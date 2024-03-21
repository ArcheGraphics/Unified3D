//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "Buffer.h"
#include <Metal/Metal.hpp>
#include <unified3d/metal/Metal.h>
#include <unified3d/metal/Device.h>

namespace u3d::core::metal {
void* Buffer::CpuAddress() const {
    return (uint8_t*)ptr_->contents() + offset_;
}

uint64_t Buffer::GpuAddress() const { return ptr_->gpuAddress() + offset_; }

Buffer Buffer::view(uint64_t offset) const {
    return Buffer{ptr_, offset + offset_};
}

bool Buffer::operator==(const Buffer &other) const {
    return this->ptr_ == other.ptr_ &&
           this->offset_ == other.offset_;
}

bool Buffer::operator!=(const Buffer &other) const {
    return !operator==(other);
}

namespace {

BufferCache::BufferCache(MTL::Device* device)
    : device_(device), head_(nullptr), tail_(nullptr), pool_size_(0) {}

BufferCache::~BufferCache() {
    auto thread_pool = NewScopedMemoryPool();
    Clear();
}

void BufferCache::Clear() {
    for (auto& [size, holder] : buffer_pool_) {
        if (holder->buf) holder->buf->release();
        delete holder;
    }
    buffer_pool_.clear();
    pool_size_ = 0;
    head_ = nullptr;
    tail_ = nullptr;
}

MTL::Buffer* BufferCache::ReuseFromCache(size_t size) {
    // Find the closest buffer in pool
    MTL::Buffer* pbuf = nullptr;

    auto it = buffer_pool_.lower_bound(size);

    // Make sure we use most of the available memory
    while (!pbuf && it != buffer_pool_.end() &&
           it->first < std::min(2 * size, size + 2 * vm_page_size)) {
        // Collect from the cache
        pbuf = it->second->buf;

        // Remove from cache
        RemoveFromList(it->second);
        delete it->second;
        it = buffer_pool_.erase(it);
    }

    if (pbuf) {
        pool_size_ -= pbuf->length();
    }

    return pbuf;
}

void BufferCache::RecycleToCache(MTL::Buffer* buf) {
    // Add to cache
    if (buf) {
        auto* bh = new BufferHolder(buf);
        AddAtHead(bh);
        pool_size_ += buf->length();
        buffer_pool_.insert({buf->length(), bh});
    }
}

void BufferCache::ReleaseCachedBuffers(size_t min_bytes_to_free) {
    if (min_bytes_to_free >= 0.9 * pool_size_) {
        Clear();
    } else {
        size_t total_bytes_freed = 0;

        while (tail_ && (total_bytes_freed < min_bytes_to_free)) {
            if (tail_->buf) {
                total_bytes_freed += tail_->buf->length();
                tail_->buf->release();
                tail_->buf = nullptr;
            }
            RemoveFromList(tail_);
        }
        pool_size_ -= total_bytes_freed;
    }
}

void BufferCache::AddAtHead(BufferCache::BufferHolder* to_add) {
    if (!to_add) return;

    if (!head_) {
        head_ = to_add;
        tail_ = to_add;
    } else {
        head_->prev = to_add;
        to_add->next = head_;
        head_ = to_add;
    }
}

void BufferCache::RemoveFromList(BufferCache::BufferHolder* to_remove) {
    if (!to_remove) {
        return;
    }

    // If in the middle
    if (to_remove->prev && to_remove->next) {
        to_remove->prev->next = to_remove->next;
        to_remove->next->prev = to_remove->prev;
    } else if (to_remove->prev && to_remove == tail_) {  // If tail
        tail_ = to_remove->prev;
        tail_->next = nullptr;
    } else if (to_remove == head_ && to_remove->next) {  // If head
        head_ = to_remove->next;
        head_->prev = nullptr;
    } else if (to_remove == head_ && to_remove == tail_) {  // If only element
        head_ = nullptr;
        tail_ = nullptr;
    }

    to_remove->prev = nullptr;
    to_remove->next = nullptr;
}

}  // namespace

Allocator::Allocator()
    : device_(metal::Device::GetInstance().mtl_device()),
      buffer_cache_(device_),
      block_limit_(1.5 * device_->recommendedMaxWorkingSetSize()),
      gc_limit_(0.95 * device_->recommendedMaxWorkingSetSize()),
      max_pool_size_(block_limit_) {}

Allocator& Allocator::GetInstance() {
    static Allocator allocator_;
    return allocator_;
}

size_t Allocator::SetCacheLimit(size_t limit) {
    std::swap(limit, max_pool_size_);
    return limit;
}

size_t Allocator::SetMemoryLimit(size_t limit, bool relaxed) {
    std::swap(limit, block_limit_);
    relaxed_ = relaxed;
    gc_limit_ =
            std::min(block_limit_,
                     static_cast<size_t>(
                             0.95 * device_->recommendedMaxWorkingSetSize()));
    return limit;
}

Buffer Allocator::Malloc(size_t size, bool allow_swap /* = false */) {
    // Metal doesn't like empty buffers
    size = std::max<size_t>(size, 4);

    // Align up memory
    if (size > vm_page_size) {
        size = vm_page_size * ((size + vm_page_size - 1) / vm_page_size);
    }

    // Try the cache
    std::unique_lock lk(mutex_);
    MTL::Buffer* buf = buffer_cache_.ReuseFromCache(size);
    if (!buf) {
        size_t mem_required = GetActiveMemory() + GetCacheMemory() + size;

        // If there is too much memory pressure, fail (likely causes a wait).
        if (!(allow_swap && relaxed_) && mem_required >= block_limit_) {
            return Buffer{nullptr};
        }

        auto thread_pool = NewScopedMemoryPool();

        // If we have a lot of memory pressure or are over the maximum cache
        // size, try to reclaim memory from the cache
        if (mem_required >= gc_limit_) {
            buffer_cache_.ReleaseCachedBuffers(mem_required - gc_limit_);
        }

        // Allocate new buffer if needed
        size_t res_opt = MTL::ResourceStorageModeShared;
        res_opt |= MTL::ResourceHazardTrackingModeTracked;
        lk.unlock();
        buf = device_->newBuffer(size, res_opt);
        lk.lock();
    }

    active_memory_ += buf->length();
    peak_memory_ = std::max(peak_memory_, active_memory_);

    // Maintain the cache below the requested limit
    if (GetCacheMemory() >= max_pool_size_) {
        auto thread_pool = NewScopedMemoryPool();
        buffer_cache_.ReleaseCachedBuffers(GetCacheMemory() - max_pool_size_);
    }

    return Buffer(buf);
}

void Allocator::Free(Buffer& buffer) {
    auto buf = buffer.Ptr();
    std::unique_lock lk(mutex_);
    active_memory_ -= buf->length();
    if (GetCacheMemory() < max_pool_size_) {
        buffer_cache_.RecycleToCache(buf);
    } else {
        lk.unlock();
        auto thread_pool = NewScopedMemoryPool();
        buf->release();
    }
}

}  // namespace u3d::core::metal