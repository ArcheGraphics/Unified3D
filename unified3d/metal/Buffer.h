//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <map>

namespace MTL {
class Buffer;
class Device;
}  // namespace MTL

namespace u3d::core::metal {
class Buffer {
private:
    MTL::Buffer* ptr_;
    uint64_t offset_;

public:
    explicit Buffer(MTL::Buffer* ptr = nullptr, uint64_t offset = 0)
        : ptr_(ptr), offset_{offset} {}

    // Get the raw data pointer from the buffer
    [[nodiscard]] void* CpuAddress() const;

    [[nodiscard]] uint64_t GpuAddress() const;

    // Get the buffer pointer from the buffer
    [[nodiscard]] const MTL::Buffer* Ptr() const { return ptr_; };
    MTL::Buffer* Ptr() { return ptr_; };

    [[nodiscard]] uint64_t Offset() const { return offset_; }

    [[nodiscard]] Buffer view(uint64_t offset) const;

    bool operator==(const Buffer& other) const;

    bool operator!=(const Buffer& other) const;
};

namespace {

class BufferCache {
public:
    explicit BufferCache(MTL::Device* device);
    ~BufferCache();

    MTL::Buffer* ReuseFromCache(size_t size);
    void RecycleToCache(MTL::Buffer* buf);
    void ReleaseCachedBuffers(size_t min_bytes_to_free);
    [[nodiscard]] size_t CacheSize() const { return pool_size_; }

private:
    struct BufferHolder {
    public:
        explicit BufferHolder(MTL::Buffer* buf_)
            : buf(buf_), prev(nullptr), next(nullptr) {}

        BufferHolder* prev;
        BufferHolder* next;
        MTL::Buffer* buf;
    };

    void Clear();
    void AddAtHead(BufferHolder* to_add);
    void RemoveFromList(BufferHolder* to_remove);

    MTL::Device* device_;

    std::multimap<size_t, BufferHolder*> buffer_pool_;
    BufferHolder* head_;
    BufferHolder* tail_;
    size_t pool_size_;
};

}  // namespace

class Allocator final {
public:
    static Allocator& GetInstance();

    Buffer Malloc(size_t size, bool allow_swap = false);
    void Free(Buffer& buffer);
    [[nodiscard]] size_t GetActiveMemory() const { return active_memory_; };
    [[nodiscard]] size_t GetPeakMemory() const { return peak_memory_; };
    size_t GetCacheMemory() { return buffer_cache_.CacheSize(); };
    size_t SetCacheLimit(size_t limit);
    size_t SetMemoryLimit(size_t limit, bool relaxed);

private:
    MTL::Device* device_;
    Allocator();

    // Caching allocator
    BufferCache buffer_cache_;

    // Allocation stats
    size_t block_limit_;
    size_t gc_limit_;
    size_t active_memory_{0};
    size_t peak_memory_{0};
    size_t max_pool_size_;
    bool relaxed_{true};

    std::mutex mutex_;
};
}  // namespace u3d::core::metal