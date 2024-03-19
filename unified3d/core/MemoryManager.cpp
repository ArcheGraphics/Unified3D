//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <unified3d/core/MemoryManager.h>

#include <unordered_map>
#include <mach/vm_page_size.h>

#include <unified3d/metal/Metal.h>
#include <unified3d/metal/Device.h>

namespace u3d::core {

metal::Buffer MemoryManager::Malloc(size_t byte_size, const Device& device) {
    return metal::Allocator::GetInstance().Malloc(byte_size, true);
}

void MemoryManager::Free(metal::Buffer ptr, const Device& device) {
    metal::Allocator::GetInstance().Free(ptr);
}

}  // namespace u3d::core
