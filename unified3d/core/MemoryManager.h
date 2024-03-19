//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <unified3d/core/Device.h>
#include <unified3d/metal/Buffer.h>

namespace u3d::core {
/// Top-level memory interface. Calls to any of the member functions will
/// automatically dispatch the appropriate MemoryManagerDevice instance based on
/// the provided device which is used to execute the requested functionality.
///
/// The memory managers are dispatched as follows:
///
/// DeviceType = CPU : MemoryManagerCPU
/// DeviceType = CUDA :
///   ENABLE_CACHED_CUDA_MANAGER = ON : MemoryManagerCached w/ MemoryManagerCUDA
///   Otherwise :                      MemoryManagerCUDA
///
class MemoryManager {
public:
    /// Allocates memory of \p byte_size bytes on device \p device and returns a
    /// pointer to the beginning of the allocated memory block.
    static metal::Buffer Malloc(size_t byte_size, const Device& device);

    /// Frees previously allocated memory at address \p ptr on device \p device.
    static void Free(metal::Buffer ptr, const Device& device);
};

}  // namespace u3d::core
