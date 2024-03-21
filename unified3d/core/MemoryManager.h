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
class MemoryManager {
public:
    /// Allocates memory of \p byte_size bytes on device \p device and returns a
    /// pointer to the beginning of the allocated memory block.
    static metal::Buffer Malloc(size_t byte_size, const Device& device);

    /// Frees previously allocated memory at address \p ptr on device \p device.
    static void Free(metal::Buffer& ptr, const Device& device);

    /// Copies \p num_bytes bytes of memory at address \p src_ptr on device
    /// \p src_device to address \p dst_ptr on device \p dst_device.
    static void MemcpyOnCpu(metal::Buffer& dst_ptr,
                            const metal::Buffer& src_ptr,
                            size_t num_bytes);

    template <typename T>
    static void MemcpyToHost(T* dst_ptr,
                             const metal::Buffer& src_ptr,
                             size_t num_bytes) {
        std::memcpy(dst_ptr, src_ptr.CpuAddress(), num_bytes);
    }

    template <typename T>
    static void MemcpyFromHost(metal::Buffer& dst_ptr,
                               T* src_ptr,
                               size_t num_bytes) {
        std::memcpy(dst_ptr.CpuAddress(), src_ptr, num_bytes);
    }
};

}  // namespace u3d::core
