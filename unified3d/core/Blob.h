//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cstddef>
#include <functional>
#include <iostream>
#include <string>

#include <unified3d/core/Device.h>
#include <unified3d/core/MemoryManager.h>

namespace u3d::core {

/// Blob class refers to a blob of memory in device or host.
///
/// Usually a Blob is constructed by specifying the blob size and device, memory
/// allocation happens during the Blob's construction.
///
/// A Blob's buffer can also be managed by an external memory manager. In this
/// case, a deleter function is needed to notify the external memory manager
/// that the memory is no longer needed. It does not make sense to infer the
/// total buffer size. For example, if a Tensor has a negative stride size, it
/// is necessary to access memory addresses smaller than Blob's beginning memory
/// address. The only responsibility for Blob is to hold the beginning
/// memory address and it's up to the user to access any addresses around it.
///
/// In summary:
/// - A Blob does not know about its memory size after construction.
/// - A Blob cannot be deep-copied. However, the Tensor which owns the blob can
/// be copied.
class Blob {
public:
    /// Construct Blob on a specified device.
    ///
    /// \param byte_size Size of the blob in bytes.
    /// \param device Device where the blob resides.
    Blob(int64_t byte_size, const Device& device)
        : deleter_(nullptr),
          data_holder_(MemoryManager::Malloc(byte_size, device)),
          device_(device) {}

    /// Construct Blob with externally managed memory.
    ///
    /// \param device Device where the blob resides.
    /// \param data_ptr Pointer the blob's beginning.
    /// \param deleter The deleter function is called at Blob's destruction to
    /// notify the external memory manager that the memory is no longer needed.
    /// It's up to the external manager to free the memory.
    Blob(const Device& device,
         metal::Buffer data_view,
         const std::function<void(void*)>& deleter)
        : deleter_(deleter), data_holder_(data_view), device_(device) {}

    ~Blob() {
        if (deleter_) {
            // Our custom deleter's void* argument is not used. The deleter
            // function itself shall handle destruction without the argument.
            // The void(void*) signature is kept to be consistent with DLPack's
            // deleter.
            deleter_(nullptr);
        } else {
            MemoryManager::Free(data_holder_, device_);
        }
    };

    [[nodiscard]] Device GetDevice() const { return device_; }

    [[nodiscard]] metal::Buffer& GetDataView() { return data_holder_; }

    [[nodiscard]] const metal::Buffer& GetDataView() const {
        return data_holder_;
    }

protected:
    /// For externally managed memory, deleter != nullptr.
    std::function<void(void*)> deleter_ = nullptr;

    /// Device data pointer.
    metal::Buffer data_holder_;

    /// Device context for the blob.
    Device device_;
};

}  // namespace u3d::core
