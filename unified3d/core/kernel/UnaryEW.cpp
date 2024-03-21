//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unified3d/core/kernel/UnaryEW.h"

#include "unified3d/core/ShapeUtil.h"
#include "unified3d/core/Tensor.h"
#include "unified3d/utility/Logging.h"

namespace u3d::core::kernel {

void UnaryEW(const Tensor& src, Tensor& dst, UnaryEWOpCode op_code) {
    // Check shape
    if (!shape_util::CanBeBrocastedToShape(src.GetShape(), dst.GetShape())) {
        utility::LogError("Shape {} can not be broadcasted to {}.",
                          src.GetShape(), dst.GetShape());
    }

    // Dispatch to device
    Device src_device = src.GetDevice();
    Device dst_device = dst.GetDevice();
    if (src_device != dst_device) {
        utility::LogError("Source device {} != destination device {}.",
                          src_device.ToString(), dst_device.ToString());
    }

    if (src_device.IsCPU()) {
        UnaryEWCPU(src, dst, op_code);
    } else if (src_device.IsGPU()) {
        UnaryEWGPU(src, dst, op_code);
    } else {
        utility::LogError("UnaryEW Unimplemented device");
    }
}

void Copy(const Tensor& src, Tensor& dst) {
    // Check shape
    if (!shape_util::CanBeBrocastedToShape(src.GetShape(), dst.GetShape())) {
        utility::LogError("Shape {} can not be broadcasted to {}.",
                          src.GetShape(), dst.GetShape());
    }

    // Disbatch to device
    Device src_device = src.GetDevice();
    Device dst_device = dst.GetDevice();
    if ((!src_device.IsCPU() && !src_device.IsGPU()) ||
        (!dst_device.IsCPU() && !dst_device.IsGPU())) {
        utility::LogError("Copy: Unimplemented device");
    }
    if (src_device.IsCPU() && dst_device.IsCPU()) {
        CopyCPU(src, dst);
    } else {
        CopyGPU(src, dst);
    }
}

}  // namespace u3d::core::kernel
