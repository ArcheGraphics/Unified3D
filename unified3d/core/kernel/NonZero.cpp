//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unified3d/core/kernel/NonZero.h"

#include "unified3d/core/Device.h"
#include "unified3d/core/Tensor.h"
#include "unified3d/utility/Logging.h"

namespace u3d::core::kernel {

Tensor NonZero(const Tensor& src) {
    if (src.IsCPU()) {
        return NonZeroCPU(src);
    } else if (src.IsGPU()) {
#ifdef BUILD_CUDA_MODULE
        return NonZeroCUDA(src);
#else
        utility::LogError("Not compiled with CUDA, but CUDA device is used.");
#endif
    } else {
        utility::LogError("NonZero: Unimplemented device");
    }
}

}  // namespace u3d::core::kernel
