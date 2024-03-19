//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "unified3d/core/Tensor.h"

namespace u3d {
namespace core {
namespace kernel {

Tensor NonZero(const Tensor& src);

Tensor NonZeroCPU(const Tensor& src);

#ifdef BUILD_CUDA_MODULE
Tensor NonZeroCUDA(const Tensor& src);
#endif

}  // namespace kernel
}  // namespace core
}  // namespace u3d
