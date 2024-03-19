//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <unified3d/core/Tensor.h>

namespace u3d::core::kernel {

Tensor Arange(const Tensor& start, const Tensor& stop, const Tensor& step);

void ArangeCPU(const Tensor& start,
               const Tensor& stop,
               const Tensor& step,
               Tensor& dst);

#ifdef BUILD_CUDA_MODULE
void ArangeCUDA(const Tensor& start,
                const Tensor& stop,
                const Tensor& step,
                Tensor& dst);
#endif

}  // namespace u3d::core::kernel
