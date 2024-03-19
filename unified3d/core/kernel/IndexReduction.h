//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "unified3d/core/Tensor.h"
#include "unified3d/utility/Logging.h"

namespace u3d {
namespace core {
namespace kernel {

void IndexAdd_(int64_t dim,
               const Tensor& index,
               const Tensor& src,
               Tensor& dst);

void IndexAddCPU_(int64_t dim,
                  const Tensor& index,
                  const Tensor& src,
                  Tensor& dst);

#ifdef BUILD_CUDA_MODULE
void IndexAddCUDA_(int64_t dim,
                   const Tensor& index,
                   const Tensor& src,
                   Tensor& dst);
#endif

}  // namespace kernel
}  // namespace core
}  // namespace u3d
