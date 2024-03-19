//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

// Private header. Do not include in Open3d.h.

#pragma once

#include "unified3d/core/linalg/LU.h"

namespace u3d::core {

void LUCPU(void* A_data,
           void* ipiv_data,
           int64_t rows,
           int64_t cols,
           Dtype dtype,
           const Device& device);

}  // namespace u3d::core
