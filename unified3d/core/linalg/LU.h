//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "unified3d/core/Tensor.h"

namespace u3d::core {

// See documentation for `core::Tensor::LUIpiv`.
void LUIpiv(const Tensor& A, Tensor& ipiv, Tensor& output);

// See documentation for `core::Tensor::LU`.
void LU(const Tensor& A,
        Tensor& permutation,
        Tensor& lower,
        Tensor& upper,
        bool permute_l = false);

}  // namespace u3d::core
