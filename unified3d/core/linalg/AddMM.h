//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <unified3d/core/Tensor.h>

namespace u3d::core {

/// Computes matrix multiplication C = alpha * A @ B + beta * C.
/// If matrix A is a (n x m) tensor, and B is a (m x p) tensor, C should have a
/// shape (n x p).
/// alpha and beta are scaling factors on matrix-matrix multiplication and the
/// added matrix input respectively.
void AddMM(
        const Tensor& A, const Tensor& B, Tensor& C, double alpha, double beta);

void AddMMCPU(void* A_data,
              void* B_data,
              void* C_data,
              int64_t m,
              int64_t k,
              int64_t n,
              double alpha,
              double beta,
              bool gemmTrA,
              bool gemmTrB,
              int lda,
              int ldb,
              int ldc,
              Dtype dtype);

}  // namespace u3d::core
