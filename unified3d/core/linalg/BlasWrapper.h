//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "unified3d/core/linalg/LinalgHeadersCPU.h"
#include "unified3d/utility/Logging.h"

namespace u3d::core {

template <typename scalar_t>
inline void gemm_cpu(CBLAS_ORDER layout,
                     CBLAS_TRANSPOSE trans_A,
                     CBLAS_TRANSPOSE trans_B,
                     UNIFIED3D_CPU_LINALG_INT m,
                     UNIFIED3D_CPU_LINALG_INT n,
                     UNIFIED3D_CPU_LINALG_INT k,
                     scalar_t alpha,
                     const scalar_t *A_data,
                     UNIFIED3D_CPU_LINALG_INT lda,
                     const scalar_t *B_data,
                     UNIFIED3D_CPU_LINALG_INT ldb,
                     scalar_t beta,
                     scalar_t *C_data,
                     UNIFIED3D_CPU_LINALG_INT ldc) {
    utility::LogError("Unsupported data type.");
}

template <>
inline void gemm_cpu<float>(CBLAS_ORDER layout,
                            CBLAS_TRANSPOSE trans_A,
                            CBLAS_TRANSPOSE trans_B,
                            UNIFIED3D_CPU_LINALG_INT m,
                            UNIFIED3D_CPU_LINALG_INT n,
                            UNIFIED3D_CPU_LINALG_INT k,
                            float alpha,
                            const float *A_data,
                            UNIFIED3D_CPU_LINALG_INT lda,
                            const float *B_data,
                            UNIFIED3D_CPU_LINALG_INT ldb,
                            float beta,
                            float *C_data,
                            UNIFIED3D_CPU_LINALG_INT ldc) {
    cblas_sgemm(layout, trans_A, trans_B, m, n, k, alpha, A_data, lda, B_data,
                ldb, beta, C_data, ldc);
}

template <>
inline void gemm_cpu<double>(CBLAS_ORDER layout,
                             CBLAS_TRANSPOSE trans_A,
                             CBLAS_TRANSPOSE trans_B,
                             UNIFIED3D_CPU_LINALG_INT m,
                             UNIFIED3D_CPU_LINALG_INT n,
                             UNIFIED3D_CPU_LINALG_INT k,
                             double alpha,
                             const double *A_data,
                             UNIFIED3D_CPU_LINALG_INT lda,
                             const double *B_data,
                             UNIFIED3D_CPU_LINALG_INT ldb,
                             double beta,
                             double *C_data,
                             UNIFIED3D_CPU_LINALG_INT ldc) {
    cblas_dgemm(layout, trans_A, trans_B, m, n, k, alpha, A_data, lda, B_data,
                ldb, beta, C_data, ldc);
}

}  // namespace u3d::core
