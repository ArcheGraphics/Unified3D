//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unified3d/core/linalg/LapackWrapper.h"
#include "unified3d/core/linalg/LinalgUtils.h"
#include "unified3d/core/linalg/Solve.h"

namespace u3d::core {

void SolveCPU(void* A_data,
              void* B_data,
              void* ipiv_data,
              int64_t n,
              int64_t k,
              Dtype dtype,
              const Device& device) {
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        UNIFIED3D_LAPACK_CHECK(
                gesv_cpu<scalar_t>(
                        LAPACK_COL_MAJOR, n, k, static_cast<scalar_t*>(A_data),
                        n, static_cast<UNIFIED3D_CPU_LINALG_INT*>(ipiv_data),
                        static_cast<scalar_t*>(B_data), n),
                "gels failed in SolveCPU");
    });
}

}  // namespace u3d::core
