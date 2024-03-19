//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unified3d/core/linalg/LapackWrapper.h"
#include "unified3d/core/linalg/LeastSquares.h"
#include "unified3d/core/linalg/LinalgUtils.h"

namespace u3d::core {

void LeastSquaresCPU(void* A_data,
                     void* B_data,
                     int64_t m,
                     int64_t n,
                     int64_t k,
                     Dtype dtype,
                     const Device& device) {
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        UNIFIED3D_LAPACK_CHECK(
                gels_cpu<scalar_t>(LAPACK_COL_MAJOR, 'N', m, n, k,
                                   static_cast<scalar_t*>(A_data), m,
                                   static_cast<scalar_t*>(B_data),
                                   std::max(m, n)),
                "gels failed in LeastSquaresCPU");
    });
}

}  // namespace u3d::core
