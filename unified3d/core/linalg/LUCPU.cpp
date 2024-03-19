//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unified3d/core/linalg/LUImpl.h"
#include "unified3d/core/linalg/LapackWrapper.h"
#include "unified3d/core/linalg/LinalgUtils.h"

namespace u3d::core {

void LUCPU(void* A_data,
           void* ipiv_data,
           int64_t rows,
           int64_t cols,
           Dtype dtype,
           const Device& device) {
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        UNIFIED3D_LAPACK_CHECK(
                getrf_cpu<scalar_t>(
                        LAPACK_COL_MAJOR, rows, cols,
                        static_cast<scalar_t*>(A_data), rows,
                        static_cast<UNIFIED3D_CPU_LINALG_INT*>(ipiv_data)),
                "getrf failed in LUCPU");
    });
}

}  // namespace u3d::core
