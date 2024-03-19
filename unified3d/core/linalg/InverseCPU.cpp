//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unified3d/core/linalg/Inverse.h"
#include "unified3d/core/linalg/LapackWrapper.h"
#include "unified3d/core/linalg/LinalgUtils.h"

namespace u3d::core {

void InverseCPU(void* A_data,
                void* ipiv_data,
                void* output_data,
                int64_t n,
                Dtype dtype,
                const Device& device) {
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        UNIFIED3D_LAPACK_CHECK(
                getrf_cpu<scalar_t>(
                        LAPACK_COL_MAJOR, n, n, static_cast<scalar_t*>(A_data),
                        n, static_cast<UNIFIED3D_CPU_LINALG_INT*>(ipiv_data)),
                "getrf failed in InverseCPU");
        UNIFIED3D_LAPACK_CHECK(
                getri_cpu<scalar_t>(
                        LAPACK_COL_MAJOR, n, static_cast<scalar_t*>(A_data), n,
                        static_cast<UNIFIED3D_CPU_LINALG_INT*>(ipiv_data)),
                "getri failed in InverseCPU");
    });
}

}  // namespace u3d::core
