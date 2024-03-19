//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unified3d/core/linalg/Det.h"

#include "unified3d/core/linalg/LU.h"
#include "unified3d/core/linalg/kernel/Matrix.h"

namespace u3d::core {

double Det(const Tensor& A) {
    AssertTensorDtypes(A, {Float32, Float64});
    const Dtype dtype = A.GetDtype();

    double det = 1.0;

    if (A.GetShape() == u3d::core::SizeVector({3, 3})) {
        DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
            core::Tensor A_3x3 =
                    A.To(core::Device("CPU:0"), false).Contiguous();
            const scalar_t* A_3x3_ptr =
                    (scalar_t*)A_3x3.GetDataPtr()->CpuAddress();
            det = static_cast<double>(linalg::kernel::det3x3(A_3x3_ptr));
        });
    } else if (A.GetShape() == u3d::core::SizeVector({2, 2})) {
        DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
            core::Tensor A_2x2 =
                    A.To(core::Device("CPU:0"), false).Contiguous();
            const scalar_t* A_2x2_ptr =
                    (scalar_t*)A_2x2.GetDataPtr()->CpuAddress();
            det = static_cast<double>(linalg::kernel::det2x2(A_2x2_ptr));
        });
    } else {
        Tensor ipiv, output;
        LUIpiv(A, ipiv, output);
        // Sequential loop to compute determinant from LU output, is more
        // efficient on CPU.
        Tensor output_cpu = output.To(core::Device("CPU:0"));
        Tensor ipiv_cpu = ipiv.To(core::Device("CPU:0"));
        int n = A.GetShape()[0];

        DISPATCH_FLOAT_DTYPE_TO_TEMPLATE(dtype, [&]() {
            auto* output_ptr = (scalar_t*)output_cpu.GetDataPtr()->CpuAddress();
            int* ipiv_ptr =
                    static_cast<int*>(ipiv_cpu.GetDataPtr()->CpuAddress());

            for (int i = 0; i < n; i++) {
                det *= output_ptr[i * n + i];
                if (ipiv_ptr[i] != i) {
                    det *= -1;
                }
            }
        });
    }
    return det;
}

}  // namespace u3d::core
