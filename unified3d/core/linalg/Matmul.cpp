//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unified3d/core/linalg/Matmul.h"

#include <unordered_map>

namespace u3d::core {

void Matmul(const Tensor& A, const Tensor& B, Tensor& output) {
    AssertTensorDevice(B, A.GetDevice());
    AssertTensorDtype(B, A.GetDtype());

    const Device device = A.GetDevice();
    const Dtype dtype_original = A.GetDtype();
    Dtype dtype;

    if (dtype_original != core::Float32 && dtype_original != core::Float64) {
        utility::LogDebug("Converting to Float32 dtype to from {}.",
                          dtype_original.ToString());
        dtype = core::Float32;
    } else {
        dtype = dtype_original;
    }

    // Check shapes
    SizeVector A_shape = A.GetShape();
    SizeVector B_shape = B.GetShape();

    if (A_shape.size() != 2) {
        utility::LogError("Tensor A must be 2D, but got {}D.", A_shape.size());
    }
    if (B_shape.size() != 1 && B_shape.size() != 2) {
        utility::LogError(
                "Tensor B must be 1D (vector) or 2D (matrix), but got {}D.",
                B_shape.size());
    }
    if (A_shape[1] != B_shape[0]) {
        utility::LogError("Tensor A columns {} mismatch with Tensor B rows {}.",
                          A_shape[1], B_shape[0]);
    }

    // Dispatch to backends
    int64_t m = A_shape[0];
    int64_t k = A_shape[1];
    int64_t n = B_shape.size() == 2 ? B_shape[1] : 1;

    if (m == 0 || k == 0 || n == 0) {
        utility::LogError(
                "Tensor shapes should not contain dimensions with zero.");
    }

    Tensor A_contiguous = A.Contiguous().To(dtype);
    Tensor B_contiguous = B.Contiguous().To(dtype);
    void* A_data = A_contiguous.GetDataPtr()->CpuAddress();
    void* B_data = B_contiguous.GetDataPtr()->CpuAddress();

    output = Tensor::Empty({m, n}, dtype, device);
    void* C_data = output.GetDataPtr()->CpuAddress();

    if (device.IsGPU()) {
#ifdef BUILD_CUDA_MODULE
        CUDAScopedDevice scoped_device(device);
        MatmulCUDA(B_data, A_data, C_data, n, k, m, dtype, device);
#else
        utility::LogError("Unimplemented device.");
#endif
    } else {
        MatmulCPU(B_data, A_data, C_data, n, k, m, dtype);
    }

    output = output.To(dtype_original);
}

}  // namespace u3d::core
