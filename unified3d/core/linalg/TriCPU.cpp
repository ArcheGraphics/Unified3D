//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unified3d/core/Dispatch.h"
#include "unified3d/core/Indexer.h"
#include "unified3d/core/Parallel.h"
#include "unified3d/core/Tensor.h"
#include "unified3d/core/linalg/TriImpl.h"

namespace u3d::core {

void TriuCPU(const Tensor &A, Tensor &output, const int diagonal) {
    DISPATCH_DTYPE_TO_TEMPLATE(A.GetDtype(), [&]() {
        const auto *A_ptr =
                static_cast<const scalar_t *>(A.GetDataPtr()->CpuAddress());
        auto *output_ptr =
                static_cast<scalar_t *>(output.GetDataPtr()->CpuAddress());
        int cols = A.GetShape()[1];
        int n = A.GetShape()[0] * cols;

        parallelFor(0, n, [&](int64_t workload_idx) {
            const int64_t idx = workload_idx / cols;
            const int64_t idy = workload_idx % cols;
            if (idy - idx >= diagonal) {
                output_ptr[workload_idx] = A_ptr[idx * cols + idy];
            }
        });
    });
}

void TrilCPU(const Tensor &A, Tensor &output, const int diagonal) {
    DISPATCH_DTYPE_TO_TEMPLATE(A.GetDtype(), [&]() {
        const auto *A_ptr =
                static_cast<const scalar_t *>(A.GetDataPtr()->CpuAddress());
        auto *output_ptr =
                static_cast<scalar_t *>(output.GetDataPtr()->CpuAddress());
        int cols = A.GetShape()[1];
        int n = A.GetShape()[0] * cols;

        parallelFor(0, n, [&](int64_t workload_idx) {
            const int64_t idx = workload_idx / cols;
            const int64_t idy = workload_idx % cols;
            if (idy - idx <= diagonal) {
                output_ptr[workload_idx] = A_ptr[idx * cols + idy];
            }
        });
    });
}

void TriulCPU(const Tensor &A,
              Tensor &upper,
              Tensor &lower,
              const int diagonal) {
    DISPATCH_DTYPE_TO_TEMPLATE(A.GetDtype(), [&]() {
        const auto *A_ptr =
                static_cast<const scalar_t *>(A.GetDataPtr()->CpuAddress());
        auto *upper_ptr =
                static_cast<scalar_t *>(upper.GetDataPtr()->CpuAddress());
        auto *lower_ptr =
                static_cast<scalar_t *>(lower.GetDataPtr()->CpuAddress());
        int cols = A.GetShape()[1];
        int n = A.GetShape()[0] * cols;

        parallelFor(0, n, [&](int64_t workload_idx) {
            const int64_t idx = workload_idx / cols;
            const int64_t idy = workload_idx % cols;
            if (idy - idx < diagonal) {
                lower_ptr[workload_idx] = A_ptr[idx * cols + idy];
            } else if (idy - idx > diagonal) {
                upper_ptr[workload_idx] = A_ptr[idx * cols + idy];
            } else {
                lower_ptr[workload_idx] = 1;
                upper_ptr[workload_idx] = A_ptr[idx * cols + idy];
            }
        });
    });
}

}  // namespace u3d::core
