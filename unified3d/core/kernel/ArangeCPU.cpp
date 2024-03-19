//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unified3d/core/Dispatch.h"
#include "unified3d/core/Parallel.h"
#include "unified3d/core/Tensor.h"
#include "unified3d/core/kernel/Arange.h"

namespace u3d::core::kernel {

void ArangeCPU(const Tensor& start,
               const Tensor& stop,
               const Tensor& step,
               Tensor& dst) {
    Dtype dtype = start.GetDtype();
    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        auto sstart = start.Item<scalar_t>();
        auto sstep = step.Item<scalar_t>();
        scalar_t* dst_ptr = (scalar_t*)dst.GetDataPtr()->CpuAddress();
        int64_t n = dst.GetLength();
        parallelFor(int64_t(0), n, [&](int64_t workload_idx) {
            dst_ptr[workload_idx] =
                    sstart + static_cast<scalar_t>(sstep * workload_idx);
        });
    });
}

}  // namespace u3d::core::kernel
