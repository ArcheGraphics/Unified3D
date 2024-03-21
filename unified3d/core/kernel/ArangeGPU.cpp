//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unified3d/core/Dispatch.h"
#include "unified3d/core/Tensor.h"
#include "unified3d/core/kernel/Arange.h"
#include "unified3d/metal/Device.h"
#include <Metal/Metal.hpp>

namespace u3d::core::kernel {

void ArangeGPU(const Tensor& start,
               const Tensor& stop,
               const Tensor& step,
               Tensor& dst) {
    Dtype dtype = start.GetDtype();
    DISPATCH_DTYPE_TO_TEMPLATE(dtype, [&]() {
        auto sstart = start.Item<scalar_t>();
        auto sstep = step.Item<scalar_t>();
        auto dst_ptr = dst.GetDataView().GpuAddress();
        int64_t n = dst.GetLength();

        auto& d = metal::Device::GetInstance();
        // Encode and dispatch kernel
        auto compute_encoder = d.get_command_encoder(0);
        auto kernel = d.get_kernel("arrange");
        compute_encoder->setComputePipelineState(kernel);
        compute_encoder->setBytes(&sstart, sizeof(scalar_t), 0);
        compute_encoder->setBytes(&sstep, sizeof(scalar_t), 1);
        compute_encoder->setBytes(&dst_ptr, sizeof(uint64_t), 2);
        compute_encoder->dispatchThreads(MTL::Size(n, 1, 1), MTL::Size(128, 1, 1));
    });
}

}  // namespace u3d::core::kernel
