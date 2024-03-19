//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unified3d/core/kernel/IndexReduction.h"

#include "unified3d/utility/Logging.h"

namespace u3d::core::kernel {

void IndexAdd_(int64_t dim,
               const Tensor& index,
               const Tensor& src,
               Tensor& dst) {
    // Permute the reduction dimension to the first.
    SizeVector permute = {};
    for (int64_t d = 0; d <= dim; ++d) {
        if (d == 0) {
            permute.push_back(dim);
        } else {
            permute.push_back(d - 1);
        }
    }
    for (int64_t d = dim + 1; d < src.NumDims(); ++d) {
        permute.push_back(d);
    }

    auto src_permute = src.Permute(permute);
    auto dst_permute = dst.Permute(permute);

    if (dst.IsCPU()) {
        IndexAddCPU_(dim, index, src_permute, dst_permute);
    } else if (dst.IsGPU()) {
#ifdef BUILD_CUDA_MODULE
        IndexAddCUDA_(dim, index, src_permute, dst_permute);
#endif
    } else {
        utility::LogError("IndexAdd_: Unimplemented device");
    }
}

}  // namespace u3d::core::kernel
