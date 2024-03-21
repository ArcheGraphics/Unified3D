//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <numeric>

#include "unified3d/core/Indexer.h"
#include "unified3d/core/kernel/NonZero.h"
#include "unified3d/utility/Logging.h"

namespace u3d::core::kernel {

Tensor NonZeroCPU(const Tensor& src) {
    // Get flattened non-zero indices.
    TensorIterator src_iter(src);
    const int64_t num_elements = src.NumElements();
    std::vector<int64_t> indices(static_cast<size_t>(num_elements));
    std::iota(std::begin(indices), std::end(indices), 0);
    std::vector<int64_t> non_zero_indices(num_elements);
    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src.GetDtype(), [&]() {
        auto it = std::copy_if(
                indices.begin(), indices.end(), non_zero_indices.begin(),
                [&src_iter](int64_t index) {
                    const void* src_ptr = src_iter.GetView(index).CpuAddress();
                    UNIFIED3D_ASSERT(src_ptr != nullptr && "Internal error.");
                    return static_cast<float>(
                                   *static_cast<const scalar_t*>(src_ptr)) != 0;
                });
        non_zero_indices.resize(std::distance(non_zero_indices.begin(), it));
    });

    // Transform flattened indices to indices in each dimension.
    SizeVector shape = src.GetShape();
    const int64_t num_dims = src.NumDims();
    const size_t num_non_zeros = non_zero_indices.size();

    SizeVector result_shape{num_dims, static_cast<int64_t>(num_non_zeros)};
    Tensor result(result_shape, core::Int64, src.GetDevice());
    TensorIterator result_iter(result);

    std::vector<std::vector<int64_t>> non_zero_indices_by_dimensions(
            num_dims, std::vector<int64_t>(num_non_zeros, 0));

    for (int64_t i = 0; i < static_cast<int64_t>(num_non_zeros); i++) {
        int64_t non_zero_index = non_zero_indices[i];
        for (int64_t dim = num_dims - 1; dim >= 0; dim--) {
            void* result_ptr =
                    result_iter.GetView(dim * num_non_zeros + i).CpuAddress();
            UNIFIED3D_ASSERT(result_ptr != nullptr && "Internal error.");
            *static_cast<int64_t*>(result_ptr) = non_zero_index % shape[dim];
            non_zero_index = non_zero_index / shape[dim];
        }
    }

    return result;
}

}  // namespace u3d::core::kernel
