//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <unordered_set>

#include "unified3d/core/SizeVector.h"
#include "unified3d/core/Tensor.h"
#include "unified3d/utility/Helper.h"
#include "unified3d/utility/Logging.h"

namespace u3d {
namespace core {
namespace kernel {

enum class ReductionOpCode {
    Sum,
    Prod,
    Min,
    Max,
    ArgMin,
    ArgMax,
    All,
    Any,
};

static const std::unordered_set<ReductionOpCode, utility::hash_enum_class>
        s_regular_reduce_ops = {
                ReductionOpCode::Sum,
                ReductionOpCode::Prod,
                ReductionOpCode::Min,
                ReductionOpCode::Max,
};
static const std::unordered_set<ReductionOpCode, utility::hash_enum_class>
        s_arg_reduce_ops = {
                ReductionOpCode::ArgMin,
                ReductionOpCode::ArgMax,
};
static const std::unordered_set<ReductionOpCode, utility::hash_enum_class>
        s_boolean_reduce_ops = {
                ReductionOpCode::All,
                ReductionOpCode::Any,
};

void Reduction(const Tensor& src,
               Tensor& dst,
               const SizeVector& dims,
               bool keepdim,
               ReductionOpCode op_code);

void ReductionCPU(const Tensor& src,
                  Tensor& dst,
                  const SizeVector& dims,
                  bool keepdim,
                  ReductionOpCode op_code);

#ifdef BUILD_CUDA_MODULE
void ReductionCUDA(const Tensor& src,
                   Tensor& dst,
                   const SizeVector& dims,
                   bool keepdim,
                   ReductionOpCode op_code);
#endif

}  // namespace kernel
}  // namespace core
}  // namespace u3d
