//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <unordered_set>

#include "unified3d/core/Tensor.h"
#include "unified3d/utility/Helper.h"
#include "unified3d/utility/Logging.h"

namespace u3d::core::kernel {

enum class BinaryEWOpCode {
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Minimum,
    LogicalAnd,
    LogicalOr,
    LogicalXor,
    Gt,
    Lt,
    Ge,
    Le,
    Eq,
    Ne,
};

extern const std::unordered_set<BinaryEWOpCode, utility::hash_enum_class>
        s_boolean_binary_ew_op_codes;

void BinaryEW(const Tensor& lhs,
              const Tensor& rhs,
              Tensor& dst,
              BinaryEWOpCode op_code);

void BinaryEWCPU(const Tensor& lhs,
                 const Tensor& rhs,
                 Tensor& dst,
                 BinaryEWOpCode op_code);

#ifdef BUILD_CUDA_MODULE
void BinaryEWCUDA(const Tensor& lhs,
                  const Tensor& rhs,
                  Tensor& dst,
                  BinaryEWOpCode op_code);
#endif

}  // namespace u3d::core::kernel
