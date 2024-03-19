//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unified3d/core/Dispatch.h"
#include "unified3d/core/Dtype.h"
#include "unified3d/core/Indexer.h"
#include "unified3d/core/MemoryManager.h"
#include "unified3d/core/Parallel.h"
#include "unified3d/core/SizeVector.h"
#include "unified3d/core/Tensor.h"
#include "unified3d/core/kernel/BinaryEW.h"
#include "unified3d/utility/Logging.h"

namespace u3d::core::kernel {

template <typename src_t, typename dst_t, typename element_func_t>
static void LaunchBinaryEWKernel(const Indexer& indexer,
                                 const element_func_t& element_func) {
    parallelFor(int64_t(0), indexer.NumWorkloads(),
                [&indexer, &element_func](int64_t i) {
                    element_func(indexer.GetInputPtr(0, i)->CpuAddress(),
                                 indexer.GetInputPtr(1, i)->CpuAddress(),
                                 indexer.GetOutputPtr(i)->CpuAddress());
                });
}

template <typename scalar_t>
static void CPUMaxElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<scalar_t*>(dst) = std::max(*static_cast<const scalar_t*>(lhs),
                                            *static_cast<const scalar_t*>(rhs));
}

template <typename scalar_t>
static void CPUMinElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<scalar_t*>(dst) = std::min(*static_cast<const scalar_t*>(lhs),
                                            *static_cast<const scalar_t*>(rhs));
}

template <typename scalar_t>
static void CPUAddElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(lhs) +
                                   *static_cast<const scalar_t*>(rhs);
}

template <typename scalar_t>
static void CPUSubElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(lhs) -
                                   *static_cast<const scalar_t*>(rhs);
}

template <typename scalar_t>
static void CPUMulElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(lhs) *
                                   *static_cast<const scalar_t*>(rhs);
}

template <typename scalar_t>
static void CPUDivElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<scalar_t*>(dst) = *static_cast<const scalar_t*>(lhs) /
                                   *static_cast<const scalar_t*>(rhs);
}

template <typename src_t, typename dst_t>
static void CPULogicalAndElementKernel(const void* lhs,
                                       const void* rhs,
                                       void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            static_cast<bool>(*static_cast<const src_t*>(lhs)) &&
            static_cast<bool>(*static_cast<const src_t*>(rhs)));
}

template <typename src_t, typename dst_t>
static void CPULogicalOrElementKernel(const void* lhs,
                                      const void* rhs,
                                      void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            static_cast<bool>(*static_cast<const src_t*>(lhs)) ||
            static_cast<bool>(*static_cast<const src_t*>(rhs)));
}

template <typename src_t, typename dst_t>
static void CPULogicalXorElementKernel(const void* lhs,
                                       const void* rhs,
                                       void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            static_cast<bool>(*static_cast<const src_t*>(lhs)) !=
            static_cast<bool>(*static_cast<const src_t*>(rhs)));
}

template <typename src_t, typename dst_t>
static void CPUGtElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            *static_cast<const src_t*>(lhs) > *static_cast<const src_t*>(rhs));
}

template <typename src_t, typename dst_t>
static void CPULtElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            *static_cast<const src_t*>(lhs) < *static_cast<const src_t*>(rhs));
}

template <typename src_t, typename dst_t>
static void CPUGeqElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            *static_cast<const src_t*>(lhs) >= *static_cast<const src_t*>(rhs));
}

template <typename src_t, typename dst_t>
static void CPULeqElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            *static_cast<const src_t*>(lhs) <= *static_cast<const src_t*>(rhs));
}

template <typename src_t, typename dst_t>
static void CPUEqElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            *static_cast<const src_t*>(lhs) == *static_cast<const src_t*>(rhs));
}

template <typename src_t, typename dst_t>
static void CPUNeqElementKernel(const void* lhs, const void* rhs, void* dst) {
    *static_cast<dst_t*>(dst) = static_cast<dst_t>(
            *static_cast<const src_t*>(lhs) != *static_cast<const src_t*>(rhs));
}

void BinaryEWCPU(const Tensor& lhs,
                 const Tensor& rhs,
                 Tensor& dst,
                 BinaryEWOpCode op_code) {
    Dtype src_dtype = lhs.GetDtype();
    Dtype dst_dtype = dst.GetDtype();

    if (s_boolean_binary_ew_op_codes.find(op_code) !=
        s_boolean_binary_ew_op_codes.end()) {
        if (dst_dtype == src_dtype) {
            // Inplace boolean op's output type is the same as the
            // input. e.g. np.logical_and(a, b, out=a), where a, b are
            // floats.
            Indexer indexer({lhs, rhs}, dst, DtypePolicy::ALL_SAME);

            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                switch (op_code) {
                    case BinaryEWOpCode::LogicalAnd:
                        LaunchBinaryEWKernel<scalar_t, scalar_t>(
                                indexer,
                                CPULogicalAndElementKernel<scalar_t, scalar_t>);
                        break;
                    case BinaryEWOpCode::LogicalOr:
                        LaunchBinaryEWKernel<scalar_t, scalar_t>(
                                indexer,
                                CPULogicalOrElementKernel<scalar_t, scalar_t>);
                        break;
                    case BinaryEWOpCode::LogicalXor:
                        LaunchBinaryEWKernel<scalar_t, scalar_t>(
                                indexer,
                                CPULogicalXorElementKernel<scalar_t, scalar_t>);
                        break;
                    case BinaryEWOpCode::Gt:
                        LaunchBinaryEWKernel<scalar_t, scalar_t>(
                                indexer,
                                CPUGtElementKernel<scalar_t, scalar_t>);
                        break;
                    case BinaryEWOpCode::Lt:
                        LaunchBinaryEWKernel<scalar_t, scalar_t>(
                                indexer,
                                CPULtElementKernel<scalar_t, scalar_t>);
                        break;
                    case BinaryEWOpCode::Ge:
                        LaunchBinaryEWKernel<scalar_t, scalar_t>(
                                indexer,
                                CPUGeqElementKernel<scalar_t, scalar_t>);
                        break;
                    case BinaryEWOpCode::Le:
                        LaunchBinaryEWKernel<scalar_t, scalar_t>(
                                indexer,
                                CPULeqElementKernel<scalar_t, scalar_t>);
                        break;
                    case BinaryEWOpCode::Eq:
                        LaunchBinaryEWKernel<scalar_t, scalar_t>(
                                indexer,
                                CPUEqElementKernel<scalar_t, scalar_t>);
                        break;
                    case BinaryEWOpCode::Ne:
                        LaunchBinaryEWKernel<scalar_t, scalar_t>(
                                indexer,
                                CPUNeqElementKernel<scalar_t, scalar_t>);
                        break;
                    default:
                        break;
                }
            });
        } else if (dst_dtype == core::Bool) {
            // By default, output is boolean type.
            Indexer indexer({lhs, rhs}, dst,
                            DtypePolicy::INPUT_SAME_OUTPUT_BOOL);

            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                switch (op_code) {
                    case BinaryEWOpCode::LogicalAnd:
                        LaunchBinaryEWKernel<scalar_t, bool>(
                                indexer,
                                CPULogicalAndElementKernel<scalar_t, bool>);
                        break;
                    case BinaryEWOpCode::LogicalOr:
                        LaunchBinaryEWKernel<scalar_t, bool>(
                                indexer,
                                CPULogicalOrElementKernel<scalar_t, bool>);
                        break;
                    case BinaryEWOpCode::LogicalXor:
                        LaunchBinaryEWKernel<scalar_t, bool>(
                                indexer,
                                CPULogicalXorElementKernel<scalar_t, bool>);
                        break;
                    case BinaryEWOpCode::Gt:
                        LaunchBinaryEWKernel<scalar_t, bool>(
                                indexer, CPUGtElementKernel<scalar_t, bool>);
                        break;
                    case BinaryEWOpCode::Lt:
                        LaunchBinaryEWKernel<scalar_t, bool>(
                                indexer, CPULtElementKernel<scalar_t, bool>);
                        break;
                    case BinaryEWOpCode::Ge:
                        LaunchBinaryEWKernel<scalar_t, bool>(
                                indexer, CPUGeqElementKernel<scalar_t, bool>);
                        break;
                    case BinaryEWOpCode::Le:
                        LaunchBinaryEWKernel<scalar_t, bool>(
                                indexer, CPULeqElementKernel<scalar_t, bool>);
                        break;
                    case BinaryEWOpCode::Eq:
                        LaunchBinaryEWKernel<scalar_t, bool>(
                                indexer, CPUEqElementKernel<scalar_t, bool>);
                        break;
                    case BinaryEWOpCode::Ne:
                        LaunchBinaryEWKernel<scalar_t, bool>(
                                indexer, CPUNeqElementKernel<scalar_t, bool>);
                        break;
                    default:
                        break;
                }
            });
        } else {
            utility::LogError(
                    "Boolean op's output type must be boolean or the "
                    "same type as the input.");
        }
    } else if (op_code == BinaryEWOpCode::Maximum ||
               op_code == BinaryEWOpCode::Minimum) {
        Indexer indexer({lhs, rhs}, dst, DtypePolicy::ALL_SAME);
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
            switch (op_code) {
                case BinaryEWOpCode::Maximum:
                    LaunchBinaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUMaxElementKernel<scalar_t>);
                    break;
                case BinaryEWOpCode::Minimum:
                    LaunchBinaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUMinElementKernel<scalar_t>);
                    break;
                default:
                    break;
            }
        });
    } else {
        Indexer indexer({lhs, rhs}, dst, DtypePolicy::ALL_SAME);
        DISPATCH_DTYPE_TO_TEMPLATE(src_dtype, [&]() {
            switch (op_code) {
                case BinaryEWOpCode::Add:
                    LaunchBinaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUAddElementKernel<scalar_t>);
                    break;
                case BinaryEWOpCode::Sub:
                    LaunchBinaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUSubElementKernel<scalar_t>);
                    break;
                case BinaryEWOpCode::Mul:
                    LaunchBinaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUMulElementKernel<scalar_t>);
                    break;
                case BinaryEWOpCode::Div:
                    // The vectorized Div kernel causes a crash in the Python
                    // tests, so use scalar version instead.
                    LaunchBinaryEWKernel<scalar_t, scalar_t>(
                            indexer, CPUDivElementKernel<scalar_t>);
                    break;
                default:
                    break;
            }
        });
    }
}

}  // namespace u3d::core::kernel
