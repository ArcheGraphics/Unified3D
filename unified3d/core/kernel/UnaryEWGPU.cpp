//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unified3d/core/Dispatch.h"
#include "unified3d/core/Indexer.h"
#include "unified3d/core/Tensor.h"
#include "unified3d/core/kernel/UnaryEW.h"

namespace u3d::core::kernel {

// Cannot be a static function since on Windows a function enclosing
// __host__ __device__ lambda function must have external linkage.
template <typename func_t>
void LaunchUnaryEWKernel(const Device& device,
                         const Indexer& indexer,
                         const func_t& element_kernel) {
    auto element_func = [=](int64_t i) {
        element_kernel(indexer.GetInputView(0, i), indexer.GetOutputView(i));
    };
    core::ParallelFor(device, indexer.NumWorkloads(), element_func);
}

template <typename src_t, typename dst_t, typename func_t>
void LaunchUnaryEWKernel(const Device& device,
                         const Indexer& indexer,
                         const func_t& element_kernel) {
    auto element_func = [=](int64_t i) {
        element_kernel(indexer.GetInputView<src_t>(0, i),
                       indexer.GetOutputView<dst_t>(i));
    };
    core::ParallelFor(device, indexer.NumWorkloads(), element_func);
}

template <typename src_t, typename dst_t>
static void MetalCopyElementKernel(const src_t* src, dst_t* dst) {
    *dst = static_cast<dst_t>(src);
}

static void MetalCopyObjectElementKernel(const char* src_bytes,
                                         char* dst_bytes,
                                         int64_t object_byte_size) {
    for (int i = 0; i < object_byte_size; ++i) {
        dst_bytes[i] = src_bytes[i];
    }
}

template <typename scalar_t>
static void MetalSqrtElementKernel(const scalar_t* src, scalar_t* dst) {
    *dst = static_cast<scalar_t>(sqrt(static_cast<float>(*src)));
}

template <typename scalar_t>
static void MetalSinElementKernel(const scalar_t* src, scalar_t* dst) {
    *dst = static_cast<scalar_t>(sin(static_cast<float>(*src)));
}

template <typename scalar_t>
static void MetalCosElementKernel(const scalar_t* src, scalar_t* dst) {
    *dst = static_cast<scalar_t>(cos(static_cast<float>(*src)));
}

template <typename scalar_t>
static void MetalNegElementKernel(const scalar_t* src, scalar_t* dst) {
    *dst = -*src;
}

template <typename scalar_t>
static void MetalExpElementKernel(const scalar_t* src, scalar_t* dst) {
    *dst = static_cast<scalar_t>(exp(static_cast<float>(*src)));
}

template <typename scalar_t>
static void MetalAbsElementKernel(const scalar_t* src, scalar_t* dst) {
    *dst = static_cast<scalar_t>(abs(static_cast<double>(*src)));
}

template <typename scalar_t>
static void MetalIsNanElementKernel(const scalar_t* src, bool* dst) {
    *dst = isnan(static_cast<float>(*src));
}

template <typename scalar_t>
static void MetalIsInfElementKernel(const scalar_t* src, bool* dst) {
    *dst = isinf(static_cast<float>(*src));
}

template <typename scalar_t>
static void MetalIsFiniteElementKernel(const scalar_t* src, bool* dst) {
    *dst = isfinite(static_cast<float>(*src));
}

template <typename scalar_t>
static void MetalFloorElementKernel(const scalar_t* src, scalar_t* dst) {
    *dst = static_cast<scalar_t>(floor(static_cast<float>(*src)));
}

template <typename scalar_t>
static void MetalCeilElementKernel(const scalar_t* src, scalar_t* dst) {
    *dst = static_cast<scalar_t>(ceil(static_cast<float>(*src)));
}

template <typename scalar_t>
static void MetalRoundElementKernel(const scalar_t* src, scalar_t* dst) {
    *dst = static_cast<scalar_t>(round(static_cast<float>(*src)));
}

template <typename scalar_t>
static void MetalTruncElementKernel(const scalar_t* src, scalar_t* dst) {
    *dst = static_cast<scalar_t>(trunc(static_cast<float>(*src)));
}

template <typename src_t, typename dst_t>
static void MetalLogicalNotElementKernel(const src_t* src, dst_t* dst) {
    *dst = static_cast<dst_t>(!static_cast<bool>(*src));
}

void CopyMetal(const Tensor& src, Tensor& dst) {
    // It has been checked that
    // - src and dst have the same dtype
    // - at least one of src or dst is Metal device
    SizeVector shape = src.GetShape();
    Dtype src_dtype = src.GetDtype();
    Dtype dst_dtype = dst.GetDtype();

    Device src_device = src.GetDevice();
    Device dst_device = dst.GetDevice();

    if (src_device.IsGPU() && dst_device.IsGPU()) {
        if (src.IsContiguous() && dst.IsContiguous() &&
            src.GetShape() == dst.GetShape() && src_dtype == dst_dtype) {
            // MemoryManager handles p2p and non-p2p device copy.
            MemoryManager::Memcpy(dst.GetDataView(), dst_device,
                                  src.GetDataView(), src_device,
                                  src_dtype.ByteSize() * shape.NumElements());
        } else if (dst.NumElements() > 1 && dst.IsContiguous() &&
                   src.NumElements() == 1 && !src_dtype.IsObject()) {
            int64_t num_elements = dst.NumElements();

            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dst_dtype, [&]() {
                auto scalar_element = src.To(dst_dtype).Item<scalar_t>();
                auto dst_ptr = dst.GetDataView().GpuAddress();
                ParallelFor(src_device, num_elements,
                            [=](int64_t workload_idx) {
                                dst_ptr[workload_idx] = scalar_element;
                            });
            });
        } else if (src_device == dst_device) {
            // For more optimized version, one can check if P2P from src to
            // dst is enabled, then put synchronization with streams on both
            // src and dst to wait for copy kernel to complete.
            Indexer indexer({src}, dst, DtypePolicy::NONE);
            if (src.GetDtype().IsObject()) {
                int64_t object_byte_size = src.GetDtype().ByteSize();
                LaunchUnaryEWKernel(
                        src_device, indexer,
                        [=](const metal::Buffer src, metal::Buffer dst) {
                            MetalCopyObjectElementKernel(src, dst,
                                                         object_byte_size);
                        });

            } else {
                DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                    using src_t = scalar_t;
                    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dst_dtype, [&]() {
                        using dst_t = scalar_t;
                        LaunchUnaryEWKernel<src_t, dst_t>(
                                src_device, indexer,
                                // Need to wrap as extended Metal lambda
                                // function
                                [](const metal::Buffer src, metal::Buffer dst) {
                                    MetalCopyElementKernel<src_t, dst_t>(src,
                                                                         dst);
                                });
                    });
                });
            }
        } else {
            dst.CopyFrom(src.Contiguous().To(dst_device));
        }
    } else if (src_device.IsCPU() && dst_device.IsGPU() ||
               src_device.IsGPU() && dst_device.IsCPU()) {
        Tensor src_conti = src.Contiguous();  // No op if already contiguous
        if (dst.IsContiguous() && src.GetShape() == dst.GetShape() &&
            src_dtype == dst_dtype) {
            MemoryManager::Memcpy(dst.GetDataView(), dst_device,
                                  src_conti.GetDataView(),
                                  src_conti.GetDevice(),
                                  src_dtype.ByteSize() * shape.NumElements());
        } else {
            dst.CopyFrom(src.Contiguous().To(dst_device));
        }
    } else {
        utility::LogError("Wrong device type {} -> {}", src_device.ToString(),
                          dst_device.ToString());
    }
}

void UnaryEWMetal(const Tensor& src, Tensor& dst, UnaryEWOpCode op_code) {
    // src and dst have been changed to have the same shape, dtype, device.
    Dtype src_dtype = src.GetDtype();
    Dtype dst_dtype = dst.GetDtype();
    Device src_device = src.GetDevice();

    auto assert_dtype_is_float = [](Dtype dtype) -> void {
        if (dtype != core::Float32) {
            utility::LogError("Only supports Float32, but {} is used.",
                              dtype.ToString());
        }
    };

    if (op_code == UnaryEWOpCode::LogicalNot) {
        DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
            if (dst_dtype == src_dtype) {
                Indexer indexer({src}, dst, DtypePolicy::ALL_SAME);
                LaunchUnaryEWKernel<scalar_t, scalar_t>(
                        src_device, indexer,
                        [](const metal::Buffer src, metal::Buffer dst) {
                            MetalLogicalNotElementKernel<scalar_t, scalar_t>(
                                    src, dst);
                        });
            } else if (dst_dtype == core::Bool) {
                Indexer indexer({src}, dst,
                                DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
                LaunchUnaryEWKernel<scalar_t, bool>(
                        src_device, indexer,
                        [](const metal::Buffer src, metal::Buffer dst) {
                            MetalLogicalNotElementKernel<scalar_t, bool>(src,
                                                                         dst);
                        });
            } else {
                utility::LogError(
                        "Boolean op's output type must be boolean or the "
                        "same type as the input.");
            }
        });
    } else if (op_code == UnaryEWOpCode::IsNan ||
               op_code == UnaryEWOpCode::IsInf ||
               op_code == UnaryEWOpCode::IsFinite) {
        assert_dtype_is_float(src_dtype);
        Indexer indexer({src}, dst, DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
        DISPATCH_DTYPE_TO_TEMPLATE(src_dtype, [&]() {
            if (op_code == UnaryEWOpCode::IsNan) {
                LaunchUnaryEWKernel<scalar_t, bool>(
                        src_device, indexer,
                        [](const metal::Buffer src, metal::Buffer dst) {
                            MetalIsNanElementKernel<scalar_t>(src, dst);
                        });
            } else if (op_code == UnaryEWOpCode::IsInf) {
                LaunchUnaryEWKernel<scalar_t, bool>(
                        src_device, indexer,
                        [](const metal::Buffer src, metal::Buffer dst) {
                            MetalIsInfElementKernel<scalar_t>(src, dst);
                        });
            } else if (op_code == UnaryEWOpCode::IsFinite) {
                LaunchUnaryEWKernel<scalar_t, bool>(
                        src_device, indexer,
                        [](const metal::Buffer src, metal::Buffer dst) {
                            MetalIsFiniteElementKernel<scalar_t>(src, dst);
                        });
            }
        });
    } else {
        Indexer indexer({src}, dst, DtypePolicy::ALL_SAME);
        DISPATCH_DTYPE_TO_TEMPLATE(src_dtype, [&]() {
            switch (op_code) {
                case UnaryEWOpCode::Sqrt:
                    assert_dtype_is_float(src_dtype);
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [](const metal::Buffer src, metal::Buffer dst) {
                                MetalSqrtElementKernel<scalar_t>(src, dst);
                            });
                    break;
                case UnaryEWOpCode::Sin:
                    assert_dtype_is_float(src_dtype);
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [](const metal::Buffer src, metal::Buffer dst) {
                                MetalSinElementKernel<scalar_t>(src, dst);
                            });
                    break;
                case UnaryEWOpCode::Cos:
                    assert_dtype_is_float(src_dtype);
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [](const metal::Buffer src, metal::Buffer dst) {
                                MetalCosElementKernel<scalar_t>(src, dst);
                            });
                    break;
                case UnaryEWOpCode::Neg:
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [](const metal::Buffer src, metal::Buffer dst) {
                                MetalNegElementKernel<scalar_t>(src, dst);
                            });
                    break;
                case UnaryEWOpCode::Exp:
                    assert_dtype_is_float(src_dtype);
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [](const metal::Buffer src, metal::Buffer dst) {
                                MetalExpElementKernel<scalar_t>(src, dst);
                            });
                    break;
                case UnaryEWOpCode::Abs:
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [](const metal::Buffer src, metal::Buffer dst) {
                                MetalAbsElementKernel<scalar_t>(src, dst);
                            });
                    break;
                case UnaryEWOpCode::Floor:
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [](const metal::Buffer src, metal::Buffer dst) {
                                MetalFloorElementKernel<scalar_t>(src, dst);
                            });
                    break;
                case UnaryEWOpCode::Ceil:
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [](const metal::Buffer src, metal::Buffer dst) {
                                MetalCeilElementKernel<scalar_t>(src, dst);
                            });
                    break;
                case UnaryEWOpCode::Round:
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [](const metal::Buffer src, metal::Buffer dst) {
                                MetalRoundElementKernel<scalar_t>(src, dst);
                            });
                    break;
                case UnaryEWOpCode::Trunc:
                    LaunchUnaryEWKernel<scalar_t, scalar_t>(
                            src_device, indexer,
                            [](const metal::Buffer src, metal::Buffer dst) {
                                MetalTruncElementKernel<scalar_t>(src, dst);
                            });
                    break;
                default:
                    utility::LogError("Unimplemented op_code for UnaryEWMetal");
                    break;
            }
        });
    }
}

}  // namespace u3d::core::kernel
