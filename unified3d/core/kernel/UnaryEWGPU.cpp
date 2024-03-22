//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unified3d/core/Dispatch.h"
#include "unified3d/core/Indexer.h"
#include "unified3d/core/Tensor.h"
#include "unified3d/core/kernel/UnaryEW.h"
#include "unified3d/metal/Device.h"
#include <Metal/Metal.hpp>

namespace u3d::core::kernel {
void CopyGPU(const Tensor& src, Tensor& dst) {
    // It has been checked that
    // - src and dst have the same dtype
    // - at least one of src or dst is Metal device
    SizeVector shape = src.GetShape();
    Dtype src_dtype = src.GetDtype();
    Dtype dst_dtype = dst.GetDtype();

    Device src_device = src.GetDevice();
    Device dst_device = dst.GetDevice();
    auto& d = metal::Device::GetInstance();

    if (src_device.IsGPU() && dst_device.IsGPU()) {
        if (src.IsContiguous() && dst.IsContiguous() &&
            src.GetShape() == dst.GetShape() && src_dtype == dst_dtype) {
            // MemoryManager handles p2p and non-p2p device copy.
            MemoryManager::MemcpyOnGpu(
                    dst.GetDataView(), src.GetDataView(),
                    src_dtype.ByteSize() * shape.NumElements());
        } else if (dst.NumElements() > 1 && dst.IsContiguous() &&
                   src.NumElements() == 1 && !src_dtype.IsObject()) {
            int64_t num_elements = dst.NumElements();

            DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dst_dtype, [&]() {
                auto scalar_element = src.To(dst_dtype).Item<scalar_t>();
                auto dst_ptr = dst.GetDataView().GpuAddress();
                {
                    auto compute_encoder = d.get_command_encoder(0);
                    auto kernel = d.get_kernel(fmt::format("CopySingleKernel{}", dst_dtype.ToString()));
                    compute_encoder->setComputePipelineState(kernel);
                    compute_encoder->setBytes(&dst_ptr, sizeof(uint64_t), 0);
                    compute_encoder->setBytes(&scalar_element, sizeof(scalar_t),
                                              1);
                    compute_encoder->dispatchThreads(
                            MTL::Size(num_elements, 1, 1),
                            MTL::Size(128, 1, 1));
                }
            });
        } else if (src_device == dst_device) {
            // For more optimized version, one can check if P2P from src to
            // dst is enabled, then put synchronization with streams on both
            // src and dst to wait for copy kernel to complete.
            Indexer indexer({src}, dst, DtypePolicy::NONE);
            auto compute_encoder = d.get_command_encoder(0);

            if (src.GetDtype().IsObject()) {
                int64_t object_byte_size = src.GetDtype().ByteSize();
                {
                    auto kernel = d.get_kernel("CopyObjectElementKernel");
                    compute_encoder->setComputePipelineState(kernel);
                    compute_encoder->setBytes(&indexer,
                                              sizeof(u3d::metal::Indexer), 0);
                    compute_encoder->setBytes(&object_byte_size,
                                              sizeof(int64_t), 1);
                    compute_encoder->dispatchThreads(
                            MTL::Size(indexer.NumWorkloads(), 1, 1),
                            MTL::Size(128, 1, 1));
                }

            } else {
                DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(src_dtype, [&]() {
                    using src_t = scalar_t;
                    DISPATCH_DTYPE_TO_TEMPLATE_WITH_BOOL(dst_dtype, [&]() {
                        using dst_t = scalar_t;
                        {
                            auto kernel = d.get_kernel(fmt::format("CopyElementKernel{}", dst_dtype.ToString()));
                            compute_encoder->setComputePipelineState(kernel);
                            compute_encoder->setBytes(
                                    &indexer, sizeof(u3d::metal::Indexer), 0);
                            compute_encoder->dispatchThreads(
                                    MTL::Size(indexer.NumWorkloads(), 1, 1),
                                    MTL::Size(128, 1, 1));
                        }
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
            MemoryManager::MemcpyOnCpu(
                    dst.GetDataView(), src_conti.GetDataView(),
                    src_dtype.ByteSize() * shape.NumElements());
        } else {
            dst.CopyFrom(src.Contiguous().To(dst_device));
        }
    } else {
        utility::LogError("Wrong device type {} -> {}", src_device.ToString(),
                          dst_device.ToString());
    }
}

void DispatchUnaryEW(const std::string& name, const Indexer& indexer) {
    auto& d = metal::Device::GetInstance();
    auto compute_encoder = d.get_command_encoder(0);
    auto kernel = d.get_kernel(name);
    compute_encoder->setComputePipelineState(kernel);
    compute_encoder->setBytes(&indexer, sizeof(u3d::metal::Indexer), 0);
    compute_encoder->dispatchThreads(MTL::Size(indexer.NumWorkloads(), 1, 1),
                                     MTL::Size(128, 1, 1));
}

void UnaryEWGPU(const Tensor& src, Tensor& dst, UnaryEWOpCode op_code) {
    // src and dst have been changed to have the same shape, dtype, device.
    Dtype src_dtype = src.GetDtype();
    Dtype dst_dtype = dst.GetDtype();

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
                DispatchUnaryEW(fmt::format("LogicalNotElementKernel{}", src_dtype.ToString()), indexer);
            } else if (dst_dtype == core::Bool) {
                Indexer indexer({src}, dst,
                                DtypePolicy::INPUT_SAME_OUTPUT_BOOL);
                DispatchUnaryEW(fmt::format("LogicalNotElementKernel{}", src_dtype.ToString()), indexer);
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
                DispatchUnaryEW(fmt::format("IsNanElementKernel{}", src_dtype.ToString()), indexer);
            } else if (op_code == UnaryEWOpCode::IsInf) {
                DispatchUnaryEW(fmt::format("IsInfElementKernel{}", src_dtype.ToString()), indexer);
            } else if (op_code == UnaryEWOpCode::IsFinite) {
                DispatchUnaryEW(fmt::format("IsFiniteElementKernel{}", src_dtype.ToString()), indexer);
            }
        });
    } else {
        Indexer indexer({src}, dst, DtypePolicy::ALL_SAME);
        DISPATCH_DTYPE_TO_TEMPLATE(src_dtype, [&]() {
            switch (op_code) {
                case UnaryEWOpCode::Sqrt:
                    assert_dtype_is_float(src_dtype);
                    DispatchUnaryEW(fmt::format("SqrtElementKernel{}", src_dtype.ToString()), indexer);
                    break;
                case UnaryEWOpCode::Sin:
                    assert_dtype_is_float(src_dtype);
                    DispatchUnaryEW(fmt::format("SinElementKernel{}", src_dtype.ToString()), indexer);
                    break;
                case UnaryEWOpCode::Cos:
                    assert_dtype_is_float(src_dtype);
                    DispatchUnaryEW(fmt::format("CosElementKernel{}", src_dtype.ToString()), indexer);
                    break;
                case UnaryEWOpCode::Neg:
                    DispatchUnaryEW(fmt::format("NegElementKernel{}", src_dtype.ToString()), indexer);
                    break;
                case UnaryEWOpCode::Exp:
                    assert_dtype_is_float(src_dtype);
                    DispatchUnaryEW(fmt::format("ExpElementKernel{}", src_dtype.ToString()), indexer);
                    break;
                case UnaryEWOpCode::Abs: {
                    DispatchUnaryEW(fmt::format("AbsElementKernel{}", src_dtype.ToString()), indexer);
                } break;
                case UnaryEWOpCode::Floor: {
                    DispatchUnaryEW(fmt::format("FloorElementKernel{}", src_dtype.ToString()), indexer);
                } break;
                case UnaryEWOpCode::Ceil: {
                    DispatchUnaryEW(fmt::format("CeilElementKernel{}", src_dtype.ToString()), indexer);
                } break;
                case UnaryEWOpCode::Round: {
                    DispatchUnaryEW(fmt::format("RoundElementKernel{}", src_dtype.ToString()), indexer);
                } break;
                case UnaryEWOpCode::Trunc: {
                    DispatchUnaryEW(fmt::format("TruncElementKernel{}", src_dtype.ToString()), indexer);
                } break;
                default:
                    utility::LogError("Unimplemented op_code for UnaryEWMetal");
                    break;
            }
        });
    }
}

}  // namespace u3d::core::kernel
