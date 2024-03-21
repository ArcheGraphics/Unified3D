//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <metal_stdlib>
using namespace metal;

#include "Indexer.h"

template <typename src_t, typename dst_t>
[[kernel]] void MetalCopyElementKernel(const device src_t* src, device dst_t* dst) {
    *dst = static_cast<dst_t>(src);
}

void MetalCopyObjectElementKernel(const device char* src_bytes,
                                  device char* dst_bytes,
                                  constant int32_t& object_byte_size) {
    for (int i = 0; i < object_byte_size; ++i) {
        dst_bytes[i] = src_bytes[i];
    }
}

[[kernel]] void MetalCopyObjectElementEntry(device u3d::metal::Indexer& indexer,
                                            constant int32_t& object_byte_size,
                                            uint index [[thread_position_in_grid]]) {
    MetalCopyObjectElementKernel(indexer.GetInputPtr(0, index),
                                 indexer.GetOutputPtr(index), object_byte_size);
}

template <typename scalar_t>
[[kernel]] void MetalSqrtElementKernel(const device scalar_t* src, device scalar_t* dst,
                                       uint index [[thread_position_in_grid]]) {
    *dst = static_cast<scalar_t>(sqrt(static_cast<float>(*src)));
}

template <typename scalar_t>
[[kernel]] void MetalSinElementKernel(const device scalar_t* src, device scalar_t* dst,
                                      uint index [[thread_position_in_grid]]) {
    *dst = static_cast<scalar_t>(sin(static_cast<float>(*src)));
}

template <typename scalar_t>
[[kernel]] void MetalCosElementKernel(const device scalar_t* src, device scalar_t* dst,
                                      uint index [[thread_position_in_grid]]) {
    *dst = static_cast<scalar_t>(cos(static_cast<float>(*src)));
}

template <typename scalar_t>
[[kernel]] void MetalNegElementKernel(const device scalar_t* src, device scalar_t* dst,
                                      uint index [[thread_position_in_grid]]) {
    *dst = -*src;
}

template <typename scalar_t>
[[kernel]] void MetalExpElementKernel(const device scalar_t* src, device scalar_t* dst,
                                      uint index [[thread_position_in_grid]]) {
    *dst = static_cast<scalar_t>(exp(static_cast<float>(*src)));
}

template <typename scalar_t>
[[kernel]] void MetalAbsElementKernel(const device scalar_t* src, device scalar_t* dst,
                                      uint index [[thread_position_in_grid]]) {
    *dst = static_cast<scalar_t>(abs(static_cast<float>(*src)));
}

template <typename scalar_t>
[[kernel]] void MetalIsNanElementKernel(const device scalar_t* src, device bool* dst,
                                        uint index [[thread_position_in_grid]]) {
    *dst = isnan(static_cast<float>(*src));
}

template <typename scalar_t>
[[kernel]] void MetalIsInfElementKernel(const device scalar_t* src, device bool* dst,
                                        uint index [[thread_position_in_grid]]) {
    *dst = isinf(static_cast<float>(*src));
}

template <typename scalar_t>
[[kernel]] void MetalIsFiniteElementKernel(const device scalar_t* src, device bool* dst,
                                           uint index [[thread_position_in_grid]]) {
    *dst = isfinite(static_cast<float>(*src));
}

template <typename scalar_t>
[[kernel]] void MetalFloorElementKernel(const device scalar_t* src, device scalar_t* dst,
                                        uint index [[thread_position_in_grid]]) {
    *dst = static_cast<scalar_t>(floor(static_cast<float>(*src)));
}

template <typename scalar_t>
[[kernel]] void MetalCeilElementKernel(const device scalar_t* src, device scalar_t* dst,
                                       uint index [[thread_position_in_grid]]) {
    *dst = static_cast<scalar_t>(ceil(static_cast<float>(*src)));
}

template <typename scalar_t>
[[kernel]] void MetalRoundElementKernel(const device scalar_t* src, device scalar_t* dst,
                                        uint index [[thread_position_in_grid]]) {
    *dst = static_cast<scalar_t>(round(static_cast<float>(*src)));
}

template <typename scalar_t>
[[kernel]] void MetalTruncElementKernel(const device scalar_t* src, device scalar_t* dst,
                                        uint index [[thread_position_in_grid]]) {
    *dst = static_cast<scalar_t>(trunc(static_cast<float>(*src)));
}

template <typename src_t, typename dst_t>
[[kernel]] void MetalLogicalNotElementKernel(const device src_t* src, device dst_t* dst,
                                             uint index [[thread_position_in_grid]]) {
    *dst = static_cast<dst_t>(!static_cast<bool>(*src));
}
