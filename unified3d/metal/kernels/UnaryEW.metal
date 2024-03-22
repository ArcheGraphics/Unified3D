//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <metal_stdlib>
using namespace metal;

#include "Indexer.h"

template <typename func_t>
void LaunchUnaryEWKernel(device u3d::metal::Indexer& indexer,
                         func_t element_kernel,
                         uint index) {
    element_kernel(indexer.GetInputPtr(0, index), indexer.GetOutputPtr(index));
}

template <typename src_t, typename dst_t, typename func_t>
void LaunchUnaryEWKernel(device u3d::metal::Indexer& indexer,
                         func_t element_kernel,
                         uint index) {
    element_kernel(indexer.GetInputPtr<src_t>(0, index),
                   indexer.GetOutputPtr<dst_t>(index));
}

[[kernel]] void CopyObjectElementKernel(device u3d::metal::Indexer& indexer [[buffer(0)]],
                                        constant int64_t& object_byte_size [[buffer(1)]],
                                        uint index  [[thread_position_in_grid]]) {
    struct Functor {
        int64_t object_byte_size;
        
        Functor(int64_t object_byte_size): object_byte_size(object_byte_size) {}
        
        void operator()(const device char* src_bytes,
                        device char* dst_bytes) {
            for (int i = 0; i < object_byte_size; ++i) {
                dst_bytes[i] = src_bytes[i];
            }
        }
    };
    LaunchUnaryEWKernel(indexer, Functor(object_byte_size), index);
}

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void CopySingleKernel(device scalar_t* dst_ptr [[buffer(0)]],
                                 constant scalar_t& scalar_element [[buffer(1)]],
                                 uint index [[thread_position_in_grid]]) {
    dst_ptr[index] = scalar_element;
}

#define instantiate_CopySingleKernel(tname, type) \
template [[host_name("CopySingleKernel" #tname)]] \
[[kernel]] void CopySingleKernel<type>(\
device type* dst_ptr, \
constant type& scalar_element, \
uint index [[thread_position_in_grid]]);

instantiate_CopySingleKernel(UInt8, uint8_t)
instantiate_CopySingleKernel(UInt16, uint16_t)
instantiate_CopySingleKernel(UInt32, uint32_t)
instantiate_CopySingleKernel(UInt64, uint64_t)
instantiate_CopySingleKernel(Int8, int8_t)
instantiate_CopySingleKernel(Int16, int16_t)
instantiate_CopySingleKernel(Int32, int32_t)
instantiate_CopySingleKernel(Int64, int64_t)
instantiate_CopySingleKernel(Float32, float)
instantiate_CopySingleKernel(Bool, bool)

//--------------------------------------------------------------------------------------------
template <typename src_t, typename dst_t>
[[kernel]] void CopyElementKernel(device u3d::metal::Indexer& indexer [[buffer(0)]],
                                  uint index [[thread_position_in_grid]]) {
    struct Functor {
        void operator()(const device void* src,
                        device void* dst) {
            *static_cast<device dst_t*>(dst) =
            static_cast<dst_t>(*static_cast<const device src_t*>(src));
        }
    };
    LaunchUnaryEWKernel<src_t, dst_t, Functor>(indexer, Functor(), index);
}

#define instantiate_CopyElementKernel(tname, type) \
template [[host_name("CopyElementKernel" #tname)]] \
[[kernel]] void CopyElementKernel<type, type>(\
device u3d::metal::Indexer& indexer, \
uint index [[thread_position_in_grid]]);

instantiate_CopyElementKernel(UInt8, uint8_t)
instantiate_CopyElementKernel(UInt16, uint16_t)
instantiate_CopyElementKernel(UInt32, uint32_t)
instantiate_CopyElementKernel(UInt64, uint64_t)
instantiate_CopyElementKernel(Int8, int8_t)
instantiate_CopyElementKernel(Int16, int16_t)
instantiate_CopyElementKernel(Int32, int32_t)
instantiate_CopyElementKernel(Int64, int64_t)
instantiate_CopyElementKernel(Float32, float)
instantiate_CopyElementKernel(Bool, bool)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void SqrtElementKernel(device u3d::metal::Indexer& indexer [[buffer(0)]],
                                  uint index [[thread_position_in_grid]]) {
    struct Functor {
        void operator()(const device void* src,
                        device void* dst) {
            *static_cast<device scalar_t*>(dst) = static_cast<scalar_t>(sqrt(static_cast<float>(*static_cast<const device scalar_t*>(src))));
        }
    };
    LaunchUnaryEWKernel<scalar_t, scalar_t, Functor>(indexer, Functor(), index);
}

#define instantiate_SqrtElementKernel(tname, type) \
template [[host_name("SqrtElementKernel" #tname)]] \
[[kernel]] void SqrtElementKernel<type>(\
device u3d::metal::Indexer& indexer, \
uint index [[thread_position_in_grid]]);

instantiate_SqrtElementKernel(UInt8, uint8_t)
instantiate_SqrtElementKernel(UInt16, uint16_t)
instantiate_SqrtElementKernel(UInt32, uint32_t)
instantiate_SqrtElementKernel(UInt64, uint64_t)
instantiate_SqrtElementKernel(Int8, int8_t)
instantiate_SqrtElementKernel(Int16, int16_t)
instantiate_SqrtElementKernel(Int32, int32_t)
instantiate_SqrtElementKernel(Int64, int64_t)
instantiate_SqrtElementKernel(Float32, float)
instantiate_SqrtElementKernel(Bool, bool)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void SinElementKernel(device u3d::metal::Indexer& indexer [[buffer(0)]],
                                 uint index [[thread_position_in_grid]]) {
    struct Functor {
        void operator()(const device void* src,
                        device void* dst) {
            *static_cast<device scalar_t*>(dst) = static_cast<scalar_t>(sin(static_cast<float>(*static_cast<const device scalar_t*>(src))));
        }
    };
    LaunchUnaryEWKernel<scalar_t, scalar_t, Functor>(indexer, Functor(), index);
}

#define instantiate_SinElementKernel(tname, type) \
template [[host_name("SinElementKernel" #tname)]] \
[[kernel]] void SinElementKernel<type>(\
device u3d::metal::Indexer& indexer, \
uint index [[thread_position_in_grid]]);

instantiate_SinElementKernel(UInt8, uint8_t)
instantiate_SinElementKernel(UInt16, uint16_t)
instantiate_SinElementKernel(UInt32, uint32_t)
instantiate_SinElementKernel(UInt64, uint64_t)
instantiate_SinElementKernel(Int8, int8_t)
instantiate_SinElementKernel(Int16, int16_t)
instantiate_SinElementKernel(Int32, int32_t)
instantiate_SinElementKernel(Int64, int64_t)
instantiate_SinElementKernel(Float32, float)
instantiate_SinElementKernel(Bool, bool)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void CosElementKernel(device u3d::metal::Indexer& indexer [[buffer(0)]],
                                 uint index [[thread_position_in_grid]]) {
    struct Functor {
        void operator()(const device void* src,
                        device void* dst) {
            *static_cast<device scalar_t*>(dst) = static_cast<scalar_t>(cos(static_cast<float>(*static_cast<const device scalar_t*>(src))));
        }
    };
    LaunchUnaryEWKernel<scalar_t, scalar_t, Functor>(indexer, Functor(), index);
}

#define instantiate_CosElementKernel(tname, type) \
template [[host_name("CosElementKernel" #tname)]] \
[[kernel]] void CosElementKernel<type>(\
device u3d::metal::Indexer& indexer, \
uint index [[thread_position_in_grid]]);


instantiate_CosElementKernel(UInt8, uint8_t)
instantiate_CosElementKernel(UInt16, uint16_t)
instantiate_CosElementKernel(UInt32, uint32_t)
instantiate_CosElementKernel(UInt64, uint64_t)
instantiate_CosElementKernel(Int8, int8_t)
instantiate_CosElementKernel(Int16, int16_t)
instantiate_CosElementKernel(Int32, int32_t)
instantiate_CosElementKernel(Int64, int64_t)
instantiate_CosElementKernel(Float32, float)
instantiate_CosElementKernel(Bool, bool)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void NegElementKernel(device u3d::metal::Indexer& indexer [[buffer(0)]],
                                 uint index [[thread_position_in_grid]]) {
    struct Functor {
        void operator()(const device void* src,
                        device void* dst) {
            *static_cast<device scalar_t*>(dst) = -*static_cast<const device scalar_t*>(src);
        }
    };
    LaunchUnaryEWKernel<scalar_t, scalar_t, Functor>(indexer, Functor(), index);
}

#define instantiate_NegElementKernel(tname, type) \
template [[host_name("NegElementKernel" #tname )]] \
[[kernel]] void NegElementKernel<type>(\
device u3d::metal::Indexer& indexer, \
uint index [[thread_position_in_grid]]);

instantiate_NegElementKernel(UInt8, uint8_t)
instantiate_NegElementKernel(UInt16, uint16_t)
instantiate_NegElementKernel(UInt32, uint32_t)
instantiate_NegElementKernel(UInt64, uint64_t)
instantiate_NegElementKernel(Int8, int8_t)
instantiate_NegElementKernel(Int16, int16_t)
instantiate_NegElementKernel(Int32, int32_t)
instantiate_NegElementKernel(Int64, int64_t)
instantiate_NegElementKernel(Float32, float)
instantiate_NegElementKernel(Bool, bool)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void ExpElementKernel(device u3d::metal::Indexer& indexer [[buffer(0)]],
                                 uint index [[thread_position_in_grid]]) {
    struct Functor {
        void operator()(const device void* src,
                        device void* dst) {
            *static_cast<device scalar_t*>(dst) = static_cast<scalar_t>(exp(static_cast<float>(*static_cast<const device scalar_t*>(src))));
        }
    };
    LaunchUnaryEWKernel<scalar_t, scalar_t, Functor>(indexer, Functor(), index);
}

#define instantiate_ExpElementKernel(tname, type) \
template [[host_name("ExpElementKernel" #tname)]] \
[[kernel]] void ExpElementKernel<type>(\
device u3d::metal::Indexer& indexer, \
uint index [[thread_position_in_grid]]);

instantiate_ExpElementKernel(UInt8, uint8_t)
instantiate_ExpElementKernel(UInt16, uint16_t)
instantiate_ExpElementKernel(UInt32, uint32_t)
instantiate_ExpElementKernel(UInt64, uint64_t)
instantiate_ExpElementKernel(Int8, int8_t)
instantiate_ExpElementKernel(Int16, int16_t)
instantiate_ExpElementKernel(Int32, int32_t)
instantiate_ExpElementKernel(Int64, int64_t)
instantiate_ExpElementKernel(Float32, float)
instantiate_ExpElementKernel(Bool, bool)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void AbsElementKernel(device u3d::metal::Indexer& indexer [[buffer(0)]],
                                 uint index [[thread_position_in_grid]]) {
    struct Functor {
        void operator()(const device void* src,
                        device void* dst) {
            *static_cast<device scalar_t*>(dst) = static_cast<scalar_t>(abs(static_cast<float>(*static_cast<const device scalar_t*>(src))));
        }
    };
    LaunchUnaryEWKernel<scalar_t, scalar_t, Functor>(indexer, Functor(), index);
}

#define instantiate_AbsElementKernel(tname, type) \
template [[host_name("AbsElementKernel" #tname)]] \
[[kernel]] void AbsElementKernel<type>(\
device u3d::metal::Indexer& indexer, \
uint index [[thread_position_in_grid]]);

instantiate_AbsElementKernel(UInt8, uint8_t)
instantiate_AbsElementKernel(UInt16, uint16_t)
instantiate_AbsElementKernel(UInt32, uint32_t)
instantiate_AbsElementKernel(UInt64, uint64_t)
instantiate_AbsElementKernel(Int8, int8_t)
instantiate_AbsElementKernel(Int16, int16_t)
instantiate_AbsElementKernel(Int32, int32_t)
instantiate_AbsElementKernel(Int64, int64_t)
instantiate_AbsElementKernel(Float32, float)
instantiate_AbsElementKernel(Bool, bool)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void IsNanElementKernel(device u3d::metal::Indexer& indexer [[buffer(0)]],
                                   uint index [[thread_position_in_grid]]) {
    struct Functor {
        void operator()(const device void* src,
                        device void* dst) {
            *static_cast<device bool*>(dst) = isnan(static_cast<float>(*static_cast<const device scalar_t*>(src)));
        }
    };
    LaunchUnaryEWKernel<scalar_t, scalar_t, Functor>(indexer, Functor(), index);
}

#define instantiate_IsNanElementKernel(tname, type) \
template [[host_name("IsNanElementKernel" #tname)]] \
[[kernel]] void IsNanElementKernel<type>(\
device u3d::metal::Indexer& indexer, \
uint index [[thread_position_in_grid]]);

instantiate_IsNanElementKernel(UInt8, uint8_t)
instantiate_IsNanElementKernel(UInt16, uint16_t)
instantiate_IsNanElementKernel(UInt32, uint32_t)
instantiate_IsNanElementKernel(UInt64, uint64_t)
instantiate_IsNanElementKernel(Int8, int8_t)
instantiate_IsNanElementKernel(Int16, int16_t)
instantiate_IsNanElementKernel(Int32, int32_t)
instantiate_IsNanElementKernel(Int64, int64_t)
instantiate_IsNanElementKernel(Float32, float)
instantiate_IsNanElementKernel(Bool, bool)
//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void IsInfElementKernel(device u3d::metal::Indexer& indexer [[buffer(0)]],
                                   uint index [[thread_position_in_grid]]) {
    struct Functor {
        void operator()(const device void* src,
                        device void* dst) {
            *static_cast<device bool*>(dst) = isinf(static_cast<float>(*static_cast<const device scalar_t*>(src)));
        }
    };
    LaunchUnaryEWKernel<scalar_t, scalar_t, Functor>(indexer, Functor(), index);
}

#define instantiate_IsInfElementKernel(tname, type) \
template [[host_name("IsInfElementKernel" #tname)]] \
[[kernel]] void IsInfElementKernel<type>(\
device u3d::metal::Indexer& indexer, \
uint index [[thread_position_in_grid]]);

instantiate_IsInfElementKernel(UInt8, uint8_t)
instantiate_IsInfElementKernel(UInt16, uint16_t)
instantiate_IsInfElementKernel(UInt32, uint32_t)
instantiate_IsInfElementKernel(UInt64, uint64_t)
instantiate_IsInfElementKernel(Int8, int8_t)
instantiate_IsInfElementKernel(Int16, int16_t)
instantiate_IsInfElementKernel(Int32, int32_t)
instantiate_IsInfElementKernel(Int64, int64_t)
instantiate_IsInfElementKernel(Float32, float)
instantiate_IsInfElementKernel(Bool, bool)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void IsFiniteElementKernel(device u3d::metal::Indexer& indexer [[buffer(0)]],
                                      uint index [[thread_position_in_grid]]) {
    struct Functor {
        void operator()(const device void* src,
                        device void* dst) {
            *static_cast<device bool*>(dst) = isfinite(static_cast<float>(*static_cast<const device scalar_t*>(src)));
        }
    };
    LaunchUnaryEWKernel<scalar_t, scalar_t, Functor>(indexer, Functor(), index);
}

#define instantiate_IsFiniteElementKernel(tname, type) \
template [[host_name("IsFiniteElementKernel" #tname)]] \
[[kernel]] void IsFiniteElementKernel<type>(\
device u3d::metal::Indexer& indexer, \
uint index [[thread_position_in_grid]]);

instantiate_IsFiniteElementKernel(UInt8, uint8_t)
instantiate_IsFiniteElementKernel(UInt16, uint16_t)
instantiate_IsFiniteElementKernel(UInt32, uint32_t)
instantiate_IsFiniteElementKernel(UInt64, uint64_t)
instantiate_IsFiniteElementKernel(Int8, int8_t)
instantiate_IsFiniteElementKernel(Int16, int16_t)
instantiate_IsFiniteElementKernel(Int32, int32_t)
instantiate_IsFiniteElementKernel(Int64, int64_t)
instantiate_IsFiniteElementKernel(Float32, float)
instantiate_IsFiniteElementKernel(Bool, bool)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void FloorElementKernel(device u3d::metal::Indexer& indexer [[buffer(0)]],
                                   uint index [[thread_position_in_grid]]) {
    struct Functor {
        void operator()(const device void* src,
                        device void* dst) {
            *static_cast<device scalar_t*>(dst) = static_cast<scalar_t>(floor(static_cast<float>(*static_cast<const device scalar_t*>(src))));
        }
    };
    LaunchUnaryEWKernel<scalar_t, scalar_t, Functor>(indexer, Functor(), index);
}

#define instantiate_FloorElementKernel(tname, type) \
template [[host_name("FloorElementKernel" #tname)]] \
[[kernel]] void FloorElementKernel<type>(\
device u3d::metal::Indexer& indexer, \
uint index [[thread_position_in_grid]]);

instantiate_FloorElementKernel(UInt8, uint8_t)
instantiate_FloorElementKernel(UInt16, uint16_t)
instantiate_FloorElementKernel(UInt32, uint32_t)
instantiate_FloorElementKernel(UInt64, uint64_t)
instantiate_FloorElementKernel(Int8, int8_t)
instantiate_FloorElementKernel(Int16, int16_t)
instantiate_FloorElementKernel(Int32, int32_t)
instantiate_FloorElementKernel(Int64, int64_t)
instantiate_FloorElementKernel(Float32, float)
instantiate_FloorElementKernel(Bool, bool)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void CeilElementKernel(device u3d::metal::Indexer& indexer [[buffer(0)]],
                                  uint index [[thread_position_in_grid]]) {
    struct Functor {
        void operator()(const device void* src,
                        device void* dst) {
            *static_cast<device scalar_t*>(dst) = static_cast<scalar_t>(ceil(static_cast<float>(*static_cast<const device scalar_t*>(src))));
        }
    };
    LaunchUnaryEWKernel<scalar_t, scalar_t, Functor>(indexer, Functor(), index);
}

#define instantiate_CeilElementKernel(tname, type) \
template [[host_name("CeilElementKernel" #tname)]] \
[[kernel]] void CeilElementKernel<type>(\
device u3d::metal::Indexer& indexer, \
uint index [[thread_position_in_grid]]);

instantiate_CeilElementKernel(UInt8, uint8_t)
instantiate_CeilElementKernel(UInt16, uint16_t)
instantiate_CeilElementKernel(UInt32, uint32_t)
instantiate_CeilElementKernel(UInt64, uint64_t)
instantiate_CeilElementKernel(Int8, int8_t)
instantiate_CeilElementKernel(Int16, int16_t)
instantiate_CeilElementKernel(Int32, int32_t)
instantiate_CeilElementKernel(Int64, int64_t)
instantiate_CeilElementKernel(Float32, float)
instantiate_CeilElementKernel(Bool, bool)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void RoundElementKernel(device u3d::metal::Indexer& indexer [[buffer(0)]],
                                   uint index [[thread_position_in_grid]]) {
    struct Functor {
        void operator()(const device void* src,
                        device void* dst) {
            *static_cast<device scalar_t*>(dst) = static_cast<scalar_t>(round(static_cast<float>(*static_cast<const device scalar_t*>(src))));
        }
    };
    LaunchUnaryEWKernel<scalar_t, scalar_t, Functor>(indexer, Functor(), index);
}

#define instantiate_RoundElementKernel(tname, type) \
template [[host_name("RoundElementKernel" #tname)]] \
[[kernel]] void RoundElementKernel<type>(\
device u3d::metal::Indexer& indexer, \
uint index [[thread_position_in_grid]]);

instantiate_RoundElementKernel(UInt8, uint8_t)
instantiate_RoundElementKernel(UInt16, uint16_t)
instantiate_RoundElementKernel(UInt32, uint32_t)
instantiate_RoundElementKernel(UInt64, uint64_t)
instantiate_RoundElementKernel(Int8, int8_t)
instantiate_RoundElementKernel(Int16, int16_t)
instantiate_RoundElementKernel(Int32, int32_t)
instantiate_RoundElementKernel(Int64, int64_t)
instantiate_RoundElementKernel(Float32, float)
instantiate_RoundElementKernel(Bool, bool)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void TruncElementKernel(device u3d::metal::Indexer& indexer [[buffer(0)]],
                                   uint index [[thread_position_in_grid]]) {
    struct Functor {
        void operator()(const device void* src,
                        device void* dst) {
            *static_cast<device scalar_t*>(dst) = static_cast<scalar_t>(trunc(static_cast<float>(*static_cast<const device scalar_t*>(src))));
        }
    };
    LaunchUnaryEWKernel<scalar_t, scalar_t, Functor>(indexer, Functor(), index);
}

#define instantiate_TruncElementKernel(tname, type) \
template [[host_name("TruncElementKernel" #tname)]] \
[[kernel]] void TruncElementKernel<type>(\
device u3d::metal::Indexer& indexer, \
uint index [[thread_position_in_grid]]);

instantiate_TruncElementKernel(UInt8, uint8_t)
instantiate_TruncElementKernel(UInt16, uint16_t)
instantiate_TruncElementKernel(UInt32, uint32_t)
instantiate_TruncElementKernel(UInt64, uint64_t)
instantiate_TruncElementKernel(Int8, int8_t)
instantiate_TruncElementKernel(Int16, int16_t)
instantiate_TruncElementKernel(Int32, int32_t)
instantiate_TruncElementKernel(Int64, int64_t)
instantiate_TruncElementKernel(Float32, float)
instantiate_TruncElementKernel(Bool, bool)

//--------------------------------------------------------------------------------------------
template <typename src_t, typename dst_t>
[[kernel]] void LogicalNotElementKernel(device u3d::metal::Indexer& indexer [[buffer(0)]],
                                        uint index [[thread_position_in_grid]]) {
    struct Functor {
        void operator()(const device void* src,
                        device void* dst) {
            *static_cast<device dst_t*>(dst) = static_cast<dst_t>(!static_cast<bool>(*static_cast<const device src_t*>(src)));
        }
    };
    LaunchUnaryEWKernel<src_t, dst_t, Functor>(indexer, Functor(), index);
}

#define instantiate_LogicalNotElementKernel(tname, type) \
template [[host_name("LogicalNotElementKernel" #tname)]] \
[[kernel]] void LogicalNotElementKernel<type, type>(\
device u3d::metal::Indexer& indexer, \
uint index [[thread_position_in_grid]]);

instantiate_LogicalNotElementKernel(UInt8, uint8_t)
instantiate_LogicalNotElementKernel(UInt16, uint16_t)
instantiate_LogicalNotElementKernel(UInt32, uint32_t)
instantiate_LogicalNotElementKernel(UInt64, uint64_t)
instantiate_LogicalNotElementKernel(Int8, int8_t)
instantiate_LogicalNotElementKernel(Int16, int16_t)
instantiate_LogicalNotElementKernel(Int32, int32_t)
instantiate_LogicalNotElementKernel(Int64, int64_t)
instantiate_LogicalNotElementKernel(Float32, float)
instantiate_LogicalNotElementKernel(Bool, bool)
