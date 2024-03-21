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

[[kernel]] void CopyObjectElementKernel(device u3d::metal::Indexer& indexer,
                                        constant int32_t& object_byte_size,
                                        uint index  [[thread_position_in_grid]]) {
    struct Functor {
        int32_t object_byte_size;
        
        Functor(int32_t object_byte_size): object_byte_size(object_byte_size) {}
        
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
[[kernel]] void CopySingleKernel(device scalar_t* dst_ptr,
                                 constant scalar_t& scalar_element,
                                 uint index [[thread_position_in_grid]]) {
    dst_ptr[index] = scalar_element;
}

#define instantiate_CopySingleKernel(tname, type) \
template [[host_name("CopySingleKernel" #tname)]] \
[[kernel]] void CopySingleKernel<type>(\
device type* dst_ptr, \
constant type& scalar_element, \
uint index [[thread_position_in_grid]]);

instantiate_CopySingleKernel(uint8, uint8_t)
instantiate_CopySingleKernel(uint16, uint16_t)
instantiate_CopySingleKernel(uint32, uint32_t)
instantiate_CopySingleKernel(uint64, uint64_t)
instantiate_CopySingleKernel(int8, int8_t)
instantiate_CopySingleKernel(int16, int16_t)
instantiate_CopySingleKernel(int32, int32_t)
instantiate_CopySingleKernel(int64, int64_t)
instantiate_CopySingleKernel(float16, half)
instantiate_CopySingleKernel(float32, float)

//--------------------------------------------------------------------------------------------
template <typename src_t, typename dst_t>
[[kernel]] void CopyElementKernel(device u3d::metal::Indexer& indexer,
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

instantiate_CopyElementKernel(uint8, uint8_t)
instantiate_CopyElementKernel(uint16, uint16_t)
instantiate_CopyElementKernel(uint32, uint32_t)
instantiate_CopyElementKernel(uint64, uint64_t)
instantiate_CopyElementKernel(int8, int8_t)
instantiate_CopyElementKernel(int16, int16_t)
instantiate_CopyElementKernel(int32, int32_t)
instantiate_CopyElementKernel(int64, int64_t)
instantiate_CopyElementKernel(float16, half)
instantiate_CopyElementKernel(float32, float)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void SqrtElementKernel(device u3d::metal::Indexer& indexer,
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

instantiate_SqrtElementKernel(uint8, uint8_t)
instantiate_SqrtElementKernel(uint16, uint16_t)
instantiate_SqrtElementKernel(uint32, uint32_t)
instantiate_SqrtElementKernel(uint64, uint64_t)
instantiate_SqrtElementKernel(int8, int8_t)
instantiate_SqrtElementKernel(int16, int16_t)
instantiate_SqrtElementKernel(int32, int32_t)
instantiate_SqrtElementKernel(int64, int64_t)
instantiate_SqrtElementKernel(float16, half)
instantiate_SqrtElementKernel(float32, float)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void SinElementKernel(device u3d::metal::Indexer& indexer,
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

instantiate_SinElementKernel(uint8, uint8_t)
instantiate_SinElementKernel(uint16, uint16_t)
instantiate_SinElementKernel(uint32, uint32_t)
instantiate_SinElementKernel(uint64, uint64_t)
instantiate_SinElementKernel(int8, int8_t)
instantiate_SinElementKernel(int16, int16_t)
instantiate_SinElementKernel(int32, int32_t)
instantiate_SinElementKernel(int64, int64_t)
instantiate_SinElementKernel(float16, half)
instantiate_SinElementKernel(float32, float)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void CosElementKernel(device u3d::metal::Indexer& indexer,
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

instantiate_CosElementKernel(uint8, uint8_t)
instantiate_CosElementKernel(uint16, uint16_t)
instantiate_CosElementKernel(uint32, uint32_t)
instantiate_CosElementKernel(uint64, uint64_t)
instantiate_CosElementKernel(int8, int8_t)
instantiate_CosElementKernel(int16, int16_t)
instantiate_CosElementKernel(int32, int32_t)
instantiate_CosElementKernel(int64, int64_t)
instantiate_CosElementKernel(float16, half)
instantiate_CosElementKernel(float32, float)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void NegElementKernel(device u3d::metal::Indexer& indexer,
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

instantiate_NegElementKernel(uint8, uint8_t)
instantiate_NegElementKernel(uint16, uint16_t)
instantiate_NegElementKernel(uint32, uint32_t)
instantiate_NegElementKernel(uint64, uint64_t)
instantiate_NegElementKernel(int8, int8_t)
instantiate_NegElementKernel(int16, int16_t)
instantiate_NegElementKernel(int32, int32_t)
instantiate_NegElementKernel(int64, int64_t)
instantiate_NegElementKernel(float16, half)
instantiate_NegElementKernel(float32, float)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void ExpElementKernel(device u3d::metal::Indexer& indexer,
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

instantiate_ExpElementKernel(uint8, uint8_t)
instantiate_ExpElementKernel(uint16, uint16_t)
instantiate_ExpElementKernel(uint32, uint32_t)
instantiate_ExpElementKernel(uint64, uint64_t)
instantiate_ExpElementKernel(int8, int8_t)
instantiate_ExpElementKernel(int16, int16_t)
instantiate_ExpElementKernel(int32, int32_t)
instantiate_ExpElementKernel(int64, int64_t)
instantiate_ExpElementKernel(float16, half)
instantiate_ExpElementKernel(float32, float)
//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void AbsElementKernel(device u3d::metal::Indexer& indexer,
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

instantiate_AbsElementKernel(uint8, uint8_t)
instantiate_AbsElementKernel(uint16, uint16_t)
instantiate_AbsElementKernel(uint32, uint32_t)
instantiate_AbsElementKernel(uint64, uint64_t)
instantiate_AbsElementKernel(int8, int8_t)
instantiate_AbsElementKernel(int16, int16_t)
instantiate_AbsElementKernel(int32, int32_t)
instantiate_AbsElementKernel(int64, int64_t)
instantiate_AbsElementKernel(float16, half)
instantiate_AbsElementKernel(float32, float)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void IsNanElementKernel(device u3d::metal::Indexer& indexer,
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

instantiate_IsNanElementKernel(uint8, uint8_t)
instantiate_IsNanElementKernel(uint16, uint16_t)
instantiate_IsNanElementKernel(uint32, uint32_t)
instantiate_IsNanElementKernel(uint64, uint64_t)
instantiate_IsNanElementKernel(int8, int8_t)
instantiate_IsNanElementKernel(int16, int16_t)
instantiate_IsNanElementKernel(int32, int32_t)
instantiate_IsNanElementKernel(int64, int64_t)
instantiate_IsNanElementKernel(float16, half)
instantiate_IsNanElementKernel(float32, float)
//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void IsInfElementKernel(device u3d::metal::Indexer& indexer,
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

instantiate_IsInfElementKernel(uint8, uint8_t)
instantiate_IsInfElementKernel(uint16, uint16_t)
instantiate_IsInfElementKernel(uint32, uint32_t)
instantiate_IsInfElementKernel(uint64, uint64_t)
instantiate_IsInfElementKernel(int8, int8_t)
instantiate_IsInfElementKernel(int16, int16_t)
instantiate_IsInfElementKernel(int32, int32_t)
instantiate_IsInfElementKernel(int64, int64_t)
instantiate_IsInfElementKernel(float16, half)
instantiate_IsInfElementKernel(float32, float)
//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void IsFiniteElementKernel(device u3d::metal::Indexer& indexer,
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

instantiate_IsFiniteElementKernel(uint8, uint8_t)
instantiate_IsFiniteElementKernel(uint16, uint16_t)
instantiate_IsFiniteElementKernel(uint32, uint32_t)
instantiate_IsFiniteElementKernel(uint64, uint64_t)
instantiate_IsFiniteElementKernel(int8, int8_t)
instantiate_IsFiniteElementKernel(int16, int16_t)
instantiate_IsFiniteElementKernel(int32, int32_t)
instantiate_IsFiniteElementKernel(int64, int64_t)
instantiate_IsFiniteElementKernel(float16, half)
instantiate_IsFiniteElementKernel(float32, float)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void FloorElementKernel(device u3d::metal::Indexer& indexer,
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

instantiate_FloorElementKernel(uint8, uint8_t)
instantiate_FloorElementKernel(uint16, uint16_t)
instantiate_FloorElementKernel(uint32, uint32_t)
instantiate_FloorElementKernel(uint64, uint64_t)
instantiate_FloorElementKernel(int8, int8_t)
instantiate_FloorElementKernel(int16, int16_t)
instantiate_FloorElementKernel(int32, int32_t)
instantiate_FloorElementKernel(int64, int64_t)
instantiate_FloorElementKernel(float16, half)
instantiate_FloorElementKernel(float32, float)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void CeilElementKernel(device u3d::metal::Indexer& indexer,
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

instantiate_CeilElementKernel(uint8, uint8_t)
instantiate_CeilElementKernel(uint16, uint16_t)
instantiate_CeilElementKernel(uint32, uint32_t)
instantiate_CeilElementKernel(uint64, uint64_t)
instantiate_CeilElementKernel(int8, int8_t)
instantiate_CeilElementKernel(int16, int16_t)
instantiate_CeilElementKernel(int32, int32_t)
instantiate_CeilElementKernel(int64, int64_t)
instantiate_CeilElementKernel(float16, half)
instantiate_CeilElementKernel(float32, float)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void RoundElementKernel(device u3d::metal::Indexer& indexer,
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

instantiate_RoundElementKernel(uint8, uint8_t)
instantiate_RoundElementKernel(uint16, uint16_t)
instantiate_RoundElementKernel(uint32, uint32_t)
instantiate_RoundElementKernel(uint64, uint64_t)
instantiate_RoundElementKernel(int8, int8_t)
instantiate_RoundElementKernel(int16, int16_t)
instantiate_RoundElementKernel(int32, int32_t)
instantiate_RoundElementKernel(int64, int64_t)
instantiate_RoundElementKernel(float16, half)
instantiate_RoundElementKernel(float32, float)

//--------------------------------------------------------------------------------------------
template <typename scalar_t>
[[kernel]] void TruncElementKernel(device u3d::metal::Indexer& indexer,
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

instantiate_TruncElementKernel(uint8, uint8_t)
instantiate_TruncElementKernel(uint16, uint16_t)
instantiate_TruncElementKernel(uint32, uint32_t)
instantiate_TruncElementKernel(uint64, uint64_t)
instantiate_TruncElementKernel(int8, int8_t)
instantiate_TruncElementKernel(int16, int16_t)
instantiate_TruncElementKernel(int32, int32_t)
instantiate_TruncElementKernel(int64, int64_t)
instantiate_TruncElementKernel(float16, half)
instantiate_TruncElementKernel(float32, float)

//--------------------------------------------------------------------------------------------
template <typename src_t, typename dst_t>
[[kernel]] void LogicalNotElementKernel(device u3d::metal::Indexer& indexer,
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
uint index [[thread_position_in_grid]]); \

instantiate_LogicalNotElementKernel(uint8, uint8_t)
instantiate_LogicalNotElementKernel(uint16, uint16_t)
instantiate_LogicalNotElementKernel(uint32, uint32_t)
instantiate_LogicalNotElementKernel(uint64, uint64_t)
instantiate_LogicalNotElementKernel(int8, int8_t)
instantiate_LogicalNotElementKernel(int16, int16_t)
instantiate_LogicalNotElementKernel(int32, int32_t)
instantiate_LogicalNotElementKernel(int64, int64_t)
instantiate_LogicalNotElementKernel(float16, half)
instantiate_LogicalNotElementKernel(float32, float)
