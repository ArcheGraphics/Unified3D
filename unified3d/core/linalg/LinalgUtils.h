//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <string>

#include <unified3d/core/Device.h>
#include <unified3d/core/Dtype.h>
#include <unified3d/core/MemoryManager.h>
#include <unified3d/core/linalg/LinalgHeadersCPU.h>
#include <unified3d/utility/Logging.h>

namespace u3d::core {

#define DISPATCH_LINALG_DTYPE_TO_TEMPLATE(DTYPE, ...)    \
    [&] {                                                \
        if (DTYPE == u3d::core::Float32) {               \
            using scalar_t = float;                      \
            return __VA_ARGS__();                        \
        } else if (DTYPE == u3d::core::Float64) {        \
            using scalar_t = double;                     \
            return __VA_ARGS__();                        \
        } else {                                         \
            utility::LogError("Unsupported data type."); \
        }                                                \
    }()

inline void UNIFIED3D_LAPACK_CHECK(UNIFIED3D_CPU_LINALG_INT info,
                                   const std::string& msg) {
    if (info < 0) {
        utility::LogError("{}: {}-th parameter is invalid.", msg, -info);
    } else if (info > 0) {
        utility::LogError("{}: singular condition detected.", msg);
    }
}

}  // namespace u3d::core
