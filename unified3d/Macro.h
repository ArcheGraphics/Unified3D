//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cassert>

// https://gcc.gnu.org/wiki/Visibility updated to use C++11 attribute syntax
#define UNIFIED3D_DLL_IMPORT [[gnu::visibility("default")]]
#define UNIFIED3D_DLL_EXPORT [[gnu::visibility("default")]]
#define UNIFIED3D_DLL_LOCAL [[gnu::visibility("hidden")]]

#ifdef UNIFIED3D_STATIC
#define UNIFIED3D_API
#define UNIFIED3D_LOCAL
#else
#define UNIFIED3D_LOCAL UNIFIED3D_DLL_LOCAL
#if defined(UNIFIED3D_ENABLE_DLL_EXPORTS)
#define UNIFIED3D_API UNIFIED3D_DLL_EXPORT
#else
#define UNIFIED3D_API UNIFIED3D_DLL_IMPORT
#endif
#endif

// Compiler-specific function macro.
// Ref: https://stackoverflow.com/a/4384825
#define UNIFIED3D_FUNCTION __PRETTY_FUNCTION__

// Assertion for CUDA device code.
// Usage:
//     UNIFIED3D_ASSERT(condition);
//     UNIFIED3D_ASSERT(condition && "Error message");
// For host-only code, consider using utility::LogError();
#define UNIFIED3D_ASSERT(...) assert((__VA_ARGS__))
