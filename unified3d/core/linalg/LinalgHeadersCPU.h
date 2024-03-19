//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

// This file contains headers for BLAS/LAPACK implementations. Currently we
// support int64_t interface of OpenBLAS or Intel MKL.
//
// For developers, please make sure that this file is not ultimately included in
// Open3D.h.

#pragma once

#define UNIFIED3D_CPU_LINALG_INT int32_t
#define lapack_int int32_t
#include <cblas.h>
#include <lapacke.h>