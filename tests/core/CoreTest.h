//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <vector>

#include "unified3d/core/Device.h"
#include "unified3d/core/Dtype.h"
#include "unified3d/core/SizeVector.h"
#include "tests/Tests.h"

namespace u3d::tests {

class PermuteDtypesWithBool : public testing::TestWithParam<core::Dtype> {
public:
    static std::vector<core::Dtype> TestCases();
};

/// Permute one device for each device type, in {CPU, CUDA}.
class PermuteDevices : public testing::TestWithParam<core::Device> {
public:
    static std::vector<core::Device> TestCases();
};

/// Permute device pairs, in {CPU, CUDA}.
class PermuteDevicePairs
    : public testing::TestWithParam<std::pair<core::Device, core::Device>> {
public:
    static std::vector<std::pair<core::Device, core::Device>> TestCases();
};

class PermuteSizesDefaultStrides
    : public testing::TestWithParam<
              std::pair<core::SizeVector, core::SizeVector>> {
public:
    static std::vector<std::pair<core::SizeVector, core::SizeVector>>
    TestCases();
};

class TensorSizes : public testing::TestWithParam<int64_t> {
public:
    static std::vector<int64_t> TestCases();
};

}  // namespace u3d::tests
