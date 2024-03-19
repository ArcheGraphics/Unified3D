//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "tests/Tests.h"
#include "unified3d/utility/CPUInfo.h"

int main(int argc, char** argv) {
    using namespace u3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    utility::CPUInfo::GetInstance().Print();

    testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}
