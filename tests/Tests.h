//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <sstream>
#include <string>
#include <vector>

#include "unified3d/Macro.h"
#include "unified3d/data/Dataset.h"
#include "tests/test_utility/Compare.h"
#include "tests/test_utility/Print.h"
#include "tests/test_utility/Rand.h"
#include "tests/test_utility/Raw.h"
#include "tests/test_utility/Sort.h"

namespace u3d::tests {

// Eigen Zero()
const Eigen::Vector2d Zero2d = Eigen::Vector2d::Zero();
const Eigen::Vector3d Zero3d = Eigen::Vector3d::Zero();
const Eigen::Matrix<double, 6, 1> Zero6d = Eigen::Matrix<double, 6, 1>::Zero();
const Eigen::Vector2i Zero2i = Eigen::Vector2i::Zero();

// Mechanism for reporting unit tests for which there is no implementation yet.
void NotImplemented();

}  // namespace u3d::tests
