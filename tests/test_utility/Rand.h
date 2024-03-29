//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <Eigen/Core>
#include <vector>

#include "unified3d/utility/Eigen.h"

namespace u3d::tests {

// Initialize an Eigen::Vector3d.
// Output range: [vmin:vmax].
void Rand(Eigen::Vector3d& v,
          const Eigen::Vector3d& vmin,
          const Eigen::Vector3d& vmax,
          const int& seed);

// Initialize an Eigen::Vector3d.
// Output range: [vmin:vmax].
void Rand(Eigen::Vector3d& v,
          const double& vmin,
          const double& vmax,
          const int& seed);

// Initialize an Eigen::Vector2i vector.
// Output range: [vmin:vmax].
void Rand(std::vector<Eigen::Vector2i>& v,
          const Eigen::Vector2i& vmin,
          const Eigen::Vector2i& vmax,
          const int& seed);

// Initialize an Eigen::Vector2d vector.
// Output range: [vmin:vmax].
void Rand(std::vector<Eigen::Vector2d, u3d::utility::Vector2d_allocator>& v,
          const Eigen::Vector2d& vmin,
          const Eigen::Vector2d& vmax,
          const int& seed);

// Initialize an Eigen::Vector3i vector.
// Output range: [vmin:vmax].
void Rand(std::vector<Eigen::Vector3i>& v,
          const Eigen::Vector3i& vmin,
          const Eigen::Vector3i& vmax,
          const int& seed);

// Initialize an Eigen::Vector3d vector.
// Output range: [vmin:vmax].
void Rand(std::vector<Eigen::Vector3d>& v,
          const Eigen::Vector3d& vmin,
          const Eigen::Vector3d& vmax,
          const int& seed);

// Initialize an Eigen::Vector4i vector.
// Output range: [vmin:vmax].
void Rand(std::vector<Eigen::Vector4i, u3d::utility::Vector4i_allocator>& v,
          const int& vmin,
          const int& vmax,
          const int& seed);

// Initialize an Eigen::Vector4i vector.
// Output range: [vmin:vmax].
void Rand(std::vector<Eigen::Vector4i, u3d::utility::Vector4i_allocator>& v,
          const Eigen::Vector4i& vmin,
          const Eigen::Vector4i& vmax,
          const int& seed);

// Initialize a uint8_t vector.
// Output range: [vmin:vmax].
void Rand(std::vector<uint8_t>& v,
          const uint8_t& vmin,
          const uint8_t& vmax,
          const int& seed);

// Initialize an array of int.
// Output range: [vmin:vmax].
void Rand(int* v,
          const size_t& size,
          const int& vmin,
          const int& vmax,
          const int& seed);

// Initialize an int vector.
// Output range: [vmin:vmax].
void Rand(std::vector<int>& v,
          const int& vmin,
          const int& vmax,
          const int& seed);

// Initialize a size_t vector.
// Output range: [vmin:vmax].
void Rand(std::vector<size_t>& v,
          const size_t& vmin,
          const size_t& vmax,
          const int& seed);

// Initialize an array of float.
// Output range: [vmin:vmax].
void Rand(float* v,
          const size_t& size,
          const float& vmin,
          const float& vmax,
          const int& seed);

// Initialize a float vector.
// Output range: [vmin:vmax].
void Rand(std::vector<float>& v,
          const float& vmin,
          const float& vmax,
          const int& seed);

// Initialize an array of double.
// Output range: [vmin:vmax].
void Rand(double* v,
          const size_t& size,
          const double& vmin,
          const double& vmax,
          const int& seed);

// Initialize a double vector.
// Output range: [vmin:vmax].
void Rand(std::vector<double>& v,
          const double& vmin,
          const double& vmax,
          const int& seed);

}  // namespace u3d::tests
