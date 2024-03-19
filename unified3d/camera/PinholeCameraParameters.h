//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <memory>
#include <vector>

#include "unified3d/camera/PinholeCameraIntrinsic.h"

namespace u3d::camera {

/// \class PinholeCameraParameters
///
/// \brief Contains both intrinsic and extrinsic pinhole camera parameters.
class PinholeCameraParameters : public utility::IJsonConvertible {
public:
    /// \brief Default Constructor.
    PinholeCameraParameters();
    ~PinholeCameraParameters() override;

public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    /// PinholeCameraIntrinsic object.
    PinholeCameraIntrinsic intrinsic_;
    /// Camera extrinsic parameters.
    Eigen::Matrix4d_u extrinsic_;
};
}  // namespace u3d::camera
