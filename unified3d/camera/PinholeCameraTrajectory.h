//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <memory>
#include <vector>

#include "unified3d/camera/PinholeCameraParameters.h"

namespace u3d::camera {

/// \class PinholeCameraTrajectory
///
/// Contains a list of PinholeCameraParameters, useful to storing trajectories.
class PinholeCameraTrajectory : public utility::IJsonConvertible {
public:
    /// \brief Default Constructor.
    PinholeCameraTrajectory();
    ~PinholeCameraTrajectory() override;

public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    /// List of PinholeCameraParameters objects.
    std::vector<PinholeCameraParameters> parameters_;
};

}  // namespace u3d::camera
