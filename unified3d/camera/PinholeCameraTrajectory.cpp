//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unified3d/camera/PinholeCameraTrajectory.h"

#include <json/json.h>

#include "unified3d/camera/PinholeCameraIntrinsic.h"
#include "unified3d/utility/Logging.h"

namespace u3d::camera {

PinholeCameraTrajectory::PinholeCameraTrajectory() = default;

PinholeCameraTrajectory::~PinholeCameraTrajectory() = default;

bool PinholeCameraTrajectory::ConvertToJsonValue(Json::Value &value) const {
    value["class_name"] = "PinholeCameraTrajectory";
    value["version_major"] = 1;
    value["version_minor"] = 0;
    Json::Value parameters_array;
    for (const auto &parameter : parameters_) {
        Json::Value parameter_value;
        parameter.ConvertToJsonValue(parameter_value);
        parameters_array.append(parameter_value);
    }
    value["parameters"] = parameters_array;

    return true;
}

bool PinholeCameraTrajectory::ConvertFromJsonValue(const Json::Value &value) {
    if (!value.isObject()) {
        utility::LogWarning(
                "PinholeCameraTrajectory read JSON failed: unsupported json "
                "format.");
        return false;
    }
    if (value.get("class_name", "").asString() != "PinholeCameraTrajectory" ||
        value.get("version_major", 1).asInt() != 1 ||
        value.get("version_minor", 0).asInt() != 0) {
        utility::LogWarning(
                "PinholeCameraTrajectory read JSON failed: unsupported json "
                "format.");
        return false;
    }

    const Json::Value &parameter_array = value["parameters"];

    if (parameter_array.empty()) {
        utility::LogWarning(
                "PinholeCameraTrajectory read JSON failed: empty "
                "trajectory.");
        return false;
    }
    parameters_.resize(parameter_array.size());
    for (size_t i = 0; i < parameter_array.size(); i++) {
        const Json::Value &status_object = parameter_array[int(i)];
        if (!parameters_[i].intrinsic_.ConvertFromJsonValue(
                    status_object["intrinsic"])) {
            return false;
        }
        if (!EigenMatrix4dFromJsonArray(parameters_[i].extrinsic_,
                                        status_object["extrinsic"])) {
            return false;
        }
    }
    return true;
}
}  // namespace u3d::camera
