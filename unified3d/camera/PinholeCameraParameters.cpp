//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unified3d/camera/PinholeCameraParameters.h"

#include <json/json.h>

#include "unified3d/utility/Logging.h"

namespace u3d::camera {
PinholeCameraParameters::PinholeCameraParameters() = default;

PinholeCameraParameters::~PinholeCameraParameters() = default;

bool PinholeCameraParameters::ConvertToJsonValue(Json::Value &value) const {
    Json::Value trajectory_array;
    value["class_name"] = "PinholeCameraParameters";
    value["version_major"] = 1;
    value["version_minor"] = 0;
    if (!EigenMatrix4dToJsonArray(extrinsic_, value["extrinsic"])) {
        return false;
    }
    if (!intrinsic_.ConvertToJsonValue(value["intrinsic"])) {
        return false;
    }
    return true;
}

bool PinholeCameraParameters::ConvertFromJsonValue(const Json::Value &value) {
    if (!value.isObject()) {
        utility::LogWarning(
                "PinholeCameraParameters read JSON failed: unsupported json "
                "format.");
        return false;
    }
    if (value.get("class_name", "").asString() != "PinholeCameraParameters" ||
        value.get("version_major", 1).asInt() != 1 ||
        value.get("version_minor", 0).asInt() != 0) {
        utility::LogWarning(
                "PinholeCameraParameters read JSON failed: unsupported json "
                "format.");
        return false;
    }
    if (!intrinsic_.ConvertFromJsonValue(value["intrinsic"])) {
        return false;
    }
    if (!EigenMatrix4dFromJsonArray(extrinsic_, value["extrinsic"])) {
        return false;
    }
    return true;
}
}  // namespace u3d::camera
