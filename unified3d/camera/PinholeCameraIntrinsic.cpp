//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unified3d/camera/PinholeCameraIntrinsic.h"

#include <json/json.h>

#include <Eigen/Dense>
#include <utility>

#include "unified3d/utility/Logging.h"

namespace u3d::camera {

PinholeCameraIntrinsic::PinholeCameraIntrinsic()
    : width_(-1), height_(-1), intrinsic_matrix_(Eigen::Matrix3d::Identity()) {}

PinholeCameraIntrinsic::PinholeCameraIntrinsic(int width,
                                               int height,
                                               Eigen::Matrix3d intrinsic_matrix)
    : width_(width),
      height_(height),
      intrinsic_matrix_(std::move(intrinsic_matrix)) {}

PinholeCameraIntrinsic::PinholeCameraIntrinsic(
        int width, int height, double fx, double fy, double cx, double cy) {
    SetIntrinsics(width, height, fx, fy, cx, cy);
}

PinholeCameraIntrinsic::PinholeCameraIntrinsic(
        PinholeCameraIntrinsicParameters param) {
    if (param == PinholeCameraIntrinsicParameters::PrimeSenseDefault)
        SetIntrinsics(640, 480, 525.0, 525.0, 319.5, 239.5);
    else if (param ==
             PinholeCameraIntrinsicParameters::Kinect2DepthCameraDefault)
        SetIntrinsics(512, 424, 365.456, 365.456, 254.878, 205.395);
    else if (param ==
             PinholeCameraIntrinsicParameters::Kinect2ColorCameraDefault)
        SetIntrinsics(1920, 1080, 1059.9718, 1059.9718, 975.7193, 545.9533);
}

PinholeCameraIntrinsic::~PinholeCameraIntrinsic() = default;

bool PinholeCameraIntrinsic::ConvertToJsonValue(Json::Value &value) const {
    value["width"] = width_;
    value["height"] = height_;
    if (!EigenMatrix3dToJsonArray(intrinsic_matrix_,
                                  value["intrinsic_matrix"])) {
        return false;
    }
    return true;
}

bool PinholeCameraIntrinsic::ConvertFromJsonValue(const Json::Value &value) {
    if (!value.isObject()) {
        utility::LogWarning(
                "PinholeCameraParameters read JSON failed: unsupported json "
                "format.");
        return false;
    }
    width_ = value.get("width", -1).asInt();
    height_ = value.get("height", -1).asInt();
    if (!EigenMatrix3dFromJsonArray(intrinsic_matrix_,
                                    value["intrinsic_matrix"])) {
        utility::LogWarning(
                "PinholeCameraParameters read JSON failed: wrong format.");
        return false;
    }
    return true;
}
}  // namespace u3d::camera
