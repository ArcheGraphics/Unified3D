//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "unified3d/io/PinholeCameraTrajectoryIO.h"
#include "unified3d/utility/FileSystem.h"
#include "unified3d/utility/Logging.h"

// The TUM format for camera trajectories as used in
// "A Benchmark for the Evaluation of RGB-D SLAM Systems" by
// J. Sturm and N. Engelhard and F. Endres and W. Burgard and D. Cremers
// (IROS 2012)
// See these pages for details:
// https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
// https://vision.in.tum.de/data/datasets/rgbd-dataset

namespace u3d::io {

bool ReadPinholeCameraTrajectoryFromTUM(
        const std::string &filename,
        camera::PinholeCameraTrajectory &trajectory) {
    camera::PinholeCameraIntrinsic intrinsic;
    if (!trajectory.parameters_.empty() &&
        trajectory.parameters_[0].intrinsic_.IsValid()) {
        intrinsic = trajectory.parameters_[0].intrinsic_;
    } else {
        intrinsic = camera::PinholeCameraIntrinsic(
                camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    }
    trajectory.parameters_.clear();
    FILE *f = utility::filesystem::FOpen(filename, "r");
    if (f == nullptr) {
        utility::LogWarning("Read TUM failed: unable to open file: {}",
                            filename);
        return false;
    }
    char line_buffer[DEFAULT_IO_BUFFER_SIZE];
    double ts, x, y, z, qx, qy, qz, qw;
    Eigen::Matrix4d transform;
    while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f)) {
        if (strlen(line_buffer) > 0 && line_buffer[0] != '#') {
            if (sscanf(line_buffer, "%lf %lf %lf %lf %lf %lf %lf %lf", &ts, &x,
                       &y, &z, &qx, &qy, &qz, &qw) != 8) {
                utility::LogWarning("Read TUM failed: unrecognized format.");
                fclose(f);
                return false;
            }

            transform.setIdentity();
            transform.topLeftCorner<3, 3>() =
                    Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix();
            transform.topRightCorner<3, 1>() = Eigen::Vector3d(x, y, z);
            auto param = camera::PinholeCameraParameters();
            param.intrinsic_ = intrinsic;
            param.extrinsic_ = transform.inverse();
            trajectory.parameters_.push_back(param);
        }
    }
    fclose(f);
    return true;
}

bool WritePinholeCameraTrajectoryToTUM(
        const std::string &filename,
        const camera::PinholeCameraTrajectory &trajectory) {
    FILE *f = utility::filesystem::FOpen(filename, "w");
    if (f == nullptr) {
        utility::LogWarning("Write TUM failed: unable to open file: {}",
                            filename);
        return false;
    }

    Eigen::Quaterniond q;
    fprintf(f, "# TUM trajectory, format: <t> <x> <y> <z> <qx> <qy> <qz> <qw>");
    for (size_t i = 0; i < trajectory.parameters_.size(); i++) {
        const Eigen::Matrix4d transform =
                trajectory.parameters_[i].extrinsic_.inverse();
        q = transform.topLeftCorner<3, 3>();
        fprintf(f, "%zu %lf %lf %lf %lf %lf %lf %lf\n", i, transform(0, 3),
                transform(1, 3), transform(2, 3), q.x(), q.y(), q.z(), q.w());
    }
    fclose(f);
    return true;
}

}  // namespace u3d::io
