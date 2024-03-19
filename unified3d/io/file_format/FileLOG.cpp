//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <Eigen/Dense>

#include "unified3d/io/PinholeCameraTrajectoryIO.h"
#include "unified3d/utility/FileSystem.h"
#include "unified3d/utility/Logging.h"

// The log file is the redwood-data format for camera trajectories
// See these pages for details:
// http://redwood-data.org/indoor/fileformat.html
// https://github.com/qianyizh/ElasticReconstruction/blob/f986e81a46201e28c0408a5f6303b4d3cdac7423/GraphOptimizer/helper.h

namespace u3d::io {

bool ReadPinholeCameraTrajectoryFromLOG(
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
        utility::LogWarning("Read LOG failed: unable to open file: {}",
                            filename.c_str());
        return false;
    }
    char line_buffer[DEFAULT_IO_BUFFER_SIZE];
    int i, j, k;
    Eigen::Matrix4d trans;
    while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f)) {
        if (strlen(line_buffer) > 0 && line_buffer[0] != '#') {
            if (sscanf(line_buffer, "%d %d %d", &i, &j, &k) != 3) {
                utility::LogWarning("Read LOG failed: unrecognized format.");
                fclose(f);
                return false;
            }
            if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == nullptr) {
                utility::LogWarning("Read LOG failed: unrecognized format.");
                fclose(f);
                return false;
            } else {
                sscanf(line_buffer, "%lf %lf %lf %lf", &trans(0, 0),
                       &trans(0, 1), &trans(0, 2), &trans(0, 3));
            }
            if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == nullptr) {
                utility::LogWarning("Read LOG failed: unrecognized format.");
                fclose(f);
                return false;
            } else {
                sscanf(line_buffer, "%lf %lf %lf %lf", &trans(1, 0),
                       &trans(1, 1), &trans(1, 2), &trans(1, 3));
            }
            if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == nullptr) {
                utility::LogWarning("Read LOG failed: unrecognized format.");
                fclose(f);
                return false;
            } else {
                sscanf(line_buffer, "%lf %lf %lf %lf", &trans(2, 0),
                       &trans(2, 1), &trans(2, 2), &trans(2, 3));
            }
            if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == nullptr) {
                utility::LogWarning("Read LOG failed: unrecognized format.");
                fclose(f);
                return false;
            } else {
                sscanf(line_buffer, "%lf %lf %lf %lf", &trans(3, 0),
                       &trans(3, 1), &trans(3, 2), &trans(3, 3));
            }
            auto param = camera::PinholeCameraParameters();
            param.intrinsic_ = intrinsic;
            param.extrinsic_ = trans.inverse();
            trajectory.parameters_.push_back(param);
        }
    }
    fclose(f);
    return true;
}

bool WritePinholeCameraTrajectoryToLOG(
        const std::string &filename,
        const camera::PinholeCameraTrajectory &trajectory) {
    FILE *f = utility::filesystem::FOpen(filename, "w");
    if (f == nullptr) {
        utility::LogWarning("Write LOG failed: unable to open file: {}",
                            filename);
        return false;
    }
    for (size_t i = 0; i < trajectory.parameters_.size(); i++) {
        Eigen::Matrix4d_u trans =
                trajectory.parameters_[i].extrinsic_.inverse();
        fprintf(f, "%d %d %d\n", (int)i, (int)i, (int)i + 1);
        fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(0, 0), trans(0, 1),
                trans(0, 2), trans(0, 3));
        fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(1, 0), trans(1, 1),
                trans(1, 2), trans(1, 3));
        fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(2, 0), trans(2, 1),
                trans(2, 2), trans(2, 3));
        fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(3, 0), trans(3, 1),
                trans(3, 2), trans(3, 3));
    }
    fclose(f);
    return true;
}

}  // namespace u3d::io
