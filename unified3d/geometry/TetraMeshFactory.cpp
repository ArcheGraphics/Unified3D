//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unified3d/geometry/PointCloud.h"
#include "unified3d/geometry/Qhull.h"
#include "unified3d/geometry/TetraMesh.h"
#include "unified3d/utility/Logging.h"

namespace u3d::geometry {

std::tuple<std::shared_ptr<TetraMesh>, std::vector<size_t>>
TetraMesh::CreateFromPointCloud(const PointCloud& point_cloud) {
    if (point_cloud.points_.size() < 4) {
        utility::LogError("Not enough points to create a tetrahedral mesh.");
    }
    return Qhull::ComputeDelaunayTetrahedralization(point_cloud.points_);
}

}  // namespace u3d::geometry
