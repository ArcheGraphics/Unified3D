//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "ops/mesh_op.h"
#include "ops/tree_topology_op.h"
#include "ops/point_data_op.h"

namespace u3d {
/// Helper class used internally by processTypedGrid()
template<typename GridType, typename OpType, bool IsConst /*=false*/>
struct GridProcessor {
    static inline void call(OpType &op, openvdb::GridBase::Ptr grid) {
        op.template operator()<GridType>(openvdb::gridPtrCast<GridType>(grid));
    }
};

/// Helper class used internally by processTypedGrid()
template<typename GridType, typename OpType>
struct GridProcessor<GridType, OpType, /*IsConst=*/true> {
    static inline void call(OpType &op, openvdb::GridBase::ConstPtr grid) {
        op.template operator()<GridType>(openvdb::gridConstPtrCast<GridType>(grid));
    }
};

/// Helper function used internally by processTypedGrid()
template<typename GridType, typename OpType, typename GridPtrType>
inline void
doProcessTypedGrid(GridPtrType grid, OpType &op) {
    GridProcessor<GridType, OpType,
                  std::is_const<typename GridPtrType::element_type>::value>::call(op, grid);
}

/// @brief Utility function that, given a generic grid pointer, calls
/// a functor on the fully-resolved grid, provided that the grid's
/// voxel values are scalars
template<typename GridPtrType, typename OpType>
bool processTypedScalarGrid(GridPtrType grid, OpType &op) {
    using namespace openvdb;
    if (grid->template isType<FloatGrid>())
        doProcessTypedGrid<FloatGrid>(grid, op);
    else if (grid->template isType<DoubleGrid>())
        doProcessTypedGrid<DoubleGrid>(grid, op);
    else if (grid->template isType<Int32Grid>())
        doProcessTypedGrid<Int32Grid>(grid, op);
    else if (grid->template isType<Int64Grid>())
        doProcessTypedGrid<Int64Grid>(grid, op);
    else
        return false;
    return true;
}
}// namespace u3d