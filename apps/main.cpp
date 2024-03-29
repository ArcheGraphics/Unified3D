//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <unified3d/visualization/windows.h>
#include <unified3d/visualization/renderer.h>
#include <unified3d/visualization/operations.h>

#include <vtkActor.h>
#include <vtkNew.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>

#include <openvdb/openvdb.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/PointScatter.h>
#include <openvdb/math/Operators.h>

// Populate the given grid with a narrow-band level set representation of a
// sphere. The width of the narrow band is determined by the grid's background
// value. (Example code only; use tools::createSphereSDF() in production.)
template <class GridType>
void makeSphere(GridType &grid, float radius, const openvdb::Vec3f &c) {
    using ValueT = typename GridType::ValueType;
    // Distance value for the constant region exterior to the narrow band
    const ValueT outside = grid.background();
    // Distance value for the constant region interior to the narrow band
    // (by convention, the signed distance is negative in the interior of
    // a level set)
    const ValueT inside = -outside;
    // Use the background value as the width in voxels of the narrow band.
    // (The narrow band is centered on the surface of the sphere, which
    // has distance 0.)
    int padding = int(openvdb::math::RoundUp(openvdb::math::Abs(outside)));
    // The bounding box of the narrow band is 2*dim voxels on a side.
    int dim = int(radius + padding);
    // Get a voxel accessor.
    typename GridType::Accessor accessor = grid.getAccessor();
    // Compute the signed distance from the surface of the sphere of each
    // voxel within the bounding box and insert the value into the grid
    // if it is smaller in magnitude than the background value.
    openvdb::Coord ijk;
    int &i = ijk[0], &j = ijk[1], &k = ijk[2];
    for (i = c[0] - dim; i < c[0] + dim; ++i) {
        const float x2 = openvdb::math::Pow2(i - c[0]);
        for (j = c[1] - dim; j < c[1] + dim; ++j) {
            const float x2y2 = openvdb::math::Pow2(j - c[1]) + x2;
            for (k = c[2] - dim; k < c[2] + dim; ++k) {
                // The distance from the sphere surface in voxels
                const float dist =
                        openvdb::math::Sqrt(x2y2 +
                                            openvdb::math::Pow2(k - c[2])) -
                        radius;
                // Convert the floating-point distance to the grid's value type.
                ValueT val = ValueT(dist);
                // Only insert distances that are smaller in magnitude than
                // the background value.
                if (val < inside || outside < val) continue;
                // Set the distance for voxel (i,j,k).
                accessor.setValue(ijk, val);
            }
        }
    }
    // Propagate the outside/inside sign information from the narrow band
    // throughout the grid.
    openvdb::tools::signedFloodFill(grid.tree());
}

int main(int, char *[]) {
    openvdb::initialize();
    // Create a shared pointer to a newly-allocated grid of a built-in type:
    // in this case, a FloatGrid, which stores one single-precision floating
    // point value per voxel.  Other built-in grid types include BoolGrid,
    // DoubleGrid, Int32Grid and Vec3SGrid (see openvdb.h for the complete
    // list). The grid comprises a sparse tree representation of voxel data,
    // user-supplied metadata and a voxel space to world space transform,
    // which defaults to the identity transform.
    openvdb::FloatGrid::Ptr grid =
            openvdb::FloatGrid::create(/*background value=*/2.0);
    // Populate the grid with a sparse, narrow-band level set representation
    // of a sphere with radius 50 voxels, located at (1.5, 2, 3) in index space.
    makeSphere(*grid, /*radius=*/50.0, /*center=*/openvdb::Vec3f(1.5, 2, 3));
    // Associate some metadata with the grid.
    grid->insertMeta("radius", openvdb::FloatMetadata(50.0));
    // Associate a scaling transform with the grid that sets the voxel size
    // to 0.5 units in world space.
    grid->setTransform(openvdb::math::Transform::createLinearTransform(
            /*voxel size=*/0.5));
    // Identify the grid as a level set.
    grid->setGridClass(openvdb::GRID_LEVEL_SET);

    {
        //        // Create a VDB file object.
        //        openvdb::io::File
        //        file("/Users/yangfeng/Downloads/armadillo.vdb");
        //        // Open the file.  This reads the file header, but not any
        //        grids. file.open();
        //        // Loop over all grids in the file and retrieve a shared
        //        pointer
        //        // to the one named "LevelSetSphere".  (This can also be done
        //        // more simply by calling file.readGrid("LevelSetSphere").)
        //        openvdb::GridBase::Ptr baseGrid;
        //        for (openvdb::io::File::NameIterator nameIter =
        //        file.beginName();
        //             nameIter != file.endName(); ++nameIter) {
        //            baseGrid = file.readGrid(nameIter.gridName());
        //        }
        //        file.close();
    }

    {
        vtkNew<vtkPolyData> polydata;
        u3d::MeshOp op(polydata);
        u3d::processTypedScalarGrid(grid, op);

        // Setup actor and mapper
        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputData(polydata);

        vtkNew<vtkActor> actor;
        actor->SetMapper(mapper);
        u3d::Renderer::instance().addActor(actor);
    }

    {
        vtkNew<vtkPolyData> polydata;
        u3d::TreeTopologyOp op(polydata);
        u3d::processTypedScalarGrid(grid, op);

        // Setup actor and mapper
        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputData(polydata);

        vtkNew<vtkActor> actor;
        actor->SetMapper(mapper);
        u3d::Renderer::instance().addActor(actor);
    }

    u3d::Windows windows("Viewer", 2048, 1152);
    windows.bindRenderer();
    u3d::Renderer::instance().start();
}