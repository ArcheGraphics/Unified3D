//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <vtkPolyData.h>
#include <vtkQuad.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>

#include <openvdb/openvdb.h>
#include <openvdb/tools/VolumeToMesh.h>

namespace u3d {
class MeshOp {
public:
    explicit MeshOp(vtkNew<vtkPolyData> &buffer) : mBuffer(buffer) {}

    template<typename GridType>
    void operator()(typename GridType::ConstPtr grid) {
        using openvdb::Index64;

        openvdb::tools::VolumeToMesh mesher(grid->getGridClass() == openvdb::GRID_LEVEL_SET ? 0.0 : 0.01);
        mesher(*grid);

        // Copy points and generate point normals.
        vtkNew<vtkPoints> points;
        vtkNew<vtkDoubleArray> normals;
        normals->SetNumberOfComponents(3);
        normals->SetNumberOfTuples(mesher.pointListSize());

        openvdb::tree::ValueAccessor<const typename GridType::TreeType> acc(grid->tree());
        openvdb::math::GenericMap map(grid->transform());
        openvdb::Coord ijk;

        for (Index64 n = 0, i = 0, N = mesher.pointListSize(); n < N; ++n) {
            const openvdb::Vec3s &p = mesher.pointList()[n];
            points->InsertNextPoint(p[0], p[1], p[2]);
        }

        openvdb::tools::PolygonPoolList &polygonPoolList = mesher.polygonPoolList();
        vtkNew<vtkCellArray> quads;
        openvdb::Vec3d normal, e1, e2;

        for (Index64 n = 0, N = mesher.polygonPoolListSize(); n < N; ++n) {
            const openvdb::tools::PolygonPool &polygons = polygonPoolList[n];
            for (Index64 i = 0, I = polygons.numQuads(); i < I; ++i) {
                const openvdb::Vec4I &quad = polygons.quad(i);

                // Create a quad on the four points
                vtkNew<vtkQuad> vtk_quad;
                vtk_quad->GetPointIds()->SetId(0, quad[0]);
                vtk_quad->GetPointIds()->SetId(1, quad[1]);
                vtk_quad->GetPointIds()->SetId(2, quad[2]);
                vtk_quad->GetPointIds()->SetId(3, quad[3]);
                quads->InsertNextCell(vtk_quad);

                e1 = mesher.pointList()[quad[1]];
                e1 -= mesher.pointList()[quad[0]];
                e2 = mesher.pointList()[quad[2]];
                e2 -= mesher.pointList()[quad[1]];
                normal = e1.cross(e2);

                const double length = normal.length();
                if (length > 1.0e-7) normal *= (1.0 / length);

                for (int v = 0; v < 4; ++v) {
                    double pN1[3] = {static_cast<double>(normal[0]), static_cast<double>(normal[1]),
                                     static_cast<double>(normal[2])};
                    normals->SetTuple(quad[v], pN1);
                }
            }
        }

        // Add the points and quads to the dataset
        mBuffer->SetPoints(points);
        mBuffer->SetPolys(quads);
        mBuffer->GetPointData()->SetNormals(normals);
    }

private:
    vtkNew<vtkPolyData> &mBuffer;
};

}// namespace u3d