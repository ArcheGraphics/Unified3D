//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <vtkNamedColors.h>
#include <vtkPolyData.h>
#include <vtkLine.h>
#include <vtkCellData.h>

#include <openvdb/openvdb.h>

namespace u3d {
class TreeTopologyOp {
public:
    explicit TreeTopologyOp(vtkNew<vtkPolyData> &buffer) : mBuffer(buffer) {}

    template<typename GridType>
    void operator()(typename GridType::ConstPtr grid) {
        using openvdb::Index64;

        Index64 nodeCount = grid->tree().leafCount() + grid->tree().nonLeafCount();
        const Index64 N = nodeCount * 8 * 3;

        vtkNew<vtkPoints> pts;
        vtkNew<vtkCellArray> lines;
        vtkNew<vtkUnsignedCharArray> colors;
        colors->SetNumberOfComponents(3);
        vtkNew<vtkNamedColors> namedColors;

        openvdb::Vec3d ptn;
        openvdb::CoordBBox bbox;
        Index64 pOffset = 0, iOffset = 0, cOffset = 0, idx = 0;

        for (typename GridType::TreeType::NodeCIter iter = grid->tree().cbeginNode(); iter; ++iter) {
            // node vertex color
            const int level = iter.getLevel();
            vtkStdString color = sNodeColors[(level == 0) ? 3 : (level == 1) ? 2 :
                                                                               1];

            iter.getBoundingBox(bbox);

            // Nodes are rendered as cell-centered
            const openvdb::Vec3d min(bbox.min().x() - 0.5, bbox.min().y() - 0.5, bbox.min().z() - 0.5);
            const openvdb::Vec3d max(bbox.max().x() + 0.5, bbox.max().y() + 0.5, bbox.max().z() + 0.5);

            // corner 1
            ptn = grid->indexToWorld(min);
            std::array<double, 3> p = {ptn[0], ptn[1], ptn[2]};
            pts->InsertNextPoint(p.data());

            // corner 2
            ptn = openvdb::Vec3d(min.x(), min.y(), max.z());
            ptn = grid->indexToWorld(ptn);
            p = {ptn[0], ptn[1], ptn[2]};
            pts->InsertNextPoint(p.data());

            // corner 3
            ptn = openvdb::Vec3d(max.x(), min.y(), max.z());
            ptn = grid->indexToWorld(ptn);
            p = {ptn[0], ptn[1], ptn[2]};
            pts->InsertNextPoint(p.data());

            // corner 4
            ptn = openvdb::Vec3d(max.x(), min.y(), min.z());
            ptn = grid->indexToWorld(ptn);
            p = {ptn[0], ptn[1], ptn[2]};
            pts->InsertNextPoint(p.data());

            // corner 5
            ptn = openvdb::Vec3d(min.x(), max.y(), min.z());
            ptn = grid->indexToWorld(ptn);
            p = {ptn[0], ptn[1], ptn[2]};
            pts->InsertNextPoint(p.data());

            // corner 6
            ptn = openvdb::Vec3d(min.x(), max.y(), max.z());
            ptn = grid->indexToWorld(ptn);
            p = {ptn[0], ptn[1], ptn[2]};
            pts->InsertNextPoint(p.data());

            // corner 7
            ptn = grid->indexToWorld(max);
            p = {ptn[0], ptn[1], ptn[2]};
            pts->InsertNextPoint(p.data());

            // corner 8
            ptn = openvdb::Vec3d(max.x(), max.y(), min.z());
            ptn = grid->indexToWorld(ptn);
            p = {ptn[0], ptn[1], ptn[2]};
            pts->InsertNextPoint(p.data());

            // edge 1
            {
                vtkNew<vtkLine> line0;
                line0->GetPointIds()->SetId(0, idx);
                line0->GetPointIds()->SetId(1, idx + 1);
                lines->InsertNextCell(line0);
                colors->InsertNextTypedTuple(namedColors->GetColor3ub(color).GetData());
            }

            // edge 2
            {
                vtkNew<vtkLine> line0;
                line0->GetPointIds()->SetId(0, idx + 1);
                line0->GetPointIds()->SetId(1, idx + 2);
                lines->InsertNextCell(line0);
                colors->InsertNextTypedTuple(namedColors->GetColor3ub(color).GetData());
            }

            // edge 3
            {
                vtkNew<vtkLine> line0;
                line0->GetPointIds()->SetId(0, idx + 2);
                line0->GetPointIds()->SetId(1, idx + 3);
                lines->InsertNextCell(line0);
                colors->InsertNextTypedTuple(namedColors->GetColor3ub(color).GetData());
            }

            // edge 4
            {
                vtkNew<vtkLine> line0;
                line0->GetPointIds()->SetId(0, idx + 3);
                line0->GetPointIds()->SetId(1, idx);
                lines->InsertNextCell(line0);
                colors->InsertNextTypedTuple(namedColors->GetColor3ub(color).GetData());
            }

            // edge 5
            {
                vtkNew<vtkLine> line0;
                line0->GetPointIds()->SetId(0, idx + 4);
                line0->GetPointIds()->SetId(1, idx + 5);
                lines->InsertNextCell(line0);
                colors->InsertNextTypedTuple(namedColors->GetColor3ub(color).GetData());
            }

            // edge 6
            {
                vtkNew<vtkLine> line0;
                line0->GetPointIds()->SetId(0, idx + 5);
                line0->GetPointIds()->SetId(1, idx + 6);
                lines->InsertNextCell(line0);
                colors->InsertNextTypedTuple(namedColors->GetColor3ub(color).GetData());
            }

            // edge 7
            {
                vtkNew<vtkLine> line0;
                line0->GetPointIds()->SetId(0, idx + 6);
                line0->GetPointIds()->SetId(1, idx + 7);
                lines->InsertNextCell(line0);
                colors->InsertNextTypedTuple(namedColors->GetColor3ub(color).GetData());
            }

            // edge 8
            {
                vtkNew<vtkLine> line0;
                line0->GetPointIds()->SetId(0, idx + 7);
                line0->GetPointIds()->SetId(1, idx + 4);
                lines->InsertNextCell(line0);
                colors->InsertNextTypedTuple(namedColors->GetColor3ub(color).GetData());
            }

            // edge 9
            {
                vtkNew<vtkLine> line0;
                line0->GetPointIds()->SetId(0, idx);
                line0->GetPointIds()->SetId(1, idx + 4);
                lines->InsertNextCell(line0);
                colors->InsertNextTypedTuple(namedColors->GetColor3ub(color).GetData());
            }

            // edge 10
            {
                vtkNew<vtkLine> line0;
                line0->GetPointIds()->SetId(0, idx + 1);
                line0->GetPointIds()->SetId(1, idx + 5);
                lines->InsertNextCell(line0);
                colors->InsertNextTypedTuple(namedColors->GetColor3ub(color).GetData());
            }

            // edge 11
            {
                vtkNew<vtkLine> line0;
                line0->GetPointIds()->SetId(0, idx + 2);
                line0->GetPointIds()->SetId(1, idx + 6);
                lines->InsertNextCell(line0);
                colors->InsertNextTypedTuple(namedColors->GetColor3ub(color).GetData());
            }

            // edge 12
            {
                vtkNew<vtkLine> line0;
                line0->GetPointIds()->SetId(0, idx + 3);
                line0->GetPointIds()->SetId(1, idx + 7);
                lines->InsertNextCell(line0);
                colors->InsertNextTypedTuple(namedColors->GetColor3ub(color).GetData());
            }

            idx += 8;
        }

        mBuffer->SetPoints(pts);
        mBuffer->SetLines(lines);
        mBuffer->GetCellData()->SetScalars(colors);
    }

private:
    vtkNew<vtkPolyData> &mBuffer;
    static vtkStdString sNodeColors[];
};

vtkStdString TreeTopologyOp::sNodeColors[] = {
    "DarkSlateGray",// root （11.465, 11.465, 11.465）
    "DarkGreen",    // first internal node level (11, 84, 10)
    "DarkOrange",   // intermediate internal node levels (221, 100, 4)
    "DarkTurquoise" // leaf nodes (1.5, 71, 159)
};
}// namespace u3d