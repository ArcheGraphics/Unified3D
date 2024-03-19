//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/tools/Count.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/tree/LeafManager.h>
#include <openvdb/util/logging.h>

namespace u3d {
class PointDataOp {
public:
    explicit PointDataOp(vtkNew<vtkPolyData> &buffer) : mBuffer(buffer) {}

    template<typename GridType>
    void operator()(typename GridType::ConstPtr grid) {
        const typename GridType::TreeType &tree = grid->tree();

        // obtain cumulative point offsets and total points
        std::vector<openvdb::Index64> pointOffsets;
        const openvdb::Index64 total = openvdb::points::pointOffsets(pointOffsets, tree);

        vtkNew<vtkPoints> points;
        points->SetNumberOfPoints(total);
        VectorAttributeWrapper vecWrapper{points};
        openvdb::points::convertPointDataGridPosition(vecWrapper, *grid, pointOffsets, 0);

        // gen buffers and upload data to GPU
        mBuffer->SetPoints(points);

        const auto leafIter = tree.cbeginLeaf();
        if (!leafIter) return;

        const size_t colorIdx = leafIter->attributeSet().find("Cd");
        if (colorIdx == openvdb::points::AttributeSet::INVALID_POS) return;

        const auto &colorArray = leafIter->constAttributeArray(colorIdx);
        if (colorArray.template hasValueType<openvdb::Vec3f>()) {
            openvdb::points::convertPointDataGridAttribute(vecWrapper, tree, pointOffsets,
                                                           0, static_cast<unsigned>(colorIdx));

            // gen color buffer
            vtkNew<vtkUnsignedCharArray> colors;
            colors->SetNumberOfComponents(3);
            colors->SetName("Colors");
            for (int i = 0; i < total; ++i) {
                auto color = points->GetPoint(i);
                vtkColor3ub a(color[0] * 255, color[1] * 255, color[2] * 255);
                colors->InsertNextTypedTuple(a.GetData());
            }
            mBuffer->GetPointData()->SetScalars(colors);
        }
    }

private:
    struct VectorAttributeWrapper {
        using ValueType = openvdb::Vec3f;

        struct Handle {
            explicit Handle(VectorAttributeWrapper &attribute)
                : mValues(attribute.mValues) {}

            void set(openvdb::Index offset, openvdb::Index /*unused*/, const ValueType &value) {
                double point[3] = {value[0], value[1], value[2]};
                mValues->SetPoint(offset, point);
            }

        private:
            vtkNew<vtkPoints> &mValues;
        };// struct Handle

        explicit VectorAttributeWrapper(vtkNew<vtkPoints> &values)
            : mValues(values) {}

        void expand() {}
        void compact() {}
    private:
        vtkNew<vtkPoints> &mValues;
    };// struct VectorAttributeWrapper

    vtkNew<vtkPolyData> &mBuffer;
};

}// namespace u3d