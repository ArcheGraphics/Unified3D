//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <vtkRenderer.h>
#include <vtkCameraOrientationWidget.h>
#include <vtkRenderWindowInteractor.h>

namespace u3d {
class Windows;
class Renderer {
public:
    static Renderer &instance();

    Renderer();

    vtkNew<vtkRenderer> &handle();

    vtkNew<vtkRenderWindowInteractor> &interactor();

    void resetCamera();

    void addActor(vtkProp *actor);

    void bindWindow(Windows* win);

    void start();

private:
    vtkNew<vtkRenderer> _handle;
    vtkNew<vtkRenderWindowInteractor> _interactor;
};
}// namespace u3d