//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "unified3d/core/Device.h"

namespace u3d::core {

struct Stream {
    int index;
    Device device;
    explicit Stream(int index, Device device) : index(index), device(device) {}
};

inline bool operator==(const Stream& lhs, const Stream& rhs) {
    return lhs.index == rhs.index;
}

inline bool operator!=(const Stream& lhs, const Stream& rhs) {
    return !(lhs == rhs);
}

}  // namespace u3d::core
