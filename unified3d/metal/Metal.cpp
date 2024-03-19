//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "Metal.h"
#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>

namespace u3d::core {
std::shared_ptr<void> NewScopedMemoryPool() {
    auto dtor = [](void* ptr) {
        static_cast<NS::AutoreleasePool*>(ptr)->release();
    };
    return {NS::AutoreleasePool::alloc()->init(), dtor};
}
}  // namespace u3d::core