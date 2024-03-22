//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "tests/core/CoreTest.h"

#include <algorithm>
#include <vector>

#include "unified3d/core/Device.h"
#include "unified3d/core/Dtype.h"
#include "unified3d/core/SizeVector.h"

namespace u3d::tests {

std::vector<core::Dtype> PermuteDtypesWithBool::TestCases() {
    return {
            core::Bool,  core::UInt8,   core::Int8,    core::UInt16,
            core::Int16, core::UInt32,  core::Int32,   core::UInt64,
            core::Int64, core::Float32,
    };
}

std::vector<core::Device> PermuteDevices::TestCases() {
    std::vector<core::Device> cpu_devices =
            core::Device::GetAvailableCPUDevices();
    std::vector<core::Device> gpu_devices =
            core::Device::GetAvailableGPUDevices();

    std::vector<core::Device> devices;
    if (!cpu_devices.empty()) {
        devices.push_back(cpu_devices[0]);
    }

    // Test 0, 1, or 2 GPU devices.
    // Testing 2 GPU devices is necessary for testing device switching.
//    if (gpu_devices.size() == 1) {
//        devices.push_back(gpu_devices[0]);
//    } else if (gpu_devices.size() == 2) {
//        devices.push_back(gpu_devices[0]);
//        devices.push_back(gpu_devices[1]);
//    }

    return devices;
}

std::vector<std::pair<core::Device, core::Device>>
PermuteDevicePairs::TestCases() {
    std::vector<core::Device> cpu_devices =
            core::Device::GetAvailableCPUDevices();
    std::vector<core::Device> gpu_devices =
            core::Device::GetAvailableGPUDevices();

    cpu_devices.resize(std::min(static_cast<size_t>(2), cpu_devices.size()));
    gpu_devices.resize(std::min(static_cast<size_t>(2), gpu_devices.size()));

    std::vector<core::Device> devices;
    devices.insert(devices.end(), cpu_devices.begin(), cpu_devices.end());
//    devices.insert(devices.end(), gpu_devices.begin(), gpu_devices.end());

    // Self-pairs and cross pairs (bidirectional).
    std::vector<std::pair<core::Device, core::Device>> device_pairs;
    for (auto& device : devices) {
        device_pairs.emplace_back(device, device);
    }
    for (size_t i = 0; i < devices.size(); i++) {
        for (size_t j = 0; j < devices.size(); j++) {
            if (i != j) {
                device_pairs.emplace_back(devices[i], devices[j]);
            }
        }
    }

    return device_pairs;
}

std::vector<std::pair<core::SizeVector, core::SizeVector>>
PermuteSizesDefaultStrides::TestCases() {
    return {
            {{}, {}},
            {{0}, {1}},
            {{0, 0}, {1, 1}},
            {{0, 1}, {1, 1}},
            {{1, 0}, {1, 1}},
            {{1}, {1}},
            {{1, 2}, {2, 1}},
            {{1, 2, 3}, {6, 3, 1}},
            {{4, 3, 2}, {6, 2, 1}},
            {{2, 0, 3}, {3, 3, 1}},
    };
}

std::vector<int64_t> TensorSizes::TestCases() {
    std::vector<int64_t> tensor_sizes{
            0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    // clang-format off
    std::vector<int64_t> large_sizes{
        (1 << 6 ) - 1, (1 << 6 ), (1 << 6 ) + 1,
        (1 << 10) - 6, (1 << 10), (1 << 10) + 6,
        (1 << 15) - 7, (1 << 15), (1 << 15) + 7,
        (1 << 20) - 1, (1 << 20), (1 << 20) + 1,
        (1 << 25) - 2, (1 << 25), (1 << 25) + 2, // ~128MB for float32
    };
    // clang-format on
    tensor_sizes.insert(tensor_sizes.end(), large_sizes.begin(),
                        large_sizes.end());
    return tensor_sizes;
}

}  // namespace u3d::tests
