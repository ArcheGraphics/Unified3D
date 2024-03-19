//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <unified3d/core/Device.h>

#include <string>
#include <vector>

#include <unified3d/utility/Helper.h>
#include <unified3d/utility/Logging.h>

namespace u3d::core {

static Device::DeviceType StringToDeviceType(const std::string &type_colon_id) {
    const std::vector<std::string> tokens =
            utility::SplitString(type_colon_id, ":", true);
    if (tokens.size() == 2) {
        std::string device_type_lower = utility::ToLower(tokens[0]);
        if (device_type_lower == "cpu") {
            return Device::DeviceType::CPU;
        } else if (device_type_lower == "gpu") {
            return Device::DeviceType::GPU;
        } else {
            utility::LogError(
                    "Invalid device string {}. Valid device strings are like "
                    "\"CPU:0\" or \"GPU:1\"",
                    type_colon_id);
        }
    } else {
        utility::LogError(
                "Invalid device string {}. Valid device strings are like "
                "\"CPU:0\" or \"GPU:1\"",
                type_colon_id);
    }
}

static int StringToDeviceId(const std::string &type_colon_id) {
    const std::vector<std::string> tokens =
            utility::SplitString(type_colon_id, ":", true);
    if (tokens.size() == 2) {
        return std::stoi(tokens[1]);
    } else {
        utility::LogError(
                "Invalid device string {}. Valid device strings are like "
                "\"CPU:0\" or \"GPU:1\"",
                type_colon_id);
    }
}

Device::Device(DeviceType device_type, int device_id)
    : device_type_(device_type), device_id_(device_id) {
    // Sanity checks.
    if (device_type_ == DeviceType::CPU && device_id_ != 0) {
        utility::LogError("CPU has device_id {}, but it must be 0.",
                          device_id_);
    }
}

Device::Device(const std::string &device_type, int device_id)
    : Device(device_type + ":" + std::to_string(device_id)) {}

Device::Device(const std::string &type_colon_id)
    : Device(StringToDeviceType(type_colon_id),
             StringToDeviceId(type_colon_id)) {}

bool Device::operator==(const Device &other) const {
    return this->device_type_ == other.device_type_ &&
           this->device_id_ == other.device_id_;
}

bool Device::operator!=(const Device &other) const {
    return !operator==(other);
}

bool Device::operator<(const Device &other) const {
    return ToString() < other.ToString();
}

std::string Device::ToString() const {
    std::string str;
    switch (device_type_) {
        case DeviceType::CPU:
            str += "CPU";
            break;
        case DeviceType::GPU:
            str += "GPU";
            break;
        default:
            utility::LogError("Unsupported device type");
    }
    str += ":" + std::to_string(device_id_);
    return str;
}

bool Device::IsAvailable() const {
    for (const Device &device : GetAvailableDevices()) {
        if (device == *this) {
            return true;
        }
    }
    return false;
}

std::vector<Device> Device::GetAvailableDevices() {
    const std::vector<Device> cpu_devices = GetAvailableCPUDevices();
    const std::vector<Device> gpu_devices = GetAvailableGPUDevices();
    std::vector<Device> devices;
    devices.insert(devices.end(), cpu_devices.begin(), cpu_devices.end());
    devices.insert(devices.end(), gpu_devices.begin(), gpu_devices.end());
    return devices;
}

std::vector<Device> Device::GetAvailableCPUDevices() {
    return {Device(DeviceType::CPU, 0)};
}

std::vector<Device> Device::GetAvailableGPUDevices() {
    std::vector<Device> devices;
    for (int i = 0; i < 1; i++) {
        devices.emplace_back(DeviceType::GPU, i);
    }
    return devices;
}

void Device::PrintAvailableDevices() {
    for (const auto &device : GetAvailableDevices()) {
        utility::LogInfo("Device(\"{}\")", device.ToString());
    }
}

}  // namespace u3d::core
