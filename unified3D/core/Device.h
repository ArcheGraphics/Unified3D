//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <string>
#include <vector>

namespace unified3d {
namespace core {

/// Device context specifying device type and device id.
/// For CPU, there is only one device with id 0.
class Device {
public:
    /// Type for device.
    enum class DeviceType {
        CPU = 0,
        GPU = 1
    };

    /// Default constructor -> "CPU:0".
    Device() = default;

    /// Constructor with device specified.
    explicit Device(DeviceType device_type, int device_id);

    /// Constructor from device type string and device id.
    explicit Device(const std::string &device_type, int device_id);

    /// Constructor from string, e.g. "CUDA:0".
    explicit Device(const std::string &type_colon_id);

    bool operator==(const Device &other) const;

    bool operator!=(const Device &other) const;

    bool operator<(const Device &other) const;

    /// Returns true iff device type is CPU.
    inline bool IsCPU() const { return device_type_ == DeviceType::CPU; }

    /// Returns true iff device type is GPU.
    inline bool IsGPU() const { return device_type_ == DeviceType::GPU; }

    /// Returns string representation of device, e.g. "CPU:0", "GPU:0".
    std::string ToString() const;

    /// Returns type of the device, e.g. DeviceType::CPU, DeviceType::GPU.
    inline DeviceType GetType() const { return device_type_; }

    /// Returns the device index (within the same device type).
    inline int GetID() const { return device_id_; }

    /// Returns true if the device is available.
    bool IsAvailable() const;

    /// Returns a vector of available devices.
    static std::vector<Device> GetAvailableDevices();

    /// Returns a vector of available CPU device.
    static std::vector<Device> GetAvailableCPUDevices();

    /// Returns a vector of available GPU device.
    static std::vector<Device> GetAvailableGPUDevices();

    /// Print all available devices.
    static void PrintAvailableDevices();

protected:
    DeviceType device_type_ = DeviceType::CPU;
    int device_id_ = 0;
};

/// Abstract class to provide IsXXX() functionality to check device location.
/// Need to implement GetDevice().
class IsDevice {
public:
    IsDevice() = default;

    virtual ~IsDevice() = default;

    virtual core::Device GetDevice() const = 0;

    inline bool IsCPU() const {
        return GetDevice().GetType() == Device::DeviceType::CPU;
    }

    inline bool IsGPU() const {
        return GetDevice().GetType() == Device::DeviceType::GPU;
    }
};

}
}// namespace unified3d::core

namespace std {
template<>
struct hash<unified3d::core::Device> {
    std::size_t operator()(const unified3d::core::Device &device) const {
        return std::hash<std::string>{}(device.ToString());
    }
};
}// namespace std
