//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <unified3d/utility/CPUInfo.h>

#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <sys/sysctl.h>
#include <sys/types.h>

#include <unified3d/utility/Logging.h>

namespace u3d::utility {

struct CPUInfo::Impl {
    uint32_t num_cores_;
    uint32_t num_threads_;
};

/// Returns the number of physical CPU cores.
static uint32_t PhysicalConcurrency() {
    try {
        // Ref: boost::thread::physical_concurrency().
        int count;
        size_t size = sizeof(count);
        return sysctlbyname("hw.physicalcpu", &count, &size, nullptr, 0)
                       ? 0
                       : count;

    } catch (...) {
        return std::thread::hardware_concurrency();
    }
}  // namespace utility

CPUInfo::CPUInfo() : impl_(new CPUInfo::Impl()) {
    impl_->num_cores_ = PhysicalConcurrency();
    impl_->num_threads_ = std::thread::hardware_concurrency();
}

CPUInfo& CPUInfo::GetInstance() {
    static CPUInfo instance;
    return instance;
}

uint32_t CPUInfo::NumCores() const { return impl_->num_cores_; }

uint32_t CPUInfo::NumThreads() const { return impl_->num_threads_; }

void CPUInfo::Print() const {
    utility::LogInfo("CPUInfo: {} cores, {} threads.", NumCores(),
                     NumThreads());
}

}  // namespace u3d::utility
