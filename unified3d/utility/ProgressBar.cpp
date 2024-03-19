//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <unified3d/utility/ProgressBar.h>

#include <fmt/printf.h>

namespace u3d::utility {

ProgressBar::ProgressBar(size_t expected_count,
                         const std::string &progress_info,
                         bool active) {
    Reset(expected_count, progress_info, active);
}

void ProgressBar::Reset(size_t expected_count,
                        const std::string &progress_info,
                        bool active) {
    expected_count_ = expected_count;
    current_count_ = static_cast<size_t>(-1);  // Guaranteed to wraparound
    progress_info_ = progress_info;
    progress_pixel_ = 0;
    active_ = active;
    operator++();
}

ProgressBar &ProgressBar::operator++() {
    SetCurrentCount(current_count_ + 1);
    return *this;
}

void ProgressBar::SetCurrentCount(size_t n) {
    current_count_ = n;
    if (!active_) {
        return;
    }
    if (current_count_ >= expected_count_) {
        fmt::print("{}[{}] 100%\n", progress_info_,
                   std::string(resolution_, '='));
    } else {
        size_t new_progress_pixel =
                int(current_count_ * resolution_ / expected_count_);
        if (new_progress_pixel > progress_pixel_) {
            progress_pixel_ = new_progress_pixel;
            int percent = int(current_count_ * 100 / expected_count_);
            fmt::print("{}[{}>{}] {:d}%\r", progress_info_,
                       std::string(progress_pixel_, '='),
                       std::string(resolution_ - 1 - progress_pixel_, ' '),
                       percent);
            fflush(stdout);
        }
    }
}

size_t ProgressBar::GetCurrentCount() const { return current_count_; }

}  // namespace u3d::utility
