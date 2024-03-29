//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <string>

namespace u3d::utility {

class ProgressBar {
public:
    ProgressBar(size_t expected_count,
                const std::string &progress_info,
                bool active = false);
    void Reset(size_t expected_count,
               const std::string &progress_info,
               bool active);
    virtual ProgressBar &operator++();
    void SetCurrentCount(size_t n);
    [[nodiscard]] size_t GetCurrentCount() const;

protected:
    const size_t resolution_ = 40;
    size_t expected_count_{};
    size_t current_count_{};
    std::string progress_info_;
    size_t progress_pixel_{};
    bool active_{};
};

}  // namespace u3d::utility
