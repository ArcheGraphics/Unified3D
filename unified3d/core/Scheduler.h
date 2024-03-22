//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <atomic>
#include <future>
#include <queue>
#include <thread>
#include <unordered_map>

#include "unified3d/core/Device.h"
#include "unified3d/core/Stream.h"
#include "unified3d/metal/Device.h"

namespace u3d::core::scheduler {

struct StreamThread {
    std::mutex mtx;
    std::queue<std::function<void()>> q;
    std::condition_variable cond;
    bool stop;
    Stream stream;
    std::thread thread;

    StreamThread(Stream stream);

    ~StreamThread();

    void ThreadFn();

    template <typename F>
    void Enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lk(mtx);
            if (stop) {
                throw std::runtime_error(
                        "Cannot enqueue work after stream is stopped.");
            }
            q.emplace(std::forward<F>(f));
        }
        cond.notify_one();
    }
};

class Scheduler {
public:
    static Scheduler& GetInstance();

    Scheduler(const Scheduler&) = delete;
    Scheduler(Scheduler&&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;
    Scheduler& operator=(Scheduler&&) = delete;

    Stream CreateStream(const Device& d);

    template <typename F>
    void Enqueue(const Stream& stream, F&& f);

    Stream GetDefaultStream(const Device& d);

    void SetDefaultStream(const Stream& s);

    void NotifyNewTask(const Stream& stream);

    void NotifyTaskCompletion(const Stream& stream);

    [[nodiscard]] int ActiveTasks() const;

    void WaitForOne();

    ~Scheduler();

private:
    Scheduler();

    int n_active_tasks_;
    std::vector<StreamThread*> streams_;
    std::unordered_map<Device::DeviceType, Stream> default_streams_;
    std::condition_variable completion_cv;
    std::mutex mtx;
};

template <typename F>
void Scheduler::Enqueue(const Stream& stream, F&& f) {
    streams_[stream.index]->Enqueue(std::forward<F>(f));
}

}  // namespace u3d::core::scheduler
