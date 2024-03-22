//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "unified3d/core/Scheduler.h"

namespace u3d::core::scheduler {
StreamThread::StreamThread(Stream stream)
    : stop(false), stream(stream), thread(&StreamThread::ThreadFn, this) {}

StreamThread::~StreamThread() {
    {
        std::unique_lock<std::mutex> lk(mtx);
        stop = true;
    }
    cond.notify_one();
    thread.join();
}

void StreamThread::ThreadFn() {
    bool initialized = false;
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lk(mtx);
            cond.wait(lk,
                      [this] { return !this->q.empty() || this->stop; });
            if (q.empty() && stop) {
                return;
            }
            task = std::move(q.front());
            q.pop();
        }

        // thread_fn may be called from a static initializer and we cannot
        // call metal-cpp until all static initializers have completed.
        // waiting for a task to arrive means that user code is running so
        // metal-cpp can safely be called.
        if (!initialized) {
            initialized = true;
            u3d::core::metal::Device::GetInstance().new_queue(stream.index);
        }

        task();
    }
}

Scheduler::Scheduler() : n_active_tasks_(0) {
    default_streams_.insert({Device::GPU, CreateStream(Device::GPU)});
    default_streams_.insert({Device::CPU, CreateStream(Device::CPU)});
}

Stream Scheduler::CreateStream(const Device& d) {
    auto stream = Stream(streams_.size(), d);
    streams_.push_back(new StreamThread{stream});
    return stream;
}


Scheduler& Scheduler::GetInstance() {
    static Scheduler scheduler;
    return scheduler;
}

Stream Scheduler::GetDefaultStream(const Device& d) {
    return default_streams_.at(d.GetType());
}

void Scheduler::SetDefaultStream(const Stream& s) {
    default_streams_.at(s.device.GetType()) = s;
}

void Scheduler::NotifyNewTask(const Stream& stream) {
    {
        std::unique_lock<std::mutex> lk(mtx);
        n_active_tasks_++;
    }
    completion_cv.notify_all();
}

void Scheduler::NotifyTaskCompletion(const Stream& stream) {
    {
        std::unique_lock<std::mutex> lk(mtx);
        n_active_tasks_--;
    }
    completion_cv.notify_all();
}

int Scheduler::ActiveTasks() const { return n_active_tasks_; }

void Scheduler::WaitForOne() {
    std::unique_lock<std::mutex> lk(mtx);
    int n_tasks_old = ActiveTasks();
    if (n_tasks_old > 1) {
        completion_cv.wait(lk, [this, n_tasks_old] {
            return this->ActiveTasks() != n_tasks_old;
        });
    }
}

Scheduler::~Scheduler() {
    for (auto s : streams_) {
        delete s;
    }
}

}  // namespace u3d::core::scheduler
