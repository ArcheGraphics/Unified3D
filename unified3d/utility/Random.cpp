//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <unified3d/utility/Random.h>

#include <unified3d/utility/Logging.h>

namespace u3d::utility::random {

/// Global thread-safe singleton instance for random generation.
/// Generates compiler/OS/device independent random numbers.
class RandomContext {
public:
    RandomContext(RandomContext const&) = delete;
    void operator=(RandomContext const&) = delete;

    /// Returns the singleton instance.
    static RandomContext& GetInstance() {
        static RandomContext instance;
        return instance;
    }

    /// Seed the random number generator (globally).
    void Seed(const int seed) {
        seed_ = seed;
        engine_ = std::mt19937(seed_);
    }

    /// This is used by other downstream random generators.
    /// You must also lock the GetMutex() before calling the engine.
    std::mt19937* GetEngine() { return &engine_; }

    /// Get global singleton mutex to protect the engine call.
    std::mutex* GetMutex() { return &mutex_; }

private:
    RandomContext() {
        // Randomly seed the seed by default.
        std::random_device rd;
        Seed(rd());
    }
    int seed_{};
    std::mt19937 engine_;
    std::mutex mutex_;
};

void Seed(const int seed) { RandomContext::GetInstance().Seed(seed); }

std::mt19937* GetEngine() { return RandomContext::GetInstance().GetEngine(); }

std::mutex* GetMutex() { return RandomContext::GetInstance().GetMutex(); }

uint32_t RandUint32() {
    std::lock_guard<std::mutex> lock(*GetMutex());
    return (*GetEngine())();
}

}  // namespace u3d::utility::random
