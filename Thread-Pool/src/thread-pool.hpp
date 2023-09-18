#pragma once

#include <condition_variable>
#include <functional>
#include <iterator>
#include <mutex>
#include <queue>
#include <ranges>
#include <thread>

class ThreadPool {
   public:
    void start();
    void queue_job(const std::function<void()>& job);
    void stop();
    bool busy();
    void wait();
    void map_jobs(const std::function<void(uint32_t idx)>& job, std::ranges::iota_view<uint32_t, uint32_t> iter);

   private:
    void thread_loop();

    bool should_terminate = false;            // Tells threads to stop looking for jobs
    std::mutex queue_mutex;                   // Prevents data races to the job queue
    std::condition_variable mutex_condition;  // Allows threads to wait on new jobs or termination
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> jobs;
};
