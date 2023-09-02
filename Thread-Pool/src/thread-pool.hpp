#pragma once

#include <condition_variable>
#include <functional>
#include <iterator>
#include <mutex>
#include <queue>
#include <thread>

void test_func();

class ThreadPool {
   public:
    void start();
    void queue_job(const std::function<void()>& job);
    void stop();
    bool busy();

   private:
    void thread_loop();

    bool should_terminate = false;            // Tells threads to stop looking for jobs
    std::mutex queue_mutex;                   // Prevents data races to the job queue
    std::condition_variable mutex_condition;  // Allows threads to wait on new jobs or termination
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> jobs;
    template <typename T>
    void map_jobs(const std::function<void(T args)>& job, std::vector<T> iter);
};
