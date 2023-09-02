#include "thread-pool.hpp"

#include <stdio.h>

void test_func() { printf("Test func\n"); }

void ThreadPool::start() {
    // Max # of threads the system supports
    const uint32_t num_threads = std::thread::hardware_concurrency();
    for (uint32_t ii = 0; ii < num_threads; ++ii) {
        threads.emplace_back(std::thread(&ThreadPool::thread_loop, this));
    }
}

void ThreadPool::thread_loop() {
    while (true) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            mutex_condition.wait(lock, [this] { return !jobs.empty() || should_terminate; });
            if (should_terminate) {
                return;
            }
            job = jobs.front();
            jobs.pop();
        }
        job();
    }
}

template <typename T>
void ThreadPool::map_jobs(const std::function<void(T args)>& job, std::vector<T> iter) {
    for (auto element : iter) {
        this->queue_job([element]{job(element)});
    }
}

void ThreadPool::queue_job(const std::function<void()>& job) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        jobs.push(job);
    }
    mutex_condition.notify_one();
}

bool ThreadPool::busy() {
    bool poolbusy;
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        poolbusy = !jobs.empty();
    }
    return poolbusy;
}

void ThreadPool::stop() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        should_terminate = true;
    }
    mutex_condition.notify_all();
    for (std::thread& active_thread : threads) {
        active_thread.join();
    }
    threads.clear();
}
