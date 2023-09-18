#include "alloc_vector_mem.hpp"

#include <functional>
#include <random>
#include <ranges>

#include "../../../Thread-Pool/src/thread-pool.hpp"

unsigned int seed = 435;

float* alloc_vector(uint32_t vector_size) {
    float* vector = (float*)malloc(sizeof(float) * vector_size);
    ThreadPool threads = ThreadPool{};

    uint32_t idx_start = 0;
    uint32_t idx_end = 10;
    uint32_t qtd_items = (vector_size) / (idx_end - idx_start);
    auto range = std::ranges::iota_view{idx_start, idx_end};
    std::uniform_int_distribution distribution(idx_start, idx_end);
    std::mt19937 random_number_engine;  // pseudorandom number generator
    auto random_number = std::bind(distribution, random_number_engine);

    threads.map_jobs(
        [vector, vector_size, qtd_items, random_number](auto idx) {
            uint32_t i_begin = qtd_items * idx;
            uint32_t i_end = i_begin + qtd_items;

            if (vector_size - i_end < qtd_items && vector_size - i_end > 0) {
                i_end += vector_size - i_end;
            }

            for (uint32_t i = i_begin; i < i_end; i++) {
                    vector[i] = (float)rand_r(&seed) / (float)RAND_MAX;
            }
        },
        range);

    threads.start();
    threads.wait();

    return vector;
}
