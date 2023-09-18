#include "alloc_image_mem.hpp"

#include <assert.h>

#include <functional>
#include <random>
#include <ranges>

#include "../../../Thread-Pool/src/thread-pool.hpp"

unsigned int seed = 435;

float* alloc_image(uint32_t image_size) {
    float* image = (float*)malloc(sizeof(float) * image_size * image_size);
    ThreadPool threads = ThreadPool{};

    uint32_t idx_start = 0;
    uint32_t idx_end = image_size;
    uint32_t qtd_items = (image_size * image_size) / (idx_end - idx_start);
    auto range = std::ranges::iota_view{idx_start, idx_end};

    threads.map_jobs(
        [image, image_size, qtd_items](auto idx) {
            uint32_t i_begin = (qtd_items * idx) / image_size;
            uint32_t i_end = i_begin + (qtd_items / image_size);

            for (uint32_t i = i_begin; i < i_end; i++) {
                for (uint32_t j = 0; j < image_size; j++) {
                    image[j + i * image_size] = (float)rand_r(&seed) / (float)RAND_MAX;
                }
            }
        },
        range);

    threads.start();
    threads.wait();

    for (uint64_t i = 0; i < image_size; i++) {
        image[i] = image[i + image_size];
        image[image_size * (image_size - 1) + i] = image[image_size * (image_size - 2) + i];
        image[image_size * i] = image[image_size * i + 1];
        image[image_size * i + (image_size - 1)] = image[image_size * i + (image_size - 2)];
    }

    image[0] = image[image_size + 1];
    image[image_size - 1] = image[2 * (image_size - 1)];
    image[image_size * (image_size - 1)] = image[image_size * (image_size - 2) + 1];
    image[image_size * image_size - 1] = image[image_size * (image_size - 3)];

    return image;
}

float* alloc_image_out(uint32_t image_size) {
    float* image_out = (float*)calloc(image_size * image_size, sizeof(float));
    return image_out;
}

float* alloc_kernel(uint32_t kernel_size) {
    float* kernel = (float*)malloc(sizeof(float*) * kernel_size * kernel_size);
    ThreadPool threads = ThreadPool{};

    uint32_t idx_start = 0;
    uint32_t idx_end = kernel_size;
    uint32_t qtd_items = (kernel_size * kernel_size) / (idx_end - idx_start);
    auto range = std::ranges::iota_view{idx_start, idx_end};
    threads.map_jobs(
        [kernel, kernel_size, qtd_items](auto idx) {
            uint32_t i_begin = (qtd_items * idx) / kernel_size;
            uint32_t i_end = i_begin + (qtd_items / kernel_size);

            for (uint32_t i = i_begin; i < i_end; i++) {
                for (uint32_t j = 0; j < kernel_size; j++) {
                    kernel[i + j * kernel_size] = 1.0;
                }
            }
        },
        range);

    threads.start();
    threads.wait();

    return kernel;
}
