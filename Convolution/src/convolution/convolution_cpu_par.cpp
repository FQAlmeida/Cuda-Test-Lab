#include "convolution_cpu_par.hpp"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ranges>

#include "../../../Thread-Pool/src/thread-pool.hpp"
#include "../alloc_image_mem/alloc_image_mem.hpp"
#include "convolution_commom.hpp"

void cpu_par_convolution(float* image, float* kernel, float* out, uint32_t image_size, uint32_t kernel_size) {
    ThreadPool threads = ThreadPool{};
    uint32_t padding = (kernel_size - 1) / 2;
    auto range = std::ranges::iota_view{padding, image_size - padding};
    threads.map_jobs(
        [image, out, kernel, image_size, kernel_size, padding](uint32_t idx) {
            uint32_t i = idx;
            for (size_t j = padding; j < image_size - padding; j++) {
                float sum = 0;
                for (size_t i_desloc = 0; i_desloc < kernel_size; i_desloc++) {
                    for (size_t j_desloc = 0; j_desloc < kernel_size; j_desloc++) {
                        sum += image[(j - padding + j_desloc) + (i - padding + i_desloc) * image_size] *
                               kernel[i_desloc * kernel_size + j_desloc];
                    }
                }
                out[j + i * image_size] = sum / (kernel_size * kernel_size);
            }
        },
        range);
}

void par_convolution_loop(float* image, uint32_t image_size, float* kernel, uint32_t kernel_size, float* out) {
    uint32_t padding = (kernel_size - 1) / 2;

    for (size_t i = 0; i < 100; i++) {
        for (size_t j = 0; j < 10; j++) {
            cpu_par_convolution(image, kernel, out, image_size, kernel_size);
            memcpy(image, out, sizeof(float) * image_size * image_size);
        }
    }

    // checker<<<1, 1>>>(image_in_device, image_out_device, image_size);
    // gpuErrchk(cudaGetLastError());
    // gpuErrchk(cudaDeviceSynchronize());
}

uint32_t run_convolution_par(uint32_t n) {
    uint32_t kernel_size = 3;
    uint32_t padding = (kernel_size - 1) / 2;
    // uint32_t n = 32;
    uint32_t image_size = 32 * n + 2 * padding;

    srand(345);

    float* image = alloc_image(image_size);
    float* image_out = alloc_image_out(image_size);
    float* kernel = alloc_kernel(kernel_size);
    // show_matrix(image, image_size, 15);

    // save_matrix(image, image_size, image_size, "data/convolution_matrix.txt");
    par_convolution_loop(image, image_size, kernel, kernel_size, image_out);

    // show_matrix(image_out, image_size, 15);

    // save_matrix(image_out, image_size, image_size, "data/convolution_matrix_out.txt");

    free(image);
    free(image_out);
    free(kernel);

    return 0;
}