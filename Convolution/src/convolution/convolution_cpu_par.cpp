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
    threads.start();
    threads.wait();
}

void par_convolution_loop(float* image, uint32_t image_size, float* kernel, uint32_t kernel_size, float* out, uint32_t qtd_loops) {
    uint32_t padding = (kernel_size - 1) / 2;
    float* image_aux = image;
    for (size_t i = 0; i < qtd_loops; i++) {
        cpu_par_convolution(image_aux, kernel, out, image_size, kernel_size);
        // memcpy(image, out, sizeof(float) * image_size * image_size);
        image_aux = out;
    }

    // checker<<<1, 1>>>(image_in_device, image_out_device, image_size);
    // gpuErrchk(cudaGetLastError());
    // gpuErrchk(cudaDeviceSynchronize());
}

float* run_convolution_par(uint32_t n, uint32_t qtd_loops, uint32_t padding) {
    uint32_t kernel_size = padding * 2 + 1;
    // uint32_t padding = (kernel_size - 1) / 2;
    // uint32_t n = 32;
    uint32_t image_size = 32 * n + 2 * padding;

    srand(345);

    float* image = alloc_image(image_size);
    float* image_out = alloc_image_out(image_size);
    memcpy(image_out, image, image_size * image_size * sizeof(float));
    float* kernel = alloc_kernel(kernel_size);
    // show_matrix(image, image_size, 15);

    // save_matrix(image, image_size, image_size, "data/convolution_matrix.txt");
    par_convolution_loop(image, image_size, kernel, kernel_size, image_out, qtd_loops);

    // show_matrix(image_out, image_size, 15);

    // save_matrix(image_out, image_size, 15, "data/convolution_matrix_out_cpu_par.txt");
    free(image);
    free(kernel);

    return image_out;
}