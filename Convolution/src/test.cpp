#include <assert.h>
#include <stdio.h>
#include <cmath>
#include <stdlib.h>

#include "convolution/convolution.cuh"
#include "convolution/convolution_cpu_par.hpp"
#include "convolution/convolution_serial.hpp"

int main(int argc, const char* argv[]) {
    uint32_t n = 400;
    uint32_t qtd_loops = 20;
    uint32_t padding = 1;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    float* result_gpu = run_convolution(n, qtd_loops, padding);
    printf("DONE WITH GPU %p\n", result_gpu);
    float* result_cpu_par = run_convolution_par(n, qtd_loops, padding);
    printf("DONE WITH CPU PAR %p\n", result_cpu_par);
    float* result_cpu = run_convolution_serial(n, qtd_loops, padding);
    printf("DONE WITH CPU %p\n", result_cpu);
    uint32_t line_width = 32 * n + 2 * padding;
    float MAX_ERROR = 1e-2;
    for (size_t i = 0; i < line_width; i++) {
        for (size_t j = 0; j < line_width; j++) {
            printf("%f %f %f\n", result_cpu[i * line_width + j], result_gpu[i * line_width + j],result_cpu_par[i * line_width + j]);
            assert(fabs(result_cpu[i * line_width + j] - result_gpu[i * line_width + j]) < MAX_ERROR);
            assert(fabs(result_cpu[i * line_width + j] - result_cpu_par[i * line_width + j]) < MAX_ERROR);
        }
    }
}
