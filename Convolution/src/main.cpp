#include <assert.h>
#include <stdio.h>
#include <cmath>
#include <stdlib.h>

#include "convolution/convolution.cuh"

int main(int argc, const char* argv[]) {
    uint32_t qtd_loops = 1000;
    uint32_t n = 400;
    uint32_t padding = 1;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (argc > 2) {
        qtd_loops = atoi(argv[2]);
    }
    float* result_gpu = run_convolution(n, qtd_loops);
    printf("DONE WITH GPU %p\n", result_gpu);
}
