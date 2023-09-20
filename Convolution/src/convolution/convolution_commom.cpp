#include "convolution_commom.hpp"

#include <assert.h>

void show_matrix(float* matrix, uint32_t size, uint32_t max_print) {
    for (uint32_t i = 0; i < std::min(size, max_print); i++) {
        for (uint32_t j = 0; j < std::min(size, max_print); j++) {
            printf("%.2f\t|", matrix[j + i * size]);
            // assert(matrix[j + i * size] <= 1.0);
        }
        putchar('\n');
    }
}

void save_matrix(float* matrix, uint32_t size, uint32_t max_print, const char* filename) {
    FILE* fptr;
    fptr = fopen(filename, "w+");
    if (fptr == NULL) {
        printf("Error while opening file %s!\n", filename);
        exit(1);
    }
    for (uint32_t i = 0; i < std::min(size, max_print); i++) {
        for (uint32_t j = 0; j < std::min(size, max_print); j++) {
            fprintf(fptr, "%f\t|", matrix[j + i * size]);
        }
        fputc('\n', fptr);
    }
    fclose(fptr);
}