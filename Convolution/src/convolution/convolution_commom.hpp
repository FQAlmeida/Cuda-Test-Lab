#pragma once

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void show_matrix(float* matrix, uint32_t size, uint32_t max_print);

void save_matrix(float* matrix, uint32_t size, uint32_t max_print, const char* filename);