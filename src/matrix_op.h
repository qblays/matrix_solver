#ifndef MATRIX_OP
#define MATRIX_OP
#include "defines.h"
#include <cstdio>

void
minus_RTDR_o1 (mat __restrict A, mat __restrict R1, vec __restrict D,
               mat __restrict R2, int n);

void
minus_RTDRu_ofast (mat A, mat R1, vec D, mat R2, int n);

#ifdef AVX
#include <immintrin.h>
void
minus_RTDR_o1_avx_ggggg (mat A, mat R1, vec D, mat R2, int n);

void
minus_RTDRu_o1_avx_ggggg (mat A, mat R1, vec D, mat R2, int n);
#endif

void
minus_RTDR_l_o1 (mat A, mat R1, vec D, mat R2, int n, int l);

void
minus_RTDRu_l (mat A, mat R1, vec D, int n, int l);

bool
reverse_upper (mat A, mat B, int n, double norma);

void
DRtA (mat R, mat A, vec D, int n);

void
DRtA_l (mat R, mat A, vec D, int n, int l);

void
print_matrix (double *a, size_t n);

void
print_matrix_b_upper (double *a, int n);

void
print_mat_beauty (int root, size_t n, size_t m, double **&rows_p,
                  size_t len = -1, bool log = 0, int width = 8,
                  int precision = 2);

void
print_vec (vec a, size_t n, size_t len = -1UL);
double
norma_vec2 (double *r, int n);
double
find_error (double *x, int n);
#endif