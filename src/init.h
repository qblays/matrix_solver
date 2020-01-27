#ifndef INIT_H
#define INIT_H

#include "defines.h"
#include <cstddef>
#include <mpi.h>
using initializer = double (*) (size_t i, size_t j, size_t);

size_t
compute_alloc_size (size_t n, size_t m);

double *
alloc_rows (size_t n, size_t m, size_t n_to_alloc, double **rows_p);

void
init_mat (double **rows_p, size_t n, size_t m, initializer f);

void
print_mat (double **rows_p, size_t n, size_t m);
#endif