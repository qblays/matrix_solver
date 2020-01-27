#ifndef SOLVER_H
#define SOLVER_H
#include "defines.h"
inline double *
get_bl (double **&rows_p, size_t I, size_t J, size_t m)
{
  return rows_p[J] + I * m;
}

bool
cholesky_decomp_bu_thread (double **&rows_p, vec d, size_t n, size_t m,
                           double norma);

bool
cholesky_decomp_U (mat a, vec d, int n, double norma);
#endif