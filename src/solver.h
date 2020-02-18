#ifndef SOLVER_H
#define SOLVER_H
#include "defines.h"
inline double *
get_bl (double **&rows_p, size_t I, size_t J, size_t m, size_t l = 0)
{

  return rows_p[J] + I * m * (l > 0 ? l : m);
}

bool
cholesky_decomp (double **&rows_p, vec d, size_t n, size_t m,
                           double norma);

bool
cholesky_decomp_U (mat a, vec d, int n, double norma);

bool
compute_y (double **&rows_p, vec b, vec y, size_t n, size_t m,
               double norma);

bool
compute_x (double **&rows_p, vec x, vec y, vec d, size_t n, size_t m,
           double norma, int root);
#endif
