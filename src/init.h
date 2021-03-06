#ifndef INIT_H
#define INIT_H

#include "defines.h"
#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
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

void
print_mat_triangle (double **rows_p, size_t n, size_t m);

bool
check_args (const int argc, char **argv);

bool
init_mat_file (double **rows_p, size_t n, size_t m, const char *filename);

void
gather_row (size_t i, size_t n, size_t m, int root, vec buf, double **rows_p,
            int action = 0);

void
gather_row_runner (size_t i, size_t n, size_t m, int root, vec buf,
                   double **&rows_p, int *recvcounts, int *recvcounts2,
                   int *displs, double *bbuf, double *gathered);

void
gather_col (size_t j, size_t n, size_t m, int root, vec buf, double **&rows_p);

void
init_b_get_norm (vec b, size_t n, size_t m, int root, double **&rows_p,
                 double &norm);

void
find_disrep_vec (vec b, vec x, vec r, size_t n, size_t m, int root,
                 double **&rows_p);

bool
check_res (int root, int loc_res);

void printf_root (int root, const char *format, ...);

class ScopeGuard
{
public:
  template <class Callable>
  ScopeGuard (Callable &&fn) : fn_ (std::forward<Callable> (fn))
  {
  }

  ScopeGuard (ScopeGuard &&other);

  ~ScopeGuard ();

  ScopeGuard (const ScopeGuard &) = delete;
  void
  operator= (const ScopeGuard &) = delete;

private:
  std::function<void ()> fn_;
};

#endif
