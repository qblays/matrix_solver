
#include "init.h"
#include "solver.h"
#include <cassert>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <sys/time.h>
const int Tag = 0;
const int root = 0;
using initializer = double (*) (size_t i, size_t j, size_t);

double
f1 (size_t i, size_t j, size_t n = 0)
{
  return n - std::max (i, j);
}

int
main (int argc, char **argv)
{

  MPI_Init (&argc, &argv);
  int rank, commSize;

  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
  size_t n = atoi (argv[1]);
  size_t m = atoi (argv[2]);
  if (1)
    {
      printf ("rank = %d, commSize = %d\n", rank, commSize);
      printf ("using: %s n m\n", argv[0]);
      printf ("n = %lu, m = %lu\n", n, m);
#ifdef AVX
      printf ("AVX\n");
#endif
    }
  auto total_alloc_size = compute_alloc_size (n, m);
  auto rows_p = new double *[n / m + (n % m > 0)];

  auto mat_body = alloc_rows (n, m, total_alloc_size, rows_p);
  printf ("%d th process should alloc %lu\n", rank, total_alloc_size);

  size_t sum = 0;
  MPI_Reduce (&total_alloc_size, &sum, 1, MPI_UNSIGNED_LONG, MPI_SUM, root,
              MPI_COMM_WORLD);

  if (rank == root)
    {
      printf ("Allocated: %lu, wanted: %lu\n", sum, n * (n + 1) / 2);
      assert (sum == n * (n + 1) / 2);
    }
  init_mat (rows_p, n, m, f1);
  // print_mat (rows_p, n, m);
  auto d_p = std::unique_ptr<double> (new double[n]);

  timeval t1, t2;
  gettimeofday (&t1, nullptr);
  auto res = cholesky_decomp_bu_thread (rows_p, d_p.get (), n, m, 1);
  printf ("w%d: choletsky res = %d\n", rank, res);
  gettimeofday (&t2, nullptr);
  printf ("elapsed = %lf\n",
          (t2.tv_sec - t1.tv_sec) * 1.e6 + t2.tv_usec - t1.tv_usec);
  printf ("R: \n");
  print_mat_triangle (rows_p, n, m);

  delete[] rows_p;
  delete[] mat_body;

  MPI_Finalize ();
}