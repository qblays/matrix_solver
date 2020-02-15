
#include "init.h"
#include "matrix_op.h"
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
main (int argc, const char **argv)
{

  MPI_Init (&argc, (char ***)&argv);
  int rank, commSize;

  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
  ScopeGuard close_file = [&] () {
    MPI_Finalize ();
    printf ("finish");
    fflush (stdout);
  };
  if (check_args (argc, argv) == false)
    {
      printf ("Usage: %s n m <filename>\n", argv[0]);
      return 0;
    }
  size_t n = atoi (argv[1]);
  size_t m = atoi (argv[2]);
  const char *filename = nullptr;
  if (argc == 4)
    {
      filename = argv[3];
    }
  if (1)
    {
      printf ("rank = %d, commSize = %d\n", rank, commSize);
      printf ("n = %lu, m = %lu, file = %s\n", n, m,
              filename ? filename : "\0");
#ifdef AVX
      printf ("AVX\n");
#endif
    }
  if (rank == 0)
    {
      auto fp = fopen ("log.txt", "a");
      fprintf (fp, "n=%lu, m=%lu, file=%s\n", n, m, filename);
      fclose (fp);
    }
  auto total_alloc_size = compute_alloc_size (n, m);
  auto rows_p_p = std::make_unique<double *[]> (n / m + (n % m > 0));
  auto rows_p = rows_p_p.get ();

  auto mat_body =
      std::unique_ptr<double[]> (alloc_rows (n, m, total_alloc_size, rows_p));
  printf ("%d th process should alloc %lu(%lf MB)\n", rank, total_alloc_size,
          (double)total_alloc_size * sizeof (double) / (1UL << 20));

  size_t sum = 0;
  MPI_Reduce (&total_alloc_size, &sum, 1, MPI_UNSIGNED_LONG, MPI_SUM, root,
              MPI_COMM_WORLD);

  if (rank == root)
    {
      printf ("Allocated: %lu, wanted: %lu\n", sum, n * (n + 1) / 2);
      assert (sum == n * (n + 1) / 2);
    }
  if (filename)
    {
      if (!init_mat_file (rows_p, n, m, filename))
        {
          printf ("error reading file\n");
          return 0;
        }
    }
  else
    init_mat (rows_p, n, m, f1);
  printf ("print given matrix\n");

  print_mat_beauty (0, n, m, rows_p, 7);
  auto workspace = std::make_unique<double[]> (5 * n);
  auto d = workspace.get ();
  auto b = d + n;
  auto y = b + n;
  auto x = y + n;
  auto r = x + n;
  for (size_t i = 0; i < n; i++)
    {
      y[i] = 0;
    }
  printf ("init_b\n");
  double norm_mat = 0;
  init_b_get_norm (b, n, m, 0, rows_p, norm_mat);
  printf ("norm_mat = %lf\n", norm_mat);
  MPI_Bcast (b, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  timeval t1, t2;
  printf ("cholesky_decomp_bu_thread\n");
  gettimeofday (&t1, nullptr);

  int local_res = cholesky_decomp_bu_thread (rows_p, d, n, m, norm_mat);
  gettimeofday (&t2, nullptr);
  // rows_p stores cholesky decomposed matrix
  printf ("w%d: choletsky res = %d\n", rank, local_res);
  if (check_res (0, local_res))
    return 0;
  auto elapsed = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) * 1e-6;
  printf ("R: \n");
  print_mat_beauty (0, n, m, rows_p, 5, 1);
  printf ("compute_y\n");
  local_res = compute_y_new (rows_p, b, y, n, m, norm_mat);
  if (check_res (0, local_res))
    return 0;
  printf ("compute_x\n");

  local_res = compute_x (rows_p, x, y, d, n, m, norm_mat, 0);
  if (check_res (0, local_res))
    return 0;
  if (rank == 0)
    {
      // printf ("vec b: \n");
      // print_vec (b, n, 7);
      // printf ("vec y: \n");
      // print_vec (y, n, 7);
      printf ("vec x: \n");
      print_vec (x, n, 50);
    }
  printf ("reinit mat\n");
  if (filename)
    {
      if (!init_mat_file (rows_p, n, m, filename))
        {
          printf ("error reading file\n");
          return 0;
        }
    }
  else
    init_mat (rows_p, n, m, f1);
  find_disrep_vec (b, x, r, n, m, 0, rows_p);
  if (rank == 0)
    {
      auto resid = norma_vec2 (r, n);
      auto fp = fopen ("log.txt", "a");
      printf ("n=%lu, m=%lu, file=%s, Residual = %e, elapsed = %lf\n", n, m,
              filename, resid, elapsed);
      fprintf (fp, "-----Residual = %e, elapsed = %lf\n", resid, elapsed);
      fclose (fp);
    }
  return 0;
}