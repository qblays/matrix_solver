
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
  return i * 1000 + j;
}

inline size_t
get_elU (size_t i, size_t j, size_t n)
{
  size_t d = i;
  d = (d * d + d) / 2;
  return i * n + j - d;
}

size_t
compute_alloc_size (size_t n, size_t m)
{
  int rank, commSize;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
  int columns_n = n / m + (n % m > 0);
  printf ("columns_n = %d\n", columns_n);
  int reminder = n - (n / m) * m;
  size_t sum = 0;
  for (int i = 0; i < columns_n; i++)
    {
      int col_width = m;
      if (i == columns_n - 1 && reminder > 0)
        {
          col_width = reminder;
        }
      if (i % commSize == rank)
        {
          // printf ("%d th col size %d\n", i, col_width);
          sum += i * m * col_width;
          sum += (col_width * col_width + col_width) / 2;
        }
    }
  return sum;
}

double *
alloc_rows (size_t n, size_t m, size_t n_to_alloc, double **rows_p)
{
  int rank, commSize;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
  int columns_n = n / m + (n % m > 0);
  int reminder = n - (n / m) * m;
  auto mat = new double[n_to_alloc];
  printf ("allocated = %p\n", mat);
  size_t sum = 0;
  for (int i = 0; i < columns_n; i++)
    {
      int col_width = m;
      if (i == columns_n - 1 && reminder > 0)
        {
          col_width = reminder;
        }
      if (i % commSize == rank)
        {
          // printf ("%d th col size %d\n", i, col_width);
          rows_p[i] = mat + sum;
          printf ("%d th row in %p\n", i, mat + sum);
          sum += i * m * col_width;
          sum += (col_width * (col_width + 1)) / 2;
          size_t t = i * m * col_width + (col_width * (col_width + 1)) / 2;
          printf ("row %dth have size %lu\n", i, t);
        }
      else
        {
          rows_p[i] = nullptr;
        }
    }
  return mat;
}

void
init_mat (double **rows_p, size_t n, size_t m, initializer f)
{
  int rank, commSize;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
  int columns_n = n / m + (n % m > 0);
  int reminder = n - (n / m) * m;
  // size_t sum = 0;
  for (int I = 0; I < columns_n; I++)
    {
      auto col_width = m;
      if (I == columns_n - 1 && reminder > 0)
        {
          col_width = reminder;
        }
      if (I % commSize == rank)
        {
          double *a = rows_p[I];
          printf ("init column %d\n", I);
          for (size_t i = 0; i < I * m; i++)
            {
              for (size_t j = I * m; j < I * m + col_width; j++)
                {
                  a[i * col_width + j - I * m] = f (i, j, n);
                  printf ("%lu ", i * col_width + j);
                }
              printf ("\n");
            }
          a += I * m * col_width;
          for (size_t i = 0; i < col_width; i++)
            {
              for (size_t j = 0; j < i; j++)
                {
                  // a[get_elU (i, j, col_width)] = f (i, j, n);
                  // printf ("f(%lu, %lu) = %lf ", i + I * m, j + I * m, 0.);
                }
              for (size_t j = i; j < col_width; j++)
                {
                  a[get_elU (i, j, col_width)] = f (i + I * m, j + I * m, n);
                  // printf ("f(%lu, %lu) = %lf ", i + I * m, j + I * m,
                  //         f (i + I * m, j + I * m, n));
                  printf ("%lu ", get_elU (i, j, col_width));
                }
              printf ("\n");
            }
        }
    }
}

void
print_mat (double **rows_p, size_t n, size_t m)
{
  int rank, commSize;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
  int columns_n = n / m + (n % m > 0);
  int reminder = n - (n / m) * m;
  // size_t sum = 0;
  for (int I = 0; I < columns_n; I++)
    {
      auto col_width = m;
      if (I == columns_n - 1 && reminder > 0)
        {
          col_width = reminder;
        }
      if (I % commSize == rank)
        {
          printf ("COlumn number %d\n", I);
          double *a = rows_p[I];
          for (size_t i = 0; i < I * m; i++)
            {
              for (size_t j = I * m; j < I * m + col_width; j++)
                {
                  printf ("%lf ", a[i * col_width + j - I * m]);
                }
              printf ("\n");
            }
          a += I * m * col_width;
          printf ("square size= %lu\n", I * m * col_width);
          printf ("printing triangle\n");
          for (size_t i = 0; i < col_width; i++)
            {
              for (size_t j = 0; j < i; j++)
                {
                  printf ("%lf ", 0.);
                }
              for (size_t j = i; j < col_width; j++)
                {
                  printf ("%lf ", a[get_elU (i, j, col_width)]);
                }
              printf ("\n");
            }
        }
    }
}

int
main (int argc, char **argv)
{
  MPI_Init (NULL, NULL);
  // MPI_Comm comm;
  int rank, commSize;

  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
  size_t n = atoi (argv[1]);
  size_t m = atoi (argv[2]);
  if (rank == root)
    {
      printf ("using: %s n m\n", argv[0]);
      printf ("n = %lu, m = %lu\n", n, m);
    }
  size_t sendbuf[2] = {n, m};
  size_t recvbuf[2];
  MPI_Scatter (sendbuf, 2, MPI_UNSIGNED_LONG, recvbuf, 2, MPI_UNSIGNED_LONG,
               root, MPI_COMM_WORLD);
  auto total_alloc_size = compute_alloc_size (n, m);
  auto rows_p = new double *[n / m + (n % m > 0)];

  auto mat_body = alloc_rows (n, m, total_alloc_size, rows_p);
  printf ("%d th process should alloc %lu\n", rank, total_alloc_size);

  size_t sum = 0;
  MPI_Reduce (&total_alloc_size, &sum, 1, MPI_UNSIGNED_LONG, MPI_SUM, root,
              MPI_COMM_WORLD);

  if (rank == root)
    {
      printf ("sum = %lu, wanted = %lu\n", sum, n * (n + 1) / 2);
    }
  printf ("n = %lu, m = %lu\n", n, m);
  init_mat (rows_p, n, m, f1);
  print_mat (rows_p, n, m);

  delete[] rows_p;
  delete[] mat_body;

  MPI_Finalize ();
}