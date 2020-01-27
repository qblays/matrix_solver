#include "init.h"

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
  // printf ("allocated = %p\n", mat);
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
          printf ("%d th row\n", i);
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
                  printf ("%3lu ", i * col_width + j);
                }
              printf ("\n");
            }
          a += I * m * col_width;
          for (size_t i = 0; i < col_width; i++)
            {
              for (size_t j = 0; j < i; j++)
                {
                  // a[get_elU (i, j, col_width)] = f (i, j, n);
                  printf ("    ");
                }
              for (size_t j = i; j < col_width; j++)
                {
                  a[get_elU (i, j, col_width)] = f (i + I * m, j + I * m, n);
                  // printf ("f(%lu, %lu) = %lf ", i + I * m, j + I * m,
                  //         f (i + I * m, j + I * m, n));
                  printf ("%3lu ", get_elU (i, j, col_width));
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
                  printf ("%0*.2lf ", 4, a[i * col_width + j - I * m]);
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
                  printf ("%4.2lf ", 0.);
                }
              for (size_t j = i; j < col_width; j++)
                {
                  printf ("%4.2lf ", a[get_elU (i, j, col_width)]);
                }
              printf ("\n");
            }
        }
    }
}