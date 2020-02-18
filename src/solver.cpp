#include "solver.h"
#include "init.h"
#include <cstring>
#include <matrix_op.h>
#include <memory>
#include <mpi.h>

bool
cholesky_decomp (double **&rows_p, vec d, size_t n, size_t m,
                           double norma)
{
  bool ret = 0;
  int rank, commSize;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);

  auto Revrsd_p = std::unique_ptr<double[]> (new double[m * (m + 1) / 2]);
  auto Column_thread = std::unique_ptr<double[]> (new double[m * n]);
  auto D_p = std::unique_ptr<double[]> (new double[m]);
  auto Revrsd = Revrsd_p.get ();
  double *Temp = nullptr;
  auto column_thread = Column_thread.get ();
  auto D = D_p.get ();

  int N = n / m;
  int I = 0;
  for (I = 0; I < N; I++)
    {
      if (I % commSize == rank)
        {
          memcpy (column_thread, rows_p[I],
                  (m * m * I + m * (m + 1) / 2) * sizeof (double));
        }
      MPI_Bcast (column_thread, m * m * I + m * (m + 1) / 2, MPI_DOUBLE,
                 I % commSize, MPI_COMM_WORLD);
      Temp = column_thread + m * m * I;
      for (int k = 0; k < I; k++)
        {
          minus_RTDRu_ofast (Temp, column_thread + k * m * m, d + k * m,
                             column_thread + k * m * m, m);
        }

      ret = std::max (cholesky_decomp_U (Temp, D, m, norma), ret);
      if (I % commSize == rank)
        {
          memcpy (get_bl (rows_p, I, I, m), Temp,
                  m * (m + 1) / 2 * sizeof (double));
        }
      memcpy (d + I * m, D, m * sizeof (double));
      ret = std::max (reverse_upper (Temp, Revrsd, m, norma), ret);
      if (ret == 1)
        return ret;
      int J;
      for (J = I + 1; J < N; J += 1)
        {
          if (J % commSize == rank)
            {
              double *A = get_bl (rows_p, I, J, m);

              for (int k = 0; k < I; k++)
                {
#ifdef AVX
                  minus_RTDR_o1_avx_ggggg (A, column_thread + k * m * m,
                                           d + k * m, get_bl (rows_p, k, J, m),
                                           m);
#else
                  minus_RTDR_o1 (A, column_thread + k * m * m, d + k * m,
                                 get_bl (rows_p, k, J, m), m);
#endif
                }
              DRtA (Revrsd, A, D, m);
            }
        }
      size_t l;
      if ((n - N * m) && ((J == N)) && ((J) % commSize == rank))
        {
          l = n - N * m;
          double *A = get_bl (rows_p, I, J, m, l);
          for (int k = 0; k < I; k++)
            {
              minus_RTDR_l_o1 (A, column_thread + k * m * m, d + k * m,
                               get_bl (rows_p, k, J, m, l), m, l);
            }
          DRtA_l (Revrsd, A, D, m, l);
        }
    }
  size_t l;
  if ((l = n - N * m) && (I % commSize == rank))
    {
      double *A = get_bl (rows_p, I, I, m, l);
      for (int k = 0; k < I; k++)
        {
          auto p = rows_p[I] + k * m * l;
          minus_RTDRu_l (A, p, d + k * m, m, l);
        }
      ret = std::max (cholesky_decomp_U (A, D, l, norma), ret);
      memcpy (d + I * m, D, l * sizeof (double));
    }
  if ((l = n - N * m))
    {
      MPI_Bcast (d + I * m, l, MPI_DOUBLE, I % commSize, MPI_COMM_WORLD);
    }
  return ret;
}

bool
cholesky_decomp_U (mat a, vec d, int n, double norma)
{
  for (int i = 0; i < n; i++)
    {
      double sum = a[get_elU (i, i, n)];
      for (int k = 0; k < i; k++)
        {
          sum -= a[get_elU (k, i, n)] * a[get_elU (k, i, n)] * d[k];
        }
      d[i] = sgn (sum);
      if (is_double_too_small (a[get_elU (i, i, n)], EPS * norma))
        {
          return 1;
        }
      a[get_elU (i, i, n)] = sqrt (std::abs (sum));
      if (is_double_too_small (a[get_elU (i, i, n)], EPS * norma))
        {
          return 1;
        }

      int j;

      for (j = i + 1; j < n - 3; j += 4)
        {
          double sum1 = a[get_elU (i, j, n)];
          double sum2 = a[get_elU (i, j + 1, n)];
          double sum3 = a[get_elU (i, j + 2, n)];
          double sum4 = a[get_elU (i, j + 3, n)];
          for (int k = 0; k < i; k++)
            {
              sum1 -= a[get_elU (k, i, n)] * a[get_elU (k, j, n)] * d[k];
              sum2 -= a[get_elU (k, i, n)] * a[get_elU (k, j + 1, n)] * d[k];
              sum3 -= a[get_elU (k, i, n)] * a[get_elU (k, j + 2, n)] * d[k];
              sum4 -= a[get_elU (k, i, n)] * a[get_elU (k, j + 3, n)] * d[k];
            }

          a[get_elU (i, j, n)] = sum1 / (a[get_elU (i, i, n)] * d[i]);
          a[get_elU (i, j + 1, n)] = sum2 / (a[get_elU (i, i, n)] * d[i]);
          a[get_elU (i, j + 2, n)] = sum3 / (a[get_elU (i, i, n)] * d[i]);
          a[get_elU (i, j + 3, n)] = sum4 / (a[get_elU (i, i, n)] * d[i]);
        }
      for (; j < n; j += 1)
        {
          double sum1 = a[get_elU (i, j, n)];
          for (int k = 0; k < i; k++)
            {
              sum1 -= a[get_elU (k, i, n)] * a[get_elU (k, j, n)] * d[k];
            }

          a[get_elU (i, j, n)] = sum1 / (a[get_elU (i, i, n)] * d[i]);
        }
    }
  return 0;
}

bool
compute_y (double **&rows_p, vec b, vec y, size_t n, size_t m, double norma)
{
  int rank, commSize;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
  int columns_n = n / m + (n % m > 0);
  int reminder = n - (n / m) * m;
  for (int I = 0; I < columns_n; I++)
    {
      auto col_width = m;
      if (I == columns_n - 1 && reminder > 0)
        {
          col_width = reminder;
        }

      if (I % commSize == rank)
        {
          for (size_t j = 0; j < col_width; j++)
            {
              double *a = rows_p[I];
              double sum = 0;
              // square
              for (size_t i = 0; i < I * m; i++)
                {
                  sum += a[col_width * i + j] * y[i];
                }
              // triangle
              a += I * m * col_width;
              for (size_t i = I * m; i < I * m + j; i++)
                {
                  sum += a[get_elU (i - I * m, j, col_width)] * y[i];
                }
              if (is_double_too_small (a[get_elU (j, j, col_width)],
                                       EPS * norma))
                {
                  return 1;
                }
              y[I * m + j] =
                  (b[I * m + j] - sum) / a[get_elU (j, j, col_width)];
            }
        }
      MPI_Bcast (y + I * m, col_width, MPI_DOUBLE, I % commSize,
                 MPI_COMM_WORLD);
    }
  return 0;
}

bool
compute_x (double **&rows_p, vec x, vec y, vec d, size_t n, size_t m,
           double norma, int root)
{
  int rank, commSize;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
  double *buf = nullptr;
  if (rank == root)
    {
      buf = new double[n];
    }
  for (size_t i = n - 1; i < n; i--)
    {
      gather_row (i, n, m, root, buf, rows_p);
      if (rank == root)
        {
          double sum = 0;
          for (size_t j = i + 1; j < n; j++)
            {
              sum += buf[j] * x[j];
            }
          if (is_double_too_small ((buf[i]), EPS * norma))
            {
              delete[] buf;
              return 1;
            }
          x[i] = d[i] * (y[i] - d[i] * sum) / buf[i];
        }
    }
  if (rank == root)
    {
      delete[] buf;
    }
  return 0;
}
