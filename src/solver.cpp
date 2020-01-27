#include "solver.h"
#include <cstring>
#include <matrix_op.h>
#include <memory>
#include <mpi.h>

bool
cholesky_decomp_bu_thread (double **&rows_p, vec d, size_t n, size_t m,
                           double norma)
{
  bool ret = 0;
  int rank, commSize;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);

  auto Revrsd_p = std::unique_ptr<double[]> (new double[m * (m + 1) / 2]);
  auto Temp_p = std::unique_ptr<double[]> (new double[m * m]);
  auto Column_thread = std::unique_ptr<double[]> (new double[m * n]);
  auto D_p = std::unique_ptr<double[]> (new double[m]);
  // auto Test_p = std::unique_ptr<double[]> (new double[m * m * 3 + m]);
  auto Revrsd = Revrsd_p.get ();
  auto Temp = Temp_p.get ();
  // auto Test_mat1 = Test_p.get ();
  // auto Test_mat3 = Test_mat1 + m * m;
  // auto Test_mat2 = Test_mat3 + m * m;
  // auto Test_vec = Test_mat2 + m * m;
  auto column_thread = Column_thread.get ();
  auto D = D_p.get ();

  // get_ub (n, m, u, b);
  int N = n / m;
  int I = 0;
  // fill_random (Test_mat1, m * m);
  // fill_random (Test_vec, m);
  // fill_random (Test_mat3, m * m);
  for (I = 0; I < N; I++)
    {
      //! memcpy (Temp, &a[get_bl (I, I, u, b)], m * (m + 1) / 2 * sizeof
      //! (double));
      // memcpy (Temp, get_bl(rows_p, I, I, m), m * (m + 1) / 2 * sizeof
      // (double));
      // printf ("I = %d\n", I);
      // MPI_Scatter (get_bl (rows_p, I, I, m), m * (m + 1) / 2, MPI_DOUBLE,
      // Temp,
      //              m * m, MPI_DOUBLE, I % commSize, MPI_COMM_WORLD);
      if (I % commSize == rank)
        {
          memcpy (Temp, get_bl (rows_p, I, I, m),
                  m * (m + 1) / 2 * sizeof (double));
        }
      MPI_Bcast (Temp, m * (m + 1) / 2, MPI_DOUBLE, I % commSize,
                 MPI_COMM_WORLD);

      //! pthread_barrier_wait (bar);
      //! fill_column (column_thread, a, n, m, m, m, I, I);
      if (I % commSize == rank)
        {
          memcpy (column_thread, get_bl (rows_p, 0, I, m),
                  (m * m * I) * sizeof (double));
        }
      // MPI_Scatter (get_bl (rows_p, 0, I, m), m * m * I, MPI_DOUBLE,
      //              column_thread, m * n, MPI_DOUBLE, I % commSize,
      //              MPI_COMM_WORLD);
      MPI_Bcast (column_thread, m * m * I, MPI_DOUBLE, I % commSize,
                 MPI_COMM_WORLD);
      // printf ("Temp: \n");
      // print_matrix_b_upper (Temp, m);
      if (I > 0)
        {
          // printf ("column thread: \n");
          // print_matrix (column_thread, m);
        }
      for (int k = 0; k < I; k++)
        {
// minus_RTDRu_o1 (Temp, column_thread + k * m * m, d + k * m, m);
#ifdef AVX
          
          minus_RTDRu_o1_avx_ggggg (Temp, column_thread + k * m * m, d + k * m,
                                    column_thread + k * m * m, m);
#else
          minus_RTDRu_ofast (Temp, column_thread + k * m * m, d + k * m,
                             column_thread + k * m * m, m);
#endif
        }

      ret = std::max (cholesky_decomp_U (Temp, D, m, norma), ret);
      if (I % commSize == rank)
        {
          // !memcpy (&a[get_bl (I, I, u, b)], Temp,
          //         m * (m + 1) / 2 * sizeof (double));
          // !memcpy (d + I * m, D, m * sizeof (double));
          memcpy (get_bl (rows_p, I, I, m), Temp,
                  m * (m + 1) / 2 * sizeof (double));
          // memcpy (d + I * m, D, m * sizeof (double));
        }
      // !!!!MPI_Scatter (D, m, MPI_DOUBLE, d + I * m, m, MPI_DOUBLE, I %
      // commSize,
      //              MPI_COMM_WORLD);
      // printf ("Temp: \n");
      // print_matrix_b_upper (Temp, m);
      memcpy (d + I * m, D, m * sizeof (double));
      ret = std::max (reverse_upper (Temp, Revrsd, m, norma), ret);
      // printf ("Revrsd: \n");
      // print_matrix_b_upper (Revrsd, m);
      if (ret == 1)
        return ret;
      int J;
      for (J = I + 1; J < N; J += 1)
        {
          if (J % commSize == rank)
            {
              // pthread_mutex_lock (mut);
              // printf ("tid %d doing block (%d, %d)\n", tid, I, J);
              // pthread_mutex_unlock (mut);
              //! double *A = &a[get_bl (I, J, u, b)];
              double *A = get_bl (rows_p, I, J, m);
              // printf ("I= %d, J=%d\n", I, J);
              // print_matrix (A, m);
              // memcpy (Temp, A, m * m * sizeof (double));

              for (int k = 0; k < I; k++)
                {
#ifdef AVX
                  minus_RTDR_o1_avx_ggggg (Temp, column_thread + k * m * m,
                                           d + k * m, get_bl (rows_p, k, J, m),
                                           m);
#else
                  // fill_random (Test_mat1, m * m);
                  // fill_random (Test_vec, m);
                  // fill_random (Test_mat3, m * m);
                  // minus_RTDR_o1 (Test_mat3, Test_mat1, Test_vec, Test_mat1,
                  // m);
                  minus_RTDR_o1 (A, column_thread + k * m * m, d + k * m,
                                 get_bl (rows_p, k, J, m), m);
                  // fun (3);
#endif
                }
              DRtA (Revrsd, A, D, m);
              // printf ("drta I= %d, J=%d\n", I, J);
              // print_matrix (A, m);
              // memcpy (A, Temp, m * m * sizeof (double));
            }
        }
      size_t l;
      if ((n - N * m) && ((J == N)) && ((J) % commSize == rank))
        {
          l = n - N * m;
          // !double *A = &a[get_bl (I, J, u, b)];
          double *A = get_bl (rows_p, I, J, m, l);
          // !memcpy (Temp, A, m * l * sizeof (double));
          for (int k = 0; k < I; k++)
            {
              minus_RTDR_l_o1 (A, column_thread + k * m * m, d + k * m,
                               get_bl (rows_p, k, J, m, l), m, l);
            }
          DRtA_l (Revrsd, A, D, m, l);
          // !memcpy (A, Temp, m * l * sizeof (double));
        }
      // pthread_barrier_wait (bar);
    }
  size_t l;
  // !pthread_barrier_wait (bar);
  if ((l = n - N * m) && (I % commSize == rank))
    {
      double *A = get_bl (rows_p, I, I, m, l);
      // !memcpy (Temp, A, l * (l + 1) / 2 * sizeof (double));
      for (int k = 0; k < I; k++)
        {
          // !minus_RTDRu_l (Temp, &a[get_bl (k, I, u, b)], d + k * m, m, l);
          minus_RTDRu_l (A, get_bl (rows_p, k, I, m, l), d + k * m, m, l);
        }
      ret = std::max (cholesky_decomp_U (A, D, l, norma), ret);
      // !memcpy (d + I * m, D, l * sizeof (double));
      memcpy (d + I * m, D, l * sizeof (double));
      
      // MPI_Scatter (D, m, MPI_DOUBLE, d + I * m, m, MPI_DOUBLE, I / commSize,
      //              MPI_COMM_WORLD);
      // !memcpy (A, Temp, l * (l + 1) / 2 * sizeof (double));
    }
  if ((l = n - N * m))
    {
      MPI_Bcast (d + I * m, l, MPI_DOUBLE, I % commSize,
                 MPI_COMM_WORLD);
    }
  // printf ("total time spent on barrier %lf\n", totaldelta);
  return ret;
}

bool
cholesky_decomp_U (mat a, vec d, int n, double norma)
{
  for (int i = 0; i < n; i++)
    {
      // compute d_ii, r_ii:
      // double sum = a[i * n + i];
      double sum = a[get_elU (i, i, n)];
      for (int k = 0; k < i; k++)
        {
          sum -= a[get_elU (k, i, n)] * a[get_elU (k, i, n)] * d[k];
        }
      d[i] = sgn (sum);
      // printf("r %e\n", a[get_elU (i, i, n)]);
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