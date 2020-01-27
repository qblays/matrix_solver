#include "matrix_op.h"

void
minus_RTDR_o1 (mat __restrict A, mat __restrict R1, vec __restrict D,
               mat __restrict R2, int n)
{
  for (int i = 0; i < n; i++)
    {
      int j = 0;
      for (j = 0; j < n - 1; j += 2)
        {
          double sum1 = 0;
          double sum2 = 0;
          // double sum3 = 0;
          // double sum4 = 0;
          for (int k = 0; k < n; k++)
            {
              sum1 += R1[k * n + i] * R2[k * n + j] * D[k];
              sum2 += R1[k * n + i] * R2[k * n + j + 1] * D[k];
              // sum3 += R1[k * n + i] * R2[k * n + j + 2] * D[k];
              // sum4 += R1[k * n + i] * R2[k * n + j + 3] * D[k];
            }
          A[i * n + j] -= sum1;
          A[i * n + j + 1] -= sum2;
          // A[i * n + j + 2] -= sum3;
          // A[i * n + j + 3] -= sum4;
        }
      for (; j < n - 0; j += 1)
        {
          double sum1 = 0;
          for (int k = 0; k < n; k++)
            {
              sum1 += R1[k * n + i] * R2[k * n + j] * D[k];
            }
          A[i * n + j] -= sum1;
        }
    }
}

void
minus_RTDRu_ofast (mat A, mat R1, vec D, mat R2, int n)
{
  for (int i = 0; i < n; i += 1)
    {
      int j = 0;
      for (j = i; j < n - 15; j += 16)
        {
          double sums[16] = {0};
          for (int k = 0; k < n; k++)
            {
              sums[0] += R1[k * n + i] * R2[k * n + j + 0] * D[k];
              sums[1] += R1[k * n + i] * R2[k * n + j + 1] * D[k];
              sums[2] += R1[k * n + i] * R2[k * n + j + 2] * D[k];
              sums[3] += R1[k * n + i] * R2[k * n + j + 3] * D[k];
              sums[4] += R1[k * n + i] * R2[k * n + j + 4] * D[k];
              sums[5] += R1[k * n + i] * R2[k * n + j + 5] * D[k];
              sums[6] += R1[k * n + i] * R2[k * n + j + 6] * D[k];
              sums[7] += R1[k * n + i] * R2[k * n + j + 7] * D[k];
              sums[8] += R1[k * n + i] * R2[k * n + j + 8] * D[k];
              sums[9] += R1[k * n + i] * R2[k * n + j + 9] * D[k];
              sums[10] += R1[k * n + i] * R2[k * n + j + 10] * D[k];
              sums[11] += R1[k * n + i] * R2[k * n + j + 11] * D[k];
              sums[12] += R1[k * n + i] * R2[k * n + j + 12] * D[k];
              sums[13] += R1[k * n + i] * R2[k * n + j + 13] * D[k];
              sums[14] += R1[k * n + i] * R2[k * n + j + 14] * D[k];
              sums[15] += R1[k * n + i] * R2[k * n + j + 15] * D[k];
            }
          A[get_elU (i, j + 0, n)] -= sums[0];
          A[get_elU (i, j + 1, n)] -= sums[1];
          A[get_elU (i, j + 2, n)] -= sums[2];
          A[get_elU (i, j + 3, n)] -= sums[3];
          A[get_elU (i, j + 4, n)] -= sums[4];
          A[get_elU (i, j + 5, n)] -= sums[5];
          A[get_elU (i, j + 6, n)] -= sums[6];
          A[get_elU (i, j + 7, n)] -= sums[7];
          A[get_elU (i, j + 8, n)] -= sums[8];
          A[get_elU (i, j + 9, n)] -= sums[9];
          A[get_elU (i, j + 10, n)] -= sums[10];
          A[get_elU (i, j + 11, n)] -= sums[11];
          A[get_elU (i, j + 12, n)] -= sums[12];
          A[get_elU (i, j + 13, n)] -= sums[13];
          A[get_elU (i, j + 14, n)] -= sums[14];
          A[get_elU (i, j + 15, n)] -= sums[15];
        }
      for (; j < n; j++)
        {
          double sum = 0;
          for (int k = 0; k < n; k++)
            {
              // opt??
              // sum += R1[get_elU (k, i, n)] * R1[get_elU (k, j, n)] * D[k];
              sum += R1[k * n + i] * R1[k * n + j] * D[k];
              // sum += R1[k * n + i] * R1[k * n + j] * D[k];
            }
          A[get_elU (i, j, n)] -= sum;
        }
    }
}

#ifdef AVX
void
minus_RTDR_o1_avx_ggggg (mat A, mat R1, vec D, mat R2, int n)
{
  for (int i = 0; i < n - 2; i += 3)
    {
      int j;

      for (j = 0; j < n - 11; j += 12)
        {
          auto s1 = _mm256_loadu_pd (&A[i * n + j]);
          auto s2 = _mm256_loadu_pd (&A[i * n + j + 4]);
          auto s2_3 = _mm256_loadu_pd (&A[i * n + j + 8]);
          auto s3 = _mm256_loadu_pd (&A[(i + 1) * n + j]);
          auto s4 = _mm256_loadu_pd (&A[(i + 1) * n + j + 4]);
          auto s4_3 = _mm256_loadu_pd (&A[(i + 1) * n + j + 8]);
          auto s5 = _mm256_loadu_pd (&A[(i + 2) * n + j]);
          auto s6 = _mm256_loadu_pd (&A[(i + 2) * n + j + 4]);
          auto s6_3 = _mm256_loadu_pd (&A[(i + 2) * n + j + 8]);
          for (int k = 0; k < n; k++)
            {
              // double coef1 = R1[k * n + i] * D[k];
              // auto c1 = _mm256_broadcast_sd (&coef1);
              // double coef2 = R1[k * n + i + 1] * D[k];
              // auto c2 = _mm256_broadcast_sd (&coef2);
              // double coef3 = R1[k * n + i + 2] * D[k];
              // auto c3 = _mm256_broadcast_sd (&coef3);
              auto Dk = _mm256_broadcast_sd (&D[k]);

              // double coef1 = R1[k * n + i] * D[k];
              auto c1 = _mm256_broadcast_sd (&R1[k * n + i]);
              // double coef2 = R1[k * n + i + 1] * D[k];
              auto c2 = _mm256_broadcast_sd (&R1[k * n + i + 1]);
              // double coef3 = R1[k * n + i + 2] * D[k];
              auto c3 = _mm256_broadcast_sd (&R1[k * n + i + 2]);
              c1 = _mm256_mul_pd (c1, Dk);
              c2 = _mm256_mul_pd (c2, Dk);
              c2 = _mm256_mul_pd (c2, Dk);
              auto c2_1 = _mm256_loadu_pd (&R2[k * n + j]);
              auto c2_2 = _mm256_loadu_pd (&R2[k * n + j + 4]);
              auto c2_3 = _mm256_loadu_pd (&R2[k * n + j + 8]);
              s1 = _mm256_sub_pd (s1, _mm256_mul_pd (c1, c2_1));
              s2 = _mm256_sub_pd (s2, _mm256_mul_pd (c1, c2_2));
              s2_3 = _mm256_sub_pd (s2_3, _mm256_mul_pd (c1, c2_3));
              s3 = _mm256_sub_pd (s3, _mm256_mul_pd (c2, c2_1));
              s4 = _mm256_sub_pd (s4, _mm256_mul_pd (c2, c2_2));
              s4_3 = _mm256_sub_pd (s4_3, _mm256_mul_pd (c2, c2_3));
              s5 = _mm256_sub_pd (s5, _mm256_mul_pd (c3, c2_1));
              s6 = _mm256_sub_pd (s6, _mm256_mul_pd (c3, c2_2));
              s6_3 = _mm256_sub_pd (s6_3, _mm256_mul_pd (c3, c2_3));
            }
          _mm256_storeu_pd (&A[i * n + j], s1);
          _mm256_storeu_pd (&A[i * n + j + 4], s2);
          _mm256_storeu_pd (&A[i * n + j + 8], s2_3);
          _mm256_storeu_pd (&A[(i + 1) * n + j], s3);
          _mm256_storeu_pd (&A[(i + 1) * n + j + 4], s4);
          _mm256_storeu_pd (&A[(i + 1) * n + j + 8], s4_3);
          _mm256_storeu_pd (&A[(i + 2) * n + j], s5);
          _mm256_storeu_pd (&A[(i + 2) * n + j + 4], s6);
          _mm256_storeu_pd (&A[(i + 2) * n + j + 8], s6_3);
        }
    }
}

void
minus_RTDRu_o1_avx_ggggg (mat A, mat R1, vec D, mat R2, int n)
{
  for (int i = 0; i < n - 0; i += 1)
    {
      int j;

      for (j = i; j < n - 15; j += 16)
        {
          auto s1 = _mm256_loadu_pd (&A[get_elU (i, j, n)]);
          auto s2 = _mm256_loadu_pd (&A[get_elU (i, j + 4, n)]);
          auto s2_3 = _mm256_loadu_pd (&A[get_elU (i, j + 8, n)]);
          auto s2_4 = _mm256_loadu_pd (&A[get_elU (i, j + 12, n)]);
          for (int k = 0; k < n; k++)
            {
              auto Dk = _mm256_broadcast_sd (&D[k]);
              auto c1 = _mm256_broadcast_sd (&R1[k * n + i]);
              c1 = _mm256_mul_pd (c1, Dk);
              auto c2_1 = _mm256_loadu_pd (&R2[k * n + j]);
              auto c2_2 = _mm256_loadu_pd (&R2[k * n + j + 4]);
              auto c2_3 = _mm256_loadu_pd (&R2[k * n + j + 8]);
              auto c2_4 = _mm256_loadu_pd (&R2[k * n + j + 12]);
              s1 = _mm256_sub_pd (s1, _mm256_mul_pd (c1, c2_1));
              s2 = _mm256_sub_pd (s2, _mm256_mul_pd (c1, c2_2));
              s2_3 = _mm256_sub_pd (s2_3, _mm256_mul_pd (c1, c2_3));
              s2_4 = _mm256_sub_pd (s2_4, _mm256_mul_pd (c1, c2_4));
            }
          _mm256_storeu_pd (&A[get_elU (i, j, n)], s1);
          _mm256_storeu_pd (&A[get_elU (i, j + 4, n)], s2);
          _mm256_storeu_pd (&A[get_elU (i, j + 8, n)], s2_3);
          _mm256_storeu_pd (&A[get_elU (i, j + 12, n)], s2_4);
        }
      for (; j < n; j++)
        {
          double sum = 0;
          for (int k = 0; k < n; k++)
            {
              sum += R1[k * n + i] * R1[k * n + j] * D[k];
            }
          A[get_elU (i, j, n)] -= sum;
        }
    }
}
#endif

void
minus_RTDR_l_o1 (mat A, mat R1, vec D, mat R2, int n, int l)
{
  for (int i = 0; i < n; i++)
    {
      int j = 0;
      for (j = 0; j < l - 3; j += 4)
        {
          double sum1 = 0;
          double sum2 = 0;
          double sum3 = 0;
          double sum4 = 0;
          for (int k = 0; k < n; k++)
            {
              // opt??
              sum1 += R1[k * n + i] * R2[k * l + j] * D[k];
              sum2 += R1[k * n + i] * R2[k * l + j + 1] * D[k];
              sum3 += R1[k * n + i] * R2[k * l + j + 2] * D[k];
              sum4 += R1[k * n + i] * R2[k * l + j + 3] * D[k];
            }
          A[i * l + j] -= sum1;
          A[i * l + j + 1] -= sum2;
          A[i * l + j + 2] -= sum3;
          A[i * l + j + 3] -= sum4;
        }
      if (j == l - 3)
        {
          double sum1 = 0;
          double sum2 = 0;
          double sum3 = 0;
          for (int k = 0; k < n; k++)
            {
              // opt??
              sum1 += R1[k * n + i] * R2[k * l + j] * D[k];
              sum2 += R1[k * n + i] * R2[k * l + j + 1] * D[k];
              sum3 += R1[k * n + i] * R2[k * l + j + 2] * D[k];
            }
          A[i * l + j] -= sum1;
          A[i * l + j + 1] -= sum2;
          A[i * l + j + 2] -= sum3;
          j += 3;
        }
      else if (j == l - 2)
        {
          double sum1 = 0;
          double sum2 = 0;
          for (int k = 0; k < n; k++)
            {
              sum1 += R1[k * n + i] * R2[k * l + j] * D[k];
              sum2 += R1[k * n + i] * R2[k * l + j + 1] * D[k];
            }
          A[i * l + j] -= sum1;
          A[i * l + j + 1] -= sum2;
          j += 2;
        }
      else if (j == l - 1)
        {
          double sum1 = 0;
          for (int k = 0; k < n; k++)
            {
              sum1 += R1[k * n + i] * R2[k * l + j] * D[k];
            }
          A[i * l + j] -= sum1;
          j++;
        }
    }
}

void
minus_RTDRu_l (mat A, mat R1, vec D, int n, int l)
{
  for (int i = 0; i < l; i++)
    {
      for (int j = i; j < l; j++)
        {
          double sum = 0;
          for (int k = 0; k < n; k++)
            {
              // opt??
              // sum += R1[get_elU (k, i, n)] * R1[get_elU (k, j, n)] * D[k];
              sum += R1[k * l + i] * R1[k * l + j] * D[k];
              // sum += R1[k * n + i] * R1[k * n + j] * D[k];
            }
          A[get_elU (i, j, l)] -= sum;
        }
    }
}

bool
reverse_upper (mat A, mat B, int n, double norma)
{

  for (int i = n - 1; i >= 0; i--)
    {
      B[get_elU (i, i, n)] = 1 / A[get_elU (i, i, n)];
      int j;
      if (is_double_too_small (A[get_elU (i, i, n)], EPS * norma))
        {
          return 1;
        }
      for (j = i + 1; j < n - 1; j += 2)
        {
          double sum1 = 0;
          double sum2 = 0;
          int k = i + 1;
          for (k = i + 1; k <= j; k++)
            {
              sum1 += A[get_elU (i, k, n)] * B[get_elU (k, j, n)];
              sum2 += A[get_elU (i, k, n)] * B[get_elU (k, j + 1, n)];
            }

          sum2 += A[get_elU (i, k, n)] * B[get_elU (k, j + 1, n)];

          B[get_elU (i, j, n)] = -sum1 / A[get_elU (i, i, n)];
          B[get_elU (i, j + 1, n)] = -sum2 / A[get_elU (i, i, n)];
        }
      if (j == n - 1)
        {
          double sum1 = 0;
          for (int k = i + 1; k <= j; k++)
            {
              sum1 += A[get_elU (i, k, n)] * B[get_elU (k, j, n)];
            }
          B[get_elU (i, j, n)] = -sum1 / A[get_elU (i, i, n)];
          j++;
        }
    }
  return 0;
}

void
DRtA (mat R, mat A, vec D, int n)
{

  for (int i = n - 1; i >= 0; i--)
    {
      int j;
      for (j = 0; j < n - 3; j += 4)
        {
          double sum[4] = {0};

          for (int k = 0; k <= i; k++)
            {
              sum[0] += R[get_elU (k, i, n)] * A[k * n + j];
              sum[1] += R[get_elU (k, i, n)] * A[k * n + j + 1];
              sum[2] += R[get_elU (k, i, n)] * A[k * n + j + 2];
              sum[3] += R[get_elU (k, i, n)] * A[k * n + j + 3];
            }
          A[i * n + j] = sum[0] * D[i];
          A[i * n + j + 1] = sum[1] * D[i];
          A[i * n + j + 2] = sum[2] * D[i];
          A[i * n + j + 3] = sum[3] * D[i];
        }
      for (; j < n - 0; j += 1)
        {
          double sum[1] = {0};

          for (int k = 0; k <= i; k++)
            {
              sum[0] += R[get_elU (k, i, n)] * A[k * n + j];
            }
          A[i * n + j] = sum[0] * D[i];
        }
    }
}

void
DRtA_l (mat R, mat A, vec D, int n, int l)
{
  for (int i = n - 1; i >= 0; i--)
    {
      int j;
      for (j = 0; j < l - 2; j += 3)
        {
          double sum1 = 0;
          double sum2 = 0;
          double sum3 = 0;
          for (int k = 0; k <= i; k++)
            {
              sum1 += R[get_elU (k, i, n)] * A[k * l + j];
              sum2 += R[get_elU (k, i, n)] * A[k * l + j + 1];
              sum3 += R[get_elU (k, i, n)] * A[k * l + j + 2];
            }
          A[i * l + j] = sum1 * D[i];
          A[i * l + j + 1] = sum2 * D[i];
          A[i * l + j + 2] = sum3 * D[i];
        }
      if (j == l - 2)
        {
          double sum1 = 0;
          double sum2 = 0;
          for (int k = 0; k <= i; k++)
            {
              sum1 += R[get_elU (k, i, n)] * A[k * l + j];
              sum2 += R[get_elU (k, i, n)] * A[k * l + j + 1];
            }
          A[i * l + j] = sum1 * D[i];
          A[i * l + j + 1] = sum2 * D[i];
          j += 2;
        }
      else if (j == l - 1)
        {
          double sum = 0;
          for (int k = 0; k <= i; k++)
            {
              sum += R[get_elU (k, i, n)] * A[k * l + j];
            }
          A[i * l + j] = sum * D[i];
          j++;
        }
    }
}

void
print_matrix (double *a, size_t n)
{
  size_t i, j, N = n, l = 0;
  if (n > MAX)
    {
      N = MAX;
      l = 1;
    }
  for (i = 0; i < N; i++)
    {
      for (j = 0; j < N; j++)
        {
          printf ("%*.2lf ", 6, a[i * n + j]);
        }
      if (l == 1)
        printf ("\t...");
      printf ("\n");
    }
  if (l)
    {
      printf ("\t...\t...\t...\n");
      for (i = n - N; i < n; i++)
        {
          printf ("\t...");
          for (j = n - N; j < n; j++)
            {
              printf ("%*.2lf ", 6, a[i * n + j]);
            }
          printf ("\n");
        }
    }
}

void
print_matrix_b_upper (double *a, int n)
{
  int i, j, N = n, l = 0;
  if (n > MAX)
    {
      N = MAX;
      l = 1;
    }
  for (i = 0; i < N; i++)
    {
      for (j = 0; j < N; j++)
        {
          i <= j ? printf ("%e ", a[get_elU (i, j, n)]) : printf ("%e ", 0.0);
          /*i <= j ? printf ("%*.2lf ", 6, a[get_el (i, j, u, b, n)])
                 : printf ("%*.2lf ", 6, 0.0);*/
        }
      if (l == 1)
        printf ("\t...");
      printf ("\n");
    }
  if (l)
    {
      printf ("\t...\t...\t...\n");
      for (i = n - N; i < n; i++)
        {
          printf ("\t...");
          for (j = n - N; j < n; j++)
            {
              i <= j ? printf ("%e ", a[get_elU (i, j, n)])
                     : printf ("%e ", 0.0);
              /*i <= j ? printf ("%*.2lf ", 6, a[get_el (i, j, u, b, n)])
                     : printf ("%*.2lf ", 6, 0.0);*/
            }
          printf ("\n");
        }
    }
}