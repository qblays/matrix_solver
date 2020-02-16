#include "init.h"
#include <memory>
size_t
compute_alloc_size (size_t n, size_t m)
{
  int rank, commSize;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
  int columns_n = n / m + (n % m > 0);
  // printf ("columns_n = %d\n", columns_n);
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
          // printf ("%d th row\n", i);
          sum += i * m * col_width;
          sum += (col_width * (col_width + 1)) / 2;
          // size_t t = i * m * col_width + (col_width * (col_width + 1)) / 2;
          // printf ("row %dth have size %lu\n", i, t);
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
          // printf ("init column %d\n", I);
          for (size_t i = 0; i < I * m; i++)
            {
              for (size_t j = I * m; j < I * m + col_width; j++)
                {
                  a[i * col_width + j - I * m] = f (i, j, n);
                  // printf ("%3lu ", i * col_width + j);
                }
              // printf ("\n");
            }
          a += I * m * col_width;
          for (size_t i = 0; i < col_width; i++)
            {
              for (size_t j = 0; j < i; j++)
                {
                  // a[get_elU (i, j, col_width)] = f (i, j, n);
                  // printf ("    ");
                }
              for (size_t j = i; j < col_width; j++)
                {
                  a[get_elU (i, j, col_width)] = f (i + I * m, j + I * m, n);
                  // printf ("f(%lu, %lu) = %lf ", i + I * m, j + I * m,
                  //         f (i + I * m, j + I * m, n));
                  // printf ("%3lu ", get_elU (i, j, col_width));
                }
              // printf ("\n");
            }
        }
    }
}

bool
read_string_of_doubles (FILE *fd, double *buf, const size_t n)
{
  for (size_t i = 0; i < n; i++)
    {
      int ret = fscanf (fd, "%lf", &buf[i]);
      if (ret != 1)
        {
          return 0;
        }
    }
  return 1;
}

void
test_bcast_root (int a)
{
  int rank, commSize;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
  assert (rank == 0);

  MPI_Bcast (&a, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void
test_bcast_others ()
{
  int rank, commSize;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
  assert (rank != 0);
  int a;
  MPI_Bcast (&a, 1, MPI_INT, 0, MPI_COMM_WORLD);
  printf ("%d: a = %d\n", rank, a);
}

bool
init_mat_file_root (double **rows_p, size_t n, size_t m, const char *filename)
{

  int rank, commSize;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
  assert (rank == 0);
  FILE *fd;
  fd = fopen (filename, "r");
  MPI_Bcast (&fd, 1, MPI_INT, 0, MPI_COMM_WORLD); // 1
  if (!fd)
    {
      printf ("cant open %s\n", filename);
      return 0;
    }
  int val = 1;
  double *buf = new double[n];
  auto exec_on_ret = [&] () {
    if (val == 0)
      printf ("unexpected eof of %s\n", filename);
    fclose (fd);
    delete[] buf;
  };

  int columns_n = n / m + (n % m > 0);
  int reminder = n - (n / m) * m;

  for (size_t i = 0; i < n; i++)
    {
      val = read_string_of_doubles (fd, buf, n);
      MPI_Bcast (&val, 1, MPI_INT, 0, MPI_COMM_WORLD); // 2
      if (!val)
        {
          exec_on_ret ();
          return 0;
        }
      // printf ("successfully read line %d:\n", i);
      // for (int k = i; k < n; k++)
      //   {
      //     printf ("%lf ", buf[k]);
      //   }
      // printf ("\n");
      for (int I = 0; I < columns_n; I++)
        {
          auto col_width = m;
          if (I == columns_n - 1 && reminder > 0)
            {
              col_width = reminder;
            }

          // printf ("init column %d\n", I);

          double *sendbuf;
          int sendcount;
          size_t offset;
          bool send_triangle = 0;
          if (i <= I * m)
            {
              offset = I * m;
              sendbuf = buf + offset;
              sendcount = col_width;
            }
          else if (i >= I * m + m)
            {
              offset = 0;
              sendbuf = nullptr;
              sendcount = 0;
            }
          else
            {
              offset = i;
              sendbuf = buf + offset;
              sendcount = col_width - i % m;
              send_triangle = 1;
            }
          double *a = rows_p[I];
          double *recvbuf;
          if (send_triangle)
            {
              a += col_width * (i / m) * m;
              recvbuf = &a[get_elU (i % m, i % m, col_width)];
            }
          else
            {
              recvbuf = a + col_width * i;
            }
          if (sendcount > 0)
            {
              if (I % commSize == 0)
                { // not send to root
                  // printf ("root copied buf\n");
                  memcpy (recvbuf, sendbuf, sendcount * sizeof (double));
                }
              else
                {
                  // printf ("root sended buf to %d\n", I % commSize);
                  MPI_Send (sendbuf, sendcount, MPI_DOUBLE, I % commSize, 0,
                            MPI_COMM_WORLD);
                }
            }
        }
    }
  exec_on_ret ();
  return 1;
}

bool
init_mat_file_others (double **rows_p, size_t n, size_t m, const char *filename)
{

  int rank, commSize;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
  assert (rank != 0);

  int ret;
  MPI_Bcast (&ret, 1, MPI_INT, 0, MPI_COMM_WORLD); // 1
  if (!ret)
    {
      printf ("cant open %s\n", filename);
      return 0;
    }

  int columns_n = n / m + (n % m > 0);
  int reminder = n - (n / m) * m;
  int val;

  for (size_t i = 0; i < n; i++)
    {
      MPI_Bcast (&val, 1, MPI_INT, 0, MPI_COMM_WORLD); // 2
      if (!val)
        {
          printf ("unexpected eof of %s\n", filename);
          return 0;
        }
      // printf ("successfully read line %d\n", i);
      for (int I = rank; I < columns_n; I += commSize)
        {
          auto col_width = m;
          if (I == columns_n - 1 && reminder > 0)
            {
              col_width = reminder;
            }

          // printf ("init column %d\n", I);

          // double *sendbuf;
          int sendcount;
          bool send_triangle = 0;
          if (i <= I * m)
            {
              sendcount = col_width;
            }
          else if (i >= I * m + m)
            {
              sendcount = 0;
            }
          else
            {
              sendcount = col_width - i % m;
              send_triangle = 1;
            }
          double *a = rows_p[I];
          double *recvbuf;
          if (send_triangle)
            {
              a += col_width * (i / m) * m;
              recvbuf = &a[get_elU (i % m, i % m, col_width)];
            }
          else
            {
              recvbuf = a + col_width * i;
            }
          if (sendcount > 0)
            {
              if (MPI_Recv (recvbuf, sendcount, MPI_DOUBLE, 0, 0,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE) == MPI_SUCCESS)
                {
                  // printf ("successfuly received message!\n");
                }
              else
                {
                  printf ("ERROR IN RECV\n");
                }
            }
        }
    }
  return 1;
}

bool
init_mat_file (double **rows_p, size_t n, size_t m, const char *filename)
{
  int rank, commSize;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
  if (rank == 0)
    return init_mat_file_root (rows_p, n, m, filename);
  else
    return init_mat_file_others (rows_p, n, m, filename);
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

void
print_mat_triangle (double **rows_p, size_t n, size_t m)
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
          // print only last triangle
          if (I + commSize >= columns_n)
            {
              double *a = rows_p[I];
              a += I * m * col_width;
              printf ("printing triangle\n");
              for (size_t i = 0; i < col_width; i++)
                {
                  for (size_t j = 0; j < i; j++)
                    {
                      printf ("%4.6lf ", 0.);
                    }
                  for (size_t j = i; j < col_width; j++)
                    {
                      printf ("%4.6lf ", a[get_elU (i, j, col_width)]);
                    }
                  printf ("\n");
                }
            }
        }
    }
}

bool
check_args (const int argc, char **argv)
{
  if (argc > 1 && argc < 5)
    {
      if (atoi (argv[1]) != 0 && atoi (argv[2]) != 0)
        {
          return 1;
        }
    }
  return 0;
}

void
gather_row_slow (size_t i, size_t n, size_t m, int root, vec buf,
                 double **&rows_p)
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
      double *sendbuf;
      int sendcount;
      size_t offset;
      // bool send_triangle = 0;
      double *a = rows_p[I];
      double *recvbuf;
      if (i <= I * m)
        { // send full
          offset = I * m;
          sendbuf = a + col_width * i;
          sendcount = col_width;
        }
      else if (i >= I * m + m)
        { // send nothing
          offset = 0;
          sendbuf = nullptr;
          sendcount = 0;
        }
      else
        { // send from trian
          offset = i;
          auto aa = a + col_width * (i / m) * m;
          sendbuf = &aa[get_elU (i % m, i % m, col_width)];
          sendcount = col_width - i % m;
          // send_triangle = 1;
        }

      recvbuf = buf + offset;

      if (sendcount > 0)
        {
          if (rank == root)
            { // not send to root
              // printf ("root copied buf\n");
              if (I % commSize == rank)
                memcpy (recvbuf, sendbuf, sendcount * sizeof (double));
              else
                {
                  MPI_Recv (recvbuf, sendcount, MPI_DOUBLE, I % commSize, 0,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
          else
            {
              if (I % commSize == rank)
                {
                  // printf ("%d sended buf to root\n", I % commSize);
                  MPI_Send (sendbuf, sendcount, MPI_DOUBLE, root, 0,
                            MPI_COMM_WORLD);
                }
            }
        }
    }
}

void
gather_row (size_t i, size_t n, size_t m, int root, vec buf, double **rows_p,
            int action)
{
  static int commSize;
  static int idx = 0;
  static int *recvcounts;
  static int *recvcounts2;
  static int *displs;
  static double *bbuf;
  static double *gathered;
  if (idx == 0)
    {
      idx = 1;
      MPI_Comm_size (MPI_COMM_WORLD, &commSize);
      recvcounts = new int[commSize];
      recvcounts2 = new int[commSize];
      displs = new int[commSize];
      bbuf = new double[n];
      gathered = new double[n];
    }
  if (action == 0)
    gather_row_runner (i, n, m, root, buf, rows_p, recvcounts, recvcounts2,
                       displs, bbuf, gathered);
  else
    {
      delete[] recvcounts;
      delete[] recvcounts2;
      delete[] displs;
      delete[] bbuf;
      delete[] gathered;
    }
}

void
gather_row_runner (size_t i, size_t n, size_t m, int root, vec buf,
                   double **&rows_p, int *recvcounts, int *recvcounts2,
                   int *displs, double *bbuf, double *gathered)
{
  int rank, commSize;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
  int columns_n = n / m + (n % m > 0);
  int reminder = n - (n / m) * m;
  size_t bbuf_size = 0;
  for (int i = 0; i < commSize; i++)
    {
      recvcounts[i] = 0;
      recvcounts2[i] = 0;
    }
  for (int I = 0; I < columns_n; I++)
    {
      auto col_width = m;
      if (I == columns_n - 1 && reminder > 0)
        {
          col_width = reminder;
        }
      double *sendbuf;
      int sendcount;
      // bool send_triangle = 0;
      double *a = rows_p[I];
      if (i <= I * m)
        { // send full
          sendbuf = a + col_width * i;
          sendcount = col_width;
        }
      else if (i >= I * m + m)
        { // send nothing
          sendbuf = nullptr;
          sendcount = 0;
        }
      else
        { // send from trian
          auto aa = a + col_width * (i / m) * m;
          sendbuf = &aa[get_elU (i % m, i % m, col_width)];
          sendcount = col_width - i % m;
          // send_triangle = 1;
        }
      recvcounts[I % commSize] += sendcount;

      if (I % commSize == rank && sendcount > 0)
        {
          memcpy (bbuf + bbuf_size, sendbuf, sendcount * sizeof (double));
          bbuf_size += sendcount;
          assert (recvcounts[I % commSize] == (int)bbuf_size);
        }
    }
  displs[0] = 0;
  for (int i = 1; i < commSize; i++)
    {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
  // for (int i = 0; i < commSize; i++)
  //   {
  //     printf ("recvcounts[%d] = %d\ndispls[%d] = %d\n", i, recvcounts[i], i,
  //             displs[i]);
  //   }
  MPI_Gatherv (bbuf, bbuf_size, MPI_DOUBLE, gathered, recvcounts, displs,
               MPI_DOUBLE, root, MPI_COMM_WORLD);
  if (rank == root)
    {
      for (int I = 0; I < columns_n; I++)
        {
          auto col_width = m;
          if (I == columns_n - 1 && reminder > 0)
            {
              col_width = reminder;
            }
          double *sendbuf;
          int sendcount;
          size_t offset;
          // bool send_triangle = 0;
          double *a = &gathered[displs[I % commSize]];
          double *recvbuf;
          if (i <= I * m)
            { // send full
              offset = I * m;
              sendbuf = a + recvcounts2[I % commSize];
              sendcount = col_width;
            }
          else if (i >= I * m + m)
            { // send nothing
              offset = 0;
              sendbuf = nullptr;
              sendcount = 0;
            }
          else
            { // send from trian
              offset = i;
              sendbuf = a + recvcounts2[I % commSize];
              sendcount = col_width - i % m;
            }
          recvcounts2[I % commSize] += sendcount;
          recvbuf = buf + offset;

          if (sendcount > 0)
            {
              memcpy (recvbuf, sendbuf, sendcount * sizeof (double));
            }
        }
    }
}

void
gather_col (size_t j, size_t n, size_t m, int root, vec buf, double **&rows_p)
{
  int rank, commSize;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
  int columns_n = n / m + (n % m > 0);
  int reminder = n - (n / m) * m;
  int I = j / m;
  if (I % commSize == rank)
    {
      auto col = new double[j + 1];
      auto col_width = m;
      if (I == columns_n - 1 && reminder > 0)
        {
          col_width = reminder;
        }
      double *sendbuf = col;
      // bool send_triangle = 0;
      double *a = rows_p[I];
      for (size_t i = 0; i < I * m; i++)
        {
          col[i] = a[i * col_width + j % m];
        }
      a += I * m * col_width;
      for (size_t i = I * m; i < j + 1; i++)
        {
          col[i] = a[get_elU (i - I * m, j - I * m, col_width)];
        }
      if (rank == root)
        {
          memcpy (buf, sendbuf, (j + 1) * sizeof (double));
        }
      else
        {
          MPI_Send (sendbuf, j + 1, MPI_DOUBLE, root, 0, MPI_COMM_WORLD);
        }

      delete[] col;
    }
  if (rank == root && (I % commSize != rank))
    {
      MPI_Recv (buf, j + 1, MPI_DOUBLE, I % commSize, 0, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
    }
}

void
init_b_get_norm (vec b, size_t n, size_t m, int root, double **&rows_p,
                 double &norm)
{
  double *row = nullptr;
  int rank, commSize;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
  if (rank == root)
    {
      row = new double[n];
    }
  double max = 0;
  for (size_t i = 0; i < n; i++)
    {
      gather_row (i, n, m, root, row, rows_p);
      gather_col (i, n, m, root, row, rows_p);
      if (rank == root)
        {
          double sum = 0;
          double sum_norm = 0;
          for (size_t k = 0; k < n; k++)
            {
              if (k % 2 == 0)
                {
                  sum += row[k];
                }
              sum_norm += std::abs (row[k]);
            }
          b[i] = sum;
          max = sum_norm > max ? sum_norm : max;
        }
    }
  norm = max;
  MPI_Bcast (&norm, 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
  if (rank == root)
    {
      delete[] row;
    }
}

void
find_disrep_vec (vec b, vec x, vec r, size_t n, size_t m, int root,
                 double **&rows_p)
{
  double *row = nullptr;
  int rank, commSize;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
  if (rank == root)
    {
      row = new double[n];
    }
  for (size_t i = 0; i < n; i++)
    {
      gather_row (i, n, m, root, row, rows_p);
      gather_col (i, n, m, root, row, rows_p);
      if (rank == root)
        {
          double sum = 0;
          for (size_t k = 0; k < n; k++)
            {
              sum += row[k] * x[k];
            }
          r[i] = sum - b[i];
        }
    }
  if (rank == root)
    {
      delete[] row;
    }
}

bool
check_res (int root, int loc_res)
{
  static int e = 0;
  int rank, commSize;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);
  int all_res = 0;
  MPI_Allreduce (&loc_res, &all_res, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (all_res > 0)
    {
      e++;
      if (rank == root)
        printf ("Cant apply cholesky decomposition er%d\n", e);
      return 1;
    }
  return 0;
}