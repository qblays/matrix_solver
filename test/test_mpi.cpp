#include <iostream>
#include <mpi.h>
#include <sys/time.h>
using namespace std;
const int Tag = 0;
const int root = 0;
double
sum_array (double *array, int n)
{
  double sum = 0;
  for (int i = 0; i < 100000; ++i)
    {
      for (int j = 0; j < n; ++j)
        {
          sum += array[j];
        }
      // sum += array[i];
    }
  return sum;
}
int
main ()
{
  int rank, commSize;

  MPI_Init (NULL, NULL);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &commSize);

  double *arr, sum = 0, buffer;
  int n;
  MPI_Status status;

  if (root == rank)
    {
      int k;
      timeval t1, t2;
      cout << "n : ";
      cin >> n;

      gettimeofday (&t1, nullptr);
      arr = new double[n];
      for (int i = 0; i < n; ++i)
        arr[i] = i;

      int partSize = n / commSize;

      int shift = n % commSize;
      MPI_Request request;
      for (int i = root + 1; i < commSize; ++i)
        {
          MPI_Isend (arr + shift + partSize * i, partSize, MPI_DOUBLE, i, Tag,
                     MPI_COMM_WORLD, &request);
          MPI_Request_free (&request);
        }

      sum = sum_array (arr, shift + partSize);

      for (int i = root + 1; i < commSize; ++i)
        {
          MPI_Recv (&buffer, 1, MPI_DOUBLE, i, Tag, MPI_COMM_WORLD, &status);
          sum += buffer;
        }
      gettimeofday (&t2, nullptr);
      printf ("elapsed = %lf, res = %lf\n",
              t2.tv_sec - t1.tv_sec + 1e-6 * (t2.tv_usec - t1.tv_usec), sum);
    }
  else
    {
      MPI_Probe (root, Tag, MPI_COMM_WORLD, &status);
      MPI_Get_count (&status, MPI_DOUBLE, &n);

      arr = new double[n];
      MPI_Recv (arr, n, MPI_DOUBLE, root, Tag, MPI_COMM_WORLD, &status);

      sum = sum_array (arr, n);

      MPI_Send (&sum, 1, MPI_DOUBLE, root, Tag, MPI_COMM_WORLD);
    }

  delete[] arr;

  cout << rank << " : " << sum << endl;
  MPI_Finalize ();
}