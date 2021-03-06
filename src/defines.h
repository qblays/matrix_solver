#ifndef DEFINES_H
#define DEFINES_H
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
using mat = double *;
using vec = double *;
constexpr double EPS = 1.e-14;
constexpr int MAX = 8;

inline size_t
get_elU (size_t i, size_t j, size_t n)
{
  size_t d = i;
  d = (d * d + d) / 2;
  return i * n + j - d;
}

inline bool
is_double_too_small (double a, double eps)
{
  return fabs (a) < eps;
}

inline int
sgn (double a)
{
  return a > 0 ? 1 : -1;
}

inline void
assert_and_do (bool x, std::function<void ()> fn)
{
  if (!x)
    {
      fn ();
      assert (x); // always true
    }
}
#endif