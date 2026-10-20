#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The benchmark families the run layer evaluates in C++ (task 11 of
// 2026-10-18, §1): ZDT, DTLZ and its variants, C-DTLZ, MaF and ZCAT, and the
// shifts and rotations over them. python/mootation/benchmarks hands a problem's
// parameters to the binding (benchmarks/native.py), and its evaluate,
// evaluate_batch and constraints_batch, and a run that evaluates without
// crossing into Python, call this code.
//
// NUMPY'S ORDER. The values follow the NumPy code of benchmarks/ (it stays
// as the reference in the tests, and is what runs without the extension), so
// every expression repeats the order of its NumPy counterpart:
//   * a sum along a row (np.sum(..., axis=1), np.mean) is NumPy's pairwise sum
//     of the row from 0.0 (numpy_sum.hpp);
//   * a product along a row (np.prod(..., axis=1)) is sequential, left to
//     right;
//   * x ** 2 of an array is x * x, ** 0.5 its sqrt; every other power, and
//     sin, cos, exp, log and hypot, are the library's own (dm::,
//     detmath.hpp; task 14 of 2026-10-20, supplement 1, D1), the same bits on
//     every platform. NumPy calls the C library instead, so the two agree to
//     a few ulps, not to the bit (the tests compare them with a tolerance);
//   * a Python expression a * b / c is (a * b) / c and 2 * pi * x is
//     (2 * pi) * x: each line below keeps the grouping of the one it follows;
//   * np.clip and np.minimum as NumPy's scalar loops decide (clip, nan_min).
// What NumPy draws or factorises (the rotations, the shift centres, MaF9's
// polygons) and the constants Python computes once are handed over: C++ does
// the arithmetic only.
//
// Built without contraction into fused multiply-adds (-ffp-contract=off,
// /fp:precise; python/CMakeLists.txt): a fused a * b + c rounds once where
// NumPy rounds twice. Problem version 2 (task 14): the values of version 1
// were the C library's of the machine.
// ============================================================================

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "../numpy_sum.hpp"
#include "native.hpp"

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace mootation::suite {

constexpr double kPi = 3.141592653589793;           // math.pi
constexpr double kHalfPi = kPi / 2.0;               // zcat.HALF_PI
constexpr double kE = 2.718281828459045;            // math.e

// np.sum of n values in a row (0.0 plus the pairwise sum).
inline double row_sum(const double* a, std::size_t n) { return np::sum(a, n); }
inline double row_sum(const std::vector<double>& a) { return np::sum(a.data(), a.size()); }

// np.prod of n values in a row: left to right; 1.0 of none.
inline double row_prod(const double* a, std::size_t n)
{
    if (n == 0) return 1.0;
    double r = a[0];
    for (std::size_t i = 1; i < n; ++i) r = r * a[i];
    return r;
}

// np.clip(v, lo, hi): NumPy's min(max(v, lo), hi) with its comparisons (a
// NaN stays NaN; -0.0 against a bound of 0.0 gives the bound).
inline double clip(double v, double lo, double hi)
{
    const double a = std::isnan(v) ? v : (v > lo ? v : lo);
    return std::isnan(a) ? a : (a < hi ? a : hi);
}

// np.minimum(a, b) and a step of .min(): a NaN wins.
inline double nan_min(double a, double b)
{
    return (a <= b || std::isnan(a)) ? a : b;
}

// np.mod(a, b) of floats (npy_divmod): the remainder with the sign of b.
inline double np_mod(double a, double b)
{
    double m = std::fmod(a, b);
    if (b == 0.0) return m;
    if (m != 0.0) {
        if ((b < 0.0) != (m < 0.0)) m += b;
    } else {
        m = std::copysign(0.0, b);
    }
    return m;
}

// A x of a k x k row-major matrix A, each row the pairwise sum of its
// products: np.sum(A * x, axis=1) (benchmarks/_rows.py, matvec_pairwise).
inline void matvec(const std::vector<double>& A, std::size_t k, const double* x, double* out)
{
    std::vector<double> p(k);
    for (std::size_t i = 0; i < k; ++i) {
        const double* a = A.data() + i * k;
        for (std::size_t j = 0; j < k; ++j) p[j] = a[j] * x[j];
        out[i] = row_sum(p);
    }
}

}   // namespace mootation::suite
