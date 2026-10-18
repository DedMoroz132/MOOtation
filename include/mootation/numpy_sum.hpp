#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Sums in NumPy's order, for the C++ halves of the Python run layer.
//
// The run layer (python/mootation/run/) computes its indicators, its archive's
// grid and its subset selection with NumPy, and the C++ that replaces those
// loops must give the same numbers bit for bit: a campaign's files must not
// change. Elementwise operations are the same IEEE operations in both; a sum
// is not. NumPy adds the n terms of np.sum, of np.mean and of a reduction along
// the last axis pairwise (numpy/_core/src/umath/loops_utils.h.src,
// pairwise_sum): fewer than 8 terms one after the other from 0; up to 128
// terms in eight running sums, term i going to sum i mod 8, which are then
// added as ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7)) before the terms
// left over by the multiple of eight, one by one; more than 128 terms in two
// halves, the first of a multiple of eight terms, each summed so. np.sum of a
// whole array is 0.0 plus that sum, and np.mean that sum over n. A loop that
// adds the terms in order agrees with it up to seven terms only: from eight
// objectives on, and for every mean over a reference front, the results differ
// in the last bits.
//
// Checked against np.sum (NumPy 2.5) for every n up to 300 and for lengths up
// to 20 000, and against sums along the last axis of 2-D and 3-D arrays: equal
// bit for bit (tests/test_run_layer_numerics.cpp and python/test_run_layer.py
// keep checks).
// ============================================================================

#include <cstddef>

namespace mootation::np {

// The pairwise sum of a[0], a[stride], ..., a[(n - 1) * stride].
inline double pairwise_sum(const double* a, std::size_t n, std::size_t stride = 1)
{
    if (n < 8) {
        double res = 0.0;
        for (std::size_t i = 0; i < n; ++i) res += a[i * stride];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (std::size_t j = 0; j < 8; ++j) r[j] = a[j * stride];
        std::size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (std::size_t j = 0; j < 8; ++j) r[j] += a[(i + j) * stride];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i * stride];
        return res;
    }
    std::size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_sum(a, n2, stride) + pairwise_sum(a + n2 * stride, n - n2, stride);
}

// np.sum(a) of a contiguous array of n values.
inline double sum(const double* a, std::size_t n) { return 0.0 + pairwise_sum(a, n); }

// np.mean(a) of a contiguous array of n > 0 values.
inline double mean(const double* a, std::size_t n)
{
    return sum(a, n) / static_cast<double>(n);
}

}   // namespace mootation::np
