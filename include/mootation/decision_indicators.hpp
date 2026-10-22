#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The decision-space indicators of python/mootation/run/metrics.py in C++
// (task 17, supplement 1, stage 2a): igdx, ps_dmin and pairwise_distance,
// value for value. The variables are normalised by the bounds as _unit does,
// (x - lo) / (hi - lo) with a width of 1 where hi <= lo; a difference of two
// such points, B - A, wraps a cyclic variable as np.mod(d + 0.5, 1) - 0.5; a
// distance is the root of NumPy's sum of the squares (numpy_sum.hpp). The
// chunks of 128 rows metrics.py takes are kept where they change the result:
// pairwise_distance adds one NumPy sum per chunk, and ps_dmin takes Python's
// min over the chunks' NumPy minima (a NaN chunk left out, as min(best, nan)
// keeps best). metrics.py keeps its NumPy code as the reference.
//
// spread_select, the second answer's choice (archive.spread_select; task 17,
// supplement 1, §3): Tanabe & Ishibuchi's farthest-point selection from a
// first point, every step archive.py's NumPy operations. The first point is
// drawn by spread_first from a std::mt19937 of its own, seeded with the run's
// seed (its low 32 bits) XOR kSecondAnswerStream, by rnd::below: not NumPy's
// generator (the owner's decision of 2026-10-23), the same draw on every
// platform and in archive.py without the extension.
// ============================================================================

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "detmath.hpp"
#include "numpy_sum.hpp"
#include "random.hpp"

namespace mootation::indicators {

namespace detail_x {

// np.mod(a, 1.0) of a float (npy_divmod): the remainder with the sign of 1
inline double mod1(double a)
{
    double m = std::fmod(a, 1.0);
    if (m != 0.0) {
        if (m < 0.0) m += 1.0;
    } else {
        m = 0.0;
    }
    return m;
}

// np.minimum's step of .min(): a NaN wins
inline double nan_min(double a, double b)
{
    if (std::isnan(a)) return a;
    if (std::isnan(b)) return b;
    return b < a ? b : a;
}

}   // namespace detail_x

// The rows of X (n x d) in [0, 1] by the bounds lo, hi (_unit).
inline std::vector<double> unit_rows(const double* X, std::size_t n, std::size_t d, const double* lo,
                                     const double* hi)
{
    std::vector<double> U(n * d);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < d; ++j)
            U[i * d + j] = (X[i * d + j] - lo[j]) / (hi[j] > lo[j] ? hi[j] - lo[j] : 1.0);
    return U;
}

// The distance from row b of B to row a of A, normalised points of d
// variables, the cyclic ones (cyclic[j] true) wrapped; buf d values.
inline double unit_distance(const double* b, const double* a, std::size_t d, const std::vector<char>& cyclic,
                            std::vector<double>& buf)
{
    for (std::size_t j = 0; j < d; ++j) {
        double v = b[j] - a[j];
        if (!cyclic.empty() && cyclic[j]) v = detail_x::mod1(v + 0.5) - 0.5;
        buf[j] = v * v;
    }
    return std::sqrt(np::sum(buf.data(), d));
}

// For every row of R (the Pareto-set sample, nr x d) the distance to the
// nearest row of U (nu x d, nu > 0): igdx's best.
inline std::vector<double> nearest_to_each(const double* U, std::size_t nu, const double* R, std::size_t nr,
                                           std::size_t d, const std::vector<char>& cyclic)
{
    std::vector<double> best(nr), buf(d);
    for (std::size_t r = 0; r < nr; ++r) {
        double b = 0.0;
        for (std::size_t u = 0; u < nu; ++u) {
            const double v = unit_distance(R + r * d, U + u * d, d, cyclic, buf);
            b = u == 0 ? v : detail_x::nan_min(b, v);
        }
        best[r] = b;
    }
    return best;
}

// metrics.igdx of normalised U and R: the mean of nearest_to_each
inline double igdx(const double* U, std::size_t nu, const double* R, std::size_t nr, std::size_t d,
                   const std::vector<char>& cyclic)
{
    if (nu == 0) return std::numeric_limits<double>::infinity();
    const std::vector<double> best = nearest_to_each(U, nu, R, nr, d, cyclic);
    return np::mean(best.data(), nr);
}

// metrics.ps_dmin of normalised U and R: Python's min over the chunks of 128
// rows of R of each chunk's NumPy minimum
inline double ps_dmin(const double* U, std::size_t nu, const double* R, std::size_t nr, std::size_t d,
                      const std::vector<char>& cyclic)
{
    double best = std::numeric_limits<double>::infinity();
    if (nu == 0) return best;
    const std::vector<double> near = nearest_to_each(U, nu, R, nr, d, cyclic);
    for (std::size_t a = 0; a < nr; a += 128) {
        double m = near[a];
        for (std::size_t r = a + 1; r < a + 128 && r < nr; ++r) m = detail_x::nan_min(m, near[r]);
        if (m < best) best = m;                         // min(best, m): a NaN leaves best
    }
    return best;
}

// metrics.pairwise_distance of normalised U: the sum over the chunks of 128
// rows of the NumPy sum of their distances to every row, over n (n - 1)
inline double pairwise_distance(const double* U, std::size_t n, std::size_t d, const std::vector<char>& cyclic)
{
    if (n < 2) return 0.0;
    double total = 0.0;
    std::vector<double> chunk, buf(d);
    for (std::size_t a = 0; a < n; a += 128) {
        const std::size_t rows = a + 128 < n ? 128 : n - a;
        chunk.resize(rows * n);
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < n; ++j)
                chunk[i * n + j] = unit_distance(U + (a + i) * d, U + j * d, d, cyclic, buf);
        total += np::sum(chunk.data(), chunk.size());
    }
    return total / static_cast<double>(n * (n - 1));
}

// metrics.cover_rate (Tanabe & Ishibuchi 2019, Eqs. 7-8) of X (n x D) and the
// Pareto-set sample P (k x D), in the problem's units: delta_i the squared
// overlap of the ranges of x_i over the sample's, 1 where the sample has a
// single value, 0 where the ranges do not meet, clipped to [0, 1]; their
// product left to right (np.prod), then ^(1 / 2D) by dm::pow — the library's
// own (the owner's decision of 2026-10-23, task 17, stage 2a: the C library's
// pow differed between platforms in the last bit; metrics.CR_VERSION 2).
inline double cover_rate(const double* X, std::size_t n, const double* P, std::size_t k, std::size_t D)
{
    if (n == 0) return 0.0;
    double prod = 1.0;
    for (std::size_t j = 0; j < D; ++j) {
        double smin = P[j], smax = P[j], xmin = X[j], xmax = X[j];
        for (std::size_t r = 1; r < k; ++r) {
            const double v = P[r * D + j];
            smin = detail_x::nan_min(smin, v);
            smax = std::isnan(smax) ? smax : (std::isnan(v) ? v : (v > smax ? v : smax));
        }
        for (std::size_t r = 1; r < n; ++r) {
            const double v = X[r * D + j];
            xmin = detail_x::nan_min(xmin, v);
            xmax = std::isnan(xmax) ? xmax : (std::isnan(v) ? v : (v > xmax ? v : xmax));
        }
        const double span = smax - smin;
        double delta = 1.0;
        if (span > 0.0) {
            const double hi = std::isnan(smax) || std::isnan(xmax) ? smax + xmax : (xmax < smax ? xmax : smax);
            const double lo = std::isnan(smin) || std::isnan(xmin) ? smin + xmin : (xmin > smin ? xmin : smin);
            const double t = (hi - lo) / span;
            delta = t * t;
            if (xmin >= smax || xmax <= smin) delta = 0.0;
        }
        // np.clip(delta, 0, 1)
        delta = std::isnan(delta) ? delta : (delta < 0.0 ? 0.0 : (delta > 1.0 ? 1.0 : delta));
        prod = j == 0 ? delta : prod * delta;
    }
    return dm::pow(prod, 1.0 / (2.0 * static_cast<double>(D)));
}

// For every row of R (n x m) the distance to the nearest other row, NumPy's
// sqrt of the sum of the squares: reference_step's distances (metrics.py),
// by every pair (it replaces SciPy's KD-tree, task 17, stage 2a).
inline std::vector<double> nearest_other(const double* R, std::size_t n, std::size_t m)
{
    std::vector<double> out(n), buf(m);
    for (std::size_t i = 0; i < n; ++i) {
        double best = std::numeric_limits<double>::infinity();
        bool first = true;
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            for (std::size_t c = 0; c < m; ++c) {
                const double v = R[i * m + c] - R[j * m + c];
                buf[c] = v * v;
            }
            const double d = std::sqrt(np::sum(buf.data(), m));
            best = first ? d : detail_x::nan_min(best, d);
            first = false;
        }
        out[i] = best;
    }
    return out;
}

// the second answer's stream: the run's seed XOR this seeds its generator
constexpr std::uint32_t kSecondAnswerStream = 0x2C9277B5u;

// The first point of spread_select among `rows`, from the run's seed.
inline std::size_t spread_first(std::uint64_t seed, std::size_t rows)
{
    std::mt19937 g(static_cast<std::uint32_t>(seed) ^ kSecondAnswerStream);
    return static_cast<std::size_t>(rnd::below(g, rows));
}

// archive.spread_select of the normalised rows Z (rows x d) from `first`: n
// indices, the first then each time the farthest from those chosen (the
// lowest index of a tie; a NaN distance counts as the largest, as np.argmax
// takes it). Every row when rows <= n.
inline std::vector<std::size_t> spread_select(const double* Z, std::size_t rows, std::size_t d, std::size_t n,
                                              std::size_t first)
{
    std::vector<std::size_t> chosen;
    if (rows <= n) {
        for (std::size_t i = 0; i < rows; ++i) chosen.push_back(i);
        return chosen;
    }
    std::vector<double> dist(rows), buf(d);
    auto distance = [&](std::size_t i, std::size_t c) {
        for (std::size_t j = 0; j < d; ++j) {
            const double v = Z[i * d + j] - Z[c * d + j];
            buf[j] = v * v;
        }
        return std::sqrt(np::sum(buf.data(), d));
    };
    chosen.push_back(first);
    for (std::size_t i = 0; i < rows; ++i) dist[i] = distance(i, first);
    dist[first] = -1.0;
    while (chosen.size() < n) {
        std::size_t c = 0;                              // np.argmax: the first maximum, a NaN first
        for (std::size_t i = 1; i < rows; ++i) {
            if (std::isnan(dist[c])) break;
            if (std::isnan(dist[i]) || dist[i] > dist[c]) c = i;
        }
        chosen.push_back(c);
        for (std::size_t i = 0; i < rows; ++i) dist[i] = detail_x::nan_min(dist[i], distance(i, c));
        for (std::size_t k : chosen) dist[k] = -1.0;
    }
    return chosen;
}

}   // namespace mootation::indicators
