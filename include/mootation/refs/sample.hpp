#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// What every reference front of the cache is made with — task 17, §2.1: the
// reference builders in C++ (Python's benchmarks/registry.py and its
// samplers stay as the cross-check, addendum 4).
//
//   linspace(a, b, n)    numpy.linspace: i * step + a, the last value b
//   distinct_rows(F)     registry._distinct_rows: F without the rows equal to
//                        an earlier one when rounded to 12 decimals
//                        (np.round: x * 1e12, rint, / 1e12), in F's order
//   thin(F, n, seed, piece)
//                        registry._thin: n rows of F in F's order — every
//                        objective's first smallest and first largest row
//                        of every piece of the front (piece[i]: row i's;
//                        none given, one piece), so that an isolated point
//                        (a piece of one) and the ends of every piece stay
//                        (addendum 7, section 1: only where the sampler
//                        knows the pieces); more of them than n is an
//                        error; the others drawn without replacement; the draw is
//                        the library's (std::mt19937 seeded with `seed`,
//                        rnd::below, a partial Fisher-Yates over the others
//                        in their order), not NumPy's default_rng: one
//                        generator in the library (addendum 1, §3), and the
//                        reference version 5 says so
//   exactly_n(pf, n, seed)
//                        registry._exactly_n: pf(n) without repeated rows; a
//                        sample with fewer drawn again at 2n, 4n, ... (up to
//                        64n) while it grows; thinned to n rows (pf(k, piece)
//                        gives the pieces of its rows where it knows them)
//   simplex(M, n)        dtlz_variants._simplex: the points of the grid of
//                        step 1/p on [0, 1]^(M-1) (p = round(n^(1/(M-1))))
//                        whose sum is at most 1 + 1e-9, with 1 - sum after
//                        them, in itertools.product's order; topped up to n
//                        with Dirichlet(1, ..., 1) draws (rnd::gamma of shape
//                        1, divided by their sum) where the grid has fewer
//   sphere_oct(M, n)     dtlz_variants._sphere_oct: simplex(M, n), each row
//                        divided by its norm (NumPy's sum of squares)
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../archive_select.hpp"
#include "../detmath.hpp"
#include "../numpy_sum.hpp"
#include "../random.hpp"

namespace mootation::refs {

// n rows of m values, row-major (the library's point set)
using Rows = select::Rows;

inline Rows rows_of(std::size_t n, std::size_t m)
{
    Rows R;
    R.n = n;
    R.m = m;
    R.v.assign(n * m, 0.0);
    return R;
}

inline void push_row(Rows& R, const double* r)
{
    R.v.insert(R.v.end(), r, r + R.m);
    ++R.n;
}

// the rows of R at `idx`, in that order
inline Rows take(const Rows& R, const std::vector<std::size_t>& idx)
{
    Rows out = rows_of(0, R.m);
    out.v.reserve(idx.size() * R.m);
    for (auto i : idx) push_row(out, R.row(i));
    return out;
}

inline std::vector<double> linspace(double a, double b, std::size_t n)
{
    std::vector<double> y(n);
    if (n == 0) return y;
    if (n == 1) {
        y[0] = a;
        return y;
    }
    const double div = static_cast<double>(n - 1), delta = b - a, step = delta / div;
    for (std::size_t i = 0; i < n; ++i)
        y[i] = step == 0.0 ? static_cast<double>(i) / div * delta + a : static_cast<double>(i) * step + a;
    y[n - 1] = b;
    return y;
}

// np.round(x, 12)
inline double round12(double x)
{
    return std::nearbyint(x * 1e12) / 1e12;
}

// the rows of F not equal to an earlier one (rounded to 12 decimals)
inline std::vector<std::size_t> distinct_index(const Rows& F)
{
    std::map<std::vector<double>, char> seen;     // compared value by value: 0.0 == -0.0
    std::vector<std::size_t> keep;
    std::vector<double> key(F.m);
    for (std::size_t i = 0; i < F.n; ++i) {
        for (std::size_t j = 0; j < F.m; ++j) key[j] = round12(F.row(i)[j]);
        if (seen.emplace(key, char{0}).second) keep.push_back(i);
    }
    return keep;
}

inline Rows distinct_rows(const Rows& F)
{
    return take(F, distinct_index(F));
}

inline Rows thin(const Rows& F, std::size_t n, std::uint32_t seed, const std::vector<std::size_t>& piece = {})
{
    std::vector<char> kept(F.n, 0);
    std::map<std::size_t, std::vector<std::size_t>> pieces;   // piece -> its rows, in F's order
    for (std::size_t i = 0; i < F.n; ++i) pieces[piece.empty() ? 0 : piece.at(i)].push_back(i);
    for (const auto& [id, rows] : pieces)
        for (std::size_t j = 0; j < F.m; ++j) {
            std::size_t lo = rows[0], hi = rows[0];
            for (std::size_t i : rows) {
                if (F.row(i)[j] < F.row(lo)[j]) lo = i;
                if (F.row(i)[j] > F.row(hi)[j]) hi = i;
            }
            kept[lo] = kept[hi] = 1;
        }
    std::vector<std::size_t> rest;
    std::size_t n_keep = 0;
    for (std::size_t i = 0; i < F.n; ++i) {
        if (kept[i]) ++n_keep;
        else rest.push_back(i);
    }
    if (n_keep > n)
        throw std::invalid_argument("thinning to " + std::to_string(n) + " rows: the ends of the front's " +
                                    std::to_string(pieces.size()) + " pieces are " + std::to_string(n_keep) +
                                    " rows already");
    const std::size_t extra = n > n_keep ? std::min(n - n_keep, rest.size()) : 0;
    std::mt19937 g(seed);
    for (std::size_t k = 0; k < extra; ++k) {             // the first `extra` places: the draw
        const std::size_t j = k + static_cast<std::size_t>(rnd::below(g, rest.size() - k));
        std::swap(rest[k], rest[j]);
    }
    for (std::size_t k = 0; k < extra; ++k) kept[rest[k]] = 1;
    std::vector<std::size_t> idx;
    for (std::size_t i = 0; i < F.n; ++i)
        if (kept[i]) idx.push_back(i);
    return take(F, idx);
}

// pf(k, piece): about k rows of the front, and the piece of each in *piece
// where the sampler knows the pieces (left empty where not)
using PiecedSampler = std::function<Rows(std::size_t, std::vector<std::size_t>*)>;

inline Rows exactly_n(const PiecedSampler& pf, std::size_t n, std::uint32_t seed)
{
    constexpr std::size_t kMaxGrowth = 64;
    auto draw = [&](std::size_t k, std::vector<std::size_t>& piece) {
        std::vector<std::size_t> all;
        const Rows R = pf(k, &all);
        const std::vector<std::size_t> keep = distinct_index(R);
        piece.clear();
        if (!all.empty())
            for (std::size_t i : keep) piece.push_back(all.at(i));
        return take(R, keep);
    };
    std::vector<std::size_t> piece, more_piece;
    Rows F = draw(n, piece);
    std::size_t k = n;
    while (F.n < n && k < kMaxGrowth * n) {
        k *= 2;
        Rows more = draw(k, more_piece);
        if (more.n <= F.n) break;                         // the sampler gives no more
        F = std::move(more);
        piece.swap(more_piece);
    }
    return F.n > n ? thin(F, n, seed, piece) : F;
}

inline Rows exactly_n(const std::function<Rows(std::size_t)>& pf, std::size_t n, std::uint32_t seed)
{
    return exactly_n(PiecedSampler([&](std::size_t k, std::vector<std::size_t>*) { return pf(k); }), n, seed);
}

// Python's round() of a positive value: half to even
inline long long round_half_even(double x)
{
    return static_cast<long long>(std::nearbyint(x));
}

inline Rows simplex(std::size_t M, std::size_t n)
{
    if (M < 2) {
        Rows R = rows_of(1, M);
        if (M == 1) R.v[0] = 1.0;
        return R;
    }
    const long long p = std::max<long long>(
        1, round_half_even(dm::pow(static_cast<double>(n), 1.0 / static_cast<double>(M - 1))));
    const std::vector<double> grid = linspace(0.0, 1.0, static_cast<std::size_t>(p + 1));
    Rows W = rows_of(0, M);
    std::vector<std::size_t> digit(M - 1, 0);            // itertools.product: the last digit fastest
    std::vector<double> row(M);
    for (;;) {
        double s = 0.0;
        for (std::size_t j = 0; j + 1 < M; ++j) s += grid[digit[j]];
        if (s <= 1.0 + 1e-9) {
            for (std::size_t j = 0; j + 1 < M; ++j) row[j] = grid[digit[j]];
            row[M - 1] = 1.0 - s;
            push_row(W, row.data());
        }
        std::size_t j = M - 1;
        while (j > 0 && ++digit[j - 1] == grid.size()) digit[--j] = 0;
        if (j == 0) break;
    }
    if (W.n < n) {
        std::mt19937 g(static_cast<std::uint32_t>(20260709 + 1000 * M + n));
        for (std::size_t k = W.n; k < n; ++k) {
            double s = 0.0;
            for (std::size_t j = 0; j < M; ++j) s += (row[j] = rnd::gamma(g, 1.0, 1.0));
            for (std::size_t j = 0; j < M; ++j) row[j] /= s;
            push_row(W, row.data());
        }
    }
    return W;
}

// each row divided by its norm (a zero row left as it is)
inline void unit_rows(Rows& W)
{
    std::vector<double> sq(W.m);
    for (std::size_t i = 0; i < W.n; ++i) {
        double* r = W.row(i);
        for (std::size_t j = 0; j < W.m; ++j) sq[j] = r[j] * r[j];
        double nrm = std::sqrt(np::sum(sq.data(), W.m));
        if (nrm == 0.0) nrm = 1.0;
        for (std::size_t j = 0; j < W.m; ++j) r[j] /= nrm;
    }
}

inline Rows sphere_oct(std::size_t M, std::size_t n)
{
    Rows W = simplex(M, n);
    unit_rows(W);
    return W;
}

}   // namespace mootation::refs
