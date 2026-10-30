#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// What JADE, SHADE and L-SHADE (jade.hpp, shade.hpp, lshade.hpp) share, each
// piece as JADE introduced it (J. Zhang, A.C. Sanderson, IEEE TEVC 13(5),
// 2009; source jade_zhang2009) and SHADE and L-SHADE repeat it (Tanabe &
// Fukunaga 2013, §IV, §VI (21); 2014, §II-A, II-B):
//   F_i = randc(loc, 0.1), drawn again while <= 0, 1 where > 1 (JADE (10));
//   CR_i = randn(loc, 0.1), clipped to [0, 1] (JADE (8));
//   current-to-pbest/1 with the archive, v = x_i + F (x_pbest - x_i) + F (x_r1
//   - x~_r2), x_r1 != x_i from P, x~_r2 != x_r1, x_i from P u A (JADE (7));
//   a mutant coordinate outside the box becomes the middle of the bound and
//   the parent's coordinate (JADE §II; SHADE (21); L-SHADE (4));
//   binomial crossover with j_rand;
//   the archive cut to its size by removing random members.
//
// Readings (OURS), shared by the three:
//   PB-1. The trial's draws: x_pbest, r1, r2, j_rand, then one U(0, 1) a
//     coordinate, j_rand's included, v_j taken where U < CR or j = j_rand
//     (de.hpp draws the same way). r1 and r2 are drawn again until they
//     differ as asked; r2 indexes P (0 ... |P| - 1), then A.
//   PB-2. "The best" of P: ranked by the comparison of evo.hpp, ties by
//     index; x_pbest is drawn from the first `top`, and may be x_i or x_r1.
// ============================================================================

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

#include "../random.hpp"
#include "evo.hpp"

namespace mootation::scalar::pbest {

using evo::Rows;
using evo::Vec;

inline double draw_F(std::mt19937& rng, double loc)
{
    double F;
    do F = rnd::cauchy(rng, loc, 0.1); while (F <= 0.0);
    return std::min(F, 1.0);
}

inline double draw_CR(std::mt19937& rng, double loc)
{
    return std::clamp(loc + 0.1 * rnd::normal01(rng), 0.0, 1.0);
}

// the indices of P, best first (PB-2)
inline std::vector<std::size_t> ranking(const Vec& f, const Vec& v)
{
    std::vector<std::size_t> order(f.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return evo::better(f[a], v[a], f[b], v[b]); });
    return order;
}

// The trial of x_i by current-to-pbest/1/bin, x_pbest one of the first `top`
// of `order` (PB-1).
inline Vec trial(const Rows& P, const Rows& A, std::size_t i, const std::vector<std::size_t>& order,
                 std::size_t top, double F, double CR, const Vec& lo, const Vec& hi, std::mt19937& rng)
{
    const std::size_t n = P.size(), d = lo.size();
    rnd::uniform_int_distribution<std::size_t> pick_top(0, top - 1), pick_p(0, n - 1), pick_pa(0, n + A.size() - 1),
        dim(0, d - 1);
    rnd::uniform_real_distribution<double> U01(0.0, 1.0);
    const Vec& xi = P[i];
    const Vec& xb = P[order[pick_top(rng)]];
    std::size_t r1, r2;
    do r1 = pick_p(rng); while (r1 == i);
    do r2 = pick_pa(rng); while (r2 == i || r2 == r1);
    const Vec& x1 = P[r1];
    const Vec& x2 = r2 < n ? P[r2] : A[r2 - n];
    const std::size_t jr = dim(rng);
    Vec u(d);
    for (std::size_t j = 0; j < d; ++j) {
        double v = xi[j] + F * (xb[j] - xi[j]) + F * (x1[j] - x2[j]);
        if (v < lo[j]) v = (lo[j] + xi[j]) / 2.0;
        else if (v > hi[j]) v = (hi[j] + xi[j]) / 2.0;
        u[j] = U01(rng) < CR || j == jr ? v : xi[j];
    }
    return u;
}

// remove random members of A until it holds at most `size`
inline void cut(Rows& A, std::size_t size, std::mt19937& rng)
{
    while (A.size() > size) {
        rnd::uniform_int_distribution<std::size_t> k(0, A.size() - 1);
        A.erase(A.begin() + static_cast<std::ptrdiff_t>(k(rng)));
    }
}

// all points of P and A one point
inline bool collapsed(const Rows& P, const Rows& A)
{
    if (P.empty()) return false;
    for (const Vec& x : P) if (x != P[0]) return false;
    for (const Vec& x : A) if (x != P[0]) return false;
    return true;
}

}   // namespace mootation::scalar::pbest
