#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// What the population methods of scalar/ (de, jade, shade, lshade, ga, pso,
// spso2011, sep_cmaes) share: the comparison of two points and the uniform
// initial population.
//
// The comparison is the constrained-domination of NSGA-II at M = 1 (K. Deb,
// A. Pratap, S. Agarwal, T. Meyarivan, "A fast and elitist multiobjective
// genetic algorithm: NSGA-II", IEEE TEVC 6(2):182-197, 2002, §V-A, Definition
// 1; source nsga2_deb2002), for constraints g <= 0 whose overall violation v
// (the caller's sum of max(0, g_j)) comes with every value:
//   a feasible point (v <= 0) is better than an infeasible one;
//   of two feasible points, the one with the smaller f;
//   of two infeasible points, the one with the smaller v.
// It is not in the papers of the methods, which treat unconstrained problems:
// each method lists it among its deviations. Without constraints (v = 0
// everywhere) it is the plain comparison of f.
//
// OURS:
//   EVO-1. A NaN value (f or v) counts as +infinity: worse than every number,
//     equal to another NaN.
// ============================================================================

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

#include "../random.hpp"

namespace mootation::scalar::evo {

using Vec = std::vector<double>;
using Rows = std::vector<Vec>;

inline double key(double v)                                    // EVO-1
{
    return std::isnan(v) ? std::numeric_limits<double>::infinity() : v;
}

// a is better than b (Deb's Definition 1 at M = 1)
inline bool better(double fa, double va, double fb, double vb)
{
    const bool pa = key(va) <= 0.0, pb = key(vb) <= 0.0;
    if (pa != pb) return pa;
    if (!pa) return key(va) < key(vb);
    return key(fa) < key(fb);
}

// a is not worse than b
inline bool not_worse(double fa, double va, double fb, double vb) { return !better(fb, vb, fa, va); }

// The index of the best of f, v; the first of equals.
inline std::size_t best(const Vec& f, const Vec& v)
{
    std::size_t b = 0;
    for (std::size_t i = 1; i < f.size(); ++i)
        if (better(f[i], v[i], f[b], v[b])) b = i;
    return b;
}

// np points, uniform in the box: point by point, coordinate by coordinate,
// lo + U(0, 1) (hi - lo) — the draw of gde3's setup().
inline Rows uniform(const Vec& lo, const Vec& hi, std::size_t np, std::mt19937& rng)
{
    rnd::uniform_real_distribution<double> U01(0.0, 1.0);
    Rows x(np, Vec(lo.size()));
    for (std::size_t i = 0; i < np; ++i)
        for (std::size_t j = 0; j < lo.size(); ++j) x[i][j] = lo[j] + U01(rng) * (hi[j] - lo[j]);
    return x;
}

}   // namespace mootation::scalar::evo
