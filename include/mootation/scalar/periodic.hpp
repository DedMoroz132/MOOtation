#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Periodic variables (task 24, §24.4.2; Problem::periodic, evaluator.hpp): a
// variable whose box [lo, hi) is one period, hi - lo — an angle, a phase, the
// rotation of a square modulo pi/2. Where a method would repair a coordinate
// that left the box (reflect, a midpoint, a clip and its penalty), a periodic
// one is wrapped into [lo, hi) instead, and distances between points take
// the shorter way round. The seam is one point: hi is lo.
//
// Where: de, jade, shade, lshade, g3_pcx, pso, spso2011 (their repairs),
// the normalized box of cmaes, nelder_mead and slsqp (no edge: they go
// round, their points wrapped), lm and lbfgsb (no bound on the variable,
// their points wrapped), sep_cmaes, ipop and bipop's samples, ils and
// basin_hopping's steps, the strategies' minima log, the symmetries' align
// and distance (symmetry.hpp, SYM-11). eki keeps the box (its ensemble is
// laid out in it); chc never leaves it (its genes are bits of the box).
//
// READINGS (all of them off by default: with Problem::periodic only)
//   PER-1. Off by default. The wrap is lo + fmod(v - lo, hi - lo), negatives
//     raised by a period; a result that rounds to hi is lo (no point on the
//     seam).
//   PER-2. Off by default. A wrapped coordinate is not a repair
//     (RepairCounts, task 20, §20.2.2, does not count it) and costs no
//     penalty (SCALAR-7, POP-4): the point is inside the problem's domain.
//   PER-3. Off by default. A difference d of a periodic coordinate is
//     min(|d|, period - |d|).
// ============================================================================

#include <cmath>
#include <cstddef>
#include <vector>

namespace mootation::scalar::periodic {

// v in [lo, hi), modulo hi - lo (PER-1)
inline double wrap_into(double v, double lo, double hi)
{
    const double d = hi - lo;
    if (!(d > 0.0)) return lo;
    double t = std::fmod(v - lo, d);
    if (t < 0.0) t += d;
    const double r = lo + t;
    return r < hi ? r : lo;
}

// y of the box normalized to [0, 1), modulo 1
inline double wrap01(double y) { return wrap_into(y, 0.0, 1.0); }

// variable j is periodic in the mask (empty: none)
inline bool is(const std::vector<char>& mask, std::size_t j) { return !mask.empty() && mask[j] != 0; }

// |a - b| of variable j, the shorter way round where it is periodic of period p (PER-3)
inline double gap(double a, double b, bool periodic, double p)
{
    const double d = std::fabs(a - b);
    return periodic && d > 0.5 * p ? p - d : d;
}

}   // namespace mootation::scalar::periodic
