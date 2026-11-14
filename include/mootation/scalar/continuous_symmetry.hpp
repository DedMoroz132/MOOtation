#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Continuous symmetries (task 28, §28.5; off by default): a problem may give
// the action of a continuous group on its variables, group_act(x, g) with g
// in R^group_dim and g = 0 the identity, f(group_act(x, g)) = f(x)
// (evaluator.hpp, Problem). A generic mechanism only: a problem's own
// canonical form (a calibration fixed by a factorization, say) is its own
// canon_key and distance (task 24, §24.1.3-4).
//
//   align_continuous(P, x, y)  x moved along its orbit as close to y as it
//                              goes: first the permutations (the problem's
//                              align, else symmetry::align on its blocks and
//                              canon), then min_g ||group_act(x, g) - y||^2
//                              by lm.hpp from g = 0; distance_of (hooks.hpp)
//                              is ||that - y|| where the hooks are given
//   tangent_check(P, x)        the group's tangent directions at g = 0 (the
//                              columns d group_act(x, g) / d g_k) against
//                              the problem's gauge_basis(x): their ranks and
//                              the largest sine of the principal angles
//                              between the two spans (0: the same span)
//
// Readings (OURS):
//   CS-1. The continuous step: lm.hpp's solve on the residuals group_act(x',
//     g) - y (a periodic coordinate's the shorter way, SYM-11), g from 0,
//     forward differences, no bounds, lm's tolerances and its own limit on
//     evaluations; the moved point group_act(x', g*). Its calls of
//     group_act are not the problem's evaluations (no f is computed).
//   CS-2. Permutations and the group alternate: after the continuous step
//     the permutations are aligned again, and while they change the point
//     the continuous step is redone, at most max_passes (5) times; the first
//     pass is the task's order (permutations, then the group). Without
//     permutations (no blocks, no canon, no align) one continuous step.
//   CS-3. tangent_check's tangents: central differences at g = 0 with step
//     h (1e-6 by default); both sets of columns orthonormalized by
//     modified Gram-Schmidt (twice), a column whose remaining norm is below
//     1e-8 of its own norm dropped (the ranks); the sine: max over the
//     orthonormal tangents q of ||q - N N^T q|| and over N's columns of the
//     same against the tangents (1 where the ranks differ).
//   CS-4. Refused: group_dim < 0, group_dim > 0 without group_act or the
//     reverse, group_act giving other than n values, tangent_check without
//     gauge_basis.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "../fp_contract.hpp"
#include "evaluator.hpp"
#include "lm.hpp"
#include "symmetry.hpp"

namespace mootation::scalar {

struct ContinuousAlignment {
    std::vector<double> x;          // the moved point
    std::vector<double> g;          // the last continuous step's g
    double distance = 0.0;          // ||x - y|| (periodic coordinates the shorter way)
    int passes = 0;                 // CS-2
};

namespace detail {

inline void check_group(const Problem& P, std::size_t n)
{
    if (P.group_dim < 0) throw std::invalid_argument("group_dim must be >= 0 (CS-4)");
    if ((P.group_dim > 0) != static_cast<bool>(P.group_act))
        throw std::invalid_argument("group_dim and group_act go together (CS-4)");
    (void)n;
}

inline std::vector<double> act(const Problem& P, const std::vector<double>& x, const std::vector<double>& g)
{
    std::vector<double> out;
    P.group_act(x, g, out);
    if (out.size() != x.size()) throw std::invalid_argument("group_act must give n values (CS-4)");
    return out;
}

inline bool has_permutations(const Problem& P) { return P.align || P.canon || !P.sym_blocks.empty(); }

inline std::vector<double> permuted(const Problem& P, const std::vector<double>& x, const std::vector<double>& y,
                                    const std::vector<double>& period)
{
    if (P.align) {
        std::vector<double> a;
        P.align(x, y, a);
        return a;
    }
    return symmetry::align(x, y, P.sym_blocks, P.canon, period);
}

}   // namespace detail

inline ContinuousAlignment align_continuous(const Problem& P, const std::vector<double>& x,
                                            const std::vector<double>& y, int max_passes = 5)
{
    const std::size_t n = x.size();
    if (y.size() != n) throw std::invalid_argument("align_continuous: x and y differ in length");
    detail::check_group(P, n);
    if (P.group_dim == 0) throw std::invalid_argument("align_continuous: the problem gives no group (CS-4)");
    const std::size_t k = static_cast<std::size_t>(P.group_dim);
    const std::vector<double> period = symmetry::periods(P.periodic, P.lo, P.hi);
    const bool perms = detail::has_permutations(P);
    ContinuousAlignment out;
    std::vector<double> xa = perms ? detail::permuted(P, x, y, period) : x;   // CS-2: the task's order
    for (int pass = 1; pass <= max_passes; ++pass) {
        out.passes = pass;
        lm::Problem p;                                 // CS-1
        p.n = k;
        p.m = n;
        p.residuals = [&](const lm::Vec& g, lm::Vec& r) {
            const std::vector<double> z = detail::act(P, xa, g);
            r.assign(n, 0.0);
            for (std::size_t i = 0; i < n; ++i)
                r[i] = period.empty() ? z[i] - y[i] : symmetry::difference(z[i], y[i], period[i]);
        };
        const lm::Result s = lm::solve(p, lm::Vec(k, 0.0), lm::Options{});
        out.g = s.x;
        xa = detail::act(P, xa, s.x);
        if (!perms) break;
        const std::vector<double> again = detail::permuted(P, xa, y, period);
        if (again == xa) break;
        xa = again;
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = period.empty() ? xa[i] - y[i] : symmetry::difference(xa[i], y[i], period[i]);
        sum += d * d;
    }
    out.x = std::move(xa);
    out.distance = std::sqrt(sum);
    return out;
}

struct TangentCheck {
    std::size_t tangent_rank = 0, gauge_rank = 0;
    double max_sine = 0.0;                             // CS-3
};

namespace detail {

// the columns orthonormalized, the dependent ones dropped (CS-3)
inline std::vector<std::vector<double>> orthonormal(const std::vector<std::vector<double>>& cols)
{
    std::vector<std::vector<double>> q;
    for (const std::vector<double>& c : cols) {
        std::vector<double> v = c;
        double n0 = 0.0;
        for (double t : v) n0 += t * t;
        n0 = std::sqrt(n0);
        if (n0 == 0.0) continue;
        for (int twice = 0; twice < 2; ++twice)
            for (const std::vector<double>& b : q) {
                double d = 0.0;
                for (std::size_t i = 0; i < v.size(); ++i) d += b[i] * v[i];
                for (std::size_t i = 0; i < v.size(); ++i) v[i] -= d * b[i];
            }
        double nv = 0.0;
        for (double t : v) nv += t * t;
        nv = std::sqrt(nv);
        if (nv <= 1e-8 * n0) continue;
        for (double& t : v) t /= nv;
        q.push_back(std::move(v));
    }
    return q;
}

// the largest ||a - B B^T a|| over the orthonormal columns a of A
inline double largest_sine(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B)
{
    double worst = 0.0;
    for (const std::vector<double>& a : A) {
        std::vector<double> r = a;
        for (const std::vector<double>& b : B) {
            double d = 0.0;
            for (std::size_t i = 0; i < r.size(); ++i) d += b[i] * a[i];
            for (std::size_t i = 0; i < r.size(); ++i) r[i] -= d * b[i];
        }
        double s = 0.0;
        for (double t : r) s += t * t;
        worst = std::max(worst, std::sqrt(s));
    }
    return worst;
}

}   // namespace detail

inline TangentCheck tangent_check(const Problem& P, const std::vector<double>& x, double h = 1e-6)
{
    const std::size_t n = x.size();
    detail::check_group(P, n);
    if (P.group_dim == 0) throw std::invalid_argument("tangent_check: the problem gives no group (CS-4)");
    if (!P.gauge_basis) throw std::invalid_argument("tangent_check: the problem gives no gauge_basis (CS-4)");
    const std::size_t k = static_cast<std::size_t>(P.group_dim);
    std::vector<std::vector<double>> T;
    for (std::size_t c = 0; c < k; ++c) {
        std::vector<double> gp(k, 0.0), gm(k, 0.0);
        gp[c] = h;
        gm[c] = -h;
        const std::vector<double> a = detail::act(P, x, gp), b = detail::act(P, x, gm);
        std::vector<double> col(n);
        for (std::size_t i = 0; i < n; ++i) col[i] = (a[i] - b[i]) / (2.0 * h);
        T.push_back(std::move(col));
    }
    Rows N;
    P.gauge_basis(x, N);
    if (N.size() != n) throw std::invalid_argument("gauge_basis: n rows are needed");
    const std::size_t kn = N.empty() ? 0 : N[0].size();
    std::vector<std::vector<double>> G(kn, std::vector<double>(n));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t c = 0; c < kn; ++c) G[c][i] = N[i].at(c);
    const auto QT = detail::orthonormal(T), QG = detail::orthonormal(G);
    TangentCheck r;
    r.tangent_rank = QT.size();
    r.gauge_rank = QG.size();
    r.max_sine = QT.size() != QG.size() ? 1.0
                                        : std::max(detail::largest_sine(QT, QG), detail::largest_sine(QG, QT));
    return r;
}

}   // namespace mootation::scalar
