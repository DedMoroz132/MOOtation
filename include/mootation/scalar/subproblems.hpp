#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The scalarized subproblems.
//
// Every scalarization here is linear in the normalized objectives
// F̄ = (F - z*) / (z^N - z*) and in at most one extra variable t, so one form
// holds them all: minimize  a0·F̄ + b0·t + c0  subject to rows  a·F̄ + b·t + c
// >= 0 (inequalities) and  a·F̄ + b·t + c = 0  (equalities), beside the
// problem's own constraints. A gradient-based solver (sqp) takes them as they
// are, with t a variable of its own; a derivative-free one (nm, cmaes) gets t
// eliminated — the least t the inequalities allow, or the least-squares t of
// the equalities — and what the rows still violate as a violation (merit).
//
// The normalization is the anchors' (Frame): z* the utopia point (each
// objective's minimum among the anchors) and z^N the pseudo-nadir (each
// objective's maximum among them), as NBI's pay-off matrix and NNC's
// normalized space take it (Das & Dennis 1998, §3; Messac et al., as
// gnc_messac2026 §2.2 gives it). Before the anchors are known the frame is
// the identity.
//
//   eps   ε-constraint: min F̄_k  s.t.  F̄_j <= ε̄_j (j != k)
//   tch   Chebyshev, smooth form: min t  s.t.  w_i F̄_i <= t; the lattice gives
//         a direction d on the simplex and w_i = 1 / max(d_i, 1e-6), so that
//         the solution's ray F̄ ∝ d is what the lattice spreads (SCALAR-3);
//         with tch_weights = lattice, w = d (the plain weight grid)
//   nbi   NBI (Das & Dennis, SIAM J. Optim. 8(3), 1998, Eq. (1) and §4.3):
//         max t  s.t.  Φ̄β + t·n̂ = F̄,  n̂ = -Φ̄e the quasi-normal
//   nnc   NNC (Messac, Ismail-Yahaya & Mattson, Struct. Multidisc. Optim.
//         25(2), 2003, Problem P3, Eqs. 36-41): min F̄_m  s.t.
//         (Ā_m - Ā_i)ᵀ(F̄ - X̄) <= 0, i = 1..m-1, X̄ = Σ α_i Ā_i (SCALAR-4)
//   ps    Pascoletti-Serafini (J. Optim. Theory Appl. 42(4), 1984, Problem
//         P(p, q): max ξ s.t. F(x) = p + ξq + λ, λ in the cone, written for
//         maximization; here its minimization form): min t  s.t.
//         ā + t·r - F̄ >= 0 (the ordering cone R^m_+), ā = Φ̄β on the
//         anchors' simplex and r = Φ̄e: NBI's targets and its quasi-normal
//         reversed, with an inequality where NBI has an equality (SCALAR-5).
//         ε-constraint (r = e_k, ā_j = ε_j) and Chebyshev (ā = 0, r_i = 1/w_i)
//         are its special cases.
// (SCALAR-n: the readings declared in scalar.hpp.)
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "../fp_contract.hpp"
#include "lsq.hpp"

namespace mootation::scalar {

using Vec = std::vector<double>;

inline double dot(const Vec& a, const Vec& b)
{
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

// The normalization the anchors give.
struct Frame {
    std::size_t m = 0;
    Vec z, s;
    std::vector<Vec> A;              // A[i] = F̄ of the i-th anchor point

    explicit Frame(std::size_t m_ = 0) : m(m_), z(m_, 0.0), s(m_, 1.0) {}

    // z* = each objective's best among the anchors, z^N its worst; a range
    // that vanishes is taken as 1 (a constant objective).
    static Frame from_anchors(const std::vector<Vec>& F)
    {
        Frame fr(F[0].size());
        for (std::size_t j = 0; j < fr.m; ++j) {
            double lo = F[0][j], hi = F[0][j];
            for (const Vec& f : F) {
                lo = std::min(lo, f[j]);
                hi = std::max(hi, f[j]);
            }
            const double range = hi - lo;
            fr.z[j] = lo;
            fr.s[j] = range > 1e-12 * std::max(1.0, std::fabs(lo)) ? range : 1.0;
        }
        for (const Vec& f : F) fr.A.push_back(fr.bar(f));
        return fr;
    }

    Vec bar(const Vec& f) const
    {
        Vec out(m);
        for (std::size_t j = 0; j < m; ++j) out[j] = (f[j] - z[j]) / s[j];
        return out;
    }

    // Φ̄ β: the anchors' combination (Φ̄'s column i = Ā_i, the origin at z*)
    Vec phi(const Vec& beta) const
    {
        Vec out(m, 0.0);
        for (std::size_t j = 0; j < m; ++j)
            for (std::size_t i = 0; i < m; ++i) out[j] += A[i][j] * beta[i];
        return out;
    }

    Vec quasi_normal() const
    {
        Vec out = phi(Vec(m, 1.0));
        for (double& v : out) v = -v;
        return out;
    }
};

struct Row {
    Vec a;
    double b = 0.0, c = 0.0;
};

// One scalarized subproblem: minimize a0·F̄ + b0·t + c0 under the rows.
struct Sub {
    std::string kind;
    Vec a0;
    double b0 = 0.0, c0 = 0.0;
    bool aux = false;
    std::vector<Row> ineq, eq;
    Vec param;

    // The t a point implies: least squares on the equalities if there are any
    // with t, else the least t every inequality with t allows.
    double t_star(const Vec& Fb) const
    {
        if (!aux) return 0.0;
        bool any_b = false;
        for (const Row& r : eq) any_b = any_b || r.b != 0.0;
        if (!eq.empty() && any_b) {
            double num = 0.0, den = 0.0;
            for (const Row& r : eq) {
                num += r.b * (dot(r.a, Fb) + r.c);
                den += r.b * r.b;
            }
            return -num / den;
        }
        bool have = false;
        double lo = 0.0;
        for (const Row& r : ineq)
            if (r.b > 0.0) {
                const double t = -(dot(r.a, Fb) + r.c) / r.b;
                lo = have ? std::max(lo, t) : t;
                have = true;
            }
        return have ? lo : 0.0;
    }

    // (value, violation) of a point with t eliminated.
    void merit(const Vec& Fb, double& value, double& viol) const
    {
        const double t = t_star(Fb);
        value = dot(a0, Fb) + b0 * t + c0;
        viol = 0.0;
        for (const Row& r : ineq) viol += std::max(0.0, -(dot(r.a, Fb) + r.b * t + r.c));
        for (const Row& r : eq) viol += std::fabs(dot(r.a, Fb) + r.b * t + r.c);
    }
};

// ── the subproblems ─────────────────────────────────────────────────────────
inline Vec unit(std::size_t m, std::size_t k, double v = 1.0)
{
    Vec a(m, 0.0);
    a[k] = v;
    return a;
}

// min F̄_k  s.t.  F̄_j <= ε̄_j for j != k (levels: m - 1 values, in order).
inline Sub eps(std::size_t m, std::size_t k, const Vec& levels)
{
    Sub s;
    s.kind = "eps";
    s.a0 = unit(m, k);
    std::size_t l = 0;
    for (std::size_t j = 0; j < m; ++j) {
        if (j == k) continue;
        const double e = levels[l++];
        if (std::isfinite(e)) s.ineq.push_back({unit(m, j, -1.0), 0.0, e});
    }
    s.param = levels;
    return s;
}

// min t  s.t.  t - w_i F̄_i >= 0: w_i = 1 / max(d_i, 1e-6) (inverse), or the
// lattice point itself, zeros as 1e-6 (lattice: the plain weight grid).
inline Sub tch(std::size_t m, const Vec& d, const std::string& weights)
{
    Vec w(m);
    double sum = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        w[i] = weights == "inverse" ? 1.0 / std::max(d[i], 1e-6) : std::max(d[i], 1e-6);
        sum += w[i];
    }
    for (double& v : w) v /= sum;
    Sub s;
    s.kind = "tch";
    s.a0 = Vec(m, 0.0);
    s.b0 = 1.0;
    s.aux = true;
    for (std::size_t i = 0; i < m; ++i) s.ineq.push_back({unit(m, i, -w[i]), 1.0, 0.0});
    s.param = d;
    return s;
}

// max t  s.t.  Φ̄β + t·n̂ - F̄ = 0.
inline Sub nbi(const Frame& fr, const Vec& beta)
{
    const Vec target = fr.phi(beta), nh = fr.quasi_normal();
    Sub s;
    s.kind = "nbi";
    s.a0 = Vec(fr.m, 0.0);
    s.b0 = -1.0;
    s.aux = true;
    for (std::size_t i = 0; i < fr.m; ++i) s.eq.push_back({unit(fr.m, i, -1.0), nh[i], target[i]});
    s.param = beta;
    return s;
}

// min t  s.t.  ā + t·r - F̄ >= 0,  ā = Φ̄β,  r = Φ̄e.
inline Sub ps(const Frame& fr, const Vec& beta)
{
    const Vec target = fr.phi(beta), nh = fr.quasi_normal();
    Sub s;
    s.kind = "ps";
    s.a0 = Vec(fr.m, 0.0);
    s.b0 = 1.0;
    s.aux = true;
    for (std::size_t i = 0; i < fr.m; ++i) s.ineq.push_back({unit(fr.m, i, -1.0), -nh[i], target[i]});
    s.param = beta;
    return s;
}

// min F̄_m  s.t.  (Ā_m - Ā_i)ᵀ(F̄ - X̄) <= 0, i < m,  X̄ = Σ α_i Ā_i.
inline Sub nnc(const Frame& fr, const Vec& alpha)
{
    const std::size_t m = fr.m;
    Vec X(m, 0.0);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j) X[j] += alpha[i] * fr.A[i][j];
    Sub s;
    s.kind = "nnc";
    s.a0 = unit(m, m - 1);
    for (std::size_t i = 0; i + 1 < m; ++i) {
        Vec N(m), negN(m);
        for (std::size_t j = 0; j < m; ++j) {
            N[j] = fr.A[m - 1][j] - fr.A[i][j];
            negN[j] = -N[j];
        }
        s.ineq.push_back({negN, 0.0, dot(N, X)});
    }
    s.param = alpha;
    return s;
}

// ── where a found point sits in a plan's parameter space (sub_plan = gap) ───
// The parameter a subproblem of this kind would need to aim at F̄: the
// constrained objectives' values (eps), the direction on the simplex (tch),
// the barycentric coordinates of the projection onto the anchors' simplex —
// along the quasi-normal (nbi, ps) or along e (nnc), the least squares
// solution of [Φ̄ dir; eᵀ 0] (β, t) = (F̄, 1) (lsq::hfti).
inline Vec coords(const std::string& kind, const Frame& fr, const Vec& Fb, std::size_t k)
{
    const std::size_t m = fr.m;
    if (kind == "eps") {
        Vec out;
        for (std::size_t j = 0; j < m; ++j)
            if (j != k) out.push_back(Fb[j]);
        return out;
    }
    if (kind == "tch") {
        Vec P(m);
        double s = 0.0;
        for (std::size_t j = 0; j < m; ++j) {
            P[j] = std::max(Fb[j], 0.0);
            s += P[j];
        }
        for (std::size_t j = 0; j < m; ++j) P[j] = s > 0 ? P[j] / s : 1.0 / static_cast<double>(m);
        return P;
    }
    const Vec dir = (kind == "nbi" || kind == "ps") ? fr.quasi_normal() : Vec(m, 1.0);
    lsq::Matrix M(m + 1, m + 1);
    for (std::size_t j = 0; j < m; ++j) {
        for (std::size_t i = 0; i < m; ++i) M(j, i) = fr.A[i][j];
        M(j, m) = dir[j];
        M(m, j) = 1.0;
    }
    Vec rhs(Fb);
    rhs.push_back(1.0);
    Vec sol = lsq::hfti(M, rhs);
    sol.resize(m);
    return sol;
}

}   // namespace mootation::scalar
