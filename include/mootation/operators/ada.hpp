#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// ADA — the framework of Tanabe & Ishibuchi for multi-modal multi-objective
// optimization in the algorithms that divide the objective space between
// reference vectors: "A framework to handle multi-modal multi-objective
// optimization in decomposition-based evolutionary algorithms", IEEE TEVC
// 24(4), 2020 (source mmo-decomposition-ada_tanabe2020; Algorithm 1 and §III)
// and its supplement (mmo-decomposition-ada_tanabe2020_supplement: Algorithms
// S.10 NSGA-III-ADA, S.11 θ-DEA-ADA, S.12 RVEA-ADA). Task 15 of 2026-10-21, §6.
//
// One child a step (μ + 1): parents a ≠ b uniform from P, SBX's first child,
// polynomial mutation (§III-A: "the same mating selection scheme, regardless of
// the mating selection schemes in their original EMOAs"); P ∪ {u} normalized
// by the host's rule (line 10); u assigned to subproblem j (line 11); X = the
// members of P assigned to j among the L = ⌊0.1 μ⌋ nearest to u in the
// variables normalized by the bounds (§III-D, after [17]); u removes every x
// of X it beats and enters P when it beat one or X is empty (lines 13-21). A
// member keeps the subproblem it was assigned when it entered (lines 2-4 for
// the initial population). P grows.
//
//   Rule::NsgaIII  normalization: the hyperplane through the extreme points
//                  (ASF) of P ∪ {u}, the nadir of the set where it degenerates
//                  (as nsga3.hpp's NSGA3-N2, over the set at hand: ADA-1);
//                  assignment: the least perpendicular distance PD to w_k;
//                  u beats x: u dominates x, or neither dominates and PD(u) <
//                  PD(x) (S.10, line 18)
//   Rule::ThetaDEA the same normalization and assignment; u beats x: u
//                  dominates x, or neither dominates and u θ-dominates x —
//                  d1 + θ d2 smaller, θ = theta (5) and 10^6 on the axes'
//                  vectors (S.11, line 18; θ-DEA's PBI)
//   Rule::Rvea     translation by the least values of P ∪ {u}; assignment:
//                  the least angle to v_k; u beats x: APD(u | v_j) <= APD(x |
//                  v_j), APD = (1 + M (t / t_max)^α θ / γ_j) ||f'||, α = 2, γ_j
//                  the least angle between v_j and another vector, t / t_max
//                  the share of the budget spent (S.12, line 18)
//
// With constraints (FEASIBILITY, beyond the paper: ADA-2) a point with less
// violation beats one with more before any of the rules.
//
// The answer (§III-H, A^primary): for every subproblem the best member
// assigned to it by the rule, then those no other of them dominates; at most
// N points, published into the vault. The decision-space indicators take their
// own answer from every point evaluated (campaign set_answer, §III-H's
// A^tertiary rule).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../detmath.hpp"
#include "../random.hpp"
#include "poly_mutation.hpp"
#include "sbx.hpp"

namespace mootation {
namespace ada {

enum class Rule { NsgaIII, ThetaDEA, Rvea };

struct Member {
    std::vector<double> x, f;
    double cv = 0.0;
    int j = -1;                      // the subproblem it was assigned on entering
    VaultPoint pt{};
};

// The hyperplane normalization of a set (NSGA-III's, over the rows given):
// ideal = the least values; extremes by the achievement function with weights
// (1, 1e-6, ...); intercepts from the hyperplane through them; the set's nadir
// on an axis whose intercept is not positive, on every axis when the system
// is singular (ADA-1).
inline std::vector<std::vector<double>> hyperplane_normalize(const std::vector<std::vector<double>>& F)
{
    const std::size_t n = F.size(), m = F.empty() ? 0 : F[0].size();
    std::vector<double> z(m, std::numeric_limits<double>::max()), nad(m, -std::numeric_limits<double>::max());
    for (const auto& f : F)
        for (std::size_t k = 0; k < m; ++k) { z[k] = std::min(z[k], f[k]); nad[k] = std::max(nad[k], f[k]); }
    std::vector<std::size_t> ext(m, 0);
    for (std::size_t i = 0; i < m; ++i) {
        double best = std::numeric_limits<double>::max();
        for (std::size_t r = 0; r < n; ++r) {
            double a = -std::numeric_limits<double>::max();
            for (std::size_t k = 0; k < m; ++k) a = std::max(a, (F[r][k] - z[k]) / (k == i ? 1.0 : 1e-6));
            if (a < best) { best = a; ext[i] = r; }
        }
    }
    std::vector<double> icpt(m, -1.0);
    bool singular = false;
    std::vector<std::vector<double>> A(m, std::vector<double>(m + 1, 1.0));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t k = 0; k < m; ++k) A[i][k] = F[ext[i]][k] - z[k];
    for (std::size_t c = 0; c < m && !singular; ++c) {
        std::size_t piv = c;
        for (std::size_t r = c + 1; r < m; ++r) if (std::abs(A[r][c]) > std::abs(A[piv][c])) piv = r;
        std::swap(A[c], A[piv]);
        if (std::abs(A[c][c]) < 1e-12) { singular = true; break; }
        for (std::size_t r = c + 1; r < m; ++r) {
            const double t = A[r][c] / A[c][c];
            for (std::size_t k = c; k <= m; ++k) A[r][k] -= t * A[c][k];
        }
    }
    if (!singular) {
        std::vector<double> xs(m);
        for (std::size_t ii = m; ii-- > 0;) {
            xs[ii] = A[ii][m];
            for (std::size_t k = ii + 1; k < m; ++k) xs[ii] -= A[ii][k] * xs[k];
            xs[ii] /= A[ii][ii];
        }
        for (std::size_t k = 0; k < m; ++k) icpt[k] = xs[k] > 1e-12 ? 1.0 / xs[k] : -1.0;
    }
    for (std::size_t k = 0; k < m; ++k)
        if (singular || icpt[k] <= 1e-12) icpt[k] = std::max(nad[k] - z[k], 1e-12);
    std::vector<std::vector<double>> G(n, std::vector<double>(m));
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t k = 0; k < m; ++k) G[r][k] = (F[r][k] - z[k]) / icpt[k];
    return G;
}

// RVEA's: translated by the least values
inline std::vector<std::vector<double>> translate(const std::vector<std::vector<double>>& F)
{
    const std::size_t m = F.empty() ? 0 : F[0].size();
    std::vector<double> z(m, std::numeric_limits<double>::max());
    for (const auto& f : F) for (std::size_t k = 0; k < m; ++k) z[k] = std::min(z[k], f[k]);
    std::vector<std::vector<double>> G = F;
    for (auto& g : G) for (std::size_t k = 0; k < m; ++k) g[k] -= z[k];
    return G;
}

inline double dot(const std::vector<double>& a, const std::vector<double>& b)
{
    double s = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) s += a[k] * b[k];
    return s;
}

// d1: the length of f along w; d2: its perpendicular distance to w's line
inline void pbi_parts(const std::vector<double>& f, const std::vector<double>& w, double& d1, double& d2)
{
    const double ww = std::sqrt(dot(w, w));
    d1 = dot(f, w) / ww;
    double s = 0.0;
    for (std::size_t k = 0; k < f.size(); ++k) {
        const double e = f[k] - d1 * w[k] / ww;
        s += e * e;
    }
    d2 = std::sqrt(s);
}

inline double angle(const std::vector<double>& f, const std::vector<double>& v)
{
    const double nf = std::sqrt(dot(f, f)), nv = std::sqrt(dot(v, v));
    if (nf <= 0.0 || nv <= 0.0) return 0.0;
    return dm::acos(std::max(-1.0, std::min(1.0, dot(f, v) / (nf * nv))));
}

inline bool dominates(const std::vector<double>& a, const std::vector<double>& b)
{
    bool strict = false;
    for (std::size_t k = 0; k < a.size(); ++k) {
        if (a[k] > b[k]) return false;
        if (a[k] < b[k]) strict = true;
    }
    return strict;
}

class Population {
public:
    Rule rule = Rule::NsgaIII;
    std::vector<std::vector<double>> W;     // reference points (simplex) or vectors
    double theta = 5.0;                      // θ-DEA's
    double alpha = 2.0;                      // RVEA's
    double eta_c = 30.0, eta_m = 20.0, pc = 1.0;
    ConstraintMode constraint_mode = ConstraintMode::NONE;

    std::size_t size() const { return P_.size(); }
    const std::vector<Member>& members() const { return P_; }

    // The initial population (vault [0, active)) taken in and assigned (lines 1-4).
    template <typename Ind_t>
    void start(DataVault<Ind_t>& vault)
    {
        if (vault.bin_vars_n() > 0)
            throw std::invalid_argument("ada: continuous variables only");
        read_bounds(vault);
        P_.clear();
        for (std::size_t v = 0; v < vault.active_n(); ++v) {
            Member e;
            e.x = vault.variables_of(v);
            e.f = vault.objectives_of(v);
            e.cv = vault.get_cv(v);
            e.pt = vault.point_of(v);
            P_.push_back(std::move(e));
        }
        gamma_.assign(W.size(), std::numeric_limits<double>::max());
        for (std::size_t a = 0; a < W.size(); ++a)
            for (std::size_t b = 0; b < W.size(); ++b)
                if (a != b) gamma_[a] = std::min(gamma_[a], angle(W[a], W[b]));
        std::vector<std::vector<double>> F;
        for (const auto& e : P_) F.push_back(e.f);
        const auto G = normalize(F);
        for (std::size_t i = 0; i < P_.size(); ++i) P_[i].j = assign(G[i]);
        publish(vault);
    }

    // One child (lines 6-21); t_frac: the share of the budget spent (RVEA's APD).
    template <typename Ind_t, typename RNG>
    void step(DataVault<Ind_t>& vault, RNG& rng, double t_frac)
    {
        t_frac_ = std::max(0.0, std::min(1.0, t_frac));
        const std::size_t mu = P_.size();
        rnd::uniform_int_distribution<std::size_t> any(0, mu - 1);
        const std::size_t a = any(rng);
        std::size_t b = any(rng);
        while (mu > 1 && b == a) b = any(rng);
        std::vector<double> c1, c2;
        ops::sbx(P_[a].x, P_[b].x, c1, c2, vault.get_bounds(), eta_c, pc, rng);
        const double pm = c1.empty() ? 0.0 : 1.0 / static_cast<double>(c1.size());
        ops::polynomial_mutation(c1, vault.get_bounds(), eta_m, pm, rng);
        Member u = evaluate(vault, c1);

        std::vector<std::vector<double>> F;                       // line 10: P ∪ {u}
        for (const auto& e : P_) F.push_back(e.f);
        F.push_back(u.f);
        const auto G = normalize(F);
        u.j = assign(G[mu]);                                      // lines 11-12

        // line 13: the L = ⌊0.1 μ⌋ nearest in the normalized variables, those of j
        const std::size_t L = mu / 10;
        std::vector<std::pair<double, std::size_t>> d;
        for (std::size_t i = 0; i < mu; ++i) {
            double s = 0.0;
            for (std::size_t k = 0; k < u.x.size(); ++k) {
                const double e = (P_[i].x[k] - u.x[k]) / span_[k];
                s += e * e;
            }
            d.push_back({s, i});
        }
        std::partial_sort(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(std::min(L, mu)), d.end());
        std::vector<char> gone(mu, 0);
        bool winner = false, explorer = true;
        for (std::size_t q = 0; q < std::min(L, mu); ++q) {
            const std::size_t i = d[q].second;
            if (P_[i].j != u.j) continue;
            explorer = false;                                      // lines 14-16
            if (beats(u, G[mu], P_[i], G[i], u.j)) { gone[i] = 1; winner = true; }   // lines 17-19
        }
        if (winner || explorer) {                                  // lines 20-21
            std::vector<Member> next;
            for (std::size_t i = 0; i < mu; ++i) if (!gone[i]) next.push_back(std::move(P_[i]));
            next.push_back(std::move(u));
            P_ = std::move(next);
        }
        publish(vault);
    }

private:
    std::vector<Member> P_;
    std::vector<double> lo_, span_, gamma_;
    double t_frac_ = 0.0;

    template <typename Ind_t>
    void read_bounds(DataVault<Ind_t>& vault)
    {
        const auto& bd = vault.get_bounds();
        lo_.assign(bd.size(), 0.0);
        span_.assign(bd.size(), 1.0);
        for (std::size_t k = 0; k < bd.size(); ++k) {
            const double lo = bd[k].first.value_or(0.0), hi = bd[k].second.value_or(1.0);
            lo_[k] = lo;
            span_[k] = hi > lo ? hi - lo : 1.0;
        }
    }

    std::vector<std::vector<double>> normalize(const std::vector<std::vector<double>>& F) const
    {
        return rule == Rule::Rvea ? translate(F) : hyperplane_normalize(F);
    }

    int assign(const std::vector<double>& g) const
    {
        std::size_t best = 0;
        double bv = std::numeric_limits<double>::max();
        for (std::size_t k = 0; k < W.size(); ++k) {
            double v;
            if (rule == Rule::Rvea) v = angle(g, W[k]);
            else { double d1, d2; pbi_parts(g, W[k], d1, d2); v = d2; }
            if (v < bv) { bv = v; best = k; }
        }
        return static_cast<int>(best);
    }

    bool on_axis(std::size_t j) const
    {
        int nz = 0;
        for (double w : W[j]) nz += w > 1e-12;
        return nz == 1;
    }

    // the rule's value of a normalized point for subproblem j: smaller is better
    double value(const std::vector<double>& g, std::size_t j) const
    {
        if (rule == Rule::Rvea) {
            const double M = static_cast<double>(g.size());
            const double pen = M * dm::pow(t_frac_, alpha) * angle(g, W[j]) / std::max(gamma_[j], 1e-12);
            return (1.0 + pen) * std::sqrt(dot(g, g));
        }
        double d1, d2;
        pbi_parts(g, W[j], d1, d2);
        if (rule == Rule::NsgaIII) return d2;
        return d1 + (on_axis(j) ? 1e6 : theta) * d2;
    }

    bool beats(const Member& u, const std::vector<double>& gu, const Member& x,
               const std::vector<double>& gx, int j) const
    {
        if (constraint_mode != ConstraintMode::NONE && (u.cv > 0.0 || x.cv > 0.0))
            return u.cv < x.cv;                                                       // ADA-2
        if (rule == Rule::Rvea) return value(gu, static_cast<std::size_t>(j)) <= value(gx, static_cast<std::size_t>(j));
        if (dominates(u.f, x.f)) return true;
        if (dominates(x.f, u.f)) return false;
        return value(gu, static_cast<std::size_t>(j)) < value(gx, static_cast<std::size_t>(j));
    }

    template <typename Ind_t>
    Member evaluate(DataVault<Ind_t>& vault, const std::vector<double>& x)
    {
        const int keep = static_cast<int>(vault.active_n());
        const int base = vault.expand(1);
        vault.set_variables(static_cast<std::size_t>(base), x);
        vault.sync();
        const std::size_t v = static_cast<std::size_t>(base);
        Member e;
        e.x = vault.variables_of(v);
        e.f = vault.objectives_of(v);
        e.cv = vault.get_cv(v);
        e.pt = vault.point_of(v);
        vault.reduce(keep);
        return e;
    }

    // A^primary (§III-G, H): the best member of each subproblem, then those no
    // other of them dominates, into the vault
    template <typename Ind_t>
    void publish(DataVault<Ind_t>& vault)
    {
        std::vector<std::vector<double>> F;
        for (const auto& e : P_) F.push_back(e.f);
        const auto G = normalize(F);
        std::vector<long long> best(W.size(), -1);
        for (std::size_t i = 0; i < P_.size(); ++i) {
            const std::size_t j = static_cast<std::size_t>(P_[i].j);
            const long long b = best[j];
            if (b < 0 || beats(P_[i], G[i], P_[static_cast<std::size_t>(b)], G[static_cast<std::size_t>(b)], P_[i].j))
                best[j] = static_cast<long long>(i);
        }
        std::vector<std::size_t> A;
        for (long long b : best) if (b >= 0) A.push_back(static_cast<std::size_t>(b));
        std::vector<std::size_t> nd;
        for (std::size_t a : A) {
            bool beaten = false;
            for (std::size_t c : A) beaten = beaten || (c != a && dominates(P_[c].f, P_[a].f));
            if (!beaten) nd.push_back(a);
        }
        std::sort(nd.begin(), nd.end());
        vault.reduce(0);
        vault.expand(static_cast<int>(nd.size()));
        for (std::size_t i = 0; i < nd.size(); ++i) vault.put_point(i, P_[nd[i]].pt);
    }
};

}   // namespace ada
}   // namespace mootation
