#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// IREA — Indicator and Reference Points Co-Guided Evolutionary Algorithm
//        for Many-Objective Optimization Problems
// C. Zhou, G. Dai, M. Wang, X. Li — Knowledge-Based Systems 140 (2018)
// doi:10.1016/j.knosys.2017.10.025
//
// Generational scheme (Algorithm 1):
//   1. MatingSelection (Alg. 2): binary tournament — dominance decides the
//      winner; if both are non-dominated and share a reference line, the
//      smaller d_perp wins; otherwise a coin flip.
//   2. Variation: SBX with a large eta_c plus polynomial mutation;
//      S_t = P u Q (2N).
//   3. UpdateIdealPoint (z* is persistent) -> normalization of S_t (Eq. 5,
//      NSGA-III extreme points by ASF -> hyperplane intercepts).
//   4. The eps indicator I_{eps+} (Eq. 2) on the normalized objectives; fitness
//      (Eq. 7): F(x) = Sum_{y!=x} −exp(−I_{eps+}(x,y)/kappa), kappa = 0.05;
//      smaller F is better (§3.6).
//   5. Associate(S_t) (Alg. 3, Eq. 6): pi(s) = argmin d_perp to the reference
//      lines (Das-Dennis, two-layer lattice, Eq. 3–4); pi and d_perp survive
//      the generation and feed the next tournament.
//   6. IndicatorBasedSelection (Alg. 4): within clusters, sort by F into layers
//      L_1, L_2, ...; fill layer by layer; an overflowing layer is
//      ascending-sorted by F and its first N−|P| taken.
//
// PAPER DEFAULTS (§4.3(2)): eta_c=30 ("For NSGA-III, theta-DEA, IREA,
//   distribution index of SBX operator is set to 30 according to [27]"),
//   p_c=1.0, eta_m=20, p_m=1/n.
// DECLARED DEVIATIONS / AMBIGUITIES IN THE PAPER:
//   * Eq. 7 and §3.6/Alg. 4 are internally contradictory: the literal Eq. 7
//     (F = Sum −e^{−I(x2,x1)/kappa}), combined with selecting the MINIMUM F,
//     would pick the worst solutions. The code implements the consistent
//     resolution: F(x) = Sum −e^{−I(x,y)/kappa} with minimum selection, so a
//     dominating solution gets a strongly negative F and is preferred. That
//     preserves the Pareto compliance of the indicator (IBEA semantics).
//   * Eq. 6 as printed is a distance to a point, not a perpendicular. The text
//     of §3.5 ("perpendicular distance... similar with the association operator
//     of NSGA-III") and Fig. 3 are unambiguous, so the code implements the
//     perpendicular; the equation is a typo.
//   * §3.4: "if extreme point set contains duplicate extreme points, or
//     intercepts are nonnegative [an evident typo for non-positive], then
//     intercept can be replaced with each maximum objective functions" — a
//     degenerate matrix, or a missing or non-positive intercept, falls back for
//     ALL denominators to max f_i − z*_i. Subtracting z* is the Eq. 5 /
//     NSGA-III convention; the paper literally says "maximum objective
//     functions".
//   * das_dennis::generate_exact requires N to equal the lattice point count,
//     which is stricter than Table II of the paper, where N may differ
//     slightly.
// EXTENSIONS BEYOND THE PAPER: ConstraintMode::FEASIBILITY — CDP dominance in
//   the tournament (the paper does not consider constraints); off by default.
//   Binary variables (uniform crossover + bit-flip) when bin_vars_n() > 0.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../constraint_mode.hpp"
#include "../das_dennis.hpp"
#include "../data_vault.hpp"
#include "../individuals.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

// ── IREA individual ─────────────────────────────────────────────────────────
// pi is the index of the associated reference line; dperp the perpendicular
// distance; F the fitness (Eq.7); rank the layer index used in selection.
// pi and dperp persist across generations to feed the next binary tournament.
struct IREA_Individual : public Based_Individual {
    int    rank  = 0;
    int    pi    = 0;
    double dperp = 0.0;
    double F     = 0.0;
};

template <typename Ind_t>
class IREACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    static constexpr double KAPPA_ = 0.05;   // kappa is fixed in Eq. 7 of the paper

    double       eta_c_ = 30.0;   // §4.3(2): eta_c = 30 for IREA
    double       eta_m_ = 20.0;   // §4.3(2): η_m = 20
    double       pc_    = 1.0;    // §4.3(2): p_c = 1.0
    double       pm_    = -1.0;   // §4.3(2): p_m = 1/n (<0 → 1/n)
    std::mt19937 rng_{std::random_device{}()};

    std::vector<std::vector<double>> W_;   // reference lines (Das-Dennis)
    std::vector<double>              z_;   // ideal point z*

    bool dominates(DataVault<Ind_t>& vault, int a, int b) const {
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            double ca = vault.get_cv(a), cb = vault.get_cv(b);
            bool af = (ca <= 0.0), bf = (cb <= 0.0);
            if (af && bf == false) return true;
            if (af == false && bf) return false;
            if (af == false && bf == false) return ca < cb;
        }
        const auto& fa = vault.objectives_of(a);
        const auto& fb = vault.objectives_of(b);
        bool strict = false;
        for (std::size_t k = 0; k < fa.size(); ++k) {
            if (fa[k] > fb[k]) return false;
            if (fa[k] < fb[k]) strict = true;
        }
        return strict;
    }

    void update_ideal(DataVault<Ind_t>& vault, int pool) {
        int m = vault.objs_n();
        if (static_cast<int>(z_.size()) != m)
            z_.assign(m, std::numeric_limits<double>::max());
        for (int i = 0; i < pool; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) z_[j] = std::min(z_[j], o[j]);
        }
    }

    // ── Normalization (Eq. 5): NSGA-III extremes -> intercepts (z* = z_). ──
    // §3.4: degenerate extreme points, or missing or non-positive intercepts,
    // fall back for all denominators to max f_i − z*_i.
    std::vector<std::vector<double>>
    normalise(DataVault<Ind_t>& vault, int pool) {
        int m = vault.objs_n();
        const std::vector<double>& zstar = z_;
        const double eps = 1e-6;
        std::vector<int> extreme(m, -1);
        for (int i = 0; i < m; ++i) {
            double best_asf = std::numeric_limits<double>::max();
            for (int v = 0; v < pool; ++v) {
                const auto& o = vault.objectives_of(v);
                double asf = 0.0;
                for (int j = 0; j < m; ++j) {
                    double w = (i == j) ? 1.0 : eps;
                    double val = (o[j] - zstar[j]) / w;
                    if (val > asf) asf = val;
                }
                if (asf < best_asf) { best_asf = asf; extreme[i] = v; }
            }
        }
        std::vector<double> intercepts(m);
        bool degenerate = false;
        for (int i = 0; i < m; ++i) if (extreme[i] < 0) { degenerate = true; break; }
        if (degenerate == false) {
            std::vector<std::vector<double>> A(m, std::vector<double>(m + 1));
            for (int i = 0; i < m; ++i) {
                const auto& o = vault.objectives_of(extreme[i]);
                for (int j = 0; j < m; ++j) A[i][j] = o[j] - zstar[j];
                A[i][m] = 1.0;
            }
            for (int col = 0; col < m && degenerate == false; ++col) {
                int pivot = col;
                for (int row = col + 1; row < m; ++row)
                    if (std::abs(A[row][col]) > std::abs(A[pivot][col])) pivot = row;
                std::swap(A[col], A[pivot]);
                if (std::abs(A[col][col]) < 1e-12) { degenerate = true; break; }
                for (int row = col + 1; row < m; ++row) {
                    double factor = A[row][col] / A[col][col];
                    for (int k = col; k <= m; ++k) A[row][k] -= factor * A[col][k];
                }
            }
            if (degenerate == false) {
                std::vector<double> x(m);
                for (int i = m - 1; i >= 0; --i) {
                    x[i] = A[i][m];
                    for (int j = i + 1; j < m; ++j) x[i] -= A[i][j] * x[j];
                    x[i] /= A[i][i];
                }
                // The intercept a_i = 1/x_i is valid only when x_i > 0, i.e.
                // positive and finite. A negative intercept would invert
                // the normalization (unphysical f~ in the eps indicator);
                // §3.4 requires
                // the fallback. Any invalid component triggers a full fallback.
                for (int i = 0; i < m; ++i) {
                    if (x[i] < 1e-12 || std::isnan(x[i])) { degenerate = true; break; }
                    intercepts[i] = 1.0 / x[i];
                }
            }
        }
        if (degenerate) {
            std::vector<double> nadir(m, -std::numeric_limits<double>::max());
            for (int v = 0; v < pool; ++v) {
                const auto& o = vault.objectives_of(v);
                for (int j = 0; j < m; ++j) nadir[j] = std::max(nadir[j], o[j]);
            }
            for (int j = 0; j < m; ++j) intercepts[j] = nadir[j] - zstar[j];
        }
        std::vector<std::vector<double>> fn(pool, std::vector<double>(m));
        for (int v = 0; v < pool; ++v) {
            const auto& o = vault.objectives_of(v);
            for (int j = 0; j < m; ++j) {
                double denom = std::max(intercepts[j], 1e-12);
                fn[v][j] = (o[j] - zstar[j]) / denom;   // ≥ 0: f ≥ z* (running min)
            }
        }
        return fn;
    }

    // ── Binary eps indicator (Eq.2) ────────────────────────────────────────
    static double eps_plus(const std::vector<double>& a, const std::vector<double>& b) {
        double w = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < a.size(); ++i) {
            double d = a[i] - b[i];
            if (d > w) w = d;
        }
        return w;
    }

    // ── Perpendicular distance (Eq.6, per the text of §3.5 / Fig. 3) ───────
    static double perp(const std::vector<double>& f, const std::vector<double>& w) {
        double dot = 0.0, ww = 0.0;
        for (std::size_t j = 0; j < f.size(); ++j) { dot += f[j] * w[j]; ww += w[j] * w[j]; }
        ww = std::max(ww, 1e-30);
        double t = dot / ww;
        double s = 0.0;
        for (std::size_t j = 0; j < f.size(); ++j) { double d = f[j] - t * w[j]; s += d * d; }
        return std::sqrt(s);
    }

    void rearrange(DataVault<Ind_t>& vault,
                   const std::vector<int>& survivors, int pool_size) {
        int n = static_cast<int>(survivors.size());
        std::vector<int> pos(pool_size), at_pos(pool_size);
        std::iota(pos.begin(), pos.end(), 0);
        std::iota(at_pos.begin(), at_pos.end(), 0);
        for (int i = 0; i < n; ++i) {
            int want = survivors[i], cur = pos[want];
            if (cur == i) continue;
            int other = at_pos[i];
            vault.swap_active(i, cur);
            pos[want] = i; pos[other] = cur;
            at_pos[i] = want; at_pos[cur] = other;
        }
        vault.reduce(n);
    }

    // ── Associate (Algorithm 3): pi / dperp plus the clusters ──────────────
    std::vector<std::vector<int>>
    associate(DataVault<Ind_t>& vault, int pool,
              const std::vector<std::vector<double>>& Fn) {
        int Nref = static_cast<int>(W_.size());
        std::vector<std::vector<int>> clusters(Nref);
        for (int s = 0; s < pool; ++s) {
            int    bj = 0; double bd = std::numeric_limits<double>::max();
            for (int j = 0; j < Nref; ++j) {
                double d = perp(Fn[s], W_[j]);
                if (d < bd) { bd = d; bj = j; }
            }
            vault.get_ind(s).pi    = bj;
            vault.get_ind(s).dperp = bd;
            clusters[bj].push_back(s);
        }
        return clusters;
    }

public:
    IREACore() = default;

    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_pc           (double p) { pc_    = p; }
    void set_pm           (double p) { pm_    = p; }
    void set_seed(unsigned s)        { rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size(), m = vault.objs_n();
        W_ = das_dennis::generate_exact(m, n);
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dr(0.0, 1.0);
        std::uniform_int_distribution<int>     db(0, 1);
        std::vector<double> vars(vault.vars_n());
        std::vector<int>    bvars(vault.bin_vars_n());
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < vault.vars_n(); ++j) {
                double lo = bounds[j].first.value_or(0.0);
                double hi = bounds[j].second.value_or(1.0);
                vars[j] = lo + dr(rng_) * (hi - lo);
            }
            for (int j = 0; j < vault.bin_vars_n(); ++j) bvars[j] = db(rng_);
            if (vault.bin_vars_n() > 0) vault.set_all_variables(i, vars, bvars);
            else                        vault.set_variables(i, vars);
        }
        vault.sync();
        z_.assign(m, std::numeric_limits<double>::max());
        update_ideal(vault, n);
        auto Fn = normalise(vault, n);
        associate(vault, n, Fn);                       // Associate(P0)
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int n = vault.pop_size(), m = vault.objs_n();
        if (W_.empty()) W_ = das_dennis::generate_exact(m, n);
        z_.assign(m, std::numeric_limits<double>::max());
        update_ideal(vault, n);
        auto Fn = normalise(vault, n);
        associate(vault, n, Fn);
    }

    // ── step: one generation (Algorithm 1) ─────────────────────────────────
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int N = n;
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_N(0, n - 1);

        // (7) MatingSelection: binary tournament -> N parents (Algorithm 2).
        std::vector<int> parents;
        parents.reserve(n);
        for (int t = 0; t < n; ++t) {
            int s1 = dist_N(rng_), s2 = dist_N(rng_);
            int win;
            if (dominates(vault, s1, s2))      win = s1;
            else if (dominates(vault, s2, s1)) win = s2;
            else {
                int p1 = vault.get_ind(s1).pi, p2 = vault.get_ind(s2).pi;
                if (p1 == p2)
                    win = (vault.get_ind(s1).dperp < vault.get_ind(s2).dperp) ? s1 : s2;
                else
                    win = (std::uniform_real_distribution<double>(0.0,1.0)(rng_) < 0.5) ? s1 : s2;
            }
            parents.push_back(win);
        }

        // §4.3(2): p_c = 1.0, p_m = 1/n, with n the number of real variables.
        double pm = (pm_ >= 0.0) ? pm_
                                 : 1.0 / static_cast<double>(std::max(1, vault.vars_n()));

        // (8) Variation: SBX + polynomial mutation over parent pairs ->
        // offspring in [N, 2N).
        int off_base = vault.expand(n);
        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        for (int i = 0; i < n; i += 2) {
            int p1 = parents[i], p2 = parents[(i + 1) % n];
            for (int j = 0; j < vault.vars_n(); ++j) {
                pv1[j] = vault.get_variable(p1, j);
                pv2[j] = vault.get_variable(p2, j);
            }
            ops::sbx(pv1, pv2, c1, c2, bounds, eta_c_, pc_, rng_);
            ops::polynomial_mutation(c1, bounds, eta_m_, pm, rng_);
            ops::polynomial_mutation(c2, bounds, eta_m_, pm, rng_);
            if (vault.bin_vars_n() > 0) {
                std::vector<int> bv1(vault.bin_vars_n()), bv2(vault.bin_vars_n()), bc1, bc2;
                for (int j = 0; j < vault.bin_vars_n(); ++j) {
                    bv1[j] = vault.get_bin_variable(p1, j);
                    bv2[j] = vault.get_bin_variable(p2, j);
                }
                ops::binary_crossover(bv1, bv2, bc1, bc2, rng_);
                ops::bit_flip_mutation(bc1, vault.bin_vars_n(), rng_);
                if (i + 1 < n) ops::bit_flip_mutation(bc2, vault.bin_vars_n(), rng_);
                vault.set_all_variables(off_base + i, c1, bc1);
                if (i + 1 < n) vault.set_all_variables(off_base + i + 1, c2, bc2);
            } else {
                vault.set_variables(off_base + i, c1);
                if (i + 1 < n) vault.set_variables(off_base + i + 1, c2);
            }
        }
        vault.sync();

        int pool = n * 2;

        // (9-10) update the ideal point; normalize St.
        update_ideal(vault, pool);
        auto Fn = normalise(vault, pool);

        // (11-12) fitness F(x) = Sum_{y!=x} -exp(-I(x,y)/kappa).
        std::vector<double> F(pool, 0.0);
        for (int i = 0; i < pool; ++i) {
            double s = 0.0;
            for (int j = 0; j < pool; ++j) {
                if (i == j) continue;
                double ind = eps_plus(Fn[i], Fn[j]);
                s += -std::exp(-ind / KAPPA_);
            }
            F[i] = s;
            vault.get_ind(i).F = s;
        }

        // (13) Associate(St) -> clusters.
        auto clusters = associate(vault, pool, Fn);

        // (14) IndicatorBasedSelection: layers by F within clusters (Alg. 4).
        for (auto& cl : clusters)
            std::sort(cl.begin(), cl.end(),
                      [&](int a, int b){ return F[a] < F[b]; });
        int Nf = 0;
        for (const auto& cl : clusters) Nf = std::max(Nf, static_cast<int>(cl.size()));

        std::vector<std::vector<int>> layers(Nf);
        for (const auto& cl : clusters)
            for (int l = 0; l < static_cast<int>(cl.size()); ++l) {
                vault.get_ind(cl[l]).rank = l;
                layers[l].push_back(cl[l]);
            }

        std::vector<int> survivors;
        survivors.reserve(N);
        for (int l = 0; l < Nf && static_cast<int>(survivors.size()) < N; ++l) {
            int room = N - static_cast<int>(survivors.size());
            if (static_cast<int>(layers[l].size()) <= room) {
                for (int v : layers[l]) survivors.push_back(v);
            } else {
                // the last layer overflows: AscendingSort by F (Alg. 4).
                std::sort(layers[l].begin(), layers[l].end(),
                          [&](int a, int b){ return F[a] < F[b]; });
                for (int k = 0; k < room; ++k) survivors.push_back(layers[l][k]);
            }
        }

        rearrange(vault, survivors, pool);
    }
};

} // namespace mootation
