#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// AGE-MOEA — An Adaptive Evolutionary Algorithm Based on Non-Euclidean
//            Geometry for Many-Objective Optimization
// A. Panichella — GECCO 2019
// doi:10.1145/3321707.3321839
//
// Generational scheme (Algorithm 1):
//   1. Q: binary tournament (rank, then survival score), SBX (eta_c=30, pc=1)
//      + PM (eta_m=20, pm=1/n)
//   2. NDS(P ∪ Q); normalize from F1 (z^min plus the hyperplane intercepts),
//      applied to ALL fronts; "abnormal normalization results" — a degenerate
//      system or a non-positive intercept — fall back to the min-max
//      a_i = z_i^max − z_i^min over F1 (§3.2)
//   3. p = GET-GEOMETRY (Eq.8): C = argmin of the PERPENDICULAR distance to the
//      bisector β (Eq.6: dist⊥ = √(‖f‖² − (Σf_i)²/M));
//      p = log(M)/(log(M) − log(ΣC_i))
//   4. Survival score (Alg.2): the extremes (argmax f^n_i within the front,
//      §3.2) get +∞; then greedily value = diversity/proximity, with
//      diversity = min1 + min2 of the distances to Σ; for d > 1,
//      score = 1/‖f^n‖_p
//   5. Fronts are accepted; the partial front is taken by decreasing score
//
// PAPER DEFAULTS (Table 1): pc=1, eta_c=30, pm=1/n, eta_m=20.
// DECLARED DEVIATIONS:
//   - compute_norm_params builds the hyperplane from the ASF extremes, the
//     NSGA-III practice. The letter of §3.2 defines Z^max through per-objective
//     maxima; the paper is internally ambiguous, saying "the same formula used
//     in NSGA-III". The +inf score extremes follow the letter of §3.2
//     (argmax f^n_i), so the two sets may differ;
//   - diversity = min1 + min2, the two smallest L_p distances from the
//     candidate to the ALREADY-SELECTED set Σ. The paper specifies this
//     quantity three mutually inconsistent ways and none of them is the one
//     implemented: §3.4's prose says "the minimum distance (L_p norm) with the
//     other solutions in the front F_1" — a SINGLE minimum over all of F_1,
//     which is not even greedy-selection-dependent; Eq.11 writes it as
//     diversity(S, F_1), again over the whole front; Alg.2 line 13 writes
//     min_{T∈Ω̄} dist + min_{T∈Ω} dist — one minimum over the REMAINING set
//     plus one over the selected set. This port follows the authors' own
//     reference implementation (two nearest neighbours among Σ), which is what
//     downstream ports such as pymoo's AGEMOEA also do. The difference is not
//     cosmetic: line 13's first term rewards a candidate for being far from
//     other unselected candidates, which under greedy selection makes the score
//     of every remaining solution change as the pool empties, whereas the
//     implemented form measures only the gap the candidate would fill in the
//     set actually being built. Note the Alg.2 line-13 form is also degenerate
//     at the last step, where Ω̄ = {S} and its first term is 0 by convention or
//     undefined by inspection;
//   - guards beyond the paper: clamp f^n >= 0, sum_C in (0,M), p in [0.1, 20],
//     proximity >= 1e-14 — inactive on regular data.
// EXTENSIONS BEYOND THE PAPER: constraint_mode FEASIBILITY (CDP; off by
//   default); mixed real+binary genome.
//
// Individual: AGEMOEA_Individual (individuals.hpp): rank, survival_score.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

template <typename Ind_t>
class AGEMOEACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double       eta_c_ = 30.0;  // Table 1
    double       eta_m_ = 20.0;  // Table 1
    double       pc_    = 1.0;   // Table 1: «SBX probability pc = 1»
    std::mt19937 rng_{std::random_device{}()};

    // ── Dominance ──────────────────────────────────────────────────────────
    bool dominates(DataVault<Ind_t>& vault, int a, int b) const {
        const auto& fa = vault.objectives_of(a);
        const auto& fb = vault.objectives_of(b);
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            double ca = vault.get_cv(a), cb = vault.get_cv(b);
            bool af = (ca <= 0.0), bf = (cb <= 0.0);
            if ( af && !bf) return true;
            if (!af &&  bf) return false;
            if (!af && !bf) return ca < cb;
        }
        bool better = false;
        for (std::size_t i = 0; i < fa.size(); ++i) {
            if (fa[i] > fb[i]) return false;
            if (fa[i] < fb[i]) better = true;
        }
        return better;
    }

    // ── Fast non-dominated sort ────────────────────────────────────────────
    std::vector<std::vector<int>>
    fast_nondominated_sort(DataVault<Ind_t>& vault, int n) {
        std::vector<std::vector<int>> S(n);
        std::vector<int> np(n, 0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                if (dominates(vault, i, j)) S[i].push_back(j);
                else if (dominates(vault, j, i)) ++np[i];
            }
        std::vector<std::vector<int>> fronts;
        std::vector<int> f0;
        for (int i = 0; i < n; ++i) if (np[i] == 0) { vault.get_ind(i).rank = 0; f0.push_back(i); }
        fronts.push_back(f0);
        int k = 0;
        while (!fronts[k].empty()) {
            std::vector<int> nxt;
            for (int i : fronts[k])
                for (int j : S[i])
                    if (--np[j] == 0) { vault.get_ind(j).rank = k + 1; nxt.push_back(j); }
            fronts.push_back(nxt);
            ++k;
        }
        fronts.pop_back();
        return fronts;
    }

    // ── Normalisation (NSGA-III style, §3.2) ──────────────────────────────
    // Returns z_min and intercepts a from F1.
    // Normalised coords stored in fn[pool_idx][obj].
    struct NormParams { std::vector<double> zmin, intercepts; };

    NormParams compute_norm_params(DataVault<Ind_t>& vault,
                                   const std::vector<int>& F1) const {
        int m = vault.objs_n();
        NormParams np;
        np.zmin.assign(m, std::numeric_limits<double>::max());
        for (int v : F1) {
            const auto& o = vault.objectives_of(v);
            for (int j = 0; j < m; ++j) np.zmin[j] = std::min(np.zmin[j], o[j]);
        }
        // Extreme points via ASF.
        const double eps = 1e-6;
        std::vector<int> extreme(m, -1);
        for (int i = 0; i < m; ++i) {
            double best = std::numeric_limits<double>::max();
            for (int v : F1) {
                const auto& o = vault.objectives_of(v);
                double asf = 0.0;
                for (int j = 0; j < m; ++j) {
                    double w = (i == j) ? 1.0 : eps;
                    double val = (o[j] - np.zmin[j]) / w;
                    if (val > asf) asf = val;
                }
                if (asf < best) { best = asf; extreme[i] = v; }
            }
        }
        // Solve hyperplane to get intercepts.
        np.intercepts.assign(m, 1.0);
        bool degenerate = false;
        for (int i = 0; i < m && !degenerate; ++i) if (extreme[i] < 0) degenerate = true;
        if (!degenerate) {
            std::vector<std::vector<double>> A(m, std::vector<double>(m + 1));
            for (int i = 0; i < m; ++i) {
                const auto& o = vault.objectives_of(extreme[i]);
                for (int j = 0; j < m; ++j) A[i][j] = o[j] - np.zmin[j];
                A[i][m] = 1.0;
            }
            for (int col = 0; col < m && !degenerate; ++col) {
                int pivot = col;
                for (int row = col + 1; row < m; ++row)
                    if (std::abs(A[row][col]) > std::abs(A[pivot][col])) pivot = row;
                std::swap(A[col], A[pivot]);
                if (std::abs(A[col][col]) < 1e-12) { degenerate = true; break; }
                for (int row = col + 1; row < m; ++row) {
                    double f = A[row][col] / A[col][col];
                    for (int k = col; k <= m; ++k) A[row][k] -= f * A[col][k];
                }
            }
            if (!degenerate) {
                std::vector<double> x(m);
                for (int i = m - 1; i >= 0; --i) {
                    x[i] = A[i][m];
                    for (int j = i + 1; j < m; ++j) x[i] -= A[i][j] * x[j];
                    x[i] /= A[i][i];
                }
                // §3.2: «the system Z^max a = 1 may be … leading to abnormal
                // normalization results": a non-positive intercept (x_i <= 0)
                // is treated as abnormal -> full min-max fallback.
                for (int i = 0; i < m; ++i) {
                    if (x[i] <= 1e-12) { degenerate = true; break; }
                    np.intercepts[i] = 1.0 / x[i];
                }
            }
        }
        if (degenerate) {
            // Min-max fallback (§3.2): a_i = z_i^max − z_i^min over F1.
            std::vector<double> nadir(m, -std::numeric_limits<double>::max());
            for (int v : F1) {
                const auto& o = vault.objectives_of(v);
                for (int j = 0; j < m; ++j) nadir[j] = std::max(nadir[j], o[j]);
            }
            for (int j = 0; j < m; ++j)
                np.intercepts[j] = std::max(nadir[j] - np.zmin[j], 1e-12);
        }
        return np;
    }

    // Compute normalised objective vector for a vault slot.
    std::vector<double> normalise(DataVault<Ind_t>& vault, int v,
                                  const NormParams& np) const {
        int m = vault.objs_n();
        const auto& o = vault.objectives_of(v);
        std::vector<double> fn(m);
        for (int j = 0; j < m; ++j) {
            double denom = np.intercepts[j];
            fn[j] = (std::abs(denom) > 1e-12) ? (o[j] - np.zmin[j]) / denom : 0.0;
            fn[j] = std::max(fn[j], 0.0);   // clamp ≥ 0
        }
        return fn;
    }

    // ── Lp norm ────────────────────────────────────────────────────────────
    double lp_norm(const std::vector<double>& v, double p) const {
        double s = 0.0;
        for (double x : v) s += std::pow(std::max(x, 0.0), p);
        return std::pow(s, 1.0 / p);
    }

    double lp_dist(const std::vector<double>& a,
                   const std::vector<double>& b, double p) const {
        double s = 0.0;
        int m = static_cast<int>(a.size());
        for (int j = 0; j < m; ++j) s += std::pow(std::abs(a[j] - b[j]), p);
        return std::pow(s, 1.0 / p);
    }

    // ── Geometry estimation (§3.3, Eq. 6 + Eq. 8) ──────────────────────────
    // Central point C (Eq.6): the F1 solution with the MINIMUM perpendicular
    // distance to the bisector beta (from Z^min=0 to Z^max=1):
    //   dist⊥(f, β) = ‖f‖·sin θ = √(‖f‖² − (f·β̂)²) = √(‖f‖² − (Σf_i)²/M).
    // (Taking argmax cos theta — the minimum ANGLE — is not equivalent: the
    //  minimum angle favours points with a large norm, whereas Eq.6 favours a
    //  small one.)
    // p = log(M) / (log(M) - log(Σ C_i))   (Eq.8)
    double estimate_p(const std::vector<std::vector<double>>& fn_F1, int m) const {
        int best_idx = -1;
        double best_d = std::numeric_limits<double>::max();
        for (int i = 0; i < static_cast<int>(fn_F1.size()); ++i) {
            double sum = 0.0, norm2 = 0.0;
            for (int j = 0; j < m; ++j) {
                sum   += fn_F1[i][j];
                norm2 += fn_F1[i][j] * fn_F1[i][j];
            }
            if (norm2 < 1e-28) continue;   // point at the ideal: degenerate
            double d2 = norm2 - (sum * sum) / static_cast<double>(m);
            double dperp = std::sqrt(std::max(d2, 0.0));
            if (dperp < best_d) { best_d = dperp; best_idx = i; }
        }
        if (best_idx < 0) return 1.0;   // fallback: linear

        const auto& C = fn_F1[best_idx];
        double sum_C = 0.0;
        for (int j = 0; j < m; ++j) sum_C += C[j];
        // Guard: sum_C must be in (0, M) for log to be valid.
        sum_C = std::max(sum_C, 1e-10);
        sum_C = std::min(sum_C, static_cast<double>(m) - 1e-10);
        double log_M    = std::log(static_cast<double>(m));
        double log_sumC = std::log(sum_C);
        double p = log_M / (log_M - log_sumC);
        // Clamp to reasonable range.
        return std::max(0.1, std::min(p, 20.0));
    }

    // ── Survival score (Algorithm 2) ───────────────────────────────────────
    // Assigns vault.get_ind(v).survival_score for all v in front.
    void survival_score(DataVault<Ind_t>& vault,
                        const std::vector<int>& front,
                        int d, double p,
                        const NormParams& np) {
        int m = vault.objs_n();
        int sz = static_cast<int>(front.size());

        // Compute normalised objectives for this front.
        std::vector<std::vector<double>> fn(sz);
        for (int i = 0; i < sz; ++i) fn[i] = normalise(vault, front[i], np);

        if (d == 1) {
            // ── First front: full survival score algorithm ─────────────────

            // Extreme points: for each axis i, find solution with max f^n_i.
            std::vector<bool> is_extreme(sz, false);
            for (int i = 0; i < m; ++i) {
                int best = 0;
                for (int k = 1; k < sz; ++k)
                    if (fn[k][i] > fn[best][i]) best = k;
                if (!is_extreme[best]) {
                    is_extreme[best] = true;
                    vault.get_ind(front[best]).survival_score =
                        std::numeric_limits<double>::infinity();
                }
            }

            // Proximity for all.
            std::vector<double> prox(sz);
            for (int i = 0; i < sz; ++i) prox[i] = lp_norm(fn[i], p);

            // Pairwise Lp distances.
            std::vector<std::vector<double>> dist(sz, std::vector<double>(sz, 0.0));
            for (int i = 0; i < sz; ++i)
                for (int j = i + 1; j < sz; ++j)
                    dist[i][j] = dist[j][i] = lp_dist(fn[i], fn[j], p);

            // Σ = extreme, Ω = rest.
            std::vector<int> sigma, omega;
            for (int i = 0; i < sz; ++i)
                if (is_extreme[i]) sigma.push_back(i); else omega.push_back(i);

            // Greedy selection loop.
            while (!omega.empty()) {
                int best_idx = -1;
                double best_val = -std::numeric_limits<double>::max();

                for (int oi : omega) {
                    // diversity = min + min2 distances from oi to sigma.
                    double min1 = std::numeric_limits<double>::max();
                    double min2 = std::numeric_limits<double>::max();
                    for (int si : sigma) {
                        double d_val = dist[oi][si];
                        if (d_val < min1) { min2 = min1; min1 = d_val; }
                        else if (d_val < min2) { min2 = d_val; }
                    }
                    double diversity = min1 + min2;
                    double prox_oi   = std::max(prox[oi], 1e-14);
                    double value     = diversity / prox_oi;
                    if (value > best_val) { best_val = value; best_idx = oi; }
                }
                if (best_idx < 0) break;

                vault.get_ind(front[best_idx]).survival_score = best_val;
                sigma.push_back(best_idx);
                omega.erase(std::find(omega.begin(), omega.end(), best_idx));
            }
        } else {
            // ── Dominated fronts: score = 1 / proximity ───────────────────
            for (int i = 0; i < sz; ++i) {
                double prox = std::max(lp_norm(fn[i], p), 1e-14);
                vault.get_ind(front[i]).survival_score = 1.0 / prox;
            }
        }
    }

    // ── Binary tournament ─────────────────────────────────────────────────
    // Rank first, then survival score (higher wins).
    int tournament(DataVault<Ind_t>& vault,
                   std::uniform_int_distribution<int>& dist) {
        int a = dist(rng_), b = dist(rng_);
        int ra = vault.get_ind(a).rank, rb = vault.get_ind(b).rank;
        if (ra != rb) return (ra < rb) ? a : b;
        double sa = vault.get_ind(a).survival_score;
        double sb = vault.get_ind(b).survival_score;
        return (sa >= sb) ? a : b;
    }

    // ── Rearrange vault ────────────────────────────────────────────────────
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
            pos[want] = i;    pos[other] = cur;
            at_pos[i] = want; at_pos[cur] = other;
        }
        vault.reduce(n);
    }

public:
    AGEMOEACore() = default;

    void set_eta_crossover(double e)  { eta_c_ = e; }
    void set_eta_mutation (double e)  { eta_m_ = e; }
    void set_pc           (double p)  { pc_ = p; }
    void set_seed         (unsigned s){ rng_.seed(s); }

    // ── setup ──────────────────────────────────────────────────────────────
    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dr(0.0, 1.0);
        std::uniform_int_distribution<int>     db(0, 1);
        std::vector<double> vars (vault.vars_n());
        std::vector<int>    bvars(vault.bin_vars_n());
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < vault.vars_n(); ++j) {
                double lo = bounds[j].first .value_or(0.0);
                double hi = bounds[j].second.value_or(1.0);
                vars[j] = lo + dr(rng_) * (hi - lo);
            }
            for (int j = 0; j < vault.bin_vars_n(); ++j) bvars[j] = db(rng_);
            if (vault.bin_vars_n() > 0) vault.set_all_variables(i, vars, bvars);
            else                        vault.set_variables    (i, vars);
        }
        vault.sync();
        // Initial ranks/scores.
        auto fronts = fast_nondominated_sort(vault, n);
        for (int i = 0; i < n; ++i)
            vault.get_ind(i).survival_score = 0.0;
        (void)fronts;
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        auto fronts = fast_nondominated_sort(vault, n);
        for (int i = 0; i < n; ++i) vault.get_ind(i).survival_score = 0.0;
        (void)fronts;
    }

    // ── step: one full generation ──────────────────────────────────────────
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int m = vault.objs_n();
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_int(0, n - 1);

        // ── expand: active = 2n ────────────────────────────────────────────
        int off_base = vault.expand(n);  // [off_base, off_base+n) = offspring slots

        // ── breed n offspring into [n, 2n) ────────────────────────────────
        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        for (int i = 0; i < n; i += 2) {
            int p1 = tournament(vault, dist_int);
            int p2 = tournament(vault, dist_int);
            for (int j = 0; j < vault.vars_n(); ++j) {
                pv1[j] = vault.get_variable(p1, j);
                pv2[j] = vault.get_variable(p2, j);
            }
            // Table 1: pc=1, pm=1/n
            double pm = (vault.vars_n() > 0) ? 1.0 / vault.vars_n() : 0.0;
            ops::sbx(pv1, pv2, c1, c2, bounds, eta_c_, pc_, rng_);
            ops::polynomial_mutation(c1, bounds, eta_m_, pm, rng_);
            ops::polynomial_mutation(c2, bounds, eta_m_, pm, rng_);
            if (vault.bin_vars_n() > 0) {
                std::vector<int> bv1(vault.bin_vars_n()), bv2(vault.bin_vars_n()),
                                 bc1, bc2;
                for (int j = 0; j < vault.bin_vars_n(); ++j) {
                    bv1[j] = vault.get_bin_variable(p1, j);
                    bv2[j] = vault.get_bin_variable(p2, j);
                }
                ops::binary_crossover(bv1, bv2, bc1, bc2, rng_);
                ops::bit_flip_mutation(bc1, vault.bin_vars_n(), rng_);
                if (i + 1 < n) ops::bit_flip_mutation(bc2, vault.bin_vars_n(), rng_);
                vault.set_all_variables(off_base + i,     c1, bc1);
                if (i + 1 < n) vault.set_all_variables(off_base + i + 1, c2, bc2);
            } else {
                vault.set_variables(off_base + i, c1);
                if (i + 1 < n) vault.set_variables(off_base + i + 1, c2);
            }
        }

        // ── evaluate offspring ─────────────────────────────────────────────
        vault.sync();

        // ── non-dominated sort of 2n combined pool ─────────────────────────
        auto fronts = fast_nondominated_sort(vault, n * 2);

        // ── normalise from F1 (first front of combined pool) ───────────────
        NormParams np = compute_norm_params(vault, fronts[0]);

        // ── estimate geometry p from normalised F1 ─────────────────────────
        std::vector<std::vector<double>> fn_F1(fronts[0].size());
        for (std::size_t i = 0; i < fronts[0].size(); ++i)
            fn_F1[i] = normalise(vault, fronts[0][i], np);
        double p_geom = estimate_p(fn_F1, m);

        // ── greedy front acceptance + survival score ───────────────────────
        std::vector<int> survivors;
        survivors.reserve(n);

        int d = 0;
        while (d < static_cast<int>(fronts.size())) {
            int sz = static_cast<int>(fronts[d].size());
            if (static_cast<int>(survivors.size()) + sz <= n) {
                // Whole front fits: score it and accept.
                survival_score(vault, fronts[d], d + 1, p_geom, np);
                for (int v : fronts[d]) survivors.push_back(v);
                ++d;
                if (static_cast<int>(survivors.size()) == n) break;
            } else {
                // Partial front: score and sort descending, take first K.
                survival_score(vault, fronts[d], d + 1, p_geom, np);
                std::vector<int> sorted_front = fronts[d];
                std::sort(sorted_front.begin(), sorted_front.end(),
                    [&](int a, int b) {
                        return vault.get_ind(a).survival_score
                             > vault.get_ind(b).survival_score;
                    });
                int K = n - static_cast<int>(survivors.size());
                for (int k = 0; k < K; ++k) survivors.push_back(sorted_front[k]);
                break;
            }
        }

        // ── move survivors to [0, n), reduce ──────────────────────────────
        rearrange(vault, survivors, n * 2);
    }
};

} // namespace mootation
