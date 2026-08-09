#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// NSGA-III — An Evolutionary Many-Objective Optimization Algorithm Using
//            Reference-Point-Based Nondominated Sorting Approach, Part I
// K. Deb, H. Jain — IEEE TEVC 18(4), 2014
// doi:10.1109/TEVC.2013.2281535          (source: deb2014)
// Constrained mode (CDP, tournament): Part II — H. Jain, K. Deb,
// IEEE TEVC 18(4), 2014, doi:10.1109/TEVC.2013.2281534 (source: jain2014).
//
// Generation scheme (Alg.1):
//   1. Q_t: parents drawn at random from P_t (§IV-F, no tournament; with
//      constraints — the CV tournament of Part II Alg.1), SBX (eta_c=30, pc=1)
//      + PM (eta_m=20, pm=1/n)
//   2. R_t = P_t ∪ Q_t → fast non-dominated sort; accept fronts until |S_t| ≥ N
//   3. Normalize (Alg.2, §IV-C): historical z^min over ∪_τ S_τ; extreme
//      points — min ASF, accumulated "ever found from the start";
//      hyperplane intercepts a_i; f^n = (f − z^min)/a (Eq.4);
//      on degeneracy/a_i ≤ 0 — per-objective fallback to nadir − z^min
//      ("Special care … nonnegative intercepts", §IV-C)
//   4. Associate (Alg.3): min perpendicular distance to the reference lines
//   5. Niching (Alg.4): K picks from F_l by argmin ρ_j (random tie-break)
//
// Defaults = Tables I-II: eta_c=30, eta_m=20, pc=1.0, pm=1/n.
// Deviations:
//   - Path-A: pop_size must exactly equal the size of the Das-Dennis lattice
//     (paper: N = nearest multiple of 4 > H — the 92/212/... configurations
//     of Table I are literally unattainable; a deliberate decision);
//   - set_reference_points: user-supplied (aspiration) points are used as
//     is — the mapping onto the normalized hyperplane via Eq.4
//     (Alg.2 lines 8-9) is not performed; supply points on the unit simplex.
// FIX 2026-07-08 (internal audit — consistency with a_nsga3.hpp; the changes
//   manifest ONLY with constraint_mode≠NONE / user points, the default
//   unconstrained path is NOT changed):
//   (a) update_norm_state in constrained mode now updates
//       z^min/extremes ONLY over feasible solutions (§IV-C). With
//       constraint_mode==NONE the filter is a no-op (Sf==St);
//   (b) set_reference_points with USER-SUPPLIED reference points adds M
//       extreme points, one on each axis (§VI-A). The default
//       Das-Dennis generation already contains the axis unit vectors (lattice
//       vertices) — no duplication there.
// Extensions beyond the paper: constraint_mode FEASIBILITY ≡ CDP (off by
//   default; CDP — Part II §III-A); mixed real+binary genome.
//
// Individual: NSGAIII_Individual (individuals.hpp): rank, ref_point_idx,
//   niche_count, norm_distance.
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
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

template <typename Ind_t>
class NSGAIIICore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    // ── hyperparameters ────────────────────────────────────────────────────
    double       eta_c_ = 30.0;   // SBX distribution index (paper Table II)
    double       eta_m_ = 20.0;   // polynomial mutation distribution index
    double       pc_    = 1.0;    // SBX crossover probability (paper Table II)
    std::mt19937 rng_{std::random_device{}()};

    // Reference points on the unit hyperplane, shape [H][m].
    std::vector<std::vector<double>> ref_points_;

    // ── Persistent normalization state (§IV-C) ────────────────────────────
    // z^min — over ∪_{τ=0}^t S_τ ("in ⋃ S_τ"); extreme points —
    // "extreme points ever found from the start of the simulation".
    std::vector<double>              zmin_hist_;     // historical ideal point
    std::vector<std::vector<double>> extreme_hist_;  // [m] historical F-vectors
                                                     // of extremes (empty = none)

    // ── Das-Dennis reference point generation ─────────────────────────────
    void generate_reference_points(int n, int m) {
        ref_points_ = das_dennis::generate_exact(m, n);
    }

    // ── Dominance helpers ──────────────────────────────────────────────────
    bool dominates_plain(const std::vector<double>& a,
                         const std::vector<double>& b) const {
        bool better = false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (a[i] > b[i]) return false;
            if (a[i] < b[i]) better = true;
        }
        return better;
    }

    bool dominates_cdp(const std::vector<double>& fa, double cva,
                       const std::vector<double>& fb, double cvb) const {
        bool af = (cva <= 0.0), bf = (cvb <= 0.0);
        if ( af && !bf) return true;
        if (!af &&  bf) return false;
        if (!af && !bf) return cva < cvb;
        return dominates_plain(fa, fb);
    }

    // ── Fast non-dominated sort ────────────────────────────────────────────
    // Returns fronts as vault-index vectors; sets ind.rank for each solution.
    std::vector<std::vector<int>>
    fast_nondominated_sort(DataVault<Ind_t>& vault, int n) {
        std::vector<double> cvs(n, 0.0);
        if (constraint_mode == ConstraintMode::CDP ||
            constraint_mode == ConstraintMode::FEASIBILITY)
            for (int i = 0; i < n; ++i) cvs[i] = vault.get_cv(i);

        std::vector<std::vector<int>> S(n);
        std::vector<int>              np(n, 0);
        for (int i = 0; i < n; ++i) {
            const auto& fi = vault.objectives_of(i);
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                const auto& fj = vault.objectives_of(j);
                bool i_dom_j = (constraint_mode == ConstraintMode::CDP ||
                                constraint_mode == ConstraintMode::FEASIBILITY)
                               ? dominates_cdp(fi, cvs[i], fj, cvs[j])
                               : dominates_plain(fi, fj);
                bool j_dom_i = (constraint_mode == ConstraintMode::CDP ||
                                constraint_mode == ConstraintMode::FEASIBILITY)
                               ? dominates_cdp(fj, cvs[j], fi, cvs[i])
                               : dominates_plain(fj, fi);
                if (i_dom_j) S[i].push_back(j);
                else if (j_dom_i) ++np[i];
            }
            if (np[i] == 0) vault.get_ind(i).rank = 0;
        }
        std::vector<std::vector<int>> fronts;
        std::vector<int> f0;
        for (int i = 0; i < n; ++i) if (np[i] == 0) f0.push_back(i);
        fronts.push_back(f0);
        int k = 0;
        while (!fronts[k].empty()) {
            std::vector<int> next;
            for (int i : fronts[k])
                for (int j : S[i])
                    if (--np[j] == 0) {
                        vault.get_ind(j).rank = k + 1;
                        next.push_back(j);
                    }
            fronts.push_back(next);
            ++k;
        }
        fronts.pop_back();
        return fronts;
    }

    // ── Update of the persistent normalization state (§IV-C) ──────────────
    // z^min — minimum over ∪_τ S_τ (historical); extreme points — min ASF
    // among {previously found extremes} ∪ S_t ("ever found from the start").
    // ASF(s, w^i) = max_j f'_j(s)/w^i_j, w^i_j = 1 (j=i), ε=1e-6 otherwise
    // (the paper does not specify ε; 1e-6 is the reference-code standard).
    // FIX 2026-07-08: in constrained mode the persistent
    //   z^min/extremes are updated ONLY over feasible solutions (jain2014
    //   §III-A, "we then update the population ideal (z^min)
    //   and nadir points (z^max) using the objective values of feasible
    //   solutions"). Same feasibility filter as in a_nsga3.hpp (the reference
    //   implementation). In unconstrained mode (constraint_mode==NONE →
    //   use_cv=false) the filter is a no-op: Sf==St in full, prior behavior.
    void update_norm_state(DataVault<Ind_t>& vault,
                           const std::vector<int>& St) {
        int m = vault.objs_n();
        if (zmin_hist_.empty())
            zmin_hist_.assign(m, std::numeric_limits<double>::max());
        if (extreme_hist_.empty())
            extreme_hist_.assign(m, {});

        // Feasible subset of St (with no constraints — all of St).
        bool use_cv = (constraint_mode == ConstraintMode::CDP ||
                       constraint_mode == ConstraintMode::FEASIBILITY);
        std::vector<int> Sf;
        Sf.reserve(St.size());
        for (int v : St)
            if (!use_cv || vault.get_cv(v) <= 0.0) Sf.push_back(v);
        // All solutions infeasible → do not distort history (no feasible signal).
        if (Sf.empty()) return;

        for (int v : Sf) {
            const auto& o = vault.objectives_of(v);
            for (int j = 0; j < m; ++j)
                zmin_hist_[j] = std::min(zmin_hist_[j], o[j]);
        }

        const double eps = 1e-6;
        auto asf = [&](const std::vector<double>& o, int axis) {
            double val = -std::numeric_limits<double>::max();
            for (int j = 0; j < m; ++j) {
                double w = (axis == j) ? 1.0 : eps;
                val = std::max(val, (o[j] - zmin_hist_[j]) / w);
            }
            return val;
        };
        for (int i = 0; i < m; ++i) {
            std::vector<double> best = extreme_hist_[i];
            double best_asf = best.empty() ? std::numeric_limits<double>::max()
                                           : asf(best, i);
            for (int v : Sf) {
                const auto& o = vault.objectives_of(v);
                double a = asf(o, i);
                if (a < best_asf) { best_asf = a; best = o; }
            }
            extreme_hist_[i] = std::move(best);
        }
    }

    // ── Normalisation (paper §IV-C, Alg.2) ─────────────────────────────────
    // 1. z^min — the historical ideal point (update_norm_state).
    // 2. Extreme points — historical (update_norm_state).
    // 3. Hyperplane through the M extremes → intercepts a_i.
    //    "Special care is taken to handle degenerate cases and nonnegative
    //    intercepts": degenerate system OR a_i ≤ 0 → per-objective fallback
    //    to nadir_i − z^min_i (nadir over the current St).
    // 4. f^n_i(s) = (f_i(s) − z^min_i) / a_i   (Eq.4; a_i is already stored in
    //    translated coordinates — z^min is not subtracted a second time).
    std::vector<std::vector<double>>
    normalise(DataVault<Ind_t>& vault,
              const std::vector<int>& St) {
        int m = vault.objs_n();
        int sz = static_cast<int>(St.size());

        update_norm_state(vault, St);
        const std::vector<double>& zstar = zmin_hist_;

        // Intercepts: solve the system f'(extreme_i) · (1/a) = 1.
        // Build matrix A[m][m] where A[i][j] = f_j(extreme_i) - z*_j.
        // Then A · x = ones, a_i = 1/x_i.
        std::vector<double> intercepts(m, -1.0);   // ≤0 → fallback on this axis
        bool degenerate = false;
        for (int i = 0; i < m; ++i)
            if (extreme_hist_[i].empty()) { degenerate = true; break; }

        if (!degenerate) {
            // Gaussian elimination to solve A * x = ones.
            std::vector<std::vector<double>> A(m, std::vector<double>(m + 1));
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < m; ++j)
                    A[i][j] = extreme_hist_[i][j] - zstar[j];
                A[i][m] = 1.0;
            }
            // Forward elimination.
            for (int col = 0; col < m; ++col) {
                // Pivot.
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
            if (!degenerate) {
                // Back substitution.
                std::vector<double> x(m);
                for (int i = m - 1; i >= 0; --i) {
                    x[i] = A[i][m];
                    for (int j = i + 1; j < m; ++j) x[i] -= A[i][j] * x[j];
                    x[i] /= A[i][i];
                }
                // intercept a_i = 1/x_i (= the Eq.4 denominator in translated
                // coordinates); x_i ≤ 0 (negative/zero intercept) is kept ≤ 0 —
                // that axis goes to the per-objective fallback below.
                for (int i = 0; i < m; ++i)
                    intercepts[i] = (x[i] > 1e-12) ? (1.0 / x[i]) : -1.0;
            }
        }

        // Per-objective fallback (§IV-C "nonnegative intercepts"):
        // degenerate system — all axes; otherwise only axes with a_i ≤ 0.
        {
            std::vector<double> nadir(m, -std::numeric_limits<double>::max());
            for (int v : St) {
                const auto& o = vault.objectives_of(v);
                for (int j = 0; j < m; ++j) nadir[j] = std::max(nadir[j], o[j]);
            }
            for (int j = 0; j < m; ++j)
                if (degenerate || intercepts[j] <= 1e-12)
                    intercepts[j] = std::max(nadir[j] - zstar[j], 1e-12);
        }

        // Build normalised objectives (Eq.4).
        std::vector<std::vector<double>> fn(sz, std::vector<double>(m));
        for (int si = 0; si < sz; ++si) {
            const auto& o = vault.objectives_of(St[si]);
            for (int j = 0; j < m; ++j)
                fn[si][j] = (o[j] - zstar[j]) / intercepts[j];
        }
        return fn;
    }

    // ── Association ─────────────────────────────────────────────────────────
    // For each solution in St (indexed by St_idx), find the reference line
    // (from origin through ref_points_[r]) with minimum perpendicular distance.
    //
    // d(s, line_r) = ||f'(s) - proj_r(f'(s))||
    //   proj_r(f') = (f' · r / ||r||²) * r
    //
    // Sets vault.get_ind(St[si]).ref_point_idx and norm_distance.
    void associate(DataVault<Ind_t>& vault,
                   const std::vector<int>& St,
                   const std::vector<std::vector<double>>& fn) {
        int m = vault.objs_n();
        int nref = static_cast<int>(ref_points_.size());
        int sz   = static_cast<int>(St.size());

        for (int si = 0; si < sz; ++si) {
            double best_d   = std::numeric_limits<double>::max();
            int    best_ref = 0;
            for (int r = 0; r < nref; ++r) {
                const auto& rp = ref_points_[r];
                // Squared norm of reference point vector.
                double rr = 0.0;
                for (int j = 0; j < m; ++j) rr += rp[j] * rp[j];
                if (rr < 1e-14) continue;
                // Projection scalar: (f' · r) / ||r||²
                double dot = 0.0;
                for (int j = 0; j < m; ++j) dot += fn[si][j] * rp[j];
                double t = dot / rr;
                // Perpendicular distance squared.
                double d2 = 0.0;
                for (int j = 0; j < m; ++j) {
                    double diff = fn[si][j] - t * rp[j];
                    d2 += diff * diff;
                }
                double d = std::sqrt(d2);
                if (d < best_d) { best_d = d; best_ref = r; }
            }
            vault.get_ind(St[si]).ref_point_idx = best_ref;
            vault.get_ind(St[si]).norm_distance  = best_d;
        }
    }

    // ── Niche-preservation (niching) ─────────────────────────────────────
    // Selects K solutions from Fl to complete Pt+1.
    // chosen: output vault indices selected from Fl.
    void niching(DataVault<Ind_t>& vault,
                 int K,
                 const std::vector<int>& Fl,
                 const std::vector<int>& St_minus_Fl,
                 std::vector<int>& chosen) {
        int nref = static_cast<int>(ref_points_.size());

        // ρ_j = niche count based on St \ Fl.
        std::vector<int> rho(nref, 0);
        for (int v : St_minus_Fl)
            ++rho[vault.get_ind(v).ref_point_idx];

        // Working copy of Fl (remaining candidates).
        std::vector<int> fl_rem = Fl;

        for (int pick = 0; pick < K; ++pick) {
            if (fl_rem.empty()) break;

            // Find minimum niche count among reference points that have
            // at least one associated solution in fl_rem.
            std::vector<int> refs_with_fl;
            for (int r = 0; r < nref; ++r) {
                bool has = false;
                for (int v : fl_rem)
                    if (vault.get_ind(v).ref_point_idx == r) { has = true; break; }
                if (has) refs_with_fl.push_back(r);
            }
            if (refs_with_fl.empty()) break;

            int min_rho = std::numeric_limits<int>::max();
            for (int r : refs_with_fl) min_rho = std::min(min_rho, rho[r]);

            // All references with that minimum rho (and having Fl members).
            std::vector<int> candidates;
            for (int r : refs_with_fl)
                if (rho[r] == min_rho) candidates.push_back(r);

            // Pick one at random.
            std::uniform_int_distribution<int> dist_c(
                0, static_cast<int>(candidates.size()) - 1);
            int j_star = candidates[dist_c(rng_)];

            // Find associated Fl member(s).
            std::vector<int> assoc;
            for (int v : fl_rem)
                if (vault.get_ind(v).ref_point_idx == j_star) assoc.push_back(v);

            // Select: if ρ_j* == 0 pick closest (min norm_distance); else random.
            int selected;
            if (rho[j_star] == 0) {
                selected = *std::min_element(assoc.begin(), assoc.end(),
                    [&](int a, int b) {
                        return vault.get_ind(a).norm_distance
                             < vault.get_ind(b).norm_distance;
                    });
            } else {
                std::uniform_int_distribution<int> dist_a(
                    0, static_cast<int>(assoc.size()) - 1);
                selected = assoc[dist_a(rng_)];
            }

            chosen.push_back(selected);
            ++rho[j_star];
            fl_rem.erase(std::find(fl_rem.begin(), fl_rem.end(), selected));
        }
    }

    // ── Binary tournament (NSGA-III variant) ─────────────────────────────
    // When both feasible: random selection (paper Part II §III-B).
    // When one feasible and other not: feasible wins.
    // When both infeasible: smaller CV wins.
    int tournament(DataVault<Ind_t>& vault,
                   std::uniform_int_distribution<int>& dist) {
        int a = dist(rng_), b = dist(rng_);
        if (constraint_mode == ConstraintMode::CDP ||
            constraint_mode == ConstraintMode::FEASIBILITY) {
            double cva = vault.get_cv(a), cvb = vault.get_cv(b);
            bool af = (cva <= 0.0), bf = (cvb <= 0.0);
            if ( af && !bf) return a;
            if (!af &&  bf) return b;
            if (!af && !bf) {
                // Part II Alg.1 lines 6-9: smaller CV; lines 10-11: equal CV — random.
                if (cva < cvb) return a;
                if (cvb < cva) return b;
                std::uniform_int_distribution<int> coin(0, 1);
                return coin(rng_) ? a : b;
            }
            // Both feasible: random.
            std::uniform_int_distribution<int> coin(0, 1);
            return coin(rng_) ? a : b;
        }
        // No constraint handling: random between any two (NSGA-III uses
        // no tournament pressure on feasible population).
        std::uniform_int_distribution<int> coin(0, 1);
        return coin(rng_) ? a : b;
    }

    // ── Rearrange vault to place survivors[i] in slot i ───────────────────
    void rearrange(DataVault<Ind_t>& vault,
                   const std::vector<int>& survivors, int pool_size) {
        int n = static_cast<int>(survivors.size());
        std::vector<int> pos(pool_size), at_pos(pool_size);
        std::iota(pos.begin(), pos.end(), 0);
        std::iota(at_pos.begin(), at_pos.end(), 0);
        for (int i = 0; i < n; ++i) {
            int want = survivors[i];
            int cur  = pos[want];
            if (cur == i) continue;
            int other = at_pos[i];
            vault.swap_active(i, cur);
            pos[want] = i;    pos[other] = cur;
            at_pos[i] = want; at_pos[cur] = other;
        }
        vault.reduce(n);
    }

public:
    NSGAIIICore() = default;

    void set_eta_crossover(double e)  { eta_c_ = e; }
    void set_eta_mutation (double e)  { eta_m_ = e; }
    void set_pc           (double p)  { pc_ = p; }
    void set_seed         (unsigned s){ rng_.seed(s); }

    // Allow user to supply custom reference points (optional).
    // WARNING (see file header): the points are used as is, the mapping of
    // aspiration points onto the normalized hyperplane (Alg.2 lines 8-9)
    // is not performed — supply points on the unit simplex.
    // FIX 2026-07-08: with USER-SUPPLIED (preferred)
    //   reference points we add M extreme points
    //   (1,0,…,0)^T,(0,1,…,0)^T,… following a_nsga3.hpp (the reference
    //   implementation) ("by default we also supply M
    //   additional reference points, one at each objective axis at intercept
    //   unity"; §VI-A). The extreme points are needed for a correct
    //   ideal/nadir estimate during normalization. Duplicates (the user
    //   already supplied a vertex) are not added (tol=1e-9). The default
    //   Das-Dennis generation already contains the axis unit vectors (lattice
    //   vertices) — no duplication there (see generate_reference_points).
    void set_reference_points(std::vector<std::vector<double>> rp) {
        ref_points_ = std::move(rp);
        if (!ref_points_.empty()) {
            int m = static_cast<int>(ref_points_[0].size());
            const double tol = 1e-9;
            auto has_point = [&](const std::vector<double>& p) {
                for (const auto& q : ref_points_) {
                    if (static_cast<int>(q.size()) != m) continue;
                    bool same = true;
                    for (int i = 0; i < m; ++i)
                        if (std::abs(q[i] - p[i]) > tol) { same = false; break; }
                    if (same) return true;
                }
                return false;
            };
            for (int k = 0; k < m; ++k) {
                std::vector<double> ext(m, 0.0);
                ext[k] = 1.0;                       // (…,1,…): k-th simplex vertex
                if (!has_point(ext)) ref_points_.push_back(std::move(ext));
            }
        }
    }

    const std::vector<std::vector<double>>& reference_points() const {
        return ref_points_;
    }

    // ── setup ──────────────────────────────────────────────────────────────
    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int m = vault.objs_n();

        if (ref_points_.empty()) generate_reference_points(n, m);

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

        // Initial rank assignment.
        auto fronts = fast_nondominated_sort(vault, n);
        (void)fronts;
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int m = vault.objs_n();
        if (ref_points_.empty()) generate_reference_points(n, m);
        auto fronts = fast_nondominated_sort(vault, n);
        (void)fronts;
    }

    // ── step: one full generation ──────────────────────────────────────────
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_int(0, n - 1);

        // ── expand: active = 2n ────────────────────────────────────────────
        int off_base = vault.expand(n);  // [off_base, off_base+n) — offspring slots

        // ── breed n offspring into [n, 2n) ────────────────────────────────
        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        for (int i = 0; i < n; i += 2) {
            int p1 = tournament(vault, dist_int);
            int p2 = tournament(vault, dist_int);
            for (int j = 0; j < vault.vars_n(); ++j) {
                pv1[j] = vault.get_variable(p1, j);
                pv2[j] = vault.get_variable(p2, j);
            }
            // Table II: pc=1.0, pm=1/n
            double pm = (vault.vars_n() > 0) ? 1.0 / vault.vars_n() : 0.0;
            ops::sbx(pv1, pv2, c1, c2, bounds, eta_c_, pc_, rng_);
            ops::polynomial_mutation(c1, bounds, eta_m_, pm, rng_);
            ops::polynomial_mutation(c2, bounds, eta_m_, pm, rng_);
            if (vault.bin_vars_n() > 0) {
                std::vector<int> bv1(vault.bin_vars_n()), bv2(vault.bin_vars_n());
                std::vector<int> bc1, bc2;
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

        // ── non-dominated sort of Rt = Pt ∪ Qt ────────────────────────────
        auto fronts = fast_nondominated_sort(vault, n * 2);

        // ── greedily accept fronts until |St| ≥ n ─────────────────────────
        std::vector<int> St;           // accepted vault indices
        std::vector<int> Fl;           // last (partial) front
        for (auto& front : fronts) {
            int total = static_cast<int>(St.size() + front.size());
            if (total <= n) {
                for (int v : front) St.push_back(v);
                if (static_cast<int>(St.size()) == n) break;
            } else {
                Fl = front;
                break;
            }
        }

        std::vector<int> survivors;

        if (Fl.empty()) {
            // Exact fit or all fronts accepted. The normalization state
            // (z^min, extremes) is accumulated in these generations too (§IV-C).
            update_norm_state(vault, St);
            survivors = St;
        } else {
            // St\Fl already selected; pick K more from Fl.
            std::vector<int> St_minus_Fl = St;   // does not include Fl yet
            int K = n - static_cast<int>(St.size());

            // Assemble full St for normalisation.
            std::vector<int> St_full = St;
            for (int v : Fl) St_full.push_back(v);

            // Normalise.
            auto fn = normalise(vault, St_full);

            // Associate every member of St_full with a reference point.
            associate(vault, St_full, fn);

            // Run niche-preservation on Fl.
            std::vector<int> chosen;
            niching(vault, K, Fl, St_minus_Fl, chosen);

            survivors = St_minus_Fl;
            for (int v : chosen) survivors.push_back(v);
        }

        // ── move survivors to [0, n), reduce ──────────────────────────────
        rearrange(vault, survivors, n * 2);
    }
};

} // namespace mootation
