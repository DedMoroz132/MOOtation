#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A-NSGA-III — An Evolutionary Many-Objective Optimization Algorithm Using
//              Reference-Point-Based Nondominated Sorting Approach, Part II:
//              Handling Constraints and Extending to an Adaptive Approach
// H. Jain, K. Deb — IEEE TEVC 18(4), 2014
// doi:10.1109/TEVC.2013.2281534
// NSGA-III base (Part I): doi:10.1109/TEVC.2013.2281535.
//
// Generational scheme (= NSGA-III plus the adaptation of §VII):
//   1. Q_t: parents drawn at random from P_t (under constraints, the CV
//      tournament of Alg.1),
//      SBX (eta_c=30, pc=1) + PM (eta_m=20, pm=1/n)
//   2. R_t = P_t ∪ Q_t -> NDS (CDP) -> accept fronts, then
//      Normalize/Associate/Niching (historical z^min and extreme points, with a
//      per-objective fallback when a_i <= 0 — as in nsga3.hpp)
//   3. Reference-point adaptation over P_{t+1} (§VII; feasible members only,
//      per §III-A):
//      Addition — around every point with rho_j >= 2, a sub-simplex of M points
//      whose inter-point distance equals the step of the original lattice
//      (Fig.25); checks: (i) outside the first quadrant is REJECTED,
//      (ii) a duplicate is REJECTED; cooldown: a point is not operated on again
//      until every original point has had a chance (§VII-A).
//      Deletion — added (non-original) points with rho_j = 0 are removed;
//      original points are always kept (§VII-B).
//
// PAPER DEFAULTS (Part I, Tables I-II): eta_c=30, eta_m=20, pc=1.0, pm=1/n.
// DECLARED DEVIATIONS:
//   - Path-A: pop_size = the Das-Dennis lattice size (see nsga3.hpp);
//   - deletion is NOT gated on the paper's "exactly N points with rho_j = 1"
//     perfect-scenario condition. §VII-B opens by tying the step to the
//     UPDATED niche counts, and its literal gate is nearly unsatisfiable
//     (Σρ_j = N always, so it demands a perfectly even spread); the
//     unconditional reading is the one every reference implementation uses.
//     Deletion here runs every generation, including those in which nothing
//     was added — note that the paper's own "perfect scenario" is exactly the
//     case where no point is crowded and therefore nothing IS added, so a
//     gate on additions would skip the one situation §VII-B names;
//   - cooldown reset: the inclusion flags are cleared once ALL original points
//     have been operated on (a reading of "have a chance"). This is STRICTER
//     than the paper's wording: the flag is set only on the rho_j >= 2 path, so
//     an original point that is never crowded blocks the global reset — each
//     original then fires inclusion at most once per run, and further
//     refinement proceeds outward through the newly added (unflagged) points,
//     which the addition loop rescans every generation;
//   - set_reference_points does not project the aspiration points through
//     Eq.4 — supply points already on the unit simplex; the M extreme points
//     (1,0,...,0)^T, ... are appended per §VI.
// EXTENSIONS BEYOND THE PAPER: a safety cap on |Z| (the paper sets no limit;
//   it rarely triggers thanks to checks (i)-(ii), the cooldown and deletion).
//   The cap is 2·max(N, |Z_0|), where |Z_0| is the initial reference set AFTER
//   the M extreme points of §VI are appended — set_reference_points runs before
//   setup() and has no access to pop_size, so at that moment it can only scale
//   by |Z_0|; setup() then re-arms it with the real N. Also:
//   constraint_mode FEASIBILITY == CDP; mixed real+binary genome.
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

struct ANSGA3_Individual : public Based_Individual {
    int    rank          = 0;
    int    ref_point_idx = 0;
    double norm_distance = 0.0;
    int    niche_count   = 0;
};

template <typename Ind_t>
class ANSGA3Core {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    // ── hyperparameters ────────────────────────────────────────────────────
    double       eta_c_ = 30.0;   // SBX distribution index (Part I Table II)
    double       eta_m_ = 20.0;   // polynomial mutation distribution index
    double       pc_    = 1.0;    // SBX crossover probability (Part I Table II)
    std::mt19937 rng_{std::random_device{}()};

    // Reference points on the unit hyperplane, shape [H][m] (adapted at runtime).
    std::vector<std::vector<double>> ref_points_;
    std::vector<char> is_original_;     // 1 for original points (lattice/user)
    std::vector<char> inclusion_done_;  // §VII-A cooldown: already operated on
    double spacing_ = 0.0;              // step of the original lattice (median NN)
    int    cap_ = 0;                    // safety cap on |ref_points_|

    // ── Persistent normalization state (Part I §IV-C) ─────────────────────
    std::vector<double>              zmin_hist_;     // historical ideal point
    std::vector<std::vector<double>> extreme_hist_;  // [m] historical F vectors

    // spacing_ = median nearest-neighbour distance among the original points.
    double compute_spacing(int m) const {
        std::vector<double> nn;
        int H = static_cast<int>(ref_points_.size());
        for (int i = 0; i < H; ++i) {
            double best = std::numeric_limits<double>::max();
            for (int j = 0; j < H; ++j) {
                if (i == j) continue;
                double d = 0.0;
                for (int k = 0; k < m; ++k) {
                    double t = ref_points_[i][k] - ref_points_[j][k];
                    d += t * t;
                }
                best = std::min(best, d);
            }
            if (best < std::numeric_limits<double>::max()) nn.push_back(std::sqrt(best));
        }
        if (nn.empty()) return 1.0 / std::max(1, m);
        std::sort(nn.begin(), nn.end());
        return nn[nn.size() / 2];
    }

    // Initializes the adaptation state. Also called for user-supplied points
    // from set_reference_points: without it, adapt would read is_original_ out
    // of bounds and produce duplicates when spacing_ = 0.
    void init_adaptation_state(int m, int pop_hint) {
        is_original_.assign(ref_points_.size(), 1);
        inclusion_done_.assign(ref_points_.size(), 0);
        spacing_ = compute_spacing(m);
        cap_ = 2 * std::max<int>(pop_hint, static_cast<int>(ref_points_.size()));
    }

    // ── Das-Dennis reference point generation ─────────────────────────────
    void generate_reference_points(int n, int m) {
        ref_points_ = das_dennis::generate_exact(m, n);
        init_adaptation_state(m, n);
    }

    // ── Reference-point adaptation (§VII): add-and-delete ─────────────────
    // Runs over P_{t+1}; normalization and association use the feasible members
    // (§III-A: «update … using the objective values of feasible solutions»).
    void adapt_reference_points(DataVault<Ind_t>& vault, int n) {
        int m = vault.objs_n();

        // P = feasible members of P_{t+1} (all of them when unconstrained).
        bool use_cv = (constraint_mode == ConstraintMode::CDP ||
                       constraint_mode == ConstraintMode::FEASIBILITY);
        std::vector<int> P;
        P.reserve(n);
        for (int i = 0; i < n; ++i)
            if (!use_cv || vault.get_cv(i) <= 0.0) P.push_back(i);
        if (P.empty()) return;   // no feasible members: adaptation is uninformative

        auto fn = normalise(vault, P);
        associate(vault, P, fn);
        int nref = static_cast<int>(ref_points_.size());
        std::vector<int> rho(nref, 0);
        for (int v : P) ++rho[vault.get_ind(v).ref_point_idx];

        // ── Addition (§VII-A) ──────────────────────────────────────────────
        // Around every crowded point (rho_j >= 2), a simplex of M points
        // (p=1) whose inter-point distance is spacing_ ("distance between them
        // same as the distance between two consecutive reference points on
        // the original simplex», Fig.25): p = z_j + (e_k − 𝟙/M)·spacing_/√2;
        // Sum p_i = Sum z_i = 1 is preserved by construction.
        // Checks: (i) outside the first quadrant -> the point is REJECTED;
        //         (ii) a duplicate of an existing or pending point -> REJECTED.
        // Cooldown: point j is not operated on again until every original point
        // has had a chance (the flags reset once all originals are done).
        const double tol = 1e-9;
        auto exists = [&](const std::vector<double>& p,
                          const std::vector<std::vector<double>>& pending) {
            auto eq = [&](const std::vector<double>& q) {
                for (int i = 0; i < m; ++i)
                    if (std::abs(p[i] - q[i]) > tol) return false;
                return true;
            };
            for (const auto& q : ref_points_) if (eq(q)) return true;
            for (const auto& q : pending)     if (eq(q)) return true;
            return false;
        };

        double off = spacing_ / std::sqrt(2.0);
        std::vector<std::vector<double>> added;
        for (int j = 0; j < nref; ++j) {
            if (rho[j] < 2) continue;
            if (inclusion_done_[j]) continue;   // §VII-A cooldown
            inclusion_done_[j] = 1;
            for (int k = 0; k < m; ++k) {
                std::vector<double> p = ref_points_[j];
                for (int i = 0; i < m; ++i)
                    p[i] += off * (((i == k) ? 1.0 : 0.0) - 1.0 / m);
                bool in_quadrant = true;
                for (int i = 0; i < m; ++i)
                    if (p[i] < 0.0) { in_quadrant = false; break; }
                if (!in_quadrant) continue;          // check (i): reject
                if (exists(p, added)) continue;      // check (ii): reject
                added.push_back(p);
            }
        }
        // Cooldown reset: every original point has had its chance.
        bool all_orig_done = true;
        for (int j = 0; j < nref; ++j)
            if (is_original_[j] && !inclusion_done_[j]) { all_orig_done = false; break; }
        if (all_orig_done)
            inclusion_done_.assign(inclusion_done_.size(), 0);

        for (auto& p : added) {
            ref_points_.push_back(std::move(p));
            is_original_.push_back(0);
            inclusion_done_.push_back(0);
        }

        // ── Deletion (§VII-B) ─────────────────────────────────────────────
        // Re-associate; added points with rho=0 are removed, original points
        // are always kept ("the original reference points are always kept").
        // Runs EVERY generation, including those in which nothing was added:
        // §VII-B ties the step to the updated niche counts, and its own literal
        // gate ("exactly N reference points with rho_j = 1") is precisely the
        // well-distributed case in which no point is crowded and therefore
        // nothing gets added — an early return there would skip the one
        // scenario the paper names. normalise/associate draw no RNG, so the
        // stream is unaffected.
        fn = normalise(vault, P);
        associate(vault, P, fn);
        nref = static_cast<int>(ref_points_.size());
        rho.assign(nref, 0);
        for (int v : P) ++rho[vault.get_ind(v).ref_point_idx];

        std::vector<std::vector<double>> keptW;
        std::vector<char> keptOrig, keptIncl;
        for (int j = 0; j < nref; ++j) {
            if (is_original_[j] || rho[j] > 0) {
                keptW.push_back(ref_points_[j]);
                keptOrig.push_back(is_original_[j]);
                keptIncl.push_back(inclusion_done_[j]);
            }
        }
        // Size safety cap (an extension beyond the paper, see the header):
        // drop surplus NON-original points from the end.
        while (static_cast<int>(keptW.size()) > cap_) {
            int idx = -1;
            for (int j = static_cast<int>(keptW.size()) - 1; j >= 0; --j)
                if (!keptOrig[j]) { idx = j; break; }
            if (idx < 0) break;
            keptW.erase(keptW.begin() + idx);
            keptOrig.erase(keptOrig.begin() + idx);
            keptIncl.erase(keptIncl.begin() + idx);
        }
        ref_points_.swap(keptW);
        is_original_.swap(keptOrig);
        inclusion_done_.swap(keptIncl);
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

    // ── Update of the persistent normalization state (Part I §IV-C) ───────
    // z^min is the minimum over the union of all S_tau; the extreme points are
    // the min-ASF members of {previously found extremes} u St ("ever found from
    // the start"). ASF: w^i_j = 1 when j=i, eps=1e-6 otherwise (the convention
    // of the reference code).
    // In constrained mode the persistent z^min and extreme points are updated
    //   ONLY from feasible solutions (jain2014 §III-A: "we then
    //   update the population ideal (z^min) and nadir points (z^max) using the
    //   objective values of feasible solutions"). This is the same CV filter
    //   already present in adapt_reference_points (see use_cv/get_cv). When
    //   unconstrained (use_cv=false) the filter is off, St is taken whole, and
    //   the behaviour is unchanged.
    void update_norm_state(DataVault<Ind_t>& vault,
                           const std::vector<int>& St) {
        int m = vault.objs_n();
        if (zmin_hist_.empty())
            zmin_hist_.assign(m, std::numeric_limits<double>::max());
        if (extreme_hist_.empty())
            extreme_hist_.assign(m, {});

        // Feasible subset of St (all of St when unconstrained).
        bool use_cv = (constraint_mode == ConstraintMode::CDP ||
                       constraint_mode == ConstraintMode::FEASIBILITY);
        std::vector<int> Sf;
        Sf.reserve(St.size());
        for (int v : St)
            if (!use_cv || vault.get_cv(v) <= 0.0) Sf.push_back(v);
        // Every solution infeasible: leave the history alone (no feasible signal).
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

    // ── Normalization (Part I §IV-C, Alg.2) — as in nsga3.hpp ──────────────
    // Historical z^min and extreme points; intercepts a_i; on degeneracy, or
    // when a_i <= 0, a per-objective fallback to nadir_i − z^min_i
    // («Special care … nonnegative intercepts»). f^n = (f − z^min)/a (Eq.4).
    std::vector<std::vector<double>>
    normalise(DataVault<Ind_t>& vault,
              const std::vector<int>& St) {
        int m = vault.objs_n();
        int sz = static_cast<int>(St.size());

        update_norm_state(vault, St);
        const std::vector<double>& zstar = zmin_hist_;

        std::vector<double> intercepts(m, -1.0);   // <=0 -> fallback on that axis
        bool degenerate = false;
        for (int i = 0; i < m; ++i)
            if (extreme_hist_[i].empty()) { degenerate = true; break; }

        if (!degenerate) {
            // Gaussian elimination to solve A * x = ones, a_i = 1/x_i.
            std::vector<std::vector<double>> A(m, std::vector<double>(m + 1));
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < m; ++j)
                    A[i][j] = extreme_hist_[i][j] - zstar[j];
                A[i][m] = 1.0;
            }
            // Forward elimination.
            for (int col = 0; col < m; ++col) {
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
                // x_i <= 0 (non-positive intercept) -> fall back on this axis.
                for (int i = 0; i < m; ++i)
                    intercepts[i] = (x[i] > 1e-12) ? (1.0 / x[i]) : -1.0;
            }
        }

        // Per-objective fallback (§IV-C «nonnegative intercepts»).
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

        // Build normalized objectives (Eq.4; the intercepts are already in
        // translated coordinates, so z^min is not subtracted a second time).
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
                // Alg.1 lines 6-9: smaller CV wins; lines 10-11: equal CV -> random.
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
    ANSGA3Core() = default;

    void set_eta_crossover(double e)  { eta_c_ = e; }
    void set_eta_mutation (double e)  { eta_m_ = e; }
    void set_pc           (double p)  { pc_ = p; }
    void set_seed         (unsigned s){ rng_.seed(s); }

    // Allow user to supply custom reference points (optional).
    // Initializes the adaptation state (is_original_ / spacing_ / cap_ /
    // cooldown). Without it, adapt_reference_points invoked undefined behaviour.
    // Points must already lie on the unit simplex (the Eq.4 projection is not
    // applied).
    // For USER-supplied (preferred) reference points, the M extreme points
    //   (1,0,...,0)^T, (0,1,...,0)^T, ... are appended (jain2014 §VI,
    //   «we include M extreme reference points … to make the
    //   normalization process to work well … make a total of |H_p|+M reference
    //   points"). The extreme points are needed for a correct ideal/nadir
    //   estimate. Duplicates — where the user already supplied a vertex — are
    //   not added (tol=1e-9).
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
                ext[k] = 1.0;                       // (...,1,...): k-th simplex vertex
                if (!has_point(ext)) ref_points_.push_back(std::move(ext));
            }
            init_adaptation_state(m, static_cast<int>(ref_points_.size()));
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
        // Guard: user-supplied points may have arrived without init.
        if (is_original_.size() != ref_points_.size())
            init_adaptation_state(m, n);
        cap_ = std::max(cap_, 2 * n);

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
        if (is_original_.size() != ref_points_.size())
            init_adaptation_state(m, n);
        cap_ = std::max(cap_, 2 * n);
        auto fronts = fast_nondominated_sort(vault, n);
        (void)fronts;
    }

    // ── step: one full generation ──────────────────────────────────────────
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
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
            // Part I Table II: pc=1.0, pm=1/n
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
            // Exact fit, or all fronts accepted. The normalization state keeps
            // accumulating in these generations too (Part I §IV-C).
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

        // ── Adaptive reference-point add-and-delete (Section VII) ──────────
        adapt_reference_points(vault, n);
    }
};

} // namespace mootation
