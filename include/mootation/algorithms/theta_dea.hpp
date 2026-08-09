#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// θ-DEA — A New Dominance Relation-Based Evolutionary Algorithm for
//         Many-Objective Optimization
// Y. Yuan, H. Xu, B. Wang, X. Yao — IEEE Trans. Evol. Comput. 20(1), 2016
// doi:10.1109/TEVC.2015.2420112
//
// Generational scheme (Algorithm 1):
//   1. Q_t (§III-C): parent pairs are chosen AT RANDOM from P_t, with no
//      tournament, as in NSGA-III; SBX with a large eta_c plus polynomial
//      mutation.
//   2. R_t = P_t u Q_t (2N); Step 9: S_t = union_{i=1..tau} F_i from a Pareto
//      NDS of R_t, where tau is the first level at which Sum |F_i| >= N.
//      Everything downstream works on S_t.
//   3. Step 10: UpdateIdealPoint(S_t) — z* is persistent ("minimum value found
//      so far", §III-A) and only improves.
//   4. Step 11, Normalize(S_t) (§III-D): the extreme points come from the ASF
//      of Eq. 10, dividing |f_i − z*_i| by the (z^nad_i − z*_i) of the PREVIOUS
//      generation; the intercepts of Eq. 11 (E^{-1}u) give z^nad_i := a_i;
//      invalid cases (rank(E) < m, no intercept, a_i <= z*_i) set ALL
//      z^nad_i := max f_i over the non-dominated members of S_t; the
//      normalization of Eq. 9 is f~ = (f − z*)/(z^nad − z*).
//   5. Step 12, Clustering(S_t, Lambda) (Alg. 2): argmin d2 (Eq. 12–13);
//      Step 13: theta-NDS per Definition 7 (F_j = d1 + theta*d2, compared only
//      within a cluster); theta = 5, and theta = 10^6 for axial directions
//      lambda (§III-F).
//   6. Steps 14–21: fill P_{t+1} with theta levels; the last accepted level is
//      RandomSort followed by the first N−|P_{t+1}| (Step 20).
//
// PAPER DEFAULTS (§IV-D, Experimental Settings): theta=5 (§IV-D-2: "For the
//   proposed theta-DEA, theta is also set to 5"); eta_c=30 (§IV-D-3: "As for
//   NSGA-III and theta-DEA, the settings are only a bit different according to
//   [28], where eta_c is set to 30"); p_c=1.0, eta_m=20, p_m=1/n (Table V).
// DECLARED DEVIATIONS: none. Reading of §III-D: in any invalid case ALL
//   components of z^nad are rolled back ("In all the above cases, z_i^nad is
//   assigned ... for each i"). das_dennis::generate_exact requires N to equal
//   the lattice point count exactly, which is stricter than the paper.
// EXTENSIONS BEYOND THE PAPER: ConstraintMode::FEASIBILITY — CDP dominance in
//   the prefilter and feasible-priority in the theta comparison (the paper does
//   not consider constraints); off by default. Binary variables (uniform
//   crossover + bit-flip), active only when bin_vars_n() > 0.
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
class ThetaDEACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    // ── hyperparameters ────────────────────────────────────────────────────
    double       theta_default_ = 5.0;       // θ (§IV-D-2: θ = 5)
    double       theta_axis_    = 1e6;       // theta for axial directions (§III-F)
    double       eta_c_         = 30.0;      // SBX index (§IV-D-3: η_c = 30)
    double       eta_m_         = 20.0;      // PM index (Table V: η_m = 20)
    double       pc_            = 1.0;       // crossover prob. (Table V: p_c = 1.0)
    double       pm_            = -1.0;      // mutation prob.; <0 → 1/n (Table V)
    std::mt19937 rng_{std::random_device{}()};

    // Persistent ideal and nadir points (Alg. 1, Steps 3–4, 10; §III-D).
    // zstar_ is the minimum of each objective found over the whole search;
    // znad_  is the nadir estimate, refreshed by the normalization procedure;
    //        its previous-generation value feeds the ASF (Eq. 10).
    std::vector<double> zstar_;
    std::vector<double> znad_;

    // Reference points (un-normalised Das-Dennis lattice, shape [N][m]).
    std::vector<std::vector<double>> ref_points_;
    std::vector<double>              theta_per_ref_; // θ value per reference point

    // ── Das-Dennis reference point generation ─────────────────────────────
    void generate_reference_points(int n, int m) {
        ref_points_ = das_dennis::generate_exact(m, n);
        // CRITICAL: theta_per_ref_ must be filled here.
        // theta_dominates() reads theta_per_ref_[ra]; without this call the
        // vector is empty and the first step() segfaults. This logic used to
        // live only in set_reference_points(), so the auto-generation path
        // never ran it.
        compute_per_ref_theta();
    }

    // Marks the axial reference points (exactly one coordinate equal to 1, the
    // rest 0) and assigns them theta_axis; everything else gets theta_default.
    // Per-reference theta (§III-F): a large theta on the axial points helps
    // capture the nadir during normalization.
    void compute_per_ref_theta() {
        int N = static_cast<int>(ref_points_.size());
        int m = static_cast<int>(ref_points_.empty() ? 0 : ref_points_[0].size());
        theta_per_ref_.assign(N, theta_default_);
        for (int i = 0; i < N; ++i) {
            int cnt1 = 0, cnt0 = 0;
            for (int j = 0; j < m; ++j) {
                if (ref_points_[i][j] > 1.0 - 1e-9) ++cnt1;
                else if (ref_points_[i][j] < 1e-9)  ++cnt0;
            }
            if (cnt1 == 1 && cnt0 == m - 1) {
                theta_per_ref_[i] = theta_axis_;
            }
        }
    }

    // ── Pareto dominance (minimization) ────────────────────────────────────
    // Under FEASIBILITY, the CDP extension (beyond the paper) applies: feasible
    // precedes infeasible, and among infeasible the smaller CV wins.
    bool pareto_dominates(DataVault<Ind_t>& vault, int a, int b) const {
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            double ca = vault.get_cv(a), cb = vault.get_cv(b);
            bool af = (ca <= 0.0), bf = (cb <= 0.0);
            if (af != bf) return af;
            if (af == false) return ca < cb;
        }
        const auto& oa = vault.objectives_of(a);
        const auto& ob = vault.objectives_of(b);
        bool better = false;
        for (std::size_t j = 0; j < oa.size(); ++j) {
            if (oa[j] > ob[j]) return false;
            if (oa[j] < ob[j]) better = true;
        }
        return better;
    }

    // ── Alg. 1, Step 9: S_t = GetParetoNondominatedFronts(R_t) ────────────
    // S_t = union_{i=1..tau} F_i, where tau is the first level with
    // Sum |F_i| >= N (§III-A). first_front (F_1) is also returned, for the
    // §III-D nadir fallback
    // («the largest value of f_i in the non-dominated solutions of S_t»).
    void pareto_prefilter(DataVault<Ind_t>& vault, int pool, int N,
                          std::vector<int>& st,
                          std::vector<int>& first_front) {
        std::vector<std::vector<int>> S(pool);
        std::vector<int> cnt(pool, 0);
        for (int i = 0; i < pool; ++i)
            for (int j = 0; j < pool; ++j) {
                if (i == j) continue;
                if (pareto_dominates(vault, i, j)) S[i].push_back(j);
                else if (pareto_dominates(vault, j, i)) ++cnt[i];
            }
        std::vector<int> cur;
        for (int i = 0; i < pool; ++i)
            if (cnt[i] == 0) cur.push_back(i);
        first_front = cur;
        st.clear();
        while (cur.empty() == false) {
            for (int v : cur) st.push_back(v);
            if (static_cast<int>(st.size()) >= N) break;
            std::vector<int> nxt;
            for (int i : cur)
                for (int j : S[i])
                    if (--cnt[j] == 0) nxt.push_back(j);
            cur = std::move(nxt);
        }
    }

    // ── Normalisation (§III-D, Eq. 9–11) ──────────────────────────────────
    struct NormParams {
        std::vector<double> zmin;   // z* (the persistent ideal point)
        std::vector<double> denom;  // z^nad − z* (the denominator of Eq. 9)
    };

    // Step 10 (UpdateIdealPoint) + Step 11 (Normalize): refreshes zstar_ and
    // znad_, and returns the Eq. 9 normalization parameters for S_t.
    NormParams normalize(DataVault<Ind_t>& vault,
                         const std::vector<int>& st,
                         const std::vector<int>& first_front) {
        int m = vault.objs_n();

        // Step 10: z* is the minimum found over the whole history (§III-A).
        for (int v : st) {
            const auto& o = vault.objectives_of(v);
            for (int j = 0; j < m; ++j) zstar_[j] = std::min(zstar_[j], o[j]);
        }

        // The PREVIOUS generation's nadir range feeds the ASF (Eq. 10:
        // "z_i^nad is
        // the i-th dimension of nadir point estimated in the previous one
        // generation»).
        std::vector<double> prev_range(m);
        for (int j = 0; j < m; ++j)
            prev_range[j] = std::max(znad_[j] - zstar_[j], 1e-12);

        // Extreme points: argmin ASF(x, w_j) over S_t (Eq. 10), with
        // w_{j,i} = 1 when i == j and 10^{-6} otherwise.
        const double eps_w = 1e-6;
        std::vector<int> extreme(m, -1);
        for (int i = 0; i < m; ++i) {
            double best = std::numeric_limits<double>::max();
            for (int v : st) {
                const auto& o = vault.objectives_of(v);
                double asf = 0.0;
                for (int j = 0; j < m; ++j) {
                    double w   = (i == j) ? 1.0 : eps_w;
                    double val = std::abs(o[j] - zstar_[j]) / prev_range[j] / w;
                    if (val > asf) asf = val;
                }
                if (asf < best) { best = asf; extreme[i] = v; }
            }
        }

        // Hyperplane intercepts (Eq. 11): E · x = u, with
        // (a_i − z*_i)^{-1} = x_i. valid becomes false when rank(E) < m, when
        // an intercept is missing, or when a_i <= z*_i (x_i <= 0). In those
        // cases §III-D requires rolling back ALL components of z^nad to
        // max f_i over the non-dominated solutions of S_t.
        bool valid = true;
        for (int i = 0; i < m; ++i) if (extreme[i] < 0) valid = false;
        std::vector<double> x(m, 0.0);
        if (valid) {
            std::vector<std::vector<double>> A(m, std::vector<double>(m + 1));
            for (int i = 0; i < m; ++i) {
                const auto& o = vault.objectives_of(extreme[i]);
                for (int j = 0; j < m; ++j) A[i][j] = o[j] - zstar_[j];
                A[i][m] = 1.0;
            }
            for (int col = 0; col < m && valid; ++col) {
                int pivot = col;
                for (int row = col + 1; row < m; ++row)
                    if (std::abs(A[row][col]) > std::abs(A[pivot][col])) pivot = row;
                std::swap(A[col], A[pivot]);
                if (std::abs(A[col][col]) < 1e-12) { valid = false; break; }
                for (int row = col + 1; row < m; ++row) {
                    double f = A[row][col] / A[col][col];
                    for (int k = col; k <= m; ++k) A[row][k] -= f * A[col][k];
                }
            }
            if (valid) {
                for (int i = m - 1; i >= 0; --i) {
                    x[i] = A[i][m];
                    for (int j = i + 1; j < m; ++j) x[i] -= A[i][j] * x[j];
                    x[i] /= A[i][i];
                }
                // a_i > z*_i iff x_i > 0 and finite; otherwise the intercept
                // is invalid.
                for (int i = 0; i < m; ++i)
                    if (x[i] < 1e-12 || std::isnan(x[i])) { valid = false; break; }
            }
        }

        if (valid) {
            // z^nad_i := a_i = z*_i + 1/x_i (§III-D: «the value of z_i^nad is
            // updated as a_i»).
            for (int i = 0; i < m; ++i) znad_[i] = zstar_[i] + 1.0 / x[i];
        } else {
            // Roll back all components: max f_i over the non-dominated S_t.
            for (int i = 0; i < m; ++i) {
                double mx = -std::numeric_limits<double>::max();
                for (int v : first_front)
                    mx = std::max(mx, vault.objectives_of(v)[i]);
                znad_[i] = mx;
            }
        }

        NormParams np;
        np.zmin = zstar_;
        np.denom.assign(m, 1.0);
        for (int j = 0; j < m; ++j)
            np.denom[j] = std::max(znad_[j] - zstar_[j], 1e-12);
        return np;
    }

    // Normalised objective for vault slot v (Eq. 9).
    std::vector<double> fn(DataVault<Ind_t>& vault, int v,
                           const NormParams& np) const {
        int m = vault.objs_n();
        const auto& o = vault.objectives_of(v);
        std::vector<double> f(m);
        for (int j = 0; j < m; ++j) {
            // f − z* >= 0 by construction, since z* is the global minimum;
            // the clamp is a numerical guard. Values above 1 are allowed —
            // Eq. 9 is not clamped.
            f[j] = std::max((o[j] - np.zmin[j]) / np.denom[j], 0.0);
        }
        return f;
    }

    // ── PBI distances d1 and d2 for solution v w.r.t. reference point r ──
    // d1 = projection length onto ref line = (f̃·λ_r) / ||λ_r||   (Eq. 12)
    // d2 = perpendicular distance = ||f̃ - d1 · λ_r/||λ_r||||      (Eq. 13)
    void compute_d1_d2(const std::vector<double>& fp,
                       int r,
                       double& d1, double& d2) const {
        const auto& v = ref_points_[r];
        double vv = 0.0;
        for (double x : v) vv += x * x;
        double norm_v = std::sqrt(vv);
        if (norm_v < 1e-14) { d1 = 0.0; d2 = 0.0; return; }

        double dot = 0.0;
        for (std::size_t j = 0; j < fp.size(); ++j) dot += fp[j] * v[j];
        d1 = dot / norm_v;                 // scalar projection

        // d2 = ||f̃ - d1 * (λ/||λ||)||
        double d2sq = 0.0;
        for (std::size_t j = 0; j < fp.size(); ++j) {
            double proj_j = d1 * v[j] / norm_v;
            double diff = fp[j] - proj_j;
            d2sq += diff * diff;
        }
        d2 = std::sqrt(d2sq);
    }

    // ── Clustering (Algorithm 2): each solution in S_t → argmin d2 ────────
    void cluster(DataVault<Ind_t>& vault, const std::vector<int>& st,
                 const NormParams& np) {
        int N = static_cast<int>(ref_points_.size());
        for (int v : st) {
            auto fp = fn(vault, v, np);
            double best_d2 = std::numeric_limits<double>::max();
            int    best_r  = 0;
            double best_d1 = 0.0;
            for (int r = 0; r < N; ++r) {
                double d1, d2;
                compute_d1_d2(fp, r, d1, d2);
                if (d2 < best_d2) { best_d2 = d2; best_r = r; best_d1 = d1; }
            }
            vault.get_ind(v).ref_point_idx = best_r;
            vault.get_ind(v).d1 = best_d1;
            vault.get_ind(v).d2 = best_d2;
        }
    }

    // ── θ-Dominance check (Definition 7) ──────────────────────────────────
    // Returns true if solution a θ-dominates b.
    bool theta_dominates(DataVault<Ind_t>& vault, int a, int b) const {
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            double ca = vault.get_cv(a), cb = vault.get_cv(b);
            bool af = (ca <= 0.0), bf = (cb <= 0.0);
            if ( af && bf == false) return true;
            if (af == false &&  bf) return false;
            if (af == false && bf == false) return ca < cb;
        }

        int ra = vault.get_ind(a).ref_point_idx;
        int rb = vault.get_ind(b).ref_point_idx;

        if (ra == rb) {
            // Same cluster: compare PBI values F_j = d1 + θ·d2.
            double theta = theta_per_ref_[ra];
            double Fa = vault.get_ind(a).d1 + theta * vault.get_ind(a).d2;
            double Fb = vault.get_ind(b).d1 + theta * vault.get_ind(b).d2;
            return Fa < Fb;
        }
        // Different clusters are theta-incomparable (Definition 7: x
        // theta-dominates y requires
        // x, y ∈ C_j; «there is no competitive relationship between clusters»).
        return false;
    }

    // ── θ-dominance based fast non-dominated sort over S_t ────────────────
    // Returns the theta levels as vault indices; rank is written into the
    // individuals.
    std::vector<std::vector<int>>
    theta_nondominated_sort(DataVault<Ind_t>& vault,
                            const std::vector<int>& st) {
        int n = static_cast<int>(st.size());
        std::vector<std::vector<int>> S(n);
        std::vector<int> np_cnt(n, 0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                if (theta_dominates(vault, st[i], st[j])) S[i].push_back(j);
                else if (theta_dominates(vault, st[j], st[i])) ++np_cnt[i];
            }
        std::vector<std::vector<int>> fronts;
        std::vector<int> f0;
        for (int i = 0; i < n; ++i)
            if (np_cnt[i] == 0) { vault.get_ind(st[i]).rank = 0; f0.push_back(i); }
        fronts.push_back(f0);
        int k = 0;
        while (fronts[k].empty() == false) {
            std::vector<int> nxt;
            for (int i : fronts[k])
                for (int j : S[i])
                    if (--np_cnt[j] == 0) {
                        vault.get_ind(st[j]).rank = k + 1;
                        nxt.push_back(j);
                    }
            fronts.push_back(nxt);
            ++k;
        }
        fronts.pop_back();
        // Local indices -> vault indices.
        for (auto& fr : fronts)
            for (int& v : fr) v = st[v];
        return fronts;
    }

    // ── Rearrange vault ────────────────────────────────────────────────────
    void rearrange(DataVault<Ind_t>& vault,
                   const std::vector<int>& survivors, int pool) {
        int n = static_cast<int>(survivors.size());
        std::vector<int> pos(pool), at_pos(pool);
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

    // Alg. 1, Steps 3–4: initialize z* and z^nad from the initial population P_0.
    void init_ideal_nadir(DataVault<Ind_t>& vault, int n) {
        int m = vault.objs_n();
        zstar_.assign(m,  std::numeric_limits<double>::max());
        znad_ .assign(m, -std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) {
                zstar_[j] = std::min(zstar_[j], o[j]);
                znad_[j]  = std::max(znad_[j],  o[j]);
            }
        }
    }

public:
    ThetaDEACore() = default;

    // NOTE on ordering: set_theta / set_theta_axis only record the values; the
    // per-reference table is built by compute_per_ref_theta(), which runs when
    // the reference points are created. Calling them AFTER
    // set_reference_points (or after setup()) therefore has no effect on
    // already-built references — set them first.
    void set_theta        (double t)  { theta_default_ = t; }
    void set_theta_axis   (double t)  { theta_axis_    = t; }
    void set_eta_crossover(double e)  { eta_c_         = e; }
    void set_eta_mutation (double e)  { eta_m_         = e; }
    void set_pc           (double p)  { pc_            = p; }
    void set_pm           (double p)  { pm_            = p; }
    void set_seed         (unsigned s){ rng_.seed(s); }

    void set_reference_points(std::vector<std::vector<double>> rp) {
        ref_points_ = std::move(rp);
        compute_per_ref_theta();
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

        // Steps 3–4: z* and z^nad from P_0. Mating is random (§III-C), so the
        // ranks and clusters of the initial population are not needed.
        init_ideal_nadir(vault, n);
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int m = vault.objs_n();
        if (ref_points_.empty()) generate_reference_points(n, m);
        init_ideal_nadir(vault, n);
    }

    // ── step: one full generation (Algorithm 1, Steps 7–22) ───────────────
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();

        // Actual parent count recorded by expand().
        int n_parents = vault.parents_n();

        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_int(0, n_parents - 1);

        // ── expand: add n offspring slots ─────────────────────────────────
        vault.expand(n);

        // Table V: p_c = 1.0, p_m = 1/n, with n the number of real variables.
        double pm = (pm_ >= 0.0) ? pm_
                                 : 1.0 / static_cast<double>(std::max(1, vault.vars_n()));

        // ── breed n offspring ─────────────────────────────────────────────
        // §III-C: «two parent solutions are randomly selected from the
        // current population P_t" — with no binary tournament.
        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        for (int i = 0; i < n; i += 2) {
            int p1 = dist_int(rng_);
            int p2 = dist_int(rng_);
            for (int j = 0; j < vault.vars_n(); ++j) {
                pv1[j] = vault.get_variable(p1, j);
                pv2[j] = vault.get_variable(p2, j);
            }
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
                vault.set_all_variables(n_parents + i,     c1, bc1);
                if (i + 1 < n) vault.set_all_variables(n_parents + i + 1, c2, bc2);
            } else {
                vault.set_variables(n_parents + i, c1);
                if (i + 1 < n) vault.set_variables(n_parents + i + 1, c2);
            }
        }

        // ── evaluate offspring ─────────────────────────────────────────────
        vault.sync();

        int pool_size = n_parents + n;

        // ── Step 9: Pareto prefilter of S_t out of R_t ────────────────────
        std::vector<int> st, first_front;
        pareto_prefilter(vault, pool_size, n, st, first_front);

        // ── Steps 10–11: UpdateIdealPoint + Normalize (persistent z*/z^nad)
        NormParams np = normalize(vault, st, first_front);

        // ── Step 12: clustering S_t ────────────────────────────────────────
        cluster(vault, st, np);

        // ── Step 13: θ-dominance sort over S_t ─────────────────────────────
        auto fronts = theta_nondominated_sort(vault, st);

        // ── Steps 14–21: select N survivors ────────────────────────────────
        std::vector<int> survivors;
        survivors.reserve(n);

        for (auto& front : fronts) {
            if (static_cast<int>(survivors.size()) + static_cast<int>(front.size()) <= n) {
                for (int v : front) survivors.push_back(v);
                if (static_cast<int>(survivors.size()) == n) break;
            } else {
                // The last accepted theta level, Step 20: "RandomSort(F'_i)";
                // §III-A: «we just randomly select solutions in the last
                // accepted level … because θ-dominance has stressed both
                // convergence and diversity».
                std::shuffle(front.begin(), front.end(), rng_);
                int K = n - static_cast<int>(survivors.size());
                for (int k = 0; k < K; ++k) survivors.push_back(front[k]);
                break;
            }
        }

        rearrange(vault, survivors, pool_size);
    }
};

} // namespace mootation
