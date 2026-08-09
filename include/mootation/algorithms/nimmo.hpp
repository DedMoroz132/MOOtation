#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// NIMMO — A niching indicator-based multi-modal many-objective optimizer
// Ryoji Tanabe, Hisao Ishibuchi — Swarm and Evolutionary Computation 50, 2019
// doi:10.1016/j.swevo.2019.06.001
//
//
// Iteration scheme (Alg.1, steady-state (mu+1); one offspring per step()):
//   1. Parents x^a, x^b are drawn at random from P with a != b (line 3; no
//      fitness tournament — randomized mating selection, §4.3).
//   2. Offspring u = SBX(x^a, x^b), taking one child, plus polynomial
//      mutation (lines 4–5).
//   3. d_i = normalizedEuclideanDistance(x^i, u) in DECISION SPACE, normalized
//      by the variable bounds (lines 6–7).
//   4. R = the T individuals nearest to u, plus u itself, |R| = T+1
//      (lines 8–10).
//   5. Fitness is computed only inside R (Eq.2):
//      F(x) = Σ_{y∈R\{x}} exp(−I(y,x)/(κ·I^max)), with I = I_eps+ on objectives
//      normalized within R (Eq.3); I^max = max_{x,y∈R} |I(x,y)|. SMALLER F is
//      better — the sign of Eq.2 differs from the original IBEA, and that is
//      the paper's convention.
//   6. x_worst = argmax F is removed from R; the rest return to P
//      (lines 12–13).
//
// PAPER DEFAULTS (§3.1/§4.3): κ=0.05, T=floor(0.1mu) (set_T; T=mu is
// permitted and equals the steady-state IBEA of the paper's experiments,
// §4.3), SBX p_c=1, η_c=20; polynomial mutation p_m=1/D, η_m=20.
// DECLARED DEVIATIONS: floor(0.1mu) versus the paper's "[0.1mu]", which may
// mean rounding — identical for mu in {200, 210, 230}, but off by one for the
// paper's mu=156 (15 vs 16) and mu=135 (13 vs 14) settings;
// initialization is uniform random, which the paper does not specify.
// EXTENSIONS BEYOND THE PAPER (off by default): ConstraintMode FEASIBILITY —
// infeasible solutions receive a penalty fitness of +1e12·(1+cv), worse than
// any feasible one (max feasible F <= T·e^{1/κ} ~= T·4.9e8); the indicator
// pairs and I^max are computed over feasible solutions only, so the penalty
// stays isolated from the indicator normalization.
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
class NIMMOCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    // ── hyperparameters ────────────────────────────────────────────────────
    int          T_     = -1;     // neighbourhood size; -1 → ⌊0.1·µ⌋ (§4.3)
    double       kappa_ = 0.05;   // §3.1: κ usually set to 0.05
    double       eta_c_ = 20.0;   // §4.3: η_c = 20
    double       eta_m_ = 20.0;   // §4.3: η_m = 20
    double       pc_    = 1.0;    // §4.3: p_c = 1
    double       pm_    = -1.0;   // <0 → 1/D (§4.3: p_m = 1/D)
    std::mt19937 rng_{std::random_device{}()};

    // ── Normalised Euclidean distance in decision space (Alg.1 line 7) ─────
    // d(x, y) = sqrt( Σ_j ((x_j - y_j) / range_j)^2 ), range = variable bounds
    double norm_euclidean(DataVault<Ind_t>& vault, int a, int b) const {
        const auto& bounds = vault.get_bounds();
        int nv = vault.vars_n();
        double sum = 0.0;
        for (int j = 0; j < nv; ++j) {
            double lo    = bounds[j].first .value_or(0.0);
            double hi    = bounds[j].second.value_or(1.0);
            double range = std::max(hi - lo, 1e-14);
            double diff  = (vault.get_variable(a, j) - vault.get_variable(b, j)) / range;
            sum += diff * diff;
        }
        // Binary variables contribute using range = 1.
        int nb = vault.bin_vars_n();
        for (int j = 0; j < nb; ++j) {
            double diff = vault.get_bin_variable(a, j) - vault.get_bin_variable(b, j);
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    // ── IBEA Iε+ fitness assignment within neighbourhood R (Eq.2–3) ───────
    // Returns fitness vector F[0..R_size-1]; smallest is best.
    std::vector<double> ibea_fitness(DataVault<Ind_t>& vault,
                                     const std::vector<int>& R) const {
        int sz = static_cast<int>(R.size());
        int m  = vault.objs_n();

        // Eq.3: normalise objectives within R (not P).
        std::vector<double> fmin(m,  std::numeric_limits<double>::max());
        std::vector<double> fmax(m, -std::numeric_limits<double>::max());
        for (int idx : R) {
            const auto& o = vault.objectives_of(idx);
            for (int j = 0; j < m; ++j) {
                fmin[j] = std::min(fmin[j], o[j]);
                fmax[j] = std::max(fmax[j], o[j]);
            }
        }
        // Normalised objectives: f'_i(x) = (f_i(x) - fmin_i) / range_i
        auto fnorm = [&](int idx, int j) -> double {
            double range = std::max(fmax[j] - fmin[j], 1e-14);
            return (vault.objectives_of(idx)[j] - fmin[j]) / range;
        };

        // Eq.3: Iε+(y, x) = max_i (f'_i(y) - f'_i(x))
        auto indicator = [&](int y_idx, int x_idx) -> double {
            double val = -std::numeric_limits<double>::max();
            for (int j = 0; j < m; ++j)
                val = std::max(val, fnorm(y_idx, j) - fnorm(x_idx, j));
            return val;
        };

        // FEASIBILITY (an extension beyond the paper): infeasible solutions
        // are excluded from the indicator pairs, so the penalty never enters
        // I^max and cannot degenerate the exponential fitness of the feasible
        // individuals.
        std::vector<double> cvs(sz, 0.0);
        std::vector<char>   feas(sz, 1);
        if (constraint_mode == ConstraintMode::FEASIBILITY)
            for (int i = 0; i < sz; ++i) {
                cvs[i]  = vault.get_cv(R[i]);
                feas[i] = (cvs[i] <= 0.0) ? 1 : 0;
            }

        // I^max = max_{x,y∈R} |I_eps+(x,y)| (feasible pairs only under FEASIBILITY)
        double I_max = 0.0;
        for (int i = 0; i < sz; ++i)
            for (int j = 0; j < sz; ++j) {
                if (i == j || !feas[i] || !feas[j]) continue;
                I_max = std::max(I_max, std::abs(indicator(R[i], R[j])));
            }
        if (I_max < 1e-14) I_max = 1.0;

        // Eq.2: F(x) = Σ_{y∈R\{x}} exp(-I_eps+(y,x) / (κ·I^max)); smaller is better.
        std::vector<double> F(sz, 0.0);
        for (int i = 0; i < sz; ++i) {
            if (!feas[i]) {
                // The penalty deliberately exceeds the maximum feasible
                // fitness (sz−1)·e^{1/κ}; among infeasible solutions the worse
                // one is the one with the larger CV.
                F[i] = 1e12 * (1.0 + cvs[i]);
                continue;
            }
            for (int j = 0; j < sz; ++j) {
                if (i == j || !feas[j]) continue;
                F[i] += std::exp(-indicator(R[j], R[i]) / (kappa_ * I_max));
            }
        }
        return F;
    }

    double resolved_pm(const DataVault<Ind_t>& vault) const {
        if (pm_ >= 0.0) return pm_;
        int nv = vault.vars_n();
        return (nv > 0) ? 1.0 / static_cast<double>(nv) : 0.0;
    }

public:
    NIMMOCore() = default;

    // Neighbourhood size T; the default is floor(0.1*mu) (§4.3). T = mu is
    // permitted and equals the steady-state IBEA of the paper's experiments.
    void set_T         (int t)    { T_      = t; }
    void set_kappa     (double k) { kappa_  = k; }
    void set_eta_crossover(double e){ eta_c_ = e; }
    void set_eta_mutation (double e){ eta_m_ = e; }
    void set_pc        (double p) { pc_     = p; }
    void set_pm        (double p) { pm_     = p; }
    void set_seed      (unsigned s){ rng_.seed(s); }

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
    }

    void setup_seeded(DataVault<Ind_t>& /*vault*/) {}

    // ── step: one (µ+1) iteration ─────────────────────────────────────────
    // Note: NIMMO is steady-state. Typically called µ times per "generation".
    void step(DataVault<Ind_t>& vault) {
        int mu = vault.pop_size();
        int T  = (T_ < 0) ? static_cast<int>(std::floor(0.1 * mu)) : T_;
        T = std::max(T, 1);
        T = std::min(T, mu);   // T = mu is allowed: R = the whole population
                               // plus the offspring, i.e. the (mu+1) IBEA
                               // selection of §4.3

        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_pop(0, mu - 1);
        double pm = resolved_pm(vault);

        // ── Alg.1 lines 3–5: reproduce one child ───────────────────────────
        // Expand by 1; the child goes into the slot expand() reports. Taking
        // the return value rather than assuming `mu` is the documented-safe
        // form: `n + i` is wrong whenever active_n() != pop_size() at entry
        // (see data_vault.hpp). Here the two coincide because step() always
        // reduces back to mu, but deriving the index removes the assumption.
        int child_slot = vault.expand(1);   // active = mu + 1

        int pa = dist_pop(rng_), pb;
        do { pb = dist_pop(rng_); } while (pb == pa && mu > 1);

        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        for (int j = 0; j < vault.vars_n(); ++j) {
            pv1[j] = vault.get_variable(pa, j);
            pv2[j] = vault.get_variable(pb, j);
        }
        ops::sbx(pv1, pv2, c1, c2, bounds, eta_c_, pc_, rng_);
        // Use c1 for the child; apply polynomial mutation.
        ops::polynomial_mutation(c1, bounds, eta_m_, pm, rng_);

        if (vault.bin_vars_n() > 0) {
            std::vector<int> bv1(vault.bin_vars_n()), bv2(vault.bin_vars_n()),
                             bc1, bc2;
            for (int j = 0; j < vault.bin_vars_n(); ++j) {
                bv1[j] = vault.get_bin_variable(pa, j);
                bv2[j] = vault.get_bin_variable(pb, j);
            }
            ops::binary_crossover(bv1, bv2, bc1, bc2, rng_);
            ops::bit_flip_mutation(bc1, vault.bin_vars_n(), rng_);
            vault.set_all_variables(child_slot, c1, bc1);
        } else {
            vault.set_variables(child_slot, c1);
        }
        vault.refresh_objectives(child_slot);

        // ── Alg.1 lines 6–8: find T closest neighbours in decision space ───
        std::vector<std::pair<double, int>> dists(mu);
        for (int i = 0; i < mu; ++i)
            dists[i] = { norm_euclidean(vault, i, child_slot), i };

        // Partial sort: smallest T.
        std::partial_sort(dists.begin(), dists.begin() + T, dists.end());

        // R = T neighbours + child.  Build index list.
        std::vector<int> R_indices;
        R_indices.reserve(T + 1);
        for (int k = 0; k < T; ++k) R_indices.push_back(dists[k].second);
        R_indices.push_back(child_slot);

        // ── Alg.1 lines 11–12: IBEA fitness within R; remove worst ─────────
        auto F = ibea_fitness(vault, R_indices);
        int worst_local = static_cast<int>(
            std::max_element(F.begin(), F.end()) - F.begin());
        int worst_vault = R_indices[worst_local];

        // Write local fitness to individual for debugging/inspection.
        for (int k = 0; k < static_cast<int>(R_indices.size()); ++k)
            vault.get_ind(R_indices[k]).nimmo_fitness = F[k];

        // ── Alg.1 line 13: discard worst, keep the rest ─────────────────────
        // Swap worst to last slot and reduce to mu.
        if (worst_vault != child_slot) {
            // Worst is one of the neighbours (in [0, mu)).
            // Move child to worst's slot, drop child slot.
            vault.swap_active(worst_vault, child_slot);
            vault.reduce(mu);
        } else {
            // Child is the worst — just drop it.
            vault.reduce(mu);
        }
    }
};

} // namespace mootation
