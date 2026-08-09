#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// IBEA-HD — Indicator-Based Evolutionary Algorithm (adaptive, I_HD indicator)
// Eckart Zitzler, Simon Künzli — PPSN VIII (LNCS 3242), 2004
// doi:10.1007/978-3-540-30217-9_84          (source: zitzler2004)
//
// Generation scheme (Alg.1 + adaptive version Alg.2, indicator §3.2):
//   1. Mating: binary tournament with replacement on F (larger F is better),
//      µ parents → µ offspring.
//   2. Variation: SBX (η_c=20, p_c=1.0) + polynomial mutation (η_m=20, p_m);
//      offspring are added to P → pool of 2µ (generational µ+λ, λ=µ).
//   3. Objective scaling to [0,1] by the pool bounds (Alg.2 Step 2.1–2.2).
//   4. I_HD(A={y},B={x}) = IH(B)−IH(A) if A dominates B; otherwise
//      IH(A+B)−IH(A); IH taken w.r.t. reference point r (§3.2);
//      c = max_{x,y∈P} |I_HD(x,y)| (Step 2.3).
//   5. F(x) = Σ_{y∈P\{x}} −exp(−I_HD({y},{x})/(c·κ)), κ=0.05 (Step 2.4).
//   6. Env-selection (Step 3): iteratively remove argmin F, updating
//      F(x) += exp(−I_HD({x*},{x})/(c·κ)); normalisation and c stay fixed for the cycle.
//
// Reference point (§4.1): in the normalised [0,1] space «for the
// reference point we used a value of 2 for all objectives» → default
// ref_point_ = 2.0 (set_ref_point). The two-point I_HD is computed by the
// exact inclusion-exclusion formula: IH({p}) = Π_j max(r−p_j,0),
// IH({y}∪{x}) = IH({y})+IH({x})−Π_j max(r−max(y_j,x_j),0) — O(n) for a pair
// of points for any number of objectives (§3.2).
//
// Defaults (§4.1 + footnote 3): κ=0.05; ref_point=2.0; SBX-20
// (η_c=20) with p_c=1.0; p_m=0.01 (footnote 3: «recombination and mutation
// probabilities were set to 1.0 and to 0.01, resp.»). η_m — conventional 20:
// footnote 3 only specifies «a polynomial distribution for mutation», the η_m
// NUMBER is not given.
// FIX 2026-07-08 (source-fidelity review):
//   comment brought in line with the facts. The code is paper-exact
//   (pm_=0.01, footnote 3); the previous header falsely claimed «p_m defaults
//   to 1/n_vars» and
//   attributed η_m=20 to footnote 3. Deviation: weak dominance in the I_HD
//   branching — functionally neutral (equal vectors yield 0 in both branches)
//   (internal audit, IHD-1/IHD-3).
// Extensions beyond the paper (off by default): ConstraintMode FEASIBILITY —
// infeasible individuals get penalty fitness −1e12·(1+cv) (guaranteed below any
// sum of env-selection exp-increments); indicator pairs and c use feasible
// individuals only, consistently between assign_fitness and env_select.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
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
class IBEAhdCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double       kappa_     = 0.05;   // §4.1: κ = 0.05
    double       ref_point_ = 2.0;    // §4.1: reference point = 2 (normalised space)
    double       eta_c_     = 20.0;   // footnote 3: SBX-20 → η_c=20 (verified)
    // FIX 2026-07-08 (source-fidelity review):
    // footnote 3 only specifies «a polynomial distribution for mutation»
    // — the η_m NUMBER is not given; 20 is a convention (not from the paper).
    // The previous comment «footnote 3: η_mutation = 20» falsely attributed the
    // value to the paper.
    double       eta_m_     = 20.0;   // convention (footnote 3 gives no numeric η_m)
    double       pc_        = 1.0;    // footnote 3: recombination probability 1.0
    double       pm_        = 0.01;   // footnote 3: mutation probability 0.01 (fixed)
    // FIX 2026-06: same paper zitzler2004 as the base IBEA → pm=0.01.
    std::mt19937 rng_{std::random_device{}()};

    // ── Objective normalisation ────────────────────────────────────────────
    void calc_bounds(DataVault<Ind_t>& vault, int n,
                     std::vector<double>& fmin, std::vector<double>& fmax) const {
        int m = vault.objs_n();
        fmin.assign(m,  std::numeric_limits<double>::max());
        fmax.assign(m, -std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) {
                fmin[j] = std::min(fmin[j], o[j]);
                fmax[j] = std::max(fmax[j], o[j]);
            }
        }
    }

    std::vector<double> normalise(const std::vector<double>& o,
                                  const std::vector<double>& fmin,
                                  const std::vector<double>& fmax) const {
        int m = static_cast<int>(o.size());
        std::vector<double> fn(m);
        for (int j = 0; j < m; ++j) {
            double range = fmax[j] - fmin[j];
            fn[j] = (range > 1e-14) ? (o[j] - fmin[j]) / range : 0.0;
        }
        return fn;
    }

    // ── I_HD(A={y}, B={x}) — Zitzler & Künzli 2004, §3.2 ──────────────────
    //   I_HD(A,B) = IH(B) - IH(A)      if A dominates all of B
    //             = IH(A+B) - IH(A)    otherwise
    // IH({p})       = Π_j max(r - p_j, 0)            (single point)
    // IH({y}∩{x})   = Π_j max(r - max(y_j,x_j), 0)
    // IH({y}∪{x})   = IH({y}) + IH({x}) - IH({y}∩{x})
    // Sign: y dominates x → I_HD(y,x) = IH(x)-IH(y) < 0  (x penalised in F).
    double ihd(const std::vector<double>& fn_y,    // A = {y}
               const std::vector<double>& fn_x)    // B = {x}
        const {
        int m = static_cast<int>(fn_y.size());
        double r = ref_point_;     // §4.1: 2.0 in the normalised space

        double hv_y = 1.0, hv_x = 1.0, hv_inter = 1.0;
        for (int j = 0; j < m; ++j) {
            hv_y     *= std::max(r - fn_y[j], 0.0);
            hv_x     *= std::max(r - fn_x[j], 0.0);
            hv_inter *= std::max(r - std::max(fn_y[j], fn_x[j]), 0.0);
        }
        double hv_union = hv_y + hv_x - hv_inter;   // IH({y}∪{x})

        // y dominates x  ⟺  fn_y[j] <= fn_x[j] for all j
        bool y_dominates_x = true;
        for (int j = 0; j < m; ++j)
            if (fn_y[j] > fn_x[j]) { y_dominates_x = false; break; }

        if (y_dominates_x)
            return hv_x - hv_y;          // IH(B) - IH(A)   (branch 1, < 0)
        else
            return hv_union - hv_y;      // IH(A+B) - IH(A)  (branch 2, >= 0)
    }

    // c = max|I_HD| over pairs; in FEASIBILITY pairs involving infeasible
    // individuals are excluded (consistent with assign_fitness).
    double calc_c(const std::vector<std::vector<double>>& fn,
                  const std::vector<double>& cvs) const {
        int n = static_cast<int>(fn.size());
        double c = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                if (constraint_mode == ConstraintMode::FEASIBILITY &&
                    (cvs[i] > 0.0 || cvs[j] > 0.0)) continue;
                double v = std::abs(ihd(fn[j], fn[i]));
                if (v > c) c = v;
            }
        return (c > 1e-14) ? c : 1.0;
    }

    // ── Fitness assignment over pool [0, n) ───────────────────────────────
    void assign_fitness(DataVault<Ind_t>& vault, int n) {
        std::vector<double> fmin, fmax;
        calc_bounds(vault, n, fmin, fmax);

        // Normalised objective vectors.
        std::vector<std::vector<double>> fn(n);
        for (int i = 0; i < n; ++i)
            fn[i] = normalise(vault.objectives_of(i), fmin, fmax);

        // CV values.
        std::vector<double> cvs(n, 0.0);
        if (constraint_mode == ConstraintMode::FEASIBILITY)
            for (int i = 0; i < n; ++i) cvs[i] = vault.get_cv(i);

        double c = calc_c(fn, cvs);

        // F(x) = Σ_{y≠x} -exp(-I_HD(y,x) / (c·κ))
        for (int i = 0; i < n; ++i) {
            if (constraint_mode == ConstraintMode::FEASIBILITY && cvs[i] > 0.0) {
                // The penalty base is guaranteed below the lowest possible fitness
                // of a feasible individual even after all env-selection
                // exp-increments (each ≤ e^{1/κ} ≈ 4.9e8).
                vault.get_ind(i).fitness = -1e12 * (1.0 + cvs[i]);
                continue;
            }
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                if (constraint_mode == ConstraintMode::FEASIBILITY && cvs[j] > 0.0)
                    continue;
                sum += -std::exp(-ihd(fn[j], fn[i]) / (c * kappa_));
            }
            vault.get_ind(i).fitness = sum;
        }
    }

    // ── Environmental selection ────────────────────────────────────────────
    void env_select(DataVault<Ind_t>& vault, int target_n) {
        // Zitzler & Künzli 2004, Alg.2: normalisation and c=max|I_HD| are
        // computed ONCE over the initial pool and stay fixed for the whole
        // removal cycle. Objectives are normalised on the fly with the fixed
        // bounds, so slot permutations (swap_active) are safe. In FEASIBILITY
        // c uses feasible pairs only (consistent with assign_fitness).
        int n0 = static_cast<int>(vault.active_n());
        std::vector<double> fmin, fmax;
        calc_bounds(vault, n0, fmin, fmax);

        std::vector<double> cvs0(n0, 0.0);
        if (constraint_mode == ConstraintMode::FEASIBILITY)
            for (int i = 0; i < n0; ++i) cvs0[i] = vault.get_cv(i);

        double c;
        {
            std::vector<std::vector<double>> fn0(n0);
            for (int i = 0; i < n0; ++i)
                fn0[i] = normalise(vault.objectives_of(i), fmin, fmax);
            c = calc_c(fn0, cvs0);
        }

        while (static_cast<int>(vault.active_n()) > target_n) {
            int curr = static_cast<int>(vault.active_n());

            // Find worst (minimum fitness).
            int worst = 0;
            if (constraint_mode == ConstraintMode::FEASIBILITY) {
                double wf = vault.get_ind(0).fitness;
                double wc = vault.get_cv(0);
                for (int i = 1; i < curr; ++i) {
                    double fi = vault.get_ind(i).fitness;
                    double ci = vault.get_cv(i);
                    bool wi = (wc > 0.0), ii = (ci > 0.0);
                    if ((ii && !wi) || (ii && wi && ci > wc) || (!ii && !wi && fi < wf)) {
                        worst = i; wf = fi; wc = ci;
                    }
                }
            } else {
                for (int i = 1; i < curr; ++i)
                    if (vault.get_ind(i).fitness < vault.get_ind(worst).fitness)
                        worst = i;
            }

            // Update fitness of remaining: F(z) += exp(-I_HD(worst,z)/(c·κ))
            // with the fixed bounds/c (objectives are normalised on the fly).
            // FEASIBILITY: increments only between feasible individuals
            // (infeasible ones took no part in the indicator sums and keep
            // their pure penalty).
            double cv_worst = (constraint_mode == ConstraintMode::FEASIBILITY)
                              ? vault.get_cv(worst) : 0.0;
            std::vector<double> fnw = normalise(vault.objectives_of(worst), fmin, fmax);
            for (int i = 0; i < curr; ++i) {
                if (i == worst) continue;
                if (constraint_mode == ConstraintMode::FEASIBILITY &&
                    (cv_worst > 0.0 || vault.get_cv(i) > 0.0)) continue;
                std::vector<double> fni = normalise(vault.objectives_of(i), fmin, fmax);
                vault.get_ind(i).fitness +=
                    std::exp(-ihd(fnw, fni) / (c * kappa_));
            }

            vault.swap_active(worst, curr - 1);
            vault.reduce(curr - 1);
        }
    }

    // ── Tournament (larger fitness = better) ───────────────────────────────
    int tournament(DataVault<Ind_t>& vault,
                   std::uniform_int_distribution<int>& dist) {
        int a = dist(rng_), b = dist(rng_);
        return (vault.get_ind(a).fitness > vault.get_ind(b).fitness) ? a : b;
    }

    double resolved_pm(const DataVault<Ind_t>& vault) const {
        if (pm_ >= 0.0) return pm_;
        int nv = vault.vars_n();
        return (nv > 0) ? 1.0 / static_cast<double>(nv) : 0.0;
    }

public:
    IBEAhdCore() = default;

    void set_kappa        (double k) { kappa_     = k; }
    // Reference point in the normalised [0,1] space; paper §4.1: 2.0.
    void set_ref_point    (double r) { ref_point_ = r; }
    void set_eta_crossover(double e) { eta_c_     = e; }
    void set_eta_mutation (double e) { eta_m_     = e; }
    void set_pc           (double p) { pc_        = p; }
    // p_m of polynomial mutation. Default pm_=0.01 (paper-exact, footnote 3:
    // «mutation probability … set to … 0.01»). FIX 2026-07-08 (source-fidelity
    // review): comment brought in line with the facts — the default is NOT
    // 1/n_vars but a fixed 0.01.
    void set_pm           (double p) { pm_        = p; }
    void set_seed         (unsigned s){ rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dr(0.0, 1.0);
        std::uniform_int_distribution<int>     db(0, 1);
        std::vector<double> vars(vault.vars_n());
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
        assign_fitness(vault, n);
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        assign_fitness(vault, vault.pop_size());
    }

    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_int(0, n - 1);
        int off_base = vault.expand(n);  // [off_base, off_base+n) — offspring slots
        double pm = resolved_pm(vault);

        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        for (int i = 0; i < n; i += 2) {
            int p1 = tournament(vault, dist_int);
            int p2 = tournament(vault, dist_int);
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
                vault.set_all_variables(off_base + i,     c1, bc1);
                if (i + 1 < n) vault.set_all_variables(off_base + i + 1, c2, bc2);
            } else {
                vault.set_variables(off_base + i, c1);
                if (i + 1 < n) vault.set_variables(off_base + i + 1, c2);
            }
        }
        vault.sync();
        assign_fitness(vault, n * 2);
        env_select(vault, n);
    }
};

} // namespace mootation
