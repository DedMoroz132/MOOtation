#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// IBEA-ε+ — Indicator-Based Evolutionary Algorithm (adaptive, Iε+ indicator)
// Eckart Zitzler, Simon Künzli — PPSN VIII (LNCS 3242), 2004
// doi:10.1007/978-3-540-30217-9_84          (source: zitzler2004)
//
// Generation scheme (Alg.1 + adaptive version Alg.2):
//   1. Mating: binary tournament with replacement on F (larger F is better),
//      µ parents → µ offspring.
//   2. Variation: SBX (η_c=20, p_c=1.0) + polynomial mutation (η_m=20, p_m);
//      offspring are added to P → pool of 2µ (generational µ+λ, λ=µ).
//   3. Objective scaling to [0,1] by the pool bounds (Alg.2 Step 2.1–2.2).
//   4. Iε+(a,b) = max_i (f'_i(a) − f'_i(b)); c = max_{x,y∈P} |I(x,y)| (Step 2.3).
//   5. F(x) = Σ_{y∈P\{x}} −exp(−I({y},{x})/(c·κ)), κ=0.05 (Step 2.4).
//   6. Env-selection (Step 3): iteratively remove argmin F, updating
//      F(x) += exp(−I({x*},{x})/(c·κ)); normalisation and c stay fixed for the cycle.
//
// Defaults (§4.1 + footnote 3): κ=0.05; SBX-20 (η_c=20) with p_c=1.0;
// p_m=0.01 — footnote 3 gives the number: «The recombination and mutation
// probabilities were set to 1.0 and to 0.01, resp.». η_m — conventional 20:
// footnote 3 only specifies «a polynomial distribution for mutation», the η_m
// NUMBER is not given.
// FIX 2026-07-08 (source-fidelity review):
//   comment brought in line with the facts. The code is paper-exact
//   (pm_=0.01, footnote 3); the previous header falsely claimed «p_m defaults
//   to 1/n_vars …
//   paper-exact via set_pm(0.01)», although the default is ALREADY 0.01. The
//   binary operators (uniform crossover +
// bit-flip 1/n_bits) are general-purpose; the paper's knapsack experiments
// used one-point 0.8 + bit-flip 0.04 (footnote 3)
// (internal audit, IEP-1/IEP-3).
// Extensions beyond the paper (off by default): ConstraintMode
//   FEASIBILITY    — infeasible individuals get penalty fitness −1e12·(1+cv)
//                    (guaranteed below any sum of env-selection exp-increments);
//                    indicator pairs and c use feasible individuals only,
//                    consistently between calculate_fitness and environmental_selection;
//   EPS_CONSTRAINT — Iε+ with a CV shift; c is computed with the same CV-shifted
//                    indicator (otherwise an underestimated c caused exp overflow).
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
class IBEAePlusCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    // Operators STRICTLY per the original IBEA source (Zitzler & Künzli 2004,
    // footnote 3 — verified against the zitzler2004 source): «the SBX-20 operator
    // is used for recombination and a polynomial distribution for mutation.
    // The recombination and mutation probabilities were set to 1.0 and to
    // 0.01, resp.» → η_c=20 (SBX-20), pc=1.0, pm=0.01 (FIXED, not 1/n!).
    // η_mut is not given numerically in the paper → conventional 20.
    double       kappa_ = 0.05;   // §4.1: κ = 0.05
    double       eta_c_ = 20.0;   // footnote 3: SBX-20 → η_recombination = 20
    double       eta_m_ = 20.0;   // not given numerically → convention (polynomial, η=20)
    double       pc_    = 1.0;    // footnote 3: recombination probability 1.0
    double       pm_    = 0.01;   // footnote 3: mutation probability 0.01 (fixed)
    std::mt19937 rng_{std::random_device{}()};

    // Iε+(a,b) with a CV shift — for EPS_CONSTRAINT (extension beyond the paper)
    double eps_indicator_with_cv(const std::vector<double>& a,
                                 const std::vector<double>& b,
                                 double cv_a, double cv_b,
                                 const std::vector<double>& fmin,
                                 const std::vector<double>& fmax) const
    {
        double worst = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < a.size(); ++i) {
            double range   = fmax[i] - fmin[i];
            double shift_a = (cv_a > 0.0) ? cv_a / (range > 1e-14 ? range : 1.0) : 0.0;
            double shift_b = (cv_b > 0.0) ? cv_b / (range > 1e-14 ? range : 1.0) : 0.0;
            double an = (range > 1e-14) ? (a[i] - fmin[i]) / range : 0.0;
            double bn = (range > 1e-14) ? (b[i] - fmin[i]) / range : 0.0;
            an += shift_a;
            bn += shift_b;
            worst = std::max(worst, an - bn);
        }
        return worst;
    }

    // Iε+(a,b) = max_i (f'_i(a) − f'_i(b)) over normalised objectives (Alg.2)
    double eps_indicator_norm(const std::vector<double>& a,
                              const std::vector<double>& b,
                              const std::vector<double>& fmin,
                              const std::vector<double>& fmax) const
    {
        double worst = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < a.size(); ++i) {
            double range = fmax[i] - fmin[i];
            double an = (range > 1e-14) ? (a[i] - fmin[i]) / range : 0.0;
            double bn = (range > 1e-14) ? (b[i] - fmin[i]) / range : 0.0;
            worst = std::max(worst, an - bn);
        }
        return worst;
    }

    void calc_obj_bounds(DataVault<Ind_t>& vault, int n,
                         std::vector<double>& fmin,
                         std::vector<double>& fmax) const
    {
        int m = vault.objs_n();
        fmin.assign(m,  std::numeric_limits<double>::max());
        fmax.assign(m, -std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int k = 0; k < m; ++k) {
                fmin[k] = std::min(fmin[k], o[k]);
                fmax[k] = std::max(fmax[k], o[k]);
            }
        }
    }

    // c = max|I| over pool pairs (Alg.2 Step 2.3). Consistency with fitness:
    // FEASIBILITY — pairs involving infeasible individuals are excluded;
    // EPS_CONSTRAINT — the same CV-shifted indicator as in fitness.
    double calc_c(DataVault<Ind_t>& vault, int n,
                  const std::vector<double>& fmin,
                  const std::vector<double>& fmax,
                  const std::vector<double>& cvs) const
    {
        double c = 0.0;
        for (int i = 0; i < n; ++i) {
            const auto& oi = vault.objectives_of(i);
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                double v;
                if (constraint_mode == ConstraintMode::FEASIBILITY) {
                    if (cvs[i] > 0.0 || cvs[j] > 0.0) continue;
                    v = std::abs(
                        eps_indicator_norm(vault.objectives_of(j), oi, fmin, fmax));
                } else if (constraint_mode == ConstraintMode::EPS_CONSTRAINT) {
                    v = std::abs(
                        eps_indicator_with_cv(vault.objectives_of(j), oi,
                                              cvs[j], cvs[i], fmin, fmax));
                } else {
                    v = std::abs(
                        eps_indicator_norm(vault.objectives_of(j), oi, fmin, fmax));
                }
                if (v > c) c = v;
            }
        }
        return (c > 1e-14) ? c : 1.0;
    }

    double resolved_pm(const DataVault<Ind_t>& vault) const {
        if (pm_ >= 0.0) return pm_;
        int nv = vault.vars_n();
        return (nv > 0) ? 1.0 / static_cast<double>(nv) : 0.0;
    }

public:
    IBEAePlusCore() = default;

    void set_kappa(double k)         { kappa_ = k; }
    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation(double e)  { eta_m_ = e; }
    void set_pc(double p)            { pc_    = p; }
    // p_m of polynomial mutation. Default pm_=0.01 (paper-exact, footnote 3:
    // «mutation probability … set to … 0.01»). FIX 2026-07-08 (source-fidelity
    // review): comment brought in line with the facts — the default is NOT
    // 1/n_vars but a fixed 0.01.
    void set_pm(double p)            { pm_    = p; }
    void set_seed(unsigned s)        { rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault)
    {
        int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::uniform_int_distribution<int>    dist_bin(0, 1);
        std::vector<double> vars(vault.vars_n());
        std::vector<int>    bvars(vault.bin_vars_n());

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < vault.vars_n(); ++j) {
                double lo = bounds[j].first .value_or(0.0);
                double hi = bounds[j].second.value_or(1.0);
                vars[j] = lo + dist(rng_) * (hi - lo);
            }
            for (int j = 0; j < vault.bin_vars_n(); ++j)
                bvars[j] = dist_bin(rng_);

            if (vault.bin_vars_n() > 0)
                vault.set_all_variables(i, vars, bvars);
            else
                vault.set_variables(i, vars);
        }
        vault.sync();
        calculate_fitness(vault, n);
    }

    // ============================================================
    //  setup_seeded — alternative setup for resume.
    //  Assumes: the DataVault is already seeded (seed_individual
    //  for all pop_size slots); random generation is skipped.
    //  Only fitness is recomputed — IBEA needs it for selection.
    // ============================================================
    void setup_seeded(DataVault<Ind_t>& vault)
    {
        int n = vault.pop_size();
        calculate_fitness(vault, n);
    }

    void calculate_fitness(DataVault<Ind_t>& vault, int n)
    {
        std::vector<double> fmin, fmax;
        calc_obj_bounds(vault, n, fmin, fmax);

        std::vector<double> cvs(n, 0.0);
        if (constraint_mode != ConstraintMode::NONE)
            for (int i = 0; i < n; ++i) cvs[i] = vault.get_cv(i);

        double c = calc_c(vault, n, fmin, fmax, cvs);

        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            const auto& oi = vault.objectives_of(i);

            if (constraint_mode == ConstraintMode::FEASIBILITY) {
                if (cvs[i] > 0.0) {
                    // The penalty base is guaranteed below the lowest possible
                    // fitness of a feasible individual even after all
                    // env-selection exp-increments (each ≤ e^{1/κ} ≈ 4.9e8).
                    vault.get_ind(i).fitness = -1e12 * (1.0 + cvs[i]);
                    continue;
                }
                for (int j = 0; j < n; ++j) {
                    if (i == j || cvs[j] > 0.0) continue;
                    sum += -std::exp(
                        -eps_indicator_norm(vault.objectives_of(j), oi, fmin, fmax)
                        / (c * kappa_));
                }
            } else if (constraint_mode == ConstraintMode::EPS_CONSTRAINT) {
                for (int j = 0; j < n; ++j) {
                    if (i == j) continue;
                    double ind = eps_indicator_with_cv(
                        vault.objectives_of(j), oi,
                        cvs[j], cvs[i], fmin, fmax);
                    sum += -std::exp(-ind / (c * kappa_));
                }
            } else {
                for (int j = 0; j < n; ++j) {
                    if (i == j) continue;
                    sum += -std::exp(
                        -eps_indicator_norm(vault.objectives_of(j), oi, fmin, fmax)
                        / (c * kappa_));
                }
            }
            vault.get_ind(i).fitness = sum;
        }
    }

    void environmental_selection(DataVault<Ind_t>& vault, int target_n)
    {
        // Zitzler & Künzli 2004, Alg.2: objective normalisation and c = max|I|
        // are computed ONCE over the initial pool and stay FIXED for the whole
        // removal cycle. The incremental F(x) += exp(-I(x*,x)/(c·κ)) preserves
        // the invariant F(x)=Σ_{y≠x} -exp(-I(y,x)/(c·κ)).
        int n0 = static_cast<int>(vault.active_n());
        std::vector<double> fmin, fmax;
        calc_obj_bounds(vault, n0, fmin, fmax);

        std::vector<double> cvs0(n0, 0.0);
        if (constraint_mode != ConstraintMode::NONE)
            for (int i = 0; i < n0; ++i) cvs0[i] = vault.get_cv(i);
        double c = calc_c(vault, n0, fmin, fmax, cvs0);

        while (static_cast<int>(vault.active_n()) > target_n) {
            int curr = static_cast<int>(vault.active_n());

            int worst = 0;
            if (constraint_mode == ConstraintMode::FEASIBILITY) {
                double worst_cv  = vault.get_cv(0);
                double worst_fit = vault.get_ind(0).fitness;
                for (int i = 1; i < curr; ++i) {
                    double cv_i  = vault.get_cv(i);
                    double fit_i = vault.get_ind(i).fitness;
                    bool i_inf = (cv_i    > 0.0);
                    bool w_inf = (worst_cv > 0.0);
                    if (i_inf && !w_inf) {
                        worst = i; worst_cv = cv_i; worst_fit = fit_i;
                    } else if (!i_inf && w_inf) {
                        // worst stays
                    } else if (i_inf && w_inf) {
                        if (cv_i > worst_cv) { worst = i; worst_cv = cv_i; worst_fit = fit_i; }
                    } else {
                        if (fit_i < worst_fit) { worst = i; worst_cv = cv_i; worst_fit = fit_i; }
                    }
                }
            } else {
                for (int i = 1; i < curr; ++i)
                    if (vault.get_ind(i).fitness < vault.get_ind(worst).fitness)
                        worst = i;
            }

            const std::vector<double> ow = vault.objectives_of(worst);
            double cv_worst = (constraint_mode != ConstraintMode::NONE)
                              ? vault.get_cv(worst) : 0.0;

            for (int i = 0; i < curr; ++i) {
                if (i == worst) continue;
                double ind_val;
                if (constraint_mode == ConstraintMode::FEASIBILITY) {
                    // Consistency with calculate_fitness: infeasible individuals
                    // take no part in the indicator sums — neither as the removed
                    // x* nor as the updated x (its fitness is a pure penalty).
                    if (cv_worst > 0.0 || vault.get_cv(i) > 0.0) continue;
                    ind_val = eps_indicator_norm(ow, vault.objectives_of(i),
                                                 fmin, fmax);
                } else if (constraint_mode == ConstraintMode::EPS_CONSTRAINT) {
                    double cv_i = vault.get_cv(i);
                    ind_val = eps_indicator_with_cv(ow, vault.objectives_of(i),
                                                    cv_worst, cv_i, fmin, fmax);
                } else {
                    ind_val = eps_indicator_norm(ow, vault.objectives_of(i),
                                                 fmin, fmax);
                }
                vault.get_ind(i).fitness += std::exp(-ind_val / (c * kappa_));
            }
            vault.swap_active(worst, curr - 1);
            vault.reduce(curr - 1);
        }
    }

    void step(DataVault<Ind_t>& vault)
    {
        int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_int(0, n - 1);
        int off_base = vault.expand(n);  // [off_base, off_base+n) — offspring slots
        double pm = resolved_pm(vault);

        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        for (int i = 0; i < n; i += 2) {
            int a = dist_int(rng_), b = dist_int(rng_);
            int p1 = (vault.get_ind(a).fitness > vault.get_ind(b).fitness) ? a : b;
            int cc = dist_int(rng_), d = dist_int(rng_);
            int p2 = (vault.get_ind(cc).fitness > vault.get_ind(d).fitness) ? cc : d;

            for (int j = 0; j < vault.vars_n(); ++j) {
                pv1[j] = vault.get_variable(p1, j);
                pv2[j] = vault.get_variable(p2, j);
            }
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
                vault.set_all_variables(off_base + i, c1, bc1);
                if (i + 1 < n) vault.set_all_variables(off_base + i + 1, c2, bc2);
            } else {
                vault.set_variables(off_base + i, c1);
                if (i + 1 < n) vault.set_variables(off_base + i + 1, c2);
            }
        }
        vault.sync();
        calculate_fitness(vault, n * 2);
        environmental_selection(vault, n);
    }
};

} // namespace mootation
