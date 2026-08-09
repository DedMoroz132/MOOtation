#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// mIBEA — A Modified Indicator-based Evolutionary Algorithm
// Wenwen Li, Ender Özcan, Robert John, John H. Drake, Aneta Neumann,
// Markus Wagner — 2017 IEEE Congress on Evolutionary Computation (CEC),
//   pp. 1047-1054.
// doi:10.1109/CEC.2017.7969423
//
// Generational scheme (Alg.2 = the adaptive Alg.1 of IBEA_HD plus Step 2.1):
//   1. Mating (Step 6): binary tournament on F over the current parent pool.
//   2. Variation (Step 7): SBX (eta_c=20, p_c=0.9) + polynomial mutation
//      (eta_m=20, p_m=1/n); the offspring are appended to P.
//   3. Step 2.1 (the key modification): NSGA-II fast non-dominated sorting,
//      P <- front 0, so the dominated solutions are removed BEFORE scaling.
//      The pool size after the filter is variable and may fall below mu.
//   4. Step 2.2: objective scaling; Step 3: c = max|I_HD| and
//      F(x) = Sum_{y!=x} −exp(−I_HD(y,x)/(c*kappa)), kappa = 0.05 — the
//      NEGATED form; Alg.1 Step 3.2 prints it without the leading minus
//      (see the deviation note below).
//   5. Step 4: environmental selection — iteratively remove argmin F, updating
//      F(x) += exp(−I_HD(x*,x)/(c*kappa)) until |P| = mu; the bounds and c stay
//      fixed.
//   Step 2.1 is applied in setup() as well, before the first fitness
//   computation (the Alg.2 flow: Step 1 -> Step 2.1 -> Step 2.2 -> Step 3).
//
// The rho parameter (Alg.1 Step 2, the "objective values scaling factor"):
//   b'_i = b_i + rho*(b_bar_i − b_i), which in the [0,1] normalization is the
//   reference point r = rho. The default is rho = 2.0 (§IV-B: "the default
//   setting of mIBEA, with rho = 2.0"); per §IV-C / Table III, at
//   rho in [1.0, 1.1] the mIBEA effect disappears, and from rho >= 2 the fronts
//   are stable. set_rho takes rho in the paper's semantics (r = rho).
//
// PAPER DEFAULTS (§IV-A): kappa=0.05, eta_c=eta_m=20, p_c=0.9, p_m=1/n,
//   rho=2.0.
// DECLARED DEVIATIONS: the generational scheme is generational (mu offspring
//   per step), although the letter of Alg.1 Steps 6–7 describes steady-state
//   (mu+1, one offspring, m += 1). The paper is internally ambiguous: Alg.1 is
//   presented as a restatement of [4,17], the generational original IBEA, and
//   the experiments were run in jMetal, which is also generational. It is kept
//   generational by arbitration.
//   Weak dominance in the I_HD branch is functionally neutral.
//   The fitness sign follows the original IBEA, not mIBEA's own printing.
//   Alg.1 Step 3.2 prints F(x1) = Σ exp(−I(x2,x1)/c·κ) with NO leading minus,
//   but Step 4.1 removes the SMALLEST F and Step 4.3 then ADDS exp(...) to the
//   survivors — a positive-sum F with a positive increment is coherent only
//   with the negated form. Zitzler & Kuenzli 2004 (Alg.2, adaptive IBEA) print
//   the minus explicitly, so it is read as a typo. The exponent is likewise
//   read as −I/(c·κ), not (−I/c)·κ.
// EXTENSIONS BEYOND THE PAPER (off by default): ConstraintMode CDP/FEASIBILITY
//   — Step 2.1 non-dominance uses constrained domination; under FEASIBILITY the
//   infeasible solutions receive a penalty fitness of −1e12*(1+cv), and the
//   indicator pairs and c are computed over feasible solutions only,
//   consistently between assign_fitness and env_select.
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
class mIBEACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double       kappa_ = 0.05;   // §IV-A: the original IBEA defaults
    double       rho_   = 2.0;    // §IV-B: ρ = 2.0; r = ρ (Alg.1 Step 2)
    double       eta_c_ = 20.0;   // §IV-A
    double       eta_m_ = 20.0;   // §IV-A
    double       pc_    = 0.9;    // §IV-A: SBX probability 0.9
    double       pm_    = -1.0;   // <0 → 1/n (§IV-A: 1/number of parameters)
    std::mt19937 rng_{std::random_device{}()};

    // ── Dominance helpers (for Step 2.1) ───────────────────────────────────
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

    // ── Step 2.1: filter the active pool down to the non-dominated front ───
    // NSGA-II fast non-dominated sorting, keeping only rank 0.
    // The active pool shrinks to front 0 and may fall below pop_size.
    void filter_to_nondominated(DataVault<Ind_t>& vault) {
        int n = static_cast<int>(vault.active_n());

        std::vector<double> cvs(n, 0.0);
        if (constraint_mode == ConstraintMode::CDP ||
            constraint_mode == ConstraintMode::FEASIBILITY)
            for (int i = 0; i < n; ++i) cvs[i] = vault.get_cv(i);

        std::vector<int> np(n, 0);
        for (int i = 0; i < n; ++i) {
            const auto& fi = vault.objectives_of(i);
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                const auto& fj = vault.objectives_of(j);
                bool j_dom_i;
                if (constraint_mode == ConstraintMode::CDP ||
                    constraint_mode == ConstraintMode::FEASIBILITY)
                    j_dom_i = dominates_cdp(fj, cvs[j], fi, cvs[i]);
                else
                    j_dom_i = dominates_plain(fj, fi);
                if (j_dom_i) ++np[i];
            }
            vault.get_ind(i).rank = (np[i] == 0) ? 0 : 1;
        }

        // Partition: move rank 0 to the front of the active range.
        int write = 0;
        std::vector<int> pos(n), at_pos(n);
        std::iota(pos.begin(), pos.end(), 0);
        std::iota(at_pos.begin(), at_pos.end(), 0);
        for (int i = 0; i < n; ++i) {
            if (vault.get_ind(i).rank != 0) continue;
            if (write != i) {
                vault.swap_active(write, i);
                int other = at_pos[write];
                pos[i]     = write;
                pos[other] = i;
                at_pos[write] = i;
                at_pos[i]     = other;
            }
            ++write;
        }
        if (write < n) vault.reduce(write);
    }

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

    // ── I_HD(A={y}, B={x}) — Zitzler & Künzli 2004, §3.2; li2017 Eq.(1) ────
    //   I_HD(A,B) = IH(B) - IH(A)      if A dominates all of B
    //             = IH(A+B) - IH(A)    otherwise
    // Reference point r = rho in the [0,1] normalization (Alg.1 Step 2:
    // b̄′ᵢ = bᵢ + ρ·(b̄ᵢ − bᵢ)).
    double ihd(const std::vector<double>& fn_y,    // A = {y}
               const std::vector<double>& fn_x)    // B = {x}
        const {
        int m = static_cast<int>(fn_y.size());
        double r = rho_;

        double hv_y = 1.0, hv_x = 1.0, hv_inter = 1.0;
        for (int j = 0; j < m; ++j) {
            hv_y     *= std::max(r - fn_y[j], 0.0);
            hv_x     *= std::max(r - fn_x[j], 0.0);
            hv_inter *= std::max(r - std::max(fn_y[j], fn_x[j]), 0.0);
        }
        double hv_union = hv_y + hv_x - hv_inter;   // IH({y}∪{x})

        bool y_dominates_x = true;
        for (int j = 0; j < m; ++j)
            if (fn_y[j] > fn_x[j]) { y_dominates_x = false; break; }

        if (y_dominates_x)
            return hv_x - hv_y;          // IH(B) - IH(A)   (branch 1, < 0)
        else
            return hv_union - hv_y;      // IH(A+B) - IH(A)  (branch 2, >= 0)
    }

    // c = max|I_HD| over the pairs (Step 3.1); under FEASIBILITY, pairs
    // involving an infeasible solution are excluded, consistently with
    // assign_fitness.
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
    // F(x) = Σ_{y≠x} -exp(-I_HD(y,x) / (c·κ)),  c = max|I_HD|.
    void assign_fitness(DataVault<Ind_t>& vault, int n) {
        std::vector<double> fmin, fmax;
        calc_bounds(vault, n, fmin, fmax);

        std::vector<std::vector<double>> fn(n);
        for (int i = 0; i < n; ++i)
            fn[i] = normalise(vault.objectives_of(i), fmin, fmax);

        std::vector<double> cvs(n, 0.0);
        if (constraint_mode == ConstraintMode::FEASIBILITY)
            for (int i = 0; i < n; ++i) cvs[i] = vault.get_cv(i);

        double c = calc_c(fn, cvs);

        for (int i = 0; i < n; ++i) {
            if (constraint_mode == ConstraintMode::FEASIBILITY && cvs[i] > 0.0) {
                // The penalty base is deliberately below the smallest fitness
                // any feasible individual can reach, even after all the
                // exponential increments of environmental selection
                // (each ≤ e^{1/κ} ≈ 4.9e8).
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

    // ── Environmental selection (Step 4: iteratively remove the worst) ────
    void env_select(DataVault<Ind_t>& vault, int target_n) {
        // Zitzler & Kuenzli 2004 / li2017 Steps 3-4: the normalization bounds
        // and c = max|I_HD| are computed ONCE over the initial pool, already
        // filtered to front 0, and held fixed for the whole removal loop.
        // Normalizing on the fly against fixed bounds makes slot permutations
        // safe. Under FEASIBILITY, c uses feasible pairs only, consistently
        // with
        // assign_fitness).
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

            // FEASIBILITY: increments only among feasible solutions; the
            // infeasible ones never entered the indicator sums and hold a pure
            // penalty.
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

    // ── Tournament (larger fitness is better) ───────────────────────────
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
    mIBEACore() = default;

    void set_kappa        (double k) { kappa_ = k; }
    // rho in the paper's semantics (Alg.1 Step 2): the reference point
    // r = rho in the [0,1] normalization. Default 2.0 (§IV-B); per §IV-C,
    // rho in [1.0, 1.1] is a dead zone and rho >= 2 is the recommended range.
    void set_rho          (double r) { rho_   = r; }
    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_pc           (double p) { pc_    = p; }
    void set_pm           (double p) { pm_    = p; }
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
        // Alg.2: Step 1 -> Step 2.1 (front-0 filter) -> Step 2.2/Step 3
        // (fitness): the first fitness computation already runs on the
        // non-dominated P0.
        filter_to_nondominated(vault);
        assign_fitness(vault, static_cast<int>(vault.active_n()));
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        // The same Alg.2 sequence as in setup().
        filter_to_nondominated(vault);
        assign_fitness(vault, static_cast<int>(vault.active_n()));
    }

    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();

        // CRITICAL: after the previous step()'s filter_to_nondominated(),
        // active_n() may be below pop_size(). The real parent count is read
        // BEFORE expand(), and offspring go into the floating slots
        // n_parents+i.
        int n_parents = vault.parents_n();
        std::uniform_int_distribution<int> dist_int(0, n_parents - 1);

        vault.expand(n);   // n offspring slots: [n_parents, n_parents+n)
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
                vault.set_all_variables(n_parents + i,     c1, bc1);
                if (i + 1 < n) vault.set_all_variables(n_parents + i + 1, c2, bc2);
            } else {
                vault.set_variables(n_parents + i, c1);
                if (i + 1 < n) vault.set_variables(n_parents + i + 1, c2);
            }
        }
        vault.sync();

        // Step 2.1 (mIBEA): filter the merged pool to front 0 BEFORE scaling
        // and before computing the indicator.
        filter_to_nondominated(vault);

        // Step 2.2 + Steps 3-7: scaling, fitness by I_HD, then environmental
        // selection down to pop_size.
        assign_fitness(vault, static_cast<int>(vault.active_n()));
        env_select(vault, n);
    }
};

} // namespace mootation
