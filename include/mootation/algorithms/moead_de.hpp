#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MOEA/D-DE — the algorithm of "Multiobjective Optimization Problems With
// Complicated Pareto Sets, MOEA/D and NSGA-II" (MOEA/D with a differential
// evolution operator; MOEA/D-DE is the paper's own name for it).
// H. Li, Q. Zhang — IEEE Transactions on Evolutionary Computation 13(2), 2009,
// pp. 284-302
// doi:10.1109/TEVC.2008.925798          (source: huili2009)
//
// Generation scheme (Step 2 of the paper, SEQUENTIALLY for i = 1..N):
//   1. Step 2.1: P = B(i) with probability δ, otherwise the whole population {1..N}.
//   2. Step 2.2: r1 = i; r2, r3 drawn at random from P; DE (Eq.6):
//      ȳ_k = x^i_k + F·(x^r2_k − x^r3_k) with prob. CR, otherwise x^i_k;
//      then polynomial mutation (Eq.7) with probability p_m.
//   3. Step 2.3 (repair): a component of y outside the bounds Ω is reset to a
//      RANDOM value inside the domain (DERepair::RandomReset).
//   4. Step 2.4: update z with the component-wise minimum of f(y).
//   5. Step 2.5: up to n_r replacements — random j from P without replacement;
//      if g(y|λ^j,z) ≤ g(x^j|λ^j,z): x^j ← y, FV^j ← F(y) by COPYING
//      (vault.seed_individual, without re-evaluation). FE per generation = exactly N.
//
// Defaults = §IV-A: CR = 1.0, F = 0.5; PM η = 20, p_m = 1/n;
//   T = 20, δ = 0.9, n_r = 2. N is set by the user (paper: N=300 for
//   2-objective, 595 for 3-objective problems); weights — Das–Dennis lattice
//   (= scheme H of the paper, §IV-A-4).
// Deviations:
//   - MDE-3 (MINOR): guaranteed j_rand gene in the binomial crossover —
//     absent from Eq.6 of the paper; with the default CR=1.0 it has no effect.
//   - MDE-4 (MINOR): r2, r3 are kept distinct from i and from each other
//     (the paper does not require distinctness) — a harmless strengthening.
//     Not inert: the rejection loop draws until distinct, so the RNG stream
//     differs from a plain uniform pick.
//   - MDE-5 (MINOR): the Step 2.3 random reset is applied to the DE mutant v
//     INSIDE de_rand_1_bin — i.e. before the binomial crossover select and
//     before PM — whereas Step 2.3 repairs the final y. The invariant y ∈ Ω
//     still holds because the bounded PM (G3) cannot leave the box, but two
//     things differ: at CR<1 a reset draw is consumed for genes the crossover
//     then discards (different RNG stream), and PM sees an in-box value rather
//     than a possibly out-of-box ȳ, which matters because the bounded PM's
//     δ1/δ2 depend on the position of x inside the box.
//   - G3 (MINOR): PM is the bounded (position-dependent δ1/δ2) NSGA-II
//     variant — see operators/poly_mutation.hpp — not the literal Eq.7, whose
//     σ_k does not depend on where x sits in the box.
// Extensions beyond the paper (disabled by default):
//   - EP archive (vault.archive_*): the 2009 paper keeps NO archive
//     (Output = {x^1..x^N}); enabled via set_use_ep(true), adds no extra FE.
//   - ConstraintMode::FEASIBILITY — g^te + penalty·cv (the paper does not
//     consider constraints beyond box domains).
//   - Binary variables: inheritance from x^i + bit-flip 1/n_bin.
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
#include "../operators/de_mutation.hpp"
#include "../operators/poly_mutation.hpp"

namespace mootation {

template <typename Ind_t>
class MOEADDECore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    // ── hyperparameters (all defaults = §IV-A of the paper) ────────────────
    int    T_           = 20;    // §IV-A-6: T = 20
    double delta_       = 0.9;   // §IV-A-6: δ = 0.9
    int    nr_          = 2;     // §IV-A-6: n_r = 2
    double F_           = 0.5;   // §IV-A-1: F = 0.5
    double CR_          = 1.0;   // §IV-A-1: CR = 1.0
    double eta_m_       = 20.0;  // §IV-A-1: η = 20
    double pm_          = -1.0;  // §IV-A-1: p_m = 1/n; <0 → auto 1/n
    double feas_penalty_= 1e6;   // extension beyond the paper (FEASIBILITY)
    bool   use_ep_      = false; // EP archive — beyond the paper, off by default
    std::mt19937 rng_{std::random_device{}()};

    // ── runtime state ──────────────────────────────────────────────────────
    std::vector<double>              ideal_;
    std::vector<std::vector<double>> W_;   // weight vectors [N][m]
    std::vector<std::vector<int>>    B_;   // neighbourhoods [N][T]

    double pm_eff(int nv) const {
        return (pm_ > 0.0) ? pm_ : (nv > 0 ? 1.0 / nv : 0.0);
    }

    // ── Das-Dennis weight vector generation (≡ scheme H, §IV-A-4) ─────────
    void generate_weight_vectors(int n, int m) {
        W_ = das_dennis::generate_exact(m, n);
    }

    // ── Neighbourhood construction (Step 1.1) ─────────────────────────────
    double weight_dist(const std::vector<double>& a,
                       const std::vector<double>& b) const {
        double d = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) {
            double diff = a[j] - b[j];
            d += diff * diff;
        }
        return std::sqrt(d);
    }

    void build_neighbourhoods(int n) {
        int T_eff = std::min(T_, n); B_.resize(n);
        for (int i = 0; i < n; ++i) {
            std::vector<std::pair<double, int>> dists; dists.reserve(n);
            for (int j = 0; j < n; ++j)
                dists.emplace_back(weight_dist(W_[i], W_[j]), j);
            std::partial_sort(dists.begin(), dists.begin() + T_eff, dists.end());
            B_[i].resize(T_eff);
            for (int k = 0; k < T_eff; ++k) B_[i][k] = dists[k].second;
        }
    }

    // ── Tchebycheff scalar function (Eq.5 of the paper) ───────────────────
    double tchebycheff(const std::vector<double>& f,
                       const std::vector<double>& lambda) const {
        double g = 0.0;
        for (std::size_t j = 0; j < f.size(); ++j) {
            double val = lambda[j] * std::abs(f[j] - ideal_[j]);
            if (val > g) g = val;
        }
        return g;
    }

    // Extension beyond the paper (only under ConstraintMode::FEASIBILITY).
    double tchebycheff_feas(const std::vector<double>& f,
                            const std::vector<double>& lambda,
                            double cv) const {
        return tchebycheff(f, lambda) + (cv > 0.0 ? feas_penalty_ * cv : 0.0);
    }

    // ── Ideal point update (Step 2.4) ──────────────────────────────────────
    void update_ideal(const std::vector<double>& f) {
        for (std::size_t j = 0; j < f.size(); ++j)
            ideal_[j] = std::min(ideal_[j], f[j]);
    }

    // ── EP update via vault.archive_* (extension beyond the paper) ────────
    void update_ep(DataVault<Ind_t>& vault, int scratch_slot) {
        const auto& f_y = vault.objectives_of(scratch_slot);
        for (std::size_t k = 0; k < vault.archive_size(); ++k) {
            const auto& ep_f = vault.archive_objectives_of(k);
            bool ep_dom = true, any_b = false;
            for (std::size_t j = 0; j < f_y.size(); ++j) {
                if (ep_f[j] > f_y[j]) { ep_dom = false; break; }
                if (ep_f[j] < f_y[j]) any_b = true;
            }
            if (ep_dom && any_b) return;
        }
        for (int k = static_cast<int>(vault.archive_size()) - 1; k >= 0; --k) {
            const auto& ep_f = vault.archive_objectives_of(static_cast<std::size_t>(k));
            bool y_dom = true, any_b = false;
            for (std::size_t j = 0; j < f_y.size(); ++j) {
                if (f_y[j] > ep_f[j]) { y_dom = false; break; }
                if (f_y[j] < ep_f[j]) any_b = true;
            }
            if (y_dom && any_b)
                vault.archive_erase(static_cast<std::size_t>(k));
        }
        vault.archive_push(scratch_slot);
    }

    // ── DE/rand/1/bin offspring (Step 2.2, Eq.6: r1 = i) ──────────────────
    // Mutation base = x^i (r1=i); r2, r3 — from the pool. Repair Step 2.3 —
    // random reset inside the domain (DERepair::RandomReset, fix G2).
    std::vector<double> de_offspring(DataVault<Ind_t>& vault, int i,
                                     const std::vector<int>& mating_pool,
                                     const std::vector<std::pair<
                                         std::optional<double>,
                                         std::optional<double>>>& bounds) {
        int pool_sz = static_cast<int>(mating_pool.size());
        std::uniform_int_distribution<int> dist_pool(0, pool_sz - 1);

        auto pick_distinct = [&](int exclude1, int exclude2) -> int {
            for (int attempt = 0; attempt < 10; ++attempt) {
                int idx = mating_pool[dist_pool(rng_)];
                if (idx != exclude1 && idx != exclude2) return idx;
            }
            return mating_pool[dist_pool(rng_)];
        };

        int b = pick_distinct(i, -1);
        int c = pick_distinct(i, b);
        int nv = vault.vars_n();

        std::vector<double> x_b(nv), x_c(nv), x_i(nv);
        for (int j = 0; j < nv; ++j) {
            x_b[j] = vault.get_variable(b, j);
            x_c[j] = vault.get_variable(c, j);
            x_i[j] = vault.get_variable(i, j);
        }

        std::vector<double> y;
        // base = x_i, difference pair (x_b − x_c), crossover with x_i;
        // out-of-bound → random reset (Step 2.3).
        ops::de_rand_1_bin(x_i, x_b, x_c, x_i, y, bounds, F_, CR_,
                           ops::DERepair::RandomReset, rng_);
        return y;
    }

    // ── Update step for one subproblem i (Steps 2.1–2.5) ──────────────────
    void update_subproblem(DataVault<Ind_t>& vault, int i, int n) {
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        const auto& bounds = vault.get_bounds();

        // Step 2.1: mating/update range P.
        const auto& B = B_[i];
        std::vector<int> mating_pool;
        if (U01(rng_) < delta_) {
            mating_pool = B;
        } else {
            mating_pool.resize(n);
            std::iota(mating_pool.begin(), mating_pool.end(), 0);
        }

        // Step 2.2: DE offspring (Eq.6) + polynomial mutation (Eq.7).
        auto y_vars = de_offspring(vault, i, mating_pool, bounds);
        ops::polynomial_mutation(y_vars, bounds, eta_m_,
                                 pm_eff(vault.vars_n()), rng_);

        // Binary variables (extension beyond the paper): inherited from x^i
        // + bit-flip with probability 1/n_bin.
        int nb = vault.bin_vars_n();

        // Evaluate the offspring in the scratch slot (index n): exactly ONE
        // objective function call per offspring.
        int scratch = n;
        if (nb > 0) {
            std::vector<int> y_bvars(nb);
            std::uniform_real_distribution<double> Ub(0.0, 1.0);
            double p_bin = 1.0 / static_cast<double>(nb);
            for (int j = 0; j < nb; ++j) {
                y_bvars[j] = vault.get_bin_variable(i, j);
                if (Ub(rng_) < p_bin) y_bvars[j] ^= 1;
            }
            vault.set_all_variables(scratch, y_vars, y_bvars);
        } else {
            vault.set_variables(scratch, y_vars);
        }
        vault.refresh_objectives(scratch);

        const auto& f_y = vault.objectives_of(scratch);
        double cv_y = (constraint_mode != ConstraintMode::NONE)
                      ? vault.get_cv(scratch) : 0.0;

        // Step 2.4: update z.
        update_ideal(f_y);

        // Step 2.5: up to n_r replacements at random j from P without
        // replacement (shuffle is equivalent). The replacement x^j ← y,
        // FV^j ← F(y) is done by COPYING via seed_individual — without
        // re-evaluation: re-evaluating would spend one extra FE per
        // replacement, so a generation would cost N + (replacements), not N.
        std::vector<int> upd_order = mating_pool;
        std::shuffle(upd_order.begin(), upd_order.end(), rng_);

        int count = 0;
        for (int j : upd_order) {
            if (count >= nr_) break;
            const auto& lam_j = W_[j];
            double g_y, g_xj;
            if (constraint_mode == ConstraintMode::FEASIBILITY) {
                g_y  = tchebycheff_feas(f_y, lam_j, cv_y);
                g_xj = tchebycheff_feas(vault.objectives_of(j), lam_j,
                                        vault.get_cv(j));
            } else {
                g_y  = tchebycheff(f_y,  lam_j);
                g_xj = tchebycheff(vault.objectives_of(j), lam_j);
            }
            if (g_y <= g_xj) {
                vault.seed_individual(j,
                                      vault.variables_of(scratch),
                                      vault.objectives_of(scratch),
                                      vault.binary_variables_of(scratch),
                                      vault.limits_of(scratch));
                vault.get_ind(j).scalar_fitness = g_y;
                ++count;
            }
        }

        // EP (beyond the paper, optional).
        if (use_ep_) update_ep(vault, scratch);
    }

public:
    MOEADDECore() = default;

    void set_neighbourhood_size(int T)   { T_            = T; }
    void set_T                 (int T)   { T_            = T; }   // alias
    void set_delta             (double d){ delta_         = d; }
    void set_nr                (int n)   { nr_            = n; }
    void set_F                 (double f){ F_             = f; }
    void set_CR                (double c){ CR_            = c; }
    void set_eta_crossover     (double)  {}  // DE uses CR/F, not SBX eta_c
    void set_eta_mutation      (double e){ eta_m_         = e; }
    void set_pm                (double p){ pm_            = p; }
    void set_feas_penalty      (double p){ feas_penalty_  = p; }
    void set_use_ep            (bool b)  { use_ep_        = b; }
    void set_seed              (unsigned s){ rng_.seed(s); }

    std::size_t external_population_size(DataVault<Ind_t>& vault) const {
        return vault.archive_size();
    }

    // ── setup (Step 1) ─────────────────────────────────────────────────────
    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int m = vault.objs_n();

        generate_weight_vectors(n, m);
        build_neighbourhoods(n);

        // Step 1.2: random initialisation.
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dist_real(0.0, 1.0);
        std::uniform_int_distribution<int>     dist_bin (0, 1);
        std::vector<double> vars (vault.vars_n());
        std::vector<int>    bvars(vault.bin_vars_n());

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < vault.vars_n(); ++j) {
                double lo = bounds[j].first .value_or(0.0);
                double hi = bounds[j].second.value_or(1.0);
                vars[j] = lo + dist_real(rng_) * (hi - lo);
            }
            for (int j = 0; j < vault.bin_vars_n(); ++j) bvars[j] = dist_bin(rng_);
            if (vault.bin_vars_n() > 0) vault.set_all_variables(i, vars, bvars);
            else                        vault.set_variables    (i, vars);
        }
        vault.sync();

        // Step 1.3: z_j = min_i f_j(x^i).
        ideal_.assign(m, std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) update_ideal(vault.objectives_of(i));

        vault.archive_clear();

        for (int i = 0; i < n; ++i) {
            const auto& lam = W_[i];
            double cv = (constraint_mode != ConstraintMode::NONE)
                        ? vault.get_cv(i) : 0.0;
            vault.get_ind(i).scalar_fitness =
                (constraint_mode == ConstraintMode::FEASIBILITY)
                ? tchebycheff_feas(vault.objectives_of(i), lam, cv)
                : tchebycheff(vault.objectives_of(i), lam);
        }

        // Keep one scratch slot at index n.
        vault.expand(1);
        vault.reduce(n + 1);
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int m = vault.objs_n();

        if (W_.empty()) { generate_weight_vectors(n, m); build_neighbourhoods(n); }

        ideal_.assign(m, std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) update_ideal(vault.objectives_of(i));

        vault.archive_clear();
        if (use_ep_)
            for (int i = 0; i < n; ++i) update_ep(vault, i);

        for (int i = 0; i < n; ++i) {
            const auto& lam = W_[i];
            double cv = (constraint_mode != ConstraintMode::NONE)
                        ? vault.get_cv(i) : 0.0;
            vault.get_ind(i).scalar_fitness =
                (constraint_mode == ConstraintMode::FEASIBILITY)
                ? tchebycheff_feas(vault.objectives_of(i), lam, cv)
                : tchebycheff(vault.objectives_of(i), lam);
        }

        vault.expand(1);
        vault.reduce(n + 1);
    }

    // ── step: one generation = Step 2 for i = 1..N SEQUENTIALLY ───────────
    // ("For i = 1, ..., N, do" — fix G4).
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        for (int i = 0; i < n; ++i)
            update_subproblem(vault, i, n);
    }
};

} // namespace mootation
