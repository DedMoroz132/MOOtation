#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MOEA/D-DRA — the algorithm of "The performance of a new version of MOEA/D
// on CEC09 unconstrained MOP test instances" (MOEA/D with dynamical resource
// allocation; MOEA/D-DRA is the paper's own name for it).
// Q. Zhang, W. Liu, H. Li — IEEE CEC 2009, pp. 203-208 (winner of the CEC09
// MOEA competition)
// doi:10.1109/CEC.2009.4982949          (source: zhang2009)
//
// Generation scheme:
//   1. Step 2: I = indexes of the subproblems whose objectives are individual
//      f_j of the MOP (unit-vector weights e_j); a 10-tournament on π adds
//      ⌊N/5⌋ − m more indexes, so |I| = max(m, ⌊N/5⌋) in total. The paper
//      writes ⌊N/5⌋, which silently assumes ⌊N/5⌋ ≥ m; when it is not (small N
//      at large m — at m=10 the smallest Das–Dennis lattice is N=10, giving
//      ⌊N/5⌋=2), the m unit-vector indexes are already more than the target
//      and nothing is added. See DRA-9.
//   2. Step 3 (for each i ∈ I): 3.1 P = B(i) with prob. δ, otherwise {1..N};
//      3.2 r1 = i, r2, r3 from P, DE (Eq.4) + polynomial mutation (Eq.5);
//      3.3 repair — random reset inside the domain (DERepair::RandomReset);
//      3.4 update z; 3.5 up to n_r replacements at random j ∈ P without
//      replacement, x^j ← y / FV^j ← F(y) by COPYING (seed_individual,
//      without re-evaluation; FE per generation = exactly |I|).
//   3. Step 5: every 50 generations π^i ← 1 if Δ^i > 0.001, otherwise
//      (0.95 + 0.05·Δ^i/0.001)·π^i; Δ^i = (old − new)/old WITHOUT clipping.
//
// Defaults = Sec.III: T = 0.1N, n_r = 0.01N (computed in setup; the
//   set_T/set_nr setters override), δ = 0.9; CR = 1.0, F = 0.5, η = 20,
//   p_m = 1/n; π update period = 50 (Step 5).
// Deviations:
//   - DRA-5 (MINOR): weights — Das–Dennis lattice instead of the authors'
//     maximin selection of N out of 5000 random vectors (Sec.III); the unit
//     vectors e_j are present in the lattice.
//   - G3 (MINOR): PM is the bounded (position-dependent δ1/δ2) NSGA-II
//     variant — see operators/poly_mutation.hpp — not the literal Eq.5.
//   - Numerical guard old_g > 1e-14 for the division in Δ (the paper does
//     not address division by 0).
//   - DRA-6 (MINOR): guaranteed j_rand gene in the binomial crossover — Eq.4
//     of the paper has no such term; with the default CR=1.0 it has no effect.
//     Same deviation as MDE-3 in moead_de.hpp (shared operator).
//   - DRA-7 (MINOR): r2, r3 are kept distinct from i and from each other;
//     Step 3.2 says only "randomly select two indexes r_2 and r_3 from P".
//     Not inert — the rejection loop draws until distinct, so the RNG stream
//     differs. Same as MDE-4.
//   - DRA-8 (MINOR): the Step 3.3 random reset is applied to the DE mutant v
//     inside de_rand_1_bin, before the crossover select and before PM, not to
//     the final y. Same as MDE-5 — see moead_de.hpp for the consequences.
//   - DRA-9 (MINOR): the 10-tournament draws WITHOUT replacement from the
//     not-yet-selected indexes (Step 2 says "select OTHER ⌊N/5⌋ − m indexes",
//     which reads as "not already in I"); it shrinks to min(10, |candidates|)
//     when fewer than ten remain, a case the paper does not contemplate. Note
//     moead_awa implements the same step with the opposite convention.
// Extensions beyond the paper (disabled by default):
//   - EP archive (vault.archive_*): in the paper Output = {x^1..x^N};
//     enabled via set_use_ep(true), adds no extra FE.
//   - ConstraintMode::FEASIBILITY — g^te + penalty·cv.
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
class MOEADDRACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    // ── hyperparameters (defaults = Sec.III of the paper) ─────────────────
    int    T_            = -1;    // Sec.III: T = 0.1N; <0 → auto in setup
    int    nr_           = -1;    // Sec.III: n_r = 0.01N; <0 → auto in setup
    int    eta_update_   = 50;    // Step 5: π update period
    double delta_        = 0.9;   // Sec.III: δ = 0.9
    double F_            = 0.5;   // Sec.III: F = 0.5
    double CR_           = 1.0;   // Sec.III: CR = 1.0
    double eta_m_        = 20.0;  // Sec.III: η = 20
    double pm_           = -1.0;  // Sec.III: p_m = 1/n; <0 → auto 1/n
    double feas_penalty_ = 1e6;   // extension beyond the paper (FEASIBILITY)
    bool   use_ep_       = false; // EP — beyond the paper, off by default
    std::mt19937 rng_{std::random_device{}()};

    int T_eff_  = 0;   // effective T (after setup)
    int nr_eff_ = 0;   // effective n_r (after setup)

    // ── runtime state ────────────────────────────────────────────────────────
    std::vector<std::vector<double>> W_;      // weight vectors [N][m]
    std::vector<std::vector<int>>    B_;      // neighbourhoods [N][T]
    std::vector<double>              ideal_;  // z

    std::vector<double> utility_;    // π^i
    std::vector<double> old_g_;      // snapshot of g^te at last utility update
    int                 gen_count_ = 0;

    double pm_eff(int nv) const {
        return (pm_ > 0.0) ? pm_ : (nv > 0 ? 1.0 / nv : 0.0);
    }

    // ── EP update via vault.archive_* (extension beyond the paper) ─────────
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

    // ── Das-Dennis weight vectors (DRA-5: instead of maximin from 5000) ────
    void init_weights(int n, int m) {
        W_ = das_dennis::generate_exact(m, n);
    }

    void build_neighbourhood(int n) {
        int T = std::min(T_eff_, n); B_.resize(n);
        for (int i = 0; i < n; ++i) {
            std::vector<std::pair<double,int>> dists; dists.reserve(n);
            for (int j = 0; j < n; ++j) {
                double d = 0.0;
                for (std::size_t k = 0; k < W_[i].size(); ++k) {
                    double diff = W_[i][k]-W_[j][k]; d += diff*diff;
                }
                dists.emplace_back(d, j);
            }
            std::partial_sort(dists.begin(), dists.begin()+T, dists.end());
            B_[i].resize(T);
            for (int k = 0; k < T; ++k) B_[i][k] = dists[k].second;
        }
    }

    // ── Tchebycheff (Eq.3 of the paper) ─────────────────────────────────────
    double tche(const std::vector<double>& f, const std::vector<double>& w) const {
        double g = 0.0;
        for (std::size_t j = 0; j < f.size(); ++j)
            g = std::max(g, w[j] * std::abs(f[j] - ideal_[j]));
        return g;
    }
    double tche_feas(const std::vector<double>& f, const std::vector<double>& w,
                     double cv) const {
        return tche(f, w) + feas_penalty_ * cv;
    }
    double g_of(DataVault<Ind_t>& vault, int slot) const {
        double cv = (constraint_mode == ConstraintMode::FEASIBILITY)
                    ? vault.get_cv(slot) : 0.0;
        return (constraint_mode == ConstraintMode::FEASIBILITY)
               ? tche_feas(vault.objectives_of(slot), W_[slot], cv)
               : tche(vault.objectives_of(slot), W_[slot]);
    }

    // ── Utility update (Step 5) ──────────────────────────────────────────────
    // Δ^i = (old − new)/old. The paper does NOT clip negative Δ, and this port
    // does not either: for Δ < 0 the factor (0.95 + 0.05·Δ/0.001) drops below
    // 0.95, which is what Step 5 says even though it is never discussed.
    void update_utility(DataVault<Ind_t>& vault, int n) {
        for (int i = 0; i < n; ++i) {
            double g_new = g_of(vault, i);
            double delta = 0.0;
            if (old_g_[i] > 1e-14)
                delta = (old_g_[i] - g_new) / old_g_[i];

            if (delta > 0.001)
                utility_[i] = 1.0;
            else
                utility_[i] = (0.95 + 0.05 * delta / 0.001) * utility_[i];

            old_g_[i] = g_new;
        }
    }

    // ── Step 2: building I ──────────────────────────────────────────────────
    // The initial I — indexes of the subproblems "whose objectives are MOP
    // individual objectives f_i", i.e. subproblems with unit-vector weights
    // e_j (for them g^te = λ_j·|f_j − z_j| = |f_j − z_j|). Then a
    // 10-tournament on π adds ⌊N/5⌋ − m more indexes; |I| = max(m, ⌊N/5⌋) in
    // total — see DRA-9 for the max(). Two earlier readings were rejected:
    // taking the argmin-f_j holders instead of the e_j subproblems, and adding
    // ⌊N/5⌋ indexes ON TOP of the m rather than up to ⌊N/5⌋.
    std::vector<int> build_active_set(DataVault<Ind_t>& vault, int n) {
        int m = vault.objs_n();
        std::vector<bool> in_I(n, false);
        std::vector<int>  I;
        I.reserve(std::max(n / 5, m));

        // Boundary subproblems: the weight with the maximal j-th component
        // (in the Das–Dennis lattice this is exactly the unit vector e_j).
        for (int j = 0; j < m; ++j) {
            int best = 0;
            for (int i = 1; i < n; ++i)
                if (W_[i][j] > W_[best][j]) best = i;
            if (!in_I[best]) { in_I[best] = true; I.push_back(best); }
        }

        // 10-tournament on π: ⌊N/5⌋ − m additions (without repeats).
        int extra = n / 5 - static_cast<int>(I.size());
        std::vector<int> candidates;
        candidates.reserve(n - static_cast<int>(I.size()));
        for (int i = 0; i < n; ++i)
            if (!in_I[i]) candidates.push_back(i);

        for (int pick = 0; pick < extra && !candidates.empty(); ++pick) {
            int t_size = std::min(10, static_cast<int>(candidates.size()));
            std::uniform_int_distribution<int> dc(0, static_cast<int>(candidates.size())-1);
            int winner_idx = candidates[dc(rng_)];
            for (int t = 1; t < t_size; ++t) {
                int cand_idx = candidates[dc(rng_)];
                if (utility_[cand_idx] > utility_[winner_idx])
                    winner_idx = cand_idx;
            }
            auto it = std::find(candidates.begin(), candidates.end(), winner_idx);
            if (it != candidates.end()) candidates.erase(it);
            in_I[winner_idx] = true;
            I.push_back(winner_idx);
        }

        return I;
    }

    // ── Step 3 for one subproblem i ─────────────────────────────────────────
    void update_subproblem(DataVault<Ind_t>& vault, int i, int scratch) {
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        int N = static_cast<int>(W_.size());

        // Step 3.1: mating/update range P.
        std::vector<int> P;
        if (U01(rng_) < delta_) {
            P = B_[i];
        } else {
            P.resize(N);
            std::iota(P.begin(), P.end(), 0);
        }

        // Step 3.2: DE/rand/1 — r1 = i, r2, r3 from P.
        int psz = static_cast<int>(P.size());
        std::uniform_int_distribution<int> dP(0, psz - 1);
        auto pick = [&](int e1, int e2) -> int {
            for (int t = 0; t < 10; ++t) {
                int idx = P[dP(rng_)];
                if (idx != e1 && idx != e2) return idx;
            }
            return P[dP(rng_)];
        };
        int r2 = pick(i, -1);
        int r3 = pick(i, r2);

        int nv = vault.vars_n();
        std::vector<double> x_i(nv), x_2(nv), x_3(nv), y;
        for (int j = 0; j < nv; ++j) {
            x_i[j] = vault.get_variable(i,  j);
            x_2[j] = vault.get_variable(r2, j);
            x_3[j] = vault.get_variable(r3, j);
        }
        // ȳ = x_i + F·(x_2 − x_3), trial with x_i; Step 3.3: out-of-bound →
        // random reset inside the domain (DERepair::RandomReset, fix G2).
        ops::de_rand_1_bin(x_i, x_2, x_3, x_i, y, bounds, F_, CR_,
                           ops::DERepair::RandomReset, rng_);
        ops::polynomial_mutation(y, bounds, eta_m_, pm_eff(nv), rng_);

        // Evaluate the offspring in the scratch slot: exactly ONE objective
        // function call.
        if (vault.bin_vars_n() > 0) {
            // Binary variables (beyond the paper): inherited from x_i + bit-flip.
            int nb = vault.bin_vars_n();
            std::vector<int> by(nb);
            std::uniform_real_distribution<double> Ub(0.0, 1.0);
            double pb = 1.0 / static_cast<double>(nb);
            for (int j = 0; j < nb; ++j) {
                by[j] = vault.get_bin_variable(i, j);
                if (Ub(rng_) < pb) by[j] ^= 1;
            }
            vault.set_all_variables(scratch, y, by);
        } else {
            vault.set_variables(scratch, y);
        }
        vault.refresh_objectives(scratch);

        // Step 3.4: update z.
        const auto& fy = vault.objectives_of(scratch);
        for (int j = 0; j < static_cast<int>(ideal_.size()); ++j)
            ideal_[j] = std::min(ideal_[j], fy[j]);

        double cv_y = (constraint_mode == ConstraintMode::FEASIBILITY)
                      ? vault.get_cv(scratch) : 0.0;

        // Step 3.5: up to n_r replacements at random j ∈ P without replacement.
        // x^j ← y, FV^j ← F(y) — by COPYING (seed_individual), without
        // re-evaluation (fix G1 / X2, internal audit: DRA-6).
        std::vector<int> upd = P;
        std::shuffle(upd.begin(), upd.end(), rng_);
        int c = 0;
        for (int j : upd) {
            if (c >= nr_eff_) break;
            double g_y  = (constraint_mode == ConstraintMode::FEASIBILITY)
                          ? tche_feas(fy, W_[j], cv_y)
                          : tche(fy, W_[j]);
            double cv_j = (constraint_mode == ConstraintMode::FEASIBILITY)
                          ? vault.get_cv(j) : 0.0;
            double g_xj = (constraint_mode == ConstraintMode::FEASIBILITY)
                          ? tche_feas(vault.objectives_of(j), W_[j], cv_j)
                          : tche(vault.objectives_of(j), W_[j]);
            if (g_y <= g_xj) {
                vault.seed_individual(j,
                                      vault.variables_of(scratch),
                                      vault.objectives_of(scratch),
                                      vault.binary_variables_of(scratch),
                                      vault.limits_of(scratch));
                vault.get_ind(j).scalar_fitness = g_y;
                ++c;
            }
        }

        // EP (beyond the paper, optional).
        if (use_ep_) update_ep(vault, scratch);
    }

    void resolve_params(int n) {
        // Sec.III: T = 0.1N, n_r = 0.01N (unless set explicitly via setters).
        T_eff_  = (T_  > 0) ? T_  : std::max(2, static_cast<int>(0.1  * n));
        nr_eff_ = (nr_ > 0) ? nr_ : std::max(1, static_cast<int>(0.01 * n));
        T_eff_  = std::min(T_eff_, n);
    }

public:
    MOEADDRACore() = default;

    void set_T               (int t)    { T_            = t; }
    void set_nr              (int n)    { nr_           = n; }
    void set_eta_update      (int e)    { eta_update_   = e; }
    void set_delta           (double d) { delta_        = d; }
    void set_F               (double f) { F_            = f; }
    void set_CR              (double c) { CR_           = c; }
    void set_eta_crossover   (double)   {}  // DE uses CR/F, not SBX eta_c
    void set_eta_mutation    (double e) { eta_m_        = e; }
    void set_pm              (double p) { pm_           = p; }
    void set_feas_penalty    (double p) { feas_penalty_ = p; }
    void set_use_ep          (bool b)   { use_ep_       = b; }
    void set_seed            (unsigned s){ rng_.seed(s); }

    int effective_T()  const { return T_eff_;  }
    int effective_nr() const { return nr_eff_; }

    std::size_t external_population_size(DataVault<Ind_t>& vault) const {
        return vault.archive_size();
    }

    // ── setup (Step 1) ───────────────────────────────────────────────────────
    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int m = vault.objs_n();

        resolve_params(n);
        init_weights(n, m);
        build_neighbourhood(n);

        ideal_.assign(m, std::numeric_limits<double>::max());

        // Step 1.2: random initialisation.
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

        // Step 1.3: z_j = min_i f_j(x^i).
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) ideal_[j] = std::min(ideal_[j], o[j]);
        }

        // Step 1.4: gen = 0, π^i = 1; snapshot old_g.
        utility_.assign(n, 1.0);
        old_g_.resize(n);
        for (int i = 0; i < n; ++i) old_g_[i] = g_of(vault, i);

        vault.archive_clear();
        if (use_ep_)
            for (int i = 0; i < n; ++i) update_ep(vault, i);

        // Scratch slot at index n.
        vault.expand(1);
        vault.reduce(n + 1);
        gen_count_ = 0;
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int m = vault.objs_n();
        resolve_params(n);
        if (W_.empty()) { init_weights(n, m); build_neighbourhood(n); }
        ideal_.assign(m, std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) ideal_[j] = std::min(ideal_[j], o[j]);
        }
        utility_.assign(n, 1.0);
        old_g_.resize(n);
        for (int i = 0; i < n; ++i) old_g_[i] = g_of(vault, i);
        vault.archive_clear();
        if (use_ep_)
            for (int i = 0; i < n; ++i) update_ep(vault, i);
        vault.expand(1); vault.reduce(n + 1);
        gen_count_ = 0;
    }

    // ── step ─────────────────────────────────────────────────────────────────
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int scratch = n;   // scratch slot always at index n

        // Step 2: build I (boundary e_j + 10-tournament, |I| = max(m, ⌊N/5⌋)).
        auto I = build_active_set(vault, n);

        // Step 3: "For each i ∈ I, do" — in the order I was built (fix G4:
        // no additional order randomisation is prescribed by the paper).
        for (int i : I)
            update_subproblem(vault, i, scratch);

        // Step 5: update π every eta_update_ (=50) generations.
        ++gen_count_;
        if (gen_count_ % eta_update_ == 0)
            update_utility(vault, n);
    }
};

} // namespace mootation
