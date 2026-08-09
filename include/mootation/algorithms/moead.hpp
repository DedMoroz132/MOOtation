#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MOEA/D — A Multiobjective Evolutionary Algorithm Based on Decomposition
// Q. Zhang, H. Li — IEEE Transactions on Evolutionary Computation 11(6), 2007
// doi:10.1109/TEVC.2007.892759          (source: qingfuzhang2007)
//
// Generation scheme (Step 2 of the paper, SEQUENTIALLY for i = 1..N):
//   1. Step 2.1: two random indexes k, l from B(i); offspring y = SBX(x^k, x^l)
//      (one offspring from the pair, §V-E) + polynomial mutation.
//   2. Step 2.2 (repair): for box domains this is handled by operator clamping.
//   3. Step 2.3: update z with the component-wise minimum of f(y).
//   4. Step 2.4: for each j ∈ B(i): if g^te(y|λ^j,z) ≤ g^te(x^j|λ^j,z),
//      then x^j ← y and FV^j ← F(y) — by COPYING the ready objectives
//      (vault.seed_individual), without re-evaluation. FE per generation = exactly N.
//   5. Step 2.5: update of the external archive EP (vault.archive_*).
//
// Defaults = §V-E: T = 20; SBX η_c = 20, p_c = 1.0; PM η_m = 20, p_m = 1/n.
//   N is set by the user (paper: N=100 for 2-objective, N=300 for 3-objective
//   continuous problems); weights — Das–Dennis lattice (das_dennis::generate_exact,
//   N must be attainable by the lattice).
// ORIENTATION (not a deviation — the paper's own footnote prescribes it). The
//   2007 paper is written for MAXIMIZATION: Eq.3 defines the reference point as
//   z*_i = max{f_i(x) | x ∈ Ω}, and Step 2.3 accordingly reads "if z_j < f_j(y')
//   then set z_j = f_j(y')". Footnote 3 gives the minimization form outright —
//   "In the case when the goal of (1) is minimization, z*_i = min{f_i(x)...}" —
//   and footnote 2 does the same for the objective sense. This library
//   minimizes, so z is updated with the component-wise MINIMUM. Reading Step
//   2.3 literally under a minimizing objective would drive z to the nadir and
//   invert the whole scalarization; the flip is the paper's instruction, not an
//   edit to it.
// Deviations:
//   - MOEAD-2 (MINOR): up to 5 attempts to ensure k ≠ l (the paper allows k = l).
//   - EP may contain duplicate F-vectors (does not contradict the letter of Step 2.5).
// Extensions beyond the paper (disabled by default):
//   - ConstraintMode::FEASIBILITY — g^te + penalty·cv; the 2007 paper handles
//     constraints with a problem-specific repair heuristic (Step 2.2), not a penalty.
//   - Binary variables: UNIFORM crossover + bit-flip (general-purpose; active
//     only when bin_vars_n() > 0). Note the paper's MOKP implementation of §IV
//     specifies one-point crossover ("the genetic operators used are the
//     one-point crossover operator and the standard mutation operator"); this
//     library provides no one-point operator, so the binary path is NOT a
//     reproduction of §IV.
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
class MOEADCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    // ── hyperparameters (for sources see the file header) ──────────────────
    int    T_           = 20;     // §V-E: "T is set to be 20"
    double eta_c_       = 20.0;   // §V-E: distribution index SBX = 20
    double eta_m_       = 20.0;   // §V-E: distribution index PM  = 20
    double pc_          = 1.0;    // §V-E: "The crossover rate is 1.00"
    double pm_          = -1.0;   // §V-E: "mutation rate is 1/n"; <0 → auto 1/n
    double feas_penalty_= 1e6;    // extension beyond the paper (FEASIBILITY)
    std::mt19937 rng_{std::random_device{}()};

    // ── runtime state ──────────────────────────────────────────────────────
    std::vector<double>              ideal_;   // z: substitute z*, size m
    std::vector<std::vector<double>> W_;       // weight vectors [N][m]
    std::vector<std::vector<int>>    B_;       // neighbourhoods [N][T]
    // EP: external non-dominated archive — stored in vault.archive_*

    double pm_eff(int nv) const {
        return (pm_ > 0.0) ? pm_ : (nv > 0 ? 1.0 / nv : 0.0);
    }

    // ── Das-Dennis weight vector generation (Step 1, §V-E) ─────────────────
    void generate_weight_vectors(int n, int m) {
        W_ = das_dennis::generate_exact(m, n);
    }

    // ── Euclidean distance between two weight vectors ──────────────────────
    double weight_dist(const std::vector<double>& a,
                       const std::vector<double>& b) const {
        double d = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) {
            double diff = a[j] - b[j];
            d += diff * diff;
        }
        return std::sqrt(d);
    }

    // ── Build neighbourhood B(i) for all subproblems (Step 1.2) ───────────
    void build_neighbourhoods(int n) {
        int T_eff = std::min(T_, n);
        B_.resize(n);
        for (int i = 0; i < n; ++i) {
            std::vector<std::pair<double, int>> dists;
            dists.reserve(n);
            for (int j = 0; j < n; ++j)
                dists.emplace_back(weight_dist(W_[i], W_[j]), j);
            std::partial_sort(dists.begin(), dists.begin() + T_eff, dists.end());
            B_[i].resize(T_eff);
            for (int k = 0; k < T_eff; ++k) B_[i][k] = dists[k].second;
        }
    }

    // ── Tchebycheff scalar function (Eq.6 of the paper) ───────────────────
    // g^te(x | λ, z) = max_j { λ_j * |f_j(x) - z_j| }
    double tchebycheff(const std::vector<double>& f,
                       const std::vector<double>& lambda) const {
        double g = 0.0;
        for (std::size_t j = 0; j < f.size(); ++j) {
            double val = lambda[j] * std::abs(f[j] - ideal_[j]);
            if (val > g) g = val;
        }
        return g;
    }

    // Extension BEYOND the paper (active only under ConstraintMode::FEASIBILITY):
    // g^te + penalty·cv. The 2007 paper handles constraints with the Step 2.2
    // repair heuristic and does not define any penalty form.
    double tchebycheff_feasibility(const std::vector<double>& f,
                                   const std::vector<double>& lambda,
                                   double cv) const {
        double g = tchebycheff(f, lambda);
        if (cv > 0.0) g += feas_penalty_ * cv;
        return g;
    }

    // ── Ideal point update (Step 2.3) ──────────────────────────────────────
    void update_ideal(const std::vector<double>& f) {
        for (std::size_t j = 0; j < f.size(); ++j)
            ideal_[j] = std::min(ideal_[j], f[j]);
    }

    // ── EP update via vault.archive_* (Step 2.5) ─────────────────────────
    // scratch_slot: active index of the offspring (already evaluated).
    void update_ep(DataVault<Ind_t>& vault, int scratch_slot) {
        const auto& f_y = vault.objectives_of(scratch_slot);
        // Check if any archive member dominates y.
        for (std::size_t k = 0; k < vault.archive_size(); ++k) {
            const auto& ep_f = vault.archive_objectives_of(k);
            bool ep_dom = true, any_better = false;
            for (std::size_t j = 0; j < f_y.size(); ++j) {
                if (ep_f[j] > f_y[j]) { ep_dom = false; break; }
                if (ep_f[j] < f_y[j]) any_better = true;
            }
            if (ep_dom && any_better) return;  // y is dominated — don't add
        }
        // Remove all archive members dominated by y (iterate backwards).
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

    // ── One SBX + PM offspring from two parents (Step 2.1, §V-E) ──────────
    std::vector<double> breed_one(DataVault<Ind_t>& vault,
                                  int p1, int p2,
                                  const std::vector<std::pair<
                                      std::optional<double>,
                                      std::optional<double>>>& bounds) {
        int nv = vault.vars_n();
        std::vector<double> pv1(nv), pv2(nv), c1, c2;
        for (int j = 0; j < nv; ++j) {
            pv1[j] = vault.get_variable(p1, j);
            pv2[j] = vault.get_variable(p2, j);
        }
        // §V-E: p_c = 1.0; "the crossover operator generates one offspring,
        // which is then modified by the mutation operator" — we take c1.
        ops::sbx(pv1, pv2, c1, c2, bounds, eta_c_, pc_, rng_);
        ops::polynomial_mutation(c1, bounds, eta_m_, pm_eff(nv), rng_);
        return c1;
    }

    // Binary variables version helper (extension beyond the paper).
    std::vector<int> breed_bin(DataVault<Ind_t>& vault, int p1, int p2) {
        int nb = vault.bin_vars_n();
        if (nb == 0) return {};
        std::vector<int> bv1(nb), bv2(nb), bc1, bc2;
        for (int j = 0; j < nb; ++j) {
            bv1[j] = vault.get_bin_variable(p1, j);
            bv2[j] = vault.get_bin_variable(p2, j);
        }
        ops::binary_crossover(bv1, bv2, bc1, bc2, rng_);
        ops::bit_flip_mutation(bc1, nb, rng_);
        return bc1;
    }

    // ── Update step for one subproblem i (Steps 2.1–2.5) ─────────────────
    void update_subproblem(DataVault<Ind_t>& vault, int i) {
        const auto& B = B_[i];
        int T_eff = static_cast<int>(B.size());

        // Step 2.1: pick two indexes k, l from B(i).
        std::uniform_int_distribution<int> dist_B(0, T_eff - 1);
        int k_idx = dist_B(rng_);
        int l_idx = dist_B(rng_);
        // MOEAD-2 (MINOR, beyond the letter of the paper): up to 5 attempts to ensure k ≠ l.
        for (int attempt = 0; attempt < 5 && l_idx == k_idx; ++attempt)
            l_idx = dist_B(rng_);
        int p1 = B[k_idx], p2 = B[l_idx];

        // Breed offspring y.
        const auto& bounds = vault.get_bounds();
        auto y_vars = breed_one(vault, p1, p2, bounds);
        auto y_bvars = breed_bin(vault, p1, p2);

        // Evaluate the offspring in the scratch slot (active index n): exactly
        // ONE objective function call per offspring (Step 1.3/2.3: FV = F(y)).
        int scratch = static_cast<int>(vault.active_n()) - 1;
        if (y_bvars.empty()) vault.set_variables(scratch, y_vars);
        else                  vault.set_all_variables(scratch, y_vars, y_bvars);
        vault.refresh_objectives(scratch);

        const auto& f_y   = vault.objectives_of(scratch);
        double cv_y = (constraint_mode != ConstraintMode::NONE)
                      ? vault.get_cv(scratch) : 0.0;

        // Step 2.3: update z.
        update_ideal(f_y);

        // Step 2.4: update neighbouring solutions.
        // "set x^j = y' and FV^j = F(y')" — objectives are TRANSFERRED from scratch
        // (seed_individual), the objective function is NOT called again.
        // Re-evaluating here would spend one extra FE per replacement, so a
        // generation would cost N + (replacements) instead of N.
        for (int j : B) {
            const auto& lam_j = W_[j];
            double g_y, g_xj;
            if (constraint_mode == ConstraintMode::FEASIBILITY) {
                g_y  = tchebycheff_feasibility(f_y, lam_j, cv_y);
                g_xj = tchebycheff_feasibility(vault.objectives_of(j),
                                               lam_j, vault.get_cv(j));
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
            }
        }

        // Step 2.5: update EP (scratch is outside [0,N) — not overwritten by Step 2.4).
        update_ep(vault, scratch);
    }

public:
    MOEADCore() = default;

    void set_neighbourhood_size(int T)  { T_           = T; }
    void set_T                 (int T)  { T_           = T; }   // alias
    void set_feas_penalty      (double p){ feas_penalty_ = p; }
    void set_eta_crossover     (double e){ eta_c_        = e; }
    void set_eta_mutation      (double e){ eta_m_        = e; }
    void set_pc                (double p){ pc_           = p; }
    void set_pm                (double p){ pm_           = p; }
    void set_seed              (unsigned s){ rng_.seed(s); }

    // Access EP (archive) size after optimisation.
    std::size_t external_population_size(DataVault<Ind_t>& vault) const {
        return vault.archive_size();
    }

    // ── setup (Step 1) ─────────────────────────────────────────────────────
    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int m = vault.objs_n();

        // Step 1 (input): weight vectors + Step 1.2: neighbourhoods.
        generate_weight_vectors(n, m);
        build_neighbourhoods(n);

        // Step 1.3: random initialisation.
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

        // Step 1.4: z_j = min_i f_j(x^i)  (§V-E).
        ideal_.assign(m, std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i)
            update_ideal(vault.objectives_of(i));

        // Step 1.1: EP = ∅.
        vault.archive_clear();

        // Compute initial scalar fitness.
        for (int i = 0; i < n; ++i) {
            double cv = (constraint_mode != ConstraintMode::NONE)
                        ? vault.get_cv(i) : 0.0;
            vault.get_ind(i).scalar_fitness =
                (constraint_mode == ConstraintMode::FEASIBILITY)
                ? tchebycheff_feasibility(vault.objectives_of(i), W_[i], cv)
                : tchebycheff(vault.objectives_of(i), W_[i]);
        }

        // Expand vault by one slot for the scratch offspring.
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
        for (int i = 0; i < n; ++i) update_ep(vault, i);

        for (int i = 0; i < n; ++i) {
            double cv = (constraint_mode != ConstraintMode::NONE)
                        ? vault.get_cv(i) : 0.0;
            vault.get_ind(i).scalar_fitness =
                (constraint_mode == ConstraintMode::FEASIBILITY)
                ? tchebycheff_feasibility(vault.objectives_of(i), W_[i], cv)
                : tchebycheff(vault.objectives_of(i), W_[i]);
        }

        vault.expand(1);
        vault.reduce(n + 1);
    }

    // ── step: one generation = Step 2 for i = 1..N SEQUENTIALLY ───────────
    // ("For i = 1, ..., N, do" — fix G4; previously the order was randomised,
    //  which is a MOEA/D-DRA technique, not part of the 2007 paper).
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        for (int i = 0; i < n; ++i)
            update_subproblem(vault, i);
    }
};

} // namespace mootation
