#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MOEA/D-AWA — MOEA/D with Adaptive Weight Vector Adjustment
// Y. Qi, X. Ma, F. Liu, L. Jiao, J. Sun, J. Wu —
//   Evolutionary Computation 22(2), 2014
// doi:10.1162/EVCO_a_00109          (source: qi2014)
//
// Generation scheme (Alg.4; the skeleton inherits MOEA/D-DRA, Zhang et al. 2009):
//   1. Step 2.1–2.2: every 50 generations Δ^i=(old−new)/old on g^tc(x^i|λ^i),
//      π^i ← 1 if Δ>0.001, otherwise (0.95+0.05·Δ/0.001)·π^i.
//   2. Step 2.3: I = m corner subproblems (weights ≈ permutations of
//      (1,0,…,0)) + (⌊N/5⌋−m) subproblems by a 10-tournament on π^i.
//   3. Step 3.1–3.3 (for i∈I): pool P = B(i) with probability δ=0.9,
//      otherwise {1..N}; r1=i, r2,r3 random from P; SBX + polynomial
//      mutation → y.
//   4. Step 3.4–3.5: z*_j = f_j(y)−10⁻⁷ on improvement; replacements: random
//      j from P, g^tc(y|λ^j)≤g^tc(x^j|λ^j) → x^j=y, at most n_r replacements.
//   5. Step 4.1 + §4.3: EP archive of non-dominated solutions, cap 1.5N,
//      truncation by vicinity distance (the most crowded one is removed).
//   6. Step 4 (gen ≥ rate_evol·G_max, gen mod wag = 0): AWA —
//      Alg.2: Step 1 reallocation (x^j ← argmin_i g^tc(x^i|λ^j)); iterative
//      removal of the nus subproblems with min SL (Eq.4, recomputed after
//      each removal); Alg.3: remove from EP the members dominated by the
//      population, iteratively add the nus sparsest EP solutions (SL
//      relative to the POPULATION), weight from the solution via Eq.5/6;
//      Step 4.4: rebuild B(i).
//
// Scalarisation Eq.2: g^tc = max_j λ_j·|f_j − z*_j|. Weights: WS
// initialisation (Alg.1): λ = WS(λ'), w_i=(1/λ'_i)/Σ_j(1/λ'_j). SL Eq.4
// (vicinity distance, Kukkonen&Deb 2006): SL = ∏_{i=1}^{m} of the L2
// distances to the m nearest neighbours.
//
// Defaults = §4.3: T=0.1N, δ=0.9, nus=0.05N, rate_evol=0.8, |EP|≤1.5N;
//   SBX/PM: η_c=η_m=20, p_c=1.0, p_m=1/n (Table 1). The paper sets wag
//   per problem (100–250, §4.3 for Adaptive-MOEA/D); the default here is 100
//   (set_wag). The paper does not set n_r — we inherit n_r=2 from MOEA/D-DRA.
// ASSUMPTIONS (each is a gap in the paper, not a departure from it):
//   (1) Step 3.2 reads "construct a solution ȳ from x^{r1}, x^{r2} and x^{r3}
//       by the SBX operator" — but SBX is a BINARY operator and the step names
//       three parents. This port crosses x^{r1} with x^{r2}; r3 is still drawn
//       (so the RNG stream matches a three-index reading) and then unused. The
//       paper offers no three-parent SBX and no rule for picking two of three.
//   (2) After a weight adjustment the utility π is reset to 1. The paper does
//       not say what happens to π for a subproblem whose weight just moved,
//       and a stale π would rank it by the performance of a different
//       subproblem.
//   (3) Corner subproblems = argmax_i λ^i_j per objective j. After the WS
//       transformation the exact permutations of (1,0,…,0) survive with an
//       ε-clamp, so the argmax picks them out.
// Deviations:
//   AWA-12 (MINOR). If the EP left after Alg.3 Step 1 supplies fewer than nus
//     candidates, the last-removed subproblems are restored so that
//     |evol_pop| stays N; Alg.3 defines no behaviour for that case.
//   (AWA-1…AWA-11 were closed by the internal audit.)
// Assumption made explicit: EP maintenance runs EVERY generation. Alg.4 nests
//   Step 4.1 inside the gate (gen >= rate_evol·G_max AND gen mod wag == 0), but
//   §3.4 defines EP as holding the non-dominated solutions visited DURING the
//   search, and §4.3 caps |EP| at 1.5N — a cap that could never bind under the
//   gated reading. The prose definition is followed.
// Extensions beyond the paper (disabled by default): ConstraintMode::FEASIBILITY —
//   an additive penalty·cv term added to g^tc; binary variables (uniform
//   crossover + bit-flip).
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

struct MOEADAWA_Individual : public Based_Individual {
    double scalar_fitness = 0.0;
};

template <typename Ind_t>
class MOEADAWACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    // ── hyperparameters (§4.3 + the DRA skeleton) ─────────────────────────
    int    T_            = -1;     // neighbourhood; -1 → auto 0.1N (§4.3)
    double delta_        = 0.9;    // probability of mating in B(i) (§4.3)
    int    nr_           = 2;      // Step 3.5 replacement limit (inherited from DRA)
    int    wag_          = 100;    // AWA period (paper: 100–250, per problem)
    double rate_evol_    = 0.8;    // fraction of pure MOEA/D generations (§4.3)
    double rate_update_  = 0.05;   // nus = rate_update_weight·N (§4.3)
    double ep_cap_rate_  = 1.5;    // |EP| ≤ 1.5N (§4.3)
    int    util_period_  = 50;     // π recomputation period (Step 2.1)
    double eta_c_        = 20.0;   // Table 1
    double eta_m_        = 20.0;   // Table 1
    double pc_           = 1.0;    // Table 1
    double feas_penalty_ = 1e6;    // FEASIBILITY extension
    int    t_max_        = 1000;   // G_max
    int    current_gen_  = 0;
    std::mt19937 rng_{std::random_device{}()};

    // ── runtime state ──────────────────────────────────────────────────────
    std::vector<double>              ideal_;    // z* (with the −1e-7 shift)
    std::vector<std::vector<double>> W_;        // weights [N][m]
    std::vector<std::vector<int>>    B_;        // neighbourhoods [N][T]
    std::vector<double>              utility_;  // π^i (Step 2)
    std::vector<double>              old_g_;    // snapshot of g^tc for Δ^i

    int T_eff(int n) const {
        int t = (T_ > 0) ? T_ : std::max(2, static_cast<int>(std::lround(0.1 * n)));
        return std::min(t, n);
    }

    // ── Alg.1: Das-Dennis + WS transformation ─────────────────────────────
    void generate_weight_vectors(int n, int m) {
        W_ = das_dennis::generate_exact(m, n);
        // WS(λ'): w_i = (1/λ'_i)/Σ_j(1/λ'_j); ε-clamp for zero components.
        for (auto& w : W_) {
            std::vector<double> t(m);
            double s = 0.0;
            for (int j = 0; j < m; ++j) { t[j] = 1.0 / std::max(w[j], 1e-6); s += t[j]; }
            for (int j = 0; j < m; ++j) w[j] = (s > 1e-30) ? t[j] / s : 1.0 / m;
        }
    }

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
        int T = T_eff(n);
        B_.assign(n, {});
        for (int i = 0; i < n; ++i) {
            std::vector<std::pair<double, int>> dists;
            dists.reserve(n);
            for (int j = 0; j < n; ++j)
                dists.emplace_back(weight_dist(W_[i], W_[j]), j);
            std::partial_sort(dists.begin(), dists.begin() + T, dists.end());
            B_[i].resize(T);
            for (int k = 0; k < T; ++k) B_[i][k] = dists[k].second;
        }
    }

    // ── Eq.2: g^tc = max λ_j |f_j − z*_j| ──────────────────────────────────
    double tchebycheff(const std::vector<double>& f,
                       const std::vector<double>& lambda) const {
        double g = 0.0;
        for (std::size_t j = 0; j < f.size(); ++j) {
            double val = lambda[j] * std::abs(f[j] - ideal_[j]);
            if (val > g) g = val;
        }
        return g;
    }

    double tchebycheff_cv(const std::vector<double>& f,
                          const std::vector<double>& lambda,
                          double cv) const {
        double g = tchebycheff(f, lambda);
        if (constraint_mode == ConstraintMode::FEASIBILITY && cv > 0.0)
            g += feas_penalty_ * cv;
        return g;
    }

    // ── Step 1.2 / 3.4: z*_j = min f_j − 10⁻⁷ ─────────────────────────────
    void update_ideal(const std::vector<double>& f) {
        for (std::size_t j = 0; j < f.size(); ++j)
            if (f[j] - 1e-7 < ideal_[j]) ideal_[j] = f[j] - 1e-7;
    }

    static double obj_dist(const std::vector<double>& a,
                           const std::vector<double>& b) {
        double s = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) { double t = a[j] - b[j]; s += t * t; }
        return std::sqrt(s);
    }

    // ── Eq.4: SL(p, pop) = ∏_{i=1}^{m} L2 to the m nearest neighbours in pop
    // self_in_pop: point p itself belongs to pop (skip the zero distance).
    static double sparsity_of(const std::vector<double>& p,
                              const std::vector<const std::vector<double>*>& pop,
                              bool self_in_pop) {
        int m = static_cast<int>(p.size());
        std::vector<double> d;
        d.reserve(pop.size());
        int skipped_self = 0;
        for (const auto* q : pop) {
            double dist = obj_dist(p, *q);
            if (self_in_pop && skipped_self == 0 && dist < 1e-30) { skipped_self = 1; continue; }
            d.push_back(dist);
        }
        if (d.empty()) return std::numeric_limits<double>::max();
        std::sort(d.begin(), d.end());
        int L = std::min<int>(m, static_cast<int>(d.size()));
        double prod = 1.0;
        for (int i = 0; i < L; ++i) prod *= d[i];
        return prod;
    }

    // ── Eq.5/6: λ^sp = WS(F − z*) with ε protection ────────────────────────
    std::vector<double> weight_from_solution(const std::vector<double>& f) const {
        int m = static_cast<int>(f.size());
        std::vector<double> w(m);
        double s = 0.0;
        for (int j = 0; j < m; ++j) {
            double r = 1.0 / std::max(f[j] - ideal_[j], 1e-6);   // ε from Eq.6
            w[j] = r; s += r;
        }
        for (int j = 0; j < m; ++j) w[j] = (s > 1e-30) ? w[j] / s : 1.0 / m;
        return w;
    }

    // ── EP (Alg.4 Step 4.1): non-dominated + cap 1.5N (vicinity truncation) ─
    void update_ep(DataVault<Ind_t>& vault, int scratch_slot, int n) {
        const auto& f_y = vault.objectives_of(scratch_slot);
        for (std::size_t k = 0; k < vault.archive_size(); ++k) {
            const auto& ep_f = vault.archive_objectives_of(k);
            bool ep_dom = true, any_better = false;
            for (std::size_t j = 0; j < f_y.size(); ++j) {
                if (ep_f[j] > f_y[j]) { ep_dom = false; break; }
                if (ep_f[j] < f_y[j]) any_better = true;
            }
            if (ep_dom && any_better) return;   // y is dominated — don't add
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
        // §4.3: cap 1.5N; truncation by vicinity distance (remove the most crowded).
        int cap = std::max(1, static_cast<int>(std::lround(ep_cap_rate_ * n)));
        while (static_cast<int>(vault.archive_size()) > cap) {
            int A = static_cast<int>(vault.archive_size());
            std::vector<const std::vector<double>*> pop(A);
            for (int i = 0; i < A; ++i) pop[i] = &vault.archive_objectives_of(i);
            int worst = 0; double wsl = std::numeric_limits<double>::max();
            for (int i = 0; i < A; ++i) {
                double sl = sparsity_of(*pop[i], pop, true);
                if (sl < wsl) { wsl = sl; worst = i; }
            }
            vault.archive_erase(static_cast<std::size_t>(worst));
        }
    }

    // ── Step 2.1–2.2: utility π ────────────────────────────────────────────
    void update_utility(DataVault<Ind_t>& vault, int n) {
        for (int i = 0; i < n; ++i) {
            double new_g = tchebycheff(vault.objectives_of(i), W_[i]);
            double old_g = old_g_[i];
            double delta = (old_g > 1e-30) ? (old_g - new_g) / old_g : 0.0;
            if (delta > 0.001) utility_[i] = 1.0;
            else utility_[i] = (0.95 + 0.05 * delta / 0.001) * utility_[i];
            old_g_[i] = new_g;
        }
    }

    // ── Step 2.3: I = corners + 10-tournament on π ─────────────────────────
    std::vector<int> select_subproblems(int n, int m) {
        std::vector<int> I;
        // m corner subproblems: argmax_i λ^i_j for each objective j.
        std::vector<char> is_corner(n, 0);
        for (int j = 0; j < m; ++j) {
            int best = 0;
            for (int i = 1; i < n; ++i) if (W_[i][j] > W_[best][j]) best = i;
            if (!is_corner[best]) { is_corner[best] = 1; I.push_back(best); }
        }
        int total = std::max(static_cast<int>(I.size()), n / 5);
        std::uniform_int_distribution<int> pick(0, n - 1);
        while (static_cast<int>(I.size()) < total) {
            int winner = pick(rng_);
            for (int k = 1; k < 10; ++k) {
                int cand = pick(rng_);
                if (utility_[cand] > utility_[winner]) winner = cand;
            }
            I.push_back(winner);
        }
        return I;
    }

    // ── Step 3: evolve one subproblem i ────────────────────────────────────
    void evolve_subproblem(DataVault<Ind_t>& vault, int i, int n) {
        // Step 3.1: mating pool.
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        bool local = (u01(rng_) < delta_);
        const std::vector<int>* poolB = &B_[i];
        std::vector<int> pool_all;
        if (!local) {
            pool_all.resize(n);
            std::iota(pool_all.begin(), pool_all.end(), 0);
        }
        const std::vector<int>& P = local ? *poolB : pool_all;

        // Step 3.2: r1 = i; r2, r3 from P. SBX(x^{r1}, x^{r2}) + PM.
        std::uniform_int_distribution<int> pickP(0, static_cast<int>(P.size()) - 1);
        int r2 = P[pickP(rng_)];
        int r3 = P[pickP(rng_)];
        for (int attempt = 0; attempt < 5 && r3 == r2; ++attempt) r3 = P[pickP(rng_)];
        (void)r3;   // SBX is binary; r3 of Step 3.2 is unused (see the file header)

        const auto& bounds = vault.get_bounds();
        int nv = vault.vars_n();
        std::vector<double> pv1(nv), pv2(nv), c1, c2;
        for (int j = 0; j < nv; ++j) {
            pv1[j] = vault.get_variable(i,  j);
            pv2[j] = vault.get_variable(r2, j);
        }
        ops::sbx(pv1, pv2, c1, c2, bounds, eta_c_, pc_, rng_);
        double pm = (nv > 0) ? 1.0 / nv : 0.0;
        ops::polynomial_mutation(c1, bounds, eta_m_, pm, rng_);
        // Step 3.3 (repair): the bounded operators yield y ∈ Ω by construction.

        std::vector<int> bc1;
        int nb = vault.bin_vars_n();
        if (nb > 0) {
            std::vector<int> bv1(nb), bv2(nb), bc2;
            for (int j = 0; j < nb; ++j) {
                bv1[j] = vault.get_bin_variable(i,  j);
                bv2[j] = vault.get_bin_variable(r2, j);
            }
            ops::binary_crossover(bv1, bv2, bc1, bc2, rng_);
            ops::bit_flip_mutation(bc1, nb, rng_);
        }

        // Evaluate y in the scratch slot (the only FE per offspring).
        int scratch = static_cast<int>(vault.active_n()) - 1;
        if (bc1.empty()) vault.set_variables(scratch, c1);
        else             vault.set_all_variables(scratch, c1, bc1);
        vault.refresh_objectives(scratch);

        std::vector<double> f_y  = vault.objectives_of(scratch);
        std::vector<double> lm_y = vault.limits_of(scratch);
        double cv_y = (constraint_mode != ConstraintMode::NONE)
                      ? vault.get_cv(scratch) : 0.0;

        // Step 3.4: z*.
        update_ideal(f_y);

        // Step 3.5: replacements — random j from P, at most n_r.
        std::vector<int> Pp = P;
        std::shuffle(Pp.begin(), Pp.end(), rng_);
        int c = 0;
        for (int j : Pp) {
            if (c >= nr_) break;
            double g_y  = tchebycheff_cv(f_y, W_[j], cv_y);
            double g_xj = tchebycheff_cv(vault.objectives_of(j), W_[j],
                                         (constraint_mode != ConstraintMode::NONE)
                                         ? vault.get_cv(j) : 0.0);
            if (g_y <= g_xj) {
                // X2: transfer vars+objs without re-evaluation (FE budget).
                vault.seed_individual(j, c1, f_y, bc1, lm_y);
                vault.get_ind(j).scalar_fitness = g_y;
                ++c;
            }
        }

        // Alg.4 Step 4.1: EP (run every generation — see the header note).
        update_ep(vault, scratch, n);
    }

    // ── AWA: Alg.2 + Alg.3 ────────────────────────────────────────────────
    struct Entry {
        std::vector<double> w, vars, objs, lims;
        std::vector<int>    bvars;
    };

    Entry gather_slot(DataVault<Ind_t>& vault, int i) {
        Entry e;
        e.w    = W_[i];
        e.objs = vault.objectives_of(i);
        e.lims = vault.limits_of(i);
        int nv = vault.vars_n(), nb = vault.bin_vars_n();
        e.vars.resize(nv);
        for (int j = 0; j < nv; ++j) e.vars[j] = vault.get_variable(i, j);
        e.bvars.resize(nb);
        for (int j = 0; j < nb; ++j) e.bvars[j] = vault.get_bin_variable(i, j);
        return e;
    }

    static bool dom_obj(const std::vector<double>& fa, const std::vector<double>& fb) {
        bool strict = false;
        for (std::size_t k = 0; k < fa.size(); ++k) {
            if (fa[k] > fb[k]) return false;
            if (fa[k] < fb[k]) strict = true;
        }
        return strict;
    }

    void awa(DataVault<Ind_t>& vault, int n) {
        int nus = std::max(1, static_cast<int>(std::lround(rate_update_ * n)));
        if (vault.archive_size() == 0) return;

        std::vector<Entry> pop;
        pop.reserve(n);
        for (int i = 0; i < n; ++i) pop.push_back(gather_slot(vault, i));

        // ── Alg.2 Step 1: reallocation of x^i over the subproblems ─────────
        // if g^tc(x^i|λ^j,z) < g^tc(x^j|λ^j,z) then x^j = x^i, FV^j = FV^i.
        for (int j = 0; j < n; ++j) {
            int best = -1;
            double gbest = tchebycheff(pop[j].objs, pop[j].w);
            for (int i = 0; i < n; ++i) {
                if (i == j) continue;
                double gi = tchebycheff(pop[i].objs, pop[j].w);
                if (gi < gbest) { gbest = gi; best = i; }
            }
            if (best >= 0) {
                pop[j].vars  = pop[best].vars;
                pop[j].objs  = pop[best].objs;
                pop[j].lims  = pop[best].lims;
                pop[j].bvars = pop[best].bvars;
            }
        }

        // ── Alg.2 Steps 2–3: iterative removal of the nus most crowded ─────
        std::vector<Entry> deleted;   // for the guard restoration (|EP| < nus)
        for (int t = 0; t < nus && static_cast<int>(pop.size()) > 1; ++t) {
            int sz = static_cast<int>(pop.size());
            std::vector<const std::vector<double>*> ptrs(sz);
            for (int i = 0; i < sz; ++i) ptrs[i] = &pop[i].objs;
            int worst = 0; double wsl = std::numeric_limits<double>::max();
            for (int i = 0; i < sz; ++i) {
                double sl = sparsity_of(pop[i].objs, ptrs, true);
                if (sl < wsl) { wsl = sl; worst = i; }
            }
            deleted.push_back(pop[worst]);
            pop.erase(pop.begin() + worst);
        }

        // ── Alg.3 Step 1: drop EP members dominated by the population ─────
        struct EpCand { std::vector<double> vars, objs, lims; std::vector<int> bvars; };
        std::vector<EpCand> ep;
        for (std::size_t k = 0; k < vault.archive_size(); ++k) {
            const auto& fo = vault.archive_objectives_of(k);
            bool dominated = false;
            for (const auto& e : pop)
                if (dom_obj(e.objs, fo)) { dominated = true; break; }
            if (dominated) continue;
            EpCand cand;
            cand.objs = fo;
            cand.vars = vault.archive_variables_of(k);
            cand.lims = vault.archive_limits_of(k);
            if (vault.bin_vars_n() > 0) cand.bvars = vault.archive_bin_variables_of(k);
            ep.push_back(std::move(cand));
        }

        // ── Alg.3 Steps 2–4: iteratively add the sparsest ones ────────────
        while (static_cast<int>(pop.size()) < n && !ep.empty()) {
            int sz = static_cast<int>(pop.size());
            std::vector<const std::vector<double>*> ptrs(sz);
            for (int i = 0; i < sz; ++i) ptrs[i] = &pop[i].objs;
            // Step 2: SL of the EP candidates relative to the POPULATION evol_pop′.
            int best = 0; double bsl = -1.0;
            for (std::size_t i = 0; i < ep.size(); ++i) {
                double sl = sparsity_of(ep[i].objs, ptrs, false);
                if (sl > bsl) { bsl = sl; best = static_cast<int>(i); }
            }
            // Step 3.1: the new subproblem's weight from the solution (Eq.5/6).
            Entry e;
            e.w     = weight_from_solution(ep[best].objs);
            e.vars  = std::move(ep[best].vars);
            e.objs  = std::move(ep[best].objs);
            e.lims  = std::move(ep[best].lims);
            e.bvars = std::move(ep[best].bvars);
            pop.push_back(std::move(e));
            ep.erase(ep.begin() + best);
        }
        // Guard beyond the paper: if EP is smaller than nus — return the last
        // removed ones so that the population size stays N.
        while (static_cast<int>(pop.size()) < n && !deleted.empty()) {
            pop.push_back(std::move(deleted.back()));
            deleted.pop_back();
        }

        // ── write back + Step 4.4 ──────────────────────────────────────────
        for (int i = 0; i < n; ++i) {
            W_[i] = pop[i].w;
            vault.seed_individual(i, pop[i].vars, pop[i].objs,
                                  pop[i].bvars, pop[i].lims);
        }
        build_neighbourhoods(n);
        for (int i = 0; i < n; ++i)
            vault.get_ind(i).scalar_fitness =
                tchebycheff(vault.objectives_of(i), W_[i]);
        // π of the new/renumbered subproblems: reset (assumption, see the file header).
        utility_.assign(n, 1.0);
        for (int i = 0; i < n; ++i) old_g_[i] = vault.get_ind(i).scalar_fitness;
    }

    void init_state(DataVault<Ind_t>& vault, int n, int m) {
        ideal_.assign(m, std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) update_ideal(vault.objectives_of(i));
        utility_.assign(n, 1.0);
        old_g_.resize(n);
        for (int i = 0; i < n; ++i) {
            double g = tchebycheff(vault.objectives_of(i), W_[i]);
            vault.get_ind(i).scalar_fitness = g;
            old_g_[i] = g;
        }
        current_gen_ = 0;
    }

public:
    MOEADAWACore() = default;

    void set_neighbourhood_size(int T)   { T_ = T; }
    void set_T                 (int T)   { T_ = T; }   // alias
    void set_delta             (double d){ delta_ = d; }
    void set_nr                (int nr)  { nr_ = std::max(1, nr); }
    void set_wag               (int w)   { wag_ = std::max(1, w); }
    void set_rate_evol         (double r){ rate_evol_ = r; }
    void set_feas_penalty      (double p){ feas_penalty_ = p; }
    void set_eta_crossover     (double e){ eta_c_ = e; }
    void set_eta_mutation      (double e){ eta_m_ = e; }
    void set_seed              (unsigned s){ rng_.seed(s); }
    void set_t_max             (int t)   { t_max_ = (t > 0) ? t : 1; }

    std::size_t external_population_size(DataVault<Ind_t>& vault) const {
        return vault.archive_size();
    }

    // ── setup ──────────────────────────────────────────────────────────────
    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int m = vault.objs_n();

        generate_weight_vectors(n, m);
        build_neighbourhoods(n);

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

        vault.archive_clear();            // EP = ∅ (Step 1)
        init_state(vault, n, m);

        vault.expand(1);                  // scratch slot for the offspring
        vault.reduce(n + 1);
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int m = vault.objs_n();
        if (W_.empty()) { generate_weight_vectors(n, m); build_neighbourhoods(n); }

        vault.archive_clear();
        init_state(vault, n, m);
        vault.expand(1);
        vault.reduce(n + 1);
        for (int i = 0; i < n; ++i) update_ep(vault, i, n);
    }

    // ── step: one generation (Alg.4, Steps 2–5) ────────────────────────────
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int m = vault.objs_n();

        // Step 2.1–2.2: utility every 50 generations.
        if (current_gen_ > 0 && current_gen_ % util_period_ == 0)
            update_utility(vault, n);

        // Step 2.3 + Step 3.
        auto I = select_subproblems(n, m);
        for (int i : I) evolve_subproblem(vault, i, n);

        // Step 4: AWA in the window gen ≥ rate_evol·G_max, every wag generations.
        if (current_gen_ >= static_cast<int>(rate_evol_ * t_max_) &&
            current_gen_ % wag_ == 0) {
            awa(vault, n);
        }

        ++current_gen_;   // Step 5
    }
};

} // namespace mootation
