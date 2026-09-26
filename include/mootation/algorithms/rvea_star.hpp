#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// RVEA* — RVEA with the reference vector regeneration strategy for irregular
//         Pareto fronts
// R. Cheng, Y. Jin, M. Olhofer, B. Sendhoff — IEEE Trans. Evol. Comput. 20(5), 2016,
// §VI, Algorithm 4
// doi:10.1109/TEVC.2016.2519378          (source: rvea_cheng2016)
//
// The paper's own variant of RVEA (rvea.hpp) for degenerate, disconnected and
// inverted fronts, where part of a uniform vector set points past the front and
// RVEA's population shrinks with it. §VI keeps the uniform set V and adds a
// second set V* of N vectors; every generation, each vector of V* that is the
// nearest one to no non-dominated solution is moved to a random direction
// inside the population's range. What is not listed below is RVEA as in
// rvea.hpp.
//
// Generation scheme (Algorithm 1 with Algorithm 4 inserted "between Line 10 and
// Line 11", and "V_t ∪ V*_t (instead V_t)" guiding Algorithm 2 — §VI):
//   1. Offspring (§III-B): ⌊N/2⌋ random pairs WITHOUT mating selection; SBX +
//      polynomial mutation (+ uniform crossover/bit-flip for binary variables).
//   2. Pool R = P_t ∪ Q_t; objective translation f' = f − z_min over the pool (Eq. 5).
//   3. Partitioning (Eq. 6–7) over the 2N vectors of V_t ∪ V*_t; APD (Eq. 8–10),
//      d = (1 + M·(t/t_max)^α·θ/γ_v)·‖f'‖, γ_v the smallest angle from v to the
//      other 2N − 1 vectors; one min-APD elitist per non-empty subpopulation;
//      N of them survive (RVEA*-1).
//   4. Adaptation of V (Alg. 3, Eq. 11) when (t/t_max mod fr) == 0, including
//      t = 0, as in RVEA. V* is not adapted: its vectors are drawn inside the
//      population's range already.
//   5. Regeneration of V* (Alg. 4) on P_{t+1}: the dominated solutions are
//      removed (line 3); translation and partitioning over V*, the algorithm's
//      only vector set (lines 5–7); every v*_i with an empty subpopulation
//      becomes u/‖u‖, u_j uniform in [0, z^max_j] of the translated values —
//      "the minimum value of each objective is always 0" after the translation
//      (footnote 6) — and the others stay (lines 9–19).
//
// Defaults = RVEA's, §IV-C: α=2, fr=0.1; η_c=30, p_c=1.0, η_m=20, p_m=1/n.
// DECLARED READINGS:
//   RVEA*-1 POPULATION N. With one elitist per non-empty vector of V ∪ V* the
//     paper's population can grow to 2N (Table VII gives RVEA a second, random
//     set of vectors too, "for fair comparisons"). Here every generation keeps
//     N: the pool is ordered by (place in its own subpopulation, V before V*,
//     APD) and the first N survive. When N or more vectors are non-empty, the
//     survivors are all the elitists of V and the elitists of V* with the
//     smallest APD; when fewer are, the runners-up of the subpopulations fill
//     the rest, V first. V goes first because §VI keeps the uniform set, as
//     NSGA-III's adaptation does, "to guarantee a wide spread", and introduces
//     V* "to perform exploration".
//   RVEA*-2 Alg. 4 line 12 runs "for j = 1 to N", but u_r has one component per
//     objective (line 13, "within [0, z^max_{t,j}]"), so j runs to M.
//   RVEA*-3 γ_v (Eq. 10, the smallest angle to "the other reference vectors in
//     the current generation") is taken over V ∪ V*, the set that guides the
//     selection.
//   RVEA*-4 The first V* is "randomly initialized" (§VI, of the second set given
//     to RVEA): u_j uniform in [0, 1), normalised.
//   RVEA*-5 When every z^max_j is 0 (all non-dominated solutions coincide) u is
//     0 and cannot be normalised; the empty v*_i then stays as it is.
//   A point at the ideal point has no direction; as in rvea.hpp it is assigned
//   to the first vector with angle 0.
// Extensions beyond the paper: binary variables (uniform crossover + bit-flip),
//   active only when bin_vars_n()>0, as in rvea.hpp. ConstraintMode::FEASIBILITY
//   is Algorithm 5 (C-RVEA, §VII) inside every subpopulation, as in rvea.hpp:
//   all-infeasible -> minimum CV, otherwise feasible-only -> minimum APD. For
//   RVEA*-1 a feasible member goes before an infeasible one at the same place,
//   and infeasible members are ordered by CV. Off by default (NONE).
// What to expect, measured 2026-09-27 against rvea (pop x 500 generations,
//   median IGD over seeds 1-5): the population stays N where RVEA's shrinks —
//   55-57 of 91 on DTLZ5 (M = 3), 12-18 of 126 on IDTLZ1 (M = 5) — and IGD
//   falls from 0.070 to 0.018 on DTLZ5, 0.145 to 0.068 on IDTLZ1 (M = 5),
//   0.104 to 0.090 and 0.675 to 0.398 on DTLZ7 at M = 3 and 6, where the
//   hypervolume is slightly lower (0.385 against 0.401 at M = 3, 0.243
//   against 0.246 at M = 6); on DTLZ2 (M = 3) the two agree (0.0545). The
//   price is speed: 2N vectors split a pool of 2N, so most subpopulations hold
//   one candidate and the pull towards the front is weaker than RVEA's — ZDT1
//   0.031 against 0.0095, DTLZ6 (M = 3) 0.77 against 0.43 at that budget; at
//   four times the budget ZDT1 is 0.0064 against 0.0049 and on DTLZ6 RVEA* is
//   ahead, 0.094 against 0.136.
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
#include "../detail/constrained.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

template <typename Ind_t>
class RVEAStarCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    // ── hyperparameters (RVEA's, §IV-C) ────────────────────────────────────
    double       alpha_  = 2.0;   // APD penalty growth exponent
    double       fr_     = 0.1;   // frequency of the adaptation of V
    int          t_max_  = 1000;  // total generations (caller should set this)
    double       eta_c_  = 30.0;  // SBX distribution index (§IV-C-1: η_c = 30)
    double       eta_m_  = 20.0;  // polynomial mutation index (§IV-C-1: η_m = 20)
    double       pc_     = 1.0;   // crossover probability (§IV-C-1: p_c = 1.0)
    double       pm_     = -1.0;  // mutation probability; <0 → 1/n (§IV-C-1)
    std::mt19937 rng_{std::random_device{}()};

    int current_gen_ = 0;         // generation counter (incremented in step())

    // Unit vectors: V0 the uniform set (Eq. 2–3), Vt the same set adapted to
    // the objective ranges (Alg. 3), Vstar the additional set V* (Alg. 4).
    std::vector<std::vector<double>> V0_;
    std::vector<std::vector<double>> Vt_;
    std::vector<std::vector<double>> Vstar_;

    // ── vectors ────────────────────────────────────────────────────────────
    static double norm_of(const std::vector<double>& v) {
        double s = 0.0;
        for (double x : v) s += x * x;
        return std::sqrt(s);
    }

    // The vector of `vecs` with the largest cosine to f' (Eq. 6–7), and the angle
    // between them. A point at the ideal point has no direction: the first
    // vector, angle 0, as in rvea.hpp.
    static int nearest(const std::vector<double>& fp, double fp_norm,
                       const std::vector<std::vector<double>>& vecs, double& angle) {
        angle = 0.0;
        if (fp_norm < 1e-14) return 0;
        int    best     = 0;
        double best_cos = -std::numeric_limits<double>::max();
        for (int r = 0; r < static_cast<int>(vecs.size()); ++r) {
            double dot = 0.0;
            for (std::size_t j = 0; j < fp.size(); ++j) dot += fp[j] * vecs[r][j];
            const double c = dot / fp_norm;   // the vectors are unit vectors
            if (c > best_cos) { best_cos = c; best = r; }
        }
        angle = std::acos(std::max(-1.0, std::min(1.0, best_cos)));
        return best;
    }

    // γ_v (Eq. 10): the smallest angle from each vector to the others.
    static std::vector<double> smallest_angles(const std::vector<std::vector<double>>& vecs) {
        const int n = static_cast<int>(vecs.size());
        std::vector<double> gamma(n, std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                double dot = 0.0;
                for (std::size_t k = 0; k < vecs[i].size(); ++k) dot += vecs[i][k] * vecs[j][k];
                gamma[i] = std::min(gamma[i], std::acos(std::max(-1.0, std::min(1.0, dot))));
            }
        return gamma;
    }

    // v = u/‖u‖ with u_j = U[0, 1)·scale_j (Alg. 4 line 13; RVEA*-4 with scale 1).
    // Returns false and leaves v as it is when u = 0 (RVEA*-5).
    bool random_direction(const std::vector<double>& scale, std::vector<double>& v) {
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::vector<double> u(scale.size());
        for (std::size_t j = 0; j < scale.size(); ++j) u[j] = U(rng_) * scale[j];
        const double n = norm_of(u);
        if (n <= 0.0) return false;
        for (std::size_t j = 0; j < u.size(); ++j) v[j] = u[j] / n;
        return true;
    }

    // V0 = Vt: the Das-Dennis lattice on the unit sphere (Eq. 2–3);
    // V*: N random unit vectors (RVEA*-4).
    void make_vectors(int n, int m) {
        V0_ = das_dennis::generate_exact(m, n);
        for (auto& v : V0_) {
            const double s = norm_of(v);   // > 0: a lattice point sums to 1
            for (double& x : v) x /= s;
        }
        Vt_ = V0_;
        const std::vector<double> ones(m, 1.0);
        Vstar_.assign(n, std::vector<double>(m, 0.0));
        for (auto& v : Vstar_)
            while (!random_direction(ones, v)) {}
    }

    // ── Reference vector adaptation (Algorithm 3, Eq. 11), as in rvea.hpp ──
    // v_{t+1,i} = normalise(V0_[i] ⊙ (z_max − z_min)) for all N vectors of V,
    // with z_min/z_max over the n survivors in P_{t+1}.
    void adapt_reference_vectors(DataVault<Ind_t>& vault, int n) {
        int m = vault.objs_n();
        std::vector<double> zmin(m,  std::numeric_limits<double>::max());
        std::vector<double> zmax(m, -std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) {
                zmin[j] = std::min(zmin[j], o[j]);
                zmax[j] = std::max(zmax[j], o[j]);
            }
        }
        std::vector<double> range(m);
        for (int j = 0; j < m; ++j) range[j] = std::max(zmax[j] - zmin[j], 1e-14);

        for (int i = 0; i < static_cast<int>(V0_.size()); ++i) {
            double norm = 0.0;
            for (int j = 0; j < m; ++j) norm += (V0_[i][j] * range[j]) * (V0_[i][j] * range[j]);
            norm = std::sqrt(norm);
            for (int j = 0; j < m; ++j)
                Vt_[i][j] = (norm > 1e-14) ? V0_[i][j] * range[j] / norm : V0_[i][j];
        }
    }

    // ── Selection: Algorithm 2 guided by V ∪ V* (§VI), N survivors ────────
    // pool: vault indices [0, pool_size). Writes the n_keep survivors (RVEA*-1).
    void select_survivors(DataVault<Ind_t>& vault, int pool_size, int n_keep,
                          std::vector<int>& survivors) {
        const int m   = vault.objs_n();
        const int n_v = static_cast<int>(Vt_.size());   // W[0, n_v) is V, the rest V*

        std::vector<std::vector<double>> W = Vt_;
        W.insert(W.end(), Vstar_.begin(), Vstar_.end());
        const std::vector<double> gamma = smallest_angles(W);   // RVEA*-3

        const double t_frac = static_cast<double>(current_gen_) /
                              static_cast<double>(std::max(t_max_, 1));

        // Translation (Eq. 5): z_min over the pool.
        std::vector<double> zmin(m, std::numeric_limits<double>::max());
        for (int i = 0; i < pool_size; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) zmin[j] = std::min(zmin[j], o[j]);
        }

        // Partition (Eq. 6–7) and APD (Eq. 8–9).
        std::vector<int>    vec(pool_size);   // index into W of each member's vector
        std::vector<double> apd(pool_size);
        std::vector<double> fp(m);
        for (int i = 0; i < pool_size; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) fp[j] = std::max(o[j] - zmin[j], 0.0);
            const double fp_norm = norm_of(fp);
            double theta = 0.0;
            vec[i] = nearest(fp, fp_norm, W, theta);
            const double gamma_v = (gamma[vec[i]] > 1e-14) ? gamma[vec[i]] : 1.0;
            const double P = static_cast<double>(m) * std::pow(t_frac, alpha_) * theta / gamma_v;
            apd[i] = (1.0 + P) * fp_norm;
            vault.get_ind(i).ref_vector_idx = vec[i];
            vault.get_ind(i).apd            = apd[i];
        }

        // Algorithm 5 in FEASIBILITY mode: an infeasible member is ordered by
        // its CV and after every feasible one.
        std::vector<int>    infeasible(pool_size, 0);
        std::vector<double> value = apd;
        if (constraint_mode == ConstraintMode::FEASIBILITY)
            for (int i = 0; i < pool_size; ++i)
                if (vault.get_cv(i) > 0.0) { infeasible[i] = 1; value[i] = vault.get_cv(i); }

        // Place of each member in its own subpopulation: 0 is the elitist that
        // Alg. 2 (Alg. 5) keeps, 1 the runner-up, and so on.
        std::vector<std::vector<int>> members(W.size());
        for (int i = 0; i < pool_size; ++i) members[vec[i]].push_back(i);
        std::vector<int> place(pool_size, 0);
        for (auto& sub : members) {
            std::sort(sub.begin(), sub.end(), [&](int a, int b) {
                if (infeasible[a] != infeasible[b]) return infeasible[a] < infeasible[b];
                if (value[a] != value[b]) return value[a] < value[b];
                return a < b;
            });
            for (int k = 0; k < static_cast<int>(sub.size()); ++k) place[sub[k]] = k;
        }

        // RVEA*-1: (place, feasible first, V before V*, APD or CV); first n_keep.
        std::vector<int> order(pool_size);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (place[a] != place[b]) return place[a] < place[b];
            if (infeasible[a] != infeasible[b]) return infeasible[a] < infeasible[b];
            const bool a_star = vec[a] >= n_v, b_star = vec[b] >= n_v;
            if (a_star != b_star) return b_star;
            if (value[a] != value[b]) return value[a] < value[b];
            return a < b;
        });
        survivors.assign(order.begin(), order.begin() + std::min(n_keep, pool_size));
    }

    // ── Reference vector regeneration (Algorithm 4) on P_{t+1} ────────────
    void regenerate_additional_vectors(DataVault<Ind_t>& vault, int n) {
        const int m = vault.objs_n();

        // Line 3: remove the dominated solutions.
        std::vector<int> nd;
        for (int i = 0; i < n; ++i) {
            bool dominated = false;
            for (int k = 0; k < n && !dominated; ++k)
                if (k != i && detail::pareto_dominates(vault.objectives_of(k),
                                                       vault.objectives_of(i)))
                    dominated = true;
            if (!dominated) nd.push_back(i);
        }

        // Lines 5 and 9: translation, then the maximal translated values.
        std::vector<double> zmin(m, std::numeric_limits<double>::max());
        for (int i : nd) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) zmin[j] = std::min(zmin[j], o[j]);
        }
        std::vector<double> zmax(m, 0.0);
        for (int i : nd) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) zmax[j] = std::max(zmax[j], o[j] - zmin[j]);
        }

        // Line 7: partition over V* alone.
        std::vector<int>    used(Vstar_.size(), 0);
        std::vector<double> fp(m);
        for (int i : nd) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) fp[j] = o[j] - zmin[j];
            double theta = 0.0;
            used[nearest(fp, norm_of(fp), Vstar_, theta)] = 1;
        }

        // Lines 10–19: an empty v*_i becomes u/‖u‖, u_j in [0, z^max_j], j over
        // the M objectives (RVEA*-2).
        for (std::size_t i = 0; i < Vstar_.size(); ++i)
            if (!used[i]) random_direction(zmax, Vstar_[i]);
    }

    // ── Rearrange vault ────────────────────────────────────────────────────
    void rearrange(DataVault<Ind_t>& vault,
                   const std::vector<int>& survivors, int pool_size) {
        int n = static_cast<int>(survivors.size());
        std::vector<int> pos(pool_size), at_pos(pool_size);
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

public:
    RVEAStarCore() = default;

    void set_alpha          (double a)  { alpha_  = a; }
    void set_fr             (double f)  { fr_     = f; }
    void set_t_max          (int t)     { t_max_  = t; }
    void set_eta_crossover  (double e)  { eta_c_  = e; }
    void set_eta_mutation   (double e)  { eta_m_  = e; }
    void set_pc             (double p)  { pc_     = p; }
    void set_pm             (double p)  { pm_     = p; }
    void set_seed           (unsigned s){ rng_.seed(s); }

    const std::vector<std::vector<double>>& reference_vectors() const { return Vt_; }
    const std::vector<std::vector<double>>& additional_reference_vectors() const { return Vstar_; }

    // ── setup ──────────────────────────────────────────────────────────────
    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int m = vault.objs_n();
        current_gen_ = 0;
        make_vectors(n, m);

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

    void setup_seeded(DataVault<Ind_t>& vault) {
        current_gen_ = 0;
        make_vectors(vault.pop_size(), vault.objs_n());
    }

    // ── step: one generation (Algorithm 1 with Algorithm 4) ────────────────
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int n_parents = vault.parents_n();   // n: every generation keeps N (RVEA*-1)

        const auto& bounds = vault.get_bounds();

        // ── expand: add n offspring slots ─────────────────────────────────
        vault.expand(n);

        // ── breed n offspring into [n_parents, n_parents+n) ───────────────
        std::vector<int> perm(n_parents);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng_);

        // §IV-C-1: p_c = 1.0, p_m = 1/n (n is the number of real-valued variables).
        double pm = (pm_ >= 0.0) ? pm_
                                 : 1.0 / static_cast<double>(std::max(1, vault.vars_n()));

        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        for (int i = 0; i < n; i += 2) {
            int p1 = perm[i % n_parents];
            int p2 = perm[(i + 1) % n_parents];
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
                vault.set_all_variables(n_parents + i,     c1, bc1);
                if (i + 1 < n) vault.set_all_variables(n_parents + i + 1, c2, bc2);
            } else {
                vault.set_variables(n_parents + i, c1);
                if (i + 1 < n) vault.set_variables(n_parents + i + 1, c2);
            }
        }

        // ── evaluate offspring ─────────────────────────────────────────────
        vault.sync();

        // ── selection guided by V ∪ V* (Algorithm 2, §VI; RVEA*-1) ────────
        int pool_size = n_parents + n;
        std::vector<int> survivors;
        select_survivors(vault, pool_size, n, survivors);
        rearrange(vault, survivors, pool_size);

        // ── adaptation of V (Algorithm 3; Alg. 1 line 10), as in rvea.hpp ─
        if (t_max_ > 0 && fr_ > 0.0) {
            int period = std::max(1, static_cast<int>(std::round(fr_ * t_max_)));
            if (current_gen_ % period == 0)
                adapt_reference_vectors(vault, static_cast<int>(vault.active_n()));
        }

        // ── regeneration of V* (Algorithm 4, between Lines 10 and 11) ──────
        regenerate_additional_vectors(vault, static_cast<int>(vault.active_n()));

        ++current_gen_;
    }
};

} // namespace mootation
