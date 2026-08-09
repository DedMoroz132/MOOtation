#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// RVEA — A Reference Vector Guided Evolutionary Algorithm for Many-Objective
//        Optimization
// R. Cheng, Y. Jin, M. Olhofer, B. Sendhoff — IEEE Trans. Evol. Comput. 20(5), 2016
// doi:10.1109/TEVC.2016.2519378          (source: cheng2016)
//
// Generation scheme (Algorithm 1):
//   1. Offspring (§III-B): ⌊N/2⌋ random pairs WITHOUT mating selection;
//      SBX + polynomial mutation (+ uniform crossover/bit-flip for binary variables).
//   2. Pool R = P_t ∪ Q_t; objective translation f' = f − z_min over the pool (Eq. 5).
//   3. Partitioning (Eq. 6–7): individual → reference vector with max cos θ;
//      vectors are a Das-Dennis lattice normalised onto the unit sphere (Eq. 2–3).
//   4. APD (Eq. 8–10): d = (1 + M·(t/t_max)^α·θ/γ_v)·‖f'‖, γ_v — the minimum
//      angle from v to the other vectors; from each non-empty subpopulation a
//      single min-APD individual survives (empty niches → |P_{t+1}| ≤ N, recovers later).
//   5. Vector adaptation (Alg. 3, Eq. 11): when (t/t_max mod fr) == 0 —
//      including t = 0 — v_i = normalise(v0_i ∘ (z_max − z_min)), z over P_{t+1}.
//
// Defaults = §IV-C: α=2, fr=0.1; η_c=30, p_c=1.0, η_m=20, p_m=1/n (§IV-C-1).
// DECLARED DEVIATIONS: none.
//   RVEA-1 (resolved): the reference-vector adaptation was skipped at t=0
//   because t was incremented before the divisibility check; the check now
//   runs first.
//   RVEA* (Alg. 4, vector regeneration for irregular PFs) is not implemented —
//   a deliberate implementation boundary (baseline RVEA).
// Extensions beyond the paper: binary variables (uniform crossover + bit-flip),
//   active only when bin_vars_n()>0. ConstraintMode::FEASIBILITY is
//   Algorithm 5 (C-RVEA, §VII) of the paper itself; off by default (NONE).
// What to expect from constraint handling here: Algorithm 5 re-orders only
//   WITHIN one reference vector's subspace (all-infeasible -> minimum CV,
//   otherwise feasible-only -> minimum APD), and elitism keeps exactly one
//   survivor per non-empty reference vector. It therefore cannot place a
//   feasible individual into a subspace that has none, and its effect on the
//   count of feasible solutions is small. Measured on the constraint suite's
//   problem (DTLZ2 m=3 n=7, x0 <= 0.5, pop 91, 60 generations) over 20 seeds:
//   mean feasible 63.4 -> 64.1 of 91, better on 13 seeds and worse on 6. The
//   direction is right; the magnitude is below the seed-to-seed spread, which
//   is why tests/test_constraints.cpp exempts rvea from its per-seed
//   feasibility comparison and says so.
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
class RVEACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    // ── hyperparameters ────────────────────────────────────────────────────
    double       alpha_  = 2.0;   // APD penalty growth exponent (paper default)
    double       fr_     = 0.1;   // reference vector adaptation frequency
    int          t_max_  = 1000;  // total generations (caller should set this)
    double       eta_c_  = 30.0;  // SBX distribution index (§IV-C-1: η_c = 30)
    double       eta_m_  = 20.0;  // polynomial mutation index (§IV-C-1: η_m = 20)
    double       pc_     = 1.0;   // crossover probability (§IV-C-1: p_c = 1.0)
    double       pm_     = -1.0;  // mutation probability; <0 → 1/n (§IV-C-1)
    std::mt19937 rng_{std::random_device{}()};

    int current_gen_ = 0;         // generation counter (incremented in step())

    // Reference vectors: V0 (original, fixed) and Vt (current, adapted).
    std::vector<std::vector<double>> V0_;   // unit reference vectors, shape [N][m]
    std::vector<std::vector<double>> Vt_;   // adapted reference vectors

    // Pre-computed smallest inter-vector angles γ_j for each reference vector.
    std::vector<double> gamma_;

    // ── Das-Dennis + unit-sphere reference vector generation ──────────────
    // Step 1: generate simplex-lattice points u_i as in MOEA/D.
    // Step 2: normalise to unit sphere: v_i = u_i / ||u_i||.
    void generate_reference_vectors(int n, int m) {
        V0_ = das_dennis::generate_exact(m, n);
        // Normalise each vector to unit sphere
        for (auto& v : V0_) {
            double norm = 0.0;
            for (double x : v) norm += x * x;
            norm = std::sqrt(std::max(norm, 1e-28));
            for (double& x : v) x /= norm;
        }
        Vt_ = V0_;
        // CRITICAL: gamma_ (the smallest inter-vector angles) MUST be
        // computed here. The APD computation in reference_vector_guided_selection
        // reads gamma_[r]; without this call gamma_ is empty → segfault on the
        // first step(). adapt_reference_vectors() calls precompute_gamma
        // only starting from its first activation (period fr*t_max).
        precompute_gamma();
    }


    // Compute angle between two unit vectors.
    double angle_between(const std::vector<double>& a,
                         const std::vector<double>& b) const {
        double dot = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) dot += a[j] * b[j];
        // Clamp to [-1,1] for numerical safety.
        dot = std::max(-1.0, std::min(1.0, dot));
        return std::acos(dot);
    }

    void precompute_gamma() {
        int n = static_cast<int>(Vt_.size());
        gamma_.assign(n, std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                double ang = angle_between(Vt_[i], Vt_[j]);
                gamma_[i] = std::min(gamma_[i], ang);
            }
    }

    // ── Reference vector adaptation (Algorithm 3, Eq. 11) ────────────────
    // Triggered when (t / t_max) mod fr == 0.
    // v_{t+1,i} = normalise( V0_[i] ⊙ (z_max - z_min) )
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

        for (int i = 0; i < n; ++i) {
            double norm = 0.0;
            for (int j = 0; j < m; ++j) norm += (V0_[i][j] * range[j]) * (V0_[i][j] * range[j]);
            norm = std::sqrt(norm);
            for (int j = 0; j < m; ++j)
                Vt_[i][j] = (norm > 1e-14) ? V0_[i][j] * range[j] / norm : V0_[i][j];
        }
        precompute_gamma();
    }

    // ── Reference-vector-guided selection (Algorithm 2) ───────────────────
    // pool: vault indices [0, pool_size). Selects at most N individuals
    // (one per non-empty subspace) and writes them to `survivors`.
    void reference_vector_guided_selection(DataVault<Ind_t>& vault,
                                           int pool_size,
                                           std::vector<int>& survivors) {
        int m   = vault.objs_n();
        int N   = static_cast<int>(Vt_.size());
        double t_frac = static_cast<double>(current_gen_) /
                        static_cast<double>(std::max(t_max_, 1));

        // Step 1: ideal point z_min from combined pool (translated objectives).
        std::vector<double> zmin(m, std::numeric_limits<double>::max());
        for (int i = 0; i < pool_size; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) zmin[j] = std::min(zmin[j], o[j]);
        }

        // Step 2: population partition — associate each individual with its
        // closest reference vector (maximum cosine = minimum angle).
        std::vector<int>    assoc(pool_size, -1);   // ref vector index for each ind
        std::vector<double> theta(pool_size, 0.0);  // angle to associated vector
        std::vector<double> fnorm(pool_size, 0.0);  // ||f'_i||

        for (int i = 0; i < pool_size; ++i) {
            const auto& o = vault.objectives_of(i);
            // Translated objective: f' = f - z_min.
            std::vector<double> fp(m);
            double fp_norm = 0.0;
            for (int j = 0; j < m; ++j) {
                fp[j] = std::max(o[j] - zmin[j], 0.0);   // clamp ≥ 0
                fp_norm += fp[j] * fp[j];
            }
            fp_norm = std::sqrt(fp_norm);
            fnorm[i] = fp_norm;

            if (fp_norm < 1e-14) {
                // Degenerate point at ideal — associate with ref 0, angle 0.
                assoc[i] = 0; theta[i] = 0.0; continue;
            }

            double best_cos = -std::numeric_limits<double>::max();
            int    best_ref = 0;
            for (int r = 0; r < N; ++r) {
                double dot = 0.0;
                for (int j = 0; j < m; ++j) dot += fp[j] * Vt_[r][j];
                double cos_val = dot / fp_norm;   // Vt_ are unit vectors
                if (cos_val > best_cos) { best_cos = cos_val; best_ref = r; }
            }
            assoc[i]  = best_ref;
            // Compute actual angle (acos of the dot product / norms).
            double clamped = std::max(-1.0, std::min(1.0,
                [&]() {
                    double dot = 0.0;
                    for (int j = 0; j < m; ++j) dot += fp[j] * Vt_[best_ref][j];
                    return dot / fp_norm;   // Vt_ already unit
                }()));
            theta[i] = std::acos(clamped);
        }

        // Collect constraint violations if needed.
        std::vector<double> cvs(pool_size, 0.0);
        if (constraint_mode == ConstraintMode::FEASIBILITY)
            for (int i = 0; i < pool_size; ++i) cvs[i] = vault.get_cv(i);

        // Step 3 & 4: for each subspace (reference vector), compute APD and
        // select the individual with the minimum APD.
        survivors.clear();
        survivors.reserve(N);

        for (int r = 0; r < N; ++r) {
            // Gather individuals in this subspace.
            std::vector<int> subpop;
            for (int i = 0; i < pool_size; ++i)
                if (assoc[i] == r) subpop.push_back(i);

            if (subpop.empty()) continue;   // no individual for this ref vector

            // Constraint handling (Algorithm 5).
            if (constraint_mode == ConstraintMode::FEASIBILITY) {
                // Check if all infeasible.
                bool all_infeas = true;
                for (int i : subpop)
                    if (cvs[i] <= 0.0) { all_infeas = false; break; }

                if (all_infeas) {
                    // Select minimum CV.
                    int best = *std::min_element(subpop.begin(), subpop.end(),
                        [&](int a, int b){ return cvs[a] < cvs[b]; });
                    vault.get_ind(best).ref_vector_idx = r;
                    vault.get_ind(best).apd = fnorm[best];
                    survivors.push_back(best);
                    continue;
                }
                // Keep only feasible candidates.
                std::vector<int> feas;
                for (int i : subpop) if (cvs[i] <= 0.0) feas.push_back(i);
                subpop = std::move(feas);
            }

            // APD = (1 + P(θ)) · ||f'||
            // P(θ) = M · (t/t_max)^α · θ / γ_r
            double gamma_r = (gamma_[r] > 1e-14) ? gamma_[r] : 1.0;
            int best = subpop[0];
            double best_apd = std::numeric_limits<double>::max();

            for (int i : subpop) {
                double P = static_cast<double>(m)
                         * std::pow(t_frac, alpha_)
                         * theta[i] / gamma_r;
                double apd_val = (1.0 + P) * fnorm[i];
                vault.get_ind(i).ref_vector_idx = r;
                vault.get_ind(i).apd = apd_val;
                if (apd_val < best_apd) { best_apd = apd_val; best = i; }
            }
            survivors.push_back(best);
        }
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
    RVEACore() = default;

    void set_alpha          (double a)  { alpha_  = a; }
    void set_fr             (double f)  { fr_     = f; }
    void set_t_max          (int t)     { t_max_  = t; }
    void set_eta_crossover  (double e)  { eta_c_  = e; }
    void set_eta_mutation   (double e)  { eta_m_  = e; }
    void set_pc             (double p)  { pc_     = p; }
    void set_pm             (double p)  { pm_     = p; }
    void set_seed           (unsigned s){ rng_.seed(s); }

    // Allow user to supply custom reference vectors (must be unit vectors).
    void set_reference_vectors(std::vector<std::vector<double>> v) {
        V0_ = std::move(v); Vt_ = V0_; precompute_gamma();
    }

    const std::vector<std::vector<double>>& reference_vectors() const { return Vt_; }

    // ── setup ──────────────────────────────────────────────────────────────
    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int m = vault.objs_n();
        current_gen_ = 0;

        if (V0_.empty()) generate_reference_vectors(n, m);

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
        int n = vault.pop_size();
        int m = vault.objs_n();
        current_gen_ = 0;
        if (V0_.empty()) generate_reference_vectors(n, m);
    }

    // ── step: one full generation (Algorithm 1) ────────────────────────────
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();

        // Actual parent count recorded by expand().
        int n_parents = vault.parents_n();

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

        // ── reference vector guided selection (Algorithm 2) ───────────────
        int pool_size = n_parents + n;
        std::vector<int> survivors;
        reference_vector_guided_selection(vault, pool_size, survivors);

        // ── move survivors to [0, |survivors|) ────────────────────────────
        rearrange(vault, survivors, pool_size);
        // vault.active_n() may now be < n if some subspaces were empty.
        // This is correct behaviour; population recovers next generation.

        // ── reference vector adaptation (Algorithm 3, line 3) ─────────────
        // The paper's condition: "if (t/t_max mod f_r) == 0" — true for
        // t ∈ {0, fr·t_max, 2·fr·t_max, ...} (§III-D: "the reference vector
        // will only be adapted at generation t = 0, t = 0.2×t_max, ...").
        // The divisibility check is performed BEFORE incrementing t, at the
        // end of iteration t, over P_{t+1} — as in Alg. 1 (line 10) / Alg. 3.
        if (t_max_ > 0 && fr_ > 0.0) {
            int period = std::max(1, static_cast<int>(std::round(fr_ * t_max_)));
            if (current_gen_ % period == 0) {
                int cur_n = static_cast<int>(vault.active_n());
                adapt_reference_vectors(vault, cur_n);
            }
        }

        ++current_gen_;
    }
};

} // namespace mootation
