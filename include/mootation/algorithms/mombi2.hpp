#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MOMBI-II — Improved Metaheuristic Based on the R2 Indicator for
//            Many-Objective Optimization
// R. Hernández Gómez, C. A. Coello Coello — GECCO 2015, pp. 679-686
// doi:10.1145/2739480.2754776
//
// Generational scheme (Alg. 3):
//   1. Parent selection (Line 6): binary tournament (§5.1). The paper does not
//      specify the criterion; here it is (R2 rank ascending, tie-break on the
//      smaller L2 norm, then random). rank and L2 come from the previous
//      generation's ranking; in the first generation every rank is 1, so L2 and
//      chance decide.
//   2. Variation (Line 7): SBX p_c = 1.0, η_c = 30; polynomial mutation
//      p_m = 1/n, η_m = 20 (§5.1, following the NSGA-III recommendation).
//   3. Evaluate the offspring; the L2 norm is taken on the RAW objectives
//      (Lines 3/9, before normalization).
//   4. Normalize P ∪ P' per Eq. 9 using the [z^min, z^max] of the PREVIOUS
//      update (Line 10); the division is protected by a numerical guard
//      (range >= 1e-14).
//   5. R2 ranking (Alg. 2): for each w, sort the pool by
//      (u_ASF(F_norm; 0, w), raw L2) ascending; rank = the minimum position;
//      guard w_j -> max(w_j, 1e-6), per footnote 4 of the paper.
//   6. Reduction (Line 12): the N best by (rank, L2).
//   7. Update Reference Points (Line 13, Alg. 1), computed on the already
//      reduced population P_{i+1} for the next generation:
//        z^min is a persistent minimum (Line 2); the z^nad of the current P per
//        Def. 9 (the max over the first non-dominated front) is kept in a
//        record of 5 generations;
//        if max_j var_j(record) > α, then ALL components z^max <- max_j z_j^nad
//        (Lines 5-6); otherwise, per i (Lines 8-18):
//        |z_i^max − z_i^min| < ε -> z_i^max <- max_j z_j^max (a snapshot taken
//        before any edits);
//        z_i^nad > z_i^max -> z_i^max <- 2 z_i^nad − z_i^max (reflection);
//        var_i = 0 and not recently marked -> z_i^max <- (z_i^max + max_rec)/2.
//        A mark lives as many generations as the record (= 5). z^max persists
//        across generations; initialization is Alg. 3 Line 4 (z^min <- z*,
//        z^max <- the z^nad of the starting population).
//
// PAPER DEFAULTS (§5.1): α = 0.5, ε = 1e-3 ("the parameters ε and α were set
// to 1e-3 and 0.5, respectively"), p_c = 1.0, η_c = 30, p_m = 1/n, η_m = 20;
// BEYOND THE PAPER — record = 5. The paper never gives the record LENGTH. §4.2
// describes it only as holding "the nadir vector of a few generations", and
// Alg.1 uses it through "obtain the vector of variances from record" and "the
// mark lasts the same number of generations that record is kept" — so the
// length sets both the variance window and the mark lifetime, and neither §5.1
// nor Table 1 pins it. The 5 used here is the value the wider literature
// settled on: Tian et al. 2018 (AR-MOEA, §IV-A.2) states it explicitly for its
// MOMBI-II baseline — "the threshold of variance α, the tolerance threshold ε
// and the record size of nadir vectors are set to 0.5, 0.001 and 5" — matching
// the paper on the two parameters the paper does give. Settable.
// Weights are SLD/Das-Dennis with |W| = |P| — except for
// m = 3, where §5.1 raises |P| to |W| + 1 = 92 "to fulfill the requirement of
// adopting even numbers in the binary tournament selection".
// DECLARED DEVIATIONS: the binary tournament criterion (the paper gives none;
// (rank, L2) was chosen); the max_j z_j^max snapshot for Line 10 is taken
// before the edits of the current call (the paper does not say);
// Path-A forces |P| = |W| exactly, so the paper's m=3 setting has to be run as
// 91 — 92 is not an attainable lattice size for m=3 (single-layer sizes are
// 3,6,10,...,91,105 and no two-layer pair sums to 92), and generate_exact
// throws on it; and where |P| is attainable only as a two-layer lattice, the
// weight set is the Deb-Jain two-layer construction rather than the
// single-layer SLD of Eq.10;
// u_ASF is evaluated WITHOUT the |·| of Eq.8 / footnote 3 ("We consider the
// absolute value from its original definition"). Because z^min is persistent
// and refreshed only at the end of the step (Alg.1 Line 13), the normalized
// F is negative for any offspring that improves the running ideal, so such a
// point is ranked BETTER instead of being penalised by the absolute value.
// Inserting std::abs would be the paper-exact form and would change selection.
// EXTENSIONS BEYOND THE PAPER (off by default): ConstraintMode::FEASIBILITY
// (CDP dominance when estimating the nadir, feasible-first in the tournament),
// binary variables.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../constraint_mode.hpp"
#include "../das_dennis.hpp"
#include "../data_vault.hpp"
#include "../individuals.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

struct MOMBI2_Individual : public Based_Individual {
    int    rank = 1;     // R2-rank of the last ranking (binary tournament)
    double l2   = 0.0;   // L2-norm of RAW objectives (Alg. 3 Lines 3/9)
};

template <typename Ind_t>
class MOMBI2Core {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    static constexpr double ALPHA_ = 0.5;      // variance threshold (§5.1)
    static constexpr double EPS_   = 1e-3;     // tolerance threshold (§5.1)
    static constexpr int    REC_   = 5;        // record length (§5.1)

    double       eta_c_ = 30.0;                // SBX index (§5.1)
    double       eta_m_ = 20.0;                // PM index (§5.1)
    double       pc_    = 1.0;                 // crossover rate (§5.1)
    std::mt19937 rng_{std::random_device{}()};

    std::vector<std::vector<double>> W_;       // weight vectors (SLD)
    std::vector<double>              zmin_;    // z^min — persistent ideal
    std::vector<double>              zmax_;    // z^max — persistent (Alg. 1)
    std::deque<std::vector<double>>  record_;  // nadir history (REC_ gens)
    std::vector<int>                 mark_;    // mark countdowns (Alg. 1)

    bool dominates(DataVault<Ind_t>& vault, int a, int b) const {
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            double ca = vault.get_cv(a), cb = vault.get_cv(b);
            bool af = (ca <= 0.0), bf = (cb <= 0.0);
            if (af && !bf) return true;
            if (!af && bf) return false;
            if (!af && !bf) return ca < cb;
        }
        const auto& fa = vault.objectives_of(a);
        const auto& fb = vault.objectives_of(b);
        bool strict = false;
        for (std::size_t k = 0; k < fa.size(); ++k) {
            if (fa[k] > fb[k]) return false;
            if (fa[k] < fb[k]) strict = true;
        }
        return strict;
    }

    // Nadir of the population slots [0, n) per Definition 9:
    // componentwise max over the first non-dominated front.
    std::vector<double> compute_nadir(DataVault<Ind_t>& vault, int n) {
        int m = vault.objs_n();
        std::vector<double> nad(m, -std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) {
            bool dom = false;
            for (int j = 0; j < n; ++j)
                if (i != j && dominates(vault, j, i)) { dom = true; break; }
            if (dom) continue;
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) nad[j] = std::max(nad[j], o[j]);
        }
        return nad;
    }

    // L2-norm of RAW objectives (Alg. 3 Lines 3/9 — before normalisation).
    void store_l2(DataVault<Ind_t>& vault, int from, int to) {
        for (int i = from; i < to; ++i) {
            const auto& o = vault.objectives_of(i);
            double s = 0.0;
            for (double f : o) s += f * f;
            vault.get_ind(i).l2 = std::sqrt(s);
        }
    }

    // ── Algorithm 1: Update Reference Points ──────────────────────────────
    // Called per Alg. 3 Line 13 on the already reduced population P_{i+1}
    // (slots [0, n)); the resulting z^min/z^max are used by the
    // normalisation (Line 10) of the NEXT generation.
    void update_reference_points(DataVault<Ind_t>& vault, int n) {
        int m = vault.objs_n();

        // Line 1: z* and z^nad of P.
        std::vector<double> zcur(m, std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) zcur[j] = std::min(zcur[j], o[j]);
        }
        std::vector<double> nad = compute_nadir(vault, n);

        // Line 2: persistent ideal.
        for (int j = 0; j < m; ++j) zmin_[j] = std::min(zmin_[j], zcur[j]);

        // Line 3: store z^nad in record (length REC_).
        record_.push_back(nad);
        while (static_cast<int>(record_.size()) > REC_) record_.pop_front();

        // Line 4: per-component variances of z^nad over the record.
        int R = static_cast<int>(record_.size());
        std::vector<double> v(m, 0.0);
        for (int j = 0; j < m; ++j) {
            double mean = 0.0;
            for (const auto& r : record_) mean += r[j];
            mean /= R;
            double var = 0.0;
            for (const auto& r : record_) { double d = r[j] - mean; var += d * d; }
            v[j] = var / R;
        }

        // Marks age one generation per update ("the mark lasts the same
        // number of generations that record is kept").
        for (int j = 0; j < m; ++j) if (mark_[j] > 0) --mark_[j];

        // Line 5: global variance test, α = 0.5.
        double vmax = *std::max_element(v.begin(), v.end());
        if (vmax > ALPHA_) {
            // Line 6: all components ← max component of the current nadir.
            double mx = *std::max_element(nad.begin(), nad.end());
            zmax_.assign(m, mx);
        } else {
            // Lines 8-18. Snapshot of max_j z_j^max taken before this call's
            // edits (the paper does not specify; documented choice).
            double zmax_old_max = *std::max_element(zmax_.begin(), zmax_.end());
            for (int j = 0; j < m; ++j) {
                if (std::abs(zmax_[j] - zmin_[j]) < EPS_) {        // Line 9
                    zmax_[j] = zmax_old_max;                       // Line 10
                    mark_[j] = REC_;                               // Line 11
                } else if (nad[j] > zmax_[j]) {                    // Line 12
                    zmax_[j] = 2.0 * nad[j] - zmax_[j];            // Line 13
                    mark_[j] = REC_;                               // Line 14
                } else if (v[j] <= 0.0 && mark_[j] == 0) {         // Line 15
                    double a = -std::numeric_limits<double>::max();
                    for (const auto& r : record_)                  // Line 16
                        a = std::max(a, r[j]);
                    zmax_[j] = (zmax_[j] + a) / 2.0;               // Line 17
                    mark_[j] = REC_;                               // Line 18
                }
            }
        }
    }

    // ── binary tournament (Alg. 3 Line 6, §5.1) ───────────────────────────
    // Criterion (documented choice): lower R2-rank, then lower L2, then
    // uniformly at random. FEASIBILITY (extension): feasible-first.
    int tournament(DataVault<Ind_t>& vault,
                   std::uniform_int_distribution<int>& dist) {
        int a = dist(rng_);
        int b = a;
        if (dist.b() > dist.a())
            do { b = dist(rng_); } while (b == a);
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            double cva = vault.get_cv(a), cvb = vault.get_cv(b);
            bool af = (cva <= 0.0), bf = (cvb <= 0.0);
            if ( af && !bf) return a;
            if (!af &&  bf) return b;
            if (!af && !bf) return (cva < cvb) ? a : b;
        }
        const auto& ia = vault.get_ind(a);
        const auto& ib = vault.get_ind(b);
        if (ia.rank != ib.rank) return (ia.rank < ib.rank) ? a : b;
        if (ia.l2   != ib.l2)   return (ia.l2   < ib.l2)   ? a : b;
        return (std::uniform_int_distribution<int>(0, 1)(rng_) == 0) ? a : b;
    }

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
            pos[want] = i; pos[other] = cur;
            at_pos[i] = want; at_pos[cur] = other;
        }
        vault.reduce(n);
    }

public:
    MOMBI2Core() = default;

    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_pc           (double p) { pc_ = p; }
    void set_seed         (unsigned s){ rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size(), m = vault.objs_n();
        W_ = das_dennis::generate_exact(m, n);
        record_.clear();
        mark_.assign(m, 0);
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

        // Alg. 3 Line 3: L2-norms of raw objectives; ranks start equal.
        store_l2(vault, 0, n);
        for (int i = 0; i < n; ++i) vault.get_ind(i).rank = 1;

        // Alg. 3 Line 4: z^min ← z*, z^max ← z^nad of the initial population.
        zmin_.assign(m, std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) zmin_[j] = std::min(zmin_[j], o[j]);
        }
        zmax_ = compute_nadir(vault, n);
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int n = vault.pop_size(), m = vault.objs_n();
        if (W_.empty()) W_ = das_dennis::generate_exact(m, n);
        record_.clear();
        mark_.assign(m, 0);
        store_l2(vault, 0, n);
        for (int i = 0; i < n; ++i) vault.get_ind(i).rank = 1;
        zmin_.assign(m, std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) zmin_[j] = std::min(zmin_[j], o[j]);
        }
        zmax_ = compute_nadir(vault, n);
    }

    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int N = n, m = vault.objs_n();
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_N(0, n - 1);

        // ── Lines 6-8: parent selection (binary tournament) + variation ────
        int off_base = vault.expand(n);
        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        for (int i = 0; i < n; i += 2) {
            int p1 = tournament(vault, dist_N);
            int p2 = tournament(vault, dist_N);
            for (int j = 0; j < vault.vars_n(); ++j) {
                pv1[j] = vault.get_variable(p1, j);
                pv2[j] = vault.get_variable(p2, j);
            }
            ops::sbx(pv1, pv2, c1, c2, bounds, eta_c_, pc_, rng_);
            ops::polynomial_mutation(c1, bounds, eta_m_, rng_);
            ops::polynomial_mutation(c2, bounds, eta_m_, rng_);
            if (vault.bin_vars_n() > 0) {
                std::vector<int> bv1(vault.bin_vars_n()), bv2(vault.bin_vars_n()), bc1, bc2;
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

        int pool = n * 2;

        // ── Line 9: L2-norm of RAW objectives for the offspring ───────────
        store_l2(vault, n, pool);

        // ── Line 10: normalise P ∪ P' (Eq. 9) with the reference points of
        //    the previous update (Line 13 of the previous generation) ───────
        std::vector<std::vector<double>> Fn(pool, std::vector<double>(m, 0.0));
        for (int i = 0; i < pool; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) {
                double range = zmax_[j] - zmin_[j];
                // numeric guard only; Alg. 1 Line 9 handles ε-collapse
                Fn[i][j] = (range > 1e-14) ? (o[j] - zmin_[j]) / range : 0.0;
            }
        }

        // ── Line 11: R2 ranking (Alg. 2), ASF with r = 0, tie-break L2 ─────
        std::vector<int> rank(pool, std::numeric_limits<int>::max());
        std::vector<int> order(pool);
        std::vector<double> mu(pool);
        for (const auto& w : W_) {
            for (int i = 0; i < pool; ++i) {
                double best = -std::numeric_limits<double>::max();
                for (int j = 0; j < m; ++j) {
                    double wj = (w[j] > 1e-6) ? w[j] : 1e-6;   // footnote 4
                    best = std::max(best, Fn[i][j] / wj);
                }
                mu[i] = best;
            }
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b){
                if (mu[a] != mu[b]) return mu[a] < mu[b];
                return vault.get_ind(a).l2 < vault.get_ind(b).l2;
            });
            for (int r = 0; r < pool; ++r) {
                int p = order[r];
                if (r + 1 < rank[p]) rank[p] = r + 1;
            }
        }

        // ── Line 12: reduce — N best by (rank, L2) ─────────────────────────
        std::vector<int> all(pool);
        std::iota(all.begin(), all.end(), 0);
        std::sort(all.begin(), all.end(), [&](int a, int b){
            if (rank[a] != rank[b]) return rank[a] < rank[b];
            return vault.get_ind(a).l2 < vault.get_ind(b).l2;
        });
        std::vector<int> survivors(all.begin(), all.begin() + N);

        // Persist ranks for the next generation's tournaments.
        for (int i = 0; i < pool; ++i) vault.get_ind(i).rank = rank[i];

        rearrange(vault, survivors, pool);

        // ── Line 13: update reference points on P_{i+1} (reduced) ──────────
        update_reference_points(vault, N);
    }
};

} // namespace mootation
