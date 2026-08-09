#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// crEA — A clustering-ranking method for many-objective optimization
// Cai, Qu, Yuan, Yao — Applied Soft Computing 35 (2015) 681–694
// doi:10.1016/j.asoc.2015.06.020
//
// Generational scheme (Algorithm 1):
//   1. Recombination + Mutation: random parent pairs (the paper specifies no
//      mating selection), SBX (p_c=0.9) + PM (p_m=1/n) -> N offspring;
//      Rt = P u Q (2N).
//   2. UpdateIdealPoint: z* is the cumulative minimum; AdaptiveNormalize
//      (Eq.4): f~_j = (f_j − z*_j)/(z^max_j − z*_j), with z^max the
//      component-wise max over the current Rt.
//   3. Clustering (Alg.4, Eq.5-6): each member of Rt goes to the nearest
//      Das-Dennis reference LINE (Eq.2-3) by perpendicular distance
//      d = ||f~ − lambda*(f~*lambda)/||lambda||^2||; Nf is the size of the
//      largest cluster.
//   4. Ranking (Alg.5, Eq.7): FT(x) = max_k f~_k(x)/lambda_k, using the lambda
//      of the point's own cluster; within a cluster, sort by increasing FT;
//      layer F_l is the l-th member of every cluster, l = 1..Nf.
//   5. Selection (Alg.6): take layers F_1, F_2, ... whole until N; the
//      incomplete last layer is shuffled and filled at random up to
//      |P_{t+1}| = N.
//
// PAPER DEFAULTS (Table 1 / §4.4): p_c=0.9, eta_c=30, p_m=1/n, eta_m=20; the
//   number of reference points equals N.
// DECLARED DEVIATIONS AND READINGS:
//   (a) CLAMPING of lambda_k (Eq.7): on boundary reference points lambda_k = 0
//       and the division blows up; the denominator is clamped to 1e-6, the
//       standard huge-penalty form. This is an interpretation — the paper does
//       not address the case.
//   (b) TIE-BREAKS: on equal perpendicular distances a solution goes to the
//       LOWEST-indexed line (a strict "<"); on equal FT within a cluster the
//       order is that of std::sort, which is unstable; in the last layer the
//       choice is random (a shuffle, per Alg.6). The paper specifies no
//       tie-breaks.
//   (c) The reference-line count of N is produced by das_dennis::generate_auto.
//       When N is exactly attainable by a lattice — including the "paper" N of
//       Table 2 — this is exactly N single-layer vectors; otherwise
//       generate_auto picks the nearest configuration (single- or two-layer
//       Deb-Jain, beyond the paper) and returns approximately N vectors. It
//       does not throw, so an arbitrary pop_size degrades gracefully.
//   (d) The pseudocode of Alg.1 contradicts the text of §3.1, "normalize all 2N
//       individuals"; the code follows the text, so the pool is Rt = 2N.
//   (e) crEA has no dominance step, so constraint_mode FEASIBILITY/CDP acts on
//       its one preference relation instead: the within-cluster ranking by FT
//       becomes feasibility-first (feasible, then ascending CV, then ascending
//       FT). Because layer F_l takes the l-th member of every cluster and the
//       layers are consumed in order, this places every feasible member of a
//       cluster ahead of its infeasible ones. The paper is unconstrained, so
//       this is an extension.
// EXTENSIONS BEYOND THE PAPER: binary variables (off by default).
// ============================================================================
// Notable fixes: (1) CREA-1: p_c=0.9 (Table 1: "p_c=0.9, eta_c=30"); SBX used
// to be applied unconditionally, i.e. effectively p_c=1.0; (2) the RNG seed is
// std::random_device plus set_seed, replacing time(nullptr); (3) explicit
// p_c/p_m in the current SBX/PM signatures; (4) the DOI was corrected (it read
// ...06.024, the correct value is ...06.020); (5) the lambda_k clamp and the
// tie-breaks are now documented (CREA-2).
// CR-1: init_refs moved from das_dennis::generate_exact to generate_auto.
// generate_exact threw on any non-lattice N, so setup failed at an arbitrary
// pop_size, while note (c) in this header promised a smooth two-layer fallback
// — the header was lying. generate_auto does not throw (as in lis_lcs and
// rd_emo), and note (c) now matches the behaviour.

#include <algorithm>
#include <cmath>
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

// ── crEA individual ─────────────────────────────────────────────────────────
// rank is the layer index after ranking; cluster is the associated reference
// line. Both are stored for readability; the selection uses local arrays.
struct crEA_Individual : public Based_Individual {
    int rank    = 0;
    int cluster = 0;
};

template <typename Ind_t>
class crEACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double       eta_c_ = 30.0;     // Table 1: η_c = 30
    double       eta_m_ = 20.0;     // Table 1: η_m = 20
    double       pc_    = 0.9;      // Table 1 / §4.4: p_c = 0.9 (FIX, CREA-1)
    std::mt19937 rng_{std::random_device{}()};

    std::vector<std::vector<double>> W_;   // reference lines Lambda (Das-Dennis)
    std::vector<double>              z_;   // ideal point z*

    void init_refs(int m, int n) {
        // FIX 2026-07-08: CR-1 —
        // This used to be das_dennis::generate_exact(m,n), which THROWS
        // std::invalid_argument for any n that is not exactly a Das-Dennis
        // lattice size, so setup failed at an arbitrary pop_size.
        // generate_auto (as in lis_lcs and rd_emo) does NOT throw: it picks the
        // nearest configuration (single- or two-layer) and returns about n
        // vectors. The reference-line count Nc = W_.size() may therefore differ
        // from n by a few.
        W_ = das_dennis::generate_auto(m, n);
    }

    void update_ideal(DataVault<Ind_t>& vault, int pool) {
        int m = vault.objs_n();
        if (static_cast<int>(z_.size()) != m)
            z_.assign(m, std::numeric_limits<double>::max());
        for (int i = 0; i < pool; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) z_[j] = std::min(z_[j], o[j]);
        }
    }

    // ── Adaptive normalization (Eq.4) over the pool [0, pool) ──────────────
    std::vector<std::vector<double>>
    normalize(DataVault<Ind_t>& vault, int pool) {
        int m = vault.objs_n();
        std::vector<double> zmax(m, -std::numeric_limits<double>::max());
        for (int i = 0; i < pool; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) zmax[j] = std::max(zmax[j], o[j]);
        }
        std::vector<std::vector<double>> F(pool, std::vector<double>(m));
        for (int i = 0; i < pool; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) {
                double range = zmax[j] - z_[j];
                F[i][j] = (range > 1e-14) ? (o[j] - z_[j]) / range : 0.0;
            }
        }
        return F;
    }

    // ── Perpendicular distance from f~ to the reference line w (Eq.6) ──────
    static double perp_dist(const std::vector<double>& f,
                            const std::vector<double>& w) {
        double dot = 0.0, ww = 0.0;
        for (std::size_t j = 0; j < f.size(); ++j) { dot += f[j] * w[j]; ww += w[j] * w[j]; }
        ww = std::max(ww, 1e-30);
        double t = dot / ww;
        double s = 0.0;
        for (std::size_t j = 0; j < f.size(); ++j) {
            double d = f[j] - t * w[j];
            s += d * d;
        }
        return std::sqrt(s);
    }

    // ── Fitness FT(x) (Eq.7): max_k f̃_k/λ_k ───────────────────────────────
    static double ft_value(const std::vector<double>& f,
                           const std::vector<double>& w) {
        double best = -std::numeric_limits<double>::max();
        for (std::size_t k = 0; k < f.size(); ++k) {
            double wk = (w[k] > 1e-6) ? w[k] : 1e-6;   // avoid division by zero
            best = std::max(best, f[k] / wk);
        }
        return best;
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
    crEACore() = default;

    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_pc           (double p) { pc_   = p; }
    void set_seed(unsigned s)        { rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size(), m = vault.objs_n();
        init_refs(m, n);
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
        z_.assign(m, std::numeric_limits<double>::max());
        update_ideal(vault, n);                       // InitializeIdealPoint(P0)
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int n = vault.pop_size(), m = vault.objs_n();
        if (W_.empty()) init_refs(m, n);
        z_.assign(m, std::numeric_limits<double>::max());
        update_ideal(vault, n);
    }

    // ── step: one generation (Algorithm 1) ─────────────────────────────────
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int N = n;
        int Nc = static_cast<int>(W_.size());
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_int(0, n - 1);

        // (6-7) Recombination + Mutation -> offspring in [off_base, +n).
        int off_base = vault.expand(n);
        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        // Table 1 / §4.4: p_c = 0.9, p_m = 1/n (FIX 2026-06, CREA-1).
        double pm = (vault.vars_n() > 0) ? 1.0 / vault.vars_n() : 0.0;
        for (int i = 0; i < n; i += 2) {
            int p1 = dist_int(rng_), p2 = dist_int(rng_);
            for (int j = 0; j < vault.vars_n(); ++j) {
                pv1[j] = vault.get_variable(p1, j);
                pv2[j] = vault.get_variable(p2, j);
            }
            ops::sbx(pv1, pv2, c1, c2, bounds, eta_c_, pc_, rng_);
            ops::polynomial_mutation(c1, bounds, eta_m_, pm, rng_);
            ops::polynomial_mutation(c2, bounds, eta_m_, pm, rng_);
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

        // (8-10) Rt = 2N; update the ideal point; normalize.
        int pool = n * 2;
        update_ideal(vault, pool);                    // UpdateIdealPoint
        auto F = normalize(vault, pool);              // Eq.4

        // (11) Clustering (Eq.6): each solution to its nearest reference line.
        std::vector<std::vector<int>> clusters(Nc);
        for (int i = 0; i < pool; ++i) {
            int    best_j = 0;
            double best_d = std::numeric_limits<double>::max();
            for (int j = 0; j < Nc; ++j) {
                double d = perp_dist(F[i], W_[j]);
                if (d < best_d) { best_d = d; best_j = j; }
            }
            vault.get_ind(i).cluster = best_j;
            clusters[best_j].push_back(i);
        }

        // (12) Ranking (Eq.7): sort each cluster by increasing FT.
        int Nf = 0;
        for (int j = 0; j < Nc; ++j) {
            auto& cl = clusters[j];
            if (cl.empty()) continue;
            // Under FEASIBILITY/CDP the within-cluster order is
            // feasibility-first (crEA-C): feasible before infeasible, then
            // ascending CV, then ascending FT. Since the layers are formed by
            // taking the l-th member of every cluster, a feasible solution
            // therefore always lands in an earlier layer than an infeasible
            // one from the same cluster, and earlier layers are taken first.
            const bool cm = (constraint_mode != ConstraintMode::NONE);
            std::sort(cl.begin(), cl.end(), [&](int a, int b) {
                if (cm) {
                    double ca = vault.get_cv(a), cb = vault.get_cv(b);
                    bool fa = (ca <= 0.0), fb = (cb <= 0.0);
                    if (fa != fb) return fa;
                    if (!fa && ca != cb) return ca < cb;
                }
                return ft_value(F[a], W_[j]) < ft_value(F[b], W_[j]);
            });
            Nf = std::max(Nf, static_cast<int>(cl.size()));
        }
        // Layers: F_l is the l-th member of every cluster (l = 0..Nf-1).
        std::vector<std::vector<int>> layers(Nf);
        for (int j = 0; j < Nc; ++j) {
            const auto& cl = clusters[j];
            for (int l = 0; l < static_cast<int>(cl.size()); ++l) {
                vault.get_ind(cl[l]).rank = l;
                layers[l].push_back(cl[l]);
            }
        }

        // (13) Selection: layer by layer up to N; the last layer at random.
        std::vector<int> survivors;
        survivors.reserve(N);
        for (int l = 0; l < Nf && static_cast<int>(survivors.size()) < N; ++l) {
            int room = N - static_cast<int>(survivors.size());
            if (static_cast<int>(layers[l].size()) <= room) {
                for (int v : layers[l]) survivors.push_back(v);
            } else {
                std::shuffle(layers[l].begin(), layers[l].end(), rng_);
                for (int k = 0; k < room; ++k) survivors.push_back(layers[l][k]);
            }
        }

        rearrange(vault, survivors, pool);
    }
};

} // namespace mootation
