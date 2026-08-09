#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// AdaW — Weight Adaptation for Decomposition-Based EMO (Any Pareto Front Shape)
// M. Li, X. Yao — Evolutionary Computation 28(2), 2020, 227-253
// doi:10.1162/evco_a_00269
//
// Generational scheme (Alg.1; the MOEA/D framework of Li & Zhang 2009, §3.7):
//   1. For each subproblem i (Steps 8–17): the mating pool is B(i) with
//      probability delta=0.9, otherwise the whole population; the parents are
//      x^i and a random member of the pool; SBX(p_c=1) + polynomial mutation
//      (p_m=1/d, eta=20) -> offspring p.
//   2. z* = (best value − 1e-4) per objective (footnote 2), refreshed by the
//      offspring and used in the scalarization.
//   3. Population update: random j from the pool, replaced when
//      g(p|w_j) <= g(x^j|w_j), at most n_r = 0.01N replacements.
//   4. Archive (Steps 13–16): insert p if it is not dominated; remove the
//      members p dominates.
//   5. Archive maintenance EVERY generation while |A| > N_A = 2N
//      (Steps 18–20): iteratively remove argmax D(p) = 1 − Prod R(p,q)
//      (Eq.1–2, §3.2); the objectives are normalized by the set's min/max;
//      r is the median distance to the k-th nearest neighbour, k = m.
//   6. Weight update every 5% of the generations, never in the last 10%
//      (Steps 21–25): addition §3.3–3.4 (undeveloped: a niche of radius r_a
//      containing no population solution, with r_a the median NN distance of
//      the archive; promising: a candidate q beats ALL T neighbours on ITS OWN
//      weight w_q, Eq.3, with the Eq.4 tie-break on Sum f; the weight comes
//      from the solution via Eq.5–6; once admitted, q updates its neighbours
//      with no cap on the number of replacements) plus deletion §3.5 (the
//      solution shared by the most weights: remove argmax g(p,w_i), Eq.7; on a
//      tie, the one whose worst weight is worse; if every solution is unique,
//      fall back to the crowding of §3.2); then rebuild the neighbourhoods.
//
// Scalarization (§4): Tchebycheff DIVIDED by the weight,
// g = max_j (f_j − z*_j)/w_j; the optimal weight of a solution (Eq.5–6) is
// w = (f − z*)/Sum_i(f_i − z*_i).
//
// PAPER DEFAULTS — AdaW's own (§4, "Several specific parameters are required
//   in the proposed AdaW"): N_A = 2N, wag = 5%·Gen_max, weights frozen over the
//   last 10%·Gen_max.
// PAPER DEFAULTS — operators (§4, common to every algorithm compared):
//   eta_c = eta_m = 20, p_c = 1.0, p_m = 1/d.
// INHERITED, NOT STATED FOR AdaW: T = 0.1N, delta = 0.9, n_r = 0.01N. §4 gives
//   those three for the MOEA/D PEER, not for AdaW. They reach AdaW through
//   §3.7, which says the initialisation, mating selection (Step 9), variation,
//   reference-point update and population update "follow the practice in Li and
//   Zhang (2009)" — and T is a Require of Alg.1 that the paper never assigns a
//   value to anywhere else, so a value has to come from somewhere. This is the
//   only source in the paper, so the values are effectively forced, but the
//   provenance is delegation and not a direct statement.
// ASSUMPTIONS (gaps in the paper):
//   (1) §3.5 deletes a weight from among those that "share one solution", and
//       illustrates it with two weights sharing s_5 — but never says what makes
//       two entries the same solution. This port uses exact equality of the
//       OBJECTIVE vectors, which is well defined here because a population
//       update copies the solution rather than referencing it, so genuine
//       sharing produces bit-identical objectives.
//   (2) r1 = i as one of the parents. §3.7 delegates mating selection to "the
//       practice in Li and Zhang (2009)", where the current subproblem's own
//       solution is the first parent.
// DECLARED DEVIATIONS: none.
// CONFORMANCE NOTE. §3.2 states, under Eq.2, that "all the objectives are
//   normalised with respect to their minimum and maximum in the considered set
//   in AdaW". That applies to the §3.3 weight-addition geometry as well, so the
//   niche radius r_a and the "undeveloped" test are computed on objectives
//   normalized over A ∪ P — the set the comparison actually spans. Everything
//   downstream of that test (Eq.5-6 optimal weight, the Eq.3 Tchebycheff
//   comparison, the objectives copied into the new population entry) stays in
//   the RAW space, because those are defined against z*. An earlier version
//   computed r_a and the niche test on raw objectives, which made the radius
//   scale-dependent per objective.
// EXTENSIONS BEYOND THE PAPER (off by default): ConstraintMode::FEASIBILITY —
//   an additive penalty*cv term on g; binary variables.
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

struct AdaW_Individual : public Based_Individual {
    double scalar_fitness = 0.0;
};

template <typename Ind_t>
class AdaWCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    // ── hyperparameters (§4) ───────────────────────────────────────────────
    int    T_            = -1;     // -1 -> auto 0.1N
    double delta_        = 0.9;    // probability of mating within B(i)
    int    nr_           = -1;     // -1 -> auto max(1, 0.01N)
    double archive_rate_ = 2.0;    // N_A = 2N
    double eta_c_        = 20.0;
    double eta_m_        = 20.0;
    double pc_           = 1.0;
    double feas_penalty_ = 1e6;    // FEASIBILITY extension
    int    t_max_        = 1000;   // Gen_max
    int    current_gen_  = 0;
    std::mt19937 rng_{std::random_device{}()};

    // ── runtime state ──────────────────────────────────────────────────────
    std::vector<double>              ideal_;   // z* = best − 1e-4 (footnote 2)
    std::vector<std::vector<double>> W_;       // weights [N][m]
    std::vector<std::vector<int>>    B_;       // neighbourhoods [N][T]

    int T_eff(int n) const {
        int t = (T_ > 0) ? T_ : std::max(2, static_cast<int>(std::lround(0.1 * n)));
        return std::min(t, n);
    }
    int nr_eff(int n) const {
        return (nr_ > 0) ? nr_ : std::max(1, static_cast<int>(std::lround(0.01 * n)));
    }

    void generate_weight_vectors(int n, int m) {
        W_ = das_dennis::generate_exact(m, n);
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

    // ── Tchebycheff divided by the weight (§4): g = max_j (f_j − z*_j)/w_j ─
    double tchebycheff(const std::vector<double>& f,
                       const std::vector<double>& w) const {
        double g = -std::numeric_limits<double>::max();
        for (std::size_t j = 0; j < f.size(); ++j) {
            double val = (f[j] - ideal_[j]) / std::max(w[j], 1e-6);
            if (val > g) g = val;
        }
        return g;
    }

    double tchebycheff_cv(const std::vector<double>& f,
                          const std::vector<double>& w, double cv) const {
        double g = tchebycheff(f, w);
        if (constraint_mode == ConstraintMode::FEASIBILITY && cv > 0.0)
            g += feas_penalty_ * cv;
        return g;
    }

    // ── z* = best − 1e-4 (footnote 2), monotone ────────────────────────────
    void update_ideal(const std::vector<double>& f) {
        for (std::size_t j = 0; j < f.size(); ++j)
            if (f[j] - 1e-4 < ideal_[j]) ideal_[j] = f[j] - 1e-4;
    }

    // ── Eq.5–6: the optimal weight of a solution, w = (f − z*)/Σ(f − z*) ───
    std::vector<double> optimal_weight(const std::vector<double>& f) const {
        int m = static_cast<int>(f.size());
        std::vector<double> w(m);
        double s = 0.0;
        for (int j = 0; j < m; ++j) {
            w[j] = std::max(f[j] - ideal_[j], 1e-12);
            s += w[j];
        }
        for (int j = 0; j < m; ++j) w[j] = (s > 1e-30) ? w[j] / s : 1.0 / m;
        return w;
    }

    static double edist(const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) { double t = a[j] - b[j]; s += t * t; }
        return std::sqrt(s);
    }

    // ── §3.2: the argmax D(p) = 1 − Prod R(p,q) index over the objective
    //    set F. The objectives are normalized by the set's min/max; r is the
    //    median distance to the k-th nearest neighbour, k = m. ───────────────
    static int most_crowded(const std::vector<std::vector<double>>& F) {
        int n = static_cast<int>(F.size());
        if (n <= 1) return 0;
        int m = static_cast<int>(F[0].size());
        // normalization by the set's min/max
        std::vector<double> lo(m, std::numeric_limits<double>::max()),
                            hi(m, -std::numeric_limits<double>::max());
        for (const auto& f : F)
            for (int j = 0; j < m; ++j) {
                lo[j] = std::min(lo[j], f[j]);
                hi[j] = std::max(hi[j], f[j]);
            }
        std::vector<std::vector<double>> Fn(n, std::vector<double>(m));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j) {
                double rng = hi[j] - lo[j];
                Fn[i][j] = (rng > 1e-14) ? (F[i][j] - lo[j]) / rng : 0.0;
            }
        // r = the median distance to the k-th (k=m) nearest neighbour
        int k = std::min(m, n - 1);
        std::vector<double> kth(n);
        std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i) {
            std::vector<double> d;
            d.reserve(n - 1);
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                D[i][j] = edist(Fn[i], Fn[j]);
                d.push_back(D[i][j]);
            }
            std::nth_element(d.begin(), d.begin() + (k - 1), d.end());
            kth[i] = d[k - 1];
        }
        std::sort(kth.begin(), kth.end());
        double r = kth[n / 2];
        if (r < 1e-12) r = 1e-12;
        // D(p) = 1 − Prod R(p,q), with R = d/r when d <= r and 1 otherwise (Eq.1–2)
        int worst = 0; double maxD = -1.0;
        for (int p = 0; p < n; ++p) {
            double prod = 1.0;
            for (int q = 0; q < n; ++q) {
                if (p == q) continue;
                double d = D[p][q];
                prod *= (d <= r) ? (d / r) : 1.0;
            }
            double Dp = 1.0 - prod;
            if (Dp > maxD) { maxD = Dp; worst = p; }
        }
        return worst;
    }

    // ── Alg.1 Steps 13–16: update the archive with the offspring ───────────
    void update_archive(DataVault<Ind_t>& vault, int scratch_slot) {
        const auto& f_y = vault.objectives_of(scratch_slot);
        for (std::size_t k = 0; k < vault.archive_size(); ++k) {
            const auto& af = vault.archive_objectives_of(k);
            bool a_dom = true, any_better = false;
            for (std::size_t j = 0; j < f_y.size(); ++j) {
                if (af[j] > f_y[j]) { a_dom = false; break; }
                if (af[j] < f_y[j]) any_better = true;
            }
            if (a_dom && any_better) return;   // some q in A dominates p: skip
        }
        for (int k = static_cast<int>(vault.archive_size()) - 1; k >= 0; --k) {
            const auto& af = vault.archive_objectives_of(static_cast<std::size_t>(k));
            bool y_dom = true, any_b = false;
            for (std::size_t j = 0; j < f_y.size(); ++j) {
                if (f_y[j] > af[j]) { y_dom = false; break; }
                if (f_y[j] < af[j]) any_b = true;
            }
            if (y_dom && any_b)
                vault.archive_erase(static_cast<std::size_t>(k));
        }
        vault.archive_push(scratch_slot);
    }

    // ── Alg.1 Steps 18–20: archive maintenance down to N_A (§3.2) ──────────
    void maintain_archive(DataVault<Ind_t>& vault, int cap) {
        while (static_cast<int>(vault.archive_size()) > cap) {
            int A = static_cast<int>(vault.archive_size());
            std::vector<std::vector<double>> F(A);
            for (int i = 0; i < A; ++i) F[i] = vault.archive_objectives_of(i);
            int worst = most_crowded(F);
            vault.archive_erase(static_cast<std::size_t>(worst));
        }
    }

    // ── one subproblem step for i (Steps 9–16) ─────────────────────────────
    void evolve_subproblem(DataVault<Ind_t>& vault, int i, int n) {
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        bool local = (u01(rng_) < delta_);
        std::vector<int> pool_all;
        if (!local) {
            pool_all.resize(n);
            std::iota(pool_all.begin(), pool_all.end(), 0);
        }
        const std::vector<int>& P = local ? B_[i] : pool_all;

        // parents: x^i plus a random pool member (Li & Zhang 2009 practice).
        std::uniform_int_distribution<int> pickP(0, static_cast<int>(P.size()) - 1);
        int r2 = P[pickP(rng_)];
        for (int attempt = 0; attempt < 5 && r2 == i; ++attempt) r2 = P[pickP(rng_)];

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

        // evaluate the offspring in a scratch slot (the single FE).
        int scratch = static_cast<int>(vault.active_n()) - 1;
        if (bc1.empty()) vault.set_variables(scratch, c1);
        else             vault.set_all_variables(scratch, c1, bc1);
        vault.refresh_objectives(scratch);

        std::vector<double> f_y  = vault.objectives_of(scratch);
        std::vector<double> lm_y = vault.limits_of(scratch);
        double cv_y = (constraint_mode != ConstraintMode::NONE)
                      ? vault.get_cv(scratch) : 0.0;

        // Step 11: z*.
        update_ideal(f_y);

        // Step 12: population update, at most n_r replacements from the pool.
        std::vector<int> Pp = P;
        std::shuffle(Pp.begin(), Pp.end(), rng_);
        int c = 0, nrmax = nr_eff(n);
        for (int j : Pp) {
            if (c >= nrmax) break;
            double g_y  = tchebycheff_cv(f_y, W_[j], cv_y);
            double g_xj = tchebycheff_cv(vault.objectives_of(j), W_[j],
                                         (constraint_mode != ConstraintMode::NONE)
                                         ? vault.get_cv(j) : 0.0);
            if (g_y <= g_xj) {
                // move vars+objs without re-evaluating, to respect the FE budget.
                vault.seed_individual(j, c1, f_y, bc1, lm_y);
                vault.get_ind(j).scalar_fitness = g_y;
                ++c;
            }
        }

        // Steps 13–16: the archive.
        update_archive(vault, scratch);
    }

    // ── Weight update (Steps 21–25; §3.3–3.5) ──────────────────────────────
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

    void weight_update(DataVault<Ind_t>& vault, int n) {
        int A = static_cast<int>(vault.archive_size());
        if (A == 0) return;
        int T = T_eff(n);

        // The paper (§3.2, under Eq.2): "all the objectives are normalised with
        // respect to their minimum and maximum in the considered set in AdaW".
        // The considered set here is A ∪ P, because r_a is compared against
        // archive-to-population distances. Both operands must share one frame.
        const int m = vault.objs_n();
        std::vector<double> lo(m, std::numeric_limits<double>::max()),
                            hi(m, -std::numeric_limits<double>::max());
        auto observe = [&](const std::vector<double>& f) {
            for (int j = 0; j < m; ++j) {
                lo[j] = std::min(lo[j], f[j]);
                hi[j] = std::max(hi[j], f[j]);
            }
        };
        for (int i = 0; i < A; ++i) observe(vault.archive_objectives_of(i));
        for (int i = 0; i < n; ++i) observe(vault.objectives_of(i));
        auto normed = [&](const std::vector<double>& f) {
            std::vector<double> r(m);
            for (int j = 0; j < m; ++j) {
                double rng = hi[j] - lo[j];
                r[j] = (rng > 1e-14) ? (f[j] - lo[j]) / rng : 0.0;
            }
            return r;
        };

        // AF stays RAW: it feeds optimal_weight / tchebycheff and is copied into
        // the new population entry, all of which live in the raw objective
        // space with z*. Only the niche geometry below uses the normalized copy.
        std::vector<std::vector<double>> AF(A);
        for (int i = 0; i < A; ++i) AF[i] = vault.archive_objectives_of(i);
        std::vector<std::vector<double>> AFn(A), PFn(n);
        for (int i = 0; i < A; ++i) AFn[i] = normed(AF[i]);
        for (int i = 0; i < n; ++i) PFn[i] = normed(vault.objectives_of(i));

        // §3.3: the niche radius r_a is the median nearest-neighbour distance
        // within the archive.
        double ra = 1.0;
        if (A >= 2) {
            std::vector<double> nn(A, std::numeric_limits<double>::max());
            for (int a = 0; a < A; ++a)
                for (int b = 0; b < A; ++b) {
                    if (a == b) continue;
                    nn[a] = std::min(nn[a], edist(AFn[a], AFn[b]));
                }
            std::sort(nn.begin(), nn.end());
            ra = nn[A / 2];
            if (ra < 1e-12) ra = 1e-12;
        }

        // working set: the current population (weight plus solution).
        std::vector<Entry> entries;
        entries.reserve(n);
        for (int i = 0; i < n; ++i) entries.push_back(gather_slot(vault, i));

        // undeveloped candidates: an r_a niche holding no population solution.
        std::vector<int> cand;
        for (int a = 0; a < A; ++a) {
            bool undeveloped = true;
            for (int i = 0; i < n; ++i)
                if (edist(AFn[a], PFn[i]) <= ra) { undeveloped = false; break; }
            if (undeveloped) cand.push_back(a);
        }

        // §3.3 addition: the promising test of Eq.3/4 against all T neighbours.
        for (int a : cand) {
            std::vector<double> wq = optimal_weight(AF[a]);
            // the T neighbouring weights of w_q in the current working set.
            int sz = static_cast<int>(entries.size());
            int Tn = std::min(T, sz);
            std::vector<std::pair<double, int>> dist(sz);
            for (int i = 0; i < sz; ++i)
                dist[i] = { weight_dist(wq, entries[i].w), i };
            std::partial_sort(dist.begin(), dist.begin() + Tn, dist.end());

            double gq = tchebycheff(AF[a], wq);
            double sq = std::accumulate(AF[a].begin(), AF[a].end(), 0.0);
            bool promising = true;
            for (int k = 0; k < Tn; ++k) {
                const auto& fp = entries[dist[k].second].objs;
                double gp = tchebycheff(fp, wq);
                double sp = std::accumulate(fp.begin(), fp.end(), 0.0);
                // Eq.3: g(q,w_q) < g(p,w_q); Eq.4: equality plus Σf(q) < Σf(p).
                if (!(gq < gp || (gq == gp && sq < sp))) { promising = false; break; }
            }
            if (!promising) continue;

            // the candidate is admitted together with its weight.
            Entry e;
            e.w    = wq;
            e.objs = AF[a];
            e.vars = vault.archive_variables_of(a);
            e.lims = vault.archive_limits_of(a);
            if (vault.bin_vars_n() > 0) e.bvars = vault.archive_bin_variables_of(a);
            else                        e.bvars.clear();

            // «neighbouring information of q's weight is updated by q»
            // — with no cap on the number of replacements.
            for (int k = 0; k < Tn; ++k) {
                Entry& nb = entries[dist[k].second];
                if (tchebycheff(e.objs, nb.w) <= tchebycheff(nb.objs, nb.w)) {
                    nb.vars  = e.vars;
                    nb.objs  = e.objs;
                    nb.lims  = e.lims;
                    nb.bvars = e.bvars;
                }
            }
            entries.push_back(std::move(e));
        }

        // §3.5 deletion: down to |W| = N.
        while (static_cast<int>(entries.size()) > n) {
            int sz = static_cast<int>(entries.size());
            // group by identical solution (exact equality of the objectives).
            std::vector<int> group(sz, -1);
            int ng = 0;
            for (int i = 0; i < sz; ++i) {
                if (group[i] >= 0) continue;
                group[i] = ng;
                for (int j = i + 1; j < sz; ++j)
                    if (group[j] < 0 && entries[i].objs == entries[j].objs)
                        group[j] = ng;
                ++ng;
            }
            std::vector<int> count(ng, 0);
            for (int i = 0; i < sz; ++i) ++count[group[i]];
            int maxc = *std::max_element(count.begin(), count.end());

            if (maxc > 1) {
                // among the solutions carrying the most weights, remove the
                // weight argmax g(p,w_i) (Eq.7); on a tie between solutions,
                // the one whose worst weight is worse.
                int del = -1; double worst_g = -std::numeric_limits<double>::max();
                for (int g = 0; g < ng; ++g) {
                    if (count[g] != maxc) continue;
                    int gi = -1; double gw = -std::numeric_limits<double>::max();
                    for (int i = 0; i < sz; ++i) {
                        if (group[i] != g) continue;
                        double v = tchebycheff(entries[i].objs, entries[i].w);
                        if (v > gw) { gw = v; gi = i; }
                    }
                    if (gw > worst_g) { worst_g = gw; del = gi; }
                }
                entries.erase(entries.begin() + del);
            } else {
                // every solution is unique -> fall back to the §3.2 crowding.
                std::vector<std::vector<double>> F(sz);
                for (int i = 0; i < sz; ++i) F[i] = entries[i].objs;
                int worst = most_crowded(F);
                entries.erase(entries.begin() + worst);
            }
        }

        // write back, then Step 24: rebuild the neighbourhoods.
        for (int i = 0; i < n; ++i) {
            W_[i] = entries[i].w;
            vault.seed_individual(i, entries[i].vars, entries[i].objs,
                                  entries[i].bvars, entries[i].lims);
        }
        build_neighbourhoods(n);
        for (int i = 0; i < n; ++i)
            vault.get_ind(i).scalar_fitness =
                tchebycheff(vault.objectives_of(i), W_[i]);
    }

    void init_state(DataVault<Ind_t>& vault, int n, int m) {
        ideal_.assign(m, std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) update_ideal(vault.objectives_of(i));
        for (int i = 0; i < n; ++i)
            vault.get_ind(i).scalar_fitness =
                tchebycheff(vault.objectives_of(i), W_[i]);
        current_gen_ = 0;
    }

public:
    AdaWCore() = default;

    void set_neighbourhood_size(int T)   { T_ = T; }
    void set_T                 (int T)   { T_ = T; }   // alias
    void set_delta             (double d){ delta_ = d; }
    void set_nr                (int nr)  { nr_ = std::max(1, nr); }
    void set_feas_penalty      (double p){ feas_penalty_ = p; }
    void set_eta_crossover     (double e){ eta_c_ = e; }
    void set_eta_mutation      (double e){ eta_m_ = e; }
    void set_seed              (unsigned s){ rng_.seed(s); }
    void set_t_max             (int t)   { t_max_ = (t > 0) ? t : 1; }

    std::size_t external_population_size(DataVault<Ind_t>& vault) const {
        return vault.archive_size();
    }

    // ── setup (Steps 1–6) ──────────────────────────────────────────────────
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

        init_state(vault, n, m);

        // Step 5: the non-dominated members of P go into the archive.
        vault.archive_clear();
        vault.expand(1);
        vault.reduce(n + 1);
        for (int i = 0; i < n; ++i) update_archive(vault, i);
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int m = vault.objs_n();
        if (W_.empty()) { generate_weight_vectors(n, m); build_neighbourhoods(n); }

        init_state(vault, n, m);
        vault.archive_clear();
        vault.expand(1);
        vault.reduce(n + 1);
        for (int i = 0; i < n; ++i) update_archive(vault, i);
    }

    // ── step: one generation (Alg.1 Steps 8–26) ────────────────────────────
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();

        // Steps 8–17: over all subproblems in sequence.
        for (int i = 0; i < n; ++i)
            evolve_subproblem(vault, i, n);

        // Steps 18-20: archive maintenance every generation while |A| > N_A.
        int cap = std::max(1, static_cast<int>(std::lround(archive_rate_ * n)));
        maintain_archive(vault, cap);

        // Steps 21-25: weight update every 5% of generations, not in the last 10%.
        ++current_gen_;
        int wag = std::max(1, static_cast<int>(std::lround(0.05 * t_max_)));
        if (current_gen_ % wag == 0 &&
            current_gen_ < static_cast<int>(0.9 * t_max_)) {
            weight_update(vault, n);
        }
    }
};

} // namespace mootation
