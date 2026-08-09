#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MaOEA-ARV — Many-Objective Evolutionary Algorithm with Adaptive Reference Vector
// Zhang, Wang, Li, Hu, Li, Wu — Information Sciences, 2021
// doi:10.1016/j.ins.2021.01.015          (source: zhang2021)
//
// Generation scheme (Algorithm 1):
//   1. NDS of population P; Con ← Adaptive_reference_vector(P,N,M) (Alg.2):
//      z*, z^nad over P; Z=(z*+z^nad)/2 (Eq.10); knee X_T = argmin of Σ_{q≠k}Σ_m
//      positive exceedances (Eq.11); R = (Z−F(X_T))/‖Z−F(X_T)‖ — the sign follows
//      the numerical example of Fig.2 (the printed Eq.9 "(F−Z)/‖·‖" is a sign
//      typo, arbitration #1); Con(X) = F(X)·R (Eq.8), smaller Con is better.
//   2. Mating (Alg.3): binary tournament — dominance → smaller Con → coin flip.
//   3. SBX + PM → N offspring; P' = P ∪ Q (2N individuals).
//   4. Env. selection (Alg.4): complete fronts are taken whole; the critical front
//      F_i is split by hierarchical clustering (Euclidean metric) into K clusters;
//      from each cluster — argmin Con (the ARV is recomputed per-cluster
//      over the |S_k| members, Alg.4 line 5).
//
// Defaults = §5.1.1: p_c=1.0, p_m=1/D. Deviations: η_c=η_m=20 and the linkage
//   of the hierarchical clustering (single-linkage here) are NOT specified
//   by the paper — implementation conventions were chosen.
// Extensions beyond the paper: constraint_mode FEASIBILITY (CDP), binary
//   variables (off by default).
// ============================================================================
// FIX 2026-06 (arbitration #1 CRITICAL BUG): the sign of R was
// inverted — the code followed the printed Eq.9 R=(F(X_knee)−Z)/‖·‖, whereas the
// numerical example of Fig.2 in the paper is unambiguous: Z=(5.5,5.5), knee=C(3,4),
// R' = (5√34/34, 3√34/34) = (Z−F(C))/‖Z−F(C)‖, with ordering
// Con'(C)<Con'(B)<Con'(D)<Con'(A)<Con'(E)<Con'(F). With the previous sign,
// Con = −Con' and argmin (mating Alg.3 + env-selection Alg.4) selected the WORST
// solutions per the paper. After the fix, a manual re-computation of the example:
// Con·√34 = C:27 < B:31 < D:34 < A:35 < E:46 < F:53 — matches the paper, argmin = C.
//
// ── Individual: MaOEAARV_Individual (see individuals.hpp) ────────────────────
//   rank    — Pareto non-domination front index (0 = best)
//   fitness — Con(X) = F(X)·R; smaller = better (per-generation, mating)

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

template <typename Ind_t>
class MaOEAARVCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double       eta_c_ = 20.0;   // not specified by the paper (convention, see header)
    double       eta_m_ = 20.0;   // not specified by the paper (convention, see header)
    double       pc_    = 1.0;    // §5.1.1: "crossover probability … 1.0"
    std::mt19937 rng_{std::random_device{}()};

    // ── Dominance ──────────────────────────────────────────────────────────
    bool dominates(DataVault<Ind_t>& vault, int a, int b) const {
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            double ca = vault.get_cv(a), cb = vault.get_cv(b);
            bool af = (ca<=0.0), bf = (cb<=0.0);
            if( af&&!bf) return true;
            if(!af&& bf) return false;
            if(!af&&!bf) return ca < cb;
        }
        const auto& fa = vault.objectives_of(a);
        const auto& fb = vault.objectives_of(b);
        bool better = false;
        for (std::size_t i = 0; i < fa.size(); ++i) {
            if (fa[i] > fb[i]) return false;
            if (fa[i] < fb[i]) better = true;
        }
        return better;
    }

    // ── Fast NDS ───────────────────────────────────────────────────────────
    std::vector<std::vector<int>> fast_nds(DataVault<Ind_t>& vault, int n) {
        std::vector<std::vector<int>> S(n);
        std::vector<int> np(n, 0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                if (dominates(vault, i, j)) S[i].push_back(j);
                else if (dominates(vault, j, i)) ++np[i];
            }
        std::vector<std::vector<int>> fronts;
        std::vector<int> f0;
        for (int i = 0; i < n; ++i)
            if (np[i] == 0) { vault.get_ind(i).rank = 0; f0.push_back(i); }
        fronts.push_back(f0);
        int k = 0;
        while (!fronts[k].empty()) {
            std::vector<int> nxt;
            for (int i : fronts[k])
                for (int j : S[i])
                    if (--np[j] == 0) { vault.get_ind(j).rank = k+1; nxt.push_back(j); }
            fronts.push_back(nxt); ++k;
        }
        fronts.pop_back();
        return fronts;
    }

    // ── Adaptive Reference Vector + Con(X) (Algorithm 2) ─────────────────
    // Computes R from the knee point of `members`, then sets ind.fitness =
    // Con(X) = F(X)·R for each member.
    void compute_arv_fitness(DataVault<Ind_t>& vault,
                             const std::vector<int>& members) {
        int K = static_cast<int>(members.size());
        int m = vault.objs_n();
        if (K == 0) return;

        // z_nad, z* over members.
        std::vector<double> znad(m, -std::numeric_limits<double>::max());
        std::vector<double> zstar(m,  std::numeric_limits<double>::max());
        for (int v : members) {
            const auto& o = vault.objectives_of(v);
            for (int j = 0; j < m; ++j) {
                znad[j]  = std::max(znad[j],  o[j]);
                zstar[j] = std::min(zstar[j], o[j]);
            }
        }

        // Z = (z_nad + z*) / 2
        std::vector<double> Z(m);
        for (int j = 0; j < m; ++j) Z[j] = (znad[j] + zstar[j]) * 0.5;

        // Knee point Xi = argmin_k Σ_{q≠k} Σ_m ((f_m(Xk)-f_m(Xq))·C_m)
        // C_m = 1 if f_m(Xk) > f_m(Xq), else 0
        // i.e. sum of objectives where Xk is worse than Xq, summed over all q≠k.
        int knee_idx = members[0];
        double knee_score = std::numeric_limits<double>::max();
        for (int ki = 0; ki < K; ++ki) {
            int k = members[ki];
            const auto& fk = vault.objectives_of(k);
            double score = 0.0;
            for (int qi = 0; qi < K; ++qi) {
                if (qi == ki) continue;
                const auto& fq = vault.objectives_of(members[qi]);
                for (int j = 0; j < m; ++j)
                    if (fk[j] > fq[j]) score += fk[j] - fq[j];
            }
            if (score < knee_score) { knee_score = score; knee_idx = k; }
        }

        // R = (Z - F(Xi)) / ||Z - F(Xi)||
        // FIX 2026-06 (arbitration #1): the printed Eq.9 "(F(Xi)−Z)/‖·‖" is a sign
        // typo; the numerical example of Fig.2 (R'=(5,3)/√34=(Z−F(C))/‖·‖ with
        // Z=(5.5,5.5), C=(3,4)) requires Z−F(knee). With the previous sign, argmin
        // Con inverted the selection (the worst solutions per the paper were chosen).
        const auto& fknee = vault.objectives_of(knee_idx);
        std::vector<double> R(m);
        double rnorm = 0.0;
        for (int j = 0; j < m; ++j) {
            R[j] = Z[j] - fknee[j];
            rnorm += R[j] * R[j];
        }
        rnorm = std::sqrt(rnorm);
        if (rnorm < 1e-14) {
            // Degenerate: fallback to uniform R.
            for (int j = 0; j < m; ++j) R[j] = 1.0 / std::sqrt(m);
        } else {
            for (int j = 0; j < m; ++j) R[j] /= rnorm;
        }

        // Con(X) = F(X)·R for each member.
        for (int v : members) {
            const auto& o = vault.objectives_of(v);
            double con = 0.0;
            for (int j = 0; j < m; ++j) con += o[j] * R[j];
            vault.get_ind(v).fitness = con;
        }
    }

    // ── Hierarchical clustering (Euclidean, single-linkage for speed) ─────
    // Divides `members` into exactly k clusters.
    // Returns cluster assignments: cluster_of[i] = cluster index for members[i].
    // Uses agglomerative approach: start with n singletons, merge until k remain.
    std::vector<int> hierarchical_cluster(
            DataVault<Ind_t>& vault,
            const std::vector<int>& members, int k) const {
        int n = static_cast<int>(members.size());
        if (k >= n) {
            // Each its own cluster.
            std::vector<int> c(n);
            std::iota(c.begin(), c.end(), 0);
            return c;
        }

        // cluster_of[i] = label of members[i]. Start: each is own cluster.
        std::vector<int> label(n);
        std::iota(label.begin(), label.end(), 0);
        int n_clusters = n;

        // Pairwise Euclidean distances.
        int m = vault.objs_n();
        auto dist = [&](int a, int b) {
            const auto& fa = vault.objectives_of(members[a]);
            const auto& fb = vault.objectives_of(members[b]);
            double s = 0.0;
            for (int j = 0; j < m; ++j) { double d = fa[j]-fb[j]; s += d*d; }
            return std::sqrt(s);
        };

        // Cluster members: group_of[label] = list of member indices with that label.
        std::vector<std::vector<int>> groups(n);
        for (int i = 0; i < n; ++i) groups[i] = {i};

        while (n_clusters > k) {
            // Find two active clusters with smallest single-linkage distance.
            double best = std::numeric_limits<double>::max();
            int best_a = -1, best_b = -1;

            // Collect active labels.
            std::vector<int> active;
            for (int i = 0; i < n; ++i)
                if (!groups[i].empty()) active.push_back(i);

            for (int ai = 0; ai < static_cast<int>(active.size()); ++ai)
                for (int bi = ai+1; bi < static_cast<int>(active.size()); ++bi) {
                    int la = active[ai], lb = active[bi];
                    // Single-linkage: min pairwise distance.
                    for (int ia : groups[la])
                        for (int ib : groups[lb]) {
                            double d = dist(ia, ib);
                            if (d < best) { best = d; best_a = la; best_b = lb; }
                        }
                }

            if (best_a < 0) break;

            // Merge best_b into best_a.
            for (int idx : groups[best_b]) {
                label[idx] = best_a;
                groups[best_a].push_back(idx);
            }
            groups[best_b].clear();
            --n_clusters;
        }

        // Remap labels to 0..k-1.
        std::unordered_map<int,int> remap;
        int next = 0;
        for (int i = 0; i < n; ++i)
            if (remap.find(label[i]) == remap.end())
                remap[label[i]] = next++;

        std::vector<int> result(n);
        for (int i = 0; i < n; ++i) result[i] = remap[label[i]];
        return result;
    }

    // ── Rearrange vault ────────────────────────────────────────────────────
    void rearrange(DataVault<Ind_t>& vault,
                   const std::vector<int>& survivors, int pool) {
        int n = static_cast<int>(survivors.size());
        std::vector<int> pos(pool), at_pos(pool);
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

    // ── Binary tournament (Algorithm 3) ───────────────────────────────────
    int tournament(DataVault<Ind_t>& vault,
                   std::uniform_int_distribution<int>& dist) {
        int a = dist(rng_), b = dist(rng_);
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            double ca = vault.get_cv(a), cb = vault.get_cv(b);
            bool af = (ca<=0.0), bf = (cb<=0.0);
            if( af&&!bf) return a;
            if(!af&& bf) return b;
            if(!af&&!bf) return (ca < cb) ? a : b;
        }
        // Xa ≺ Xb → Xa; non-dominated → lower Con; tie → random.
        bool a_dom_b = dominates(vault, a, b);
        bool b_dom_a = dominates(vault, b, a);
        if ( a_dom_b && !b_dom_a) return a;
        if (!a_dom_b &&  b_dom_a) return b;
        // Non-dominated: lower Con (fitness) is better.
        if (vault.get_ind(a).fitness < vault.get_ind(b).fitness) return a;
        if (vault.get_ind(b).fitness < vault.get_ind(a).fitness) return b;
        // Tie: random.
        std::uniform_int_distribution<int> coin(0, 1);
        return coin(rng_) ? a : b;
    }

public:
    MaOEAARVCore() = default;

    void set_eta_crossover(double e)  { eta_c_ = e; }
    void set_eta_mutation (double e)  { eta_m_ = e; }
    void set_pc           (double p)  { pc_   = p; }
    void set_seed         (unsigned s){ rng_.seed(s); }

    // ── setup ──────────────────────────────────────────────────────────────
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
        // Initial NDS + ARV fitness for mating.
        fast_nds(vault, n);
        std::vector<int> all(n); std::iota(all.begin(), all.end(), 0);
        compute_arv_fitness(vault, all);
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        fast_nds(vault, n);
        std::vector<int> all(n); std::iota(all.begin(), all.end(), 0);
        compute_arv_fitness(vault, all);
    }

    // ── step (Algorithm 1) ─────────────────────────────────────────────────
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_int(0, n - 1);

        // ── Breed N offspring ─────────────────────────────────────────────
        vault.expand(vault.pop_size());
        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        // §5.1.1: p_c = 1.0, p_m = 1/D.
        double pm = (vault.vars_n() > 0) ? 1.0 / vault.vars_n() : 0.0;
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
                if (i+1 < n) ops::bit_flip_mutation(bc2, vault.bin_vars_n(), rng_);
                vault.set_all_variables(n+i, c1, bc1);
                if (i+1 < n) vault.set_all_variables(n+i+1, c2, bc2);
            } else {
                vault.set_variables(n+i, c1);
                if (i+1 < n) vault.set_variables(n+i+1, c2);
            }
        }
        vault.sync();

        // ── NDS over combined 2N pool ─────────────────────────────────────
        auto fronts = fast_nds(vault, n * 2);

        // ── Environmental selection (Algorithm 4) ─────────────────────────
        // Accumulate complete fronts until adding next would exceed N.
        std::vector<int> survivors;
        survivors.reserve(n);
        const std::vector<int>* last_front = nullptr;

        for (auto& front : fronts) {
            if (static_cast<int>(survivors.size() + front.size()) <= n) {
                for (int v : front) survivors.push_back(v);
                if (static_cast<int>(survivors.size()) == n) break;
            } else {
                last_front = &front;
                break;
            }
        }

        if (last_front && static_cast<int>(survivors.size()) < n) {
            int K = n - static_cast<int>(survivors.size());
            const auto& Fi = *last_front;

            // Hierarchical clustering of Fi into K clusters.
            auto cluster_of = hierarchical_cluster(vault, Fi, K);

            // Build clusters.
            std::vector<std::vector<int>> clusters(K);
            for (int i = 0; i < static_cast<int>(Fi.size()); ++i)
                clusters[cluster_of[i]].push_back(Fi[i]);

            // From each cluster, compute ARV fitness and pick argmin Con.
            for (int ci = 0; ci < K; ++ci) {
                if (clusters[ci].empty()) continue;
                compute_arv_fitness(vault, clusters[ci]);
                int best = *std::min_element(
                    clusters[ci].begin(), clusters[ci].end(),
                    [&](int a, int b) {
                        return vault.get_ind(a).fitness < vault.get_ind(b).fitness;
                    });
                survivors.push_back(best);
            }
        }

        // Trim to exactly N.
        if (static_cast<int>(survivors.size()) > n)
            survivors.resize(n);

        rearrange(vault, survivors, n * 2);

        // ── Recompute ARV fitness for next generation's mating ────────────
        // Use all N survivors together to find knee point + compute Con.
        std::vector<int> all(n); std::iota(all.begin(), all.end(), 0);
        compute_arv_fitness(vault, all);
    }
};

} // namespace mootation
