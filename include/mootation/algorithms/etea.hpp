#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// ETEA — A Euclidean Minimum Spanning Tree-Based Evolutionary Algorithm for
//        Multi-Objective Optimization.
// Miqing Li, Shengxiang Yang, Jinhua Zheng, Xiaohui Liu — Evolutionary
// Computation 22(2):189-230, MIT Press, 2014.
// doi:10.1162/EVCO_a_00106
//
// IDEA. A Euclidean minimum spanning tree (EMST, Prim's algorithm) is built
// over the objective vectors. The EMST edge lengths give a density estimate
// (ETCD, Eq.1). Convergence — the "distance count" D(i) (Eq.2-3): for a
// dominated individual we take the nearest nondominated neighbor j that
// dominates it, and count the number of nondominated k that are closer to j
// than i is (+1). Fitness F(i)=D(i)+1/(ETCD(i)+1) (Eq.4): nondom. in (0,1],
// dom. >1, smaller = better.
//
// GENERATIONAL SCHEME (Algorithm 1):
//   R = Copy(P, Q)                                              (line 3)
//   Fitness_assignment(R)                                       (line 4, §3.2)
//   Q' = nondominated(R)                                        (line 5, elitism)
//   if |Q'| < N:  Q' ∪= Fitness_adjustment(R)  (Alg.2, §3.3)    (line 7)
//   else:         Q' = Archive_truncation(Q')  (Alg.3, §3.4)    (line 9)
//   P  = Mating_selection(Q')                                   (line 11)
//   P  = Variation(P)                                           (line 12)
//   return Q                                                    (line 15)
//
// EMST (Def.4, Eq.1): ETCD(X)=( Σ_i L_{XY_i}^{0.5} / d )^2, where d is the
//   node degree of X.
// Archive_truncation (Alg.3): Prim; we find the minimum-weight edge L_{pq};
//   if the degree of one of the ends is 1 — we remove the OTHER one;
//   otherwise we compare the modified ETCD' (Eq. in lines 12-13, edge L_{pq}
//   excluded), and remove the individual with the SMALLER ETCD' (the more
//   crowded one).
//
// PAPER DEFAULTS (§4.3): L=N=pop_size; SBX η_c=20, p_c=1.0;
//   PM η_m=20, p_m=1/n.
//   Mating (line 11) — binary tournament on F. The paper deliberately leaves
//   this open, and the explicit sentence is in §3.1, not §3.3: "this paper
//   only focuses on fitness assignment (line 4) and environmental selection
//   ... In other words, mating selection and variation schemes in ETEA are not
//   determined and can be freely selected by users". §3.3 adds only that
//   mating "is usually performed in a random way". Binary tournament on F is
//   the NSGA-II/SPEA2 standard and is what this port fills the hole with — a
//   choice this library makes, not a value the paper reports.
//
// DECLARED DEVIATIONS:
//   ETEA-1 (INVARIANT, not a deviation). All Euclidean distances — EMST, ETCD
//     and the distance count — are computed on RAW objective values, per Def.4
//     and Eq.1. The paper prescribes no normalization, so none is applied.
//     (An earlier version of this port normalized min-max over the pool R and
//     declared that as a deviation; the normalization was removed and the
//     declaration outlived it. distance_matrix is the sole distance producer
//     and feeds all three sites.)
//   ETEA-2 (MINOR). Mating — binary tournament on F (the paper leaves this
//     free; see the note above for the exact §3.1 wording). Variation —
//     SBX + PM, which §4.3 does state: "the distribution indexes in both SBX
//     and the polynomial mutation are set to 20", p_c = 1.0, p_m = 1/n.
//   ETEA-3 (MINOR). Findout_neighbor (Alg.2 line 4): the neighbors of p are
//     the dominated individuals within a radius = the distance from p to
//     its nearest dominating nondom. neighbor (§3.3, text after Alg.2).
//     Penalty F(q_i)+=i by decreasing distance (line 5-8): the far ones get
//     a smaller penalty (i is smaller).
//   ETEA-4 (MINOR). Real-valued genome; binary is outside the paper's coverage.
// CONSTRAINTS (beyond the paper, off by default). constraint_mode
//   FEASIBILITY/CDP makes the Alg.1 line-5 nondomination test a CONSTRAINED
//   one, which is the single gate every later step reads: the EMST density and
//   the distance count are computed on the same pool, but membership of the
//   nondominated set — and therefore F(i) ∈ (0,1] versus F(i) > 1 — follows
//   Deb's feasibility rules. li2014 studies unconstrained problems only.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../constraint_mode.hpp"
#include "../detail/constrained.hpp"
#include "../data_vault.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

template <typename Ind_t>
class ETEACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

    ETEACore() = default;
    void set_seed(unsigned s) { rng_.seed(s); }
    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation(double e)  { eta_m_ = e; }

    void setup(DataVault<Ind_t>& vault) {
        m_ = vault.objs_n();
        N_ = vault.pop_size();
        const auto& bd = vault.get_bounds();
        std::uniform_real_distribution<double> d(0.0, 1.0);
        std::vector<double> vars(vault.vars_n());
        for (int i = 0; i < N_; ++i) {
            for (int j = 0; j < vault.vars_n(); ++j) {
                double lo = bd[j].first.value_or(0.0), hi = bd[j].second.value_or(1.0);
                vars[j] = lo + d(rng_) * (hi - lo);
            }
            vault.set_variables(i, vars);
        }
        vault.sync();
        // P_0 = current population; Q_0 = ∅ (Alg.1 line 1).
        P_.clear();
        for (int i = 0; i < N_; ++i)
            P_.push_back(Sol{vault.variables_of(i), vault.objectives_of(i),
                             cv_of(vault, i)});
        Q_.clear();
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        m_ = vault.objs_n();
        N_ = vault.pop_size();
        P_.clear();
        for (int i = 0; i < (int)vault.active_n(); ++i)
            P_.push_back(Sol{vault.variables_of(i), vault.objectives_of(i),
                             cv_of(vault, i)});
        Q_.clear();
    }

    void step(DataVault<Ind_t>& vault) {
        // ── R = Copy(P, Q) (Alg.1 line 3) ────────────────────────────────
        std::vector<Sol> R;
        R.reserve(P_.size() + Q_.size());
        for (auto& s : P_) R.push_back(s);
        for (auto& s : Q_) R.push_back(s);
        int n = (int)R.size();

        // ── Fitness_assignment(R) (Alg.1 line 4, §3.2) ───────────────────
        std::vector<double> F(n);
        std::vector<int>    Dcount(n);
        std::vector<char>   nondom(n);
        assign_fitness(R, F, Dcount, nondom);

        // ── Elitism: Q' = nondominated(R) (Alg.1 line 5) ─────────────────
        std::vector<int> ndIdx;
        for (int i = 0; i < n; ++i) if (nondom[i]) ndIdx.push_back(i);

        std::vector<Sol> Qnext;
        if ((int)ndIdx.size() < N_) {
            // Fitness_adjustment: top up with the best dominated ones
            // (Alg.1 line 7).
            for (int i : ndIdx) Qnext.push_back(R[i]);
            int need = N_ - (int)ndIdx.size();
            auto extra = fitness_adjustment(R, F, Dcount, nondom, need);
            for (int i : extra) Qnext.push_back(R[i]);
        } else if ((int)ndIdx.size() == N_) {
            for (int i : ndIdx) Qnext.push_back(R[i]);
        } else {
            // Archive_truncation (Alg.1 line 9, Alg.3).
            std::vector<Sol> Qcand;
            for (int i : ndIdx) Qcand.push_back(R[i]);
            Qnext = archive_truncation(Qcand, N_);
        }

        Q_ = Qnext;

        // ── Mating_selection(Q') + Variation (Alg.1 line 11-12) ──────────
        breed_population(vault);

        // Write ARCHIVE Q (Alg.1 line 15: return Q) as the final population.
        store_result(vault);
    }

private:
    struct Sol { std::vector<double> vars, objs; double cv = 0.0; };

    std::mt19937 rng_{std::random_device{}()};
    int          m_ = 0, N_ = 0;
    double       eta_c_ = 20.0, eta_m_ = 20.0, pc_ = 1.0;
    std::vector<Sol> P_, Q_;

    double pm_eff(int nv) const { return nv > 0 ? 1.0 / nv : 0.0; }

    double cv_of(DataVault<Ind_t>& vault, int slot) const {
        return (constraint_mode == ConstraintMode::NONE) ? 0.0 : vault.get_cv(slot);
    }

    // Constraint-aware domination between two pool members.
    bool dominates(const Sol& a, const Sol& b) const {
        return detail::dominates(constraint_mode, a.objs, a.cv, b.objs, b.cv);
    }
    // Matrix of RAW Euclidean distances in objective space, per Def.4/Eq.1 —
    // the paper specifies no normalization, so none is applied here. This is
    // the sole distance producer: assign_fitness, fitness_adjustment and
    // archive_truncation all consume it (see ETEA-1).
    std::vector<std::vector<double>>
    distance_matrix(const std::vector<Sol>& R) const {
        int n = (int)R.size();
        std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i)
            for (int k = i + 1; k < n; ++k) {
                double s = 0.0;
                for (int j = 0; j < m_; ++j) {
                    double d = R[i].objs[j] - R[k].objs[j];
                    s += d * d;
                }
                D[i][k] = D[k][i] = std::sqrt(s);
            }
        return D;
    }

    // ── EMST via Prim's algorithm over distance matrix D on indices idx ────
    // Returns the edge list (a,b) (indices into idx) and each node's degree.
    // ETCD(X)=( Σ L_{XY}^{0.5} / deg )^2  (Def.4, Eq.1).
    void build_emst(const std::vector<int>& idx,
                    const std::vector<std::vector<double>>& D,
                    std::vector<std::pair<int,int>>& edges,  // indices into idx
                    std::vector<int>& degree,
                    std::vector<double>& etcd) const {
        int s = (int)idx.size();
        edges.clear();
        degree.assign(s, 0);
        etcd.assign(s, 0.0);
        if (s <= 1) return;

        // Prim: inTree, minimum edge to the tree.
        std::vector<char>   inTree(s, 0);
        std::vector<double> minw(s, std::numeric_limits<double>::max());
        std::vector<int>    parent(s, -1);
        minw[0] = 0.0;
        std::vector<double> sumSqrt(s, 0.0);
        for (int it = 0; it < s; ++it) {
            int u = -1; double best = std::numeric_limits<double>::max();
            for (int v = 0; v < s; ++v)
                if (!inTree[v] && minw[v] < best) { best = minw[v]; u = v; }
            if (u < 0) break;
            inTree[u] = 1;
            if (parent[u] >= 0) {
                edges.emplace_back(u, parent[u]);
                ++degree[u]; ++degree[parent[u]];
                double w = D[idx[u]][idx[parent[u]]];
                double sq = std::sqrt(w);
                sumSqrt[u] += sq; sumSqrt[parent[u]] += sq;
            }
            for (int v = 0; v < s; ++v) {
                if (inTree[v]) continue;
                double w = D[idx[u]][idx[v]];
                if (w < minw[v]) { minw[v] = w; parent[v] = u; }
            }
        }
        for (int v = 0; v < s; ++v)
            etcd[v] = (degree[v] > 0)
                ? std::pow(sumSqrt[v] / degree[v], 2.0)
                : 0.0;
    }

    // ── Fitness_assignment (§3.2, Eq.1-4) over the whole pool R ───────────
    void assign_fitness(const std::vector<Sol>& R,
                        std::vector<double>& F,
                        std::vector<int>& Dcount,
                        std::vector<char>& nondom) const {
        int n = (int)R.size();
        auto D = distance_matrix(R);

        // Nondomination.
        nondom.assign(n, 1);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (i != j && dominates(R[j], R[i])) { nondom[i] = 0; break; }

        std::vector<int> NDS;
        for (int i = 0; i < n; ++i) if (nondom[i]) NDS.push_back(i);

        // Distance count D(i) (Eq.2-3).
        Dcount.assign(n, 0);
        for (int i = 0; i < n; ++i) {
            if (nondom[i]) { Dcount[i] = 0; continue; }
            // j = nearest NONDOM. neighbor dominating i (Eq.3).
            int    jbest = -1;
            double dbest = std::numeric_limits<double>::max();
            for (int j : NDS) {
                if (!dominates(R[j], R[i])) continue;
                if (D[i][j] < dbest) { dbest = D[i][j]; jbest = j; }
            }
            if (jbest < 0) {
                // no dominating nondom. neighbor — take the nearest nondom.
                for (int j : NDS)
                    if (D[i][j] < dbest) { dbest = D[i][j]; jbest = j; }
            }
            int cnt = 0;
            if (jbest >= 0) {
                double Lij = D[i][jbest];
                for (int k : NDS) {
                    if (k == jbest) continue;
                    if (D[jbest][k] < Lij) ++cnt;   // L_{jk} < L_{ij}
                }
            }
            Dcount[i] = cnt + 1;   // +1 penalty for the dominated (Eq.2).
        }

        // ETCD from the EMST of the whole pool (Def.4, Eq.1).
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        std::vector<std::pair<int,int>> edges;
        std::vector<int> degree;
        std::vector<double> etcd;
        build_emst(all, D, edges, degree, etcd);

        // F(i)=D(i)+1/(ETCD(i)+1) (Eq.4).
        F.assign(n, 0.0);
        for (int i = 0; i < n; ++i)
            F[i] = (double)Dcount[i] + 1.0 / (etcd[i] + 1.0);
    }

    // ── Fitness_adjustment (Alg.2, §3.3): select the `need` best dominated
    //    individuals, with a hierarchical penalty applied to the neighbors.
    //    Returns indices into R. ─────────────────────────────────────────
    std::vector<int> fitness_adjustment(const std::vector<Sol>& R,
                                        std::vector<double> F,    // a copy (we edit it)
                                        const std::vector<int>& /*Dcount*/,
                                        const std::vector<char>& nondom,
                                        int need) const {
        int n = (int)R.size();
        auto D = distance_matrix(R);

        std::vector<int> NDS;
        for (int i = 0; i < n; ++i) if (nondom[i]) NDS.push_back(i);

        // Neighborhood radius of a dominated individual p = the distance to
        // its nearest dominating nondom. neighbor (§3.3, text after Alg.2).
        auto radius_of = [&](int p) -> double {
            double best = std::numeric_limits<double>::max();
            for (int j : NDS)
                if (dominates(R[j], R[p]) && D[p][j] < best) best = D[p][j];
            if (best == std::numeric_limits<double>::max()) {
                for (int j : NDS) if (D[p][j] < best) best = D[p][j];
            }
            return (best == std::numeric_limits<double>::max()) ? 0.0 : best;
        };

        std::vector<char> removed(n, 0);
        for (int i = 0; i < n; ++i) if (nondom[i]) removed[i] = 1;  // dominated only
        std::vector<int> S;
        S.reserve(need);

        while ((int)S.size() < need) {
            // Findout_best: min. F among the remaining dominated (Alg.2 line 3).
            int    p = -1; double fp = std::numeric_limits<double>::max();
            for (int i = 0; i < n; ++i)
                if (!removed[i] && F[i] < fp) { fp = F[i]; p = i; }
            if (p < 0) break;

            // Findout_neighbor: the dominated within the radius (Alg.2 line 4).
            double rad = radius_of(p);
            std::vector<int> T;
            for (int q = 0; q < n; ++q) {
                if (q == p || removed[q]) continue;
                if (D[p][q] <= rad) T.push_back(q);
            }
            // Sort(T,p): decreasing distance from p (Alg.2 line 5).
            std::sort(T.begin(), T.end(),
                      [&](int a, int b){ return D[p][a] > D[p][b]; });
            // F(q_i) += i  (Alg.2 line 6-8): far ones (small i) — mild penalty.
            for (int i = 0; i < (int)T.size(); ++i)
                F[T[i]] += (double)(i + 1);

            S.push_back(p);          // Alg.2 line 9
            removed[p] = 1;          // Alg.2 line 10
        }
        return S;
    }

    // ── Archive_truncation (Alg.3, §3.4): remove iteratively while |Q|>N ──
    std::vector<Sol> archive_truncation(std::vector<Sol> Q, int target) const {
        while ((int)Q.size() > target) {
            int s = (int)Q.size();
            auto D = distance_matrix(Q);
            std::vector<int> idx(s);
            std::iota(idx.begin(), idx.end(), 0);
            std::vector<std::pair<int,int>> edges;
            std::vector<int> degree;
            std::vector<double> etcd;
            build_emst(idx, D, edges, degree, etcd);   // Prim (Alg.3 line 3)

            if (edges.empty()) { Q.pop_back(); continue; }

            // Find_minimum: minimum-weight edge L_{pq} (Alg.3 line 4).
            int ep = -1, eq = -1; double emin = std::numeric_limits<double>::max();
            for (auto& e : edges) {
                double w = D[e.first][e.second];
                if (w < emin) { emin = w; ep = e.first; eq = e.second; }
            }

            int toRemove;
            if (degree[ep] == 1 || degree[eq] == 1) {
                // degree of one =1 → remove the OTHER (Alg.3 line 5-10).
                toRemove = (degree[ep] == 1) ? eq : ep;
            } else {
                // modified ETCD' without edge L_{pq} (Alg.3 line 12-13).
                double sq = std::sqrt(emin);
                double ep_p = std::pow(
                    (std::sqrt(etcd[ep]) * degree[ep] - sq) / (degree[ep] - 1), 2.0);
                double eq_p = std::pow(
                    (std::sqrt(etcd[eq]) * degree[eq] - sq) / (degree[eq] - 1), 2.0);
                // remove the individual with the SMALLER ETCD' (the more
                // crowded one, line 14-18).
                toRemove = (ep_p < eq_p) ? ep : eq;
            }
            Q.erase(Q.begin() + toRemove);
        }
        return Q;
    }

    // ── Mating (line 11) + Variation (line 12) → new P_ of size N ─────────
    void breed_population(DataVault<Ind_t>& vault) {
        int scratch = vault.expand(1);
        const auto& bd = vault.get_bounds();
        int nv = vault.vars_n();

        // Fitness of Q_ members for the tournament: recompute F over Q_ (as
        // a pool). §3.3: mating relies on the same fitness information.
        std::vector<double> Fq;
        {
            std::vector<int>  dc; std::vector<char> nd;
            assign_fitness(Q_, Fq, dc, nd);
        }
        int qn = (int)Q_.size();
        std::uniform_int_distribution<int> pick(0, qn - 1);
        auto tourn = [&]() -> int {
            int a = pick(rng_), b = pick(rng_);
            return (Fq[a] <= Fq[b]) ? a : b;   // smaller F = better
        };

        std::vector<Sol> newP;
        newP.reserve(N_);
        std::vector<double> c1, c2;
        while ((int)newP.size() < N_) {
            int a = tourn(), b = tourn();
            ops::sbx(Q_[a].vars, Q_[b].vars, c1, c2, bd, eta_c_, pc_, rng_);
            ops::polynomial_mutation(c1, bd, eta_m_, pm_eff(nv), rng_);
            ops::polynomial_mutation(c2, bd, eta_m_, pm_eff(nv), rng_);

            vault.set_variables(scratch, c1);
            vault.refresh_objectives(scratch);
            newP.push_back(Sol{c1, vault.objectives_of(scratch),
                               cv_of(vault, scratch)});
            if ((int)newP.size() < N_) {
                vault.set_variables(scratch, c2);
                vault.refresh_objectives(scratch);
                newP.push_back(Sol{c2, vault.objectives_of(scratch),
                                   cv_of(vault, scratch)});
            }
        }
        P_ = std::move(newP);
    }

    // Write archive Q_ as the final active population (Alg.1 line 15).
    void store_result(DataVault<Ind_t>& vault) {
        vault.reduce(0);
        vault.expand((int)Q_.size());
        for (int i = 0; i < (int)Q_.size(); ++i)
            vault.seed_individual((std::size_t)i, Q_[i].vars, Q_[i].objs, {}, {});
    }
};

} // namespace mootation
