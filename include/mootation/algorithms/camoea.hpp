#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// CA-MOEA — A Clustering-Based Adaptive Evolutionary Algorithm for
//   Multiobjective Optimization With Irregular Pareto Fronts
// Y. Hua, Y. Jin, K. Hao — IEEE Transactions on Cybernetics, 2019
// doi:10.1109/TCYB.2018.2834466
//
// Generational scheme (Algorithm 1):
//   1. Q <- Crossover+Mutation(P): SBX + polynomial mutation, |Q| = N.
//   2. P u Q (2N) -> fast non-dominated sort (CDP under FEASIBILITY).
//   3. Accept fronts f_1..f_{l-1} while |F_{l-1}| < N strictly; if |f_l| closes
//      the gap at exactly N, take the whole critical front.
//   4. Otherwise K = N − |F_{l-1}|; normalization (3) by the min/max WITHIN
//      f_l; Algorithm 2: agglomerative Ward clustering (Eq.1) into K clusters,
//      whose centres are the centroids (2).
//   5. Algorithm 3: each individual of f_l goes to its nearest centre
//      (Euclidean); T1 is the nearest individual of each non-empty centre; when
//      n_Cj >= 3 the rest go to T2, otherwise to T3; R = T1, then random picks
//      from T2, then from T3, up to K.
//
// PAPER DEFAULTS (§IV-B): p_c=1.0, p_m=1/V, eta_c=20, eta_m=20; the paper
//   states there are no problem-dependent parameters.
// DECLARED DEVIATIONS: none.
// READINGS of points the paper leaves unspecified:
//   (a) Ward criterion: the minimized quantity is
//       Delta(r,s) = n_r*n_s/(n_r+n_s)*||C_r − C_s||^2, which is monotonically
//       equivalent to d(r,s) = sqrt(2 n_r n_s/(n_r+n_s))*||.||_2 (MATLAB
//       linkage 'ward', d^2 = 2 Delta), so the merge order is identical. The
//       printed Eq.(1) of the paper omits the square on the norm — a typo,
//       confirmed by Eq.(1) of CAVA-MOEA.
//   (b) The T1 tie-break at equal distances takes the first strict minimum and
//       is deterministic; exact ties do not occur in floating point.
//   (c) The paper does not specify mating selection ("Crossover and
//       Mutation(P)"), so random pairs are used, as in the reference practice
//       (NSGA-III / PlatEMO).
//
// Formulas:
//   (2) C_i^c = (Sum_j p_{j,i}^c)/n   — the centroid of cluster c
//   (3) y'_i = (y_i − y_{i,min})/(y_{i,max} − y_{i,min}) — normalization over f_l
//
// EXTENSIONS BEYOND THE PAPER: binary variables (binary_crossover / bit_flip),
//   active only when bin_vars_n() > 0.
// FEASIBILITY: the fast non-dominated sort uses constraint domination (CDP);
//   normalization and clustering work on the objectives.
//
// CAMOEA_Individual stores only rank, the index of the non-dominated front.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../constraint_mode.hpp"
#include "../data_vault.hpp"
#include "../individuals.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

// ── CA-MOEA individual ──────────────────────────────────────────────────────
// rank is the non-dominated front index (0 = best), assigned by the fast NDS.
struct CAMOEA_Individual : public Based_Individual {
    int rank = 0;
};

template <typename Ind_t>
class CAMOEACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double       eta_c_ = 20.0;   // §IV-B: n_c = 20
    double       eta_m_ = 20.0;   // §IV-B: n_m = 20
    double       pc_    = 1.0;    // §IV-B: p_c = 1.0
    std::mt19937 rng_{std::random_device{}()};

    // ── Pareto dominance (with CDP under FEASIBILITY) ──────────────────────
    bool dominates(DataVault<Ind_t>& vault, int a, int b) const {
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            double ca = vault.get_cv(a), cb = vault.get_cv(b);
            bool af = (ca <= 0.0), bf = (cb <= 0.0);
            if (af && !bf) return true;
            if (!af && bf) return false;
            if (!af && !bf) return ca < cb;     // both infeasible -> smaller CV
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

    // ── Fast non-dominated sort over the pool [0, pool) -> fronts ──────────
    std::vector<std::vector<int>> fast_nds(DataVault<Ind_t>& vault, int pool) {
        std::vector<std::vector<int>> S(pool);
        std::vector<int> ndom(pool, 0);
        std::vector<int> f0;
        for (int i = 0; i < pool; ++i) {
            for (int j = 0; j < pool; ++j) {
                if (i == j) continue;
                if (dominates(vault, i, j)) S[i].push_back(j);
                else if (dominates(vault, j, i)) ++ndom[i];
            }
            if (ndom[i] == 0) { vault.get_ind(i).rank = 0; f0.push_back(i); }
        }
        std::vector<std::vector<int>> fronts;
        fronts.push_back(f0);
        std::vector<int> cur = f0;
        int r = 0;
        while (!cur.empty()) {
            std::vector<int> next;
            for (int i : cur)
                for (int j : S[i])
                    if (--ndom[j] == 0) { vault.get_ind(j).rank = r + 1; next.push_back(j); }
            if (next.empty()) break;
            fronts.push_back(next);
            cur = next;
            ++r;
        }
        return fronts;
    }

    // ── Normalization (3): by the min/max within the set idx ───────────────
    std::vector<std::vector<double>>
    normalize(DataVault<Ind_t>& vault, const std::vector<int>& idx) {
        int m = vault.objs_n();
        int n = static_cast<int>(idx.size());
        std::vector<double> ymin(m,  std::numeric_limits<double>::max());
        std::vector<double> ymax(m, -std::numeric_limits<double>::max());
        for (int v : idx) {
            const auto& o = vault.objectives_of(v);
            for (int j = 0; j < m; ++j) {
                ymin[j] = std::min(ymin[j], o[j]);
                ymax[j] = std::max(ymax[j], o[j]);
            }
        }
        std::vector<std::vector<double>> Y(n, std::vector<double>(m));
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(idx[i]);
            for (int j = 0; j < m; ++j) {
                double range = ymax[j] - ymin[j];
                Y[i][j] = (range > 1e-14) ? (o[j] - ymin[j]) / range : 0.0;
            }
        }
        return Y;
    }

    static double sq_dist(const std::vector<double>& a,
                          const std::vector<double>& b) {
        double s = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) {
            double d = a[j] - b[j];
            s += d * d;
        }
        return s;
    }

    // ── Algorithm 2: hierarchical Ward clustering -> K centroids ───────────
    // Agglomerative: every point starts as a cluster (centroid = the point,
    // size 1); while the cluster count exceeds K, merge the pair with the
    // smallest Ward increment Delta(r,s) = n_r*n_s/(n_r+n_s)*||C_r−C_s||^2. The
    // new centre is the weighted mean.
    std::vector<std::vector<double>>
    calc_cluster_centers(const std::vector<std::vector<double>>& Y, int K) {
        int n = static_cast<int>(Y.size());
        int m = (n > 0) ? static_cast<int>(Y[0].size()) : 0;
        std::vector<std::vector<double>> cen = Y;     // cluster centroids
        std::vector<int>  sz(n, 1);
        std::vector<char> alive(n, 1);
        int nclust = n;
        while (nclust > K) {
            double best = std::numeric_limits<double>::max();
            int ba = -1, bb = -1;
            for (int a = 0; a < n; ++a) {
                if (!alive[a]) continue;
                for (int b = a + 1; b < n; ++b) {
                    if (!alive[b]) continue;
                    double w = (static_cast<double>(sz[a]) * sz[b])
                             / (sz[a] + sz[b]) * sq_dist(cen[a], cen[b]);
                    if (w < best) { best = w; ba = a; bb = b; }
                }
            }
            if (ba < 0) break;
            int na = sz[ba], nb = sz[bb];
            for (int j = 0; j < m; ++j)
                cen[ba][j] = (cen[ba][j] * na + cen[bb][j] * nb) / (na + nb);
            sz[ba] = na + nb;
            alive[bb] = 0;
            --nclust;
        }
        std::vector<std::vector<double>> C;
        C.reserve(K);
        for (int i = 0; i < n; ++i) if (alive[i]) C.push_back(cen[i]);
        return C;   // K centroids (fewer on degenerate data)
    }

    // ── Algorithm 3: Clustering-Based Selection ────────────────────────────
    // Returns K indices (values drawn from fl), selected by the centres C.
    std::vector<int>
    clustering_based_selection(const std::vector<std::vector<double>>& C,
                               const std::vector<std::vector<double>>& Y,
                               const std::vector<int>& fl, int K) {
        int n  = static_cast<int>(fl.size());
        int Kc = static_cast<int>(C.size());

        // Assignment: each point goes to its nearest centre (Euclidean).
        std::vector<std::vector<int>> members(Kc);    // positions i (within fl) per centre
        std::vector<double>           dist(n, 0.0);
        for (int i = 0; i < n; ++i) {
            int    best_j = 0;
            double best_d = std::numeric_limits<double>::max();
            for (int j = 0; j < Kc; ++j) {
                double d = sq_dist(Y[i], C[j]);
                if (d < best_d) { best_d = d; best_j = j; }
            }
            dist[i] = best_d;
            members[best_j].push_back(i);
        }

        // T1: the individual nearest each non-empty centre; T2/T3: the rest.
        std::vector<int> T1, T2, T3;
        for (int j = 0; j < Kc; ++j) {
            if (members[j].empty()) continue;
            int imin = members[j][0];
            for (int i : members[j]) if (dist[i] < dist[imin]) imin = i;
            T1.push_back(fl[imin]);
            bool big = (static_cast<int>(members[j].size()) >= 3);
            for (int i : members[j]) {
                if (i == imin) continue;
                (big ? T2 : T3).push_back(fl[i]);
            }
        }

        std::vector<int> R = T1;
        int need = K - static_cast<int>(T1.size());
        if (need <= 0) { R.resize(K); return R; }

        if (static_cast<int>(T2.size()) >= need) {
            std::shuffle(T2.begin(), T2.end(), rng_);
            for (int i = 0; i < need; ++i) R.push_back(T2[i]);
        } else {
            for (int v : T2) R.push_back(v);
            need -= static_cast<int>(T2.size());
            std::shuffle(T3.begin(), T3.end(), rng_);
            for (int i = 0; i < need && i < static_cast<int>(T3.size()); ++i)
                R.push_back(T3[i]);
        }
        return R;
    }

    // ── Move the survivors into [0, n) and truncate the pool ───────────────
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
    CAMOEACore() = default;

    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_pc(double p)            { pc_ = p; }
    void set_seed(unsigned s)        { rng_.seed(s); }

    // ── setup: random initialization ───────────────────────────────────────
    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
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
    }

    void setup_seeded(DataVault<Ind_t>& /*vault*/) { /* population already seeded */ }

    // ── step: one generation (Algorithm 1) ─────────────────────────────────
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int N = n;
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_int(0, n - 1);

        // (3-4) Reproduction: random parent pairing + SBX + polynomial mutation.
        int off_base = vault.expand(n);
        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        const double pm = (vault.vars_n() > 0)
            ? 1.0 / static_cast<double>(vault.vars_n()) : 0.0;   // §IV-B: p_m=1/V
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

        // (5) Merged pool of 2N -> fast non-dominated sort.
        int pool_size = n * 2;
        auto fronts = fast_nds(vault, pool_size);

        // Accept fronts f_1..f_{l-1} while their total size is < N (strictly).
        std::vector<int> survivors;
        survivors.reserve(N);
        std::size_t fi = 0;
        while (fi < fronts.size() &&
               static_cast<int>(survivors.size() + fronts[fi].size()) < N) {
            for (int v : fronts[fi]) survivors.push_back(v);
            ++fi;
        }

        if (fi < fronts.size()) {
            const std::vector<int>& fl = fronts[fi];          // the critical front
            int K = N - static_cast<int>(survivors.size());   // = N − |F_{l-1}|
            if (static_cast<int>(fl.size()) == K) {
                // (6-7) |F_l| = N: take the whole critical front.
                for (int v : fl) survivors.push_back(v);
            } else {
                // (9-12) cluster f_l and select K by the centres.
                auto Y = normalize(vault, fl);                 // (3)
                auto C = calc_cluster_centers(Y, K);           // Algorithm 2
                auto R = clustering_based_selection(C, Y, fl, K); // Algorithm 3
                for (int v : R) survivors.push_back(v);
            }
        }

        rearrange(vault, survivors, pool_size);
    }
};

} // namespace mootation
