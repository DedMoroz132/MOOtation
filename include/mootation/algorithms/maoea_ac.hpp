#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MaOEA/AC — An Adaptive Clustering-based Evolutionary Algorithm for
//   Many-objective Optimization Problems
// Liu, Yu, Lin, Tan — Information Sciences 537 (2020) 261–283
// doi:10.1016/j.ins.2020.03.104
//
// Generational scheme (Algorithm 1):
//   1. Q <- SBX+PM on P (the paper specifies no mating selection, so pairs are
//      random); U = P u Q (2N).
//   2. ACS (Alg.2): NDS on U; S <- fronts until |S| >= N; normalization Eq.2:
//      f~_i = (f_i − z*_i)/(z^nad_i − z*_i); z* = the min over S; z^nad by the
//      NSGA-III [13] method (ASF extremes of S -> hyperplane -> intercepts;
//      on degeneracy or negative intercepts, fall back to z^nad_i = max over S).
//   3. AdaptiveEstimation of p (Alg.3): discard f~ > 1 from F1;
//      dis(x) = (Sum f~_i − 1)/sqrt(M) (3); avg (4), std (5);
//      d(p) = (M^{1−1/p} − 1)/sqrt(M) (6); p in {0.5, 1, 2} by (7) with
//      mu = 0.5 − 0.02M; correction (8): std/|avg| < 0.1 -> p = 1.0.
//   4. HierarchicalClustering(S, N, p) (Alg.4): agglomerative; projection onto
//      H_p as f~(c)/(Sum f~_k(c)^p)^{1/p} (9); Sim = the Euclidean distance
//      between projections (10); merge the minimum pair (11); the centroid is
//      the mean f~ of the members (12).
//   5. EliteSelection (Alg.5): the argmax-Sim pair (13), taken from different
//      clusters, enters P; then M−2 more by Strategy I (max-min projection
//      distance to P); from the remaining clusters, argmin CF(x) = Sum_j f~_j
//      (14).
//
// PAPER DEFAULTS (§4.2): p_c=1.0 (Table 1 says "p_c=1/n", which is a typo; the
//   text of §4.2 says "p_c = 1.0"), p_m=1/n, eta_c=30, eta_m=20;
//   mu = 0.5 − 0.02M.
// DECLARED DEVIATIONS AND READINGS:
//   (a) Eq.13: the pair is taken from DIFFERENT clusters. Alg.5 line 4 removes
//       "two clusters", although the formula alone does not require it — this
//       is a reading by intent;
//   (b) Eq.8: "||std/avg||" is read as the coefficient of variation std/|avg|
//       with a guard |avg| > 1e-12 (when avg is near zero, Eq.7 already yields
//       p = 1.0);
//   (c) the paper does not say which population z* and z^nad are estimated
//       from; S is used, since Alg.2 line 7 normalizes exactly S;
//   (d) the paper specifies no mating selection, so random pairs are used as
//       the neutral reading;
//   (e) the safety top-up to exactly N by CF is beyond the paper and is
//       normally dead code.
// EXTENSIONS BEYOND THE PAPER (off by default): constraint_mode FEASIBILITY
//   (CDP inside NDS), binary variables.
// ============================================================================
// Notable fixes: (1) MAC-1: eta_c 20 -> 30 (§4.2); (2) MAC-2: z^nad used to be
// the component-wise max over U, which forced f~ into [0,1] and made the
// "f~ > 1" filter of Alg.3 line 1 dead code; it now uses the NSGA-III [13]
// intercept method with a per-objective fallback to the max over S; (3) AC-5:
// the distance matrix in the clustering is cached, giving O(M·N^2) + O(N^3)
// instead of O(M·N^3); (4) the RNG seed is std::random_device plus set_seed,
// replacing time(nullptr); (5) explicit p_c/p_m in the current SBX/PM
// signatures.

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

struct MaOEAAC_Individual : public Based_Individual {
    int rank = 0;
};

template <typename Ind_t>
class MaOEAACCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double       eta_c_ = 30.0;   // §4.2: η_c = 30 (FIX 2026-06, MAC-1)
    double       eta_m_ = 20.0;   // §4.2: η_m = 20
    double       pc_    = 1.0;    // §4.2: p_c = 1.0 (Table 1's "1/n" is a typo)
    std::mt19937 rng_{std::random_device{}()};
    int          objs_ = 0;

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

    std::vector<std::vector<int>> fast_nds(DataVault<Ind_t>& vault, int pool) {
        std::vector<std::vector<int>> S(pool);
        std::vector<int> ndom(pool, 0), cur;
        for (int i = 0; i < pool; ++i) {
            for (int j = 0; j < pool; ++j) {
                if (i == j) continue;
                if (dominates(vault, i, j)) S[i].push_back(j);
                else if (dominates(vault, j, i)) ++ndom[i];
            }
            if (ndom[i] == 0) cur.push_back(i);
        }
        std::vector<std::vector<int>> fronts;
        while (!cur.empty()) {
            fronts.push_back(cur);
            std::vector<int> next;
            for (int i : cur)
                for (int j : S[i])
                    if (--ndom[j] == 0) next.push_back(j);
            cur = next;
        }
        return fronts;
    }

    // ── L^p projection onto H_p (Eq.9) ─────────────────────────────────────
    static std::vector<double> project(const std::vector<double>& f, double p) {
        double s = 0.0;
        for (double fk : f) s += std::pow(std::max(0.0, fk), p);
        double lp = std::pow(s, 1.0 / p);
        std::vector<double> pr(f.size());
        if (lp < 1e-30) return f;
        for (std::size_t i = 0; i < f.size(); ++i) pr[i] = f[i] / lp;
        return pr;
    }

    static double eucl(const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) { double d = a[j] - b[j]; s += d * d; }
        return std::sqrt(s);
    }

    // ── Normalization Eq.2: z* = min over S; z^nad by NSGA-III [13] ────────
    // MAC-2: ASF extremes of S -> hyperplane -> intercepts a_i; the denominator
    // of Eq.2 is a_i (z^nad_i − z*_i). Per-objective fallback
    // a_i = max_S f_i − z*_i when x_i <= 0 ("no/negative intercepts"), and a
    // full fallback on a degenerate linear system — as in canonical NSGA-III.
    struct NormParams { std::vector<double> zmin, intercepts; };
    NormParams compute_norm(DataVault<Ind_t>& vault, const std::vector<int>& S) const {
        int m = vault.objs_n();
        NormParams np;
        np.zmin.assign(m, std::numeric_limits<double>::max());
        for (int v : S) {
            const auto& o = vault.objectives_of(v);
            for (int j = 0; j < m; ++j) np.zmin[j] = std::min(np.zmin[j], o[j]);
        }
        const double eps = 1e-6;
        std::vector<int> extreme(m, -1);
        for (int i = 0; i < m; ++i) {
            double best = std::numeric_limits<double>::max();
            for (int v : S) {
                const auto& o = vault.objectives_of(v);
                double asf = 0.0;
                for (int j = 0; j < m; ++j) {
                    double w = (i == j) ? 1.0 : eps;
                    asf = std::max(asf, (o[j] - np.zmin[j]) / w);
                }
                if (asf < best) { best = asf; extreme[i] = v; }
            }
        }
        np.intercepts.assign(m, 1.0);
        std::vector<double> nadir(m, -std::numeric_limits<double>::max());
        for (int v : S) {
            const auto& o = vault.objectives_of(v);
            for (int j = 0; j < m; ++j) nadir[j] = std::max(nadir[j], o[j]);
        }
        bool deg = false;
        for (int i = 0; i < m && deg == false; ++i) if (extreme[i] < 0) deg = true;
        if (deg == false) {
            std::vector<std::vector<double>> A(m, std::vector<double>(m + 1));
            for (int i = 0; i < m; ++i) {
                const auto& o = vault.objectives_of(extreme[i]);
                for (int j = 0; j < m; ++j) A[i][j] = o[j] - np.zmin[j];
                A[i][m] = 1.0;
            }
            for (int col = 0; col < m && deg == false; ++col) {
                int piv = col;
                for (int r = col + 1; r < m; ++r)
                    if (std::abs(A[r][col]) > std::abs(A[piv][col])) piv = r;
                std::swap(A[col], A[piv]);
                if (std::abs(A[col][col]) < 1e-12) { deg = true; break; }
                for (int r = col + 1; r < m; ++r) {
                    double f = A[r][col] / A[col][col];
                    for (int k = col; k <= m; ++k) A[r][k] -= f * A[col][k];
                }
            }
            if (deg == false) {
                std::vector<double> x(m);
                for (int i = m - 1; i >= 0; --i) {
                    x[i] = A[i][m];
                    for (int j = i + 1; j < m; ++j) x[i] -= A[i][j] * x[j];
                    x[i] /= A[i][i];
                }
                for (int i = 0; i < m; ++i) {
                    if (x[i] > 1e-12) np.intercepts[i] = 1.0 / x[i];
                    else              np.intercepts[i] = nadir[i] - np.zmin[i];
                }
            }
        }
        if (deg)
            for (int j = 0; j < m; ++j) np.intercepts[j] = nadir[j] - np.zmin[j];
        return np;
    }

    // ── Algorithm 3: adaptive estimation of p from the front F1 ────────────
    double adaptive_estimate_p(const std::vector<int>& F1, int M,
                               const std::vector<std::vector<double>>& Fn) {
        std::vector<int> cand;
        for (int v : F1) {
            bool ok = true;
            for (int j = 0; j < M; ++j) if (Fn[v][j] > 1.0 + 1e-9) { ok = false; break; }
            if (ok) cand.push_back(v);
        }
        if (cand.empty()) return 1.0;
        double sm = std::sqrt(static_cast<double>(M));
        std::vector<double> dis(cand.size());
        double avg = 0.0;
        for (std::size_t t = 0; t < cand.size(); ++t) {
            double s = 0.0;
            for (int j = 0; j < M; ++j) s += Fn[cand[t]][j];
            dis[t] = (s - 1.0) / sm;
            avg += dis[t];
        }
        avg /= static_cast<double>(cand.size());
        double var = 0.0;
        if (cand.size() > 1) {
            for (double d : dis) var += (d - avg) * (d - avg);
            var /= static_cast<double>(cand.size() - 1);
        }
        double sd = std::sqrt(var);

        double d05 = (std::pow(static_cast<double>(M), 1.0 - 1.0 / 0.5) - 1.0) / sm;
        double d20 = (std::pow(static_cast<double>(M), 1.0 - 1.0 / 2.0) - 1.0) / sm;
        double mu  = 0.5 - M * 0.02;

        double p;
        if (avg < mu * d05)      p = 0.5;
        else if (avg > mu * d20) p = 2.0;
        else                     p = 1.0;

        if (std::abs(avg) > 1e-12 && sd / std::abs(avg) < 0.1) p = 1.0;   // (8)
        return p;
    }

    // ── Algorithm 4: hierarchical clustering on the H_p projections ────────
    std::vector<std::vector<int>>
    hierarchical_clustering(const std::vector<int>& S, int N, double p,
                            const std::vector<std::vector<double>>& Fn) {
        int M = objs_;
        int n = static_cast<int>(S.size());
        std::vector<std::vector<int>>    members(n);
        std::vector<std::vector<double>> cen(n), proj(n);
        std::vector<char> alive(n, 1);
        for (int i = 0; i < n; ++i) {
            members[i] = {S[i]};
            cen[i]  = Fn[S[i]];
            proj[i] = project(cen[i], p);
        }
        // AC-5 (performance only, semantics unchanged): the matrix of
        // projection distances (10) is cached; after a merge only the row and
        // column of the new centroid are refreshed.
        std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));
        for (int a = 0; a < n; ++a)
            for (int b = a + 1; b < n; ++b)
                D[a][b] = D[b][a] = eucl(proj[a], proj[b]);        // (10)
        int nclust = n, target = std::min(N, n);
        while (nclust > target) {
            double best = std::numeric_limits<double>::max();
            int ba = -1, bb = -1;
            for (int a = 0; a < n; ++a) {
                if (!alive[a]) continue;
                for (int b = a + 1; b < n; ++b) {
                    if (!alive[b]) continue;
                    if (D[a][b] < best) { best = D[a][b]; ba = a; bb = b; }  // (11)
                }
            }
            if (ba < 0) break;
            for (int v : members[bb]) members[ba].push_back(v);
            std::vector<double> nc(M, 0.0);
            for (int v : members[ba])
                for (int j = 0; j < M; ++j) nc[j] += Fn[v][j];
            double inv = 1.0 / static_cast<double>(members[ba].size());
            for (int j = 0; j < M; ++j) nc[j] *= inv;               // (12)
            cen[ba]  = nc;
            proj[ba] = project(nc, p);                             // (9)
            alive[bb] = 0; --nclust;
            for (int b = 0; b < n; ++b)
                if (alive[b] && b != ba)
                    D[ba][b] = D[b][ba] = eucl(proj[ba], proj[b]);
        }
        std::vector<std::vector<int>> out;
        for (int i = 0; i < n; ++i) if (alive[i]) out.push_back(members[i]);
        return out;
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
    MaOEAACCore() = default;

    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_pc           (double p) { pc_   = p; }
    void set_seed(unsigned s)        { rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        objs_ = vault.objs_n();
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

    void setup_seeded(DataVault<Ind_t>& vault) { objs_ = vault.objs_n(); }

    // ── step: one generation (Algorithm 1) ─────────────────────────────────
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int N = n;
        int M = vault.objs_n();
        objs_ = M;
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_int(0, n - 1);

        // Reproduction with random pairing -> offspring in [off_base, +n).
        int off_base = vault.expand(n);
        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        // §4.2: p_c = 1.0, p_m = 1/n.
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

        int pool = n * 2;

        // ── AdaptiveClustering (Algorithm 2) ───────────────────────────────
        auto fronts = fast_nds(vault, pool);
        std::vector<int> Sset;
        std::size_t fi = 0;
        for (; fi < fronts.size(); ++fi) {
            for (int v : fronts[fi]) Sset.push_back(v);
            if (static_cast<int>(Sset.size()) >= N) break;
        }

        // Normalization (2): z* = min over S; z^nad by the NSGA-III [13]
        // method. MAC-2: z^nad used to be the component-wise max over U, which
        // forced f~ into [0,1] and made the "f~ > 1" filter of Alg.3 line 1
        // dead code. The paper (§3.2): "To estimate z* and z^nad, the method
        // used in [20] and [13] is adopted", with f~ "basically" — not
        // strictly — in [0,1].
        NormParams np = compute_norm(vault, Sset);
        std::vector<std::vector<double>> Fn(pool, std::vector<double>(M, 0.0));
        std::vector<double> CF(pool, 0.0);
        for (int v = 0; v < pool; ++v) {
            const auto& o = vault.objectives_of(v);
            double cf = 0.0;
            for (int j = 0; j < M; ++j) {
                double d = np.intercepts[j];
                double val = (std::abs(d) > 1e-12) ? (o[j] - np.zmin[j]) / d : 0.0;
                Fn[v][j] = std::max(val, 0.0);   // >1.0 is allowed (Alg.3 filter)
                cf += Fn[v][j];
            }
            CF[v] = cf;                                            // (14)
        }

        double p = adaptive_estimate_p(fronts.empty() ? Sset : fronts[0], M, Fn);
        auto clusters = hierarchical_clustering(Sset, N, p, Fn);   // N clusters

        // ── EliteSelection (Algorithm 5) ───────────────────────────────────
        int Kc = static_cast<int>(clusters.size());
        std::vector<int> cluster_of(pool, -1);
        std::vector<std::vector<double>> proj(pool);
        for (int t = 0; t < Kc; ++t)
            for (int v : clusters[t]) { cluster_of[v] = t; proj[v] = project(Fn[v], p); }

        std::vector<int>  P;
        std::vector<char> used_cl(Kc, 0);
        int Ksel = std::min(M, Kc);

        // (13) the pair of solutions with the maximum projection distance,
        // taken from different clusters.
        if (Ksel >= 2) {
            int bu = -1, bv = -1; double bd = -1.0;
            for (int t = 0; t < Kc; ++t)
                for (int u : clusters[t])
                    for (int t2 = t + 1; t2 < Kc; ++t2)
                        for (int w : clusters[t2]) {
                            double d = eucl(proj[u], proj[w]);
                            if (d > bd) { bd = d; bu = u; bv = w; }
                        }
            if (bu >= 0) {
                P.push_back(bu); used_cl[cluster_of[bu]] = 1;
                P.push_back(bv); used_cl[cluster_of[bv]] = 1;
            }
        } else if (Ksel == 1 && Kc >= 1) {
            // a single cluster for diversity: take an arbitrary representative
            int u = clusters[0][0];
            P.push_back(u); used_cl[0] = 1;
        }

        // Strategy I: (M−2) more by maximum minimum projection distance to P.
        while (static_cast<int>(P.size()) < Ksel) {
            int best = -1; double bestd = -1.0;
            for (int t = 0; t < Kc; ++t) {
                if (used_cl[t]) continue;
                for (int v : clusters[t]) {
                    double dmin = std::numeric_limits<double>::max();
                    for (int y : P) dmin = std::min(dmin, eucl(proj[v], proj[y]));
                    if (dmin > bestd) { bestd = dmin; best = v; }
                }
            }
            if (best < 0) break;
            P.push_back(best); used_cl[cluster_of[best]] = 1;
        }

        // Remaining clusters: the solution with the best (minimum) CF.
        for (int t = 0; t < Kc; ++t) {
            if (used_cl[t]) continue;
            int qbest = clusters[t][0];
            for (int v : clusters[t]) if (CF[v] < CF[qbest]) qbest = v;
            P.push_back(qbest); used_cl[t] = 1;
        }

        // safety top-up to exactly N (if fewer than N clusters emerged).
        if (static_cast<int>(P.size()) < N) {
            std::vector<char> inP(pool, 0);
            for (int v : P) inP[v] = 1;
            std::vector<int> rest;
            for (int v = 0; v < pool; ++v) if (!inP[v]) rest.push_back(v);
            std::sort(rest.begin(), rest.end(),
                      [&](int a, int b){ return CF[a] < CF[b]; });
            for (int v : rest) {
                if (static_cast<int>(P.size()) >= N) break;
                P.push_back(v);
            }
        }
        if (static_cast<int>(P.size()) > N) {
            std::sort(P.begin(), P.end(),
                      [&](int a, int b){ return CF[a] < CF[b]; });
            P.resize(N);
        }

        rearrange(vault, P, pool);
    }
};

} // namespace mootation
