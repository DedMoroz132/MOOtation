#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MaOEA/C — A Clustering-based Evolutionary Algorithm for Many-objective
//   Optimization Problems
// Q. Lin, S. Liu, K.-C. Wong, M. Gong, C.A. Coello Coello, J. Chen, J. Zhang —
//   IEEE Transactions on Evolutionary Computation, 2019
// doi:10.1109/TEVC.2018.2866927
//
// Generational scheme (Algorithm 4):
//   1. PCM(P,m) (Alg.1): m axial centroids e_1..e_m with cs = |P|/m; the axes,
//      in random order, take the cs individuals nearest by angle (2).
//   2. Mating: for each C_i^PCM, produce |C_i^PCM| offspring; with probability
//      eps = 0.8 both parents come from C_i^PCM (similarity mating), otherwise
//      from all of P; SBX -> u, polynomial mutation -> v; v joins Q (|Q| = N).
//   3. Environmental_Selection(P,Q,N,m) (Alg.3): U = P u Q (2N); normalization
//      (3); the convergence indicator c(u) = Sum_j f~_j(u) (6); PCM(U,m).
//   4. For each axis i: NDS within C_i^PCM, accumulating fronts into S until
//      |S| >= k (k = N/m); HCM(S,k) (Alg.2): agglomeratively merge the pair of
//      clusters whose centroids subtend the smallest angle (4), the centroid
//      being the mean f~ of the members (5).
//   5. The cluster C_d nearest the axis by (4) contributes the individual
//      nearest the axis by (2) — diversity; the other k−1 clusters contribute
//      their minimum c(.) — convergence. In total m*k = N.
//
// PAPER DEFAULTS (Table I): p_c=1.0, p_m=1/n, eta_c=30, eta_m=20, eps=0.8.
// DECLARED DEVIATIONS: none. (At m < 4 the f^min/f^max of Eq.3 are taken over
//   the non-dominated individuals of the set only — the paper's own caveat
//   after Eq.3.)
// READINGS of points the paper leaves unspecified:
//   (a) HCM performs an exact global search for the minimum pair at every step,
//       instead of the "lazy" theta/index updates of the Alg.2 pseudocode. That
//       is the canonical agglomerative algorithm and preserves the semantics of
//       Eq.(4)–(5); the linkage is by centroids, as Eq.(4)–(5) explicitly
//       define — the paper's prose "average-link" contradicts its own formulas.
//   (b) When N is not a multiple of m (the paper requires that it be), the
//       remainder N%m goes to the lowest-indexed axes, and the last PCM axis
//       absorbs the whole remainder |S|%m.
//   (c) similarity mating draws parents from C_i with replacement, so p1 may
//       equal p2; the paper does not say.
//   (d) The safety top-up / truncation by c(.) to exactly |P| = N in the case
//       of degenerate clusters is beyond the paper and normally does not fire.
// EXTENSIONS BEYOND THE PAPER: binary variables (binary_crossover / bit_flip),
//   active only when bin_vars_n() > 0.
// FEASIBILITY: the NDS uses CDP.
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

struct MaOEAC_Individual : public Based_Individual {
    int rank = 0;
};

template <typename Ind_t>
class MaOEACCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    static constexpr double PI_ = 3.14159265358979323846;

    double       eta_c_ = 30.0;    // Table I
    double       eta_m_ = 20.0;    // Table I
    double       pc_    = 1.0;     // Table I: p_c = 1.0
    double       delta_ = 0.8;     // eps, the similarity-mating probability (Table I)
    std::mt19937 rng_{std::random_device{}()};

    // ── Pareto dominance (CDP under FEASIBILITY) ───────────────────────────
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

    // ── NDS over an arbitrary subset idx -> fronts (values from idx) ────────
    std::vector<std::vector<int>>
    nds_subset(DataVault<Ind_t>& vault, const std::vector<int>& idx) {
        int n = static_cast<int>(idx.size());
        std::vector<std::vector<int>> S(n);
        std::vector<int> ndom(n, 0);
        std::vector<int> cur;
        for (int a = 0; a < n; ++a) {
            for (int b = 0; b < n; ++b) {
                if (a == b) continue;
                if (dominates(vault, idx[a], idx[b])) S[a].push_back(b);
                else if (dominates(vault, idx[b], idx[a])) ++ndom[a];
            }
            if (ndom[a] == 0) cur.push_back(a);
        }
        std::vector<std::vector<int>> fronts;
        while (!cur.empty()) {
            std::vector<int> front;
            front.reserve(cur.size());
            for (int a : cur) front.push_back(idx[a]);
            fronts.push_back(front);
            std::vector<int> next;
            for (int a : cur)
                for (int b : S[a])
                    if (--ndom[b] == 0) next.push_back(b);
            cur = next;
        }
        return fronts;
    }

    // ── Normalization (3) over the set idx -> Fn (indexed by pool position) ─
    // The paper's caveat to Eq.(3): "when the number of objectives is less than
    // four, f_i^max and f_i^min should be found only from non-dominated
    // individuals" — at m < 4, min/max are taken over the non-dominated
    // members of idx.
    std::vector<std::vector<double>>
    normalize(DataVault<Ind_t>& vault, const std::vector<int>& idx, int pool) {
        int m = vault.objs_n();
        std::vector<int> ref = idx;            // source of f^min / f^max
        if (m < 4) {
            std::vector<int> nd;
            for (int a : idx) {
                bool dom = false;
                for (int b : idx) {
                    if (b == a) continue;
                    if (dominates(vault, b, a)) { dom = true; break; }
                }
                if (dom == false) nd.push_back(a);
            }
            if (nd.empty() == false) ref = nd;
        }
        std::vector<double> fmin(m,  std::numeric_limits<double>::max());
        std::vector<double> fmax(m, -std::numeric_limits<double>::max());
        for (int v : ref) {
            const auto& o = vault.objectives_of(v);
            for (int j = 0; j < m; ++j) {
                fmin[j] = std::min(fmin[j], o[j]);
                fmax[j] = std::max(fmax[j], o[j]);
            }
        }
        std::vector<std::vector<double>> Fn(pool, std::vector<double>(m, 0.0));
        for (int v : idx) {
            const auto& o = vault.objectives_of(v);
            for (int j = 0; j < m; ++j) {
                double range = fmax[j] - fmin[j];
                Fn[v][j] = (range > 1e-14) ? (o[j] - fmin[j]) / range : 0.0;
            }
        }
        return Fn;
    }

    static double vnorm(const std::vector<double>& f) {
        double s = 0.0;
        for (double x : f) s += x * x;
        return std::sqrt(s);
    }

    // ── Angle from an individual to the axis vector e_axis (Eq.2) ──────────
    static double angle_to_axis(const std::vector<double>& f, int axis) {
        double nf = vnorm(f);
        if (nf < 1e-30) return PI_ / 2.0;
        double c = f[axis] / nf;
        c = std::max(-1.0, std::min(1.0, c));
        return std::acos(c);
    }

    // ── Angle between two vectors (Eq.4): arccos|cos|, absolute per the paper ─
    static double angle_vec(const std::vector<double>& a,
                            const std::vector<double>& b) {
        double dot = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) dot += a[j] * b[j];
        double denom = vnorm(a) * vnorm(b);
        if (denom < 1e-30) return PI_ / 2.0;
        double c = std::abs(dot) / denom;
        c = std::min(1.0, c);
        return std::acos(c);
    }

    // ── Algorithm 1: PCM(idx, m) -> m clusters (index = axis) ───────────────
    std::vector<std::vector<int>>
    pcm(const std::vector<int>& idx, int m,
        const std::vector<std::vector<double>>& Fn) {
        std::vector<std::vector<int>> clusters(m);
        std::vector<int> remaining = idx;
        int total = static_cast<int>(idx.size());
        int cs = total / m;
        std::vector<int> order(m);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng_);
        for (int a = 0; a < m; ++a) {
            int axis = order[a];
            if (a == m - 1) { clusters[axis] = remaining; break; }
            // the cs individuals nearest by angle (2) to the axis
            std::sort(remaining.begin(), remaining.end(), [&](int p, int q) {
                return angle_to_axis(Fn[p], axis) < angle_to_axis(Fn[q], axis);
            });
            int take = std::min(cs, static_cast<int>(remaining.size()));
            clusters[axis].assign(remaining.begin(), remaining.begin() + take);
            remaining.erase(remaining.begin(), remaining.begin() + take);
        }
        return clusters;
    }

    // ── Algorithm 2: HCM(S, k) -> k clusters and their centroids ────────────
    void hcm(const std::vector<int>& S, int k,
             const std::vector<std::vector<double>>& Fn,
             std::vector<std::vector<int>>& out_clusters,
             std::vector<std::vector<double>>& out_centroids) {
        int m = vault_objs_;
        int n = static_cast<int>(S.size());
        std::vector<std::vector<int>>    members(n);
        std::vector<std::vector<double>> cen(n);
        std::vector<char> alive(n, 1);
        for (int i = 0; i < n; ++i) { members[i] = {S[i]}; cen[i] = Fn[S[i]]; }
        int nclust = n;
        int target = std::min(k, n);
        while (nclust > target) {
            double best = std::numeric_limits<double>::max();
            int ba = -1, bb = -1;
            for (int a = 0; a < n; ++a) {
                if (!alive[a]) continue;
                for (int b = a + 1; b < n; ++b) {
                    if (!alive[b]) continue;
                    double ang = angle_vec(cen[a], cen[b]);
                    if (ang < best) { best = ang; ba = a; bb = b; }
                }
            }
            if (ba < 0) break;
            for (int v : members[bb]) members[ba].push_back(v);
            std::vector<double> nc(m, 0.0);
            for (int v : members[ba])
                for (int j = 0; j < m; ++j) nc[j] += Fn[v][j];
            double inv = 1.0 / static_cast<double>(members[ba].size());
            for (int j = 0; j < m; ++j) nc[j] *= inv;       // (5) the mean
            cen[ba] = nc;
            alive[bb] = 0; --nclust;
        }
        out_clusters.clear(); out_centroids.clear();
        for (int i = 0; i < n; ++i) if (alive[i]) {
            out_clusters.push_back(members[i]);
            out_centroids.push_back(cen[i]);
        }
    }

    int vault_objs_ = 0;

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
    MaOEACCore() = default;

    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_pc(double p)            { pc_ = p; }
    void set_delta(double d)         { delta_ = d; }   // ε
    void set_seed(unsigned s)        { rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        vault_objs_ = vault.objs_n();
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

    void setup_seeded(DataVault<Ind_t>& vault) { vault_objs_ = vault.objs_n(); }

    // ── step: one generation (Algorithm 4) ─────────────────────────────────
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int N = n;
        int m = vault.objs_n();
        vault_objs_ = m;
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dr(0.0, 1.0);
        std::uniform_int_distribution<int> dist_N(0, n - 1);

        // ── Mating (Alg.4 lines 4-17): PCM(P,m) + similarity-mating ────────
        std::vector<int> P_idx(n);
        std::iota(P_idx.begin(), P_idx.end(), 0);
        auto Fn_P  = normalize(vault, P_idx, n);
        auto PCM_P = pcm(P_idx, m, Fn_P);

        int off_base = vault.expand(n);
        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        const double pm = (vault.vars_n() > 0)
            ? 1.0 / static_cast<double>(vault.vars_n()) : 0.0;   // p_m=1/n (Table I)
        int off = 0;
        for (int i = 0; i < m && off < n; ++i) {
            const auto& Ci = PCM_P[i];
            int cz = static_cast<int>(Ci.size());
            for (int j = 0; j < cz && off < n; ++j) {
                int p1, p2;
                if (dr(rng_) < delta_ && cz >= 1) {       // similarity-mating
                    p1 = Ci[std::uniform_int_distribution<int>(0, cz - 1)(rng_)];
                    p2 = Ci[std::uniform_int_distribution<int>(0, cz - 1)(rng_)];
                } else {                                   // global pairing
                    p1 = dist_N(rng_); p2 = dist_N(rng_);
                }
                for (int d = 0; d < vault.vars_n(); ++d) {
                    pv1[d] = vault.get_variable(p1, d);
                    pv2[d] = vault.get_variable(p2, d);
                }
                ops::sbx(pv1, pv2, c1, c2, bounds, eta_c_, pc_, rng_); // take u = c1
                ops::polynomial_mutation(c1, bounds, eta_m_, pm, rng_);
                if (vault.bin_vars_n() > 0) {
                    std::vector<int> bv1(vault.bin_vars_n()), bv2(vault.bin_vars_n()), bc1, bc2;
                    for (int d = 0; d < vault.bin_vars_n(); ++d) {
                        bv1[d] = vault.get_bin_variable(p1, d);
                        bv2[d] = vault.get_bin_variable(p2, d);
                    }
                    ops::binary_crossover(bv1, bv2, bc1, bc2, rng_);
                    ops::bit_flip_mutation(bc1, vault.bin_vars_n(), rng_);
                    vault.set_all_variables(off_base + off, c1, bc1);
                } else {
                    vault.set_variables(off_base + off, c1);
                }
                ++off;
            }
        }
        // if PCM rounding left fewer than n offspring, top up at random
        for (; off < n; ++off) {
            int p1 = dist_N(rng_), p2 = dist_N(rng_);
            for (int d = 0; d < vault.vars_n(); ++d) {
                pv1[d] = vault.get_variable(p1, d);
                pv2[d] = vault.get_variable(p2, d);
            }
            ops::sbx(pv1, pv2, c1, c2, bounds, eta_c_, pc_, rng_);
            ops::polynomial_mutation(c1, bounds, eta_m_, pm, rng_);
            if (vault.bin_vars_n() > 0) {
                std::vector<int> bv1(vault.bin_vars_n()), bv2(vault.bin_vars_n()), bc1, bc2;
                for (int d = 0; d < vault.bin_vars_n(); ++d) {
                    bv1[d] = vault.get_bin_variable(p1, d);
                    bv2[d] = vault.get_bin_variable(p2, d);
                }
                ops::binary_crossover(bv1, bv2, bc1, bc2, rng_);
                ops::bit_flip_mutation(bc1, vault.bin_vars_n(), rng_);
                vault.set_all_variables(off_base + off, c1, bc1);
            } else {
                vault.set_variables(off_base + off, c1);
            }
        }
        vault.sync();

        // ── Environmental_Selection (Algorithm 3) ──────────────────────────
        int pool = n * 2;
        std::vector<int> U(pool);
        std::iota(U.begin(), U.end(), 0);
        auto Fn = normalize(vault, U, pool);                 // (3)
        std::vector<double> conv(pool, 0.0);                 // (6)
        for (int i = 0; i < pool; ++i)
            for (int j = 0; j < m; ++j) conv[i] += Fn[i][j];

        auto PCM_U = pcm(U, m, Fn);                          // m clusters (2k)

        // k_i: the base k = N/m plus the distributed remainder (Sum k_i = N).
        int base = N / m, rem = N % m;
        std::vector<int> kk(m, base);
        for (int i = 0; i < rem; ++i) ++kk[i];

        std::vector<int> survivors;
        survivors.reserve(N);
        std::vector<char> used(pool, 0);
        for (int i = 0; i < m; ++i) {
            int k = kk[i];
            if (k <= 0) continue;
            const auto& Ci = PCM_U[i];
            if (Ci.empty()) continue;

            // NDS over C_i^PCM, accumulating the leading fronts until |S| >= k.
            auto fronts = nds_subset(vault, Ci);
            std::vector<int> Sset;
            for (const auto& fr : fronts) {
                for (int v : fr) Sset.push_back(v);
                if (static_cast<int>(Sset.size()) >= k) break;
            }
            if (Sset.empty()) continue;

            // HCM(S, k) -> k clusters and their centroids.
            std::vector<std::vector<int>>    hc;
            std::vector<std::vector<double>> hcen;
            hcm(Sset, k, Fn, hc, hcen);

            // the cluster C_d nearest to axis i by (4).
            int d = 0; double dmin = std::numeric_limits<double>::max();
            for (int t = 0; t < static_cast<int>(hcen.size()); ++t) {
                double ang = angle_to_axis(hcen[t], i);      // centroid angle to e_i
                if (ang < dmin) { dmin = ang; d = t; }
            }
            // p from C_d, nearest to axis i by (2) — diversity.
            {
                int pbest = -1; double pmin = std::numeric_limits<double>::max();
                for (int v : hc[d]) {
                    if (used[v]) continue;
                    double ang = angle_to_axis(Fn[v], i);
                    if (ang < pmin) { pmin = ang; pbest = v; }
                }
                if (pbest >= 0) { survivors.push_back(pbest); used[pbest] = 1; }
            }
            // from the other clusters, the best (minimum) c(.) — convergence.
            for (int t = 0; t < static_cast<int>(hc.size()); ++t) {
                if (t == d) continue;
                int qbest = -1; double qmin = std::numeric_limits<double>::max();
                for (int v : hc[t]) {
                    if (used[v]) continue;
                    if (conv[v] < qmin) { qmin = conv[v]; qbest = v; }
                }
                if (qbest >= 0) { survivors.push_back(qbest); used[qbest] = 1; }
            }
        }

        // safety: bring the population to exactly N (top up with the best by
        // c(.), truncate the worst).
        if (static_cast<int>(survivors.size()) < N) {
            std::vector<int> rest;
            for (int v = 0; v < pool; ++v) if (!used[v]) rest.push_back(v);
            std::sort(rest.begin(), rest.end(),
                      [&](int a, int b){ return conv[a] < conv[b]; });
            for (int v : rest) {
                if (static_cast<int>(survivors.size()) >= N) break;
                survivors.push_back(v); used[v] = 1;
            }
        }
        if (static_cast<int>(survivors.size()) > N) {
            std::sort(survivors.begin(), survivors.end(),
                      [&](int a, int b){ return conv[a] < conv[b]; });
            survivors.resize(N);
        }

        rearrange(vault, survivors, pool);
    }
};

} // namespace mootation
