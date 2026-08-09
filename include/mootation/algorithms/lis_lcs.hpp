#pragma once
// SPDX-License-Identifier: Apache-2.0
// ╔══════════════════════════════════════════════════════════════════════════╗
// ║ LIS/LCS (CSMOEA) — An evolutionary algorithm with clustering-based        ║
// ║   selection strategies for multi-objective optimization.                  ║
// ║   Shenghao Zhou, Xiaomei Mo, Zidong Wang, Qi Li, Tianxiang Chen,          ║
// ║   Yujun Zheng, Weiguo Sheng,                                              ║
// ║   Information Sciences 624 (2023) 217-234.                                ║
// ║   doi:10.1016/j.ins.2022.12.076                                           ║
// ║                       ║
// ╚══════════════════════════════════════════════════════════════════════════╝
//
// Two-population co-evolution (Algorithm 1), one generation:
//   1. M_DP <- (N−K) random individuals of DP; OffspringPool_1 <- DE/rand/1/bin
//      (vectors from the T-neighbourhood of the weights with probability δ,
//      otherwise from all of DP;
//      repair RandomReset) + PM.
//   2. [K, M_PP] <- LIS(PP ∪ OffspringPool_1, K) (Algorithm 2): NDS filtering
//      -> convergence fitness Fitness(x)=Σ −e^{I_{ε+}(x',x)/0.05} (Eq.2–3,
//      smaller is better) -> k-means into K clusters -> best member of each
//      cluster. K adapts per Eq.4–9: EI = Σ Δf^tch of the offspring divided by
//      the pool size (Eq.4–5, Δf against the ACTUAL parent), normalized
//      (Eq.8),
//      K = max(min([EI_DP'·N], N−3), 3) (Eq.9).
//   3. OffspringPool_2 <- SBX + PM from M_PP (K offspring).
//   4. DP <- decomposition-based environmental selection: every offspring, from
//      both pools, updates at most n_r subproblems by Tchebycheff g^tch
//      (Eq.6–7).
//   5. PP <- LCS(PP ∪ OffspringPool_1 ∪ OffspringPool_2) (Algorithm 3): NDS up
//      to >=N -> k-means into N clusters -> cluster crowding distance (Eq.10)
//      -> best member of each cluster.
//   6. Output (Algorithm 1, line 18) is the CCD strategy (Eq.11–12) over
//      DP ∪ PP: NDS filtering -> k-means into N clusters -> t(p) = size of
//      cluster p (1 -> 3) -> niche radius r = mean distance to the t-th
//      neighbour -> D(p)=1−Π R(p,q) -> iteratively remove the individual with
//      the largest D, recomputing its neighbours.
//
// Paper defaults, §4.1: "The crossover and mutation probabilities are set
//   to be 1 and 1/n, respectively. The distribution indexes of SBX and PM are
//   configured as 20. For the DE operator, the crossover and scaling rates
//   are set to be 1 and 0.5, respectively. The rest parameters, i.e., the
//   size of neighborhood for weight vectors T, the probability to select
//   parents from T neighbors and the maximum number of parent solutions to
//   be updated by each offspring, are configured as 20, 0.9 and 2»;
//   K_0 = 0.5N (Algorithm 1, line 2); κ = 0.05 (Eq.2); t = 3 for singleton
//   clusters in CCD (§3.3, following the recommendation of [23]).
//
// DECLARED DEVIATIONS:
//   - Eq.2–3, k-means and the CCD distances are computed on the min-max
//     normalized objectives of the current set. The paper does not mention
//     normalization, but e^{I/0.05} overflows on raw scales (DTLZ1, DTLZ3).
//   - Alg.1 lines 12–14 («DP ← DP ∪ Pool₁ ∪ Pool₂; environmental selection
//     (DP,W,Z)") are implemented with the canonical MOEA/D-DE scheme: each
//     offspring replaces at most n_r subproblems (the mating scope for Pool_1,
//     all of DP for Pool_2). That is the scheme the §4.1 parameters T/δ/n_r
//     were stated for.
//   - Δf (Eq.5) is clamped from below at zero to guarantee 0 <= EI' <= 1
//     (Eq.8); when EI_PP + EI_DP = 0, and in the first generation, K is left
//     unchanged (the previous generation's EI_PP does not exist yet).
//   - The weight w_i in Eq.5: for DP offspring it is the parent subproblem's
//     weight; for PP offspring it is the weight at the actual parent's position
//     (the PP index, or the DP parent of a Pool_1 individual). The paper does
//     not specify the PP-to-W association.
//   - CCD runs at the end of EVERY step(). This follows the step() framework
//     contract — the vault's active population must be a valid result at any
//     moment — whereas the paper runs it once on termination. DP and PP live in
//     the Core's internal state; the vault is used to evaluate offspring and to
//     present the CCD result.
//   - Eq.10: "C1!=C2!=C3" is read strictly — both axis neighbours come from
//     clusters different from X's cluster and from each other; a boundary gives
//     BIG (the ∞ of NSGA-II).
//   - Weights: das_dennis::generate_auto, with the excess truncated to N (an
//     exact lattice does not exist for every N). If the generator comes back
//     SHORTER than N the tail is padded by CYCLING, which duplicates weight
//     vectors — several subproblems then carry the same direction. This is
//     reachable, not theoretical: at m=2 the largest size any generate_auto
//     strategy can produce is 501, so pop_size >= 502 falls through to the
//     2-vector terminal fallback and 500 of the 502 subproblems are duplicates;
//     k-means: <=20 iterations,
//     random initialization, empty centroids reinitialized; CF/CD ties are
//     broken by lowest index; the ideal point z* is the cumulative minimum over
//     all evaluated individuals (Eq.7 states the minimum over DP).
//   - FE PER step(): (N - K_{t-1}) + K_t, NOT the flat N of Algorithm 1 line 16.
//     This is the paper's own arithmetic, not a port artefact: line 4 sizes
//     M_DP with the OLD K, line 8 then replaces K, and line 10 sizes M_PP with
//     the NEW one, so line 16's "ev <- ev + N" only holds while K is stationary.
//     Eq.9 clamps K to [3, N-3], so a single generation can in principle cost
//     anywhere in [6, 2N-6]; in practice K moves by a few units per generation
//     and the cost stays within a few percent of N. A caller budgeting by
//     max_ev therefore gets the paper's budget, not a fixed generation count.
//   - dp_update replaces on a STRICT improvement (g^tch(y) < g^tch(x^j)); the
//     paper gives no tie rule for line 14. MOEA/D-DE's "<=" would let equal
//     scalarizations propagate one copy of y over up to n_r subproblems.
//   - setup() initializes both populations at random (2N evaluations, as in the
//     paper); setup_seeded() copies the seeded population into both DP and PP.
//
// EXTENSIONS BEYOND THE PAPER (off by default / neutral):
//   - constraint_mode FEASIBILITY: CV dominance inside NDS (LIS/LCS/CCD); the
//     decomposition update of DP ignores constraints.
//   - Mixed real+binary genome: uniform crossover + bit-flip.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "../constraint_mode.hpp"
#include "../das_dennis.hpp"
#include "../data_vault.hpp"
#include "../warn.hpp"
#include "../individuals.hpp"
#include "../operators/binary_crossover.hpp"
#include "../operators/bit_flip.hpp"
#include "../operators/de_mutation.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

struct LISLCS_Individual : public Based_Individual {
    int rank = 0;
};

template <typename Ind_t>
class LISLCSCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    static constexpr double KAPPA_ = 0.05;   // Eq.2
    static constexpr double BIG_   = 1e9;    // boundary CD (Eq.10)

    // §4.1 (see the header)
    double eta_c_ = 20.0;
    double eta_m_ = 20.0;
    double pc_    = 1.0;
    double de_F_  = 0.5;
    double de_CR_ = 1.0;
    int    T_nb_  = 20;
    double delta_ = 0.9;
    int    nr_    = 2;

    int          K_ = 0;                       // adaptive K (Eq.4–9)
    std::mt19937 rng_{std::random_device{}()};

    // Internal co-evolution individual (DP / PP / offspring). The vault holds
    // only the presented (CCD) population; both co-populations live here.
    struct Sol {
        std::vector<double> x;        // real variables
        std::vector<int>    b;        // binary variables (extension)
        std::vector<double> f;        // objectives
        std::vector<double> g;        // limits
        double cv       = 0.0;        // constraint violation
        int    widx     = 0;          // associated weight (Eq.5)
        int    parent   = -1;         // DP parent (OffspringPool_1 only)
        bool   nb_scope = false;      // mated within the T-neighbourhood (Pool_1)
    };

    std::vector<Sol>                 dp_, pp_;   // co-populations (Algorithm 1)
    std::vector<std::vector<double>> W_;         // weight vectors
    std::vector<std::vector<int>>    B_;         // T-neighbourhoods of the weights
    std::vector<double>              z_;         // ideal point (Eq.7)
    double ei_pp_raw_  = 0.0;   // Σ max(0,Δf) of last generation's PP offspring
    bool   have_ei_pp_ = false;

    // ── Dominance / NDS ────────────────────────────────────────────────────
    bool dominates(const Sol& a, const Sol& b) const {
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            bool af = (a.cv <= 0.0), bf = (b.cv <= 0.0);
            if (af && !bf) return true;
            if (!af && bf) return false;
            if (!af && !bf) return a.cv < b.cv;
        }
        bool strict = false;
        for (std::size_t k = 0; k < a.f.size(); ++k) {
            if (a.f[k] > b.f[k]) return false;
            if (a.f[k] < b.f[k]) strict = true;
        }
        return strict;
    }

    // Fast non-dominated sorting [8] -> fronts of local indices.
    std::vector<std::vector<int>> nds(const std::vector<const Sol*>& P) const {
        int n = static_cast<int>(P.size());
        std::vector<std::vector<int>> S(n);
        std::vector<int> ndom(n, 0), cur;
        for (int a = 0; a < n; ++a) {
            for (int b = 0; b < n; ++b) {
                if (a == b) continue;
                if (dominates(*P[a], *P[b]))      S[a].push_back(b);
                else if (dominates(*P[b], *P[a])) ++ndom[a];
            }
            if (ndom[a] == 0) cur.push_back(a);
        }
        std::vector<std::vector<int>> fronts;
        while (!cur.empty()) {
            fronts.push_back(cur);
            std::vector<int> next;
            for (int a : cur)
                for (int b : S[a])
                    if (--ndom[b] == 0) next.push_back(b);
            cur = std::move(next);
        }
        return fronts;
    }

    // NDS filtering of solutions far from the PF (Alg.2/3, lines 2–7):
    // accumulate fronts until |Q| >= need.
    std::vector<int> nds_prune(const std::vector<const Sol*>& P, int need) const {
        auto fronts = nds(P);
        std::vector<int> Q;
        for (const auto& fr : fronts) {
            for (int v : fr) Q.push_back(v);
            if (static_cast<int>(Q.size()) >= need) break;
        }
        return Q;
    }

    // ── Normalization (min-max over the set; see the declared deviations) ──
    std::vector<std::vector<double>>
    normalise(const std::vector<const Sol*>& P) const {
        int n = static_cast<int>(P.size());
        int m = (n > 0) ? static_cast<int>(P[0]->f.size()) : 0;
        std::vector<double> fmin(m,  std::numeric_limits<double>::max());
        std::vector<double> fmax(m, -std::numeric_limits<double>::max());
        for (const Sol* s : P)
            for (int j = 0; j < m; ++j) {
                fmin[j] = std::min(fmin[j], s->f[j]);
                fmax[j] = std::max(fmax[j], s->f[j]);
            }
        std::vector<std::vector<double>> Fn(n, std::vector<double>(m, 0.0));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j) {
                double range = fmax[j] - fmin[j];
                Fn[i][j] = (range > 1e-14) ? (P[i]->f[j] - fmin[j]) / range : 0.0;
            }
        return Fn;
    }

    // Eq.3: I_{ε+}(a,b) = max_j (a_j − b_j) — the smallest ε for which a−ε
    // weakly dominates b.
    static double eps_plus(const std::vector<double>& a,
                           const std::vector<double>& b) {
        double w = -std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < a.size(); ++j)
            w = std::max(w, a[j] - b[j]);
        return w;
    }

    // Eq.2: Fitness(x) = Σ_{x'≠x} −e^{I_{ε+}(x',x)/0.05}; smaller is better.
    static std::vector<double>
    conv_fitness(const std::vector<std::vector<double>>& Fn) {
        int n = static_cast<int>(Fn.size());
        std::vector<double> cf(n, 0.0);
        for (int i = 0; i < n; ++i) {
            double s = 0.0;
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                s += -std::exp(eps_plus(Fn[j], Fn[i]) / KAPPA_);
            }
            cf[i] = s;
        }
        return cf;
    }

    // ── k-means (normalized objective space) ───────────────────────────────
    std::vector<int> kmeans(const std::vector<std::vector<double>>& X, int K) {
        int n = static_cast<int>(X.size());
        std::vector<int> label(n, 0);
        if (n == 0 || K <= 1) return label;
        if (K >= n) { std::iota(label.begin(), label.end(), 0); return label; }
        int m = static_cast<int>(X[0].size());
        std::vector<int> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng_);
        std::vector<std::vector<double>> cen(K);
        for (int c = 0; c < K; ++c) cen[c] = X[perm[c]];
        for (int iter = 0; iter < 20; ++iter) {
            bool changed = false;
            for (int i = 0; i < n; ++i) {
                int bc = 0; double bd = std::numeric_limits<double>::max();
                for (int c = 0; c < K; ++c) {
                    double d = 0.0;
                    for (int j = 0; j < m; ++j) {
                        double t = X[i][j] - cen[c][j];
                        d += t * t;
                    }
                    if (d < bd) { bd = d; bc = c; }
                }
                if (label[i] != bc) { label[i] = bc; changed = true; }
            }
            std::vector<std::vector<double>> sum(K, std::vector<double>(m, 0.0));
            std::vector<int> cnt(K, 0);
            for (int i = 0; i < n; ++i) {
                ++cnt[label[i]];
                for (int j = 0; j < m; ++j) sum[label[i]][j] += X[i][j];
            }
            for (int c = 0; c < K; ++c) {
                if (cnt[c] > 0)
                    for (int j = 0; j < m; ++j) cen[c][j] = sum[c][j] / cnt[c];
                else
                    cen[c] = X[std::uniform_int_distribution<int>(0, n - 1)(rng_)];
            }
            if (!changed && iter > 0) break;
        }
        return label;
    }

    // ── Eq.10: cluster crowding distance (LCS); larger is better ──────────
    std::vector<double> crowding(const std::vector<const Sol*>& Q,
                                 const std::vector<int>& label) const {
        int n = static_cast<int>(Q.size());
        int m = (n > 0) ? static_cast<int>(Q[0]->f.size()) : 0;
        std::vector<double> cd(n, 0.0);
        std::vector<double> fmin(m,  std::numeric_limits<double>::max());
        std::vector<double> fmax(m, -std::numeric_limits<double>::max());
        for (const Sol* s : Q)
            for (int j = 0; j < m; ++j) {
                fmin[j] = std::min(fmin[j], s->f[j]);
                fmax[j] = std::max(fmax[j], s->f[j]);
            }
        for (int j = 0; j < m; ++j) {
            double range = fmax[j] - fmin[j];
            if (range < 1e-14) continue;
            std::vector<int> ord(n);
            std::iota(ord.begin(), ord.end(), 0);
            std::sort(ord.begin(), ord.end(), [&](int a, int b) {
                return Q[a]->f[j] < Q[b]->f[j];
            });
            for (int p = 0; p < n; ++p) {
                int x = ord[p];
                // X_{n−1}: nearest below, from a different cluster (C1 != C2)
                int lo = -1;
                for (int q = p - 1; q >= 0; --q)
                    if (label[ord[q]] != label[x]) { lo = ord[q]; break; }
                // X_{n+1}: nearest above, from a cluster different from those
                // of X_n and X_{n−1} (C1 != C2 != C3)
                int up = -1;
                if (lo >= 0)
                    for (int q = p + 1; q < n; ++q) {
                        int l = label[ord[q]];
                        if (l != label[x] && l != label[lo]) { up = ord[q]; break; }
                    }
                if (lo < 0 || up < 0) { cd[x] += BIG_; continue; }   // boundary
                cd[x] += (Q[up]->f[j] - Q[lo]->f[j]) / range;
            }
        }
        return cd;
    }

    // ── Eq.6: g^tch(x|w,z*) = max_j w_j·|f_j(x) − z*_j| ────────────────────
    double gtch(const std::vector<double>& f, const std::vector<double>& w) const {
        double g = -std::numeric_limits<double>::max();
        for (std::size_t j = 0; j < f.size(); ++j) {
            double wj = (w[j] > 1e-6) ? w[j] : 1e-6;
            g = std::max(g, wj * std::abs(f[j] - z_[j]));
        }
        return g;
    }

    void update_z(const std::vector<double>& f) {
        for (std::size_t j = 0; j < f.size(); ++j) z_[j] = std::min(z_[j], f[j]);
    }

    // Eq.9 clamp: K = max(min([·], N−3), 3).
    int clamp_K(long v, int N) const {
        long kmax = std::max(3, N - 3);
        long k = std::max(3L, std::min(v, kmax));
        if (k > N) k = N;   // degenerately small N
        return static_cast<int>(k);
    }

    // ── Decomposition-based environmental selection of DP (Alg.1, 12–14) ───
    // An offspring replaces at most n_r subproblems (by g^tch) within its
    // mating scope: the T-neighbourhood for Pool_1 when nb_scope is set,
    // otherwise all of DP. See the declared deviations.
    void dp_update(const Sol& y) {
        std::vector<int> scope;
        if (y.parent >= 0 && y.nb_scope) scope = B_[y.parent];
        else {
            scope.resize(dp_.size());
            std::iota(scope.begin(), scope.end(), 0);
        }
        std::shuffle(scope.begin(), scope.end(), rng_);
        int c = 0;
        for (int j : scope) {
            if (gtch(y.f, W_[j]) < gtch(dp_[j].f, W_[j])) {
                Sol t = y;
                t.widx = j; t.parent = -1; t.nb_scope = false;
                dp_[j] = std::move(t);
                if (++c >= nr_) break;
            }
        }
    }

    // Three distinct DE vectors from the scope, != self where possible.
    void pick3(const std::vector<int>& scope_in, int self,
               int& a, int& b, int& c) {
        std::vector<int> fallback;
        const std::vector<int>* scope = &scope_in;
        if (static_cast<int>(scope_in.size()) < 4) {
            fallback.resize(dp_.size());
            std::iota(fallback.begin(), fallback.end(), 0);
            scope = &fallback;
        }
        std::uniform_int_distribution<int> d(0, static_cast<int>(scope->size()) - 1);
        if (static_cast<int>(scope->size()) < 4) {           // tiny DP
            a = (*scope)[d(rng_)]; b = (*scope)[d(rng_)]; c = (*scope)[d(rng_)];
            return;
        }
        do { a = (*scope)[d(rng_)]; } while (a == self);
        do { b = (*scope)[d(rng_)]; } while (b == self || b == a);
        do { c = (*scope)[d(rng_)]; } while (c == self || c == a || c == b);
    }

    // ── Evaluate a batch of offspring via the vault scratch slots [0,cnt) ──
    // (cnt <= N always: |Pool_1| = N−K <= N−3, |Pool_2| = K <= N−3.)
    void evaluate(DataVault<Ind_t>& vault, std::vector<Sol>& sols) {
        int cnt = static_cast<int>(sols.size());
        if (cnt == 0) return;
        for (int i = 0; i < cnt; ++i) {
            if (vault.bin_vars_n() > 0) vault.set_all_variables(i, sols[i].x, sols[i].b);
            else                        vault.set_variables(i, sols[i].x);
        }
        vault.sync();
        for (int i = 0; i < cnt; ++i) {
            sols[i].x  = vault.variables_of(i);     // after bound clamping
            sols[i].f  = vault.objectives_of(i);
            sols[i].g  = vault.limits_of(i);
            sols[i].cv = vault.get_cv(i);
            update_z(sols[i].f);
        }
    }

    // ── CCD (Eq.11–12, §3.3): extract N individuals from P = DP ∪ PP ───────
    std::vector<int> ccd_select(const std::vector<const Sol*>& P, int N) {
        std::vector<int> Q = nds_prune(P, N);
        int nQ = static_cast<int>(Q.size());
        if (nQ <= N) return Q;

        std::vector<const Sol*> Qs;
        Qs.reserve(nQ);
        for (int v : Q) Qs.push_back(P[v]);
        auto Fn = normalise(Qs);

        // k-means into N clusters -> t(p) = size of cluster p (1 -> 3, [23])
        auto label = kmeans(Fn, N);
        std::vector<int> csz(std::max(N, nQ), 0);
        for (int l : label) ++csz[l];
        std::vector<int> t(nQ);
        for (int i = 0; i < nQ; ++i)
            t[i] = (csz[label[i]] <= 1) ? 3 : csz[label[i]];

        // distances and niche radius r = mean d(p, t(p)-th neighbour)
        std::vector<std::vector<double>> d(nQ, std::vector<double>(nQ, 0.0));
        int m = static_cast<int>(Fn[0].size());
        for (int i = 0; i < nQ; ++i)
            for (int j = i + 1; j < nQ; ++j) {
                double s = 0.0;
                for (int k = 0; k < m; ++k) {
                    double dx = Fn[i][k] - Fn[j][k];
                    s += dx * dx;
                }
                d[i][j] = d[j][i] = std::sqrt(s);
            }
        double r = 0.0;
        std::vector<double> row;
        row.reserve(nQ - 1);
        for (int i = 0; i < nQ; ++i) {
            row.clear();
            for (int j = 0; j < nQ; ++j) if (j != i) row.push_back(d[i][j]);
            int tt = std::min(t[i], nQ - 1);
            if (tt < 1) tt = 1;
            std::nth_element(row.begin(), row.begin() + (tt - 1), row.end());
            r += row[tt - 1];
        }
        r /= nQ;

        // Eq.11–12: D(p) = 1 − Π_{q≠p, d≤r} d(p,q)/r
        std::vector<char> alive(nQ, 1);
        int na = nQ;
        auto computeD = [&](int p) -> double {
            if (r < 1e-14) return 1.0;       // degenerate: all points coincide
            double prod = 1.0;
            for (int q = 0; q < nQ; ++q) {
                if (q == p || !alive[q]) continue;
                if (d[p][q] <= r) prod *= d[p][q] / r;
            }
            return 1.0 - prod;
        };
        std::vector<double> D(nQ);
        for (int i = 0; i < nQ; ++i) D[i] = computeD(i);

        // iteratively remove the maximum D (ties broken at random) and
        // recompute D for the neighbours inside the removed point's niche (§3.3)
        while (na > N) {
            double best = -std::numeric_limits<double>::max();
            for (int i = 0; i < nQ; ++i)
                if (alive[i] && D[i] > best) best = D[i];
            std::vector<int> cands;
            for (int i = 0; i < nQ; ++i)
                if (alive[i] && D[i] >= best - 1e-12) cands.push_back(i);
            int rem = cands[std::uniform_int_distribution<int>(
                0, static_cast<int>(cands.size()) - 1)(rng_)];
            alive[rem] = 0;
            --na;
            if (r >= 1e-14)
                for (int q = 0; q < nQ; ++q)
                    if (alive[q] && d[rem][q] <= r) D[q] = computeD(q);
        }
        std::vector<int> keep;
        keep.reserve(N);
        for (int i = 0; i < nQ; ++i) if (alive[i]) keep.push_back(Q[i]);
        return keep;
    }

    // ── Initialize the weights W and the neighbourhoods B (T = 20) ─────────
    void init_weights(int N, int m) {
        W_ = das_dennis::generate_auto(m, N);
        if (static_cast<int>(W_.size()) > N) W_.resize(N);
        if (W_.empty()) W_.assign(1, std::vector<double>(m, 1.0 / m));
        // Pad a short lattice by cycling. This DUPLICATES directions — see the
        // weights note in the header for when it is reachable and what it costs.
        std::size_t base = W_.size();
        if (static_cast<int>(base) < N)
            warn_lazy([&]{ return "lis_lcs: the weight lattice returned only " +
                           std::to_string(base) + " vectors for N=" +
                           std::to_string(N) + " (m=" + std::to_string(m) +
                           "); the remaining subproblems will carry DUPLICATE "
                           "direction vectors"; });
        while (static_cast<int>(W_.size()) < N)
            W_.push_back(W_[W_.size() % base]);

        int T = std::min(T_nb_, N);
        B_.assign(N, {});
        std::vector<std::pair<double, int>> dist(N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double s = 0.0;
                for (int k = 0; k < m; ++k) {
                    double dx = W_[i][k] - W_[j][k];
                    s += dx * dx;
                }
                dist[j] = {s, j};
            }
            std::partial_sort(dist.begin(), dist.begin() + T, dist.end());
            B_[i].reserve(T);
            for (int t = 0; t < T; ++t) B_[i].push_back(dist[t].second);
        }
    }

    std::vector<Sol> random_population(DataVault<Ind_t>& vault, int N) {
        const auto& bounds = vault.get_bounds();
        std::uniform_real_distribution<double> dr(0.0, 1.0);
        std::uniform_int_distribution<int>     db(0, 1);
        std::vector<Sol> P(N);
        for (int i = 0; i < N; ++i) {
            P[i].x.resize(vault.vars_n());
            for (int j = 0; j < vault.vars_n(); ++j) {
                double lo = bounds[j].first.value_or(0.0);
                double hi = bounds[j].second.value_or(1.0);
                P[i].x[j] = lo + dr(rng_) * (hi - lo);
            }
            P[i].b.resize(vault.bin_vars_n());
            for (int j = 0; j < vault.bin_vars_n(); ++j) P[i].b[j] = db(rng_);
            P[i].widx = i;
        }
        return P;
    }

public:
    LISLCSCore() = default;

    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_seed(unsigned s)        { rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault) {
        const int N = vault.pop_size(), m = vault.objs_n();
        init_weights(N, m);
        K_ = clamp_K(std::lround(0.5 * N), N);     // Algorithm 1 line 2
        z_.assign(m, std::numeric_limits<double>::max());
        have_ei_pp_ = false;
        ei_pp_raw_  = 0.0;

        // two populations of N random individuals (Alg.1 line 1; 2N evaluations)
        pp_ = random_population(vault, N);
        evaluate(vault, pp_);
        dp_ = random_population(vault, N);
        evaluate(vault, dp_);

        // present PP in the vault (CCD is not informative on a random start)
        for (int i = 0; i < N; ++i)
            vault.seed_individual(i, pp_[i].x, pp_[i].f, pp_[i].b, pp_[i].g);
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        const int N = vault.pop_size(), m = vault.objs_n();
        if (W_.empty()) init_weights(N, m);
        if (K_ <= 0) K_ = clamp_K(std::lround(0.5 * N), N);
        z_.assign(m, std::numeric_limits<double>::max());
        have_ei_pp_ = false;
        ei_pp_raw_  = 0.0;

        pp_.assign(N, Sol{});
        for (int i = 0; i < N; ++i) {
            pp_[i].x    = vault.variables_of(i);
            pp_[i].b    = vault.binary_variables_of(i);
            pp_[i].f    = vault.objectives_of(i);
            pp_[i].g    = vault.limits_of(i);
            pp_[i].cv   = vault.get_cv(i);
            pp_[i].widx = i;
            update_z(pp_[i].f);
        }
        dp_ = pp_;   // resume-path deviation: one seeded population feeds both
    }

    // ── step: one generation of Algorithm 1 (N evaluations) ────────────────
    void step(DataVault<Ind_t>& vault) {
        const int N  = vault.pop_size();
        const int nv = vault.vars_n();
        const int nb = vault.bin_vars_n();
        const auto& bounds = vault.get_bounds();
        const double pm = (nv > 0) ? 1.0 / nv : 0.0;   // §4.1: pm = 1/n
        std::uniform_real_distribution<double> U01(0.0, 1.0);

        const int K_l = K_;   // K of the previous generation (Eq.4: K_l)

        // ── 1) M_DP: (N−K) random DP individuals; OffspringPool_1 = DE + PM ─
        int n1 = std::max(0, N - K_l);
        std::vector<int> permDP(dp_.size());
        std::iota(permDP.begin(), permDP.end(), 0);
        std::shuffle(permDP.begin(), permDP.end(), rng_);
        std::vector<int> allDP = permDP;   // full scope (same contents)

        std::vector<Sol> off1;
        off1.reserve(n1);
        for (int t = 0; t < n1; ++t) {
            int i = permDP[t % permDP.size()];
            bool nb_scope = (U01(rng_) < delta_);       // §4.1: δ = 0.9
            const std::vector<int>& scope = nb_scope ? B_[i] : allDP;
            int a, b, c;
            pick3(scope, i, a, b, c);
            Sol ch;
            ops::de_rand_1_bin(dp_[a].x, dp_[b].x, dp_[c].x, dp_[i].x, ch.x,
                               bounds, de_F_, de_CR_,
                               ops::DERepair::RandomReset, rng_);
            ops::polynomial_mutation(ch.x, bounds, eta_m_, pm, rng_);
            if (nb > 0) {                                // extension: binary
                std::vector<int> bc1, bc2;
                ops::binary_crossover(dp_[i].b, dp_[a].b, bc1, bc2, rng_);
                ops::bit_flip_mutation(bc1, nb, rng_);
                ch.b = std::move(bc1);
            }
            ch.parent   = i;          // the actual parent
            ch.nb_scope = nb_scope;
            ch.widx     = i;          // weight of the parent subproblem (Eq.5)
            off1.push_back(std::move(ch));
        }
        evaluate(vault, off1);

        // Δf^tch of the DP offspring against their ACTUAL parents (Eq.5)
        double ei_dp_raw = 0.0;
        for (const Sol& s : off1) {
            double df = gtch(dp_[s.parent].f, W_[s.widx]) - gtch(s.f, W_[s.widx]);
            if (df > 0.0) ei_dp_raw += df;               // clamped at 0 (see header)
        }

        // ── 2) Adaptive K (Eq.4, 8, 9) ─────────────────────────────────────
        if (have_ei_pp_ && K_l > 0 && N - K_l > 0) {
            double EIpp = ei_pp_raw_ / K_l;              // Eq.4
            double EIdp = ei_dp_raw / (N - K_l);         // Eq.4
            double s = EIpp + EIdp;
            if (s > 1e-14) {
                double EIdpN = EIdp / s;                 // Eq.8
                K_ = clamp_K(std::lround(EIdpN * N), N); // Eq.9
            }
        }

        // ── 3) LIS (Algorithm 2) over PP ∪ OffspringPool_1 ────────────────
        for (int i = 0; i < static_cast<int>(pp_.size()); ++i) pp_[i].widx = i;
        std::vector<const Sol*> lisP;
        lisP.reserve(pp_.size() + off1.size());
        for (const Sol& s : pp_)  lisP.push_back(&s);
        for (const Sol& s : off1) lisP.push_back(&s);
        std::vector<int> Q1 = nds_prune(lisP, N);        // lines 2–7
        std::vector<const Sol*> Qs1;
        Qs1.reserve(Q1.size());
        for (int v : Q1) Qs1.push_back(lisP[v]);
        auto Fn1 = normalise(Qs1);
        auto cf  = conv_fitness(Fn1);                    // line 8 (Eq.2)
        int KK = std::min(K_, static_cast<int>(Qs1.size()));
        auto lab1 = kmeans(Fn1, KK);                     // line 9
        std::vector<int>    bestI(KK, -1);
        std::vector<double> bestCF(KK, std::numeric_limits<double>::max());
        for (int i = 0; i < static_cast<int>(Qs1.size()); ++i) {
            int c = lab1[i];
            if (c < KK && cf[i] < bestCF[c]) { bestCF[c] = cf[i]; bestI[c] = i; }
        }
        std::vector<const Sol*> M;                       // lines 10–13
        M.reserve(KK);
        for (int c = 0; c < KK; ++c) if (bestI[c] >= 0) M.push_back(Qs1[bestI[c]]);
        if (M.empty()) M = Qs1;                          // guard

        // ── 4) OffspringPool_2 = SBX + PM from M_PP (K offspring) ──────────
        int n2 = K_;
        std::vector<Sol> off2;
        std::vector<const Sol*> par2;                    // the actual parents
        off2.reserve(n2);
        par2.reserve(n2);
        std::uniform_int_distribution<int> dm(0, static_cast<int>(M.size()) - 1);
        std::vector<double> c1, c2;
        for (int i = 0; i < n2; i += 2) {
            const Sol* p1 = M[dm(rng_)];
            const Sol* p2 = M[dm(rng_)];
            ops::sbx(p1->x, p2->x, c1, c2, bounds, eta_c_, pc_, rng_);
            ops::polynomial_mutation(c1, bounds, eta_m_, pm, rng_);
            ops::polynomial_mutation(c2, bounds, eta_m_, pm, rng_);
            std::vector<int> bc1, bc2;
            if (nb > 0) {
                ops::binary_crossover(p1->b, p2->b, bc1, bc2, rng_);
                ops::bit_flip_mutation(bc1, nb, rng_);
                if (i + 1 < n2) ops::bit_flip_mutation(bc2, nb, rng_);
            }
            Sol s1;
            s1.x = c1; s1.b = std::move(bc1); s1.widx = p1->widx;
            off2.push_back(std::move(s1));
            par2.push_back(p1);
            if (i + 1 < n2) {
                Sol s2;
                s2.x = c2; s2.b = std::move(bc2); s2.widx = p2->widx;
                off2.push_back(std::move(s2));
                par2.push_back(p2);
            }
        }
        evaluate(vault, off2);

        // Δf of the PP offspring, feeding next generation's EI_PP (Eq.4–5)
        ei_pp_raw_ = 0.0;
        for (std::size_t i = 0; i < off2.size(); ++i) {
            double df = gtch(par2[i]->f, W_[off2[i].widx])
                      - gtch(off2[i].f,  W_[off2[i].widx]);
            if (df > 0.0) ei_pp_raw_ += df;
        }
        have_ei_pp_ = true;

        // ── 5) DP: decomposition environmental selection (lines 12, 14) ───
        for (const Sol& s : off1) dp_update(s);
        for (const Sol& s : off2) dp_update(s);          // scope = all of DP

        // ── 6) PP: LCS (Algorithm 3; lines 13, 15) ────────────────────────
        std::vector<const Sol*> lcsP;
        lcsP.reserve(pp_.size() + off1.size() + off2.size());
        for (const Sol& s : pp_)  lcsP.push_back(&s);
        for (const Sol& s : off1) lcsP.push_back(&s);
        for (const Sol& s : off2) lcsP.push_back(&s);
        std::vector<int> Q2 = nds_prune(lcsP, N);        // lines 2–7
        std::vector<Sol> newPP;
        newPP.reserve(N);
        if (static_cast<int>(Q2.size()) <= N) {          // line 8 (|Q| == N)
            for (int v : Q2) newPP.push_back(*lcsP[v]);
        } else {
            std::vector<const Sol*> Qs2;
            Qs2.reserve(Q2.size());
            for (int v : Q2) Qs2.push_back(lcsP[v]);
            auto Fn2  = normalise(Qs2);
            auto lab2 = kmeans(Fn2, N);                  // line 9
            auto cd   = crowding(Qs2, lab2);             // line 10 (Eq.10)
            std::vector<int>    bI(N, -1);
            std::vector<double> bCD(N, -std::numeric_limits<double>::max());
            for (int i = 0; i < static_cast<int>(Qs2.size()); ++i) {
                int c = lab2[i];
                if (c < N && cd[i] > bCD[c]) { bCD[c] = cd[i]; bI[c] = i; }
            }
            std::vector<char> taken(Qs2.size(), 0);
            for (int c = 0; c < N; ++c)                  // lines 12–15
                if (bI[c] >= 0 && !taken[bI[c]]) {
                    newPP.push_back(*Qs2[bI[c]]);
                    taken[bI[c]] = 1;
                }
            if (static_cast<int>(newPP.size()) < N) {    // guard: empty clusters
                std::vector<int> rest;
                for (int i = 0; i < static_cast<int>(Qs2.size()); ++i)
                    if (!taken[i]) rest.push_back(i);
                std::sort(rest.begin(), rest.end(),
                          [&](int a, int b) { return cd[a] > cd[b]; });
                for (int ri : rest) {
                    if (static_cast<int>(newPP.size()) >= N) break;
                    newPP.push_back(*Qs2[ri]);
                }
            }
        }
        for (Sol& s : newPP) { s.parent = -1; s.nb_scope = false; }
        pp_ = std::move(newPP);

        // ── 7) Output: CCD (Eq.11–12) over DP ∪ PP -> presented in the vault ─
        std::vector<const Sol*> all;
        all.reserve(dp_.size() + pp_.size());
        for (const Sol& s : dp_) all.push_back(&s);
        for (const Sol& s : pp_) all.push_back(&s);
        std::vector<int> keep = ccd_select(all, N);
        if (static_cast<int>(keep.size()) > N) keep.resize(N);
        if (static_cast<int>(keep.size()) < N) {         // guard (not expected)
            std::vector<char> in(all.size(), 0);
            for (int v : keep) in[v] = 1;
            for (int v = 0; v < static_cast<int>(all.size()); ++v) {
                if (static_cast<int>(keep.size()) >= N) break;
                if (!in[v]) { keep.push_back(v); in[v] = 1; }
            }
        }
        for (int i = 0; i < N; ++i) {
            const Sol& s = *all[keep[i]];
            vault.seed_individual(i, s.x, s.f, s.b, s.g);
        }
    }
};

} // namespace mootation
