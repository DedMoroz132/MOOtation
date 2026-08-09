#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// HCCA (HCCA) — A Multi-Objective Evolutionary Algorithm With Hierarchical
//   Clustering-Based Selection.
// S. Zhou, Z. Chen, Q. Li, M. Gu, Z. Bao, W. He, W. Sheng — IEEE Access, 2023.
// doi:10.1109/ACCESS.2023.3234226
//
// MECHANISM (Algorithm 1). Two populations of N individuals each: DP
//   (decomposition) and PP (Pareto). Per generation:
//   - M <- N−K random individuals from DP; DE/rand/1 + PM on each produces the
//     offspring pool Q1 (lines 4-8).
//   - [K,Q2] <- local area selection(PP u Q1, K) (Alg.2): mating driven by
//     hierarchical clustering plus the convergence fitness I_eps+ (Eq.2);
//     K adapts to the fitness improvement FI'_PP (Eq.3-8).
//   - PP ← PP∪Q1∪Q2; DP ← DP∪Q1∪Q2.
//   - [DP] ← environment selection(DP,W,Z): MOEA/D-Tchebycheff (Eq.5-6).
//   - PP <- local coverage selection(PP) (Alg.3): hierarchical clustering in
//     the normalized (Eq.10) objective space; the per-cluster survivor count
//     follows the coverage area (Eq.11-12); within a cluster, the best by
//     crowding
//     distance (Eq.9).
//   - Finally: P <- crowding degree strategy(PP,DP) [38] — the niche-product
//     crowding degree (BCE, Li/Yang/Liu TEVC 2016): NDS filtering of PP u DP up
//     to >=N -> k-means into N clusters -> t(p) = cluster size (singleton -> 3)
//     -> r = mean distance to the t-th neighbour -> D(p) = 1 − Π R(p,q) ->
//     iteratively remove the max-D point, recomputing its niche (see HCCA-5).
//
// CLUSTERING: agglomerative, average-linkage (mean pairwise Euclidean distance
//   in the normalized objective space); the nearest pair of clusters is merged
//   while their count exceeds the target.
//   LCS additionally performs "cluster recombination": a singleton cluster is
//   merged with its nearest neighbour if that neighbour is also a singleton.
//
// PAPER DEFAULTS (Table 2, HCCA row): P_c=1, P_m=1/n, η_c=η_m=20, δ=0.9,
//   n_r=2, CR=1, F=0.5. Table 2 gives NO neighbourhood size T for HCCA.
//   Eq.8 fixes the clamp: min(K)=3, max(K)=N−3.
// NOT FROM TABLE 2:
//   N is per problem in Table 1 (200 for the 2-objective WFG set, 500 for the
//     3-objective DTLZ set, 300/600 for UF) — the library takes it from
//     pop_size instead, and then REVISES it to the Das–Dennis weight count,
//     which the paper does not do (see HCCA-8).
//   K0 = 0.5N is not a stated default either. §5 sweeps the initial K over
//     {0, 0.25N, 0.5N, 0.75N, N} and reports that HCCA is robust to the
//     choice; 0.5N is the midpoint of that sweep, adopted here.
//   T (neighbourhood) = max(2, N/10), a MOEA/D convention, since Table 2
//     omits it for HCCA.
//
// DECLARED DEVIATIONS:
//   HCCA-1 (MINOR). The hierarchical clustering is AGGLOMERATIVE with
//     AVERAGE-LINKAGE (mean pairwise distance). The paper says "hierarchical
//     clustering" without naming the linkage; the EMyO/C description in §II
//     merges the pair with the minimum SUM of pairwise distances, and
//     average-linkage is the closest scale-stable analogue. The header
//     previously claimed single-linkage while the code always used average —
//     the header was wrong, not the code.
//   HCCA-2 (MINOR). DP environmental selection is the classical
//     MOEA/D-Tchebycheff scheme (Eq.5-6, delta=0.9, n_r=2): an offspring
//     updates the neighbouring subproblems. Weights W = Das–Dennis; z* is the
//     running ideal over DP (Eq.6).
//   HCCA-3 (MINOR). FI (Eq.3-7) is computed from the actual Δf of the current
//     generation's PP/DP offspring; in the first generation, with no history,
//     K = K0. Eq.4's subproblem index i is undefined for offspring not bound to
//     a subproblem: DE children inherit their DP parent's subproblem, while
//     SBX/PP children are scored against the subproblem their PARENT associates
//     to. Δf uses the recorded parent objectives, not a re-evaluation.
//   HCCA-9 (MINOR, resolved). Alg.1 line 12's DP update is the MOEA/D
//     scheme, so an offspring updates the neighbourhood of the subproblem it
//     was generated for. DE children carry that index (Sol::sp, set at
//     breeding); SBX/PP children are not bred against any subproblem, so for
//     them — and only them — a random subproblem is drawn. An earlier version
//     drew a random subproblem for every offspring, which discarded the index
//     it had just computed and turned the decomposition update into an
//     undirected one.
//   HCCA-8 (MINOR). N is revised to the Das–Dennis weight count, so the
//     effective population can differ from pop_size when no exact lattice of
//     that size exists (Path-A, as in nsga3.hpp). K0 and every N-derived
//     quantity follow the revised N.
//   HCCA-4 (MINOR). LCS Num(C) (Eq.12): floor of the area proportion, at least
//     1 per non-empty cluster; if the sum of Num falls short of the target
//     size, the remainder is filled with the best by crowding (deterministic
//     tie-break).
//   HCCA-5. FIX 2026-07-07:
//     the final "crowding degree strategy" [38] is BCE (Li/Yang/Liu, IEEE TEVC
//     20(5):645-665, 2016), the niche-product crowding degree. It had been
//     substituted by an NSGA-II truncation, which was a deviation. It is now
//     implemented following the CCD of the sibling CSMOEA (same authors; its
//     [23] is the same BCE) — compare lis_lcs.hpp::ccd_select: NDS filtering up
//     to >=N -> k-means into N clusters -> t(p) = size of cluster p (singleton
//     -> 3, as [38] recommends) -> r = mean over individuals of the distance to
//     the t-th neighbour -> D(p) = 1 − Π R(p,q), with R = d/r when d <= r and 1
//     otherwise -> iteratively remove the max-D point (random tie-break),
//     recomputing D for the neighbours inside the removed point's niche.
//     Distances and k-means run on min-max normalized objectives (cf. Eq.10).
//     This runs at the end of every step() to present the vault population (the
//     step() framework contract); DP_ and PP_ evolve independently of it.
//   HCCA-6 (MINOR). Real-valued genome; binary is out of scope.
//   HCCA-7 (DEVIATION, reference set + normalization of Eq.2).
//     Eq.2 is evaluated over the NDS-ACCUMULATED set Q built by Alg.2 lines
//     2-8, not over the full input pool PP ∪ Q1 that Alg.2 line 9 and §III
//     name. This follows the sibling CSMOEA paper, whose Alg.2 line 8 says Q,
//     against HCCA's own text, which says PP. Separately, the objectives are
//     min-max normalized before the exponential: e^{I/0.05} overflows on the
//     raw DTLZ1/DTLZ3 scales (the same guard as in lis_lcs.hpp).
//
// CONSTRAINTS (beyond the paper, off by default). constraint_mode
//   FEASIBILITY/CDP makes the non-dominated sorts constrained — the Alg.2 NDS
//   that builds Q, the NDS filter inside the final crowding-degree strategy,
//   and therefore what survives into PP. The Tchebycheff DP branch and the
//   clustering are untouched. The paper is unconstrained.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "../constraint_mode.hpp"
#include "../detail/constrained.hpp"
#include "../das_dennis.hpp"
#include "../data_vault.hpp"
#include "../operators/poly_mutation.hpp"
#include "../operators/sbx.hpp"

namespace mootation {

template <typename Ind_t>
class HCCACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    double eta_c_ = 20.0, eta_m_ = 20.0, pc_ = 1.0, pm_ = -1.0;
    double CR_ = 1.0, F_ = 0.5;
    double delta_ = 0.9;       // MOEA/D neighbor mating prob
    int    nr_    = 2;         // MOEA/D max replacements
    std::mt19937 rng_{std::random_device{}()};

    struct Sol {
        std::vector<double> vars, objs;
        std::vector<double> parent_objs;  // objectives of the associated parent x (Eq.4)
        double              cv = 0.0;      // constraint violation (0 when unconstrained)
        int                 sp = -1;       // associated subproblem/weight index for g^tch (Eq.4/5)
    };

    int m_ = 0, N_ = 0, K_ = 0;
    std::vector<Sol> DP_, PP_;
    std::vector<std::vector<double>> W_;   // weight vectors (Das–Dennis)
    std::vector<std::vector<int>>    B_;   // neighborhoods of subproblems
    std::vector<double> z_;                // ideal point
    int Tnb_ = 0;                          // neighborhood size

    double pm_eff(int nv) const { return (pm_ > 0.0) ? pm_ : (nv > 0 ? 1.0 / nv : 0.0); }

    // ── dominance ─────────────────────────────────────────────────────────
    static bool dominates(const std::vector<double>& a, const std::vector<double>& b) {
        bool better = false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (a[i] > b[i]) return false;
            if (a[i] < b[i]) better = true;
        }
        return better;
    }
    // Constraint-aware form (Deb's constrained domination when the mode is on).
    bool dominates(const Sol& a, const Sol& b) const {
        return detail::dominates(constraint_mode, a.objs, a.cv, b.objs, b.cv);
    }

    static double euclid(const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0; for (std::size_t i = 0; i < a.size(); ++i) { double d = a[i]-b[i]; s += d*d; }
        return std::sqrt(s);
    }

    // ── Tchebycheff (Eq.5) ──────────────────────────────────────────────────
    double gtch(const std::vector<double>& f, const std::vector<double>& w,
                const std::vector<double>& z) const {
        double g = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < m_; ++j) {
            double lam = (w[j] < 1e-6) ? 1e-6 : w[j];
            double v = lam * std::fabs(f[j] - z[j]);
            if (v > g) g = v;
        }
        return g;
    }

    // ── normalize objectives of a set to [0,1] (Eq.10) ──────────────────────
    static std::vector<std::vector<double>> normalize(const std::vector<Sol>& S, int m) {
        std::vector<std::vector<double>> out(S.size(), std::vector<double>(m, 0.0));
        std::vector<double> lo(m,  std::numeric_limits<double>::max());
        std::vector<double> hi(m, -std::numeric_limits<double>::max());
        for (const auto& s : S) for (int k = 0; k < m; ++k) {
            lo[k] = std::min(lo[k], s.objs[k]); hi[k] = std::max(hi[k], s.objs[k]);
        }
        for (std::size_t i = 0; i < S.size(); ++i)
            for (int k = 0; k < m; ++k) {
                double r = hi[k] - lo[k];
                out[i][k] = (r < 1e-14) ? 0.0 : (S[i].objs[k] - lo[k]) / r;
            }
        return out;
    }

    // ── agglomerative average-linkage clustering down to `target` clusters ──
    // (The implementation has always been average-linkage; the comment used to
    // say otherwise.)
    // pts: points in (normalized) objective space; returns cluster index per pt.
    static std::vector<int> agglomerative(const std::vector<std::vector<double>>& pts,
                                          int target) {
        int n = static_cast<int>(pts.size());
        std::vector<int> label(n);
        std::iota(label.begin(), label.end(), 0);
        if (n <= target || target < 1) return label;

        // active cluster ids = members lists
        std::vector<std::vector<int>> members(n);
        for (int i = 0; i < n; ++i) members[i] = {i};
        std::vector<bool> alive(n, true);
        int clusters = n;

        while (clusters > target) {
            double best = std::numeric_limits<double>::max();
            int ba = -1, bb = -1;
            for (int a = 0; a < n; ++a) {
                if (!alive[a]) continue;
                for (int b = a + 1; b < n; ++b) {
                    if (!alive[b]) continue;
                    // average-linkage: mean pairwise distance between clusters
                    // (closer to the EMyO/C the paper cites than single-linkage).
                    double sum = 0.0; int cnt = 0;
                    for (int ia : members[a]) for (int ib : members[b]) {
                        sum += euclid(pts[ia], pts[ib]); ++cnt;
                    }
                    double d = (cnt > 0) ? sum / (double)cnt : 0.0;
                    if (d < best) { best = d; ba = a; bb = b; }
                }
            }
            if (ba < 0) break;
            for (int x : members[bb]) members[ba].push_back(x);
            members[bb].clear();
            alive[bb] = false;
            --clusters;
        }
        for (int a = 0; a < n; ++a)
            if (alive[a]) for (int x : members[a]) label[x] = a;
        // compact labels to 0..C-1
        std::vector<int> remap(n, -1); int c = 0;
        for (int i = 0; i < n; ++i) {
            if (remap[label[i]] < 0) remap[label[i]] = c++;
        }
        for (int i = 0; i < n; ++i) label[i] = remap[label[i]];
        return label;
    }

    // ── k-means (normalized objective space) — used by CCD (HCCA-5) ────────
    // Added for the final crowding degree strategy [38]; cf. lis_lcs.hpp, which
    // uses the same method.
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

    // ── fast non-dominated sort over a set of Sol ───────────────────────────
    std::vector<std::vector<int>> nd_sort(const std::vector<Sol>& S) const {
        int n = static_cast<int>(S.size());
        std::vector<std::vector<int>> dom(n);
        std::vector<int> np(n, 0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                if (dominates(S[i],S[j])) dom[i].push_back(j);
                else if (dominates(S[j],S[i])) ++np[i];
            }
        std::vector<std::vector<int>> fronts;
        std::vector<int> f0;
        for (int i = 0; i < n; ++i) if (np[i] == 0) f0.push_back(i);
        fronts.push_back(f0);
        int k = 0;
        while (!fronts[k].empty()) {
            std::vector<int> nf;
            for (int i : fronts[k]) for (int j : dom[i])
                if (--np[j] == 0) nf.push_back(j);
            fronts.push_back(nf); ++k;
        }
        fronts.pop_back();
        return fronts;
    }

    // ── crowding distance over an index subset of S (Eq.9) ──────────────────
    std::vector<double> crowding(const std::vector<Sol>& S,
                                 const std::vector<int>& idx) const {
        int l = static_cast<int>(idx.size());
        std::vector<double> cd(l, 0.0);
        if (l == 0) return cd;
        if (l <= 2) { for (auto& v : cd) v = std::numeric_limits<double>::infinity(); return cd; }
        for (int k = 0; k < m_; ++k) {
            std::vector<int> ord(l); std::iota(ord.begin(), ord.end(), 0);
            std::sort(ord.begin(), ord.end(), [&](int a, int b) {
                return S[idx[a]].objs[k] < S[idx[b]].objs[k];
            });
            cd[ord.front()] = cd[ord.back()] = std::numeric_limits<double>::infinity();
            double fmin = S[idx[ord.front()]].objs[k];
            double fmax = S[idx[ord.back()]].objs[k];
            double r = fmax - fmin; if (r < 1e-14) continue;
            for (int i = 1; i < l - 1; ++i)
                cd[ord[i]] += (S[idx[ord[i+1]]].objs[k] - S[idx[ord[i-1]]].objs[k]) / r;
        }
        return cd;
    }

    // ── convergence fitness via I_ε+ indicator (Eq.2, IBEA) ─────────────────
    // I_ε+(x2,x1) = min over j eps s.t. f_j(x2) - eps <= f_j(x1)
    //             = max_j ( f_j(x2) - f_j(x1) )   (after normalization).
    static double Ieps(const std::vector<double>& x2, const std::vector<double>& x1) {
        double e = -std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < x2.size(); ++j)
            e = std::max(e, x2[j] - x1[j]);
        return e;
    }

    // Fitness(x1) = sum_{x2!=x1} -exp( I_ε+(x2,x1) / kappa ), kappa=0.05.
    // NOTE the sign: because the exponent is +I/kappa (no minus on I), a
    // well-converged x1 produces a LARGE NEGATIVE fit, so SMALLER means BETTER
    // convergence. That matches the paper (Eq.2: "smaller is better").
    // Selection in LAS therefore takes the MINIMUM fit.
    std::vector<double> conv_fitness(const std::vector<std::vector<double>>& norm) const {
        int n = static_cast<int>(norm.size());
        std::vector<double> fit(n, 0.0);
        for (int i = 0; i < n; ++i) {
            double s = 0.0;
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                s += -std::exp(Ieps(norm[j], norm[i]) / 0.05);
            }
            fit[i] = s;
        }
        return fit;
    }

    // ── DE/rand/1 + PM offspring; evaluated via scratch slot ────────────────
    Sol de_breed(const Sol& base, int base_sp, DataVault<Ind_t>& vault, int scratch) {
        int nv = vault.vars_n();
        const auto& bnd = vault.get_bounds();
        std::uniform_int_distribution<int> di(0, (int)DP_.size() - 1);
        int r1 = di(rng_), r2 = di(rng_), r3 = di(rng_);
        std::uniform_real_distribution<double> ur(0.0, 1.0);
        std::vector<double> child(nv);
        int jrand = std::uniform_int_distribution<int>(0, nv - 1)(rng_);
        for (int j = 0; j < nv; ++j) {
            double lo = bnd[j].first.value_or(0.0), hi = bnd[j].second.value_or(1.0);
            double v;
            if (ur(rng_) < CR_ || j == jrand)
                v = DP_[r1].vars[j] + F_ * (DP_[r2].vars[j] - DP_[r3].vars[j]);
            else
                v = base.vars[j];
            if (v < lo) v = lo; if (v > hi) v = hi;
            child[j] = v;
        }
        ops::polynomial_mutation(child, bnd, eta_m_, pm_eff(nv), rng_);
        Sol o; o.vars = child;
        vault.set_variables(scratch, child); vault.refresh_objectives(scratch);
        o.objs = vault.objectives_of(scratch);
        if (constraint_mode != ConstraintMode::NONE) o.cv = vault.get_cv(scratch);
        o.parent_objs = base.objs;   // parent x for Eq.4 (DE target)
        o.sp = base_sp;              // subproblem/weight of the DP parent (DP[i] <-> W_[i])
        return o;
    }

    // ── SBX + PM offspring ──────────────────────────────────────────────────
    Sol sbx_breed(const Sol& a, const Sol& b, DataVault<Ind_t>& vault, int scratch) {
        const auto& bnd = vault.get_bounds(); int nv = vault.vars_n();
        std::vector<double> c1, c2;
        ops::sbx(a.vars, b.vars, c1, c2, bnd, eta_c_, pc_, rng_);
        ops::polynomial_mutation(c1, bnd, eta_m_, pm_eff(nv), rng_);
        Sol o; o.vars = c1;
        vault.set_variables(scratch, c1); vault.refresh_objectives(scratch);
        o.objs = vault.objectives_of(scratch);
        if (constraint_mode != ConstraintMode::NONE) o.cv = vault.get_cv(scratch);
        o.parent_objs = a.objs;   // parent x for Eq.4 (first SBX parent)
        o.sp = -1;                // PP individuals are not bound to a subproblem; pick weight in adapt_K
        return o;
    }

    void update_ideal(const std::vector<Sol>& S) {
        if (z_.empty()) z_.assign(m_, std::numeric_limits<double>::max());
        for (const auto& s : S) for (int k = 0; k < m_; ++k) z_[k] = std::min(z_[k], s.objs[k]);
    }

    // ── MOEA/D-Tchebycheff environment selection on DP (HCCA-2) ──────────────
    // Incorporates Q (offspring) into DP subproblem assignment via neighbor
    // replacement; returns new DP of size N (one solution per subproblem).
    void moead_update(std::vector<Sol>& cur_DP, const std::vector<Sol>& Q) {
        // z* = min over the CURRENT DP each generation (Eq.6): reset then take mins.
        z_.assign(m_, std::numeric_limits<double>::max());
        update_ideal(cur_DP);
        update_ideal(Q);
        std::uniform_real_distribution<double> ur(0.0, 1.0);
        for (const auto& off : Q) {
            // The offspring updates the neighbourhood OF THE SUBPROBLEM IT WAS
            // BRED FOR (MOEA/D Step 2.4), not a random one. DE children carry
            // that index in Sol::sp; SBX/PP children have no home subproblem
            // (sp = -1) and fall back to a random one, which is the only
            // defensible reading for them. An earlier version drew a random
            // subproblem for EVERY offspring, discarding the index it had just
            // computed and stored — see HCCA-9.
            int sp = (off.sp >= 0 && off.sp < N_)
                         ? off.sp
                         : std::uniform_int_distribution<int>(0, N_ - 1)(rng_);
            const std::vector<int>& pool = (ur(rng_) < delta_) ? B_[sp]
                                          : [&]{ static thread_local std::vector<int> all;
                                                 all.resize(N_); std::iota(all.begin(), all.end(), 0);
                                                 return all; }();
            int replaced = 0;
            std::vector<int> order(pool.begin(), pool.end());
            std::shuffle(order.begin(), order.end(), rng_);
            for (int idx : order) {
                if (replaced >= nr_) break;
                double go = gtch(off.objs, W_[idx], z_);
                double gc = gtch(cur_DP[idx].objs, W_[idx], z_);
                if (go <= gc) { cur_DP[idx] = off; ++replaced; }
            }
        }
    }

    // ── LCS: local coverage selection on PP (Alg.3) ─────────────────────────
    std::vector<Sol> local_coverage_selection(std::vector<Sol> pool) {
        // Step: non-dominated sorting, accept fronts until exceeding N.
        // Fronts accumulate until |Q| >= N, INCLUDING the overflowing front.
        // That is the intended semantics of Alg.3 lines 3-6 together with the
        // line 7 branch "if |Q|>N": the literal line 3 of the paper is
        // defective, since it makes the line 7 branch dead code. The correct
        // idiom is spelled out in Alg.2 lines 4-6 of the same paper:
        // "Q <- Q u F_j; if |Q| >= N break". Accepting only fronts that fit
        // entirely within N and breaking otherwise returned |Q| < N WITHOUT
        // clustering, so PP regularly fell below N and the LCS core (Eq.10-12)
        // never ran.
        auto fronts = nd_sort(pool);
        std::vector<int> Qidx;
        for (auto& fr : fronts) {
            for (int v : fr) Qidx.push_back(v);
            if ((int)Qidx.size() >= N_) break;
        }
        // Build Q set.
        std::vector<Sol> Q; Q.reserve(Qidx.size());
        for (int v : Qidx) Q.push_back(pool[v]);
        if ((int)Q.size() <= N_) return Q;

        // crowding distance over Q (Eq.9)
        std::vector<int> all(Q.size()); std::iota(all.begin(), all.end(), 0);
        std::vector<double> cd = crowding(Q, all);

        // normalize (Eq.10) + cluster into N clusters
        auto norm = normalize(Q, m_);
        int target = std::min((int)Q.size(), N_);
        std::vector<int> label = agglomerative(norm, target);
        int C = 0; for (int l : label) C = std::max(C, l + 1);

        std::vector<std::vector<int>> clu(C);
        for (int i = 0; i < (int)label.size(); ++i) clu[label[i]].push_back(i);

        // Singleton-cluster recombination (Alg.3, lines 16-20). Each singleton
        // merges with its nearest neighbour IF that neighbour is also in a
        // singleton cluster; repeat until no such pair remains.
        {
            bool merged = true;
            while (merged) {
                merged = false;
                std::vector<int> owner(norm.size(), -1);
                for (int c = 0; c < (int)clu.size(); ++c)
                    for (int i : clu[c]) owner[i] = c;
                for (int c = 0; c < (int)clu.size(); ++c) {
                    if (clu[c].size() != 1) continue;
                    int xi = clu[c][0], nn = -1;
                    double best = std::numeric_limits<double>::max();
                    for (int j = 0; j < (int)norm.size(); ++j) {
                        if (j == xi) continue;
                        double d = euclid(norm[xi], norm[j]);
                        if (d < best) { best = d; nn = j; }
                    }
                    if (nn < 0) continue;
                    int oc = owner[nn];
                    if (oc != c && clu[oc].size() == 1) {   // neighbour also a singleton -> merge
                        clu[c].push_back(nn); clu[oc].clear();
                        merged = true; break;
                    }
                }
                if (merged) {                               // drop the emptied ones
                    std::vector<std::vector<int>> nc;
                    for (auto& cc : clu) if (!cc.empty()) nc.push_back(std::move(cc));
                    clu.swap(nc);
                }
            }
        }
        C = (int)clu.size();
        // singletons remaining after recombination, for the (N − sN) budget in Eq.12.
        int sN = 0; for (auto& c : clu) if ((int)c.size() == 1) ++sN;

        // coverage area per cluster (Eq.11) using normalized objectives.
        std::vector<double> area(C, 0.0); double sumA = 0.0;
        for (int c = 0; c < C; ++c) {
            double a = 1.0;
            for (int k = 0; k < m_; ++k) {
                double lo =  std::numeric_limits<double>::max();
                double hi = -std::numeric_limits<double>::max();
                for (int i : clu[c]) { lo = std::min(lo, norm[i][k]); hi = std::max(hi, norm[i][k]); }
                a *= (hi - lo);
            }
            area[c] = a; sumA += a;
        }

        // Num(C) (Eq.12): floor(area/sumA*(N-sN)), at least 1 per cluster.
        std::vector<int> num(C, 0);
        int budget = std::max(0, N_ - sN);
        for (int c = 0; c < C; ++c) {
            int v = 1;
            if (sumA > 1e-300)
                v = std::max(1, (int)std::floor(area[c] / sumA * (double)budget));
            num[c] = std::min(v, (int)clu[c].size());
        }

        // keep Num(C) best by crowding distance per cluster.
        std::vector<int> kept;
        for (int c = 0; c < C; ++c) {
            std::vector<int> ids = clu[c];
            std::sort(ids.begin(), ids.end(), [&](int a, int b) {
                return cd[a] > cd[b];
            });
            for (int t = 0; t < num[c] && t < (int)ids.size(); ++t) kept.push_back(ids[t]);
        }

        // adjust to exactly N: trim worst-crowding or add best leftover.
        if ((int)kept.size() > N_) {
            std::sort(kept.begin(), kept.end(), [&](int a, int b) { return cd[a] > cd[b]; });
            kept.resize(N_);
        } else if ((int)kept.size() < N_) {
            std::vector<char> in(Q.size(), 0); for (int v : kept) in[v] = 1;
            std::vector<int> rest;
            for (int i = 0; i < (int)Q.size(); ++i) if (!in[i]) rest.push_back(i);
            std::sort(rest.begin(), rest.end(), [&](int a, int b) { return cd[a] > cd[b]; });
            for (int i = 0; i < (int)rest.size() && (int)kept.size() < N_; ++i)
                kept.push_back(rest[i]);
        }

        std::vector<Sol> out; out.reserve(kept.size());
        for (int v : kept) out.push_back(Q[v]);
        return out;
    }

    // ── LAS: local area selection — mating + offspring on PP (Alg.2) ────────
    // returns offspring pool (size ~K) and sets K_ for next gen.
    std::vector<Sol> local_area_selection(const std::vector<Sol>& PPunionQ1, int K,
                                          DataVault<Ind_t>& vault, int scratch) {
        // non-dominated sort; collect fronts into Q until |Q| >= N (Alg.2 l.2-8)
        auto fronts = nd_sort(PPunionQ1);
        std::vector<int> Qidx;
        for (auto& fr : fronts) {
            for (int v : fr) Qidx.push_back(v);
            if ((int)Qidx.size() >= N_) break;
        }
        std::vector<Sol> Q; for (int v : Qidx) Q.push_back(PPunionQ1[v]);
        if (Q.empty()) return {};

        // convergence fitness (Eq.2) on normalized Q
        auto norm = normalize(Q, m_);
        std::vector<double> fit = conv_fitness(norm);

        // cluster Q into K clusters
        int target = std::min((int)Q.size(), std::max(1, K));
        std::vector<int> label = agglomerative(norm, target);
        int C = 0; for (int l : label) C = std::max(C, l + 1);
        std::vector<std::vector<int>> clu(C);
        for (int i = 0; i < (int)label.size(); ++i) clu[label[i]].push_back(i);

        std::vector<Sol> OFP;   // offspring pool
        std::vector<Sol> OP;    // best-per-cluster mating pool
        for (int c = 0; c < C; ++c) {
            std::vector<int> ids = clu[c];
            // conv_fitness is SMALLER = better convergence (the -exp(+I/kappa)
            // form makes a well-converged point strongly negative). The paper,
            // Eq.2: "a smaller convergence fitness value implies a better
            // convergence". Sorting in DESCENDING order would take ids[0] = the
            // least converged individual, i.e. pressure AGAINST convergence.
            std::sort(ids.begin(), ids.end(), [&](int a, int b) { return fit[a] < fit[b]; });
            OP.push_back(Q[ids[0]]);
            if ((int)ids.size() >= 2) {
                Sol o = sbx_breed(Q[ids[0]], Q[ids[1]], vault, scratch);
                OFP.push_back(o);
            }
        }
        // fill OFP up to K from OP (Alg.2 l.21-27)
        if (!OP.empty()) {
            std::uniform_int_distribution<int> dop(0, (int)OP.size() - 1);
            while ((int)OFP.size() < K) {
                int a = dop(rng_), b = dop(rng_);
                OFP.push_back(sbx_breed(OP[a], OP[b], vault, scratch));
            }
        }
        return OFP;
    }

    // ── adaptive K (Eq.3-8) from fitness improvement ───────────────────────
    // FI_PP = sum df over PP offspring / K' ; FI_DP = sum df / (N-K')   (Eq.3).
    // df = gtch(parent | W_[sp]) - gtch(child | W_[sp])                 (Eq.4),
    // using the ACTUAL parent objectives recorded at breeding time in
    // Sol::parent_objs — not a re-evaluation and not a stand-in.
    // Eq.4 leaves the subproblem index i undefined for offspring that are not
    // bound to a subproblem: DE children inherit their DP parent's subproblem
    // (o.sp = base_sp), and SBX/PP children (o.sp = -1) are associated to the
    // closest subproblem below. See HCCA-3.
    int adapt_K(const std::vector<Sol>& offPP, const std::vector<Sol>& offDP, int Kprev) {
        if (W_.empty() || z_.empty()) return Kprev;
        auto df_sum = [&](const std::vector<Sol>& off) {
            double s = 0.0;
            for (const auto& o : off) {
                if (o.parent_objs.empty()) continue;  // no parent recorded -> skip
                int sp = o.sp;
                if (sp < 0 || sp >= (int)W_.size()) {
                    // associate child to its closest subproblem (Eq.4 weight)
                    double best = std::numeric_limits<double>::infinity();
                    sp = 0;
                    for (int wi = 0; wi < (int)W_.size(); ++wi) {
                        double g = gtch(o.objs, W_[wi], z_);
                        if (g < best) { best = g; sp = wi; }
                    }
                }
                double g_parent = gtch(o.parent_objs, W_[sp], z_);
                double g_child  = gtch(o.objs,        W_[sp], z_);
                s += (g_parent - g_child);   // Δf (Eq.4); positive means improvement
            }
            return s;
        };
        double FIpp = (Kprev > 0)        ? df_sum(offPP) / (double)Kprev          : 0.0;
        double FIdp = (N_ - Kprev > 0)   ? df_sum(offDP) / (double)(N_ - Kprev)   : 0.0;
        // FI'_PP = FI_PP/(FI_PP+FI_DP) (Eq.7); guard against non-positive sum.
        double denom = FIpp + FIdp;
        double FIpp_n = (std::fabs(denom) > 1e-300) ? FIpp / denom : 0.5;
        if (FIpp_n < 0.0) FIpp_n = 0.0;
        if (FIpp_n > 1.0) FIpp_n = 1.0;
        int K = (int)std::lround(FIpp_n * (double)N_);   // round(FI'_PP*N) (Eq.8)
        K = std::max(std::min(K, N_ - 3), 3);
        return K;
    }

    // ── final crowding degree strategy [38] (Alg.1 l.15), HCCA-5 ─────────────
    // This used to be an NSGA-II truncation (rank + crowding distance). It is
    // now the niche-product crowding degree of [38] = BCE (Li/Yang/Liu, TEVC
    // 2016), in the CCD form (Eq.11-12) of the sibling CSMOEA (whose [23] is
    // the same BCE) — cf. lis_lcs.hpp::ccd_select: NDS filtering up to >=N ->
    // k-means into N clusters -> t(p) = cluster size (singleton -> 3) -> r =
    // mean distance to the t-th neighbour -> D(p) = 1 − Π R(p,q), with R = d/r
    // when d <= r and 1 otherwise -> iteratively remove the max-D point (random
    // tie-break), recomputing D for the neighbours inside its niche.
    std::vector<Sol> final_select(const std::vector<Sol>& PP, const std::vector<Sol>& DP) {
        std::vector<Sol> pool = PP; pool.insert(pool.end(), DP.begin(), DP.end());
        // NDS filtering: fronts up to |Q| >= N (including the overflowing one).
        auto fronts = nd_sort(pool);
        std::vector<int> Q;
        for (auto& fr : fronts) {
            for (int v : fr) Q.push_back(v);
            if ((int)Q.size() >= N_) break;
        }
        int nQ = (int)Q.size();
        std::vector<Sol> out; out.reserve(std::min(nQ, N_));
        if (nQ <= N_) { for (int v : Q) out.push_back(pool[v]); return out; }

        std::vector<Sol> Qs; Qs.reserve(nQ);
        for (int v : Q) Qs.push_back(pool[v]);
        auto Fn = normalize(Qs, m_);

        // k-means into N clusters -> t(p) = size of cluster p (1 -> 3, [38])
        auto label = kmeans(Fn, N_);
        std::vector<int> csz(std::max(N_, nQ), 0);
        for (int l : label) ++csz[l];
        std::vector<int> t(nQ);
        for (int i = 0; i < nQ; ++i)
            t[i] = (csz[label[i]] <= 1) ? 3 : csz[label[i]];

        // distances and niche radius r = mean d(p, t(p)-th neighbour)
        std::vector<std::vector<double>> d(nQ, std::vector<double>(nQ, 0.0));
        for (int i = 0; i < nQ; ++i)
            for (int j = i + 1; j < nQ; ++j) {
                double s = 0.0;
                for (int k = 0; k < m_; ++k) {
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

        // D(p) = 1 − Π_{q≠p, d≤r} d(p,q)/r
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
        // recompute D for the neighbours inside the removed point's niche
        while (na > N_) {
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
        for (int i = 0; i < nQ; ++i) if (alive[i]) out.push_back(Qs[i]);
        return out;
    }

    void build_weights() {
        auto Vr = das_dennis::generate_auto(m_, N_);
        W_.clear();
        for (auto& v : Vr) {
            double s = 0; for (double x : v) s += x;
            if (s < 1e-12) s = 1.0;
            std::vector<double> w(m_); for (int k = 0; k < m_; ++k) w[k] = v[k] / s;
            W_.push_back(w);
        }
        // adjust N_ to number of weights (Das–Dennis may not equal N exactly)
        N_ = (int)W_.size();
        // neighborhoods
        Tnb_ = std::max(2, N_ / 10);
        B_.assign(N_, {});
        for (int i = 0; i < N_; ++i) {
            std::vector<int> idx(N_); std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                return euclid(W_[i], W_[a]) < euclid(W_[i], W_[b]);
            });
            for (int t = 0; t < Tnb_ && t < N_; ++t) B_[i].push_back(idx[t]);
        }
    }

    void init_pops(DataVault<Ind_t>& vault, bool seeded) {
        m_ = vault.objs_n();
        N_ = vault.pop_size();
        build_weights();              // may revise N_ to weight count
        int Nreq = vault.pop_size();
        K_ = std::max(3, std::min(N_ - 3, (int)std::floor(0.5 * N_)));

        std::vector<Sol> base;
        if (seeded) {
            for (int i = 0; i < (int)vault.active_n(); ++i) {
                Sol s; s.vars = vault.variables_of(i); s.objs = vault.objectives_of(i);
                if (constraint_mode != ConstraintMode::NONE) s.cv = vault.get_cv(i);
                base.push_back(s);
            }
        } else {
            const auto& bnd = vault.get_bounds();
            std::uniform_real_distribution<double> ur(0.0, 1.0);
            std::vector<double> vars(vault.vars_n());
            for (int i = 0; i < Nreq; ++i) {
                for (int j = 0; j < vault.vars_n(); ++j) {
                    double lo = bnd[j].first.value_or(0.0), hi = bnd[j].second.value_or(1.0);
                    vars[j] = lo + ur(rng_) * (hi - lo);
                }
                vault.set_variables(i, vars);
            }
            vault.sync();
            for (int i = 0; i < Nreq; ++i) {
                Sol s; s.vars = vault.variables_of(i); s.objs = vault.objectives_of(i);
                if (constraint_mode != ConstraintMode::NONE) s.cv = vault.get_cv(i);
                base.push_back(s);
            }
        }
        // DP and PP both initialized from the same sampled population, sized N_.
        DP_.clear(); PP_.clear();
        std::uniform_int_distribution<int> db(0, (int)base.size() - 1);
        for (int i = 0; i < N_; ++i) {
            DP_.push_back(base[i % base.size()]);
            PP_.push_back(base[i % base.size()]);
        }
        z_.assign(m_, std::numeric_limits<double>::max());
        update_ideal(DP_); update_ideal(PP_);
    }

    void store(DataVault<Ind_t>& vault, const std::vector<Sol>& P) {
        vault.reduce(0);
        vault.expand((int)P.size());
        for (int i = 0; i < (int)P.size(); ++i)
            vault.seed_individual((std::size_t)i, P[i].vars, P[i].objs, {}, {});
    }

public:
    HCCACore() = default;

    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_pc(double p)            { pc_ = p; }
    void set_pm(double p)            { pm_ = p; }
    void set_CR(double c)            { CR_ = c; }
    void set_F(double f)             { F_ = f; }
    void set_delta(double d)         { delta_ = d; }
    void set_nr(int n)               { nr_ = n; }
    void set_seed(unsigned s)        { rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault)        { init_pops(vault, false); store(vault, final_select(PP_, DP_)); }
    void setup_seeded(DataVault<Ind_t>& vault) { init_pops(vault, true);  store(vault, final_select(PP_, DP_)); }

    void step(DataVault<Ind_t>& vault) {
        int scratch = vault.expand(1);

        // (Alg.1 l.4) M <- random N-K solutions from DP
        int nM = std::max(0, N_ - K_);
        std::vector<int> didx(DP_.size()); std::iota(didx.begin(), didx.end(), 0);
        std::shuffle(didx.begin(), didx.end(), rng_);
        nM = std::min(nM, (int)DP_.size());

        // (Alg.1 l.5-8) DE offspring Q1 from DP
        std::vector<Sol> Q1;
        for (int t = 0; t < nM; ++t)
            Q1.push_back(de_breed(DP_[didx[t]], didx[t], vault, scratch));

        // (Alg.1 l.9) LAS on PP ∪ Q1 -> Q2 (offspring), and update K
        std::vector<Sol> PPuQ1 = PP_; PPuQ1.insert(PPuQ1.end(), Q1.begin(), Q1.end());
        std::vector<Sol> Q2 = local_area_selection(PPuQ1, K_, vault, scratch);

        // adaptive K (Eq.3-8): FI from PP offspring (Q2) and DP offspring (Q1)
        K_ = adapt_K(Q2, Q1, K_);

        // (Alg.1 l.10-11) augment both populations with Q1 ∪ Q2
        std::vector<Sol> allOff = Q1; allOff.insert(allOff.end(), Q2.begin(), Q2.end());

        // (Alg.1 l.12) DP environment selection (MOEA/D-Tchebycheff)
        moead_update(DP_, allOff);

        // (Alg.1 l.13) PP local coverage selection
        std::vector<Sol> PPpool = PP_;
        PPpool.insert(PPpool.end(), allOff.begin(), allOff.end());
        PP_ = local_coverage_selection(PPpool);

        // (Alg.1 l.15) final crowding-degree strategy -> output population
        std::vector<Sol> out = final_select(PP_, DP_);
        store(vault, out);
    }
};

} // namespace mootation
