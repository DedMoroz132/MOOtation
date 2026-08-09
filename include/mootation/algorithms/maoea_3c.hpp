#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MaOEA-3C — A many-objective evolutionary algorithm with estimating the
// Convexity-Concavity of Pareto fronts and Clustering.
// X. Wang, F. Zhang, M. Yao — Information Sciences (2023), article 119289.
// DOI: 10.1016/j.ins.2023.119289
//
//
// IDEA. The convexity-concavity of the PF is estimated from a periodically
// updated elitist archive EA: build the hyperplane Sum a_i f_i = 1 through the
// m extreme points, then count how many EA members lie above (L>1) vs below
// (L<1) it (Step 1, Alg.3, Eq. nothing-numbered / lines 5-14). Label is
// concave(+1) if |L>1| > min{0.45|EA|, |L<1|}, else convex(-1) if the mirror
// test holds, else linear(0) — the pseudocode's min-rule, NOT "whichever side
// is bigger"; the two differ and the paper's prose says max (see 3C-7).
// The convergence fitness used in fitness-based sorting then follows the 1by1EA
// estimator: linear -> Sum (Eq.2), concave -> EdI (Eq.4), convex -> EdN (Eq.5).
// EA also acts like reference vectors: Q is clustered by smallest angle to EA
// members (Step 2, Eq.8); fitness is ranked WITHIN each cluster -> FitFrontNo.
//
// FRAMEWORK (Alg.1). P=init(N); FrontNo=NDsort(P); EA=P(FrontNo==1).
//   loop: P'=MatingSelection(P,FrontNo); P''=Variation(P'); Q=P U P''.
//     if FE<0.1*maxFE or |EA|<N: EnvironmentalSelection1(Q,N)  (Alg.2)
//     else                     : EnvironmentalSelection2(Q,EA,N)(Alg.3)
//     if mod(ceil(FE/N), ceil(fr*maxFE/N))==0: EA=updateEA(EA U P'', N) (Alg.4)
//
// MATING (3.2). Binary tournament. Stage-1 (ES1 active): smaller NDFrontNo
//   wins, tie random. Stage-2 (ES2 active): smaller NDFrontNo, then smaller
//   FitFrontNo, tie random.
//
// ES1 (Alg.2). NDsort(Q); take fronts F_1..F_l with cumulated size >= N;
//   P=[F_1..F_{l-1}]; normalize Q (Eq.6); add m extreme points (Eq.7 ASF);
//   fill to N by max-min-angle (Eq.8: d=1-cos).
//
// ES2 (Alg.3). estimate Label; cluster Q to EA; CalFitness(Label);
//   FitSorting within cluster; NDsort -> take NDF_1..NDF_l (>=N);
//   if still >N, take FitF_1..FitF_k (>=N); if still >N put FitF_1..FitF_{k-1}
//   then max-min-angle fill (Alg.2 lines 6-11).
//
// updateEA (Alg.4). EA = nondominated(EA U P''); if |EA|>N, normalize and
//   delete densest one-by-one (Eq.9-10, niche radius r = mean k-th(=3) nearest
//   neighbour distance) until |EA|=N.
//
// DEFAULTS: fr=0.1, switch=0.1*maxFE, EA capacity=N, niche k=3, conv45=0.45,
//   SBX eta_c=20 / pc=1, PM eta_m=20 / pm=1/D. maxFE = t_max * N (FE counter
//   advanced by N each generation, matching one (mu+lambda) step).
//
// DECLARED DEVIATIONS (honest):
//   3C-1 (MINOR). Generation-driven schedule. The library drives the algorithm
//     by step()/t_max (generations), not raw FE. We set FE = t_ * N and
//     maxFE = t_max_ * N, so the 10% switch and the mod(...) EA-update schedule
//     fire at exactly the same population epochs as the paper. Equivalent at
//     N evals/generation (1 offspring per parent, see 3C-2).
//   3C-2 (MINOR). One offspring per parent: |P''|=N (each parent breeds once
//     via SBX -> first child + PM). Paper's "Variation(P')" generates a full
//     offspring population; size N is the standard library convention (cf.
//     isde_rd, theta_dea) and keeps |Q|=2N as in Alg.1.
//   3C-3 (MINOR). FindExtremePoints (Eq.7) returns the ASF argmin per axis;
//     "unique from existing solutions in P" is enforced by skipping indices
//     already in P; if all m are duplicates none are added (P then filled by
//     max-min-angle), matching the stated purpose.
//   3C-4 (MINOR). Hyperplane Sum a_i f_i = 1 is solved from the m extreme
//     points by Gaussian elimination on normalized objectives; if singular we
//     fall back to a_i = 1/(extreme_i along axis i) (the intercept form), then
//     to all-ones (Label=0). Degenerate-archive safety.
//   3C-5 (MINOR). Max-min-angle ties / zero-norm objective vectors: angle
//     distance uses normalized objectives; a zero vector is treated as cos=1
//     (d=0) so it is never preferred for spread, as in isde_rd::cosang.
//   3C-6 (MINOR). Real genome only; binary is out of scope, as across the
//     library.
//   3C-7 (AMBIGUOUS — the paper contradicts itself; pseudocode followed).
//     Alg.3 lines 6 and 9 test |L>1| > min{0.45|EA|, |L<1|} (and the mirror),
//     but the §3.3.2 prose says the count must be "more than BOTH 0.45x|EA|
//     AND |L<1|", i.e. max{...}. These are not the same rule and they can
//     return OPPOSITE labels: with |EA|=100, above=46, below=50, min gives
//     concave(+1) (46 > min(45,50)=45) while max gives convex(-1) (below=50 >
//     max(45,46)=46). This port follows the pseudocode (min), which is also the
//     reading consistent with the prose calling it a "loose condition". Label
//     selects the convergence estimator (Sum / EdI / EdN) for the whole
//     generation, so the choice is load-bearing, not cosmetic.
//   3C-8 (UNDECLARED-NO-MORE: deviation in normalization scope, ES1).
//     Alg.2 line 2 REASSIGNS Q = [F_1..F_l] before line 4 normalizes it and
//     line 5 searches it for extreme points; §3.3.1 confirms it ("solutions in
//     fronts after F_l are discarded (line 2)"), so the paper's Eq.6 z^max is
//     the max over the RETAINED fronts. This port keeps Q as the full 2N union
//     and normalizes over all of it (env_select_1), so z^max can come from a
//     discarded solution. Consequence: every axis is rescaled, which shifts the
//     Eq.7 ASF values and every Eq.8 angle distance, i.e. the fill order of the
//     critical front. The extreme points themselves are practically unaffected
//     — ASF is dominance-monotone, so its argmin is attained in F_1, which is
//     never discarded. Note this does NOT apply to ES2/Alg.3, where normalizing
//     the full Q is correct.
//   3C-9 (AMBIGUOUS -> resolved). Alg.3 line 16 normalizes Q, so CalFitness at
//     line 20 could be read as operating on normalized objectives. This port
//     uses RAW objectives with z^min/z^max taken over Q, because: line 16 sits
//     inside the "Step 2: Associate Q with EA" block and exists to make the
//     Eq.8 angle distance computable; line 20 names no operand while lines 5,
//     17 and 23 all name theirs; and the paper keeps a strict primed/unprimed
//     convention — Eq.6 defines F'(x) and Eq.7/Eq.8 use primes, while Eq.2/4/5
//     are unprimed and define z^min, z^max explicitly, which would be vacuous
//     (0 and 1) on normalized input. The normalized reading is admissible; the
//     two orderings coincide within a narrow angular cluster and diverge for
//     wide or sparse ones.
//
// CONSTRAINTS (beyond the paper, off by default). constraint_mode
//   FEASIBILITY/CDP routes every non-dominated sort through Deb's constrained
//   domination, which is where MaOEA-3C's selection pressure lives: ES1's front
//   accumulation, ES2's NDF levels, the mating tournament's NDFrontNo, and the
//   Alg.4 archive filter all consume it. The convexity estimate and the
//   Sum/EdI/EdN fitnesses are left on raw objectives — the paper defines them
//   there, and a feasible-only EA already carries the feasibility preference.
//   The paper is unconstrained (DTLZ/WFG/MaF), so this is an extension.
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
class MaOEA3CCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

    MaOEA3CCore() = default;
    void set_seed(unsigned s) { rng_.seed(s); }
    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation(double e)  { eta_m_ = e; }
    void set_t_max(int t) { t_max_ = t; }
    void set_fr(double f) { fr_ = f; }
    void set_conv45(double c) { conv45_ = c; }
    void set_niche_k(int k) { niche_k_ = k; }

private:
    std::mt19937 rng_{std::random_device{}()};
    int    t_max_ = 250, t_ = 0;
    double fr_ = 0.1;        // EA update frequency (x maxFE)
    double conv45_ = 0.45;   // dominant-side threshold (lines 6-9, Alg.3)
    int    niche_k_ = 3;     // kth-nearest for niche radius r (Eq.10)
    double eta_c_ = 20.0, eta_m_ = 20.0, pc_ = 1.0, pm_ = -1.0;

    struct Sol {
        std::vector<double> vars, objs;
        double cv = 0.0;      // constraint violation (0 when unconstrained)
        int    ndfront = 0;   // NDFrontNo
        int    fitfront = 0;  // FitFrontNo
    };

    int                m_ = 0, N_ = 0;
    std::vector<Sol>   P_;    // current population
    std::vector<Sol>   EA_;   // elitist archive
    bool               es2_active_ = false; // which mating rule applies

    double pm_eff(int nv) const { return (pm_ > 0.0) ? pm_ : (nv > 0 ? 1.0 / nv : 0.0); }

    // ── Domination (minimization). Constrained domination when
    //    constraint_mode is on, plain Pareto otherwise. ────────────────────
    bool dominates(const Sol& a, const Sol& b) const {
        return detail::dominates(constraint_mode, a.objs, a.cv, b.objs, b.cv);
    }

    // Fast non-dominated sort: returns fronts (vectors of indices into S).
    std::vector<std::vector<int>> nd_sort(const std::vector<Sol>& S) const {
        int n = (int)S.size();
        std::vector<std::vector<int>> dom(n);
        std::vector<int> cnt(n, 0);
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                if (dominates(S[i], S[j])) { dom[i].push_back(j); ++cnt[j]; }
                else if (dominates(S[j], S[i])) { dom[j].push_back(i); ++cnt[i]; }
            }
        std::vector<std::vector<int>> fronts;
        std::vector<int> cur;
        for (int i = 0; i < n; ++i) if (cnt[i] == 0) cur.push_back(i);
        while (!cur.empty()) {
            fronts.push_back(cur);
            std::vector<int> nxt;
            for (int p : cur)
                for (int q : dom[p])
                    if (--cnt[q] == 0) nxt.push_back(q);
            cur.swap(nxt);
        }
        return fronts;
    }

    // ── Normalization (Eq.6): returns normalized objective vectors ───────
    static std::vector<std::vector<double>> normalize(const std::vector<Sol>& S, int m) {
        std::vector<double> zmin(m, std::numeric_limits<double>::max());
        std::vector<double> zmax(m, -std::numeric_limits<double>::max());
        for (const auto& s : S)
            for (int j = 0; j < m; ++j) {
                zmin[j] = std::min(zmin[j], s.objs[j]);
                zmax[j] = std::max(zmax[j], s.objs[j]);
            }
        std::vector<std::vector<double>> F(S.size(), std::vector<double>(m));
        for (std::size_t i = 0; i < S.size(); ++i)
            for (int j = 0; j < m; ++j) {
                double d = zmax[j] - zmin[j];
                F[i][j] = (d > 1e-12) ? (S[i].objs[j] - zmin[j]) / d : 0.0;
            }
        return F;
    }

    // ── ASF extreme points (Eq.7): returns m indices into F ──────────────
    static std::vector<int> find_extremes(const std::vector<std::vector<double>>& F, int m) {
        std::vector<int> ex(m, -1);
        for (int axis = 0; axis < m; ++axis) {
            double best = std::numeric_limits<double>::max();
            int    bi = -1;
            for (int i = 0; i < (int)F.size(); ++i) {
                double asf = -std::numeric_limits<double>::max();
                for (int j = 0; j < m; ++j) {
                    double w = (j == axis) ? 1.0 : 1e-6;
                    asf = std::max(asf, F[i][j] / w);
                }
                if (asf < best) { best = asf; bi = i; }
            }
            ex[axis] = bi;
        }
        return ex;
    }

    // ── Angle distance d = 1 - cos (Eq.8) on normalized objectives ───────
    static double cosang(const std::vector<double>& a, const std::vector<double>& b) {
        double d = 0, na = 0, nb = 0;
        for (std::size_t i = 0; i < a.size(); ++i) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        double q = std::sqrt(na) * std::sqrt(nb);
        if (q < 1e-300) return 1.0;   // zero vector -> treated as aligned (3C-5)
        return std::clamp(d / q, -1.0, 1.0);
    }
    static double angle_dist(const std::vector<double>& a, const std::vector<double>& b) {
        return 1.0 - cosang(a, b);
    }

    // ── Max-min-angle fill (Alg.2 lines 6-11). sel/cand index into F. ────
    // Moves indices from cand to sel until |sel| == target.
    static void maxmin_angle_fill(const std::vector<std::vector<double>>& F,
                                  std::vector<int>& sel, std::vector<int>& cand,
                                  int target) {
        while ((int)sel.size() < target && !cand.empty()) {
            int    bestc = -1;
            double bestmin = -1.0;
            for (int ci = 0; ci < (int)cand.size(); ++ci) {
                double mind = std::numeric_limits<double>::max();
                for (int s : sel)
                    mind = std::min(mind, angle_dist(F[cand[ci]], F[s]));
                if (sel.empty()) mind = 0.0;
                if (mind > bestmin) { bestmin = mind; bestc = ci; }
            }
            if (bestc < 0) break;
            sel.push_back(cand[bestc]);
            cand.erase(cand.begin() + bestc);
        }
    }

    // ── Hyperplane Sum a_i f_i = 1 from m extreme points (3C-4) ──────────
    // Solve A a = 1 where rows of A are the extreme points' normalized objs.
    static std::vector<double> build_hyperplane(const std::vector<std::vector<double>>& F,
                                                const std::vector<int>& ex, int m) {
        // Gaussian elimination on the augmented system.
        std::vector<std::vector<double>> A(m, std::vector<double>(m + 1, 0.0));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) A[i][j] = F[ex[i]][j];
            A[i][m] = 1.0;
        }
        std::vector<double> a(m, 0.0);
        bool ok = true;
        for (int col = 0; col < m && ok; ++col) {
            int piv = col;
            for (int r = col + 1; r < m; ++r)
                if (std::fabs(A[r][col]) > std::fabs(A[piv][col])) piv = r;
            if (std::fabs(A[piv][col]) < 1e-12) { ok = false; break; }
            std::swap(A[col], A[piv]);
            for (int r = 0; r < m; ++r) {
                if (r == col) continue;
                double f = A[r][col] / A[col][col];
                for (int c = col; c <= m; ++c) A[r][c] -= f * A[col][c];
            }
        }
        if (ok) {
            for (int i = 0; i < m; ++i) a[i] = A[i][m] / A[i][i];
        } else {
            // fallback: intercept form a_i = 1 / (extreme_i value on axis i)
            ok = true;
            for (int i = 0; i < m; ++i) {
                double v = F[ex[i]][i];
                if (std::fabs(v) < 1e-12) { ok = false; break; }
                a[i] = 1.0 / v;
            }
            if (!ok) std::fill(a.begin(), a.end(), 1.0); // all-ones -> Label likely 0
        }
        return a;
    }

    // ── Step 1 of Alg.3: estimate Label from EA ──────────────────────────
    int estimate_label() const {
        if ((int)EA_.size() < m_ + 1) return 0; // too few -> linear
        auto Fea = normalize(EA_, m_);
        auto ex  = find_extremes(Fea, m_);
        auto a   = build_hyperplane(Fea, ex, m_);
        int above = 0, below = 0;
        for (const auto& f : Fea) {
            double L = 0;
            for (int j = 0; j < m_; ++j) L += a[j] * f[j];
            if (L > 1.0 + 1e-9) ++above;
            else if (L < 1.0 - 1e-9) ++below;
        }
        double thr = conv45_ * (double)EA_.size();
        // line 6: |L>1| > min{0.45|EA|, |L<1|} -> concave (+1)
        if ((double)above > std::min(thr, (double)below)) return 1;
        // line 9: |L<1| > min{0.45|EA|, |L>1|} -> convex (-1)
        if ((double)below > std::min(thr, (double)above)) return -1;
        return 0; // linear
    }

    // ── Fitness by estimator (Step 3, Eq.2/4/5) on RAW objectives ────────
    // Smaller fitness preferred. EdN (Eq.5) is 1/dist(nadir): smaller = farther
    // from nadir = better, so we keep its value (small is good).
    std::vector<double> cal_fitness(const std::vector<Sol>& S, int label) const {
        std::vector<double> fit(S.size());
        std::vector<double> zmin(m_, std::numeric_limits<double>::max());
        std::vector<double> zmax(m_, -std::numeric_limits<double>::max());
        for (const auto& s : S)
            for (int j = 0; j < m_; ++j) {
                zmin[j] = std::min(zmin[j], s.objs[j]);
                zmax[j] = std::max(zmax[j], s.objs[j]);
            }
        for (std::size_t i = 0; i < S.size(); ++i) {
            const auto& f = S[i].objs;
            if (label == 0) {              // Sum (Eq.2)
                double s = 0; for (int j = 0; j < m_; ++j) s += f[j];
                fit[i] = s;
            } else if (label == 1) {       // EdI (Eq.4)
                double s = 0; for (int j = 0; j < m_; ++j) { double d = f[j] - zmin[j]; s += d * d; }
                fit[i] = std::sqrt(s);
            } else {                       // EdN (Eq.5): 1/dist to nadir
                double s = 0; for (int j = 0; j < m_; ++j) { double d = f[j] - zmax[j]; s += d * d; }
                double dn = std::sqrt(s);
                fit[i] = (dn > 1e-12) ? (1.0 / dn) : std::numeric_limits<double>::max();
            }
        }
        return fit;
    }

    // ── Breed one offspring from two parents (3C-2) ──────────────────────
    Sol breed(const Sol& x, const Sol& y, DataVault<Ind_t>& vault, int scratch) {
        const auto& b = vault.get_bounds();
        int nv = vault.vars_n();
        std::vector<double> c1, c2;
        ops::sbx(x.vars, y.vars, c1, c2, b, eta_c_, pc_, rng_);
        ops::polynomial_mutation(c1, b, eta_m_, pm_eff(nv), rng_);
        Sol z; z.vars = c1;
        vault.set_variables(scratch, c1);
        vault.refresh_objectives(scratch);
        z.objs = vault.objectives_of(scratch);
        if (constraint_mode != ConstraintMode::NONE) z.cv = vault.get_cv(scratch);
        return z;
    }

    // ── Mating selection (3.2): produce N parents (indices into P_) ──────
    std::vector<int> mating_selection() {
        std::uniform_int_distribution<int> pick(0, N_ - 1);
        std::uniform_real_distribution<double> coin(0.0, 1.0);
        std::vector<int> pool(N_);
        for (int i = 0; i < N_; ++i) {
            int a = pick(rng_), b = pick(rng_);
            int win;
            if (P_[a].ndfront != P_[b].ndfront)
                win = (P_[a].ndfront < P_[b].ndfront) ? a : b;
            else if (es2_active_ && P_[a].fitfront != P_[b].fitfront)
                win = (P_[a].fitfront < P_[b].fitfront) ? a : b;
            else
                win = (coin(rng_) < 0.5) ? a : b;
            pool[i] = win;
        }
        return pool;
    }

    // ── EnvironmentalSelection1 (Alg.2). Sets P_ (with ndfront). ─────────
    void env_select_1(std::vector<Sol>& Q) {
        auto fronts = nd_sort(Q);
        for (std::size_t fi = 0; fi < fronts.size(); ++fi)
            for (int idx : fronts[fi]) Q[idx].ndfront = (int)fi + 1;

        // l: minimal cumulated front count >= N
        std::vector<int> Pidx;          // selected (indices into Q)
        std::vector<int> lastFront;
        int cum = 0; std::size_t l = 0;
        for (; l < fronts.size(); ++l) {
            cum += (int)fronts[l].size();
            if (cum >= N_) { lastFront = fronts[l]; break; }
        }
        // P = F_1..F_{l-1}
        for (std::size_t fi = 0; fi < l; ++fi)
            for (int idx : fronts[fi]) Pidx.push_back(idx);

        // candidate pool for max-min-angle = the last front F_l
        // normalize whole Q (Eq.6)
        auto F = normalize(Q, m_);

        if ((int)Pidx.size() < N_) {
            // add m extreme points (from full Q), unique from Pidx (3C-3)
            auto ex = find_extremes(F, m_);
            std::vector<char> inP(Q.size(), 0);
            for (int p : Pidx) inP[p] = 1;
            std::vector<char> inCand(Q.size(), 0);
            for (int c : lastFront) inCand[c] = 1;
            for (int axis = 0; axis < m_ && (int)Pidx.size() < N_; ++axis) {
                int e = ex[axis];
                if (e >= 0 && !inP[e]) {
                    Pidx.push_back(e); inP[e] = 1;
                    inCand[e] = 0; // if it was a candidate, remove it
                }
            }
            // build candidate list = last front minus already selected
            std::vector<int> cand;
            for (int c : lastFront) if (!inP[c]) cand.push_back(c);
            maxmin_angle_fill(F, Pidx, cand, N_);
        }
        if ((int)Pidx.size() > N_) Pidx.resize(N_);

        std::vector<Sol> nextP;
        nextP.reserve(Pidx.size());
        for (int idx : Pidx) nextP.push_back(Q[idx]);
        P_.swap(nextP);
    }

    // ── EnvironmentalSelection2 (Alg.3). Sets P_ (ndfront + fitfront). ───
    void env_select_2(std::vector<Sol>& Q) {
        // Step 1: estimate Label from EA
        int label = estimate_label();

        // Step 2: associate Q with EA by smallest angle (Eq.8) -> cluster id
        auto Fq  = normalize(Q, m_);
        auto Fea = normalize(EA_, m_);
        std::vector<int> clust(Q.size(), 0);
        for (std::size_t i = 0; i < Q.size(); ++i) {
            int best = 0; double bd = std::numeric_limits<double>::max();
            for (std::size_t k = 0; k < EA_.size(); ++k) {
                double d = angle_dist(Fq[i], Fea[k]);
                if (d < bd) { bd = d; best = (int)k; }
            }
            clust[i] = best;
        }

        // Step 3: fitness within cluster -> FitFrontNo
        auto fit = cal_fitness(Q, label);
        // group indices by cluster, sort each by fitness ascending; assign rank
        std::vector<std::vector<int>> groups(EA_.size());
        for (std::size_t i = 0; i < Q.size(); ++i) groups[clust[i]].push_back((int)i);
        for (auto& g : groups) {
            std::sort(g.begin(), g.end(), [&](int a, int b) { return fit[a] < fit[b]; });
            for (std::size_t r = 0; r < g.size(); ++r) Q[g[r]].fitfront = (int)r + 1;
        }

        // Step 4: NDsort -> NDFrontNo
        auto fronts = nd_sort(Q);
        for (std::size_t fi = 0; fi < fronts.size(); ++fi)
            for (int idx : fronts[fi]) Q[idx].ndfront = (int)fi + 1;

        // Q = [NDF_1..NDF_l], l minimal with cumulated >= N
        std::vector<int> sel;
        int cum = 0; std::size_t l = 0;
        for (; l < fronts.size(); ++l) {
            for (int idx : fronts[l]) sel.push_back(idx);
            cum += (int)fronts[l].size();
            if (cum >= N_) break;
        }

        std::vector<int> Pidx;
        if ((int)sel.size() <= N_) {
            Pidx = sel; // |Q| <= N -> P = Q (lines 33-34)
        } else {
            // compare by FitFrontNo within the NDF_1..NDF_l set (line 26)
            int maxfit = 0;
            for (int idx : sel) maxfit = std::max(maxfit, Q[idx].fitfront);
            std::vector<std::vector<int>> byfit(maxfit + 1);
            for (int idx : sel) byfit[Q[idx].fitfront].push_back(idx);
            // k: minimal cumulated FitF count >= N
            std::vector<int> kept; std::vector<int> kfront;
            int c2 = 0; int k = 1;
            for (; k <= maxfit; ++k) {
                for (int idx : byfit[k]) kept.push_back(idx);
                c2 += (int)byfit[k].size();
                if (c2 >= N_) { kfront = byfit[k]; break; }
            }
            if ((int)kept.size() <= N_) {
                Pidx = kept; // |Q| == N -> P = Q (lines 30-31)
            } else {
                // put FitF_1..FitF_{k-1} into P, then max-min-angle fill (lines 28-29)
                std::vector<char> inLast(Q.size(), 0);
                for (int idx : kfront) inLast[idx] = 1;
                for (int idx : kept) if (!inLast[idx]) Pidx.push_back(idx);
                // normalize Q already in Fq; candidate = last fit-front
                std::vector<int> cand = kfront;
                maxmin_angle_fill(Fq, Pidx, cand, N_);
            }
        }
        if ((int)Pidx.size() > N_) Pidx.resize(N_);

        std::vector<Sol> nextP;
        nextP.reserve(Pidx.size());
        for (int idx : Pidx) nextP.push_back(Q[idx]);
        P_.swap(nextP);
    }

    // ── updateEA (Alg.4) ─────────────────────────────────────────────────
    void update_ea(const std::vector<Sol>& offspring) {
        std::vector<Sol> U = EA_;
        for (const auto& s : offspring) U.push_back(s);
        // EA = nondominated(U)
        auto fronts = nd_sort(U);
        std::vector<Sol> ea;
        if (!fronts.empty())
            for (int idx : fronts[0]) ea.push_back(U[idx]);
        if ((int)ea.size() > N_) {
            // normalize then delete densest one-by-one (Eq.9-10)
            // recompute normalized objs of ea on the fly each removal-free;
            // niche radius r = mean over members of their k-th nearest distance.
            std::vector<std::vector<double>> F = normalize(ea, m_);
            std::vector<int> alive(ea.size());
            std::iota(alive.begin(), alive.end(), 0);
            while ((int)alive.size() > N_) {
                int na = (int)alive.size();
                // pairwise distances among alive
                std::vector<std::vector<double>> dist(na, std::vector<double>(na, 0.0));
                for (int i = 0; i < na; ++i)
                    for (int j = i + 1; j < na; ++j) {
                        double s = 0;
                        for (int d = 0; d < m_; ++d) {
                            double diff = F[alive[i]][d] - F[alive[j]][d];
                            s += diff * diff;
                        }
                        double dd = std::sqrt(s);
                        dist[i][j] = dist[j][i] = dd;
                    }
                // niche radius r = mean of each member's k-th nearest distance
                double rsum = 0; int kk = std::min(niche_k_, na - 1);
                if (kk < 1) kk = 1;
                for (int i = 0; i < na; ++i) {
                    std::vector<double> row;
                    for (int j = 0; j < na; ++j) if (j != i) row.push_back(dist[i][j]);
                    std::sort(row.begin(), row.end());
                    rsum += row[std::min((int)row.size() - 1, kk - 1)];
                }
                double r = (na > 0) ? rsum / na : 1.0;
                if (r < 1e-12) r = 1e-12;
                // density D(p) = 1 - prod_{q!=p} R(p,q) (Eq.9-10)
                int    worst = -1; double worstD = -1.0;
                for (int i = 0; i < na; ++i) {
                    double prod = 1.0;
                    for (int j = 0; j < na; ++j) {
                        if (j == i) continue;
                        double R = (dist[i][j] <= r) ? (dist[i][j] / r) : 1.0;
                        prod *= R;
                    }
                    double D = 1.0 - prod;
                    if (D > worstD) { worstD = D; worst = i; }
                }
                if (worst < 0) worst = 0;
                alive.erase(alive.begin() + worst);
            }
            std::vector<Sol> trimmed;
            for (int a : alive) trimmed.push_back(ea[a]);
            ea.swap(trimmed);
        }
        EA_.swap(ea);
    }

    // ── read full population from vault into P_ ──────────────────────────
    void read_pop(DataVault<Ind_t>& vault) {
        P_.clear();
        for (int i = 0; i < N_; ++i) {
            Sol s; s.vars = vault.variables_of(i); s.objs = vault.objectives_of(i);
            if (constraint_mode != ConstraintMode::NONE) s.cv = vault.get_cv(i);
            P_.push_back(s);
        }
    }
    // ── write P_ back to the vault as the active population ──────────────
    void store_pop(DataVault<Ind_t>& vault) {
        vault.reduce(0);
        vault.expand((int)P_.size());
        for (int i = 0; i < (int)P_.size(); ++i)
            vault.seed_individual((std::size_t)i, P_[i].vars, P_[i].objs, {}, {});
    }

    // initialize EA and front numbers after population is set
    void init_archive() {
        auto fronts = nd_sort(P_);
        for (std::size_t fi = 0; fi < fronts.size(); ++fi)
            for (int idx : fronts[fi]) P_[idx].ndfront = (int)fi + 1;
        EA_.clear();
        if (!fronts.empty())
            for (int idx : fronts[0]) EA_.push_back(P_[idx]);
    }

public:
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
        read_pop(vault);
        init_archive();
        t_ = 0; es2_active_ = false;
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        m_ = vault.objs_n();
        N_ = vault.pop_size();
        read_pop(vault);
        init_archive();
        t_ = 0; es2_active_ = false;
    }

    void step(DataVault<Ind_t>& vault) {
        int scratch = vault.expand(1);

        // FE bookkeeping (3C-1): FE = consumed evals BEFORE this generation.
        long FE     = (long)t_ * (long)N_;
        long maxFE  = (long)t_max_ * (long)N_;

        // Mating (3.2)
        auto pool = mating_selection();

        // Variation -> offspring P'' (one per parent, 3C-2)
        std::vector<Sol> offspring;
        offspring.reserve(N_);
        for (int i = 0; i < N_; ++i) {
            int a = pool[i];
            int b = pool[(i + 1) % N_];
            offspring.push_back(breed(P_[a], P_[b], vault, scratch));
        }

        // Q = P U P''
        std::vector<Sol> Q = P_;
        for (auto& s : offspring) Q.push_back(s);

        // Environmental selection (Alg.1 lines 8-12)
        bool use_es1 = (FE < (long)(0.1 * (double)maxFE)) || ((int)EA_.size() < N_);
        if (use_es1) { es2_active_ = false; env_select_1(Q); }
        else         { es2_active_ = true;  env_select_2(Q); }

        // EA update schedule (Alg.1 lines 13-15): mod(ceil(FE'/N), ceil(fr*maxFE/N))==0
        // Use the FE consumed AFTER this generation (FE + N), matching paper epochs.
        long feAfter = FE + (long)N_;
        long period  = (long)std::ceil(fr_ * (double)maxFE / (double)N_);
        if (period < 1) period = 1;
        long epoch   = (long)std::ceil((double)feAfter / (double)N_);
        if (epoch % period == 0)
            update_ea(offspring);

        store_pop(vault);
        ++t_;
    }
};

} // namespace mootation
