#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// AR-MOEA — An Indicator-Based Multiobjective Evolutionary Algorithm With
//           Reference Point Adaptation for Better Versatility
// Y. Tian, R. Cheng, X. Zhang, F. Cheng, Y. Jin — IEEE TEVC 22(4), 2018
// doi:10.1109/TEVC.2017.2749619
//
// Four sets (§III-B, Fig.3): the population P, the original reference points R
// (Das-Dennis), the external archive A (living in vault.archive_*), and the
// adapted reference points R'. Generational scheme (Alg.1):
//   1. MatingSelection (Alg.2): f_i(p) -= the min over P; binary tournament on
//      fitness_p = IGD-NS(P\{p}, R') (Eq.4), where the larger contribution
//      wins.
//   2. Variation: SBX (pc=1.0, eta_c=20) + polynomial mutation (pm=1/D,
//      eta_m=20) — §IV-A.3.
//   3. RefPointAdaption (Alg.3): z* and z^nad from P; the objectives of A u O
//      and of P are SHIFTED by z*, while the points of R are SCALED by
//      (z^nad − z*) — the objectives are not divided by the range;
//      deduplicate A and drop its dominated members;
//      R <- AdjustLocation(R, A) (Alg.4); A^con (the contributing set of Eq.2)
//      becomes A', topped up by max-min angle to min(|R|,|A|); R^valid (the
//      points nearest to the solutions of A^con) becomes R', topped up with the
//      F(p) points of A' by max-min angle to min(|R|,|A'|);
//      R' <- AdjustLocation(R', P).
//   4. EnvironmentalSelection (Alg.5): f_i(p) -= the min over P u O; NDS; whole
//      fronts up to the critical one; within the critical front, iteratively
//      REMOVE p* = argmin_p IGD-NS(Front_k\{p}, R'), recomputing after each
//      removal.
//
// IGD-NS (Eq.2-3): IGD-NS(X,Y) = Sum_{y in Y} min_{x in X} dis(y,x)
//                              + Sum_{x' in X*} min_{y in Y} dis(y,x'),
//   where X* is the set of noncontributing solutions: those x' that are nearest
//   to no reference point y in Y (Eq.2).
//
// AdjustLocation (Alg.4, Fig.5): for each r, take the solution p with the
//   smallest perpendicular distance ||F(p)||*sin(angle(z*r, F(p))) to the ray
//   z* -> r; r is moved to the orthogonal projection of F(p) onto that ray:
//   r' = r/||r|| * ||F(p)||*cos(angle). This preserves the extremes on convex
//   fronts (Fig.4b).
//
// PAPER DEFAULTS (§IV-A.3): pc=1.0, pm=1/D, eta_c=20, eta_m=20 ("The
//   probabilities of crossover and mutation are set to 1.0 and 1/D ... The
//   distribution indexes of both SBX and polynomial mutation are set to 20").
//   Beyond the operators the algorithm is parameter-free ("for NSGA-II,
//   NSGA-III, A-NSGA-III and AR-MOEA ... there is no additional parameter").
//   |R| = N_R: by default the nearest Das-Dennis lattice >= pop_size
//   (das_dennis::generate_auto; Table II gives N_R = 105/126/275 for
//   M = 3/5/10, two-layer at 10 objectives). The population size is arbitrary
//   and NOT tied to the lattice: N may be below N_R, and there is no
//   solution-to-point association (§III-D, §IV-D), so no exception is thrown
//   over pop_size.
// DECLARED DEVIATIONS: none. Details the paper leaves unspecified:
//   - argmin/argmax ties are broken deterministically (the first extremum); on
//     an exact distance tie there is a single contributing carrier;
//   - ENS/T-ENS (§III-D) is replaced by a fast NDS with an equivalent result;
//   - numerical guards in AdjustLocation: ||r|| < 1e-12 leaves r in place, and
//     a negative projection is clamped to 0;
//   - archive duplicates are detected by exact equality of the objective
//     vectors.
// EXTENSIONS BEYOND THE PAPER (off by default): ConstraintMode::FEASIBILITY
//   (CDP dominance, feasible-first in the tournament); mixed real+binary genome
//   (uniform crossover + bit-flip); the set_pc / set_pm / set_n_ref setters.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
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

struct ARMOEA_Individual : public Based_Individual {
    int rank = 0;   // NDS front from the last environmental selection
};

template <typename Ind_t>
class ARMOEACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    static constexpr double EPS_ = 1e-12;

    double       eta_c_ = 20.0;   // §IV-A.3
    double       eta_m_ = 20.0;   // §IV-A.3
    double       pc_    = 1.0;    // §IV-A.3
    double       pm_    = -1.0;   // <0 → 1/D (§IV-A.3)
    int          n_ref_ = 0;      // 0 -> nearest lattice >= pop_size
    std::mt19937 rng_{std::random_device{}()};

    std::vector<std::vector<double>> R0_;  // R  = original points (on the simplex)
    std::vector<std::vector<double>> Rp_;  // R' = adapted points
                                           //      (in translated coordinates)

    // ── Geometry ────────────────────────────────────────────────────────────

    static double edist(const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) { double t = a[j] - b[j]; s += t * t; }
        return std::sqrt(s);
    }
    static double dot(const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) s += a[j] * b[j];
        return s;
    }
    static double norm(const std::vector<double>& a) { return std::sqrt(dot(a, a)); }

    // Acute angle between vectors (Alg.3 lines 16/22: arccos(F(p),F(q))).
    static double acute_angle(const std::vector<double>& a, const std::vector<double>& b) {
        double na = norm(a), nb = norm(b);
        if (na < EPS_ || nb < EPS_) return 0.0;
        double c = dot(a, b) / (na * nb);
        c = std::min(1.0, std::max(-1.0, c));
        return std::acos(c);
    }

    // ── IGD-NS (Eq.2-3) ─────────────────────────────────────────────────────

    // fitness_p = IGD-NS(X\{p}, Y) for every p at once (Eq.4) in O(|X|*|Y|):
    // each y keeps its two nearest solutions; when p is removed, the reference
    // points whose argmin was p fall through to the second nearest, which then
    // becomes contributing; the noncontributing sum is corrected over the
    // affected x.
    static std::vector<double> igdns_drop_one(
            const std::vector<std::vector<double>>& X,
            const std::vector<std::vector<double>>& Y)
    {
        const double INF = std::numeric_limits<double>::max();
        const int nx = static_cast<int>(X.size());
        const int ny = static_cast<int>(Y.size());
        std::vector<double> fit(nx, 0.0);
        if (nx <= 1 || ny == 0) return fit;

        std::vector<int>    i1(ny, -1), i2(ny, -1);
        std::vector<double> d1(ny, INF), d2(ny, INF);
        std::vector<double> dref(nx, INF);          // min_y dis(y,x), for X*
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                double d = edist(X[x], Y[y]);
                if (d < d1[y]) { d2[y] = d1[y]; i2[y] = i1[y]; d1[y] = d; i1[y] = x; }
                else if (d < d2[y]) { d2[y] = d; i2[y] = x; }
                if (d < dref[x]) dref[x] = d;
            }
        }
        std::vector<int>              cnt(nx, 0);   // how many y have argmin == x
        std::vector<std::vector<int>> ys(nx);       // those y
        double base_ref = 0.0;
        for (int y = 0; y < ny; ++y) {
            base_ref += d1[y];
            ++cnt[i1[y]];
            ys[i1[y]].push_back(y);
        }
        double S0 = 0.0;                            // noncontributing sum of X
        for (int x = 0; x < nx; ++x) if (cnt[x] == 0) S0 += dref[x];

        std::vector<char> fl(nx, 0);
        std::vector<int>  touched;
        for (int p = 0; p < nx; ++p) {
            double sref = base_ref;
            double snc  = S0 - (cnt[p] == 0 ? dref[p] : 0.0);
            touched.clear();
            for (int y : ys[p]) {
                sref += d2[y] - d1[y];
                int q = i2[y];                      // becomes contributing
                if (q >= 0 && cnt[q] == 0 && !fl[q]) {
                    fl[q] = 1; touched.push_back(q); snc -= dref[q];
                }
            }
            for (int q : touched) fl[q] = 0;
            fit[p] = sref + snc;
        }
        return fit;
    }

    // ── AdjustLocation (Alg.4) ───────────────────────────────────────────────
    // For each r: p = argmin_p ||F(p)||*sin(angle(z*r, F(p))), the
    // perpendicular to the ray z* -> r; then r' = r/||r|| * ||F(p)||*cos(angle),
    // the orthogonal projection of F(p).
    static std::vector<std::vector<double>> adjust_location(
            const std::vector<std::vector<double>>& R,
            const std::vector<std::vector<double>>& F)
    {
        std::vector<std::vector<double>> out = R;
        if (F.empty()) return out;
        for (std::size_t k = 0; k < R.size(); ++k) {
            double nr = norm(R[k]);
            if (nr < EPS_) continue;                // numerical guard
            double best_perp = std::numeric_limits<double>::max();
            double best_proj = 0.0;
            for (const auto& f : F) {
                double proj  = dot(f, R[k]) / nr;   // ||F||·cos∠
                double perp2 = dot(f, f) - proj * proj;
                double perp  = std::sqrt(std::max(0.0, perp2));
                if (perp < best_perp) { best_perp = perp; best_proj = proj; }
            }
            best_proj = std::max(0.0, best_proj);   // guard: projection points backwards
            for (std::size_t j = 0; j < R[k].size(); ++j)
                out[k][j] = R[k][j] / nr * best_proj;
        }
        return out;
    }

    // ── Dominance / NDS ─────────────────────────────────────────────────

    static bool dom_plain(const std::vector<double>& fa, const std::vector<double>& fb) {
        bool strict = false;
        for (std::size_t k = 0; k < fa.size(); ++k) {
            if (fa[k] > fb[k]) return false;
            if (fa[k] < fb[k]) strict = true;
        }
        return strict;
    }

    bool dominates(DataVault<Ind_t>& vault, int a, int b) {
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            double ca = vault.get_cv(a), cb = vault.get_cv(b);
            bool af = (ca <= 0.0), bf = (cb <= 0.0);
            if (af && !bf) return true;
            if (!af && bf) return false;
            if (!af && !bf) return ca < cb;
        }
        return dom_plain(vault.objectives_of(a), vault.objectives_of(b));
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
        int k = 0;
        while (!cur.empty()) {
            for (int i : cur) vault.get_ind(i).rank = k;
            fronts.push_back(cur);
            std::vector<int> next;
            for (int i : cur) for (int j : S[i]) if (--ndom[j] == 0) next.push_back(j);
            cur = std::move(next);
            ++k;
        }
        return fronts;
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

    // ── MatingSelection (Alg.2) ─────────────────────────────────────────────
    std::vector<int> mating_selection(DataVault<Ind_t>& vault, int n) {
        const int m = vault.objs_n();
        // line 3: f_i(p) ← f_i(p) − min_{q∈P} f_i(q)
        std::vector<double> zmin(m, std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) zmin[j] = std::min(zmin[j], o[j]);
        }
        std::vector<std::vector<double>> Pt(n, std::vector<double>(m));
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) Pt[i][j] = o[j] - zmin[j];
        }
        // line 4: fitness per Eq.4
        std::vector<double> fit = igdns_drop_one(Pt, Rp_);
        // lines 6-11: binary tournament, the larger contribution wins
        std::uniform_int_distribution<int> di(0, n - 1);
        std::vector<int> mpool(n);
        for (int i = 0; i < n; ++i) {
            int a = di(rng_), b = di(rng_);
            if (constraint_mode == ConstraintMode::FEASIBILITY) {
                double ca = vault.get_cv(a), cb = vault.get_cv(b);
                bool af = (ca <= 0.0), bf = (cb <= 0.0);
                if (af != bf)        { mpool[i] = af ? a : b; continue; }
                if (!af && !bf)      { mpool[i] = (ca < cb) ? a : b; continue; }
            }
            mpool[i] = (fit[a] > fit[b]) ? a : b;
        }
        return mpool;
    }

    // ── RefPointAdaption (Alg.3) ────────────────────────────────────────────
    // The archive A persists in vault.archive_*; the input is A u O, with the
    // offspring [n, pool) appended to the archive here.
    void ref_point_adaption(DataVault<Ind_t>& vault, int n, int pool) {
        const int m  = vault.objs_n();
        const int NR = static_cast<int>(R0_.size());

        // Operation 1 (lines 1-8): z* and z^nad from P; translate A u P, scale R
        std::vector<double> zmin(m,  std::numeric_limits<double>::max());
        std::vector<double> znad(m, -std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) {
                zmin[j] = std::min(zmin[j], o[j]);
                znad[j] = std::max(znad[j], o[j]);
            }
        }
        for (int i = n; i < pool; ++i)                       // A ← A ∪ O
            vault.archive_push(static_cast<std::size_t>(i));
        const int na = static_cast<int>(vault.archive_size());
        std::vector<std::vector<double>> A(na, std::vector<double>(m));
        std::vector<double> Acv(na, 0.0);
        for (int i = 0; i < na; ++i) {
            const auto& o = vault.archive_objectives_of(i);
            for (int j = 0; j < m; ++j) A[i][j] = o[j] - zmin[j];
            if (constraint_mode == ConstraintMode::FEASIBILITY)
                Acv[i] = vault.archive_cv(i);
        }
        std::vector<std::vector<double>> Rs(NR, std::vector<double>(m));
        for (int k = 0; k < NR; ++k)
            for (int j = 0; j < m; ++j)
                Rs[k][j] = R0_[k][j] * (znad[j] - zmin[j]);

        // Operation 2 (lines 10-17): duplicates, dominated, A^con, top up A'
        std::vector<char> dead(na, 0);
        for (int i = 0; i < na; ++i) {                       // line 10: duplicates
            if (dead[i]) continue;
            for (int j = i + 1; j < na; ++j)
                if (!dead[j] && A[i] == A[j]) dead[j] = 1;
        }
        auto cdp_dom = [&](int a, int b) -> bool {           // line 11
            if (constraint_mode == ConstraintMode::FEASIBILITY) {
                bool af = (Acv[a] <= 0.0), bf = (Acv[b] <= 0.0);
                if (af && !bf) return true;
                if (!af && bf) return false;
                if (!af && !bf) return Acv[a] < Acv[b];
            }
            return dom_plain(A[a], A[b]);
        };
        for (int i = 0; i < na; ++i) {
            if (dead[i]) continue;
            for (int j = 0; j < na; ++j)
                // Line 11 runs on what line 10 left behind, so an entry already
                // removed as a duplicate must not act as a dominator. Inert
                // under ConstraintMode::NONE (a duplicate never strictly
                // dominates), but not under FEASIBILITY: there cdp_dom compares
                // constraint violations, and A[i] == A[j] only compares
                // objectives — so a killed duplicate with a smaller CV could
                // turn round and kill its own survivor.
                if (j != i && !dead[j] && cdp_dom(j, i)) { dead[i] = 1; break; }
        }
        std::vector<int> kept;                               // A after cleaning
        for (int i = 0; i < na; ++i) if (!dead[i]) kept.push_back(i);
        const int nk = static_cast<int>(kept.size());
        std::vector<std::vector<double>> AK(nk);
        for (int i = 0; i < nk; ++i) AK[i] = A[kept[i]];

        if (nk == 0) { Rp_ = Rs; return; }                   // safety net

        // line 12: R ← AdjustLocation(R, A)
        std::vector<std::vector<double>> Radj = adjust_location(Rs, AK);

        // line 13: A^con = the solutions nearest to at least one point of R
        std::vector<char> is_con(nk, 0);
        for (const auto& r : Radj) {
            int    arg = 0;
            double dm  = std::numeric_limits<double>::max();
            for (int i = 0; i < nk; ++i) {
                double d = edist(r, AK[i]);
                if (d < dm) { dm = d; arg = i; }
            }
            is_con[arg] = 1;
        }
        std::vector<int>  sel;                               // A' (indices into kept)
        std::vector<char> in_sel(nk, 0);
        for (int i = 0; i < nk; ++i)
            if (is_con[i]) { sel.push_back(i); in_sel[i] = 1; }

        // lines 15-17: top up to min(|R|,|A|) by max-min angle
        const int target_a = std::min(NR, nk);
        std::vector<double> minang(nk, std::numeric_limits<double>::max());
        for (int i = 0; i < nk; ++i)
            for (int s : sel)
                minang[i] = std::min(minang[i], acute_angle(AK[i], AK[s]));
        while (static_cast<int>(sel.size()) < target_a) {
            int best = -1;
            for (int i = 0; i < nk; ++i)
                if (!in_sel[i] && (best < 0 || minang[i] > minang[best])) best = i;
            if (best < 0) break;
            in_sel[best] = 1; sel.push_back(best);
            for (int i = 0; i < nk; ++i)
                minang[i] = std::min(minang[i], acute_angle(AK[i], AK[best]));
        }

        // Operation 3 (lines 19-24): R^valid -> R' -> top up from A' -> adjust on P
        std::vector<char> validw(Radj.size(), 0);            // line 19
        for (int i = 0; i < nk; ++i) {
            if (!is_con[i]) continue;
            int    arg = 0;
            double dm  = std::numeric_limits<double>::max();
            for (std::size_t r = 0; r < Radj.size(); ++r) {
                double d = edist(Radj[r], AK[i]);
                if (d < dm) { dm = d; arg = static_cast<int>(r); }
            }
            validw[arg] = 1;
        }
        Rp_.clear();                                         // line 20
        for (std::size_t r = 0; r < Radj.size(); ++r)
            if (validw[r]) Rp_.push_back(Radj[r]);

        const int ns = static_cast<int>(sel.size());         // lines 21-23
        const int target_r = std::min(NR, ns);
        std::vector<char>   used(ns, 0);
        std::vector<double> mang(ns, std::numeric_limits<double>::max());
        for (int i = 0; i < ns; ++i)
            for (const auto& r : Rp_)
                mang[i] = std::min(mang[i], acute_angle(r, AK[sel[i]]));
        while (static_cast<int>(Rp_.size()) < target_r) {
            int best = -1;
            for (int i = 0; i < ns; ++i)
                if (!used[i] && (best < 0 || mang[i] > mang[best])) best = i;
            if (best < 0) break;
            used[best] = 1;
            Rp_.push_back(AK[sel[best]]);                    // R' ∪ {F(p)}
            for (int i = 0; i < ns; ++i)
                mang[i] = std::min(mang[i], acute_angle(AK[sel[best]], AK[sel[i]]));
        }

        std::vector<std::vector<double>> Pt(n, std::vector<double>(m));
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) Pt[i][j] = o[j] - zmin[j];
        }
        Rp_ = adjust_location(Rp_, Pt);                      // line 24
        if (Rp_.empty()) Rp_ = Rs;                           // safety net

        // A <- A' in the vault archive: descending swap-erase of the
        // positions that did not survive
        std::vector<char> keep_pos(na, 0);
        for (int i : sel) keep_pos[kept[i]] = 1;
        for (int i = na - 1; i >= 0; --i)
            if (!keep_pos[i]) vault.archive_erase(static_cast<std::size_t>(i));
    }

    // ── EnvironmentalSelection (Alg.5) ──────────────────────────────────────
    void environmental_selection(DataVault<Ind_t>& vault, int pool, int N) {
        const int m = vault.objs_n();
        // line 3: f_i(p) <- f_i(p) - the min over the merged population
        std::vector<double> zmin(m, std::numeric_limits<double>::max());
        for (int i = 0; i < pool; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) zmin[j] = std::min(zmin[j], o[j]);
        }
        std::vector<std::vector<double>> Ft(pool, std::vector<double>(m));
        for (int i = 0; i < pool; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) Ft[i][j] = o[j] - zmin[j];
        }
        // lines 4-6: NDS, whole fronts up to the critical one
        auto fronts = fast_nds(vault, pool);
        std::vector<int> survivors, Fl;
        for (const auto& fr : fronts) {
            if (static_cast<int>(survivors.size() + fr.size()) <= N) {
                survivors.insert(survivors.end(), fr.begin(), fr.end());
                if (static_cast<int>(survivors.size()) == N) break;
            } else { Fl = fr; break; }
        }
        // lines 7-9: iteratively remove argmin IGD-NS(Front_k\{p}, R')
        if (static_cast<int>(survivors.size()) < N && !Fl.empty()) {
            const int slots = N - static_cast<int>(survivors.size());
            std::vector<int> alive = Fl;
            std::vector<std::vector<double>> X;
            while (static_cast<int>(alive.size()) > slots) {
                X.clear();
                X.reserve(alive.size());
                for (int v : alive) X.push_back(Ft[v]);
                std::vector<double> fit = igdns_drop_one(X, Rp_);
                int worst = 0;
                for (std::size_t i = 1; i < fit.size(); ++i)
                    if (fit[i] < fit[worst]) worst = static_cast<int>(i);
                alive.erase(alive.begin() + worst);
            }
            survivors.insert(survivors.end(), alive.begin(), alive.end());
        }
        rearrange(vault, survivors, pool);
    }

public:
    ARMOEACore() = default;

    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_pc  (double p)          { pc_ = p; }
    void set_pm  (double p)          { pm_ = p; }
    void set_seed(unsigned s)        { rng_.seed(s); }
    // Explicit reference-point count N_R (Table II); 0 selects the nearest
    // Das-Dennis lattice >= pop_size. Applied in setup and setup_seeded.
    void set_n_ref(int nr)           { n_ref_ = nr; }

    // IGD-NS(X, Y) per Eq.2-3 — a public indicator, used by the unit tests.
    static double igd_ns(const std::vector<std::vector<double>>& X,
                         const std::vector<std::vector<double>>& Y)
    {
        if (X.empty() || Y.empty()) return 0.0;
        const int nx = static_cast<int>(X.size());
        std::vector<char>   contrib(nx, 0);
        std::vector<double> dref(nx, std::numeric_limits<double>::max());
        double s = 0.0;
        for (const auto& y : Y) {
            int    arg = 0;
            double dm  = std::numeric_limits<double>::max();
            for (int x = 0; x < nx; ++x) {
                double d = edist(X[x], y);
                if (d < dm) { dm = d; arg = x; }
                if (d < dref[x]) dref[x] = d;
            }
            s += dm;                      // Σ_y min_x dis(y,x)
            contrib[arg] = 1;             // Eq.2: arg — contributing
        }
        for (int x = 0; x < nx; ++x)      // Σ_{x'∈X*} min_y dis(y,x')
            if (!contrib[x]) s += dref[x];
        return s;
    }

    // Alg.1 lines 1-4: P ← RandomInitialize(N); R ← UniformReferencePoint(N_R);
    // A ← P; R' ← R.
    void setup(DataVault<Ind_t>& vault) {
        const int n = vault.pop_size(), m = vault.objs_n();
        const int target = (n_ref_ > 0) ? n_ref_ : n;
        R0_ = das_dennis::generate_auto(m, target);
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
        vault.archive_clear();                               // A ← P
        for (int i = 0; i < n; ++i)
            vault.archive_push(static_cast<std::size_t>(i));
        Rp_ = R0_;                                           // R' ← R
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        const int n = vault.pop_size(), m = vault.objs_n();
        const int target = (n_ref_ > 0) ? n_ref_ : n;
        if (R0_.empty()) R0_ = das_dennis::generate_auto(m, target);
        vault.archive_clear();                               // A ← P
        for (int i = 0; i < n; ++i)
            vault.archive_push(static_cast<std::size_t>(i));
        Rp_ = R0_;                                           // R' ← R
    }

    // Alg.1 lines 6-9: one generation.
    void step(DataVault<Ind_t>& vault) {
        const int n = vault.pop_size(), N = n;
        const auto& bounds = vault.get_bounds();
        const double pm = (pm_ >= 0.0)
            ? pm_
            : (vault.vars_n() > 0 ? 1.0 / vault.vars_n() : 0.0);

        // line 6: P' ← MatingSelection(P, R')
        std::vector<int> mpool = mating_selection(vault, n);

        // line 7: O ← Variation(P', N)
        int off_base = vault.expand(n);
        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        for (int i = 0; i < n; i += 2) {
            int p1 = mpool[i], p2 = mpool[(i + 1) % n];
            for (int j = 0; j < vault.vars_n(); ++j) {
                pv1[j] = vault.get_variable(p1, j);
                pv2[j] = vault.get_variable(p2, j);
            }
            ops::sbx(pv1, pv2, c1, c2, bounds, eta_c_, pc_, rng_);
            ops::polynomial_mutation(c1, bounds, eta_m_, pm, rng_);
            if (i + 1 < n) ops::polynomial_mutation(c2, bounds, eta_m_, pm, rng_);
            if (vault.bin_vars_n() > 0) {
                std::vector<int> bv1(vault.bin_vars_n()), bv2(vault.bin_vars_n()),
                                 bc1, bc2;
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
        const int pool = off_base + n;

        // line 8: [A, R'] ← RefPointAdaption(A ∪ O, R, P)
        ref_point_adaption(vault, off_base, pool);

        // line 9: P ← EnvironmentalSelection(P ∪ O, R', N)
        environmental_selection(vault, pool, N);
    }
};

} // namespace mootation
