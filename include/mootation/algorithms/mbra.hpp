#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MBRA — A Many-objective Evolutionary Algorithm with Metric-Based
//   Reference Vector Adjustment
// X. Wang, F. Zhang, M. Yao — Complex & Intelligent Systems 10 (2024) 207-231
//   (2024 issue; online 2023)
// doi:10.1007/s40747-023-01161-w        (source: s40747-023-01161-w)
//
// Generation scheme (Algorithm 1):
//   1. MatingSelection: two random individuals; Pareto dominance → smaller Σf_j →
//      random; repeat until N parents.
//   2. Variation: SBX (p_c=1) + PM (p_m=1/D) → N offspring; pool P∪P'' (2N).
//   3. EnvironmentalSelection (Alg.2): normalisation (3) over the pool; fitness
//      F(x1)=Σ−exp(−I_{ε+}(x2,x1)/0.05) (4), larger is better; s2v association
//      by d=1−cos (5); FitNo within a subregion (larger F — smaller FitNo);
//      NDS → fronts until ≥N; if >N — complete FitNo layers, the critical
//      layer — max-min-angle (diversity).
//   4. Condition1: G ∈ [20%,90%]·Gmax AND G divisible by fr·Gmax (fr=0.1) → CM =
//      CalConvergence (Alg.3: v2s association, CM[j]=d1 projection (F'·w)/‖w‖);
//      IMR=(CM−CM_old)/CM_old (6); converted to ±1/0 with threshold α=0.01 (7).
//   5. Condition2: Σ of converted IMR ≥ 0 → AdjustVector (Alg.4): delete
//      RVs with ρ=0; add until |W|=N: w_new=(f(x_f)−z_min)/Σ(f_k(x_f)−z_min_k)
//      (8) from the farthest solution x_f in the subregion of the most crowded
//      RV; re-associate P∖{x_f}; then CM_old ← recomputed CalConvergence.
//
// Defaults = §"Parameter settings"(2): p_c=1, p_m=1/D, η_c=η_m=20; α=0.01,
//   fr=0.1; W — Das-Dennis two-layer, |W| = N.
// Gmax CONTRACT (M1): the Condition1 schedule is
//   defined relative to the ACTUAL generation budget (paper: Gmax = FEmax/N);
//   set_t_max(Gmax) is MANDATORY before setup()/setup_seeded(), otherwise a
//   std::logic_error is thrown. The silent default (previously 1000) on short
//   runs disabled the adjustment window entirely (G < 0.2·1000), i.e. it
//   silently degenerated MBRA into an algorithm with fixed RVs.
// DECLARED DEVIATIONS AND INTERPRETATIONS:
//   (a) M2: the Condition1 bounds are INCLUSIVE (G=⌊0.2·Gmax⌋ and G=⌊0.9·Gmax⌋
//       pass); the paper does not specify the strictness of the bounds;
//   (b) M3: guard CM_old≈0 → IMR=0 (the paper does not address division by 0);
//   (c) M4: max-min-angle with an empty selected set: the first individual is
//       the first by index (all dmin=0); the paper does not define this case;
//   (d) M5: degenerate range z_max=z_min → f'=0; a zero vector in
//       angle_dist → d=1 — numerical guards beyond the paper;
//   (e) M6: the protective exits of AdjustVector (rmax=0 / Nref=0) are normally
//       unreachable when n=N;
//   (f) M7: CM_old is refreshed at EVERY Condition1 trigger, not only after an
//       AdjustVector. The paper never assigns CM_old in pseudocode — Eq.(6)
//       defines it only as "the convergence metric of the last computation",
//       and Alg.1 line 8 necessarily performs a CalConvergence at every
//       trigger, so "the last computation" is read as that one.
// Extensions beyond the paper: constraint_mode FEASIBILITY (CDP in NDS/dominance),
//   binary variables (off by default).
// ============================================================================
// FIX 2026-06: (1) M1: set_t_max is mandatory (see Gmax CONTRACT);
// (2) RNG seed — std::random_device + set_seed (was time(nullptr)); (3) explicit
// p_c/p_m in the new SBX/PM signatures; (4) MINOR M2–M6 documented above.

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
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

struct MBRA_Individual : public Based_Individual {
    int rank = 0;
};

template <typename Ind_t>
class MBRACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    static constexpr double KAPPA_ = 0.05;

    double       eta_c_ = 20.0;   // §"Parameter settings": η_c = 20
    double       eta_m_ = 20.0;   // §"Parameter settings": η_m = 20
    double       pc_    = 1.0;    // §"Parameter settings": p_c = 1
    double       fr_    = 0.1;
    double       alpha_ = 0.01;
    int          t_max_ = 0;      // Gmax; 0 = NOT SET (see Gmax CONTRACT)
    int          current_gen_ = 0;
    std::mt19937 rng_{std::random_device{}()};

    void require_t_max() const {
        if (t_max_ <= 0)
            throw std::logic_error("MBRA: set_t_max(Gmax) must be called before setup()/"
                "setup_seeded(). Condition1 (adjustment window 20%-90% of "
                "Gmax, period fr*Gmax) is defined relative to the ACTUAL "
                "generation budget (paper: Gmax = FEmax/N); a silent default "
                "would shift or disable the adjustment schedule.");
    }

    std::vector<std::vector<double>> W_;      // reference vectors
    std::vector<double>              CM_old_; // convergence metric (per RV)

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

    // ── Normalisation (3) over indices → Fn (by position in the pool) ──────
    std::vector<std::vector<double>>
    normalise(DataVault<Ind_t>& vault, const std::vector<int>& idx, int pool,
              std::vector<double>* zmin_out = nullptr) {
        int m = vault.objs_n();
        std::vector<double> zmin(m,  std::numeric_limits<double>::max());
        std::vector<double> zmax(m, -std::numeric_limits<double>::max());
        for (int v : idx) {
            const auto& o = vault.objectives_of(v);
            for (int j = 0; j < m; ++j) {
                zmin[j] = std::min(zmin[j], o[j]);
                zmax[j] = std::max(zmax[j], o[j]);
            }
        }
        std::vector<std::vector<double>> fn(pool, std::vector<double>(m, 0.0));
        for (int v : idx) {
            const auto& o = vault.objectives_of(v);
            for (int j = 0; j < m; ++j) {
                double range = zmax[j] - zmin[j];
                fn[v][j] = (range > 1e-14) ? (o[j] - zmin[j]) / range : 0.0;
            }
        }
        if (zmin_out) *zmin_out = zmin;
        return fn;
    }

    static double eps_plus(const std::vector<double>& a, const std::vector<double>& b) {
        double w = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < a.size(); ++i) { double d = a[i] - b[i]; if (d > w) w = d; }
        return w;
    }

    static double vnorm(const std::vector<double>& f) {
        double s = 0.0; for (double x : f) s += x * x; return std::sqrt(s);
    }

    // ── Angular distance d=1−cos(a,b) (Eq.5) ───────────────────────────────
    static double angle_dist(const std::vector<double>& a, const std::vector<double>& b) {
        double dot = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) dot += a[j] * b[j];
        double denom = vnorm(a) * vnorm(b);
        if (denom < 1e-30) return 1.0;
        double c = dot / denom;
        c = std::max(-1.0, std::min(1.0, c));
        return 1.0 - c;
    }

    // ── d1 projection of f' onto w (convergence) ───────────────────────────
    static double d1_proj(const std::vector<double>& f, const std::vector<double>& w) {
        double dot = 0.0;
        for (std::size_t j = 0; j < f.size(); ++j) dot += f[j] * w[j];
        double nw = vnorm(w);
        return (nw > 1e-30) ? dot / nw : 0.0;
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

    // ── Algorithm 3: CalConvergence(P, W) → CM (per RV) ────────────────────
    std::vector<double> cal_convergence(DataVault<Ind_t>& vault, int n) {
        std::vector<int> P(n);
        std::iota(P.begin(), P.end(), 0);
        auto Fn = normalise(vault, P, n);
        int Nref = static_cast<int>(W_.size());
        std::vector<double> CM(Nref, 0.0);
        for (int j = 0; j < Nref; ++j) {
            int    best = -1; double bd = std::numeric_limits<double>::max();
            for (int s = 0; s < n; ++s) {           // v2s: closest solution to the RV
                double d = angle_dist(Fn[s], W_[j]);
                if (d < bd) { bd = d; best = s; }
            }
            CM[j] = (best >= 0) ? d1_proj(Fn[best], W_[j]) : 0.0;
        }
        return CM;
    }

    // ── Algorithm 4: AdjustVector(P, W) — delete invalid + add crowded ─────
    void adjust_vector(DataVault<Ind_t>& vault, int n, int N) {
        int m = vault.objs_n();
        std::vector<int> P(n);
        std::iota(P.begin(), P.end(), 0);
        std::vector<double> zmin;
        auto Fn = normalise(vault, P, n, &zmin);

        // Step 1: delete invalid RVs (rho=0).
        {
            std::vector<int> rho(W_.size(), 0);
            for (int s = 0; s < n; ++s) {
                int    bj = 0; double bd = std::numeric_limits<double>::max();
                for (int j = 0; j < static_cast<int>(W_.size()); ++j) {
                    double d = angle_dist(Fn[s], W_[j]);
                    if (d < bd) { bd = d; bj = j; }
                }
                ++rho[bj];
            }
            std::vector<std::vector<double>> kept;
            for (int j = 0; j < static_cast<int>(W_.size()); ++j)
                if (rho[j] > 0) kept.push_back(W_[j]);
            W_.swap(kept);
        }

        // Step 2: add one at a time until |W|=N.
        std::vector<char> used(n, 0);
        while (static_cast<int>(W_.size()) < N) {
            int Nref = static_cast<int>(W_.size());
            if (Nref == 0) break;
            std::vector<int> rho(Nref, 0);
            std::vector<int> assoc(n, -1);
            for (int s = 0; s < n; ++s) {
                if (used[s]) continue;
                int    bj = 0; double bd = std::numeric_limits<double>::max();
                for (int j = 0; j < Nref; ++j) {
                    double d = angle_dist(Fn[s], W_[j]);
                    if (d < bd) { bd = d; bj = j; }
                }
                assoc[s] = bj; ++rho[bj];
            }
            // the most crowded RV (random on ties)
            int rmax = 0;
            for (int j = 0; j < Nref; ++j) rmax = std::max(rmax, rho[j]);
            if (rmax == 0) break;
            std::vector<int> ties;
            for (int j = 0; j < Nref; ++j) if (rho[j] == rmax) ties.push_back(j);
            int wmost = ties[std::uniform_int_distribution<int>(0, static_cast<int>(ties.size()) - 1)(rng_)];
            // the farthest solution in the subregion of wmost (max angular distance)
            int    xf = -1; double far = -1.0;
            for (int s = 0; s < n; ++s) {
                if (used[s] || assoc[s] != wmost) continue;
                double d = angle_dist(Fn[s], W_[wmost]);
                if (d > far) { far = d; xf = s; }
            }
            if (xf < 0) break;
            // new RV (8): (f(x_f)−zmin) normalised to sum 1.
            const auto& o = vault.objectives_of(xf);
            std::vector<double> wnew(m, 0.0);
            double sum = 0.0;
            for (int j = 0; j < m; ++j) { wnew[j] = std::max(0.0, o[j] - zmin[j]); sum += wnew[j]; }
            if (sum > 1e-30) for (int j = 0; j < m; ++j) wnew[j] /= sum;
            else             wnew = Fn[xf];
            W_.push_back(wnew);
            used[xf] = 1;
        }
    }

public:
    MBRACore() = default;

    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_pc           (double p) { pc_   = p; }
    // MANDATORY before setup()/setup_seeded() (see Gmax CONTRACT in the header).
    void set_t_max(int t)            { t_max_ = (t > 0) ? t : 0; }
    void set_seed(unsigned s)        { rng_.seed(s); }

    void setup(DataVault<Ind_t>& vault) {
        require_t_max();   // FIX 2026-06 (M1): the Condition1 schedule derives from Gmax
        int n = vault.pop_size(), m = vault.objs_n();
        W_ = das_dennis::generate_exact(m, n);
        current_gen_ = 0;
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
        CM_old_ = cal_convergence(vault, n);
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        require_t_max();   // FIX 2026-06 (M1): the Condition1 schedule derives from Gmax
        int n = vault.pop_size(), m = vault.objs_n();
        if (W_.empty()) W_ = das_dennis::generate_exact(m, n);
        current_gen_ = 0;
        CM_old_ = cal_convergence(vault, n);
    }

    // ── step: one generation (Algorithm 1) ─────────────────────────────────
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int N = n;
        int m = vault.objs_n();
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_N(0, n - 1);
        ++current_gen_;

        // ── MatingSelection (dominance + objective sum) → N parents ──────────
        std::vector<int> parents;
        parents.reserve(n);
        for (int t = 0; t < n; ++t) {
            int s1 = dist_N(rng_), s2 = dist_N(rng_);
            int win;
            if (dominates(vault, s1, s2))      win = s1;
            else if (dominates(vault, s2, s1)) win = s2;
            else {
                double sum1 = 0.0, sum2 = 0.0;
                for (double v : vault.objectives_of(s1)) sum1 += v;
                for (double v : vault.objectives_of(s2)) sum2 += v;
                if      (sum1 < sum2) win = s1;
                else if (sum2 < sum1) win = s2;
                else win = (std::uniform_real_distribution<double>(0.0,1.0)(rng_) < 0.5) ? s1 : s2;
            }
            parents.push_back(win);
        }

        // ── Variation → offspring [N, 2N) ──────────────────────────────────
        int off_base = vault.expand(n);
        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        // §"Parameter settings": p_c = 1, p_m = 1/D.
        double pm = (vault.vars_n() > 0) ? 1.0 / vault.vars_n() : 0.0;
        for (int i = 0; i < n; i += 2) {
            int p1 = parents[i], p2 = parents[(i + 1) % n];
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

        // ── EnvironmentalSelection (Algorithm 2) ───────────────────────────
        std::vector<int> U(pool);
        std::iota(U.begin(), U.end(), 0);
        auto Fn = normalise(vault, U, pool);

        // fitness F(x) (4): larger is better.
        std::vector<double> F(pool, 0.0);
        for (int i = 0; i < pool; ++i) {
            double s = 0.0;
            for (int j = 0; j < pool; ++j) {
                if (i == j) continue;
                double ind = eps_plus(Fn[j], Fn[i]);     // I_{ε+}(x2=j, x1=i)
                s += -std::exp(-ind / KAPPA_);
            }
            F[i] = s;
        }

        // association (s2v) + FitNo within a subregion (larger F → FitNo=0).
        int Nref = static_cast<int>(W_.size());
        std::vector<std::vector<int>> sub(Nref);
        for (int i = 0; i < pool; ++i) {
            int    bj = 0; double bd = std::numeric_limits<double>::max();
            for (int j = 0; j < Nref; ++j) {
                double d = angle_dist(Fn[i], W_[j]);
                if (d < bd) { bd = d; bj = j; }
            }
            sub[bj].push_back(i);
        }
        std::vector<int> FitNo(pool, 0);
        for (auto& sr : sub) {
            std::sort(sr.begin(), sr.end(), [&](int a, int b){ return F[a] > F[b]; });
            for (int r = 0; r < static_cast<int>(sr.size()); ++r) FitNo[sr[r]] = r;
        }

        // NDS → take fronts until ≥N.
        auto fronts = fast_nds(vault, pool);
        std::vector<int> cand;
        for (const auto& fr : fronts) {
            for (int v : fr) cand.push_back(v);
            if (static_cast<int>(cand.size()) >= N) break;
        }

        std::vector<int> survivors;
        if (static_cast<int>(cand.size()) == N) {
            survivors = cand;
        } else {
            // layers by FitNo within cand.
            int maxfit = 0;
            for (int v : cand) maxfit = std::max(maxfit, FitNo[v]);
            std::vector<std::vector<int>> FitF(maxfit + 1);
            for (int v : cand) FitF[FitNo[v]].push_back(v);

            survivors.reserve(N);
            int kf = 0;
            // complete layers FitF_1..k-1
            for (; kf <= maxfit; ++kf) {
                if (static_cast<int>(survivors.size() + FitF[kf].size()) >= N) break;
                for (int v : FitF[kf]) survivors.push_back(v);
            }
            if (static_cast<int>(survivors.size()) < N && kf <= maxfit) {
                // critical layer FitF[kf]: max-min-angle until N.
                std::vector<int> rem = FitF[kf];
                std::vector<char> taken(rem.size(), 0);
                while (static_cast<int>(survivors.size()) < N && !rem.empty()) {
                    int best = -1; double bestd = -1.0; int bidx = -1;
                    for (int t = 0; t < static_cast<int>(rem.size()); ++t) {
                        if (taken[t]) continue;
                        int xq = rem[t];
                        double dmin = std::numeric_limits<double>::max();
                        for (int xp : survivors)
                            dmin = std::min(dmin, angle_dist(Fn[xq], Fn[xp]));
                        if (survivors.empty()) dmin = 0.0;
                        if (dmin > bestd) { bestd = dmin; best = xq; bidx = t; }
                    }
                    if (best < 0) break;
                    survivors.push_back(best); taken[bidx] = 1;
                }
            }
        }

        rearrange(vault, survivors, pool);

        // ── Reference vector adjustment (Algorithm 1: Condition1/2) ────────
        int G = current_gen_;
        int period = std::max(1, static_cast<int>(std::lround(fr_ * t_max_)));
        bool cond1 = (G >= static_cast<int>(0.2 * t_max_)) &&
                     (G <= static_cast<int>(0.9 * t_max_)) &&
                     (G % period == 0);
        if (cond1) {
            std::vector<double> CM = cal_convergence(vault, N);
            // Both sizes are bounded by the population size, so the narrowing
            // is safe; make it explicit rather than let the compiler warn.
            int sz = static_cast<int>(std::min(CM.size(), CM_old_.size()));
            int conv_sum = 0;
            for (int j = 0; j < sz; ++j) {
                double base = CM_old_[j];
                double imr = (std::abs(base) > 1e-30) ? (CM[j] - base) / base : 0.0;
                conv_sum += (imr < -alpha_) ? -1 : (imr > alpha_ ? 1 : 0);
            }
            if (conv_sum >= 0) {                       // Condition2 → adjust
                adjust_vector(vault, N, N);
                CM = cal_convergence(vault, N);
            }
            CM_old_ = CM;
        }
        (void)m;
    }
};

} // namespace mootation
