#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// NRV-MOEA — Many-objective EA guided by Normal Reference Vectors
//            (adaptive normal vector guided evolutionary algorithm)
// Y. Hua, Q. Liu, K. Hao — Complex & Intelligent Systems 10, 2024, 3709-3726
// doi:10.1007/s40747-024-01353-y
//
// Generational scheme (Alg.1):
//   1. Arc ← UpdateArchive(Arc, P): union, the non-dominated ones; when >N —
//      ε-indicator removal of the worst down to N (section Archive
//      management, [22]).
//   2. Q ← Reproduction(P ∪ Arc, N): parents from the union of the
//      population and the archive; SBX (p_c=1, η_c=20) + polynomial
//      mutation (p_m=1/V, η_m=20).
//   3. P_u = P ∪ Q (2N); fast NDS; accept whole fronts up to F^u=F_1∪…∪F_l.
//   4. Normalization: b_iN=(b_i−b_imin)/s; min/max — over the objective
//      values in F^u; s = b_imax − b_imin is frozen for f_r·MaxG
//      generations (f_r=0.1), b_imin is recomputed every generation.
//   5. Mapping (Alg.2): E ← extreme individuals of the ARCHIVE (max along
//      each objective); hyperplane A·x + D = 0 through E (solution of the
//      system, D=−1); projection of F_l along the normal A (Eq.1).
//   6. Ward-linkage clustering (Euclid) of the mapped points into N−N_n
//      clusters; Alg.3: cluster center (Eq.2), d1=‖mP−C‖, d2 = ±projection
//      distance («+» on the ideal side of H, «−» on the nadir side),
//      selection of min d=d1−d2.
//      (Second paper inconsistency: the framework text says ALL normalized
//      individuals are mapped and clustered into N clusters, while Alg.2 takes
//      F_l alone as its input and Alg.3 line 2 loops i = 1…N−N_n. The
//      Alg.2/Alg.3 reading is adopted; the two coincide only when l = 1.)
//   7. Pruning: P_c is compared against Arc — the non-dominated ones of
//      P_c ∪ Arc, when >N an ε-cull down to N → the new population P.
//      CAUTION: the paper is internally inconsistent — Alg.1 line 15 states
//      «P ← P_c» (no pruning), but the text THREE TIMES (Framework, Proposed
//      algorithm, Archive management) describes an ε-selection of P_c
//      against Arc into the population; the Alg.1 pseudocode is demonstrably
//      defective (line 8 «break»). The textual reading is adopted — a change
//      to the pseudocode's «P ← P_c» is deliberately NOT implemented.
//
// PAPER DEFAULTS = §Experimental settings: p_c=1, p_m=1/V, η_c=η_m=20,
//   f_r=0.1.
// ASSUMPTIONS (gaps in the paper, each filled deliberately):
//   (1) The archive ε-fitness formula is NOT in the paper. §"Archive
//       management" says only "we adopt the epsilon-indicator-based archive
//       management adopted in [22]" and then describes the procedure
//       (merge, keep non-dominated, delete the lowest ε-based fitness) without
//       the fitness itself. Note [22] is Zou, Zhang, Zheng & Yang (2021), KBS
//       231:107392 — a dominance-and-decomposition paper, not an IBEA one — so
//       it does not supply the formula either. This port uses the canonical
//       IBEA scheme F(x)=Σ −exp(−I_{ε+}(x',x)/(c·κ)) with κ=0.05 and
//       c=max|I_{ε+}|, the same form as ibea_eplus.hpp and two_arch2.hpp. The
//       paper's own discussion motivates the ε-indicator by pointing at
//       Two_Arch2, which is why that reading was preferred.
//   (2) The mechanism for drawing parents from P ∪ Arc is not specified —
//       uniformly at random here.
//   (3) For a degenerate system of extreme points, fall back to the intercept
//       plane Σ x_k/a_k = 1 with a_k the max of the normalized k-th coordinate
//       over the archive. The paper does not address degeneracy.
// DECLARED DEVIATIONS: topping the population up when <N survive pruning (by
//   the Σ of normalized objectives) — a guard of the fixed-N framework, the
//   case is not specified in the paper.
// EXTENSIONS BEYOND THE PAPER (off by default): ConstraintMode::FEASIBILITY
//   — CDP in NDS (fast_nds → dominates); binary variables.
//   Notable fix: this header used to promise CDP also in the «archive
//   non-dominance», but eps_archive_select (UpdateArchive and Pruning) uses
//   PURE dominance over the objectives (dom_obj, without CV) — CV/CDP is NOT
//   applied in the ε-management of the archive even under FEASIBILITY. The
//   comment has been brought in line with the fact; the code was not changed.
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

struct NRVMOEA_Individual : public Based_Individual {
    int rank = 0;
};

template <typename Ind_t>
class NRVMOEACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    static constexpr double KAPPA_ = 0.05;  // archive ε-fitness (assumption, [22])
    static constexpr double FR_    = 0.1;   // freeze period of s = f_r·MaxG

    double       eta_c_ = 20.0;
    double       eta_m_ = 20.0;
    double       pc_    = 1.0;
    int          t_max_ = 1000;
    int          current_gen_ = 0;
    std::vector<double> s_rng_;             // cache of s (frozen for f_r·MaxG)
    int          s_age_ = -1;               // when s was last updated
    std::mt19937 rng_{std::random_device{}()};

    // ── dominance (CDP) ────────────────────────────────────────────────────
    bool dom_obj(const std::vector<double>& fa, const std::vector<double>& fb) const {
        bool strict = false;
        for (std::size_t k = 0; k < fa.size(); ++k) {
            if (fa[k] > fb[k]) return false;
            if (fa[k] < fb[k]) strict = true;
        }
        return strict;
    }
    bool dominates(DataVault<Ind_t>& vault, int a, int b) const {
        if (constraint_mode == ConstraintMode::FEASIBILITY) {
            double ca = vault.get_cv(a), cb = vault.get_cv(b);
            bool af = (ca <= 0.0), bf = (cb <= 0.0);
            if (af && !bf) return true;
            if (!af && bf) return false;
            if (!af && !bf) return ca < cb;
        }
        return dom_obj(vault.objectives_of(a), vault.objectives_of(b));
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
            for (int i : cur) for (int j : S[i]) if (--ndom[j] == 0) next.push_back(j);
            cur = next;
        }
        return fronts;
    }

    static double eps_plus(const std::vector<double>& a, const std::vector<double>& b) {
        double w = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < a.size(); ++i) { double d = a[i]-b[i]; if (d > w) w = d; }
        return w;
    }
    static double edist(const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0.0; for (std::size_t j = 0; j < a.size(); ++j) { double t=a[j]-b[j]; s+=t*t; } return std::sqrt(s);
    }

    void rearrange(DataVault<Ind_t>& vault, const std::vector<int>& survivors, int pool_size) {
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

    // [code F] Ward-linkage agglomeration of mapped points → K clusters
    //          (Lance-Williams)
    std::vector<std::vector<int>>
    ward_cluster(const std::vector<int>& idx,
                 const std::vector<std::vector<double>>& MP, int K) {
        int n = static_cast<int>(idx.size());
        std::vector<std::vector<int>> cl(n);
        for (int i = 0; i < n; ++i) cl[i] = {idx[i]};
        if (n <= K) return cl;
        std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                D[i][j] = D[j][i] = edist(MP[idx[i]], MP[idx[j]]);
        std::vector<int> alive(n, 1), sz(n, 1);
        int count = n;
        while (count > K) {
            int ba = -1, bb = -1; double bd = std::numeric_limits<double>::max();
            for (int a = 0; a < n; ++a) {
                if (!alive[a]) continue;
                for (int b = a + 1; b < n; ++b) {
                    if (!alive[b]) continue;
                    if (D[a][b] < bd) { bd = D[a][b]; ba = a; bb = b; }
                }
            }
            if (ba < 0) break;
            int ni = sz[ba], nj = sz[bb];
            double dij = D[ba][bb];
            for (int c = 0; c < n; ++c) {
                if (!alive[c] || c == ba || c == bb) continue;
                int nk = sz[c];
                // Ward (Lance-Williams): d(i∪j,k)=sqrt(((ni+nk)d²ik+(nj+nk)d²jk−nk·d²ij)/(ni+nj+nk))
                double dik = D[ba][c], djk = D[bb][c];
                double num = (ni+nk)*dik*dik + (nj+nk)*djk*djk - nk*dij*dij;
                double nd = std::sqrt(std::max(0.0, num / (ni+nj+nk)));
                D[ba][c] = D[c][ba] = nd;
            }
            cl[ba].insert(cl[ba].end(), cl[bb].begin(), cl[bb].end());
            sz[ba] = ni + nj; alive[bb] = 0; --count;
        }
        std::vector<std::vector<int>> out;
        for (int i = 0; i < n; ++i) if (alive[i]) out.push_back(cl[i]);
        return out;
    }

    // [code E] archive ε-management: combine → non-dominated → (>N? ε-prune : keep)
    // Returns the indices of the survivors among items. Works in Fobj space.
    // NOTE: non-dominance here is PURE dom_obj over the objectives (without
    // CV/CDP), even under ConstraintMode::FEASIBILITY (see the file header).
    std::vector<int> eps_archive_select(const std::vector<std::vector<double>>& F,
                                        const std::vector<int>& items, int N) {
        // non-dominated among items
        std::vector<int> nd;
        for (int a : items) {
            bool dom = false;
            for (int b : items) if (a != b && dom_obj(F[b], F[a])) { dom = true; break; }
            if (!dom) nd.push_back(a);
        }
        if (static_cast<int>(nd.size()) <= N) return nd;   // «keep non-dominated only»
        // ε-prune down to N
        double c = 1e-12;
        for (int i : nd) for (int j : nd) if (i != j) c = std::max(c, std::abs(eps_plus(F[i], F[j])));
        std::vector<double> fit(F.size(), 0.0);
        std::vector<char> alive(F.size(), 0);
        for (int i : nd) alive[i] = 1;
        for (int i : nd) {
            double s = 0.0;
            for (int j : nd) if (i != j) s += -std::exp(-eps_plus(F[j], F[i])/(c*KAPPA_));
            fit[i] = s;
        }
        int rem = static_cast<int>(nd.size()) - N;
        for (int t = 0; t < rem; ++t) {
            int worst = -1; double wf = std::numeric_limits<double>::max();
            for (int i : nd) if (alive[i] && fit[i] < wf) { wf = fit[i]; worst = i; }
            if (worst < 0) break;
            alive[worst] = 0;
            for (int i : nd) if (alive[i] && i != worst) fit[i] += std::exp(-eps_plus(F[worst], F[i])/(c*KAPPA_));
        }
        std::vector<int> out;
        for (int i : nd) if (alive[i]) out.push_back(i);
        return out;
    }

    // Solution of the system E·A = 1 (hyperplane through m extreme points,
    // D = −1) by Gaussian elimination with partial pivoting. false → fallback.
    static bool solve_hyperplane(std::vector<std::vector<double>> E,
                                 std::vector<double>& A) {
        int m = static_cast<int>(E.size());
        std::vector<double> rhs(m, 1.0);
        for (int col = 0; col < m; ++col) {
            int piv = col;
            for (int r = col + 1; r < m; ++r)
                if (std::abs(E[r][col]) > std::abs(E[piv][col])) piv = r;
            if (std::abs(E[piv][col]) < 1e-10) return false;
            std::swap(E[piv], E[col]);
            std::swap(rhs[piv], rhs[col]);
            for (int r = col + 1; r < m; ++r) {
                double f = E[r][col] / E[col][col];
                for (int cc = col; cc < m; ++cc) E[r][cc] -= f * E[col][cc];
                rhs[r] -= f * rhs[col];
            }
        }
        A.assign(m, 0.0);
        for (int r = m - 1; r >= 0; --r) {
            double s = rhs[r];
            for (int cc = r + 1; cc < m; ++cc) s -= E[r][cc] * A[cc];
            A[r] = s / E[r][r];
        }
        for (double a : A)
            if (!std::isfinite(a) || a <= 0.0) return false;   // degenerate orientation
        return true;
    }

public:
    NRVMOEACore() = default;

    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_seed(unsigned s)        { rng_.seed(s); }
    void set_t_max(int t)            { t_max_ = (t > 0) ? t : 1; }

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
        // Arc ← P  (archive initialized with the initial population)  [code A]
        vault.archive_clear();
        for (int i = 0; i < n; ++i) vault.archive_push(i);
        s_age_ = -1; current_gen_ = 0;
    }

    void setup_seeded(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        if (vault.archive_size() == 0)
            for (int i = 0; i < n; ++i) vault.archive_push(i);
    }

    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int N = n, m = vault.objs_n();
        const auto& bounds = vault.get_bounds();
        ++current_gen_;

        // ── [code E] UpdateArchive(Arc, P): merge with P, ε-management ─────
        {
            int A = static_cast<int>(vault.archive_size());
            int total = A + n;
            std::vector<std::vector<double>> F(total);
            for (int i = 0; i < A; ++i) F[i] = vault.archive_objectives_of(i);
            for (int i = 0; i < n; ++i) F[A + i] = vault.objectives_of(i);
            std::vector<int> items(total); std::iota(items.begin(), items.end(), 0);
            auto keep = eps_archive_select(F, items, N);
            // rewrite the archive: keep vars/objs/lims of the non-dominated survivors
            std::vector<std::vector<double>> kv, ko, kl;
            std::vector<std::vector<int>> kb;
            for (int idx : keep) {
                if (idx < A) { kv.push_back(vault.archive_variables_of(idx));
                               ko.push_back(vault.archive_objectives_of(idx));
                               kl.push_back(vault.archive_limits_of(idx));
                               kb.push_back(vault.bin_vars_n() ? vault.archive_bin_variables_of(idx) : std::vector<int>()); }
                else { int pi = idx - A; std::vector<double> vv(vault.vars_n());
                       for (int j = 0; j < vault.vars_n(); ++j) vv[j] = vault.get_variable(pi, j);
                       kv.push_back(vv); ko.push_back(vault.objectives_of(pi));
                       kl.push_back(vault.limits_of(pi));
                       std::vector<int> bb(vault.bin_vars_n());
                       for (int j = 0; j < vault.bin_vars_n(); ++j) bb[j] = vault.get_bin_variable(pi, j);
                       kb.push_back(bb); }
            }
            vault.archive_clear();
            for (std::size_t i = 0; i < kv.size(); ++i)
                vault.archive_push_data(kv[i], ko[i], kb[i], kl[i]);
        }
        int A = static_cast<int>(vault.archive_size());

        // ── [code B] Q ← Reproduction(P ∪ Arc, N): SBX+PM, Alg.1 line 4 ────
        // Parents chosen uniformly from population [0,n) ∪ archive [0,A).
        std::uniform_int_distribution<int> dist_PA(0, n + A - 1);
        auto parent_vars = [&](int k, std::vector<double>& out) {
            int nv = vault.vars_n();
            out.resize(nv);
            if (k < n) for (int j = 0; j < nv; ++j) out[j] = vault.get_variable(k, j);
            else       out = vault.archive_variables_of(k - n);
        };
        auto parent_bvars = [&](int k, std::vector<int>& out) {
            int nb = vault.bin_vars_n();
            out.resize(nb);
            if (k < n) for (int j = 0; j < nb; ++j) out[j] = vault.get_bin_variable(k, j);
            else       out = vault.archive_bin_variables_of(k - n);
        };
        int off_base = vault.expand(n);
        int nv = vault.vars_n();
        double pm = (nv > 0) ? 1.0 / nv : 0.0;   // p_m = 1/V
        std::vector<double> pv1, pv2, c1, c2;
        for (int i = 0; i < n; i += 2) {
            int p1 = dist_PA(rng_), p2 = dist_PA(rng_);
            parent_vars(p1, pv1);
            parent_vars(p2, pv2);
            ops::sbx(pv1, pv2, c1, c2, bounds, eta_c_, pc_, rng_);
            ops::polynomial_mutation(c1, bounds, eta_m_, pm, rng_);
            ops::polynomial_mutation(c2, bounds, eta_m_, pm, rng_);
            if (vault.bin_vars_n() > 0) {
                std::vector<int> bv1, bv2, bc1, bc2;
                parent_bvars(p1, bv1);
                parent_bvars(p2, bv2);
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

        // ── NDS, accept whole fronts, critical front Fl ────────────────────
        auto fronts = fast_nds(vault, pool);
        std::vector<int> Pc;
        std::vector<int> Fl;
        for (const auto& fr : fronts) {
            if (static_cast<int>(Pc.size() + fr.size()) <= N) {
                for (int v : fr) Pc.push_back(v);
                if (static_cast<int>(Pc.size()) == N) break;
            } else { Fl = fr; break; }
        }

        // ── [code C] Normalization: min/max over F^u = F_1∪…∪F_l;
        //    s is frozen every f_r·MaxG generations, min — every gen ────────
        std::vector<int> Fu = Pc;
        for (int v : Fl) Fu.push_back(v);
        std::vector<double> fmin(m,  std::numeric_limits<double>::max());
        std::vector<double> fmax(m, -std::numeric_limits<double>::max());
        for (int v : Fu) {
            const auto& o = vault.objectives_of(v);
            for (int j = 0; j < m; ++j) {
                fmin[j] = std::min(fmin[j], o[j]);
                fmax[j] = std::max(fmax[j], o[j]);
            }
        }
        int s_period = std::max(1, static_cast<int>(FR_ * t_max_));
        if (s_age_ < 0 || (current_gen_ - s_age_) >= s_period ||
            static_cast<int>(s_rng_.size()) != m) {
            s_rng_.assign(m, 1.0);
            for (int j = 0; j < m; ++j)
                s_rng_[j] = (fmax[j] - fmin[j] > 1e-14) ? (fmax[j] - fmin[j]) : 1.0;
            s_age_ = current_gen_;
        }
        auto normalize = [&](const std::vector<double>& o) {
            std::vector<double> r(m);
            for (int j = 0; j < m; ++j) r[j] = (o[j] - fmin[j]) / s_rng_[j];
            return r;
        };
        std::vector<std::vector<double>> Fn(pool);
        for (int i = 0; i < pool; ++i) Fn[i] = normalize(vault.objectives_of(i));

        if (static_cast<int>(Pc.size()) < N && !Fl.empty()) {
            int kk = N - static_cast<int>(Pc.size());
            // ── [code D] Mapping (Alg.2): E ← Find Extreme Individuals(Arc);
            //    H from archive extremes (D = −1); fallback — intercepts ────
            std::vector<double> A_(m, 1.0);
            {
                std::vector<std::vector<double>> arcN(A);
                for (int i = 0; i < A; ++i)
                    arcN[i] = normalize(vault.archive_objectives_of(i));
                bool ok = false;
                if (A >= m) {
                    // extreme individual per objective — max of that objective in Arc
                    std::vector<std::vector<double>> E(m);
                    for (int j = 0; j < m; ++j) {
                        int best = 0;
                        for (int i = 1; i < A; ++i)
                            if (arcN[i][j] > arcN[best][j]) best = i;
                        E[j] = arcN[best];
                    }
                    ok = solve_hyperplane(E, A_);
                }
                if (!ok) {
                    // fallback: intercept plane Σ x_k/a_k = 1 from max over Arc
                    std::vector<double> amax(m, 1e-12);
                    for (int i = 0; i < A; ++i)
                        for (int j = 0; j < m; ++j)
                            amax[j] = std::max(amax[j], arcN[i][j]);
                    for (int j = 0; j < m; ++j) A_[j] = 1.0 / std::max(amax[j], 1e-6);
                }
            }
            double AA = 0.0; for (int j = 0; j < m; ++j) AA += A_[j]*A_[j];
            double normA = std::sqrt(AA);
            std::vector<std::vector<double>> MP(pool, std::vector<double>(m, 0.0));
            std::vector<double> d2(pool, 0.0);
            for (int v : Fl) {
                double Ab = 0.0; for (int j = 0; j < m; ++j) Ab += A_[j]*Fn[v][j];
                double t = (1.0 - Ab) / AA;                    // Eq.1: A·b + D = Ab − 1
                for (int j = 0; j < m; ++j) MP[v][j] = Fn[v][j] + A_[j]*t;
                // signed projection distance: «+» if b is on the ideal side (A·b<1)
                d2[v] = (1.0 - Ab) / normA;
            }
            // [code F] Ward → kk clusters over the mapped points
            auto clusters = ward_cluster(Fl, MP, kk);
            // [code G] Alg.3: cluster center (Eq.2), min d=d1−d2
            for (auto& cl : clusters) {
                std::vector<double> C(m, 0.0);
                for (int v : cl) for (int j = 0; j < m; ++j) C[j] += MP[v][j];
                for (int j = 0; j < m; ++j) C[j] /= cl.size();
                int best = cl[0]; double bd = std::numeric_limits<double>::max();
                for (int v : cl) {
                    double d1 = edist(MP[v], C);
                    double d = d1 - d2[v];
                    if (d < bd) { bd = d; best = v; }
                }
                Pc.push_back(best);
            }
        }
        if (static_cast<int>(Pc.size()) > N) Pc.resize(N);

        // ── [code E] Pruning: Pc against Arc, ε-cull → new population P ────
        // (the textual reading of the paper; see the file header)
        std::vector<int> finalP;
        {
            int total = static_cast<int>(Pc.size()) + A;
            std::vector<std::vector<double>> F(total);
            std::vector<int> src(total);   // >=0: pool-index; <0: −(arch+1)
            for (std::size_t i = 0; i < Pc.size(); ++i) { F[i] = vault.objectives_of(Pc[i]); src[i] = Pc[i]; }
            for (int i = 0; i < A; ++i) { F[Pc.size()+i] = vault.archive_objectives_of(i); src[Pc.size()+i] = -(i+1); }
            std::vector<int> items(total); std::iota(items.begin(), items.end(), 0);
            auto keep = eps_archive_select(F, items, N);
            // materialize the survivors into the active slots [0,N) via
            // seed_individual (objs are known — no repeated evaluations).
            std::vector<std::vector<double>> kv, ko, kl;
            std::vector<std::vector<int>> kb;
            for (int it : keep) {
                int s = src[it];
                if (s >= 0) { std::vector<double> vv(vault.vars_n());
                              for (int j = 0; j < vault.vars_n(); ++j) vv[j] = vault.get_variable(s, j);
                              kv.push_back(vv); ko.push_back(vault.objectives_of(s));
                              kl.push_back(vault.limits_of(s));
                              std::vector<int> bb(vault.bin_vars_n());
                              for (int j = 0; j < vault.bin_vars_n(); ++j) bb[j] = vault.get_bin_variable(s, j);
                              kb.push_back(bb); }
                else { int ai = -s - 1; kv.push_back(vault.archive_variables_of(ai));
                       ko.push_back(vault.archive_objectives_of(ai));
                       kl.push_back(vault.archive_limits_of(ai));
                       kb.push_back(vault.bin_vars_n() ? vault.archive_bin_variables_of(ai) : std::vector<int>()); }
            }
            // framework guard: top up to N from the rest of the pool, by the
            // Σ of normalized objectives
            if (static_cast<int>(kv.size()) < N) {
                std::vector<char> used(pool, 0);
                for (int v : Pc) used[v] = 1;
                std::vector<int> rest;
                for (int v = 0; v < pool; ++v) if (!used[v]) rest.push_back(v);
                std::sort(rest.begin(), rest.end(), [&](int a, int b){
                    double na=0,nb=0; for(int j=0;j<m;++j){na+=Fn[a][j];nb+=Fn[b][j];} return na<nb; });
                for (int v : rest) {
                    if (static_cast<int>(kv.size()) >= N) break;
                    std::vector<double> vv(vault.vars_n());
                    for (int j = 0; j < vault.vars_n(); ++j) vv[j] = vault.get_variable(v, j);
                    kv.push_back(vv); ko.push_back(vault.objectives_of(v));
                    kl.push_back(vault.limits_of(v));
                    std::vector<int> bb(vault.bin_vars_n());
                    for (int j = 0; j < vault.bin_vars_n(); ++j) bb[j] = vault.get_bin_variable(v, j);
                    kb.push_back(bb);
                }
            }
            int kept = static_cast<int>(kv.size());
            for (int i = 0; i < kept; ++i) {
                vault.seed_individual(i, kv[i], ko[i], kb[i], kl[i]);
                finalP.push_back(i);
            }
        }
        // bring active_ to exactly |finalP| (slots [0,·) already hold the new population)
        std::vector<int> keepN(finalP.size());
        std::iota(keepN.begin(), keepN.end(), 0);
        rearrange(vault, keepN, pool);
    }
};

} // namespace mootation
