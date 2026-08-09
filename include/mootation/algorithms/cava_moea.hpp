#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// CAVA-MOEA — A clustering and vector angle-based adaptive evolutionary
//   algorithm for multi-objective optimization with irregular Pareto fronts
// M. He, H. Zheng, H. Chen, Z. Wang, X. Liu, Y. Xia, H. Wang —
//   The Journal of Supercomputing 81:98 (2025 issue; online 2024)
// doi:10.1007/s11227-024-06496-w
//
// Generational scheme (Algorithm 2 + Algorithm 3):
//   1. Q <- SBX + polynomial mutation (|Q| = N, random pairs); S = P u Q (2N).
//   2. Normalization Eq.(3) over the whole pool S; Fit(x) = Sum f~_j Eq.(4);
//      norm(x) = ||f~(x)|| Eq.(5); fast non-dominated sort (CDP under
//      FEASIBILITY).
//   3. Accept fronts F_1..F_{l-1} (< N); if |F_l| closes the gap at exactly N,
//      take the whole front; otherwise Algorithm 4 selects K = N−|P|
//      individuals from F_l.
//   4. Alg.4, stage 1: Ward clustering of F_l into K clusters (Eq.1, centres
//      Eq.2); from each non-empty cluster take the individual nearest its
//      centre (tie-break by minimum Fit, lines 10–13); clusters of density >= 3
//      go to SP, those below 3 to GP.
//   5. Alg.4 lines 22–29: if |P| + |SP| > N, run Correlate_solution (Alg.5) and
//      Vector_Angle-Based_Selection (Alg.6: max-angle-first, N_S = N−|P|
//      additions; on every iteration Alg.7 replaces similar poor solutions when
//      theta(x_mu) < (pi/2)/(N+1) and Fit(y_r) > Fit(x_mu)); otherwise
//      P = P u SP u the best-Fit members of GP.
//
// PAPER DEFAULTS (§5.1): p_c=1.0, p_m=1/V, eta_c=20, eta_m=20; the Alg.7
//   threshold is (pi/2)/(N+1).
// DECLARED DEVIATIONS:
//   CAVA-2. The Alg.7 replacement admits any y_r in P. Alg.5 line 4 iterates
//     "for each y_k ∈ P" and P here already holds the accepted fronts, so y_r
//     is not restricted to the critical front and a replacement may evict a
//     member of F_{l-1}. Strict elitism relative to Alg.3 is therefore NOT
//     enforced. This follows the letter of Alg.5/6/7 and reproduces the VaEA
//     [13] behaviour the paper builds on; it is a deliberate choice, not an
//     oversight.
// EXTENSION BEYOND THE PAPER:
//   A safety top-up to exactly N — see (b) below.
// LITERAL READINGS (verbatim from the pseudocode; NOT deviations):
//   theta(x_i) := angle in case (2) of Alg.7 is Alg.7 line 16 as printed —
//     the prose gives the reason (after a replacement x_mu and y_r are taken
//     as collinear), even though Alg.5 defines theta as a minimum angle
//     elsewhere; theta/gamma are maintained incrementally per Alg.5/6/7; the
//     |P|+|SP| > N branch is Alg.4 as written.
// NOTABLE FIXES (history, kept so the current shape is legible):
//   CAVA-1 — the population could shrink below N; the top-up in (b) fixed it.
//   CAVA-2 — y_r used to be restricted to the critical front and theta used to
//     be recomputed exactly; both were changed to follow Alg.5/6/7 literally.
// READINGS of points the paper leaves unspecified:
//   (a) The stage-1 tie-break uses exact floating-point equality of the
//       distances; the paper's intent ("solutions closest ... not unique") is
//       covered formally.
//   (b) Alg.6 can under-fill (x_rho becomes null once SP is exhausted by
//       replacements); the paper does not address |P| < N on output. A top-up
//       with the best by Fit from the unused members of F_l, then from the
//       pool, brings it to exactly N. This is beyond the paper and normally
//       does not fire.
//   (c) The paper does not specify mating selection; random pairs are used
//       (PlatEMO practice).
//
// Formulas:
//   (3) f~_j = (f_j − z_j^min)/(z_j^max − z_j^min)   over the pool S
//   (4) Fit(x) = Sum_j f~_j(x)
//   (5) norm(x) = sqrt(Sum_j f~_j(x)^2)
//   (6) angle(x,y) = arccos(|F'(x)·F'(y)| / (norm(x)·norm(y)))
//   Ward merge increment for r,s: (n_r*n_s)/(n_r+n_s)*||C_r−C_s||^2 (equivalent
//   to Eq.1)
//
// EXTENSIONS BEYOND THE PAPER: binary variables (binary_crossover / bit_flip),
//   active only when bin_vars_n() > 0.
// FEASIBILITY: the fast non-dominated sort uses CDP; normalization and the
//   angle work on the objectives.
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

// ── CAVA-MOEA individual ────────────────────────────────────────────────────
// rank is the non-dominated front index, assigned by the fast NDS.
struct CAVAMOEA_Individual : public Based_Individual {
    int rank = 0;
};

template <typename Ind_t>
class CAVAMOEACore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    static constexpr double PI_ = 3.14159265358979323846;

    double       eta_c_ = 20.0;   // §5.1: n_c = 20
    double       eta_m_ = 20.0;   // §5.1: n_m = 20
    double       pc_    = 1.0;    // §5.1: p_c = 1.0
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
            cur = next; ++r;
        }
        return fronts;
    }

    static double sq_dist(const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) { double d = a[j] - b[j]; s += d * d; }
        return s;
    }

    // ── Angle between normalized vectors (Eq.6) ────────────────────────────
    static double angle(const std::vector<double>& fa, double na,
                        const std::vector<double>& fb, double nb) {
        double dot = 0.0;
        for (std::size_t j = 0; j < fa.size(); ++j) dot += fa[j] * fb[j];
        double denom = na * nb;
        if (denom < 1e-30) return PI_ / 2.0;
        double c = std::abs(dot) / denom;
        c = std::max(-1.0, std::min(1.0, c));
        return std::acos(c);
    }

    // ── Ward clustering of the normalized points Y -> K centroids (Eq.1–2) ──
    std::vector<std::vector<double>>
    ward_centers(const std::vector<std::vector<double>>& Y, int K) {
        int n = static_cast<int>(Y.size());
        int m = (n > 0) ? static_cast<int>(Y[0].size()) : 0;
        std::vector<std::vector<double>> cen = Y;
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
                    double w = (static_cast<double>(sz[a]) * sz[b]) / (sz[a] + sz[b])
                             * sq_dist(cen[a], cen[b]);
                    if (w < best) { best = w; ba = a; bb = b; }
                }
            }
            if (ba < 0) break;
            int na = sz[ba], nb = sz[bb];
            for (int j = 0; j < m; ++j)
                cen[ba][j] = (cen[ba][j] * na + cen[bb][j] * nb) / (na + nb);
            sz[ba] = na + nb; alive[bb] = 0; --nclust;
        }
        std::vector<std::vector<double>> C;
        for (int i = 0; i < n; ++i) if (alive[i]) C.push_back(cen[i]);
        return C;
    }

    // ── Algorithms 5–7: Correlate_solution + Vector_Angle-Based_Selection +
    //    Similar_poor_solutions_replaced. P is modified in place.
    //    theta and gamma are maintained incrementally, as in the paper:
    //      Alg.5 — initial correlation of each x_i in SP with the y_k of the
    //              ENTIRE current P nearest by angle, including the already
    //              accepted fronts F_{l-1};
    //      Alg.6 — N_S iterations: rho = argmax theta, mu = argmin theta (one
    //              snapshot); add x_rho and update theta/gamma only against
    //              x_rho (lines 9–17); then run Alg.7 on EVERY iteration
    //              (line 19);
    //      Alg.7 — if theta(x_mu) < (pi/2)/(N+1) and Fit(y_r) > Fit(x_mu), then
    //              y_r := x_mu (y_r being any member of P, by the letter of the
    //              paper), followed by the update: gamma(x_i) != gamma(x_mu)
    //              gives a min-update; otherwise theta(x_i) := angle directly
    //              (case (2), line 16 — even when the angle is larger).
    void vaea_select(std::vector<int>& P, const std::vector<int>& SP, int need,
                     const std::vector<std::vector<double>>& Fn,
                     const std::vector<double>& Nm,
                     const std::vector<double>& Fit, int Ntot) {
        int T = static_cast<int>(SP.size());                  // Alg.6 line 1
        std::vector<char>   sel(T, 0);                        // line 2
        std::vector<double> theta(T, std::numeric_limits<double>::infinity());
        std::vector<int>    gamma(T, -1);
        // Algorithm 5: Correlate_solution(P, SP).
        for (int i = 0; i < T; ++i) {
            int s = SP[i];
            for (std::size_t k = 0; k < P.size(); ++k) {
                double a = angle(Fn[s], Nm[s], Fn[P[k]], Nm[P[k]]);
                if (a < theta[i]) { theta[i] = a; gamma[i] = static_cast<int>(k); }
            }
        }
        const double thr = (PI_ / 2.0) / (Ntot + 1);          // Alg.7 line 1
        for (int it = 0; it < need; ++it) {                   // Alg.6 line 3
            int rho = -1, mu = -1;                            // lines 4–5
            double mx = -1.0, mn = std::numeric_limits<double>::infinity();
            for (int i = 0; i < T; ++i) {
                if (sel[i]) continue;
                if (theta[i] > mx) { mx = theta[i]; rho = i; }
                if (theta[i] < mn) { mn = theta[i]; mu = i; }
            }
            if (rho >= 0) {                                   // line 6
                int xr = SP[rho];
                P.push_back(xr); sel[rho] = 1;                // lines 7–8
                int newpos = static_cast<int>(P.size()) - 1;
                for (int i = 0; i < T; ++i) {                 // lines 9–17
                    if (sel[i]) continue;
                    int s = SP[i];
                    double a = angle(Fn[s], Nm[s], Fn[xr], Nm[xr]);
                    if (a < theta[i]) { theta[i] = a; gamma[i] = newpos; }
                }
            }
            // Algorithm 7 (Alg.6 line 19 — on every iteration).
            if (mu >= 0 && sel[mu] == 0 && theta[mu] < thr) {
                int r = gamma[mu];                            // line 2
                if (r >= 0 && Fit[P[r]] > Fit[SP[mu]]) {      // line 3
                    sel[mu] = 1;                              // line 4
                    int xm = SP[mu];
                    P[r] = xm;                                // replacement y_r <- x_mu
                    for (int i = 0; i < T; ++i) {             // lines 6–18
                        if (sel[i]) continue;
                        int s = SP[i];
                        double a = angle(Fn[s], Nm[s], Fn[xm], Nm[xm]);
                        if (gamma[i] != r) {                  // case (1)
                            if (a < theta[i]) { theta[i] = a; gamma[i] = r; }
                        } else {                              // case (2)
                            theta[i] = a;                     // set directly
                        }
                    }
                }
            }
        }
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
    CAVAMOEACore() = default;

    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }
    void set_pc(double p)            { pc_ = p; }
    void set_seed(unsigned s)        { rng_.seed(s); }

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

    void setup_seeded(DataVault<Ind_t>& /*vault*/) {}

    // ── step: one generation (Algorithm 2 + Algorithm 3) ───────────────
    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int N = n;
        int m = vault.objs_n();
        const auto& bounds = vault.get_bounds();
        std::uniform_int_distribution<int> dist_int(0, n - 1);

        // (Algorithm 2) random reproduction -> offspring in [off_base, +n).
        int off_base = vault.expand(n);
        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        const double pm = (vault.vars_n() > 0)
            ? 1.0 / static_cast<double>(vault.vars_n()) : 0.0;   // §5.1: p_m=1/V
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

        // (Algorithm 3, lines 1-3) normalization (3), Fit (4), norm (5) over the pool.
        std::vector<double> ymin(m,  std::numeric_limits<double>::max());
        std::vector<double> ymax(m, -std::numeric_limits<double>::max());
        for (int i = 0; i < pool; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) {
                ymin[j] = std::min(ymin[j], o[j]);
                ymax[j] = std::max(ymax[j], o[j]);
            }
        }
        std::vector<std::vector<double>> Fn(pool, std::vector<double>(m));
        std::vector<double> Nm(pool, 0.0), Fit(pool, 0.0);
        for (int i = 0; i < pool; ++i) {
            const auto& o = vault.objectives_of(i);
            double s = 0.0, fsum = 0.0;
            for (int j = 0; j < m; ++j) {
                double range = ymax[j] - ymin[j];
                double v = (range > 1e-14) ? (o[j] - ymin[j]) / range : 0.0;
                Fn[i][j] = v; s += v * v; fsum += v;
            }
            Nm[i]  = std::sqrt(s);
            Fit[i] = fsum;
        }

        // (line 4) fast non-dominated sort; accept fronts F_1..F_{l-1} (< N).
        auto fronts = fast_nds(vault, pool);
        std::vector<int> survivors;
        survivors.reserve(N);
        std::size_t fi = 0;
        while (fi < fronts.size() &&
               static_cast<int>(survivors.size() + fronts[fi].size()) < N) {
            for (int v : fronts[fi]) survivors.push_back(v);
            ++fi;
        }

        if (fi < fronts.size()) {
            const std::vector<int>& fl = fronts[fi];
            int K = N - static_cast<int>(survivors.size());
            if (static_cast<int>(fl.size()) == K) {
                for (int v : fl) survivors.push_back(v);
            } else {
                // ── Algorithm 4: three-stage selection of K individuals from
                //    the critical front ────────────────────────────────────
                int nfl = static_cast<int>(fl.size());
                std::vector<std::vector<double>> Y(nfl);
                for (int i = 0; i < nfl; ++i) Y[i] = Fn[fl[i]];
                auto C = ward_centers(Y, K);
                int Kc = static_cast<int>(C.size());

                std::vector<std::vector<int>> members(Kc);  // positions i within fl
                for (int i = 0; i < nfl; ++i) {
                    int    bj = 0; double bd = std::numeric_limits<double>::max();
                    for (int j = 0; j < Kc; ++j) {
                        double d = sq_dist(Y[i], C[j]);
                        if (d < bd) { bd = d; bj = j; }
                    }
                    members[bj].push_back(i);
                }

                // P is the archive under construction: the accepted fronts
                // F_{l-1} plus the picks from F_l.
                std::vector<int> P = survivors;
                std::vector<int> SP, GP;
                for (int j = 0; j < Kc; ++j) {
                    if (members[j].empty()) continue;
                    // Stage 1 (lines 10-13): nearest to C_j; tie-break by min Fit.
                    int imin = members[j][0];
                    double dmin = sq_dist(Y[imin], C[j]);
                    for (int i : members[j]) {
                        double d = sq_dist(Y[i], C[j]);
                        if (d < dmin ||
                            (d == dmin && Fit[fl[i]] < Fit[fl[imin]])) {
                            dmin = d; imin = i;
                        }
                    }
                    P.push_back(fl[imin]);
                    bool dense = (static_cast<int>(members[j].size()) >= 3); // lines 15–18
                    for (int i : members[j]) {
                        if (i == imin) continue;
                        (dense ? SP : GP).push_back(fl[i]);
                    }
                }

                // Alg.4 lines 22-29: branch on |P| + |SP| > N.
                if (static_cast<int>(P.size()) < N) {
                    if (static_cast<int>(P.size() + SP.size()) > N) {
                        // lines 23-25: Alg.5 (correlation) + Alg.6/7 (VaEA selection).
                        vaea_select(P, SP, N - static_cast<int>(P.size()),
                                    Fn, Nm, Fit, N);
                    } else {
                        // lines 26-28: P = P u SP u the best(N-|P|-|SP|) of GP by Fit.
                        for (int v : SP) P.push_back(v);
                        int need = N - static_cast<int>(P.size());
                        if (need > 0) {
                            std::sort(GP.begin(), GP.end(),
                                      [&](int a, int b){ return Fit[a] < Fit[b]; });
                            for (int k = 0; k < need && k < static_cast<int>(GP.size()); ++k)
                                P.push_back(GP[k]);
                        }
                    }
                }

                // safety (beyond the paper, see (b) in the header): top up to
                // exactly N with the best by Fit from the unused members of
                // F_l, then from the pool.
                if (static_cast<int>(P.size()) < N) {
                    std::vector<char> inP(pool, 0);
                    for (int v : P) inP[v] = 1;
                    std::vector<int> rest;
                    for (int v : fl) if (inP[v] == 0) rest.push_back(v);
                    std::sort(rest.begin(), rest.end(),
                              [&](int a, int b){ return Fit[a] < Fit[b]; });
                    for (int v : rest) {
                        if (static_cast<int>(P.size()) >= N) break;
                        P.push_back(v); inP[v] = 1;
                    }
                    for (int v = 0; v < pool && static_cast<int>(P.size()) < N; ++v)
                        if (inP[v] == 0) { P.push_back(v); inP[v] = 1; }
                }
                if (static_cast<int>(P.size()) > N) P.resize(N);
                survivors = P;                       // the resulting population
            }
        }

        rearrange(vault, survivors, pool);
    }
};

} // namespace mootation
