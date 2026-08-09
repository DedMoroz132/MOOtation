#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// MaOEA-IAMD — An Indicator-Guided Many-Objective Evolutionary Algorithm
//              with Adaptive Mapping Distance
// Ren et al. — The Journal of Supercomputing 82:280 (2026)
// doi:10.1007/s11227-026-08362-3
//
// Generational scheme (Alg. 1):
//   0. setup (once, Alg. 1 line 2): classify the variables into convergence
//      (CV) and diversity (DV) groups with the LMEA [32] clustering method:
//      per variable, nS individuals x nP perturbations; the angle between the
//      principal axis of the normalized objectives and the diagonal (1,...,1);
//      k-means (k=2) over the rows of angles; the cluster with the smaller mean
//      angle becomes CV.
//   1. KP = Identify knee points(P) (Alg. 2, the KnEA [36] method without NDS):
//      hyperplane L through the extreme points (max f_i), vertical distance to
//      L, greedy selection by decreasing d with neighbour removal
//      |f_a^i − f_x^i| ≤ R_i;  R_i = (f_i^max − f_i^min)·r_g,
//      r_g = r_{g−1}·e^{−(1−t_{g−1}/T)/M}, r_0 = 1 (KnEA).
//   2. Mating selection (Alg. 3): normalize P (Eq. 2); Fit (Eq. 5,
//      κ = 0.05); AMDC over P (Eq. 8–10) -> Φ (Eq. 6–7, niche radius r = median
//      of each point's distance to its M nearest neighbours); N binary
//      tournaments: a winner that is better on both Fit and Φ takes it;
//      otherwise the tie is broken by the sign of I_ε+ (Eq. 3); otherwise at
//      random.
//   3. SBX over the pairs of P1 (pc = 1.0, η_c = 20; §4.3(2)).
//   4. KPCM (Alg. 4): E_c = the knee point with max Fit; E_d = the top
//      ceil(N/2) knee points by Φ; with probability 0.5 a convergence mutation
//      x(v) = E_c(v) (v a random CV variable), otherwise a diversity mutation
//      x(DV) = y(DV) (y random from E_d); then noise on every variable
//      (Alg.4 line 13 — scale unspecified by the paper, see MaOEA-IAMD-N).
//   5. U = P ∪ Q (2N), normalized (Eq. 2); AMDC (Alg. 5): the curvature q by
//      Newton–Raphson (Eq. 8; q_0 = 1, stop at |Δq| <= 0.001 or 100 iterations;
//      f_i = 0 => f_i^q·log f_i := 0) taken from the non-dominated individual
//      with the smallest perpendicular distance to the diagonal; the mapping of
//      Eq. 9 is the radial projection f_bar = f'/(Σf'^q)^{1/q} onto the
//      estimated PF; d_AM is the Euclidean distance in the mapped space
//      (Eq. 10).
//   6. Environmental selection (Alg. 6): NDS; accept whole fronts while
//      |P|+|F_i| <= N; Fit (Eq. 5) over P ∪ F_k; if P is empty, bootstrap with
//      the M best by Fit from F_k; selection-replacement (Alg. 7): associate by
//      min d_AM; add x_l (max d_AM to its reference); UPDATE the associations;
//      pick x_s (min d_AM); if Fit(x_s) > Fit(p_t), replace p_t with x_s and
//      recompute Fit over P ∪ F_k (incrementally but exactly).
//
// PAPER DEFAULTS (§4.3): SBX pc = 1.0, η_c = 20; κ = 0.05 (Eq. 5); the two
//   KPCM branches at 0.5 each (Alg.4 line 5); Newton–Raphson q_0 = 1,
//   tol = 1e-3, <=100 iterations (§3.5). KPCM does replace PM — §4.5.2 ablates
//   it by "replacing it with the polynomial mutation used in the compared
//   algorithm" — so §4.3(2)'s "all compared algorithms employ SBX and PM"
//   describes the peer set, not MaOEA-IAMD's own operator.
//
// MaOEA-IAMD-N (OUR CHOICE, NOT A PAPER VALUE — the largest free parameter in
//   this file). Alg.4 line 13 is the entire specification of the KPCM noise:
//   "Add appropriate noise to each individual's decision variables." No
//   distribution, no scale, no number, anywhere in the paper. This port uses a
//   Gaussian with sigma = noise_frac * (hi - lo) per variable, noise_frac =
//   0.01, settable with set_noise_frac.
//   Why 0.01 and not something larger: the noise is applied to EVERY variable
//   of EVERY offspring, so its scale dominates convergence. Measured on
//   DTLZ2 (M=3, n=12, pop 91, 200 generations, seed 20260804), mean distance
//   to the front:
//       noise_frac  0.316  0.20   0.10   0.05   0.02   0.01   0.00
//       mean        0.289  0.127  0.065  0.046  0.015  0.005  0.014
//   An earlier version of this file used 0.316 (= sqrt(0.1)) and attributed
//   "N(0, 0.1)" to §3.4; that number is not in the paper and is ~50x worse than
//   0.01 here. At 0.0 the population collapses to a single point (mean == best),
//   so some noise IS required — the paper is right about that much.
//   This value is ours and was picked from the sweep above; a reader who
//   disagrees has the numbers to argue with and a setter to override it.
// Parameters the paper does not give numerically, taken from the primary
//   sources it cites:
//   nS = 2, nP = 4 — the LMEA [32] variable-classification defaults;
//   T = 0.5, r_0 = 1 — the KnEA [36] knee-point neighbourhood defaults
//   ("neighborhood size is set the same as in [36]"; t is the fraction of knee
//   points in P, since the paper deliberately omits the NDS step).
// DECLARED DEVIATIONS (deliberate guards):
//   - the replacement in Alg. 7 runs only while |F_k|−1 >= N−|P|; otherwise the
//     loop can exhaust F_k before |P| = N, a case the paper leaves undefined;
//   - Φ (Eq. 6) is stored as log Φ — a monotone guard against underflow;
//   - degenerate cases the paper does not cover: a singular knee hyperplane
//     falls back to d = −Σ of the normalized objectives; a zero derivative or
//     an excursion to q <= 0 in Newton–Raphson gives q = 1; zero perturbation
//     covariance gives an angle of π/2, putting the variable in DV; an empty
//     k-means cluster gives CV = DV = all variables;
//   - eta_m_ / set_eta_mutation are kept for interface compatibility but are
//     unused, since the paper replaces PM with KPCM.
// EXTENSIONS BEYOND THE PAPER (off by default): constraint_mode FEASIBILITY
//   (feasible-first in dominance and in the tournament); binary genome (uniform
//   crossover + bit-flip; KPCM and the classification stay real-valued only).
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
#include "../operators/sbx.hpp"

namespace mootation {

struct MaOEAIAMD_Individual : public Based_Individual {
    int rank = 0;
};

template <typename Ind_t>
class MaOEAIAMDCore {
public:
    ConstraintMode constraint_mode = ConstraintMode::NONE;

private:
    static constexpr double KAPPA_     = 0.05;   // Eq. 5
    static constexpr double NR_TOL_    = 1e-3;   // §3.5: |q_{k+1}−q_k| ≤ 0.001
    static constexpr int    NR_MAXIT_  = 100;    // §3.5
    static constexpr double KNEE_T_    = 0.5;    // KnEA [36]: rate T

    // Alg.4 line 13 is the whole specification of the KPCM noise: "Add
    // appropriate noise to each individual's decision variables". No
    // distribution, no scale, nowhere in the paper. It is therefore a free
    // parameter of this port, not a paper default, and must be settable.
    // Expressed as a fraction of each variable's own range, so it does not
    // silently depend on how the problem is scaled.
    double noise_frac_ = 0.01;          // see MaOEA-IAMD-N in the header

    double eta_c_ = 20.0;   // §4.3(2)
    double pc_    = 1.0;    // §4.3(2)
    double eta_m_ = 20.0;   // unused: the paper replaces PM with KPCM
    int    nsel_  = 2;      // nS (LMEA [32])
    int    nper_  = 4;      // nP (LMEA [32])
    std::mt19937 rng_{std::random_device{}()};

    std::vector<int> cv_, dv_;          // Alg. 1 line 2: CV/DV variable indices
    double knee_r_      = 1.0;          // r_{g−1} (KnEA), r_0 = 1
    double knee_t_prev_ = 0.0;          // t_{g−1} (KnEA)

    // ── basic predicates ─────────────────────────────────────────────────────

    bool dominates(DataVault<Ind_t>& vault, int a, int b) {
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
            for (int i : cur) for (int j : S[i]) if (--ndom[j] == 0) next.push_back(j);
            cur = next;
        }
        return fronts;
    }

    // I_{ε+}(a, b) = max_i (a_i − b_i)  (Eq. 3)
    static double eps_plus(const std::vector<double>& a, const std::vector<double>& b) {
        double w = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < a.size(); ++i) { double d = a[i] - b[i]; if (d > w) w = d; }
        return w;
    }
    static double mdist(const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) { double t = a[j] - b[j]; s += t * t; }
        return std::sqrt(s);
    }
    static double median_of(std::vector<double> v) {
        if (v.empty()) return 0.0;
        std::size_t k = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + static_cast<long>(k), v.end());
        return v[k];
    }

    // ── Eq. 2: normalization by the set's ideal and nadir points ───────────
    std::vector<std::vector<double>> normalize_set(DataVault<Ind_t>& vault, int count) {
        int m = vault.objs_n();
        std::vector<double> zmin(m,  std::numeric_limits<double>::max());
        std::vector<double> zmax(m, -std::numeric_limits<double>::max());
        for (int i = 0; i < count; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) {
                zmin[j] = std::min(zmin[j], o[j]);
                zmax[j] = std::max(zmax[j], o[j]);
            }
        }
        std::vector<std::vector<double>> Fn(count, std::vector<double>(m, 0.0));
        for (int i = 0; i < count; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) {
                double range = zmax[j] - zmin[j];
                Fn[i][j] = (range > 1e-14) ? (o[j] - zmin[j]) / range : 0.0;
            }
        }
        return Fn;
    }

    // ── Eq. 5: Fit over the member set (values are valid for members) ──────
    std::vector<double> fitness_over(const std::vector<std::vector<double>>& Fn,
                                     const std::vector<int>& members) {
        std::vector<double> fit(Fn.size(), 0.0);
        for (int a : members) {
            double s = 0.0;
            for (int b : members) {
                if (a == b) continue;
                s -= std::exp(-eps_plus(Fn[b], Fn[a]) / KAPPA_);
            }
            fit[a] = s;
        }
        return fit;
    }

public:
    // ── Eq. 8: PF curvature estimate by Newton–Raphson (§3.5) ──────────────
    // f is the normalized objective vector of a non-dominated individual.
    // q_0 = 1; stop at |Δq| <= 0.001 or 100 iterations; f_i = 0 =>
    // f_i^q·log f_i := 0 (per the paper).
    // Guards beyond the paper: a zero derivative, or an excursion to q <= 0,
    // yields q = 1.
    // public static, so the curvature unit test can reach it.
    static double newton_raphson_curvature(const std::vector<double>& f) {
        double q = 1.0;
        for (int it = 0; it < NR_MAXIT_; ++it) {
            double S = 0.0, T = 0.0;
            for (double fi : f) {
                if (fi <= 0.0) continue;                  // zero contribution (paper)
                double t = std::pow(fi, q);
                S += t;
                T += t * std::log(fi);                    // log(1) = 0 — safe
            }
            if (S <= 0.0 || std::abs(T) < 1e-300) return 1.0;
            double qn = q - std::log(S) / (T / S);        // Eq. 8
            if (!std::isfinite(qn) || qn <= 0.0) return 1.0;
            if (std::abs(qn - q) <= NR_TOL_) return qn;
            q = qn;
        }
        return q;
    }

private:
    // §3.5: the non-dominated individual closest (perpendicularly) to the
    // diagonal of the normalized space feeds Newton–Raphson.
    double estimate_curvature(const std::vector<std::vector<double>>& Fn,
                              const std::vector<int>& nd) const {
        if (nd.empty()) return 1.0;
        int m = static_cast<int>(Fn[nd[0]].size());
        double u = 1.0 / std::sqrt(static_cast<double>(m));
        int pick = nd[0];
        double best = std::numeric_limits<double>::max();
        for (int v : nd) {
            double n2 = 0.0, pr = 0.0;
            for (int j = 0; j < m; ++j) { n2 += Fn[v][j] * Fn[v][j]; pr += Fn[v][j] * u; }
            double perp2 = n2 - pr * pr;
            if (perp2 < best) { best = perp2; pick = v; }
        }
        return newton_raphson_curvature(Fn[pick]);
    }

    // Eq. 9: radial projection onto the estimated PF, Σ f_bar^q = 1
    static std::vector<double> map_point(const std::vector<double>& f, double q) {
        double s = 0.0;
        for (double fi : f) if (fi > 0.0) s += std::pow(fi, q);
        if (s <= 1e-300) return f;                        // point at the ideal (guard)
        double denom = std::pow(s, 1.0 / q);
        std::vector<double> out(f.size());
        for (std::size_t j = 0; j < f.size(); ++j) out[j] = f[j] / denom;
        return out;
    }
    static std::vector<std::vector<double>> map_all(
            const std::vector<std::vector<double>>& Fn, double q) {
        std::vector<std::vector<double>> MF(Fn.size());
        for (std::size_t i = 0; i < Fn.size(); ++i) MF[i] = map_point(Fn[i], q);
        return MF;
    }

    // ── Eq. 6–7: diversity Φ (stored as log Φ; the niche radius r is the
    //    median over points of the distance to their M nearest neighbours in
    //    d_AM) ────────────────────────────────────────────────────────────────
    std::vector<double> log_phi(const std::vector<std::vector<double>>& MF, int m) const {
        int n = static_cast<int>(MF.size());
        std::vector<double> lp(n, 0.0);
        if (n <= 1) return lp;
        std::vector<std::vector<double>> dam(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                dam[i][j] = dam[j][i] = mdist(MF[i], MF[j]);
        int k = std::min(m, n - 1);
        std::vector<double> rec;
        rec.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
        std::vector<double> tmp;
        for (int i = 0; i < n; ++i) {
            tmp.clear();
            for (int j = 0; j < n; ++j) if (j != i) tmp.push_back(dam[i][j]);
            std::partial_sort(tmp.begin(), tmp.begin() + k, tmp.end());
            for (int t = 0; t < k; ++t) rec.push_back(tmp[t]);
        }
        double r = median_of(rec);
        if (r <= 0.0) return lp;                          // all coincide -> Φ = 1
        for (int i = 0; i < n; ++i) {
            double s = 0.0;
            for (int j = 0; j < n; ++j) {
                if (j == i) continue;
                if (dam[i][j] < r)
                    s += std::log(std::max(dam[i][j] / r, 1e-300));
            }
            lp[i] = s;
        }
        return lp;
    }

    // ── Alg. 2: knee points (the KnEA [36] method, no NDS — over all of P) ─
    static bool solve_linear(std::vector<std::vector<double>> A, std::vector<double>& w) {
        int n = static_cast<int>(A.size());
        for (int i = 0; i < n; ++i) A[i].push_back(1.0);  // right-hand side = 1
        for (int c = 0; c < n; ++c) {
            int piv = c;
            for (int r = c + 1; r < n; ++r)
                if (std::abs(A[r][c]) > std::abs(A[piv][c])) piv = r;
            if (std::abs(A[piv][c]) < 1e-12) return false;
            std::swap(A[c], A[piv]);
            for (int r = 0; r < n; ++r) {
                if (r == c) continue;
                double k = A[r][c] / A[c][c];
                for (int j = c; j <= n; ++j) A[r][j] -= k * A[c][j];
            }
        }
        w.assign(n, 0.0);
        for (int i = 0; i < n; ++i) w[i] = A[i][n] / A[i][i];
        return true;
    }

    std::vector<int> identify_knee_points(DataVault<Ind_t>& vault, int n) {
        int m = vault.objs_n();
        std::vector<double> fmax(m, -std::numeric_limits<double>::max());
        std::vector<double> fmin(m,  std::numeric_limits<double>::max());
        std::vector<int>    ext (m, 0);                   // extreme points
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) {
                if (o[j] > fmax[j]) { fmax[j] = o[j]; ext[j] = i; }
                if (o[j] < fmin[j])   fmin[j] = o[j];
            }
        }
        // hyperplane L through the extreme points: Σ w_k f_k = 1
        std::vector<std::vector<double>> E(m);
        for (int j = 0; j < m; ++j) E[j] = vault.objectives_of(ext[j]);
        std::vector<double> w;
        bool ok = solve_linear(E, w);
        std::vector<double> d(n, 0.0);
        if (ok) {
            double wn = 0.0;
            for (double wi : w) wn += wi * wi;
            wn = std::sqrt(std::max(wn, 1e-300));
            for (int i = 0; i < n; ++i) {
                const auto& o = vault.objectives_of(i);
                double dot = 0.0;
                for (int j = 0; j < m; ++j) dot += w[j] * o[j];
                d[i] = (1.0 - dot) / wn;                  // larger = closer to ideal
            }
        } else {
            // fallback guard beyond the paper: −Σ of the normalized objectives
            for (int i = 0; i < n; ++i) {
                const auto& o = vault.objectives_of(i);
                double s = 0.0;
                for (int j = 0; j < m; ++j) {
                    double range = fmax[j] - fmin[j];
                    s += (range > 1e-14) ? (o[j] - fmin[j]) / range : 0.0;
                }
                d[i] = -s;
            }
        }
        // KnEA: r_g = r_{g−1}·e^{−(1−t_{g−1}/T)/M};  R_i = (f_i^max−f_i^min)·r_g
        knee_r_ *= std::exp(-(1.0 - knee_t_prev_ / KNEE_T_) / static_cast<double>(m));
        std::vector<double> R(m);
        for (int j = 0; j < m; ++j) R[j] = (fmax[j] - fmin[j]) * knee_r_;

        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return d[a] > d[b]; });
        std::vector<char> removed(n, 0);
        std::vector<int> KP;
        for (int x : order) {
            if (removed[x]) continue;
            KP.push_back(x);
            const auto& fx = vault.objectives_of(x);
            for (int a = 0; a < n; ++a) {
                if (removed[a]) continue;
                const auto& fa = vault.objectives_of(a);
                bool nb = true;
                for (int j = 0; j < m; ++j)
                    if (std::abs(fa[j] - fx[j]) > R[j]) { nb = false; break; }
                if (nb) removed[a] = 1;                   // neighbours (incl. x)
            }
        }
        knee_t_prev_ = static_cast<double>(KP.size()) / static_cast<double>(std::max(n, 1));
        return KP;
    }

    // ── Alg. 1 line 2: CV/DV variable classification (LMEA [32]) ───────────
    // nP perturbed points for nS individuals per variable; the angle between
    // the principal axis of the normalized objective cloud and the diagonal
    // (1,...,1); then k-means (k=2).
    double principal_angle(const std::vector<std::vector<double>>& pts, int m) const {
        const double HALF_PI = 1.57079632679489662;
        int k = static_cast<int>(pts.size());
        if (k < 2) return HALF_PI;
        std::vector<double> mean(m, 0.0);
        for (const auto& p : pts) for (int j = 0; j < m; ++j) mean[j] += p[j];
        for (int j = 0; j < m; ++j) mean[j] /= k;
        std::vector<double> C(static_cast<std::size_t>(m) * m, 0.0);
        for (const auto& p : pts)
            for (int a = 0; a < m; ++a)
                for (int b = 0; b < m; ++b)
                    C[static_cast<std::size_t>(a) * m + b] += (p[a] - mean[a]) * (p[b] - mean[b]);
        // power iteration -> principal direction
        std::vector<double> v(m, 1.0 / std::sqrt(static_cast<double>(m))), nv(m);
        for (int it = 0; it < 60; ++it) {
            double nn = 0.0;
            for (int a = 0; a < m; ++a) {
                double s = 0.0;
                for (int b = 0; b < m; ++b) s += C[static_cast<std::size_t>(a) * m + b] * v[b];
                nv[a] = s; nn += s * s;
            }
            nn = std::sqrt(nn);
            if (nn < 1e-30) return HALF_PI;               // variable has no effect -> DV
            for (int a = 0; a < m; ++a) v[a] = nv[a] / nn;
        }
        double dot = 0.0, u = 1.0 / std::sqrt(static_cast<double>(m));
        for (int a = 0; a < m; ++a) dot += v[a] * u;
        double c = std::min(std::abs(dot), 1.0);
        return std::acos(c);
    }

    void classify_variables(DataVault<Ind_t>& vault) {
        cv_.clear(); dv_.clear();
        int n = vault.pop_size(), D = vault.vars_n(), m = vault.objs_n();
        if (D == 0) return;
        const auto& bounds = vault.get_bounds();
        // normalize the perturbed points by the current population's min/max
        std::vector<double> fmin(m,  std::numeric_limits<double>::max());
        std::vector<double> fmax(m, -std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(i);
            for (int j = 0; j < m; ++j) {
                fmin[j] = std::min(fmin[j], o[j]);
                fmax[j] = std::max(fmax[j], o[j]);
            }
        }
        int ns = std::max(1, std::min(nsel_, n));
        std::vector<int> samp(n);
        std::iota(samp.begin(), samp.end(), 0);
        std::shuffle(samp.begin(), samp.end(), rng_);
        samp.resize(ns);

        int base = vault.expand(ns * nper_);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        std::vector<double> vars(D);
        std::vector<std::vector<double>> angle(D, std::vector<double>(ns, 0.0));
        for (int j = 0; j < D; ++j) {
            double lo = bounds[j].first .value_or(0.0);
            double hi = bounds[j].second.value_or(1.0);
            for (int s = 0; s < ns; ++s) {
                for (int p = 0; p < nper_; ++p) {
                    for (int t = 0; t < D; ++t) vars[t] = vault.get_variable(samp[s], t);
                    vars[j] = lo + uni(rng_) * (hi - lo);
                    int slot = base + s * nper_ + p;
                    if (vault.bin_vars_n() > 0)
                        vault.set_all_variables(slot, vars,
                                                vault.binary_variables_of(samp[s]));
                    else
                        vault.set_variables(slot, vars);
                }
            }
            vault.sync();
            for (int s = 0; s < ns; ++s) {
                std::vector<std::vector<double>> pts(nper_, std::vector<double>(m, 0.0));
                for (int p = 0; p < nper_; ++p) {
                    const auto& o = vault.objectives_of(base + s * nper_ + p);
                    for (int t = 0; t < m; ++t) {
                        double range = fmax[t] - fmin[t];
                        pts[p][t] = (range > 1e-14) ? (o[t] - fmin[t]) / range : 0.0;
                    }
                }
                angle[j][s] = principal_angle(pts, m);
            }
        }
        vault.reduce(n);

        // k-means (k = 2) over the angle rows; smaller mean angle -> CV
        std::vector<double> mean_a(D, 0.0);
        for (int j = 0; j < D; ++j) {
            for (int s = 0; s < ns; ++s) mean_a[j] += angle[j][s];
            mean_a[j] /= ns;
        }
        int lo_j = 0, hi_j = 0;
        for (int j = 1; j < D; ++j) {
            if (mean_a[j] < mean_a[lo_j]) lo_j = j;
            if (mean_a[j] > mean_a[hi_j]) hi_j = j;
        }
        if (D < 2 || mean_a[hi_j] - mean_a[lo_j] < 1e-12) {
            // indistinguishable (guard beyond the paper): all variables in both
            cv_.resize(D); std::iota(cv_.begin(), cv_.end(), 0);
            dv_ = cv_;
            return;
        }
        std::vector<double> c1 = angle[lo_j], c2 = angle[hi_j];
        std::vector<int> lab(D, 0);
        for (int it = 0; it < 100; ++it) {
            bool changed = false;
            for (int j = 0; j < D; ++j) {
                double d1 = 0.0, d2 = 0.0;
                for (int s = 0; s < ns; ++s) {
                    double a = angle[j][s] - c1[s], b = angle[j][s] - c2[s];
                    d1 += a * a; d2 += b * b;
                }
                int l = (d1 <= d2) ? 0 : 1;
                if (l != lab[j]) { lab[j] = l; changed = true; }
            }
            std::vector<double> n1(ns, 0.0), n2(ns, 0.0);
            int k1 = 0, k2 = 0;
            for (int j = 0; j < D; ++j) {
                if (lab[j] == 0) { ++k1; for (int s = 0; s < ns; ++s) n1[s] += angle[j][s]; }
                else             { ++k2; for (int s = 0; s < ns; ++s) n2[s] += angle[j][s]; }
            }
            if (k1 == 0 || k2 == 0) break;
            for (int s = 0; s < ns; ++s) { c1[s] = n1[s] / k1; c2[s] = n2[s] / k2; }
            if (!changed) break;
        }
        double m1 = 0.0, m2 = 0.0;
        int k1 = 0, k2 = 0;
        for (int j = 0; j < D; ++j) {
            if (lab[j] == 0) { m1 += mean_a[j]; ++k1; }
            else             { m2 += mean_a[j]; ++k2; }
        }
        if (k1 == 0 || k2 == 0) {
            cv_.resize(D); std::iota(cv_.begin(), cv_.end(), 0);
            dv_ = cv_;
            return;
        }
        m1 /= k1; m2 /= k2;
        int cv_label = (m1 <= m2) ? 0 : 1;
        for (int j = 0; j < D; ++j)
            (lab[j] == cv_label ? cv_ : dv_).push_back(j);
    }

    // ── Alg. 3: mating selection (binary tournament on Fit and Φ) ──────────
    std::vector<int> mating_selection(DataVault<Ind_t>& vault, int n,
                                      const std::vector<std::vector<double>>& Fn,
                                      const std::vector<double>& fit,
                                      const std::vector<double>& lphi) {
        std::uniform_int_distribution<int> di(0, n - 1);
        std::uniform_int_distribution<int> coin(0, 1);
        std::vector<int> P1;
        P1.reserve(n);
        for (int i = 0; i < n; ++i) {
            int x = di(rng_), y = di(rng_);
            // FEASIBILITY extension (beyond the paper, off by default)
            if (constraint_mode == ConstraintMode::FEASIBILITY) {
                double cx = vault.get_cv(x), cy = vault.get_cv(y);
                bool xf = (cx <= 0.0), yf = (cy <= 0.0);
                if (xf && !yf)        { P1.push_back(x); continue; }
                if (!xf && yf)        { P1.push_back(y); continue; }
                if (!xf && !yf)       { P1.push_back(cx < cy ? x : y); continue; }
            }
            if      (fit[x] > fit[y] && lphi[x] > lphi[y]) P1.push_back(x);   // lines 8–9
            else if (fit[y] > fit[x] && lphi[y] > lphi[x]) P1.push_back(y);   // lines 10–11
            else if (eps_plus(Fn[x], Fn[y]) < 0.0)         P1.push_back(x);   // lines 13–14
            else if (eps_plus(Fn[y], Fn[x]) < 0.0)         P1.push_back(y);   // lines 15–16
            else                                           P1.push_back(coin(rng_) ? x : y);
        }
        return P1;
    }

    // ── Alg. 4: KPCM mutation of a single offspring ────────────────────────
    void kpcm_mutate(std::vector<double>& c, DataVault<Ind_t>& vault,
                     int ec, const std::vector<int>& ed) {
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        bool conv = (uni(rng_) <= 0.5);                   // line 5
        bool can_conv = (ec >= 0 && !cv_.empty());
        bool can_div  = (!ed.empty() && !dv_.empty());
        if ((conv && can_conv) || (!can_div && can_conv)) {
            std::uniform_int_distribution<int> dv(0, static_cast<int>(cv_.size()) - 1);
            int v = cv_[dv(rng_)];                        // lines 6–7
            c[v] = vault.get_variable(ec, v);
        } else if (can_div) {                             // lines 9–10
            std::uniform_int_distribution<int> dy(0, static_cast<int>(ed.size()) - 1);
            int y = ed[dy(rng_)];
            for (int v : dv_) c[v] = vault.get_variable(y, v);
        }
        // Alg.4 line 13: "Add appropriate noise to each individual's decision
        // variables". The paper fixes neither the distribution nor its scale,
        // so this is our choice, not a transcription (MaOEA-IAMD-N). Gaussian
        // with sigma = noise_frac_ * (hi - lo) per variable, so the strength is
        // relative to that variable's own range rather than to whatever units
        // the problem happens to use.
        const auto& bnd = vault.get_bounds();
        for (std::size_t j = 0; j < c.size(); ++j) {
            const double lo = bnd[j].first .value_or(0.0);
            const double hi = bnd[j].second.value_or(1.0);
            const double sd = noise_frac_ * (hi - lo);
            if (sd <= 0.0) continue;
            std::normal_distribution<double> gauss(0.0, sd);
            c[j] += gauss(rng_);
        }
        // vault.set_variables performs the clamp into the bounds
    }

    // ── Alg. 6 + Alg. 7: environmental selection ───────────────────────────
    std::vector<int> environmental_selection(DataVault<Ind_t>& vault, int pool, int N) {
        int m = vault.objs_n();
        auto Fn     = normalize_set(vault, pool);         // Eq. 2 (Alg. 1 line 9)
        auto fronts = fast_nds(vault, pool);              // Alg. 6 line 2
        // Alg. 5 (AMDC): curvature from the non-dominated individual nearest
        // the diagonal, then the mapping
        double q = estimate_curvature(Fn, fronts[0]);
        auto MF = map_all(Fn, q);                         // Eq. 9

        std::vector<int> P, Fk;
        for (const auto& fr : fronts) {                   // Alg. 6 lines 3–7
            if (static_cast<int>(P.size() + fr.size()) <= N) {
                P.insert(P.end(), fr.begin(), fr.end());
                if (static_cast<int>(P.size()) == N) break;
            } else { Fk = fr; break; }
        }
        if (static_cast<int>(P.size()) == N || Fk.empty()) {
            if (static_cast<int>(P.size()) > N) P.resize(N);
            return P;                                     // Alg. 6 lines 8–9
        }

        // Alg. 6 line 11: Fit over P ∪ F_k (Eq. 5, κ = 0.05)
        std::vector<int> S = P;
        S.insert(S.end(), Fk.begin(), Fk.end());
        auto fit = fitness_over(Fn, S);

        // Alg. 6 lines 12–13: bootstrap with the M best by Fit when P is empty
        if (P.empty()) {
            std::sort(Fk.begin(), Fk.end(), [&](int a, int b) { return fit[a] > fit[b]; });
            int take = std::min({m, static_cast<int>(Fk.size()), N});
            P.assign(Fk.begin(), Fk.begin() + take);
            Fk.erase(Fk.begin(), Fk.begin() + take);
        }

        // Alg. 7: selection-replacement
        // association: ad[k] = min d_AM to P, aref[k] = the nearest reference
        std::vector<double> ad(Fk.size());
        std::vector<int>    aref(Fk.size());
        auto assoc_one = [&](std::size_t k) {
            double best = std::numeric_limits<double>::max();
            int who = -1;
            for (int p : P) {
                double dd = mdist(MF[Fk[k]], MF[p]);
                if (dd < best) { best = dd; who = p; }
            }
            ad[k] = best; aref[k] = who;
        };
        for (std::size_t k = 0; k < Fk.size(); ++k) assoc_one(k);   // line 2
        auto erase_k = [&](std::size_t k) {
            Fk[k] = Fk.back();   Fk.pop_back();
            ad[k] = ad.back();   ad.pop_back();
            aref[k] = aref.back(); aref.pop_back();
        };

        while (static_cast<int>(P.size()) < N && !Fk.empty()) {
            // line 3: x_l = the maximum distance to its own reference
            std::size_t li = 0;
            for (std::size_t k = 1; k < Fk.size(); ++k) if (ad[k] > ad[li]) li = k;
            int xl = Fk[li];
            P.push_back(xl);                              // line 4
            erase_k(li);
            if (Fk.empty()) break;
            // line 5: update the associations (exact: x_l is the only new reference)
            for (std::size_t k = 0; k < Fk.size(); ++k) {
                double dd = mdist(MF[Fk[k]], MF[xl]);
                if (dd < ad[k]) { ad[k] = dd; aref[k] = xl; }
            }
            // line 6: x_s = the minimum distance to its own reference
            std::size_t si = 0;
            for (std::size_t k = 1; k < Fk.size(); ++k) if (ad[k] < ad[si]) si = k;
            int xs = Fk[si], pt = aref[si];
            // lines 7–9; termination guard |F_k|−1 >= N−|P| (see the header)
            if (pt >= 0 && fit[xs] > fit[pt] &&
                static_cast<int>(Fk.size()) - 1 >= N - static_cast<int>(P.size())) {
                for (int& slot : P) if (slot == pt) { slot = xs; break; }   // line 8
                erase_k(si);
                // re-associate: reference p_t is gone, x_s becomes the new one
                for (std::size_t k = 0; k < Fk.size(); ++k) {
                    if (aref[k] == pt) assoc_one(k);
                    else {
                        double dd = mdist(MF[Fk[k]], MF[xs]);
                        if (dd < ad[k]) { ad[k] = dd; aref[k] = xs; }
                    }
                }
                // line 9: recompute Fit over P ∪ F_k without p_t (incrementally exact)
                for (int a : P)  fit[a] += std::exp(-eps_plus(Fn[pt], Fn[a]) / KAPPA_);
                for (int a : Fk) fit[a] += std::exp(-eps_plus(Fn[pt], Fn[a]) / KAPPA_);
            }
        }

        // defensive top-up (unreachable in practice given the termination guard)
        if (static_cast<int>(P.size()) < N) {
            std::vector<char> in(pool, 0);
            for (int v : P) in[v] = 1;
            std::vector<int> rest;
            for (int i = 0; i < pool; ++i) if (in[i] == 0) rest.push_back(i);
            std::sort(rest.begin(), rest.end(), [&](int a, int b) { return fit[a] > fit[b]; });
            for (int v : rest) {
                if (static_cast<int>(P.size()) >= N) break;
                P.push_back(v);
            }
        }
        if (static_cast<int>(P.size()) > N) P.resize(N);
        return P;
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
    MaOEAIAMDCore() = default;

    void set_eta_crossover(double e) { eta_c_ = e; }
    void set_eta_mutation (double e) { eta_m_ = e; }   // compatibility shim; unused
    void set_pc(double p)            { pc_ = p; }
    void set_nsel(int v)             { nsel_ = std::max(1, v); }
    void set_nper(int v)             { nper_ = std::max(2, v); }
    void set_noise_frac(double v)    { noise_frac_ = std::max(0.0, v); }
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
                double lo = bounds[j].first .value_or(0.0);
                double hi = bounds[j].second.value_or(1.0);
                vars[j] = lo + dr(rng_) * (hi - lo);
            }
            for (int j = 0; j < vault.bin_vars_n(); ++j) bvars[j] = db(rng_);
            if (vault.bin_vars_n() > 0) vault.set_all_variables(i, vars, bvars);
            else                        vault.set_variables(i, vars);
        }
        vault.sync();
        knee_r_ = 1.0; knee_t_prev_ = 0.0;
        classify_variables(vault);                        // Alg. 1 line 2
    }

    // Resume via Optimizer::setup_with_seed: the vault is already seeded, so
    // only the CV/DV classification and the knee-state reset are performed.
    void setup_seeded(DataVault<Ind_t>& vault) {
        knee_r_ = 1.0; knee_t_prev_ = 0.0;
        classify_variables(vault);
    }

    void step(DataVault<Ind_t>& vault) {
        int n = vault.pop_size();
        int m = vault.objs_n();
        const auto& bounds = vault.get_bounds();

        // ── Alg. 1 line 4: knee points of the current population (Alg. 2) ──
        std::vector<int> KP = identify_knee_points(vault, n);

        // ── Alg. 3 lines 1–5: normalize P, Fit (Eq. 5), AMDC over P, Φ ─────
        auto FnP = normalize_set(vault, n);
        std::vector<int> allP(n);
        std::iota(allP.begin(), allP.end(), 0);
        auto fitP = fitness_over(FnP, allP);
        // non-dominated members of P -> curvature (Alg. 5, lines 1–2)
        std::vector<int> ndP;
        for (int i = 0; i < n; ++i) {
            bool dom = false;
            for (int j = 0; j < n && !dom; ++j)
                if (j != i && dominates(vault, j, i)) dom = true;
            if (!dom) ndP.push_back(i);
        }
        double qP = estimate_curvature(FnP, ndP);
        auto MFP  = map_all(FnP, qP);                     // Eq. 9
        auto lphiP = log_phi(MFP, m);                     // Eq. 6–7

        // ── Alg. 3 lines 6–21: binary tournaments ──────────────────────────
        auto P1 = mating_selection(vault, n, FnP, fitP, lphiP);

        // ── Alg. 4 lines 1–3: elite knee points ────────────────────────────
        int ec = -1;
        for (int v : KP) if (ec < 0 || fitP[v] > fitP[ec]) ec = v;        // E_c
        std::vector<int> ed = KP;                                         // E_d
        std::sort(ed.begin(), ed.end(),
                  [&](int a, int b) { return lphiP[a] > lphiP[b]; });
        int edn = std::min<int>(static_cast<int>(ed.size()), (n + 1) / 2);
        ed.resize(edn);

        // ── Alg. 1 lines 6–7: SBX (pc=1.0, η_c=20) + KPCM ──────────────────
        int off_base = vault.expand(n);
        std::vector<double> pv1(vault.vars_n()), pv2(vault.vars_n()), c1, c2;
        for (int i = 0; i < n; i += 2) {
            int p1 = P1[i];
            int p2 = P1[(i + 1) % n];
            for (int j = 0; j < vault.vars_n(); ++j) {
                pv1[j] = vault.get_variable(p1, j);
                pv2[j] = vault.get_variable(p2, j);
            }
            ops::sbx(pv1, pv2, c1, c2, bounds, eta_c_, pc_, rng_);
            kpcm_mutate(c1, vault, ec, ed);
            if (i + 1 < n) kpcm_mutate(c2, vault, ec, ed);
            if (vault.bin_vars_n() > 0) {
                // extension beyond the paper: binary genome
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

        // ── Alg. 1 lines 8–11: U = P ∪ Q → env. selection (Alg. 5–7) ───────
        int pool = off_base + n;
        auto survivors = environmental_selection(vault, pool, n);
        rearrange(vault, survivors, pool);
    }
};

} // namespace mootation
