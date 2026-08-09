#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// SRVStrategy — Self-Guided Reference Vector (SRV) extraction strategy
// Liu, Lin, Wong, Coello Coello, Li, Ming, Zhang — IEEE Trans. Cybernetics, 2022
// doi:10.1109/TCYB.2020.2971638
//
// The extract scheme (Algorithm 1; the carrier calls it every generation):
//   1. Normalize Pc per Eq.2: f'(x) = (f(x)−z*)/(z^nad−z*);
//      EdI(x) = ||f'(x)|| (Eq.7); f*(x) = f'(x)/EdI(x) lies on the unit sphere
//      (Eq.14).
//   2. ADM initialization (Alg.2):
//      rho(x) = Sum_{y!=x, theta(y,x)<=theta_c} exp(−(theta/theta_c)^2) (Eq.9);
//      delta(x) = min_{y: rho(y)>rho(x)} theta(x,y); delta(argmax rho) = pi/2
//      (Eq.10).
//   3. Centroids: the m extreme points x^{e(i)} = argmin theta(x, e_i)
//      (Eq.11), then N−m individuals by decreasing delta(x).
//   4. Self-Adjustment (Alg.3): a modified k-means on angles, at most 2m
//      iterations; the first m centroid SLOTS are held fixed — extreme where
//      distinct extremes were found, see the deviation below; Eq.13–14.
//   5. rv_i = c_i / EdI(c_i), giving N unit RVs (Eq.15).
//   The theta_c schedule (Eq.17):
//   theta_c(G) = (pi − theta_c^min)(G−1)/(2·G_max) + theta_c^min/2,
//   where theta_c^min (Eq.16) is the minimum angle between neighbouring
//   ORIGINAL RVs. Throughout the paper "angle" means
//   arccos|a·b/(||a||·||b||)| (Eq.6/Eq.8).
//
// PAPER DEFAULTS: §III.
// DECLARED DEVIATIONS: the k-means loop runs t < 2m against the paper's
//   "while t <= 2m", i.e. one iteration fewer; a duplicate extreme centroid
//   shared by two axes is skipped, whereas Alg.2 lines 5-8 set c^i = x^{e(i)}
//   for every i = 1..m with no distinctness test and therefore insert the
//   duplicate. Consequence, since the freeze is POSITIONAL: with k_e < m
//   distinct extremes the slots [k_e, m) hold delta-ranked ORDINARY points,
//   and those are frozen through the whole k-means in place of centroids the
//   paper would have adjusted — so up to m−k_e fewer boundary directions are
//   emitted. No boundary individual is dropped: the shared argmax stays a
//   centroid, only its second copy is skipped;
//   z^nad = the max over Pc (Table I; acceptable for a standalone carrier).
// EXTENSIONS BEYOND THE PAPER: guards for empty clusters, zero norms and
//   N <= 1 — numerical protection only.
// ============================================================================
// A critical fix worth recording: theta_min() used to compute acos of the raw
// dot product over UNNORMALIZED Das-Dennis simplex points (Sum w = 1,
// ||w||_2 in [1/sqrt(m), 1]), so theta_c^min was systematically overestimated
// — at m=3, H=2 it gave 60 degrees instead of the true 45; at m=3, H=16 it gave
// 20.4 instead of 3.81. By Eq.16, theta(r^i, r^j) is the true angle (Eq.6/8:
// arccos after dividing by the norms). Fixed by normalizing inside
// theta_min(), so the input may be at any scale.

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "../data_vault.hpp"

namespace mootation {

// SRV_PI is not guaranteed by the C++ standard (absent on MSVC without
// _USE_MATH_DEFINES). Define a local constant instead.
#ifndef SRV_PI
inline constexpr double SRV_PI = 3.14159265358979323846;
#endif

template <typename Ind_t>
class SRVStrategy {
public:
    // ── θ_c schedule (Eq. 16-17) ─────────────────────────────────────────────
    // V0: the original Das-Dennis points lie on the unit SIMPLEX, not the
    // sphere. The angles are computed after dividing by the norms (Eq.6/8), so
    // the input scale does not matter.
    static double theta_min(const std::vector<std::vector<double>>& V0) {
        double tmin = std::numeric_limits<double>::max();
        int N = static_cast<int>(V0.size());
        for (int i = 0; i < N; ++i)
            for (int j = i + 1; j < N; ++j) {
                double dot = 0.0, na = 0.0, nb = 0.0;
                for (std::size_t k = 0; k < V0[i].size(); ++k) {
                    dot += V0[i][k] * V0[j][k];
                    na  += V0[i][k] * V0[i][k];
                    nb  += V0[j][k] * V0[j][k];
                }
                double denom = std::sqrt(na) * std::sqrt(nb);
                if (denom < 1e-14) continue;
                // Eq.6/8: theta = arccos|a·b / (||a||·||b||)| — the acute angle.
                double c = std::min(1.0, std::abs(dot) / denom);
                tmin = std::min(tmin, std::acos(c));
            }
        return (N <= 1 || tmin == std::numeric_limits<double>::max())
            ? (SRV_PI / 4.0) : tmin;
    }

    static double compute_theta_c(double theta_min_c, int G, int G_max) {
        // Eq. 17: θ_c = (π − θ_min_c) × (G−1) / (2 × G_max) + θ_min_c/2
        double t = (G_max > 1)
            ? (SRV_PI - theta_min_c) * (G - 1) / (2.0 * G_max) + theta_min_c / 2.0
            : theta_min_c / 2.0;
        return std::max(theta_min_c / 2.0, std::min(t, SRV_PI / 2.0));
    }

    // Wrapper: Pc = vault[0..n) (backward compatibility).
    std::vector<std::vector<double>>
    extract(DataVault<Ind_t>& vault, int n, double theta_c) const {
        std::vector<int> Pc(n);
        for (int i = 0; i < n; ++i) Pc[i] = i;
        return extract(vault, Pc, n, theta_c);
    }

    // ── Main extraction: Algorithm 1 ─────────────────────────────────────────
    // Returns N unit reference vectors extracted from a candidate union
    // population Pc (set of active indices). Per Liu et al. 2022, Pc = St —
    // the leading fronts of the merged pool (>= N individuals).
    // An overload taking an explicit index set Pc was added; previously only
    // extract(vault, n, ...) over [0,n) existed. The thin wrapper below keeps
    // that older form.
    std::vector<std::vector<double>>
    extract(DataVault<Ind_t>& vault, const std::vector<int>& Pc,
            int N, double theta_c) const {
        int m = vault.objs_n();
        int n = static_cast<int>(Pc.size());   // number of clustering points
        // N is the number of clusters/RVs (the population size); it may be
        // smaller than n when Pc = St.

        // ── Step 0: normalise objectives → f'(x), then unit-sphere → f*(x) ──
        // f'_j(x) = (f_j(x) - z*_j) / (z^nad_j - z*_j)   [Eq. 2]
        // EdI(x)  = ||f'(x)||₂                              [Eq. 7]
        // f*(x)   = f'(x) / EdI(x)                          [Eq. 14]
        std::vector<double> zstar(m,  std::numeric_limits<double>::max());
        std::vector<double> znad (m, -std::numeric_limits<double>::max());
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(Pc[i]);
            for (int j = 0; j < m; ++j) {
                zstar[j] = std::min(zstar[j], o[j]);
                znad [j] = std::max(znad [j], o[j]);
            }
        }
        // f_prime[i] = f'(x_i), fstar[i] = f*(x_i), edi[i] = EdI(x_i)
        std::vector<std::vector<double>> f_prime(n, std::vector<double>(m));
        std::vector<std::vector<double>> fstar  (n, std::vector<double>(m));
        std::vector<double>              edi    (n, 0.0);
        for (int i = 0; i < n; ++i) {
            const auto& o = vault.objectives_of(Pc[i]);
            for (int j = 0; j < m; ++j) {
                double range = std::max(znad[j] - zstar[j], 1e-14);
                f_prime[i][j] = (o[j] - zstar[j]) / range;
                edi[i] += f_prime[i][j] * f_prime[i][j];
            }
            edi[i] = std::sqrt(edi[i]);
            double d = std::max(edi[i], 1e-14);
            for (int j = 0; j < m; ++j) fstar[i][j] = f_prime[i][j] / d;
        }

        // ── Algorithm 2: ADM-based initialisation ────────────────────────────
        // Pairwise angles θ(x_i, x_j) = arccos(fstar[i]·fstar[j])
        std::vector<std::vector<float>> ang(n, std::vector<float>(n, 0.0f));
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                double dot = 0.0;
                for (int k = 0; k < m; ++k) dot += fstar[i][k] * fstar[j][k];
                float a = static_cast<float>(
                    std::acos(std::max(-1.0, std::min(1.0, dot))));
                ang[i][j] = ang[j][i] = a;
            }

        // Local density ρ(x) [Eq. 9]: Σ_{y≠x, θ(y,x)≤θ_c} exp(-(θ/θ_c)²)
        float tc = static_cast<float>(theta_c);
        std::vector<double> rho(n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                if (ang[i][j] <= tc) {
                    double r = ang[i][j] / tc;
                    rho[i] += std::exp(-(r * r));
                }
            }

        // Separation distance δ(x) [Eq. 10]:
        //   δ(x) = min_{y: ρ(y)>ρ(x)} θ(x,y)
        //   for x with highest ρ: δ = π/2
        std::vector<double> delta(n, SRV_PI / 2.0);
        for (int i = 0; i < n; ++i) {
            double best = SRV_PI / 2.0;
            bool found = false;
            for (int j = 0; j < n; ++j) {
                if (j == i) continue;
                if (rho[j] > rho[i]) {
                    best  = std::min(best, static_cast<double>(ang[i][j]));
                    found = true;
                }
            }
            delta[i] = found ? best : SRV_PI / 2.0;
        }

        // Step 1: m extreme centroids — one per axis e_i = (0,..,1,..,0)
        // xe(i) = argmin_{x ∈ Pc} θ(x, e_i) = argmax dot(fstar[x], e_i) = argmax fstar[x][i]
        std::vector<bool> is_centroid(n, false);
        std::vector<int>  centroids;
        centroids.reserve(n);
        for (int axis = 0; axis < m; ++axis) {
            int best = -1; double best_val = -1.0;
            for (int i = 0; i < n; ++i) {
                if (fstar[i][axis] > best_val) { best_val = fstar[i][axis]; best = i; }
            }
            if (best >= 0 && !is_centroid[best]) {
                is_centroid[best] = true;
                centroids.push_back(best);
            }
        }

        // Step 2: remaining N−m centroids by descending δ(x) among unchosen
        // Sort all by δ descending, pick first N−m that are not yet centroids
        std::vector<int> by_delta(n);
        std::iota(by_delta.begin(), by_delta.end(), 0);
        std::sort(by_delta.begin(), by_delta.end(),
                  [&](int a, int b){ return delta[a] > delta[b]; });
        for (int idx : by_delta) {
            if (static_cast<int>(centroids.size()) >= N) break;
            if (!is_centroid[idx]) {
                is_centroid[idx] = true;
                centroids.push_back(idx);
            }
        }
        // Ensure exactly N centroids (edge cases: N < m or duplicate axes)
        for (int i = 0; i < n && static_cast<int>(centroids.size()) < N; ++i)
            if (!is_centroid[i]) { is_centroid[i] = true; centroids.push_back(i); }

        // ── Algorithm 3: Self-Adjustment ─────────────────────────────────────
        // c[k] = current centroid (as f* index into population for k<m,
        //        or as a free vector for k≥m).
        // We store centroids as vectors in fstar-space.
        int n_free = static_cast<int>(centroids.size());  // = N
        std::vector<std::vector<double>> C(n_free);
        for (int k = 0; k < n_free; ++k) C[k] = fstar[centroids[k]];

        // Initial assignment: each x → nearest centroid by angle
        std::vector<int> cl(n, 0);
        auto assign_all = [&]() -> int {
            int changed = 0;
            for (int i = 0; i < n; ++i) {
                double best_cos = -std::numeric_limits<double>::max();
                int    best_k   = 0;
                for (int k = 0; k < n_free; ++k) {
                    double dot = 0.0;
                    for (int j = 0; j < m; ++j) dot += fstar[i][j] * C[k][j];
                    if (dot > best_cos) { best_cos = dot; best_k = k; }
                }
                if (cl[i] != best_k) { cl[i] = best_k; ++changed; }
            }
            return changed;
        };
        assign_all();

        // Iterate: update free centroids (k ≥ m), keep extreme (k < m) fixed
        // Max 2m iterations per Algorithm 3 [line 8: t ≤ 2m && flag=false]
        for (int t = 0; t < 2 * m; ++t) {
            // Update centroids k = m..N-1 by Eq. 13-14:
            //   Eq.13: c_k = mean_{x in C_k} f*(x) — a centroid INSIDE the
            //          sphere (the mean of unit vectors, ||c_k|| <= 1);
            //   Eq.14: projection of the centroid onto the UNIT hypersphere,
            //          c_k/||c_k||.
            for (int k = m; k < n_free; ++k) {
                std::vector<double> nc(m, 0.0);
                int cnt = 0;
                for (int i = 0; i < n; ++i)
                    if (cl[i] == k) { for (int j=0;j<m;++j) nc[j] += fstar[i][j]; ++cnt; }
                if (cnt > 0) {
                    // FIX 2026-07-08:
                    // The comment used to mis-attribute this as "Eq.13:
                    // centroid on hypersphere". Eq.13 gives a centroid INSIDE
                    // the sphere (a mean, hence shorter); the projection onto
                    // the SURFACE of the unit hypersphere is Eq.14. Text only —
                    // the behaviour was already correct.
                    // Eq.14: normalize the centroid onto the unit hypersphere.
                    double norm = 0.0;
                    for (int j = 0; j < m; ++j) { nc[j] /= cnt; norm += nc[j]*nc[j]; }
                    norm = std::sqrt(std::max(norm, 1e-28));
                    for (int j = 0; j < m; ++j) nc[j] /= norm;
                    C[k] = nc;
                }
            }
            // Reassign; stop if nothing changed [flag=true in Algorithm 3]
            int changed = assign_all();
            if (changed == 0) break;
        }

        // ── Step 3: map centroids to unit hypersphere (Eq. 15) ───────────────
        // rv_i = C[i] (already on unit sphere from normalisation above)
        // For the extreme centroids (k < m): still the original fstar values.
        return C;   // N unit reference vectors
    }
};

} // namespace mootation
