#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Deterministic near-uniform direction vectors on the unit sphere in the first
// octant, for an ARBITRARY count K.
//
// The M2M family (moead_m2m, sms_m2m, isde_rd) partitions the objective space
// with K direction vectors. A Das-Dennis lattice only realizes the counts
// C(H+m-1, m-1), so an arbitrary K is unattainable by a lattice at m >= 3 and
// the usual workaround — round K UP to the next attainable size — silently
// changes the subregion count and hence S_k = floor(N/K).
//
// This generator hits K exactly and uses NO RNG, so a core switching to it
// keeps its random stream: candidates from the smallest Das-Dennis lattice with
// at least 2K points, projected on the sphere; farthest-point sampling by angle
// starting from candidate 0; then 60 iterations of Riesz s-energy repulsion
// (s = 2, step 0.05, clamped into the first octant and renormalized) — the
// deterministic Riesz s-energy (C2 Energy) scheme.
//
// Extracted verbatim from moead_m2m.hpp / sms_m2m.hpp, which held two
// byte-identical private copies; all three cores now share this one.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <vector>

#include "../das_dennis.hpp"

namespace mootation {
namespace detail {

inline std::vector<std::vector<double>>
uniform_sphere_directions(int m, int K) {
        auto normalize = [](std::vector<double>& v) {
            double n2 = 0.0; for (double x : v) n2 += x*x;
            double nn = std::sqrt(std::max(n2, 1e-300));
            for (double& x : v) x /= nn;
        };
        // 1) candidates
        int H = 1;
        while (H < 500 && das_dennis::n_vectors(m, H) < 2LL*K) ++H;
        auto cand = das_dennis::generate(m, H);
        for (auto& c : cand) normalize(c);
        int nc = static_cast<int>(cand.size());
        auto cosv = [m](const std::vector<double>& a, const std::vector<double>& b) {
            double d = 0.0;
            for (int j = 0; j < m; ++j) d += a[j]*b[j];
            return std::clamp(d, -1.0, 1.0);   // candidates are already unit-length
        };
        // 2) FPS by angle (min max-cos), deterministic start at cand[0]
        std::vector<std::vector<double>> R;
        R.reserve(K);
        std::vector<char>   used(nc, 0);
        std::vector<double> maxc(nc, -2.0);   // max cos to the selected ones
        int cur = 0;
        for (int t = 0; t < K && t < nc; ++t) {
            R.push_back(cand[cur]); used[cur] = 1;
            int nxt = -1; double best = 2.0;
            for (int i = 0; i < nc; ++i) {
                if (used[i]) continue;
                maxc[i] = std::max(maxc[i], cosv(cand[i], cand[cur]));
                if (maxc[i] < best) { best = maxc[i]; nxt = i; }
            }
            if (nxt < 0) break;
            cur = nxt;
        }
        for (std::size_t g = 0; static_cast<int>(R.size()) < K && !R.empty(); ++g)
            R.push_back(R[g % R.size()]);                 // guard (K > nc; dead: nc≥2K)
        // 3) Riesz s-energy: s=2, 60 iterations, step 0.05 (deterministic direction generator)
        int Kf = static_cast<int>(R.size());
        const double s = 2.0, step = 0.05;
        for (int it = 0; it < 60; ++it) {
            std::vector<std::vector<double>> Frc(Kf, std::vector<double>(m, 0.0));
            for (int a = 0; a < Kf; ++a)
                for (int b = 0; b < Kf; ++b) {
                    if (a == b) continue;
                    double d2 = 0.0;
                    for (int j = 0; j < m; ++j) {
                        double t = R[a][j] - R[b][j]; d2 += t*t;
                    }
                    double d = std::sqrt(std::max(d2, 1e-6));
                    double w = 1.0 / std::pow(d, s + 1.0);
                    for (int j = 0; j < m; ++j)
                        Frc[a][j] += (R[a][j] - R[b][j]) / d * w;
                }
            for (int a = 0; a < Kf; ++a) {
                std::vector<double> prev = R[a];
                double n2 = 0.0;
                for (int j = 0; j < m; ++j) {
                    R[a][j] += step * Frc[a][j];
                    if (R[a][j] < 0.0) R[a][j] = 0.0;     // first octant
                    n2 += R[a][j]*R[a][j];
                }
                if (n2 < 1e-18) R[a] = prev;              // degeneration → rollback
                else normalize(R[a]);
            }
        }
        return R;
    }

} // namespace detail
} // namespace mootation
