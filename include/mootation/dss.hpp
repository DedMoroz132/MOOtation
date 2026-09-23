#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Distance-based subset selection (DSS) of k points from a set, in C++: the
// same selection as python/mootation/run/archive.py `dss_order`, which every
// campaign uses to reduce a point set to N (reference fronts, the run archive,
// the sampling baselines). A core whose state is a set rather than a
// population (DMS) answers with it.
//
// In objectives normalised by the set's own range ((f − min)/(max − min), a
// zero range counting as 1): first the best point of every objective, then
// repeatedly the candidate the selected set covers WORST, "cover" being the
// IGD+ distance d+(s, c) = ||max(s − c, 0)|| minimised over the selected s.
// Ties go to the lowest index, so the order is deterministic, and it is
// incremental: the first j of order(F, k) are order(F, j).
//
// Sources: Singh, Bhattacharjee & Ray, IEEE TEVC 23(5), 2019 (DSS); the IGD+
// distance in its place, Chen, Ishibuchi & Shang, 2020 — neither is in this
// project's corpus; the procedure is the one specified for MOOtation on
// 2026-09-22 (archive.py).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace mootation::dss {

inline std::vector<std::size_t> order(const std::vector<std::vector<double>>& F, std::size_t k)
{
    const std::size_t n = F.size();
    k = std::min(k, n);
    std::vector<std::size_t> out;
    if (k == 0) return out;
    const std::size_t m = F[0].size();
    std::vector<double> lo(m, std::numeric_limits<double>::infinity());
    std::vector<double> hi(m, -std::numeric_limits<double>::infinity());
    for (const auto& f : F)
        for (std::size_t j = 0; j < m; ++j) {
            lo[j] = std::min(lo[j], f[j]);
            hi[j] = std::max(hi[j], f[j]);
        }
    std::vector<std::vector<double>> G(n, std::vector<double>(m));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j) {
            const double span = hi[j] - lo[j];
            G[i][j] = (F[i][j] - lo[j]) / (span > 0.0 ? span : 1.0);
        }
    std::vector<char> taken(n, 0);
    std::vector<double> cover(n, std::numeric_limits<double>::infinity());
    out.reserve(k);
    auto take = [&](std::size_t i) {
        taken[i] = 1;
        out.push_back(i);
        for (std::size_t c = 0; c < n; ++c) {
            double s = 0.0;
            for (std::size_t j = 0; j < m; ++j) {
                const double d = G[i][j] - G[c][j];
                if (d > 0.0) s += d * d;
            }
            cover[c] = std::min(cover[c], std::sqrt(s));
        }
    };
    for (std::size_t j = 0; j < m && out.size() < k; ++j) {       // every objective's best
        std::size_t best = 0;
        for (std::size_t i = 1; i < n; ++i)
            if (G[i][j] < G[best][j]) best = i;
        if (!taken[best]) take(best);
    }
    while (out.size() < k) {
        std::size_t arg = n;
        double worst = -1.0;
        for (std::size_t i = 0; i < n; ++i)
            if (!taken[i] && cover[i] > worst) { worst = cover[i]; arg = i; }
        if (arg == n) break;
        take(arg);
    }
    return out;
}

}   // namespace mootation::dss
