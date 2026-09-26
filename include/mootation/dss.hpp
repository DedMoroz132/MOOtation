#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Distance-based subset selection (DSS) of k points from a set, in C++: the
// same selection as python/mootation/run/archive.py `dss_order`, which every
// campaign uses to reduce a point set to N (reference fronts, the run archive,
// the sampling baselines). A core whose state is a set rather than a
// population (DMS) answers with it.
//
// In objectives normalised by the range of the points kept ((f − min)/(max −
// min), a zero range counting as 1): first the best kept point of every
// objective, then repeatedly the kept candidate the selected set covers WORST,
// "cover" being the IGD+ distance d+(c, s) = ||max(s − c, 0)|| minimised over
// the selected s (the candidate c in the place of IGD+'s reference point);
// then, the same way, the points set aside. Ties go to the lowest index, so
// the order is deterministic, and it is incremental: the first j of
// order(F, k) are order(F, j). It is archive.py's dss_order with ideal and
// nadir left None.
//
// ALMOST-DOMINATED POINTS GO LAST (2026-09-27). A point p is set aside when
// some other point q is better than p by more than DRS_LOSS in some objective
// while p is better than q by at most DRS_GAIN in every objective, in the
// normalised coordinates: a trade-off worse than 100 to 1 over a tenth of the
// range. Such points — the best value of one objective bought with a much
// worse value of another — were exactly what the per-objective seeds picked,
// and far from everything else, what the max-min steps picked next. The range
// is taken again from the points kept after every round that sets points
// aside, until a round sets none aside, so an outlier does not stretch the
// frame the rest is measured in; a round never sets aside every point left. A
// set with nothing to set aside is ordered exactly as before the rule.
//
// NO APPROXIMATION GUARANTEE. No bound holds for the covering radius DSS
// leaves — the largest d+ from a point of the set to the selection — against
// the best achievable with the same k: d+ is not symmetric and the seeds are
// forced, so the argument that bounds greedy max-min selection under a metric
// does not apply. It is a spread heuristic, not an optimal subset.
//
// Sources: DSS is Singh, Bhattacharjee & Ray, "Distance based subset
// selection for benchmarking in evolutionary multi/many-objective
// optimization", IEEE TEVC, 2019, Algo. 1 (source dss_singh2019). The procedure
// here is the one specified for MOOtation on 2026-09-22 and departs from
// Algo. 1 three times: the seeds are the M per-objective minima, the rule of
// Tanabe, Ishibuchi & Oyama (2017) that Singh et al. replace by ONE extreme
// point; the distance is d+ (Ishibuchi, Masuda, Tanigaki & Nojima, EMO 2015,
// Eq. 18, source igd-plus_ishibuchi2015 — its use in DSS after Chen, Ishibuchi &
// Shang, 2020, not in the corpus), not the Euclidean; the almost-dominated
// points go last (Singh et al. filter nothing: they select from reference
// sets). Singh et al.'s results on the spacing of the selected points assume
// the Euclidean distance and are not claimed here.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace mootation::dss {

inline constexpr double DRS_GAIN = 1e-3;   // p's largest gain over q, at most
inline constexpr double DRS_LOSS = 0.1;    // p's largest loss to q, more than

inline std::vector<std::size_t> order(const std::vector<std::vector<double>>& F, std::size_t k)
{
    const std::size_t n = F.size();
    k = std::min(k, n);
    std::vector<std::size_t> out;
    if (k == 0) return out;
    const std::size_t m = F[0].size();
    const double inf = std::numeric_limits<double>::infinity();

    // Normalise by the kept points' range and set aside the almost-dominated
    // ones, until a round sets none aside (archive.py set_aside).
    std::vector<char> kept(n, 1);
    std::vector<std::vector<double>> G(n, std::vector<double>(m));
    for (;;) {
        std::vector<double> lo(m, inf), hi(m, -inf);
        for (std::size_t i = 0; i < n; ++i) {
            if (!kept[i]) continue;
            for (std::size_t j = 0; j < m; ++j) {
                lo[j] = std::min(lo[j], F[i][j]);
                hi[j] = std::max(hi[j], F[i][j]);
            }
        }
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < m; ++j) {
                const double span = hi[j] - lo[j];
                G[i][j] = (F[i][j] - lo[j]) / (span > 0.0 ? span : 1.0);
            }
        std::vector<char> drop(n, 0);
        std::size_t n_kept = 0, n_drop = 0;
        for (std::size_t p = 0; p < n; ++p) {
            if (!kept[p]) continue;
            ++n_kept;
            for (std::size_t q = 0; q < n && !drop[p]; ++q) {
                if (!kept[q] || q == p) continue;
                double loss = -inf;
                bool small_gain = true;              // p better than q by <= DRS_GAIN everywhere
                for (std::size_t j = 0; j < m; ++j) {
                    const double d = G[p][j] - G[q][j];
                    if (-d > DRS_GAIN) { small_gain = false; break; }
                    loss = std::max(loss, d);
                }
                if (small_gain && loss > DRS_LOSS) drop[p] = 1;
            }
            if (drop[p]) ++n_drop;
        }
        if (n_drop == 0 || n_drop == n_kept) break;
        for (std::size_t i = 0; i < n; ++i)
            if (drop[i]) kept[i] = 0;
    }

    std::vector<char> taken(n, 0);
    std::vector<double> cover(n, inf);
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
    for (std::size_t j = 0; j < m && out.size() < k; ++j) {       // every objective's best kept
        std::size_t best = n;
        for (std::size_t i = 0; i < n; ++i)
            if (kept[i] && (best == n || G[i][j] < G[best][j])) best = i;
        if (!taken[best]) take(best);
    }
    while (out.size() < k) {
        bool kept_left = false;                                   // then the points set aside
        for (std::size_t i = 0; i < n; ++i)
            if (kept[i] && !taken[i]) { kept_left = true; break; }
        std::size_t arg = n;
        double worst = -1.0;
        for (std::size_t i = 0; i < n; ++i)
            if (!taken[i] && (kept[i] || !kept_left) && cover[i] > worst) {
                worst = cover[i];
                arg = i;
            }
        if (arg == n) break;
        take(arg);
    }
    return out;
}

}   // namespace mootation::dss
