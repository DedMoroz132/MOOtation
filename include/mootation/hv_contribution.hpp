#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Hypervolume contributions of the points of a non-dominated set — the
// selection criterion of SMS-EMOA and of the (μ+1)-MO-CMA-ES:
//   ΔS(p, F) = S(F) − S(F \ {p})
// against a reference point r (Emmerich, Beume & Naujoks, EMO 2005, Eq. 3;
// source sms-emoa_emmerich2005). F must be mutually non-dominated; copies
// are allowed and contribute nothing.
//
//   M = 2, exact: Eq. 4 of that paper. Sorted by f1 ascending (so f2
//     descending), ΔS(s_i) = (f1(s_{i+1}) − f1(s_i))·(f2(s_{i−1}) − f2(s_i)),
//     the neighbours past either end being the reference point.
//   M ≥ 3, exact: ΔS(p) = Hyp({p}) − Hyp({limit(q, p) : q ≠ p}), limit taking
//     the worse of the two values in every objective — the exclusive
//     hypervolume of While, Bradstreet & Barone (TEVC 16(1), 2012, Eqs. 5-7),
//     computed with hypervolume.hpp's WFG.
//   Any M, Monte-Carlo: p's exclusive region lies in the box [p, b] with
//     b_k = min(r_k, min{q_k : q ≠ p, q_j ≤ p_j for every j ≠ k}) — a point no
//     worse than p in all objectives but k caps p's region in k. Samples
//     uniform in that box; the share no other point weakly dominates, times
//     the box's volume, estimates ΔS(p). Each point gets its own samples, so
//     the small contributions — the ones a selection compares — are measured
//     as finely as the large ones.
// A point not strictly better than r in every objective contributes nothing.
// ============================================================================

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

#include "hypervolume.hpp"

namespace mootation::hv {

inline std::vector<double> contributions_exact(const std::vector<std::vector<double>>& F,
                                               const std::vector<double>& r)
{
    const std::size_t n = F.size(), m = r.size();
    std::vector<double> c(n, 0.0);
    if (n == 0) return c;
    auto inside = [&](const std::vector<double>& p) {
        for (std::size_t k = 0; k < m; ++k) if (!(p[k] < r[k])) return false;
        return true;
    };
    if (m == 2) {
        std::vector<std::size_t> o;
        for (std::size_t i = 0; i < n; ++i) if (inside(F[i])) o.push_back(i);
        std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) {
            return F[a][0] < F[b][0] || (F[a][0] == F[b][0] && F[a][1] < F[b][1]);
        });
        for (std::size_t t = 0; t < o.size(); ++t) {
            const double right = t + 1 < o.size() ? F[o[t + 1]][0] : r[0];
            const double above = t > 0 ? F[o[t - 1]][1] : r[1];
            c[o[t]] = (right - F[o[t]][0]) * (above - F[o[t]][1]);
        }
        return c;
    }
    std::vector<std::vector<double>> L;
    for (std::size_t i = 0; i < n; ++i) {
        const auto& p = F[i];
        if (!inside(p)) continue;
        double own = 1.0;
        for (std::size_t k = 0; k < m; ++k) own *= r[k] - p[k];
        L.clear();
        for (std::size_t q = 0; q < n; ++q) {
            if (q == i) continue;
            std::vector<double> l(m);
            for (std::size_t k = 0; k < m; ++k) l[k] = std::max(F[q][k], p[k]);
            L.push_back(std::move(l));
        }
        c[i] = std::max(0.0, own - (L.empty() ? 0.0 : hypervolume::wfg(L, r)));
    }
    return c;
}

template <typename RNG>
inline std::vector<double> contributions_mc(const std::vector<std::vector<double>>& F,
                                            const std::vector<double>& r,
                                            std::size_t samples_per_point, RNG& rng)
{
    const std::size_t n = F.size(), m = r.size();
    std::vector<double> c(n, 0.0);
    if (n == 0 || samples_per_point == 0) return c;
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::vector<double> others, S(samples_per_point * m), b(m);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& p = F[i];
        for (std::size_t k = 0; k < m; ++k) b[k] = r[k];
        others.clear();
        bool covered_whole = false;
        for (std::size_t q = 0; q < n && !covered_whole; ++q) {
            if (q == i) continue;
            std::size_t worse = 0, at = m;           // objectives where q is worse than p
            for (std::size_t j = 0; j < m; ++j)
                if (F[q][j] > p[j]) { ++worse; at = j; }
            covered_whole = worse == 0;               // a copy of p: no region of its own
            if (worse == 1) b[at] = std::min(b[at], F[q][at]);
            others.insert(others.end(), F[q].begin(), F[q].end());
        }
        if (covered_whole) continue;
        double vol = 1.0;
        for (std::size_t k = 0; k < m; ++k) vol *= std::max(0.0, b[k] - p[k]);
        if (vol <= 0.0) continue;
        for (std::size_t s = 0; s < samples_per_point; ++s)
            for (std::size_t k = 0; k < m; ++k)
                S[s * m + k] = p[k] + U01(rng) * (b[k] - p[k]);
        const std::size_t hit =
            others.empty() ? 0 : hypervolume::covered(others.data(), others.size() / m, m, S.data(), samples_per_point);
        c[i] = vol * static_cast<double>(samples_per_point - hit) /
               static_cast<double>(samples_per_point);
    }
    return c;
}

}   // namespace mootation::hv
