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
//     computed with hypervolume.hpp's WFG; contributions_box computes the
//     limited sets' hypervolumes by box decomposition instead (hv_box.hpp,
//     hv_algo = "box"), the same values up to the order of the sums.
//   Any M, Monte-Carlo: p's exclusive region lies in the box [p, b] with
//     b_k = min(r_k, min{q_k : q ≠ p, q_j ≤ p_j for every j ≠ k}) — a point no
//     worse than p in all objectives but k caps p's region in k. Samples
//     uniform in that box; the share no other point weakly dominates, times
//     the box's volume, estimates ΔS(p), without bias. Each point gets its
//     own samples, but how finely its contribution is measured depends on the
//     contribution's share of the box: near the front's edges the box can be
//     far larger than the contribution, few samples or none fall in it, and
//     the estimate is often 0 though the contribution is not small.
// A point not strictly better than r in every objective contributes nothing.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hv_box.hpp"
#include "hypervolume.hpp"
#include "random.hpp"

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
    // Every point's limited set in one flat buffer and one WFG evaluator for
    // them all: hypervolume::wfg on the same rows, which keeps the rows
    // strictly better than r, drops the dominated ones and recurses — the
    // same operations, without a vector per row and an evaluator per point.
    auto inside_row = [m](const double* p, const std::vector<double>& ref) {
        std::size_t j = 0;
        while (j < m && p[j] < ref[j]) ++j;           // false for NaN as well, as in wfg
        return j == m;
    };
    hypervolume::detail::Wfg w(r.data(), m);
    std::vector<double> L;
    L.reserve(n * m);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& p = F[i];
        if (!inside(p)) continue;
        double own = 1.0;
        for (std::size_t k = 0; k < m; ++k) own *= r[k] - p[k];
        L.clear();
        std::size_t rows = 0;
        for (std::size_t q = 0; q < n; ++q) {
            if (q == i) continue;
            const std::size_t at = L.size();
            for (std::size_t k = 0; k < m; ++k) L.push_back(std::max(F[q][k], p[k]));
            if (inside_row(L.data() + at, r)) ++rows;
            else L.resize(at);                        // as wfg drops it
        }
        double sub = 0.0;
        if (rows > 0) {
            rows = w.nds(L, rows, m);
            sub = w.hv(L, rows, m);
        }
        c[i] = std::max(0.0, own - sub);
    }
    return c;
}

// contributions_exact() with every limited set's hypervolume by box
// decomposition (hv_box.hpp): the same values up to the last bits.
inline std::vector<double> contributions_box(const std::vector<std::vector<double>>& F,
                                             const std::vector<double>& r)
{
    const std::size_t n = F.size(), m = r.size();
    if (m <= 2) return contributions_exact(F, r);          // Eq. 4: nothing to decompose
    std::vector<double> c(n, 0.0);
    auto inside = [&](const double* p) {
        for (std::size_t k = 0; k < m; ++k) if (!(p[k] < r[k])) return false;
        return true;
    };
    std::vector<double> L;
    L.reserve(n * m);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& p = F[i];
        if (!inside(p.data())) continue;
        double own = 1.0;
        for (std::size_t k = 0; k < m; ++k) own *= r[k] - p[k];
        L.clear();
        std::size_t rows = 0;
        for (std::size_t q = 0; q < n; ++q) {
            if (q == i) continue;
            const std::size_t at = L.size();
            for (std::size_t k = 0; k < m; ++k) L.push_back(std::max(F[q][k], p[k]));
            if (inside(L.data() + at)) ++rows;
            else L.resize(at);
        }
        const double sub = rows > 0 ? hypervolume::box(L.data(), rows, m, r.data()) : 0.0;
        c[i] = std::max(0.0, own - sub);
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
    rnd::uniform_real_distribution<double> U01(0.0, 1.0);
    std::vector<double> others, S(samples_per_point * m), b(m);
    // the points in increasing first objective, once: the order the count
    // reads them in (hypervolume::covered), whichever of them it reads
    std::vector<std::size_t> by_f1(n);
    for (std::size_t q = 0; q < n; ++q) by_f1[q] = q;
    std::stable_sort(by_f1.begin(), by_f1.end(),
                     [&F](std::size_t a, std::size_t b2) { return F[a][0] < F[b2][0]; });
    for (std::size_t i = 0; i < n; ++i) {
        const auto& p = F[i];
        for (std::size_t k = 0; k < m; ++k) b[k] = r[k];
        bool covered_whole = false;
        for (std::size_t q = 0; q < n && !covered_whole; ++q) {
            if (q == i) continue;
            std::size_t worse = 0, at = m;           // objectives where q is worse than p
            for (std::size_t j = 0; j < m; ++j)
                if (F[q][j] > p[j]) { ++worse; at = j; }
            covered_whole = worse == 0;               // a copy of p: no region of its own
            if (worse == 1) b[at] = std::min(b[at], F[q][at]);
        }
        if (covered_whole) continue;
        // Only a point no worse than b in every objective can cover a sample
        // of the box [p, b]; the others are left out of the count, which
        // they could not change.
        others.clear();
        for (std::size_t q : by_f1) {
            if (q == i) continue;
            bool reach = true;
            for (std::size_t j = 0; j < m && reach; ++j) reach = F[q][j] <= b[j];
            if (reach) others.insert(others.end(), F[q].begin(), F[q].end());
        }
        double vol = 1.0;
        for (std::size_t k = 0; k < m; ++k) vol *= std::max(0.0, b[k] - p[k]);
        if (vol <= 0.0) continue;
        for (std::size_t s = 0; s < samples_per_point; ++s)
            for (std::size_t k = 0; k < m; ++k)
                S[s * m + k] = p[k] + U01(rng) * (b[k] - p[k]);
        const std::size_t hit = others.empty() ? 0
            : hypervolume::covered_sorted(others.data(), others.size() / m, m, S.data(),
                                          samples_per_point);
        c[i] = vol * static_cast<double>(samples_per_point - hit) /
               static_cast<double>(samples_per_point);
    }
    return c;
}

// How SMS-EMOA and MO-CMA-ES compute a step's contributions (task 13 of
// 2026-10-19, §3; sms_emoa.hpp SMS-2, mo_cma_es.hpp MOCMA-5), from their knobs:
//   M = 2, 3                exact (M = 3: HVC3D unless hv_contrib = wfg)
//   4 <= M <= hv_exact_max_m exact (default threshold 5): M = 4 HVC4D unless
//                           hv_contrib = wfg, else WFG (hv_algo = box: boxes)
//   M > hv_exact_max_m      Monte-Carlo, 10 000 samples a step
//   n_samples given         from M = 4 it decides: > 0 Monte-Carlo with that
//                           many samples, 0 exact, at every M
// The exact contributions' cost grows steeply with M, hence the threshold.
struct ContribChoice {
    static constexpr int kSamples = 10000;
    std::optional<int> n_samples;       // given by the caller
    int  exact_max_m = 5;               // hv_exact_max_m
    bool hvc = true;                    // hv_contrib: hvc (default) or wfg
    bool box = false;                   // hv_algo = box
    std::size_t m = 0;                  // the objectives of the last step (settings())

    bool mc(std::size_t M) const
    {
        if (M < 4) return false;
        if (n_samples) return *n_samples > 0;
        return static_cast<int>(M) > exact_max_m;
    }
    std::size_t samples() const { return static_cast<std::size_t>(n_samples.value_or(kSamples)); }
    bool use_hvc(std::size_t M) const { return hvc && (M == 3 || M == 4); }

    // The contributions of F against r, as the knobs say (rng: Monte-Carlo's).
    template <typename Hvc, typename Rng>
    std::vector<double> compute(const std::vector<std::vector<double>>& F,
                                const std::vector<double>& r, Hvc& kept, Rng& rng)
    {
        m = r.size();
        if (mc(m)) return contributions_mc(F, r, std::max<std::size_t>(1, samples() / F.size()), rng);
        if (use_hvc(m)) return kept.contributions(F, r);
        return box ? contributions_box(F, r) : contributions_exact(F, r);
    }

    bool set_param(const std::string& k, double v)
    {
        if (k == "n_samples") {
            if (!(v >= 0.0 && v <= 1e9 && v == std::floor(v)))
                throw std::invalid_argument("n_samples must be a whole number >= 0");
            n_samples = static_cast<int>(v);
            return true;
        }
        if (k == "hv_exact_max_m") {
            if (!(v >= 0.0 && v <= 1000.0 && v == std::floor(v)))
                throw std::invalid_argument("hv_exact_max_m must be a whole number >= 0");
            exact_max_m = static_cast<int>(v);
            return true;
        }
        return false;
    }
    bool set_word(const std::string& k, const std::string& w)
    {
        if (k == "hv_contrib") {
            if (w != "wfg" && w != "hvc")
                throw std::invalid_argument("hv_contrib = '" + w + "': wfg or hvc");
            hvc = w == "hvc";
            return true;
        }
        if (k == "hv_algo") {                           // auto: WFG for the contributions
            if (w != "auto" && w != "wfg" && w != "box")
                throw std::invalid_argument("hv_algo = '" + w + "': auto, wfg or box");
            box = w == "box";
            return true;
        }
        return false;
    }

    // The knobs and the method of the run (Result.settings, a campaign's meta.json).
    std::vector<std::pair<std::string, std::string>> settings() const
    {
        std::string how;
        if (m == 0)           how = "none (no step compared contributions)";
        else if (m == 2)      how = "exact (two objectives)";
        else if (mc(m))       how = "monte-carlo, " + std::to_string(samples()) + " samples a step (" +
                                    (n_samples ? "n_samples given" : "above hv_exact_max_m") + ")";
        else if (use_hvc(m))  how = m == 3 ? "exact, hvc3d" : "exact, hvc4d";
        else                  how = box ? "exact, box" : "exact, wfg";
        return {{"hv_exact_max_m", std::to_string(exact_max_m)},
                {"n_samples", n_samples ? std::to_string(*n_samples) : "default"},
                {"hv_contrib", hvc ? "hvc" : "wfg"},
                {"hv_algo", box ? "box" : "auto"},
                {"contributions", how + " at M = " + std::to_string(m)}};
    }
};

}   // namespace mootation::hv
