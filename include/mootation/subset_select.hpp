#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Subset selection: k of the n points of a non-dominated set (task 15 of
// 2026-10-21, §4). An archive reduced to N, or the answer a hypervolume is
// taken of, depends on which k: these are the ways to choose them.
//
//   dss        distance-based subset selection as the library does it
//              (dss.hpp, dss::order): the first k in its order, bit for bit
//   hv_greedy  the hypervolume's greedy choice: one point at a time, the one
//              that adds most to the hypervolume of those chosen; lazily
//              (a point's gain only shrinks as the set grows — the
//              hypervolume is submodular —, so a gain counted at an earlier
//              step bounds it now and is counted again only when it heads the
//              queue; Minoux's lazy greedy). Ties go to the lowest index.
//   hv_ls_rn   local search by swaps from the greedy choice, with candidate
//              lists: Korogi & Tanabe, "Speeding up local search for the
//              indicator-based subset selection problem by a candidate list
//              strategy", IEEE TEVC 2025 (source
//              issp-local-search-candidate-list_korogi2025_arxiv), LS-RN of
//              their §IV-D, Algorithm 4: first-improvement swaps (Algorithm 2)
//              of a chosen point s for a point of its list L_s not chosen,
//              until no swap of a pass improves the hypervolume, first with
//              lists of l^R = 20 random points (drawn once, from seed), then
//              with lists of the l^N = 20 nearest (Euclidean, in the
//              objectives; Algorithm 3). Their search starts from a random
//              subset; this one from hv_greedy's.
//
// The extreme points — the least value of each objective, the first such
// point by index — are always chosen and never swapped out (as the
// library's archive keeps them); with k at most their number, the first k of
// them in the objectives' order.
//
// The hypervolume against ref (each objective of every point below it to
// count). A gain or a contribution is an exclusive hypervolume computed by
// WFG (hv_contribution.hpp, detail::exact_one: the point's box less WFG's
// hypervolume of its limited set) — the computation contributions_exact
// makes from five objectives on; at three and four the cores' hvc gives the
// same values up to the last bits. A swap improves when it adds more than
// 1e-12 of the reference box's volume above the ideal point, so rounding
// cannot make the search go round.
//
// Indicators other than the hypervolume (the additive epsilon) are left out
// on purpose: Korogi & Tanabe, GECCO 2025 (source
// issp-landscape_korogi2025_arxiv), find that landscape multimodal, and the
// candidate lists fail on it. The IGD+ greedy choice is Python's
// (mootation.subset_select, method "igdp_greedy": run/cover.py's floor).
// ============================================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dss.hpp"
#include "hv_contribution.hpp"
#include "hypervolume.hpp"
#include "random.hpp"

namespace mootation {
namespace subset {

// The least value of each objective, the first such point by index; each
// point once, in the objectives' order.
inline std::vector<std::size_t> extremes(const std::vector<std::vector<double>>& F)
{
    std::vector<std::size_t> E;
    if (F.empty()) return E;
    const std::size_t m = F[0].size();
    for (std::size_t j = 0; j < m; ++j) {
        std::size_t best = 0;
        for (std::size_t i = 1; i < F.size(); ++i)
            if (F[i][j] < F[best][j]) best = i;
        if (std::find(E.begin(), E.end(), best) == E.end()) E.push_back(best);
    }
    return E;
}

// The hypervolume p adds to the points `chosen` of F: its exclusive
// hypervolume in chosen ∪ {p} (WFG).
inline double gain(const std::vector<std::vector<double>>& F, const std::vector<std::size_t>& chosen,
                   std::size_t p, const std::vector<double>& ref,
                   hypervolume::detail::Wfg& w, std::vector<double>& L,
                   std::vector<std::vector<double>>& rows)
{
    rows.clear();
    for (std::size_t c : chosen) rows.push_back(F[c]);
    rows.push_back(F[p]);
    return hv::detail::exact_one(rows, ref, rows.size() - 1, w, L, nullptr);
}

// hv_greedy: the extremes, then lazily the point that adds most.
inline std::vector<std::size_t> hv_greedy(const std::vector<std::vector<double>>& F, std::size_t k,
                                          const std::vector<double>& ref)
{
    const std::size_t n = F.size(), m = ref.size();
    std::vector<std::size_t> S = extremes(F);
    if (k <= S.size()) { S.resize(k); return S; }
    hypervolume::detail::Wfg w(ref.data(), m);
    std::vector<double> L;
    std::vector<std::vector<double>> rows;
    std::vector<char> in(n, 0);
    for (std::size_t s : S) in[s] = 1;
    // (bound, -index): the largest bound first, the lowest index on a tie
    using Entry = std::pair<double, long long>;
    std::priority_queue<Entry> q;
    for (std::size_t i = 0; i < n; ++i)
        if (!in[i]) q.push({gain(F, S, i, ref, w, L, rows), -static_cast<long long>(i)});
    std::vector<std::size_t> stamp(n, S.size());       // the |S| a point's bound was counted at
    while (S.size() < k && !q.empty()) {
        const auto [bound, neg] = q.top();
        q.pop();
        const std::size_t i = static_cast<std::size_t>(-neg);
        if (stamp[i] == S.size()) {                     // counted against the present S: it is the best
            S.push_back(i);
            in[i] = 1;
            continue;
        }
        stamp[i] = S.size();
        q.push({gain(F, S, i, ref, w, L, rows), neg});
        (void)bound;
    }
    return S;
}

// hv_ls_rn: the swaps of Korogi & Tanabe's LS-RN from the greedy choice.
inline std::vector<std::size_t> hv_ls_rn(const std::vector<std::vector<double>>& F, std::size_t k,
                                         const std::vector<double>& ref, std::uint32_t seed,
                                         std::size_t l = 20)
{
    const std::size_t n = F.size(), m = ref.size();
    std::vector<std::size_t> S = hv_greedy(F, k, ref);
    if (S.size() >= n || S.size() <= extremes(F).size()) return S;
    const std::size_t fixed = std::min(extremes(F).size(), S.size());   // S[0, fixed): the extremes
    std::vector<char> in(n, 0);
    for (std::size_t s : S) in[s] = 1;
    double vol = 1.0;                                   // the reference box above the ideal point
    for (std::size_t j = 0; j < m; ++j) {
        double lo = ref[j];
        for (const auto& f : F) lo = std::min(lo, f[j]);
        vol *= ref[j] - lo;
    }
    const double tol = 1e-12 * vol;
    hypervolume::detail::Wfg w(ref.data(), m);
    std::vector<double> L;
    std::vector<std::vector<double>> rows;

    // Algorithm 2 with the lists `lists`: first improvement, until a pass
    // improves nothing
    auto search = [&](const std::vector<std::vector<std::size_t>>& lists) {
        bool improved = true;
        while (improved) {
            improved = false;
            for (std::size_t t = fixed; t < S.size(); ++t) {
                std::vector<std::size_t> rest(S.begin(), S.end());
                rest.erase(rest.begin() + static_cast<std::ptrdiff_t>(t));
                const double own = gain(F, rest, S[t], ref, w, L, rows);   // s's contribution to S
                for (std::size_t p : lists[S[t]]) {
                    if (in[p]) continue;
                    if (gain(F, rest, p, ref, w, L, rows) - own > tol) {
                        in[S[t]] = 0;
                        in[p] = 1;
                        S[t] = p;
                        improved = true;
                        break;                          // the next s of the new S
                    }
                }
            }
        }
    };

    // L^R: l points of P \ {p} drawn without replacement, once (from seed)
    std::mt19937 rng(seed);
    std::vector<std::vector<std::size_t>> lists(n);
    const std::size_t lr = std::min(l, n - 1);
    for (std::size_t p = 0; p < n; ++p) {
        std::vector<std::size_t> others;
        for (std::size_t q = 0; q < n; ++q) if (q != p) others.push_back(q);
        for (std::size_t a = 0; a < lr; ++a) {              // a partial Fisher-Yates
            rnd::uniform_int_distribution<std::size_t> pick(a, others.size() - 1);
            std::swap(others[a], others[pick(rng)]);
        }
        others.resize(lr);
        lists[p] = std::move(others);
    }
    search(lists);
    // L^N: the l nearest (Euclidean in the objectives; ties by index)
    for (std::size_t p = 0; p < n; ++p) {
        std::vector<std::pair<double, std::size_t>> d;
        for (std::size_t q = 0; q < n; ++q) {
            if (q == p) continue;
            double s2 = 0.0;
            for (std::size_t j = 0; j < m; ++j) s2 += (F[p][j] - F[q][j]) * (F[p][j] - F[q][j]);
            d.push_back({s2, q});
        }
        std::partial_sort(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(lr), d.end());
        lists[p].clear();
        for (std::size_t a = 0; a < lr; ++a) lists[p].push_back(d[a].second);
    }
    search(lists);
    return S;
}

// k of the rows of F by `method` ("dss", "hv_greedy", "hv_ls_rn"). ref: the
// hypervolume's reference point (the hv methods; every point must be below
// it to count). seed: hv_ls_rn's random lists.
inline std::vector<std::size_t> select(const std::vector<std::vector<double>>& F, std::size_t k,
                                       const std::string& method, const std::vector<double>& ref = {},
                                       std::uint32_t seed = 1)
{
    if (k >= F.size()) {
        std::vector<std::size_t> all(F.size());
        for (std::size_t i = 0; i < all.size(); ++i) all[i] = i;
        return all;
    }
    if (method == "dss") return dss::order(F, k);
    if (method != "hv_greedy" && method != "hv_ls_rn")
        throw std::invalid_argument("subset::select: method '" + method + "': dss, hv_greedy or hv_ls_rn");
    if (ref.size() != F[0].size())
        throw std::invalid_argument("subset::select: " + method + " needs ref, one value an objective");
    return method == "hv_greedy" ? hv_greedy(F, k, ref) : hv_ls_rn(F, k, ref, seed);
}

}   // namespace subset
}   // namespace mootation
