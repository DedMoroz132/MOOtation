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
#include <cstring>
#include <limits>
#include <map>
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

// TASK 14, SUPPLEMENT 3, Д8.3. Between the steps of SMS-EMOA and MO-CMA-ES
// ((mu + 1): one offspring in, one point out) most points' exact counts do
// not change. A point's count below is its box less WFG's hypervolume of its
// limited set (every other point raised to it, the dominated rows dropped by
// nds); WFG's answer depends on nothing but those rows in that order and the
// reference point. ExactKept keeps, for every point of the step before, that
// input and the answer; a point whose input is the same this step, bit for
// bit (the point, r, the rows after nds), takes the answer without WFG. So
// the contributions — and the run — are those without it, bit for bit; only
// the time differs. A point the offspring's or the removed point's row
// touches, or any point when r moves (SMS-EMOA: r = nad(Q) + 1), is counted
// again.
struct ExactKept {
    std::vector<double> r;                                          // the last step's
    std::map<std::vector<double>, std::pair<std::vector<double>, double>> prev, next;
    std::size_t counted = 0, reused = 0;                            // over the run
};

inline bool same_bits(const double* a, const double* b, std::size_t n)
{
    return std::memcmp(a, b, n * sizeof(double)) == 0;
}

inline std::vector<double> contributions_exact(const std::vector<std::vector<double>>& F,
                                               const std::vector<double>& r,
                                               ExactKept* kept = nullptr)
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
            if (!kept) {
                sub = w.hv(L, rows, m);
            } else {                                                // Д8.3
                std::vector<double> in(L.begin(), L.begin() + static_cast<std::ptrdiff_t>(rows * m));
                const auto it = kept->prev.find(p);
                if (it != kept->prev.end() && kept->r.size() == m && same_bits(kept->r.data(), r.data(), m) &&
                    same_bits(it->first.data(), p.data(), m) && it->second.first.size() == in.size() &&
                    same_bits(it->second.first.data(), in.data(), in.size())) {
                    sub = it->second.second;
                    ++kept->reused;
                } else {
                    sub = w.hv(L, rows, m);                         // hv may reorder L: in is the copy
                    ++kept->counted;
                }
                kept->next[p] = {std::move(in), sub};
            }
        }
        c[i] = std::max(0.0, own - sub);
    }
    if (kept) {
        kept->r = r;
        kept->prev.swap(kept->next);
        kept->next.clear();
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

// The settings of the race for the least contributor (contributions_race).
struct RaceSettings {
    double eps = 1e-2;                 // race_eps: (1 + eps) times the least contribution
    double delta = 1e-6;               // race_delta: the probability of a wrong answer
    std::size_t max_samples = 1000000; // race_max_samples: the samples of one step, at most
    double gamma = 1.0;                // gamma of Eq. (3), in (0, 1]: not given in the paper
    double alpha = 0.2;                // §3.4.1: the current least sampled to alpha * Delta
    double abs_tol = 1e-12;            // race_abs_tol: a share of prod(r - ideal), see below
};

// What the last race did (the cores' settings report sums of these).
struct RaceStats {
    std::size_t samples = 0, exact = 0, rounds = 0;
    bool capped = false;
};

// The least hypervolume contributor by the race of Bringmann & Friedrich,
// "Approximating the least hypervolume contributor: NP-hard in general, but
// fast in practice", Theor. Comput. Sci. 425:104-116, 2012 (source
// hv-least-contributor_bringmann2012), Algorithm 1 with the heuristics of
// §3.4: for every candidate the bounding box [p, b] of its exclusive region
// (§3.1: b cut by every point no worse than p in all objectives but one — the
// box of contributions_mc), samples uniform in it, the estimate
//   V~_A = hits / samples * VOL(BB_A)                                   (2)
// and its deviation bound
//   Delta_R(A) = sqrt(log(2 n R^(1+gamma) (1+gamma) / (gamma delta)) / (2 samples)) VOL(BB_A)   (3)
// In round R every candidate still in the race is sampled until
// Delta_R(A) <= Delta_R = Delta_{R-1} / 2 (Delta_0 = the largest box), the
// current least one until alpha * Delta_R (§3.4.1, alpha = 0.2); a candidate
// with V~_A - Delta_R(A) > V~_LC + Delta_R(LC) leaves the race (4). The race
// ends with one candidate left, or when V~_A - Delta_R(A) > 0 and
// (V~_LC + Delta_R(LC)) / (V~_A - Delta_R(A)) <= 1 + eps for every other A:
// then with probability at least 1 - delta the current least contributes at
// most (1 + eps) times the least (Corollary 5). §3.4.2: a sample is tested
// only against the points that can cover part of the box, the largest share
// first. §3.4.3: a candidate whose sampling has cost more coordinate
// comparisons than the exact count of its few covering points would (the
// paper's bound for HSO, n_A C(n_A + d - 2, d - 1)) is counted exactly
// instead (WFG of those points clipped to its box), its Delta then 0.
// READINGS: gamma is not given in the paper — 1, the bound's end; the switch
// to the exact count is checked before a candidate's samples of a round are
// drawn (the comparisons they would cost included), not after; the number of
// samples a round needs is computed from (3), not tested sample by sample (the
// same count); at max_samples a step the race stops (stats.capped).
// TASK 14, SUPPLEMENT 3, Д8.5. A race stopped at max_samples answers with the
// candidate of the least upper bound on its contribution — V~ + Delta for one
// with samples (the bound (3) at its samples), its box's volume for one
// without, V for one counted exactly — not with the least estimate: where the
// least contributions are tiny shares of their boxes (M >= 5, measured on task
// 13's sets) most candidates have no hit at the cap, their estimates are all
// 0, and the first of them was an arbitrary choice.
// TASK 14, SUPPLEMENT 1, Д5. Before any sample a point that contributes
// nothing — a copy, or one another point weakly dominates, or one not
// strictly better than r — is known exactly (0); when there is one the race
// is over: 0 is the least possible, and the zero points are the answer,
// their tie left to the caller's rule (SMS-EMOA: the first; MO-CMA-ES: one
// at random). A relative eps means nothing when the least contribution is
// about 0, so the race also stops when every other candidate's lower bound is
// within abs_tol * prod(r - ideal) of the current least's upper bound (ideal:
// the set's least value in every objective) — 1e-12 of the region the
// reference point spans, the scale of the rounding in an exact count.
// Returns V~ for the candidates left in the race (stopped at max_samples:
// their upper bounds) and +inf for the others and for the points `skip` marks
// (they still cover the others' boxes): its argmin is the race's answer, ties
// left to the caller's rule.
template <typename RNG>
inline std::vector<double> contributions_race(const std::vector<std::vector<double>>& F,
                                              const std::vector<double>& r,
                                              const RaceSettings& set, RNG& rng,
                                              const std::vector<char>* skip = nullptr,
                                              RaceStats* stats = nullptr)
{
    const double inf = std::numeric_limits<double>::infinity();
    const std::size_t n = F.size(), m = r.size();
    std::vector<double> c(n, inf);
    RaceStats st;
    if (n == 0) { if (stats) *stats = st; return c; }

    struct Cand {
        std::size_t i;
        std::vector<double> lo, hi;            // the box [p, b]
        double vol = 0.0;
        std::vector<std::size_t> cover;        // the points that can cover part of it
        double hso = 0.0;                      // §3.4.3: the exact count's estimated cost
        std::size_t samples = 0, hits = 0;
        double ops = 0.0;
        bool exact = false;
        double v = 0.0, dev = 0.0;             // V~ and Delta_R
    };
    std::vector<Cand> S;
    for (std::size_t i = 0; i < n; ++i) {
        if (skip && (*skip)[i]) continue;
        const auto& p = F[i];
        Cand a;
        a.i = i;
        a.lo = p;
        a.hi = r;
        bool inside = true;
        for (std::size_t k = 0; k < m; ++k) inside = inside && p[k] < r[k];
        bool covered_whole = !inside;
        for (std::size_t q = 0; q < n && !covered_whole; ++q) {
            if (q == i) continue;
            std::size_t worse = 0, at = m;
            for (std::size_t j = 0; j < m; ++j)
                if (F[q][j] > p[j]) { ++worse; at = j; }
            covered_whole = worse == 0;            // a copy of p: no region of its own
            if (worse == 1) a.hi[at] = std::min(a.hi[at], F[q][at]);
        }
        if (!covered_whole) {
            a.vol = 1.0;
            for (std::size_t k = 0; k < m; ++k) a.vol *= std::max(0.0, a.hi[k] - a.lo[k]);
        }
        if (a.vol <= 0.0) {                         // nothing of its own: exactly 0
            a.exact = true;
            S.push_back(std::move(a));
            continue;
        }
        std::vector<std::pair<double, std::size_t>> share;   // §3.4.2: the largest share first
        for (std::size_t q = 0; q < n; ++q) {
            if (q == i) continue;
            bool reach = true;
            double v = 1.0;
            for (std::size_t j = 0; j < m && reach; ++j) {
                reach = F[q][j] < a.hi[j];
                v *= a.hi[j] - std::max(F[q][j], a.lo[j]);
            }
            if (reach) share.emplace_back(-v, q);
        }
        std::stable_sort(share.begin(), share.end());
        for (const auto& s : share) a.cover.push_back(s.second);
        const double na = static_cast<double>(a.cover.size());
        double binom = 1.0;                         // C(n_A + d - 2, d - 1)
        for (std::size_t t = 1; t + 1 <= m; ++t)
            binom = binom * (na + static_cast<double>(m) - 2.0 - static_cast<double>(t) + 1.0) /
                    static_cast<double>(t);
        a.hso = na * binom;
        S.push_back(std::move(a));
    }

    // the exact contribution of a: its box less what its covering points take of it
    auto count_exactly = [&](Cand& a) {
        std::vector<double> L;
        L.reserve(a.cover.size() * m);
        for (std::size_t q : a.cover)
            for (std::size_t k = 0; k < m; ++k) L.push_back(std::max(F[q][k], a.lo[k]));
        const double taken = a.cover.empty() ? 0.0
            : hypervolume::wfg(L.data(), a.cover.size(), m, a.hi.data());
        a.v = std::max(0.0, a.vol - taken);
        a.dev = 0.0;
        a.exact = true;
        ++st.exact;
    };
    auto sample = [&](Cand& a, std::size_t k) {
        std::vector<double> x(m);
        for (std::size_t s = 0; s < k; ++s) {
            for (std::size_t j = 0; j < m; ++j) x[j] = a.lo[j] + rnd::uniform01(rng) * (a.hi[j] - a.lo[j]);
            bool covered = false;
            for (std::size_t q : a.cover) {
                std::size_t j = 0;
                while (j < m && F[q][j] <= x[j]) ++j;
                a.ops += static_cast<double>(j + 1);
                if (j == m) { covered = true; break; }
            }
            if (!covered) ++a.hits;
        }
        a.samples += k;
        st.samples += k;
        a.v = static_cast<double>(a.hits) / static_cast<double>(a.samples) * a.vol;
    };

    // a contribution known to be 0: the least possible, the race is over
    {
        bool zero = false;
        for (const auto& a : S) zero = zero || (a.exact && a.v == 0.0);
        if (zero) {
            for (const auto& a : S) if (a.exact && a.v == 0.0) c[a.i] = 0.0;
            if (stats) *stats = st;
            return c;
        }
    }
    double tol = set.abs_tol;                       // abs_tol * prod(r - ideal)
    for (std::size_t k = 0; k < m; ++k) {
        double lo_k = inf;
        for (const auto& f : F) lo_k = std::min(lo_k, f[k]);
        tol *= std::max(0.0, r[k] - lo_k);
    }
    const double nn = static_cast<double>(S.size());
    double D = 0.0;
    for (const auto& a : S) D = std::max(D, a.vol);
    std::vector<std::size_t> live(S.size());
    for (std::size_t t = 0; t < S.size(); ++t) live[t] = t;
    std::size_t lc = S.size();                      // the current least: none before round 1
    for (std::size_t R = 1; !live.empty(); ++R) {
        st.rounds = R;
        D *= 0.5;
        const double g = set.gamma;
        const double logterm = std::log(2.0 * nn * std::pow(static_cast<double>(R), 1.0 + g) *
                                        (1.0 + g) / (g * set.delta));
        for (std::size_t t : live) {
            Cand& a = S[t];
            if (a.exact) continue;
            const double target = (t == lc ? set.alpha : 1.0) * D;
            // (3) solved for the samples: Delta_R(a) <= target
            const double want = std::ceil(logterm * a.vol * a.vol / (2.0 * target * target));
            if (want > static_cast<double>(a.samples)) {
                const double more = want - static_cast<double>(a.samples);
                if (a.ops + more * (static_cast<double>(a.cover.size()) + 1.0) > a.hso) {
                    count_exactly(a);                 // §3.4.3
                    continue;
                }
                if (static_cast<double>(st.samples) + more > static_cast<double>(set.max_samples)) {
                    st.capped = true;
                    break;
                }
                sample(a, static_cast<std::size_t>(more));
            }
            a.dev = std::sqrt(logterm / (2.0 * static_cast<double>(a.samples))) * a.vol;
        }
        lc = live.front();
        for (std::size_t t : live) if (S[t].v < S[lc].v) lc = t;   // the first of equal ones
        if (st.capped) break;
        const double up = S[lc].v + S[lc].dev;
        std::vector<std::size_t> keep;
        for (std::size_t t : live)
            if (t == lc || !(S[t].v - S[t].dev > up)) keep.push_back(t);
        live.swap(keep);
        if (live.size() == 1) break;
        bool done = true, all_exact = true;
        for (std::size_t t : live) {
            all_exact = all_exact && S[t].exact;
            if (t == lc) continue;
            const double down = S[t].v - S[t].dev;
            done = done && ((down > 0.0 && up / down <= 1.0 + set.eps) || up - down <= tol);
        }
        if (done || all_exact) break;
    }
    for (std::size_t t : live) {
        const Cand& a = S[t];
        if (!st.capped || a.exact) c[a.i] = a.v;
        else c[a.i] = a.samples == 0 ? a.vol : std::min(a.vol, a.v + a.dev);   // Д8.5
    }
    if (stats) *stats = st;
    return c;
}

// How SMS-EMOA and MO-CMA-ES compute a step's contributions (task 13 of
// 2026-10-19, §3; task 14 of 2026-10-20, §11; sms_emoa.hpp SMS-2, mo_cma_es.hpp
// MOCMA-5), from their knobs:
//   M = 2, 3                 exact (M = 3: HVC3D unless hv_contrib = wfg)
//   4 <= M <= hv_exact_max_m exact (default threshold 5): M = 4 HVC4D unless
//                            hv_contrib = wfg, else WFG (hv_algo = box: boxes)
//   M > hv_exact_max_m       the race for the least contributor (race, the
//                            default; contributions_race) or, with
//                            hv_contrib = mc, Monte-Carlo with n_samples
//                            (10 000) a step shared out equally
//   n_samples given, hv_contrib neither race nor mc
//                            from M = 4 n_samples decides, as before task 14:
//                            0 exact at every M, N > 0 Monte-Carlo with N
// The exact contributions' cost grows steeply with M, hence the threshold.
// exact_only (MO-CMA-ES; task 14, supplement 3, Д8.1): exact at every M;
// n_samples > 0 and hv_contrib = race | mc are refused (kExactOnly says why),
// hv_exact_max_m and race_* are not taken (the caller reports them ignored).
struct ContribChoice {
    static constexpr const char* kExactOnly =
        "MO-CMA-ES's step-size adaptation needs exact hypervolume contributions: an "
        "offspring that survives counts as a success, and a wrongly chosen point inflates "
        "sigma (task 14, measurement of section 11: DTLZ2 at 6 objectives, the race gave "
        "HV 0 and sigma 1.13, the exact contributions HV 0.71 and sigma 0.145)";
    static constexpr int kSamples = 10000;
    enum class Approx { race, mc };
    std::optional<int> n_samples;       // given by the caller
    int  exact_max_m = 5;               // hv_exact_max_m
    bool hvc = true;                    // hv_contrib: hvc (default) or wfg, the exact way
    Approx approx = Approx::race;       // hv_contrib: race (default) or mc, above the threshold
    bool approx_given = false;
    bool box = false;                   // hv_algo = box
    RaceSettings race;                  // race_eps, race_delta, race_max_samples
    std::size_t m = 0;                  // the objectives of the last step (settings())
    std::size_t race_steps = 0, race_samples = 0, race_exact = 0, race_capped = 0;
    std::size_t steps = 0;              // every step that compared contributions
    bool exact_only = false;            // MO-CMA-ES (Д8.1): exact at every M
    bool incremental = false;           // hv_incremental (Д8.3): WFG's counts kept
    ExactKept kept_exact;               // between the steps (ExactKept)

    enum class Way { exact, mc, race };
    Way way(std::size_t M) const
    {
        if (M < 4 || exact_only) return Way::exact;
        if (n_samples && !approx_given) return *n_samples > 0 ? Way::mc : Way::exact;
        if (static_cast<int>(M) <= exact_max_m) return Way::exact;
        return approx == Approx::race ? Way::race : Way::mc;
    }
    bool mc(std::size_t M) const { return way(M) == Way::mc; }
    std::size_t samples() const { return static_cast<std::size_t>(n_samples.value_or(kSamples)); }
    bool use_hvc(std::size_t M) const { return hvc && (M == 3 || M == 4); }

    // The contributions of F against r, as the knobs say (rng: the samples').
    // skip marks the points the caller keeps whatever (MO-CMA-ES's boundary
    // elements): the race leaves them out of the candidates; the other ways
    // compute them as any.
    template <typename Hvc, typename Rng>
    std::vector<double> compute(const std::vector<std::vector<double>>& F,
                                const std::vector<double>& r, Hvc& kept, Rng& rng,
                                const std::vector<char>* skip = nullptr)
    {
        m = r.size();
        ++steps;
        const Way w = way(m);
        if (w == Way::race) {
            RaceStats st;
            auto c = contributions_race(F, r, race, rng, skip, &st);
            ++race_steps;
            race_samples += st.samples;
            race_exact += st.exact;
            race_capped += st.capped ? 1 : 0;
            return c;
        }
        if (w == Way::mc) return contributions_mc(F, r, std::max<std::size_t>(1, samples() / F.size()), rng);
        if (use_hvc(m)) return kept.contributions(F, r);
        return box ? contributions_box(F, r) : contributions_exact(F, r, incremental ? &kept_exact : nullptr);
    }

    bool set_param(const std::string& k, double v)
    {
        if (exact_only && (k == "hv_exact_max_m" || k.rfind("race_", 0) == 0)) return false;
        if (k == "n_samples") {
            if (!(v >= 0.0 && v <= 1e9 && v == std::floor(v)))
                throw std::invalid_argument("n_samples must be a whole number >= 0");
            if (exact_only && v > 0.0)
                throw std::invalid_argument(std::string("n_samples > 0: ") + kExactOnly +
                                            "; n_samples must be 0 or not given");
            n_samples = static_cast<int>(v);
            return true;
        }
        if (k == "hv_exact_max_m") {
            if (!(v >= 0.0 && v <= 1000.0 && v == std::floor(v)))
                throw std::invalid_argument("hv_exact_max_m must be a whole number >= 0");
            exact_max_m = static_cast<int>(v);
            return true;
        }
        if (k == "race_eps") {
            if (!(v > 0.0 && v <= 1.0)) throw std::invalid_argument("race_eps must be in (0, 1]");
            race.eps = v;
            return true;
        }
        if (k == "race_delta") {
            if (!(v > 0.0 && v < 1.0)) throw std::invalid_argument("race_delta must be in (0, 1)");
            race.delta = v;
            return true;
        }
        if (k == "hv_incremental") {
            if (v != 0.0 && v != 1.0) throw std::invalid_argument("hv_incremental must be 0 or 1");
            incremental = v == 1.0;
            return true;
        }
        if (k == "race_abs_tol") {
            if (!(v >= 0.0 && v < 1.0)) throw std::invalid_argument("race_abs_tol must be in [0, 1)");
            race.abs_tol = v;
            return true;
        }
        if (k == "race_max_samples") {
            if (!(v >= 1.0 && v <= 1e12 && v == std::floor(v)))
                throw std::invalid_argument("race_max_samples must be a whole number >= 1");
            race.max_samples = static_cast<std::size_t>(v);
            return true;
        }
        return false;
    }
    bool set_word(const std::string& k, const std::string& w)
    {
        if (k == "hv_contrib") {
            if (w == "hvc" || w == "wfg") { hvc = w == "hvc"; return true; }
            if (w == "race" || w == "mc") {
                if (exact_only)
                    throw std::invalid_argument("hv_contrib = '" + w + "': " + kExactOnly +
                                                "; hv_contrib must be hvc or wfg");
                approx = w == "race" ? Approx::race : Approx::mc;
                approx_given = true;
                return true;
            }
            throw std::invalid_argument("hv_contrib = '" + w + "': hvc, wfg, race or mc");
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
        auto num = [](double v) {
            std::string s = std::to_string(v);
            s.erase(s.find_last_not_of('0') + 1);
            if (!s.empty() && s.back() == '.') s.pop_back();
            return s;
        };
        std::string how;
        const Way w = m == 0 ? Way::exact : way(m);
        if (m == 0)                how = "none (no step compared contributions)";
        else if (m == 2)           how = "exact (two objectives)";
        else if (w == Way::race)   how = "race (eps " + num(race.eps) + ", delta " + num(race.delta) + "; " +
                                         std::to_string(race_steps) + " steps, " + std::to_string(race_samples) +
                                         " samples, " + std::to_string(race_exact) + " counted exactly, " +
                                         std::to_string(race_capped) + " stopped at race_max_samples)";
        else if (w == Way::mc)     how = "monte-carlo, " + std::to_string(samples()) + " samples a step (" +
                                         (n_samples && !approx_given ? "n_samples given" :
                                          approx_given ? "hv_contrib = mc" : "above hv_exact_max_m") + ")";
        else if (use_hvc(m))       how = m == 3 ? "exact, hvc3d" : "exact, hvc4d";
        else                       how = box ? "exact, box" : "exact, wfg";
        std::vector<std::pair<std::string, std::string>> out;
        if (!exact_only) out.emplace_back("hv_exact_max_m", std::to_string(exact_max_m));
        out.emplace_back("n_samples", n_samples ? std::to_string(*n_samples) : "default");
        out.emplace_back("hv_contrib", std::string(hvc ? "hvc" : "wfg") +
                                           (exact_only ? "" : approx == Approx::race ? ", race" : ", mc"));
        out.emplace_back("hv_algo", box ? "box" : "auto");
        out.emplace_back("hv_incremental", incremental ? "1" : "0");
        if (kept_exact.counted + kept_exact.reused > 0)
            out.emplace_back("hv_incremental_reused",
                             num(static_cast<double>(kept_exact.reused) /
                                 static_cast<double>(kept_exact.counted + kept_exact.reused)));
        out.emplace_back("contributions", how + " at M = " + std::to_string(m));
        if (race_steps > 0) {
            // the steps the guarantee does not cover; a campaign warns above 1 %
            out.emplace_back("race_capped_steps", std::to_string(race_capped));
            out.emplace_back("race_capped_share",
                             num(static_cast<double>(race_capped) / static_cast<double>(steps)));
        }
        return out;
    }
};

}   // namespace mootation::hv
