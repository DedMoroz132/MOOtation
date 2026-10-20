#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// SDR — the strengthened dominance relation, and Deb's fast non-dominated sort
// by it.
// Y. Tian, R. Cheng, X. Zhang, Y. Su, Y. Jin, "A Strengthened Dominance
// Relation Considering Convergence and Diversity for Evolutionary
// Many-Objective Optimization", IEEE TEVC 23(2):331-345, 2019,
// doi:10.1109/TEVC.2018.2866854          (source: nsga2-sdr_tian2019_accepted)
//
// §III-A, for a set P of n points (NSGA-II/SDR: R = P ∪ Q):
//   normalise    f̄_i = (f_i − z_i^min) / (z_i^max − z_i^min), z^min and z^max
//                the minima and maxima of the set (the text after (14));
//   (13)         Con(x) = Σ_i f̄_i(x), summed in the order of the objectives;
//   (14)         θ_xy = arccos(f̄(x)·f̄(y) / (|f̄(x)| |f̄(y)|));
//   (15)         θ̄ = the ⌊n/2⌋-th smallest of the DISTINCT values of
//                { min_{q ∈ P∖{p}} θ_pq | p ∈ P } (Fig. 4: "the unique values");
//   (12)         x ≺ y  ⟺  Con(x) · max(1, θ_xy / θ̄) < Con(y), the two cases
//                of (12) in one expression (θ_xy ≤ θ̄ gives a factor of 1).
// From (12), x ≺ y implies Con(x) < Con(y): the relation is irreflexive,
// antisymmetric (Properties 1, 2) and acyclic, but not transitive (Property
// 3). So the layers come from Deb's fast non-dominated sort, which compares
// every pair directly (fast_sort below); a sort that leans on transitivity
// (the efficient non-dominated sort, deductive sort, an ND-tree) gives wrong
// layers, as the paper says after Property 3.
//
// The library's rules where the paper says nothing:
//   SDR-1 an objective with z^max = z^min (zero range) has f̄_i = 0 for every
//         point;
//   SDR-2 θ is computed as acos of the cosine clipped to [−1, 1]. A point
//         with f̄ = 0 (it is z^min in every objective) has no angle with any
//         point: it is left out of θ̄'s estimate — (15) runs over the n points
//         with f̄ ≠ 0 alone, each one's minimum angle to the others of them —
//         and in (12) its θ is taken as 0, a factor of 1: with Con = 0 it
//         dominates every point with Con > 0, and nothing dominates it;
//   SDR-3 fewer than ⌊n/2⌋ distinct minimum angles: θ̄ is the largest of them;
//         θ̄ = 0 (the ⌊n/2⌋-th is 0, which happens for n = 2 or 3): the
//         smallest positive one; none positive (every point on one ray, or
//         fewer than two points with f̄ ≠ 0): θ̄ = 0, and the relation compares
//         Con alone — the same as (12) with every θ_xy = 0.
// The arithmetic is fixed so that a reference can repeat it bit for bit:
// sums and dot products run over the objectives in order, the cosine is
// (f̄(x)·f̄(y) / |f̄(x)|) / |f̄(y)|, the angle std::acos of it.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "detmath.hpp"

namespace mootation {
namespace sdr {

// A set in the frame of (12)-(15): Con of every point, every pairwise angle
// and the niche size.
struct Frame {
    std::size_t n = 0;
    std::vector<double> con;      // Con(x), (13)
    std::vector<double> theta;    // θ_xy, n × n, row-major, (14)
    double theta_bar = 0.0;       // θ̄, (15); 0: Con alone decides (SDR-3)
};

// The niche size (15) from the minimum angle of every point of a set of n
// points (SDR-3 for the fallbacks).
inline double niche_size(std::vector<double> nearest, std::size_t n)
{
    if (nearest.empty() || n < 2) return 0.0;
    std::stable_sort(nearest.begin(), nearest.end());
    nearest.erase(std::unique(nearest.begin(), nearest.end()), nearest.end());
    const std::size_t k = n / 2;                               // ⌊n/2⌋, from 1
    const double t = k <= nearest.size() ? nearest[k - 1] : nearest.back();
    if (t > 0.0) return t;
    return nearest.size() > 1 ? nearest[1] : 0.0;             // nearest[0] is the 0
}

// The minima and maxima normalisation of the rows of F (SDR-1).
inline std::vector<std::vector<double>> normalize(const std::vector<std::vector<double>>& F)
{
    std::vector<std::vector<double>> out(F.size());
    if (F.empty()) return out;
    const std::size_t m = F[0].size();
    std::vector<double> lo = F[0], hi = F[0];
    for (const auto& f : F)
        for (std::size_t k = 0; k < m; ++k) {
            lo[k] = std::min(lo[k], f[k]);
            hi[k] = std::max(hi[k], f[k]);
        }
    for (std::size_t i = 0; i < F.size(); ++i) {
        out[i].resize(m);
        for (std::size_t k = 0; k < m; ++k) {
            const double range = hi[k] - lo[k];
            out[i][k] = range > 0.0 ? (F[i][k] - lo[k]) / range : 0.0;
        }
    }
    return out;
}

// Con, the angles and θ̄ of points already normalised (or, for the paper's
// Property 3, given as they are).
inline Frame frame_of(const std::vector<std::vector<double>>& fbar)
{
    Frame fr;
    const std::size_t n = fbar.size();
    fr.n = n;
    fr.con.assign(n, 0.0);
    fr.theta.assign(n * n, 0.0);
    std::vector<double> len(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double c = 0.0, s = 0.0;
        for (double v : fbar[i]) {
            c += v;
            s += v * v;
        }
        fr.con[i] = c;
        len[i] = std::sqrt(s);
    }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) {
            double t = 0.0;                                    // SDR-2
            if (len[i] > 0.0 && len[j] > 0.0) {
                double d = 0.0;
                for (std::size_t k = 0; k < fbar[i].size(); ++k) d += fbar[i][k] * fbar[j][k];
                const double c = d / len[i] / len[j];
                t = dm::acos(std::min(1.0, std::max(-1.0, c)));
            }
            fr.theta[i * n + j] = t;
            fr.theta[j * n + i] = t;
        }
    // (15) over the points with f̄ ≠ 0 (SDR-2)
    std::vector<double> nearest;
    for (std::size_t i = 0; i < n; ++i) {
        if (!(len[i] > 0.0)) continue;
        double t = std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < n; ++j)
            if (j != i && len[j] > 0.0) t = std::min(t, fr.theta[i * n + j]);
        nearest.push_back(t);
    }
    fr.theta_bar = nearest.size() > 1 ? niche_size(nearest, nearest.size()) : 0.0;
    return fr;
}

// The frame of a set of objective vectors: normalised by the set's own
// minima and maxima, as NSGA-II/SDR does R in every generation.
inline Frame make_frame(const std::vector<std::vector<double>>& F)
{
    return frame_of(normalize(F));
}

// x ≺_SDR y, (12): x and y are indices into the frame.
inline bool dominates(const Frame& fr, std::size_t x, std::size_t y)
{
    double cx = fr.con[x];
    if (fr.theta_bar > 0.0) {
        const double r = fr.theta[x * fr.n + y] / fr.theta_bar;
        if (r > 1.0) cx *= r;
    }
    return cx < fr.con[y];
}

// Deb's fast non-dominated sort (Deb et al. 2002, §III-A) by any relation
// dom(i, j) over the points 0 .. n−1: every pair compared directly, once in
// each direction; the layers in the order the procedure builds them, each
// member's dominated ones in index order. A cycle in dom would leave its
// points out of every layer; SDR has none.
template <class Dom>
std::vector<std::vector<int>> fast_sort(int n, Dom dom)
{
    std::vector<std::vector<int>> S(static_cast<std::size_t>(n));
    std::vector<int> count(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            if (dom(i, j)) {
                S[static_cast<std::size_t>(i)].push_back(j);
                ++count[static_cast<std::size_t>(j)];
            } else if (dom(j, i)) {
                S[static_cast<std::size_t>(j)].push_back(i);
                ++count[static_cast<std::size_t>(i)];
            }
        }
    std::vector<std::vector<int>> fronts(1);
    for (int i = 0; i < n; ++i)
        if (count[static_cast<std::size_t>(i)] == 0) fronts[0].push_back(i);
    while (!fronts.back().empty()) {
        std::vector<int> next;
        for (int i : fronts.back())
            for (int j : S[static_cast<std::size_t>(i)])
                if (--count[static_cast<std::size_t>(j)] == 0) next.push_back(j);
        fronts.push_back(std::move(next));
    }
    fronts.pop_back();
    return fronts;
}

// The layers of a set of objective vectors by SDR alone (no constraints):
// rank[i] = the index of i's layer, 0 for the non-dominated.
inline std::vector<int> layers(const std::vector<std::vector<double>>& F, double* theta_bar = nullptr)
{
    const Frame fr = make_frame(F);
    if (theta_bar) *theta_bar = fr.theta_bar;
    const auto fronts = fast_sort(static_cast<int>(F.size()), [&](int i, int j) {
        return dominates(fr, static_cast<std::size_t>(i), static_cast<std::size_t>(j));
    });
    std::vector<int> rank(F.size(), -1);
    for (std::size_t k = 0; k < fronts.size(); ++k)
        for (int i : fronts[k]) rank[static_cast<std::size_t>(i)] = static_cast<int>(k);
    return rank;
}

} // namespace sdr
} // namespace mootation
