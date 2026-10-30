#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// An entry of the cache of references built in C++ — task 17, §2.1
// (run/refs.py's build_entry; Python's stays as the cross-check):
//
//   ref     the reference front of n_ref rows (fronts.hpp)
//   ref3    the dense reference (registry.dense_reference), where the
//           problem has one: the first size of 1 000 ... 100 000 whose
//           reference step — the median distance of a point to its nearest
//           other, in the frame — is at most a third of the step of a front
//           of the population's size, or the largest; with both steps
//   floors  cover.igdp_floors at the problem's population N: IGD+ of the
//           best N rows the greedy choice finds (added one at a time, each
//           lowering IGD+ most, then single swaps while one improves, at
//           most three passes; ties to the lowest index) and of the first N
//           in DSS order, both against the normalised front itself
//   rows, distinct
//           the front's rows, and those distinct to 12 decimals
//
// The frame (ideal, nadir) is the registry's, as Python's entry has it.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "../archive_select.hpp"
#include "../decision_indicators.hpp"
#include "../indicators.hpp"
#include "../numpy_sum.hpp"
#include "../run_metrics.hpp"
#include "fronts.hpp"

namespace mootation::refs {

// registry.REFERENCE_VERSION of the references C++ builds: 5 (the library's
// draw where Python's took default_rng's)
constexpr int kReferenceVersion = 5;

constexpr std::size_t kDenseSizes[] = {1000, 2000, 5000, 10000, 20000, 50000, 100000};
constexpr double kDenseRatio = 3.0;

struct Frame {
    std::vector<double> ideal, nadir;
};

// (R - ideal) / span, span = nadir - ideal where that is positive, else 1
// (metrics._normalised; registry.dense_reference's np.where(hi > lo, ...))
inline Rows normalised(const Rows& R, const Frame& f)
{
    Rows G = R;
    for (std::size_t j = 0; j < R.m; ++j) {
        double span = f.nadir[j] - f.ideal[j];
        if (!(span > 0.0)) span = 1.0;
        for (std::size_t i = 0; i < R.n; ++i) G.row(i)[j] = (R.row(i)[j] - f.ideal[j]) / span;
    }
    return G;
}

// The exact frame of the polygon problems (task 17, addendum 6, §3): Polygon
// (Ishibuchi, Akedo & Nojima 2011) and MaF8 (Cheng et al. 2017), f_m the
// distance to vertex m of the regular M-gon on the unit circle, and MaF9, f_m
// the distance to the line through edge m. Their Pareto set is the polygon's
// interior, which holds every vertex and edge: the ideal is 0 everywhere. A
// distance is convex, so its largest value over the polygon is at a vertex:
// the nadir is the farthest vertex from vertex m (from line m for MaF9).
// Nothing for the other kinds (rotated copies take their base's spec).
inline std::optional<Frame> exact_frame(const json::Value& spec)
{
    const json::Value* kind = spec.find("kind");
    if (!kind || (kind->s != "polygon" && kind->s != "maf8" && kind->s != "maf9")) return std::nullopt;
    const std::size_t M = detail::count(spec, "M");
    const std::vector<double> V = detail::unit_polygon(M);
    Rows P = rows_of(M, 2);
    P.v = V;
    const Rows D = kind->s == "maf9" ? detail::edge_line_distances(P, V) : detail::vertex_distances(P, V);
    Frame f{std::vector<double>(M, 0.0), std::vector<double>(M, 0.0)};
    for (std::size_t m = 0; m < M; ++m)
        for (std::size_t v = 0; v < M; ++v) f.nadir[m] = std::max(f.nadir[m], D.row(v)[m]);
    return f;
}

// metrics.reference_step: the median distance of a row to its nearest other
inline double reference_step(const Rows& R)
{
    if (R.n < 2) return std::numeric_limits<double>::quiet_NaN();
    return run_metrics::median(indicators::nearest_other(R.v.data(), R.n, R.m));
}

struct Dense {
    Rows F;
    double step = 0.0, population_step = 0.0;
};

// registry.dense_reference
inline Dense dense_reference(const json::Value& spec, const Frame& frame, std::size_t pop, const Data& data = {})
{
    Dense d;
    d.population_step = reference_step(normalised(front(spec, pop, data), frame));
    for (std::size_t n : kDenseSizes) {
        d.F = front(spec, n, data);
        d.step = reference_step(normalised(d.F, frame));
        if (d.step <= d.population_step / kDenseRatio) break;
    }
    return d;
}

// IGD+ of the rows `idx` of G against G (metrics.igd_plus)
inline double igdp_of(const Rows& G, const std::vector<std::size_t>& idx)
{
    if (idx.empty()) return std::numeric_limits<double>::infinity();
    const Rows A = take(G, idx);
    std::vector<double> to_set(G.n);
    indicators::front_distances(A.v.data(), A.n, G.v.data(), G.n, G.m, to_set.data(), nullptr, nullptr);
    return np::mean(to_set.data(), to_set.size());
}

// cover.greedy_subset
inline std::vector<std::size_t> greedy_subset(const Rows& G, std::size_t n, int sweeps = 3)
{
    const std::size_t r = G.n, m = G.m;
    const double inf = std::numeric_limits<double>::infinity();
    // D[c, z] = d+(z, c) = ||max(c - z, 0)||
    std::vector<double> D(r * r), sq(m);
    for (std::size_t c = 0; c < r; ++c)
        for (std::size_t z = 0; z < r; ++z) {
            for (std::size_t j = 0; j < m; ++j) {
                const double t = std::max(G.row(c)[j] - G.row(z)[j], 0.0);
                sq[j] = t * t;
            }
            D[c * r + z] = std::sqrt(np::sum(sq.data(), m));
        }
    std::vector<double> row(r);
    // np.minimum(base[None, :], D).mean(axis=1): every candidate's IGD+ beside base
    auto costs = [&](const std::vector<double>& base) {
        std::vector<double> cost(r);
        for (std::size_t c = 0; c < r; ++c) {
            for (std::size_t z = 0; z < r; ++z) row[z] = std::min(base[z], D[c * r + z]);
            cost[c] = np::mean(row.data(), r);
        }
        return cost;
    };
    auto argmin = [](const std::vector<double>& v) {
        std::size_t best = 0;
        for (std::size_t k = 1; k < v.size(); ++k)
            if (v[k] < v[best]) best = k;
        return best;
    };
    std::vector<double> cur(r, inf);
    std::vector<std::size_t> chosen;
    for (std::size_t step = 0; step < std::min(n, r); ++step) {
        std::vector<double> cost = costs(cur);
        for (auto c : chosen) cost[c] = inf;
        const std::size_t c = argmin(cost);
        chosen.push_back(c);
        for (std::size_t z = 0; z < r; ++z) cur[z] = std::min(cur[z], D[c * r + z]);
    }
    for (int pass = 0; pass < sweeps; ++pass) {
        bool improved = false;
        for (std::size_t i = 0; i < chosen.size(); ++i) {
            std::vector<double> base(r, inf);
            for (std::size_t k = 0; k < chosen.size(); ++k)
                if (k != i)
                    for (std::size_t z = 0; z < r; ++z) base[z] = std::min(base[z], D[chosen[k] * r + z]);
            std::vector<double> cost = costs(base);
            for (std::size_t k = 0; k < chosen.size(); ++k)
                if (k != i) cost[chosen[k]] = inf;
            const std::size_t c = argmin(cost);
            for (std::size_t z = 0; z < r; ++z) row[z] = std::min(base[z], D[chosen[i] * r + z]);
            if (cost[c] < np::mean(row.data(), r) - 1e-15) {
                chosen[i] = c;
                improved = true;
            }
        }
        if (!improved) break;
    }
    return chosen;
}

struct Floors {
    double greedy = 0.0, dss = 0.0;
    std::size_t rows = 0, distinct = 0;
};

// cover.igdp_floors at one population
inline Floors igdp_floors(const Rows& ref, const Frame& frame, std::size_t pop)
{
    Floors f;
    const Rows G = normalised(ref, frame);
    f.rows = ref.n;
    f.distinct = distinct_rows(ref).n;
    f.dss = igdp_of(G, select::dss_order(ref, pop, frame.ideal, frame.nadir));
    f.greedy = igdp_of(G, greedy_subset(G, pop));
    return f;
}

}   // namespace mootation::refs
