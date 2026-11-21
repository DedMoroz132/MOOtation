#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// sweep: a plan over the problem's box that moves the box's faces outward
// where its best points sit on them, up to hard bounds (task 32, layer 4).
// Included at the end of strategy.hpp, after param.hpp.
//
// THE PLAN. The box is the problem's (lo, hi): what a person would sweep
// first. plan = sobol | lhs | uniform | halton_rand | sobol_owen (designs.hpp:
// n_points points of the design, its generator's seed the run's) or grid (a
// full grid: grid_n levels lo + (hi - lo) i / (grid_n - 1) on a real
// coordinate, every value on a listed one, coordinate 0 the slowest). Every
// point is evaluated once, in the plan's order, on one thread.
//
// THE COORDINATES. A coordinate with values (Problem::discrete: minimize's
// var_types, sorted and in the box) is listed: its points are its values, a
// design's u in [0, 1) the value of index floor(u K) of its K values (each
// value an equal share). A listed coordinate whose values are consecutive
// whole numbers is an integer coordinate: its faces move by whole numbers,
// its hard bounds rounded inward (ceil hard_lo, floor hard_hi: the genome's
// [ceil lo, floor hi], run_core.hpp). Any other coordinate is real: lo + u
// (hi - lo).
//
// THE EDGE RULE. After the plan and after each step: the top_k best points
// evaluated so far (the less violating, then the lower f, the first of
// equals; failed points not among them). Point x is on coordinate j's lower
// face where x_j - lo_j <= edge_tol w0_j, on its upper face where hi_j - x_j
// <= edge_tol w0_j (lo, hi the current box, w0_j the starting box's width:
// the band stays as thick as the plan's spacing, which the layers keep;
// SWEEP-7). A face fires where
// at least edge_share of the top_k points are on it and it is not at its hard
// bound (sweep::edge_faces; the grid of keys of campaign/components.hpp,
// COMP-15, judges its faces by it too). A step moves every face that fired outward, in the order coordinate
// 0's lower, its upper, coordinate 1's lower, ...:
//   real        by expand_step w0_j (w0_j: the starting box's width), at
//               most to the hard bound;
//   integer     by max(1, ceil(expand_step w0_j)) of its values, at most to
//               the hard bound;
//   grid, real  by K = max(1, ceil(expand_step (grid_n - 1))) levels of the
//               starting spacing w0_j / (grid_n - 1), the last one the hard
//               bound where the next level would pass it.
// The new layer of a face is the box between the new face and the old one
// (the new face in, the old one out), the other coordinates over the box as
// it is at that face's turn (a face moved before it in the step included):
// design plans lay ceil(n_points V_layer / V_0) points of the design in it
// (V the product over the coordinates of the width, a listed one's count of
// values; a coordinate of starting width 0 left out), seeded start_seed(seed,
// t) for the t-th layer of the run (strategy.hpp STRAT-1); the grid lays the
// full grid of its new levels times every other coordinate's levels. The
// steps go on while a face fires, the budget lasts and the stop rule allows
// (stop_rule.hpp: a step is one move of the box; stall, max_steps).
//
// THE BUDGET is the run's (task 31, layer 1): the plan's evaluations are
// counted by the run's evaluator, sweep_budget (0: none) caps what the plan
// may spend, and the inner solver (below) has what is left.
//
// THE POINTS AS STARTS. With inner = {"solver": ...} the plan's carry_points
// (1) best distinct points start the inner solver (param.hpp carry_level,
// resumable.hpp RES-7: once from all of them where it takes a population,
// else once from each with the budget left split among them) on the problem
// with the box the sweep ended with (an integer coordinate's values with
// it). Their spending goes to the run's ledger as a derived run's (STOP-6).
//
// SETTINGS: the knobs read, sweep_boxes (the box at the start and after
// every step: a JSON list of [lo, hi] pairs of lists), sweep_expansions (one
// [step, coordinate, "lo" | "hi", from, to, points of its layer] a moved
// face), sweep_fixed (the coordinates whose faces cannot move though the hard
// bounds are wider: SWEEP-3), sweep_points (the plan's points asked),
// sweep_steps; with inner sweep_carried (the points carried) and
// sweep_inner_runs.
//
// Knobs: plan (sobol), n_points (64), grid_n (5), hard_lo, hard_hi (a JSON
// list of n numbers, or one number for every coordinate; default the box:
// no face moves), top_k (5), edge_share (0.4; SWEEP-7), edge_tol (a share of the
// starting width; default the plan's spacing: min(0.25, n_points^(-1/n)),
// the grid's 0.5 / (grid_n - 1); SWEEP-7), expand_step (0.5: a share of the coordinate's
// STARTING width w0_j, not its units; an integer coordinate rounds it up to
// one value at least, SWEEP-1), sweep_budget (0), inner, carry_points (1),
// stall, max_steps, budget_units.
//
// READINGS:
//   SWEEP-1 (only when asked: the sweep strategy). The step of a face is a
//     share of the coordinate's starting width, for real and integer
//     coordinates alike, not a step in the coordinate's units: one knob for
//     coordinates of any scale (an integer count beside a length of
//     hundredths), a growth that is linear, a layer of the same thickness
//     every step, and the steps to a minimum at distance D past a face
//     ceil(D / (expand_step w0)). An integer coordinate moves by one value at least, else its
//     face would stay and the rule fire again without a new point.
//   SWEEP-2 (only when asked: the sweep strategy). With the hard bounds equal
//     to the box (their default) or the edge rule never firing, the run is
//     the plain plan: the same points in the same order, its answer and
//     counts bit for bit; the settings then show no expansion.
//   SWEEP-3 (only when asked: the sweep strategy). Faces that cannot move: a
//     listed coordinate whose values are not consecutive whole numbers (no
//     value lies outside its list), a periodic coordinate (its period is the
//     box), a real coordinate of starting width 0 (a share of it is 0). They
//     keep the box whatever the hard bounds say, and sweep_fixed names them.
//   SWEEP-4 (only when asked: the sweep strategy). The plan never evaluates a
//     point outside the hard bounds: a face stops at its hard bound, a grid's
//     last level is the bound itself, and every design point of a layer lies
//     between the old face and the new one.
//   SWEEP-5 (only when asked: the sweep strategy). The run's start (x0, seed
//     points) is not a point of the plan; with inner it starts the inner
//     solver only where the plan gave no point.
//   SWEEP-6 (only when asked: the sweep strategy). With inner the answer is
//     the better (the less violating, then the lower f; the plan's on a tie)
//     of the plan's best point and the inner runs' best (param.hpp
//     carry_level); improvements are the plan's records only.
//   SWEEP-7 (only when asked: the sweep strategy). The face's band is a
//     share of the starting width, and by default the plan's own spacing: a
//     space-filling design of N points in n coordinates leaves about N^(-1/n)
//     of the width between neighbours, so a narrower band would often hold
//     none of the best points though the minimum lies past the face (32
//     Sobol points in 2 coordinates, the minimum outside a corner: the best
//     point 0.11 of the width from the face). Capped at 0.25, so that in
//     many coordinates the bands do not cover the box; the grid's points lie
//     on its levels, half a spacing holds the outer level only. edge_share
//     0.4, two of the five best: a minimum past a corner of the box splits
//     the best points between two faces, and with 0.5 (three of five)
//     neither face fired there (32 Owen-scrambled Sobol points in 2
//     coordinates); a minimum well inside the box puts none of them in a
//     band. One nearer a face than about the band and the best points'
//     spread together may move that face once (the next step finds the best
//     points inside and stops): 32 Sobol points in 2 coordinates, a minimum
//     0.35 of the width from a face, one seed in ten; with 64 none.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../designs.hpp"
#include "../json.hpp"
#include "param.hpp"
#include "strategy.hpp"

namespace mootation::scalar {

namespace sweep {

// a coordinate of the sweep's box
struct Axis {
    bool listed = false;              // its values are the points (Problem::discrete)
    bool whole = false;               // listed, consecutive whole numbers: an integer coordinate
    bool movable = false;             // its faces may move (SWEEP-3)
    Vec values;                       // listed: the box's values; grid, real: its levels; ascending
    double lo = 0.0, hi = 0.0;        // the current box
    double lo0 = 0.0, hi0 = 0.0, w0 = 0.0;   // the starting box and its width (an integer's in values)
    double hard_lo = 0.0, hard_hi = 0.0;
    double h = 0.0;                   // grid, real: the starting spacing of the levels
    long long i_lo = 0, i_hi = 0;     // grid, real: the levels' indices at the faces (level i = lo0 + i h)
    long long last = 0;               // grid, real: the index of hi0 (grid_n - 1)
};

// what a layer lays on a coordinate: its values (listed, or a grid's levels)
// or the reals between a (the new face, in) and b (the old one, out)
struct Range {
    bool listed = false;
    Vec values;
    double a = 0.0, b = 0.0;
};

inline std::string g17(double v)
{
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.17g", v);
    return buf;
}

inline std::string list_text(const Vec& v)
{
    std::string s = "[";
    for (std::size_t j = 0; j < v.size(); ++j) s += (j ? "," : "") + g17(v[j]);
    return s + "]";
}

// a knob of n bounds: a JSON list of n numbers, or one number for all
inline Vec bounds_of(const std::string& name, const Knob& k, std::size_t n)
{
    if (!k.is_word) return Vec(n, k.number);
    const json::Value v = json::parse(k.word);
    if (v.kind == json::Value::Kind::Int || v.kind == json::Value::Kind::Float)
        return Vec(n, v.kind == json::Value::Kind::Int ? static_cast<double>(v.i) : v.f);
    if (v.kind != json::Value::Kind::Array || v.items.size() != n)
        throw std::invalid_argument("sweep: " + name + " must be a number or a JSON list of " + std::to_string(n) +
                                    " numbers");
    Vec out;
    for (const auto& it : v.items) {
        if (it.kind != json::Value::Kind::Int && it.kind != json::Value::Kind::Float)
            throw std::invalid_argument("sweep: " + name + " must be a JSON list of numbers");
        out.push_back(it.kind == json::Value::Kind::Int ? static_cast<double>(it.i) : it.f);
    }
    return out;
}

// grid, real: level i of the starting spacing (the starting faces exactly)
inline double level(const Axis& A, long long i)
{
    if (i == 0) return A.lo0;
    if (i == A.last) return A.hi0;
    return A.lo0 + static_cast<double>(i) * A.h;
}

// the point of a design's row u over the ranges
inline Vec design_point(const std::vector<Range>& R, const double* u)
{
    Vec x(R.size());
    for (std::size_t j = 0; j < R.size(); ++j) {
        if (R[j].listed) {
            const std::size_t K = R[j].values.size();
            const std::size_t at = (std::min)(K - 1, static_cast<std::size_t>(u[j] * static_cast<double>(K)));
            x[j] = R[j].values[at];
        } else {
            x[j] = R[j].a + u[j] * (R[j].b - R[j].a);
        }
    }
    return x;
}

// every point of the full grid of the lists, coordinate 0 the slowest
template <class Fn>
inline void walk_grid(const std::vector<Vec>& L, Fn&& fn)
{
    for (const Vec& l : L)
        if (l.empty()) return;
    std::vector<std::size_t> at(L.size(), 0);
    Vec x(L.size());
    for (;;) {
        for (std::size_t j = 0; j < L.size(); ++j) x[j] = L[j][at[j]];
        fn(x);
        std::size_t j = L.size();          // the next point: the last coordinate the fastest
        for (;;) {
            if (j == 0) return;
            --j;
            if (++at[j] < L[j].size()) break;
            at[j] = 0;
        }
    }
}

// THE EDGE RULE, a coordinate's side of it
struct Edge {
    double lo = 0.0, hi = 0.0;                 // the current faces
    double tol = 0.0;                          // the band: x on the lower face where x - lo <= tol, the upper hi - x <= tol
    bool open_lo = false, open_hi = false;     // the face may move (the coordinate movable, the face not at its hard bound)
};

// THE EDGE RULE (the sweep's, and campaign/components.hpp's grid of keys, COMP-15): the faces that fire for the best
// points, in the order coordinate 0's lower, its upper, coordinate 1's lower, ...: a face fires where at least
// edge_share of the points are on it and it is open. No points: none.
inline std::vector<std::pair<std::size_t, bool>> edge_faces(const std::vector<Vec>& best, const std::vector<Edge>& E,
                                                            double edge_share)
{
    std::vector<std::pair<std::size_t, bool>> faces;
    if (best.empty()) return faces;
    for (std::size_t j = 0; j < E.size(); ++j) {
        const Edge& e = E[j];
        if (!e.open_lo && !e.open_hi) continue;
        long long on_lo = 0, on_hi = 0;
        for (const Vec& x : best) {
            on_lo += x[j] - e.lo <= e.tol ? 1 : 0;
            on_hi += e.hi - x[j] <= e.tol ? 1 : 0;
        }
        const double share_lo = static_cast<double>(on_lo) / static_cast<double>(best.size());
        const double share_hi = static_cast<double>(on_hi) / static_cast<double>(best.size());
        if (share_lo >= edge_share && e.open_lo) faces.push_back({j, false});
        if (share_hi >= edge_share && e.open_hi) faces.push_back({j, true});
    }
    return faces;
}

struct PlanCap {};       // the plan has spent sweep_budget

}   // namespace sweep

inline Result run_sweep(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size, long long budget,
                        unsigned seed, const Knobs& knobs, Observer observer, int record_every, const Start& start)
{
    using namespace strategy;
    using sweep::Axis;
    using sweep::Range;
    StrategyKnobs k("sweep", knobs, {"plan", "n_points", "grid_n", "hard_lo", "hard_hi", "top_k", "edge_share",
                                     "edge_tol", "expand_step", "sweep_budget", "inner", "carry_points"});
    const std::string plan = k.word("plan", "sobol", {"sobol", "lhs", "uniform", "halton_rand", "sobol_owen", "grid"});
    const bool grid = plan == "grid";
    const long long n_points = grid ? 0 : k.whole("n_points", 64, 1);
    const long long grid_n = grid ? k.whole("grid_n", 5, 2) : 0;
    const long long top_k = k.whole("top_k", 5, 1);
    const double edge_share = k.number("edge_share", 0.4);
    const double spacing = grid ? 0.5 / static_cast<double>(grid_n - 1)                // SWEEP-7
                                : (std::min)(0.25, dm::pow(static_cast<double>(n_points),
                                                           -1.0 / static_cast<double>(P.lo.size())));
    const double edge_tol = k.number("edge_tol", spacing);
    const double step = k.number("expand_step", 0.5);
    const long long cap = k.whole("sweep_budget", 0, 0);
    const bool has_inner = k.find("inner") != nullptr;
    Inner inner;
    long long carry_n = 0;
    if (has_inner) {
        inner = k.inner("inner");
        carry_n = k.whole("carry_points", 1, 1);
    }
    if (P.m != 1) throw std::invalid_argument("sweep is a single-objective strategy (one objective)");
    if (!(edge_share > 0.0 && edge_share <= 1.0)) throw std::invalid_argument("sweep: edge_share in (0, 1]");
    if (!(edge_tol >= 0.0 && edge_tol < 0.5)) throw std::invalid_argument("sweep: edge_tol in [0, 0.5)");
    if (!(step > 0.0 && std::isfinite(step))) throw std::invalid_argument("sweep: expand_step must be a number > 0");
    const std::size_t n = P.lo.size();
    if (n == 0 || P.hi.size() != n) throw std::invalid_argument("sweep: the problem needs a box");

    // the coordinates (THE COORDINATES), the hard bounds (default the box)
    std::vector<Axis> ax(n);
    for (std::size_t j = 0; j < n; ++j) {
        if (!(std::isfinite(P.lo[j]) && std::isfinite(P.hi[j]) && P.lo[j] <= P.hi[j]))
            throw std::invalid_argument("sweep sweeps the box: every bound must be finite, lo <= hi");
        Axis& A = ax[j];
        A.listed = j < P.discrete.size() && !P.discrete[j].empty();
        if (A.listed) {
            A.values = P.discrete[j];
            A.whole = true;
            for (std::size_t q = 0; q < A.values.size(); ++q)
                A.whole = A.whole && A.values[q] == std::floor(A.values[q]) &&
                          (q == 0 || A.values[q] == A.values[q - 1] + 1.0);
            A.lo = A.values.front();
            A.hi = A.values.back();
        } else {
            A.lo = P.lo[j];
            A.hi = P.hi[j];
        }
        A.lo0 = A.lo;
        A.hi0 = A.hi;
        A.w0 = A.hi - A.lo;
        const bool periodic = j < P.periodic.size() && P.periodic[j];
        A.movable = !periodic && (A.whole || (!A.listed && A.w0 > 0.0));                        // SWEEP-3
        if (grid && !A.listed) {
            A.last = grid_n - 1;
            A.h = A.w0 / static_cast<double>(A.last);
            A.i_lo = 0;
            A.i_hi = A.last;
            for (long long i = 0; i <= A.last; ++i) A.values.push_back(sweep::level(A, i));
        }
    }
    const Knob* kl = k.find("hard_lo");
    const Knob* kh = k.find("hard_hi");
    const Vec hard_lo = kl ? sweep::bounds_of("hard_lo", *kl, n) : Vec();
    const Vec hard_hi = kh ? sweep::bounds_of("hard_hi", *kh, n) : Vec();
    if (kl) k.settings.push_back({"hard_lo", sweep::list_text(hard_lo)});
    if (kh) k.settings.push_back({"hard_hi", sweep::list_text(hard_hi)});
    std::string fixed = "[";
    bool any_fixed = false;
    for (std::size_t j = 0; j < n; ++j) {
        Axis& A = ax[j];
        const double hl = kl ? hard_lo[j] : P.lo[j], hh = kh ? hard_hi[j] : P.hi[j];
        if (!(std::isfinite(hl) && std::isfinite(hh) && hl <= P.lo[j] && hh >= P.hi[j]))
            throw std::invalid_argument("sweep: the hard bounds must be finite and hold the box (hard_lo <= lo, "
                                        "hard_hi >= hi)");
        A.hard_lo = A.whole ? (std::min)(A.lo, std::ceil(hl)) : A.listed ? A.lo : hl;
        A.hard_hi = A.whole ? (std::max)(A.hi, std::floor(hh)) : A.listed ? A.hi : hh;
        const bool wider = hl < P.lo[j] || hh > P.hi[j];
        if (!A.movable) {
            if (wider) {
                fixed += std::string(any_fixed ? "," : "") + std::to_string(j);
                any_fixed = true;
            }
            A.hard_lo = A.lo;
            A.hard_hi = A.hi;
        }
    }
    fixed += "]";

    // the problem the plan evaluates: the box the hard bounds
    Problem Ph = P;
    for (std::size_t j = 0; j < n; ++j) {
        Ph.lo[j] = (std::min)(P.lo[j], ax[j].hard_lo);
        Ph.hi[j] = (std::max)(P.hi[j], ax[j].hard_hi);
    }
    Evaluator ev(Ph, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    StopRule stop = stop_of(k, ev);                                                  // stop_rule.hpp: a step a move

    long long asked = 0, layers = 0, steps = 0;
    bool capped = false;
    auto evaluate = [&](const Vec& x) {
        if (cap > 0 && ev.counted() >= static_cast<double>(cap)) throw sweep::PlanCap{};
        ++asked;
        ev.index(x);
    };
    auto box_text = [&]() {
        Vec lo(n), hi(n);
        for (std::size_t j = 0; j < n; ++j) {
            lo[j] = ax[j].lo;
            hi[j] = ax[j].hi;
        }
        return "[" + sweep::list_text(lo) + "," + sweep::list_text(hi) + "]";
    };
    // the ranges of the box as it is (each coordinate whole)
    auto whole_box = [&]() {
        std::vector<Range> R(n);
        for (std::size_t j = 0; j < n; ++j) {
            R[j].listed = ax[j].listed || grid;
            if (R[j].listed) R[j].values = ax[j].values;
            R[j].a = ax[j].lo;
            R[j].b = ax[j].hi;
        }
        return R;
    };
    // a layer's points by the plan, evaluated in order
    auto lay = [&](const std::vector<Range>& R, long long count, unsigned layer_seed) {
        if (grid) {
            std::vector<Vec> L(n);
            for (std::size_t j = 0; j < n; ++j) L[j] = R[j].values;
            sweep::walk_grid(L, evaluate);
            return;
        }
        const std::vector<double> U =
            designs::design(plan, static_cast<std::size_t>(count), static_cast<int>(n), layer_seed);
        for (long long i = 0; i < count; ++i) evaluate(sweep::design_point(R, &U[static_cast<std::size_t>(i) * n]));
    };
    // a design layer's count: n_points times its measure over the starting box's
    auto count_of = [&](const std::vector<Range>& R) {
        double ratio = 1.0;
        for (std::size_t j = 0; j < n; ++j) {
            const Axis& A = ax[j];
            const double m0 = A.listed ? static_cast<double>(P.discrete[j].size()) : A.w0;
            if (!(m0 > 0.0)) continue;
            const double m = R[j].listed ? static_cast<double>(R[j].values.size()) : R[j].b - R[j].a;
            ratio *= (m < 0.0 ? -m : m) / m0;
        }
        return (std::max)(1LL, static_cast<long long>(std::ceil(static_cast<double>(n_points) * ratio)));
    };
    // the top_k best points of the archive (failed ones out)
    auto top = [&]() {
        std::vector<std::size_t> idx;
        for (std::size_t i = 0; i < ev.X.size(); ++i)
            if (!ev.failed[i]) idx.push_back(i);
        std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
            const double ca = ev.cv(a), cb = ev.cv(b);
            if (ca != cb) return ca < cb;
            return ev.F[a][0] < ev.F[b][0];
        });
        if (idx.size() > static_cast<std::size_t>(top_k)) idx.resize(static_cast<std::size_t>(top_k));
        return idx;
    };
    // moves face (j, upper) outward; the layer between the new face and the old
    auto move = [&](std::size_t j, bool upper) {
        Axis& A = ax[j];
        Range r;
        const double from = upper ? A.hi : A.lo;
        if (A.whole) {                                                              // SWEEP-1
            const double d = (std::max)(1.0, std::ceil(step * A.w0));
            const double to = upper ? (std::min)(A.hard_hi, A.hi + d) : (std::max)(A.hard_lo, A.lo - d);
            r.listed = true;
            Vec add;
            if (upper)
                for (double v = A.hi + 1.0; v <= to; v += 1.0) add.push_back(v);
            else
                for (double v = to; v < A.lo; v += 1.0) add.push_back(v);
            r.values = add;
            if (upper) {
                A.values.insert(A.values.end(), add.begin(), add.end());
                A.hi = to;
            } else {
                A.values.insert(A.values.begin(), add.begin(), add.end());
                A.lo = to;
            }
        } else if (grid) {
            const long long K = (std::max)(1LL, static_cast<long long>(std::ceil(step * static_cast<double>(A.last))));
            Vec add;
            for (long long t = 1; t <= K; ++t) {
                const long long i = upper ? A.i_hi + t : A.i_lo - t;
                const double v = sweep::level(A, i);
                if (upper ? v >= A.hard_hi : v <= A.hard_lo) {                     // SWEEP-4: the bound itself
                    add.push_back(upper ? A.hard_hi : A.hard_lo);
                    break;
                }
                add.push_back(v);
                if (upper) A.i_hi = i;
                else A.i_lo = i;
            }
            r.listed = true;
            if (upper) {
                r.values = add;
                A.values.insert(A.values.end(), add.begin(), add.end());
                A.hi = add.back();
            } else {
                r.values.assign(add.rbegin(), add.rend());
                A.values.insert(A.values.begin(), r.values.begin(), r.values.end());
                A.lo = add.back();
            }
        } else {
            const double to = upper ? (std::min)(A.hard_hi, A.hi + step * A.w0) : (std::max)(A.hard_lo, A.lo - step * A.w0);
            r.a = to;
            r.b = from;
            if (upper) A.hi = to;
            else A.lo = to;
        }
        return r;
    };

    std::string boxes = "[" + box_text();
    std::string expansions = "[";
    bool any_expansion = false;
    try {
        try {
            const std::vector<Range> R0 = whole_box();
            lay(R0, grid ? 0 : n_points, seed);                                     // the plan (SWEEP-2)
            stop.mark(ev);
            for (;;) {
                // THE EDGE RULE: the faces that fire, judged on the top before the step
                std::vector<Vec> best;
                for (std::size_t i : top()) best.push_back(ev.X[i]);
                std::vector<sweep::Edge> E(n);
                for (std::size_t j = 0; j < n; ++j) {
                    const Axis& A = ax[j];
                    E[j].lo = A.lo;
                    E[j].hi = A.hi;
                    E[j].tol = edge_tol * A.w0;
                    E[j].open_lo = A.movable && A.lo > A.hard_lo;
                    E[j].open_hi = A.movable && A.hi < A.hard_hi;
                }
                const std::vector<std::pair<std::size_t, bool>> faces = sweep::edge_faces(best, E, edge_share);
                if (faces.empty()) break;                                           // the sweep's own end
                if (!stop.go(ev)) break;
                ++steps;
                // the moves first (each layer over the box at its face's turn), then the layers' points
                std::vector<std::vector<Range>> todo;
                std::vector<long long> counts;
                for (const auto& fc : faces) {
                    const std::size_t j = fc.first;
                    const double from = fc.second ? ax[j].hi : ax[j].lo;
                    std::vector<Range> R = whole_box();
                    R[j] = move(j, fc.second);
                    const double to = fc.second ? ax[j].hi : ax[j].lo;
                    long long count = 0;
                    if (grid) {
                        count = 1;
                        for (std::size_t q = 0; q < n; ++q) count *= static_cast<long long>(R[q].values.size());
                    } else {
                        count = count_of(R);
                    }
                    expansions += std::string(any_expansion ? "," : "") + "[" + std::to_string(steps) + "," +
                                  std::to_string(j) + ",\"" + (fc.second ? "hi" : "lo") + "\"," + sweep::g17(from) +
                                  "," + sweep::g17(to) + "," + std::to_string(count) + "]";
                    any_expansion = true;
                    todo.push_back(R);
                    counts.push_back(count);
                }
                boxes += "," + box_text();
                for (std::size_t q = 0; q < todo.size(); ++q) {
                    ++layers;
                    lay(todo[q], counts[q], start_seed(seed, static_cast<std::uint64_t>(layers)));
                }
                stop.step(ev);
            }
        } catch (const sweep::PlanCap&) {
            capped = true;
        }
    } catch (const BudgetSpent&) {
    }
    boxes += "]";
    expansions += "]";
    if (!ev.recorded) ev.record();
    std::string stopped = stop.stopped_by(ev, capped ? "sweep_budget" : "rule");

    // THE POINTS AS STARTS: the plan's best points start the inner solver
    detail::CarryLevel cl;
    bool inner_ran = false;
    long long carried = 0;
    if (has_inner && !ev.target_reached && ev.left() > 0 && !past_deadline(P)) {
        Problem Q = P;
        for (std::size_t j = 0; j < n; ++j) {
            Q.lo[j] = (std::min)(P.lo[j], ax[j].lo);
            Q.hi[j] = (std::max)(P.hi[j], ax[j].hi);
            if (ax[j].whole) Q.discrete[j] = ax[j].values;
        }
        Q = derived(P, Q);
        Rows from;
        for (std::size_t i : top()) {
            if (static_cast<long long>(from.size()) >= carry_n) break;
            if (std::find(from.begin(), from.end(), ev.X[i]) == from.end()) from.push_back(ev.X[i]);
        }
        carried = static_cast<long long>(from.size());
        cl = detail::carry_level(inner, Q, eq_tol, has_tol0, eq_tol0, pop_size, ev.left(), seed, start, from,
                                 static_cast<std::size_t>(carry_n));
        inner_ran = true;
        bool target = false;
        for (const Result& part : cl.parts) {                                        // STOP-6: the run's ledger
            ev.spend_outside(part.evaluations, part.grad_calls);
            ev.hook_calls += part.hook_calls;
            ev.hook_spent += part.hook_spent;
            ev.hook_seconds += part.hook_seconds;
            ev.success_calls += part.success_calls;
            ev.success_spent += part.success_spent;
            ev.delta_calls += part.delta_calls;
            ev.delta_spent += part.delta_spent;
            target = target || part.stopped_by == "target" || reached(Q, part);
        }
        stopped = target ? "target" : ev.left() <= 0 ? "budget" : past_deadline(P) ? "deadline" : "rule";
    }

    ev.over = true;
    const std::vector<std::size_t> idx = ev.answer();
    ev.record(true);
    Result r = detail::result_of(ev, idx);
    if (inner_ran && !cl.best.objectives.empty()) {                                  // SWEEP-6
        const Result& b = cl.best;
        const double cb = b.cv.empty() ? 0.0 : b.cv[0];
        bool better = r.objectives.empty();
        if (!better) {
            const double ca = r.cv.empty() ? 0.0 : r.cv[0];
            better = cb != ca ? cb < ca : b.objectives[0][0] < r.objectives[0][0];
        }
        if (better) {
            r.objectives = {b.objectives[0]};
            r.variables = {b.variables[0]};
            r.cv = {cb};
        }
    }
    r.ignored = k.ignored();
    r.settings = k.settings;
    r.settings.push_back({"sweep_boxes", boxes});
    r.settings.push_back({"sweep_expansions", expansions});
    if (any_fixed) r.settings.push_back({"sweep_fixed", fixed});
    r.settings.push_back({"sweep_points", std::to_string(asked)});
    r.settings.push_back({"sweep_steps", std::to_string(steps)});
    if (has_inner) {
        r.settings.push_back({"sweep_carried", std::to_string(carried)});
        r.settings.push_back({"sweep_inner_runs", std::to_string(cl.parts.size())});
    }
    r.stopped_by = stopped;
    r.settings.push_back({"method_version", std::to_string(METHOD_VERSION)});
    return r;
}

}   // namespace mootation::scalar
