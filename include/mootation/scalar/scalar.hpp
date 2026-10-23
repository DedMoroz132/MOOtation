#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Scalarization with a local solver (task 9 of 2026-10-18, part A1; in C++
// since task 17, stage 6): the front as a sequence of single-objective
// subproblems, each solved by a local method — and, beside them, the methods
// of task 9 on the same evaluator, answer and observer: mosqp (mosqp.hpp) and
// boostdms (boostdms.hpp). Not evolutionary, and not a core of
// algorithms.def: run() takes the problem as a function and every evaluation
// goes through it (evaluator.hpp). The local solvers are the library's own
// (slsqp.hpp, nelder_mead.hpp, cmaes.hpp).
//
// THE ALGORITHM `scalar`, and its presets under their own names:
//
//   eps_sqp  tch_sqp  nbi_sqp  nnc_sqp  eps_nm  eps_cmaes
//
//   scalarization  eps | tch | nbi | nnc | ps            (subproblems.hpp)
//   local          sqp | nm | cmaes                      (solvers.hpp)
//   sub_plan       grid | laumanns | gap
//   tch_weights    inverse | lattice: tch's weights from the lattice point d,
//                  1/d (the rays F̄ ∝ d, SCALAR-3) or d itself (the plain grid)
//   n_sub          subproblems after the anchors; 0: from the budget (SCALAR-1)
//   sub_budget     evaluations a subproblem may take; 0: what is left over the
//                  subproblems left, so that what one does not use the next get
//   fd_step        sqp's forward-difference step in the normalized box (1.49e-8)
//   local_tol      the solvers' tolerance (1e-14)
//   cma_sigma0     cmaes's initial step in the normalized box (0.3; the
//                  tutorial's 0.3 (b - a), footnote to Figure 6)
//   eps_objective  the objective eps minimizes, 1-based (1)
//
// THE PLAN.
//   1. The initial plan: pop_size points drawn uniformly from the seed
//      (rnd::uniform01 of std::mt19937(seed)), or the rows of a design given
//      (init), or the seed population — the points the first subproblems
//      start from.
//   2. The anchors: for every objective k, the lexicographic minimum of
//      (f_k, f_{k+1}, ..., f_{k-1}) — min f_k, then min f_{k+1} with f_k no
//      worse, and so on (SCALAR-9). They give the normalization (z*, z^N;
//      subproblems.hpp Frame).
//   3. The subproblems of the plan, each started from the archive's best
//      point for it (feasible first, then its value): with sub_plan =
//        grid      the lattice, in lattice order (neighbours one after another:
//                  NBI's "continuation", Das & Dennis §5.3);
//        laumanns  eps only: the adaptive scheme of Laumanns, Thiele & Zitzler
//                  (Dagstuhl 04461, 2005, Fig. 1-2; EJOR 169(3), 2006): the
//                  constraint levels come from the points already found, every
//                  cell is searched once, and an empty or dominated cell is
//                  never searched again; a cell yields a point by two runs, min
//                  f_k, then min the others with f_k no worse. On a continuous
//                  problem the strict bound is f_j <= e_j - δ with δ = 1/(L + 1)
//                  of the normalized range, L the grid's levels per objective
//                  (the increments §6 of the paper asks for there; SCALAR-10);
//        gap       OUR rule, not Laumanns's: of a lattice four times finer, the
//                  subproblem whose parameter lies farthest from those of the
//                  points found so far (subproblems.hpp coords), one after
//                  another.
//   4. The answer: the archive's feasible non-dominated points, reduced to
//      pop_size by DSS (evaluator.hpp). The run ends when the plan is done or
//      the budget spent, whichever comes first: a plan that ends early leaves
//      evaluations unspent, as DMS does.
//
// CONSTRAINTS. sqp takes the problem's inequalities (with a margin of 1e-10:
// SCALAR-8) and equalities directly; nm and cmaes through the penalty in which
// a feasible point is always better (solvers.hpp, SCALAR-2). Only the
// feasibility rule: constraint_mode = eps_level is refused (the caller's
// check). Real variables only.
//
// READINGS (declared):
//   SCALAR-1. n_sub from the budget: what is left after the plan's start and
//     the anchors, over 10 (n + 1) evaluations a subproblem for sqp — about
//     ten iterations from a neighbour's solution — and 100 n for nm and
//     cmaes; at most pop_size, at least 1. A subproblem may take what is left
//     over the subproblems left; an anchor's stage at least a quarter of the
//     budget left after the initial plan over the m² stages — the frame rests
//     on them.
//   SCALAR-2. nm, cmaes: an infeasible point is worth the worst feasible
//     value known for its subproblem plus its violation (Deb's rule as a
//     penalty).
//   SCALAR-3. tch: w_i = 1 / max(d_i, 1e-6) for the lattice's direction d, so
//     that the solutions lie on the rays F̄ ∝ d the lattice spreads evenly;
//     the reference point is z*.
//   SCALAR-4. nnc: the normal constraints (Ā_m − Ā_i)ᵀ(F̄ − X̄) <= 0 and min
//     F̄_m, Problem P3 of the 2003 paper (Eqs. 36-41). Its points X̄ on the
//     utopia plane are spaced along each N̄_k in proportion to ‖N̄_k‖ (Eq. 32);
//     here they are the sub_plan's lattice of α. (The 2026 summary's Eq.
//     (80), (F̄ − X̄)ᵀ(A_i − X̄) <= 0 with min F̄_k, is not the 2003 constraint.)
//   SCALAR-5. ps: ā on the anchors' simplex, r its quasi-normal reversed
//     (NBI's geometry); the 1984 paper leaves (p, q) to the user.
//     Eichfelder's adaptive choice of ā could not be checked against her
//     texts and is not implemented.
//   SCALAR-6. sqp stops a subproblem at a failed evaluation of an iterate, and
//     at a finite-difference step that fails both ways.
//   SCALAR-7. cmaes: a point outside the box is evaluated at its clip, plus
//     the squared distance to it times (1 + |value|); one run per subproblem,
//     until σ·max D < local_tol or the allowance is spent.
//   SCALAR-8. sqp asks g(x) <= -1e-10 of the problem's inequalities, so that a
//     solution on the boundary is feasible by the vault's rule (g <= 0).
//   SCALAR-9. The anchors are lexicographic minima in cyclic order: ours. Das
//     & Dennis and Messac et al. take the individual minima as they come,
//     which on a front with a degenerate edge (DTLZ2: (0, 0, 1) minimizes f1
//     and f2 alike) makes two anchors one and the simplex flat; a stage keeps
//     the earlier objectives within 1e-9 (relative) of their minima. A stage
//     runs from the archive's best point for it and, when it differs, from
//     the point of least value too: the best feasible point can sit where the
//     gradient vanishes (DTLZ2's corner x1 = 1 for f3), and sqp never leaves
//     it.
//   SCALAR-10. laumanns: the grid's first bound is +∞ (no constraint), the
//     mirror of the paper's lower bound 0 for maximization.
//   SCALAR-11. A subproblem whose solver ends where it started runs again,
//     with what is left of its allowance, from the initial plan's best point
//     for it: the archive's best is often an anchor or a neighbour's solution
//     at a corner of the parameterization where the gradients vanish (DTLZ's
//     x_i in {0, 1}), and a gradient method stays there.
//   SCALAR-12. The anchors' stages see the objectives scaled by the initial
//     plan's range (its minimum to 0, its maximum to 1): SQP stops early on a
//     raw objective of 1e7 (bbob-biobj f2's ellipsoid).
//
// METHOD_VERSION: the version of these methods' trajectories, in every
// result's settings (a campaign's meta): 1 — SciPy's SLSQP and Nelder-Mead and
// NumPy's generator (task 9), 2 — the library's own solvers and rnd:: (task
// 17, stage 6).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../detmath.hpp"
#include "../fp_contract.hpp"
#include "../random.hpp"
#include "boostdms.hpp"
#include "mosqp.hpp"
#include "solvers.hpp"

namespace mootation::scalar {

inline constexpr int METHOD_VERSION = 2;

// A knob as the caller gives it: a word or a number.
struct Knob {
    bool is_word = false;
    std::string word;
    double number = 0.0;
};
using Knobs = std::vector<std::pair<std::string, Knob>>;   // in the caller's order

struct Result {
    Rows objectives, variables;
    Vec cv;
    std::vector<std::string> ignored;
    long long evaluations = 0;
    long long n_failed = 0;          // failed evaluations in the archive
    std::vector<std::pair<std::string, std::string>> settings;
    Rows solutions;                  // scalar: every subproblem's solution, anchors first
};

// How the first points come: seed points (planted), rows of a design
// (evaluated in order), or pop_size uniform points from the seed.
struct Start {
    std::vector<Point> seed_points;
    Rows design;
    bool has_seed_points = false, has_design = false;
};

// The names: the presets of `scalar`, then the methods beside it.
inline const std::vector<std::string>& preset_names()
{
    static const std::vector<std::string> v{"scalar", "eps_sqp", "tch_sqp", "nbi_sqp",
                                            "nnc_sqp", "eps_nm", "eps_cmaes"};
    return v;
}
inline const std::vector<std::string>& names()
{
    static const std::vector<std::string> v{"scalar", "eps_sqp", "tch_sqp", "nbi_sqp", "nnc_sqp",
                                            "eps_nm", "eps_cmaes", "boostdms", "mosqp"};
    return v;
}

namespace detail {

// Python's f"{v:g}"
inline std::string g_format(double v)
{
    char buf[64];
    std::snprintf(buf, sizeof buf, "%g", v);
    return buf;
}

// Python's str() of a knob's value, for messages.
inline std::string shown(const Knob& k)
{
    if (k.is_word) return k.word;
    if (k.number == std::floor(k.number) && std::fabs(k.number) < 1e15) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.0f", k.number);
        return buf;
    }
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.17g", k.number);
    return buf;
}

inline double number_of(const std::string& name, const Knob& k)
{
    if (!k.is_word) return k.number;
    char* end = nullptr;
    const double v = std::strtod(k.word.c_str(), &end);
    if (end == k.word.c_str() || *end != '\0')
        throw std::invalid_argument("could not convert string to float: '" + k.word + "' (" + name + ")");
    return v;
}

inline bool listed(const std::vector<std::string>& v, const std::string& s)
{
    return std::find(v.begin(), v.end(), s) != v.end();
}

inline std::string joined(const std::vector<std::string>& v)
{
    std::string out;
    for (std::size_t i = 0; i < v.size(); ++i) out += (i ? ", " : "") + v[i];
    return out;
}

// The Das-Dennis lattice of H = h in its recursive order (the first
// component from h down), every component v / h.
inline void lattice(std::size_t m, int h, std::vector<Vec>& out)
{
    std::vector<int> prefix;
    std::function<void(int, std::size_t)> rec = [&](int left, std::size_t k) {
        if (k == 1) {
            Vec row;
            for (int v : prefix) row.push_back(static_cast<double>(v) / static_cast<double>(h));
            row.push_back(static_cast<double>(left) / static_cast<double>(h));
            out.push_back(row);
            return;
        }
        for (int v = left; v >= 0; --v) {
            prefix.push_back(v);
            rec(left - v, k - 1);
            prefix.pop_back();
        }
    };
    rec(h, m);
}

inline long long comb(long long n, long long k)
{
    long long r = 1;
    for (long long i = 1; i <= k; ++i) r = r * (n - k + i) / i;
    return r;
}

// Das-Dennis lattice points without the vertices (the anchors), at least
// `count` of them (the smallest H that gives as many).
inline std::vector<Vec> simplex(std::size_t m, long long count)
{
    std::vector<Vec> out;
    if (m < 2) return out;
    int h = 2;
    while (comb(h + static_cast<long long>(m) - 1, static_cast<long long>(m) - 1) -
               static_cast<long long>(m) < count)
        ++h;
    std::vector<Vec> L;
    lattice(m, h, L);
    for (const Vec& r : L)
        if (*std::max_element(r.begin(), r.end()) < 1.0) out.push_back(r);
    return out;
}

// eps: per constrained objective L levels j/(L + 1), L^(m-1) >= count.
inline std::vector<Vec> levels(std::size_t m, long long count, long long& L)
{
    std::vector<Vec> out;
    L = 1;
    if (m < 2) return out;
    L = std::max<long long>(1, static_cast<long long>(std::ceil(
            dm::pow(static_cast<double>(count), 1.0 / static_cast<double>(m - 1)) - 1e-9)));
    Vec axis;
    for (long long j = 1; j <= L; ++j) axis.push_back(static_cast<double>(j) / static_cast<double>(L + 1));
    // the grid in "ij" order: the first coordinate the slowest
    std::vector<std::size_t> digit(m - 1, 0);
    for (;;) {
        Vec row;
        for (std::size_t d : digit) row.push_back(axis[d]);
        out.push_back(row);
        std::size_t r = m - 1;
        while (r > 0) {
            if (++digit[r - 1] < axis.size()) break;
            digit[r - 1] = 0;
            --r;
        }
        if (r == 0) break;
    }
    return out;
}

}   // namespace detail

// ── scalar ──────────────────────────────────────────────────────────────────
struct ScalarSettings {
    std::string scalarization = "eps", local = "sqp", sub_plan = "grid", tch_weights = "inverse";
    double n_sub = 0, sub_budget = 0, fd_step = 1.4901161193847656e-08, local_tol = 1e-14,
           cma_sigma0 = 0.3, eps_objective = 1;
};

// The preset's settings with the knobs given over them, checked; the knobs
// it does not know go to `ignored`.
inline ScalarSettings scalar_settings(const std::string& name, const Knobs& knobs,
                                      std::vector<std::string>& ignored)
{
    static const std::map<std::string, std::vector<std::string>> words{
        {"scalarization", {"eps", "tch", "nbi", "nnc", "ps"}},
        {"tch_weights", {"inverse", "lattice"}},
        {"local", {"sqp", "nm", "cmaes"}},
        {"sub_plan", {"grid", "laumanns", "gap"}}};
    ScalarSettings s;
    if (name == "eps_sqp") { s.scalarization = "eps"; s.local = "sqp"; }
    else if (name == "tch_sqp") { s.scalarization = "tch"; s.local = "sqp"; }
    else if (name == "nbi_sqp") { s.scalarization = "nbi"; s.local = "sqp"; }
    else if (name == "nnc_sqp") { s.scalarization = "nnc"; s.local = "sqp"; }
    else if (name == "eps_nm") { s.scalarization = "eps"; s.local = "nm"; }
    else if (name == "eps_cmaes") { s.scalarization = "eps"; s.local = "cmaes"; }
    std::map<std::string, double*> numbers{{"n_sub", &s.n_sub}, {"sub_budget", &s.sub_budget},
                                           {"fd_step", &s.fd_step}, {"local_tol", &s.local_tol},
                                           {"cma_sigma0", &s.cma_sigma0},
                                           {"eps_objective", &s.eps_objective}};
    std::map<std::string, std::string*> word_of{{"scalarization", &s.scalarization},
                                                {"tch_weights", &s.tch_weights},
                                                {"local", &s.local}, {"sub_plan", &s.sub_plan}};
    for (const auto& [k, v] : knobs) {
        auto w = words.find(k);
        if (w != words.end()) {
            const std::string val = detail::shown(v);
            if (!v.is_word || !detail::listed(w->second, val))
                throw std::invalid_argument(k + " = '" + val + "': " + detail::joined(w->second));
            *word_of[k] = val;
        } else if (numbers.count(k)) {
            *numbers[k] = detail::number_of(k, v);
        } else {
            ignored.push_back(k);
        }
    }
    std::sort(ignored.begin(), ignored.end());
    if (s.n_sub < 0 || s.n_sub != std::floor(s.n_sub))
        throw std::invalid_argument("n_sub must be a whole number >= 0 (0: from the budget)");
    if (s.sub_budget < 0 || s.sub_budget != std::floor(s.sub_budget))
        throw std::invalid_argument("sub_budget must be a whole number >= 0 (0: a share of what is left)");
    for (const auto& [k, p] : std::vector<std::pair<std::string, double*>>{
             {"fd_step", &s.fd_step}, {"local_tol", &s.local_tol}, {"cma_sigma0", &s.cma_sigma0}})
        if (!(*p > 0.0)) throw std::invalid_argument(k + " must be > 0");
    if (s.sub_plan == "laumanns" && s.scalarization != "eps")
        throw std::invalid_argument("sub_plan = 'laumanns' is a scheme of the eps-constraint method: "
                                    "scalarization = 'eps'");
    return s;
}

inline std::vector<std::pair<std::string, std::string>> settings_list(const ScalarSettings& s)
{
    using detail::g_format;
    return {{"scalarization", s.scalarization}, {"local", s.local}, {"sub_plan", s.sub_plan},
            {"tch_weights", s.tch_weights}, {"n_sub", g_format(s.n_sub)},
            {"sub_budget", g_format(s.sub_budget)}, {"fd_step", g_format(s.fd_step)},
            {"local_tol", g_format(s.local_tol)}, {"cma_sigma0", g_format(s.cma_sigma0)},
            {"eps_objective", g_format(s.eps_objective)},
            {"method_version", std::to_string(METHOD_VERSION)}};
}

namespace detail {

// The first points of a run (THE PLAN, 1); the archive indices.
inline std::vector<std::size_t> first_points(Evaluator& ev, const Start& st, int pop_size,
                                             std::mt19937& rng, bool uniform_default)
{
    std::vector<std::size_t> out;
    const std::size_t n = ev.lo.size();
    if (st.has_seed_points) {
        for (const Point& p : st.seed_points) out.push_back(ev.plant(p.x, p.f, p.g));
    } else if (st.has_design) {
        for (const Vec& x : st.design) out.push_back(ev.index(x));
    } else if (uniform_default) {
        for (int k = 0; k < pop_size; ++k) {
            Vec x(n);
            for (std::size_t j = 0; j < n; ++j)
                x[j] = ev.lo[j] + rnd::uniform01(rng) * (ev.hi[j] - ev.lo[j]);
            out.push_back(ev.index(x));
        }
    }
    return out;
}

inline Result result_of(Evaluator& ev, const std::vector<std::size_t>& idx)
{
    Result r;
    for (std::size_t i : idx) {
        r.objectives.push_back(ev.F[i]);
        r.variables.push_back(ev.X[i]);
        r.cv.push_back(ev.cv(i));
    }
    r.evaluations = ev.spent;
    for (char f : ev.failed) r.n_failed += f ? 1 : 0;
    return r;
}

}   // namespace detail

inline Result run_scalar(const std::string& name, const Problem& P, double eq_tol, bool has_tol0,
                         double eq_tol0, int pop_size, long long budget, unsigned seed,
                         const Knobs& knobs, Observer observer, int record_every, const Start& start)
{
    std::vector<std::string> ignored;
    ScalarSettings s = scalar_settings(name, knobs, ignored);
    const std::size_t m = static_cast<std::size_t>(P.m), n = P.lo.size();
    const long long k_eps1 = static_cast<long long>(s.eps_objective);
    if (!(1 <= k_eps1 && k_eps1 <= static_cast<long long>(m)))
        throw std::invalid_argument("eps_objective = " + detail::g_format(s.eps_objective) + ": 1 to " +
                                    std::to_string(m));
    const std::size_t k_eps = static_cast<std::size_t>(k_eps1 - 1);
    std::mt19937 rng(seed);
    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);

    // The archive's best point for sub: feasible first, then its value; or,
    // by_value, the least value whatever it violates. among: these indices only.
    auto best_start = [&](const Sub& sub, const Frame& frame, bool by_value,
                          const std::vector<std::size_t>* among) -> long long {
        std::vector<std::size_t> ok;
        const std::size_t N = among ? among->size() : ev.X.size();
        for (std::size_t t = 0; t < N; ++t) {
            const std::size_t i = among ? (*among)[t] : t;
            if (!ev.failed[i]) ok.push_back(i);
        }
        if (ok.empty()) return -1;
        Vec value(ok.size()), viol(ok.size());
        for (std::size_t t = 0; t < ok.size(); ++t) sub.merit(frame.bar(ev.F[ok[t]]), value[t], viol[t]);
        auto argmin = [](const Vec& v, const std::vector<char>* mask) {
            long long j = -1;
            for (std::size_t t = 0; t < v.size(); ++t)
                if ((!mask || (*mask)[t]) && (j < 0 || v[t] < v[static_cast<std::size_t>(j)]))
                    j = static_cast<long long>(t);
            return j;
        };
        if (by_value) return static_cast<long long>(ok[static_cast<std::size_t>(argmin(value, nullptr))]);
        std::vector<char> feas(ok.size(), 0);
        bool any = false;
        for (std::size_t t = 0; t < ok.size(); ++t) {
            viol[t] = (viol[t] > SLACK ? viol[t] : 0.0) + ev.cv(ok[t]);
            feas[t] = viol[t] <= 0.0;
            any = any || feas[t];
        }
        const long long j = any ? argmin(value, &feas) : argmin(viol, nullptr);
        return static_cast<long long>(ok[static_cast<std::size_t>(j)]);
    };

    long long plan_left = 0;                     // subproblems still to come, this one included
    std::vector<std::size_t> initial;
    std::vector<std::size_t> solutions;

    // One subproblem: its start, its allowance, its solver; its best point.
    // two_starts (the anchors): from the best point and from the one of least
    // value too, when they differ (SCALAR-9).
    auto solve = [&](const Sub& sub, const Frame& frame, bool two_starts, long long floor) -> long long {
        std::vector<std::size_t> starts;
        const long long s0 = best_start(sub, frame, false, nullptr);
        if (s0 < 0) return -1;
        starts.push_back(static_cast<std::size_t>(s0));
        if (two_starts) {
            const long long other = best_start(sub, frame, true, nullptr);
            if (other != s0) starts.push_back(static_cast<std::size_t>(other));
        }
        const long long left = ev.budget - ev.spent;
        const long long share = left / std::max<long long>(1, plan_left);
        const long long allowance = s.sub_budget > 0
            ? std::min(left, static_cast<long long>(s.sub_budget))
            : std::min(left, std::max(share, floor));
        plan_left = std::max<long long>(0, plan_left - 1);
        const long long end = ev.spent + std::max<long long>(1, allowance);
        Run run(sub, ev, frame);

        auto from = [&](std::size_t st, long long limit) {
            ev.limit = limit;
            run.touched.push_back(st);
            const Vec y0 = run.y_of(st);
            if (s.local == "sqp")
                sqp(run, y0, sub.t_star(frame.bar(ev.F[st])), s.fd_step, s.local_tol);
            else if (s.local == "nm")
                nm(run, y0, {st}, s.local_tol);
            else
                cmaes(run, y0, {st}, s.cma_sigma0, s.local_tol, rng);
        };
        try {
            for (std::size_t k = 0; k < starts.size(); ++k)
                // the second start gets what the first left of the allowance
                from(starts[k], k + 1 == starts.size() ? end
                                                        : ev.spent + std::max<long long>(1, (end - ev.spent) / 2));
            const long long b = best_of(run, run.touched);
            if (!moved(run, starts, static_cast<std::size_t>(b)) && ev.spent < end) {
                // not moved from where it started: again from the initial
                // plan's best point for this subproblem (SCALAR-11)
                const long long again = best_start(sub, frame, false, &initial);
                if (again >= 0 && std::find(starts.begin(), starts.end(),
                                            static_cast<std::size_t>(again)) == starts.end())
                    from(static_cast<std::size_t>(again), end);
            }
        } catch (...) {
            ev.limit = ev.budget;
            throw;
        }
        ev.limit = ev.budget;
        return best_of(run, run.touched);
    };

    // Did subproblem sub yield a feasible point at i?
    auto found = [&](const Sub& sub, const Frame& frame, long long i) {
        if (i < 0 || ev.failed[static_cast<std::size_t>(i)] || ev.cv(static_cast<std::size_t>(i)) > 0.0)
            return false;
        double v, c;
        sub.merit(frame.bar(ev.F[static_cast<std::size_t>(i)]), v, c);
        return c <= SLACK;
    };

    long long k_sub = static_cast<long long>(s.n_sub);
    try {
        // 1. the initial plan
        detail::first_points(ev, start, pop_size, rng, true);
        for (std::size_t i = 0; i < ev.X.size(); ++i) initial.push_back(i);
        if (!ev.recorded) ev.record();               // gen 0, unless the plan's last point gave it

        // 2. the anchors, in the scale of the initial plan (SCALAR-12)
        Frame identity(m);
        {
            std::vector<Vec> okF;
            for (std::size_t i : initial)
                if (!ev.failed[i]) okF.push_back(ev.F[i]);
            if (!okF.empty()) {
                Vec lo = okF[0], hi = okF[0];
                for (const Vec& f : okF)
                    for (std::size_t j = 0; j < m; ++j) {
                        lo[j] = std::min(lo[j], f[j]);
                        hi[j] = std::max(hi[j], f[j]);
                    }
                identity = Frame::from_anchors({lo, hi});
            }
        }
        const long long mm = static_cast<long long>(m * m);
        const long long cost = s.local == "sqp" ? 10 * (static_cast<long long>(n) + 1)
                                                : 100 * static_cast<long long>(n);
        if (k_sub == 0) k_sub = std::min<long long>(pop_size, std::max<long long>(1, (ev.budget - ev.spent) / cost - mm));
        s.n_sub = static_cast<double>(k_sub);
        plan_left = mm + k_sub;
        const long long floor = (ev.budget - ev.spent) / (4 * mm);   // SCALAR-1
        std::vector<Vec> anchors;
        for (std::size_t k = 0; k < m; ++k) {
            // lexicographic, by cycle: f_k, then f_{k+1}, ..., each stage
            // keeping the ones before no worse (SCALAR-9)
            std::vector<Row> rows;
            long long best = -1;
            for (std::size_t r = 0; r < m; ++r) {
                const std::size_t o = (k + r) % m;
                Sub stage;
                stage.kind = "lexi";
                stage.a0 = unit(m, o);
                stage.ineq = rows;
                const long long b = solve(stage, identity, true, floor);
                if (b < 0 || !found(stage, identity, b)) break;
                best = b;
                const double fo = identity.bar(ev.F[static_cast<std::size_t>(b)])[o];
                rows.push_back({unit(m, o, -1.0), 0.0, fo + 1e-9 * std::max(1.0, std::fabs(fo))});
            }
            if (best >= 0) {
                anchors.push_back(ev.F[static_cast<std::size_t>(best)]);
                solutions.push_back(static_cast<std::size_t>(best));
            }
        }
        if (anchors.size() < m) throw BudgetSpent{};
        const Frame frame = Frame::from_anchors(anchors);

        // 3. the plan
        const std::string& kind = s.scalarization;
        auto build = [&](const Vec& p) {
            if (kind == "eps") return eps(m, k_eps, p);
            if (kind == "tch") return tch(m, p, s.tch_weights);
            if (kind == "nbi") return nbi(frame, p);
            if (kind == "ps") return ps(frame, p);
            return nnc(frame, p);
        };
        auto params = [&](long long count) {
            long long L = 0;
            return kind == "eps" ? detail::levels(m, count, L) : detail::simplex(m, count);
        };
        if (m == 1) {
        } else if (s.sub_plan == "grid") {
            const std::vector<Vec> grid = params(k_sub);
            plan_left = static_cast<long long>(grid.size());
            for (const Vec& p : grid) {
                const Sub sub = build(p);
                const long long i = solve(sub, frame, false, 0);
                if (found(sub, frame, i)) solutions.push_back(static_cast<std::size_t>(i));
            }
        } else if (s.sub_plan == "gap") {
            std::vector<Vec> cand = params(4 * k_sub);
            std::vector<Vec> got;
            for (const Vec& a : anchors) got.push_back(frame.bar(a));
            plan_left = k_sub;
            const long long rounds = std::min<long long>(k_sub, static_cast<long long>(cand.size()));
            for (long long t = 0; t < rounds; ++t) {
                std::vector<Vec> have;
                for (const Vec& g : got) have.push_back(coords(kind, frame, g, k_eps));
                std::size_t j = 0;
                double far = -1.0;
                for (std::size_t c = 0; c < cand.size(); ++c) {
                    double near = std::numeric_limits<double>::infinity();
                    for (const Vec& h : have) {
                        double s2 = 0.0;
                        for (std::size_t q = 0; q < h.size(); ++q) s2 += (cand[c][q] - h[q]) * (cand[c][q] - h[q]);
                        near = std::min(near, std::sqrt(s2));
                    }
                    if (near > far) { far = near; j = c; }
                }
                const Sub sub = build(cand[j]);
                cand.erase(cand.begin() + static_cast<std::ptrdiff_t>(j));
                const long long i = solve(sub, frame, false, 0);
                if (found(sub, frame, i)) {
                    got.push_back(frame.bar(ev.F[static_cast<std::size_t>(i)]));
                    solutions.push_back(static_cast<std::size_t>(i));
                }
            }
        } else {
            // Laumanns, Thiele & Zitzler's adaptive ε-constraint scheme (Fig. 1
            // and 2), for minimization in the normalized objectives: the
            // bounds are upper ones, sorted from +∞ down, and a cell's strict
            // bound f_j < e_j is f_j <= e_j - δ.
            std::vector<std::size_t> others;
            for (std::size_t j = 0; j < m; ++j)
                if (j != k_eps) others.push_back(j);
            long long L = 0;
            detail::levels(m, k_sub, L);
            const double delta = 1.0 / static_cast<double>(L + 1);
            std::map<std::size_t, Vec> e;                 // descending: index 0 is +∞
            for (std::size_t j : others) e[j] = {std::numeric_limits<double>::infinity()};
            std::vector<Vec> P_found, Q, R;
            plan_left = k_sub;
            for (;;) {
                const long long p = static_cast<long long>(P_found.size());
                long long i = 1;
                for (std::size_t t = 0; t + 1 < m; ++t) i *= p + 1;
                bool fresh = false;
                while (i >= 0) {
                    Vec epsv;
                    long long rr = i;
                    for (std::size_t j : others) {        // getBounds
                        const long long d = rr % (p + 1);
                        rr /= p + 1;
                        epsv.push_back(e[j][static_cast<std::size_t>(d)]);
                    }
                    bool skip = std::find(R.begin(), R.end(), epsv) != R.end();
                    for (const Vec& q : Q) {
                        bool all_ge = true;
                        for (std::size_t t = 0; t < q.size(); ++t) all_ge = all_ge && q[t] >= epsv[t];
                        skip = skip || all_ge;
                    }
                    if (!skip) {
                        Vec lv;
                        for (double x : epsv) lv.push_back(x - delta);
                        const Sub first = eps(m, k_eps, lv);
                        const long long y = solve(first, frame, false, 0);
                        bool dominated = false;
                        Vec Fy;
                        if (y >= 0) {
                            Fy = frame.bar(ev.F[static_cast<std::size_t>(y)]);
                            for (const Vec& z : P_found) {
                                bool le = true;
                                for (std::size_t t = 0; t < m; ++t) le = le && z[t] <= Fy[t];
                                dominated = dominated || le;
                            }
                        }
                        if (!found(first, frame, y) || dominated) {
                            Q.push_back(epsv);
                        } else {
                            Sub second;
                            second.kind = "eps";
                            second.a0 = Vec(m, 1.0);
                            second.a0[k_eps] = 0.0;
                            second.ineq = first.ineq;
                            second.ineq.push_back({unit(m, k_eps, -1.0), 0.0, Fy[k_eps]});
                            long long x = solve(second, frame, false, 0);
                            if (!found(second, frame, x)) x = y;
                            const Vec Fx = frame.bar(ev.F[static_cast<std::size_t>(x)]);
                            P_found.push_back(Fx);
                            solutions.push_back(static_cast<std::size_t>(x));
                            R.push_back(epsv);
                            for (std::size_t j : others) {    // updateBounds
                                e[j].push_back(Fx[j]);
                                std::sort(e[j].begin(), e[j].end(), [](double a, double b) { return a > b; });
                            }
                            fresh = true;
                            break;
                        }
                    }
                    --i;
                }
                if (!fresh) break;
            }
        }
    } catch (const BudgetSpent&) {
    } catch (const SubBudgetSpent&) {
    }
    ev.over = true;
    const std::vector<std::size_t> idx = ev.answer();
    ev.record(true);
    Result r = detail::result_of(ev, idx);
    r.ignored = ignored;
    r.settings = settings_list(s);
    for (std::size_t i : solutions) r.solutions.push_back(ev.F[i]);
    return r;
}

// ── mosqp ───────────────────────────────────────────────────────────────────
inline Result run_mosqp_named(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0,
                              int pop_size, long long budget, const Knobs& knobs, Observer observer,
                              int record_every)
{
    MOSQPKnobs s;
    std::vector<std::string> ignored;
    std::map<std::string, double*> own{{"mosqp_tau", &s.tau}, {"mosqp_mu", &s.mu},
                                       {"mosqp_beta", &s.beta}, {"mosqp_spread", &s.spread},
                                       {"fd_step", &s.fd_step}};
    for (const auto& [k, v] : knobs) {
        if (own.count(k)) *own[k] = detail::number_of(k, v);
        else ignored.push_back(k);
    }
    std::sort(ignored.begin(), ignored.end());
    if (!(s.tau > 0 && 0 < s.mu && s.mu < 1 && 0 < s.beta && s.beta < 1 && s.spread >= 0 && s.fd_step > 0))
        throw std::invalid_argument("mosqp: mosqp_tau > 0, mosqp_mu and mosqp_beta in (0, 1), "
                                    "mosqp_spread >= 0, fd_step > 0");
    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    run_mosqp(ev, P.m, P.n_eq, pop_size, s);
    ev.over = true;
    const std::vector<std::size_t> idx = ev.answer();
    ev.record(true);
    Result r = detail::result_of(ev, idx);
    r.ignored = ignored;
    using detail::g_format;
    r.settings = {{"mosqp_tau", g_format(s.tau)}, {"mosqp_mu", g_format(s.mu)},
                  {"mosqp_beta", g_format(s.beta)}, {"mosqp_spread", g_format(s.spread)},
                  {"fd_step", g_format(s.fd_step)},
                  {"method_version", std::to_string(METHOD_VERSION)}};
    return r;
}

// ── boostdms ────────────────────────────────────────────────────────────────
inline Result run_boostdms_named(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0,
                                 int pop_size, long long budget, const Knobs& knobs, Observer observer,
                                 int record_every, const Start& start)
{
    std::vector<std::string> ignored;
    std::string how = "eps";
    for (const auto& [k, v] : knobs) {
        if (k == "boost_scalarization") how = detail::shown(v);
        else ignored.push_back(k);
    }
    std::sort(ignored.begin(), ignored.end());
    if (how != "eps" && how != "minmax")
        throw std::invalid_argument("boost_scalarization = '" + how + "': eps or minmax");
    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    std::vector<std::size_t> first;
    std::mt19937 unused(0);
    try {
        first = detail::first_points(ev, start, pop_size, unused, false);
    } catch (const BudgetSpent&) {
    }
    const std::vector<std::size_t> idx = run_boostdms(ev, P.m, pop_size, how == "minmax", first);
    ev.record(true);
    Result r = detail::result_of(ev, idx);
    r.ignored = ignored;
    r.settings = {{"boost_scalarization", how}, {"method_version", std::to_string(METHOD_VERSION)}};
    return r;
}

// Any of names(): the scalar presets, mosqp or boostdms.
inline Result run(const std::string& name, const Problem& P, double eq_tol, bool has_tol0,
                  double eq_tol0, int pop_size, long long budget, unsigned seed, const Knobs& knobs,
                  Observer observer, int record_every, const Start& start)
{
    if (name == "mosqp")
        return run_mosqp_named(P, eq_tol, has_tol0, eq_tol0, pop_size, budget, knobs, std::move(observer),
                               record_every);
    if (name == "boostdms")
        return run_boostdms_named(P, eq_tol, has_tol0, eq_tol0, pop_size, budget, knobs,
                                  std::move(observer), record_every, start);
    if (!detail::listed(preset_names(), name))
        throw std::invalid_argument("unknown scalarization method '" + name + "'");
    return run_scalar(name, P, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, knobs,
                      std::move(observer), record_every, start);
}

}   // namespace mootation::scalar
