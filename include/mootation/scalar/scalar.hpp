#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Scalarization with a local solver (task 9 of 2026-10-18, part A1; in C++
// since task 17, stage 6): the front as a sequence of single-objective
// subproblems, each solved by a local method — and, beside them, the methods
// of task 9 on the same evaluator, answer and observer: mosqp (mosqp.hpp) and
// boostdms (boostdms.hpp); and the local solvers alone as single-objective
// algorithms, slsqp, nelder_mead and cmaes (run_single), and lm and lbfgsb in
// the problem's own coordinates (run_local), at the end. Not
// evolutionary, and not a core of algorithms.def: run() takes the problem as
// a function and every evaluation goes through it (evaluator.hpp). The local
// solvers are the library's own (slsqp.hpp, nelder_mead.hpp, cmaes.hpp).
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
//   sqp_line_search  stop | accept | powell: sqp after 10 step lengths without
//                  Armijo's decrease stops (NW §3.5) or takes the last one
//                  (Kraft's code), slsqp.hpp SQP-6; or its merit and step
//                  length are Powell 1978's, §4 (powell; the owner's word,
//                  26.10)
//   fd_step        sqp's forward-difference step in the normalized box (1.49e-8)
//   local_tol      the solvers' tolerance (1e-14)
//   nm_coefficients  adaptive | standard: nm's coefficients, Gao & Han's (4.1)
//                  or Lagarias et al.'s (1.3) (nelder_mead.hpp; task 19,
//                  §19.4.2); in the settings where local = nm
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
//   SCALAR-14. Result::repairs counts a point when a method makes it and
//     it is looked at against the box, as Kononova, Caraffini and Bäck
//     count (Information Sciences 581, 2021; task 20, section 20.2.2) — de's, jade's, shade's and lshade's mutants,
//     g3_pcx's and the BLX/SBX children, pso's and spso2011's moves, cmaes's,
//     sep_cmaes's and ipop/bipop's samples (outside: clipped with the
//     penalty, or not, cma_bounds = none), nelder_mead's and slsqp's points
//     (clipped, NM-2), lbfgsb's x_0 and its search's points (projected,
//     LB-9); bound_repair = native counts what it leaves outside. Not
//     counted: lbfgsb's Cauchy path (the projected gradient path is the
//     method's geometry, not a point outside), lm (Coleman-Li's interior
//     steps stay inside; x_0 outside is refused), eki. A point made but not
//     evaluated (the budget ends within a generation) is counted.
//   SCALAR-15. A campaign's trajectory rows carry the counts at their fe
//     (job_so.hpp, campaign_so.py) where the run is on one thread; a strategy
//     on more than one (threads > 1, or an inner table's) gives null rows:
//     the counter is thread_local, and between rounds the job's thread holds
//     only what the rounds added, not the starts' points as they are made.
//     The run's totals (meta.json so.repairs, the last row) are exact on any
//     number of threads (strategy.hpp, round). A campaign refuses threads > 1
//     for now (task 18's 3g, config.hpp, until a job counts at the merge of
//     the starts), so its rows are full; the null rows are the rule for when
//     it takes them again.
//
// METHOD_VERSION: the version of these methods' trajectories, in every
// result's settings (a campaign's meta): 1 — SciPy's SLSQP and Nelder-Mead and
// NumPy's generator (task 9), 2 — the library's own solvers and rnd:: (task
// 17, stage 6), 3 — sqp_line_search = accept by default (the presets with
// local = sqp change; polish, mosqp and boostdms run as in 2), 4 —
// sqp_line_search = powell by default (the same presets change again).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
#include "../operators/bound_repair.hpp"
#include "../random.hpp"
#include "boostdms.hpp"
#include "chc.hpp"
#include "de.hpp"
#include "eki.hpp"
#include "ga.hpp"
#include "jade.hpp"
#include "lbfgsb.hpp"
#include "lm.hpp"
#include "lshade.hpp"
#include "mosqp.hpp"
#include "population.hpp"
#include "pso.hpp"
#include "sep_cmaes.hpp"
#include "shade.hpp"
#include "solvers.hpp"
#include "spso2011.hpp"

namespace mootation::scalar {

inline constexpr int METHOD_VERSION = 4;

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
    long long grad_calls = 0;        // the problem's gradient or Jacobian (evaluator.hpp THE COUNTS)
    long long n_failed = 0;          // failed evaluations in the archive
    std::vector<std::pair<std::string, std::string>> settings;
    Rows solutions;                  // scalar: every subproblem's solution, anchors first
    std::vector<Improvement> improvements;   // one objective: the best point's records (improvements.hpp)
    // the points the method made, how many of them left the box and by how
    // many variables (operators/bound_repair.hpp, RepairCounts; Kononova et
    // al. 2021): every repair, clip and projection of the run, its strategy's
    // starts included (task 20, §20.2.2)
    ops::RepairCounts repairs;
};

// How the first points come: seed points (planted), rows of a design
// (evaluated in order), or pop_size uniform points from the seed.
struct Start {
    std::vector<Point> seed_points;
    Rows design;
    bool has_seed_points = false, has_design = false;
    Vec x0;                          // a start in the problem's coordinates (task 18, addendum 1, §3.1)
    bool has_x0 = false;
};

// The names: the presets of `scalar`, then the methods beside it, then the
// single-objective solvers (run_single, run_local, run_population), then the
// strategies (strategy.hpp).
inline const std::vector<std::string>& preset_names()
{
    static const std::vector<std::string> v{"scalar", "eps_sqp", "tch_sqp", "nbi_sqp",
                                            "nnc_sqp", "eps_nm", "eps_cmaes"};
    return v;
}
inline const std::vector<std::string>& names()
{
    static const std::vector<std::string> v{"scalar", "eps_sqp", "tch_sqp", "nbi_sqp", "nnc_sqp",
                                            "eps_nm", "eps_cmaes", "boostdms", "mosqp",
                                            "slsqp", "nelder_mead", "cmaes", "lm", "lbfgsb", "eki",
                                            "de", "jade", "shade", "lshade", "ga_sbx", "ssga",
                                            "g3_pcx", "chc", "pso", "spso2011", "sep_cmaes",
                                            "multistart", "ipop", "bipop", "bipop_2009", "halving", "mlsl",
                                            "basin_hopping", "ils", "memetic"};
    return v;
}

// the strategies (strategy.hpp, included at the end)
inline const std::vector<std::string>& strategy_names();
inline Result run_strategy(const std::string& name, const Problem& P, double eq_tol, bool has_tol0,
                           double eq_tol0, int pop_size, long long budget, unsigned seed, const Knobs& knobs,
                           Observer observer, int record_every, const Start& start);

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
    std::string scalarization = "eps", local = "sqp", sub_plan = "grid", tch_weights = "inverse",
                sqp_line_search = "powell", nm_coefficients = "adaptive";
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
        {"sub_plan", {"grid", "laumanns", "gap"}},
        {"sqp_line_search", {"stop", "accept", "powell"}},
        {"nm_coefficients", {"adaptive", "standard"}}};
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
                                                {"local", &s.local}, {"sub_plan", &s.sub_plan},
                                                {"sqp_line_search", &s.sqp_line_search},
                                                {"nm_coefficients", &s.nm_coefficients}};
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
    std::vector<std::pair<std::string, std::string>> v{
        {"scalarization", s.scalarization}, {"local", s.local}, {"sub_plan", s.sub_plan},
        {"tch_weights", s.tch_weights}, {"sqp_line_search", s.sqp_line_search},
        {"n_sub", g_format(s.n_sub)},
        {"sub_budget", g_format(s.sub_budget)}, {"fd_step", g_format(s.fd_step)},
        {"local_tol", g_format(s.local_tol)}, {"cma_sigma0", g_format(s.cma_sigma0)},
        {"eps_objective", g_format(s.eps_objective)}};
    if (s.local == "nm") v.push_back({"nm_coefficients", s.nm_coefficients});
    v.push_back({"method_version", std::to_string(METHOD_VERSION)});
    return v;
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
    r.grad_calls = ev.grad_calls;
    r.improvements = ev.improvements.records();
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
                sqp(run, y0, sub.t_star(frame.bar(ev.F[st])), s.fd_step, s.local_tol,
                    s.sqp_line_search == "stop"     ? slsqp::LineSearch::stop
                    : s.sqp_line_search == "accept" ? slsqp::LineSearch::accept
                                                    : slsqp::LineSearch::powell);
            else if (s.local == "nm")
                nm(run, y0, {st}, s.local_tol, s.nm_coefficients == "standard");
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

// ── one objective: slsqp, nelder_mead, cmaes ────────────────────────────────
// The local solvers of `scalar` as single-objective algorithms (task 18, item
// 2.3), each with its knobs there: slsqp (sqp: fd_step, local_tol,
// sqp_line_search), nelder_mead (nm: local_tol, nm_coefficients), cmaes (cma_sigma0,
// local_tol, and cma_weights = active | positive, cma_cmu = tutorial2016 |
// hansen2009, cma_stop = spread | h09: the tutorial's weights and c_mu or
// those of BIPOP-CMA-ES 2009, the run stopped by its spread at local_tol or
// by H09's termination criteria, cmaes.hpp); the others are ignored and
// reported. n_objs = 1 only. The run:
//   1. the initial plan, as scalar's (THE PLAN, 1);
//   2. the solver once, from the plan's best point (feasible first, then f;
//      without a feasible point, the least violating one), until its own
//      tolerance or the budget;
//   3. the answer as every method's (evaluator.hpp): at one objective, the
//      feasible points of least f.
// Constraints as in scalar (CONSTRAINTS above).
//
// READINGS (declared):
//   SINGLE-1. One run of the solver: one that stops before the budget leaves
//     the rest unspent, as DMS does. Restarts are strategies of their own
//     (task 18, stage 3), and so are scalar's second start (SCALAR-9) and its
//     start again from the plan (SCALAR-11): none of them here. A run is the
//     same for every budget up to where the budget stops it, so it does not
//     depend on the budget (scalar.OTHERS in Python).
//   SINGLE-2. The solver sees f scaled by the initial plan's range (its least
//     value to 0, its greatest to 1), as scalar's anchors do (SCALAR-12), in
//     the box normalized to [0, 1]^n, as every solver of solvers.hpp: a bound
//     that is not finite is refused (task 18, addendum 1, section 3.1).
//   SINGLE-3. A start x0 (the problem's coordinates) is the plan: one point,
//     and the solver from it; the frame of SINGLE-2 is then f(x0) to 0 with
//     scale 1 (subproblems.hpp: a range of 0 is a scale of 1). slsqp takes the
//     problem's gradient where it gives one (solvers.hpp SCALAR-13), and
//     grad_cost then counts it (evaluator.hpp THE COUNTS; reported in the
//     settings only where there is a gradient).
inline const std::vector<std::string>& single_names()
{
    static const std::vector<std::string> v{"slsqp", "nelder_mead", "cmaes"};
    return v;
}

inline Result run_single(const std::string& name, const Problem& P, double eq_tol, bool has_tol0,
                         double eq_tol0, int pop_size, long long budget, unsigned seed,
                         const Knobs& knobs, Observer observer, int record_every, const Start& start)
{
    using detail::g_format;
    const std::string local = name == "slsqp" ? "sqp" : name == "nelder_mead" ? "nm" : "cmaes";
    if (P.m != 1)
        throw std::invalid_argument(name + " is a single-objective algorithm: n_objs must be 1, got " +
                                    std::to_string(P.m));
    for (std::size_t j = 0; j < P.lo.size(); ++j)
        if (!std::isfinite(P.lo[j]) || !std::isfinite(P.hi[j]))
            throw std::invalid_argument(name + " works in the box normalized to [0, 1]^n: every bound must be "
                                               "finite (SINGLE-2)");
    std::vector<std::string> own{"local_tol", "grad_cost"};
    if (local == "sqp") own.insert(own.end(), {"fd_step", "sqp_line_search"});
    if (local == "nm") own.push_back("nm_coefficients");
    if (local == "cmaes") own.insert(own.end(), {"cma_sigma0", "cma_weights", "cma_cmu", "cma_stop"});
    Knobs taken;
    std::vector<std::string> ignored, none;
    double grad_cost = 1.0;
    for (const auto& kv : knobs) {
        if (kv.first == "grad_cost") {
            grad_cost = detail::number_of(kv.first, kv.second);
            if (!std::isfinite(grad_cost) || grad_cost < 0.0) throw std::invalid_argument("grad_cost must be a number >= 0");
        } else if (detail::listed(own, kv.first)) {
            taken.push_back(kv);
        } else {
            ignored.push_back(kv.first);
        }
    }
    std::sort(ignored.begin(), ignored.end());
    const ScalarSettings s = scalar_settings("scalar", taken, none);   // the words and numbers checked
    std::map<std::string, std::string> cma{{"cma_weights", "active"}, {"cma_cmu", "tutorial2016"},
                                           {"cma_stop", "spread"}};
    for (const auto& [k, v] : taken)
        if (cma.count(k)) {
            static const std::map<std::string, std::vector<std::string>> words{
                {"cma_weights", {"active", "positive"}}, {"cma_cmu", {"tutorial2016", "hansen2009"}},
                {"cma_stop", {"spread", "h09"}}};
            const std::string w = detail::shown(v);
            if (!v.is_word || !detail::listed(words.at(k), w))
                throw std::invalid_argument(k + " = '" + w + "': " + detail::joined(words.at(k)));
            cma[k] = w;
        }
    CMAESOptions cma_options;
    if (cma["cma_weights"] == "positive") cma_options.weights = CMAESOptions::Weights::positive;
    if (cma["cma_cmu"] == "hansen2009") cma_options.cmu = CMAESOptions::Cmu::hansen2009;
    if (start.has_x0 && start.x0.size() != P.lo.size())
        throw std::invalid_argument(name + ": x0 has " + std::to_string(start.x0.size()) + " values for " +
                                    std::to_string(P.lo.size()) + " variables");

    std::mt19937 rng(seed);
    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    ev.grad_cost = grad_cost;
    try {
        if (start.has_x0) ev.index(start.x0);         // the plan: x0 alone (SINGLE-3)
        else detail::first_points(ev, start, pop_size, rng, true);
        if (!ev.recorded) ev.record();               // gen 0, unless the plan's last point gave it
        long long st = -1;
        for (std::size_t i = 0; i < ev.X.size(); ++i) {
            if (ev.failed[i]) continue;
            if (st < 0) {
                st = static_cast<long long>(i);
                continue;
            }
            const std::size_t b = static_cast<std::size_t>(st);
            const double ci = ev.cv(i), cb = ev.cv(b);
            if (ci <= 0.0 && cb <= 0.0 ? ev.F[i][0] < ev.F[b][0] : ci < cb) st = static_cast<long long>(i);
        }
        if (st >= 0) {
            double lo = ev.F[static_cast<std::size_t>(st)][0], hi = lo;
            for (std::size_t i = 0; i < ev.X.size(); ++i)
                if (!ev.failed[i]) {
                    lo = std::min(lo, ev.F[i][0]);
                    hi = std::max(hi, ev.F[i][0]);
                }
            const Frame frame = Frame::from_anchors({{lo}, {hi}});   // SINGLE-2
            Sub sub;
            sub.kind = "lexi";
            sub.a0 = unit(1, 0);
            Run run(sub, ev, frame);
            const std::size_t i0 = static_cast<std::size_t>(st);
            run.touched.push_back(i0);
            const Vec y0 = run.y_of(i0);
            if (local == "sqp")
                sqp(run, y0, sub.t_star(frame.bar(ev.F[i0])), s.fd_step, s.local_tol,
                    s.sqp_line_search == "stop"     ? slsqp::LineSearch::stop
                    : s.sqp_line_search == "accept" ? slsqp::LineSearch::accept
                                                    : slsqp::LineSearch::powell);
            else if (local == "nm")
                nm(run, y0, {i0}, s.local_tol, s.nm_coefficients == "standard");
            else
                cmaes(run, y0, {i0}, s.cma_sigma0, s.local_tol, rng, cma_options, cma["cma_stop"] == "h09");
        }
    } catch (const BudgetSpent&) {
    } catch (const SubBudgetSpent&) {
    }
    ev.over = true;
    const std::vector<std::size_t> idx = ev.answer();
    ev.record(true);
    Result r = detail::result_of(ev, idx);
    r.ignored = ignored;
    r.settings = {{"local", local}, {"local_tol", g_format(s.local_tol)}};
    if (local == "sqp")
        r.settings.insert(r.settings.end(),
                          {{"sqp_line_search", s.sqp_line_search}, {"fd_step", g_format(s.fd_step)}});
    if (local == "nm") r.settings.push_back({"nm_coefficients", s.nm_coefficients});
    if (local == "cmaes")
        r.settings.insert(r.settings.end(), {{"cma_sigma0", g_format(s.cma_sigma0)},
                                             {"cma_weights", cma["cma_weights"]}, {"cma_cmu", cma["cma_cmu"]},
                                             {"cma_stop", cma["cma_stop"]}});
    if (P.has_gradient()) r.settings.push_back({"grad_cost", g_format(grad_cost)});
    r.settings.push_back({"method_version", std::to_string(METHOD_VERSION)});
    return r;
}

// ── one objective in the problem's coordinates: lm, lbfgsb ─────────────────
// The least-squares and the bound-constrained quasi-Newton solvers of stage 2
// as single-objective algorithms (task 18, stage 1, §1.1; addendum 1, §3.1):
// lm (lm.hpp, Moré 1978; the problem's residuals), lbfgsb (lbfgsb.hpp,
// Byrd, Lu, Nocedal, Zhu 1995) and eki (eki.hpp, ensemble Kalman inversion,
// Iglesias, Law, Stuart 2013; the residuals). All work in the problem's own coordinates,
// bounds -inf / +inf allowed, and take the problem's derivatives where it
// gives them (the residuals' Jacobian; the gradient), else forward
// differences of their own, every one an evaluation. The run:
//   1. the start: start.x0 (the problem's coordinates) where it is given;
//      else the initial plan, as scalar's (THE PLAN, 1), and its best point,
//      which needs finite bounds;
//   2. the solver once from there, until its own tolerance or the budget
//      (SINGLE-1);
//   3. the answer as every method's (evaluator.hpp).
// Knobs: lm_ftol, lm_xtol, lm_gtol (Moré's (8.3)-(8.4); 1e-8, 1e-8, 0),
// lm_loss (linear | huber | cauchy; lm.hpp's IRLS), lm_f_scale (1) and
// lm_gauge (off | on: the step in the complement of the problem's
// gauge_basis, lm.hpp LMN; task 19, §19.3.1) for lm; lbfgsb_m (5),
// lbfgsb_pgtol (1e-5), lbfgsb_ftol (0) for lbfgsb; grad_cost (1;
// evaluator.hpp THE COUNTS) for both; for eki (eki.hpp's Options, task 19,
// §19.4.1) eki_ensemble (J; 0: max(10, 2 n), EKI-4), eki_init (around |
// sobol | lhs; LOCAL-5), eki_sigma0 (0.1), eki_gamma (1), eki_perturb (yes |
// no), eki_noise (||eta^dagger||_Gamma of the discrepancy principle; 0: off)
// and eki_tau (1.5).
//
// READINGS (declared):
//   LOCAL-1. Bounds only: a problem with constraints is refused, and lm
//     refuses a problem without residuals (its objective is 1/2 ||r~||^2).
//   LOCAL-2. Neither solver's own limit on evaluations (lm's max_nfev,
//     lbfgsb's max_nfev and max_iter) is put: the budget stops them.
//   LOCAL-3. A failed evaluation ends the run, as sqp's (SCALAR-6): neither
//     method can step over a hole in the function.
//   LOCAL-4. Without x0, a bound that is not finite is refused: there is no
//     box to draw the plan in, and no frame is put on x (addendum 1, §3.1).
//   LOCAL-5. eki starts as lm does, from x0 or the plan's best point: its
//     initial ensemble around that point by default (eki_init = around,
//     EKI-3), or a design in the box (sobol, lhs; finite bounds). Its seed
//     is the run's; its own stop on evaluations is off (LOCAL-2), so its
//     last mean (EKI-5) is evaluated only where the budget leaves room: the
//     answer is the evaluator's, as every method's.
//   LOCAL-6. lm_gauge = on needs the problem's gauge_basis; its k and ||J N||
//     / ||J|| at lm's solution (lm.hpp LMN-4) go to the settings
//     (lm_gauge_k, lm_gauge_ratio).
inline const std::vector<std::string>& local_names()
{
    static const std::vector<std::string> v{"lm", "lbfgsb", "eki"};
    return v;
}

inline Result run_local(const std::string& name, const Problem& P, double eq_tol, bool has_tol0,
                        double eq_tol0, int pop_size, long long budget, unsigned seed,
                        const Knobs& knobs, Observer observer, int record_every, const Start& start)
{
    using detail::g_format;
    const std::size_t n = P.lo.size();
    if (P.m != 1)
        throw std::invalid_argument(name + " is a single-objective algorithm: n_objs must be 1, got " +
                                    std::to_string(P.m));
    if (P.n_cons != 0)
        throw std::invalid_argument(name + " takes bounds only: the problem's constraints are not for it (LOCAL-1)");
    if ((name == "lm" || name == "eki") && !P.has_residuals())
        throw std::invalid_argument(name + " solves least squares: the problem gives no residuals (LOCAL-1)");
    if (start.has_x0 && start.x0.size() != n)
        throw std::invalid_argument(name + ": x0 has " + std::to_string(start.x0.size()) + " values for " +
                                    std::to_string(n) + " variables");
    if (!start.has_x0)
        for (std::size_t j = 0; j < n; ++j)
            if (!std::isfinite(P.lo[j]) || !std::isfinite(P.hi[j]))
                throw std::invalid_argument(name + ": a bound that is not finite needs a start x0 (LOCAL-4)");
    const std::vector<std::string> own =
        name == "lm"    ? std::vector<std::string>{"lm_ftol", "lm_xtol", "lm_gtol", "lm_loss", "lm_f_scale", "lm_gauge",
                                                   "grad_cost"}
        : name == "eki" ? std::vector<std::string>{"eki_ensemble", "eki_init", "eki_sigma0", "eki_gamma", "eki_perturb",
                                                   "eki_noise", "eki_tau"}
                        : std::vector<std::string>{"lbfgsb_m", "lbfgsb_pgtol", "lbfgsb_ftol", "grad_cost"};
    static const std::map<std::string, std::vector<std::string>> words{
        {"lm_loss", {"linear", "huber", "cauchy"}}, {"lm_gauge", {"off", "on"}},
        {"eki_init", {"around", "sobol", "lhs"}}, {"eki_perturb", {"yes", "no"}}};
    lm::Options lo;
    lbfgsb::Options bo;
    eki::Options eo;
    std::map<std::string, std::string> word{{"lm_loss", "linear"}, {"lm_gauge", "off"},
                                            {"eki_init", "around"}, {"eki_perturb", "yes"}};
    double grad_cost = 1.0;
    std::vector<std::string> ignored;
    for (const auto& [k, v] : knobs) {
        if (!detail::listed(own, k)) {
            ignored.push_back(k);
            continue;
        }
        if (words.count(k)) {
            const std::string w = detail::shown(v);
            if (!v.is_word || !detail::listed(words.at(k), w))
                throw std::invalid_argument(k + " = '" + w + "': " + detail::joined(words.at(k)));
            word[k] = w;
            continue;
        }
        const double x = detail::number_of(k, v);
        if (!std::isfinite(x) || x < 0.0) throw std::invalid_argument(k + " must be a number >= 0");
        if (k == "lm_ftol") lo.ftol = x;
        else if (k == "lm_xtol") lo.xtol = x;
        else if (k == "lm_gtol") lo.gtol = x;
        else if (k == "lm_f_scale") lo.f_scale = x;
        else if (k == "lbfgsb_pgtol") bo.pgtol = x;
        else if (k == "lbfgsb_ftol") bo.ftol = x;
        else if (k == "grad_cost") grad_cost = x;
        else if (k == "eki_sigma0") eo.sigma0 = x;
        else if (k == "eki_gamma") eo.gamma = x;
        else if (k == "eki_noise") eo.noise_norm = x;
        else if (k == "eki_tau") eo.tau = x;
        else if (k == "eki_ensemble") {
            if (x != std::floor(x) || x == 1.0 || x > 1e6)
                throw std::invalid_argument("eki_ensemble must be a whole number >= 2 (0: max(10, 2 n))");
            eo.ensemble = static_cast<int>(x);
        } else {
            if (x < 1.0 || x != std::floor(x)) throw std::invalid_argument("lbfgsb_m must be a whole number >= 1");
            bo.m = static_cast<int>(x);
        }
    }
    std::sort(ignored.begin(), ignored.end());
    if (lo.f_scale <= 0.0) throw std::invalid_argument("lm_f_scale must be > 0");
    if (!(eo.sigma0 > 0.0) || !(eo.gamma > 0.0) || !(eo.tau > 0.0))
        throw std::invalid_argument("eki_sigma0, eki_gamma and eki_tau must be > 0");
    const std::string loss = word["lm_loss"];
    const bool gauge = word["lm_gauge"] == "on";
    if (gauge && !P.gauge_basis)
        throw std::invalid_argument("lm_gauge = 'on' needs the problem's gauge_basis (LOCAL-6)");
    eo.init = word["eki_init"] == "sobol" ? eki::Options::Init::sobol
            : word["eki_init"] == "lhs"   ? eki::Options::Init::lhs : eki::Options::Init::around;
    eo.perturb = word["eki_perturb"] == "yes";
    eo.seed = static_cast<std::uint32_t>(seed);
    lo.loss = loss == "huber" ? lm::Options::Loss::huber
            : loss == "cauchy" ? lm::Options::Loss::cauchy : lm::Options::Loss::linear;
    const long long no_limit = std::numeric_limits<long long>::max() / 4;   // LOCAL-2
    lo.max_nfev = no_limit;
    bo.max_nfev = no_limit;
    bo.max_iter = no_limit;
    eo.max_evals = no_limit;
    lm::Result lr;

    std::mt19937 rng(seed);
    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    ev.grad_cost = grad_cost;
    try {
        long long st = -1;
        if (start.has_x0) {
            st = static_cast<long long>(ev.index(start.x0));
        } else {
            detail::first_points(ev, start, pop_size, rng, true);
            for (std::size_t i = 0; i < ev.X.size(); ++i)
                if (!ev.failed[i] && (st < 0 || ev.F[i][0] < ev.F[static_cast<std::size_t>(st)][0]))
                    st = static_cast<long long>(i);
        }
        if (!ev.recorded) ev.record();                // gen 0
        if (st >= 0 && !ev.failed[static_cast<std::size_t>(st)]) {
            const Vec x0 = ev.X[static_cast<std::size_t>(st)];
            auto value = [&](const Vec& x) {
                const std::size_t i = ev.index(x);
                if (ev.failed[i]) throw FailedPoint{};     // LOCAL-3
                return i;
            };
            if (name == "lm") {
                lm::Problem p;
                p.n = n;
                p.m = static_cast<std::size_t>(ev.n_res);
                p.lo = P.lo;
                p.hi = P.hi;
                p.residuals = [&](const Vec& x, Vec& r) { r = ev.R[value(x)]; };
                if (P.has_gradient())
                    p.jacobian = [&](const Vec& x, lm::Matrix& J) {
                        const std::size_t i = ev.index_grad(x);
                        if (ev.failed[i]) throw FailedPoint{};
                        const Rows& Jr = ev.jac(i);
                        for (std::size_t a = 0; a < J.rows; ++a)
                            for (std::size_t b = 0; b < J.cols; ++b) J(a, b) = Jr[a][b];
                    };
                if (gauge)
                    p.gauge = [&](const Vec& x, lm::Matrix& N) {
                        Rows Nr;
                        P.gauge_basis(x, Nr);
                        const std::size_t k = Nr.empty() ? 0 : Nr[0].size();
                        if (Nr.size() != n) throw std::invalid_argument("gauge_basis: n rows are needed");
                        N = lm::Matrix(n, k);
                        for (std::size_t a = 0; a < n; ++a) {
                            if (Nr[a].size() != k) throw std::invalid_argument("gauge_basis: rows of one length");
                            for (std::size_t b = 0; b < k; ++b) N(a, b) = Nr[a][b];
                        }
                    };
                lr = lm::solve_robust(p, x0, lo);
            } else if (name == "eki") {
                eki::minimize([&](const Vec& x, Vec& r) { r = ev.R[value(x)]; },
                              static_cast<std::size_t>(ev.n_res), P.lo, P.hi, x0, eo);
            } else {
                lbfgsb::Problem p;
                p.n = n;
                p.lo = P.lo;
                p.hi = P.hi;
                p.f = [&](const Vec& x) { return ev.F[value(x)][0]; };
                if (P.has_gradient())
                    p.fg = [&](const Vec& x, Vec& g) {
                        const std::size_t i = ev.index_grad(x);
                        if (ev.failed[i]) throw FailedPoint{};
                        g = ev.grad(i);
                        return ev.F[i][0];
                    };
                lbfgsb::solve(p, x0, bo);
            }
        }
    } catch (const BudgetSpent&) {
    } catch (const SubBudgetSpent&) {
    } catch (const FailedPoint&) {
    }
    ev.over = true;
    const std::vector<std::size_t> idx = ev.answer();
    ev.record(true);
    Result r = detail::result_of(ev, idx);
    r.ignored = ignored;
    if (name == "lm") {
        r.settings = {{"lm_ftol", g_format(lo.ftol)}, {"lm_xtol", g_format(lo.xtol)},
                      {"lm_gtol", g_format(lo.gtol)}, {"lm_loss", loss}, {"lm_f_scale", g_format(lo.f_scale)},
                      {"lm_gauge", word["lm_gauge"]}};
        if (gauge)                                    // LOCAL-6 (0 where the run ended before lm did)
            r.settings.insert(r.settings.end(), {{"lm_gauge_k", std::to_string(lr.gauge_k)},
                                                 {"lm_gauge_ratio", g_format(lr.gauge_ratio)}});
    } else if (name == "eki") {
        r.settings = {{"eki_ensemble", std::to_string(eo.ensemble > 0 ? eo.ensemble
                                                                      : std::max<int>(10, 2 * static_cast<int>(n)))},
                      {"eki_init", word["eki_init"]}, {"eki_sigma0", g_format(eo.sigma0)},
                      {"eki_gamma", g_format(eo.gamma)}, {"eki_perturb", word["eki_perturb"]},
                      {"eki_noise", g_format(eo.noise_norm)}, {"eki_tau", g_format(eo.tau)}};
    } else {
        r.settings = {{"lbfgsb_m", std::to_string(bo.m)}, {"lbfgsb_pgtol", g_format(bo.pgtol)},
                      {"lbfgsb_ftol", g_format(bo.ftol)}};
    }
    if (name != "eki") r.settings.push_back({"grad_cost", g_format(grad_cost)});
    r.settings.push_back({"method_version", std::to_string(METHOD_VERSION)});
    return r;
}

// ── one objective, a population: stage 4's methods ──────────────────────────
// The global single-objective methods of stage 4 (task 18, §4.1-4.4) as
// algorithms, on the evaluator (population.hpp: the loop, POP-1 ... POP-4):
//   de         DE/rand/1/bin, Storn & Price 1997 (de.hpp)
//   jade       Zhang & Sanderson 2009 (jade.hpp)
//   shade      Tanabe & Fukunaga 2013 (shade.hpp)
//   lshade     Tanabe & Fukunaga 2014 (lshade.hpp)
//   ga_sbx     Deb & Deb 2014's real-coded GA; ssga its steady state (ga.hpp)
//   g3_pcx     Deb, Anand, Joshi 2002 (ga.hpp, G3)
//   chc        Eshelman 1991 with BLX-0.5, Eshelman & Schaffer 1993 (chc.hpp)
//   pso        Clerc & Kennedy 2002, Type 1'' (pso.hpp)
//   spso2011   Zambrano-Bigiarini, Clerc, Rojas 2013 (spso2011.hpp)
//   sep_cmaes  Ros & Hansen 2008 (sep_cmaes.hpp)
// Each with the defaults of its paper and these knobs (0 for np: the paper's
// size; every other knob's default is the paper's value its header names):
//   all        np: the population (de NP, jade NP, shade N, lshade N_init,
//              ga mu, g3 N, chc M, pso and spso2011 the swarm, sep_cmaes
//              lambda)
//   de         F (0.5), CR (0.1), bound_repair (reflect)
//   jade       jade_p (0.05), jade_c (0.1), de_archive (yes)
//   shade      shade_h (100), shade_p_max (0.2), de_archive (yes)
//   lshade     shade_h (6), lshade_p (0.11), lshade_r_arc (2.6),
//              lshade_n_min (4), lshade_max_nfe (0: the run's budget)
//   ga_sbx     pc (0.9), eta_c (2), pm (0: 1/n), eta_m (20),
//   ssga       ga_tournament (2); ssga is (100 + 1) with the coin
//   g3_pcx     g3_mu (3), g3_lambda (2), g3_sigma (0.1: sigma_zeta and
//              sigma_eta), bound_repair (reflect)
//   chc        chc_divergence (0.35), blx_alpha (0.5), chc_bits (10),
//              bound_repair (reflect)
//   pso        pso_phi (4.1), kappa (1), pso_topology (ring | global),
//              pso_v0 (random | zero), bound_repair (reflect)
//   spso2011   spso_k (3)
//   sep_cmaes  cma_sigma0 (0.3, in the normalized box), local_tol (1e-14)
// The other knobs are ignored and reported. n_objs = 1 only; the answer as
// every method's (evaluator.hpp): the best point.
//
// READINGS (declared):
//   POP-5. Every bound must be finite: the methods draw their first
//     population uniformly in the box (sep_cmaes its mean), and the papers'
//     boxes are where they search.
//   POP-6. The population is the method's own (np), not pop_size: the papers'
//     sizes are part of the methods (lshade's N_init = 18 D, pso's 20).
//     pop_size still sets the observer's stride (evaluator.hpp).
//   POP-7. The first population is the method's draw: x0, a design (init)
//     and seed points are not used and are reported as ignored, except x0
//     for sep_cmaes, its mean (normalized and clipped to the box); without
//     x0 the mean is a uniform point of the box, drawn first.
//   POP-8. lshade's MAX_NFE is the run's budget unless lshade_max_nfe is
//     set: §III's MAX_NFE is the budget the run is given.
inline const std::vector<std::string>& population_names()
{
    static const std::vector<std::string> v{"de", "jade", "shade", "lshade", "ga_sbx", "ssga",
                                            "g3_pcx", "chc", "pso", "spso2011", "sep_cmaes"};
    return v;
}

namespace detail {

// The knobs of a population method.
inline std::vector<std::string> population_knobs(const std::string& name)
{
    std::vector<std::string> k{"np"};
    if (name == "de") k.insert(k.end(), {"F", "CR", "bound_repair"});
    else if (name == "jade") k.insert(k.end(), {"jade_p", "jade_c", "de_archive"});
    else if (name == "shade") k.insert(k.end(), {"shade_h", "shade_p_max", "de_archive"});
    else if (name == "lshade")
        k.insert(k.end(), {"shade_h", "lshade_p", "lshade_r_arc", "lshade_n_min", "lshade_max_nfe"});
    else if (name == "ga_sbx" || name == "ssga") k.insert(k.end(), {"pc", "eta_c", "pm", "eta_m", "ga_tournament"});
    else if (name == "g3_pcx") k.insert(k.end(), {"g3_mu", "g3_lambda", "g3_sigma", "bound_repair"});
    else if (name == "chc") k.insert(k.end(), {"chc_divergence", "blx_alpha", "chc_bits", "bound_repair"});
    else if (name == "pso") k.insert(k.end(), {"pso_phi", "kappa", "pso_topology", "pso_v0", "bound_repair"});
    else if (name == "spso2011") k.insert(k.end(), {"spso_k"});
    else k.insert(k.end(), {"cma_sigma0", "local_tol"});
    return k;
}

// The knobs given to a population method, read by name with their defaults.
class PopulationKnobs {
public:
    PopulationKnobs(const std::string& name, const Knobs& knobs) : own_(population_knobs(name))
    {
        for (const auto& kv : knobs) {
            if (listed(own_, kv.first)) given_.push_back(kv);
            else ignored.push_back(kv.first);
        }
    }

    std::vector<std::string> ignored;
    std::vector<std::pair<std::string, std::string>> settings;   // what the run used, in reading order

    double number(const std::string& k, double fallback)
    {
        double x = fallback;
        if (const Knob* v = find(k)) {
            x = number_of(k, *v);
            if (!std::isfinite(x)) throw std::invalid_argument(k + " must be a finite number");
        }
        settings.push_back({k, g_format(x)});
        return x;
    }

    std::size_t whole(const std::string& k, std::size_t fallback)
    {
        double x = static_cast<double>(fallback);
        if (const Knob* v = find(k)) {
            x = number_of(k, *v);
            if (!(x >= 0.0) || x != std::floor(x) || x > 1e9)
                throw std::invalid_argument(k + " must be a whole number >= 0");
        }
        settings.push_back({k, g_format(x)});
        return static_cast<std::size_t>(x);
    }

    std::string word(const std::string& k, const std::string& fallback, const std::vector<std::string>& words)
    {
        std::string w = fallback;
        if (const Knob* v = find(k)) {
            w = shown(*v);
            if (!v->is_word || !listed(words, w)) throw std::invalid_argument(k + " = '" + w + "': " + joined(words));
        }
        settings.push_back({k, w});
        return w;
    }

    ops::BoundRepair repair()
    {
        static const std::vector<std::string> words{"clip", "reflect", "random", "midpoint",
                                                    "resample", "wrap", "native", "r_reflect"};
        return *ops::parse_bound_repair(word("bound_repair", "reflect", words));
    }

    // np as given (0: the paper's), and the size the method took, first in the settings
    std::size_t np()
    {
        const std::size_t n = whole("np", 0);
        settings.pop_back();
        return n;
    }
    void size(std::size_t n) { settings.insert(settings.begin(), {"np", std::to_string(n)}); }

private:
    std::vector<std::string> own_;
    Knobs given_;

    const Knob* find(const std::string& k) const
    {
        for (const auto& kv : given_)
            if (kv.first == k) return &kv.second;
        return nullptr;
    }
};

}   // namespace detail

inline Result run_population(const std::string& name, const Problem& P, double eq_tol, bool has_tol0,
                             double eq_tol0, int pop_size, long long budget, unsigned seed,
                             const Knobs& knobs, Observer observer, int record_every, const Start& start)
{
    if (P.m != 1)
        throw std::invalid_argument(name + " is a single-objective algorithm: n_objs must be 1, got " +
                                    std::to_string(P.m));
    for (std::size_t j = 0; j < P.lo.size(); ++j)
        if (!std::isfinite(P.lo[j]) || !std::isfinite(P.hi[j]))
            throw std::invalid_argument(name + " draws its population in the box: every bound must be finite (POP-5)");
    if (start.has_x0 && start.x0.size() != P.lo.size())
        throw std::invalid_argument(name + ": x0 has " + std::to_string(start.x0.size()) + " values for " +
                                    std::to_string(P.lo.size()) + " variables");
    detail::PopulationKnobs k(name, knobs);
    if (start.has_x0 && name != "sep_cmaes") k.ignored.push_back("x0");             // POP-7
    if (start.has_design) k.ignored.push_back("init");
    if (start.has_seed_points) k.ignored.push_back("seed_points");
    std::sort(k.ignored.begin(), k.ignored.end());
    const std::vector<std::string> yes_no{"yes", "no"};
    const Vec lo = P.lo, hi = P.hi;
    const std::size_t np = k.np();

    std::mt19937 rng(seed);
    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    try {
        if (name == "de") {
            de::Options o;
            o.np = np;
            o.F = k.number("F", o.F);
            o.CR = k.number("CR", o.CR);
            o.repair = k.repair();
            de::DE m(lo, hi, o, rng);
            k.size(m.np());
            population::drive(m, ev, [&] { return m.done(); });
        } else if (name == "jade") {
            jade::Options o;
            o.np = np;
            o.p = k.number("jade_p", o.p);
            o.c = k.number("jade_c", o.c);
            o.archive = k.word("de_archive", "yes", yes_no) == "yes";
            jade::JADE m(lo, hi, o, rng);
            k.size(m.np());
            population::drive(m, ev, [&] { return m.done(); });
        } else if (name == "shade") {
            shade::Options o;
            if (np) o.np = np;
            o.H = k.whole("shade_h", o.H);
            o.p_max = k.number("shade_p_max", o.p_max);
            o.archive = k.word("de_archive", "yes", yes_no) == "yes";
            shade::SHADE m(lo, hi, o, rng);
            k.size(m.np());
            population::drive(m, ev, [&] { return m.done(); });
        } else if (name == "lshade") {
            lshade::Options o;
            if (np) o.r_ninit = static_cast<double>(np) / static_cast<double>(lo.size());   // N_init = np
            o.H = k.whole("shade_h", o.H);
            o.p = k.number("lshade_p", o.p);
            o.r_arc = k.number("lshade_r_arc", o.r_arc);
            o.n_min = k.whole("lshade_n_min", o.n_min);
            o.max_nfe = static_cast<long long>(k.whole("lshade_max_nfe", 0));
            if (o.max_nfe == 0) o.max_nfe = budget;                                       // POP-8
            lshade::LSHADE m(lo, hi, o, rng);
            k.size(m.np());
            population::drive(m, ev, [&] { return m.done(); });
        } else if (name == "ga_sbx" || name == "ssga") {
            ga::Options o = name == "ssga" ? ga::ssga() : ga::ga_sbx();
            if (np) o.mu = np;
            o.pc = k.number("pc", o.pc);
            o.eta_c = k.number("eta_c", o.eta_c);
            o.pm = k.number("pm", o.pm);
            o.eta_m = k.number("eta_m", o.eta_m);
            o.tournament = k.whole("ga_tournament", o.tournament);
            ga::GA m(lo, hi, o, rng);
            k.size(m.mu());
            population::drive(m, ev, [&] { return m.done(); });
        } else if (name == "g3_pcx") {
            ga::G3Options o;
            if (np) o.n = np;
            o.mu = k.whole("g3_mu", o.mu);
            o.lambda = k.whole("g3_lambda", o.lambda);
            o.sigma_zeta = o.sigma_eta = k.number("g3_sigma", o.sigma_zeta);
            o.repair = k.repair();
            ga::G3 m(lo, hi, o, rng);
            k.size(o.n);
            population::drive(m, ev, [] { return false; });
        } else if (name == "chc") {
            chc::Options o;
            if (np) o.m = np;
            o.divergence = k.number("chc_divergence", o.divergence);
            o.alpha = k.number("blx_alpha", o.alpha);
            o.bits = static_cast<unsigned>(k.whole("chc_bits", o.bits));
            o.repair = k.repair();
            chc::CHC m(lo, hi, o, rng);
            k.size(o.m);
            population::drive(m, ev, [] { return false; });
        } else if (name == "pso") {
            pso::Options o;
            if (np) o.n = np;
            o.phi_max = k.number("pso_phi", o.phi_max);
            o.kappa = k.number("kappa", o.kappa);
            o.ring = k.word("pso_topology", "ring", {"ring", "global"}) == "ring";
            o.v0_zero = k.word("pso_v0", "random", {"random", "zero"}) == "zero";
            o.repair = k.repair();
            pso::PSO m(lo, hi, o, rng);
            k.size(o.n);
            population::drive(m, ev, [&] { return m.done(); });
        } else if (name == "spso2011") {
            spso2011::Options o;
            if (np) o.n = np;
            o.K = k.whole("spso_k", o.K);
            spso2011::SPSO2011 m(lo, hi, o, rng);
            k.size(o.n);
            population::drive(m, ev, [] { return false; });
        } else {
            const double sigma0 = k.number("cma_sigma0", 0.3);
            const double tol = k.number("local_tol", 1e-14);
            if (!(sigma0 > 0.0) || !(tol > 0.0)) throw std::invalid_argument("cma_sigma0 and local_tol must be > 0");
            Vec y0(lo.size());
            for (std::size_t j = 0; j < lo.size(); ++j)                                    // POP-7
                y0[j] = start.has_x0 ? std::min(1.0, std::max(0.0, (start.x0[j] - lo[j]) / (hi[j] - lo[j])))
                                     : rnd::uniform01(rng);
            sep_cmaes::SepCMAES m(y0, sigma0, rng, np);
            k.size(m.lambda());
            population::drive_normalized(m, ev, tol);
        }
    } catch (const BudgetSpent&) {
    }
    ev.over = true;
    const std::vector<std::size_t> idx = ev.answer();
    ev.record(true);
    Result r = detail::result_of(ev, idx);
    r.ignored = k.ignored;
    r.settings = k.settings;
    r.settings.push_back({"method_version", std::to_string(METHOD_VERSION)});
    return r;
}

// Any of names(): the scalar presets, mosqp, boostdms or a single-objective solver.
namespace detail {

inline Result run_named(const std::string& name, const Problem& P, double eq_tol, bool has_tol0,
                        double eq_tol0, int pop_size, long long budget, unsigned seed, const Knobs& knobs,
                        Observer observer, int record_every, const Start& start);

}   // namespace detail

// The run's own counts of the box (Result::repairs): what the thread's
// counter gained during the run. The counter only grows, so a run inside a
// run (a strategy's start on this thread) adds to the outer one, and a
// caller may read it between evaluations (job_so.hpp's trajectory); a
// strategy on several threads adds its starts' counts to it (strategy.hpp,
// round). No method reads it: the run's points are the same with it or
// without.
inline Result run(const std::string& name, const Problem& P, double eq_tol, bool has_tol0,
                  double eq_tol0, int pop_size, long long budget, unsigned seed, const Knobs& knobs,
                  Observer observer, int record_every, const Start& start)
{
    const ops::RepairCounts before = ops::repair_counts();
    Result r = detail::run_named(name, P, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, knobs,
                                 std::move(observer), record_every, start);
    r.repairs = ops::repair_counts() - before;
    return r;
}

inline Result detail::run_named(const std::string& name, const Problem& P, double eq_tol, bool has_tol0,
                                double eq_tol0, int pop_size, long long budget, unsigned seed,
                                const Knobs& knobs, Observer observer, int record_every, const Start& start)
{
    if (detail::listed(local_names(), name))
        return run_local(name, P, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, knobs,
                         std::move(observer), record_every, start);
    if (detail::listed(single_names(), name))
        return run_single(name, P, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, knobs,
                          std::move(observer), record_every, start);
    if (detail::listed(population_names(), name))
        return run_population(name, P, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, knobs,
                              std::move(observer), record_every, start);
    if (detail::listed(strategy_names(), name))
        return run_strategy(name, P, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, knobs,
                            std::move(observer), record_every, start);
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

#include "strategy.hpp"
