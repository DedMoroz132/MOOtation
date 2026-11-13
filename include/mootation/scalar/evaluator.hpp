#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The one road to the problem for the scalarization methods (scalar.hpp,
// mosqp.hpp, boostdms.hpp, polish.hpp): every evaluation is counted against
// the budget, kept in the archive and reported to the observer.
//
// THE BUDGET. A call of the problem is an evaluation, whoever asks for it: a
// solver's iterate, a finite-difference step, a trial point of a line search,
// a point of the initial plan. A point asked for twice is evaluated once (the
// archive keeps it by its bytes), so the number of calls is the number
// counted. Asking for a point past the budget throws BudgetSpent, and past a
// subproblem's allowance SubBudgetSpent: the solver stops there, and what it
// found is in the archive.
//
// THE ARCHIVE holds every evaluation: x, f, the constraint values and whether
// it failed. A point is feasible when every inequality is <= 0 and every
// equality within the tolerance of the moment (|h| <= eq_tol), and its
// violation cv is the vault's (data_vault.hpp): sum max(g, 0) + sum max(|h| -
// eq_tol, 0). A failed evaluation (a value not finite: mootation.failures's
// NaN) is in the archive, marked, and never in the answer.
//
// THE ANSWER, as every algorithm's, is pop_size points: the feasible
// non-dominated points of the archive, reduced by DSS (dss.hpp) when there are
// more; without a feasible point, the least violating ones. The observer gets
// it after the initial plan (gen 0), then every record_every * pop_size
// evaluations and at the end, with the variables beside the objectives. At
// one objective the answer is the best point, and every improvement of it is
// a record (improvements.hpp; SO-4 ... SO-6).
//
// WHAT ELSE A PROBLEM MAY GIVE (task 18, stage 1, §1.1; addendum 1, §3-4),
// all of it optional and in the problem's own coordinates:
//   * gradient: the objective's gradient (and the constraints' Jacobian);
//     value_grad: values and gradient in one call;
//   * residuals r(x) of n_res values, the objective then f = 1/2 ||r~||^2
//     with r~ the residuals of block b times sqrt(w_b) (blocks: their sizes;
//     weights: one each; none: one block of weight 1), and their Jacobian
//     (jacobian, or residuals_jac in one call with r).
// The bounds may be -inf / +inf, coordinate by coordinate.
//
// THE TARGET (task 18, stage 3, §3.6): with Problem::stop_at, the first
// feasible point of one objective at or below it ends the run as the budget
// would: the evaluator keeps it, records it, and its budget becomes what it
// has spent (BudgetSpent), so that nothing evaluates after it — a strategy's
// rounds and restarts included (as COCO's runs end at the final target).
//
// THE SUCCESS (task 24, §24.1.5; off by default): with Problem::
// accept_success, the target is the user's measure of success instead of
// stop_at: a feasible point of one objective with f <= post_at is handed to
// accept_success(x), and the first it accepts ends the run as THE TARGET
// does (stop_at is then not looked at). Each call costs success_cost of the
// budget (declared; 0 by default: the call spends nothing, as a hook that
// does not evaluate f) and is counted apart (success_calls, success_spent;
// success_at: the evaluations spent when it accepted, -1 where it did not).
// READINGS (all of them off by default: with accept_success only):
//   SUCC-1. Off by default. A point is handed over once: where it is met
//     again it is not evaluated again (the archive), and neither is its
//     verdict asked again.
//   SUCC-2. Off by default. The cost is charged after the call, so a call
//     may take the spending past the budget; the run then ends at its next
//     ask (BudgetSpent), as after a gradient's cost.
//   SUCC-3. Off by default. In a strategy's start (strategy.hpp, View) the
//     start's own run asks, its View charges the cost to the start's
//     allowance and keeps the verdict, and the run's evaluator takes that
//     verdict when the round merges, counting the call and its cost then,
//     once; a point the start did not hand over (infeasible at its tolerance
//     of the moment) the run's evaluator hands over itself at the merge, in
//     the merge's order. A start merged after the round's target has its
//     calls uncounted, as its evaluations are (the run ended before them),
//     though the hook was called. The hook may be called from a strategy's
//     threads: it is const and without state, as the problem is.
//   SUCC-4. Off by default. An exception of the hook ends the run with its
//     message and the number of the evaluation (UO-1).
//
// THE CACHE (task 24, §24.1.4; Problem::cache, off by default):
//   CACHE-1. Off by default. exact: the archive's own rule (a point asked for
//     again is not evaluated), its repeats counted in cache_hits.
//   CACHE-2. Off by default. canon: a point whose structure (canon_key_of: the
//     problem's canon_key, else the bytes of canon(x)) was evaluated in this run
//     is not evaluated: it is kept with that structure's values, free, and
//     counted in cache_hits; a gradient asked for it is computed (it is x's, not
//     the structure's representative's). The lean mode keeps the structures'
//     values too.
//   CACHE-3. Off by default. A structure's values are its first point's, in
//     the order the run met them.
//   CACHE-4. Off by default. A strategy's start (strategy.hpp, View) reads the
//     run's cache as it was at the round's start and keeps its own; its hits go
//     to the run at the merge, its points as known ones (free).
//   CACHE-5. Off by default (with the cache). As many points in a row as the
//     budget, each a cache hit (CACHE-1, CACHE-2) and none evaluated between,
//     end the run as its budget's end would (Result::cache_ended): on a finite
//     space of structures a run that has met them all asks only for repeats,
//     spends nothing and would not end (as STRAT-8, SRCH-9, SRCH-10, RST-8).
//     An evaluation starts the count again, so a run that still finds new
//     structures goes on; it makes at most budget x (budget + 1) points. The
//     point that ends the run is not kept, nor counted in cache_hits.
//   CACHE-6. Off by default (with the cache). The run's cache hits in all are
//     at most Problem::cache_hit_cap (100) times the budget: the next one
//     ends the run as its budget's end would (Result::cache_capped). CACHE-5
//     alone bounds a run by budget x (budget + 1) points, 1e10 at a budget of
//     1e5; task 27's GA on its grammar met about 12 hits an evaluation.
// THE DELTA (task 24, §24.1.6): Evaluator::delta(x, block, values) is the
// problem's delta, its declared delta_cost charged after the call
// (DELTA-1, as SUCC-2), counted in delta_calls, delta_spent; a value not
// finite ends the run (DELTA-2); without the hook the two points are
// evaluated (counted, archived).
//
// THE LEAN MODE (task 18, group 3g; lean = true, set before the first
// evaluation, one objective only): no archive. The evaluator keeps two
// points, the best so far (index 0) and the last (index 1, overwritten by the
// next), and looks no point up: a point asked for again is evaluated again
// and counted. The answer is the best point; the counts, the records of the
// best, the observer and THE TARGET are as without it. For long runs of
// methods that never ask for a point twice (ipop, bipop: keep_points = best),
// whose archive of every point would not fit in memory (1e6 D evaluations).
// keep_points = best (task 24, §24.3.3) asks for it in the single and local
// solvers (SINGLE-5, LOCAL-7), the population methods, multistart, ils,
// basin_hopping, mlsl, tiktak and local_search too (a gradient is kept by
// slot, and goes with its point); halving
// and memetic refuse it (their reruns are free only through the archive,
// SRCH-6, SRCH-7). Without it, a run of one objective whose archive would
// pass kArchiveWarnBytes (budget times the bytes of a point) warns once
// through mootation::warn, before its first point.
//
// THE COUNTS (addendum 1, §4). evaluations: the calls of the problem's
// values (evaluate, residuals); grad_calls: the calls of a gradient or a
// Jacobian; a joint call is one of each. What the budget is spent by is
// evaluations + grad_cost * grad_calls (grad_cost a knob, 1 by default);
// finite differences are evaluations. Without a gradient both are the one
// count of before, bit for bit.
//
// READINGS:
//   SO-1. A problem with residuals takes its objective from them: evaluate,
//     if it is given too, is not called, and such a problem has no
//     constraints (n_cons = 0; the least-squares solvers of this library
//     take none). The archive keeps r~ beside f, so that a solver asking
//     for a point again gets its residuals without a call.
//   SO-2. A gradient asked for at a point not yet evaluated is a joint call
//     where the problem has one (value_grad, residuals_jac), else the values
//     and then the gradient: either way one evaluation and one grad_call. A
//     gradient asked for twice at the same point is computed once.
//   SO-3. A problem that gives only the joint call (value_grad without
//     evaluate, residuals_jac without residuals) is evaluated by it every
//     time: every evaluation is then a grad_call too.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../dss.hpp"
#include "../fp_contract.hpp"
#include "../warn.hpp"
#include "genome.hpp"
#include "periodic.hpp"
#include "improvements.hpp"
#include "symmetry.hpp"

namespace mootation::scalar {

struct BudgetSpent {};       // the run's evaluations are spent
struct SubBudgetSpent {};    // the current subproblem's allowance is spent
struct FailedPoint {};       // sqp met a failed evaluation (solvers.hpp)

using Rows = std::vector<std::vector<double>>;

struct Problem;

// THE USER'S LOCAL SEARCH (task 24, §24.1.1; scalar.hpp, run_user_local):
// what Problem::local_search is handed — the problem through the run's
// evaluator (every call counted against the budget, kept in the archive), the
// budget it may spend, the run's generator (draw with rnd:: only), and the
// variables it may move (empty: all) — and what it gives back: x', f(x')
// where it knows it (x' is then not evaluated again), the budget it spent
// besides the problem's calls (declared), and its state (a vector of the
// hook's own meaning: lambda and scales of LM, multipliers of ALM, a step;
// handed back at the next search of the same start where a strategy keeps it).
struct LocalContext {
    const Problem& problem;
    long long budget;
    std::mt19937& rng;
    std::vector<char> mask;
};

struct LocalOutcome {
    std::vector<double> x;
    bool has_f = false;
    double f = 0.0;
    double cost = 0.0;
    std::vector<double> state;
};

// The problem as the methods see it: n = lo.size() variables in [lo, hi], m
// objectives, n_cons constraint values (the inequalities, g <= 0, then the
// n_eq equalities, h = 0). evaluate fills f (m values) and g (n_cons); a value
// that is not finite marks the evaluation failed.
struct Problem {
    using Vec = std::vector<double>;
    std::vector<double> lo, hi;
    int m = 0, n_cons = 0, n_eq = 0;
    std::function<void(const Vec& x, Vec& f, Vec& g)> evaluate;
    // the optional parts (WHAT ELSE A PROBLEM MAY GIVE): gf the objective's
    // gradient (n values, one objective), JG the constraints' Jacobian
    // (n_cons rows of n; left empty where there are no constraints)
    std::function<void(const Vec& x, Vec& gf, Rows& JG)> gradient;
    std::function<void(const Vec& x, Vec& f, Vec& g, Vec& gf, Rows& JG)> value_grad;
    int n_res = 0;                    // 0: as many as the first call gives
    std::function<void(const Vec& x, Vec& r)> residuals;
    std::function<void(const Vec& x, Rows& J)> jacobian;            // n_res rows of n
    std::function<void(const Vec& x, Vec& r, Rows& J)> residuals_jac;
    std::vector<int> blocks;          // the residual blocks' sizes (sum n_res); empty: one block
    std::vector<double> weights;      // a weight per block; empty: 1
    // the directions along which the residuals do not change to first order,
    // n rows of k (task 19, §19.3.1; lm with lm_gauge = on)
    std::function<void(const Vec& x, Rows& N)> gauge_basis;
    // one objective: the run ends at the first feasible point with f <= stop_at
    // (THE TARGET; -inf: never)
    double stop_at = -std::numeric_limits<double>::infinity();
    // symmetries (task 19, §19.2.1; symmetry.hpp): groups of interchangeable
    // blocks of coordinates, the problem's canonical representative (f(canon(x))
    // = f(x), canon(canon(x)) = canon(x)) and its alignment of x to ref; empty:
    // the built-in ones on sym_blocks (the kernels' sym knob)
    std::vector<symmetry::Blocks> sym_blocks;
    std::function<void(Vec& x)> canon;
    std::function<void(const Vec& x, const Vec& ref, Vec& out)> align;
    // THE SUCCESS (task 24, §24.1.5): the user's measure of success, asked
    // at the feasible points with f <= post_at; empty: stop_at's target
    std::function<bool(const Vec& x)> accept_success;
    double post_at = std::numeric_limits<double>::infinity();
    double success_cost = 0.0;        // the budget one call spends (declared)
    // THE USER'S LOCAL SEARCH (task 24, §24.1.1): the solver "local_search"
    // (scalar.hpp) calls it once from its start: (x, ctx, state) -> outcome
    std::function<LocalOutcome(const Vec& x, LocalContext& ctx, const Vec& state)> local_search;
    // THE USER'S GENOME (task 24, §24.1.2; ga.hpp GA-9 ... GA-11): operators
    // on the genome of fixed length for ga_sbx and ssga (ga_crossover,
    // ga_mutation, ga_repair, ga_init = user), and its decoding, written back
    std::function<void(const Vec& p1, const Vec& p2, Vec& c1, Vec& c2, std::mt19937& rng)> crossover;
    std::function<void(Vec& x, std::mt19937& rng)> mutate;
    std::function<void(Vec& x)> repair;
    std::function<Decoded(const Vec& x)> decode;
    std::function<void(Vec& x, std::mt19937& rng)> sample;     // x arrives with n values
    // PERIODIC VARIABLES (task 24, §24.4.2; periodic.hpp): 1 where variable j
    // is periodic, its period hi - lo; empty: none
    std::vector<char> periodic;
    // a variable's own repair (task 24, §24.4.2, "bounded(method)"): -1 the
    // method's, else an ops::BoundRepair (operators/bound_repair.hpp) for
    // de, jade, shade, lshade, g3_pcx, pso, spso2011; empty: the method's
    std::vector<int> var_repair;
    // set by a strategy's start (strategy.hpp, View), not by the user: a
    // point whose values are known, for the run's archive at the merge (ULS-2)
    std::function<void(const Vec& x, const Vec& f)> known_point;
    // THE MOVE, THE DISTANCE, THE SHARES (task 24, §24.1.3; hooks.hpp): a
    // move of x in place (strength: the strategy's step, its meaning the
    // hook's), a dissimilarity (>= 0, symmetric; default distance_of), each
    // element's share of f (its length the hook's). The strategies that use
    // them are tasks 25 and 26's.
    std::function<void(Vec& x, double strength, std::mt19937& rng)> perturb;
    std::function<double(const Vec& x, const Vec& y)> distance;
    std::function<void(const Vec& x, Vec& c)> contributions;
    // THE CACHE (task 24, §24.1.4; CACHE-1 ... CACHE-4): the bytes of x's
    // canonical structure (default canon_key_of), and the run's cache by it
    std::function<std::string(const Vec& x)> canon_key;
    enum class Cache { off, exact, canon };
    Cache cache = Cache::off;
    double cache_hit_cap = 100.0;     // CACHE-6: the run's cache hits, at most this times the budget
    // THE SUBPROBLEM AND THE DELTA (task 24, §24.1.6; hooks.hpp): the user's
    // subproblem on the variables idx, the others frozen at x_frozen
    // (default restricted()); f(x with x[block] = values) - f(x), at the
    // declared cost delta_cost (Evaluator::delta); the variables each one
    // interacts with
    std::function<Problem(const Vec& x_frozen, const std::vector<std::size_t>& idx)> restrict_to;
    std::function<double(const Vec& x, const std::vector<std::size_t>& block, const Vec& values)> delta;
    double delta_cost = 1.0;
    std::function<std::vector<std::vector<std::size_t>>(const Vec& x)> neighbors;
    // MIXED VARIABLES (task 25, §24.3.4): each variable's values where the
    // caller's f takes it rounded to one of them (minimize's var_types: an
    // integer, a discrete value, a bit), sorted and in the box; empty, and
    // empty lists: continuous. The methods search the reals as before; cmaes
    // with cma_margin = on takes them as HM22's discrete values (scalar.hpp
    // SINGLE-6).
    std::vector<Vec> discrete;
    // SEPARABLE LEAST SQUARES (task 25, §24.3.5): the variables the residuals
    // are linear in, r(x) = y - Phi(alpha) a with a = x[linear] and y not
    // depending on alpha (suite::Native's linear; scalar.hpp LOCAL-9, lm's
    // lm_varpro); empty: none.
    std::vector<int> linear;

    bool has_residuals() const { return residuals || residuals_jac; }
    bool has_gradient() const { return gradient || value_grad || jacobian || residuals_jac; }
};

// THE CACHE's key (task 24, §24.1.4): the problem's canon_key, else the bytes
// of canon(x) — the problem's canon, else its blocks sorted (symmetry.hpp
// SYM-5) — else the bytes of x
inline std::string canon_key_of(const Problem& P, const std::vector<double>& x)
{
    if (P.canon_key) return P.canon_key(x);
    std::vector<double> y = x;
    if (P.canon) P.canon(y);
    else if (!P.sym_blocks.empty()) symmetry::sort_blocks(y, P.sym_blocks);
    std::string k(y.size() * sizeof(double), '\0');
    if (!y.empty()) std::memcpy(&k[0], y.data(), k.size());
    return k;
}

// A point whose values are known (a seed population, an algorithm's evaluations).
struct Point {
    std::vector<double> x, f, g;
};

// observer(gen, F, X): the answer's objectives and variables.
using Observer = std::function<void(long long gen, const Rows& F, const Rows& X)>;

class Evaluator {
public:
    using Vec = std::vector<double>;
    const Problem& P;
    std::vector<double> lo, hi;
    int m = 0, n_cons = 0, n_eq = 0;
    double eq_tol_end = 0.0;
    bool has_tol0 = false;
    double eq_tol0 = 0.0;
    long long budget = 0;
    int pop_size = 1;
    Observer observer;
    long long stride = 1, next_record = 1;
    Rows X, F, G;
    Rows R;                            // r~ of every point (residual problems; else empty)
    std::vector<char> failed;
    long long spent = 0;               // evaluations
    long long grad_calls = 0;
    double grad_cost = 1.0;
    int n_res = 0;                     // the residuals' count (the problem's, or its first call's)
    long long limit = 0;               // the current subproblem's last evaluation
    std::vector<std::size_t> nd;       // feasible non-dominated archive indices
    Rows ndF;                          // ... and their objectives
    bool nd_set = false;               // nd was judged at nd_tol
    double nd_tol = 0.0;
    bool over = false;                 // the run has ended: the final tolerance
    bool recorded = false;             // gen 0 has gone to the observer
    long long gen0 = 0;                // gens before this evaluator (polish.hpp)
    bool lean = false;                 // THE LEAN MODE: the best and the last point only
    static constexpr double kArchiveWarnBytes = 2147483648.0;     // 2 GiB (task 24, §24.3.3)
    ImprovementLog improvements;       // m = 1: the best point's records (improvements.hpp)
    long long success_calls = 0;       // THE SUCCESS: accept_success's calls,
    double success_spent = 0.0;        // ... their cost in the budget,
    long long success_at = -1;         // ... and the evaluations spent when it accepted
    long long hook_calls = 0;          // THE USER'S LOCAL SEARCH: its calls,
    double hook_spent = 0.0;           // ... the budget it declared spent,
    double hook_seconds = 0.0;         // ... and its time (reported, never in a trajectory)
    long long decode_invalid = 0;      // THE USER'S GENOME: genomes that did not decode, not evaluated
    long long cache_hits = 0;          // THE CACHE: points not evaluated, their structure's values taken
    bool cache_ended = false;          // ... the run ended by CACHE-5
    bool cache_capped = false;         // ... or by CACHE-6
    long long delta_calls = 0;         // THE DELTA: the hook's calls,
    double delta_spent = 0.0;          // ... their declared cost in the budget

    Evaluator(const Problem& p, double eq_tol, bool has_eq_tol0, double eq_tol0_, long long budget_,
              int pop_size_, Observer obs, int record_every)
        : P(p), lo(p.lo), hi(p.hi), m(p.m), n_cons(p.n_cons), n_eq(p.n_eq), eq_tol_end(eq_tol),
          has_tol0(has_eq_tol0), eq_tol0(eq_tol0_), budget(budget_), pop_size(pop_size_), n_res(p.n_res)
    {
        if (record_every > 0 && obs) observer = std::move(obs);
        stride = static_cast<long long>(std::max(1, record_every)) * pop_size;
        next_record = stride;
        limit = budget;
        if (p.has_residuals()) check_residual_problem();
        if (!p.var_repair.empty()) {                                // a variable's own repair
            if (p.var_repair.size() != p.lo.size())
                throw std::invalid_argument("var_repair: " + std::to_string(p.var_repair.size()) +
                                            " entries for " + std::to_string(p.lo.size()) + " variables");
            for (std::size_t j = 0; j < p.var_repair.size(); ++j)
                if (p.var_repair[j] < -1 || p.var_repair[j] > 7 || p.var_repair[j] == 4)
                    throw std::invalid_argument("var_repair: variable " + std::to_string(j) +
                                                ": -1 or a bound repair but resample (0 ... 7, not 4)");
        }
        if (p.cache != Problem::Cache::off && !(p.cache_hit_cap > 0.0 && std::isfinite(p.cache_hit_cap)))
            throw std::invalid_argument("cache_hit_cap must be a number > 0 (CACHE-6)");
        if (!p.periodic.empty()) {                                  // periodic.hpp
            if (p.periodic.size() != p.lo.size())
                throw std::invalid_argument("periodic: " + std::to_string(p.periodic.size()) + " flags for " +
                                            std::to_string(p.lo.size()) + " variables");
            for (std::size_t j = 0; j < p.lo.size(); ++j)
                if (p.periodic[j] && !(std::isfinite(p.lo[j]) && std::isfinite(p.hi[j]) && p.lo[j] < p.hi[j]))
                    throw std::invalid_argument("periodic: variable " + std::to_string(j) +
                                                " needs a finite box of positive width, its period");
        }
    }

    // ── evaluating ──────────────────────────────────────────────────────────
    // The archive index of x, evaluating it if it is new.
    std::size_t index(const std::vector<double>& x)
    {
        if (lean) {                                                             // THE LEAN MODE: no lookup
            if (const Cached* c = cached(x)) return from_cache(x, *c, std::string());
            return fresh(x, std::string(), false);
        }
        const std::string key = key_of(x);
        auto it = where_.find(key);
        if (it != where_.end()) {
            if (P.cache == Problem::Cache::exact) free_point();                // CACHE-1, CACHE-5, CACHE-6
            return it->second;
        }
        if (const Cached* c = cached(x)) return from_cache(x, *c, key);
        return fresh(x, key, false);
    }

    // THE CACHE (task 24, §24.1.4): a structure's values, by canon_key_of
    struct Cached {
        Vec f, g, r;
        bool bad = false;
    };

    // the values of x's structure, where the cache (canon) has them
    const Cached* cached(const std::vector<double>& x) const
    {
        if (P.cache != Problem::Cache::canon) return nullptr;
        const auto it = canon_.find(canon_key_of(P, x));
        return it == canon_.end() ? nullptr : &it->second;
    }

    // x kept with its structure's values, not evaluated (CACHE-2)
    std::size_t from_cache(const std::vector<double>& x, const Cached& c, const std::string& key)
    {
        free_point();                                                          // CACHE-5, CACHE-6
        r_ = c.r;
        return keep(x, c.f, c.g, c.bad, key, false);
    }

    // a point the cache gives free, counted in cache_hits; it ends the run as
    // its budget's end would where it is the budget's worth in a row (CACHE-5)
    // or past cache_hit_cap x budget in all (CACHE-6), and is then not kept
    void free_point()
    {
        if (static_cast<double>(cache_hits) >= P.cache_hit_cap * static_cast<double>(budget)) {
            cache_capped = true;
            throw BudgetSpent{};
        }
        if (free_run_ + 1 >= budget) {
            cache_ended = true;
            throw BudgetSpent{};
        }
        ++cache_hits;
        ++free_run_;
    }

    // THE DELTA (task 24, §24.1.6): f(x with x[block] = values) - f(x) — the
    // problem's delta, its declared cost charged (DELTA-1), or the two points
    // through this evaluator
    double delta(const std::vector<double>& x, const std::vector<std::size_t>& block, const std::vector<double>& values)
    {
        if (block.size() != values.size())
            throw std::invalid_argument("delta: " + std::to_string(values.size()) + " values for a block of " +
                                        std::to_string(block.size()));
        for (std::size_t j : block)
            if (j >= x.size()) throw std::invalid_argument("delta: variable " + std::to_string(j) + " past n");
        if (P.delta) {
            ++delta_calls;
            delta_spent += P.delta_cost;
            const double d = P.delta(x, block, values);
            if (!std::isfinite(d)) throw std::runtime_error("delta: the hook gave a value not finite (DELTA-2)");
            return d;
        }
        std::vector<double> y = x;
        for (std::size_t k = 0; k < block.size(); ++k) y[block[k]] = values[k];
        const double fx = F[index(x)][0];                                     // read before y's evaluation
        return F[index(y)][0] - fx;
    }

    // What the budget is spent by: evaluations + grad_cost * grad_calls.
    double used() const
    {
        return static_cast<double>(spent) + grad_cost * static_cast<double>(grad_calls) + success_spent + hook_spent +
               delta_spent;
    }

    // ── gradients (SO-2) ────────────────────────────────────────────────────
    // The archive index of x with its gradient known: the objective's (grad),
    // and the constraints' Jacobian (jac_g) or the residuals' (jac).
    std::size_t index_grad(const std::vector<double>& x)
    {
        if (!P.has_gradient()) throw std::logic_error("index_grad: the problem gives no gradient");
        if (lean) {                                   // THE LEAN MODE: its two slots, a gradient kept by slot
            std::size_t i = X.size();
            for (std::size_t s = 0; s < X.size(); ++s)
                if (X[s] == x && (i == X.size() || (gf_.count(s) && !gf_.count(i)))) i = s;
            if (i == X.size()) i = fresh(x, std::string(), true);
            return with_gradient(x, i);
        }
        const std::string key = key_of(x);
        auto it = where_.find(key);
        const std::size_t i = it != where_.end() ? it->second : fresh(x, key, true);
        return with_gradient(x, i);
    }

    // point i (x) with its gradient: computed where it is not known (SO-2)
    std::size_t with_gradient(const std::vector<double>& x, std::size_t i)
    {
        if (!gf_.count(i)) {
            charge(grad_cost);
            ++grad_calls;
            Rows J;
            Vec gf;
            if (P.has_residuals()) {
                if (P.jacobian) {
                    P.jacobian(x, J);
                } else {
                    Vec r;
                    P.residuals_jac(x, r, J);
                }
                scale_rows(J);
            } else if (P.gradient) {
                P.gradient(x, gf, J);
            } else {
                Vec f(static_cast<std::size_t>(m)), g(static_cast<std::size_t>(n_cons));
                P.value_grad(x, f, g, gf, J);
            }
            put_grad(i, gf, J);
        }
        return i;
    }

    const Vec& grad(std::size_t i) const { return gf_.at(i); }
    const Rows& jac(std::size_t i) const { return jac_.at(i); }       // residuals: r~'s Jacobian
    const Rows& jac_g(std::size_t i) const { return jac_.at(i); }     // otherwise: the constraints'

    // A point whose values are known (a seed population): archived, not counted.
    std::size_t plant(const std::vector<double>& x, const std::vector<double>& f,
                      const std::vector<double>& g)
    {
        const std::string key = key_of(x);
        auto it = where_.find(key);
        if (it != where_.end()) return it->second;
        bool bad = false;
        for (double v : f) bad = bad || !std::isfinite(v);
        for (double v : g) bad = bad || !std::isfinite(v);
        return keep(x, f, g, bad, key, false);
    }

    bool known(const std::vector<double>& x) const { return where_.count(key_of(x)) != 0; }

    std::size_t find(const std::vector<double>& x) const { return where_.at(key_of(x)); }

    bool has_grad(std::size_t i) const { return gf_.count(i) != 0; }

    // ── a strategy's round (strategy.hpp) ───────────────────────────────────
    // A start's point merged at the end of its round, in the order of the
    // starts: f, g and r~ as the start found them, `evals` (0 or 1) the
    // evaluation and `grads` (0 or 1) the gradient call it made for it, gf and
    // J the gradient where it has one. What an earlier start of the round
    // brought already costs nothing again (the round's rule: a point two
    // starts evaluated counts for the first).
    // verdict: accept_success's answer at x where the start asked it (SUCC-3;
    // -1: not asked)
    std::size_t adopt(const Vec& x, const Vec& f, const Vec& g, const Vec& r, bool bad, int evals, int grads,
                      const Vec* gf, const Rows* J, int verdict = -1)
    {
        const std::string key = key_of(x);
        auto it = where_.find(key);
        std::size_t i;
        const bool with_grad = grads > 0 && gf != nullptr && J != nullptr;
        if (it != where_.end()) {
            i = it->second;
            if (with_grad && !gf_.count(i)) {
                grad_calls += grads;
                gf_[i] = *gf;
                jac_[i] = *J;
            }
            return i;
        }
        if (n_res == 0) n_res = static_cast<int>(r.size());
        grad_calls += grads;                      // before keep(), as fresh() counts a joint call
        r_ = r;
        i = keep(x, f, g, bad, key, evals > 0, verdict);
        if (with_grad) {
            gf_[i] = *gf;
            jac_[i] = *J;
        }
        return i;
    }

    // ── feasibility ─────────────────────────────────────────────────────────
    // The equalities' tolerance of the moment (eq_tol_schedule = linear: from
    // eq_tol0 to eq_tol in proportion to the evaluations spent; eq_tol itself
    // once the run is over, for the answer).
    double eq_tol() const
    {
        if (!has_tol0 || budget <= 0 || over) return eq_tol_end;
        const double share = std::min(1.0, static_cast<double>(spent) / static_cast<double>(budget));
        return eq_tol0 + (eq_tol_end - eq_tol0) * share;
    }

    double violation(const std::vector<double>& g, double tol) const
    {
        if (g.empty()) return 0.0;
        const std::size_t k = g.size() - static_cast<std::size_t>(n_eq);
        double v = 0.0;
        for (std::size_t j = 0; j < k; ++j) v += std::max(g[j], 0.0);
        double ve = 0.0;
        for (std::size_t j = k; j < g.size(); ++j) ve += std::max(std::fabs(g[j]) - tol, 0.0);
        return n_eq ? v + ve : v;
    }

    double cv(std::size_t i) const { return cv(i, eq_tol()); }

    double cv(std::size_t i, double tol) const
    {
        return failed[i] ? std::numeric_limits<double>::infinity() : violation(G[i], tol);
    }

    // ── the answer ──────────────────────────────────────────────────────────
    // Archive indices of the answer: up to k (pop_size) feasible
    // non-dominated points by DSS, or the least violating ones.
    std::vector<std::size_t> answer(long long k = -1)
    {
        if (lean) return X.empty() ? std::vector<std::size_t>{} : std::vector<std::size_t>{0};   // THE LEAN MODE
        const std::size_t kk = static_cast<std::size_t>(k < 0 ? pop_size : k);
        if (!nd_set || nd_tol != eq_tol()) rebuild();
        if (!nd.empty()) {
            if (nd.size() <= kk) return nd;
            Rows Fn;
            for (std::size_t i : nd) Fn.push_back(F[i]);
            std::vector<std::size_t> order = dss::order(Fn, kk);
            std::vector<std::size_t> out;
            for (std::size_t j : order) out.push_back(nd[j]);
            return out;
        }
        std::vector<std::size_t> ok;
        for (std::size_t i = 0; i < X.size(); ++i)
            if (!failed[i]) ok.push_back(i);
        std::vector<double> c(X.size(), 0.0);
        for (std::size_t i : ok) c[i] = cv(i);
        std::stable_sort(ok.begin(), ok.end(), [&](std::size_t a, std::size_t b) {
            return c[a] < c[b] || (c[a] == c[b] && a < b);
        });
        if (ok.size() > kk) ok.resize(kk);
        return ok;
    }

    // The answer to the observer: gen 0 the first time (after the initial
    // plan), then the evaluations spent in units of pop_size, rounded up at
    // the end.
    void record(bool final_ = false)
    {
        if (!observer) return;
        std::vector<std::size_t> idx = answer();
        Rows Fo, Xo;
        for (std::size_t i : idx) {
            Fo.push_back(F[i]);
            Xo.push_back(X[i]);
        }
        const long long p = pop_size;
        const long long gen = !recorded ? 0 : gen0 + (final_ ? (spent + p - 1) / p : spent / p);
        recorded = true;
        observer(gen, Fo, Xo);
    }

private:
    std::unordered_map<std::string, std::size_t> where_;   // x's bytes -> archive index
    std::unordered_map<std::string, Cached> canon_;          // THE CACHE: a structure's key -> its values
    long long free_run_ = 0;                                 // CACHE-5: cache hits since the last evaluation

    // THE CACHE: the first point of a structure gives it its values (CACHE-3)
    void remember(const std::vector<double>& x, const Vec& f, const Vec& g, bool bad)
    {
        if (P.cache != Problem::Cache::canon) return;
        canon_.emplace(canon_key_of(P, x), Cached{f, g, r_, bad});
    }

public:
    // a structure's values in this run's cache, or nullptr (a strategy's start reads them: strategy.hpp)
    const Cached* cached_key(const std::string& k) const
    {
        const auto it = canon_.find(k);
        return it == canon_.end() ? nullptr : &it->second;
    }

private:
    std::unordered_map<std::size_t, Vec> gf_;              // archive index -> the objective's gradient
    std::unordered_map<std::size_t, Rows> jac_;            // ... -> r~'s or the constraints' Jacobian
    Vec r_;                                                // r~ of the point being kept

    static std::string key_of(const std::vector<double>& x)
    {
        std::string k(x.size() * sizeof(double), '\0');
        if (!x.empty()) std::memcpy(&k[0], x.data(), k.size());
        return k;
    }

    // Throws when cost more would spend past the budget or the subproblem's
    // allowance (for one evaluation and no gradient: spent >= budget, as before).
    void charge(double cost) const
    {
        const double after = used() + cost;
        if (after > static_cast<double>(budget)) throw BudgetSpent{};
        if (after > static_cast<double>(limit)) throw SubBudgetSpent{};
    }

    void check_residual_problem() const
    {
        if (m != 1 || n_cons != 0)
            throw std::invalid_argument("a problem with residuals has one objective and no constraints (SO-1)");
        if (P.n_res < 0) throw std::invalid_argument("a problem with residuals needs n_res >= 0");
        if (P.n_res == 0 && !P.blocks.empty())
            throw std::invalid_argument("residual blocks need n_res, the residuals' count");
        if (!P.blocks.empty()) {
            long long sum = 0;
            for (int b : P.blocks) {
                if (b <= 0) throw std::invalid_argument("residual blocks: every size must be >= 1");
                sum += b;
            }
            if (sum != P.n_res) throw std::invalid_argument("residual blocks: the sizes must add up to n_res");
            if (P.weights.size() != P.blocks.size())
                throw std::invalid_argument("residual blocks: one weight per block");
        } else if (P.weights.size() > 1) {
            throw std::invalid_argument("residual weights: one per block, and there is one block");
        }
        for (double w : P.weights)
            if (!(w >= 0.0) || !std::isfinite(w)) throw std::invalid_argument("residual weights must be finite and >= 0");
    }

    // block b's rows times sqrt(w_b) (SO-1)
    template <typename RowOp>
    void by_block(RowOp op) const
    {
        if (P.weights.empty()) return;
        std::size_t at = 0;
        for (std::size_t b = 0; b < P.weights.size(); ++b) {
            const std::size_t size = P.blocks.empty() ? static_cast<std::size_t>(n_res)
                                                      : static_cast<std::size_t>(P.blocks[b]);
            const double s = std::sqrt(P.weights[b]);
            for (std::size_t k = 0; k < size; ++k) op(at + k, s);
            at += size;
        }
    }

    void scale_rows(Rows& J) const
    {
        by_block([&](std::size_t i, double s) {
            for (double& v : J[i]) v *= s;
        });
    }

    void put_grad(std::size_t i, Vec gf, Rows J)
    {
        if (P.has_residuals()) {                      // grad f = J~^T r~
            const Vec& r = R[i];
            gf.assign(lo.size(), 0.0);
            for (std::size_t k = 0; k < r.size(); ++k)
                for (std::size_t j = 0; j < gf.size(); ++j) gf[j] += J[k][j] * r[k];
        }
        gf_[i] = std::move(gf);
        jac_[i] = std::move(J);
    }

    // x evaluated for the first time; with_grad: the gradient is wanted too,
    // by the joint call where there is one (SO-2, SO-3)
    std::size_t fresh(const Vec& x, const std::string& key, bool with_grad)
    {
        const bool res = P.has_residuals();
        const bool joint = res ? (P.residuals_jac && (with_grad || !P.residuals))
                               : (P.value_grad && (with_grad || !P.evaluate));
        charge(joint ? 1.0 + grad_cost : 1.0);
        free_run_ = 0;                                                         // CACHE-5
        Vec f(static_cast<std::size_t>(m), 0.0), g(static_cast<std::size_t>(n_cons), 0.0), gf;
        Rows J;
        r_.clear();
        if (res) {
            if (joint) P.residuals_jac(x, r_, J);
            else P.residuals(x, r_);
            if (n_res == 0) n_res = static_cast<int>(r_.size());
            if (r_.empty() || r_.size() != static_cast<std::size_t>(n_res))
                throw std::invalid_argument("residuals gave " + std::to_string(r_.size()) + " values for n_res = " +
                                            std::to_string(n_res));
            by_block([&](std::size_t k, double s) { r_[k] *= s; });
            double sum = 0.0;
            for (double v : r_) sum += v * v;
            f.assign(1, 0.5 * sum);
        } else if (joint) {
            P.value_grad(x, f, g, gf, J);
        } else {
            P.evaluate(x, f, g);
        }
        bool bad = false;
        for (double v : f) bad = bad || !std::isfinite(v);
        for (double v : g) bad = bad || !std::isfinite(v);
        if (joint) ++grad_calls;       // counted before keep(): the record holds its own call
        const std::size_t i = keep(x, f, g, bad, key, true);
        if (joint) {
            if (res) scale_rows(J);
            put_grad(i, std::move(gf), std::move(J));
        }
        return i;
    }

    std::size_t keep(const std::vector<double>& x, const std::vector<double>& f,
                     const std::vector<double>& g, bool bad, const std::string& key, bool count, int verdict = -1)
    {
        if (lean) return keep_lean(x, f, g, bad, count, verdict);
        if (X.empty() && m == 1) warn_archive(x.size());
        const std::size_t i = X.size();
        X.push_back(x);
        F.push_back(f);
        G.push_back(g);
        R.push_back(r_);
        remember(x, f, g, bad);                                                 // THE CACHE
        r_.clear();
        failed.push_back(static_cast<char>(bad));
        where_[key] = i;
        if (count) ++spent;
        if (!bad && nd_set && nd_tol == eq_tol() && cv(i) <= 0.0) offer(i);
        if (m == 1 && !bad) improvements.offer(f[0], cv(i), spent, grad_calls);
        if (observer && count && spent >= next_record) {
            while (next_record <= spent) next_record += stride;
            record();
        }
        if (m == 1 && count && !bad && reached(x, f[0], i, verdict)) {          // THE TARGET
            budget = spent;
            limit = std::min(limit, spent);
            throw BudgetSpent{};
        }
        return i;
    }

    // the archive's size at the budget, warned of where it is large (THE LEAN MODE)
    void warn_archive(std::size_t n) const
    {
        const double per = 16.0 * static_cast<double>(n) + 8.0 * static_cast<double>(m + n_cons + n_res) + 96.0;
        const double bytes = per * static_cast<double>(budget);
        if (!(bytes > kArchiveWarnBytes)) return;
        mootation::warn("scalar: the archive of every point may take about " +
                        std::to_string(static_cast<long long>(bytes / 1073741824.0 + 0.5)) + " GiB (budget " +
                        std::to_string(budget) + ", " + std::to_string(n) +
                        " variables); keep_points = best keeps the best point and the last");
    }

    // THE TARGET at point i (x, f, feasible or not): f <= stop_at, or THE
    // SUCCESS's verdict — the start's (SUCC-3), or asked now
    bool reached(const std::vector<double>& x, double f, std::size_t i, int verdict)
    {
        if (!P.accept_success) return f <= P.stop_at && cv(i) <= 0.0;
        if (!(f <= P.post_at) || cv(i) > 0.0) return false;
        ++success_calls;
        success_spent += P.success_cost;
        bool ok = verdict == 1;
        if (verdict < 0) {
            try {
                ok = P.accept_success(x);
            } catch (const std::exception& e) {
                throw std::runtime_error("accept_success failed at evaluation " + std::to_string(spent) + ": " +
                                         e.what() + " (SUCC-4)");
            }
        }
        if (ok) success_at = spent;
        return ok;
    }

    // THE LEAN MODE's keep: the point into slot 1 (slot 0 for the first), the
    // best of the two into slot 0 (feasible first, then f; Deb)
    std::size_t keep_lean(const std::vector<double>& x, const std::vector<double>& f,
                          const std::vector<double>& g, bool bad, bool count, int verdict = -1)
    {
        if (m != 1) throw std::logic_error("the lean evaluator is for one objective");
        std::size_t i = 0;
        if (X.size() < 2) {
            i = X.size();
            X.push_back(x);
            F.push_back(f);
            G.push_back(g);
            R.push_back(r_);
            failed.push_back(static_cast<char>(bad));
        } else {
            i = 1;
            X[1] = x;
            gf_.erase(1);                             // slot 1's gradient was the point before
            jac_.erase(1);
            F[1] = f;
            G[1] = g;
            R[1] = r_;
            failed[1] = static_cast<char>(bad);
        }
        remember(x, f, g, bad);                                                 // THE CACHE
        r_.clear();
        if (count) ++spent;
        if (!bad) improvements.offer(f[0], cv(i), spent, grad_calls);
        if (i == 1 && !bad) {
            const double c1 = cv(1), c0 = cv(0);
            const bool better = failed[0] || (c1 <= 0.0 && c0 <= 0.0 ? F[1][0] < F[0][0] : c1 < c0);
            if (better) {
                X[0] = X[1];
                if (gf_.count(1)) {                   // the gradient goes with the point
                    gf_[0] = gf_[1];
                    jac_[0] = jac_[1];
                } else {
                    gf_.erase(0);
                    jac_.erase(0);
                }
                F[0] = F[1];
                G[0] = G[1];
                R[0] = R[1];
                failed[0] = 0;
            }
        }
        if (observer && count && spent >= next_record) {
            while (next_record <= spent) next_record += stride;
            record();
        }
        if (count && !bad && reached(x, f[0], i, verdict)) {                    // THE TARGET
            budget = spent;
            limit = std::min(limit, spent);
            throw BudgetSpent{};
        }
        return i;
    }

    void offer(std::size_t i)
    {
        const std::vector<double>& f = F[i];
        if (!nd.empty()) {
            for (const auto& q : ndF) {                    // dominated or equal: not added
                bool weak = true;
                for (int j = 0; j < m; ++j) weak = weak && q[static_cast<std::size_t>(j)] <= f[static_cast<std::size_t>(j)];
                if (weak) return;
            }
            std::vector<std::size_t> keep_i;
            Rows keep_f;
            for (std::size_t a = 0; a < nd.size(); ++a) {
                bool dominated = true;                     // f <= ndF[a] everywhere
                for (int j = 0; j < m; ++j)
                    dominated = dominated && f[static_cast<std::size_t>(j)] <= ndF[a][static_cast<std::size_t>(j)];
                if (!dominated) {
                    keep_i.push_back(nd[a]);
                    keep_f.push_back(ndF[a]);
                }
            }
            nd = keep_i;
            ndF = keep_f;
        }
        nd.push_back(i);
        ndF.push_back(f);
    }

    void rebuild()
    {
        const double tol = eq_tol();
        nd.clear();
        ndF.clear();
        nd_set = true;
        nd_tol = tol;
        for (std::size_t i = 0; i < X.size(); ++i)
            if (!failed[i] && cv(i, tol) <= 0.0) offer(i);
    }
};

}   // namespace mootation::scalar
