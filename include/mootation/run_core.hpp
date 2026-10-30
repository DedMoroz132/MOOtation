#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// One algorithm run, by name — the run the Python package's minimize() makes
// (_core.run), the campaign engine in C++ makes (task 17, stage 5, step 4b)
// and embed.hpp's run() and Session make (task 17, section 2.5): the same
// code, so the same run, bit for bit; the library's one driver of a run. Moved here from
// python/mootation_module.cpp, which now wraps its Python callables and
// calls run() below.
//
// What a run is given (RunProblem): the box, the sizes, and either a problem
// of the C++ suite (native: evaluated here, its rows handed to on_rows a
// batch at a time) or evaluators to call (evaluate, evaluate_batch,
// limits_batch). A caller that has a lock to hold around its evaluators and
// to let go of while the core steps (the GIL) gives hold and release; they
// are called where the binding took and dropped the GIL before.
//
// The run in progress (RunState) is reached by the generated Problem<> below
// through state(), one per thread: the problem, the evaluations counted, the
// rows not yet handed over, the operator statistics, and what an observer
// may ask for (current_variables, current_points, current_eq_tol,
// current_cv_level). The cores' own records (operators used, repair counts,
// the inner archive, mosa's) are thread_local in their headers.
//
// Every algorithm of algorithms.def is instantiated by run(): a translation
// unit that includes this (or embed.hpp) pays for all of them.
// ============================================================================

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mootation.hpp"
#include "settings.hpp"
#include "suite/native.hpp"

namespace mootation::runner {

// ── The problem of a run ────────────────────────────────────────────────────
struct RunProblem {
    std::vector<std::pair<double, double>> bounds;
    int    n_objectives = 2;
    int    n_limits     = 0;
    // Binary variables after the real ones (task 7, supplement 2, §3): the
    // evaluators receive the real values, then the bits as 0.0 and 1.0.
    int    n_bin        = 0;
    // A noisy measurement (task 17, supplement 2, §1.3): problem.hpp's
    // is_stochastic(); the cores' caches are off by default where it is set.
    bool   stochastic   = false;
    // One of the two; evaluate_batch is preferred (one call per generation).
    std::function<std::vector<double>(std::vector<double>)>               evaluate;
    std::function<std::vector<std::vector<double>>(
        std::vector<std::vector<double>>)>                                evaluate_batch;
    // Optional; must return one row of limits per individual, same convention
    // as the C++ side: <= 0 satisfied, > 0 is the violation.
    std::function<std::vector<std::vector<double>>(
        std::vector<std::vector<double>>)>                                limits_batch;
    // A problem of the C++ suite (include/mootation/suite; task 11 of
    // 2026-10-18, §1.3). Set, the run evaluates it here and calls none of the
    // three above; bounds, n_objectives and n_limits must be its sizes. The
    // rows it evaluates go to on_rows(X, F, G, n) — n rows of n_var, n_obj and
    // n_con values, in the order evaluated — a batch at a time: before every
    // on_generation call, at the end of the run and every kNativeRowsCap rows
    // in between. Without on_rows nothing is kept.
    std::shared_ptr<suite::Native>                                        native;
    std::function<void(const std::vector<double>& X, const std::vector<double>& F,
                       const std::vector<double>& G, std::size_t n)>      on_rows;
    // The caller's lock around its evaluators and observer, and its release
    // while the core steps: each returns a token whose end undoes it (the
    // binding: the GIL taken, the GIL let go). Unset: nothing to hold.
    std::function<std::shared_ptr<void>()>                                hold, release;
};

// The rows a run of a suite problem has evaluated and not yet handed to
// on_rows: variables, objectives and constraint values, row after row.
struct NativeRows {
    std::vector<double> X, F, G;
    std::size_t         n = 0;
};
constexpr std::size_t kNativeRowsCap = 4096;

// Integer and discrete variables of a suite problem (task 13 of 2026-10-19,
// §3b; RunConfig::var_kinds): the core's genome decoded before the problem
// sees it, as minimize's Python path decodes it (var_types.Layout.decode_one)
// — an integer to the nearest, ties to even, kept in [ceil lo, floor hi]; a
// discrete value to the nearest of its values, ties to the lower — whether
// or not the vault has already rounded the genome (int_repair). The decoded
// row is the one on_rows receives, as the Python path's evaluator receives
// it; the core keeps its genome. Binary and categorical variables are not
// decoded here: minimize.direct_path leaves them to the Python path.
struct NativeDecode {
    std::vector<int>                 kinds;     // per variable: 0 real, 1 integer, 2 discrete; empty: none
    std::vector<std::vector<double>> values;    // a discrete variable's values, sorted
    std::vector<double>              lo, hi;    // an integer's ceil lo and floor hi
    std::vector<double>              x;         // the point decoded
};

// ── Operator statistics (the 2026-09-23 task, A2/B4/D2) ─────────────────────
// Asked for with RunConfig::operator_stats; nothing below runs otherwise, and
// nothing below draws a random number or evaluates anything, so a run's
// result is the same with or without it (tests/compat_dump.cpp, and
// test_run_layer's operator-statistics test compares the two).
//
// Per step: the population before it (the "parents"), every point the step
// evaluated (the "offspring") and the population after it. The library does
// not track which individuals an offspring came from across its cores, so
// the relations are to the parent POPULATION:
//   survival    an offspring is in the next population. By value: the next
//               population holds more copies of its variable vector than the
//               parents did (a clone of a surviving parent is not counted, a
//               child copied into several MOEA/D neighbours counts once);
//   nd          no parent dominates the offspring (objectives only);
//   step        distance to the nearest parent in variables, over the box
//               diagonal ||ub - lb||.
// The out-of-box counts come from the repair sites (operators/bound_repair.hpp).
struct OpStats {
    bool on = false;
    std::vector<std::vector<double>> log_x, log_f;           // this step's evaluations
    std::vector<std::vector<double>> par_x, par_f;           // the population before it
    // running sums: since the last operator_stats() read, and for the run
    long long off = 0, surv = 0, nd = 0, step_n = 0;
    double    step = 0.0;
    long long t_off = 0, t_surv = 0, t_nd = 0, t_step_n = 0;
    double    t_step = 0.0;
    long long evals_read = 0;
    long long setup_evals = 0;                                // not offspring
    mootation::ops::RepairCounts rc_read;
};

// The points of the population and of the vault's archive as the vault holds
// them, for current_points(): (variables, objectives, constraint values,
// violation, failed). The run layer's tests check every core with it (task 8).
using PointRow = std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
                            double, bool>;

// ── The run in progress, one per thread ─────────────────────────────────────
struct RunState {
    // The active problem. A raw pointer is enough because the problem outlives
    // the vault.
    RunProblem* problem = nullptr;
    // Evaluations spent by the current run, counted where the evaluator is
    // actually called: one per individual through eval_into, the batch size
    // through the batch executors. A warm start plants objectives without
    // evaluating and costs nothing.
    std::atomic<long long> evaluations{0};
    // The wall time the run spends in the caller's evaluators, and how many
    // times it calls them: from before the lock is taken to after the answer
    // is copied back, the on_generation observer not included.
    // Result.callback_seconds and .callbacks.
    double    callback_seconds = 0.0;
    long long callbacks        = 0;
    // The wall time the run spends evaluating a problem of the suite:
    // Result.native_seconds.
    double native_seconds = 0.0;
    NativeRows   rows;
    NativeDecode decode;
    OpStats      ops;
    // The variables of the rows on_generation receives, for the run in
    // progress: what current_variables() returns to an observer that asks for
    // them. Set by run_core, cleared by run(); asked for only at the records,
    // so a run that never asks pays nothing.
    std::function<std::vector<std::vector<double>>()> answer_variables;
    std::function<std::pair<std::vector<PointRow>, std::vector<PointRow>>()> points;
    // The equalities' tolerance and the vault's epsilon level of the run in
    // progress, for current_eq_tol() and current_cv_level().
    std::function<double()> eq_tol;
    std::function<double()> cv_level;
};

inline RunState& st()
{
    thread_local RunState s;
    return s;
}

inline std::shared_ptr<void> hold_lock()
{
    const RunProblem* p = st().problem;
    return p && p->hold ? p->hold() : nullptr;
}

inline std::shared_ptr<void> release_lock()
{
    const RunProblem* p = st().problem;
    return p && p->release ? p->release() : nullptr;
}

struct CallbackClock {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~CallbackClock()
    {
        st().callback_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        ++st().callbacks;
    }
};

// Hand the rows kept to on_rows; the caller holds the lock.
inline void flush_native_rows()
{
    RunState& s = st();
    if (!s.problem || !s.problem->on_rows || s.rows.n == 0) return;
    CallbackClock clock;
    const std::size_t n = s.rows.n;
    s.rows.n = 0;
    // the rows handed over, and the buffers emptied as the binding's arrays did
    std::vector<double> X, F, G;
    X.swap(s.rows.X);
    F.swap(s.rows.F);
    G.swap(s.rows.G);
    s.problem->on_rows(X, F, G, n);
}

inline void keep_native_row(const std::vector<double>& x, const std::vector<double>& f,
                            const std::vector<double>& g)
{
    RunState& s = st();
    if (!s.problem->on_rows) return;
    s.rows.X.insert(s.rows.X.end(), x.begin(), x.end());
    s.rows.F.insert(s.rows.F.end(), f.begin(), f.end());
    s.rows.G.insert(s.rows.G.end(), g.begin(), g.end());
    if (++s.rows.n >= kNativeRowsCap) {
        const auto gil = hold_lock();
        flush_native_rows();
    }
}

// The point the suite problem is evaluated at: vars, or vars decoded.
inline const std::vector<double>& native_point(const std::vector<double>& vars)
{
    NativeDecode& d = st().decode;
    if (d.kinds.empty()) return vars;
    std::vector<double>& x = d.x;
    x = vars;
    for (std::size_t j = 0; j < x.size() && j < d.kinds.size(); ++j) {
        if (d.kinds[j] == 1) {
            // + 0.0: a -0 rounded from (-0.5, 0) is Python's int 0, i.e. +0
            x[j] = std::clamp(std::nearbyint(x[j]), d.lo[j], d.hi[j]) + 0.0;
        } else if (d.kinds[j] == 2 && !d.values[j].empty()) {
            const auto& v = d.values[j];
            auto it = std::lower_bound(v.begin(), v.end(), x[j]);
            if (it == v.end()) x[j] = v.back();
            else if (it == v.begin()) x[j] = v.front();
            else x[j] = (x[j] - *(it - 1) <= *it - x[j]) ? *(it - 1) : *it;
        }
    }
    return x;
}

// One point of a suite problem, evaluated here.
inline void native_eval(const std::vector<double>& vars, std::vector<double>& objs,
                        std::vector<double>& lims)
{
    const auto& p = *st().problem->native;
    objs.assign(static_cast<std::size_t>(p.n_obj), 0.0);
    lims.assign(static_cast<std::size_t>(p.n_con), 0.0);
    const auto start = std::chrono::steady_clock::now();
    p.evaluate(vars.data(), objs.data(), lims.empty() ? nullptr : lims.data());
    st().native_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

inline std::string vec_key(const std::vector<double>& x)
{
    return std::string(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(double));
}

template <typename Vault>
void ops_before_step(Vault& v)
{
    st().ops.log_x.clear();
    st().ops.log_f.clear();
    st().ops.par_x.clear();
    st().ops.par_f.clear();
    const std::size_t n = std::min<std::size_t>(v.active_n(), static_cast<std::size_t>(v.pop_size()));
    for (std::size_t i = 0; i < n; ++i) {
        if (v.is_dirty(i)) continue;              // never evaluate on the observer's behalf
        st().ops.par_x.push_back(v.variables_of(i));
        st().ops.par_f.push_back(v.objectives_of(i));
    }
}

template <typename Vault>
void ops_after_step(Vault& v)
{
    const auto& O = st().ops.log_x;
    if (O.empty()) return;
    // survival, by multiset count of exact variable vectors
    std::unordered_map<std::string, long long> cp, cq, co;
    for (const auto& x : st().ops.par_x) ++cp[vec_key(x)];
    for (const auto& x : O) ++co[vec_key(x)];
    const std::size_t n = std::min<std::size_t>(v.active_n(), static_cast<std::size_t>(v.pop_size()));
    for (std::size_t i = 0; i < n; ++i) {
        auto k = vec_key(v.variables_of(i));
        if (co.count(k)) ++cq[k];
    }
    long long surv = 0;
    for (const auto& kv : co) {
        auto q = cq.find(kv.first);
        auto pp = cp.find(kv.first);
        long long extra = (q == cq.end() ? 0 : q->second) - (pp == cp.end() ? 0 : pp->second);
        surv += std::min(kv.second, std::max(0LL, extra));
    }
    // nondominated by the parents, and the step to the nearest parent
    double diag = 0.0;
    for (const auto& b : st().problem->bounds) diag += (b.second - b.first) * (b.second - b.first);
    diag = std::sqrt(diag);
    long long nd = 0, step_n = 0;
    double step = 0.0;
    for (std::size_t c = 0; c < O.size(); ++c) {
        const auto& f = st().ops.log_f[c];
        bool dominated = false;
        for (const auto& pf : st().ops.par_f) {
            bool le = true, lt = false;
            for (std::size_t j = 0; j < f.size() && le; ++j) {
                le = pf[j] <= f[j];
                lt = lt || pf[j] < f[j];
            }
            if (le && lt) { dominated = true; break; }
        }
        nd += dominated ? 0 : 1;
        if (!st().ops.par_x.empty() && diag > 0.0) {
            double best = 1e300;
            for (const auto& px : st().ops.par_x) {
                double d2 = 0.0;
                for (std::size_t j = 0; j < px.size(); ++j) d2 += (O[c][j] - px[j]) * (O[c][j] - px[j]);
                best = std::min(best, d2);
            }
            step += std::sqrt(best) / diag;
            ++step_n;
        }
    }
    const long long off = static_cast<long long>(O.size());
    st().ops.off += off;   st().ops.surv += surv;   st().ops.nd += nd;
    st().ops.step += step; st().ops.step_n += step_n;
    st().ops.t_off += off; st().ops.t_surv += surv; st().ops.t_nd += nd;
    st().ops.t_step += step; st().ops.t_step_n += step_n;
}

inline void ops_log(const std::vector<double>& x, const std::vector<double>& f)
{
    if (!st().ops.on) return;
    // the real variables only: the statistics compare them with the parents'
    st().ops.log_x.push_back(st().problem && st().problem->n_bin > 0
        ? std::vector<double>(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(
                                             st().problem->bounds.size()))
        : x);
    st().ops.log_f.push_back(f);
}

// Objective/limit evaluation, shared by every generated Problem<> below.
inline void eval_into(const std::vector<double>& vars,
                      std::vector<double>&       objs,
                      std::vector<double>&       lims)
{
    RunProblem* problem = st().problem;
    if (!problem) throw std::runtime_error("mootation: no active problem");
    if (problem->native) {
        const std::vector<double>& x = native_point(vars);
        native_eval(x, objs, lims);
        ++st().evaluations;
        ops_log(vars, objs);
        keep_native_row(x, objs, lims);
        return;
    }
    CallbackClock clock;
    const auto gil = hold_lock();

    if (problem->evaluate_batch) {
        auto out = problem->evaluate_batch({vars});
        if (out.empty()) throw std::runtime_error("evaluate_batch returned nothing");
        objs = out.front();
    } else if (problem->evaluate) {
        objs = problem->evaluate(vars);
    } else {
        throw std::runtime_error("set problem.evaluate or problem.evaluate_batch");
    }
    ++st().evaluations;
    if (static_cast<int>(objs.size()) != problem->n_objectives)
        throw std::runtime_error("the evaluator returned the wrong number of objectives");
    ops_log(vars, objs);

    if (problem->n_limits > 0) {
        if (!problem->limits_batch)
            throw std::runtime_error("n_limits > 0 requires problem.limits_batch");
        auto lim = problem->limits_batch({vars});
        if (lim.empty()) throw std::runtime_error("limits_batch returned nothing");
        lims = lim.front();
        if (static_cast<int>(lims.size()) != problem->n_limits)
            throw std::runtime_error("limits_batch returned the wrong width");
    }
}

// The BatchExecutor installed when the caller supplied evaluate_batch: the
// whole dirty set in ONE call, which is what "batched evaluation" promises.
// Without it DataVault falls back to calc_objs one individual at a time and
// a batch function only ever sees single rows.
inline void callback_run_batch(const BatchRequest& req, BatchResponse& resp)
{
    RunProblem* problem = st().problem;
    if (!problem) throw std::runtime_error("mootation: no active problem");
    CallbackClock clock;
    const auto gil = hold_lock();

    // the bits after the real variables (RunProblem::n_bin)
    std::vector<std::vector<double>> rows = req.variables;
    if (problem->n_bin > 0)
        for (std::size_t i = 0; i < rows.size(); ++i)
            for (int b : req.binary_variables[i]) rows[i].push_back(static_cast<double>(b));
    auto out = problem->evaluate_batch(rows);
    if (out.size() != req.size())
        throw std::runtime_error(
            "evaluate_batch returned " + std::to_string(out.size()) +
            " rows for a batch of " + std::to_string(req.size()));
    for (const auto& row : out)
        if (static_cast<int>(row.size()) != problem->n_objectives)
            throw std::runtime_error("the evaluator returned the wrong number of objectives");
    st().evaluations += static_cast<long long>(req.size());
    for (std::size_t i = 0; i < req.size(); ++i) ops_log(rows[i], out[i]);
    resp.objectives = std::move(out);

    if (problem->n_limits > 0) {
        if (!problem->limits_batch)
            throw std::runtime_error("n_limits > 0 requires problem.limits_batch");
        auto lim = problem->limits_batch(rows);
        if (lim.size() != req.size())
            throw std::runtime_error(
                "limits_batch returned " + std::to_string(lim.size()) +
                " rows for a batch of " + std::to_string(req.size()));
        for (const auto& row : lim)
            if (static_cast<int>(row.size()) != problem->n_limits)
                throw std::runtime_error("limits_batch returned the wrong width");
        resp.limits = std::move(lim);
    } else {
        resp.limits.clear();
    }
}

// The BatchExecutor of a suite problem (RunProblem::native): the batch's rows
// evaluated here, in order, as eval_into evaluates one.
inline void native_run_batch(const BatchRequest& req, BatchResponse& resp)
{
    RunProblem* problem = st().problem;
    if (!problem) throw std::runtime_error("mootation: no active problem");
    const auto& p = *problem->native;
    resp.resize(req.size(), p.n_obj, p.n_con);
    std::vector<double> none;
    for (std::size_t i = 0; i < req.size(); ++i) {
        std::vector<double>& lims = p.n_con > 0 ? resp.limits[i] : none;
        const std::vector<double>& x = native_point(req.variables[i]);
        native_eval(x, resp.objectives[i], lims);
        ++st().evaluations;
        ops_log(req.variables[i], resp.objectives[i]);
        keep_native_row(x, resp.objectives[i], lims);
    }
}

}   // namespace mootation::runner

// Each algorithm needs ITS OWN individual type — crEA reads .cluster/.rank,
// NSGA-II reads .rank/.crowding_distance — so one tag per algorithm is
// generated from the same X-macro that drives the dispatch. The Problem<>
// specialization has to be written out in full: Based_Individual befriends
// Problem<T> only, so nothing else may touch ind.variables/objectives/limits.
#define MOOTATION_RUN_PROBLEM(TAG, BASE)                                       \
    namespace mootation::runner { struct TAG : public mootation::BASE {}; }       \
    namespace mootation {                                                      \
    template <>                                                                \
    class Problem<runner::TAG> {                                                  \
    public:                                                                    \
        std::vector<std::pair<std::optional<double>, std::optional<double>>>   \
            bounds;                                                            \
        int get_vars_n() const { return static_cast<int>(bounds.size()); }     \
        int get_bin_vars_n() const                                             \
        {                                                                      \
            return runner::st().problem ? runner::st().problem->n_bin : 0;           \
        }                                                                      \
        bool is_stochastic() const                                             \
        {                                                                      \
            return runner::st().problem && runner::st().problem->stochastic;         \
        }                                                                      \
        int get_objs_n() const                                                 \
        {                                                                      \
            return runner::st().problem ? runner::st().problem->n_objectives : 0;    \
        }                                                                      \
        int get_lims_n() const                                                 \
        {                                                                      \
            return runner::st().problem ? runner::st().problem->n_limits : 0;        \
        }                                                                      \
        void calc_objs(runner::TAG& ind) const                                    \
        {                                                                      \
            if (runner::st().problem && runner::st().problem->n_bin > 0) {           \
                std::vector<double> x = ind.variables;                         \
                for (int b : ind.binary_variables) x.push_back(b);             \
                runner::eval_into(x, ind.objectives, ind.limits);                 \
            } else {                                                           \
                runner::eval_into(ind.variables, ind.objectives, ind.limits);     \
            }                                                                  \
        }                                                                      \
    };                                                                         \
    }

#define MOOTATION_ALG(KEY, IND, CORE) MOOTATION_RUN_PROBLEM(RunTag_##KEY, IND)
#include "algorithms.def"
#undef MOOTATION_ALG
#undef MOOTATION_RUN_PROBLEM

namespace mootation::runner {

// ── Optional-setter detection ───────────────────────────────────────────────
// One macro per knob: a trait that asks "does this core have set_X(T)?", and an
// apply() that calls it when it does and reports when it does not.
#define MOOTATION_OPTIONAL_SETTER(NAME, SETTER, TYPE)                          \
    template <typename C, typename = void>                                     \
    struct has_##NAME : std::false_type {};                                    \
    template <typename C>                                                      \
    struct has_##NAME<C, std::void_t<decltype(std::declval<C&>().SETTER(       \
                             std::declval<TYPE>()))>> : std::true_type {};     \
    template <typename C>                                                      \
    bool apply_##NAME(C& core, TYPE v)                                         \
    {                                                                          \
        if constexpr (has_##NAME<C>::value) { core.SETTER(v); return true; }    \
        else { (void)core; (void)v; return false; }                            \
    }

MOOTATION_OPTIONAL_SETTER(t_max,      set_t_max,             int)
MOOTATION_OPTIONAL_SETTER(eta_c,      set_eta_crossover,     double)
MOOTATION_OPTIONAL_SETTER(eta_m,      set_eta_mutation,      double)
MOOTATION_OPTIONAL_SETTER(pc,         set_pc,                double)
MOOTATION_OPTIONAL_SETTER(pm,         set_pm,                double)
MOOTATION_OPTIONAL_SETTER(T,          set_T,                 int)
MOOTATION_OPTIONAL_SETTER(delta,      set_delta,             double)
MOOTATION_OPTIONAL_SETTER(nr,         set_nr,                int)
MOOTATION_OPTIONAL_SETTER(kappa,      set_kappa,             double)
MOOTATION_OPTIONAL_SETTER(K,          set_K,                 int)
MOOTATION_OPTIONAL_SETTER(n_clusters, set_n_clusters,        int)
MOOTATION_OPTIONAL_SETTER(theta,      set_theta,             double)
MOOTATION_OPTIONAL_SETTER(alpha,      set_alpha,             double)
MOOTATION_OPTIONAL_SETTER(F,          set_F,                 double)
MOOTATION_OPTIONAL_SETTER(CR,         set_CR,                double)
MOOTATION_OPTIONAL_SETTER(div,        set_div,               int)
MOOTATION_OPTIONAL_SETTER(normalize,  set_normalize,         bool)
MOOTATION_OPTIONAL_SETTER(bound_repair, set_bound_repair,    mootation::ops::BoundRepair)
MOOTATION_OPTIONAL_SETTER(crossover,  set_crossover,         mootation::ops::Crossover)
MOOTATION_OPTIONAL_SETTER(mutation,   set_mutation,          mootation::ops::Mutation)
MOOTATION_OPTIONAL_SETTER(mutation_scale, set_mutation_scale, double)
MOOTATION_OPTIONAL_SETTER(mixture_q,  set_mixture_q,         double)
MOOTATION_OPTIONAL_SETTER(blx_alpha,  set_blx_alpha,         double)
MOOTATION_OPTIONAL_SETTER(crowding_space, set_crowding_space, mootation::CrowdingSpace)
MOOTATION_OPTIONAL_SETTER(dms_init,   set_init,              mootation::DMSInit)
MOOTATION_OPTIONAL_SETTER(dms_poll,   set_poll,              mootation::DMSPoll)
MOOTATION_OPTIONAL_SETTER(step_share, set_step_share,        double)
MOOTATION_OPTIONAL_SETTER(F_spread,   set_F_spread,          double)

#undef MOOTATION_OPTIONAL_SETTER

// The knobs after the per-knob setters (settings.hpp, is_core_param): a core
// takes them through set_param / set_word, which answer whether it has them.
template <typename C, typename = void>
struct has_set_param : std::false_type {};
template <typename C>
struct has_set_param<C, std::void_t<decltype(std::declval<C&>().set_param(
                            std::declval<const std::string&>(), std::declval<double>()))>>
    : std::true_type {};
template <typename C, typename = void>
struct has_set_word : std::false_type {};
template <typename C>
struct has_set_word<C, std::void_t<decltype(std::declval<C&>().set_word(
                           std::declval<const std::string&>(), std::declval<const std::string&>()))>>
    : std::true_type {};
template <typename C>
bool apply_param(C& core, const std::string& k, double v)
{
    if constexpr (has_set_param<C>::value) return core.set_param(k, v);
    else { (void)core; (void)k; (void)v; return false; }
}
template <typename C>
bool apply_word(C& core, const std::string& k, const std::string& w)
{
    if constexpr (has_set_word<C>::value) return core.set_word(k, w);
    else { (void)core; (void)k; (void)w; return false; }
}

// A core with an inner archive (operators/inner_archive.hpp): its members.
template <typename C, typename = void>
struct has_inner_archive : std::false_type {};
template <typename C>
struct has_inner_archive<C, std::void_t<decltype(std::declval<const C&>().inner_archive())>>
    : std::true_type {};

// A core whose knobs come from a preset (smpso): every knob as the run used it.
template <typename C, typename = void>
struct has_effective_settings : std::false_type {};
template <typename C>
struct has_effective_settings<C, std::void_t<decltype(std::declval<const C&>().effective_settings())>>
    : std::true_type {};

// A core that can tell it has converged (DMS: every step size below its
// tolerance) ends the run early instead of stepping on without evaluating.
template <typename C, typename = void>
struct has_finished : std::false_type {};
template <typename C>
struct has_finished<C, std::void_t<decltype(std::declval<const C&>().finished())>>
    : std::true_type {};
template <typename C>
bool core_finished(const C& c)
{
    if constexpr (has_finished<C>::value) return c.finished();
    else { (void)c; return false; }
}

// ── Run configuration ───────────────────────────────────────────────────────
// NOT named PyConfig: CPython's own Python.h defines that (PEP 587).
// Every knob is optional. `has_*` is false until the caller assigns, so an
// unset knob leaves the algorithm's paper default alone.
struct RunConfig {
    int      pop_size = 100;
    int      n_gen    = 250;
    unsigned seed     = 0;
    mootation::ConstraintMode constraint_mode = mootation::ConstraintMode::NONE;

    std::optional<double> eta_c, eta_m, pc, pm, delta, kappa, theta, alpha, F, CR;
    std::optional<int>    T, nr, K, n_clusters, div;
    // The objective normalization that ibea_eplus, r2ibea, two_arch2 and
    // moead_am2m can switch; each header says which setting is the paper's.
    std::optional<bool>   normalize;
    // What an operator that can leave the box does there (text_knobs() in
    // settings.hpp); the algorithms without such an operator report it ignored.
    std::optional<std::string> bound_repair;
    // Switchable operators (nsga2, nsga2_sdr, ibea_eplus, spea2_sde, agemoea; moead_de the
    // mutation only) and their parameters: text_knobs() / knob_names().
    std::optional<std::string> crossover, mutation;
    // nsga2: crowding distance over the objectives or the decision variables
    std::optional<std::string> crowding_space;
    // dms: the initial list, line (n points on the diagonal), single or random;
    // the next poll centre, first or random (dms.hpp, DMS-6)
    std::optional<std::string> dms_init, dms_poll;
    std::optional<double>      mutation_scale, mixture_q, blx_alpha;
    // emyo_c: the clamp of the difference vector (a share of the range) and
    // the spread of its factor F (emyo_c.hpp, the knobs for the mechanism)
    std::optional<double>      step_share, F_spread;
    // The SBX operator's share of crossed variables (ops::sbx_var_prob, 0.5),
    // for this run only.
    std::optional<double>      sbx_var_prob;
    // The knobs after the per-knob setters (settings.hpp, is_core_param and
    // is_core_word): name -> value, handed to the core's set_param / set_word.
    std::map<std::string, double>      params;
    std::map<std::string, std::string> words;
    // Integer and discrete variables (task 7, §10): per real variable 0 real,
    // 1 integer, 2 discrete (its values in discrete_values, sorted); with
    // lamarck (the default) the vault writes the rounded value into the genome
    // (DataVault::set_snap), else the evaluator rounds (minimize's wrapper).
    // eta_int: polynomial mutation's index for them (ops::pm_eta_per_variable).
    std::vector<int>                 var_kinds;
    std::vector<std::vector<double>> discrete_values;
    bool                             lamarck = true;
    std::optional<double>            eta_int;
    // Categorical variables (supplement 3, §1): the levels of every gene of
    // the integer genome, 2 for a bit (ops::gene_levels); empty: all bits.
    std::vector<int>                 gene_levels;
    // the bits of a warm start, beside seed_variables (PyProblem::n_bin)
    std::vector<std::vector<int>>    seed_binary;

    // Warm start. Empty means a fresh random population. Vars and objs are
    // what every algorithm shares — which is what lets a population saved by
    // NSGA-II seed a MOEA/D run. A constrained run also needs the individual
    // constraint values (the algorithms compare those, not the aggregate cv):
    // minimize() recomputes them from `constraints`, and the binding refuses
    // a constrained warm start without them rather than planting a population
    // every one of whose constraints reads as satisfied.
    std::vector<std::vector<double>> seed_variables;
    std::vector<std::vector<double>> seed_objectives;
    std::vector<std::vector<double>> seed_limits;

    // Per-generation observer. When set and record_every > 0, the run is
    // stepped one generation at a time and on_generation(gen, objectives) is
    // called after setup (gen = 0) and after every record_every-th generation
    // (and the last one). `objectives` are the current answer set's rows —
    // the same view the final Result returns — so a caller can record IGD or
    // hypervolume against the evaluation count it keeps in its own evaluator.
    // Off by default: optimize() runs uninterrupted.
    std::function<void(int, std::vector<std::vector<double>>)> on_generation;
    int record_every = 0;
    // Called after setup (0) and after every step (its number), whatever
    // record_every: embed.hpp's Session counts the generations with it.
    std::function<void(int)> on_step;

    // Evaluation budget. 0 = run n_gen steps. > 0 = step until this many
    // evaluations have been spent; n_gen is then ignored. A step is not a
    // generation of pop_size evaluations for every core: NIMMO evaluates one
    // offspring per step and MOEA/D-DRA and -AWA a fifth of the population,
    // so a budget in steps is not a budget. The check sits between steps, so
    // a generational core overshoots by less than one generation. The
    // observer then fires every record_every * pop_size evaluations instead
    // of every record_every steps, which keeps trajectories comparable.
    long long max_evaluations = 0;
    // Operator statistics along the run (the per-step bookkeeping above); read
    // them with operator_stats() from the on_generation observer.
    bool operator_stats = false;
    // A failed evaluation (a non-finite value; minimize turns the user's
    // exception into NaN): the vault's penalty (false, on_failure =
    // "infeasible", the default) or an error (true, "raise"). See
    // data_vault.hpp, FAILED EVALUATIONS.
    bool fail_raise = false;
    // Equality constraints (data_vault.hpp, EQUALITY CONSTRAINTS): the last
    // n_eq of the problem's n_limits values are equalities h = 0, satisfied
    // within eq_tol. eq_linear: the tolerance falls from eq_tol0 to eq_tol in
    // proportion to the evaluations spent, set before every step (it needs
    // max_evaluations); at the end it is eq_tol either way.
    int    n_eq      = 0;
    double eq_tol    = 1e-4;
    double eq_tol0   = 1e-4;
    bool   eq_linear = false;
    // Distinct decoded points (supplement 3, §1.2; mootation/unique.py): the
    // caller's count of them, and whether the evaluation budget, its schedules
    // and the observer's stride count them instead of calls (budget_counts =
    // "unique": a repeat answered from memory costs nothing).
    std::function<long long()> unique_evaluations;
    bool                       budget_unique = false;
    // EPS_LEVEL (supplement 3, §4.1; data_vault.hpp, EPSILON LEVEL): the core
    // ranks by feasibility (constraint_mode), the vault counts a violation up
    // to eps as none, and eps follows Fan et al.'s improved setting (Soft
    // Computing 23(23), 2019, Eq. 8) once per generation — pop_size
    // evaluations under max_evaluations, a step under n_gen: eps(0) the
    // violation of the initial population's (eps_theta N)-th worst point,
    // then (1 - eps_tau) eps while fewer than eps_alpha of the population are
    // feasible, (1 + eps_tau) times the largest violation seen once that many
    // are, and 0 from eps_tc of the budget on. The paper: tau 0.1, alpha 0.95,
    // theta 0.05, Tc 800 of its 1000 generations — eps_tc = 0.8 here, of the
    // budget the run has, not 800 generations of any budget (a shorter budget
    // would never reach eps = 0). The answer's cv is the violation.
    bool   eps_level = false;
    double eps_tau = 0.1, eps_alpha = 0.95, eps_theta = 0.05, eps_tc = 0.8;
};

struct RunResult {
    std::vector<std::vector<double>> objectives;
    std::vector<std::vector<double>> variables;
    std::vector<double>              cv;
    std::size_t                      active_n = 0;
    // Knobs the caller set that this algorithm does not have. Reported rather
    // than dropped: a run configured with an ignored parameter is not the run
    // the caller asked for.
    std::vector<std::string>         ignored;
    // Evaluations the run spent, setup included.
    long long                        evaluations = 0;
    // The steps the core took, and the answer's per-constraint values (n_limits
    // of them a row, `<= 0` satisfied; empty without constraints), beside cv.
    int                              generations = 0;
    std::vector<std::vector<double>> limits;
    bool                             finished = false;   // the core stopped by its own rule (task 16)
    // Every variation operator the run used, with its bound repair ("none" for
    // one that cannot leave the box), in order of first use.
    std::vector<std::pair<std::string, std::string>> operators;
    // Every knob as the run used it, for a core whose knobs come from a preset
    // (smpso); empty for the others.
    std::vector<std::pair<std::string, std::string>> settings;
    // The run's operator statistics (operator_stats() over the whole run);
    // empty unless RunConfig::operator_stats.
    std::map<std::string, double>    operator_totals;
    // The inner archive (operators/inner_archive.hpp) at the end, for the cores
    // that keep one and with inner_archive = "nd"; empty otherwise.
    std::vector<std::vector<double>> inner_objectives;
    std::vector<std::vector<double>> inner_variables;
    // the bits of the answer, beside variables (PyProblem::n_bin)
    std::vector<std::vector<int>>    binary_variables;
    // Evaluations that failed (data_vault.hpp, FAILED EVALUATIONS). A failed
    // member of the final population is not in the answer above.
    long long                        n_failed = 0;
    // The equalities' tolerance the answer's cv was taken with (RunConfig::
    // eq_tol); 0 without equalities.
    double                           eq_tol = 0.0;
    // Distinct decoded points evaluated (RunConfig::unique_evaluations); -1
    // where the caller did not count them.
    long long                        unique_evaluations = -1;
    // The time spent in the Python evaluators and the number of calls (the
    // counters beside st().evaluations): what a run of cheap problems pays for
    // the crossing; tools/perf reads them.
    double                           callback_seconds = 0.0;
    long long                        callbacks = 0;
    // The time spent evaluating a problem of the suite (PyProblem::native).
    double                           native_seconds = 0.0;
};

template <typename Tag, typename Core>
RunResult run_core(const RunConfig& cfg)
{
    using namespace mootation;

    Problem<Tag> prob;
    prob.bounds.reserve(st().problem->bounds.size());
    for (auto& b : st().problem->bounds)
        prob.bounds.emplace_back(std::optional<double>(b.first),
                                 std::optional<double>(b.second));

    DataVault<Tag>      vault(cfg.pop_size, prob);
    if (st().problem->native) vault.set_batch_executor(&native_run_batch);
    else if (st().problem->evaluate_batch) vault.set_batch_executor(&callback_run_batch);
    vault.set_fail_raise(cfg.fail_raise);
    if (cfg.n_eq > 0) {
        if (cfg.eq_linear && cfg.max_evaluations <= 0)
            throw std::invalid_argument("mootation: the linear schedule of eq_tol runs over "
                                        "the evaluation budget; set max_evaluations");
        vault.set_equalities(static_cast<std::size_t>(cfg.n_eq),
                             cfg.eq_linear ? cfg.eq_tol0 : cfg.eq_tol);
    }
    // integer and discrete variables (RunConfig::var_kinds): the Lamarckian
    // rounding at every write — integers to the nearest, ties to even, kept in
    // [ceil lo, floor hi]; discrete ones to the nearest value, ties to the lower
    if (!cfg.var_kinds.empty() && cfg.lamarck) {
        std::vector<double> lo, hi;
        for (const auto& b : st().problem->bounds) { lo.push_back(b.first); hi.push_back(b.second); }
        vault.set_snap([kinds = cfg.var_kinds, values = cfg.discrete_values, lo, hi](
                           std::vector<double>& x) {
            for (std::size_t j = 0; j < x.size() && j < kinds.size(); ++j) {
                if (kinds[j] == 1) {
                    x[j] = std::clamp(std::nearbyint(x[j]), std::ceil(lo[j]), std::floor(hi[j]));
                } else if (kinds[j] == 2 && !values[j].empty()) {
                    const auto& v = values[j];
                    auto it = std::lower_bound(v.begin(), v.end(), x[j]);
                    if (it == v.end()) x[j] = v.back();
                    else if (it == v.begin()) x[j] = v.front();
                    else x[j] = (x[j] - *(it - 1) <= *it - x[j]) ? *(it - 1) : *it;
                }
            }
        });
    }
    {
        auto& per = ops::pm_eta_per_variable();
        per.clear();
        if (cfg.eta_int && !cfg.var_kinds.empty())
            for (int k : cfg.var_kinds) per.push_back(k != 0 ? *cfg.eta_int : 0.0);
    }
    ops::gene_levels() = cfg.gene_levels;
    Optimizer<Tag, Core> opt(std::move(vault), defer_setup);

    auto&                    alg = opt.get_algorithm();
    std::vector<std::string> ignored;
    auto note = [&](bool applied, const char* name) {
        if (!applied) ignored.emplace_back(name);
    };

    alg.set_seed(cfg.seed);
    alg.constraint_mode = cfg.constraint_mode;

    // t_max is not a user knob but a schedule input: several algorithms anneal
    // against it, and a default of 1000 against a 250-generation run is not the
    // paper's schedule. Always pass the real budget where the core takes it.
    // Under an evaluation budget the number of steps is not known yet: start
    // from budget / pop_size and correct it after the first step (below).
    const bool by_evaluations = cfg.max_evaluations > 0;
    if (by_evaluations)
        apply_t_max(alg, static_cast<int>(std::min<long long>(
            INT_MAX, (cfg.max_evaluations + cfg.pop_size - 1) / std::max(1, cfg.pop_size))));
    else
        apply_t_max(alg, cfg.n_gen);

    if (cfg.eta_c)      note(apply_eta_c(alg, *cfg.eta_c),           "eta_c");
    if (cfg.eta_m)      note(apply_eta_m(alg, *cfg.eta_m),           "eta_m");
    if (cfg.pc)         note(apply_pc(alg, *cfg.pc),                 "pc");
    if (cfg.pm)         note(apply_pm(alg, *cfg.pm),                 "pm");
    if (cfg.T)          note(apply_T(alg, *cfg.T),                   "T");
    if (cfg.delta)      note(apply_delta(alg, *cfg.delta),           "delta");
    if (cfg.nr)         note(apply_nr(alg, *cfg.nr),                 "nr");
    if (cfg.kappa)      note(apply_kappa(alg, *cfg.kappa),           "kappa");
    if (cfg.K)          note(apply_K(alg, *cfg.K),                   "K");
    if (cfg.n_clusters) note(apply_n_clusters(alg, *cfg.n_clusters), "n_clusters");
    if (cfg.theta)      note(apply_theta(alg, *cfg.theta),           "theta");
    if (cfg.alpha)      note(apply_alpha(alg, *cfg.alpha),           "alpha");
    if (cfg.F)          note(apply_F(alg, *cfg.F),                   "F");
    if (cfg.CR)         note(apply_CR(alg, *cfg.CR),                 "CR");
    if (cfg.div)        note(apply_div(alg, *cfg.div),               "div");
    if (cfg.normalize)  note(apply_normalize(alg, *cfg.normalize),   "normalize");
    if (cfg.bound_repair) {
        auto how = ops::parse_bound_repair(*cfg.bound_repair);
        if (!how)
            throw std::invalid_argument(
                "bound_repair = '" + *cfg.bound_repair + "': one of clip, reflect, random, "
                "midpoint, resample, wrap, native");
        note(apply_bound_repair(alg, *how), "bound_repair");
    }
    if (cfg.crossover) {
        auto c = ops::parse_crossover(*cfg.crossover);
        if (!c)
            throw std::invalid_argument("crossover = '" + *cfg.crossover +
                                        "': one of sbx, uniform, blx_alpha, spx, rex, undx, pcx, "
                                        "de, de_pbest, portfolio");
        note(apply_crossover(alg, *c), "crossover");
    }
    if (cfg.mutation) {
        auto m = ops::parse_mutation(*cfg.mutation);
        if (!m)
            throw std::invalid_argument("mutation = '" + *cfg.mutation + "': one of polynomial, "
                                        "gaussian, cauchy, uniform_reset, mixture, mixture_cauchy, "
                                        "none");
        note(apply_mutation(alg, *m), "mutation");
    }
    if (cfg.dms_init) {
        auto i = parse_dms_init(*cfg.dms_init);
        if (!i) throw std::invalid_argument("dms_init = '" + *cfg.dms_init +
                                            "': line, single or random");
        note(apply_dms_init(alg, *i), "dms_init");
    }
    if (cfg.dms_poll) {
        auto p = parse_dms_poll(*cfg.dms_poll);
        if (!p) throw std::invalid_argument("dms_poll = '" + *cfg.dms_poll + "': first or random");
        note(apply_dms_poll(alg, *p), "dms_poll");
    }
    if (cfg.crowding_space) {
        auto c = parse_crowding_space(*cfg.crowding_space);
        if (!c)
            throw std::invalid_argument("crowding_space = '" + *cfg.crowding_space +
                                        "': objectives or decision");
        note(apply_crowding_space(alg, *c), "crowding_space");
    }
    if (cfg.mutation_scale) note(apply_mutation_scale(alg, *cfg.mutation_scale), "mutation_scale");
    if (cfg.mixture_q)      note(apply_mixture_q(alg, *cfg.mixture_q),           "mixture_q");
    if (cfg.blx_alpha)      note(apply_blx_alpha(alg, *cfg.blx_alpha),           "blx_alpha");
    if (cfg.step_share)     note(apply_step_share(alg, *cfg.step_share),         "step_share");
    if (cfg.F_spread)       note(apply_F_spread(alg, *cfg.F_spread),             "F_spread");
    for (const auto& kv : cfg.params) {
        if (!mootation::is_core_param(kv.first))
            throw std::invalid_argument("params: unknown knob '" + kv.first + "'");
        note(apply_param(alg, kv.first, kv.second), kv.first.c_str());
    }
    for (const auto& kv : cfg.words) {
        if (!mootation::is_core_word(kv.first))
            throw std::invalid_argument("words: unknown knob '" + kv.first + "'");
        if (!mootation::text_knob_accepts(kv.first, kv.second))
            throw std::invalid_argument(kv.first + " = '" + kv.second + "' is not one of its values");
        note(apply_word(alg, kv.first, kv.second), kv.first.c_str());
    }
    long long steps_done = 0;                 // the steps taken (RunResult::generations)
    std::optional<ops::ScopedSbxVarProb> var_prob;
    if (cfg.sbx_var_prob) {
        if (!(*cfg.sbx_var_prob >= 0.0 && *cfg.sbx_var_prob <= 1.0))
            throw std::invalid_argument("sbx_var_prob must be in [0, 1]");
        var_prob.emplace(*cfg.sbx_var_prob);
    }

    {
        const auto unlock = release_lock();   // the callback re-acquires per batch
        if (!cfg.seed_variables.empty()) {
            // setup_with_seed plants the population WITHOUT evaluating it: the
            // objectives come from the file, so a warm start costs zero
            // function evaluations. That is the whole point when an evaluation
            // is a solver run.
            if (st().problem->n_limits > 0 && cfg.seed_limits.empty())
                throw std::invalid_argument(
                    "mootation: this run has " + std::to_string(st().problem->n_limits) +
                    " constraints but the seed population carries no per-constraint "
                    "values; pass seed_limits (minimize() recomputes them from "
                    "`constraints`)");
            opt.setup_with_seed(cfg.seed_variables, cfg.seed_objectives, cfg.seed_binary,
                                cfg.seed_limits);
        } else {
            opt.setup();
        }
        if (cfg.on_step) cfg.on_step(0);
        // EPS_LEVEL: eps(0) from the initial population (Eq. 8, rule 1)
        int eps_gen = 0;                      // generations whose eps is set
        if (cfg.eps_level) {
            auto& v = opt.get_vault();
            const std::size_t n = std::min<std::size_t>(v.active_n(),
                                                        static_cast<std::size_t>(v.pop_size()));
            std::vector<double> cv;
            for (std::size_t i = 0; i < n; ++i)
                if (!v.is_dirty(i)) cv.push_back(v.raw_cv(i));
            std::stable_sort(cv.begin(), cv.end(), std::greater<double>());
            const std::size_t theta = static_cast<std::size_t>(
                std::max(1L, std::lround(cfg.eps_theta * static_cast<double>(v.pop_size()))));
            v.set_cv_level(cv.empty() ? 0.0 : cv[std::min(theta, cv.size()) - 1]);
            // rule 4 from generation 0 when tc = 0: a core whose setup spends
            // fewer than pop_size evaluations (paes: one) would otherwise run
            // its first steps at eps(0) before the first update
            if (cfg.eps_tc <= 0.0) v.set_cv_level(0.0);
        }
        // the initial population is not offspring: the counts start here
        st().ops.setup_evals = st().ops.evals_read = st().evaluations.load();
        st().ops.rc_read = ops::repair_counts();
        const bool observe = cfg.on_generation && cfg.record_every > 0;
        // the variables of the rows emit() hands over, when the observer asks:
        // the real genome and then the integer one (bits, categorical
        // variables' label indices), as the evaluator receives a point
        st().answer_variables = [&opt]() {
            auto& v = opt.get_vault();
            std::size_t n = std::min<std::size_t>(
                v.active_n(), static_cast<std::size_t>(v.pop_size()));
            std::vector<std::vector<double>> X;
            X.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                if (v.is_failed(i)) continue;                           // as emit() below
                X.push_back(v.variables_of(i));
                if (v.bin_vars_n() > 0)
                    for (int b : v.binary_variables_of(i)) X.back().push_back(static_cast<double>(b));
            }
            return X;
        };
        st().points = [&opt]() {
            auto& v = opt.get_vault();
            auto row = [](const mootation::VaultPoint& p) {
                return PointRow{p.variables, p.objectives, p.limits, p.cv_raw, p.failed};
            };
            std::pair<std::vector<PointRow>, std::vector<PointRow>> out;
            const std::size_t n = std::min<std::size_t>(
                v.active_n(), static_cast<std::size_t>(v.pop_size()));
            for (std::size_t i = 0; i < n; ++i)
                if (!v.is_dirty(i)) out.first.push_back(row(v.point_of(i)));   // never evaluate
            for (std::size_t i = 0; i < v.archive_size(); ++i)
                out.second.push_back(row(v.archive_point_of(i)));
            return out;
        };
        st().eq_tol = [&opt]() { return opt.get_vault().eq_tol(); };
        // the evaluations the budget counts: calls, or distinct points
        auto spent = [&]() -> long long {
            if (!cfg.budget_unique || !cfg.unique_evaluations) return st().evaluations.load();
            const auto gil = hold_lock();
            return cfg.unique_evaluations();
        };
        st().cv_level = [&opt]() { return opt.get_vault().cv_level(); };
        // EPS_LEVEL, Eq. 8 rules 2-4, once for every generation reached
        auto eps_update = [&]() {
            auto& v = opt.get_vault();
            const double pop = static_cast<double>(std::max(1, cfg.pop_size));
            const double gens_total = by_evaluations
                ? static_cast<double>(cfg.max_evaluations) / pop : static_cast<double>(cfg.n_gen);
            const int gen_now = by_evaluations
                ? static_cast<int>(static_cast<double>(spent()) / pop)
                : static_cast<int>(steps_done);
            for (; eps_gen < gen_now; ++eps_gen) {
                const double k = static_cast<double>(eps_gen + 1);
                if (k >= cfg.eps_tc * gens_total) { v.set_cv_level(0.0); continue; }
                const std::size_t n = std::min<std::size_t>(
                    v.active_n(), static_cast<std::size_t>(v.pop_size()));
                std::size_t feasible = 0, seen = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    if (v.is_dirty(i)) continue;
                    ++seen;
                    feasible += v.raw_cv(i) <= 0.0;
                }
                const double r = seen ? static_cast<double>(feasible) / static_cast<double>(seen) : 0.0;
                v.set_cv_level(r < cfg.eps_alpha ? (1.0 - cfg.eps_tau) * v.cv_level()
                                                 : (1.0 + cfg.eps_tau) * v.max_raw_cv());
            }
        };
        auto one_step = [&]() {
            if (cfg.eps_level) eps_update();
            if (cfg.n_eq > 0 && cfg.eq_linear) {
                // the share of the budget spent so far, setup included
                const double t = std::min(1.0, static_cast<double>(spent()) /
                                                   static_cast<double>(cfg.max_evaluations));
                opt.get_vault().set_eq_tol(cfg.eq_tol0 + (cfg.eq_tol - cfg.eq_tol0) * t);
            }
            if (st().ops.on) ops_before_step(opt.get_vault());
            opt.step();
            ++steps_done;
            if (st().ops.on) ops_after_step(opt.get_vault());
            if (cfg.on_step) cfg.on_step(static_cast<int>(steps_done));
        };
        auto emit = [&](int g) {
            auto& v = opt.get_vault();
            std::size_t n = std::min<std::size_t>(
                v.active_n(), static_cast<std::size_t>(v.pop_size()));
            std::vector<std::vector<double>> P;
            P.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                const auto& f = v.objectives_of(i);              // evaluates a dirty slot
                if (!v.is_failed(i)) P.push_back(f);             // a failure is no answer
            }
            const auto gil = hold_lock();
            flush_native_rows();                         // the rows behind this population
            cfg.on_generation(g, std::move(P));
        };
        if (!by_evaluations) {
            if (observe) {
                emit(0);
                for (int g = 1; g <= cfg.n_gen; ++g) {
                    one_step();
                    const bool done = core_finished(alg);
                    if (g % cfg.record_every == 0 || g == cfg.n_gen || done) emit(g);
                    if (done) break;
                }
            } else {
                for (int g = 1; g <= cfg.n_gen && !core_finished(alg); ++g) one_step();
            }
        } else {
            if (observe) emit(0);
            const long long start  = spent();
            const long long stride = static_cast<long long>(cfg.record_every) *
                                     std::max(1, cfg.pop_size);
            long long next_record  = start + stride;
            long long idle         = 0;   // consecutive steps that evaluated nothing
            for (int g = 1; spent() < cfg.max_evaluations; ++g) {
                const long long before = spent();
                one_step();
                const long long now = spent();
                if (now > before) {
                    idle = 0;
                } else if (++idle >= 10000) {
                    // counting distinct points: the core asks only for points it
                    // has seen, a space too small for the budget; the run ends
                    if (cfg.budget_unique) break;
                    throw std::runtime_error(
                        "mootation: 10000 steps in a row evaluated nothing; this core "
                        "cannot run on an evaluation budget, use n_gen");
                }
                if (g == 1) {
                    // The real cost of a step is known now: give the schedule the
                    // number of steps the budget buys (MOEA/D-AWA anneals over
                    // five times more steps than a generational core).
                    const long long per  = std::max(1LL, now - start);
                    const long long left = std::max(0LL, cfg.max_evaluations - now);
                    apply_t_max(alg, static_cast<int>(std::min<long long>(
                        INT_MAX, 1 + (left + per - 1) / per)));
                }
                const bool finished = core_finished(alg);
                const bool last = now >= cfg.max_evaluations || finished;
                if (observe && (now >= next_record || last)) {
                    emit(g);
                    while (next_record <= now) next_record += stride;
                }
                if (finished) break;
            }
        }
    }

    flush_native_rows();                          // the run's last rows
    auto&    v = opt.get_vault();
    if (cfg.n_eq > 0) v.set_eq_tol(cfg.eq_tol);   // the answer at the final tolerance
    v.set_cv_level(0.0);                          // and its cv the violation itself
    RunResult r;
    // NOT active_n(): steady-state cores (moead_de, moead_dd, hlmea, adaw,
    // ...) park a persistent scratch slot at active index pop_size(), so
    // active_n() is pop_size() + 1 for them. Returning that extra slot hands
    // the caller an unselected offspring dressed as a member of the answer
    // set — enough to skew any IGD or hypervolume computed from it, and
    // invisible unless you count the rows. See data_vault.hpp, "SCRATCH SLOTS
    // AND CONSUMERS".
    r.active_n = std::min<std::size_t>(v.active_n(),
                                       static_cast<std::size_t>(v.pop_size()));
    r.ignored  = std::move(ignored);
    r.evaluations = st().evaluations.load();
    r.generations = static_cast<int>(steps_done);
    r.finished = core_finished(alg);
    for (const auto& u : ops::operators_used()) r.operators.emplace_back(u.op, u.repair);
    if constexpr (has_effective_settings<Core>::value) r.settings = alg.effective_settings();
    if (var_prob) {
        bool used = false;
        for (const auto& u : ops::operators_used()) used = used || std::string(u.op) == "sbx";
        if (!used) r.ignored.emplace_back("sbx_var_prob");
    }
    if (st().ops.on) {
        const auto& rc = ops::repair_counts();
        const double n_vars = static_cast<double>(st().problem->bounds.size());
        const double ev = static_cast<double>(r.evaluations - st().ops.setup_evals);
        r.operator_totals["offspring"] = static_cast<double>(st().ops.t_off);
        if (ev > 0) {
            r.operator_totals["oob_share"] = static_cast<double>(rc.out_children) / ev;
            r.operator_totals["oob_var_share"] = static_cast<double>(rc.out_vars) / (ev * n_vars);
        }
        if (st().ops.t_off > 0) {
            r.operator_totals["survival_share"] = static_cast<double>(st().ops.t_surv) / st().ops.t_off;
            r.operator_totals["offspring_nd_share"] = static_cast<double>(st().ops.t_nd) / st().ops.t_off;
        }
        if (st().ops.t_step_n > 0)
            r.operator_totals["step_mean"] = st().ops.t_step / static_cast<double>(st().ops.t_step_n);
        for (const auto& kv : ops::crossover_stats(true)) r.operator_totals[kv.first] = kv.second;
        for (const auto& kv : mosa_record()) r.operator_totals[kv.first] = kv.second;
        for (const auto& kv : ops::inner_archive_stats(true)) r.operator_totals[kv.first] = kv.second;
    }
    if constexpr (has_inner_archive<Core>::value) {
        for (const auto& m : alg.inner_archive()) {
            r.inner_objectives.push_back(m.f);
            r.inner_variables.push_back(m.x);
        }
    }
    ops::inner_archive_record().live = nullptr;          // the core goes with this scope
    r.objectives.reserve(r.active_n);
    r.variables.reserve(r.active_n);
    const std::size_t rows = r.active_n;
    for (std::size_t i = 0; i < rows; ++i) {
        const auto& f = v.objectives_of(i);
        if (v.is_failed(i)) { --r.active_n; continue; }  // a failure is no answer
        r.objectives.push_back(f);
        r.variables.push_back(v.variables_of(i));
        if (v.bin_vars_n() > 0) r.binary_variables.push_back(v.binary_variables_of(i));
        r.cv.push_back(v.get_cv(i));
        if (st().problem->n_limits > 0) r.limits.push_back(v.limits_of(i));
    }
    r.n_failed = v.failed_n();
    r.eq_tol   = cfg.n_eq > 0 ? v.eq_tol() : 0.0;
    r.unique_evaluations = cfg.unique_evaluations ? cfg.unique_evaluations() : -1;
    r.callback_seconds = st().callback_seconds;
    r.callbacks = st().callbacks;
    r.native_seconds = st().native_seconds;
    return r;
}

// operator_stats(): the operator statistics since the previous call, for the
// run in progress (called from the on_generation observer: the binding's
// operator_stats(), the C++ campaign engine's records). In order: offspring
// evaluated (integer), the shares of them that left the box before repair
// (and of their variables), that survived, that no parent dominates, the mean
// step — none where nothing was counted; the last three need
// RunConfig::operator_stats — then the crossover's, mosa's and the inner
// archive's records.
struct OperatorStat {
    std::string name;
    std::optional<double> value;        // none: nothing counted
    bool integer = false;
};

inline std::vector<OperatorStat> operator_stats()
{
    std::vector<OperatorStat> d;
    RunState& s = st();
    const long long ev = s.evaluations.load() - s.ops.evals_read;
    const auto& rc = ops::repair_counts();
    const double n_vars = s.problem ? static_cast<double>(s.problem->bounds.size()) : 0.0;
    d.push_back({"offspring", static_cast<double>(ev), true});
    if (ev > 0) {
        d.push_back({"oob_share", static_cast<double>(rc.out_children - s.ops.rc_read.out_children) / ev});
        d.push_back({"oob_var_share", n_vars > 0 ? std::optional<double>(static_cast<double>(
                                                       rc.out_vars - s.ops.rc_read.out_vars) / (ev * n_vars))
                                                 : std::nullopt});
    } else {
        d.push_back({"oob_share", std::nullopt});
        d.push_back({"oob_var_share", std::nullopt});
    }
    if (s.ops.on && s.ops.off > 0) {
        d.push_back({"survival_share", static_cast<double>(s.ops.surv) / s.ops.off});
        d.push_back({"offspring_nd_share", static_cast<double>(s.ops.nd) / s.ops.off});
    } else {
        d.push_back({"survival_share", std::nullopt});
        d.push_back({"offspring_nd_share", std::nullopt});
    }
    d.push_back({"step_mean", s.ops.on && s.ops.step_n > 0
                                  ? std::optional<double>(s.ops.step / static_cast<double>(s.ops.step_n))
                                  : std::nullopt});
    // de, de_pbest and portfolio (operators/real_crossover.hpp)
    for (const auto& kv : ops::crossover_stats(false)) d.push_back({kv.first, kv.second});
    // mosa: the annealing's state (algorithms/mosa.hpp)
    for (const auto& kv : mosa_record()) d.push_back({kv.first, kv.second});
    // the inner archive (operators/inner_archive.hpp)
    for (const auto& kv : ops::inner_archive_stats(false)) d.push_back({kv.first, kv.second});
    s.ops.evals_read = s.evaluations.load();
    s.ops.rc_read = rc;
    s.ops.off = s.ops.surv = s.ops.nd = s.ops.step_n = 0;
    s.ops.step = 0.0;
    return d;
}

// ── Dispatch, generated from include/mootation/algorithms.def ───────────────
// One run of the algorithm `name` on `problem`. The problem is the run's
// until it returns; the state of the thread (st()) is cleared afterwards.
inline RunResult run(const std::string& name, RunProblem& problem, const RunConfig& cfg)
{
    if (problem.bounds.empty() && problem.n_bin <= 0)
        throw std::invalid_argument("problem.bounds is empty");

    if (problem.native) {
        const auto& p = *problem.native;
        if (static_cast<int>(problem.bounds.size()) != p.n_var || problem.n_bin != 0 ||
            problem.n_objectives != p.n_obj || problem.n_limits != p.n_con)
            throw std::invalid_argument(
                "problem.native: bounds, n_objectives and n_limits must be the suite "
                "problem's (" + std::to_string(p.n_var) + " variables, " + std::to_string(p.n_obj) +
                " objectives, " + std::to_string(p.n_con) + " constraints), with no binary variables");
    }
    RunState& s = st();
    s.problem = &problem;
    s.evaluations = 0;
    s.callback_seconds = 0.0;
    s.callbacks = 0;
    s.native_seconds = 0.0;
    s.rows = NativeRows{};
    s.decode = NativeDecode{};
    if (problem.native && !cfg.var_kinds.empty()) {
        s.decode.kinds  = cfg.var_kinds;
        s.decode.values = cfg.discrete_values;
        s.decode.values.resize(cfg.var_kinds.size());
        for (const auto& b : problem.bounds) {          // var_types.Layout.core_bounds
            s.decode.lo.push_back(std::ceil(b.first) + 0.0);
            s.decode.hi.push_back(std::floor(b.second));
        }
    }
    ops::reset_operator_records();
    ops::reset_crossover_records();
    mosa_record().clear();
    ops::inner_archive_record() = ops::InnerArchiveRecord{};
    s.ops = OpStats{};
    s.ops.on = cfg.operator_stats;
    struct Guard {
        ~Guard()
        {
            RunState& g = st();
            g.problem = nullptr;
            g.answer_variables = nullptr;
            g.eq_tol = nullptr;
            g.points = nullptr;
            g.cv_level = nullptr;
        }
    } guard;

#define MOOTATION_ALG(KEY, IND, CORE) \
    if (name == #KEY) return run_core<RunTag_##KEY, mootation::CORE<RunTag_##KEY>>(cfg);
#include "algorithms.def"
#undef MOOTATION_ALG

    throw std::invalid_argument("unknown algorithm: " + name);
}

inline std::vector<std::string> algorithm_names()
{
    std::vector<std::string> v;
#define MOOTATION_ALG(KEY, IND, CORE) v.emplace_back(#KEY);
#include "algorithms.def"
#undef MOOTATION_ALG
    return v;
}

}   // namespace mootation::runner
