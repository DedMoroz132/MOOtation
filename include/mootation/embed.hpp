#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// embed.hpp — the two ways to drive MOOtation from another program.
//
// Both take the same `Settings` (settings.hpp) and pick the algorithm by name
// at run time, so switching from NSGA-II to MOEA/D-DE is a one-word change in
// a config file.
//
// ── 1. YOU OWN THE LOOP (ask / tell) ────────────────────────────────────────
// Use this when the objective values come from somewhere the optimizer cannot
// call into: a simulator, a solver process, a measurement rig, another
// language, a job queue.
//
//     mootation::Session s(mootation::Settings::from_file("run.cfg"));
//
//     for (auto X = s.ask(); !X.empty(); X = s.tell(F)) {
//         F.assign(X.size(), std::vector<double>(s.n_objs()));
//         for (std::size_t i = 0; i < X.size(); ++i)
//             F[i] = my_simulator(X[i]);       // X[i] is one candidate's variables
//     }
//     mootation::Result r = s.result();        // the final population
//
// `tell` returns the next batch to evaluate — exactly "give me objective
// values, get back new candidates". An empty batch means the run is over.
//
// BATCH SIZE IS THE ALGORITHM'S CHOICE, not pop_size. Generational algorithms
// (NSGA-II, NSGA-III, ...) hand you a whole offspring generation at once;
// steady-state ones (MOEA/D-DE, MOEA/DD, ...) hand you one candidate at a
// time, because that is how those algorithms are defined. Always size your
// loop off `X.size()`; never assume it.
//
// ── 2. THE OPTIMIZER OWNS THE LOOP (run) ────────────────────────────────────
// Use this when you can just hand over a function.
//
//     auto r = mootation::run(settings,
//         [](const std::vector<std::vector<double>>& X,
//            std::vector<std::vector<double>>& F,
//            std::vector<std::vector<double>>& /*G*/) {
//             for (std::size_t i = 0; i < X.size(); ++i)
//                 F[i] = my_simulator(X[i]);
//         });
//
// F is pre-sized to X.size() x n_objs and G to X.size() x n_cons before the
// call, so the callback only has to fill them in. Same batching rule as above.
//
// ── CONSTRAINTS ─────────────────────────────────────────────────────────────
// Set `n_cons` and `constraints` in Settings, then fill G: one row per
// candidate, `n_cons` values, each <= 0 meaning satisfied. The library derives
// the total violation from those. With n_cons = 0, G is empty and ignored.
//
// ── ONE DRIVER ──────────────────────────────────────────────────────────────
// Both run through run_core.hpp's runner::run, the driver the Python package
// and the C++ campaign engine use: the same Settings give the same run there
// and here (task 17, section 2.5).
//
// ── COST ────────────────────────────────────────────────────────────────────
// Including this header instantiates all 76 algorithms in the translation
// unit, because the dispatch is by string. That is a real compile-time cost
// (tens of seconds). A program that always uses one algorithm can skip it and
// build `Optimizer<Ind, Core>` directly — see docs/writing-an-algorithm.md.
//
// ── THREADING ───────────────────────────────────────────────────────────────
// `run` is single-threaded: the callback is invoked on the calling thread.
// `Session` runs the algorithm on one worker thread and blocks it while you
// evaluate; your code always runs on your own thread and never re-enters the
// library. Sessions are independent — several may run at once — but a single
// Session must be driven from one thread at a time.
// ============================================================================

#include <algorithm>
#include <climits>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "io/population.hpp"
#include "mootation.hpp"
#include "run_core.hpp"
#include "settings.hpp"

namespace mootation {

// ── Result ──────────────────────────────────────────────────────────────────
struct Result {
    std::vector<std::vector<double>> variables;   // [n][n_vars]
    std::vector<std::vector<double>> objectives;  // [n][n_objs]
    std::vector<double>              cv;          // total constraint violation
    // Per-constraint values, `<= 0` meaning satisfied. Empty when n_cons == 0.
    // Kept alongside cv rather than folded into it because cv is a SUM: it
    // cannot be taken apart again, and a population handed to the next run
    // needs the individual values, not the total.
    std::vector<std::vector<double>> limits;      // [n][n_cons]
    std::vector<std::string>         ignored;     // knobs this algorithm lacks
    int                              generations = 0;
    int                              evaluations = 0;

    std::size_t size() const { return objectives.size(); }
};

// The evaluator. X is the batch to evaluate; F and G arrive pre-sized
// (X.size() x n_objs and X.size() x n_cons) and must be filled in place.
using Evaluate = std::function<void(const std::vector<std::vector<double>>& X,
                                    std::vector<std::vector<double>>&       F,
                                    std::vector<std::vector<double>>&       G)>;

namespace embed_detail {

// ── Warm start ──────────────────────────────────────────────────────────────
// Everything that can be wrong with a seed population is checked HERE, once,
// before any algorithm is instantiated — so the message names the file and the
// mismatch instead of surfacing as a length error deep inside setup_with_seed,
// or worse, on a Session's worker thread after the caller has stopped looking.
inline void validate_seed(const io::Population& pop, const Settings& s,
                          const std::string& origin)
{
    const std::string at = "mootation: seed population " + origin + ": ";

    if (static_cast<int>(pop.size()) != s.pop_size)
        throw std::invalid_argument(
            at + "has " + std::to_string(pop.size()) + " individuals but "
            "pop_size is " + std::to_string(s.pop_size) +
            " (set on_size_mismatch to truncate or pad to resize it)");

    if (pop.n_vars() != s.n_vars())
        throw std::invalid_argument(
            at + "has " + std::to_string(pop.n_vars()) + " variables but this "
            "run has " + std::to_string(s.n_vars()) +
            " — it was saved from a different problem");

    if (pop.n_objs() != s.n_objs)
        throw std::invalid_argument(
            at + "has " + std::to_string(pop.n_objs()) + " objectives but this "
            "run has " + std::to_string(s.n_objs) +
            " — it was saved from a different problem");

    // A constrained run cannot be seeded from a file that carries only the
    // aggregate cv: the sum cannot be taken apart into the individual
    // constraint values the algorithms actually compare. Starting anyway would
    // plant a population every one of whose constraints reads as satisfied,
    // which is a silently wrong run rather than a failed one.
    if (s.n_cons > 0) {
        if (pop.limits.empty())
            throw std::invalid_argument(
                at + "this run has " + std::to_string(s.n_cons) + " constraints, "
                "but the file carries no per-constraint columns (g1...). It was "
                "written either by an unconstrained run or by a version before "
                "those columns existed; re-save it from a constrained run");
        if (pop.n_lims() != s.n_cons)
            throw std::invalid_argument(
                at + "has " + std::to_string(pop.n_lims()) + " constraint "
                "values per individual but this run has " +
                std::to_string(s.n_cons));
    }
}

// Load, resize to pop_size under the caller's policy, then check.
inline io::Population load_seed(const Settings& s)
{
    io::Population pop = io::load_population(s.seed_population);
    io::fit_population(pop, s.pop_size, s.on_size_mismatch);
    validate_seed(pop, s, "'" + s.seed_population + "'");
    return pop;
}

// ── The run: run_core.hpp's (task 17, section 2.5) ──────────────────────────
// One driver of a run in the library: the one the Python package and the C++
// campaign engine use (runner::run). Settings become its RunConfig — the
// knobs it names first, then every other core knob and word, as run_core
// applies them — and the evaluator its RunProblem: one call per batch, the
// constraint values of the batch kept for the limits the run asks for next.
inline runner::RunConfig config_of(const Settings& s, const io::Population* seed)
{
    runner::RunConfig c;
    c.pop_size = s.pop_size;
    c.n_gen = s.max_gen;
    c.max_evaluations = s.max_evaluations;
    c.seed = s.seed;
    c.constraint_mode = s.constraints;
    auto num = [&](const char* k) -> std::optional<double> {
        const auto it = s.params.find(k);
        return it == s.params.end() ? std::nullopt : std::optional<double>(it->second);
    };
    auto whole = [&](const char* k) -> std::optional<int> {
        const auto v = num(k);
        return v ? std::optional<int>(static_cast<int>(*v)) : std::nullopt;
    };
    auto word = [&](const char* k) -> std::optional<std::string> {
        const auto it = s.text_params.find(k);
        return it == s.text_params.end() ? std::nullopt : std::optional<std::string>(it->second);
    };
    c.eta_c = num("eta_c");
    c.eta_m = num("eta_m");
    c.pc = num("pc");
    c.pm = num("pm");
    c.T = whole("T");
    c.delta = num("delta");
    c.nr = whole("nr");
    c.kappa = num("kappa");
    c.K = whole("K");
    c.n_clusters = whole("n_clusters");
    c.theta = num("theta");
    c.alpha = num("alpha");
    c.F = num("F");
    c.CR = num("CR");
    c.div = whole("div");
    if (const auto v = num("normalize")) c.normalize = *v != 0.0;
    c.mutation_scale = num("mutation_scale");
    c.mixture_q = num("mixture_q");
    c.blx_alpha = num("blx_alpha");
    c.step_share = num("step_share");
    c.F_spread = num("F_spread");
    c.sbx_var_prob = num("sbx_var_prob");
    c.bound_repair = word("bound_repair");
    c.crossover = word("crossover");
    c.mutation = word("mutation");
    c.dms_init = word("dms_init");
    c.dms_poll = word("dms_poll");
    c.crowding_space = word("crowding_space");
    for (const auto& kv : s.params)
        if (is_core_param(kv.first)) c.params.insert(kv);
    for (const auto& kv : s.text_params)
        if (is_core_word(kv.first)) c.words.insert(kv);
    if (seed) {
        // the objectives come from the population: a warm start costs no
        // evaluation (setup_with_seed); bits only where the population has them
        c.seed_variables = seed->variables;
        c.seed_objectives = seed->objectives;
        c.seed_limits = seed->limits;
        if (!seed->binary_variables.empty() && !seed->binary_variables.front().empty())
            c.seed_binary = seed->binary_variables;
    }
    return c;
}

// the evaluator as a RunProblem's: F returned, G kept for limits_batch, which
// the run calls next on the same rows; the shapes checked
inline runner::RunProblem problem_of(const Settings& s, Evaluate eval)
{
    runner::RunProblem p;
    for (std::size_t i = 0; i < s.lower.size(); ++i) p.bounds.emplace_back(s.lower[i], s.upper[i]);
    p.n_objectives = s.n_objs;
    p.n_limits = s.n_cons;
    const auto G = std::make_shared<std::vector<std::vector<double>>>();
    const int n_objs = s.n_objs, n_cons = s.n_cons;
    p.evaluate_batch = [eval = std::move(eval), G, n_objs, n_cons](std::vector<std::vector<double>> X) {
        std::vector<std::vector<double>> F(X.size(), std::vector<double>(static_cast<std::size_t>(n_objs), 0.0));
        G->assign(n_cons > 0 ? X.size() : 0, std::vector<double>(static_cast<std::size_t>(n_cons), 0.0));
        eval(X, F, *G);
        if (F.size() != X.size())
            throw std::runtime_error("mootation: evaluator returned " + std::to_string(F.size()) +
                                     " objective rows for a batch of " + std::to_string(X.size()));
        for (const auto& row : F)
            if (row.size() != static_cast<std::size_t>(n_objs))
                throw std::runtime_error("mootation: evaluator returned a row of " + std::to_string(row.size()) +
                                         " objectives, expected " + std::to_string(n_objs));
        if (n_cons > 0) {
            if (G->size() != X.size())
                throw std::runtime_error("mootation: n_cons = " + std::to_string(n_cons) +
                                         " but the evaluator returned " + std::to_string(G->size()) +
                                         " constraint rows for a batch of " + std::to_string(X.size()));
            for (const auto& row : *G)
                if (row.size() != static_cast<std::size_t>(n_cons))
                    throw std::runtime_error("mootation: evaluator returned a row of " +
                                             std::to_string(row.size()) + " constraint values, expected " +
                                             std::to_string(n_cons));
        }
        return F;
    };
    if (n_cons > 0)
        p.limits_batch = [G](std::vector<std::vector<double>> X) {
            if (G->size() != X.size())
                throw std::logic_error("mootation: the limits asked for are not the batch just evaluated");
            return *G;
        };
    return p;
}

// One run: on_step(g) after setup (0) and after every step
inline Result run(const Settings& s, Evaluate eval, const io::Population* seed,
                  const std::function<void(int)>& on_step)
{
    runner::RunProblem problem = problem_of(s, std::move(eval));
    runner::RunConfig cfg = config_of(s, seed);
    cfg.on_step = on_step;
    runner::RunResult rr = runner::run(s.algorithm, problem, cfg);
    Result r;
    r.variables = std::move(rr.variables);
    r.objectives = std::move(rr.objectives);
    r.cv = std::move(rr.cv);
    r.limits = std::move(rr.limits);
    r.ignored = std::move(rr.ignored);
    r.generations = rr.generations;
    r.evaluations = static_cast<int>(rr.evaluations);
    return r;
}

} // namespace embed_detail

// ── Public entry points ─────────────────────────────────────────────────────

// Every name accepted by Settings::algorithm.
inline std::vector<std::string> algorithm_names() {
    std::vector<std::string> v;
#define MOOTATION_ALG(KEY, IND, CORE) v.emplace_back(#KEY);
#include "mootation/algorithms.def"
#undef MOOTATION_ALG
    return v;
}

// Settings::validate() cannot do this: settings.hpp deliberately knows nothing
// about the algorithm list. Checking here means a typo in `algorithm` is caught
// when the run is created rather than deep inside the first step — which for a
// Session would surface on the worker thread at the first ask(), long after the
// caller could act on it.
inline void validate_algorithm(const std::string& name) {
    const auto names = algorithm_names();
    for (const auto& n : names)
        if (n == name) return;
    std::string msg = "mootation: unknown algorithm '" + name + "'. Known: ";
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (i) msg += ", ";
        msg += names[i];
    }
    throw std::invalid_argument(msg);
}

// The optimizer owns the loop; `eval` is called once per batch.
//
// Starts from Settings::seed_population when that is set, and from a fresh
// random population otherwise.
inline Result run(const Settings& s, Evaluate eval) {
    s.validate();
    validate_algorithm(s.algorithm);
    if (!eval)
        throw std::invalid_argument("mootation::run: evaluator is empty");

    // Loaded BEFORE the context is installed: a bad seed file must fail as a
    // plain exception from run(), not partway into a configured run.
    std::optional<io::Population> seed;
    if (!s.seed_population.empty()) seed = embed_detail::load_seed(s);

    return embed_detail::run(s, std::move(eval), seed ? &*seed : nullptr, nullptr);
}

// The same, from a population you already have in memory — no file involved.
// Use this when the previous run's Result is still to hand, or when the
// population comes from somewhere that is not a file at all. An explicit
// argument wins over Settings::seed_population.
inline Result run(const Settings& s, Evaluate eval, io::Population seed) {
    s.validate();
    validate_algorithm(s.algorithm);
    if (!eval)
        throw std::invalid_argument("mootation::run: evaluator is empty");

    io::fit_population(seed, s.pop_size, s.on_size_mismatch);
    embed_detail::validate_seed(seed, s, "passed to run()");

    return embed_detail::run(s, std::move(eval), &seed, nullptr);
}

// A Result is what a finished run hands back; this turns it into something the
// next run can start from, without a round trip through a file.
inline io::Population as_population(const Result& r) {
    io::Population p;
    p.variables  = r.variables;
    p.objectives = r.objectives;
    p.cv         = r.cv;
    p.limits     = r.limits;      // empty unless the run had constraints
    p.binary_variables.assign(r.variables.size(), {});
    return p;
}

// ── Session: you own the loop ───────────────────────────────────────────────
// The algorithm runs on a worker thread. Its batch executor blocks there while
// the batch sits in `pending_`; ask() picks it up, tell() posts the answers and
// releases the worker. Your code never runs inside the library.
class Session {
public:
    // Starts from Settings::seed_population when that is set, and from a fresh
    // random population otherwise.
    explicit Session(Settings s) : settings_(std::move(s)) {
        settings_.validate();
        // Before the worker starts: a bad algorithm name must be a constructor
        // failure the caller can catch, not a surprise on the worker thread.
        validate_algorithm(settings_.algorithm);
        if (!settings_.seed_population.empty())
            seed_ = embed_detail::load_seed(settings_);
        init();
    }

    // The same, from a population already in memory — typically
    // as_population(previous.result()) or a file you loaded yourself. An
    // explicit argument wins over Settings::seed_population.
    Session(Settings s, io::Population seed) : settings_(std::move(s)) {
        settings_.validate();
        validate_algorithm(settings_.algorithm);
        io::fit_population(seed, settings_.pop_size, settings_.on_size_mismatch);
        embed_detail::validate_seed(seed, settings_, "passed to Session");
        seed_ = std::move(seed);
        init();
    }

    Session(const Session&)            = delete;
    Session& operator=(const Session&) = delete;

    ~Session() { abort_and_join(); }

    int n_objs() const { return settings_.n_objs; }
    int n_cons() const { return settings_.n_cons; }
    int n_vars() const { return settings_.n_vars(); }
    const Settings& settings() const { return settings_; }

    // The batch waiting to be evaluated. Empty once the run has finished.
    // Safe to call repeatedly; it does not advance anything.
    //
    // Returned BY VALUE deliberately. A reference into pending_ would be
    // handed out without the lock and then overwritten by the worker the
    // moment tell() released it, so `const auto& X = ask(); tell(F); use(X);`
    // would be a data race that happens to work most of the time. The copy is
    // a few kilobytes against a batch of objective evaluations.
    std::vector<std::vector<double>> ask() {
        std::unique_lock<std::mutex> lk(m_);
        cv_main_.wait(lk, [this] { return have_batch_ || finished_; });
        rethrow_if_failed();
        if (!have_batch_) return {};
        return pending_;
    }

    // Post objective values for the batch ask() returned and get the next
    // batch. An empty return means the run is over — call result().
    //   F : one row per candidate, n_objs values.
    //   G : one row per candidate, n_cons values (omit when n_cons == 0);
    //       each value <= 0 means the constraint is satisfied.
    std::vector<std::vector<double>> tell(
            const std::vector<std::vector<double>>& F,
            const std::vector<std::vector<double>>& G = {}) {
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_main_.wait(lk, [this] { return have_batch_ || finished_; });
            rethrow_if_failed();
            if (!have_batch_)
                throw std::logic_error("mootation::Session::tell: the run has "
                                       "already finished");
            if (F.size() != pending_.size())
                throw std::invalid_argument(
                    "mootation::Session::tell: got " + std::to_string(F.size()) +
                    " objective rows for a batch of " + std::to_string(pending_.size()) +
                    " — size the answer off ask().size()");
            for (const auto& row : F)
                if (row.size() != static_cast<std::size_t>(settings_.n_objs))
                    throw std::invalid_argument(
                        "mootation::Session::tell: an objective row has " +
                        std::to_string(row.size()) + " values, expected " +
                        std::to_string(settings_.n_objs));
            if (settings_.n_cons > 0) {
                if (G.size() != pending_.size())
                    throw std::invalid_argument(
                        "mootation::Session::tell: n_cons = " +
                        std::to_string(settings_.n_cons) + " but got " +
                        std::to_string(G.size()) + " constraint rows for a batch of " +
                        std::to_string(pending_.size()));
                for (const auto& row : G)
                    if (row.size() != static_cast<std::size_t>(settings_.n_cons))
                        throw std::invalid_argument(
                            "mootation::Session::tell: a constraint row has " +
                            std::to_string(row.size()) + " values, expected " +
                            std::to_string(settings_.n_cons));
            }
            answer_F_    = F;
            answer_G_    = G;
            have_batch_  = false;
            have_answer_ = true;
        }
        cv_worker_.notify_one();
        return ask();
    }

    // True once the algorithm has run its course.
    bool done() {
        std::unique_lock<std::mutex> lk(m_);
        cv_main_.wait(lk, [this] { return have_batch_ || finished_; });
        rethrow_if_failed();
        return !have_batch_;
    }

    // Generations completed so far.
    int generation() {
        std::lock_guard<std::mutex> lk(m_);
        return generation_;
    }

    // The final population. Blocks until the run ends; throws if it is still
    // waiting for a tell().
    const Result& result() {
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_main_.wait(lk, [this] { return have_batch_ || finished_; });
            rethrow_if_failed();
            if (have_batch_)
                throw std::logic_error("mootation::Session::result: the run is "
                                       "still waiting for tell()");
        }
        join();
        return result_;
    }

private:
    // Thrown into the worker to unwind it when a Session is destroyed early.
    struct Aborted {};

    // Both constructors end here: the context has to be built from the settled
    // settings and the worker started last, once every member it touches is in
    // place.
    void init() {
        start();
    }

    void start() {
        worker_ = std::thread([this] {
            try {
                Result r = embed_detail::run(
                    settings_,
                    [this](const std::vector<std::vector<double>>& X,
                           std::vector<std::vector<double>>&       F,
                           std::vector<std::vector<double>>&       G) { this->handoff(X, F, G); },
                    // seed_ is written in the constructor and only read here,
                    // after the thread is created — no synchronisation needed
                    // because the thread's creation is the happens-before edge.
                    seed_ ? &*seed_ : nullptr,
                    [this](int g) {
                        std::lock_guard<std::mutex> lk(m_);
                        generation_ = g;
                    });
                std::lock_guard<std::mutex> lk(m_);
                result_ = std::move(r);
            } catch (const Aborted&) {
                // Expected during ~Session; nothing to report.
            } catch (...) {
                std::lock_guard<std::mutex> lk(m_);
                failure_ = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lk(m_);
                finished_   = true;
                have_batch_ = false;
            }
            cv_main_.notify_all();
        });
    }

    // Runs on the worker thread, inside the algorithm.
    void handoff(const std::vector<std::vector<double>>& X,
                 std::vector<std::vector<double>>&       F,
                 std::vector<std::vector<double>>&       G) {
        std::unique_lock<std::mutex> lk(m_);
        pending_     = X;
        have_batch_  = true;
        have_answer_ = false;
        lk.unlock();
        cv_main_.notify_one();

        lk.lock();
        cv_worker_.wait(lk, [this] { return have_answer_ || abort_; });
        if (abort_) throw Aborted{};
        F = std::move(answer_F_);
        G = std::move(answer_G_);
        have_answer_ = false;
    }

    void rethrow_if_failed() {
        if (failure_) {
            auto e = failure_;
            failure_ = nullptr;
            std::rethrow_exception(e);
        }
    }

    void join() {
        if (worker_.joinable()) worker_.join();
    }

    void abort_and_join() {
        {
            std::lock_guard<std::mutex> lk(m_);
            abort_ = true;
        }
        cv_worker_.notify_all();
        join();
    }

    Settings                      settings_;
    std::optional<io::Population> seed_;   // empty = fresh random start
    Result                 result_;

    std::thread             worker_;
    std::mutex              m_;
    std::condition_variable cv_main_, cv_worker_;

    std::vector<std::vector<double>> pending_, answer_F_, answer_G_;
    bool               have_batch_  = false;
    bool               have_answer_ = false;
    bool               finished_    = false;
    bool               abort_       = false;
    int                generation_  = 0;
    std::exception_ptr failure_;
};

} // namespace mootation
