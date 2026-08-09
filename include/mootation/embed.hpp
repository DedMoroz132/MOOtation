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
// ── COST ────────────────────────────────────────────────────────────────────
// Including this header instantiates all 60 algorithms in the translation
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

#include "mootation.hpp"
#include "settings.hpp"

namespace mootation {

// ── Result ──────────────────────────────────────────────────────────────────
struct Result {
    std::vector<std::vector<double>> variables;   // [n][n_vars]
    std::vector<std::vector<double>> objectives;  // [n][n_objs]
    std::vector<double>              cv;          // total constraint violation
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

// ── The evaluation context ──────────────────────────────────────────────────
// One per running optimization, reached by Problem<Tag>::calc_objs through a
// thread_local pointer. thread_local rather than global so that independent
// runs on different threads do not collide.
struct Context {
    int      n_objs = 0;
    int      n_cons = 0;
    Evaluate eval;
    std::vector<std::pair<std::optional<double>, std::optional<double>>> bounds;
    int      evaluations = 0;
};

inline Context*& context() {
    static thread_local Context* ctx = nullptr;
    return ctx;
}

struct ContextGuard {
    Context* prev;
    explicit ContextGuard(Context* c) : prev(context()) { context() = c; }
    ~ContextGuard() { context() = prev; }
};

// Evaluate exactly one point through the context. Used by Problem<Tag> for the
// non-batched path; the batched path goes through BatchExecutor below.
inline void eval_one(const std::vector<double>& x,
                     std::vector<double>&       f,
                     std::vector<double>&       g) {
    Context* c = context();
    if (!c)
        throw std::logic_error("mootation: Problem::calc_objs called with no "
                               "evaluation context — use run() or Session");
    std::vector<std::vector<double>> X{x};
    std::vector<std::vector<double>> F(1, std::vector<double>(static_cast<std::size_t>(c->n_objs), 0.0));
    std::vector<std::vector<double>> G;
    if (c->n_cons > 0)
        G.assign(1, std::vector<double>(static_cast<std::size_t>(c->n_cons), 0.0));
    c->eval(X, F, G);
    c->evaluations += 1;
    if (F.size() != 1 || F[0].size() != static_cast<std::size_t>(c->n_objs))
        throw std::runtime_error("mootation: evaluator returned the wrong shape for F");
    f = F[0];
    if (c->n_cons > 0) {
        if (G.size() != 1 || G[0].size() != static_cast<std::size_t>(c->n_cons))
            throw std::runtime_error("mootation: evaluator returned the wrong shape for G");
        g = G[0];
    } else {
        g.clear();
    }
}

// The BatchExecutor installed on every vault: hands the whole dirty set to the
// evaluator in one call. This is what makes ask() return real batches.
inline void run_batch(const BatchRequest& req, BatchResponse& resp) {
    Context* c = context();
    if (!c)
        throw std::logic_error("mootation: batch executor called with no "
                               "evaluation context — use run() or Session");
    resp.resize(req.size(), c->n_objs, c->n_cons);
    c->eval(req.variables, resp.objectives, resp.limits);
    c->evaluations += static_cast<int>(req.size());

    if (resp.objectives.size() != req.size())
        throw std::runtime_error(
            "mootation: evaluator returned " + std::to_string(resp.objectives.size()) +
            " objective rows for a batch of " + std::to_string(req.size()));
    for (const auto& row : resp.objectives)
        if (row.size() != static_cast<std::size_t>(c->n_objs))
            throw std::runtime_error(
                "mootation: evaluator returned a row of " + std::to_string(row.size()) +
                " objectives, expected " + std::to_string(c->n_objs));
    if (c->n_cons > 0) {
        if (resp.limits.size() != req.size())
            throw std::runtime_error(
                "mootation: n_cons = " + std::to_string(c->n_cons) +
                " but the evaluator returned " + std::to_string(resp.limits.size()) +
                " constraint rows for a batch of " + std::to_string(req.size()));
        for (const auto& row : resp.limits)
            if (row.size() != static_cast<std::size_t>(c->n_cons))
                throw std::runtime_error(
                    "mootation: evaluator returned a row of " + std::to_string(row.size()) +
                    " constraint values, expected " + std::to_string(c->n_cons));
    } else {
        resp.limits.clear();
    }
}

// ── One tag + one Problem specialization per algorithm ──────────────────────
// Each core needs its own Individual type, and Based_Individual befriends
// Problem<T> only, so calc_objs has to live in the specialization. The tags
// are generated from the same algorithms.def the tests iterate over.
#define MOOTATION_EMBED_PROBLEM(TAG, BASE)                                     \
    namespace mootation { namespace embed_detail {                             \
        struct TAG : public ::mootation::BASE {};                              \
    } }                                                                        \
    namespace mootation {                                                      \
    template <>                                                                \
    class Problem<embed_detail::TAG> {                                         \
    public:                                                                    \
        int get_vars_n() const {                                               \
            return static_cast<int>(embed_detail::context()->bounds.size());   \
        }                                                                      \
        int get_bin_vars_n() const { return 0; }                               \
        int get_objs_n() const { return embed_detail::context()->n_objs; }     \
        int get_lims_n() const { return embed_detail::context()->n_cons; }     \
        std::vector<std::pair<std::optional<double>, std::optional<double>>>   \
            bounds;                                                            \
        void calc_objs(embed_detail::TAG& ind) const {                         \
            embed_detail::eval_one(ind.variables, ind.objectives, ind.limits); \
        }                                                                      \
    };                                                                         \
    }

} // namespace embed_detail
} // namespace mootation

#define MOOTATION_ALG(KEY, IND, CORE) MOOTATION_EMBED_PROBLEM(EmbedTag_##KEY, IND)
#include "mootation/algorithms.def"
#undef MOOTATION_ALG
#undef MOOTATION_EMBED_PROBLEM

namespace mootation {
namespace embed_detail {

// ── Optional-setter detection ───────────────────────────────────────────────
// A knob is applied when the core has the setter and reported when it does
// not. Silently dropping it would mean running a different configuration than
// the caller asked for.
#define MOOTATION_OPTIONAL_SETTER(NAME, SETTER, TYPE)                          \
    template <typename C, typename = void>                                     \
    struct has_##NAME : std::false_type {};                                    \
    template <typename C>                                                      \
    struct has_##NAME<C, std::void_t<decltype(std::declval<C&>().SETTER(       \
                             std::declval<TYPE>()))>> : std::true_type {};     \
    template <typename C>                                                      \
    bool apply_##NAME(C& core, TYPE v) {                                       \
        if constexpr (has_##NAME<C>::value) { core.SETTER(v); return true; }   \
        else { (void)core; (void)v; return false; }                            \
    }

MOOTATION_OPTIONAL_SETTER(t_max,      set_t_max,      int)
MOOTATION_OPTIONAL_SETTER(eta_c,      set_eta_crossover, double)
MOOTATION_OPTIONAL_SETTER(eta_m,      set_eta_mutation,  double)
MOOTATION_OPTIONAL_SETTER(pc,         set_pc,         double)
MOOTATION_OPTIONAL_SETTER(pm,         set_pm,         double)
MOOTATION_OPTIONAL_SETTER(T,          set_T,          int)
MOOTATION_OPTIONAL_SETTER(delta,      set_delta,      double)
MOOTATION_OPTIONAL_SETTER(nr,         set_nr,         int)
MOOTATION_OPTIONAL_SETTER(kappa,      set_kappa,      double)
MOOTATION_OPTIONAL_SETTER(K,          set_K,          int)
MOOTATION_OPTIONAL_SETTER(n_clusters, set_n_clusters, int)
MOOTATION_OPTIONAL_SETTER(theta,      set_theta,      double)
MOOTATION_OPTIONAL_SETTER(alpha,      set_alpha,      double)
MOOTATION_OPTIONAL_SETTER(F,          set_F,          double)
MOOTATION_OPTIONAL_SETTER(CR,         set_CR,         double)
MOOTATION_OPTIONAL_SETTER(div,        set_div,        int)

#undef MOOTATION_OPTIONAL_SETTER

template <typename Core>
inline std::vector<std::string> apply_knobs(Core& alg, const Settings& s) {
    std::vector<std::string> ignored;
    auto get = [&](const char* k) -> const double* {
        auto it = s.params.find(k);
        return it == s.params.end() ? nullptr : &it->second;
    };
    auto note = [&](bool applied, const char* name) {
        if (!applied) ignored.emplace_back(name);
    };
    if (auto* v = get("eta_c"))      note(apply_eta_c(alg, *v), "eta_c");
    if (auto* v = get("eta_m"))      note(apply_eta_m(alg, *v), "eta_m");
    if (auto* v = get("pc"))         note(apply_pc(alg, *v), "pc");
    if (auto* v = get("pm"))         note(apply_pm(alg, *v), "pm");
    if (auto* v = get("T"))          note(apply_T(alg, static_cast<int>(*v)), "T");
    if (auto* v = get("delta"))      note(apply_delta(alg, *v), "delta");
    if (auto* v = get("nr"))         note(apply_nr(alg, static_cast<int>(*v)), "nr");
    if (auto* v = get("kappa"))      note(apply_kappa(alg, *v), "kappa");
    if (auto* v = get("K"))          note(apply_K(alg, static_cast<int>(*v)), "K");
    if (auto* v = get("n_clusters")) note(apply_n_clusters(alg, static_cast<int>(*v)), "n_clusters");
    if (auto* v = get("theta"))      note(apply_theta(alg, *v), "theta");
    if (auto* v = get("alpha"))      note(apply_alpha(alg, *v), "alpha");
    if (auto* v = get("F"))          note(apply_F(alg, *v), "F");
    if (auto* v = get("CR"))         note(apply_CR(alg, *v), "CR");
    if (auto* v = get("div"))        note(apply_div(alg, static_cast<int>(*v)), "div");
    return ignored;
}

// ── The generic run body, one instantiation per algorithm ───────────────────
// `hook` is called once after setup and once after every step, so Session can
// drive the loop generation by generation without duplicating this function.
template <typename Tag, typename Core>
inline Result run_core(const Settings& s, Context& ctx,
                       const std::function<void(int)>& per_generation) {
    Problem<Tag> prob;
    prob.bounds = ctx.bounds;

    DataVault<Tag> vault(s.pop_size, prob);
    vault.set_batch_executor(&run_batch);

    Optimizer<Tag, Core> opt(std::move(vault), defer_setup);
    Core& alg = opt.get_algorithm();

    alg.set_seed(s.seed);
    alg.constraint_mode = s.constraints;
    // t_max is a schedule input, not a user knob: several cores anneal against
    // it, and their 1000-generation default against a 250-generation budget is
    // not the paper's schedule. Always pass the real budget where it is taken.
    apply_t_max(alg, s.max_gen);

    Result r;
    r.ignored = apply_knobs(alg, s);

    opt.setup();
    if (per_generation) per_generation(0);
    for (int g = 1; g <= s.max_gen; ++g) {
        opt.step();
        r.generations = g;
        if (per_generation) per_generation(g);
    }

    auto& v = opt.get_vault();
    // Steady-state cores park a scratch slot at active index pop_size(); the
    // answer set is [0, pop_size()), never [0, active_n()).
    std::size_t n = std::min<std::size_t>(v.active_n(),
                                          static_cast<std::size_t>(v.pop_size()));
    r.variables.reserve(n);
    r.objectives.reserve(n);
    r.cv.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        r.variables.push_back(v.variables_of(i));
        r.objectives.push_back(v.objectives_of(i));
        r.cv.push_back(v.get_cv(i));
    }
    r.evaluations = ctx.evaluations;
    return r;
}

inline Result dispatch(const Settings& s, Context& ctx,
                       const std::function<void(int)>& per_generation) {
#define MOOTATION_ALG(KEY, IND, CORE)                                          \
    if (s.algorithm == #KEY)                                                   \
        return run_core<embed_detail::EmbedTag_##KEY,                          \
                        CORE<embed_detail::EmbedTag_##KEY>>(s, ctx, per_generation);
#include "mootation/algorithms.def"
#undef MOOTATION_ALG
    throw std::invalid_argument("mootation: unknown algorithm '" + s.algorithm +
                                "' — see algorithm_names()");
}

inline Context make_context(const Settings& s, Evaluate eval) {
    Context ctx;
    ctx.n_objs = s.n_objs;
    ctx.n_cons = s.n_cons;
    ctx.eval   = std::move(eval);
    ctx.bounds.reserve(s.lower.size());
    for (std::size_t i = 0; i < s.lower.size(); ++i)
        ctx.bounds.emplace_back(std::optional<double>(s.lower[i]),
                                std::optional<double>(s.upper[i]));
    return ctx;
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
inline Result run(const Settings& s, Evaluate eval) {
    s.validate();
    validate_algorithm(s.algorithm);
    if (!eval)
        throw std::invalid_argument("mootation::run: evaluator is empty");
    embed_detail::Context ctx = embed_detail::make_context(s, std::move(eval));
    embed_detail::ContextGuard guard(&ctx);
    return embed_detail::dispatch(s, ctx, nullptr);
}

// ── Session: you own the loop ───────────────────────────────────────────────
// The algorithm runs on a worker thread. Its batch executor blocks there while
// the batch sits in `pending_`; ask() picks it up, tell() posts the answers and
// releases the worker. Your code never runs inside the library.
class Session {
public:
    explicit Session(Settings s) : settings_(std::move(s)) {
        settings_.validate();
        // Before the worker starts: a bad algorithm name must be a constructor
        // failure the caller can catch, not a surprise on the worker thread.
        validate_algorithm(settings_.algorithm);
        ctx_ = embed_detail::make_context(settings_,
            [this](const std::vector<std::vector<double>>& X,
                   std::vector<std::vector<double>>&       F,
                   std::vector<std::vector<double>>&       G) {
                this->handoff(X, F, G);
            });
        start();
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

    void start() {
        worker_ = std::thread([this] {
            embed_detail::ContextGuard guard(&ctx_);
            try {
                Result r = embed_detail::dispatch(
                    settings_, ctx_,
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

    Settings               settings_;
    embed_detail::Context  ctx_;
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
