#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Search strategies: a strategy takes an inner solver (an algorithm of
// scalar::run, or a strategy) and is a solver itself (task 18, stage 3, §3.1
// and §3.2; addendum 1, §3.4). Included at the end of scalar.hpp.
//
// THE INNER SOLVER is a knob whose value is a JSON object, its "solver" the
// name and its other members the solver's knobs: inner = {"solver": "cmaes",
// "cma_sigma0": 0.2}. The config and minimize() hand it over as a word, the
// object's canonical text (keys sorted, no spaces).
//
// ROUNDS. The budget goes by synchronous rounds: a round is a set of starts,
// each with its allowance and its generator. Every start of a round runs the
// inner solver on a view of the problem (View): a point the run's archive
// held at the round's beginning is answered from it, free; a new one is
// evaluated and kept by the view. When the round is over the views' points
// go into the archive in the order of the starts, and a point an earlier
// start of the round brought costs nothing again (Evaluator::adopt). The
// archive does not change during a round, so its starts may run in any
// number of threads: 1 thread and 12 give the same run, bit for bit.
//
// THE DEFERRED COUNT. What a problem's functions do besides evaluating (a
// campaign job's count of the evaluations and their records) is not
// thread-safe: a start in a thread of its own (a round in threads > 1) hands
// it to defer_or_run(), which holds it in the start's list; when the starts
// are done, the round runs the lists start by start, in the order of the
// starts, which is the order one thread runs them in. So any number of
// threads gives the count of 1 thread, bit for bit. Elsewhere (one thread, or
// the run outside a round) defer_or_run() does it at once. A held call keeps
// the box's repair counts its start had made so far (ops::repair_counts, a
// thread's own); the round sets the counter to what 1 thread shows at that
// call (the earlier starts' totals, then this one's so far) when it runs it,
// and leaves it at every start's total (SCALAR-15).
//
// THE ANSWER is every algorithm's (evaluator.hpp): at one objective the best
// point of the archive. Each start's own answer goes to the minima log, which
// groups the starts that found the same minimum (min_dist, min_df).
//
// THE STRATEGIES (and ipop, bipop, bipop_2009: restarts.hpp; halving, mlsl,
// basin_hopping, ils, memetic: search.hpp):
//   tiktak      Arnoud, Guvenen, Kleineberg's TikTak (tiktak.hpp; task 19).
//   data_continuation  stages of the residuals' weights (continuation.hpp; task 19).
//   pap         Peng, Tang, Chen & Yao's algorithm portfolio (pap.hpp; task 19).
//   multistart  n_starts starts in one round, the budget shared evenly:
//               start 0 is the inner solver's own (its plan, or x0); start k
//               >= 1 from point k of a design (start_points: sobol, lhs,
//               uniform, halton_rand, sobol_owen) in the box. Knobs: inner,
//               n_starts (10), start_points (sobol), threads (1), min_dist
//               (1e-3), min_df (1e-8).
//
// READINGS (declared):
//   STRAT-1. A start's generator: the run's seed for start 0, else a 32-bit
//     word of std::seed_seq{seed, k} (the standard defines its words), so
//     one start is the inner solver's own run, bit for bit (§3.1's test).
//   STRAT-2. The inner solver counts what it asks for against its allowance,
//     the points it gets free from the archive included; the strategy's
//     budget counts only what is new. A start that meets old points thus
//     stops before the run has spent its allowance.
//   STRAT-3. The budget shared evenly: start k's allowance is budget / n,
//     the first budget mod n starts one more.
//   STRAT-10. start_points = around_x0 (task 24, §24.2.5; off by default):
//     start 0 is the start (x0, which it needs), start k >= 1 is x0 + N(0,
//     sigma^2) coordinate by coordinate, sigma = start_sigma times the
//     coordinate's width where the box is finite, start_sigma itself where
//     it is not, clipped to the box; the normal draws from mt19937(seed),
//     start by start, coordinate by coordinate. start_sigma = 0: every start
//     is x0. allowance and start_sigma go to the settings only where given.
//   STRAT-12. keep_points = best (task 24, §24.3.3; off by default: all):
//     the run's evaluator keeps its best point and its last (evaluator.hpp
//     THE LEAN MODE) — multistart, ils, basin_hopping, mlsl, tiktak; a
//     start's own run keeps its points as before. A round then runs its
//     starts kLeanChunk at a time, each chunk merged before the next starts,
//     so that no more than a chunk's points are held at once; the chunk does
//     not depend on threads (1 thread = N threads). A point met by two
//     starts is counted in both (there is no archive to find it in).
//   STRAT-13. minima_bayes = on (task 24, §24.2.5; off by default): the
//     settings get Boender's posterior expectations (Rinnooy Kan & Timmer
//     1987, part I, Proposition 2): the number of minima w (N - 1) / (N - w -
//     2) (11), "inf" where N <= w + 2, and the share of the space in the
//     regions of attraction not yet found w (w + 1) / (N (N - 1)) (12) — w the
//     minima log's minima, N the local searches that gave an answer
//     (multistart, tiktak), the reduced sample gamma k N for mlsl (part I,
//     §3). The proposition's prior: uniform starting points; at a design
//     (sobol, lhs) or about x0 they are an indication.
//   STRAT-14. start_filter = oqnlp (task 24, §24.2.5; off by default: every
//     start runs): Ugray, Lasdon, Plummer, Glover, Kelly & Marti 2007, §3.4,
//     Table 2. multistart's starts become trial points, one after another:
//     the first oq_stage1 (200) are evaluated, the inner solver runs from
//     the best; then each later trial point is evaluated and the inner
//     solver runs from it only if it passes both filters — the distance
//     filter (its Euclidean distance to every local solution found is at
//     least oq_distfactor (0.75) times that solution's maxdist, the largest
//     distance from a start that led to it) and the merit filter (its P1 at
//     most the threshold, which becomes its P1); after oq_waitcycle (20) trial
//     points in a row above the threshold, threshold += oq_threshfactor (0.2)
//     * (1 + |threshold|). The threshold starts at the first solution's P1
//     (feasible), else at the best stage 1 point's. Each run of the inner
//     solver has all the budget left (STRAT-11). Readings: OQ-1 the
//     threshold's rule of §3.4, not §5's; OQ-2 §5's proposals (automatic
//     threshfactor, disjoint spheres) not made; OQ-3 P1 = f + cv (5) with
//     the weights 1: the library's solvers give no multipliers (the weights'
//     positive lower limit); OQ-4 the run of rejections restarts after an
//     increase and at a point not above the threshold; OQ-5 a local solution
//     is the minima log's (min_dist, min_df), the same minimum found again
//     raises its maxdist; an infeasible one is not kept; OQ-6 the trial
//     points are start 0's x0, where the start has one, and the design's;
//     OQ-7 the number of searches is not the paper's: on the six-hump
//     camelback, 1000 uniform trial points start 22 searches where the
//     paper's 846 OptQuest points start 9 (§4, Figure 2). The likely causes:
//     the trial points (scatter search crowds them about good points, where
//     the distance filter rejects them; uniform points reach every basin,
//     and the merit filter's threshold rises over the bad ones), and maxdist
//     (a minimum found from one start has the radius of that start only, so
//     the distance filter is weak until a basin was entered twice). The
//     filters are §3.4's; the comparison is of the order, a few per cent.
//   STRAT-11. allowance = sequential (task 24, §24.2.5; off by default:
//     even, STRAT-3) runs multistart's starts one after another, each with
//     all the budget left (its floor), until the budget or the target is
//     spent; one thread, and the starts' points are the same.
//   STRAT-4. The minima log: a start's answer joins the first minimum within
//     min_dist (the box normalized to [0, 1]^n where it is finite, else the
//     problem's coordinates; the largest coordinate difference) and min_df in
//     f, else is a minimum of its own; a minimum keeps its best point.
//   STRAT-5. The run's equalities' tolerance schedule (eq_tol0) is the inner
//     solver's, on its own allowance.
//   STRAT-6. A start may spend on new points at most what it was given (its
//     allowance, or less where a strategy reruns a start whose earlier
//     points are free, search.hpp SRCH-6): past it the view ends the inner
//     run as the budget does, so a round never spends past the run's budget.
//   STRAT-7. A round within a start (a strategy as another's inner solver)
//     runs its starts in the start's own thread, one after another, whatever
//     its threads knob: its views reach the outer start's view, which one
//     thread owns. The run is the same (1 thread and 12 give the same run).
//   STRAT-8. Every strategy ends below its budget too where its points cost
//     nothing (the archive answers them: points clipped onto known ones, a
//     box of zero width): a loop over the budget ends when a pass of it
//     spends nothing. multistart is one round and halving ceil(log2 n); ipop,
//     bipop and bipop_2009 end on a run that evaluates nothing (RST-8), each
//     CMA-ES run on its own criteria (max_iter among them); memetic on a phase
//     whose global run spends nothing (SRCH-9); mlsl on an iteration that
//     spends nothing, ils and basin_hopping on 100 steps in a row that spend
//     nothing (SRCH-10; one free step, clipped onto a known point, is fine);
//     tiktak is its n_star starts, data_continuation its stages, and pap ends
//     on POP-2's generations in a row with no new point.
//     test_search runs every strategy, with nelder_mead, cmaes and de inside,
//     on a minimum in the box's corner, on CEC 2013 F1 (its minima on the
//     bounds) and in a box of zero width (data_continuation, for least
//     squares only, aside).
//   STRAT-9. min_sym = off | canon | align (task 19, §19.2.2, S5; off by
//     default): the log's distance between a start's answer and a minimum
//     taken after the answer is aligned to the minimum (align), or both are
//     canonical (canon), the problem's symmetries' (symmetry.hpp; its
//     sym_blocks, canon, align), so that minima equal but for a permutation
//     of the blocks are one. In the settings only where given.
// ============================================================================

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../designs.hpp"
#include "../json.hpp"
#include "scalar.hpp"

namespace mootation::scalar {

namespace strategy {

// STRAT-1
inline unsigned start_seed(unsigned seed, std::uint64_t k)
{
    if (k == 0) return seed;
    std::seed_seq q{seed, static_cast<unsigned>(k & 0xFFFFFFFFu), static_cast<unsigned>(k >> 32)};
    std::uint32_t w[1];
    q.generate(w, w + 1);
    return w[0];
}

// One start's road to the problem: the archive at the round's beginning, read
// only, and the start's own new points.
// THE DEFERRED COUNT: this thread's list, while it runs a start of a round
// in threads; none elsewhere
struct DeferredCall {
    std::function<void()> what;
    ops::RepairCounts at;             // the start's repair counts at the call
};
using Deferred = std::vector<DeferredCall>;

inline Deferred*& deferred_here()
{
    thread_local Deferred* list = nullptr;
    return list;
}

// this thread's repair counts when its start began
inline ops::RepairCounts& start_counts0()
{
    thread_local ops::RepairCounts c;
    return c;
}

// how many starts this thread is running within one another (STRAT-7)
inline int& start_depth()
{
    thread_local int depth = 0;
    return depth;
}

inline void defer_or_run(std::function<void()> what)
{
    if (Deferred* list = deferred_here()) list->push_back({std::move(what), ops::repair_counts() - start_counts0()});
    else what();
}

class View {
public:
    using Vec = std::vector<double>;
    Problem problem;                  // what the inner run evaluates: every call through here

    struct Known {
        Vec x, f, g, r, gf;
        Rows J;
        bool bad = false, has_grad = false;
        int verdict = -1;             // accept_success's answer here (evaluator.hpp SUCC-3); -1: not asked
    };
    std::vector<Known> own;           // the start's points, in the order it met them

    // fresh: what the start may spend on new points (evaluations + grad_cost
    // * gradient calls); past it a call throws BudgetSpent, which ends the
    // inner run as its budget would (STRAT-6)
    explicit View(const Evaluator& parent, double fresh = std::numeric_limits<double>::infinity())
        : parent_(parent), P_(parent.P), fresh_(fresh)
    {
        problem = P_;
        problem.cache = Problem::Cache::off;   // THE CACHE is the view's (evaluator.hpp CACHE-4)
        problem.blocks.clear();       // r~ comes scaled already (SO-1)
        problem.weights.clear();
        if (problem.n_res == 0) problem.n_res = parent.n_res;
        problem.evaluate = nullptr;
        problem.gradient = nullptr;
        problem.value_grad = nullptr;
        problem.residuals = nullptr;
        problem.jacobian = nullptr;
        problem.residuals_jac = nullptr;
        if (P_.evaluate)
            problem.evaluate = [this](const Vec& x, Vec& f, Vec& g) {
                const Known& k = get(x, Call::values);
                f = k.f;
                g = k.g;
            };
        if (P_.value_grad)
            problem.value_grad = [this](const Vec& x, Vec& f, Vec& g, Vec& gf, Rows& J) {
                const Known& k = get(x, Call::joint);
                f = k.f;
                g = k.g;
                gf = k.gf;
                J = k.J;
            };
        if (P_.gradient)
            problem.gradient = [this](const Vec& x, Vec& gf, Rows& J) {
                const Known& k = get(x, Call::grad);
                gf = k.gf;
                J = k.J;
            };
        if (P_.residuals)
            problem.residuals = [this](const Vec& x, Vec& r) { r = get(x, Call::values).r; };
        if (P_.residuals_jac)
            problem.residuals_jac = [this](const Vec& x, Vec& r, Rows& J) {
                const Known& k = get(x, Call::joint);
                r = k.r;
                J = k.J;
            };
        if (P_.jacobian)
            problem.jacobian = [this](const Vec& x, Rows& J) { J = get(x, Call::grad).J; };
        problem.known_point = nullptr;
        if (P_.m == 1 && P_.n_cons == 0 && !P_.has_residuals())
            problem.known_point = [this](const Vec& x, const Vec& f) {      // planted (scalar.hpp ULS-2)
                const std::string key = key_of(x);
                if (where_.count(key)) return;
                Known k;
                k.x = x;
                k.f = f;
                for (double v : f) k.bad = k.bad || !std::isfinite(v);
                own.push_back(std::move(k));
                where_[key] = own.size() - 1;
                events_.push_back({own.size() - 1, 0, 0});
            };
        // THE SUCCESS (evaluator.hpp SUCC-3): the start's run asks here; the
        // cost goes to the start's allowance, the verdict to the merge
        if (P_.accept_success) {
            problem.success_cost = 0.0;
            problem.accept_success = [this](const Vec& x) {
                const auto it = where_.find(key_of(x));
                if (it == where_.end()) return P_.accept_success(x);     // not a point of this start
                Known& k = own[it->second];
                if (k.verdict < 0) {
                    k.verdict = P_.accept_success(x) ? 1 : 0;
                    spent_ += P_.success_cost;
                }
                return k.verdict == 1;
            };
        }
    }

    View(const View&) = delete;
    View& operator=(const View&) = delete;

    // the start's calls into the run's archive (the end of the round), in
    // the order it made them: the counts of every record as the start's own
    void merge_into(Evaluator& ev) const
    {
        for (const Event& e : events_) {
            const Known& k = own[e.at];
            const bool g = e.grads > 0;
            ev.adopt(k.x, k.f, k.g, k.r, k.bad, e.evals, e.grads, g ? &k.gf : nullptr, g ? &k.J : nullptr,
                     k.verdict);
        }
    }

private:
    enum class Call { values, grad, joint };
    struct Event {                    // a call of the problem: an evaluation, a gradient, or both
        std::size_t at;
        int evals, grads;
    };
    const Evaluator& parent_;
    const Problem& P_;
    std::unordered_map<std::string, Evaluator::Cached> own_canon_;   // THE CACHE: this start's structures
    long long cache_hits_ = 0;

    // the values of x's structure: the run's (at the round's start) or this start's
    const Evaluator::Cached* cached(const Vec& x) const
    {
        if (P_.cache != Problem::Cache::canon) return nullptr;
        const std::string k = canon_key_of(P_, x);
        if (const Evaluator::Cached* c = parent_.cached_key(k)) return c;
        const auto it = own_canon_.find(k);
        return it == own_canon_.end() ? nullptr : &it->second;
    }

public:
    long long cache_hits() const { return cache_hits_; }

private:
    std::unordered_map<std::string, std::size_t> where_;
    std::vector<Event> events_;
    double fresh_, spent_ = 0.0;

    void charge(double cost)
    {
        if (spent_ + cost > fresh_) throw BudgetSpent{};
        spent_ += cost;
    }

    static std::string key_of(const Vec& x)
    {
        std::string k(x.size() * sizeof(double), '\0');
        if (!x.empty()) std::memcpy(&k[0], x.data(), k.size());
        return k;
    }

    // r (or J's rows) of block b times sqrt(w_b), as the evaluator scales them
    template <typename RowOp>
    void by_block(std::size_t n_res, RowOp op) const
    {
        if (P_.weights.empty()) return;
        std::size_t at = 0;
        for (std::size_t b = 0; b < P_.weights.size(); ++b) {
            const std::size_t size = P_.blocks.empty() ? n_res : static_cast<std::size_t>(P_.blocks[b]);
            const double s = std::sqrt(P_.weights[b]);
            for (std::size_t k = 0; k < size; ++k) op(at + k, s);
            at += size;
        }
    }

    void values(Known& k, std::size_t at, bool joint)
    {
        charge(joint ? 1.0 + parent_.grad_cost : 1.0);
        k.f.assign(static_cast<std::size_t>(P_.m), 0.0);
        k.g.assign(static_cast<std::size_t>(P_.n_cons), 0.0);
        if (P_.has_residuals()) {
            if (joint) P_.residuals_jac(k.x, k.r, k.J);
            else P_.residuals(k.x, k.r);
            by_block(k.r.size(), [&](std::size_t i, double s) { k.r[i] *= s; });
            double sum = 0.0;
            for (double v : k.r) sum += v * v;
            k.f.assign(1, 0.5 * sum);
        } else if (joint) {
            P_.value_grad(k.x, k.f, k.g, k.gf, k.J);
        } else {
            P_.evaluate(k.x, k.f, k.g);
        }
        for (double v : k.f) k.bad = k.bad || !std::isfinite(v);
        for (double v : k.g) k.bad = k.bad || !std::isfinite(v);
        if (joint) finish_grad(k);
        events_.push_back({at, 1, joint ? 1 : 0});
    }

    void finish_grad(Known& k)
    {
        if (P_.has_residuals()) {
            by_block(k.r.size(), [&](std::size_t i, double s) {
                for (double& v : k.J[i]) v *= s;
            });
            k.gf.assign(k.x.size(), 0.0);                  // grad f = J~^T r~
            for (std::size_t a = 0; a < k.r.size(); ++a)
                for (std::size_t j = 0; j < k.gf.size(); ++j) k.gf[j] += k.J[a][j] * k.r[a];
        }
        k.has_grad = true;
    }

    void gradient(Known& k, std::size_t at)
    {
        charge(parent_.grad_cost);
        if (P_.has_residuals()) {
            if (P_.jacobian) {
                P_.jacobian(k.x, k.J);
            } else {
                Vec r;
                P_.residuals_jac(k.x, r, k.J);
            }
        } else if (P_.gradient) {
            P_.gradient(k.x, k.gf, k.J);
        } else {
            Vec f(static_cast<std::size_t>(P_.m)), g(static_cast<std::size_t>(P_.n_cons));
            P_.value_grad(k.x, f, g, k.gf, k.J);
        }
        finish_grad(k);
        events_.push_back({at, 0, 1});
    }

    const Known& get(const Vec& x, Call call)
    {
        const std::string key = key_of(x);
        auto it = where_.find(key);
        if (it == where_.end()) {
            Known k;
            k.x = x;
            if (parent_.known(x)) {                        // the archive's, free
                const std::size_t i = parent_.find(x);
                k.f = parent_.F[i];
                k.g = parent_.G[i];
                k.r = parent_.R[i];
                k.bad = parent_.failed[i] != 0;
                if (parent_.has_grad(i)) {
                    k.gf = parent_.grad(i);
                    k.J = parent_.jac(i);
                    k.has_grad = true;
                }
            } else if (const Evaluator::Cached* c = cached(x)) {   // THE CACHE (CACHE-4): free
                k.f = c->f;
                k.g = c->g;
                k.r = c->r;
                k.bad = c->bad;
                events_.push_back({own.size(), 0, 0});
                ++cache_hits_;
            } else {
                values(k, own.size(), call == Call::joint);
                if (P_.cache == Problem::Cache::canon)
                    own_canon_.emplace(canon_key_of(P_, x), Evaluator::Cached{k.f, k.g, k.r, k.bad});
            }
            it = where_.emplace(key, own.size()).first;
            own.push_back(std::move(k));
        }
        Known& k = own[it->second];
        if (call != Call::values && !k.has_grad) gradient(k, it->second);
        return k;
    }
};

// The inner solver of a knob: its name and knobs (THE INNER SOLVER).
struct Inner {
    std::string name;
    Knobs knobs;
};

inline Inner inner_of(const std::string& knob, const Knob& v)
{
    if (!v.is_word || v.word.empty() || v.word.front() != '{')
        throw std::invalid_argument(knob + " must be a solver: {\"solver\": name, its knobs ...}");
    const json::Value o = json::parse(v.word);
    const json::Value* s = o.find("solver");
    if (o.kind != json::Value::Kind::Object || s == nullptr || s->kind != json::Value::Kind::String)
        throw std::invalid_argument(knob + " must name its solver: {\"solver\": name, ...}");
    Inner in;
    in.name = s->s;
    for (std::size_t k = 0; k < o.keys.size(); ++k) {
        if (o.keys[k] == "solver") continue;
        const json::Value& x = o.items[k];
        Knob kv;
        if (x.kind == json::Value::Kind::String) {
            kv.is_word = true;
            kv.word = x.s;
        } else if (x.kind == json::Value::Kind::Int) {
            kv.number = static_cast<double>(x.i);
        } else if (x.kind == json::Value::Kind::Float) {
            kv.number = x.f;
        } else if (x.kind == json::Value::Kind::Object) {
            kv.is_word = true;                            // a solver within: its canonical text
            kv.word = json::dump(x, -1, ",", ":");
        } else {
            throw std::invalid_argument(knob + "." + o.keys[k] + ": a word, a number or a solver");
        }
        in.knobs.push_back({o.keys[k], kv});
    }
    return in;
}

// What a start runs: its allowance (what the inner solver counts), its
// generator's seed, its start, and what it may spend on new points (fresh;
// -1: its allowance).
struct StartSpec {
    long long allowance = 0;
    unsigned seed = 0;
    Start start;
    long long fresh = -1;
};

// One round (ROUNDS): every start's inner run on its view, in `threads`
// threads, then the views merged in the order of the starts. Returns each
// start's result.
inline constexpr std::size_t kLeanChunk = 64;   // STRAT-12

inline std::vector<Result> round(Evaluator& ev, const Inner& inner, const std::vector<StartSpec>& starts,
                                 int threads, int pop_size)
{
    const std::size_t n = starts.size();
    if (ev.lean && n > kLeanChunk) {                  // STRAT-12: a chunk at a time
        std::vector<Result> all;
        for (std::size_t c = 0; c < n; c += kLeanChunk) {
            const std::vector<StartSpec> chunk(starts.begin() + static_cast<std::ptrdiff_t>(c),
                                               starts.begin() + static_cast<std::ptrdiff_t>(std::min(n, c + kLeanChunk)));
            for (Result& r : round(ev, inner, chunk, threads, pop_size)) all.push_back(std::move(r));
        }
        return all;
    }
    std::vector<std::unique_ptr<View>> views;
    for (std::size_t k = 0; k < n; ++k)
        views.push_back(std::make_unique<View>(
            ev, static_cast<double>(starts[k].fresh >= 0 ? starts[k].fresh : starts[k].allowance)));
    std::vector<Result> out(n);
    std::vector<std::exception_ptr> error(n);
    std::vector<Deferred> later(n);                   // THE DEFERRED COUNT, start by start
    std::vector<ops::RepairCounts> made(n);            // each start's repair counts
    const double eq_tol = ev.eq_tol_end, eq_tol0 = ev.eq_tol0;
    const bool has_tol0 = ev.has_tol0;
    auto one = [&](std::size_t k) {
        ++start_depth();
        try {
            if (starts[k].allowance > 0)
                out[k] = run(inner.name, views[k]->problem, eq_tol, has_tol0, eq_tol0, pop_size,
                             starts[k].allowance, starts[k].seed, inner.knobs, {}, 0, starts[k].start);
        } catch (...) {
            error[k] = std::current_exception();
        }
        --start_depth();
    };
    const std::size_t T = start_depth() > 0 ? 1                                       // STRAT-7
                                            : std::min<std::size_t>(n, static_cast<std::size_t>(std::max(1, threads)));
    if (T <= 1) {
        for (std::size_t k = 0; k < n; ++k) one(k);
    } else {
        std::atomic<std::size_t> next{0};
        std::vector<std::thread> pool;
        for (std::size_t t = 0; t < T; ++t)
            pool.emplace_back([&]() {
                for (std::size_t k = next++; k < n; k = next++) {
                    deferred_here() = &later[k];
                    start_counts0() = ops::repair_counts();
                    one(k);
                    made[k] = ops::repair_counts() - start_counts0();
                    deferred_here() = nullptr;
                }
            });
        for (auto& th : pool) th.join();
        const ops::RepairCounts base = ops::repair_counts();
        ops::RepairCounts before{};                   // the earlier starts' totals
        for (std::size_t k = 0; k < n; ++k) {         // in the order of the starts (a round within a
            for (DeferredCall& call : later[k]) {     // start's thread: into its list)
                ops::repair_counts() = base;
                ops::repair_counts() += before;
                ops::repair_counts() += call.at;
                defer_or_run(std::move(call.what));
            }
            before += made[k];
        }
        ops::repair_counts() = base;
        ops::repair_counts() += before;
    }
    for (std::size_t k = 0; k < n; ++k)
        if (error[k]) std::rethrow_exception(error[k]);
    for (std::size_t k = 0; k < n; ++k) {
        views[k]->merge_into(ev);
        ev.hook_calls += out[k].hook_calls;           // the user's local search (scalar.hpp ULS-5)
        ev.hook_spent += out[k].hook_spent;
        ev.hook_seconds += out[k].hook_seconds;
        ev.cache_hits += views[k]->cache_hits();          // THE CACHE (evaluator.hpp CACHE-4)
        ev.delta_calls += out[k].delta_calls;             // THE DELTA
        ev.delta_spent += out[k].delta_spent;
    }
    return out;
}

// The minima log (STRAT-4).
struct Minimum {
    Vec x;
    double f = 0.0;
    long long start = 0;              // the start that found its best point
    long long found = 0;              // how many starts found it
};

class MinimaLog {
public:
    MinimaLog(const Vec& lo, const Vec& hi, double min_dist, double min_df)
        : lo_(lo), hi_(hi), dist_(min_dist), df_(min_df) {}

    // STRAT-9: the distance between symmetric points
    void symmetric(const symmetry::Sym& s) { sym_ = s; }
    // the problem's periodic variables (periodic.hpp PER-3; task 24, §24.4.2)
    void periodic(const std::vector<char>& mask) { periodic_ = mask; }

    void add(const Vec& x, double f, long long start)
    {
        for (std::size_t q = 0; q < list_.size(); ++q) {
            Minimum& m = list_[q];
            if (distance(m.x, x) <= dist_ && std::fabs(m.f - f) <= df_) {
                joined_ = q;
                ++m.found;
                if (f < m.f) {
                    m.x = x;
                    m.f = f;
                    m.start = start;
                }
                return;
            }
        }
        joined_ = list_.size();
        list_.push_back({x, f, start, 1});
    }

    // the minimum the last add() joined or opened
    std::size_t joined() const { return joined_; }

    const std::vector<Minimum>& list() const { return list_; }

    // [[f, start, found, x1, ..., xn], ...] in compact JSON
    std::string json_text() const
    {
        json::Value a = json::Value::array();
        for (const Minimum& m : list_) {
            json::Value row = json::Value::array();
            row.items.push_back(json::Value::number(m.f));
            row.items.push_back(json::Value::integer(m.start));
            row.items.push_back(json::Value::integer(m.found));
            for (double v : m.x) row.items.push_back(json::Value::number(v));
            a.items.push_back(row);
        }
        return json::dump(a, -1, ",", ":");
    }

private:
    Vec lo_, hi_;
    double dist_, df_;
    std::vector<Minimum> list_;
    symmetry::Sym sym_;
    std::vector<char> periodic_;
    std::size_t joined_ = 0;

    double distance(const Vec& a0, const Vec& b0) const
    {
        Vec ka, kb;                                   // STRAT-9's copies
        const Vec& a = sym_.mode == symmetry::Mode::canon ? sym_.view(a0, a0, ka) : a0;
        const Vec& b = sym_.view(b0, a0, kb);
        double d = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) {
            const double w = hi_[j] - lo_[j];
            const double s = std::isfinite(w) && w > 0.0 ? w : 1.0;
            d = std::max(d, periodic::gap(a[j], b[j], periodic::is(periodic_, j), w) / s);
        }
        return d;
    }
};

// ── the knobs of a strategy ─────────────────────────────────────────────────
class StrategyKnobs {
public:
    StrategyKnobs(const std::string& name, const Knobs& knobs, std::vector<std::string> own)
        : name_(name), knobs_(knobs), own_(std::move(own)) {}

    const Knob* find(const std::string& k) const
    {
        for (const auto& kv : knobs_)
            if (kv.first == k) return &kv.second;
        return nullptr;
    }

    double number(const std::string& k, double fallback)
    {
        double x = fallback;
        if (const Knob* v = find(k)) x = detail::number_of(k, *v);
        settings.push_back({k, detail::g_format(x)});
        return x;
    }

    long long whole(const std::string& k, long long fallback, long long least)
    {
        const double x = number(k, static_cast<double>(fallback));
        if (!(x >= static_cast<double>(least)) || x != std::floor(x))
            throw std::invalid_argument(name_ + ": " + k + " must be a whole number >= " + std::to_string(least));
        return static_cast<long long>(x);
    }

    std::string word(const std::string& k, const std::string& fallback, const std::vector<std::string>& words)
    {
        std::string w = fallback;
        if (const Knob* v = find(k)) {
            w = detail::shown(*v);
            if (!v->is_word || !detail::listed(words, w))
                throw std::invalid_argument(k + " = '" + w + "': " + detail::joined(words));
        }
        settings.push_back({k, w});
        return w;
    }

    Inner inner(const std::string& k)
    {
        const Knob* v = find(k);
        if (v == nullptr) throw std::invalid_argument(name_ + " needs its inner solver: " + k + " = {\"solver\": ...}");
        Inner in = inner_of(k, *v);
        settings.push_back({k, v->word});
        return in;
    }

    std::vector<std::string> ignored() const
    {
        std::vector<std::string> out;
        for (const auto& kv : knobs_)
            if (!detail::listed(own_, kv.first) && kv.first != "min_sym") out.push_back(kv.first);   // STRAT-9
        std::sort(out.begin(), out.end());
        return out;
    }

    std::vector<std::pair<std::string, std::string>> settings;

private:
    std::string name_;
    const Knobs& knobs_;
    std::vector<std::string> own_;
};

// STRAT-13's estimates, as settings
inline std::vector<std::pair<std::string, std::string>> bayes_estimates(long long w, long long N)
{
    const double dw = static_cast<double>(w), dN = static_cast<double>(N);
    auto full = [](double v) {                        // all 17 digits, not the settings' 6
        char b[32];
        std::snprintf(b, sizeof b, "%.17g", v);
        return std::string(b);
    };
    const std::string expected = N > w + 2 ? full(dw * (dN - 1.0) / (dN - dw - 2.0)) : "inf";
    const std::string uncovered = N > 1 ? full(dw * (dw + 1.0) / (dN * (dN - 1.0))) : "1";
    return {{"minima_found", std::to_string(w)}, {"minima_searches", std::to_string(N)},
            {"minima_expected", expected}, {"minima_uncovered", uncovered}};
}

inline bool bayes_of(StrategyKnobs& k)
{
    return k.find("minima_bayes") && k.word("minima_bayes", "off", {"off", "on"}) == "on";
}

// the lean mode, where keep_points = best is given (see STRAT-12)
inline bool lean_of(StrategyKnobs& k)
{
    return k.find("keep_points") && k.word("keep_points", "all", {"all", "best"}) == "best";
}

// STRAT-9: the log grouped by the problem's symmetries where min_sym is given
inline void group_by_symmetry(StrategyKnobs& k, const Problem& P, MinimaLog& log)
{
    if (!P.periodic.empty()) log.periodic(P.periodic);           // periodic.hpp PER-3
    if (k.find("min_sym") == nullptr) return;
    const std::string w = k.word("min_sym", "off", {"off", "canon", "align"});
    symmetry::Sym s;
    s.mode = w == "canon" ? symmetry::Mode::canon : w == "align" ? symmetry::Mode::align : symmetry::Mode::off;
    if (!s.on()) return;
    if (P.sym_blocks.empty() && !(s.mode == symmetry::Mode::canon ? bool(P.canon) : bool(P.align)))
        throw std::invalid_argument("min_sym = " + w + ": the problem gives no symmetries (sym_blocks, canon or align)");
    if (!P.sym_blocks.empty()) symmetry::check(P.sym_blocks, P.lo.size());
    s.groups = P.sym_blocks;
    s.canon = P.canon;
    s.align = P.align;
    s.period = symmetry::periods(P.periodic, P.lo, P.hi);                           // SYM-11
    log.symmetric(s);
}

}   // namespace strategy

inline const std::vector<std::string>& strategy_names()
{
    static const std::vector<std::string> v{"multistart", "ipop", "bipop", "bipop_2009", "halving", "mlsl",
                                            "basin_hopping", "ils", "memetic", "tiktak", "data_continuation", "pap",
                                            "pbh", "mde"};
    return v;
}

// halving, mlsl, basin_hopping, ils, memetic (search.hpp, included at the end)
inline Result run_mlsl(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size, long long budget,
                       unsigned seed, const Knobs& knobs, Observer observer, int record_every);
inline Result run_walk(const std::string& name, const Problem& P, double eq_tol, bool has_tol0, double eq_tol0,
                       int pop_size, long long budget, unsigned seed, const Knobs& knobs, Observer observer,
                       int record_every, const Start& start);
inline Result run_halving(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size,
                          long long budget, unsigned seed, const Knobs& knobs, Observer observer, int record_every,
                          const Start& start);
inline Result run_memetic(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size,
                          long long budget, unsigned seed, const Knobs& knobs, Observer observer, int record_every,
                          const Start& start);

// pbh (pbh.hpp, included at the end; task 25, §24.2.2)
inline Result run_pbh(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size, long long budget,
                      unsigned seed, const Knobs& knobs, Observer observer, int record_every, const Start& start);

// mde (mde.hpp, included at the end; task 25, §24.2.3)
inline Result run_mde(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size, long long budget,
                      unsigned seed, const Knobs& knobs, Observer observer, int record_every, const Start& start);

// tiktak (tiktak.hpp, included at the end; task 19, §19.6)
inline Result run_tiktak(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size,
                         long long budget, unsigned seed, const Knobs& knobs, Observer observer, int record_every);

// data_continuation (continuation.hpp, included at the end; task 19, §19.6)
inline Result run_data_continuation(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size,
                                    long long budget, unsigned seed, const Knobs& knobs, Observer observer,
                                    int record_every, const Start& start);

// pap (pap.hpp, included at the end; task 19, §19.6)
inline Result run_pap(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size, long long budget,
                      unsigned seed, const Knobs& knobs, Observer observer, int record_every);

// ipop, bipop, bipop_2009 (restarts.hpp, included at the end)
inline Result run_restarts(const std::string& name, const Problem& P, double eq_tol, bool has_tol0, double eq_tol0,
                           int pop_size, long long budget, unsigned seed, const Knobs& knobs, Observer observer,
                           int record_every, const Start& start);

// ── multistart ──────────────────────────────────────────────────────────────
inline Result run_multistart(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size,
                             long long budget, unsigned seed, const Knobs& knobs, Observer observer,
                             int record_every, const Start& start)
{
    using namespace strategy;
    StrategyKnobs k("multistart", knobs, {"inner", "n_starts", "start_points", "threads", "min_dist", "min_df",
                                          "allowance", "start_sigma", "keep_points", "minima_bayes", "start_filter",
                                          "oq_stage1", "oq_waitcycle", "oq_threshfactor", "oq_distfactor"});
    const Inner inner = k.inner("inner");
    const long long n_starts = k.whole("n_starts", 10, 1);
    const std::string points =
        k.word("start_points", "sobol", {"sobol", "lhs", "uniform", "halton_rand", "sobol_owen", "around_x0"});
    const long long threads = k.whole("threads", 1, 1);
    const double min_dist = k.number("min_dist", 1e-3), min_df = k.number("min_df", 1e-8);
    const bool sequential =                                                          // STRAT-11
        k.find("allowance") && k.word("allowance", "even", {"even", "sequential"}) == "sequential";
    const bool around = points == "around_x0";                                       // STRAT-10
    const double sigma = around ? k.number("start_sigma", 0.1) : 0.0;
    const bool bayes = bayes_of(k);                                                  // STRAT-13
    const bool oqnlp =                                                               // STRAT-14
        k.find("start_filter") && k.word("start_filter", "off", {"off", "oqnlp"}) == "oqnlp";
    const long long oq_stage1 = oqnlp ? k.whole("oq_stage1", 200, 1) : 0;
    const long long oq_wait = oqnlp ? k.whole("oq_waitcycle", 20, 1) : 0;
    const double oq_thresh = oqnlp ? k.number("oq_threshfactor", 0.2) : 0.0;
    const double oq_dist = oqnlp ? k.number("oq_distfactor", 0.75) : 0.0;
    if (oqnlp && (threads > 1 || sequential))
        throw std::invalid_argument("multistart: start_filter = oqnlp runs its starts one by one (threads = 1, "
                                    "allowance even)");
    if (oqnlp && !(oq_thresh >= 0.0 && oq_dist >= 0.0))
        throw std::invalid_argument("multistart: oq_threshfactor and oq_distfactor must be >= 0");
    const std::size_t n = P.lo.size();
    if (sequential && threads > 1)
        throw std::invalid_argument("multistart: allowance = sequential runs one start at a time (threads = 1)");
    if (around && !start.has_x0) throw std::invalid_argument("multistart: start_points = around_x0 needs x0");
    if (around && start.x0.size() != n)
        throw std::invalid_argument("multistart: x0 has " + std::to_string(start.x0.size()) + " values for " +
                                    std::to_string(n) + " variables");
    if (around && !(sigma >= 0.0 && std::isfinite(sigma)))
        throw std::invalid_argument("multistart: start_sigma must be a finite number >= 0");
    if (n_starts > 1 && !around)
        for (std::size_t j = 0; j < n; ++j)
            if (!std::isfinite(P.lo[j]) || !std::isfinite(P.hi[j]))
                throw std::invalid_argument("multistart draws its starts in the box: every bound must be finite");

    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    ev.lean = lean_of(k);                                                            // STRAT-12
    std::vector<StartSpec> starts(static_cast<std::size_t>(n_starts));
    const std::vector<double> U =
        n_starts > 1 && !around
            ? designs::design(points, static_cast<std::size_t>(n_starts - 1), static_cast<int>(n), seed)
            : std::vector<double>();
    std::mt19937 jitter(seed);                                                       // STRAT-10
    for (std::size_t s = 0; s < starts.size(); ++s) {
        StartSpec& sp = starts[s];
        sp.allowance = budget / n_starts + (static_cast<long long>(s) < budget % n_starts ? 1 : 0);   // STRAT-3
        sp.seed = start_seed(seed, s);
        if (s == 0) {
            sp.start = start;                                                 // the inner solver's own
        } else if (around) {
            sp.start.has_x0 = true;
            sp.start.x0 = start.x0;
            for (std::size_t j = 0; j < n; ++j) {
                const bool box = std::isfinite(P.lo[j]) && std::isfinite(P.hi[j]);
                const double z = rnd::normal01(jitter);
                const double step = sigma * (box ? P.hi[j] - P.lo[j] : 1.0);
                sp.start.x0[j] = std::min(P.hi[j], std::max(P.lo[j], start.x0[j] + step * z));
            }
        } else {
            sp.start.has_x0 = true;
            sp.start.x0.resize(n);
            for (std::size_t j = 0; j < n; ++j)
                sp.start.x0[j] = P.lo[j] + U[(s - 1) * n + j] * (P.hi[j] - P.lo[j]);
        }
    }
    std::vector<Result> runs;
    MinimaLog log(P.lo, P.hi, min_dist, min_df);
    group_by_symmetry(k, P, log);
    long long oq_starts = 0, oq_rejected_merit = 0, oq_rejected_distance = 0;
    try {
        if (oqnlp) {                                                                  // STRAT-14
            auto p1 = [&](const Vec& x) {                                             // OQ-3
                const std::size_t i = ev.index(x);
                if (ev.failed[i]) return std::numeric_limits<double>::infinity();
                return ev.F[i][0] + ev.cv(i);
            };
            auto x0_of = [&](const StartSpec& sp) { return sp.start.has_x0 ? sp.start.x0 : Vec(); };
            std::vector<double> maxdist;                                              // a solution's, by log index
            auto run_from = [&](StartSpec sp, const Vec& from) {
                const double left = static_cast<double>(ev.budget) - ev.used();
                if (left < 1.0) throw BudgetSpent{};
                sp.allowance = static_cast<long long>(left);
                const Result r = round(ev, inner, {sp}, 1, pop_size).at(0);
                runs.push_back(r);
                ++oq_starts;
                if (P.m != 1 || r.objectives.empty() || !(r.cv.empty() || r.cv[0] <= 0.0)) return;   // OQ-5
                const std::size_t before = log.list().size();
                log.add(r.variables[0], r.objectives[0][0], static_cast<long long>(runs.size() - 1));
                double d2 = 0.0;
                for (std::size_t j = 0; j < n; ++j) d2 += (r.variables[0][j] - from[j]) * (r.variables[0][j] - from[j]);
                const double d = std::sqrt(d2);
                const std::size_t at = log.joined();                                  // the solution it joined
                (void)before;
                if (at >= maxdist.size()) maxdist.resize(at + 1, 0.0);
                maxdist[at] = std::max(maxdist[at], d);
            };
            // the trial points (OQ-6): start 0's x0 where the start has one, then
            // the starts of the design; stage 1: the first oq_stage1 evaluated,
            // the inner solver from the best
            std::vector<Vec> trial;
            std::vector<StartSpec> specs;
            for (std::size_t q = 0; q < starts.size(); ++q) {
                const Vec x = x0_of(starts[q]);
                if (x.empty()) continue;
                trial.push_back(x);
                StartSpec sp = starts[q];
                sp.start = Start();
                sp.start.has_x0 = true;
                sp.start.x0 = x;
                specs.push_back(sp);
            }
            if (trial.empty()) throw std::invalid_argument("multistart: start_filter = oqnlp needs trial points "
                                                           "(n_starts >= 2, or x0)");
            const std::size_t s1 = std::min<std::size_t>(trial.size(), static_cast<std::size_t>(oq_stage1));
            std::size_t best = 0;
            double best_p = std::numeric_limits<double>::infinity();
            for (std::size_t t = 0; t < s1; ++t) {
                const double v = p1(trial[t]);
                if (v < best_p) {
                    best_p = v;
                    best = t;
                }
            }
            if (!ev.recorded) ev.record();
            double threshold = best_p;
            run_from(specs[best], trial[best]);
            if (!runs.back().objectives.empty() && (runs.back().cv.empty() || runs.back().cv[0] <= 0.0))
                threshold = runs.back().objectives[0][0] + (runs.back().cv.empty() ? 0.0 : runs.back().cv[0]);
            // stage 2: the filters
            long long above = 0;
            for (std::size_t t = s1; t < trial.size(); ++t) {
                const double v = p1(trial[t]);
                bool far = true;
                for (std::size_t q = 0; q < log.list().size() && far; ++q) {
                    double d2 = 0.0;
                    for (std::size_t j = 0; j < n; ++j)
                        d2 += (trial[t][j] - log.list()[q].x[j]) * (trial[t][j] - log.list()[q].x[j]);
                    far = !(std::sqrt(d2) < oq_dist * (q < maxdist.size() ? maxdist[q] : 0.0));
                }
                const bool merit = v <= threshold;
                if (!merit) ++oq_rejected_merit;
                if (!far) ++oq_rejected_distance;
                if (merit && far) {
                    threshold = v;
                    above = 0;
                    run_from(specs[t], trial[t]);
                    continue;
                }
                if (merit) {
                    above = 0;                                                        // OQ-4
                    continue;
                }
                if (++above >= oq_wait) {                                             // OQ-1
                    threshold += oq_thresh * (1.0 + std::fabs(threshold));
                    above = 0;
                }
            }
        } else if (!sequential) {
            runs = round(ev, inner, starts, static_cast<int>(threads), pop_size);
        } else {                                                                      // STRAT-11
            for (StartSpec& sp : starts) {
                const double left = static_cast<double>(ev.budget) - ev.used();
                if (left < 1.0) break;
                sp.allowance = static_cast<long long>(left);
                runs.push_back(round(ev, inner, {sp}, 1, pop_size).at(0));
            }
        }
    } catch (const BudgetSpent&) {
    }
    if (!ev.recorded) ev.record();
    long long answered = 0;
    if (P.m == 1)
        for (std::size_t s = 0; s < runs.size(); ++s)
            if (!runs[s].objectives.empty()) {
                if (!oqnlp) log.add(runs[s].variables[0], runs[s].objectives[0][0], static_cast<long long>(s));
                ++answered;
            }
    ev.over = true;
    const std::vector<std::size_t> idx = ev.answer();
    ev.record(true);
    Result r = detail::result_of(ev, idx);
    r.ignored = k.ignored();
    r.settings = k.settings;
    if (oqnlp) {                                                                      // STRAT-14
        r.settings.push_back({"oq_starts", std::to_string(oq_starts)});
        r.settings.push_back({"oq_rejected_merit", std::to_string(oq_rejected_merit)});
        r.settings.push_back({"oq_rejected_distance", std::to_string(oq_rejected_distance)});
    }
    if (bayes)                                                                        // STRAT-13
        for (auto& e : bayes_estimates(static_cast<long long>(log.list().size()), answered)) r.settings.push_back(e);
    r.settings.push_back({"minima", log.json_text()});
    r.settings.push_back({"method_version", std::to_string(METHOD_VERSION)});
    return r;
}

inline Result run_strategy(const std::string& name, const Problem& P, double eq_tol, bool has_tol0,
                           double eq_tol0, int pop_size, long long budget, unsigned seed, const Knobs& knobs,
                           Observer observer, int record_every, const Start& start)
{
    if (name == "multistart")
        return run_multistart(P, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, knobs, std::move(observer),
                              record_every, start);
    if (name == "ipop" || name == "bipop" || name == "bipop_2009")
        return run_restarts(name, P, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, knobs, std::move(observer),
                            record_every, start);
    if (name == "mlsl")
        return run_mlsl(P, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, knobs, std::move(observer), record_every);
    if (name == "ils" || name == "basin_hopping")
        return run_walk(name, P, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, knobs, std::move(observer),
                        record_every, start);
    if (name == "halving")
        return run_halving(P, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, knobs, std::move(observer),
                           record_every, start);
    if (name == "memetic")
        return run_memetic(P, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, knobs, std::move(observer),
                           record_every, start);
    if (name == "pap")
        return run_pap(P, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, knobs, std::move(observer), record_every);
    if (name == "data_continuation")
        return run_data_continuation(P, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, knobs, std::move(observer),
                                     record_every, start);
    if (name == "pbh")
        return run_pbh(P, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, knobs, std::move(observer), record_every,
                       start);
    if (name == "mde")
        return run_mde(P, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, knobs, std::move(observer), record_every,
                       start);
    if (name == "tiktak")
        return run_tiktak(P, eq_tol, has_tol0, eq_tol0, pop_size, budget, seed, knobs, std::move(observer),
                          record_every);
    throw std::invalid_argument("unknown strategy '" + name + "'");
}

}   // namespace mootation::scalar

#include "restarts.hpp"
#include "search.hpp"
#include "tiktak.hpp"
#include "continuation.hpp"
#include "pap.hpp"
#include "pbh.hpp"
#include "mde.hpp"
