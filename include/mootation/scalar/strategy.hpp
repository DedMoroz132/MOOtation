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
// ============================================================================

#include <algorithm>
#include <atomic>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <exception>
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
class View {
public:
    using Vec = std::vector<double>;
    Problem problem;                  // what the inner run evaluates: every call through here

    struct Known {
        Vec x, f, g, r, gf;
        Rows J;
        bool bad = false, has_grad = false;
    };
    std::vector<Known> own;           // the start's points, in the order it met them

    // fresh: what the start may spend on new points (evaluations + grad_cost
    // * gradient calls); past it a call throws BudgetSpent, which ends the
    // inner run as its budget would (STRAT-6)
    explicit View(const Evaluator& parent, double fresh = std::numeric_limits<double>::infinity())
        : parent_(parent), P_(parent.P), fresh_(fresh)
    {
        problem = P_;
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
            ev.adopt(k.x, k.f, k.g, k.r, k.bad, e.evals, e.grads, g ? &k.gf : nullptr, g ? &k.J : nullptr);
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
            } else {
                values(k, own.size(), call == Call::joint);
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
inline std::vector<Result> round(Evaluator& ev, const Inner& inner, const std::vector<StartSpec>& starts,
                                 int threads, int pop_size)
{
    const std::size_t n = starts.size();
    std::vector<std::unique_ptr<View>> views;
    for (std::size_t k = 0; k < n; ++k)
        views.push_back(std::make_unique<View>(
            ev, static_cast<double>(starts[k].fresh >= 0 ? starts[k].fresh : starts[k].allowance)));
    std::vector<Result> out(n);
    std::vector<std::exception_ptr> error(n);
    const double eq_tol = ev.eq_tol_end, eq_tol0 = ev.eq_tol0;
    const bool has_tol0 = ev.has_tol0;
    auto one = [&](std::size_t k) {
        try {
            if (starts[k].allowance > 0)
                out[k] = run(inner.name, views[k]->problem, eq_tol, has_tol0, eq_tol0, pop_size,
                             starts[k].allowance, starts[k].seed, inner.knobs, {}, 0, starts[k].start);
        } catch (...) {
            error[k] = std::current_exception();
        }
    };
    const std::size_t T = std::min<std::size_t>(n, static_cast<std::size_t>(std::max(1, threads)));
    if (T <= 1) {
        for (std::size_t k = 0; k < n; ++k) one(k);
    } else {
        std::atomic<std::size_t> next{0};
        std::vector<std::thread> pool;
        for (std::size_t t = 0; t < T; ++t)
            pool.emplace_back([&]() {
                for (std::size_t k = next++; k < n; k = next++) one(k);
            });
        for (auto& th : pool) th.join();
        for (const Result& r : out) ops::repair_counts() += r.repairs;   // counted on the pool's threads
    }
    for (std::size_t k = 0; k < n; ++k)
        if (error[k]) std::rethrow_exception(error[k]);
    for (std::size_t k = 0; k < n; ++k) views[k]->merge_into(ev);
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

    void add(const Vec& x, double f, long long start)
    {
        for (Minimum& m : list_)
            if (distance(m.x, x) <= dist_ && std::fabs(m.f - f) <= df_) {
                ++m.found;
                if (f < m.f) {
                    m.x = x;
                    m.f = f;
                    m.start = start;
                }
                return;
            }
        list_.push_back({x, f, start, 1});
    }

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

    double distance(const Vec& a, const Vec& b) const
    {
        double d = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) {
            const double w = hi_[j] - lo_[j];
            const double s = std::isfinite(w) && w > 0.0 ? w : 1.0;
            d = std::max(d, std::fabs(a[j] - b[j]) / s);
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
            if (!detail::listed(own_, kv.first)) out.push_back(kv.first);
        std::sort(out.begin(), out.end());
        return out;
    }

    std::vector<std::pair<std::string, std::string>> settings;

private:
    std::string name_;
    const Knobs& knobs_;
    std::vector<std::string> own_;
};

}   // namespace strategy

inline const std::vector<std::string>& strategy_names()
{
    static const std::vector<std::string> v{"multistart", "ipop", "bipop", "bipop_2009", "halving", "mlsl",
                                            "basin_hopping", "ils", "memetic", "tiktak", "data_continuation", "pap"};
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
    StrategyKnobs k("multistart", knobs,
                    {"inner", "n_starts", "start_points", "threads", "min_dist", "min_df"});
    const Inner inner = k.inner("inner");
    const long long n_starts = k.whole("n_starts", 10, 1);
    const std::string points = k.word("start_points", "sobol", {"sobol", "lhs", "uniform", "halton_rand", "sobol_owen"});
    const long long threads = k.whole("threads", 1, 1);
    const double min_dist = k.number("min_dist", 1e-3), min_df = k.number("min_df", 1e-8);
    const std::size_t n = P.lo.size();
    if (n_starts > 1)
        for (std::size_t j = 0; j < n; ++j)
            if (!std::isfinite(P.lo[j]) || !std::isfinite(P.hi[j]))
                throw std::invalid_argument("multistart draws its starts in the box: every bound must be finite");

    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    std::vector<StartSpec> starts(static_cast<std::size_t>(n_starts));
    const std::vector<double> U =
        n_starts > 1 ? designs::design(points, static_cast<std::size_t>(n_starts - 1), static_cast<int>(n), seed)
                     : std::vector<double>();
    for (std::size_t s = 0; s < starts.size(); ++s) {
        StartSpec& sp = starts[s];
        sp.allowance = budget / n_starts + (static_cast<long long>(s) < budget % n_starts ? 1 : 0);   // STRAT-3
        sp.seed = start_seed(seed, s);
        if (s == 0) {
            sp.start = start;                                                 // the inner solver's own
        } else {
            sp.start.has_x0 = true;
            sp.start.x0.resize(n);
            for (std::size_t j = 0; j < n; ++j)
                sp.start.x0[j] = P.lo[j] + U[(s - 1) * n + j] * (P.hi[j] - P.lo[j]);
        }
    }
    std::vector<Result> runs;
    try {
        runs = round(ev, inner, starts, static_cast<int>(threads), pop_size);
    } catch (const BudgetSpent&) {
    }
    if (!ev.recorded) ev.record();
    MinimaLog log(P.lo, P.hi, min_dist, min_df);
    if (P.m == 1)
        for (std::size_t s = 0; s < runs.size(); ++s)
            if (!runs[s].objectives.empty())
                log.add(runs[s].variables[0], runs[s].objectives[0][0], static_cast<long long>(s));
    ev.over = true;
    const std::vector<std::size_t> idx = ev.answer();
    ev.record(true);
    Result r = detail::result_of(ev, idx);
    r.ignored = k.ignored();
    r.settings = k.settings;
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
