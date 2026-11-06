#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// data_continuation (task 19, §19.6; B8): a least-squares problem solved in
// stages, each stage the same residuals seen with its own weights — a vector
// of the blocks' weights, or a window of the residuals — the solution of
// stage k the start of stage k + 1, the problem itself the last stage. A
// strategy of stage 3's frame (strategy.hpp); included from it.
//
//   dc_stages   the stages before the last, a JSON array: each element the
//               blocks' weights [w_1, ..., w_B] (B the problem's residual
//               blocks; Problem::blocks, weights) or {"window": [a, b]}, the
//               residuals a ... b - 1 at weight 1 and the others at 0
//   dc_share    the share of the budget for those stages, evenly (0.5)
//   inner       the solver of every stage (lm, lbfgsb, nelder_mead, ...)
//
// There is no source to transcribe (the task's item B8: a schedule of the
// data, as continuation methods do with a parameter); what follows are its
// readings.
//
// Readings (OURS):
//   DC-1. The problem itself is always the last stage: the stages given
//     lead up to it, and the answer is the problem's (its archive's best
//     point, which may come from an earlier stage).
//   DC-2. Stage 1 starts where the inner solver would (its plan, or x0);
//     stage k + 1 from stage k's answer. Stage k's generator is
//     start_seed(seed, k - 1) (STRAT-1): one stage is the inner solver's own
//     run, bit for bit.
//   DC-3. The budget: the stages before the last share dc_share of it
//     evenly (the first ones one more, as STRAT-3); the last stage has what
//     is left.
//   DC-4. A stage evaluates through its own view (Stage below): a point an
//     earlier stage, or this one, met is answered from the problem's raw
//     residuals kept then (every point of the run comes through a stage), a
//     new one calls the problem's residuals. At the stage's end its
//     new points go into the run's archive, in the order they came, with the
//     problem's own weights (r~, f = 1/2 ||r~||^2, and J~, grad f = J~^T r~
//     where the stage called the Jacobian), as a round's starts' do.
//   DC-5. A window [a, b) becomes three blocks (a, b - a, n_res - b; empty
//     ones dropped) at weights 0, 1, 0; a weight vector needs the problem's
//     blocks, one weight each, finite and >= 0, and at least one > 0. A
//     window needs the residuals' count (Problem::n_res, or its blocks'
//     sizes): none is drawn from a call outside the budget.
// ============================================================================

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../json.hpp"
#include "strategy.hpp"

namespace mootation::scalar {

namespace continuation {

// A stage's blocks and weights over the problem's n_res residuals (DC-5).
struct Weights {
    std::vector<int> blocks;
    std::vector<double> weights;
};

inline std::vector<Weights> stages_of(const std::string& text, const Problem& P, int n_res)
{
    const json::Value a = json::parse(text);
    if (a.kind != json::Value::Kind::Array) throw std::invalid_argument("data_continuation: dc_stages must be an array");
    const std::size_t B = P.blocks.empty() ? 1 : P.blocks.size();
    std::vector<Weights> out;
    for (const json::Value& e : a.items) {
        Weights w;
        if (e.kind == json::Value::Kind::Object) {
            const json::Value* win = e.find("window");
            if (win == nullptr || win->kind != json::Value::Kind::Array || win->items.size() != 2)
                throw std::invalid_argument("data_continuation: a window is {\"window\": [a, b]}");
            const auto bound = [](const json::Value& v) {
                if (v.kind != json::Value::Kind::Int) throw std::invalid_argument("data_continuation: a window's ends are whole numbers");
                return static_cast<long long>(v.i);
            };
            const long long lo = bound(win->items[0]), hi = bound(win->items[1]);
            if (n_res <= 0) throw std::invalid_argument("data_continuation: a window needs the problem's n_res");
            if (!(0 <= lo && lo < hi && hi <= n_res))
                throw std::invalid_argument("data_continuation: a window [a, b) needs 0 <= a < b <= n_res (" +
                                            std::to_string(n_res) + ")");
            const long long sizes[3] = {lo, hi - lo, n_res - hi};
            const double ws[3] = {0.0, 1.0, 0.0};
            for (int k = 0; k < 3; ++k)
                if (sizes[k] > 0) {
                    w.blocks.push_back(static_cast<int>(sizes[k]));
                    w.weights.push_back(ws[k]);
                }
        } else if (e.kind == json::Value::Kind::Array) {
            if (e.items.size() != B)
                throw std::invalid_argument("data_continuation: a stage gives one weight per residual block (" +
                                            std::to_string(B) + ")");
            bool some = false;
            for (const json::Value& v : e.items) {
                const double x = v.kind == json::Value::Kind::Int ? static_cast<double>(v.i)
                                 : v.kind == json::Value::Kind::Float ? v.f
                                                                      : std::numeric_limits<double>::quiet_NaN();
                if (!(x >= 0.0) || !std::isfinite(x))
                    throw std::invalid_argument("data_continuation: the weights must be finite numbers >= 0");
                some = some || x > 0.0;
                w.weights.push_back(x);
            }
            if (!some) throw std::invalid_argument("data_continuation: a stage needs a weight > 0");
            w.blocks = P.blocks;
        } else {
            throw std::invalid_argument("data_continuation: a stage is [w_1, ...] or {\"window\": [a, b]}");
        }
        out.push_back(std::move(w));
    }
    return out;
}

// r (or J's rows) times sqrt(w_b), block by block, as the evaluator scales them
template <typename RowOp>
void scaled(const std::vector<int>& blocks, const std::vector<double>& weights, std::size_t n_res, RowOp op)
{
    if (weights.empty()) return;
    std::size_t at = 0;
    for (std::size_t b = 0; b < weights.size(); ++b) {
        const std::size_t size = blocks.empty() ? n_res : static_cast<std::size_t>(blocks[b]);
        const double s = std::sqrt(weights[b]);
        for (std::size_t k = 0; k < size; ++k) op(at + k, s);
        at += size;
    }
}

// Every point the stages met, with the problem's raw residuals (DC-4).
struct Raw {
    std::vector<double> x, r;
    Rows J;
    bool has_jac = false;
};

struct Seen {
    std::vector<Raw> own;
    std::unordered_map<std::string, std::size_t> where;
};

// A stage's road to the problem (DC-4): the problem with the stage's weights,
// its residuals from what the stages met already, else called and kept.
class Stage {
public:
    using Vec = std::vector<double>;
    Problem problem;

    Stage(const Problem& P, const Weights& w, Seen& seen) : P_(P), seen_(seen)
    {
        problem = P_;
        problem.blocks = w.blocks;
        problem.weights = w.weights;
        problem.evaluate = nullptr;
        problem.gradient = nullptr;
        problem.value_grad = nullptr;
        problem.residuals = nullptr;
        problem.jacobian = nullptr;
        problem.residuals_jac = nullptr;
        if (P_.residuals) problem.residuals = [this](const Vec& x, Vec& r) { r = get(x, false).r; };
        if (P_.residuals_jac)
            problem.residuals_jac = [this](const Vec& x, Vec& r, Rows& J) {
                const Raw& k = get(x, true);
                r = k.r;
                J = k.J;
            };
        if (P_.jacobian) problem.jacobian = [this](const Vec& x, Rows& J) { J = get(x, true).J; };
    }

    Stage(const Stage&) = delete;
    Stage& operator=(const Stage&) = delete;

    // the stage's calls into the run's archive, with the problem's weights
    void merge_into(Evaluator& ev) const
    {
        for (const Event& e : events_) {
            const Raw& k = seen_.own[e.at];
            Vec r = k.r;
            scaled(P_.blocks, P_.weights, r.size(), [&](std::size_t i, double s) { r[i] *= s; });
            double sum = 0.0;
            for (double v : r) sum += v * v;
            const Vec f{0.5 * sum};
            const bool bad = !std::isfinite(f[0]);
            if (!e.grads) {
                ev.adopt(k.x, f, {}, r, bad, e.evals, 0, nullptr, nullptr);
                continue;
            }
            Rows J = k.J;
            scaled(P_.blocks, P_.weights, r.size(), [&](std::size_t i, double s) {
                for (double& v : J[i]) v *= s;
            });
            Vec gf(k.x.size(), 0.0);
            for (std::size_t a = 0; a < r.size(); ++a)
                for (std::size_t j = 0; j < gf.size(); ++j) gf[j] += J[a][j] * r[a];
            ev.adopt(k.x, f, {}, r, bad, e.evals, 1, &gf, &J);
        }
    }

private:
    struct Event {
        std::size_t at;
        int evals, grads;
    };
    const Problem& P_;
    Seen& seen_;
    std::vector<Event> events_;

    static std::string key_of(const Vec& x)
    {
        std::string k(x.size() * sizeof(double), '\0');
        if (!x.empty()) std::memcpy(&k[0], x.data(), k.size());
        return k;
    }

    const Raw& get(const Vec& x, bool jac)
    {
        const std::string key = key_of(x);
        auto it = seen_.where.find(key);
        if (it == seen_.where.end()) {
            Raw k;
            k.x = x;
            if (jac && P_.residuals_jac) {
                P_.residuals_jac(x, k.r, k.J);
                k.has_jac = true;
            } else if (P_.residuals) {
                P_.residuals(x, k.r);
            } else {
                P_.residuals_jac(x, k.r, k.J);
                k.has_jac = true;
            }
            events_.push_back({seen_.own.size(), 1, k.has_jac ? 1 : 0});
            it = seen_.where.emplace(key, seen_.own.size()).first;
            seen_.own.push_back(std::move(k));
        }
        Raw& k = seen_.own[it->second];
        if (jac && !k.has_jac) {
            if (P_.jacobian) {
                P_.jacobian(x, k.J);
            } else {
                Vec r;
                P_.residuals_jac(x, r, k.J);
            }
            k.has_jac = true;
            events_.push_back({it->second, 0, 1});
        }
        return k;
    }
};

}   // namespace continuation

inline Result run_data_continuation(const Problem& P, double eq_tol, bool has_tol0, double eq_tol0, int pop_size,
                                    long long budget, unsigned seed, const Knobs& knobs, Observer observer,
                                    int record_every, const Start& start)
{
    using namespace strategy;
    StrategyKnobs k("data_continuation", knobs, {"inner", "dc_stages", "dc_share", "min_dist", "min_df"});
    const Inner inner = k.inner("inner");
    if (!P.has_residuals()) throw std::invalid_argument("data_continuation: the problem's residuals are needed");
    std::string text = "[]";
    if (const Knob* v = k.find("dc_stages")) {
        if (!v->is_word) throw std::invalid_argument("data_continuation: dc_stages must be a JSON array");
        text = v->word;
    }
    k.settings.push_back({"dc_stages", text});
    const double share = k.number("dc_share", 0.5);
    const double min_dist = k.number("min_dist", 1e-3), min_df = k.number("min_df", 1e-8);
    if (!(share >= 0.0 && share < 1.0)) throw std::invalid_argument("data_continuation: dc_share in [0, 1)");

    Evaluator ev(P, eq_tol, has_tol0, eq_tol0, budget, pop_size, std::move(observer), record_every);
    MinimaLog log(P.lo, P.hi, min_dist, min_df);
    strategy::group_by_symmetry(k, P, log);
    long long done = 0;
    try {
        int n_res = P.n_res;                                    // DC-5: a window needs the count
        if (n_res == 0)
            for (int b : P.blocks) n_res += b;
        std::vector<continuation::Weights> stages = continuation::stages_of(text, P, n_res);
        stages.push_back({P.blocks, P.weights});                                                       // DC-1
        const long long before = static_cast<long long>(stages.size()) - 1;
        const long long early = static_cast<long long>(share * static_cast<double>(budget));            // DC-3
        Start from = start;
        continuation::Seen seen;                                                                       // DC-4
        for (long long s = 0; s <= before; ++s) {
            const long long rest = search::left(ev);
            if (rest <= 0) break;
            const long long allowance =
                s < before ? early / before + (s < early % before ? 1 : 0) : rest;
            continuation::Stage view(P, stages[static_cast<std::size_t>(s)], seen);
            Result r;
            try {
                r = run(inner.name, view.problem, eq_tol, has_tol0, eq_tol0, pop_size, std::min(allowance, rest),
                        start_seed(seed, static_cast<std::uint64_t>(s)), inner.knobs, {}, 0, from);   // DC-2
            } catch (const BudgetSpent&) {
            }
            view.merge_into(ev);
            ++done;
            const search::Found b = search::found(r);
            if (!b.ok) continue;
            log.add(b.x, b.f, s);
            from = search::at(b.x);
        }
    } catch (const BudgetSpent&) {
    }
    return search::finish(ev, k, log, {{"stages", std::to_string(done)}});
}

}   // namespace mootation::scalar
