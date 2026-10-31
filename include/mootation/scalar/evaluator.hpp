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
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../dss.hpp"
#include "../fp_contract.hpp"
#include "improvements.hpp"

namespace mootation::scalar {

struct BudgetSpent {};       // the run's evaluations are spent
struct SubBudgetSpent {};    // the current subproblem's allowance is spent
struct FailedPoint {};       // sqp met a failed evaluation (solvers.hpp)

using Rows = std::vector<std::vector<double>>;

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

    bool has_residuals() const { return residuals || residuals_jac; }
    bool has_gradient() const { return gradient || value_grad || jacobian || residuals_jac; }
};

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
    ImprovementLog improvements;       // m = 1: the best point's records (improvements.hpp)

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
    }

    // ── evaluating ──────────────────────────────────────────────────────────
    // The archive index of x, evaluating it if it is new.
    std::size_t index(const std::vector<double>& x)
    {
        const std::string key = key_of(x);
        auto it = where_.find(key);
        if (it != where_.end()) return it->second;
        return fresh(x, key, false);
    }

    // What the budget is spent by: evaluations + grad_cost * grad_calls.
    double used() const
    {
        return static_cast<double>(spent) + grad_cost * static_cast<double>(grad_calls);
    }

    // ── gradients (SO-2) ────────────────────────────────────────────────────
    // The archive index of x with its gradient known: the objective's (grad),
    // and the constraints' Jacobian (jac_g) or the residuals' (jac).
    std::size_t index_grad(const std::vector<double>& x)
    {
        if (!P.has_gradient()) throw std::logic_error("index_grad: the problem gives no gradient");
        const std::string key = key_of(x);
        auto it = where_.find(key);
        const std::size_t i = it != where_.end() ? it->second : fresh(x, key, true);
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
                     const std::vector<double>& g, bool bad, const std::string& key, bool count)
    {
        const std::size_t i = X.size();
        X.push_back(x);
        F.push_back(f);
        G.push_back(g);
        R.push_back(r_);
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
