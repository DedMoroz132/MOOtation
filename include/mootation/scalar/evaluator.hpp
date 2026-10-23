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
// evaluations and at the end, with the variables beside the objectives.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "../dss.hpp"
#include "../fp_contract.hpp"

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
    std::vector<double> lo, hi;
    int m = 0, n_cons = 0, n_eq = 0;
    std::function<void(const std::vector<double>& x, std::vector<double>& f, std::vector<double>& g)>
        evaluate;
};

// observer(gen, F, X): the answer's objectives and variables.
using Observer = std::function<void(long long gen, const Rows& F, const Rows& X)>;

class Evaluator {
public:
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
    std::vector<char> failed;
    long long spent = 0;
    long long limit = 0;               // the current subproblem's last evaluation
    std::vector<std::size_t> nd;       // feasible non-dominated archive indices
    Rows ndF;                          // ... and their objectives
    bool nd_set = false;               // nd was judged at nd_tol
    double nd_tol = 0.0;
    bool over = false;                 // the run has ended: the final tolerance
    bool recorded = false;             // gen 0 has gone to the observer
    long long gen0 = 0;                // gens before this evaluator (polish.hpp)

    Evaluator(const Problem& p, double eq_tol, bool has_eq_tol0, double eq_tol0_, long long budget_,
              int pop_size_, Observer obs, int record_every)
        : P(p), lo(p.lo), hi(p.hi), m(p.m), n_cons(p.n_cons), n_eq(p.n_eq), eq_tol_end(eq_tol),
          has_tol0(has_eq_tol0), eq_tol0(eq_tol0_), budget(budget_), pop_size(pop_size_)
    {
        if (record_every > 0 && obs) observer = std::move(obs);
        stride = static_cast<long long>(std::max(1, record_every)) * pop_size;
        next_record = stride;
        limit = budget;
    }

    // ── evaluating ──────────────────────────────────────────────────────────
    // The archive index of x, evaluating it if it is new.
    std::size_t index(const std::vector<double>& x)
    {
        const std::string key = key_of(x);
        auto it = where_.find(key);
        if (it != where_.end()) return it->second;
        if (spent >= budget) throw BudgetSpent{};
        if (spent >= limit) throw SubBudgetSpent{};
        std::vector<double> f(static_cast<std::size_t>(m), 0.0), g(static_cast<std::size_t>(n_cons), 0.0);
        P.evaluate(x, f, g);
        bool bad = false;
        for (double v : f) bad = bad || !std::isfinite(v);
        for (double v : g) bad = bad || !std::isfinite(v);
        return keep(x, f, g, bad, key, true);
    }

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

    static std::string key_of(const std::vector<double>& x)
    {
        std::string k(x.size() * sizeof(double), '\0');
        if (!x.empty()) std::memcpy(&k[0], x.data(), k.size());
        return k;
    }

    std::size_t keep(const std::vector<double>& x, const std::vector<double>& f,
                     const std::vector<double>& g, bool bad, const std::string& key, bool count)
    {
        const std::size_t i = X.size();
        X.push_back(x);
        F.push_back(f);
        G.push_back(g);
        failed.push_back(static_cast<char>(bad));
        where_[key] = i;
        if (count) ++spent;
        if (!bad && nd_set && nd_tol == eq_tol() && cv(i) <= 0.0) offer(i);
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
