#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The local solvers of one subproblem.
//
// All three work in the box normalized to [0, 1]^n (y = (x - lo) / (hi - lo))
// and spend evaluations only through the Evaluator, so the budget and the
// archive see every point. A solver stops at its own tolerance, at the
// subproblem's allowance (SubBudgetSpent) or at the run's budget (BudgetSpent,
// passed on); what it found is in the archive either way, and the
// subproblem's result is the best point it evaluated (best_of).
//
//   sqp    slsqp.hpp. Gradients by forward differences with step fd_step in
//          y (backwards at the upper bound; a step that fails is taken the
//          other way once): n evaluations a gradient, every one counted. The
//          problem's inequalities and equalities go to the solver as
//          constraints of their own (equalities as h = 0). A failed
//          evaluation ends the subproblem: SQP cannot step over a hole in
//          the function (SCALAR-6). With the problem's own gradient
//          (evaluator.hpp) at one objective, that instead (SCALAR-13).
//  nm      nelder_mead.hpp: Lagarias et al.'s algorithm with Gao & Han's
//          adaptive coefficients, in the box.
//  cmaes   cmaes.hpp; the box by evaluating the clipped point and adding the
//          squared distance to it times (1 + |value|) (SCALAR-7).
// For nm and cmaes the subproblem's own constraints and the problem's are a
// penalty in which a feasible point is always better (SCALAR-2): an
// infeasible point is worth the worst feasible value known for this
// subproblem — over the archive at its start, then over what it evaluates —
// plus its violation.
//
// SCALAR-13 (task 18, addendum 1, §3.2). The problem's gradient is in its
// coordinates; sqp works in y, so it takes d/dy_j = (hi_j - lo_j) d/dx_j, and
// the objective's divided by the frame's scale as F̄ is. A constrained problem
// that gives a gradient gives the constraints' Jacobian too (n_cons rows).
// Only at one objective: the gradient of a problem is its one objective's.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "../fp_contract.hpp"
#include "cmaes.hpp"
#include "evaluator.hpp"
#include "nelder_mead.hpp"
#include "slsqp.hpp"
#include "subproblems.hpp"

namespace mootation::scalar {

// A subproblem's own rows count as met within this (normalized units): a
// solver meets an equality to rounding, not exactly. The problem's
// constraints keep the vault's rule (cv <= 0).
inline constexpr double SLACK = 1e-9;

inline constexpr double kInf = std::numeric_limits<double>::infinity();

// One subproblem's bookkeeping: the points it evaluated, and their worth.
struct Run {
    const Sub& sub;
    Evaluator& ev;
    const Frame& frame;
    std::vector<double> width;
    std::vector<std::size_t> touched;

    Run(const Sub& s, Evaluator& e, const Frame& f) : sub(s), ev(e), frame(f)
    {
        for (std::size_t i = 0; i < ev.lo.size(); ++i) width.push_back(ev.hi[i] - ev.lo[i]);
    }

    std::vector<double> x_of(const std::vector<double>& y) const
    {
        std::vector<double> x(width.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = ev.lo[i] + std::min(1.0, std::max(0.0, y[i])) * width[i];
        return x;
    }

    std::size_t at(const std::vector<double>& y)
    {
        const std::size_t i = ev.index(x_of(y));
        touched.push_back(i);
        return i;
    }

    std::vector<double> y_of(std::size_t i) const
    {
        std::vector<double> y(width.size());
        for (std::size_t j = 0; j < y.size(); ++j) y[j] = (ev.X[i][j] - ev.lo[j]) / width[j];
        return y;
    }

    // (value, violation): the subproblem's merit, its rows' violation beyond
    // SLACK plus the problem's cv.
    void worth(std::size_t i, double& value, double& viol) const
    {
        if (ev.failed[i]) { value = viol = kInf; return; }
        double v, c;
        sub.merit(frame.bar(ev.F[i]), v, c);
        value = v;
        viol = (c > SLACK ? c : 0.0) + ev.cv(i);
    }
};

// The feasible point of least value, else the least violating one.
inline long long best_of(const Run& run, const std::vector<std::size_t>& indices)
{
    long long best = -1;
    int kb = 0;
    double vb = 0.0;
    std::size_t ib = 0;
    for (std::size_t i : indices) {
        double v, c;
        run.worth(i, v, c);
        const int k = c <= 0.0 ? 0 : 1;
        const double val = c <= 0.0 ? v : c;
        if (best < 0 || std::tie(k, val, i) < std::tie(kb, vb, ib)) {
            best = static_cast<long long>(i);
            kb = k;
            vb = val;
            ib = i;
        }
    }
    return best;
}

// Did the subproblem get better than its best start, beyond 1e-9 relative?
inline bool moved(const Run& run, const std::vector<std::size_t>& starts, std::size_t best)
{
    double v1, c1;
    run.worth(best, v1, c1);
    double v0 = 0.0, c0 = 0.0;
    bool first = true;
    for (std::size_t i : starts) {                  // min by (infeasible, violation, value)
        double v, c;
        run.worth(i, v, c);
        if (first || std::make_tuple(c > 0.0, c, v) < std::make_tuple(c0 > 0.0, c0, v0)) {
            v0 = v;
            c0 = c;
            first = false;
        }
    }
    if (c0 > 0.0) return c1 < c0 * (1.0 - 1e-9);
    return c1 <= 0.0 && v1 < v0 - 1e-9 * std::max(1.0, std::fabs(v0));
}

// The problem's own gradient, where sqp takes it (SCALAR-13): one objective
// and a gradient of it.
inline bool gradient_usable(const Evaluator& ev)
{
    return ev.m == 1 && ev.P.has_gradient();
}

// ── sqp ─────────────────────────────────────────────────────────────────────
inline void sqp(Run& run, const std::vector<double>& y0, double t0, double fd_step, double tol,
                slsqp::LineSearch line_search = slsqp::LineSearch::stop)
{
    const Sub& sub = run.sub;
    Evaluator& ev = run.ev;
    const std::size_t n = y0.size(), m = static_cast<std::size_t>(ev.m);
    const std::size_t k_ineq = static_cast<std::size_t>(ev.n_cons - ev.n_eq);
    const std::size_t n_eq = static_cast<std::size_t>(ev.n_eq);
    const std::size_t nz = n + (sub.aux ? 1 : 0);

    auto clip01 = [](std::vector<double> y) {
        for (double& v : y) v = std::min(1.0, std::max(0.0, v));
        return y;
    };
    auto point = [&](const std::vector<double>& z) {
        std::vector<double> y(z.begin(), z.begin() + static_cast<std::ptrdiff_t>(n));
        const std::size_t i = run.at(y);
        if (ev.failed[i]) throw FailedPoint{};
        return i;
    };
    auto t_of = [&](const std::vector<double>& z) { return sub.aux ? z[n] : 0.0; };

    // c = (sub's equalities, problem's equalities, sub's inequalities,
    // problem's inequalities - 1e-10 (SCALAR-8)) for F̄ and G at t
    auto rows_at = [&](const Vec& Fb, const Vec& G, double t, std::vector<double>& c) {
        c.clear();
        for (const Row& r : sub.eq) c.push_back(dot(r.a, Fb) + r.b * t + r.c);
        for (std::size_t j = 0; j < n_eq; ++j) c.push_back(G[k_ineq + j]);
        for (const Row& r : sub.ineq) c.push_back(dot(r.a, Fb) + r.b * t + r.c);
        for (std::size_t j = 0; j < k_ineq; ++j) c.push_back(-G[j] - 1e-10);
    };

    std::map<std::string, std::pair<lsq::Matrix, lsq::Matrix>> memo;   // y's bytes -> (JF, JG)
    const bool user_gradient = gradient_usable(ev);

    slsqp::NLP nlp;
    nlp.n = nz;
    nlp.n_eq = sub.eq.size() + n_eq;
    nlp.n_ineq = sub.ineq.size() + k_ineq;
    nlp.lo.assign(nz, 0.0);
    nlp.hi.assign(nz, 1.0);
    if (sub.aux) {
        nlp.lo[n] = -kInf;
        nlp.hi[n] = kInf;
    }
    nlp.values = [&](const std::vector<double>& z, std::vector<double>& c) {
        const std::size_t i = point(z);
        const Vec Fb = run.frame.bar(ev.F[i]);
        rows_at(Fb, ev.G[i], t_of(z), c);
        return dot(sub.a0, Fb) + sub.b0 * t_of(z) + sub.c0;
    };
    nlp.derivatives = [&](const std::vector<double>& z, std::vector<double>& g, lsq::Matrix& A) {
        std::vector<double> y = clip01(std::vector<double>(z.begin(), z.begin() + static_cast<std::ptrdiff_t>(n)));
        std::string key(n * sizeof(double), '\0');
        std::memcpy(&key[0], y.data(), key.size());
        auto it = memo.find(key);
        if (it == memo.end() && user_gradient) {          // SCALAR-13
            const std::size_t i0 = ev.index_grad(run.x_of(y));
            run.touched.push_back(i0);
            if (ev.failed[i0]) throw FailedPoint{};
            const Vec& gf = ev.grad(i0);
            const Rows& JGu = ev.jac_g(i0);
            if (JGu.size() != static_cast<std::size_t>(ev.n_cons))
                throw std::invalid_argument("gradient: the constraints' Jacobian needs n_cons = " +
                                            std::to_string(ev.n_cons) + " rows, got " + std::to_string(JGu.size()));
            lsq::Matrix JF(m, n), JG(static_cast<std::size_t>(ev.n_cons), n);
            for (std::size_t j = 0; j < n; ++j) {
                JF(0, j) = gf[j] * run.width[j] / run.frame.s[0];
                for (std::size_t r = 0; r < JG.rows; ++r) JG(r, j) = JGu[r][j] * run.width[j];
            }
            it = memo.emplace(key, std::make_pair(JF, JG)).first;
        }
        if (it == memo.end()) {
            const std::size_t i0 = point(z);
            const Vec Fb = run.frame.bar(ev.F[i0]);
            const Vec G0 = ev.G[i0];
            lsq::Matrix JF(m, n), JG(G0.size(), n);
            for (std::size_t j = 0; j < n; ++j) {
                double h = y[j] + fd_step <= 1.0 ? fd_step : -fd_step;
                auto at = [&](double step, std::size_t& out) {
                    std::vector<double> yj = y;
                    yj[j] += step;
                    out = run.at(yj);
                    return !ev.failed[out];
                };
                std::size_t i1 = 0;
                bool got = at(h, i1);
                if (!got && 0.0 <= y[j] - h && y[j] - h <= 1.0) {
                    h = -h;                                  // a failed step: the other way once
                    got = at(h, i1);
                }
                if (!got) throw FailedPoint{};
                const Vec Fj = run.frame.bar(ev.F[i1]);
                for (std::size_t r = 0; r < m; ++r) JF(r, j) = (Fj[r] - Fb[r]) / h;
                for (std::size_t r = 0; r < G0.size(); ++r) JG(r, j) = (ev.G[i1][r] - G0[r]) / h;
            }
            it = memo.emplace(key, std::make_pair(JF, JG)).first;
        }
        const lsq::Matrix& JF = it->second.first;
        const lsq::Matrix& JG = it->second.second;
        g.assign(nz, 0.0);
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t r = 0; r < m; ++r) g[j] += sub.a0[r] * JF(r, j);
        if (sub.aux) g[n] = sub.b0;
        A = lsq::Matrix(nlp.n_eq + nlp.n_ineq, nz);
        std::size_t row = 0;
        auto sub_row = [&](const Row& rw) {
            for (std::size_t j = 0; j < n; ++j) {
                double s = 0.0;
                for (std::size_t r = 0; r < m; ++r) s += rw.a[r] * JF(r, j);
                A(row, j) = s;
            }
            if (sub.aux) A(row, n) = rw.b;
            ++row;
        };
        for (const Row& rw : sub.eq) sub_row(rw);
        for (std::size_t q = 0; q < n_eq; ++q, ++row)
            for (std::size_t j = 0; j < n; ++j) A(row, j) = JG(k_ineq + q, j);
        for (const Row& rw : sub.ineq) sub_row(rw);
        for (std::size_t q = 0; q < k_ineq; ++q, ++row)
            for (std::size_t j = 0; j < n; ++j) A(row, j) = -JG(q, j);
    };
    std::vector<double> z0 = y0;
    if (sub.aux) z0.push_back(t0);
    try {
        slsqp::minimize(nlp, z0, {tol, 100000, line_search});
    } catch (const SubBudgetSpent&) {
    } catch (const FailedPoint&) {
    }
}

// ── nm and cmaes: a penalty in which feasible is better ─────────────────────
// value if feasible, else (worst feasible value known) + violation.
struct Penalty {
    Run& run;
    double worst = -kInf;

    Penalty(Run& r, const std::vector<std::size_t>& start) : run(r)
    {
        for (std::size_t i : start) {
            double v, c;
            run.worth(i, v, c);
            if (c <= 0.0 && std::isfinite(v)) worst = std::max(worst, v);
        }
        if (!std::isfinite(worst)) worst = 0.0;
    }

    double operator()(const std::vector<double>& y)
    {
        const std::size_t i = run.at(y);
        double v, c;
        run.worth(i, v, c);
        if (!std::isfinite(c)) return kInf;                 // a failed evaluation
        if (c <= 0.0) {
            worst = std::max(worst, v);
            return v;
        }
        return worst + c;
    }
};

// The penalty's value of a point grows as the run meets worse feasible values
// (Penalty::worst), so a simplex collapsed onto points already evaluated can
// keep a stale best value and cycle for ever without evaluating anything (a
// run of eps_nm on RE21 did, until memory ran out): the method stops after
// 100 (n + 1) calls in a row that evaluate nothing.
inline void nm(Run& run, const std::vector<double>& y0, const std::vector<std::size_t>& start, double tol)
{
    struct Idle {};
    Penalty pen(run, start);
    const std::size_t n = y0.size();
    nelder_mead::Options o;
    o.tol = tol;
    o.max_iter = 1000000000LL;
    const long long cap = 100 * static_cast<long long>(n + 1);
    long long idle = 0, spent = run.ev.spent;
    try {
        nelder_mead::minimize(
            [&](const std::vector<double>& y) {
                const double v = pen(y);
                if (run.ev.spent != spent) {
                    spent = run.ev.spent;
                    idle = 0;
                } else if (++idle > cap) {
                    throw Idle{};
                }
                return v;
            },
            y0, std::vector<double>(n, 0.0), std::vector<double>(n, 1.0), o);
    } catch (const SubBudgetSpent&) {
    } catch (const Idle&) {
    }
}

inline void cmaes(Run& run, const std::vector<double>& y0, const std::vector<std::size_t>& start,
                  double sigma0, double tol, std::mt19937& rng)
{
    Penalty pen(run, start);
    CMAES es(y0, sigma0, rng);
    try {
        while (es.spread() > tol) {
            const std::vector<std::vector<double>> Y = es.ask();
            std::vector<double> vals;
            for (const auto& y : Y) {
                std::vector<double> yc = y;
                double d2 = 0.0;
                for (std::size_t j = 0; j < yc.size(); ++j) {
                    yc[j] = std::min(1.0, std::max(0.0, yc[j]));
                    d2 += (y[j] - yc[j]) * (y[j] - yc[j]);
                }
                const double v = pen(yc);
                vals.push_back(std::isfinite(v) ? v + d2 * (1.0 + std::fabs(v)) : kInf);
            }
            es.tell(vals);
        }
    } catch (const SubBudgetSpent&) {
    }
}

}   // namespace mootation::scalar
