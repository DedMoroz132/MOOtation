#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// SQP with a least-squares subproblem — the method D. Kraft's SLSQP stands for
// (D. Kraft, "Algorithm 733: TOMP — Fortran modules for optimal control
// calculations", ACM TOMS 20(3):262-281, 1994, §3: "(NLP) is solved by
// sequential linear least squares problems ... using a slightly modified NNLS
// of Lawson and Hanson"; source slsqp-tomp_kraft1994), written from the
// theory, not from Kraft's code: Kraft 1994 describes the interface and not
// the algorithm, whose report (DFVLR-FB 88-28, 1988) is not at hand. The
// iteration is the line-search SQP of J. Nocedal, S.J. Wright, Numerical
// Optimization, 2nd ed., Springer 2006 (source numerical-optimization_
// nocedal2006; "NW"), Algorithm 18.3; its quadratic subproblem is solved as a
// least-squares problem of Lawson & Hanson (lsq.hpp).
//
//   min f(x)  s.t.  c_E(x) = 0,  c_I(x) >= 0,  lo <= x <= hi
//
// One iteration at x, with B positive definite (B_0 = I):
//   1. the step d of the QP (NW (18.11)) min grad_f^T d + d^T B d / 2 under
//      A_E d + c_E = 0, A_I d + c_I >= 0 and lo - x <= d <= hi - x, solved as
//      min ||R d + R^-T grad_f|| (B = R^T R, Cholesky) under the same rows:
//      lsq::lsei, which returns the multipliers lambda as well;
//   2. the penalty mu of the l1 merit phi(x) = f + mu (sum |c_E| + sum
//      max(0, -c_I)) (NW (18.27)): the least that satisfies (18.36) with
//      sigma = 1 (18.37) and rho = 1/2, doubled when it has to grow; mu never
//      decreases;
//   3. the step length: alpha = 1, then by the minimizer of the quadratic
//      through phi(0), phi'(0) and phi(alpha) (NW (3.58)), at least 0.1 alpha
//      (NW Exercise 3.14), until Armijo's condition (18.28) holds with
//      eta = 1e-4 (NW (3.4)); at most 10 trials (NW §3.5); phi'(0) is the
//      directional derivative (18.29);
//   4. B by the damped BFGS update (NW Procedure 18.2) with s = x+ - x and y
//      the change of the gradient of the Lagrangian f - lambda^T c at the new
//      multipliers (18.13).
//
// The derivatives are asked for at the accepted points only; values at every
// trial point (an evaluation each, as a caller counting evaluations sees).
//
// OURS (Kraft 1994 and NW leave them open; SciPy's SLSQP, Kraft's code, makes
// other choices for some of them):
//   SQP-1. Incompatible linearized constraints: the QP is solved again with
//     one more variable delta in [0, 1], the equalities and the violated
//     inequalities taken as a^T d + (1 - delta) c, and rho delta^2 / 2 added to
//     the objective, rho = 1e4 (the relaxation of Powell 1978, which Kraft's
//     code follows; NW (18.12) is the related elastic mode, whose linear
//     penalty does not fit a least-squares objective). d = 0, delta = 1 is
//     always feasible.
//   SQP-2. B is reset to I and the iteration made again when the QP fails,
//     when no step length is found, or when B loses positive definiteness
//     to round-off; a second failure in a row ends the run.
//   SQP-3. Stopping: after the QP when |d|_inf <= tol and the violation
//     (sum |c_E| + sum max(0, -c_I)) <= tol; after a step when |f+ - f| <= tol
//     and the violation <= tol; when a step leaves x unchanged bit for bit;
//     after max_iter iterations. tol plays the part of SciPy's ftol.
//   SQP-4. The bounds are rows of the QP (infinite ones none); the point
//     x + alpha d is clipped to the box (a step of an ulp at most: the QP's
//     step keeps the box) and the start too.
//   SQP-5. No scaling of x, f or c: the caller's.
//   SQP-6. When no step length passes Armijo's test in 10 trials, the run
//     stops (LineSearch::stop, NW §3.5's stopping test; with B reset first,
//     SQP-2) or takes the last trial point and goes on (LineSearch::accept:
//     what Kraft's code does). The knob sqp_line_search of the scalarization
//     methods; its third value, powell, is the next paragraph.
//
// LineSearch::powell — steps 2 and 3 as M.J.D. Powell, "A fast algorithm for
// nonlinearly constrained optimization calculations", in: G.A. Watson (ed.),
// Numerical Analysis, Dundee 1977, LNM 630, Springer 1978, pp. 144-157
// (source sqp_powell1978), §4, whose rule Kraft's code follows:
//   2. a penalty mu_i of its own for every constraint, in the merit (4.2)
//      phi(x) = f + sum_E mu_i |c_i| + sum_I mu_i |min(0, c_i)|: mu_i = |lambda_i|
//      on the first iteration, then (4.6) mu_i = max(|lambda_i|,
//      (mu_i + |lambda_i|) / 2), mu_i the previous iteration's;
//   3. alpha_0 = 1, alpha_k the greater of 0.1 alpha_{k-1} and the minimizer
//      of the quadratic through phi(0), Delta and phi(alpha_{k-1}) (4.7),
//      until phi(alpha_k) <= phi(0) + 0.1 alpha_k Delta (4.8); Delta is
//      phi(1) - phi(0) with f and c taken linear (§4, after (4.8)), not the
//      derivative; no limit on the trials, as in the paper.
// OURS for it:
//   SQP-7. The trials end, failed, when x + alpha d is x bit for bit (alpha
//     falls by 1.8 at least a trial, so this comes); then SQP-2 as for stop.
//     Delta >= 0 (a relaxed QP's step, SQP-1) fails the same way at once.
//   SQP-8. An iteration made again after a reset (SQP-2) takes (4.6) from the
//     mu_i of the last finished iteration, not from those of the failed try.
//   SQP-9. Not done: the paper's trap for cycling (an error return after five
//     iterations with mu constant and no decrease of phi; "This error return
//     has never occurred", §4). Steps 1 and 4 and SQP-1 to SQP-5 are as above.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

#include "../fp_contract.hpp"
#include "lsq.hpp"

namespace mootation::slsqp {

using lsq::Matrix;

struct NLP {
    std::size_t n = 0, n_eq = 0, n_ineq = 0;
    std::vector<double> lo, hi;          // +-infinity: no bound
    // f(x), and c = (c_E, c_I) into c (n_eq + n_ineq values)
    std::function<double(const std::vector<double>& x, std::vector<double>& c)> values;
    // grad f into g, the Jacobian of c (n_eq + n_ineq rows, n columns) into A
    std::function<void(const std::vector<double>& x, std::vector<double>& g, Matrix& A)> derivatives;
};

enum class LineSearch { stop, accept, powell };   // SQP-6; powell: Powell 1978, §4

struct Options {
    double tol = 1e-6;
    int max_iter = 100;
    LineSearch line_search = LineSearch::stop;
};

enum class Status { converged, max_iter, qp_failed, line_search_failed, no_progress };

struct Result {
    std::vector<double> x;
    double f = 0.0;
    std::vector<double> c;
    Status status = Status::max_iter;
    int iterations = 0;
    int relaxed = 0;                     // QPs solved relaxed (SQP-1)
    int forced = 0;                      // steps taken without Armijo's decrease (SQP-6)
};

inline constexpr double kEta = 1e-4;     // Armijo's constant, NW (3.4)
inline constexpr double kRho = 0.5;      // NW (18.36)
inline constexpr double kRelax = 1e4;    // SQP-1
inline constexpr int kTrials = 10;       // NW §3.5
inline constexpr double kEtaPowell = 0.1;   // Powell 1978, (4.8)

namespace detail {

inline double violation(const std::vector<double>& c, std::size_t me)
{
    double v = 0.0;
    for (std::size_t j = 0; j < c.size(); ++j)
        v += j < me ? std::fabs(c[j]) : std::max(0.0, -c[j]);
    return v;
}

// The penalty part of Powell's merit (4.2), mu_i a constraint.
inline double penalty(const std::vector<double>& c, const std::vector<double>& mu, std::size_t me)
{
    double v = 0.0;
    for (std::size_t j = 0; j < c.size(); ++j)
        v += mu[j] * (j < me ? std::fabs(c[j]) : std::max(0.0, -c[j]));
    return v;
}

// B = R^T R with R upper triangular; false when B is not (numerically)
// positive definite.
inline bool cholesky(const Matrix& B, Matrix& R)
{
    const std::size_t n = B.rows;
    R = Matrix(n, n);
    for (std::size_t j = 0; j < n; ++j) {
        double s = B(j, j);
        for (std::size_t k = 0; k < j; ++k) s -= R(k, j) * R(k, j);
        if (!(s > 0.0)) return false;
        R(j, j) = std::sqrt(s);
        for (std::size_t i = j + 1; i < n; ++i) {
            double t = B(j, i);
            for (std::size_t k = 0; k < j; ++k) t -= R(k, j) * R(k, i);
            R(j, i) = t / R(j, j);
        }
    }
    return true;
}

struct Step {
    std::vector<double> d, lam;          // lam: c_E's, then c_I's
    bool ok = false, relaxed = false;
};

// The QP of step 1 (relaxed by SQP-1 when its rows are incompatible).
inline Step qp(const Matrix& B, const std::vector<double>& g, const Matrix& A,
               const std::vector<double>& c, const std::vector<double>& x,
               const std::vector<double>& lo, const std::vector<double>& hi, std::size_t me)
{
    const std::size_t n = x.size(), mi = c.size() - me;
    Step st;
    Matrix R;
    if (!cholesky(B, R)) return st;
    std::vector<double> fl = lsq::solve_upper_transposed(R, n, g);   // R^T v = g
    for (double& v : fl) v = -v;

    std::size_t nb = 0;
    for (std::size_t i = 0; i < n; ++i)
        nb += static_cast<std::size_t>(std::isfinite(lo[i])) + static_cast<std::size_t>(std::isfinite(hi[i]));

    for (int relax = 0; relax < 2; ++relax) {
        const std::size_t k = n + static_cast<std::size_t>(relax);
        Matrix E(k, k), C(me, k), G(mi + nb + 2 * static_cast<std::size_t>(relax), k);
        std::vector<double> f(k, 0.0), dC(me), h(G.rows);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i; j < n; ++j) E(i, j) = R(i, j);
            f[i] = fl[i];
        }
        if (relax) E(n, n) = std::sqrt(kRelax);
        for (std::size_t r = 0; r < me; ++r) {
            for (std::size_t j = 0; j < n; ++j) C(r, j) = A(r, j);
            if (relax) C(r, n) = -c[r];
            dC[r] = -c[r];
        }
        std::size_t row = 0;
        for (std::size_t r = 0; r < mi; ++r, ++row) {
            for (std::size_t j = 0; j < n; ++j) G(row, j) = A(me + r, j);
            if (relax && c[me + r] < 0.0) G(row, n) = -c[me + r];
            h[row] = -c[me + r];
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (std::isfinite(lo[i])) { G(row, i) = 1.0; h[row] = lo[i] - x[i]; ++row; }
            if (std::isfinite(hi[i])) { G(row, i) = -1.0; h[row] = x[i] - hi[i]; ++row; }
        }
        if (relax) {
            G(row, n) = 1.0;  h[row] = 0.0;  ++row;          // delta >= 0
            G(row, n) = -1.0; h[row] = -1.0; ++row;          // delta <= 1
        }
        lsq::LSIResult s = lsq::lsei(E, f, C, dC, G, h);
        if (s.status == lsq::Status::ok) {
            st.d.assign(s.x.begin(), s.x.begin() + static_cast<std::ptrdiff_t>(n));
            st.lam.assign(me + mi, 0.0);
            for (std::size_t r = 0; r < me; ++r) st.lam[r] = s.lam_C[r];
            for (std::size_t r = 0; r < mi; ++r) st.lam[me + r] = s.lam_G[r];
            st.ok = true;
            st.relaxed = relax != 0;
            return st;
        }
        if (s.status != lsq::Status::incompatible) return st;
    }
    return st;
}

inline void clip(std::vector<double>& x, const std::vector<double>& lo, const std::vector<double>& hi)
{
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i] < lo[i]) x[i] = lo[i];
        if (x[i] > hi[i]) x[i] = hi[i];
    }
}

}   // namespace detail

inline Result minimize(const NLP& p, std::vector<double> x, const Options& o)
{
    const std::size_t n = p.n, me = p.n_eq, m = p.n_eq + p.n_ineq;
    Result r;
    detail::clip(x, p.lo, p.hi);
    std::vector<double> c(m), g(n);
    Matrix A(m, n);
    double f = p.values(x, c);
    p.derivatives(x, g, A);
    Matrix B(n, n);
    for (std::size_t i = 0; i < n; ++i) B(i, i) = 1.0;
    bool fresh = true;                                   // B = I since the last reset
    const bool powell = o.line_search == LineSearch::powell;
    double mu = 0.0;                                     // NW (18.36)
    std::vector<double> mus;                             // Powell (4.6); empty before the first iteration

    auto reset = [&]() {
        B = Matrix(n, n);
        for (std::size_t i = 0; i < n; ++i) B(i, i) = 1.0;
        fresh = true;
    };

    while (true) {
        if (r.iterations >= o.max_iter) { r.status = Status::max_iter; break; }
        ++r.iterations;
        detail::Step st = detail::qp(B, g, A, c, x, p.lo, p.hi, me);
        if (!st.ok) {
            if (fresh) { r.status = Status::qp_failed; break; }
            reset();                                     // SQP-2
            --r.iterations;
            continue;
        }
        r.relaxed += st.relaxed;
        const double viol = detail::violation(c, me);
        double dmax = 0.0;
        for (double v : st.d) dmax = std::max(dmax, std::fabs(v));
        if (dmax <= o.tol && viol <= o.tol) { r.status = Status::converged; break; }   // SQP-3

        double gd = 0.0;
        for (std::size_t i = 0; i < n; ++i) gd += g[i] * st.d[i];
        double slope = 0.0, phi0 = 0.0;
        std::vector<double> mut;                         // the mu_i of this iteration (Powell)
        if (powell) {
            // the penalties (4.6), SQP-8
            mut.resize(m);
            for (std::size_t j = 0; j < m; ++j) {
                const double l = std::fabs(st.lam[j]);
                mut[j] = mus.empty() ? l : std::max(l, 0.5 * (mus[j] + l));
            }
            // Delta: phi(1) - phi(0) with f and c linear
            std::vector<double> cl(m);
            for (std::size_t j = 0; j < m; ++j) {
                double v = c[j];
                for (std::size_t i = 0; i < n; ++i) v += A(j, i) * st.d[i];
                cl[j] = v;
            }
            const double p0 = detail::penalty(c, mut, me);
            slope = gd + (detail::penalty(cl, mut, me) - p0);
            phi0 = f + p0;
        } else {
            // the penalty (NW (18.36), sigma = 1)
            double dBd = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                double s = 0.0;
                for (std::size_t j = 0; j < n; ++j) s += B(i, j) * st.d[j];
                dBd += st.d[i] * s;
            }
            if (viol > 0.0) {
                const double need = (gd + 0.5 * dBd) / ((1.0 - kRho) * viol);
                if (mu < need) mu = 2.0 * need;
            }
            slope = gd - mu * viol;                      // NW (18.29)
            phi0 = f + mu * viol;
        }
        const double eta = powell ? kEtaPowell : kEta;

        // the step length
        bool accepted = false, evaluated = false;
        double alpha = 1.0, ft = 0.0;
        std::vector<double> xt(n), ct(m);
        if (slope < 0.0) {
            for (int trial = 0; powell || trial < kTrials; ++trial) {   // SQP-7
                for (std::size_t i = 0; i < n; ++i) xt[i] = x[i] + alpha * st.d[i];
                detail::clip(xt, p.lo, p.hi);
                if (xt == x) break;
                ft = p.values(xt, ct);
                evaluated = true;
                const double phi = ft + (powell ? detail::penalty(ct, mut, me)
                                                : mu * detail::violation(ct, me));
                if (phi <= phi0 + eta * alpha * slope) { accepted = true; break; }
                const double a = -slope * alpha * alpha / (2.0 * (phi - phi0 - slope * alpha));
                alpha = std::isfinite(a) ? std::max(0.1 * alpha, a) : 0.1 * alpha;
            }
        }
        if (!accepted && o.line_search == LineSearch::accept && evaluated && !(xt == x)) {
            accepted = true;                             // SQP-6: the last trial point
            ++r.forced;
        }
        if (!accepted) {
            if (fresh) {
                r.status = xt == x ? Status::no_progress : Status::line_search_failed;
                break;
            }
            reset();                                     // SQP-2
            --r.iterations;
            continue;
        }
        if (powell) mus = mut;
        const bool done = std::fabs(ft - f) <= o.tol && detail::violation(ct, me) <= o.tol;
        std::vector<double> s(n);
        for (std::size_t i = 0; i < n; ++i) s[i] = xt[i] - x[i];
        x = xt;
        f = ft;
        c = ct;
        if (done) { r.status = Status::converged; break; }   // SQP-3

        // the damped BFGS update (NW Procedure 18.2)
        std::vector<double> gn(n);
        Matrix An(m, n);
        p.derivatives(x, gn, An);
        std::vector<double> y(n), Bs(n);
        for (std::size_t i = 0; i < n; ++i) {
            double v = gn[i] - g[i];
            for (std::size_t j = 0; j < m; ++j) v -= (An(j, i) - A(j, i)) * st.lam[j];
            y[i] = v;
            double b = 0.0;
            for (std::size_t j = 0; j < n; ++j) b += B(i, j) * s[j];
            Bs[i] = b;
        }
        double sBs = 0.0, sy = 0.0;
        for (std::size_t i = 0; i < n; ++i) { sBs += s[i] * Bs[i]; sy += s[i] * y[i]; }
        if (sBs > 0.0) {
            const double theta = sy >= 0.2 * sBs ? 1.0 : 0.8 * sBs / (sBs - sy);   // (18.15)
            std::vector<double> rv(n);
            double sr = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                rv[i] = theta * y[i] + (1.0 - theta) * Bs[i];
                sr += s[i] * rv[i];
            }
            if (sr > 0.0) {
                for (std::size_t i = 0; i < n; ++i)
                    for (std::size_t j = 0; j < n; ++j)
                        B(i, j) += rv[i] * rv[j] / sr - Bs[i] * Bs[j] / sBs;   // (18.16)
                fresh = false;
            }
        }
        g = gn;
        A = An;
    }
    r.x = x;
    r.f = f;
    r.c = c;
    return r;
}

}   // namespace mootation::slsqp
