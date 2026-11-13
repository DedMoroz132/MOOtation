#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The best fit by a positive sum of kernel columns: an atomic measure (task
// 25, item 24.3.7):
//
//     minimize over a >= 0, lambda_k in [alpha, beta]:  1/2 || y - E(l) a ||_W^2,
//     E(l) = (e(lambda_1), ..., e(lambda_p)),  ||v||_W^2 = sum_i w_i v_i^2,
//
// by the algorithm of A. Ruhe, "Fitting empirical data by positive sums of
// exponentials", SIAM J. Sci. Stat. Comput. 1(4):481-498, 1980 (source
// fitting-empirical-data_ruhe1980; "R80" below), §4, Algorithms 1-3, with
// the grids and the tolerance of §5. One code for any kernel e(lambda) given
// with its first two derivatives in lambda: exp_kernel(t) is R80's
// e(lambda)_i = exp(-lambda t_i) (a Prony spectrum); a play operator, a creep
// element or Chaboche's e^{-gamma p} are kernels of the same form (the
// problems' side).
//
//   Phase 1 (Algorithm 1): the grid lambda_j = alpha + (beta - alpha)
//     exp(-(m - j)/d), j = 1..m, m = 10, d = 1, then m = 20, 40, ..., d = 2,
//     4, ... (§5); NNLS (Lawson-Hanson, lsq.hpp); each cluster of consecutive
//     positive coefficients condensed to one exponent, sum alpha_j lambda_j /
//     sum alpha_j.
//   Phase 2 (Algorithm 2, the least squares variant): Wedin's perturbation
//     method — the Gauss-Newton matrix J^T W J plus the absolute values of the
//     diagonal of the (2,2) block of the second part of the Hessian (R80
//     (3.7): |alpha_k rho''(lambda_k)|), solved as an augmented least squares
//     problem; a line search; converged when || J h ||_W < tol = 1e-7 (§5).
//     A coefficient that reaches 0 is a failure and restarts phase 1 on the
//     next grid; an exponent at alpha or beta is kept there, its column out
//     of the step (§4 after Algorithm 2).
//   Phase 3 (Algorithm 3): the certificate rho(lambda) = (y - y*, e(lambda))_W
//     <= 0 (R80 (2.2)) on a grid finer than phase 1's; if it fails, phase 1
//     again on the next grid.
//
// fit(...) is phase 2 alone, from given exponents: the best fit with a fixed
// number of atoms (D.W. Kammler, "Least squares approximation of completely
// monotonic functions by sums of exponentials", SIAM J. Numer. Anal. 16(5),
// 1979, Table 1, is a check of it: tests/test_atomic_measure.cpp).
//
// READINGS:
//   AM-1. Phase 2's start a = E(l)_W^+ y (Algorithm 2, step 1): when that has
//     a coefficient <= 0, the start is NNLS on E(l) instead and the exponents
//     with a zero coefficient are dropped (R80 assumes phase 1's clusters give
//     a positive solution).
//   AM-2. The line search (R80: "line search to determine mu_s", no rule):
//     mu = 1, 1/2, 1/4, ... (at most 40 halvings); a trial is taken when Phi
//     decreases and every coefficient stays > 0; exponents are clipped to
//     [alpha, beta], and one that lands on a bound stays there (§4). No
//     acceptable mu: the failure of "a coefficient hits zero" when the full
//     step would make one <= 0, else convergence at the current point.
//   AM-3. The certificate is checked with a tolerance and a scale: rho(lambda)
//     / (||y||_W ||e(lambda)||_W) <= tol_cert (1e-9), on the grid of phase 1's
//     formula with 10 m points and 10 d (the same span, ten times denser).
//     R80 leave the grid to the user ("nontrivial", Gustafson); the task's
//     preparation (04 Р8) asks for a tolerance of the order of the noise.
//   AM-4. Phase 2 runs at most max_iter = 100 steps; the convergence test is
//     R80's absolute || J h ||_W < tol (scale the data, or set tol).
//   AM-5. Phase 1's grids stop at m_max = 160; the result is then the last
//     phase 2's, with certified = false.
//   AM-6. NNLS takes the problem scaled by sqrt(w_i) row by row.
//   AM-7. Phase 2 converges quadratically only at the global optimum, where
//     -alpha_k rho''(lambda_k) >= 0 and |G| is the true (2,2) block. With
//     fewer atoms than the global optimum has (fit with a fixed p, as in
//     Kammler's table) the block has negative curvature (R80 §3, "at a local
//     optimum ... there will be negative curvatures"), |G| overstates it and
//     the steps converge linearly (Kammler n = 1: about 0.68 a step); as R80,
//     not changed here.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../detmath.hpp"
#include "../fp_contract.hpp"
#include "lsq.hpp"

namespace mootation::atomic {

// A kernel: the column e(lambda) and its first two derivatives in lambda
// (d1, d2 may be null: not needed)
using Kernel = std::function<void(double lambda, std::vector<double>& e, std::vector<double>* d1,
                                  std::vector<double>* d2)>;

// R80's e(lambda)_i = exp(-lambda t_i)
inline Kernel exp_kernel(const std::vector<double>& t)
{
    return [t](double lambda, std::vector<double>& e, std::vector<double>* d1, std::vector<double>* d2) {
        e.resize(t.size());
        if (d1) d1->resize(t.size());
        if (d2) d2->resize(t.size());
        for (std::size_t i = 0; i < t.size(); ++i) {
            e[i] = dm::exp(-lambda * t[i]);
            if (d1) (*d1)[i] = -t[i] * e[i];
            if (d2) (*d2)[i] = t[i] * t[i] * e[i];
        }
    };
}

struct Options {
    int m0 = 10;                  // the first grid (§5)
    double d0 = 1.0;
    int m_max = 160;              // AM-5
    double tol = 1e-7;            // AM-4
    int max_iter = 100;
    double tol_cert = 1e-9;       // AM-3
    int cert_factor = 10;
};

struct Atom {
    double lambda = 0.0, a = 0.0;
    bool at_bound = false;        // lambda at alpha or beta
};

struct Result {
    std::vector<Atom> atoms;      // by increasing lambda
    double residual = 0.0;        // || y - y* ||_W
    double rho_max = 0.0;         // the largest scaled rho on the certificate's grid (AM-3)
    bool certified = false;
    bool converged = false;       // phase 2
    int grids = 0;                // phase 1's grids tried
    int iterations = 0;           // phase 2's steps, the last run
};

namespace detail {

inline double wdot(const std::vector<double>& u, const std::vector<double>& v, const std::vector<double>& w)
{
    double s = 0.0;
    for (std::size_t i = 0; i < u.size(); ++i) s += w[i] * u[i] * v[i];
    return s;
}

// f = y - E(l) a and Phi = 1/2 ||f||_W^2
inline double residual(const Kernel& K, const std::vector<double>& y, const std::vector<double>& w,
                       const std::vector<double>& l, const std::vector<double>& a, std::vector<double>& f)
{
    f = y;
    std::vector<double> e;
    for (std::size_t k = 0; k < l.size(); ++k) {
        K(l[k], e, nullptr, nullptr);
        for (std::size_t i = 0; i < f.size(); ++i) f[i] -= a[k] * e[i];
    }
    return 0.5 * wdot(f, f, w);
}

// R80 §5's grid
inline std::vector<double> grid(double alpha, double beta, int m, double d)
{
    std::vector<double> g;
    for (int j = 1; j <= m; ++j) g.push_back(alpha + (beta - alpha) * dm::exp(-static_cast<double>(m - j) / d));
    return g;
}

// NNLS on sqrt(w) E (AM-6)
inline std::vector<double> nnls(const Kernel& K, const std::vector<double>& y, const std::vector<double>& w,
                                const std::vector<double>& l)
{
    lsq::Matrix E(y.size(), l.size());
    std::vector<double> e, ys(y.size());
    for (std::size_t k = 0; k < l.size(); ++k) {
        K(l[k], e, nullptr, nullptr);
        for (std::size_t i = 0; i < y.size(); ++i) E(i, k) = std::sqrt(w[i]) * e[i];
    }
    for (std::size_t i = 0; i < y.size(); ++i) ys[i] = std::sqrt(w[i]) * y[i];
    return lsq::nnls(E, ys).x;
}

}   // namespace detail

// Phase 2 alone (Algorithm 2): from the exponents l0 in [alpha, beta]
inline Result fit(const Kernel& K, const std::vector<double>& y, const std::vector<double>& w,
                  const std::vector<double>& l0, double alpha, double beta, const Options& o = {})
{
    const std::size_t n = y.size();
    if (w.size() != n) throw std::invalid_argument("atomic: a weight per datum");
    Result res;
    std::vector<double> l = l0;
    std::sort(l.begin(), l.end());

    // the start: a = E(l)_W^+ y, else NNLS (AM-1)
    std::vector<double> a;
    {
        lsq::Matrix E(n, l.size());
        std::vector<double> e, ys(n);
        for (std::size_t k = 0; k < l.size(); ++k) {
            K(l[k], e, nullptr, nullptr);
            for (std::size_t i = 0; i < n; ++i) E(i, k) = std::sqrt(w[i]) * e[i];
        }
        for (std::size_t i = 0; i < n; ++i) ys[i] = std::sqrt(w[i]) * y[i];
        bool positive = lsq::least_squares(E, ys, a);
        for (double v : a) positive = positive && v > 0.0;
        if (!positive) {
            a = lsq::nnls(E, ys).x;
            std::vector<double> l2, a2;
            for (std::size_t k = 0; k < l.size(); ++k)
                if (a[k] > 0.0) {
                    l2.push_back(l[k]);
                    a2.push_back(a[k]);
                }
            l = l2;
            a = a2;
        }
    }
    const std::size_t p = l.size();
    std::vector<bool> fixed(p, false);
    for (std::size_t k = 0; k < p; ++k) fixed[k] = l[k] <= alpha || l[k] >= beta;

    std::vector<double> f;
    double phi = detail::residual(K, y, w, l, a, f);
    std::vector<double> e, d1, d2, ft;
    bool failed = false;
    for (int it = 0; it < o.max_iter && p > 0; ++it) {
        // the augmented least squares problem [sqrt(W) J; sqrt(D)] h = [-sqrt(W) f; 0]
        std::vector<std::size_t> free_l;
        for (std::size_t k = 0; k < p; ++k)
            if (!fixed[k]) free_l.push_back(k);
        const std::size_t nv = p + free_l.size();
        lsq::Matrix A(n + free_l.size(), nv);
        std::vector<double> rhs(n + free_l.size(), 0.0);
        std::vector<std::vector<double>> cols_e(p), cols_d1(p);
        std::vector<double> dkk(p, 0.0);
        for (std::size_t k = 0; k < p; ++k) {
            K(l[k], e, &d1, &d2);
            cols_e[k] = e;
            cols_d1[k] = d1;
            dkk[k] = std::fabs(a[k] * detail::wdot(f, d2, w));            // |alpha_k rho''(lambda_k)|
        }
        for (std::size_t i = 0; i < n; ++i) {
            const double s = std::sqrt(w[i]);
            for (std::size_t k = 0; k < p; ++k) A(i, k) = -s * cols_e[k][i];
            for (std::size_t j = 0; j < free_l.size(); ++j) A(i, p + j) = -s * a[free_l[j]] * cols_d1[free_l[j]][i];
            rhs[i] = -s * f[i];
        }
        for (std::size_t j = 0; j < free_l.size(); ++j) A(n + j, p + j) = std::sqrt(dkk[free_l[j]]);
        std::vector<double> h;
        if (!lsq::least_squares(A, rhs, h)) h = lsq::hfti(A, rhs);
        ++res.iterations;

        // || J h ||_W
        double jh2 = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (std::size_t c = 0; c < nv; ++c) s += A(i, c) * h[c];
            jh2 += s * s;
        }
        if (std::sqrt(jh2) < o.tol) {
            res.converged = true;
            break;
        }

        // the line search (AM-2)
        bool full_nonpositive = false;
        for (std::size_t k = 0; k < p; ++k) full_nonpositive = full_nonpositive || a[k] + h[k] <= 0.0;
        double mu = 1.0;
        bool taken = false;
        std::vector<double> at(p), lt(p);
        for (int s = 0; s <= 40 && !taken; ++s, mu *= 0.5) {
            bool positive = true;
            for (std::size_t k = 0; k < p; ++k) {
                at[k] = a[k] + mu * h[k];
                positive = positive && at[k] > 0.0;
            }
            if (!positive) continue;
            lt = l;
            for (std::size_t j = 0; j < free_l.size(); ++j) {
                const std::size_t k = free_l[j];
                lt[k] = std::min(beta, std::max(alpha, l[k] + mu * h[p + j]));
            }
            const double pt = detail::residual(K, y, w, lt, at, ft);
            if (pt < phi) {
                taken = true;
                a = at;
                l = lt;
                f = ft;
                phi = pt;
                for (std::size_t k = 0; k < p; ++k) fixed[k] = fixed[k] || l[k] <= alpha || l[k] >= beta;
            }
        }
        if (!taken) {
            if (full_nonpositive) failed = true;
            else res.converged = true;
            break;
        }
    }
    res.converged = res.converged && !failed;

    std::vector<std::size_t> order(p);
    for (std::size_t k = 0; k < p; ++k) order[k] = k;
    std::sort(order.begin(), order.end(), [&](std::size_t i, std::size_t j) { return l[i] < l[j]; });
    for (std::size_t k : order) res.atoms.push_back({l[k], a[k], l[k] <= alpha || l[k] >= beta});
    res.residual = std::sqrt(2.0 * phi);
    return res;
}

// The largest scaled rho(lambda) over the grid (AM-3)
inline double rho_max(const Kernel& K, const std::vector<double>& y, const std::vector<double>& w,
                      const Result& r, const std::vector<double>& lambdas)
{
    std::vector<double> l, a, f, e;
    for (const Atom& at : r.atoms) {
        l.push_back(at.lambda);
        a.push_back(at.a);
    }
    detail::residual(K, y, w, l, a, f);
    const double ny = std::sqrt(detail::wdot(y, y, w));
    double worst = -std::numeric_limits<double>::infinity();
    for (double lam : lambdas) {
        K(lam, e, nullptr, nullptr);
        const double ne = std::sqrt(detail::wdot(e, e, w));
        if (ne == 0.0 || ny == 0.0) continue;
        worst = std::max(worst, detail::wdot(f, e, w) / (ny * ne));
    }
    return worst;
}

// The whole algorithm: phase 1, 2, 3 on refining grids
inline Result solve(const Kernel& K, const std::vector<double>& y, const std::vector<double>& w, double alpha,
                    double beta, const Options& o = {})
{
    if (!(alpha < beta)) throw std::invalid_argument("atomic: alpha < beta");
    Result best;
    bool have = false;
    int grids = 0;
    for (int m = o.m0; m <= o.m_max; m *= 2) {
        const double d = o.d0 * static_cast<double>(m) / static_cast<double>(o.m0);
        ++grids;
        // phase 1
        const std::vector<double> g = detail::grid(alpha, beta, m, d);
        const std::vector<double> c = detail::nnls(K, y, w, g);
        std::vector<double> l0;
        for (std::size_t j = 0; j < g.size();) {
            if (c[j] <= 0.0) {
                ++j;
                continue;
            }
            double s = 0.0, sl = 0.0;
            while (j < g.size() && c[j] > 0.0) {
                s += c[j];
                sl += c[j] * g[j];
                ++j;
            }
            l0.push_back(sl / s);
        }
        if (l0.empty()) continue;
        // phase 2
        Result r = fit(K, y, w, l0, alpha, beta, o);
        // phase 3
        r.rho_max = rho_max(K, y, w, r, detail::grid(alpha, beta, o.cert_factor * m, o.cert_factor * d));
        r.certified = r.converged && r.rho_max <= o.tol_cert;
        r.grids = grids;
        if (!have || r.residual < best.residual || r.certified) {
            best = r;
            have = true;
        }
        if (r.certified) return r;
    }
    best.grids = grids;
    return best;
}

}   // namespace mootation::atomic
