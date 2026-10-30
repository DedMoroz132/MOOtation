#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// L-BFGS-B (task 18, stage 2): min f(x) over l <= x <= u, the bounds -inf/+inf
// where there are none, the gradient the caller's (or forward differences).
//
// R.H. Byrd, P. Lu, J. Nocedal, C. Zhu, "A limited memory algorithm for bound
// constrained optimization", SIAM J. Sci. Comput. 16(5) 1995, pp. 1190-1208
// (source lbfgsb_byrd1995; "BLNZ"), with the subspace step of J.L. Morales,
// J. Nocedal, "Remark on Algorithm 778", ACM TOMS 38(1) 2011 (source
// lbfgsb-remark_morales2011; "MN") and the line search of Moré & Thuente
// 1994 (line_search.hpp), as BLNZ §6 use it. One iteration at x (BLNZ §6,
// "L-BFGS-B Algorithm"):
//   1. stop when ||P(x - g, l, u) - x||_inf < pgtol (BLNZ (6.1));
//   2. the generalized Cauchy point x^c by Algorithm CP (BLNZ §4), B = theta I
//      - W M W^T the compact limited memory matrix (BLNZ (3.2)-(3.6));
//   3. the subspace step by the direct primal method (BLNZ §5.1): the free
//      variables at x^c, the reduced gradient (5.4), the unconstrained
//      minimizer d^u = -B^-1 r^c by Sherman-Morrison-Woodbury (5.10); then MN
//      §1.1: xbar = P(x^c + Z d^u), d = xbar - x, unless d is not a descent
//      direction, where BLNZ's truncation (5.8)-(5.9) gives xbar;
//   4. the Moré-Thuente search along d from the unit step, (2.5) with
//      alpha = 1e-4, (2.6) with beta = 0.9 (BLNZ step 4), the largest step the
//      one to the nearest bound along d (BLNZ §6);
//   5. the pair (s, y) kept if s^T y > eps ||y||^2, eps = 2.2e-16 (BLNZ (3.9),
//      step 6), the oldest dropped beyond m; theta = y^T y / y^T s (step 7).
//
// Readings:
//   LB-1. (5.11) as printed has the sign of B^-1 r^c; d^u = -B^-1 r^c is
//     (5.7)'s, which we follow.
//   LB-2. The free variables at x^c are those strictly between their bounds
//     there (Algorithm CP's last lines leave the set to be read this way).
//   LB-3. MN's "not a direction of strong descent": g^T d >= 0.
//   LB-4. OURS: where even the truncated d is not a descent direction (B
//     lost positive definiteness to rounding), the pairs are dropped and the
//     iteration made again with B = I.
//   LB-5. OURS: the breakpoints are sorted (BLNZ use a heap: the same order;
//     ties by index); m = 5 (BLNZ's Table 1, first column); a line search
//     that ends without a lower f ends the run (BLNZ's F2); the point of the
//     search is projected onto the box against rounding; ftol (off by
//     default) stops where (f_k - f_k+1) / max(|f_k|, |f_k+1|, 1) <= ftol
//     (the test Algorithm 778 has; that paper is not at hand); the forward
//     difference gradient takes the relative step sqrt(eps) |x_j| (sqrt(eps)
//     at 0), backward where x_j + h passes u_j, n evaluations.
//   LB-7. M of (3.4) by the Schur complement of -D, T = theta S^T S + L D^-1
//     L^T (positive definite: s_i^T y_i > 0), by Cholesky (the factorization
//     of Byrd, Nocedal, Schnabel, BLNZ's [6]; BLNZ only say the inverse is
//     cheap): an elimination of the whole indefinite matrix lost B's
//     positive definiteness where theta is large (Brown's badly scaled).
//   LB-6. OURS: the new point is the one the search accepts; where it stops
//     short of (2.5)-(2.6) (its own limits), the trial with the lowest f.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../fp_contract.hpp"
#include "line_search.hpp"

namespace mootation::scalar::lbfgsb {

using Vec = std::vector<double>;
inline constexpr double kInf = std::numeric_limits<double>::infinity();
inline constexpr double kEps = std::numeric_limits<double>::epsilon();

// f, or f and its gradient in one call (fg); empty bounds: none
struct Problem {
    std::size_t n = 0;
    std::function<double(const Vec& x)> f;
    std::function<double(const Vec& x, Vec& g)> fg;   // empty: forward differences of f
    Vec lo, hi;
};

struct Options {
    int m = 5;                              // LB-5
    double pgtol = 1e-5;                    // BLNZ (6.1)
    double ftol = 0.0;                      // LB-5, off
    long long max_nfev = 15000;             // evaluations of f (the differences' included)
    long long max_iter = 15000;
    double alpha = 1e-4, beta = 0.9;        // BLNZ (2.5), (2.6)
};

enum class Status { pgtol, ftol, line_search, max_nfev, max_iter };

struct Result {
    Vec x, g;
    double f = 0.0;
    long long nfev = 0, ngev = 0, iterations = 0;
    Status status = Status::max_iter;
};

namespace detail {

// A x = b for a small dense A (row by row, k x k), Gaussian elimination with partial pivoting
inline Vec solve(std::vector<double> A, Vec b)
{
    const std::size_t k = b.size();
    for (std::size_t c = 0; c < k; ++c) {
        std::size_t p = c;
        for (std::size_t r = c + 1; r < k; ++r)
            if (std::fabs(A[r * k + c]) > std::fabs(A[p * k + c])) p = r;
        if (p != c) {
            for (std::size_t j = 0; j < k; ++j) std::swap(A[c * k + j], A[p * k + j]);
            std::swap(b[c], b[p]);
        }
        for (std::size_t r = c + 1; r < k; ++r) {
            const double t = A[r * k + c] / A[c * k + c];
            for (std::size_t j = c; j < k; ++j) A[r * k + j] -= t * A[c * k + j];
            b[r] -= t * b[c];
        }
    }
    for (std::size_t c = k; c-- > 0;) {
        double s = b[c];
        for (std::size_t j = c + 1; j < k; ++j) s -= A[c * k + j] * b[j];
        b[c] = s / A[c * k + c];
    }
    return b;
}

// B = theta I - W M W^T, W = [Y theta S] (BLNZ (3.2)-(3.6)); M kept explicitly (2k x 2k)
struct Memory {
    std::vector<Vec> S, Y;                  // oldest first
    double theta = 1.0;
    std::vector<double> M;                  // row by row

    std::size_t k() const { return S.size(); }
    // W^T v
    Vec wt(const Vec& v) const
    {
        const std::size_t kk = k();
        Vec r(2 * kk, 0.0);
        for (std::size_t j = 0; j < kk; ++j) {
            double a = 0.0, b = 0.0;
            for (std::size_t i = 0; i < v.size(); ++i) {
                a += Y[j][i] * v[i];
                b += S[j][i] * v[i];
            }
            r[j] = a;
            r[kk + j] = theta * b;
        }
        return r;
    }
    Vec w_row(std::size_t b) const          // row b of W
    {
        const std::size_t kk = k();
        Vec r(2 * kk);
        for (std::size_t j = 0; j < kk; ++j) {
            r[j] = Y[j][b];
            r[kk + j] = theta * S[j][b];
        }
        return r;
    }
    Vec mul_m(const Vec& v) const
    {
        const std::size_t q = v.size();
        Vec r(q, 0.0);
        for (std::size_t i = 0; i < q; ++i)
            for (std::size_t j = 0; j < q; ++j) r[i] += M[i * q + j] * v[j];
        return r;
    }
    // M = [[-D, L^T], [L, theta S^T S]]^-1 (BLNZ (3.4)-(3.6)) by its Schur
    // complement T = theta S^T S + L D^-1 L^T, positive definite (LB-7):
    // M = [[-D^-1 + D^-1 L^T T^-1 L D^-1, D^-1 L^T T^-1], [T^-1 L D^-1, T^-1]]
    void form()
    {
        const std::size_t kk = k(), q = 2 * kk;
        auto dot = [](const Vec& a, const Vec& b) {
            double s = 0.0;
            for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
            return s;
        };
        Vec D(kk);
        std::vector<double> L(kk * kk, 0.0), T(kk * kk, 0.0);
        for (std::size_t i = 0; i < kk; ++i) {
            D[i] = dot(S[i], Y[i]);
            for (std::size_t j = 0; j < i; ++j) L[i * kk + j] = dot(S[i], Y[j]);
        }
        for (std::size_t i = 0; i < kk; ++i)
            for (std::size_t j = 0; j < kk; ++j) {
                double s = theta * dot(S[i], S[j]);
                for (std::size_t l = 0; l < kk; ++l) s += L[i * kk + l] * L[j * kk + l] / D[l];
                T[i * kk + j] = s;
            }
        std::vector<double> C(kk * kk, 0.0);           // T = C C^T (Cholesky)
        for (std::size_t j = 0; j < kk; ++j) {
            double s = T[j * kk + j];
            for (std::size_t l = 0; l < j; ++l) s -= C[j * kk + l] * C[j * kk + l];
            C[j * kk + j] = std::sqrt(s);
            for (std::size_t i = j + 1; i < kk; ++i) {
                double r = T[i * kk + j];
                for (std::size_t l = 0; l < j; ++l) r -= C[i * kk + l] * C[j * kk + l];
                C[i * kk + j] = r / C[j * kk + j];
            }
        }
        std::vector<double> Ti(kk * kk, 0.0);          // T^-1, column by column
        for (std::size_t c = 0; c < kk; ++c) {
            Vec z(kk, 0.0);
            for (std::size_t i = 0; i < kk; ++i) {
                double s = i == c ? 1.0 : 0.0;
                for (std::size_t l = 0; l < i; ++l) s -= C[i * kk + l] * z[l];
                z[i] = s / C[i * kk + i];
            }
            for (std::size_t i = kk; i-- > 0;) {
                double s = z[i];
                for (std::size_t l = i + 1; l < kk; ++l) s -= C[l * kk + i] * z[l];
                z[i] = s / C[i * kk + i];
            }
            for (std::size_t i = 0; i < kk; ++i) Ti[i * kk + c] = z[i];
        }
        std::vector<double> E(kk * kk, 0.0);           // E = T^-1 L D^-1
        for (std::size_t i = 0; i < kk; ++i)
            for (std::size_t j = 0; j < kk; ++j) {
                double s = 0.0;
                for (std::size_t l = 0; l < kk; ++l) s += Ti[i * kk + l] * L[l * kk + j];
                E[i * kk + j] = s / D[j];
            }
        M.assign(q * q, 0.0);
        for (std::size_t i = 0; i < kk; ++i)
            for (std::size_t j = 0; j < kk; ++j) {
                double s = i == j ? -1.0 / D[i] : 0.0;  // -D^-1 + D^-1 L^T E
                for (std::size_t l = 0; l < kk; ++l) s += L[l * kk + i] * E[l * kk + j] / D[i];
                M[i * q + j] = s;
                M[i * q + kk + j] = E[j * kk + i];     // D^-1 L^T T^-1 = E^T
                M[(kk + i) * q + j] = E[i * kk + j];
                M[(kk + i) * q + kk + j] = Ti[i * kk + j];
            }
    }
};

// Algorithm CP (BLNZ §4): x^c, and c = W^T (x^c - x) (4.13)
inline Vec cauchy(const Vec& x, const Vec& g, const Vec& lo, const Vec& hi, const Memory& B, Vec& c)
{
    const std::size_t n = x.size();
    const double theta = B.theta;
    Vec t(n), d(n), xc = x;
    for (std::size_t i = 0; i < n; ++i) {
        if (g[i] < 0.0 && hi[i] < kInf) t[i] = (x[i] - hi[i]) / g[i];
        else if (g[i] > 0.0 && lo[i] > -kInf) t[i] = (x[i] - lo[i]) / g[i];
        else t[i] = kInf;
        d[i] = t[i] == 0.0 ? 0.0 : -g[i];
    }
    std::vector<std::size_t> order;                   // F, by increasing t_i (LB-5)
    for (std::size_t i = 0; i < n; ++i)
        if (t[i] > 0.0 && t[i] < kInf) order.push_back(i);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return t[a] < t[b]; });
    Vec p = B.wt(d);
    c.assign(p.size(), 0.0);
    double fp = 0.0;
    for (double v : d) fp -= v * v;                   // f' = -d^T d
    double fpp = -theta * fp;
    if (!p.empty()) {
        const Vec Mp = B.mul_m(p);
        for (std::size_t i = 0; i < p.size(); ++i) fpp -= p[i] * Mp[i];
    }
    double dtmin = fpp > 0.0 ? -fp / fpp : kInf;
    double told = 0.0;
    std::size_t next = 0;
    double tt = next < order.size() ? t[order[next]] : kInf;
    double dt = tt - told;
    while (dtmin >= dt && next < order.size()) {
        const std::size_t b = order[next++];
        xc[b] = d[b] > 0.0 ? hi[b] : lo[b];
        const double zb = xc[b] - x[b];
        for (std::size_t i = 0; i < c.size(); ++i) c[i] += dt * p[i];
        const double gb = g[b];
        const Vec wb = B.w_row(b);
        double wMc = 0.0, wMp = 0.0, wMw = 0.0;
        if (!wb.empty()) {
            const Vec Mc = B.mul_m(c), Mp = B.mul_m(p), Mw = B.mul_m(wb);
            for (std::size_t i = 0; i < wb.size(); ++i) {
                wMc += wb[i] * Mc[i];
                wMp += wb[i] * Mp[i];
                wMw += wb[i] * Mw[i];
            }
        }
        fp = fp + dt * fpp + gb * gb + theta * gb * zb - gb * wMc;
        fpp = fpp - theta * gb * gb - 2.0 * gb * wMp - gb * gb * wMw;
        for (std::size_t i = 0; i < p.size(); ++i) p[i] += gb * wb[i];
        d[b] = 0.0;
        dtmin = fpp > 0.0 ? -fp / fpp : kInf;
        told = tt;
        tt = next < order.size() ? t[order[next]] : kInf;
        dt = tt - told;
    }
    dtmin = std::max(dtmin, 0.0);
    if (!std::isfinite(dtmin)) dtmin = 0.0;           // every variable at a bound: d = 0
    told += dtmin;
    for (std::size_t i = 0; i < n; ++i)
        if (d[i] != 0.0) xc[i] = x[i] + told * d[i];
    for (std::size_t i = 0; i < c.size(); ++i) c[i] += dtmin * p[i];
    return xc;
}

// The direct primal method (BLNZ §5.1) and MN's projection: xbar
inline Vec subspace(const Vec& x, const Vec& g, const Vec& lo, const Vec& hi, const Memory& B, const Vec& xc,
                    const Vec& c)
{
    const std::size_t n = x.size();
    const double theta = B.theta;
    std::vector<std::size_t> Z;                       // LB-2
    for (std::size_t i = 0; i < n; ++i)
        if (xc[i] > lo[i] && xc[i] < hi[i]) Z.push_back(i);
    if (Z.empty()) return xc;
    const std::size_t t = Z.size(), q = 2 * B.k();
    Vec Mc = q ? B.mul_m(c) : Vec();
    Vec r(t);                                         // (5.4)
    for (std::size_t a = 0; a < t; ++a) {
        const std::size_t i = Z[a];
        double s = g[i] + theta * (xc[i] - x[i]);
        if (q) {
            const Vec wi = B.w_row(i);
            for (std::size_t j = 0; j < q; ++j) s -= wi[j] * Mc[j];
        }
        r[a] = s;
    }
    Vec du(t);
    for (std::size_t a = 0; a < t; ++a) du[a] = -r[a] / theta;
    if (q) {                                          // (5.10): - (1/theta^2) Z^T W N^-1 M W^T Z r
        std::vector<Vec> U(t);                        // the rows of Z^T W
        for (std::size_t a = 0; a < t; ++a) U[a] = B.w_row(Z[a]);
        Vec v(q, 0.0);
        for (std::size_t a = 0; a < t; ++a)
            for (std::size_t j = 0; j < q; ++j) v[j] += U[a][j] * r[a];
        v = B.mul_m(v);
        std::vector<double> UtU(q * q, 0.0), N(q * q, 0.0);
        for (std::size_t a = 0; a < t; ++a)
            for (std::size_t i = 0; i < q; ++i)
                for (std::size_t j = 0; j < q; ++j) UtU[i * q + j] += U[a][i] * U[a][j] / theta;
        for (std::size_t i = 0; i < q; ++i)
            for (std::size_t j = 0; j < q; ++j) {
                double s = 0.0;
                for (std::size_t l = 0; l < q; ++l) s += B.M[i * q + l] * UtU[l * q + j];
                N[i * q + j] = (i == j ? 1.0 : 0.0) - s;
            }
        v = solve(N, v);
        for (std::size_t a = 0; a < t; ++a) {
            double s = 0.0;
            for (std::size_t j = 0; j < q; ++j) s += U[a][j] * v[j];
            du[a] -= s / (theta * theta);
        }
    }
    Vec xbar = xc;                                    // MN: the projection of x^c + Z d^u
    for (std::size_t a = 0; a < t; ++a) {
        const std::size_t i = Z[a];
        xbar[i] = std::min(std::max(xc[i] + du[a], lo[i]), hi[i]);
    }
    double gd = 0.0;
    for (std::size_t i = 0; i < n; ++i) gd += g[i] * (xbar[i] - x[i]);
    if (gd < 0.0) return xbar;                        // LB-3
    double a_star = 1.0;                              // (5.8)
    for (std::size_t a = 0; a < t; ++a) {
        const std::size_t i = Z[a];
        if (du[a] > 0.0 && hi[i] < kInf) a_star = std::min(a_star, (hi[i] - xc[i]) / du[a]);
        else if (du[a] < 0.0 && lo[i] > -kInf) a_star = std::min(a_star, (lo[i] - xc[i]) / du[a]);
    }
    xbar = xc;
    for (std::size_t a = 0; a < t; ++a) xbar[Z[a]] = xc[Z[a]] + a_star * du[a];   // (5.9)
    return xbar;
}

}   // namespace detail

inline Result solve(const Problem& p, Vec x, const Options& o)
{
    const std::size_t n = p.n;
    Vec lo = p.lo.empty() ? Vec(n, -kInf) : p.lo, hi = p.hi.empty() ? Vec(n, kInf) : p.hi;
    if (lo.size() != n || hi.size() != n) throw std::invalid_argument("lbfgsb: lo and hi need n entries");
    for (std::size_t i = 0; i < n; ++i) {
        if (!(lo[i] <= hi[i])) throw std::invalid_argument("lbfgsb: lo_i <= hi_i is needed");
        x[i] = std::min(std::max(x[i], lo[i]), hi[i]);   // x_0 projected onto the box
    }
    Result res;
    auto eval = [&](const Vec& v, Vec& g) {           // f and g, counted
        if (p.fg) {
            ++res.nfev;
            ++res.ngev;
            return p.fg(v, g);
        }
        const double f0 = p.f(v);
        ++res.nfev;
        g.assign(n, 0.0);
        Vec vh = v;
        const double e = std::sqrt(kEps);
        for (std::size_t j = 0; j < n; ++j) {          // LB-5
            double h = e * std::fabs(v[j]);
            if (h == 0.0) h = e;
            if (v[j] + h > hi[j]) h = -h;
            vh[j] = v[j] + h;
            g[j] = (p.f(vh) - f0) / h;
            vh[j] = v[j];
            ++res.nfev;
        }
        return f0;
    };
    Vec g(n);
    double f = eval(x, g);
    detail::Memory B;
    for (;;) {
        double pg = 0.0;                              // BLNZ (6.1)
        for (std::size_t i = 0; i < n; ++i)
            pg = std::max(pg, std::fabs(std::min(std::max(x[i] - g[i], lo[i]), hi[i]) - x[i]));
        if (pg < o.pgtol) { res.status = Status::pgtol; break; }
        if (res.iterations >= o.max_iter) { res.status = Status::max_iter; break; }
        if (res.nfev >= o.max_nfev) { res.status = Status::max_nfev; break; }
        Vec c;
        Vec xc = detail::cauchy(x, g, lo, hi, B, c);
        Vec xbar = detail::subspace(x, g, lo, hi, B, xc, c);
        Vec d(n);
        double gd = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            d[i] = xbar[i] - x[i];
            gd += g[i] * d[i];
        }
        if (!(gd < 0.0)) {                            // LB-4
            if (B.k() == 0) { res.status = Status::line_search; break; }
            B = detail::Memory();
            continue;
        }
        double amax = kInf;                           // the step to the nearest bound along d
        for (std::size_t i = 0; i < n; ++i) {
            if (d[i] > 0.0 && hi[i] < kInf) amax = std::min(amax, (hi[i] - x[i]) / d[i]);
            else if (d[i] < 0.0 && lo[i] > -kInf) amax = std::min(amax, (lo[i] - x[i]) / d[i]);
        }
        Vec xa(n), ga_(n), xt = x, gt = g;            // the trial; the best one (LB-6)
        double ft = f;
        auto phi = [&](double a, double& fa, double& ga) {
            for (std::size_t i = 0; i < n; ++i) xa[i] = std::min(std::max(x[i] + a * d[i], lo[i]), hi[i]);
            fa = eval(xa, ga_);
            ga = 0.0;
            for (std::size_t i = 0; i < n; ++i) ga += ga_[i] * d[i];
            if (fa < ft) {
                ft = fa;
                xt = xa;
                gt = ga_;
            }
        };
        line_search::Options lo_opts;
        lo_opts.mu = o.alpha;
        lo_opts.eta = o.beta;
        lo_opts.a_max = amax;
        const line_search::Result ls = line_search::search(phi, f, gd, std::min(1.0, amax), lo_opts);
        if (ls.info == line_search::Info::converged && ls.f <= ft) {   // the point the search accepted
            ft = ls.f;
            xt = xa;
            gt = ga_;
        }
        if (!(ft < f)) { res.status = Status::line_search; break; }
        Vec s(n), y(n);
        double sy = 0.0, yy = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            s[i] = xt[i] - x[i];
            y[i] = gt[i] - g[i];
            sy += s[i] * y[i];
            yy += y[i] * y[i];
        }
        const double fold = f;
        x = xt;
        g = gt;
        f = ft;
        ++res.iterations;
        if (sy > 2.2e-16 * yy) {                      // BLNZ (3.9), step 6
            B.S.push_back(s);
            B.Y.push_back(y);
            if (static_cast<int>(B.S.size()) > o.m) {
                B.S.erase(B.S.begin());
                B.Y.erase(B.Y.begin());
            }
            B.theta = yy / sy;                        // step 7
            B.form();
        }
        if (o.ftol > 0.0 && (fold - f) / std::max({std::fabs(fold), std::fabs(f), 1.0}) <= o.ftol) {
            res.status = Status::ftol;
            break;
        }
    }
    res.x = x;
    res.g = g;
    res.f = f;
    return res;
}

}   // namespace mootation::scalar::lbfgsb
