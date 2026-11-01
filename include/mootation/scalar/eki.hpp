#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Ensemble Kalman inversion (task 19, §19.4.1): Algorithm 1 of M.A. Iglesias,
// K.J.H. Law, A.M. Stuart, "Ensemble Kalman methods for inverse problems",
// Inverse Problems 29 (2013) 045001, arXiv:1209.2736 (source eki-inverse-
// problems_iglesias2013_arxiv), on a least-squares problem given by its
// residuals r(u) = G(u) - y (the task-18 contract: residual blocks already
// weighted). An ensemble of J members u^(j); each iteration:
//   prediction (9)-(11): r^(j) = r(u^(j)), one evaluation a member; the
//     centred members u~^(j) = u^(j) - mean u, p~^(j) = r^(j) - mean r (the
//     sample covariances (11) with the divisor J);
//   analysis (12)-(15), in the form (25): u^(j) += C^up (C^pp + Gamma)^-1
//     (y^(j) - G(u^(j))), C^up = (1/J) sum u~ p~^T, C^pp = (1/J) sum p~
//     p~^T, y^(j) - G(u^(j)) = eta^(j) - r^(j), eta^(j) ~ N(0, Gamma)
//     independent over (j, n) (15) or 0 (perturb = false);
//   the estimate (16): the ensemble's mean.
// Gamma = gamma^2 I on the residuals. The members move by combinations of
// the centred members (their (28)), so the ensemble stays in the affine hull
// of the initial one (their Theorem 2.1 says the linear span; the weights of
// (28) sum to zero, which gives the affine hull, tighter).
// Stopping: the budget of evaluations (an iteration is begun only where it
// and the answer's evaluation fit), or with noise_norm > 0 the
// discrepancy principle of their §2.3, ||y - G(u_n)||_Gamma <= tau
// ||eta^dagger||_Gamma, the mean evaluated each iteration (one evaluation
// more).
//
// OURS:
//   EKI-1. (C^pp + Gamma)^-1 v: where J <= m by the Sherman-Morrison-Woodbury
//     identity, gamma^-2 (v - P~ (J gamma^2 I + P~^T P~)^-1 P~^T v), P~ the
//     m x J matrix of the p~^(j), a J x J Cholesky system; where J > m the
//     m x m system C^pp + Gamma itself. The same numbers to rounding.
//   EKI-2. (22) prints C^pp with p-hat (uncentred); we take the covariance
//     (11) prints, centred, which (37)-(38) use as well.
//   EKI-3. The initial ensemble (their §2.2 leaves it to the user as a draw
//     from a prior): J points of a design in the box (init = sobol: the
//     scrambled Sobol points of designs.hpp; lhs: its Latin hypercube; seed
//     the design's), or around x0: u = x0 + sigma0 s N(0, 1) per coordinate,
//     s the box's width where it is finite, max(1, |x0_i|) where not. The box
//     serves the initial ensemble alone: the members are not held in it (the
//     method has no bounds, and clipping would break the affine hull).
//   EKI-4. J = max(10, 2 n) by default; gamma = 1 (the residuals taken as
//     divided by the noise's sigma); perturb = true (their p. 7: without the
//     perturbation the results were less accurate); tau = 1.5 (their tau >
//     1).
//   EKI-5. The answer: the mean of the last ensemble, evaluated once; the best
//     member evaluated on the way is returned beside it.
//   EKI-6. The draws: one std::mt19937 seeded with seed; the initial
//     ensemble around x0 member after member, then eta per iteration member
//     after member, rnd::normal01.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "../designs.hpp"
#include "../fp_contract.hpp"
#include "../random.hpp"

namespace mootation::eki {

using Vec = std::vector<double>;

struct Options {
    int ensemble = 0;                       // J (0: EKI-4)
    enum class Init { sobol, lhs, around } init = Init::sobol;
    double sigma0 = 0.1;                    // init = around (EKI-3)
    double gamma = 1.0;                     // Gamma = gamma^2 I
    bool perturb = true;                    // (15)
    long long max_evals = 10000;
    double noise_norm = 0.0;                // ||eta^dagger||_Gamma: the discrepancy principle (0: off)
    double tau = 1.5;
    std::uint32_t seed = 0;
};

struct Result {
    Vec x;                                  // the last ensemble's mean (16)
    double f = 0.0;                         // 1/2 ||r(x)||^2
    Vec best_x;                             // the best member evaluated
    double best_f = std::numeric_limits<double>::infinity();
    std::vector<Vec> initial, ensemble;     // the first and the last ensemble
    long long evaluations = 0, iterations = 0;
    bool discrepancy = false;               // stopped by the discrepancy principle
};

using Residuals = std::function<void(const Vec& x, Vec& r)>;

namespace detail {

inline double half_squares(const Vec& r)
{
    double s = 0.0;
    for (double v : r) s += v * v;
    return 0.5 * s;
}

// S x = b, S symmetric positive definite (k x k, row-major), by Cholesky
inline Vec cholesky_solve(std::vector<double> S, Vec b)
{
    const std::size_t k = b.size();
    for (std::size_t j = 0; j < k; ++j) {
        double d = S[j * k + j];
        for (std::size_t l = 0; l < j; ++l) d -= S[j * k + l] * S[j * k + l];
        if (!(d > 0.0)) throw std::runtime_error("eki: the gain's system is not positive definite");
        const double L = std::sqrt(d);
        S[j * k + j] = L;
        for (std::size_t i = j + 1; i < k; ++i) {
            double s = S[i * k + j];
            for (std::size_t l = 0; l < j; ++l) s -= S[i * k + l] * S[j * k + l];
            S[i * k + j] = s / L;
        }
    }
    for (std::size_t i = 0; i < k; ++i) {           // L y = b
        double s = b[i];
        for (std::size_t l = 0; l < i; ++l) s -= S[i * k + l] * b[l];
        b[i] = s / S[i * k + i];
    }
    for (std::size_t ii = k; ii-- > 0;) {           // L^T x = y
        double s = b[ii];
        for (std::size_t l = ii + 1; l < k; ++l) s -= S[l * k + ii] * b[l];
        b[ii] = s / S[ii * k + ii];
    }
    return b;
}

}   // namespace detail

inline Result minimize(const Residuals& residuals, std::size_t m, const Vec& lo, const Vec& hi, const Vec& x0,
                       const Options& o)
{
    const std::size_t n = lo.size();
    if (hi.size() != n || (o.init == Options::Init::around && x0.size() != n))
        throw std::invalid_argument("eki: lo, hi (and x0) need n entries");
    if (o.init != Options::Init::around)
        for (std::size_t i = 0; i < n; ++i)
            if (!(std::isfinite(lo[i]) && std::isfinite(hi[i]) && lo[i] < hi[i]))
                throw std::invalid_argument("eki: a design in the box needs finite bounds; init = around takes x0");
    const std::size_t J = o.ensemble > 0 ? static_cast<std::size_t>(o.ensemble) : std::max<std::size_t>(10, 2 * n);
    if (J < 2) throw std::invalid_argument("eki: two members at least");
    std::mt19937 g(o.seed);
    Result res;
    std::vector<Vec> u(J, Vec(n));
    if (o.init == Options::Init::around) {           // EKI-3
        for (Vec& uj : u)
            for (std::size_t i = 0; i < n; ++i) {
                const double s = std::isfinite(lo[i]) && std::isfinite(hi[i]) ? hi[i] - lo[i]
                                                                                : std::max(1.0, std::fabs(x0[i]));
                uj[i] = x0[i] + o.sigma0 * s * rnd::normal01(g);
            }
    } else {
        const Vec pts = o.init == Options::Init::sobol ? designs::sobol_lms(J, static_cast<int>(n), o.seed)
                                                       : designs::lhs(J, static_cast<int>(n), o.seed);
        for (std::size_t j = 0; j < J; ++j)
            for (std::size_t i = 0; i < n; ++i) u[j][i] = lo[i] + (hi[i] - lo[i]) * pts[j * n + i];
    }
    res.initial = u;
    const double gamma2 = o.gamma * o.gamma, dJ = static_cast<double>(J);
    std::vector<Vec> r(J, Vec(m));
    auto evaluate = [&](const Vec& x, Vec& rx) {
        rx.assign(m, 0.0);
        residuals(x, rx);
        ++res.evaluations;
        const double f = detail::half_squares(rx);
        if (f < res.best_f) {
            res.best_f = f;
            res.best_x = x;
        }
        return f;
    };
    auto mean_of = [&]() {
        Vec mu(n, 0.0);
        for (const Vec& uj : u)
            for (std::size_t i = 0; i < n; ++i) mu[i] += uj[i];
        for (double& v : mu) v /= dJ;
        return mu;
    };
    const long long per = static_cast<long long>(J) + (o.noise_norm > 0.0 ? 1 : 0);   // and one for the answer
    while (res.evaluations + per + 1 <= o.max_evals) {
        for (std::size_t j = 0; j < J; ++j) evaluate(u[j], r[j]);     // prediction (9)
        Vec ubar = mean_of(), rbar(m, 0.0);
        for (const Vec& rj : r)
            for (std::size_t i = 0; i < m; ++i) rbar[i] += rj[i];
        for (double& v : rbar) v /= dJ;
        std::vector<Vec> ut(J, Vec(n)), pt(J, Vec(m));                // centred (11)
        for (std::size_t j = 0; j < J; ++j) {
            for (std::size_t i = 0; i < n; ++i) ut[j][i] = u[j][i] - ubar[i];
            for (std::size_t i = 0; i < m; ++i) pt[j][i] = r[j][i] - rbar[i];
        }
        const bool woodbury = J <= m;                                 // EKI-1
        std::vector<double> S;                                        // J gamma^2 I + P~^T P~, or C^pp + Gamma
        if (woodbury) {
            S.assign(J * J, 0.0);
            for (std::size_t a = 0; a < J; ++a)
                for (std::size_t b = 0; b <= a; ++b) {
                    double s = 0.0;
                    for (std::size_t i = 0; i < m; ++i) s += pt[a][i] * pt[b][i];
                    S[a * J + b] = S[b * J + a] = s + (a == b ? dJ * gamma2 : 0.0);
                }
        } else {
            S.assign(m * m, 0.0);
            for (std::size_t a = 0; a < m; ++a)
                for (std::size_t b = 0; b <= a; ++b) {
                    double s = 0.0;
                    for (std::size_t k = 0; k < J; ++k) s += pt[k][a] * pt[k][b];
                    S[a * m + b] = S[b * m + a] = s / dJ + (a == b ? gamma2 : 0.0);
                }
        }
        std::vector<Vec> step(J, Vec(n, 0.0));
        for (std::size_t j = 0; j < J; ++j) {                         // analysis (25)
            Vec v(m);
            for (std::size_t i = 0; i < m; ++i) {
                const double eta = o.perturb ? o.gamma * rnd::normal01(g) : 0.0;   // (15)
                v[i] = eta - r[j][i];
            }
            Vec d(m);                                                 // (C^pp + Gamma)^-1 v
            if (woodbury) {
                Vec ptv(J, 0.0);
                for (std::size_t k = 0; k < J; ++k)
                    for (std::size_t i = 0; i < m; ++i) ptv[k] += pt[k][i] * v[i];
                const Vec w = detail::cholesky_solve(S, ptv);
                for (std::size_t i = 0; i < m; ++i) {
                    double s = v[i];
                    for (std::size_t k = 0; k < J; ++k) s -= pt[k][i] * w[k];
                    d[i] = s / gamma2;
                }
            } else {
                d = detail::cholesky_solve(S, v);
            }
            for (std::size_t k = 0; k < J; ++k) {                     // (1/J) <p~^(k), d> u~^(k) (28)
                double a = 0.0;
                for (std::size_t i = 0; i < m; ++i) a += pt[k][i] * d[i];
                a /= dJ;
                for (std::size_t i = 0; i < n; ++i) step[j][i] += a * ut[k][i];
            }
        }
        for (std::size_t j = 0; j < J; ++j)
            for (std::size_t i = 0; i < n; ++i) u[j][i] += step[j][i];
        ++res.iterations;
        if (o.noise_norm > 0.0) {                                     // the discrepancy principle
            Vec rm;
            const double f = evaluate(mean_of(), rm);
            if (std::sqrt(2.0 * f) / o.gamma <= o.tau * o.noise_norm) {
                res.discrepancy = true;
                break;
            }
        }
    }
    res.x = mean_of();                                                // EKI-5
    Vec rx;
    res.f = evaluate(res.x, rx);
    res.ensemble = u;
    return res;
}

}   // namespace mootation::eki
