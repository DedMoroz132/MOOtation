#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The Robust Gauss-Newton algorithm (RGN) for least squares in a box (task
// 25, item 24.3.6): Y. Qin, D. Kavetski, G. Kuczera, "A robust Gauss-Newton
// algorithm for the optimization of hydrological models: from standard
// Gauss-Newton to robust Gauss-Newton", Water Resour. Res. 54, 2018 (source
// robust-gauss-newton-standard-to-robust_qin2018a; "Q18" below), Appendix A
// (the algorithm), Appendix B (the bounds), §4 (the three heuristics and the
// termination). Written from the paper; the authors' Fortran code was not
// read.
//
// Phi(theta) = 1/2 r^T r. An iteration (Appendix A, steps 1-12):
//   the Jacobian by central differences with the large sampling scale h
//     (LSS, §4.1): theta +- h_k e_k clipped to the box, the column divided by
//     the actual difference;
//   g = J^T r, H = J^T J; the active set (B1-B3) fixes the variables at a
//     bound whose downhill direction leaves the box;
//   the step solves H_free delta = -g_free by the symmetric eigen-
//     decomposition, inverse eigenvalues at or below beta_NSJ sqrt(eps) w_max
//     set to 0 (the null-space jump, NSJ, §4.3; beta_NSJ = 0.001), and is
//     projected onto the box (B4);
//   Armijo backtracking: sigma = 1, 0.6, 0.36, ... (N_ls + 1 = 5 trials),
//     accept at Phi(theta + sigma delta) <= Phi + c_ls sigma min(0, delta^T g);
//   the best-sampling point (BSP, §4.2): the best of the difference points
//     replaces the line search's point when it is better (and counts as a
//     success);
//   h <- min(10 h, h_max) after a success, max(h / 10, h_min) after a failure;
//     h_max = phi_hmax (hi - lo), phi_hmax = 0.5, h_0 = h_max, h_min = 1e-8.
// The run ends (§4.4, step 11) after 100 iterations, or when Phi has not
// decreased over 4 iterations in a row, or the relative change of Phi
// (|Phi_{i-1} - Phi_i| / max(Phi_i, Phi_scale)) or of theta (max_k
// |dtheta_k| / max(|theta_k|, theta_scale)) has stayed below 1e-5 over 5
// iterations in a row; theta_scale = 10, Phi_scale = 1.
//
// READINGS:
//   RGN-1. Step 1 evaluates r at theta_i again every iteration; here theta_i
//     and its residuals come from the line search or the difference points,
//     so no evaluation is repeated (the library's evaluator does not count a
//     point twice either, scalar/evaluator.hpp THE COUNTS).
//   RGN-2. Every line search trial is an evaluation (the pseudocode adds 1 to
//     N_Phi only for the accepted one). A zero step (no free variable, or
//     every direction truncated) makes no trial: theta itself passes Armijo's
//     test, as the pseudocode's first trial would.
//   RGN-3. The expected reduction of B2, delta^T g + 1/2 delta^T H delta, is
//     taken for the projected step (Appendix A writes it for delta_rdc before
//     the projection of step 7); B2 takes the step and the reductions of the
//     previous iteration.
//   RGN-4. "The SVD" of the symmetric positive semidefinite H_free is its
//     eigendecomposition (eigen.hpp): W the eigenvalues; those at or below the
//     threshold, the slightly negative ones of round-off included, are
//     dropped.
//   RGN-5. A budget (max_evals, the library's) may end an iteration midway:
//     the run then returns the best point it has evaluated (the current one or
//     a difference point).
//   RGN-6. A residual that is not finite gives Phi = +inf, worse than any
//     finite value; a Jacobian column with a nonfinite residual is set to 0.
//     The start must give finite residuals.
//   RGN-7. The box must be finite (h_max is a fraction of its width) and
//     contain the start.
//   RGN-8. theta_scale = 10 and Phi_scale = 1 are Q18's absolute scales: in a
//     problem whose parameters are of order 1, theta changes below 1e-4 count
//     as "no change" (§4.4, criterion 3).
//   RGN-9. Free / locked / releasable as B1-B3; B3's "comparable gradient"
//     tests use |g_k| max(|theta_k|, theta_scale), as written.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../fp_contract.hpp"
#include "eigen.hpp"
#include "lsq.hpp"

namespace mootation::rgn {

using Residuals = std::function<void(const std::vector<double>& x, std::vector<double>& r)>;

struct Options {
    double phi_hmax = 0.5;          // Q18 Appendix A: the maximum sampling fraction
    double alpha_lss = 10.0;
    double h_min = 1e-8;
    double beta_nsj = 1e-3;
    int n_no_decrease = 4;          // §4.4 criteria
    double tol_f = 1e-5;
    int n_small_df = 5;
    double tol_x = 1e-5;
    int n_small_dx = 5;
    int max_iter = 100;
    double x_scale = 10.0;          // RGN-8
    double f_scale = 1.0;
    double sigma_ls = 1.0;          // the line search
    double rho_ls = 0.6;
    double c_ls = 1e-4;
    int n_ls = 4;
    double tau_sg = 1e-10;          // B2
    double tau_ss = 1e-11;
    double tau_sf = 1e-10;
    double tau_gmin = 0.1;
    double tau_rfix = 0.1;          // B3
    double tau_rforce = 0.1;
    long long max_evals = std::numeric_limits<long long>::max();
};

enum class Stop { max_iter, no_decrease, small_df, small_dx, budget };

struct Result {
    std::vector<double> x, r;
    double f = 0.0;                 // 1/2 r^T r
    long long evaluations = 0;      // residual vectors computed
    int iterations = 0;
    Stop stop = Stop::max_iter;
};

namespace detail {

enum Flag { Free, Lower, Upper, FreeLower, FreeUpper };   // B^F, B^L, B^H, B^FL, B^FH

inline double phi(const std::vector<double>& r)
{
    double s = 0.0;
    for (double v : r) {
        if (!std::isfinite(v)) return std::numeric_limits<double>::infinity();   // RGN-6
        s += v * v;
    }
    return 0.5 * s;
}

inline double scaled(double v, double x, double scale) { return std::fabs(v) * std::max(std::fabs(x), scale); }

}   // namespace detail

inline Result minimize(const Residuals& residuals, const std::vector<double>& x0, const std::vector<double>& lo,
                       const std::vector<double>& hi, const Options& o = {})
{
    using detail::Flag;
    const std::size_t n = x0.size();
    if (n == 0 || lo.size() != n || hi.size() != n) throw std::invalid_argument("rgn: x0, lo and hi of one nonzero length");
    for (std::size_t k = 0; k < n; ++k)
        if (!std::isfinite(lo[k]) || !std::isfinite(hi[k]) || !(lo[k] <= x0[k] && x0[k] <= hi[k]))
            throw std::invalid_argument("rgn: a finite box that holds x0 (RGN-7)");
    const double eps = std::numeric_limits<double>::epsilon();

    Result res;
    auto eval = [&](const std::vector<double>& x, std::vector<double>& r) {
        residuals(x, r);
        ++res.evaluations;
        return detail::phi(r);
    };

    std::vector<double> x = x0, r;
    double f = eval(x, r);
    if (!std::isfinite(f)) throw std::invalid_argument("rgn: the residuals at x0 are not finite (RGN-6)");
    const std::size_t m = r.size();

    std::vector<double> h(n), hmax(n);
    for (std::size_t k = 0; k < n; ++k) h[k] = hmax[k] = o.phi_hmax * (hi[k] - lo[k]);

    // the previous iteration's step and reductions (B2, RGN-3)
    std::vector<double> delta_prev(n, 0.0);
    double dphi_exp = 0.0, dphi_act = 0.0;
    int no_decrease = 0, small_df = 0, small_dx = 0;
    std::vector<double> best_s_x, best_s_r, rp, rm, xt, rt;

    auto finish = [&](Stop why) {
        res.x = x;
        res.r = r;
        res.f = f;
        res.stop = why;
        return res;
    };

    for (;;) {
        // 2) the Jacobian by central differences; the best sampled point
        lsq::Matrix J(m, n);
        double f_best_s = f;
        best_s_x = x;
        best_s_r = r;
        for (std::size_t k = 0; k < n; ++k) {
            if (res.evaluations + 2 > o.max_evals) {                      // RGN-5
                if (f_best_s < f) {
                    x = best_s_x;
                    r = best_s_r;
                    f = f_best_s;
                }
                return finish(Stop::budget);
            }
            std::vector<double> xm = x, xp = x;
            xm[k] = std::max(x[k] - h[k], lo[k]);
            xp[k] = std::min(x[k] + h[k], hi[k]);
            const double fm = eval(xm, rm), fp = eval(xp, rp);
            const double d = xp[k] - xm[k];
            const bool ok = d > 0.0 && std::isfinite(fm) && std::isfinite(fp);
            for (std::size_t t = 0; t < m; ++t) J(t, k) = ok ? (rp[t] - rm[t]) / d : 0.0;   // RGN-6
            if (fp < f_best_s) {
                f_best_s = fp;
                best_s_x = xp;
                best_s_r = rp;
            }
            if (fm < f_best_s) {
                f_best_s = fm;
                best_s_x = xm;
                best_s_r = rm;
            }
        }

        // 3) g = J^T r, H = J^T J
        std::vector<double> g(n, 0.0);
        lsq::Matrix H(n, n);
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t t = 0; t < m; ++t) g[j] += J(t, j) * r[t];
            for (std::size_t k = 0; k <= j; ++k) {
                double s = 0.0;
                for (std::size_t t = 0; t < m; ++t) s += J(t, j) * J(t, k);
                H(j, k) = H(k, j) = s;
            }
        }

        // 4) the active set: B1, B2 (from the second iteration), B3
        std::vector<int> a(n, Flag::Free);
        for (std::size_t k = 0; k < n; ++k) {
            if (x[k] <= lo[k] + 10.0 * eps * std::max(std::fabs(lo[k]), o.x_scale))
                a[k] = g[k] < 0.0 ? Flag::FreeLower : Flag::Lower;
            else if (x[k] >= hi[k] - 10.0 * eps * std::max(std::fabs(hi[k]), o.x_scale))
                a[k] = g[k] > 0.0 ? Flag::FreeUpper : Flag::Upper;
        }
        bool converged_free = false;
        if (res.iterations > 0) {
            double gbar = 0.0, dbar = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                if (a[k] == Flag::Free || a[k] == Flag::FreeLower || a[k] == Flag::FreeUpper)
                    gbar = std::max(gbar, detail::scaled(g[k], x[k], o.x_scale));
                dbar = std::max(dbar, std::fabs(delta_prev[k]) / std::max(std::fabs(x[k]), o.x_scale));
            }
            const double fs = std::max(std::fabs(f), o.f_scale);
            gbar /= fs;
            converged_free = gbar <= o.tau_sg || (dbar <= o.tau_ss && gbar <= o.tau_gmin) ||
                             (std::fabs(dphi_exp) / fs <= o.tau_sf && std::fabs(dphi_act) / fs <= o.tau_sf &&
                              gbar <= o.tau_gmin);
        }
        {
            std::size_t n_free = 0, n_rls = 0;
            double g_max_free = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                if (a[k] == Flag::Free) {
                    ++n_free;
                    g_max_free = std::max(g_max_free, detail::scaled(g[k], x[k], o.x_scale));
                }
                if (a[k] == Flag::FreeLower || a[k] == Flag::FreeUpper) ++n_rls;
            }
            const bool release = n_free == 0 || converged_free;
            if (n_rls > 0) {
                if (n_free > 0)
                    for (std::size_t k = 0; k < n; ++k)
                        if ((a[k] == Flag::FreeLower || a[k] == Flag::FreeUpper) &&
                            detail::scaled(g[k], x[k], o.x_scale) >= o.tau_rfix * g_max_free)
                            a[k] = Flag::Free;
                if (release) {
                    double g_max_rls = -1.0;
                    std::size_t l = n;
                    for (std::size_t k = 0; k < n; ++k)
                        if ((a[k] == Flag::FreeLower || a[k] == Flag::FreeUpper) &&
                            detail::scaled(g[k], x[k], o.x_scale) > g_max_rls) {
                            g_max_rls = detail::scaled(g[k], x[k], o.x_scale);
                            l = k;
                        }
                    if (l < n) {
                        a[l] = Flag::Free;
                        for (std::size_t k = 0; k < n; ++k)
                            if ((a[k] == Flag::FreeLower || a[k] == Flag::FreeUpper) &&
                                detail::scaled(g[k], x[k], o.x_scale) >= o.tau_rforce * g_max_rls)
                                a[k] = Flag::Free;
                    }
                }
            }
        }

        // 5)-6) the reduced normal equations by the eigendecomposition (RGN-4)
        std::vector<std::size_t> idx;
        for (std::size_t k = 0; k < n; ++k)
            if (a[k] == Flag::Free) idx.push_back(k);
        std::vector<double> delta(n, 0.0);
        if (!idx.empty()) {
            const std::size_t nf = idx.size();
            lsq::Matrix Hr(nf, nf), V;
            for (std::size_t i = 0; i < nf; ++i)
                for (std::size_t j = 0; j < nf; ++j) Hr(i, j) = H(idx[i], idx[j]);
            std::vector<double> w;
            eigen::symmetric(Hr, w, V);
            double wmax = 0.0;
            for (double v : w) wmax = std::max(wmax, std::fabs(v));
            const double cut = o.beta_nsj * std::sqrt(eps) * wmax;
            for (std::size_t j = 0; j < nf; ++j) {
                if (!(w[j] > cut)) continue;
                double ug = 0.0;
                for (std::size_t i = 0; i < nf; ++i) ug += V(i, j) * g[idx[i]];
                for (std::size_t i = 0; i < nf; ++i) delta[idx[i]] -= ug / w[j] * V(i, j);
            }
        }

        // 7) the projection onto the box (B4)
        for (std::size_t k = 0; k < n; ++k) {
            const double xf = x[k] + delta[k];
            delta[k] = (xf < lo[k] ? lo[k] : xf > hi[k] ? hi[k] : xf) - x[k];
        }

        // 8) Armijo backtracking (RGN-2)
        double dg = 0.0, dHd = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            dg += delta[j] * g[j];
            for (std::size_t k = 0; k < n; ++k) dHd += delta[j] * H(j, k) * delta[k];
        }
        dphi_exp = dg + 0.5 * dHd;                                           // RGN-3
        bool success = false;
        std::vector<double> x_ls = x, r_ls = r;
        double f_ls = f;
        bool zero = true;
        for (double v : delta) zero = zero && v == 0.0;
        if (zero) {
            success = true;
        } else {
            double sigma = o.sigma_ls;
            for (int t = 0; t <= o.n_ls; ++t) {
                if (res.evaluations + 1 > o.max_evals) break;                 // RGN-5
                xt = x;
                for (std::size_t k = 0; k < n; ++k) xt[k] = x[k] + sigma * delta[k];
                const double ft = eval(xt, rt);
                if (ft <= f + o.c_ls * sigma * std::min(0.0, dg)) {
                    x_ls = xt;
                    r_ls = rt;
                    f_ls = ft;
                    success = true;
                    break;
                }
                sigma *= o.rho_ls;
            }
        }
        dphi_act = f_ls - f;

        // 9) the best-sampling point
        const std::vector<double> x_prev = x;
        const double f_prev = f;
        ++res.iterations;
        if (f_best_s < f_ls) {
            x = best_s_x;
            r = best_s_r;
            f = f_best_s;
            success = true;
        } else {
            x = x_ls;
            r = r_ls;
            f = f_ls;
        }
        for (std::size_t k = 0; k < n; ++k) delta_prev[k] = x[k] - x_prev[k];

        // 10) the sampling scale
        for (std::size_t k = 0; k < n; ++k)
            h[k] = success ? std::min(h[k] * o.alpha_lss, hmax[k]) : std::max(h[k] / o.alpha_lss, o.h_min);

        // 11) termination
        no_decrease = f < f_prev ? 0 : no_decrease + 1;
        small_df = std::fabs(f_prev - f) / std::max(f, o.f_scale) < o.tol_f ? small_df + 1 : 0;
        double dx = 0.0;
        for (std::size_t k = 0; k < n; ++k) dx = std::max(dx, std::fabs(x[k] - x_prev[k]) / std::max(std::fabs(x[k]), o.x_scale));
        small_dx = dx < o.tol_x ? small_dx + 1 : 0;
        if (res.iterations >= o.max_iter) return finish(Stop::max_iter);
        if (no_decrease >= o.n_no_decrease) return finish(Stop::no_decrease);
        if (small_df >= o.n_small_df) return finish(Stop::small_df);
        if (small_dx >= o.n_small_dx) return finish(Stop::small_dx);
        if (res.evaluations >= o.max_evals) return finish(Stop::budget);
    }
}

}   // namespace mootation::rgn
