#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The Nelder-Mead simplex method in a box.
//
// The iteration is Algorithm NM of J.C. Lagarias, J.A. Reeds, M.H. Wright,
// P.E. Wright, "Convergence properties of the Nelder-Mead simplex method in
// low dimensions", SIAM J. Optim. 9(1):112-147, 1998, §2.1 (source
// nelder-mead-convergence_lagarias1998): order, reflect, expand (the better of
// the reflected and the expanded point), outside or inside contraction, shrink
// — with its tie-breaking rules: a new vertex takes the highest index its
// value allows, and in a shrink x_1 stays first when tied.
//
// The coefficients are the adaptive ones of F. Gao, L. Han, "Implementing the
// Nelder-Mead simplex algorithm with adaptive parameters", Comput. Optim.
// Appl. 51(1):259-277, 2012, (4.1), for n >= 2: rho = 1, chi = 1 + 2/n,
// gamma = 0.75 - 1/(2n), sigma = 1 - 1/n (source nelder-mead-adaptive_
// gao2012); for n = 1, where (4.1) would give sigma = 0, the standard ones of
// Lagarias et al. (2.2). The run ends by Gao & Han's test (T1): max |f_i -
// f_1| <= tol and max ||x_i - x_1||_inf <= tol, or when max_iter iterations
// are done.
//
// OURS:
//   NM-1. The initial simplex: x_0 and x_0 + tau e_i, tau = 0.05 in every
//     coordinate, x_0 - tau e_i where x_0 + tau e_i would leave the box (the
//     box is meant normalized to [0, 1]^n: Gao & Han's (3.1), fminsearch's,
//     takes 0.00025 for a zero coordinate, a rule for unscaled variables).
//   NM-2. Every point the method makes is clipped to the box before it is
//     evaluated and becomes a vertex (Lagarias et al. and Gao & Han treat
//     unconstrained problems).
//   NM-3. The initial vertices are ordered by value, ties in their order of
//     construction (x_0 first).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include "../fp_contract.hpp"

namespace mootation::nelder_mead {

struct Options {
    double tol = 1e-8;
    double tau = 0.05;              // NM-1
    long long max_iter = 1000000;
};

struct Result {
    std::vector<double> x;          // the best vertex
    double f = 0.0;
    long long iterations = 0;
    bool converged = false;
};

// f is called at every vertex and trial point; whatever it throws ends the run.
inline Result minimize(const std::function<double(const std::vector<double>&)>& f,
                       const std::vector<double>& x0, const std::vector<double>& lo,
                       const std::vector<double>& hi, const Options& o)
{
    const std::size_t n = x0.size();
    const double dn = static_cast<double>(n);
    const double rho = 1.0;
    const double chi = n >= 2 ? 1.0 + 2.0 / dn : 2.0;
    const double gamma = n >= 2 ? 0.75 - 1.0 / (2.0 * dn) : 0.5;
    const double sigma = n >= 2 ? 1.0 - 1.0 / dn : 0.5;

    auto clip = [&](std::vector<double>& x) {
        for (std::size_t i = 0; i < n; ++i) x[i] = std::min(hi[i], std::max(lo[i], x[i]));
    };
    struct Vertex {
        std::vector<double> x;
        double f;
    };
    std::vector<Vertex> s;                                  // ordered: s[0] best
    {
        std::vector<double> v = x0;
        clip(v);
        s.push_back({v, f(v)});
        for (std::size_t i = 0; i < n; ++i) {               // NM-1
            std::vector<double> w = s[0].x;
            w[i] = w[i] + o.tau <= hi[i] ? w[i] + o.tau : w[i] - o.tau;
            clip(w);
            s.push_back({w, f(w)});
        }
        std::stable_sort(s.begin(), s.end(),                // NM-3
                         [](const Vertex& a, const Vertex& b) { return a.f < b.f; });
    }

    Result r;
    // the accepted point replaces the worst vertex and goes after every
    // vertex whose value is not greater (the nonshrink ordering rule)
    auto accept = [&](std::vector<double> x, double fx) {
        s.pop_back();
        std::size_t j = 0;
        while (j < s.size() && !(fx < s[j].f)) ++j;
        s.insert(s.begin() + static_cast<std::ptrdiff_t>(j), Vertex{std::move(x), fx});
    };
    auto point = [&](const std::vector<double>& c, double t) {   // c + t (c - x_{n+1})
        std::vector<double> x(n);
        for (std::size_t i = 0; i < n; ++i) x[i] = c[i] + t * (c[i] - s[n].x[i]);
        clip(x);                                            // NM-2
        return x;
    };

    while (r.iterations < o.max_iter) {
        double df = 0.0, dx = 0.0;                          // (T1)
        for (std::size_t k = 1; k <= n; ++k) {
            df = std::max(df, std::fabs(s[k].f - s[0].f));
            for (std::size_t i = 0; i < n; ++i) dx = std::max(dx, std::fabs(s[k].x[i] - s[0].x[i]));
        }
        if (df <= o.tol && dx <= o.tol) { r.converged = true; break; }
        ++r.iterations;

        std::vector<double> c(n, 0.0);                      // the centroid of the n best
        for (std::size_t k = 0; k < n; ++k)
            for (std::size_t i = 0; i < n; ++i) c[i] += s[k].x[i];
        for (double& v : c) v /= dn;

        std::vector<double> xr = point(c, rho);             // 2. reflect
        const double fr = f(xr);
        if (s[0].f <= fr && fr < s[n - 1].f) { accept(xr, fr); continue; }
        if (fr < s[0].f) {                                  // 3. expand
            std::vector<double> xe = point(c, rho * chi);
            const double fe = f(xe);
            if (fe < fr) accept(xe, fe);
            else accept(xr, fr);
            continue;
        }
        if (fr < s[n].f) {                                  // 4a. outside contraction
            std::vector<double> xc = point(c, rho * gamma);
            const double fc = f(xc);
            if (fc <= fr) { accept(xc, fc); continue; }
        } else {                                            // 4b. inside contraction
            std::vector<double> xcc = point(c, -gamma);
            const double fcc = f(xcc);
            if (fcc < s[n].f) { accept(xcc, fcc); continue; }
        }
        for (std::size_t k = 1; k <= n; ++k) {              // 5. shrink
            for (std::size_t i = 0; i < n; ++i)
                s[k].x[i] = s[0].x[i] + sigma * (s[k].x[i] - s[0].x[i]);
            s[k].f = f(s[k].x);
        }
        std::stable_sort(s.begin(), s.end(),                // x_1 first when tied
                         [](const Vertex& a, const Vertex& b) { return a.f < b.f; });
    }
    r.x = s[0].x;
    r.f = s[0].f;
    return r;
}

}   // namespace mootation::nelder_mead
