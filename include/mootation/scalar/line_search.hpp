#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The line search of J.J. Moré, D.J. Thuente, "Line search algorithms with
// guaranteed sufficient decrease", ACM TOMS 20(3) 1994, pp. 286-307 (source
// line-search_more1994; "MT"): for phi(a) = f(x + a p), phi'(0) < 0, an a in
// [a_min, a_max] with
//   (1.1) phi(a) <= phi(0) + mu phi'(0) a,   (1.2) |phi'(a)| <= eta |phi'(0)|.
// The search algorithm of §2: nested intervals I_k by the updating algorithm
// (cases U1-U3) on psi(a) = phi(a) - phi(0) - mu phi'(0) a, until a trial
// has psi <= 0 and phi' >= 0; from then on phi itself (§3's modified updating,
// cases a-c; LS-3 for the trials). The trial
// values by §4's cases 1-4 (alpha_c the cubic's minimizer through (a_l, f_l,
// g_l) and (a_t, f_t, g_t), alpha_q the quadratic's through f_l, f_t, g_l,
// alpha_s the secant's through g_l, g_t; the cubic and quadratic minimizers as
// J. Nocedal, S.J. Wright, Numerical Optimization, 2nd ed. 2006, (3.59) and
// (3.57) give them). Safeguards (§2): while no interval is bracketed the trial
// lies in [a_t + 1.1 (a_t - a_l), a_t + 4 (a_t - a_l)] (MT's delta in [1.1,
// 4]), within [a_min, a_max]; once bracketed, a bisection when the interval
// has not shrunk by 0.66 in two trials; case 3 keeps within a_t + 0.66 (a_u -
// a_t). Termination (§2): (1.1) and (1.2); at a_max with psi <= 0, psi' < 0;
// at a_min with psi > 0 or psi' >= 0.
//
// Readings where §4's words and §5's tables disagree: the tables were made
// by MT's program, and these three readings reproduce all 24 runs of Tables
// I-VI (trials m and a_m; tests/test_line_search.cpp), where the words taken
// literally give 12 of 24:
//   LS-1. Unbracketed, case 4 takes the far end a_t + 4 (a_t - a_l) (MT give
//     no interpolant there without a_u; §5: "in a typical situation the
//     extrapolation process generates iterates by setting a_t+ = a_t +
//     delta (a_t - a_l) with delta = 4", the iterates 0.005, 0.021, 0.085,
//     0.341, 1.365 of Table I).
//   LS-6. Unbracketed, case 3 takes the farther of alpha_c and alpha_s from
//     a_t (the words: the closer, "to be cautious"), alpha_c being the far end
//     where the cubic has no minimizer beyond a_t (the words: alpha_s); then
//     (2.2)'s bound a_t + 4 (a_t - a_l). Bracketed, the words: the closer.
//   LS-3. Before the switch to phi, psi stands in for phi in §4's cases only
//     where the trial has a lower phi than a_l but not the sufficient
//     decrease (phi_t <= phi_l, psi_t > 0), phi elsewhere (the words: psi
//     until the switch). The switch at the first trial with psi <= 0 and
//     phi' >= 0, as §4 words it (§3 writes phi' > 0).
// OURS:
//   LS-2. Case 3, the cubic "tends to infinity in the direction of the step"
//     where its cubic coefficient has the sign of a_t - a_l, and its minimizer
//     exists (the discriminant of NW (3.59) is positive).
//   LS-4. Also stops where the bracket is narrower than xtol times its right
//     end (xtol = 1e-10 by default: rounding), and after max_nfev trials.
//   LS-5. A trial a bracket cannot hold (outside it after rounding) is
//     replaced by the bisection.
//   LS-7. A trial where phi or phi' is not finite (overflow) becomes the right
//     end, and the next trial halves the way back to a_l.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

#include "../fp_contract.hpp"

namespace mootation::scalar::line_search {

struct Options {
    double mu = 1e-3;                       // (1.1)
    double eta = 0.9;                       // (1.2)
    double a_min = 0.0, a_max = std::numeric_limits<double>::infinity();
    double xtol = 1e-10;                    // LS-4
    int max_nfev = 20;
};

enum class Info { converged, at_max, at_min, width, max_nfev };

struct Result {
    double a = 0.0, f = 0.0, g = 0.0;       // the last trial, phi and phi' there
    int nfev = 0;
    Info info = Info::max_nfev;
};

namespace detail {

// the minimizer of the cubic through (a, fa, ga), (b, fb, gb) (NW (3.59)); ok = false where it has none
inline double cubic(double a, double fa, double ga, double b, double fb, double gb, bool& ok)
{
    const double d1 = ga + gb - 3.0 * (fa - fb) / (a - b);
    const double disc = d1 * d1 - ga * gb;
    ok = disc >= 0.0;
    if (!ok) return b;
    const double d2 = (b > a ? 1.0 : -1.0) * std::sqrt(disc);
    const double den = gb - ga + 2.0 * d2;
    if (den == 0.0) {
        ok = false;
        return b;
    }
    return b - (b - a) * (gb + d2 - d1) / den;
}

// the minimizer of the quadratic through fa, ga at a and fb at b (NW (3.57), shifted)
inline double quadratic(double a, double fa, double ga, double b, double fb)
{
    const double h = b - a;
    return a - ga * h * h / (2.0 * (fb - fa - ga * h));
}

// the secant step: the minimizer of the quadratic with derivatives ga at a, gb at b
inline double secant(double a, double ga, double b, double gb)
{
    return b - gb * (b - a) / (gb - ga);
}

}   // namespace detail

// phi(a) -> f, g = phi'(a); phi0, g0 at 0 (g0 < 0); a0 the first trial
inline Result search(const std::function<void(double a, double& f, double& g)>& phi, double phi0, double g0,
                     double a0, const Options& o)
{
    Result res;
    const double gtest = o.mu * g0;
    bool use_phi = false, bracketed = false;
    double al = 0.0, fl = phi0, gl = g0;              // phi's values at the ends
    double au = 0.0, fu = phi0, gu = g0;
    double width = o.a_max - o.a_min, width1 = 2.0 * width;
    double a = std::min(std::max(a0, o.a_min), o.a_max);
    for (;;) {
        double f, g;
        phi(a, f, g);
        ++res.nfev;
        res.a = a;
        res.f = f;
        res.g = g;
        const double psi = f - phi0 - gtest * a, dpsi = g - gtest;
        if (psi <= 0.0 && std::fabs(g) <= o.eta * -g0) { res.info = Info::converged; return res; }
        if (a == o.a_max && psi <= 0.0 && dpsi < 0.0) { res.info = Info::at_max; return res; }
        if (a == o.a_min && (psi > 0.0 || dpsi >= 0.0)) { res.info = Info::at_min; return res; }
        if (bracketed && std::fabs(au - al) <= o.xtol * std::max(al, au)) { res.info = Info::width; return res; }
        if (res.nfev >= o.max_nfev) { res.info = Info::max_nfev; return res; }
        if (!std::isfinite(f) || !std::isfinite(g)) {    // LS-7: back toward a_l
            bracketed = true;
            au = a;
            fu = f;
            gu = g;
            a = al + 0.5 * (a - al);
            continue;
        }
        if (!use_phi && psi <= 0.0 && g >= 0.0) use_phi = true;   // LS-3
        // the three points in the function the search uses (LS-3)
        const bool modified = !use_phi && f <= fl && psi > 0.0;
        auto val = [&](double x, double fx) { return modified ? fx - phi0 - gtest * x : fx; };
        auto der = [&](double gx) { return modified ? gx - gtest : gx; };
        const double Fl = val(al, fl), Gl = der(gl), Ft = val(a, f), Gt = der(g), Fu = val(au, fu), Gu = der(gu);
        // §4: the next trial
        double next;
        bool ok = false;
        if (Ft > Fl) {                                    // case 1
            const double c = detail::cubic(al, Fl, Gl, a, Ft, Gt, ok);
            const double q = detail::quadratic(al, Fl, Gl, a, Ft);
            next = ok && std::fabs(c - al) < std::fabs(q - al) ? c : ok ? 0.5 * (q + c) : q;
            bracketed = true;
        } else if (Gt * Gl < 0.0) {                       // case 2
            const double c = detail::cubic(al, Fl, Gl, a, Ft, Gt, ok);
            const double s = detail::secant(al, Gl, a, Gt);
            next = ok && std::fabs(c - a) >= std::fabs(s - a) ? c : s;
            bracketed = true;
        } else if (std::fabs(Gt) <= std::fabs(Gl)) {      // case 3
            const double s = detail::secant(al, Gl, a, Gt);
            const double h = a - al;
            const double c3 = (Gl + Gt - 2.0 * (Ft - Fl) / h) / (h * h);   // the cubic coefficient (LS-2)
            const double c = detail::cubic(al, Fl, Gl, a, Ft, Gt, ok);
            const bool beyond = ok && c3 * h > 0.0 && (c - a) * h > 0.0;
            if (bracketed) {
                next = beyond && std::fabs(c - a) < std::fabs(s - a) ? c : s;
            } else {                                      // LS-6
                const double cc = beyond ? c : a + 4.0 * h;
                next = std::fabs(cc - a) > std::fabs(s - a) ? cc : s;
            }
            if (bracketed)
                next = a > al ? std::min(a + 0.66 * (au - a), next) : std::max(a + 0.66 * (au - a), next);
        } else {                                          // case 4
            if (bracketed) next = detail::cubic(au, Fu, Gu, a, Ft, Gt, ok);
            else next = a + 4.0 * (a - al);               // LS-1
            if (bracketed && !ok) next = 0.5 * (al + au);
        }
        // the updating algorithm (U1-U3; a-c once phi is used)
        const double al_old = al;
        if (Ft > Fl) {
            au = a; fu = f; gu = g;
        } else {
            if (Gt * (al - a) < 0.0) { au = al; fu = fl; gu = gl; }
            al = a; fl = f; gl = g;
        }
        if (bracketed) {                                  // shrink, else bisect
            const double w = std::fabs(au - al);
            if (w >= 0.66 * width1) next = al + 0.5 * (au - al);
            width1 = width;
            width = w;
            const double lo = std::min(al, au), hi = std::max(al, au);
            if (!(next > lo && next < hi)) next = al + 0.5 * (au - al);   // LS-5
        } else {                                          // (2.2): delta in [1.1, 4]
            const double lo = a + 1.1 * (a - al_old), hi = a + 4.0 * (a - al_old);
            next = std::min(std::max(next, std::min(lo, hi)), std::max(lo, hi));
        }
        a = std::min(std::max(next, o.a_min), o.a_max);
    }
}

}   // namespace mootation::scalar::line_search
