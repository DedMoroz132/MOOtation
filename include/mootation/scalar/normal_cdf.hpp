#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The standard normal distribution function, its inverse and the quantile of
// chi-squared with one degree of freedom, from the library's own exp and log
// (detmath.hpp, dm::), the same bits on every platform (the C library's erfc
// is not allowed in include/). For the margin of CMA-ES with margin (task 25,
// item 24.3.4; Hamano et al. 2022 §4.3-4.4: CI = sqrt(chi2_ppf(1 - 2 alpha))
// sigma <A C A>_j^(1/2), and the marginal probabilities p_low, p_up).
//
//   normal_cdf(x)        Phi(x) = 1/2 erfc(-x / sqrt 2): for |x| <= 2 by the
//                        power series of erf (Abramowitz & Stegun 7.1.5),
//                        beyond by the continued fraction of erfc (A&S
//                        7.1.14), evaluated by the modified Lentz method;
//                        each until its terms change the sum by less than
//                        1e-17 of it
//   normal_quantile(p)   Phi^-1(p), 0 < p < 1: bisection on [-40, 40] to a
//                        bracket of 1e-3, then Newton's steps x <- x - (Phi(x)
//                        - p) / phi(x) until the step is below 1e-15 (1 + |x|)
//   chi2_1_quantile(q)   the q-quantile of chi-squared with one degree of
//                        freedom: Phi^-1((1 + q) / 2)^2
//
// READINGS:
//   NCDF-1. The series and the continued fraction meet at |x| = 2 (z =
//     sqrt 2): there both converge in about 40 terms; the lower tail keeps its
//     relative accuracy (erfc is never 1 - erf there).
//   NCDF-2. The inverse takes the tail it is in (p > 1/2: -Phi^-1(1 - p)), so
//     that p near 1 is not lost to 1 - p's rounding more than the input
//     already is.
// ============================================================================

#include <cmath>
#include <stdexcept>

#include "../detmath.hpp"
#include "../fp_contract.hpp"

namespace mootation::ncdf {

inline constexpr double kSqrt2 = 1.4142135623730950488;
inline constexpr double kSqrtPi = 1.7724538509055160273;
inline constexpr double kSqrt2Pi = 2.5066282746310005024;

// erfc(z) for z >= 0
inline double erfc_nonneg(double z)
{
    if (z <= kSqrt2) {                        // 1 - erf(z), the series (A&S 7.1.5)
        double term = z, sum = z;
        for (int n = 1; n < 200; ++n) {
            term *= -z * z / n;
            const double add = term / (2 * n + 1);
            sum += add;
            if (std::fabs(add) < 1e-17 * std::fabs(sum)) break;
        }
        return 1.0 - 2.0 / kSqrtPi * sum;
    }
    // erfc(z) = exp(-z^2)/sqrt(pi) * 1/(z + (1/2)/(z + 1/(z + (3/2)/(z + ...)))) (A&S 7.1.14):
    // the fraction b0 + a1/(b1 + a2/(b2 + ...)), b_k = z, a_k = k/2, by Lentz
    const double tiny = 1e-300;
    double f = z, C = z, D = 0.0;
    for (int k = 1; k < 500; ++k) {
        const double a = 0.5 * k;
        D = z + a * D;
        if (D == 0.0) D = tiny;
        C = z + a / C;
        if (C == 0.0) C = tiny;
        D = 1.0 / D;
        const double delta = C * D;
        f *= delta;
        if (std::fabs(delta - 1.0) < 1e-17) break;
    }
    return dm::exp(-z * z) / kSqrtPi / f;
}

inline double normal_cdf(double x)
{
    if (std::isnan(x)) return x;
    const double z = std::fabs(x) / kSqrt2;
    const double tail = 0.5 * erfc_nonneg(z);  // P(X > |x|)
    return x < 0.0 ? tail : 1.0 - tail;
}

inline double normal_pdf(double x) { return dm::exp(-0.5 * x * x) / kSqrt2Pi; }

inline double normal_quantile(double p)
{
    if (!(p > 0.0 && p < 1.0)) throw std::invalid_argument("normal_quantile: 0 < p < 1");
    if (p > 0.5) return -normal_quantile(1.0 - p);          // NCDF-2
    double lo = -40.0, hi = 0.0;
    while (hi - lo > 1e-3) {
        const double mid = 0.5 * (lo + hi);
        (normal_cdf(mid) < p ? lo : hi) = mid;
    }
    double x = 0.5 * (lo + hi);
    for (int it = 0; it < 50; ++it) {
        const double step = (normal_cdf(x) - p) / normal_pdf(x);
        x -= step;
        if (std::fabs(step) < 1e-15 * (1.0 + std::fabs(x))) break;
    }
    return x;
}

inline double chi2_1_quantile(double q)
{
    if (!(q >= 0.0 && q < 1.0)) throw std::invalid_argument("chi2_1_quantile: 0 <= q < 1");
    if (q == 0.0) return 0.0;
    const double x = normal_quantile(0.5 * (1.0 + q));
    return x * x;
}

}   // namespace mootation::ncdf
