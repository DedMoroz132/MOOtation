// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// include/mootation/scalar/normal_cdf.hpp (task 25, for 24.3.4): the standard
// normal distribution function and its inverse on known values
//   Phi(0) = 1/2, Phi(-1) = 0.15865525393145705, Phi(-3) =
//   0.0013498980316300946, Phi(-8) = 6.2209605742717841e-16 (relative 1e-13),
//   Phi(1.959963984540054) = 0.975;
//   Phi^-1(0.975) = 1.959963984540054, Phi^-1(1e-10) = -6.3613409024040557;
//   chi2_1 quantiles 0.95 -> 3.8414588206941236, 0.99 -> 6.6348966010212145;
//   Phi(Phi^-1(p)) = p for p from 1e-12 to 1 - 1e-12; the series and the
//   continued fraction agree where they meet (|x| = 2).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include <mootation/scalar/normal_cdf.hpp>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

namespace nc = mootation::ncdf;

bool rel(double a, double b, double tol) { return std::fabs(a - b) <= tol * std::fabs(b); }

}   // namespace

int main()
{
    check(nc::normal_cdf(0.0) == 0.5, "Phi(0) = 1/2");
    std::printf("      Phi(-1) = %.17g, Phi(-3) = %.17g, Phi(-8) = %.17g\n", nc::normal_cdf(-1.0), nc::normal_cdf(-3.0),
                nc::normal_cdf(-8.0));
    check(rel(nc::normal_cdf(-1.0), 0.15865525393145705, 1e-15) && rel(nc::normal_cdf(-3.0), 0.0013498980316300946, 1e-14) &&
              rel(nc::normal_cdf(-8.0), 6.2209605742717841e-16, 1e-13),
          "Phi(-1), Phi(-3), Phi(-8)");
    check(rel(nc::normal_cdf(1.959963984540054), 0.975, 1e-15), "Phi(1.959963984540054) = 0.975");
    std::printf("      Phi^-1(0.975) = %.17g, Phi^-1(1e-10) = %.17g\n", nc::normal_quantile(0.975), nc::normal_quantile(1e-10));
    check(rel(nc::normal_quantile(0.975), 1.959963984540054, 1e-14) && rel(nc::normal_quantile(1e-10), -6.3613409024040557, 1e-13),
          "Phi^-1(0.975), Phi^-1(1e-10)");
    std::printf("      chi2_1: %.17g, %.17g\n", nc::chi2_1_quantile(0.95), nc::chi2_1_quantile(0.99));
    check(rel(nc::chi2_1_quantile(0.95), 3.8414588206941236, 1e-13) && rel(nc::chi2_1_quantile(0.99), 6.6348966010212145, 1e-13) &&
              nc::chi2_1_quantile(0.0) == 0.0,
          "chi-squared with one degree of freedom: the 0.95 and 0.99 quantiles");
    double worst = 0.0;
    for (int k = -12; k <= 12; ++k) {
        const double p = k < 0 ? std::pow(10.0, k) : k == 0 ? 0.5 : 1.0 - std::pow(10.0, -k);
        const double back = nc::normal_cdf(nc::normal_quantile(p));
        worst = std::max(worst, std::fabs(back - p) / std::min(p, 1.0 - p));
    }
    std::printf("      round trip: worst relative error (to the smaller tail) %.2g\n", worst);
    check(worst < 1e-12, "Phi(Phi^-1(p)) = p from 1e-12 to 1 - 1e-12");
    const double z = 2.0 * nc::kSqrt2 / 2.0;   // x = 2: z = sqrt 2
    const double below = nc::erfc_nonneg(std::nextafter(z, 0.0)), above = nc::erfc_nonneg(std::nextafter(z, 10.0));
    check(rel(below, above, 1e-14), "erfc: the series and the continued fraction agree at z = sqrt 2");
    std::printf("\n%s\n", g_failed == 0 ? "all passed" : "FAILURES");
    return g_failed == 0 ? 0 : 1;
}
