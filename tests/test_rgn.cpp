// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// include/mootation/scalar/rgn.hpp (task 25, item 24.3.6):
//   linear least squares: the first step lands on the solution of
//     lsq::least_squares (central differences are exact for linear residuals
//     at any sampling scale);
//   Rosenbrock's residuals (10 (x2 - x1^2), 1 - x1) from (-1.2, 1): (1, 1);
//   a minimum outside the box: the active set stops at the bound (B1-B4);
//   one residual of two variables (a singular J^T J): the truncated
//     eigendecomposition gives the minimum-norm step to f = 0;
//   the budget: never more evaluations than max_evals;
//   the counts: evaluations = the residual calls made.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <mootation/scalar/lsq.hpp>
#include <mootation/scalar/rgn.hpp>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

namespace rgn = mootation::rgn;
namespace lsq = mootation::lsq;

void linear()
{
    const double A[5][3] = {{1, 2, 0}, {0, 1, 3}, {2, 0, 1}, {1, 1, 1}, {3, -1, 2}};
    const double b[5] = {1, -2, 3, 0.5, 4};
    auto res = [&](const std::vector<double>& x, std::vector<double>& r) {
        r.assign(5, 0.0);
        for (int t = 0; t < 5; ++t) r[t] = A[t][0] * x[0] + A[t][1] * x[1] + A[t][2] * x[2] - b[t];
    };
    lsq::Matrix M(5, 3);
    for (std::size_t t = 0; t < 5; ++t)
        for (std::size_t k = 0; k < 3; ++k) M(t, k) = A[t][k];
    std::vector<double> xs;
    lsq::least_squares(M, std::vector<double>(b, b + 5), xs);
    long long calls = 0;
    const rgn::Result R = rgn::minimize(
        [&](const std::vector<double>& x, std::vector<double>& r) {
            ++calls;
            res(x, r);
        },
        {0.0, 0.0, 0.0}, {-10, -10, -10}, {10, 10, 10});
    double err = 0.0;
    for (int k = 0; k < 3; ++k) err = std::max(err, std::fabs(R.x[k] - xs[k]));
    std::vector<double> rs;
    res(xs, rs);
    double fs = 0.0;
    for (double v : rs) fs += 0.5 * v * v;
    std::printf("      linear: |x - x*| = %.3g, f = %.17g (lsq %.17g), %d iterations, %lld evaluations\n", err,
                R.f, fs, R.iterations, R.evaluations);
    check(err < 1e-10 && std::fabs(R.f - fs) <= 1e-12 * fs, "linear least squares: the solution of lsq::least_squares");
    check(R.evaluations == calls, "evaluations = the residual calls");
    check(R.stop == rgn::Stop::no_decrease || R.stop == rgn::Stop::small_df || R.stop == rgn::Stop::small_dx,
          "it ends by a lack-of-progress criterion");
}

void rosenbrock()
{
    auto res = [](const std::vector<double>& x, std::vector<double>& r) {
        r = {10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]};
    };
    const rgn::Result R = rgn::minimize(res, {-1.2, 1.0}, {-2, -2}, {2, 2});
    std::printf("      Rosenbrock: x = (%.15g, %.15g), f = %.3g, %d iterations, %lld evaluations\n", R.x[0], R.x[1], R.f,
                R.iterations, R.evaluations);
    check(std::fabs(R.x[0] - 1.0) < 1e-8 && std::fabs(R.x[1] - 1.0) < 1e-8 && R.f < 1e-16,
          "Rosenbrock from (-1.2, 1): (1, 1)");
}

void bounds()
{
    auto res = [](const std::vector<double>& x, std::vector<double>& r) { r = {x[0] - 2.0, x[1] - 0.5, 0.1 * x[0] * x[1]}; };
    const rgn::Result R = rgn::minimize(res, {0.2, 0.2}, {0, 0}, {1, 1});
    std::printf("      bound: x = (%.15g, %.15g), f = %.15g\n", R.x[0], R.x[1], R.f);
    bool inside = R.x[0] >= 0.0 && R.x[0] <= 1.0 && R.x[1] >= 0.0 && R.x[1] <= 1.0;
    // at x1 = 1: d/dx2 of 1/2 ((x2 - 0.5)^2 + 0.01 x2^2) = 0 -> x2 = 0.5 / 1.01
    check(inside && R.x[0] == 1.0 && std::fabs(R.x[1] - 0.5 / 1.01) < 1e-8,
          "the minimum lies past x1 = 1: x1 stops at the bound, x2 = 0.5 / 1.01");
}

void singular()
{
    auto res = [](const std::vector<double>& x, std::vector<double>& r) { r = {x[0] + x[1] - 1.0}; };
    const rgn::Result R = rgn::minimize(res, {0.0, 0.0}, {-5, -5}, {5, 5});
    check(R.f < 1e-28 && std::fabs(R.x[0] - 0.5) < 1e-12 && std::fabs(R.x[1] - 0.5) < 1e-12,
          "one residual, two variables: the minimum-norm step from 0 to (0.5, 0.5)");
}

void budget()
{
    auto res = [](const std::vector<double>& x, std::vector<double>& r) {
        r = {10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]};
    };
    bool ok = true;
    for (long long b = 1; b <= 40; ++b) {
        rgn::Options o;
        o.max_evals = b;
        const rgn::Result R = rgn::minimize(res, {-1.2, 1.0}, {-2, -2}, {2, 2}, o);
        std::vector<double> r;
        res(R.x, r);
        ok = ok && R.evaluations <= b && R.f <= 0.5 * (4.4 * 4.4 + 2.2 * 2.2) && r == R.r;
    }
    check(ok, "max_evals = 1 ... 40: never more evaluations; the answer's residuals are its own; f never above f(x0)");
}

}   // namespace

int main()
{
    linear();
    rosenbrock();
    bounds();
    singular();
    budget();
    std::printf("\n%s\n", g_failed == 0 ? "all passed" : "FAILURES");
    return g_failed == 0 ? 0 : 1;
}
