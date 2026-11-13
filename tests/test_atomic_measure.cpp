// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// include/mootation/scalar/atomic_measure.hpp (task 25, item 24.3.7):
//   Kammler 1979, Table 1: the best least squares sums of n = 1, 2, 6
//     exponentials to F(t) = 1/(1 + t) on (0, inf), dmu = dt, by phase 2
//     (fit) on a quadrature of the integral (t = e^u, the trapezoidal rule
//     in u on [-30, 30]): ||F - Y_n|| = 3.18e-1, 1.31e-1, 9.73e-3 and the
//     table's exponents;
//   the whole algorithm (solve) on exact data of three exponentials recovers
//     them, certified (rho <= 0 on the fine grid);
//   one atom fitted to that data: a stationary point that the certificate
//     refuses (rho > 0 somewhere).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <mootation/scalar/atomic_measure.hpp>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

namespace am = mootation::atomic;

bool rel(double a, double b, double tol) { return std::fabs(a - b) <= tol * std::fabs(b); }

void kammler()
{
    // t = e^u, dt = e^u du: the trapezoidal rule in u
    std::vector<double> t, w, y;
    const double du = 0.01;
    for (int k = -3000; k <= 3000; ++k) {
        const double u = du * k;
        t.push_back(std::exp(u));
        w.push_back(du * std::exp(u) * (k == -3000 || k == 3000 ? 0.5 : 1.0));
        y.push_back(1.0 / (1.0 + std::exp(u)));
    }
    double norm2 = 0.0;
    for (std::size_t i = 0; i < t.size(); ++i) norm2 += w[i] * y[i] * y[i];
    check(std::fabs(norm2 - 1.0) < 1e-9, "the quadrature gives ||F||^2 = 1");
    const am::Kernel K = am::exp_kernel(t);

    struct Case {
        int n;
        double norm;
        std::vector<double> a, l;
    };
    const std::vector<Case> table = {
        {1, 3.18e-1, {.68238477}, {.25894690}},
        {2, 1.31e-1, {.11886378, .75040414}, {.03614215, .57059921}},
        {6, 9.73e-3, {.00045813, .00488099, .03049140, .13602053, .39569802, .42271966},
         {.00013437, .00211543, .01664790, .09319351, .42202932, 1.67013802}}};
    for (const Case& c : table) {
        std::vector<double> l0;
        for (double v : c.l) l0.push_back(1.2 * v);                   // 20 % off the table
        am::Options o;
        o.max_iter = 5000;                                            // AM-7: linear convergence at a fixed p
        o.tol = 1e-12;
        const am::Result r = am::fit(K, y, w, l0, 0.0, 100.0, o);
        double worst_l = 0.0, worst_a = 0.0;
        bool sizes = r.atoms.size() == c.l.size();
        for (std::size_t k = 0; sizes && k < c.l.size(); ++k) {
            std::printf("        lambda %.10g (table %.8g)  a %.10g (table %.8g)\n", r.atoms[k].lambda, c.l[k], r.atoms[k].a, c.a[k]);
            worst_l = std::max(worst_l, std::fabs(r.atoms[k].lambda - c.l[k]) / c.l[k]);
            worst_a = std::max(worst_a, std::fabs(r.atoms[k].a - c.a[k]) / c.a[k]);
        }
        std::printf("      n = %d: ||F - Y|| = %.6g (table %.3g), largest relative difference: lambda %.2g, a %.2g, "
                    "%d steps\n",
                    c.n, r.residual, c.norm, worst_l, worst_a, r.iterations);
        check(sizes && r.converged && rel(r.residual, c.norm, 5e-3) && worst_l < 1e-4 && worst_a < 1e-4,
              "Kammler Table 1, n = " + std::to_string(c.n) + ": the norm to 3 digits, the parameters to 1e-4");
    }
}

std::vector<double> three(const std::vector<double>& t)
{
    std::vector<double> y;
    for (double s : t) y.push_back(1.0 * std::exp(-0.3 * s) + 2.0 * std::exp(-1.5 * s) + 0.5 * std::exp(-6.0 * s));
    return y;
}

void whole()
{
    std::vector<double> t;
    for (int i = 0; i < 60; ++i) t.push_back(0.1 * i);
    const std::vector<double> y = three(t), w(t.size(), 1.0);
    const am::Kernel K = am::exp_kernel(t);
    const am::Result r = am::solve(K, y, w, 0.01, 20.0);
    std::printf("      solve: %zu atoms, residual %.3g, rho_max %.3g, %d grid(s), certified %d\n", r.atoms.size(),
                r.residual, r.rho_max, r.grids, r.certified ? 1 : 0);
    for (const am::Atom& a : r.atoms) std::printf("        lambda %.10g  a %.10g\n", a.lambda, a.a);
    const double L[3] = {0.3, 1.5, 6.0}, A[3] = {1.0, 2.0, 0.5};
    bool ok = r.atoms.size() == 3 && r.certified;
    for (std::size_t k = 0; ok && k < 3; ++k) ok = rel(r.atoms[k].lambda, L[k], 1e-6) && rel(r.atoms[k].a, A[k], 1e-6);
    check(ok, "exact data of three exponentials: recovered to 1e-6, certified");

    const am::Result one = am::fit(K, y, w, {1.0}, 0.01, 20.0);
    const double rho = am::rho_max(K, y, w, one, {0.01, 0.1, 0.3, 1.0, 3.0, 6.0, 10.0, 20.0});
    std::printf("      one atom: lambda %.6g, rho_max %.3g\n", one.atoms[0].lambda, rho);
    check(one.converged && rho > 1e-6, "one atom to the same data: converged, refused by the certificate");
}

}   // namespace

int main()
{
    kammler();
    whole();
    std::printf("\n%s\n", g_failed == 0 ? "all passed" : "FAILURES");
    return g_failed == 0 ? 0 : 1;
}
