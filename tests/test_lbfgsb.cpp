// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/lbfgsb.hpp: MGH's unconstrained minimization list (Moré, Garbow,
// Hillstrom 1981, §3; f = sum f_i^2, g = 2 J^T f, suite/mgh.hpp) from x_s:
// the minimum MGH's (d) gives (relatively 1e-5, or 1e-10 absolutely where it
// is 0), but for four noted below (where SciPy's L-BFGS-B, the same m, ends
// no nearer); infinite bounds as no bounds, bit for bit; Rosenbrock with x1
// <= 1/2 ends at (1/2, 1/4); bounded variants end where the projected
// gradient vanishes (BLNZ (6.1)).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <mootation/scalar/lbfgsb.hpp>
#include <mootation/suite/mgh.hpp>

namespace {

namespace LB = mootation::scalar::lbfgsb;
namespace mgh = mootation::suite::mgh;
using LB::Vec;

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    if (!ok) ++g_failed;
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
}

LB::Problem sum_of_squares(int id, std::size_t n, std::size_t m)
{
    LB::Problem p;
    p.n = n;
    p.fg = [=](const Vec& x, Vec& g) {
        Vec f, J;
        mgh::evaluate(id, n, m, x, f, &J);
        double s = 0.0;
        g.assign(n, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            s += f[i] * f[i];
            for (std::size_t j = 0; j < n; ++j) g[j] += 2.0 * J[i * n + j] * f[i];
        }
        return s;
    };
    return p;
}

}   // namespace

int main()
{
    struct Case { int id; std::size_t n, m; double fmin; };
    // MGH's unconstrained list and the minima of their (d)
    const Case cases[] = {
        {7, 3, 3, 0.0},          {18, 6, 13, 5.65565e-3}, {9, 3, 15, 1.12793e-8}, {3, 2, 2, 0.0},
        {12, 3, 10, 0.0},        {25, 10, 12, 0.0},       {20, 6, 31, 2.28767e-3}, {20, 9, 31, 1.39976e-6},
        {20, 12, 31, 4.72238e-10}, {23, 4, 5, 2.24997e-5}, {23, 10, 11, 7.08765e-5}, {24, 4, 8, 9.37629e-6},
        {24, 10, 20, 2.93660e-4}, {4, 2, 3, 0.0},         {16, 4, 20, 85822.2},     {11, 3, 99, 0.0},
        {26, 10, 10, 0.0},       {21, 10, 10, 0.0},       {22, 12, 12, 0.0},        {5, 2, 3, 0.0},
        {14, 4, 6, 0.0},         {35, 8, 8, 3.51687e-3},  {35, 10, 10, 6.50395e-3},
    };
    for (const Case& k : cases) {
        LB::Options o;
        o.pgtol = 1e-10;
        o.ftol = 1e-15;
        const LB::Result r = LB::solve(sum_of_squares(k.id, k.n, k.m), mgh::start(k.id, k.n, 1.0), o);
        std::printf("      %-24s n %2zu: f %.9g (MGH %.6g) nfev %lld it %lld status %d\n", mgh::name(k.id), k.n, r.f,
                    k.fmin, r.nfev, r.iterations, static_cast<int>(r.status));
        bool ok = k.fmin == 0.0 ? r.f <= 1e-10 : std::fabs(r.f - k.fmin) <= 1e-5 * k.fmin;
        std::string note = ": MGH's minimum";
        if (k.id == 26) {                             // a local minimum; SciPy ends there too
            ok = std::fabs(r.f - 2.79505612e-5) <= 1e-6 * 2.79505612e-5;
            note = ": the local minimum 2.79506e-5 (SciPy's too)";
        }
        // ill-conditioned: L-BFGS creeps (SciPy: 1.9e-7, 2.1e-6, 9.2e-8); on Powell's the path
        // follows the rounding of exp: 5.8e-7 with MSVC's, 5.1e-4 and 5.2e-4 with macOS's and glibc's
        if (k.id == 3 || (k.id == 20 && k.n > 6)) {
            ok = r.f <= (k.id == 3 ? 1e-3 : k.n == 9 ? 1.6e-6 : 1e-7);
            note = ": below 1e-3, 1.6e-6, 1e-7 (MGH's minimum not reached, nor by SciPy)";
        }
        check(ok, std::string("MGH ") + mgh::name(k.id) + " n=" + std::to_string(k.n) + note);
    }
    // bounds
    {
        LB::Problem p = sum_of_squares(21, 10, 10);
        const LB::Result a = LB::solve(p, mgh::start(21, 10, 1.0), LB::Options{});
        p.lo.assign(10, -LB::kInf);
        p.hi.assign(10, LB::kInf);
        const LB::Result b = LB::solve(p, mgh::start(21, 10, 1.0), LB::Options{});
        check(a.x == b.x && a.nfev == b.nfev, "bounds: infinite bounds give the unbounded run bit for bit");
        LB::Problem q = sum_of_squares(1, 2, 2);
        q.lo = {-LB::kInf, -LB::kInf};
        q.hi = {0.5, LB::kInf};
        LB::Options o;
        o.pgtol = 1e-10;
        const LB::Result c = LB::solve(q, {-1.2, 1.0}, o);
        std::printf("      rosenbrock x1 <= 1/2: x %.17g %.17g f %.17g nfev %lld\n", c.x[0], c.x[1], c.f, c.nfev);
        check(c.x[0] == 0.5 && std::fabs(c.x[1] - 0.25) <= 1e-9 && std::fabs(c.f - 0.25) <= 1e-12,
              "bounds: Rosenbrock with x1 <= 1/2 ends at (1/2, 1/4), f = 1/4");
        struct Box { int id; std::size_t n, m; double lo, hi; };
        const Box boxes[] = {{24, 4, 8, 0.0, 0.5}, {35, 8, 8, 0.0, 0.6}, {23, 4, 5, 0.4, LB::kInf},
                             {14, 4, 6, -LB::kInf, 0.8}};
        for (const Box& k : boxes) {
            LB::Problem r = sum_of_squares(k.id, k.n, k.m);
            r.lo.assign(k.n, k.lo);
            r.hi.assign(k.n, k.hi);
            const LB::Result z = LB::solve(r, mgh::start(k.id, k.n, 1.0), o);
            double pg = 0.0;
            for (std::size_t i = 0; i < k.n; ++i)
                pg = std::max(pg, std::fabs(std::min(std::max(z.x[i] - z.g[i], k.lo), k.hi) - z.x[i]));
            std::printf("      %s in [%g, %g]: f %.10g pg %.2g status %d nfev %lld\n", mgh::name(k.id), k.lo, k.hi, z.f,
                        pg, static_cast<int>(z.status), z.nfev);
            check((z.status == LB::Status::pgtol || z.status == LB::Status::line_search) && pg < 1e-7,
                  std::string("bounds: ") + mgh::name(k.id) + " in a box ends with the projected gradient < 1e-7 (pgtol, or the search at rounding)");
        }
    }
    std::printf(g_failed ? "FAILED\n" : "all passed\n");
    return g_failed ? 1 : 0;
}
