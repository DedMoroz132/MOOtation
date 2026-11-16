// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/lbfgsb.hpp: MGH's unconstrained minimization list (Moré, Garbow,
// Hillstrom 1981, §3; f = sum f_i^2, g = 2 J^T f, suite/mgh.hpp) from x_s:
// the minimum MGH's (d) gives (relatively 1e-5, or 1e-10 absolutely where it
// is 0), but for four noted below (where SciPy's L-BFGS-B, the same m, ends
// no nearer); infinite bounds as no bounds, bit for bit; Rosenbrock with x1
// <= 1/2 ends at (1/2, 1/4); bounded variants end where the projected
// gradient vanishes (BLNZ (6.1)); a checksum of the MGH runs, the same bits
// on every platform (dm:: in the problems, + - * / sqrt in the solver); at
// n = 300 in a box, m = 5, 12, 20: the cost per iteration against m (LB-11)
// and the runs' bits; at n = 300 and 3000 the solver's own cost per
// evaluation against n (LB-12) and every evaluation's bits.
// ============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
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

std::uint64_t g_hash = 1469598103934665603ull;     // FNV-1a over the runs' bits (MSVC = g++ = clang)

void mix(double v)
{
    std::uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    g_hash = (g_hash ^ b) * 1099511628211ull;
}

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
        for (double v : r.x) mix(v);
        mix(static_cast<double>(r.nfev));
        std::printf("      %-24s n %2zu: f %.9g (MGH %.6g) nfev %lld it %lld status %d\n", mgh::name(k.id), k.n, r.f,
                    k.fmin, r.nfev, r.iterations, static_cast<int>(r.status));
        bool ok = k.fmin == 0.0 ? r.f <= 1e-10 : std::fabs(r.f - k.fmin) <= 1e-5 * k.fmin;
        std::string note = ": MGH's minimum";
        if (k.id == 26) {                             // a local minimum; SciPy ends there too
            ok = std::fabs(r.f - 2.79505612e-5) <= 1e-6 * 2.79505612e-5;
            note = ": the local minimum 2.79506e-5 (SciPy's too)";
        }
        // ill-conditioned: L-BFGS creeps (SciPy: 1.9e-7, 2.1e-6, 9.2e-8); the run is the same bits
        // everywhere (the checksum below): Powell's ends at 3.8e-7, Watson's at 1.39996e-6, 8.5e-8
        if (k.id == 3 || (k.id == 20 && k.n > 6)) {
            ok = r.f <= (k.id == 3 ? 1e-6 : k.n == 9 ? 1.6e-6 : 1e-7);
            note = ": below 1e-6, 1.6e-6, 1e-7 (MGH's minimum not reached, nor by SciPy)";
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
    std::printf("checksum %016llx\n", static_cast<unsigned long long>(g_hash));
    check(g_hash == 0x8ae4a65d9e843c71ull, "the runs' bits as on every platform (dm:: and + - * / sqrt only)");
    // speed (task 30, §30.4; LB-11): n = 300 in a box, m = 5, 12, 20. From m = 5 to 20 the
    // time per iteration grew some 150 times (the k^4 n of M's forming), now some 7; the
    // runs keep their bits (the checksum of the runs before LB-11)
    {
        const std::size_t n = 300;
        auto rosen = [](const Vec& x, Vec& g) {        // extended Rosenbrock
            double f = 0.0;
            g.assign(x.size(), 0.0);
            for (std::size_t i = 0; i + 1 < x.size(); i += 2) {
                const double t = x[i + 1] - x[i] * x[i], u = 1.0 - x[i];
                f += 100.0 * t * t + u * u;
                g[i] += -400.0 * t * x[i] - 2.0 * u;
                g[i + 1] += 200.0 * t;
            }
            return f;
        };
        auto quad = [](const Vec& x, Vec& g) {         // ill-conditioned, chained; a third outside the box
            const std::size_t k = x.size();
            double f = 0.0;
            g.assign(k, 0.0);
            for (std::size_t i = 0; i < k; ++i) {
                const double d = 1.0 + 1e4 * static_cast<double>(i * 37 % k) / static_cast<double>(k);
                const double c = (i % 3 == 0) ? 1.5 : 0.3 * static_cast<double>(i % 7) / 7.0;
                const double r = x[i] - c;
                f += d * r * r;
                g[i] += 2.0 * d * r;
                if (i + 1 < k) {
                    const double e = x[i] - x[i + 1];
                    f += 10.0 * e * e;
                    g[i] += 20.0 * e;
                    g[i + 1] -= 20.0 * e;
                }
            }
            return f;
        };
        std::uint64_t hash = 1469598103934665603ull;
        double per_it[2][3] = {};
        const int ms[] = {5, 12, 20};
        for (int which = 0; which < 2; ++which)
            for (int a = 0; a < 3; ++a) {
                LB::Problem p;
                p.n = n;
                p.lo.assign(n, which == 0 ? -2.0 : -1.0);
                p.hi.assign(n, which == 0 ? 2.0 : 1.0);
                Vec x0(n, 0.0);
                if (which == 0) {
                    p.fg = rosen;
                    for (std::size_t i = 0; i < n; ++i) {
                        x0[i] = i % 2 == 0 ? -1.2 : 1.0;
                        if (i % 3 == 0) p.hi[i] = 0.8;
                    }
                } else {
                    p.fg = quad;
                }
                LB::Options o;
                o.m = ms[a];
                o.pgtol = 1e-12;
                o.max_nfev = 3000;
                LB::Result r;
                double best = 1e300;
                for (int rep = 0; rep < 3; ++rep) {    // the least of three against the machine's noise
                    const auto t0 = std::chrono::steady_clock::now();
                    r = LB::solve(p, x0, o);
                    best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                }
                per_it[which][a] = best / static_cast<double>(std::max(r.iterations, 1LL));
                std::uint64_t b;
                for (double v : r.x) {
                    std::memcpy(&b, &v, sizeof b);
                    hash = (hash ^ b) * 1099511628211ull;
                }
                for (const double v : {static_cast<double>(r.nfev), static_cast<double>(r.iterations)}) {
                    std::memcpy(&b, &v, sizeof b);
                    hash = (hash ^ b) * 1099511628211ull;
                }
                std::printf("      n 300 %s m %2d: %lld iterations, %lld evaluations, %.4f ms per iteration, f %.10g\n",
                            which == 0 ? "rosenbrock" : "quadratic ", ms[a], r.iterations, r.nfev, 1e3 * per_it[which][a], r.f);
            }
        std::printf("      n 300 checksum %016llx\n", static_cast<unsigned long long>(hash));
        check(hash == 0x1281c1698fa2cdeaull, "n = 300, m = 5, 12, 20: the runs' bits as before LB-11 (on every platform)");
        for (int which = 0; which < 2; ++which) {
            const double ratio = per_it[which][2] / per_it[which][0];
            check(ratio < 30.0, std::string("n = 300: m = 20 costs less than 30 times m = 5 per iteration (") +
                                    (which == 0 ? "rosenbrock" : "quadratic") + ", " + std::to_string(ratio) + ")");
        }
    }
    // speed (task 30, §30.11; LB-12): the solver's own time per evaluation (the problem's
    // taken out) at n = 3000 against n = 300, m = 5, the quadratic above in [-1, 1]: some 11
    // times, linear in n (before LB-12 as well, at some 2.5 times the cost); a growth as n^2
    // would give some 100. And the runs' bits: every evaluation's x, f and g, as before LB-12
    {
        auto quad = [](const Vec& x, Vec& g) {
            const std::size_t k = x.size();
            double f = 0.0;
            g.assign(k, 0.0);
            for (std::size_t i = 0; i < k; ++i) {
                const double d = 1.0 + 1e4 * static_cast<double>(i * 37 % k) / static_cast<double>(k);
                const double c = (i % 3 == 0) ? 1.5 : 0.3 * static_cast<double>(i % 7) / 7.0;
                const double r = x[i] - c;
                f += d * r * r;
                g[i] += 2.0 * d * r;
                if (i + 1 < k) {
                    const double e = x[i] - x[i + 1];
                    f += 10.0 * e * e;
                    g[i] += 20.0 * e;
                    g[i + 1] -= 20.0 * e;
                }
            }
            return f;
        };
        using clock = std::chrono::steady_clock;
        std::uint64_t trail = 1469598103934665603ull;
        double own[2] = {};
        const std::size_t sizes[] = {300, 3000};
        for (int a = 0; a < 2; ++a) {
            const std::size_t n = sizes[a];
            LB::Problem p;
            p.n = n;
            p.lo.assign(n, -1.0);
            p.hi.assign(n, 1.0);
            double fg_seconds = 0.0;
            bool on_trail = false;
            p.fg = [&](const Vec& x, Vec& g) {
                const auto t0 = clock::now();
                const double f = quad(x, g);
                fg_seconds += std::chrono::duration<double>(clock::now() - t0).count();
                if (on_trail) {
                    std::uint64_t b;
                    auto put = [&](double v) {
                        std::memcpy(&b, &v, sizeof b);
                        trail = (trail ^ b) * 1099511628211ull;
                    };
                    for (double v : x) put(v);
                    put(f);
                    for (double v : g) put(v);
                }
                return f;
            };
            LB::Options o;
            o.pgtol = 1e-12;
            o.max_nfev = 600;
            LB::Result r;
            double best = 1e300;
            for (int rep = 0; rep < 3; ++rep) {        // the least of three against the machine's noise
                fg_seconds = 0.0;
                const auto t0 = clock::now();
                r = LB::solve(p, Vec(n, 0.0), o);
                best = std::min(best, std::chrono::duration<double>(clock::now() - t0).count() - fg_seconds);
            }
            own[a] = best / static_cast<double>(std::max(r.nfev, 1LL));
            on_trail = true;
            LB::solve(p, Vec(n, 0.0), o);
            std::printf("      n %4zu quadratic m 5: %lld iterations, %lld evaluations, %.1f us of the solver's own per evaluation\n",
                        n, r.iterations, r.nfev, 1e6 * own[a]);
        }
        std::printf("      n 300 and 3000 trail %016llx\n", static_cast<unsigned long long>(trail));
        check(trail == 0x30c1a3b0a5bf8136ull, "n = 300 and 3000: every evaluation's x, f, g as before LB-12 (on every platform)");
        const double ratio = own[1] / own[0];
        check(ratio < 40.0, "n = 3000 costs the solver less than 40 times n = 300 per evaluation (" +
                                std::to_string(ratio) + ")");
    }
    std::printf(g_failed ? "FAILED\n" : "all passed\n");
    return g_failed ? 1 : 0;
}
