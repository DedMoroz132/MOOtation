// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/line_search.hpp against Moré & Thuente 1994, §5: their six functions
// (5.1)-(5.4) from a_0 = 1e-3, 1e-1, 1e1, 1e3 with their mu and eta, Tables
// I-VI: the trials m and the a_m they print (within one unit of its last
// printed digit), and that a_m satisfies (1.1) and (1.2).
// ============================================================================

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>

#include <mootation/scalar/line_search.hpp>

namespace {

namespace LS = mootation::scalar::line_search;

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    if (!ok) ++g_failed;
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
}

const double kPi = 3.141592653589793;

void f1(double a, double& f, double& g)              // (5.1), beta = 2
{
    const double b = 2.0, d = a * a + b;
    f = -a / d;
    g = (a * a - b) / (d * d);
}

void f2(double a, double& f, double& g)              // (5.2), beta = 0.004
{
    const double t = a + 0.004;
    f = std::pow(t, 5) - 2.0 * std::pow(t, 4);
    g = 5.0 * std::pow(t, 4) - 8.0 * std::pow(t, 3);
}

void f3(double a, double& f, double& g)              // (5.3), beta = 0.01, l = 39
{
    const double b = 0.01, l = 39.0;
    double p0, d0;
    if (a <= 1.0 - b) { p0 = 1.0 - a; d0 = -1.0; }
    else if (a >= 1.0 + b) { p0 = a - 1.0; d0 = 1.0; }
    else { p0 = (a - 1.0) * (a - 1.0) / (2.0 * b) + 0.5 * b; d0 = (a - 1.0) / b; }
    f = p0 + 2.0 * (1.0 - b) / (l * kPi) * std::sin(l * kPi / 2.0 * a);
    g = d0 + (1.0 - b) * std::cos(l * kPi / 2.0 * a);
}

std::function<void(double, double&, double&)> f4(double b1, double b2)   // (5.4)
{
    auto gam = [](double b) { return std::sqrt(1.0 + b * b) - b; };
    const double g1 = gam(b1), g2 = gam(b2);
    return [=](double a, double& f, double& g) {
        const double r1 = std::sqrt((1.0 - a) * (1.0 - a) + b2 * b2), r2 = std::sqrt(a * a + b1 * b1);
        f = g1 * r1 + g2 * r2;
        g = g1 * (a - 1.0) / r1 + g2 * a / r2;
    };
}

}   // namespace

int main()
{
    struct Table {
        const char* name;
        std::function<void(double, double&, double&)> phi;
        double mu, eta;
        int m[4];
        double am[4];
        double unit[4];      // one unit of a_m's last printed digit
    };
    const Table tables[] = {
        {"I (5.1)", f1, 0.001, 0.1, {6, 3, 1, 4}, {1.4, 1.4, 10.0, 37.0}, {0.1, 0.1, 1.0, 1.0}},
        {"II (5.2)", f2, 0.1, 0.1, {12, 8, 8, 11}, {1.6, 1.6, 1.6, 1.6}, {0.1, 0.1, 0.1, 0.1}},
        {"III (5.3)", f3, 0.1, 0.1, {12, 12, 10, 13}, {1.0, 1.0, 1.0, 1.0}, {0.1, 0.1, 0.1, 0.1}},
        {"IV (5.4) 0.001 0.001", f4(0.001, 0.001), 0.001, 0.001, {4, 1, 3, 4}, {0.08, 0.10, 0.35, 0.83}, {0.01, 0.01, 0.01, 0.01}},
        {"V (5.4) 0.01 0.001", f4(0.01, 0.001), 0.001, 0.001, {6, 3, 7, 8}, {0.075, 0.078, 0.073, 0.076}, {0.001, 0.001, 0.001, 0.001}},
        {"VI (5.4) 0.001 0.01", f4(0.001, 0.01), 0.001, 0.001, {13, 11, 8, 11}, {0.93, 0.93, 0.92, 0.92}, {0.01, 0.01, 0.01, 0.01}},
    };
    for (const Table& t : tables) {
        double f0, g0;
        t.phi(0.0, f0, g0);
        double a0 = 1e-3;
        for (int k = 0; k < 4; ++k, a0 *= 100.0) {
            LS::Options o;
            o.mu = t.mu;
            o.eta = t.eta;
            const LS::Result r = LS::search(t.phi, f0, g0, a0, o);
            const bool wolfe = r.f <= f0 + t.mu * g0 * r.a && std::fabs(r.g) <= t.eta * std::fabs(g0);
            const bool same_a = std::fabs(r.a - t.am[k]) <= t.unit[k] + 1e-12;
            std::printf("      Table %s a0 %g: m %d (theirs %d), a_m %.6g (theirs %g), phi' %.2g, info %d\n", t.name, a0,
                        r.nfev, t.m[k], r.a, t.am[k], r.g, static_cast<int>(r.info));
            check(wolfe && r.info == LS::Info::converged,
                  std::string("Table ") + t.name + " a0 " + std::to_string(a0) + ": (1.1) and (1.2) hold");
            check(r.nfev == t.m[k] && same_a, std::string("Table ") + t.name + " a0 " + std::to_string(a0) +
                                                  ": their m and a_m");
        }
    }
    std::printf(g_failed ? "FAILED\n" : "all passed\n");
    return g_failed ? 1 : 0;
}
