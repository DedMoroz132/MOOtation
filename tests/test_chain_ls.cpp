// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// include/mootation/scalar/chain_ls.hpp (task 25, item 24.2.1, АП-2):
//   the chain property — calls of k1, k2, ... evaluations carrying the state
//     (and the generator) give the point, value and state of one call of
//     k1 + k2 + ..., bit for bit, for Solis-Wets, SSW, MTS-LS1 and MTS-LS2,
//     also through pack/unpack;
//   Solis-Wets at rho = 0 never moves, and rho halves after 4 failures
//     (SW-1);
//   MTS-LS1 by hand: x - SR better kept; worse -> restored + 0.5 SR (MTS-1);
//     an equal value restores; SR halves after a pass without improvement;
//   MTS-LS1 and MTS-LS2 reach the minimum of a separable quadratic;
//   SSW draws dif on floor(0.2 n) consecutive coordinates (mod n) only.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <mootation/scalar/chain_ls.hpp>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

namespace cls = mootation::chain_ls;

// A rotated ill-conditioned quadratic in [0, 1]^n: not separable; its
// minimum 0 at c_i = 0.1 + 0.07 i
double rotated(const std::vector<double>& x)
{
    double s = 0.0;
    for (std::size_t i = 0; i + 1 < x.size(); ++i) {
        const double a = x[i] - (0.1 + 0.07 * static_cast<double>(i));
        const double b = x[i + 1] - (0.1 + 0.07 * static_cast<double>(i + 1));
        s += (a + b) * (a + b) + 10.0 * (a - b) * (a - b);
    }
    return s;
}

// A separable quadratic with its minimum at c_i = 0.1 + 0.07 i
double separable(const std::vector<double>& x)
{
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double d = x[i] - (0.1 + 0.07 * static_cast<double>(i));
        s += static_cast<double>(i + 1) * d * d;
    }
    return s;
}

const std::vector<long long> kSplits = {1, 2, 3, 97, 250, 7, 640};   // 1000 in all

template <class Call>
bool chain_equals_one(std::size_t n, Call call)
{
    const std::vector<double> x0(n, 0.9);
    std::vector<double> xa = x0, xb = x0;
    double fa = rotated(x0), fb = fa;
    std::mt19937 ra(42), rb(42);
    std::vector<double> sa, sb;   // packed states
    long long ua = call(xa, fa, sa, 1000, ra), ub = 0;
    for (long long k : kSplits) ub += call(xb, fb, sb, k, rb);
    return ua == 1000 && ub == 1000 && xa == xb && fa == fb && sa == sb && ra() == rb();
}

void chains()
{
    const std::size_t n = 6;
    const std::vector<double> lo(n, 0.0), hi(n, 1.0);
    check(chain_equals_one(n,
                           [&](std::vector<double>& x, double& fx, std::vector<double>& s, long long k, std::mt19937& r) {
                               cls::SWState st = cls::SWState::unpack(s, n);
                               const long long u = cls::solis_wets(rotated, x, fx, st, k, lo, hi, r);
                               s = st.pack();
                               return u;
                           }),
          "Solis-Wets: calls of 1, 2, 3, 97, 250, 7, 640 = one call of 1000, bit for bit (through pack)");
    cls::SWOptions ssw;
    ssw.subgroups = true;
    ssw.subset_evals = 25;
    check(chain_equals_one(n,
                           [&](std::vector<double>& x, double& fx, std::vector<double>& s, long long k, std::mt19937& r) {
                               cls::SWState st = cls::SWState::unpack(s, n);
                               const long long u = cls::solis_wets(rotated, x, fx, st, k, lo, hi, r, ssw);
                               s = st.pack();
                               return u;
                           }),
          "SSW (subset_evals = 25): the same chain = one call");
    check(chain_equals_one(n,
                           [&](std::vector<double>& x, double& fx, std::vector<double>& s, long long k, std::mt19937&) {
                               cls::MTSState st = cls::MTSState::unpack(s, n);
                               const long long u = cls::mts_ls1(rotated, x, fx, st, k, lo, hi);
                               s = st.pack();
                               return u;
                           }),
          "MTS-LS1: the same chain = one call");
    check(chain_equals_one(n,
                           [&](std::vector<double>& x, double& fx, std::vector<double>& s, long long k, std::mt19937& r) {
                               cls::MTSState st = cls::MTSState::unpack(s, n);
                               const long long u = cls::mts_ls2(rotated, x, fx, st, k, lo, hi, r);
                               s = st.pack();
                               return u;
                           }),
          "MTS-LS2: the same chain = one call");
}

void solis_wets_rules()
{
    const std::vector<double> lo(3, 0.0), hi(3, 1.0);
    std::vector<double> x = {0.2, 0.5, 0.7};
    double fx = rotated(x);
    const double f0 = fx;
    cls::SWState st;
    cls::SWOptions o;
    o.rho0 = 0.0;
    std::mt19937 r(1);
    const long long u = cls::solis_wets(rotated, x, fx, st, 200, lo, hi, r, o);
    check(u == 200 && x == std::vector<double>{0.2, 0.5, 0.7} && fx == f0 && st.rho == 0.0,
          "rho = 0: the point does not move in 200 evaluations");

    // A flat function: every step fails (two evaluations each); rho halves at
    // the 4th failure (SW-1), so after 8 steps = 16 evaluations rho = 0.2 / 4
    cls::SWState flat;
    std::vector<double> y = {0.5, 0.5, 0.5};
    double fy = 1.0;
    std::mt19937 r2(3);
    cls::solis_wets([](const std::vector<double>&) { return 1.0; }, y, fy, flat, 16, lo, hi, r2);
    check(flat.rho == 0.05 && flat.n_failed == 0 && y == std::vector<double>{0.5, 0.5, 0.5},
          "always failing: rho halves after the 4th failure in a row: 0.2 -> 0.05 after 8 steps");
    cls::solis_wets([](const std::vector<double>&) { return 1.0; }, y, fy, flat, 7, lo, hi, r2);
    check(flat.pending && flat.n_failed == 3, "7 more evaluations: 3 failed steps, the 4th step's second trial pending");
}

void mts_by_hand()
{
    const std::vector<double> lo = {0.0}, hi = {1.0};
    auto f = [](const std::vector<double>& x) { return (x[0] - 0.3) * (x[0] - 0.3); };

    std::vector<double> x = {0.6};
    double fx = f(x);
    cls::MTSState st;
    cls::mts_ls1(f, x, fx, st, 1, lo, hi);
    check(x[0] == 0.6 - 0.4 && st.improved, "x = 0.6, SR = 0.4: 0.2 is better, kept");

    x = {0.1};
    fx = f(x);
    st = cls::MTSState{};
    const long long u = cls::mts_ls1(f, x, fx, st, 2, lo, hi);
    check(u == 2 && x[0] == 0.1 + 0.5 * 0.4 && fx == f({0.1 + 0.5 * 0.4}),
          "x = 0.1: 0.1 - 0.4 clipped to 0 is worse; restored, 0.1 + 0.2 kept (MTS-1, not 0 + 0.2)");

    x = {0.5};
    fx = f(x);
    st = cls::MTSState{};
    cls::mts_ls1([](const std::vector<double>& y) { return std::fabs(y[0] - 0.5) < 0.15 ? 0.0 : 1.0; }, x, fx, st, 1, lo,
                 hi);
    check(x[0] == 0.5 && st.second && st.in_pass, "worse at 0.1 -> the second trial is next");
    x = {0.5};
    fx = 0.0;
    st = cls::MTSState{};
    cls::mts_ls1([](const std::vector<double>&) { return 0.0; }, x, fx, st, 1, lo, hi);
    check(x[0] == 0.5 && !st.second && !st.improved, "an equal value restores, no second trial (MTS-3)");
    cls::mts_ls1([](const std::vector<double>&) { return 0.0; }, x, fx, st, 1, lo, hi);
    check(st.sr == 0.2, "a pass without improvement: SR halves at the next pass, 0.4 -> 0.2");
}

void quadratics()
{
    const std::size_t n = 8;
    const std::vector<double> lo(n, 0.0), hi(n, 1.0);
    double worst1 = 0.0, worst2 = 0.0;
    {
        std::vector<double> x(n, 0.95);
        double fx = separable(x);
        cls::MTSState st;
        cls::mts_ls1(separable, x, fx, st, 20000, lo, hi);
        for (std::size_t i = 0; i < n; ++i) worst1 = std::max(worst1, std::fabs(x[i] - (0.1 + 0.07 * static_cast<double>(i))));
    }
    {
        std::vector<double> x(n, 0.95);
        double fx = separable(x);
        cls::MTSState st;
        std::mt19937 r(5);
        cls::mts_ls2(separable, x, fx, st, 20000, lo, hi, r);
        for (std::size_t i = 0; i < n; ++i) worst2 = std::max(worst2, std::fabs(x[i] - (0.1 + 0.07 * static_cast<double>(i))));
    }
    std::printf("      MTS-LS1 max |x_i - c_i| = %.3g, MTS-LS2 = %.3g\n", worst1, worst2);
    check(worst1 < 1e-6 && worst2 < 1e-6, "separable quadratic, n = 8, 20000 evaluations: both within 1e-6 of the minimum");

    std::vector<double> x(n, 0.95);
    double fx = rotated(x);
    const double f0 = fx;
    cls::SWState st;
    std::mt19937 r(9);
    cls::solis_wets(rotated, x, fx, st, 20000, lo, hi, r);
    std::printf("      Solis-Wets on the rotated quadratic: f %.3g -> %.3g\n", f0, fx);
    check(fx < 1e-6 * f0, "Solis-Wets lowers the rotated quadratic a millionfold in 20000 evaluations");
}

void subsets()
{
    const std::size_t n = 20;
    const std::vector<double> lo(n, -10.0), hi(n, 10.0);
    // linear: one of x + d, x - d is better unless d . c = 0
    auto lin = [](const std::vector<double>& x) {
        double s = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) s += (1.0 + static_cast<double>(i)) * x[i];
        return s;
    };
    cls::SWOptions o;
    o.subgroups = true;
    o.subset_evals = 100;   // by default max_evals / 10 = 0 -> 1: a new subset every evaluation
    bool ok = true;
    for (unsigned seed = 0; seed < 50; ++seed) {
        std::vector<double> x(n, 0.0);
        double fx = lin(x);
        cls::SWState st;
        std::mt19937 r(seed);
        cls::solis_wets(lin, x, fx, st, 2, lo, hi, r, o);
        std::vector<std::size_t> moved;
        for (std::size_t i = 0; i < n; ++i)
            if (x[i] != 0.0) moved.push_back(i);
        // 4 coordinates from sub_start on, mod n
        bool contiguous = moved.size() == 4 && st.sub_count == 4;
        for (std::size_t k = 0; contiguous && k < 4; ++k) {
            const std::size_t want = static_cast<std::size_t>((st.sub_start + static_cast<long long>(k)) % 20);
            contiguous = std::find(moved.begin(), moved.end(), want) != moved.end();
        }
        ok = ok && contiguous && fx < 0.0;
    }
    check(ok, "SSW, n = 20: a step moves exactly the 4 consecutive coordinates of its subset (50 seeds)");
}

}   // namespace

int main()
{
    chains();
    solis_wets_rules();
    mts_by_hand();
    quadratics();
    subsets();
    std::printf("\n%s\n", g_failed == 0 ? "all passed" : "FAILURES");
    return g_failed == 0 ? 0 : 1;
}
