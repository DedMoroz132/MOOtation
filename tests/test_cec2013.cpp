// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// suite/cec2013.hpp and so_metrics.hpp's peaks_found (task 18, stage 3, §3.5;
// Li, Engelbrecht, Epitropakis 2013):
//   - every problem's global optima, from their closed forms (F1, F2, F4,
//     F7, F8, the compositions' o_i) or refined from a grid (F3, F5, F6), at
//     the height to full precision (cec2013::best) within 1e-9 relative
//     (CEC-1: F5 without the factor 4), and
//     Algorithm 1 counts exactly Table IV's number of them at ε = 1e-4;
//   - the heights to full precision (cec2013::best) agree with Table IV's
//     printed digits;
//   - no point of 20000 random ones above the height;
//   - Algorithm 1: points below the height's ε are not counted, nor a second
//     one within r of a first; the sort by F, largest first;
//   - the suite's Native: -F, the boxes (F5's two ranges).
// ============================================================================

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <mootation/scalar/nelder_mead.hpp>
#include <mootation/so_metrics.hpp>
#include <mootation/suite/suite.hpp>

namespace {

namespace cec = mootation::suite::cec2013;
using Vec = std::vector<double>;

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    if (!ok) {
        ++g_failed;
        std::printf("FAIL  %s\n", what.c_str());
    } else {
        std::printf("ok    %s\n", what.c_str());
    }
}

constexpr double kPi = 3.14159265358979323846;

// a local maximum of F near x, by Nelder-Mead in a small box about it
Vec climb(const cec::Problem& P, const Vec& x, double half)
{
    namespace nm = mootation::nelder_mead;
    nm::Options o;
    o.tol = 1e-15;
    o.max_iter = 100000;
    Vec lo(x.size()), hi(x.size());
    for (std::size_t j = 0; j < x.size(); ++j) {
        lo[j] = x[j] - half;
        hi[j] = x[j] + half;
    }
    const auto r = nm::minimize(
        [&](const Vec& u) {
            Vec z(u.size());
            for (std::size_t j = 0; j < u.size(); ++j) z[j] = lo[j] + u[j] * (hi[j] - lo[j]);
            return -P.value(z);
        },
        Vec(x.size(), 0.5), Vec(x.size(), 0.0), Vec(x.size(), 1.0), o);
    Vec z(x.size());
    for (std::size_t j = 0; j < x.size(); ++j) z[j] = lo[j] + r.x[j] * (hi[j] - lo[j]);
    return z;
}

// the global optima of problem p
std::vector<Vec> optima(int p)
{
    const cec::Info i = cec::info(p);
    const cec::Problem P(p);
    std::vector<Vec> out;
    switch (i.function) {
    case 1: return {{0.0}, {30.0}};
    case 2: return {{0.1}, {0.3}, {0.5}, {0.7}, {0.9}};
    case 3: return {climb(P, {0.08}, 0.01)};
    case 4: return {{3.0, 2.0}, {-2.805118086952745, 3.131312518250573}, {-3.779310253377747, -3.283185991286170},
                    {3.584428340330492, -1.848126526964404}};
    case 5: return {climb(P, {0.0898, -0.7126}, 0.01), climb(P, {-0.0898, 0.7126}, 0.01)};
    case 6: {                                   // a grid of step 0.25, each point climbed, the optima kept
        const int D = i.D;
        const int m = 81;
        std::vector<int> k(static_cast<std::size_t>(D), 0);
        while (true) {
            Vec x(static_cast<std::size_t>(D));
            for (int j = 0; j < D; ++j) x[static_cast<std::size_t>(j)] = -10.0 + 0.25 * k[static_cast<std::size_t>(j)];
            if (P.value(x) > 0.5 * i.height) {
                const Vec z = climb(P, climb(P, x, 0.2), 0.005);
                if (std::fabs(P.value(z) - i.height) < 1e-3) out.push_back(z);
            }
            int j = 0;
            while (j < D && ++k[static_cast<std::size_t>(j)] == m) k[static_cast<std::size_t>(j++)] = 0;
            if (j == D) break;
        }
        return out;
    }
    case 7: {
        std::vector<double> c;
        for (int q = -2; q <= 3; ++q) c.push_back(std::exp((kPi / 2.0 + 2.0 * kPi * q) / 10.0));
        if (i.D == 2) {
            for (double a : c)
                for (double b : c) out.push_back({a, b});
        } else {
            for (double a : c)
                for (double b : c)
                    for (double d : c) out.push_back({a, b, d});
        }
        return out;
    }
    case 8: {
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 4; ++b) out.push_back({(2.0 * a + 1.0) / 6.0, (2.0 * b + 1.0) / 8.0});
        return out;
    }
    default: return P.optima();
    }
}

void problems()
{
    for (int p = 1; p <= cec::kProblems; ++p) {
        const cec::Info i = cec::info(p);
        const cec::Problem P(p);
        const std::vector<Vec> X = optima(p);
        const cec::Best b = cec::best(p);
        Vec F;
        double worst = 0.0;
        for (const Vec& x : X) {
            F.push_back(P.value(x));
            worst = std::max(worst, std::fabs(F.back() - b.value));
        }
        const int found = mootation::so::peaks_found(X, F, i.height, 1e-4, i.r);
        std::mt19937 g(static_cast<unsigned>(p));
        const Vec lo = cec::lower(p), hi = cec::upper(p);
        double above = -1e300;
        for (int k = 0; k < 20000; ++k) {
            Vec x(lo.size());
            for (std::size_t j = 0; j < x.size(); ++j) x[j] = lo[j] + (hi[j] - lo[j]) * mootation::rnd::uniform01(g);
            above = std::max(above, P.value(x));
        }
        check(worst < 1e-9 * std::max(1.0, std::fabs(b.value)) && found == i.n_global && above <= b.value + 1e-12 &&
                  std::fabs(b.value - i.height) < 0.5e-5 * std::max(1.0, std::fabs(i.height)) + (p == 3 ? 2e-7 : 0.0),
              "problem " + std::to_string(p) + " (F" + std::to_string(i.function) + ", D " + std::to_string(i.D) +
                  "): " + std::to_string(found) + " of " + std::to_string(i.n_global) + " optima at Table IV's height " +
                  std::to_string(i.height) + " (|F - F*| at most " + std::to_string(worst) + "), none above F*");
    }
}

void algorithm_1()
{
    using mootation::so::peaks_found;
    const std::vector<Vec> X = {{0.0}, {0.005}, {0.5}, {1.0}};
    const Vec F = {1.0, 1.0 + 1e-9, 0.99, 1.0};
    check(peaks_found(X, F, 1.0, 1e-4, 0.01) == 2 && peaks_found(X, F, 1.0, 0.1, 0.01) == 3 &&
              peaks_found(X, F, 1.0, 1e-4, 2.0) == 1,
          "Algorithm 1: a point within r of one counted is not counted, one below the height's eps neither");
}

void native()
{
    namespace s = mootation::suite;
    s::Spec spec;
    spec.family = "CEC2013";
    spec.n_var = 2;
    spec.n_obj = 1;
    spec.cec_problem = 5;
    const s::Problem nat(spec);
    const double x[2] = {0.0898, -0.7126};
    double f = 0.0;
    nat.evaluate(x, &f, nullptr);
    const auto b = nat.bounds();
    check(std::fabs(f + cec::Problem(5).value({0.0898, -0.7126})) == 0.0 && b.size() == 2 && b[0].first == -1.9 &&
              b[1].second == 1.1,
          "Native CEC2013 problem 5: -F, the box [-1.9, 1.9] x [-1.1, 1.1]");
}

}   // namespace

int main()
{
    problems();
    algorithm_1();
    native();
    std::printf("%s (%d failed)\n", g_failed == 0 ? "PASS" : "FAIL", g_failed);
    return g_failed == 0 ? 0 : 1;
}
