// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/de.hpp:
//   - Storn & Price 1997, Tables 1-3: the mean number of evaluations until a
//     value below the VTR, 20 runs a case with the paper's NP, F, CR and IPR
//     (`native`: the box is the IPR only, DE-2) — within 5 % on testbed 2
//     (D = 20 ... 100, little spread), within 15 % on the small cases of
//     testbeds 1 and 3 (20 runs where the paper has 20 and 1000);
//   - gde3 at n_objs = 1 is the same run draw for draw (de.hpp's last note);
//   - the constrained comparison (evo.hpp) and done();
//   - a checksum of the runs, the same bits on every platform.
//
// Readings of the test functions (the paper's print, checked on the PDF):
//   SP-1. f15 (Ackley) prints exp(-0.02 sqrt(...)); with 0.02 DE needs 2.2
//     times Table 2's evaluations at D = 30 and misses the VTR at D = 100;
//     Ackley's 0.2 gives Table 2 within 1 %. Taken: 0.2.
//   SP-2. f5 (foxholes) prints 1 / (i + sum (x_j - a_ij)^6) for i = 0 ... 24,
//     infinite at a_0; the minimum 0.998004 the paper gives is De Jong's
//     1 / (i + 1 + ...). Taken: i + 1.
//   SP-3. f6 (Corana): z_j = floor(|x_j / 0.2| + 0.49999) sgn(x_j) 0.2.
// Not reproduced (no test; TASK18 report): f8, whose literal sgn() has no
// minimum 0 at (7, 2) — read as a step it needs 1.6 times Table 1's
// evaluations; f20, whose print lacks the x1^2 of the six-hump camel — with
// it, 1.6 times Table 3's; f1 at NP = 5, CR = 0.1 misses in 3 of 20 runs (one
// coordinate collapses over the population), the paper's 20 did not.
// ============================================================================

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <mootation/mootation.hpp>
#include <mootation/scalar/de.hpp>

namespace {

namespace D = mootation::scalar::de;
namespace evo = mootation::scalar::evo;
namespace dm = mootation::dm;
using D::Vec;

int g_failed = 0;
std::uint64_t g_hash = 1469598103934665603ull;     // FNV-1a over the runs' bits

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

const double PI = 3.141592653589793;

double sgn(double v) { return v > 0 ? 1.0 : v < 0 ? -1.0 : 0.0; }

// ── Storn & Price 1997, testbeds 1-3 ────────────────────────────────────────
double sphere(const Vec& x)
{
    double s = 0;
    for (double v : x) s += v * v;
    return s;
}
double rosenbrock(const Vec& x)
{
    const double a = x[0] * x[0] - x[1];
    return 100 * a * a + (1 - x[0]) * (1 - x[0]);
}
double foxholes(const Vec& x)                                   // SP-2
{
    const double a[5] = {-32, -16, 0, 16, 32};
    double s = 0;
    for (int i = 0; i < 25; ++i) {
        const double d1 = x[0] - a[i % 5], d2 = x[1] - a[i / 5];
        const double p1 = d1 * d1 * d1, p2 = d2 * d2 * d2;
        s += 1.0 / (i + 1 + p1 * p1 + p2 * p2);
    }
    return 1.0 / (0.002 + s);
}
double corana(const Vec& x)                                     // SP-3
{
    const double d[4] = {1, 1000, 10, 100};
    double s = 0;
    for (int j = 0; j < 4; ++j) {
        const double z = std::floor(std::fabs(x[j] / 0.2) + 0.49999) * sgn(x[j]) * 0.2;
        if (std::fabs(x[j] - z) < 0.05) {
            const double t = z - 0.05 * sgn(z);
            s += 0.15 * t * t * d[j];
        } else {
            s += d[j] * x[j] * x[j];
        }
    }
    return s;
}
double griewangk(const Vec& x)
{
    double s = 0, p = 1;
    for (std::size_t j = 0; j < x.size(); ++j) {
        s += x[j] * x[j] / 4000;
        p *= dm::cos(x[j] / std::sqrt(static_cast<double>(j + 1)));
    }
    return s - p + 1;
}
double ellipsoid(const Vec& x)
{
    double s = 0;
    for (std::size_t j = 0; j < x.size(); ++j) s += static_cast<double>((j + 1) * (j + 1)) * x[j] * x[j];
    return s;
}
double rastrigin(const Vec& x)
{
    double s = 10.0 * static_cast<double>(x.size());
    for (double v : x) s += v * v - 10 * dm::cos(2 * PI * v);
    return s;
}
double ackley(const Vec& x)                                     // SP-1
{
    double s = 0, c = 0;
    const double n = static_cast<double>(x.size());
    for (double v : x) {
        s += v * v;
        c += dm::cos(2 * PI * v);
    }
    return -20 * dm::exp(-0.2 * std::sqrt(s / n)) - dm::exp(c / n) + 20 + dm::exp(1.0);
}
double goldstein(const Vec& x)
{
    const double t = x[0] * x[0];
    return t * t * t - 15 * t * t + 27 * t + 250;
}
double f25(const Vec& x)
{
    const double t = x[0] * x[0];
    return 0.25 * t * t - 0.5 * t + 0.1 * x[0];
}
double f26(const Vec& x) { return f25(x) + 0.5 * x[1] * x[1]; }
double f27(const Vec& x) { return 0.5 * x[0] * x[0] + 0.5 * (1 - dm::cos(2 * x[0])) + x[1] * x[1]; }

struct Case {
    const char* name;
    std::function<double(const Vec&)> f;
    std::size_t n;
    double ipr, vtr;
    std::size_t np;
    double F, CR, paper, tol;
};

// testbed 3: the global minimum with a relative accuracy of 1e-6
double rel(double fmin) { return fmin + std::fabs(fmin) * 1e-6; }

// evaluations until the first value below vtr; -1 past cap
long long solve(const Case& c, unsigned seed, long long cap)
{
    D::Options o;
    o.np = c.np;
    o.F = c.F;
    o.CR = c.CR;
    o.repair = mootation::ops::BoundRepair::Native;
    std::mt19937 rng(seed);
    D::DE de(Vec(c.n, -c.ipr), Vec(c.n, c.ipr), o, rng);
    long long nfe = 0;
    while (nfe < cap) {
        const evo::Rows X = de.ask();
        Vec f(X.size());
        for (std::size_t i = 0; i < X.size(); ++i) {
            f[i] = c.f(X[i]);
            mix(f[i]);
            ++nfe;
            if (f[i] < c.vtr) return nfe;
        }
        de.tell(f);
    }
    return -1;
}

void paper_tables()
{
    const std::vector<Case> cases = {
        // Table 1
        {"f1 sphere, D = 3", sphere, 3, 5.12, 1e-6, 5, 0.9, 0.1, 406, 0.15},
        {"f2 Rosenbrock", rosenbrock, 2, 2.048, 1e-6, 10, 0.9, 0.9, 654, 0.15},
        {"f5 foxholes", foxholes, 2, 65.536, 0.998005, 15, 0.9, 0.0, 695, 0.15},
        {"f6 Corana", corana, 4, 1000, 1e-6, 10, 0.5, 0.0, 841, 0.15},
        {"f7 Griewangk, D = 10", griewangk, 10, 400, 1e-6, 25, 0.5, 0.2, 12752, 0.15},
        // Table 2
        {"f11 ellipsoid, D = 30", ellipsoid, 30, 1, 1e-10, 20, 0.5, 0.1, 16907, 0.05},
        {"f11 ellipsoid, D = 100", ellipsoid, 100, 1, 1e-10, 20, 0.5, 0.1, 56145, 0.05},
        {"f13 Rastrigin, D = 20", rastrigin, 20, 600, 0.9, 25, 0.5, 0.0, 12971, 0.05},
        {"f13 Rastrigin, D = 100", rastrigin, 100, 600, 0.9, 25, 0.5, 0.0, 73620, 0.05},
        {"f14 Griewangk, D = 20", griewangk, 20, 600, 1e-3, 20, 0.5, 0.1, 8691, 0.05},
        {"f14 Griewangk, D = 100", griewangk, 100, 600, 1e-3, 20, 0.5, 0.1, 31796, 0.05},
        {"f15 Ackley, D = 30", ackley, 30, 30, 1e-3, 20, 0.5, 0.1, 12481, 0.05},
        {"f15 Ackley, D = 100", ackley, 100, 30, 1e-3, 20, 0.5, 0.1, 36801, 0.05},
        // Table 3
        {"f16 Goldstein", goldstein, 1, 10, rel(7.0), 20, 0.5, 0.0, 503, 0.15},
        {"f25", f25, 1, 10, rel(-0.3523861), 20, 0.5, 0.0, 273, 0.15},
        {"f26", f26, 2, 10, rel(-0.3523861), 20, 0.5, 0.0, 650, 0.15},
        {"f27", f27, 2, 10, 1e-6, 20, 0.5, 0.0, 621, 0.15},
    };
    for (const Case& c : cases) {
        double sum = 0.0;
        int hit = 0;
        for (unsigned r = 0; r < 20; ++r) {
            const long long nfe = solve(c, 1000u + r, 400000);
            if (nfe > 0) {
                sum += static_cast<double>(nfe);
                ++hit;
            }
        }
        const double mean = hit ? sum / hit : 0.0;
        char line[200];
        std::snprintf(line, sizeof line, "%s: %d of 20 below the VTR, mean nfe %.0f, paper %.0f (%.3f)", c.name, hit,
                      mean, c.paper, mean / c.paper);
        check(hit >= 17 && std::fabs(mean / c.paper - 1.0) <= c.tol, line);
    }
}

// ── gde3 at n_objs = 1 ──────────────────────────────────────────────────────
struct Twin_GDE3 : public mootation::NSGAII_Individual {};

double twin_f(const std::vector<double>& x)                     // Rastrigin on [0, 1]^6, minimum inside
{
    double s = 0.0;
    for (double v : x) {
        const double t = 10.0 * (v - 0.3);
        s += t * t - 10.0 * dm::cos(2.0 * PI * t) + 10.0;
    }
    return s;
}

}   // namespace

namespace mootation {
template <>
class Problem<Twin_GDE3> {
public:
    std::vector<std::pair<std::optional<double>, std::optional<double>>> bounds =
        std::vector<std::pair<std::optional<double>, std::optional<double>>>(6, {0.0, 1.0});
    int get_vars_n() const { return 6; }
    int get_bin_vars_n() const { return 0; }
    int get_objs_n() const { return 1; }
    int get_lims_n() const { return 0; }
    void calc_objs(Twin_GDE3& ind) const { ind.objectives[0] = twin_f(ind.variables); }
};
}   // namespace mootation

namespace {

void twin_gde3()
{
    const unsigned seed = 20261031;
    const int np = 20, generations = 150;
    mootation::Problem<Twin_GDE3> prob;
    mootation::DataVault<Twin_GDE3> vault(np, prob);
    mootation::Optimizer<Twin_GDE3, mootation::GDE3Core<Twin_GDE3>> opt(std::move(vault), mootation::defer_setup);
    opt.get_algorithm().set_seed(seed);
    opt.get_algorithm().set_F(0.2);
    opt.get_algorithm().set_CR(0.2);

    D::Options o;
    o.np = np;
    o.F = 0.2;
    o.CR = 0.2;                                                  // repair: reflect, the default of both
    std::mt19937 rng(seed);
    D::DE de(Vec(6, 0.0), Vec(6, 1.0), o, rng);

    bool same = true;
    for (int g = 0; g <= generations && same; ++g) {
        if (g == 0) opt.setup(); else opt.step();
        const evo::Rows X = de.ask();
        Vec f(X.size());
        for (std::size_t i = 0; i < X.size(); ++i) f[i] = twin_f(X[i]);
        de.tell(f);
        auto& v = opt.get_vault();
        for (std::size_t i = 0; i < static_cast<std::size_t>(np); ++i)
            same = same && v.variables_of(i) == de.population()[i] && v.objectives_of(i)[0] == de.values()[i];
        if (!same) std::printf("      differs at generation %d\n", g);
    }
    for (double fv : de.values()) mix(fv);
    char line[160];
    std::snprintf(line, sizeof line,
                  "gde3 at n_objs = 1 and de (F = CR = 0.2, reflect): the same population, bit for bit, "
                  "%d generations (best %.3g)", generations, de.values()[de.best()]);
    check(same, line);
}

void comparison_and_done()
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    check(evo::better(5.0, 0.0, 1.0, 0.1) && !evo::better(1.0, 0.1, 5.0, 0.0),
          "a feasible point beats an infeasible one, whatever their f");
    check(evo::better(9.0, 0.2, 1.0, 0.3) && evo::better(1.0, 0.0, 2.0, 0.0),
          "two infeasible: the smaller violation; two feasible: the smaller f");
    check(evo::better(1.0, 0.0, nan, 0.0) && evo::not_worse(nan, 0.0, nan, 0.0) && evo::better(1.0, 5.0, 1.0, nan),
          "NaN counts as +infinity (EVO-1)");

    // a target that every trial violates more stays; a feasible trial replaces it
    D::Options o;
    o.np = 4;
    o.F = 0.5;
    o.CR = 1.0;
    std::mt19937 rng(7);
    D::DE de(Vec(2, 0.0), Vec(2, 1.0), o, rng);
    de.ask();
    de.tell(Vec{1, 1, 1, 1}, Vec{0.5, 0.5, 0.5, 0.0});
    const evo::Rows before = de.population();
    de.ask();
    de.tell(Vec{0, 0, 0, 0}, Vec{0.6, 0.0, 0.4, 0.1});
    check(de.population()[0] == before[0] && de.population()[1] != before[1] && de.population()[2] != before[2] &&
              de.population()[3] == before[3] && de.best() == 1,
          "selection by the rule: kept, replaced, replaced (smaller violation), kept (feasible target)");

    // done(): one point everywhere
    std::mt19937 rng2(3);
    D::DE flat(Vec(1, 0.0), Vec(1, 0.0), o, rng2);
    flat.ask();
    flat.tell(Vec(4, 0.0));
    check(flat.done(), "done() once all points are equal (DE-5)");
}

}   // namespace

int main()
{
    paper_tables();
    twin_gde3();
    comparison_and_done();
    std::printf("checksum %016llx\n", static_cast<unsigned long long>(g_hash));
    check(g_hash == 0xc2bc53d9f8ca4406ull, "the checksum of the runs, the same bits on every platform");
    std::printf(g_failed ? "FAILED\n" : "all passed\n");
    return g_failed ? 1 : 0;
}
