// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/pso.hpp:
//   - chi of (4.17) at phi = 4.1, kappa = 1: 0.7298;
//   - Clerc & Kennedy 2002, Table V, the Type 1'' column (20 particles, 2000
//     iterations, 20 runs, the initial ranges of Table IV as the only box):
//     the zeros of f1, f2, f4 below the table's six decimals; f5 at 0.998004;
//     Schaffer's f6, Griewank, Ackley, Rastrigin and Rosenbrock within a
//     factor 2.5 of the table's means (they come out at 2.0, 1.00, 0.65,
//     1.04 and 0.76 of them);
//   - the twin at n_objs = 1 (task 18, supplement 3, §4): smpso with χ by
//     modulus, w = 1, C1 = C2 = 2.05, r per variable, no turbulence, no
//     velocity bound, the velocity kept at the bound, against pso with the
//     global best, v = 0 at the start and clip. They differ in the order of
//     draws and in p_i on ties (smpso keeps the new point, pso the old), so
//     the check is the statistical one: 10 seeds, final bests, two-sided
//     Mann-Whitney at 0.05, no difference;
//   - a checksum of the runs, the same bits on every platform.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <mootation/mootation.hpp>
#include <mootation/scalar/pso.hpp>

namespace {

namespace P = mootation::scalar::pso;
namespace evo = mootation::scalar::evo;
namespace dm = mootation::dm;
using P::Vec;

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

// Two-sided Mann-Whitney U for two samples of 10: a difference at 0.05 when
// min(U1, U2) <= 23; ties take their mean rank.
bool differ(const Vec& a, const Vec& b)
{
    std::vector<std::pair<double, int>> all;
    for (double v : a) all.push_back({v, 0});
    for (double v : b) all.push_back({v, 1});
    std::sort(all.begin(), all.end());
    double r1 = 0.0;
    for (std::size_t i = 0; i < all.size();) {
        std::size_t j = i;
        while (j < all.size() && all[j].first == all[i].first) ++j;
        const double rank = (static_cast<double>(i + 1) + static_cast<double>(j)) / 2.0;
        for (std::size_t k = i; k < j; ++k)
            if (all[k].second == 0) r1 += rank;
        i = j;
    }
    const double u1 = r1 - 55.0, u2 = 100.0 - u1;
    return std::min(u1, u2) <= 23.0;
}

// ── Table III of Clerc & Kennedy ────────────────────────────────────────────
double foxholes(const Vec& x)
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

void table_v()
{
    check(std::fabs(P::constriction(4.1, 1.0) - 0.729843788) < 1e-9, "chi (4.17) at phi = 4.1, kappa = 1: 0.729844");
    struct Case {
        const char* name;
        std::function<double(const Vec&)> f;
        std::size_t n;
        double range, paper;
    };
    const std::vector<Case> cases = {
        {"f1", [](const Vec& x) { double s = 0; for (double v : x) s += v * v; return s; }, 30, 20, 0.0},
        {"f2", [](const Vec& x) { const double a = x[0] * x[0] - x[1]; return 100 * a * a + (1 - x[0]) * (1 - x[0]); }, 2, 50, 0.0},
        {"f4", [](const Vec& x) { double s = 0; for (std::size_t i = 0; i < x.size(); ++i) { const double t = x[i] * x[i]; s += static_cast<double>(i + 1) * t * t; } return s; }, 30, 20, 0.0},
        {"f5", foxholes, 2, 50, 0.998004},
        {"Schaffer's f6", [](const Vec& x) { const double r = x[0] * x[0] + x[1] * x[1]; const double s = dm::sin(std::sqrt(r)), q = 1 + 0.001 * r; return 0.5 + (s * s - 0.5) / (q * q); }, 2, 100, 0.001459},
        {"Griewank", [](const Vec& x) { double s = 0, p = 1; for (std::size_t i = 0; i < x.size(); ++i) { const double t = x[i] - 100; s += t * t; p *= dm::cos(t / std::sqrt(static_cast<double>(i + 1))); } return s / 4000 - p + 1; }, 30, 300, 0.003944},
        {"Ackley", [](const Vec& x) { double s = 0, c = 0; const double n = static_cast<double>(x.size()); for (double v : x) { s += v * v; c += dm::cos(2 * PI * v); } return -20 * dm::exp(-0.2 * std::sqrt(s / n)) - dm::exp(c / n) + 20 + dm::exp(1.0); }, 30, 32, 0.204988},
        {"Rastrigin", [](const Vec& x) { double s = 0; for (double v : x) s += v * v - 10 * dm::cos(2 * PI * v) + 10; return s; }, 30, 5.12, 82.95618},
        {"Rosenbrock", [](const Vec& x) { double s = 0; for (std::size_t i = 0; i + 1 < x.size(); ++i) { const double a = x[i + 1] - x[i] * x[i]; s += 100 * a * a + (x[i] - 1) * (x[i] - 1); } return s; }, 30, 10, 50.193877},
    };
    for (const Case& c : cases) {
        double sum = 0.0;
        for (unsigned r = 0; r < 20; ++r) {
            P::Options o;
            o.repair = mootation::ops::BoundRepair::Native;
            std::mt19937 rng(200u + r);
            P::PSO swarm(Vec(c.n, -c.range), Vec(c.n, c.range), o, rng);
            for (int it = 0; it <= 2000; ++it) {
                const evo::Rows X = swarm.ask();
                Vec f(X.size());
                for (std::size_t i = 0; i < X.size(); ++i) f[i] = c.f(X[i]);
                swarm.tell(f);
            }
            sum += swarm.values()[swarm.best()];
            mix(swarm.values()[swarm.best()]);
        }
        const double mean = sum / 20.0;
        char line[200];
        std::snprintf(line, sizeof line, "Type 1'', %s: mean best %.6f, Table V %.6f", c.name, mean, c.paper);
        const bool ok = c.paper == 0.0      ? mean < 5e-7
                        : c.paper < 1.0 && c.paper > 0.9 ? std::fabs(mean - c.paper) < 1e-6
                                            : std::fabs(std::log10(mean / c.paper)) <= std::log10(2.5);
        check(ok, line);
    }
}

// ── the smpso twin ──────────────────────────────────────────────────────────
struct TwinPSO : public mootation::SMPSO_Individual {};

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
class Problem<TwinPSO> {
public:
    std::vector<std::pair<std::optional<double>, std::optional<double>>> bounds =
        std::vector<std::pair<std::optional<double>, std::optional<double>>>(6, {0.0, 1.0});
    int get_vars_n() const { return 6; }
    int get_bin_vars_n() const { return 0; }
    int get_objs_n() const { return 1; }
    int get_lims_n() const { return 0; }
    void calc_objs(TwinPSO& ind) const { ind.objectives[0] = twin_f(ind.variables); }
};
}   // namespace mootation

namespace {

void twin_smpso()
{
    Vec a, b;
    for (unsigned s = 0; s < 10; ++s) {
        mootation::Problem<TwinPSO> prob;
        mootation::DataVault<TwinPSO> vault(20, prob);
        mootation::Optimizer<TwinPSO, mootation::SMPSOCore<TwinPSO>> opt(std::move(vault), mootation::defer_setup);
        auto& c = opt.get_algorithm();
        c.set_seed(600u + s);
        c.set_word("chi_mode", "abs");
        c.set_param("w_min", 1.0);
        c.set_param("w_max", 1.0);
        c.set_param("c_min", 2.05);
        c.set_param("c_max", 2.05);
        c.set_param("vmax_frac", 0.0);
        c.set_param("vel_on_bound_low", 1.0);
        c.set_param("vel_on_bound_up", 1.0);
        c.set_word("r_per_dim", "yes");
        c.set_word("turbulence", "off");
        opt.optimize(200);
        auto& v = opt.get_vault();
        double best = INFINITY;
        for (std::size_t i = 0; i < v.active_n(); ++i) best = std::min(best, v.objectives_of(i)[0]);
        a.push_back(best);

        P::Options o;
        o.ring = false;
        o.v0_zero = true;
        o.repair = mootation::ops::BoundRepair::Clip;
        std::mt19937 rng(600u + s);
        P::PSO swarm(Vec(6, 0.0), Vec(6, 1.0), o, rng);
        for (int it = 0; it <= 200; ++it) {
            const evo::Rows X = swarm.ask();
            Vec f(X.size());
            for (std::size_t i = 0; i < X.size(); ++i) f[i] = twin_f(X[i]);
            swarm.tell(f);
        }
        b.push_back(swarm.values()[swarm.best()]);
        mix(a.back());
        mix(b.back());
    }
    double ma = 0.0, mb = 0.0;
    for (std::size_t i = 0; i < 10; ++i) {
        ma += a[i] / 10.0;
        mb += b[i] / 10.0;
    }
    char line[200];
    std::snprintf(line, sizeof line,
                  "pso and smpso with its knobs at n_objs = 1: 10 seeds, mean bests %.3g and %.3g, Mann-Whitney 0.05: no difference",
                  mb, ma);
    check(!differ(a, b), line);
}

void rule()
{
    P::Options o;
    o.n = 3;
    std::mt19937 rng(4);
    P::PSO swarm(Vec(1, 0.0), Vec(1, 1.0), o, rng);
    swarm.ask();
    swarm.tell(Vec{5, 1, 3}, Vec{0, 0.2, 0});
    check(swarm.best() == 2, "p_g by the rule: the feasible 3 before the infeasible 1");
}

}   // namespace

int main()
{
    table_v();
    twin_smpso();
    rule();
    std::printf("checksum %016llx\n", static_cast<unsigned long long>(g_hash));
    check(g_hash == 0x30d807fc47120afaull, "the checksum of the runs, the same bits on every platform");
    std::printf(g_failed ? "FAILED\n" : "all passed\n");
    return g_failed ? 1 : 0;
}
