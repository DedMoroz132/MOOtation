// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/ga.hpp:
//   - twins at n_objs = 1 (task 18, supplement 3, §4): GA with nsga2's knobs
//     against nsga2 (100 generations), ssga with sms_emoa's against sms_emoa
//     (3000 steps), 10 seeds each: the same populations, bit for bit, at
//     every generation or step (ga.hpp, last note); the final bests also pass
//     the statistical form of the check (two-sided Mann-Whitney at 0.05: no
//     difference);
//   - G3 with PCX, Deb, Anand, Joshi 2002 (n = 20, N = 100, mu = 3, lambda =
//     2, start in [-10, -5]^20, no box): every run of 5 reaches 1e-20 on the
//     ellipsoid and on Schwefel's function. With the library's PCX, whose
//     index parent is random (G3-1), it takes about 10 times Table 2's
//     evaluations; the same model with the best parent as index parent gives
//     Table 2 (the TASK18 report, §13);
//   - CHC with BLX-0.5 (scalar/chc.hpp), Eshelman & Schaffer 1993, Table 3,
//     De Jong's F1 and F2 on their bit grids, 50 runs as the paper: every run
//     finds the optimum; the mean trials within 20 % (F1) and 50 % (F2) of
//     the paper's — the paper's children are coded back to bits, ours stay
//     real (CHC-1); the threshold and the divergence;
//   - tournament, truncation and the steady-state replacement by the rule of
//     evo.hpp;
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
#include <mootation/scalar/chc.hpp>
#include <mootation/scalar/ga.hpp>

namespace {

namespace G = mootation::scalar::ga;
namespace evo = mootation::scalar::evo;
namespace dm = mootation::dm;
using G::Vec;

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

double twin_f(const std::vector<double>& x)                     // Rastrigin on [0, 1]^6, minimum inside
{
    double s = 0.0;
    for (double v : x) {
        const double t = 10.0 * (v - 0.3);
        s += t * t - 10.0 * dm::cos(2.0 * PI * t) + 10.0;
    }
    return s;
}

// Two-sided Mann-Whitney U for two samples of 10: a difference at 0.05 when
// min(U1, U2) <= 23 (the exact critical value for n1 = n2 = 10); ties take
// their mean rank.
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
    const double u1 = r1 - 10.0 * 11.0 / 2.0, u2 = 100.0 - u1;
    return std::min(u1, u2) <= 23.0;
}

struct TwinGA : public mootation::NSGAII_Individual {};

}   // namespace

namespace mootation {
template <>
class Problem<TwinGA> {
public:
    std::vector<std::pair<std::optional<double>, std::optional<double>>> bounds =
        std::vector<std::pair<std::optional<double>, std::optional<double>>>(6, {0.0, 1.0});
    int get_vars_n() const { return 6; }
    int get_bin_vars_n() const { return 0; }
    int get_objs_n() const { return 1; }
    int get_lims_n() const { return 0; }
    void calc_objs(TwinGA& ind) const { ind.objectives[0] = twin_f(ind.variables); }
};
}   // namespace mootation

namespace {

// One twin run: the core and the GA step by step; returns {first step where
// the populations differ (steps if never), best of the core, best of the GA}.
template <typename Core, typename Knobs>
std::vector<double> twin_run(Knobs knobs, const G::Options& o, unsigned seed, int np, int steps)
{
    mootation::Problem<TwinGA> prob;
    mootation::DataVault<TwinGA> vault(np, prob);
    mootation::Optimizer<TwinGA, Core> opt(std::move(vault), mootation::defer_setup);
    knobs(opt.get_algorithm());
    opt.get_algorithm().set_seed(seed);
    std::mt19937 rng(seed);
    G::GA ga(Vec(6, 0.0), Vec(6, 1.0), o, rng);
    int split = steps;
    for (int g = 0; g <= steps; ++g) {
        if (g == 0) opt.setup(); else opt.step();
        const evo::Rows X = ga.ask();
        Vec f(X.size());
        for (std::size_t i = 0; i < X.size(); ++i) f[i] = twin_f(X[i]);
        ga.tell(f);
        if (split == steps) {
            auto& v = opt.get_vault();
            bool same = true;
            for (std::size_t i = 0; i < static_cast<std::size_t>(np); ++i)
                same = same && v.variables_of(i) == ga.population()[i] && v.objectives_of(i)[0] == ga.values()[i];
            if (!same) split = g;
        }
    }
    auto& v = opt.get_vault();
    double best = INFINITY;
    for (std::size_t i = 0; i < static_cast<std::size_t>(np); ++i) best = std::min(best, v.objectives_of(i)[0]);
    mix(best);
    mix(ga.values()[ga.best()]);
    return {static_cast<double>(split), best, ga.values()[ga.best()]};
}

void twin_nsga2()
{
    G::Options o;
    o.mu = 40;
    o.lambda = 40;
    o.tournament = 2;
    o.pc = 0.9;
    o.eta_c = 20.0;
    o.eta_m = 20.0;
    Vec a, b;
    int first = 1 << 30;
    for (unsigned s = 0; s < 10; ++s) {
        const Vec r = twin_run<mootation::NSGAIICore<TwinGA>>(
            [](mootation::NSGAIICore<TwinGA>& c) {
                c.set_pc(0.9);
                c.set_eta_crossover(20.0);
                c.set_eta_mutation(20.0);
            },
            o, 900u + s, 40, 100);
        first = std::min(first, static_cast<int>(r[0]));
        a.push_back(r[1]);
        b.push_back(r[2]);
    }
    check(first == 100, "GA with nsga2's knobs: nsga2's populations, bit for bit, every generation, 10 seeds");
    check(!differ(a, b), "GA and nsga2 at n_objs = 1: final bests of 10 seeds, Mann-Whitney 0.05: no difference");
}

void twin_sms_emoa()
{
    G::Options o = G::ssga();
    o.mu = 40;
    o.tournament = 1;
    o.pc = 1.0;
    o.eta_c = 15.0;
    o.eta_m = 20.0;
    Vec a, b;
    int first = 1 << 30;
    for (unsigned s = 0; s < 10; ++s) {
        const Vec r = twin_run<mootation::SMSEMOACore<TwinGA>>(
            [](mootation::SMSEMOACore<TwinGA>& c) {
                c.set_pc(1.0);
                c.set_eta_crossover(15.0);
                c.set_eta_mutation(20.0);
            },
            o, 950u + s, 40, 3000);
        first = std::min(first, static_cast<int>(r[0]));
        a.push_back(r[1]);
        b.push_back(r[2]);
    }
    check(first == 3000, "ssga with sms_emoa's knobs: sms_emoa's populations, bit for bit, every step, 10 seeds");
    check(!differ(a, b), "ssga and sms_emoa at n_objs = 1: final bests of 10 seeds, Mann-Whitney 0.05: no difference");
}

// ── G3 with PCX, Deb et al. 2002 ────────────────────────────────────────────
double felp(const Vec& x)
{
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) s += static_cast<double>(i + 1) * x[i] * x[i];
    return s;
}
double fsch(const Vec& x)
{
    double s = 0.0, c = 0.0;
    for (double v : x) {
        c += v;
        s += c * c;
    }
    return s;
}

long long g3_evaluations(const std::function<double(const Vec&)>& f, unsigned seed, long long cap)
{
    G::G3Options o;
    o.repair = mootation::ops::BoundRepair::Native;
    std::mt19937 rng(seed);
    G::G3 g3(Vec(20, -10.0), Vec(20, -5.0), o, rng);
    long long nfe = 0;
    while (nfe < cap) {
        const evo::Rows X = g3.ask();
        Vec v(X.size());
        for (std::size_t i = 0; i < X.size(); ++i) {
            v[i] = f(X[i]);
            ++nfe;
            if (v[i] <= 1e-20) {
                mix(static_cast<double>(nfe));
                return nfe;
            }
        }
        g3.tell(v);
    }
    return -1;
}

void g3_table_2()
{
    struct Case {
        const char* name;
        std::function<double(const Vec&)> f;
        double median;
    };
    const Case cases[] = {{"ellipsoid", felp, 6624}, {"Schwefel's", fsch, 16326}};
    for (const Case& c : cases) {
        std::vector<long long> n;
        for (unsigned r = 0; r < 5; ++r) n.push_back(g3_evaluations(c.f, 300u + r, 300000));
        std::sort(n.begin(), n.end());
        char line[200];
        std::snprintf(line, sizeof line,
                      "G3-PCX, %s, n = 20: 5 of 5 to 1e-20, evaluations %lld ... %lld (median %lld; Table 2's %.0f)",
                      c.name, n.front(), n.back(), n[2], c.median);
        check(n.front() > 0, line);
    }
}

// ── CHC with BLX-0.5, Eshelman & Schaffer 1993, Table 3 ─────────────────────
// De Jong's F1 (3 parameters, 10 bits, step 0.01) and F2 (2 parameters, 12
// bits, step 0.001), each value read on its grid point as the bit strings
// would code it; "the optimum found" is the grid's minimum, 0.
double dejong_f1(const Vec& x)
{
    double s = 0.0;
    for (double v : x) {
        const double g = -5.12 + std::round((v + 5.12) / 0.01) * 0.01;
        s += g * g;
    }
    return s;
}
double dejong_f2(const Vec& x)
{
    auto g = [](double v) { return -2.048 + std::round((v + 2.048) / 0.001) * 0.001; };
    const double a = g(x[0]), b = g(x[1]);
    return 100.0 * (a * a - b) * (a * a - b) + (1.0 - a) * (1.0 - a);
}

void chc_table_3()
{
    struct Case {
        const char* name;
        std::function<double(const Vec&)> f;
        std::size_t n;
        double lo, hi;
        unsigned bits;
        double paper, tol;
    };
    const Case cases[] = {{"f1 parabola", dejong_f1, 3, -5.12, 5.11, 10, 874, 0.2},
                          {"f2 Rosenbrock's saddle", dejong_f2, 2, -2.048, 2.047, 12, 4893, 0.5}};
    for (const Case& c : cases) {
        mootation::scalar::chc::Options o;
        o.bits = c.bits;
        double sum = 0.0;
        int hit = 0;
        for (unsigned r = 0; r < 50; ++r) {
            std::mt19937 rng(100u + r);
            mootation::scalar::chc::CHC chc(Vec(c.n, c.lo), Vec(c.n, c.hi), o, rng);
            long long nfe = 0;
            bool found = false;
            while (nfe < 50000 && !found) {
                const evo::Rows X = chc.ask();
                Vec f(X.size());
                for (std::size_t i = 0; i < X.size(); ++i) {
                    f[i] = c.f(X[i]);
                    ++nfe;
                    if (f[i] < 1e-12 && !found) {
                        found = true;
                        sum += static_cast<double>(nfe);
                        ++hit;
                    }
                }
                chc.tell(f);
            }
            mix(static_cast<double>(nfe));
        }
        const double mean = hit ? sum / hit : 0.0;
        char line[200];
        std::snprintf(line, sizeof line, "CHC, BLX-0.5, %s: %d of 50 find the optimum, mean trials %.0f, Table 3's %.0f (%.2f)",
                      c.name, hit, mean, c.paper, mean / c.paper);
        check(hit == 50 && std::fabs(mean / c.paper - 1.0) <= c.tol, line);
    }

    // a pairing that cannot mate lowers the threshold; below zero the population diverges, the best kept
    mootation::scalar::chc::Options o;
    o.m = 4;
    o.bits = 4;
    std::mt19937 rng(3);
    mootation::scalar::chc::CHC chc(Vec(2, 0.0), Vec(2, 1.0), o, rng);
    chc.ask();
    chc.tell(Vec{1, 2, 3, 4});
    const Vec best = chc.population()[0];
    long long t0 = chc.threshold(), restarts = 0;
    bool kept = true;
    for (int g = 0; g < 40; ++g) {
        const evo::Rows X = chc.ask();
        Vec f(X.size(), 9.0);                                          // children never accepted
        chc.tell(f);
        restarts = chc.restarts();
        kept = kept && chc.population()[chc.best()] == best;
    }
    check(t0 == 2 && restarts > 0 && kept,
          "CHC: the threshold starts at L/4 and drops while no child is accepted; then divergence, the best kept");
}

void rules()
{
    // truncation keeps the best mu, in order
    G::Options o;
    o.mu = 4;
    o.lambda = 4;
    std::mt19937 rng(1);
    G::GA ga(Vec(2, 0.0), Vec(2, 1.0), o, rng);
    ga.ask();
    ga.tell(Vec{4, 3, 2, 1});
    ga.ask();
    ga.tell(Vec{0.5, 9, 9, 9}, Vec{0.1, 0, 0, 0});
    check(ga.values() == Vec({1, 2, 3, 4}) && ga.violations() == Vec(4, 0.0),
          "(mu + lambda): the feasible first, then by f; the infeasible child of f = 0.5 is out");

    G::Options s = G::ssga();
    s.mu = 3;
    std::mt19937 rng2(2);
    G::GA st(Vec(1, 0.0), Vec(1, 1.0), s, rng2);
    st.ask();
    st.tell(Vec{1, 5, 3});
    st.ask();
    st.tell(Vec{2});
    check(st.values() == Vec({1, 2, 3}), "steady state: the child takes the place of the worst");
    st.ask();
    st.tell(Vec{7});
    check(st.values() == Vec({1, 2, 3}), "steady state: a child worse than all is dropped");
}

}   // namespace

int main()
{
    twin_nsga2();
    twin_sms_emoa();
    g3_table_2();
    chc_table_3();
    rules();
    std::printf("checksum %016llx\n", static_cast<unsigned long long>(g_hash));
    check(g_hash == 0xf020aa5fdc800c0bull, "the checksum of the runs, the same bits on every platform");
    std::printf(g_failed ? "FAILED\n" : "all passed\n");
    return g_failed ? 1 : 0;
}
