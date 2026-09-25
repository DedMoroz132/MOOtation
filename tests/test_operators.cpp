// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Variation operators.
//
// Checked properties are the ones the papers guarantee, not implementation
// details: children stay inside the bounds, unbounded variables are rejected
// rather than silently clamped, probability-zero means "no change", and the
// DE repair modes do what their names say.
//
// SBX:  Deb & Agrawal (1995), Complex Systems 9(2).
// PM:   NSGA-II reference implementation (mutation.c).
// DE:   Storn & Price (1997), rand/1/bin.
// SPX, REX, UNDX, PCX: the sources in operators/multi_parent.hpp.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness.hpp"

namespace {

using namespace mootation;
using namespace mootation::testing;

using Bounds = std::vector<std::pair<std::optional<double>, std::optional<double>>>;

Bounds unit_bounds(int n) { return Bounds(n, {0.0, 1.0}); }

bool within(const std::vector<double>& x, const Bounds& b)
{
    for (std::size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) return false;
        if (x[j] < *b[j].first - 1e-12) return false;
        if (x[j] > *b[j].second + 1e-12) return false;
    }
    return true;
}

void test_sbx_bounds()
{
    std::cout << "\n-- SBX keeps children inside the bounds --\n";
    std::mt19937 rng(12345);
    const int  n = 10;
    const auto b = unit_bounds(n);
    std::uniform_real_distribution<double> u(0.0, 1.0);

    bool all_in = true;
    for (int trial = 0; trial < 2000; ++trial) {
        std::vector<double> p1(n), p2(n), c1, c2;
        for (int j = 0; j < n; ++j) { p1[j] = u(rng); p2[j] = u(rng); }
        ops::sbx(p1, p2, c1, c2, b, 20.0, 1.0, rng);
        if (!within(c1, b) || !within(c2, b)) { all_in = false; break; }
    }
    check(all_in, "2000 SBX crossovers stayed within [0,1]");
}

void test_sbx_pc_zero()
{
    std::cout << "-- SBX with pc = 0 copies the parents --\n";
    std::mt19937 rng(7);
    const int  n = 8;
    const auto b = unit_bounds(n);
    std::uniform_real_distribution<double> u(0.0, 1.0);

    std::vector<double> p1(n), p2(n), c1, c2;
    for (int j = 0; j < n; ++j) { p1[j] = u(rng); p2[j] = u(rng); }
    ops::sbx(p1, p2, c1, c2, b, 20.0, 0.0, rng);
    check(c1 == p1 && c2 == p2, "pc=0 leaves both children equal to the parents");
}

void test_sbx_identical_parents()
{
    std::cout << "-- SBX on identical parents produces identical children --\n";
    // dy = 0 would divide by zero; the implementation skips such variables.
    std::mt19937 rng(99);
    const int  n = 6;
    const auto b = unit_bounds(n);
    std::vector<double> p(n, 0.42), c1, c2;
    ops::sbx(p, p, c1, c2, b, 20.0, 1.0, rng);
    check(c1 == p && c2 == p, "identical parents give identical children (no NaN)");
}

void test_sbx_requires_bounds()
{
    std::cout << "-- SBX rejects an unbounded variable instead of guessing --\n";
    // The 2026-06 audit removed a silent [0,1] default that produced a silent
    // clamp on problems with other domains. Throwing is the contract now.
    std::mt19937 rng(1);
    Bounds b(3, {0.0, 1.0});
    b[1].second = std::nullopt;          // no upper bound on variable 1

    // Force every variable to participate. With the canonical per-variable
    // probability of 0.5, variable 1 is only *visited* about half the time, so
    // the throw is probabilistic and the test would be flaky.
    //
    // Worth recording as a design wart rather than a test inconvenience: with
    // the default settings a missing bound surfaces nondeterministically, which
    // is an unpleasant way for a user to meet a configuration error.
    const double saved = ops::sbx_var_prob();
    ops::sbx_var_prob() = 1.0;

    std::vector<double> p1{0.1, 0.2, 0.3}, p2{0.4, 0.5, 0.6}, c1, c2;
    bool threw = false;
    try {
        ops::sbx(p1, p2, c1, c2, b, 20.0, 1.0, rng);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ops::sbx_var_prob() = saved;
    check(threw, "SBX throws std::invalid_argument on a missing bound");
}

void test_pm_bounds()
{
    std::cout << "-- polynomial mutation keeps values inside the bounds --\n";
    std::mt19937 rng(2024);
    const int  n = 12;
    const auto b = unit_bounds(n);
    std::uniform_real_distribution<double> u(0.0, 1.0);

    bool all_in = true;
    for (int trial = 0; trial < 2000; ++trial) {
        std::vector<double> x(n);
        for (int j = 0; j < n; ++j) x[j] = u(rng);
        ops::polynomial_mutation(x, b, 20.0, 1.0, rng);   // pm = 1: mutate everything
        if (!within(x, b)) { all_in = false; break; }
    }
    check(all_in, "2000 polynomial mutations stayed within [0,1]");
}

void test_pm_zero_probability()
{
    std::cout << "-- polynomial mutation with pm = 0 changes nothing --\n";
    std::mt19937 rng(5);
    const int  n = 10;
    const auto b = unit_bounds(n);
    std::uniform_real_distribution<double> u(0.0, 1.0);

    std::vector<double> x(n);
    for (int j = 0; j < n; ++j) x[j] = u(rng);
    const auto before = x;
    ops::polynomial_mutation(x, b, 20.0, 0.0, rng);
    check(x == before, "pm=0 leaves the vector untouched");
}

void test_pm_at_bounds()
{
    std::cout << "-- polynomial mutation is finite at the bounds --\n";
    // x exactly at a bound makes one of delta_1/delta_2 zero; a naive
    // implementation produces NaN here.
    std::mt19937 rng(31337);
    const int  n = 5;
    const auto b = unit_bounds(n);

    bool finite = true;
    for (double v : {0.0, 1.0}) {
        for (int trial = 0; trial < 500; ++trial) {
            std::vector<double> x(n, v);
            ops::polynomial_mutation(x, b, 20.0, 1.0, rng);
            for (double c : x)
                if (!std::isfinite(c)) finite = false;
            if (!within(x, b)) finite = false;
        }
    }
    check(finite, "mutation at x=0 and x=1 stays finite and in range");
}

void test_de_bounds_and_repair()
{
    std::cout << "-- DE rand/1/bin: both repair modes stay in range --\n";
    std::mt19937 rng(808);
    const int  n = 10;
    const auto b = unit_bounds(n);
    std::uniform_real_distribution<double> u(0.0, 1.0);

    for (auto repair : {ops::DERepair::Clip, ops::DERepair::RandomReset}) {
        bool all_in = true;
        for (int trial = 0; trial < 1000; ++trial) {
            std::vector<double> xa(n), xb(n), xc(n), xi(n), y;
            for (int j = 0; j < n; ++j) {
                xa[j] = u(rng); xb[j] = u(rng); xc[j] = u(rng); xi[j] = u(rng);
            }
            // F = 2.0 deliberately overshoots the box so repair has to act.
            ops::de_rand_1_bin(xa, xb, xc, xi, y, b, 2.0, 0.9, repair, rng);
            if (y.size() != static_cast<std::size_t>(n) || !within(y, b)) {
                all_in = false;
                break;
            }
        }
        check(all_in, std::string("DE with ") +
                          (repair == ops::DERepair::Clip ? "Clip" : "RandomReset") +
                          " repair stayed within [0,1] under F=2.0");
    }
}

void test_de_zero_vars()
{
    std::cout << "-- DE on a zero-length genome does not invoke UB --\n";
    // OP-1 from the 2026-07-08 audit: uniform_int_distribution(0, -1) is UB.
    std::mt19937 rng(3);
    Bounds b;
    std::vector<double> e, y{1.0, 2.0};
    ops::de_rand_1_bin(e, e, e, e, y, b, 0.5, 0.9, ops::DERepair::Clip, rng);
    check(y.empty(), "zero real variables yields an empty child, no UB");
}

void test_bit_flip()
{
    std::cout << "-- binary operators produce valid 0/1 genomes --\n";
    std::mt19937 rng(77);
    const int nb = 16;

    std::vector<int> p1(nb), p2(nb), c1, c2;
    std::uniform_int_distribution<int> coin(0, 1);
    for (int j = 0; j < nb; ++j) { p1[j] = coin(rng); p2[j] = coin(rng); }

    ops::binary_crossover(p1, p2, c1, c2, rng);
    check(c1.size() == static_cast<std::size_t>(nb) &&
              c2.size() == static_cast<std::size_t>(nb),
          "binary crossover preserves the genome length");

    ops::bit_flip_mutation(c1, nb, rng);
    bool valid = true;
    for (int v : c1)
        if (v != 0 && v != 1) valid = false;
    check(valid, "bit-flip mutation leaves every gene in {0,1}");
}

void test_determinism()
{
    std::cout << "-- identical seeds give identical operator output --\n";
    const int  n = 10;
    const auto b = unit_bounds(n);

    auto run = [&](unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        std::vector<double> out;
        for (int trial = 0; trial < 50; ++trial) {
            std::vector<double> p1(n), p2(n), c1, c2;
            for (int j = 0; j < n; ++j) { p1[j] = u(rng); p2[j] = u(rng); }
            ops::sbx(p1, p2, c1, c2, b, 20.0, 0.9, rng);
            ops::polynomial_mutation(c1, b, 20.0, 1.0 / n, rng);
            out.insert(out.end(), c1.begin(), c1.end());
        }
        return out;
    };

    check(run(4242) == run(4242), "same seed reproduces the operator stream exactly");
    check(run(4242) != run(4243), "different seeds produce different streams");
}

// ── Multi-parent crossovers (operators/multi_parent.hpp) ────────────────────
// The properties their sources give: SPX with n + 1 parents and the factor
// √(n + 2), and REX with σ² = 1/(μ − 1), hand the parents' mean g and
// covariance on (Higuchi, Tsutsui & Yamamura 2000, Theorem 3; Tanabe &
// Ishibuchi 2019, §2.2) — SPX the covariance (1/μ)·S, REX (1/(μ − 1))·S,
// S = Σ (x_j − g)(x_j − g)ᵀ. UNDX's pair is symmetric about the parents'
// midpoint with spread α·d1 along their line and β·d2/√n in every direction
// across it (Ono, Kita & Kobayashi 1999, Eq. 1-2); PCX's child spreads
// σ_ζ·|d| along d and σ_η·D̄ across it (Deb, Anand & Joshi, Evol. Comput.
// 2002, Eq. 2).

using Vec = std::vector<double>;

double dot(const Vec& a, const Vec& b)
{
    double s = 0.0;
    for (std::size_t j = 0; j < a.size(); ++j) s += a[j] * b[j];
    return s;
}

void test_spx_rex_statistics()
{
    std::cout << "-- SPX and REX hand the parents' mean and covariance on --\n";
    const std::size_t n = 3, mu = n + 1;
    const ops::ParentSet P = {{0.1, 0.2, 0.7}, {0.8, 0.1, 0.3}, {0.4, 0.9, 0.5}, {0.6, 0.5, 0.1}};
    const Vec g = ops::mp_detail::centre(P);
    Vec S(n * n, 0.0);
    for (const auto& x : P)
        for (std::size_t a = 0; a < n; ++a)
            for (std::size_t b = 0; b < n; ++b) S[a * n + b] += (x[a] - g[a]) * (x[b] - g[b]);

    // Mean and covariance about g of 200 000 children, against scale·S.
    auto moments = [&](auto child, double scale, const std::string& name) {
        std::mt19937 rng(2026);
        const int draws = 200000;
        Vec mean(n, 0.0), C(n * n, 0.0);
        for (int t = 0; t < draws; ++t) {
            const Vec y = child(rng);
            for (std::size_t a = 0; a < n; ++a) {
                mean[a] += y[a] / draws;
                for (std::size_t b = 0; b < n; ++b)
                    C[a * n + b] += (y[a] - g[a]) * (y[b] - g[b]) / draws;
            }
        }
        double mean_err = 0.0, cov_err = 0.0, largest = 0.0;
        for (std::size_t a = 0; a < n; ++a) {
            mean_err = std::max(mean_err, std::abs(mean[a] - g[a]));
            for (std::size_t b = 0; b < n; ++b) {
                cov_err = std::max(cov_err, std::abs(C[a * n + b] - scale * S[a * n + b]));
                largest = std::max(largest, std::abs(scale * S[a * n + b]));
            }
        }
        check(mean_err < 0.005, name + ": the children's mean is the parents' centre");
        check(cov_err < 0.03 * largest, name + ": the children's covariance is the parents'");
    };
    moments([&](std::mt19937& r) { return ops::spx_child(P, std::sqrt(n + 2.0), r); },
            1.0 / static_cast<double>(mu), "SPX, e = sqrt(n + 2)");
    moments([&](std::mt19937& r) { return ops::rex_child(P, r); },
            1.0 / static_cast<double>(mu - 1), "REX, sigma^2 = 1/(mu - 1)");
}

void test_undx_pair()
{
    std::cout << "-- UNDX: a symmetric pair, alpha*d1 along the parents, beta*d2/sqrt(n) across --\n";
    const std::size_t n = 4;
    const Vec x1{0.2, 0.3, 0.4, 0.5}, x2{0.6, 0.1, 0.4, 0.7}, x3{0.3, 0.8, 0.2, 0.4};
    Vec m(n), e1(n), v(n);
    for (std::size_t j = 0; j < n; ++j) { m[j] = 0.5 * (x1[j] + x2[j]); e1[j] = x2[j] - x1[j]; }
    const double d1 = std::sqrt(dot(e1, e1));
    for (double& c : e1) c /= d1;
    for (std::size_t j = 0; j < n; ++j) v[j] = x3[j] - x1[j];
    const double along3 = dot(v, e1);
    for (std::size_t j = 0; j < n; ++j) v[j] -= along3 * e1[j];
    const double d2 = std::sqrt(dot(v, v));

    std::mt19937 rng(7);
    const int draws = 100000;
    bool symmetric = true;
    double along = 0.0, across = 0.0;
    for (int t = 0; t < draws; ++t) {
        const auto pr = ops::undx_pair(x1, x2, x3, 0.5, 0.35, rng);
        Vec s(n);
        for (std::size_t j = 0; j < n; ++j) {
            if (std::abs(pr.first[j] + pr.second[j] - 2.0 * m[j]) > 1e-12) symmetric = false;
            s[j] = pr.first[j] - m[j];
        }
        const double a = dot(s, e1);
        along  += a * a / draws;
        across += (dot(s, s) - a * a) / draws;
    }
    check(symmetric, "c1 + c2 = 2m in every pair");
    const double want_along = 0.5 * d1;
    const double want_across = (n - 1.0) * std::pow(0.35 * d2 / std::sqrt(static_cast<double>(n)), 2);
    check_close(std::sqrt(along), want_along, 0.02 * want_along, "spread along the parents is alpha*d1");
    check_close(across, want_across, 0.02 * want_across,
                "spread across is (n - 1)(beta*d2/sqrt(n))^2 in total");
}

void test_pcx_child()
{
    std::cout << "-- PCX: sigma_zeta*|d| along d, sigma_eta*Dbar across --\n";
    const std::size_t n = 4;
    const ops::ParentSet P = {{0.2, 0.3, 0.4, 0.5}, {0.6, 0.1, 0.4, 0.7}, {0.3, 0.8, 0.2, 0.4}};
    const Vec g = ops::mp_detail::centre(P);

    std::mt19937 rng(11);
    const int draws = 100000;
    double along_ratio = 0.0, across_ratio = 0.0;
    for (int t = 0; t < draws; ++t) {
        const auto [y, p] = ops::pcx_child(P, 0.1, 0.1, rng);
        Vec d(n), s(n);
        for (std::size_t j = 0; j < n; ++j) { d[j] = P[p][j] - g[j]; s[j] = y[j] - P[p][j]; }
        const double dn2 = dot(d, d);
        double Dbar = 0.0;
        for (std::size_t i = 0; i < P.size(); ++i) {
            if (i == p) continue;
            Vec r(n);
            for (std::size_t j = 0; j < n; ++j) r[j] = P[i][j] - g[j];
            const double k = dot(r, d) / dn2;
            for (std::size_t j = 0; j < n; ++j) r[j] -= k * d[j];
            Dbar += std::sqrt(dot(r, r)) / static_cast<double>(P.size() - 1);
        }
        const double a = dot(s, d) / std::sqrt(dn2);            // w_zeta*|d|
        along_ratio  += a * a / dn2 / draws;                     // -> sigma_zeta^2
        across_ratio += (dot(s, s) - a * a) / (Dbar * Dbar) / draws;   // -> (n-1) sigma_eta^2
    }
    check_close(along_ratio, 0.01, 0.0003, "the step along d has variance sigma_zeta^2 |d|^2");
    check_close(across_ratio, (n - 1.0) * 0.01, 0.0009,
                "the step across has variance sigma_eta^2 Dbar^2 per direction");
}

void test_multi_parent_host_entry()
{
    std::cout << "-- apply_any: the host's pair, further parents drawn only when used --\n";
    const std::size_t n = 5;
    const auto b = unit_bounds(static_cast<int>(n));
    std::mt19937 pool_rng(3);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<Vec> pool(40, Vec(n));
    for (auto& x : pool) for (double& c : x) c = u(pool_rng);
    const Vec p1 = pool[0], p2 = pool[1];

    int drawn = 0;
    std::mt19937 draw_rng(5);
    std::uniform_int_distribution<std::size_t> any(0, pool.size() - 1);
    auto draw = [&] { ++drawn; return pool[any(draw_rng)]; };

    ops::CrossoverSpec sbx_spec;                                  // the default
    Vec a1, a2, b1, b2;
    std::mt19937 r1(9), r2(9);
    sbx_spec.apply(p1, p2, a1, a2, b, 20.0, 0.9, r1);
    sbx_spec.apply_any(p1, p2, draw, b1, b2, b, 20.0, 0.9, r2);
    check(a1 == b1 && a2 == b2 && drawn == 0,
          "SBX through apply_any: the same children, no further parent drawn");

    ops::CrossoverSpec spx;
    spx.kind = ops::Crossover::SPX;
    bool refused = false;
    try { spx.apply(p1, p2, a1, a2, b, 20.0, 1.0, r1); }
    catch (const std::invalid_argument&) { refused = true; }
    check(refused, "apply() refuses a kind that takes more than two parents");

    spx.apply_any(p1, p2, draw, a1, a2, b, 20.0, 0.0, r1);
    check(drawn == 0 && a1 == p1 && a2 == p2, "pc = 0: the pair is copied and nothing is drawn");

    bool inside = true;
    for (auto kind : {ops::Crossover::SPX, ops::Crossover::REX, ops::Crossover::UNDX,
                      ops::Crossover::PCX}) {
        ops::CrossoverSpec c;
        c.kind = kind;
        for (int t = 0; t < 500; ++t) {
            drawn = 0;
            c.apply_any(p1, p2, draw, a1, a2, b, 20.0, 1.0, r1);
            if (!within(a1, b) || !within(a2, b)) inside = false;
        }
        check(drawn >= c.parents(static_cast<int>(n)) - 2,
              std::string(ops::crossover_name(kind)) + " draws its parents beyond the pair");
    }
    check(inside, "the four multi-parent kinds repair their children into the box");

    // MP-6: a pool of two distinct members cannot give SPX n + 1 different
    // parents; the redraws run out and the application still completes.
    std::vector<Vec> two = {p1, p2};
    auto draw_two = [&] { return two[any(draw_rng) % 2]; };
    bool completed = true;
    try { spx.apply_any(p1, p2, draw_two, a1, a2, b, 20.0, 1.0, r1); }
    catch (...) { completed = false; }
    check(completed && within(a1, b), "a converged pool still crosses, repeating a parent");
}

}   // namespace

int main()
{
    std::cout << "Variation operators\n";
    test_sbx_bounds();
    test_sbx_pc_zero();
    test_sbx_identical_parents();
    test_sbx_requires_bounds();
    test_pm_bounds();
    test_pm_zero_probability();
    test_pm_at_bounds();
    test_de_bounds_and_repair();
    test_de_zero_vars();
    test_bit_flip();
    test_determinism();
    test_spx_rex_statistics();
    test_undx_pair();
    test_pcx_child();
    test_multi_parent_host_entry();
    return mootation::testing::report("operators");
}
