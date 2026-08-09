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
// ============================================================================

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
    return mootation::testing::report("operators");
}
