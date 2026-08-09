// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Reference-point generators.
//
// These are checked against closed-form identities rather than golden files:
// the Das-Dennis lattice has an exact cardinality C(H+m-1, m-1) and every point
// lies on the unit simplex. Both are properties the paper states, so a
// violation is a real defect and not a changed convention.
//
// Das & Dennis (1998), "Normal-Boundary Intersection", SIAM J. Optim. 8(3).
// Two-layer variant: Deb & Jain (2014), NSGA-III, §V-A.
// ============================================================================

#include <cmath>
#include <cstddef>
#include <iostream>
#include <set>
#include <vector>

#include "harness.hpp"

namespace {

using namespace mootation;
using namespace mootation::testing;

// C(n, k) computed in long double to survive the larger lattices.
long long binom(long long n, long long k)
{
    if (k < 0 || k > n) return 0;
    k = std::min(k, n - k);
    long double r = 1.0L;
    for (long long i = 1; i <= k; ++i) r = r * static_cast<long double>(n - k + i) / i;
    return static_cast<long long>(r + 0.5L);
}

void test_cardinality()
{
    std::cout << "\n-- lattice cardinality = C(H+m-1, m-1) --\n";
    for (int m = 2; m <= 6; ++m) {
        for (int H = 1; H <= 8; ++H) {
            const auto pts     = das_dennis::generate(m, H);
            const long long ex = binom(H + m - 1, m - 1);
            check(static_cast<long long>(pts.size()) == ex,
                  "generate(m=" + std::to_string(m) + ", H=" + std::to_string(H) +
                      "): got " + std::to_string(pts.size()) + ", expected " +
                      std::to_string(ex));
            // n_vectors() must agree with the generator it advertises.
            check(das_dennis::n_vectors(m, H) == ex,
                  "n_vectors(m=" + std::to_string(m) + ", H=" + std::to_string(H) + ")");
        }
    }
}

void test_on_simplex()
{
    std::cout << "-- every point lies on the unit simplex, coordinates >= 0 --\n";
    for (int m = 2; m <= 6; ++m) {
        const auto pts = das_dennis::generate(m, 6);
        bool sum_ok = true, nonneg_ok = true, dim_ok = true;
        for (const auto& p : pts) {
            if (static_cast<int>(p.size()) != m) { dim_ok = false; continue; }
            double s = 0.0;
            for (double c : p) {
                s += c;
                if (c < -1e-12) nonneg_ok = false;
            }
            if (std::abs(s - 1.0) > 1e-9) sum_ok = false;
        }
        const std::string tag = "m=" + std::to_string(m);
        check(dim_ok,    tag + ": every point has m coordinates");
        check(sum_ok,    tag + ": coordinates sum to 1");
        check(nonneg_ok, tag + ": coordinates are non-negative");
    }
}

void test_distinct()
{
    std::cout << "-- no duplicate points --\n";
    for (int m = 3; m <= 5; ++m) {
        const auto pts = das_dennis::generate(m, 5);
        std::set<std::vector<long long>> seen;
        for (const auto& p : pts) {
            std::vector<long long> key;
            key.reserve(p.size());
            for (double c : p) key.push_back(static_cast<long long>(std::llround(c * 1e9)));
            seen.insert(key);
        }
        check(seen.size() == pts.size(),
              "m=" + std::to_string(m) + ": " + std::to_string(pts.size()) +
                  " points, " + std::to_string(seen.size()) + " distinct");
    }
}

void test_generate_auto()
{
    std::cout << "-- generate_auto returns at least the requested count --\n";
    // The documented contract: the nearest lattice size >= n. Algorithms rely
    // on this when the caller's population size is not a lattice cardinality.
    for (int m = 3; m <= 8; ++m) {
        for (int n : {20, 50, 91, 100, 200}) {
            const auto pts = das_dennis::generate_auto(m, n);
            check(static_cast<int>(pts.size()) >= n,
                  "generate_auto(m=" + std::to_string(m) + ", n=" + std::to_string(n) +
                      "): got " + std::to_string(pts.size()) + ", must be >= n");
        }
    }
}

void test_exact_and_attainable()
{
    std::cout << "-- is_attainable agrees with generate_exact --\n";
    for (int m = 3; m <= 5; ++m) {
        for (long long n = 3; n <= 130; ++n) {
            const bool attainable = das_dennis::is_attainable(m, n);
            bool threw = false;
            std::size_t got = 0;
            try {
                got = das_dennis::generate_exact(m, n).size();
            } catch (const std::exception&) {
                threw = true;
            }
            if (attainable) {
                check(!threw && static_cast<long long>(got) == n,
                      "m=" + std::to_string(m) + ", n=" + std::to_string(n) +
                          ": is_attainable=true but generate_exact " +
                          (threw ? "threw" : "returned " + std::to_string(got)));
            } else {
                check(threw,
                      "m=" + std::to_string(m) + ", n=" + std::to_string(n) +
                          ": is_attainable=false but generate_exact returned " +
                          std::to_string(got));
            }
        }
    }
}

void test_two_layer()
{
    std::cout << "-- two-layer lattice: |boundary| + |inner| --\n";
    // NSGA-III §V-A: for many objectives a boundary layer is combined with a
    // scaled inner layer. The total is the sum of two Das-Dennis lattices.
    for (int m = 5; m <= 8; ++m) {
        const int Hb = 3, Hi = 2;
        const auto pts = das_dennis::generate_two_layer(m, Hb, Hi);
        const long long ex = binom(Hb + m - 1, m - 1) + binom(Hi + m - 1, m - 1);
        check(static_cast<long long>(pts.size()) == ex,
              "generate_two_layer(m=" + std::to_string(m) + ", " +
                  std::to_string(Hb) + ", " + std::to_string(Hi) + "): got " +
                  std::to_string(pts.size()) + ", expected " + std::to_string(ex));

        bool sum_ok = true;
        for (const auto& p : pts) {
            double s = 0.0;
            for (double c : p) s += c;
            if (std::abs(s - 1.0) > 1e-9) sum_ok = false;
        }
        check(sum_ok, "m=" + std::to_string(m) + ": two-layer points sum to 1");
    }
}

void test_degenerate()
{
    std::cout << "-- degenerate inputs do not corrupt memory --\n";
    // These must either return something sane or throw. What they must not do
    // is read out of bounds; the sanitizer job is what actually enforces that,
    // this test only makes sure the paths are executed.
    for (int m : {1, 2}) {
        try {
            const auto pts = das_dennis::generate(m, 1);
            check(!pts.empty(), "generate(m=" + std::to_string(m) + ", H=1) is non-empty");
        } catch (const std::exception&) {
            check(true, "generate(m=" + std::to_string(m) + ", H=1) threw (acceptable)");
        }
    }
}

}   // namespace

int main()
{
    std::cout << "Das-Dennis reference-point generators\n";
    test_cardinality();
    test_on_simplex();
    test_distinct();
    test_generate_auto();
    test_exact_and_attainable();
    test_two_layer();
    test_degenerate();
    return mootation::testing::report("das_dennis");
}
