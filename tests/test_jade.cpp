// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/jade.hpp:
//   - Zhang & Sanderson 2009, Table IV (D = 30, NP = 100, p = 0.05, c = 0.1):
//     the mean best error at the paper's generation, 5 runs a case where the
//     paper has 50, within 1.5 decades of the paper's mean (the means of
//     values near 1e-60 are set by the worst run) — with the archive on f1,
//     f6, f9, f10, f12, f13 and without it on f1, f10;
//   - the archive never holds more than NP points; done();
//   - the constrained comparison of evo.hpp decides (5);
//   - a checksum of the runs, the same bits on every platform.
// ============================================================================

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <mootation/scalar/jade.hpp>

#include "yao1999.hpp"

namespace {

namespace J = mootation::scalar::jade;
using J::Vec;

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

// the best value after `generations` generations (the initial one not counted)
double run(int id, std::size_t n, bool archive, int generations, unsigned seed, std::size_t* largest_a = nullptr)
{
    J::Options o;
    o.archive = archive;
    std::mt19937 rng(seed);
    J::JADE de(Vec(n, -yao1999::range(id)), Vec(n, yao1999::range(id)), o, rng);
    double best = INFINITY;
    for (int g = 0; g <= generations; ++g) {
        const mootation::scalar::evo::Rows X = de.ask();
        Vec f(X.size());
        for (std::size_t i = 0; i < X.size(); ++i) {
            f[i] = yao1999::f(id, X[i]);
            best = std::min(best, f[i]);
        }
        de.tell(f);
        if (largest_a) *largest_a = std::max(*largest_a, de.archive().size());
    }
    mix(best);
    mix(de.mu_f());
    mix(de.mu_cr());
    return best;
}

void table_iv()
{
    struct Case {
        int id, generations;
        bool archive;
        double paper;
    };
    const Case cases[] = {
        {1, 1500, true, 1.3e-54}, {6, 100, true, 5.6e+00},  {9, 1000, true, 1.4e-04}, {10, 500, true, 3.0e-09},
        {12, 500, true, 3.8e-16}, {13, 500, true, 1.2e-15}, {1, 1500, false, 1.8e-60}, {10, 500, false, 8.2e-10},
    };
    for (const Case& c : cases) {
        double sum = 0.0;
        std::size_t largest = 0;
        for (unsigned r = 0; r < 5; ++r) sum += run(c.id, 30, c.archive, c.generations, 500u + r, &largest);
        const double mean = sum / 5.0;
        char line[200];
        std::snprintf(line, sizeof line, "f%d, D = 30, %s archive, generation %d: mean %.2e, paper %.1e", c.id,
                      c.archive ? "with" : "without", c.generations, mean, c.paper);
        check(mean > 0.0 && std::fabs(std::log10(mean / c.paper)) <= 1.5, line);
        if (c.archive) check(largest <= 100, "  the archive holds at most NP = 100 points");
        else check(largest == 0, "  without the archive, A stays empty");
    }
}

void rule_and_done()
{
    // a trial must be strictly better (5): equal values keep the parent
    J::Options o;
    o.np = 6;
    std::mt19937 rng(11);
    J::JADE de(Vec(3, 0.0), Vec(3, 1.0), o, rng);
    de.ask();
    de.tell(Vec(6, 1.0));
    const auto before = de.population();
    de.ask();
    de.tell(Vec(6, 1.0));
    check(de.population() == before && de.archive().empty() && de.mu_f() == 0.5 && de.mu_cr() == 0.5,
          "equal values: no parent replaced, A empty, mu_F and mu_CR kept (JADE-3)");
    de.ask();
    de.tell(Vec{1, 1, 1, 1, 1, 1}, Vec{0, 0, 0, 0, 0, 0.0});
    de.ask();
    de.tell(Vec{5, 0, 5, 5, 5, 5}, Vec{0, 0, 0.1, 0, 0, 0});
    check(de.population()[1] != before[1] && de.population()[2] == before[2] && de.archive().size() == 1 &&
              de.best() == 1,
          "a feasible better trial replaces its parent, which goes to A; an infeasible one does not");

    std::mt19937 rng2(3);
    J::JADE flat(Vec(2, 0.5), Vec(2, 0.5), o, rng2);
    flat.ask();
    flat.tell(Vec(6, 0.0));
    check(flat.done(), "done() once P and A are one point (JADE-8)");
}

}   // namespace

int main()
{
    table_iv();
    rule_and_done();
    std::printf("checksum %016llx\n", static_cast<unsigned long long>(g_hash));
    check(g_hash == 0xa042abb46c3f7541ull, "the checksum of the runs, the same bits on every platform");
    std::printf(g_failed ? "FAILED\n" : "all passed\n");
    return g_failed ? 1 : 0;
}
