// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// include/mootation/so_metrics.hpp (task 18, stage 1, §1.3), on records worked
// out by hand:
//   targets   COCO's 51 precisions 10^2 ... 10^-8 a fifth of a decade apart,
//             its table's seven, Moré and Wild's four; absolute and relative
//             targets;
//   runtime   the cost at the first feasible record at or below the target,
//             grad_cost counted; none where no record is;
//   ert       the runs' costs over the hits (H16 §4.1), +inf without a hit;
//   ecdf      misses in the denominator;
//   value_at  the best feasible f within the budget;
//   simulated_runtimes (SOM-4, SOM-6): every run hit and N = K, each runtime
//             once (the first runs a permutation); one hit and one miss of
//             cost 100, every sample 10 + 100 j and the mean near ERT = 110,
//             with the first run permuted or drawn; no hit, no sample; the
//             same seed, the same samples.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <mootation/so_metrics.hpp>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

namespace so = mootation::so;

bool near(double a, double b) { return std::fabs(a - b) <= 4e-16 * std::fabs(b); }

void targets()
{
    const std::vector<double> p = so::coco_precisions();
    bool grid = p.size() == 51 && p[0] == 100.0 && p[10] == 1.0 && near(p[50], 1e-8);
    for (std::size_t i = 0; grid && i < p.size(); ++i)
        grid = near(p[i], std::pow(10.0, static_cast<double>(10 - static_cast<int>(i)) / 5.0));   // (5 k_max - i) / 5
    check(grid, "COCO's 51 precisions, 10^2 to 10^-8 a fifth of a decade apart");
    check(so::coco_precisions(-6).size() == 41 && so::coco_table_precisions().size() == 7 &&
              so::more_wild_factors() == std::vector<double>{1e-1, 1e-3, 1e-5, 1e-7},
          "k_min = -6: 41; the tables' seven; Moré and Wild's four");
    check(so::default_precisions(so::TargetKind::Absolute) == p &&
              so::default_precisions(so::TargetKind::Relative) == so::more_wild_factors(),
          "the kinds' precisions: COCO's for absolute, Moré and Wild's for relative");
    const std::vector<double> a = so::targets(so::TargetKind::Absolute, {1.0, 1e-2}, -3.0, 99.0);
    const std::vector<double> r = so::targets(so::TargetKind::Relative, {1e-1, 1e-3}, 2.0, 12.0);
    check(a == std::vector<double>{-2.0, -3.0 + 1e-2} && r == std::vector<double>{2.0 + 1e-1 * 10.0, 2.0 + 1e-3 * 10.0},
          "absolute: f_ref + eps; relative: f_ref + eps (f(x0) - f_ref)");
}

void runtimes()
{
    // (evaluations, grad_calls, f, cv): infeasible first, then improving
    const std::vector<so::Improvement> rec = {{1, 0, -5.0, 2.0}, {3, 1, 10.0, 0.0}, {6, 2, 3.0, 0.0}, {9, 4, 0.5, 0.0}};
    check(so::runtime(rec, 3.0) == std::optional<double>(8.0) && so::runtime(rec, 10.0) == std::optional<double>(4.0),
          "runtime: the first feasible record at or below the target, evaluations + grad_calls");
    check(so::runtime(rec, 0.5, 0.5) == std::optional<double>(11.0) && !so::runtime(rec, 0.4) && !so::runtime(rec, -5.0),
          "runtime: grad_cost 0.5; a target no feasible record reaches has none (the infeasible -5 does not count)");
    check(so::value_at(rec, 4.0) == std::optional<double>(10.0) && so::value_at(rec, 8.0) == std::optional<double>(3.0) &&
              !so::value_at(rec, 2.0) && so::value_at(rec, 1e9) == std::optional<double>(0.5),
          "value_at: the best feasible f within the budget, none before the first feasible record");
}

void ert_ecdf()
{
    const std::vector<std::optional<double>> t = {10.0, std::nullopt, 30.0};
    check(so::ert(t, {100.0, 200.0, 300.0}) == 120.0, "ERT: (10 + 200 + 30) / 2 hits");
    check(std::isinf(so::ert({std::nullopt, std::nullopt}, {5.0, 7.0})) && so::ert({}, {}) == 0.0,
          "ERT: no hit +inf; no run 0 / 0 = 0");
    const std::vector<double> e = so::ecdf({1.0, std::nullopt, 5.0, 10.0}, {0.0, 1.0, 5.0, 100.0});
    check(e == std::vector<double>{0.0, 0.25, 0.5, 0.75}, "ECDF: misses in the denominator");
}

void simulated()
{
    using O = std::optional<double>;
    std::vector<double> got = so::simulated_runtimes({O(3.0), O(1.0), O(2.0), O(5.0)}, {3.0, 1.0, 2.0, 5.0}, 4, 7);
    std::sort(got.begin(), got.end());
    check(got == std::vector<double>{1.0, 2.0, 3.0, 5.0}, "every run hit, N = K: each runtime once");
    for (bool permuted : {true, false}) {
        const std::vector<double> s = so::simulated_runtimes({O(), O(10.0)}, {100.0, 10.0}, 20000, 11, permuted);
        bool shape = s.size() == 20000;
        double mean = 0.0;
        for (double v : s) {
            const double j = (v - 10.0) / 100.0;
            shape = shape && j >= 0.0 && j == std::floor(j);
            mean += v / static_cast<double>(s.size());
        }
        check(shape && std::fabs(mean - 110.0) < 0.03 * 110.0,
              std::string("a hit of 10 and a miss of 100, first ") + (permuted ? "permuted" : "drawn") +
                  ": every sample 10 + 100 j, mean " + std::to_string(mean) + " near ERT 110");
    }
    check(so::simulated_runtimes({O(), O()}, {5.0, 6.0}, 10, 1).empty(), "no hit: no sample");
    check(so::simulated_runtimes({O(), O(4.0), O()}, {9.0, 4.0, 2.0}, 50, 3) ==
              so::simulated_runtimes({O(), O(4.0), O()}, {9.0, 4.0, 2.0}, 50, 3),
          "the same seed, the same samples");
}

}   // namespace

int main()
{
    targets();
    runtimes();
    ert_ecdf();
    simulated();
    if (g_failed) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
