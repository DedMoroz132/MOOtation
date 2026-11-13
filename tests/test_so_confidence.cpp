// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// include/mootation/so_confidence.hpp (task 25, item 24.7.3), on numbers
// worked out by hand:
//   characteristic_effort  M_95 = 29 / 9 / 5 / 2 at R = 0.1 / 0.3 / 0.5 / 0.9
//                          (K18 (12)); the safeguard at R = 0 and R = 1; an
//                          integer quotient not rounded up (KAV-2)
//   batch_reliability      K18 (9); M_95 runs reach 95 %, M_95 - 1 do not
//   characteristic_cost, efficiency_ratio   K18 (13), (15)
//   typ, spread            median and IQR, linear between order statistics
//   success                one-sided, relative and absolute (KAV-1)
//   values_at, share_at    Q22's distribution at a budget (KAV-3, KAV-5)
// ============================================================================

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/so_confidence.hpp>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

namespace so = mootation::so;

void effort()
{
    check(so::characteristic_effort(0.1, 0.95, 100) == 29 && so::characteristic_effort(0.3, 0.95, 100) == 9 &&
              so::characteristic_effort(0.5, 0.95, 100) == 5 && so::characteristic_effort(0.9, 0.95, 100) == 2,
          "M_95 = 29 / 9 / 5 / 2 at R = 0.1 / 0.3 / 0.5 / 0.9");
    // R = 0 on 99 runs: eps_R = 0.01, log(0.05)/log(0.99) = 298.07 -> 299
    check(so::characteristic_effort(0.0, 0.95, 99) == 299, "R = 0 is safeguarded to 1/(n + 1): M_95 = 299 at n = 99");
    // R = 1 on 99 runs: 0.99, log(0.05)/log(0.01) = 0.65 -> 1
    check(so::characteristic_effort(1.0, 0.95, 99) == 1, "R = 1 is safeguarded to 1 - 1/(n + 1): one run");
    check(so::characteristic_effort(0.5, 0.75, 100) == 2, "log(0.25)/log(0.5) = 2 is not rounded up to 3 (KAV-2)");
    check(so::characteristic_effort(0.5, 0.5, 100) == 1 && so::characteristic_effort(0.6, 0.5, 100) == 1,
          "alpha at or below R: one run");
    bool all = true;
    for (int i = 1; i < 100; ++i) {
        const double R = i / 100.0;
        const long long m = so::characteristic_effort(R, 0.95, 1000);
        all = all && so::batch_reliability(R, m) >= 0.95 - 1e-12 && (m == 1 || so::batch_reliability(R, m - 1) < 0.95);
    }
    check(all, "R = 0.01 ... 0.99: M_95 runs reach 95 %, one run fewer does not (K18 (9), (12))");
    check(std::fabs(so::batch_reliability(0.5, 5) - 0.96875) < 1e-15 && so::batch_reliability(0.3, 0) == 0.0,
          "batch reliability 1 - 0.5^5 = 0.96875; no runs: 0");
    bool threw = false;
    try {
        (void)so::characteristic_effort(0.5, 1.0, 10);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "alpha = 1 refused");
}

void cost_and_spread()
{
    check(so::characteristic_cost(9, 250.0) == 2250.0, "C = M c = 9 * 250");
    // A: M = 5 at 400 calls a run; B: M = 2 at 3000: kappa^{A/B} = 6000 / 2000 = 3
    check(so::efficiency_ratio(so::characteristic_cost(5, 400.0), so::characteristic_cost(2, 3000.0)) == 3.0,
          "kappa^{A/B} = C^B / C^A = 3");
    check(so::typ({4, 1, 3, 2}) == 2.5 && so::spread({4, 1, 3, 2}) == 1.5,
          "1, 2, 3, 4: median 2.5, quartiles 1.75 and 3.25, IQR 1.5");
    check(so::typ({7}) == 7.0 && so::spread({7}) == 0.0 && so::typ({1, 2, 100}) == 2.0,
          "one problem: its M, spread 0; the median ignores an outlier");
    check(so::reliability({true, false, true, true}) == 0.75 && so::reliability({}) == 0.0, "R = 3/4; no runs: 0");
}

void success()
{
    const double tol_rel = so::success_tolerance(0.01, -200.0, true);
    check(tol_rel == 2.0 && so::success(-198.5, -200.0, tol_rel) && !so::success(-197.5, -200.0, tol_rel),
          "relative 1 % of |f_best| = 200: -198.5 succeeds, -197.5 does not");
    check(so::success(-250.0, -200.0, tol_rel), "below the best known value: a success (KAV-1)");
    check(so::success_tolerance(1e-8, 0.0, false) == 1e-8 && so::success(5e-9, 0.0, 1e-8) && !so::success(2e-8, 0.0, 1e-8),
          "absolute 1e-8 at f_best = 0");
}

void budgets()
{
    // (evaluations, grad_calls, f, cv)
    const std::vector<std::vector<so::Improvement>> runs = {
        {{1, 0, 50.0, 0.0}, {150, 0, 10.0, 0.0}, {900, 0, 1.0, 0.0}},
        {{1, 0, 40.0, 0.0}, {450, 0, 5.0, 0.0}},
        {{1, 0, 9.0, 1.0}},                        // never feasible (KAV-5)
        {{300, 0, 30.0, 0.0}}};                    // nothing before 300
    check(so::values_at(runs, 200.0) == std::vector<double>{10.0, 40.0}, "B = 200: 10 and 40; the others have none");
    check(so::values_at(runs, 500.0) == std::vector<double>{5.0, 10.0, 30.0}, "B = 500: 5, 10, 30, sorted");
    check(so::values_at(runs, 1000.0) == std::vector<double>{1.0, 5.0, 30.0}, "B = 1000: 1, 5, 30");
    check(so::share_at(runs, 500.0, 10.0) == 0.5 && so::share_at(runs, 1000.0, 1.0) == 0.25,
          "share at or below the target over all four runs: 2/4 at B = 500, 1/4 at B = 1000");
    check(so::values_at({{{100, 100, 3.0, 0.0}}}, 150.0).empty() && so::values_at({{{100, 100, 3.0, 0.0}}}, 150.0, 0.0).size() == 1,
          "grad_cost counts: 100 + 100 > 150; with grad_cost 0 it is in");
    check(so::q22_budgets() == std::vector<double>{200.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0}, "Q22's budgets");
}

}   // namespace

int main()
{
    effort();
    cost_and_spread();
    success();
    budgets();
    std::printf("\n%s\n", g_failed == 0 ? "all passed" : "FAILURES");
    return g_failed == 0 ? 0 : 1;
}
