// SPDX-License-Identifier: Apache-2.0
// The solver catcma (task 26; scalar/catcma_run.hpp over task 19's
// catcma.hpp): on f = sum (x_j - 0.3)^2 + [c_0 != 2] + 0.5 [c_1 != 0], three
// continuous variables in [-1, 1] and two categorical ones of 4 and 3
// levels, it finds x = 0.3 and the categories 2, 0; the same run twice;
// every evaluation through the run's evaluator (the budget kept); its
// refusals (CCR-3).

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/scalar/scalar.hpp>

namespace {

namespace sc = mootation::scalar;
using Vec = std::vector<double>;

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

sc::Problem mixed()
{
    sc::Problem P;
    P.lo = {-1.0, -1.0, -1.0, 0.0, 0.0};
    P.hi = {1.0, 1.0, 1.0, 3.0, 2.0};
    P.m = 1;
    P.cat_levels = {4, 3};
    P.evaluate = [](const Vec& x, Vec& f, Vec&) {
        double s = 0.0;
        for (int j = 0; j < 3; ++j) s += (x[static_cast<std::size_t>(j)] - 0.3) * (x[static_cast<std::size_t>(j)] - 0.3);
        s += (x[3] == 2.0 ? 0.0 : 1.0) + (x[4] == 0.0 ? 0.0 : 0.5);
        f.assign(1, s);
    };
    return P;
}

sc::Result run(const sc::Problem& P, long long budget, unsigned seed = 4)
{
    return sc::run("catcma", P, 1e-6, false, 0.0, 10, budget, seed, {}, {}, 0, {});
}

}   // namespace

int main()
{
    const sc::Problem P = mixed();
    const sc::Result a = run(P, 6000), b = run(P, 6000);
    const bool ok = !a.objectives.empty();
    check(ok && a.objectives[0][0] < 1e-6 && a.variables[0][3] == 2.0 && a.variables[0][4] == 0.0,
          "catcma: x = 0.3 and the categories 2, 0: f = " + (ok ? std::to_string(a.objectives[0][0]) : std::string("-")));
    check(ok && a.objectives == b.objectives && a.variables == b.variables && a.evaluations == b.evaluations,
          "catcma: the same run twice");
    check(a.evaluations <= 6000, "catcma: within the budget");
    bool threw = false;
    sc::Problem none = P;
    none.cat_levels.clear();
    try { run(none, 100); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "catcma without categorical variables refused (CCR-3)");
    threw = false;
    sc::Problem con = P;
    con.n_cons = 1;
    try { run(con, 100); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "catcma with constraints refused (CCR-3)");
    std::printf("%s\n", g_failed ? "FAILED" : "all passed");
    return g_failed ? 1 : 0;
}
