// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// ma_chains (scalar/ma_chains.hpp; task 25, §24.2.1):
//   the local searches sw, ssw, mts_ls1, mts_ls2 and inner (lbfgsb) on
//     Rastrigin and Rosenbrock in 10-D: the values reached (printed); MTS-LS1
//     (built for separable functions, M11 §3.4) solves the separable
//     Rastrigin; Solis-Wets chains bring Rosenbrock below 1;
//   lamarck = share with lamarck_share = 0 is lamarck = off bit for bit, 1 is
//     on bit for bit (the coins have their own stream, MAC-6); off differs;
//   ma_delta_min huge: S_LS empties and the population restarts (MAC-1);
//   one seed, one run.
// ============================================================================

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

sc::Knob word(const std::string& w) { return {true, w, 0.0}; }
sc::Knob number(double v) { return {false, "", v}; }

std::string setting(const sc::Result& r, const std::string& key)
{
    for (const auto& [k, v] : r.settings)
        if (k == key) return v;
    return "";
}

sc::Problem rastrigin(int n)
{
    sc::Problem P;
    P.lo.assign(static_cast<std::size_t>(n), -5.12);
    P.hi.assign(static_cast<std::size_t>(n), 5.12);
    P.m = 1;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) {
        double s = 10.0 * static_cast<double>(x.size());
        for (double v : x) s += v * v - 10.0 * mootation::dm::cos(2.0 * 3.14159265358979323846 * v);
        f.assign(1, s);
    };
    return P;
}

sc::Problem rosenbrock(int n)
{
    sc::Problem P;
    P.lo.assign(static_cast<std::size_t>(n), -5.0);
    P.hi.assign(static_cast<std::size_t>(n), 5.0);
    P.m = 1;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) {
        double s = 0.0;
        for (std::size_t i = 0; i + 1 < x.size(); ++i)
            s += 100.0 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) + (1.0 - x[i]) * (1.0 - x[i]);
        f.assign(1, s);
    };
    return P;
}

sc::Result run(const sc::Problem& P, sc::Knobs k, long long budget, unsigned seed = 3)
{
    return sc::run("ma_chains", P, 1e-6, false, 0.0, 10, budget, seed, k, {}, 0, {});
}

bool same(const sc::Result& a, const sc::Result& b)
{
    return a.objectives == b.objectives && a.variables == b.variables && a.evaluations == b.evaluations &&
           setting(a, "ls_applications") == setting(b, "ls_applications");
}

void local_searches()
{
    const sc::Problem R = rastrigin(10), B = rosenbrock(10);
    double f_mts1 = 1e300, f_sw_rosen = 1e300;
    for (const char* ls : {"sw", "ssw", "mts_ls1", "mts_ls2", "inner"}) {
        sc::Knobs k{{"ma_local", word(ls)}};
        if (std::string(ls) == "inner") k.push_back({"inner", word("{\"solver\":\"lbfgsb\"}")});
        const sc::Result r = run(R, k, 100000), b = run(B, k, 100000);
        std::printf("      ma_local = %-7s: Rastrigin-10 f = %.3g (%s searches), Rosenbrock-10 f = %.3g\n", ls,
                    r.objectives[0][0], setting(r, "ls_applications").c_str(), b.objectives[0][0]);
        if (std::string(ls) == "mts_ls1") f_mts1 = r.objectives[0][0];
        if (std::string(ls) == "sw") f_sw_rosen = b.objectives[0][0];
    }
    check(f_mts1 < 1e-8, "mts_ls1 chains on the separable Rastrigin-10: the global minimum");
    check(f_sw_rosen < 1.0, "Solis-Wets chains on Rosenbrock-10: below 1 in 100000 evaluations");
}

void lamarck_and_restart()
{
    const sc::Problem R = rastrigin(5);
    const sc::Result off = run(R, {{"lamarck", word("off")}}, 30000), on = run(R, {}, 30000);
    const sc::Result s0 = run(R, {{"lamarck", word("share")}, {"lamarck_share", number(0.0)}}, 30000);
    const sc::Result s1 = run(R, {{"lamarck", word("share")}, {"lamarck_share", number(1.0)}}, 30000);
    check(same(s0, off) && same(s1, on) && !same(off, on),
          "lamarck = share: 0 is off and 1 is on bit for bit; off differs from on");
    const sc::Result rs = run(R, {{"ma_delta_min", number(1e9)}, {"ma_pop", number(10)}, {"ma_istr", number(100)}}, 30000);
    std::printf("      delta_min = 1e9: %s restarts, %s searches\n", setting(rs, "restarts").c_str(),
                setting(rs, "ls_applications").c_str());
    check(std::stoll(setting(rs, "restarts")) > 0, "an empty S_LS restarts the population (MAC-1)");
    const sc::Result a = run(R, {}, 20000), b = run(R, {}, 20000);
    check(same(a, b), "one seed, one run");
    bool threw = false;
    try {
        (void)run(R, {{"ma_rlg", number(1.0)}}, 1000);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "ma_rlg = 1 refused");
}

}   // namespace

int main()
{
    local_searches();
    lamarck_and_restart();
    std::printf("\n%s\n", g_failed == 0 ? "all passed" : "FAILURES");
    return g_failed == 0 ? 0 : 1;
}
