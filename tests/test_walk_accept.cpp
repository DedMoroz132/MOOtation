// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// ils and basin_hopping, task 25's knobs (search.hpp, SRCH-15 ... SRCH-18):
//   neutral values give the old runs bit for bit (the answer, the evaluations,
//     the steps and the accepted): step_units = absolute on a box of width 1;
//     ils_accept = threshold with walk_th0 = 1e-300, walk_th_mu = 1 (accept
//     dE < 0 only, as better); ladder_min = ladder_max = ils_step,
//     ladder_step = 0; bh_accept = metropolis;
//   threshold (DTS, Lai et al. 2022) and lsmc_rel (T from the median |dE|)
//     find Rastrigin's global minimum in 2-D from a far start (ils) and
//     report their final threshold / temperature;
//   the ladder moves the step (a run with a ladder differs from one without).
// ============================================================================

#include <cmath>
#include <cstdio>
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

sc::Problem rastrigin(int n, double lo, double hi)
{
    sc::Problem P;
    P.lo.assign(static_cast<std::size_t>(n), lo);
    P.hi.assign(static_cast<std::size_t>(n), hi);
    P.m = 1;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) {
        double s = 10.0 * static_cast<double>(x.size());
        for (double v : x) s += v * v - 10.0 * mootation::dm::cos(2.0 * 3.14159265358979323846 * v);
        f.assign(1, s);
    };
    return P;
}

const char* kNM = "{\"local_tol\":1e-10,\"solver\":\"nelder_mead\"}";

sc::Start far_start(double v)
{
    sc::Start s;
    s.has_x0 = true;
    s.x0 = {v, -v};
    return s;
}

bool same(const sc::Result& a, const sc::Result& b)
{
    return a.objectives == b.objectives && a.variables == b.variables && a.evaluations == b.evaluations &&
           setting(a, "steps") == setting(b, "steps") && setting(a, "accepted") == setting(b, "accepted");
}

sc::Result run(const std::string& name, const sc::Problem& P, sc::Knobs k, long long budget, const sc::Start& s)
{
    k.push_back({"inner", word(kNM)});
    return sc::run(name, P, 1e-6, false, 0.0, 10, budget, 3, k, {}, 0, s);
}

void neutral()
{
    // a box of width 1: absolute = box
    const sc::Problem unit = rastrigin(2, -0.5, 0.5);
    const sc::Start s0 = far_start(0.45);
    check(same(run("ils", unit, {{"ils_step", number(0.3)}}, 4000, s0),
               run("ils", unit, {{"ils_step", number(0.3)}, {"step_units", word("absolute")}}, 4000, s0)),
          "ils, step_units = absolute on a box of width 1: the old run bit for bit");
    const sc::Problem P = rastrigin(2, -5.12, 5.12);
    const sc::Start far = far_start(4.0);
    const sc::Result better = run("ils", P, {{"ils_accept", word("better")}}, 8000, far);
    const sc::Result th0 = run("ils", P,
                               {{"ils_accept", word("threshold")}, {"walk_th0", number(1e-300)}, {"walk_th_mu", number(1.0)}},
                               8000, far);
    check(same(better, th0), "ils_accept = threshold, Th = 1e-300 and mu = 1: = better bit for bit (" +
                                 setting(th0, "accepted") + " of " + setting(th0, "steps") + " accepted)");
    const sc::Result plain = run("ils", P, {{"ils_step", number(0.1)}}, 8000, far);
    const sc::Result flat = run("ils", P,
                                {{"ils_step", number(0.1)}, {"ladder_min", number(0.1)}, {"ladder_max", number(0.1)},
                                 {"ladder_step", number(0.0)}},
                                8000, far);
    check(same(plain, flat), "ladder_min = ladder_max = ils_step, ladder_step = 0: the old run bit for bit");
    const sc::Result bh = run("basin_hopping", P, {}, 8000, far);
    const sc::Result bhm = run("basin_hopping", P, {{"bh_accept", word("metropolis")}}, 8000, far);
    check(same(bh, bhm), "bh_accept = metropolis: the old basin_hopping bit for bit");
}

void new_rules()
{
    const sc::Problem P = rastrigin(2, -5.12, 5.12);
    const sc::Start far = far_start(4.0);
    const sc::Result th = run("ils", P, {{"ils_accept", word("threshold")}, {"ils_step", number(0.2)}}, 20000, far);
    std::printf("      threshold: f = %.3g, %s of %s accepted, final Th = %s\n", th.objectives[0][0],
                setting(th, "accepted").c_str(), setting(th, "steps").c_str(), setting(th, "walk_th_end").c_str());
    check(th.objectives[0][0] < 1e-8 && !setting(th, "walk_th_end").empty(),
          "ils_accept = threshold (DTS): Rastrigin's global minimum from (4, -4)");
    const sc::Result rel = run("ils", P, {{"ils_accept", word("lsmc_rel")}, {"ils_step", number(0.2)}}, 20000, far);
    std::printf("      lsmc_rel: f = %.3g, T = %s\n", rel.objectives[0][0], setting(rel, "walk_t_rel_end").c_str());
    check(rel.objectives[0][0] < 1e-8 && std::stod(setting(rel, "walk_t_rel_end")) > 0.0,
          "ils_accept = lsmc_rel: the global minimum, a temperature from the median |dE| of the first steps");
    const sc::Result bht = run("basin_hopping", P, {{"bh_accept", word("threshold")}}, 20000, far);
    std::printf("      basin_hopping, threshold: f = %.3g, %s of %s accepted, final step %s, Th %s\n",
                bht.objectives[0][0], setting(bht, "accepted").c_str(), setting(bht, "steps").c_str(),
                setting(bht, "bh_step_end").c_str(), setting(bht, "walk_th_end").c_str());
    check(std::isfinite(bht.objectives[0][0]) && !setting(bht, "walk_th_end").empty(),
          "bh_accept = threshold: runs and reports its threshold (with bh's small, shrinking step most new minima "
          "are the current one, dE = 0 counts as refused, and Th grows: SRCH-16)");
    const sc::Result lad = run("ils", P,
                               {{"ils_step", number(0.1)}, {"ladder_min", number(0.05)}, {"ladder_max", number(0.5)},
                                {"ladder_step", number(0.05)}},
                               20000, far);
    const sc::Result plain = run("ils", P, {{"ils_step", number(0.1)}}, 20000, far);
    check(!same(lad, plain) && lad.objectives[0][0] < 1e-8, "the ladder 0.05 ... 0.5: another run, the global minimum");
    bool threw = false;
    try {
        (void)run("ils", P, {{"ils_accept", word("threshold")}, {"walk_th_mu", number(1.5)}}, 1000, far);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "walk_th_mu > 1 refused");
}

}   // namespace

int main()
{
    neutral();
    new_rules();
    std::printf("\n%s\n", g_failed == 0 ? "all passed" : "FAILURES");
    return g_failed == 0 ? 0 : 1;
}
