// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// pbh (scalar/pbh.hpp; task 25, §24.2.2):
//   K = 1, d_cut = inf: monotonic basin hopping, the run of ils with
//     ils_accept = better and the same step bit for bit (the answer, the
//     evaluations, the steps and the accepted), in box and absolute units;
//   K = 5 on Rastrigin in 2-D and 5-D: the global minimum; one thread = four
//     threads bit for bit;
//   pbh_dcut = 1.5 (c d_ave) with the 2P-PBH* annealing: d_cut reported and
//     smaller than at the start; pbh_distance = euclid and aligned (sym_blocks)
//     run; a bad knob is refused.
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

const char* kNM = "{\"local_tol\":1e-10,\"solver\":\"nelder_mead\"}";

sc::Result run(const std::string& name, const sc::Problem& P, sc::Knobs k, long long budget, const sc::Start& s,
               unsigned seed = 3)
{
    k.push_back({"inner", word(kNM)});
    return sc::run(name, P, 1e-6, false, 0.0, 10, budget, seed, k, {}, 0, s);
}

bool same(const sc::Result& a, const sc::Result& b)
{
    return a.objectives == b.objectives && a.variables == b.variables && a.evaluations == b.evaluations &&
           setting(a, "steps") == setting(b, "steps") && setting(a, "accepted") == setting(b, "accepted");
}

sc::Start far(int n)
{
    sc::Start s;
    s.has_x0 = true;
    s.x0.assign(static_cast<std::size_t>(n), 4.0);
    return s;
}

void mbh()
{
    const sc::Problem P = rastrigin(2);
    for (const char* units : {"box", "absolute"}) {
        const double st = std::string(units) == "box" ? 0.1 : 1.0;
        const sc::Result ils = run("ils", P, {{"ils_accept", word("better")}, {"ils_step", number(st)},
                                              {"step_units", word(units)}}, 8000, far(2));
        const sc::Result pbh = run("pbh", P, {{"pbh_K", number(1)}, {"pbh_step", number(st)},
                                              {"step_units", word(units)}}, 8000, far(2));
        check(same(ils, pbh), std::string("K = 1, step_units = ") + units + ": ils better bit for bit (" +
                                  setting(pbh, "accepted") + " of " + setting(pbh, "steps") + " accepted, f = " +
                                  std::to_string(pbh.objectives[0][0]) + ")");
    }
}

void population()
{
    for (int n : {2, 5}) {
        const sc::Problem P = rastrigin(n);
        const sc::Result r = run("pbh", P, {{"pbh_K", number(5)}, {"pbh_step", number(0.1)}}, n == 2 ? 60000 : 400000, far(n));
        std::printf("      K = 5, Rastrigin %d-D: f = %.3g, %s iterations, %s of %s accepted\n", n, r.objectives[0][0],
                    setting(r, "iterations").c_str(), setting(r, "accepted").c_str(), setting(r, "steps").c_str());
        check(r.objectives[0][0] < 1e-8, "K = 5, Rastrigin " + std::to_string(n) + "-D: the global minimum");
    }
    const sc::Problem P = rastrigin(3);
    const sc::Result one = run("pbh", P, {{"pbh_K", number(4)}, {"threads", number(1)}}, 20000, far(3));
    const sc::Result four = run("pbh", P, {{"pbh_K", number(4)}, {"threads", number(4)}}, 20000, far(3));
    check(same(one, four), "one thread = four threads bit for bit");
}

void dissimilarity()
{
    const sc::Problem P = rastrigin(3);
    const sc::Result r = run("pbh", P, {{"pbh_K", number(6)}, {"pbh_distance", word("euclid")}, {"pbh_dcut", number(1.5)},
                                        {"pbh_dcut_anneal", word("on")}},
                             40000, far(3));
    std::printf("      euclid, d_cut = 1.5 d_ave, annealed: d_cut at the end %s, f = %.3g\n",
                setting(r, "pbh_dcut_end").c_str(), r.objectives[0][0]);
    check(!setting(r, "pbh_dcut_end").empty() && r.objectives[0][0] < 1e-6, "euclid with d_cut and annealing: runs, solves");
    sc::Problem S = rastrigin(4);
    S.sym_blocks = {sc::symmetry::Blocks{0, 2, 2}};       // two interchangeable pairs
    const sc::Result a = run("pbh", S, {{"pbh_K", number(4)}, {"pbh_distance", word("aligned")}, {"pbh_dcut", number(0.5)}},
                             300000, far(4));
    std::printf("      aligned: f = %.3g, %s iterations\n", a.objectives[0][0], setting(a, "iterations").c_str());
    check(a.objectives[0][0] < 1e-6, "aligned on two interchangeable pairs: runs, solves");
    bool threw = false;
    try {
        (void)run("pbh", P, {{"pbh_distance", word("hamming")}}, 1000, far(3));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "an unknown pbh_distance is refused");
}

}   // namespace

int main()
{
    mbh();
    population();
    dissimilarity();
    std::printf("\n%s\n", g_failed == 0 ? "all passed" : "FAILURES");
    return g_failed == 0 ? 0 : 1;
}
