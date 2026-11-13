// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// mde (scalar/mde.hpp; task 25, §24.2.3):
//   the four variants (mde, greedy, distance, hybrid), population 10, inner
//     lbfgsb, find Rastrigin's global minimum in 5-D; one seed, one run;
//   a convex quadratic: the population collapses and CL16's f_var test ends
//     the run before the budget;
//   LMS14 Algorithm 4: greedy_match relabels a permuted point back onto the
//     record; mde_match = align and greedy run on interchangeable blocks;
//   mde_pop < 4 for mde, greedy matching without sym_blocks: refused.
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

// an accurate local solver, as LMS14 and CL16 (SNOPT, MINOS): with Nelder-Mead
// the populations of 10 collapse early in 5-D
const char* kLS = "{\"solver\":\"lbfgsb\"}";

sc::Result run(const sc::Problem& P, sc::Knobs k, long long budget, unsigned seed = 3)
{
    k.push_back({"inner", word(kLS)});
    return sc::run("mde", P, 1e-6, false, 0.0, 10, budget, seed, k, {}, 0, {});
}

void variants()
{
    const sc::Problem P = rastrigin(5);
    for (const char* v : {"mde", "greedy", "distance", "hybrid"}) {
        const sc::Result r = run(P, {{"mde_variant", word(v)}, {"mde_pop", number(10)}}, 400000);
        std::printf("      %s: f = %.3g, %s sweeps, %s local searches, %lld evaluations\n", v, r.objectives[0][0],
                    setting(r, "sweeps").c_str(), setting(r, "local_searches").c_str(), r.evaluations);
        check(r.objectives[0][0] < 1e-6, std::string("mde_variant = ") + v + ", pop 10: Rastrigin 5-D's global minimum");
    }
    const sc::Result a = run(P, {{"mde_pop", number(8)}}, 50000), b = run(P, {{"mde_pop", number(8)}}, 50000);
    check(a.objectives == b.objectives && a.evaluations == b.evaluations, "one seed, one run");
}

void collapse()
{
    sc::Problem Q;
    Q.lo.assign(4, -3.0);
    Q.hi.assign(4, 3.0);
    Q.m = 1;
    Q.evaluate = [](const Vec& x, Vec& f, Vec&) {
        double s = 0.0;
        for (std::size_t j = 0; j < x.size(); ++j) s += (1.0 + j) * (x[j] - 0.5) * (x[j] - 0.5);
        f.assign(1, s);
    };
    const sc::Result r = run(Q, {{"mde_pop", number(6)}}, 1000000);
    std::printf("      quadratic: %lld of 1000000 evaluations, %s sweeps\n", r.evaluations, setting(r, "sweeps").c_str());
    check(r.evaluations < 1000000 && r.objectives[0][0] < 1e-10, "a convex quadratic: the population collapses, the run ends early");
}

void matching()
{
    const std::vector<sc::symmetry::Blocks> g = {sc::symmetry::Blocks{0, 2, 3}};
    const Vec ref = {0.1, 0.2, 1.1, 1.2, 2.1, 2.2};
    const Vec x = {2.0, 2.3, 0.0, 0.25, 1.15, 1.1};             // ref's blocks permuted and moved a little
    const Vec m = sc::search::greedy_match(x, ref, g);
    check(m == Vec{0.0, 0.25, 1.15, 1.1, 2.0, 2.3}, "greedy_match (LMS14 Algorithm 4): the blocks back in ref's order");
    sc::Problem S = rastrigin(4);
    S.sym_blocks = {sc::symmetry::Blocks{0, 2, 2}};
    for (const char* how : {"align", "greedy"}) {
        const sc::Result r = run(S, {{"mde_pop", number(8)}, {"mde_match", word(how)}}, 200000);
        check(r.objectives[0][0] < 1e-8, std::string("mde_match = ") + how + " on two interchangeable pairs: solves");
    }
    bool t1 = false, t2 = false;
    try {
        (void)run(rastrigin(2), {{"mde_pop", number(3)}}, 1000);
    } catch (const std::invalid_argument&) {
        t1 = true;
    }
    try {
        (void)run(rastrigin(2), {{"mde_match", word("greedy")}}, 1000);
    } catch (const std::invalid_argument&) {
        t2 = true;
    }
    check(t1 && t2, "mde_pop = 3 for mde, greedy matching without sym_blocks: refused");
}

}   // namespace

int main()
{
    variants();
    collapse();
    matching();
    std::printf("\n%s\n", g_failed == 0 ? "all passed" : "FAILURES");
    return g_failed == 0 ? 0 : 1;
}
