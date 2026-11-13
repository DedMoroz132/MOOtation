// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// cmaes's knobs of task 25 through the single-objective path (scalar.hpp
// SINGLE-6; §24.3.1, §24.3.4):
//   - a problem's discrete values change nothing without cma_margin, and
//     cma_eigen_every = 1 is the default run: bit for bit, no new setting
//     but the one given;
//   - cma_eigen_every = 0 (H09's interval) still solves the ellipsoid;
//   - cma_margin = on on SphereInt (Hamano et al. 2022, §5.1: 5 continuous and
//     5 integer variables in [-10, 10], f evaluated at the rounded integers as
//     minimize's var_types does) reaches f = 0 on every seed, the alpha
//     1/(N lambda) = 1/(10 * 10) in the settings; without the margin no
//     better (the count printed);
//   - the refusals: cma_margin = on without discrete values; an alpha outside
//     [0, 1/2); cma_eigen_every not whole.
// ============================================================================

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/scalar/scalar.hpp>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

namespace sc = mootation::scalar;
using Vec = std::vector<double>;

sc::Knob word(const std::string& w) { return {true, w, 0.0}; }
sc::Knob number(double v) { return {false, "", v}; }

// HM22's SphereInt at N = 10: x_0..x_4 continuous in [-5, 5], z_5..z_9
// integers in [-10, 10], their box [-10.4999999, 10.4999999] as minimize's
// var_types (Layout.core_bounds), f at the rounded values
sc::Problem sphere_int(bool with_values)
{
    sc::Problem P;
    P.m = 1;
    for (int j = 0; j < 10; ++j) {
        P.lo.push_back(j < 5 ? -5.0 : -10.4999999);
        P.hi.push_back(j < 5 ? 5.0 : 10.4999999);
    }
    P.evaluate = [](const Vec& x, Vec& f, Vec&) {
        double s = 0.0;
        for (int j = 0; j < 10; ++j) {
            const double v = j < 5 ? x[static_cast<std::size_t>(j)] : std::nearbyint(x[static_cast<std::size_t>(j)]);
            s += v * v;
        }
        f = {s};
    };
    if (with_values) {
        P.discrete.assign(10, {});
        for (std::size_t j = 5; j < 10; ++j)
            for (int v = -10; v <= 10; ++v) P.discrete[j].push_back(v);
    }
    return P;
}

sc::Problem ellipsoid()
{
    sc::Problem P;
    P.m = 1;
    P.lo.assign(10, -5.0);
    P.hi.assign(10, 5.0);
    P.evaluate = [](const Vec& x, Vec& f, Vec&) {
        double s = 0.0;
        for (std::size_t j = 0; j < 10; ++j) s += std::pow(1e6, static_cast<double>(j) / 9.0) * (x[j] - 1.0) * (x[j] - 1.0);
        f = {s};
    };
    return P;
}

std::string setting(const sc::Result& r, const std::string& k)
{
    for (const auto& [a, b] : r.settings)
        if (a == k) return b;
    return "(none)";
}

sc::Result run_cma(const sc::Problem& P, const sc::Knobs& knobs, unsigned seed, long long budget = 20000)
{
    return sc::run("cmaes", P, 1e-6, false, 0.0, 20, budget, seed, knobs, {}, 0, sc::Start{});
}

bool refused(const sc::Problem& P, const sc::Knobs& knobs)
{
    try {
        run_cma(P, knobs, 1, 200);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}   // namespace

int main()
{
    // 1. nothing changes without the knobs
    const sc::Result plain = run_cma(sphere_int(false), {}, 3);
    const sc::Result valued = run_cma(sphere_int(true), {}, 3);
    const sc::Result every1 = run_cma(sphere_int(false), {{"cma_eigen_every", number(1)}}, 3);
    check(valued.variables == plain.variables && valued.evaluations == plain.evaluations &&
              valued.settings == plain.settings,
          "cmaes: the problem's discrete values change nothing without cma_margin, bit for bit");
    check(every1.variables == plain.variables && every1.evaluations == plain.evaluations &&
              setting(every1, "cma_eigen_every") == "1" && setting(plain, "cma_eigen_every") == "(none)" &&
              setting(plain, "cma_margin") == "(none)",
          "cmaes: cma_eigen_every = 1 is the default run, bit for bit; the setting only where given");

    // 2. the lazy decomposition
    const sc::Result lazy = run_cma(ellipsoid(), {{"cma_eigen_every", number(0)}}, 1, 60000);
    std::printf("      ellipsoid D = 10, cma_eigen_every = 0: f = %.3g after %lld evaluations\n",
                lazy.objectives[0][0], lazy.evaluations);
    check(lazy.objectives[0][0] < 1e-8 && setting(lazy, "cma_eigen_every") == "0",
          "cmaes: cma_eigen_every = 0 (H09's interval) solves the ellipsoid");

    // 3. the margin on SphereInt
    int with = 0, without = 0;
    std::string alpha;
    for (unsigned seed = 1; seed <= 10; ++seed) {
        const sc::Result on = run_cma(sphere_int(true), {{"cma_margin", word("on")}}, seed);
        const sc::Result off = run_cma(sphere_int(true), {}, seed);
        with += on.objectives[0][0] < 1e-10 ? 1 : 0;
        without += off.objectives[0][0] < 1e-10 ? 1 : 0;
        alpha = setting(on, "cma_margin_alpha");
        std::printf("      seed %u: margin f = %.3g (%lld evaluations), without f = %.3g\n", seed,
                    on.objectives[0][0], on.evaluations, off.objectives[0][0]);
    }
    std::printf("      SphereInt N = 10: f < 1e-10 on %d of 10 seeds with the margin, %d without\n", with, without);
    check(with == 10, "cma_margin = on: SphereInt solved on every seed (HM22 Table 2: 100 %)");
    check(without <= with, "without the margin no better (" + std::to_string(without) +
                               " of 10: at N = 10 in the normalized box plain cmaes solves it too)");
    check(alpha == "0.01", "cma_margin_alpha: 1/(N lambda) = 1/(10 * 10) in the settings, got " + alpha);

    // 4. refusals
    check(refused(sphere_int(false), {{"cma_margin", word("on")}}), "cma_margin = on without discrete values: refused");
    check(refused(sphere_int(true), {{"cma_margin", word("on")}, {"cma_margin_alpha", number(0.5)}}),
          "cma_margin_alpha = 0.5: refused");
    check(refused(sphere_int(true), {{"cma_margin", word("maybe")}}), "cma_margin = maybe: refused");
    check(refused(sphere_int(true), {{"cma_eigen_every", number(1.5)}}), "cma_eigen_every = 1.5: refused");

    std::printf(g_failed ? "%d FAILED\n" : "all passed\n", g_failed);
    return g_failed ? 1 : 0;
}
