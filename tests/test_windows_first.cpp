// SPDX-License-Identifier: Apache-2.0
// <windows.h> before the library (task 31, layer 1; the owner's choice of
// 17.11): plain windows.h, without NOMINMAX or WIN32_LEAN_AND_MEAN, so its
// macros min, max (function-like), near, far (minwindef.h), small, hyper
// (rpcndr.h), interface and the rest are defined, and then the library's
// umbrella headers — the algorithms (mootation.hpp), the single-objective
// methods and strategies (scalar/scalar.hpp) and the campaign engine
// (campaign/engine.hpp). The library calls min and max in parentheses,
// (std::min)(a, b), (std::numeric_limits<T>::max)(), and names nothing near,
// far or small. A short run of each kind shows the code runs; that it runs
// as before is the other tests' (golden_fingerprints, test_stop_rule's
// cc2d22c runs). Elsewhere the same headers without windows.h.

#if defined(_WIN32)
#include <windows.h>
#if !defined(min) || !defined(max) || !defined(near) || !defined(far) || !defined(small)
#error "windows.h's macros min, max, near, far and small are expected here"
#endif
#endif

#include <mootation/campaign/engine.hpp>
#include <mootation/mootation.hpp>
#include <mootation/scalar/scalar.hpp>

#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

}   // namespace

int main()
{
    namespace sc = mootation::scalar;
    using Vec = std::vector<double>;
    sc::Problem P;
    P.lo.assign(3, -2.0);
    P.hi.assign(3, 2.0);
    P.m = 1;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) {
        double s = 0.0;
        for (double v : x) s += (v - 0.5) * (v - 0.5);
        f.assign(1, s);
    };
    const sc::Knobs k{{"inner", sc::Knob{true, "{\"solver\":\"nelder_mead\"}", 0.0}}};
    const sc::Result r = sc::run("basin_hopping", P, 1e-6, false, 0.0, 10, 2000, 3, k, {}, 0, sc::Start{});
    check(!r.objectives.empty() && r.objectives[0][0] < 1e-6 && r.evaluations <= 2000,
          "basin_hopping after windows.h runs and finds the minimum");
    const sc::Result m = sc::run("cmaes", P, 1e-6, false, 0.0, 10, 1000, 3, {}, {}, 0, sc::Start{});
    check(!m.objectives.empty() && m.objectives[0][0] < 1e-6, "cmaes after windows.h runs and finds the minimum");
    std::printf("%s\n", g_failed ? "FAILED" : "all passed");
    return g_failed ? 1 : 0;
}
