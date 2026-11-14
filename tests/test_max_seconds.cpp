// SPDX-License-Identifier: Apache-2.0
// THE TIME (task 28, §28.6.1; scalar.hpp TIME-1 ... TIME-4): max_seconds.
//   * a method on a slow problem (1 ms an evaluation) with max_seconds =
//     0.2: the run ends near 0.2 s, far inside its budget, time_up = yes,
//     an answer kept; a strategy (multistart) alike;
//   * without max_seconds no time setting, the run as before;
//     Result::seconds every run's;
//   * a deadline long enough: the run as without it, bit for bit, time_up = no;
//   * refused: max_seconds <= 0 or a word.

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/scalar/scalar.hpp>

namespace sc = mootation::scalar;
using Vec = std::vector<double>;

static int failures = 0;

static void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// the sphere, each evaluation taking `ms` milliseconds (a busy wait)
static sc::Problem slow_sphere(double ms)
{
    sc::Problem P;
    P.m = 1;
    P.lo.assign(3, -5.0);
    P.hi.assign(3, 5.0);
    P.evaluate = [ms](const Vec& x, Vec& f, Vec&) {
        const auto until = std::chrono::steady_clock::now() + std::chrono::duration<double, std::milli>(ms);
        while (std::chrono::steady_clock::now() < until) {
        }
        double s = 0.0;
        for (double v : x) s += v * v;
        f[0] = s;
    };
    return P;
}

static sc::Knob number(double v)
{
    sc::Knob k;
    k.number = v;
    return k;
}

static std::string setting(const sc::Result& r, const std::string& k)
{
    for (const auto& [a, b] : r.settings)
        if (a == k) return b;
    return "";
}

static void ends_early()
{
    const sc::Problem P = slow_sphere(1.0);
    sc::Knob inner;                                  // multistart's inner solver
    inner.is_word = true;
    inner.word = "{\"solver\":\"nelder_mead\"}";
    for (const char* name : {"de", "multistart"}) {
        sc::Knobs k{{"max_seconds", number(0.2)}};
        if (std::string(name) == "multistart") k.push_back({"inner", inner});
        const sc::Result r = sc::run(name, P, 1e-6, false, 0.0, 20, 1000000, 1, k, {}, 0, {});
        char line[200];
        std::snprintf(line, sizeof line, "%s, 1 ms an evaluation, max_seconds 0.2: %.3f s, %lld evaluations, time_up %s",
                      name, r.seconds, r.evaluations, setting(r, "time_up").c_str());
        check(r.seconds >= 0.2 && r.seconds < 2.0 && r.evaluations < 5000 && setting(r, "time_up") == "yes" &&
                  setting(r, "max_seconds") == "0.2" && !r.variables.empty(),
              line);
    }
}

static void as_before()
{
    const sc::Problem P = slow_sphere(0.0);
    const sc::Result a = sc::run("de", P, 1e-6, false, 0.0, 20, 2000, 3, {}, {}, 0, {});
    const sc::Result b = sc::run("de", P, 1e-6, false, 0.0, 20, 2000, 3, {{"max_seconds", number(3600.0)}}, {}, 0, {});
    check(setting(a, "time_up").empty() && a.seconds > 0.0 && a.variables == b.variables && a.objectives == b.objectives &&
              a.evaluations == b.evaluations && setting(b, "time_up") == "no",
          "without max_seconds no time setting; an hour's deadline: the run as without it, time_up = no");
}

static void refusals()
{
    const sc::Problem P = slow_sphere(0.0);
    auto refused = [&](const sc::Knob& k) {
        try {
            sc::run("de", P, 1e-6, false, 0.0, 20, 100, 1, {{"max_seconds", k}}, {}, 0, {});
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    sc::Knob w;
    w.is_word = true;
    w.word = "soon";
    check(refused(number(0.0)) && refused(number(-1.0)) && refused(w), "refused: max_seconds 0, -1, a word");
}

int main()
{
    ends_early();
    as_before();
    refusals();
    std::printf(failures ? "FAILED\n" : "all passed\n");
    return failures ? 1 : 0;
}
