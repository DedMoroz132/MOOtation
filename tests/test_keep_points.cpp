// SPDX-License-Identifier: Apache-2.0
// keep_points = best beyond ipop and bipop (task 24, §24.3.3; evaluator.hpp
// THE LEAN MODE, scalar.hpp POP-10, strategy.hpp STRAT-12):
//   * a run that asks for no point twice is the run with every point kept,
//     bit for bit: de, local_search; multistart and tiktak over lbfgsb the
//     same answer, a point two starts meet counted in both (STRAT-12);
//   * a run that does (a strategy's local search from a point of its own
//     sample, a population's repeated point) pays for the repeats and is
//     otherwise a run of the same method;
//   * multistart's 200 starts go in chunks of 64: one thread and three alike;
//   * halving and memetic refuse it;
//   * a large archive is warned of, once; a lean run is not.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/scalar/scalar.hpp>
#include <mootation/warn.hpp>

namespace sc = mootation::scalar;
using Vec = std::vector<double>;

static int failures = 0;

static void check(bool ok, const std::string& what)
{
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}

static sc::Problem rastrigin(std::size_t n)
{
    const double pi = 3.14159265358979323846;
    sc::Problem P;
    P.lo.assign(n, -5.12);
    P.hi.assign(n, 5.12);
    P.m = 1;
    P.evaluate = [pi](const Vec& x, Vec& f, Vec&) {
        double s = 10.0 * static_cast<double>(x.size());
        for (double v : x) s += v * v - 10.0 * mootation::dm::cos(2.0 * pi * v);
        f[0] = s;
    };
    P.gradient = [pi](const Vec& x, Vec& gf, sc::Rows&) {
        gf.assign(x.size(), 0.0);
        for (std::size_t i = 0; i < x.size(); ++i) gf[i] = 2.0 * x[i] + 20.0 * pi * mootation::dm::sin(2.0 * pi * x[i]);
    };
    return P;
}

static bool same(const sc::Result& a, const sc::Result& b)
{
    if (a.improvements.size() != b.improvements.size()) return false;
    for (std::size_t i = 0; i < a.improvements.size(); ++i)
        if (std::memcmp(&a.improvements[i], &b.improvements[i], sizeof a.improvements[i]) != 0) return false;
    return a.objectives == b.objectives && a.variables == b.variables && a.evaluations == b.evaluations &&
           a.grad_calls == b.grad_calls;
}

static sc::Knob word(const std::string& w) { return {true, w, 0.0}; }
static sc::Knob number(double v) { return {false, "", v}; }

int main()
{
    const sc::Knob L = word("{\"solver\":\"lbfgsb\"}");
    struct Case {
        std::string name;
        sc::Knobs knobs;
        bool repeats;                   // asks for a point twice: the lean run pays for it
    };
    sc::Problem H = rastrigin(3);
    H.local_search = [](const Vec& x, sc::LocalContext& ctx, const Vec&) {
        Vec y = x, f(1), g;
        for (int k = 0; k < 30; ++k) {
            y[0] = x[0] + 0.01 * k;
            ctx.problem.evaluate(y, f, g);
        }
        sc::LocalOutcome o;
        o.x = y;
        return o;
    };
    for (const Case& c : std::vector<Case>{{"multistart", {{"inner", L}, {"n_starts", number(8)}}, true},
                                           {"tiktak", {{"inner", L}, {"tiktak_n", number(40)}}, true},
                                           {"de", {}, false},
                                           {"local_search", {}, false},
                                           {"mlsl", {{"inner", L}}, true},
                                           {"basin_hopping", {{"inner", L}}, true},
                                           {"ils", {{"inner", L}}, true},
                                           {"ga_sbx", {}, true}}) {
        sc::Knobs lean = c.knobs;
        lean.push_back({"keep_points", word("best")});
        const sc::Problem& P = c.name == "local_search" ? H : rastrigin(3);
        sc::Start st;
        if (c.name == "local_search") {
            st.has_x0 = true;
            st.x0 = {1.0, 2.0, -1.0};
        }
        const sc::Result a = sc::run(c.name, P, 1e-6, false, 0.0, 20, 3000, 5, c.knobs, {}, 0, st);
        const sc::Result b = sc::run(c.name, P, 1e-6, false, 0.0, 20, 3000, 5, lean, {}, 0, st);
        bool listed = false;
        for (const auto& w : b.ignored) listed = listed || w == "keep_points";
        check(!listed, c.name + ": keep_points taken, not ignored");
        if (!c.repeats) check(same(a, b), c.name + ": keep_points = best, no point asked twice: the same run");
        else check(b.evaluations <= 3000 && !b.objectives.empty(), c.name + ": keep_points = best runs");
        if (c.name == "multistart" || c.name == "tiktak")
            check(a.objectives == b.objectives && a.variables == b.variables && b.evaluations >= a.evaluations,
                  c.name + ": the same answer, the points two starts met counted twice");
        std::printf("  %-13s all: f %-10.4g %5lld evaluations; best: f %-10.4g %5lld evaluations%s\n", c.name.c_str(),
                    a.objectives[0][0], a.evaluations, b.objectives[0][0], b.evaluations,
                    same(a, b) ? " (the same run)" : "");
    }

    // the single and local solvers (SINGLE-5, LOCAL-7); lbfgsb's gradients in
    // the lean mode's slots (a point's gradient never another's)
    {
        sc::Problem R;                                    // Rosenbrock's residuals, with their Jacobian
        R.lo = {-5.0, -5.0};
        R.hi = {5.0, 5.0};
        R.m = 1;
        R.residuals = [](const Vec& x, Vec& r) { r = {10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]}; };
        R.jacobian = [](const Vec& x, sc::Rows& J) { J = {{-20.0 * x[0], 10.0}, {-1.0, 0.0}}; };
        sc::Start from;
        from.has_x0 = true;
        from.x0 = {-1.2, 1.0};
        struct Single {
            std::string name;
            bool repeats;
        };
        for (const Single& c : std::vector<Single>{{"lbfgsb", false}, {"lm", true}, {"cmaes", false},
                                                   {"nelder_mead", true}, {"slsqp", true}}) {
            const sc::Problem& P = c.name == "lm" ? R : rastrigin(3);
            const sc::Knobs best{{"keep_points", word("best")}};
            for (bool x0 : {false, true}) {
                sc::Start st;
                if (x0) st = c.name == "lm" ? from : sc::Start{{}, {}, false, false, {1.0, 2.0, -1.0}, true, {}};
                const sc::Result a = sc::run(c.name, P, 1e-6, false, 0.0, 20, 3000, 5, {}, {}, 0, st);
                const sc::Result b = sc::run(c.name, P, 1e-6, false, 0.0, 20, 3000, 5, best, {}, 0, st);
                const std::string label = c.name + (x0 ? " from x0" : " from its plan");
                if (!c.repeats) check(same(a, b), label + ": keep_points = best is the same run");
                else check(!b.objectives.empty() && b.evaluations <= 3000, label + ": keep_points = best runs");
                bool listed = false;
                for (const auto& w : b.ignored) listed = listed || w == "keep_points";
                check(!listed, label + ": keep_points taken");
                std::printf("  %-26s all: f %-10.4g %5lld + %lld; best: f %-10.4g %5lld + %lld%s\n", label.c_str(),
                            a.objectives[0][0], a.evaluations, a.grad_calls, b.objectives[0][0], b.evaluations,
                            b.grad_calls, same(a, b) ? " (the same run)" : "");
            }
        }
        // a local search on ctx.problem's gradients in the lean mode
        sc::Problem G = rastrigin(3);
        G.local_search = [](const Vec& x, sc::LocalContext& ctx, const Vec&) {
            sc::Start s;
            s.has_x0 = true;
            s.x0 = x;
            const sc::Result r = sc::run("lbfgsb", ctx.problem, 1e-6, false, 0.0, 20, ctx.budget, 1, {}, {}, 0, s);
            sc::LocalOutcome o;
            o.x = r.variables.at(0);
            o.has_f = true;
            o.f = r.objectives.at(0).at(0);
            return o;
        };
        sc::Start st;
        st.has_x0 = true;
        st.x0 = {1.0, 2.0, -1.0};
        const sc::Result a = sc::run("lbfgsb", rastrigin(3), 1e-6, false, 0.0, 20, 3000, 5, {}, {}, 0, st);
        const sc::Result b =
            sc::run("local_search", G, 1e-6, false, 0.0, 20, 3000, 5, {{"keep_points", word("best")}}, {}, 0, st);
        check(same(a, b), "local_search over lbfgsb, keep_points = best: lbfgsb's run (the gradients by slot)");
    }

    // 200 starts in chunks of 64: one thread and three alike
    {
        const sc::Knobs k1{{"inner", L}, {"n_starts", number(200)}, {"keep_points", word("best")}};
        sc::Knobs k3 = k1;
        k3.push_back({"threads", number(3)});
        const sc::Knobs all{{"inner", L}, {"n_starts", number(200)}};
        const sc::Result a = sc::run("multistart", rastrigin(4), 1e-6, false, 0.0, 20, 20000, 3, k1, {}, 0, {});
        const sc::Result b = sc::run("multistart", rastrigin(4), 1e-6, false, 0.0, 20, 20000, 3, k3, {}, 0, {});
        const sc::Result c = sc::run("multistart", rastrigin(4), 1e-6, false, 0.0, 20, 20000, 3, all, {}, 0, {});
        check(same(a, b), "multistart, 200 starts, best: one thread = three");
        check(a.objectives == c.objectives && a.variables == c.variables && a.evaluations >= c.evaluations,
              "multistart, 200 starts: best's answer = all's");
    }

    // halving and memetic refuse it
    for (const std::string m : {"halving", "memetic"}) {
        sc::Knobs k{{"inner", L}, {"keep_points", word("best")}};
        if (m == "memetic") k.push_back({"local_solver", L});
        std::string why;
        try {
            sc::run(m, rastrigin(3), 1e-6, false, 0.0, 20, 1000, 5, k, {}, 0, {});
        } catch (const std::invalid_argument& e) {
            why = e.what();
        }
        check(why.find(m + ": keep_points = best is refused") != std::string::npos, m + ": refused: '" + why + "'");
    }

    // the warning of a large archive, once; none for a lean run
    {
        std::vector<std::string> said;
        const auto before = mootation::set_warn_handler([&said](const std::string& m) { said.push_back(m); });
        sc::run("nelder_mead", rastrigin(2), 1e-6, false, 0.0, 20, 100000000, 5, {}, {}, 0, {});
        check(said.size() == 1 && said[0].find("GiB (budget 100000000, 2 variables)") != std::string::npos,
              "a large archive warned of once" + (said.empty() ? std::string() : ": " + said[0]));
        said.clear();
        sc::Problem T = rastrigin(2);
        T.stop_at = 1e-3;                                // a long budget, ended by the target
        sc::run("de", T, 1e-6, false, 0.0, 20, 100000000, 5, {{"keep_points", word("best")}}, {}, 0, {});
        check(said.empty(), "keep_points = best: no warning");
        said.clear();
        sc::run("nelder_mead", rastrigin(2), 1e-6, false, 0.0, 20, 5000, 5, {}, {}, 0, {});
        check(said.empty(), "a small archive: no warning");
        mootation::set_warn_handler(before);
    }

    if (failures) {
        std::printf("test_keep_points: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_keep_points: all checks passed\n");
    return 0;
}
