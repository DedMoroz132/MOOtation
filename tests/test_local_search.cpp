// SPDX-License-Identifier: Apache-2.0
// THE USER'S LOCAL SEARCH (task 24, §24.1.1; scalar.hpp run_user_local,
// LS-1 ... LS-6): Problem::local_search as the solver "local_search".
//   * a hook that runs the library's lbfgsb on ctx.problem is lbfgsb, bit for
//     bit: alone and as the inner solver of multistart (one thread and
//     three), basin_hopping, ils, mlsl and memetic's local_solver;
//   * the hook's calls of the problem are the run's evaluations, past the
//     budget the search ends there;
//   * a declared cost is spent exactly, a hook without calls and cost spends
//     nothing; the calls, the cost and the time are counted apart, a
//     strategy's starts' added up;
//   * x' without f is evaluated, with f planted;
//   * the state goes in and comes back;
//   * a stochastic hook on ctx.rng: one thread and three alike;
//   * x' of the wrong length, not finite, or an exception: refused (LS-4).

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/random.hpp>
#include <mootation/scalar/scalar.hpp>

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

// Rastrigin in [-5.12, 5.12]^3 with its gradient; calls counted
static sc::Problem rastrigin(std::shared_ptr<long long> calls)
{
    const double pi = 3.14159265358979323846;
    sc::Problem P;
    P.lo.assign(3, -5.12);
    P.hi.assign(3, 5.12);
    P.m = 1;
    P.evaluate = [calls, pi](const Vec& x, Vec& f, Vec&) {
        if (calls) ++*calls;
        double s = 30.0;
        for (double v : x) s += v * v - 10.0 * mootation::dm::cos(2.0 * pi * v);
        f[0] = s;
    };
    P.gradient = [pi](const Vec& x, Vec& gf, sc::Rows&) {
        gf.assign(x.size(), 0.0);
        for (std::size_t i = 0; i < x.size(); ++i) gf[i] = 2.0 * x[i] + 20.0 * pi * mootation::dm::sin(2.0 * pi * x[i]);
    };
    return P;
}

// the hook: the library's lbfgsb from x on the counted problem
static sc::LocalOutcome lbfgsb_hook(const Vec& x, sc::LocalContext& ctx, const Vec&)
{
    sc::Start st;
    st.has_x0 = true;
    st.x0 = x;
    const sc::Result r = sc::run("lbfgsb", ctx.problem, 1e-6, false, 0.0, 20, ctx.budget, 1, {}, {}, 0, st);
    sc::LocalOutcome out;
    out.x = r.variables.at(0);
    out.has_f = true;
    out.f = r.objectives.at(0).at(0);
    return out;
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
    const std::string L = "{\"solver\":\"lbfgsb\"}", U = "{\"solver\":\"local_search\"}";
    sc::Start from;
    from.has_x0 = true;
    from.x0 = {2.2, -3.1, 0.7};

    // the hook that is lbfgsb: alone and inside the strategies, bit for bit
    {
        sc::Problem A = rastrigin(nullptr), B = rastrigin(nullptr);
        B.local_search = lbfgsb_hook;
        const sc::Result a = sc::run("lbfgsb", A, 1e-6, false, 0.0, 20, 3000, 5, {}, {}, 0, from);
        const sc::Result b = sc::run("local_search", B, 1e-6, false, 0.0, 20, 3000, 5, {}, {}, 0, from);
        check(same(a, b) && b.hook_calls == 1 && b.hook_spent == 0.0, "local_search = lbfgsb, alone");
        struct Case {
            std::string name;
            sc::Knobs extra;
        };
        for (const Case& c : std::vector<Case>{{"multistart", {{"n_starts", number(6)}}},
                                               {"multistart", {{"n_starts", number(6)}, {"threads", number(3)}}},
                                               {"basin_hopping", {}},
                                               {"ils", {}},
                                               {"mlsl", {}},
                                               {"memetic", {}}}) {
            sc::Knobs ka, kb;
            const bool memetic = c.name == "memetic";
            ka.push_back({"inner", word(memetic ? "{\"solver\":\"de\"}" : L)});
            kb.push_back({"inner", word(memetic ? "{\"solver\":\"de\"}" : U)});
            if (memetic) {
                ka.push_back({"local_solver", word(L)});
                kb.push_back({"local_solver", word(U)});
            }
            for (const auto& kv : c.extra) {
                ka.push_back(kv);
                kb.push_back(kv);
            }
            const sc::Result ra = sc::run(c.name, A, 1e-6, false, 0.0, 20, 4000, 5, ka, {}, 0, from);
            const sc::Result rb = sc::run(c.name, B, 1e-6, false, 0.0, 20, 4000, 5, kb, {}, 0, from);
            const std::string label = c.name + (c.extra.size() > 1 ? " (3 threads)" : "");
            check(same(ra, rb) && rb.hook_calls >= 1,
                  label + ": inner local_search = lbfgsb, bit for bit (" + std::to_string(rb.hook_calls) + " calls)");
            std::printf("  %-26s evaluations %5lld, f %.6g, the hook called %lld times\n", label.c_str(),
                        rb.evaluations, rb.objectives[0][0], rb.hook_calls);
        }
    }

    // the hook's calls are the run's evaluations; past the budget it stops there
    {
        auto calls = std::make_shared<long long>(0);
        sc::Problem P = rastrigin(calls);
        P.local_search = [](const Vec& x, sc::LocalContext& ctx, const Vec&) {
            Vec y = x, f(1), g;
            for (int k = 0; k < 1000000; ++k) {                 // until the budget ends it
                y[0] = x[0] + 1e-3 * k;
                ctx.problem.evaluate(y, f, g);
            }
            sc::LocalOutcome out;
            out.x = y;
            return out;
        };
        const sc::Result r = sc::run("local_search", P, 1e-6, false, 0.0, 20, 500, 5, {}, {}, 0, from);
        check(r.evaluations == 500 && *calls == 500, "the hook's calls counted, the budget ends the search");
        check(!r.objectives.empty() && r.ls_state.empty(), "the archive's best is the answer");
    }

    // a declared cost, spent exactly; x' with f planted; a strategy adds the starts' counts
    {
        auto calls = std::make_shared<long long>(0);
        sc::Problem P = rastrigin(calls);
        P.local_search = [](const Vec& x, sc::LocalContext&, const Vec& state) {
            sc::LocalOutcome out;                                // a search of its own (B2-like): no f calls
            out.x = x;
            for (double& v : out.x) v *= 0.5;
            out.has_f = true;
            double s = 30.0;
            for (double v : out.x) s += v * v - 10.0 * mootation::dm::cos(2.0 * 3.14159265358979323846 * v);
            out.f = s;
            out.cost = 7.0;
            out.state = state;
            out.state.push_back(static_cast<double>(state.size()) + 1.0);
            return out;
        };
        sc::Start st = from;
        st.ls_state = {10.0, 20.0};
        const sc::Result r = sc::run("local_search", P, 1e-6, false, 0.0, 20, 100, 5, {}, {}, 0, st);
        check(r.evaluations == 0 && *calls == 0 && r.hook_spent == 7.0 && r.hook_calls == 1,
              "a cost of 7 declared: 7 spent, no evaluation (x' planted)");
        check(r.ls_state == Vec({10.0, 20.0, 3.0}), "the state goes in and comes back");
        check(r.variables[0] == Vec({1.1, -1.55, 0.35}), "x' is the answer");
        *calls = 0;
        const sc::Knobs k{{"inner", word(U)}, {"n_starts", number(4)}};
        const sc::Result m = sc::run("multistart", P, 1e-6, false, 0.0, 20, 1000, 5, k, {}, 0, from);
        check(m.hook_calls == 4 && m.hook_spent == 28.0 && m.evaluations == 0 && *calls == 0,
              "multistart: 4 starts, 28 spent, the counts added up");
        // no cost, no f: x' is evaluated, nothing else is spent
        P.local_search = [](const Vec& x, sc::LocalContext&, const Vec&) {
            sc::LocalOutcome out;
            out.x = x;
            out.x[1] = 0.0;
            return out;
        };
        *calls = 0;
        const sc::Result z = sc::run("local_search", P, 1e-6, false, 0.0, 20, 100, 5, {}, {}, 0, from);
        check(z.evaluations == 1 && *calls == 1 && z.hook_spent == 0.0, "x' without f: evaluated once, no more");
    }

    // a stochastic hook on ctx.rng: one thread and three alike
    {
        sc::Problem P = rastrigin(nullptr);
        P.local_search = [](const Vec& x, sc::LocalContext& ctx, const Vec&) {
            Vec best = x, f(1), g;
            ctx.problem.evaluate(best, f, g);
            double fb = f[0];
            for (int k = 0; k < 40; ++k) {                       // a random walk of 40 steps
                Vec y = best;
                for (double& v : y) v = std::min(5.12, std::max(-5.12, v + 0.3 * mootation::rnd::normal01(ctx.rng)));
                ctx.problem.evaluate(y, f, g);
                if (f[0] < fb) {
                    fb = f[0];
                    best = y;
                }
            }
            sc::LocalOutcome out;
            out.x = best;
            out.has_f = true;
            out.f = fb;
            return out;
        };
        const sc::Knobs k1{{"inner", word(U)}, {"n_starts", number(8)}};
        const sc::Knobs k3{{"inner", word(U)}, {"n_starts", number(8)}, {"threads", number(3)}};
        const sc::Result a = sc::run("multistart", P, 1e-6, false, 0.0, 20, 2000, 9, k1, {}, 0, from);
        const sc::Result b = sc::run("multistart", P, 1e-6, false, 0.0, 20, 2000, 9, k3, {}, 0, from);
        check(same(a, b) && a.hook_calls == b.hook_calls && a.hook_calls == 8, "a random hook: one thread = three");
    }

    // memetic's lamarck (search.hpp SRCH-12): the local searches' answers in
    // the global solver's later runs. The global solver here is a hook that
    // writes down the point it starts from, then walks at random until its
    // allowance ends; the local solver is lbfgsb.
    {
        struct Seen {
            std::vector<Vec> starts;
        };
        auto memetic = [&](const std::string& lamarck, double share, std::shared_ptr<Seen> seen, int threads) {
            sc::Problem P = rastrigin(nullptr);
            P.local_search = [seen](const Vec& x, sc::LocalContext& ctx, const Vec&) -> sc::LocalOutcome {
                if (seen) seen->starts.push_back(x);
                Vec y = x, f(1), g;
                for (;;) {                                   // until the allowance ends it
                    for (double& v : y)
                        v = std::min(5.12, std::max(-5.12, v + 0.5 * mootation::rnd::normal01(ctx.rng)));
                    ctx.problem.evaluate(y, f, g);
                }
            };
            sc::Knobs k{{"inner", word(U)}, {"local_solver", word(L)}, {"memetic_every", number(60)},
                        {"threads", number(threads)}};
            if (!lamarck.empty()) k.push_back({"lamarck", word(lamarck)});
            if (lamarck == "share") k.push_back({"lamarck_share", number(share)});
            return sc::run("memetic", P, 1e-6, false, 0.0, 20, 900, 5, k, {}, 0, from);
        };
        auto off = std::make_shared<Seen>(), on = std::make_shared<Seen>();
        const sc::Result base = memetic("", 0.0, nullptr, 1);
        const sc::Result r_off = memetic("off", 0.0, off, 1);
        const sc::Result r_on = memetic("on", 0.0, on, 1);
        check(same(base, r_off), "lamarck = off, given, is the default run");
        check(off->starts.size() >= 2 && off->starts[1] == from.x0, "lamarck = off: the global solver starts anew");
        bool improved = on->starts.size() >= 2 && on->starts[0] == from.x0 && on->starts[1] != from.x0;
        if (improved) {
            Vec f(1), g;
            rastrigin(nullptr).evaluate(on->starts[1], f, g);
            Vec f0(1);
            rastrigin(nullptr).evaluate(from.x0, f0, g);
            improved = f[0] < f0[0];
        }
        check(improved, "lamarck = on: the global solver's next run starts from the local search's answer");
        const sc::Result r_zero = memetic("share", 0.0, nullptr, 1);
        check(same(base, r_zero), "lamarck = share at 0: nothing written, the default run");
        const sc::Result a = memetic("share", 0.5, nullptr, 1), b = memetic("share", 0.5, nullptr, 3);
        check(same(a, b), "lamarck = share 0.5: one thread and three alike");
        std::printf("  memetic, a random global walk + lbfgsb: f %.4g (off), %.4g (on), %.4g (share 0.5)\n",
                    r_off.objectives[0][0], r_on.objectives[0][0], a.objectives[0][0]);
    }

    // the mistakes (LS-4)
    auto refused = [&](std::function<sc::LocalOutcome(const Vec&, sc::LocalContext&, const Vec&)> hook,
                       const std::string& part) {
        sc::Problem P = rastrigin(nullptr);
        P.local_search = std::move(hook);
        std::string why;
        try {
            sc::run("local_search", P, 1e-6, false, 0.0, 20, 100, 5, {}, {}, 0, from);
        } catch (const std::exception& e) {
            why = e.what();
        }
        check(why.find(part) != std::string::npos, "refused: '" + why + "' (wanted: " + part + ")");
    };
    refused([](const Vec&, sc::LocalContext&, const Vec&) { return sc::LocalOutcome{{1.0}, false, 0.0, 0.0, {}}; },
            "x' of 1 values for 3 variables");
    refused([](const Vec& x, sc::LocalContext&, const Vec&) {
        sc::LocalOutcome o;
        o.x = x;
        o.x[2] = std::nan("");
        return o;
    }, "not finite");
    refused([](const Vec&, sc::LocalContext&, const Vec&) -> sc::LocalOutcome { throw std::runtime_error("no B2"); },
            "the hook failed: no B2");
    {
        std::string why;
        try {
            sc::run("local_search", rastrigin(nullptr), 1e-6, false, 0.0, 20, 100, 5, {}, {}, 0, from);
        } catch (const std::exception& e) {
            why = e.what();
        }
        check(why.find("gives no local_search hook") != std::string::npos, "no hook: refused");
    }

    if (failures) {
        std::printf("test_local_search: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_local_search: all checks passed\n");
    return 0;
}
