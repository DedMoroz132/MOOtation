// SPDX-License-Identifier: Apache-2.0
// THE SUCCESS (task 24, §24.1.5; scalar/evaluator.hpp): the user's measure of
// success in place of stop_at's target.
//   * accept_success accepting every point with f <= post_at = stop_at is
//     stop_at's target, bit for bit: the answer, the counts, the records —
//     single methods, population methods, strategies on one thread and on
//     three, and the lean evaluator (ipop with keep_points = best);
//   * a stricter measure goes on past such points, asked at each one once;
//   * its declared cost is spent from the budget, nothing else is;
//   * multistart on one thread and on three asks it as often and ends alike;
//   * an exception of the hook ends the run with its message (SUCC-4).

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

// a shifted sphere in [-1, 1]^5, with its gradient
static sc::Problem sphere()
{
    sc::Problem P;
    P.lo.assign(5, -1.0);
    P.hi.assign(5, 1.0);
    P.m = 1;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) {
        double s = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) s += (static_cast<double>(i) + 1.0) * (x[i] - 0.3) * (x[i] - 0.3);
        f[0] = s;
    };
    P.gradient = [](const Vec& x, Vec& gf, sc::Rows&) {
        gf.assign(x.size(), 0.0);
        for (std::size_t i = 0; i < x.size(); ++i) gf[i] = 2.0 * (static_cast<double>(i) + 1.0) * (x[i] - 0.3);
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
    const double target = 1e-6;
    struct Case {
        std::string name;
        sc::Knobs knobs;
    };
    const std::vector<Case> cases = {
        {"nelder_mead", {}},
        {"cmaes", {}},
        {"lbfgsb", {}},
        {"de", {}},
        {"pso", {}},
        {"multistart", {{"inner", word("{\"solver\":\"lbfgsb\"}")}, {"n_starts", number(6)}}},
        {"multistart", {{"inner", word("{\"solver\":\"nelder_mead\"}")}, {"n_starts", number(6)}, {"threads", number(3)}}},
        {"basin_hopping", {{"inner", word("{\"solver\":\"lbfgsb\"}")}}},
        {"memetic", {{"inner", word("{\"solver\":\"de\"}")}, {"local_solver", word("{\"solver\":\"lbfgsb\"}")}}},
        {"ipop", {{"keep_points", word("best")}}},
    };
    for (const Case& c : cases) {
        bool threads = false;
        for (const auto& [key, v] : c.knobs) threads = threads || key == "threads";
        const std::string label = c.name + (threads ? " (3 threads)" : "");
        sc::Problem A = sphere();
        A.stop_at = target;
        sc::Problem B = sphere();
        auto asked = std::make_shared<long long>(0);
        B.accept_success = [asked](const Vec&) {
            ++*asked;
            return true;
        };
        B.post_at = target;
        const sc::Result ra = sc::run(c.name, A, 1e-6, false, 0.0, 20, 6000, 5, c.knobs, {}, 0, {});
        const sc::Result rb = sc::run(c.name, B, 1e-6, false, 0.0, 20, 6000, 5, c.knobs, {}, 0, {});
        check(same(ra, rb), label + ": accept_success = (f <= stop_at) is stop_at's target, bit for bit");
        const bool hit = !ra.objectives.empty() && ra.objectives[0][0] <= target;
        check(ra.success_calls == 0 && ra.success_at == -1, label + ": without the hook nothing is asked");
        check(hit == (rb.success_at >= 0), label + ": the success is stop_at's hit");
        check(!hit || (rb.success_at == rb.evaluations && rb.success_calls >= 1),
              label + ": accepted at the run's last evaluation");
        check(rb.success_spent == 0.0, label + ": no cost declared, none spent");
        std::printf("  %-28s evaluations %5lld, target %s, asked %lld (counted %lld)\n", label.c_str(),
                    rb.evaluations, hit ? "hit" : "missed", *asked, rb.success_calls);
    }

    // a stricter measure: only points near x = 0.3 in every coordinate,
    // f <= post_at being a loose 1e-2; the run goes past the loose points
    {
        sc::Problem P = sphere();
        P.post_at = 1e-2;
        P.accept_success = [](const Vec& x) {
            for (double v : x)
                if (v < 0.3 - 1e-4 || v > 0.3 + 1e-4) return false;
            return true;
        };
        const sc::Result r = sc::run("nelder_mead", P, 1e-6, false, 0.0, 20, 6000, 5, {}, {}, 0, {});
        sc::Problem Q = sphere();
        Q.stop_at = 1e-2;
        const sc::Result q = sc::run("nelder_mead", Q, 1e-6, false, 0.0, 20, 6000, 5, {}, {}, 0, {});
        check(q.success_at == -1 && r.evaluations > q.evaluations, "stricter: the run goes past f <= post_at");
        check(r.success_calls > 1, "stricter: asked at each point with f <= post_at");
        check(r.success_at < 0 || r.success_at == r.evaluations, "stricter: ends where it accepts");
        std::printf("  stricter measure: stop_at 1e-2 ends at %lld, the measure at %lld (asked %lld times)\n",
                    q.evaluations, r.evaluations, r.success_calls);
    }

    // the declared cost: every feasible point asked (post_at = +inf), none
    // accepted; each call spends 0.5 of the budget, so 1.5 an evaluation
    {
        sc::Problem P = sphere();
        P.accept_success = [](const Vec&) { return false; };
        P.success_cost = 0.5;
        const sc::Result r = sc::run("de", P, 1e-6, false, 0.0, 20, 3000, 5, {}, {}, 0, {});
        check(r.success_calls == r.evaluations, "cost: every evaluation asked once");
        check(r.success_spent == 0.5 * static_cast<double>(r.success_calls), "cost: 0.5 a call");
        check(static_cast<double>(r.evaluations) + r.success_spent <= 3000.0 + 0.5 &&
                  static_cast<double>(r.evaluations) + r.success_spent > 3000.0 - 1.5 - 1e-9,
              "cost: evaluations + cost = the budget (SUCC-2)");
        check(r.success_at == -1, "cost: never accepted");
        P.success_cost = 0.0;
        sc::Problem Q = sphere();
        const sc::Result q = sc::run("de", Q, 1e-6, false, 0.0, 20, 3000, 5, {}, {}, 0, {});
        const sc::Result z = sc::run("de", P, 1e-6, false, 0.0, 20, 3000, 5, {}, {}, 0, {});
        check(same(q, z), "no cost: a hook that never accepts spends nothing, the run as without it");
    }

    // a strategy on one thread and on three: the same calls and the same end
    {
        sc::Problem P = sphere();
        P.post_at = 1e-3;
        P.accept_success = [](const Vec& x) { return x[4] > 0.3 - 1e-3 && x[4] < 0.3 + 1e-3; };
        P.success_cost = 0.25;
        const sc::Knobs k1{{"inner", word("{\"solver\":\"nelder_mead\"}")}, {"n_starts", number(8)}, {"threads", number(1)}};
        const sc::Knobs k3{{"inner", word("{\"solver\":\"nelder_mead\"}")}, {"n_starts", number(8)}, {"threads", number(3)}};
        const sc::Result a = sc::run("multistart", P, 1e-6, false, 0.0, 20, 8000, 9, k1, {}, 0, {});
        const sc::Result b = sc::run("multistart", P, 1e-6, false, 0.0, 20, 8000, 9, k3, {}, 0, {});
        check(same(a, b) && a.success_calls == b.success_calls && a.success_at == b.success_at &&
                  a.success_spent == b.success_spent,
              "multistart: one thread and three ask alike and end alike");
        std::printf("  multistart, measure with cost 0.25: %lld evaluations, asked %lld, accepted at %lld\n",
                    a.evaluations, a.success_calls, a.success_at);
    }

    // an exception of the hook: the run ends with its message
    {
        sc::Problem P = sphere();
        P.accept_success = [](const Vec&) -> bool { throw std::runtime_error("no packing"); };
        std::string why;
        try {
            sc::run("nelder_mead", P, 1e-6, false, 0.0, 20, 1000, 5, {}, {}, 0, {});
        } catch (const std::exception& e) {
            why = e.what();
        }
        check(why.find("accept_success failed at evaluation") != std::string::npos &&
                  why.find("no packing") != std::string::npos && why.find("SUCC-4") != std::string::npos,
              "an exception: '" + why + "'");
    }

    if (failures) {
        std::printf("test_success: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_success: all checks passed\n");
    return 0;
}
