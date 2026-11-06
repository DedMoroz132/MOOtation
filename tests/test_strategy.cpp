// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/strategy.hpp (task 18, stage 3, §3.1-3.2): a strategy is a solver.
//   - one start is the inner solver's run, bit for bit (§3.1): multistart
//     with n_starts = 1 against cmaes, nelder_mead, de and lbfgsb alone;
//   - 1 thread and 12 give the same run, bit for bit (§3.2);
//   - THE DEFERRED COUNT: what the problem's function hands to
//     defer_or_run() happens in the order of 1 thread, with 12 threads and
//     with a strategy in threads within one in threads;
//   - a round's views: a point two starts evaluated counts once, for the
//     first; a point of an earlier round is free;
//   - the minima log finds the two minima of (x^2 - 1)^2 + y^2;
//   - nesting: a multistart within a multistart; refusals.
// ============================================================================

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <mootation/scalar/scalar.hpp>

namespace {

namespace sc = mootation::scalar;
using Vec = std::vector<double>;

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    if (!ok) {
        ++g_failed;
        std::printf("FAIL  %s\n", what.c_str());
    } else {
        std::printf("ok    %s\n", what.c_str());
    }
}

// the weighted sphere sum (i + 1)(x_i - 0.3)^2 in [-1, 1]^4, its calls counted
sc::Problem sphere(std::shared_ptr<long long> calls, bool with_gradient = false)
{
    sc::Problem P;
    P.lo.assign(4, -1.0);
    P.hi.assign(4, 1.0);
    P.m = 1;
    P.evaluate = [calls](const Vec& x, Vec& f, Vec&) {
        ++*calls;
        double s = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) s += (static_cast<double>(i) + 1.0) * (x[i] - 0.3) * (x[i] - 0.3);
        f.assign(1, s);
    };
    if (with_gradient)
        P.gradient = [](const Vec& x, Vec& gf, sc::Rows&) {
            gf.assign(x.size(), 0.0);
            for (std::size_t i = 0; i < x.size(); ++i) gf[i] = 2.0 * (static_cast<double>(i) + 1.0) * (x[i] - 0.3);
        };
    return P;
}

sc::Knob word(const std::string& w) { return {true, w, 0.0}; }
sc::Knob number(double v) { return {false, "", v}; }

bool same(const sc::Result& a, const sc::Result& b)
{
    if (a.improvements.size() != b.improvements.size()) return false;
    for (std::size_t i = 0; i < a.improvements.size(); ++i)
        if (std::memcmp(&a.improvements[i], &b.improvements[i], sizeof a.improvements[i]) != 0) return false;
    return a.objectives == b.objectives && a.variables == b.variables && a.evaluations == b.evaluations &&
           a.grad_calls == b.grad_calls;
}

void one_start_is_the_inner_run()
{
    struct Case {
        std::string inner, json;
        bool gradient;
    };
    for (const Case& c : std::vector<Case>{{"cmaes", "{\"solver\":\"cmaes\"}", false},
                                           {"nelder_mead", "{\"solver\":\"nelder_mead\"}", false},
                                           {"de", "{\"solver\":\"de\"}", false},
                                           {"lbfgsb", "{\"solver\":\"lbfgsb\"}", true}}) {
        auto c1 = std::make_shared<long long>(0), c2 = std::make_shared<long long>(0);
        const sc::Result alone = sc::run(c.inner, sphere(c1, c.gradient), 1e-6, false, 0.0, 20, 3000, 7, {}, {}, 0, {});
        const sc::Knobs kn{{"inner", word(c.json)}, {"n_starts", number(1)}};
        const sc::Result ms = sc::run("multistart", sphere(c2, c.gradient), 1e-6, false, 0.0, 20, 3000, 7, kn, {}, 0, {});
        check(same(alone, ms) && *c1 == *c2,
              "multistart, one start: " + c.inner + "'s own run bit for bit (" + std::to_string(ms.evaluations) +
                  " evaluations, " + std::to_string(ms.improvements.size()) + " records)");
    }
}

void threads_change_nothing()
{
    sc::Result r[2];
    std::string minima[2];
    int k = 0;
    for (int threads : {1, 12}) {
        auto calls = std::make_shared<long long>(0);
        const sc::Knobs kn{{"inner", word("{\"cma_sigma0\":0.2,\"solver\":\"cmaes\"}")}, {"n_starts", number(8)},
                           {"threads", number(threads)}};
        r[k] = sc::run("multistart", sphere(calls), 1e-6, false, 0.0, 20, 8000, 11, kn, {}, 0, {});
        for (const auto& [key, v] : r[k].settings)
            if (key == "minima") minima[k] = v;
        ++k;
    }
    check(same(r[0], r[1]) && minima[0] == minima[1] && !minima[0].empty() && r[0].evaluations <= 8000,
          "multistart of 8 cmaes: 1 thread and 12 the same run (" + std::to_string(r[0].evaluations) + " evaluations)");
}

// THE DEFERRED COUNT: the problem's function logs each x through
// defer_or_run() into a plain vector (no lock): in threads it is filled at the
// rounds' ends, in the order of the starts, as 1 thread fills it at once
void the_count_is_deferred()
{
    auto run = [](const std::string& inner, int threads, std::vector<Vec>& log, long long& calls) {
        auto made = std::make_shared<std::atomic<long long>>(0);
        sc::Problem P = sphere(std::make_shared<long long>(0));
        const auto f = P.evaluate;
        P.evaluate = [f, made, &log](const Vec& x, Vec& fx, Vec& g) {
            ++*made;
            sc::strategy::defer_or_run([&log, x] { log.push_back(x); });
            f(x, fx, g);
        };
        const sc::Knobs kn{{"inner", word(inner)}, {"n_starts", number(6)}, {"threads", number(threads)}};
        const sc::Result r = sc::run("multistart", P, 1e-6, false, 0.0, 20, 6000, 3, kn, {}, 0, {});
        calls = *made;
        return r;
    };
    const std::string cma = "{\"cma_sigma0\":0.2,\"solver\":\"cmaes\"}";
    std::vector<Vec> one, twelve;
    long long c1 = 0, c12 = 0;
    const sc::Result r1 = run(cma, 1, one, c1), r12 = run(cma, 12, twelve, c12);
    check(same(r1, r12) && one == twelve && static_cast<long long>(one.size()) == c1 && c12 == c1,
          "multistart of 6 cmaes: the problem's deferred log in 12 threads is the log of 1 (" +
              std::to_string(one.size()) + " entries)");
    const std::string nested =
        "{\"inner\":{\"solver\":\"nelder_mead\"},\"n_starts\":3,\"solver\":\"multistart\",\"threads\":3}";
    const std::string nested1 = "{\"inner\":{\"solver\":\"nelder_mead\"},\"n_starts\":3,\"solver\":\"multistart\"}";
    std::vector<Vec> a, b;
    long long ca = 0, cb = 0;
    const sc::Result ra = run(nested1, 1, a, ca), rb = run(nested, 4, b, cb);
    check(same(ra, rb) && a == b && static_cast<long long>(a.size()) == ca && cb == ca,
          "a multistart in 3 threads within one in 4: the deferred log of 1 thread (" + std::to_string(a.size()) +
              " entries)");
}

void a_point_counts_once()
{
    auto calls = std::make_shared<long long>(0);
    const sc::Problem P = sphere(calls);
    sc::Evaluator ev(P, 1e-6, false, 0.0, 100, 1, {}, 0);
    const Vec x{0.1, 0.2, 0.3, 0.4}, y{0.0, 0.0, 0.0, 0.0};
    {
        sc::strategy::View a(ev), b(ev);
        Vec f, g;
        a.problem.evaluate(x, f, g);
        b.problem.evaluate(x, f, g);
        b.problem.evaluate(y, f, g);
        a.merge_into(ev);
        b.merge_into(ev);
    }
    const long long after_one = ev.spent, calls_one = *calls;
    sc::strategy::View c(ev);
    Vec f, g;
    c.problem.evaluate(x, f, g);
    c.merge_into(ev);
    check(after_one == 2 && calls_one == 3 && ev.spent == 2 && *calls == 3 && ev.X.size() == 2,
          "a round: x evaluated by two starts counts once (spent " + std::to_string(after_one) + ", calls " +
              std::to_string(calls_one) + "); the next round gets it free");
}

void two_minima()
{
    sc::Problem P;
    P.lo = {-2.0, -2.0};
    P.hi = {2.0, 2.0};
    P.m = 1;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) { f.assign(1, (x[0] * x[0] - 1.0) * (x[0] * x[0] - 1.0) + x[1] * x[1]); };
    const sc::Knobs kn{{"inner", word("{\"solver\":\"nelder_mead\",\"local_tol\":1e-12}")}, {"n_starts", number(10)},
                       {"min_dist", number(1e-3)}};
    const sc::Result r = sc::run("multistart", P, 1e-6, false, 0.0, 10, 20000, 3, kn, {}, 0, {});
    std::string minima;
    for (const auto& [key, v] : r.settings)
        if (key == "minima") minima = v;
    const mootation::json::Value a = mootation::json::parse(minima);
    bool both = a.items.size() == 2;
    long long found = 0;
    for (const auto& m : a.items) {
        both = both && m.items[0].f < 1e-10 && std::fabs(std::fabs(m.items[3].f) - 1.0) < 1e-5;
        found += m.items[2].i;
    }
    check(both && found == 10 && r.objectives[0][0] < 1e-10,
          "multistart of 10 nelder_mead on (x^2 - 1)^2 + y^2: two minima, x = +-1, every start in one: " + minima.substr(0, 80));
}

void nesting_and_refusals()
{
    auto calls = std::make_shared<long long>(0);
    const sc::Knobs kn{{"inner", word("{\"inner\":{\"solver\":\"nelder_mead\"},\"n_starts\":2,\"solver\":\"multistart\"}")},
                       {"n_starts", number(3)}};
    const sc::Result r = sc::run("multistart", sphere(calls), 1e-6, false, 0.0, 20, 6000, 5, kn, {}, 0, {});
    check(!r.objectives.empty() && r.objectives[0][0] < 1e-8 && r.evaluations <= 6000 && *calls >= r.evaluations,
          "a multistart of multistarts: f = " + std::to_string(r.objectives[0][0]) + " in " +
              std::to_string(r.evaluations) + " evaluations (" + std::to_string(*calls) + " calls)");
    std::string why;
    try {
        sc::run("multistart", sphere(calls), 1e-6, false, 0.0, 20, 600, 5, {}, {}, 0, {});
    } catch (const std::invalid_argument& e) {
        why = e.what();
    }
    check(why.find("needs its inner solver") != std::string::npos, "multistart without inner: " + why);
    sc::Problem u = sphere(calls);
    u.hi[1] = std::numeric_limits<double>::infinity();
    why.clear();
    try {
        sc::run("multistart", u, 1e-6, false, 0.0, 20, 600, 5, {{"inner", word("{\"solver\":\"cmaes\"}")}}, {}, 0, {});
    } catch (const std::invalid_argument& e) {
        why = e.what();
    }
    check(why.find("every bound must be finite") != std::string::npos, "multistart with an infinite bound: " + why);
}

}   // namespace

int main()
{
    one_start_is_the_inner_run();
    threads_change_nothing();
    the_count_is_deferred();
    a_point_counts_once();
    two_minima();
    nesting_and_refusals();
    std::printf("%s (%d failed)\n", g_failed == 0 ? "PASS" : "FAIL", g_failed);
    return g_failed == 0 ? 0 : 1;
}
