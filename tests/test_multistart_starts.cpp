// SPDX-License-Identifier: Apache-2.0
// multistart's budget by start and its starts about x0 (task 24, §24.2.5;
// strategy.hpp STRAT-3, STRAT-10):
//   * allowance = even, given, is the default run;
//   * allowance = sequential with one start is the inner solver's run; with
//     many, its first start is the inner solver's run with the whole budget
//     (its records first), the run ends at the budget or the target;
//   * start_points = around_x0 with start_sigma = 0: every start is x0, so
//     the run is the inner solver's from x0 (the other starts' points are in
//     the archive); with start_sigma > 0 the starts lie in the box, about x0,
//     and three threads give one thread's run; it works without a box;
//   * the mistakes are refused.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/scalar/scalar.hpp>

namespace sc = mootation::scalar;
using Vec = std::vector<double>;
using Rows = std::vector<Vec>;

static int failures = 0;

static void check(bool ok, const std::string& what)
{
    if (!ok) {
        ++failures;
        std::printf("FAIL %s\n", what.c_str());
    }
}

// Rastrigin in [-5.12, 5.12]^2 (or without a box), with its gradient; the
// points asked for are written down
static sc::Problem rastrigin(std::shared_ptr<Rows> asked, bool box = true)
{
    const double pi = 3.14159265358979323846;
    sc::Problem P;
    P.lo.assign(2, box ? -5.12 : -std::numeric_limits<double>::infinity());
    P.hi.assign(2, box ? 5.12 : std::numeric_limits<double>::infinity());
    P.m = 1;
    P.evaluate = [asked, pi](const Vec& x, Vec& f, Vec&) {
        if (asked) asked->push_back(x);
        double s = 20.0;
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
    const sc::Knob lbfgsb = word("{\"solver\":\"lbfgsb\"}");
    sc::Start from;
    from.has_x0 = true;
    from.x0 = {2.2, -3.1};

    // even, given: the default run
    {
        const sc::Knobs a{{"inner", lbfgsb}, {"n_starts", number(8)}};
        const sc::Knobs b{{"inner", lbfgsb}, {"n_starts", number(8)}, {"allowance", word("even")}};
        const sc::Result ra = sc::run("multistart", rastrigin(nullptr), 1e-6, false, 0.0, 10, 2000, 4, a, {}, 0, {});
        const sc::Result rb = sc::run("multistart", rastrigin(nullptr), 1e-6, false, 0.0, 10, 2000, 4, b, {}, 0, {});
        check(same(ra, rb), "allowance = even is the default run");
        bool listed = false;
        for (const auto& [key, v] : ra.settings) listed = listed || key == "allowance";
        check(!listed, "allowance in the settings only where given");
    }
    // sequential, one start: the inner solver's run
    {
        const sc::Knobs k{{"inner", lbfgsb}, {"n_starts", number(1)}, {"allowance", word("sequential")}};
        const sc::Result alone = sc::run("lbfgsb", rastrigin(nullptr), 1e-6, false, 0.0, 10, 2000, 4, {}, {}, 0, from);
        const sc::Result seq = sc::run("multistart", rastrigin(nullptr), 1e-6, false, 0.0, 10, 2000, 4, k, {}, 0, from);
        check(same(alone, seq), "sequential with one start is the inner solver's run");
    }
    // sequential, many starts: the first start the inner solver's run with
    // the whole budget, its records first; the budget spent, not passed
    {
        const sc::Knobs k{{"inner", lbfgsb}, {"n_starts", number(30)}, {"allowance", word("sequential")}};
        const sc::Result alone = sc::run("lbfgsb", rastrigin(nullptr), 1e-6, false, 0.0, 10, 2000, 4, {}, {}, 0, from);
        const sc::Result seq = sc::run("multistart", rastrigin(nullptr), 1e-6, false, 0.0, 10, 2000, 4, k, {}, 0, from);
        bool prefix = seq.improvements.size() >= alone.improvements.size();
        for (std::size_t i = 0; prefix && i < alone.improvements.size(); ++i)
            prefix = std::memcmp(&alone.improvements[i], &seq.improvements[i], sizeof alone.improvements[i]) == 0;
        check(prefix, "sequential: the first start's records are the inner run's");
        check(seq.evaluations + seq.grad_calls <= 2000 && seq.evaluations > alone.evaluations,
              "sequential: the next starts take the rest of the budget");
        const sc::Knobs e{{"inner", lbfgsb}, {"n_starts", number(30)}};
        const sc::Result even = sc::run("multistart", rastrigin(nullptr), 1e-6, false, 0.0, 10, 2000, 4, e, {}, 0, from);
        std::printf("  rastrigin 2-D, 30 starts of lbfgsb, budget 2000: even f = %.3g, sequential f = %.3g "
                    "(%lld + %lld)\n",
                    even.objectives[0][0], seq.objectives[0][0], seq.evaluations, seq.grad_calls);
        sc::Problem T = rastrigin(nullptr);
        T.stop_at = 1e-8;
        const sc::Result hit = sc::run("multistart", T, 1e-6, false, 0.0, 10, 20000, 4, k, {}, 0, from);
        check(hit.objectives[0][0] <= 1e-8 && hit.evaluations + hit.grad_calls < 20000,
              "sequential: ends at the target");
    }
    // around_x0, start_sigma = 0: every start x0
    {
        const sc::Knobs k{{"inner", lbfgsb}, {"n_starts", number(5)}, {"start_points", word("around_x0")},
                          {"start_sigma", number(0.0)}};
        const sc::Result alone = sc::run("lbfgsb", rastrigin(nullptr), 1e-6, false, 0.0, 10, 2000, 4, {}, {}, 0, from);
        const sc::Result ms = sc::run("multistart", rastrigin(nullptr), 1e-6, false, 0.0, 10, 2000, 4, k, {}, 0, from);
        check(alone.evaluations == ms.evaluations && alone.objectives == ms.objectives &&
                  alone.variables == ms.variables,
              "around_x0 (0): every start is x0, the inner solver's run");
    }
    // around_x0, start_sigma = 0.05: about x0, in the box; one thread = three
    {
        auto asked = std::make_shared<Rows>();
        const sc::Knobs k1{{"inner", word("{\"solver\":\"nelder_mead\"}")}, {"n_starts", number(6)},
                           {"start_points", word("around_x0")}, {"start_sigma", number(0.05)}};
        sc::Knobs k3 = k1;
        k3.push_back({"threads", number(3)});
        const sc::Result a = sc::run("multistart", rastrigin(asked), 1e-6, false, 0.0, 10, 3000, 4, k1, {}, 0, from);
        const sc::Result b = sc::run("multistart", rastrigin(nullptr), 1e-6, false, 0.0, 10, 3000, 4, k3, {}, 0, from);
        check(same(a, b), "around_x0: three threads give one thread's run");
        bool inside = true, close = true;
        for (const Vec& x : *asked)
            for (std::size_t j = 0; j < 2; ++j) {
                inside = inside && x[j] >= -5.12 && x[j] <= 5.12;
                close = close && std::fabs(x[j] - from.x0[j]) < 5.0;
            }
        check(inside && close, "around_x0: the points in the box, about x0");
        // without a box: start_sigma in the coordinates' own units
        const sc::Knobs u{{"inner", lbfgsb}, {"n_starts", number(4)}, {"start_points", word("around_x0")},
                          {"start_sigma", number(0.5)}};
        const sc::Result r = sc::run("multistart", rastrigin(nullptr, false), 1e-6, false, 0.0, 10, 2000, 4, u, {}, 0, from);
        check(!r.objectives.empty() && std::isfinite(r.objectives[0][0]), "around_x0 without a box");
    }
    // the mistakes
    auto refused = [](const sc::Knobs& k, const sc::Start& st, const std::string& part) {
        std::string why;
        try {
            sc::run("multistart", rastrigin(nullptr), 1e-6, false, 0.0, 10, 500, 4, k, {}, 0, st);
        } catch (const std::invalid_argument& e) {
            why = e.what();
        }
        check(why.find(part) != std::string::npos, "refused: '" + why + "' (wanted: " + part + ")");
    };
    refused({{"inner", lbfgsb}, {"start_points", word("around_x0")}}, {}, "around_x0 needs x0");
    refused({{"inner", lbfgsb}, {"allowance", word("sequential")}, {"threads", number(2)}}, from, "one start at a time");
    refused({{"inner", lbfgsb}, {"start_points", word("around_x0")}, {"start_sigma", number(-1.0)}}, from,
            "start_sigma must be");
    refused({{"inner", lbfgsb}, {"allowance", word("all")}}, from, "allowance = 'all'");

    if (failures) {
        std::printf("test_multistart_starts: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_multistart_starts: all checks passed\n");
    return 0;
}
