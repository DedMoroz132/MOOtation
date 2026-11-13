// SPDX-License-Identifier: Apache-2.0
// The multistarts' filters and estimates (task 24, §24.2.5; strategy.hpp
// STRAT-13, STRAT-14, search.hpp SRCH-13):
//   * Boender's estimates on a hand example (w = 3, N = 100: 3 * 99 / 95);
//     minima_bayes = on adds them and changes nothing else;
//   * start_filter = oqnlp on the six-hump camelback of Ugray et al. 2007
//     (§4, Figures 1-2: 9 starts of 846 trial points), 1000 trial points:
//     few starts, the global minimum -1.0316; with the filters open
//     (distfactor 0, a threshold that rises at once) a start at every point;
//   * mlsl_stop = bayes stops mlsl before its budget once (3) holds;
//   * the mistakes are refused.

#include <cmath>
#include <cstdio>
#include <cstring>
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

// the six-hump camelback on [-3, 3] x [-2, 2], with its gradient
static sc::Problem camelback()
{
    sc::Problem P;
    P.lo = {-3.0, -2.0};
    P.hi = {3.0, 2.0};
    P.m = 1;
    P.evaluate = [](const Vec& z, Vec& f, Vec&) {
        const double x = z[0], y = z[1], x2 = x * x;
        f[0] = (4.0 - 2.1 * x2 + x2 * x2 / 3.0) * x2 + x * y + (-4.0 + 4.0 * y * y) * y * y;
    };
    P.gradient = [](const Vec& z, Vec& g, sc::Rows&) {
        const double x = z[0], y = z[1], x2 = x * x;
        g = {8.0 * x - 8.4 * x2 * x + 2.0 * x2 * x2 * x + y, x - 8.0 * y + 16.0 * y * y * y};
    };
    return P;
}

static std::string setting(const sc::Result& r, const std::string& key)
{
    for (const auto& [k, v] : r.settings)
        if (k == key) return v;
    return "";
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
    // the hand example
    {
        const auto e = sc::strategy::bayes_estimates(3, 100);
        check(e[2].second == "3.1263157894736842" || std::fabs(std::stod(e[2].second) - 3.0 * 99.0 / 95.0) < 1e-15,
              "3 minima of 100 searches: 3 * 99 / 95 expected (" + e[2].second + ")");
        check(std::fabs(std::stod(e[3].second) - 12.0 / 9900.0) < 1e-18, "uncovered 3 * 4 / (100 * 99)");
        check(sc::strategy::bayes_estimates(3, 5)[2].second == "inf", "N <= w + 2: inf");
    }
    // minima_bayes: the estimates added, nothing else changed
    {
        const sc::Knobs a{{"inner", L}, {"n_starts", number(40)}};
        sc::Knobs b = a;
        b.push_back({"minima_bayes", word("on")});
        const sc::Result ra = sc::run("multistart", camelback(), 1e-6, false, 0.0, 20, 5000, 3, a, {}, 0, {});
        const sc::Result rb = sc::run("multistart", camelback(), 1e-6, false, 0.0, 20, 5000, 3, b, {}, 0, {});
        check(same(ra, rb) && setting(ra, "minima_expected").empty() && !setting(rb, "minima_expected").empty(),
              "minima_bayes: the same run, the estimates in the settings");
        std::printf("  camelback, 40 starts of lbfgsb: %s minima found of %s searches, expected %s, uncovered %s\n",
                    setting(rb, "minima_found").c_str(), setting(rb, "minima_searches").c_str(),
                    setting(rb, "minima_expected").c_str(), setting(rb, "minima_uncovered").c_str());
    }
    // OQNLP's filters on the camelback, 1000 trial points (uniform, as OptQuest's are spread)
    {
        const sc::Knobs k{{"inner", L},          {"n_starts", number(1001)},       {"start_points", word("uniform")},
                          {"start_filter", word("oqnlp")}, {"minima_bayes", word("on")}};
        const sc::Result r = sc::run("multistart", camelback(), 1e-6, false, 0.0, 20, 100000, 3, k, {}, 0, {});
        const long long starts = std::stoll(setting(r, "oq_starts"));
        check(starts >= 2 && starts <= 50, "oqnlp: a few starts of 1000 trial points (" + std::to_string(starts) + ")");
        check(std::fabs(r.objectives[0][0] + 1.0316284534898) < 1e-6, "oqnlp: the global minimum -1.0316");
        std::printf("  oqnlp, camelback, 1000 trial points: %lld starts (Ugray et al.: 9 of 846), merit rejected %s, "
                    "distance rejected %s, f = %.10f, %lld evaluations, minima %s\n",
                    starts, setting(r, "oq_rejected_merit").c_str(), setting(r, "oq_rejected_distance").c_str(),
                    r.objectives[0][0], r.evaluations, setting(r, "minima_found").c_str());
        // the filters open: a start at every trial point
        const sc::Knobs open{{"inner", L},           {"n_starts", number(31)},          {"start_points", word("uniform")},
                             {"start_filter", word("oqnlp")}, {"oq_stage1", number(1)},
                             {"oq_distfactor", number(0.0)},  {"oq_waitcycle", number(1)},
                             {"oq_threshfactor", number(1e300)}};
        const sc::Result o = sc::run("multistart", camelback(), 1e-6, false, 0.0, 20, 100000, 3, open, {}, 0, {});
        const long long all = std::stoll(setting(o, "oq_starts"));
        check(all >= 15, "oqnlp, filters open: starts at most trial points (" + std::to_string(all) + " of 30)");
    }
    // mlsl_stop = bayes
    {
        const sc::Knobs a{{"inner", L}, {"mlsl_n", number(100)}};
        sc::Knobs b = a;
        b.push_back({"mlsl_stop", word("bayes")});
        b.push_back({"minima_bayes", word("on")});
        const sc::Result ra = sc::run("mlsl", camelback(), 1e-6, false, 0.0, 20, 20000, 3, a, {}, 0, {});
        const sc::Result rb = sc::run("mlsl", camelback(), 1e-6, false, 0.0, 20, 20000, 3, b, {}, 0, {});
        const double w = std::stod(setting(rb, "minima_found")), gN = std::stod(setting(rb, "minima_searches"));
        check(rb.evaluations < ra.evaluations && w >= 1.0 && w * (gN - 1.0) / (gN - w - 2.0) <= w + 0.5,
              "mlsl_stop = bayes: stopped once (3) held, before the budget");
        std::printf("  mlsl on the camelback: budget run %lld evaluations; bayes stop %lld evaluations, %s iterations, "
                    "w = %s, gamma k N = %s, expected %s\n",
                    ra.evaluations, rb.evaluations, setting(rb, "iterations").c_str(),
                    setting(rb, "minima_found").c_str(), setting(rb, "minima_searches").c_str(),
                    setting(rb, "minima_expected").c_str());
    }
    // the mistakes
    {
        std::string why;
        try {
            sc::run("multistart", camelback(), 1e-6, false, 0.0, 20, 1000, 3,
                    {{"inner", L}, {"start_filter", word("oqnlp")}, {"threads", number(2)}}, {}, 0, {});
        } catch (const std::invalid_argument& e) {
            why = e.what();
        }
        check(why.find("start_filter = oqnlp runs its starts one by one") != std::string::npos, "oqnlp + threads: " + why);
    }

    if (failures) {
        std::printf("test_multistart_filters: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_multistart_filters: all checks passed\n");
    return 0;
}
