// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/tiktak.hpp (task 19, §19.6):
//   - theta_i = min(max(0.1, (i / N*)^(1/2)), 0.995) at the ends and between;
//   - the starts: with each search one evaluation (start_budget = 1), the
//     problem sees the N points of the sample, then S_i = (1 - theta_i) s_i
//     + theta_i p_low for i = 2 ... N* (S_1 = s_1, free), s_i the sample in
//     ascending order of f and p_low the best start so far — bit for bit;
//   - one search = the inner solver: N = 1 is the sample's point, then the
//     inner solver from it on the rest of the budget, bit for bit (the
//     answer, the evaluations);
//   - two minima: the lower found, every search in the log; the budget kept; the same seed, the
//     same run; refusals (no inner solver, a bound infinite, share past 1).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <mootation/scalar/scalar.hpp>

namespace {

namespace sc = mootation::scalar;
using Vec = std::vector<double>;

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    if (!ok) ++g_failed;
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
}

sc::Knob word(const std::string& w) { return {true, w, 0.0}; }
sc::Knob number(double v) { return {false, "", v}; }

std::string setting(const sc::Result& r, const std::string& key)
{
    for (const auto& [k, v] : r.settings)
        if (k == key) return v;
    return "";
}

// (x^2 - 1)^2 + y^2 + 0.1 x on [-2, 2]^2: two minima, the left one lower; every call recorded
sc::Problem two_minima(std::shared_ptr<std::vector<Vec>> seen)
{
    sc::Problem P;
    P.lo = {-2.0, -2.0};
    P.hi = {2.0, 2.0};
    P.m = 1;
    P.evaluate = [seen](const Vec& x, Vec& f, Vec&) {
        seen->push_back(x);
        f.assign(1, (x[0] * x[0] - 1.0) * (x[0] * x[0] - 1.0) + x[1] * x[1] + 0.1 * x[0]);
    };
    return P;
}

double value(const Vec& x) { return (x[0] * x[0] - 1.0) * (x[0] * x[0] - 1.0) + x[1] * x[1] + 0.1 * x[0]; }

const char* kNM = "{\"local_tol\":1e-10,\"solver\":\"nelder_mead\"}";

void theta()
{
    using sc::tiktak::theta;
    check(theta(1, 100) == 0.1 && theta(1, 10) == std::sqrt(0.1) && theta(10, 10) == 0.995 && theta(25, 100) == 0.5,
          "theta_i: 0.1 at first, (i/N*)^1/2 between, 0.995 at the end");
}

void the_starts()
{
    const long long N = 40, n_star = 4;
    auto seen = std::make_shared<std::vector<Vec>>();
    const sc::Result r = sc::run("tiktak", two_minima(seen), 1e-6, false, 0.0, 10, 1000, 3,
                                 {{"inner", word(kNM)}, {"tiktak_n", number(static_cast<double>(N))},
                                  {"start_budget", number(1)}},
                                 {}, 0, {});
    bool sample = seen->size() == static_cast<std::size_t>(N + n_star - 1);
    // the expected starts from the sample the problem saw
    std::vector<std::size_t> order(static_cast<std::size_t>(N));
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return value((*seen)[a]) < value((*seen)[b]); });
    bool starts = sample;
    if (sample) {
        Vec low = (*seen)[order[0]];                                   // search 1: S_1 = s_1, its own minimum
        double f_low = value(low);
        for (long long i = 2; i <= n_star; ++i) {
            const Vec& s = (*seen)[order[static_cast<std::size_t>(i - 1)]];
            const double t = sc::tiktak::theta(i, n_star);
            Vec x(2);
            for (std::size_t j = 0; j < 2; ++j) x[j] = (1.0 - t) * s[j] + t * low[j];
            const Vec& got = (*seen)[static_cast<std::size_t>(N + i - 2)];
            starts = starts && got == x;
            if (value(x) < f_low) {
                low = x;
                f_low = value(x);
            }
        }
    }
    check(sample && starts && setting(r, "tiktak_n_star") == "4" && setting(r, "local_searches") == "4",
          "the starts: N = 40 points, then S_i = (1 - theta_i) s_i + theta_i p_low, i = 2 ... 4, bit for bit");
}

void one_search_is_the_inner_solver()
{
    const long long budget = 600;
    auto seen = std::make_shared<std::vector<Vec>>();
    const sc::Result t = sc::run("tiktak", two_minima(seen), 1e-6, false, 0.0, 10, budget, 7,
                                 {{"inner", word(kNM)}, {"tiktak_n", number(1)}}, {}, 0, {});
    const Vec s1 = seen->at(0);
    auto seen2 = std::make_shared<std::vector<Vec>>();
    sc::Start st;
    st.has_x0 = true;
    st.x0 = s1;
    const sc::Result nm = sc::run("nelder_mead", two_minima(seen2), 1e-6, false, 0.0, 10, budget - 1,
                                  sc::strategy::start_seed(7, 1), {{"local_tol", number(1e-10)}}, {}, 0, st);
    check(t.variables == nm.variables && t.objectives == nm.objectives && t.evaluations == nm.evaluations &&
              *seen == *seen2,
          "N = 1: the sample's point, then nelder_mead from it on the rest: the same run, bit for bit (" +
              std::to_string(t.evaluations) + " evaluations)");
}

void a_run()
{
    auto seen = std::make_shared<std::vector<Vec>>();
    const sc::Knobs kb{{"inner", word(kNM)}, {"tiktak_n", number(50)}};
    const sc::Result a = sc::run("tiktak", two_minima(seen), 1e-6, false, 0.0, 10, 3000, 11, kb, {}, 0, {});
    auto seen2 = std::make_shared<std::vector<Vec>>();
    const sc::Result b = sc::run("tiktak", two_minima(seen2), 1e-6, false, 0.0, 10, 3000, 11, kb, {}, 0, {});
    const auto minima = mootation::json::parse(setting(a, "minima"));
    long long found = 0;
    for (const auto& m : minima.items) found += m.items[2].i;
    check(a.evaluations <= 3000 && static_cast<long long>(seen->size()) == a.evaluations && a.variables[0][0] < 0.0 &&
              a.objectives[0][0] < -0.09 && found == 5 && setting(a, "local_searches") == "5",
          "two minima: the lower found (f = " + std::to_string(a.objectives[0][0]) + ") by the N* = 5 searches, " +
              std::to_string(a.evaluations) + " of 3000 evaluations (TT-5: the run ends after them)");
    check(a.variables == b.variables && a.evaluations == b.evaluations && *seen == *seen2, "the same seed, the same run");
    auto refused = [](const sc::Problem& P, const sc::Knobs& k) {
        try {
            sc::run("tiktak", P, 1e-6, false, 0.0, 10, 100, 1, k, {}, 0, {});
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    sc::Problem open = two_minima(seen);
    open.hi[1] = std::numeric_limits<double>::infinity();
    check(refused(two_minima(seen), {}) && refused(open, kb) &&
              refused(two_minima(seen), {{"inner", word(kNM)}, {"tiktak_share", number(1.5)}}),
          "refused: no inner solver, a bound infinite, tiktak_share past 1");
}

}   // namespace

int main()
{
    theta();
    the_starts();
    one_search_is_the_inner_solver();
    a_run();
    if (g_failed) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
