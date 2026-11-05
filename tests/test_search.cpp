// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/search.hpp (task 18, stage 3, §3.3 and §3.6):
//   - mlsl: Γ(1 + n/2) exactly; the property of Rinnooy Kan & Timmer (part
//     II, Theorem 1 with σ = 4): the number of local searches stays bounded
//     as the sample grows; both minima of (x^2 - 1)^2 + y^2 found;
//   - basin_hopping on Rastrigin from a far start: the global minimum, the
//     step adjusted; ils: better never accepts a worse minimum, rw accepts
//     every one;
//   - halving: ⌈log2 8⌉ = 3 rounds of 8, 4, 2 arms, each round's allowance
//     B/(|S| 3) more per arm (Jamieson & Talwalkar, Algorithm 1);
//   - memetic: de with nelder_mead as it goes;
//   - every strategy within its budget, also with lshade (whose run depends
//     on the budget) inside: the reruns' new points capped (STRAT-6).
// ============================================================================

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

sc::Knob word(const std::string& w) { return {true, w, 0.0}; }
sc::Knob number(double v) { return {false, "", v}; }

std::string setting(const sc::Result& r, const std::string& key)
{
    for (const auto& [k, v] : r.settings)
        if (k == key) return v;
    return "";
}

sc::Problem two_minima(std::shared_ptr<long long> calls)
{
    sc::Problem P;
    P.lo = {-2.0, -2.0};
    P.hi = {2.0, 2.0};
    P.m = 1;
    P.evaluate = [calls](const Vec& x, Vec& f, Vec&) {
        ++*calls;
        f.assign(1, (x[0] * x[0] - 1.0) * (x[0] * x[0] - 1.0) + x[1] * x[1]);
    };
    return P;
}

sc::Problem rastrigin(int n)
{
    sc::Problem P;
    P.lo.assign(static_cast<std::size_t>(n), -5.12);
    P.hi.assign(static_cast<std::size_t>(n), 5.12);
    P.m = 1;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) {
        double s = 10.0 * static_cast<double>(x.size());
        for (double v : x) s += v * v - 10.0 * mootation::dm::cos(2.0 * 3.14159265358979323846 * v);   // dm: the same bits everywhere
        f.assign(1, s);
    };
    return P;
}

const char* kNM = "{\"local_tol\":1e-10,\"solver\":\"nelder_mead\"}";

void mlsl()
{
    using sc::search::gamma_one_plus_half;
    check(gamma_one_plus_half(2) == 1.0 && gamma_one_plus_half(4) == 2.0 &&
              std::fabs(gamma_one_plus_half(3) - 1.3293403881791355) < 1e-14 &&
              std::fabs(gamma_one_plus_half(1) - 0.886226925452758) < 1e-14,
          "Gamma(1 + n/2) for n = 1 ... 4: 0.886..., 1, 1.329..., 2");
    long long searches[2] = {0, 0};
    std::string minima[2];
    int k = 0;
    for (long long budget : {20000LL, 80000LL}) {
        auto calls = std::make_shared<long long>(0);
        const sc::Result r = sc::run("mlsl", two_minima(calls), 1e-6, false, 0.0, 10, budget, 5,
                                     {{"inner", word(kNM)}, {"mlsl_nu", number(0.05)}}, {}, 0, {});
        searches[k] = std::stoll(setting(r, "local_searches"));
        minima[k] = setting(r, "minima");
        check(r.evaluations <= budget && *calls == r.evaluations && r.objectives[0][0] < 1e-10,
              "mlsl on (x^2 - 1)^2 + y^2, budget " + std::to_string(budget) + ": " + setting(r, "iterations") +
                  " iterations, " + std::to_string(searches[k]) + " local searches, f = " +
                  std::to_string(r.objectives[0][0]));
        ++k;
    }
    const auto m = mootation::json::parse(minima[1]);
    check(searches[1] == searches[0] && searches[0] <= 6 && m.items.size() == 2,
          "mlsl, nu = 0.05: the sample four times larger, no more local searches (" + std::to_string(searches[0]) +
              ", " + std::to_string(searches[1]) + "); both minima: " + minima[1].substr(0, 90));
    auto calls = std::make_shared<long long>(0);
    const sc::Result r0 = sc::run("mlsl", two_minima(calls), 1e-6, false, 0.0, 10, 20000, 5, {{"inner", word(kNM)}},
                                  {}, 0, {});
    check(std::stoll(setting(r0, "local_searches")) > searches[0],
          "mlsl, nu = 0 (the paper's experiments): searches near the minima found start again (" +
              setting(r0, "local_searches") + ")");
}

void walks()
{
    const sc::Knobs kb{{"inner", word(kNM)}, {"bh_step", number(0.1)}};
    sc::Start far;
    far.has_x0 = true;
    far.x0 = {4.0, -4.0};
    const sc::Result r = sc::run("basin_hopping", rastrigin(2), 1e-6, false, 0.0, 10, 20000, 3, kb, {}, 0, far);
    const sc::Result again = sc::run("basin_hopping", rastrigin(2), 1e-6, false, 0.0, 10, 20000, 3, kb, {}, 0, far);
    check(r.objectives[0][0] < 1e-8 && r.variables == again.variables && r.evaluations <= 20000,
          "basin_hopping on Rastrigin from (4, -4): f = " + std::to_string(r.objectives[0][0]) + " after " +
              setting(r, "steps") + " steps, " + setting(r, "accepted") + " accepted, the step " +
              setting(r, "bh_step_end") + "; the same seed, the same run");
    const sc::Result rw = sc::run("ils", rastrigin(2), 1e-6, false, 0.0, 10, 8000, 3,
                                  {{"inner", word(kNM)}, {"ils_accept", word("rw")}}, {}, 0, far);
    check(setting(rw, "steps") == setting(rw, "accepted") && std::stoll(setting(rw, "steps")) > 10,
          "ils, rw: every one of " + setting(rw, "steps") + " minima accepted");
    for (const char* acc : {"better", "restart", "lsmc"}) {
        const sc::Result b = sc::run("ils", rastrigin(2), 1e-6, false, 0.0, 10, 8000, 3,
                                     {{"inner", word(kNM)}, {"ils_accept", word(acc)}}, {}, 0, far);
        check(b.evaluations <= 8000 && !b.objectives.empty() && b.objectives[0][0] < 3.0,
              std::string("ils, ") + acc + ": f = " + std::to_string(b.objectives[0][0]) + ", " + setting(b, "accepted") +
                  " of " + setting(b, "steps") + " accepted");
    }
    const sc::Result bt = sc::run("ils", rastrigin(2), 1e-6, false, 0.0, 10, 8000, 3,
                                  {{"inner", word(kNM)}, {"ils_accept", word("better")}}, {}, 0, far);
    check(std::stoll(setting(bt, "accepted")) < std::stoll(setting(bt, "steps")),
          "ils, better: fewer accepted than steps (" + setting(bt, "accepted") + " of " + setting(bt, "steps") + ")");
}

void halving()
{
    auto calls = std::make_shared<long long>(0);
    const long long B = 24000;
    const sc::Result r = sc::run("halving", two_minima(calls), 1e-6, false, 0.0, 10, B, 7,
                                 {{"inner", word("{\"solver\":\"cmaes\"}")}, {"n_starts", number(8)}}, {}, 0, {});
    const auto plan = mootation::json::parse(setting(r, "rounds"));
    const bool shape = plan.items.size() == 3 && plan.items[0].items[0].i == 8 && plan.items[1].items[0].i == 4 &&
                       plan.items[2].items[0].i == 2 && plan.items[0].items[1].i == 1000 &&
                       plan.items[1].items[1].i == 3000 && plan.items[2].items[1].i == 7000;
    check(shape && r.evaluations <= B && *calls >= r.evaluations && r.objectives[0][0] < 1e-8,
          "halving, 8 cmaes arms, B = 24000: rounds of 8, 4, 2 arms with 1000, 3000, 7000 evaluations each so far; " +
              std::to_string(r.evaluations) + " spent");
}

void memetic()
{
    auto calls = std::make_shared<long long>(0);
    const sc::Result r = sc::run("memetic", rastrigin(3), 1e-6, false, 0.0, 10, 20000, 2,
                                 {{"inner", word("{\"solver\":\"de\"}")}, {"local_solver", word(kNM)},
                                  {"memetic_every", number(2000)}, {"memetic_points", number(2)}},
                                 {}, 0, {});
    check(r.evaluations <= 20000 && r.objectives[0][0] < 1e-6 && std::stoll(setting(r, "phases")) >= 2,
          "memetic, de and nelder_mead on Rastrigin n = 3: f = " + std::to_string(r.objectives[0][0]) + " in " +
              setting(r, "phases") + " phases, " + setting(r, "local_searches") + " local searches");
}

void within_the_budget()
{
    std::uint64_t h = 1469598103934665603ull;          // the runs' answers, the same with MSVC and clang
    const char* lshade = "{\"solver\":\"lshade\"}";
    for (const char* name : {"multistart", "halving", "memetic", "mlsl", "ils", "basin_hopping"}) {
        auto calls = std::make_shared<long long>(0);
        sc::Knobs kn{{"inner", word(lshade)}};
        if (std::string(name) == "memetic") kn.push_back({"local_solver", word(lshade)});
        if (std::string(name) == "multistart" || std::string(name) == "halving") kn.push_back({"n_starts", number(4)});
        const sc::Result r = sc::run(name, rastrigin(2), 1e-6, false, 0.0, 10, 3000, 1, kn, {}, 0, {});
        check(r.evaluations <= 3000 && r.evaluations >= 1000,
              std::string(name) + " with lshade inside, budget 3000: " + std::to_string(r.evaluations) + " spent");
        for (const Vec& row : r.variables)
            for (double v : row) {
                std::uint64_t b;
                std::memcpy(&b, &v, sizeof b);
                h = (h ^ b) * 1099511628211ull;
            }
    }
    char buf[32];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    check(h == 0x661eff1f409cdb6eull, std::string("the six runs' answers: checksum ") + buf);
}

}   // namespace

int main()
{
    mlsl();
    walks();
    halving();
    memetic();
    within_the_budget();
    std::printf("%s (%d failed)\n", g_failed == 0 ? "PASS" : "FAIL", g_failed);
    return g_failed == 0 ? 0 : 1;
}
