// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/restarts.hpp (task 18, stage 3, §3.3): ipop and bipop.
//   - ipop: every run's population twice the last (AH05), σ⁰ half the box's
//     width; on bbob f1 below 1e-8;
//   - bipop's schedule on a constant f, where every run stops at once (H09
//     §2.1, RST-2 ... RST-5): a first run with λ_def, then regime 1 first;
//     regime 1's λ = 2^i λ_def up to 2^9, the last run of regime 1; regime
//     2's λ in [λ_def, λ_ℓ / 2], σ⁰ in [0.02 σ⁰_large, σ⁰_large], at most
//     half the last large run's evaluations, and run only while its budget
//     is the smaller;
//   - bipop_2009 on bbob f15 (Rastrigin, rotated), D = 3: below 1e-8; its
//     settings; the same seed, the same run; a checksum, MSVC = clang;
//   - cma_bounds = none evaluates beyond the box, penalty does not;
//   - refusals: a bound that is not finite, constraints.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <mootation/scalar/scalar.hpp>
#include <mootation/suite/bbob.hpp>

namespace {

namespace sc = mootation::scalar;
namespace B = mootation::suite::bbob;
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

sc::Problem bbob(int fid, int D, int iid = 1)
{
    auto obj = std::make_shared<B::Objective>(B::instance(fid, iid, D), D);
    sc::Problem P;
    P.lo.assign(static_cast<std::size_t>(D), -5.0);
    P.hi.assign(static_cast<std::size_t>(D), 5.0);
    P.m = 1;
    P.evaluate = [obj](const Vec& x, Vec& f, Vec&) { f.assign(1, (*obj)(x.data())); };
    return P;
}

double f_opt(int fid, int D) { return B::instance(fid, 1, D).f_opt; }

void ipop()
{
    const sc::Result r = sc::run("ipop", bbob(1, 4), 1e-6, false, 0.0, 10, 20000, 1, {}, {}, 0, {});
    const auto runs = mootation::json::parse(setting(r, "restarts"));
    bool doubling = runs.items.size() >= 2;
    long long spent = 0;
    for (std::size_t k = 0; k < runs.items.size(); ++k) {
        const auto& row = runs.items[k].items;
        doubling = doubling && row[1].i == static_cast<std::int64_t>(8u << k) && row[2].f == 5.0;
        spent += row[3].i;
    }
    check(doubling && spent == r.evaluations && r.objectives[0][0] - f_opt(1, 4) < 1e-8,
          "ipop, bbob f1 D = 4: lambda 8, 16, 32, ..., sigma0 5; f - f* = " +
              std::to_string(r.objectives[0][0] - f_opt(1, 4)) + " in " + std::to_string(runs.items.size()) + " runs");
}

void bipop_schedule()
{
    sc::Problem P;
    P.lo.assign(3, -5.0);
    P.hi.assign(3, 5.0);
    P.m = 1;
    P.evaluate = [](const Vec&, Vec& f, Vec&) { f.assign(1, 1.0); };
    const sc::Result r = sc::run("bipop", P, 1e-6, false, 0.0, 10, 10000000, 2, {}, {}, 0, {});
    const auto runs = mootation::json::parse(setting(r, "restarts"));
    const std::int64_t lam_def = 7;                   // 4 + floor(3 ln 3)
    bool ok = runs.items.size() > 10 && runs.items[0].items[0].i == 0 && runs.items[1].items[0].i == 1 &&
              runs.items.back().items[0].i == 1;
    long long used_large = 0, used_small = 0, last_large = 0, large = 0;
    std::int64_t lam_last = lam_def;
    std::string why;
    for (std::size_t k = 1; k < runs.items.size() && ok; ++k) {
        const auto& row = runs.items[k].items;
        const std::int64_t regime = row[0].i, lam = row[1].i, evals = row[3].i;
        const double sigma0 = row[2].f;
        if (regime == 1) {
            ok = k == 1 || used_small >= used_large;                 // RST-4: the smaller budget runs
            ++large;
            ok = ok && lam == (lam_def << large) && sigma0 == 2.0;
            used_large += evals;
            last_large = evals;
            lam_last = lam;
            if (!ok) why = "large run " + std::to_string(k);
        } else {
            ok = used_small < used_large && lam >= lam_def && lam <= lam_last / 2 && sigma0 >= 0.02 && sigma0 <= 2.0 &&
                 evals <= std::max(1LL, last_large / 2) + lam;       // the allowance, the generation it ends in
            used_small += evals;
            if (!ok) why = "small run " + std::to_string(k);
        }
    }
    check(ok && large == 9 && r.evaluations < 10000000,
          "bipop on a constant f: first run lambda 7, then regime 1, lambda 7 * 2^i to 2^9, the last run of regime 1; "
          "regime 2 within its bounds and only while its budget is smaller (" + std::to_string(runs.items.size()) +
              " runs, " + std::to_string(r.evaluations) + " evaluations" + (why.empty() ? "" : ", " + why) + ")");
}

void bipop_2009()
{
    const sc::Result r = sc::run("bipop_2009", bbob(15, 3), 1e-6, false, 0.0, 10, 300000, 4, {}, {}, 0, {});
    const sc::Result again = sc::run("bipop_2009", bbob(15, 3), 1e-6, false, 0.0, 10, 300000, 4, {}, {}, 0, {});
    const double df = r.objectives[0][0] - f_opt(15, 3);
    check(df < 1e-8 && setting(r, "cma_weights") == "positive" && setting(r, "cma_cmu") == "hansen2009" &&
              r.variables == again.variables && r.evaluations == again.evaluations,
          "bipop_2009, bbob f15 D = 3: f - f* = " + std::to_string(df) + " in " + std::to_string(r.evaluations) +
              " evaluations; its CMA-ES H09's; the same seed, the same run");
    std::uint64_t h = 1469598103934665603ull;
    for (const auto& imp : r.improvements) {
        std::uint64_t b;
        std::memcpy(&b, &imp.f, sizeof b);
        h = (h ^ b ^ static_cast<std::uint64_t>(imp.evaluations)) * 1099511628211ull;
    }
    char buf[32];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    check(h == 0x1d09598eca0783e4ull, std::string("bipop_2009, bbob f15 D = 3: checksum of the records ") + buf);
}

void bounds()
{
    sc::Problem P;
    P.lo.assign(2, -5.0);
    P.hi.assign(2, 5.0);
    P.m = 1;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) { f.assign(1, (x[0] - 7.0) * (x[0] - 7.0) + (x[1] - 7.0) * (x[1] - 7.0)); };
    const sc::Result none = sc::run("ipop", P, 1e-6, false, 0.0, 10, 5000, 3, {{"cma_bounds", word("none")}}, {}, 0, {});
    const sc::Result pen = sc::run("ipop", P, 1e-6, false, 0.0, 10, 5000, 3, {}, {}, 0, {});
    check(none.objectives[0][0] < 1e-10 && std::fabs(pen.objectives[0][0] - 8.0) < 1e-10 && pen.variables[0][0] == 5.0,
          "the optimum at (7, 7) beyond the box [-5, 5]^2: cma_bounds = none finds it, penalty stops at (5, 5) (f = " +
              std::to_string(none.objectives[0][0]) + ", " + std::to_string(pen.objectives[0][0]) + ")");
}

// THE TARGET (evaluator.hpp): a run ends at the first point at or below
// Problem::stop_at, whatever the strategy; the point is the answer and the last
// evaluation, and nothing loops on a budget that no longer shrinks
void target()
{
    const double fopt = f_opt(1, 4), at = fopt + 1e-8;
    for (const char* name : {"ipop", "bipop_2009", "multistart", "ils", "mlsl"}) {
        sc::Problem P = bbob(1, 4);
        auto calls = std::make_shared<long long>(0);
        const auto inner_eval = P.evaluate;
        P.evaluate = [inner_eval, calls](const Vec& x, Vec& f, Vec& g) {
            ++*calls;
            inner_eval(x, f, g);
        };
        P.stop_at = at;
        sc::Knobs kn;
        if (std::string(name) == "multistart" || std::string(name) == "ils" || std::string(name) == "mlsl")
            kn.push_back({"inner", word("{\"solver\":\"cmaes\",\"local_tol\":1e-14}")});
        const sc::Result r = sc::run(name, P, 1e-6, false, 0.0, 10, 1000000, 6, kn, {}, 0, {});
        const bool last = !r.improvements.empty() && r.improvements.back().f <= at &&
                          r.improvements.back().evaluations == r.evaluations;
        check(r.objectives[0][0] <= at && last && r.evaluations < 1000000,
              std::string(name) + " with stop_at = f* + 1e-8, bbob f1 D = 4: ends at the target's first point, " +
                  std::to_string(r.evaluations) + " evaluations (" + std::to_string(*calls) + " calls)");
    }
}

// RST-8: the target met in a run of regime 2 ends bipop (the H2 sample's
// bbob f15, D = 10, instance 2, seed 2 looped there for ever: regime 2's own
// allowance cleared the budget's end against the run's whole budget)
void target_in_regime_2()
{
    sc::Problem P = bbob(15, 10, 2);
    const double at = B::instance(15, 2, 10).f_opt + 1e-8;
    P.stop_at = at;
    const sc::Result r = sc::run("bipop_2009", P, 1e-6, false, 0.0, 10, 1000000, 2,
                                 {{"cma_bounds", word("none")}}, {}, 0, {});
    const auto runs = mootation::json::parse(setting(r, "restarts"));
    const auto& last = runs.items.back().items;
    check(r.objectives[0][0] <= at && r.evaluations < 1000000 && last[0].i == 2,
          "bipop_2009, bbob f15 D = 10 instance 2, the target met in a run of regime 2: the strategy ends (" +
              std::to_string(runs.items.size()) + " runs, " + std::to_string(r.evaluations) + " evaluations)");
}

// THE LEAN MODE (evaluator.hpp; group 3g): 50000 points kept as two, the
// records and the best those of the full archive; bipop_2009 with
// keep_points = best the run of keep_points = all, bit for bit (bbob is
// evaluated beyond the box, cma_bounds = none: no point repeats)
void lean()
{
    sc::Problem P = bbob(15, 5);
    sc::Evaluator full(P, 1e-6, false, 0.0, 100000, 1, {}, 0), thin(P, 1e-6, false, 0.0, 100000, 1, {}, 0);
    thin.lean = true;
    std::mt19937 g(9);
    for (int k = 0; k < 50000; ++k) {
        Vec x(5);
        for (double& v : x) v = -5.0 + 10.0 * mootation::rnd::uniform01(g);
        full.index(x);
        thin.index(x);
    }
    const auto a = full.improvements.records(), b = thin.improvements.records();
    bool same = a.size() == b.size();
    for (std::size_t i = 0; same && i < a.size(); ++i)
        same = a[i].evaluations == b[i].evaluations && a[i].f == b[i].f;
    const std::size_t best = full.answer()[0];
    check(thin.X.size() == 2 && full.X.size() == 50000 && thin.spent == 50000 && same &&
              thin.answer() == std::vector<std::size_t>{0} && thin.X[0] == full.X[best],
          "the lean evaluator: 50000 points kept as 2, the same " + std::to_string(b.size()) +
              " records and the same best point");
    const sc::Knobs all{{"cma_bounds", word("none")}};
    const sc::Knobs best_only{{"cma_bounds", word("none")}, {"keep_points", word("best")}};
    const sc::Result r1 = sc::run("bipop_2009", bbob(15, 5), 1e-6, false, 0.0, 10, 200000, 3, all, {}, 0, {});
    const sc::Result r2 = sc::run("bipop_2009", bbob(15, 5), 1e-6, false, 0.0, 10, 200000, 3, best_only, {}, 0, {});
    bool same_run = r1.improvements.size() == r2.improvements.size();
    for (std::size_t i = 0; same_run && i < r1.improvements.size(); ++i)
        same_run = r1.improvements[i].evaluations == r2.improvements[i].evaluations &&
                   r1.improvements[i].f == r2.improvements[i].f;
    check(same_run && r1.evaluations == r2.evaluations && r1.variables[0] == r2.variables[0] &&
              setting(r2, "keep_points") == "best" && setting(r1, "restarts") == setting(r2, "restarts"),
          "bipop_2009, bbob f15 D = 5, keep_points = best: the run of keep_points = all, bit for bit (" +
              std::to_string(r2.evaluations) + " evaluations)");
}

void refusals()
{
    std::string why;
    sc::Problem P = bbob(1, 2);
    P.hi[0] = std::numeric_limits<double>::infinity();
    try {
        sc::run("bipop", P, 1e-6, false, 0.0, 10, 500, 1, {}, {}, 0, {});
    } catch (const std::invalid_argument& e) {
        why = e.what();
    }
    check(why.find("every bound must be finite") != std::string::npos, "bipop with an infinite bound: " + why);
    sc::Problem Q = bbob(1, 2);
    Q.n_cons = 1;
    why.clear();
    try {
        sc::run("ipop", Q, 1e-6, false, 0.0, 10, 500, 1, {{"cma_sigma0", number(0.3)}}, {}, 0, {});
    } catch (const std::invalid_argument& e) {
        why = e.what();
    }
    check(why == "ipop takes bounds only (RST-6)", "ipop with constraints: " + why);
}

}   // namespace

int main()
{
    ipop();
    bipop_schedule();
    bipop_2009();
    bounds();
    target();
    target_in_regime_2();
    lean();
    refusals();
    std::printf("%s (%d failed)\n", g_failed == 0 ? "PASS" : "FAIL", g_failed);
    return g_failed == 0 ? 0 : 1;
}
