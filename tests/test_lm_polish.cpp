// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// lm's knobs of task 25, §24.3.5 (lm.hpp LM-10, scalar.hpp LOCAL-8):
//   - max_iter = k stops lm after k accepted steps, at the k-th point of the
//     unlimited run, bit for bit, with k Jacobians; a limit it never meets
//     changes nothing; solve_robust counts its rounds' steps together;
//   - through minimize's path: no new knob, no new setting; lm_max_iter = 0
//     runs as without it; lm_max_iter = 10 is lm.hpp's run with max_iter =
//     10 (Kommenda et al. 2020's short polish);
//   - lm_failed: with stop a trial point whose residuals are not finite ends
//     the run (LOCAL-3); with refuse lm steps back from it (LM-9) and goes
//     on to the minimum, the failed points counted;
//   - lm_geodesic = on is lm.hpp's geodesic run; refused with a finite bound;
//   - the knobs' values are checked.
// ============================================================================

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/scalar/scalar.hpp>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

namespace sc = mootation::scalar;
namespace lm = mootation::scalar::lm;
using Vec = std::vector<double>;

const double kInf = std::numeric_limits<double>::infinity();
const double kNaN = std::numeric_limits<double>::quiet_NaN();

sc::Knob word(const std::string& w) { return {true, w, 0.0}; }
sc::Knob number(double v) { return {false, "", v}; }

lm::Problem rosenbrock_lm()
{
    lm::Problem p;
    p.n = 2;
    p.m = 2;
    p.residuals = [](const Vec& x, Vec& r) { r = {10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]}; };
    p.jacobian = [](const Vec& x, lm::Matrix& J) {
        J(0, 0) = -20.0 * x[0];
        J(0, 1) = 10.0;
        J(1, 0) = -1.0;
        J(1, 1) = 0.0;
    };
    return p;
}

sc::Problem rosenbrock_sc()
{
    sc::Problem P;
    P.lo = {-kInf, -kInf};
    P.hi = {kInf, kInf};
    P.m = 1;
    P.n_res = 2;
    P.residuals = [](const Vec& x, Vec& r) { r = {10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0]}; };
    P.jacobian = [](const Vec& x, sc::Rows& J) { J = sc::Rows{{-20.0 * x[0], 10.0}, {-1.0, 0.0}}; };
    return P;
}

// r = (x - 3)^2 with a hole: NaN on [1.4, 1.6], +inf on (1.6, 1.7]
sc::Problem holed()
{
    sc::Problem P;
    P.lo = {-kInf};
    P.hi = {kInf};
    P.m = 1;
    P.n_res = 1;
    P.residuals = [](const Vec& x, Vec& r) {
        if (x[0] >= 1.4 && x[0] <= 1.6) r = {kNaN};
        else if (x[0] > 1.6 && x[0] <= 1.7) r = {kInf};
        else r = {(x[0] - 3.0) * (x[0] - 3.0)};
    };
    P.jacobian = [](const Vec& x, sc::Rows& J) { J = sc::Rows{{2.0 * (x[0] - 3.0)}}; };
    return P;
}

sc::Start from(const Vec& x0)
{
    sc::Start s;
    s.has_x0 = true;
    s.x0 = x0;
    return s;
}

std::string setting(const sc::Result& r, const std::string& k)
{
    for (const auto& [a, b] : r.settings)
        if (a == k) return b;
    return "(none)";
}

sc::Result run_lm(const sc::Problem& P, const sc::Knobs& knobs, const Vec& x0, long long budget = 5000)
{
    return sc::run("lm", P, 1e-6, false, 0.0, 1, budget, 1, knobs, {}, 0, from(x0));
}

}   // namespace

int main()
{
    const lm::Problem p = rosenbrock_lm();
    // 1. lm.hpp: max_iter = k is the unlimited run cut at its k-th accepted point
    std::vector<Vec> path;
    lm::Options o;
    o.iterates = &path;
    const lm::Result full = lm::solve(p, {-1.2, 1.0}, o);
    std::printf("      unlimited: %lld accepted steps, %lld Jacobians, %lld evaluations\n", full.iterations, full.njev,
                full.nfev);
    check(full.iterations > 10 && static_cast<long long>(path.size()) == full.iterations,
          "lm: Rosenbrock from (-1.2, 1) takes more than 10 accepted steps");
    for (long long k : {1LL, 3LL, 10LL}) {
        lm::Options ok = o;
        ok.iterates = nullptr;
        ok.max_iter = k;
        const lm::Result cut = lm::solve(p, {-1.2, 1.0}, ok);
        check(cut.status == lm::Status::max_iter && cut.iterations == k && cut.njev == k &&
                  cut.x == path[static_cast<std::size_t>(k - 1)],
              "lm: max_iter = " + std::to_string(k) + " stops at the unlimited run's point " + std::to_string(k) +
                  ", bit for bit, with " + std::to_string(k) + " Jacobians");
    }
    {
        lm::Options big = o;
        big.iterates = nullptr;
        big.max_iter = 1000;
        const lm::Result same = lm::solve(p, {-1.2, 1.0}, big);
        check(same.x == full.x && same.nfev == full.nfev && same.njev == full.njev && same.status == full.status,
              "lm: a max_iter it never meets changes nothing");
    }
    {
        lm::Options h;
        h.loss = lm::Options::Loss::huber;
        h.f_scale = 0.5;
        const lm::Result all = lm::solve_robust(p, {-1.2, 1.0}, h);
        h.max_iter = 5;
        const lm::Result five = lm::solve_robust(p, {-1.2, 1.0}, h);
        check(all.iterations > 5 && five.iterations == 5 && five.status == lm::Status::max_iter,
              "lm: solve_robust's rounds count their steps together (" + std::to_string(all.iterations) +
                  " without the limit, 5 with max_iter = 5)");
    }

    // 2. through minimize's path
    const sc::Problem R = rosenbrock_sc();
    const sc::Result plain = run_lm(R, {}, {-1.2, 1.0});
    check(setting(plain, "lm_max_iter") == "(none)" && setting(plain, "lm_iterations") == "(none)" &&
              setting(plain, "lm_failed") == "(none)" && setting(plain, "lm_geodesic") == "(none)",
          "minimize's lm: without the new knobs no new setting");
    const sc::Result zero = run_lm(R, {{"lm_max_iter", number(0)}}, {-1.2, 1.0});
    check(zero.variables == plain.variables && zero.evaluations == plain.evaluations &&
              zero.grad_calls == plain.grad_calls && setting(zero, "lm_max_iter") == "0",
          "minimize's lm: lm_max_iter = 0 runs as without it");
    {
        const sc::Result ten = run_lm(R, {{"lm_max_iter", number(10)}}, {-1.2, 1.0});
        lm::Options o10;
        o10.max_iter = 10;
        o10.max_nfev = std::numeric_limits<long long>::max() / 4;
        const lm::Result own = lm::solve(p, {-1.2, 1.0}, o10);
        check(ten.variables[0] == own.x && ten.grad_calls == 10 && setting(ten, "lm_iterations") == "10" &&
                  ten.evaluations < plain.evaluations,
              "minimize's lm: lm_max_iter = 10 is lm.hpp's run with max_iter = 10 (" +
                  std::to_string(ten.evaluations) + " evaluations against " + std::to_string(plain.evaluations) +
                  ")");
    }

    // 3. lm_failed
    const sc::Problem H = holed();
    const sc::Result stop = run_lm(H, {}, {0.0});
    const sc::Result stop2 = run_lm(H, {{"lm_failed", word("stop")}}, {0.0});
    const sc::Result refuse = run_lm(H, {{"lm_failed", word("refuse")}}, {0.0});
    std::printf("      stop: f = %.6g after %lld evaluations (%lld failed); refuse: f = %.3g at x = %.12g after %lld "
                "(%lld failed)\n",
                stop.objectives[0][0], stop.evaluations, stop.n_failed, refuse.objectives[0][0],
                refuse.variables[0][0], refuse.evaluations, refuse.n_failed);
    check(stop.n_failed == 1 && stop.objectives[0][0] == 40.5 && stop2.variables == stop.variables &&
              stop2.evaluations == stop.evaluations,
          "lm_failed = stop (the default): the first trial point in the hole ends the run at x0 (LOCAL-3)");
    check(refuse.n_failed >= 1 && refuse.objectives[0][0] < 1e-12 && std::fabs(refuse.variables[0][0] - 3.0) < 1e-3,
          "lm_failed = refuse: lm steps back from the hole (LM-9) and goes on to x = 3");
    {
        sc::Problem N = holed();
        N.residuals = [](const Vec& x, Vec& r) { r = {x[0] > 1.0 ? kNaN : (x[0] - 3.0) * (x[0] - 3.0)}; };
        const sc::Result nan_all = run_lm(N, {{"lm_failed", word("refuse")}}, {0.0}, 300);
        check(nan_all.evaluations <= 300 && std::isfinite(nan_all.objectives[0][0]) &&
                  nan_all.objectives[0][0] < 40.5 && nan_all.variables[0][0] <= 1.0,
              "lm_failed = refuse: NaN everywhere beyond x = 1 — no crash, the answer finite and below f(x0), "
              "within the budget (" +
                  std::to_string(nan_all.evaluations) + " evaluations, x = " +
                  std::to_string(nan_all.variables[0][0]) + ")");
    }

    // 4. lm_geodesic
    {
        const sc::Result geo = run_lm(R, {{"lm_geodesic", word("on")}}, {-1.2, 1.0});
        lm::Options og;
        og.geodesic = true;
        og.max_nfev = std::numeric_limits<long long>::max() / 4;
        const lm::Result own = lm::solve(p, {-1.2, 1.0}, og);
        check(geo.variables[0] == own.x && setting(geo, "lm_geodesic") == "on",
              "minimize's lm: lm_geodesic = on is lm.hpp's geodesic run, bit for bit");
        sc::Problem B = R;
        B.lo = {-2.0, -2.0};
        B.hi = {2.0, 2.0};
        bool refused = false;
        try {
            run_lm(B, {{"lm_geodesic", word("on")}}, {-1.2, 1.0});
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        check(refused, "lm_geodesic = on with a finite bound: refused (LMB-5)");
    }

    // 5. lm_varpro (LOCAL-9): y(t) = 2 e^{-0.5 t} + 1 e^{-3 t} at t = 0, 0.1, ..., 3;
    //    x = (a_1, alpha_1, a_2, alpha_2), a linear (x_0, x_2)
    {
        sc::Problem E;
        const double inf = kInf;
        E.lo.assign(4, -inf);
        E.hi.assign(4, inf);
        E.m = 1;
        E.n_res = 31;
        E.linear = {0, 2};
        E.residuals = [](const Vec& x, Vec& r) {
            r.assign(31, 0.0);
            for (int i = 0; i <= 30; ++i) {
                const double t = 0.1 * i;
                const double d = 2.0 * mootation::dm::exp(-0.5 * t) + mootation::dm::exp(-3.0 * t);
                r[static_cast<std::size_t>(i)] =
                    x[0] * mootation::dm::exp(-x[1] * t) + x[2] * mootation::dm::exp(-x[3] * t) - d;
            }
        };
        const Vec start{0.0, 0.3, 0.0, 5.0};
        for (const char* how : {"on", "kaufman"}) {
            const sc::Result v = run_lm(E, {{"lm_varpro", word(how)}}, start, 3000);
            const Vec& x = v.variables[0];
            std::printf("      lm_varpro = %s: f = %.3g, x = (%.10g, %.10g, %.10g, %.10g), %lld evaluations\n", how,
                        v.objectives[0][0], x[0], x[1], x[2], x[3], v.evaluations);
            check(v.objectives[0][0] < 1e-20 && std::fabs(x[1] - 0.5) < 1e-8 && std::fabs(x[3] - 3.0) < 1e-8 &&
                      std::fabs(x[0] - 2.0) < 1e-8 && std::fabs(x[2] - 1.0) < 1e-8 &&
                      setting(v, "lm_varpro") == how && setting(v, "lm_varpro_rank") == "2",
                  std::string("lm_varpro = ") + how + ": the two exponentials found, Phi of rank 2");
        }
        sc::Problem B = E;                            // y depends on alpha: refused
        B.residuals = [](const Vec& x, Vec& r) {
            r.assign(31, 0.0);
            for (int i = 0; i <= 30; ++i) {
                const double t = 0.1 * i;
                r[static_cast<std::size_t>(i)] = x[0] * mootation::dm::exp(-x[1] * t) + x[2] * t + x[3] * x[3] - 1.0;
            }
        };
        bool refused = false;
        try {
            run_lm(B, {{"lm_varpro", word("on")}}, start, 3000);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        check(refused, "lm_varpro: residuals whose r(alpha, a = 0) moves with alpha are refused");
        sc::Problem C = E;
        C.linear.clear();
        refused = false;
        try {
            run_lm(C, {{"lm_varpro", word("on")}}, start, 3000);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        check(refused, "lm_varpro without the problem's linear variables: refused");
    }

    // 6. the knobs' values
    for (const auto& [k, v] : std::vector<std::pair<std::string, sc::Knob>>{{"lm_max_iter", number(2.5)},
                                                                           {"lm_max_iter", number(-1)},
                                                                           {"lm_failed", word("maybe")},
                                                                           {"lm_geodesic_alpha", number(0)},
                                                                           {"lm_geodesic_h", number(0)}}) {
        bool refused = false;
        try {
            run_lm(R, {{k, v}}, {-1.2, 1.0});
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        check(refused, k + " = " + (v.is_word ? v.word : std::to_string(v.number)) + ": refused");
    }

    std::printf(g_failed ? "%d FAILED\n" : "all passed\n", g_failed);
    return g_failed ? 1 : 0;
}
