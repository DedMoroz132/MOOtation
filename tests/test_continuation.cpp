// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/continuation.hpp (task 19, §19.6): data_continuation on r_i = a
// exp(-b t_i) - y_i, 20 points in two blocks of 10 (noisy data):
//   - one stage (dc_stages = []) is the inner solver, bit for bit: lm and
//     nelder_mead (the answer, the evaluations, the gradient calls);
//   - a window stage, then the problem: stage 1 is lm on the problem with
//     weights (1, 0) by itself, bit for bit; the last stage starts from its
//     answer (free from the archive); every call of the residuals is one
//     evaluation of the run; the answer's f is the problem's 1/2 ||r||^2;
//   - a weight vector stage; the budget kept; refusals.
// ============================================================================

#include <cmath>
#include <cstdio>
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

struct Calls {
    long long r = 0, J = 0;
};

// r_i = a exp(-b t_i) - y_i, y from a = 2, b = 0.7 with a little noise; blocks of 10
sc::Problem decay(std::shared_ptr<Calls> calls)
{
    sc::Problem P;
    P.lo = {0.1, 0.01};
    P.hi = {10.0, 5.0};
    P.m = 1;
    P.n_res = 20;
    P.blocks = {10, 10};
    P.weights = {1.0, 1.0};
    Vec t(20), y(20);
    for (int i = 0; i < 20; ++i) {
        t[static_cast<std::size_t>(i)] = 0.25 * i;
        y[static_cast<std::size_t>(i)] = 2.0 * std::exp(-0.7 * 0.25 * i) + 0.01 * std::sin(3.0 * i);
    }
    P.residuals = [calls, t, y](const Vec& x, Vec& r) {
        ++calls->r;
        r.resize(20);
        for (std::size_t i = 0; i < 20; ++i) r[i] = x[0] * std::exp(-x[1] * t[i]) - y[i];
    };
    P.jacobian = [calls, t](const Vec& x, sc::Rows& J) {
        ++calls->J;
        J.assign(20, Vec(2));
        for (std::size_t i = 0; i < 20; ++i) {
            const double e = std::exp(-x[1] * t[i]);
            J[i][0] = e;
            J[i][1] = -x[0] * t[i] * e;
        }
    };
    return P;
}

sc::Start from(double a, double b)
{
    sc::Start st;
    st.has_x0 = true;
    st.x0 = {a, b};
    return st;
}

const char* kLM = "{\"solver\":\"lm\"}";
const char* kNM = "{\"local_tol\":1e-12,\"solver\":\"nelder_mead\"}";

void one_stage()
{
    for (const auto& [inner, name] : {std::pair<std::string, std::string>{kLM, "lm"}, {kNM, "nelder_mead"}}) {
        auto c1 = std::make_shared<Calls>(), c2 = std::make_shared<Calls>();
        const sc::Knobs own = name == "nelder_mead" ? sc::Knobs{{"local_tol", number(1e-12)}} : sc::Knobs{};
        const sc::Result d = sc::run("data_continuation", decay(c1), 1e-6, false, 0.0, 10, 400, 3,
                                     {{"inner", word(inner)}}, {}, 0, from(5.0, 2.0));
        const sc::Result s = sc::run(name, decay(c2), 1e-6, false, 0.0, 10, 400, 3, own, {}, 0, from(5.0, 2.0));
        check(d.variables == s.variables && d.objectives == s.objectives && d.evaluations == s.evaluations &&
                  d.grad_calls == s.grad_calls && c1->r == c2->r && c1->J == c2->J,
              "one stage = " + name + " itself, bit for bit (" + std::to_string(d.evaluations) + " evaluations, " +
                  std::to_string(d.grad_calls) + " Jacobians)");
    }
}

void a_window_then_the_problem()
{
    auto c = std::make_shared<Calls>();
    const sc::Knobs kb{{"inner", word(kLM)}, {"dc_stages", word("[{\"window\":[0,10]}]")}, {"dc_share", number(0.5)}};
    const sc::Result d = sc::run("data_continuation", decay(c), 1e-6, false, 0.0, 10, 400, 3, kb, {}, 0, from(5.0, 2.0));
    // stage 1 by itself: lm on the problem with weights (1, 0), 200 evaluations
    auto c1 = std::make_shared<Calls>();
    sc::Problem W = decay(c1);
    W.weights = {1.0, 0.0};
    const sc::Result s1 = sc::run("lm", W, 1e-6, false, 0.0, 10, 200, 3, {}, {}, 0, from(5.0, 2.0));
    const auto minima = mootation::json::parse(setting(d, "minima"));
    bool stage1 = false;                                  // the log's minimum of stage 0 is s1's answer
    for (const auto& m : minima.items)
        if (m.items[1].i == 0)
            stage1 = m.items[0].f == s1.objectives[0][0] && m.items[3].f == s1.variables[0][0] &&
                     m.items[4].f == s1.variables[0][1];
    double f_ans = 0.0;
    {
        Vec r;
        auto c3 = std::make_shared<Calls>();
        decay(c3).residuals(d.variables[0], r);
        for (double v : r) f_ans += v * v;
        f_ans *= 0.5;
    }
    check(stage1, "stage 1 is lm on the problem with weights (1, 0) by itself, bit for bit (a = " +
                      std::to_string(s1.variables[0][0]) + ")");
    check(setting(d, "stages") == "2" && c->r == d.evaluations && c->J == d.grad_calls && d.evaluations <= 400 &&
              d.objectives[0][0] == f_ans,
          "a window [0, 10), then the problem: 2 stages, every call one evaluation (" + std::to_string(d.evaluations) +
              ", " + std::to_string(d.grad_calls) + " Jacobians), the answer's f the problem's 1/2 ||r||^2");
}

void weights_and_refusals()
{
    auto c = std::make_shared<Calls>();
    const sc::Result w = sc::run("data_continuation", decay(c), 1e-6, false, 0.0, 10, 400, 5,
                                 {{"inner", word(kLM)}, {"dc_stages", word("[[1,0.01],[1,0.1]]")}}, {}, 0, from(5.0, 2.0));
    check(setting(w, "stages") == "3" && c->r == w.evaluations && w.evaluations <= 400 && w.objectives[0][0] < 1e-2,
          "two weight stages, then the problem: 3 stages, f = " + std::to_string(w.objectives[0][0]));
    auto refused = [](const sc::Problem& P, const sc::Knobs& k) {
        try {
            sc::run("data_continuation", P, 1e-6, false, 0.0, 10, 100, 1, k, {}, 0, from(5.0, 2.0));
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    sc::Problem plain;
    plain.lo = {0.0};
    plain.hi = {1.0};
    plain.m = 1;
    plain.evaluate = [](const Vec& x, Vec& f, Vec&) { f.assign(1, x[0] * x[0]); };
    const auto P = decay(c);
    check(refused(plain, {{"inner", word(kLM)}}) && refused(P, {{"inner", word(kLM)}, {"dc_stages", word("[[1]]")}}) &&
              refused(P, {{"inner", word(kLM)}, {"dc_stages", word("[{\"window\":[5,30]}]")}}) &&
              refused(P, {{"inner", word(kLM)}, {"dc_stages", word("[[0,0]]")}}) &&
              refused(P, {{"inner", word(kLM)}, {"dc_share", number(1.0)}}) && refused(P, {}),
          "refused: no residuals, a weight per block missing, a window past n_res, all weights 0, dc_share 1, no "
          "inner solver");
}

}   // namespace

int main()
{
    one_stage();
    a_window_then_the_problem();
    weights_and_refusals();
    if (g_failed) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
