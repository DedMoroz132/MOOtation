// SPDX-License-Identifier: Apache-2.0
// A parameter of the problem and other formulations (task 26, items 24.2.7
// and 24.2.11; scalar/param.hpp, scalar/reformulation.hpp):
//   * shrink with one level is the inner solver on the problem at L_0, bit
//     for bit (the degenerate case);
//   * on f_L(x) = (x_0 - 1)^2 + x_1^2 in [0, L]^2, solvable exactly when L >= 1:
//     geometric ends on the last L >= 1 of 2 * 0.9^k, bisect closes on 1,
//     ilsqn's best stays >= 1, ladder visits every level and answers the
//     last;
//   * problem_at's refusals (PARAM-1); carried clips into the new box;
//   * reformulation without other formulations is the inner solver, bit for
//     bit; with a scaled formulation its answer is mapped back into the box;
//     its refusals.

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/scalar/scalar.hpp>

namespace {

namespace sc = mootation::scalar;
using Vec = std::vector<double>;

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

sc::Knob word(const std::string& w) { return {true, w, 0.0}; }
sc::Knob number(double v) { return {false, "", v}; }

std::string setting(const sc::Result& r, const std::string& key)
{
    for (const auto& kv : r.settings)
        if (kv.first == key) return kv.second;
    return "";
}

sc::Problem at(double L)
{
    sc::Problem P;
    P.lo = {0.0, 0.0};
    P.hi = {L, L};
    P.m = 1;
    P.stop_at = 1e-12;
    P.param = L;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) { f.assign(1, (x[0] - 1.0) * (x[0] - 1.0) + x[1] * x[1]); };
    P.at_param = [](double L2) { return at(L2); };
    P.rescale = [](Vec& x, double from, double to) {
        for (double& v : x) v *= to / from;
    };
    return P;
}

// L-BFGS-B: the optimum x_1 = 0 is on the bound, where Nelder-Mead's clipped simplex
// can collapse onto the corner (nelder_mead.hpp NM-2) and stop short
const char* kNM = "{\"lbfgsb_pgtol\":1e-10,\"solver\":\"lbfgsb\"}";   // a level's success needs f <= 1e-12

sc::Result run(const std::string& name, const sc::Problem& P, sc::Knobs k, long long budget, const sc::Start& s,
               unsigned seed = 7)
{
    k.push_back({"inner", word(kNM)});
    return sc::run(name, P, 1e-6, false, 0.0, 10, budget, seed, k, {}, 0, s);
}

bool same(const sc::Result& a, const sc::Result& b)
{
    return a.objectives == b.objectives && a.variables == b.variables && a.evaluations == b.evaluations;
}

double best_param(const sc::Result& r)
{
    const std::string s = setting(r, "param_best");
    return s.empty() ? NAN : std::stod(s);
}

}   // namespace

int main()
{
    sc::Start from;
    from.has_x0 = true;
    from.x0 = {1.7, 0.4};
    const sc::Problem P = at(2.0);

    // ── the degenerate case ─────────────────────────────────────────────────
    {
        const sc::Result s = run("shrink", P, {{"shrink_rounds", number(1)}}, 4000, from);
        const sc::Result i = sc::run("lbfgsb", sc::problem_at(P, 2.0), 1e-6, false, 0.0, 10, 4000, 7,
                                     {{"lbfgsb_pgtol", number(1e-10)}}, {}, 0, from);
        check(same(s, i), "shrink with one level = the inner solver on the problem at L_0, bit for bit");
    }

    // ── the rules ───────────────────────────────────────────────────────────
    {
        const sc::Result g = run("shrink", P, {{"shrink_step", word("geometric")}, {"shrink_rho", number(0.9)}}, 40000, from);
        check(std::fabs(best_param(g) - 2.0 * std::pow(0.9, 6)) < 1e-12,
              "geometric: the last L >= 1 of 2 * 0.9^k, " + setting(g, "param_best"));
        const sc::Result b = run("shrink", P, {{"shrink_step", word("bisect")}, {"shrink_lo", number(0.5)}}, 60000, from);
        check(best_param(b) >= 1.0 - 1e-6 && best_param(b) < 1.0 + 3e-6, "bisect closes on 1: " + setting(b, "param_best") + " " + setting(b, "shrink_trail"));
        const sc::Result q = run("shrink", P, {{"shrink_step", word("ilsqn")}, {"shrink_rounds", number(40)}}, 80000, from);
        check(best_param(q) >= 1.0 - 1e-6 && best_param(q) < 1.05, "ilsqn: the best L stays >= 1, near it: " + setting(q, "param_best") + " " + setting(q, "shrink_trail"));
        const sc::Result l = run("shrink", P, {{"shrink_step", word("ladder")}, {"shrink_ladder", word("[2, 1.5, 1.2]")}}, 9000, from);
        check(setting(l, "shrink_levels") == "3" && best_param(l) == 1.2, "ladder: every level, the last answers");
        check(same(g, run("shrink", P, {{"shrink_step", word("geometric")}, {"shrink_rho", number(0.9)}}, 40000, from)),
              "shrink: the same run twice");
    }

    // ── PARAM-1, carried ────────────────────────────────────────────────────
    {
        bool threw = false;
        try { sc::problem_at(P, NAN); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "problem_at: a non-finite L refused");
        sc::Problem bad = P;
        bad.at_param = [](double L) { sc::Problem Q = at(L); Q.param = NAN; return Q; };
        threw = false;
        try { sc::problem_at(bad, 1.5); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "problem_at: a problem whose param is not L refused");
        sc::Problem none = P;
        none.at_param = nullptr;
        threw = false;
        try { run("shrink", none, {}, 1000, from); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "shrink without at_param refused");
        sc::Problem plain = P;
        plain.rescale = nullptr;
        const Vec c = sc::carried(plain, at(1.0), {1.7, 0.4}, 2.0, 1.0);
        check(c[0] == 1.0 && c[1] == 0.4, "carried without rescale: unchanged, clipped to the new box");
        const Vec r = sc::carried(P, at(1.0), {1.7, 0.4}, 2.0, 1.0);
        check(std::fabs(r[0] - 0.85) < 1e-15 && std::fabs(r[1] - 0.2) < 1e-15, "carried: the source's rescale");
    }

    // ── reformulation ───────────────────────────────────────────────────────
    {
        const sc::Result a = run("reformulation", P, {}, 4000, from);
        const sc::Result i = sc::run("lbfgsb", P, 1e-6, false, 0.0, 10, 4000, 7, {{"lbfgsb_pgtol", number(1e-10)}}, {},
                                     0, from);
        check(same(a, i), "reformulation without other formulations = the inner solver, bit for bit");
        sc::Problem F = P;
        F.n_formulations = 1;
        F.formulation = [](std::size_t) {                     // y = 10 x
            sc::Problem Q;
            Q.lo = {0.0, 0.0};
            Q.hi = {20.0, 20.0};
            Q.m = 1;
            Q.evaluate = [](const Vec& y, Vec& f, Vec&) {
                f.assign(1, (y[0] / 10.0 - 1.0) * (y[0] / 10.0 - 1.0) + (y[1] / 10.0) * (y[1] / 10.0));
            };
            return Q;
        };
        F.to_formulation = [](std::size_t, const Vec& x, Vec& y) { y = {10.0 * x[0], 10.0 * x[1]}; };
        F.from_formulation = [](std::size_t, const Vec& y, Vec& x) { x = {y[0] / 10.0, y[1] / 10.0}; };
        const sc::Result b = run("reformulation", F, {}, 8000, from), b2 = run("reformulation", F, {}, 8000, from);
        check(!b.objectives.empty() && b.objectives[0][0] < 1e-10 && b.variables[0][0] <= 2.0 && same(b, b2),
              "reformulation with a scaled formulation: an answer in the box, the same run twice");
        sc::Problem half = F;
        half.from_formulation = nullptr;
        bool threw = false;
        try { run("reformulation", half, {}, 1000, from); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "reformulation: a formulation without its maps refused");
    }

    // ── replica_ladder (24.2.8; RL-1 ... RL-4) ──────────────────────────────
    {
        const sc::Result one = run("replica_ladder", P, {{"rl_over", word("param")}, {"rl_levels", word("[1.5]")}}, 3000, from);
        const sc::Result i = sc::run("lbfgsb", sc::problem_at(P, 1.5), 1e-6, false, 0.0, 10, 3000, 7,
                                     {{"lbfgsb_pgtol", number(1e-10)}}, {}, 0, from);
        check(same(one, i), "replica_ladder with one level, no exchange = the inner solver at that level, bit for bit");
        const sc::Result ind = run("replica_ladder", P, {{"rl_over", word("param")}, {"rl_levels", word("[1.2, 1.6, 2]")}},
                                   3000, from);
        bool alike = true;
        std::string want = "[";
        const double levels[3] = {1.2, 1.6, 2.0};
        for (int m = 0; m < 3; ++m) {
            const sc::Result r = sc::run("lbfgsb", sc::problem_at(P, levels[m]), 1e-6, false, 0.0, 10, 1000,
                                         sc::strategy::start_seed(7, static_cast<std::uint64_t>(m)),
                                         {{"lbfgsb_pgtol", number(1e-10)}}, {}, 0, from);
            want += (m ? "," : "") + sc::detail::g_format(r.objectives[0][0]);
            alike = alike && !r.objectives.empty();
        }
        want += "]";
        check(alike && setting(ind, "replica_best") == want,
              "replica_ladder without exchange: three independent runs, " + setting(ind, "replica_best"));
        const sc::Knobs sw{{"rl_over", word("param")}, {"rl_levels", word("[1.2, 1.6, 2]")}, {"swap_every", number(200)}};
        const sc::Result a = run("replica_ladder", P, sw, 6000, from), b = run("replica_ladder", P, sw, 6000, from);
        check(same(a, b) && setting(a, "swaps_tried") != "0", "replica_ladder over a parameter: exchanges tried, the same run twice");
        const sc::Knobs kt{{"rl_knob", word("lbfgsb_m")}, {"rl_levels", word("[3, 7]")}, {"swap_every", number(300)}};
        const sc::Result c = run("replica_ladder", P, kt, 4000, from);
        check(!c.objectives.empty() && setting(c, "swaps_tried") != "0", "replica_ladder over a knob of the inner solver");
        bool threw = false;
        try { run("replica_ladder", P, {{"rl_over", word("param")}, {"rl_levels", word("[2]")}, {"rl_fs", number(0)}}, 100, from); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "replica_ladder: rl_fs <= 0 refused");
    }

    std::printf("%s\n", g_failed ? "FAILED" : "all passed");
    return g_failed ? 1 : 0;
}
