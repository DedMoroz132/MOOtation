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
//     its refusals;
//   * max_seconds (task 30, §30.1; PARAM-7): problem_at carries the run's
//     deadline; with max_seconds every rule of shrink ends by the deadline
//     (plus a level), stopped_by = deadline where the rule would go on; no
//     stopped_by without it;
//   * shrink_count (task 30, §30.2; PARAM-8; budget by default, method
//     version 5): with an inner solver on a gradient the levels charged by
//     evaluations go past the run's budget, charged by budget they stay
//     within it; evaluations gives the numbers of the baseline 8909670 bit
//     for bit (kOld*, from that commit's headers); without gradients the two
//     are one run; the default is kDefaultCount;
//   * shrink_carry = population (task 30, §30.6; PARAM-9): cluster_ga's
//     last population goes to the next level, the number of members kept;
//     point is the default run bit for bit; an inner solver that cannot
//     take it refused.

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

// the same box at every level (param L, any L): geometric, ilsqn and idts
// never end by themselves on it, every level a success at its first point
sc::Problem flat(double L)
{
    sc::Problem P = at(2.0);
    P.param = L;
    P.at_param = [](double L2) { return flat(L2); };
    P.rescale = nullptr;
    return P;
}

// the default of shrink_count (PARAM-8), and the scenario below at 8909670
// (no shrink_count then): what shrink_count = evaluations gives again
const char* kDefaultCount = "budget";
const long long kOldEvaluations = 400, kOldGradCalls = 398;
const double kOldF = 0x1.8ac3333333333p+12;

// Rosenbrock in [-L, L]^10 with its gradient; every level starts again at
// (-1.2, ..., -1.2) (the rescale hook), so that each spends its allowance
sc::Problem rosen(double L)
{
    sc::Problem P;
    P.lo.assign(10, -L);
    P.hi.assign(10, L);
    P.m = 1;
    P.param = L;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) {
        double s = 0.0;
        for (std::size_t i = 0; i + 1 < x.size(); ++i)
            s += 100.0 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i]) + (1.0 - x[i]) * (1.0 - x[i]);
        f.assign(1, s);
    };
    P.gradient = [](const Vec& x, Vec& g, sc::Rows&) {
        g.assign(x.size(), 0.0);
        for (std::size_t i = 0; i + 1 < x.size(); ++i) {
            const double t = x[i + 1] - x[i] * x[i];
            g[i] += -400.0 * x[i] * t - 2.0 * (1.0 - x[i]);
            g[i + 1] += 200.0 * t;
        }
    };
    P.at_param = [](double L2) { return rosen(L2); };
    P.rescale = [](Vec& x, double, double) { x.assign(x.size(), -1.2); };
    return P;
}

// the sum of column c of shrink_counts' rows
double column(const sc::Result& r, std::size_t c)
{
    double s = 0.0;
    for (const auto& row : mootation::json::parse(setting(r, "shrink_counts")).items)
        s += row.items[c].kind == mootation::json::Value::Kind::Int ? static_cast<double>(row.items[c].i) : row.items[c].f;
    return s;
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

    // ── max_seconds (task 30, §30.1; PARAM-7) ──────────────────────────────
    {
        sc::Problem D = P;
        D.deadline = 5.0;
        check(sc::problem_at(D, 1.5).deadline == 5.0 && std::isinf(sc::problem_at(P, 1.5).deadline),
              "problem_at: the problem at L keeps the run's deadline (PARAM-7)");
        const sc::Result plain = run("shrink", P, {{"shrink_step", word("geometric")}, {"shrink_rho", number(0.9)}}, 40000, from);
        check(setting(plain, "stopped_by").empty() && setting(plain, "time_up").empty(),
              "shrink without max_seconds: no stopped_by, as before");
        const sc::Problem F = flat(2.0);
        for (const char* rule : {"geometric", "bisect", "ilsqn", "idts", "ladder"}) {
            sc::Knobs k{{"shrink_step", word(rule)}, {"max_seconds", number(0.3)}, {"shrink_rounds", number(200000)}};
            if (std::string(rule) == "bisect") k.push_back({"shrink_lo", number(0.5)});
            if (std::string(rule) == "ladder") k.push_back({"shrink_ladder", word("[2, 1.5, 1.2]")});
            const sc::Result r = run("shrink", F, k, 10000000000LL, from);
            const bool endless = std::string(rule) != "bisect" && std::string(rule) != "ladder";
            check(r.seconds < 1.3 && (endless ? setting(r, "stopped_by") == "deadline" && r.time_up
                                              : setting(r, "stopped_by") == "rule"),
                  std::string("shrink_step = ") + rule + " with max_seconds = 0.3: " + std::to_string(r.seconds) +
                      " s, stopped_by = " + setting(r, "stopped_by") + ", levels " + setting(r, "shrink_levels"));
        }
    }

    // ── shrink_count (task 30, §30.2; PARAM-8) ─────────────────────────────
    {
        const sc::Problem R = rosen(2.0);
        sc::Start far;
        far.has_x0 = true;
        far.x0.assign(10, -1.2);
        std::string ladder = "[2";
        for (int i = 1; i < 40; ++i) ladder += ",2";
        ladder += "]";
        auto with = [&](const sc::Problem& Q, const char* count) {
            sc::Knobs k{{"shrink_step", word("ladder")}, {"shrink_ladder", word(ladder)}, {"start_budget", number(40)}};
            if (count) k.push_back({"shrink_count", word(count)});
            k.push_back({"inner", word("{\"solver\":\"lbfgsb\"}")});
            return sc::run("shrink", Q, 1e-6, false, 0.0, 10, 400, 7, k, {}, 0, far);
        };
        const sc::Result d = with(R, nullptr), e = with(R, "evaluations"), b = with(R, "budget");
        check(same(d, with(R, kDefaultCount)) && setting(d, "shrink_count").empty() && !setting(d, "shrink_counts").empty(),
              std::string("shrink_count: the default is ") + kDefaultCount + ", the levels' counts in the settings");
        check(e.evaluations == kOldEvaluations && e.grad_calls == kOldGradCalls && !e.objectives.empty() &&
                  e.objectives[0][0] == kOldF,
              "shrink_count = evaluations: the baseline's run bit for bit (" + std::to_string(e.evaluations) + ", " +
                  std::to_string(e.grad_calls) + ")");
        const long long used_e = e.evaluations + e.grad_calls, used_b = b.evaluations + b.grad_calls;
        check(e.grad_calls > 0 && used_e > 400,
              "shrink_count = evaluations with gradients: the levels spend past the budget, " + std::to_string(used_e) + " > 400");
        check(used_b <= 400 && column(b, 2) <= 400.0 && static_cast<long long>(column(b, 0)) == b.evaluations &&
                  static_cast<long long>(column(b, 1)) == b.grad_calls,
              "shrink_count = budget: evaluations + gradient calls " + std::to_string(used_b) + " <= 400, the levels' rows add up");
        sc::Problem plain = rosen(2.0);
        plain.gradient = nullptr;
        plain.at_param = [](double L2) { sc::Problem Q = rosen(L2); Q.gradient = nullptr; return Q; };
        check(same(with(plain, "evaluations"), with(plain, "budget")),
              "shrink_count: without gradients evaluations and budget are one run");
    }

    // ── shrink_carry (task 30, §30.6; PARAM-9) ─────────────────────────────
    {
        const char* cga = "{\"cga_init\":\"seed_points\",\"cga_pop\":4,\"inner\":{\"solver\":\"nelder_mead\"},"
                          "\"solver\":\"cluster_ga\",\"start_budget\":200}";
        // without the target: a level that reached it in its first round would end before its population
        sc::Problem N = P;
        N.stop_at = -INFINITY;
        N.at_param = [](double L2) {
            sc::Problem Q = at(L2);
            Q.stop_at = -INFINITY;
            return Q;
        };
        auto with = [&](const char* carry) {
            sc::Knobs k{{"shrink_step", word("ladder")}, {"shrink_ladder", word("[2, 1.5, 1.2]")},
                        {"start_budget", number(2000)}, {"inner", word(cga)}};
            if (carry) k.push_back({"shrink_carry", word(carry)});
            return sc::run("shrink", N, 1e-6, false, 0.0, 10, 6000, 7, k, {}, 0, from);
        };
        const sc::Result d = with(nullptr), pt = with("point"), pop = with("population"), pop2 = with("population");
        check(same(d, pt) && setting(d, "shrink_carried").empty(), "shrink_carry = point: the default run, bit for bit");
        check(same(pop, pop2) && setting(pop, "shrink_carried") == "[0,4,4]" && setting(pop, "init_seeded") == "3",
              "shrink_carry = population: 4 members carried to every level, " + setting(pop, "shrink_carried"));
        bool threw = false;
        try {
            sc::Knobs k{{"shrink_carry", word("population")}, {"inner", word(kNM)}};
            sc::run("shrink", P, 1e-6, false, 0.0, 10, 1000, 7, k, {}, 0, from);
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "shrink_carry = population with an inner solver other than cluster_ga refused");
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
