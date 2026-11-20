// SPDX-License-Identifier: Apache-2.0
// The sweep with the box's faces moved outward (task 32, layer 4;
// scalar/sweep.hpp), on a shifted sphere f(x) = sum (x_j - c_j)^2:
//   * the minimum outside the starting box, inside the hard bounds: the box
//     reaches it (design plans and the grid; a real and an integer
//     coordinate);
//   * the minimum inside the box: no face moves, and the run is the plain
//     plan bit for bit — the run without hard bounds, and the plan's points
//     evaluated here;
//   * the hard bound is never crossed: the box stops at it, and no point
//     beyond it is evaluated;
//   * the plan's best points start the inner solver (carry_points), the
//     budget one for both;
//   * the stop rule, sweep_budget, the budget, and the refusals.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mootation/designs.hpp>
#include <mootation/json.hpp>
#include <mootation/scalar/scalar.hpp>

namespace {

namespace sc = mootation::scalar;
namespace json = mootation::json;
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

// what the problem was asked: every point's coordinates, smallest and largest
struct Seen {
    Vec lo, hi;
    long long calls = 0;
    bool whole = true;            // coordinate 1 always a whole number (the integer test)
};

// f(x) = sum (x_j - c_j)^2 in the box [lo, hi]
sc::Problem sphere(const Vec& c, double lo, double hi, Seen* seen = nullptr)
{
    sc::Problem P;
    P.lo.assign(c.size(), lo);
    P.hi.assign(c.size(), hi);
    P.m = 1;
    P.evaluate = [c, seen](const Vec& x, Vec& f, Vec&) {
        double s = 0.0;
        for (std::size_t j = 0; j < x.size(); ++j) s += (x[j] - c[j]) * (x[j] - c[j]);
        f.assign(1, s);
        if (seen) {
            if (seen->lo.empty()) seen->lo = seen->hi = x;
            for (std::size_t j = 0; j < x.size(); ++j) {
                seen->lo[j] = (std::min)(seen->lo[j], x[j]);
                seen->hi[j] = (std::max)(seen->hi[j], x[j]);
            }
            if (x.size() > 1 && x[1] != std::floor(x[1])) seen->whole = false;
            ++seen->calls;
        }
    };
    return P;
}

sc::Result run(const sc::Problem& P, const sc::Knobs& k, long long budget, unsigned seed = 7)
{
    return sc::run("sweep", P, 0.0, false, 0.0, 1, budget, seed, k, {}, 0, sc::Start());
}

// the run's last box from sweep_boxes: [lo, hi]
std::pair<Vec, Vec> last_box(const sc::Result& r)
{
    const json::Value v = json::parse(setting(r, "sweep_boxes"));
    const json::Value& b = v.items.back();
    auto list = [](const json::Value& l) {
        Vec out;
        for (const auto& e : l.items) out.push_back(e.kind == json::Value::Kind::Int ? static_cast<double>(e.i) : e.f);
        return out;
    };
    return {list(b.items[0]), list(b.items[1])};
}

bool holds(const std::pair<Vec, Vec>& box, const Vec& c)
{
    for (std::size_t j = 0; j < c.size(); ++j)
        if (!(box.first[j] <= c[j] && c[j] <= box.second[j])) return false;
    return true;
}

bool same(const sc::Result& a, const sc::Result& b)
{
    if (a.objectives != b.objectives || a.variables != b.variables || a.cv != b.cv) return false;
    if (a.evaluations != b.evaluations || a.used != b.used || a.improvements.size() != b.improvements.size())
        return false;
    for (std::size_t i = 0; i < a.improvements.size(); ++i)
        if (a.improvements[i].f != b.improvements[i].f || a.improvements[i].evaluations != b.improvements[i].evaluations) return false;
    return true;
}

std::string text(const Vec& x)
{
    std::string s = "(";
    for (std::size_t j = 0; j < x.size(); ++j) {
        char b[32];
        std::snprintf(b, sizeof b, "%s%.4g", j ? ", " : "", x[j]);
        s += b;
    }
    return s + ")";
}

}   // namespace

int main()
{
    const sc::Knobs hard5 = {{"hard_lo", number(-5.0)}, {"hard_hi", number(5.0)}};

    // ── the minimum outside the box, inside the hard bounds ────────────────
    for (const char* plan : {"sobol", "lhs", "uniform", "halton_rand", "sobol_owen", "grid"}) {
        const Vec c{3.3, -2.7};
        Seen seen;
        const sc::Problem P = sphere(c, -1.0, 1.0, &seen);
        sc::Knobs k = hard5;
        k.push_back({"plan", word(plan)});
        if (std::string(plan) != "grid") k.push_back({"n_points", number(32)});
        const sc::Result r = run(P, k, 20000);
        const auto box = last_box(r);
        check(holds(box, c) && setting(r, "sweep_steps") != "0",
              std::string(plan) + ": the minimum " + text(c) + " outside [-1, 1]^2 is reached: box " + text(box.first) +
                  " - " + text(box.second) + ", steps " + setting(r, "sweep_steps") + ", best " + text(r.variables[0]));
        check(r.evaluations == seen.calls && r.evaluations == std::stoll(setting(r, "sweep_points")) &&
                  r.used <= 20000.0 && r.stopped_by == "rule",
              std::string(plan) + ": " + std::to_string(r.evaluations) + " evaluations, all counted, the sweep's own end");
        const sc::Result plain = run(P, {{"plan", word(plan)}, {"n_points", number(32)}}, 20000);
        check(r.objectives[0][0] < plain.objectives[0][0],
              std::string(plan) + ": better than the plan in the starting box");
    }

    // ── the minimum inside: no face moves, the plain plan bit for bit ──────
    for (const char* plan : {"sobol", "lhs", "grid"}) {
        const Vec c{0.3, -0.2};
        const sc::Problem P = sphere(c, -1.0, 1.0);
        sc::Knobs k = hard5, k0;
        k.push_back({"plan", word(plan)});
        k0.push_back({"plan", word(plan)});
        const sc::Result r = run(P, k, 5000), plain = run(P, k0, 5000);
        check(same(r, plain) && setting(r, "sweep_expansions") == "[]" && setting(r, "sweep_steps") == "0",
              std::string(plan) + ": the minimum inside, hard bounds wider: no expansion, = the plain plan, bit for bit");
    }
    // the defaults over seeds 1 ... 10 (not one seed's luck): the minimum past
    // a corner reached with 32 and 64 points; one 0.35 of the width from a
    // face moves none with the default 64 points (with 32 the best points'
    // spread may reach the band, SWEEP-7: counted, not required)
    for (const char* plan : {"sobol", "sobol_owen", "lhs"})
        for (const long long np : {32LL, 64LL}) {
            int reached = 0, quiet = 0;
            for (unsigned s = 1; s <= 10; ++s) {
                sc::Knobs k = hard5;
                k.push_back({"plan", word(plan)});
                k.push_back({"n_points", number(static_cast<double>(np))});
                const Vec out{3.3, -2.7}, in{0.3, -0.2};
                reached += holds(last_box(run(sphere(out, -1.0, 1.0), k, 20000, s)), out) ? 1 : 0;
                quiet += setting(run(sphere(in, -1.0, 1.0), k, 20000, s), "sweep_steps") == "0" ? 1 : 0;
            }
            check(reached == 10 && (np == 32 || quiet == 10),
                  std::string(plan) + ", " + std::to_string(np) + " points, seeds 1 ... 10: the corner minimum reached " +
                      std::to_string(reached) + " times, no move for the inner one " + std::to_string(quiet) + " times");
        }
    {
        // the plan's points evaluated here: its best, the first of equals
        const Vec c{0.3, -0.2};
        const sc::Problem P = sphere(c, -1.0, 1.0);
        const sc::Result r = run(P, hard5, 5000, 11);
        const std::vector<double> U = mootation::designs::design("sobol", 64, 2, 11);
        double best = 1e300;
        Vec xb;
        for (std::size_t i = 0; i < 64; ++i) {
            const Vec x{-1.0 + U[2 * i] * 2.0, -1.0 + U[2 * i + 1] * 2.0};
            const double f = (x[0] - c[0]) * (x[0] - c[0]) + (x[1] - c[1]) * (x[1] - c[1]);
            if (f < best) {
                best = f;
                xb = x;
            }
        }
        check(r.objectives[0][0] == best && r.variables[0] == xb && r.evaluations == 64,
              "sobol, 64 points (the defaults): the answer is the design's best point, bit for bit");
    }

    // ── the hard bound is never crossed ────────────────────────────────────
    for (const char* plan : {"sobol", "grid"}) {
        const Vec c{8.0, 0.0};
        Seen seen;
        const sc::Problem P = sphere(c, -1.0, 1.0, &seen);
        sc::Knobs k = hard5;
        k.push_back({"plan", word(plan)});
        const sc::Result r = run(P, k, 20000);
        const auto box = last_box(r);
        check(box.second[0] == 5.0 && seen.hi[0] <= 5.0 && seen.lo[0] >= -5.0 && seen.hi[1] <= 5.0 &&
                  seen.lo[1] >= -5.0 && r.stopped_by == "rule",
              std::string(plan) + ": minimum at x0 = 8 beyond hard_hi = 5: the box stops at 5 (" + text(box.second) +
                  "), the largest x0 evaluated " + text(seen.hi));
    }

    // ── an integer coordinate ──────────────────────────────────────────────
    for (const char* plan : {"sobol", "grid"}) {
        const Vec c{0.2, 7.0};
        Seen seen;
        sc::Problem P = sphere(c, 0.0, 3.0, &seen);
        P.lo[0] = -1.0;
        P.hi[0] = 1.0;
        P.discrete = {Vec(), Vec{0.0, 1.0, 2.0, 3.0}};
        sc::Knobs k = {{"plan", word(plan)}, {"hard_lo", word("[-1, 0]")}, {"hard_hi", word("[1, 10.5]")}};
        const sc::Result r = run(P, k, 20000);
        const auto box = last_box(r);
        check(holds(box, c) && box.second[1] <= 10.0 && seen.whole && seen.hi[1] <= 10.0 && r.variables[0][1] == 7.0 &&
                  box.first[0] == -1.0 && box.second[0] == 1.0,
              std::string(plan) + ": integer x1 in 0 ... 3, minimum at 7, hard 10.5 -> 10: box " + text(box.second) +
                  ", every x1 whole, best x1 = 7, x0 fixed by its hard bounds; " + setting(r, "sweep_expansions"));
    }

    // ── a listed coordinate that is not whole numbers keeps its values ──────
    {
        const Vec c{0.2, 0.9};
        sc::Problem P = sphere(c, -1.0, 1.0);
        P.discrete = {Vec(), Vec{-0.5, 0.25, 0.5}};
        const sc::Result r = run(P, hard5, 5000);
        const auto box = last_box(r);
        check(setting(r, "sweep_fixed") == "[1]" && box.first[1] == -0.5 && box.second[1] == 0.5,
              "a discrete coordinate of other values: its faces stay, sweep_fixed = " + setting(r, "sweep_fixed"));
    }

    // ── the plan's best points start the inner solver ──────────────────────
    {
        const Vec c{3.3, -2.7};
        const sc::Problem P = sphere(c, -1.0, 1.0);
        sc::Knobs k = hard5;
        k.push_back({"inner", word("{\"solver\":\"nelder_mead\"}")});
        k.push_back({"carry_points", number(3)});
        const sc::Result r = run(P, k, 3000), alone = run(P, hard5, 3000);
        check(r.objectives[0][0] < alone.objectives[0][0] && r.objectives[0][0] < 1e-6 && r.used <= 3000.0 &&
                  setting(r, "sweep_carried") == "3" && setting(r, "sweep_inner_runs") == "3" && r.outside_evals > 0 &&
                  r.evaluations == alone.evaluations,
              "inner nelder_mead from the 3 best plan points: f " + std::to_string(r.objectives[0][0]) + ", used " +
                  std::to_string(r.used) + " of 3000, the plan's evaluations " + std::to_string(r.evaluations));
        const sc::Result again = run(P, k, 3000);
        check(same(r, again), "the same run twice, bit for bit");
    }

    // ── the stop rule, sweep_budget, the budget ────────────────────────────
    {
        const Vec c{3.3, -2.7};
        const sc::Problem P = sphere(c, -1.0, 1.0);
        sc::Knobs k = hard5;
        k.push_back({"max_steps", number(1)});
        const sc::Result r = run(P, k, 20000);
        check(setting(r, "sweep_steps") == "1" && r.stopped_by == "steps", "max_steps = 1: one move, stopped_by steps");
        sc::Knobs kc = hard5;
        kc.push_back({"sweep_budget", number(100)});
        const sc::Result rc = run(P, kc, 20000);
        check(rc.evaluations == 100 && rc.stopped_by == "sweep_budget", "sweep_budget = 100: the plan stops at 100");
        const sc::Result rb = run(P, hard5, 90);
        check(rb.evaluations == 90 && rb.stopped_by == "budget", "budget 90: 90 evaluations, stopped_by budget");
    }

    // ── the refusals ───────────────────────────────────────────────────────
    {
        const sc::Problem P = sphere({0.0, 0.0}, -1.0, 1.0);
        auto refused = [&](const sc::Knobs& k, const std::string& what) {
            bool threw = false;
            try {
                run(P, k, 100);
            } catch (const std::invalid_argument&) {
                threw = true;
            }
            check(threw, "refused: " + what);
        };
        refused({{"hard_lo", number(0.0)}}, "hard_lo inside the box");
        refused({{"hard_hi", word("[2, 3, 4]")}}, "hard_hi of 3 numbers for 2 variables");
        refused({{"edge_share", number(0.0)}}, "edge_share = 0");
        refused({{"expand_step", number(0.0)}}, "expand_step = 0");
        refused({{"plan", word("halton")}}, "an unknown plan");
        sc::Problem P2 = sphere({0.0, 0.0}, -1.0, 1.0);
        P2.m = 2;
        bool threw = false;
        try {
            run(P2, {}, 100);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "refused: two objectives");
    }

    std::printf("%s\n", g_failed ? "FAILED" : "all passed");
    return g_failed ? 1 : 0;
}
