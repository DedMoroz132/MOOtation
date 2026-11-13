// SPDX-License-Identifier: Apache-2.0
// The rest of task 24's §24.4.2 and §24.2.5:
//   * periodic variables in lbfgsb, lm and slsqp: met from 0.9, a minimum at
//     0.02 found across the seam, every point in [lo, hi);
//   * the symmetries modulo a period (symmetry.hpp SYM-11): align brings a
//     periodic coordinate next to ref, distance takes the shorter way;
//   * a variable's own repair (Problem::var_repair): every entry the
//     method's own (-1, or de's reflect) is the run without it; clip on one
//     variable puts points on its bounds and on no other; resample and a
//     wrong length are refused;
//   * mlsl_cells (search.hpp SRCH-14): the full scan's decisions, so the
//     same run, in 2 and in 5 dimensions; n > 10 refused.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/scalar/scalar.hpp>

namespace sc = mootation::scalar;
namespace sym = mootation::scalar::symmetry;
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

static bool same(const sc::Result& a, const sc::Result& b)
{
    if (a.improvements.size() != b.improvements.size()) return false;
    for (std::size_t i = 0; i < a.improvements.size(); ++i)
        if (std::memcmp(&a.improvements[i], &b.improvements[i], sizeof a.improvements[i]) != 0) return false;
    return a.objectives == b.objectives && a.variables == b.variables && a.evaluations == b.evaluations &&
           a.grad_calls == b.grad_calls;
}

static const double kPi = 3.14159265358979323846;

// x0 an angle on [0, 1) with its minimum at 0.02, x1 in [-1, 1]; as an
// objective with its gradient, or as residuals with their Jacobian
static sc::Problem angle(std::shared_ptr<Rows> asked, bool residuals)
{
    sc::Problem P;
    P.lo = {0.0, -1.0};
    P.hi = {1.0, 1.0};
    P.m = 1;
    P.periodic = {1, 0};
    if (!residuals) {
        P.evaluate = [asked](const Vec& x, Vec& f, Vec&) {
            if (asked) asked->push_back(x);
            f[0] = 1.0 - mootation::dm::cos(2.0 * kPi * (x[0] - 0.02)) + (x[1] - 0.3) * (x[1] - 0.3);
        };
        P.gradient = [](const Vec& x, Vec& g, Rows&) {
            g = {2.0 * kPi * mootation::dm::sin(2.0 * kPi * (x[0] - 0.02)), 2.0 * (x[1] - 0.3)};
        };
    } else {
        P.residuals = [asked](const Vec& x, Vec& r) {
            if (asked) asked->push_back(x);
            r = {mootation::dm::sin(kPi * (x[0] - 0.02)), x[1] - 0.3};
        };
        P.jacobian = [](const Vec& x, Rows& J) {
            J = {{kPi * mootation::dm::cos(kPi * (x[0] - 0.02)), 0.0}, {0.0, 1.0}};
        };
    }
    return P;
}

static sc::Problem rastrigin(std::size_t n, std::shared_ptr<Rows> asked = nullptr)
{
    sc::Problem P;
    P.lo.assign(n, -5.12);
    P.hi.assign(n, 5.12);
    P.m = 1;
    P.evaluate = [asked](const Vec& x, Vec& f, Vec&) {
        if (asked) asked->push_back(x);
        double s = 10.0 * static_cast<double>(x.size());
        for (double v : x) s += v * v - 10.0 * mootation::dm::cos(2.0 * kPi * v);
        f[0] = s;
    };
    P.gradient = [](const Vec& x, Vec& g, Rows&) {
        g.assign(x.size(), 0.0);
        for (std::size_t i = 0; i < x.size(); ++i) g[i] = 2.0 * x[i] + 20.0 * kPi * mootation::dm::sin(2.0 * kPi * x[i]);
    };
    return P;
}

static sc::Knob word(const std::string& w) { return {true, w, 0.0}; }
static sc::Knob number(double v) { return {false, "", v}; }

int main()
{
    // periodic variables in the local solvers
    for (const std::string m : {"lbfgsb", "lm", "slsqp"}) {
        auto asked = std::make_shared<Rows>();
        sc::Start st;
        st.has_x0 = true;
        st.x0 = {0.9, 0.0};
        const sc::Result r = sc::run(m, angle(asked, m == "lm"), 1e-6, false, 0.0, 20, 2000, 3, {}, {}, 0, st);
        bool inside = !asked->empty();
        for (const Vec& x : *asked) inside = inside && x[0] >= 0.0 && x[0] < 1.0;
        const double off = sc::periodic::gap(r.variables[0][0], 0.02, true, 1.0);
        check(inside, m + ": every point in [lo, hi)");
        check(off < 1e-5, m + ": the minimum across the seam (off " + std::to_string(off) + ")");
        std::printf("  %-7s from 0.9: x0 = %.8f, %zu points\n", m.c_str(), r.variables[0][0], asked->size());
    }

    // the symmetries modulo a period (SYM-11)
    {
        const Vec period{1.0, 0.0};
        const Vec a = sym::align({0.98, 0.5}, {0.01, 0.5}, {}, {}, period);
        check(std::fabs(a[0] + 0.02) < 1e-15 && a[1] == 0.5, "align: the periodic coordinate next to ref");
        check(std::fabs(sym::distance({0.98, 0.5}, {0.01, 0.5}, {}, {}, period) - 0.03) < 1e-15,
              "distance: the shorter way round");
        const Vec both{1.0, 1.0};
        const std::vector<sym::Blocks> g{{0, 1, 2}};
        const Vec b = sym::align({0.98, 0.5}, {0.5, 0.01}, g, {}, both);
        check(std::fabs(b[0] - 0.5) < 1e-15 && std::fabs(b[1] + 0.02) < 1e-15,
              "align with blocks: matched the short way, then next to ref");
        check(sym::align({0.98, 0.5}, {0.01, 0.5}, {}) == Vec({0.98, 0.5}), "without periods: as before");
    }

    // a variable's own repair
    {
        auto plain = std::make_shared<Rows>(), clip = std::make_shared<Rows>();
        sc::Problem A = rastrigin(3, plain), B = rastrigin(3), C = rastrigin(3, clip), D = rastrigin(3);
        B.var_repair = {-1, -1, -1};
        C.var_repair = {0, -1, -1};                                              // clip on x0
        D.var_repair = {1, 1, 1};                                                // reflect: de's own
        const sc::Result a = sc::run("de", A, 1e-6, false, 0.0, 20, 3000, 3, {}, {}, 0, {});
        const sc::Result b = sc::run("de", B, 1e-6, false, 0.0, 20, 3000, 3, {}, {}, 0, {});
        const sc::Result d = sc::run("de", D, 1e-6, false, 0.0, 20, 3000, 3, {}, {}, 0, {});
        sc::run("de", C, 1e-6, false, 0.0, 20, 3000, 3, {}, {}, 0, {});
        check(same(a, b) && same(a, d), "var_repair: the method's own repair everywhere is the run without it");
        long long on0 = 0, on1 = 0;
        for (const Vec& x : *clip) {
            on0 += x[0] == -5.12 || x[0] == 5.12;
            on1 += x[1] == -5.12 || x[1] == 5.12 || x[2] == -5.12 || x[2] == 5.12;
        }
        check(on0 > 0 && on1 == 0, "var_repair: clip on x0 puts points on its bounds (" + std::to_string(on0) +
                                       "), none on the others'");
        for (const std::string m : {"pso", "jade", "g3_pcx", "spso2011"}) {
            sc::Problem E = rastrigin(3);
            E.var_repair = {0, -1, -1};
            const sc::Result e = sc::run(m, E, 1e-6, false, 0.0, 20, 2000, 3, {}, {}, 0, {});
            check(!e.objectives.empty(), m + ": runs with a variable's own repair");
        }
        auto refused = [](std::vector<int> mask, const std::string& part) {
            sc::Problem P = rastrigin(3);
            P.var_repair = std::move(mask);
            std::string why;
            try {
                sc::run("de", P, 1e-6, false, 0.0, 20, 100, 3, {}, {}, 0, {});
            } catch (const std::invalid_argument& e) {
                why = e.what();
            }
            check(why.find(part) != std::string::npos, "refused: '" + why + "'");
        };
        refused({4, -1, -1}, "not 4");
        refused({0, -1}, "2 entries for 3 variables");
    }

    // mlsl_cells: the full scan's decisions
    for (std::size_t n : {std::size_t{2}, std::size_t{5}}) {
        const sc::Knobs a{{"inner", word("{\"solver\":\"lbfgsb\"}")}, {"mlsl_n", number(200)}};
        sc::Knobs b = a;
        b.push_back({"mlsl_cells", word("on")});
        const sc::Result ra = sc::run("mlsl", rastrigin(n), 1e-6, false, 0.0, 20, 20000, 7, a, {}, 0, {});
        const sc::Result rb = sc::run("mlsl", rastrigin(n), 1e-6, false, 0.0, 20, 20000, 7, b, {}, 0, {});
        check(same(ra, rb), "mlsl_cells, n = " + std::to_string(n) + ": the full scan's run");
        std::printf("  mlsl, rastrigin n = %zu: %lld evaluations, f = %.4g, with cells the same: %s\n", n,
                    ra.evaluations, ra.objectives[0][0], same(ra, rb) ? "yes" : "no");
    }
    {
        std::string why;
        try {
            sc::run("mlsl", rastrigin(11), 1e-6, false, 0.0, 20, 1000, 7,
                    {{"inner", word("{\"solver\":\"lbfgsb\"}")}, {"mlsl_cells", word("on")}}, {}, 0, {});
        } catch (const std::invalid_argument& e) {
            why = e.what();
        }
        check(why.find("n <= 10") != std::string::npos, "mlsl_cells at n = 11 refused");
    }

    if (failures) {
        std::printf("test_bounds_mask: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_bounds_mask: all checks passed\n");
    return 0;
}
