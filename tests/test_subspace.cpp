// SPDX-License-Identifier: Apache-2.0
// Strategies over subspaces (task 26, items 24.2.9 and 24.2.10;
// scalar/subspace.hpp):
//   * coevolution with one group of every variable and one cycle, from an
//     x0, is the inner solver from x0 with the whole budget: the run of
//     multistart with one start, bit for bit (the degenerate case);
//   * coevolution on overlapping windows and on the neighbors hook's groups
//     solves a chain-coupled problem, the same run twice; its refusals;
//   * lns from a start improves the point by windows, monotonely; worst
//     needs the contributions hook; the same run twice.

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

// f = sum (x_i - x_{i+1})^2 + sum (x_i - 0.3)^2: each variable meets its two
// neighbours; the minimum 0 at x = 0.3
sc::Problem chain(int n)
{
    sc::Problem P;
    P.lo.assign(static_cast<std::size_t>(n), -1.0);
    P.hi.assign(static_cast<std::size_t>(n), 1.0);
    P.m = 1;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) {
        double s = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            s += (x[i] - 0.3) * (x[i] - 0.3);
            if (i + 1 < x.size()) s += (x[i] - x[i + 1]) * (x[i] - x[i + 1]);
        }
        f.assign(1, s);
    };
    return P;
}

void with_neighbors(sc::Problem& P)
{
    P.neighbors = [](const Vec& x) {
        std::vector<std::vector<std::size_t>> nb(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) {
            if (i > 0) nb[i].push_back(i - 1);
            if (i + 1 < x.size()) nb[i].push_back(i + 1);
        }
        return nb;
    };
}

const char* kNM = "{\"local_tol\":1e-10,\"solver\":\"nelder_mead\"}";

sc::Result run(const std::string& name, const sc::Problem& P, sc::Knobs k, long long budget, const sc::Start& s,
               unsigned seed = 5)
{
    k.push_back({"inner", word(kNM)});
    return sc::run(name, P, 1e-6, false, 0.0, 10, budget, seed, k, {}, 0, s);
}

bool same(const sc::Result& a, const sc::Result& b)
{
    return a.objectives == b.objectives && a.variables == b.variables && a.evaluations == b.evaluations;
}

double best(const sc::Result& r) { return r.objectives.empty() ? INFINITY : r.objectives[0][0]; }

}   // namespace

int main()
{
    const int n = 12;
    sc::Start from;
    from.has_x0 = true;
    from.x0.assign(static_cast<std::size_t>(n), -0.5);

    // ── the degenerate case: one group, one cycle = the inner solver ────────
    {
        const sc::Problem P = chain(n);
        const sc::Result cc = run("coevolution", P, {{"cc_window", number(n)}, {"cc_cycles", number(1)}}, 3000, from);
        const sc::Result ms = run("multistart", P, {{"n_starts", number(1)}}, 3000, from);
        check(same(cc, ms), "coevolution with one group and one cycle = multistart with one start, bit for bit");
    }

    // ── coevolution on windows and on neighbors ─────────────────────────────
    {
        const sc::Problem P = chain(n);
        const sc::Knobs win{{"cc_window", number(4)}, {"cc_stride", number(2)}};
        const sc::Result a = run("coevolution", P, win, 40000, from), b = run("coevolution", P, win, 40000, from);
        check(best(a) < 1e-6, "coevolution on windows of 4 (stride 2) solves the chain: f = " + std::to_string(best(a)));
        check(same(a, b), "coevolution: the same run twice");
        check(setting(a, "groups") == "5", "coevolution: windows 0, 2, 4, 6, 8 of 4 over 12 variables");
        sc::Problem Q = chain(n);
        with_neighbors(Q);
        const sc::Result c = run("coevolution", Q, {{"cc_groups", word("neighbors")}}, 40000, from);
        check(best(c) < 1e-6 && setting(c, "groups") == std::to_string(n),
              "coevolution on the neighbors hook's groups (CC-2) solves the chain");
        bool threw = false;
        try { run("coevolution", P, {{"cc_groups", word("neighbors")}}, 1000, from); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "coevolution: cc_groups = neighbors without the hook refused");
        threw = false;
        try { run("coevolution", P, {{"cc_window", number(3)}, {"cc_stride", number(4)}}, 1000, from); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "coevolution: groups that do not overlap (stride > window) refused");
        const sc::Result u = run("coevolution", P, win, 20000, {});
        check(best(u) < 1e-3, "coevolution from a uniform point (CC-3)");
    }

    // ── lns ─────────────────────────────────────────────────────────────────
    {
        sc::Problem P = chain(n);
        const sc::Knobs k{{"lns_size", number(3)}};
        const sc::Result a = run("lns", P, k, 30000, from), b = run("lns", P, k, 30000, from);
        check(best(a) < 1e-4, "lns by windows of 3 improves the chain: f = " + std::to_string(best(a)));
        check(same(a, b), "lns: the same run twice");
        bool threw = false;
        try { run("lns", P, {{"lns_window", word("worst")}}, 1000, from); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "lns: lns_window = worst without the contributions hook refused (LNS-2)");
        P.contributions = [](const Vec& x, Vec& c) {
            c.assign(x.size(), 0.0);
            for (std::size_t i = 0; i < x.size(); ++i) c[i] = (x[i] - 0.3) * (x[i] - 0.3);
        };
        with_neighbors(P);
        const sc::Result w = run("lns", P, {{"lns_window", word("worst")}}, 30000, from);
        check(best(w) < 1e-4, "lns: the worst contribution's window with its neighbors");
    }

    std::printf("%s\n", g_failed ? "FAILED" : "all passed");
    return g_failed ? 1 : 0;
}
