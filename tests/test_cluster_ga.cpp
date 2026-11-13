// SPDX-License-Identifier: Apache-2.0
// cluster_ga (task 26, item 24.2.9; scalar/cluster_ga.hpp):
//   * cga_pop = 1, cga_de = 0 is pbh with pbh_K = 1 (monotonic basin
//     hopping), bit for bit — the degenerate case;
//   * the built-in cut and splice (CGA-1): k points of the first parent as
//     they are, N - k of the second moved into its frame, inside the box;
//   * the refusals; the crossover hook's children; the same run twice;
//   * a short check on the 7-atom Lennard-Jones cluster: the global minimum
//     -16.505384 (Wales & Doye 1997, Table I).

#include <cmath>
#include <cstdio>
#include <random>
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

sc::Problem rastrigin(int n)
{
    sc::Problem P;
    P.lo.assign(static_cast<std::size_t>(n), -5.12);
    P.hi.assign(static_cast<std::size_t>(n), 5.12);
    P.m = 1;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) {
        double s = 10.0 * static_cast<double>(x.size());
        for (double v : x) s += v * v - 10.0 * mootation::dm::cos(2.0 * 3.14159265358979323846 * v);
        f.assign(1, s);
    };
    return P;
}

// E = 4 sum (r^-12 - r^-6), N atoms in [-L, L]^3N, its gradient
sc::Problem lj(int N)
{
    sc::Problem P;
    const double L = 1.0 + std::cbrt(3.0 * N / (4.0 * 3.14159265358979323846));
    P.lo.assign(static_cast<std::size_t>(3 * N), -L);
    P.hi.assign(static_cast<std::size_t>(3 * N), L);
    P.m = 1;
    P.sym_blocks = {sc::symmetry::Blocks{0, 3, static_cast<std::size_t>(N)}};
    auto energy = [N](const Vec& x, Vec* g) {
        double e = 0.0;
        if (g) g->assign(x.size(), 0.0);
        for (int i = 0; i < N; ++i)
            for (int j = i + 1; j < N; ++j) {
                double r2 = 0.0;
                for (int c = 0; c < 3; ++c) r2 += (x[3 * i + c] - x[3 * j + c]) * (x[3 * i + c] - x[3 * j + c]);
                const double s = 1.0 / r2, s3 = s * s * s;
                e += 4.0 * (s3 * s3 - s3);
                if (g) {
                    const double de = 8.0 * s * s3 * (3.0 - 6.0 * s3);
                    for (int c = 0; c < 3; ++c) {
                        const double t = de * (x[3 * i + c] - x[3 * j + c]);
                        (*g)[static_cast<std::size_t>(3 * i + c)] += t;
                        (*g)[static_cast<std::size_t>(3 * j + c)] -= t;
                    }
                }
            }
        return e;
    };
    P.evaluate = [energy](const Vec& x, Vec& f, Vec&) { f.assign(1, energy(x, nullptr)); };
    P.gradient = [energy](const Vec& x, Vec& gf, sc::Rows&) { energy(x, &gf); };
    return P;
}

const char* kNM = "{\"local_tol\":1e-10,\"solver\":\"nelder_mead\"}";

sc::Result run(const std::string& name, const sc::Problem& P, sc::Knobs k, long long budget, const sc::Start& s,
               const char* inner = kNM, unsigned seed = 3)
{
    k.push_back({"inner", word(inner)});
    return sc::run(name, P, 1e-6, false, 0.0, 10, budget, seed, k, {}, 0, s);
}

bool same(const sc::Result& a, const sc::Result& b)
{
    return a.objectives == b.objectives && a.variables == b.variables && a.evaluations == b.evaluations;
}

}   // namespace

int main()
{
    sc::Start from;
    from.has_x0 = true;
    from.x0 = {1.1, -2.2, 3.3, 0.4};

    // ── the degenerate case ─────────────────────────────────────────────────
    {
        const sc::Problem P = rastrigin(4);
        const sc::Result g = run("cluster_ga", P, {{"cga_pop", number(1)}, {"cga_de", number(0)}}, 6000, from);
        const sc::Result h = run("pbh", P, {{"pbh_K", number(1)}}, 6000, from);
        check(same(g, h), "cluster_ga with one member and cga_de = 0 = pbh with K = 1, bit for bit");
        check(setting(g, "crossings") == "0" && setting(g, "mutations") != "0", "one member: mutants only (CGA-2)");
    }

    // ── the cut and splice (CGA-1) ──────────────────────────────────────────
    {
        sc::Problem P = lj(6);
        Vec a(18), b(18);
        for (int i = 0; i < 18; ++i) {
            a[static_cast<std::size_t>(i)] = 0.1 * i - 0.8;
            b[static_cast<std::size_t>(i)] = 0.9 - 0.07 * i;
        }
        std::mt19937 rng(11);
        const Vec c = sc::cga::cut_and_splice(P, P.sym_blocks[0], a, b, rng);
        int from_a = 0, inside = 1;
        for (int s = 0; s < 6; ++s) {
            bool in_a = false;
            for (int i = 0; i < 6; ++i)
                in_a = in_a || (c[static_cast<std::size_t>(3 * s)] == a[static_cast<std::size_t>(3 * i)] &&
                                c[static_cast<std::size_t>(3 * s + 1)] == a[static_cast<std::size_t>(3 * i + 1)] &&
                                c[static_cast<std::size_t>(3 * s + 2)] == a[static_cast<std::size_t>(3 * i + 2)]);
            from_a += in_a ? 1 : 0;
        }
        for (std::size_t j = 0; j < c.size(); ++j) inside = inside && c[j] >= P.lo[j] && c[j] <= P.hi[j];
        check(c.size() == 18 && from_a >= 1 && from_a <= 5 && inside,
              "cut and splice: " + std::to_string(from_a) + " of 6 points from the first parent, the rest from the second, in the box");
    }

    // ── refusals, the crossover hook, determinism ───────────────────────────
    {
        const sc::Problem R = rastrigin(4);
        const sc::Result uni = run("cluster_ga", R, {{"cga_pop", number(4)}}, 4000, from);
        check(setting(uni, "crossings") != "0", "cluster_ga: without point blocks the built-in crossover is uniform (CGA-1)");
        bool threw = false;
        try { run("cluster_ga", R, {{"crossover", word("user")}}, 1000, from); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "cluster_ga: crossover = user without the hook refused");
        sc::Problem U = rastrigin(4);
        long long calls = 0;
        U.crossover = [&](const Vec& p1, const Vec& p2, Vec& c1, Vec& c2, std::mt19937& rng) {
            ++calls;
            for (std::size_t j = 0; j < p1.size(); ++j) {
                const bool first = mootation::rnd::uniform01(rng) < 0.5;
                c1[j] = first ? p1[j] : p2[j];
                c2[j] = first ? p2[j] : p1[j];
            }
        };
        const sc::Knobs k{{"cga_pop", number(4)}, {"crossover", word("user")}};
        const sc::Result a = run("cluster_ga", U, k, 8000, from);
        const long long first_calls = calls;
        const sc::Result b = run("cluster_ga", U, k, 8000, from);
        check(first_calls > 0 && setting(a, "crossings") == std::to_string(first_calls),
              "cluster_ga: the crossover hook makes the children");
        check(same(a, b), "cluster_ga: the same run twice");
    }

    // ── A2 multi_dynamic, A3 afpo (cga_replace; MD-1, MD-2, AF-1, AF-2) ─────
    {
        const sc::Problem R = rastrigin(4);
        const sc::Knobs md{{"cga_pop", number(6)}, {"cga_replace", word("multi_dynamic")}};
        const sc::Result a = run("cluster_ga", R, md, 8000, from), b = run("cluster_ga", R, md, 8000, from);
        check(same(a, b) && setting(a, "md_d_initial") == "1", "multi_dynamic: D_I = 0.5 sqrt(4) = 1 (MD-1); the same run twice");
        check(!a.objectives.empty() && a.objectives[0][0] < 10.0, "multi_dynamic: the answer is a low minimum");
        const sc::Knobs af{{"cga_pop", number(6)}, {"cga_replace", word("afpo")}};
        const sc::Result c = run("cluster_ga", R, af, 8000, from), d = run("cluster_ga", R, af, 8000, from);
        const std::string ages = setting(c, "ages");
        int count = ages.empty() ? 0 : 1, young = 0;
        for (char ch : ages) count += ch == ',' ? 1 : 0;
        std::size_t at = 0;
        while (at < ages.size()) {
            const std::size_t comma = ages.find(',', at);
            const long long v = std::stoll(ages.substr(at, comma == std::string::npos ? std::string::npos : comma - at));
            young += v < 2 ? 1 : 0;
            if (comma == std::string::npos) break;
            at = comma + 1;
        }
        check(same(c, d) && count == 6 && young == 0,
              "afpo: the target population of 6, every age >= 2 after the generation (AF-1): " + ages);
        bool threw = false;
        try { run("cluster_ga", R, {{"cga_replace", word("tournament")}}, 1000, from); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "cluster_ga: an unknown cga_replace refused");
    }

    // ── LJ7 (short) ─────────────────────────────────────────────────────────
    {
        const sc::Problem P = lj(7);
        const char* lb = "{\"solver\":\"lbfgsb\"}";
        const sc::Result r = run("cluster_ga", P, {{"cga_pop", number(4)}, {"cga_de", number(5e-4)}}, 60000, {}, lb);
        const double e = r.objectives.empty() ? 0.0 : r.objectives[0][0];
        check(e < -16.5053, "cluster_ga on LJ7: E = " + std::to_string(e) + " (global -16.505384)");
    }

    std::printf("%s\n", g_failed ? "FAILED" : "all passed");
    return g_failed ? 1 : 0;
}
