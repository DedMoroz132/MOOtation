// SPDX-License-Identifier: Apache-2.0
// The problem's hooks of task 24, §24.1.3, §24.1.4, §24.1.6 (evaluator.hpp,
// hooks.hpp):
//   * distance_of: Euclidean without symmetries; invariant to the blocks'
//     permutations, to canon and, through the problem's align, to D4 of
//     points in a square; periodic coordinates the short way;
//   * canon_key_of: one key for a structure's permutations;
//   * the cache: a synthetic run of 60 points, 20 structures each met three
//     times permuted, evaluates 20 and hits 40 (with an archive and in the
//     lean mode); exact counts the archive's repeats; off is the run as
//     before; multistart with the cache, one thread = three;
//   * restricted: all the variables = the problem, bit for bit; a part's f
//     the whole's at the assembled point, its gradient the whole's columns;
//   * delta: a pair energy's incremental delta, f(x) + delta = f(x') within
//     ulp times the pairs; the hook's declared cost charged, no evaluation;
//     without it the two points evaluated.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/random.hpp>
#include <mootation/scalar/hooks.hpp>
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

// k points in [0, 1]^2, blocks of 2 coordinates, a pair energy (symmetric
// under the points' permutations and under D4 about the square's centre)
static sc::Problem points(std::size_t k, std::shared_ptr<long long> calls)
{
    sc::Problem P;
    P.lo.assign(2 * k, 0.0);
    P.hi.assign(2 * k, 1.0);
    P.m = 1;
    P.sym_blocks = {{0, 2, k}};
    P.evaluate = [k, calls](const Vec& x, Vec& f, Vec&) {
        if (calls) ++*calls;
        double s = 0.0;
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = i + 1; j < k; ++j) {
                const double dx = x[2 * i] - x[2 * j], dy = x[2 * i + 1] - x[2 * j + 1];
                s += 1.0 / (1e-3 + dx * dx + dy * dy);
            }
        f[0] = s;
    };
    return P;
}

// D4 about (1/2, 1/2): the k-th of the 8 maps on every point
static Vec d4(const Vec& x, int t)
{
    Vec y = x;
    for (std::size_t i = 0; i + 1 < x.size(); i += 2) {
        double a = x[i] - 0.5, b = x[i + 1] - 0.5;
        for (int r = 0; r < t % 4; ++r) {
            const double c = a;
            a = -b;
            b = c;
        }
        if (t >= 4) a = -a;
        y[i] = a + 0.5;
        y[i + 1] = b + 0.5;
    }
    return y;
}

int main()
{
    std::mt19937 rng(5);
    const std::size_t k = 4;
    Vec x(2 * k);
    for (double& v : x) v = mootation::rnd::uniform01(rng);
    Vec perm = x;                                                     // the points in another order
    std::swap(perm[0], perm[4]);
    std::swap(perm[1], perm[5]);
    std::swap(perm[2], perm[6]);
    std::swap(perm[3], perm[7]);

    // distance_of
    {
        sc::Problem plain;
        plain.lo.assign(2 * k, 0.0);
        plain.hi.assign(2 * k, 1.0);
        double e = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) e += (x[i] - perm[i]) * (x[i] - perm[i]);
        check(sc::distance_of(plain, x, perm) == std::sqrt(e), "no symmetries: the Euclidean distance");
        sc::Problem P = points(k, nullptr);
        check(sc::distance_of(P, perm, x) == 0.0, "the blocks: a permutation at distance 0");
        Vec y(2 * k);
        for (double& v : y) v = mootation::rnd::uniform01(rng);
        check(std::fabs(sc::distance_of(P, perm, y) - sc::distance_of(P, x, y)) < 1e-15,
              "the blocks: distance invariant to the permutation");
        sc::Problem C = P;
        C.canon = [](Vec& v) { sym::sort_blocks(v, {{0, 2, 4}}); };
        Vec cx = x;
        C.canon(cx);
        check(sc::distance_of(C, cx, x) == 0.0, "canon: canon(x) at distance 0");
        // D4 through the problem's align: the best of the 8 maps, then the blocks
        sc::Problem D = P;
        D.align = [](const Vec& a, const Vec& ref, Vec& out) {
            double best = std::numeric_limits<double>::infinity();
            for (int t = 0; t < 8; ++t) {
                const Vec cand = sym::align(d4(a, t), ref, {{0, 2, 4}});
                double s = 0.0;
                for (std::size_t i = 0; i < cand.size(); ++i) s += (cand[i] - ref[i]) * (cand[i] - ref[i]);
                if (s < best) {
                    best = s;
                    out = cand;
                }
            }
        };
        double worst = 0.0;
        for (int t = 0; t < 8; ++t) worst = std::max(worst, sc::distance_of(D, d4(perm, t), x));
        check(worst < 1e-15, "D4 and the permutations: distance 0 (worst " + std::to_string(worst) + ")");
        sc::Problem Q = plain;
        Q.lo.assign(1, 0.0);
        Q.hi.assign(1, 1.0);
        Q.periodic = {1};
        check(std::fabs(sc::distance_of(Q, {0.98}, {0.01}) - 0.03) < 1e-15, "periodic: the short way");
        check(sc::distance_of(plain, x, perm) == sc::distance_of(plain, perm, x), "symmetric");
        sc::Problem U = plain;
        U.distance = [](const Vec&, const Vec&) { return 7.0; };
        check(sc::distance_of(U, x, perm) == 7.0, "the problem's own distance");
    }

    // canon_key_of
    {
        sc::Problem P = points(k, nullptr);
        check(sc::canon_key_of(P, x) == sc::canon_key_of(P, perm), "one key for a structure's permutations");
        sc::Problem plain = P;
        plain.sym_blocks.clear();
        check(sc::canon_key_of(plain, x) != sc::canon_key_of(plain, perm), "without symmetries: x's bytes");
        sc::Problem U = P;
        U.canon_key = [](const Vec&) { return std::string("one"); };
        check(sc::canon_key_of(U, x) == "one", "the problem's own key");
    }

    // the cache on a synthetic run: 20 structures, each met three times permuted
    {
        Rows pts;
        for (int s = 0; s < 20; ++s) {
            Vec v(2 * k);
            for (double& c : v) c = mootation::rnd::uniform01(rng);
            pts.push_back(v);
            Vec w = v;
            std::swap(w[0], w[2]);
            std::swap(w[1], w[3]);
            pts.push_back(w);
            Vec u = v;
            std::swap(u[4], u[6]);
            std::swap(u[5], u[7]);
            pts.push_back(u);
        }
        for (bool lean : {false, true}) {
            auto calls = std::make_shared<long long>(0);
            sc::Problem P = points(k, calls);
            P.cache = sc::Problem::Cache::canon;
            sc::Evaluator ev(P, 1e-6, false, 0.0, 1000, 20, {}, 0);
            ev.lean = lean;
            bool equal = true;
            for (std::size_t i = 0; i < pts.size(); ++i) {
                const double f = ev.F[ev.index(pts[i])][0];
                Vec ff(1), g;
                points(k, nullptr).evaluate(pts[i], ff, g);
                equal = equal && std::fabs(f - ff[0]) <= 1e-12 * std::fabs(ff[0]);
            }
            check(*calls == 20 && ev.cache_hits == 40 && ev.spent == 20,
                  std::string(lean ? "lean: " : "") + "canon: 20 evaluated, 40 hits (" + std::to_string(*calls) + ", " +
                      std::to_string(ev.cache_hits) + ")");
            check(equal, "canon: a hit has its structure's value");
        }
        auto calls = std::make_shared<long long>(0);
        sc::Problem E = points(k, calls);
        E.cache = sc::Problem::Cache::exact;
        sc::Evaluator ev(E, 1e-6, false, 0.0, 1000, 20, {}, 0);
        for (int r = 0; r < 3; ++r) ev.index(pts[0]);
        check(*calls == 1 && ev.cache_hits == 2, "exact: the archive's repeats counted");
        sc::Problem O = points(k, nullptr), F = points(k, nullptr);
        O.cache = sc::Problem::Cache::off;
        const sc::Result a = sc::run("de", O, 1e-6, false, 0.0, 20, 2000, 3, {}, {}, 0, {});
        const sc::Result b = sc::run("de", F, 1e-6, false, 0.0, 20, 2000, 3, {}, {}, 0, {});
        check(same(a, b) && a.cache_hits == 0, "off: the run as before");
        // multistart with the cache: one thread = three; evaluations the problem's calls
        auto c1 = std::make_shared<long long>(0), c3 = std::make_shared<long long>(0);
        sc::Problem M1 = points(k, c1), M3 = points(k, c3);
        M1.cache = M3.cache = sc::Problem::Cache::canon;
        const sc::Knob nm{true, "{\"solver\":\"nelder_mead\"}", 0.0};
        const sc::Knobs k1{{"inner", nm}, {"n_starts", {false, "", 8.0}}};
        const sc::Knobs k3{{"inner", nm}, {"n_starts", {false, "", 8.0}}, {"threads", {false, "", 3.0}}};
        const sc::Result m1 = sc::run("multistart", M1, 1e-6, false, 0.0, 20, 4000, 3, k1, {}, 0, {});
        const sc::Result m3 = sc::run("multistart", M3, 1e-6, false, 0.0, 20, 4000, 3, k3, {}, 0, {});
        check(same(m1, m3) && m1.cache_hits == m3.cache_hits, "multistart with the cache: one thread = three");
        std::printf("  multistart (8 starts of nelder_mead), 4 points, cache canon: %lld evaluations, %lld hits\n",
                    m1.evaluations, m1.cache_hits);
    }

    // restricted
    {
        sc::Problem P;
        P.lo.assign(4, -5.0);
        P.hi.assign(4, 5.0);
        P.m = 1;
        P.evaluate = [](const Vec& z, Vec& f, Vec&) {
            double s = 0.0;
            for (std::size_t i = 0; i + 1 < z.size(); ++i)
                s += 100.0 * (z[i + 1] - z[i] * z[i]) * (z[i + 1] - z[i] * z[i]) + (1.0 - z[i]) * (1.0 - z[i]);
            f[0] = s;
        };
        P.gradient = [](const Vec& z, Vec& g, Rows&) {
            g.assign(z.size(), 0.0);
            for (std::size_t i = 0; i + 1 < z.size(); ++i) {
                g[i] += -400.0 * z[i] * (z[i + 1] - z[i] * z[i]) - 2.0 * (1.0 - z[i]);
                g[i + 1] += 200.0 * (z[i + 1] - z[i] * z[i]);
            }
        };
        const Vec xf{0.3, -0.2, 0.7, 1.1};
        std::vector<std::size_t> all(4);
        std::iota(all.begin(), all.end(), std::size_t{0});
        sc::Start st;
        st.has_x0 = true;
        st.x0 = xf;
        const sc::Result a = sc::run("lbfgsb", P, 1e-6, false, 0.0, 20, 2000, 3, {}, {}, 0, st);
        const sc::Result b = sc::run("lbfgsb", sc::restricted(P, xf, all), 1e-6, false, 0.0, 20, 2000, 3, {}, {}, 0, st);
        check(same(a, b), "restricted to every variable: the problem, bit for bit");
        const sc::Problem Q = sc::restricted(P, xf, {1, 3});
        Vec fq(1), ff(1), g, gq, gf;
        Rows J;
        Q.evaluate({0.4, 0.9}, fq, g);
        P.evaluate({0.3, 0.4, 0.7, 0.9}, ff, g);
        Q.gradient({0.4, 0.9}, gq, J);
        P.gradient({0.3, 0.4, 0.7, 0.9}, gf, J);
        check(fq[0] == ff[0] && gq == Vec({gf[1], gf[3]}) && Q.lo.size() == 2,
              "a part: the whole's f and gradient columns at the assembled point");
        sc::Start s2;
        s2.has_x0 = true;
        s2.x0 = {-0.2, 1.1};
        const sc::Result c = sc::run("lbfgsb", Q, 1e-6, false, 0.0, 20, 2000, 3, {}, {}, 0, s2);
        check(!c.objectives.empty() && c.objectives[0][0] <= ff[0] + 10.0, "lbfgsb on a part runs");
        sc::Problem U = P;
        U.restrict_to = [](const Vec&, const std::vector<std::size_t>& idx) {
            sc::Problem R;
            R.lo.assign(idx.size(), 0.0);
            R.hi.assign(idx.size(), 1.0);
            return R;
        };
        check(sc::restricted(U, xf, {0}).hi == Vec({1.0}), "the problem's own subproblem");
    }

    // delta: a pair energy, one variable moved
    {
        const std::size_t n = 50;
        auto energy = [](const Vec& z) {
            double s = 0.0;
            for (std::size_t i = 0; i < z.size(); ++i)
                for (std::size_t j = i + 1; j < z.size(); ++j) s += (z[i] - z[j]) * (z[i] - z[j]);
            return s;
        };
        auto calls = std::make_shared<long long>(0);
        sc::Problem P;
        P.lo.assign(n, -1.0);
        P.hi.assign(n, 1.0);
        P.m = 1;
        P.evaluate = [energy, calls](const Vec& z, Vec& f, Vec&) {
            ++*calls;
            f[0] = energy(z);
        };
        P.delta = [](const Vec& z, const std::vector<std::size_t>& block, const Vec& values) {
            const std::size_t q = block.at(0);
            double d = 0.0;
            for (std::size_t j = 0; j < z.size(); ++j)
                if (j != q) d += (values[0] - z[j]) * (values[0] - z[j]) - (z[q] - z[j]) * (z[q] - z[j]);
            return d;
        };
        P.delta_cost = 0.25;
        Vec z(n);
        for (double& v : z) v = 2.0 * mootation::rnd::uniform01(rng) - 1.0;
        sc::Evaluator ev(P, 1e-6, false, 0.0, 1000, 20, {}, 0);
        const double d = ev.delta(z, {7}, {0.123});
        Vec z2 = z;
        z2[7] = 0.123;
        const double fz = energy(z), fz2 = energy(z2);
        const double pairs = static_cast<double>(n * (n - 1) / 2);
        check(std::fabs(fz + d - fz2) <= pairs * std::numeric_limits<double>::epsilon() * fz2,
              "delta: f(x) + delta = f(x') within ulp times the pairs");
        check(ev.delta_calls == 1 && ev.delta_spent == 0.25 && *calls == 0 && ev.used() == 0.25,
              "delta: the hook's cost charged, nothing evaluated");
        sc::Problem N = P;
        N.delta = nullptr;
        *calls = 0;
        sc::Evaluator ev2(N, 1e-6, false, 0.0, 1000, 20, {}, 0);
        const double d2 = ev2.delta(z, {7}, {0.123});
        check(*calls == 2 && ev2.spent == 2 && d2 == fz2 - fz, "delta without the hook: the two points evaluated");
        std::string why;
        try {
            ev.delta(z, {1, 2}, {0.0});
        } catch (const std::invalid_argument& e) {
            why = e.what();
        }
        check(why.find("1 values for a block of 2") != std::string::npos, "delta: a wrong block refused");
    }

    if (failures) {
        std::printf("test_hooks: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_hooks: all checks passed\n");
    return 0;
}
