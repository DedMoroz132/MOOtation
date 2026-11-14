// SPDX-License-Identifier: Apache-2.0
// Continuous symmetries (task 28, §28.5; continuous_symmetry.hpp, CS;
// hooks.hpp distance_of):
//   * a scaling gauge (p, q, r) -> (p e^g, q e^-g, r): x and group_act(x, g)
//     for random g at distance 0 (to lm's accuracy); a point off the orbit
//     not;
//   * three points in the plane, interchangeable and turned together (blocks
//     of 2 and the rotation): a permutation and a turn of x at distance 0
//     (permutations first, then the group; CS-2);
//   * no group: distance_of as before, bit for bit;
//   * tangent_check: the scaling's tangent (p, -q, 0) is the gauge basis'
//     span (sine 0); a basis that is not, a sine far from 0;
//   * refused: group_dim without group_act, group_act of the wrong length
//     (CS-4).

#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/random.hpp>
#include <mootation/scalar/hooks.hpp>

namespace sc = mootation::scalar;
using Vec = std::vector<double>;

static int failures = 0;

static void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

static sc::Problem scaling()
{
    sc::Problem P;
    P.m = 1;
    P.lo = {0.1, 0.1, -5.0};
    P.hi = {10.0, 10.0, 5.0};
    P.group_dim = 1;
    P.group_act = [](const Vec& x, const Vec& g, Vec& out) {
        out = {x[0] * std::exp(g[0]), x[1] * std::exp(-g[0]), x[2]};
    };
    P.gauge_basis = [](const Vec& x, sc::Rows& N) { N = {{x[0]}, {-x[1]}, {0.0}}; };
    return P;
}

static Vec turn(const Vec& x, double t)
{
    Vec out(x.size());
    for (std::size_t i = 0; i + 1 < x.size(); i += 2) {
        out[i] = std::cos(t) * x[i] - std::sin(t) * x[i + 1];
        out[i + 1] = std::sin(t) * x[i] + std::cos(t) * x[i + 1];
    }
    return out;
}

static void scaling_orbit()
{
    const sc::Problem P = scaling();
    std::mt19937 g(9);
    const Vec x{2.0, 0.5, 1.5};
    double worst = 0.0;
    for (int k = 0; k < 8; ++k) {
        const double t = -1.0 + 2.0 * mootation::rnd::uniform01(g);
        Vec y;
        P.group_act(x, {t}, y);
        worst = std::max(worst, sc::distance_of(P, x, y));
    }
    const double off = sc::distance_of(P, x, {2.0, 0.6, 1.5});
    char line[200];
    std::snprintf(line, sizeof line, "scaling gauge: x and group_act(x, g) at distance %.1e (8 random g); off the orbit %.3f",
                  worst, off);
    check(worst < 1e-7 && off > 1e-3, line);
}

static void points_in_the_plane()
{
    sc::Problem P;
    P.m = 1;
    P.lo.assign(6, -5.0);
    P.hi.assign(6, 5.0);
    P.sym_blocks = {{0, 2, 3}};
    P.group_dim = 1;
    P.group_act = [](const Vec& x, const Vec& g, Vec& out) { out = turn(x, g[0]); };
    const Vec x{1.0, 0.0, 0.0, 2.0, -1.5, -0.5};
    const Vec perm{x[4], x[5], x[0], x[1], x[2], x[3]};
    const Vec y = turn(perm, 0.3);
    const sc::ContinuousAlignment a = sc::align_continuous(P, x, y);
    const double d = sc::distance_of(P, x, y);
    char line[200];
    std::snprintf(line, sizeof line,
                  "three points, permuted and turned by 0.3: distance %.1e, the turn found %.6f, %d passes", d,
                  a.g.empty() ? 0.0 : a.g[0], a.passes);
    check(d < 1e-7 && std::fabs(a.g[0] - 0.3) < 1e-7, line);
}

static void no_group()
{
    sc::Problem P;
    P.m = 1;
    P.lo.assign(6, -5.0);
    P.hi.assign(6, 5.0);
    P.sym_blocks = {{0, 2, 3}};
    const Vec x{1.0, 0.0, 0.0, 2.0, -1.5, -0.5}, y{0.1, 2.0, 1.0, 0.2, -1.0, -1.0};
    check(sc::distance_of(P, x, y) == mootation::scalar::symmetry::distance(x, y, P.sym_blocks),
          "no group: distance_of as symmetry::distance, bit for bit");
}

static void tangents()
{
    sc::Problem P = scaling();
    const Vec x{2.0, 0.5, 1.5};
    const sc::TangentCheck t = sc::tangent_check(P, x);
    P.gauge_basis = [](const Vec&, sc::Rows& N) { N = {{1.0}, {0.0}, {0.0}}; };
    const sc::TangentCheck u = sc::tangent_check(P, x);
    char line[200];
    std::snprintf(line, sizeof line, "tangent_check: the gauge basis' span, sine %.1e (ranks %zu, %zu); another, %.3f",
                  t.max_sine, t.tangent_rank, t.gauge_rank, u.max_sine);
    check(t.max_sine < 1e-8 && t.tangent_rank == 1 && t.gauge_rank == 1 && u.max_sine > 0.1, line);
}

static void refusals()
{
    auto refused = [](const sc::Problem& P) {
        try {
            sc::distance_of(P, {1.0, 1.0, 1.0}, {1.0, 1.0, 1.0});
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    sc::Problem a = scaling();
    a.group_act = nullptr;
    sc::Problem b = scaling();
    b.group_act = [](const Vec&, const Vec&, Vec& out) { out = {1.0}; };
    check(refused(a) && refused(b), "refused: group_dim without group_act, group_act of the wrong length (CS-4)");
}

int main()
{
    scaling_orbit();
    points_in_the_plane();
    no_group();
    tangents();
    refusals();
    std::printf(failures ? "FAILED\n" : "all passed\n");
    return failures ? 1 : 0;
}
