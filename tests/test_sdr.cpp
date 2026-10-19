// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// sdr.hpp and NSGA-II/SDR (algorithms/nsga2_sdr.hpp):
//   - the niche size (15) on the two populations of the paper's Fig. 4, from
//     their minimum angles and from points with those angles, and the
//     fallbacks SDR-3;
//   - Property 3 on the paper's own points (the relation as given, without
//     normalisation, θ̄ = π/4);
//   - on random sets (2 to 15 objectives, up to 550 points, copies, points on
//     one ray, a constant objective): irreflexive, antisymmetric, x ≺ y only
//     with Con(x) < Con(y) (so no cycle), and layers as Deb's sort defines
//     them — every point in one, none dominated within its layer, each one
//     after the first dominated from the layer before;
//   - a point at z^min (SDR-2) and copies of a point;
//   - the core: its ranks after setup are sdr::layers of the population, θ̄
//     and the first layer's share the sort's; under constraints the feasible
//     points come first, sorted by SDR among themselves alone, the infeasible
//     ones after them by violation.
// The comparison with a reference written from the paper, bit for bit, is in
// python/test_run_layer.py (sdr_layers_match_the_reference).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "harness.hpp"

namespace mootation::testing {

struct CSpec {          // DTLZ2 at three objectives, feasible where x0 <= 0.5
    static constexpr int NVARS = 7;
    static constexpr int NOBJS = 3;
    static void eval(const std::vector<double>& x, std::vector<double>& f, std::vector<double>& lim)
    {
        double g = 0.0;
        for (int i = NOBJS - 1; i < NVARS; ++i) g += (x[i] - 0.5) * (x[i] - 0.5);
        const double h = std::acos(-1.0) / 2.0;
        f[0] = (1.0 + g) * std::cos(x[0] * h) * std::cos(x[1] * h);
        f[1] = (1.0 + g) * std::cos(x[0] * h) * std::sin(x[1] * h);
        f[2] = (1.0 + g) * std::sin(x[0] * h);
        lim[0] = x[0] - 0.5;
    }
};

} // namespace mootation::testing

MOOTATION_TEST_PROBLEM(SdrTag, NSGAII_Individual, DTLZ2Spec)

namespace mootation::testing {
struct SdrCTag : public mootation::NSGAII_Individual {};
}
namespace mootation {
template <>
class Problem<testing::SdrCTag> {
public:
    std::vector<std::pair<std::optional<double>, std::optional<double>>> bounds =
        std::vector<std::pair<std::optional<double>, std::optional<double>>>(
            testing::CSpec::NVARS, {0.0, 1.0});
    int get_vars_n() const { return testing::CSpec::NVARS; }
    int get_bin_vars_n() const { return 0; }
    int get_objs_n() const { return testing::CSpec::NOBJS; }
    int get_lims_n() const { return 1; }
    void calc_objs(testing::SdrCTag& ind) const
    {
        testing::CSpec::eval(ind.variables, ind.objectives, ind.limits);
    }
};
} // namespace mootation

using namespace mootation;
using namespace mootation::testing;

namespace {

void fig4()
{
    // the minimum angles of the two populations, as the paper lists them
    check(sdr::niche_size({0.1, 0.1, 0.2, 0.2, 0.3, 0.3}, 6) == 0.3, "Fig. 4 left: θ̄ = 0.3");
    check(sdr::niche_size({0.1, 0.1, 0.2, 0.3, 0.4, 0.5}, 6) == 0.3, "Fig. 4 right: θ̄ = 0.3");
    // SDR-3: fewer distinct values than ⌊n/2⌋, θ̄ = 0, none positive
    check(sdr::niche_size({0.2, 0.2, 0.1, 0.1, 0.1, 0.1}, 6) == 0.2, "fewer distinct: the largest");
    check(sdr::niche_size({0.0, 0.0, 0.4}, 3) == 0.4, "θ̄ = 0: the smallest positive");
    check(sdr::niche_size({0.0, 0.0, 0.0, 0.0}, 4) == 0.0, "none positive: 0");

    // points on the unit circle with those angles between neighbours
    auto at = [](const std::vector<double>& gaps) {
        std::vector<std::vector<double>> P;
        double a = 0.05;
        P.push_back({std::cos(a), std::sin(a)});
        for (double g : gaps) {
            a += g;
            P.push_back({std::cos(a), std::sin(a)});
        }
        return P;
    };
    // left: a-b 0.1, c-d 0.2, e-f 0.3, the pairs 0.35 apart
    check_close(sdr::frame_of(at({0.1, 0.35, 0.2, 0.35, 0.3})).theta_bar, 0.3, 1e-12,
                "Fig. 4 left from points");
    check_close(sdr::frame_of(at({0.1, 0.2, 0.3, 0.4, 0.5})).theta_bar, 0.3, 1e-12,
                "Fig. 4 right from points");
}

void property3()
{
    sdr::Frame fr = sdr::frame_of({{0.0, 3.0}, {2.0, 2.0}, {5.0, 0.0}});
    fr.theta_bar = std::acos(-1.0) / 4.0;
    check(sdr::dominates(fr, 0, 1), "Property 3: x ≺ y");
    check(sdr::dominates(fr, 1, 2), "Property 3: y ≺ z");
    check(!sdr::dominates(fr, 0, 2), "Property 3: not x ≺ z");
    check(!sdr::dominates(fr, 1, 0) && !sdr::dominates(fr, 2, 1) && !sdr::dominates(fr, 2, 0),
          "Property 3: no reverse");
}

// A random set: a front-like cloud, with copies, a ray and a constant
// objective now and then.
std::vector<std::vector<double>> random_set(std::mt19937& rng, int n, int m)
{
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<std::vector<double>> F(static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(m)));
    const int kind = static_cast<int>(rng() % 4);
    for (auto& f : F) {
        double s = 0.0;
        for (auto& v : f) { v = u(rng); s += v; }
        if (kind == 1) for (auto& v : f) v = v / s * (1.0 + 0.3 * u(rng));      // near a simplex
        if (kind == 2) for (auto& v : f) v = std::round(v * 4.0) / 4.0;          // a grid: ties
    }
    if (kind == 3 && n > 2) {
        F[1] = F[0];                                                             // a copy
        for (std::size_t k = 0; k < F[2].size(); ++k) F[2][k] = 2.0 * F[0][k];   // a ray
    }
    if (rng() % 5 == 0)
        for (auto& f : F) f[0] = 1.5;                                            // zero range
    return F;
}

void properties()
{
    std::mt19937 rng(20261019u);
    int sets = 0;
    bool irreflexive = true, antisymmetric = true, con = true, covered = true, inside = true, chained = true;
    for (int t = 0; t < 400; ++t) {
        const int m = 2 + static_cast<int>(rng() % 14);
        const int n = 1 + static_cast<int>(rng() % 550);
        const auto F = random_set(rng, n, m);
        const sdr::Frame fr = sdr::make_frame(F);
        for (int i = 0; i < n; ++i) {
            irreflexive = irreflexive && !sdr::dominates(fr, i, i);
            for (int j = 0; j < n; ++j) {
                if (!sdr::dominates(fr, i, j)) continue;
                antisymmetric = antisymmetric && !sdr::dominates(fr, j, i);
                con = con && fr.con[i] < fr.con[j];
            }
        }
        const auto fronts = sdr::fast_sort(n, [&](int i, int j) { return sdr::dominates(fr, i, j); });
        std::size_t total = 0;
        for (std::size_t k = 0; k < fronts.size(); ++k) {
            total += fronts[k].size();
            for (int a : fronts[k]) {
                for (int b : fronts[k]) inside = inside && !sdr::dominates(fr, a, b);
                if (k == 0) continue;
                bool any = false;
                for (int b : fronts[k - 1]) any = any || sdr::dominates(fr, b, a);
                chained = chained && any;
            }
        }
        covered = covered && total == static_cast<std::size_t>(n);
        ++sets;
    }
    check(irreflexive, "irreflexive on " + std::to_string(sets) + " random sets");
    check(antisymmetric, "antisymmetric");
    check(con, "x ≺ y only with Con(x) < Con(y): acyclic");
    check(covered, "every point in one layer");
    check(inside, "no point dominates another of its layer");
    check(chained, "every point after the first layer dominated from the layer before");
}

void degenerate()
{
    // a point at z^min: every angle with it 0, θ̄ = 0, Con alone
    const std::vector<std::vector<double>> F = {{0.0, 0.0, 0.0}, {1.0, 0.2, 0.5}, {0.2, 1.0, 0.4},
                                                {0.5, 0.5, 1.0}, {0.9, 0.9, 0.1}};
    double tb = -1.0;
    const auto rank = sdr::layers(F, &tb);
    check(tb == 0.0, "a point at z^min: θ̄ = 0");
    // Con: 0, 1.7/1.0 ... by Con alone the layers follow the sums of f̄
    const sdr::Frame fr = sdr::make_frame(F);
    bool by_con = true;
    for (std::size_t i = 0; i < F.size(); ++i)
        for (std::size_t j = 0; j < F.size(); ++j)
            by_con = by_con && (sdr::dominates(fr, i, j) == (fr.con[i] < fr.con[j]));
    check(by_con, "a point at z^min: the relation is Con alone");
    check(rank[0] == 0, "the point at z^min is in the first layer");

    // copies never dominate each other
    const std::vector<std::vector<double>> C = {{0.3, 0.7}, {0.3, 0.7}, {0.7, 0.3}, {1.0, 1.0}};
    const sdr::Frame fc = sdr::make_frame(C);
    check(!sdr::dominates(fc, 0, 1) && !sdr::dominates(fc, 1, 0), "copies: neither dominates");
    check(sdr::layers(C)[0] == sdr::layers(C)[1], "copies: one layer");

    // a single point, and none
    check(sdr::layers({{1.0, 2.0}}) == std::vector<int>{0}, "one point: layer 0");
    check(sdr::layers({}).empty(), "no points");
}

template <typename Ind>
std::vector<std::vector<double>> objectives(DataVault<Ind>& v, int n)
{
    std::vector<std::vector<double>> F;
    for (int i = 0; i < n; ++i) F.push_back(v.objectives_of(i));
    return F;
}

void core()
{
    {
        Problem<SdrTag> prob;
        Optimizer<SdrTag, NSGAIISDRCore<SdrTag>> opt(DataVault<SdrTag>(100, prob), defer_setup);
        auto& alg = opt.get_algorithm();
        alg.set_seed(7);
        opt.setup();
        auto& v = opt.get_vault();
        double tb = -1.0;
        const auto rank = sdr::layers(objectives(v, 100), &tb);
        bool same = true;
        int first = 0;
        for (int i = 0; i < 100; ++i) {
            same = same && v.get_ind(i).rank == rank[static_cast<std::size_t>(i)];
            first += rank[static_cast<std::size_t>(i)] == 0;
        }
        check(same, "setup: the ranks are sdr::layers of P");
        check(alg.theta_bar() == tb, "setup: θ̄ of P");
        check(alg.first_layer_share() == first / 100.0, "setup: the first layer's share");
        opt.optimize(30);
        check(v.active_n() == 100, "30 generations: the population is N");
        check(alg.first_layer_share() > 0.0 && alg.first_layer_share() <= 1.0, "share in (0, 1]");
    }
    for (ConstraintMode mode : {ConstraintMode::FEASIBILITY, ConstraintMode::CDP}) {
        Problem<SdrCTag> prob;
        Optimizer<SdrCTag, NSGAIISDRCore<SdrCTag>> opt(DataVault<SdrCTag>(120, prob), defer_setup);
        auto& alg = opt.get_algorithm();
        alg.set_seed(3);
        alg.constraint_mode = mode;
        opt.setup();
        auto& v = opt.get_vault();
        std::vector<int> feas, infeas;
        for (int i = 0; i < 120; ++i) (v.get_cv(i) <= 0.0 ? feas : infeas).push_back(i);
        std::vector<std::vector<double>> F;
        for (int i : feas) F.push_back(v.objectives_of(i));
        double tb = -1.0;
        const auto rank = sdr::layers(F, &tb);
        int layers = 0;
        bool same = true;
        for (std::size_t k = 0; k < feas.size(); ++k) {
            same = same && v.get_ind(feas[k]).rank == rank[k];
            layers = std::max(layers, rank[k] + 1);
        }
        check(!feas.empty() && !infeas.empty(), "constrained: both kinds in P");
        check(same, "constrained: the feasible ranks are sdr::layers of the feasible part");
        check(alg.theta_bar() == tb, "constrained: θ̄ of the feasible part");
        bool after = true;
        for (int i : infeas) {
            int below = 0;                         // distinct smaller violations
            std::vector<double> seen;
            for (int j : infeas)
                if (v.get_cv(j) < v.get_cv(i) &&
                    std::find(seen.begin(), seen.end(), v.get_cv(j)) == seen.end()) {
                    seen.push_back(v.get_cv(j));
                    ++below;
                }
            after = after && v.get_ind(i).rank == layers + below;
        }
        check(after, "constrained: the infeasible after the feasible, layered by violation");
        opt.optimize(20);
        int feasible = 0;
        for (int i = 0; i < 120; ++i) feasible += v.get_cv(i) <= 0.0;
        check(feasible == 120, "constrained: after 20 generations every member feasible");
    }
}

} // namespace

int main()
{
    fig4();
    property3();
    properties();
    degenerate();
    core();
    return report("test_sdr");
}
