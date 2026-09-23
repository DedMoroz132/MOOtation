// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// operators/bound_repair.hpp and the operators that use it.
//
//   (1) every repair lands inside the box, with its own arithmetic: clip on
//       the violated bound, reflect as a triangle wave (any overshoot), wrap as
//       a torus, midpoint halfway to the parent, random uniform inside;
//   (2) the historical DE names are the same draws as the new ones:
//       DERepair::RandomReset == random, DERepair::Clip == clip, for
//       de_rand_1_bin and repair_out_of_box, number for number;
//   (3) Liu & Li's operators: the default is the paper's rule, unchanged; any
//       other repair keeps children inside; the crossover refuses resample;
//   (4) the counters: a tally books one offspring and its variables outside,
//       and the operator list records what ran with which repair;
//   (5) the settings file takes bound_repair as a word and refuses an unknown one;
//   (6) the mutations of real_mutation.hpp: inside the box under every repair,
//       nothing moves at p_m = 0, the polynomial choice is polynomial_mutation;
//   (7) the crossovers of real_crossover.hpp: uniform children are complementary,
//       BLX-alpha's inside the box, SBX is ops::sbx;
//   (8) DE/rand/2/bin and DE/best/1/bin are Storn & Price's formulas at CR = 1.
// ============================================================================

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <mootation/operators/bound_repair.hpp>
#include <mootation/operators/de_mutation.hpp>
#include <mootation/operators/liuli_crossover.hpp>
#include <mootation/operators/real_crossover.hpp>
#include <mootation/operators/real_mutation.hpp>
#include <mootation/settings.hpp>

namespace {

int g_checks = 0, g_failed = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failed;
        std::printf("  FAIL  %s\n", what.c_str());
    }
}

using Bounds = std::vector<std::pair<std::optional<double>, std::optional<double>>>;

}  // namespace

int main() {
    using namespace mootation::ops;
    std::mt19937 rng(20260923u);

    // ── (1) the arithmetic ──────────────────────────────────────────────────
    check(repair_value(1.3, 0.0, 1.0, 0.5, BoundRepair::Clip, rng) == 1.0, "clip high");
    check(repair_value(-0.2, 0.0, 1.0, 0.5, BoundRepair::Clip, rng) == 0.0, "clip low");
    check(std::abs(repair_value(1.3, 0.0, 1.0, 0.5, BoundRepair::Reflect, rng) - 0.7) < 1e-12,
          "reflect once across the upper bound");
    check(std::abs(repair_value(-0.2, 0.0, 1.0, 0.5, BoundRepair::Reflect, rng) - 0.2) < 1e-12,
          "reflect once across the lower bound");
    check(std::abs(repair_value(2.3, 0.0, 1.0, 0.5, BoundRepair::Reflect, rng) - 0.3) < 1e-12,
          "reflect again while outside: 2.3 -> -0.3 -> 0.3");
    check(std::abs(repair_value(1.3, 0.0, 1.0, 0.5, BoundRepair::Wrap, rng) - 0.3) < 1e-12,
          "wrap: 1.3 -> 0.3");
    check(std::abs(repair_value(-0.2, 0.0, 1.0, 0.5, BoundRepair::Wrap, rng) - 0.8) < 1e-12,
          "wrap: -0.2 -> 0.8");
    check(std::abs(repair_value(1.3, 0.0, 1.0, 0.4, BoundRepair::Midpoint, rng) - 0.7) < 1e-12,
          "midpoint between the parent 0.4 and the upper bound");
    check(std::abs(repair_value(-5.0, 2.0, 4.0, 3.0, BoundRepair::Midpoint, rng) - 2.5) < 1e-12,
          "midpoint on a box that does not start at zero");
    bool inside = true;
    for (int i = 0; i < 1000; ++i) {
        for (BoundRepair b : {BoundRepair::Clip, BoundRepair::Reflect, BoundRepair::Random,
                              BoundRepair::Midpoint, BoundRepair::Wrap, BoundRepair::Resample,
                              BoundRepair::Native}) {
            const double v = std::uniform_real_distribution<double>(-10.0, 10.0)(rng);
            if (v >= -1.0 && v <= 2.0) continue;
            const double r = repair_value(v, -1.0, 2.0, 0.3, b, rng);
            inside = inside && r >= -1.0 && r <= 2.0;
        }
    }
    check(inside, "every repair lands inside [-1, 2] from anywhere in [-10, 10]");
    for (BoundRepair b : {BoundRepair::Clip, BoundRepair::Reflect, BoundRepair::Random,
                          BoundRepair::Midpoint, BoundRepair::Resample, BoundRepair::Wrap,
                          BoundRepair::Native}) {
        auto back = parse_bound_repair(bound_repair_name(b));
        check(back && *back == b, std::string("name round trip: ") + bound_repair_name(b));
    }
    check(!parse_bound_repair("clamp"), "an unknown name is refused");

    // ── (2) the historical DE names draw the same numbers ───────────────────
    {
        const Bounds bd(8, {0.0, 1.0});
        std::vector<double> a(8, 0.9), b(8, 0.95), c(8, 0.05), x(8, 0.5), y1, y2;
        for (DERepair old : {DERepair::RandomReset, DERepair::Clip}) {
            std::mt19937 r1(7u), r2(7u);
            de_rand_1_bin(a, b, c, x, y1, bd, 0.5, 0.9, old, r1);
            de_rand_1_bin(a, b, c, x, y2, bd, 0.5, 0.9, to_bound_repair(old), r2);
            check(y1 == y2 && r1() == r2(), "de_rand_1_bin: DERepair and BoundRepair agree");
            std::vector<double> z1(8, 1.4), z2(8, 1.4);
            std::mt19937 s1(9u), s2(9u);
            repair_out_of_box(z1, bd, old, s1);
            repair_out_of_box(z2, bd, to_bound_repair(old), s2);
            check(z1 == z2 && s1() == s2(), "repair_out_of_box: DERepair and BoundRepair agree");
        }
        bool threw = false;
        try { de_rand_1_bin(a, b, c, x, y1, bd, 0.5, 0.9, BoundRepair::Resample, rng); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "DE refuses resample: it cannot redraw one variable");
    }

    // ── (3) Liu & Li ─────────────────────────────────────────────────────────
    {
        const Bounds bd(6, {0.0, 1.0});
        std::vector<double> x = {0.99, 0.01, 0.5, 0.97, 0.02, 0.6};
        std::vector<double> y = {0.01, 0.99, 0.4, 0.03, 0.98, 0.5};
        std::vector<double> c1, c2;
        std::mt19937 r1(11u), r2(11u);
        liuli_crossover(x, y, c1, bd, 1, 100, r1);
        liuli_crossover(x, y, c2, bd, 1, 100, r2, BoundRepair::Native);
        check(c1 == c2 && r1() == r2(), "liuli_crossover: the default is the paper's rule");
        bool ok = true;
        for (BoundRepair b : {BoundRepair::Clip, BoundRepair::Reflect, BoundRepair::Random,
                              BoundRepair::Midpoint, BoundRepair::Wrap}) {
            for (int t = 0; t < 200; ++t) {
                liuli_crossover(x, y, c1, bd, 1, 100, rng, b);
                std::vector<double> m = c1;
                liuli_mutation(m, bd, 1.0, 1, 100, rng, b);
                for (double v : c1) ok = ok && v >= 0.0 && v <= 1.0;
                for (double v : m) ok = ok && v >= 0.0 && v <= 1.0;
            }
        }
        std::vector<double> m = x;
        for (int t = 0; t < 200; ++t) {
            liuli_mutation(m, bd, 1.0, 1, 100, rng, BoundRepair::Resample);
            for (double v : m) ok = ok && v >= 0.0 && v <= 1.0;
        }
        check(ok, "Liu-Li children stay inside under every repair");
        bool threw = false;
        try { liuli_crossover(x, y, c1, bd, 1, 100, rng, BoundRepair::Resample); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "liuli_crossover refuses resample: one rc per offspring");
    }

    // ── (4) the counters ─────────────────────────────────────────────────────
    {
        reset_operator_records();
        { RepairTally t; t.out(); t.out(); }
        { RepairTally t; }
        const auto& c = repair_counts();
        check(c.checked == 2 && c.out_children == 1 && c.out_vars == 2,
              "two offspring booked, one with two variables outside");
        const Bounds bd(4, {0.0, 1.0});
        std::vector<double> z = {1.5, 0.5, -0.5, 0.5};
        repair_out_of_box(z, bd, BoundRepair::Reflect, rng);
        check(repair_counts().out_vars == 4 && repair_counts().out_children == 2,
              "repair_out_of_box books its two variables");
        const auto& used = operators_used();
        check(used.size() == 1 && std::string(used[0].op) == "box repair" &&
              std::string(used[0].repair) == "reflect",
              "the operator list names the site and its repair");
        reset_operator_records();
        check(operators_used().empty() && repair_counts().checked == 0, "a reset clears both");
    }

    // ── (5) the settings file ────────────────────────────────────────────────
    {
        auto s = mootation::Settings::from_string(
            "n_vars = 2\nlower = 0\nupper = 1\nbound_repair = reflect\n");
        check(s.text_params.count("bound_repair") && s.text_params["bound_repair"] == "reflect",
              "bound_repair is read as a word");
        auto back = mootation::Settings::from_string(s.to_string());
        check(back.text_params == s.text_params, "and survives the settings text");
        bool threw = false;
        try {
            mootation::Settings::from_string("n_vars = 2\nlower = 0\nupper = 1\nbound_repair = clamp\n");
        } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "an unknown repair is refused when the file is read");
    }

    // ── (6) the mutations ────────────────────────────────────────────────────
    {
        const Bounds bd(10, {-2.0, 3.0});
        bool ok = true, still = true;
        for (Mutation m : {Mutation::Polynomial, Mutation::Gaussian, Mutation::Cauchy,
                           Mutation::UniformReset, Mutation::Mixture, Mutation::MixtureCauchy}) {
            for (BoundRepair b : {BoundRepair::Clip, BoundRepair::Reflect, BoundRepair::Random,
                                  BoundRepair::Midpoint, BoundRepair::Resample, BoundRepair::Wrap}) {
                MutationSpec spec;
                spec.kind = m;
                spec.repair = b;
                spec.scale = 0.5;               // large: plenty of steps leave the box
                for (int t = 0; t < 100; ++t) {
                    std::vector<double> x(10);
                    for (double& v : x) v = std::uniform_real_distribution<double>(-2.0, 3.0)(rng);
                    spec.apply(x, bd, 20.0, 1.0, rng);
                    for (double v : x) ok = ok && v >= -2.0 && v <= 3.0;
                    std::vector<double> y = x;
                    spec.apply(y, bd, 20.0, 0.0, rng);   // p_m = 0: nothing moves
                    still = still && y == x;
                }
            }
            check(parse_mutation(mutation_name(m)) == m, std::string("name: ") + mutation_name(m));
        }
        check(ok, "every mutation stays inside the box under every repair");
        check(still, "at p_m = 0 no mutation changes anything");
        MutationSpec poly;
        std::vector<double> a(10, 0.3), b2(10, 0.3);
        std::mt19937 r1(5u), r2(5u);
        poly.apply(a, bd, 20.0, 0.5, r1);
        polynomial_mutation(b2, bd, 20.0, 0.5, r2);
        check(a == b2 && r1() == r2(), "the polynomial choice is polynomial_mutation, draw for draw");
        MutationSpec raw;
        raw.kind = Mutation::Gaussian;
        raw.scale = 5.0;
        std::vector<double> z(10, 0.0);
        raw.apply(z, bd, 20.0, 1.0, rng, /*caller_repairs=*/true);
        bool left = false;
        for (double v : z) left = left || v < -2.0 || v > 3.0;
        check(left, "with the caller repairing, the steps are left raw");
    }

    // ── (7) the crossovers ───────────────────────────────────────────────────
    {
        const Bounds bd(6, {0.0, 1.0});
        const std::vector<double> p1 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.95};
        const std::vector<double> p2 = {0.9, 0.8, 0.7, 0.6, 0.5, 0.05};
        std::vector<double> c1, c2;
        uniform_crossover(p1, p2, c1, c2, 1.0, rng);
        bool comp = true;
        for (std::size_t j = 0; j < p1.size(); ++j)
            comp = comp && ((c1[j] == p1[j] && c2[j] == p2[j]) || (c1[j] == p2[j] && c2[j] == p1[j]));
        check(comp, "uniform: every variable from one parent, the other child's from the other");
        bool ok = true;
        for (BoundRepair b : {BoundRepair::Clip, BoundRepair::Reflect, BoundRepair::Random,
                              BoundRepair::Midpoint, BoundRepair::Resample, BoundRepair::Wrap})
            for (int t = 0; t < 300; ++t) {
                blx_alpha_crossover(p1, p2, c1, c2, bd, 2.0, 1.0, b, rng);
                for (double v : c1) ok = ok && v >= 0.0 && v <= 1.0;
                for (double v : c2) ok = ok && v >= 0.0 && v <= 1.0;
            }
        check(ok, "BLX-alpha children stay inside under every repair, even at alpha = 2");
        blx_alpha_crossover(p1, p2, c1, c2, bd, 0.0, 1.0, BoundRepair::Clip, rng);
        bool between = true;
        for (std::size_t j = 0; j < p1.size(); ++j)
            between = between && c1[j] >= std::min(p1[j], p2[j]) && c1[j] <= std::max(p1[j], p2[j]);
        check(between, "BLX-0 draws between the parents");
        CrossoverSpec sbx_spec;
        std::vector<double> a1, a2, b1, b2;
        std::mt19937 r1(6u), r2(6u);
        sbx_spec.apply(p1, p2, a1, a2, bd, 20.0, 0.9, r1);
        sbx(p1, p2, b1, b2, bd, 20.0, 0.9, r2);
        check(a1 == b1 && a2 == b2 && r1() == r2(), "the SBX choice is ops::sbx, draw for draw");
    }

    // ── (8) DE/rand/2/bin and DE/best/1/bin ──────────────────────────────────
    {
        const Bounds bd(3, {-10.0, 10.0});
        const std::vector<double> r1 = {1, 2, 3}, r2 = {0.5, 0.5, 0.5}, r3 = {1, 1, 1},
                                  r4 = {0, 1, 0}, r5 = {2, 0, 1}, x = {9, 9, 9};
        std::vector<double> y;
        de_rand_2_bin(r1, r2, r3, r4, r5, x, y, bd, 0.5, 1.0, BoundRepair::Clip, rng);
        bool f = true;
        for (int j = 0; j < 3; ++j)
            f = f && std::abs(y[j] - (r1[j] + 0.5 * (r2[j] + r3[j] - r4[j] - r5[j]))) < 1e-12;
        check(f, "DE/rand/2: v = x_r1 + F(x_r2 + x_r3 - x_r4 - x_r5)");
        de_best_1_bin(r1, r2, r3, x, y, bd, 0.5, 1.0, BoundRepair::Clip, rng);
        f = true;
        for (int j = 0; j < 3; ++j) f = f && std::abs(y[j] - (r1[j] + 0.5 * (r2[j] - r3[j]))) < 1e-12;
        check(f, "DE/best/1: v = x_best + F(x_r1 - x_r2)");
    }

    std::printf("bound repair: %d/%d checks passed%s\n", g_checks - g_failed, g_checks,
                g_failed ? ", FAILED" : "");
    return g_failed == 0 ? 0 : 1;
}
