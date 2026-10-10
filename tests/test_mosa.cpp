// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// mosa — multi-objective simulated annealing (algorithms/mosa.hpp; task 7 of
// 2026-10-10, §1.5, §1.2a and §6): the energies on hand-made archives, the
// archive's rule, the schedules' ends, the reseeding rate, the differential
// step, the box, the answer, PSA's weights and the adaptive step.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "harness.hpp"

MOOTATION_TEST_PROBLEM(MosaZDT1, NSGAII_Individual, ZDT1Spec)
MOOTATION_TEST_PROBLEM(MosaDTLZ2, NSGAII_Individual, DTLZ2Spec)

namespace {

using namespace mootation;
using namespace mootation::testing;
using Core = MOSACore<MosaZDT1>;
using P = Core::Point;

bool mutually_nondominated(const std::vector<P>& A)
{
    for (const auto& a : A)
        for (const auto& b : A) {
            if (&a == &b) continue;
            bool le = true, lt = false;
            for (std::size_t j = 0; j < a.f.size(); ++j) { le = le && a.f[j] <= b.f[j]; lt = lt || a.f[j] < b.f[j]; }
            if (le && lt) return false;
        }
    return true;
}

template <typename Ind>
struct Run {
    Problem<Ind> prob;
    DataVault<Ind> vault;
    Optimizer<Ind, MOSACore<Ind>> opt;
    Run(int pop) : vault(pop, prob), opt(std::move(vault), defer_setup) {}
    MOSACore<Ind>& core() { return opt.get_algorithm(); }
};

void test_energies()
{
    std::cout << "-- the energies on a hand-made archive --\n";
    Core c;
    c.set_word("sa_energy", "dom");
    c.offer({{0.0}, {0.0, 1.0}});
    c.offer({{0.0}, {0.5, 0.5}});
    c.offer({{0.0}, {1.0, 0.0}});
    check(c.archive().size() == 3, "three mutually non-dominated points all kept");
    check(c.energy({0.2, 0.8}, 0) == 0.0, "dom: 0 for a non-dominated point");
    check(std::abs(c.energy({1.0, 1.0}, 0) - 1.0) < 1e-15, "dom: (1, 1) is dominated by all three");
    c.set_word("sa_energy", "cmb");
    check(c.energy({0.2, 0.2}, 0) < 0.0, "cmb: below 0 for a point that extends the front");
    check(c.energy({0.6, 0.6}, 0) > 0.0, "cmb: above 0 for a dominated one");
    check(c.energy({0.5, 0.5}, 0) <= 0.0, "cmb: an archive member's energy leaves itself out");
    c.set_word("sa_energy", "eps");
    // eps of (0.2, 0.2): -min over A of max_i (a_i - f_i)/span_i, span (1, 1)
    check_close(c.energy({0.2, 0.2}, 0), -0.3, 1e-15, "eps: minus the additive epsilon");
    check(!c.offer({{0.0}, {0.6, 0.6}}), "a dominated point is refused");
    check(c.offer({{0.0}, {0.4, 0.4}}) && c.archive().size() == 3,
          "a point dominating a member replaces it");
}

void test_run_and_schedules()
{
    std::cout << "-- a run: the archive, the schedules' ends, the box, the answer --\n";
    const int steps = 3000;
    Run<MosaZDT1> r(40);
    auto& c = r.core();
    c.set_seed(7);
    c.set_t_max(steps);
    c.set_param("sa_s0", 0.9);                 // big steps: the box is reached often
    r.opt.setup();
    bool nd = true;
    for (int t = 0; t < steps; ++t) {
        r.opt.step();
        if (t % 300 == 0) nd = nd && mutually_nondominated(c.archive());
    }
    check(nd && mutually_nondominated(c.archive()), "the archive stays mutually non-dominated");
    check_close(c.temperature(), c.initial_temperature() * 1e-4,
                1e-12 * c.initial_temperature(), "the last step's T is T_0 sa_t_ratio");
    check_close(c.step_scale(), 0.002, 1e-12, "the last step's s is sa_s_end");
    bool inside = true;
    for (const auto& a : c.archive())
        for (double x : a.x) inside = inside && x >= 0.0 && x <= 1.0;
    check(inside, "reflection keeps every point in the box");
    auto& v = r.opt.get_vault();
    const std::size_t n = std::min<std::size_t>(v.active_n(), static_cast<std::size_t>(v.pop_size()));
    bool from_archive = n <= 40 && n > 0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto& f = v.objectives_of(i);
        bool found = false;
        for (const auto& a : c.archive()) found = found || a.f == f;
        from_archive = from_archive && found;
    }
    check(from_archive, "the population: at most N points, all from the archive");
}

void test_reseeding_rate()
{
    std::cout << "-- reseeding: 1/50 of 10^5 steps --\n";
    const int steps = 100000;
    Run<MosaZDT1> r(20);
    auto& c = r.core();
    c.set_seed(3);
    c.set_t_max(steps);
    r.opt.setup();
    for (int t = 0; t < steps; ++t) r.opt.step();
    const double p = 1.0 / 50.0;
    const double share = static_cast<double>(c.reseeds()) / steps;
    check(std::abs(share - p) <= 3.0 * std::sqrt(p * (1.0 - p) / steps),
          "the share of reseeding steps is 1/50 within three standard errors");
}

void test_differential_step()
{
    std::cout << "-- the differential step moves every variable (CR = 1) --\n";
    Run<MosaZDT1> r(20);
    auto& c = r.core();
    c.set_seed(5);
    r.opt.setup();
    bool all = true;
    for (int t = 0; t < 100; ++t) {
        const std::vector<double> x(ZDT1Spec::NVARS, 0.5);
        const auto y = c.differential(x);
        for (std::size_t j = 0; j < x.size(); ++j) all = all && y[j] != x[j];
    }
    check(all, "x + F (x_a - x_b): no variable left as it was");
}

void test_psa_weights()
{
    std::cout << "-- PSA: weights of two chains apart along objective 1 --\n";
    MOSACore<MosaDTLZ2> c;
    c.set_word("sa_energy", "tch");
    c.offer({{0.0}, {0.1, 0.8, 0.5}});
    c.offer({{0.0}, {0.9, 0.2, 0.5}});
    const std::vector<double> w0 = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
    c.set_weights({w0, w0});
    c.set_current({{{0.0}, {0.1, 0.8, 0.5}}, {{0.0}, {0.9, 0.2, 0.5}}});
    c.psa_update(0);
    c.psa_update(1);
    const auto& W = c.weights();
    check(W[0][0] > w0[0] && W[1][0] < w0[0], "objective 1: chain 0 (better there) up, chain 1 down");
    bool ok = true;
    for (const auto& w : W) {
        double s = 0.0;
        for (double v : w) { s += v; ok = ok && v > 0.0; }
        ok = ok && std::abs(s - 1.0) < 1e-12;
    }
    check(ok, "the weights stay positive and sum to 1");

    // a run: psa moves the weights, fixed does not
    for (const char* mode : {"psa", "fixed"}) {
        Run<MosaDTLZ2> r(30);
        auto& d = r.core();
        d.set_seed(9);
        d.set_word("sa_energy", "tch");
        d.set_word("sa_weights", mode);
        d.set_t_max(2000);
        r.opt.setup();
        const auto start = d.weights();
        for (int t = 0; t < 2000; ++t) r.opt.step();
        bool sums = true;
        for (const auto& w : d.weights()) {
            double s = 0.0;
            for (double v : w) { s += v; sums = sums && v > 0.0; }
            sums = sums && std::abs(s - 1.0) < 1e-9;
        }
        const bool moved = d.weights() != start;
        check(sums && (std::string(mode) == "psa" ? moved : !moved),
              std::string("sa_weights = ") + mode + ": " + (std::string(mode) == "psa" ? "the weights move" : "the weights stay"));
    }
}

void test_adaptive_step()
{
    std::cout << "-- the adaptive step: credit over a window --\n";
    Core c;
    c.set_word("sa_step", "adaptive");
    for (int t = 0; t < 150; ++t) c.note_step(1, true);
    for (int t = 0; t < 50; ++t) c.note_step(t % 2 == 0 ? 0 : 2, false);
    const auto& p = c.step_probabilities();
    check(std::abs(p[0] + p[1] + p[2] - 1.0) < 1e-12 && p[0] > 0.0 && p[2] > 0.0,
          "the probabilities sum to 1, every one above 0");
    check(p[1] > 0.9, "the step that keeps succeeding takes most of the probability");
    for (int t = 0; t < 200; ++t) c.note_step(0, true);
    check(c.step_probabilities()[0] > 0.9, "the window forgets: now the other step leads");
}

}   // namespace

int main()
{
    std::cout << "mosa\n";
    test_energies();
    test_run_and_schedules();
    test_reseeding_rate();
    test_differential_step();
    test_psa_weights();
    test_adaptive_step();
    return mootation::testing::report("mosa");
}
