// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// hvc.hpp: HVC3D and HVC4D (Guerreiro & Fonseca, IEEE TEVC 2018) against
// contributions_exact (WFG, hv_contribution.hpp) on 10 000 random sets at 3
// and 4 objectives — fronts, dominated points, copies, equal coordinates,
// points on the reference point's faces and outside it — every contribution
// within 1e-12 of the set's hypervolume. The kept structure (HvcKept) through
// streams of single-point additions and removals, SMS-EMOA's add one and
// remove the least among them: its contributions equal those of a structure
// built from scratch for the same set bit for bit. The knob hv_contrib of
// SMS-EMOA and MO-CMA-ES. Small cases by hand.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/hv_contribution.hpp>
#include <mootation/hvc.hpp>
#include <mootation/hypervolume.hpp>

#include "harness.hpp"

MOOTATION_TEST_PROBLEM(HvcSMS, NSGAII_Individual, DTLZ2Spec)
MOOTATION_TEST_PROBLEM(HvcCMA, MOCMAES_Individual, DTLZ2Spec)

namespace {

int checks = 0, failed = 0;

void check(bool ok, const std::string& what)
{
    ++checks;
    if (!ok) {
        ++failed;
        if (failed <= 20) std::printf("  FAIL  %s\n", what.c_str());
    }
}

using Set = std::vector<std::vector<double>>;

// n points at m objectives: on the sphere's positive part, some behind it,
// some copies, some on a grid of q steps (equal coordinates)
Set random_set(std::mt19937_64& rng, std::size_t m, std::size_t n, double behind, double grid)
{
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::normal_distribution<double> g(0.0, 1.0);
    const int q = 1 << (2 + rng() % 4);
    Set F;
    for (std::size_t i = 0; i < n; ++i) {
        const double kind = u(rng);
        if (i > 0 && kind < 0.1) { F.push_back(F[rng() % F.size()]); continue; }   // a copy
        std::vector<double> f(m);
        double s = 0.0;
        for (auto& v : f) { v = std::abs(g(rng)); s += v * v; }
        for (auto& v : f) {
            v = (1.0 + 0.3 * u(rng) * (kind < behind)) * v / std::sqrt(s);
            if (kind > 1.0 - grid) v = std::floor(v * q) / q;
        }
        F.push_back(f);
    }
    return F;
}

// A run of 3000 steps on DTLZ2 at three objectives with hv_contrib = hvc:
// how many of its steps the kept structure took by updates
template <typename Ind, typename Core>
void run_with_hvc(const char* name)
{
    mootation::Problem<Ind> prob;
    mootation::DataVault<Ind> vault(50, prob);
    mootation::Optimizer<Ind, Core> opt(std::move(vault), mootation::defer_setup);
    auto& alg = opt.get_algorithm();
    alg.set_seed(3);
    alg.set_word("hv_contrib", "hvc");
    opt.setup();
    opt.optimize(3000);
    const auto& k = alg.hvc_kept();
    std::printf("%s, DTLZ2 at M = 3, 3000 steps: %ld builds, %ld single-point updates\n",
                name, k.builds(), k.updates());
    check(k.updates() > 2 * k.builds(), std::string(name) + ": the structure kept by updates");
    bool finite = true;
    for (std::size_t i = 0; i < 50; ++i)
        for (double f : opt.get_vault().objectives_of(i)) finite = finite && std::isfinite(f);
    check(finite, std::string(name) + ": a run with hvc");
}

double largest_gap(const std::vector<double>& a, const std::vector<double>& b)
{
    double d = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

}   // namespace

int main()
{
    namespace hv = mootation::hv;
    namespace H = mootation::hypervolume;

    // ── by hand ─────────────────────────────────────────────────────────────
    {
        const std::vector<double> r3 = {1.0, 1.0, 1.0}, r4 = {1.0, 1.0, 1.0, 1.0};
        auto c = hv::contributions_hvc({{0.5, 0.5, 0.5}}, r3);
        check(c.size() == 1 && c[0] == 0.125, "one point at M = 3");
        c = hv::contributions_hvc({{0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}}, r3);
        check(c[0] == 0.125 && c[1] == 0.125 && c[2] == 0.125, "the 3-D staircase: 1/8 each");
        c = hv::contributions_hvc({{0.5, 0.5, 0.5}, {0.6, 0.6, 0.6}}, r3);
        check(std::abs(c[0] - (0.125 - 0.064)) < 1e-15 && c[1] == 0.0,
              "a point dominated by one: the other's contribution less its box");
        c = hv::contributions_hvc({{0.0, 0.5, 0.5}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}}, r3);
        check(c[0] == 0.0 && c[1] == 0.0 && std::abs(c[2] - 0.125) < 1e-15,
              "two copies contribute nothing, the third point its own");
        c = hv::contributions_hvc({{0.5, 0.5, 1.0}, {0.2, 0.2, 2.0}, {0.5, 0.5, 0.5}}, r3);
        check(c[0] == 0.0 && c[1] == 0.0 && c[2] == 0.125, "on the face and outside: nothing");
        c = hv::contributions_hvc({{0.5, 0.5, 0.5, 0.5}}, r4);
        check(c[0] == 0.0625, "one point at M = 4");
        c = hv::contributions_hvc({{0.5, 0.5, 0.5, 0.5}, {0.5, 0.5, 0.5, 0.6}}, r4);
        check(std::abs(c[0] - 0.0625 * 0.2) < 1e-15 && c[1] == 0.0,
              "M = 4: equal in x, y, z, the lower w dominates");
        c = hv::contributions_hvc({{0.0, 0.5, 0.5, 0.5}, {0.5, 0.0, 0.5, 0.5},
                                   {0.5, 0.5, 0.0, 0.5}, {0.5, 0.5, 0.5, 0.0}}, r4);
        bool all = true;
        for (double x : c) all = all && std::abs(x - 0.0625) < 1e-15;
        check(all, "the 4-D staircase: 1/16 each");
    }

    // ── 10 000 random sets against WFG ──────────────────────────────────────
    std::mt19937_64 rng(20261019);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double worst = 0.0, worst_rel = 0.0;
    long compared = 0, equal_bits = 0;
    for (int t = 0; t < 10000; ++t) {
        const std::size_t m = 3 + t % 2;
        std::size_t n = 1 + rng() % (m == 3 ? 80 : 60);
        if (t % 500 == 0) n = m == 3 ? 400 : 150;          // a few large ones
        Set F = random_set(rng, m, n, 0.3, 0.2);
        std::vector<double> r(m, 1.1);
        if (t % 7 == 0) r[0] = F[0][0];                      // a point on the reference's face
        if (t % 11 == 0) r[1] = 0.9;                         // some points outside
        const double Hs = H::wfg(F, r);
        const auto ch = hv::contributions_hvc(F, r), cw = hv::contributions_exact(F, r);
        const double d = largest_gap(ch, cw);
        worst = std::max(worst, Hs > 0.0 ? d / Hs : d);
        check(Hs > 0.0 ? d <= 1e-12 * Hs : d == 0.0,
              "set " + std::to_string(t) + " (M = " + std::to_string(m) + ", n = " +
                  std::to_string(n) + "): largest difference " + std::to_string(d / Hs));
        for (std::size_t i = 0; i < n; ++i) {
            if (cw[i] > 1e-6 * Hs) {
                worst_rel = std::max(worst_rel, std::abs(ch[i] - cw[i]) / cw[i]);
                ++compared;
            }
            equal_bits += ch[i] == cw[i];
        }
    }
    std::printf("hvc against wfg: largest difference %.3g of the set's hypervolume; "
                "largest relative difference %.3g over %ld contributions above 1e-6 of it; "
                "%ld equal bit for bit\n", worst, worst_rel, compared, equal_bits);

    // ── the kept structure: single-point updates against a fresh build ──────
    long steps = 0, same = 0;
    long builds = 0, updates = 0, builds_front = 0, updates_front = 0;
    double worst_k = 0.0;
    for (int stream = 0; stream < 60; ++stream) {
        const std::size_t mu = 5 + rng() % 120;
        const double grid = stream % 3 == 0 ? 0.5 : 0.0;     // equal coordinates in a third
        Set P = random_set(rng, 3, mu, 0.0, grid);
        hv::HvcKept kept;
        for (int s = 0; s < 300; ++s) {
            // SMS-EMOA's step: one in; the contributions; the least out
            Set Q = P;
            const double kind = u(rng);
            if (kind < 0.05) Q.push_back(Q[rng() % Q.size()]);                 // a copy
            else if (kind < 0.10) {                                            // dominated or dominating
                auto f = Q[rng() % Q.size()];
                for (auto& v : f) v *= kind < 0.075 ? 1.05 : 0.95;
                Q.push_back(f);
            } else Q.push_back(random_set(rng, 3, 1, 0.0, grid)[0]);
            std::vector<double> r(3, -1e300);
            for (const auto& f : Q)
                for (std::size_t k = 0; k < 3; ++k) r[k] = std::max(r[k], f[k]);
            for (auto& x : r) x += stream % 2 == 0 ? 1.0 : 0.01;              // r = nad + 1, or tight
            const auto ck = kept.contributions(Q, r);
            const auto cf = hv::contributions_hvc(Q, r);
            ++steps;
            bool bits = ck.size() == cf.size();
            for (std::size_t i = 0; bits && i < ck.size(); ++i) bits = ck[i] == cf[i];
            same += bits;
            check(bits, "stream " + std::to_string(stream) + ", step " + std::to_string(s) +
                            ": the updated structure's contributions are the rebuilt one's");
            if (s % 3 == 0) {                                // and against WFG
                const auto cw = hv::contributions_exact(Q, r);
                const double Hs = H::wfg(Q, r);
                const double d = largest_gap(ck, cw);
                worst_k = std::max(worst_k, d / Hs);
                check(d <= 1e-12 * Hs, "stream " + std::to_string(stream) + ", step " +
                                           std::to_string(s) + ": against wfg");
            }
            std::size_t arg = 0;
            for (std::size_t i = 1; i < ck.size(); ++i) if (ck[i] < ck[arg]) arg = i;
            Q.erase(Q.begin() + static_cast<std::ptrdiff_t>(arg));
            P = Q;
        }
        // single changes away from SMS-EMOA's pattern: two out, one in, none
        for (int s = 0; s < 20; ++s) {
            Set Q = P;
            if (Q.size() > 3 && s % 3 != 2) Q.erase(Q.begin() + static_cast<std::ptrdiff_t>(rng() % Q.size()));
            if (Q.size() > 3 && s % 3 == 0) Q.erase(Q.begin() + static_cast<std::ptrdiff_t>(rng() % Q.size()));
            if (s % 3 == 1) Q.push_back(random_set(rng, 3, 1, 0.0, grid)[0]);
            std::vector<double> r(3, 1.5);
            const auto ck = kept.contributions(Q, r);
            const auto cf = hv::contributions_hvc(Q, r);
            bool bits = true;
            for (std::size_t i = 0; i < ck.size(); ++i) bits = bits && ck[i] == cf[i];
            ++steps;
            same += bits;
            check(bits, "stream " + std::to_string(stream) + ", change " + std::to_string(s));
            P = Q;
        }
        builds += kept.builds();
        updates += kept.updates();
        if (grid == 0.0) { builds_front += kept.builds(); updates_front += kept.updates(); }
    }
    // without equal coordinates a dominated point comes only with the 5 % of
    // offspring made so (a dominating one makes several, removed a step at a
    // time, each step a rebuild): the other steps are updates
    check(updates_front > 5 * builds_front, "the streams without ties ran on updates");
    std::printf("kept structure: %ld/%ld steps equal to a fresh build bit for bit "
                "(%ld builds, %ld single-point updates; without equal coordinates %ld and "
                "%ld); largest difference to wfg %.3g of the set's hypervolume\n",
                same, steps, builds, updates, builds_front, updates_front, worst_k);

    // ── the knob ────────────────────────────────────────────────────────────
    {
        mootation::SMSEMOACore<mootation::NSGAII_Individual> sms;
        check(sms.set_word("hv_contrib", "hvc") && sms.set_word("hv_contrib", "wfg"),
              "sms_emoa takes hv_contrib = hvc and wfg");
        bool refused = false;
        try { sms.set_word("hv_contrib", "box"); } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "sms_emoa refuses hv_contrib = box");
        mootation::MOCMAESCore<mootation::MOCMAES_Individual> cma;
        check(cma.set_word("hv_contrib", "hvc"), "mo_cma_es takes hv_contrib = hvc");
        refused = false;
        try { cma.set_word("hv_contrib", "HVC"); } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "mo_cma_es refuses hv_contrib = HVC");
        // MO-CMA-ES: exact at every M (task 14, supplement 3, Д8.1)
        for (const char* w : {"race", "mc"}) {
            refused = false;
            try { cma.set_word("hv_contrib", w); } catch (const std::invalid_argument&) { refused = true; }
            check(refused, "mo_cma_es refuses hv_contrib = race and mc");
        }
        refused = false;
        try { cma.set_param("n_samples", 10000); } catch (const std::invalid_argument&) { refused = true; }
        check(refused && cma.set_param("n_samples", 0), "mo_cma_es refuses n_samples > 0, takes 0");
        check(!cma.set_param("hv_exact_max_m", 6) && !cma.set_param("race_eps", 0.1) &&
                  !cma.set_param("race_max_samples", 100),
              "mo_cma_es does not take hv_exact_max_m and race_*");
        {
            hv::ContribChoice e;
            e.exact_only = true;
            check(e.way(4) == hv::ContribChoice::Way::exact && e.way(10) == hv::ContribChoice::Way::exact &&
                      e.use_hvc(4),
                  "exact_only: exact at every M, HVC4D at M = 4");
        }
        mootation::R2EMOACore<mootation::NSGAII_Individual> r2;
        check(!r2.set_word("hv_contrib", "hvc"), "r2_emoa has no hypervolume: hv_contrib not taken");
        check(!r2.set_word("hv_algo", "box"), "r2_emoa has no hypervolume: hv_algo not taken");
        check(!r2.set_param("n_samples", 0) && !r2.set_param("hv_exact_max_m", 3),
              "r2_emoa has no hypervolume: n_samples and hv_exact_max_m not taken");
    }
    // ── exact, race or Monte-Carlo: hv::ContribChoice (task 13, §3; task 14, §11) ──
    {
        using Way = hv::ContribChoice::Way;
        hv::ContribChoice c;                                   // the defaults
        check(c.way(2) == Way::exact && c.way(3) == Way::exact && c.way(4) == Way::exact &&
                  c.way(5) == Way::exact && c.way(6) == Way::race && c.way(10) == Way::race,
              "default: exact up to M = 5, the race above");
        check(c.use_hvc(3) && c.use_hvc(4) && !c.use_hvc(5), "default: hvc at M = 3 and 4");
        check(c.samples() == 10000, "default: 10 000 samples a step");
        c.set_param("hv_exact_max_m", 6);
        check(c.way(6) == Way::exact && c.way(7) == Way::race, "hv_exact_max_m = 6: exact at M = 6");
        {
            hv::ContribChoice e;
            e.set_word("hv_contrib", "mc");
            check(e.way(5) == Way::exact && e.way(6) == Way::mc && e.use_hvc(4),
                  "hv_contrib = mc: exact up to the threshold, Monte-Carlo above (task 13's defaults)");
            e.set_param("hv_exact_max_m", 3);
            e.set_word("hv_contrib", "race");
            check(e.way(4) == Way::race, "hv_exact_max_m = 3 with race: the race from M = 4");
            e.set_param("n_samples", 500);
            check(e.way(4) == Way::race, "n_samples with hv_contrib = race: still the race");
            bool refused = false;
            try { e.set_word("hv_contrib", "box"); } catch (const std::invalid_argument&) { refused = true; }
            check(refused, "hv_contrib refuses a word it does not know");
            refused = false;
            try { e.set_param("race_eps", 0.0); } catch (const std::invalid_argument&) { refused = true; }
            check(refused, "race_eps must be > 0");
        }
        c.set_param("n_samples", 0);
        check(!c.mc(10), "n_samples = 0: exact at every M");
        c.set_param("n_samples", 500);
        check(!c.mc(3) && c.mc(4) && c.samples() == 500, "n_samples = 500: Monte-Carlo from M = 4");
        c.set_word("hv_contrib", "wfg");
        check(!c.use_hvc(3) && !c.use_hvc(4), "hv_contrib = wfg");
        bool refused = false;
        try { c.set_param("hv_exact_max_m", 2.5); } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "hv_exact_max_m must be whole");
        check(!c.set_param("ref_delta", 1.0) && !c.set_word("survival", "greedy"),
              "other knobs are not ContribChoice's");
        // the method named in the settings, at the M of the last step
        hv::ContribChoice d;
        std::mt19937 g(1);
        hv::HvcKept kept;
        const std::vector<std::vector<double>> F4 = {{0.1, 0.5, 0.6, 0.7}, {0.5, 0.1, 0.6, 0.7},
                                                    {0.6, 0.5, 0.1, 0.7}, {0.7, 0.6, 0.5, 0.1}};
        const std::vector<double> r4(4, 1.0);
        const auto exact = d.compute(F4, r4, kept, g);
        const auto want = hv::contributions_exact(F4, r4);
        double gap = 0.0;
        for (std::size_t i = 0; i < 4; ++i) gap = std::max(gap, std::abs(exact[i] - want[i]));
        check(gap <= 1e-15, "default at M = 4: the exact contributions (HVC4D)");
        check(d.settings().back().second == "exact, hvc4d at M = 4", "settings: exact, hvc4d");
        d.set_param("n_samples", 1000);
        d.compute(F4, r4, kept, g);
        check(d.settings().back().second ==
                  "monte-carlo, 1000 samples a step (n_samples given) at M = 4",
              "settings: monte-carlo, n_samples given");
    }

    // ── the race (Bringmann & Friedrich 2012; task 14, §11) ─────────────────
    {
        std::mt19937 gen(20261020);
        std::normal_distribution<double> nd(0.0, 1.0);
        std::uniform_real_distribution<double> ud(0.0, 1.0);
        int sets = 0, within = 0, twice = 0, never_skipped = 0, capped = 0, full = 0, full_within = 0;
        hv::RaceSettings set;
        for (int t = 0; t < 400; ++t) {
            const std::size_t m = 4 + static_cast<std::size_t>(t % 5);       // 4 to 8 objectives
            const std::size_t n = 10 + static_cast<std::size_t>(t % 31);
            std::vector<std::vector<double>> F(n, std::vector<double>(m));
            for (auto& row : F) {                                            // a front on the sphere
                double s2 = 0.0;
                for (auto& v : row) { v = std::abs(nd(gen)) + 1e-3; s2 += v * v; }
                for (auto& v : row) v /= std::sqrt(s2);
            }
            const std::vector<double> r(m, 1.1);
            const auto exact = hv::contributions_exact(F, r);
            double least = exact[0];
            for (double v : exact) least = std::min(least, v);
            std::vector<char> skip(n, 0);
            skip[static_cast<std::size_t>(t) % n] = 1;                       // one point kept whatever
            double least_left = std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < n; ++i) if (!skip[i]) least_left = std::min(least_left, exact[i]);
            std::mt19937 a(static_cast<unsigned>(t)), b(static_cast<unsigned>(t));
            hv::RaceStats st;
            const auto c1 = hv::contributions_race(F, r, set, a, &skip, &st);
            const auto c2 = hv::contributions_race(F, r, set, b, &skip);
            std::size_t pick = 0;
            for (std::size_t i = 1; i < n; ++i) if (c1[i] < c1[pick]) pick = i;
            ++sets;
            const bool ok = exact[pick] <= (1.0 + set.eps) * least_left + 1e-15;
            within += ok;
            full += !st.capped;
            full_within += !st.capped && ok;
            twice += c1 == c2;
            never_skipped += !skip[pick];
            capped += st.capped;
            (void)least;
            (void)ud;
        }
        std::printf("race, 400 random sets at M = 4 to 8: within (1 + eps) of the least %d "
                    "(of the %d races that ended by the paper's rule: %d), the same answer twice %d, "
                    "a skipped point never %d, stopped at race_max_samples %d\n",
                    within, full, full_within, twice, never_skipped, capped);
        // the guarantee (Corollary 5) holds for a race that ends by its rule;
        // one stopped at race_max_samples takes the current least
        check(full_within == full && full > sets / 2,
              "the race: the answer within (1 + eps) of the least contribution");
        check(twice == sets, "the race: the same answer at the same seed");
        check(never_skipped == sets, "the race: a skipped point is never the answer");
        // a copy contributes nothing: the race finds it exactly
        std::vector<std::vector<double>> F = {{0.2, 0.5, 0.5, 0.6}, {0.5, 0.2, 0.5, 0.6},
                                              {0.5, 0.5, 0.2, 0.6}, {0.5, 0.5, 0.2, 0.6},
                                              {0.6, 0.6, 0.6, 0.1}};
        std::mt19937 g(3);
        hv::RaceStats zst;
        const auto c = hv::contributions_race(F, std::vector<double>(4, 1.0), set, g, nullptr, &zst);
        check(c[2] == 0.0 && c[3] == 0.0 && zst.samples == 0 && std::isinf(c[0]),
              "the race: copies contribute 0, known without a sample, and are the answer (Д5)");
        // stopped at race_max_samples, the race answers with the least upper
        // bound (Д8.5): at a cap of 1 no candidate has a sample, every bound is
        // the volume of the candidate's box [p, b] — b cut by every point no
        // worse than p in all objectives but one
        {
            const std::size_t m = 6, n = 40;
            std::vector<std::vector<double>> G(n, std::vector<double>(m));
            for (auto& row : G) {
                double s2 = 0.0;
                for (auto& v : row) { v = std::abs(nd(gen)) + 1e-3; s2 += v * v; }
                for (auto& v : row) v /= std::sqrt(s2);
            }
            const std::vector<double> r(m, 1.1);
            std::vector<double> box(n);
            for (std::size_t i = 0; i < n; ++i) {
                std::vector<double> hi = r;
                for (std::size_t q = 0; q < n; ++q) {
                    std::size_t worse = 0, at = 0;
                    for (std::size_t j = 0; j < m; ++j) if (G[q][j] > G[i][j]) { ++worse; at = j; }
                    if (q != i && worse == 1) hi[at] = std::min(hi[at], G[q][at]);
                }
                box[i] = 1.0;
                for (std::size_t j = 0; j < m; ++j) box[i] *= hi[j] - G[i][j];
            }
            hv::RaceSettings one;
            one.max_samples = 1;
            std::mt19937 h(5);
            hv::RaceStats cst;
            const auto cc = hv::contributions_race(G, r, one, h, nullptr, &cst);
            std::size_t pick = 0, small = 0;
            for (std::size_t i = 1; i < n; ++i) {
                if (cc[i] < cc[pick]) pick = i;
                if (box[i] < box[small]) small = i;
            }
            check(cst.capped && cst.samples == 0 && cc == box && pick == small,
                  "the race stopped at race_max_samples: the least upper bound, here the least box (Д8.5)");
        }
    }

    // ── the kept exact counts (task 14, supplement 3, Д8.3) ─────────────────
    // a (mu + 1) sequence: one point in, the least contributor out, now and
    // then r moved; with ExactKept the contributions are those without it,
    // bit for bit, and some counts are taken from the step before (random
    // points on the sphere touch most others' limited sets: about a tenth
    // here; on SMS-EMOA's and MO-CMA-ES's runs 37-54 %)
    {
        std::mt19937 gen(8301);
        std::normal_distribution<double> nd(0.0, 1.0);
        int kept_same = 0, kept_steps = 0;
        std::size_t reused = 0, counted = 0;
        for (std::size_t m : {std::size_t(5), std::size_t(6)}) {
            auto point = [&] {
                std::vector<double> row(m);
                double s2 = 0.0;
                for (auto& v : row) { v = std::abs(nd(gen)) + 1e-3; s2 += v * v; }
                for (auto& v : row) v /= std::sqrt(s2);
                return row;
            };
            std::vector<std::vector<double>> F;
            for (int i = 0; i < 30; ++i) F.push_back(point());
            std::vector<double> r(m, 1.1);
            hv::ExactKept kept;
            for (int t = 0; t < 150; ++t) {
                F.push_back(point());
                if (t % 50 == 49) r[static_cast<std::size_t>(t) % m] += 0.01;
                const auto a = hv::contributions_exact(F, r);
                const auto b = hv::contributions_exact(F, r, &kept);
                kept_same += a.size() == b.size() &&
                        std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
                ++kept_steps;
                std::size_t least = 0;
                for (std::size_t i = 1; i < a.size(); ++i) if (a[i] < a[least]) least = i;
                F.erase(F.begin() + static_cast<std::ptrdiff_t>(least));
            }
            reused += kept.reused;
            counted += kept.counted;
        }
        std::printf("kept exact counts: %d of %d steps the same bit for bit, %zu counts reused, %zu counted\n",
                    kept_same, kept_steps, reused, counted);
        check(kept_same == kept_steps, "ExactKept: the contributions without it, bit for bit (Д8.3)");
        check(reused > 0, "ExactKept: counts reused between (mu + 1) steps");
    }

    run_with_hvc<mootation::testing::HvcSMS, mootation::SMSEMOACore<mootation::testing::HvcSMS>>("sms_emoa");
    run_with_hvc<mootation::testing::HvcCMA, mootation::MOCMAESCore<mootation::testing::HvcCMA>>("mo_cma_es");

    std::printf("hvc: %d/%d checks passed%s\n", checks - failed, checks, failed ? ", FAILED" : "");
    return failed == 0 ? 0 : 1;
}
