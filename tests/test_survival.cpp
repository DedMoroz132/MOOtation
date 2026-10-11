// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// survival = "stochastic" and the inner archive (task 7 of 2026-10-10,
// supplement 2): Bian, Zhou, Li & Qian's stochastic population update in
// SMS-EMOA, NSGA-II and NSGA-III, and operators/inner_archive.hpp.
//   * Lemma 1: a member dominated by every other survives a step with
//     probability at least 1/2 (SMS-EMOA) and 1/4 (NSGA-II, NSGA-III); never
//     with the greedy selection.
//   * The sizes: μ after a step, and Q \ Q' all in the new population.
//   * The selection inside Q' reads Q' alone: SMS-4's extremes of Q'.
//   * spu_frac = 1 is greedy bit for bit; too small a Q' is refused.
//   * The archive: mutually non-dominated, every non-dominated point seen
//     (no cap), at most the cap after every addition; archive_reuse_p = 1
//     draws every parent from it.
//   * The cut: dss::order (the fast one) leaves the archive's members, in
//     their order, bit for bit as a cut by dss::order_plain after every
//     addition of long streams; and orders random sets (dominated points,
//     duplicates, ties) as order_plain does, at every k.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "harness.hpp"

namespace mootation::testing {
// f = x: points put where the test wants them
struct IdentSpec {
    static constexpr int NVARS = 2;
    static constexpr int NOBJS = 2;
    static void eval(const std::vector<double>& x, std::vector<double>& f)
    {
        f[0] = x[0];
        f[1] = x[1];
    }
};
}   // namespace mootation::testing

MOOTATION_TEST_PROBLEM(SpuSMS, NSGAII_Individual, IdentSpec)
MOOTATION_TEST_PROBLEM(SpuNS2, NSGAII_Individual, IdentSpec)
MOOTATION_TEST_PROBLEM(SpuNS3, NSGAIII_Individual, IdentSpec)
MOOTATION_TEST_PROBLEM(SpuZDT1, NSGAII_Individual, ZDT1Spec)

namespace {

using namespace mootation;
using namespace mootation::testing;
using Vec = std::vector<double>;

const Vec DOMINATED = {0.95, 0.95};

// k points of the front f2 = 1 - f1 inside (0.1, 0.9), each dominating DOMINATED
std::vector<Vec> front(int k)
{
    std::vector<Vec> P;
    for (int i = 0; i < k; ++i) {
        const double t = 0.1 + 0.8 * i / std::max(1, k - 1);
        P.push_back({t, 1.0 - t});
    }
    return P;
}

template <typename Tag>
bool holds(DataVault<Tag>& v, const Vec& f)
{
    for (std::size_t i = 0; i < v.active_n(); ++i)
        if (v.objectives_of(i) == f) return true;
    return false;
}

void test_lemma_sms()
{
    std::cout << "-- SMS-EMOA: the dominated member survives with p >= 1/2 --\n";
    const int n = 10, reps = 10000;
    for (const char* mode : {"stochastic", "greedy"}) {
        SMSEMOACore<SpuSMS> core;
        core.set_seed(3);
        core.set_word("survival", mode);
        int kept = 0;
        bool sizes = true;
        for (int r = 0; r < reps; ++r) {
            DataVault<SpuSMS> v(n, Problem<SpuSMS>{});
            const auto P = front(n);
            for (int i = 0; i < n - 1; ++i) v.set_variables(static_cast<std::size_t>(i), P[i]);
            v.set_variables(static_cast<std::size_t>(n - 1), DOMINATED);
            const int q = v.expand(1);
            v.set_variables(static_cast<std::size_t>(q), P[n - 1]);
            v.sync();
            core.reduce_once(v);
            sizes = sizes && v.active_n() == static_cast<std::size_t>(n);
            kept += holds(v, DOMINATED);
        }
        const double share = static_cast<double>(kept) / reps;
        if (std::string(mode) == "stochastic")
            check(share >= 0.5 - 3.0 * std::sqrt(0.25 / reps) && sizes,
                  "stochastic: survives " + std::to_string(share) + " >= 1/2 - 3 sigma; mu left");
        else
            check(kept == 0, "greedy: never survives");
    }
}

template <typename Core, typename Tag>
void lemma_two(const char* name)
{
    const int n = 10, reps = 10000;
    for (const char* mode : {"stochastic", "greedy"}) {
        Core core;
        core.set_seed(5);
        core.set_word("survival", mode);
        int kept = 0;
        bool sizes = true, untouched = true;
        for (int r = 0; r < reps; ++r) {
            DataVault<Tag> v(n, Problem<Tag>{});
            if (r == 0) core.setup(v);                     // NSGA-III: its reference points
            const auto P = front(2 * n - 1);
            for (int i = 0; i < n; ++i) v.set_variables(static_cast<std::size_t>(i), P[i]);
            const int base = v.expand(n);
            for (int i = 0; i < n - 1; ++i)
                v.set_variables(static_cast<std::size_t>(base + i), P[n + i]);
            v.set_variables(static_cast<std::size_t>(base + n - 1), DOMINATED);
            v.sync();
            core.select(v);
            sizes = sizes && v.active_n() == static_cast<std::size_t>(n);
            if (std::string(mode) == "stochastic")
                for (const auto& f : core.last_untouched()) untouched = untouched && holds(v, f);
            kept += holds(v, DOMINATED);
        }
        const double share = static_cast<double>(kept) / reps;
        if (std::string(mode) == "stochastic")
            check(share >= 0.25 - 3.0 * std::sqrt(0.25 * 0.75 / reps) && sizes && untouched,
                  std::string(name) + " stochastic: survives " + std::to_string(share) +
                  " >= 1/4 - 3 sigma; mu left; Q \\ Q' all kept");
        else
            check(kept == 0, std::string(name) + " greedy: never survives");
    }
}

void test_lemma_nsga()
{
    std::cout << "-- NSGA-II, NSGA-III: the dominated member survives with p >= 1/4 --\n";
    lemma_two<NSGAIICore<SpuNS2>, SpuNS2>("nsga2");
    lemma_two<NSGAIIICore<SpuNS3>, SpuNS3>("nsga3");
}

void test_within_subset()
{
    std::cout << "-- SMS-EMOA's Reduce on Q' alone: the extremes of Q' --\n";
    // P1 (0, 1), P2 (0.2, 0.45), P3 (0.6, 0.15), P4 (1, 0): on all four the
    // extremes P1 and P4 stay and P3 (0.4 x 0.3 = 0.12 < 0.4 x 0.55) goes; on
    // Q' = {P1, P2, P3} the extremes are P1 and P3, and P2 goes
    DataVault<SpuSMS> v(4, Problem<SpuSMS>{});
    const std::vector<Vec> P = {{0.0, 1.0}, {0.2, 0.45}, {0.6, 0.15}, {1.0, 0.0}};
    for (std::size_t i = 0; i < 4; ++i) v.set_variables(i, P[i]);
    v.sync();
    SMSEMOACore<SpuSMS> core;
    check(core.worst_of_first(v, 4) == 2, "on Q: P3 goes");
    check(core.worst_of_first(v, 3) == 1, "on Q' = {P1, P2, P3}: P2 goes, P3 now an extreme");
}

template <typename Core>
std::vector<Vec> run(const std::string& survival, double frac, const std::string& archive = "",
                     double reuse = 0.0)
{
    Problem<SpuZDT1> prob;
    DataVault<SpuZDT1> vault(20, prob);
    Optimizer<SpuZDT1, Core> opt(std::move(vault), defer_setup);
    auto& core = opt.get_algorithm();
    core.set_seed(9);
    if (!survival.empty()) {
        core.set_word("survival", survival);
        core.set_param("spu_frac", frac);
    }
    if (!archive.empty()) {
        core.set_word("inner_archive", archive);
        core.set_param("archive_reuse_p", reuse);
    }
    opt.setup();
    for (int g = 0; g < 40; ++g) opt.step();
    auto& v = opt.get_vault();
    std::vector<Vec> out;
    for (std::size_t i = 0; i < v.active_n(); ++i) out.push_back(v.variables_of(i));
    return out;
}

// NSGA-III at an odd N: Opris's |Q'| = ceil(3N/2), so floor(N/2) members of
// R_t stay untouched (NSGA-II's floor would leave ceil(N/2)).
void test_nsga3_odd_population()
{
    std::cout << "-- NSGA-III at N = 11: |Q'| = ceil(3N/2) = 17 (Opris) --\n";
    const int n = 11;
    NSGAIIICore<SpuNS3> core;
    core.set_seed(3);
    core.set_word("survival", "stochastic");
    DataVault<SpuNS3> v(n, Problem<SpuNS3>{});
    core.setup(v);
    const auto P = front(2 * n);
    for (int i = 0; i < n; ++i) v.set_variables(static_cast<std::size_t>(i), P[i]);
    const int base = v.expand(n);
    for (int i = 0; i < n; ++i) v.set_variables(static_cast<std::size_t>(base + i), P[n + i]);
    v.sync();
    core.select(v);
    check(core.last_untouched().size() == 5 && v.active_n() == static_cast<std::size_t>(n),
          "five of the 22 untouched, eleven left");
}

void test_bounds_of_spu_frac()
{
    std::cout << "-- spu_frac = 1 is greedy, too small a Q' refused --\n";
    check(run<NSGAIICore<SpuZDT1>>("", 0) == run<NSGAIICore<SpuZDT1>>("stochastic", 1.0),
          "nsga2: spu_frac = 1 runs the greedy run, bit for bit");
    check(run<SMSEMOACore<SpuZDT1>>("", 0) == run<SMSEMOACore<SpuZDT1>>("stochastic", 1.0),
          "sms_emoa: the same");
    check(run<NSGAIICore<SpuZDT1>>("", 0) != run<NSGAIICore<SpuZDT1>>("stochastic", 0.75),
          "nsga2: spu_frac = 3/4 is another run");
    check(run<NSGAIICore<SpuZDT1>>("", 0) == run<NSGAIICore<SpuZDT1>>("", 0, "nd", 0.0),
          "nsga2: an archive no parent is drawn from changes nothing");
    bool small = false, tiny = false;
    try { run<NSGAIICore<SpuZDT1>>("stochastic", 0.4); } catch (const std::invalid_argument&) { small = true; }
    try { run<SMSEMOACore<SpuZDT1>>("stochastic", 0.04); } catch (const std::invalid_argument&) { tiny = true; }
    check(small, "nsga2: |Q'| = floor(0.4 * 40) < 20 refused");
    check(tiny, "sms_emoa: |Q'| = floor(0.04 * 21) = 0 refused");
}

void test_archive()
{
    std::cout << "-- the inner archive --\n";
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<Vec> seen;
    ops::InnerArchive all, capped;
    all.on = capped.on = true;
    all.cap = 0;
    capped.cap = 15;
    all.start(10, false);
    capped.start(10, false);
    bool within = true;
    for (int t = 0; t < 3000; ++t) {
        const double a = u(rng), b = u(rng);
        const Vec f = {a, 1.0 - a + 0.3 * b * b};
        seen.push_back(f);
        all.add(f, {a, b});
        capped.add(f, {a, b});
        within = within && capped.members().size() <= 15;
    }
    std::set<Vec> nd;
    for (const auto& f : seen) {
        bool beaten = false;
        for (const auto& g : seen)
            if (g != f && g[0] <= f[0] && g[1] <= f[1]) { beaten = true; break; }
        if (!beaten) nd.insert(f);
    }
    std::set<Vec> got;
    for (const auto& m : all.members()) got.insert(m.f);
    check(got == nd, "no cap: exactly the non-dominated points seen (" + std::to_string(nd.size()) + ")");
    check(within, "cap 15: at most 15 members after every addition");
    bool refused = false;
    ops::InnerArchive off;
    off.reuse_p = 0.5;
    try { off.start(10, false); } catch (const std::invalid_argument&) { refused = true; }
    check(refused, "archive_reuse_p > 0 without the archive is refused");

    ops::inner_archive_record() = ops::InnerArchiveRecord{};
    run<NSGAIICore<SpuZDT1>>("", 0, "nd", 1.0);
    const auto& r = ops::inner_archive_record();
    check(r.t_drawn > 0 && r.t_from_archive == r.t_drawn,
          "nsga2, archive_reuse_p = 1: every parent from the archive");
    ops::inner_archive_record() = ops::InnerArchiveRecord{};
    run<SMSEMOACore<SpuZDT1>>("", 0, "nd", 0.5);
    const double share = static_cast<double>(ops::inner_archive_record().t_from_archive) /
                         static_cast<double>(ops::inner_archive_record().t_drawn);
    check(std::abs(share - 0.5) < 0.15, "sms_emoa, archive_reuse_p = 1/2: about half");
}

// InnerArchive::add as first written: every addition beyond the cap cut by
// dss::order_plain, the selection written out plainly.
struct PlainCutArchive {
    std::size_t cap;
    std::vector<ops::InnerArchiveMember> members;
    long long cuts = 0;

    void add(const Vec& f, const Vec& x)
    {
        auto dominates = [](const Vec& a, const Vec& c) {
            bool lt = false;
            for (std::size_t j = 0; j < a.size(); ++j) {
                if (a[j] > c[j]) return false;
                lt = lt || a[j] < c[j];
            }
            return lt;
        };
        for (const auto& m : members)
            if (m.f == f || dominates(m.f, f)) return;
        members.erase(std::remove_if(members.begin(), members.end(),
                                     [&](const ops::InnerArchiveMember& m) { return dominates(f, m.f); }),
                      members.end());
        members.push_back({f, x, {}});
        if (members.size() > cap) {
            ++cuts;
            std::vector<Vec> F;
            for (const auto& m : members) F.push_back(m.f);
            std::vector<ops::InnerArchiveMember> kept;
            for (std::size_t i : dss::order_plain(F, cap)) kept.push_back(members[i]);
            members = std::move(kept);
        }
    }
};

bool same_bits(const Vec& a, const Vec& b)
{
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

void test_archive_cut()
{
    std::cout << "-- the archive cut: dss::order as dss::order_plain, bit for bit --\n";
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    // Two objectives: a front with a flat tail and far outliers (points set
    // aside); three: values on a grid of 1/40 (ties); five objectives.
    const int ms[] = {2, 3, 5};
    const std::size_t caps[] = {100, 80, 50};
    const int adds[] = {4000, 3000, 2000};
    for (int s = 0; s < 3; ++s) {
        const int m = ms[s];
        ops::InnerArchive fast;
        fast.on = true;
        fast.cap = static_cast<long long>(caps[s]);
        fast.start(10, false);
        PlainCutArchive plain{caps[s], {}};
        bool same = true;
        for (int t = 0; t < adds[s] && same; ++t) {
            Vec f(static_cast<std::size_t>(m));
            if (m == 2) {
                const double a = u(rng);
                f = {a, 1.0 - std::sqrt(a) + 0.01 * u(rng)};
                if (a > 0.8) f[1] = 1e-4 * (1.0 - a);
                if (u(rng) < 0.01) f = {3.0 + u(rng), -1e-3 * u(rng)};
            } else {
                double norm = 0.0;
                for (auto& v : f) {
                    v = -std::log(u(rng) + 1e-300);
                    norm += v * v;
                }
                const double r = (1.0 + 0.05 * u(rng)) / std::sqrt(norm);
                for (auto& v : f) v = m == 3 ? std::round(v * r * 40.0) / 40.0 : v * r;
            }
            const Vec x = {u(rng), u(rng)};
            fast.add(f, x);
            plain.add(f, x);
            const auto& got = fast.members();
            same = got.size() == plain.members.size();
            for (std::size_t i = 0; same && i < got.size(); ++i)
                same = same_bits(got[i].f, plain.members[i].f) && same_bits(got[i].x, plain.members[i].x);
        }
        const std::string what = std::to_string(m) + " objectives, cap " + std::to_string(caps[s]) + ", " +
                                 std::to_string(plain.cuts) + " cuts";
        std::cout << "   " << what << '\n';
        check(same && plain.cuts > 1000, what + ": the same members after every addition");
    }

    // Sets an archive never holds: dominated points, exact duplicates, a
    // constant objective, large values; every k.
    bool same = true;
    for (int trial = 0; trial < 400 && same; ++trial) {
        const std::size_t m = 1 + rng() % 5, n = 1 + rng() % 120;
        const unsigned kind = rng() % 4;
        std::vector<Vec> F(n, Vec(m));
        for (auto& f : F) {
            for (auto& v : f) v = u(rng);
            if (kind == 1)
                for (auto& v : f) v = std::round(v * 4.0) / 4.0;
            if (kind == 2 && u(rng) < 0.3) f[0] = 0.25;
            if (kind == 3)
                for (auto& v : f) v = 1e8 + std::round(v * 3.0);
        }
        if (n > 3) F[n - 1] = F[0];
        for (std::size_t k : {std::size_t(0), std::size_t(1), n / 2, n - 1, n, n + 3})
            same = same && dss::order(F, k) == dss::order_plain(F, k);
    }
    check(same, "400 sets of up to 120 points, 1 to 5 objectives: the same order at every k");
    std::vector<Vec> F = {{0.1, 0.9}, {0.5, 0.5}, {0.9, 0.1}, {0.3, std::nan("")}};
    check(dss::order(F, 4) == dss::order_plain(F, 4), "a value not finite: order_plain's order");
}

}   // namespace

int main()
{
    std::cout << "survival and the inner archive\n";
    test_lemma_sms();
    test_lemma_nsga();
    test_within_subset();
    test_nsga3_odd_population();
    test_bounds_of_spu_frac();
    test_archive();
    test_archive_cut();
    return mootation::testing::report("survival");
}
