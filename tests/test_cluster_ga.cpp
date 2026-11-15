// SPDX-License-Identifier: Apache-2.0
// cluster_ga (task 26, item 24.2.9; scalar/cluster_ga.hpp):
//   * cga_pop = 1, cga_de = 0 is pbh with pbh_K = 1 (monotonic basin
//     hopping), bit for bit — the degenerate case;
//   * the built-in cut and splice (CGA-1): k points of the first parent as
//     they are, N - k of the second moved into its frame, inside the box;
//   * the refusals; the crossover hook's children; the same run twice;
//   * minima_log (task 30, §30.3; strategy.hpp STRAT-15): all is the
//     default run bit for bit; best keeps min(k, found) minima, the lowest
//     among them; off has no minima row and keeps nothing however many
//     answers come; a strategy reading its log refuses a log not whole;
//   * round_stop (task 30, §30.9; STRAT-16; target by default, method
//     version 5): with the start already at the target, round_stop = off
//     runs the first round's other members (their calls of the problem
//     made, not in the result) as the baseline 8909670 did, its numbers bit
//     for bit (kOld*); target leaves them out, the result bit for bit;
//     several threads run them all, the same result; the default is
//     kDefaultStop;
//   * cga_init (task 30, §30.5; CGA-5): uniform is the default run bit for
//     bit; around_x0 calls the mutate hook (K - 1) times cga_init_moves in
//     the first round; seed_points starts members 2 ... from the seeds;
//   * cga_islands (task 30, §30.8; CGA-6): one island is the default run
//     bit for bit; three islands on 1 thread and on 3 give the same run,
//     every island's members in Result::population, migrants placed;
//   * a short check on the 7-atom Lennard-Jones cluster: the global minimum
//     -16.505384 (Wales & Doye 1997, Table I).

#include <algorithm>
#include <atomic>
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

// the default of round_stop (STRAT-16), and the round_stop scenario at
// 8909670 (no round_stop then): what round_stop = off gives again
const char* kDefaultStop = "target";
const long long kOldCalls = 1002, kOldEvaluations = 130;

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

    // ── minima_log (task 30, §30.3; STRAT-15) ───────────────────────────────
    {
        const sc::Problem R = rastrigin(4);
        const sc::Knobs base{{"cga_pop", number(6)}};
        auto with = [&](sc::Knobs extra) {
            sc::Knobs k = base;
            for (auto& e : extra) k.push_back(e);
            return run("cluster_ga", R, k, 8000, from);
        };
        auto rows = [&](const sc::Result& r) {
            const std::string m = setting(r, "minima");
            return m.empty() ? -1 : static_cast<int>(mootation::json::parse(m).items.size());
        };
        auto lowest = [&](const sc::Result& r) {
            double lo = INFINITY;
            for (const auto& row : mootation::json::parse(setting(r, "minima")).items) {
                const auto& v = row.items[0];
                lo = std::min(lo, v.kind == mootation::json::Value::Kind::Int ? static_cast<double>(v.i) : v.f);
            }
            return lo;
        };
        const sc::Result d = with({});
        const sc::Result all = with({{"minima_log", word("all")}});
        check(same(d, all) && setting(d, "minima") == setting(all, "minima") && setting(d, "minima_log").empty() &&
                  setting(all, "minima_log") == "all",
              "minima_log = all: the default run and log, bit for bit (" + std::to_string(rows(d)) + " minima)");
        const sc::Result b3 = with({{"minima_log", word("best")}, {"minima_best", number(3)}});
        check(same(d, b3) && rows(b3) == std::min(3, rows(d)) && lowest(b3) == lowest(d),
              "minima_log = best, minima_best = 3: the same run, " + std::to_string(rows(b3)) + " minima, the lowest kept");
        const sc::Result big = with({{"minima_log", word("best")}, {"minima_best", number(100000)}});
        check(setting(big, "minima") == setting(d, "minima"), "minima_log = best with k above the count: the whole log");
        const sc::Result off = with({{"minima_log", word("off")}});
        bool no_row = true;
        for (const auto& kv : off.settings) no_row = no_row && kv.first != "minima";
        check(same(d, off) && no_row, "minima_log = off: the same run, no minima row in the settings");
        const sc::Result stray = with({{"minima_best", number(3)}});
        check(std::find(stray.ignored.begin(), stray.ignored.end(), "minima_best") != stray.ignored.end() &&
                  std::find(b3.ignored.begin(), b3.ignored.end(), "minima_best") == b3.ignored.end(),
              "minima_best without minima_log = best: ignored");
        bool threw = false;
        try { run("multistart", R, {{"minima_log", word("best")}, {"minima_bayes", word("on")}}, 2000, from); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "multistart: minima_bayes with a log not whole refused");
        threw = false;
        try { with({{"minima_log", word("some")}}); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "cluster_ga: an unknown minima_log refused");

        // the log itself: 10 000 answers far apart, their f from a generator
        std::mt19937 gen(5);
        std::vector<double> fs;
        sc::strategy::MinimaLog keep_off({0.0, 0.0}, {1.0, 1.0}, 1e-3, 1e-8), keep5({0.0, 0.0}, {1.0, 1.0}, 1e-3, 1e-8);
        keep_off.keep(sc::strategy::MinimaLog::Keep::off, 0);
        keep5.keep(sc::strategy::MinimaLog::Keep::best, 5);
        for (int i = 0; i < 10000; ++i) {
            const Vec x{0.01 * (i % 100), 0.01 * (i / 100)};
            const double f = mootation::rnd::uniform01(gen);
            fs.push_back(f);
            keep_off.add(x, f, i);
            keep5.add(x, f, i);
        }
        std::sort(fs.begin(), fs.end());
        std::vector<double> kept;
        for (const auto& m : keep5.list()) kept.push_back(m.f);
        std::sort(kept.begin(), kept.end());
        check(keep_off.list().empty() && keep_off.json_text() == "[]", "minima_log = off: 10 000 answers, nothing kept");
        check(kept == std::vector<double>(fs.begin(), fs.begin() + 5), "minima_log = best: the 5 lowest of 10 000 answers");
    }

    // ── round_stop (task 30, §30.9; STRAT-16) ───────────────────────────────
    {
        std::atomic<long long> calls{0};              // the starts may call from two threads
        sc::Problem S;
        S.lo.assign(4, -5.0);
        S.hi.assign(4, 5.0);
        S.m = 1;
        S.stop_at = 1e-6;
        S.evaluate = [&calls](const Vec& x, Vec& f, Vec&) {
            ++calls;
            double s = 0.0;
            for (double v : x) s += v * v;
            f.assign(1, s);
        };
        sc::Start near;
        near.has_x0 = true;
        near.x0 = {0.1, -0.1, 0.05, 0.02};
        auto with = [&](const char* stop, int threads, long long& made) {
            sc::Knobs k{{"cga_pop", number(6)}, {"start_budget", number(500)}, {"threads", number(threads)}};
            if (stop) k.push_back({"round_stop", word(stop)});
            calls = 0;
            const sc::Result r = run("cluster_ga", S, k, 20000, near);
            made = calls.load();
            return r;
        };
        long long c_default = 0, c_off = 0, c_target = 0, c_threads = 0, c_kept = 0;
        const sc::Result d = with(nullptr, 1, c_default), off = with("off", 1, c_off), tgt = with("target", 1, c_target);
        const sc::Result thr = with("target", 2, c_threads), kept = with(kDefaultStop, 1, c_kept);
        check(!d.objectives.empty() && d.objectives[0][0] <= 1e-6 && c_off > d.evaluations,
              "the start reaches the target in the first round; the other members still run: " + std::to_string(c_off) +
                  " calls, " + std::to_string(d.evaluations) + " in the result (30.9)");
        check(same(d, off) && same(d, tgt) && same(d, thr), "round_stop = target: the same result, bit for bit, 1 or 2 threads");
        check(c_target < c_off && setting(tgt, "round_skipped") == "5" && setting(thr, "round_skipped") == "0",
              "round_stop = target: the 5 members after the start left out, " + std::to_string(c_target) + " calls (threads: all run)");
        check(same(d, kept) && c_default == c_kept && setting(d, "round_stop").empty() && setting(d, "round_skipped").empty(),
              std::string("round_stop: the default is ") + kDefaultStop);
        check(c_off == kOldCalls && off.evaluations == kOldEvaluations,
              "round_stop = off: the baseline's calls and evaluations, " + std::to_string(c_off) + " and " +
                  std::to_string(off.evaluations));
    }

    // ── cga_init (task 30, §30.5; CGA-5) ────────────────────────────────────
    {
        const sc::Problem R = rastrigin(4);
        const sc::Knobs base{{"cga_pop", number(6)}};
        const sc::Result d = run("cluster_ga", R, base, 6000, from);
        sc::Knobs ku = base;
        ku.push_back({"cga_init", word("uniform")});
        check(same(d, run("cluster_ga", R, ku, 6000, from)), "cga_init = uniform: the default run, bit for bit");
        sc::Problem M = rastrigin(4);
        long long moves = 0;
        M.mutate = [&moves](Vec& y, std::mt19937& g) {
            ++moves;
            for (double& v : y) v += 0.2 * (2.0 * mootation::rnd::uniform01(g) - 1.0);
        };
        const sc::Knobs ka{{"cga_pop", number(6)}, {"cga_init", word("around_x0")}, {"cga_init_moves", number(3)},
                           {"mutation", word("user")}, {"cga_mut_rate", number(0)}};
        const sc::Result a = run("cluster_ga", M, ka, 6000, from);
        check(moves == 15 && setting(a, "init_moves") == "15" && setting(a, "mutations") == "0",
              "cga_init = around_x0: the mutate hook (6 - 1) x 3 times in the first round, " + std::to_string(moves));
        bool threw = false;
        try { run("cluster_ga", R, {{"cga_init", word("around_x0")}}, 1000, {}); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "cga_init = around_x0 without x0 refused");
        sc::Start seeded = from;
        seeded.has_seed_points = true;
        seeded.seed_points.push_back(sc::Point{{0.0, 0.0, 0.0, 0.0}, {}, {}});
        seeded.seed_points.push_back(sc::Point{{1.0, 1.0, 1.0, 1.0}, {}, {}});
        const sc::Knobs ks{{"cga_pop", number(6)}, {"cga_init", word("seed_points")}};
        const sc::Result s1 = run("cluster_ga", R, ks, 6000, seeded), s2 = run("cluster_ga", R, ks, 6000, seeded);
        check(same(s1, s2) && setting(s1, "init_seeded") == "2" && !s1.objectives.empty() && s1.objectives[0][0] < 1e-6,
              "cga_init = seed_points: two members from the seeds (one at the minimum), the rest uniform");
    }

    // ── cga_islands (task 30, §30.8; CGA-6) ─────────────────────────────────
    {
        const sc::Problem R = rastrigin(4);
        const sc::Result d = run("cluster_ga", R, {{"cga_pop", number(4)}}, 8000, from);
        const sc::Result one = run("cluster_ga", R, {{"cga_pop", number(4)}, {"cga_islands", number(1)}}, 8000, from);
        check(same(d, one) && d.population.size() == 4, "cga_islands = 1: the default run, bit for bit");
        auto isl = [&](int threads) {
            return run("cluster_ga", R,
                       {{"cga_pop", number(4)}, {"cga_islands", number(3)}, {"cga_migrate_every", number(2)},
                        {"threads", number(threads)}},
                       8000, from);
        };
        const sc::Result t1 = isl(1), t3 = isl(3);
        check(same(t1, t3) && t1.population == t3.population && t1.population.size() == 12 &&
                  setting(t1, "migrated") == setting(t3, "migrated") && !setting(t1, "migrated").empty(),
              "cga_islands = 3: 1 thread = 3 threads, 12 members, migrated " + setting(t1, "migrated"));
        bool threw = false;
        try { run("cluster_ga", R, {{"cga_islands", number(2)}, {"cga_replace", word("afpo")}}, 1000, from); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "cga_islands > 1 with afpo refused");
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
