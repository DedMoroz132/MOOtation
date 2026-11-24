// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The local search of the population's members (task 34.1; member_search.hpp,
// MLS-1 ... MLS-11; run_core.hpp, RunConfig::local_search; embed.hpp).
//
//   (1) the settings text: the defaults are not written, other values are
//       written and read back, unknown words and values out of range refused;
//   (2) local_search = off with a hook given: the run without one, bit for
//       bit, and the hook never called (MLS-1);
//   (3) ZDT1 with a local well in every distance variable (below), NSGA-II:
//       with lamarck and a parabolic search along the distance variables (its
//       evaluations declared as its cost) the run's spending at its first
//       answer of IGD+ at most 4 times the coverage floor of pop_size points
//       is smaller than off's for at least 3 of 5 seeds (the observer after
//       every pop_size spent); SPEA2 measured the same way and printed;
//   (4) a hook of declared cost c and no evaluations of its own: every call
//       is made with the run's spending below the budget, so at most
//       ceil(budget / c) calls (MLS-2); the settings' ls_calls, ls_cost and
//       used agree with what the hook was called and declared (MLS-10);
//   (5) every answer row is a pair of the problem's, f = f(x), with lamarck
//       and with baldwin (MLS-8);
//   (6) a Session (ask / tell), its local search requests answered with the
//       same hook, is the run of run() bit for bit;
//   (7) four runs on four threads at once are the runs made one after the
//       other (MLS-11);
//   (8) a hook that gives back a point of the wrong size, not finite, out of
//       the box, a negative cost, or throws, ends the run with
//       std::runtime_error (MLS-5); lamarck without a hook is refused;
//   (9) every algorithm of the dispatch on DTLZ2 at 2 and 3 objectives with
//       lamarck and baldwin and a hook that puts the distance variables at
//       their optimum: the cores of member_search::refusal() are refused with
//       their reason, the others run without failing, every answer row a pair
//       of the problem's, and the run is not the run with off (the core took
//       the members written); the table says how many answer rows are such
//       points (kept) and how many of the evaluations after the first call
//       inherit such a variable (bred).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "mootation/embed.hpp"

#include "harness.hpp"

namespace {

using mootation::testing::check;
using Vec = std::vector<double>;
using Rows = std::vector<std::vector<double>>;

// ── ZDT4 (Zitzler, Deb and Thiele, Evol. Comput. 8(2), 2000): x1 in [0, 1],
// x2..x10 in [-5, 5]; f1 = x1, f2 = g (1 - sqrt(x1 / g)), g = 1 + 10 (n - 1) +
// sum (x_i^2 - 10 cos(4 pi x_i)). The front: g = 1, f2 = 1 - sqrt(f1).
constexpr int kZdtVars = 10;

Vec zdt4(const Vec& x)
{
    const double pi = 3.141592653589793;
    double g = 1.0 + 10.0 * static_cast<double>(x.size() - 1);
    for (std::size_t i = 1; i < x.size(); ++i) g += x[i] * x[i] - 10.0 * std::cos(4.0 * pi * x[i]);
    return {x[0], g * (1.0 - std::sqrt(x[0] / g))};
}

mootation::Evaluate zdt4_batch(std::shared_ptr<long long> count = nullptr)
{
    return [count](const Rows& X, Rows& F, Rows&) {
        for (std::size_t i = 0; i < X.size(); ++i) F[i] = zdt4(X[i]);
        if (count) *count += static_cast<long long>(X.size());
    };
}

// a compass search of the distance variables x2..x10 from x: steps of 0.25
// halved when a sweep finds nothing better, until 1e-3 or 60 evaluations;
// minimizing f2 at the same x1 lowers g, a Pareto improvement. Its
// evaluations are its declared cost.
mootation::LocalStep compass(const Vec& x0, const Vec& f0)
{
    mootation::LocalStep out;
    out.x = x0;
    out.f = f0;
    int used = 0;
    for (double step = 0.25; step >= 1e-3 && used < 60; step *= 0.5) {
        bool better = true;
        while (better && used < 60) {
            better = false;
            for (std::size_t i = 1; i < out.x.size() && used < 60; ++i)
                for (double s : {step, -step}) {
                    Vec y = out.x;
                    y[i] = (std::min)(5.0, (std::max)(-5.0, y[i] + s));
                    const Vec fy = zdt4(y);
                    ++used;
                    if (fy[1] < out.f[1]) { out.x = y; out.f = fy; better = true; break; }
                }
        }
    }
    out.cost = used;
    return out;
}

// ── (3)'s problem: ZDT1 (Zitzler, Deb and Thiele 2000) with a local well in
// every distance variable: x in [0, 1]^n, f1 = x1, f2 = g (1 - sqrt(x1 / g)),
// g = 1 + 9 / (n - 1) sum h(x_i), h(u) = min(u^2, depth + 4 (u - 0.6)^2): the
// global minimum of h at 0, a local one of value depth at 0.6. The front: g = 1,
// f2 = 1 - sqrt(f1).
struct Well {
    int    n     = 30;
    int    pop   = 100;
    int    every = 5;
    double share = 0.05;
    double times = 4.0;       // the target: times the coverage floor of pop points
    double depth = 0.02;
};

Vec zdt1w(const Vec& x, double depth)
{
    double s = 0.0;
    for (std::size_t i = 1; i < x.size(); ++i)
        s += (std::min)(x[i] * x[i], depth + 4.0 * (x[i] - 0.6) * (x[i] - 0.6));
    const double g = 1.0 + 9.0 * s / static_cast<double>(x.size() - 1);
    return {x[0], g * (1.0 - std::sqrt(x[0] / g))};
}

// A parabolic search along each distance variable in turn: f2 at u - 0.05 and u
// + 0.05 (moved inside [0, 1] at a bound), the vertex of the parabola through
// the three (the best of them where it does not open upwards), the best kept;
// three evaluations a variable, declared as the cost. At the same x1 a lower f2
// is a lower g: a Pareto improvement.
mootation::LocalStep parabolic(const Vec& x0, const Vec& f0, double depth)
{
    mootation::LocalStep out;
    out.x = x0;
    out.f = f0;
    int used = 0;
    const double d = 0.05;
    for (std::size_t i = 1; i < out.x.size(); ++i) {
        const double u = out.x[i];
        const double a = (std::min)(1.0 - 2.0 * d, (std::max)(0.0, u - d));
        const double p[3] = {a, a + d, a + 2.0 * d};
        double y[3];
        Vec best_x = out.x, best_f = out.f;
        for (int k = 0; k < 3; ++k) {
            Vec t = out.x;
            t[i] = p[k];
            const Vec ft = zdt1w(t, depth);
            ++used;
            y[k] = ft[1];
            if (ft[1] < best_f[1]) { best_x = t; best_f = ft; }
        }
        const double curv = (y[0] - 2.0 * y[1] + y[2]) / (d * d);
        if (curv > 0.0) {
            const double v = (std::min)(1.0, (std::max)(0.0, p[1] - d * (y[2] - y[0]) / (2.0 * d * d * curv)));
            Vec t = out.x;
            t[i] = v;
            const Vec ft = zdt1w(t, depth);
            ++used;
            if (ft[1] < best_f[1]) { best_x = t; best_f = ft; }
        }
        out.x = best_x;
        out.f = best_f;
    }
    out.cost = used;
    return out;
}

mootation::Settings zdt4_settings(const std::string& algorithm, int pop, int budget, unsigned seed)
{
    mootation::Settings s;
    s.algorithm = algorithm;
    s.pop_size = pop;
    s.max_evaluations = budget;
    s.n_objs = 2;
    s.seed = seed;
    s.lower.assign(kZdtVars, -5.0);
    s.upper.assign(kZdtVars, 5.0);
    s.lower[0] = 0.0;
    s.upper[0] = 1.0;
    return s;
}

// IGD+ (Ishibuchi et al., EMO 2015) of an answer against 500 points of the front of
// ZDT1 and ZDT4 (the same front)
double igd_plus_zdt(const Rows& F)
{
    const int R = 500;
    double sum = 0.0;
    for (int k = 0; k < R; ++k) {
        const double t = static_cast<double>(k) / (R - 1);
        const double z[2] = {t, 1.0 - std::sqrt(t)};
        double best = std::numeric_limits<double>::infinity();
        for (const auto& a : F) {
            double d2 = 0.0;
            for (int j = 0; j < 2; ++j) {
                const double d = (std::max)(a[static_cast<std::size_t>(j)] - z[j], 0.0);
                d2 += d * d;
            }
            best = (std::min)(best, std::sqrt(d2));
        }
        sum += best;
    }
    return sum / R;
}

// ── DTLZ2 with m objectives over m + 9 variables in [0, 1] (Deb et al. 2005).
Vec dtlz2_of(const Vec& x, int m)
{
    const double half_pi = 1.5707963267948966;
    double g = 0.0;
    for (std::size_t j = static_cast<std::size_t>(m) - 1; j < x.size(); ++j) g += (x[j] - 0.5) * (x[j] - 0.5);
    Vec f(static_cast<std::size_t>(m));
    for (int k = 0; k < m; ++k) {
        double v = 1.0 + g;
        for (int j = 0; j < m - 1 - k; ++j) v *= std::cos(x[static_cast<std::size_t>(j)] * half_pi);
        if (k > 0) v *= std::sin(x[static_cast<std::size_t>(m - 1 - k)] * half_pi);
        f[static_cast<std::size_t>(k)] = v;
    }
    return f;
}

std::string setting(const mootation::Result& r, const std::string& key)
{
    for (const auto& kv : r.settings)
        if (kv.first == key) return kv.second;
    return "";
}

bool same_run(const mootation::Result& a, const mootation::Result& b)
{
    return a.objectives == b.objectives && a.variables == b.variables && a.cv == b.cv &&
           a.evaluations == b.evaluations && a.generations == b.generations && a.settings == b.settings;
}

// (5): every row f = f(x)
template <typename Fn>
bool honest(const mootation::Result& r, const Fn& f_of)
{
    for (std::size_t i = 0; i < r.size(); ++i)
        if (f_of(r.variables[i]) != r.objectives[i]) return false;
    return true;
}

template <typename Fn>
bool throws_runtime(const Fn& fn, const std::string& part)
{
    try {
        fn();
    } catch (const std::runtime_error& e) {
        return std::string(e.what()).find(part) != std::string::npos;
    } catch (...) {
        return false;
    }
    return false;
}

void test_settings_text()
{
    using mootation::Settings;
    Settings s = zdt4_settings("nsga2", 40, 1000, 1);
    const std::string plain = s.to_string();
    check(plain.find("local_search") == std::string::npos && plain.find("ls_") == std::string::npos,
          "(1) the defaults are not written: the text of a run without the local search is as it was");
    s.local_search = "baldwin";
    s.ls_every = 3;
    s.ls_share = 0.25;
    s.ls_pick = "objective";
    const Settings back = Settings::from_string(s.to_string());
    check(back.local_search == "baldwin" && back.ls_every == 3 && back.ls_share == 0.25 && back.ls_pick == "objective",
          "(1) local_search, ls_every, ls_share, ls_pick written and read back");
    const Settings none = Settings::from_string("pop_size = 10\nlower = 0\nupper = 1\n");
    check(none.local_search == "off" && none.ls_every == 1 && none.ls_share == 0.1 && none.ls_pick == "spread",
          "(1) not given: off, 1, 0.1, spread");
    for (const char* bad : {"local_search = lamark", "ls_every = 0", "ls_share = 0", "ls_share = 1.5",
                            "ls_pick = best"}) {
        bool threw = false;
        try {
            (void)Settings::from_string(std::string(bad) + "\nlower = 0\nupper = 1\n");
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, std::string("(1) refused: ") + bad);
    }
}

void test_off()
{
    for (const char* name : {"nsga2", "moead", "spea2", "smpso"}) {
        const mootation::Settings s = zdt4_settings(name, 40, 2000, 2);
        int calls = 0;
        const mootation::Result a = mootation::run(s, zdt4_batch());
        const mootation::Result b = mootation::run(s, zdt4_batch(), [&calls](const Vec& x, const Vec& f, const Vec&) {
            ++calls;
            return compass(x, f);
        });
        check(same_run(a, b) && calls == 0 && setting(b, "ls_calls").empty(),
              std::string("(2) ") + name + ": off with a hook is the run without one, the hook not called");
    }
}

// (3)'s coverage floor: IGD+ of pop points of the front itself, f1 evenly spaced
// in [0, 1] — what no answer of pop points does much better than
double coverage_floor(int pop)
{
    Rows F;
    for (int k = 0; k < pop; ++k) {
        const double t = static_cast<double>(k) / (pop - 1);
        F.push_back({t, 1.0 - std::sqrt(t)});
    }
    return igd_plus_zdt(F);
}

// (3): for one seed, a run of 64000 with the observer after every pop_size
// spent (the local search's cost counted, MLS-2): the spending at the first
// answer whose IGD+ is at most eps; -1 where none is
long long spent_to_target(const std::string& algorithm, const std::string& mode, const Well& w, unsigned seed,
                          double eps)
{
    mootation::Settings s = zdt4_settings(algorithm, w.pop, 64000, seed);
    s.set_box(w.n, 0.0, 1.0);
    s.local_search = mode;
    s.ls_every = w.every;
    s.ls_share = w.share;
    const double depth = w.depth;
    mootation::runner::RunProblem problem = mootation::embed_detail::problem_of(
        s, [depth](const Rows& X, Rows& F, Rows&) {
            for (std::size_t i = 0; i < X.size(); ++i) F[i] = zdt1w(X[i], depth);
        });
    mootation::runner::RunConfig cfg = mootation::embed_detail::config_of(s, nullptr);
    if (mode != "off")
        cfg.ls_hook = [depth](const Vec& x, const Vec& f, const Vec&) { return parabolic(x, f, depth); };
    long long reached = -1;
    cfg.record_every = 1;
    cfg.on_generation = [&reached, eps](int, std::vector<std::vector<double>> F) {
        if (reached >= 0 || igd_plus_zdt(F) > eps) return;
        const auto& st = mootation::runner::st();
        reached = st.evaluations.load() + static_cast<long long>(std::ceil(st.ls_cost));
    };
    (void)mootation::runner::run(algorithm, problem, cfg);
    return reached;
}

void test_well(const Well& w)
{
    const double eps = w.times * coverage_floor(w.pop);
    std::printf("\n(3) ZDT1 with a local well, n = %d, depth %g, pop_size %d, ls_every %d, ls_share %g: the spending "
                "at which IGD+ first is <= %g x the coverage floor = %.5f, seeds 1-5, runs of 64000 (-1: never)\n",
                w.n, w.depth, w.pop, w.every, w.share, w.times, eps);
    for (const char* name : {"nsga2", "spea2"}) {
        int wins = 0;
        for (unsigned seed = 1; seed <= 5; ++seed) {
            const long long off = spent_to_target(name, "off", w, seed, eps);
            const long long lam = spent_to_target(name, "lamarck", w, seed, eps);
            const bool win = lam >= 0 && (off < 0 || lam < off);
            wins += win ? 1 : 0;
            std::printf("    %-6s seed %u: off %6lld, lamarck %6lld%s\n", name, seed, off, lam,
                        win ? "  lamarck first" : "");
        }
        std::printf("  %s: lamarck first for %d of 5 seeds\n", name, wins);
        // the check of the task on NSGA-II; SPEA2 measured the same way, for the record
        if (std::string(name) == "nsga2")
            check(wins >= 3, "(3) nsga2: lamarck reaches the target at a smaller spending than off for most seeds (" +
                                 std::to_string(wins) + " of 5)");
    }
}

void test_budget()
{
    for (double c : {1.0, 7.5, 333.0, 5000.0}) {
        for (const char* name : {"nsga2", "moead_de", "spea2"}) {
            const int budget = 4000;
            mootation::Settings s = zdt4_settings(name, 40, budget, 4);
            s.local_search = "lamarck";
            s.ls_share = 1.0;
            const auto evals = std::make_shared<long long>(0);
            long long calls = 0;
            double declared = 0.0;
            bool below = true;
            const mootation::Result r =
                mootation::run(s, zdt4_batch(evals), [&](const Vec& x, const Vec& f, const Vec&) {
                    below = below && *evals + static_cast<long long>(std::ceil(declared)) < budget;
                    ++calls;
                    declared += c;
                    mootation::LocalStep out;
                    out.x = x;                     // nothing moves: the cost alone
                    out.f = f;
                    out.cost = c;
                    return out;
                });
            const long long cap = static_cast<long long>(std::ceil(budget / c));
            const std::string at = std::string("(4) ") + name + ", cost " + mootation::member_search::number(c) + ": ";
            check(below, at + "every call made with the spending below the budget");
            check(calls >= 1 && calls <= cap, at + std::to_string(calls) + " calls, at most ceil(budget / c) = " +
                                                  std::to_string(cap));
            check(setting(r, "ls_calls") == std::to_string(calls) && setting(r, "ls_accepted") == "0",
                  at + "ls_calls in the settings, none accepted (nothing moved)");
            check(setting(r, "ls_cost") == mootation::member_search::number(declared) &&
                      setting(r, "used") == mootation::member_search::number(static_cast<double>(r.evaluations) + declared),
                  at + "ls_cost and used in the settings");
            check(r.evaluations == *evals, at + "the evaluations are the problem's calls");
        }
    }
}

void test_honest()
{
    for (const char* mode : {"lamarck", "baldwin"}) {
        for (const char* pick : {"spread", "random", "objective", "feasible"}) {
            mootation::Settings s = zdt4_settings("nsga2", 40, 6000, 5);
            s.local_search = mode;
            s.ls_pick = pick;
            const mootation::Result r =
                mootation::run(s, zdt4_batch(), [](const Vec& x, const Vec& f, const Vec&) { return compass(x, f); });
            const long long calls = std::atoll(setting(r, "ls_calls").c_str());
            const long long taken = std::atoll(setting(r, "ls_accepted").c_str());
            check(honest(r, zdt4), std::string("(5) ") + mode + ", " + pick + ": every answer row is f(x)");
            check(calls > 0 && taken > 0 && taken <= calls,
                  std::string("(5) ") + mode + ", " + pick + ": " + std::to_string(taken) + " of " +
                      std::to_string(calls) + " calls taken");
        }
    }
}

void test_session()
{
    for (const char* mode : {"lamarck", "baldwin"}) {
        mootation::Settings s = zdt4_settings("nsga2", 40, 3000, 6);
        s.local_search = mode;
        const mootation::Result a =
            mootation::run(s, zdt4_batch(), [](const Vec& x, const Vec& f, const Vec&) { return compass(x, f); });
        mootation::Session session(s);
        const auto eval = zdt4_batch();
        int requests = 0;
        for (auto X = session.ask(); !X.empty();) {
            if (session.asks_local()) {
                ++requests;
                X = session.tell_local(compass(X[0], session.local_objectives()));
            } else {
                Rows F(X.size(), Vec(2)), G;
                eval(X, F, G);
                X = session.tell(F);
            }
        }
        const mootation::Result& b = session.result();
        check(same_run(a, b) && std::to_string(requests) == setting(b, "ls_calls"),
              std::string("(6) ") + mode + ": the Session is the run of run(), bit for bit");
    }
}

void test_threads()
{
    auto one = [](unsigned seed, mootation::Result& out) {
        mootation::Settings s = zdt4_settings("nsga2", 40, 3000, seed);
        s.local_search = seed % 2 ? "lamarck" : "baldwin";
        s.ls_pick = seed % 3 ? "spread" : "random";
        out = mootation::run(s, zdt4_batch(), [](const Vec& x, const Vec& f, const Vec&) { return compass(x, f); });
    };
    std::vector<mootation::Result> alone(4), together(4);
    for (unsigned k = 0; k < 4; ++k) one(10 + k, alone[k]);
    std::vector<std::thread> threads;
    for (unsigned k = 0; k < 4; ++k) threads.emplace_back(one, 10 + k, std::ref(together[k]));
    for (auto& t : threads) t.join();
    bool same = true;
    for (unsigned k = 0; k < 4; ++k) same = same && same_run(alone[k], together[k]);
    check(same, "(7) four runs on four threads are the runs one after the other");
}

void test_bad_hooks()
{
    mootation::Settings s = zdt4_settings("nsga2", 40, 1000, 7);
    s.local_search = "lamarck";
    auto with = [&s](mootation::LocalSearch h) { return [&s, h]() { (void)mootation::run(s, zdt4_batch(), h); }; };
    check(throws_runtime(with([](const Vec& x, const Vec& f, const Vec&) {
              mootation::LocalStep o{x, f, {}, 1.0};
              o.x.pop_back();
              return o;
          }), "MLS-5"), "(8) x' of the wrong size");
    check(throws_runtime(with([](const Vec& x, const Vec& f, const Vec&) {
              mootation::LocalStep o{x, f, {}, 1.0};
              o.f[0] = std::numeric_limits<double>::quiet_NaN();
              return o;
          }), "MLS-5"), "(8) f' not finite");
    check(throws_runtime(with([](const Vec& x, const Vec& f, const Vec&) {
              mootation::LocalStep o{x, f, {}, 1.0};
              o.x[1] = 7.0;
              return o;
          }), "MLS-5"), "(8) x' out of the box");
    check(throws_runtime(with([](const Vec& x, const Vec& f, const Vec&) {
              return mootation::LocalStep{x, f, {}, -1.0};
          }), "MLS-5"), "(8) a negative cost");
    check(throws_runtime(with([](const Vec&, const Vec&, const Vec&) -> mootation::LocalStep {
              throw std::runtime_error("the solver diverged");
          }), "the solver diverged"), "(8) the hook's exception, its message kept");
    bool refused = false;
    try {
        (void)mootation::run(s, zdt4_batch());
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    check(refused, "(8) lamarck without a hook is refused");
}

void test_every_algorithm()
{
    std::printf("\n(9) every algorithm on DTLZ2, pop_size 100 (nearest), budget 3000, ls_share 0.1\n");
    std::printf("%-14s %2s  %-8s %6s %6s %6s  %s\n", "algorithm", "m", "mode", "calls", "kept", "bred", "");
    int runs = 0, failures = 0, refusals = 0;
    for (const std::string& name : mootation::algorithm_names()) {
        for (int m : {2, 3}) {
            for (const char* mode : {"lamarck", "baldwin"}) {
                mootation::Settings s;
                s.algorithm = name;
                s.pop_size = 100;
                s.pop_size_mode = "nearest";
                s.max_evaluations = 3000;
                s.n_objs = m;
                s.seed = 8;
                s.set_box(m + 9, 0.0, 1.0);
                s.local_search = mode;
                const std::size_t first = static_cast<std::size_t>(m) - 1;
                auto at_optimum = [first](const Vec& x) {
                    for (std::size_t j = first; j < x.size(); ++j)
                        if (x[j] != 0.5) return false;
                    return true;
                };
                long long calls = 0, bred = 0;
                auto eval = [m, first, &calls, &bred](const Rows& X, Rows& F, Rows&) {
                    for (std::size_t i = 0; i < X.size(); ++i) {
                        F[i] = dtlz2_of(X[i], m);
                        if (calls == 0) continue;
                        for (std::size_t j = first; j < X[i].size(); ++j)
                            if (X[i][j] == 0.5) { ++bred; break; }
                    }
                };
                auto hook = [m, first, &calls](const Vec& x, const Vec&, const Vec&) {
                    ++calls;
                    mootation::LocalStep o;
                    o.x = x;
                    for (std::size_t j = first; j < o.x.size(); ++j) o.x[j] = 0.5;
                    o.f = dtlz2_of(o.x, m);
                    o.cost = 1.0;
                    return o;
                };
                std::string what = "ok";
                long long kept = 0;
                ++runs;
                const bool to_refuse = mootation::member_search::refusal(name) != nullptr;
                try {
                    const mootation::Result r = mootation::run(s, eval, hook);
                    for (const auto& x : r.variables) kept += at_optimum(x) ? 1 : 0;
                    if (!honest(r, [m](const Vec& x) { return dtlz2_of(x, m); }))
                        what = "FAILED: an answer row is not f(x)";
                    if (setting(r, "ls_calls") != std::to_string(calls)) what = "FAILED: ls_calls";
                    mootation::Settings off = s;
                    off.local_search = "off";
                    const mootation::Result o = mootation::run(off, [m](const Rows& X, Rows& F, Rows&) {
                        for (std::size_t i = 0; i < X.size(); ++i) F[i] = dtlz2_of(X[i], m);
                    });
                    if (o.objectives == r.objectives && o.variables == r.variables)
                        what = "FAILED: the run with off";
                    if (to_refuse) what = "FAILED: not refused";
                } catch (const std::invalid_argument& e) {
                    what = std::string("refused: ") + e.what();
                    ++refusals;
                    if (!to_refuse) what = "FAILED: " + what;
                } catch (const std::exception& e) {
                    what = std::string("FAILED: ") + e.what();
                }
                if (what.rfind("FAILED", 0) == 0) ++failures;
                std::printf("%-14s %2d  %-8s %6lld %6lld %6lld  %s\n", name.c_str(), m, mode, calls, kept, bred,
                            what == "ok" ? "" : what.c_str());
                check(what.rfind("FAILED", 0) != 0, "(9) " + name + ", m = " + std::to_string(m) + ", " + mode +
                                                        ": " + what);
            }
        }
    }
    std::printf("(9) %d runs, %d refused, %d failed\n", runs, refusals, failures);
}

} // namespace

int main(int argc, char** argv)
{
    std::printf("local search of members (task 34.1)\n");
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "--well") {                    // the calibration of (3): --well n pop every share times depth
        Well w;
        if (argc > 2) w.n = std::atoi(argv[2]);
        if (argc > 3) w.pop = std::atoi(argv[3]);
        if (argc > 4) w.every = std::atoi(argv[4]);
        if (argc > 5) w.share = std::atof(argv[5]);
        if (argc > 6) w.times = std::atof(argv[6]);
        if (argc > 7) w.depth = std::atof(argv[7]);
        test_well(w);
        return mootation::testing::report("test_member_search --well");
    }
    if (mode != "--every") {
        test_settings_text();
        test_off();
        test_budget();
        test_honest();
        test_session();
        test_threads();
        test_bad_hooks();
        test_well(Well{});
    }
    test_every_algorithm();
    return mootation::testing::report("test_member_search");
}
