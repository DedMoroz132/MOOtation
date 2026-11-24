// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// pop_size_mode = exact | nearest (task 34.2; run_core.hpp, RunConfig::
// pop_size_mode, POP-1, POP-2; settings.hpp).
//
//   (1) the settings text: exact is the default and is not written, nearest is
//       written and read back, any other word is refused;
//   (2) nearest_pop_size: the size taken is accepted, none accepted is nearer,
//       and of two equally near the smaller is taken (POP-1) — against every
//       n up to 300 for the Das-Dennis lattices at 2, 3, 5 and 8 objectives
//       and for M2M's multiples of K;
//   (3) NSGA-III at 3 and 5 objectives and MOEA/D-M2M at pop_size 105: exact
//       refuses the size, nearest runs with the size it takes, which is in
//       the result and in its settings (pop_size_used); the evaluation budget
//       stays as given; a size the core takes runs as with exact, bit for bit;
//       a Session (ask / tell) the same; a seed population of the other size
//       is refused (POP-2);
//   (4) every algorithm of the dispatch on DTLZ2 at pop_size 100 and 2, 3 and
//       5 objectives with nearest: none fails (the list printed: what exact
//       does and the size nearest took).
// ============================================================================

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mootation/das_dennis.hpp"
#include "mootation/embed.hpp"

#include "harness.hpp"

namespace {

using mootation::testing::check;

// DTLZ2 with m objectives over m + 9 variables in [0, 1] (Deb et al. 2005).
mootation::Evaluate dtlz2(int m)
{
    return [m](const std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& F,
               std::vector<std::vector<double>>&) {
        const double half_pi = 1.5707963267948966;
        for (std::size_t i = 0; i < X.size(); ++i) {
            const auto& x = X[i];
            double g = 0.0;
            for (std::size_t j = static_cast<std::size_t>(m) - 1; j < x.size(); ++j)
                g += (x[j] - 0.5) * (x[j] - 0.5);
            for (int k = 0; k < m; ++k) {
                double v = 1.0 + g;
                for (int j = 0; j < m - 1 - k; ++j) v *= std::cos(x[static_cast<std::size_t>(j)] * half_pi);
                if (k > 0) v *= std::sin(x[static_cast<std::size_t>(m - 1 - k)] * half_pi);
                F[i][static_cast<std::size_t>(k)] = v;
            }
        }
    };
}

mootation::Settings settings_of(const std::string& algorithm, int m, int pop, const std::string& mode,
                                int budget)
{
    mootation::Settings s;
    s.algorithm = algorithm;
    s.pop_size = pop;
    s.pop_size_mode = mode;
    s.max_evaluations = budget;
    s.n_objs = m;
    s.seed = 3;
    s.set_box(m + 9, 0.0, 1.0);
    return s;
}

std::string setting(const mootation::Result& r, const std::string& key)
{
    for (const auto& kv : r.settings)
        if (kv.first == key) return kv.second;
    return "";
}

bool refused(const mootation::Settings& s)
{
    try {
        (void)mootation::run(s, dtlz2(s.n_objs));
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

// (2): the property of POP-1 for one predicate
template <typename Takes>
bool nearest_is_nearest(int n, const Takes& takes, std::string& why)
{
    const int r = mootation::runner::nearest_pop_size(n, takes);
    const int d = r > n ? r - n : n - r;
    if (!takes(r)) { why = "takes " + std::to_string(r) + ", which is not accepted"; return false; }
    for (int e = 0; e < d; ++e)
        if ((n - e >= 2 && takes(n - e)) || takes(n + e)) {
            why = "takes " + std::to_string(r) + ", but one at distance " + std::to_string(e) + " is accepted";
            return false;
        }
    if (r > n && n - d >= 2 && takes(n - d)) {
        why = "takes " + std::to_string(r) + ", but " + std::to_string(n - d) + " is as near and smaller";
        return false;
    }
    return true;
}

void test_settings_text()
{
    using mootation::Settings;
    Settings s = settings_of("nsga3", 3, 92, "exact", 0);
    check(s.to_string().find("pop_size_mode") == std::string::npos,
          "(1) exact, the default, is not written: the text of an exact run is as it was");
    s.pop_size_mode = "nearest";
    const Settings back = Settings::from_string(s.to_string());
    check(back.pop_size_mode == "nearest", "(1) nearest is written and read back");
    check(Settings::from_string("pop_size = 10\nlower = 0\nupper = 1\n").pop_size_mode == "exact",
          "(1) not given: exact");
    bool threw = false;
    try {
        (void)Settings::from_string("pop_size_mode = closest\nlower = 0\nupper = 1\n");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "(1) pop_size_mode = closest is refused");
}

void test_nearest()
{
    std::string why;
    bool ok = true;
    for (int m : {2, 3, 5, 8}) {
        auto lattice = [m](int n) { return mootation::das_dennis::is_attainable(m, n); };
        for (int n = 2; n <= 300 && ok; ++n)
            if (!nearest_is_nearest(n, lattice, why)) {
                check(false, "(2) lattice at m = " + std::to_string(m) + ", n = " + std::to_string(n) + ": " + why);
                ok = false;
            }
    }
    for (int K : {3, 10, 17}) {
        auto m2m = [K](int n) { return n % K == 0 && n / K >= 2; };
        for (int n = 2; n <= 300 && ok; ++n)
            if (!nearest_is_nearest(n, m2m, why)) {
                check(false, "(2) M2M with K = " + std::to_string(K) + ", n = " + std::to_string(n) + ": " + why);
                ok = false;
            }
    }
    check(ok, "(2) the size taken is the nearest accepted, the smaller of two equally near");
    auto m5 = [](int n) { return mootation::das_dennis::is_attainable(5, n); };
    check(mootation::runner::nearest_pop_size(100, m5) == 105, "(2) m = 5, 100: 105 (35 + 70)");
    auto k10 = [](int n) { return n % 10 == 0 && n / 10 >= 2; };
    check(mootation::runner::nearest_pop_size(105, k10) == 100, "(2) K = 10, 105: 100 and 110 tie, 100");
    check(mootation::runner::nearest_pop_size(7, [](int) { return false; }) == 7,
          "(2) none accepted: the size asked, for the core's own error");
}

// (3) one algorithm at one size: exact refuses, nearest takes `want`
void test_one(const std::string& algorithm, int m, int pop, int want)
{
    const std::string at = "(3) " + algorithm + ", m = " + std::to_string(m) + ", pop_size " +
                           std::to_string(pop) + ": ";
    const int budget = 4 * pop;
    check(refused(settings_of(algorithm, m, pop, "exact", budget)), at + "exact refuses it");
    const mootation::Result r = mootation::run(settings_of(algorithm, m, pop, "nearest", budget), dtlz2(m));
    check(r.pop_size_used == want, at + "nearest takes " + std::to_string(want) + " (took " +
                                       std::to_string(r.pop_size_used) + ")");
    check(setting(r, "pop_size_used") == std::to_string(want), at + "pop_size_used in the run's settings");
    check(static_cast<int>(r.size()) == want, at + "the answer has the size taken");
    check(r.evaluations >= budget && r.evaluations < budget + want,
          at + "the budget as given (" + std::to_string(r.evaluations) + " evaluations of " +
              std::to_string(budget) + ")");

    // the size taken asked for: nearest = exact, bit for bit
    const mootation::Result a = mootation::run(settings_of(algorithm, m, want, "exact", budget), dtlz2(m));
    const mootation::Result b = mootation::run(settings_of(algorithm, m, want, "nearest", budget), dtlz2(m));
    check(a.objectives == b.objectives && a.variables == b.variables && a.evaluations == b.evaluations,
          at + "at the size taken, nearest runs as exact, bit for bit");
    check(a.objectives == r.objectives && a.variables == r.variables,
          at + "and so does the run that asked for " + std::to_string(pop));
    check(a.pop_size_used == want && a.settings.empty() && setting(b, "pop_size_used") == std::to_string(want),
          at + "exact has no pop_size_used entry, nearest has it");
}

void test_session_and_seed()
{
    // ask / tell: the size taken in the result, the batches of the size taken
    mootation::Settings s = settings_of("nsga3", 3, 92, "nearest", 0);
    s.max_gen = 3;
    mootation::Session session(s);
    const auto eval = dtlz2(3);
    bool batches_ok = true;
    for (auto X = session.ask(); !X.empty();) {
        batches_ok = batches_ok && X.size() == 91;
        std::vector<std::vector<double>> F(X.size(), std::vector<double>(3)), G;
        eval(X, F, G);
        X = session.tell(F);
    }
    const mootation::Result& r = session.result();
    check(batches_ok, "(3) Session, nsga3 at 92: every batch of 91");
    check(r.pop_size_used == 91 && setting(r, "pop_size_used") == "91" && r.size() == 91,
          "(3) Session, nsga3 at 92: pop_size_used 91");

    // POP-2: a seed population of the size asked, not of the size taken
    mootation::Settings first = settings_of("nsga2", 3, 92, "exact", 0);
    first.max_gen = 1;
    const mootation::Result pop92 = mootation::run(first, dtlz2(3));
    bool threw = false;
    std::string message;
    try {
        mootation::Settings t = settings_of("nsga3", 3, 92, "nearest", 0);
        t.max_gen = 1;
        (void)mootation::run(t, dtlz2(3), mootation::as_population(pop92));
    } catch (const std::invalid_argument& e) {
        threw = true;
        message = e.what();
    }
    check(threw && message.find("91") != std::string::npos,
          "(3) POP-2: a seed of 92 rows for a run that takes 91 is refused, the message names 91");
}

// (4) every algorithm at pop_size 100 and m = 2, 3, 5, nearest
void test_every_algorithm()
{
    std::printf("\n(4) every algorithm on DTLZ2, pop_size 100, nearest (exact: what it does)\n");
    std::printf("%-14s %3s  %-6s %-6s  %s\n", "algorithm", "m", "exact", "used", "nearest");
    int runs = 0, failures = 0, moved = 0;
    for (const std::string& name : mootation::algorithm_names()) {
        for (int m : {2, 3, 5}) {
            const int budget = 300;
            std::string exact = "ok";
            try {
                (void)mootation::run(settings_of(name, m, 100, "exact", budget), dtlz2(m));
            } catch (const std::exception&) {
                exact = "error";
            }
            std::string what = "ok";
            int used = 0;
            try {
                const mootation::Result r = mootation::run(settings_of(name, m, 100, "nearest", budget), dtlz2(m));
                used = r.pop_size_used;
                if (setting(r, "pop_size_used") != std::to_string(used)) what = "no pop_size_used in settings";
            } catch (const std::exception& e) {
                what = std::string("FAILED: ") + e.what();
            }
            ++runs;
            if (used != 100) ++moved;
            if (what != "ok") ++failures;
            if (exact != "ok" && used == 100) what += " (exact refused 100, nearest kept it)";
            std::printf("%-14s %3d  %-6s %-6d  %s\n", name.c_str(), m, exact.c_str(), used, what.c_str());
            check(what == "ok", "(4) " + name + ", m = " + std::to_string(m) + ": " + what);
        }
    }
    std::printf("(4) %d runs, %d with another size, %d failed\n", runs, moved, failures);
}

} // namespace

int main()
{
    std::printf("pop_size_mode (task 34.2)\n");
    test_settings_text();
    test_nearest();
    test_one("nsga3", 3, 92, 91);
    test_one("nsga3", 5, 100, 105);
    test_one("moead_m2m", 2, 105, 100);
    test_one("moead_m2m", 3, 105, 100);
    test_session_and_seed();
    test_every_algorithm();
    return mootation::testing::report("test_pop_size_mode");
}
