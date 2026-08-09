// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Warm start through the embedding layer.
//
// A run that costs money per evaluation — a simulator, a solver, a rig — must
// be able to continue from where a previous one stopped without paying for the
// same points twice. This checks the property that makes that worth doing:
//
//   (1) seeding costs ZERO function evaluations. The objectives come out of the
//       population, not out of the evaluator. If this ever regresses, warm
//       start becomes a way to spend a full population's budget on arithmetic
//       that was already done.
//   (2) the seeded run really starts from the seeded population, rather than
//       from a fresh draw that happens to be near it.
//   (3) a file written by save_population loads back and seeds a run.
//   (4) the size policies do what they say: Error refuses, Truncate shortens,
//       Pad lengthens.
//   (5) a mismatched seed is refused with a message that names the mismatch,
//       not accepted and silently reshaped.
//   (6) a CONSTRAINED run cannot be seeded from a population that carries only
//       the aggregate cv. This is the one that would be silently wrong rather
//       than loudly broken: every constraint would read as satisfied.
//
// What is NOT claimed anywhere: that a resumed run reproduces the uninterrupted
// one. No RNG position and no per-algorithm state are saved. It starts from the
// same place, which is the part that costs money.
// ============================================================================

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "mootation/embed.hpp"
#include "mootation/io/population.hpp"

#include "harness.hpp"

namespace {

int g_failed = 0;
int g_checks = 0;

void check(bool ok, const std::string& what)
{
    ++g_checks;
    if (!ok) {
        ++g_failed;
        std::printf("  FAIL  %s\n", what.c_str());
    }
}

// ZDT1: cheap, two objectives, a front everyone recognises.
void zdt1(const std::vector<std::vector<double>>& X,
          std::vector<std::vector<double>>&       F,
          std::vector<std::vector<double>>&       /*G*/)
{
    for (std::size_t i = 0; i < X.size(); ++i) {
        double g = 0.0;
        for (std::size_t j = 1; j < X[i].size(); ++j) g += X[i][j];
        g = 1.0 + 9.0 * g / static_cast<double>(X[i].size() - 1);
        F[i][0] = X[i][0];
        F[i][1] = g * (1.0 - std::sqrt(F[i][0] / g));
    }
}

// The same, plus one constraint: x0 <= 0.5.
void zdt1_constrained(const std::vector<std::vector<double>>& X,
                      std::vector<std::vector<double>>&       F,
                      std::vector<std::vector<double>>&       G)
{
    zdt1(X, F, G);
    for (std::size_t i = 0; i < X.size(); ++i) G[i][0] = X[i][0] - 0.5;
}

mootation::Settings base_settings(int pop, int gens)
{
    mootation::Settings s;
    s.algorithm = "nsga2";
    s.pop_size  = pop;
    s.max_gen   = gens;
    s.seed      = 11;
    s.n_objs    = 2;
    s.set_box(6, 0.0, 1.0);
    return s;
}

bool same_objectives(const mootation::Result& a, const mootation::Result& b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t k = 0; k < a.objectives[i].size(); ++k)
            if (a.objectives[i][k] != b.objectives[i][k]) return false;
    return true;
}

} // namespace

int main()
{
    using namespace mootation;

    std::printf("warm start through embed\n");

    // ── A run to continue from ──────────────────────────────────────────────
    Settings s = base_settings(20, 15);
    const Result first = run(s, zdt1);
    check(first.size() == 20, "the first run returns pop_size individuals");
    check(first.evaluations > 0, "the first run actually evaluated something");

    // ── (1) and (2): zero generations from a seed = the seed, for free ──────
    {
        Settings cont = base_settings(20, 0);       // no generations at all
        const Result r = run(cont, zdt1, as_population(first));
        check(r.evaluations == 0,
              "seeding costs zero function evaluations (got " +
              std::to_string(r.evaluations) + ")");
        check(same_objectives(r, first),
              "a zero-generation seeded run returns exactly the seed population");
    }

    // ── (3): through a file ─────────────────────────────────────────────────
    // Written by hand rather than by save_population because that one takes a
    // DataVault, which the embedding layer deliberately does not expose. The
    // format is the same one, and load_population is what reads it back.
    {
        const std::string path = "warm_start_pop.csv";
        {
            std::ofstream f(path);
            check(static_cast<bool>(f), "the seed file can be created");
            if (!f) return 1;
            f << "# mootation population v1\n";
            f << "# n_vars=6 n_bin=0 n_objs=2 n_lims=0\n";
            f << "x1,x2,x3,x4,x5,x6,f1,f2,cv\n";
            f << std::setprecision(17);
            for (std::size_t i = 0; i < first.size(); ++i) {
                for (double v : first.variables[i]) f << v << ",";
                f << first.objectives[i][0] << "," << first.objectives[i][1] << ",0\n";
            }
        }

        io::Population loaded = io::load_population(path);
        check(loaded.size() == first.size(), "the file loads back with every row");
        check(loaded.n_vars() == 6 && loaded.n_objs() == 2,
              "the file loads back with the right shape");

        Settings cont = base_settings(20, 0);
        cont.seed_population = path;
        const Result r = run(cont, zdt1);
        check(r.evaluations == 0, "a file-seeded run also costs zero evaluations");
        check(same_objectives(r, first),
              "a file round trip preserves the population exactly");

        std::remove(path.c_str());
    }

    // ── (4): the size policies ──────────────────────────────────────────────
    {
        Settings smaller = base_settings(12, 0);
        smaller.on_size_mismatch = SizeMismatch::Truncate;
        const Result r = run(smaller, zdt1, as_population(first));
        check(r.size() == 12, "truncate resizes 20 -> 12");

        Settings bigger = base_settings(28, 0);
        bigger.on_size_mismatch = SizeMismatch::Pad;
        const Result r2 = run(bigger, zdt1, as_population(first));
        check(r2.size() == 28, "pad resizes 20 -> 28");
        check(r2.evaluations == 0, "padding does not evaluate the added rows");

        Settings strict = base_settings(12, 0);
        strict.on_size_mismatch = SizeMismatch::Error;
        bool threw = false;
        try { run(strict, zdt1, as_population(first)); }
        catch (const std::exception&) { threw = true; }
        check(threw, "error refuses a population of the wrong size");
    }

    // ── (5): a seed from a different problem is named, not reshaped ─────────
    {
        Settings other = base_settings(20, 0);
        other.set_box(9, 0.0, 1.0);          // 9 variables, the seed has 6
        bool threw = false;
        std::string msg;
        try { run(other, zdt1, as_population(first)); }
        catch (const std::exception& e) { threw = true; msg = e.what(); }
        check(threw, "a seed with the wrong number of variables is refused");
        check(msg.find("variables") != std::string::npos,
              "and the message says what did not match");
    }

    // ── (6): a constrained run needs the per-constraint values ──────────────
    {
        Settings c = base_settings(20, 6);
        c.n_cons      = 1;
        c.constraints = ConstraintMode::FEASIBILITY;
        const Result cr = run(c, zdt1_constrained);
        check(cr.limits.size() == cr.size(),
              "a constrained run reports per-constraint values, not just cv");

        // Round trip through a population keeps them, so seeding works.
        Settings cont = base_settings(20, 0);
        cont.n_cons      = 1;
        cont.constraints = ConstraintMode::FEASIBILITY;
        const Result r = run(cont, zdt1_constrained, as_population(cr));
        check(r.evaluations == 0, "a constrained run seeds for free too");
        check(same_objectives(r, cr), "and returns exactly the seed population");

        // Without them it must refuse rather than start with every constraint
        // reading as satisfied.
        io::Population stripped = as_population(cr);
        stripped.limits.clear();
        bool threw = false;
        std::string msg;
        try { run(cont, zdt1_constrained, stripped); }
        catch (const std::exception& e) { threw = true; msg = e.what(); }
        check(threw, "a constrained run refuses a seed with no constraint values");
        check(msg.find("constraint") != std::string::npos,
              "and the message says why");
    }

    // ── Session takes a seed too ────────────────────────────────────────────
    {
        Settings cont = base_settings(20, 3);
        Session sess(cont, as_population(first));
        std::vector<std::vector<double>> F;
        int batches = 0;
        for (auto X = sess.ask(); !X.empty(); X = sess.tell(F)) {
            F.assign(X.size(), std::vector<double>(2, 0.0));
            std::vector<std::vector<double>> G;
            zdt1(X, F, G);
            ++batches;
        }
        const Result r = sess.result();
        check(r.size() == 20, "a seeded Session finishes with pop_size");
        check(batches > 0, "a seeded Session still hands out offspring batches");
        // 3 generations of offspring and nothing for the seeded generation.
        check(r.evaluations > 0 && r.evaluations <= 20 * 3,
              "a seeded Session pays only for the generations it ran (" +
              std::to_string(r.evaluations) + ")");
    }

    std::printf("warm_start: %d/%d checks passed%s\n",
                g_checks - g_failed, g_checks, g_failed ? ", FAILED" : "");
    return g_failed == 0 ? 0 : 1;
}
