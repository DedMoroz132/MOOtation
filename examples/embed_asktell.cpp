// SPDX-License-Identifier: Apache-2.0
//
// Embedding MOOtation in a program that already has its own evaluation loop.
//
// Two shapes, same Settings:
//   1. Session — you own the loop. ask() gives you candidates, tell() takes
//      objective values back and returns the next candidates. Use this when
//      the objectives come from a simulator, a solver process, a rig, or
//      another language.
//   2. run() — the optimizer owns the loop and calls your function.
//
// The "simulator" here is ZDT1 evaluated by hand, deliberately outside the
// library: nothing below ever tells MOOtation what the problem is.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "mootation/embed.hpp"

namespace {

// Pretend this is a solver in another process. It only ever sees a vector of
// decision variables and returns a vector of objective values.
std::vector<double> simulate(const std::vector<double>& x) {
    double g = 0.0;
    for (std::size_t i = 1; i < x.size(); ++i) g += x[i];
    g = 1.0 + 9.0 * g / static_cast<double>(x.size() - 1);
    const double f1 = x[0];
    const double f2 = g * (1.0 - std::sqrt(f1 / g));
    return {f1, f2};
}

double best_f1_plus_f2(const mootation::Result& r) {
    double best = 1e300;
    for (const auto& f : r.objectives) best = std::min(best, f[0] + f[1]);
    return best;
}

void require(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAILED: %s\n", what.c_str());
        std::exit(1);
    }
}

} // namespace

int main() {
    // ── Settings: a plain struct. Equivalent to loading the file below. ──────
    mootation::Settings s;
    s.algorithm = "nsga2";
    s.pop_size  = 40;
    s.max_gen   = 60;
    s.seed      = 7;
    s.n_objs    = 2;
    s.set_box(/*n_vars=*/10, /*lower=*/0.0, /*upper=*/1.0);

    // The same thing as text, so a caller can keep it in a config file.
    const std::string cfg_text =
        "algorithm = nsga2\n"
        "pop_size  = 40\n"
        "max_gen   = 60\n"
        "seed      = 7\n"
        "n_vars    = 10\n"
        "n_objs    = 2\n"
        "lower     = 0\n"
        "upper     = 1\n";
    const mootation::Settings from_text =
        mootation::Settings::from_string(cfg_text, "embedded.cfg");
    require(from_text.algorithm == s.algorithm && from_text.pop_size == s.pop_size &&
            from_text.n_vars() == s.n_vars() && from_text.seed == s.seed,
            "the text config and the struct describe the same run");

    // ── 1. You own the loop ─────────────────────────────────────────────────
    mootation::Result via_session;
    {
        mootation::Session sess(s);

        std::vector<std::vector<double>> F;
        std::size_t batches = 0, evaluated = 0;

        for (auto X = sess.ask(); !X.empty(); X = sess.tell(F)) {
            // The batch size is the ALGORITHM's choice, never assume pop_size:
            // generational algorithms hand over a whole offspring generation,
            // steady-state ones hand over one candidate at a time.
            F.assign(X.size(), std::vector<double>(2, 0.0));
            for (std::size_t i = 0; i < X.size(); ++i) F[i] = simulate(X[i]);
            ++batches;
            evaluated += X.size();
        }
        via_session = sess.result();

        std::printf("session : %zu batches, %zu evaluations, %d generations, "
                    "%zu solutions\n",
                    batches, evaluated, via_session.generations,
                    via_session.size());
        require(batches > 0, "the session handed out at least one batch");
        require(via_session.generations == s.max_gen, "all generations ran");
        require(via_session.size() == static_cast<std::size_t>(s.pop_size),
                "the answer set is pop_size individuals");
    }

    // ── 2. The optimizer owns the loop ──────────────────────────────────────
    mootation::Result via_run = mootation::run(
        s,
        [](const std::vector<std::vector<double>>& X,
           std::vector<std::vector<double>>&       F,
           std::vector<std::vector<double>>& /*G*/) {
            for (std::size_t i = 0; i < X.size(); ++i) F[i] = simulate(X[i]);
        });

    std::printf("run     : %d evaluations, %d generations, %zu solutions\n",
                via_run.evaluations, via_run.generations, via_run.size());

    // Both shapes drive the same algorithm with the same seed over the same
    // problem, so they must agree exactly — that is the point of the pair.
    require(via_run.size() == via_session.size(), "both shapes return the same count");
    for (std::size_t i = 0; i < via_run.size(); ++i)
        for (std::size_t j = 0; j < 2; ++j)
            require(via_run.objectives[i][j] == via_session.objectives[i][j],
                    "Session and run() agree objective by objective");

    // Sanity check against the analytic front. ZDT1 is f2 = 1 - sqrt(f1), so
    // f1 + f2 = f1 + 1 - sqrt(f1), which is minimized at f1 = 1/4 with the
    // value 3/4. Nothing may go below that; a converged run sits just above it.
    const double kFrontMin = 0.75;
    const double best = best_f1_plus_f2(via_run);
    std::printf("best f1+f2 = %.6f (analytic minimum on the front is %.2f)\n",
                best, kFrontMin);
    require(best >= kFrontMin - 1e-9, "no solution beats the true front");
    require(best < kFrontMin + 0.05, "the run actually converged");

    // ── Switching algorithms is one word ────────────────────────────────────
    // The numbers below are not a comparison: 40 individuals for 60
    // generations is a smoke test, not a budget. MOEA/D-DE in particular is a
    // decomposition method its authors ran at N = 300 for 300k evaluations, so
    // it has barely started here. Each is only asserted to stay on the
    // feasible side of the analytic front.
    for (const char* name : {"spea2", "moead_de", "ibea_eplus"}) {
        mootation::Settings t = s;
        t.algorithm = name;
        if (std::string(name) == "moead_de") {
            // MOEA/D needs its population to be an attainable Das-Dennis
            // lattice size; at m = 2 every size is attainable, so 40 is fine.
            t.params["T"] = 8;
        }
        mootation::Result r = mootation::run(
            t, [](const std::vector<std::vector<double>>& X,
                  std::vector<std::vector<double>>&       F,
                  std::vector<std::vector<double>>&) {
                for (std::size_t i = 0; i < X.size(); ++i) F[i] = simulate(X[i]);
            });
        std::printf("%-11s: %zu solutions, best f1+f2 = %.6f", name, r.size(),
                    best_f1_plus_f2(r));
        require(best_f1_plus_f2(r) >= kFrontMin - 1e-9,
                std::string(name) + " stayed on the feasible side of the front");
        if (!r.ignored.empty()) {
            std::printf(", ignored knobs:");
            for (const auto& k : r.ignored) std::printf(" %s", k.c_str());
        }
        std::printf("\n");
        require(r.size() > 0, std::string(name) + " returned a population");
    }

    // A knob the algorithm does not have is reported, not silently dropped.
    {
        mootation::Settings t = s;
        t.algorithm       = "nsga2";
        t.params["kappa"] = 0.05;      // an IBEA knob; NSGA-II has no such thing
        mootation::Result r = mootation::run(
            t, [](const std::vector<std::vector<double>>& X,
                  std::vector<std::vector<double>>&       F,
                  std::vector<std::vector<double>>&) {
                for (std::size_t i = 0; i < X.size(); ++i) F[i] = simulate(X[i]);
            });
        require(r.ignored.size() == 1 && r.ignored[0] == "kappa",
                "an inapplicable knob is reported in Result::ignored");
    }

    std::printf("OK\n");
    return 0;
}
