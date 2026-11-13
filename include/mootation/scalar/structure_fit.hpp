#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The parameters of a decoded genome fitted by a local search (task 27, item
// 27.1: "the decoded structure gets its parameters by a local fit with the
// Lamarckian write-back"; 24.1.1 and 24.1.2 of task 24).
//
// set_active_fit(P, inner) gives the problem a local_search hook
// (evaluator.hpp, 24.1.1) for a genome with a decoder (Problem::decode, task
// 24's genome contract; structure_hooks.hpp for trees and grammatical
// evolution): x is decoded, its active genes (the used parameter slots) are
// the free variables of the subproblem hooks.hpp's restricted() makes with
// the other genes frozen at the canonical genome, and the inner solver (a
// solver as a strategy names it, {"solver": name, its knobs ...}) runs on it
// from the genome's own values. The answer is the canonical genome with the
// fitted values in the active genes; a strategy that writes its local
// searches back (memetic with lamarck = on, SRCH-12) puts it into the
// population. The structure's genes are never moved: a builtin local solver
// would move every variable, the structure's slots too.
//
// Two ways to give a structure its parameters (04 Р3, Р5):
//   - this hook: f(x) is the error of the structure with the genome's
//     parameters, the fit is a local search the strategy pays for and writes
//     back; the cache by structure (cache = canon) must stay off then — two
//     genomes of one structure differ in f (structure_hooks.hpp HOOK-6);
//   - a fit inside evaluate (Kommenda et al. 2020: ten LM iterations within
//     the evaluation): f(x) is the structure's error after its fit, one
//     evaluation a structure, and cache = canon spares the repeats; nothing
//     is written back (the parameters are not searched by the GA).
//
// Readings (OURS):
//   FIT-1. The inner run's budget is the hook's allowance (ctx.budget), at
//     most fit_budget evaluations when that is > 0; its generator's seed is
//     drawn from the hook's generator (rnd::below(2^32)), its eq_tol 1e-6
//     (a structure's fit has no equalities in mind; the problem's are kept
//     by restricted()). Every evaluation of the inner run is the strategy's,
//     through ctx.problem (ULS-1).
//   FIT-2. A genome that does not decode, or that has no active gene, comes
//     back canonical (decode's x_canon, or x itself when invalid) with
//     nothing spent.
//   FIT-3. The answer is the inner run's best point embedded into the
//     canonical genome; its f is not declared (it is in the archive, so free:
//     ULS-2).
//   FIT-4. An empty active mask means "every gene active" in task 24's
//     contract (genome.hpp), and the fit then moves every gene — right only
//     for a genome of parameters alone. structure_hooks.hpp's decoders always
//     give the full mask, so their structure's genes are never moved.
// ============================================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../random.hpp"
#include "hooks.hpp"
#include "scalar.hpp"

namespace mootation::scalar::structure {

struct ActiveFit {
    std::string inner = "{\"solver\":\"nelder_mead\"}";   // the solver of the fit
    long long fit_budget = 0;                             // 0: the hook's allowance (FIT-1)
    int pop_size = 20;                                    // the inner solver's plan, where it has one
};

// The problem's local_search set to the fit of the active genes.
inline void set_active_fit(Problem& P, const ActiveFit& a)
{
    if (!P.decode) throw std::invalid_argument("set_active_fit: the problem gives no decode");
    const strategy::Inner in = strategy::inner_of("set_active_fit", Knob{true, a.inner, 0.0});
    P.local_search = [in, a](const std::vector<double>& x, LocalContext& ctx,
                             const std::vector<double>& state) -> LocalOutcome {
        LocalOutcome out;
        out.state = state;
        const Decoded d = ctx.problem.decode(x);
        if (!d.valid) {                                                              // FIT-2
            out.x = x;
            return out;
        }
        std::vector<std::size_t> idx;
        for (std::size_t k = 0; k < d.x_canon.size(); ++k)
            if (d.active.empty() || d.active[k]) idx.push_back(k);                   // FIT-4
        out.x = d.x_canon;
        if (idx.empty() || ctx.budget <= 0) return out;                             // FIT-2
        const Problem sub = restricted(ctx.problem, d.x_canon, idx);
        Start from;
        from.has_x0 = true;
        for (std::size_t k : idx) from.x0.push_back(d.x_canon[k]);
        const long long budget = a.fit_budget > 0 ? std::min(a.fit_budget, ctx.budget) : ctx.budget;
        const auto seed = static_cast<unsigned>(rnd::below(ctx.rng, std::uint64_t{1} << 32));
        const Result r = run(in.name, sub, 1e-6, false, 0.0, a.pop_size, budget, seed, in.knobs, {}, 0, from);
        if (!r.variables.empty()) out.x = embed(d.x_canon, idx, r.variables[0]);    // FIT-3
        return out;
    };
}

}  // namespace mootation::scalar::structure
