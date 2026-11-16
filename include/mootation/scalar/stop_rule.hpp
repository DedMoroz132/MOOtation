#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// THE STOP RULE AND THE LEDGER of the strategies (task 31, layer 1; items
// 30.10, 30.12, 30.13 of task 30, 30.1 made general). Included by
// strategy.hpp.
//
// THE LEDGER. Whatever a strategy's run spends goes through its Evaluator
// (evaluator.hpp THE COUNTS): the problem's evaluations, its gradient calls
// (grad_cost each), the declared costs of the hooks (local_search's cost,
// success_cost, delta_cost), and the evaluations of runs on a derived problem
// that evaluates outside the archive (STOP-6). A strategy whose runs have
// evaluators of their own (shrink's levels, replica_ladder's pieces,
// reformulation's searches) adds each run's result to a Ledger instead.
// Every strategy checks its budget by Evaluator::left() (or Ledger::left()),
// nowhere by the evaluations alone.
//
// THE STOP RULE, the same for every strategy: before each step the strategy
// asks StopRule::go, after it tells StopRule::step. A step is the strategy's
// own unit: a generation (cluster_ga, pbh, mde, ma_chains), a step of the
// walk (ils, basin_hopping), an iteration (mlsl), a round (halving,
// multistart's one), a phase (memetic), a cycle (coevolution), a move (lns),
// a start (tiktak), a stage (data_continuation), a run of CMA-ES (ipop,
// bipop, bipop_2009), a level (shrink), a round of pieces (replica_ladder),
// a search (reformulation), a generation of the portfolio (pap). The run
// ends at the first of:
//   budget    nothing left (STOP-1, STOP-2);
//   target    the target reached (evaluator.hpp THE TARGET, THE SUCCESS);
//   deadline  max_seconds passed on the run's clock (clocks.hpp);
//   stall     stall steps in a row without a better best (STOP-3);
//   steps     max_steps steps;
//   the strategy's own end: idle (100 steps in a row that spend nothing,
//             SRCH-10, kept as the guard against a loop that spends
//             nothing), rule (its own criteria: cga_maxgen, its rounds or
//             stages done, the CMA-ES runs' criteria, shrink's rule).
//
// Knobs of every strategy: stall (0: off), max_steps (0: off), budget_units
// (budget | evaluations); of scalar::run: time_clock (wall | unbiased | cpu).
//
// READINGS:
//   STOP-1. A strategy's budget is checked by Evaluator::left(): the budget
//     less counted(), rounded up — the evaluations, the gradient calls at
//     grad_cost each (STOP-2), the hooks' declared costs and the derived
//     runs' evaluations: what the budget is spent by (evaluator.hpp used()).
//     Before, its loops looked at the evaluations alone, so a local_search
//     hook that declares its cost and calls nothing through the evaluator
//     never ended a run (30.10), and with a gradient a strategy of several
//     rounds spent past its budget in used()'s units from its second round
//     on. Without gradients, hooks and derived runs the count is the
//     evaluations, bit for bit.
//   STOP-2. budget_units = budget | evaluations (budget by default since the
//     owner's choice of 17.11, method version 6; evaluations: the runs
//     before, the hooks' declared costs still counted). budget: the gradient
//     calls are in what a strategy's loops check, at grad_cost each, as
//     shrink_count = budget (PARAM-8);
//     evaluations: they are not — the count of before, so a run with a
//     gradient is the run of method version 5 bit for bit; the hooks'
//     declared costs count either way (STOP-1, the fix of 30.10). A round's
//     starts count their gradients against their allowances either way
//     (STRAT-6). shrink keeps its own shrink_count.
//   STOP-3. Off by default (stall = 0). A better best is a new record of
//     the run's best point (improvements.hpp: feasible first, then f); a
//     strategy without an evaluator of its own (shrink, replica_ladder,
//     reformulation) looks at its runs' best f, feasible first. stall
//     counts steps, whatever they spent; cluster_ga's cga_stall of 30.13 is
//     stall with a generation as the step.
//   STOP-4. Off by default (where none of stall, max_steps, budget_units,
//     max_seconds, time_clock is given and no hook declared a cost: then the
//     settings are as before). The settings say stopped_by — the
//     evaluator's end (target, budget, deadline: the call that found it),
//     else the rule's (budget, deadline, stall, steps: the check before a
//     step), else the strategy's own (idle, rule) — and spend: {evaluations,
//     grad_calls, hook_cost, success_spent, delta_spent, outside, used}.
//   STOP-5. Off by default (with a hook's declared cost only). A round's
//     merge stops at the start whose merge leaves nothing of the budget: the
//     starts after it are not merged (their points, counts and costs not in
//     the run; their results empty), as a start after the target's (STRAT-
//     16); 1 thread and N give the same run. On one thread a start is not
//     run at all where the declared costs of the starts before it (merged
//     exactly, without the archive's discount) already spend the budget.
//     Without declared costs every start's spending stays within its
//     allowance and the allowances within what is left, so the merge never
//     stops early.
//   STOP-6. Off by default (a derived problem only). A problem made from the
//     run's for a run of its own (evaluator.hpp derived: a level of
//     at_param, a formulation, the user's restrict_to) keeps the run's
//     deadline and clock; its run's evaluations and gradient calls go to the
//     ledger: the user's subproblem of coevolution and lns evaluates outside
//     the run's archive and was not counted at all before.
// ============================================================================

#include <cmath>
#include <cstddef>
#include <string>

#include "evaluator.hpp"

namespace mootation::scalar {

namespace strategy {

class StopRule {
public:
    long long stall = 0;              // STOP-3: steps in a row without a better best (0: off)
    long long max_steps = 0;          // steps that end the run (0: off)
    bool asked = false;               // a knob of the rule given (STOP-4)

    // before a step: whether the strategy may take it
    bool go(const Evaluator& ev) { return go(ev.left(), ev.P); }

    bool go(long long left, const Problem& P)
    {
        if (left <= 0) return no("budget");
        if (past_deadline(P)) return no("deadline");
        if (stall > 0 && still_ >= stall) return no("stall");
        if (max_steps > 0 && steps_ >= max_steps) return no("steps");
        return true;
    }

    // after a step: whether the run's best got better in it (STOP-3)
    void step(bool improved)
    {
        ++steps_;
        still_ = improved ? 0 : still_ + 1;
    }

    // ... told by the evaluator's records of the best
    void step(const Evaluator& ev)
    {
        const std::size_t r = ev.improvements.records().size();
        step(r != records_);
        records_ = r;
    }

    // the records the run starts the steps with (the first population's)
    void mark(const Evaluator& ev) { records_ = ev.improvements.records().size(); }

    long long steps() const { return steps_; }

    // what ended the run (STOP-4): the evaluator's end, the rule's, else the strategy's own
    std::string stopped_by(const Evaluator& ev, const std::string& own) const
    {
        if (ev.target_reached) return "target";
        if (!ev.ended.empty()) return ev.ended;
        if (!why_.empty()) return why_;
        if (ev.left() <= 0) return "budget";          // a loop that left by its own test on a round cut short
        if (past_deadline(ev.P)) return "deadline";
        return own;
    }

    std::string stopped_by(const std::string& own) const { return why_.empty() ? own : why_; }

private:
    long long steps_ = 0, still_ = 0;
    std::size_t records_ = 0;
    std::string why_;

    bool no(const char* why)
    {
        why_ = why;
        return false;
    }
};

// THE LEDGER of a strategy whose runs have evaluators of their own: every
// run's counts and costs, and what its budget is checked by (STOP-1, STOP-2)
struct Ledger {
    long long evaluations = 0, grad_calls = 0, hook_calls = 0, success_calls = 0, delta_calls = 0;
    double grad_spent = 0.0, hook_spent = 0.0, success_spent = 0.0, delta_spent = 0.0, hook_seconds = 0.0;
    bool budget_grads = false;

    // a run's result, its gradient calls at grad_cost each
    void add(const Result& r, double grad_cost)
    {
        evaluations += r.evaluations;
        grad_calls += r.grad_calls;
        grad_spent += grad_cost * static_cast<double>(r.grad_calls);
        hook_calls += r.hook_calls;
        hook_spent += r.hook_spent;
        hook_seconds += r.hook_seconds;
        success_calls += r.success_calls;
        success_spent += r.success_spent;
        delta_calls += r.delta_calls;
        delta_spent += r.delta_spent;
    }

    double counted() const
    {
        return static_cast<double>(evaluations) + (budget_grads ? grad_spent : 0.0) + hook_spent + success_spent +
               delta_spent;
    }

    long long left(long long budget) const { return budget - static_cast<long long>(std::ceil(counted())); }

    // the run's result gets the ledger's counts
    void into(Result& out) const
    {
        out.evaluations = evaluations;
        out.grad_calls = grad_calls;
        out.hook_calls = hook_calls;
        out.hook_spent = hook_spent;
        out.hook_seconds = hook_seconds;
        out.success_calls = success_calls;
        out.success_spent = success_spent;
        out.delta_calls = delta_calls;
        out.delta_spent = delta_spent;
        out.used = static_cast<double>(evaluations) + grad_spent + hook_spent + success_spent + delta_spent;
    }
};

}   // namespace strategy

}   // namespace mootation::scalar
