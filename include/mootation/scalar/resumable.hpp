// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// THE RESUMABLE RUN of a strategy (task 31, layer 3; the base of 30.16, 30.6
// and replica_ladder's pieces). A strategy that has it can be started, taken
// on by pieces of its budget, asked for its members and handed others, and
// its state between the pieces is kept, never evaluated again: the
// population or the walk's point, the step, the operators' weights, the
// generator, the evaluator (archive, counts) and the stop rule (stop_rule.hpp,
// layer 1). Strategies without it run as they always did.
//
//   start_resumable(name, P, ... the arguments of scalar::run ...)
//       the run, nothing evaluated yet; begin() then makes its first
//       population (or the walk's first local search) from the start: x0,
//       the seed points, as the strategy's one-call run reads them.
//   advance(piece)
//       whole steps — a step the strategy's own unit (stop_rule.hpp: a
//       generation, a step of the walk) — until the piece is used up or the
//       run ends; begin() first where it has not run. Returns whether the
//       run can go on.
//   members()          the current members: the population (cluster_ga: every
//                      island's; ga: its mu parents) or the walk's point.
//   admit(in, how)     members from elsewhere in place of the receiver's.
//   best_points(k)     the k best distinct points the run holds (carry_points).
//   operators(), set_operators(o)
//                      the move's step and the portfolios' weights.
//   best()             the run's best point so far (its archive's answer).
//   summary()          the strategy's own settings rows so far.
//   result()           the end: the answer and the settings, as scalar::run
//                      gives them; the run takes no step after it.
//
// The resumable strategies: resumable_names() (cluster_ga, ga_sbx, ssga,
// basin_hopping). The others, and what each would need: the report of task
// 31, layer 3.
//
// READINGS:
//   RES-1 (only when asked: the resumable API). A run cut into pieces is the
//     run in one piece, bit for bit (its evaluations, its generator's draws,
//     its answer and settings): a piece ends between two steps only, and every
//     step is taken as in the one-call run — a round's allowances still come
//     from what is left of the whole budget, so the step that crosses a
//     piece's spend finishes, and the next piece pays for it less. advance({})
//     runs to the end; scalar::run of a resumable strategy is start_resumable,
//     advance({}), result().
//   RES-2 (only when asked: the resumable API). Piece::spend is in the
//     budget's units (Evaluator::counted(), stop_rule.hpp STOP-1), from the
//     call on; begin()'s spending is the first piece's. Piece::steps counts
//     steps, begin() not one of them. Both 0: to the run's end. The run's own
//     end (budget, target, deadline, stall, max_steps, idle, its rule) ends it
//     as in one piece.
//   RES-3 (only when asked: the resumable API). A member (Member) is x with
//     its values as the giver's archive has them: f, the constraints' g, cv at
//     the giver's tolerance. admit plants it in the receiver's archive
//     (Evaluator::plant: known values, not evaluated, not counted — the giver
//     paid for it; one whose g is not of n_cons values, or any member of a
//     problem with residuals, is evaluated and counted, scalar.hpp SEED-1,
//     SEED-2) and the receiver's cv is its own
//     tolerance's. A member of a problem with another meaning (a level of
//     at_param) is not a member: it is a start point (carry_start).
//   RES-4 (only when asked: the resumable API). admit(in, Place): worst — each
//     member in place of the worst of the members not admitted in this call;
//     worst_if_better — the same, only where it is better (search.hpp SRCH-8:
//     less violating, then lower f; cluster_ga: its step 3 with cga_de, as its
//     islands' migrants); all — the members become in's (as many as the
//     receiver has, the rest kept). A PlaceRule picks the index replaced for
//     each member in turn (-1: none). Returns the members taken. A walk has
//     one member.
//   RES-5 (only when asked: the resumable API). set_budget(b) between pieces:
//     the run's budget is b from then on (Evaluator::budget). What the
//     strategy read from the budget at its start stays (lshade's MAX_NFE,
//     cluster_ga's multi_dynamic T_End); a b below what is spent ends the run
//     at its next step.
//   RES-6 (only when asked: the resumable API). operators(): the move's
//     strength where it adapts (has_step: basin_hopping's bh_step, SRCH-5) and
//     the portfolios (portfolio.hpp, their P, Q and counts; on() false where
//     the strategy has none). set_operators replaces what the receiver has of
//     them and ignores the rest; a portfolio of other operators (names) is
//     refused.
//   RES-7. carry_points = N (the owner's knob of 17.11; off by default, and
//     then the next run gets the one point of before): the next run (a shrink
//     level, an ensemble member, the next strategy) gets the N best points —
//     the run's answer first, then best_points' others. A strategy that takes
//     a population (takes_population: cluster_ga with cga_init = seed_points,
//     ga_sbx and ssga with warm_start = on) gets them as its start
//     (carry_start: x0 the first, seed points the rest, their values unknown;
//     ga_sbx and ssga take the rest as the design's rows, evaluated: POP-7
//     plants seed points as known values); any other is run once from each
//     point, the budget split (split_budget). N = 1 is the run of before, bit
//     for bit.
//   RES-8 (only when asked: the resumable API). The problem: a run keeps a
//     reference to P (Evaluator), so P outlives it; with max_seconds or
//     time_clock in the knobs the run holds its own copy of P with the
//     deadline (scalar.hpp TIME-1), as scalar::run does.
// ============================================================================

#include "scalar.hpp"

// Included by scalar.hpp after Result and Start; the guard (not #pragma once)
// lets a program include this header first.
#ifndef MOOTATION_SCALAR_RESUMABLE_HPP
#define MOOTATION_SCALAR_RESUMABLE_HPP

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "evaluator.hpp"
#include "portfolio.hpp"

namespace mootation::scalar {

// a member handed between runs (RES-3)
struct Member {
    Vec x;
    double f = std::numeric_limits<double>::infinity();
    Vec g;                                            // the constraints' values (empty: none)
    double cv = std::numeric_limits<double>::infinity();
};

// a piece of the run (RES-2): 0 is no limit
struct Piece {
    double spend = 0.0;          // the budget's units, from the call on
    long long steps = 0;         // the strategy's steps
};

// the operators' state (RES-6)
struct Operators {
    bool has_step = false;
    double step = 0.0;           // the move's strength (basin_hopping's bh_step)
    moves::Portfolio mutation;   // the move's portfolio (mutate_ops)
    moves::Portfolio crossover;  // the crossover's (crossover_ops)
};

// where admitted members go (RES-4)
enum class Place { worst, worst_if_better, all };
// the caller's rule: the index of members replaced by in, -1 for none
using PlaceRule = std::function<long long(const std::vector<Member>& members, const Member& in)>;

class Resumable {
public:
    virtual ~Resumable() = default;

    virtual const std::string& name() const = 0;
    virtual void begin() = 0;                                              // the first population (once)
    virtual bool advance(const Piece& piece = Piece()) = 0;               // RES-1, RES-2
    virtual bool done() const = 0;
    virtual long long steps() const = 0;
    virtual const Evaluator& ledger() const = 0;                          // counted(), left(), used(), budget
    virtual void set_budget(long long budget) = 0;                        // RES-5
    virtual std::vector<Member> members() const = 0;
    virtual std::vector<Member> best_points(std::size_t k) const = 0;      // RES-7
    virtual long long admit(const std::vector<Member>& in, Place how) = 0;          // RES-3, RES-4
    virtual long long admit(const std::vector<Member>& in, const PlaceRule& rule) = 0;
    virtual Operators operators() const = 0;                              // RES-6
    virtual void set_operators(const Operators& ops) = 0;
    virtual Member best() const = 0;
    virtual std::vector<std::pair<std::string, std::string>> summary() const = 0;
    virtual Result result() = 0;                                          // the end
};

// the strategies that have it (resumable_run.hpp, at the end of strategy.hpp)
inline const std::vector<std::string>& resumable_names();

// the run of a resumable strategy, nothing evaluated yet: the arguments of
// scalar::run (RES-8); unknown and other names are refused
inline std::unique_ptr<Resumable> start_resumable(const std::string& name, const Problem& P, double eq_tol,
                                                  bool has_tol0, double eq_tol0, int pop_size, long long budget,
                                                  unsigned seed, const Knobs& knobs, Observer observer,
                                                  int record_every, const Start& start);

// x as a member of the run of ev (RES-3): its values from the archive where
// it is there, else f and cv as given and no g
inline Member member_of(const Evaluator& ev, const Vec& x, double f, double cv)
{
    Member m;
    m.x = x;
    m.f = f;
    m.cv = cv;
    if (!ev.known(x)) return m;
    const std::size_t i = ev.find(x);
    m.f = ev.F[i][0];
    m.g = ev.G[i];
    m.cv = ev.cv(i);
    return m;
}

// the run's best point (Resumable::best): the lowest f of the feasible points
// of the archive, else the least violating, the first of equals (THE LEAN
// MODE: the archive's first, its best). The archive is only read
// (Evaluator::answer would build its cache of the front, and the run's
// later steps would then keep it: a run asked for its best between pieces
// stays the run in one piece, RES-1).
inline Member best_member(const Evaluator& ev)
{
    if (ev.X.empty()) return Member();
    std::size_t b = 0;
    if (!ev.lean) {
        bool found = false;
        double bf = 0.0, bc = 0.0;
        for (std::size_t i = 0; i < ev.X.size(); ++i) {
            if (ev.failed[i]) continue;
            const double c = ev.cv(i), f = ev.F[i][0];
            const bool feasible = c <= 0.0;
            const bool b_feasible = found && bc <= 0.0;
            const bool better = !found || (feasible && !b_feasible) ||
                                (feasible && b_feasible && f < bf) || (!feasible && !b_feasible && c < bc);
            if (better) {
                b = i;
                bf = f;
                bc = c;
                found = true;
            }
        }
        if (!found) return Member();
    }
    if (ev.failed[b]) return Member();
    return member_of(ev, ev.X[b], ev.F[b][0], ev.cv(b));
}

// whether the strategy with these knobs starts from many points (RES-7)
inline bool takes_population(const std::string& name, const Knobs& knobs)
{
    auto has = [&](const char* key, const char* word) {
        for (const auto& kv : knobs)
            if (kv.first == key && kv.second.is_word && kv.second.word == word) return true;
        return false;
    };
    if (name == "cluster_ga") return has("cga_init", "seed_points");
    if (name == "ga_sbx" || name == "ssga") return has("warm_start", "on");
    return false;
}

// the start of a run from carried points (RES-7): x0 the first, the others
// seed points whose values are unknown (each a start of the next run)
inline Start carry_start(const std::vector<Vec>& points)
{
    Start st;
    if (points.empty()) return st;
    st.has_x0 = true;
    st.x0 = points[0];
    if (points.size() > 1) {
        st.has_seed_points = true;
        for (std::size_t q = 1; q < points.size(); ++q) st.seed_points.push_back(Point{points[q], {}, {}});
    }
    return st;
}

// a budget split among parts runs (RES-7), the first ones a unit more where
// it does not divide
inline std::vector<long long> split_budget(long long budget, std::size_t parts)
{
    std::vector<long long> b(parts, 0);
    if (parts == 0) return b;
    const long long n = static_cast<long long>(parts), each = budget / n, rest = budget % n;
    for (std::size_t q = 0; q < parts; ++q) b[q] = each + (static_cast<long long>(q) < rest ? 1 : 0);
    return b;
}

}   // namespace mootation::scalar

#endif   // MOOTATION_SCALAR_RESUMABLE_HPP
