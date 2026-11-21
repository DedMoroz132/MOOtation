#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A FAMILY OF PROBLEMS: the bank of the best solutions and the transfer
// between instances (task 32, layer 2). Many problems are not one problem
// but a family of instances over a whole parameter n (a size, a number of
// objects, a level) with one objective of the form sum_n w(n) f(n), and the
// best solutions of neighbouring instances are related: one for n is made
// into a solution for n - 1 or n + 1 (a part taken off or added, cut,
// extended). This header holds the campaign over such a family; the runs
// themselves are scalar::run's (any strategy: a resumable one, layer 1's
// portfolio_run ...). Nothing outside it changes: without a family every run
// is the run of before.
//
// THE FAMILY (Family): the instances, a set of whole n; problem_at(n), the
// problem of instance n (its own number of variables, if it likes; a
// problem with at_param gives it by levels_of); the weight w(n) (a hook, or a
// table by table_of; none: 1); and three hooks, each optional:
//   reference(n)   the value instance n is measured against (pick = gap);
//   certify(n, x)  {ok, f_certified, cost}: the strict check from outside
//                  (interval arithmetic, exact feasibility); without it the
//                  bank takes the f of the run (or of the evaluation);
//   transfer(n, x) [(n_to, x'), ...]: candidates for other instances made
//                  from instance n's point.
//
// THE BANK (Bank): one record per instance, the best one found — the point,
// its f and cv, where it came from (run, transfer, given), the piece and the
// campaign's used() when it came — and the campaign's state: what it has
// spent, its pieces, when each instance was last given a piece, the cursor
// of round_robin, a closure not finished, the counts of every path, the
// history of the total objective over used(). It is written to a file and
// read back (save_bank, load_bank): a campaign goes on from there.
//
// THE CAMPAIGN (run): pieces of the budget, each a run of the solver on one
// instance from its record, until the budget, max_pieces, or nothing to do:
//   1. a closure not finished goes first (the file's, or that of records
//      given);
//   2. pick the instance (pick = round_robin | gap | gap_age | gap_utility;
//      with fresh_first, the records not yet worked on first: FAM-13);
//   3. the piece: scalar::run(solver, problem_at(n), piece budget, the
//      piece's seed, knobs, start x0 = the bank's point of n);
//   4. its answer is offered to the bank; if the bank takes it, the closure
//      from it: transfer from every record that got better, the candidates
//      certified (or evaluated) and offered, round by round, until nothing
//      gets better or closure_rounds rounds;
//   5. the bank to its file (every save_every pieces, and at the end).
// With diminish_ratio or diminish_abs (off by default) the campaign also ends
// when a round of pieces gains too little in the total objective (FAM-11).
//
// THE FILE (owner's decision 2 of task 32, the developer's choice; the
// argument in the task's report): one JSON text (UTF-8, "\n" line ends,
// numbers as Python's repr: the shortest digits that read back to the same
// double, json.hpp), written whole to <file>.tmp and moved over the file, so
// that a reader sees the old bank or the new one, never half of one. Any
// script reads it with json.load; family_store.hpp exports it into a run
// store.
//
// READINGS:
//   FAM-1. The instances are distinct whole numbers, kept in ascending order
//     (round_robin goes up them, the ties of the picks are broken toward the
//     lower n, the bank and the total go up them). A hook's value is asked
//     once per instance and call of run: the problem, the weight and the
//     reference are kept for the call. A weight or reference not finite is
//     refused, as a table without the instance.
//   FAM-2. A record is better than another when it is less violating, then
//     when its f is lower; an equal one is not taken (the bank keeps the
//     first). A failed evaluation is never taken. The total objective is
//     sum w(n) f(n) over the instances that have a record, added up in
//     ascending n; the settings say how many have none (family_missing).
//   FAM-3. certify(n, x): ok false rejects the point (family_rejected, the
//     bank unchanged); ok true gives the record f = f_certified and cv = 0
//     (a certified point is feasible). The hook's declared cost goes to
//     used(). With certify, a transfer's candidate is not evaluated: the
//     certificate is its value.
//   FAM-4. Without certify a candidate of the transfer (or a record given)
//     is evaluated on the problem of its instance by an Evaluator of its own
//     (the campaign's eq_tol; the budget the campaign's, checked before):
//     f, cv and the evaluation's cost in used(); a failed evaluation, or one
//     the problem's deadline refused, is offered and not taken. A candidate for an instance outside the family, or whose point
//     is not of that problem's length or not within its box, is skipped
//     (family_transfer_skipped); the hook's candidates are taken in its
//     order.
//   FAM-5. THE CLOSURE. Round 1 transfers from the records that just got
//     better (a run's piece: its instance; records given: all of them; in
//     ascending n), each from its record as it is when its turn comes;
//     round k + 1 from the records that got better in round k (ascending n,
//     each once). It ends when a round makes nothing better, or after
//     closure_rounds rounds (0: no limit), or when the budget is spent:
//     before every candidate the campaign checks that something is left of
//     its budget (a declared cost of certify may pass it by its size). A
//     closure the budget ended is kept in the bank (where it stopped, to the
//     candidate) and goes on first when the campaign does.
//   FAM-6. THE PICK. round_robin: the instances in turn (the cursor in the
//     bank). gap: the largest w(n) (f(n) - reference(n)); an instance
//     without a record, or with an infeasible one, first (its gap
//     infinite); ties to the instance untouched longer, then to the lower
//     n; it needs a reference for every instance. gap_age: the largest w(n)
//     age(n), age(n) the pieces since instance n was last given one (an
//     instance never given one: the pieces so far + 1); ties to the lower n.
//     gap_utility: gap's priority times the utility of the instance
//     (FAM-12), so that pieces which do not better an instance move the
//     next ones on to others.
//   FAM-7. THE PIECE. Piece p (0, 1, ... over the whole campaign, its file
//     included) runs the solver on problem_at(n) with the budget
//     min(piece_budget, what is left of the campaign's), the seed
//     strategy::start_seed(seed, p) (piece 0: the campaign's seed, STRAT-1),
//     the knobs, and the start x0 = the bank's point of n where it has one
//     (else the solver's own start). Its answer (the best point of its
//     result) is offered to the bank as source run. One instance, one piece,
//     no transfer: the bank's record is that scalar::run's answer, bit for
//     bit.
//   FAM-8. THE COUNT (stop_rule.hpp: one count, used()). The campaign's used()
//     is the sum of its pieces' Result::used, of the evaluations of the
//     transfer's candidates and of records given, and of certify's declared
//     costs. A piece's run keeps within its budget as every run does (STOP-1;
//     with budget_units = evaluations its gradients may pass it).
//   FAM-9. THE FILE and going on. save_bank writes the bank (every record,
//     the state, the counts, the history, the last call's options for the
//     scripts) as one JSON text to <file>.tmp, then moves it over <file>
//     (tried again for a few seconds where another process holds the file
//     open, Windows). load_bank reads it back exactly (the numbers' text
//     reads back to the same doubles). A campaign stopped between pieces
//     (max_pieces, a crash after a save) or by its budget inside a closure,
//     loaded and run on, is the campaign run in one piece, bit for bit: the
//     bank, the counts and the history. A piece is never cut and resumed:
//     where less than piece_budget was left, the last piece had the smaller
//     budget, and a campaign that goes on after it is not the one in one
//     piece from that piece on. The options
//     may differ between the calls (a stage of another solver goes on from
//     the bank of the one before); the file keeps the last call's.
//   FAM-10. THE SETTINGS (Report::settings, and the same in the file):
//     family_total, family_instances, family_missing, family_pieces,
//     family_used, family_offered / family_accepted / family_rejected
//     ({"run", "transfer", "given"}), family_transfer_skipped,
//     family_closure_rounds (the rounds of every closure, in order),
//     family_history ([used, total] after every change of the total),
//     family_gains (with the diminishing returns' measure: FAM-11),
//     family_stopped_by (budget, pieces, diminishing, or done: nothing to
//     pick).
//   FAM-11. Off by default (task 32, layer 5; diminish_ratio or
//     diminish_abs > 0). THE DIMINISHING RETURNS (scalar/stop_rule.hpp
//     STOP-8). The round is round_pieces pieces (0, the default: as many as
//     the family has instances, so that round_robin gives each one piece),
//     each with the closures it starts; it ends at the top of the loop, after
//     the closure of its last piece. The total objective is the bank's
//     (FAM-2) over the instances that have a record; a round's gain is
//     measured where the records before and after it are of the same
//     instances (none new in the round; at least one) and all feasible (cv
//     not > 0): the total before less the total after. A record that came
//     in the round (no point before) or a violating one: not measured. The
//     campaign ends after the round that ends the rule (stopped_by =
//     diminishing). The measure lives in the bank and its file (state's
//     "diminish": the piece the round began at, its total and instances, the
//     gains, g1, whether the rule ended it): a campaign loaded and run on
//     goes on with it, the campaign in one piece (FAM-9). A call whose
//     measure the rule has ended, or a bank without one, begins a new
//     measure at its first piece (a stage that follows a stage the rule
//     ended starts its own); a call without diminish_* drops the measure
//     (the bank and its file then as before). A loop of stages (calls of
//     run with other solvers or knobs) is measured by the caller: a Diminish
//     of its own told each cycle's gain, the totals of the reports before
//     and after.
//   FAM-12. pick = gap_utility (task 33.1; the other picks as before). THE
//     UTILITY of MOEA/D-DRA: Q. Zhang, W. Liu, H. Li, "The performance of a
//     new version of MOEA/D on CEC09 unconstrained MOP test instances", IEEE
//     CEC 2009, pp. 203-208, doi:10.1109/CEC.2009.4982949 (source:
//     moead-dra_zhang2009), Step 1.4 and Step 5: every subproblem i has a
//     utility pi^i, 1 at the start; after a stretch of the search, with
//     Delta^i the relative decrease of its objective (old - new) / old,
//       pi^i = 1                                  if Delta^i > 0.001,
//       pi^i = (0.95 + 0.05 Delta^i / 0.001) pi^i  otherwise,
//     and the effort goes to the subproblems of the highest utility (the
//     paper's 10-tournament). Here an instance is the subproblem and its gap
//     f(n) - reference(n) its objective (as the paper's Tchebycheff value,
//     measured from a reference point); the stretch is a piece on the
//     instance: Delta = (gap before the piece - gap after its answer was
//     offered) / gap before, the closure that follows the piece not counted
//     (a transfer that betters the instance lowers its gap, not its
//     utility). An instance without a feasible record before the piece:
//     utility 1 where the piece gave it one, else Delta = 0; a gap before not
//     above 0: Delta = 0 (moead_dra.hpp's guard on old). utility_factor (0.95,
//     the paper's; 0 < it < 1) is the factor of a piece that betters nothing,
//     the 0.05 its complement 1 - utility_factor, the factor floored at 0 as
//     moead_dra.hpp's DRA-11 (the bank takes no worse record, so Delta is not
//     below 0 here). The pick: the largest pi(n) w(n) (f(n) - reference(n))
//     where w(n) (f(n) - reference(n)) > 0, else that product alone (an
//     instance at or below its reference is not raised by a small utility);
//     an instance without a record, or with an infeasible one, first; ties
//     as gap's. The tournament's draws are left out: the pick stays
//     deterministic, the highest priority taken, as gap takes it. The
//     utilities live in the bank and its file (state's "utility", [n, pi]
//     of every instance given a piece by this pick, ascending n; absent in a
//     bank without one, all utilities then 1): a campaign loaded and run on
//     goes on with them (FAM-9).
//   FAM-13. fresh_first (off by default; task 33.1): THE FRESH RECORDS
//     FIRST. A record is fresh where it did not come from a piece (source
//     transfer, given, or a label given, as components.hpp's assembled) and
//     no piece has been given to its instance since it came (the record's
//     piece not below the instance's last piece; a record of the piece p
//     came after piece p's run). While some instance has a fresh record, the
//     pick chooses among those instances alone, by its own rule (round_robin:
//     the first of them from the cursor on, the cursor then after it; the
//     others: their largest priority, ties as theirs); with none, it is the
//     pick of before. An instance without a record is not fresh: a fresh
//     record goes before it even under gap. The bank already holds what this
//     needs (the records' source and piece, touched): nothing new in its
//     file but the option.
//   FAM-14. budget_more (off by default; task 33.3): SO MUCH MORE. budget
//     counts the bank's used() in all, the file's included (FAM-8), so a
//     stage that goes on from a file is given its end, not its share. With
//     budget_more = B (budget then 0: one of the two) the call's budget is
//     ceil(used()) when the campaign is made, plus B: the stage spends B more
//     than the file had. The file's options keep that budget and
//     budget_more. Two calls of budget_more = B from a bank are one of
//     budget = 2B where the first ended at its budget exactly (no closure or
//     declared cost past it), bit for bit. A budget not above what the bank
//     had spent when the campaign was made is no longer silent: the call
//     gives no piece (stopped_by budget, as before) and its settings say so
//     (family_budget_spent: what the bank had spent).
//
// Options: solver ("nelder_mead"), knobs, eq_tol (1e-4), pop_size (100),
// budget (the campaign's used() in all, the file's included), piece_budget,
// max_pieces (0: none), pick (round_robin), closure_rounds (0: none), seed
// (1), bank_file ("": none), save_every (1), diminish_ratio, diminish_abs (0:
// off), round_pieces (0: the number of instances; read with the diminishing
// returns), utility_factor (0.95; read with pick = gap_utility), fresh_first
// (false), budget_more (0: off; FAM-14). The file's options name utility_factor only with pick =
// gap_utility and fresh_first only where it is on.
// ============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "../json.hpp"
#include "../scalar/scalar.hpp"
#include "../store/pack.hpp"

namespace mootation::campaign::fam {

using scalar::Problem;
using scalar::Vec;

// certify's answer (FAM-3)
struct Certified {
    bool ok = false;
    double f = 0.0;              // the certified value
    double cost = 0.0;           // the check's declared cost in used()
};

// a transfer's candidate (FAM-4)
struct Candidate {
    long long n = 0;
    Vec x;
};

struct Family {
    std::vector<long long> instances;                                         // FAM-1
    std::function<Problem(long long n)> problem_at;
    std::function<double(long long n)> weight;                                // none: 1
    std::function<double(long long n)> reference;                             // none: pick = gap refused
    std::function<Certified(long long n, const Vec& x)> certify;               // none: the run's f
    std::function<std::vector<Candidate>(long long n, const Vec& x)> transfer; // none: no closure
};

// a weight or reference given as a table; an instance it lacks is refused
inline std::function<double(long long)> table_of(const std::map<long long, double>& t, const std::string& what)
{
    return [t, what](long long n) {
        const auto it = t.find(n);
        if (it == t.end()) throw std::invalid_argument("family: the " + what + " table has no instance " + std::to_string(n));
        return it->second;
    };
}

// the instances of a problem with at_param: instance n is the problem at the
// level n (param.hpp problem_at, checked as there: PARAM-1)
inline std::function<Problem(long long)> levels_of(const Problem& P)
{
    return [P](long long n) { return scalar::problem_at(P, static_cast<double>(n)); };
}

struct Record {
    Vec x;
    double f = std::numeric_limits<double>::infinity();
    double cv = std::numeric_limits<double>::infinity();
    std::string source;          // run, transfer, given
    long long from = -1;         // transfer: the instance it was made from
    long long piece = -1;        // the piece it came in (-1: before the first)
    double used = 0.0;           // the campaign's used() when the bank took it
};

// what a path offered, what the bank took and what certify rejected
struct Counts {
    long long offered = 0, accepted = 0, rejected = 0;
};

// THE DIMINISHING RETURNS' measure (FAM-11; components.hpp COMP-12): the
// round's start and its total, and the rule with the rounds' gains
struct Measure {
    bool on = false;                     // a measure begun (kept in the file)
    long long start = 0;                 // the piece the round began at
    double total = 0.0;                  // the total objective then ...
    std::vector<long long> over;         // ... over these instances (ascending)
    bool feasible = false;               // ... all of them feasible
    scalar::strategy::Diminish rule;     // ratio and abs: the call's
};

// a piece of the campaign, as the bank keeps it
struct PieceLog {
    long long n = 0;
    double used_at = 0.0;        // the campaign's used() before the piece
    long long budget = 0;        // the piece's budget
    bool taken = false;          // the bank took its answer
};

struct Bank {
    std::map<long long, Record> records;
    // the campaign's state
    double used = 0.0;
    std::vector<PieceLog> pieces;
    std::size_t cursor = 0;                              // round_robin
    std::map<long long, long long> touched;              // the last piece given to the instance
    Counts run, transfer, given;
    long long transfer_skipped = 0;
    std::vector<std::pair<double, double>> history;      // [used, total] after every change
    std::vector<long long> closure_rounds;               // the rounds of every closure ended
    // a closure not finished (FAM-5): its round, the records of this round
    // and of the next, the record and the candidate it is at
    long long round = 0;
    std::vector<long long> now, next;
    std::size_t at = 0, cand = 0;
    json::Value options = json::Value::object();         // the last call's (for the scripts)
    Measure measure;                                     // FAM-11 (off: none)
    std::map<long long, double> utility;                 // FAM-12 (pick = gap_utility; none: 1)
};

struct Options {
    std::string solver = "nelder_mead";
    scalar::Knobs knobs;
    double eq_tol = 1e-4;
    int pop_size = 100;
    long long budget = 0;
    long long piece_budget = 0;
    long long max_pieces = 0;
    std::string pick = "round_robin";
    long long closure_rounds = 0;
    unsigned seed = 1;
    std::string bank_file;
    long long save_every = 1;
    double diminish_ratio = 0.0;                         // FAM-11 (0: off)
    double diminish_abs = 0.0;
    long long round_pieces = 0;                          // 0: the number of instances
    double utility_factor = 0.95;                        // FAM-12 (pick = gap_utility)
    bool fresh_first = false;                            // FAM-13 (off)
    long long budget_more = 0;                           // FAM-14 (0: off; budget then 0)
};

struct Report {
    double total = 0.0;
    std::string stopped_by;
    std::vector<std::pair<std::string, std::string>> settings;
};

namespace detail {

inline const char* const kFormat = "mootation-family-bank";
constexpr int kVersion = 1;

// FAM-2: a better than b
inline bool better(double fa, double cva, double fb, double cvb)
{
    if (cva != cvb) return cva < cvb;
    return fa < fb;
}

inline double number_of(const json::Value& v, const std::string& what)
{
    if (v.kind == json::Value::Kind::Int) return static_cast<double>(v.i);
    if (v.kind == json::Value::Kind::Float) return v.f;
    throw std::invalid_argument("family bank: " + what + " is not a number");
}

inline long long whole_of(const json::Value& v, const std::string& what)
{
    if (v.kind != json::Value::Kind::Int) throw std::invalid_argument("family bank: " + what + " is not a whole number");
    return static_cast<long long>(v.i);
}

// key by value: a reference to o's member comes back, and GCC's
// -Wdangling-reference takes a temporary string argument for its source (as
// config.hpp's req)
inline const json::Value& field(const json::Value& o, std::string_view key_)
{
    const std::string key(key_);
    const json::Value* v = o.find(key);
    if (!v) throw std::invalid_argument("family bank: no \"" + key + "\"");
    return *v;
}

inline json::Value numbers(const Vec& x)
{
    json::Value a = json::Value::array();
    for (double v : x) a.items.push_back(json::Value::number(v));
    return a;
}

inline json::Value wholes(const std::vector<long long>& x)
{
    json::Value a = json::Value::array();
    for (long long v : x) a.items.push_back(json::Value::integer(v));
    return a;
}

inline json::Value counts_json(const Bank& B, long long Counts::*c)
{
    json::Value o = json::Value::object();
    o.set("run", json::Value::integer(B.run.*c));
    o.set("transfer", json::Value::integer(B.transfer.*c));
    o.set("given", json::Value::integer(B.given.*c));
    return o;
}

inline json::Value history_json(const Bank& B)
{
    json::Value a = json::Value::array();
    for (const auto& h : B.history) {
        json::Value p = json::Value::array();
        p.items.push_back(json::Value::number(h.first));
        p.items.push_back(json::Value::number(h.second));
        a.items.push_back(p);
    }
    return a;
}

inline json::Value record_json(long long n, const Record& r)
{
    json::Value o = json::Value::object();
    o.set("n", json::Value::integer(n));
    o.set("f", json::Value::number(r.f));
    o.set("cv", json::Value::number(r.cv));
    o.set("source", json::Value::string(r.source));
    o.set("from", r.from >= 0 ? json::Value::integer(r.from) : json::Value::null());
    o.set("piece", json::Value::integer(r.piece));
    o.set("used", json::Value::number(r.used));
    o.set("x", numbers(r.x));
    return o;
}

inline json::Value options_json(const Options& o)
{
    json::Value v = json::Value::object(), k = json::Value::object();
    for (const auto& kv : o.knobs)
        k.set(kv.first, kv.second.is_word ? json::Value::string(kv.second.word) : json::Value::number(kv.second.number));
    v.set("solver", json::Value::string(o.solver));
    v.set("knobs", k);
    v.set("eq_tol", json::Value::number(o.eq_tol));
    v.set("pop_size", json::Value::integer(o.pop_size));
    v.set("budget", json::Value::integer(o.budget));
    v.set("piece_budget", json::Value::integer(o.piece_budget));
    v.set("max_pieces", json::Value::integer(o.max_pieces));
    v.set("pick", json::Value::string(o.pick));
    v.set("closure_rounds", json::Value::integer(o.closure_rounds));
    v.set("seed", json::Value::integer(o.seed));
    v.set("save_every", json::Value::integer(o.save_every));
    if (o.diminish_ratio > 0.0 || o.diminish_abs > 0.0) {                     // FAM-11: only where asked
        v.set("diminish_ratio", json::Value::number(o.diminish_ratio));
        v.set("diminish_abs", json::Value::number(o.diminish_abs));
        v.set("round_pieces", json::Value::integer(o.round_pieces));
    }
    if (o.pick == "gap_utility") v.set("utility_factor", json::Value::number(o.utility_factor));   // FAM-12
    if (o.fresh_first) v.set("fresh_first", json::Value::boolean(true));                         // FAM-13
    if (o.budget_more > 0) v.set("budget_more", json::Value::integer(o.budget_more));            // FAM-14
    return v;
}

// FAM-11: the measure as the file keeps it
inline json::Value measure_json(const Measure& m)
{
    json::Value o = json::Value::object();
    o.set("start", json::Value::integer(m.start));
    o.set("total", json::Value::number(m.total));
    o.set("over", wholes(m.over));
    o.set("feasible", json::Value::boolean(m.feasible));
    o.set("gains", m.rule.gains_json());
    o.set("ref", m.rule.has_ref ? json::Value::number(m.rule.ref) : json::Value::null());
    o.set("ended", json::Value::boolean(m.rule.ended));
    return o;
}

inline Measure measure_of(const json::Value& o, const std::string& who)
{
    Measure m;
    m.on = true;
    m.start = whole_of(field(o, "start"), who + " start");
    m.total = number_of(field(o, "total"), who + " total");
    for (const auto& n : field(o, "over").items) m.over.push_back(whole_of(n, who + " over"));
    const json::Value& f = field(o, "feasible");
    const json::Value& e = field(o, "ended");
    if (f.kind != json::Value::Kind::Bool || e.kind != json::Value::Kind::Bool)
        throw std::invalid_argument(who + ": feasible and ended are true or false");
    m.feasible = f.b;
    m.rule.ended = e.b;
    for (const auto& g : field(o, "gains").items)
        m.rule.gains.push_back(g.kind == json::Value::Kind::Null ? std::numeric_limits<double>::quiet_NaN()
                                                                 : number_of(g, who + " gains"));
    m.rule.rounds = static_cast<long long>(m.rule.gains.size());
    const json::Value& r = field(o, "ref");
    m.rule.has_ref = r.kind != json::Value::Kind::Null;
    m.rule.ref = m.rule.has_ref ? number_of(r, who + " ref") : 0.0;
    return m;
}

// FAM-11: a round's gain from the totals at its two ends; NaN where they are not comparable
inline double gain_of(const Measure& before, double total, const std::vector<long long>& over, bool feasible)
{
    if (over.empty() || over != before.over || !feasible || !before.feasible)
        return std::numeric_limits<double>::quiet_NaN();
    return before.total - total;
}

// FAM-9: <path>.tmp moved over <path>; a file another process holds open
// (Windows) is tried again for a few seconds
inline void write_atomic(const std::string& path, const std::string& text)
{
    const std::filesystem::path p = store::file::path_of(path);   // UTF-8, as in C++17 and C++20
    std::filesystem::path tmp = p;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.write(text.data(), static_cast<std::streamsize>(text.size()));
        f.flush();
        if (!f) throw std::runtime_error("family bank: cannot write " + path + ".tmp");
    }
    std::chrono::milliseconds delay(10);
    for (int attempt = 0; attempt < 39; ++attempt) {
        std::error_code ec;
        std::filesystem::rename(tmp, p, ec);
        if (!ec) return;
        std::this_thread::sleep_for(delay);
        delay = (std::min)(2 * delay, std::chrono::milliseconds(250));
    }
    std::filesystem::rename(tmp, p);
}

}   // namespace detail

// FAM-2: the total objective, in ascending n
inline double total_of(const Bank& B, const std::function<double(long long)>& w)
{
    double s = 0.0;
    for (const auto& [n, r] : B.records) s += (w ? w(n) : 1.0) * r.f;
    return s;
}

// FAM-10: the settings of a bank (its total by the weights w)
inline std::vector<std::pair<std::string, std::string>> settings_of(const Bank& B, double total,
                                                                  std::size_t instances,
                                                                  const std::string& stopped_by)
{
    std::vector<std::pair<std::string, std::string>> s;
    s.push_back({"family_total", json::float_repr(total)});
    s.push_back({"family_instances", std::to_string(instances)});
    s.push_back({"family_missing", std::to_string(instances - (std::min)(instances, B.records.size()))});
    s.push_back({"family_pieces", std::to_string(B.pieces.size())});
    s.push_back({"family_used", json::float_repr(B.used)});
    s.push_back({"family_offered", json::dump(detail::counts_json(B, &Counts::offered))});
    s.push_back({"family_accepted", json::dump(detail::counts_json(B, &Counts::accepted))});
    s.push_back({"family_rejected", json::dump(detail::counts_json(B, &Counts::rejected))});
    s.push_back({"family_transfer_skipped", std::to_string(B.transfer_skipped)});
    s.push_back({"family_closure_rounds", json::dump(detail::wholes(B.closure_rounds))});
    s.push_back({"family_history", json::dump(detail::history_json(B))});
    if (B.measure.on) s.push_back({"family_gains", json::dump(B.measure.rule.gains_json())});   // FAM-11
    if (!stopped_by.empty()) s.push_back({"family_stopped_by", stopped_by});
    return s;
}

// FAM-9: the bank as the file's text — one record a line
inline std::string bank_text(const Bank& B, double total, std::size_t instances, const std::string& stopped_by)
{
    std::string t = "{\n";
    t += "\"format\": \"" + std::string(detail::kFormat) + "\",\n";
    t += "\"version\": " + std::to_string(detail::kVersion) + ",\n";
    json::Value s = json::Value::object();
    for (const auto& [k, v] : settings_of(B, total, instances, stopped_by)) {
        if (k == "family_history") continue;                     // its own line, below
        s.set(k, json::parse(k == "family_stopped_by" ? json::dump(json::Value::string(v)) : v));
    }
    t += "\"settings\": " + json::dump(s) + ",\n";
    t += "\"options\": " + json::dump(B.options) + ",\n";
    json::Value st = json::Value::object(), touched = json::Value::array(), pieces = json::Value::array();
    for (const auto& [n, p] : B.touched) touched.items.push_back(detail::wholes({n, p}));
    for (const PieceLog& p : B.pieces) {
        json::Value q = json::Value::array();
        q.items.push_back(json::Value::integer(p.n));
        q.items.push_back(json::Value::number(p.used_at));
        q.items.push_back(json::Value::integer(p.budget));
        q.items.push_back(json::Value::boolean(p.taken));
        pieces.items.push_back(q);
    }
    json::Value cl = json::Value::object();
    cl.set("round", json::Value::integer(B.round));
    cl.set("now", detail::wholes(B.now));
    cl.set("next", detail::wholes(B.next));
    cl.set("at", json::Value::integer(static_cast<std::int64_t>(B.at)));
    cl.set("cand", json::Value::integer(static_cast<std::int64_t>(B.cand)));
    st.set("used", json::Value::number(B.used));
    st.set("cursor", json::Value::integer(static_cast<std::int64_t>(B.cursor)));
    st.set("touched", touched);
    st.set("closure", cl);
    if (B.measure.on) st.set("diminish", detail::measure_json(B.measure));      // FAM-11: only where begun
    if (!B.utility.empty()) {                                                    // FAM-12: only where given
        json::Value u = json::Value::array();
        for (const auto& [n, pi] : B.utility) {
            json::Value q = json::Value::array();
            q.items.push_back(json::Value::integer(n));
            q.items.push_back(json::Value::number(pi));
            u.items.push_back(q);
        }
        st.set("utility", u);
    }
    t += "\"state\": " + json::dump(st) + ",\n";
    t += "\"pieces\": " + json::dump(pieces) + ",\n";
    t += "\"history\": " + json::dump(detail::history_json(B)) + ",\n";
    t += "\"records\": [";
    bool first = true;
    for (const auto& [n, r] : B.records) {
        t += first ? "\n" : ",\n";
        t += json::dump(detail::record_json(n, r));
        first = false;
    }
    t += "\n]\n}\n";
    return t;
}

inline void save_bank(const std::string& path, const Bank& B, double total, std::size_t instances,
                      const std::string& stopped_by = "")
{
    detail::write_atomic(path, bank_text(B, total, instances, stopped_by));
}

// FAM-9: a bank from its file's text
inline Bank bank_of_text(const std::string& text)
{
    using detail::field;
    using detail::number_of;
    using detail::whole_of;
    const json::Value v = json::parse(text);
    if (v.kind != json::Value::Kind::Object) throw std::invalid_argument("family bank: not a JSON object");
    const json::Value& fmt = field(v, "format");
    if (fmt.kind != json::Value::Kind::String || fmt.s != detail::kFormat)
        throw std::invalid_argument("family bank: the format is not " + std::string(detail::kFormat));
    if (whole_of(field(v, "version"), "version") != detail::kVersion)
        throw std::invalid_argument("family bank: version " + std::to_string(whole_of(field(v, "version"), "version")) +
                                    " (this library reads " + std::to_string(detail::kVersion) + ")");
    Bank B;
    const json::Value& s = field(v, "settings");
    auto counts = [&](long long Counts::*c, const std::string& key) {
        const json::Value& o = field(s, key);
        B.run.*c = whole_of(field(o, "run"), key);
        B.transfer.*c = whole_of(field(o, "transfer"), key);
        B.given.*c = whole_of(field(o, "given"), key);
    };
    counts(&Counts::offered, "family_offered");
    counts(&Counts::accepted, "family_accepted");
    counts(&Counts::rejected, "family_rejected");
    B.transfer_skipped = whole_of(field(s, "family_transfer_skipped"), "family_transfer_skipped");
    for (const auto& r : field(s, "family_closure_rounds").items) B.closure_rounds.push_back(whole_of(r, "a closure's rounds"));
    B.options = field(v, "options");
    const json::Value& st = field(v, "state");
    B.used = number_of(field(st, "used"), "used");
    B.cursor = static_cast<std::size_t>(whole_of(field(st, "cursor"), "cursor"));
    for (const auto& p : field(st, "touched").items) {
        if (p.items.size() != 2) throw std::invalid_argument("family bank: touched is a list of [n, piece]");
        B.touched[whole_of(p.items[0], "touched")] = whole_of(p.items[1], "touched");
    }
    const json::Value& cl = field(st, "closure");
    B.round = whole_of(field(cl, "round"), "round");
    for (const auto& n : field(cl, "now").items) B.now.push_back(whole_of(n, "closure now"));
    for (const auto& n : field(cl, "next").items) B.next.push_back(whole_of(n, "closure next"));
    B.at = static_cast<std::size_t>(whole_of(field(cl, "at"), "closure at"));
    B.cand = static_cast<std::size_t>(whole_of(field(cl, "cand"), "closure cand"));
    if (const json::Value* d = st.find("diminish")) B.measure = detail::measure_of(*d, "family bank: diminish");
    if (const json::Value* u = st.find("utility")) {                            // FAM-12
        for (const auto& p : u->items) {
            if (p.items.size() != 2) throw std::invalid_argument("family bank: utility is a list of [n, pi]");
            B.utility[whole_of(p.items[0], "utility")] = number_of(p.items[1], "utility");
        }
    }
    for (const auto& p : field(v, "pieces").items) {
        if (p.items.size() != 4 || p.items[3].kind != json::Value::Kind::Bool)
            throw std::invalid_argument("family bank: a piece is [n, used_at, budget, taken]");
        B.pieces.push_back(PieceLog{whole_of(p.items[0], "piece n"), number_of(p.items[1], "piece used_at"),
                                    whole_of(p.items[2], "piece budget"), p.items[3].b});
    }
    for (const auto& h : field(v, "history").items) {
        if (h.items.size() != 2) throw std::invalid_argument("family bank: history is a list of [used, total]");
        B.history.push_back({number_of(h.items[0], "history"), number_of(h.items[1], "history")});
    }
    for (const auto& r : field(v, "records").items) {
        Record rec;
        const long long n = whole_of(field(r, "n"), "a record's n");
        rec.f = number_of(field(r, "f"), "a record's f");
        rec.cv = number_of(field(r, "cv"), "a record's cv");
        const json::Value& src = field(r, "source");
        if (src.kind != json::Value::Kind::String) throw std::invalid_argument("family bank: a record's source");
        rec.source = src.s;
        const json::Value& from = field(r, "from");
        rec.from = from.kind == json::Value::Kind::Null ? -1 : whole_of(from, "a record's from");
        rec.piece = whole_of(field(r, "piece"), "a record's piece");
        rec.used = number_of(field(r, "used"), "a record's used");
        for (const auto& x : field(r, "x").items) rec.x.push_back(number_of(x, "a record's x"));
        if (!B.records.emplace(n, std::move(rec)).second)
            throw std::invalid_argument("family bank: two records of instance " + std::to_string(n));
    }
    return B;
}

inline Bank load_bank(const std::string& path)
{
    std::ifstream f(store::file::path_of(path), std::ios::binary);
    if (!f) throw std::runtime_error("family bank: cannot read " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return bank_of_text(ss.str());
}

// ── the campaign ────────────────────────────────────────────────────────────

class Campaign {
public:
    // the family and the options are copied (their hooks with them); the bank is the caller's
    Campaign(const Family& F, Bank& B, const Options& o) : F_(F), B_(B), o_(o)
    {
        check();
        spent_ = static_cast<long long>(std::ceil(B_.used));                  // FAM-14
        budget_ = o_.budget_more > 0 ? spent_ + o_.budget_more : o_.budget;
    }

    // records given (source given: the record's label, counted on the path
    // given; layer 3's loop gives its assemblies as "assembled",
    // components.hpp COMP-7): each certified or evaluated and offered; those
    // the bank takes start a closure (FAM-5)
    void give(long long n, const Vec& x, const std::string& source = "given")
    {
        if (!in_family(n)) throw std::invalid_argument("family: instance " + std::to_string(n) + " is not in the family");
        const Problem& Q = problem(n);
        if (x.size() != Q.lo.size())
            throw std::invalid_argument("family: the point given for instance " + std::to_string(n) + " has " +
                                        std::to_string(x.size()) + " values, the problem " +
                                        std::to_string(Q.lo.size()));
        if (offer_point(n, x, B_.given, source, -1)) begin_closure(n);
    }

    // the closure alone, from where the bank has it (FAM-5; layer 3's loop
    // after its assemblies); false where the budget ended it
    bool closure() { return close(); }

    Report run()
    {
        Options given = o_;                                                    // FAM-14: the call's budget
        given.budget = budget_;
        B_.options = detail::options_json(given);
        const bool dim = o_.diminish_ratio > 0.0 || o_.diminish_abs > 0.0;   // FAM-11
        if (!dim) B_.measure = Measure();
        std::string stopped;
        long long since_save = 0;
        for (;;) {
            if (!close()) {                                                    // FAM-5: the budget ended it
                stopped = "budget";
                break;
            }
            if (dim && diminished()) {                                         // FAM-11
                stopped = "diminishing";
                break;
            }
            if (o_.max_pieces > 0 && static_cast<long long>(B_.pieces.size()) >= o_.max_pieces) {
                stopped = "pieces";
                break;
            }
            const long long left = budget_ - static_cast<long long>(std::ceil(B_.used));
            if (left <= 0) {
                stopped = "budget";
                break;
            }
            piece(pick(), (std::min)(o_.piece_budget, left));
            if (!o_.bank_file.empty() && ++since_save >= o_.save_every) {
                close();                                                       // the closure of the piece first
                save("");
                since_save = 0;
            }
        }
        if (!o_.bank_file.empty()) save(stopped);
        Report r;
        r.total = total();
        r.stopped_by = stopped;
        r.settings = settings_of(B_, r.total, F_.instances.size(), stopped);
        if (o_.budget_more <= 0 && o_.budget <= spent_)                        // FAM-14: not silent
            r.settings.push_back({"family_budget_spent", std::to_string(spent_)});
        return r;
    }

    double total() { return total_of(B_, [this](long long n) { return weight(n); }); }

private:
    const Family F_;
    Bank& B_;
    const Options o_;
    long long spent_ = 0;                                                      // FAM-14: ceil(used()) when made
    long long budget_ = 0;                                                     // the call's budget
    std::vector<long long> inst_;                                              // FAM-1: ascending
    std::map<long long, Problem> problems_;
    std::map<long long, double> weights_, references_;

    void check()
    {
        if (F_.instances.empty()) throw std::invalid_argument("family: no instances");
        if (!F_.problem_at) throw std::invalid_argument("family: no problem_at");
        inst_ = F_.instances;
        std::sort(inst_.begin(), inst_.end());
        if (std::adjacent_find(inst_.begin(), inst_.end()) != inst_.end())
            throw std::invalid_argument("family: an instance given twice");
        if (o_.pick != "round_robin" && o_.pick != "gap" && o_.pick != "gap_age" && o_.pick != "gap_utility")
            throw std::invalid_argument("family: pick = round_robin | gap | gap_age | gap_utility, not " + o_.pick);
        if ((o_.pick == "gap" || o_.pick == "gap_utility") && !F_.reference)
            throw std::invalid_argument("family: pick = " + o_.pick + " needs reference(n)");
        if (!(o_.utility_factor > 0.0 && o_.utility_factor < 1.0))
            throw std::invalid_argument("family: 0 < utility_factor < 1");
        if (o_.closure_rounds < 0) throw std::invalid_argument("family: closure_rounds >= 0");
        if (o_.max_pieces < 0) throw std::invalid_argument("family: max_pieces >= 0");
        if (o_.save_every < 1) throw std::invalid_argument("family: save_every >= 1");
        if (o_.budget_more < 0) throw std::invalid_argument("family: budget_more >= 0");
        if (o_.budget_more > 0 && o_.budget != 0)
            throw std::invalid_argument("family: budget or budget_more, not both");
        scalar::strategy::Diminish::check(o_.diminish_ratio, o_.diminish_abs, "family");
        if (o_.round_pieces < 0) throw std::invalid_argument("family: round_pieces >= 0");
        for (const auto& [n, r] : B_.records) {                                // FAM-9: a bank of this family
            if (!in_family(n))
                throw std::invalid_argument("family: the bank has instance " + std::to_string(n) + ", not in the family");
            if (r.x.size() != problem(n).lo.size())
                throw std::invalid_argument("family: the bank's point of instance " + std::to_string(n) +
                                            " is not of its problem's length");
        }
        for (const auto& [n, p] : B_.touched)
            if (!in_family(n)) throw std::invalid_argument("family: the bank touched instance " + std::to_string(n));
    }

    bool in_family(long long n) const { return std::binary_search(inst_.begin(), inst_.end(), n); }

    const Problem& problem(long long n)
    {
        auto it = problems_.find(n);
        if (it == problems_.end()) it = problems_.emplace(n, F_.problem_at(n)).first;
        return it->second;
    }

    double weight(long long n)
    {
        if (!F_.weight) return 1.0;
        auto it = weights_.find(n);
        if (it == weights_.end()) {
            const double w = F_.weight(n);
            if (!std::isfinite(w)) throw std::invalid_argument("family: w(" + std::to_string(n) + ") is not finite");
            it = weights_.emplace(n, w).first;
        }
        return it->second;
    }

    double reference(long long n)
    {
        auto it = references_.find(n);
        if (it == references_.end()) {
            const double r = F_.reference(n);
            if (!std::isfinite(r))
                throw std::invalid_argument("family: reference(" + std::to_string(n) + ") is not finite");
            it = references_.emplace(n, r).first;
        }
        return it->second;
    }

    long long left() const { return budget_ - static_cast<long long>(std::ceil(B_.used)); }

    // FAM-2, FAM-3: a point with its values (from a run or an evaluation)
    // offered; whether the bank took it
    bool offer(long long n, const Vec& x, double f, double cv, Counts& c, const std::string& source, long long from)
    {
        ++c.offered;
        if (F_.certify) {
            const Certified z = F_.certify(n, x);
            B_.used += z.cost;
            if (!z.ok) {
                ++c.rejected;
                return false;
            }
            f = z.f;
            cv = 0.0;
        }
        const auto it = B_.records.find(n);
        if (it != B_.records.end() && !detail::better(f, cv, it->second.f, it->second.cv)) return false;
        Record r;
        r.x = x;
        r.f = f;
        r.cv = cv;
        r.source = source;
        r.from = from;
        r.piece = static_cast<long long>(B_.pieces.size()) - 1;
        r.used = B_.used;
        B_.records[n] = std::move(r);
        ++c.accepted;
        B_.history.push_back({B_.used, total()});
        return true;
    }

    // FAM-4: a point without values: certified, else evaluated, then offered
    bool offer_point(long long n, const Vec& x, Counts& c, const std::string& source, long long from)
    {
        if (F_.certify) return offer(n, x, 0.0, 0.0, c, source, from);
        const Problem& Q = problem(n);
        // one evaluation (with a gradient where the problem gives its value only so); the
        // budget is the campaign's check, not the evaluator's
        scalar::Evaluator ev(Q, o_.eq_tol, false, 0.0, std::int64_t{1} << 40, 1, {}, 0);
        std::size_t i = 0;
        try {
            i = ev.index(x);
        } catch (const scalar::BudgetSpent&) {                                  // the problem's deadline
            ++c.offered;
            return false;
        }
        B_.used += ev.used();
        if (ev.failed[i]) {
            ++c.offered;
            return false;
        }
        return offer(n, x, ev.F[i][0], ev.cv(i), c, source, from);
    }

    // FAM-5: instance n's record got better. A closure that has not begun its
    // round (a new one; records given; a closure loaded between rounds) takes
    // it in this round, else in the next.
    void begin_closure(long long n)
    {
        if (!F_.transfer) return;
        if (B_.now.empty()) {
            B_.round = 1;
            B_.at = B_.cand = 0;
        }
        const bool fresh_round = B_.at == 0 && B_.cand == 0;
        std::vector<long long>& to = fresh_round ? B_.now : B_.next;
        if (std::find(to.begin(), to.end(), n) != to.end()) return;
        to.push_back(n);
        if (fresh_round) std::sort(B_.now.begin(), B_.now.end());
    }

    // FAM-5: the closure from where it is; false where the budget ended it
    bool close()
    {
        if (!F_.transfer) {                                                    // a bank's closure, a family without one
            B_.now.clear();
            B_.next.clear();
            B_.round = 0;
            B_.at = B_.cand = 0;
        }
        while (!B_.now.empty()) {
            const bool limit = o_.closure_rounds > 0 && B_.round > o_.closure_rounds;
            while (!limit && B_.at < B_.now.size()) {
                const long long n = B_.now[B_.at];
                const std::vector<Candidate> cands = F_.transfer(n, B_.records.at(n).x);
                while (B_.cand < cands.size()) {
                    if (left() <= 0) return false;
                    const Candidate& c = cands[B_.cand];
                    if (!in_family(c.n) || !fits(c)) {
                        ++B_.transfer_skipped;
                    } else if (offer_point(c.n, c.x, B_.transfer, "transfer", n)) {
                        if (std::find(B_.next.begin(), B_.next.end(), c.n) == B_.next.end()) B_.next.push_back(c.n);
                    }
                    ++B_.cand;
                }
                B_.cand = 0;
                ++B_.at;
            }
            if (limit || B_.next.empty()) {
                B_.closure_rounds.push_back(limit ? B_.round - 1 : B_.round);
                B_.now.clear();
                B_.next.clear();
                B_.round = 0;
                B_.at = B_.cand = 0;
                break;
            }
            std::sort(B_.next.begin(), B_.next.end());
            B_.now = std::move(B_.next);
            B_.next.clear();
            B_.at = B_.cand = 0;
            ++B_.round;
        }
        return true;
    }

    // FAM-11: at the top of the loop; whether the round just ended ends the campaign
    bool diminished()
    {
        Measure& m = B_.measure;
        const long long p = static_cast<long long>(B_.pieces.size());
        std::vector<long long> over;
        bool feasible = true;
        for (const auto& [n, r] : B_.records) {
            over.push_back(n);
            feasible = feasible && !(r.cv > 0.0);
        }
        auto at_start = [&]() {
            m.start = p;
            m.total = total();
            m.over = over;
            m.feasible = feasible;
        };
        if (!m.on || m.rule.ended) {                                           // a new measure from here
            m = Measure();
            m.on = true;
            m.rule.ratio = o_.diminish_ratio;
            m.rule.abs = o_.diminish_abs;
            at_start();
            return false;
        }
        m.rule.ratio = o_.diminish_ratio;                                      // the call's
        m.rule.abs = o_.diminish_abs;
        const long long R = o_.round_pieces > 0 ? o_.round_pieces : static_cast<long long>(inst_.size());
        if (p - m.start < R) return false;
        const double now = total();
        const double g = detail::gain_of(m, now, over, feasible);
        at_start();
        return m.rule.round(g);
    }

    bool fits(const Candidate& c)
    {
        const Problem& Q = problem(c.n);
        if (c.x.size() != Q.lo.size()) return false;
        for (std::size_t j = 0; j < c.x.size(); ++j)
            if (!(c.x[j] >= Q.lo[j] && c.x[j] <= Q.hi[j])) return false;
        return true;
    }

    long long age(long long n) const
    {
        const auto it = B_.touched.find(n);
        return static_cast<long long>(B_.pieces.size()) - (it == B_.touched.end() ? -1 : it->second);
    }

    // FAM-13: the record of n did not come from a piece, and no piece was given to n since it came
    bool fresh(long long n) const
    {
        const auto it = B_.records.find(n);
        if (it == B_.records.end() || it->second.source == "run") return false;
        const auto t = B_.touched.find(n);
        return t == B_.touched.end() || t->second <= it->second.piece;
    }

    double utility(long long n) const
    {
        const auto it = B_.utility.find(n);
        return it == B_.utility.end() ? 1.0 : it->second;
    }

    // FAM-6, FAM-12, FAM-13
    long long pick()
    {
        std::vector<long long> fresh_ones;                                     // FAM-13
        if (o_.fresh_first)
            for (long long n : inst_)
                if (fresh(n)) fresh_ones.push_back(n);
        if (o_.pick == "round_robin") {
            for (std::size_t k = 0; k < inst_.size(); ++k) {
                const std::size_t i = (B_.cursor + k) % inst_.size();
                if (!fresh_ones.empty() && !fresh(inst_[i])) continue;
                B_.cursor = (i + 1) % inst_.size();
                return inst_[i];
            }
        }
        const std::vector<long long>& among = fresh_ones.empty() ? inst_ : fresh_ones;
        const bool by_gap = o_.pick == "gap" || o_.pick == "gap_utility";
        const double inf = std::numeric_limits<double>::infinity();
        long long best = among[0];
        double best_p = -inf;
        long long best_age = -1;
        bool first = true;
        for (long long n : among) {
            double p;
            if (by_gap) {
                const auto it = B_.records.find(n);
                p = (it == B_.records.end() || it->second.cv > 0.0) ? inf
                                                                     : weight(n) * (it->second.f - reference(n));
                if (o_.pick == "gap_utility" && p > 0.0 && p < inf) p *= utility(n);   // FAM-12
            } else {
                p = weight(n) * static_cast<double>(age(n));
            }
            const long long a = age(n);
            const bool take = first || p > best_p || (by_gap && p == best_p && a > best_age);
            if (take) {
                best = n;
                best_p = p;
                best_age = a;
                first = false;
            }
        }
        return best;
    }

    // FAM-7
    void piece(long long n, long long budget)
    {
        const long long p = static_cast<long long>(B_.pieces.size());
        B_.pieces.push_back(PieceLog{n, B_.used, budget, false});
        B_.touched[n] = p;
        scalar::Start st;
        const auto it = B_.records.find(n);
        if (it != B_.records.end()) {
            st.has_x0 = true;
            st.x0 = it->second.x;
        }
        const scalar::Result r =
            scalar::run(o_.solver, problem(n), o_.eq_tol, false, 0.0, o_.pop_size, budget,
                        scalar::strategy::start_seed(o_.seed, static_cast<std::uint64_t>(p)), o_.knobs, {}, 0, st);
        B_.used += r.used;
        const double gap_before = o_.pick == "gap_utility" ? gap(n) : 0.0;     // FAM-12
        bool taken = false;
        if (!r.variables.empty() && !r.objectives.empty())
            taken = offer(n, r.variables[0], r.objectives[0][0], r.cv.empty() ? 0.0 : r.cv[0], B_.run, "run", -1);
        B_.pieces.back().taken = taken;
        if (o_.pick == "gap_utility") update_utility(n, gap_before);
        if (taken) begin_closure(n);
    }

    // FAM-12: the gap f(n) - reference(n) of n's record; infinite without a feasible one
    double gap(long long n)
    {
        const auto it = B_.records.find(n);
        if (it == B_.records.end() || it->second.cv > 0.0) return std::numeric_limits<double>::infinity();
        return it->second.f - reference(n);
    }

    // FAM-12: MOEA/D-DRA's Step 5 after a piece on n, its gap before the piece given
    void update_utility(long long n, double before)
    {
        const double after = gap(n);
        double& pi = B_.utility.emplace(n, 1.0).first->second;
        if (std::isinf(before) && !std::isinf(after)) {                        // its first feasible record
            pi = 1.0;
            return;
        }
        const double delta = (!std::isinf(before) && before > 0.0) ? (before - after) / before : 0.0;
        if (delta > 0.001)
            pi = 1.0;
        else
            pi = (std::max)(0.0, o_.utility_factor + (1.0 - o_.utility_factor) * delta / 0.001) * pi;
    }

    void save(const std::string& stopped)
    {
        save_bank(o_.bank_file, B_, total(), inst_.size(), stopped);
    }
};

// the campaign over the family from the bank B (FAM-7 ... FAM-10)
inline Report run(const Family& F, Bank& B, const Options& o)
{
    if (o.budget_more <= 0 && o.budget <= 0) throw std::invalid_argument("family: budget > 0");
    if (o.piece_budget <= 0) throw std::invalid_argument("family: piece_budget > 0");
    Campaign c(F, B, o);
    return c.run();
}

}   // namespace mootation::campaign::fam
