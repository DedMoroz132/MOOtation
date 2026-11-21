#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// ASSEMBLY FROM PARTS AND THE BOTTLENECK ANALYSIS (task 32, layer 3). The
// solution of a large instance is often assembled from parts (pieces, blocks,
// sub-solutions) that are searched apart and kept in a table; one better part
// makes many instances better at once. This header holds the table of parts,
// the analysis of which part limits which instance and what its improvement
// is worth in the family's total objective (family.hpp), and the loop that
// searches parts and carries what they give into the family's bank. The
// assembly itself is the problem's hook (a dynamic programme, a greedy
// choice, a search over schemes). The table lives here, in campaign/, not in
// the core: a part's search is a run of any strategy on the part's problem,
// which the core has. Nothing outside this header changes: without a table of
// parts the family's campaign and every run are those of before.
//
// THE PARTS (Components): the keys of the parts (a key is a vector of whole
// numbers; the keys distinct), and two hooks:
//   component_problem(key)  the problem of the part (scalar::Problem, one
//                           objective): its search is scalar::run of any
//                           strategy on it (a resumable one, layer 1's
//                           portfolio_run ...);
//   assemble(n, view)       {ok, f, x, used_keys, cost}: instance n made from
//                           the parts the view shows. f its value, x its
//                           point in instance n's problem (empty: nothing for
//                           the bank), used_keys the parts it used, cost the
//                           assembly's declared cost in used().
//
// THE TABLE (Table): one record per key, the best one found — the part's
// point, its quality q (the value of the part's problem at the point; lower
// is better) and cv, where it came from (run, given), the piece and the
// loop's used() when it came — and the loop's state. It is written to a file
// and read back like the family's bank (save_table, load_table: family.hpp's
// file conventions, format "mootation-component-table").
//
// THE SENSITIVITY (sensitivity): for instance n and part k, the change
// delta_k f(n) = f(n) - f_eps(n), f_eps(n) the assembly of n from the view in
// which part k's quality is q(k) - eps (one more assemble, no search); the
// contribution of k is sum_n w(n) delta_k f(n), and the parts are ranked by
// it: the first is the part whose improvement is worth most in the total.
//
// THE LOOP (run): pieces of one budget, each a run of the solver on the
// problem of the part picked (pick = round_robin | sensitivity), started from
// the part's record; a better part goes into the table, the instances it
// touches are assembled again, their points given to the family's bank and
// the bank's closure (family.hpp FAM-5) run from them. With diminish_ratio or
// diminish_abs (off by default) the loop also ends when a round of pieces
// gains too little in the total objective (COMP-12).
//
// READINGS:
//   COMP-1. A part's record is better than another when it is less
//     violating, then when its q is lower; an equal one is not taken (the
//     table keeps the first; family.hpp FAM-2). A failed evaluation is never
//     taken. The keys are kept in ascending (lexicographic) order: the
//     round_robin goes up them, the ties of the picks and of the ranking go
//     to the lower key.
//   COMP-2. THE VIEW. assemble reads the table only through its view: has(k),
//     quality(k), cv(k), point(k), keys() (the parts present, ascending). The
//     view of the sensitivity differs from the table in one thing: the
//     quality of one part is q(k) - eps. An assembly that keeps its own copy
//     of the qualities is not seen by the analysis.
//   COMP-3. THE ASSEMBLY. Its used_keys are kept sorted and each once; a key
//     that is not a part of its view is refused, as a cost that is not a
//     finite number >= 0. ok false: instance n cannot be assembled from this
//     view (a part missing, none feasible). Every call of assemble counts:
//     its declared cost goes to used() and the call to components_assembles.
//   COMP-4. THE SENSITIVITY. Instances in ascending n: one assemble of n on
//     the table's view (f(n)); then, for every part k of scope — all: every
//     part of the table; used: the parts n's assembly used, the others'
//     delta 0 without a call — one assemble on the view with q(k) - eps. An
//     instance whose assembly fails (ok false or f not finite) adds nothing
//     (failed); a changed view whose assembly fails gives delta 0. A part's
//     contribution adds w(n) delta_k f(n) in ascending n; its deltas list
//     the instances where delta is not 0. The ranking: every part of the
//     table, the largest contribution first, ties to the lower key. Positive
//     means the total goes down when the part gets better. eps > 0 is an
//     absolute amount in the quality's units.
//   COMP-5. THE PICK. round_robin: the keys in turn (the cursor in the
//     table). sensitivity: a key without a part first (the one untouched
//     longest, then the lower key); else the sensitivity of the table
//     (COMP-4, its cost in used()) and the largest contribution, ties to the
//     part untouched longer, then to the lower key. The key picked is kept
//     in the table until its piece runs (a loop stopped between the pick and
//     the piece does not pick again when it goes on).
//   COMP-6. THE PIECE. Piece p (0, 1, ... over the whole loop, its file
//     included) runs the solver on component_problem(key) with the budget
//     min(piece_budget, what is left of the loop's), the seed
//     strategy::start_seed(seed, p) (piece 0: the loop's seed), the knobs,
//     and the start x0 = the part's point where the table has one. Its answer
//     (the best point of its result) is offered to the table as source run.
//   COMP-7. THE INSTANCES TOUCHED. When the table takes a part k: every
//     instance where k is new or reassemble = all; else the instances whose
//     last assembly used k and those without an assembly. They wait in the
//     table (pending, ascending n) and are assembled one by one, the budget
//     checked before each; an assembly that is ok replaces the instance's
//     last one, one that fails removes it. With a bank, an assembled point x
//     is given to it (Campaign::give, the record's source "assembled", the
//     count on the path given; certified or evaluated, FAM-3, FAM-4); after
//     the instances pending, the bank's closure from the records that got
//     better, within what is left of the loop's budget (a closure the budget
//     ended stays in the bank, FAM-5).
//   COMP-8. THE COUNT (stop_rule.hpp: one count, used()). The loop's used()
//     is the sum of its pieces' Result::used, of the evaluations of parts
//     given, of every assembly's declared cost (the sensitivity's included),
//     and of what the bank spent in the loop (its evaluations or certify's
//     costs, its closures). The bank's own used() counts the bank's part of
//     it, as family.hpp does.
//   COMP-9. THE FILE and going on. save_table writes the table (every part,
//     the state, the counts, the last ranking, the last call's options) as
//     one JSON text through <file>.tmp and a rename (family.hpp FAM-9);
//     load_table reads it back exactly. The loop saves the bank's file (when
//     it has one) and then the table's, every save_every pieces and at the
//     end. A loop stopped between pieces (max_pieces) or by its budget
//     (before an assembly, in the bank's closure, after the pick), loaded
//     and run on, is the loop run in one piece, bit for bit.
//   COMP-10. THE SETTINGS (Report::settings, and the same in the file):
//     components_keys, components_missing, components_pieces,
//     components_used, components_offered / components_accepted ({"run",
//     "given"}), components_assembles, components_sensitivities,
//     components_bank_given, components_ranking ([key, contribution] of the
//     last sensitivity), components_gains (with the diminishing returns'
//     measure: COMP-12), components_stopped_by (budget, pieces, diminishing,
//     or done: no keys).
//   COMP-11. OFF BY DEFAULT. Without keys the loop does nothing (stopped_by
//     done): it does not call a hook, touch the bank or the problems, nor
//     write the bank's file, and the family's campaign is that of before.
//   COMP-12. Off by default (task 32, layer 5; diminish_ratio or
//     diminish_abs > 0). THE DIMINISHING RETURNS (scalar/stop_rule.hpp
//     STOP-8), as the family's FAM-11. The round is round_pieces pieces (0,
//     the default: as many as there are keys, so that round_robin gives each
//     part one piece), each with its assemblies and the bank's closure; it
//     ends at the top of the loop, after the assemblies and the closure of
//     its last piece. The total objective is the family's: with a bank the
//     bank's total (FAM-2) over the instances that have a record, all
//     feasible to be measured; without one, sum w(n) f(n) of the instances'
//     last assemblies that were ok (COMP-7), in ascending n. A round's gain
//     is measured where the total before and after it is over the same
//     instances (at least one): the total before less the total after. The
//     loop ends after the round that ends the rule (stopped_by =
//     diminishing). The measure lives in the table and its file (state's
//     "diminish", as the bank's): a loop loaded and run on goes on with it,
//     the loop in one piece (COMP-9); a call whose measure the rule ended, or
//     a table without one, begins a new measure at its first piece; a call
//     without diminish_* drops it (the table and its file then as before).
//     Without keys the loop does nothing, as COMP-11 says.
//   COMP-13. Off by default (task 33; table_journal = true, with a
//     table_file). THE JOURNAL. A table of many parts is a long text, and
//     its parts change one at a time. With the journal the loop writes the
//     table's file whole (as COMP-9, the same text) at its first save, and
//     at the later saves appends to <file>.journal what changed: the parts
//     the table took since the last save, the pieces run since, and the
//     settings, options and state. When the journal would grow past the
//     whole file's length the save writes the file whole again and removes
//     the journal (so the whole file is written again only after a journal
//     as long as it).
//     The journal is JSON lines, each a JSON object of one kind:
//       {"format": "mootation-component-journal", "version": 1,
//        "base_bytes": N, "base_sha256": "<hex>"}: the first line; the
//         journal goes with the table's file of N bytes and that SHA-256
//         only (another file: a journal left from before, not read);
//       {"part": {...}}: a part's record, as in "records" (it replaces the
//         part of its key);
//       {"pieces_from": i, "pieces": [...]}: the pieces from the i-th on (the
//         earlier ones kept);
//       {"head": {"settings": ..., "options": ..., "state": ...}}: the end of
//         a save, the three as in the file (they replace them).
//     A save is one append of its lines, the head last. load_table reads the
//     file, then the journal's saves that end with a head, up to the first
//     line that is not whole (no newline, or not JSON: a write cut off);
//     the lines after the last head are not read. A write cut off anywhere
//     therefore reads as the last whole save: in an append, the saves
//     before it; in the whole file's (.tmp and a rename, COMP-9), the file
//     and journal before it; after the rename, the new file (the old
//     journal goes with the old file). The table so read is the table, and
//     its table_text the whole file's text, byte for byte. A call that goes
//     on from it writes the file whole at its first save, so that no
//     journal is appended after a part cut off. A script reads the table
//     with Python's json and hashlib alone: the file, then the journal
//     where its first line names the file, line by line in this order.
//     Without table_journal the file and its writes are those of before,
//     and no journal is written; load_table reads a file without a journal
//     as before.
//
// Options: solver ("nelder_mead"), knobs, eq_tol (1e-4), pop_size (100),
// budget (the loop's used() in all, the file's included), piece_budget,
// max_pieces (0: none), pick (round_robin), epsilon (pick = sensitivity:
// > 0), scope (all), reassemble (used), closure_rounds (the bank's, 0: none),
// seed (1), table_file ("": none), bank_file ("": none; needs a bank),
// save_every (1), diminish_ratio, diminish_abs (0: off), round_pieces (0: the
// number of keys; read with the diminishing returns), table_journal (false;
// COMP-13).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "../json.hpp"
#include "../scalar/scalar.hpp"
#include "../store/sha256.hpp"
#include "family.hpp"

namespace mootation::campaign::comp {

using scalar::Problem;
using scalar::Vec;
using Key = std::vector<long long>;

inline std::string key_text(const Key& k)
{
    std::string s = "[";
    for (std::size_t i = 0; i < k.size(); ++i) s += (i ? ", " : "") + std::to_string(k[i]);
    return s + "]";
}

struct Part {
    Vec x;
    double q = std::numeric_limits<double>::infinity();
    double cv = std::numeric_limits<double>::infinity();
    std::string source;          // run, given
    long long piece = -1;        // the piece it came in (-1: before the first)
    double used = 0.0;           // the loop's used() when the table took it
};

// COMP-2: the table as assemble sees it
class View {
public:
    explicit View(const std::map<Key, Part>& parts) : parts_(&parts) {}
    // the view in which part `bettered` has the quality q - by (the sensitivity)
    View(const std::map<Key, Part>& parts, const Key& bettered, double by) : parts_(&parts), bettered_(&bettered), by_(by) {}

    bool has(const Key& k) const { return parts_->count(k) != 0; }
    double quality(const Key& k) const { return bettered_ && k == *bettered_ ? at(k).q - by_ : at(k).q; }
    double cv(const Key& k) const { return at(k).cv; }
    const Vec& point(const Key& k) const { return at(k).x; }
    std::vector<Key> keys() const
    {
        std::vector<Key> ks;
        for (const auto& kv : *parts_) ks.push_back(kv.first);
        return ks;
    }

private:
    const std::map<Key, Part>* parts_;
    const Key* bettered_ = nullptr;
    double by_ = 0.0;

    const Part& at(const Key& k) const
    {
        const auto it = parts_->find(k);
        if (it == parts_->end()) throw std::out_of_range("components: the table has no part " + key_text(k));
        return it->second;
    }
};

// assemble's answer (COMP-3)
struct Assembled {
    bool ok = false;
    double f = std::numeric_limits<double>::infinity();
    Vec x;                         // instance n's point (empty: nothing for the bank)
    std::vector<Key> used_keys;
    double cost = 0.0;             // the assembly's declared cost in used()
};

struct Components {
    std::vector<Key> keys;
    std::function<Problem(const Key& key)> component_problem;
    std::function<Assembled(long long n, const View& view)> assemble;
};

// COMP-4
struct Contribution {
    Key key;
    double value = 0.0;                                     // sum_n w(n) (f(n) - f_eps(n))
    std::vector<std::pair<long long, double>> deltas;       // (n, f(n) - f_eps(n)) where not 0, ascending n
};

struct Sensitivity {
    std::vector<Contribution> ranked;                       // the largest value first, ties to the lower key
    std::map<long long, double> f;                          // the instances' assembled values
    std::vector<long long> failed;                          // instances assemble could not make
    long long assembles = 0;
    double cost = 0.0;
};

// a piece of the loop, as the table keeps it
struct PieceLog {
    Key key;
    double used_at = 0.0;        // the loop's used() before the piece
    long long budget = 0;
    bool taken = false;          // the table took its answer
};

// an instance's last assembly that was ok
struct Assembly {
    double f = 0.0;
    std::vector<Key> used_keys;
};

struct Table {
    std::map<Key, Part> parts;
    // the loop's state
    double used = 0.0;
    std::vector<PieceLog> pieces;
    std::size_t cursor = 0;                              // round_robin
    std::map<Key, long long> touched;                    // the last piece given to the part
    fam::Counts run, given;
    std::map<long long, Assembly> assembled;
    long long assembles = 0, sensitivities = 0, bank_given = 0;
    std::vector<long long> pending;                      // instances to assemble (COMP-7), ascending
    bool has_next = false;                               // a key picked, its piece not run (COMP-5)
    Key next;
    std::vector<std::pair<Key, double>> ranking;         // the last sensitivity's
    json::Value options = json::Value::object();         // the last call's (for the scripts)
    fam::Measure measure;                                // COMP-12 (off: none)
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
    double epsilon = 0.0;
    std::string scope = "all";
    std::string reassemble = "used";
    long long closure_rounds = 0;
    unsigned seed = 1;
    std::string table_file;
    std::string bank_file;
    long long save_every = 1;
    double diminish_ratio = 0.0;                         // COMP-12 (0: off)
    double diminish_abs = 0.0;
    long long round_pieces = 0;                          // 0: the number of keys
    bool table_journal = false;                          // COMP-13 (false: the file whole at every save)
};

struct Report {
    std::string stopped_by;
    std::vector<std::pair<std::string, std::string>> settings;
};

namespace detail {

inline const char* const kFormat = "mootation-component-table";
constexpr int kVersion = 1;

using fam::detail::field;
using fam::detail::number_of;
using fam::detail::numbers;
using fam::detail::whole_of;
using fam::detail::wholes;

inline Key key_of(const json::Value& v, const std::string& what)
{
    if (v.kind != json::Value::Kind::Array) throw std::invalid_argument("component table: " + what + " is not a key");
    Key k;
    for (const auto& i : v.items) k.push_back(whole_of(i, what));
    return k;
}

// the instances of a family, ascending, each once (FAM-1)
inline std::vector<long long> instances_of(const fam::Family& F)
{
    std::vector<long long> inst = F.instances;
    std::sort(inst.begin(), inst.end());
    if (std::adjacent_find(inst.begin(), inst.end()) != inst.end())
        throw std::invalid_argument("components: an instance given twice");
    return inst;
}

inline std::vector<double> weights_of(const fam::Family& F, const std::vector<long long>& inst)
{
    std::vector<double> w;
    for (long long n : inst) {
        const double v = F.weight ? F.weight(n) : 1.0;
        if (!std::isfinite(v)) throw std::invalid_argument("components: w(" + std::to_string(n) + ") is not finite");
        w.push_back(v);
    }
    return w;
}

// COMP-3
inline Assembled assemble_checked(const Components& C, long long n, const View& v)
{
    Assembled a = C.assemble(n, v);
    if (!std::isfinite(a.cost) || a.cost < 0.0)
        throw std::invalid_argument("components: the cost of instance " + std::to_string(n) +
                                    "'s assembly is not a finite number >= 0");
    std::sort(a.used_keys.begin(), a.used_keys.end());
    a.used_keys.erase(std::unique(a.used_keys.begin(), a.used_keys.end()), a.used_keys.end());
    for (const Key& k : a.used_keys)
        if (!v.has(k))
            throw std::invalid_argument("components: instance " + std::to_string(n) + "'s assembly used part " +
                                        key_text(k) + ", not in the table");
    return a;
}

// COMP-4
inline Sensitivity sensitivity_of(const std::vector<long long>& inst, const std::vector<double>& w,
                                  const Components& C, const std::map<Key, Part>& parts, double eps,
                                  const std::string& scope)
{
    Sensitivity S;
    const View base(parts);
    std::map<Key, Contribution> by_key;
    for (const auto& kv : parts) by_key[kv.first].key = kv.first;
    const std::vector<Key> all = base.keys();
    for (std::size_t i = 0; i < inst.size(); ++i) {
        const long long n = inst[i];
        const Assembled a = assemble_checked(C, n, base);
        ++S.assembles;
        S.cost += a.cost;
        if (!a.ok || !std::isfinite(a.f)) {
            S.failed.push_back(n);
            continue;
        }
        S.f[n] = a.f;
        for (const Key& k : scope == "used" ? a.used_keys : all) {
            const Assembled b = assemble_checked(C, n, View(parts, k, eps));
            ++S.assembles;
            S.cost += b.cost;
            if (!b.ok || !std::isfinite(b.f)) continue;
            const double d = a.f - b.f;
            if (d == 0.0) continue;
            Contribution& c = by_key[k];
            c.value += w[i] * d;
            c.deltas.push_back({n, d});
        }
    }
    for (auto& kv : by_key) S.ranked.push_back(std::move(kv.second));
    std::stable_sort(S.ranked.begin(), S.ranked.end(),
                     [](const Contribution& a, const Contribution& b) { return a.value > b.value; });
    return S;
}

inline void check_sensitivity(double eps, const std::string& scope)
{
    if (!(eps > 0.0) || !std::isfinite(eps)) throw std::invalid_argument("components: epsilon > 0");
    if (scope != "all" && scope != "used") throw std::invalid_argument("components: scope = all | used, not " + scope);
}

inline json::Value counts_json(const Table& T, long long fam::Counts::*c)
{
    json::Value o = json::Value::object();
    o.set("run", json::Value::integer(T.run.*c));
    o.set("given", json::Value::integer(T.given.*c));
    return o;
}

inline json::Value ranking_json(const Table& T)
{
    json::Value a = json::Value::array();
    for (const auto& r : T.ranking) {
        json::Value p = json::Value::array();
        p.items.push_back(wholes(r.first));
        p.items.push_back(json::Value::number(r.second));
        a.items.push_back(p);
    }
    return a;
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
    v.set("epsilon", json::Value::number(o.epsilon));
    v.set("scope", json::Value::string(o.scope));
    v.set("reassemble", json::Value::string(o.reassemble));
    v.set("closure_rounds", json::Value::integer(o.closure_rounds));
    v.set("seed", json::Value::integer(o.seed));
    v.set("save_every", json::Value::integer(o.save_every));
    if (o.diminish_ratio > 0.0 || o.diminish_abs > 0.0) {                     // COMP-12: only where asked
        v.set("diminish_ratio", json::Value::number(o.diminish_ratio));
        v.set("diminish_abs", json::Value::number(o.diminish_abs));
        v.set("round_pieces", json::Value::integer(o.round_pieces));
    }
    return v;
}

inline json::Value part_json(const Key& k, const Part& p)
{
    json::Value o = json::Value::object();
    o.set("key", wholes(k));
    o.set("q", json::Value::number(p.q));
    o.set("cv", json::Value::number(p.cv));
    o.set("source", json::Value::string(p.source));
    o.set("piece", json::Value::integer(p.piece));
    o.set("used", json::Value::number(p.used));
    o.set("x", numbers(p.x));
    return o;
}

}   // namespace detail

// COMP-4: the sensitivity of the family's total to every part of the table
inline Sensitivity sensitivity(const fam::Family& F, const Components& C, const std::map<Key, Part>& parts,
                               double epsilon, const std::string& scope = "all")
{
    detail::check_sensitivity(epsilon, scope);
    if (!C.assemble) throw std::invalid_argument("components: no assemble");
    const std::vector<long long> inst = detail::instances_of(F);
    return detail::sensitivity_of(inst, detail::weights_of(F, inst), C, parts, epsilon, scope);
}

// COMP-10: the settings of a table of `keys` keys
inline std::vector<std::pair<std::string, std::string>> settings_of(const Table& T, std::size_t keys,
                                                                  const std::string& stopped_by)
{
    std::vector<std::pair<std::string, std::string>> s;
    s.push_back({"components_keys", std::to_string(keys)});
    s.push_back({"components_missing", std::to_string(keys - (std::min)(keys, T.parts.size()))});
    s.push_back({"components_pieces", std::to_string(T.pieces.size())});
    s.push_back({"components_used", json::float_repr(T.used)});
    s.push_back({"components_offered", json::dump(detail::counts_json(T, &fam::Counts::offered))});
    s.push_back({"components_accepted", json::dump(detail::counts_json(T, &fam::Counts::accepted))});
    s.push_back({"components_assembles", std::to_string(T.assembles)});
    s.push_back({"components_sensitivities", std::to_string(T.sensitivities)});
    s.push_back({"components_bank_given", std::to_string(T.bank_given)});
    s.push_back({"components_ranking", json::dump(detail::ranking_json(T))});
    if (T.measure.on) s.push_back({"components_gains", json::dump(T.measure.rule.gains_json())});   // COMP-12
    if (!stopped_by.empty()) s.push_back({"components_stopped_by", stopped_by});
    return s;
}

namespace detail {

// COMP-9: the file's settings, state and pieces (from the from-th on; COMP-13)
inline json::Value settings_value(const Table& T, std::size_t keys, const std::string& stopped_by)
{
    json::Value s = json::Value::object();
    for (const auto& [k, v] : settings_of(T, keys, stopped_by))
        s.set(k, json::parse(k == "components_stopped_by" ? json::dump(json::Value::string(v)) : v));
    return s;
}

inline json::Value state_value(const Table& T)
{
    json::Value st = json::Value::object(), touched = json::Value::array(), assembled = json::Value::array();
    for (const auto& [k, p] : T.touched) {
        json::Value q = json::Value::array();
        q.items.push_back(wholes(k));
        q.items.push_back(json::Value::integer(p));
        touched.items.push_back(q);
    }
    for (const auto& [n, a] : T.assembled) {
        json::Value q = json::Value::array(), ks = json::Value::array();
        for (const Key& k : a.used_keys) ks.items.push_back(wholes(k));
        q.items.push_back(json::Value::integer(n));
        q.items.push_back(json::Value::number(a.f));
        q.items.push_back(ks);
        assembled.items.push_back(q);
    }
    st.set("used", json::Value::number(T.used));
    st.set("cursor", json::Value::integer(static_cast<std::int64_t>(T.cursor)));
    st.set("touched", touched);
    st.set("pending", wholes(T.pending));
    st.set("next", T.has_next ? wholes(T.next) : json::Value::null());
    st.set("assembled", assembled);
    if (T.measure.on) st.set("diminish", fam::detail::measure_json(T.measure));   // COMP-12: only where begun
    return st;
}

inline json::Value pieces_value(const Table& T, std::size_t from)
{
    json::Value pieces = json::Value::array();
    for (std::size_t i = from; i < T.pieces.size(); ++i) {
        const PieceLog& p = T.pieces[i];
        json::Value q = json::Value::array();
        q.items.push_back(wholes(p.key));
        q.items.push_back(json::Value::number(p.used_at));
        q.items.push_back(json::Value::integer(p.budget));
        q.items.push_back(json::Value::boolean(p.taken));
        pieces.items.push_back(q);
    }
    return pieces;
}

}   // namespace detail

// COMP-9: the table as the file's text — one part a line
inline std::string table_text(const Table& T, std::size_t keys, const std::string& stopped_by)
{
    std::string t = "{\n";
    t += "\"format\": \"" + std::string(detail::kFormat) + "\",\n";
    t += "\"version\": " + std::to_string(detail::kVersion) + ",\n";
    t += "\"settings\": " + json::dump(detail::settings_value(T, keys, stopped_by)) + ",\n";
    t += "\"options\": " + json::dump(T.options) + ",\n";
    t += "\"state\": " + json::dump(detail::state_value(T)) + ",\n";
    t += "\"pieces\": " + json::dump(detail::pieces_value(T, 0)) + ",\n";
    t += "\"records\": [";
    bool first = true;
    for (const auto& [k, p] : T.parts) {
        t += first ? "\n" : ",\n";
        t += json::dump(detail::part_json(k, p));
        first = false;
    }
    t += "\n]\n}\n";
    return t;
}

inline void save_table(const std::string& path, const Table& T, std::size_t keys, const std::string& stopped_by = "")
{
    fam::detail::write_atomic(path, table_text(T, keys, stopped_by));
}

namespace detail {

// COMP-9, COMP-13: the settings, options and state of the file's text or of a journal's head (every field anew)
inline void read_head(const json::Value& v, Table& T)
{
    const json::Value& s = field(v, "settings");
    auto counts = [&](long long fam::Counts::*c, const std::string& key) {
        const json::Value& o = field(s, key);
        T.run.*c = whole_of(field(o, "run"), key);
        T.given.*c = whole_of(field(o, "given"), key);
    };
    counts(&fam::Counts::offered, "components_offered");
    counts(&fam::Counts::accepted, "components_accepted");
    T.assembles = whole_of(field(s, "components_assembles"), "components_assembles");
    T.sensitivities = whole_of(field(s, "components_sensitivities"), "components_sensitivities");
    T.bank_given = whole_of(field(s, "components_bank_given"), "components_bank_given");
    T.ranking.clear();
    for (const auto& r : field(s, "components_ranking").items) {
        if (r.items.size() != 2) throw std::invalid_argument("component table: the ranking is a list of [key, value]");
        T.ranking.push_back({key_of(r.items[0], "a ranking's key"), number_of(r.items[1], "a ranking's value")});
    }
    T.options = field(v, "options");
    const json::Value& st = field(v, "state");
    T.used = number_of(field(st, "used"), "used");
    T.cursor = static_cast<std::size_t>(whole_of(field(st, "cursor"), "cursor"));
    T.touched.clear();
    for (const auto& p : field(st, "touched").items) {
        if (p.items.size() != 2) throw std::invalid_argument("component table: touched is a list of [key, piece]");
        T.touched[key_of(p.items[0], "touched")] = whole_of(p.items[1], "touched");
    }
    T.pending.clear();
    for (const auto& n : field(st, "pending").items) T.pending.push_back(whole_of(n, "pending"));
    const json::Value& next = field(st, "next");
    T.has_next = false;
    T.next.clear();
    if (next.kind != json::Value::Kind::Null) {
        T.has_next = true;
        T.next = key_of(next, "next");
    }
    T.assembled.clear();
    for (const auto& a : field(st, "assembled").items) {
        if (a.items.size() != 3) throw std::invalid_argument("component table: assembled is a list of [n, f, keys]");
        Assembly as;
        as.f = number_of(a.items[1], "an assembly's f");
        for (const auto& k : a.items[2].items) as.used_keys.push_back(key_of(k, "an assembly's key"));
        T.assembled[whole_of(a.items[0], "an assembly's n")] = std::move(as);
    }
    T.measure = fam::Measure();
    if (const json::Value* d = st.find("diminish")) T.measure = fam::detail::measure_of(*d, "component table: diminish");
}

// the pieces of a list [[key, used_at, budget, taken], ...], after those of T
inline void read_pieces(const json::Value& pieces, Table& T)
{
    for (const auto& p : pieces.items) {
        if (p.items.size() != 4 || p.items[3].kind != json::Value::Kind::Bool)
            throw std::invalid_argument("component table: a piece is [key, used_at, budget, taken]");
        T.pieces.push_back(PieceLog{key_of(p.items[0], "a piece's key"), number_of(p.items[1], "piece used_at"),
                                    whole_of(p.items[2], "piece budget"), p.items[3].b});
    }
}

inline std::pair<Key, Part> part_of(const json::Value& r)
{
    Part part;
    const Key k = key_of(field(r, "key"), "a part's key");
    part.q = number_of(field(r, "q"), "a part's q");
    part.cv = number_of(field(r, "cv"), "a part's cv");
    const json::Value& src = field(r, "source");
    if (src.kind != json::Value::Kind::String) throw std::invalid_argument("component table: a part's source");
    part.source = src.s;
    part.piece = whole_of(field(r, "piece"), "a part's piece");
    part.used = number_of(field(r, "used"), "a part's used");
    for (const auto& x : field(r, "x").items) part.x.push_back(number_of(x, "a part's x"));
    return {k, std::move(part)};
}

// COMP-13
inline const char* const kJournalFormat = "mootation-component-journal";
constexpr int kJournalVersion = 1;

inline std::string sha256_hex(const std::string& text)
{
    const auto h = store::sha256(reinterpret_cast<const std::uint8_t*>(text.data()), text.size());
    const char* const hex = "0123456789abcdef";
    std::string s;
    for (const std::uint8_t b : h) {
        s += hex[b >> 4];
        s += hex[b & 15];
    }
    return s;
}

// the journal's first line: the file's text it goes with
inline std::string journal_header(std::size_t base_bytes, const std::string& base_sha256)
{
    json::Value h = json::Value::object();
    h.set("format", json::Value::string(kJournalFormat));
    h.set("version", json::Value::integer(kJournalVersion));
    h.set("base_bytes", json::Value::integer(static_cast<std::int64_t>(base_bytes)));
    h.set("base_sha256", json::Value::string(base_sha256));
    return json::dump(h) + "\n";
}

// one save's lines: the parts changed (ascending keys), the pieces from the from-th on, the head
inline std::string journal_save(const Table& T, std::size_t keys, const std::string& stopped_by,
                                const std::set<Key>& changed, std::size_t from)
{
    std::string t;
    for (const Key& k : changed) {
        json::Value part_line = json::Value::object();
        part_line.set("part", part_json(k, T.parts.at(k)));
        t += json::dump(part_line) + "\n";
    }
    if (T.pieces.size() > from) {
        json::Value pieces_line = json::Value::object();
        pieces_line.set("pieces_from", json::Value::integer(static_cast<std::int64_t>(from)));
        pieces_line.set("pieces", pieces_value(T, from));
        t += json::dump(pieces_line) + "\n";
    }
    json::Value head = json::Value::object(), head_line = json::Value::object();
    head.set("settings", settings_value(T, keys, stopped_by));
    head.set("options", T.options);
    head.set("state", state_value(T));
    head_line.set("head", head);
    return t + json::dump(head_line) + "\n";
}

inline std::string read_text(const std::string& path, bool& found)
{
    std::ifstream f(store::file::path_of(path), std::ios::binary);
    found = static_cast<bool>(f);
    std::stringstream ss;
    if (found) ss << f.rdbuf();
    return ss.str();
}

}   // namespace detail

// COMP-9: a table from its file's text
inline Table table_of_text(const std::string& text)
{
    using detail::field;
    using detail::whole_of;
    const json::Value v = json::parse(text);
    if (v.kind != json::Value::Kind::Object) throw std::invalid_argument("component table: not a JSON object");
    const json::Value& fmt = field(v, "format");
    if (fmt.kind != json::Value::Kind::String || fmt.s != detail::kFormat)
        throw std::invalid_argument("component table: the format is not " + std::string(detail::kFormat));
    if (whole_of(field(v, "version"), "version") != detail::kVersion)
        throw std::invalid_argument("component table: version " + std::to_string(whole_of(field(v, "version"), "version")) +
                                    " (this library reads " + std::to_string(detail::kVersion) + ")");
    Table T;
    detail::read_head(v, T);
    detail::read_pieces(field(v, "pieces"), T);
    for (const auto& r : field(v, "records").items) {
        std::pair<Key, Part> kp = detail::part_of(r);
        if (!T.parts.emplace(kp.first, std::move(kp.second)).second)
            throw std::invalid_argument("component table: two parts of key " + key_text(kp.first));
    }
    return T;
}

// COMP-13: the journal of the table's file
inline std::string journal_path(const std::string& table_file) { return table_file + ".journal"; }

// COMP-13: T, read from the file's text `base`, brought on by the journal's text: its saves that end with a head, up
// to its first line that is not whole; a journal of another file is not read. The saves read (0: none).
inline long long apply_journal(Table& T, const std::string& base, const std::string& journal)
{
    using detail::field;
    using detail::whole_of;
    std::size_t at = 0;
    json::Value v;
    auto next_line = [&]() {                                       // false: no whole line is left
        const std::size_t end = journal.find('\n', at);
        if (end == std::string::npos) return false;
        try {
            v = json::parse(journal.substr(at, end - at));
        } catch (const std::exception&) {
            return false;
        }
        at = end + 1;
        return v.kind == json::Value::Kind::Object;
    };
    if (!next_line()) return 0;
    const json::Value* fmt = v.find("format");
    if (!fmt || fmt->kind != json::Value::Kind::String || fmt->s != detail::kJournalFormat)
        throw std::invalid_argument("component journal: the format is not " + std::string(detail::kJournalFormat));
    if (whole_of(field(v, "version"), "the journal's version") != detail::kJournalVersion)
        throw std::invalid_argument("component journal: version " +
                                    std::to_string(whole_of(field(v, "version"), "the journal's version")) +
                                    " (this library reads " + std::to_string(detail::kJournalVersion) + ")");
    const json::Value& sha = field(v, "base_sha256");
    if (whole_of(field(v, "base_bytes"), "base_bytes") != static_cast<long long>(base.size()) ||
        sha.kind != json::Value::Kind::String || sha.s != detail::sha256_hex(base))
        return 0;                                                  // a journal left from another file
    std::vector<std::pair<Key, Part>> parts;                       // a save's, until its head
    std::vector<std::pair<long long, json::Value>> pieces;
    long long saves = 0;
    while (next_line()) {
        if (const json::Value* rec = v.find("part")) {
            parts.push_back(detail::part_of(*rec));
        } else if (const json::Value* pf = v.find("pieces_from")) {
            pieces.push_back({whole_of(*pf, "pieces_from"), field(v, "pieces")});
        } else if (const json::Value* head = v.find("head")) {
            for (auto& kp : parts) T.parts[kp.first] = std::move(kp.second);
            for (const auto& fp : pieces) {
                if (fp.first < 0 || fp.first > static_cast<long long>(T.pieces.size()))
                    throw std::invalid_argument("component journal: pieces from " + std::to_string(fp.first) +
                                                " where the table has " + std::to_string(T.pieces.size()));
                T.pieces.resize(static_cast<std::size_t>(fp.first));
                detail::read_pieces(fp.second, T);
            }
            detail::read_head(*head, T);
            parts.clear();
            pieces.clear();
            ++saves;
        } else {
            throw std::invalid_argument("component journal: a line that is not a part, pieces or a head");
        }
    }
    return saves;
}

// COMP-9, COMP-13: the table's file, with its journal where it has one
inline Table load_table(const std::string& path)
{
    bool found = false;
    const std::string text = detail::read_text(path, found);
    if (!found) throw std::runtime_error("component table: cannot read " + path);
    Table T = table_of_text(text);
    const std::string journal = detail::read_text(journal_path(path), found);
    if (found) apply_journal(T, text, journal);
    return T;
}

namespace detail {

// COMP-13: the loop's writer of the table's file and its journal; a call's first save writes the file whole
struct JournalWriter {
    std::size_t base_bytes = 0;          // the file's length (0: not written yet)
    std::string base_sha256;
    std::size_t bytes = 0;               // the journal's length since (0: none)
    std::size_t pieces = 0;              // the pieces the file and the journal hold
    std::set<Key> changed;               // the parts the table took since

    void save(const std::string& path, const Table& T, std::size_t keys, const std::string& stopped_by)
    {
        const std::filesystem::path jp = store::file::path_of(journal_path(path));
        if (base_bytes > 0) {
            const std::string t = (bytes == 0 ? journal_header(base_bytes, base_sha256) : std::string()) +
                                  journal_save(T, keys, stopped_by, changed, pieces);
            if (bytes + t.size() <= base_bytes) {                   // appended: one write, the head last
                std::ofstream f(jp, std::ios::binary | (bytes == 0 ? std::ios::trunc : std::ios::app));
                f.write(t.data(), static_cast<std::streamsize>(t.size()));
                f.flush();
                if (!f) throw std::runtime_error("component table: cannot write " + journal_path(path));
                bytes += t.size();
                pieces = T.pieces.size();
                changed.clear();
                return;
            }
        }
        const std::string t = table_text(T, keys, stopped_by);    // whole, as save_table
        fam::detail::write_atomic(path, t);
        std::error_code ec;
        std::filesystem::remove(jp, ec);   // where it cannot be removed it goes with the file before: not read
        base_bytes = t.size();
        base_sha256 = sha256_hex(t);
        bytes = 0;
        pieces = T.pieces.size();
        changed.clear();
    }
};

}   // namespace detail

// ── the loop ────────────────────────────────────────────────────────────────

class Loop {
public:
    // the family, the parts and the options are copied (their hooks with them); the table and the
    // bank (none: nullptr) are the caller's
    Loop(const fam::Family& F, const Components& C, Table& T, fam::Bank* B, const Options& o)
        : F_(F), C_(C), T_(T), B_(B), o_(o)
    {
        check();
    }

    // a part given (source given): evaluated on its problem and offered (COMP-1)
    void give(const Key& k, const Vec& x)
    {
        if (!std::binary_search(keys_.begin(), keys_.end(), k))
            throw std::invalid_argument("components: " + key_text(k) + " is not a key of the parts");
        const Problem& Q = problem(k);
        if (x.size() != Q.lo.size())
            throw std::invalid_argument("components: the point given for part " + key_text(k) + " has " +
                                        std::to_string(x.size()) + " values, its problem " +
                                        std::to_string(Q.lo.size()));
        scalar::Evaluator ev(Q, o_.eq_tol, false, 0.0, std::int64_t{1} << 40, 1, {}, 0);
        std::size_t i = 0;
        try {
            i = ev.index(x);
        } catch (const scalar::BudgetSpent&) {                                  // the problem's deadline
            ++T_.given.offered;
            return;
        }
        T_.used += ev.used();
        if (ev.failed[i]) {
            ++T_.given.offered;
            return;
        }
        offer(k, x, ev.F[i][0], ev.cv(i), T_.given, "given");
    }

    Report run()
    {
        T_.options = detail::options_json(o_);
        journal_ = detail::JournalWriter();                                    // COMP-13: the first save whole
        const bool dim = o_.diminish_ratio > 0.0 || o_.diminish_abs > 0.0;   // COMP-12
        if (!dim) T_.measure = fam::Measure();
        std::string stopped;
        long long since_save = 0;
        if (keys_.empty()) stopped = "done";                                   // COMP-11
        while (stopped.empty()) {
            if (!settle()) {
                stopped = "budget";
                break;
            }
            if (dim && diminished()) {                                         // COMP-12
                stopped = "diminishing";
                break;
            }
            if (o_.max_pieces > 0 && static_cast<long long>(T_.pieces.size()) >= o_.max_pieces) {
                stopped = "pieces";
                break;
            }
            if (left() <= 0) {
                stopped = "budget";
                break;
            }
            if (!T_.has_next) {                                                // COMP-5
                T_.next = pick();
                T_.has_next = true;
            }
            const long long l = left();
            if (l <= 0) {
                stopped = "budget";
                break;
            }
            const Key k = T_.next;
            T_.has_next = false;
            T_.next.clear();
            piece(k, (std::min)(o_.piece_budget, l));
            if (files() && ++since_save >= o_.save_every) {
                settle();                                                      // the piece's assemblies first
                save("");
                since_save = 0;
            }
        }
        if (files()) save(stopped);
        Report r;
        r.stopped_by = stopped;
        r.settings = settings_of(T_, keys_.size(), stopped);
        if (B_ && !keys_.empty())
            r.settings.push_back({"components_bank_total", json::float_repr(bank_total())});
        return r;
    }

private:
    const fam::Family F_;
    const Components C_;
    Table& T_;
    fam::Bank* B_;
    const Options o_;
    std::vector<Key> keys_;                                                    // COMP-1: ascending
    std::vector<long long> inst_;                                              // FAM-1: ascending
    std::vector<double> w_;
    std::map<Key, Problem> problems_;
    detail::JournalWriter journal_;                                            // COMP-13

    void check()
    {
        keys_ = C_.keys;
        std::sort(keys_.begin(), keys_.end());
        if (std::adjacent_find(keys_.begin(), keys_.end()) != keys_.end())
            throw std::invalid_argument("components: a key given twice");
        if (o_.pick != "round_robin" && o_.pick != "sensitivity")
            throw std::invalid_argument("components: pick = round_robin | sensitivity, not " + o_.pick);
        if (o_.pick == "sensitivity") detail::check_sensitivity(o_.epsilon, o_.scope);
        if (o_.scope != "all" && o_.scope != "used")
            throw std::invalid_argument("components: scope = all | used, not " + o_.scope);
        if (o_.reassemble != "used" && o_.reassemble != "all")
            throw std::invalid_argument("components: reassemble = used | all, not " + o_.reassemble);
        if (o_.max_pieces < 0) throw std::invalid_argument("components: max_pieces >= 0");
        if (o_.closure_rounds < 0) throw std::invalid_argument("components: closure_rounds >= 0");
        if (o_.save_every < 1) throw std::invalid_argument("components: save_every >= 1");
        if (!o_.bank_file.empty() && !B_) throw std::invalid_argument("components: a bank_file without a bank");
        if (o_.table_journal && o_.table_file.empty())
            throw std::invalid_argument("components: a table_journal without a table_file");
        scalar::strategy::Diminish::check(o_.diminish_ratio, o_.diminish_abs, "components");
        if (o_.round_pieces < 0) throw std::invalid_argument("components: round_pieces >= 0");
        if (keys_.empty()) return;                                             // COMP-11: nothing asked
        if (!C_.component_problem) throw std::invalid_argument("components: no component_problem");
        if (!C_.assemble) throw std::invalid_argument("components: no assemble");
        inst_ = detail::instances_of(F_);
        if (inst_.empty()) throw std::invalid_argument("components: the family has no instances");
        if (B_ && !F_.problem_at) throw std::invalid_argument("components: a bank needs the family's problem_at");
        w_ = detail::weights_of(F_, inst_);
        for (const auto& kv : T_.parts) {                                      // COMP-9: a table of these parts
            if (!std::binary_search(keys_.begin(), keys_.end(), kv.first))
                throw std::invalid_argument("components: the table has part " + key_text(kv.first) + ", not a key");
            if (kv.second.x.size() != problem(kv.first).lo.size())
                throw std::invalid_argument("components: the table's point of part " + key_text(kv.first) +
                                            " is not of its problem's length");
        }
        for (const auto& kv : T_.touched)
            if (!std::binary_search(keys_.begin(), keys_.end(), kv.first))
                throw std::invalid_argument("components: the table touched " + key_text(kv.first) + ", not a key");
        if (T_.has_next && !std::binary_search(keys_.begin(), keys_.end(), T_.next))
            throw std::invalid_argument("components: the table's next part " + key_text(T_.next) + " is not a key");
        for (const auto& kv : T_.assembled)
            if (!in_family(kv.first))
                throw std::invalid_argument("components: the table assembled instance " + std::to_string(kv.first) +
                                            ", not in the family");
        for (long long n : T_.pending)
            if (!in_family(n))
                throw std::invalid_argument("components: instance " + std::to_string(n) + " pending, not in the family");
    }

    bool in_family(long long n) const { return std::binary_search(inst_.begin(), inst_.end(), n); }
    bool files() const { return !o_.table_file.empty() || !o_.bank_file.empty(); }
    long long left() const { return o_.budget - static_cast<long long>(std::ceil(T_.used)); }

    const Problem& problem(const Key& k)
    {
        auto it = problems_.find(k);
        if (it == problems_.end()) it = problems_.emplace(k, C_.component_problem(k)).first;
        return it->second;
    }

    double weight(long long n) const
    {
        return w_[static_cast<std::size_t>(std::lower_bound(inst_.begin(), inst_.end(), n) - inst_.begin())];
    }

    double bank_total() const { return fam::total_of(*B_, [this](long long n) { return weight(n); }); }

    long long age(const Key& k) const
    {
        const auto it = T_.touched.find(k);
        return static_cast<long long>(T_.pieces.size()) - (it == T_.touched.end() ? -1 : it->second);
    }

    // COMP-1; COMP-7: the instances the new part touches wait for their assembly
    bool offer(const Key& k, const Vec& x, double q, double cv, fam::Counts& c, const std::string& source)
    {
        ++c.offered;
        const auto it = T_.parts.find(k);
        const bool is_new = it == T_.parts.end();
        if (!is_new && !fam::detail::better(q, cv, it->second.q, it->second.cv)) return false;
        Part p;
        p.x = x;
        p.q = q;
        p.cv = cv;
        p.source = source;
        p.piece = static_cast<long long>(T_.pieces.size()) - 1;
        p.used = T_.used;
        T_.parts[k] = std::move(p);
        if (o_.table_journal) journal_.changed.insert(k);                      // COMP-13
        ++c.accepted;
        for (long long n : inst_) {
            bool touched = is_new || o_.reassemble == "all";
            if (!touched) {
                const auto a = T_.assembled.find(n);
                touched = a == T_.assembled.end() ||
                          std::binary_search(a->second.used_keys.begin(), a->second.used_keys.end(), k);
            }
            if (touched) T_.pending.push_back(n);
        }
        std::sort(T_.pending.begin(), T_.pending.end());
        T_.pending.erase(std::unique(T_.pending.begin(), T_.pending.end()), T_.pending.end());
        return true;
    }

    fam::Options bank_options(long long budget) const
    {
        fam::Options b;
        b.eq_tol = o_.eq_tol;
        b.closure_rounds = o_.closure_rounds;
        b.budget = budget;
        b.piece_budget = 1;
        return b;
    }

    // COMP-7: the instances pending, then the bank's closure; false where the budget ended it
    bool settle()
    {
        std::optional<fam::Campaign> to_bank;
        while (!T_.pending.empty()) {
            if (left() <= 0) return false;
            const long long n = T_.pending.front();
            T_.pending.erase(T_.pending.begin());
            const Assembled a = detail::assemble_checked(C_, n, View(T_.parts));
            ++T_.assembles;
            T_.used += a.cost;
            if (!a.ok) {
                T_.assembled.erase(n);
                continue;
            }
            T_.assembled[n] = Assembly{a.f, a.used_keys};
            if (B_ && !a.x.empty()) {
                if (!to_bank) to_bank.emplace(F_, *B_, bank_options(1));
                const double was = B_->used;
                to_bank->give(n, a.x, "assembled");
                T_.used += B_->used - was;
                ++T_.bank_given;
            }
        }
        if (B_ && !B_->now.empty()) {
            const long long l = left();
            if (l <= 0) return false;
            fam::Campaign c(F_, *B_, bank_options(static_cast<long long>(std::ceil(B_->used)) + l));
            const double was = B_->used;
            const bool done = c.closure();
            T_.used += B_->used - was;
            if (!done) return false;
        }
        return true;
    }

    // COMP-12: at the top of the loop; whether the round just ended ends the loop
    bool diminished()
    {
        fam::Measure& m = T_.measure;
        const long long p = static_cast<long long>(T_.pieces.size());
        std::vector<long long> over;
        bool feasible = true;
        double now = 0.0;
        if (B_) {                                                              // the bank's total
            for (const auto& [n, r] : B_->records) {
                over.push_back(n);
                feasible = feasible && !(r.cv > 0.0);
            }
            now = bank_total();
        } else {                                                               // the last assemblies'
            for (const auto& [n, a] : T_.assembled) {
                over.push_back(n);
                now += weight(n) * a.f;
            }
        }
        auto at_start = [&]() {
            m.start = p;
            m.total = now;
            m.over = over;
            m.feasible = feasible;
        };
        if (!m.on || m.rule.ended) {                                           // a new measure from here
            m = fam::Measure();
            m.on = true;
            m.rule.ratio = o_.diminish_ratio;
            m.rule.abs = o_.diminish_abs;
            at_start();
            return false;
        }
        m.rule.ratio = o_.diminish_ratio;                                      // the call's
        m.rule.abs = o_.diminish_abs;
        const long long R = o_.round_pieces > 0 ? o_.round_pieces : static_cast<long long>(keys_.size());
        if (p - m.start < R) return false;
        const double g = fam::detail::gain_of(m, now, over, feasible);
        at_start();
        return m.rule.round(g);
    }

    // COMP-5
    Key pick()
    {
        if (o_.pick == "round_robin") {
            const Key k = keys_[T_.cursor % keys_.size()];
            T_.cursor = (T_.cursor + 1) % keys_.size();
            return k;
        }
        const Key* missing = nullptr;
        for (const Key& k : keys_)
            if (!T_.parts.count(k) && (!missing || age(k) > age(*missing))) missing = &k;
        if (missing) return *missing;
        const Sensitivity S = detail::sensitivity_of(inst_, w_, C_, T_.parts, o_.epsilon, o_.scope);
        T_.used += S.cost;
        T_.assembles += S.assembles;
        ++T_.sensitivities;
        T_.ranking.clear();
        std::map<Key, double> value;
        for (const Contribution& c : S.ranked) {
            T_.ranking.push_back({c.key, c.value});
            value[c.key] = c.value;
        }
        const Key* best = nullptr;
        double best_v = 0.0;
        long long best_age = 0;
        for (const Key& k : keys_) {
            const double v = value.at(k);
            const long long a = age(k);
            if (!best || v > best_v || (v == best_v && a > best_age)) {
                best = &k;
                best_v = v;
                best_age = a;
            }
        }
        return *best;
    }

    // COMP-6
    void piece(const Key& k, long long budget)
    {
        const long long p = static_cast<long long>(T_.pieces.size());
        T_.pieces.push_back(PieceLog{k, T_.used, budget, false});
        T_.touched[k] = p;
        scalar::Start st;
        const auto it = T_.parts.find(k);
        if (it != T_.parts.end()) {
            st.has_x0 = true;
            st.x0 = it->second.x;
        }
        const scalar::Result r =
            scalar::run(o_.solver, problem(k), o_.eq_tol, false, 0.0, o_.pop_size, budget,
                        scalar::strategy::start_seed(o_.seed, static_cast<std::uint64_t>(p)), o_.knobs, {}, 0, st);
        T_.used += r.used;
        if (r.variables.empty() || r.objectives.empty()) return;
        T_.pieces.back().taken =
            offer(k, r.variables[0], r.objectives[0][0], r.cv.empty() ? 0.0 : r.cv[0], T_.run, "run");
    }

    // COMP-9: the bank first, then the table (COMP-11: without keys the bank's file is not written; COMP-13)
    void save(const std::string& stopped)
    {
        if (!o_.bank_file.empty() && !keys_.empty()) fam::save_bank(o_.bank_file, *B_, bank_total(), inst_.size());
        if (o_.table_file.empty()) return;
        if (o_.table_journal)
            journal_.save(o_.table_file, T_, keys_.size(), stopped);           // COMP-13
        else
            save_table(o_.table_file, T_, keys_.size(), stopped);
    }
};

// the loop over the parts of the family's instances from the table T and into the bank B (nullptr:
// none; COMP-5 ... COMP-10)
inline Report run(const fam::Family& F, const Components& C, Table& T, fam::Bank* B, const Options& o)
{
    if (o.budget <= 0) throw std::invalid_argument("components: budget > 0");
    if (o.piece_budget <= 0) throw std::invalid_argument("components: piece_budget > 0");
    Loop l(F, C, T, B, o);
    return l.run();
}

}   // namespace mootation::campaign::comp
