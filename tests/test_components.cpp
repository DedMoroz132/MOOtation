// SPDX-License-Identifier: Apache-2.0
// Assembly from parts and the bottleneck analysis (task 32, layer 3;
// campaign/components.hpp) on a synthetic family:
//   * instance n (n = 1 ... 8) is the sum of the parts of a partition of n
//     into threes and the rest (7 = 3 + 3 + 1), its quality the sum of the
//     parts' qualities; part {s} is the problem b(s) + sum (x_j - 0.5)^2 in
//     dimension s. The sensitivity finds exactly the parts of the
//     partitions, each instance's delta eps times the part's multiplicity,
//     the contribution sum_n w(n) eps mult; parts no assembly uses
//     (including one the table has and an assembly leaves out) get 0;
//     scope = used gives the same with fewer assemblies;
//   * the loop: parts searched, the instances assembled into the family's
//     bank, the bank's closure from them (instance 9 is not assembled, it is
//     reached by the transfer from 8); the count of used(), exactly;
//     reassemble = used | all; pick = sensitivity picks the largest
//     contribution;
//   * the table's file read back is the table; a loop stopped by max_pieces
//     or by its budget (between assemblies, in the bank's closure), loaded
//     and run on, is the loop in one piece (the table and the bank, byte for
//     byte);
//   * one part, one piece: the table's record is the scalar::run, bit for
//     bit; without keys the layer does nothing and the family's campaign is
//     that of before; the refusals.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/campaign/components.hpp>
#include <mootation/campaign/family.hpp>
#include <mootation/scalar/scalar.hpp>

namespace {

namespace sc = mootation::scalar;
namespace fam = mootation::campaign::fam;
namespace comp = mootation::campaign::comp;
using Vec = std::vector<double>;
using Key = comp::Key;

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
}

std::string read_text(const std::filesystem::path& p)
{
    std::ifstream in(p, std::ios::binary);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

// a path as UTF-8 text (u8string is a std::u8string in C++20)
std::string utf8(const std::filesystem::path& p)
{
    const auto s = p.u8string();
    return std::string(s.begin(), s.end());
}

std::filesystem::path temp_dir(const std::string& what)
{
    const std::filesystem::path d = std::filesystem::temp_directory_path() /
                                    ("mootation_test_components_" + what + "_" + std::to_string(std::rand()) + "_" +
                                     std::to_string(std::rand()));
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

// ── the synthetic family ────────────────────────────────────────────────────

// the parts: {1} ... {5} and {7, 7}; only {1}, {2}, {3} are in a partition
const std::vector<Key> kKeys = {{1}, {2}, {3}, {4}, {5}, {7, 7}};

std::size_t dim_of(const Key& k) { return k.size() == 1 ? static_cast<std::size_t>(k[0]) : 2; }
double b_of(const Key& k)
{
    double s = 0.0;
    for (long long v : k) s += 0.25 * static_cast<double>(v);
    return s;
}

// n into threes and the rest: 7 = {3} {3} {1}; 9 = {3} {3} {3}
std::vector<Key> partition(long long n)
{
    std::vector<Key> p(static_cast<std::size_t>(n / 3), Key{3});
    if (n % 3) p.push_back(Key{n % 3});
    return p;
}

// a part's value: b(k) + sum (x_j - 0.5)^2, added in this order
double part_value(const Key& k, const double* x)
{
    double s = b_of(k);
    for (std::size_t j = 0; j < dim_of(k); ++j) s += (x[j] - 0.5) * (x[j] - 0.5);
    return s;
}

sc::Problem component_problem(const Key& k)
{
    sc::Problem P;
    P.lo.assign(dim_of(k), -2.0);
    P.hi.assign(dim_of(k), 2.0);
    P.m = 1;
    P.evaluate = [k](const Vec& x, Vec& f, Vec&) { f.assign(1, part_value(k, x.data())); };
    return P;
}

// instance n: the sum of its partition's parts, the parts' points one after another
sc::Problem instance_problem(long long n)
{
    sc::Problem P;
    P.lo.assign(static_cast<std::size_t>(n), -2.0);
    P.hi.assign(static_cast<std::size_t>(n), 2.0);
    P.m = 1;
    P.evaluate = [n](const Vec& x, Vec& f, Vec&) {
        double t = 0.0;
        std::size_t at = 0;
        for (const Key& k : partition(n)) {
            t += part_value(k, x.data() + at);
            at += dim_of(k);
        }
        f.assign(1, t);
    };
    return P;
}

const double kAssembleCost = 0.5;

// instances up to 8 are assembled; 9 is not (only the transfer from 8 reaches it)
comp::Assembled assemble_partition(long long n, const comp::View& v)
{
    comp::Assembled a;
    a.cost = kAssembleCost;
    if (n > 8) return a;
    double t = 0.0;
    for (const Key& k : partition(n)) {
        if (!v.has(k)) return a;
        t += v.quality(k);
        a.x.insert(a.x.end(), v.point(k).begin(), v.point(k).end());
        a.used_keys.push_back(k);
    }
    a.ok = true;
    a.f = t;
    return a;
}

comp::Components parts_of(const std::function<comp::Assembled(long long, const comp::View&)>& assemble)
{
    comp::Components C;
    C.keys = {{7, 7}, {5}, {4}, {3}, {2}, {1}};                         // kept in ascending order (COMP-1)
    C.component_problem = component_problem;
    C.assemble = assemble;
    return C;
}

fam::Family family(bool with_nine)
{
    fam::Family F;
    F.instances = {1, 2, 3, 4, 5, 6, 7, 8};
    if (with_nine) F.instances.push_back(9);
    F.problem_at = instance_problem;
    F.weight = [](long long n) { return static_cast<double>(n); };
    F.transfer = [](long long n, const Vec& x) {
        std::vector<fam::Candidate> out;
        if (n == 8) {
            fam::Candidate c{9, x};
            c.x.push_back(0.5);
            out.push_back(c);
        }
        return out;
    };
    return F;
}

// every part at its optimum: q = b(k), exact
std::map<Key, comp::Part> exact_parts()
{
    std::map<Key, comp::Part> parts;
    for (const Key& k : kKeys) {
        comp::Part p;
        p.x.assign(dim_of(k), 0.5);
        p.q = b_of(k);
        p.cv = 0.0;
        p.source = "given";
        parts[k] = p;
    }
    return parts;
}

long long mult(const Key& k, long long n)
{
    long long m = 0;
    for (const Key& p : partition(n)) m += p == k ? 1 : 0;
    return m;
}

const comp::Contribution& contribution(const comp::Sensitivity& S, const Key& k)
{
    for (const auto& c : S.ranked)
        if (c.key == k) return c;
    throw std::runtime_error("no contribution of " + comp::key_text(k));
}

comp::Options loop_options(const std::string& pick, long long budget)
{
    comp::Options o;
    o.solver = "nelder_mead";
    o.pop_size = 10;
    o.budget = budget;
    o.piece_budget = 60;
    o.pick = pick;
    o.epsilon = 1.0 / 1024.0;
    o.seed = 11;
    return o;
}

std::string bank_text(const fam::Bank& B, const fam::Family& F)
{
    return fam::bank_text(B, fam::total_of(B, F.weight), F.instances.size(), "");
}

// ── the sensitivity ─────────────────────────────────────────────────────────

void sensitivity_finds_the_partition()
{
    const fam::Family F = family(false);
    const comp::Components C = parts_of(assemble_partition);
    const auto parts = exact_parts();
    const double eps = 1.0 / 1024.0;
    const comp::Sensitivity S = comp::sensitivity(F, C, parts, eps, "all");
    bool deltas = true, values = true;
    for (const Key& k : kKeys) {
        const comp::Contribution& c = contribution(S, k);
        double v = 0.0;
        std::vector<std::pair<long long, double>> want;
        for (long long n = 1; n <= 8; ++n)
            if (mult(k, n)) {
                want.push_back({n, eps * static_cast<double>(mult(k, n))});
                v += static_cast<double>(n) * eps * static_cast<double>(mult(k, n));
            }
        deltas = deltas && c.deltas == want;
        values = values && c.value == v;
    }
    check(deltas, "sensitivity: delta f(n) = eps * the part's multiplicity in n's partition, exactly, and only there");
    check(values, "sensitivity: the contribution = sum_n w(n) eps mult, exactly");
    std::vector<Key> order;
    for (const auto& c : S.ranked) order.push_back(c.key);
    check(order == std::vector<Key>{{3}, {2}, {1}, {4}, {5}, {7, 7}},
          "sensitivity: ranked {3} (54 eps), {2} (15 eps), {1} (12 eps), then the zeros by key");
    check(contribution(S, {4}).value == 0.0 && contribution(S, {5}).deltas.empty() &&
              contribution(S, {7, 7}).value == 0.0,
          "sensitivity: the parts no assembly uses contribute 0");
    check(S.assembles == 8 + 8 * 6 && S.cost == kAssembleCost * 56.0 && S.failed.empty() && S.f.at(7) == 1.75,
          "sensitivity, scope all: 8 + 8 * 6 assemblies, their declared cost, f(7) = 0.75 + 0.75 + 0.25");

    const comp::Sensitivity U = comp::sensitivity(F, C, parts, eps, "used");
    bool same = U.ranked.size() == S.ranked.size();
    for (std::size_t i = 0; same && i < S.ranked.size(); ++i)
        same = U.ranked[i].key == S.ranked[i].key && U.ranked[i].value == S.ranked[i].value &&
               U.ranked[i].deltas == S.ranked[i].deltas;
    check(same && U.assembles == 8 + 12 && U.cost == kAssembleCost * 20.0,
          "sensitivity, scope used: the same ranking with 8 + 12 assemblies");

    // an assembly that leaves part {3} out (n in ones and twos): {3} is in the table, its contribution 0
    const comp::Components C2 = parts_of([](long long n, const comp::View& v) {
        comp::Assembled a;
        double t = 0.0;
        for (long long i = 0; i < n / 2; ++i) {
            t += v.quality({2});
            a.used_keys.push_back({2});
        }
        if (n % 2) {
            t += v.quality({1});
            a.used_keys.push_back({1});
        }
        a.ok = true;
        a.f = t;
        return a;
    });
    const comp::Sensitivity T = comp::sensitivity(F, C2, parts, eps, "all");
    check(contribution(T, {3}).value == 0.0 && contribution(T, {3}).deltas.empty() &&
              contribution(T, {2}).value == eps * (2 + 3 + 4 * 2 + 5 * 2 + 6 * 3 + 7 * 3 + 8 * 4) &&
              contribution(T, {1}).value == eps * (1 + 3 + 5 + 7),
          "sensitivity: an assembly that does not use part {3} gives it 0; {2} and {1} by their multiplicities");
    // a weight 0 and an assembly that fails
    fam::Family F0 = F;
    F0.weight = [](long long n) { return n == 3 ? 0.0 : 1.0; };
    const comp::Components C3 = parts_of([](long long n, const comp::View& v) {
        if (n == 5) return comp::Assembled{};
        return assemble_partition(n, v);
    });
    const comp::Sensitivity W = comp::sensitivity(F0, C3, parts, eps, "all");
    check(W.failed == std::vector<long long>{5} && contribution(W, {3}).value == eps * (1 + 2 + 2 + 2) &&
              contribution(W, {3}).deltas.size() == 5,
          "sensitivity: w(3) = 0 adds nothing to the value (its delta is listed); instance 5 failed, left out");
}

// ── the loop ────────────────────────────────────────────────────────────────

void the_count_and_the_closure()
{
    const fam::Family F = family(true);
    const comp::Components C = parts_of(assemble_partition);
    comp::Table T;
    fam::Bank B;
    comp::Options o = loop_options("round_robin", 20);
    {
        comp::Loop l(F, C, T, &B, o);
        for (const Key& k : kKeys) l.give(k, Vec(dim_of(k), 0.5));
        check(T.pending == std::vector<long long>{1, 2, 3, 4, 5, 6, 7, 8, 9} && T.used == 6.0,
              "loop: six parts given (6 evaluations), every instance pending");
        const comp::Report r = l.run();
        // 6 + 9 assemblies * 0.5 + 8 records given to the bank + 1 candidate of the closure (8 -> 9)
        check(T.used == 19.5 && B.used == 9.0 && T.assembles == 9 && T.bank_given == 8 && r.stopped_by == "budget" &&
                  T.pieces.empty(),
              "loop: used = 6 + 9 * 0.5 + 9 (the bank's), exactly; no piece (left 0)");
    }
    bool same = true;
    for (long long n = 1; n <= 8; ++n)
        same = same && B.records.at(n).f == T.assembled.at(n).f && B.records.at(n).source == "assembled";
    check(same, "loop: the bank's records of 1 ... 8 are the assemblies (source assembled, f the same)");
    check(B.records.at(9).source == "transfer" && B.records.at(9).from == 8 && B.records.at(9).f == 2.25 &&
              !T.assembled.count(9),
          "loop: instance 9 (not assembled) by the bank's closure from 8, f = 3 * 0.75");
    check(B.given.offered == 8 && B.given.accepted == 8 && B.transfer.accepted == 1 &&
              B.closure_rounds == std::vector<long long>{2},
          "loop: the bank's counts (path given) and its closure of 2 rounds");

    // reassemble: a better part {1}: used -> the instances using it and those without an assembly
    {
        comp::Loop l(F, C, T, &B, o);
        Vec x(1, 0.5);
        T.parts.at({1}).q = 1.0;                                         // a worse record to replace
        l.give({1}, x);
        check(T.pending == std::vector<long long>{1, 4, 7, 9}, "reassemble = used: 1, 4, 7 (use {1}) and 9 (none)");
        T.pending.clear();
        comp::Options oa = o;
        oa.reassemble = "all";
        comp::Loop la(F, C, T, &B, oa);
        T.parts.at({1}).q = 1.0;
        la.give({1}, x);
        check(T.pending.size() == 9, "reassemble = all: every instance");
    }
}

void loop_into_the_bank(const std::string& pick)
{
    const fam::Family F = family(true);
    const comp::Components C = parts_of(assemble_partition);
    comp::Table T;
    fam::Bank B;
    const comp::Report r = comp::run(F, C, T, &B, loop_options(pick, 2400));
    check(T.parts.size() == 6 && B.records.size() == 9 && r.stopped_by == "budget",
          "loop " + pick + ": every part searched, every instance in the bank");
    bool same = true;
    for (long long n = 1; n <= 8; ++n) same = same && B.records.at(n).f == T.assembled.at(n).f;
    check(same, "loop " + pick + ": the bank's f of 1 ... 8 = their last assemblies");
    if (pick == "sensitivity") {
        // the first six pieces fill the table (missing first, lower key first), then the ranking decides
        bool first_six = true;
        const std::vector<Key> ascending = {{1}, {2}, {3}, {4}, {5}, {7, 7}};
        for (std::size_t i = 0; i < 6; ++i) first_six = first_six && T.pieces[i].key == ascending[i];
        check(first_six && T.pieces[6].key == Key{3} && T.sensitivities > 0 && T.ranking.front().first == Key{3},
              "loop sensitivity: the missing parts first, then {3}, the largest contribution");
        long long useless = 0;
        for (std::size_t i = 6; i < T.pieces.size(); ++i)
            useless += (T.pieces[i].key == Key{4} || T.pieces[i].key == Key{5} || T.pieces[i].key == Key{7, 7}) ? 1 : 0;
        check(useless == 0, "loop sensitivity: no piece after the first six on a part no instance uses");
    }
}

void file_and_going_on()
{
    const fam::Family F = family(true);
    const comp::Components C = parts_of(assemble_partition);
    const std::filesystem::path d = temp_dir("file");
    const std::string tf = utf8(d / "parts.json"), bf = utf8(d / "bank.json");
    for (const std::string pick : {"round_robin", "sensitivity"}) {
        comp::Options o = loop_options(pick, 1500);
        comp::Table T1;
        fam::Bank B1;
        comp::run(F, C, T1, &B1, o);
        const std::string t1 = comp::table_text(T1, 6, ""), b1 = bank_text(B1, F);

        for (long long stop : {1LL, 4LL, 9LL}) {
            comp::Options o1 = o;
            o1.max_pieces = stop;
            o1.table_file = tf;
            o1.bank_file = bf;
            {
                comp::Table T;
                fam::Bank B;
                comp::run(F, C, T, &B, o1);
            }
            comp::Table T = comp::load_table(tf);
            fam::Bank B = fam::load_bank(bf);
            comp::Options o2 = o;
            o2.table_file = tf;
            o2.bank_file = bf;
            comp::run(F, C, T, &B, o2);
            check(comp::table_text(T, 6, "") == t1 && bank_text(B, F) == b1,
                  "file " + pick + ": stopped after " + std::to_string(stop) +
                      " pieces, loaded, run on = the loop in one piece (table and bank, byte for byte)");
        }
    }
    // stopped by the budget: between assemblies (15), in the bank's closure (19), at the loop's top (20)
    {
        comp::Options o = loop_options("round_robin", 900);
        auto given = [&](comp::Table& T, fam::Bank& B) {
            comp::Loop l(F, C, T, &B, o);
            for (const Key& k : kKeys) l.give(k, Vec(dim_of(k), 0.75));
        };
        comp::Table T1;
        fam::Bank B1;
        given(T1, B1);
        comp::run(F, C, T1, &B1, o);
        const std::string t1 = comp::table_text(T1, 6, ""), b1 = bank_text(B1, F);
        for (long long budget : {15LL, 19LL, 20LL}) {
            comp::Table T;
            fam::Bank B;
            given(T, B);
            comp::Options o1 = o;
            o1.budget = budget;
            o1.table_file = tf;
            o1.bank_file = bf;
            const comp::Report r = comp::run(F, C, T, &B, o1);
            const bool where = budget == 15 ? !T.pending.empty() : budget == 19 ? !B.now.empty() : T.pending.empty();
            comp::Table T2 = comp::load_table(tf);
            fam::Bank B2 = fam::load_bank(bf);
            const bool exact = comp::table_text(T2, 6, "") == comp::table_text(T, 6, "") && bank_text(B2, F) == bank_text(B, F);
            comp::run(F, C, T2, &B2, o);
            check(r.stopped_by == "budget" && where && exact && comp::table_text(T2, 6, "") == t1 && bank_text(B2, F) == b1,
                  "file: stopped by the budget " + std::to_string(budget) + " (" +
                      (budget == 15 ? "between assemblies" : budget == 19 ? "in the closure" : "at the top") +
                      "), read back exactly, run on = the loop in one piece");
        }
    }
    {
        const std::string text = read_text(tf);
        check(text.rfind("{\n\"format\": \"mootation-component-table\",\n\"version\": 1,\n", 0) == 0,
              "file: format and version first");
        std::string other = text;
        other.replace(other.find("\"version\": 1"), 12, "\"version\": 2");
        bool refused = false;
        try {
            comp::table_of_text(other);
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        check(refused, "file: a table of another version is refused");
    }
    std::filesystem::remove_all(d);
}

// ── bit for bit ─────────────────────────────────────────────────────────────

void one_part_is_the_run()
{
    fam::Family F;
    F.instances = {3};
    F.problem_at = instance_problem;
    comp::Components C;
    C.keys = {{3}};
    C.component_problem = component_problem;
    C.assemble = assemble_partition;
    comp::Table T;
    comp::Options o = loop_options("round_robin", 100);
    o.seed = 7;
    o.max_pieces = 1;
    comp::run(F, C, T, nullptr, o);
    const sc::Result r = sc::run("nelder_mead", component_problem({3}), 1e-4, false, 0.0, 10, 60, 7, {}, {}, 0, sc::Start{});
    const comp::Part& p = T.parts.at({3});
    check(p.x == r.variables[0] && p.q == r.objectives[0][0] && T.used == static_cast<double>(r.used) + kAssembleCost,
          "one part, one piece: the table's record is scalar::run's answer, bit for bit");
}

void without_keys_nothing()
{
    const fam::Family F = family(true);
    fam::Options fo;
    fo.pop_size = 10;
    fo.budget = 400;
    fo.piece_budget = 60;
    fo.seed = 5;
    fam::Bank B0;
    fam::run(F, B0, fo);
    fam::Bank B = B0;
    const std::string before = bank_text(B, F);
    long long calls = 0;
    comp::Components C;
    C.component_problem = [&](const Key& k) {
        ++calls;
        return component_problem(k);
    };
    C.assemble = [&](long long n, const comp::View& v) {
        ++calls;
        return assemble_partition(n, v);
    };
    const std::filesystem::path d = temp_dir("off");
    comp::Table T;
    comp::Options o = loop_options("sensitivity", 1000);
    o.bank_file = utf8(d / "bank.json");
    const comp::Report r = comp::run(F, C, T, &B, o);
    check(r.stopped_by == "done" && calls == 0 && bank_text(B, F) == before && T.used == 0.0 && T.parts.empty() &&
              !std::filesystem::exists(d / "bank.json"),
          "without keys: done, no hook called, the bank as it was, its file not written");
    fam::Options fo2 = fo;
    fo2.budget = 800;
    fam::Bank B1 = B0;
    fam::run(F, B1, fo2);
    fam::run(F, B, fo2);
    check(bank_text(B, F) == bank_text(B1, F), "without keys: the family's campaign goes on as it would have");
    std::filesystem::remove_all(d);
}

// ── refusals ────────────────────────────────────────────────────────────────

template <class F>
bool refused(F f)
{
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void refusals()
{
    const fam::Family F = family(false);
    const auto parts = exact_parts();
    check(refused([&] {
              comp::sensitivity(F, parts_of([](long long, const comp::View&) {
                                    comp::Assembled a;
                                    a.ok = true;
                                    a.f = 0.0;
                                    a.used_keys = {{6}};
                                    return a;
                                }),
                                parts, 0.1);
          }),
          "refused: an assembly that used a part not in the table");
    check(refused([&] {
              comp::sensitivity(F, parts_of([](long long, const comp::View&) {
                                    comp::Assembled a;
                                    a.cost = -1.0;
                                    return a;
                                }),
                                parts, 0.1);
          }),
          "refused: an assembly's cost < 0");
    const comp::Components C = parts_of(assemble_partition);
    check(refused([&] { comp::sensitivity(F, C, parts, 0.0); }), "refused: epsilon 0");
    check(refused([&] { comp::sensitivity(F, C, parts, 0.1, "some"); }), "refused: scope some");
    comp::Table T;
    check(refused([&] { (void)comp::run(F, C, T, nullptr, loop_options("sensitivity", 100)); }) == false,
          "a loop without a bank runs");
    comp::Options o = loop_options("sensitivity", 100);
    o.epsilon = 0.0;
    comp::Table T0;
    check(refused([&] { comp::run(F, C, T0, nullptr, o); }), "refused: pick = sensitivity with epsilon 0");
    o = loop_options("gap", 100);
    check(refused([&] { comp::run(F, C, T0, nullptr, o); }), "refused: pick = gap");
    o = loop_options("round_robin", 100);
    o.bank_file = "x.json";
    check(refused([&] { comp::run(F, C, T0, nullptr, o); }), "refused: a bank_file without a bank");
    comp::Components C2 = C;
    C2.keys.push_back({3});
    check(refused([&] { comp::run(F, C2, T0, nullptr, loop_options("round_robin", 100)); }), "refused: a key twice");
    comp::Table T1;
    T1.parts[{6}] = comp::Part{};
    check(refused([&] { comp::run(F, C, T1, nullptr, loop_options("round_robin", 100)); }),
          "refused: a table with a part that is not a key");
    comp::Table T2;
    T2.parts[{2}] = comp::Part{};
    check(refused([&] { comp::run(F, C, T2, nullptr, loop_options("round_robin", 100)); }),
          "refused: a table's point not of its problem's length");
    check(refused([&] { comp::table_of_text("{\"format\": \"mootation-family-bank\", \"version\": 1}"); }),
          "refused: a file of another format");
}

}   // namespace

int main()
{
    try {
        sensitivity_finds_the_partition();
        the_count_and_the_closure();
        loop_into_the_bank("round_robin");
        loop_into_the_bank("sensitivity");
        file_and_going_on();
        one_part_is_the_run();
        without_keys_nothing();
        refusals();
    } catch (const std::exception& e) {
        std::printf("FAIL  exception: %s\n", e.what());
        ++g_failed;
    }
    if (g_failed) {
        std::printf("%d failure(s)\n", g_failed);
        return 1;
    }
    std::printf("all passed\n");
    return 0;
}
