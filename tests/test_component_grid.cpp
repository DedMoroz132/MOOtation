// SPDX-License-Identifier: Apache-2.0
// New parts (task 33.2; campaign/components.hpp COMP-15) on synthetic tables:
// part `key` the problem base(key) + (x_1 - 0.5)^2 + (x_2 - 0.5)^2 in the box
// [-2, 2]^2, make_part's construction the point (0.8, 0.3) (0.13 above the
// part's best), instance n assembled at no cost from the best part it may
// use (f = its q + n / 100; the line: any part; the plane: the parts (a, n)).
//   * the line, keys {1} {2} {3}, base 0.1 (k - 8)^2: the best part {8} lies
//     past the grid's upper face, and the grid grows to it ({1} ... {9}: {8}
//     inside), with pick = sensitivity and round_robin; a hard bound stops
//     it; the lower face (base 0.1 (k - 2)^2 from keys {5} {6} {7});
//     grow_step 2 and edge_band 1;
//   * the plane, keys (a, b) in {1, 2, 3}^2, base 0.1 (a - 6)^2 + 0.05 b,
//     instances 1, 2, 3: the three best parts (3, 1), (3, 2), (3, 3) on
//     coordinate 0's upper face move it to a = 7, one layer of three keys a
//     step; coordinate 1's faces hold one of three (1/3 < edge_share 0.4)
//     and stay; with edge_share 0.3 they move once, the layers made on the
//     box as it is at their turn (sweep.hpp's order);
//   * a loop stopped (by its budget at every whole number where no piece
//     was cut short: before an assembly, after a pick, after a make; by
//     max_pieces) and run on is the loop in one piece, bit for bit (the
//     file; the journal); a key made is not made again;
//   * make_part's cost and its point's evaluation are in used(); make_part
//     alone (no grid) makes the keys without a part, an empty x makes none;
//   * off by default: without make_part and the hard bounds the file has no
//     new field, and make_part with every part in the table is never asked;
//     the refusals.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
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
namespace json = mootation::json;
using Vec = std::vector<double>;
using Key = comp::Key;

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_failed;
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
                                    ("mootation_test_component_grid_" + what + "_" + std::to_string(std::rand()) + "_" +
                                     std::to_string(std::rand()));
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

std::string read(const std::string& path)
{
    std::ifstream f(mootation::store::file::path_of(path), std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string setting(const std::vector<std::pair<std::string, std::string>>& s, const std::string& key)
{
    for (const auto& kv : s)
        if (kv.first == key) return kv.second;
    return "";
}

bool has_setting(const std::vector<std::pair<std::string, std::string>>& s, const std::string& key)
{
    for (const auto& kv : s)
        if (kv.first == key) return true;
    return false;
}

std::string keys_text(const std::vector<Key>& ks)
{
    std::string s;
    for (const Key& k : ks) s += (s.empty() ? "" : " ") + comp::key_text(k);
    return s;
}

// the expansions as "sensitivity:coordinate:lo|hi:from:to:keys" a face
std::string expansions_text(const comp::Table& T)
{
    std::string s;
    for (const comp::Expansion& e : T.expansions)
        s += (s.empty() ? "" : " ") + std::to_string(e.sensitivity) + ":" + std::to_string(e.coordinate) + ":" +
             (e.upper ? "hi" : "lo") + ":" + std::to_string(e.from) + ":" + std::to_string(e.to) + ":" +
             std::to_string(e.keys);
    return s;
}

// the keys from (a, b) ... as a list, coordinate 0 the slowest
std::vector<Key> line_keys(long long first, long long last)
{
    std::vector<Key> ks;
    for (long long k = first; k <= last; ++k) ks.push_back({k});
    return ks;
}

std::vector<Key> plane_keys(long long a0, long long a1, long long b0, long long b1)
{
    std::vector<Key> ks;
    for (long long a = a0; a <= a1; ++a)
        for (long long b = b0; b <= b1; ++b) ks.push_back({a, b});
    return ks;
}

// the part's best key: the lowest q of the table
Key best_part(const comp::Table& T)
{
    Key best;
    double q = std::numeric_limits<double>::infinity();
    for (const auto& kv : T.parts)
        if (kv.second.q < q) {
            q = kv.second.q;
            best = kv.first;
        }
    return best;
}

sc::Problem part_problem(double base)
{
    sc::Problem P;
    P.lo.assign(2, -2.0);
    P.hi.assign(2, 2.0);
    P.m = 1;
    P.evaluate = [base](const Vec& x, Vec& f, Vec&) {
        f.assign(1, base + (x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5));
    };
    return P;
}

struct Maker {
    long long calls = 0;
    double cost = 0.0;
    Key empty_for;                // make_part gives no point for this key
    std::size_t length = 2;       // the length of the point it gives
};

// the parts: base(key); instance n from the best part (by_b: of the parts (a, n))
comp::Components parts(const std::vector<Key>& keys, std::function<double(const Key&)> base, bool by_b,
                       Maker* maker)
{
    comp::Components C;
    C.keys = keys;
    C.component_problem = [base](const Key& k) { return part_problem(base(k)); };
    C.assemble = [by_b](long long n, const comp::View& v) {
        comp::Assembled a;
        double q = std::numeric_limits<double>::infinity();
        Key best;
        for (const Key& k : v.keys()) {
            if (by_b && k[1] != n) continue;
            if (v.cv(k) <= 0.0 && v.quality(k) < q) {
                q = v.quality(k);
                best = k;
            }
        }
        if (best.empty()) return a;
        a.ok = true;
        a.f = q + 0.01 * static_cast<double>(n);
        a.used_keys = {best};
        return a;
    };
    if (maker)
        C.make_part = [maker](const Key& k) {
            ++maker->calls;
            comp::Made m;
            m.cost = maker->cost;
            if (k != maker->empty_for) m.x = maker->length == 2 ? Vec{0.8, 0.3} : Vec(maker->length, 0.8);
            return m;
        };
    return C;
}

double line_base(const Key& k, double centre)
{
    const double d = static_cast<double>(k[0]) - centre;
    return 0.1 * d * d;
}

double plane_base(const Key& k)
{
    const double d = static_cast<double>(k[0]) - 6.0;
    return 0.1 * d * d + 0.05 * static_cast<double>(k[1]);
}

fam::Family family(long long instances)
{
    fam::Family F;
    for (long long n = 1; n <= instances; ++n) F.instances.push_back(n);
    return F;
}

comp::Options options(const std::string& pick, const Key& hard_lo, const Key& hard_hi)
{
    comp::Options o;
    o.solver = "nelder_mead";
    o.pop_size = 10;
    o.piece_budget = 40;
    o.budget = 100000;
    o.pick = pick;
    o.epsilon = 1e-6;
    o.seed = 5;
    o.key_hard_lo = hard_lo;
    o.key_hard_hi = hard_hi;
    return o;
}

// ── the line ───────────────────────────────────────────────────────────────

void line_grows(const std::string& pick, long long max_pieces)
{
    Maker mk;
    mk.cost = 2.5;
    const comp::Components C = parts(line_keys(1, 3), [](const Key& k) { return line_base(k, 8.0); }, false, &mk);
    comp::Options o = options(pick, {1}, {12});
    o.max_pieces = max_pieces;
    comp::Table T;
    const comp::Report r = comp::run(family(4), C, T, nullptr, o);
    std::printf("      line, %s: keys %s; grown %s; expansions %s; best %s q %.3g; %zu pieces\n", pick.c_str(),
                setting(r.settings, "components_keys").c_str(), keys_text(T.grown).c_str(),
                expansions_text(T).c_str(), comp::key_text(best_part(T)).c_str(), T.parts.at(best_part(T)).q,
                T.pieces.size());
    check(r.stopped_by == "pieces" && setting(r.settings, "components_keys") == "9" && T.parts.size() == 9 &&
              T.grown == line_keys(4, 9),
          "line, " + pick + ": the best part {8} past the upper face; the grid grew {4} ... {9} and stopped");
    check(best_part(T) == Key{8} && T.parts.at({8}).q < 0.05, "line, " + pick + ": part {8} found, the best");
    std::string want;                                  // pick = sensitivity: sensitivity k - 3 grows k
    for (long long k = 4; k <= 9; ++k)
        want += (want.empty() ? "" : " ") + std::to_string(k - 3) + ":0:hi:" + std::to_string(k - 1) + ":" +
                std::to_string(k) + ":1";
    if (pick == "sensitivity")
        check(expansions_text(T) == want, "line, sensitivity: one key a step, on sensitivities 1 ... 6: " + want);
    else
        check(T.expansions.size() == 6 && T.sensitivities >= 7,
              "line, round_robin: six steps of one key, a sensitivity at each round's start (" +
                  std::to_string(T.sensitivities) + ")");
    check(mk.calls == 9 && T.made.offered == 9 && T.made_keys.size() == 9,
          "line, " + pick + ": make_part asked once a key, every key made (" + std::to_string(mk.calls) + ")");
    const json::Value offered = json::parse(setting(r.settings, "components_offered"));
    check(offered.find("made") && offered.find("made")->i == 9 && has_setting(r.settings, "components_grown") &&
              has_setting(r.settings, "components_expansions"),
          "line, " + pick + ": the settings: offered " + setting(r.settings, "components_offered") + ", grown, expansions");
}

void line_bounds()
{
    Maker mk;
    {
        const comp::Components C = parts(line_keys(1, 3), [](const Key& k) { return line_base(k, 8.0); }, false, &mk);
        comp::Options o = options("sensitivity", {1}, {6});
        o.max_pieces = 20;
        comp::Table T;
        const comp::Report r = comp::run(family(4), C, T, nullptr, o);
        check(T.grown == line_keys(4, 6) && best_part(T) == Key{6} && setting(r.settings, "components_keys") == "6",
              "line, key_hard_hi {6}: the grid stops at its hard bound, the best part {6} on its face (grown " +
                  keys_text(T.grown) + ")");
    }
    {
        const comp::Components C = parts(line_keys(5, 7), [](const Key& k) { return line_base(k, 2.0); }, false, &mk);
        comp::Options o = options("sensitivity", {1}, {12});
        o.max_pieces = 20;
        comp::Table T;
        comp::run(family(4), C, T, nullptr, o);
        check(T.grown == line_keys(1, 4) && best_part(T) == Key{2} &&
                  expansions_text(T) == "1:0:lo:5:4:1 2:0:lo:4:3:1 3:0:lo:3:2:1 4:0:lo:2:1:1",
              "line, the best part {2} past the lower face: the grid grew down to {1} (" + expansions_text(T) + ")");
    }
    {
        const comp::Components C = parts(line_keys(1, 3), [](const Key& k) { return line_base(k, 8.0); }, false, &mk);
        comp::Options o = options("sensitivity", {1}, {12});
        o.max_pieces = 20;
        o.grow_step = 2;
        comp::Table T;
        comp::run(family(4), C, T, nullptr, o);
        check(T.grown == line_keys(4, 9) && best_part(T) == Key{8} &&
                  expansions_text(T) == "1:0:hi:3:5:2 2:0:hi:5:7:2 3:0:hi:7:9:2",
              "line, grow_step 2: two keys a step to {9} (" + expansions_text(T) + ")");
    }
    {
        const comp::Components C = parts(line_keys(1, 3), [](const Key& k) { return line_base(k, 8.0); }, false, &mk);
        comp::Options o = options("sensitivity", {1}, {12});
        o.max_pieces = 20;
        o.edge_band = 1;
        comp::Table T;
        comp::run(family(4), C, T, nullptr, o);
        check(T.grown == line_keys(4, 10) && best_part(T) == Key{8},
              "line, edge_band 1: {8} one value from the face {9} still fires, the grid to {10} (grown " +
                  keys_text(T.grown) + ")");
    }
}

// ── the plane ──────────────────────────────────────────────────────────────

void plane(double share)
{
    Maker mk;
    const comp::Components C = parts(plane_keys(1, 3, 1, 3), plane_base, true, &mk);
    comp::Options o = options("sensitivity", {1, 0}, {10, 5});
    o.edge_share = share;
    o.max_pieces = 80;
    comp::Table T;
    const comp::Report r = comp::run(family(3), C, T, nullptr, o);
    std::printf("      plane, edge_share %g: keys %s; expansions %s; best %s\n", share,
                setting(r.settings, "components_keys").c_str(), expansions_text(T).c_str(),
                comp::key_text(best_part(T)).c_str());
    std::vector<Key> keys;
    for (const auto& kv : T.parts) keys.push_back(kv.first);
    if (share == 0.4) {
        check(keys == plane_keys(1, 7, 1, 3) &&
                  expansions_text(T) == "1:0:hi:3:4:3 2:0:hi:4:5:3 3:0:hi:5:6:3 4:0:hi:6:7:3",
              "plane, edge_share 0.4: a = 7, three keys a layer; b's faces (one of three best points each) stay");
    } else {
        check(keys == plane_keys(1, 7, 0, 4) &&
                  expansions_text(T) == "1:0:hi:3:4:3 1:1:lo:1:0:4 1:1:hi:3:4:4 2:0:hi:4:5:5 3:0:hi:5:6:5 4:0:hi:6:7:5",
              "plane, edge_share 0.3: b's faces move once, each layer over the box at its turn (a = 4 included), "
              "then a to 7 by layers of five");
    }
    bool sixes = true;
    for (long long b = 1; b <= 3; ++b) {
        Key best;
        double q = std::numeric_limits<double>::infinity();
        for (const auto& kv : T.parts)
            if (kv.first[1] == b && kv.second.q < q) {
                q = kv.second.q;
                best = kv.first;
            }
        sixes = sixes && best == Key{6, b};
    }
    check(sixes, "plane: every instance's best part (6, b) found");
}

// ── stopped and run on ─────────────────────────────────────────────────────

void resumed(const std::string& pick, long long B, long long step)
{
    Maker mk;
    mk.cost = 2.5;
    const comp::Components C = parts(line_keys(1, 3), [](const Key& k) { return line_base(k, 8.0); }, false, &mk);
    const fam::Family F = family(4);
    const std::filesystem::path d = temp_dir("resumed");
    comp::Options o = options(pick, {1}, {12});
    o.budget = B;
    o.save_every = 1000000;
    o.table_file = utf8(d / "one.json");
    comp::Table T1;
    comp::run(F, C, T1, nullptr, o);
    const std::string one = read(o.table_file);
    // a stop where the budget cut a piece short is not one (COMP-9: its piece ran with less); the others are
    // stops before an assembly, after a pick, after a make (its assemblies or its piece not run yet)
    long long same = 0, runs = 0, made_once = 0, cut = 0, after_make = 0;
    for (long long b1 = 1; b1 < B; b1 += step) {
        comp::Options a = o;
        a.budget = b1;
        a.table_file = utf8(d / "two.json");
        comp::Table T2;
        mk.calls = 0;
        comp::run(F, C, T2, nullptr, a);
        bool short_piece = false;
        for (const comp::PieceLog& p : T2.pieces) short_piece = short_piece || p.budget != o.piece_budget;
        if (short_piece) {
            ++cut;
            continue;
        }
        for (const Key& k : T2.made_keys)
            if (!T2.touched.count(k)) {
                ++after_make;
                break;
            }
        comp::Table L = comp::load_table(a.table_file);
        a.budget = B;
        comp::run(F, C, L, nullptr, a);
        ++runs;
        same += read(a.table_file) == one ? 1 : 0;
        made_once += mk.calls == static_cast<long long>(L.made_keys.size()) ? 1 : 0;
    }
    check(same == runs && made_once == runs && runs > 0 && after_make > 0,
          "line, " + pick + ": stopped by its budget between pieces at " + std::to_string(runs) + " budgets of 1 ... " +
              std::to_string(B - 1) + " (" + std::to_string(after_make) + " of them after a make, before its piece; " +
              std::to_string(cut) + " that cut a piece short left out) and run on = one call of budget " +
              std::to_string(B) + " (the file), " + std::to_string(same) + " same; every key made once");
}

void resumed_pieces(bool journal)
{
    Maker mk;
    const comp::Components C = parts(plane_keys(1, 3, 1, 3), plane_base, true, &mk);
    const fam::Family F = family(3);
    const std::filesystem::path d = temp_dir("pieces");
    comp::Options o = options("sensitivity", {1, 0}, {10, 5});
    o.edge_share = 0.3;
    o.budget = 2400;
    o.table_file = utf8(d / "one.json");
    comp::Table T1;
    const comp::Report r1 = comp::run(F, C, T1, nullptr, o);
    const std::string one = read(o.table_file);
    long long same = 0, runs = 0;
    for (long long m = 1; m <= static_cast<long long>(T1.pieces.size()); ++m) {
        comp::Options a = o;
        a.max_pieces = m;
        a.table_file = utf8(d / "two.json");
        a.table_journal = journal;
        std::filesystem::remove(mootation::store::file::path_of(comp::journal_path(a.table_file)));
        comp::Table T2;
        comp::run(F, C, T2, nullptr, a);
        comp::Table L = comp::load_table(a.table_file);
        a.max_pieces = 0;
        const comp::Report r2 = comp::run(F, C, L, nullptr, a);
        const comp::Table R = comp::load_table(a.table_file);
        ++runs;
        same += comp::table_text(R, std::stoul(setting(r2.settings, "components_keys")), r2.stopped_by) == one ? 1 : 0;
    }
    check(same == runs && setting(r1.settings, "components_keys") == "35",
          std::string("plane, edge_share 0.3") + (journal ? ", with the journal" : "") + ": stopped after 1 ... " +
              std::to_string(runs) + " pieces and run on = the loop in one piece (the table), " +
              std::to_string(same) + " same");
}

// ── the count ──────────────────────────────────────────────────────────────

void count()
{
    double used[2] = {0.0, 0.0};
    std::size_t pieces[2] = {0, 0};
    long long calls[2] = {0, 0};
    for (int i = 0; i < 2; ++i) {
        Maker mk;
        mk.cost = i == 0 ? 0.0 : 2.5;
        const comp::Components C = parts(line_keys(1, 3), [](const Key& k) { return line_base(k, 8.0); }, false, &mk);
        comp::Options o = options("sensitivity", {1}, {12});
        o.max_pieces = 15;
        comp::Table T;
        comp::run(family(4), C, T, nullptr, o);
        used[i] = T.used;
        pieces[i] = T.pieces.size();
        calls[i] = mk.calls;
    }
    check(pieces[0] == pieces[1] && calls[0] == 9 && calls[1] == 9 && used[1] - used[0] == 2.5 * 9,
          "make_part's cost in used(): 9 parts made at 2.5 add " + json::float_repr(used[1] - used[0]));

    // the made point's evaluation: the table's used() = every evaluation of the parts' problems (no other cost)
    long long evaluations = 0;
    Maker mk;
    comp::Components C = parts(line_keys(1, 3), [](const Key& k) { return line_base(k, 8.0); }, false, &mk);
    C.component_problem = [&evaluations](const Key& k) {
        sc::Problem P = part_problem(line_base(k, 8.0));
        const auto f = P.evaluate;
        P.evaluate = [f, &evaluations](const Vec& x, Vec& y, Vec& g) {
            ++evaluations;
            f(x, y, g);
        };
        return P;
    };
    comp::Options o = options("sensitivity", {1}, {12});
    o.max_pieces = 15;
    comp::Table T;
    comp::run(family(4), C, T, nullptr, o);
    check(T.used == static_cast<double>(evaluations),
          "the made points' evaluations in used(): used " + json::float_repr(T.used) + " = " +
              std::to_string(evaluations) + " evaluations");
}

// ── make_part alone; off by default ────────────────────────────────────────

void make_alone()
{
    Maker mk;
    mk.empty_for = {2};
    const comp::Components C = parts(line_keys(1, 3), [](const Key& k) { return line_base(k, 8.0); }, false, &mk);
    const std::filesystem::path d = temp_dir("alone");
    comp::Options o = options("round_robin", {}, {});
    o.max_pieces = 6;
    o.table_file = utf8(d / "t.json");
    comp::Table T;
    const comp::Report r = comp::run(family(4), C, T, nullptr, o);
    const std::string text = read(o.table_file);
    check(mk.calls == 3 && T.made_keys.size() == 3 && T.made.offered == 2 && T.made.accepted == 2 && T.grown.empty() &&
              setting(r.settings, "components_keys") == "3",
          "make_part alone: the three keys made once each, key {2} given no point (made offered 2), no key grown");
    check(T.parts.at({2}).source == "run" && T.parts.at({1}).piece >= 0 && T.parts.count({3}) == 1,
          "make_part alone: {2}'s piece searched it from no point, source " + T.parts.at({2}).source);
    check(text.find("\"made\"") != std::string::npos && text.find("\"grown\"") == std::string::npos &&
              text.find("key_hard_lo") == std::string::npos && !has_setting(r.settings, "components_grown"),
          "make_part alone: the file keeps the keys made, no grid in it");
}

void off_by_default()
{
    const std::filesystem::path d = temp_dir("off");
    const fam::Family F = family(4);
    comp::Options o = options("sensitivity", {}, {});
    o.max_pieces = 8;
    o.table_file = utf8(d / "plain.json");
    comp::Table T0;
    const comp::Report r0 =
        comp::run(F, parts(line_keys(1, 3), [](const Key& k) { return line_base(k, 8.0); }, false, nullptr), T0,
                  nullptr, o);
    const std::string plain = read(o.table_file);
    check(plain.find("\"made\"") == std::string::npos && plain.find("\"grown\"") == std::string::npos &&
              plain.find("expansions") == std::string::npos && plain.find("key_hard") == std::string::npos &&
              plain.find("edge_") == std::string::npos && plain.find("grow_step") == std::string::npos &&
              !has_setting(r0.settings, "components_grown") && T0.grown.empty() && T0.made_keys.empty(),
          "off by default: no new field in the file, no new setting");

    // make_part where every key has a part: never asked, the file the same
    auto given = [&](const comp::Components& C, const std::string& file) {
        comp::Options a = o;
        a.table_file = utf8(d / file);
        comp::Table T;
        comp::Loop l(F, C, T, nullptr, a);
        for (long long k = 1; k <= 3; ++k) l.give({k}, {0.1 * static_cast<double>(k), 0.2});
        l.run();
        return read(a.table_file);
    };
    Maker mk;
    const std::string without = given(parts(line_keys(1, 3), [](const Key& k) { return line_base(k, 8.0); }, false, nullptr), "g0.json");
    const std::string with = given(parts(line_keys(1, 3), [](const Key& k) { return line_base(k, 8.0); }, false, &mk), "g1.json");
    check(mk.calls == 0 && with == without, "make_part where every key has a part: never asked, the file the same");
}

void refusals()
{
    const fam::Family F = family(4);
    Maker mk;
    const comp::Components C = parts(line_keys(1, 3), [](const Key& k) { return line_base(k, 8.0); }, false, &mk);
    auto refused = [&](const comp::Components& Cx, const comp::Options& o, const std::string& what) {
        bool threw = false;
        try {
            comp::Table T;
            comp::run(F, Cx, T, nullptr, o);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "refused: " + what);
    };
    comp::Options o = options("sensitivity", {1}, {12});
    o.max_pieces = 4;
    comp::Options x = o;
    x.key_hard_lo = {1, 1};
    refused(C, x, "a hard bound of another length than the keys'");
    x = o;
    x.key_hard_hi = {2};
    refused(C, x, "a key outside the hard bounds");
    x = o;
    x.edge_share = 0.0;
    refused(C, x, "edge_share 0");
    x.edge_share = 1.5;
    refused(C, x, "edge_share 1.5");
    x = o;
    x.edge_top_k = 0;
    refused(C, x, "edge_top_k 0");
    x = o;
    x.edge_band = -1;
    refused(C, x, "edge_band -1");
    x = o;
    x.grow_step = 0;
    refused(C, x, "grow_step 0");
    x = o;
    x.pick = "round_robin";
    x.epsilon = 0.0;
    refused(C, x, "a grid with round_robin and no epsilon (the rule's sensitivity)");
    comp::Components Cl = C;
    Cl.keys.push_back({4, 1});
    refused(Cl, o, "a grid of keys of two lengths");
    Maker nan;
    nan.cost = std::numeric_limits<double>::quiet_NaN();
    refused(parts(line_keys(1, 3), [](const Key& k) { return line_base(k, 8.0); }, false, &nan), options("round_robin", {}, {}),
            "make_part's cost NaN");
    Maker three;
    three.length = 3;
    refused(parts(line_keys(1, 3), [](const Key& k) { return line_base(k, 8.0); }, false, &three),
            options("round_robin", {}, {}), "make_part's point not of the part's problem's length");
}

}   // namespace

int main()
{
    try {
        line_grows("sensitivity", 30);
        line_grows("round_robin", 60);
        line_bounds();
        plane(0.4);
        plane(0.3);
        resumed("sensitivity", 700, 1);
        resumed("round_robin", 2000, 7);
        resumed_pieces(false);
        resumed_pieces(true);
        count();
        make_alone();
        off_by_default();
        refusals();
    } catch (const std::exception& e) {
        std::printf("FAIL  an exception: %s\n", e.what());
        ++g_failed;
    }
    if (g_failed) {
        std::printf("%d FAILED\n", g_failed);
        return 1;
    }
    std::printf("all passed\n");
    return 0;
}
