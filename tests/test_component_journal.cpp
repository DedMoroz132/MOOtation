// SPDX-License-Identifier: Apache-2.0
// The journal of the table of parts (task 33; campaign/components.hpp COMP-13)
// on a synthetic family: twelve parts {1} ... {12}, part {k} the problem
// k / 4 + sum (x_j - 0.5)^2 in dimension 100; instance n (1 ... 12) assembled
// from parts {n} and {n mod 12 + 1}.
//   * the loop with the journal is the loop without it, and the table's file
//     with its journal reads as the whole file, byte for byte; the journal
//     holds saves (the file was written whole again in the run);
//   * a write cut off anywhere in the journal (at every line's end, a byte
//     before it, in its middle) reads as the last whole save: the table of
//     the loop stopped after that save's piece (max_pieces), byte for byte;
//   * a write cut off in writing the file whole: before the rename (the old
//     file and its journal), after it (the new file; the old journal is not
//     read); a loop that goes on from a cut is the loop in one piece;
//   * off by default: no journal written, a file without one reads as before,
//     a journal of another file is not read; the refusals.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

void write_text(const std::filesystem::path& p, const std::string& t)
{
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(t.data(), static_cast<std::streamsize>(t.size()));
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
                                    ("mootation_test_component_journal_" + what + "_" + std::to_string(std::rand()) +
                                     "_" + std::to_string(std::rand()));
    std::filesystem::remove_all(d);
    std::filesystem::create_directories(d);
    return d;
}

// ── the synthetic family ────────────────────────────────────────────────────

const std::size_t kParts = 12, kDim = 100;

sc::Problem component_problem(const Key& k)
{
    sc::Problem P;
    P.lo.assign(kDim, -2.0);
    P.hi.assign(kDim, 2.0);
    P.m = 1;
    const double b = 0.25 * static_cast<double>(k.at(0));
    P.evaluate = [b](const Vec& x, Vec& f, Vec&) {
        double s = b;
        for (double v : x) s += (v - 0.5) * (v - 0.5);
        f.assign(1, s);
    };
    return P;
}

Key other_of(long long n) { return Key{n % static_cast<long long>(kParts) + 1}; }

comp::Assembled assemble(long long n, const comp::View& v)
{
    comp::Assembled a;
    a.cost = 0.25;
    if (!v.has(Key{n}) || !v.has(other_of(n))) return a;
    a.ok = true;
    a.f = v.quality(Key{n}) + v.quality(other_of(n));
    a.used_keys = {Key{n}, other_of(n)};
    return a;
}

comp::Components parts()
{
    comp::Components C;
    for (long long k = 1; k <= static_cast<long long>(kParts); ++k) C.keys.push_back(Key{k});
    C.component_problem = component_problem;
    C.assemble = assemble;
    return C;
}

fam::Family family()
{
    fam::Family F;
    for (long long n = 1; n <= static_cast<long long>(kParts); ++n) F.instances.push_back(n);
    F.weight = [](long long n) { return static_cast<double>(n); };
    return F;
}

comp::Options options(const std::string& pick, long long max_pieces)
{
    comp::Options o;
    o.solver = "nelder_mead";
    o.pop_size = 10;
    o.budget = 1000000;
    o.piece_budget = 120;
    o.max_pieces = max_pieces;
    o.pick = pick;
    o.epsilon = 1.0 / 1024.0;
    o.seed = 5;
    return o;
}

// the table of the loop stopped after p pieces, with the options `as` (those of the loop it is compared with)
std::string table_after(const std::string& pick, long long p, const mootation::json::Value& as)
{
    comp::Table T;
    comp::run(family(), parts(), T, nullptr, options(pick, p));
    T.options = as;
    return comp::table_text(T, kParts, "");
}

// the heads whole in the journal's first `cut` bytes
long long heads_in(const std::string& journal, std::size_t cut)
{
    long long h = 0;
    for (std::size_t at = 0;;) {
        const std::size_t end = journal.find('\n', at);
        if (end == std::string::npos || end + 1 > cut) return h;
        if (journal.compare(at, 9, "{\"head\": ") == 0) ++h;
        at = end + 1;
    }
}

// ── the journal is the whole file ───────────────────────────────────────────

void journal_is_the_whole_file()
{
    const fam::Family F = family();
    const comp::Components C = parts();
    const std::filesystem::path d = temp_dir("whole");
    const std::string wf = utf8(d / "whole.json"), jf = utf8(d / "journal.json");
    for (const std::string pick : {"round_robin", "sensitivity"}) {
        comp::Options o = options(pick, 40);
        o.table_file = wf;
        comp::Table T0;
        const comp::Report r0 = comp::run(F, C, T0, nullptr, o);
        const std::string whole = read_text(wf);
        check(!std::filesystem::exists(comp::journal_path(wf)) && whole == comp::table_text(T0, kParts, r0.stopped_by),
              pick + ": without table_journal the file whole, no journal");

        o.table_file = jf;
        o.table_journal = true;
        comp::Table T1;
        const comp::Report r1 = comp::run(F, C, T1, nullptr, o);
        check(r1.stopped_by == r0.stopped_by && r1.settings == r0.settings &&
                  comp::table_text(T1, kParts, r1.stopped_by) == whole,
              pick + ": the loop with the journal is the loop without it");
        const std::string base = read_text(jf), journal = read_text(comp::journal_path(jf));
        const comp::Table B = comp::table_of_text(base);
        comp::Table J = comp::table_of_text(base);
        const long long saves = comp::apply_journal(J, base, journal);
        std::printf("      %s: the file %zu bytes at piece %zu, the journal %zu bytes, %lld saves\n", pick.c_str(),
                    base.size(), B.pieces.size(), journal.size(), saves);
        check(B.pieces.size() > 1 && B.pieces.size() < 40 && saves >= 2,
              pick + ": the file written whole again in the run, the journal holds saves after it");
        check(comp::table_text(comp::load_table(jf), kParts, r1.stopped_by) == whole &&
                  comp::table_text(J, kParts, r1.stopped_by) == whole,
              pick + ": the file and its journal read as the whole file, byte for byte");
        check(journal.find("{\"part\": ") != std::string::npos &&
                  journal.find("{\"pieces_from\": ") != std::string::npos &&
                  journal.rfind("{\"format\": \"mootation-component-journal\", \"version\": 1, \"base_bytes\": " +
                                    std::to_string(base.size()) + ", \"base_sha256\": \"",
                                0) == 0,
              pick + ": the journal's first line names the file; parts and pieces in its saves");
    }
    std::filesystem::remove_all(d);
}

// ── a write cut off ─────────────────────────────────────────────────────────

void cut_in_the_journal()
{
    const fam::Family F = family();
    const comp::Components C = parts();
    const std::filesystem::path d = temp_dir("cut");
    const std::string jf = utf8(d / "parts.json");
    for (const std::string pick : {"round_robin", "sensitivity"})
        for (long long P : {25LL, 33LL, 40LL}) {
            comp::Options o = options(pick, P);
            o.table_file = jf;
            o.table_journal = true;
            o.save_every = 1;
            comp::Table T;
            comp::run(F, C, T, nullptr, o);
            const std::string base = read_text(jf), journal = read_text(comp::journal_path(jf));
            const std::size_t at = comp::table_of_text(base).pieces.size();
            std::vector<std::string> after(static_cast<std::size_t>(P) + 1);   // the table after piece p
            std::vector<std::size_t> cuts = {0, journal.size()};
            for (std::size_t i = 0; i < journal.size(); ++i)
                if (journal[i] == '\n') {
                    cuts.push_back(i);                                         // the newline cut off
                    cuts.push_back(i + 1);                                     // the line whole
                }
            for (std::size_t i = 0, start = 0; i < journal.size(); ++i)
                if (journal[i] == '\n') {
                    cuts.push_back((start + i) / 2);                           // in the line's middle
                    start = i + 1;
                }
            bool same = true, counted = true;
            for (std::size_t c : cuts) {
                const long long h = heads_in(journal, c);
                const std::size_t p = (std::min)(at + static_cast<std::size_t>(h), static_cast<std::size_t>(P));
                if (after[p].empty()) after[p] = table_after(pick, static_cast<long long>(p), T.options);
                comp::Table R = comp::table_of_text(base);
                counted = counted && comp::apply_journal(R, base, journal.substr(0, c)) == h;
                same = same && comp::table_text(R, kParts, "") == after[p];
            }
            check(same && counted && cuts.size() > 20,
                  pick + ", " + std::to_string(P) + " pieces: the journal cut at " + std::to_string(cuts.size()) +
                      " places reads as its last whole save (the loop stopped after its piece), byte for byte");
            // a loop that goes on from a cut in the journal's last save is the loop in one piece
            if (P == 40) {
                const std::size_t c = journal.size() - 3;
                write_text(comp::journal_path(jf), journal.substr(0, c));
                comp::Table R = comp::load_table(jf);
                comp::Options on = options(pick, 60);
                on.table_file = jf;
                on.table_journal = true;
                const comp::Report r = comp::run(F, C, R, nullptr, on);
                comp::Table one;
                comp::run(F, C, one, nullptr, options(pick, 60));
                check(comp::table_text(R, kParts, "") == comp::table_text(one, kParts, "") &&
                          comp::table_text(comp::load_table(jf), kParts, r.stopped_by) ==
                              comp::table_text(one, kParts, r.stopped_by),
                      pick + ": a loop that goes on from a journal cut off is the loop in one piece (its file too)");
            }
        }
    std::filesystem::remove_all(d);
}

void cut_in_the_whole_write()
{
    const fam::Family F = family();
    const comp::Components C = parts();
    const std::filesystem::path d = temp_dir("rename");
    const std::string jf = utf8(d / "parts.json");
    const std::string pick = "sensitivity";
    comp::Options o = options(pick, 20);
    o.table_file = jf;
    o.table_journal = true;
    comp::Table T1;
    comp::run(F, C, T1, nullptr, o);
    const std::string b1 = read_text(jf), j1 = read_text(comp::journal_path(jf));
    const std::string s1 = comp::table_text(T1, kParts, "");
    {
        comp::Table T2 = comp::load_table(jf);
        comp::Options o2 = o;
        o2.max_pieces = 30;
        comp::run(F, C, T2, nullptr, o2);
    }
    const std::string b2 = read_text(jf);
    check(b2 != b1 && !j1.empty(), "rename: the second call wrote the file anew");

    // cut off before the rename: the old file, its journal and a .tmp half written
    write_text(jf, b1);
    write_text(comp::journal_path(jf), j1);
    write_text(jf + ".tmp", b2.substr(0, b2.size() / 2));
    check(comp::table_text(comp::load_table(jf), kParts, "") == s1,
          "rename: cut off before the rename, the old file and its journal are read (the .tmp is not)");
    {
        comp::Table R = comp::load_table(jf);
        comp::Options on = options(pick, 40);
        on.table_file = jf;
        on.table_journal = true;
        comp::run(F, C, R, nullptr, on);
        comp::Table one;
        comp::run(F, C, one, nullptr, options(pick, 40));
        check(comp::table_text(R, kParts, "") == comp::table_text(one, kParts, ""),
              "rename: a loop that goes on from there is the loop in one piece");
    }
    // cut off after the rename, before the old journal is removed: the new file alone
    write_text(jf, b2);
    write_text(comp::journal_path(jf), j1);
    comp::Table R = comp::table_of_text(b2);
    check(comp::apply_journal(R, b2, j1) == 0 &&
              comp::table_text(comp::load_table(jf), kParts, "") == comp::table_text(comp::table_of_text(b2), kParts, ""),
          "rename: cut off after the rename, the new file is read, the old journal is not");
    std::filesystem::remove_all(d);
}

// ── off by default and the refusals ─────────────────────────────────────────

template <class Fn>
bool refused(Fn f)
{
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void off_by_default_and_refusals()
{
    const fam::Family F = family();
    const comp::Components C = parts();
    const std::filesystem::path d = temp_dir("off");
    const std::string tf = utf8(d / "parts.json"), jf = utf8(d / "journal.json");
    check(!comp::Options().table_journal, "off by default: table_journal false");
    comp::Options o = options("round_robin", 15);
    o.table_file = jf;
    o.table_journal = true;
    comp::Table T;
    comp::run(F, C, T, nullptr, o);
    const std::string base = read_text(jf), journal = read_text(comp::journal_path(jf));

    // a whole file with the journal of another file beside it: the file alone
    o.table_file = tf;
    o.table_journal = false;
    comp::Table W;
    comp::run(F, C, W, nullptr, o);
    const std::string whole = read_text(tf);
    write_text(comp::journal_path(tf), journal);
    check(comp::table_text(comp::load_table(tf), kParts, "") == comp::table_text(comp::table_of_text(whole), kParts, ""),
          "off: a file with a journal of another file beside it reads as the file alone");

    comp::Options bad = options("round_robin", 1);
    bad.table_journal = true;
    comp::Table B0;
    check(refused([&] { comp::run(F, C, B0, nullptr, bad); }), "refused: a table_journal without a table_file");
    auto apply = [&](const std::string& j) {
        comp::Table R = comp::table_of_text(base);
        return comp::apply_journal(R, base, j);
    };
    const std::size_t eol = journal.find('\n');
    std::string other = journal;
    other.replace(other.find("\"version\": 1"), 12, "\"version\": 2");
    check(refused([&] { apply(other); }), "refused: a journal of another version");
    check(refused([&] { apply("{\"format\": \"mootation-family-bank\", \"version\": 1}\n"); }),
          "refused: a journal of another format");
    check(refused([&] { apply(journal.substr(0, eol + 1) + "{\"parts\": []}\n"); }),
          "refused: a journal's line that is not a part, pieces or a head");
    check(apply(journal.substr(0, eol)) == 0 && apply("") == 0, "a journal without a whole first line: nothing read");
    std::filesystem::remove_all(d);
}

}   // namespace

int main()
{
    try {
        journal_is_the_whole_file();
        cut_in_the_journal();
        cut_in_the_whole_write();
        off_by_default_and_refusals();
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
