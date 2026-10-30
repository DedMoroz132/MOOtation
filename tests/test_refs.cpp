// SPDX-License-Identifier: Apache-2.0
// The reference builders in C++ (include/mootation/refs, task 17, §2.1) and
// what they stand on:
//   - nearest_other: the k-d tree gives every row's nearest distance bit for
//     bit as every pair does (random rows, copies among them, a grid);
//   - deflate, gzip, zip (io/deflate.hpp): what npz.hpp's inflate reads back
//     is the input, for text, noise, runs, one byte and none; an .npz of
//     npy_bytes reads back as its arrays;
//   - distinct_rows, thin: copies to 12 decimals go, the extremes stay, the
//     draw is the same for the same seed;
//   - the fronts: DTLZ1's rows sum to 1/2, DTLZ2's have norm 1, WFG4's
//     sum (f_m / 2m)^2 = 1, the simplex grid has the size of its formula,
//     exactly_n gives n rows;
//   - an entry of the cache as a record: decoded, the same arrays, meta and
//     hash; the floors of a front are finite, DSS's at least the greedy's.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <mootation/campaign/references.hpp>
#include <mootation/decision_indicators.hpp>
#include <mootation/io/deflate.hpp>
#include <mootation/io/npz.hpp>
#include <mootation/random.hpp>
#include <mootation/refs/build.hpp>
#include <mootation/refs/sets.hpp>

namespace rf = mootation::refs;
namespace js = mootation::json;

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

static void nearest_other_tree()
{
    std::mt19937 g(5);
    for (std::size_t m : {2u, 3u, 5u, 8u}) {
        const std::size_t n = 3000;
        std::vector<double> R(n * m);
        for (double& v : R) v = mootation::rnd::uniform01(g);
        for (std::size_t c = 0; c < m; ++c) R[7 * m + c] = R[3 * m + c];      // a copy: distance 0
        const auto a = mootation::indicators::nearest_other(R.data(), n, m);
        const auto b = mootation::indicators::nearest_other_pairs(R.data(), n, m);
        CHECK(a == b);
        CHECK(a[3] == 0.0 && a[7] == 0.0);
    }
    // a grid: many ties
    std::vector<double> G;
    for (int i = 0; i < 50; ++i)
        for (int j = 0; j < 50; ++j) {
            G.push_back(i / 49.0);
            G.push_back(j / 49.0);
        }
    CHECK(mootation::indicators::nearest_other(G.data(), 2500, 2) ==
          mootation::indicators::nearest_other_pairs(G.data(), 2500, 2));
}

static std::vector<std::uint8_t> bytes_of(const std::string& s)
{
    return std::vector<std::uint8_t>(s.begin(), s.end());
}

static void deflate_round_trip()
{
    std::mt19937 g(9);
    std::string text;
    for (int i = 0; i < 20000; ++i) text += "problem,algorithm," + std::to_string(i % 97) + ",0.12345678\n";
    std::vector<std::uint8_t> noise(100000), runs(70000, 'a');
    for (auto& b : noise) b = static_cast<std::uint8_t>(mootation::rnd::below(g, 256));
    for (std::size_t k = 0; k < runs.size(); k += 1000) runs[k] = 'b';
    const std::vector<std::vector<std::uint8_t>> inputs = {bytes_of(text), noise, runs, {42}, {}};
    for (const auto& in : inputs) {
        const auto z = mootation::deflate::compress(in.data(), in.size());
        CHECK(mootation::npz::inflate(z.data(), z.size()) == in);
        const auto gz = mootation::deflate::gzip(in.data(), in.size());
        CHECK(gz.size() >= 18 && gz[0] == 0x1f && gz[1] == 0x8b && gz[2] == 8);
        CHECK(mootation::npz::inflate(gz.data() + 10, gz.size() - 18) == in);
    }
    CHECK(mootation::deflate::compress(reinterpret_cast<const std::uint8_t*>(text.data()), text.size()).size() <
          text.size() / 5);
    // an .npz of two arrays, read back
    std::vector<double> a = {1.0, -2.5, 3.25, 0.0, 1e-300, 7.0};
    mootation::npz::NpyArray A{"F", "<f8", {3, 2}, {}}, B{"v", "<f8", {6}, {}};
    const auto* p = reinterpret_cast<const std::uint8_t*>(a.data());
    A.data.assign(p, p + 48);
    B.data = A.data;
    const auto npz = mootation::npz::npz_bytes({A, B});
    const auto path = std::filesystem::temp_directory_path() / ("mootation_test_refs_" + std::to_string(std::rand()) + ".npz");
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(npz.data()),
                                                static_cast<std::streamsize>(npz.size()));
    const auto got = mootation::npz::load(path);
    std::filesystem::remove(path);
    CHECK(got.size() == 2 && got.at("F").shape == (std::vector<std::size_t>{3, 2}) && got.at("F").doubles() == a);
    CHECK(got.at("v").shape == std::vector<std::size_t>{6} && got.at("v").doubles() == a);
}

static void thinning()
{
    rf::Rows F = rf::rows_of(0, 2);
    for (int i = 0; i <= 100; ++i) {
        const double r[2] = {i / 100.0, 1.0 - i / 100.0};
        rf::push_row(F, r);
        const double copy[2] = {r[0] + 1e-14, r[1]};          // the same row to 12 decimals
        rf::push_row(F, copy);
    }
    const rf::Rows D = rf::distinct_rows(F);
    CHECK(D.n == 101);
    const rf::Rows T = rf::thin(D, 10, 77u), U = rf::thin(D, 10, 77u);
    CHECK(T.n == 10 && T.v == U.v);
    CHECK(T.row(0)[0] == 0.0 && T.row(9)[0] == 1.0);          // the extremes, in order
    const rf::Rows E = rf::exactly_n([](std::size_t k) { return rf::simplex(3, k); }, 500, 3u);
    CHECK(E.n == 500);
}

static void fronts()
{
    auto spec = [](const std::string& kind, int M) {
        js::Value s = js::Value::object();
        s.set("kind", js::Value::string(kind));
        s.set("M", js::Value::integer(M));
        s.set("seed", js::Value::integer(1));
        return s;
    };
    // the grid of step 1/p on the simplex: C(p + M - 1, M - 1) points
    CHECK(rf::simplex(3, 1000).n == 1000);
    CHECK(rf::detail::simplex_grid(3, 1000).n == 33 * 34 / 2);
    const rf::Rows d1 = rf::front(spec("dtlz1", 5), 1000), d2 = rf::front(spec("dtlz2", 4), 1000),
                   w4 = rf::front(spec("wfg_concave", 3), 1000);
    CHECK(d1.n == 1000 && d2.n == 1000 && w4.n == 1000);
    double worst = 0.0;
    for (std::size_t i = 0; i < d1.n; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < d1.m; ++j) s += d1.row(i)[j];
        worst = std::max(worst, std::fabs(s - 0.5));
    }
    for (std::size_t i = 0; i < d2.n; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < d2.m; ++j) s += d2.row(i)[j] * d2.row(i)[j];
        worst = std::max(worst, std::fabs(s - 1.0));
    }
    for (std::size_t i = 0; i < w4.n; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < w4.m; ++j) {
            const double h = w4.row(i)[j] / (2.0 * static_cast<double>(j + 1));
            s += h * h;
        }
        worst = std::max(worst, std::fabs(s - 1.0));
    }
    CHECK(worst < 1e-12);
    // the Pareto set of DTLZ2_3D: the cube's corners, then Halton, the rest at 1/2
    js::Value ps = js::Value::object();
    ps.set("kind", js::Value::string("dtlz"));
    ps.set("M", js::Value::integer(3));
    ps.set("n_vars", js::Value::integer(12));
    ps.set("rest", js::Value::number(0.5));
    const rf::Rows X = rf::pareto_set(ps, 100);
    CHECK(X.n == 100 && X.m == 12 && X.row(0)[0] == 0.0 && X.row(3)[0] == 1.0 && X.row(3)[1] == 1.0);
    CHECK(X.row(4)[0] == 0.5 && X.row(4)[1] == 1.0 / 3.0 && X.row(50)[11] == 0.5);
}

static void entry_record()
{
    mootation::campaign::ReferenceEntry e;
    e.ideal = mootation::campaign::Matrix{{2}, {0.0, 0.0}};
    e.nadir = mootation::campaign::Matrix{{2}, {1.0, 1.0}};
    e.ref = mootation::campaign::Matrix{{3, 2}, {0.0, 1.0, 0.5, 0.5, 1.0, 0.0}};
    e.meta = js::Value::object();
    e.meta.set("problem", js::Value::string("X"));
    e.meta.set("reference_version", js::Value::integer(5));
    const auto bytes = mootation::campaign::encode_entry("X|3", e);
    const auto back = mootation::campaign::decode_entry(bytes, "test");
    CHECK(back.ref && back.ref->shape == e.ref->shape && back.ref->values == e.ref->values);
    CHECK(back.ideal && back.nadir && !back.ref3 && !back.pset);
    CHECK(js::dump(back.meta) == js::dump(e.meta));
    using mootation::campaign::reference_hash;
    CHECK(reference_hash(back.ideal, back.nadir, back.ref, back.ref3, back.pset) ==
          reference_hash(e.ideal, e.nadir, e.ref, e.ref3, e.pset));
    auto broken = bytes;
    broken[broken.size() - 3] ^= 0x10;
    bool refused = false;
    try {
        mootation::campaign::decode_entry(broken, "test");
    } catch (const std::exception&) {
        refused = true;
    }
    CHECK(refused);
    // the floors of a front of 200 points at a population of 20
    js::Value s = js::Value::object();
    s.set("kind", js::Value::string("zdt1"));
    s.set("seed", js::Value::integer(1));
    const rf::Rows F = rf::front(s, 200);
    const rf::Floors f = rf::igdp_floors(F, rf::Frame{{0.0, 0.0}, {1.0, 1.0}}, 20);
    CHECK(f.rows == 200 && f.distinct == 200 && std::isfinite(f.greedy) && f.greedy > 0.0 && f.dss >= f.greedy);
}

int main()
{
    try {
        nearest_other_tree();
        deflate_round_trip();
        thinning();
        fronts();
        entry_record();
    } catch (const std::exception& ex) {
        std::printf("FAIL: %s\n", ex.what());
        return 1;
    }
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("refs: nearest_other, deflate, thinning, fronts, entries\n");
    return 0;
}
