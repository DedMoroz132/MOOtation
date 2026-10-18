// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// nd_tree.hpp: the ND-tree gives the answers of comparing a point with every
// point of the set, so the run archive (grid_archive.hpp) and the
// nondominated filter (nondominated.hpp) are the same with it as without.
// 10 000 random streams of points at 4, 5, 8 and 10 objectives (and some at 3),
// with copies, equal coordinates and points far out: the archive with the
// tree and the archive with the pass over every stored point hold the same
// points in the same order, the same extremes and the same frame, bit for
// bit; the filter keeps the rows the pairwise comparison keeps.
// ============================================================================

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <mootation/grid_archive.hpp>
#include <mootation/nd_tree.hpp>
#include <mootation/nondominated.hpp>

namespace {

int checks = 0, failed = 0;

void check(bool ok, const std::string& what)
{
    ++checks;
    if (!ok) {
        ++failed;
        if (failed <= 20) std::printf("  FAIL  %s\n", what.c_str());
    }
}

// n points of m objectives: a noisy front (many mutually nondominated),
// coordinates on a grid of 1/q now and then (equal values), copies of earlier
// points, and a few far out (the archive's own frame regrids)
std::vector<double> stream(std::mt19937_64& rng, std::size_t n, std::size_t m)
{
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<double> F;
    F.reserve(n * m);
    const double q = 1 << (2 + rng() % 5);
    const double spread = 0.02 + 0.3 * u(rng);
    for (std::size_t i = 0; i < n; ++i) {
        const double r = u(rng);
        if (i > 0 && r < 0.1) {                                    // a copy
            const std::size_t k = rng() % i;
            for (std::size_t j = 0; j < m; ++j) F.push_back(F[k * m + j]);
            continue;
        }
        std::vector<double> w(m);
        double s = 0.0;
        for (auto& v : w) { v = -std::log(1.0 - u(rng)); s += v; }
        const double depth = 1.0 + spread * u(rng);
        for (std::size_t j = 0; j < m; ++j) {
            double v = depth * w[j] / s;
            if (r < 0.4) v = std::floor(v * q) / q;                 // equal coordinates
            if (r > 0.995) v = v * 50.0 - 10.0;                     // far out
            F.push_back(v);
        }
    }
    return F;
}

bool same_archive(const mootation::GridArchive& a, const mootation::GridArchive& b)
{
    std::vector<double> Fa, Xa, Fb, Xb, la, ha, lb, hb;
    std::vector<char> ea, eb;
    a.points(Fa, Xa, ea);
    b.points(Fb, Xb, eb);
    const bool fa = a.frame(la, ha), fb = b.frame(lb, hb);
    return Fa == Fb && Xa == Xb && ea == eb && a.offered() == b.offered() &&
           a.accepted() == b.accepted() && fa == fb && la == lb && ha == hb;
}

// the filter as it was: every row against the nondominated rows before it
std::vector<char> pairwise(const double* F, std::size_t m, const std::vector<std::size_t>& rows)
{
    std::vector<char> keep(rows.size(), 1);
    std::vector<const double*> front;
    for (std::size_t t = 0; t < rows.size(); ++t) {
        const double* f = F + rows[t] * m;
        if (mootation::nd::has_nan(f, m)) continue;
        bool dominated = false;
        for (const double* p : front)
            if (mootation::nd::weakly_dominates(p, f, m)) { dominated = true; break; }
        if (dominated) { keep[t] = 0; continue; }
        front.push_back(f);
    }
    return keep;
}

}   // namespace

int main()
{
    std::mt19937_64 rng(20261018);
    const std::size_t sizes[] = {3, 4, 5, 8, 10};
    int streams = 0;
    for (int t = 0; t < 10000; ++t) {
        const std::size_t m = sizes[t % 5];
        const std::size_t n = 20 + rng() % 400;
        const std::vector<double> F = stream(rng, n, m);
        std::vector<double> X(n * 2);
        for (std::size_t i = 0; i < X.size(); ++i) X[i] = static_cast<double>(i);
        const bool own_frame = t % 3 == 0;
        const std::vector<double> ideal = own_frame ? std::vector<double>{} : std::vector<double>(m, 0.0);
        const std::vector<double> nadir = own_frame ? std::vector<double>{} : std::vector<double>(m, 1.0);
        mootation::GridArchive tree(static_cast<int>(m), 2, ideal, nadir, t % 2 ? 1e-3 : 0.05);
        mootation::GridArchive pass(static_cast<int>(m), 2, ideal, nadir, t % 2 ? 1e-3 : 0.05);
        tree.set_tree(true);
        pass.set_tree(false);
        bool same_each = true;
        for (std::size_t i = 0; i < n; ++i) {
            const bool a = tree.add(F.data() + i * m, X.data() + i * 2);
            const bool b = pass.add(F.data() + i * m, X.data() + i * 2);
            same_each = same_each && a == b;
        }
        check(same_each && same_archive(tree, pass),
              "stream " + std::to_string(t) + " (M = " + std::to_string(m) + "): the same archive");
        const std::vector<std::size_t> rows = mootation::nd::unique_rows(F.data(), n, m);
        check(mootation::nd::nondominated_sorted(F.data(), m, rows) == pairwise(F.data(), m, rows),
              "stream " + std::to_string(t) + ": the filter keeps the same rows");
        ++streams;
    }
    // a long stream, whose archive compacts its slots and rebuilds the tree
    for (std::size_t m : {4u, 6u}) {
        const std::vector<double> F = stream(rng, 6000, m);
        mootation::GridArchive tree(static_cast<int>(m), 0, {}, {}, 0.0);
        mootation::GridArchive pass(static_cast<int>(m), 0, {}, {}, 0.0);
        tree.set_tree(true);
        pass.set_tree(false);
        for (std::size_t i = 0; i < 6000; ++i) {
            tree.add(F.data() + i * m, nullptr);
            pass.add(F.data() + i * m, nullptr);
        }
        check(same_archive(tree, pass), "a long stream: the same archive, M = " + std::to_string(m));
    }
    // the tree alone: removals, a set emptied and filled again
    {
        mootation::nd::NDTree t(3, 2, 2);
        const double a[] = {0.0, 1.0, 2.0}, b[] = {1.0, 0.0, 2.0}, c[] = {2.0, 2.0, 0.0},
                     d[] = {0.5, 0.5, 3.0}, e[] = {3.0, 3.0, 3.0};
        t.insert(0, a);
        t.insert(1, b);
        t.insert(2, c);
        t.insert(3, d);
        std::vector<std::size_t> beaten;
        check(t.covered(e, &beaten), "a point every point dominates is covered");
        const double f[] = {0.0, 0.0, 0.0};
        check(!t.covered(f, &beaten) && beaten == std::vector<std::size_t>{0, 1, 2, 3},
              "a point that dominates all: every id, ascending");
        for (std::size_t id : {2u, 0u, 3u, 1u}) t.erase(id);
        check(t.size() == 0 && !t.covered(e, &beaten) && beaten.empty(), "emptied: covers nothing");
        t.insert(7, c);
        check(t.covered(e, nullptr) && !t.covered(a, nullptr), "filled again");
    }
    std::printf("nd_tree: %d/%d checks passed (%d streams)%s\n", checks - failed, checks, streams,
                failed ? ", FAILED" : "");
    return failed == 0 ? 0 : 1;
}
