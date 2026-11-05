#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Symmetries of a problem (task 19, §19.2.1): groups of interchangeable
// blocks of coordinates, a canonical representative, the alignment of one
// point to another and the distance that follows from it.
//
//   Blocks{offset, size, count}: count blocks of size coordinates from offset
//     on may be permuted (block k is x[offset + k size, offset + (k + 1)
//     size)); the coordinates in no group are free.
//   canon(x): the problem's own canonical representative, f(canon(x)) = f(x),
//     canon(canon(x)) = canon(x) (optional).
//   align(x, ref): canon(x) where the problem gives canon, then in every
//     group the permutation of x's blocks that minimizes the sum of squared
//     distances to ref's blocks; block j of the result is the block of x
//     assigned to ref's block j.
//   distance(x, y) = ||align(x, y) - y||.
//
// The assignment: the Hungarian method of H.W. Kuhn, "The Hungarian method
// for the assignment problem", Naval Res. Logist. Q. 2 (1955) 83-97, in the
// form of J. Munkres, "Algorithms for the assignment and transportation
// problems", J. SIAM 5 (1957) 32-38, §1 (sources hungarian_kuhn1955,
// hungarian_munkres1957), on the count x count matrix of squared block
// distances: the Preliminaries (each row, then each column, less its least
// element; a zero starred where its row and column have none; the starred
// columns covered), Step 1 (a non-covered zero primed; no starred zero in
// its row: Step 2; else its row covered, the star's column uncovered),
// Step 2 (the alternating sequence of primes and stars from it: the stars
// unstarred, the primes starred; primes erased, rows uncovered, the starred
// columns covered; all covered: done), Step 3 (h the least non-covered
// element, added to the covered rows, taken from the uncovered columns; to
// Step 1 with nothing else changed). Munkres shows the method finite for
// real entries (after his operation count): at most n applications of
// Step 3 before the number of starred zeros grows.
//
// Readings (OURS):
//   SYM-1. Ties: Step 1 takes the first non-covered zero in row-major order
//     and the Preliminaries star in row-major order, so the assignment is a
//     function of the matrix: equal blocks or equal costs give the same
//     answer every time.
//   SYM-2. A zero is an entry equal to 0.0. In floating point Step 3 makes
//     the least non-covered entry exactly 0 (y - y), so the zero the method
//     needs is there; the other entries carry the rounding of the
//     subtractions, which can change the choice only between assignments
//     whose costs differ by that rounding. Checked against every permutation
//     for count <= 7 (tests/test_symmetry.cpp).
//   SYM-3. align(x, ref) with x a permutation of ref's blocks returns ref
//     bit for bit: the permutation's cost is 0, the least possible, and the
//     result copies x's blocks, which are ref's.
//   SYM-4. The groups are aligned one after another, each on its own
//     coordinates (the task's blocks do not overlap; overlapping groups are
//     refused).
//   SYM-5. sym = canon where the problem gives no canon: each group's blocks
//     sorted lexicographically (stably), the least member of x's orbit under
//     the group: f(canon(x)) = f(x), canon(canon(x)) = canon(x) and
//     canon(P(x)) = canon(x) for every permutation P of the blocks.
//   SYM-6. A group's blocks have the same bounds, coordinate by coordinate,
//     so that a permutation stays in the box; refused otherwise. A problem's
//     own canon and align keep the box (its contract; not checked).
//   Where the kernels use it (§19.2.2; sym = off | canon | align, off by
//   default: the kernels as before, bit for bit; Sym below):
//   SYM-7. ga (GA's crossovers): align — the second parent aligned to the
//     first, canon — both parents canonical, before the crossover; the
//     children are in the first parent's representation (align).
//   SYM-8. de, jade, shade, lshade: align — the base vector and the
//     difference vectors aligned to the target x_i before the mutation;
//     canon — the target too canonical, and the trial's crossover and the
//     bound repair take canon(x_i). JADE's x_pbest, x_r1 and x~_r2 (an
//     archive member included) likewise.
//   SYM-9. pso, spso2011: align only — the particle's own best and its
//     neighbourhood's best aligned to its position before it moves; the
//     velocity stays in the particle's representation (§19.2.2). canon is
//     refused: it would change the position's representation and not the
//     velocity's.
//   SYM-10. The points a kernel makes are evaluated and stored as made; the
//     Evaluator's archive (repeats) is by the raw x.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../fp_contract.hpp"

namespace mootation::scalar::symmetry {

using Vec = std::vector<double>;

struct Blocks {
    std::size_t offset = 0, size = 1, count = 0;
};

// The Hungarian method on the n x n matrix c (row-major): assignment[i] =
// the column given to row i, the sum of c[i][assignment[i]] least.
inline std::vector<std::size_t> hungarian(std::size_t n, Vec c)
{
    std::vector<std::size_t> assignment(n);
    if (n == 0) return assignment;
    if (c.size() != n * n) throw std::invalid_argument("hungarian: an n x n matrix is needed");
    for (double v : c)
        if (!std::isfinite(v)) throw std::invalid_argument("hungarian: the entries must be finite");
    auto at = [&](std::size_t i, std::size_t j) -> double& { return c[i * n + j]; };
    // Preliminaries
    for (std::size_t i = 0; i < n; ++i) {
        double m = at(i, 0);
        for (std::size_t j = 1; j < n; ++j) m = std::min(m, at(i, j));
        for (std::size_t j = 0; j < n; ++j) at(i, j) -= m;
    }
    for (std::size_t j = 0; j < n; ++j) {
        double m = at(0, j);
        for (std::size_t i = 1; i < n; ++i) m = std::min(m, at(i, j));
        for (std::size_t i = 0; i < n; ++i) at(i, j) -= m;
    }
    const std::size_t none = n;
    std::vector<std::size_t> star_in_row(n, none), star_in_col(n, none), prime_in_row(n, none);
    std::vector<char> row_covered(n, 0), col_covered(n, 0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (at(i, j) == 0.0 && star_in_row[i] == none && star_in_col[j] == none) {
                star_in_row[i] = j;
                star_in_col[j] = i;
            }
    auto cover_starred_columns = [&]() {
        std::size_t covered = 0;
        for (std::size_t j = 0; j < n; ++j) {
            col_covered[j] = star_in_col[j] != none;
            covered += col_covered[j] ? 1 : 0;
        }
        return covered;
    };
    std::size_t covered = cover_starred_columns();
    while (covered < n) {
        // Step 1: the first non-covered zero, row-major (SYM-1)
        std::size_t zi = none, zj = none;
        for (std::size_t i = 0; i < n && zi == none; ++i) {
            if (row_covered[i]) continue;
            for (std::size_t j = 0; j < n; ++j)
                if (!col_covered[j] && at(i, j) == 0.0) {
                    zi = i;
                    zj = j;
                    break;
                }
        }
        if (zi == none) {                             // Step 3
            double h = std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < n; ++i)
                if (!row_covered[i])
                    for (std::size_t j = 0; j < n; ++j)
                        if (!col_covered[j]) h = std::min(h, at(i, j));
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j) {
                    if (row_covered[i]) at(i, j) += h;
                    if (!col_covered[j]) at(i, j) -= h;
                }
            continue;
        }
        prime_in_row[zi] = zj;
        if (star_in_row[zi] != none) {
            row_covered[zi] = 1;
            col_covered[star_in_row[zi]] = 0;
            continue;
        }
        // Step 2: the alternating sequence from the primed zero (zi, zj)
        std::size_t i = zi, j = zj;
        for (;;) {
            const std::size_t i_star = star_in_col[j];   // the starred zero in the prime's column
            star_in_row[i] = j;                       // the prime starred
            star_in_col[j] = i;
            if (i_star == none) break;
            const std::size_t j_prime = prime_in_row[i_star];   // the prime in the star's row
            star_in_row[i_star] = none;               // the star unstarred (its column is now the prime's)
            i = i_star;
            j = j_prime;
        }
        prime_in_row.assign(n, none);
        row_covered.assign(n, 0);
        covered = cover_starred_columns();
    }
    for (std::size_t i = 0; i < n; ++i) assignment[i] = star_in_row[i];
    return assignment;
}

// The groups checked: within x's length, not overlapping (SYM-4).
inline void check(const std::vector<Blocks>& groups, std::size_t n)
{
    std::vector<char> used(n, 0);
    for (const Blocks& b : groups) {
        if (b.size == 0 || b.count == 0 || b.offset + b.size * b.count > n)
            throw std::invalid_argument("symmetry: a group of blocks needs size, count >= 1 within the variables");
        for (std::size_t k = b.offset; k < b.offset + b.size * b.count; ++k) {
            if (used[k]) throw std::invalid_argument("symmetry: the groups of blocks overlap");
            used[k] = 1;
        }
    }
}

// align(x, ref): canon (where given), then each group's blocks permuted to
// ref's by the Hungarian method on the squared distances.
inline Vec align(const Vec& x, const Vec& ref, const std::vector<Blocks>& groups,
                 const std::function<void(Vec&)>& canon = {})
{
    if (x.size() != ref.size()) throw std::invalid_argument("symmetry: x and ref differ in length");
    Vec y = x;
    if (canon) canon(y);
    Vec out = y;
    for (const Blocks& b : groups) {
        const std::size_t m = b.count, s = b.size;
        Vec cost(m * m);
        for (std::size_t i = 0; i < m; ++i)           // x's block i against ref's block j
            for (std::size_t j = 0; j < m; ++j) {
                double sum = 0.0;
                for (std::size_t k = 0; k < s; ++k) {
                    const double d = y[b.offset + i * s + k] - ref[b.offset + j * s + k];
                    sum += d * d;
                }
                cost[i * m + j] = sum;
            }
        const std::vector<std::size_t> a = hungarian(m, cost);
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t k = 0; k < s; ++k) out[b.offset + a[i] * s + k] = y[b.offset + i * s + k];
    }
    return out;
}

inline double distance(const Vec& x, const Vec& y, const std::vector<Blocks>& groups,
                       const std::function<void(Vec&)>& canon = {})
{
    const Vec a = align(x, y, groups, canon);
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) sum += (a[i] - y[i]) * (a[i] - y[i]);
    return std::sqrt(sum);
}

// The built-in canon (SYM-5): each group's blocks sorted lexicographically.
inline void sort_blocks(Vec& x, const std::vector<Blocks>& groups)
{
    for (const Blocks& b : groups) {
        std::vector<Vec> blocks(b.count);
        for (std::size_t k = 0; k < b.count; ++k)
            blocks[k].assign(x.begin() + static_cast<std::ptrdiff_t>(b.offset + k * b.size),
                             x.begin() + static_cast<std::ptrdiff_t>(b.offset + (k + 1) * b.size));
        std::stable_sort(blocks.begin(), blocks.end());
        for (std::size_t k = 0; k < b.count; ++k)
            std::copy(blocks[k].begin(), blocks[k].end(), x.begin() + static_cast<std::ptrdiff_t>(b.offset + k * b.size));
    }
}

// The groups' bounds (SYM-6): every block of a group has its first block's.
inline void check_bounds(const std::vector<Blocks>& groups, const Vec& lo, const Vec& hi)
{
    for (const Blocks& b : groups)
        for (std::size_t k = 1; k < b.count; ++k)
            for (std::size_t c = 0; c < b.size; ++c) {
                const std::size_t i = b.offset + c, j = b.offset + k * b.size + c;
                if (lo[j] != lo[i] || hi[j] != hi[i])
                    throw std::invalid_argument("symmetry: the blocks of a group need the same bounds (SYM-6)");
            }
}

enum class Mode { off, canon, align };

// What a kernel does with the points it combines (§19.2.2, SYM-7 ... 10):
// nothing (off), canon(x) or x aligned to the point it is combined with. The
// problem's canon and align where it gives them, else SYM-5's canon and
// align() above on the groups.
struct Sym {
    Mode mode = Mode::off;
    std::vector<Blocks> groups;
    std::function<void(Vec&)> canon;
    std::function<void(const Vec& x, const Vec& ref, Vec& out)> align;

    bool on() const { return mode != Mode::off; }

    Vec canonical(const Vec& x) const
    {
        Vec y = x;
        if (canon) canon(y);
        else sort_blocks(y, groups);
        return y;
    }

    Vec aligned(const Vec& x, const Vec& ref) const
    {
        if (!align) return symmetry::align(x, ref, groups, canon);
        Vec out;
        align(x, ref, out);
        return out;
    }

    // x as the kernel combines it beside ref: x itself (off), canon(x) or x
    // aligned to ref; `keep` holds the copy
    const Vec& view(const Vec& x, const Vec& ref, Vec& keep) const
    {
        if (mode == Mode::off) return x;
        keep = mode == Mode::canon ? canonical(x) : aligned(x, ref);
        return keep;
    }
};

}   // namespace mootation::scalar::symmetry
