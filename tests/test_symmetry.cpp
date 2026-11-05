// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/symmetry.hpp (task 19, §19.2.4):
//   - the Hungarian method against every permutation for n = 1 ... 7: the
//     same least cost, on random matrices, on integer matrices with many
//     ties and on a constant one;
//   - align(P(x), x) = x bit for bit for permutations P of the blocks, with
//     two groups and free coordinates between them; distance(P(x), x) = 0;
//   - align with canon: canon applied first;
//   - overlapping groups and a group past the variables refused;
//   - the built-in canon (SYM-5): canon(P(x)) = canon(x), idempotent;
//   - the kernels' steps with sym = align on x and on P(x) (every point but
//     the one the others are aligned to permuted) and with sym = canon (every
//     point permuted) give the same offspring bit for bit, from the same
//     generator state: de::trial, pbest::trial (jade, shade, lshade),
//     ga::cross (with and without block_uniform), pso::move, spso2011::move;
//     sym = off on the same inputs differs (the check can fail);
//   - block_uniform and block_mutation do what GA-7 and GA-8 say;
//   - through scalar::run: sym = off given runs as no knob, the knobs go to
//     the settings only where given, and the refusals (no symmetries, pso
//     with canon, block operators without blocks, blocks with two boxes).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/random.hpp>
#include <mootation/scalar/de.hpp>
#include <mootation/scalar/ga.hpp>
#include <mootation/scalar/pbest.hpp>
#include <mootation/scalar/pso.hpp>
#include <mootation/scalar/scalar.hpp>
#include <mootation/scalar/spso2011.hpp>
#include <mootation/scalar/symmetry.hpp>

namespace {

namespace S = mootation::scalar::symmetry;
using S::Vec;
using Rows = std::vector<Vec>;

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    if (!ok) ++g_failed;
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
}

double cost_of(std::size_t n, const Vec& c, const std::vector<std::size_t>& a)
{
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += c[i * n + a[i]];
    return s;
}

double brute_force(std::size_t n, const Vec& c)
{
    std::vector<std::size_t> p(n);
    std::iota(p.begin(), p.end(), std::size_t{0});
    double best = cost_of(n, c, p);
    while (std::next_permutation(p.begin(), p.end())) best = std::min(best, cost_of(n, c, p));
    return best;
}

bool is_permutation(const std::vector<std::size_t>& a)
{
    std::vector<std::size_t> s = a;
    std::sort(s.begin(), s.end());
    for (std::size_t i = 0; i < s.size(); ++i)
        if (s[i] != i) return false;
    return true;
}

void hungarian_against_every_permutation()
{
    std::mt19937 rng(19);
    double worst = 0.0;
    bool perms = true;
    int cases = 0;
    for (std::size_t n = 1; n <= 7; ++n)
        for (int rep = 0; rep < 60; ++rep) {
            Vec c(n * n);
            for (double& v : c) {
                if (rep % 3 == 0) v = mootation::rnd::uniform01(rng);
                else if (rep % 3 == 1) v = std::floor(4.0 * mootation::rnd::uniform01(rng));   // ties
                else v = 1e3 * mootation::rnd::uniform01(rng) * mootation::rnd::uniform01(rng);
            }
            const std::vector<std::size_t> a = S::hungarian(n, c);
            perms = perms && a.size() == n && is_permutation(a);
            const double b = brute_force(n, c), h = cost_of(n, c, a);
            worst = std::max(worst, (h - b) / std::max(1.0, std::fabs(b)));
            ++cases;
        }
    check(perms && worst <= 1e-12, "hungarian: the least cost of every permutation, n = 1 ... 7 (" +
                                       std::to_string(cases) + " matrices, worst excess " + std::to_string(worst) + ")");
    const std::vector<std::size_t> flat = S::hungarian(5, Vec(25, 2.5));
    check(is_permutation(flat), "hungarian: a constant matrix gives a permutation");
    bool same = true;
    for (int k = 0; k < 3; ++k) same = same && S::hungarian(5, Vec(25, 2.5)) == flat;
    check(same, "hungarian: ties broken the same way every time (SYM-1)");
}

void align_of_a_permutation()
{
    // two groups: 4 blocks of 3 from 1 on, 5 blocks of 2 from 14 on; 0 and 13 free
    const std::vector<S::Blocks> groups{{1, 3, 4}, {14, 2, 5}};
    std::mt19937 rng(7);
    bool exact = true, zero = true;
    for (int rep = 0; rep < 200; ++rep) {
        Vec x(24);
        for (double& v : x) v = 2.0 * mootation::rnd::uniform01(rng) - 1.0;
        Vec px = x;
        for (const S::Blocks& b : groups) {
            std::vector<std::size_t> p(b.count);
            std::iota(p.begin(), p.end(), std::size_t{0});
            std::shuffle(p.begin(), p.end(), rng);
            for (std::size_t k = 0; k < b.count; ++k)
                for (std::size_t c = 0; c < b.size; ++c)
                    px[b.offset + k * b.size + c] = x[b.offset + p[k] * b.size + c];
        }
        exact = exact && S::align(px, x, groups) == x;
        zero = zero && S::distance(px, x, groups) == 0.0;
    }
    check(exact, "align(P(x), x) = x bit for bit, 200 permutations of two groups (SYM-3)");
    check(zero, "distance(P(x), x) = 0");
    // canon first: here a sign flip of every block's first coordinate where it is negative
    auto canon = [](Vec& v) {
        for (std::size_t k = 0; k < 4; ++k)
            if (v[1 + 3 * k] < 0.0)
                for (std::size_t c = 0; c < 3; ++c) v[1 + 3 * k + c] = -v[1 + 3 * k + c];
    };
    Vec x{0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
    Vec y = x;
    for (std::size_t c = 0; c < 3; ++c) y[4 + c] = -y[4 + c];    // block 1 flipped
    std::swap_ranges(y.begin() + 1, y.begin() + 4, y.begin() + 10);   // blocks 0 and 3 swapped
    check(S::align(y, x, {{1, 3, 4}}, canon) == x, "align with canon: the flip undone, then the blocks");
    bool refused = true;
    for (const std::vector<S::Blocks>& bad :
         {std::vector<S::Blocks>{{0, 3, 2}, {4, 2, 2}}, std::vector<S::Blocks>{{0, 4, 4}}}) {
        try {
            S::check(bad, 13);
            refused = false;
        } catch (const std::invalid_argument&) {
        }
    }
    check(refused, "overlapping groups and a group past the variables refused");
}

// x with each group's blocks permuted at random
Vec permuted(const Vec& x, const std::vector<S::Blocks>& groups, std::mt19937& rng)
{
    Vec y = x;
    for (const S::Blocks& b : groups) {
        std::vector<std::size_t> p(b.count);
        std::iota(p.begin(), p.end(), std::size_t{0});
        std::shuffle(p.begin(), p.end(), rng);
        for (std::size_t k = 0; k < b.count; ++k)
            for (std::size_t c = 0; c < b.size; ++c) y[b.offset + k * b.size + c] = x[b.offset + p[k] * b.size + c];
    }
    return y;
}

Rows random_rows(std::size_t n, std::size_t d, std::mt19937& rng)
{
    Rows X(n, Vec(d));
    for (Vec& x : X)
        for (double& v : x) v = 4.0 * mootation::rnd::uniform01(rng) - 2.0;
    return X;
}

// every row but `keep` permuted (keep = n: every row)
Rows permuted_rows(const Rows& X, std::size_t keep, const std::vector<S::Blocks>& groups, std::mt19937& rng)
{
    Rows Y = X;
    for (std::size_t i = 0; i < X.size(); ++i)
        if (i != keep) Y[i] = permuted(X[i], groups, rng);
    return Y;
}

const std::vector<S::Blocks> kGroups{{1, 2, 4}, {10, 3, 3}};   // 0 and 9 free, 19 coordinates
constexpr std::size_t kN = 19;

void built_in_canon()
{
    std::mt19937 rng(3);
    bool same = true, idem = true;
    for (int rep = 0; rep < 100; ++rep) {
        Vec x(kN);
        for (double& v : x) v = mootation::rnd::uniform01(rng);
        Vec a = x, b = permuted(x, kGroups, rng);
        S::sort_blocks(a, kGroups);
        S::sort_blocks(b, kGroups);
        same = same && a == b;
        Vec c = a;
        S::sort_blocks(c, kGroups);
        idem = idem && c == a;
    }
    check(same && idem, "the built-in canon: canon(P(x)) = canon(x), canon(canon(x)) = canon(x) (SYM-5)");
}

S::Sym sym_of(S::Mode mode)
{
    S::Sym s;
    s.mode = mode;
    s.groups = kGroups;
    return s;
}

// the DE family: de::trial and pbest::trial on the population and on its permutation
void de_steps()
{
    namespace sc = mootation::scalar;
    std::mt19937 rng(11);
    const Vec lo(kN, -2.0), hi(kN, 2.0);
    const std::size_t n = 12;
    bool de_align = true, de_canon = true, pb_align = true, pb_canon = true, de_off = true;
    for (int rep = 0; rep < 40; ++rep) {
        const Rows X = random_rows(n, kN, rng), A = random_rows(5, kN, rng);
        const std::size_t i = static_cast<std::size_t>(rep) % n;
        const Rows Xi = permuted_rows(X, i, kGroups, rng), Xall = permuted_rows(X, n, kGroups, rng);
        Rows Ap;
        for (const Vec& a : A) Ap.push_back(permuted(a, kGroups, rng));
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), std::size_t{0});
        const unsigned seed = 100u + static_cast<unsigned>(rep);
        auto de_trial = [&](const Rows& P, S::Mode mode) {
            sc::de::Options o;
            o.CR = 0.5;
            o.sym = sym_of(mode);
            std::mt19937 r(seed);
            return sc::de::trial(P, i, o, lo, hi, r);
        };
        auto pb_trial = [&](const Rows& P, const Rows& AA, S::Mode mode) {
            std::mt19937 r(seed);
            return sc::pbest::trial(P, AA, i, order, 4, 0.7, 0.5, lo, hi, r, sym_of(mode));
        };
        de_align = de_align && de_trial(Xi, S::Mode::align) == de_trial(X, S::Mode::align);
        de_canon = de_canon && de_trial(Xall, S::Mode::canon) == de_trial(X, S::Mode::canon);
        de_off = de_off && de_trial(Xi, S::Mode::off) == de_trial(X, S::Mode::off);
        pb_align = pb_align && pb_trial(Xi, Ap, S::Mode::align) == pb_trial(X, A, S::Mode::align);
        pb_canon = pb_canon && pb_trial(Xall, Ap, S::Mode::canon) == pb_trial(X, A, S::Mode::canon);
    }
    check(de_align && de_canon, "de::trial: align on P(x) but the target, canon on P(x): the same trial (SYM-8)");
    check(pb_align && pb_canon, "pbest::trial (jade, shade, lshade): the same, the archive permuted too (PB-3)");
    check(!de_off, "de::trial with sym = off on P(x): another trial");
}

// ga::cross with and without block_uniform; block_mutation
void ga_steps()
{
    namespace ga = mootation::scalar::ga;
    std::mt19937 rng(13);
    const Vec lo(kN, -2.0), hi(kN, 2.0);
    const ga::Bounds b = ga::detail::bounds(lo, hi);
    bool align = true, canon = true, off_same = true, whole = true;
    for (int rep = 0; rep < 60; ++rep) {
        const Rows X = random_rows(2, kN, rng);
        const Vec q2 = permuted(X[1], kGroups, rng), q1 = permuted(X[0], kGroups, rng);
        const unsigned seed = 200u + static_cast<unsigned>(rep);
        auto cross = [&](const Vec& p1, const Vec& p2, S::Mode mode, double bu, Vec& c1, Vec& c2) {
            ga::Options o;
            o.pc = 1.0;
            o.sym = sym_of(mode);
            o.block_uniform = bu;
            std::mt19937 r(seed);
            ga::cross(p1, p2, c1, c2, b, o, r);
        };
        for (double bu : {0.0, 1.0}) {
            Vec a1, a2, b1, b2;
            cross(X[0], X[1], S::Mode::align, bu, a1, a2);
            cross(X[0], q2, S::Mode::align, bu, b1, b2);
            align = align && a1 == b1 && a2 == b2;
            cross(X[0], X[1], S::Mode::canon, bu, a1, a2);
            cross(q1, q2, S::Mode::canon, bu, b1, b2);
            canon = canon && a1 == b1 && a2 == b2;
            cross(X[0], X[1], S::Mode::off, bu, a1, a2);
            cross(X[0], q2, S::Mode::off, bu, b1, b2);
            off_same = off_same && a1 == b1 && a2 == b2;
        }
        // GA-7 with p_c = 0 (the crossover gives the parents back): every block whole from one
        // parent, the other child's from the other, the free coordinates the parents'
        ga::Options o;
        o.pc = 0.0;
        o.sym = sym_of(S::Mode::off);
        o.block_uniform = 1.0;
        std::mt19937 r(seed);
        Vec c1, c2;
        ga::cross(X[0], X[1], c1, c2, b, o, r);
        whole = whole && c1[0] == X[0][0] && c1[9] == X[0][9] && c2[0] == X[1][0] && c2[9] == X[1][9];
        for (const S::Blocks& g : kGroups)
            for (std::size_t k = 0; k < g.count; ++k) {
                const std::size_t j0 = g.offset + k * g.size;
                const bool from1 = std::equal(c1.begin() + static_cast<std::ptrdiff_t>(j0),
                                              c1.begin() + static_cast<std::ptrdiff_t>(j0 + g.size),
                                              X[0].begin() + static_cast<std::ptrdiff_t>(j0));
                const bool from2 = std::equal(c1.begin() + static_cast<std::ptrdiff_t>(j0),
                                              c1.begin() + static_cast<std::ptrdiff_t>(j0 + g.size),
                                              X[1].begin() + static_cast<std::ptrdiff_t>(j0));
                const Vec& other = from1 ? X[1] : X[0];
                const bool rest = std::equal(c2.begin() + static_cast<std::ptrdiff_t>(j0),
                                             c2.begin() + static_cast<std::ptrdiff_t>(j0 + g.size),
                                             other.begin() + static_cast<std::ptrdiff_t>(j0));
                whole = whole && (from1 || from2) && rest;
            }
    }
    check(align && canon, "ga::cross: align on P(p2), canon on P(p1), P(p2): the same children, block_uniform "
                          "0 and 1 (GA-6, GA-7)");
    check(!off_same, "ga::cross with sym = off on P(p2): other children");
    check(whole, "block_uniform: each block whole from one parent, the other child's from the other (GA-7)");
    // GA-8: one block changes, within the box
    bool one = true, inside = true;
    int changed_total = 0;
    for (int rep = 0; rep < 200; ++rep) {
        ga::Options o;
        o.sym = sym_of(S::Mode::off);
        o.block_mutation = 1.0;
        Vec c = random_rows(1, kN, rng)[0];
        const Vec before = c;
        ga::block_mutation(c, b, o, rng);
        int blocks_changed = 0;
        for (const S::Blocks& g : kGroups)
            for (std::size_t k = 0; k < g.count; ++k) {
                bool ch = false;
                for (std::size_t j = g.offset + k * g.size; j < g.offset + (k + 1) * g.size; ++j) ch = ch || c[j] != before[j];
                blocks_changed += ch ? 1 : 0;
            }
        one = one && blocks_changed <= 1 && c[0] == before[0] && c[9] == before[9];
        changed_total += blocks_changed;
        for (double v : c) inside = inside && v >= -2.0 && v <= 2.0;
    }
    check(one && inside && changed_total > 150, "block_mutation: one block mutated, in the box (GA-8; " +
                                                    std::to_string(changed_total) + " of 200 changed)");
}

// pso::move, spso2011::move: the bests permuted, the particle not
void pso_steps()
{
    namespace sc = mootation::scalar;
    std::mt19937 rng(17);
    const Vec lo(kN, -2.0), hi(kN, 2.0);
    bool pso = true, spso = true, off = true;
    for (int rep = 0; rep < 40; ++rep) {
        const Rows X = random_rows(4, kN, rng);           // x, v (scaled below), p, g
        Vec v0 = X[1];
        for (double& t : v0) t *= 0.1;
        const Vec pp = permuted(X[2], kGroups, rng), gp = permuted(X[3], kGroups, rng);
        const unsigned seed = 300u + static_cast<unsigned>(rep);
        auto pmove = [&](const Vec& p, const Vec& g, S::Mode mode) {
            sc::pso::Options o;
            o.sym = sym_of(mode);
            Vec x = X[0], v = v0;
            std::mt19937 r(seed);
            sc::pso::move(x, v, p, g, o, sc::pso::constriction(o.phi_max, o.kappa), lo, hi, r);
            x.insert(x.end(), v.begin(), v.end());
            return x;
        };
        auto smove = [&](const Vec& p, const Vec& g) {
            sc::spso2011::Options o;
            o.sym = sym_of(S::Mode::align);
            Vec x = X[0], v = v0;
            std::mt19937 r(seed);
            sc::spso2011::move(x, v, p, g, o, lo, hi, r);
            x.insert(x.end(), v.begin(), v.end());
            return x;
        };
        pso = pso && pmove(pp, gp, S::Mode::align) == pmove(X[2], X[3], S::Mode::align);
        off = off && pmove(pp, gp, S::Mode::off) == pmove(X[2], X[3], S::Mode::off);
        spso = spso && smove(pp, gp) == smove(X[2], X[3]);
    }
    check(pso && spso, "pso::move, spso2011::move: align with the bests permuted, the same move (SYM-9)");
    check(!off, "pso::move with sym = off and the bests permuted: another move");
}

// through scalar::run: the knobs, the settings and the refusals
void through_run()
{
    namespace sc = mootation::scalar;
    sc::Problem P;
    P.lo.assign(kN, -2.0);
    P.hi.assign(kN, 2.0);
    P.m = 1;
    P.evaluate = [](const Vec& x, Vec& f, Vec&) {           // invariant under kGroups
        double s = (x[0] - 0.5) * (x[0] - 0.5) + (x[9] + 0.5) * (x[9] + 0.5);
        for (const S::Blocks& g : kGroups)
            for (std::size_t k = 0; k < g.count; ++k) {
                double r = 0.0;
                for (std::size_t c = 0; c < g.size; ++c) r += x[g.offset + k * g.size + c] * x[g.offset + k * g.size + c];
                s += (r - 1.0) * (r - 1.0);
            }
        f[0] = s;
    };
    P.sym_blocks = kGroups;
    auto run = [&](const std::string& name, const sc::Knobs& k, const sc::Problem& Q) {
        return sc::run(name, Q, 1e-6, false, 0.0, 20, 600, 5, k, {}, 0, {});
    };
    auto has = [](const sc::Result& r, const std::string& k) {
        for (const auto& kv : r.settings)
            if (kv.first == k) return true;
        return false;
    };
    bool same = true, listed = true, ran = true;
    for (const std::string name : {"de", "jade", "shade", "lshade", "ga_sbx", "ssga", "pso", "spso2011"}) {
        const sc::Result a = run(name, {}, P);
        const sc::Result b = run(name, {{"sym", {true, "off", 0.0}}}, P);
        same = same && a.variables == b.variables && a.objectives == b.objectives && a.evaluations == b.evaluations;
        listed = listed && !has(a, "sym") && has(b, "sym");
        for (const std::string mode : {"align", "canon"}) {
            if (mode == "canon" && (name == "pso" || name == "spso2011")) continue;
            const sc::Result c = run(name, {{"sym", {true, mode, 0.0}}}, P);
            ran = ran && c.evaluations == a.evaluations && std::isfinite(c.objectives[0][0]);
        }
    }
    const sc::Result g = run("ga_sbx", {{"sym", {true, "align", 0.0}}, {"ga_block_uniform", {false, "", 0.5}},
                                        {"ga_block_mutation", {false, "", 0.2}}}, P);
    ran = ran && std::isfinite(g.objectives[0][0]) && has(g, "ga_block_uniform");
    check(same, "sym = off given: every population method as without the knob, bit for bit");
    check(listed, "sym goes to the settings only where given (POP-9)");
    check(ran, "sym = align / canon and the block operators run on every method that takes them");
    auto refused = [&](const std::string& name, const sc::Knobs& k, const sc::Problem& Q) {
        try {
            run(name, k, Q);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    sc::Problem bare = P;
    bare.sym_blocks.clear();
    sc::Problem two_boxes = P;
    two_boxes.hi[3] = 1.5;                                   // block 1 of the first group
    check(refused("de", {{"sym", {true, "align", 0.0}}}, bare) &&
              refused("pso", {{"sym", {true, "canon", 0.0}}}, P) &&
              refused("ga_sbx", {{"ga_block_mutation", {false, "", 0.5}}}, bare) &&
              refused("de", {{"sym", {true, "align", 0.0}}}, two_boxes) &&
              refused("ga_sbx", {{"ga_block_uniform", {false, "", 1.5}}}, P),
          "refused: sym without symmetries, pso with canon, block operators without blocks, a group with two "
          "boxes (SYM-6), a probability past 1");
    // a run on a problem with two boxes in a group but no sym knob: as before
    check(!refused("de", {}, two_boxes), "no sym knob: the groups are not checked, the run as before");
}

}   // namespace

int main()
{
    hungarian_against_every_permutation();
    align_of_a_permutation();
    built_in_canon();
    de_steps();
    ga_steps();
    pso_steps();
    through_run();
    if (g_failed) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
