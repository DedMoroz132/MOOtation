// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// suite/: the C++ suite of benchmark problems, on what holds whatever the C
// library (python/test_run_layer.py holds it to NumPy's values, bit for bit,
// on every registered problem): the fronts the families are built to have
// at their optimum, NumPy's clip, mod and pairwise sums, the maps of the shifts
// and rotations, and the specs make() refuses.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/suite/suite.hpp>

namespace S = mootation::suite;

namespace {

int checks = 0, failed = 0;

void check(bool ok, const std::string& what)
{
    ++checks;
    if (!ok) {
        ++failed;
        std::printf("  FAIL  %s\n", what.c_str());
    }
}

S::Spec spec(const std::string& family, int n, int M)
{
    S::Spec s;
    s.family = family;
    s.n_var = n;
    s.n_obj = M;
    return s;
}

std::vector<double> eval(const S::Native& p, const std::vector<double>& x, std::vector<double>* g = nullptr)
{
    std::vector<double> f(static_cast<std::size_t>(p.n_obj)), c(static_cast<std::size_t>(p.n_con));
    p.evaluate(x.data(), f.data(), c.empty() ? nullptr : c.data());
    if (g) *g = c;
    return f;
}

bool refused(const S::Spec& s)
{
    try {
        S::make(s);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}   // namespace

int main()
{
    // NumPy's scalar loops
    check(std::signbit(S::clip(-0.0, 0.0, 1.0)) == false, "clip(-0.0, 0, 1) is the bound +0.0");
    check(std::isnan(S::clip(std::nan(""), 0.0, 1.0)), "clip keeps NaN");
    check(S::clip(2.0, 0.0, 1.0) == 1.0 && S::clip(-1.0, 0.0, 1.0) == 0.0, "clip to the bounds");
    check(S::np_mod(-0.25, 1.0) == 0.75 && S::np_mod(1.0, 1.0) == 0.0 &&
              !std::signbit(S::np_mod(-1.0, 1.0)), "np.mod: the sign of the divisor");
    check(std::isnan(S::nan_min(1.0, std::nan(""))) && std::isnan(S::nan_min(std::nan(""), 1.0)),
          "np.minimum: a NaN wins");
    {
        std::vector<double> a(9);
        for (int i = 0; i < 9; ++i) a[static_cast<std::size_t>(i)] = 0.1 * (i + 1);
        const double r8 = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
        check(S::row_sum(a) == 0.0 + (r8 + a[8]), "a row of nine: eight running sums, then the ninth");
        check(S::row_prod(a.data(), 0) == 1.0 && S::row_prod(a.data(), 1) == a[0], "products from 1");
    }

    // the fronts at the optimum: DTLZ1 sums to 1/2, DTLZ2-4 lie on the sphere
    {
        for (int M : {2, 3, 5, 10}) {
            const int n = M + 9;
            std::vector<double> x(static_cast<std::size_t>(n), 0.5);
            for (int i = 0; i < M - 1; ++i) x[static_cast<std::size_t>(i)] = 0.1 + 0.8 * i / M;
            auto d1 = S::make(spec("DTLZ1", n, M));
            double sum = 0.0;
            for (double v : eval(*d1, x)) sum += v;
            check(std::abs(sum - 0.5) < 1e-14, "DTLZ1 at g = 0 sums to 1/2, M = " + std::to_string(M));
            for (const char* fam : {"DTLZ2", "DTLZ3", "DTLZ4"}) {
                auto p = S::make(spec(fam, n, M));
                double sq = 0.0;
                for (double v : eval(*p, x)) sq += v * v;
                check(std::abs(sq - 1.0) < 1e-14, std::string(fam) + " at g = 0 on the sphere");
            }
            auto i1 = S::make(spec("IDTLZ1", n, M));
            const auto fi = eval(*i1, x), fd = eval(*d1, x);
            bool inverted = true;
            for (int i = 0; i < M; ++i) inverted = inverted && fi[static_cast<std::size_t>(i)] == 0.5 - fd[static_cast<std::size_t>(i)];
            check(inverted, "IDTLZ1 = 1/2 - DTLZ1 at g = 0");
        }
    }

    // ZDT at g = 1
    {
        std::vector<double> x(30, 0.0);
        x[0] = 0.36;
        auto z1 = S::make(spec("ZDT1", 30, 2));
        const auto f = eval(*z1, x);
        check(f[0] == 0.36 && std::abs(f[1] - 0.4) < 1e-15, "ZDT1 at g = 1: 1 - sqrt(f1)");
        auto z2 = S::make(spec("ZDT2", 30, 2));
        check(std::abs(eval(*z2, x)[1] - (1.0 - 0.36 * 0.36)) < 1e-15, "ZDT2 at g = 1: 1 - f1^2");
        std::vector<double> x4(10, 0.0);
        x4[0] = 0.25;
        auto z4 = S::make(spec("ZDT4", 10, 2));
        check(std::abs(eval(*z4, x4)[1] - 0.5) < 1e-15, "ZDT4 at g = 1");
    }

    // a fold and a shift move the optimum to the centres; a rotation turns about them
    {
        const int n = 12, M = 3, k = n - M + 1;
        S::Spec s = spec("DTLZ2", n, M);
        s.map = "rotate_half";
        s.first = M - 1;
        s.centre.assign(static_cast<std::size_t>(k), 0.3);
        s.matrices.push_back(std::vector<double>(static_cast<std::size_t>(k * k), 0.0));
        for (int i = 0; i < k; ++i)                       // a permutation: a rotation by its rows
            s.matrices[0][static_cast<std::size_t>(i * k + (i + 1) % k)] = 1.0;
        auto p = S::make(s);
        std::vector<double> x(static_cast<std::size_t>(n), 0.3);
        x[0] = 0.2;
        x[1] = 0.7;
        double sq = 0.0;
        for (double v : eval(*p, x)) sq += v * v;
        check(std::abs(sq - 1.0) < 1e-14, "rotate_half: at the centre, DTLZ2's front");
        S::Spec c = spec("DTLZ1", n, M);
        c.map = "cyclic";
        c.first = M - 1;
        c.centre.assign(static_cast<std::size_t>(k), 0.65);
        auto q = S::make(c);
        std::vector<double> y(static_cast<std::size_t>(n), 0.65);
        y[0] = 0.4;
        y[1] = 0.1;
        double sum = 0.0;
        for (double v : eval(*q, y)) sum += v;
        check(std::abs(sum - 0.5) < 1e-14, "cyclic: at the centres, DTLZ1's front");
    }

    // constraints: C1-DTLZ1 holds on DTLZ1's front, C3-DTLZ1 does not
    {
        const int M = 3, n = M + 4;
        std::vector<double> x(static_cast<std::size_t>(n), 0.5);
        x[0] = 0.3;
        x[1] = 0.6;
        std::vector<double> g;
        auto c1 = S::make(spec("C1-DTLZ1", n, M));
        eval(*c1, x, &g);
        check(c1->n_con == 1 && g[0] <= 0.0, "C1-DTLZ1: its front feasible");
        auto c3 = S::make(spec("C3-DTLZ1", n, M));
        eval(*c3, x, &g);
        check(c3->n_con == M && g[0] > 0.0, "C3-DTLZ1: DTLZ1's front infeasible");
    }

    // ZCAT at its Pareto set: beta = 0, f_i = i^2 F_i
    {
        const int M = 3, n = 30;
        S::Spec s = spec("ZCAT4", n, M);
        auto p = S::make(s);
        std::vector<double> Y(static_cast<std::size_t>(n), 0.5);
        Y[0] = 0.3;
        Y[1] = 0.8;
        std::vector<double> g(static_cast<std::size_t>(n - (M - 1)));
        S::zcat::topology(S::zcat::kTopology[4], Y.data(), M - 1, n, g.data());
        std::vector<double> x(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            const double y = i < M - 1 ? Y[static_cast<std::size_t>(i)] : g[static_cast<std::size_t>(i - (M - 1))];
            x[static_cast<std::size_t>(i)] = (y - 0.5) * (i + 1);
        }
        const auto f = eval(*p, x);
        check(std::abs(f[0] - 0.3) < 1e-12 && std::abs(f[1] - 4 * 0.8) < 1e-12 &&
                  std::abs(f[2] - 9 * (1.0 - 1.1 / 2)) < 1e-12,
              "ZCAT4 on its Pareto set: alpha alone");
    }

    // MaF9: the distances to the M-gon's lines, a point inside it feasible
    {
        const int M = 5;
        S::Spec s = spec("MaF9", 2, M);
        for (int m = 0; m < M; ++m) {
            const double ax = std::cos(2 * S::kPi * m / M), ay = std::sin(2 * S::kPi * m / M);
            const double bx = std::cos(2 * S::kPi * (m + 1) / M), by = std::sin(2 * S::kPi * (m + 1) / M);
            s.lines.insert(s.lines.end(), {ax, ay, bx - ax, by - ay, std::hypot(bx - ax, by - ay)});
        }
        s.polygons.push_back({3.0, 3.0, 4.0, 3.0, 4.0, 4.0, 3.0, 4.0});
        auto p = S::make(s);
        std::vector<double> g;
        const auto f = eval(*p, {0.0, 0.0}, &g);
        const double apothem = std::cos(S::kPi / M);
        bool equal = true;
        for (double v : f) equal = equal && std::abs(v - apothem) < 1e-14;
        check(equal && p->n_con == 1 && g[0] == 0.0, "MaF9: the centre at the apothem, feasible");
        eval(*p, {3.25, 3.5}, &g);
        check(std::abs(g[0] - 0.25) < 1e-15, "MaF9: inside a polygon, the distance to its nearest edge");
    }

    // MaF8: the distances to the M-gon's vertices
    {
        const int M = 4;
        S::Spec s = spec("MaF8", 2, M);
        for (int m = 0; m < M; ++m) s.vertices.insert(s.vertices.end(), {std::cos(2 * S::kPi * m / M), std::sin(2 * S::kPi * m / M)});
        auto p = S::make(s);
        bool unit = true;
        for (double v : eval(*p, {0.0, 0.0})) unit = unit && v == 1.0;
        const auto f = eval(*p, {s.vertices[0], s.vertices[1]});
        check(unit && f[0] == 0.0 && std::abs(f[2] - 2.0) < 1e-15, "MaF8: 1 from the centre, 0 and 2 at a vertex");
    }

    // MaF14 and MaF15 on their Pareto sets: the simplex, and 1 minus the sphere
    {
        const int M = 3, n = 60, ns = n - M + 1;
        const std::vector<double> parts = {6, 6, 15, 15, 8, 8};
        for (int which : {14, 15}) {
            S::Spec s = spec(which == 14 ? "MaF14" : "MaF15", n, M);
            s.parts = parts;
            for (int i = 1; i <= ns; ++i) {
                const double r = static_cast<double>(i) / ns;
                s.link.insert(s.link.end(), {which == 14 ? 1.0 + r : 1.0 + std::cos(0.5 * S::kPi * r), 0.0,
                                             which == 14 ? 10.0 : 1.0});
            }
            auto p = S::make(s);
            std::vector<double> x(static_cast<std::size_t>(n));
            x[0] = 0.3;
            x[1] = 0.6;
            for (int i = 0; i < ns; ++i) {
                // MaF14's group 2 (columns 12 .. 41) is Rosenbrock's, its optimum z = 1
                const double z = which == 14 && i >= 12 && i < 42 ? 1.0 : 0.0;
                const double* l = s.link.data() + 3 * i;
                x[static_cast<std::size_t>(M - 1 + i)] = l[1] + (z + x[0] * l[2]) / l[0];
            }
            const auto f = eval(*p, x);
            double sum = 0.0, sq = 0.0;
            for (double v : f) {
                sum += v;
                sq += (1.0 - v) * (1.0 - v);
            }
            check(which == 14 ? std::abs(sum - 1.0) < 1e-14 : std::abs(sq - 1.0) < 1e-14,
                  which == 14 ? "MaF14 on its Pareto set: the simplex" : "MaF15 on its Pareto set: 1 - the sphere");
            x[30] += 0.5;
            const auto g = eval(*p, x);
            check(g[1] > f[1] && (which == 14 ? g[0] == f[0] : g[0] > f[0]) && g[2] == f[2],
                  "MaF" + std::to_string(which) + ": group 2 read by objective 2 (MaF15: and 1)");
            S::Spec bad = s;
            bad.parts.back() = 7.0;
            check(refused(bad), "MaF" + std::to_string(which) + " whose subcomponents do not take x^s");
            bad = s;
            bad.link.pop_back();
            check(refused(bad), "MaF" + std::to_string(which) + " without a link per variable");
        }
    }

    // bbob: the products fuse where OpenBLAS's kernels do. a b = 1 - 2^-60
    // rounds to 1; fused, -1 + a b keeps the -2^-60
    {
        namespace B = S::bbob;
        const double a = 1.0 + std::ldexp(1.0, -30), b = 1.0 - std::ldexp(1.0, -30), tiny = -std::ldexp(1.0, -60);
        const double u[2] = {1.0, a}, v[2] = {-1.0, b};
        check(B::dot(u, v, 2) == tiny, "bbob: x . y in fused multiply-adds");
        // fused (the instruction, where MSVC has it) = std::fma, products near their sums
        std::uint64_t state = 20261019;
        auto next = [&]() {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            return std::ldexp(static_cast<double>(state >> 11), -53) - 0.5;
        };
        bool same = true;
        for (int i = 0; i < 100000; ++i) {
            const double x = std::ldexp(next(), static_cast<int>(40 * next())),
                         y = std::ldexp(next(), static_cast<int>(40 * next()));
            const double z = i % 2 ? -x * y * (1.0 + std::ldexp(next(), -40)) : next();
            same = same && B::fused(x, y, z) == std::fma(x, y, z);
        }
        check(same, "bbob: fused = std::fma on 100 000 triples");
        std::vector<double> d(2), R2 = {1.0, 0.0, 0.0, 1.0};
        B::gemm_row(u, R2, 2, d.data());
        check(d[0] == 1.0 && d[1] == a, "bbob: (x - y) R^T, the identity");
        // D = 5: rows 0-3 a block of four (fused lanes), row 4 the single (products)
        std::vector<double> M(25, 0.0);
        const double x[5] = {-1.0, 0.0, 0.0, 0.0, b};
        M[0] = 1.0;
        M[4] = a;                                       // row 0: -1 + a b, the fifth term fused
        M[20] = 1.0;
        M[24] = a;                                      // row 4: the same
        check(B::matvec_row(M.data(), x, 5, 0) == tiny && B::matvec_row(M.data() + 20, x, 5, 4) == tiny,
              "bbob: M x, a lone fifth term fused onto the sum");
        const double x8[8] = {-1.0, 0.0, 0.0, 0.0, b, 0.0, 0.0, 0.0};
        std::vector<double> M8(64, 0.0);
        M8[0] = 1.0;
        M8[4] = a;
        check(B::matvec_row(M8.data(), x8, 8, 0) == tiny, "bbob: M x, a block row's lane fused");
        M8[56] = 1.0;
        M8[60] = a;
        check(B::matvec_row(M8.data() + 56, x8, 8, 7) == tiny, "bbob: M x, the four lanes of the block");
        const double x6[6] = {-1.0, 0.0, b, 0.0, 0.0, 0.0};
        std::vector<double> M6(36, 0.0);
        M6[24] = 1.0;
        M6[26] = a;                                     // row 4 of D = 6: the pair, products then sums
        check(B::matvec_row(M6.data() + 24, x6, 6, 4) == 0.0, "bbob: M x, the pair's lanes not fused");
    }

    // bbob: every one of the ten at its optimum is f_opt, on instances made up
    // here (the identity and a permutation for R and Q); F = (f_a, f_b)
    {
        namespace B = S::bbob;
        const int D = 5;
        auto function = [&](int fid) {
            B::Function f;
            f.fid = fid;
            f.f_opt = fid - 40.25;
            for (int i = 0; i < D; ++i) f.x_opt.push_back(0.5 * i - 1.25);
            f.R.assign(D * D, 0.0);
            f.Q.assign(D * D, 0.0);
            for (int i = 0; i < D; ++i) {
                f.R[static_cast<std::size_t>(i * D + i)] = 1.0;
                f.Q[static_cast<std::size_t>(i * D + (i + 2) % D)] = 1.0;
                f.l10.push_back(std::pow(10.0, 0.5 * i / (D - 1.0)));
                f.sign.push_back(i % 2 ? -1.0 : 1.0);
            }
            if (fid == 20)
                for (int i = 0; i < D; ++i) f.x_opt[static_cast<std::size_t>(i)] = 0.5 * 4.2096874633 * f.sign[static_cast<std::size_t>(i)];
            for (int k = 0; k < B::kPeaks; ++k) {
                f.w.push_back(k == 0 ? 10.0 : 1.1 + 8.0 * (k - 1) / 99.0);
                for (int i = 0; i < D; ++i) {
                    f.C.push_back(1.0);
                    f.y.push_back(k == 0 ? f.x_opt[static_cast<std::size_t>(i)] : 4.5 - 0.09 * k);
                }
            }
            return f;
        };
        bool all = true;
        for (int fid : {1, 2, 6, 8, 13, 14, 15, 17, 20, 21}) {
            const B::Objective o(function(fid), D);
            const double v = o(o.function().x_opt.data());
            const bool ok = fid == 20 ? std::abs(v - o.function().f_opt) < 1e-9 : v == o.function().f_opt;
            if (!ok) std::printf("  f%d at x_opt: %.17g, f_opt %.17g\n", fid, v, o.function().f_opt);
            all = all && ok;
        }
        check(all, "bbob: f(x_opt) = f_opt for the ten");

        S::Spec s = spec("bbob-biobj", D, 2);
        s.bbob = {function(1), function(2)};
        s.bbob[1].x_opt[0] += 1.0;
        auto p = S::make(s);
        const auto f = eval(*p, s.bbob[0].x_opt);
        check(f[0] == s.bbob[0].f_opt && f[1] > s.bbob[1].f_opt, "bbob-biobj: (f_alpha, f_beta)");

        // mixint: the integer j stands for the level z_{j+1}, j rounded half to even into 0 .. l-1
        S::Spec m = spec("bbob-biobj-mixint", D, 2);
        m.bbob = {function(1), function(1)};
        for (int o = 0; o < 2; ++o)
            for (int i = 0; i < 4; ++i)
                m.bbob[static_cast<std::size_t>(o)].levels.push_back({o - 1.0, o + 0.0, o + 1.0});
        auto q = S::make(m);
        const auto at = [&](std::vector<double> x) { return eval(*q, x); };
        const double last = s.bbob[0].x_opt[4];
        check(at({0.5, 1.5, 2.5, -7.0, last}) == at({0.0, 2.0, 2.0, 0.0, last}),
              "bbob-biobj-mixint: the integers rounded half to even and kept in their range");
        S::Spec plain = spec("bbob-biobj", D, 2);
        plain.bbob = {function(1), function(1)};
        auto r = S::make(plain);
        const std::vector<double> lv = {-1.0, 0.0, 1.0, 1.0, last};
        check(at({0.0, 1.0, 2.0, 2.0, last})[0] == eval(*r, lv)[0], "bbob-biobj-mixint: each value its level");
    }

    // bbob, one objective (task 18): all 24 at their optimum, on the instances
    // drawn here, in every size of the suite; the optimum in the box
    {
        bool all = true, inside = true;
        for (int fid = 1; fid <= 24; ++fid)
            for (int D : {2, 3, 5, 10, 20, 40})
                for (int iid = 1; iid <= 3; ++iid) {
                    S::Spec s = spec("bbob", D, 1);
                    s.bbob_fid = fid;
                    s.bbob_iid = iid;
                    auto p = S::make(s);
                    const S::bbob::Function f = S::bbob::instance(fid, iid, D);
                    const double v = eval(*p, f.x_opt)[0];
                    const bool ok = std::abs(v - f.f_opt) <= 1e-9 * std::max(1.0, std::abs(f.f_opt));
                    if (!ok) std::printf("  bbob f%d D=%d i%d at x_opt: %.17g, f_opt %.17g\n", fid, D, iid, v, f.f_opt);
                    all = all && ok;
                    for (double x : f.x_opt) inside = inside && std::abs(x) <= 5.0;
                }
        check(all, "bbob: f(x_opt) = f_opt for the 24 in 2..40 variables");
        check(inside, "bbob: x_opt in [-5, 5]^D");
        S::Spec s = spec("bbob", 41, 1);
        s.bbob_fid = 1;
        check(refused(s), "bbob beyond 40 variables");
        s = spec("bbob", 5, 2);
        s.bbob_fid = 1;
        check(refused(s), "bbob with two objectives");
        s = spec("bbob", 5, 1);
        s.bbob_fid = 25;
        check(refused(s), "bbob f25");
    }

    // what make() refuses
    {
        namespace B = S::bbob;
        B::Function f;
        f.fid = 1;
        f.x_opt.assign(5, 0.0);
        S::Spec s = spec("bbob-biobj", 5, 2);
        s.bbob = {f};
        check(refused(s), "bbob-biobj with one function");
        s.bbob = {f, f};
        check(!refused(s), "bbob-biobj of two spheres");
        s.n_var = 16;
        s.bbob[0].x_opt.assign(16, 0.0);
        s.bbob[1].x_opt.assign(16, 0.0);
        check(refused(s), "bbob-biobj beyond the sizes its products were read off for");
        S::Spec t = spec("bbob-biobj", 5, 2);
        f.fid = 6;
        t.bbob = {f, f};
        check(refused(t), "bbob f6 without its rotations");
        f.fid = 3;
        t.bbob = {f, f};
        check(refused(t), "bbob f3: not one of the ten");
        f.fid = 1;
        t.bbob = {f, f};
        t.family = "bbob-biobj-mixint";
        check(refused(t), "bbob-biobj-mixint without integer variables");
        t.bbob[0].levels = {{0.0, 1.0}};
        check(refused(t), "bbob-biobj-mixint with levels in one objective only");
    }
    check(refused(spec("WFG10", 6, 2)), "a family the suite does not have");
    check(refused(spec("WFG1", 4, 3)), "WFG with no distance variable");
    check(!refused(spec("MaF8", 2, 3)) && refused(spec("MaF8", 3, 3)),
          "MaF8: its vertices drawn here (task 17, supplement 5, §3), two variables");
    check(refused(spec("MaF11", 12, 3)), "MaF11 is family WFG2 with k = M - 1");
    check(refused(spec("DTLZ2", 2, 3)), "fewer variables than objectives");
    check(refused(spec("ZDT1", 30, 3)), "ZDT at three objectives");
    {
        S::Spec s = spec("SDTLZ1", 7, 3);
        s.scale = {1.0, 10.0};
        check(refused(s), "SDTLZ without a scale per objective");
        S::Spec r = spec("DTLZ2", 12, 3);
        r.map = "rotate_half";
        r.first = 2;
        r.centre.assign(10, 0.5);
        r.matrices.push_back(std::vector<double>(81, 0.0));
        check(refused(r), "a rotation of the wrong size");
        r.map = "turn";
        check(refused(r), "an unknown map");
        S::Spec z = spec("ZCAT19", 30, 3);
        z.map = "rotate_zcat";
        z.matrices.push_back(std::vector<double>(28 * 28, 0.0));
        check(refused(z), "rZCAT19 without its second matrix");
    }

    // task 17, supplement 3, stage 3a: the constrained, multimodal and
    // real-world families where their papers state the values, their Pareto
    // sets on their fronts, and the specs make() refuses
    {
        const double pi = 3.141592653589793;
        S::Spec d = spec("DAS-CMOP1", 30, 2);
        d.constants = {-1.0, 0.0, std::numeric_limits<double>::infinity(), 0.0};   // the triplet (0, 0, 0)
        auto das = S::make(d);
        std::vector<double> x(30, mootation::dm::sin(0.5 * pi * 0.3)), g;
        x[0] = 0.3;
        const auto f = eval(*das, x, &g);
        check(das->n_con == 11 && f[0] == 0.3 && f[1] == 1.0 - 0.3 * 0.3 && g[1] == 0.0,
              "DAS-CMOP1: g = 0 on x_j = sin(pi x1 / 2), c2 = g - d at zeta = 0");

        auto lir = S::make(spec("LIR-CMOP13", 30, 3));
        std::vector<double> y(30, 0.5);
        y[0] = 0.2;
        y[1] = 0.7;
        const auto fl = eval(*lir, y, &g);
        check(std::abs(std::sqrt(fl[0] * fl[0] + fl[1] * fl[1] + fl[2] * fl[2]) - 1.7057) < 1e-12 && g.size() == 2,
              "LIR-CMOP13: the sphere of radius 1.7057 at g1 = 0");

        auto mw = S::make(spec("MW3", 15, 2));
        std::vector<double> z(15, 0.0);
        z[0] = 0.4;
        for (int i = 1; i < 15; ++i) z[static_cast<std::size_t>(i)] = 1.0 - (z[static_cast<std::size_t>(i - 1)] - 0.5) * (z[static_cast<std::size_t>(i - 1)] - 0.5);
        const auto fm = eval(*mw, z, &g);
        check(std::abs(fm[0] + fm[1] - 1.0) < 1e-12 && g.size() == 2, "MW3: f1 + f2 = 1 at g3's optimum");

        auto re = S::make(spec("RE21", 4, 2));
        const auto fr = eval(*re, {2.0, 2.0, 2.0, 2.0});
        const double s2 = std::sqrt(2.0);
        check(std::abs(fr[0] - 200.0 * (4.0 + 2.0 * s2 + s2 + 2.0)) < 1e-9 && std::abs(fr[1] - 0.01 * 2.0) < 1e-15,
              "RE21 at (2, 2, 2, 2)");
        check(S::make(spec("CRE24", 7, 2))->n_con == 11, "CRE24: the speed reducer's eleven constraints");

        // the Pareto sets
        const auto mmf = S::make(spec("MMF1", 2, 2));
        const std::vector<double> P = dynamic_cast<const S::Problem&>(*mmf).pareto_set(100);
        double worst = 0.0;
        for (std::size_t i = 0; i + 1 < P.size(); i += 2) {
            const auto fp = eval(*mmf, {P[i], P[i + 1]});
            worst = std::max(worst, std::abs(fp[1] - (1.0 - std::sqrt(fp[0]))));
        }
        check(P.size() == 200 && worst < 1e-12, "MMF1: its Pareto set on f2 = 1 - sqrt f1");
        S::Spec ip = spec("IPolygon", 2, 3);
        for (double cx : {25.0, 75.0})
            for (double cy : {25.0, 75.0})
                ip.polygons.push_back({cx, cy + 8.0, cx - 6.9282, cy - 4.0, cx + 6.9282, cy - 4.0});
        const std::vector<double> Q = dynamic_cast<const S::Problem&>(*S::make(ip)).pareto_set(40);
        bool inside = Q.size() == 80;
        for (std::size_t i = 0; i + 1 < Q.size(); i += 2)
            inside = inside && S::polygon::inside(Q[i], Q[i + 1], ip.polygons[i / 20]);
        check(inside, "IPolygon: ten points in each polygon, each inside its own");
        S::Spec rot = spec("Polygon", 10, 3);
        rot.vertices = {1.0, 0.0, -0.5, 0.8660254037844386, -0.5, -0.8660254037844387};
        check(dynamic_cast<const S::Problem&>(*S::make(rot)).has_pareto_set(), "Polygon has its Pareto set");
        rot.map = "rotate";
        rot.first = 2;
        rot.centre.assign(8, 0.0);
        rot.matrices.push_back(std::vector<double>(64, 0.0));
        check(!dynamic_cast<const S::Problem&>(*S::make(rot)).has_pareto_set(), "rcPolygon: not the suite's");

        S::Spec das0 = spec("DAS-CMOP1", 30, 2);
        check(refused(das0), "DAS-CMOP without its triplet's constants");
        check(refused(spec("MW1", 15, 3)), "MW1 at three objectives");
        check(refused(spec("DOC1", 7, 2)), "DOC1 at seven variables");
        check(refused(spec("RE21", 5, 2)), "RE21 at five variables");
        check(refused(spec("MMF1", 3, 2)), "MMF1 at three variables");
        check(!refused(spec("IPolygon", 2, 3)), "IPolygon builds its polygons itself (spec_numbers.hpp)");
        check(refused(spec("LIR-CMOP15", 30, 2)), "LIR-CMOP15: there is none");
    }

    // the Haar rotations (instances.hpp; task 17, supplement 5, §3): Q^T Q = I
    // to a few ulps, R = Q^T A upper triangular with a positive diagonal (the
    // signs fixed), and over 4 000 rotations of size 4 the moments of the
    // Haar distribution: E q_ij = 0, E q_ij^2 = 1/4, E tr Q = 0, E (tr Q)^2 = 1
    // (within five standard errors)
    {
        namespace I = S::instances;
        check(I::crc32("DTLZ2") == 0xde3d019bu && I::crc32("") == 0u && I::crc32("ZCAT19") == 0xd2c19e25u,
              "instances: crc32 is zlib's");
        double worst_orth = 0.0, worst_low = 0.0;
        bool positive = true;
        for (int k : {1, 2, 3, 5, 9, 20, 48})
            for (int inst = 1; inst <= 5; ++inst) {
                std::mt19937 g(I::seed({7u, static_cast<std::uint64_t>(k), static_cast<std::uint64_t>(inst)}));
                std::mt19937 again = g;
                const std::vector<double> Q = I::haar(g, k);
                std::vector<double> A(static_cast<std::size_t>(k * k));
                for (double& a : A) a = mootation::rnd::normal01(again);
                for (int i = 0; i < k; ++i)
                    for (int j = 0; j < k; ++j) {
                        double qq = 0.0, r = 0.0;
                        for (int l = 0; l < k; ++l) {
                            qq += Q[static_cast<std::size_t>(l * k + i)] * Q[static_cast<std::size_t>(l * k + j)];
                            r += Q[static_cast<std::size_t>(l * k + i)] * A[static_cast<std::size_t>(l * k + j)];
                        }
                        worst_orth = std::max(worst_orth, std::abs(qq - (i == j ? 1.0 : 0.0)));
                        if (i > j) worst_low = std::max(worst_low, std::abs(r));
                        if (i == j) positive = positive && r > 0.0;
                    }
            }
        std::printf("  haar: largest |Q^T Q - I| %.3g, largest |R below the diagonal| %.3g\n", worst_orth, worst_low);
        check(worst_orth < 64 * std::numeric_limits<double>::epsilon(), "haar: Q^T Q = I to a few ulps");
        check(worst_low < 1e-13 && positive, "haar: Q^T A upper triangular with a positive diagonal");
        const int N = 4000, k = 4;
        double m00 = 0.0, sq = 0.0, tr = 0.0, tr2 = 0.0;
        for (int t = 0; t < N; ++t) {
            std::mt19937 g(I::seed({11u, static_cast<std::uint64_t>(t)}));
            const std::vector<double> Q = I::haar(g, k);
            double trace = 0.0;
            for (int i = 0; i < k; ++i) trace += Q[static_cast<std::size_t>(i * k + i)];
            for (double q : Q) sq += q * q;
            m00 += Q[0];
            tr += trace;
            tr2 += trace * trace;
        }
        m00 /= N;
        sq /= N * k * k;
        tr /= N;
        tr2 /= N;
        // standard errors: sqrt(1/4 / N), sqrt(1/16 / N) (per element, the
        // squares' variance E q^4 - 1/16 = 3/24 - 1/16), sqrt(1/N), sqrt(2/N)
        check(std::abs(m00) < 5.0 * std::sqrt(0.25 / N) && std::abs(sq - 0.25) < 5.0 * std::sqrt(0.0625 / N) &&
                  std::abs(tr) < 5.0 * std::sqrt(1.0 / N) && std::abs(tr2 - 1.0) < 5.0 * std::sqrt(2.0 / N),
              "haar: E q_11 = 0, E q^2 = 1/4, E tr = 0, E tr^2 = 1 (4 000 rotations of size 4)");
    }

    std::printf("suite: %d/%d checks passed%s\n", checks - failed, checks, failed ? ", FAILED" : "");
    return failed == 0 ? 0 : 1;
}
