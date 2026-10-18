// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// suite/: the C++ suite of benchmark problems, on what holds whatever the C
// library (python/test_run_layer.py holds it to NumPy's values, bit for bit,
// on every registered problem): the fronts the families are built to have
// at their optimum, NumPy's clip, mod and pairwise sums, the maps of the shifts
// and rotations, and the specs make() refuses.
// ============================================================================

#include <cmath>
#include <cstdint>
#include <cstdio>
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
    check(refused(spec("WFG1", 6, 2)), "a family the suite does not have");
    check(refused(spec("MaF8", 2, 3)), "MaF8 stays in Python");
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

    std::printf("suite: %d/%d checks passed%s\n", checks - failed, checks, failed ? ", FAILED" : "");
    return failed == 0 ? 0 : 1;
}
