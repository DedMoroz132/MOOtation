// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// suite/: the C++ suite of benchmark problems, on what holds whatever the C
// library (python/test_run_layer.py holds it to NumPy's values, bit for bit,
// on every registered problem): the fronts the families are built to have
// at their optimum, NumPy's clip, mod and pairwise sums, the maps of the shifts
// and rotations, and the specs make() refuses.
// ============================================================================

#include <cmath>
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

    // what make() refuses
    check(refused(spec("WFG1", 6, 2)), "a family the suite does not have");
    check(refused(spec("MaF8", 2, 3)), "MaF8 without its vertices");
    check(refused(spec("MaF11", 12, 3)), "MaF11 (WFG2) stays in Python");
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
