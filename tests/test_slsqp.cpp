// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/lsq.hpp, scalar/slsqp.hpp, scalar/nelder_mead.hpp on problems whose
// solutions are known (task 17, supplement 3, §3):
//   lsq     Lawson & Hanson's LSE example (20.25)-(20.29); NNLS and LDP
//           against their Kuhn-Tucker conditions on random problems;
//   slsqp   quadratics with equalities and inequalities, active bounds,
//           degenerate constraints (repeated and redundant rows), a start on
//           the boundary, infeasible starts, a start where the linearized
//           constraints are incompatible, Rosenbrock, Hock & Schittkowski 71,
//           the post office parcel and Powell's problems of Powell 1978, §5;
//           with each line search (stop, accept, powell);
//   nm      Rosenbrock, a quadratic in 5 variables, a minimum on the boundary.
// ============================================================================

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <mootation/scalar/lsq.hpp>
#include <mootation/scalar/nelder_mead.hpp>
#include <mootation/scalar/slsqp.hpp>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    if (!ok) {
        ++g_failed;
        std::printf("FAIL  %s\n", what.c_str());
    } else {
        std::printf("ok    %s\n", what.c_str());
    }
}

bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

using mootation::lsq::Matrix;
namespace slsqp = mootation::slsqp;
namespace nm = mootation::nelder_mead;

constexpr double kInf = std::numeric_limits<double>::infinity();

// ── lsq ─────────────────────────────────────────────────────────────────────
void lse_example()
{
    // LH (20.25)-(20.28): min ||E x - f|| s.t. C x = d; LH's x to 12 figures
    Matrix E(2, 2), C(1, 2), G(0, 2);
    E(0, 0) = 0.4302; E(0, 1) = 0.3516;
    E(1, 0) = 0.6246; E(1, 1) = 0.3384;
    C(0, 0) = 0.4087; C(0, 1) = 0.1593;
    auto r = mootation::lsq::lsei(E, {0.6593, 0.9666}, C, {0.1376}, G, {});
    check(r.status == mootation::lsq::Status::ok && near(r.x[0], -1.17749898217, 1e-10) &&
          near(r.x[1], 3.88476983058, 1e-10), "lsei: Lawson-Hanson (20.29)");
}

void nnls_ldp_random()
{
    std::mt19937 gen(7);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    int bad_nnls = 0, bad_ldp = 0, ldp_feasible = 0;
    for (int t = 0; t < 2000; ++t) {
        const std::size_t m = 1 + gen() % 12, n = 1 + gen() % 10;
        Matrix E(m, n);
        std::vector<double> f(m);
        for (double& v : E.a) v = u(gen);
        for (double& v : f) v = u(gen);
        auto r = mootation::lsq::nnls(E, f);
        // Kuhn-Tucker (23.11)-(23.16): x >= 0, w = E^T (f - E x), w_j <= 0
        // where x_j = 0, w_j = 0 where x_j > 0 (to round-off)
        double scale = 1.0;
        for (double v : f) scale += std::fabs(v);
        for (std::size_t j = 0; j < n; ++j) {
            double w = 0.0;
            for (std::size_t i = 0; i < m; ++i) {
                double res = f[i];
                for (std::size_t k = 0; k < n; ++k) res -= E(i, k) * r.x[k];
                w += E(i, j) * res;
            }
            if (r.x[j] < 0.0 || (r.x[j] > 0.0 && std::fabs(w) > 1e-9 * scale) ||
                (r.x[j] == 0.0 && w > 1e-9 * scale))
                ++bad_nnls;
        }
        // LDP: min ||x|| s.t. G x >= h
        Matrix G(m, n);
        std::vector<double> h(m);
        for (double& v : G.a) v = u(gen);
        for (double& v : h) v = u(gen);
        auto l = mootation::lsq::ldp(G, h);
        if (l.status == mootation::lsq::Status::ok) {
            ++ldp_feasible;
            for (std::size_t i = 0; i < m; ++i) {
                double gx = 0.0;
                for (std::size_t j = 0; j < n; ++j) gx += G(i, j) * l.x[j];
                if (gx < h[i] - 1e-9 || l.lam[i] < 0.0) ++bad_ldp;
            }
            for (std::size_t j = 0; j < n; ++j) {          // x = G^T lam
                double s = 0.0;
                for (std::size_t i = 0; i < m; ++i) s += G(i, j) * l.lam[i];
                if (std::fabs(s - l.x[j]) > 1e-9 * (1.0 + std::fabs(l.x[j]))) ++bad_ldp;
            }
        }
    }
    check(bad_nnls == 0, "nnls: Kuhn-Tucker conditions on 2000 random problems");
    check(bad_ldp == 0 && ldp_feasible > 500,
          "ldp: feasible, x = G^T lam, lam >= 0 (" + std::to_string(ldp_feasible) + " feasible)");
}

// ── slsqp ───────────────────────────────────────────────────────────────────
// A problem from its functions; derivatives exact.
struct Prob {
    std::size_t n, me, mi;
    std::vector<double> lo, hi;
    std::function<double(const std::vector<double>&)> f;
    std::function<std::vector<double>(const std::vector<double>&)> grad;
    std::function<std::vector<double>(const std::vector<double>&)> c;     // c_E then c_I
    std::function<Matrix(const std::vector<double>&)> jac;
};

slsqp::LineSearch g_ls = slsqp::LineSearch::stop;   // the line search solve() uses
std::string g_tag = "slsqp";                          // and its name in the checks

slsqp::Result solve(const Prob& p, std::vector<double> x0, double tol = 1e-12, int it = 200)
{
    slsqp::NLP nlp;
    nlp.n = p.n;
    nlp.n_eq = p.me;
    nlp.n_ineq = p.mi;
    nlp.lo = p.lo;
    nlp.hi = p.hi;
    nlp.values = [&](const std::vector<double>& x, std::vector<double>& c) {
        if (p.me + p.mi) c = p.c(x);
        return p.f(x);
    };
    nlp.derivatives = [&](const std::vector<double>& x, std::vector<double>& g, Matrix& A) {
        g = p.grad(x);
        if (p.me + p.mi) A = p.jac(x);
    };
    return slsqp::minimize(nlp, std::move(x0), {tol, it, g_ls});
}

std::vector<double> lin(std::initializer_list<double> v) { return v; }

Matrix rows(std::initializer_list<std::initializer_list<double>> r)
{
    Matrix A(r.size(), r.begin()->size());
    std::size_t i = 0;
    for (auto& row : r) {
        std::size_t j = 0;
        for (double v : row) A(i, j++) = v;
        ++i;
    }
    return A;
}

void slsqp_problems()
{
    // 1. equality: min (x-1)^2 + (y-2)^2 s.t. x + y = 1 -> (0, 1)
    Prob eq{2, 1, 0, {-kInf, -kInf}, {kInf, kInf},
            [](auto& x) { return (x[0] - 1) * (x[0] - 1) + (x[1] - 2) * (x[1] - 2); },
            [](auto& x) { return lin({2 * (x[0] - 1), 2 * (x[1] - 2)}); },
            [](auto& x) { return lin({x[0] + x[1] - 1}); },
            [](auto&) { return rows({{1, 1}}); }};
    auto r = solve(eq, {5, -3});
    check(r.status == slsqp::Status::converged && near(r.x[0], 0, 1e-9) && near(r.x[1], 1, 1e-9),
          g_tag + ": quadratic, one equality (" + std::to_string(r.iterations) + " iterations)");

    // 2. inequality: min x^2 + y^2 s.t. x + y >= 1 -> (0.5, 0.5); infeasible start
    Prob in{2, 0, 1, {-kInf, -kInf}, {kInf, kInf},
            [](auto& x) { return x[0] * x[0] + x[1] * x[1]; },
            [](auto& x) { return lin({2 * x[0], 2 * x[1]}); },
            [](auto& x) { return lin({x[0] + x[1] - 1}); },
            [](auto&) { return rows({{1, 1}}); }};
    r = solve(in, {-3, -3});
    check(r.status == slsqp::Status::converged && near(r.x[0], 0.5, 1e-9) && near(r.x[1], 0.5, 1e-9),
          g_tag + ": quadratic, one inequality, infeasible start (" + std::to_string(r.iterations) +
              " iterations)");

    // 3. bounds active: min (x-2)^2 + (y+1)^2 on [0, 1]^2 -> (1, 0); from a corner
    Prob bx{2, 0, 0, {0, 0}, {1, 1},
            [](auto& x) { return (x[0] - 2) * (x[0] - 2) + (x[1] + 1) * (x[1] + 1); },
            [](auto& x) { return lin({2 * (x[0] - 2), 2 * (x[1] + 1)}); }, nullptr, nullptr};
    r = solve(bx, {0, 1});
    check(r.status == slsqp::Status::converged && near(r.x[0], 1, 1e-12) && near(r.x[1], 0, 1e-12),
          g_tag + ": active bounds, start at a corner (" + std::to_string(r.iterations) + " iterations)");
    r = solve(bx, {5, -5});
    check(near(r.x[0], 1, 1e-12) && near(r.x[1], 0, 1e-12), g_tag + ": a start outside the box is clipped");

    // 4. degenerate: the equality twice, and redundant active inequalities
    Prob dup{2, 2, 3, {-kInf, -kInf}, {kInf, kInf},
             [](auto& x) { return (x[0] - 1) * (x[0] - 1) + (x[1] - 2) * (x[1] - 2); },
             [](auto& x) { return lin({2 * (x[0] - 1), 2 * (x[1] - 2)}); },
             [](auto& x) {
                 return lin({x[0] + x[1] - 1, 2 * x[0] + 2 * x[1] - 2, 0.5 - x[0], 0.5 - x[0],
                             1 - 2 * x[0]});
             },
             [](auto&) { return rows({{1, 1}, {2, 2}, {-1, 0}, {-1, 0}, {-2, 0}}); }};
    r = solve(dup, {3, 3});
    // min under x + y = 1, x <= 0.5: (0, 1) satisfies x <= 0.5 -> (0, 1)
    check(r.status == slsqp::Status::converged && near(r.x[0], 0, 1e-9) && near(r.x[1], 1, 1e-9),
          g_tag + ": repeated equality, redundant inequalities (" + std::to_string(r.iterations) +
              " iterations)");
    Prob act{2, 0, 3, {-kInf, -kInf}, {kInf, kInf},
             [](auto& x) { return (x[0] - 1) * (x[0] - 1) + x[1] * x[1]; },
             [](auto& x) { return lin({2 * (x[0] - 1), 2 * x[1]}); },
             [](auto& x) { return lin({0.5 - x[0], 0.5 - x[0], 1 - 2 * x[0]}); },
             [](auto&) { return rows({{-1, 0}, {-1, 0}, {-2, 0}}); }};
    r = solve(act, {0.1, 0.3});
    check(r.status == slsqp::Status::converged && near(r.x[0], 0.5, 1e-9) && near(r.x[1], 0, 1e-9),
          g_tag + ": three active rows of one constraint (" + std::to_string(r.iterations) +
              " iterations)");

    // 5. incompatible linearization at the start: x <= 1, x^2 >= 0.25 at x = 0
    Prob inc{1, 0, 2, {-kInf}, {kInf},
             [](auto& x) { return (x[0] - 0.8) * (x[0] - 0.8); },
             [](auto& x) { return lin({2 * (x[0] - 0.8)}); },
             [](auto& x) { return lin({1 - x[0], x[0] * x[0] - 0.25}); },
             [](auto& x) { return rows({{-1}, {2 * x[0]}}); }};
    r = solve(inc, {0.0});
    check(r.status == slsqp::Status::converged && near(r.x[0], 0.8, 1e-9) && r.relaxed > 0,
          g_tag + ": incompatible linearization, relaxed (" + std::to_string(r.relaxed) + " QPs)");

    // 6. nonlinear infeasible start: min x + y s.t. x^2 + y^2 <= 2 -> (-1, -1)
    Prob circ{2, 0, 1, {-kInf, -kInf}, {kInf, kInf},
              [](auto& x) { return x[0] + x[1]; },
              [](auto&) { return lin({1, 1}); },
              [](auto& x) { return lin({2 - x[0] * x[0] - x[1] * x[1]}); },
              [](auto& x) { return rows({{-2 * x[0], -2 * x[1]}}); }};
    r = solve(circ, {3, 2});
    check(r.status == slsqp::Status::converged && near(r.x[0], -1, 1e-7) && near(r.x[1], -1, 1e-7),
          g_tag + ": disc, infeasible start (" + std::to_string(r.iterations) + " iterations)");

    // 7. Rosenbrock from (-1.2, 1)
    Prob ros{2, 0, 0, {-kInf, -kInf}, {kInf, kInf},
             [](auto& x) { return 100 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]) +
                                  (1 - x[0]) * (1 - x[0]); },
             [](auto& x) { return lin({-400 * x[0] * (x[1] - x[0] * x[0]) - 2 * (1 - x[0]),
                                       200 * (x[1] - x[0] * x[0])}); }, nullptr, nullptr};
    r = solve(ros, {-1.2, 1}, 1e-14, 500);
    check(near(r.x[0], 1, 1e-6) && near(r.x[1], 1, 1e-6),
          g_tag + ": Rosenbrock (" + std::to_string(r.iterations) + " iterations)");

    // 8. Hock & Schittkowski 71: f* = 17.0140173 at (1, 4.743, 3.82115, 1.37941)
    Prob hs71{4, 1, 1, {1, 1, 1, 1}, {5, 5, 5, 5},
              [](auto& x) { return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]; },
              [](auto& x) {
                  return lin({x[3] * (2 * x[0] + x[1] + x[2]), x[0] * x[3], x[0] * x[3] + 1,
                              x[0] * (x[0] + x[1] + x[2])});
              },
              [](auto& x) {
                  return lin({x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] - 40,
                              x[0] * x[1] * x[2] * x[3] - 25});
              },
              [](auto& x) {
                  return rows({{2 * x[0], 2 * x[1], 2 * x[2], 2 * x[3]},
                               {x[1] * x[2] * x[3], x[0] * x[2] * x[3], x[0] * x[1] * x[3],
                                x[0] * x[1] * x[2]}});
              }};
    r = solve(hs71, {1, 5, 5, 1});
    check(r.status == slsqp::Status::converged && near(r.f, 17.0140173, 1e-6) &&
          near(r.x[1], 4.743, 1e-3) && near(r.x[2], 3.82115, 1e-4),
          g_tag + ": Hock-Schittkowski 71 (" + std::to_string(r.iterations) + " iterations)");

    // 9. two problems of Powell 1978, §5 (Table 1: 5 and 6 iterations there,
    // to five significant digits): the post office parcel problem from
    // (10, 10, 10) -> (24, 12, 12), and Powell's problem with three nonlinear
    // equalities from (-2, 2, 2, -1, -1) -> f* = 0.0539498; checked as there, by
    // the point (round-off in c keeps the violation over tol = 1e-12 on the first)
    Prob pop{3, 0, 1, {0, 0, 0}, {42, 42, 42},
             [](auto& x) { return -x[0] * x[1] * x[2]; },
             [](auto& x) { return lin({-x[1] * x[2], -x[0] * x[2], -x[0] * x[1]}); },
             [](auto& x) { return lin({72 - x[0] - 2 * x[1] - 2 * x[2]}); },
             [](auto&) { return rows({{-1, -2, -2}}); }};
    r = solve(pop, {10, 10, 10});
    check(near(r.x[0], 24, 1e-6) && near(r.x[1], 12, 1e-6) &&
              near(r.x[2], 12, 1e-6),
          g_tag + ": post office parcel (" + std::to_string(r.iterations) + " iterations)");
    Prob pw{5, 3, 0, std::vector<double>(5, -kInf), std::vector<double>(5, kInf),
            [](auto& x) { return std::exp(x[0] * x[1] * x[2] * x[3] * x[4]); },
            [](auto& x) {
                const double e = std::exp(x[0] * x[1] * x[2] * x[3] * x[4]);
                return lin({e * x[1] * x[2] * x[3] * x[4], e * x[0] * x[2] * x[3] * x[4],
                            e * x[0] * x[1] * x[3] * x[4], e * x[0] * x[1] * x[2] * x[4],
                            e * x[0] * x[1] * x[2] * x[3]});
            },
            [](auto& x) {
                return lin({x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] + x[4] * x[4] - 10,
                            x[1] * x[2] - 5 * x[3] * x[4], x[0] * x[0] * x[0] + x[1] * x[1] * x[1] + 1});
            },
            [](auto& x) {
                return rows({{2 * x[0], 2 * x[1], 2 * x[2], 2 * x[3], 2 * x[4]},
                             {0, x[2], x[1], -5 * x[4], -5 * x[3]},
                             {3 * x[0] * x[0], 3 * x[1] * x[1], 0, 0, 0}});
            }};
    r = solve(pw, {-2, 2, 2, -1, -1});
    check(near(r.f, 0.0539498, 1e-7) && near(r.x[0], -1.717143, 1e-5),
          g_tag + ": Powell's problem (" + std::to_string(r.iterations) + " iterations)");
}

// ── nelder-mead ─────────────────────────────────────────────────────────────
void nm_problems()
{
    nm::Options o;
    o.tol = 1e-10;
    long long calls = 0;
    auto ros = [&](const std::vector<double>& x) {
        ++calls;
        return 100 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]) + (1 - x[0]) * (1 - x[0]);
    };
    auto r = nm::minimize(ros, {-1.2, 1}, {-2, -2}, {2, 2}, o);
    check(r.converged && near(r.x[0], 1, 1e-5) && near(r.x[1], 1, 1e-5),
          "nm: Rosenbrock (" + std::to_string(calls) + " evaluations)");
    calls = 0;
    auto quad = [&](const std::vector<double>& x) {
        ++calls;
        double s = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) s += (i + 1.0) * (x[i] - 0.3) * (x[i] - 0.3);
        return s;
    };
    r = nm::minimize(quad, {1, 1, 1, 1, 1}, std::vector<double>(5, 0.0), std::vector<double>(5, 1.0), o);
    bool ok = r.converged;
    for (double v : r.x) ok = ok && near(v, 0.3, 1e-5);
    check(ok, "nm: quadratic in 5 variables from a corner (" + std::to_string(calls) + " evaluations)");
    auto edge = [](const std::vector<double>& x) { return (x[0] - 2) * (x[0] - 2) + (x[1] - 0.5) * (x[1] - 0.5); };
    r = nm::minimize(edge, {0.2, 0.2}, {0, 0}, {1, 1}, o);
    check(r.x[0] == 1.0 && near(r.x[1], 0.5, 1e-5), "nm: minimum on the boundary");
}

}   // namespace

int main()
{
    lse_example();
    nnls_ldp_random();
    slsqp_problems();
    g_ls = slsqp::LineSearch::accept;
    g_tag = "slsqp accept";
    slsqp_problems();
    g_ls = slsqp::LineSearch::powell;
    g_tag = "slsqp powell";
    slsqp_problems();
    nm_problems();
    if (g_failed) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
