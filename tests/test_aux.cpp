// SPDX-License-Identifier: Apache-2.0
// Population-dependent objectives (task 26, item 24.5.1, A1; data_vault.hpp,
// POPULATION-DEPENDENT OBJECTIVES; run_core.hpp, RunConfig::aux):
//   * aux = "none" is the run it was: the same answer, bit for bit, as a run
//     whose configuration never names aux;
//   * the builtin "dcn" by hand: max DCN - DCN(i) in the scaled box (Tran,
//     Brockhoff and Derbel 2013, §2.1; AUX-1);
//   * a run with aux = "dcn" and with a user function hands back the
//     problem's M objectives only, the same on a second run (determinism),
//     and its evaluations are the run's own (k values cost none);
//   * the vault: the function sees the evaluated members that did not fail;
//     a failed slot gets one span above the worst; a seed of M values is
//     completed at the first read.

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <mootation/run_core.hpp>

#include "harness.hpp"

namespace {

using mootation::runner::RunConfig;
using mootation::runner::RunProblem;
using mootation::runner::RunResult;

RunProblem zdt1(int n)
{
    RunProblem p;
    p.bounds.assign(static_cast<std::size_t>(n), {0.0, 1.0});
    p.n_objectives = 2;
    p.evaluate = [n](std::vector<double> x) {
        double s = 0.0;
        for (int i = 1; i < n; ++i) s += x[static_cast<std::size_t>(i)];
        const double g = 1.0 + 9.0 * s / (n - 1);
        return std::vector<double>{x[0], g * (1.0 - std::sqrt(x[0] / g))};
    };
    return p;
}

RunResult run(const std::string& alg, RunConfig cfg)
{
    RunProblem p = zdt1(6);
    return mootation::runner::run(alg, p, cfg);
}

bool same(const RunResult& a, const RunResult& b)
{
    return a.objectives == b.objectives && a.variables == b.variables &&
           a.evaluations == b.evaluations;
}

// The vault alone, on the library's default Problem<> (ZDT1, 6 variables): k
// values per slot after the problem's M, one call per sync over the active
// set (parents and offspring), a lazy evaluation refreshes, a seed of M
// values is completed at its first read, a problem writing its M values in
// place into the M + k slots.
void vault_checks()
{
    using mootation::testing::check;
    using Ind = mootation::Based_Individual;
    mootation::Problem<Ind> prob;
    mootation::DataVault<Ind> v(4, prob);
    long long calls = 0;
    v.set_aux(2, [&](const std::vector<const std::vector<double>*>& X,
                     const std::vector<std::vector<double>>& F, std::vector<std::vector<double>>& A) {
        ++calls;
        for (std::size_t i = 0; i < X.size(); ++i) {
            A[i][0] = (*X[i])[0] * 10.0;
            A[i][1] = F[i][0] + static_cast<double>(X.size());
        }
    });
    check(v.objs_n() == 4 && v.user_objs_n() == 2 && v.aux_n() == 2, "vault: M + k objectives");
    for (int i = 0; i < 4; ++i) {
        std::vector<double> x(6, 0.0);
        x[0] = 0.1 * (i + 1);
        v.set_variables(static_cast<std::size_t>(i), x);
    }
    v.sync();
    check(calls == 1, "vault: one call per sync");
    bool values = true;
    for (int i = 0; i < 4; ++i) {
        const auto& f = v.objectives_of(static_cast<std::size_t>(i));
        values = values && f.size() == 4 && std::fabs(f[0] - 0.1 * (i + 1)) < 1e-15 &&
                 std::fabs(f[2] - (i + 1)) < 1e-12 && std::fabs(f[3] - (f[0] + 4.0)) < 1e-15;
    }
    check(values, "vault: the problem's values, then the function's");
    const int base = v.expand(2);
    std::vector<double> y(6, 0.0);
    y[0] = 0.9;
    v.set_variables(static_cast<std::size_t>(base), y);
    v.set_variables(static_cast<std::size_t>(base + 1), y);
    v.sync();
    check(calls == 2 && std::fabs(v.objectives_of(0)[3] - (0.1 + 6.0)) < 1e-15,
          "vault: refreshed over parents and offspring");
    v.reduce(4);
    v.set_variables(0, y);
    check(std::fabs(v.objectives_of(0)[2] - 9.0) < 1e-12 && calls == 3, "vault: a lazy evaluation refreshes");
    mootation::DataVault<Ind> w(2, prob);
    w.set_aux(1, [](const std::vector<const std::vector<double>*>& X,
                    const std::vector<std::vector<double>>&, std::vector<std::vector<double>>& A) {
        for (std::size_t i = 0; i < X.size(); ++i) A[i][0] = 7.0;
    });
    w.seed_individual(0, std::vector<double>(6, 0.2), {0.2, 0.5});
    w.seed_individual(1, std::vector<double>(6, 0.3), {0.3, 0.4});
    check(w.objectives_of(1)[2] == 7.0 && w.objectives_of(1)[0] == 0.3, "vault: a seed of M values completed");
    mootation::DataVault<Ind> u(2, prob);
    check(u.objs_n() == 2 && u.aux_n() == 0, "vault: without set_aux, M objectives");
}

}   // namespace

int main()
{
    using mootation::testing::check;

    vault_checks();

    // ── aux = "none" is the run it was ──────────────────────────────────────
    for (const char* alg : {"nsga2", "sms_emoa", "spea2", "ibea_eplus", "moead"}) {
        RunConfig base;
        base.pop_size = 20;
        base.n_gen = 15;
        base.seed = 7;
        RunConfig none = base;
        none.aux = "none";
        check(same(run(alg, base), run(alg, none)), std::string(alg) + ": aux = none, bit for bit");
    }

    // ── dcn by hand ─────────────────────────────────────────────────────────
    {
        std::vector<std::pair<double, double>> box{{0.0, 1.0}, {0.0, 1.0}};
        std::vector<double> a{0.0, 0.0}, b{0.1, 0.0}, c{1.0, 1.0};
        std::vector<const std::vector<double>*> X{&a, &b, &c};
        std::vector<std::vector<double>> A(3, std::vector<double>(1, -1.0));
        mootation::runner::aux_dcn(box, X, A);
        const double far = std::sqrt(0.9 * 0.9 + 1.0);
        check(std::fabs(A[0][0] - (far - 0.1)) < 1e-15 && std::fabs(A[1][0] - (far - 0.1)) < 1e-15 &&
                  A[2][0] == 0.0,
              "dcn: max DCN - DCN(i) by hand");
        std::vector<std::pair<double, double>> wide{{0.0, 2.0}, {0.0, 2.0}};
        std::vector<double> d{2.0, 2.0};
        std::vector<const std::vector<double>*> Y{&a, &d};
        std::vector<std::vector<double>> B(2, std::vector<double>(1, -1.0));
        mootation::runner::aux_dcn(wide, Y, B);
        check(B[0][0] == 0.0 && B[1][0] == 0.0, "dcn: two points are each other's nearest");
        std::vector<const std::vector<double>*> one{&a};
        std::vector<std::vector<double>> C(1, std::vector<double>(1, -1.0));
        mootation::runner::aux_dcn(box, one, C);
        check(C[0][0] == 0.0, "dcn: a population of one (AUX-1)");
    }

    // ── a run with aux: the problem's M back, deterministic ─────────────────
    for (const char* alg : {"nsga2", "sms_emoa", "spea2", "ibea_eplus"}) {
        RunConfig cfg;
        cfg.pop_size = 20;
        cfg.n_gen = 10;
        cfg.seed = 3;
        cfg.aux = "dcn";
        RunConfig plain = cfg;
        plain.aux = "none";
        const RunResult r1 = run(alg, cfg), r2 = run(alg, cfg), r0 = run(alg, plain);
        bool m_only = !r1.objectives.empty();
        for (const auto& f : r1.objectives) m_only = m_only && f.size() == 2;
        check(m_only, std::string(alg) + ": aux = dcn, the answer carries the problem's two objectives");
        check(same(r1, r2), std::string(alg) + ": aux = dcn, two runs alike");
        check(r1.evaluations == r0.evaluations,
              std::string(alg) + ": aux = dcn spends the evaluations the run would");
        check(r1.objectives != r0.objectives, std::string(alg) + ": aux = dcn changes the selection");
    }
    {
        // a user function: the members' first objective's rank among them
        RunConfig cfg;
        cfg.pop_size = 12;
        cfg.n_gen = 6;
        cfg.seed = 11;
        cfg.aux = "user";
        cfg.aux_n = 2;
        long long calls = 0;
        std::size_t widest = 0;
        cfg.aux_fn = [&](const std::vector<const std::vector<double>*>& X,
                         const std::vector<std::vector<double>>& F, std::vector<std::vector<double>>& A) {
            ++calls;
            widest = std::max(widest, X.size());
            for (std::size_t i = 0; i < F.size(); ++i) {
                double r = 0.0;
                for (std::size_t j = 0; j < F.size(); ++j) r += F[j][0] < F[i][0] ? 1.0 : 0.0;
                A[i][0] = r;
                A[i][1] = -r;
            }
        };
        const RunResult r = run("nsga2", cfg);
        check(calls > 0 && widest == 24, "user aux: called on parents and offspring (2N)");
        bool m_only = true;
        for (const auto& f : r.objectives) m_only = m_only && f.size() == 2;
        check(m_only, "user aux: the answer carries the problem's two objectives");
        RunConfig bad = cfg;
        bad.aux_fn = nullptr;
        bool threw = false;
        try { run("nsga2", bad); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "user aux without a function is refused");
        RunConfig word = cfg;
        word.aux = "novelty";
        threw = false;
        try { run("nsga2", word); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "an unknown aux is refused");
    }

    // ── a warm start of the problem's M objectives ──────────────────────────
    {
        RunConfig cfg;
        cfg.pop_size = 4;
        cfg.n_gen = 3;
        cfg.seed = 5;
        cfg.aux = "dcn";
        cfg.seed_variables = {{0.1, 0, 0, 0, 0, 0}, {0.2, 0, 0, 0, 0, 0},
                              {0.5, 0, 0, 0, 0, 0}, {0.9, 0, 0, 0, 0, 0}};
        for (const auto& x : cfg.seed_variables) cfg.seed_objectives.push_back({x[0], 1.0 - std::sqrt(x[0])});
        const RunResult r = run("nsga2", cfg);
        bool ok = !r.objectives.empty();
        for (const auto& f : r.objectives) ok = ok && f.size() == 2 && std::isfinite(f[1]);
        check(ok, "aux = dcn: a warm start of two objectives runs");
    }

    // ── sym in the multi-objective cores (24.5.4; MSYM-1, MSYM-2, AUX-2) ────
    {
        using mootation::scalar::symmetry::Blocks;
        // two objectives invariant under permutations of three blocks of two
        auto blocky = [] {
            RunProblem p;
            p.bounds.assign(6, {0.0, 1.0});
            p.n_objectives = 2;
            p.evaluate = [](std::vector<double> x) {
                double a = 0.0, b = 0.0;
                for (int k = 0; k < 3; ++k) {
                    const double u = x[static_cast<std::size_t>(2 * k)], v = x[static_cast<std::size_t>(2 * k + 1)];
                    a += u * u + v;
                    b += (u - 1.0) * (u - 1.0) + v;
                }
                return std::vector<double>{a, b};
            };
            return p;
        };
        RunConfig cfg;
        cfg.pop_size = 16;
        cfg.n_gen = 8;
        cfg.seed = 9;
        cfg.sym = "canon";
        cfg.sym_blocks = {Blocks{0, 2, 3}};
        RunProblem p1 = blocky(), p2 = blocky();
        const RunResult r1 = mootation::runner::run("nsga2", p1, cfg), r2 = mootation::runner::run("nsga2", p2, cfg);
        bool sorted = !r1.variables.empty();
        for (const auto& x : r1.variables) {
            const std::vector<double> b0(x.begin(), x.begin() + 2), b1(x.begin() + 2, x.begin() + 4),
                b2(x.begin() + 4, x.begin() + 6);
            sorted = sorted && !(b1 < b0) && !(b2 < b1);
        }
        check(sorted, "sym = canon: every member's blocks in canonical (sorted) order");
        check(same(r1, r2), "sym = canon: two runs alike");
        // aux_metric = sym: a permutation of a point is at distance 0
        mootation::scalar::symmetry::Sym s;
        s.groups = {Blocks{0, 2, 3}};
        std::vector<double> a{0.1, 0.2, 0.5, 0.6, 0.9, 0.0}, b{0.9, 0.0, 0.1, 0.2, 0.5, 0.6}, c{0.3, 0.3, 0.3, 0.3, 0.3, 0.3};
        std::vector<const std::vector<double>*> X{&a, &b, &c};
        std::vector<std::vector<double>> A(3, std::vector<double>(1, -1.0)), E(3, std::vector<double>(1, -1.0));
        std::vector<std::pair<double, double>> box(6, {0.0, 1.0});
        mootation::runner::aux_dcn(box, X, A, &s);
        mootation::runner::aux_dcn(box, X, E);
        check(A[0][0] == A[1][0] && A[0][0] > A[2][0] && A[2][0] == 0.0,
              "aux_metric = sym: a and b, one point permuted, are nearest at distance 0");
        check(E[0][0] < A[0][0], "aux_metric = euclid sees them apart");
        RunConfig dsym = cfg;
        dsym.sym = "off";
        dsym.aux = "dcn";
        dsym.aux_metric = "sym";
        RunProblem p3 = blocky();
        bool ok = true;
        try { mootation::runner::run("nsga2", p3, dsym); } catch (const std::exception&) { ok = false; }
        check(ok, "aux = dcn with aux_metric = sym runs");
        auto refused = [&](RunConfig c, RunProblem p) {
            try { mootation::runner::run("nsga2", p, c); } catch (const std::invalid_argument&) { return true; }
            return false;
        };
        RunConfig al = cfg;
        al.sym = "align";
        check(refused(al, blocky()), "sym = align refused in the multi-objective cores (MSYM-1)");
        RunConfig nob = cfg;
        nob.sym_blocks.clear();
        check(refused(nob, blocky()), "sym = canon without blocks refused");
        RunProblem uneven = blocky();
        uneven.bounds[4] = {0.0, 2.0};
        check(refused(cfg, uneven), "blocks of a group in other bounds refused (SYM-6)");
        RunConfig kinds = cfg;
        kinds.var_kinds = {1, 0, 0, 0, 0, 0};
        check(refused(kinds, blocky()), "blocks of a group with other variable kinds refused (MSYM-2)");
    }

    // ── local search in the cores (A7; LS-1 ... LS-4) ───────────────────────
    {
        auto sphere = [](int n) {
            RunProblem p;
            p.bounds.assign(static_cast<std::size_t>(n), {-1.0, 1.0});
            p.n_objectives = 1;
            p.evaluate = [](std::vector<double> x) {
                double s = 0.0;
                for (double v : x) s += (v - 0.3) * (v - 0.3);
                return std::vector<double>{s};
            };
            return p;
        };
        RunConfig cfg;
        cfg.pop_size = 8;
        cfg.n_gen = 1;
        cfg.seed = 4;
        cfg.aux = "dcn";
        cfg.local = "nm";
        cfg.local_evals = 400;
        RunProblem a = sphere(2), b = sphere(2);
        const RunResult r1 = mootation::runner::run("nsga2", a, cfg), r2 = mootation::runner::run("nsga2", b, cfg);
        double best = std::numeric_limits<double>::infinity();
        for (const auto& f : r1.objectives) best = std::min(best, f[0]);
        check(best < 1e-12, "local = nm: the sphere's minimum in one generation (A7's check)");
        check(same(r1, r2), "local = nm: two runs alike");
        RunConfig plain = cfg;
        plain.local = "none";
        RunProblem c = sphere(2);
        const RunResult r0 = mootation::runner::run("nsga2", c, plain);
        check(r1.evaluations > r0.evaluations && r1.evaluations <= r0.evaluations * 400,
              "local = nm: the searches' evaluations counted, at most local_evals per point");
        auto refused = [&](RunConfig k, RunProblem p) {
            try { mootation::runner::run("nsga2", p, k); } catch (const std::invalid_argument&) { return true; }
            return false;
        };
        check(refused(cfg, zdt1(6)), "local = nm with two objectives refused (LS-1)");
        RunProblem con = sphere(2);
        con.n_limits = 1;
        con.limits_batch = [](std::vector<std::vector<double>> X) {
            return std::vector<std::vector<double>>(X.size(), std::vector<double>{-1.0});
        };
        check(refused(cfg, con), "local = nm with constraints refused (LS-2)");
        RunConfig word = cfg;
        word.local = "sqp";
        check(refused(word, sphere(2)), "an unknown local refused");
    }

    return mootation::testing::report("test_aux");
}
