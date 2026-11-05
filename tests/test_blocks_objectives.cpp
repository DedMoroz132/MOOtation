// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// suite/blocks_objectives.hpp (task 19, §19.5):
//   - the families' residual blocks: chaboche1d's three amplitudes (80, 160,
//     240 steps), sumexp's two halves of the grid; they add up to n_res;
//   - f_b = 1/2 w_b ||r_b||^2 from one call of the residuals, block by block
//     (BO-1), at random points;
//   - sum_b f_b against the problem's objective at one objective: within 2
//     gamma_{n_res} of it (BO-2); one block of weight 1 (MGH, no res_blocks)
//     the same bit for bit;
//   - the weights scale each block; what passes through (residuals, canon,
//     start, the symmetries) is the problem's;
//   - refused: a problem without residuals, weights of the wrong count or
//     negative, blocks that do not add up.
// ============================================================================

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/suite/blocks_objectives.hpp>
#include <mootation/suite/suite.hpp>

namespace {

namespace S = mootation::suite;
using Vec = std::vector<double>;

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    if (!ok) ++g_failed;
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
}

S::Spec sym_spec(const std::string& family, const S::sym::Instance& in)
{
    S::Spec s;
    s.family = family;
    s.n_obj = 1;
    s.sym = in;
    return s;
}

std::shared_ptr<const S::Native> chaboche()
{
    S::sym::Instance in;
    in.teacher = 2;
    in.noise = 1e-3;
    return std::shared_ptr<const S::Native>(S::make(sym_spec("chaboche1d", in)));
}

std::shared_ptr<const S::Native> sumexp(int points)
{
    S::sym::Instance in;
    in.teacher = 3;
    in.points = points;
    in.ratio = 4.0;
    in.noise = 1e-3;
    return std::shared_ptr<const S::Native>(S::make(sym_spec("sumexp", in)));
}

// a point of the problem's box, uniform
Vec point(const S::Native& p, std::mt19937& rng)
{
    const S::Bounds b = dynamic_cast<const S::Problem&>(p).bounds();
    std::uniform_real_distribution<double> U(0.0, 1.0);
    Vec x(static_cast<std::size_t>(p.n_var));
    for (std::size_t j = 0; j < x.size(); ++j) x[j] = b[j].first + U(rng) * (b[j].second - b[j].first);
    return x;
}

double objective(const S::Native& p, const Vec& x)
{
    double f = 0.0;
    p.evaluate(x.data(), &f, nullptr);
    return f;
}

void the_families_blocks()
{
    const auto c = chaboche();
    check(c->res_blocks == std::vector<int>{80, 160, 240} && c->n_res == 480,
          "chaboche1d: the residuals' blocks are the three amplitudes, 80 + 160 + 240 = 480 (CH-4)");
    const auto s = sumexp(41);
    check(s->res_blocks == std::vector<int>{20, 21} && s->n_res == 41, "sumexp: the grid's two halves, 20 + 21 (SE-4)");
}

void objectives_and_their_sum()
{
    std::mt19937 rng(5);
    const double u = std::numeric_limits<double>::epsilon() / 2.0;
    for (const auto& p : {chaboche(), sumexp(41)}) {
        const S::BlocksObjectives a(p);
        const std::size_t B = a.sizes().size(), m = static_cast<std::size_t>(p->n_res);
        bool same = true, near = true;
        double worst = 0.0;
        for (int rep = 0; rep < 50; ++rep) {
            const Vec x = point(*p, rng);
            Vec r(m), f(B);
            p->residuals(x.data(), r.data(), nullptr);
            a.evaluate(x.data(), f.data(), nullptr);
            std::size_t at = 0;
            for (std::size_t b = 0; b < B; ++b) {                     // BO-1, by hand
                double s = 0.0;
                for (int k = 0; k < a.sizes()[b]; ++k, ++at) s += r[at] * r[at];
                same = same && f[b] == 0.5 * s;
            }
            const double f1 = objective(*p, x), t = a.total(f.data());
            const double gamma = static_cast<double>(m) * u / (1.0 - static_cast<double>(m) * u);
            near = near && std::fabs(t - f1) <= 2.0 * gamma * f1;
            worst = std::max(worst, std::fabs(t - f1) / (gamma * f1));
        }
        check(a.n_obj == static_cast<int>(B) && same,
              "f_b = 1/2 ||r_b||^2 from one call of the residuals, " + std::to_string(B) + " blocks (BO-1)");
        check(near, "sum_b f_b within 2 gamma_{n_res} of the objective at M = 1 (BO-2; worst " +
                        std::to_string(worst) + " gamma)");
    }
    // one block, weight 1: MGH Rosenbrock has no res_blocks
    S::Spec s;
    s.family = "MGH";
    s.n_var = 2;
    s.n_obj = 1;
    s.mgh_id = 1;
    s.mgh_m = 2;
    const std::shared_ptr<const S::Native> mgh(S::make(s));
    const S::BlocksObjectives one(mgh);
    bool bits = true;
    for (int rep = 0; rep < 50; ++rep) {
        std::uniform_real_distribution<double> U(-3.0, 3.0);
        const Vec x{U(rng), U(rng)};
        double f = 0.0;
        one.evaluate(x.data(), &f, nullptr);
        bits = bits && f == objective(*mgh, x) && one.total(&f) == f;
    }
    check(one.n_obj == 1 && bits, "one block of weight 1: the objective at M = 1 bit for bit (BO-2)");
}

void weights_and_what_passes_through()
{
    std::mt19937 rng(9);
    const auto p = chaboche();
    const S::BlocksObjectives a(p), w(p, {2.0, 0.5, 0.0});
    bool scaled = true, through = true;
    for (int rep = 0; rep < 20; ++rep) {
        const Vec x = point(*p, rng);
        Vec fa(3), fw(3), ra(480), rp(480), ca = x, cp = x;
        a.evaluate(x.data(), fa.data(), nullptr);
        w.evaluate(x.data(), fw.data(), nullptr);
        scaled = scaled && fw[0] == 2.0 * fa[0] && fw[1] == 0.5 * fa[1] && fw[2] == 0.0;
        a.residuals(x.data(), ra.data(), nullptr);
        p->residuals(x.data(), rp.data(), nullptr);
        a.canon(ca.data());
        p->canon(cp.data());
        through = through && ra == rp && ca == cp;
    }
    Vec s1(static_cast<std::size_t>(p->n_var)), s2 = s1;
    a.start(s1.data());
    p->start(s2.data());
    through = through && s1 == s2 && a.blocks.size() == p->blocks.size() && a.has_canon && !a.has_gradient;
    check(scaled, "the weights: f_b = (0.5 w_b) s_b, w = (2, 0.5, 0)");
    check(through, "residuals, canon, start and the symmetries are the problem's (BO-4)");
}

void refusals()
{
    auto refused = [](auto make) {
        try {
            make();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    S::sym::Instance in;
    in.dim = 7;
    const std::shared_ptr<const S::Native> lj(S::make(sym_spec("lj_cluster", in)));
    const auto c = chaboche();
    check(refused([&] { S::BlocksObjectives a(lj); }) && refused([&] { S::BlocksObjectives a(c, {1.0, 1.0}); }) &&
              refused([&] { S::BlocksObjectives a(c, {1.0, -1.0, 1.0}); }) &&
              refused([] { S::residual_blocks({2, 3}, 6); }) && refused([] { S::residual_blocks({0, 6}, 6); }),
          "refused: no residuals (lj_cluster), weights of the wrong count or negative, blocks that do not add up");
}

}   // namespace

int main()
{
    the_families_blocks();
    objectives_and_their_sum();
    weights_and_what_passes_through();
    refusals();
    if (g_failed) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
