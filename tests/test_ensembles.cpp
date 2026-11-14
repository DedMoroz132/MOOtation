// SPDX-License-Identifier: Apache-2.0
// Ensembles by residual blocks (task 28, §28.2, §28.3):
// stacking.hpp —
//   * a member with no error among others: its weight 1, the others' 0;
//   * the weights >= 0 on members that pull both ways; one member: alpha = 1
//     (STK-1);
//   * sum_to_one: the weights add up to 1 and the fit is no better than
//     without it; two members alike: the penalty's path, sum 1 within 1e-5
//     (STK-2);
// bootstrap.hpp —
//   * every count 1 (unit_counts): member k is the plain run of seed
//     seed + 1 + k, bit for bit (BT-1);
//   * out of bag: a share of the blocks near (1 - 1/B)^B (B = 200, 40
//     members); one seed one ensemble;
//   * probe blocks: weight 0 in every member, their rows' mean and spread;
//     refit's members start from the plain fit (BT-4);
//   * refused: one block, no residuals, a probe out of range (BT-5).

#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/random.hpp>
#include <mootation/scalar/bootstrap.hpp>
#include <mootation/scalar/stacking.hpp>

namespace sc = mootation::scalar;
namespace st = mootation::scalar::stacking;
namespace bt = mootation::scalar::bootstrap;
using Vec = std::vector<double>;

static int failures = 0;

static void check(bool ok, const std::string& what)
{
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

static void stacking()
{
    const std::size_t m = 30;
    Vec y(m);
    std::vector<Vec> Y(m, Vec(3));
    for (std::size_t i = 0; i < m; ++i) {
        const double t = 0.1 * static_cast<double>(i);
        y[i] = std::sin(t) + 0.5 * t;
        Y[i] = {y[i] + 0.3 * std::cos(3.0 * t), y[i], 0.8 * y[i] - 0.2};   // the second exact
    }
    const st::Result a = st::stack(Y, y);
    check(std::fabs(a.alpha[1] - 1.0) < 1e-12 && a.alpha[0] < 1e-12 && a.alpha[2] < 1e-12 && a.method == "nnls",
          "a member with no error: its weight 1, the others' 0");
    std::vector<Vec> Z(m, Vec(3));
    for (std::size_t i = 0; i < m; ++i) {
        const double t = 0.1 * static_cast<double>(i);
        Z[i] = {y[i] + 1.0, y[i] - 1.0, -y[i] * y[i]};   // over, under, opposite (independent of the two)
        (void)t;
    }
    const st::Result b = st::stack(Z, y);
    bool nonneg = true;
    for (double v : b.alpha) nonneg = nonneg && v >= 0.0;
    std::vector<Vec> one(m, Vec(1));
    for (std::size_t i = 0; i < m; ++i) one[i][0] = 2.0 * y[i];
    const st::Result c = st::stack(one, y);
    check(nonneg && b.alpha[2] == 0.0 && c.alpha == Vec{1.0} && c.method == "one",
          "the weights >= 0 (the opposite member 0); one member: alpha = 1 (STK-1)");
    st::Options so;
    so.sum_to_one = true;
    const st::Result d = st::stack(Z, y, {}, so);
    double sum = 0.0;
    for (double v : d.alpha) sum += v;
    check(std::fabs(sum - 1.0) < 1e-12 && d.rss >= b.rss - 1e-12 * (1.0 + b.rss) && d.method == "lsei",
          "sum_to_one: the weights add up to 1 (" + std::to_string(sum) + "), the fit no better than without");
    std::vector<Vec> twin(m, Vec(3));
    for (std::size_t i = 0; i < m; ++i) twin[i] = {Z[i][0], Z[i][0], Z[i][1]};
    const st::Result e = st::stack(twin, y, {}, so);
    double s2 = 0.0;
    for (double v : e.alpha) s2 += v;
    check(std::fabs(s2 - 1.0) < 1e-5 && e.method == "nnls_penalty",
          "two members alike with sum_to_one: the penalty's path, sum " + std::to_string(s2) + " (STK-2)");
}

// y = 1 + 2 t + noise, one residual a block, B blocks
static sc::Problem line(std::size_t B)
{
    sc::Problem P;
    P.m = 1;
    P.lo = {-10.0, -10.0};
    P.hi = {10.0, 10.0};
    P.n_res = static_cast<int>(B);
    P.blocks.assign(B, 1);
    P.weights.assign(B, 1.0);
    std::mt19937 g(5);
    auto data = std::make_shared<Vec>(B);
    for (std::size_t i = 0; i < B; ++i) (*data)[i] = 1.0 + 2.0 * (0.01 * static_cast<double>(i)) + 0.1 * (mootation::rnd::uniform01(g) - 0.5);
    P.residuals = [data](const Vec& x, Vec& r) {
        r.assign(data->size(), 0.0);
        for (std::size_t i = 0; i < data->size(); ++i) r[i] = x[0] + x[1] * (0.01 * static_cast<double>(i)) - (*data)[i];
    };
    return P;
}

static bool same(const sc::Result& a, const sc::Result& b)
{
    return a.variables == b.variables && a.objectives == b.objectives && a.evaluations == b.evaluations;
}

static void bootstrap()
{
    const sc::Problem P = line(200);
    sc::Start s0;
    s0.has_x0 = true;
    s0.x0 = {0.0, 0.0};
    bt::Options u;
    u.members = 3;
    u.seed = 10;
    u.unit_counts = true;
    const bt::Result a = bt::fit("lm", P, 2000, {}, u, s0);
    bool plain = true;
    for (std::size_t k = 0; k < 3; ++k)
        plain = plain && same(a.members[k].run, sc::run("lm", P, 1e-6, false, 0.0, 1, 2000, 11 + static_cast<unsigned>(k),
                                                        {}, {}, 0, s0));
    check(plain && a.oob_share == 0.0, "every count 1: member k the plain run of seed 11 + k, bit for bit (BT-1)");

    bt::Options o;
    o.members = 40;
    o.seed = 3;
    const bt::Result b = bt::fit("lm", P, 2000, {}, o, s0), c = bt::fit("lm", P, 2000, {}, o, s0);
    const double expect = std::pow(1.0 - 1.0 / 200.0, 200.0);
    bool alike = b.members.size() == c.members.size();
    for (std::size_t k = 0; alike && k < b.members.size(); ++k)
        alike = b.members[k].counts == c.members[k].counts && b.members[k].x == c.members[k].x &&
                b.members[k].oob == c.members[k].oob;
    char line1[200];
    std::snprintf(line1, sizeof line1, "out of bag: %.4f of the blocks, (1 - 1/200)^200 = %.4f; one seed one ensemble",
                  b.oob_share, expect);
    check(std::fabs(b.oob_share - expect) < 0.02 && alike && b.extra_evaluations == 40, line1);

    bt::Options p = o;
    p.members = 10;
    p.probe = {0, 199};
    p.mode = bt::Options::Mode::refit;
    const bt::Result d = bt::fit("lm", P, 2000, {}, p, s0);
    bool held = true, fin = d.probe_mean.size() == 2 && d.probe_sd.size() == 2;
    for (const bt::Member& mb : d.members) held = held && mb.counts[0] == 0 && mb.counts[199] == 0;
    for (std::size_t i = 0; fin && i < 2; ++i) fin = std::isfinite(d.probe_mean[i]) && d.probe_sd[i] > 0.0;
    char line2[200];
    std::snprintf(line2, sizeof line2,
                  "probes 0 and 199: weight 0 in every member; their rows' spread %.3g, %.3g; refit's base fit",
                  d.probe_sd[0], d.probe_sd[1]);
    check(held && fin && !d.base.variables.empty(), line2);

    auto refused = [](const std::function<void()>& g) {
        try {
            g();
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    sc::Problem one = P;
    one.blocks = {200};
    one.weights = {1.0};
    bt::Options bad = o;
    bad.probe = {200};
    check(refused([&] { bt::fit("lm", one, 100, {}, o, s0); }) && refused([&] { bt::fit("lm", P, 100, {}, bad, s0); }),
          "refused: one block, a probe out of range (BT-5)");
}

int main()
{
    stacking();
    bootstrap();
    std::printf(failures ? "FAILED\n" : "all passed\n");
    return failures ? 1 : 0;
}
