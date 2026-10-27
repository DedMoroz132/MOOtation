#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The two ablation baselines of python/mootation/run/ablations.py on the
// library's own generator (task 17, stage 5): std::mt19937(seed) and rnd::
// (random.hpp), the same numbers on every platform, where ablations.py drew
// from NumPy's; the random choices in ablations.py's order, a float's power
// dm::pow. BASELINE version 2 (designs.hpp's for the sampling baselines).
//
//   random_selection_ea  NSGA-II's variation (bounded SBX, eta_c = 20,
//                        p_c = 0.9; polynomial mutation, eta_m = 20,
//                        p_m = 1/n) with N of the 2N parents and children
//                        kept uniformly at random; its answer its last
//                        population
//   gsemo                global SEMO on real variables: the nondominated set,
//                        one polynomially mutated member a step, its answer
//                        that set reduced to N by `select`
//
// The evaluation and the observer are the caller's (the campaign's, which
// feed its archive); a failed evaluation (a non-finite value) is no answer.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "detmath.hpp"
#include "random.hpp"

namespace mootation::ablations {

constexpr double kEtaC = 20.0, kPc = 0.9, kEtaM = 20.0;

using Rows = std::vector<std::vector<double>>;
using Evaluate = std::function<std::vector<double>(const std::vector<double>&)>;
using Observe = std::function<void(int, const Rows&, const Rows&)>;          // gen, objectives, variables
using Select = std::function<std::vector<std::size_t>(const Rows&, std::size_t)>;

struct Answer {
    Rows F, X;
    int steps = 0;
};

// bounded SBX on one pair (sbx.hpp; ablations.sbx): two children
template <class G>
inline void sbx(const std::vector<double>& p1, const std::vector<double>& p2, const std::vector<double>& lo,
                const std::vector<double>& hi, G& g, std::vector<double>& c1, std::vector<double>& c2)
{
    c1 = p1;
    c2 = p2;
    if (kPc < 1.0 && rnd::uniform01(g) > kPc) return;
    const double e = 1.0 / (kEtaC + 1.0);
    for (std::size_t j = 0; j < p1.size(); ++j) {
        if (rnd::uniform01(g) > 0.5) continue;
        if (std::fabs(p1[j] - p2[j]) < 1e-14) continue;
        const double y1 = std::min(p1[j], p2[j]), y2 = std::max(p1[j], p2[j]);
        const double dy = y2 - y1;
        const double u = rnd::uniform01(g);
        const double bl = 1.0 + 2.0 * (y1 - lo[j]) / dy;
        const double al = 2.0 - dm::pow(bl, -(kEtaC + 1.0));
        double bq = u <= 1.0 / al ? dm::pow(u * al, e) : dm::pow(1.0 / (2.0 - u * al), e);
        c1[j] = std::min(std::max(0.5 * ((y1 + y2) - bq * dy), lo[j]), hi[j]);
        const double bh = 1.0 + 2.0 * (hi[j] - y2) / dy;
        const double ah = 2.0 - dm::pow(bh, -(kEtaC + 1.0));
        bq = u <= 1.0 / ah ? dm::pow(u * ah, e) : dm::pow(1.0 / (2.0 - u * ah), e);
        c2[j] = std::min(std::max(0.5 * ((y1 + y2) + bq * dy), lo[j]), hi[j]);
        if (rnd::uniform01(g) < 0.5) std::swap(c1[j], c2[j]);
    }
}

// bounded polynomial mutation (poly_mutation.hpp; ablations.polynomial_mutation), in place
template <class G>
inline void polynomial_mutation(std::vector<double>& x, const std::vector<double>& lo, const std::vector<double>& hi,
                                G& g)
{
    const double pm = 1.0 / static_cast<double>(x.size());
    const double e = 1.0 / (kEtaM + 1.0);
    for (std::size_t j = 0; j < x.size(); ++j) {
        if (rnd::uniform01(g) > pm) continue;
        const double dx = hi[j] - lo[j];
        if (dx < 1e-14) continue;
        const double u = rnd::uniform01(g);
        double dq;
        if (u < 0.5) {
            const double b = std::min(std::max((x[j] - lo[j]) / dx, 0.0), 1.0);
            const double tmp = 2.0 * u + (1.0 - 2.0 * u) * dm::pow(1.0 - b, kEtaM + 1.0);
            dq = dm::pow(tmp, e) - 1.0;
        } else {
            const double b = std::min(std::max((hi[j] - x[j]) / dx, 0.0), 1.0);
            const double tmp = 2.0 * (1.0 - u) + (2.0 * u - 1.0) * dm::pow(1.0 - b, kEtaM + 1.0);
            dq = 1.0 - dm::pow(tmp, e);
        }
        x[j] = std::min(std::max(x[j] + dq * dx, lo[j]), hi[j]);
    }
}

inline bool finite(const std::vector<double>& f)
{
    for (double v : f)
        if (!std::isfinite(v)) return false;
    return true;
}

// the rows of a failed evaluation are no answer
inline void keep_ok(const Rows& F, const Rows& X, Rows& Fo, Rows& Xo)
{
    Fo.clear();
    Xo.clear();
    for (std::size_t i = 0; i < F.size(); ++i)
        if (finite(F[i])) {
            Fo.push_back(F[i]);
            Xo.push_back(X[i]);
        }
}

inline Answer random_selection_ea(const Evaluate& evaluate, const std::vector<double>& lo,
                                  const std::vector<double>& hi, std::size_t pop, std::size_t max_evaluations,
                                  std::int64_t seed, const Observe& observe, int record_every = 1)
{
    std::mt19937 g(static_cast<std::uint32_t>(seed));
    const std::size_t d = lo.size();
    Rows X(pop, std::vector<double>(d));
    for (auto& x : X)
        for (std::size_t j = 0; j < d; ++j) x[j] = lo[j] + rnd::uniform01(g) * (hi[j] - lo[j]);
    const std::size_t k0 = std::min(pop, max_evaluations);
    X.resize(k0);
    Rows F;
    for (const auto& x : X) F.push_back(evaluate(x));
    std::size_t spent = k0;
    int gen = 0;
    Rows Fo, Xo;
    if (observe) {
        keep_ok(F, X, Fo, Xo);
        observe(0, Fo, Xo);
    }
    std::vector<double> c1, c2;
    while (spent < max_evaluations) {
        const std::size_t k = std::min(pop, max_evaluations - spent);
        Rows kids;
        while (kids.size() < k) {
            const std::size_t a = static_cast<std::size_t>(rnd::below(g, X.size()));
            const std::size_t b = static_cast<std::size_t>(rnd::below(g, X.size()));
            sbx(X[a], X[b], lo, hi, g, c1, c2);
            polynomial_mutation(c1, lo, hi, g);
            kids.push_back(c1);
            if (kids.size() < k) {
                polynomial_mutation(c2, lo, hi, g);
                kids.push_back(c2);
            }
        }
        Rows Fk;
        for (const auto& x : kids) Fk.push_back(evaluate(x));
        spent += k;
        ++gen;
        Rows Xa = X, Fa = F;
        Xa.insert(Xa.end(), kids.begin(), kids.end());
        Fa.insert(Fa.end(), Fk.begin(), Fk.end());
        // N of the parents and children uniformly at random
        std::vector<std::size_t> order(Xa.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        rnd::shuffle(order.begin(), order.end(), g);
        order.resize(std::min(pop, Xa.size()));
        X.clear();
        F.clear();
        for (auto i : order) {
            X.push_back(Xa[i]);
            F.push_back(Fa[i]);
        }
        if (observe && (gen % std::max(1, record_every) == 0 || spent >= max_evaluations)) {
            keep_ok(F, X, Fo, Xo);
            observe(gen, Fo, Xo);
        }
    }
    Answer out;
    keep_ok(F, X, out.F, out.X);
    out.steps = gen;
    return out;
}

inline Answer gsemo(const Evaluate& evaluate, const std::vector<double>& lo, const std::vector<double>& hi,
                    std::size_t pop, std::size_t max_evaluations, std::int64_t seed, const Select& select,
                    const Observe& observe, int record_every = 1)
{
    std::mt19937 g(static_cast<std::uint32_t>(seed));
    const std::size_t d = lo.size();
    std::vector<double> x0(d);
    for (std::size_t j = 0; j < d; ++j) x0[j] = lo[j] + rnd::uniform01(g) * (hi[j] - lo[j]);
    std::vector<double> f0 = evaluate(x0);
    // a failed evaluation reads as infinitely bad: dominated by every point
    // that did not fail, and gone once one arrives
    if (!finite(f0)) std::fill(f0.begin(), f0.end(), std::numeric_limits<double>::infinity());
    Rows PX{x0}, PF{f0};
    std::size_t spent = 1;
    int step = 0;
    auto answer = [&](Rows& F, Rows& X) {
        F.clear();
        X.clear();
        for (auto i : select(PF, pop)) {
            F.push_back(PF[i]);
            X.push_back(PX[i]);
        }
    };
    Rows Fa, Xa;
    if (observe) {
        answer(Fa, Xa);
        observe(0, Fa, Xa);
    }
    while (spent < max_evaluations) {
        std::vector<double> y = PX[static_cast<std::size_t>(rnd::below(g, PX.size()))];
        polynomial_mutation(y, lo, hi, g);
        std::vector<double> fy = evaluate(y);
        if (!finite(fy)) std::fill(fy.begin(), fy.end(), std::numeric_limits<double>::infinity());
        ++spent;
        auto weakly = [](const std::vector<double>& a, const std::vector<double>& b) {   // a <= b everywhere
            for (std::size_t m = 0; m < a.size(); ++m)
                if (!(a[m] <= b[m])) return false;
            return true;
        };
        bool dominated = false;
        for (const auto& f : PF) dominated = dominated || weakly(f, fy);
        if (!dominated) {                        // nothing weakly dominates y: it enters, and
            Rows KX, KF;                         // drops what it weakly dominates
            for (std::size_t i = 0; i < PF.size(); ++i)
                if (!weakly(fy, PF[i])) {
                    KX.push_back(PX[i]);
                    KF.push_back(PF[i]);
                }
            KX.push_back(y);
            KF.push_back(fy);
            PX.swap(KX);
            PF.swap(KF);
        }
        // a "generation" is pop evaluations, so records line up with the others
        if (spent % pop == 0 || spent >= max_evaluations) {
            ++step;
            if (observe && (step % std::max(1, record_every) == 0 || spent >= max_evaluations)) {
                answer(Fa, Xa);
                observe(step, Fa, Xa);
            }
        }
    }
    Answer out;
    answer(out.F, out.X);
    out.steps = step;
    return out;
}

}   // namespace mootation::ablations
