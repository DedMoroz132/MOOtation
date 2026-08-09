#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A small set of standard benchmark problems, as plain math plus one macro.
//
// WHY IT LOOKS LIKE THIS. Based_Individual keeps variables/objectives/limits
// private and befriends Problem<T> only, so no helper class can fill them in on
// your behalf — the evaluation has to happen inside the Problem<>
// specialization itself. Hence the split:
//
//   * a SPEC is a plain struct of static functions over std::vector<double>.
//     It has no idea what an individual is and is trivially unit-testable.
//   * MOOTATION_DEFINE_PROBLEM(Tag, Base, Spec) generates the tag individual
//     and the whole Problem<Tag> specialization around a spec. It is variadic
//     on purpose: a spec like DTLZ2<3, 12> contains a comma, which the
//     preprocessor would otherwise split into two macro arguments.
//
// Usage (at namespace scope, once per problem you want):
//
//     #include <mootation/mootation.hpp>
//     #include <mootation/problems/benchmarks.hpp>
//
//     MOOTATION_DEFINE_PROBLEM(MyDTLZ2, NSGAII_Individual,
//                              mootation::problems::DTLZ2<3, 12>)
//
//     int main() {
//         using namespace mootation;
//         Problem<MyDTLZ2>   prob;
//         DataVault<MyDTLZ2> vault(91, prob);
//         Optimizer<MyDTLZ2, NSGAIICore<MyDTLZ2>> opt(std::move(vault), defer_setup);
//         opt.get_algorithm().set_seed(1);
//         opt.setup();
//         opt.optimize(200);
//     }
//
// The objective and variable counts are template parameters, so a spec is a
// compile-time constant and the vault sizes itself with no runtime plumbing.
//
// SCOPE. These are the problems the library's own tests use, not a benchmark
// suite. DTLZ1-4 are Deb, Thiele, Laumanns & Zitzler (2005); ZDT1-3 are
// Zitzler, Deb & Thiele (2000). The full WFG/MaF families are deliberately not
// here — they are large, and their reference implementations are the authors'
// own. Write your own spec when you need them; the macro takes anything with
// the same members.
// ============================================================================

#include <cmath>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace mootation::problems {

namespace detail {

inline double half_pi() { return std::acos(-1.0) / 2.0; }

// g of the DTLZ "distance" block: sum of (x_i - 0.5)^2 over the tail.
inline double g_sphere(const std::vector<double>& x, int m)
{
    double g = 0.0;
    for (std::size_t i = static_cast<std::size_t>(m) - 1; i < x.size(); ++i) {
        const double d = x[i] - 0.5;
        g += d * d;
    }
    return g;
}

// g of DTLZ1/DTLZ3: the multimodal variant.
inline double g_multimodal(const std::vector<double>& x, int m)
{
    const double pi = std::acos(-1.0);
    double s = 0.0;
    for (std::size_t i = static_cast<std::size_t>(m) - 1; i < x.size(); ++i) {
        const double d = x[i] - 0.5;
        s += d * d - std::cos(20.0 * pi * d);
    }
    const double k = static_cast<double>(x.size()) - (static_cast<double>(m) - 1.0);
    return 100.0 * (k + s);
}

} // namespace detail

// ── DTLZ1: linear front, sum f_i = 0.5. Multimodal, hard to converge. ───────
template <int M = 3, int N = 7>
struct DTLZ1 {
    static constexpr int    NVARS = N;
    static constexpr int    NOBJS = M;
    static constexpr int    NLIMS = 0;
    static constexpr double LO = 0.0, HI = 1.0;

    static void eval(const std::vector<double>& x, std::vector<double>& f)
    {
        const double g = detail::g_multimodal(x, M);
        for (int j = 0; j < M; ++j) {
            double v = 0.5 * (1.0 + g);
            for (int i = 0; i < M - 1 - j; ++i)
                v *= x[static_cast<std::size_t>(i)];
            if (j > 0)
                v *= (1.0 - x[static_cast<std::size_t>(M - 1 - j)]);
            f[static_cast<std::size_t>(j)] = v;
        }
    }
    static void eval_limits(const std::vector<double>&, std::vector<double>&) {}
};

// ── DTLZ2: concave front, the unit sphere ||f|| = 1. ────────────────────────
template <int M = 3, int N = 12>
struct DTLZ2 {
    static constexpr int    NVARS = N;
    static constexpr int    NOBJS = M;
    static constexpr int    NLIMS = 0;
    static constexpr double LO = 0.0, HI = 1.0;

    static void eval(const std::vector<double>& x, std::vector<double>& f)
    {
        const double g  = detail::g_sphere(x, M);
        const double hp = detail::half_pi();
        for (int j = 0; j < M; ++j) {
            double v = 1.0 + g;
            for (int i = 0; i < M - 1 - j; ++i)
                v *= std::cos(x[static_cast<std::size_t>(i)] * hp);
            if (j > 0)
                v *= std::sin(x[static_cast<std::size_t>(M - 1 - j)] * hp);
            f[static_cast<std::size_t>(j)] = v;
        }
    }
    static void eval_limits(const std::vector<double>&, std::vector<double>&) {}
};

// ── DTLZ3: DTLZ2's front with DTLZ1's multimodal g. ─────────────────────────
template <int M = 3, int N = 12>
struct DTLZ3 {
    static constexpr int    NVARS = N;
    static constexpr int    NOBJS = M;
    static constexpr int    NLIMS = 0;
    static constexpr double LO = 0.0, HI = 1.0;

    static void eval(const std::vector<double>& x, std::vector<double>& f)
    {
        const double g  = detail::g_multimodal(x, M);
        const double hp = detail::half_pi();
        for (int j = 0; j < M; ++j) {
            double v = 1.0 + g;
            for (int i = 0; i < M - 1 - j; ++i)
                v *= std::cos(x[static_cast<std::size_t>(i)] * hp);
            if (j > 0)
                v *= std::sin(x[static_cast<std::size_t>(M - 1 - j)] * hp);
            f[static_cast<std::size_t>(j)] = v;
        }
    }
    static void eval_limits(const std::vector<double>&, std::vector<double>&) {}
};

// ── DTLZ4: DTLZ2 with x_i^alpha, alpha = 100 — biased density on the front. ─
template <int M = 3, int N = 12>
struct DTLZ4 {
    static constexpr int    NVARS = N;
    static constexpr int    NOBJS = M;
    static constexpr int    NLIMS = 0;
    static constexpr double LO = 0.0, HI = 1.0;
    static constexpr double ALPHA = 100.0;

    static void eval(const std::vector<double>& x, std::vector<double>& f)
    {
        const double g  = detail::g_sphere(x, M);
        const double hp = detail::half_pi();
        for (int j = 0; j < M; ++j) {
            double v = 1.0 + g;
            for (int i = 0; i < M - 1 - j; ++i)
                v *= std::cos(std::pow(x[static_cast<std::size_t>(i)], ALPHA) * hp);
            if (j > 0)
                v *= std::sin(std::pow(x[static_cast<std::size_t>(M - 1 - j)], ALPHA) * hp);
            f[static_cast<std::size_t>(j)] = v;
        }
    }
    static void eval_limits(const std::vector<double>&, std::vector<double>&) {}
};

// ── ZDT1: convex front, f2 = 1 - sqrt(f1). ──────────────────────────────────
template <int N = 30>
struct ZDT1 {
    static constexpr int    NVARS = N;
    static constexpr int    NOBJS = 2;
    static constexpr int    NLIMS = 0;
    static constexpr double LO = 0.0, HI = 1.0;

    static void eval(const std::vector<double>& x, std::vector<double>& f)
    {
        double s = 0.0;
        for (std::size_t i = 1; i < x.size(); ++i) s += x[i];
        const double g = 1.0 + 9.0 * s / (static_cast<double>(x.size()) - 1.0);
        f[0] = x[0];
        f[1] = g * (1.0 - std::sqrt(f[0] / g));
    }
    static void eval_limits(const std::vector<double>&, std::vector<double>&) {}
};

// ── ZDT2: concave front, f2 = 1 - (f1/g)^2. ─────────────────────────────────
template <int N = 30>
struct ZDT2 {
    static constexpr int    NVARS = N;
    static constexpr int    NOBJS = 2;
    static constexpr int    NLIMS = 0;
    static constexpr double LO = 0.0, HI = 1.0;

    static void eval(const std::vector<double>& x, std::vector<double>& f)
    {
        double s = 0.0;
        for (std::size_t i = 1; i < x.size(); ++i) s += x[i];
        const double g = 1.0 + 9.0 * s / (static_cast<double>(x.size()) - 1.0);
        f[0] = x[0];
        const double r = f[0] / g;
        f[1] = g * (1.0 - r * r);
    }
    static void eval_limits(const std::vector<double>&, std::vector<double>&) {}
};

// ── ZDT3: DISCONNECTED front. Note f2 goes NEGATIVE — this is the problem
//    that motivated several deviations in this library (RDE-3, ISDE-3): an
//    algorithm that assumes f >= 0 loses whole segments of the front here. ───
template <int N = 30>
struct ZDT3 {
    static constexpr int    NVARS = N;
    static constexpr int    NOBJS = 2;
    static constexpr int    NLIMS = 0;
    static constexpr double LO = 0.0, HI = 1.0;

    static void eval(const std::vector<double>& x, std::vector<double>& f)
    {
        const double pi = std::acos(-1.0);
        double s = 0.0;
        for (std::size_t i = 1; i < x.size(); ++i) s += x[i];
        const double g = 1.0 + 9.0 * s / (static_cast<double>(x.size()) - 1.0);
        f[0] = x[0];
        const double r = f[0] / g;
        f[1] = g * (1.0 - std::sqrt(r) - r * std::sin(10.0 * pi * f[0]));
    }
    static void eval_limits(const std::vector<double>&, std::vector<double>&) {}
};

// ── C-DTLZ2: DTLZ2 with one inequality constraint, x0 <= NUM/DEN.
//    Exists so the constraint machinery has something to run on; the library's
//    own tests use exactly this. Convention: limits[k] <= 0 is satisfied,
//    > 0 is the violation, and CV = sum of the positive parts. ───────────────
template <int M = 3, int N = 7, int NUM = 1, int DEN = 2>
struct CDTLZ2 {
    static constexpr int    NVARS = N;
    static constexpr int    NOBJS = M;
    static constexpr int    NLIMS = 1;
    static constexpr double LO = 0.0, HI = 1.0;

    static void eval(const std::vector<double>& x, std::vector<double>& f)
    {
        DTLZ2<M, N>::eval(x, f);
    }
    static void eval_limits(const std::vector<double>& x, std::vector<double>& lim)
    {
        lim[0] = x[0] - static_cast<double>(NUM) / static_cast<double>(DEN);
    }
};

} // namespace mootation::problems

// Declares `TAG` (an individual of type BASE) and the Problem<TAG>
// specialization that evaluates the spec. Use at namespace scope.
#define MOOTATION_DEFINE_PROBLEM(TAG, BASE, ...)                              \
    struct TAG : public mootation::BASE {};                                   \
    namespace mootation {                                                     \
    template <>                                                               \
    class Problem<TAG> {                                                      \
        using Spec_ = __VA_ARGS__;                                            \
                                                                              \
    public:                                                                   \
        std::vector<std::pair<std::optional<double>, std::optional<double>>>  \
            bounds = std::vector<std::pair<std::optional<double>,             \
                                           std::optional<double>>>(           \
                Spec_::NVARS,                                                 \
                std::pair<std::optional<double>, std::optional<double>>{      \
                    Spec_::LO, Spec_::HI});                                   \
        int  get_vars_n()     const { return Spec_::NVARS; }                  \
        int  get_bin_vars_n() const { return 0; }                             \
        int  get_objs_n()     const { return Spec_::NOBJS; }                  \
        int  get_lims_n()     const { return Spec_::NLIMS; }                  \
        void calc_objs(TAG& ind) const                                        \
        {                                                                     \
            Spec_::eval(ind.variables, ind.objectives);                       \
            if (Spec_::NLIMS > 0)                                             \
                Spec_::eval_limits(ind.variables, ind.limits);                \
        }                                                                     \
    };                                                                        \
    }
