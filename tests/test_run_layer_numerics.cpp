// SPDX-License-Identifier: Apache-2.0
// The run layer's numerics in C++ (numpy_sum.hpp, nondominated.hpp,
// archive_select.hpp, grid_archive.hpp, indicators.hpp): what can be checked
// without NumPy. python/test_run_layer.py compares them with the NumPy
// reference bit for bit; here, the sums against values NumPy 2.5 gives, the
// filter and the archive against brute force, the threads against one thread,
// the incremental front against a full recomputation, and the slot order of
// the archive's dethroned extremes against Python 3.13's sets.
#include <mootation/archive_select.hpp>
#include <mootation/grid_archive.hpp>
#include <mootation/indicators.hpp>
#include <mootation/nondominated.hpp>
#include <mootation/numpy_sum.hpp>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int failures = 0;
#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                \
            ++failures;                                                                \
        }                                                                              \
    } while (0)

static bool dominates(const double* p, const double* q, std::size_t m)
{
    bool lt = false;
    for (std::size_t j = 0; j < m; ++j) {
        if (p[j] > q[j]) return false;
        lt = lt || p[j] < q[j];
    }
    return lt;
}

static void sums()
{
    // np.sum((np.arange(1, 20) / 7.0)[:n]) for n < 20, and of the 19 values
    // repeated, for larger n (NumPy 2.5)
    std::vector<double> a;
    for (int r = 0; r < 20; ++r)
        for (int i = 1; i < 20; ++i) a.push_back(i / 7.0);
    CHECK(mootation::np::sum(a.data(), 1) == 0.14285714285714285);
    CHECK(mootation::np::sum(a.data(), 7) == 4.0);
    CHECK(mootation::np::sum(a.data(), 8) == 5.142857142857143);
    CHECK(mootation::np::sum(a.data(), 9) == 6.428571428571429);
    CHECK(mootation::np::sum(a.data(), 17) == 21.857142857142858);
    CHECK(mootation::np::sum(a.data(), 130) == 182.28571428571428);
    CHECK(mootation::np::sum(a.data(), 300) == 424.2857142857143);
}

static void filter()
{
    std::mt19937_64 rng(1);
    std::uniform_int_distribution<int> grid(0, 3);
    for (int t = 0; t < 400; ++t) {
        const std::size_t m = 2 + static_cast<std::size_t>(t % 6), n = 5 + static_cast<std::size_t>(t % 60);
        std::vector<double> F(n * m);
        for (auto& v : F) v = static_cast<double>(grid(rng));
        // brute force: distinct rows that no row dominates
        std::vector<char> want(n, 1);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < n && want[i]; ++k)
                if (dominates(&F[k * m], &F[i * m], m)) want[i] = 0;
        const auto mask = mootation::nd::mask(F.data(), n, m);
        for (std::size_t i = 0; i < n; ++i) CHECK(mask[i] == want[i]);
        const auto keep = mootation::nd::filter(F.data(), n, m);
        std::size_t distinct = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (!want[i]) continue;
            bool first = true;
            for (std::size_t k = 0; k < i && first; ++k) {
                bool same = true;
                for (std::size_t j = 0; j < m; ++j) same = same && F[k * m + j] == F[i * m + j];
                first = !same;
            }
            distinct += first;
        }
        CHECK(keep.size() == distinct);
        for (std::size_t s = 1; s < keep.size(); ++s) {      // in lexicographic order
            const double* p = &F[keep[s - 1] * m];
            const double* q = &F[keep[s] * m];
            std::size_t j = 0;
            while (j < m && p[j] == q[j]) ++j;
            CHECK(j < m && p[j] < q[j]);
        }
    }
}

static void archive()
{
    // the slot order of the dethroned extremes: list({k for k in keys}) in Python 3.13
    using mootation::detail::cpython_set_order;
    CHECK((cpython_set_order({12, 5}) == std::vector<long long>{12, 5}));
    CHECK((cpython_set_order({33, 1}) == std::vector<long long>{33, 1}));
    CHECK((cpython_set_order({7, 15, 23, 31}) == std::vector<long long>{31, 15, 23, 7}));
    CHECK((cpython_set_order({100, 3, 77, 260, 9, 1000, 4}) ==
           std::vector<long long>{3, 260, 100, 4, 1000, 9, 77}));
    CHECK((cpython_set_order({513, 1, 257, 769, 1025, 2049, 3, 11, 19}) ==
           std::vector<long long>{1025, 513, 769, 1, 257, 2049, 3, 11, 19}));
    // what the archive keeps: mutually nondominated points, every objective's best
    std::mt19937_64 rng(2);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (std::size_t m : {2u, 3u, 5u}) {
        mootation::GridArchive a(static_cast<int>(m), 0, {}, {}, 0.0);
        std::vector<double> best(m, 1e300), f(m);
        for (int i = 0; i < 4000; ++i) {
            double s = 0.0;
            for (auto& v : f) { v = u(rng); s += v; }
            for (auto& v : f) v = v / s * (1.0 + 2.0 * u(rng) * (1.0 - i / 4000.0));
            a.add(f.data(), nullptr);
            for (std::size_t j = 0; j < m; ++j) best[j] = std::min(best[j], f[j]);
        }
        std::vector<double> F, X;
        std::vector<char> E;
        a.points(F, X, E);
        const std::size_t n = E.size();
        CHECK(n == a.size() && n > 0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t k = 0; k < n; ++k)
                CHECK(i == k || !dominates(&F[k * m], &F[i * m], m));
        for (std::size_t j = 0; j < m; ++j) {
            bool found = false;
            for (std::size_t i = 0; i < n; ++i) found = found || (E[i] && F[i * m + j] == best[j]);
            CHECK(found);
        }
    }
}

static void indicators()
{
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (std::size_t m : {2u, 3u, 8u, 11u}) {
        const std::size_t n = 70, r = 500;
        std::vector<double> F(n * m), R(r * m);
        for (auto& v : F) v = 1.2 * u(rng);
        for (auto& v : R) v = u(rng);
        std::vector<double> s1(r), t1(n), e1(r), s4(r), t4(n), e4(r);
        mootation::indicators::front_distances(F.data(), n, R.data(), r, m, s1.data(), t1.data(),
                                               e1.data(), 1);
        mootation::indicators::front_distances(F.data(), n, R.data(), r, m, s4.data(), t4.data(),
                                               e4.data(), 4);
        CHECK(s1 == s4 && t1 == t4 && e1 == e4);
        // the incremental front, fed in pieces, against the whole set at once
        std::vector<double> lo(m, 0.0), hi(m, 1.0);
        mootation::indicators::IncrementalFront inc(R.data(), r, m, lo, hi, {10.0, 0.3, 0.0});
        std::vector<long long> fe(n);
        for (std::size_t i = 0; i < n; ++i) fe[i] = static_cast<long long>(i + 1);
        inc.update(F.data(), 30, fe.data());
        inc.update(F.data() + 30 * m, n - 30, fe.data() + 30);
        CHECK(inc.distances() == s1);
        CHECK(inc.igd_plus() == mootation::np::mean(s1.data(), r));
        double e = e1[0];
        for (double v : e1) e = std::max(e, v);
        CHECK(inc.eps_plus() == e);
        CHECK(inc.reached()[0] == 1 && inc.reached()[2] == -1);
    }
    // DSS: the first k of the order are the order of k (incremental)
    mootation::select::Rows P;
    P.m = 3;
    P.n = 300;
    for (std::size_t i = 0; i < P.n * P.m; ++i) P.v.push_back(u(rng));
    const auto all = mootation::select::dss_order(P, P.n);
    const auto some = mootation::select::dss_order(P, 40);
    CHECK(all.size() == P.n);
    CHECK(std::equal(some.begin(), some.end(), all.begin()));
}

int main()
{
    sums();
    filter();
    archive();
    indicators();
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("run layer numerics OK\n");
    return EXIT_SUCCESS;
}
