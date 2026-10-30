// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// scalar/shade.hpp and scalar/lshade.hpp:
//   - SHADE, Tanabe & Fukunaga 2013, Table III (the 13 classical functions,
//     D = 30, N = H = 100): the mean best error at the paper's generation, 5
//     runs where the paper has 50, within 1.5 decades of the paper's mean —
//     f1, f6, f7 (noise from its own generator), f9, f10, f12, f13 (f3 and
//     f4 end 15 and 25 decades below the paper's means, whose deviations are
//     four times the means; the TASK18 report);
//   - the weighted means (13), (19) and L-SHADE's Algorithm 1 with ⊥;
//   - L-SHADE's LPSR (10): the population sizes over a run, the archive size
//     round(2.6 N), N_min = 4 at MAX_NFE; L-SHADE on f1 and f9 of the same
//     set at D = 10 within 10000 D evaluations below 1e-8 (CEC 2014's zero);
//   - a checksum of the runs, the same bits on every platform.
// ============================================================================

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <mootation/scalar/lshade.hpp>
#include <mootation/scalar/shade.hpp>

#include "yao1999.hpp"

namespace {

namespace S = mootation::scalar::shade;
namespace L = mootation::scalar::lshade;
using S::Vec;

int g_failed = 0;
std::uint64_t g_hash = 1469598103934665603ull;     // FNV-1a over the runs' bits

void mix(double v)
{
    std::uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    g_hash = (g_hash ^ b) * 1099511628211ull;
}

void check(bool ok, const std::string& what)
{
    if (!ok) ++g_failed;
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
}

double shade_run(int id, int generations, unsigned seed)
{
    S::Options o;
    std::mt19937 rng(seed), noise(seed + 7777u);
    S::SHADE de(Vec(30, -yao1999::range(id)), Vec(30, yao1999::range(id)), o, rng);
    double best = INFINITY;
    for (int g = 0; g <= generations; ++g) {
        const mootation::scalar::evo::Rows X = de.ask();
        Vec f(X.size());
        for (std::size_t i = 0; i < X.size(); ++i) {
            f[i] = yao1999::f(id, X[i], &noise);
            best = std::min(best, f[i]);
        }
        de.tell(f);
    }
    mix(best);
    for (double m : de.memory_f()) mix(m);
    return best;
}

void table_iii()
{
    struct Case {
        int id, generations;
        double paper;
    };
    const Case cases[] = {{1, 1500, 1.0e-70}, {6, 100, 2.7e+00},  {7, 3000, 5.8e-04}, {9, 1000, 1.6e-02},
                          {10, 500, 2.5e-10}, {12, 500, 3.7e-19}, {13, 500, 3.9e-18}};
    for (const Case& c : cases) {
        double sum = 0.0;
        for (unsigned r = 0; r < 5; ++r) sum += shade_run(c.id, c.generations, 700u + r);
        const double mean = sum / 5.0;
        char line[200];
        std::snprintf(line, sizeof line, "SHADE f%d, D = 30, generation %d: mean %.2e, paper %.1e", c.id,
                      c.generations, mean, c.paper);
        check(mean > 0.0 && std::fabs(std::log10(mean / c.paper)) <= 1.5, line);
    }
}

void means()
{
    const Vec gains = {1.0, 3.0};
    const Vec w = S::detail::weights(gains);
    const Vec s = {0.2, 0.6};
    check(w[0] == 0.25 && w[1] == 0.75, "w_k = dF_k / sum dF (14)");
    check(std::fabs(S::detail::mean_wa(w, s) - 0.5) < 1e-15, "mean_WA = sum w_k S_k (13)");
    check(std::fabs(S::detail::mean_wl(w, s) - (0.25 * 0.04 + 0.75 * 0.36) / 0.5) < 1e-15,
          "mean_WL = sum w S^2 / sum w S (19)");
    const Vec eq = S::detail::weights({INFINITY, 1.0});
    check(eq[0] == 0.5 && eq[1] == 0.5, "weights that do not sum to a finite number: all equal (SHADE-3)");
}

void lpsr()
{
    // D = 2: N_init = 36, MAX_NFE = 20000; the sizes follow (10) after every generation
    L::Options o;
    std::mt19937 rng(5);
    L::LSHADE de(Vec(2, -5.0), Vec(2, 5.0), o, rng);
    bool sizes = de.np() == 36, archive = true;
    long long nfe = 0;
    while (nfe < 20000) {
        const auto X = de.ask();
        Vec f(X.size());
        for (std::size_t i = 0; i < X.size(); ++i) f[i] = X[i][0] * X[i][0] + X[i][1] * X[i][1];
        de.tell(f);
        nfe += static_cast<long long>(X.size());
        const double want = std::round((4.0 - 36.0) / 20000.0 * static_cast<double>(nfe) + 36.0);
        sizes = sizes && de.np() == static_cast<std::size_t>(std::max(4.0, want)) && de.nfe() == nfe;
        archive = archive && de.archive().size() <= static_cast<std::size_t>(std::llround(2.6 * double(de.np())));
    }
    check(sizes && de.np() == 4, "LPSR: N after each generation is round((4 - N_init) / MAX_NFE * NFE + N_init) (10)");
    check(archive, "the archive holds at most round(2.6 N) points");
    mix(de.values()[de.best()]);

    for (int id : {1, 9}) {
        L::Options q;
        std::mt19937 r2(40u + static_cast<unsigned>(id));
        L::LSHADE ls(Vec(10, -yao1999::range(id)), Vec(10, yao1999::range(id)), q, r2);
        double best = INFINITY;
        while (ls.nfe() < 100000) {
            const auto X = ls.ask();
            Vec f(X.size());
            for (std::size_t i = 0; i < X.size(); ++i) {
                f[i] = yao1999::f(id, X[i]);
                best = std::min(best, f[i]);
            }
            ls.tell(f);
        }
        mix(best);
        char line[160];
        std::snprintf(line, sizeof line, "L-SHADE f%d, D = 10, 100000 evaluations: best %.2e below 1e-8", id, best);
        check(best < 1e-8, line);
    }
}

}   // namespace

int main()
{
    table_iii();
    means();
    lpsr();
    std::printf("checksum %016llx\n", static_cast<unsigned long long>(g_hash));
    check(g_hash == 0xa5351d765c84ea86ull, "the checksum of the runs, the same bits on every platform");
    std::printf(g_failed ? "FAILED\n" : "all passed\n");
    return g_failed ? 1 : 0;
}
