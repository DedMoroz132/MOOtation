// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// sobol.hpp: the Sobol sequence of Joe & Kuo's direction numbers, against
// SciPy's unscrambled qmc.Sobol (tests/data/scipy-1.18.1_sobol.txt: its
// points 1..40 in 100 dimensions). Every value is a dyadic fraction, so the
// comparison is exact; and the lower dimensions are the prefix of the higher.
// ============================================================================

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <mootation/sobol.hpp>

int main()
{
    int checks = 0, failed = 0;
    auto check = [&](bool ok, const char* what) {
        ++checks;
        if (!ok) { ++failed; std::printf("  FAIL  %s\n", what); }
    };
    std::ifstream in(std::string(MOOTATION_TEST_DATA) + "/scipy-1.18.1_sobol.txt");
    check(static_cast<bool>(in), "the reference file opens");
    std::vector<std::vector<double>> ref;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        std::vector<double> p;
        for (double v; row >> v;) p.push_back(v);
        ref.push_back(p);
    }
    check(ref.size() == 40 && ref.front().size() == 100, "40 points of 100 dimensions");
    mootation::sobol::Generator g(100);
    bool same = true;
    for (const auto& r : ref) {
        const auto p = g.next();
        for (std::size_t j = 0; j < r.size(); ++j) same = same && p[j] == r[j];
    }
    check(same, "points 1..40 equal SciPy's, bit for bit");
    mootation::sobol::Generator a(100), b(7);
    bool prefix = true;
    for (int i = 0; i < 40; ++i) {
        const auto pa = a.next(), pb = b.next();
        for (std::size_t j = 0; j < pb.size(); ++j) prefix = prefix && pa[j] == pb[j];
    }
    check(prefix, "7 dimensions are the prefix of 100");
    std::printf("sobol: %d/%d checks passed%s\n", checks - failed, checks, failed ? ", FAILED" : "");
    return failed == 0 ? 0 : 1;
}
