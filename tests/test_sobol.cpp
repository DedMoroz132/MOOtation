// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// sobol.hpp: the Sobol sequence of Joe & Kuo's direction numbers, against
// SciPy's unscrambled qmc.Sobol (tests/data/scipy-1.18.1_sobol.txt: its
// points 1..40 in 100 dimensions). Every value is a dyadic fraction, so the
// comparison is exact; and the lower dimensions are the prefix of the higher.
// ============================================================================

#include <cstdint>
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

    // task 24, §24.3.2: all 21 201 dimensions of the file (sobol_joe_kuo.hpp
    // beyond table()'s 100), against SciPy's points 1..40 in dimensions
    // 1..300 and the FNV-1a of its points 1..40 in all 21 201
    // (tests/data/scipy-1.18.1_sobol_21201.txt, tools/refs/gen_sobol_ref.py)
    std::ifstream wide(std::string(MOOTATION_TEST_DATA) + "/scipy-1.18.1_sobol_21201.txt");
    check(static_cast<bool>(wide), "the wide reference file opens");
    std::vector<std::vector<double>> wref;
    std::string fnv;
    for (std::string line; std::getline(wide, line);) {
        if (line.rfind("# fnv1a64 ", 0) == 0) fnv = line.substr(10, 16);
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        std::vector<double> p;
        for (double v; row >> v;) p.push_back(v);
        wref.push_back(p);
    }
    check(wref.size() == 40 && wref.front().size() == 300 && fnv.size() == 16, "40 points of 300 dimensions, a hash");
    mootation::sobol::Generator w(300), w100(100);
    bool same300 = true, prefix100 = true;
    for (const auto& r : wref) {
        const auto p = w.next(), q = w100.next();
        for (std::size_t j = 0; j < r.size(); ++j) same300 = same300 && p[j] == r[j];
        for (std::size_t j = 0; j < q.size(); ++j) prefix100 = prefix100 && p[j] == q[j];
    }
    check(same300, "300 dimensions: points 1..40 equal SciPy's, bit for bit");
    check(prefix100, "the first 100 of 300 dimensions are the 100 dimensions' points");
    mootation::sobol::Generator all(mootation::sobol::max_dims_joe_kuo);
    std::uint64_t h = 0xCBF29CE484222325ull;
    for (int i = 0; i < 40; ++i)
        for (double v : all.next()) {
            const auto x = static_cast<std::uint32_t>(v * 4294967296.0);
            for (int k = 0; k < 4; ++k) {
                h ^= (x >> (8 * k)) & 0xFFu;
                h *= 0x100000001B3ull;
            }
        }
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(h));
    check(mootation::sobol::max_dims_joe_kuo == 21201 && fnv == hex,
          "21 201 dimensions: points 1..40 hash as SciPy's");
    bool refused = false;
    try {
        mootation::sobol::directions(21202);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    check(refused && mootation::sobol::max_dims == 100, "21 202 dimensions refused; max_dims stays 100 (multiglods)");
    std::printf("sobol: %d/%d checks passed%s\n", checks - failed, checks, failed ? ", FAILED" : "");
    return failed == 0 ? 0 : 1;
}
