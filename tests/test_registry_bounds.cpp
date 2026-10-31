// SPDX-License-Identifier: Apache-2.0
// The box of every problem of the repository's registry (data/registry.json)
// from its definition in C++ (registry.hpp's definition_bounds, suite::bounds;
// task 17, supplement 7, section 2.4) is the file's, bit for bit: the file's
// numbers are for this comparison only. Every problem has its box from C++,
// none keeps the file's for want of one.

#include <cstdio>
#include <string>

#include <mootation/campaign/registry.hpp>

namespace cg = mootation::campaign;

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

int main()
{
    const cg::ProblemRegistry reg = cg::load_registry(std::string(MOOTATION_REPO_DATA) + "/registry.json");
    CHECK(reg.problems.size() >= 1366);
    std::size_t same = 0, shown = 0;
    for (const auto& [name, p] : reg.problems) {
        // from C++: a problem of the suite, a copy of one, or a probe
        CHECK(p.spec || p.noise || p.failing_region || p.uninformative);
        CHECK(static_cast<int>(p.bounds.size()) == p.n_var);
        if (p.bounds == p.file_bounds) {
            ++same;
            continue;
        }
        ++failures;
        if (shown++ < 10)
            for (std::size_t i = 0; i < p.bounds.size() && i < p.file_bounds.size(); ++i)
                if (p.bounds[i] != p.file_bounds[i]) {
                    std::printf("FAIL %s x%zu: C++ [%.17g, %.17g], file [%.17g, %.17g]\n", name.c_str(), i + 1,
                                p.bounds[i].first, p.bounds[i].second, p.file_bounds[i].first,
                                p.file_bounds[i].second);
                    break;
                }
    }
    std::printf("%zu of %zu problems: the box from C++ is the file's\n", same, reg.problems.size());
    if (failures == 0) std::printf("test_registry_bounds: OK\n");
    return failures == 0 ? 0 : 1;
}
