// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Settings::validate() at n_objs = 1 (task 18, supplement 1, §5): the 22 cores
// that crash or hang with one objective are refused with a message that names
// the core and its reason; every other core of algorithms.def passes, and at
// n_objs = 2 all of them pass (the multi-objective path is untouched).
// mootation::run() validates before it builds a core, so the refusal comes
// before any evaluation.
// ============================================================================

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/settings.hpp>

namespace {

int g_failed = 0;

void check(bool ok, const std::string& what)
{
    if (!ok) ++g_failed;
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what.c_str());
}

const std::vector<std::string>& all_algorithms()
{
    static const std::vector<std::string> names = {
#define MOOTATION_ALG(key, IND, CORE) #key,
#include <mootation/algorithms.def>
#undef MOOTATION_ALG
    };
    return names;
}

// "" when validate() passes, else its message
std::string refusal(const std::string& algorithm, int n_objs)
{
    mootation::Settings s;
    s.algorithm = algorithm;
    s.n_objs = n_objs;
    s.set_box(3, 0.0, 1.0);
    try {
        s.validate();
    } catch (const std::invalid_argument& e) {
        return e.what();
    }
    return "";
}

}   // namespace

int main()
{
    const std::vector<std::string> refused = {
        "nsga3", "a_nsga3", "theta_dea", "moead", "moead_de", "moead_dra", "moead_dd", "moead_awa",
        "adaw", "mombi2", "irea", "edv", "rvea", "rvea_star", "mbra", "srv", "srv_nsga3", "ctaea",
        "rd_emo", "r2ibea", "if_maoea", "r2_emoa"};
    check(refused.size() == 22, "22 cores on the list");
    int listed = 0;
    for (const std::string& a : all_algorithms()) {
        bool on_list = false;
        for (const std::string& r : refused) on_list = on_list || r == a;
        listed += on_list;
        const std::string m1 = refusal(a, 1), m2 = refusal(a, 2);
        if (on_list) {
            const bool named = m1.find(a + " needs n_objs >= 2: ") != std::string::npos &&
                               m1.size() > a.size() + 40;
            if (!named) std::printf("      %s: '%s'\n", a.c_str(), m1.c_str());
            check(named, a + " at n_objs = 1: refused, the core and its reason in the message");
        } else {
            check(m1.empty(), a + " at n_objs = 1: passes validate()");
        }
        check(m2.empty(), a + " at n_objs = 2: passes validate()");
    }
    check(listed == 22, "all 22 are cores of algorithms.def");
    const std::string lattice = refusal("nsga3", 1), loop = refusal("if_maoea", 1);
    check(lattice.find("simplex lattice") != std::string::npos && loop.find("loop does not end") != std::string::npos,
          "the reasons: the lattice of one point; the loop that does not end");
    std::printf(g_failed ? "FAILED\n" : "all passed\n");
    return g_failed ? 1 : 0;
}
