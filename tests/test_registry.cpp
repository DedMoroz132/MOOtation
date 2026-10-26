// SPDX-License-Identifier: Apache-2.0
// The problem registry read in C++ (include/mootation/campaign/registry.hpp) on
// a few problems of the exported registry (tests/data/registry_sample.json,
// tools/refs/gen_registry_sample.py): each problem of the C++ suite built from
// its exported spec gives, at the file's point, the objectives and
// constraints Python's registry gave, bit for bit — a shift's centre handed
// over, C-DTLZ's constants, a rotation and a bbob instance drawn here — and
// the engine refuses the problem it cannot run without Python.

#include <cstdio>
#include <string>
#include <vector>

#include <mootation/campaign/registry.hpp>

namespace cg = mootation::campaign;
namespace js = mootation::json;

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

static std::vector<double> numbers(const js::Value& v)
{
    std::vector<double> out;
    for (const auto& x : v.items) out.push_back(x.kind == js::Value::Kind::Float ? x.f : static_cast<double>(x.i));
    return out;
}

int main()
{
    const std::string path = std::string(MOOTATION_TEST_DATA) + "/registry_sample.json";
    const cg::ProblemRegistry reg = cg::load_registry(path);
    CHECK(reg.problems.size() == 5);
    CHECK(!reg.code_stamp.empty() && reg.reference_version >= 1);

    // the values the file records, from the problems built here
    const auto doc = js::parse([&] {
        const auto b = mootation::npz::read_file(path);
        return std::string(b.begin(), b.end());
    }());
    int checked = 0;
    for (const auto& item : doc.find("problems")->items) {
        const js::Value* check = item.find("check");
        if (!check) continue;
        const cg::RegistryProblem& p = reg.problems.at(item.find("name")->s);
        const auto native = cg::make_native(p);
        const auto x = numbers(*check->find("x"));
        std::vector<double> f(static_cast<std::size_t>(native->n_obj)), g(static_cast<std::size_t>(native->n_con));
        native->evaluate(x.data(), f.data(), g.empty() ? nullptr : g.data());
        if (f != numbers(*check->find("F")) || g != numbers(*check->find("G"))) {
            std::printf("FAIL %s: other values\n", p.name.c_str());
            ++failures;
        }
        CHECK(static_cast<int>(p.bounds.size()) == p.n_var && native->n_var == p.n_var);
        ++checked;
    }
    CHECK(checked == 4);

    CHECK(!cg::engine_refusal(reg.problems.at("shiftDTLZ2_3D")));
    CHECK(reg.problems.at("C2-DTLZ2_3D").has_cons && reg.problems.at("C2-DTLZ2_3D").n_con == 1);
    const auto why = cg::engine_refusal(reg.problems.at("catZDT1"));
    CHECK(why && !why->empty());
    CHECK(reg.problems.at("rDTLZ2_3D").modification.kind == js::Value::Kind::Object);

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("registry: the exported problems' values\n");
    return 0;
}
