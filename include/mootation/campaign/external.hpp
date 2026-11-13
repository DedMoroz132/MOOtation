#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// External problems (task 24, §24.7.1): a problem in C++ that lives outside
// this repository — a private folder of its own — run by the campaign engine
// as a problem of the registry, at one objective.
//
// THE FOLDER (the build's MOOTATION_EXTERNAL_PROBLEMS=<folder>; engine/
// CMakeLists.txt, docs/running.md) holds
//   * its sources, *.cpp, compiled into the engine with the same compiler
//     and flags as the library (no contraction into fused multiply-adds:
//     a seed gives the same bits as for a problem of the suite);
//   * the registration below, defined in one of them;
//   * problems.toml, the problems' rows (external_rows.hpp): name, family,
//     size, box, f_opt, start, the family's parameters.
// Nothing of it enters the repository: the registry (data/registry.json),
// suite/ and the Python modules stay as they are.
//
// THE REGISTRATION: mootation_external_problems(reg) adds each family once,
// reg.add(family, make, version): make(params, n_var) builds the problem of a
// row — the row's [problem.params] table as a JSON tree (EXT-1) and its
// n_var — as a suite::Native (native.hpp): evaluate, and where it has them
// residuals and their Jacobian, gradient, canon and blocks, gauge directions.
// The single-objective methods and strategies take it as they take a
// problem of the suite. version: the family's code version, written into
// every run's meta.json (external: family, version, the row's CRC-32C,
// the folder's sources' SHA-256 — engine/external_stamp.cmake): what a
// result was made of, traceable without publishing the problem.
//
// The problem is const and without state (native.hpp): one object serves
// every thread of a strategy. A random stream it needs is drawn from rnd::
// with a seed of its row, never from the clock.
// ============================================================================

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "../json.hpp"
#include "../suite/native.hpp"

namespace mootation::campaign {

using ExternalMake = std::function<std::shared_ptr<suite::Native>(const json::Value& params, int n_var)>;

struct ExternalFamily {
    ExternalMake make;
    std::string version;
};

class ExternalProblems {
public:
    void add(const std::string& family, ExternalMake make, const std::string& version = "")
    {
        if (family.empty()) throw std::invalid_argument("external problems: a family without a name");
        if (!make) throw std::invalid_argument("external problems: " + family + ": no function makes its problems");
        if (families_.count(family))
            throw std::invalid_argument("external problems: the family '" + family + "' is registered twice");
        families_[family] = {std::move(make), version};
    }

    // the family, or nullptr
    const ExternalFamily* find(const std::string& family) const
    {
        const auto it = families_.find(family);
        return it == families_.end() ? nullptr : &it->second;
    }

    const std::map<std::string, ExternalFamily>& families() const { return families_; }

private:
    std::map<std::string, ExternalFamily> families_;
};

// The one refusal of a campaign that names an external problem where it
// cannot run: the Python engine's config check (run/config.py,
// external_refusal: the same text), and the C++ engine's where a job of it
// would be left to the Python engine (engine.hpp).
inline std::string external_refusal(const std::string& name)
{
    return "external problem '" + name +
           "': the C++ engine alone runs it (mootation run <config>; task 24, section 24.7.1), "
           "the Python engine does not know it";
}

}   // namespace mootation::campaign

// Defined by the external folder, called by the engine where the build has
// one (MOOTATION_EXTERNAL_PROBLEMS_DIR).
void mootation_external_problems(mootation::campaign::ExternalProblems& reg);
