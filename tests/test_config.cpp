// SPDX-License-Identifier: Apache-2.0
// A run description read and checked in C++ (include/mootation/campaign/config.hpp):
// the error or the list of problems that run/config.py's loads() and validate()
// give, word for word. The expected texts are Python's (Python 3.13); the
// registry here is a small stand-in holding what these cases look up.

#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include <mootation/campaign/config.hpp>

namespace cg = mootation::campaign;

static int failures = 0;

static const char* EXT = R"([run]
name = "t"
scratch = "s/{worker:02d}"
ledger = "l.jsonl"

[problem]
kind = "external"
n_vars = 2
n_objs = 2
bounds = [[0.0, 1.0], [0.0, 1.0]]

[[problem.steps]]
name = "solve"
run = ["python", "-c", "pass"]

[problem.output]
parser = "csv"
from = "out.csv"
objectives = ["f1", "f2"]

[[algorithms]]
name = "nsga2"
pop = 20
gens = 5
)";

static const char* BUILTIN = R"([run]
name = "b"

[problem]
kind = "builtin"

[benchmarks]
problems = ["ZDT1", "DTLZ2_3D"]
runs = 2

[[algorithms]]
name = "nsga3"
pop = 92
gens = 5
)";

static std::string replaced(std::string text, const std::string& from, const std::string& to)
{
    const auto at = text.find(from);
    if (at == std::string::npos) {
        std::printf("FAIL: '%s' not in the case\n", from.c_str());
        ++failures;
        return text;
    }
    return text.replace(at, from.size(), to);
}

static cg::Registry registry()
{
    cg::Registry reg;
    reg.algorithms = {"moead_de", "nsga2", "nsga3"};
    reg.problems = {{"DTLZ2_3D", 3}, {"ZDT1", 2}};
    reg.executable_findable = [](const std::string& argv0) { return argv0 == "python"; };
    reg.regex_groups = [](const std::string&, std::string&) { return 0; };
    return reg;
}

// the error loads() raises, or validate()'s problems and the benchmark problems
static void same(const char* label, const std::string& text, const std::string& error,
                 const std::vector<std::string>& problems = {}, const std::vector<std::string>& benchmark = {})
{
    const cg::Registry reg = registry();
    std::string got_error;
    std::vector<std::string> got_problems, got_benchmark;
    try {
        cg::Config cfg = cg::loads(text, reg, std::filesystem::path("cases/x.toml"));
        got_problems = cg::validate(cfg, reg);
        got_benchmark = cfg.benchmark_problems;
    } catch (const cg::ConfigError& e) {
        got_error = e.what();
    }
    const bool ok = !error.empty() && error.back() == '*' ? got_error.rfind(error.substr(0, error.size() - 1), 0) == 0
                                        : got_error == error;
    if (!ok || got_problems != problems || got_benchmark != benchmark) {
        std::printf("FAIL %s: error '%s'\n", label, got_error.c_str());
        for (const auto& p : got_problems) std::printf("  problem: %s\n", p.c_str());
        ++failures;
    }
}

int main()
{
    same("a clean external problem", EXT, "");
    same("true for a count", replaced(EXT, "n_objs = 2", "n_objs = true"), "problem.n_objs: expected int, got bool");
    same("a step's timeout as a word",
         replaced(EXT, "run = [\"python\", \"-c\", \"pass\"]", "run = [\"python\", \"-c\", \"pass\"]\ntimeout = \"zz\""),
         "problem.steps[0].timeout: expected int/float, got str");
    same("a column given as a word", replaced(EXT, "parser = \"csv\"", "parser = \"columns\"\nfields = [0, \"1\"]"),
         "problem.output.fields: expected a list of integers, got [0, '1']");
    // one objective: every lattice is one point; the search for a larger one used to loop for ever
    same("one objective",
         replaced(replaced(replaced(EXT, "n_objs = 2", "n_objs = 1"), "objectives = [\"f1\", \"f2\"]",
                           "objectives = [\"f1\"]"),
                  "name = \"nsga2\"", "name = \"moead_de\""),
         "",
         {"algorithms[0] (moead_de): pop = 20 is not a Das-Dennis lattice size for n_objs = 1; moead_de requires "
          "an exact lattice. Nearest attainable: 1"});
    same("the benchmarks", BUILTIN, "",
         {"algorithms[0] (nsga3): for the 3-objective problems (DTLZ2_3D): pop = 92 is not a Das-Dennis lattice size "
          "for n_objs = 3; nsga3 requires an exact lattice. Nearest attainable: 91, 105"},
         {"ZDT1", "DTLZ2_3D"});
    same("problems as a number", replaced(BUILTIN, "problems = [\"ZDT1\", \"DTLZ2_3D\"]", "problems = 3"), "",
         {"benchmarks.problems: expected a list of problem names, got 3"});
    same("families as a word", replaced(BUILTIN, "problems = [\"ZDT1\", \"DTLZ2_3D\"]", "families = \"DTLZ\"\nobjectives = [3]"),
         "", {"benchmarks: 'families' and 'objectives' must be lists"});
    same("not TOML", "[run]\nname = = 1\n", "TOML does not parse: *");

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("config: run/config.py's verdicts\n");
    return 0;
}
