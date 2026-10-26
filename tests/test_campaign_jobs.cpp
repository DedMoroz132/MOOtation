// SPDX-License-Identifier: Apache-2.0
// A campaign's job list made in C++ (include/mootation/campaign/jobs.hpp): the
// jobs run/campaign.py's expand_jobs() lists for the same config — a fitted
// population, an algorithm's own budget, a budget ladder's rungs — and the
// part of 7 shard_of() puts each in; [campaign] errors word for word. The
// expected values are Python's (Python 3.13).

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <mootation/campaign/jobs.hpp>

namespace cg = mootation::campaign;

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

static const char* CAMPAIGN = R"([run]
name = "j"

[problem]
kind = "builtin"

[benchmarks]
problems = ["ZDT1", "DTLZ2_3D"]
runs = 2

[campaign]
budget_fe = 1000
pop_fit = "nearest"
ladder = [300, 5000]

[[algorithms]]
name = "moead_de"
pop = 100
gens = 1

[[algorithms]]
name = "sms_m2m"
pop = 104
gens = 1
params = { K = 7 }

[[algorithms]]
name = "nsga2"
pop = 30
gens = 2
evaluations = 95

[[algorithms]]
name = "smpso"
pop = 50
gens = 1
params = { turbulence = "thirds" }
)";

struct Want {
    std::int64_t index;
    const char* problem;
    const char* key;
    std::int64_t seed, pop, gens, budget;
    const char* note;
    const char* ladder_of;
    std::int64_t shard;
};

static const Want WANT[] = {
    {0, "ZDT1", "moead_de", 1, 100, 10, 1000, "", "", 4},
    {1, "ZDT1", "moead_de", 2, 100, 10, 1000, "", "", 4},
    {2, "ZDT1", "sms_m2m", 1, 105, 10, 1050, "pop 104 -> 105 (multiple of K=7)", "", 0},
    {3, "ZDT1", "sms_m2m", 2, 105, 10, 1050, "pop 104 -> 105 (multiple of K=7)", "", 1},
    {4, "ZDT1", "nsga2", 1, 30, 4, 95, "", "", 1},
    {5, "ZDT1", "nsga2", 2, 30, 4, 95, "", "", 4},
    {6, "ZDT1", "smpso", 1, 50, 20, 1000, "", "", 0},
    {7, "ZDT1", "smpso", 2, 50, 20, 1000, "", "", 5},
    {8, "DTLZ2_3D", "moead_de", 1, 100, 10, 1000, "", "", 2},
    {9, "DTLZ2_3D", "moead_de", 2, 100, 10, 1000, "", "", 3},
    {10, "DTLZ2_3D", "sms_m2m", 1, 105, 10, 1050, "pop 104 -> 105 (multiple of K=7)", "", 0},
    {11, "DTLZ2_3D", "sms_m2m", 2, 105, 10, 1050, "pop 104 -> 105 (multiple of K=7)", "", 2},
    {12, "DTLZ2_3D", "nsga2", 1, 30, 4, 95, "", "", 0},
    {13, "DTLZ2_3D", "nsga2", 2, 30, 4, 95, "", "", 4},
    {14, "DTLZ2_3D", "smpso", 1, 50, 20, 1000, "", "", 6},
    {15, "DTLZ2_3D", "smpso", 2, 50, 20, 1000, "", "", 4},
    {16, "ZDT1", "smpso@300", 1, 50, 6, 300, "", "smpso", 0},
    {17, "ZDT1", "smpso@300", 2, 50, 6, 300, "", "smpso", 0},
    {18, "DTLZ2_3D", "smpso@300", 1, 50, 6, 300, "", "smpso", 6},
    {19, "DTLZ2_3D", "smpso@300", 2, 50, 6, 300, "", "smpso", 3},
};

static cg::Registry registry()
{
    cg::Registry reg;
    reg.algorithms = {"moead_de", "nsga2", "smpso", "sms_m2m"};
    reg.knob_names = {"K", "turbulence"};
    reg.text_knobs = {{"turbulence", {"every_k", "thirds"}}};
    reg.problems = {{"DTLZ2_3D", 3}, {"ZDT1", 2}};
    reg.executable_findable = [](const std::string&) { return true; };
    reg.regex_groups = [](const std::string&, std::string&) { return 0; };
    return reg;
}

static cg::JobsRegistry jobs_registry()
{
    cg::JobsRegistry reg;
    reg.metric_names = {"igd", "igdp", "hv"};
    reg.problems = {{"DTLZ2_3D", {3, 91, 400}}, {"ZDT1", {2, 100, 250}}};
    return reg;
}

static std::string campaign_error(const std::string& table)
{
    const cg::Registry reg = registry();
    std::string text = CAMPAIGN;
    text.replace(text.find("budget_fe = 1000\n"), 17, table);
    cg::Config cfg = cg::loads(text, reg);
    try {
        cg::campaign_spec(cfg, jobs_registry());
    } catch (const cg::ConfigError& e) {
        return e.what();
    }
    return "";
}

int main()
{
    const cg::Registry reg = registry();
    const cg::JobsRegistry jreg = jobs_registry();
    cg::Config cfg = cg::loads(CAMPAIGN, reg);
    const auto problems = cg::validate(cfg, reg);
    for (const auto& p : problems) std::printf("FAIL problem: %s\n", p.c_str());
    CHECK(problems.empty());
    const cg::CampaignSpec spec = cg::campaign_spec(cfg, jreg);
    CHECK(spec.pop_fit == "nearest" && spec.budget_fe == 1000 && spec.ladder.size() == 2);
    const auto jobs = cg::expand_jobs(cfg, spec, jreg);
    CHECK(jobs.size() == sizeof(WANT) / sizeof(WANT[0]));
    for (std::size_t k = 0; k < jobs.size() && k < sizeof(WANT) / sizeof(WANT[0]); ++k) {
        const auto& j = jobs[k];
        const Want& w = WANT[k];
        if (j.index != w.index || j.problem != w.problem || j.key() != w.key || j.seed != w.seed || j.pop != w.pop ||
            j.gens != w.gens || j.budget() != w.budget || j.pop_note != w.note || j.ladder_of != w.ladder_of ||
            cg::shard_of(j, 7) != w.shard) {
            std::printf("FAIL job %zu: %s %s seed %lld pop %lld gens %lld fe %lld '%s' part %lld\n", k,
                        j.problem.c_str(), j.key().c_str(), static_cast<long long>(j.seed),
                        static_cast<long long>(j.pop), static_cast<long long>(j.gens),
                        static_cast<long long>(j.budget()), j.pop_note.c_str(),
                        static_cast<long long>(cg::shard_of(j, 7)));
            ++failures;
        }
    }

    // [campaign] errors, as campaign_spec() words them
    CHECK(campaign_error("budget_fe = 1.5\n") == "campaign.budget_fe: an integer, not 1.5");
    CHECK(campaign_error("budget_fe = true\n") == "campaign.budget_fe: an integer, not True");
    CHECK(campaign_error("metrics = \"igd\"\n") == "campaign.metrics: a list of metric names, not 'igd'");
    CHECK(campaign_error("metrics = [\"igd\", \"nope\"]\n") == "campaign.metrics: unknown metric 'nope'; known: igd, igdp, hv");
    CHECK(campaign_error("seeds = [1, true]\n") == "campaign.seeds: a list of integers, not [1, True]");
    CHECK(campaign_error("record_grid = \"daily\"\n") ==
          "campaign.record_grid: 'budgets', 'generations' or 'log', not 'daily'");
    CHECK(campaign_error("hard_budgets = [100]\n").empty());     // the grid then defaults to "generations"
    CHECK(campaign_error("hard_budgets = [100]\nrecord_grid = \"budgets\"\n") ==
          "campaign.hard_budgets: reads the last record at or below each budget: needs a record every generation "
          "(record_grid = \"generations\", record_every = 1)");
    CHECK(campaign_error("noise_cell = 2\n") == "campaign.noise_cell: a share in (0, 1], not 2");
    CHECK(campaign_error("storage = \"cloud\"\n") == "campaign.storage: \"files\" or \"store\", not 'cloud'");

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("campaign jobs: campaign.py's job list\n");
    return 0;
}
