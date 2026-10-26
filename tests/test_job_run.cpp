// SPDX-License-Identifier: Apache-2.0
// A campaign's jobs run by the C++ engine (include/mootation/campaign/job_run.hpp):
// a small campaign (tests/data/job_sample, tools/refs/gen_job_sample.py) — its
// config read, its jobs listed, each problem built from the exported registry,
// its references read from the cache's entries — gives every job the
// trajectory the Python engine wrote (run/campaign.py's _run_job), record for
// record, "t" left out, and its meta.json and tables (the process's fields
// left out): the cores' runs, the archive, everything evaluated, the
// indicators and the Pareto-set ones, the same on every platform.

#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <mootation/campaign/config.hpp>
#include <mootation/campaign/job_run.hpp>
#include <mootation/campaign/jobs.hpp>
#include <mootation/campaign/registry.hpp>

namespace cg = mootation::campaign;
namespace js = mootation::json;

static int failures = 0;

static std::string read_text(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

int main()
{
    const std::string dir = std::string(MOOTATION_TEST_DATA) + "/job_sample/";
    cg::Registry reg;
    reg.algorithms = {"moead_de", "nsga2"};
    reg.problems = {{"DTLZ2_3D", 3}, {"ZDT1", 2}};
    reg.executable_findable = [](const std::string&) { return true; };
    reg.regex_groups = [](const std::string&, std::string&) { return 0; };
    const cg::ProblemRegistry preg = cg::load_registry(dir + "registry.json");
    cg::JobsRegistry jreg;
    jreg.metric_names = {"igd", "igdp", "hv"};
    for (const auto& [name, p] : preg.problems) jreg.problems[name] = {p.n_obj, p.pop_size, p.n_gen};

    cg::Config cfg = cg::loads(read_text(dir + "campaign.toml"), reg, std::filesystem::path(dir + "campaign.toml"));
    if (!cg::validate(cfg, reg).empty()) {
        std::printf("FAIL: the sample campaign has problems\n");
        return 1;
    }
    const cg::CampaignSpec spec = cg::campaign_spec(cfg, jreg);
    const auto jobs = cg::expand_jobs(cfg, spec, jreg);

    std::ifstream expected(dir + "expected.jsonl", std::ios::binary);
    std::string line;
    std::size_t checked = 0;
    while (std::getline(expected, line)) {
        if (line.empty()) continue;
        const js::Value want = js::parse(line);
        const cg::Job& job = jobs.at(static_cast<std::size_t>(want.find("index")->i));
        const cg::RegistryProblem& p = preg.problems.at(job.problem);
        if (auto why = cg::refusal(job, spec, p, mootation::run::algorithm_names())) {
            std::printf("FAIL %s: refused (%s)\n", job.problem.c_str(), why->c_str());
            ++failures;
            continue;
        }
        const auto entry = cg::load_entry(dir + "refs/v" + std::to_string(preg.reference_version) + "/" +
                                          job.problem + "/n" + std::to_string(spec.n_ref) + ".npz");
        cg::JobRun run(job, spec, p, cg::shared_references(entry, true, spec.dense_reference, p.has_pareto_set));
        run.set_registry(preg, jreg);
        cg::Process still;
        still.seconds = [] { return 0.0; };            // the clock stands still: "t" is 0.0
        cg::JobOutcome out = run.run(still);
        const cg::JobFiles files = run.finish(out);
        for (auto& rec : out.trajectory) {            // the samples have no "t"
            const auto at = rec.find("\"t\":0.0,");
            if (at != std::string::npos) rec.erase(at, 8);
        }
        // meta.json without the process's fields
        js::Value meta = files.meta, kept = js::Value::object();
        for (std::size_t k = 0; k < meta.keys.size(); ++k) {
            static const std::set<std::string> process = {"mootation", "revision", "host", "platform", "python",
                                                          "pid", "fp_env", "started", "finished", "seconds",
                                                          "record_seconds", "timing_us"};
            if (!process.count(meta.keys[k])) kept.set(meta.keys[k], meta.items[k]);
        }
        if (js::dump(kept) != js::dump(*want.find("meta"))) {
            std::printf("FAIL %s %s: meta.json\n  got  %s\n  want %s\n", job.problem.c_str(), job.key().c_str(),
                        js::dump(kept).substr(0, 600).c_str(), js::dump(*want.find("meta")).substr(0, 600).c_str());
            ++failures;
        }
        // the tables: head (a CSV's line end the platform's), shape, the numbers' SHA-256
        const auto& heads = want.find("tables")->items;
        bool tables_same = heads.size() == files.tables.size();
        for (std::size_t k = 0; tables_same && k < heads.size(); ++k) {
            const cg::Table& t = files.tables[k];
            js::Value head = *heads[k].find("head");
            const bool gz = t.name.size() > 3 && t.name.compare(t.name.size() - 3, 3, ".gz") == 0;
#if defined(_WIN32)
            if (!gz) head.set("newline", js::Value::string("\r\n"));
#else
            if (!gz) head.set("newline", js::Value::string("\n"));
#endif
            const auto h = mootation::store::sha256(reinterpret_cast<const std::uint8_t*>(t.values.data()),
                                                    t.values.size() * sizeof(double));
            static const char* hex = "0123456789abcdef";
            std::string digest;
            for (auto c : h) {
                digest += hex[c >> 4];
                digest += hex[c & 15];
            }
            const auto& shape = heads[k].find("shape")->items;
            tables_same = heads[k].find("name")->s == t.name && js::dump(head) == js::dump(t.head) &&
                          static_cast<std::size_t>(shape.at(0).i) == t.rows &&
                          (shape.size() < 2 || static_cast<std::size_t>(shape.at(1).i) == t.cols) &&
                          heads[k].find("sha256")->s == digest;
        }
        if (!tables_same) {
            std::printf("FAIL %s %s: the tables\n", job.problem.c_str(), job.key().c_str());
            ++failures;
        }
        const auto& lines = want.find("lines")->items;
        bool same = out.status == "done" && out.trajectory.size() == lines.size();
        for (std::size_t k = 0; same && k < lines.size(); ++k) same = out.trajectory[k] == lines[k].s;
        if (!same) {
            std::printf("FAIL %s %s: %zu records, %zu expected; status %s %s\n", job.problem.c_str(),
                        job.key().c_str(), out.trajectory.size(), lines.size(), out.status.c_str(), out.error.c_str());
            for (std::size_t k = 0; k < lines.size() && k < out.trajectory.size(); ++k)
                if (out.trajectory[k] != lines[k].s) {
                    std::printf("  got  %s\n  want %s\n", out.trajectory[k].c_str(), lines[k].s.c_str());
                    break;
                }
            ++failures;
        }
        ++checked;
    }
    if (checked != jobs.size()) {
        std::printf("FAIL: %zu of %zu jobs checked\n", checked, jobs.size());
        ++failures;
    }
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("job run: the Python engine's trajectories\n");
    return 0;
}
