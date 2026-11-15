// SPDX-License-Identifier: Apache-2.0
// A job's record in the store, made by the C++ engine (include/mootation/campaign/
// run_store.hpp): the sample campaign's jobs (tests/data/job_sample) run, their
// blocks — "#format", meta.json, the trajectory column by column, the tables —
// encoded and read back the same, block for block; the trajectory's lines come
// back from its columns; a line the columns cannot give back stays text; the
// records appended to a segment are found again whole, by their index.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <mootation/campaign/config.hpp>
#include <mootation/campaign/engine.hpp>
#include <mootation/campaign/job_run.hpp>
#include <mootation/campaign/jobs.hpp>
#include <mootation/campaign/registry.hpp>
#include <mootation/campaign/run_store.hpp>

namespace cg = mootation::campaign;
namespace st = mootation::store;

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

static std::string read_text(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

static bool same_blocks(const std::vector<st::Block>& a, const std::vector<st::Block>& b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t k = 0; k < a.size(); ++k)
        if (a[k].name != b[k].name || a[k].dtype != b[k].dtype || a[k].shape != b[k].shape || a[k].data != b[k].data)
            return false;
    return true;
}

int main()
{
    const std::string dir = std::string(MOOTATION_TEST_DATA) + "/job_sample/";
    // the config and the jobs checked against the sample's exported registry,
    // its problems and lists, as the engine checks them (engine.hpp)
    const cg::ProblemRegistry preg = cg::load_registry(dir + "registry.json");
    const cg::Registry reg = cg::config_registry(preg);
    const cg::JobsRegistry jreg = cg::jobs_registry(preg);
    cg::Config cfg = cg::loads(read_text(dir + "campaign.toml"), reg, std::filesystem::path(dir + "campaign.toml"));
    CHECK(cg::validate(cfg, reg).empty());
    const cg::CampaignSpec spec = cg::campaign_spec(cfg, jreg);
    const auto jobs = cg::expand_jobs(cfg, spec, jreg);

    const std::filesystem::path store =
        std::filesystem::temp_directory_path() / ("mootation_test_run_store_" + std::to_string(std::rand()));
    std::filesystem::remove_all(store);
    cg::SegmentWriter writer(store, 1 << 20, 4242, "host",
                             {{"code_stamp", preg.code_stamp}, {"trajectory_version", "2"}, {"detmath_path", "detmath"}});
    std::vector<std::pair<std::string, std::vector<st::Block>>> written;
    for (const auto& job : jobs) {
        const cg::RegistryProblem& p = preg.problems.at(job.problem);
        // the entry where refs.entry_path() puts it under the sample's refs/ (references.hpp:
        // records, .mrf, from reference version 5; .npz before)
        const auto entry = cg::load_entry(std::filesystem::path(dir) / "refs" /
                                          ("v" + std::to_string(preg.reference_version)) /
                                          mootation::store::file::path_of(job.problem) /
                                          ("n" + std::to_string(spec.n_ref) +
                                           (preg.reference_version >= cg::kRecordEntries ? ".mrf" : ".npz")));
        cg::JobRun run(job, spec, p, cg::shared_references(entry, true, spec.dense_reference, p.has_pareto_set));
        run.set_registry(preg, jreg);
        const cg::JobOutcome out = run.run();
        const cg::JobFiles files = run.finish(out);
        const auto blocks = cg::record_blocks(files, out.trajectory);
        // "#format", meta.json, the trajectory's head, cells and values, then two blocks a table
        CHECK(blocks.size() == 5 + 2 * files.tables.size());
        CHECK(blocks[0].name == "#format" && blocks[1].name == "meta.json" && blocks[2].name == "trajectory.jsonl#head");
        const auto body = st::encode_run(blocks);
        CHECK(same_blocks(st::decode_run(body.data(), body.size()), blocks));
        const std::string key = cg::job_key(job);
        writer.append(key, blocks);
        written.emplace_back(key, blocks);
    }

    // a line the columns cannot give back (a float written as an integer would) stays text
    CHECK(!cg::trajectory_blocks("t.jsonl", {"{\"a\":1.0}", "{\"a\":2}"}).has_value() ||
          cg::trajectory_blocks("t.jsonl", {"{\"a\":1.0}", "{\"a\":2}"})->size() == 3);
    CHECK(!cg::trajectory_blocks("t.jsonl", {"{\"a\":[1,2]}"}).has_value());
    const auto plain = cg::trajectory_blocks("t.jsonl", {"{\"gen\":0,\"x\":null}", "{\"gen\":1,\"y\":\"w\",\"x\":0.5}"});
    CHECK(plain.has_value() && (*plain)[1].shape == std::vector<std::uint64_t>({2, 3}));

    // every record found again whole, in the order appended, by its index
    std::size_t found = 0;
    for (const auto& f : std::filesystem::directory_iterator(store)) {
        if (f.path().extension() != ".raw") continue;
        const std::string raw = read_text(f.path().string());
        const auto* buf = reinterpret_cast<const std::uint8_t*>(raw.data());
        std::filesystem::path idx = f.path();
        idx.replace_extension(".idx");
        for (const auto& e : st::index_entries(read_text(idx.string()))) {
            const st::RecordAt r = st::parse_record(buf, raw.size(), static_cast<std::size_t>(e.offset));
            CHECK(r.ok && r.key == e.key && found < written.size() && e.key == written[found].first);
            if (r.ok && found < written.size())
                CHECK(same_blocks(st::decode_run(buf + r.body, r.body_size), written[found].second));
            CHECK(e.status == "done");
            ++found;
        }
        std::filesystem::path mhd = f.path();
        mhd.replace_extension(".mhd");
        const std::string head = read_text(mhd.string());
        CHECK(head.rfind("format = mootation-store\nformat_version = " + std::to_string(cg::kStoreFormatVersion) +
                             "\nkind = segment\n",
                         0) == 0);
        CHECK(head.find("\ncode_stamp = " + preg.code_stamp + "\n") != std::string::npos);
    }
    CHECK(found == jobs.size());
    std::filesystem::remove_all(store);

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("run store: the engine's records, written and read back\n");
    return 0;
}
