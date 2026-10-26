// SPDX-License-Identifier: Apache-2.0
// A campaign run by the C++ engine (include/mootation/campaign/engine.hpp) on the
// sample campaign (tests/data/job_sample) written to a store: four threads run
// its jobs and their records are the bytes one thread writes; run again, it
// runs nothing (the store says every job is done); a record cut off in the
// middle is cut away by the repair and the jobs before it stay done; a slice
// of the jobs (shard_of) runs only its jobs.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <mootation/campaign/engine.hpp>

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

static std::string read_text(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

// key -> the record's blocks, every record of a store's segments, meta.json
// without the fields that are the process's (timing_us: the run's seconds)
static std::map<std::string, std::string> bodies(const std::filesystem::path& store)
{
    std::map<std::string, std::string> out;
    for (const auto& f : std::filesystem::directory_iterator(store)) {
        if (f.path().extension() != ".raw") continue;
        const std::string raw = read_text(f.path());
        const auto* buf = reinterpret_cast<const std::uint8_t*>(raw.data());
        for (std::size_t at = 0; at < raw.size();) {
            const st::RecordAt r = st::parse_record(buf, raw.size(), at);
            if (!r.ok) break;
            std::string text;
            for (const auto& b : st::decode_run(buf + r.body, r.body_size)) {
                std::string data(b.data.begin(), b.data.end());
                if (b.name == "meta.json") {
                    mootation::json::Value meta = mootation::json::parse(data), kept = mootation::json::Value::object();
                    for (std::size_t k = 0; k < meta.keys.size(); ++k)
                        if (meta.keys[k] != "timing_us") kept.set(meta.keys[k], meta.items[k]);
                    data = mootation::json::dump(kept);
                }
                text += b.name + "|" + b.dtype + "|" + data + "\n";
            }
            out[r.key] = text;
            at = r.end;
        }
    }
    return out;
}

static void set_env(const char* name, const std::string& value)
{
#if defined(_WIN32)
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

static int body();

int main()
{
    try {
        return body();
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}

static int body()
{
    const std::string data = std::string(MOOTATION_TEST_DATA) + "/job_sample/";
    set_env("MOOTATION_REFS_DIR", data + "refs");
    const cg::ProblemRegistry preg = cg::load_registry(data + "registry.json");
    std::string text = read_text(data + "campaign.toml");
    text.replace(text.find("[campaign]\n"), 11, "[campaign]\nstorage = \"store\"\n");
    const auto tmp = std::filesystem::temp_directory_path() / ("mootation_test_engine_" + std::to_string(std::rand()));
    std::filesystem::remove_all(tmp);

    auto run = [&](const std::string& dir, int workers, std::optional<std::pair<std::int64_t, std::int64_t>> shard = {}) {
        std::filesystem::create_directories(tmp / dir);
        const auto source = tmp / dir / "campaign.toml";
        std::ofstream(source, std::ios::binary) << text;
        cg::EngineOptions opt;
        opt.workers = workers;
        opt.shard = shard;
        opt.pid = 7;
        opt.host = "test";
        return cg::run_campaign(text, source, preg, opt);
    };

    // four threads, then one: the same records
    const cg::EngineCounts four = run("four", 4);
    CHECK(four.total == 4 && four.done == 4 && four.failed == 0 && four.skipped == 0 && four.left.empty());
    const cg::EngineCounts one = run("one", 1);
    CHECK(one.done == 4);
    const auto store4 = tmp / "four" / "results" / "sample" / "store";
    const auto a = bodies(store4), b = bodies(tmp / "one" / "results" / "sample" / "store");
    CHECK(a.size() == 4 && a == b);
    CHECK(std::filesystem::is_regular_file(tmp / "four" / "results" / "sample" / "campaign.json"));

    // run again: every job done
    const cg::EngineCounts again = run("four", 4);
    CHECK(again.done == 0 && again.skipped == 4);

    // a record cut off in the middle (a writer stopped): the repair cuts it
    // away, the jobs written before it stay done, its job runs again
    std::filesystem::path seg;
    for (const auto& f : std::filesystem::directory_iterator(store4))
        if (f.path().extension() == ".raw") seg = f.path();
    const auto size = std::filesystem::file_size(seg);
    {
        const std::string raw = read_text(seg);
        std::ofstream out(seg, std::ios::binary | std::ios::app);
        out << raw.substr(0, raw.size() / 2 > 100 ? 100 : raw.size() / 2);     // the head of a record, no more
    }
    CHECK(std::filesystem::file_size(seg) > size);
    const auto repaired = cg::repair_store(store4);
    CHECK(repaired.size() == 1 && std::filesystem::file_size(seg) == size);
    CHECK(run("four", 2).done == 0);

    // a slice: its jobs only
    std::size_t sliced = 0;
    for (std::int64_t i = 0; i < 3; ++i) sliced += run("slices", 1, std::make_pair(i, std::int64_t{3})).done;
    CHECK(sliced == 4);
    CHECK(bodies(tmp / "slices" / "results" / "sample" / "store") == a);

    std::filesystem::remove_all(tmp);
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("engine: a campaign's records by a pool of threads, resumed and repaired\n");
    return 0;
}
