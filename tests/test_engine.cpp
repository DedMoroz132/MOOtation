// SPDX-License-Identifier: Apache-2.0
// A campaign run by the C++ engine (include/mootation/campaign/engine.hpp) on the
// sample campaign (tests/data/job_sample) written to a store: four threads run
// its jobs and their records are the bytes one thread writes; run again, it
// runs nothing (the store says every job is done); a record cut off in the
// middle is cut away by the repair and the jobs before it stay done, but
// not while another runner's heartbeat says it runs; the heartbeat and
// _workers.txt read as the Python runner's; a slice
// of the jobs (shard_of) runs only its jobs; a run past run_timeout is
// stopped and recorded "timeout"; snapshots and archive checkpoints are
// where meta.json says, in the shapes it says.

#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
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

    auto run_with = [&](const std::string& dir, const std::string& toml, int workers,
                        std::optional<std::pair<std::int64_t, std::int64_t>> shard, std::function<double()> seconds) {
        std::filesystem::create_directories(tmp / dir);
        const auto source = tmp / dir / "campaign.toml";
        std::ofstream(source, std::ios::binary) << toml;
        cg::EngineOptions opt;
        opt.workers = workers;
        opt.shard = shard;
        opt.pid = 7;
        opt.process.seconds = std::move(seconds);
        opt.host = "test";
        return cg::run_campaign(toml, source, preg, opt);
    };
    auto run = [&](const std::string& dir, int workers, std::optional<std::pair<std::int64_t, std::int64_t>> shard = {}) {
        // the clock stands still: "t" the same whichever thread ran it
        return run_with(dir, text, workers, shard, [] { return 0.0; });
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

    // the heartbeat of the last run: finished, its jobs counted; the pool's
    // size in _workers.txt
    const auto root4 = tmp / "four" / "results" / "sample";
    const auto beat = cg::runner_state(root4);
    CHECK(beat && beat->find("state")->s == "finished" && !beat->find("alive")->b);
    CHECK(beat && beat->find("done")->i == 4 && beat->find("todo")->i == 4 && beat->find("slice")->s == "all");
    CHECK(cg::read_workers(root4, 5) == 4);
    for (const auto& [content, want] : std::vector<std::pair<std::string, int>>{
             {" 8 \n", 8}, {"-3", 0}, {"x", 5}, {"", 5}, {"+4", 4}, {"99999999999", INT_MAX}}) {
        std::ofstream(root4 / "_workers.txt", std::ios::binary) << content;
        CHECK(cg::read_workers(root4, 5) == want);
    }

    // a runner that beats (another process writing): the store is not
    // repaired, its unfinished record stays
    {
        std::ofstream out(seg, std::ios::binary | std::ios::app);
        out << read_text(seg).substr(0, 100);
    }
    const auto cut = std::filesystem::file_size(seg);
    {
        const double now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        std::ofstream(root4 / "_runner.json", std::ios::binary)
            << "{\"state\": \"running\", \"pid\": 1, \"updated\": " << std::to_string(now) << "}";
    }
    CHECK(cg::runner_state(root4)->find("alive")->b);
    run("four", 1);
    CHECK(std::filesystem::file_size(seg) == cut);
    cg::repair_store(store4);
    CHECK(std::filesystem::file_size(seg) == size);

    // a slice: its jobs only
    std::size_t sliced = 0;
    for (std::int64_t i = 0; i < 3; ++i) sliced += run("slices", 1, std::make_pair(i, std::int64_t{3})).done;
    CHECK(sliced == 4);
    CHECK(bodies(tmp / "slices" / "results" / "sample" / "store") == a);

    // run_timeout: a clock that moves a second each time it is read stops
    // every run after its first record; the record says "timeout" with the
    // last trajectory record, and the job is not run again
    std::string timed = text;
    timed.replace(timed.find("[campaign]\n"), 11, "[campaign]\nrun_timeout = 2.5\n");
    auto ticking = [] { return [n = 0.0]() mutable { return n += 1.0; }; };
    const cg::EngineCounts t = run_with("timeout", timed, 2, std::nullopt, ticking());
    CHECK(t.timeout == 4 && t.done == 0 && t.failed == 0);
    const auto store_t = tmp / "timeout" / "results" / "sample" / "store";
    for (const auto& [key, status] : cg::store_statuses(store_t)) CHECK(status == "timeout");
    for (const auto& [key, body] : bodies(store_t)) {
        CHECK(body.find("\"status\": \"timeout\"") != std::string::npos);
        CHECK(body.find("\"timeout_seconds\": 2.5") != std::string::npos);
        CHECK(body.find("\"last_record\": {\"gen\": 0, ") != std::string::npos);
    }
    CHECK(run_with("timeout", timed, 2, std::nullopt, ticking()).skipped == 4);

    // snapshots and archive checkpoints: snapshots.npz holds every record's
    // population (n rows each, variables too), archive_at.npz the archive
    // scenario at each record_at count below the budget, meta.json says both
    std::string snapped = text;
    snapped.replace(snapped.find("[campaign]\n"), 11,
                    "[campaign]\nkeep = \"full\"\nsnapshots = true\nsnapshot_variables = \"all\"\n"
                    "archive_checkpoints = true\nrecord_at = [200, 400, 5000]\n");
    CHECK(run_with("snapshots", snapped, 2, std::nullopt, [] { return 0.0; }).done == 4);
    for (const auto& f : std::filesystem::directory_iterator(tmp / "snapshots" / "results" / "sample" / "store")) {
        if (f.path().extension() != ".raw") continue;
        const std::string raw = read_text(f.path());
        const auto* buf = reinterpret_cast<const std::uint8_t*>(raw.data());
        for (std::size_t at = 0; at < raw.size();) {
            const st::RecordAt r = st::parse_record(buf, raw.size(), at);
            if (!r.ok) break;
            std::map<std::string, st::Block> by;
            for (auto& blk : st::decode_run(buf + r.body, r.body_size)) by[blk.name] = blk;
            const auto meta = mootation::json::parse(std::string(by["meta.json"].data.begin(), by["meta.json"].data.end()));
            const auto* snaps = meta.find("snapshots");
            const auto* ck = meta.find("archive_at");
            CHECK(snaps && ck && ck->keys == std::vector<std::string>({"200", "400"}));
            const auto& n = by["snapshots.npz/n"];
            const auto& F = by["snapshots.npz/F"];
            const auto& X = by["snapshots.npz/X"];
            CHECK(n.dtype == "<i8" && F.dtype == "<f4" && X.dtype == "<f4" && F.shape.size() == 2);
            std::int64_t rows = 0;
            for (std::size_t k = 0; k < n.data.size() / 8; ++k) {
                std::int64_t v = 0;
                std::memcpy(&v, n.data.data() + 8 * k, 8);
                rows += v;
            }
            CHECK(snaps && static_cast<std::int64_t>(n.shape.at(0)) == snaps->find("records")->i);
            CHECK(F.shape.at(0) == static_cast<std::uint64_t>(rows) && X.shape.at(0) == F.shape.at(0));
            CHECK(F.data.size() == 4 * F.shape[0] * F.shape[1]);
            const auto& budget = by["archive_at.npz/budget"];
            std::int64_t b0 = 0, b1 = 0;
            CHECK(budget.data.size() == 16);
            if (budget.data.size() == 16) {
                std::memcpy(&b0, budget.data.data(), 8);
                std::memcpy(&b1, budget.data.data() + 8, 8);
            }
            CHECK(b0 == 200 && b1 == 400);
            at = r.end;
        }
    }

    std::filesystem::remove_all(tmp);
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("engine: a campaign's records by a pool of threads, resumed and repaired\n");
    return 0;
}
