// SPDX-License-Identifier: Apache-2.0
// Workers by free memory (task 32, layer 5; include/mootation/mem_workers.hpp,
// AUTO-MEM-1 ... 6, and the campaign engine's pool, engine.hpp POOL-MEM-1 ...
// 5): the count min(threads, floor(mem_share x (free + held) / peak)), at
// least 1; a trial's peak from the process's readings; the readings of the
// system there and sane; small free memory gives one worker, not a refusal;
// a reading that cannot be made turns the protection off; a decision read
// again sees other programs' memory. The engine on the sample campaign
// (tests/data/job_sample) with readers of the test's own: a trial, then the
// pool's size, the same records as one thread's, the decision in the counts
// and the heartbeat; without auto_mem nothing of it.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mootation/campaign/engine.hpp>
#include <mootation/mem_workers.hpp>

namespace cg = mootation::campaign;
namespace mem = mootation::mem;
namespace st = mootation::store;

namespace {

int failures = 0;

void check(bool ok, const char* what, int line)
{
    if (!ok) {
        std::printf("FAIL line %d: %s\n", line, what);
        ++failures;
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

constexpr std::uint64_t MB = std::uint64_t{1} << 20;

std::string read_text(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

// key -> the record's blocks, meta.json without timing_us (test_engine's)
std::map<std::string, std::string> bodies(const std::filesystem::path& store)
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

void set_env(const char* name, const std::string& value)
{
#if defined(_WIN32)
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

// readers of the test's own: the free memory and the process's levels given
mem::Readers fixed(std::optional<std::uint64_t> free_mem, std::uint64_t current, std::uint64_t peak)
{
    mem::Readers r;
    r.free_memory = [free_mem] { return free_mem; };
    r.process = [current, peak] { return std::optional<mem::ProcessMemory>(mem::ProcessMemory{current, peak}); };
    return r;
}

void the_count()
{
    // floor(0.5 x 1024 / 100) = 5 of 8
    CHECK(mem::workers_for(8, 0.5, 1024 * MB, 100 * MB) == 5);
    CHECK(mem::workers_for(3, 0.5, 1024 * MB, 100 * MB) == 3);              // the cap
    CHECK(mem::workers_for(8, 1.0, 300 * MB, 100 * MB) == 3);               // exactly 3
    CHECK(mem::workers_for(8, 1.0, 300 * MB - 1, 100 * MB) == 2);
    CHECK(mem::workers_for(8, 0.5, 10 * MB, 100 * MB) == 1);                // AUTO-MEM-5: one, not none
    CHECK(mem::workers_for(8, 0.5, 0, 100 * MB) == 1);
    CHECK(mem::workers_for(8, 0.5, 1024 * MB, 0) == 8);                     // a peak of nothing
    CHECK(mem::workers_for(0, 0.5, 1024 * MB, 100 * MB) == 1);
}

void the_trial_and_the_decisions()
{
    // a trial from a level of 100 MB to a peak of 400 MB: a job's peak 300 MB
    std::uint64_t current = 100 * MB, peak = 100 * MB;
    std::optional<std::uint64_t> free_mem = 64 * MB;
    mem::Readers r;
    r.free_memory = [&free_mem] { return free_mem; };
    r.process = [&current, &peak] {
        return std::optional<mem::ProcessMemory>(mem::ProcessMemory{current, peak});
    };
    mem::AutoMem am(8, 0.5, r);
    CHECK(!am.peak());
    am.start_trial();
    current = 150 * MB;
    peak = 400 * MB;
    am.end_trial();
    CHECK(am.peak() && *am.peak() == 300 * MB);

    // small free memory: one worker, not a refusal (AUTO-MEM-5)
    mem::Decision d = am.decide();
    CHECK(d.workers == 1 && d.threads == 8 && d.mem_share == 0.5);
    CHECK(d.peak && *d.peak == 300 * MB && d.free_memory && *d.free_memory == 64 * MB && d.held == 50 * MB);

    // the decision read again: other programs gave memory back
    free_mem = 6000 * MB;
    d = am.decide();
    CHECK(d.workers == 8);                                  // floor(0.5 x 6050 / 300) = 10, capped at 8
    free_mem = 1800 * MB;
    d = am.decide();
    CHECK(d.workers == 3);                                  // floor(0.5 x 1850 / 300) = 3

    // what the pool holds counts as free (AUTO-MEM-3): 4 jobs at their peak
    current = 100 * MB + 4 * 300 * MB;
    free_mem = 1200 * MB;
    d = am.decide();
    CHECK(d.held == 1200 * MB && d.workers == 4);           // floor(0.5 x 2400 / 300) = 4

    // a peak measured otherwise
    am.set_peak(600 * MB);
    CHECK(am.decide().workers == 2);
    am.set_peak(0);
    CHECK(am.peak() && *am.peak() == 1);

    // a level after the trial below the one before: the peak is at least a byte
    mem::AutoMem low(4, 0.5, fixed(64 * MB, 100 * MB, 100 * MB));
    low.start_trial();
    low.end_trial();
    CHECK(low.peak() && *low.peak() == 1);

    // a reading that cannot be made: the protection is off (AUTO-MEM-4)
    mem::AutoMem blind(6, 0.5, fixed(std::nullopt, 100 * MB, 200 * MB));
    blind.start_trial();
    blind.end_trial();
    d = blind.decide();
    CHECK(d.workers == 6 && !d.free_memory && d.peak);
    mem::Readers no_process = fixed(64 * MB, 0, 0);
    no_process.process = [] { return std::optional<mem::ProcessMemory>(); };
    mem::AutoMem unmeasured(6, 0.5, no_process);
    unmeasured.start_trial();
    unmeasured.end_trial();
    d = unmeasured.decide();
    CHECK(d.workers == 6 && !d.peak);
    // no current level (getrusage alone): the level before is 0, the peak the whole process's
    mem::Readers peak_only;
    peak_only.free_memory = [] { return std::optional<std::uint64_t>(64 * MB); };
    peak_only.process = [] {
        return std::optional<mem::ProcessMemory>(mem::ProcessMemory{std::nullopt, 250 * MB});
    };
    mem::AutoMem whole(4, 0.5, peak_only);
    whole.start_trial();
    whole.end_trial();
    CHECK(whole.peak() && *whole.peak() == 250 * MB && whole.decide().held == 0);

    // the knobs' ranges
    for (const auto& [threads, share] : std::vector<std::pair<int, double>>{{0, 0.5}, {4, 0.0}, {4, 1.5}, {4, -1.0}}) {
        bool threw = false;
        try {
            mem::AutoMem bad(threads, share, r);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw);
    }
}

void the_system()
{
#if defined(_WIN32) || defined(__linux__)
    // the readings of Windows and Linux are there
    const auto free_mem = mem::free_bytes();
    CHECK(free_mem && *free_mem > 0);
    const auto before = mem::process_memory();
    CHECK(before && before->current && *before->current > 0 && before->peak >= *before->current);
#endif
    // 64 MB written page by page: the process's peak is at least most of it,
    // and a trial around the writing measures at least most of it
    mem::AutoMem am(4, 0.5);
    am.start_trial();
    {
        std::vector<char> block(64 * MB);
        for (std::size_t k = 0; k < block.size(); k += 4096) block[k] = static_cast<char>(k >> 12);
        std::uint64_t sum = 0;
        for (std::size_t k = 0; k < block.size(); k += 4096) sum += static_cast<unsigned char>(block[k]);
        CHECK(sum > 0);
    }
    am.end_trial();
    const auto after = mem::process_memory();
    if (after) {
        CHECK(after->peak >= 48 * MB);
        CHECK(am.peak() && *am.peak() >= 48 * MB);
        const mem::Decision d = am.decide();
        CHECK(d.workers >= 1 && d.workers <= 4);
    }
}

int body()
{
    the_count();
    the_trial_and_the_decisions();
    the_system();

    const std::string data = std::string(MOOTATION_TEST_DATA) + "/job_sample/";
    set_env("MOOTATION_REFS_DIR", data + "refs");
    const cg::ProblemRegistry preg = cg::load_registry(data + "registry.json");
    std::string text = read_text(data + "campaign.toml");
    text.replace(text.find("[campaign]\n"), 11, "[campaign]\nstorage = \"store\"\n");
    const auto tmp = std::filesystem::temp_directory_path() /
                     ("mootation_test_auto_mem_" + std::to_string(std::rand()) + "_" + std::to_string(std::rand()));
    std::filesystem::remove_all(tmp);

    std::vector<std::string> said;
    auto run = [&](const std::string& dir, int workers, bool auto_mem, mem::Readers readers,
                   std::optional<std::vector<std::int64_t>> only = std::nullopt) {
        std::filesystem::create_directories(tmp / dir);
        const auto source = tmp / dir / "campaign.toml";
        std::ofstream(source, std::ios::binary) << text;
        cg::EngineOptions opt;
        opt.workers = workers;
        opt.auto_mem = auto_mem;
        opt.mem_readers = std::move(readers);
        opt.only = std::move(only);
        opt.pid = 7;
        opt.process.seconds = [] { return 0.0; };   // the clock stands still (test_engine)
        opt.host = "test";
        opt.say = [&said](const std::string& line) { said.push_back(line); };
        return cg::run_campaign(text, source, preg, opt);
    };
    auto store = [&](const std::string& dir) { return tmp / dir / "results" / "sample" / "store"; };
    auto heartbeat = [&](const std::string& dir) { return cg::runner_state(tmp / dir / "results" / "sample"); };

    // a numbered pool: no decision, no auto_mem in the heartbeat
    const cg::EngineCounts one = run("one", 1, false, {});
    CHECK(one.done == 4 && one.auto_mem.empty());
    const auto b1 = heartbeat("one");
    CHECK(b1 && !b1->find("auto_mem"));
    const auto records = bodies(store("one"));
    CHECK(records.size() == 4);

    // auto_mem, 1 MB free: a trial of 50 MB, then one worker, not a refusal
    said.clear();
    const cg::EngineCounts tiny = run("tiny", 4, true, fixed(1 * MB, 100 * MB, 150 * MB));
    CHECK(tiny.done == 4 && tiny.failed == 0);
    CHECK(tiny.auto_mem.size() == 1);
    if (!tiny.auto_mem.empty()) {
        const mem::Decision& d = tiny.auto_mem.front();
        CHECK(d.workers == 1 && d.threads == 4 && d.mem_share == 0.5);
        CHECK(d.peak && *d.peak == 50 * MB && d.free_memory && *d.free_memory == 1 * MB && d.held == 0);
    }
    CHECK(bodies(store("tiny")) == records);
    const auto bt = heartbeat("tiny");
    const mootation::json::Value* m = bt ? bt->find("auto_mem") : nullptr;
    CHECK(m && m->find("workers")->i == 1 && m->find("peak")->i == static_cast<std::int64_t>(50 * MB) &&
          m->find("free")->i == static_cast<std::int64_t>(1 * MB) && !m->find("trial")->b);
    bool told = false;
    for (const auto& line : said)
        told = told || line.find("auto_mem: 1 worker(s) of 4 (a job's peak 50 MB, free 1 MB") == 0;
    CHECK(told);

    // plenty free: the cap; the campaign's memory_share as the share (POOL-MEM-3)
    std::string shared = text;
    shared.replace(shared.find("[campaign]\n"), 11, "[campaign]\nmemory_share = 0.25\n");
    std::swap(text, shared);
    const cg::EngineCounts wide = run("wide", 4, true, fixed(1024 * 1024 * MB, 100 * MB, 150 * MB));
    std::swap(text, shared);
    CHECK(wide.done == 4 && wide.auto_mem.size() == 1);
    if (!wide.auto_mem.empty()) CHECK(wide.auto_mem.front().workers == 4 && wide.auto_mem.front().mem_share == 0.25);
    CHECK(bodies(store("wide")) == records);

    // the free memory unknown: the cap, the protection off (AUTO-MEM-4)
    said.clear();
    const cg::EngineCounts blind = run("blind", 3, true, fixed(std::nullopt, 100 * MB, 150 * MB));
    CHECK(blind.done == 4 && blind.auto_mem.size() == 1);
    if (!blind.auto_mem.empty()) CHECK(blind.auto_mem.front().workers == 3 && !blind.auto_mem.front().free_memory);
    bool off = false;
    for (const auto& line : said) off = off || line.find("the protection is off") != std::string::npos;
    CHECK(off);
    CHECK(bodies(store("blind")) == records);

    // the system's readings: a trial, a size within the cap, the same records
    const cg::EngineCounts sys = run("system", 2, true, {});
    CHECK(sys.done == 4 && !sys.auto_mem.empty());
    if (!sys.auto_mem.empty()) CHECK(sys.auto_mem.front().workers >= 1 && sys.auto_mem.front().workers <= 2);
    CHECK(bodies(store("system")) == records);

    // the trial the only job: its peak still in the counts
    const cg::EngineCounts single =
        run("single", 4, true, fixed(1 * MB, 100 * MB, 180 * MB), std::vector<std::int64_t>{2});
    CHECK(single.done == 1 && single.auto_mem.size() == 1);
    if (!single.auto_mem.empty()) CHECK(single.auto_mem.front().peak && *single.auto_mem.front().peak == 80 * MB);
    const auto bs = heartbeat("single");
    CHECK(bs && bs->find("auto_mem") && bs->find("auto_mem")->find("workers")->i == 1);

    // run again: nothing to run, no trial
    const cg::EngineCounts done_again = run("tiny", 4, true, fixed(1 * MB, 100 * MB, 150 * MB));
    CHECK(done_again.skipped == 4 && done_again.auto_mem.empty());

    std::filesystem::remove_all(tmp);
    return failures;
}

}   // namespace

int main()
{
    try {
        if (body()) {
            std::printf("%d failure(s)\n", failures);
            return 1;
        }
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    std::printf("auto_mem: the count, the trial, the readings, the engine's pool by memory\n");
    return 0;
}
