#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A campaign run by the C++ engine — task 17, stage 5, step 5: run/campaign.py's
// run_campaign for a campaign with storage = "store": campaign.json, the
// slice of the jobs (--shard i/n, --only), a store left by a stopped run
// repaired (store.repair), the jobs already done skipped (their records in
// the store's segments and packs: store.index), and the others run by a pool
// of threads, each appending its records to a segment of its own.
//
// What it needs: the config's text, the exported problem registry
// (registry.hpp; it also carries the Python modules' lists a config is
// checked against) and the cache of references (references.hpp). A job the
// engine does not run (job_run.hpp's refusal) is left pending and counted
// apart with its reason: the Python engine runs it.
// ============================================================================

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../run_core.hpp"
#include "../settings.hpp"
#include "../store/pack.hpp"
#include "../store/record.hpp"
#include "../store/run_json.hpp"
#include "config.hpp"
#include "job_run.hpp"
#include "jobs.hpp"
#include "references.hpp"
#include "registry.hpp"
#include "run_store.hpp"

namespace mootation::campaign {

// store.STORE_DIR, SCRATCH_DIR, LOG_FILE
constexpr const char* kStoreDir = "store";
constexpr const char* kScratchDir = "tmp";
constexpr const char* kStoreLog = "store.log";

namespace detail {

inline std::string read_all(const std::filesystem::path& p)
{
    std::ifstream in(p, std::ios::binary);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

inline std::string utf8(const std::filesystem::path& p)
{
    const auto s = p.u8string();
    return std::string(s.begin(), s.end());
}

inline std::string now_text()
{
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%S", &tm);
    return buf;
}

// run/config.py's _executable_findable: an existing file, or a name on PATH
inline bool executable_findable(const std::string& name)
{
    std::error_code ec;
    if (name.find('/') != std::string::npos || name.find('\\') != std::string::npos)
        return std::filesystem::is_regular_file(store::file::path_of(name), ec);
    const std::string path = getenv_str("PATH");
#if defined(_WIN32)
    const char sep = ';';
    std::vector<std::string> exts = {""};
    std::string pathext = getenv_str("PATHEXT");
    if (pathext.empty()) pathext = ".COM;.EXE;.BAT;.CMD";
    std::stringstream es(pathext);
    for (std::string e; std::getline(es, e, ';');) exts.push_back(e);
#else
    const char sep = ':';
    const std::vector<std::string> exts = {""};
#endif
    std::stringstream ps(path);
    for (std::string dir; std::getline(ps, dir, sep);) {
        if (dir.empty()) continue;
        for (const auto& e : exts)
            if (std::filesystem::is_regular_file(store::file::path_of(dir) / store::file::path_of(name + e), ec))
                return true;
    }
    return false;
}

}   // namespace detail

// What a config is checked against: the library's algorithms and knobs and
// the registry's problems and lists.
inline Registry config_registry(const ProblemRegistry& preg)
{
    Registry r;
    r.algorithms = run::algorithm_names();
    r.baselines = preg.names.baselines;
    r.scalar_names = preg.names.scalar_names;
    r.design_kinds = preg.names.design_kinds;
    r.knob_names = preg.names.knob_names;
    r.text_knobs = preg.names.text_knobs;
    r.list_knobs = preg.names.list_knobs;
    for (const auto& [name, p] : preg.problems) r.problems.emplace_back(name, p.n_obj);
#if defined(_WIN32)
    r.platform = "windows";
#elif defined(__APPLE__)
    r.platform = "darwin";
#else
    r.platform = "linux";
#endif
    r.executable_findable = detail::executable_findable;
    r.regex_groups = [](const std::string&, std::string& error) {
        error = "regular expressions are checked by the Python engine";
        return 0;
    };
    return r;
}

inline JobsRegistry jobs_registry(const ProblemRegistry& preg)
{
    JobsRegistry j;
    j.metric_names = preg.names.metric_names;
    j.budget_dependent = preg.names.budget_dependent;
    j.scalar_names = preg.names.scalar_names;
    j.scalar_others = preg.names.scalar_others;
    for (const auto& [name, p] : preg.problems) j.problems[name] = {p.n_obj, p.pop_size, p.n_gen};
    return j;
}

// ── the store's account of its runs ─────────────────────────────────────────

struct StoreEntry {
    std::string key, status;
    std::uint64_t offset = 0, length = 0;
    std::filesystem::path file;          // the segment or pack it is in (store_index)
};

// the status a record's meta.json says
inline std::string record_status(const std::uint8_t* body, std::size_t size)
{
    for (const auto& b : store::decode_run(body, size))
        if (b.name == "meta.json") {
            const json::Value meta = json::parse(std::string(b.data.begin(), b.data.end()));
            const json::Value* s = meta.find("status");
            return s && s->kind == json::Value::Kind::String ? s->s : "?";
        }
    return "?";
}

// store.scan_raw: every whole record of a segment, from its data; and the end of the last
inline std::vector<StoreEntry> scan_raw(const std::string& data, std::uint64_t* end_out)
{
    std::vector<StoreEntry> out;
    const auto* buf = reinterpret_cast<const std::uint8_t*>(data.data());
    std::size_t at = 0;
    while (at < data.size()) {
        const store::RecordAt r = store::parse_record(buf, data.size(), at);
        if (!r.ok) break;
        std::string status = "?";
        try {
            status = record_status(buf + r.body, r.body_size);
        } catch (const std::exception&) {
        }
        out.push_back({r.key, status, at, r.end - at, {}});
        at = r.end;
    }
    if (end_out) *end_out = at;
    return out;
}

// store._read_idx: the index's entries where they cover the data up to its
// last whole record, else nothing
inline std::optional<std::vector<StoreEntry>> read_idx(const std::filesystem::path& raw)
{
    std::filesystem::path idx = raw;
    idx.replace_extension(".idx");
    std::error_code ec;
    if (!std::filesystem::is_regular_file(idx, ec)) return std::nullopt;
    std::vector<StoreEntry> out;
    std::uint64_t end = 0;
    std::stringstream lines(detail::read_all(idx));
    for (std::string line; std::getline(lines, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::vector<std::string> parts;
        std::stringstream ls(line);
        for (std::string p; std::getline(ls, p, '\t');) parts.push_back(p);
        if (parts.size() != 5) return std::nullopt;
        StoreEntry e;
        try {
            e.offset = std::stoull(parts[0]);
            e.length = std::stoull(parts[1]);
        } catch (const std::exception&) {
            return std::nullopt;
        }
        if (e.offset != end) return std::nullopt;
        e.status = parts[3];
        e.key = parts[4];
        end = e.offset + e.length;
        out.push_back(std::move(e));
    }
    const std::uint64_t size = std::filesystem::file_size(raw, ec);
    if (end > size) return std::nullopt;
    if (end < size) {            // a record whole past the index: read from the data
        std::ifstream in(raw, std::ios::binary);
        in.seekg(static_cast<std::streamoff>(end));
        std::string tail((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (store::parse_record(reinterpret_cast<const std::uint8_t*>(tail.data()), tail.size(), 0).ok)
            return std::nullopt;
    }
    return out;
}

// a pack's entries from its footer and index alone (pack.hpp's pack_entries,
// without reading the records)
inline std::vector<StoreEntry> pack_index(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    in.seekg(0, std::ios::end);
    const std::uint64_t size = static_cast<std::uint64_t>(in.tellg());
    if (size < 64) throw std::runtime_error(detail::utf8(path.filename()) + ": shorter than a pack's footer");
    std::uint8_t foot[64];
    in.seekg(static_cast<std::streamoff>(size - 64));
    in.read(reinterpret_cast<char*>(foot), 64);
    // the footer: MOOTPEND, six places (the index's the last two), the
    // file's CRC and the index's
    const std::string name = detail::utf8(path.filename());
    const std::uint64_t index_at = store::rec::get_le(foot + 8 + 8 * 4, 8);
    const std::uint64_t index_size = store::rec::get_le(foot + 8 + 8 * 5, 8);
    const std::uint32_t index_crc = static_cast<std::uint32_t>(store::rec::get_le(foot + 60, 4));
    if (std::memcmp(foot, "MOOTPEND", 8) != 0 || index_size == 0 || index_at > size - 64 ||
        index_size > size - 64 - index_at)
        throw std::runtime_error(name + ": no pack footer at its end");
    std::vector<std::uint8_t> idx(static_cast<std::size_t>(index_size));
    in.seekg(static_cast<std::streamoff>(index_at));
    in.read(reinterpret_cast<char*>(idx.data()), static_cast<std::streamsize>(index_size));
    if (store::crc32c(idx.data(), idx.size()) != index_crc) throw std::runtime_error(name + ": the index's checksum does not hold");
    std::string lines;
    if (idx[0] == 'T') {
        const auto t = store::decode_text(idx.data() + 1, idx.size() - 1);
        lines.assign(t.begin(), t.end());
    } else if (idx[0] == 'R') {
        lines.assign(reinterpret_cast<const char*>(idx.data() + 1), idx.size() - 1);
    } else {
        throw std::runtime_error(name + ": an index of unknown kind");
    }
    std::vector<StoreEntry> out;
    for (const auto& e : store::index_entries(lines)) out.push_back({e.key, e.status, e.offset, e.length, {}});
    return out;
}

// store.packs and store.segments: the packs (.mpk, then the packs of before,
// .raw), the segments in the order they were started
inline void store_files(const std::filesystem::path& store_dir, std::vector<std::filesystem::path>& packs,
                        std::vector<std::filesystem::path>& segs)
{
    std::vector<std::filesystem::path> mpk, old_packs;
    std::error_code ec;
    if (std::filesystem::is_directory(store_dir, ec))
        for (const auto& f : std::filesystem::directory_iterator(store_dir)) {
            const std::string n = detail::utf8(f.path().filename());
            if (n.rfind("pack-", 0) == 0 && f.path().extension() == ".mpk") mpk.push_back(f.path());
            else if (n.rfind("pack-", 0) == 0 && f.path().extension() == ".raw") old_packs.push_back(f.path());
            else if (n.rfind("seg-", 0) == 0 && f.path().extension() == ".raw") segs.push_back(f.path());
        }
    std::sort(mpk.begin(), mpk.end());
    std::sort(old_packs.begin(), old_packs.end());
    std::sort(segs.begin(), segs.end());
    packs = mpk;
    packs.insert(packs.end(), old_packs.begin(), old_packs.end());
}

// store.segment_entries: a pack's index, a segment's index where it agrees
// with the data, else the data's account
inline std::vector<StoreEntry> file_entries(const std::filesystem::path& raw)
{
    std::vector<StoreEntry> out;
    if (raw.extension() == ".mpk") out = pack_index(raw);
    else if (auto listed = read_idx(raw)) out = *listed;
    else out = scan_raw(detail::read_all(raw), nullptr);
    for (auto& e : out) e.file = raw;
    return out;
}

// store.index: key -> where its record is, the packs first, then the
// segments in the order they were started (a run done again: its later record)
inline std::map<std::string, StoreEntry> store_index(const std::filesystem::path& store_dir)
{
    std::vector<std::filesystem::path> packs, segs;
    store_files(store_dir, packs, segs);
    std::map<std::string, StoreEntry> out;
    for (const auto* list : {&packs, &segs})
        for (const auto& raw : *list)
            for (auto& e : file_entries(raw)) out[e.key] = e;
    return out;
}

// store.index's statuses: key -> the status of its record
inline std::map<std::string, std::string> store_statuses(const std::filesystem::path& store_dir)
{
    std::map<std::string, std::string> out;
    for (const auto& [key, e] : store_index(store_dir)) out[key] = e.status;
    return out;
}

// store.repair: cut every segment's unfinished tail, rebuild an index that
// disagrees with its data, remove the scratch folders; what was done
inline std::vector<std::string> repair_store(const std::filesystem::path& store_dir)
{
    std::vector<std::string> done;
    std::error_code ec;
    auto log = [&](const std::string& line) {
        std::ofstream f(store_dir / kStoreLog, std::ios::binary | std::ios::app);
        f << detail::now_text() << " " << line << (kTextCrlf ? "\r\n" : "\n");
    };
    const auto scratch = store_dir / kScratchDir;
    if (std::filesystem::is_directory(scratch, ec) && !std::filesystem::is_empty(scratch, ec)) {
        std::filesystem::remove_all(scratch, ec);
        done.push_back(std::string(kScratchDir) + "/: removed the folders of runs stopped before their record");
    }
    std::vector<std::filesystem::path> raws;
    for (const auto& f : std::filesystem::directory_iterator(store_dir, ec)) {
        const std::string n = detail::utf8(f.path().filename());
        if (f.path().extension() == ".raw" && (n.rfind("seg-", 0) == 0 || n.rfind("pack-", 0) == 0))
            raws.push_back(f.path());
    }
    std::sort(raws.begin(), raws.end());
    for (const auto& raw : raws) {
        const std::uint64_t size = std::filesystem::file_size(raw, ec);
        auto listed = read_idx(raw);
        std::vector<StoreEntry> entries;
        std::uint64_t end = 0;
        if (listed) {
            entries = *listed;
            end = entries.empty() ? 0 : entries.back().offset + entries.back().length;
        } else {
            entries = scan_raw(detail::read_all(raw), &end);
        }
        const std::string name = detail::utf8(raw.filename());
        if (end < size) {
            std::filesystem::resize_file(raw, end, ec);
            const std::string msg = name + ": cut an unfinished record off at " + std::to_string(end) + " (" +
                                    std::to_string(size - end) + " bytes)";
            log(msg);
            done.push_back(msg);
        }
        if (!listed) {
            std::string text;
            const std::string data = detail::read_all(raw);
            for (const auto& e : entries) {
                const store::RecordAt r = store::parse_record(reinterpret_cast<const std::uint8_t*>(data.data()),
                                                              data.size(), static_cast<std::size_t>(e.offset));
                store::Entry se;
                se.key = e.key;
                se.offset = e.offset;
                se.length = e.length;
                se.crc = r.crc;
                se.status = e.status;
                text += store::index_line(se);
            }
            std::filesystem::path idx = raw;
            idx.replace_extension(".idx");
            std::ofstream f(idx, std::ios::binary | std::ios::trunc);
            f << text;
            const std::string msg = name + ": index rebuilt from the data (" + std::to_string(entries.size()) + " records)";
            log(msg);
            done.push_back(msg);
        }
    }
    return done;
}

// ── live control of a running campaign (campaign.py) ────────────────────────
//
// Two small files in the results root, the Python runner's: `_workers.txt`
// holds the number of workers the runner keeps, re-read every 1.5 s (a higher
// number starts threads at once, a lower one lets the surplus finish the job
// it is on and leave, 0 drains the campaign and stops it with the unfinished
// jobs left pending); `_runner.json` is the runner's heartbeat, rewritten every
// two seconds. A runner of either engine sees the other's: neither repairs a
// store whose runner is alive.

constexpr const char* kWorkersFile = "_workers.txt";
constexpr const char* kRunnerFile = "_runner.json";
constexpr double kHeartbeatStale = 15.0;     // seconds without a beat: the runner is gone

namespace detail {

// time.time()
inline double unix_time()
{
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// _write_text: a temporary file moved over the old one. On Windows a file
// another process holds open cannot be replaced; a reader holds it for
// milliseconds, so the move is tried again for a few seconds.
inline void write_text_file(const std::filesystem::path& path, const std::string& text)
{
    auto tmp = path;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f << text_mode(text);
    }
    std::chrono::milliseconds delay(10);
    for (int attempt = 0; attempt < 39; ++attempt) {
        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
        if (!ec) return;
        std::this_thread::sleep_for(delay);
        delay = std::min(2 * delay, std::chrono::milliseconds(250));
    }
    std::filesystem::rename(tmp, path);
}

}   // namespace detail

// read_workers: the worker count `_workers.txt` says, or `fallback`
inline int read_workers(const std::filesystem::path& root, int fallback)
{
    std::string s = detail::read_all(root / kWorkersFile);
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return fallback;
    s = s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
    std::size_t at = s[0] == '+' || s[0] == '-' ? 1 : 0;
    if (at == s.size() || s.find_first_not_of("0123456789", at) != std::string::npos) return fallback;
    if (s[0] == '-') return 0;
    if (s.size() - at > 9) return INT_MAX;                  // more than any machine has
    return std::stoi(s);
}

inline void write_workers(const std::filesystem::path& root, int n)
{
    std::filesystem::create_directories(root);
    detail::write_text_file(root / kWorkersFile, std::to_string(std::max(0, n)) + "\n");
}

// runner_state: the last heartbeat of this results root's runner with
// "alive" added (running, and a beat in the last 15 s); none if no runner
// ever wrote one
inline std::optional<json::Value> runner_state(const std::filesystem::path& root)
{
    std::error_code ec;
    if (!std::filesystem::is_regular_file(root / kRunnerFile, ec)) return std::nullopt;
    json::Value st;
    try {
        st = json::parse(detail::read_all(root / kRunnerFile));
    } catch (const std::exception&) {
        return std::nullopt;
    }
    if (st.kind != json::Value::Kind::Object) return std::nullopt;
    const json::Value* state = st.find("state");
    const json::Value* updated = st.find("updated");
    double when = 0.0;
    if (updated && updated->kind == json::Value::Kind::Float) when = updated->f;
    if (updated && updated->kind == json::Value::Kind::Int) when = static_cast<double>(updated->i);
    const bool running = state && state->kind == json::Value::Kind::String && state->s == "running";
    st.set("alive", json::Value::boolean(running && detail::unix_time() - when <= kHeartbeatStale));
    return st;
}

// ── the campaign ────────────────────────────────────────────────────────────

struct EngineOptions {
    int workers = 1;
    std::optional<std::pair<std::int64_t, std::int64_t>> shard;   // part i of n (shard_of)
    std::optional<std::vector<std::int64_t>> only;                // these job indices
    bool force = false;
    Process process;                                              // meta.json's process fields
    long long pid = 0;
    std::string host;
    std::function<void(const std::string&)> say;                  // progress lines; unset: silent
};

struct EngineCounts {
    std::size_t total = 0, selected = 0, skipped = 0, done = 0, failed = 0, timeout = 0;
    std::size_t pending = 0;                                      // not run: the campaign drained (_workers.txt 0)
    std::map<std::string, std::size_t> left;                      // to Python, by reason
};

// campaign.json as run_campaign writes it (json.dumps(..., indent=1), text mode)
inline void write_campaign_json(const std::filesystem::path& root, const Config& cfg, const CampaignSpec& spec,
                                const std::vector<Job>& jobs)
{
    json::Value c = json::Value::object(), problems = json::Value::array(), algs = json::Value::array(),
                seeds = json::Value::array();
    std::set<std::string> ps;
    std::set<std::int64_t> ss;
    for (const auto& j : jobs) {
        ps.insert(j.problem);
        ss.insert(j.seed);
    }
    for (const auto& p : ps) problems.items.push_back(json::Value::string(p));
    for (const auto& a : cfg.algorithms) algs.items.push_back(json::Value::string(a.key()));
    for (auto s : ss) seeds.items.push_back(json::Value::integer(s));
    c.set("name", json::Value::string(cfg.name));
    c.set("config", json::Value::string(cfg.source_path ? detail::utf8(*cfg.source_path) : std::string()));
    c.set("problems", problems);
    c.set("algorithms", algs);
    c.set("seeds", seeds);
    c.set("jobs", json::Value::integer(static_cast<std::int64_t>(jobs.size())));
    c.set("spec", to_value(spec));
    std::filesystem::create_directories(root);
    const auto tmp = root / "campaign.json.tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f << text_mode(json::dump(c, 1));
    }
    std::filesystem::rename(tmp, root / "campaign.json");
}

// run_campaign: the campaign's jobs not done yet, run by `workers` threads
inline EngineCounts run_campaign(const std::string& text, const std::filesystem::path& source_path,
                                 const ProblemRegistry& preg, const EngineOptions& opt)
{
    const Registry reg = config_registry(preg);
    const JobsRegistry jreg = jobs_registry(preg);
    Config cfg = loads(text, reg, source_path);
    const auto problems = validate(cfg, reg);
    if (!problems.empty()) throw ConfigError("", "the config has problems: " + join(problems, "; "));
    if (cfg.kind != "builtin") throw ConfigError("problem.kind", "the C++ engine runs builtin campaigns");
    const CampaignSpec spec = campaign_spec(cfg, jreg);
    if (spec.storage != "store") throw ConfigError("campaign.storage", "the C++ engine writes a store");
    const std::vector<Job> jobs = expand_jobs(cfg, spec, jreg);
    const auto root = out_root(cfg, spec);
    if (!root) throw ConfigError("campaign.out", "a path with {name} and nothing else in braces");
    write_campaign_json(*root, cfg, spec, jobs);
    EngineCounts counts;
    counts.total = jobs.size();
    std::vector<const Job*> selected;
    for (const auto& j : jobs) {
        if (opt.only && std::find(opt.only->begin(), opt.only->end(), j.index) == opt.only->end()) continue;
        if (!opt.only && opt.shard && shard_of(j, opt.shard->second) != opt.shard->first) continue;
        selected.push_back(&j);
    }
    counts.selected = selected.size();
    const auto store_dir = *root / kStoreDir;
    std::error_code ec;
    // a campaign stopped while a worker wrote: its unfinished record goes and
    // the run is done again. Not while a runner of this folder beats: its
    // workers may be writing
    const auto runner = runner_state(*root);
    const bool alive = runner && runner->find("alive")->b;
    if (std::filesystem::is_directory(store_dir, ec) && !alive)
        for (const auto& line : repair_store(store_dir))
            if (opt.say) opt.say(line);
    const auto status = store_statuses(store_dir);
    // the jobs to run, and those the engine leaves to Python
    std::vector<const Job*> todo;
    const auto algorithms = run::algorithm_names();
    for (const Job* j : selected) {
        const auto it = status.find(job_key(*j));
        const std::string st = it == status.end() ? "pending" : it->second;
        if (!opt.force && (st == "done" || st == "timeout")) {
            ++counts.skipped;
            continue;
        }
        const RegistryProblem& p = preg.problems.at(j->problem);
        if (auto why = refusal(*j, spec, p, algorithms)) {
            ++counts.left[*why];
            continue;
        }
        todo.push_back(j);
    }
    // every reference the jobs read, there before the first runs
    std::set<std::string> missing;
    for (const Job* j : todo) {
        const auto path = entry_path(preg.reference_version, j->problem, spec.n_ref);
        if (!std::filesystem::is_regular_file(path, ec)) missing.insert(j->problem);
    }
    if (!missing.empty())
        throw std::runtime_error("the cache of references (" + detail::utf8(refs_root()) + ") lacks " +
                                 std::to_string(missing.size()) + " entries at n_ref = " + std::to_string(spec.n_ref) +
                                 " (build them with: mootation refs build --n-ref " + std::to_string(spec.n_ref) +
                                 " --problems " + join({missing.begin(), missing.end()}, ",") + ")");
    if (opt.say)
        opt.say(cfg.name + ": " + std::to_string(jobs.size()) + " jobs total, " + std::to_string(selected.size()) +
                " in this slice, " + std::to_string(todo.size()) + " to run -> " + detail::utf8(*root));
    if (todo.empty()) return counts;
    // the pool: each thread takes the next job and appends its record to its
    // own segment; this thread keeps the pool at what _workers.txt says and
    // writes the heartbeat
    std::mutex lock;
    std::condition_variable changed;          // a job finished, a thread left
    std::size_t next = 0, finished = 0;       // under `lock`, as the rest
    int active = 0, want = 0;
    std::set<std::int64_t> running;
    std::exception_ptr crash;
    auto worker = [&]() {
        SegmentWriter writer(store_dir, static_cast<std::uint64_t>(spec.segment_mb) << 20, opt.pid, opt.host,
                             {{"code_stamp", preg.code_stamp}, {"trajectory_version", "2"}, {"detmath_path", "detmath"}});
        for (;;) {
            std::size_t k = 0;
            {
                std::lock_guard<std::mutex> g(lock);
                // surplus, or nothing left: leave
                if (active > want || next >= todo.size() || crash) {
                    --active;
                    changed.notify_all();
                    return;
                }
                k = next++;
                running.insert(todo[k]->index);
            }
            const Job& job = *todo[k];
            try {
                const RegistryProblem& p = preg.problems.at(job.problem);
                bool front = false, with_set = spec.set_metrics && p.has_pareto_set;
                for (const auto* list : {&spec.metrics, &spec.final_metrics})
                    for (const auto& w : *list) {
                        front = front || needs_front().count(w);
                        with_set = with_set || needs_set().count(w);
                    }
                const auto entry = load_entry(entry_path(preg.reference_version, job.problem, spec.n_ref));
                JobRun run(job, spec, p, shared_references(entry, front, spec.dense_reference, with_set));
                run.set_registry(preg, jreg);
                const JobOutcome out = run.run(opt.process);
                const JobFiles files = run.finish(out);
                writer.append(job_key(job), record_blocks(files, out.trajectory));
                std::lock_guard<std::mutex> g(lock);
                (out.status == "done" ? counts.done : out.status == "timeout" ? counts.timeout : counts.failed) += 1;
                ++finished;
                running.erase(job.index);
                changed.notify_all();
                if (opt.say)
                    opt.say("[" + std::to_string(job.index) + "] " + job.problem + " " + job.key() + " seed " +
                            std::to_string(job.seed) + ": " + out.status);
            } catch (...) {
                std::lock_guard<std::mutex> g(lock);
                if (!crash) crash = std::current_exception();    // the others stop after their job
                running.erase(job.index);
                --active;
                changed.notify_all();
                return;
            }
        }
    };
    const std::string label = opt.shard ? "shard " + std::to_string(opt.shard->first) + "/" + std::to_string(opt.shard->second)
                              : opt.only ? std::to_string(opt.only->size()) + " job(s)"
                                         : "all";
    const double started = detail::unix_time();
    auto beat = [&](const std::string& state) {          // under `lock`
        json::Value st = json::Value::object(), ids = json::Value::array();
        for (auto i : running) ids.items.push_back(json::Value::integer(i));
        st.set("state", json::Value::string(state));
        st.set("pid", json::Value::integer(opt.pid));
        st.set("slice", json::Value::string(label));
        st.set("started", json::Value::number(started));
        st.set("updated", json::Value::number(detail::unix_time()));
        st.set("target", json::Value::integer(want));
        st.set("active", json::Value::integer(active));
        st.set("todo", json::Value::integer(static_cast<std::int64_t>(todo.size())));
        st.set("done", json::Value::integer(static_cast<std::int64_t>(counts.done)));
        st.set("failed", json::Value::integer(static_cast<std::int64_t>(counts.failed)));
        st.set("running", ids);
        detail::write_text_file(*root / kRunnerFile, json::dump(st, 1));
    };
    write_workers(*root, opt.workers);
    const int ceiling = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    std::vector<std::thread> pool;
    std::string state = "running";
    {
        using clock = std::chrono::steady_clock;
        clock::time_point last_read{}, last_beat{};
        std::unique_lock<std::mutex> g(lock);
        while (finished < todo.size() && !(crash && active == 0)) {
            const auto now = clock::now();
            if (pool.empty() || now - last_read >= std::chrono::milliseconds(1500)) {
                last_read = now;
                g.unlock();
                const int asked = read_workers(*root, opt.workers);
                g.lock();
                want = std::min(asked, ceiling);
                const int wanted = static_cast<int>(std::min<std::size_t>(want, running.size() + (todo.size() - next)));
                for (; active < wanted && !crash; ++active) pool.emplace_back(worker);
                if (want == 0 && active == 0) {
                    state = "stopped";                   // drained on request
                    break;
                }
            }
            if (now - last_beat >= std::chrono::seconds(2)) {
                last_beat = now;
                beat("running");
            }
            changed.wait_for(g, std::chrono::milliseconds(250));
        }
        if (crash) state = "broken";
    }
    for (auto& t : pool) t.join();
    std::lock_guard<std::mutex> g(lock);
    beat(state == "running" ? "finished" : state);
    counts.pending = todo.size() - finished;
    if (crash) std::rethrow_exception(crash);
    return counts;
}

}   // namespace mootation::campaign
