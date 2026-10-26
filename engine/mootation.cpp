// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// mootation — the campaign engine in C++ (task 17, stage 5, step 6).
//
//     mootation run <campaign.toml> [--workers N] [--shard i/n] [--only i,j,...]
//                   [--force] [--registry FILE] [--quiet]
//     mootation store ls|check|repair|compact|merge <campaign folder> ...
//     mootation --version
//
// `run` runs a campaign with storage = "store" as `python -m
// mootation.run.campaign` does (include/mootation/campaign/engine.hpp): the
// jobs not done yet, by a pool of threads, each record appended to the
// store; the jobs it does not run it names, with their reasons, for the
// Python engine. It reads the exported problem registry (`python -m
// mootation.run.registry_export`; the newest in the reference cache's
// registry/ folder unless --registry names one) and the cache of references
// (`mootation refs build`, Python).
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include <mootation/campaign/engine.hpp>
#include <mootation/campaign/store_tools.hpp>
#include <mootation/fp_env.hpp>
#include <mootation/version.hpp>

#if __has_include("mootation_git_stamp.h")
#include "mootation_git_stamp.h"
#endif
#ifndef MOOTATION_GIT_COMMIT
#define MOOTATION_GIT_COMMIT "unknown"
#endif
#ifndef MOOTATION_GIT_DIRTY
#define MOOTATION_GIT_DIRTY 0
#endif

namespace cg = mootation::campaign;
namespace js = mootation::json;

namespace {

const char* kUsage =
    "usage: mootation run <campaign.toml> [--workers N] [--shard i/n] [--only i,j,...] [--force]\n"
    "                     [--registry FILE] [--quiet]\n"
    "       mootation --version\n";

long long process_id()
{
#if defined(_WIN32)
    return _getpid();
#else
    return static_cast<long long>(getpid());
#endif
}

std::string host_name()
{
#if defined(_WIN32)
    const std::string h = cg::detail::getenv_str("COMPUTERNAME");
#else
    char buf[256] = {0};
    const std::string h = gethostname(buf, sizeof buf - 1) == 0 ? std::string(buf) : std::string();
#endif
    return h.empty() ? "unknown" : h;
}

// the newest export in the reference cache's registry folder
std::filesystem::path newest_registry()
{
    const auto dir = cg::refs_root() / "registry";
    std::filesystem::path best;
    std::filesystem::file_time_type when{};
    std::error_code ec;
    if (std::filesystem::is_directory(dir, ec))
        for (const auto& f : std::filesystem::directory_iterator(dir)) {
            if (f.path().extension() != ".json") continue;
            const auto t = std::filesystem::last_write_time(f.path(), ec);
            if (best.empty() || t > when) {
                best = f.path();
                when = t;
            }
        }
    if (best.empty())
        throw std::runtime_error("no exported problem registry in " + cg::detail::utf8(dir) +
                                 " (write one with: python -m mootation.run.registry_export)");
    return best;
}

// meta.json's process fields, as the engine says them of itself
cg::Process process_fields()
{
    cg::Process p;
    p.mootation = js::Value::string(mootation::version());
    js::Value rev = js::Value::object();
    const std::string commit = MOOTATION_GIT_COMMIT;
    const bool known = commit != "unknown";
    rev.set("commit", js::Value::string(commit));
    rev.set("dirty", known ? js::Value::boolean(MOOTATION_GIT_DIRTY != 0) : js::Value::null());
    rev.set("source", js::Value::string(known ? "build" : "unknown"));
    rev.set("core_commit", js::Value::string(commit));
    rev.set("core_dirty", known ? js::Value::boolean(MOOTATION_GIT_DIRTY != 0) : js::Value::null());
    p.revision = rev;
    p.host = js::Value::string(host_name());
#if defined(_WIN32)
    p.platform = js::Value::string("Windows (C++ engine)");
#elif defined(__APPLE__)
    p.platform = js::Value::string("Darwin (C++ engine)");
#else
    p.platform = js::Value::string("Linux (C++ engine)");
#endif
    p.python = js::Value::null();
    p.pid = js::Value::integer(process_id());
    const mootation::FpEnv e = mootation::fp_env();
    js::Value fp = js::Value::object();
    fp.set("arch", js::Value::string(e.arch));
    fp.set("ftz", e.known ? js::Value::boolean(e.ftz) : js::Value::null());
    fp.set("daz", e.known ? js::Value::boolean(e.daz) : js::Value::null());
    if (e.known) fp.set("register", js::Value::integer(static_cast<std::int64_t>(e.reg)));
    p.fp_env = fp;
    p.now = cg::detail::now_text;
    return p;
}

int run(const std::vector<std::string>& args)
{
    std::string config;
    cg::EngineOptions opt;
    opt.workers = static_cast<int>(std::max(1u, std::thread::hardware_concurrency() / 2));
    std::filesystem::path registry;
    bool quiet = false;
    for (std::size_t k = 0; k < args.size(); ++k) {
        const std::string& a = args[k];
        auto value = [&]() -> const std::string& {
            if (k + 1 >= args.size()) throw std::invalid_argument(a + " needs a value");
            return args[++k];
        };
        if (a == "--workers") opt.workers = std::stoi(value());
        else if (a == "--shard") {
            const std::string v = value();
            const auto slash = v.find('/');
            if (slash == std::string::npos) throw std::invalid_argument("--shard i/n");
            opt.shard = std::make_pair(std::stoll(v.substr(0, slash)), std::stoll(v.substr(slash + 1)));
            if (opt.shard->second < 1 || opt.shard->first < 0 || opt.shard->first >= opt.shard->second)
                throw std::invalid_argument("--shard i/n: 0 <= i < n");
        } else if (a == "--only") {
            std::vector<std::int64_t> only;
            std::stringstream s(value());
            for (std::string i; std::getline(s, i, ',');)
                if (!i.empty()) only.push_back(std::stoll(i));
            opt.only = only;
        } else if (a == "--force") opt.force = true;
        else if (a == "--registry") registry = mootation::store::file::path_of(value());
        else if (a == "--quiet") quiet = true;
        else if (!a.empty() && a[0] == '-') throw std::invalid_argument("unknown option " + a);
        else if (config.empty()) config = a;
        else throw std::invalid_argument("one campaign at a time: " + a);
    }
    if (config.empty()) throw std::invalid_argument("run: which campaign?");
    if (registry.empty()) registry = newest_registry();
    const cg::ProblemRegistry preg = cg::load_registry(registry);
    const auto source = mootation::store::file::path_of(config);
    std::ifstream in(source, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + config);
    std::stringstream text;
    text << in.rdbuf();
    opt.process = process_fields();
    opt.pid = process_id();
    opt.host = host_name();
    if (!quiet) opt.say = [](const std::string& line) { std::fprintf(stderr, "%s\n", line.c_str()); };
    const cg::EngineCounts c = cg::run_campaign(text.str(), source, preg, opt);
    std::size_t left = 0;
    for (const auto& [why, n] : c.left) left += n;
    std::fprintf(stderr, "finished: done %zu, failed %zu, skipped %zu, left to the Python engine %zu\n", c.done,
                 c.failed, c.skipped, left);
    for (const auto& [why, n] : c.left) std::fprintf(stderr, "  %zu: %s\n", n, why.c_str());
    return c.failed ? 1 : 0;
}

// the keys a campaign plans (its campaign.json names its config), or nothing
std::optional<std::vector<std::string>> planned_keys(const std::filesystem::path& camp, const cg::ProblemRegistry& preg)
{
    try {
        const js::Value info = js::parse(cg::detail::read_all(camp / "campaign.json"));
        const std::string config = info.find("config")->s;
        const auto source = mootation::store::file::path_of(config);
        const cg::Registry reg = cg::config_registry(preg);
        const cg::JobsRegistry jreg = cg::jobs_registry(preg);
        cg::Config cfg = cg::loads(cg::detail::read_all(source), reg, source);
        cg::validate(cfg, reg);
        std::vector<std::string> keys;
        for (const auto& j : cg::expand_jobs(cfg, cg::campaign_spec(cfg, jreg), jreg)) keys.push_back(cg::job_key(j));
        return keys;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

const char* kStoreUsage =
    "usage: mootation store ls|check|repair <campaign folder>\n"
    "       mootation store compact <campaign folder> [--remove-segments] [--registry FILE]\n"
    "       mootation store merge <campaign folder> <part folder>...\n"
    "       (export: python -m mootation.store export ...)\n";

int store_command(const std::vector<std::string>& args)
{
    if (args.size() < 2) throw std::invalid_argument(std::string("store: which command and campaign?\n") + kStoreUsage);
    const std::string cmd = args[0];
    const auto camp = mootation::store::file::path_of(args[1]);
    const auto dir = camp / cg::kStoreDir;
    bool remove_segments = false;
    std::filesystem::path registry;
    std::vector<std::filesystem::path> parts;
    for (std::size_t k = 2; k < args.size(); ++k) {
        if (args[k] == "--remove-segments") remove_segments = true;
        else if (args[k] == "--registry" && k + 1 < args.size()) registry = mootation::store::file::path_of(args[++k]);
        else if (cmd == "merge") parts.push_back(mootation::store::file::path_of(args[k]) / cg::kStoreDir);
        else throw std::invalid_argument("store " + cmd + ": unknown argument " + args[k]);
    }
    if (cmd == "ls") {
        if (registry.empty()) registry = newest_registry();
        const cg::ProblemRegistry preg = cg::load_registry(registry);
        std::map<std::string, std::map<std::string, std::size_t>> by;
        const auto idx = cg::store_index(dir);
        for (const auto& [key, e] : idx) {
            const std::string problem = cg::KeyOrder::parts(key)[0];
            const auto it = preg.problems.find(problem);
            ++by[it != preg.problems.end() ? it->second.family : problem][e.status];
        }
        for (const auto& [fam, st] : by) {
            std::string line;
            for (const auto& [s, n] : st) line += (line.empty() ? "" : ", ") + s + " " + std::to_string(n);
            std::printf("%-16s %s\n", fam.c_str(), line.c_str());
        }
        std::vector<std::filesystem::path> packs, segs;
        cg::store_files(dir, packs, segs);
        std::uintmax_t size = 0;
        std::error_code ec;
        for (const auto& f : std::filesystem::directory_iterator(dir, ec))
            if (f.is_regular_file()) size += f.file_size();
        std::printf("%zu runs, %zu segments, %zu packs, %.1f MiB\n", idx.size(), segs.size(), packs.size(),
                    static_cast<double>(size) / (1 << 20));
        return 0;
    }
    if (cmd == "check") {
        const cg::StoreCheck c = cg::check_store(dir);
        js::Value o = js::Value::object(), d = js::Value::array(), t = js::Value::array();
        for (const auto& s : c.damaged) d.items.push_back(js::Value::string(s));
        for (const auto& s : c.tails) t.items.push_back(js::Value::string(s));
        o.set("records", js::Value::integer(static_cast<std::int64_t>(c.records)));
        o.set("damaged", d);
        o.set("tails", t);
        o.set("duplicates", js::Value::integer(static_cast<std::int64_t>(c.duplicates)));
        std::printf("%s\n", js::dump(o, 1).c_str());
        return c.damaged.empty() && c.tails.empty() ? 0 : 1;
    }
    if (cmd == "repair") {
        const auto done = cg::repair_store(dir);
        if (done.empty()) std::printf("nothing to repair\n");
        for (const auto& line : done) std::printf("%s\n", line.c_str());
        return 0;
    }
    if (cmd == "compact") {
        if (registry.empty()) registry = newest_registry();
        const cg::ProblemRegistry preg = cg::load_registry(registry);
        auto family_of = [&preg](const std::string& problem) {
            const auto it = preg.problems.find(problem);
            if (it == preg.problems.end() || it->second.family.empty())
                throw std::runtime_error(problem + ": not in the problem registry (its family is the pack's)");
            return it->second.family;
        };
        const cg::Compacted c = cg::compact_store(dir, family_of, remove_segments, planned_keys(camp, preg), process_id());
        js::Value o = js::Value::object(), packs = js::Value::object();
        for (const auto& [fam, n] : c.packs) packs.set(fam, js::Value::integer(static_cast<std::int64_t>(n)));
        o.set("packs", packs);
        o.set("removed", js::Value::integer(static_cast<std::int64_t>(c.removed.size())));
        o.set("missing", js::Value::integer(static_cast<std::int64_t>(c.missing.size())));
        o.set("failed", js::Value::integer(static_cast<std::int64_t>(c.failed.size())));
        std::printf("%s\n", js::dump(o, 1).c_str());
        for (std::size_t k = 0; k < c.missing.size() && k < 50; ++k) std::printf("missing: %s\n", c.missing[k].c_str());
        return 0;
    }
    if (cmd == "merge") {
        if (parts.empty()) throw std::invalid_argument("store merge: which parts?");
        const auto [runs, copied] = cg::merge_stores(dir, parts, process_id(), host_name());
        std::printf("{\n \"runs\": %zu,\n \"copied\": %zu\n}\n", runs, copied);
        return 0;
    }
    throw std::invalid_argument("store: unknown command " + cmd + "\n" + kStoreUsage);
}

}   // namespace

int main(int argc, char** argv)
{
    std::vector<std::string> args(argv + 1, argv + argc);
    try {
        if (args.empty() || args[0] == "-h" || args[0] == "--help") {
            std::fputs(kUsage, args.empty() ? stderr : stdout);
            return args.empty() ? 2 : 0;
        }
        if (args[0] == "--version") {
            std::printf("mootation %s (%s%s)\n", mootation::version(), MOOTATION_GIT_COMMIT,
                        MOOTATION_GIT_DIRTY ? ", dirty" : "");
            return 0;
        }
        if (args[0] == "run") return run(std::vector<std::string>(args.begin() + 1, args.end()));
        if (args[0] == "store") return store_command(std::vector<std::string>(args.begin() + 1, args.end()));
        std::fprintf(stderr, "unknown command %s\n%s", args[0].c_str(), kUsage);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mootation: %s\n", e.what());
        return 1;
    }
}
