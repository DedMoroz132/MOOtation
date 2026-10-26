// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// mootation — the campaign engine in C++ (task 17, stage 5, step 6).
//
//     mootation run <campaign.toml> [--workers N] [--shard i/n] [--only i,j,...]
//                   [--force] [--registry FILE] [--quiet]
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
        std::fprintf(stderr, "unknown command %s\n%s", args[0].c_str(), kUsage);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mootation: %s\n", e.what());
        return 1;
    }
}
