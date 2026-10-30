#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The cache of references built in C++ — task 17, §2.1: run/refs.py's build,
// check and status for `mootation refs build|check` (engine/mootation.cpp),
// the entries those of reference version 5 (refs/build.hpp; records,
// references.hpp).
//
// A problem is built here when the registry's export gives its front's
// sampler, and its Pareto set's where it has a set (benchmarks/front_specs.py);
// the others are named, with that reason.
//
// An entry's meta, in refs.build_entry's order: problem, n_ref, n_obj,
// problem_version, reference_version, code_stamp (the builder's: "cxx " and
// the library's version and commit), mootation, reference_hash, rows,
// distinct, info3, floors {"<pop>": {greedy, dss}}, extends (with --extend),
// seconds.
// ============================================================================

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "../refs/build.hpp"
#include "../refs/sets.hpp"
#include "../suite/instances.hpp"
#include "engine.hpp"
#include "references.hpp"
#include "registry.hpp"

namespace mootation::campaign {

// refs.SLOW_FIRST: the families whose references take longest, built first
inline const std::vector<std::string>& slow_first()
{
    static const std::vector<std::string> v{"rZCAT", "ZCAT", "DAS-CMOP", "rWFG", "WFG", "MW", "LIR-CMOP", "DOC"};
    return v;
}

// failing.SAMPLE: the base's Pareto-set sample a failing copy keeps the rows of
constexpr std::size_t kFailSample = 2000;

// why C++ does not build the entry of p, or nothing
inline std::optional<std::string> refs_refusal(const RegistryProblem& p, const ProblemRegistry& preg)
{
    if (!p.shared_reference) return "no reference of its own in the cache (no exact front)";
    if (p.failing) {
        if (!p.failing_region) return "a failing copy of a problem outside the C++ suite";
        const auto it = preg.problems.find(p.failing_region->of);
        if (it == preg.problems.end() || it->second.front.kind != json::Value::Kind::Object)
            return "a failing copy whose base's front is not built in C++";
        if (it->second.pset.kind != json::Value::Kind::Object && it->second.family != "WFG")
            return "a failing copy whose base's Pareto set is not built in C++";
        return std::nullopt;
    }
    if (p.front.kind != json::Value::Kind::Object) return "its front's sampler is not in C++ yet";
    if (p.has_pareto_set && p.pset.kind != json::Value::Kind::Object) return "its Pareto set's sampler is not in C++ yet";
    if (p.ideal.size() != static_cast<std::size_t>(p.n_obj) || p.nadir.size() != static_cast<std::size_t>(p.n_obj))
        return "no frame in the registry's export";
    return std::nullopt;
}

// refs.status: "ok", "missing", or why the entry is old
inline std::string refs_status(const std::filesystem::path& path, const RegistryProblem& p, int reference_version)
{
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) return "missing";
    json::Value meta;
    try {
        meta = load_entry(path).meta;
    } catch (const std::exception&) {
        return "missing";
    }
    std::string why;
    auto int_of = [&meta](const char* k) -> long long {
        const json::Value* v = meta.find(k);
        return v && v->kind == json::Value::Kind::Int ? v->i : -1;
    };
    if (int_of("problem_version") != p.problem_version)
        why += "problem version " + std::to_string(int_of("problem_version")) + " -> " + std::to_string(p.problem_version);
    if (int_of("reference_version") != reference_version)
        why += std::string(why.empty() ? "" : ", ") + "builder version " + std::to_string(int_of("reference_version")) +
               " -> " + std::to_string(reference_version);
    return why.empty() ? "ok" : "old: " + why;
}

inline Matrix matrix_of_rows(const refs::Rows& R)
{
    return Matrix{{R.n, R.m}, R.v};
}

inline Matrix vector_matrix(const std::vector<double>& v)
{
    return Matrix{{v.size()}, v};
}

// refs._extends: against the largest smaller entry, whether its front is the
// first rows of `ref`
inline json::Value extends_of(const RegistryProblem& p, std::size_t n_ref, const refs::Rows& ref, int version)
{
    const auto dir = entry_path(version, p.name, static_cast<long long>(n_ref)).parent_path();
    long long best = -1;
    std::error_code ec;
    for (const auto& f : std::filesystem::directory_iterator(dir, ec)) {
        const std::string stem = f.path().stem().string();
        if (f.path().extension() != ".mrf" || stem.size() < 2 || stem[0] != 'n') continue;
        if (stem.find_first_not_of("0123456789", 1) != std::string::npos) continue;
        const long long k = std::stoll(stem.substr(1));
        if (k < static_cast<long long>(n_ref) && k > best) best = k;
    }
    if (best < 0) return json::Value::null();
    ReferenceEntry old;
    try {
        old = load_entry(entry_path(version, p.name, best));
    } catch (const std::exception&) {
        return json::Value::null();
    }
    if (!old.ref) return json::Value::null();
    const std::vector<double>& a = old.ref->values;
    const bool prefix = old.ref->cols() == ref.m && a.size() <= ref.v.size() &&
                        std::equal(a.begin(), a.end(), ref.v.begin());
    json::Value out = json::Value::object();
    out.set("n_ref", json::Value::integer(best));
    out.set("prefix", json::Value::boolean(prefix));
    return out;
}

// registry._failing_parts: the base's Pareto-set sample (failing.SAMPLE rows;
// WFG4's — it has no sampler — failing.wfg4_reference_sample: the k
// positions uniform in [0, 2i], the distance variables at 0.35 * 2i, from the
// library's generator seeded with instances::seed of 20261011 and the CRC-32
// of "<copy>:ps"), and the rows outside the region
inline refs::Rows failing_kept(const RegistryProblem& p, const RegistryProblem& base, std::size_t& total)
{
    refs::Rows S;
    if (base.pset.kind == json::Value::Kind::Object) {
        S = refs::pareto_set(base.pset, kFailSample);
    } else {
        const std::size_t k = 2 * static_cast<std::size_t>(base.n_obj - 1), nv = static_cast<std::size_t>(base.n_var);
        const std::string key = p.name + ":ps";
        std::mt19937 g(suite::instances::seed(
            {20261011u, store::crc32_zlib(reinterpret_cast<const std::uint8_t*>(key.data()), key.size())}));
        S = refs::rows_of(kFailSample, nv);
        for (std::size_t i = 0; i < kFailSample; ++i)
            for (std::size_t j = 0; j < nv; ++j)
                S.row(i)[j] = j < k ? rnd::uniform01(g) * (2.0 * static_cast<double>(j + 1)) : 0.35 * (2.0 * static_cast<double>(j + 1));
    }
    total = S.n;
    const auto native = make_native(p);
    const auto* failing = dynamic_cast<const FailingNative*>(native.get());
    std::vector<char> keep(S.n, 1);
    for (std::size_t i = 0; i < S.n; ++i) keep[i] = !failing->inside(S.row(i));
    return refs::detail::keep_rows(S, keep);
}

// refs.build_entry in C++: the entry of p at n_ref (not written)
inline ReferenceEntry build_entry(const RegistryProblem& p, const ProblemRegistry& preg, std::size_t n_ref,
                                  const refs::Data& data, const std::string& code_stamp, const std::string& version,
                                  bool extend)
{
    const auto t0 = std::chrono::steady_clock::now();
    // the frame: the exact one where C++ knows it (addendum 6, §3), else the registry's
    const auto exact = refs::exact_frame(p.front);
    const refs::Frame frame = exact ? *exact : refs::Frame{p.ideal, p.nadir};
    const std::size_t pop = static_cast<std::size_t>(p.pop_size);
    ReferenceEntry e;
    e.ideal = vector_matrix(frame.ideal);
    e.nadir = vector_matrix(frame.nadir);
    refs::Rows ref;
    if (p.failing) {
        // the base's front where no row of the sample fails or every one does,
        // else the base's images of the rows kept; the Pareto set those rows
        const RegistryProblem& base = preg.problems.at(p.failing_region->of);
        std::size_t total = 0;
        const refs::Rows kept = failing_kept(p, base, total);
        if (kept.n == 0 || kept.n == total) {
            ref = refs::front(base.front, n_ref, data);
        } else {
            const auto native = std::shared_ptr<suite::Native>(suite::make(spec_of(p.failing_region->spec)));
            ref = refs::rows_of(kept.n, static_cast<std::size_t>(base.n_obj));
            std::vector<double> g(static_cast<std::size_t>(std::max(native->n_con, 1)));
            for (std::size_t i = 0; i < kept.n; ++i) native->evaluate(kept.row(i), ref.row(i), g.data());
            ref = refs::detail::nondominated(ref);
        }
        e.pset = Matrix{{kept.n, static_cast<std::size_t>(base.n_var)}, kept.v};
    } else {
        ref = refs::front(p.front, n_ref, data);
    }
    e.ref = matrix_of_rows(ref);
    json::Value info3 = json::Value::null();
    if (p.dense_reference) {
        const refs::Dense d = refs::dense_reference(p.front, frame, pop, data);
        e.ref3 = matrix_of_rows(d.F);
        info3 = json::Value::object();
        info3.set("rows", json::Value::integer(static_cast<std::int64_t>(d.F.n)));
        info3.set("step", json::Value::number(d.step));
        info3.set("population_step", json::Value::number(d.population_step));
    }
    if (!p.failing && p.pset.kind == json::Value::Kind::Object) e.pset = matrix_of_rows(refs::pareto_set(p.pset, n_ref));
    const refs::Floors f = refs::igdp_floors(ref, frame, pop);
    json::Value meta = json::Value::object();
    meta.set("problem", json::Value::string(p.name));
    meta.set("n_ref", json::Value::integer(static_cast<std::int64_t>(n_ref)));
    meta.set("n_obj", json::Value::integer(p.n_obj));
    meta.set("problem_version", json::Value::integer(p.problem_version));
    meta.set("reference_version", json::Value::integer(refs::kReferenceVersion));
    meta.set("code_stamp", json::Value::string(code_stamp));
    meta.set("mootation", json::Value::string(version));
    meta.set("reference_hash", json::Value::string(reference_hash(e.ideal, e.nadir, e.ref, e.ref3, e.pset)));
    meta.set("rows", json::Value::integer(static_cast<std::int64_t>(f.rows)));
    meta.set("distinct", json::Value::integer(static_cast<std::int64_t>(f.distinct)));
    meta.set("info3", info3);
    json::Value at = json::Value::object(), floors = json::Value::object();
    at.set("greedy", json::Value::number(f.greedy));
    at.set("dss", json::Value::number(f.dss));
    floors.set(std::to_string(pop), at);
    meta.set("floors", floors);
    if (extend) meta.set("extends", extends_of(p, n_ref, ref, refs::kReferenceVersion));
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    meta.set("seconds", json::Value::number(std::nearbyint(sec * 1000.0) / 1000.0));
    e.meta = std::move(meta);
    return e;
}

struct RefsBuildOptions {
    std::size_t n_ref = 1000, workers = 1;
    bool rebuild = false, extend = false;
    refs::Data data;
    std::string code_stamp, version;
    long long pid = 0;
    std::function<void(const std::string&)> log;      // a line of progress, as refs.build prints it
};

// refs.build: the entries of `names` missing or old (all with rebuild or
// extend), `workers` threads; {name: "built" | "ok" | "failed: ..." | why
// C++ does not build it}
inline std::map<std::string, std::string> build_references(const ProblemRegistry& preg,
                                                           const std::vector<std::string>& names,
                                                           const RefsBuildOptions& opt)
{
    std::map<std::string, std::string> out;
    std::vector<const RegistryProblem*> todo;
    for (const auto& name : names) {
        const auto it = preg.problems.find(name);
        if (it == preg.problems.end()) {
            out[name] = "not in the registry";
            continue;
        }
        if (const auto why = refs_refusal(it->second, preg)) {
            out[name] = "not built: " + *why;
            continue;
        }
        const std::string st = refs_status(entry_path(refs::kReferenceVersion, name, static_cast<long long>(opt.n_ref)),
                                           it->second, refs::kReferenceVersion);
        if (opt.rebuild || opt.extend || st != "ok") todo.push_back(&it->second);
        else out[name] = "ok";
    }
    auto rank = [](const RegistryProblem* p) {
        const auto& s = slow_first();
        return static_cast<std::size_t>(std::find(s.begin(), s.end(), p->family) - s.begin());
    };
    std::stable_sort(todo.begin(), todo.end(), [&rank](const RegistryProblem* a, const RegistryProblem* b) {
        return rank(a) != rank(b) ? rank(a) < rank(b) : a->name < b->name;
    });
    if (todo.empty()) return out;
    if (opt.log)
        opt.log("references: " + std::to_string(todo.size()) + " of " + std::to_string(names.size()) +
                " to build at n_ref = " + std::to_string(opt.n_ref) + " -> " +
                detail::utf8(refs_root() / ("v" + std::to_string(refs::kReferenceVersion))));
    std::mutex lock;
    std::atomic<std::size_t> next{0}, done{0};
    auto work = [&] {
        for (;;) {
            const std::size_t k = next++;
            if (k >= todo.size()) return;
            const RegistryProblem& p = *todo[k];
            const auto t0 = std::chrono::steady_clock::now();
            std::string what = "built";
            try {
                const ReferenceEntry e = build_entry(p, preg, opt.n_ref, opt.data, opt.code_stamp, opt.version, opt.extend);
                save_entry(entry_path(refs::kReferenceVersion, p.name, static_cast<long long>(opt.n_ref)),
                           p.name + "|" + std::to_string(opt.n_ref), e, opt.pid);
            } catch (const std::exception& ex) {
                what = std::string("failed: ") + ex.what();
            }
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::lock_guard<std::mutex> g(lock);
            out[p.name] = what;
            char buf[64];
            std::snprintf(buf, sizeof buf, " (%.1f s)", sec);
            if (opt.log)
                opt.log("  [" + std::to_string(++done) + "/" + std::to_string(todo.size()) + "] " + p.name + ": " + what + buf);
        }
    };
    std::vector<std::thread> pool;
    const std::size_t n = std::max<std::size_t>(1, std::min(opt.workers, todo.size()));
    for (std::size_t t = 1; t < n; ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
    return out;
}

}   // namespace mootation::campaign
