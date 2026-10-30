#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// What a run is measured against, read in C++ — task 17, stage 5, step 4a:
// the cache of references (run/refs.py: <root>/v<REFERENCE_VERSION>/<problem>/
// n<n_ref>.<ext>) and campaign.reference_hash() over it.
//
// An entry holds the frame (ideal, nadir), the reference front (ref), the
// dense reference (ref3) and the Pareto-set sample (pset) — each where the
// problem has it — and json.dumps of its meta (refs.build_entry).
//
// From reference version 5 (§2.1: the references built in C++, refs/) an
// entry is one record of the store's format (store/record.hpp, the key
// "<problem>|<n_ref>"): the block meta.json (its JSON text), then the arrays
// as <f8 blocks under their names — n<n_ref>.mrf. The entries of before, by
// Python, are .npz files (numpy.savez_compressed; "meta" json.dumps of the
// meta as a 0-d string array): read, never written.
// ============================================================================

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "../io/npz.hpp"
#include "../json.hpp"
#include "../store/pack.hpp"
#include "../store/record.hpp"
#include "../store/run_json.hpp"
#include "../store/sha256.hpp"

namespace mootation::campaign {

// a float64 array with its shape (the cache's arrays are 1-d or 2-d)
struct Matrix {
    std::vector<std::size_t> shape;
    std::vector<double> values;          // C order

    std::size_t rows() const { return shape.empty() ? 1 : shape[0]; }
    std::size_t cols() const { return shape.size() < 2 ? 1 : shape[1]; }
};

struct ReferenceEntry {
    std::optional<Matrix> ideal, nadir, ref, ref3, pset;
    json::Value meta;
};

namespace detail {

inline std::string getenv_str(const char* name)
{
#if defined(_MSC_VER)
    char* v = nullptr;
    std::size_t n = 0;
    std::string out;
    if (_dupenv_s(&v, &n, name) == 0 && v) out = v;
    std::free(v);
    return out;
#else
    const char* v = std::getenv(name);
    return v ? v : "";
#endif
}

}   // namespace detail

// refs.refs_root(): MOOTATION_REFS_DIR, else the user's cache folder
inline std::filesystem::path refs_root()
{
    const std::string env = detail::getenv_str("MOOTATION_REFS_DIR");
    if (!env.empty()) return store::file::path_of(env);
#if defined(_WIN32)
    std::string base = detail::getenv_str("LOCALAPPDATA");
    std::filesystem::path root = base.empty() ? store::file::path_of(detail::getenv_str("USERPROFILE")) / "AppData" / "Local"
                                              : store::file::path_of(base);
#else
    std::string base = detail::getenv_str("XDG_CACHE_HOME");
    std::filesystem::path root = base.empty() ? std::filesystem::path(detail::getenv_str("HOME")) / ".cache"
                                              : std::filesystem::path(base);
#endif
    return root / "mootation" / "references";
}

// the first reference version whose entries are records (.mrf)
constexpr int kRecordEntries = 5;

// refs.entry_path()
inline std::filesystem::path entry_path(int reference_version, const std::string& name, long long n_ref)
{
    return refs_root() / ("v" + std::to_string(reference_version)) / store::file::path_of(name) /
           ("n" + std::to_string(n_ref) + (reference_version >= kRecordEntries ? ".mrf" : ".npz"));
}

inline Matrix matrix_of(const npz::Array& a)
{
    return Matrix{a.shape, a.doubles()};
}

namespace detail {

inline const std::pair<const char*, std::optional<Matrix> ReferenceEntry::*> kArrays[] = {
    {"ideal", &ReferenceEntry::ideal}, {"nadir", &ReferenceEntry::nadir}, {"ref", &ReferenceEntry::ref},
    {"ref3", &ReferenceEntry::ref3},   {"pset", &ReferenceEntry::pset}};

}   // namespace detail

// An entry as the bytes of its record (the key "<problem>|<n_ref>").
inline std::vector<std::uint8_t> encode_entry(const std::string& key, const ReferenceEntry& e)
{
    std::vector<store::Block> blocks;
    const std::string meta = json::dump(e.meta);
    blocks.push_back(store::Block{"meta.json", "json", {}, {meta.begin(), meta.end()}, ""});
    for (const auto& [name, member] : detail::kArrays) {
        const std::optional<Matrix>& a = e.*member;
        if (!a) continue;
        store::Block b{name, "<f8", {}, {}, ""};
        for (auto s : a->shape) b.shape.push_back(s);
        const auto* p = reinterpret_cast<const std::uint8_t*>(a->values.data());
        b.data.assign(p, p + 8 * a->values.size());
        blocks.push_back(std::move(b));
    }
    return store::encode_record(key, store::encode_run(blocks));
}

inline ReferenceEntry decode_entry(const std::vector<std::uint8_t>& bytes, const std::string& what)
{
    const store::RecordAt r = store::parse_record(bytes.data(), bytes.size(), 0);
    if (!r.ok) throw std::runtime_error(what + ": not a whole record (length or CRC)");
    ReferenceEntry e;
    for (const store::Block& b : store::decode_run(bytes.data() + r.body, r.body_size)) {
        if (b.name == "meta.json") {
            e.meta = json::parse(std::string(b.data.begin(), b.data.end()));
            continue;
        }
        for (const auto& [name, member] : detail::kArrays) {
            if (b.name != name) continue;
            Matrix m;
            for (auto s : b.shape) m.shape.push_back(static_cast<std::size_t>(s));
            m.values.resize(b.data.size() / 8);
            std::memcpy(m.values.data(), b.data.data(), 8 * m.values.size());
            e.*member = std::move(m);
        }
    }
    return e;
}

// An entry written whole: to a file beside it, renamed over it (a copy built
// at the same time elsewhere is the same bytes).
inline void save_entry(const std::filesystem::path& path, const std::string& key, const ReferenceEntry& e,
                       long long pid)
{
    std::filesystem::create_directories(path.parent_path());
    const auto bytes = encode_entry(key, e);
    auto tmp = path;
    tmp += "." + std::to_string(pid) + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!out) throw std::runtime_error("cannot write " + tmp.string());
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {                                        // Windows: another process holds the copy it wrote
        std::filesystem::remove(tmp, ec);
        if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("cannot write " + path.string());
    }
}

// an entry's arrays and meta (refs.load_entry), from its file
inline ReferenceEntry load_entry(const std::filesystem::path& path)
{
    ReferenceEntry e;
    if (path.extension() == ".mrf") return decode_entry(npz::read_file(path), path.string());
    for (const auto& [name, a] : npz::load(path)) {
        if (name == "meta") e.meta = json::parse(a.text());
        else if (name == "ideal") e.ideal = matrix_of(a);
        else if (name == "nadir") e.nadir = matrix_of(a);
        else if (name == "ref") e.ref = matrix_of(a);
        else if (name == "ref3") e.ref3 = matrix_of(a);
        else if (name == "pset") e.pset = matrix_of(a);
    }
    return e;
}

// str(ndarray.shape) as Python writes it: (), (3,), (1000, 3)
inline std::string shape_text(const std::vector<std::size_t>& shape)
{
    std::string s = "(";
    for (std::size_t k = 0; k < shape.size(); ++k) s += (k ? ", " : "") + std::to_string(shape[k]);
    if (shape.size() == 1) s += ",";
    return s + ")";
}

// campaign.reference_hash(): the first 16 hex digits of the SHA-256 of each
// array's name, then its shape's text and bytes ("none" where absent)
inline std::string reference_hash(const std::optional<Matrix>& ideal, const std::optional<Matrix>& nadir,
                                  const std::optional<Matrix>& ref, const std::optional<Matrix>& ref3,
                                  const std::optional<Matrix>& pset)
{
    std::vector<std::uint8_t> buf;
    auto put = [&buf](const std::string& s) { buf.insert(buf.end(), s.begin(), s.end()); };
    const std::pair<const char*, const std::optional<Matrix>*> parts[] = {
        {"ideal", &ideal}, {"nadir", &nadir}, {"ref", &ref}, {"ref3", &ref3}, {"pset", &pset}};
    for (const auto& [name, a] : parts) {
        put(name);
        if (!*a) {
            put("none");
            continue;
        }
        put(shape_text((*a)->shape));
        const auto* p = reinterpret_cast<const std::uint8_t*>((*a)->values.data());
        buf.insert(buf.end(), p, p + 8 * (*a)->values.size());
    }
    const auto h = store::sha256(buf.data(), buf.size());
    static const char* hex = "0123456789abcdef";
    std::string out;
    for (std::size_t k = 0; k < 8; ++k) {
        out += hex[h[k] >> 4];
        out += hex[h[k] & 15];
    }
    return out;
}

}   // namespace mootation::campaign
