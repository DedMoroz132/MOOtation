#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// What a run is measured against, read in C++ — task 17, stage 5, step 4a:
// the cache of references (run/refs.py: <root>/v<REFERENCE_VERSION>/<problem>/
// n<n_ref>.npz, built by Python) and campaign.reference_hash() over it.
//
// An entry holds the frame (ideal, nadir), the reference front (ref), the
// dense reference (ref3) and the Pareto-set sample (pset) — each where the
// problem has it — and json.dumps of its meta (refs.build_entry).
// ============================================================================

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "../io/npz.hpp"
#include "../json.hpp"
#include "../store/pack.hpp"
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

// refs.entry_path()
inline std::filesystem::path entry_path(int reference_version, const std::string& name, long long n_ref)
{
    return refs_root() / ("v" + std::to_string(reference_version)) / store::file::path_of(name) /
           ("n" + std::to_string(n_ref) + ".npz");
}

inline Matrix matrix_of(const npz::Array& a)
{
    return Matrix{a.shape, a.doubles()};
}

// an entry's arrays and meta (refs.load_entry), from its file
inline ReferenceEntry load_entry(const std::filesystem::path& path)
{
    ReferenceEntry e;
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
