#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A campaign's store looked after in C++ — task 17, stage 5, step 6:
// mootation/store.py's check, compact and merge, for `mootation store`.
//
//   check    every record read back (length, key hash, CRC; a pack by
//            check_pack): the records, the damaged ones and where, the
//            unfinished tails, the runs a segment holds twice
//   compact  one pack per family (the registry's family of a problem), its
//            records sorted by key; a record of format 3 gives its meta.json
//            to the pack's meta table and keeps a mark; the metrics table;
//            the segments go only on request, and only after every record
//            read back from its pack is the segment's
//   merge    the stores of a campaign's parts into one, a run in two parts
//            the same blocks in both
//
// export (a run's files, the CSV of the metrics) stays Python's: it writes
// gzip, which the library does not.
// ============================================================================

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "../io/npz.hpp"
#include "../store/meta_table.hpp"
#include "../store/metrics_table.hpp"
#include "../store/pack.hpp"
#include "../store/record.hpp"
#include "../store/run_json.hpp"
#include "engine.hpp"

namespace mootation::campaign {

// store.sort_key: problem, algorithm, variant, then seed and budget as numbers
struct KeyOrder {
    static std::vector<std::string> parts(const std::string& key)
    {
        std::vector<std::string> p;
        std::size_t at = 0;
        for (;;) {
            const auto bar = key.find('|', at);
            p.push_back(key.substr(at, bar == std::string::npos ? std::string::npos : bar - at));
            if (bar == std::string::npos) break;
            at = bar + 1;
        }
        return p;
    }
    bool operator()(const std::string& a, const std::string& b) const
    {
        const auto x = parts(a), y = parts(b);
        if (x.size() != 5 || y.size() != 5) return a < b;
        for (int k = 0; k < 3; ++k)
            if (x[k] != y[k]) return x[k] < y[k];
        for (int k = 3; k < 5; ++k) {
            const long long u = std::stoll(x[k]), v = std::stoll(y[k]);
            if (u != v) return u < v;
        }
        return false;
    }
};

// a record's bytes, and its blocks as store.read_record gives them
inline std::string record_bytes(const StoreEntry& e)
{
    std::ifstream in(e.file, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(e.offset));
    std::string out(static_cast<std::size_t>(e.length), '\0');
    in.read(&out[0], static_cast<std::streamsize>(e.length));
    return out;
}

// a pack's meta table: key -> its run's meta.json (the .mpk's meta part, a
// pack of before's .meta file); empty where it has none
inline std::map<std::string, std::vector<std::uint8_t>> pack_metas(const std::filesystem::path& pack)
{
    std::vector<std::uint8_t> data;
    if (pack.extension() == ".mpk") {
        std::ifstream in(pack, std::ios::binary);
        in.seekg(0, std::ios::end);
        const std::uint64_t size = static_cast<std::uint64_t>(in.tellg());
        std::uint8_t foot[64];
        in.seekg(static_cast<std::streamoff>(size - 64));
        in.read(reinterpret_cast<char*>(foot), 64);
        const std::uint64_t at = store::rec::get_le(foot + 8, 8), n = store::rec::get_le(foot + 16, 8);
        data.resize(static_cast<std::size_t>(n));
        in.seekg(static_cast<std::streamoff>(at));
        in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(n));
    } else {
        auto meta = pack;
        meta.replace_extension(".meta");
        const std::string s = detail::read_all(meta);
        data.assign(s.begin(), s.end());
    }
    std::map<std::string, std::vector<std::uint8_t>> out;
    if (data.empty()) return out;
    for (auto& [key, bytes] : store::decode_meta_table(data.data(), data.size())) out[key] = std::move(bytes);
    return out;
}

// the packs' meta tables read once
struct MetaCache {
    std::map<std::string, std::map<std::string, std::vector<std::uint8_t>>> by_file;
};

// zlib's stream (a record of format 2): the two bytes of its header, the
// deflate data, the four of its checksum
inline std::vector<std::uint8_t> zlib_inflate(const std::vector<std::uint8_t>& z)
{
    if (z.size() < 6) throw std::runtime_error("a zlib block shorter than its frame");
    return npz::inflate(z.data() + 2, z.size() - 6);
}

// store.read_record: the blocks of the record at e, a block of format 2
// decompressed, a pack's record with its meta.json put back at its mark
inline std::vector<store::Block> entry_blocks(const StoreEntry& e, MetaCache& cache)
{
    const std::string rec = record_bytes(e);
    const auto* buf = reinterpret_cast<const std::uint8_t*>(rec.data());
    const store::RecordAt r = store::parse_record(buf, rec.size(), 0);
    if (!r.ok) throw std::runtime_error(detail::utf8(e.file.filename()) + "@" + std::to_string(e.offset) +
                                        ": not a whole record (length or CRC)");
    std::vector<store::Block> blocks = store::decode_run(buf + r.body, r.body_size);
    bool mark = false;
    for (auto& b : blocks) {
        if (b.how == "zlib") {
            b.data = zlib_inflate(b.data);
        } else if (b.how == "shuffle+zlib") {
            const store::rec::Dtype d = store::rec::dtype_of(b.dtype);
            b.data = store::rec::unshuffle(zlib_inflate(b.data), d.array ? d.size : 1);
        }
        b.how.clear();
        mark = mark || b.name == "#meta";
    }
    if (!mark) return blocks;
    const std::string file = detail::utf8(e.file);
    auto it = cache.by_file.find(file);
    if (it == cache.by_file.end()) it = cache.by_file.emplace(file, pack_metas(e.file)).first;
    const auto m = it->second.find(e.key);
    if (m == it->second.end())
        throw std::runtime_error(detail::utf8(e.file.filename()) + ": " + e.key + ": its meta.json is not in the pack's table");
    for (auto& b : blocks)
        if (b.name == "#meta") b = store::Block{"meta.json", "json", {}, m->second, ""};
    return store::final_back(std::move(blocks));
}

inline bool same_blocks(const std::vector<store::Block>& a, const std::vector<store::Block>& b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t k = 0; k < a.size(); ++k)
        if (a[k].name != b[k].name || a[k].dtype != b[k].dtype || a[k].shape != b[k].shape || a[k].data != b[k].data)
            return false;
    return true;
}

struct StoreCheck {
    std::size_t records = 0, duplicates = 0;
    std::vector<std::string> damaged, tails;
};

// store.check
inline StoreCheck check_store(const std::filesystem::path& store_dir)
{
    StoreCheck out;
    std::vector<std::filesystem::path> packs, segs;
    store_files(store_dir, packs, segs);
    std::set<std::string> seen;
    for (const auto* list : {&packs, &segs})
        for (const auto& raw : *list) {
            const std::string name = detail::utf8(raw.filename());
            if (raw.extension() == ".mpk") {
                const std::string data = detail::read_all(raw);
                const auto bad = store::check_pack(reinterpret_cast<const std::uint8_t*>(data.data()), data.size(), name);
                out.damaged.insert(out.damaged.end(), bad.begin(), bad.end());
                if (bad.empty()) out.records += pack_index(raw).size();
                continue;
            }
            std::uint64_t end = 0;
            const auto entries = scan_raw(detail::read_all(raw), &end);
            std::error_code ec;
            const auto size = std::filesystem::file_size(raw, ec);
            if (end < size) out.tails.push_back(name + ": " + std::to_string(size - end) + " bytes after the last whole record");
            const auto listed = file_entries(raw);
            if (listed.size() != entries.size())
                out.damaged.push_back(name + ": index " + std::to_string(listed.size()) + " records, data " +
                                      std::to_string(entries.size()));
            for (const auto& e : entries) {
                ++out.records;
                if (name.rfind("seg-", 0) == 0) {
                    out.duplicates += seen.count(e.key);
                    seen.insert(e.key);
                }
            }
        }
    return out;
}

// store._safe: a family's name in a file's
inline std::string safe_name(const std::string& s)
{
    std::string out;
    for (char c : s) out += (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.') ? c : '_';
    return out;
}

struct Compacted {
    std::map<std::string, std::size_t> packs;    // family -> records
    std::vector<std::string> removed, missing, failed;
};

// store.compact. family_of: a problem's family (the registry's); expected:
// the keys the campaign plans, for the list of the missing
inline Compacted compact_store(const std::filesystem::path& store_dir,
                               const std::function<std::string(const std::string&)>& family_of,
                               bool remove_segments, const std::optional<std::vector<std::string>>& expected,
                               long long pid)
{
    const StoreCheck bad = check_store(store_dir);
    if (!bad.damaged.empty()) {
        std::string list;
        for (std::size_t k = 0; k < bad.damaged.size() && k < 10; ++k) list += (k ? "; " : "") + bad.damaged[k];
        throw std::runtime_error("damaged records, run repair first: " + list);
    }
    std::error_code ec;
    std::filesystem::remove_all(store_dir / kScratchDir, ec);
    const auto idx = store_index(store_dir);
    std::map<std::string, std::vector<std::string>> fams;
    std::vector<std::string> keys;
    for (const auto& [key, e] : idx) keys.push_back(key);
    std::sort(keys.begin(), keys.end(), KeyOrder{});
    for (const auto& key : keys) fams[family_of(KeyOrder::parts(key)[0])].push_back(key);
    Compacted out;
    MetaCache metas;
    std::map<std::string, StoreEntry> packed;          // key -> its record in the new pack
    for (const auto& [fam, fkeys] : fams) {
        const auto path = store_dir / ("pack-" + safe_name(fam) + ".mpk");
        std::vector<store::PackRecord> records;
        std::vector<std::pair<std::string, std::vector<std::uint8_t>>> table;
        std::vector<std::pair<std::string, std::vector<store::Block>>> runs;
        std::set<std::string> stamps, versions;
        for (const auto& key : fkeys) {
            const StoreEntry& e = idx.at(key);
            const std::string rec = record_bytes(e);
            const auto blocks = entry_blocks(e, metas);
            runs.emplace_back(key, blocks);
            const store::Block* meta = nullptr;
            const store::Block* fmt = nullptr;
            for (const auto& b : blocks) {
                if (b.name == "meta.json") meta = &b;
                if (b.name == "#format") fmt = &b;
            }
            int format = 1;
            if (fmt) {
                const json::Value v = json::parse(std::string(fmt->data.begin(), fmt->data.end()));
                if (const json::Value* x = v.find("version")) format = static_cast<int>(x->i);
            }
            std::vector<std::uint8_t> bytes(rec.begin(), rec.end());
            if (format >= 3 && meta) {
                // meta.json into the pack's table (as final_once keeps it), a mark in its place
                for (const auto& b : store::final_once(blocks))
                    if (b.name == "meta.json") table.emplace_back(key, b.data);
                std::vector<store::Block> marked;
                for (const auto& b : blocks)
                    marked.push_back(b.name == "meta.json" ? store::Block{"#meta", "json", {}, {}, ""} : b);
                bytes = store::encode_record(key, store::encode_run(marked));
            }
            records.push_back({key, std::move(bytes), e.status});
            if (meta) {
                const json::Value m = json::parse(std::string(meta->data.begin(), meta->data.end()));
                auto text = [](const json::Value* v) { return v->kind == json::Value::Kind::String ? v->s : json::dump(*v); };
                if (const json::Value* v = m.find("code_stamp"); v && v->kind != json::Value::Kind::Null) stamps.insert(text(v));
                if (const json::Value* v = m.find("trajectory_version"); v && v->kind != json::Value::Kind::Null)
                    versions.insert(text(v));
            }
        }
        auto joined = [](const std::set<std::string>& s) {
            std::string r;
            for (const auto& x : s) r += (r.empty() ? "" : ",") + x;
            return r;
        };
        // the header's "key = value" lines, as compact writes them
        const std::string header = "format = mootation-store\nformat_version = " + std::to_string(kStoreFormatVersion) +
                                   "\nkind = pack\nfamily = " + fam + "\nrecords = " + std::to_string(records.size()) +
                                   "\nsorted_by = problem, algorithm, variant, seed, budget\ncode_stamp = " +
                                   joined(stamps) + "\ntrajectory_version = " + joined(versions) +
                                   "\nlayout = header, meta, metrics, records, index; a footer of 64 bytes at the end"
                                   "\nalignment = 64\nbyte_order = little\n";
        const auto meta_table = table.empty() ? std::vector<std::uint8_t>{} : store::encode_meta_table(table);
        const auto metrics = store::encode_metrics_table(fam, runs);
        const std::string tmp = detail::utf8(path) + "." + std::to_string(pid) + ".tmp";
        const auto entries = store::write_pack(detail::utf8(path), tmp, header, meta_table, metrics, records);
        out.packs[fam] = entries.size();
        for (const auto& e : entries) packed[e.key] = StoreEntry{e.key, e.status, e.offset, e.length, path};
        for (const char* stem : {"pack-", "metrics-"})
            for (const char* suffix : {".raw", ".idx", ".mhd", ".meta"})
                std::filesystem::remove(store_dir / (std::string(stem) + safe_name(fam) + suffix), ec);
    }
    if (remove_segments) {
        // every segment's record against its record in the new pack (not the
        // store's index, which still gives the segment's while it is there)
        MetaCache fresh;
        for (const auto& [key, e] : idx) {
            if (detail::utf8(e.file.filename()).rfind("seg-", 0) != 0) continue;
            if (!same_blocks(entry_blocks(packed.at(key), fresh), entry_blocks(e, fresh)))
                throw std::runtime_error(key + ": the pack's record is not the segment's; nothing removed");
        }
        std::vector<std::filesystem::path> packs, segs;
        store_files(store_dir, packs, segs);
        for (const auto& raw : segs) {
            for (const char* suffix : {".raw", ".idx", ".mhd"}) {
                auto p = raw;
                p.replace_extension(suffix);
                std::filesystem::remove(p, ec);
            }
            out.removed.push_back(detail::utf8(raw.filename()));
        }
        std::ofstream log(store_dir / kStoreLog, std::ios::binary | std::ios::app);
        log << detail::now_text() << " compact: removed " << out.removed.size()
            << " segments after checking every record" << (kTextCrlf ? "\r\n" : "\n");
    }
    if (expected)
        for (const auto& k : *expected)
            if (!idx.count(k)) out.missing.push_back(k);
    std::sort(out.missing.begin(), out.missing.end());
    for (const auto& [key, e] : idx)
        if (e.status != "done") out.failed.push_back(key);
    return out;
}

// store.merge: the parts' stores into `dst`; a run in two parts must be the
// same blocks in both; returns (runs, copied)
inline std::pair<std::size_t, std::size_t> merge_stores(const std::filesystem::path& dst,
                                                        const std::vector<std::filesystem::path>& parts, long long pid,
                                                        const std::string& host)
{
    std::filesystem::create_directories(dst);
    std::map<std::string, std::vector<store::Block>, KeyOrder> have;
    MetaCache metas;
    for (const auto& [key, e] : store_index(dst)) have.emplace(key, entry_blocks(e, metas));
    const auto mine = have;
    std::vector<std::string> clash;
    for (const auto& part : parts)
        for (const auto& [key, e] : store_index(part)) {
            auto blocks = entry_blocks(e, metas);
            auto it = have.find(key);
            if (it != have.end() && !same_blocks(it->second, blocks)) clash.push_back(key + ": " + detail::utf8(part) + " differs");
            if (it == have.end()) have.emplace(key, std::move(blocks));
        }
    if (!clash.empty()) {
        std::string list;
        for (std::size_t k = 0; k < clash.size() && k < 20; ++k) list += (k ? "; " : "") + clash[k];
        throw std::runtime_error(std::to_string(clash.size()) + " runs differ between the parts: " + list);
    }
    std::string from;
    for (const auto& p : parts) from += (from.empty() ? "" : ",") + detail::utf8(p);
    SegmentWriter w(dst, std::uint64_t{1} << 30, pid, host, {{"merged_from", from}});
    std::size_t copied = 0;
    for (const auto& [key, blocks] : have) {
        if (mine.count(key)) continue;
        w.append(key, blocks);
        ++copied;
    }
    return {have.size(), copied};
}

}   // namespace mootation::campaign
