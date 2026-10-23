#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The store's files (task 17, §3а.3: one implementation in C++): a segment's
// record appended and synced, and a family's pack — one file, written whole
// and checked part by part — byte for byte as docs/store-format.md says.
//
// A PACK (pack-<family>.mpk):
//     header    "key = value" lines, then newlines to a multiple of 64
//     meta      the runs' meta.json as a table (the caller's bytes; may be empty)
//     metrics   the metrics table (the caller's bytes)
//     records   each at a multiple of 64 (zero bytes between)
//     index     "T" and the text codec's bytes of the lines
//               "offset \t length \t checksum \t status \t key"
//     footer    64 bytes: "MOOTPEND", the offset and length of meta, metrics
//               and index (u64 each), the CRC-32C of the file before the
//               footer and of the index's bytes (u32 each)
// It is written to a temporary file beside it, synced and renamed.
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#include <share.h>
#else
#include <unistd.h>
#endif

#include "crc32c.hpp"
#include "record.hpp"
#include "text_codec.hpp"

namespace mootation::store {

constexpr std::size_t kAlign = 64;
constexpr char kPackFoot[9] = "MOOTPEND";

// A record's place in a segment or a pack.
struct Entry {
    std::string key;
    std::uint64_t offset = 0, length = 0;
    std::uint32_t crc = 0;
    std::string status;
};

struct PackFooter {
    std::uint64_t meta_at = 0, meta_size = 0, metrics_at = 0, metrics_size = 0, index_at = 0, index_size = 0;
    std::uint32_t crc = 0, index_crc = 0;
    std::uint64_t end = 0;               // where the footer starts
};

namespace file {

// A UTF-8 path as the system's: UTF-16 on Windows (decoded here, the same in
// C++17 and C++20, where u8path is deprecated), the bytes as they are elsewhere.
inline std::filesystem::path path_of(const std::string& utf8)
{
#if defined(_WIN32)
    std::wstring w;
    for (std::size_t i = 0; i < utf8.size();) {
        const auto b = static_cast<unsigned char>(utf8[i]);
        const int n = b < 0x80 ? 1 : b < 0xE0 ? 2 : b < 0xF0 ? 3 : 4;
        if (i + static_cast<std::size_t>(n) > utf8.size() || (n > 1 && b < 0xC2) || b > 0xF4)
            throw std::invalid_argument("a path that is not UTF-8");
        std::uint32_t c = n == 1 ? b : b & (0x7Fu >> n);
        for (int k = 1; k < n; ++k) {
            const auto t = static_cast<unsigned char>(utf8[i + static_cast<std::size_t>(k)]);
            if ((t & 0xC0) != 0x80) throw std::invalid_argument("a path that is not UTF-8");
            c = (c << 6) | (t & 0x3Fu);
        }
        i += static_cast<std::size_t>(n);
        if (c >= 0x10000) {
            c -= 0x10000;
            w.push_back(static_cast<wchar_t>(0xD800 + (c >> 10)));
            w.push_back(static_cast<wchar_t>(0xDC00 + (c & 0x3FF)));
        } else {
            w.push_back(static_cast<wchar_t>(c));
        }
    }
    return std::filesystem::path(w);
#else
    return std::filesystem::path(utf8);
#endif
}

// A file by its UTF-8 path, opened as fopen's mode says.
inline std::FILE* open(const std::string& path, const char* mode)
{
#if defined(_WIN32)
    const std::wstring w = path_of(path).wstring();
    const std::wstring m(mode, mode + std::char_traits<char>::length(mode));
    std::FILE* f = _wfsopen(w.c_str(), m.c_str(), _SH_DENYNO);   // others may read it meanwhile
#else
    std::FILE* f = std::fopen(path.c_str(), mode);
#endif
    if (!f) throw std::runtime_error("cannot open " + path);
    return f;
}

inline void write(std::FILE* f, const void* p, std::size_t n, const std::string& path)
{
    if (n && std::fwrite(p, 1, n, f) != n) throw std::runtime_error("cannot write " + path);
}

// written through to the disk (fsync), so that a crash keeps what was written
inline void sync(std::FILE* f, const std::string& path)
{
    if (std::fflush(f) != 0) throw std::runtime_error("cannot write " + path);
#if defined(_WIN32)
    const int failed = _commit(_fileno(f));
#else
    const int failed = fsync(fileno(f));
#endif
    if (failed) throw std::runtime_error("cannot sync " + path);
}

}   // namespace file

// One line of an index: offset, length, checksum (8 hex digits), status, key.
inline std::string index_line(const Entry& e)
{
    char crc[9];
    std::snprintf(crc, sizeof crc, "%08x", e.crc);
    return std::to_string(e.offset) + "\t" + std::to_string(e.length) + "\t" + crc + "\t" + e.status + "\t" + e.key +
           "\n";
}

// The lines of an index back as entries; an error where one is not a line of five.
inline std::vector<Entry> index_entries(const std::string& lines)
{
    std::vector<Entry> out;
    std::size_t at = 0;
    while (at < lines.size()) {
        std::size_t end = lines.find('\n', at);
        if (end == std::string::npos) end = lines.size();
        std::string line = lines.substr(at, end - at);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        at = end + 1;
        std::size_t p[4], from = 0;
        for (auto& t : p) {
            t = line.find('\t', from);
            if (t == std::string::npos) throw std::runtime_error("index: not a line of five fields");
            from = t + 1;
        }
        Entry e;
        e.offset = std::stoull(line.substr(0, p[0]));
        e.length = std::stoull(line.substr(p[0] + 1, p[1] - p[0] - 1));
        e.crc = static_cast<std::uint32_t>(std::stoul(line.substr(p[1] + 1, p[2] - p[1] - 1), nullptr, 16));
        e.status = line.substr(p[2] + 1, p[3] - p[2] - 1);
        e.key = line.substr(p[3] + 1);
        out.push_back(std::move(e));
    }
    return out;
}

// A record appended to a segment and synced, then its line to the segment's
// index; its entry (status: the caller's).
inline Entry append_record(const std::string& raw, const std::string& idx, const std::string& key,
                           const std::vector<std::uint8_t>& record, const std::string& status)
{
    const RecordAt r = parse_record(record.data(), record.size(), 0);
    if (!r.ok || r.key != key) throw std::invalid_argument("append_record: not a whole record of " + key);
    Entry e;
    e.key = key;
    e.length = record.size();
    e.crc = r.crc;
    e.status = status;
    std::FILE* f = file::open(raw, "ab");
    try {
        std::fseek(f, 0, SEEK_END);
        const long at = std::ftell(f);
        if (at < 0) throw std::runtime_error("cannot tell the end of " + raw);
        e.offset = static_cast<std::uint64_t>(at);
        file::write(f, record.data(), record.size(), raw);
        file::sync(f, raw);
    } catch (...) {
        std::fclose(f);
        throw;
    }
    std::fclose(f);
    const std::string line = index_line(e);
    std::FILE* g = file::open(idx, "ab");
    file::write(g, line.data(), line.size(), idx);
    std::fclose(g);
    return e;
}

// A pack written: `header` its "key = value" lines, then the meta and metrics
// tables' bytes, the records [(key, record, status)] in order; their entries.
struct PackRecord {
    std::string key;
    std::vector<std::uint8_t> record;
    std::string status;
};

inline std::vector<Entry> write_pack(const std::string& path, const std::string& tmp, const std::string& header,
                                     const std::vector<std::uint8_t>& meta, const std::vector<std::uint8_t>& metrics,
                                     const std::vector<PackRecord>& records)
{
    std::vector<Entry> entries;
    std::FILE* f = file::open(tmp, "wb");
    std::uint64_t at = 0;
    std::uint32_t whole = 0;
    auto put = [&](const std::uint8_t* p, std::size_t n) {
        file::write(f, p, n, tmp);
        whole = crc32c(p, n, whole);
        at += n;
    };
    const std::vector<std::uint8_t> zeros(kAlign, 0);
    auto align = [&] { put(zeros.data(), static_cast<std::size_t>((kAlign - at % kAlign) % kAlign)); };
    try {
        std::vector<std::uint8_t> head(header.begin(), header.end());
        const std::size_t pad = (kAlign - head.size() % kAlign) % kAlign;
        head.insert(head.end(), pad ? pad : kAlign, '\n');
        put(head.data(), head.size());
        PackFooter ft;
        ft.meta_at = at;
        ft.meta_size = meta.size();
        put(meta.data(), meta.size());
        align();
        ft.metrics_at = at;
        ft.metrics_size = metrics.size();
        put(metrics.data(), metrics.size());
        align();
        std::string lines;
        for (const PackRecord& pr : records) {
            const RecordAt r = parse_record(pr.record.data(), pr.record.size(), 0);
            if (!r.ok || r.key != pr.key) throw std::invalid_argument("write_pack: not a whole record of " + pr.key);
            Entry e{pr.key, at, pr.record.size(), r.crc, pr.status};
            put(pr.record.data(), pr.record.size());
            align();
            lines += index_line(e);
            entries.push_back(std::move(e));
        }
        std::vector<std::uint8_t> index = {'T'};
        const std::vector<std::uint8_t> coded =
            encode_text(reinterpret_cast<const std::uint8_t*>(lines.data()), lines.size());
        index.insert(index.end(), coded.begin(), coded.end());
        ft.index_at = at;
        ft.index_size = index.size();
        put(index.data(), index.size());
        align();
        std::vector<std::uint8_t> foot(kPackFoot, kPackFoot + 8);
        for (std::uint64_t v : {ft.meta_at, ft.meta_size, ft.metrics_at, ft.metrics_size, ft.index_at, ft.index_size})
            rec::put_le(foot, v, 8);
        rec::put_le(foot, whole, 4);
        rec::put_le(foot, crc32c(index.data(), index.size()), 4);
        file::write(f, foot.data(), foot.size(), tmp);
        file::sync(f, tmp);
    } catch (...) {
        std::fclose(f);
        std::error_code ec;
        std::filesystem::remove(file::path_of(tmp), ec);
        throw;
    }
    std::fclose(f);
    std::filesystem::rename(file::path_of(tmp), file::path_of(path));   // replaces a pack there
    return entries;
}

// A pack's footer, from the file's bytes; an error where there is none.
inline PackFooter pack_footer(const std::uint8_t* data, std::size_t size, const std::string& name)
{
    if (size < 64) throw std::runtime_error(name + ": shorter than a pack's footer");
    const std::uint8_t* p = data + size - 64;
    if (std::memcmp(p, kPackFoot, 8) != 0) throw std::runtime_error(name + ": no pack footer at its end");
    PackFooter f;
    std::uint64_t* fields[6] = {&f.meta_at, &f.meta_size, &f.metrics_at, &f.metrics_size, &f.index_at, &f.index_size};
    for (int i = 0; i < 6; ++i) *fields[i] = rec::get_le(p + 8 + 8 * i, 8);
    f.crc = static_cast<std::uint32_t>(rec::get_le(p + 56, 4));
    f.index_crc = static_cast<std::uint32_t>(rec::get_le(p + 60, 4));
    f.end = size - 64;
    for (int i = 0; i < 3; ++i)
        if (*fields[2 * i] > f.end || *fields[2 * i + 1] > f.end - *fields[2 * i])
            throw std::runtime_error(name + ": the footer's places are past the file");
    return f;
}

// A pack's entries, from its index (its checksum checked); "R": lines as they are.
inline std::vector<Entry> pack_entries(const std::uint8_t* data, std::size_t size, const std::string& name)
{
    const PackFooter f = pack_footer(data, size, name);
    const std::uint8_t* p = data + f.index_at;
    if (crc32c(p, static_cast<std::size_t>(f.index_size)) != f.index_crc)
        throw std::runtime_error(name + ": the index's checksum does not hold");
    if (f.index_size == 0) throw std::runtime_error(name + ": no index");
    std::string lines;
    if (p[0] == 'T') {
        const std::vector<std::uint8_t> t = decode_text(p + 1, static_cast<std::size_t>(f.index_size - 1));
        lines.assign(t.begin(), t.end());
    } else if (p[0] == 'R') {
        lines.assign(reinterpret_cast<const char*>(p + 1), static_cast<std::size_t>(f.index_size - 1));
    } else {
        throw std::runtime_error(name + ": an index of unknown kind");
    }
    return index_entries(lines);
}

// What is wrong with a pack, and where (docs/store-format.md): every part by
// its own checksum — the meta and metrics tables (a magic and the CRC-32C of
// their bodies), the index, each record — and the bytes between the parts
// against zero; where only the file's checksum fails, the header or the
// footer's checksum. Empty: whole.
inline std::vector<std::string> check_pack(const std::uint8_t* data, std::size_t size, const std::string& name)
{
    std::vector<std::string> bad;
    PackFooter f;
    try {
        f = pack_footer(data, size, name);
    } catch (const std::runtime_error& e) {
        return {e.what()};
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> parts;
    const char* magic[2] = {"MOOTMET1", "MOOTMTR1"};
    const char* part[2] = {"meta", "metrics"};
    const std::uint64_t at[2] = {f.meta_at, f.metrics_at}, n[2] = {f.meta_size, f.metrics_size};
    for (int i = 0; i < 2; ++i) {
        parts.emplace_back(at[i], at[i] + n[i]);
        if (n[i] == 0) continue;
        const std::uint8_t* p = data + at[i];
        if (n[i] < 12 || std::memcmp(p, magic[i], 8) != 0 ||
            crc32c(p + 12, static_cast<std::size_t>(n[i] - 12)) != static_cast<std::uint32_t>(rec::get_le(p + 8, 4)))
            bad.push_back(name + "@" + std::to_string(at[i]) + ": the " + part[i] + " table is damaged (its checksum)");
    }
    std::vector<Entry> entries;
    if (crc32c(data + f.index_at, static_cast<std::size_t>(f.index_size)) != f.index_crc) {
        bad.push_back(name + "@" + std::to_string(f.index_at) + ": the index is damaged (its checksum)");
    } else {
        try {
            entries = pack_entries(data, size, name);
        } catch (const std::exception& e) {
            bad.push_back(name + "@" + std::to_string(f.index_at) + ": the index is damaged (" + e.what() + ")");
        }
    }
    parts.emplace_back(f.index_at, f.index_at + f.index_size);
    for (const Entry& e : entries) {
        const bool inside = e.offset <= f.end && e.length <= f.end - e.offset;
        parts.emplace_back(e.offset, inside ? e.offset + e.length : e.offset);
        const RecordAt r = inside ? parse_record(data, static_cast<std::size_t>(e.offset + e.length),
                                                 static_cast<std::size_t>(e.offset))
                                  : RecordAt{};
        if (!r.ok || r.key != e.key || r.crc != e.crc || r.end != e.offset + e.length)
            bad.push_back(name + "@" + std::to_string(e.offset) + ": the record of " + e.key +
                          " is damaged (its checksum)");
    }
    std::sort(parts.begin(), parts.end());
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const std::uint64_t from = parts[i].second, to = i + 1 < parts.size() ? parts[i + 1].first : f.end;
        for (std::uint64_t b = from; b < to; ++b)
            if (data[b] != 0) {
                bad.push_back(name + "@" + std::to_string(b) + ": a byte that should be zero (between the parts)");
                break;
            }
    }
    if (bad.empty() && crc32c(data, static_cast<std::size_t>(f.end)) != f.crc)
        bad.push_back(name + ": the header (bytes 0-" + std::to_string(parts.front().first) +
                      ") or the footer's checksum is damaged");
    return bad;
}

}   // namespace mootation::store
