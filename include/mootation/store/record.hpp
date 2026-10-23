#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A run's record in the store, format 3 (task 17, §3а.3: one implementation
// in C++): its blocks, its body and the record's frame, byte for byte as
// docs/store-format.md says.
//
//   * A block: its name, dtype (a NumPy dtype's text, or json, text, bytes),
//     shape and bytes, kept by the shortest of the library's codecs that
//     apply — float64 by the float64 column, integers by the integer column,
//     anything else by the text codec, an array also shuffled — or raw.
//   * A body: the count of blocks and the blocks; a record's point tables go
//     together into one block "#points" (record_tables.hpp) with their places,
//     after the other blocks.
//   * A record: "MOOTREC2", the length of the rest, the key, the first 8
//     bytes of the key's SHA-256, the body's CRC-32C and the body.
//
// Which blocks are point tables, and their columns, the caller says (it reads
// their "#head"); so does what meta.json keeps (store.py). A block of the
// records of before (formats 1 and 2: "zlib", "shuffle+zlib") is returned
// with its method and its payload as it is, for the caller to inflate.
// Little-endian, whatever the machine: every number is put and read a byte
// at a time.
// ============================================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "column_codec.hpp"
#include "crc32c.hpp"
#include "record_tables.hpp"
#include "sha256.hpp"
#include "text_codec.hpp"

namespace mootation::store {

struct Block {
    std::string name;
    std::string dtype;                   // "json", "text", "bytes", or a NumPy dtype's text ("<f8", "|u1" ...)
    std::vector<std::uint64_t> shape;    // empty for json, text and bytes
    std::vector<std::uint8_t> data;      // the bytes; for "zlib", "shuffle+zlib": the payload as stored
    std::string how;                     // decoded blocks: "", or the method left to the caller
};

// A point table of a body: the block's place among the blocks and its columns.
struct PointTable {
    std::size_t place = 0;
    std::vector<std::string> columns;
};

constexpr char kRecordMagic[9] = "MOOTREC2";        // CRC-32C of the body
constexpr char kRecordMagicZlib[9] = "MOOTREC1";    // zlib's CRC-32 (read only)

namespace rec {

inline void put_le(std::vector<std::uint8_t>& out, std::uint64_t v, int bytes)
{
    for (int i = 0; i < bytes; ++i) out.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}

inline std::uint64_t get_le(const std::uint8_t* p, int bytes)
{
    std::uint64_t v = 0;
    for (int i = 0; i < bytes; ++i) v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    return v;
}

// what a block's dtype says: an array or not, its item's size and kind
struct Dtype {
    bool array = false;
    char order = '|';
    char kind = 'B';
    std::size_t size = 1;
};

inline Dtype dtype_of(const std::string& s)
{
    Dtype d;
    if (s == "json" || s == "text" || s == "bytes") return d;
    d.array = true;
    if (s.size() < 3 || std::string("<>|=").find(s[0]) == std::string::npos)
        throw std::invalid_argument("block dtype '" + s + "': not a NumPy dtype's text");
    d.order = s[0];
    d.kind = s[1];
    std::size_t i = 2, n = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') n = 10 * n + static_cast<std::size_t>(s[i++] - '0');
    const bool unit = i < s.size() && s[i] == '[' && s.back() == ']' && (d.kind == 'M' || d.kind == 'm');
    if (n == 0 || (i != s.size() && !unit) || std::string("biufcUSVaMm").find(d.kind) == std::string::npos)
        throw std::invalid_argument("block dtype '" + s + "': not a NumPy dtype's text");
    d.size = d.kind == 'U' ? 4 * n : n;
    return d;
}

inline bool is_f64(const std::string& dtype) { return dtype == "<f8"; }

// integers the integer column keeps: signed up to 8 bytes, unsigned up to 4
inline bool is_ints(const Dtype& d)
{
    return d.array && (d.kind == 'i' || d.kind == 'u') && d.size <= 8 && !(d.kind == 'u' && d.size == 8) &&
           (d.order == '<' || d.order == '|' || d.order == '=');
}

inline std::vector<std::uint8_t> shuffle(const std::vector<std::uint8_t>& data, std::size_t size)
{
    if (size <= 1 || data.empty()) return data;
    const std::size_t n = data.size() / size;
    std::vector<std::uint8_t> out(data.size());
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < size; ++j) out[j * n + i] = data[i * size + j];
    return out;
}

inline std::vector<std::uint8_t> unshuffle(const std::vector<std::uint8_t>& data, std::size_t size)
{
    if (size <= 1 || data.empty()) return data;
    const std::size_t n = data.size() / size;
    std::vector<std::uint8_t> out(data.size());
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < size; ++j) out[i * size + j] = data[j * n + i];
    return out;
}

inline std::vector<double> doubles_of(const std::vector<std::uint8_t>& data)
{
    std::vector<double> x(data.size() / 8);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const std::uint64_t b = get_le(&data[8 * i], 8);
        std::memcpy(&x[i], &b, 8);
    }
    return x;
}

inline std::vector<std::int64_t> ints_of(const std::vector<std::uint8_t>& data, const Dtype& d)
{
    std::vector<std::int64_t> x(data.size() / d.size);
    const int bits = static_cast<int>(8 * d.size);
    for (std::size_t i = 0; i < x.size(); ++i) {
        std::uint64_t v = get_le(&data[d.size * i], static_cast<int>(d.size));
        if (d.kind == 'i' && bits < 64 && ((v >> (bits - 1)) & 1u)) v |= ~std::uint64_t{0} << bits;   // sign
        x[i] = static_cast<std::int64_t>(v);
    }
    return x;
}

inline std::size_t count_of(const std::vector<std::uint64_t>& shape)
{
    std::size_t n = 1;
    for (std::uint64_t s : shape) n *= static_cast<std::size_t>(s);
    return n;
}

struct Reader {
    const std::uint8_t* p;
    std::size_t size, at = 0;
    void need(std::size_t n) const
    {
        if (n > size - at) throw std::runtime_error("record body: the bytes end inside a block");
    }
    std::uint64_t le(int bytes)
    {
        need(static_cast<std::size_t>(bytes));
        const std::uint64_t v = get_le(p + at, bytes);
        at += static_cast<std::size_t>(bytes);
        return v;
    }
    std::string text(std::size_t n)
    {
        need(n);
        std::string s(reinterpret_cast<const char*>(p + at), n);
        at += n;
        return s;
    }
};

}   // namespace rec

// A block's bytes in a body: u16 name length, name, u8 dtype length, dtype,
// u8 dimensions, u64 each, u8 method length, method, u64 payload length,
// payload.
inline std::vector<std::uint8_t> encode_block(const Block& b)
{
    const rec::Dtype d = rec::dtype_of(b.dtype);
    std::string how = "raw";
    std::vector<std::uint8_t> payload;
    if (b.name == "#points") {
        payload = b.data;                                   // coded already
    } else if (rec::is_f64(b.dtype)) {
        const std::vector<double> x = rec::doubles_of(b.data);
        how = "f64";
        payload = encode_f64(x.data(), x.size());
    } else if (rec::is_ints(d)) {
        const std::vector<std::int64_t> x = rec::ints_of(b.data, d);
        how = "i64";
        payload = encode_i64(x.data(), x.size());
    } else {
        how = "text";
        payload = encode_text(b.data.data(), b.data.size());
        if (d.array) {
            const std::vector<std::uint8_t> s = rec::shuffle(b.data, d.size);
            std::vector<std::uint8_t> shuffled = encode_text(s.data(), s.size());
            if (shuffled.size() < payload.size()) {
                how = "shuffle+text";
                payload.swap(shuffled);
            }
        }
    }
    if (payload.size() >= b.data.size()) {
        how = "raw";
        payload = b.data;
    }
    if (b.name.size() > 0xFFFF || b.dtype.size() > 0xFF || b.shape.size() > 0xFF)
        throw std::invalid_argument("block '" + b.name + "': its name, dtype or shape is too long");
    std::vector<std::uint8_t> out;
    out.reserve(payload.size() + b.name.size() + 64);
    rec::put_le(out, b.name.size(), 2);
    out.insert(out.end(), b.name.begin(), b.name.end());
    rec::put_le(out, b.dtype.size(), 1);
    out.insert(out.end(), b.dtype.begin(), b.dtype.end());
    rec::put_le(out, b.shape.size(), 1);
    for (std::uint64_t s : b.shape) rec::put_le(out, s, 8);
    rec::put_le(out, how.size(), 1);
    out.insert(out.end(), how.begin(), how.end());
    rec::put_le(out, payload.size(), 8);
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

// A body: the point tables (their places among `blocks`) into "#points" — a
// u32 count, a u32 place each, the tables' bytes — after the other blocks.
inline std::vector<std::uint8_t> encode_body(const std::vector<Block>& blocks, const std::vector<PointTable>& tables)
{
    std::vector<const Block*> order;
    Block points;
    if (!tables.empty()) {
        std::vector<Table> ts;
        std::vector<bool> taken(blocks.size(), false);
        points.name = "#points";
        points.dtype = "bytes";
        rec::put_le(points.data, tables.size(), 4);
        for (const PointTable& t : tables) {
            if (t.place >= blocks.size() || taken[t.place]) throw std::invalid_argument("body: a point table's place");
            const Block& b = blocks[t.place];
            if (!rec::is_f64(b.dtype) || b.shape.size() != 2 || b.shape[1] != t.columns.size() ||
                b.data.size() != 8 * rec::count_of(b.shape))
                throw std::invalid_argument("body: block '" + b.name + "' is not a float64 table of its columns");
            taken[t.place] = true;
            rec::put_le(points.data, t.place, 4);
            Table tb;
            tb.name = b.name;
            tb.columns = t.columns;
            tb.rows = static_cast<std::size_t>(b.shape[0]);
            tb.values = rec::doubles_of(b.data);
            ts.push_back(std::move(tb));
        }
        const std::vector<std::uint8_t> coded = encode_tables(ts);
        points.data.insert(points.data.end(), coded.begin(), coded.end());
        for (std::size_t i = 0; i < blocks.size(); ++i)
            if (!taken[i]) order.push_back(&blocks[i]);
        order.push_back(&points);
    } else {
        for (const Block& b : blocks) order.push_back(&b);
    }
    std::vector<std::uint8_t> out;
    rec::put_le(out, order.size(), 4);
    for (const Block* b : order) {
        const std::vector<std::uint8_t> one = encode_block(*b);
        out.insert(out.end(), one.begin(), one.end());
    }
    return out;
}

// The blocks of a body, the point tables back at their places; a block of
// "zlib" or "shuffle+zlib" as stored, its method in `how`.
inline std::vector<Block> decode_body(const std::uint8_t* data, std::size_t size)
{
    rec::Reader r{data, size};
    const std::uint64_t count = r.le(4);
    if (count > size) throw std::runtime_error("record body: more blocks than bytes");
    std::vector<Block> out;
    for (std::uint64_t k = 0; k < count; ++k) {
        Block b;
        b.name = r.text(static_cast<std::size_t>(r.le(2)));
        b.dtype = r.text(static_cast<std::size_t>(r.le(1)));
        const std::uint64_t nd = r.le(1);
        for (std::uint64_t j = 0; j < nd; ++j) b.shape.push_back(r.le(8));
        const std::string how = r.text(static_cast<std::size_t>(r.le(1)));
        const std::uint64_t n = r.le(8);
        r.need(static_cast<std::size_t>(n));
        const std::uint8_t* payload = data + r.at;
        r.at += static_cast<std::size_t>(n);
        const rec::Dtype d = rec::dtype_of(b.dtype);
        const std::size_t values = b.shape.empty() ? 1 : rec::count_of(b.shape);
        if (how == "raw") {
            b.data.assign(payload, payload + n);
        } else if (how == "zlib" || how == "shuffle+zlib") {
            b.data.assign(payload, payload + n);
            b.how = how;
        } else if (how == "text") {
            b.data = decode_text(payload, static_cast<std::size_t>(n));
        } else if (how == "shuffle+text") {
            b.data = rec::unshuffle(decode_text(payload, static_cast<std::size_t>(n)), d.size);
        } else if (how == "f64") {
            const std::vector<double> x = decode_f64(payload, static_cast<std::size_t>(n), values);
            b.data.reserve(8 * x.size());
            for (double v : x) {
                std::uint64_t bits;
                std::memcpy(&bits, &v, 8);
                rec::put_le(b.data, bits, 8);
            }
        } else if (how == "i64") {
            if (!d.array) throw std::runtime_error("block '" + b.name + "': i64 of no array");
            const std::vector<std::int64_t> x = decode_i64(payload, static_cast<std::size_t>(n), values);
            b.data.reserve(d.size * x.size());
            for (std::int64_t v : x) rec::put_le(b.data, static_cast<std::uint64_t>(v), static_cast<int>(d.size));
        } else {
            throw std::runtime_error("block '" + b.name + "': unknown compression '" + how + "'");
        }
        out.push_back(std::move(b));
    }
    if (r.at != size) throw std::runtime_error("record body: bytes past the blocks");
    // #points: the tables back at their places
    const auto at = std::find_if(out.begin(), out.end(), [](const Block& b) { return b.name == "#points"; });
    if (at == out.end()) return out;
    const std::vector<std::uint8_t> pts = std::move(at->data);
    out.erase(at);
    if (pts.size() < 4) throw std::runtime_error("block '#points': no count");
    const std::size_t k = static_cast<std::size_t>(rec::get_le(pts.data(), 4));
    if (pts.size() < 4 + 4 * k) throw std::runtime_error("block '#points': the places end early");
    std::vector<std::pair<std::size_t, Block>> placed;
    const std::vector<Table> tables = decode_tables(pts.data() + 4 + 4 * k, pts.size() - 4 - 4 * k);
    if (tables.size() != k) throw std::runtime_error("block '#points': its tables and places differ in number");
    for (std::size_t i = 0; i < k; ++i) {
        Block b;
        b.name = tables[i].name;
        b.dtype = "<f8";
        b.shape = {tables[i].rows, tables[i].columns.size()};
        b.data.reserve(8 * tables[i].values.size());
        for (double v : tables[i].values) {
            std::uint64_t bits;
            std::memcpy(&bits, &v, 8);
            rec::put_le(b.data, bits, 8);
        }
        placed.emplace_back(static_cast<std::size_t>(rec::get_le(pts.data() + 4 + 4 * i, 4)), std::move(b));
    }
    std::stable_sort(placed.begin(), placed.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (auto& [place, b] : placed) {
        if (place > out.size()) throw std::runtime_error("block '#points': a place past the blocks");
        out.insert(out.begin() + static_cast<std::ptrdiff_t>(place), std::move(b));
    }
    return out;
}

// A record: "MOOTREC2", u64 the length of the rest, u16 key length, key,
// the first 8 bytes of the key's SHA-256, u32 CRC-32C of the body, the body.
inline std::vector<std::uint8_t> encode_record(const std::string& key, const std::vector<std::uint8_t>& body)
{
    if (key.size() > 0xFFFF) throw std::invalid_argument("record: a key longer than 65535 bytes");
    std::vector<std::uint8_t> out(kRecordMagic, kRecordMagic + 8);
    rec::put_le(out, 2 + key.size() + 8 + 4 + body.size(), 8);
    rec::put_le(out, key.size(), 2);
    out.insert(out.end(), key.begin(), key.end());
    const auto h = sha256(reinterpret_cast<const std::uint8_t*>(key.data()), key.size());
    out.insert(out.end(), h.begin(), h.begin() + 8);
    rec::put_le(out, crc32c(body.data(), body.size()), 4);
    out.insert(out.end(), body.begin(), body.end());
    return out;
}

// The record at `at` in buf: whole and sound, its key, body, CRC and end.
struct RecordAt {
    bool ok = false;
    std::string key;
    std::size_t body = 0, body_size = 0, end = 0;
    std::uint32_t crc = 0;
};

inline RecordAt parse_record(const std::uint8_t* buf, std::size_t size, std::size_t at)
{
    RecordAt r;
    if (at > size || size - at < 16) return r;
    const bool c = std::memcmp(buf + at, kRecordMagic, 8) == 0;
    if (!c && std::memcmp(buf + at, kRecordMagicZlib, 8) != 0) return r;
    const std::uint64_t rest = rec::get_le(buf + at + 8, 8);
    if (rest < 14 || rest > size - at - 16) return r;
    const std::size_t end = at + 16 + static_cast<std::size_t>(rest);
    const std::size_t n = static_cast<std::size_t>(rec::get_le(buf + at + 16, 2));
    if (at + 30 + n > end) return r;
    const std::uint8_t* k = buf + at + 18;
    const auto h = sha256(k, n);
    if (std::memcmp(h.data(), k + n, 8) != 0) return r;
    r.crc = static_cast<std::uint32_t>(rec::get_le(k + n + 8, 4));
    r.body = at + 30 + n;
    r.body_size = end - r.body;
    if ((c ? crc32c(buf + r.body, r.body_size) : crc32_zlib(buf + r.body, r.body_size)) != r.crc) return r;
    r.key.assign(reinterpret_cast<const char*>(k), n);
    r.end = end;
    r.ok = true;
    return r;
}

}   // namespace mootation::store
