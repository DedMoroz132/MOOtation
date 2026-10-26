#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// NumPy's .npz and .npy read in C++ — task 17, stage 5: the campaign engine
// reads the reference cache (run/refs.py, campaign.REFERENCES_DIR) as Python
// wrote it, without Python.
//
// An .npz is a zip archive of .npy files (numpy.savez: stored; savez_compressed:
// deflated). Read here: the central directory, each member's local header,
// the stored and the deflated members (RFC 1951, inflate below), each
// checked against its CRC-32. An .npy (NumPy's format.py, versions 1 to 3): the
// magic, the header's dict — descr, fortran_order, shape — and the raw bytes.
// What the cache holds is read: little-endian float64 and int64 arrays, bool
// arrays, and a unicode string (a 0-d '<U' array, json.dumps of the meta).
// Not read: zip64, encryption, a data descriptor (numpy writes none of them
// for these files), big-endian and object arrays — an error names what it
// met.
//
// inflate follows the decoder of RFC 1951 §3.2 as Mark Adler's puff.c lays it
// out: canonical Huffman codes given by their lengths, decoded a bit at a time
// by counting the codes of each length.
// ============================================================================

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../store/crc32c.hpp"

namespace mootation::npz {

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& what) : std::runtime_error("npz: " + what) {}
};

// ── inflate (RFC 1951) ──────────────────────────────────────────────────────

namespace detail {

constexpr int MAXBITS = 15;              // the longest code
constexpr int MAXLCODES = 286;           // literal/length codes
constexpr int MAXDCODES = 30;            // distance codes
constexpr int FIXLCODES = 288;           // literal/length codes of the fixed table

struct Bits {
    const std::uint8_t* in;
    std::size_t n, pos = 0;
    std::uint32_t buf = 0;               // bits not used yet, the next lowest
    int cnt = 0;

    int need(int k)                      // the next k bits, least significant first
    {
        std::uint32_t v = buf;
        while (cnt < k) {
            if (pos == n) throw Error("deflate stream ends early");
            v |= static_cast<std::uint32_t>(in[pos++]) << cnt;
            cnt += 8;
        }
        buf = v >> k;
        cnt -= k;
        return static_cast<int>(v & ((1u << k) - 1));
    }
};

// a canonical code: how many codes of each length, and the symbols in code order
struct Huffman {
    std::vector<short> count, symbol;
};

// the code of these lengths; < 0 for an over-subscribed set, > 0 for an
// incomplete one, 0 for a complete one
inline int construct(Huffman& h, const short* length, int n)
{
    h.count.assign(MAXBITS + 1, 0);
    h.symbol.assign(static_cast<std::size_t>(n), 0);
    for (int s = 0; s < n; ++s) ++h.count[static_cast<std::size_t>(length[s])];
    if (h.count[0] == n) return 0;       // no codes: complete, and decode() fails on it
    int left = 1;
    for (int len = 1; len <= MAXBITS; ++len) {
        left <<= 1;
        left -= h.count[static_cast<std::size_t>(len)];
        if (left < 0) return left;
    }
    short offs[MAXBITS + 1];
    offs[1] = 0;
    for (int len = 1; len < MAXBITS; ++len)
        offs[len + 1] = static_cast<short>(offs[len] + h.count[static_cast<std::size_t>(len)]);
    for (int s = 0; s < n; ++s)
        if (length[s] != 0) h.symbol[static_cast<std::size_t>(offs[length[s]]++)] = static_cast<short>(s);
    return left;
}

inline int decode(Bits& b, const Huffman& h)
{
    int code = 0, first = 0, index = 0;
    for (int len = 1; len <= MAXBITS; ++len) {
        code |= b.need(1);
        const int count = h.count[static_cast<std::size_t>(len)];
        if (code - count < first) return h.symbol[static_cast<std::size_t>(index + (code - first))];
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    throw Error("deflate stream holds a code that is not in its table");
}

inline void codes(Bits& b, std::vector<std::uint8_t>& out, const Huffman& lencode, const Huffman& distcode)
{
    static const short lbase[29] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
                                    35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
    static const short lext[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                                   3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
    static const short dbase[30] = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129,
                                    193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097,
                                    6145, 8193, 12289, 16385, 24577};
    static const short dext[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6,
                                   7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
    for (;;) {
        int symbol = decode(b, lencode);
        if (symbol < 256) {
            out.push_back(static_cast<std::uint8_t>(symbol));
        } else if (symbol == 256) {
            return;
        } else {
            symbol -= 257;
            if (symbol >= 29) throw Error("deflate length code out of range");
            const std::size_t len = static_cast<std::size_t>(lbase[symbol] + b.need(lext[symbol]));
            symbol = decode(b, distcode);
            if (symbol >= 30) throw Error("deflate distance code out of range");
            const std::size_t dist = static_cast<std::size_t>(dbase[symbol] + b.need(dext[symbol]));
            if (dist > out.size()) throw Error("deflate distance before the start");
            const std::size_t from = out.size() - dist;
            for (std::size_t k = 0; k < len; ++k) out.push_back(out[from + k]);
        }
    }
}

inline void fixed(Bits& b, std::vector<std::uint8_t>& out)
{
    struct Tables {
        Huffman lencode, distcode;
    };
    static const Tables t = [] {        // built once, thread-safe
        Tables r;
        short lengths[FIXLCODES];
        int s = 0;
        for (; s < 144; ++s) lengths[s] = 8;
        for (; s < 256; ++s) lengths[s] = 9;
        for (; s < 280; ++s) lengths[s] = 7;
        for (; s < FIXLCODES; ++s) lengths[s] = 8;
        construct(r.lencode, lengths, FIXLCODES);
        for (s = 0; s < MAXDCODES; ++s) lengths[s] = 5;
        construct(r.distcode, lengths, MAXDCODES);
        return r;
    }();
    codes(b, out, t.lencode, t.distcode);
}

inline void dynamic(Bits& b, std::vector<std::uint8_t>& out)
{
    static const short order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
    const int nlen = b.need(5) + 257, ndist = b.need(5) + 1, ncode = b.need(4) + 4;
    if (nlen > MAXLCODES || ndist > MAXDCODES) throw Error("deflate dynamic block has too many codes");
    short lengths[MAXLCODES + MAXDCODES];
    int index = 0;
    for (; index < ncode; ++index) lengths[order[index]] = static_cast<short>(b.need(3));
    for (; index < 19; ++index) lengths[order[index]] = 0;
    Huffman lencode, distcode;
    if (construct(lencode, lengths, 19) != 0) throw Error("deflate code-length code is not complete");
    index = 0;
    while (index < nlen + ndist) {
        int symbol = decode(b, lencode);
        if (symbol < 16) {
            lengths[index++] = static_cast<short>(symbol);
        } else {
            short len = 0;
            if (symbol == 16) {
                if (index == 0) throw Error("deflate repeats a length before the first");
                len = lengths[index - 1];
                symbol = 3 + b.need(2);
            } else if (symbol == 17) {
                symbol = 3 + b.need(3);
            } else {
                symbol = 11 + b.need(7);
            }
            if (index + symbol > nlen + ndist) throw Error("deflate lengths run past the codes");
            while (symbol--) lengths[index++] = len;
        }
    }
    if (lengths[256] == 0) throw Error("deflate block without an end code");
    const int err = construct(lencode, lengths, nlen);
    if (err < 0 || (err > 0 && nlen - lencode.count[0] != 1)) throw Error("deflate literal/length code is not complete");
    const int errd = construct(distcode, lengths + nlen, ndist);
    if (errd < 0 || (errd > 0 && ndist - distcode.count[0] != 1)) throw Error("deflate distance code is not complete");
    codes(b, out, lencode, distcode);
}

}   // namespace detail

// a raw deflate stream (no zlib or gzip wrapper) decompressed
inline std::vector<std::uint8_t> inflate(const std::uint8_t* in, std::size_t n, std::size_t size_hint = 0)
{
    std::vector<std::uint8_t> out;
    out.reserve(size_hint);
    detail::Bits b{in, n};
    int last;
    do {
        last = b.need(1);
        const int type = b.need(2);
        if (type == 0) {                                  // stored: from the next byte
            b.buf = 0;
            b.cnt = 0;
            if (b.pos + 4 > n) throw Error("deflate stored block ends early");
            const unsigned len = in[b.pos] | (in[b.pos + 1] << 8);
            const unsigned nlen = in[b.pos + 2] | (in[b.pos + 3] << 8);
            if (len != (~nlen & 0xffffu)) throw Error("deflate stored block's length check fails");
            b.pos += 4;
            if (b.pos + len > n) throw Error("deflate stored block ends early");
            out.insert(out.end(), in + b.pos, in + b.pos + len);
            b.pos += len;
        } else if (type == 1) {
            detail::fixed(b, out);
        } else if (type == 2) {
            detail::dynamic(b, out);
        } else {
            throw Error("deflate block of type 3");
        }
    } while (!last);
    return out;
}

// ── zip ─────────────────────────────────────────────────────────────────────

namespace detail {

inline std::uint32_t u16(const std::uint8_t* p) { return p[0] | (p[1] << 8); }
inline std::uint32_t u32(const std::uint8_t* p)
{
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

}   // namespace detail

// every member of a zip archive held in memory: name -> its bytes, CRC checked
inline std::map<std::string, std::vector<std::uint8_t>> unzip(const std::vector<std::uint8_t>& z)
{
    using detail::u16;
    using detail::u32;
    if (z.size() < 22) throw Error("not a zip archive (too short)");
    // the end of the central directory: the last signature within the comment's reach
    std::size_t eocd = std::string::npos;
    for (std::size_t k = z.size() - 22 + 1; k-- > 0 && z.size() - k <= 22 + 0xffff;)
        if (u32(&z[k]) == 0x06054b50u) {
            eocd = k;
            break;
        }
    if (eocd == std::string::npos) throw Error("not a zip archive (no end of central directory)");
    const std::size_t entries = u16(&z[eocd + 10]);
    std::size_t at = u32(&z[eocd + 16]);
    std::map<std::string, std::vector<std::uint8_t>> out;
    for (std::size_t e = 0; e < entries; ++e) {
        if (at + 46 > z.size() || u32(&z[at]) != 0x02014b50u) throw Error("zip central directory is damaged");
        const std::uint32_t flags = u16(&z[at + 8]), method = u16(&z[at + 10]), crc = u32(&z[at + 16]);
        const std::size_t csize = u32(&z[at + 20]), usize = u32(&z[at + 24]);
        const std::size_t nlen = u16(&z[at + 28]), xlen = u16(&z[at + 30]), clen = u16(&z[at + 32]);
        const std::size_t local = u32(&z[at + 42]);
        const std::string name(reinterpret_cast<const char*>(&z[at + 46]), nlen);
        at += 46 + nlen + xlen + clen;
        if (csize == 0xffffffffu || usize == 0xffffffffu || local == 0xffffffffu)
            throw Error(name + ": zip64 members are not read");
        if (flags & 1u) throw Error(name + ": encrypted");
        if (local + 30 > z.size() || u32(&z[local]) != 0x04034b50u) throw Error(name + ": local header is damaged");
        const std::size_t data = local + 30 + u16(&z[local + 26]) + u16(&z[local + 28]);
        if (data + csize > z.size()) throw Error(name + ": member runs past the archive");
        std::vector<std::uint8_t> bytes;
        if (method == 0) bytes.assign(z.begin() + static_cast<std::ptrdiff_t>(data),
                                      z.begin() + static_cast<std::ptrdiff_t>(data + csize));
        else if (method == 8) bytes = inflate(&z[data], csize, usize);
        else throw Error(name + ": compression method " + std::to_string(method) + " is not read");
        if (bytes.size() != usize) throw Error(name + ": size is not the directory's");
        if (store::crc32_zlib(bytes.data(), bytes.size()) != crc) throw Error(name + ": CRC-32 does not match");
        out[name] = std::move(bytes);
    }
    return out;
}

// ── npy ─────────────────────────────────────────────────────────────────────

struct Array {
    std::string descr;                   // '<f8', '<i8', '|b1', '<U342', ...
    bool fortran_order = false;
    std::vector<std::size_t> shape;      // () for a 0-d array
    std::vector<std::uint8_t> data;

    std::size_t size() const
    {
        std::size_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
    // the values as doubles, in C order ('<f8' and '<i8')
    std::vector<double> doubles() const
    {
        const std::size_t n = size();
        std::vector<double> v(n);
        if (descr == "<f8") {
            if (data.size() != 8 * n) throw Error("float64 array of the wrong size");
            std::memcpy(v.data(), data.data(), 8 * n);
        } else if (descr == "<i8") {
            if (data.size() != 8 * n) throw Error("int64 array of the wrong size");
            for (std::size_t k = 0; k < n; ++k) {
                std::int64_t x;
                std::memcpy(&x, &data[8 * k], 8);
                v[k] = static_cast<double>(x);
            }
        } else {
            throw Error("an array of " + descr + " read as numbers");
        }
        if (fortran_order && shape.size() == 2) {          // to C order
            std::vector<double> c(n);
            for (std::size_t i = 0; i < shape[0]; ++i)
                for (std::size_t j = 0; j < shape[1]; ++j) c[i * shape[1] + j] = v[j * shape[0] + i];
            return c;
        }
        return v;
    }
    // a 0-d unicode array as UTF-8, its trailing NULs dropped as str(array) does
    std::string text() const
    {
        if (descr.rfind("<U", 0) != 0) throw Error("an array of " + descr + " read as text");
        std::string out;
        std::size_t end = data.size() / 4;
        auto cp = [this](std::size_t k) {
            std::uint32_t c;
            std::memcpy(&c, &data[4 * k], 4);
            return c;
        };
        while (end > 0 && cp(end - 1) == 0) --end;
        for (std::size_t k = 0; k < end; ++k) {
            const std::uint32_t c = cp(k);
            if (c < 0x80) {
                out += static_cast<char>(c);
            } else if (c < 0x800) {
                out += static_cast<char>(0xc0 | (c >> 6));
                out += static_cast<char>(0x80 | (c & 0x3f));
            } else if (c < 0x10000) {
                out += static_cast<char>(0xe0 | (c >> 12));
                out += static_cast<char>(0x80 | ((c >> 6) & 0x3f));
                out += static_cast<char>(0x80 | (c & 0x3f));
            } else {
                out += static_cast<char>(0xf0 | (c >> 18));
                out += static_cast<char>(0x80 | ((c >> 12) & 0x3f));
                out += static_cast<char>(0x80 | ((c >> 6) & 0x3f));
                out += static_cast<char>(0x80 | (c & 0x3f));
            }
        }
        return out;
    }
};

// an .npy file's bytes read
inline Array parse_npy(const std::vector<std::uint8_t>& b)
{
    static const char magic[] = "\x93NUMPY";
    if (b.size() < 10 || std::memcmp(b.data(), magic, 6) != 0) throw Error("not an .npy (magic)");
    const int major = b[6];
    std::size_t hlen, start;
    if (major == 1) {
        hlen = detail::u16(&b[8]);
        start = 10;
    } else if (major == 2 || major == 3) {
        if (b.size() < 12) throw Error("not an .npy (header)");
        hlen = detail::u32(&b[8]);
        start = 12;
    } else {
        throw Error(".npy version " + std::to_string(major) + " is not read");
    }
    if (start + hlen > b.size()) throw Error(".npy header runs past the file");
    const std::string h(reinterpret_cast<const char*>(&b[start]), hlen);
    Array a;
    // {'descr': '<f8', 'fortran_order': False, 'shape': (1000, 3), }
    auto value_of = [&h](const std::string& key) {
        const auto k = h.find("'" + key + "'");
        if (k == std::string::npos) throw Error(".npy header without " + key);
        auto v = h.find(':', k) + 1;
        while (v < h.size() && h[v] == ' ') ++v;
        return v;
    };
    std::size_t v = value_of("descr");
    if (h[v] != '\'') throw Error(".npy structured descr is not read");
    a.descr = h.substr(v + 1, h.find('\'', v + 1) - v - 1);
    v = value_of("fortran_order");
    a.fortran_order = h.compare(v, 4, "True") == 0;
    v = value_of("shape");
    if (h[v] != '(') throw Error(".npy shape is not a tuple");
    const std::size_t close = h.find(')', v);
    std::string inner = h.substr(v + 1, close - v - 1);
    std::stringstream ss(inner);
    std::string item;
    while (std::getline(ss, item, ',')) {
        std::size_t p = 0;
        while (p < item.size() && item[p] == ' ') ++p;
        if (p == item.size()) continue;
        a.shape.push_back(static_cast<std::size_t>(std::stoull(item.substr(p))));
    }
    if (a.descr.size() > 1 && a.descr[0] == '>') throw Error("big-endian " + a.descr + " is not read");
    if (a.descr == "|O") throw Error("object arrays are not read");
    a.data.assign(b.begin() + static_cast<std::ptrdiff_t>(start + hlen), b.end());
    return a;
}

inline std::vector<std::uint8_t> read_file(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) throw Error("cannot open " + path.string());
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// an .npz's arrays by name (the member's name without ".npy")
inline std::map<std::string, Array> load(const std::filesystem::path& path)
{
    std::map<std::string, Array> out;
    for (auto& [name, bytes] : unzip(read_file(path))) {
        const std::string key = name.size() > 4 && name.compare(name.size() - 4, 4, ".npy") == 0
                                    ? name.substr(0, name.size() - 4)
                                    : name;
        out[key] = parse_npy(bytes);
    }
    return out;
}

}   // namespace mootation::npz
