#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Deflate (RFC 1951) written in C++, with the gzip (RFC 1952) and zip (PKWARE
// APPNOTE 6.3, without zip64) frames around it — task 17, §3а.3: `mootation
// store export` writes the .csv.gz and .npz files of the files layout without
// zlib. The reading side is npz.hpp's inflate.
//
// LZ77 over the 32 KiB window: a hash of the next three bytes and the chain
// of the earlier places with that hash (at most kMaxChain of them looked at);
// the longest match is taken (greedy). Each block of up to kBlockSymbols
// symbols gets its own Huffman codes (a dynamic block, RFC 1951 §3.2.7), at
// most 15 bits long (7 for the code-length code); a block whose codes would
// take more bits than its bytes is stored as it is. The bytes are not zlib's;
// what they inflate to is the same.
// ============================================================================

#include <cstdint>
#include <functional>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../store/crc32c.hpp"

namespace mootation::deflate {

namespace detail {

constexpr int kWindow = 32768;
constexpr int kMinMatch = 3;
constexpr int kMaxMatch = 258;
constexpr int kMaxChain = 128;
constexpr int kHashBits = 15;
constexpr std::uint32_t kHashMask = (1u << kHashBits) - 1;
constexpr std::size_t kBlockSymbols = std::size_t{1} << 15;

// RFC 1951 §3.2.5: length codes 257..285 and distance codes 0..29
constexpr int kLengthBase[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
constexpr int kLengthExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr int kDistBase[30] = {1,   2,   3,   4,   5,   7,    9,    13,   17,   25,   33,   49,   65,    97,    129,
                               193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
constexpr int kDistExtra[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
// §3.2.7: the order the code-length code's lengths are written in
constexpr int kClOrder[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

// a literal (len == 0) or a match of len bytes dist bytes back
struct Symbol {
    std::uint16_t len = 0;
    std::uint16_t dist = 0;
    std::uint8_t lit = 0;
};

inline int length_code(int len)      // index into kLengthBase
{
    int k = 28;
    while (kLengthBase[k] > len) --k;
    return k;
}

inline int dist_code(int dist)
{
    int k = 29;
    while (kDistBase[k] > dist) --k;
    return k;
}

// bits written from the least significant end, as deflate packs them
struct BitWriter {
    std::vector<std::uint8_t> out;
    std::uint64_t acc = 0;
    int n = 0;

    void put(std::uint32_t bits, int count)
    {
        acc |= static_cast<std::uint64_t>(bits) << n;
        n += count;
        while (n >= 8) {
            out.push_back(static_cast<std::uint8_t>(acc & 0xFF));
            acc >>= 8;
            n -= 8;
        }
    }
    void align()
    {
        if (n > 0) out.push_back(static_cast<std::uint8_t>(acc & 0xFF));
        acc = 0;
        n = 0;
    }
};

// Huffman code lengths of the frequencies, at most `limit` bits, a complete
// code (Kraft's sum exactly 1: zlib's inflate takes no other): the depths of
// the Huffman tree, those past the limit cut to it; then the longest code
// that can still grow lengthened by a bit while the sum is over 1, and the
// longest one that may shrink shortened by a bit while it is under. A single
// symbol gets a code of one bit and a partner of one bit (as zlib does).
inline std::vector<int> code_lengths(const std::vector<std::uint32_t>& freq, int limit)
{
    const std::size_t n = freq.size();
    std::vector<int> len(n, 0);
    std::vector<std::uint64_t> weight;
    std::vector<std::size_t> parent, leaf_of;
    using Item = std::pair<std::uint64_t, std::size_t>;          // (weight, node), the lightest first
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
    for (std::size_t s = 0; s < n; ++s)
        if (freq[s]) {
            heap.emplace(freq[s], weight.size());
            weight.push_back(freq[s]);
            leaf_of.push_back(s);
        }
    const std::size_t leaves = weight.size();
    if (leaves == 0) return len;
    if (leaves == 1) {
        len[leaf_of[0]] = 1;
        len[leaf_of[0] == 0 ? 1 : 0] = 1;
        return len;
    }
    parent.assign(leaves, 0);
    while (heap.size() > 1) {
        const Item a = heap.top();
        heap.pop();
        const Item b = heap.top();
        heap.pop();
        const std::size_t node = weight.size();
        weight.push_back(a.first + b.first);
        parent.push_back(node);                           // the root's own, never read
        parent[a.second] = parent[b.second] = node;
        heap.emplace(a.first + b.first, node);
    }
    const std::size_t root = weight.size() - 1;
    for (std::size_t k = 0; k < leaves; ++k) {
        int depth = 0;
        for (std::size_t v = k; v != root; v = parent[v]) ++depth;
        len[leaf_of[k]] = depth < limit ? depth : limit;
    }
    // Kraft's sum in units of 2^-limit
    const std::uint64_t full = std::uint64_t{1} << limit;
    std::uint64_t kraft = 0;
    for (int l : len)
        if (l) kraft += std::uint64_t{1} << (limit - l);
    while (kraft > full) {
        std::size_t pick = n;
        for (std::size_t s = 0; s < n; ++s)
            if (len[s] && len[s] < limit && (pick == n || len[s] > len[pick] || (len[s] == len[pick] && freq[s] < freq[pick])))
                pick = s;
        kraft -= std::uint64_t{1} << (limit - len[pick] - 1);
        ++len[pick];
    }
    // every term is a multiple of the longest code's, so is what is missing:
    // the longest code shortened fills it, step by step
    while (kraft < full) {
        std::size_t pick = n;
        for (std::size_t s = 0; s < n; ++s)
            if (len[s] > 1 && kraft + (std::uint64_t{1} << (limit - len[s])) <= full &&
                (pick == n || len[s] > len[pick] || (len[s] == len[pick] && freq[s] > freq[pick])))
                pick = s;
        kraft += std::uint64_t{1} << (limit - len[pick]);
        --len[pick];
    }
    return len;
}

// RFC 1951 §3.2.2: the canonical codes of the lengths, bit-reversed for the writer
inline std::vector<std::uint32_t> codes_of(const std::vector<int>& len)
{
    int max_len = 0;
    for (int l : len) max_len = l > max_len ? l : max_len;
    std::vector<std::uint32_t> count(static_cast<std::size_t>(max_len) + 1, 0), next(count.size(), 0);
    for (int l : len)
        if (l) ++count[static_cast<std::size_t>(l)];
    std::uint32_t code = 0;
    for (int b = 1; b <= max_len; ++b) {
        code = (code + count[static_cast<std::size_t>(b - 1)]) << 1;
        next[static_cast<std::size_t>(b)] = code;
    }
    std::vector<std::uint32_t> out(len.size(), 0);
    for (std::size_t s = 0; s < len.size(); ++s) {
        const int l = len[s];
        if (!l) continue;
        const std::uint32_t c = next[static_cast<std::size_t>(l)]++;
        std::uint32_t r = 0;
        for (int b = 0; b < l; ++b) r |= ((c >> b) & 1u) << (l - 1 - b);
        out[s] = r;
    }
    return out;
}

// the lengths of both codes as code-length symbols (0..15 a length, 16 the
// previous length 3..6 times, 17 zero 3..10 times, 18 zero 11..138 times),
// each with the value of its extra bits
inline std::vector<std::pair<int, int>> length_runs(const std::vector<int>& lengths)
{
    std::vector<std::pair<int, int>> out;
    for (std::size_t i = 0; i < lengths.size();) {
        const int l = lengths[i];
        std::size_t run = 1;
        while (i + run < lengths.size() && lengths[i + run] == l) ++run;
        i += run;
        if (l == 0) {
            while (run >= 11) {
                const std::size_t k = run < 138 ? run : 138;
                out.emplace_back(18, static_cast<int>(k - 11));
                run -= k;
            }
            if (run >= 3) {
                out.emplace_back(17, static_cast<int>(run - 3));
                run = 0;
            }
            for (; run; --run) out.emplace_back(0, 0);
            continue;
        }
        out.emplace_back(l, 0);
        --run;
        while (run >= 3) {
            const std::size_t k = run < 6 ? run : 6;
            out.emplace_back(16, static_cast<int>(k - 3));
            run -= k;
        }
        for (; run; --run) out.emplace_back(l, 0);
    }
    return out;
}

inline void stored_block(BitWriter& w, const std::uint8_t* data, std::size_t n, bool last)
{
    std::size_t at = 0;
    do {
        const std::size_t k = n - at < 65535 ? n - at : 65535;
        w.put(last && at + k == n ? 1u : 0u, 1);
        w.put(0, 2);
        w.align();
        w.put(static_cast<std::uint32_t>(k), 16);
        w.put(static_cast<std::uint32_t>(~k & 0xFFFF), 16);
        w.out.insert(w.out.end(), data + at, data + at + k);
        at += k;
    } while (at < n);
}

// one block of symbols that stand for data[begin, end)
inline void write_block(BitWriter& w, const std::vector<Symbol>& syms, const std::uint8_t* data, std::size_t begin,
                        std::size_t end, bool last)
{
    std::vector<std::uint32_t> lf(286, 0), df(30, 0);
    for (const Symbol& s : syms) {
        if (s.len == 0) {
            ++lf[s.lit];
        } else {
            ++lf[static_cast<std::size_t>(257 + length_code(s.len))];
            ++df[static_cast<std::size_t>(dist_code(s.dist))];
        }
    }
    lf[256] = 1;                                         // the end of the block
    const std::vector<int> ll = code_lengths(lf, 15);
    std::vector<int> dl = code_lengths(df, 15);
    bool any_dist = false;
    for (int l : dl) any_dist = any_dist || l;
    if (!any_dist) dl[0] = 1;                            // one distance code, unused (as zlib writes)
    std::size_t hlit = 286, hdist = 30;
    while (hlit > 257 && ll[hlit - 1] == 0) --hlit;
    while (hdist > 1 && dl[hdist - 1] == 0) --hdist;
    std::vector<int> both(ll.begin(), ll.begin() + static_cast<std::ptrdiff_t>(hlit));
    both.insert(both.end(), dl.begin(), dl.begin() + static_cast<std::ptrdiff_t>(hdist));
    const auto runs = length_runs(both);
    std::vector<std::uint32_t> cf(19, 0);
    for (const auto& r : runs) ++cf[static_cast<std::size_t>(r.first)];
    const std::vector<int> cl = code_lengths(cf, 7);
    std::size_t hclen = 19;
    while (hclen > 4 && cl[static_cast<std::size_t>(kClOrder[hclen - 1])] == 0) --hclen;

    // the block's bits, against its bytes stored
    std::uint64_t bits = 3 + 5 + 5 + 4 + 3 * hclen;
    for (const auto& r : runs)
        bits += static_cast<std::uint64_t>(cl[static_cast<std::size_t>(r.first)]) +
                (r.first == 16 ? 2 : r.first == 17 ? 3 : r.first == 18 ? 7 : 0);
    for (const Symbol& s : syms) {
        if (s.len == 0) {
            bits += static_cast<std::uint64_t>(ll[s.lit]);
        } else {
            const int lc = length_code(s.len), dc = dist_code(s.dist);
            bits += static_cast<std::uint64_t>(ll[static_cast<std::size_t>(257 + lc)] + kLengthExtra[lc] +
                                               dl[static_cast<std::size_t>(dc)] + kDistExtra[dc]);
        }
    }
    bits += static_cast<std::uint64_t>(ll[256]);
    const std::size_t n = end - begin;
    const std::uint64_t stored_bits = 8 * (static_cast<std::uint64_t>(n) + 5 * (n / 65535 + 1)) + 7;
    if (bits >= stored_bits) {
        stored_block(w, data + begin, n, last);
        return;
    }

    const auto lcode = codes_of(ll), dcode = codes_of(dl), ccode = codes_of(cl);
    w.put(last ? 1u : 0u, 1);
    w.put(2, 2);                                         // dynamic Huffman codes
    w.put(static_cast<std::uint32_t>(hlit - 257), 5);
    w.put(static_cast<std::uint32_t>(hdist - 1), 5);
    w.put(static_cast<std::uint32_t>(hclen - 4), 4);
    for (std::size_t k = 0; k < hclen; ++k) w.put(static_cast<std::uint32_t>(cl[static_cast<std::size_t>(kClOrder[k])]), 3);
    for (const auto& r : runs) {
        const std::size_t c = static_cast<std::size_t>(r.first);
        w.put(ccode[c], cl[c]);
        if (r.first == 16) w.put(static_cast<std::uint32_t>(r.second), 2);
        if (r.first == 17) w.put(static_cast<std::uint32_t>(r.second), 3);
        if (r.first == 18) w.put(static_cast<std::uint32_t>(r.second), 7);
    }
    for (const Symbol& s : syms) {
        if (s.len == 0) {
            w.put(lcode[s.lit], ll[s.lit]);
            continue;
        }
        const int lc = length_code(s.len), dc = dist_code(s.dist);
        const std::size_t li = static_cast<std::size_t>(257 + lc), di = static_cast<std::size_t>(dc);
        w.put(lcode[li], ll[li]);
        w.put(static_cast<std::uint32_t>(s.len - kLengthBase[lc]), kLengthExtra[lc]);
        w.put(dcode[di], dl[di]);
        w.put(static_cast<std::uint32_t>(s.dist - kDistBase[dc]), kDistExtra[dc]);
    }
    w.put(lcode[256], ll[256]);
}

}   // namespace detail

// The raw deflate stream of data[0, n).
inline std::vector<std::uint8_t> compress(const std::uint8_t* data, std::size_t n)
{
    using namespace detail;
    BitWriter w;
    if (n == 0) {
        write_block(w, {}, data, 0, 0, true);
        w.align();
        return w.out;
    }
    auto hash = [data](std::size_t i) {
        return ((static_cast<std::uint32_t>(data[i]) << 10) ^ (static_cast<std::uint32_t>(data[i + 1]) << 5) ^ data[i + 2]) &
               kHashMask;
    };
    std::vector<std::int64_t> head(std::size_t{1} << kHashBits, -1), prev(n, -1);
    auto insert = [&](std::size_t i) {
        if (i + kMinMatch > n) return;
        const std::uint32_t h = hash(i);
        prev[i] = head[h];
        head[h] = static_cast<std::int64_t>(i);
    };
    std::vector<Symbol> syms;
    std::size_t block_begin = 0;
    for (std::size_t i = 0; i < n;) {
        int best = 0, best_dist = 0;
        if (i + kMinMatch <= n) {
            const std::size_t most = n - i < static_cast<std::size_t>(kMaxMatch) ? n - i : kMaxMatch;
            int chain = 0;
            for (std::int64_t j = head[hash(i)]; j >= 0 && i - static_cast<std::size_t>(j) <= static_cast<std::size_t>(kWindow) &&
                                                 chain < kMaxChain;
                 j = prev[static_cast<std::size_t>(j)], ++chain) {
                const std::uint8_t* a = data + j;
                const std::uint8_t* b = data + i;
                std::size_t len = 0;
                while (len < most && a[len] == b[len]) ++len;
                if (static_cast<int>(len) > best) {
                    best = static_cast<int>(len);
                    best_dist = static_cast<int>(i - static_cast<std::size_t>(j));
                    if (len == most) break;
                }
            }
        }
        Symbol s;
        if (best >= kMinMatch) {
            s.len = static_cast<std::uint16_t>(best);
            s.dist = static_cast<std::uint16_t>(best_dist);
            for (int k = 0; k < best; ++k) insert(i + static_cast<std::size_t>(k));
            i += static_cast<std::size_t>(best);
        } else {
            s.lit = data[i];
            insert(i);
            ++i;
        }
        syms.push_back(s);
        if (syms.size() == kBlockSymbols || i == n) {
            write_block(w, syms, data, block_begin, i, i == n);
            syms.clear();
            block_begin = i;
        }
    }
    w.align();
    return w.out;
}

namespace detail {
inline void put_le(std::vector<std::uint8_t>& out, std::uint64_t v, int bytes)
{
    for (int k = 0; k < bytes; ++k) out.push_back(static_cast<std::uint8_t>(v >> (8 * k)));
}
}   // namespace detail

// A gzip file of data[0, n): no name, no time stamp (as Python's
// GzipFile(filename="", mtime=0) writes it), the OS "unknown".
inline std::vector<std::uint8_t> gzip(const std::uint8_t* data, std::size_t n)
{
    std::vector<std::uint8_t> out = {0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 2, 255};
    const auto body = compress(data, n);
    out.insert(out.end(), body.begin(), body.end());
    detail::put_le(out, store::crc32_zlib(data, n), 4);
    detail::put_le(out, static_cast<std::uint64_t>(n) & 0xFFFFFFFFu, 4);
    return out;
}

// A zip archive of the files [(name, bytes)], each deflated, dated
// 1980-01-01 00:00 (so that the same files give the same archive).
inline std::vector<std::uint8_t> zip(const std::vector<std::pair<std::string, std::vector<std::uint8_t>>>& files)
{
    using detail::put_le;
    constexpr std::uint32_t kDate = (0u << 9) | (1u << 5) | 1u;   // DOS date: 1980, January, 1st
    std::vector<std::uint8_t> out, central;
    for (const auto& file : files) {
        const std::string& name = file.first;
        const std::vector<std::uint8_t>& bytes = file.second;
        const auto packed = compress(bytes.data(), bytes.size());
        const std::uint32_t crc = store::crc32_zlib(bytes.data(), bytes.size());
        if (bytes.size() >= 0xFFFFFFFFu || packed.size() >= 0xFFFFFFFFu || out.size() >= 0xFFFFFFFFu || files.size() >= 0xFFFF)
            throw std::runtime_error("zip: " + name + ": past 4 GiB or 65535 files (zip64 is not written)");
        const std::uint64_t at = out.size();
        auto common = [&](std::vector<std::uint8_t>& v) {
            put_le(v, 20, 2);                            // the version needed: deflate
            put_le(v, 0, 2);                             // flags
            put_le(v, 8, 2);                             // method: deflate
            put_le(v, 0, 2);                             // time
            put_le(v, kDate, 2);
            put_le(v, crc, 4);
            put_le(v, packed.size(), 4);
            put_le(v, bytes.size(), 4);
            put_le(v, name.size(), 2);
            put_le(v, 0, 2);                             // extra field
        };
        put_le(out, 0x04034b50, 4);
        common(out);
        out.insert(out.end(), name.begin(), name.end());
        out.insert(out.end(), packed.begin(), packed.end());
        put_le(central, 0x02014b50, 4);
        put_le(central, 20, 2);                          // made by
        common(central);
        put_le(central, 0, 2);                           // comment
        put_le(central, 0, 2);                           // disk
        put_le(central, 0, 2);                           // internal attributes
        put_le(central, 0, 4);                           // external attributes
        put_le(central, at, 4);
        central.insert(central.end(), name.begin(), name.end());
    }
    const std::uint64_t cd_at = out.size();
    out.insert(out.end(), central.begin(), central.end());
    put_le(out, 0x06054b50, 4);
    put_le(out, 0, 2);
    put_le(out, 0, 2);
    put_le(out, files.size(), 2);
    put_le(out, files.size(), 2);
    put_le(out, central.size(), 4);
    put_le(out, cd_at, 4);
    put_le(out, 0, 2);
    return out;
}

}   // namespace mootation::deflate
