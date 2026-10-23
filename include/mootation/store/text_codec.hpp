#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The store's own compression of bytes without a model of their own — the
// text of meta.json, a table's header, a trajectory's texts, any other bytes
// (task 17, §3а.2: "LZ77 and an entropy coder; Deflate is not needed").
//
// LZ77 (Ziv & Lempel 1977): the bytes are a sequence of literals and matches,
// a match a copy of `length` bytes from `distance` bytes back (it may overlap
// what it writes: a run). The matches are found greedily — at each byte the
// longest match among the last kChain positions whose next three bytes hash
// alike, up to kWindow back — with one byte of lookahead (a match at the next
// byte longer by two or more wins over the one here, as gzip's lazy
// matching does).
//
// THE CODING, every bit through the adaptive range coder of range_coder.hpp:
//   * literal or match: a bit, its context the kind of the previous token;
//   * a literal: its 8 bits by a binary tree, one tree for each of 8 classes
//     of the byte before it (its top three bits: digits, letters, ... );
//   * a match's length - 3: its bit length by a tree of 5 bits, the bits
//     below the top one by a tree for each bit length (up to 4 of them,
//     the rest direct);
//   * a match's distance - 1: the same, a tree of 5 bits for the bit length,
//     the 4 bits below the top one by a tree per bit length, the rest direct.
//
// THE BYTES: the length of the input (varint), the range coder's bytes. A
// decoder checks that the length comes out and the coder's bytes end there.
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "column_codec.hpp"
#include "range_coder.hpp"

namespace mootation::store {

namespace text {

constexpr std::size_t kMinMatch = 3;
constexpr std::size_t kMaxMatch = 3 + (std::size_t{1} << 16) - 1;   // length - 3 in 16 bits
constexpr std::size_t kWindow = std::size_t{1} << 20;
constexpr int kChain = 64;
constexpr int kHashBits = 15;

struct Models {
    BitModel kind[2];
    BitModel literal[8][256];
    BitModel len_width[32];                 // the bit length of length - 3 (0..16): a tree of 5 bits
    BitModel len_bits[17][16];              // per bit length, up to 4 bits below the leading one
    BitModel dist_width[32];                // the bit length of distance - 1 (0..20)
    BitModel dist_bits[21][16];
};

inline void put_tree(RangeEncoder& rc, BitModel* tree, std::uint32_t v, int bits)
{
    std::uint32_t node = 1;
    for (int i = bits - 1; i >= 0; --i) {
        const int b = static_cast<int>((v >> i) & 1u);
        rc.bit(tree[node], b);
        node = (node << 1) | static_cast<std::uint32_t>(b);
    }
}

inline std::uint32_t get_tree(RangeDecoder& rc, BitModel* tree, int bits)
{
    std::uint32_t node = 1;
    for (int i = 0; i < bits; ++i) node = (node << 1) | static_cast<std::uint32_t>(rc.bit(tree[node]));
    return node - (std::uint32_t{1} << bits);
}

// a number by its bit length w (a tree of `wbits` bits) and the bits below
// its top one: up to 4 of them by a tree for w, the rest direct
inline void put_number(RangeEncoder& rc, BitModel* width_tree, int wbits, BitModel (*below)[16],
                       std::uint32_t v)
{
    const int w = col::bit_width(v);
    put_tree(rc, width_tree, static_cast<std::uint32_t>(w), wbits);
    if (w <= 1) return;
    const int rest = w - 1;
    const int modelled = rest < 4 ? rest : 4;
    put_tree(rc, below[w], (v >> (rest - modelled)) & ((1u << modelled) - 1u), modelled);
    rc.direct(v, rest - modelled);
}

inline std::uint32_t get_number(RangeDecoder& rc, BitModel* width_tree, int wbits, BitModel (*below)[16])
{
    const int w = static_cast<int>(get_tree(rc, width_tree, wbits));
    if (w <= 1) return static_cast<std::uint32_t>(w);
    const int rest = w - 1;
    const int modelled = rest < 4 ? rest : 4;
    std::uint32_t v = 1;
    v = (v << modelled) | get_tree(rc, below[w], modelled);
    v = (v << (rest - modelled)) | static_cast<std::uint32_t>(rc.direct(rest - modelled));
    return v;
}

inline std::uint32_t hash3(const std::uint8_t* p)
{
    const std::uint32_t v = static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
                            (static_cast<std::uint32_t>(p[2]) << 16);
    return (v * 2654435761u) >> (32 - kHashBits);
}

struct Matcher {
    const std::uint8_t* d;
    std::size_t n;
    std::vector<std::int64_t> head, prev;

    Matcher(const std::uint8_t* data, std::size_t size)
        : d(data), n(size), head(std::size_t{1} << kHashBits, -1), prev(size, -1) {}

    void insert(std::size_t i)
    {
        if (i + kMinMatch > n) return;
        const std::uint32_t h = hash3(d + i);
        prev[i] = head[h];
        head[h] = static_cast<std::int64_t>(i);
    }

    // the longest match at i among the chain: (length, distance), length 0 for none
    std::pair<std::size_t, std::size_t> best(std::size_t i) const
    {
        std::size_t len = 0, dist = 0;
        if (i + kMinMatch > n) return {0, 0};
        std::int64_t j = head[hash3(d + i)];
        const std::size_t limit = n - i < kMaxMatch ? n - i : kMaxMatch;
        for (int c = 0; c < kChain && j >= 0; ++c, j = prev[static_cast<std::size_t>(j)]) {
            const std::size_t s = static_cast<std::size_t>(j);
            if (i - s > kWindow) break;
            std::size_t l = 0;
            while (l < limit && d[s + l] == d[i + l]) ++l;
            if (l > len) {
                len = l;
                dist = i - s;
                if (l == limit) break;
            }
        }
        return len >= kMinMatch ? std::make_pair(len, dist) : std::make_pair(std::size_t{0}, std::size_t{0});
    }
};

}   // namespace text

inline std::vector<std::uint8_t> encode_text(const std::uint8_t* d, std::size_t n)
{
    std::vector<std::uint8_t> out;
    col::put_varint(out, n);
    RangeEncoder rc;
    auto m = std::make_unique<text::Models>();
    text::Matcher mt(d, n);
    int last = 0;
    std::size_t i = 0;
    while (i < n) {
        auto here = mt.best(i);
        if (here.first && i + 1 < n) {               // one byte of lookahead
            mt.insert(i);
            const auto next = mt.best(i + 1);
            if (next.first >= here.first + 2) here = {0, 0};
        } else {
            mt.insert(i);
        }
        if (here.first == 0) {
            rc.bit(m->kind[last], 0);
            last = 0;
            const int cls = i ? d[i - 1] >> 5 : 0;
            text::put_tree(rc, m->literal[cls], d[i], 8);
            ++i;
        } else {
            rc.bit(m->kind[last], 1);
            last = 1;
            text::put_number(rc, m->len_width, 5, m->len_bits, static_cast<std::uint32_t>(here.first - text::kMinMatch));
            text::put_number(rc, m->dist_width, 5, m->dist_bits, static_cast<std::uint32_t>(here.second - 1));
            for (std::size_t k = 1; k < here.first; ++k) mt.insert(i + k);
            i += here.first;
        }
    }
    const std::vector<std::uint8_t> coded = rc.finish();
    out.insert(out.end(), coded.begin(), coded.end());
    return out;
}

inline std::vector<std::uint8_t> decode_text(const std::uint8_t* data, std::size_t size)
{
    std::size_t pos = 0;
    const std::uint64_t n = col::get_varint(data, size, pos);
    if (n > (std::uint64_t{1} << 36)) throw std::runtime_error("text: a length past any record");
    std::vector<std::uint8_t> out;
    out.reserve(static_cast<std::size_t>(n));
    RangeDecoder rc(data + pos, size - pos);
    auto m = std::make_unique<text::Models>();
    int last = 0;
    while (out.size() < n) {
        const int kind = rc.bit(m->kind[last]);
        last = kind;
        if (kind == 0) {
            const int cls = out.empty() ? 0 : out.back() >> 5;
            out.push_back(static_cast<std::uint8_t>(text::get_tree(rc, m->literal[cls], 8)));
        } else {
            const std::size_t len = text::get_number(rc, m->len_width, 5, m->len_bits) + text::kMinMatch;
            const std::size_t dist = text::get_number(rc, m->dist_width, 5, m->dist_bits) + std::size_t{1};
            if (dist > out.size() || len > n - out.size()) throw std::runtime_error("text: a copy past the bytes");
            const std::size_t from = out.size() - dist;
            for (std::size_t k = 0; k < len; ++k) out.push_back(out[from + k]);
        }
    }
    if (rc.consumed() != size - pos) throw std::runtime_error("text: the coded bytes do not end where the text does");
    return out;
}

}   // namespace mootation::store
