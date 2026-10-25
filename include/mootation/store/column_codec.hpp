#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The store's codecs of a column (task 17, stage 4; the owner's decisions,
// supplement 5, §2): float64 and integers, without loss, bit for bit.
//
// FLOAT64 (§2.1). The values are coded as their 64 bits, so every value comes
// back as it was: NaN with its payload, -0.0, infinities, subnormals. Each
// value is either
//   * one met before in the column: a flag and its index among the column's
//     distinct values (in the order they first came), the index's bits by an
//     adaptive binary tree (a node per prefix, for each width of the index);
//     the measure of step 1: 10-24 % of a column repeats (a population's
//     points that are in the archive, genes copied by a crossover);
//   * new: a flag, its top k bits — sign, exponent and the mantissa's start —
//     by an adaptive binary tree, the 64 - k below as direct bits. k (8..28)
//     is the column's, in its header: the one of the least cost, the tree's
//     cost estimated from the counts at its nodes (the Krichevsky-Trofimov
//     code length, which the adaptive nodes follow; logarithms by detmath, so
//     that every platform chooses the same k and writes the same bytes).
// No predictor: none was shorter on any column of step 1 (the points of a set
// are not a series). The flag's probability depends on the previous flag.
// A column the model does not shorten is kept raw (8 bytes a value).
//
// DECIMAL (the idea of ALP, Afroozeh, Kuffo & Boncz 2023): a column whose every
// value is an integer d over 10^e (e = 0..18, the least that holds for all),
// clocks rounded to milliseconds, counts kept as float64, is kept as e and the
// integers d (below), where that is shorter. A value is such an integer when
// d = llround(x 10^e) gives x back bit for bit as d / 10^e (one IEEE
// division, correctly rounded, the same everywhere; 10^e is exact in a
// double); -0.0, NaN and the infinities never are.
//
// INTEGERS (§2.2): the shortest of raw, one constant, frame of reference (the
// minimum and every value's offset from it in a fixed number of bits), the
// same over the differences after the first value (zigzag: 0, -1, 1, -2 ...
// to 0, 1, 2, 3 ...), and runs of equal values (Lemire & Boytsov 2015 for
// the frame of reference and the differences).
//
// A COLUMN'S BYTES: u8 method, the count of values (varint), the method's
// data. A decoder checks that it read the bytes and the count it was given;
// the container checks the bytes themselves (CRC32C).
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../detmath.hpp"
#include "range_coder.hpp"

namespace mootation::store {

namespace col {

enum Method : std::uint8_t {
    kRaw = 0,
    kModel = 1,        // float64: references and top k bits by the range coder
    kConstant = 2,     // integers: one value
    kFor = 3,          // integers: frame of reference
    kDeltaFor = 4,     // integers: the first value, then the differences' frame of reference
    kRuns = 5,         // integers: (value, run) pairs
    kDecimal = 6       // float64: e, then the integers x 10^e (an integer column)
};

// 10^0 .. 10^18, each exact in a double
constexpr double kPow10[19] = {1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8, 1e9,
                               1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18};

inline void put_varint(std::vector<std::uint8_t>& out, std::uint64_t v)
{
    while (v >= 0x80) {
        out.push_back(static_cast<std::uint8_t>(v | 0x80));
        v >>= 7;
    }
    out.push_back(static_cast<std::uint8_t>(v));
}

inline std::uint64_t get_varint(const std::uint8_t* data, std::size_t size, std::size_t& pos)
{
    std::uint64_t v = 0;
    for (int shift = 0; shift < 64; shift += 7) {
        if (pos >= size) throw std::runtime_error("column: the bytes end inside a number");
        const std::uint8_t b = data[pos++];
        v |= static_cast<std::uint64_t>(b & 0x7F) << shift;
        if (!(b & 0x80)) return v;
    }
    throw std::runtime_error("column: a number longer than 64 bits");
}

inline void put_u64(std::vector<std::uint8_t>& out, std::uint64_t v)
{
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}

inline std::uint64_t get_u64(const std::uint8_t* data, std::size_t size, std::size_t& pos)
{
    if (pos > size || size - pos < 8) throw std::runtime_error("column: the bytes end inside a value");
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(data[pos + i]) << (8 * i);
    pos += 8;
    return v;
}

inline int bit_width(std::uint64_t v)
{
    int n = 0;
    while (v) {
        ++n;
        v >>= 1;
    }
    return n;
}

// bits written least significant first, in bytes
class BitWriter {
public:
    explicit BitWriter(std::vector<std::uint8_t>& out) : out_(out) {}
    void put(std::uint64_t v, int n)
    {
        for (int i = 0; i < n; ++i) {
            if (used_ == 0) out_.push_back(0);
            if ((v >> i) & 1u) out_.back() = static_cast<std::uint8_t>(out_.back() | (1u << used_));
            used_ = (used_ + 1) & 7;
        }
    }

private:
    std::vector<std::uint8_t>& out_;
    int used_ = 0;
};

class BitReader {
public:
    BitReader(const std::uint8_t* data, std::size_t size, std::size_t pos) : data_(data), size_(size), pos_(pos) {}
    std::uint64_t get(int n)
    {
        std::uint64_t v = 0;
        for (int i = 0; i < n; ++i) {
            if (pos_ >= size_) throw std::runtime_error("column: the bytes end inside the packed bits");
            v |= static_cast<std::uint64_t>((data_[pos_] >> used_) & 1u) << i;
            if (++used_ == 8) {
                used_ = 0;
                ++pos_;
            }
        }
        return v;
    }
    std::size_t end() const { return pos_ + (used_ ? 1 : 0); }

private:
    const std::uint8_t* data_;
    std::size_t size_, pos_;
    int used_ = 0;
};

// The models of a tree's nodes by their key (not 0): open addressing, grown
// as the tree is (a node exists once a value has passed it).
class NodeTable {
public:
    BitModel& at(std::uint64_t key)
    {
        if (2 * (used_ + 1) > keys_.size()) grow();
        const std::size_t i = slot(key);
        if (keys_[i] == 0) {
            keys_[i] = key;
            ++used_;
        }
        return models_[i];
    }

private:
    std::size_t slot(std::uint64_t key) const
    {
        std::uint64_t h = key ^ (key >> 33);            // a 64-bit mix (MurmurHash3's finaliser)
        h *= 0xff51afd7ed558ccdull;
        h ^= h >> 33;
        const std::size_t mask = keys_.size() - 1;
        std::size_t i = static_cast<std::size_t>(h) & mask;
        while (keys_[i] != 0 && keys_[i] != key) i = (i + 1) & mask;
        return i;
    }

    void grow()
    {
        std::vector<std::uint64_t> keys = std::move(keys_);
        std::vector<BitModel> models = std::move(models_);
        keys_.assign(keys.empty() ? 1024 : 2 * keys.size(), 0);
        models_.assign(keys_.size(), BitModel{});
        for (std::size_t j = 0; j < keys.size(); ++j)
            if (keys[j] != 0) {
                const std::size_t i = slot(keys[j]);
                keys_[i] = keys[j];
                models_[i] = models[j];
            }
    }

    std::vector<std::uint64_t> keys_;
    std::vector<BitModel> models_;
    std::size_t used_ = 0;
};

// The models of a column: the flags, the trees of the indices (one per width
// of the index, a node at its prefix with a leading 1: an array of 2^width,
// made when the width is first used) and the tree of the values' top bits
// (its top 12 levels, the nodes every value passes, in an array; the deeper
// nodes by their prefix in a NodeTable).
constexpr std::uint64_t kDense = 1u << 12;

struct FloatModels {
    BitModel flag[2];
    std::vector<BitModel> top;              // the nodes of prefix below its size, in an array
    std::vector<BitModel> index[33];
    NodeTable value;

    // n: the column's values; the array holds no more of the tree's top than
    // n values can reach (8 n nodes, at least 64), so that a short column does
    // not fill 4096 of them. Where a node is kept changes nothing it codes.
    explicit FloatModels(std::size_t n)
    {
        std::uint64_t size = 64;
        while (size < kDense && size < 8 * static_cast<std::uint64_t>(n)) size <<= 1;
        top.assign(static_cast<std::size_t>(size), BitModel{});
    }
};

inline BitModel& value_node(FloatModels& m, std::uint64_t prefix)
{
    return prefix < m.top.size() ? m.top[prefix] : m.value.at(prefix);
}

inline std::vector<BitModel>& index_tree(FloatModels& m, int width)
{
    if (m.index[width].empty()) m.index[width].assign(std::size_t{1} << width, BitModel{});
    return m.index[width];
}

inline void code_value(RangeEncoder& rc, FloatModels& m, std::uint64_t sym, int k)
{
    std::uint64_t prefix = 1;
    for (int d = 0; d < k; ++d) {
        const int b = static_cast<int>((sym >> (k - 1 - d)) & 1u);
        rc.bit(value_node(m, prefix), b);
        prefix = (prefix << 1) | static_cast<std::uint64_t>(b);
    }
}

inline std::uint64_t decode_value(RangeDecoder& rc, FloatModels& m, int k)
{
    std::uint64_t prefix = 1;
    for (int d = 0; d < k; ++d) prefix = (prefix << 1) | static_cast<std::uint64_t>(rc.bit(value_node(m, prefix)));
    return prefix & ((std::uint64_t{1} << k) - 1);
}

inline void code_index(RangeEncoder& rc, FloatModels& m, std::uint64_t sym, int width)
{
    std::vector<BitModel>& tree = index_tree(m, width);
    std::uint64_t prefix = 1;
    for (int d = 0; d < width; ++d) {
        const int b = static_cast<int>((sym >> (width - 1 - d)) & 1u);
        rc.bit(tree[prefix], b);
        prefix = (prefix << 1) | static_cast<std::uint64_t>(b);
    }
}

inline std::uint64_t decode_index(RangeDecoder& rc, FloatModels& m, int width)
{
    std::vector<BitModel>& tree = index_tree(m, width);
    std::uint64_t prefix = 1;
    for (int d = 0; d < width; ++d) prefix = (prefix << 1) | static_cast<std::uint64_t>(rc.bit(tree[prefix]));
    return prefix & ((std::uint64_t{1} << width) - 1);
}

// Each value's index among the column's distinct values (in the order they
// first came), and those values: open addressing over the values' bits.
inline void distinct(const std::uint64_t* v, std::size_t n, std::vector<std::uint32_t>& index,
                     std::vector<std::uint64_t>& fresh)
{
    if (n >= 0xFFFFFFFFu) throw std::invalid_argument("float64 column: 2^32 values or more");
    constexpr std::uint32_t kEmpty = 0xFFFFFFFFu;
    std::size_t size = 16;
    while (size < 2 * n) size <<= 1;
    std::vector<std::uint32_t> slots(size, kEmpty);
    const std::size_t mask = size - 1;
    index.resize(n);
    fresh.clear();
    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t h = v[i];                          // MurmurHash3's finaliser
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdull;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ull;
        h ^= h >> 33;
        std::size_t s = static_cast<std::size_t>(h) & mask;
        while (slots[s] != kEmpty && fresh[slots[s]] != v[i]) s = (s + 1) & mask;
        if (slots[s] == kEmpty) {
            slots[s] = static_cast<std::uint32_t>(fresh.size());
            fresh.push_back(v[i]);
        }
        index[i] = slots[s];
    }
}

constexpr int kMinTop = 8, kMaxTop = 28;

// rest[t] = sum_{s<t} log2(s + 1), half[c] = sum_{j<c} log2(j + 1/2), up to
// n: a node's KT code length is rest[n0 + n1] - half[n0] - half[n1]. Kept
// for the thread's next columns up to 2^16 (each sum is the same whatever
// length it is made to: the terms in one order).
struct LogSums {
    std::vector<double> rest{0.0}, half{0.0};
    void reach(std::size_t n)
    {
        const double ln2 = dm::log(2.0);
        for (std::size_t t = rest.size() - 1; t < n; ++t) {
            rest.push_back(rest[t] + dm::log(static_cast<double>(t) + 1.0) / ln2);
            half.push_back(half[t] + dm::log(static_cast<double>(t) + 0.5) / ln2);
        }
    }
};

inline const LogSums& log_sums(std::size_t n, LogSums& own)
{
    thread_local LogSums kept;
    constexpr std::size_t kKept = std::size_t{1} << 16;
    LogSums& s = n <= kKept ? kept : own;
    s.reach(n);
    return s;
}

// The KT code lengths of the tree of the sorted top bits a[lo, hi), which
// share their top d bits: cost[d'] gains each node's at depth d' that holds
// two values or more; a node of one value costs exactly one bit at its depth
// and every depth below it (the KT estimate of one symbol: 1/2), counted in
// single[d'] where it first stands alone.
inline void tree_cost(const std::vector<std::uint64_t>& a, std::size_t lo, std::size_t hi, int d,
                      const std::vector<double>& rest, const std::vector<double>& half, double* cost,
                      std::size_t* single)
{
    while (d < kMaxTop) {
        const std::size_t m = hi - lo;
        if (m == 1) {
            ++single[d];
            return;
        }
        // the first value with bit d set (the values are sorted and share their top d bits)
        const std::uint64_t bit = std::uint64_t{1} << (kMaxTop - 1 - d);
        const std::size_t mid = static_cast<std::size_t>(
            std::partition_point(a.begin() + static_cast<std::ptrdiff_t>(lo), a.begin() + static_cast<std::ptrdiff_t>(hi),
                                 [bit](std::uint64_t x) { return (x & bit) == 0; }) - a.begin());
        cost[d] += rest[m] - half[mid - lo] - half[hi - mid];
        if (mid == lo || mid == hi) {                   // one child: the same values one level down
            ++d;
            continue;
        }
        tree_cost(a, lo, mid, d + 1, rest, half, cost, single);
        lo = mid;
        ++d;
    }
}

// The k of least cost for the values met first (`fresh`, in order): the tree
// of their top k bits at the KT code length of every node's counts, and the
// 64 - k raw bits of each.
inline int choose_k(std::vector<std::uint64_t> fresh)
{
    const std::size_t n = fresh.size();
    if (n == 0) return kMinTop;
    for (auto& a : fresh) a >>= 64 - kMaxTop;
    std::sort(fresh.begin(), fresh.end());
    LogSums own;
    const LogSums& sums = log_sums(n, own);
    double cost[kMaxTop] = {};
    std::size_t single[kMaxTop] = {};
    tree_cost(fresh, 0, n, 0, sums.rest, sums.half, cost, single);
    double tree = 0.0, best_cost = 0.0;
    std::size_t alone = 0;
    int best = kMinTop;
    for (int d = 0; d < kMaxTop; ++d) {
        alone += single[d];                             // the values alone at depth d or above
        tree += cost[d] + static_cast<double>(alone);
        const int k = d + 1;
        const double c = tree + static_cast<double>(64 - k) * static_cast<double>(n);
        if (k == kMinTop || (k > kMinTop && c < best_cost)) {
            best = k;
            best_cost = c;
        }
    }
    return best;
}

// index[i]: v[i]'s index among the distinct values; one met before where it
// is below the count of those met so far
inline std::vector<std::uint8_t> model_bytes(const std::uint64_t* v, const std::uint32_t* index, std::size_t n, int k)
{
    RangeEncoder rc;
    FloatModels m(n);
    std::uint64_t seen = 0;
    int last = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const int repeat = index[i] < seen ? 1 : 0;
        rc.bit(m.flag[last], repeat);
        last = repeat;
        if (repeat) {
            code_index(rc, m, index[i], bit_width(seen - 1));
        } else {
            code_value(rc, m, v[i] >> (64 - k), k);
            rc.direct(v[i], 64 - k);
            ++seen;
        }
    }
    return rc.finish();
}

// x as an integer over 10^e: d, if d / 10^e gives x's bits back
inline bool decimal_at(double x, int e, std::int64_t& d)
{
    const double s = x * kPow10[e];
    if (!(std::fabs(s) < 1125899906842624.0)) return false;    // 2^50; NaN and the infinities too
    d = std::llround(s);
    const double back = static_cast<double>(d) / kPow10[e];
    std::uint64_t a, b;
    std::memcpy(&a, &x, 8);
    std::memcpy(&b, &back, 8);
    return a == b;
}

// The least e for which every value is an integer over 10^e (d), or -1: no
// e below the largest of the values' own least e holds, so the search starts
// there (one value with none ends it).
inline int decimal_digits(const double* x, std::size_t n, std::vector<std::int64_t>& d)
{
    d.resize(n);
    // the e at which a value is an integer d over 10^e are an interval: from
    // its least up to where |x 10^e| reaches 2^50 — below that bound x 10^(e+1)
    // is 10 d to within 1/4, so d / 10^e and 10 d / 10^(e+1), the same number,
    // read back alike. The column's least e is then the largest of the values'
    // least: each value is tried first at the largest so far, then above it.
    int from = 0;
    for (std::size_t i = 0; i < n; ++i) {
        int e = from;
        while (e <= 18 && !decimal_at(x[i], e, d[i])) ++e;
        if (e > 18) return -1;
        from = e;
    }
    for (int e = from; e <= 18; ++e) {
        bool all = true;
        for (std::size_t i = 0; i < n && all; ++i) all = decimal_at(x[i], e, d[i]);
        if (all) return e;
    }
    return -1;
}

}   // namespace col

inline std::vector<std::uint8_t> encode_i64(const std::int64_t* x, std::size_t n);
inline std::vector<std::int64_t> decode_i64(const std::uint8_t* data, std::size_t size, std::size_t n);

// The bytes of a float64 column (its values' bits).
inline std::vector<std::uint8_t> encode_f64(const double* x, std::size_t n)
{
    std::vector<std::uint64_t> v(n);
    if (n) std::memcpy(v.data(), x, n * 8);
    std::vector<std::uint8_t> out;
    if (n) {
        // the values met for the first time, in order: their top bits choose k
        std::vector<std::uint32_t> index;
        std::vector<std::uint64_t> fresh;
        col::distinct(v.data(), n, index, fresh);
        const int k = col::choose_k(std::move(fresh));
        out.push_back(col::kModel);
        col::put_varint(out, n);
        out.push_back(static_cast<std::uint8_t>(k));
        const std::vector<std::uint8_t> coded = col::model_bytes(v.data(), index.data(), n, k);
        out.insert(out.end(), coded.begin(), coded.end());
    }
    std::vector<std::int64_t> d;
    const int e = n ? col::decimal_digits(x, n, d) : -1;
    if (e >= 0) {
        std::vector<std::uint8_t> dec;
        dec.push_back(col::kDecimal);
        col::put_varint(dec, n);
        dec.push_back(static_cast<std::uint8_t>(e));
        const std::vector<std::uint8_t> ints = encode_i64(d.data(), n);
        dec.insert(dec.end(), ints.begin(), ints.end());
        if (dec.size() < out.size()) out.swap(dec);
    }
    std::vector<std::uint8_t> raw;
    raw.push_back(col::kRaw);
    col::put_varint(raw, n);
    if (n && out.size() < raw.size() + 8 * n) return out;
    raw.reserve(raw.size() + 8 * n);
    for (std::size_t i = 0; i < n; ++i) col::put_u64(raw, v[i]);
    return raw;
}

// The column's n values back from its bytes; an error where they do not hold
// n values of a known method.
inline std::vector<double> decode_f64(const std::uint8_t* data, std::size_t size, std::size_t n)
{
    std::size_t pos = 0;
    if (size < 1) throw std::runtime_error("float64 column: no bytes");
    const std::uint8_t method = data[pos++];
    if (col::get_varint(data, size, pos) != n) throw std::runtime_error("float64 column: another count of values");
    std::vector<std::uint64_t> v(n);
    if (method == col::kRaw) {
        for (std::size_t i = 0; i < n; ++i) v[i] = col::get_u64(data, size, pos);
        if (pos != size) throw std::runtime_error("float64 column: bytes past the values");
    } else if (method == col::kModel) {
        if (pos >= size) throw std::runtime_error("float64 column: no k");
        const int k = data[pos++];
        if (k < col::kMinTop || k > col::kMaxTop) throw std::runtime_error("float64 column: k out of range");
        RangeDecoder rc(data + pos, size - pos);
        col::FloatModels m(n);
        std::vector<std::uint64_t> seen;
        int last = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const int repeat = rc.bit(m.flag[last]);
            last = repeat;
            if (repeat) {
                if (seen.empty()) throw std::runtime_error("float64 column: a reference with no value before it");
                const int width = col::bit_width(seen.size() - 1);
                const std::uint64_t j = col::decode_index(rc, m, width);
                if (j >= seen.size()) throw std::runtime_error("float64 column: a reference past the values");
                v[i] = seen[j];
            } else {
                const std::uint64_t top = col::decode_value(rc, m, k);
                v[i] = (top << (64 - k)) | rc.direct(64 - k);
                seen.push_back(v[i]);
            }
        }
        if (rc.consumed() != size - pos)
            throw std::runtime_error("float64 column: the coded bytes do not end where the values do");
    } else if (method == col::kDecimal) {
        if (pos >= size) throw std::runtime_error("float64 column: no e");
        const int e = data[pos++];
        if (e > 18) throw std::runtime_error("float64 column: e out of range");
        const std::vector<std::int64_t> d = decode_i64(data + pos, size - pos, n);
        std::vector<double> x(n);
        for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<double>(d[i]) / col::kPow10[e];
        return x;
    } else {
        throw std::runtime_error("float64 column: unknown method " + std::to_string(method));
    }
    std::vector<double> x(n);
    if (n) std::memcpy(x.data(), v.data(), n * 8);
    return x;
}

namespace col {

inline std::uint64_t zigzag(std::int64_t d)
{
    return (static_cast<std::uint64_t>(d) << 1) ^ static_cast<std::uint64_t>(d >> 63);
}

inline std::int64_t unzigzag(std::uint64_t z)
{
    return static_cast<std::int64_t>((z >> 1) ^ (~(z & 1u) + 1u));
}

// frame of reference: the minimum and every offset in the bits of the largest
inline void put_for(std::vector<std::uint8_t>& out, const std::vector<std::uint64_t>& u)
{
    std::uint64_t lo = u[0], span = 0;
    for (std::uint64_t a : u) lo = a < lo ? a : lo;
    for (std::uint64_t a : u) span = (a - lo) > span ? (a - lo) : span;
    const int w = bit_width(span);
    put_u64(out, lo);
    out.push_back(static_cast<std::uint8_t>(w));
    BitWriter bw(out);
    for (std::uint64_t a : u) bw.put(a - lo, w);
}

inline std::vector<std::uint64_t> get_for(const std::uint8_t* data, std::size_t size, std::size_t& pos, std::size_t n)
{
    const std::uint64_t lo = get_u64(data, size, pos);
    if (pos >= size) throw std::runtime_error("integer column: no width");
    const int w = data[pos++];
    if (w > 64) throw std::runtime_error("integer column: a width past 64 bits");
    BitReader br(data, size, pos);
    std::vector<std::uint64_t> u(n);
    for (std::size_t i = 0; i < n; ++i) u[i] = lo + br.get(w);
    pos = br.end();
    return u;
}

}   // namespace col

// The bytes of an integer column.
inline std::vector<std::uint8_t> encode_i64(const std::int64_t* x, std::size_t n)
{
    std::vector<std::vector<std::uint8_t>> tries;
    auto start = [&](col::Method m) {
        tries.emplace_back();
        tries.back().push_back(m);
        col::put_varint(tries.back(), n);
        return &tries.back();
    };
    {
        auto* out = start(col::kRaw);
        for (std::size_t i = 0; i < n; ++i) col::put_u64(*out, static_cast<std::uint64_t>(x[i]));
    }
    if (n) {
        bool same = true;
        for (std::size_t i = 1; i < n; ++i) same = same && x[i] == x[0];
        if (same) {
            col::put_u64(*start(col::kConstant), static_cast<std::uint64_t>(x[0]));
        } else {
            // offsets from the minimum in int64's order (the sign bit flipped)
            std::vector<std::uint64_t> u(n), d(n - 1);
            for (std::size_t i = 0; i < n; ++i)
                u[i] = static_cast<std::uint64_t>(x[i]) ^ (std::uint64_t{1} << 63);
            col::put_for(*start(col::kFor), u);
            for (std::size_t i = 1; i < n; ++i)
                d[i - 1] = col::zigzag(static_cast<std::int64_t>(static_cast<std::uint64_t>(x[i]) -
                                                                 static_cast<std::uint64_t>(x[i - 1])));
            auto* delta = start(col::kDeltaFor);
            col::put_u64(*delta, static_cast<std::uint64_t>(x[0]));
            col::put_for(*delta, d);
            auto* out = start(col::kRuns);
            std::size_t i = 0;
            while (i < n) {
                std::size_t j = i + 1;
                while (j < n && x[j] == x[i]) ++j;
                col::put_varint(*out, col::zigzag(x[i]));
                col::put_varint(*out, j - i);
                i = j;
            }
        }
    }
    std::size_t best = 0;
    for (std::size_t t = 1; t < tries.size(); ++t)
        if (tries[t].size() < tries[best].size()) best = t;
    return tries[best];
}

inline std::vector<std::int64_t> decode_i64(const std::uint8_t* data, std::size_t size, std::size_t n)
{
    std::size_t pos = 0;
    if (size < 1) throw std::runtime_error("integer column: no bytes");
    const std::uint8_t method = data[pos++];
    if (col::get_varint(data, size, pos) != n) throw std::runtime_error("integer column: another count of values");
    std::vector<std::int64_t> x(n);
    if (method == col::kRaw) {
        for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<std::int64_t>(col::get_u64(data, size, pos));
    } else if (method == col::kConstant) {
        const std::int64_t c = static_cast<std::int64_t>(col::get_u64(data, size, pos));
        for (auto& a : x) a = c;
    } else if (method == col::kFor) {
        const auto u = col::get_for(data, size, pos, n);
        for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<std::int64_t>(u[i] ^ (std::uint64_t{1} << 63));
    } else if (method == col::kDeltaFor) {
        if (n < 2) throw std::runtime_error("integer column: differences of fewer than two values");
        x[0] = static_cast<std::int64_t>(col::get_u64(data, size, pos));
        const auto d = col::get_for(data, size, pos, n - 1);
        for (std::size_t i = 1; i < n; ++i)
            x[i] = static_cast<std::int64_t>(static_cast<std::uint64_t>(x[i - 1]) +
                                             static_cast<std::uint64_t>(col::unzigzag(d[i - 1])));
    } else if (method == col::kRuns) {
        std::size_t i = 0;
        while (i < n) {
            const std::int64_t value = col::unzigzag(col::get_varint(data, size, pos));
            const std::uint64_t run = col::get_varint(data, size, pos);
            if (run == 0 || run > n - i) throw std::runtime_error("integer column: a run past the values");
            for (std::uint64_t r = 0; r < run; ++r) x[i++] = value;
        }
    } else {
        throw std::runtime_error("integer column: unknown method " + std::to_string(method));
    }
    if (pos != size) throw std::runtime_error("integer column: bytes past the values");
    return x;
}

}   // namespace mootation::store
