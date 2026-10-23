#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A binary range coder with adaptive bit probabilities, for the store's own
// codecs (task 17, stage 4; supplement 5, §2: "the top k bits by an adaptive
// binary range coder, the rest raw").
//
// Range coding (Martin 1979) is arithmetic coding over an integer interval
// renormalised a byte at a time: the interval [low, low + range) of 32 bits
// is split for each bit in proportion to the probability of a 0, the coder
// keeps the part of the bit coded, and a byte leaves the top whenever the
// range falls below 2^24. A sum that runs past 2^32 carries into the bytes
// already written: the whole output is in memory here, so the carry is added
// to them directly (a run of 0xFF bytes becomes 0x00 and the byte before it
// grows by one); a decoder reading the bytes sees the sum carried.
//
// THE PROBABILITY of a 0 at a node is 16 bits and learns from every bit coded
// there: p moves towards the bit seen by 1/(n + 2) of the distance, n the
// bits it has seen, which is the Krichevsky-Trofimov estimate (n0 + 1/2) /
// (n + 1) exactly while it is not clamped; from 30 bits on the step stays
// 1/32, so the node keeps following a source that changes. Clamped to
// [kMinP, 2^16 - kMinP], so that no bit costs more than ~11 bits.
//
// DIRECT BITS cost one bit each: m of them (up to 16) split the range into
// 2^m equal parts. The coder codes the low bits of a value that way, which no
// model predicts (supplement 5, §2).
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace mootation::store {

struct BitModel {
    std::uint16_t p = 1u << 15;     // the probability of a 0, in 1/65536
    std::uint8_t n = 0;             // bits seen, up to kMaxCount
};

constexpr std::uint32_t kTop = 1u << 24;
constexpr std::uint32_t kMinP = 32;
constexpr std::uint8_t kMaxCount = 30;

inline void adapt(BitModel& m, int bit)
{
    const std::uint32_t div = static_cast<std::uint32_t>(m.n) + 2;
    std::uint32_t p = m.p;
    if (bit == 0)
        p += (65536u - p) / div;
    else
        p -= p / div;
    if (p < kMinP) p = kMinP;
    if (p > 65536u - kMinP) p = 65536u - kMinP;
    m.p = static_cast<std::uint16_t>(p);
    if (m.n < kMaxCount) ++m.n;
}

class RangeEncoder {
public:
    void bit(BitModel& m, int b)
    {
        const std::uint32_t bound = (range_ >> 16) * m.p;
        if (b == 0) {
            range_ = bound;
        } else {
            low_ += bound;
            range_ -= bound;
        }
        adapt(m, b);
        normalise();
    }

    // n bits of v (the low ones, the highest first), at one bit each: up to
    // 16 at a time, the range split into 2^m equal parts
    void direct(std::uint64_t v, int n)
    {
        while (n > 0) {
            const int m = n < 16 ? n : 16;
            n -= m;
            const std::uint32_t part = static_cast<std::uint32_t>((v >> n) & ((1u << m) - 1u));
            range_ >>= m;
            low_ += static_cast<std::uint64_t>(part) * range_;
            normalise();
        }
    }

    // the bytes: what was coded and the four bytes of low that end it
    std::vector<std::uint8_t> finish()
    {
        for (int i = 0; i < 4; ++i) {
            out_.push_back(static_cast<std::uint8_t>(low_ >> 24));
            low_ = (low_ << 8) & 0xFFFFFFFFu;
        }
        return std::move(out_);
    }

private:
    void normalise()
    {
        if (low_ >> 32) {                       // the carry, into the bytes written
            std::size_t i = out_.size();
            while (i > 0 && out_[i - 1] == 0xFF) out_[--i] = 0x00;
            if (i == 0) throw std::logic_error("range coder: a carry past the first byte");
            ++out_[i - 1];
            low_ &= 0xFFFFFFFFu;
        }
        while (range_ < kTop) {
            out_.push_back(static_cast<std::uint8_t>(low_ >> 24));
            low_ = (low_ << 8) & 0xFFFFFFFFu;
            range_ <<= 8;
        }
    }

    std::uint64_t low_ = 0;
    std::uint32_t range_ = 0xFFFFFFFFu;
    std::vector<std::uint8_t> out_;
};

class RangeDecoder {
public:
    RangeDecoder(const std::uint8_t* data, std::size_t size) : data_(data), size_(size)
    {
        for (int i = 0; i < 4; ++i) code_ = (code_ << 8) | next();
    }

    int bit(BitModel& m)
    {
        const std::uint32_t bound = (range_ >> 16) * m.p;
        int b;
        if (code_ < bound) {
            range_ = bound;
            b = 0;
        } else {
            code_ -= bound;
            range_ -= bound;
            b = 1;
        }
        adapt(m, b);
        normalise();
        return b;
    }

    std::uint64_t direct(int n)
    {
        std::uint64_t v = 0;
        while (n > 0) {
            const int m = n < 16 ? n : 16;
            n -= m;
            range_ >>= m;
            std::uint32_t part = code_ / range_;
            const std::uint32_t top = (1u << m) - 1u;
            if (part > top) part = top;                 // damaged bytes: a value, not a crash
            code_ -= part * range_;
            v = (v << m) | part;
            normalise();
        }
        return v;
    }

    // the bytes read: the encoder's, all of them and no more, at the last value
    std::size_t consumed() const { return pos_; }

private:
    std::uint32_t next()
    {
        const std::uint32_t b = pos_ < size_ ? data_[pos_] : 0u;
        ++pos_;
        return b;
    }

    void normalise()
    {
        while (range_ < kTop) {
            code_ = (code_ << 8) | next();
            range_ <<= 8;
        }
    }

    const std::uint8_t* data_;
    std::size_t size_, pos_ = 0;
    std::uint32_t code_ = 0, range_ = 0xFFFFFFFFu;
};

}   // namespace mootation::store
