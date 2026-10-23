// SPDX-License-Identifier: Apache-2.0
// The store's column codecs (include/mootation/store/column_codec.hpp; task
// 17, stage 4): every column comes back bit for bit — NaN payloads, -0.0,
// infinities, subnormals, the extremes of int64, an empty column, one value,
// all equal, all different, huge and tiny exponents, repeats — and a column
// with structure is coded shorter than raw. A count or bytes that do not
// match are an error, not other numbers.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include <mootation/store/column_codec.hpp>
#include <mootation/store/crc32c.hpp>
#include <mootation/store/record_tables.hpp>

namespace st = mootation::store;

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

static double from_bits(std::uint64_t b)
{
    double x;
    std::memcpy(&x, &b, 8);
    return x;
}

static bool same_bits(const std::vector<double>& a, const std::vector<double>& b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * 8) == 0);
}

static std::size_t f64_round_trip(const std::vector<double>& x, const char* what)
{
    const auto bytes = st::encode_f64(x.data(), x.size());
    std::vector<double> back;
    try {
        back = st::decode_f64(bytes.data(), bytes.size(), x.size());
    } catch (const std::exception& e) {
        std::printf("FAIL %s: decode threw: %s\n", what, e.what());
        ++failures;
        return bytes.size();
    }
    if (!same_bits(x, back)) {
        std::printf("FAIL %s: not the same bits\n", what);
        ++failures;
    }
    return bytes.size();
}

static std::size_t i64_round_trip(const std::vector<std::int64_t>& x, const char* what)
{
    const auto bytes = st::encode_i64(x.data(), x.size());
    std::vector<std::int64_t> back;
    try {
        back = st::decode_i64(bytes.data(), bytes.size(), x.size());
    } catch (const std::exception& e) {
        std::printf("FAIL %s: decode threw: %s\n", what, e.what());
        ++failures;
        return bytes.size();
    }
    if (back != x) {
        std::printf("FAIL %s: not the same values\n", what);
        ++failures;
    }
    return bytes.size();
}

int main()
{
    std::mt19937_64 g(17);
    auto u01 = [&] { return static_cast<double>(g() >> 11) * (1.0 / 9007199254740992.0); };

    // ── float64 ────────────────────────────────────────────────────────────
    f64_round_trip({}, "empty");
    f64_round_trip({0.25}, "one value");
    f64_round_trip(std::vector<double>(500, 3.0), "all equal");
    {
        std::vector<double> x(2000);
        for (auto& v : x) v = from_bits(g());                   // every bit pattern, NaNs included
        f64_round_trip(x, "random bit patterns");
    }
    {
        const double inf = std::numeric_limits<double>::infinity();
        std::vector<double> x = {0.0, -0.0, inf, -inf, from_bits(0x7FF8000000000001ull),
                                 from_bits(0xFFF0000000000ABCull), from_bits(0x7FF4000000000000ull),
                                 std::numeric_limits<double>::denorm_min(), -std::numeric_limits<double>::denorm_min(),
                                 from_bits(0x000FFFFFFFFFFFFFull), std::numeric_limits<double>::max(),
                                 -std::numeric_limits<double>::max(), std::numeric_limits<double>::min(), 1e-300,
                                 1e300, -1e-308, 0.0, -0.0, inf};
        for (int r = 0; r < 3; ++r) {                            // and their repeats
            const std::vector<double> copy = x;
            x.insert(x.end(), copy.begin(), copy.end());
        }
        f64_round_trip(x, "special values");
    }
    {
        std::vector<double> x(3000);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::ldexp(u01(), static_cast<int>(g() % 2000) - 1000);
        f64_round_trip(x, "huge and tiny exponents");
    }
    {
        // a front's values: one smooth column and its repeats, as a record's
        std::vector<double> x;
        for (int i = 0; i < 1000; ++i) x.push_back(std::sqrt(u01()));
        for (int i = 0; i < 400; ++i) x.push_back(x[g() % 1000]);
        for (int i = 0; i < 200; ++i) x.push_back(i % 2 ? 0.0 : 1.0);      // variables on their bounds
        const std::size_t size = f64_round_trip(x, "a front with repeats");
        CHECK(size < 8 * x.size());
    }

    // a count or bytes that do not match: an error
    {
        std::vector<double> x(100);
        for (auto& v : x) v = u01();
        const auto bytes = st::encode_f64(x.data(), x.size());
        bool threw = false;
        try { st::decode_f64(bytes.data(), bytes.size(), 99); } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
        threw = false;
        auto cut = bytes;
        cut.resize(cut.size() / 2);
        try { st::decode_f64(cut.data(), cut.size(), 100); } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    }

    // ── integers ───────────────────────────────────────────────────────────
    i64_round_trip({}, "empty ints");
    i64_round_trip({42}, "one int");
    i64_round_trip(std::vector<std::int64_t>(300, -7), "constant");
    {
        std::vector<std::int64_t> x(500);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = 100 + 3 * static_cast<std::int64_t>(i);
        const std::size_t size = i64_round_trip(x, "a counter");
        CHECK(size < 100);                                       // differences of 3: a few bits each
    }
    {
        std::vector<std::int64_t> x(1000);
        for (auto& v : x) v = static_cast<std::int64_t>(g());
        i64_round_trip(x, "random ints");
    }
    {
        const std::int64_t lo = std::numeric_limits<std::int64_t>::min(), hi = std::numeric_limits<std::int64_t>::max();
        i64_round_trip({lo, hi, 0, -1, 1, lo, lo, hi}, "the extremes");
        i64_round_trip({hi, lo}, "the widest difference");
    }
    {
        std::vector<std::int64_t> x;
        for (int r = 0; r < 20; ++r) x.insert(x.end(), 50 + static_cast<std::size_t>(g() % 50), static_cast<std::int64_t>(r % 3));
        const std::size_t size = i64_round_trip(x, "runs");
        CHECK(size < 80);
    }

    // ── CRC-32C: the check value, and in parts as at once ──────────────────
    {
        const char* nine = "123456789";
        const auto* p = reinterpret_cast<const std::uint8_t*>(nine);
        CHECK(st::crc32c(p, 9) == 0xE3069283u);
        CHECK(st::crc32c(p + 4, 5, st::crc32c(p, 4)) == 0xE3069283u);
        CHECK(st::crc32c(p, 0) == 0u);
    }

    // ── a record's tables: back with their names, columns and bits ─────────
    {
        std::vector<st::Table> tables(3);
        tables[0].name = "archive";
        tables[0].columns = {"f1", "f2", "extreme"};
        tables[1].name = "final";
        tables[1].columns = {"f1", "f2", "x1"};
        tables[2].name = "empty";
        tables[2].columns = {"f1", "f2"};
        tables[0].rows = 200;
        tables[1].rows = 50;
        for (std::size_t i = 0; i < 200; ++i) {
            const double a = u01();
            tables[0].values.insert(tables[0].values.end(), {a, 1.0 - std::sqrt(a), i < 2 ? 1.0 : 0.0});
        }
        for (std::size_t i = 0; i < 50; ++i)                   // half of them archive points
            tables[1].values.insert(tables[1].values.end(),
                                    {i % 2 ? tables[0].values[3 * i] : u01(),
                                     i % 2 ? tables[0].values[3 * i + 1] : u01(), u01()});
        const auto bytes = st::encode_tables(tables);
        const auto back = st::decode_tables(bytes.data(), bytes.size());
        CHECK(back.size() == 3);
        for (std::size_t t = 0; t < back.size() && t < tables.size(); ++t) {
            CHECK(back[t].name == tables[t].name && back[t].columns == tables[t].columns &&
                  back[t].rows == tables[t].rows);
            CHECK(same_bits(back[t].values, tables[t].values));
        }
    }

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("store codec: every column back bit for bit\n");
    return 0;
}
