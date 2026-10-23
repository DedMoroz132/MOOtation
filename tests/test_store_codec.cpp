// SPDX-License-Identifier: Apache-2.0
// The store's column codecs (include/mootation/store/column_codec.hpp; task
// 17, stage 4): every column comes back bit for bit — NaN payloads, -0.0,
// infinities, subnormals, the extremes of int64, an empty column, one value,
// all equal, all different, huge and tiny exponents, repeats — and a column
// with structure is coded shorter than raw. A count or bytes that do not
// match are an error, not other numbers.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <random>
#include <stdexcept>
#include <vector>

#include <mootation/store/column_codec.hpp>
#include <mootation/store/crc32c.hpp>
#include <mootation/store/record.hpp>
#include <mootation/store/record_tables.hpp>
#include <mootation/store/text_codec.hpp>

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
    {
        // decimal columns: clocks in milliseconds, counts as float64 — kept as integers
        std::vector<double> ms, counts, mixed;
        for (int i = 0; i < 1000; ++i) ms.push_back(static_cast<double>(g() % 100000) / 1000.0);
        for (int i = 0; i < 1000; ++i) counts.push_back(static_cast<double>(100 * i + g() % 50));
        mixed = ms;
        mixed[500] = -0.0;                                      // not a decimal: the column is not
        const auto bytes = st::encode_f64(ms.data(), ms.size());
        CHECK(bytes[0] == st::col::kDecimal && bytes[3] == 3);  // method, count (2 bytes), e = 3
        CHECK(f64_round_trip(ms, "milliseconds") < 3 * ms.size());
        const auto cb = st::encode_f64(counts.data(), counts.size());
        CHECK(cb[0] == st::col::kDecimal && cb[3] == 0);
        CHECK(f64_round_trip(counts, "counts") < 2 * counts.size());
        f64_round_trip(mixed, "milliseconds and a -0.0");
        CHECK(st::encode_f64(mixed.data(), mixed.size())[0] != st::col::kDecimal);
        f64_round_trip({0.1, 0.2, 0.30000000000000004, 1e-18, -5e17, 123456789.123}, "decimals at the edges");
        f64_round_trip({1e19, 2e19}, "past int64's range");
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

    // ── text: LZ77 and the range coder ─────────────────────────────────────
    {
        auto round = [&](const std::vector<std::uint8_t>& t, const char* what) {
            const auto bytes = st::encode_text(t.data(), t.size());
            std::vector<std::uint8_t> back;
            try {
                back = st::decode_text(bytes.data(), bytes.size());
            } catch (const std::exception& e) {
                std::printf("FAIL %s: decode threw: %s\n", what, e.what());
                ++failures;
                return bytes.size();
            }
            if (back != t) {
                std::printf("FAIL %s: not the same bytes\n", what);
                ++failures;
            }
            return bytes.size();
        };
        round({}, "empty text");
        round({'x'}, "one byte");
        std::vector<std::uint8_t> all;
        for (int r = 0; r < 4; ++r)
            for (int b = 0; b < 256; ++b) all.push_back(static_cast<std::uint8_t>(b));
        round(all, "every byte value");
        std::vector<std::uint8_t> noise(5000);
        for (auto& b : noise) b = static_cast<std::uint8_t>(g());
        round(noise, "random bytes");
        const std::vector<std::uint8_t> run(10000, 'a');
        CHECK(round(run, "a run (an overlapping copy)") < 60);
        std::string json;
        for (int i = 0; i < 200; ++i)
            json += "{\"gen\": " + std::to_string(i) + ", \"fe\": " + std::to_string(100 * i) +
                    ", \"igdp\": 0." + std::to_string(g() % 1000000) + "},\n";
        const std::vector<std::uint8_t> jt(json.begin(), json.end());
        CHECK(round(jt, "json-like lines") < jt.size() / 2);
        std::vector<std::uint8_t> far;                       // repeats past 64 KB back
        for (int i = 0; i < 70000; ++i) far.push_back(static_cast<std::uint8_t>(g()));
        const std::size_t alone = round(far, "70000 random bytes");
        const std::vector<std::uint8_t> head(far.begin(), far.begin() + 3000);
        far.insert(far.end(), head.begin(), head.end());
        CHECK(round(far, "a repeat 70000 bytes back") < alone + 20);
    }

    // ── a record: SHA-256, zlib's CRC, blocks, body, frame (record.hpp) ──────
    {
        auto hex = [](const std::array<std::uint8_t, 32>& h) {
            std::string s;
            char buf[3];
            for (auto c : h) {
                std::snprintf(buf, sizeof buf, "%02x", c);
                s += buf;
            }
            return s;
        };
        const std::uint8_t* abc = reinterpret_cast<const std::uint8_t*>("abc");
        CHECK(hex(st::sha256(abc, 3)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        CHECK(hex(st::sha256(abc, 0)) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        const std::string long_msg = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";   // 56 bytes: two blocks
        CHECK(hex(st::sha256(reinterpret_cast<const std::uint8_t*>(long_msg.data()), long_msg.size())) ==
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
        CHECK(st::crc32_zlib(reinterpret_cast<const std::uint8_t*>("123456789"), 9) == 0xCBF43926u);
    }
    {
        auto bytes_of = [](const void* p, std::size_t n) {
            const auto* b = static_cast<const std::uint8_t*>(p);
            return std::vector<std::uint8_t>(b, b + n);
        };
        std::vector<double> front(30 * 3), pop(20 * 3);
        for (auto& v : front) v = u01();
        for (std::size_t i = 0; i < pop.size(); ++i) pop[i] = i < 30 ? front[i] : u01();    // shared points
        std::vector<float> f32(500);
        for (std::size_t i = 0; i < f32.size(); ++i) f32[i] = static_cast<float>(i % 7);
        std::vector<std::int32_t> i32 = {-5, 7, -2147483647 - 1, 2147483647, 0};
        std::vector<std::uint32_t> u32 = {4000000000u, 0u, 17u};
        std::vector<std::uint8_t> u8(300, 4);
        const std::string json = R"({"algorithm": "nsga2", "seed": 1})";
        std::vector<st::Block> blocks = {
            {"meta.json", "json", {}, bytes_of(json.data(), json.size()), ""},
            {"archive", "<f8", {30, 3}, bytes_of(front.data(), 8 * front.size()), ""},
            {"f32", "<f4", {500}, bytes_of(f32.data(), 4 * f32.size()), ""},
            {"population", "<f8", {20, 3}, bytes_of(pop.data(), 8 * pop.size()), ""},
            {"i32", "<i4", {5}, bytes_of(i32.data(), 4 * i32.size()), ""},
            {"u32", "<u4", {3}, bytes_of(u32.data(), 4 * u32.size()), ""},
            {"u8", "|u1", {300}, u8, ""},
            {"scalar", "<f8", {}, bytes_of(&front[0], 8), ""},
            {"empty", "<f8", {0, 3}, {}, ""}};
        const std::vector<st::PointTable> tables = {{1, {"f1", "f2", "f3"}}, {3, {"f1", "f2", "f3"}}};
        const auto body = st::encode_body(blocks, tables);
        const auto back = st::decode_body(body.data(), body.size());
        CHECK(back.size() == blocks.size());
        for (std::size_t i = 0; i < back.size() && i < blocks.size(); ++i)
            CHECK(back[i].name == blocks[i].name && back[i].dtype == blocks[i].dtype &&
                  back[i].shape == blocks[i].shape && back[i].data == blocks[i].data && back[i].how.empty());
        CHECK(st::encode_body(back, tables) == body);                        // the same bytes again
        const auto plain = st::encode_body(blocks, {});
        const auto back2 = st::decode_body(plain.data(), plain.size());
        CHECK(back2.size() == blocks.size() && back2[1].data == blocks[1].data);

        const std::string key = "DTLZ2_3D|nsga2||1|10000";
        const auto record = st::encode_record(key, body);
        const auto r = st::parse_record(record.data(), record.size(), 0);
        CHECK(r.ok && r.key == key && r.end == record.size() && r.body_size == body.size() &&
              std::memcmp(record.data() + r.body, body.data(), body.size()) == 0);
        for (std::size_t at = 0; at < record.size(); at += 7) {               // any bit flipped: not whole
            auto bad = record;
            bad[at] ^= static_cast<std::uint8_t>(1u << (at % 8));
            CHECK(!st::parse_record(bad.data(), bad.size(), 0).ok);
        }
        CHECK(!st::parse_record(record.data(), record.size() - 1, 0).ok);     // cut

        // a block of before (zlib): back as stored, its method named
        std::vector<std::uint8_t> old = {1, 0, 0, 0};
        const std::string nm = "a", dt = "json", how = "zlib";
        st::rec::put_le(old, nm.size(), 2);
        old.insert(old.end(), nm.begin(), nm.end());
        st::rec::put_le(old, dt.size(), 1);
        old.insert(old.end(), dt.begin(), dt.end());
        st::rec::put_le(old, 0, 1);
        st::rec::put_le(old, how.size(), 1);
        old.insert(old.end(), how.begin(), how.end());
        st::rec::put_le(old, 3, 8);
        old.insert(old.end(), {7, 8, 9});
        const auto ob = st::decode_body(old.data(), old.size());
        CHECK(ob.size() == 1 && ob[0].how == "zlib" && ob[0].data == std::vector<std::uint8_t>({7, 8, 9}));
        bool threw = false;
        try { st::rec::dtype_of("<x"); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("store codec: every column back bit for bit\n");
    return 0;
}
