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
#include <filesystem>
#include <limits>
#include <string>
#include <random>
#include <stdexcept>
#include <vector>

#include <mootation/store/column_codec.hpp>
#include <mootation/store/crc32c.hpp>
#include <mootation/store/meta_table.hpp>
#include <mootation/store/metrics_table.hpp>
#include <mootation/store/pack.hpp>
#include <mootation/store/record.hpp>
#include <mootation/store/record_tables.hpp>
#include <mootation/store/run_json.hpp>
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

static void same(const std::string& got, const std::string& want)
{
    if (got != want) {
        std::printf("FAIL: got  %s\n      want %s\n", got.c_str(), want.c_str());
        ++failures;
    }
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

        // ── a pack: written, its entries, checked; a byte changed: where ─────
        namespace fs = std::filesystem;
#if defined(_WIN32)
        // a UTF-8 path to the system's UTF-16: two, three and four bytes (a surrogate pair)
        CHECK(st::file::path_of("a\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80").wstring() ==
              std::wstring({L'a', wchar_t(0x00E9), wchar_t(0x20AC), wchar_t(0xD83D), wchar_t(0xDE00)}));
#else
        CHECK(st::file::path_of("a\xC3\xA9").string() == "a\xC3\xA9");
#endif
        const fs::path dir = fs::temp_directory_path() /
                             ("mootation_test_store_" + std::to_string(std::random_device{}()) + "_" +
                              std::to_string(static_cast<unsigned long long>(g() % 1000000007ull)));
        fs::create_directories(dir);
        const std::string path = (dir / "pack-T.mpk").string(), tmp = (dir / "pack-T.mpk.tmp").string();   // ASCII
        std::vector<st::PackRecord> recs;
        for (int i = 0; i < 5; ++i) {
            const std::string k = "P" + std::to_string(i) + "|nsga2||1|100";
            recs.push_back({k, st::encode_record(k, body), i == 3 ? "failed" : "done"});
        }
        const std::vector<std::uint8_t> meta_t = {'M', 'O', 'O', 'T', 'M', 'E', 'T', '1', 0, 0, 0, 0};
        const auto entries = st::write_pack(path, tmp, "kind = pack\nfamily = T\n", meta_t, {}, recs);
        CHECK(!fs::exists(dir / "pack-T.mpk.tmp") && entries.size() == 5);
        std::vector<std::uint8_t> file(static_cast<std::size_t>(fs::file_size(dir / "pack-T.mpk")));
        {
            std::FILE* f = st::file::open(path, "rb");
            CHECK(std::fread(file.data(), 1, file.size(), f) == file.size());
            std::fclose(f);
        }
        CHECK(file.size() % 64 == 0 && std::memcmp(file.data() + file.size() - 64, "MOOTPEND", 8) == 0);
        const auto read = st::pack_entries(file.data(), file.size(), "pack-T.mpk");
        CHECK(read.size() == 5);
        for (std::size_t i = 0; i < read.size() && i < entries.size(); ++i)
            CHECK(read[i].key == entries[i].key && read[i].offset == entries[i].offset && read[i].offset % 64 == 0 &&
                  read[i].crc == entries[i].crc && read[i].status == entries[i].status);
        CHECK(st::check_pack(file.data(), file.size(), "pack-T.mpk").empty());
        auto bad = file;
        bad[static_cast<std::size_t>(entries[2].offset + entries[2].length / 2)] ^= 4;
        const auto said = st::check_pack(bad.data(), bad.size(), "pack-T.mpk");
        CHECK(said.size() == 1 && said[0].find("@" + std::to_string(entries[2].offset) + ": the record of P2") !=
                                      std::string::npos);
        bad = file;
        bad[3] ^= 1;                                                       // the header
        const auto said2 = st::check_pack(bad.data(), bad.size(), "pack-T.mpk");
        CHECK(said2.size() == 1 && said2[0].find("the header") != std::string::npos);
        // a segment: appended, synced, indexed
        const std::string raw = (dir / "seg.raw").string(), idx = (dir / "seg.idx").string();
        const auto e0 = st::append_record(raw, idx, recs[0].key, recs[0].record, "done");
        const auto e1 = st::append_record(raw, idx, recs[1].key, recs[1].record, "done");
        CHECK(e0.offset == 0 && e1.offset == recs[0].record.size() &&
              fs::file_size(dir / "seg.raw") == recs[0].record.size() + recs[1].record.size());
        std::string lines;
        {
            std::FILE* f = st::file::open(idx, "rb");
            char buf[512];
            std::size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) lines.append(buf, n);
            std::fclose(f);
        }
        const auto ie = st::index_entries(lines);
        CHECK(ie.size() == 2 && ie[1].key == recs[1].key && ie[1].offset == e1.offset && ie[1].crc == e1.crc);
        std::error_code ec;
        fs::remove_all(dir, ec);
    }

    // ── a run's JSON: final once and back; a pack's meta table ───────────────
    {
        namespace js = mootation::json;
        auto bytes_of = [](const std::string& t) { return std::vector<std::uint8_t>(t.begin(), t.end()); };
        // the trajectory: two records of columns fe (int), igd (float), ok (bool), x (null), note (text)
        js::Value head = js::Value::object(), cols = js::Value::array(), texts = js::Value::array();
        for (const char* c : {"fe", "igd", "ok", "x", "note"}) cols.items.push_back(js::Value::string(c));
        texts.items.push_back(js::Value::string("first"));
        texts.items.push_back(js::Value::string("last"));
        head.set("kind", js::Value::string("jsonl"));
        head.set("columns", cols);
        head.set("texts", texts);
        const double vals[10] = {100, 0.5, 1, 0, 0, 200, 0.25, 0, 0, 1};
        std::vector<std::uint8_t> vb, cb = {2, 1, 3, 4, 5, 2, 1, 3, 4, 5};
        for (double v : vals) {
            std::uint64_t bits;
            std::memcpy(&bits, &v, 8);
            st::rec::put_le(vb, bits, 8);
        }
        js::Value meta = js::Value::object(), final = js::Value::object();
        meta.set("algorithm", js::Value::string("nsga2"));
        final.set("igd", js::Value::number(0.25));       // as the last record: left to it
        final.set("fe", js::Value::integer(200));
        final.set("hv", js::Value::number(0.75));        // not in the trajectory: kept
        final.set("ok", js::Value::boolean(true));       // the record says false: kept
        meta.set("final", final);
        meta.set("seed", js::Value::integer(3));
        for (bool crlf : {false, true}) {
            const std::vector<std::uint8_t> text = st::meta_text(meta, crlf);
            std::vector<st::Block> run = {
                {"meta.json", "json", {}, text, ""},
                {"trajectory.jsonl#head", "json", {}, bytes_of(js::dump(head)), ""},
                {"trajectory.jsonl", "<f8", {2, 5}, vb, ""},
                {"trajectory.jsonl#cells", "|u1", {2, 5}, cb, ""}};
            const auto once = st::final_once(run);
            const std::string kept(once[0].data.begin(), once[0].data.end());
            CHECK(kept.find("\"#trajectory\": [") != std::string::npos && kept.find("\"hv\": 0.75") != std::string::npos &&
                  st::has_crlf(once[0].data) == crlf);
            const auto js_once = js::parse(kept);
            const js::Value* f = js_once.find("final");
            CHECK(f && f->find("#trajectory")->items.size() == 2 && !f->find("igd") && !f->find("fe") && f->find("ok"));
            const auto back = st::final_back(once);
            CHECK(back[0].data == text);
            const auto body = st::encode_run(run);
            const auto read = st::decode_run(body.data(), body.size());
            CHECK(read.size() == run.size() && read[0].data == text && read[2].data == vb);
        }
        // the meta table: two rows of one skeleton (one with CR LF), one kept whole
        std::vector<std::pair<std::string, std::vector<std::uint8_t>>> rows = {
            {"A|nsga2||1|100", st::meta_text(meta, false)},
            {"B|nsga2||2|100", st::meta_text(meta, true)},
            {"C|nsga2||3|100", bytes_of("{\"a\":1}")}};
        const auto table = st::encode_meta_table(rows);
        const auto got = st::decode_meta_table(table.data(), table.size());
        CHECK(got == rows);
        auto bad = table;
        bad[bad.size() / 2] ^= 1;
        bool threw = false;
        try { st::decode_meta_table(bad.data(), bad.size()); } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    }
    {
        // the metrics table: a run's trajectory by columns (its two records) and one as text (format 1)
        namespace js = mootation::json;
        auto bytes_of = [](const std::string& t) { return std::vector<std::uint8_t>(t.begin(), t.end()); };
        js::Value head = js::Value::object(), cols = js::Value::array();
        for (const char* c : {"fe", "igd", "ok"}) cols.items.push_back(js::Value::string(c));
        head.set("kind", js::Value::string("jsonl"));
        head.set("columns", cols);
        head.set("texts", js::Value::array());
        const double vals[6] = {100, 0.5, 1, 200, 0.25, 0};
        std::vector<std::uint8_t> vb, cb = {2, 1, 3, 2, 1, 3};
        for (double v : vals) {
            std::uint64_t bits;
            std::memcpy(&bits, &v, 8);
            st::rec::put_le(vb, bits, 8);
        }
        const std::vector<st::Block> a = {
            {"meta.json", "json", {}, bytes_of("{\"status\": \"done\"}"), ""},
            {"trajectory.jsonl#head", "json", {}, bytes_of(js::dump(head)), ""},
            {"trajectory.jsonl", "<f8", {2, 3}, vb, ""},
            {"trajectory.jsonl#cells", "|u1", {2, 3}, cb, ""}};
        const std::vector<st::Block> b = {
            {"trajectory.jsonl", "text", {}, bytes_of("{\"fe\": 50, \"hv\": null, \"note\": \"x\"}\r\n\n"), ""}};
        const auto table = st::encode_metrics_table("ZDT", {{"ZDT1|nsga2||1|200", a}, {"ZDT2|moead||2|100", b}});
        CHECK(table.size() > 12 && std::memcmp(table.data(), "MOOTMTR1", 8) == 0);
        std::size_t at = 12;
        const std::uint64_t n = st::col::get_varint(table.data(), table.size(), at);
        const js::Value dir = js::parse(std::string(table.begin() + static_cast<std::ptrdiff_t>(at),
                                                    table.begin() + static_cast<std::ptrdiff_t>(at + n)));
        CHECK(dir.find("rows")->i == 3 && dir.find("family")->s == "ZDT");
        std::string names;
        for (const auto& c : dir.find("columns")->items) names += c.items[0].s + ":" + c.items[1].s + " ";
        same(names, "problem#labels:json problem:<i4 algorithm#labels:json algorithm:<i4 variant#labels:json "
                    "variant:<i4 status#labels:json status:<i4 seed:<f8 budget:<f8 fe:<f8 igd:<f8 ok:<f8 hv:<f8 ");
    }

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("store codec: every column back bit for bit\n");
    return 0;
}
