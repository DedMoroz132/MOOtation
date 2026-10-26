#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A job's record in the store, made in C++ — task 17, stage 5, step 4d: what
// run/campaign.py's run_job hands mootation/store.py for a run of a campaign
// with storage = "store": the "#format" block, meta.json as the file's text,
// the trajectory column by column (store._jsonl_blocks: the keys of every
// record in one order, float64 values, a code per cell, the texts apart —
// kept as text where the lines would not come back byte for byte), and the
// run's tables (store.table_blocks); encoded by the library's codecs
// (run_json.hpp, record.hpp) and appended to this process's segment
// (pack.hpp: synced, then indexed), a new segment past its limit
// (store.SegmentWriter).
//
// The files a Python process writes in text mode end their lines with the
// platform's line end (meta.json's and the trajectory's on Windows: "\r\n");
// so do these.
// ============================================================================

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../json.hpp"
#include "../store/pack.hpp"
#include "../store/record.hpp"
#include "../store/run_json.hpp"
#include "job_run.hpp"

namespace mootation::campaign {

using store::Block;

// store.FORMAT_VERSION
constexpr int kStoreFormatVersion = 3;

#if defined(_WIN32)
constexpr bool kTextCrlf = true;
#else
constexpr bool kTextCrlf = false;
#endif

inline std::vector<std::uint8_t> bytes_of(const std::string& s) { return std::vector<std::uint8_t>(s.begin(), s.end()); }

// text written in Python's text mode: "\n" as the platform's line end
inline std::string text_mode(const std::string& s, bool crlf = kTextCrlf)
{
    if (!crlf) return s;
    std::string out;
    out.reserve(s.size() + s.size() / 16);
    for (char c : s) {
        if (c == '\n') out += '\r';
        out += c;
    }
    return out;
}

// store.format_block
inline Block format_block()
{
    return Block{"#format", "json", {}, bytes_of("{\"version\": " + std::to_string(kStoreFormatVersion) + "}"), ""};
}

// store.table_blocks: the head with numbers "float64", and the numbers
inline std::vector<Block> table_blocks(const Table& t)
{
    json::Value head = t.head;
    head.set("numbers", json::Value::string("float64"));
    std::vector<Block> out;
    out.push_back(Block{t.name + "#head", "json", {}, bytes_of(json::dump(head)), ""});
    Block b{t.name, "<f8", {t.cols ? t.rows : 0, t.cols}, {}, ""};
    const auto* p = reinterpret_cast<const std::uint8_t*>(t.values.data());
    b.data.assign(p, p + t.values.size() * sizeof(double));
    out.push_back(std::move(b));
    return out;
}

// store._jsonl_blocks: a trajectory (its lines without their line ends) column
// by column, or nothing where the lines would not come back byte for byte
inline std::optional<std::vector<Block>> trajectory_blocks(const std::string& name, const std::vector<std::string>& lines,
                                                           bool crlf = kTextCrlf)
{
    enum : std::uint8_t { kAbsent = 0, kFloat = 1, kInt = 2, kBool = 3, kNull = 4, kText = 5 };
    using Kind = json::Value::Kind;
    std::vector<json::Value> recs;
    for (const auto& line : lines) {
        try {
            recs.push_back(json::parse(line));
        } catch (const std::exception&) {
            return std::nullopt;
        }
        if (recs.back().kind != Kind::Object) return std::nullopt;
    }
    // one order of the columns that keeps every record's own order
    std::vector<std::string> cols;
    for (const auto& r : recs) {
        std::ptrdiff_t prev = -1;
        for (const auto& k : r.keys) {
            auto it = std::find(cols.begin(), cols.end(), k);
            if (it == cols.end()) it = cols.insert(cols.begin() + (prev + 1), k);
            prev = it - cols.begin();
        }
    }
    const std::size_t n = recs.size(), c = cols.size();
    std::vector<double> V(n * c, 0.0);
    std::vector<std::uint8_t> C(n * c, kAbsent);
    std::vector<std::string> texts;
    for (std::size_t i = 0; i < n; ++i) {
        const auto& r = recs[i];
        for (std::size_t k = 0; k < r.keys.size(); ++k) {
            const std::size_t j = static_cast<std::size_t>(std::find(cols.begin(), cols.end(), r.keys[k]) - cols.begin());
            const json::Value& v = r.items[k];
            switch (v.kind) {
            case Kind::Null: C[i * c + j] = kNull; break;
            case Kind::Bool:
                C[i * c + j] = kBool;
                V[i * c + j] = v.b ? 1.0 : 0.0;
                break;
            case Kind::Int:
                if (v.i >= (std::int64_t{1} << 53) || v.i <= -(std::int64_t{1} << 53)) return std::nullopt;
                C[i * c + j] = kInt;
                V[i * c + j] = static_cast<double>(v.i);
                break;
            case Kind::Float:
                C[i * c + j] = kFloat;
                V[i * c + j] = v.f;
                break;
            case Kind::String:
                if (v.datetime) return std::nullopt;
                C[i * c + j] = kText;
                V[i * c + j] = static_cast<double>(texts.size());
                texts.push_back(v.s);
                break;
            default: return std::nullopt;
            }
        }
    }
    // the lines back from the columns (store._jsonl_text), as a check
    std::string want, got;
    for (const auto& line : lines) want += line + "\n";
    for (std::size_t i = 0; i < n; ++i) {
        json::Value r = json::Value::object();
        for (std::size_t j = 0; j < c; ++j) {
            const std::uint8_t code = C[i * c + j];
            const double v = V[i * c + j];
            if (code == kAbsent) continue;
            r.set(cols[j], code == kFloat  ? json::Value::number(v)
                           : code == kInt  ? json::Value::integer(static_cast<std::int64_t>(v))
                           : code == kBool ? json::Value::boolean(v != 0.0)
                           : code == kNull ? json::Value::null()
                                           : json::Value::string(texts[static_cast<std::size_t>(v)]));
        }
        got += json::dump(r, -1, ",", ":") + "\n";
    }
    if (got != want) return std::nullopt;
    json::Value head = json::Value::object(), jc = json::Value::array(), jt = json::Value::array();
    for (const auto& k : cols) jc.items.push_back(json::Value::string(k));
    for (const auto& s : texts) jt.items.push_back(json::Value::string(s));
    head.set("kind", json::Value::string("jsonl"));
    head.set("columns", jc);
    head.set("texts", jt);
    // an empty trajectory has no line end to tell: "\n"
    head.set("newline", json::Value::string(crlf && n ? "\r\n" : "\n"));
    std::vector<Block> out;
    out.push_back(Block{name + "#head", "json", {}, bytes_of(json::dump(head)), ""});
    out.push_back(Block{name + "#cells", "|u1", {n, c}, C, ""});
    Block b{name, "<f8", {n, c}, {}, ""};
    const auto* p = reinterpret_cast<const std::uint8_t*>(V.data());
    b.data.assign(p, p + V.size() * sizeof(double));
    out.push_back(std::move(b));
    return out;
}

// The blocks of a job's record, in run_job's order: "#format", the run
// folder's files by name (meta.json, trajectory.jsonl), then the tables.
inline std::vector<Block> record_blocks(const JobFiles& files, const std::vector<std::string>& trajectory,
                                        bool crlf = kTextCrlf)
{
    std::vector<Block> blocks{format_block()};
    blocks.push_back(Block{"meta.json", "json", {}, bytes_of(text_mode(json::dump(files.meta, 1), crlf)), ""});
    if (auto t = trajectory_blocks("trajectory.jsonl", trajectory, crlf)) {
        for (auto& b : *t) blocks.push_back(std::move(b));
    } else {
        std::string text;
        for (const auto& line : trajectory) text += line + "\n";
        blocks.push_back(Block{"trajectory.jsonl", "text", {}, bytes_of(text_mode(text, crlf)), ""});
    }
    for (const auto& t : files.tables)
        for (auto& b : table_blocks(t)) blocks.push_back(std::move(b));
    return blocks;
}

// store.run_key / job_key
inline std::string job_key(const Job& job)
{
    return job.problem + "|" + job.algorithm + "|" + job.label + "|" + std::to_string(job.seed) + "|" +
           std::to_string(job.budget());
}

// store.SegmentWriter: one process's segments in one store
class SegmentWriter {
public:
    // header: the "key = value" lines a segment's .mhd adds (run_job's:
    // code_stamp, trajectory_version, detmath_path), in order
    SegmentWriter(std::filesystem::path store, std::uint64_t limit, long long pid, std::string host,
                  std::vector<std::pair<std::string, std::string>> header)
        : store_(std::move(store)), limit_(limit), pid_(pid), host_(std::move(host)), header_(std::move(header))
    {
        const std::time_t now = std::time(nullptr);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &now);
#else
        localtime_r(&now, &tm);
#endif
        char buf[32];
        std::strftime(buf, sizeof buf, "%Y%m%d-%H%M%S", &tm);
        stamp_ = buf;
        std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%S", &tm);
        created_ = buf;
    }

    store::Entry append(const std::string& key, const std::vector<Block>& blocks)
    {
        const auto body = store::encode_run(blocks);
        const auto rec = store::encode_record(key, body);
        std::error_code ec;
        if (raw_.empty() || (records_ && std::filesystem::file_size(raw_, ec) + rec.size() > limit_)) start();
        std::string status = "?";
        for (const auto& b : blocks)
            if (b.name == "meta.json") {
                const json::Value meta = json::parse(std::string(b.data.begin(), b.data.end()));
                const json::Value* s = meta.find("status");
                status = s ? (s->kind == json::Value::Kind::String ? s->s : json::dump(*s)) : "?";
            }
        auto e = store::append_record(utf8_of(raw_), idx_, key, rec, status);
        ++records_;
        return e;
    }

private:
    static std::string utf8_of(const std::filesystem::path& p)
    {
        const auto s = p.u8string();
        return std::string(s.begin(), s.end());
    }

    void start()
    {
        std::filesystem::create_directories(store_);
        for (;;) {            // a name no writer of this process took this second
            ++number_;
            char n[16];
            std::snprintf(n, sizeof n, "%04d", number_);
            const std::filesystem::path raw =
                store_ / ("seg-" + stamp_ + "-p" + std::to_string(pid_) + "-" + n + ".raw");
            std::error_code ec;
            if (std::filesystem::exists(raw, ec)) continue;
            std::FILE* f = store::file::open(utf8_of(raw), "wb");
            std::fclose(f);
            raw_ = raw;
            break;
        }
        std::filesystem::path idx = raw_;
        idx.replace_extension(".idx");
        idx_ = utf8_of(idx);
        std::fclose(store::file::open(idx_, "wb"));
        records_ = 0;
        // the segment's header (store._write_header: "key = value" lines)
        std::filesystem::path mhd = raw_;
        mhd.replace_extension(".mhd");
        std::string text = "format = mootation-store\nformat_version = " + std::to_string(kStoreFormatVersion) +
                           "\nkind = segment\ndata_file = " + utf8_of(raw_.filename()) +
                           "\nindex_file = " + utf8_of(idx.filename()) + "\nprocess = " + std::to_string(pid_) +
                           "\nhost = " + host_ + "\nsegment = " + std::to_string(number_) + "\n";
        for (const auto& [k, v] : header_) text += k + " = " + v + "\n";
        text += "created = " + created_ + "\n";
        const std::string tmp = utf8_of(mhd) + "." + std::to_string(pid_) + ".tmp";
        std::FILE* h = store::file::open(tmp, "wb");
        store::file::write(h, text.data(), text.size(), tmp);
        std::fclose(h);
        std::filesystem::rename(store::file::path_of(tmp), mhd);
    }

    std::filesystem::path store_;
    std::uint64_t limit_;
    long long pid_;
    std::string host_;
    std::vector<std::pair<std::string, std::string>> header_;
    std::string stamp_, created_;
    int number_ = 0;
    std::filesystem::path raw_;
    std::string idx_;
    std::size_t records_ = 0;
};

}   // namespace mootation::campaign
