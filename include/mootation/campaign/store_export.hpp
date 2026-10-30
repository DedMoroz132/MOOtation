#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// `mootation store export` in C++ — task 17, §2.3: mootation/store.py's
// export_files and export_csv, for a user of the library without Python.
//
//   files  every stored run (or those of the keys asked for) as the campaign
//          wrote it, under <out>/<problem>/<variant or algorithm>/run_<seed>:
//          meta.json and the other JSON and text files as they are kept; the
//          trajectory's lines back from its columns (store._jsonl_text); a
//          table's CSV back from its numbers (store._csv_text: "{:.10g}", an
//          int where the head says so); an .npz from its arrays; a name
//          ending in .gz compressed again (gzip without a time stamp)
//   csv    every pack's metrics table as metrics-<family>.csv.gz: the key's
//          problem, algorithm, variant and the run's status as text, the
//          numbers as Python's repr; a field with a comma, quote or line end
//          quoted (csv.writer's QUOTE_MINIMAL)
//
// The text is Python's byte for byte; the gzip and zip around it are the
// library's own deflate (io/deflate.hpp), so the .gz and .npz files hold the
// same content in other bytes.
// ============================================================================

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../io/deflate.hpp"
#include "../io/npz.hpp"
#include "../json.hpp"
#include "../store/column_codec.hpp"
#include "../store/meta_table.hpp"
#include "../store/pack.hpp"
#include "../store/record.hpp"
#include "store_tools.hpp"

namespace mootation::campaign {

namespace export_detail {

inline std::string with_newline(std::string text, const json::Value& head)
{
    const json::Value* nl = head.find("newline");
    if (!nl || nl->s != "\r\n") return text;
    std::string out;
    for (char c : text) {
        if (c == '\n') out += '\r';
        out += c;
    }
    return out;
}

inline double cell(const store::Block& b, std::size_t k)
{
    double v;
    std::memcpy(&v, b.data.data() + 8 * k, 8);
    return v;
}

// store._jsonl_text
inline std::string jsonl_text(const json::Value& head, const store::Block& C, const store::Block& V)
{
    enum : std::uint8_t { kAbsent = 0, kFloat = 1, kInt = 2, kBool = 3, kNull = 4, kText = 5 };
    const json::Value& cols = *head.find("columns");
    const json::Value& texts = *head.find("texts");
    const std::size_t n = V.shape.empty() ? 0 : static_cast<std::size_t>(V.shape[0]);
    const std::size_t c = cols.items.size();
    std::string out;
    for (std::size_t i = 0; i < n; ++i) {
        json::Value r = json::Value::object();
        for (std::size_t j = 0; j < c; ++j) {
            const std::uint8_t code = C.data[i * c + j];
            const double v = cell(V, i * c + j);
            if (code == kAbsent) continue;
            r.set(cols.items[j].s, code == kFloat  ? json::Value::number(v)
                                   : code == kInt  ? json::Value::integer(static_cast<std::int64_t>(v))
                                   : code == kBool ? json::Value::boolean(v != 0.0)
                                   : code == kNull ? json::Value::null()
                                                   : texts.items[static_cast<std::size_t>(v)]);
        }
        out += json::dump(r, -1, ",", ":") + "\n";
    }
    return with_newline(out, head);
}

// Python's f"{v:.10g}"
inline std::string g10(double v)
{
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.10g", v);
    return buf;
}

// store._csv_text
inline std::string csv_text(const json::Value& head, const store::Block& A)
{
    std::string out;
    if (const json::Value* c = head.find("comments"))
        for (const auto& line : c->items) out += line.s + "\n";
    const json::Value& cols = *head.find("columns");
    std::set<std::int64_t> ints;
    if (const json::Value* k = head.find("ints"))
        for (const auto& v : k->items) ints.insert(v.i);
    for (std::size_t j = 0; j < cols.items.size(); ++j) out += (j ? "," : "") + cols.items[j].s;
    out += "\n";
    const std::size_t m = cols.items.size(), n = m ? A.data.size() / 8 / m : 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            const double v = cell(A, i * m + j);
            if (j) out += ",";
            out += ints.count(static_cast<std::int64_t>(j)) ? std::to_string(static_cast<long long>(v)) : g10(v);
        }
        out += "\n";
    }
    return with_newline(out, head);
}

// Python's repr(float(v))
inline std::string py_repr(double v)
{
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    return json::float_repr(v);
}

// a field as csv.writer writes it (QUOTE_MINIMAL)
inline std::string csv_field(const std::string& s)
{
    if (s.find_first_of(",\"\r\n") == std::string::npos) return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += '"';
        out += c;
    }
    return out + "\"";
}

inline void write_bytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("cannot write " + detail::utf8(path));
}

inline bool ends_with(const std::string& s, const std::string& tail)
{
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

}   // namespace export_detail

// store.run_files + write_run_files: a run's files, (name, bytes as written)
inline std::vector<std::pair<std::string, std::vector<std::uint8_t>>> run_files(const std::vector<store::Block>& blocks)
{
    using namespace export_detail;
    std::map<std::string, const store::Block*> by;
    for (const auto& b : blocks) by[b.name] = &b;
    std::vector<std::string> names, npzs;
    for (const auto& b : blocks) {
        if (b.name.find('#') == std::string::npos && b.name.find('/') == std::string::npos) names.push_back(b.name);
        if (ends_with(b.name, ".npz#head")) npzs.push_back(b.name.substr(0, b.name.size() - 5));
    }
    std::vector<std::pair<std::string, std::vector<std::uint8_t>>> out;
    for (const auto& name : names) {
        const store::Block& b = *by.at(name);
        std::vector<std::uint8_t> content;
        if (b.dtype == "json" || b.dtype == "text" || b.dtype == "bytes") {
            content = b.data;
        } else {
            const json::Value head = json::parse(std::string(by.at(name + "#head")->data.begin(), by.at(name + "#head")->data.end()));
            const json::Value* kind = head.find("kind");
            content = bytes_of(kind && kind->s == "jsonl" ? jsonl_text(head, *by.at(name + "#cells"), b) : csv_text(head, b));
        }
        if (ends_with(name, ".gz") && b.dtype != "bytes") content = deflate::gzip(content.data(), content.size());
        out.emplace_back(name, std::move(content));
    }
    for (const auto& name : npzs) {
        const json::Value head = json::parse(std::string(by.at(name + "#head")->data.begin(), by.at(name + "#head")->data.end()));
        std::vector<npz::NpyArray> arrays;
        for (const auto& k : head.find("arrays")->items) {
            const store::Block& a = *by.at(name + "/" + k.s);
            npz::NpyArray arr{k.s, a.dtype, {}, a.data};
            for (auto s : a.shape) arr.shape.push_back(static_cast<std::size_t>(s));
            arrays.push_back(std::move(arr));
        }
        out.emplace_back(name, npz::npz_bytes(arrays));
    }
    return out;
}

// store.rel_dir_of: problem/(variant or algorithm)/run_<seed>
inline std::filesystem::path run_rel_dir(const std::string& key)
{
    const auto p = KeyOrder::parts(key);
    return store::file::path_of(p.at(0)) / store::file::path_of(p.at(2).empty() ? p.at(1) : p.at(2)) /
           ("run_" + p.at(3));
}

// store.export_files: the files of the stored runs (all, or `keys`) under out; how many
inline std::size_t export_files(const std::filesystem::path& store_dir, const std::filesystem::path& out,
                                const std::vector<std::string>& keys = {})
{
    const auto idx = store_index(store_dir);
    std::vector<std::string> order;
    for (const auto& [key, e] : idx)
        if (keys.empty() || std::find(keys.begin(), keys.end(), key) != keys.end()) order.push_back(key);
    std::sort(order.begin(), order.end(), KeyOrder{});
    MetaCache cache;
    for (const auto& key : order)
        for (const auto& [name, bytes] : run_files(entry_blocks(idx.at(key), cache)))
            export_detail::write_bytes(out / run_rel_dir(key) / store::file::path_of(name), bytes);
    return order.size();
}

// store.export_csv: metrics-<family>.csv.gz of every pack's metrics table
inline std::vector<std::filesystem::path> export_csv(const std::filesystem::path& store_dir, const std::filesystem::path& out)
{
    using namespace export_detail;
    std::vector<std::filesystem::path> packs, segs, written;
    store_files(store_dir, packs, segs);
    std::sort(packs.begin(), packs.end());
    for (const auto& pack : packs) {
        if (pack.extension() != ".mpk") continue;
        const std::string data = detail::read_all(pack);
        const auto* buf = reinterpret_cast<const std::uint8_t*>(data.data());
        const store::PackFooter f = store::pack_footer(buf, data.size(), detail::utf8(pack.filename()));
        if (f.metrics_size < 12) continue;
        const std::uint8_t* sec = buf + f.metrics_at;
        if (std::memcmp(sec, store::kMetricsMagic, 8) != 0) throw std::runtime_error(detail::utf8(pack) + ": not a metrics table");
        const std::uint8_t* body = sec + 12;
        const std::size_t n_body = static_cast<std::size_t>(f.metrics_size) - 12;
        if (store::crc32c(body, n_body) != static_cast<std::uint32_t>(store::rec::get_le(sec + 8, 4)))
            throw std::runtime_error(detail::utf8(pack) + ": the metrics table's checksum does not hold");
        std::vector<std::pair<const std::uint8_t*, std::size_t>> pieces;
        for (std::size_t at = 0; at < n_body;) {
            const std::uint64_t k = store::col::get_varint(body, n_body, at);
            pieces.emplace_back(body + at, static_cast<std::size_t>(k));
            at += static_cast<std::size_t>(k);
        }
        const json::Value dir = json::parse(std::string(pieces.at(0).first, pieces.at(0).first + pieces.at(0).second));
        std::map<std::string, store::Block> cols;
        std::vector<std::string> order;
        for (std::size_t k = 0; k < dir.find("columns")->items.size(); ++k) {
            std::vector<std::uint8_t> one{1, 0, 0, 0};
            one.insert(one.end(), pieces.at(k + 1).first, pieces.at(k + 1).first + pieces.at(k + 1).second);
            auto blocks = store::decode_body(one.data(), one.size());
            const std::string name = dir.find("columns")->items[k].items.at(0).s;
            cols[name] = std::move(blocks.at(0));
            order.push_back(name);
        }
        const std::size_t rows = static_cast<std::size_t>(dir.find("rows")->i);
        static const char* const key_columns[4] = {"problem", "algorithm", "variant", "status"};
        std::vector<std::string> header(key_columns, key_columns + 4);
        for (const auto& c : order)
            if (c.find('#') == std::string::npos && std::find(header.begin(), header.end(), c) == header.end())
                header.push_back(c);
        std::vector<json::Value> labels(4);
        for (int k = 0; k < 4; ++k)
            labels[static_cast<std::size_t>(k)] = json::parse(std::string(cols.at(std::string(key_columns[k]) + "#labels").data.begin(),
                                                                         cols.at(std::string(key_columns[k]) + "#labels").data.end()));
        std::string text;
        for (std::size_t j = 0; j < header.size(); ++j) text += (j ? "," : "") + csv_field(header[j]);
        text += "\n";
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < header.size(); ++j) {
                if (j) text += ",";
                const store::Block& b = cols.at(header[j]);
                if (j < 4) {
                    const auto code = static_cast<std::size_t>(store::rec::get_le(b.data.data() + 4 * i, 4));
                    text += csv_field(labels[j].items.at(code).s);
                } else {
                    text += py_repr(cell(b, i));
                }
            }
            text += "\n";
        }
        const std::string stem = detail::utf8(pack.stem());
        const auto path = out / ("metrics-" + stem.substr(stem.find('-') + 1) + ".csv.gz");
        const auto gz = deflate::gzip(reinterpret_cast<const std::uint8_t*>(text.data()), text.size());
        write_bytes(path, gz);
        written.push_back(path);
    }
    return written;
}

}   // namespace mootation::campaign
