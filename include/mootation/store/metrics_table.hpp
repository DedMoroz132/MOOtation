#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A pack's metrics table (task 17, §3а.3: one implementation in C++; layer 3
// of docs/store-format.md), the bytes store.py wrote: one row per trajectory
// record of every run — the key's problem, algorithm and variant, the run's
// status (meta.json's "status", "?" without), seed, budget, then every
// number the records hold (a bool as 0/1, null as NaN; text left out), a
// record's key over the key's parts where they share a name, as a dict
// would — column after column:
//
//     "MOOTMTR1" | u32 CRC-32C of the body | body
//     body: pieces — a directory (JSON {"family", "rows", "columns": [[name,
//     dtype], ...]}), then each column's block (as a record's, without the
//     count): problem, algorithm, variant, status each as "<name>#labels"
//     (JSON, its distinct texts sorted) and "<name>" (<i4 indices into them);
//     seed, budget and the records' numbers as <f8, NaN where a row has none.
//
// A trajectory kept column by column (format 2 and 3: its #head, #cells and
// values) is read without its text; one kept as text (format 1) line by line.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../json.hpp"
#include "meta_table.hpp"
#include "record.hpp"
#include "run_json.hpp"

namespace mootation::store {

constexpr char kMetricsMagic[9] = "MOOTMTR1";

namespace metrics_table {

// a record's numbers in its keys' order: (key, value), a bool as 0/1, null NaN
using Record = std::vector<std::pair<std::string, double>>;
// a number kept as the int it was (a key's seed and budget, an int cell)
struct Cell {
    double v = 0.0;
};

inline std::vector<Record> records(const std::vector<Block>& blocks, const std::string& name = "trajectory.jsonl")
{
    std::vector<Record> out;
    const Block* h = run_json::named(blocks, name + "#head");
    const auto head = h ? run_json::parse(h->data) : std::nullopt;
    const json::Value* kind = head && head->kind == json::Value::Kind::Object ? head->find("kind") : nullptr;
    if (kind && kind->kind == json::Value::Kind::String && kind->s == "jsonl") {
        const Block* V = run_json::named(blocks, name);
        const Block* C = run_json::named(blocks, name + "#cells");
        const json::Value* cols = head->find("columns");
        if (!V || !C || !cols) throw std::runtime_error(name + ": no values, cells or columns");
        const std::size_t rows = V->shape.empty() ? 0 : static_cast<std::size_t>(V->shape[0]);
        const std::size_t m = V->shape.size() > 1 ? static_cast<std::size_t>(V->shape[1]) : 1;
        const std::size_t cm = C->shape.size() > 1 ? static_cast<std::size_t>(C->shape[1]) : 1;
        if (V->data.size() < 8 * rows * m || C->data.size() < rows * cm)
            throw std::runtime_error(name + ": its values and cells do not match");
        for (std::size_t r = 0; r < rows; ++r) {
            Record rec;
            for (std::size_t j = 0; j < cm; ++j) {
                const std::uint8_t code = C->data[r * cm + j];
                if (code == run_json::kAbsent || code == run_json::kText) continue;
                if (j >= m || j >= cols->items.size()) throw std::runtime_error(name + ": a cell past the columns");
                const std::uint64_t bits = rec::get_le(&V->data[8 * (r * m + j)], 8);
                double v;
                std::memcpy(&v, &bits, 8);
                if (code == run_json::kNull) v = std::numeric_limits<double>::quiet_NaN();
                else if (code == run_json::kBool) v = v != 0.0 ? 1.0 : 0.0;
                else if (code == run_json::kInt) v = std::trunc(v);           // int(v), back to a float
                rec.emplace_back(cols->items[j].s, v);
            }
            out.push_back(std::move(rec));
        }
        return out;
    }
    // format 1: the text, a JSON object a line
    const Block* t = run_json::named(blocks, name);
    if (!t) return out;
    const std::string text(t->data.begin(), t->data.end());
    std::size_t at = 0;
    while (at < text.size()) {
        std::size_t end = text.find('\n', at);
        if (end == std::string::npos) end = text.size();
        std::string line = text.substr(at, end - at);
        at = end + 1;
        if (line.find_first_not_of(" \t\r\f\v") == std::string::npos) continue;
        const json::Value obj = json::parse(line);
        Record rec;
        for (std::size_t k = 0; k < obj.keys.size(); ++k) {
            const json::Value& x = obj.items[k];
            using K = json::Value::Kind;
            if (x.kind == K::Bool) rec.emplace_back(obj.keys[k], x.b ? 1.0 : 0.0);
            else if (x.kind == K::Null) rec.emplace_back(obj.keys[k], std::numeric_limits<double>::quiet_NaN());
            else if (x.kind == K::Int) rec.emplace_back(obj.keys[k], static_cast<double>(x.i));
            else if (x.kind == K::Float) rec.emplace_back(obj.keys[k], x.f);
        }
        out.push_back(std::move(rec));
    }
    return out;
}

inline std::vector<std::string> split_key(const std::string& key)
{
    std::vector<std::string> parts;
    std::size_t at = 0;
    while (true) {
        const std::size_t bar = key.find('|', at);
        parts.push_back(key.substr(at, bar == std::string::npos ? std::string::npos : bar - at));
        if (bar == std::string::npos) break;
        at = bar + 1;
    }
    if (parts.size() != 5) throw std::invalid_argument("a run's key of other than five parts: " + key);
    return parts;
}

}   // namespace metrics_table

// The metrics table of a pack's runs [(key, blocks)], in the pack's order.
inline std::vector<std::uint8_t> encode_metrics_table(const std::string& family,
                                                     const std::vector<std::pair<std::string, std::vector<Block>>>& runs)
{
    static const char* const key_columns[4] = {"problem", "algorithm", "variant", "status"};
    // a row: the key's texts, then the numbers by name (a dict: a later value over an earlier one)
    struct Row {
        std::string text[4];
        std::map<std::string, double> num;
    };
    std::vector<Row> rows;
    std::vector<std::string> cols;                       // the records' numbers, as first met
    std::set<std::string> seen;
    for (const auto& [key, blocks] : runs) {
        const auto parts = metrics_table::split_key(key);
        std::string status = "?";
        if (const Block* mb = run_json::named(blocks, "meta.json")) {
            const json::Value meta = json::parse(run_json::text_of(mb->data));
            const json::Value* s = meta.find("status");
            if (s) status = s->kind == json::Value::Kind::String ? s->s : json::dump(*s);   // str(): a text, else its JSON
        }
        for (const auto& rec : metrics_table::records(blocks)) {
            Row row;
            row.text[0] = parts[0];
            row.text[1] = parts[1];
            row.text[2] = parts[2];
            row.text[3] = status;
            row.num["seed"] = static_cast<double>(std::stoll(parts[3]));
            row.num["budget"] = static_cast<double>(std::stoll(parts[4]));
            for (const auto& [k, v] : rec) {
                row.num[k] = v;
                if (seen.insert(k).second) cols.push_back(k);
            }
            rows.push_back(std::move(row));
        }
    }
    std::vector<Block> blocks;
    for (int c = 0; c < 4; ++c) {
        std::vector<std::string> labels;
        for (const Row& r : rows) labels.push_back(r.text[c]);
        std::sort(labels.begin(), labels.end());
        labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
        json::Value lv = json::Value::array();
        for (const auto& l : labels) lv.items.push_back(json::Value::string(l));
        const std::string lt = json::dump(lv);
        blocks.push_back({std::string(key_columns[c]) + "#labels", "json", {}, {lt.begin(), lt.end()}, ""});
        Block codes{key_columns[c], "<i4", {rows.size()}, {}, ""};
        for (const Row& r : rows) {
            const auto at = std::lower_bound(labels.begin(), labels.end(), r.text[c]) - labels.begin();
            rec::put_le(codes.data, static_cast<std::uint64_t>(at), 4);
        }
        blocks.push_back(std::move(codes));
    }
    std::vector<std::string> numeric = {"seed", "budget"};
    numeric.insert(numeric.end(), cols.begin(), cols.end());
    for (const auto& c : numeric) {
        Block b{c, "<f8", {rows.size()}, {}, ""};
        for (const Row& r : rows) {
            const auto it = r.num.find(c);
            const double v = it == r.num.end() ? std::numeric_limits<double>::quiet_NaN() : it->second;
            std::uint64_t bits;
            std::memcpy(&bits, &v, 8);
            rec::put_le(b.data, bits, 8);
        }
        blocks.push_back(std::move(b));
    }
    json::Value dir = json::Value::object(), columns = json::Value::array();
    for (const Block& b : blocks) {
        json::Value pair = json::Value::array();
        pair.items.push_back(json::Value::string(b.name));
        pair.items.push_back(json::Value::string(b.dtype));
        columns.items.push_back(std::move(pair));
    }
    dir.set("family", json::Value::string(family));
    dir.set("rows", json::Value::integer(static_cast<std::int64_t>(rows.size())));
    dir.set("columns", std::move(columns));
    const std::string dt = json::dump(dir);
    std::vector<std::uint8_t> body;
    meta_table::put_piece(body, std::vector<std::uint8_t>(dt.begin(), dt.end()));
    for (const Block& b : blocks) meta_table::put_piece(body, encode_block(b));
    std::vector<std::uint8_t> out(kMetricsMagic, kMetricsMagic + 8);
    rec::put_le(out, crc32c(body.data(), body.size()), 4);
    out.insert(out.end(), body.begin(), body.end());
    return out;
}

}   // namespace mootation::store
