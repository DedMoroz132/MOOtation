#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// What a run's record reads from its JSON (task 17, §3а.3: one implementation
// in C++), byte for byte what store.py did:
//
//   * the point tables: a float64 block of two dimensions whose "#head" (a
//     json block "<name>#head", an object) names its columns, as many as the
//     block has;
//   * the trajectory's last record, from "trajectory.jsonl#head" (kind
//     "jsonl", its columns and texts), its values (float64, rows x columns)
//     and its cells (u1: 0 absent, 1 float, 2 int, 3 bool, 4 null, 5 text);
//   * meta.json's "final" once (supplement 5, §2.4): the keys whose value the
//     last record holds alike (json.dumps's text equal) are left to it —
//     final keeps "#keys" (all of them, in order) and "#trajectory" (those),
//     then the others —, where meta.json then comes back byte for byte
//     (json.dumps(meta, indent=1), CR LF where the stored text has them);
//     and back.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../json.hpp"
#include "record.hpp"

namespace mootation::store {

namespace run_json {

enum Cell : std::uint8_t { kAbsent = 0, kFloat = 1, kInt = 2, kBool = 3, kNull = 4, kText = 5 };

inline const Block* named(const std::vector<Block>& blocks, const std::string& name)
{
    for (const Block& b : blocks)
        if (b.name == name) return &b;
    return nullptr;
}

inline std::optional<json::Value> parse(const std::vector<std::uint8_t>& data)
{
    try {
        return json::parse(std::string(data.begin(), data.end()));
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
}

inline std::string text_of(const std::vector<std::uint8_t>& d) { return std::string(d.begin(), d.end()); }

}   // namespace run_json

// The point tables among blocks, their places and columns.
inline std::vector<PointTable> point_tables(const std::vector<Block>& blocks)
{
    std::vector<PointTable> out;
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        const Block& b = blocks[i];
        if (b.dtype != "<f8" || b.shape.size() != 2) continue;
        // the last "#head" of that name, as a dict of them would keep
        const Block* h = nullptr;
        for (const Block& c : blocks)
            if (c.name == b.name + "#head" && c.dtype == "json") h = &c;
        if (!h) continue;
        const auto head = run_json::parse(h->data);
        if (!head || head->kind != json::Value::Kind::Object) continue;
        const json::Value* cols = head->find("columns");
        if (!cols || cols->kind != json::Value::Kind::Array || cols->items.size() != b.shape[1]) continue;
        PointTable t{i, {}};
        bool strings = true;
        for (const auto& c : cols->items) {
            if (c.kind != json::Value::Kind::String) strings = false;
            else t.columns.push_back(c.s);
        }
        if (strings) out.push_back(std::move(t));
    }
    return out;
}

// The trajectory's last record as an object, or nothing.
inline std::optional<json::Value> last_record(const std::vector<Block>& blocks,
                                              const std::string& name = "trajectory.jsonl")
{
    const Block* h = run_json::named(blocks, name + "#head");
    const Block* V = run_json::named(blocks, name);
    const Block* C = run_json::named(blocks, name + "#cells");
    if (!h || !V || !C || V->dtype != "<f8" || V->shape.empty() || V->shape[0] == 0) return std::nullopt;
    const auto head = run_json::parse(h->data);
    if (!head || head->kind != json::Value::Kind::Object) return std::nullopt;
    const json::Value* kind = head->find("kind");
    if (!kind || kind->kind != json::Value::Kind::String || kind->s != "jsonl") return std::nullopt;
    const json::Value* cols = head->find("columns");
    const json::Value* texts = head->find("texts");
    if (!cols || !texts) throw std::runtime_error(name + "#head: no columns or texts");
    const std::size_t rows = static_cast<std::size_t>(V->shape[0]);
    const std::size_t m = V->shape.size() > 1 ? static_cast<std::size_t>(V->shape[1]) : 1;
    const std::size_t cm = C->shape.size() > 1 ? static_cast<std::size_t>(C->shape[1]) : 1;
    if (V->data.size() < 8 * rows * m || C->data.size() < rows * cm || C->shape.empty() ||
        C->shape[0] != V->shape[0])
        throw std::runtime_error(name + ": its values and cells do not match");
    json::Value rec = json::Value::object();
    for (std::size_t j = 0; j < cm; ++j) {
        const std::uint8_t code = C->data[(rows - 1) * cm + j];
        if (code == run_json::kAbsent) continue;
        if (j >= m || j >= cols->items.size()) throw std::runtime_error(name + ": a cell past the columns");
        const std::uint64_t bits = rec::get_le(&V->data[8 * ((rows - 1) * m + j)], 8);
        double v;
        std::memcpy(&v, &bits, 8);
        json::Value x;
        if (code == run_json::kFloat) {
            x = json::Value::number(v);
        } else if (code == run_json::kInt) {
            if (!(std::fabs(v) < 9.2e18)) throw std::runtime_error(name + ": an int past int64");
            x = json::Value::integer(static_cast<std::int64_t>(v));      // int(): towards zero
        } else if (code == run_json::kBool) {
            x = json::Value::boolean(v != 0.0);
        } else if (code == run_json::kNull) {
            x = json::Value::null();
        } else if (code == run_json::kText) {
            const auto k = static_cast<std::size_t>(v);
            if (k >= texts->items.size()) throw std::runtime_error(name + ": a text past the texts");
            x = texts->items[k];
        } else {
            throw std::runtime_error(name + ": an unknown cell");
        }
        rec.set(cols->items[j].s, std::move(x));
    }
    return rec;
}

// meta.json's text as a campaign writes it: json.dumps(meta, indent=1), CR LF where asked.
inline std::vector<std::uint8_t> meta_text(const json::Value& meta, bool crlf)
{
    const std::string t = json::dump(meta, 1);
    std::vector<std::uint8_t> out;
    out.reserve(t.size() + (crlf ? t.size() / 8 : 0));
    for (char c : t) {
        if (c == '\n' && crlf) out.push_back('\r');
        out.push_back(static_cast<std::uint8_t>(c));
    }
    return out;
}

inline bool has_crlf(const std::vector<std::uint8_t>& d)
{
    for (std::size_t i = 0; i + 1 < d.size(); ++i)
        if (d[i] == '\r' && d[i + 1] == '\n') return true;
    return false;
}

// meta.json's text with "final" rebuilt from the trajectory's last record.
inline std::vector<std::uint8_t> final_text(const std::vector<std::uint8_t>& text, const json::Value& rec)
{
    json::Value meta = json::parse(run_json::text_of(text));
    const json::Value* f = meta.find("final");
    if (!f) throw std::runtime_error("meta.json: no final");
    const json::Value* keys = f->find("#keys");
    const json::Value* from = f->find("#trajectory");
    if (!keys || !from) throw std::runtime_error("meta.json: final without #keys or #trajectory");
    json::Value nf = json::Value::object();
    for (const auto& k : keys->items) {
        bool traj = false;
        for (const auto& t : from->items) traj = traj || (t.kind == json::Value::Kind::String && t.s == k.s);
        const json::Value* v = traj ? rec.find(k.s) : f->find(k.s);
        if (!v) throw std::runtime_error("meta.json: final's " + k.s + " is in neither place");
        nf.set(k.s, *v);
    }
    meta.set("final", std::move(nf));
    return meta_text(meta, has_crlf(text));
}

// Blocks with meta.json's final values the trajectory's last record holds
// alike left to it; unchanged where meta.json would not come back byte for byte.
inline std::vector<Block> final_once(std::vector<Block> blocks)
{
    Block* mb = nullptr;
    for (Block& b : blocks)
        if (b.name == "meta.json") mb = &b;
    if (!mb) return blocks;
    const auto rec = last_record(blocks);
    if (!rec) return blocks;
    auto meta = run_json::parse(mb->data);
    if (!meta || meta->kind != json::Value::Kind::Object) return blocks;
    const json::Value* final = meta->find("final");
    if (!final || final->kind != json::Value::Kind::Object || final->keys.empty() || final->find("#keys"))
        return blocks;
    json::Value same = json::Value::array(), all = json::Value::array();
    for (std::size_t k = 0; k < final->keys.size(); ++k) {
        all.items.push_back(json::Value::string(final->keys[k]));
        const json::Value* r = rec->find(final->keys[k]);
        if (r && json::dump(*r) == json::dump(final->items[k])) same.items.push_back(json::Value::string(final->keys[k]));
    }
    if (same.items.empty()) return blocks;
    json::Value nf = json::Value::object();
    nf.set("#keys", std::move(all));
    nf.set("#trajectory", same);
    for (std::size_t k = 0; k < final->keys.size(); ++k) {
        bool taken = false;
        for (const auto& s : same.items) taken = taken || s.s == final->keys[k];
        if (!taken) nf.set(final->keys[k], final->items[k]);
    }
    meta->set("final", std::move(nf));
    std::vector<std::uint8_t> text = meta_text(*meta, has_crlf(mb->data));
    std::vector<std::uint8_t> back;
    try {
        back = final_text(text, *rec);
    } catch (const std::exception&) {
        return blocks;
    }
    if (back != mb->data) return blocks;
    mb->data = std::move(text);
    return blocks;
}

// meta.json's final rebuilt where it was left to the trajectory.
inline std::vector<Block> final_back(std::vector<Block> blocks)
{
    Block* mb = nullptr;
    for (Block& b : blocks)
        if (b.name == "meta.json") mb = &b;
    if (!mb) return blocks;
    static const std::string mark = "\"#keys\"";
    if (std::search(mb->data.begin(), mb->data.end(), mark.begin(), mark.end()) == mb->data.end()) return blocks;
    const auto rec = last_record(blocks);
    if (!rec) throw std::runtime_error("meta.json: its final is in the trajectory, which the record lacks");
    mb->data = final_text(mb->data, *rec);
    return blocks;
}

// A run's body: meta.json's final once, the point tables together (record.hpp).
inline std::vector<std::uint8_t> encode_run(const std::vector<Block>& blocks)
{
    const std::vector<Block> once = final_once(blocks);
    return encode_body(once, point_tables(once));
}

// A run's blocks back from its body, final rebuilt (a body with blocks of
// before, zlib, comes back as decode_body gives it: those have no final once).
inline std::vector<Block> decode_run(const std::uint8_t* data, std::size_t size)
{
    std::vector<Block> blocks = decode_body(data, size);
    for (const Block& b : blocks)
        if (!b.how.empty()) return blocks;
    return final_back(std::move(blocks));
}

}   // namespace mootation::store
