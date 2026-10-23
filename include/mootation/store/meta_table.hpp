#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A pack's meta table (task 17, supplement 5, §2.3; docs/store-format.md):
// meta.json of every run of a pack as one table, written and read here
// (§3а.3: one implementation in C++), the bytes store.py wrote.
//
// A run's JSON is a skeleton — its objects and arrays, every leaf a 0 —
// and its leaves in depth-first order, each in the column of its path (the
// JSON text, json.dumps's, of the keys and [index]es down to it). A column
// keeps its leaves' types (0 null, 1 false, 2 true, 3 int, 4 float, 5 text)
// and its ints, floats and texts apart: ints and types by the integer column,
// floats by the float64 column, texts as a dictionary of the column's
// distinct texts and an index each. Skeletons are a dictionary too. A run
// whose text json.dumps(..., indent=1) does not give back (CR LF kept where
// it had them) is kept whole.
//
//     "MOOTMET1" | u32 CRC-32C of the body | body
//     body: pieces, each a varint length and its bytes —
//       header (text codec: JSON {"rows", "columns"}), keys (text codec:
//       JSON), skeletons (text codec: JSON), each row's skeleton (integers;
//       -1 kept whole), CR LF (integers), the texts kept whole (text codec:
//       JSON {row: text}), then per column: types, ints, floats, the texts'
//       indices, the column's texts (text codec: JSON)
// ============================================================================

#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../json.hpp"
#include "column_codec.hpp"
#include "crc32c.hpp"
#include "run_json.hpp"
#include "text_codec.hpp"

namespace mootation::store {

constexpr char kMetaMagic[9] = "MOOTMET1";

namespace meta_table {

struct Column {
    std::vector<std::int64_t> types, ints, texts;
    std::vector<double> floats;
    json::Value dict = json::Value::array();
    std::map<std::string, std::int64_t> index;          // a text's place in dict
};

// v's skeleton; its leaves appended as (path's JSON, value)
inline json::Value skeleton(const json::Value& v, json::Value& path,
                            std::vector<std::pair<std::string, const json::Value*>>& leaves)
{
    using K = json::Value::Kind;
    if (v.kind == K::Object) {
        json::Value out = json::Value::object();
        for (std::size_t k = 0; k < v.keys.size(); ++k) {
            path.items.push_back(json::Value::string(v.keys[k]));
            out.keys.push_back(v.keys[k]);
            out.items.push_back(skeleton(v.items[k], path, leaves));
            path.items.pop_back();
        }
        return out;
    }
    if (v.kind == K::Array) {
        json::Value out = json::Value::array();
        for (std::size_t k = 0; k < v.items.size(); ++k) {
            json::Value at = json::Value::array();
            at.items.push_back(json::Value::integer(static_cast<std::int64_t>(k)));
            path.items.push_back(std::move(at));
            out.items.push_back(skeleton(v.items[k], path, leaves));
            path.items.pop_back();
        }
        return out;
    }
    leaves.emplace_back(json::dump(path), &v);
    return json::Value::integer(0);
}

// the leaves' paths of a skeleton, in order
inline void leaf_paths(const json::Value& skel, json::Value& path, std::vector<std::string>& out)
{
    using K = json::Value::Kind;
    if (skel.kind == K::Object) {
        for (std::size_t k = 0; k < skel.keys.size(); ++k) {
            path.items.push_back(json::Value::string(skel.keys[k]));
            leaf_paths(skel.items[k], path, out);
            path.items.pop_back();
        }
    } else if (skel.kind == K::Array) {
        for (std::size_t k = 0; k < skel.items.size(); ++k) {
            json::Value at = json::Value::array();
            at.items.push_back(json::Value::integer(static_cast<std::int64_t>(k)));
            path.items.push_back(std::move(at));
            leaf_paths(skel.items[k], path, out);
            path.items.pop_back();
        }
    } else {
        out.push_back(json::dump(path));
    }
}

inline void put_piece(std::vector<std::uint8_t>& out, const std::vector<std::uint8_t>& piece)
{
    col::put_varint(out, piece.size());
    out.insert(out.end(), piece.begin(), piece.end());
}

inline std::vector<std::uint8_t> text_piece(const json::Value& v)
{
    const std::string t = json::dump(v);
    return encode_text(reinterpret_cast<const std::uint8_t*>(t.data()), t.size());
}

inline json::Value text_of_piece(const std::pair<const std::uint8_t*, std::size_t>& p)
{
    const std::vector<std::uint8_t> t = decode_text(p.first, p.second);
    return json::parse(std::string(t.begin(), t.end()));
}

}   // namespace meta_table

// [(key, meta.json bytes)] as a pack's table.
inline std::vector<std::uint8_t> encode_meta_table(const std::vector<std::pair<std::string, std::vector<std::uint8_t>>>& rows)
{
    using meta_table::Column;
    json::Value keys = json::Value::array(), whole = json::Value::object(), skel_list = json::Value::array();
    std::map<std::string, std::int64_t> skel_ids_of;
    std::vector<std::int64_t> skel_ids, crlf;
    std::vector<std::string> order;                      // the columns, as first met
    std::map<std::string, Column> cols;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const auto& [key, text] = rows[r];
        keys.items.push_back(json::Value::string(key));
        const bool cr = has_crlf(text);
        crlf.push_back(cr ? 1 : 0);
        std::vector<std::pair<std::string, const json::Value*>> leaves;
        json::Value obj, skel;
        bool ok = false;
        try {
            obj = json::parse(std::string(text.begin(), text.end()));
            ok = meta_text(obj, cr) == text;
        } catch (const std::invalid_argument&) {
            ok = false;
        }
        std::string skel_text;
        if (ok) {
            json::Value path = json::Value::array();
            skel = meta_table::skeleton(obj, path, leaves);
            skel_text = json::dump(skel, -1, ",", ":");
        }
        if (!ok) {
            whole.set(std::to_string(r), json::Value::string(std::string(text.begin(), text.end())));
            skel_ids.push_back(-1);
            continue;
        }
        auto found = skel_ids_of.find(skel_text);
        if (found == skel_ids_of.end()) {
            found = skel_ids_of.emplace(skel_text, static_cast<std::int64_t>(skel_list.items.size())).first;
            skel_list.items.push_back(json::Value::string(skel_text));
        }
        skel_ids.push_back(found->second);
        for (const auto& [path, v] : leaves) {
            auto at = cols.find(path);
            if (at == cols.end()) {
                at = cols.emplace(path, Column{}).first;
                order.push_back(path);
            }
            Column& c = at->second;
            using K = json::Value::Kind;
            if (v->kind == K::Null) {
                c.types.push_back(0);
            } else if (v->kind == K::Bool) {
                c.types.push_back(v->b ? 2 : 1);
            } else if (v->kind == K::Int) {
                c.types.push_back(3);
                c.ints.push_back(v->i);
            } else if (v->kind == K::Float) {
                c.types.push_back(4);
                c.floats.push_back(v->f);
            } else {
                c.types.push_back(5);
                auto t = c.index.find(v->s);
                if (t == c.index.end()) {
                    t = c.index.emplace(v->s, static_cast<std::int64_t>(c.dict.items.size())).first;
                    c.dict.items.push_back(json::Value::string(v->s));
                }
                c.texts.push_back(t->second);
            }
        }
    }
    json::Value head = json::Value::object(), columns = json::Value::array();
    for (const auto& p : order) columns.items.push_back(json::Value::string(p));
    head.set("rows", json::Value::integer(static_cast<std::int64_t>(rows.size())));
    head.set("columns", std::move(columns));
    std::vector<std::uint8_t> body;
    meta_table::put_piece(body, meta_table::text_piece(head));
    meta_table::put_piece(body, meta_table::text_piece(keys));
    meta_table::put_piece(body, meta_table::text_piece(skel_list));
    meta_table::put_piece(body, encode_i64(skel_ids.data(), skel_ids.size()));
    meta_table::put_piece(body, encode_i64(crlf.data(), crlf.size()));
    meta_table::put_piece(body, meta_table::text_piece(whole));
    for (const auto& p : order) {
        const Column& c = cols.at(p);
        meta_table::put_piece(body, encode_i64(c.types.data(), c.types.size()));
        meta_table::put_piece(body, encode_i64(c.ints.data(), c.ints.size()));
        meta_table::put_piece(body, encode_f64(c.floats.data(), c.floats.size()));
        meta_table::put_piece(body, encode_i64(c.texts.data(), c.texts.size()));
        meta_table::put_piece(body, meta_table::text_piece(c.dict));
    }
    std::vector<std::uint8_t> out(kMetaMagic, kMetaMagic + 8);
    rec::put_le(out, crc32c(body.data(), body.size()), 4);
    out.insert(out.end(), body.begin(), body.end());
    return out;
}

// [(key, meta.json bytes)] of a pack's table, in its order; an error where it is damaged.
inline std::vector<std::pair<std::string, std::vector<std::uint8_t>>> decode_meta_table(const std::uint8_t* data,
                                                                                       std::size_t size)
{
    if (size < 12 || std::memcmp(data, kMetaMagic, 8) != 0) throw std::runtime_error("not a pack's meta table");
    const std::uint8_t* body = data + 12;
    const std::size_t n_body = size - 12;
    if (crc32c(body, n_body) != static_cast<std::uint32_t>(rec::get_le(data + 8, 4)))
        throw std::runtime_error("a pack's meta table: its checksum does not hold");
    std::vector<std::pair<const std::uint8_t*, std::size_t>> pieces;
    for (std::size_t at = 0; at < n_body;) {
        const std::uint64_t n = col::get_varint(body, n_body, at);
        if (n > n_body - at) throw std::runtime_error("a pack's meta table: a piece past its end");
        pieces.emplace_back(body + at, static_cast<std::size_t>(n));
        at += static_cast<std::size_t>(n);
    }
    if (pieces.size() < 6) throw std::runtime_error("a pack's meta table: too few pieces");
    using meta_table::text_of_piece;
    const json::Value head = text_of_piece(pieces[0]);
    const json::Value* rows_v = head.find("rows");
    const json::Value* columns = head.find("columns");
    if (!rows_v || !columns) throw std::runtime_error("a pack's meta table: no rows or columns");
    const std::size_t n = static_cast<std::size_t>(rows_v->i);
    const json::Value keys = text_of_piece(pieces[1]), skels = text_of_piece(pieces[2]), whole = text_of_piece(pieces[5]);
    const std::vector<std::int64_t> skel_ids = decode_i64(pieces[3].first, pieces[3].second, n);
    const std::vector<std::int64_t> crlf = decode_i64(pieces[4].first, pieces[4].second, n);
    if (keys.items.size() != n) throw std::runtime_error("a pack's meta table: keys and rows differ in number");
    // the skeletons, and each one's leaf paths
    std::vector<json::Value> skel_v;
    std::vector<std::vector<std::string>> skel_paths;
    for (const auto& s : skels.items) {
        skel_v.push_back(json::parse(s.s));
        json::Value path = json::Value::array();
        skel_paths.emplace_back();
        meta_table::leaf_paths(skel_v.back(), path, skel_paths.back());
    }
    std::map<std::string, std::size_t> count;
    for (std::int64_t sid : skel_ids) {
        if (sid < 0) continue;
        if (static_cast<std::size_t>(sid) >= skel_v.size()) throw std::runtime_error("a pack's meta table: a skeleton past the list");
        for (const auto& p : skel_paths[static_cast<std::size_t>(sid)]) ++count[p];
    }
    struct Cursor {
        std::vector<std::int64_t> types, ints, texts;
        std::vector<double> floats;
        json::Value dict;
        std::size_t t = 0, i = 0, f = 0, x = 0;
    };
    std::map<std::string, Cursor> cur;
    std::size_t at = 6;
    for (const auto& c : columns->items) {
        if (at + 5 > pieces.size()) throw std::runtime_error("a pack's meta table: a column's pieces missing");
        Cursor k;
        k.types = decode_i64(pieces[at].first, pieces[at].second, count[c.s]);
        std::size_t ni = 0, nf = 0, nx = 0;
        for (std::int64_t t : k.types) {
            if (t == 3) ++ni;
            if (t == 4) ++nf;
            if (t == 5) ++nx;
        }
        k.ints = decode_i64(pieces[at + 1].first, pieces[at + 1].second, ni);
        k.floats = decode_f64(pieces[at + 2].first, pieces[at + 2].second, nf);
        k.texts = decode_i64(pieces[at + 3].first, pieces[at + 3].second, nx);
        k.dict = text_of_piece(pieces[at + 4]);
        cur.emplace(c.s, std::move(k));
        at += 5;
    }
    // a row back: its skeleton filled depth first from the columns
    std::function<json::Value(const json::Value&, json::Value&)> fill = [&](const json::Value& skel,
                                                                           json::Value& path) -> json::Value {
        using K = json::Value::Kind;
        if (skel.kind == K::Object) {
            json::Value out = json::Value::object();
            for (std::size_t k = 0; k < skel.keys.size(); ++k) {
                path.items.push_back(json::Value::string(skel.keys[k]));
                out.keys.push_back(skel.keys[k]);
                out.items.push_back(fill(skel.items[k], path));
                path.items.pop_back();
            }
            return out;
        }
        if (skel.kind == K::Array) {
            json::Value out = json::Value::array();
            for (std::size_t k = 0; k < skel.items.size(); ++k) {
                json::Value idx = json::Value::array();
                idx.items.push_back(json::Value::integer(static_cast<std::int64_t>(k)));
                path.items.push_back(std::move(idx));
                out.items.push_back(fill(skel.items[k], path));
                path.items.pop_back();
            }
            return out;
        }
        auto it = cur.find(json::dump(path));
        if (it == cur.end()) throw std::runtime_error("a pack's meta table: a leaf of no column");
        Cursor& c = it->second;
        if (c.t >= c.types.size()) throw std::runtime_error("a pack's meta table: a column ends early");
        const std::int64_t t = c.types[c.t++];
        if (t == 0) return json::Value::null();
        if (t == 1 || t == 2) return json::Value::boolean(t == 2);
        if (t == 3) return json::Value::integer(c.ints.at(c.i++));
        if (t == 4) return json::Value::number(c.floats.at(c.f++));
        if (t == 5) return c.dict.items.at(static_cast<std::size_t>(c.texts.at(c.x++)));
        throw std::runtime_error("a pack's meta table: an unknown type");
    };
    std::vector<std::pair<std::string, std::vector<std::uint8_t>>> out;
    for (std::size_t r = 0; r < n; ++r) {
        const std::int64_t sid = skel_ids[r];
        if (sid < 0) {
            const json::Value* w = whole.find(std::to_string(r));
            if (!w) throw std::runtime_error("a pack's meta table: a whole text missing");
            out.emplace_back(keys.items[r].s, std::vector<std::uint8_t>(w->s.begin(), w->s.end()));
            continue;
        }
        json::Value path = json::Value::array();
        out.emplace_back(keys.items[r].s, meta_text(fill(skel_v[static_cast<std::size_t>(sid)], path), crlf[r] != 0));
    }
    return out;
}

}   // namespace mootation::store
