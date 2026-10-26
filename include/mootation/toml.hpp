#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// TOML 1.0 (toml.io/en/v1.0.0), read into the tree of json.hpp — what
// Python's tomllib.loads gives a campaign's configuration (task 17, §4: the
// campaign engine in C++ reads the same files run/config.py reads):
//
//   * a table is an object, its keys in the order they first came; an array
//     of tables is an array of objects;
//   * strings (basic, literal, both multi-line, every escape), integers
//     (decimal, 0x, 0o, 0b, underscores; int64), floats (inf and nan with a
//     sign, correctly rounded), booleans;
//   * dates and times are kept as their text (Kind::String, `datetime` set):
//     the campaign's files have none, and a reader that wants them parses the
//     text it is given.
// The rules tomllib keeps are kept: a key defined twice, a table defined twice
// or extended where it was closed (an inline table, a static array, a value),
// a table header over a key's value — an error, with its line and column.
// ============================================================================

#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

namespace mootation::toml {

class Error : public std::invalid_argument {
public:
    Error(const std::string& what, std::size_t line, std::size_t column)
        : std::invalid_argument(what + " (at line " + std::to_string(line) + ", column " + std::to_string(column) + ")"),
          line(line), column(column) {}
    std::size_t line, column;
};

namespace detail {

using json::Value;
using Kind = Value::Kind;

// The flags tomllib keeps on keys (its Flags): FROZEN — a value, an inline
// table or a static array, and all under it; EXPLICIT_NEST — a table made by
// a header, or a table a dotted key opened (pending until the next header).
struct Flags {
    struct Node {
        bool frozen = false, explicit_nest = false;          // on this key alone
        bool frozen_all = false, explicit_nest_all = false;  // on it and all under it
        std::map<std::string, std::unique_ptr<Node>> nested;
    };
    enum Flag { Frozen, ExplicitNest };
    Node root;
    std::vector<std::pair<std::vector<std::string>, Flag>> pending;

    static bool& bit(Node& n, Flag f, bool recursive)
    {
        if (f == Frozen) return recursive ? n.frozen_all : n.frozen;
        return recursive ? n.explicit_nest_all : n.explicit_nest;
    }
    void set(const std::vector<std::string>& key, Flag f, bool recursive)
    {
        Node* c = &root;
        for (const auto& k : key) {
            auto& next = c->nested[k];
            if (!next) next = std::make_unique<Node>();
            c = next.get();
        }
        bit(*c, f, recursive) = true;
    }
    void unset_all(const std::vector<std::string>& key)
    {
        Node* c = &root;
        for (std::size_t j = 0; j + 1 < key.size(); ++j) {
            auto it = c->nested.find(key[j]);
            if (it == c->nested.end()) return;
            c = it->second.get();
        }
        c->nested.erase(key.back());
    }
    bool is(const std::vector<std::string>& key, Flag f) const
    {
        if (key.empty()) return false;
        const Node* c = &root;
        for (std::size_t j = 0; j + 1 < key.size(); ++j) {
            auto it = c->nested.find(key[j]);
            if (it == c->nested.end()) return false;
            if (f == Frozen ? it->second->frozen_all : it->second->explicit_nest_all) return true;
            c = it->second.get();
        }
        auto it = c->nested.find(key.back());
        if (it == c->nested.end()) return false;
        const Node& n = *it->second;
        return f == Frozen ? (n.frozen || n.frozen_all) : (n.explicit_nest || n.explicit_nest_all);
    }
    void finalize_pending()
    {
        for (const auto& [k, f] : pending) set(k, f, false);
        pending.clear();
    }
};

struct Parser {
    const std::string& t;
    std::size_t at = 0;
    Value root = Value::object();
    Flags flags;
    std::vector<std::string> current;                     // the header the key/values go under

    [[noreturn]] void fail(const std::string& what) const
    {
        std::size_t line = 1, col = 1;
        for (std::size_t i = 0; i < at && i < t.size(); ++i) {
            if (t[i] == '\n') {
                ++line;
                col = 1;
            } else {
                ++col;
            }
        }
        throw Error(what, line, col);
    }
    bool end() const { return at >= t.size(); }
    char peek(std::size_t k = 0) const { return at + k < t.size() ? t[at + k] : '\0'; }

    static std::string path_text(const std::vector<std::string>& p)
    {
        std::string s;
        for (const auto& k : p) s += json::dump(Value::string(k)) + ".";
        return s;
    }

    void blank()                                          // spaces and tabs
    {
        while (!end() && (t[at] == ' ' || t[at] == '\t')) ++at;
    }
    void comment()
    {
        if (peek() != '#') return;
        while (!end() && t[at] != '\n') {
            const auto c = static_cast<unsigned char>(t[at]);
            if ((c < 0x20 && c != '\t' && c != '\r') || c == 0x7F) fail("Illegal character in a comment");
            ++at;
        }
    }
    void line_end()                                       // the rest of a line: blanks, a comment, the newline
    {
        blank();
        comment();
        if (end()) return;
        if (t[at] == '\r' && peek(1) == '\n') {
            at += 2;
            return;
        }
        if (t[at] == '\n') {
            ++at;
            return;
        }
        fail("Expected newline or end of document after a statement");
    }
    void blank_lines()                                    // whitespace, newlines and comments (in arrays)
    {
        while (!end()) {
            if (t[at] == ' ' || t[at] == '\t' || t[at] == '\n') {
                ++at;
            } else if (t[at] == '\r' && peek(1) == '\n') {
                at += 2;
            } else if (t[at] == '#') {
                comment();
            } else {
                break;
            }
        }
    }

    // ── keys ──
    static bool bare(char c)
    {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
    }
    std::string key_part()
    {
        if (peek() == '"') {
            if (peek(1) == '"' && peek(2) == '"') fail("Multiline strings are not allowed as keys");
            return basic_string();
        }
        if (peek() == '\'') {
            if (peek(1) == '\'' && peek(2) == '\'') fail("Multiline strings are not allowed as keys");
            return literal_string();
        }
        const std::size_t s = at;
        while (!end() && bare(t[at])) ++at;
        if (at == s) fail("Invalid initial character for a key part");
        return t.substr(s, at - s);
    }
    std::vector<std::string> key()
    {
        std::vector<std::string> k;
        blank();
        k.push_back(key_part());
        blank();
        while (peek() == '.') {
            ++at;
            blank();
            k.push_back(key_part());
            blank();
        }
        return k;
    }

    // ── strings ──
    static void put_utf8(std::string& out, std::uint32_t cp)
    {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    void escape(std::string& out)                         // after the backslash
    {
        const char e = peek();
        ++at;
        switch (e) {
        case 'b': out += '\b'; return;
        case 't': out += '\t'; return;
        case 'n': out += '\n'; return;
        case 'f': out += '\f'; return;
        case 'r': out += '\r'; return;
        case '"': out += '"'; return;
        case '\\': out += '\\'; return;
        case 'u':
        case 'U': {
            const int n = e == 'u' ? 4 : 8;
            std::uint32_t cp = 0;
            for (int k = 0; k < n; ++k) {
                const char c = peek();
                ++at;
                cp <<= 4;
                if (c >= '0' && c <= '9') cp |= static_cast<std::uint32_t>(c - '0');
                else if (c >= 'a' && c <= 'f') cp |= static_cast<std::uint32_t>(c - 'a' + 10);
                else if (c >= 'A' && c <= 'F') cp |= static_cast<std::uint32_t>(c - 'A' + 10);
                else fail("Invalid hex value");
            }
            if (cp > 0x10FFFF || (cp >= 0xD800 && cp < 0xE000)) fail("Escaped character is not a Unicode scalar value");
            put_utf8(out, cp);
            return;
        }
        default: fail("Unescaped '\\' in a string");
        }
    }
    void char_ok(unsigned char c, bool multiline)
    {
        if ((c < 0x20 && c != '\t' && !(multiline && (c == '\n' || c == '\r'))) || c == 0x7F)
            fail("Illegal character in a string");
    }
    std::string basic_string()
    {
        ++at;
        std::string out;
        while (true) {
            if (end()) fail("Unterminated string");
            const auto c = static_cast<unsigned char>(t[at]);
            if (c == '"') {
                ++at;
                return out;
            }
            if (c == '\\') {
                ++at;
                escape(out);
                continue;
            }
            char_ok(c, false);
            out += static_cast<char>(c);
            ++at;
        }
    }
    std::string literal_string()
    {
        ++at;
        const std::size_t s = at;
        while (true) {
            if (end()) fail("Unterminated string");
            const auto c = static_cast<unsigned char>(t[at]);
            if (c == '\'') break;
            char_ok(c, false);
            ++at;
        }
        std::string out = t.substr(s, at - s);
        ++at;
        return out;
    }
    std::string multiline(char q)                         // after the three quotes
    {
        at += 3;
        if (peek() == '\n') ++at;                         // a newline right after the quotes is trimmed
        else if (peek() == '\r' && peek(1) == '\n') at += 2;
        std::string out;
        while (true) {
            if (end()) fail("Unterminated string");
            if (t[at] == q && peek(1) == q && peek(2) == q) {
                // up to two quotes may close it: the longest run of quotes ends the string
                std::size_t n = 3;
                while (peek(n) == q && n < 5) ++n;
                out.append(n - 3, q);
                at += n;
                return out;
            }
            const auto c = static_cast<unsigned char>(t[at]);
            if (q == '"' && c == '\\') {
                // a line-ending backslash: the newline and the whitespace after it go
                std::size_t k = at + 1;
                while (k < t.size() && (t[k] == ' ' || t[k] == '\t')) ++k;
                if (k < t.size() && (t[k] == '\n' || (t[k] == '\r' && k + 1 < t.size() && t[k + 1] == '\n'))) {
                    at = k;
                    while (!end() && (t[at] == ' ' || t[at] == '\t' || t[at] == '\n' || t[at] == '\r')) {
                        if (t[at] == '\r' && peek(1) != '\n') fail("Illegal character in a string");
                        ++at;
                    }
                    continue;
                }
                ++at;
                escape(out);
                continue;
            }
            if (c == '\r') {
                if (peek(1) != '\n') fail("Illegal character in a string");
                out += "\r\n";
                at += 2;
                continue;
            }
            char_ok(c, true);
            out += static_cast<char>(c);
            ++at;
        }
    }

    // ── numbers, booleans, dates ──
    static bool digit(char c) { return c >= '0' && c <= '9'; }
    Value number_or_date()
    {
        const std::size_t s = at;
        // a date or time: four digits and '-', or two digits and ':'
        if (digit(peek()) && digit(peek(1)) && digit(peek(2)) && digit(peek(3)) && peek(4) == '-') return datetime(s);
        if (digit(peek()) && digit(peek(1)) && peek(2) == ':') return datetime(s);
        std::size_t e = at;
        while (e < t.size() && (std::isalnum(static_cast<unsigned char>(t[e])) || t[e] == '_' || t[e] == '+' ||
                                t[e] == '-' || t[e] == '.'))
            ++e;
        const std::string w = t.substr(s, e - s);
        auto bad = [&]() -> Value { fail("Invalid value"); };
        if (w == "inf" || w == "+inf") { at = e; return Value::number(std::numeric_limits<double>::infinity()); }
        if (w == "-inf") { at = e; return Value::number(-std::numeric_limits<double>::infinity()); }
        if (w == "nan" || w == "+nan" || w == "-nan") {
            at = e;
            const double v = std::numeric_limits<double>::quiet_NaN();
            return Value::number(w[0] == '-' ? -v : v);
        }
        if (w.size() > 2 && w[0] == '0' && (w[1] == 'x' || w[1] == 'o' || w[1] == 'b')) {
            const int base = w[1] == 'x' ? 16 : w[1] == 'o' ? 8 : 2;
            std::string d;
            for (std::size_t i = 2; i < w.size(); ++i) {
                if (w[i] == '_') {
                    if (i == 2 || i + 1 == w.size() || w[i + 1] == '_') return bad();
                    continue;
                }
                const char c = static_cast<char>(std::tolower(static_cast<unsigned char>(w[i])));
                const int v = digit(c) ? c - '0' : (c >= 'a' && c <= 'f') ? c - 'a' + 10 : 99;
                if (v >= base) return bad();
                d += c;
            }
            std::uint64_t u = 0;
            for (char c : d) {
                const std::uint64_t v = digit(c) ? static_cast<std::uint64_t>(c - '0') : static_cast<std::uint64_t>(c - 'a' + 10);
                if (u > (std::numeric_limits<std::uint64_t>::max() - v) / static_cast<std::uint64_t>(base)) return bad();
                u = u * static_cast<std::uint64_t>(base) + v;
            }
            if (u > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) fail("Integer out of range");
            at = e;
            return Value::integer(static_cast<std::int64_t>(u));
        }
        // decimal: [+-] digits with single underscores between them, then a fraction and/or an exponent
        std::string clean;
        std::size_t i = 0;
        if (i < w.size() && (w[i] == '+' || w[i] == '-')) clean += w[i++];
        auto digits = [&](bool lead_zero_ok) {
            const std::size_t from = i;
            if (i >= w.size() || !digit(w[i])) return false;
            while (i < w.size() && (digit(w[i]) || w[i] == '_')) {
                if (w[i] == '_' && (i + 1 >= w.size() || !digit(w[i + 1]))) return false;
                if (w[i] != '_') clean += w[i];
                ++i;
            }
            if (!lead_zero_ok && i - from > 1 && w[from] == '0') return false;
            return true;
        };
        if (!digits(false)) return bad();
        bool real = false;
        if (i < w.size() && w[i] == '.') {
            clean += w[i++];
            if (!digits(true)) return bad();
            real = true;
        }
        if (i < w.size() && (w[i] == 'e' || w[i] == 'E')) {
            clean += w[i++];
            if (i < w.size() && (w[i] == '+' || w[i] == '-')) clean += w[i++];
            if (!digits(true)) return bad();
            real = true;
        }
        if (i != w.size()) return bad();
        at = e;
        if (!real) {
            errno = 0;
            char* end_ = nullptr;
            const long long v = std::strtoll(clean.c_str(), &end_, 10);
            if (errno == ERANGE) fail("Integer out of range");
            return Value::integer(static_cast<std::int64_t>(v));
        }
        return Value::number(std::strtod(clean.c_str(), nullptr));     // correctly rounded (json.hpp's note)
    }
    Value datetime(std::size_t s)
    {
        // the text of an offset or local date-time, a local date or a local time, checked for its shape
        std::size_t e = s;
        auto run = [&](std::size_t n) {
            for (std::size_t k = 0; k < n; ++k)
                if (e >= t.size() || !digit(t[e++])) fail("Invalid date or time");
        };
        auto take = [&](char c) {
            if (e >= t.size() || t[e] != c) fail("Invalid date or time");
            ++e;
        };
        auto time = [&] {
            run(2);
            take(':');
            run(2);
            take(':');
            run(2);
            if (e < t.size() && t[e] == '.') {
                ++e;
                const std::size_t f = e;
                while (e < t.size() && digit(t[e])) ++e;
                if (e == f) fail("Invalid date or time");
            }
        };
        if (digit(t[s]) && s + 2 < t.size() && t[s + 2] == ':') {
            time();
        } else {
            run(4);
            take('-');
            run(2);
            take('-');
            run(2);
            if (e + 1 < t.size() && (t[e] == 'T' || t[e] == 't' || (t[e] == ' ' && digit(t[e + 1])))) {
                ++e;
                time();
                if (e < t.size() && (t[e] == 'Z' || t[e] == 'z')) {
                    ++e;
                } else if (e < t.size() && (t[e] == '+' || t[e] == '-')) {
                    ++e;
                    run(2);
                    take(':');
                    run(2);
                }
            }
        }
        Value v = Value::string(t.substr(s, e - s));
        v.datetime = true;
        at = e;
        return v;
    }

    // ── values ──
    Value array(int depth)
    {
        ++at;
        Value a = Value::array();
        while (true) {
            blank_lines();
            if (peek() == ']') {
                ++at;
                return a;
            }
            a.items.push_back(value(depth + 1));
            blank_lines();
            if (peek() == ',') {
                ++at;
                continue;
            }
            if (peek() == ']') {
                ++at;
                return a;
            }
            fail("Unclosed array");
        }
    }
    Value inline_table(int depth)
    {
        ++at;
        Value o = Value::object();
        Flags own;
        blank();
        if (peek() == '}') {
            ++at;
            return o;
        }
        while (true) {
            const std::vector<std::string> k = key();
            if (peek() != '=') fail("Expected '=' after a key in a key/value pair");
            ++at;
            blank();
            Value v = value(depth + 1);
            if (own.is(k, Flags::Frozen)) fail("Cannot mutate immutable namespace");
            Value* nest = nest_of(o, std::vector<std::string>(k.begin(), k.end() - 1), false);
            if (!nest) fail("Cannot overwrite a value");
            if (nest->find(k.back())) fail("Duplicate inline table key");
            const bool container = v.kind == Kind::Object || v.kind == Kind::Array;
            nest->set(k.back(), std::move(v));
            if (container) own.set(k, Flags::Frozen, true);
            blank();
            if (peek() == ',') {
                ++at;
                blank();
                if (peek() == '}') fail("Trailing comma in an inline table");
                continue;
            }
            if (peek() == '}') {
                ++at;
                return o;
            }
            fail("Unclosed inline table");
        }
    }
    Value value(int depth)
    {
        if (depth > 1000) fail("Nested too deep");
        const char c = peek();
        if (c == '"') {
            if (peek(1) == '"' && peek(2) == '"') return Value::string(multiline('"'));
            return Value::string(basic_string());
        }
        if (c == '\'') {
            if (peek(1) == '\'' && peek(2) == '\'') return Value::string(multiline('\''));
            return Value::string(literal_string());
        }
        if (c == '[') return array(depth);
        if (c == '{') return inline_table(depth);
        if (t.compare(at, 4, "true") == 0 && !bare(peek(4))) {
            at += 4;
            return Value::boolean(true);
        }
        if (t.compare(at, 5, "false") == 0 && !bare(peek(5))) {
            at += 5;
            return Value::boolean(false);
        }
        if (digit(c) || c == '+' || c == '-' || c == 'i' || c == 'n') return number_or_date();
        fail("Invalid value");
    }

    // ── the tree (tomllib's NestedDict) ──
    // the table behind key, made where missing; with lists, an array of
    // tables' last element; nullptr where something else stands there
    static Value* nest_of(Value& from, const std::vector<std::string>& key, bool access_lists = true)
    {
        Value* cont = &from;
        for (const auto& k : key) {
            Value* next = nullptr;
            for (std::size_t m = 0; m < cont->keys.size(); ++m)
                if (cont->keys[m] == k) next = &cont->items[m];
            if (!next) {
                cont->set(k, Value::object());
                next = &cont->items.back();
            }
            if (access_lists && next->kind == Kind::Array) {
                if (next->items.empty()) return nullptr;
                next = &next->items.back();
            }
            if (next->kind != Kind::Object) return nullptr;
            cont = next;
        }
        return cont;
    }

    void key_value()
    {
        const std::vector<std::string> k = key();
        if (peek() != '=') fail("Expected '=' after a key in a key/value pair");
        ++at;
        blank();
        Value v = value(0);
        std::vector<std::string> parent = current;
        parent.insert(parent.end(), k.begin(), k.end() - 1);
        for (std::size_t i = 1; i < k.size(); ++i) {
            std::vector<std::string> cont = current;
            cont.insert(cont.end(), k.begin(), k.begin() + static_cast<std::ptrdiff_t>(i));
            if (flags.is(cont, Flags::ExplicitNest)) fail("Cannot redefine namespace");
            flags.pending.emplace_back(cont, Flags::ExplicitNest);
        }
        if (flags.is(parent, Flags::Frozen)) fail("Cannot mutate immutable namespace");
        Value* nest = nest_of(root, parent);
        if (!nest) fail("Cannot overwrite a value");
        if (nest->find(k.back())) fail("Cannot overwrite a value");
        const bool container = v.kind == Kind::Object || v.kind == Kind::Array;
        nest->set(k.back(), std::move(v));
        if (container) {
            std::vector<std::string> full = current;
            full.insert(full.end(), k.begin(), k.end());
            flags.set(full, Flags::Frozen, true);
        }
    }

    void header()
    {
        flags.finalize_pending();
        const bool aot = peek(1) == '[';
        at += aot ? 2 : 1;
        blank();
        const std::vector<std::string> k = key();
        if (aot) {
            if (flags.is(k, Flags::Frozen)) fail("Cannot mutate immutable namespace");
            flags.unset_all(k);
            flags.set(k, Flags::ExplicitNest, false);
            Value* parent = nest_of(root, std::vector<std::string>(k.begin(), k.end() - 1));
            if (!parent) fail("Cannot overwrite a value");
            Value* list = nullptr;
            for (std::size_t m = 0; m < parent->keys.size(); ++m)
                if (parent->keys[m] == k.back()) list = &parent->items[m];
            if (list) {
                if (list->kind != Kind::Array) fail("Cannot overwrite a value");
                list->items.push_back(Value::object());
            } else {
                Value a = Value::array();
                a.items.push_back(Value::object());
                parent->set(k.back(), std::move(a));
            }
            if (peek() != ']' || peek(1) != ']') fail("Expected ']]' at the end of an array declaration");
            at += 2;
        } else {
            if (flags.is(k, Flags::ExplicitNest) || flags.is(k, Flags::Frozen)) fail("Cannot declare a table twice");
            flags.set(k, Flags::ExplicitNest, false);
            if (!nest_of(root, k)) fail("Cannot overwrite a value");
            if (peek() != ']') fail("Expected ']' at the end of a table declaration");
            ++at;
        }
        current = k;
    }

    void document()
    {
        if (t.compare(0, 3, "\xEF\xBB\xBF") == 0) at = 3;       // a byte-order mark
        while (true) {
            blank_lines();
            if (end()) return;
            if (peek() == '[') header();
            else key_value();
            line_end();
        }
    }
};

}   // namespace detail

// tomllib.loads(text): the document's tree; toml::Error with its line and column
inline json::Value parse(const std::string& text)
{
    detail::Parser p{text};
    p.document();
    return std::move(p.root);
}

}   // namespace mootation::toml
