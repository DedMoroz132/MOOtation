#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// JSON as the run layer writes and reads it (task 17, stage 5 groundwork: the
// store's compact/verify/merge/export in C++ and a campaign's meta.json),
// byte for byte what Python's json module gives:
//
//   * parse: json.loads — RFC 8259 plus NaN, Infinity and -Infinity; a number
//     with neither fraction nor exponent an integer (int64 here; Python's
//     unbounded ints beyond it are an error), any other a double, correctly
//     rounded; strings with their escapes, a surrogate pair joined, control
//     characters refused (strict=True); an object's keys in order, a key
//     given twice keeping its first place and its last value (a dict).
//   * dump: json.dumps(v, indent=..., separators=..., ensure_ascii=True) —
//     a float as repr(): the shortest digits that read back to the same
//     double (std::to_chars, C++17 §[utility.to.chars]: the shortest, the
//     nearest of those; David Gay's mode 0, which CPython uses), fixed
//     notation for decimal exponents -4 < e <= 16 with ".0" where there is no
//     fraction, else d.ddde±XX; NaN, Infinity, -Infinity; strings with \",
//     \\, \n, \r, \t, \b, \f, other control characters and every non-ASCII
//     character as \uXXXX (UTF-16, a surrogate pair above U+FFFF).
//   Reading a number goes through std::strtod in the "C" numeric locale (the
//   library never sets another); it is correctly rounded on every platform
//   the library builds on (glibc, the UCRT, Apple's libc).
// ============================================================================

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace mootation::json {

struct Value {
    enum class Kind { Null, Bool, Int, Float, String, Array, Object };
    Kind kind = Kind::Null;
    bool b = false;
    std::int64_t i = 0;
    double f = 0.0;
    std::string s;                                        // UTF-8
    std::vector<Value> items;                             // an array's, or an object's values
    std::vector<std::string> keys;                        // an object's keys, in order (one a value)

    static Value null() { return {}; }
    static Value boolean(bool v) { Value x; x.kind = Kind::Bool; x.b = v; return x; }
    static Value integer(std::int64_t v) { Value x; x.kind = Kind::Int; x.i = v; return x; }
    static Value number(double v) { Value x; x.kind = Kind::Float; x.f = v; return x; }
    static Value string(std::string v) { Value x; x.kind = Kind::String; x.s = std::move(v); return x; }
    static Value array() { Value x; x.kind = Kind::Array; return x; }
    static Value object() { Value x; x.kind = Kind::Object; return x; }

    // an object's member by key, or nullptr
    const Value* find(const std::string& key) const
    {
        for (std::size_t k = 0; k < keys.size(); ++k)
            if (keys[k] == key) return &items[k];
        return nullptr;
    }
    // an object's member set as a dict does: its place kept where it was
    void set(const std::string& key, Value v)
    {
        for (std::size_t k = 0; k < keys.size(); ++k)
            if (keys[k] == key) {
                items[k] = std::move(v);
                return;
            }
        keys.push_back(key);
        items.push_back(std::move(v));
    }
};

// ── a float as Python's repr ────────────────────────────────────────────────

inline std::string float_repr(double x)
{
    if (std::isnan(x)) return "NaN";
    if (std::isinf(x)) return x > 0 ? "Infinity" : "-Infinity";
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof buf, x, std::chars_format::scientific);
    if (r.ec != std::errc()) throw std::runtime_error("float_repr: to_chars failed");
    // buf: [-]d[.ddd]e[+-]XX — the shortest digits and the decimal exponent
    std::string t(buf, r.ptr);
    std::string sign;
    if (t[0] == '-') {
        sign = "-";
        t.erase(0, 1);
    }
    const std::size_t epos = t.find('e');
    std::string digits = t.substr(0, epos);
    digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
    const int exp10 = std::stoi(t.substr(epos + 1));
    const int decpt = exp10 + 1;                         // value = 0.digits x 10^decpt
    const int n = static_cast<int>(digits.size());
    std::string out = sign;
    if (decpt > -4 && decpt <= 16) {                     // CPython's float_repr_style "short"
        if (decpt <= 0) {
            out += "0." + std::string(static_cast<std::size_t>(-decpt), '0') + digits;
        } else if (decpt >= n) {
            out += digits + std::string(static_cast<std::size_t>(decpt - n), '0') + ".0";
        } else {
            out += digits.substr(0, static_cast<std::size_t>(decpt)) + "." +
                   digits.substr(static_cast<std::size_t>(decpt));
        }
    } else {
        out += digits.substr(0, 1);
        if (n > 1) out += "." + digits.substr(1);
        const int e = decpt - 1;
        out += e < 0 ? "e-" : "e+";
        const int a = e < 0 ? -e : e;
        if (a < 10) out += "0";
        out += std::to_string(a);
    }
    return out;
}

// ── dump ────────────────────────────────────────────────────────────────────

namespace detail {

inline void put_hex4(std::string& out, unsigned v)
{
    static const char* hex = "0123456789abcdef";
    out += "\\u";
    for (int s = 12; s >= 0; s -= 4) out += hex[(v >> s) & 0xF];
}

// a UTF-8 string as json.dumps(ensure_ascii=True) writes it
inline void put_string(std::string& out, const std::string& s)
{
    out += '"';
    for (std::size_t i = 0; i < s.size();) {
        const auto c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) put_hex4(out, c);
                else out += static_cast<char>(c);
            }
            ++i;
            continue;
        }
        const int n = c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
        if (c < 0xC2 || c > 0xF4 || i + static_cast<std::size_t>(n) > s.size())
            throw std::invalid_argument("json: a string that is not UTF-8");
        std::uint32_t cp = c & (0x7Fu >> n);
        for (int k = 1; k < n; ++k) {
            const auto t = static_cast<unsigned char>(s[i + static_cast<std::size_t>(k)]);
            if ((t & 0xC0) != 0x80) throw std::invalid_argument("json: a string that is not UTF-8");
            cp = (cp << 6) | (t & 0x3Fu);
        }
        i += static_cast<std::size_t>(n);
        if (cp >= 0x10000) {
            cp -= 0x10000;
            put_hex4(out, 0xD800 + (cp >> 10));
            put_hex4(out, 0xDC00 + (cp & 0x3FF));
        } else {
            put_hex4(out, cp);
        }
    }
    out += '"';
}

inline void dump_into(std::string& out, const Value& v, int indent, int depth, const std::string& item_sep,
                      const std::string& key_sep)
{
    auto newline = [&](int d) {
        out += '\n';
        out.append(static_cast<std::size_t>(indent * d), ' ');
    };
    switch (v.kind) {
    case Value::Kind::Null: out += "null"; return;
    case Value::Kind::Bool: out += v.b ? "true" : "false"; return;
    case Value::Kind::Int: out += std::to_string(v.i); return;
    case Value::Kind::Float: out += float_repr(v.f); return;
    case Value::Kind::String: put_string(out, v.s); return;
    case Value::Kind::Array:
        if (v.items.empty()) {
            out += "[]";
            return;
        }
        out += '[';
        for (std::size_t k = 0; k < v.items.size(); ++k) {
            if (k) out += item_sep;
            if (indent >= 0) newline(depth + 1);
            dump_into(out, v.items[k], indent, depth + 1, item_sep, key_sep);
        }
        if (indent >= 0) newline(depth);
        out += ']';
        return;
    case Value::Kind::Object:
        if (v.keys.empty()) {
            out += "{}";
            return;
        }
        out += '{';
        for (std::size_t k = 0; k < v.keys.size(); ++k) {
            if (k) out += item_sep;
            if (indent >= 0) newline(depth + 1);
            put_string(out, v.keys[k]);
            out += key_sep;
            dump_into(out, v.items[k], indent, depth + 1, item_sep, key_sep);
        }
        if (indent >= 0) newline(depth);
        out += '}';
        return;
    }
}

}   // namespace detail

// json.dumps(v, indent=indent) (indent < 0: None), separators as given or
// Python's default for the indent: (", ", ": ") without, (",", ": ") with.
inline std::string dump(const Value& v, int indent = -1, const char* item_sep = nullptr, const char* key_sep = nullptr)
{
    std::string out;
    detail::dump_into(out, v, indent, 0, item_sep ? item_sep : (indent >= 0 ? "," : ", "), key_sep ? key_sep : ": ");
    return out;
}

// ── parse ───────────────────────────────────────────────────────────────────

namespace detail {

struct Parser {
    const std::string& t;
    std::size_t at = 0;

    [[noreturn]] void fail(const std::string& what) const
    {
        throw std::invalid_argument("json: " + what + " at char " + std::to_string(at));
    }
    void space()
    {
        while (at < t.size() && (t[at] == ' ' || t[at] == '\t' || t[at] == '\n' || t[at] == '\r')) ++at;
    }
    bool take(const char* word)
    {
        const std::size_t n = std::strlen(word);
        if (t.compare(at, n, word) == 0) {
            at += n;
            return true;
        }
        return false;
    }
    unsigned hex4()
    {
        if (at + 4 > t.size()) fail("a \\u escape cut short");
        unsigned v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = t[at++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<unsigned>(c - 'A' + 10);
            else fail("a bad \\u escape");
        }
        return v;
    }
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
    std::string string()
    {
        ++at;                                             // the opening quote
        std::string out;
        while (true) {
            if (at >= t.size()) fail("an unterminated string");
            const auto c = static_cast<unsigned char>(t[at]);
            if (c == '"') {
                ++at;
                return out;
            }
            if (c < 0x20) fail("a control character in a string");
            if (c != '\\') {
                out += static_cast<char>(c);
                ++at;
                continue;
            }
            ++at;
            if (at >= t.size()) fail("an unterminated string");
            const char e = t[at++];
            switch (e) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u': {
                std::uint32_t cp = hex4();
                if (cp >= 0xD800 && cp < 0xDC00 && at + 6 <= t.size() && t[at] == '\\' && t[at + 1] == 'u') {
                    const std::size_t back = at;
                    at += 2;
                    const unsigned lo = hex4();
                    if (lo >= 0xDC00 && lo < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    else at = back;
                }
                if (cp >= 0xD800 && cp < 0xE000) fail("a lone surrogate (not UTF-8)");
                put_utf8(out, cp);
                break;
            }
            default: fail("a bad escape");
            }
        }
    }
    Value number()
    {
        const std::size_t start = at;
        if (t[at] == '-') ++at;
        if (at < t.size() && t[at] == '0') {
            ++at;
        } else if (at < t.size() && t[at] >= '1' && t[at] <= '9') {
            while (at < t.size() && t[at] >= '0' && t[at] <= '9') ++at;
        } else {
            fail("a bad number");
        }
        bool real = false;
        if (at < t.size() && t[at] == '.') {
            const std::size_t d = ++at;
            while (at < t.size() && t[at] >= '0' && t[at] <= '9') ++at;
            if (at == d) fail("a bad number");
            real = true;
        }
        if (at < t.size() && (t[at] == 'e' || t[at] == 'E')) {
            ++at;
            if (at < t.size() && (t[at] == '+' || t[at] == '-')) ++at;
            const std::size_t d = at;
            while (at < t.size() && t[at] >= '0' && t[at] <= '9') ++at;
            if (at == d) fail("a bad number");
            real = true;
        }
        const std::string s = t.substr(start, at - start);
        if (!real) {
            std::int64_t v = 0;
            const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
            if (r.ec != std::errc() || r.ptr != s.data() + s.size()) fail("an integer beyond int64");
            return Value::integer(v);
        }
        char* end = nullptr;
        const double v = std::strtod(s.c_str(), &end);   // correctly rounded; HUGE_VAL past the doubles, as float()
        if (end != s.c_str() + s.size()) fail("a bad number");
        return Value::number(v);
    }
    Value value(int depth)
    {
        if (depth > 1000) fail("nested too deep");
        space();
        if (at >= t.size()) fail("expecting a value");
        const char c = t[at];
        if (c == '{') {
            ++at;
            Value o = Value::object();
            space();
            if (at < t.size() && t[at] == '}') {
                ++at;
                return o;
            }
            while (true) {
                space();
                if (at >= t.size() || t[at] != '"') fail("expecting a key");
                std::string k = string();
                space();
                if (at >= t.size() || t[at] != ':') fail("expecting ':'");
                ++at;
                o.set(k, value(depth + 1));
                space();
                if (at < t.size() && t[at] == ',') {
                    ++at;
                    continue;
                }
                if (at < t.size() && t[at] == '}') {
                    ++at;
                    return o;
                }
                fail("expecting ',' or '}'");
            }
        }
        if (c == '[') {
            ++at;
            Value a = Value::array();
            space();
            if (at < t.size() && t[at] == ']') {
                ++at;
                return a;
            }
            while (true) {
                a.items.push_back(value(depth + 1));
                space();
                if (at < t.size() && t[at] == ',') {
                    ++at;
                    continue;
                }
                if (at < t.size() && t[at] == ']') {
                    ++at;
                    return a;
                }
                fail("expecting ',' or ']'");
            }
        }
        if (c == '"') return Value::string(string());
        if (take("null")) return Value::null();
        if (take("true")) return Value::boolean(true);
        if (take("false")) return Value::boolean(false);
        if (take("NaN")) return Value::number(std::numeric_limits<double>::quiet_NaN());
        if (take("Infinity")) return Value::number(std::numeric_limits<double>::infinity());
        if (take("-Infinity")) return Value::number(-std::numeric_limits<double>::infinity());
        return number();
    }
};

}   // namespace detail

// json.loads(text); invalid_argument where it is not JSON as Python reads it.
inline Value parse(const std::string& text)
{
    detail::Parser p{text};
    Value v = p.value(0);
    p.space();
    if (p.at != text.size()) p.fail("extra data");
    return v;
}

}   // namespace mootation::json
