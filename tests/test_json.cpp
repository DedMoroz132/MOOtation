// SPDX-License-Identifier: Apache-2.0
// JSON as Python's json module reads and writes it (include/mootation/json.hpp):
// floats as repr() — the shortest digits, fixed or exponent as CPython puts
// them —, strings escaped as ensure_ascii does, json.dumps's layouts with and
// without indent, json.loads's numbers, escapes, NaN and the infinities, a
// key given twice; malformed text is an error. The values below are
// Python 3.13's (repr, json.dumps).

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

#include <mootation/json.hpp>

namespace js = mootation::json;

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

static void same(const std::string& got, const std::string& want)
{
    if (got != want) {
        std::printf("FAIL: got  %s\n      want %s\n", got.c_str(), want.c_str());
        ++failures;
    }
}

static bool refused(const std::string& text)
{
    try {
        js::parse(text);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

int main()
{
    // repr(float)
    same(js::float_repr(0.0), "0.0");
    same(js::float_repr(-0.0), "-0.0");
    same(js::float_repr(1.0), "1.0");
    same(js::float_repr(0.1), "0.1");
    same(js::float_repr(0.1 + 0.2), "0.30000000000000004");
    same(js::float_repr(1e16), "1e+16");
    same(js::float_repr(1e15), "1000000000000000.0");
    same(js::float_repr(123456789012345678.0), "1.2345678901234568e+17");
    same(js::float_repr(0.0001), "0.0001");
    same(js::float_repr(0.00001), "1e-05");
    same(js::float_repr(1.5e-7), "1.5e-07");
    same(js::float_repr(-2.5e100), "-2.5e+100");
    same(js::float_repr(5e-324), "5e-324");
    same(js::float_repr(1.7976931348623157e308), "1.7976931348623157e+308");
    same(js::float_repr(2.2250738585072014e-308), "2.2250738585072014e-308");
    same(js::float_repr(123.456), "123.456");
    same(js::float_repr(9007199254740993.0), "9007199254740992.0");
    same(js::float_repr(std::numeric_limits<double>::quiet_NaN()), "NaN");
    same(js::float_repr(std::numeric_limits<double>::infinity()), "Infinity");
    same(js::float_repr(-std::numeric_limits<double>::infinity()), "-Infinity");

    // json.dumps
    const std::string text =
        R"({"b": [1, 2.5, -0.0, true, false, null, "x\"\\\n\t\u00e9\u20ac\ud83d\ude00\u0001"], "a": {}, "c": [], )"
        R"("d": {"e": 1e-05, "f": NaN, "g": -Infinity}, "b": [3]})";
    const js::Value v = js::parse(text);
    // "b" given twice: its first place, its last value
    same(js::dump(v), R"({"b": [3], "a": {}, "c": [], "d": {"e": 1e-05, "f": NaN, "g": -Infinity}})");
    same(js::dump(v, 1), "{\n \"b\": [\n  3\n ],\n \"a\": {},\n \"c\": [],\n \"d\": {\n  \"e\": 1e-05,\n"
                         "  \"f\": NaN,\n  \"g\": -Infinity\n }\n}");
    same(js::dump(v, -1, ",", ":"), R"({"b":[3],"a":{},"c":[],"d":{"e":1e-05,"f":NaN,"g":-Infinity}})");
    const js::Value s = js::parse(R"(["x\"\\\n\t\u00e9\u20ac\ud83d\ude00\u0001\/", "\u00E9"])");
    same(js::dump(s), R"(["x\"\\\n\t\u00e9\u20ac\ud83d\ude00\u0001/", "\u00e9"])");
    CHECK(s.items[0].s == "x\"\\\n\t\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80\x01/");

    // numbers
    const js::Value n = js::parse("[0, -0, 12, -9223372036854775808, 1E2, 1.0, 2e-1, 1e400, 0.1]");
    CHECK(n.items[0].kind == js::Value::Kind::Int && n.items[1].kind == js::Value::Kind::Int &&
          n.items[1].i == 0 && n.items[3].i == std::numeric_limits<std::int64_t>::min());
    CHECK(n.items[4].kind == js::Value::Kind::Float && n.items[4].f == 100.0 && n.items[6].f == 0.2);
    CHECK(std::isinf(n.items[7].f));
    same(js::dump(n), "[0, 0, 12, -9223372036854775808, 100.0, 1.0, 0.2, Infinity, 0.1]");

    // what json.loads refuses
    CHECK(refused("[1,]"));
    CHECK(refused("{\"a\" 1}"));
    CHECK(refused("01"));
    CHECK(refused("1."));
    CHECK(refused(".5"));
    CHECK(refused("[1] x"));
    CHECK(refused("\"a\x01\""));            // a control character
    CHECK(refused("\"\\x\""));
    CHECK(refused("\"\\ud800\""));          // a lone surrogate
    CHECK(refused("99999999999999999999"));  // past int64 (Python: an int)
    CHECK(refused("nan"));
    CHECK(refused(""));

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("json: Python's json, byte for byte\n");
    return 0;
}
