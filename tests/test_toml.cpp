// SPDX-License-Identifier: Apache-2.0
// TOML 1.0 as Python's tomllib reads it (include/mootation/toml.hpp): a
// campaign-like file, every kind of string, number and key, tables, arrays of
// tables, inline tables, dates kept as text; what tomllib refuses is refused,
// with the line and column. The trees below are json.dumps of tomllib.loads
// (Python 3.13).

#include <cstdio>
#include <stdexcept>
#include <string>

#include <mootation/toml.hpp>

namespace tml = mootation::toml;
namespace js = mootation::json;

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

static void same(const std::string& text, const std::string& want)
{
    std::string got;
    try {
        got = js::dump(tml::parse(text));
    } catch (const std::exception& e) {
        got = std::string("error: ") + e.what();
    }
    if (got != want) {
        std::printf("FAIL: %s\n  got  %s\n  want %s\n", text.c_str(), got.c_str(), want.c_str());
        ++failures;
    }
}

static std::string refused(const std::string& text)
{
    try {
        tml::parse(text);
    } catch (const tml::Error& e) {
        return e.what();
    }
    std::printf("FAIL: accepted %s\n", text.c_str());
    ++failures;
    return "";
}

int main()
{
    same("[run]\nname = \"atlas\"\n\n[benchmarks]\nproblems = [\"ZDT1\", \"DTLZ2_3D\"]\nruns = 2\n\n"
         "[[algorithms]]\nname = \"nsga2\"\npop = 0\n\n[[algorithms]]\nname = \"moead_de\"\n",
         R"({"run": {"name": "atlas"}, "benchmarks": {"problems": ["ZDT1", "DTLZ2_3D"], "runs": 2}, )"
         R"("algorithms": [{"name": "nsga2", "pop": 0}, {"name": "moead_de"}]})");
    same("a.b.c = 1\na.b.d = 2\n[x.y]\nz = 3\n[x]\nw = 4\n",
         R"({"a": {"b": {"c": 1, "d": 2}}, "x": {"y": {"z": 3}, "w": 4}})");
    same("[[p]]\nn = 1\n[p.q]\nm = 2\n[[p]]\nn = 2\n", R"({"p": [{"n": 1, "q": {"m": 2}}, {"n": 2}]})");
    same("k = \"\"\"\nline1\\\n   line2\n\"\"\"\nl = '''raw\\n'''\n", R"({"k": "line1line2\n", "l": "raw\\n"})");
    same("u = \"\\u00e9\\U0001F600\\t\"\nq = \"\"\"two \"\" quotes\"\"\"\nq2 = \"\"\"\"\"five\"\"\"\"\"\n",
         R"({"u": "\u00e9\ud83d\ude00\t", "q": "two \"\" quotes", "q2": "\"\"five\"\""})");
    same("i = 0xDEAD_beef\no = 0o755\nb = 0b1101\nn = -0\np = +17\n",
         R"({"i": 3735928559, "o": 493, "b": 13, "n": 0, "p": 17})");
    same("f = 6.626e-34\ng = -0.0\nh = 1_000.5\ni = inf\nj = -inf\nk = nan\nl = 1e1_0\n",
         R"({"f": 6.626e-34, "g": -0.0, "h": 1000.5, "i": Infinity, "j": -Infinity, "k": NaN, "l": 10000000000.0})");
    same("big = 9223372036854775807\nsmall = -9223372036854775808\n",
         R"({"big": 9223372036854775807, "small": -9223372036854775808})");
    same("t = { a = { b = 1 } }\nx = [ [1, 2], [\"a\"], [] ]\n", R"({"t": {"a": {"b": 1}}, "x": [[1, 2], ["a"], []]})");
    same("# comment\n\n  key   =   \"v\"   # trailing\n[ spaced . table ]\n'q' = 1\n\"\" = 2\n",
         R"({"key": "v", "spaced": {"table": {"q": 1, "": 2}}})");
    same("a = [\n  1,\n  2, # c\n]\r\nb = 3\r\n", R"({"a": [1, 2], "b": 3})");
    same("d = 1979-05-27T07:32:00Z\ne = 1979-05-27\nf = 07:32:00\n",
         R"({"d": "1979-05-27T07:32:00Z", "e": "1979-05-27", "f": "07:32:00"})");
    CHECK(tml::parse("d = 1979-05-27\n").find("d")->datetime);

    // what tomllib refuses
    CHECK(refused("a = 1\na = 2\n").find("(at line 2, column") != std::string::npos);
    for (const char* bad : {"[t]\n[t]\n", "a = {x = 1}\na.y = 2\n", "a = [1]\n[[a]]\n", "[a]\nb = 1\n[a.b]\n",
                            "a.b = 1\n[a]\n", "a = 1 b = 2\n", "a = \"unterminated\n", "a = 01\n", "a = 1__0\n",
                            "a = _1\n", "a = 1.\n", "a = .5\n", "a = 0x\n", "a = {x = 1,}\n", "[[a]]\n[a]\n",
                            "a = \"\\x\"\n", "a = \"\\uD800\"\n", "key\n", "= 1\n", "[a.]\n", "a = [1,,2]\n",
                            "x = { a = 1 }\nx.b = 2\n", "[a]\nb.c = 1\n[a.b]\n", "a = '''x'''''''\n",
                            "a = 9223372036854775808\n"})
        refused(bad);

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("toml: tomllib's trees\n");
    return 0;
}
