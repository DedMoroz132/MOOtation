// SPDX-License-Identifier: Apache-2.0
// External problems (task 24, §24.7.1; include/mootation/campaign/external.hpp,
// external_rows.hpp) with the example folder tests/external_example: its
// rows read beside the repository's registry; every mistake of a row refused
// with its row and reason; a campaign of the C++ engine on them — the row
// that is MGH's problem 1 gives the bits of MGH01_n02_m02_x1_1D's runs (lm,
// lbfgsb), meta.json names the family, its version and the row's checksum;
// four threads and one, a strategy's one thread and three: the same records.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <mootation/campaign/engine.hpp>
#include <mootation/campaign/external_rows.hpp>

namespace cg = mootation::campaign;
namespace js = mootation::json;
namespace st = mootation::store;

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

static std::string read_text(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

static void set_env(const char* name, const std::string& value)
{
#if defined(_WIN32)
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

// key -> block name -> its bytes, every record of a store's segments
using Records = std::map<std::string, std::map<std::string, std::string>>;

static Records records(const std::filesystem::path& store)
{
    Records out;
    for (const auto& f : std::filesystem::directory_iterator(store)) {
        if (f.path().extension() != ".raw") continue;
        const std::string raw = read_text(f.path());
        const auto* buf = reinterpret_cast<const std::uint8_t*>(raw.data());
        for (std::size_t at = 0; at < raw.size();) {
            const st::RecordAt r = st::parse_record(buf, raw.size(), at);
            if (!r.ok) break;
            for (const auto& b : st::decode_run(buf + r.body, r.body_size))
                out[r.key][b.name] = std::string(b.data.begin(), b.data.end());
            at = r.end;
        }
    }
    return out;
}

// the record's key with its problem's name replaced by `as`
static std::string renamed(const std::string& key, const std::string& from, const std::string& as)
{
    std::string k = key;
    const auto at = k.find(from);
    if (at != std::string::npos) k.replace(at, from.size(), as);
    return k;
}

// the reader's refusal of a row (its message), or "" where it reads it
static std::string refusal(const cg::ProblemRegistry& base, const cg::ExternalProblems& reg, const std::string& text)
{
    cg::ProblemRegistry preg = base;
    try {
        cg::add_external_rows(preg, text, "rows.toml", reg);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

static bool says(const std::string& message, const std::string& part)
{
    const bool yes = message.find(part) != std::string::npos;
    if (!yes) std::printf("  the message: %s\n  lacks: %s\n", message.c_str(), part.c_str());
    return yes;
}

// a problem of the given sizes, f = 0
struct Sized : mootation::suite::Native {
    Sized(int n, int m, int c)
    {
        n_var = n;
        n_obj = m;
        n_con = c;
    }
    void evaluate(const double*, double* f, double*) const override { f[0] = 0.0; }
};

static int body();

int main()
{
    try {
        return body();
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}

static int body()
{
    const std::string example = std::string(MOOTATION_TEST_DATA) + "/../external_example/";
    const cg::ProblemRegistry base = cg::load_registry(std::string(MOOTATION_REPO_DATA) + "/registry.json");
    cg::ExternalProblems reg;
    mootation_external_problems(reg);
    CHECK(reg.families().size() == 1 && reg.find("extended_rosenbrock_example") != nullptr);
    CHECK(reg.find("extended_rosenbrock_example")->version == "1");

    // ── the rows, beside the registry's ─────────────────────────────────────
    cg::ProblemRegistry preg = base;
    cg::add_external(preg, example + "problems.toml", reg);
    CHECK(preg.problems.size() == base.problems.size() + 2);
    {
        const cg::RegistryProblem& p = preg.problems.at("ext_rosenbrock_n02_x1");
        CHECK(p.n_var == 2 && p.n_obj == 1 && p.family == "extended_rosenbrock_example");
        CHECK(p.f_opt && *p.f_opt == 0.0 && p.f_opt_kind == "exact" && p.targets == "relative");
        CHECK(p.x0 == std::vector<double>({-1.2, 1.0}) && p.pop_size == 1 && p.n_gen == 1000);
        CHECK(p.bounds.size() == 2 && std::isinf(p.bounds[0].first) && std::isinf(p.bounds[1].second));
        CHECK(!cg::engine_refusal(p));
        CHECK(p.external_meta.find("family")->s == "extended_rosenbrock_example");
        CHECK(p.external_meta.find("version")->s == "1" && p.external_meta.find("row_crc32c")->s.size() == 8);
        // the same values, residuals and Jacobian as MGH's problem 1, bit for bit
        const auto ext = cg::make_native(p);
        const auto mgh = cg::make_native(base.problems.at("MGH01_n02_m02_x1_1D"));
        for (const auto& x : std::vector<std::vector<double>>{{-1.2, 1.0}, {0.3, -2.7}, {1e3, 1e-3}}) {
            double fe = 0.0, fm = 0.0;
            ext->evaluate(x.data(), &fe, nullptr);
            mgh->evaluate(x.data(), &fm, nullptr);
            std::vector<double> re(2), rm(2), Je(4), Jm(4);
            ext->residuals(x.data(), re.data(), Je.data());
            mgh->residuals(x.data(), rm.data(), Jm.data());
            CHECK(fe == fm && re == rm && Je == Jm);
        }
        const cg::RegistryProblem& q = preg.problems.at("ext_rosenbrock_n04_box");
        const std::vector<std::pair<double, double>> box(4, {-5.0, 5.0});
        CHECK(q.bounds == box && q.targets == "absolute");
        CHECK(q.x0.empty() && q.pop_size == 20 && q.n_gen == 50);
        // another row's text, another checksum
        CHECK(q.external_meta.find("row_crc32c")->s != p.external_meta.find("row_crc32c")->s);
    }

    // ── every mistake refused, with its row and reason ──────────────────────
    const std::string row = "[[problem]]\nfamily = \"extended_rosenbrock_example\"\nn_var = 2\n";
    CHECK(refusal(base, reg, "[[problem]]\nname = \"a\"\n" + row.substr(12) + "lo = -1\nhi = 1\n") == "");
    CHECK(says(refusal(base, reg, "[[problem]]\nname = \"MGH01_n02_m02_x1_1D\"\n" + row.substr(12)),
               "the name is taken (by the problem registry)"));
    CHECK(says(refusal(base, reg, "[[problem]]\nname = \"a\"\n" + row.substr(12) + "[[problem]]\nname = \"a\"\n" +
                                      row.substr(12)),
               "[[problem]] 2 (a): the name is taken (by another external row)"));
    CHECK(says(refusal(base, reg, "[[problem]]\nname = \"a b\"\n" + row.substr(12)), "a name of letters"));
    CHECK(says(refusal(base, reg, "[[problem]]\nname = \"a\"\nfamily = \"nope\"\nn_var = 2\n"),
               "the family 'nope' is not registered (registered: extended_rosenbrock_example)"));
    CHECK(says(refusal(base, reg, "[[problem]]\nname = \"a\"\nfamily = \"MGH\"\nn_var = 2\n"),
               "the family 'MGH' is one of the problem registry's"));
    CHECK(says(refusal(base, reg, "[[problem]]\nname = \"a\"\n" + row.substr(12) + "n_vars = 3\n"),
               "unknown key 'n_vars'"));
    CHECK(says(refusal(base, reg, "[[problem]]\nname = \"a\"\nfamily = \"extended_rosenbrock_example\"\n"),
               "n_var is required"));
    CHECK(says(refusal(base, reg, "[[problem]]\nname = \"a\"\n" + row.substr(12) + "n_obj = 2\n"),
               "one objective only (EXT-3"));
    CHECK(says(refusal(base, reg, "[[problem]]\nname = \"a\"\nfamily = \"extended_rosenbrock_example\"\nn_var = 3\n"),
               "the family's make refused the row: n_var even"));
    CHECK(says(refusal(base, reg, "[[problem]]\nname = \"a\"\n" + row.substr(12) + "lo = 1\nhi = 1\n"),
               "lo < hi in every variable (variable 0)"));
    CHECK(says(refusal(base, reg, "[[problem]]\nname = \"a\"\n" + row.substr(12) + "lo = [1, 2, 3]\n"),
               "lo: a number, or a list of n_var = 2 numbers"));
    CHECK(says(refusal(base, reg, "[[problem]]\nname = \"a\"\n" + row.substr(12) + "lo = 0\nbounds = [[0, 1], [0, 1]]\n"),
               "not both"));
    CHECK(says(refusal(base, reg, "[[problem]]\nname = \"a\"\n" + row.substr(12) + "f_opt = 0\n"),
               "f_opt_kind is required"));
    CHECK(says(refusal(base, reg, "[[problem]]\nname = \"a\"\n" + row.substr(12) + "f_opt = 0\nf_opt_kind = \"best\"\n"),
               "f_opt_kind: exact, computed"));
    CHECK(says(refusal(base, reg, "[[problem]]\nname = \"a\"\n" + row.substr(12) + "targets = \"relative\"\n"),
               "targets = relative needs x0"));
    CHECK(says(refusal(base, reg, "[[problem]]\nname = \"a\"\n" + row.substr(12) + "x0 = [1]\n"),
               "x0: a number, or a list of n_var = 2 numbers"));
    CHECK(says(refusal(base, reg, "[problem]\nname = \"a\"\n"), "an array of tables"));
    CHECK(says(refusal(base, reg, "[other]\nname = \"a\"\n"), "'other': only [[problem]] tables are read"));
    CHECK(says(refusal(base, reg, "[[problem]]\nname = \"a\n"), "rows.toml: "));
    {
        cg::ExternalProblems twice;
        mootation_external_problems(twice);
        std::string why;
        try {
            mootation_external_problems(twice);
        } catch (const std::invalid_argument& e) {
            why = e.what();
        }
        CHECK(says(why, "the family 'extended_rosenbrock_example' is registered twice"));
        // a family whose problems have two objectives, or constraints: refused at the row
        cg::ExternalProblems wide;
        wide.add("two", [](const js::Value&, int n) { return std::make_shared<Sized>(n, 2, 0); });
        wide.add("cons", [](const js::Value&, int n) { return std::make_shared<Sized>(n, 1, 1); });
        wide.add("short", [](const js::Value&, int n) { return std::make_shared<Sized>(n - 1, 1, 0); });
        CHECK(says(refusal(base, wide, "[[problem]]\nname = \"a\"\nfamily = \"two\"\nn_var = 2\n"),
                   "the problem made has 2 objectives"));
        CHECK(says(refusal(base, wide, "[[problem]]\nname = \"a\"\nfamily = \"cons\"\nn_var = 2\n"),
                   "constraints: none at one objective (EXT-4)"));
        CHECK(says(refusal(base, wide, "[[problem]]\nname = \"a\"\nfamily = \"short\"\nn_var = 2\n"),
                   "the problem made has 1 variables, the row says n_var = 2"));
    }

    // ── a campaign of the C++ engine ────────────────────────────────────────
    const auto tmp = std::filesystem::temp_directory_path() / ("mootation_test_external_" + std::to_string(std::rand()));
    std::filesystem::remove_all(tmp);
    set_env("MOOTATION_REFS_DIR", (tmp / "refs").string());
    // the registry with the external rows, and a copy in which MGH01 is made
    // from its own row of the suite as an external problem is, so that the
    // engine runs it alone too (no reference of the campaign to wait for):
    // the bits to compare the external row with
    cg::ProblemRegistry beside = preg;
    {
        cg::RegistryProblem& m = beside.problems.at("MGH01_n02_m02_x1_1D");
        const cg::RegistryProblem own = m;
        m.external = [own] { return cg::make_native(own); };
    }
    const cg::ProblemRegistry* used = &preg;
    auto run = [&](const std::string& dir, const std::string& toml, int workers) {
        std::filesystem::create_directories(tmp / dir);
        const auto source = tmp / dir / "campaign.toml";
        std::ofstream(source, std::ios::binary) << toml;
        cg::EngineOptions opt;
        opt.workers = workers;
        opt.pid = 7;
        opt.host = "test";
        opt.process.seconds = [] { return 0.0; };
        return cg::run_campaign(toml, source, *used, opt);
    };
    const std::string head = "[run]\nname = \"ext\"\n[problem]\nkind = \"builtin\"\n[benchmarks]\nruns = 2\n";
    const std::string local = head +
                              "problems = [\"MGH01_n02_m02_x1_1D\", \"ext_rosenbrock_n02_x1\"]\n"
                              "[campaign]\nout = \"res/{name}\"\nbudget_fe = 400\nstorage = \"store\"\n"
                              "[[algorithms]]\nname = \"lm\"\npop = 0\ngens = 0\n"
                              "[[algorithms]]\nname = \"lbfgsb\"\npop = 0\ngens = 0\n";
    // the registry's MGH01 waits for its campaign reference, which the Python
    // engine builds (as before); the external row runs
    const cg::EngineCounts alone = run("alone", local, 2);
    CHECK(alone.total == 8 && alone.done == 4 && alone.left.size() == 1 && alone.left.begin()->second == 4);
    CHECK(alone.left.begin()->first.find("a reference built for the campaign") != std::string::npos);
    used = &beside;
    const cg::EngineCounts four = run("four", local, 4);
    CHECK(four.total == 8 && four.done == 8 && four.failed == 0 && four.left.empty());
    const cg::EngineCounts one = run("one", local, 1);
    CHECK(one.done == 8);
    const Records a = records(tmp / "four" / "res" / "ext" / "store");
    const Records b = records(tmp / "one" / "res" / "ext" / "store");
    CHECK(a.size() == 8);
    std::size_t same = 0;
    for (const auto& [key, blocks] : a) {
        // one thread and four: the same records (but the run's own seconds and times)
        for (const auto& [name, data] : blocks)
            if (name != "meta.json") CHECK(b.at(key).at(name) == data);
        if (key.find("ext_rosenbrock_n02_x1") == std::string::npos) continue;
        // the external row: MGH01's records, bit for bit
        const auto& mgh = a.at(renamed(key, "ext_rosenbrock_n02_x1", "MGH01_n02_m02_x1_1D"));
        CHECK(blocks.size() == mgh.size());
        for (const auto& [name, data] : blocks) {
            if (name == "meta.json") continue;
            CHECK(mgh.at(name) == data);
            if (mgh.at(name) == data) ++same;
        }
        const js::Value meta = js::parse(blocks.at("meta.json"));
        const js::Value base_meta = js::parse(mgh.at("meta.json"));
        CHECK(meta.find("status")->s == "done");
        CHECK(meta.find("external") && meta.find("external")->find("family")->s == "extended_rosenbrock_example");
        CHECK(base_meta.find("external")->kind == js::Value::Kind::Null);
        CHECK(js::dump(*meta.find("so")) == js::dump(*base_meta.find("so")));
        CHECK(meta.find("fe")->i == base_meta.find("fe")->i);
    }
    CHECK(same >= 4);

    // a box, a population method and a strategy over threads: one thread and three the same
    used = &preg;
    // [campaign] success = "user" (task 24, §24.1.5): the problem's own measure in
    // the run, its calls and the evaluation of the success in meta.json; a
    // problem without one refused
    {
        cg::ProblemRegistry withs = preg;
        cg::add_external_rows(withs,
                              "[[problem]]\nname = \"ext_rosenbrock_n02_success\"\nfamily = "
                              "\"extended_rosenbrock_example\"\nn_var = 2\nx0 = [-1.2, 1.0]\nf_opt = 0.0\n"
                              "f_opt_kind = \"exact\"\ntargets = \"relative\"\npop_size = 1\nn_gen = 1000\n"
                              "[problem.params]\nsuccess_tol = 1e-3\npost_at = 1e-4\n",
                              "success.toml", reg);
        used = &withs;
        const std::string s = head + "problems = [\"ext_rosenbrock_n02_success\", \"ext_rosenbrock_n02_x1\"]\n"
                                     "[campaign]\nout = \"res/{name}\"\nbudget_fe = 400\nstorage = \"store\"\n"
                                     "success = \"user\"\n[[algorithms]]\nname = \"lm\"\npop = 0\ngens = 0\n";
        const cg::EngineCounts c = run("success", s, 2);
        CHECK(c.total == 4 && c.done == 2 && c.failed == 2);
        const Records r = records(tmp / "success" / "res" / "ext" / "store");
        std::size_t seen = 0;
        for (const auto& [key, blocks] : r) {
            const js::Value meta = js::parse(blocks.at("meta.json"));
            if (key.find("ext_rosenbrock_n02_success") != std::string::npos) {
                const js::Value* u = meta.find("so") ? meta.find("so")->find("success") : nullptr;
                CHECK(u && u->find("calls")->i >= 1 && u->find("at")->kind == js::Value::Kind::Int &&
                      u->find("at")->i == meta.find("fe")->i);
                ++seen;
            } else {
                CHECK(meta.find("status")->s == "failed" &&
                      meta.find("error")->s.find("no measure of success of its own") != std::string::npos);
            }
        }
        CHECK(seen == 2);
        used = &preg;
    }
    // a job of an external problem the engine would leave to the Python
    // engine (here: the learned coordinates' diagnostics): the campaign
    // refused before anything runs, with the text run/config.py gives
    CHECK(cg::external_refusal("x") ==
          "external problem 'x': the C++ engine alone runs it (mootation run <config>; task 24, section 24.7.1), "
          "the Python engine does not know it");
    {
        // run/config.py's external_refusal says the same, in its two f-strings
        const std::string py = read_text(std::string(MOOTATION_REPO_DATA) + "/../python/mootation/run/config.py");
        CHECK(says(py, "return (f\"external problem '{name}': the C++ engine alone runs it (mootation run <config>; \"\n"
                       "            f\"task 24, section 24.7.1), the Python engine does not know it\")"));
    }
    {
        std::string why;
        try {
            run("refused", head + "problems = [\"ext_rosenbrock_n02_x1\"]\n[campaign]\nout = \"res/{name}\"\n"
                                  "budget_fe = 100\nstorage = \"store\"\ncoords_diag = true\n"
                                  "[[algorithms]]\nname = \"lm\"\npop = 0\ngens = 0\n", 1);
        } catch (const cg::ConfigError& e) {
            why = e.what();
        }
        CHECK(says(why, "benchmarks.problems: " + cg::external_refusal("ext_rosenbrock_n02_x1") +
                            " (lm: the learned coordinates' diagnostics)"));
        CHECK(!std::filesystem::exists(tmp / "refused" / "res" / "ext" / "store"));
    }
    const std::string boxed = head +
                              "problems = [\"ext_rosenbrock_n04_box\"]\n"
                              "[campaign]\nout = \"res/{name}\"\nbudget_fe = 600\nstorage = \"store\"\n"
                              "[[algorithms]]\nname = \"de\"\npop = 0\ngens = 0\n"
                              "[[algorithms]]\nname = \"multistart\"\npop = 0\ngens = 0\n"
                              "params = { n_starts = 4, threads = THREADS, inner = { solver = \"lbfgsb\" } }\n";
    auto with = [&](const std::string& threads) {
        std::string t = boxed;
        t.replace(t.find("THREADS"), 7, threads);
        return t;
    };
    const cg::EngineCounts t1 = run("t1", with("1"), 2);
    const cg::EngineCounts t3 = run("t3", with("3"), 2);
    CHECK(t1.done == 4 && t3.done == 4 && t1.failed == 0 && t3.failed == 0);
    const Records r1 = records(tmp / "t1" / "res" / "ext" / "store");
    const Records r3 = records(tmp / "t3" / "res" / "ext" / "store");
    CHECK(r1.size() == 4 && r3.size() == 4);
    for (const auto& [key, blocks] : r1) {
        const std::string other = renamed(key, "\"threads\": 1", "\"threads\": 3");
        const auto it = r3.find(other);
        CHECK(it != r3.end());
        if (it == r3.end()) continue;
        for (const auto& [name, data] : blocks)
            if (name != "meta.json") CHECK(it->second.at(name) == data);
        const js::Value meta = js::parse(blocks.at("meta.json"));
        CHECK(meta.find("status")->s == "done");
    }

    std::filesystem::remove_all(tmp);
    if (failures == 0) std::printf("test_external: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
