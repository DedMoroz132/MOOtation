// SPDX-License-Identifier: Apache-2.0
// A campaign's store looked after in C++ (include/mootation/campaign/store_tools.hpp),
// on the sample campaign (tests/data/job_sample) run by the engine: check finds
// every record and nothing wrong; compact makes one pack per family whose
// records read back as the segments' (meta.json from the pack's table), then
// removes the segments; compacting the packs again writes the same bytes;
// merging two copies of the store finds no run that differs.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <mootation/campaign/engine.hpp>
#include <mootation/campaign/store_tools.hpp>

namespace cg = mootation::campaign;

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

// every pack's bytes, by name
static std::map<std::string, std::string> packs(const std::filesystem::path& store)
{
    std::map<std::string, std::string> out;
    for (const auto& f : std::filesystem::directory_iterator(store))
        if (f.path().extension() == ".mpk") out[f.path().filename().string()] = read_text(f.path());
    return out;
}

static int body()
{
    const std::string data = std::string(MOOTATION_TEST_DATA) + "/job_sample/";
    set_env("MOOTATION_REFS_DIR", data + "refs");
    const cg::ProblemRegistry preg = cg::load_registry(data + "registry.json");
    std::string text = read_text(data + "campaign.toml");
    text.replace(text.find("[campaign]\n"), 11, "[campaign]\nstorage = \"store\"\n");
    const auto tmp = std::filesystem::temp_directory_path() / ("mootation_test_store_tools_" + std::to_string(std::rand()));
    std::filesystem::remove_all(tmp);
    std::filesystem::create_directories(tmp);
    const auto source = tmp / "campaign.toml";
    std::ofstream(source, std::ios::binary) << text;
    cg::EngineOptions opt;
    opt.workers = 2;
    opt.pid = 9;
    opt.host = "test";
    CHECK(cg::run_campaign(text, source, preg, opt).done == 4);
    const auto store = tmp / "results" / "sample" / "store";

    const cg::StoreCheck c = cg::check_store(store);
    CHECK(c.records == 4 && c.damaged.empty() && c.tails.empty() && c.duplicates == 0);

    // the segments' records, then the packs'
    cg::MetaCache cache;
    std::map<std::string, std::vector<mootation::store::Block>> before;
    for (const auto& [key, e] : cg::store_index(store)) before[key] = cg::entry_blocks(e, cache);
    auto family_of = [&preg](const std::string& problem) { return preg.problems.at(problem).family; };
    const cg::Compacted done = cg::compact_store(store, family_of, true, std::nullopt, 9);
    CHECK(done.packs.size() == 2 && done.packs.at("ZDT") == 2 && done.packs.at("DTLZ") == 2);
    CHECK(done.removed.size() == 2 && done.failed.empty());
    const auto first = packs(store);
    CHECK(first.size() == 2);
    cg::MetaCache after_cache;
    const auto after = cg::store_index(store);
    CHECK(after.size() == 4);
    for (const auto& [key, e] : after) {
        CHECK(e.file.extension() == ".mpk" && e.status == "done");
        CHECK(cg::same_blocks(cg::entry_blocks(e, after_cache), before.at(key)));
    }
    CHECK(cg::check_store(store).damaged.empty());

    // the packs compacted again: the same bytes
    cg::compact_store(store, family_of, false, std::nullopt, 9);
    CHECK(packs(store) == first);

    // a copy merged in: nothing differs, nothing to copy
    const auto copy = tmp / "copy";
    std::filesystem::copy(store, copy, std::filesystem::copy_options::recursive);
    const auto [runs, copied] = cg::merge_stores(store, {copy}, 9, "test");
    CHECK(runs == 4 && copied == 0);

    std::filesystem::remove_all(tmp);
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("store tools: check, compact, merge\n");
    return 0;
}

int main()
{
    try {
        return body();
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
