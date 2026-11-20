#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// A family's bank (family.hpp) exported into a run store (task 32, layer 2;
// the owner's decision 2: the bank's own file, with an export to run_store).
// The bank goes in as one run's record (run_store.hpp SegmentWriter), so that
// the store's tools (`mootation store export`, store.py) see it beside a
// campaign's runs:
//   "#format"       the store's;
//   "meta.json"     status "done", kind "family_bank", the total, the
//                   instances, the records, the pieces, used();
//   "bank.json"     the bank's file as save_bank writes it (family.hpp FAM-9);
//   "records.csv"   a table: one row a record — n, f, cv, source (0 run,
//                   1 transfer, 2 given), from (-1: none), piece, used.
// ============================================================================

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "../json.hpp"
#include "family.hpp"
#include "run_store.hpp"

namespace mootation::campaign::fam {

// the blocks of the bank's record (total: its total objective by the
// family's weights; instances: the family's number of instances)
inline std::vector<store::Block> bank_blocks(const Bank& B, double total, std::size_t instances)
{
    std::vector<store::Block> blocks{format_block()};
    json::Value meta = json::Value::object();
    meta.set("status", json::Value::string("done"));
    meta.set("kind", json::Value::string("family_bank"));
    meta.set("family_total", json::Value::number(total));
    meta.set("family_instances", json::Value::integer(static_cast<std::int64_t>(instances)));
    meta.set("family_records", json::Value::integer(static_cast<std::int64_t>(B.records.size())));
    meta.set("family_pieces", json::Value::integer(static_cast<std::int64_t>(B.pieces.size())));
    meta.set("family_used", json::Value::number(B.used));
    blocks.push_back(store::Block{"meta.json", "json", {}, bytes_of(text_mode(json::dump(meta, 1))), ""});
    blocks.push_back(store::Block{"bank.json", "json", {}, bytes_of(bank_text(B, total, instances, "")), ""});
    Table t;
    t.name = "records.csv";
    t.head = table_head(t.name, {"# mootation family bank v1: one record an instance; source 0 run, 1 transfer, 2 given"},
                        {"n", "f", "cv", "source", "from", "piece", "used"}, {0, 3, 4, 5});
    t.cols = 7;
    t.rows = B.records.size();
    for (const auto& [n, r] : B.records) {
        const double source = r.source == "run" ? 0.0 : r.source == "transfer" ? 1.0 : 2.0;
        t.values.insert(t.values.end(), {static_cast<double>(n), r.f, r.cv, source, static_cast<double>(r.from),
                                         static_cast<double>(r.piece), r.used});
    }
    for (auto& b : table_blocks(t)) blocks.push_back(std::move(b));
    return blocks;
}

// the bank appended to the store under key (a segment of the process pid on
// host, as SegmentWriter makes them)
inline store::Entry export_bank_to_store(const std::filesystem::path& store_dir, const std::string& key, const Bank& B,
                                         double total, std::size_t instances, long long pid, const std::string& host)
{
    SegmentWriter w(store_dir, std::uint64_t{1} << 30, pid, host, {});
    return w.append(key, bank_blocks(B, total, instances));
}

}   // namespace mootation::campaign::fam
