#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// The point sets of a run's record, coded together (task 17, stage 4; the
// owner's decisions, supplement 5, §2.1 and §2.7): the tables of a record —
// the archive, the final population, the archive's selection, the second
// answer, the trajectory's values — column by column, a column of one name
// (a role: f1, x3, cv ...) taken through every table that has it, in the
// tables' order, as one float64 column (column_codec.hpp). A point the
// archive holds and the population holds again is then a reference into the
// role's values, written once.
//
// THE BYTES: the count of tables; per table its name, rows, columns and each
// column's role (an index into the roles); the count of roles; per role its
// name and its coded column. Names are UTF-8 with their length; counts are
// varints (column_codec.hpp).
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "column_codec.hpp"

namespace mootation::store {

struct Table {
    std::string name;
    std::vector<std::string> columns;
    std::size_t rows = 0;
    std::vector<double> values;          // row-major, rows x columns
};

namespace tab {

inline void put_text(std::vector<std::uint8_t>& out, const std::string& s)
{
    col::put_varint(out, s.size());
    out.insert(out.end(), s.begin(), s.end());
}

inline std::string get_text(const std::uint8_t* data, std::size_t size, std::size_t& pos)
{
    const std::uint64_t n = col::get_varint(data, size, pos);
    if (n > size - pos) throw std::runtime_error("tables: a name past the bytes");
    std::string s(reinterpret_cast<const char*>(data + pos), static_cast<std::size_t>(n));
    pos += static_cast<std::size_t>(n);
    return s;
}

}   // namespace tab

inline std::vector<std::uint8_t> encode_tables(const std::vector<Table>& tables)
{
    std::vector<std::string> roles;
    std::unordered_map<std::string, std::size_t> role_of;
    for (const auto& t : tables) {
        if (t.values.size() != t.rows * t.columns.size()) throw std::invalid_argument("tables: " + t.name + ": not rows x columns");
        for (const auto& c : t.columns)
            if (role_of.emplace(c, roles.size()).second) roles.push_back(c);
    }
    std::vector<std::uint8_t> out;
    col::put_varint(out, tables.size());
    for (const auto& t : tables) {
        tab::put_text(out, t.name);
        col::put_varint(out, t.rows);
        col::put_varint(out, t.columns.size());
        for (const auto& c : t.columns) col::put_varint(out, role_of.at(c));
    }
    col::put_varint(out, roles.size());
    std::vector<double> column;
    for (std::size_t r = 0; r < roles.size(); ++r) {
        column.clear();
        for (const auto& t : tables) {
            const std::size_t m = t.columns.size();
            for (std::size_t j = 0; j < m; ++j) {
                if (t.columns[j] != roles[r]) continue;
                for (std::size_t i = 0; i < t.rows; ++i) column.push_back(t.values[i * m + j]);
            }
        }
        const std::vector<std::uint8_t> coded = encode_f64(column.data(), column.size());
        tab::put_text(out, roles[r]);
        col::put_varint(out, coded.size());
        out.insert(out.end(), coded.begin(), coded.end());
    }
    return out;
}

inline std::vector<Table> decode_tables(const std::uint8_t* data, std::size_t size)
{
    std::size_t pos = 0;
    const std::uint64_t nt = col::get_varint(data, size, pos);
    if (nt > size) throw std::runtime_error("tables: more tables than bytes");
    std::vector<Table> tables(static_cast<std::size_t>(nt));
    std::vector<std::vector<std::size_t>> role_index(tables.size());
    for (std::size_t t = 0; t < tables.size(); ++t) {
        tables[t].name = tab::get_text(data, size, pos);
        const std::uint64_t rows = col::get_varint(data, size, pos);
        if (rows > (std::uint64_t{1} << 31)) throw std::runtime_error("tables: a table of too many rows");
        tables[t].rows = static_cast<std::size_t>(rows);
        const std::uint64_t m = col::get_varint(data, size, pos);
        if (m > size) throw std::runtime_error("tables: more columns than bytes");
        for (std::uint64_t j = 0; j < m; ++j) role_index[t].push_back(static_cast<std::size_t>(col::get_varint(data, size, pos)));
    }
    const std::uint64_t nr = col::get_varint(data, size, pos);
    if (nr > size) throw std::runtime_error("tables: more roles than bytes");
    std::vector<std::string> roles;
    std::vector<std::vector<double>> columns;
    for (std::uint64_t r = 0; r < nr; ++r) {
        roles.push_back(tab::get_text(data, size, pos));
        const std::uint64_t len = col::get_varint(data, size, pos);
        if (len > size - pos) throw std::runtime_error("tables: a column past the bytes");
        // the role's values: every table's rows that have it, in order
        std::size_t count = 0;
        for (std::size_t t = 0; t < tables.size(); ++t)
            for (std::size_t j : role_index[t])
                if (j == r) count += tables[t].rows;
        columns.push_back(decode_f64(data + pos, static_cast<std::size_t>(len), count));
        pos += static_cast<std::size_t>(len);
    }
    if (pos != size) throw std::runtime_error("tables: bytes past the roles");
    std::vector<std::size_t> taken(roles.size(), 0);
    for (std::size_t t = 0; t < tables.size(); ++t) {
        Table& tb = tables[t];
        const std::size_t m = role_index[t].size();
        tb.values.assign(tb.rows * m, 0.0);
        for (std::size_t j = 0; j < m; ++j) {
            const std::size_t r = role_index[t][j];
            if (r >= roles.size()) throw std::runtime_error("tables: a column of no role");
            tb.columns.push_back(roles[r]);
            for (std::size_t i = 0; i < tb.rows; ++i) tb.values[i * m + j] = columns[r][taken[r] + i];
            taken[r] += tb.rows;
        }
    }
    return tables;
}

}   // namespace mootation::store
