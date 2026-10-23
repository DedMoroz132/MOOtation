#pragma once
// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// CRC-32C (Castagnoli, Bräuer & Herrmann 1993: the polynomial 0x1EDC6F41,
// reflected 0x82F63B78), the store's checksum of a record (task 17, §3а.3):
// a byte at a time from a table of 256, initial value and final xor
// 0xFFFFFFFF, as iSCSI (RFC 3720, appendix B.4) and SSE4.2's crc32 compute
// it — the same value; "123456789" gives 0xE3069283.
// ============================================================================

#include <array>
#include <cstddef>
#include <cstdint>

namespace mootation::store {

namespace detail {

inline const std::array<std::uint32_t, 256>& crc32c_table()
{
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (c >> 1) ^ 0x82F63B78u : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    return table;
}

}   // namespace detail

// The CRC-32C of n bytes; `crc` the value of the bytes before them (0 to start).
inline std::uint32_t crc32c(const std::uint8_t* data, std::size_t n, std::uint32_t crc = 0)
{
    const auto& t = detail::crc32c_table();
    crc = ~crc;
    for (std::size_t i = 0; i < n; ++i) crc = t[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    return ~crc;
}

// zlib's CRC-32 (ISO 3309, the reflected polynomial 0xEDB88320), the checksum
// of the records of before (MOOTREC1), read only; "123456789" gives 0xCBF43926.
inline std::uint32_t crc32_zlib(const std::uint8_t* data, std::size_t n)
{
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (c >> 1) ^ 0xEDB88320u : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    std::uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < n; ++i) crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    return ~crc;
}

}   // namespace mootation::store
