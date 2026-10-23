# The campaign store, format 3

What a campaign with `storage = "store"` writes, byte by byte — enough to
write a reader in another language. A record's bytes — its blocks, its body,
its frame — are the library's (`include/mootation/store/record.hpp` over the
codecs beside it; the extension's `store_*` functions);
`python/mootation/store.py` keeps the files and what reads JSON (which blocks
are point tables, meta.json's final once, a pack's meta table).
Everything is little-endian.

## Files

    <campaign>/store/seg-<time>-p<pid>-<n>.{mhd,raw,idx}   segments, while a campaign runs
    <campaign>/store/pack-<family>.mpk                     a family's pack, after `compact`

- `.mhd` — a text header, one `key = value` per line (as MetaImage's):
  `format = mootation-store`, `format_version`, `kind = segment` and the
  file names beside it.
- `.raw` — records one after the other (below). A segment is only appended
  to; a record counts once it is written whole.
- `.idx` — one line per record: `offset \t length \t checksum (8 hex digits)
  \t status \t key`. It is rebuilt from the `.raw` where they disagree.

A pack holds a family's records sorted by key (problem, algorithm, variant,
seed, budget), so that a pack is the same bytes whatever segments its records
came in (below). Packs of before — `pack-<family>.{mhd,raw,idx}`, a
`pack-<family>.meta` beside them and `metrics-<family>.{mhd,raw}` — are read
as well; `compact` replaces them with the one file.

## A pack (`pack-<family>.mpk`)

Parts one after the other, each starting at a multiple of 64 bytes (zero
bytes between), and a footer of 64 bytes at the end:

| part | what |
|---|---|
| header | `key = value` lines, as an `.mhd` (`kind = pack`, `family`, `records`, `sorted_by`, `code_stamp`, `trajectory_version`, `layout`, `alignment`, `byte_order`), then newlines to a multiple of 64 |
| meta | the runs' meta.json as one table (below; empty where no record is of format 3) |
| metrics | the trajectories' table (below) |
| records | one record (below) after the other, each at a multiple of 64 |
| index | `T` and the text codec's bytes of the lines `offset 	 length 	 checksum 	 status 	 key` (offset from the start of the file); `R` and the lines as they are where written without the extension |

The footer:

    magic        8 bytes   "MOOTPEND"
    meta         u64, u64  offset, length
    metrics      u64, u64  offset, length
    index        u64, u64  offset, length
    crc          u32       CRC-32C of the file before the footer
    index crc    u32       CRC-32C of the index's bytes

A reader reads the footer, the index (its checksum), then a record where the
index says (the record's own checksum); a check reads the whole file against
`crc`, and where it fails, each part against its own checksum (the meta and
metrics tables, the index, every record) and the bytes between the parts
against zero, to say where; what is left is the header or the footer. A pack
is written to a temporary file and renamed when whole.

### The metrics table

    magic "MOOTMTR1" | u32 CRC-32C of the body | body

The body is pieces (below): a directory — JSON `{"family", "rows",
"columns": [[name, dtype], ...]}` — then each column's block (as a record's,
without the count).
One row per trajectory record of every run: `problem`, `algorithm`,
`variant`, `status` (each a block `<name>#labels`, JSON list of the
distinct texts, and `<name>`, `<i4` indices into it), `seed`, `budget`, then
every number the trajectory's records hold (`<f8`; bools 0/1, null NaN).

## A record

    magic        8 bytes   "MOOTREC2" (CRC-32C) — "MOOTREC1" (zlib's CRC32) in records of before
    rest         u64       the length of everything after this field
    key length   u16
    key          UTF-8     problem|algorithm|variant|seed|budget
    key hash     8 bytes   the first 8 bytes of SHA-256(key)
    checksum     u32       of the body: CRC-32C (Castagnoli, reflected polynomial
                           0x82F63B78, initial and final xor 0xFFFFFFFF; "123456789"
                           gives 0xE3069283) — zlib's CRC32 under "MOOTREC1"
    body

A record whose length, key hash or checksum do not hold is not whole: reading
stops there (and `repair` cuts it off a segment's tail).

## The body: blocks

    count   u32
    count times:
      name length  u16, name  UTF-8
      dtype length u8,  dtype ASCII   "json", "text", "bytes" or a NumPy dtype ("<f8", "<i8", "|u1", "<f4" ...)
      ndim         u8,  shape u64 × ndim
      method length u8, method ASCII
      payload length u64, payload

The block's bytes (its "data") are the payload decoded by its method:

| method | data |
|---|---|
| `raw` | the payload |
| `text` | the text codec's output (below) |
| `shuffle+text` | the text codec's output, unshuffled: the payload's bytes are the first byte of every item, then the second, ... (item size from the dtype) |
| `f64` | a float64 column (below), its values' bytes; the count is the product of the shape |
| `i64` | an integer column (below), each value cast to the block's dtype |
| `zlib`, `shuffle+zlib` | zlib (RFC 1950) — written by formats 1 and 2, and by a store written without the extension |

The writer keeps, per block, the shortest of the methods that apply.

### The blocks of a run

| name | dtype | what |
|---|---|---|
| `#format` | json | `{"version": 3}` |
| `meta.json` | json | the run's meta.json (but "final", below) |
| `<table>#head` | json | `{"comments": [...], "columns": [...], "ints": [...], "newline": ...}` |
| `<table>` | `<f8` | the table's numbers, rows × columns, row-major |
| `trajectory.jsonl#head` | json | `{"kind": "jsonl", "columns": [...], "texts": [...], "newline": ...}` |
| `trajectory.jsonl#cells` | `\|u1` | per record and column: 0 absent, 1 float, 2 int, 3 bool, 4 null, 5 text |
| `trajectory.jsonl` | `<f8` | the values (a text: its index in "texts") |
| `<name>.npz#head`, `<name>.npz/<array>` | json, any | an `.npz` array by array |
| `#points` | bytes | the tables (below) |

A table's CSV is its "comments" lines, the columns joined by commas, and every
row's numbers written `"{:.10g}"` (Python's format; an "ints" column as an
integer), lines ended by "newline" — the bytes a campaign with
`storage = "files"` writes. A trajectory's line is its record's
`json.dumps(..., separators=(",", ":"))`.

### `#points`: the point sets together

Every `<f8` block of two dimensions whose `#head` names as many columns as it
has is taken out of the block list and coded in `#points`:

    k        u32         the tables
    places   u32 × k     each table's index in the block list
    tables   the tables' coding (below)

Reading decodes the tables and inserts each, as a `<f8` block of its rows and
columns, at its place (in increasing order of the places): the block list is
the one written.

The tables' coding (`include/mootation/store/record_tables.hpp`):

    T        varint      tables
    T times: name (varint length, UTF-8), rows (varint), columns m (varint),
             m role indices (varint)
    R        varint      roles
    R times: role name (varint length, UTF-8), column length (varint), a float64 column

A role is a column name; its column is the role's values through every table
that has it, in the tables' order, each table's rows in order. So a point the
archive holds and the population holds again is a reference the second time.

### `meta.json`: "final" once

Where meta.json's `"final"` holds values that the trajectory's last record
holds alike (the same JSON text), the record's meta.json has, in place of
those keys, `"#keys"` (all of final's keys, in order) and `"#trajectory"` (those
read from the trajectory). Reading rebuilds final in the order of `#keys` and
writes meta.json again as `json.dumps(meta, indent=1)`, its line ends CR LF
where the stored text's are. A writer does this only where the rebuild gives
the original bytes back.

## A pack's meta table

`compact` takes every format-3 run's meta.json (as its record keeps it, final
once) out of the record — a block `#meta` (json, empty) marks its place — and
writes them as one table:

    magic "MOOTMET1" | u32 CRC-32C of the body | body

The body is pieces, each a varint length and its bytes:

| piece | what |
|---|---|
| header | text codec: JSON `{"rows": n, "columns": [path, ...]}` |
| keys | text codec: JSON list of the runs' keys, in the pack's order |
| skeletons | text codec: JSON list of the distinct skeletons |
| skeleton of each row | integer column (-1: kept whole) |
| CR LF of each row | integer column (1: the text's line ends are CR LF) |
| kept whole | text codec: JSON `{row: text}` |
| per column, five | its leaves' types (integer column: 0 null, 1 false, 2 true, 3 int, 4 float, 5 text), its ints (integer column), its floats (float64 column), its texts' indices (integer column), its texts (text codec: JSON list) |

A row's JSON is its skeleton — its dicts and lists in order, every leaf a 0
(`json.dumps(..., separators=(",", ":"))`) — with the leaves filled in depth
first, each the next of its column, a column being a leaf's path: the JSON
text of the list of keys and `[index]`s down to it. The text is
`json.dumps(meta, indent=1)`, CR LF where marked. Reading a pack's record
puts its meta.json back at `#meta` (then rebuilds final).

## The codecs

Varints are LEB128 (7 bits a byte, low first, the high bit set but on the
last byte).

### The range coder (`range_coder.hpp`)

A binary range coder over 32 bits. The state is `low` (with a carry bit) and
`range`, starting at 0 and 0xFFFFFFFF. A bit with probability `p` of a 0 (16
bits, in 1/65536) splits the range at `bound = (range >> 16) * p`: a 0 keeps
`range = bound`; a 1 adds `bound` to `low` and keeps `range - bound`. A carry
past 32 bits adds one to the bytes written (a run of 0xFF becomes 0x00 and
the byte before it grows). While `range < 2^24` the top byte of `low` is
written and both shift by 8. `m` direct bits (m ≤ 16) set `range >>= m` and
add `bits * range` to `low`. The end writes the four bytes of `low`.

A decoder reads four bytes into `code` and mirrors it: a bit is 0 where
`code < bound`; `m` direct bits are `code / (range >> m)`.

A probability starts at 1/2 and after each bit moves towards it by about
1/(n + 2) of the distance, n the bits it has seen (to at most 30), clamped
to [32, 65504]; integer arithmetic, with `s = floor(65536 / (n + 2))`:
`p += ((65536 - p) * s) >> 16` after a 0, `p -= (p * s) >> 16` after a 1.

The value tree's nodes are found by their prefix (with the leading 1); this
is the bytes' definition, whatever the coder keeps them in.

### A float64 column (`column_codec.hpp`)

    method  u8      0 raw, 1 model, 6 decimal
    count   varint
    raw:     count × u64, each value's bits
    model:   k (u8, 8..28), then the range coder's bytes
    decimal: e (u8, 0..18), then an integer column (below) of count values d

The model codes the values in order, each as its 64 bits. A value is a
repeat (a flag bit 1) when it is bit for bit one met before in the column:
then its index among the column's distinct values, in the order they first
came, by a binary tree of `w` bits, `w` the bit width of (distinct values so
far − 1). Otherwise (flag 0) its top k bits by a binary tree, then its low
64 − k bits as direct bits. The flag's probability is one of two, by the
previous flag. A tree's node is the bits coded so far with a leading 1: the
value tree's node keys are those prefixes; an index tree's are
`(w + 1) << 40 | prefix`. Every node has its own probability. The writer
chooses k as the least of the tree's estimated cost (the
Krichevsky-Trofimov code length of every node's counts) plus 64 − k bits per
new value, and keeps raw where that is shorter.

Decimal (the idea of ALP, Afroozeh, Kuffó & Boncz 2023): each value is
`d / 10^e`, one IEEE division of the integer d (as a double) by 10^e (exact
in a double for e ≤ 18). The writer takes it where every value of the column
comes back so bit for bit (d = round(x·10^e), half away from zero) at one e
— the least such e — and it is shorter than the model; -0.0, NaN and the
infinities never are decimals.

### An integer column

    method  u8      0 raw, 2 constant, 3 frame of reference, 4 differences, 5 runs
    count   varint
    raw:        count × u64
    constant:   u64
    frame:      u64 minimum (of the values with the sign bit flipped), u8 width w,
                then every value − minimum in w bits, packed low bit first
    differences: u64 first value, then the frame of the zigzag differences
                 (0, -1, 1, -2 ... as 0, 1, 2, 3 ...)
    runs:       (zigzag value varint, run length varint) pairs

### Text (`text_codec.hpp`)

    length  varint (the bytes decoded)
    the range coder's bytes

A sequence of literals and copies (LZ77): a flag (its probability by the
previous token's kind), then a literal's 8 bits by a tree (one of 8, by the
previous byte's top three bits), or a copy's length − 3 and distance − 1,
each as its bit width w by a tree of 5 bits and the bits below its top one —
up to 4 of them by a tree for w, the rest direct. A copy may overlap the
bytes it writes.

## Reading formats 1 and 2

Format 1 (task 16) kept a table as its CSV's text or numbers of ten digits;
format 2 its float64 numbers; both with zlib and the shuffle, under
"MOOTREC1". A record says its format in `#format` (none: 1).
