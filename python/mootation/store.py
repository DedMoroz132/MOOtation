# SPDX-License-Identifier: Apache-2.0
"""The campaign's own result store (task 16, §1): a few large files instead of
four small ones per run. No database and nothing outside the standard library
and NumPy: a text header in the manner of MetaImage (.mhd: "key = value"), the
binary data beside it (.raw), and an index (.idx) rebuilt from the data when it
disagrees with it.

LAYER 1, WHILE A CAMPAIGN RUNS ([campaign] storage = "store"). Every worker
process appends to its own segment, so nothing is locked:

    <campaign>/store/seg-<time>-p<pid>-<n>.mhd   the header (Notepad reads it)
    <campaign>/store/seg-<time>-p<pid>-<n>.raw   the records, only appended
    <campaign>/store/seg-<time>-p<pid>-<n>.idx   key -> offset, length, CRC32

A new segment starts at SEGMENT_BYTES (a knob: [campaign] segment_mb). One
record is one run:

    MAGIC (8 bytes) | u64 length of the rest | u16 key length | key (UTF-8)
    | 8 bytes of the key's SHA-256 | u32 checksum of the body | body

the checksum CRC-32C (MAGIC "MOOTREC2", task 17, §3а.3; the library's own,
include/mootation/store/crc32c.hpp, or the same in Python) — or, in records
written before, zlib's CRC32 (MAGIC "MOOTREC1"), still read

and the body a sequence of blocks: u32 count, then per block u16 name length,
name, u8 dtype length, dtype, u8 ndim, u64 per dimension, u8 length of the
compression's name, the name, u64 payload length, payload. All little-endian.
A record counts once it is written whole and on the disk (flush + fsync);
opening a segment reads up to its last whole record (length and CRC), and
repair() cuts an unfinished tail off, saying so in store/store.log.

FORMAT 2 (task 16, supplement 1): the run's point sets — the final
population, the archive, the second answer, the snapshots — come straight
from memory as float64 (counts as integers), never through the text of a CSV:
a table is its numbers with the header lines its CSV would have
(table_blocks), so that export_files writes that CSV, ten significant digits,
byte for byte as a campaign with storage = "files" writes it. meta.json is
kept as text, the trajectory column by column (float64 values and a code per
cell: float, int, bool, null, text, absent), checked to give its lines back
byte for byte, else kept as text. A record says its format in a block
"#format"; a record of format 1 (task 16: the CSV's ten-digit numbers, or its
text) is read as before. keep = "standard" leaves out the archives at the
intermediate points of the budget grid (archive_at.npz); "full" keeps
everything.

The key of a run: problem|algorithm|variant|seed|budget — the core, the
label of the variant ("" for none), and the budget in evaluations the run
was given.

LAYER 2, AFTER THE CAMPAIGN (compact): one pack per family of problems
(families.family), records sorted by (problem, algorithm, variant, seed,
budget), so that the pack is the same whatever process wrote which run: one
file, pack-<family>.mpk (task 17, §3а.3) — a text header, the runs' meta.json
as one table (supplement 5, §2.3; encode_meta_table), the metrics table
(layer 3), the records, an index, each at an offset that is a multiple of 64,
and a footer of 64 bytes at the end: where the parts are and the CRC-32C of
everything before it. A record of format 3 keeps a mark "#meta" where its
meta.json was, and reading puts it back — the blocks read are the blocks
written; records of formats 1 and 2 are copied byte for byte. Packs of
before (pack-<family>.raw, .idx, .mhd, .meta and metrics-<family>.mhd/.raw)
are read as well, and compact replaces them. LAYER 3: a table of the
trajectories per family (column after column; a part of the pack, before it
metrics-<family>.mhd/.raw), read by metrics() and written to CSV by
export_csv().

Compression: zlib; for an array of numbers also zlib after a byte shuffle (the
first byte of every number, then the second, ...; as Blosc does, which puts
alike high bytes together), and the smaller of the two is kept. The block says
which. Decompression gives the same bits.

FORMAT 3 (task 17, stage 4; the owner's decisions of 2026-10-24, supplement
5, §2): the library's own codecs where the extension is built
(include/mootation/store/), and no zlib. Bytes — a JSON, a text, any block —
are "text": LZ77 and an adaptive range coder; an array also "shuffle+text"
(the same after the shuffle), of float64 "f64" (its values met before by
reference, new ones by their top bits through the range coder and the rest
raw), of integers "i64" (frame of reference, differences or runs): the
shortest is kept. Without the extension a store is written with zlib, as
format 2, and read anywhere. The point sets of a record — every float64 table with its
columns named in its "#head": the archive, the final population, the
archive's selection, the second answer, the trajectory's values — go
together into one block "#points" (store_encode_tables): a column of one
name through all of them, so that a point two of them hold is written once.
"#points" holds the tables' places among the record's blocks, and reading
puts every table back where it was: the blocks read are the blocks written,
bit for bit. meta.json's "final" is not kept twice (supplement 5, §2.4):
the values the trajectory's last record holds alike (json's text equal) are
read from there — meta.json keeps the keys' order ("#keys") and which come
from the trajectory ("#trajectory"), and reading gives its bytes back (the
writer checks that it does, else keeps meta.json whole). Reading format 3
needs the extension; formats 1 and 2 are read as before.

ONE IMPLEMENTATION (task 17, §3а.3): a format-3 record's bytes — its blocks,
its body with "#points", its frame — and the files — a segment's record
appended and synced, a pack written, its footer and index read, a pack
checked — are the library's (include/mootation/store/record.hpp, pack.hpp;
the extension's store_* functions). What reads JSON
stays here: which blocks are point tables (their "#head"), meta.json's final
once and back, the pack's meta table. Without the extension this module
writes and reads format 2 (zlib) by itself.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import socket
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

FORMAT = "mootation-store"
FORMAT_VERSION = 3
MAGIC = b"MOOTREC1"                     # zlib's CRC32 of the body (read only)
MAGIC_C = b"MOOTREC2"                   # CRC-32C of the body (written)
SEGMENT_BYTES = 1 << 30                 # a new segment at 1 GiB ([campaign] segment_mb)
ZLEVEL = 6
STORE_DIR = "store"
SCRATCH_DIR = "tmp"                     # under store/: the runs being written
LOG_FILE = "store.log"
# the files a run writes that keep = "standard" leaves out
FULL_ONLY = ("archive_at.npz",)


# ── keys ────────────────────────────────────────────────────────────────────


def run_key(problem: str, algorithm: str, variant: str, seed: int, budget: int) -> str:
    """The canonical key of a run (the module's docstring)."""
    for part in (problem, algorithm, variant):
        if "|" in part:
            raise ValueError(f"a store key part holds '|': {part!r}")
    return f"{problem}|{algorithm}|{variant}|{int(seed)}|{int(budget)}"


def job_key(job) -> str:
    """The key of a campaign job (campaign.Job)."""
    return run_key(job.problem, job.algorithm, job.label, job.seed, job.budget)


def split_key(key: str) -> tuple:
    """(problem, algorithm, variant, seed, budget) of a key."""
    problem, algorithm, variant, seed, budget = key.split("|")
    return problem, algorithm, variant, int(seed), int(budget)


def rel_dir_of(key: str) -> PurePosixPath:
    """The folder the run would have in the files layout: problem/label/run_seed."""
    problem, algorithm, variant, seed, _ = split_key(key)
    return PurePosixPath(problem) / (variant or algorithm) / f"run_{seed}"


def sort_key(key: str) -> tuple:
    problem, algorithm, variant, seed, budget = split_key(key)
    return problem, algorithm, variant, seed, budget


# ── blocks ──────────────────────────────────────────────────────────────────


def shuffle(data: bytes, itemsize: int) -> bytes:
    """The first byte of every item, then the second, ... (Blosc's shuffle)."""
    import numpy as np
    if itemsize <= 1 or not data:
        return data
    return np.frombuffer(data, np.uint8).reshape(-1, itemsize).T.tobytes()


def unshuffle(data: bytes, itemsize: int) -> bytes:
    import numpy as np
    if itemsize <= 1 or not data:
        return data
    return np.frombuffer(data, np.uint8).reshape(itemsize, -1).T.tobytes()


@dataclass
class Block:
    name: str
    dtype: str          # "json", "text", "bytes", or a NumPy dtype ("<f8", "<i8", ...)
    shape: tuple        # () for json, text and bytes
    data: bytes         # uncompressed


def _itemsize(dtype: str) -> int:
    import numpy as np
    return np.dtype(dtype).itemsize if dtype not in ("json", "text", "bytes") else 1


def _codec():
    """The extension, where it has the store's codecs (format 3), else None."""
    try:
        from . import _core
    except ImportError:
        return None
    return _core if hasattr(_core, "store_encode_tables") else None


def _lib(call, *args):
    """A call into the library's store, its errors of bytes as ValueError
    (what this module raises for a damaged store)."""
    try:
        return call(*args)
    except RuntimeError as e:
        raise ValueError(str(e)) from None


class _Mapped:
    """A file's bytes, mapped (mmap) while in a with block: what the library
    reads without a copy."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def __enter__(self):
        import mmap
        self.fh = open(self.path, "rb")
        try:
            self.mm = mmap.mmap(self.fh.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError:                  # an empty file
            self.fh.close()
            raise ValueError(f"{self.path.name}: empty")
        return self.mm

    def __exit__(self, *exc):
        self.mm.close()
        self.fh.close()


def encode_block(b: Block) -> bytes:
    c = _codec()
    if c is None:
        how, payload = "zlib", zlib.compress(b.data, ZLEVEL)
        if b.dtype not in ("json", "text", "bytes"):
            # the shuffle helps where the high bytes repeat (counts, float32);
            # the ten-digit decimals of a CSV fill the mantissa, and it does
            # not: the smaller of the two, task 16, §1.3
            shuffled = zlib.compress(shuffle(b.data, _itemsize(b.dtype)), ZLEVEL)
            if len(shuffled) < len(payload):
                how, payload = "shuffle+zlib", shuffled
    else:
        # format 3: the library's (record.hpp: float64 and integers by their
        # column codecs, the rest by the text codec, an array shuffled or not)
        return bytes(c.store_encode_body([(b.name, b.dtype, list(b.shape), b.data)], []))[4:]
    if len(payload) >= len(b.data):
        how, payload = "raw", b.data
    name, dt, hw = b.name.encode(), b.dtype.encode(), how.encode()
    out = [struct.pack("<H", len(name)), name, struct.pack("<B", len(dt)), dt,
           struct.pack("<B", len(b.shape)), struct.pack(f"<{len(b.shape)}Q", *b.shape),
           struct.pack("<B", len(hw)), hw, struct.pack("<Q", len(payload)), payload]
    return b"".join(out)


def decode_blocks(body: bytes) -> list:
    c = _codec()
    if c is not None:
        out = []
        for name, dtype, shape, data, how in _lib(c.store_decode_run, body):
            if how == "zlib":
                data = zlib.decompress(data)
            elif how == "shuffle+zlib":
                data = unshuffle(zlib.decompress(data), _itemsize(dtype))
            out.append(Block(name, dtype, tuple(shape), bytes(data)))
        return out
    # without the extension: formats 1 and 2
    out = []
    (count,) = struct.unpack_from("<I", body, 0)
    at = 4
    for _ in range(count):
        (n,) = struct.unpack_from("<H", body, at); at += 2
        name = body[at:at + n].decode(); at += n
        (n,) = struct.unpack_from("<B", body, at); at += 1
        dtype = body[at:at + n].decode(); at += n
        (nd,) = struct.unpack_from("<B", body, at); at += 1
        shape = struct.unpack_from(f"<{nd}Q", body, at); at += 8 * nd
        (n,) = struct.unpack_from("<B", body, at); at += 1
        how = body[at:at + n].decode(); at += n
        (n,) = struct.unpack_from("<Q", body, at); at += 8
        payload = body[at:at + n]; at += n
        if how == "raw":
            data = payload
        elif how == "zlib":
            data = zlib.decompress(payload)
        elif how == "shuffle+zlib":
            data = unshuffle(zlib.decompress(payload), _itemsize(dtype))
        elif how in ("f64", "i64", "text", "shuffle+text"):
            raise ValueError(f"block {name!r}: format 3 ({how}) needs the compiled extension")
        else:
            raise ValueError(f"block {name!r}: unknown compression {how!r}")
        out.append(Block(name, dtype, tuple(shape), data))
    if any(b.name == "#points" for b in out):
        raise ValueError("block '#points': format 3 needs the compiled extension")
    return out


# ── format 3: the record's point sets together ──────────────────────────────


def _blocks_lib(blocks: list) -> list:
    return [(b.name, b.dtype, list(b.shape), b.data) for b in blocks]


def _blocks_py(got) -> list:
    return [Block(name, dtype, tuple(shape), bytes(data)) for name, dtype, shape, data in got]


def _meta_text(meta: dict, crlf: bool) -> bytes:
    """meta.json's text as campaign writes it (json.dumps, indent 1), its line
    ends CR LF where the file's were (a text file written on Windows)."""
    text = json.dumps(meta, indent=1).encode()
    return text.replace(b"\n", b"\r\n") if crlf else text


def _point_tables(blocks: list) -> list:
    """(index, block, columns) of the float64 tables among blocks: a 2-D
    array whose "#head" names its columns (the library's, store/run_json.hpp)."""
    c = _codec()
    if c is None:
        return []
    return [(i, blocks[i], cols) for i, cols in _lib(c.store_point_tables, _blocks_lib(blocks))]


def _final_once(blocks: list) -> list:
    """Blocks with meta.json's final values that the trajectory's last record
    holds alike left to it (the module's docstring; the library's,
    store/run_json.hpp); unchanged where meta.json would not come back byte
    for byte, or without the extension."""
    c = _codec()
    return blocks if c is None else _blocks_py(_lib(c.store_final_once, _blocks_lib(blocks)))


def _final_back(blocks: list) -> list:
    """meta.json's final rebuilt where it was left to the trajectory (the library's)."""
    c = _codec()
    if c is None:
        mb = next((b for b in blocks if b.name == "meta.json"), None)
        if mb is not None and b'"#keys"' in mb.data:
            raise ValueError("meta.json: its final is in the trajectory; reading it needs the compiled extension")
        return blocks
    return _blocks_py(_lib(c.store_final_back, _blocks_lib(blocks)))


def encode_body(blocks: list) -> bytes:
    c = _codec()
    if c is None:                                       # format 2
        return struct.pack("<I", len(blocks)) + b"".join(encode_block(b) for b in blocks)
    # final once, the point tables together (store/run_json.hpp)
    return bytes(_lib(c.store_encode_run, _blocks_lib(blocks)))


_CRC32C_TABLE: list = []


def crc32c(data: bytes) -> int:
    """CRC-32C (Castagnoli; iSCSI's): the extension's, or the same value here."""
    c = _codec()
    if c is not None:
        return int(c.crc32c(data))
    if not _CRC32C_TABLE:
        for i in range(256):
            v = i
            for _ in range(8):
                v = (v >> 1) ^ 0x82F63B78 if v & 1 else v >> 1
            _CRC32C_TABLE.append(v)
    crc = 0xFFFFFFFF
    for b in data:
        crc = _CRC32C_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def encode_record(key: str, body: bytes) -> bytes:
    c = _codec()
    if c is not None:
        return bytes(c.store_encode_record(key, body))
    k = key.encode()
    rest = (struct.pack("<H", len(k)) + k + hashlib.sha256(k).digest()[:8]
            + struct.pack("<I", crc32c(body)) + body)
    return MAGIC_C + struct.pack("<Q", len(rest)) + rest


def _parse_record(buf: bytes, at: int):
    """(key, body, crc, end) of the record at `at` in buf, or None where there
    is no whole, sound record there."""
    c = _codec()
    if c is not None:
        got = c.store_parse_record(buf, at)
        return None if got is None else (got[0], bytes(got[1]), got[2], got[3])
    magic = buf[at:at + 8]
    if magic not in (MAGIC, MAGIC_C) or at + 16 > len(buf):
        return None
    (rest,) = struct.unpack_from("<Q", buf, at + 8)
    end = at + 16 + rest
    if end > len(buf) or rest < 14:
        return None
    (n,) = struct.unpack_from("<H", buf, at + 16)
    k = buf[at + 18:at + 18 + n]
    if hashlib.sha256(k).digest()[:8] != buf[at + 18 + n:at + 26 + n]:
        return None
    (crc,) = struct.unpack_from("<I", buf, at + 26 + n)
    body = buf[at + 30 + n:end]
    if (crc32c(body) if magic == MAGIC_C else zlib.crc32(body)) != crc:
        return None
    return k.decode(), body, crc, end


# ── a run's files as blocks ─────────────────────────────────────────────────


def _csv_blocks(name: str, text: str):
    """The blocks of a CSV the campaign writes — '#' lines, the column names,
    rows of numbers written "{:.10g}" (an int for "extreme") — or None where
    the numbers do not give the same text back. Its line ends are kept (a
    text file written on Windows has CR LF)."""
    import numpy as np
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.replace("\r\n", "\n").split("\n")
    if not lines or lines[-1] != "":
        return None
    lines = lines[:-1]
    comments = []
    while lines and lines[0].startswith("#"):
        comments.append(lines.pop(0))
    if not lines:
        return None
    columns = lines.pop(0).split(",")
    ints = [i for i, c in enumerate(columns) if c == "extreme"]
    try:
        rows = [[float(v) for v in line.split(",")] for line in lines]
    except ValueError:
        return None
    if any(len(r) != len(columns) for r in rows):
        return None
    A = np.asarray(rows, "<f8").reshape(len(rows), len(columns))
    head = {"comments": comments, "columns": columns, "ints": ints, "newline": newline}
    if _csv_text(head, A) != text:
        return None
    return [Block(name + "#head", "json", (), json.dumps(head).encode()),
            Block(name, "<f8", A.shape, A.tobytes())]


def _csv_text(head: dict, A) -> str:
    ints = set(head["ints"])
    out = [c + "\n" for c in head["comments"]]
    out.append(",".join(head["columns"]) + "\n")
    for row in A.tolist():
        out.append(",".join(str(int(v)) if j in ints else f"{v:.10g}"
                            for j, v in enumerate(row)) + "\n")
    text = "".join(out)
    return text.replace("\n", "\r\n") if head.get("newline") == "\r\n" else text


def format_block() -> Block:
    """The block that says a record's format (FORMAT_VERSION); a record
    without one is of format 1."""
    return Block("#format", "json", (), json.dumps({"version": FORMAT_VERSION}).encode())


def record_format(by: dict) -> int:
    b = by.get("#format")
    return int(json.loads(b.data)["version"]) if b is not None else 1


def table_blocks(name: str, head: dict, A) -> list:
    """A table of a run (format 2): its numbers as float64, straight from
    memory, and head = {"comments": the '#' lines, "columns": the names,
    "ints": the columns written as integers, "newline": the CSV's line end},
    from which file_content writes its CSV ("{:.10g}"), as campaign does."""
    import numpy as np
    n = len(head["columns"])
    A = np.ascontiguousarray(np.asarray(A, "<f8").reshape(-1, n) if n else np.zeros((0, 0), "<f8"))
    return [Block(name + "#head", "json", (), json.dumps(dict(head, numbers="float64")).encode()),
            Block(name, "<f8", A.shape, A.tobytes())]


def array_blocks(name: str, arrays: dict) -> list:
    """Named arrays of a run (an .npz in the files layout), array by array."""
    import numpy as np
    out = [Block(name + "#head", "json", (), json.dumps({"arrays": list(arrays)}).encode())]
    for k, a in arrays.items():
        a = np.ascontiguousarray(a)
        a = a.astype(a.dtype.newbyteorder("<"))
        out.append(Block(f"{name}/{k}", a.dtype.str, a.shape, a.tobytes()))
    return out


# the trajectory's cells (format 2): what each value of a record is
_ABSENT, _FLOAT, _INT, _BOOL, _NULL, _TEXT = 0, 1, 2, 3, 4, 5


def _jsonl_blocks(name: str, text: str):
    """A trajectory column by column: the keys of every record in one order
    (each record's keys keep their order in it), float64 values, a code per
    cell, the text values apart — or None where the lines would not come back
    byte for byte (json.dumps(record, separators=(",", ":")), as campaign
    writes them)."""
    import numpy as np
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    try:
        recs = [json.loads(line) for line in lines]
    except ValueError:
        return None
    if not all(isinstance(r, dict) for r in recs):
        return None
    cols: list = []
    for r in recs:                     # one order that keeps every record's
        prev = -1
        for k in r:
            if k not in cols:
                cols.insert(prev + 1, k)
            prev = cols.index(k)
    where = {k: j for j, k in enumerate(cols)}
    V = np.zeros((len(recs), len(cols)), "<f8")
    C = np.zeros((len(recs), len(cols)), "u1")
    texts = []
    for i, r in enumerate(recs):
        for k, v in r.items():
            j = where[k]
            if v is None:
                C[i, j] = _NULL
            elif isinstance(v, bool):
                C[i, j], V[i, j] = _BOOL, float(v)
            elif isinstance(v, int):
                if abs(v) >= 2 ** 53:
                    return None
                C[i, j], V[i, j] = _INT, float(v)
            elif isinstance(v, float):
                C[i, j], V[i, j] = _FLOAT, v
            elif isinstance(v, str):
                C[i, j], V[i, j] = _TEXT, len(texts)
                texts.append(v)
            else:
                return None
    head = {"kind": "jsonl", "columns": cols, "texts": texts, "newline": newline}
    blocks = [Block(name + "#head", "json", (), json.dumps(head).encode()),
              Block(name + "#cells", "|u1", C.shape, C.tobytes()),
              Block(name, "<f8", V.shape, V.tobytes())]
    if _jsonl_text(head, C, V) != text:
        return None
    return blocks


def _jsonl_text(head: dict, C, V) -> str:
    cols, texts = head["columns"], head["texts"]
    out = []
    for c, v in zip(C.tolist(), V.tolist()):
        rec = {}
        for j, code in enumerate(c):
            if code == _ABSENT:
                continue
            rec[cols[j]] = (v[j] if code == _FLOAT else int(v[j]) if code == _INT
                            else bool(v[j]) if code == _BOOL else None if code == _NULL
                            else texts[int(v[j])])
        out.append(json.dumps(rec, separators=(",", ":")) + "\n")
    text = "".join(out)
    return text.replace("\n", "\r\n") if head.get("newline") == "\r\n" else text


def trajectory_columns(by: dict, name: str = "trajectory.jsonl"):
    """(columns, cells, values) of a trajectory kept column by column (format
    2), or None: the metrics tables read the numbers without the text."""
    h = by.get(name + "#head")
    if h is None or json.loads(h.data).get("kind") != "jsonl":
        return None
    head = json.loads(h.data)
    return head["columns"], block_array(by[name + "#cells"]), block_array(by[name])


def pack_run_dir(d: Path, keep: str = "standard") -> list:
    """The blocks of every file of a run folder (the module's docstring)."""
    import numpy as np
    blocks = []
    for f in sorted(p for p in Path(d).iterdir() if p.is_file()):
        name = f.name
        if keep == "standard" and name in FULL_ONLY:
            continue
        if name.endswith(".json"):
            blocks.append(Block(name, "json", (), f.read_bytes()))
        elif name.endswith(".jsonl"):
            raw = f.read_bytes()
            got = _jsonl_blocks(name, raw.decode("utf-8"))
            blocks += got if got is not None else [Block(name, "text", (), raw)]
        elif name.endswith((".csv", ".csv.gz")):
            raw = f.read_bytes()
            text_bytes = gzip.decompress(raw) if name.endswith(".gz") else raw
            got = _csv_blocks(name, text_bytes.decode("utf-8"))
            blocks += got if got is not None else [Block(name, "text", (), text_bytes)]
        elif name.endswith(".npz"):
            with np.load(f, allow_pickle=False) as z:
                names = list(z.files)
                arrays = [np.ascontiguousarray(z[k]) for k in names]
            blocks.append(Block(name + "#head", "json", (), json.dumps({"arrays": names}).encode()))
            for k, a in zip(names, arrays):
                a = a.astype(a.dtype.newbyteorder("<"))
                blocks.append(Block(f"{name}/{k}", a.dtype.str, a.shape, a.tobytes()))
        else:
            blocks.append(Block(name, "bytes", (), f.read_bytes()))
    return blocks


def block_array(b: Block):
    import numpy as np
    return np.frombuffer(b.data, np.dtype(b.dtype)).reshape(b.shape)


def file_names(by: dict) -> list:
    """The run's files among blocks {name: Block}."""
    return [n for n in by if "#" not in n and "/" not in n] + \
           [n[:-len("#head")] for n in by if n.endswith(".npz#head")]


def file_content(by: dict, name: str):
    """One file of a run, from its blocks {name: Block}: the text (str) of
    meta.json, trajectory.jsonl and a CSV (uncompressed, under its own name,
    .gz or not), {array name: array} of an .npz, bytes otherwise."""
    if name.endswith(".npz"):
        return {k: block_array(by[f"{name}/{k}"]) for k in json.loads(by[name + "#head"].data)["arrays"]}
    b = by[name]
    if b.dtype in ("json", "text"):
        return b.data.decode("utf-8")
    if b.dtype == "bytes":
        return b.data
    head = json.loads(by[name + "#head"].data)
    if head.get("kind") == "jsonl":                                            # a trajectory
        return _jsonl_text(head, block_array(by[name + "#cells"]), block_array(b))
    return _csv_text(head, block_array(b))                                     # a table


def run_files(blocks: list) -> dict:
    """{file name: file_content} of every file of a run."""
    by = {b.name: b for b in blocks}
    return {name: file_content(by, name) for name in file_names(by)}


def write_run_files(files: dict, d: Path) -> None:
    """The files layout of one run from run_files(): what the campaign wrote,
    the same text (a .gz compressed again, without a time stamp)."""
    import numpy as np
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        if isinstance(content, dict):
            np.savez_compressed(d / name, **content)
        elif isinstance(content, bytes):
            (d / name).write_bytes(content)
        elif name.endswith(".gz"):
            with open(d / name, "wb") as raw, \
                    gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                gz.write(content.encode("utf-8"))
        else:
            with open(d / name, "w", encoding="utf-8", newline="") as fh:
                fh.write(content)


# ── segments ────────────────────────────────────────────────────────────────


def _write_header(path: Path, fields: dict) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for k, v in fields.items():
            fh.write(f"{k} = {v}\n")
    os.replace(tmp, path)


def read_header(path: Path) -> dict:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if " = " in line:
            k, v = line.split(" = ", 1)
            out[k.strip()] = v.strip()
    return out


def header_of(raw: Path) -> dict:
    """The header of a segment or pack: its .mhd, or a pack file's first part."""
    raw = Path(raw)
    if raw.suffix != PACK_SUFFIX:
        return read_header(raw.with_suffix(".mhd"))
    at = _pack_footer(raw)["meta"][0]
    with open(raw, "rb") as fh:
        text = fh.read(at).decode("utf-8")
    return {k.strip(): v.strip() for k, v in (line.split(" = ", 1) for line in text.splitlines() if " = " in line)}


@dataclass
class Loc:
    raw: Path
    offset: int
    length: int
    crc: int
    status: str


def _status_of(blocks: list) -> str:
    for b in blocks:
        if b.name == "meta.json":
            return str(json.loads(b.data).get("status", "?"))
    return "?"


def scan_raw(raw: Path) -> tuple:
    """([(key, Loc)] of every whole record of a .raw, in order; the end of the
    last one) — the data's own account, whatever its index says."""
    buf = Path(raw).read_bytes()
    out, at = [], 0
    while at < len(buf):
        got = _parse_record(buf, at)
        if got is None:
            break
        key, body, crc, end = got
        try:
            status = _status_of(_with_meta(key, decode_blocks(body), Path(raw)))
        except ValueError:                  # format 3 without the extension: whole, status unread
            status = "?"
        out.append((key, Loc(Path(raw), at, end - at, crc, status)))
        at = end
    return out, at


def _read_idx(idx: Path, raw: Path) -> list | None:
    """The index's entries if they cover the .raw up to its last whole record
    (an unfinished tail past them is the writer's, or a crash's), else None."""
    try:
        lines = Path(idx).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    out, end = [], 0
    for line in lines:
        parts = line.split("\t")
        if len(parts) != 5:
            return None
        off, length, crc, status, key = parts
        if int(off) != end:
            return None
        out.append((key, Loc(Path(raw), int(off), int(length), int(crc, 16), status)))
        end = int(off) + int(length)
    size = Path(raw).stat().st_size
    if end > size:
        return None
    if end < size:
        # a tail past the index: a record written whole with its index line
        # lost is read from the data
        with open(raw, "rb") as fh:
            fh.seek(end)
            tail = fh.read()
        if _parse_record(tail, 0) is not None:
            return None
    return out


def segment_entries(raw: Path) -> list:
    """[(key, Loc)] of a segment or pack: from its index, or from the data
    where the index disagrees."""
    raw = Path(raw)
    if raw.suffix == PACK_SUFFIX:
        return _pack_index(raw)
    got = _read_idx(raw.with_suffix(".idx"), raw)
    return got if got is not None else scan_raw(raw)[0]


def _idx_line(key: str, loc: Loc) -> str:
    return f"{loc.offset}\t{loc.length}\t{loc.crc:08x}\t{loc.status}\t{key}\n"


def write_index(raw: Path, entries: list) -> None:
    tmp = raw.with_suffix(f".idx.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        for key, loc in entries:
            fh.write(_idx_line(key, loc))
    os.replace(tmp, raw.with_suffix(".idx"))


class SegmentWriter:
    """One process's segments in one store: records appended, a new segment
    past `limit` bytes."""

    def __init__(self, store: Path, limit: int = SEGMENT_BYTES, header: dict | None = None):
        self.store = Path(store)
        self.limit = int(limit)
        self.header = dict(header or {})
        self.stamp = time.strftime("%Y%m%d-%H%M%S")
        self.number = 0
        self.raw = None
        self.records = 0

    def _start(self) -> None:
        self.store.mkdir(parents=True, exist_ok=True)
        while True:             # a name no writer of this process took this second
            self.number += 1
            self.raw = self.store / f"seg-{self.stamp}-p{os.getpid()}-{self.number:04d}.raw"
            try:
                self.raw.touch(exist_ok=False)
                break
            except FileExistsError:
                continue
        self.raw.with_suffix(".idx").touch()
        self.records = 0
        self._header()

    def _header(self) -> None:
        # once, when the segment starts: its records are counted by its index
        _write_header(self.raw.with_suffix(".mhd"), {
            "format": FORMAT, "format_version": FORMAT_VERSION, "kind": "segment",
            "data_file": self.raw.name, "index_file": self.raw.with_suffix(".idx").name,
            "process": os.getpid(), "host": socket.gethostname(), "segment": self.number,
            **self.header,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.strptime(self.stamp, "%Y%m%d-%H%M%S")),
        })

    def append(self, key: str, blocks: list) -> Loc:
        rec = encode_record(key, encode_body(blocks))
        if self.raw is None or (self.records and self.raw.stat().st_size + len(rec) > self.limit):
            self._start()
        c = _codec()
        if c is not None:                   # the library's (pack.hpp): appended, synced, indexed
            _, offset, length, crc, status = c.store_append_record(
                str(self.raw), str(self.raw.with_suffix(".idx")), key, rec, _status_of(blocks))
            self.records += 1
            return Loc(self.raw, offset, length, crc, status)
        with open(self.raw, "ab") as fh:
            offset = fh.tell()
            fh.write(rec)
            fh.flush()
            os.fsync(fh.fileno())
        (crc,) = struct.unpack_from("<I", rec, 16 + 2 + len(key.encode()) + 8)
        loc = Loc(self.raw, offset, len(rec), crc, _status_of(blocks))
        with open(self.raw.with_suffix(".idx"), "a", encoding="utf-8", newline="\n") as fh:
            fh.write(_idx_line(key, loc))
        self.records += 1
        return loc


_WRITERS: dict = {}


def writer(store: Path, limit: int = SEGMENT_BYTES, header: dict | None = None) -> SegmentWriter:
    """This process's writer of a store (one per process and store)."""
    key = (str(Path(store).resolve()), os.getpid())
    if key not in _WRITERS:
        _WRITERS[key] = SegmentWriter(store, limit, header)
    return _WRITERS[key]


# ── a store, read ───────────────────────────────────────────────────────────


def store_dir(campaign: Path) -> Path:
    return Path(campaign) / STORE_DIR


def segments(store: Path) -> list:
    return sorted(Path(store).glob("seg-*.raw"))


def packs(store: Path) -> list:
    """The packs: one file each (.mpk), and those of before (.raw beside .idx)."""
    return sorted(Path(store).glob("pack-*.mpk")) + sorted(Path(store).glob("pack-*.raw"))


def index(store: Path) -> dict:
    """{key: Loc} of every run of a store: the packs, then the segments in the
    order they were started — a key met again (a run done again) is its later
    record."""
    out = {}
    for raw in packs(store) + segments(store):
        for key, loc in segment_entries(raw):
            out[key] = loc
    return out


def read_record(loc: Loc) -> list:
    """The blocks of the record at loc; ValueError if it is not whole and sound.
    A pack's record gets its meta.json back from the pack's table."""
    with open(loc.raw, "rb") as fh:
        fh.seek(loc.offset)
        buf = fh.read(loc.length)
    got = _parse_record(buf, 0)
    if got is None or got[2] != loc.crc:
        raise ValueError(f"{loc.raw.name}@{loc.offset}: not a whole record (length or CRC)")
    return _with_meta(got[0], decode_blocks(got[1]), Path(loc.raw))


# ── the pack file (task 17, §3а.3) ──────────────────────────────────────────
#
#     header    "key = value" lines (as an .mhd), then newlines to a multiple of 64
#     meta      the runs' meta.json as a table (encode_meta_table; may be empty)
#     metrics   the metrics table (_metrics_section)
#     records   one after the other, each at a multiple of 64 (zero bytes between)
#     index     "offset \t length \t checksum \t status \t key" lines (the
#               text codec's bytes, a "T" before them; "R" and the lines as
#               they are where the extension is not built)
#     footer    64 bytes: "MOOTPEND", the meta's, metrics' and index's offset
#               and length (u64 each), the CRC-32C of the file before the
#               footer, the CRC-32C of the index's bytes (u32 each)
#
# Every part but the footer starts at a multiple of 64. A reader reads the
# footer, then the index (its own checksum), then any record (its own); a
# check reads the whole file against the footer's checksum.

PACK_SUFFIX = ".mpk"
PACK_FOOT = b"MOOTPEND"
ALIGN = 64
_PACK_CACHE: dict = {}


def _align(out) -> None:
    out.write(b"\0" * (-out.tell() % ALIGN))


def write_pack(path: Path, fields: dict, meta: bytes, metrics: bytes, records: list) -> list:
    """Write a pack: records [(key, record bytes, status)] in order; their
    [(key, Loc)] in it."""
    path = Path(path)
    tmp = path.with_suffix(f".mpk.{os.getpid()}.tmp")
    c = _codec()
    if c is not None:                       # the library's (pack.hpp)
        got = c.store_write_pack(str(path), str(tmp), "".join(f"{k} = {v}\n" for k, v in fields.items()),
                                 meta, metrics, records)
        _PACK_CACHE.clear()
        _META_CACHE.clear()
        return [(k, Loc(path, off, n, crc, st)) for k, off, n, crc, st in got]
    entries = []
    with open(tmp, "wb") as out:
        head = "".join(f"{k} = {v}\n" for k, v in fields.items()).encode("utf-8")
        out.write(head + b"\n" * (-len(head) % ALIGN or ALIGN))
        meta_at = out.tell()
        out.write(meta)
        _align(out)
        metrics_at = out.tell()
        out.write(metrics)
        _align(out)
        for key, rec, status in records:
            at = out.tell()
            out.write(rec)
            entries.append((key, Loc(path, at, len(rec), _parse_record(rec, 0)[2], status)))
            _align(out)
        lines = "".join(_idx_line(k, loc) for k, loc in entries).encode("utf-8")
        c = _codec()
        index = (b"T" + bytes(c.store_encode_text(lines))) if c is not None else (b"R" + lines)
        index_at = out.tell()
        out.write(index)
        _align(out)
        out.flush()
    with open(tmp, "rb") as fh:
        whole = crc32c(fh.read())
    with open(tmp, "ab") as out:
        out.write(PACK_FOOT + struct.pack("<6Q", meta_at, len(meta), metrics_at, len(metrics), index_at,
                                          len(index)) + struct.pack("<II", whole, crc32c(index)))
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)
    _PACK_CACHE.clear()
    _META_CACHE.clear()
    return [(k, Loc(path, loc.offset, loc.length, loc.crc, loc.status)) for k, loc in entries]


def _pack_footer(path: Path) -> dict:
    path = Path(path)
    c = _codec()
    if c is not None:
        with _Mapped(path) as mm:
            return _lib(c.store_pack_footer, mm, path.name)
    size = path.stat().st_size
    if size < 64:
        raise ValueError(f"{path.name}: shorter than a pack's footer")
    with open(path, "rb") as fh:
        fh.seek(size - 64)
        foot = fh.read(64)
    if foot[:8] != PACK_FOOT:
        raise ValueError(f"{path.name}: no pack footer at its end")
    meta_at, meta_n, metrics_at, metrics_n, index_at, index_n = struct.unpack_from("<6Q", foot, 8)
    whole, index_crc = struct.unpack_from("<II", foot, 56)
    return {"meta": (meta_at, meta_n), "metrics": (metrics_at, metrics_n), "index": (index_at, index_n),
            "crc": whole, "index_crc": index_crc, "end": size - 64}


def _pack_part(path: Path, name: str) -> bytes:
    """A part of a pack ("meta", "metrics", "index"), read alone (mmap)."""
    import mmap
    f = _pack_footer(path)
    at, n = f[name]
    if n == 0:
        return b""
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        part = mm[at:at + n]
    if name == "index" and crc32c(part) != f["index_crc"]:
        raise ValueError(f"{Path(path).name}: the index's checksum does not hold")
    return part


def _pack_index(path: Path) -> list:
    """[(key, Loc)] of a pack, from its index."""
    path = Path(path)
    st = path.stat()
    tag = (str(path), st.st_mtime_ns, st.st_size)
    if tag not in _PACK_CACHE:
        c = _codec()
        if c is not None:
            with _Mapped(path) as mm:
                _PACK_CACHE[tag] = [(k, Loc(path, off, n, crc, st))
                                    for k, off, n, crc, st in _lib(c.store_pack_entries, mm, path.name)]
        else:
            _PACK_CACHE[tag] = _index_entries(path, _pack_part(path, "index"))
    return _PACK_CACHE[tag]


def _index_entries(path: Path, part: bytes) -> list:
    if part[:1] == b"T":
        c = _codec()
        if c is None:
            raise ValueError(f"{path.name}: its index needs the compiled extension")
        lines = bytes(c.store_decode_text(part[1:]))
    else:
        lines = part[1:]
    out = []
    for line in lines.decode("utf-8").splitlines():
        off, length, crc, status, key = line.split("\t", 4)
        out.append((key, Loc(path, int(off), int(length), int(crc, 16), status)))
    return out


def check_pack(path: Path) -> list:
    """What is wrong with a pack, and where: the footer, then every part by its
    own checksum — the meta table, the metrics table, the index, each record
    (its key and offset) — and the zero bytes between them; where only the
    file's checksum fails, the header or the footer's checksum."""
    path = Path(path)
    name = path.name
    c = _codec()
    if c is not None:                       # the library's (pack.hpp)
        if path.stat().st_size == 0:
            return [f"{name}: shorter than a pack's footer"]
        with _Mapped(path) as mm:
            return list(c.store_check_pack(mm, name))
    try:
        f = _pack_footer(path)
    except ValueError as e:
        return [str(e)]
    data = path.read_bytes()
    bad = []
    parts = []                                          # (start, end) of every part, in order
    for part in ("meta", "metrics"):
        at, n = f[part]
        parts.append((at, at + n))
        if n == 0:
            continue
        piece = data[at:at + n]
        magic = META_MAGIC if part == "meta" else METRICS_MAGIC
        if piece[:8] != magic or len(piece) < 12 or crc32c(piece[12:]) != struct.unpack_from("<I", piece, 8)[0]:
            bad.append(f"{name}@{at}: the {part} table is damaged (its checksum)")
    at, n = f["index"]
    if at + n > f["end"] or crc32c(data[at:at + n]) != f["index_crc"]:
        bad.append(f"{name}@{at}: the index is damaged (its checksum)")
        entries = []
    else:
        entries = _index_entries(path, data[at:at + n])
    parts.append((at, at + n))
    for key, loc in entries:
        parts.append((loc.offset, loc.offset + loc.length))
        got = _parse_record(data[loc.offset:loc.offset + loc.length], 0)
        if got is None or got[0] != key or got[2] != loc.crc:
            bad.append(f"{name}@{loc.offset}: the record of {key} is damaged (its checksum)")
    parts.sort()
    for (_, end), (start, _) in zip(parts, parts[1:] + [(f["end"], 0)]):
        if end <= start and any(data[end:start]):
            first = end + next(i for i, b in enumerate(data[end:start]) if b)
            bad.append(f"{name}@{first}: a byte that should be zero (between the parts)")
    if crc32c(data[:f["end"]]) != f["crc"] and not bad:
        bad.append(f"{name}: the header (bytes 0-{parts[0][0]}) or the footer's checksum is damaged")
    return bad


# ── the meta table of a pack (format 3; supplement 5, §2.3) ─────────────────
#
# meta.json of every run of a pack as a table: a run's JSON is a skeleton
# (its dicts and lists, every leaf a 0) and its leaves in depth-first order,
# each in the column of its path (the keys and list positions down to it). A
# column keeps its leaves' types (0 null, 1 false, 2 true, 3 int, 4 float,
# 5 text) and its ints, floats and texts apart, each by its codec: ints and
# types "i64", floats "f64", texts a dictionary of the column's distinct
# texts and an index each. The skeletons are a dictionary too: most runs
# share one. A run whose text json.dumps(..., indent=1) does not give back
# (its CR LF line ends are kept) is kept as its text.
#
#     magic "MOOTMET1" | u32 CRC-32C of the body | body
#     body: pieces, each a varint length and its bytes —
#       header (text codec: JSON {"rows", "columns"}), keys (text codec: JSON),
#       skeletons (text codec: JSON), each row's skeleton (i64), CR LF (i64),
#       the texts kept whole (text codec: JSON {row: text}),
#       then per column: types, ints (i64), floats (f64), the texts' indices
#       (i64), the column's texts (text codec: JSON)

META_MAGIC = b"MOOTMET1"
METRICS_MAGIC = b"MOOTMTR1"
_META_CACHE: dict = {}


def _put_piece(out: list, data: bytes) -> None:
    n = len(data)
    head = bytearray()
    while n >= 0x80:
        head.append((n & 0x7F) | 0x80)
        n >>= 7
    head.append(n)
    out.append(bytes(head) + data)


def _pieces(body: bytes):
    at = 0
    while at < len(body):
        n, shift = 0, 0
        while True:
            b = body[at]
            at += 1
            n |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                break
        yield body[at:at + n]
        at += n


def _leaves(v, path: tuple, out: list):
    """v's skeleton; its leaves appended to out as (path, value)."""
    if isinstance(v, dict):
        return {k: _leaves(x, path + (k,), out) for k, x in v.items()}
    if isinstance(v, list):
        return [_leaves(x, path + ([i],), out) for i, x in enumerate(v)]
    out.append((json.dumps(path), v))
    return 0


def _fill(skel, path: tuple, take):
    if isinstance(skel, dict):
        return {k: _fill(x, path + (k,), take) for k, x in skel.items()}
    if isinstance(skel, list):
        return [_fill(x, path + ([i],), take) for i, x in enumerate(skel)]
    return take(json.dumps(path))


def encode_meta_table(rows: list) -> bytes:
    """[(key, meta.json bytes)] as a pack's table (the comment above)."""
    import numpy as np
    c = _codec()
    keys, skels, skel_ids, crlf, whole = [], {}, [], [], {}
    cols: dict = {}
    for r, (key, text) in enumerate(rows):
        keys.append(key)
        cr = b"\r\n" in text
        crlf.append(int(cr))
        try:
            obj = json.loads(text)
            ok = _meta_text(obj, cr) == text
        except ValueError:
            ok = False
        leaves: list = []
        if ok:
            skel = json.dumps(_leaves(obj, (), leaves), separators=(",", ":"))
            ok = all(v is None or isinstance(v, (bool, float, str)) or
                     (isinstance(v, int) and -2 ** 63 <= v < 2 ** 63) for _, v in leaves)
        if not ok:
            whole[r] = text.decode("utf-8")
            skel_ids.append(-1)
            continue
        skel_ids.append(skels.setdefault(skel, len(skels)))
        for path, v in leaves:
            col = cols.setdefault(path, {"types": [], "ints": [], "floats": [], "texts": [], "dict": {}})
            if v is None:
                col["types"].append(0)
            elif isinstance(v, bool):
                col["types"].append(2 if v else 1)
            elif isinstance(v, int):
                col["types"].append(3)
                col["ints"].append(v)
            elif isinstance(v, float):
                col["types"].append(4)
                col["floats"].append(v)
            else:
                col["types"].append(5)
                col["texts"].append(col["dict"].setdefault(v, len(col["dict"])))
    def text_piece(obj):
        return bytes(c.store_encode_text(json.dumps(obj).encode()))

    def ints(v):
        return bytes(c.store_encode_i64(np.asarray(v, np.int64)))
    out: list = []
    _put_piece(out, text_piece({"rows": len(rows), "columns": list(cols)}))
    _put_piece(out, text_piece(keys))
    _put_piece(out, text_piece(list(skels)))
    _put_piece(out, ints(skel_ids))
    _put_piece(out, ints(crlf))
    _put_piece(out, text_piece({str(r): t for r, t in whole.items()}))
    for col in cols.values():
        _put_piece(out, ints(col["types"]))
        _put_piece(out, ints(col["ints"]))
        _put_piece(out, bytes(c.store_encode_f64(np.asarray(col["floats"], np.float64))))
        _put_piece(out, ints(col["texts"]))
        _put_piece(out, text_piece(list(col["dict"])))
    body = b"".join(out)
    return META_MAGIC + struct.pack("<I", crc32c(body)) + body


def decode_meta_table(data: bytes) -> dict:
    """{key: meta.json bytes} of a pack's table; ValueError where it is damaged."""
    import numpy as np
    c = _codec()
    if c is None:
        raise ValueError("a pack's meta table needs the compiled extension")
    if data[:8] != META_MAGIC or len(data) < 12:
        raise ValueError("not a pack's meta table")
    (crc,) = struct.unpack_from("<I", data, 8)
    body = data[12:]
    if crc32c(body) != crc:
        raise ValueError("a pack's meta table: its checksum does not hold")
    pieces = list(_pieces(body))

    def text(i):
        return json.loads(bytes(c.store_decode_text(pieces[i])))
    head = text(0)
    n = head["rows"]
    keys, skels, whole = text(1), text(2), text(5)
    skel_ids = c.store_decode_i64(pieces[3], n).tolist()
    crlf = c.store_decode_i64(pieces[4], n).tolist()
    # the columns' leaves, in row order: types first, then each kind's values
    cols, at = {}, 6
    counts: dict = {}
    for r, sid in enumerate(skel_ids):           # how many leaves each column holds
        if sid < 0:
            continue
        _fill(json.loads(skels[sid]), (), lambda path: counts.__setitem__(path, counts.get(path, 0) + 1))
    for path in head["columns"]:
        types = c.store_decode_i64(pieces[at], counts.get(path, 0)).tolist()
        n_int, n_float, n_text = types.count(3), types.count(4), types.count(5)
        cols[path] = {
            "types": iter(types),
            "ints": iter(c.store_decode_i64(pieces[at + 1], n_int).tolist()),
            "floats": iter(np.asarray(c.store_decode_f64(pieces[at + 2], n_float)).tolist()),
            "texts": iter(c.store_decode_i64(pieces[at + 3], n_text).tolist()),
            "dict": text(at + 4),
        }
        at += 5

    def take(path):
        col = cols[path]
        t = next(col["types"])
        if t == 0:
            return None
        if t in (1, 2):
            return t == 2
        if t == 3:
            return int(next(col["ints"]))
        if t == 4:
            return float(next(col["floats"]))
        return col["dict"][next(col["texts"])]
    out = {}
    for r, (key, sid) in enumerate(zip(keys, skel_ids)):
        if sid < 0:
            out[key] = whole[str(r)].encode("utf-8")
        else:
            out[key] = _meta_text(_fill(json.loads(skels[sid]), (), take), bool(crlf[r]))
    return out


def _pack_metas(raw: Path):
    """{key: meta.json bytes} of the pack whose data is raw, or None."""
    raw = Path(raw)
    p = raw if raw.suffix == PACK_SUFFIX else raw.with_suffix(".meta")
    if not raw.name.startswith("pack-") or not p.exists():
        return None
    st = p.stat()
    tag = (str(p), st.st_mtime_ns, st.st_size)
    if tag not in _META_CACHE:
        _META_CACHE.clear()
        data = _pack_part(p, "meta") if p.suffix == PACK_SUFFIX else p.read_bytes()
        _META_CACHE[tag] = decode_meta_table(data) if data else {}
    return _META_CACHE[tag]


def _with_meta(key: str, blocks: list, raw: Path) -> list:
    """A pack's record with its meta.json put back at its mark."""
    if not any(b.name == "#meta" for b in blocks):
        return blocks
    metas = _pack_metas(raw)
    if metas is None or key not in metas:
        raise ValueError(f"{raw.name}: {key}: its meta.json is not in the pack's table")
    blocks = [Block("meta.json", "json", (), metas[key]) if b.name == "#meta" else b for b in blocks]
    return _final_back(blocks)


def read_record_bytes(loc: Loc) -> bytes:
    with open(loc.raw, "rb") as fh:
        fh.seek(loc.offset)
        return fh.read(loc.length)


def log(store: Path, line: str) -> None:
    with open(Path(store) / LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + line + "\n")


def repair(store: Path) -> list:
    """Cut every segment's unfinished tail off, rebuild an index that
    disagrees with its data, remove the scratch folders of runs never packed;
    [what was done]. Only while no process writes. (Not needed to read: a
    tail is never read, and a new process starts a segment of its own.)"""
    done = []
    scratch = Path(store) / SCRATCH_DIR
    if scratch.is_dir() and any(scratch.iterdir()):
        shutil.rmtree(scratch, ignore_errors=True)
        done.append(f"{SCRATCH_DIR}/: removed the folders of runs stopped before their record")
    for raw in segments(store) + [p for p in packs(store) if p.suffix != PACK_SUFFIX]:
        size = raw.stat().st_size
        listed = _read_idx(raw.with_suffix(".idx"), raw)
        # an index that agrees with the data: only the tail past it is looked
        # at (a resumed atlas does not read its hundreds of GiB again)
        entries, end = ((listed, listed[-1][1].offset + listed[-1][1].length if listed else 0)
                        if listed is not None else scan_raw(raw))
        if end < size:
            with open(raw, "r+b") as fh:
                fh.truncate(end)
            msg = f"{raw.name}: cut an unfinished record off at {end} ({size - end} bytes)"
            log(store, msg)
            done.append(msg)
        if listed is None:
            write_index(raw, entries)
            msg = f"{raw.name}: index rebuilt from the data ({len(entries)} records)"
            log(store, msg)
            done.append(msg)
    return done


# ── the analysis tools' view of a stored run ────────────────────────────────


class StoredRun:
    """A stored run where the tools take a run folder: `run / "meta.json"`
    gives a StoredFile with is_file(), read_text(), read_bytes(), open().
    Read only; the record is read on first use, a file made when asked for."""

    def __init__(self, loc: Loc, key: str):
        self.loc, self.key = loc, key
        self.rel_dir = rel_dir_of(key)
        self._by = None
        self._made: dict = {}

    def blocks(self) -> dict:
        if self._by is None:
            self._by = {b.name: b for b in read_record(self.loc)}
        return self._by

    def names(self) -> list:
        return file_names(self.blocks())

    def content(self, name: str):
        if name not in self._made:
            self._made[name] = file_content(self.blocks(), name)
        return self._made[name]

    def files(self) -> dict:
        return {name: self.content(name) for name in self.names()}

    def table(self, name: str):
        """A CSV's numbers (float64, rows x columns) where the store keeps them
        as numbers — what parsing its text gives — else None."""
        b = self.blocks().get(name)
        return block_array(b) if b is not None and b.dtype == "<f8" else None

    @property
    def name(self) -> str:
        return self.rel_dir.name

    def __truediv__(self, name: str) -> "StoredFile":
        return StoredFile(self, str(name))

    def __str__(self) -> str:
        return f"store:{self.rel_dir.as_posix()}"

    __repr__ = __str__


class StoredFile:
    def __init__(self, run: StoredRun, name: str):
        self.run, self.name = run, name
        self.suffix = PurePosixPath(name).suffix

    def is_file(self) -> bool:
        return self.name in self.run.names()

    exists = is_file

    def table(self):
        return self.run.table(self.name)

    def _content(self):
        if not self.is_file():
            raise FileNotFoundError(f"{self.run}/{self.name}")
        return self.run.content(self.name)

    def read_text(self, encoding: str = "utf-8") -> str:
        """The text with its line ends read as Path.read_text reads them (CR LF
        and CR as LF); read_bytes() keeps them."""
        c = self._content()
        text = c if isinstance(c, str) else c.decode(encoding)
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def read_bytes(self) -> bytes:
        import numpy as np
        c = self._content()
        if isinstance(c, dict):                          # an .npz: written again
            buf = io.BytesIO()
            np.savez_compressed(buf, **c)
            return buf.getvalue()
        return c.encode("utf-8") if isinstance(c, str) else c

    def open(self, mode: str = "r", encoding: str = "utf-8", **_):
        """The logical content: the text of a .gz too."""
        if "b" in mode:
            return io.BytesIO(self.read_bytes())
        return io.StringIO(self.read_text(encoding))

    def __str__(self) -> str:
        return f"{self.run}/{self.name}"


def as_run_dir(d):
    """A run folder for the tools: a StoredRun as it is, anything else a Path."""
    return d if isinstance(d, StoredRun) else Path(d)


def stored_runs(campaign: Path):
    """[StoredRun] of a campaign's store, sorted by key; [] without one."""
    s = store_dir(campaign)
    if not s.is_dir():
        return []
    return [StoredRun(loc, key) for key, loc in sorted(index(s).items(), key=lambda kv: sort_key(kv[0]))]


# ── layer 2: packs per family ───────────────────────────────────────────────


def _family(problem: str) -> str:
    from .families import family
    return family(problem)


def compact(campaign: Path, *, remove_segments: bool = False, expected: list | None = None) -> dict:
    """Pack the store of a campaign: one pack per family, its records sorted
    by key and copied byte for byte; then the metrics tables (layer 3). The
    segments go only with remove_segments, and only after every record read
    back from its pack is the bytes of the segment's. `expected`: the keys the
    campaign plans, for the list of the missing."""
    s = store_dir(campaign)
    bad = check(campaign)
    if bad["damaged"]:
        raise ValueError("damaged records, run repair first: " + "; ".join(bad["damaged"][:10]))
    shutil.rmtree(s / SCRATCH_DIR, ignore_errors=True)       # after the campaign: nothing runs
    idx = index(s)
    fams: dict = {}
    for key in sorted(idx, key=sort_key):
        fams.setdefault(_family(split_key(key)[0]), []).append(key)
    written = {}
    for fam, keys in sorted(fams.items()):
        path = s / f"pack-{_safe(fam)}{PACK_SUFFIX}"
        records, metas, table, runs = [], [], [], []
        for key in keys:
            loc = idx[key]
            blocks = read_record(loc)
            by = {b.name: b for b in blocks}
            run = StoredRun(loc, key)
            run._by = by
            runs.append(run)
            rec = read_record_bytes(loc)
            if _codec() is not None and record_format(by) >= 3 and "meta.json" in by:
                # format 3: meta.json into the pack's table, a mark in its place
                once = {b.name: b for b in _final_once(blocks)}["meta.json"].data
                table.append((key, once))
                rec = encode_record(key, encode_body(
                    [Block("#meta", "json", (), b"") if b.name == "meta.json" else b for b in blocks]))
            elif rec[:8] == MAGIC_C and _parse_record(rec, 0) is None:
                raise ValueError(f"{key}: not a whole record")
            records.append((key, rec, loc.status))
            metas.append(json.loads(file_content(by, "meta.json")) if "meta.json" in by else {})
        # read from the records, so that the header too is the same whatever
        # segments they came in
        fields = {"format": FORMAT, "format_version": FORMAT_VERSION, "kind": "pack", "family": fam,
                  "records": len(records), "sorted_by": "problem, algorithm, variant, seed, budget",
                  **_common(metas, ("code_stamp", "trajectory_version")),
                  "layout": "header, meta, metrics, records, index; a footer of 64 bytes at the end",
                  "alignment": ALIGN, "byte_order": "little"}
        entries = write_pack(path, fields, encode_meta_table(table) if table else b"",
                             _metrics_section(fam, runs), records)
        written[fam] = len(entries)
        # the pack of before, now in this one
        for old in (s / f"pack-{_safe(fam)}.raw", s / f"metrics-{_safe(fam)}.raw"):
            for suffix in (".raw", ".idx", ".mhd", ".meta"):
                old.with_suffix(suffix).unlink(missing_ok=True)
    removed = []
    if remove_segments:
        new = index(s)
        for key, loc in idx.items():
            if loc.raw.name.startswith("seg-") and not _same_blocks(read_record(new[key]), read_record(loc)):
                raise ValueError(f"{key}: the pack's record is not the segment's; nothing removed")
        for raw in segments(s):
            for suffix in (".raw", ".idx", ".mhd"):
                raw.with_suffix(suffix).unlink(missing_ok=True)
            removed.append(raw.name)
        log(s, f"compact: removed {len(removed)} segments after checking every record")
    missing = sorted(set(expected or ()) - set(idx))
    failed = sorted(k for k, loc in idx.items() if loc.status != "done")
    return {"packs": written, "removed": removed, "missing": missing, "failed": failed}


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _common(metas: list, fields: tuple) -> dict:
    """Each field's values over the runs' meta.json, sorted and joined."""
    return {f: ",".join(sorted({str(m[f]) for m in metas if m.get(f) is not None})) for f in fields}


def check(campaign: Path) -> dict:
    """Every record of the store read back (length, key hash, CRC): {records,
    damaged: [where], tails: [an unfinished tail], duplicates}."""
    s = store_dir(campaign)
    out = {"records": 0, "damaged": [], "tails": [], "duplicates": 0}
    seen = set()
    for raw in packs(s) + segments(s):
        if raw.suffix == PACK_SUFFIX:
            bad = check_pack(raw)
            out["damaged"] += bad
            out["records"] += 0 if bad else len(_pack_index(raw))
            continue
        entries, end = scan_raw(raw)
        if end < raw.stat().st_size:
            out["tails"].append(f"{raw.name}: {raw.stat().st_size - end} bytes after the last whole record")
        listed = segment_entries(raw)
        if len(listed) != len(entries):
            out["damaged"].append(f"{raw.name}: index {len(listed)} records, data {len(entries)}")
        for key, loc in entries:
            out["records"] += 1
            if raw.name.startswith("seg-"):
                out["duplicates"] += key in seen
                seen.add(key)
    return out


def _same_blocks(a: list, b: list) -> bool:
    return [(x.name, x.dtype, tuple(x.shape), x.data) for x in a] == \
        [(x.name, x.dtype, tuple(x.shape), x.data) for x in b]


def merge(out_campaign: Path, parts: list) -> dict:
    """The stores of campaign parts (other machines' --shard) into one: their
    segments and packs copied in; a run in two parts must be the same bytes
    in both, else ValueError listing them."""
    dst = store_dir(out_campaign)
    dst.mkdir(parents=True, exist_ok=True)
    have = {}
    for key, loc in index(dst).items():
        have[key] = read_record(loc)
    clash, copied = [], 0
    for part in parts:
        src = store_dir(part)
        for key, loc in index(src).items():
            blocks = read_record(loc)
            if key in have and not _same_blocks(have[key], blocks):
                clash.append(f"{key}: {part} differs")
            have.setdefault(key, blocks)
    if clash:
        raise ValueError(f"{len(clash)} runs differ between the parts: " + "; ".join(clash[:20]))
    w = SegmentWriter(dst, header={"merged_from": ",".join(str(p) for p in parts)})
    mine = set(index(dst))
    for key in sorted(have, key=sort_key):
        if key in mine:
            continue
        w.append(key, have[key])
        copied += 1
    return {"runs": len(have), "copied": copied}


# ── layer 3: the metrics tables ─────────────────────────────────────────────


_KEY_COLUMNS = ("problem", "algorithm", "variant", "status")


def _records(run, names) -> list:
    """The trajectory records of a stored run, as dicts: from its columns
    (format 2: each record's keys in its own order, as its line had them)
    without the text, else from its lines."""
    got = trajectory_columns(run.blocks()) if "trajectory.jsonl" in names else None
    if got is None:
        text = run.content("trajectory.jsonl") if "trajectory.jsonl" in names else ""
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    cols, C, V = got
    out = []
    for c, v in zip(C.tolist(), V.tolist()):
        out.append({cols[j]: (None if code == _NULL else bool(v[j]) if code == _BOOL
                              else int(v[j]) if code == _INT else v[j])
                    for j, code in enumerate(c) if code not in (_ABSENT, _TEXT)})
    return out


def _metrics_section(family: str, runs: list) -> bytes:
    """The metrics table of a pack: one row per trajectory record of every run
    — the key's parts, seed, budget, then every number the records hold (bools
    as 0/1, null as NaN; text left out) — column after column: "MOOTMTR1",
    the CRC-32C of the body, the body: pieces, a directory (JSON: family,
    rows, columns [name, dtype]) and each column's block (encode_block)."""
    import numpy as np
    rows, cols = _metric_rows(runs)
    out: list = []
    blocks = []
    for c in _KEY_COLUMNS:
        labels = sorted({r[c] for r in rows})
        code = {v: i for i, v in enumerate(labels)}
        blocks.append(Block(f"{c}#labels", "json", (), json.dumps(labels).encode()))
        blocks.append(Block(c, "<i4", (len(rows),), np.asarray([code[r[c]] for r in rows], "<i4").tobytes()))
    for c in ("seed", "budget", *cols):
        a = np.asarray([r.get(c, np.nan) for r in rows], "<f8")
        blocks.append(Block(c, "<f8", a.shape, a.tobytes()))
    _put_piece(out, json.dumps({"family": family, "rows": len(rows),
                                "columns": [[b.name, b.dtype] for b in blocks]}).encode())
    for b in blocks:
        _put_piece(out, encode_block(b))
    body = b"".join(out)
    return METRICS_MAGIC + struct.pack("<I", crc32c(body)) + body


def _metrics_from_section(data: bytes) -> dict:
    """{column name: Block} of a pack's metrics part."""
    if data[:8] != METRICS_MAGIC or len(data) < 12:
        raise ValueError("not a pack's metrics table")
    body = data[12:]
    if crc32c(body) != struct.unpack_from("<I", data, 8)[0]:
        raise ValueError("a pack's metrics table: its checksum does not hold")
    pieces = list(_pieces(body))
    head = json.loads(pieces[0])
    cols = {}
    for (name, _), piece in zip(head["columns"], pieces[1:]):
        (b,) = decode_blocks(struct.pack("<I", 1) + piece)
        cols[name] = b
    return cols


def _metric_rows(runs: list) -> tuple:
    rows, cols = [], []
    for run in runs:
        problem, algorithm, variant, seed, budget = split_key(run.key)
        names = run.names()
        status = (str(json.loads(run.content("meta.json")).get("status", "?"))
                  if "meta.json" in names else "?")
        for rec in _records(run, names):
            row = {"problem": problem, "algorithm": algorithm, "variant": variant, "status": status,
                   "seed": seed, "budget": budget}
            for k, v in rec.items():
                if isinstance(v, bool):
                    v = float(v)
                if v is None:
                    v = float("nan")
                if isinstance(v, (int, float)):
                    row[k] = v
                    if k not in cols:
                        cols.append(k)
            rows.append(row)
    return rows, cols


def write_metrics(store: Path, family: str, runs: list) -> Path:
    """metrics-<family>.mhd/.raw, the metrics table as a file of its own (the
    layout of before the pack file; _metrics_section is the pack's)."""
    import numpy as np
    rows, cols = _metric_rows(runs)
    raw = Path(store) / f"metrics-{_safe(family)}.raw"
    tmp = raw.with_suffix(f".raw.{os.getpid()}.tmp")
    fields = {"format": FORMAT, "format_version": FORMAT_VERSION, "kind": "metrics", "family": family,
              "data_file": raw.name, "rows": len(rows)}
    with open(tmp, "wb") as fh:
        for c in _KEY_COLUMNS:
            labels = sorted({r[c] for r in rows})
            code = {s: i for i, s in enumerate(labels)}
            for name, b in ((f"{c}#labels", Block(f"{c}#labels", "json", (), json.dumps(labels).encode())),
                            (c, Block(c, "<i4", (len(rows),),
                                      np.asarray([code[r[c]] for r in rows], "<i4").tobytes()))):
                fields[f"column.{name}"] = f"{b.dtype} {fh.tell()} {fh.write(encode_block(b))}"
        for c in ("seed", "budget", *cols):
            a = np.asarray([r.get(c, np.nan) for r in rows], "<f8")
            b = Block(c, "<f8", a.shape, a.tobytes())
            fields[f"column.{c}"] = f"<f8 {fh.tell()} {fh.write(encode_block(b))}"
    os.replace(tmp, raw)
    _write_header(raw.with_suffix(".mhd"), fields)
    return raw


def metrics(path, family: str | None = None, *, frame: bool = False):
    """The metrics table of a family: {column: NumPy array} (problem,
    algorithm, variant and status as text), or a pandas DataFrame with
    frame=True. `path`: a campaign, its store, or a metrics-<family>.mhd;
    family=None with a campaign takes every family, one table."""
    import numpy as np
    path = Path(path)
    if path.suffix in (".mhd", PACK_SUFFIX):
        mhds = [path]
    else:
        s = path if path.name == STORE_DIR else store_dir(path)
        mhds = _metric_files(s, family)
    parts = []
    for mhd in mhds:
        if mhd.suffix == PACK_SUFFIX:
            cols = _metrics_from_section(_pack_part(mhd, "metrics"))
        else:
            h = read_header(mhd)
            buf = mhd.with_name(h["data_file"]).read_bytes()
            cols = {}
            for k, v in h.items():
                if not k.startswith("column."):
                    continue
                _, off, n = v.split()
                (b,) = decode_blocks(struct.pack("<I", 1) + buf[int(off):int(off) + int(n)])
                cols[k[len("column."):]] = b
        table = {}
        for c in _KEY_COLUMNS:
            labels = json.loads(cols.pop(f"{c}#labels").data)
            codes = block_array(cols.pop(c))
            table[c] = np.asarray(labels, object)[codes] if len(codes) else np.asarray([], object)
        for c, b in cols.items():
            table[c] = block_array(b).copy()
        parts.append(table)
    names = []
    for t in parts:
        names += [c for c in t if c not in names]
    out = {}
    for c in names:
        out[c] = np.concatenate([t[c] if c in t else np.full(len(t["problem"]), np.nan) for t in parts]) \
            if parts else np.asarray([])
    if frame:
        import pandas as pd
        return pd.DataFrame(out)
    return out


def _metric_files(s: Path, family: str | None = None) -> list:
    """The files that hold a family's metrics table (every family's with
    None): its pack, or the metrics-<family>.mhd of before."""
    names = [_safe(family)] if family else None
    out = []
    for f in sorted(s.glob(f"pack-*{PACK_SUFFIX}")):
        if names is None or f.stem[len("pack-"):] in names:
            out.append(f)
    have = {f.stem[len("pack-"):] for f in out}
    for f in sorted(s.glob("metrics-*.mhd")):
        fam = f.stem[len("metrics-"):]
        if fam not in have and (names is None or fam in names):
            out.append(f)
    return out


def export_csv(campaign: Path, out_dir: Path) -> list:
    """metrics-<family>.csv.gz of every family's table (gzip, no time stamp)."""
    import csv
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for mhd in _metric_files(store_dir(campaign)):
        t = metrics(mhd)
        fam = mhd.stem.split("-", 1)[1]
        path = out_dir / f"metrics-{fam}.csv.gz"
        with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
            w = csv.writer(text, lineterminator="\n")
            w.writerow(list(t))
            for i in range(len(t["problem"])):
                w.writerow([t[c][i] if c in _KEY_COLUMNS else repr(float(t[c][i])) for c in t])
            text.flush()
            text.detach()
        written.append(path)
    return written


def export_files(campaign: Path, out: Path, keys: list | None = None) -> int:
    """The files layout of the stored runs (all, or `keys`) under out."""
    n = 0
    for run in stored_runs(campaign):
        if keys is not None and run.key not in keys:
            continue
        write_run_files(run.files(), Path(out) / run.rel_dir)
        n += 1
    return n


# ── the command: mootation store ... ────────────────────────────────────────


def main(argv: list | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="mootation store",
                                 description="the result store of a campaign (task 16)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("ls", "check", "repair", "compact", "merge", "export"):
        s = sub.add_parser(cmd)
        s.add_argument("campaign", help="the campaign's results folder")
        if cmd == "compact":
            s.add_argument("--remove-segments", action="store_true",
                           help="delete the segments once the packs are checked")
        if cmd == "merge":
            s.add_argument("parts", nargs="+", help="the parts' results folders")
        if cmd == "export":
            s.add_argument("--csv", default="", help="write metrics-<family>.csv.gz here")
            s.add_argument("--files", default="", help="write the files layout of the runs here")
            s.add_argument("--run", action="append", default=[],
                           help="with --files: only this key (problem|algorithm|variant|seed|budget)")
    a = ap.parse_args(argv)
    camp = Path(a.campaign)
    if a.cmd == "ls":
        idx = index(store_dir(camp))
        by: dict = {}
        for key, loc in idx.items():
            fam = _family(split_key(key)[0])
            st = by.setdefault(fam, {})
            st[loc.status] = st.get(loc.status, 0) + 1
        size = sum(f.stat().st_size for f in store_dir(camp).glob("*") if f.is_file())
        for fam, st in sorted(by.items()):
            print(f"{fam:16s} " + ", ".join(f"{k} {v}" for k, v in sorted(st.items())))
        print(f"{len(idx)} runs, {len(segments(store_dir(camp)))} segments, "
              f"{len(packs(store_dir(camp)))} packs, {size / 2**20:.1f} MiB")
        return 0
    if a.cmd == "check":
        got = check(camp)
        print(json.dumps(got, indent=1))
        return 1 if got["damaged"] or got["tails"] else 0
    if a.cmd == "repair":
        for line in repair(store_dir(camp)) or ["nothing to repair"]:
            print(line)
        return 0
    if a.cmd == "compact":
        got = compact(camp, remove_segments=a.remove_segments, expected=_planned(camp))
        print(json.dumps({k: (v if k == "packs" else len(v)) for k, v in got.items()}, indent=1))
        for k in got["missing"][:50]:
            print("missing:", k)
        return 0
    if a.cmd == "merge":
        print(json.dumps(merge(camp, [Path(p) for p in a.parts]), indent=1))
        return 0
    if a.cmd == "export":
        if a.csv:
            for p in export_csv(camp, Path(a.csv)):
                print(p)
        if a.files:
            print(export_files(camp, Path(a.files), a.run or None), "runs written")
        return 0
    return 2


def _planned(camp: Path) -> list | None:
    """The keys the campaign's config plans (campaign.json names it), or None."""
    try:
        info = json.loads((Path(camp) / "campaign.json").read_text(encoding="utf-8"))
        from .run.config import load, validate
        from .run.campaign import campaign_spec, expand_jobs
        cfg = load(info["config"])
        validate(cfg)
        return [job_key(j) for j in expand_jobs(cfg, campaign_spec(cfg))]
    except (OSError, KeyError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
