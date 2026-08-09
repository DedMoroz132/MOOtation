# SPDX-License-Identifier: Apache-2.0
"""The journal: what was evaluated, and what came back (TUI_SPEC.md §4).

Scratch and journal are two stores with different properties, and mixing them
is the source of both disk bloat and the inability to resume:

              scratch                       journal
    what      meshes, VTU, solver temps     x, f, status
    grain     per worker                    per run
    mode      overwritten                   append-only
    size      gigabytes                     ~300 bytes per evaluation
    path      scratch/worker_03/            results/<run>/evaluations.jsonl

A hundred thousand evaluations is about 30 MB. Next to the VTU files, nothing.

Resume works off the JOURNAL, not the scratch: knowing that x maps to f does
not require the mesh that produced it.

The journal is also a cache, and that is not a small side benefit —
evolutionary algorithms re-evaluate identical individuals far more often than
one expects. The key is a hash of the decision vector, quantized so that two
vectors that differ only in float noise hash alike.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterator


# Decision vectors are hashed at this many significant digits. 12 is far below
# double precision (~15-17) yet far above anything an optimizer meaningfully
# distinguishes, so a re-evaluation of "the same" point hits the cache while
# genuinely different points never collide.
_HASH_DIGITS = 12


def hash_x(x: list[float]) -> str:
    """Stable content hash of a decision vector."""
    payload = ",".join(f"{v:.{_HASH_DIGITS}g}" for v in x)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


@dataclass
class Record:
    gen: int
    idx: int
    x: list[float]
    f: list[float]
    cv: float = 0.0
    status: str = "ok"          # ok | failed
    t: float = 0.0              # seconds spent evaluating
    hash: str = ""

    def __post_init__(self):
        if not self.hash:
            self.hash = hash_x(self.x)


class Ledger:
    """Append-only JSONL journal with a hash index for resume and caching.

    Opened for append, flushed per record. The cost of an fsync-free flush is
    negligible against a solver call, and the benefit is that a run killed with
    Ctrl-C loses nothing but the evaluation in flight.
    """

    def __init__(self, path: str | os.PathLike, *, resume: bool = True):
        self.path = Path(path)
        # The parent directory is created on the first append, NOT here.
        # Opening a journal is what `--check` does to report how much of a run
        # can be resumed, and a read-only command that leaves directories
        # behind is a command nobody trusts to be read-only.
        self._index: dict[str, Record] = {}
        self._fh = None
        if resume and self.path.is_file():
            self._load()

    # ── reading ─────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Index an existing journal, tolerating a truncated final line.

        A run killed mid-write leaves a partial last line. Refusing to resume
        because of it would throw away every completed evaluation, so the
        partial line is dropped and the rest is kept.
        """
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    # Only the final line may legitimately be torn.
                    continue
                try:
                    rec = Record(
                        gen=int(raw.get("gen", -1)),
                        idx=int(raw.get("idx", -1)),
                        x=[float(v) for v in raw["x"]],
                        f=[float(v) for v in raw.get("f", [])],
                        cv=float(raw.get("cv", 0.0)),
                        status=str(raw.get("status", "ok")),
                        t=float(raw.get("t", 0.0)),
                        hash=str(raw.get("hash", "")),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                # Only successful evaluations are worth reusing: replaying a
                # failure would make a transient solver crash permanent.
                if rec.status == "ok":
                    self._index[rec.hash] = rec

    def lookup(self, x: list[float]) -> Record | None:
        """A previously computed result for this decision vector, if any."""
        return self._index.get(hash_x(x))

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self) -> Iterator[Record]:
        return iter(self._index.values())

    # ── writing ─────────────────────────────────────────────────────────────

    def append(self, rec: Record) -> None:
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")
        self._fh.write(json.dumps(asdict(rec), separators=(",", ":")) + "\n")
        self._fh.flush()
        if rec.status == "ok":
            self._index[rec.hash] = rec

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── reporting ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Counts for the Monitor screen and for --check --resume."""
        return {
            "cached": len(self._index),
            "path": str(self.path),
            "exists": self.path.is_file(),
            "bytes": self.path.stat().st_size if self.path.is_file() else 0,
        }
