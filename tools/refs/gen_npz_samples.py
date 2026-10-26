# SPDX-License-Identifier: Apache-2.0
"""tests/data/npz/: what NumPy and zlib write, the reference of tests/test_npz.cpp
(include/mootation/io/npz.hpp, campaign/references.hpp).

    entry.npz       numpy.savez_compressed of a reference cache entry's arrays
                    (meta, ideal, nadir, ref, pset), as run/refs.py writes one
    stored.npz      numpy.savez (stored members): an int64 array and a
                    Fortran-ordered float64 one
    stored.deflate, fixed.deflate, dyn.deflate
                    one text deflated raw by zlib in each of the three block
                    types (level 0; Z_FIXED; level 9)

It prints what the test checks: the arrays' reference_hash and the text's
length and CRC-32.

    python tools/refs/gen_npz_samples.py
"""
import json
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mootation.run.campaign import reference_hash

out = Path(__file__).resolve().parents[2] / "tests" / "data" / "npz"
out.mkdir(parents=True, exist_ok=True)
ideal = np.array([0.0, 0.0, 0.0])
nadir = np.array([1.0, 1.0, 1.5])
t = np.linspace(0, 1, 40)
ref = np.column_stack([t, 1 - t, np.sqrt(t)])
pset = np.arange(24, dtype=float).reshape(8, 3) / 7.0
meta = {"problem": "Tést", "n_ref": 40, "info3": None}
np.savez_compressed(out / "entry.npz", meta=np.asarray(json.dumps(meta)), ideal=ideal, nadir=nadir,
                    ref=ref, pset=pset)
np.savez(out / "stored.npz", ints=np.arange(5, dtype=np.int64), F=np.asfortranarray(np.arange(6.0).reshape(2, 3)))

text = (bytes(range(256)) * 3 + b"abcabcabc" * 50
        + b",".join(str(i * i % 9973).encode() for i in range(3000)))
for name, level, strategy in (("stored", 0, zlib.Z_DEFAULT_STRATEGY), ("fixed", 9, zlib.Z_FIXED),
                              ("dyn", 9, zlib.Z_DEFAULT_STRATEGY)):
    c = zlib.compressobj(level, zlib.DEFLATED, -15, 9, strategy)
    data = c.compress(text) + c.flush()
    (out / f"{name}.deflate").write_bytes(data)
    print(name, "first block type", (data[0] >> 1) & 3)
print("reference_hash", reference_hash(SimpleNamespace(ideal=ideal, nadir=nadir), ref, None, pset))
print("text", len(text), "bytes, crc32", zlib.crc32(text))
print("ref[1]", repr(ref[1, 0]), repr(ref[1, 1]), repr(ref[1, 2]))
