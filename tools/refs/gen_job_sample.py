# SPDX-License-Identifier: Apache-2.0
"""tests/data/job_sample/: a small campaign run by the Python engine, the
reference of tests/test_job_run.cpp (include/mootation/campaign/job_run.hpp):

    campaign.toml       ZDT1 and DTLZ2_3D, nsga2 and moead_de, 600 evaluations,
                        one seed, references of 100 points
    registry.json       the two problems of the exported registry
                        (run/registry_export.py)
    refs/               their entries of the cache of references (run/refs.py)
    expected.jsonl      every job as run/campaign.py's _run_job writes it for
                        the store: its trajectory ("t", the seconds, left
                        out), meta.json (the process's fields left out) and
                        its tables (head, shape, SHA-256 of the float64
                        numbers); one line per job

    python tools/refs/gen_job_sample.py
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

out = Path(__file__).resolve().parents[2] / "tests" / "data" / "job_sample"
shutil.rmtree(out, ignore_errors=True)
(out / "refs").mkdir(parents=True)
os.environ["MOOTATION_REFS_DIR"] = str(out / "refs")       # before the package reads it

from mootation import __version__  # noqa: E402
from mootation.benchmarks import registry as R  # noqa: E402
from mootation.run import campaign as K  # noqa: E402
from mootation.run import config as C  # noqa: E402
from mootation.run.campaign import _code_stamp  # noqa: E402
from mootation.run.registry_export import FORMAT, entry, names  # noqa: E402
import numpy as np  # noqa: E402

# meta.json's fields that are the process's (campaign.py; the C++ engine's Process)
PROCESS = ("mootation", "revision", "host", "platform", "python", "pid", "fp_env", "started", "finished",
           "seconds", "record_seconds", "timing_us")

TEXT = """[run]
name = "sample"

[problem]
kind = "builtin"

[benchmarks]
problems = ["ZDT1", "DTLZ2_3D"]
runs = 1

[campaign]
budget_fe = 600
n_ref = 100

[[algorithms]]
name = "nsga2"
pop = 0
gens = 0

[[algorithms]]
name = "moead_de"
pop = 0
gens = 0
"""
(out / "campaign.toml").write_text(TEXT, encoding="utf-8", newline="\n")
cfg = C.loads(TEXT, source_path=out / "campaign.toml")
assert not C.validate(cfg)
spec = K.campaign_spec(cfg)
jobs = K.expand_jobs(cfg, spec)
doc = {"format": FORMAT, "code_stamp": _code_stamp(), "reference_version": R.REFERENCE_VERSION,
       "mootation": __version__, "names": names(), "problems": [entry(n) for n in ("ZDT1", "DTLZ2_3D")]}
(out / "registry.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
lines = []
with tempfile.TemporaryDirectory() as tmp:
    for j in jobs:
        d = Path(tmp) / j.rel_dir
        tables = {}
        K._run_job(j, Path(tmp), spec, d, quiet=True, tables=tables)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        for k in PROCESS:
            meta.pop(k, None)
        heads = [{"name": name, "head": head, "shape": list(A.shape),
                  "sha256": hashlib.sha256(np.ascontiguousarray(A, "<f8").tobytes()).hexdigest()}
                 for name, (kind, head, A) in tables.items()]
        traj = []
        for x in (d / "trajectory.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(x)
            rec.pop("t", None)
            traj.append(json.dumps(rec, separators=(",", ":")))
        lines.append(json.dumps({"index": j.index, "problem": j.problem, "algorithm": j.key, "lines": traj,
                                 "meta": meta, "tables": heads}))
(out / "expected.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print(out, len(jobs), "jobs", file=sys.stderr)
