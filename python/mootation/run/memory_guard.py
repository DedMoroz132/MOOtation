# SPDX-License-Identifier: Apache-2.0
"""Memory protection of a campaign (task 28, §28.6.2; [campaign] memory_share,
0 = off by default).

A campaign of many workers can take more memory than the machine has (twenty
processes past 31 GB froze one). With memory_share = s in (0, 1] the runner
estimates every run's memory before it starts and keeps the runs that run at
once within s times the free memory:

  * a run whose estimate alone passes s x free is refused: its meta.json says
    failed, with the estimate and the limit (MEM-3);
  * the pool's size is at most floor(s x free / the largest estimate), at
    least 1, at the campaign's start (both engines, MEM-4);
  * the Python engine's pool hands a job to a worker only while the
    estimates of the jobs running and of this one fit s x the free memory
    measured then plus what the running jobs already hold; a job that does
    not fit waits for one to end (a queue), and is run alone when nothing
    else runs (MEM-5).

READINGS (OURS)
  MEM-1. The estimate of a run, in bytes: BASE (300 MB: the interpreter or the
    engine, the library, the problem) + POINT x pop (D + M) x 4 (the
    population, its offspring and copies) + where the run keeps every point
    evaluated (one objective without keep_points = "best"; several with
    all_evaluated) budget x (D + M + 4) x 8 x 3 (the archive's rows, their
    index and its slack). POINT = 8 bytes a double. A rough upper bound,
    not a measurement.
  MEM-2. Free memory: Windows GlobalMemoryStatusEx's ullAvailPhys, Linux
    /proc/meminfo's MemAvailable, elsewhere sysconf's available pages; where
    none answers, the protection is off with a warning.
  MEM-3. A refused run is not an error of the campaign: it counts as failed
    ("refused: memory"), and a later start of the campaign with more memory
    runs it.
  MEM-4. The C++ engine's pool gets the capped worker count only; it has no
    queue of its own (its jobs are those of one size, the campaign's).
  MEM-5. The queue measures the free memory each time it hands out a job:
    other programs' memory counts, and a job waits while they hold it.
"""
from __future__ import annotations

import os
import sys

BASE = 300 * 1024 * 1024
POINT = 8


def free_bytes() -> int | None:
    """The machine's free physical memory in bytes, or None (MEM-2)."""
    if os.name == "nt":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = Status()
        st.dwLength = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullAvailPhys)
        return None
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return None


def keeps_every_point(job, spec) -> bool:
    """MEM-1: the run keeps every point it evaluates."""
    if job.n_objs == 1:
        return str(job.params.get("keep_points", "all")) != "best"
    return bool(getattr(spec, "all_evaluated", False))


def estimate(job, spec, n_vars: int) -> int:
    """MEM-1: a run's memory in bytes, a rough upper bound."""
    D, M = int(n_vars), max(1, int(job.n_objs))
    pop = max(1, int(job.pop or 1))
    b = BASE + POINT * pop * (D + M) * 4
    if keeps_every_point(job, spec):
        b += int(job.budget) * (D + M + 4) * POINT * 3
    return b


def plan(todo: list, spec, n_vars_of) -> tuple[list, list, int, int | None]:
    """(the jobs that fit, the refused ones with their estimates, the largest
    estimate of those that fit, the limit share x free; None: off)."""
    share = float(getattr(spec, "memory_share", 0.0) or 0.0)
    if share <= 0.0:
        return todo, [], 0, None
    free = free_bytes()
    if free is None:
        print("memory_share: the free memory cannot be read here; the protection is off (MEM-2)",
              file=sys.stderr)
        return todo, [], 0, None
    limit = int(share * free)
    fit, refused, largest = [], [], 0
    for j in todo:
        e = estimate(j, spec, n_vars_of(j))
        if e > limit:
            refused.append((j, e))
        else:
            fit.append(j)
            largest = max(largest, e)
    return fit, refused, largest, limit


def capped_workers(workers: int, largest: int, limit: int | None) -> int:
    """MEM-4: at most floor(limit / largest) workers, at least 1."""
    if limit is None or largest <= 0:
        return workers
    return max(1, min(workers, limit // largest))


def mb(b: int) -> str:
    return f"{b / (1024 * 1024):.0f} MB"
