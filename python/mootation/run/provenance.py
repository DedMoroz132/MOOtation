# SPDX-License-Identifier: Apache-2.0
"""Which code produced a result: the source revision, and the binary's.

`mootation.__version__` stays 0.1.0 across every commit, so on its own it cannot
tell a run made before a fix from one made after it. Every campaign result
therefore records two revisions:

  * the SOURCE — `git rev-parse HEAD` of the checkout this package was imported
    from, and whether tracked files differ from it. A pip-installed package has
    no .git; its source is then the revision stamped into the extension at
    build time, which is also when pip copied the Python files.
  * the CORE — the revision the compiled extension was built from, stamped by
    python/git_stamp.cmake. In a checkout the two can disagree: a pulled commit
    whose C++ was never rebuilt runs old algorithms under a new commit hash,
    and this is how that shows.

Both are "unknown" rather than guessed when nothing can be asked. Untracked
files never make a tree dirty: a scratch file next to the code is not a change
to the code that ran.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path


def _git(root: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                           text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _core_stamp() -> tuple[str, bool | None]:
    try:
        from .. import _core
    except ImportError:
        return "unknown", None
    commit = str(getattr(_core, "__git_commit__", "unknown") or "unknown")
    dirty = getattr(_core, "__git_dirty__", None)
    return commit, (bool(dirty) if dirty is not None and commit != "unknown" else None)


@lru_cache(maxsize=1)
def revision() -> dict:
    """{"commit", "dirty", "source", "core_commit", "core_dirty"}; cached per process.

    `source` says where `commit` came from: "git" (the checkout, live),
    "build" (the extension's stamp, for an installed package) or "unknown".
    """
    core_commit, core_dirty = _core_stamp()
    root = Path(__file__).resolve().parents[3]          # python/mootation/run -> repo
    commit = _git(root, "rev-parse", "HEAD") if (root / ".git").exists() else None
    if commit:
        status = _git(root, "status", "--porcelain", "--untracked-files=no")
        return {"commit": commit, "dirty": (bool(status) if status is not None else None),
                "source": "git", "core_commit": core_commit, "core_dirty": core_dirty}
    if core_commit != "unknown":
        return {"commit": core_commit, "dirty": core_dirty, "source": "build",
                "core_commit": core_commit, "core_dirty": core_dirty}
    return {"commit": "unknown", "dirty": None, "source": "unknown",
            "core_commit": core_commit, "core_dirty": core_dirty}
