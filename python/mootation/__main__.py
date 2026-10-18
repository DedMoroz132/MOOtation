# SPDX-License-Identifier: Apache-2.0
"""The console script `mootation` (python -m mootation):

    mootation refs build|check ...    the cache of references (mootation.run.refs)

Campaigns have their own: python -m mootation.run.campaign, mootation-run.
"""

import sys


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] != "refs":
        print("usage: mootation refs {build,check} ...  (python -m mootation.run.refs -h)",
              file=sys.stderr)
        return 2
    from .run.refs import main as refs_main
    return refs_main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
