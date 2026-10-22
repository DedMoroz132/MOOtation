# SPDX-License-Identifier: Apache-2.0
"""The console script `mootation` (python -m mootation):

    mootation refs build|check ...    the cache of references (mootation.run.refs)
    mootation selftest [-v]           this build gives the library's runs bit for bit?
                                      (mootation.selftest(), task 14 of 2026-10-20, §7)
    mootation store ls|check|repair|compact|merge|export <campaign> ...
                                      a campaign's result store (mootation.store, task 16)
    mootation report <campaign>       one page on a campaign (run.campaign_report)

Campaigns have their own: python -m mootation.run.campaign, mootation-run.
"""

import sys


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "selftest":
        from ._selftest import main as selftest_main
        return selftest_main(argv[1:])
    if argv and argv[0] == "report":
        from .run.campaign_report import main as report_main
        return report_main(argv[1:])
    if argv and argv[0] == "store":
        from .store import main as store_main
        return store_main(argv[1:])
    if not argv or argv[0] != "refs":
        print("usage: mootation refs {build,check} ...  (python -m mootation.run.refs -h)\n"
              "       mootation selftest [-v]\n"
              "       mootation store {ls,check,repair,compact,merge,export} <campaign> ...\n"
              "       mootation report <campaign>",
              file=sys.stderr)
        return 2
    from .run.refs import main as refs_main
    return refs_main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
