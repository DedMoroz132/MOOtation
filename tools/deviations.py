#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The table of every algorithm's departures from its paper, from the headers.

    python tools/deviations.py           # writes docs/deviations.md
    python tools/deviations.py --check   # exit 1 if docs/deviations.md is not what the headers give

Each core's header (the comments of its file above the class, include/
mootation/algorithms) names its departures from the paper in one of these
ways: an item with an identifier — "SMS-2.", "HLMEA-1 (RESOLVED).", "MOCMA-5
(MINOR)." —; a bullet ("- ...", "* ...", "(a) ...") or an identifier and its
text ("SDR-A duplicates stay: ...") under a heading that says "Deviations"
("DECLARED DEVIATIONS:", "Deviations and readings:"); the text after such a
heading's colon ("DECLARED DEVIATIONS: none." says there is none); "Deviation:"
inside a sentence. All are collected: the algorithm (its key in
algorithms.def), the identifier (an item without one is numbered "<key>.1",
"<key>.2", ... in its order), the first sentence of the item, and whether the
default run departs:

  no: fixed     the item says it was fixed, resolved, retired or removed, or
                that it is not a deviation (NONE, "not a deviation");
  no: option    the item is about something off by default (a knob, an
                option, a mode that has to be asked for);
  yes           otherwise: the item describes what the default run does.

The last column is read from the item's wording, not stated by it: an item
that does not say "off by default", "only when", "knob" and the like counts as
"yes". The header stays the source; this file only gathers it, so the table
and the headers cannot drift apart (CI runs --check).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALGS = ROOT / "include" / "mootation" / "algorithms"
DEF = ROOT / "include" / "mootation" / "algorithms.def"
OUT = ROOT / "docs" / "deviations.md"

# "SMS-2.", "HLMEA-1 (RESOLVED).", "- NSGA3-N1 (reading: ...)": an identifier
# with at least one hyphen and a number or a capital after the last one
ID = r"[A-Z0-9][A-Z0-9]*(?:-[A-Za-z0-9]+)*-(?:[A-Z]?[0-9]+[a-z]?|[A-Z]+)"     # SMS-2, IF-MaOEA-1, 3C-1, SDR-A
# after the identifier a tag in brackets and ".", ":" or ",", or "." or ":" and a space
ITEM_ID = re.compile(rf"^(\s*)(?:[-*•]\s+)?({ID})(?:\s*\(([^)]*)\)[.:,]|[.:](?=\s|$))\s*(.*)$")
ITEM_LOOSE = re.compile(rf"^(\s*)({ID})()\s+(.*)$")         # under a heading: "SDR-A duplicates stay: ..."
BULLET = re.compile(r"^(\s*)(?:[-*•]|\([a-z]\))\s+(.*)$")   # "- ...", "* ...", "(a) ..."
# "Deviations:", "DECLARED DEVIATIONS: none.", "DECLARED DEVIATIONS AND READINGS:"
HEADING = re.compile(r"^(?:DECLARED\s+)?DEVIATIONS?\b[^:]{0,60}:\s*(.*)$", re.I)
# "... Deviation: weak dominance in ..." inside a sentence
INLINE = re.compile(r"\b(?:DECLARED\s+)?Deviations?:\s*(.+)$", re.I)
FIXED_TAG = re.compile(r"\b(fixed|resolved(?! from)|retired|removed|none)\b|not a deviation", re.I)
FIXED_TEXT = re.compile(r"^\(?(fixed|resolved|retired|removed)\b|not a deviation", re.I)
OPTION = re.compile(r"\b(off|disabled) by default\b|\bdefault(s)? (is |are )?(off|false)\b|\bopt-in\b|"
                    r"\bonly when (asked|set|given|enabled|requested)\b|\bnot the default\b", re.I)
# 'survival = "stochastic" (...): ...', 'inner_archive, inner_archive_cap (...): ...'
KNOB_FIRST = re.compile(r'^[a-z][a-z0-9_]* *= *"|^[a-z]+_[a-z0-9_]*(?:, [a-z]+_[a-z0-9_]*)*(?: \(|:(?!:)|,)')


def cores() -> list:
    """(key, core class) of every MOOTATION_ALG line, in algorithms.def's order."""
    return re.findall(r"^MOOTATION_ALG\((\w+),\s*\w+,\s*(\w+)\)", DEF.read_text(encoding="utf-8"), re.M)


def header_of(core: str) -> tuple:
    """The file that defines `core` and its comment lines above the class, the
    "// " taken off."""
    for f in sorted(ALGS.glob("*.hpp")):
        lines = f.read_text(encoding="utf-8").splitlines()
        for n, line in enumerate(lines):
            if re.match(rf"^\s*(class|struct)\s+{core}\b(?!\s*;)", line):
                comments = [l.strip()[2:] for l in lines[:n] if l.strip().startswith("//")]
                return f.name, [c[1:] if c.startswith(" ") else c for c in comments]
    raise SystemExit(f"no class {core} in {ALGS}")


def indent(s: str) -> int:
    return len(s) - len(s.lstrip())


def first_sentence(text: str) -> str:
    """Up to the first "." or ";" that ends a sentence (followed by a capital,
    a bracket or the end; not "Eq.", "Alg.", "i.e." and the like), at least 40
    characters in; at most 220."""
    text = " ".join(text.split()).lstrip(",; ")
    for m in re.finditer(r"[.;](?=\s+[A-Z(§\"]|$)", text):
        if m.start() >= 40 and not re.search(r"\b(Eq|Eqs|Alg|Fig|Sec|Tab|i\.e|e\.g|cf|vs|al|resp)$",
                                             text[:m.start()]):
            text = text[:m.end()]
            break
    return text if len(text) <= 220 else text[:217].rstrip() + "..."


def items(key: str, lines: list) -> tuple:
    """([(identifier, status tag, text)], whether the header says "none") of
    one header.

    An item runs on over the lines indented deeper than its first; an item
    that is the text after a heading's colon (a paragraph) runs on to the next
    blank line, bullet, identifier or heading.
    """
    out, says_none = [], False
    cur = None                    # [identifier, tag, text, indent, is a paragraph]
    under_heading, heading_indent, n = False, 0, 0

    def close():
        nonlocal cur
        if cur:
            out.append(cur)
        cur = None

    def numbered(text, ind, paragraph=False):
        nonlocal cur, n
        close()
        n += 1
        cur = [f"{key}.{n}", "", text, ind, paragraph]

    for line in lines:
        if not line.strip():
            close()
            under_heading = False
            continue
        ind = indent(line)
        m, b, h = ITEM_ID.match(line), BULLET.match(line), HEADING.match(line.strip())
        if not m and under_heading and ind > heading_indent:
            m = ITEM_LOOSE.match(line)
        if m and (cur is None or cur[4] or ind <= cur[3]):
            close()
            cur = [m.group(2), m.group(3) or "", m.group(4), ind, False]
            continue
        if h:
            close()
            under_heading, heading_indent = True, ind
            rest = h.group(1).strip()
            if re.match(r"none\b", rest, re.I):
                says_none = True
            elif rest:
                numbered(rest, ind, paragraph=True)
            continue
        if under_heading and b and ind >= heading_indent and (cur is None or cur[4] or ind <= cur[3]):
            numbered(b.group(2), ind)
            continue
        if cur and (cur[4] or ind > cur[3]):
            cur[2] += " " + line.strip()
            continue
        close()
        if under_heading and ind <= heading_indent:
            under_heading = False
        i = INLINE.search(line)
        if i:
            numbered(i.group(1), ind, paragraph=True)
    close()
    return [(i, tag, text) for i, tag, text, _, _ in out], says_none


def default_column(tag: str, text: str) -> str:
    if FIXED_TAG.search(tag) or FIXED_TEXT.search(text):
        return "no: fixed"
    if KNOB_FIRST.match(text) or OPTION.search(tag + " " + first_sentence(text)):
        return "no: option"
    return "yes"


def table() -> str:
    rows, counts = [], {"yes": 0, "no: option": 0, "no: fixed": 0}
    says_none, none_found = [], []
    for key, core in cores():
        fname, lines = header_of(core)
        got, none = items(key, lines)
        if not got:
            (says_none if none else none_found).append(key)
        for ident, tag, text in got:
            col = default_column(tag, text)
            counts[col] += 1
            note = f" ({tag})" if tag else ""
            summary = first_sentence(text).replace("|", "\\|")
            rows.append(f"| `{key}` | {ident}{note} | {summary} | {col} | `{fname}` |")
    head = [
        "# Departures from the papers",
        "",
        "Written by `tools/deviations.py` from the headers of the cores (`include/mootation/algorithms`);",
        "do not edit by hand — edit the header and run the tool. The header is the source: there each item",
        "is given in full, with its reasons and references.",
        "",
        "*Default run*: `yes` — the default run does what the item describes; `no: option` — the item is",
        "about something off by default; `no: fixed` — fixed, resolved, retired, or not a deviation. The",
        "column is read from the item's wording (see the tool's docstring), not stated by the item.",
        "",
        f"{len(rows)} items: {counts['yes']} yes, {counts['no: option']} no: option, "
        f"{counts['no: fixed']} no: fixed. Headers that declare none: "
        + (", ".join(f"`{k}`" for k in says_none) if says_none else "—")
        + ". Headers with no item this tool recognises: "
        + (", ".join(f"`{k}`" for k in none_found) if none_found else "—") + ".",
        "",
        "| algorithm | item | what | default run | file |",
        "|---|---|---|---|---|",
    ]
    return "\n".join(head + rows) + "\n"


def main() -> int:
    text = table()
    if "--check" in sys.argv[1:]:
        if not OUT.is_file() or OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT.relative_to(ROOT)} is not what the headers give: run python tools/deviations.py",
                  file=sys.stderr)
            return 1
        print(f"{OUT.relative_to(ROOT)}: up to date")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {text.count(chr(10)) - 15} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
