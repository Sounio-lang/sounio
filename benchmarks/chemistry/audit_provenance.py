#!/usr/bin/env python3
"""Provenance audit of RESULTS.md: does every reported number have a command,
and does that command name a file that actually ships?

Written after RESULTS.md section 6.2b was found to have been measured with a
working copy that was never committed.  A document whose contract is "every
number carries the command that produced it" needs that contract checked
mechanically, or it degrades exactly where it is hardest to notice: in the
sections added last, under time pressure, while fixing the same defect.

Criterion, per section:
  * does it report high-precision numbers?
  * if so, does a fenced command block in that section name a file that exists
    in the tree being audited?

A section inheriting its parent's command is reported as INHERIT, not as a
pass: that is a judgement for a reader, not for a script.

Usage:
    python3 benchmarks/chemistry/audit_provenance.py [--tree DIR] [--doc PATH]

--tree defaults to the repository root, so the audit runs against the working
tree.  Point it at an unpacked release to audit what a reader actually gets.
"""
import argparse
import os
import re
import sys

NUM = re.compile(r"\d\.\d{3,}e[+-]\d{2}|\*\*\d+\.\d{3,}\*\*|\d+\.\d{4,}")
FILE = re.compile(r"([A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|sio|cpp|lean))")
PLACEHOLDER = re.compile(r"python3\s*-\s*<\s*the\s+harness|<\s*the\s+harness\s+in")


def sections(text):
    lines = text.splitlines()
    out, cur, start = [], None, 0
    for i, line in enumerate(lines):
        m = re.match(r"^(#{2,3})\s+(.*)", line)
        if m:
            if cur is not None:
                out.append((cur, "\n".join(lines[start:i])))
            cur, start = m.group(2).strip(), i
    if cur is not None:
        out.append((cur, "\n".join(lines[start:])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir, os.pardir))
    ap.add_argument("--doc", default=None)
    args = ap.parse_args()

    tree = os.path.abspath(args.tree)
    doc = args.doc or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "RESULTS.md")
    if not os.path.exists(doc):
        doc = os.path.join(tree, "RESULTS.md")

    shipped = set()
    for root, dirs, files in os.walk(tree):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "target")]
        for f in files:
            shipped.add(f)

    text = open(doc, encoding="utf-8").read()
    ok = inherit = fail = 0
    findings = []
    for name, body in sections(text):
        n = len(NUM.findall(body))
        if not n:
            continue
        blocks = re.findall(r"```(?:sh|bash|console)?\n(.*?)```", body, re.S)
        cmd = "\n".join(blocks)
        named = FILE.findall(cmd)
        present = [f for f in named if f.rsplit("/", 1)[-1] in shipped]
        if PLACEHOLDER.search(cmd):
            fail += 1
            findings.append((name, n, "PLACEHOLDER: the command block is prose, "
                                      "not a runnable command"))
        elif present:
            ok += 1
        elif named:
            fail += 1
            findings.append((name, n, "names %s, which is not in the tree"
                             % ", ".join(sorted(set(named))[:3])))
        elif blocks:
            fail += 1
            findings.append((name, n, "has a block but it names no file"))
        else:
            inherit += 1

    print(f"document : {doc}")
    print(f"tree     : {tree}")
    print(f"sections reporting numbers : {ok + inherit + fail}")
    print(f"  producer named and present: {ok}")
    print(f"  INHERIT (parent command)  : {inherit}")
    print(f"  FAIL                      : {fail}")
    for name, n, why in findings:
        print(f"    - {name}  [{n} numbers]  {why}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
