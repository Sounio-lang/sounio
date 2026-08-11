#!/usr/bin/env python3
"""Phase 0 step 3: per-file paren/brace balance delta of the SoA commit vs its parent.

Absolute counts are meaningless (parens live inside strings and comments; on this
tree lean_single.sio is +17 at rest). The only valid check is that the DELTA is
unchanged by the commit.
"""
import subprocess
import sys

REPO = "/workspace/wt-ir-soa-phase0"
COMMIT = "0ae1ebff20"
PARENT = "0ae1ebff20^"


def git(*args):
    return subprocess.run(
        ["git", "-C", REPO, *args], capture_output=True, text=True
    )


def blob(rev, path):
    r = git("show", f"{rev}:{path}")
    return r.stdout if r.returncode == 0 else None


def delta(text):
    return (
        text.count("(") - text.count(")"),
        text.count("{") - text.count("}"),
        text.count("[") - text.count("]"),
    )


files = [
    f
    for f in git("diff", "--name-only", PARENT, COMMIT).stdout.splitlines()
    if f.endswith(".sio")
]

print(f"{len(files)} .sio files touched by {COMMIT}\n")

bad, added = [], []
for path in files:
    new = blob(COMMIT, path)
    old = blob(PARENT, path)
    if new is None:
        continue
    if old is None:
        added.append((path, delta(new)))
        continue
    dn, do = delta(new), delta(old)
    if dn != do:
        bad.append((path, do, dn))

if added:
    print("NEW files (no baseline — absolute delta shown, eyeball it):")
    for p, d in added:
        flag = "  <-- nonzero" if d != (0, 0, 0) else ""
        print(f"  {p}: paren/brace/bracket {d}{flag}")
    print()

if bad:
    print("*** BALANCE DELTA CHANGED — tree is structurally corrupt ***")
    for p, do, dn in bad:
        print(f"  {p}: parent {do} -> commit {dn}")
    sys.exit(1)

print(f"BALANCE OK: all {len(files) - len(added)} modified files unchanged in delta")
