#!/usr/bin/env python3
"""Find lines where the transform put a closing paren AFTER a trailing // comment.

The balance check cannot see these: both parens are present in the bytes, so the
raw count is unchanged. Only a comment-aware scan finds them.

Signature: strip string/char literals, split at the first `//`. If the CODE part
has unbalanced open parens and the COMMENT part contains a ')', the closer landed
inside the comment and the parser never sees it.
"""
import subprocess
import sys

REPO = "/workspace/wt-ir-soa-phase0"
COMMIT = "0ae1ebff20"


def git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True)


def strip_literals(line):
    """Blank out string and char literal contents so quotes inside don't confuse us."""
    out, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c in '"\'':
            quote = c
            out.append(" ")
            i += 1
            while i < n:
                if line[i] == "\\":
                    out.append(" ")
                    i += 2
                    continue
                if line[i] == quote:
                    break
                out.append(" ")
                i += 1
            out.append(" ")
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


files = [
    f
    for f in git("diff", "--name-only", f"{COMMIT}^", COMMIT).stdout.splitlines()
    if f.endswith(".sio")
]

total = 0
per_file = {}
samples = []

for path in files:
    text = git("show", f"{COMMIT}:{path}").stdout
    hits = 0
    for lineno, raw in enumerate(text.splitlines(), 1):
        masked = strip_literals(raw)
        idx = masked.find("//")
        if idx < 0:
            continue
        code, comment = masked[:idx], masked[idx:]
        depth = code.count("(") - code.count(")")
        if depth > 0 and ")" in comment:
            hits += 1
            total += 1
            if len(samples) < 8:
                samples.append(f"{path}:{lineno}: {raw.strip()[:150]}")
    if hits:
        per_file[path] = hits

print(f"Scanned {len(files)} .sio files at {COMMIT}\n")
print(f"TOTAL comment-eaten closing parens: {total}\n")
if per_file:
    print("Per file:")
    for p, c in sorted(per_file.items(), key=lambda kv: -kv[1]):
        print(f"  {c:6d}  {p}")
    print("\nSamples:")
    for s in samples:
        print("  " + s)

sys.exit(1 if total else 0)
