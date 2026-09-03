#!/usr/bin/env python3
"""Move closing parens that the transform appended AFTER a trailing // comment
back to the end of the code, preserving comment-column alignment.

    ir_arena_store(a, b)   // note)      ->   ir_arena_store(a, b))  // note

Refuses to touch a line whose comment does not actually END with the required
number of ')' — that would mean the paren is a legitimate part of the prose and
the imbalance has some other cause.
"""
import sys

WORKTREE = "/workspace/wt-ir-soa-phase0"
TARGETS = [
    "self-hosted/compiler/main.sio",
    "self-hosted/ir/inline.sio",
    "self-hosted/linker/test_linker.sio",
]


def strip_literals(line):
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


fixed_total, refused = 0, []

for rel in TARGETS:
    path = f"{WORKTREE}/{rel}"
    with open(path) as fh:
        lines = fh.read().split("\n")

    changed = 0
    for i, raw in enumerate(lines):
        masked = strip_literals(raw)
        idx = masked.find("//")
        if idx < 0:
            continue
        depth = masked[:idx].count("(") - masked[:idx].count(")")
        if depth <= 0 or ")" not in masked[idx:]:
            continue

        code, comment = raw[:idx], raw[idx:]
        body = comment.rstrip()
        trail_ws = comment[len(body):]

        if not body.endswith(")" * depth):
            refused.append(f"{rel}:{i + 1}: {raw.strip()[:120]}")
            continue

        stripped = code.rstrip()
        pad = code[len(stripped):]
        # keep the comment in its original column by eating `depth` pad spaces
        new_pad = pad[depth:] if len(pad) > depth else " "
        lines[i] = stripped + ")" * depth + new_pad + body[:-depth] + trail_ws
        changed += 1

    if changed:
        with open(path, "w") as fh:
            fh.write("\n".join(lines))
    print(f"{changed:4d} fixed  {rel}")
    fixed_total += changed

print(f"\nTOTAL fixed: {fixed_total}")
if refused:
    print(f"\nREFUSED {len(refused)} (comment does not end with the needed parens):")
    for r in refused:
        print("  " + r)
    sys.exit(1)
