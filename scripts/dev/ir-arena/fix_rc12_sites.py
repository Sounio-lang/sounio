#!/usr/bin/env python3
"""rc=12 call-site repair.

A) ir_arena_store(A, ir_arena_load(B))  ->  ir_arena_copy_slot(A, B)
   A pure slot-to-slot copy. copy_slot already carries ARG_BASE/ARG_COUNT and
   NAME_OFF/NAME_LEN, which the store(load(..)) round trip drops.

B) var x = ir_arena_load(SRC) ... ir_arena_store(DST, x)   with DST != SRC
   ->  ir_arena_store_from(DST, x, SRC)
   The instruction moves between slots and may be modified on the way, so the
   arguments have to be named explicitly.

Same-slot read-modify-write is deliberately left alone: ir_arena_store_args now
recognises it and preserves the binding.
"""
import re
import subprocess
import sys

DRY = "--apply" not in sys.argv


def split_two_args(text, start):
    """Given text[start] == '(' of a call, return (arg1, arg2, end_index)."""
    d, args, cur = 0, [], ""
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "(":
            d += 1
            if d == 1:
                continue
        elif ch == ")":
            d -= 1
            if d == 0:
                args.append(cur)
                return args, i
        if d == 1 and ch == ",":
            args.append(cur)
            cur = ""
            continue
        cur += ch
    return None, -1


files = [
    f
    for f in subprocess.run(
        ["git", "ls-files", "self-hosted"], capture_output=True, text=True
    ).stdout.splitlines()
    if f.endswith(".sio")
]

DECL = re.compile(r"^\s*(?:let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*ir_arena_load\((.*)\)\s*$")
n_a = n_b = 0

for path in files:
    lines = open(path, errors="replace").read().split("\n")
    changed = False

    # ---- A: direct round trips ----
    for i, raw in enumerate(lines):
        if "ir_arena_store(" not in raw or "ir_arena_load(" not in raw:
            continue
        p = raw.index("ir_arena_store(")
        args, end = split_two_args(raw, p + len("ir_arena_store"))
        if not args or len(args) != 2:
            continue
        dst, src_expr = args[0], args[1].strip()
        if not src_expr.startswith("ir_arena_load("):
            continue
        inner, iend = split_two_args(src_expr, len("ir_arena_load"))
        if not inner or len(inner) != 1:
            continue
        lines[i] = (
            raw[:p]
            + f"ir_arena_copy_slot({dst},{inner[0]})"
            + raw[p + (end + 1 - p) :]
        )
        print(f"  A {path}:{i + 1}  {lines[i].strip()[:120]}")
        n_a += 1
        changed = True

    # ---- B: cross-slot via-local ----
    for i, raw in enumerate(lines):
        m = DECL.match(raw)
        if not m:
            continue
        name, src = m.group(1), m.group(2).strip()
        for j in range(i + 1, min(i + 60, len(lines))):
            sm = re.search(
                r"ir_arena_store\((.*),\s*" + re.escape(name) + r"\s*\)\s*$", lines[j]
            )
            if not sm:
                continue
            dst = sm.group(1)
            norm = lambda s: re.sub(r"\s+", "", s)
            if norm(src).replace("ir_region_slot_r", "S") == norm(dst).replace(
                "ir_region_slot_w", "S"
            ):
                break  # same slot: store_args handles it
            lines[j] = lines[j].replace(
                f"ir_arena_store({dst}, {name})",
                f"ir_arena_store_from({dst}, {name}, {src})",
            )
            print(f"  B {path}:{j + 1}  {lines[j].strip()[:130]}")
            n_b += 1
            changed = True
            break

    if changed and not DRY:
        open(path, "w").write("\n".join(lines))

print(f"\nA (copy_slot): {n_a}    B (store_from): {n_b}")
if DRY:
    print("(dry run — pass --apply to write)")
