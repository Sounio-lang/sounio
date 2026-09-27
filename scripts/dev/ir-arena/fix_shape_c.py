#!/usr/bin/env python3
"""Shape C: the transform put the RHS's closing paren at the end of the FIRST
physical line of the right-hand side, wherever the expression actually ended.

    C1   ir_arena_store(slot, ir_load_imm(          -> closer missing entirely
    C2a  ir_arena_store(slot, ir_binop(x.dst,)      -> closer after the 1st arg
    C2b  ir_arena_store(slot, ir_call()             -> inner call closed empty

C2 is syntactically VALID -- it silently re-homes the remaining arguments into
the OUTER call -- so the compiler reports arity/type noise, never the real site.
That is why every reconstruction here is checked against the callee's declared
arity and the script refuses rather than guesses.

Unified repair: drop the stray closer from the opener (C2 only), then append one
')' to the first following line at which cumulative paren depth reaches +1, i.e.
exactly one short of closing the ir_arena_store call.
"""
import re
import subprocess
import sys

DRY = "--apply" not in sys.argv

# declared arities, read from the definitions
ARITY = {
    "ir_binop": 4,            # dst, lhs, op, rhs
    "ir_load_imm": 2,         # dst, value
    "ir_call": 5,             # dst, fn_id, fn_name, args, arg_count
    "ir_arena_store": 2,      # at, instr
    "ir_merge_adjust_epistemic_instr": 12,  # instr + 11 offsets (both twins agree)
}


def strip(line):
    out, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c in "\"'":
            q = c
            out.append(" ")
            i += 1
            while i < n:
                if line[i] == "\\":
                    out.append(" ")
                    i += 2
                    continue
                if line[i] == q:
                    break
                out.append(" ")
                i += 1
            out.append(" ")
            i += 1
        else:
            out.append(c)
            i += 1
    s = "".join(out)
    j = s.find("//")
    return s if j < 0 else s[:j]


def depth(s):
    return s.count("(") - s.count(")")


def top_level_args(call_text):
    """Count top-level comma-separated arguments of name(...) -> int."""
    i = call_text.index("(")
    d, args, cur = 0, [], ""
    for ch in call_text[i:]:
        if ch == "(":
            d += 1
            if d == 1:
                continue
        elif ch == ")":
            d -= 1
            if d == 0:
                break
        if d == 1 and ch == ",":
            args.append(cur)
            cur = ""
            continue
        cur += ch
    if cur.strip():
        args.append(cur)
    return len([a for a in args if a.strip()])


files = [
    f
    for f in subprocess.run(
        ["git", "diff", "--name-only", "0ae1ebff20^", "0ae1ebff20"],
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if f.endswith(".sio")
]

report, refused, per_file = [], [], {}

for path in files:
    lines = open(path).read().split("\n")
    for n, raw in enumerate(lines):
        if "ir_arena_store" not in raw:
            continue
        code = strip(raw)
        if depth(code) == 0:
            continue
        t = code.rstrip()
        if t.endswith("{"):
            continue  # struct literal, already repaired

        if t.endswith(",)") or t.endswith("()"):
            kind = "C2"
            cut = raw.rstrip()
            assert cut.endswith(")")
            new_opener = cut[:-1]
        elif t.endswith("(") or t.endswith(","):
            kind = "C1"
            new_opener = raw.rstrip()
        else:
            refused.append(f"{path}:{n + 1} unrecognised opener {t[-4:]!r}")
            continue

        # walk forward to the line where depth first reaches +1
        d = depth(strip(new_opener))
        end = None
        for j in range(n + 1, min(n + 40, len(lines))):
            d += depth(strip(lines[j]))
            if d == 1:
                end = j
                break
            if d <= 0:
                break
        if end is None:
            refused.append(f"{path}:{n + 1} could not locate construct end")
            continue

        new_end = lines[end].rstrip() + ")"

        # ---- verification: reconstruct and check arity of BOTH calls ----
        body = [new_opener] + lines[n + 1 : end] + [new_end]
        joined = " ".join(strip(x).strip() for x in body)
        outer = joined[joined.index("ir_arena_store") :]
        m = re.search(r"\)\s*,\s*([a-z_][a-z0-9_]*)\s*\(", outer)
        inner_name = m.group(1) if m else None
        try:
            outer_n = top_level_args(outer)
            inner_n = (
                top_level_args(outer[outer.index(inner_name + "(") :])
                if inner_name
                else None
            )
        except Exception as e:
            refused.append(f"{path}:{n + 1} could not parse reconstruction ({e})")
            continue

        if outer_n != ARITY["ir_arena_store"]:
            refused.append(
                f"{path}:{n + 1} ir_arena_store would get {outer_n} args, want 2"
            )
            continue
        want = ARITY.get(inner_name)
        if want is None:
            refused.append(
                f"{path}:{n + 1} inner call `{inner_name}` arity unknown — inspect by hand"
            )
            continue
        if inner_n != want:
            refused.append(
                f"{path}:{n + 1} {inner_name} would get {inner_n} args, want {want}"
            )
            continue

        per_file.setdefault(path, []).append((n, end, new_opener, new_end))
        report.append(
            f"  {kind}  {path}:{n + 1}-{end + 1}  {inner_name}({inner_n} args) OK"
        )

for line in report:
    print(line)
print(f"\nverified: {len(report)}")
if refused:
    print(f"\nREFUSED {len(refused)} — handle individually:")
    for r in refused:
        print("  " + r)

if not DRY:
    for path, sites in per_file.items():
        lines = open(path).read().split("\n")
        for n, end, new_opener, new_end in sorted(sites, reverse=True):
            lines[n] = new_opener
            lines[end] = new_end
        open(path, "w").write("\n".join(lines))
    print(f"\nAPPLIED to {len(per_file)} files")
else:
    print("\n(dry run — pass --apply to write)")
