#!/usr/bin/env python3
"""Sweep 1: route every WRITTEN-THROUGH by-value IrFunction copy via ir_function_clone.

Selection is deliberately narrow: a site qualifies only when the local is copied
from a by-value `IrFunction` parameter AND the same local is written through
`ir_region_slot_w(<local>.region`. Read-only copies are left alone -- sharing a
region is harmless when nobody writes, and cloning them would burn arena for
nothing.

    var result = func     ->     var result = ir_function_clone(&func)
"""
import re
import subprocess
import sys

DRY = "--apply" not in sys.argv

FN_START = re.compile(r"^(pub )?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
SLOT_W = re.compile(r"ir_region_slot_w\(\s*\(?\*?([A-Za-z_][A-Za-z0-9_]*)\)?\.region")
P_VAL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*IrFunction\b")
P_REF = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*&\s*!?\s*IrFunction\b")
VAR_COPY = re.compile(r"^(\s*)var\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\*?)([A-Za-z_][A-Za-z0-9_]*)\s*$")

files = [
    f
    for f in subprocess.run(
        ["git", "ls-files", "self-hosted"], capture_output=True, text=True
    ).stdout.splitlines()
    if f.endswith(".sio")
]

total = 0
for path in files:
    lines = open(path, errors="replace").read().split("\n")
    edits = []
    i = 0
    while i < len(lines):
        m = FN_START.match(lines[i])
        if not m:
            i += 1
            continue
        fname = m.group(2)
        sig, j = "", i
        while j < len(lines):
            sig += lines[j] + " "
            if "{" in lines[j]:
                break
            j += 1
        bs = j
        d = lines[bs].count("{") - lines[bs].count("}")
        k = bs
        while k + 1 < len(lines) and d > 0:
            k += 1
            c = lines[k].split("//")[0]
            d += c.count("{") - c.count("}")
        be = k

        head = sig[: sig.index("{")] if "{" in sig else sig
        byval = set(P_VAL.findall(head)) - set(P_REF.findall(head))

        written = set()
        for n in range(bs, be + 1):
            written.update(SLOT_W.findall(lines[n].split("//")[0]))

        for n in range(bs, be + 1):
            vm = VAR_COPY.match(lines[n].split("//")[0])
            if not vm:
                continue
            indent, local, star, srcname = vm.groups()
            if srcname not in byval or local not in written:
                continue
            if star:  # `var x = *p` is a deref, not the copy shape we mean
                continue
            edits.append(
                (n, f"{indent}var {local} = ir_function_clone(&{srcname})", fname)
            )
        i = be + 1

    if edits:
        for n, new, fname in edits:
            print(f"  {path}:{n + 1}  ({fname})  {lines[n].strip()}  ->  {new.strip()}")
        total += len(edits)
        if not DRY:
            for n, new, _ in edits:
                lines[n] = new
            open(path, "w").write("\n".join(lines))

print(f"\n{'would convert' if DRY else 'CONVERTED'}: {total} sites")
if DRY:
    print("(dry run — pass --apply to write)")
