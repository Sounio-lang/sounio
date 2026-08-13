#!/usr/bin/env python3
"""Sweep 1 inventory: every write through an IrFunction bound BY VALUE.

Before the arena, `IrFunction` carried its instructions inline, so a by-value
binding was a deep copy and writing through it could not touch the caller's
function. With a region handle the copy is an ALIAS, so every such write now
mutates the caller's storage. The type checker cannot see this: the copy site is
unchanged in both worlds.

For each function we resolve how each IrFunction-typed name is bound:

    f: IrFunction        BY VALUE  -> writing through it is an alias bug
    f: &IrFunction       by ref, immutable
    f: &! IrFunction     by ref, mutable -> writing through it is INTENDED
    var f = <by-value>   BY VALUE  -> alias bug (the `var result = func` shape)

then flag `ir_region_slot_w(<name>.region` writes through the by-value ones.
"""
import re
import subprocess
from collections import defaultdict

FN_START = re.compile(r"^(pub )?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
SLOT_W = re.compile(r"ir_region_slot_w\(\s*\(?\*?([A-Za-z_][A-Za-z0-9_]*)\)?\.region")
# binding forms inside a signature
P_VAL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*IrFunction\b")
P_REF = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*&\s*!?\s*IrFunction\b")
P_MUT = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*&\s*!\s*IrFunction\b")
VAR_COPY = re.compile(r"^\s*var\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\*?([A-Za-z_][A-Za-z0-9_]*)\s*$")

files = [
    f
    for f in subprocess.run(
        ["git", "ls-files", "self-hosted"], capture_output=True, text=True
    ).stdout.splitlines()
    if f.endswith(".sio")
]

hits = defaultdict(list)
by_kind = defaultdict(int)

for path in files:
    lines = open(path, errors="replace").read().split("\n")
    # segment into function bodies by brace depth
    i = 0
    while i < len(lines):
        m = FN_START.match(lines[i])
        if not m:
            i += 1
            continue
        fname = m.group(2)
        # gather signature (may span lines) until the '{' that opens the body
        sig, j, d = "", i, 0
        while j < len(lines):
            sig += lines[j] + " "
            d += lines[j].count("{") - lines[j].count("}")
            if "{" in lines[j]:
                break
            j += 1
        body_start = j
        # find body end
        d = lines[body_start].count("{") - lines[body_start].count("}")
        k = body_start
        while k + 1 < len(lines) and d > 0:
            k += 1
            code = lines[k].split("//")[0]
            d += code.count("{") - code.count("}")
        body_end = k

        sig_head = sig[: sig.index("{")] if "{" in sig else sig
        refs = set(P_REF.findall(sig_head)) | set(P_MUT.findall(sig_head))
        byval = set(P_VAL.findall(sig_head)) - refs

        # locals copied from a by-value name are themselves by value
        local_copies = {}
        for n in range(body_start, body_end + 1):
            vm = VAR_COPY.match(lines[n].split("//")[0])
            if vm and vm.group(2) in byval:
                local_copies[vm.group(1)] = (n + 1, vm.group(2))
        aliased = byval | set(local_copies)

        for n in range(body_start, body_end + 1):
            for recv in SLOT_W.findall(lines[n].split("//")[0]):
                if recv in aliased:
                    kind = "local-copy" if recv in local_copies else "by-value-param"
                    by_kind[kind] += 1
                    hits[path].append((n + 1, fname, recv, kind))
        i = body_end + 1

total = sum(len(v) for v in hits.values())
print(f"writes through a BY-VALUE IrFunction: {total}\n")
for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1]):
    print(f"  {v:5d}  {k}")
print()
for path in sorted(hits, key=lambda p: -len(hits[p])):
    fns = sorted({h[1] for h in hits[path]})
    print(f"{len(hits[path]):5d}  {path}   ({len(fns)} functions)")
    for f in fns[:6]:
        print(f"           {f}")
    if len(fns) > 6:
        print(f"           ... +{len(fns) - 6} more")
