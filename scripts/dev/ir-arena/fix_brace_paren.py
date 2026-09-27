#!/usr/bin/env python3
"""Shape B: the transform closed the call BEFORE a multi-line struct-literal body.

    ir_arena_store(slot, IrInstr {)      ->   ir_arena_store(slot, IrInstr {
        op: ...,                                  op: ...,
    }                                         })

Brace and paren counts are both unchanged by the corruption, so no balance check
can see it. Fix = drop the ')' after '{', brace-match forward, append ')' there.
"""
SITES = [
    ("/workspace/wt-ir-soa-phase0/self-hosted/ir/normalize.sio", 114),
    ("/workspace/wt-ir-soa-phase0/self-hosted/ir/normalize.sio", 205),
    ("/workspace/wt-ir-soa-phase0/self-hosted/linker/mod.sio", 177),
]

by_file = {}
for path, lineno in SITES:
    by_file.setdefault(path, []).append(lineno)

for path, linenos in by_file.items():
    lines = open(path).read().split("\n")
    # descending so earlier edits never shift later indices
    for lineno in sorted(linenos, reverse=True):
        idx = lineno - 1
        raw = lines[idx]
        assert raw.rstrip().endswith("{)"), f"{path}:{lineno} unexpected: {raw!r}"

        # drop the stray ')' that sits right after the '{'
        stripped = raw.rstrip()
        lines[idx] = stripped[:-1] + stripped[-1:].replace(")", "")

        # brace-match forward from this line to find the body's closing '}'
        depth = 0
        end = None
        for j in range(idx, len(lines)):
            code = lines[j].split("//")[0]
            depth += code.count("{") - code.count("}")
            if j > idx and depth == 0:
                end = j
                break
        assert end is not None, f"{path}:{lineno} no matching close brace"
        assert lines[end].rstrip().endswith("}"), f"{path}:{lineno} -> {lines[end]!r}"
        lines[end] = lines[end].rstrip() + ")"
        print(f"  {path.split('/')[-1]}:{lineno} -> closer moved to line {end + 1}")

    open(path, "w").write("\n".join(lines))

print(f"\nfixed {len(SITES)} sites")
