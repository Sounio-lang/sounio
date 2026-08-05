# Transform-damage detectors for the #1649 IR arena conversion

A paren/brace **balance check cannot find any of the damage below**, because in
every shape both parens are present in the file's bytes. Balance was clean on all
46 touched files while the build produced 151,940+ errors. These scripts are the
checks that actually discriminate.

| script | finds |
|---|---|
| `balance_check.py` | per-file paren/brace/bracket delta vs the commit's parent. Necessary, **not sufficient** — it passed on a fully corrupt tree |
| `find_comment_eaten_parens.py` | **shape A** — closer emitted after a trailing `//` comment |
| `fix_comment_eaten_parens.py` | repairs shape A, preserving the comment column |
| `fix_brace_paren.py` | repairs **shape B** — closer emitted before a multi-line struct-literal body |

## The three shapes

**A — closer after a trailing comment** (51 sites, fixed)

```sounio
ir_arena_store(slot, ir_load_imm(0, 4294967296)   // a.lo = 2^32)
```

The parser never sees that `)`. `ir_arena_store(` runs forward past the closing
brace and swallows the next `fn` signature as an expression, which surfaces as
`undefined identifier <TypeName>` plus `Mut` / `Panic` / `Div` all reported on the
signature's line. **That triple is the tell for structural corruption, not a type
problem.**

**B — closer before a struct-literal body** (3 sites, fixed)

```sounio
ir_arena_store(slot, IrInstr {)
    op: base.op,
}
```

**C — multi-line right-hand side truncated** (~34 candidates, OPEN)

Two sub-shapes, and they fail differently:

```sounio
// C1: outer closer missing -> depth stays +1 -> desync, like shape A
ir_arena_store(slot, ir_load_imm(
    expected.outer_const_reg,
    combined_value,
)

// C2: inner call closed EMPTY, real arguments orphaned into the outer call.
// Syntactically valid, semantically wrong -- shows up as arity/type errors,
// never as a desync. This is the dangerous one.
ir_arena_store(slot, ir_call()
    1, -1,
    ir_name_from_bytes(100, 111, 117, 98, 6),
    Some(Box::new(args1)),
    1
)
```

C1 wants one more `)` at the construct's end. C2 wants the `)` after `ir_call(`
deleted and the closer moved to the end of the argument list. **Do not batch C2
blindly** — it is the one shape where the compiler will not tell you that you got
it wrong.

## Detecting shape C

Flag lines mentioning `ir_arena_store` / `ir_region_slot` whose comment-stripped
paren depth is non-zero with brace depth zero, then walk forward to see where
depth returns to 0. Legitimate multi-line calls close cleanly; C1 never does, and
C2 closes at the wrong arity.

## Measured, one compile each, seed pre-derived

| tree | errors |
|---|---|
| `origin/main` (baseline, still produces the 101,627,712 B artifact) | **6** |
| committed SoA tree | **≥151,940** — first at `main.sio:2985` |
| + shape A fixed | 23,035 — first at `ir.sio:940` |
| + shape B and `(*node).head` fixed | 5,051 — of which 6 are baseline |

Refs #1649, #1655.
