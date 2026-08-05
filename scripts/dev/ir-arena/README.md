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

**C — the closer landed at the end of the FIRST line of the RHS** (34 sites, fixed)

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

There is a third variant, **C2a**, which is the most common and the least
visible — the closer sits after the *first argument*:

```sounio
ir_arena_store(slot, ir_binop(bf_instr.dst,)
    am_or_var_src[bf_s2 as usize], BinaryOp::OpBitAnd, bf_s1)
```

**Do not batch C2 blindly.** `classify_c.py` buckets by how the opener line ends;
`fix_shape_c.py` applies the unified repair — drop the stray closer, then append
one `)` at the first following line where cumulative depth reaches +1 — and
**refuses any reconstruction whose argument count does not match the callee's
declaration**. That arity check is the only defence available, because C2 is
syntactically valid and the compiler never points at the site. Verified arities:
`ir_binop` 4, `ir_load_imm` 2, `ir_call` 5, `ir_merge_adjust_epistemic_instr` 12,
`ir_arena_store` 2. Result: 34 verified, 0 refused, all 37 multi-line arena
constructs closing cleanly.

## Measured, one compile each, seed pre-derived

| tree | errors |
|---|---|
| `origin/main` and merge-base `40116b661d` (baseline, ~101.6 MB artifact) | **6** |
| committed SoA tree | **≥151,940** — first at `main.sio:2985` |
| + shape A | 23,035 — first at `ir.sio:940` |
| + shape B and `(*node).head` | 5,051 |
| + shape C | **6 — the baseline set. exit 0, artifact produced.** |

## STOP: the branch compiler cannot compile a two-function program

Found 2026-08-05, and it is the top blocker.

```
fn s0(x: i64) -> i64 { x * 2 + 1 }
fn main() -> i64 { var t: i64 = 1  t = (t + s0(t)) % 1009  t % 251 }
```

| compiler | result |
|---|---|
| `origin/main` | rc=0, binary runs, exits 4 |
| this branch, before Sweep 1 | `Failed to write native binary ... rc=12` |
| this branch, after Sweep 1 | same |

Bisected to **one helper function**. Single-function programs compile and run
correctly, which is why it stayed hidden — and it explains why only 7 of 61
sampled `tests/run-pass` files compiled at all.

`rc=12` is `!compile_ok` (`native/codegen_x86_linux.sio:10839`), a generic
backend failure with no diagnostic. Lowering succeeds first (`final_fn_count 2`),
so the defect is in native codegen, downstream of the conversion. It predates
Sweep 1. The same `rc=12` blocks `ir_instr_arena_gate.sh`, so it is also what
stops the arena's runtime contract from being verified.

**The green self-compile is a FALSE GREEN.** The branch builds because the *seed*
(lean_single) compiles it, and lean_single does not carry this defect. `exit 0`
on the build says nothing about the compiler it produces. Repro kept at
`tests/known_failures/soa_arena_two_function_codegen.sio`.

Fix this before raising `IR_MAX_INSTRS` or trusting any parity measurement.

## What the green build does NOT yet buy

Measured, not assumed:

- **The memory win is real but it is NOT in bss.** bss went *up*:
  3,559,050,480 vs main's 3,418,213,544, exactly +140,836,936 = the arena arrays
  + pools + region tables (140,836,864 computed). Dropping `[IrInstr; 4096]`
  moved bss by **zero**, so #1649's "doubling `IR_MAX_INSTRS` roughly doubles
  bss" does not follow. The reduction is in **by-value / stack** storage: the
  compiler's own diagnostic during the gate run reports `ir_empty_module`
  returning **4,275,944 bytes**, against the ~2.08 GB that main's layout implies
  (4096 × 248 × 2048, matching the figure in #1649/#1650). Peak RSS on a *small*
  compile is 458 MB vs 466 MB — near parity, because the old inline array was
  only ever faulted in on the pages actually touched.
- **The 4096 cap still stands.** `IR_MAX_INSTRS: i64 = 4096` (`ir.sio:20`) is
  unchanged, so `knowledge_octonion_structure.sio` still fails with *needs 14389
  IR instructions* on **both** compilers. Storage is no longer the blocker — the
  arena holds 1,048,576 slots, 256× the cap — so raising it is a separate and now
  unblocked change.
- **The arena's runtime contract is still unverified.**
  `SOUNIO_IR_ARENA_SOUC=<compiler> bash scripts/ci/ir_instr_arena_gate.sh` FAILS
  at `build_ir_instr_arena_witness`: the composed 9,670-line translation unit
  does not compile, `rc=12` = `!compile_ok`
  (`native/codegen_x86_linux.sio:10839`), a generic codegen failure rather than
  an arena diagnostic. So a green self-compile is **not** evidence that the arena
  stores and loads correctly, and per `BOOTSTRAP.md:22` the exit code lies. Get
  the witnesses building before trusting any of this.

Refs #1649, #1655.
