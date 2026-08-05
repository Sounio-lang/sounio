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

Parity caveat: all 7 programs that compiled are **single-function**, forced by
the regression above, and their receipts show `transactions=0 applications=0` —
the cloning passes had almost nothing to do. It is a no-regression signal, not
evidence the clone is meaningfully exercised. The first parity run was worse than
that: it ran **without `-O`**, so `opt_cleanup` was skipped entirely and the
comparison was vacuous. Always confirm `optimize=1` via
`SOUNIO_REBRACKET_TRACE=1` before reading a parity number.

The arena capacity risk — functions × passes × length against 1,048,576 slots
with no reclamation — is **retired at realistic scale**. What looked like a
separate "codegen wall" was rc=12 itself; with that fixed, a generated
60-function × 100-statement program (6,304 lines) compiles clean, runs, and
returns 56 — the same as `origin/main` — with no violation, capacity error or
quarantine line in the log. `ir_arena_mark` / `ir_arena_release` remain unused
and stay the intended fix if a much larger input ever does exhaust it.

**Coverage gap worth naming.** `ir_arena_swap_slots` is the one primitive here
written from scratch, and nothing in the evidence above demonstrably executes it:
its two call sites are `ocp_licm`'s hoist and the sink-loads swap, which need
specific IR shapes that a 24-file sample and a generated arithmetic stress are
unlikely to produce. Gate green plus 18 parity matches does **not** cover it.
A witness that forces both shapes is the honest next test.

| tree | errors |
|---|---|
| `origin/main` and merge-base `40116b661d` (baseline, ~101.6 MB artifact) | **6** |
| committed SoA tree | **≥151,940** — first at `main.sio:2985` |
| + shape A | 23,035 — first at `ir.sio:940` |
| + shape B and `(*node).head` | 5,051 |
| + shape C | **6 — the baseline set. exit 0, artifact produced.** |

## rc=12 — FIXED. It was never about function count.

The branch could not compile any program containing a call **with arguments**.
Calls with none were fine, which is why single-function programs worked and
anything calling a helper did not, and why only 7 of 61 sampled `tests/run-pass`
files compiled. "Cannot compile two functions" was the symptom; the trigger was
the argument list.

**Diagnosis cost one command, not one build** — the diagnostic was already in the
binary:

```
SOUNIO_NV2_IR_TRACE=1 ./mad --native-v2-compile prog.sio -O -o /tmp/x.elf
    NV2_IR missing_call_arg fn=main i=1 arg_index=0 arg_count=1
    NV2_IR unsupported fn=1 name=main
```

`ir_arena_load` deliberately does not rebuild the argument list (that needs
`Alloc`, and forcing `Alloc` onto ~431 call sites is what desynchronised the
checker in the first place). So a loaded instruction has `arg_count > 0` with
`call_args = None`, and `ir_arena_store` walked that empty list and wrote
`IR_INVALID_REG` once per argument. **Every round trip through the arena
destroyed the call's arguments.**

Repaired by how the instruction actually travels:

| shape | count | repair |
|---|---|---|
| pure slot→slot copy | 11 | `ir_arena_copy_slot` (already carries `ARG_BASE`/`ARG_COUNT`) |
| same-slot read-modify-write | 11 | `ir_arena_store_args` recognises it, preserves the binding |
| genuine cross-slot move | 1 | `ir_arena_store_from`, naming the source slot |
| **swap** | 2 | `ir_arena_swap_slots` |

A swap cannot be repaired by naming the source: the first store overwrites the
very slot the second must read its arguments from. It has to be one primitive.

Fail-closed rather than preserve-quietly: a store with no list and a positive
count is accepted **only** when the destination already holds exactly those
arguments; anything else latches a violation. Preserving silently would hand a
call another instruction's registers — the failure this arena exists to prevent.

Two sites the scanner proposed were **rejected after reading them**: it had
crossed a function boundary and matched a load from a different function, where
the value stored is a parameter. One would not have compiled; the other resolved
to outer names and would have silently taken arguments from an unrelated slot.

Verified against `origin/main`'s compiler — identical run results:

```
noargs 7 | 1 helper 4 | 2 helpers 8 | 3 helpers 13 | 4 helpers 19
6 helpers 34 | 19-function 904-line stress 225
```

and on a 24-file `tests/run-pass` sample: **18 match, 0 mismatch**, 6 not
compiled by either.

**Lesson worth keeping.** The self-compile was green through all of this. It is
built by the *seed* (lean_single), which does not carry the defect, so `exit 0`
on the build says nothing about the compiler it produces. Repro kept at
`tests/known_failures/soa_arena_two_function_codegen.sio`.

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
- **The arena's runtime contract is now VERIFIED.** With rc=12 fixed,
  `SOUNIO_IR_ARENA_SOUC=<compiler> bash scripts/ci/ir_instr_arena_gate.sh`
  passes: all three witnesses green, generation / sealing / fail-closed capacity
  proved. Vacuity re-checked with `SOUNIO_IR_ARENA_VACUITY=1` — deleting the
  generation guard fails the stale witness (rc=30), deleting the sealing guard
  fails the seal witness (rc=20), so the gate can still fail.

  Its static contract had gone stale and was updated: it demanded the AoS
  `pub var IR_INSTR_ARENA: [IrInstr; 1048576]`, which under #1655 is precisely
  the shape that must never return. It now asserts that array's **absence** and
  holds all 16 scalar lanes to the accessor's capacity. The generation-guard
  pattern also moved from `(*r).generation` to `r.generation` — `ir_region_status_v`
  takes the handle by value now. Same guard, same strength.

Refs #1649, #1655.
