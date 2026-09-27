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
- **The cap is raised and #1649's blocker is gone.** `IR_MAX_INSTRS` is 16384;
  `knowledge_octonion_structure.sio` compiles, runs and prints `PASS`. It lowers
  to 7057 instructions. `bss` is unchanged at 3,559,050,480 — a 4× raise costing
  nothing is exactly what the arena is for.

  Raising it needed guards, not just a literal: `dce_run_impl` and `cp_run_impl`
  hold 8192-wide contexts and used to **truncate**, and a truncated liveness
  analysis is wrong rather than weak — a use past the cap is unseen, so its
  definition looks dead. Both now refuse. Pinned by the repurposed
  `irfunction_instr_capacity_coherence_gate.sh`, which is non-vacuous.

- **Open, pre-existing, now reachable: `-O` deletes `print()` above 256
  registers.** `opt_cleanup` carries ~90 register-indexed `[_; 256]` arrays; the
  octonion test uses **7088** registers, so the analysis runs on a prefix of the
  register file. Without `-O` it prints `PASS`; with `-O` its own PASS/FAIL block
  disappears. **Do not fix this by refusing the function** — both refusals were
  measured worse than the bug: skipping all of `opt_cleanup_function_mfi`
  SIGSEGVs, and skipping only the register-indexed peels returns 95 where every
  other compiler returns 56, because the peels are a pipeline, not a menu.
  Recorded in `tests/known_failures/opt_cleanup_wide_register_file.sio`.
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

## The reference hazard: hoist before you take `&` into a boxed array

Taking a reference to an element of an array inside a `Box`'d struct reads the
**wrong address** under the bootstrap seed:

```sounio
ir_float_bits_get(&(*module).functions[i as usize], r)   // garbage
var fslot = (*module).functions[i as usize]
ir_float_bits_get(&fslot, r)                             // correct
```

Same binary, same probe, one line apart:

| | `live_at_codegen` |
|---|---|
| in place | 4003 / 4643 / 4963 — varies per run |
| hoisted | **8 / 8 / 8 — exactly equal to `writes`** |

This is why `ir_module_seal_functions` sealed **0 of 7** regions until the
element was hoisted into a local. That fix was landed with its mechanism
recorded as unproven; this is the proof, and it generalises — the hazard is the
reference, not sealing.

### Three false diagnoses this produced, and what killed each

Every alarming number in the `float_reg_bits` investigation came from the broken
instrument, not from broken storage.

| claim | killed by |
|---|---|
| the module is born dirty (`untouched8=7388`) | hoisting the counter → `untouched8=0` |
| `Box::new` delivers uninitialised memory | control run on `origin/main` with only a probe added: `instr_count=0, reg_count=0` |
| the function slot is recycled memory | `IR_FLOAT_BITS_INHERITED = 0` over runs lowering 5 functions — my own falsifier |
| a by-value struct copy drops array fields (#1655 shape) | known pattern: `pattern_direct=8`, `pattern_copied=8` |

The fourth hypothesis was only reachable because the third was tested with a
**known pattern** rather than by inference. Proving copies were fine left the
reference as the only remaining difference between the clean reading and the
dirty one.

`IR_FLOAT_BITS_INHERITED` stays in the tree although it now always reads 0. It
is the falsifier that killed a wrong theory, and it is what would catch that
theory becoming true.

### Where the bitset actually stands

`writes=8`, `live_at_codegen=8`, deterministic. That pair is the round-trip
evidence, and it is the whole of it.

**The `v=2.500000` run is NOT evidence for the bitset.** With
`IR_FLOAT_BITS_TRUSTED = 0` the consumers are gated off, so that `f64` comes out
right through the existing marker-nop path. It shows the branch has not
regressed, which is worth having and is not the same claim. Commit `d55ee1ed08`
runs the two together in one sentence; read the round-trip claim off
`writes`/`live_at_codegen` alone.

`IR_FLOAT_BITS_TRUSTED` is still `0`: flipping it is a behaviour change and
belongs in its own commit with the consumers switched and a gate. The evidence
for flipping it now exists, which it did not before.

### Reproduced standalone — `scripts/dev/ir-arena/repro_boxed_element_ref.sio`

    hoisted=5 inplace=22499901309144        # both should be 5

30 lines, no dependency on this branch, on **lean_single** — the seed that
builds Madaros and therefore the compiler that generated the miscompiled code.

The first attempt ran it through `madaros --native-v2-compile` and it segfaulted
before reaching the reference, which I recorded below as "blocked". That was
blocked on the choice of compiler, not on the compiler; the paragraph is kept
because the native-v2 observation is separately true.

Scalar field reads in place through the `Box` are **correct** (`tag_inplace=77`).
The hazard is a reference to an aggregate *element*, dereferenced in a callee.

**CORRECTION (see `REF_HAZARD_SITE_AUDIT.md`): the `Box` is not the ingredient.**
The line above saying "without the `Box` the same program prints 5 and 5" was
measured with `madaros --native-v2-compile`. Under `lean_single` the same
program gives `hoisted=5 inplace=288899952421056`. A local, a reference
parameter and a `Box` all fail alike. The file name
`repro_boxed_element_ref.sio` is therefore too narrow.

Branch sweep for the shape: 5 sites — 2 are comments describing this hazard, 3
are in `compiler/pkg/{lock,registry_client}.sio` and off the IR path. Every IR
site is hoisted.

### The earlier, wrong conclusion, kept

A minimal `.sio` reproducer of the reference hazard is **blocked**, not
negative. Boxing a struct in a *user program* under `--native-v2-compile`
segfaults before the reference is even reached — `Box::new(make())` followed by
`print_int((*a).n)`, where `Flat` is `{ n: i64, xs: [i64; 32] }`, gives rc=139
on a compiler built from `origin/main`. Without the `Box` the same program
prints the right answer (`hoisted=5 inplace=5`), so the shape cannot be
demonstrated without it. Whether `main`'s own compiler code hits the hazard is
therefore **open**; the two `&(*lock).entries[i as usize]` sites in
`compiler/pkg/lock.sio` are the candidates to check first.
