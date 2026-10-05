<!-- docs:meta
topic_id: repo.docs.audit.madaros-nested-aggregate-element-store-dispatch-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-nested-aggregate-element-store-dispatch-2026-08-16
-->

# Forensic dispatch — Madaros native-v2 nested aggregate element stores: SIGSEGV and silent misaddress

**Status:** open defect, evidence collected, fix unassigned.
**Reporter:** cursor-2, lane `mli-s1` / `mli-s1-dispatch`, 2026-08-16.
**Found while:** implementing MLI S1 (`self-hosted/mli/`, WS-D) — hit in the
builder on the first instruction append, not sought out.

---

## 1. Summary

Under **Madaros native-v2 codegen** (default `bin/souc` engine), programs that
store aggregates into array elements fail in two distinct ways once the
element shape crosses a measured boundary:

- **Shape 1 — SIGSEGV:** storing a nominal-392 B struct with nesting depth 3
  (`MliInst`, which contains 4 × `MliOperand`, each containing `MliKind`) into
  an array element field — `blk.instrs[i] = inst` — crashes at runtime. The
  program type-checks and compiles clean. Crash reproduces with `blk` a
  **plain local variable** (no references involved) and via `&!` parameters.
- **Shape 2 — silent misaddress (the dangerous one):** scalar field writes
  through `&!Struct` into an array-of-struct element —
  `f.blocks[b].instr_count = v` — land at **wrong offsets**, clobbering
  sibling fields and sibling elements with no crash and no diagnostic.
  Observed effect during S1: 12 of 15 unit fixtures failing with phantom
  verifier codes because block metadata read back corrupted.

Both shapes were confirmed against a **Madaros built from source this
session** (§4.2), not only the checked-in ELF, so this is a current-source
defect, not artifact staleness (the #1689 lesson). Both were re-confirmed by
the executable witness (§5), which currently prints `OBSERVED` with 3
misaddressed cells out of 8 checked.

## 2. Measured boundary (bisection receipts)

All shapes bisected on 2026-08-16 with single-file `./bin/souc run` programs
under `/workspace/.wt/cursor-2` at work-in-progress states of what is now
commit `ab572a62d9`. Nominal sizes assume 8-byte fields, no padding (actual
layout is the compiler's; the shape distinction, not the byte count, is the
load-bearing fact). An earlier bus message quoted 80 B / 360 B from memory;
the field-count arithmetic below is the corrected record.

| Element stored into array | Nominal size | Nesting depth | Local var | Through `&!` param | Verdict |
|---|---:|---:|---|---|---|
| `Pair { a, b: i64 }` | 16 B | 1 | OK | (not tested) | clean |
| `MliKind` (6 × i64) | 48 B | 1 | OK | OK | clean |
| `MliOperand` (5 scalars + MliKind) | 88 B | 2 | OK | OK | clean |
| bare local array `[MliOperand; 4]`, element store | 88 B | 2 | OK | n/a | clean |
| `MliInst` (5 scalars + 4 × MliOperand) | 392 B | 3 | **SIGSEGV** | **SIGSEGV** | shape 1 |
| scalar field of array-of-struct element via `&!` (`f.blocks[b].field = v`, element = 3-field struct) | 8 B write | n/a | (see note) | **silent misaddress** | shape 2 |

Note on shape 2: the misaddress was proven by behaviour (writes to
`blk_id`-equivalent fields flipping sibling `has_term`/`id` cells, 12 fixtures
turning to phantom `MLI_VERR_BLOCK_ID`) and is pinned live by the witness in
§5, which performs 4 writes through `&!` helpers and checks 8 cells: 3 read
back wrong under both instruments.

Bisection narrative (receipts were live terminal output on this pod,
2026-08-16 13:30–14:04 UTC):

1. Full S1 fixture suite: compiles clean (`run_check_mode: verdict=0`,
   5 modules), SIGSEGVs at runtime on the first `mli_emit` call.
2. Probe A (constructors only): kind/operand/inst/block/function
   construction all clean, including a ~200 KB function struct returned by
   value (`magic` canary intact) — construction and large by-value returns
   are NOT the defect.
3. Probe B (emit path inline): crash isolated to `blk.instrs[0] = inst` with
   `blk` a **local** `var` — refs eliminated as a necessary condition.
4. Probe C (shape matrix): the table above. 48 B and 88 B element stores
   clean in every context; 392 B depth-3 element store crashes.
5. After moving instruction storage to an SoA pool (§6): 12/15 fixtures fail
   with phantom code 22 (`MLI_VERR_BLOCK_ID`) and spurious
   `double_terminator` builder refusals — reads/writes of scalar fields of
   `f.blocks[b]` through `&!` corrupted → shape 2 identified.
6. After moving block metadata to scalar SoA columns (`blk_id` /
   `blk_instr_count` / `blk_has_term` as `[i64; 16]`): **15/15 fixtures
   pass** under both instruments.

## 3. What this is NOT (boundary against adjacent findings — do not merge)

| Adjacent finding | Surface | Why it is distinct |
|---|---|---|
| `GLOBAL_VAR_ARRAY_INDEX_READS_ELEMENT0` (glm-cli1's claimed dispatch, 2026-08-16) | **Globals** under the **seed (lean_single)** path | This dispatch involves **no globals at all** — `self-hosted/mli/` has only scalar `pub let` module constants (the aggregate-global discipline was followed). The failing objects here are **locals and `&!` parameters** under **Madaros native-v2** codegen. Same broad "aggregate addressing" theme, different engine, different storage class, different repro. |
| #1678 / #1749 aggregate-array-element mutable-borrow miscompile | Madaros native-v2 | Same family — but the #1749 fix (`03416657fa`, **present in the binary tested here**) covers shallower shapes. This dispatch documents the residual: depth-3/392 B element stores (shape 1) and scalar-field-of-element writes through `&!` (shape 2) still miscompile after #1749. |
| fable-1's stale-base phantom failures (2026-08-16) | Instrument validity | Ruled out here: both shapes reproduce under a Madaros built from source at the exact tested commit (§4.2). |

## 4. Instrument provenance (both receipts)

### 4.1 Pod (iteration instrument)

- Engine: checked-in wrapper `./bin/souc` → Madaros v0.80.0 default engine,
  repo `/workspace/.wt/cursor-2`, branch `lane/cursor-2/mli-s1-20260816`
  (based on `origin/main` @ `6f2c4e2461`, which includes the #1749 fix).
- Result: shape 1 SIGSEGV, shape 2 misaddress, witness `OBSERVED (3 cells)`.

### 4.2 Slurm (source-built instrument — the authoritative receipt)

- Job: `srun -p cpu-ops` on `cpuops-t560-proxmox`, workdir
  `/tmp/mli-s1-W5nJTt`, 2026-08-16 13:59–14:03 UTC.
- Clone SHA **asserted** equal to the pod commit before any measurement:
  `ab572a62d9c60913f836d02ebd71df1842f655ea` (job fails closed on mismatch).
- Build: `bash scripts/ci/build_modular_madaros.sh` from that source,
  233 s, output ELF 99,937,624 B, reports `Madaros v0.80.0`.
- Gate: `MADAROS_RAW_BIN=<fresh ELF> ./bin/souc run
  self-hosted/mli/self_test_runner.sio` → 15/15 PASS (SoA layout).
- Witness: `MADAROS_RAW_BIN=<fresh ELF> ./bin/souc run
  self-hosted/mli/aggregate_store_diag.sio` →
  `OBSERVED (3 misaddressed cells)` — **the defect is in current source**.

## 5. Executable witness (mechanical fix tracking)

`self-hosted/mli/aggregate_store_diag.sio` — committed in `ab572a62d9`:

- exit-0 always (deliberately NOT a gate; it never turns CI red);
- performs shape-2 writes through `&!` helpers into a 4-element
  array-of-struct, then checks 8 cells (written values + neighbour
  integrity + a tail canary);
- prints `NOT_OBSERVED` when nested scalar field writes are clean, or
  `OBSERVED (<n> misaddressed cells)` while the defect lives;
- header comment carries the shape-1 repro recipe (revert
  `self-hosted/mli/ir.sio` to an AoS `MliBlock { instrs: [MliInst; 32] }`
  and run any builder fixture — crashes today).

Current output under both instruments: `OBSERVED (3 misaddressed cells)`.
The witness-matrix lane can track the fix by re-running this file instead of
re-deriving the repro.

## 6. Design consequence already taken (not a workaround request)

MLI S1 stores instructions as a **struct-of-arrays pool** on `MliFunction`
(scalar columns plus `[MliOperand; 528]` columns — the verified-safe shapes)
and block metadata as scalar SoA columns. This is the same remedy the IR
arena took in #1717 for the same class of pressure. It is documented as a
deliberate defect response in `self-hosted/mli/ir.sio` (storage layout note)
and in `docs/architecture/MLI_DESIGN.md` (S1 amendment), so S2's `ir_to_mli`
does not "simplify" back to AoS and re-import the crash.

## 7. Blocker contract

| Field | Value |
|---|---|
| Blocker-ID | BLK-20260816-nested-aggregate-element-store |
| Severity | HIGH (shape 2 is silent wrong-code; shape 1 is a hard crash in type-checked programs) |
| Class | compiler-defect / Madaros native-v2 lowering-or-codegen (not yet localised to `ir/lower.sio` vs `native/codegen_x86_linux.sio` — that localisation is the fix lane's first step) |
| Evidence level | executable witness + dual-instrument receipts (checked-in AND source-built at `ab572a62d9`) |
| Owner | unassigned — natural fit: witness-matrix lane (claude-3, #1678 residuals) |
| Worktree / branch | `/workspace/.wt/cursor-2` / `lane/cursor-2/mli-s1-20260816` @ `ab572a62d9` |
| Acceptance gate | `aggregate_store_diag.sio` prints `NOT_OBSERVED` under a session-built Madaros, AND the shape-1 AoS recipe (witness header) compiles and runs without SIGSEGV, AND proper run-pass/compile-fail witnesses land in the matrix |
| Next-Command | `MADAROS_RAW_BIN=<session-built ELF> ./bin/souc run self-hosted/mli/aggregate_store_diag.sio` |

---

## 8. Addendum (S2a, 2026-08-16 evening) — two further shapes, same session

Found while implementing MLI S2a (`ir_to_mli` + interpreter) in the 8-module
closure (`ir::ir` + `mli::*`). Executable witness for both:
`self-hosted/mli/inst_landing_diag.sio` (exit-0, `OBSERVED`/`NOT_OBSERVED`
per shape; currently `OBSERVED` on both).

**SHAPE 3 — struct-return landing corrupts nested operand fields,
closure-dependently.** Binding the 392 B `mli_inst_get` return to a local
(`let inst = mli_inst_get(f, slot)`) lands a copy whose operand `id` fields
read as the sibling `tag` fields (one-slot shift), while per-slot column
accessors returning scalars or one 88 B `MliOperand` read true data.
Measured: `dst.id=1 src1.id=1 src2.id=1` for a `mul v2, v0, v1` whose pool
columns hold `2 / 0 / 1` (accessor read confirms, and `mli::dump` — which
forwards the same return directly as a call argument — printed the true
program from the same pool). CRITICALLY: the S1 gate (5-module closure) ran
the SAME `let`-landing code and read true data — the corruption appears only
in the larger closure, placing this in the imported-module D3 residual
family. Consequence taken: `mli::verify`, `mli::dump` and `mli::interp` read
exclusively through per-slot accessors (`mli_slot_opcode` / `mli_slot_dst` /
…); `mli_inst_get` carries a landing-hazard warning and has no in-tree
consumers on the read path.

**SHAPE 4 — enum values passed as function arguments collapse to
discriminant 0.** `ir_binop(dst, lhs, BinaryOp::OpShl, rhs)` called from
native-v2-compiled code stores `IR_A_BIN_OP == 0` (OpShl's declaration
ordinal is 16). Enum equality also miscompiles, context-dependently: in
main-file context `e == BinaryOp::OpShl` and `e == BinaryOp::OpAdd` BOTH
return true for the same `e`; in imported-module context the translator's
enum comparisons behaved as always-false. This is adjacent to the
witness-matrix enum-discriminant residual. Consequence taken:
`mli::ir_to_mli` never touches enum values — it reads raw i64 arena columns
(`IR_A_OP`, `IR_A_BIN_OP`, …) and CALIBRATES IrOpcode ordinals at runtime
against this binary's own constructors (`s2a_calibrate`), so an enum
reordering cannot silently desynchronise the mapping; gate fixtures poke
`IR_A_BIN_OP` ordinals directly because constructor enum arguments cannot
carry them.

Acceptance gate for this addendum: `inst_landing_diag.sio` prints
`NOT_OBSERVED` for both shapes under a session-built Madaros.

---

*Reporter: cursor-2. Dispatch requested by fleet-orchestrator (claude-1)
after the S1 handoff, 2026-08-16. Addendum added during the S2a tranche.*
