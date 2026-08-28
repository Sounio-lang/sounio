<!-- docs:meta
topic_id: repo.docs.audit.madaros-for-loop-lowering-2026-06-20
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-for-loop-lowering-2026-06-20
-->

# Madaros native_v2: `for i in a..b` range-loop lowering (2026-06-20)

Branch `fix/madaros-for-loop-lowering` off `origin/main` @ `659492156`.

## Bug
`for i in 0..10 { … }` builds with `native_v2_compile: front-half failed: ir_bodies_failed`
(no ELF, rc 0). `--check` passes; `while` loops work.

Root cause: `ExprForIn` has **no case** in the IR lowerer's expression dispatch
(`lower.sio` `lower_expr_ref`) — the only reference was a debug `print("for_in")`. Every
`for` loop therefore fell through to the dispatch's final `else` → `report_error()` →
`ir_bodies_failed`. For-loops were simply never lowered.

## Fix (this commit)
Add `lower_for_in_expr_ref`, desugaring `for i in start..end { body }` into the **same IR
shape the working `while` loop emits**:
```
i = start
l_top:  cond = (i < end)        ; i <= end for inclusive (..=) ranges
        if !cond goto l_end
        body
l_cont: i = i + 1               ; continue target AND fallthrough
        goto l_top
l_end:
```
- Loop var bound via `bind_local` (mutable, stable slot), incremented via `ir_copy`.
- `push_loop_labels(l_cont, l_end)` so `continue` jumps to the increment (not skipping it)
  and `break` exits — matching the for-loop contract.
- Inclusive vs exclusive via `ExprRange.bin_op` (`OpRangeInclusive` → `OpLe`, else `OpLt`).
- Bound evaluated **once** before the loop (matches lean_single semantics — required for the
  gen2==gen3 fixed point, since the compiler's own source uses `for i in 0..n`).
- New dispatch case `ExprForIn → lower_for_in_expr_ref`.

## Safety analysis (why this is correct without a local run)
- **No regression:** before this change all `for` loops hit `report_error()`. After it,
  range-`for` lowers; non-range iterables (`for x in arr`) and half-open ranges
  (`for i in 0..`, `.right == None`) **still** `report_error()` — explicit guards, no
  silent miscompile.
- **`end_reg` survives the loop back-edge:** `build` emits the **core_ir dedicated-slot
  model** — `ir_slot_offset(vreg) = -(vreg+1)*8`, one never-reused stack slot per vreg
  (confirmed by disassembly: generated code uses sequential `-0x8/-0x10/-0x18/-0x20`
  slots). `end_reg` is written once before `l_top` and only read inside; nothing else
  writes its slot. This is the exact mechanism by which a `while` loop's outside-initialised
  counter `i` survives the back-edge — proven working (`while i<n {…}` → correct sums).
- Type-check-neutral: `--check self-hosted/compiler/main.sio` = 755 errors with and without
  the change (pre-existing prebuilt-vs-bundle noise).

## Validation status — PENDING REBUILD
The prebuilt binary cannot exercise a source change. After a madaros rebuild
(CI `madaros-prebuilt-refresh.yml` ref=`fix/madaros-for-loop-lowering`, or batched with
Codex's rebuild), run this matrix (exit-code verified, avoiding the still-broken int-println):
```sounio
fn main()->i64 { var s=0  for i in 0..10 { s=s+i }  s }          // expect 45
fn main()->i64 { let n=10  var s=0  for i in 0..n { s=s+i }  s }  // expect 45 (end as live value)
fn main()->i64 { var s=0  for i in 0..=10 { s=s+i }  s }         // expect 55 (inclusive)
fn main()->i64 { var s=0  for i in 0..0 { s=s+1 }  s }           // expect 0  (empty)
fn main()->i64 { var s=0  for i in 0..10 { if i==5 { break }  s=s+i }  s }     // expect 10
fn main()->i64 { var s=0  for i in 0..5 { if i==2 { continue }  s=s+i }  s }   // expect 8 (continue increments)
fn main()->i64 { var s=0  for i in 0..3 { for j in 0..3 { s=s+1 } }  s }       // expect 9 (nested)
fn main()->i64 { var a:[i64;3]=[10,20,30]  var s=0  for i in 0..3 { s=s+a[i] }  s }  // expect 60
// for x in arr  ->  must still fail cleanly (report_error / ir_bodies_failed), no crash
```
This fix is plausibly on the **gen2==gen3 critical path** (the compiler uses `for i in 0..n`),
unlike the int-println fix.
