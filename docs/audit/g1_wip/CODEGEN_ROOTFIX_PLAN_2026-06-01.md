<!-- docs:meta
topic_id: repo.docs.audit.g1-wip.codegen-rootfix-plan-2026-06-01
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.g1-wip.codegen-rootfix-plan-2026-06-01
-->

# Root-fix plan: bin/souc codegen bugs #2/#3 (by-value 164KB Checker copy)

**Decision (2026-06-01):** fix the bin/souc CODEGEN root of bugs #2+#3 at once,
instead of the per-function `*mut` grind (the 840-line WIP in this dir works
around the symptom; the root fix makes it unnecessary). **NOT started this
session** — editing `lean_single.sio` + re-bootstrapping is the repo's highest-
risk operation and must be done in FRESH context, never at build #33 of an
exhausted session. This doc is the read-only groundwork so the next session
executes precisely.

## Exact codegen sites (verified by source read, no build)

All in `self-hosted/compiler/lean_single.sio` (the lean_single seed = bin/souc
fixed point; 32,589 lines):

| site | line | what it does |
|------|------|--------------|
| `local_bss_spill_bytes()` | 923 | returns **524288** (512KB) — the threshold below which an aggregate is copied onto the **local stack frame** instead of BSS |
| `materialize_aggregate_expr_x86` | 7034 | materializes a struct-like value arg; if `nslots*8 < 512KB` → `emit_bulk_copy_to_slots_x86(nslots, base)` = full stack copy (7063-7065) |
| `emit_bulk_copy_to_slots_x86` | 7008 | the per-slot / `rep movsq` copy onto `[rbp - base*8]` |
| `copy_agg_into_struct_slots_x86` | 7069 | per-field aggregate copy (match-arm materialization) |
| SRET return | 7196-7225 | return-of-large-struct via r12 + `rep movsq` — **this part is correct/in-place**, not the bug |

## The root cause (precise)

`struct Checker` (check.sio:105) is ~164KB ⇒ ~20,500 slots, which is **< the
512KB spill threshold**. So EVERY by-value Checker materialization — a `match`
arm that yields `(Checker, X)`, a `(*c).method()` call whose `self: Checker` is
copied, an arg of type Checker — emits a full ~164KB slot-by-slot copy **onto
the caller's stack frame**. N such sites in one function → N×164KB frame (the
measured 11.7MB frames). Under recursion this overflows the 8MB stack fast (bug
#2); a single `(*c).method()` self-copy can blow it immediately (bug #3, VmStk
flat 132kB then crash). Bug #1 (already fixed) was the `match`-disc-0 miswire
that *routed* into this path.

## Fix options (for the fresh session to choose + prototype)

1. **Lower/condition the spill threshold for huge aggregates.** Make
   `local_bss_spill_bytes()` (or the call sites at 7040/7055/7040) send
   aggregates above e.g. 64KB to a BSS/heap temp (the 7055-7061 branch already
   does exactly this for ≥512KB — just widen when it triggers). Smallest change;
   risk = BSS temp lifetime/reentrancy for recursive calls (each level needs its
   own temp, so a bump-allocator keyed on call depth, NOT a fixed RET_AGG_BSS_OFF).
2. **Pass large aggregates by hidden reference** (caller materializes once, passes
   a pointer; callee reads through it). Closer to a real ABI fix; bigger change
   across arg-passing (1702-1785) + materialization + method `self`.
3. **Honor Sounio's linear types with move/in-place** (the original
   [[project_sota_linearity_move_arch]] idea) — soundest but largest.

Recommended first probe: **option 1**, narrowly — only for `type_is_struct_like`
aggregates above a new mid threshold, with a depth-indexed BSS bump temp — then
gate with the existing harness.

## Re-bootstrap discipline (CRITICAL — this is editing the canonical compiler)

`bin/souc` is the self-reproducing fixed point of `lean_single.sio`. Editing it
means:
1. Edit `lean_single.sio`.
2. Build gen1: `bin/souc self-hosted/compiler/lean_single.sio /tmp/g1` (old souc
   compiles the edited source).
3. Build gen2: `/tmp/g1 self-hosted/compiler/lean_single.sio /tmp/g2`.
4. Build gen3: `/tmp/g2 ... /tmp/g3`; **require gen2==gen3 (md5)** = fixed point.
5. Run the FULL canonical gate + wide run-pass/compile-fail sweep BEFORE
   replacing `bin/souc`. A miscompiling intermediate bricks the whole toolchain.
6. Only then install gen3 as `bin/souc`.
⚠️ A bad edit here breaks EVERY build in the repo, not just `--check`. Serialize;
one edit; full fixed-point + sweep each time.

## Verify the fix worked

Rebuild the modular `mc.elf` with the NEW bin/souc (no check.sio *mut changes
needed if the codegen fix is real) and confirm the original crashes are gone at
the SOURCE level:
- `( ulimit -s 8192; mc.elf --check )` on `fn main(){1+2*3}`, `if 1==1{}`,
  `while 1==1{}`, `f()` → rc=0, VmStk peaks at KB-MB not GB.
- `scripts/ci/g1_expr_recursion_gate.sh` PASS.
- The `*mut` if-chains from 0afad182c can then be KEPT (harmless) or reverted;
  the WIP 840-line patch becomes unnecessary.

## State at this doc's writing
- HEAD `56178f241` (WIP patch banked), tree clean.
- `let x=1` fix live at `0afad182c` (bug #1).
- WIP `*mut` expr-spine: `g1_mut_expr_spine_wip.patch` (applies clean), validated
  binary `mc_g2c.elf` (md5 02eb92bc…, gitignored).
- Memory: `project_g1_second_bug_compound_expr` + `project_mut_increment1_landed`.
