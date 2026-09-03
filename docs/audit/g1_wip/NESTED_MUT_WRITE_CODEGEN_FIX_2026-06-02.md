<!-- docs:meta
topic_id: repo.docs.audit.g1-wip.nested-mut-write-codegen-fix-2026-06-02
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.g1-wip.nested-mut-write-codegen-fix-2026-06-02
-->

# Nested-`*mut`-write codegen bug — FIXED + validated; E008 spurious class eliminated (2026-06-02)

Branch `codegen/nested-mut-write-fix` @ base `5082bf67e` (fall-through-fixed). Commit `89ddc753b`.

## The bug (separate from the fall-through bug)
`lean_single.sio` codegen pattern-matches assignment shapes. `(*name).field =` and
`(*name).field[i] =` had recognizers; `(*name).field.field =` and `(*name).field.field[i] =`
did NOT, so two-level nested `*mut` writes fell to the generic path, which materialised the
INLINE intermediate struct field as a value-copy, wrote into the copy, and discarded it
(read back as 0). Minimal repro (`NESTED_MUT_WRITE_REPRO_2026-06-02.sio`): `top=9 n=0 v0=0`.

## The fix
Added `stmt_is_deref_field_field_store` / `stmt_is_deref_field_field_array_store` recognizers
+ `compile_deref_field_field_store_x86` / `compile_deref_field_field_array_store_x86` (flat
`[ptr + off(f1) + off(f2)]` addressing, reusing `emit_store_to_pointer_offset_x86` and the
deref-field-array store path) + wired both into `compile_stmt` before the generic path. x86.

## Validation (all PASS)
- Repro: `top=9 n=0 v0=0` → **`top=9 n=3 v0=7`** (both scalar + array nested writes persist).
- Bootstrap fixed point: **gen2 == gen3** (md5 `ad9bf234`) — modified compiler self-reproduces.
- Run-pass divergence sweep (old `bin/souc` vs fixed `souc_gen2`): **504 total, 501 identical,
  3 divergent** — all 3 confirmed NON-deterministic (stack-address printers; same old binary
  varies per run). **0 real regressions.**

## E008 payoff (modular compiler rebuilt with the fixed compiler, NO check.sio change)
The in-place collect's lost writes (`(*c).fn_sigs.entries[i]=sig`, `(*c).fn_sigs.count=`,
`(*c).env.bindings[i]=`, `(*c).env.count=`) are exactly these shapes. Rebuilt mc.elf via
`souc_gen2` and censused 504 run-pass:

| | PASS | E008 spurious "expected ()" | E008 real | CRASH |
|---|---:|---:|---:|---:|
| baseline (old compiler) | 125 | 122 | 0 | 3 |
| nested-write-fixed compiler | 112 | **0** | 45 | 170 |

- **100% of the spurious "expected ()" E008 class (122) is eliminated** — `fn f()->i64{return 5}`
  now `check: OK`; the silent body-type hole is closed (`fn f()->i64{"hello"}` correctly E008).
- The 45 remaining E008 are REAL (e.g. `main()->i32{0}` i64-literal-vs-i32 width — a separate
  frontend int-literal-narrowing gap, not this bug).

Note: the E008 payoff exercises the fix's HARD path, not just the synthetic repro — the real
collect site `(*c).fn_sigs.entries[sig_id] = sig` is a **variable-index, aggregate (FnSig)
element** store on the live 8 MB Checker (the repro used a constant index + i64 element), and it
worked (count 0→2, `find` resolves, matching-type programs `check: OK`).

## Honest limit — the corpus is still net-negative
PASS 125→112, CRASH 3→170. The crash SET is **identical** to my earlier `check.sio` source-fix:
both produce exactly 170 crashers — **170 common, 0 unique to either** (`comm`-verified,
`census_fix3` vs this census). Two different mechanisms (source `.add`+write-back vs codegen
nested-write fix) reach the SAME programs ⇒ the crashes are a pre-existing latent class,
**definitively not introduced by this fix**. They live in the modular checker's deeper `*mut`
check spine, newly REACHED because the checker now actually type-checks bodies. Crashers crash
during "Type checking module 0" on VALID programs (`souc_gen2` compiles them fine), 2–8
functions, diverse (`array_elem_field_store`, `array_mut_ref`, `approx_*`).
=> "one codegen fix unblocks both E008 and crashes" is REFUTED: it unblocked E008, not the
crashes. The next codegen hunt is this deeper-check crash class.

## Status / merge-readiness — MERGE-READY for the canonical compiler
The lean_single.sio codegen fix is CORRECT and fully validated against the established bar:
- repro `top=9 n=3 v0=7`; bootstrap fixed point gen2==gen3 (`ad9bf234`);
- **run-pass 504: 501 identical / 3 non-deterministic / 0 real regressions**;
- **examples 847: 847 identical / 0 divergences / 2 HANG_BOTH** (`TOTAL=847 SAME=847 DIVERGE=0
  HANG_BOTH=2`). The 2 hangs (e.g. `alphageozero_final.sio`) are PRE-EXISTING — they time out
  identically on BOTH the old and fixed compiler (rc=124 each side), so they are not a
  divergence and not introduced by this fix;
- crash set proven pre-existing (170 common / 0 unique vs the source-fix census).

=> Zero behavioural change on 504 run-pass + 847 examples; the change can only affect
previously-miscompiled two-level nested `*mut` writes. Ready for canonical-compiler review/merge.
The modular-compiler E008 corpus win remains gated behind the separate deeper-check crash class
above (next codegen hunt). a64 dispatch: follow-up.

## FOLLOW-UP 2026-06-02: deeper-*mut-check crash class — dominant bug fixed (170 -> 5)
The "separate deeper-check crash class" above was hunted and largely fixed (commit 59895154d,
check.sio source — modular-compiler checker, distinct from the lean_single.sio codegen fix).

ROOT CAUSE (one bug, 165 of 170 crashers): `checker_ontology_boundary_check_call_arg_contract_inplace`
(check.sio:4012) — despite its `_inplace` name — called the by-value method
`(*c).check_call_arg_ontology_boundary(...)`, copying the 8MB Checker as `self` on EVERY
user-function call argument. That copy was the stack smash (rip in stack, page-aligned frames).
The other four arg-boundary checks (borrow/refine/knowledge/unit) were already true *mut; the
ontology one was missed. It is exposed only because fn_sigs now persists (the arg-checker runs).

HUNT METHOD (reusable): smallest crasher `ir_disasm_basic` -> minimal repro `fn id(x:i64)->i64{x}
fn main()->i32{let y=id(5) 0}` -> trigger = user-fn call with >=1 arg (builtins don't crash) ->
gdb (rip in stack, frames 0x1000 apart) -> entry markers (DBG_CCE/DBG_CCAI each fire ONCE => not
recursion) -> per-boundary markers (DBG_M5_onto last, M6 never) -> the ontology boundary -> read
the by-value method, saw its immediate early-out for non-ontology params.

FIX: hoist that early-out into the *mut wrapper (no-op unless the PARAM carries an ontology
contract) so the 8MB by-value copy only happens for real ontology args. Behaviour-identical.

RESULT (modular census, 504 run-pass):
| | PASS | FAIL | CRASH |
|---|---:|---:|---:|
| baseline (old compiler) | 125 | 376 | 3 |
| nested-write codegen fix only | 112 | 222 | 170 |
| + ontology guard | **151** | 348 | **5** |
PASS now 151 > 125 baseline (NET POSITIVE); spurious E008 class still 0; CRASH 170 -> 5.

REMAINING 5 (long tail, 3 distinct constructs, each a separate bug — same hunt method):
- typed closure `|x: T|`: approx_propagation
- Knowledge<T>: epsilon_comparison_valid, knowledge_octonion_inner
- Seq<T>: seq_borrow, seq_struct_elems

## FOLLOW-UP 2026-06-03: long tail CLOSED — modular CRASH 5 -> 0; bin/souc swapped
The 5 survivors were eliminated by four more `*mut` checker transcriptions plus one
canonical codegen feature. Each `_inplace` transcription replaces a by-value `(*c).method()`
bridge (8MB Checker copy) with a true *mut spine; the codegen feature implements a construct
the canonical compiler never lowered.

| commit | construct | site | modular CRASH |
|---|---|---|---:|
| 59895154d | (prior) ontology arg-boundary | check.sio:4012 | 170 -> 5 |
| b2fdeed5e | closure expr | `checker_check_closure_expr_inplace` | 5 -> 4 |
| 7002bf61f | method call w/ base ty | `checker_check_method_call_with_base_ty_inplace` | 4 -> 3 |
| 9d5ffac8f | for-in expr | `checker_check_for_in_expr_inplace` | 3 -> 2 |
| 5dce10672 | tuple patterns in match | lean_single.sio codegen | 2 -> **0** |

The last 2 crashers (epsilon_comparison_valid, knowledge_octonion_inner) were NOT a checker
*mut bug — they were `match (a,b){(Some(x),Some(y))=>...}` tuple patterns, which the canonical
compiler's arm parser had no `(` case for, so it mis-parsed and emitted a garbage deref
(SIGSEGV — crashed the OLD bin/souc too, pre-existing). Codegen now handles tuple patterns:
elements are stored INLINE (tag at `[tuple+off]`, payload at `[tuple+off+8]`); per-element tag
tests jump to a shared skip; payload binds are typed via the tuple cache's element Option
inner-type so nested `*ia` derefs lower correctly. x86 only (a64 dispatch follow-up).

VALIDATION (tuple-match feature):
- repros teq/teq2/teq3 correct (equal trees true, diff-kind false, no crash);
- bootstrap fixed point gen2==gen3 = **c634b38f** (compiler self-reproduces with the feature);
- run-pass 504: 501 identical / 3 non-deterministic (covid_2020_kernel, epsilon_comparison_valid,
  knightian_syntax — all confirmed: old binary's own output varies per run) / **0 real regressions**;
- modular census 504: **PASS 151 / CRASH 0** (was PASS 151 / CRASH 2).
- examples sweep (souc_gen2 vs tm_genA, tuple-match delta): **TOTAL=847 SAME=845 DIVERGE=2
  HANG_BOTH=2**. Both divergences are 10 s RUN-timeout (rc=124) artifacts on compute-heavy
  sweeps, classified non-real:
  - `examples/erdos/cdcl_proof.sio` — CDCL proofs for growing K_n; old reached mid-K_10 one
    run / mid-K_9 another, new mid-K_9; output byte-identical up to the cutoff.
  - `examples/zd_ratio_sweep.sio` — zero-divisor ratio sweep; old's own cutoff line varies by
    load (non-deterministic), and **new's output == old-run1 byte-for-byte**.
  The 2 HANG_BOTH time out at COMPILE on both sides (pre-existing, identical). **0 real
  divergences across all 847 examples.**

ARC (modular crash count across the whole hunt): baseline 3 -> nested-write-fix 170 (reached the
latent deeper-check class) -> ontology 5 -> closure 4 -> method 3 -> for-in 2 -> tuple-match **0**.
PASS 125 (baseline) -> 151. Net: every modular crasher on the 504-program corpus eliminated, +26 PASS.

CANONICAL bin/souc SWAP (commit 96528f1d7): bin/souc re-bootstrapped to the tuple-match fixed
point. `canonical_compiler_gate.sh` PASS — bin/souc md5 == self-compile md5 == c634b38f. Prior
bin/souc ff68f758 (fall-through-fixed, no tuple-match) backed up at
/tmp/souc.backup-pre-tuplematch-bin. The 4.98MB artifacts/self-hosted binary is a separate
legacy CLI (not the gate's canonical target) and was left untouched.
