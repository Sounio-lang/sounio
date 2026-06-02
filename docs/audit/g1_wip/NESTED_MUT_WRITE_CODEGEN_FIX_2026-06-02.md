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

## Status / merge-readiness
The lean_single.sio codegen fix is CORRECT and validated: repro, fixed point (gen2==gen3),
**run-pass 504: 501 identical / 3 non-deterministic / 0 regressions**, crash set proven
pre-existing. Examples divergence sweep (847, the established canonical-compiler bar) is
**IN PROGRESS** — merge-readiness is pending its result; do NOT call gate-ready until the 847
examples confirm 0 real divergences. The modular-compiler E008 corpus win is gated behind the
separate deeper-check crash class above. a64 dispatch: follow-up.
