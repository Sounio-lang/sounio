# Nested deref field/array store codegen fix — 2026-06-02

**Branch:** `codegen/deref-nested-store` (off `g1/e008-bridge-fix` @ `4bab1996a`)
**Commits:** `6d8326d37` (explicit `(*p)` forms) + `0f3628957` (auto-deref pointer-root forms)
**File:** `self-hosted/compiler/lean_single.sio` (x86 codegen)

## The bug (was the handoff's "NEXT CODEGEN BUG")

Two-level nested field/array stores through a pointer/ref did not persist — they
were lowered as a write into a **discarded by-value copy** of the inner struct, so
they read back as 0. One-level writes through the same pointer worked, which made
the bug subtle. This is the build-independent `NESTED_MUT_WRITE` bug and the direct
cause of the in-place checker's collectors losing state (the #1 front-half lever:
E008×122 + E170×27).

Four syntactic shapes, all two-hop `*.f1.f2`:

| form | example | base | fixed |
|------|---------|------|-------|
| explicit-deref scalar | `(*c).fn_sigs.count = n` | lost (0) | ✅ |
| explicit-deref array  | `(*c).fn_sigs.entries[i] = sig` | lost (0) | ✅ |
| auto-deref scalar (ptr root) | `c.algebras.count = n` | lost (0) | ✅ |
| auto-deref array (ptr root)  | `c.studies.entries[i] = x` | lost (0) | ✅ |

## Form-match against the live checker (self-hosted/check/check.sio)

- Primary `fn_sigs` collector (check.sio:2278-2281) uses **explicit** `(*c).fn_sigs.count`
  / `(*c).fn_sigs.entries[i]` → covered by `6d8326d37`.
- algebras/studies/ontologies collectors (check.sio:13082/13153/13279) use **auto-deref**
  `c.algebras.count` etc. → covered by `0f3628957`.

Both write shapes the checker uses are now handled.

## Fix

New token-shape detectors + x86 compilers mirroring the existing single-field
deref-store codegen, but resolving a second **inline** struct hop and summing both
field offsets (`emit_store_to_pointer_offset_x86(lslot, foff1+foff2, …)`; arrays
add `lea` + indexed store). Auto-deref dispatch is **gated on
`token_chain_root_is_pointer(EP)`** so value-struct roots fall through to the
generic path unchanged (no regression).

## Validation (worktree, bootstrap = main bin/souc 9d4ef541)

- repro/nested_deref_field_store_min.sio, repro/nested_autoderef_field_store_min.sio,
  repro/nested_deref_aggregate_array_elem.sio — all green.
- **tests/run-pass/nested_mut_ref_struct_field.sio** (pre-existing G1b regression test,
  `&!Outer` nested store) flips rc=1 panic("FAIL") → rc=0 "OK" — a real test-asserted win.
- Bootstrap fixed point holds: gen2==gen3 (md5 `bd35d8ed…`).
- run-pass sweep (504): identical compile set 476/476, 0 compile divergences,
  1 improvement (above), 0 real run divergences (3 apparent = ASLR address prints,
  non-deterministic in the baseline too).

## Scope / open

- **x86 only.** The a64 dispatch twin (compile_stmt a64 path, ~line 29800+) is
  **UNMODIFIED** — still falls through. No cross-arch coverage.
- 3-or-more-hop stores (`(*p).a.b.c=`) still fall through (out of scope; rare).
- Value-struct (non-pointer) `o.f1.f2` nested store is a **separate** pre-existing
  gap on this base, tracked by the main checkout's `stmt_is_nested_field_store` WIP.
  Left to the generic path here (gate returns false).

## Next step to realise the lever (NOT done here)

This is the **codegen fix only**. To convert the E008/E170 census numbers, rebuild
the modular compiler (`self-hosted/compiler/main.sio` bundle) with a bin/souc built
from this lean_single.sio, then re-run the corpus census. The prior naive source
work-around was net-negative (PASS 125→112, CRASH 3→170); this codegen fix lets the
in-place collectors use cheap nested writes instead — expected strictly better, but
**unmeasured** until that rebuild + census runs.
