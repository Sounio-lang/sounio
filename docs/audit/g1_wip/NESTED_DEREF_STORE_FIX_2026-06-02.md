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

**Type-structure precondition verified.** The fix bails (`tc_error "nested field
store requires inline struct field"`) if the intermediate field is not an inline
struct (`fty1 != 6`), which would *error-regress* the modular build if any collect-
site intermediate were boxed/pointer. Confirmed safe: all six collect-site fields
on `Checker` are value `pub struct` types (TypeEnv, FnSigTable, AlgebraTable,
StudyTable, OntologyTable, OntologyKernel — check.sio:106-116, defs.sio/env.sio),
i.e. ty 6 inline → the error branch never fires for them. A 512-element inline
array-field repro confirms large inline struct fields resolve to correct offsets
(no boxing), so the inline-offset arithmetic is valid at FnSigTable scale.

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

## Lever payoff — MEASURED (A/B census, 2026-06-02)

Built two modular compilers from the SAME `self-hosted/compiler/main.sio`, differing
only in the bootstrap that compiled them (both ~85 MB, 5 pre-existing resolve.sio
"match must be exhaustive" errors, build ~2:36 each under the lock):
- **mc_baseline** = main.sio via the UNFIXED bootstrap (ds_clean).
- **mc_fixed** = main.sio via the FIXED bootstrap (ds_fixed2).

Ran `mc --check` over the 504-prog tests/run-pass corpus (artifacts:
census_mc_baseline_2026-06-02.tsv / census_mc_fixed_2026-06-02.tsv):

| metric | baseline | fixed | Δ |
|--------|----------|-------|---|
| PASS   | 117 | 93  | **−24** |
| FAIL   | 309 | 202 | −107 |
| CRASH (all SIGSEGV) | 78 | 209 | **+131** |
| E008 progs | 137 | 51 | **−86** |
| E170 progs | 2 | 0 | −2 |

Transition matrix (baseline→fixed): 187 FAIL→FAIL, 110 **FAIL→CRASH**, 81 PASS→PASS,
78 **CRASH→CRASH (0 baseline crashers fixed)**, 21 PASS→CRASH, 15 PASS→FAIL,
12 **FAIL→PASS (genuine unblocks)**.

### Verdict (honest)

- ✅ **The lever's stated symptom is cleared.** The spurious E008 flood drops 137→51
  progs and E170 2→0 — the bridge-state root cause (in-place collect dropping
  fn_sigs) is genuinely fixed. 12 real FAIL→PASS unblocks, thematically coherent
  (issue11_ptr_deref_write, heap_vec_*, ptr_index_write).
- ❌ **But net corpus PASS goes DOWN (117→93) and CRASH explodes (78→209).** Same
  census *shape* the rejected source work-around produced (PASS 125→112, CRASH
  3→170) — because the broken collector was generating **false-passes/early-bails**;
  a working collector lets the checker actually check bodies and reach a DEEPER layer
  of latent *mut body-check SIGSEGVs. Sampled FAIL→CRASH (_diag_sobol.sio) emits 118
  lines of real type diagnostics then segfaults — progress-exposed, not a codegen
  break (the fix is proven correct + 0-regression on the bootstrap corpus).
- ❌ **Hypothesis FALSIFIED:** "one codegen fix unblocks BOTH E008 AND the ~170
  crashers." **0 of 78 baseline crashers fixed; +131 new crashers exposed.** The
  crashers are a SEPARATE disease (a deeper *mut body-check codegen layer), now
  revealed as the **dominant remaining front-half blocker** (209 SIGSEGV).

### Consequence

This codegen fix is correct and stands on its own (repros, fixed-point, 0
regressions, G1b test win, E008 root-cause cleared). But **do NOT land mc_fixed on a
green-gate basis** — net PASS is negative, same as the rejected workaround. The
roadmap is refined: E008 was a real bug *masking* the true #1 net-blocker, which is
the latent body-check SIGSEGV layer. Next lane = root-cause the 209 *mut
body-check crashers (a fresh fan-out, same gdb/repro method as this fix).
