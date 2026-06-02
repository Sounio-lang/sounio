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
- ⚠️ **Hypothesis NOT SUPPORTED (with a layout caveat):** "one codegen fix unblocks
  BOTH E008 AND the ~170 crashers." The crash count did not drop — 0 of 78 baseline
  crashers moved, +131 new crashers appear (209 SIGSEGV total). The crashers behave
  as a SEPARATE disease, not the nested-write bug.

### Determinism + layout confound (added after advisor review)

Crash/PASS sets are **per-binary deterministic** — re-running each census gives a
byte-identical crash set (mc_fixed 209≡209, mc_baseline 78≡78, PASS set ≡; symdiff=0
on all three). So the numbers are reproducible, NOT run-to-run noise.

**HOWEVER** the cross-binary crash *delta* is confounded by layout. mc_baseline and
mc_fixed differ by 386 bytes across an 85 MB binary, and this repo's modular-checker
crash is documented as layout-sensitive / non-monotonic (`project_modular_span_sensitive_crash`,
`project_modular_B_repro_verdict`: a 1-byte whitespace shift flips crash/pass on the
SAME program). Corroborating: the prior G1 census counted **6** crashers where this
build counts **78** — same checker logic, different binary, 13× crash count → crash
counts are build/layout-bound. So **"+131 exposed / 0 fixed" is a reproducible
observation but cannot be cleanly attributed to the semantic fix alone** — it is
"semantic fix ⊕ a layout perturbation," inseparable with these two binaries. The
*direction* (a nested-write fix is unlikely to repair unrelated SIGSEGVs, and a
working collector does let the checker run deeper — the sampled _diag_sobol.sio
emits 118 real diagnostic lines before crashing) is plausible, but the precise crash
counts are NOT a pure semantic measurement.

**The one robust, layout-independent signal is E008/E170** (a checker-logic
diagnostic outcome, not a memory-layout outcome): 137→51 and 2→0 stand. The
dominant-blocker reframing (front-half is now gated by a body-check SIGSEGV layer,
not E008) is the *likely* read but rests on layout-confounded crash counts — treat
as a strong lead to verify with per-program crash root-causing, not settled fact.

### Consequence

This codegen fix is correct and stands on its own (repros, fixed-point, 0
regressions, G1b test win, E008 root-cause cleared). But **do NOT land mc_fixed on a
green-gate basis** — net PASS is negative (and the crash count, the thing that would
have to improve, is layout-confounded). The *likely* roadmap read: E008 was a real
bug masking a body-check SIGSEGV layer that is now the net blocker — but because the
crash delta is layout-confounded, the next lane should first **root-cause individual
crashers** (per-program, gdb/repro, same method as this fix) to confirm they are
genuine body-check bugs rather than layout artifacts, before treating "209 crashers"
as the headline blocker count.

## Body-check crasher root-cause (2026-06-02, follow-up)

Stack-overflow split (re-run all 209 default-stack crashers under `ulimit -s
1048576`): **39 stop crashing = stack-overflow** (deep recursion / big frames, NOT
memory bugs); **170 remain genuine SIGSEGV**. Under the FAIR big-stack A/B, baseline
genuine crashers = **3** (matches the prior independent census's ~6 → layout-robust),
fixed = **170**. A 3→170 directional jump is far too large/one-sided for layout
noise.

**gdb fault clustering (all 170, big stack) — the layout-confound killer:**
- **131/170 (77%) fault at the SAME instruction `0x4c2805b`** — `mov 0x0(%rdx),%rax`
  with **rdx = -1**, reading a 16-byte `TypeEntry {ty,hash}` from address −1. A
  single shared code site rules out layout noise (which would scatter addresses).
- 38/170 fault with RIP in the stack (0x7fff…) = corrupted return / residual deep
  recursion. 1 misc (0xcd895b).

**Minimal repro of the dominant bug (131/170), build-independent, 2 lines:**
```
fn f(x: i64) -> i64 { x }
fn main() -> i64 { let y = f(5)  0 }
```
Crashes mc_fixed `--check` at 0x4c2805b. The trigger is **a call to a user function
that HAS a parameter** — `f()` (no param) PASSES, `f(5)` (one param) crashes;
`fn main()->i32{0}` alone only FAILs (E004), no crash.

**Mechanism — it is the SEPARATE, documented SRET large-struct-return bug, not the
nested-store bug:** call-argument checking (check.sio:3692 in-place path) does
`let sig = (*c).fn_sigs.get(sig_id)` — `FnSigTable::get` returns a large `FnSig`
**by value** (SRET) — then `checker_check_call_args_inner_inplace` (check.sio:3575)
calls `fn_param_list_get(params, idx) -> FnParamInfo` **by value**, whose recursive
arm `fn_param_list_get((*list).tail, idx-1)` is the exact "return another
struct-returning call's result" forwarding shape of `project_sret_forwarding_bug`.
One of these by-value struct returns corrupts `sig.params` (Box→−1) / the param info,
so the param's `TypeEntry` read derefs −1 → SIGSEGV. This path is only REACHED now
because the fn_sigs fix populates the table so call-checking proceeds past the
previously-bailing point. **The nested-store fix (this branch) and the dominant
body-check crasher are two DISTINCT codegen bugs**; fixing nested stores correctly
unmasks the pre-existing struct-return bug.

**Verified the confound is dead:** crash sets are per-binary deterministic AND the
131 cluster at one instruction with a clear param'd-call trigger and a 2-line repro —
this is a genuine bug, not layout. The "209 headline" was inflated (39 = stack); the
real dominant body-check blocker is **one struct-return bug (~131 progs), = the
already-tracked SRET forwarding family** ([[project_sret_forwarding_bug_2026-06-02]]).
Next lane: fix the large/forwarded struct-return codegen (separate from this branch);
38 stack crashers want a stack-size/frame-reduction pass.
