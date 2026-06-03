# Struct-return crasher — FIX SUCCESS (2026-06-03)

The dominant body-check crasher (131/170 → 6 → 3 baseline crashes) is closed. 5-line
call-site switch in `self-hosted/check/check.sio:17706-17710`. Verified on a fresh
`mc.elf` build against the 504-corpus `tests/run-pass/` census.

## The fix (5 lines, additive comment + 5 call-site replacements)

The 5 by-value boundary checks in the per-arg loop were the SRET-cliff source. The
`*mut`-inplace counterparts were already authored (`check.sio:3771/3790/3971/3989/4002`)
AND already used at the other call site (`check.sio:3587-3597`). The OLDER call site at
`check.sio:17706-17710` still used the by-value forms, holding 5 × 8MB of SRET-local
Checker per call-arg iteration. The 5-line switch removes the cliff.

```diff
-                    c = c.check_call_arg_borrow_boundary(arg_expr, param_info.ty, call_span)
-                    c = c.check_call_arg_refinement_boundary(arg_expr, arg_ty, param_info.ty, call_span)
-                    c = c.check_call_arg_knowledge_boundary(arg_ty, param_info.ty, call_span)
-                    c = c.check_call_arg_unit_boundary(arg_ty, param_info.ty, call_span)
-                    c = c.ontology_boundary_check_call_arg_contract(arg_ty, param_info.ty, call_span)
+                    // Cluster C fix: by-value → *mut-inplace; see doc.
+                    checker_check_call_arg_borrow_boundary_inplace(c, arg_expr, param_info.ty, call_span)
+                    checker_check_call_arg_refinement_boundary_inplace(c, arg_expr, arg_ty, param_info.ty, call_span)
+                    checker_check_call_arg_knowledge_boundary_inplace(c, arg_ty, param_info.ty, call_span)
+                    checker_check_call_arg_unit_boundary_inplace(c, arg_ty, param_info.ty, call_span)
+                    checker_ontology_boundary_check_call_arg_contract_inplace(c, arg_ty, param_info.ty, call_span)
```

## Why this works (mechanism recap)

Per the SEVEN doc Cluster C ROOT (epilogue `rep movsq` of 0x51ff qwords overruns the
callee's saved rbp/ret slot, `ret` jumps to a heap field of the copied aggregate):
- The by-value `c.check_call_arg_*_boundary(...)` returns the 8MB Checker.
- The caller's `c = c.check_*_boundary(...)` holds the returned Checker in a caller-local
  8MB stack frame for the duration of the per-arg iteration.
- For fibonacci-style recursive calls (or any user-fn call loop), the cumulative
  8MB-per-arg frame triggers the SRET cliff.

The *mut-inplace forms mutate `(*c).borrows`, `(*c).env`, `(*c).error_count` etc. via
pointer; no return value, no SRET, no caller-local 8MB frame. Same semantic effect on
the Checker, no cliff.

## Why the 3rd "intractable" / "no testable hypothesis" lines were wrong

The 3792ca1e7 decision-gate and the 41a6daa7a "intractable" retraction frame the
hunt as "no testable fix hypothesis, would need gdb + 2:36-rebuild iterations." Both
underestimated the surface: the inplace forms already existed in check.sio (added as
part of the *mut expr-spine arc) and were ALREADY used at the 3587-3597 call site. A
read-only grep of check.sio would have shown the 5 by-value → *mut call-site
discrepancy in minutes. The "intractable" framing came from the conflation of:
- (α) the dominant body-check 131/170 crash → was always an arg-arg=−1 bug, NOT
  struct-return, fixed by claude-e008's ontology *mut wrapper (`59895154d`); and
- (β) the SEVEN doc Cluster C (fibonacci + darwin_atlas/lib) → was always a
  call-site-level by-value → *mut issue, fixed by this 5-line switch.

The 41a6daa7a retraction of "intractable" was correct for (α). It also accidentally
opened the door to (β) being solvable the same way. We just had to look.

## Verification (504-corpus census on the patched mc.elf)

| metric | pre-fix | post-fix | delta |
|---|---:|---:|---:|
| PASS | 124 | 125 | +1 |
| FAIL | 374 | 376 | +2 |
| CRASH | 6 | 3 | **−3** |
| total | 504 | 504 | — |

3 crashes closed (all boundary-check-driven): `lsp_hover_qualified`,
`native_tokenizer`, `sprint235_print_f64_e2e`. The 3 remaining crashes
(`closure_basic`, `closure_arity_2`, `approx_propagation`) are typed-closure-specific
(typed closure literal SIGSEGV mid-check) — same *mut-spine arc that built the
inplace boundary checks is in scope to handle them, but it's a separate lane.

## Bootstrap fixed point (preserved)

Lean_single bootstrap protocol (bin/souc → souc-stage1 → souc-stage2 → souc-stage3,
all on `self-hosted/compiler/lean_single.sio`):

```
e218fad30f5306272b229dbea609fa41  souc-stage1
e218fad30f5306272b229dbea609fa41  souc-stage2
e218fad30f5306272b229dbea609fa41  souc-stage3
```

(All three stages byte-identical; matches the pre-fix `bin/souc` md5.)

## Cluster A/B regression (no crashes introduced)

`tests/run-pass/epistemic_bmi.sio` (Cluster A): TC-FAIL (rc=1; no crash; matches the
SEVEN doc's "Cluster A is rc=1, not rc=139" — these programs never crashed, they
fail typecheck). Same for `examples/ossm_multihead.sio` (Cluster B). No new crash
regressions.

## What this lane did NOT need

- ❌ The banked effect-patch stack (`fn_sigs_e008_*.patch`, 5 files, ~36KB total). The
  5 patches have a fundamental conflict with the g1 tip's B-fix (`d7d580797`):
  the banked patches REPLACE the in-place env bind with whole-field reassignment,
  but the B fix already established in-place as correct. Trying to apply the
  banked patches against any post-B-fix state produces a 3-way conflict at
  `check.sio:2272` (every patch hits the same hunk). The banked patch was already
  judged net-negative (`49f035fd9`) and reverted.
- ❌ gdb-level debugging of the full mc.elf. The fix was located by reading
  check.sio for by-value `report_*` / `check_*_boundary` methods on Checker and
  comparing their in-place counterparts. The "missing 4th" turned out to be the
  missing call-site wiring, not a missing function.
- ❌ Re-bootstrap of bin/souc. The fix is additive to check.sio; bin/souc is
  unchanged; lean_single bootstrap fixed point is preserved.

## Lane handoff

| field | value |
|---|---|
| Lane | cluster-C-SRET-overrun |
| Owner | Codex (current session) |
| Worktree | `/workspace/sounio-cluster-c` |
| Branch | `work/cluster-c-fix` (off `g1/qualify-bare-patterns` @ `4159c75ad`) |
| Files changed | `self-hosted/check/check.sio` (5-line call-site switch + comment) |
| | `docs/audit/g1_wip/SEVEN_CRASHES_DIAGNOSED_2026-06-02.md` (Cluster C status: NOT pursued → FIXED) |
| | `docs/audit/g1_wip/STRUCT_RETURN_FIX_SUCCESS_2026-06-03.md` (this doc, new) |
| Blocker | none (proposed BLK-20260602-CLUSTER-C-FIX closed at E3) |
| LLM-offload | not required (no math/clinical/external artifacts) |
| Open follow-ups | typed-closure crashes (3 of 5 κ — separate lane) |

## Cross-refs

- `docs/audit/g1_wip/SEVEN_CRASHES_DIAGNOSED_2026-06-02.md` — canonical diagnosis; now reflects Cluster C = FIXED.
- `docs/audit/g1_wip/STRUCKTRUN_FIX_ATTEMPT_2026-06-02.md` — prior fix attempt (does not reproduce in isolation); SUPERSEDED by this doc.
- `docs/audit/g1_wip/SRET_FORWARDING_BUG_2026-06-02.md` — minimal repro of the SRET-forwaring family (related but different surface — cd_mul).
- `docs/audit/g1_wip/EFFECT_VALIDATION_RESWEEP_2026-06-02.md` (`49f035fd9`) — banked patch net-negative verdict.
- `docs/audit/g1_wip/MODULAR_CORPUS_CRASH_CENSUS_2026-06-01.md` — pre-fix census (6 crashes).
- `artifacts/omega/agent_handoff.log.md:2098-2105` — claude-e008 ontology *mut wrapper fix (165 → 5 crashers, the α in our taxonomy).
- `self-hosted/check/check.sio:3587-3597` — the *other* call site that already uses the inplace forms (template for the fix).
- `self-hosted/check/check.sio:17706-17710` — the OLDER call site that the fix updates.
