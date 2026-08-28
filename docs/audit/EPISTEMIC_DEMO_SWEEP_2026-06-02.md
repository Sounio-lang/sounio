<!-- docs:meta
topic_id: repo.docs.audit.epistemic-demo-sweep-2026-06-02
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.epistemic-demo-sweep-2026-06-02
-->

# Epistemic / dissertation demo sweep — research-artifact integrity (2026-06-02)

Compiled + ran all `examples/dissertation_*.sio` + `epistemic_*.sio` + clinical/PBPK demos
with the working `bin/souc` (md5 `9d4ef541`), `ulimit -s 1048576`, 30s run cap.

## Result: 48 demos — **31 GREEN**, 8 compile-fail, 9 run-fail (after the olanzapine fix)

The GUM/Knowledge<T> clinical-epistemic path is healthy: vancomycin TDM (ISO 7-source
uncertainty budget + Knightian p-box gate), tirzepatide, epistemic_propagation, and the
bulk of the dissertation suite compile and run clean.

### FIXED this session
- **dissertation_olanzapine_demo** — was hung (timeout). `olanzapine_pbpk_params_smoker`
  returned `cl_hepatic=0.0` (copy-then-reassign on the 14-field PBPKParams14 miscompiles to
  0; param-into-literal hits the same large-struct bug — constant literal is the fix).
  Now rc=0, smoker CL=50.0, SS plasma halved. Commit `a3ea42082`.

### Broken — categorized by cause (for prioritization; not all are regressions)
- **GPU / requires CUDA (expected to fail CPU-only):** epistemic_gpu_pipeline,
  epistemic_quantum_vqe (compile-fail = GPU intrinsics/kernel surface).
- **ML-heavy / likely incomplete or env:** epistemic_mcts, epistemic_mcts_full,
  epistemic_ml_demo, epistemic_classifier (rc=3), epistemic_transformer (rc=3),
  epistemic_kan_trained (rc=1).
- **Compiler / language gaps (the real, fixable-at-source-but-compiler-rooted class):**
  - clinical_curvature_analysis — COMPILE: "unknown method" + "effect not declared"
    (stdlib API drift; the demo calls a renamed/removed method).
  - epistemic_dempster_shafer, epistemic_refinements, epistemic_preictal_workflow —
    COMPILE fail (uninvestigated; likely API/feature gaps).
  - epistemic_viz_demo — rc=139 (SIGSEGV at runtime).
  - epistemic_lm_fileio (rc=1), native_epistemic_pk (rc=63).
- **Clinical-numeric (real regression to triage):** dissertation_pbpk_rapamycin — runs but
  its internal test harness reports "Passed 2, Failed 5" (rc=5). 5 assertions drifted;
  needs per-test triage (clinical/numeric, Demetrios's domain).

## The compiler bug the olanzapine fix exposed (generalizable)
On the current `bin/souc`, **reassigning a field of a 14-field struct** (`var p = ctor();
p.field = ...; return p`) **AND passing a value through a fn parameter into a large struct
literal field both miscompile the field to 0.0**. Only constant literals are reliable. This
is the same large-struct value-move family as the modular-compiler issues — it bites
ordinary stdlib code, not just the compiler. Any stdlib that does copy-then-reassign on a
big params struct is suspect; prefer full constant literals.

## Honest scope note
The broken demos are mostly broken because of *compiler/language* limits (large-struct
miscompile, missing GPU/ML surface, API drift) — NOT clinical content. So "fixing the
research demos" largely routes back into compiler work. The olanzapine fix was the one
cleanly workaround-able at source (constant literal). The rest split into: GPU/ML (env),
compiler-rooted (the blocked axis), and pbpk_rapamycin's numeric test drift (clinical
triage). 31/48 green is the current research-artifact baseline.
