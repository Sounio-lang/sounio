<!-- docs:meta
topic_id: repo.docs.handoff.neurodyn-oct-mul-sign-fix-codex-execution-2026-07-07
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.neurodyn-oct-mul-sign-fix-codex-execution-2026-07-07
-->

# NeuroDyn Octonion Sign Fix Execution Report

Date: 2026-07-07
Owner: Codex
Branch: `coord/lane-8c-dossier`
Input blocker: `BLK-20260707-neurodyn-oct-mul-not-normed`

## Decision

`BLK-20260707-neurodyn-oct-mul-not-normed` is locally repaired at the source
level: the shared model/generator octonion product now uses `e2*e5 = +e7` and
`e5*e2 = -e7`.

Algebra-C remains blocked. Algebra-A/B must be regenerated and re-audited
because all previous "octonionic" results used the invalid product. A separate
runtime blocker was discovered on the default Madaros path while verifying the
compiled `.sio` multiplication harness.

## Files changed by this execution

- `examples/brain_ossm_abide.sio`
  - In `do_oct_mul`, changed `TMP_OCT[7]` from
    `... - a2 * b5 + a5 * b2 ...` to
    `... + a2 * b5 - a5 * b2 ...`.
- `scripts/research/neurodyn_octonionic_associator_manifest.py`
  - In `oct_mul`, changed the component-7 expression from
    `... - a2 * b5 + a5 * b2 ...` to
    `... + a2 * b5 - a5 * b2 ...`.

Other existing modifications in `examples/brain_ossm_abide.sio` predate this
execution and are not claimed here.

## Verification

Python generator proof:

```text
PYTHONPATH=scripts/research python3 <numeric composition/alternative harness>
composition_err 8.881784197001252e-16 alternative_err 2.059156955057807e-15
OK: valid normed alternative octonion; e2*e5=+e7
```

Static gates:

```text
python3 -m py_compile scripts/research/neurodyn_octonionic_associator_manifest.py
./bin/souc check examples/brain_ossm_abide.sio
```

Both passed.

Compiled `.sio` proof:

```text
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run /tmp/neurodyn_oct_mul_fixed_harness_scalar.sio
OK compiled oct_mul composition/alternative/e2e5
```

The same proof is not accepted as default-Madaros evidence because of the
separate runtime blocker below.

## Corrected local artifacts

Corrected local manifest packages were regenerated under `/tmp`:

- `/tmp/neurodyn_oct_mul_fixed_manifest`
  - `--triple-source continuous`
  - `--target-assoc-dim 6`
  - `--target-assoc-sign 0`
  - `--continuous-jitter 0.01`
  - `nonassociative_triples_available: 168`
  - `target_support.distinct_target_values: 56`
  - `target_support.tie_fraction: 0.0`
  - both target signs present globally and per pseudo-site
- `/tmp/neurodyn_oct_mul_fixed_unit_manifest`
  - `--triple-source unit`
  - `nonassociative_triples_available: 168`
  - associator dimensions now `1..7`; no real-part-dominant associator class

Corrected target audit:

```text
python3 scripts/research/neurodyn_algebra_c_target_audit.py \
  --targets /tmp/neurodyn_oct_mul_fixed_manifest/associator_targets.tsv \
  --manifest /tmp/neurodyn_oct_mul_fixed_manifest/octonionic_associator_manifest.tsv \
  --output-dir /tmp/neurodyn_oct_mul_fixed_target_audit --overwrite

ALGEBRA_C_TARGET_AUDIT_PASS
distinct_target_values: 56
tie_fraction: 0.0
```

Structural data audits found no bad pair construction in the corrected local
packages:

- unit package: `bad_pair_count: 0`
- continuous package: `bad_pair_count: 0`

The existing associator balance gate is not a promotion gate for the fixed-dim
continuous target and returned `ASSOCIATOR_MANIFEST_BALANCE_GATE_NOT_READY`.
That is recorded as a remaining control/gate mismatch, not as a sign-fix
failure.

## New Blocker

Blocker-ID: `BLK-20260707-madaros-f64-arg-abi-oct-mul`
Severity: B1
Class: compiler-semantics / runtime-lowering
Evidence-Level: E2 local repro
Owner: Codex / compiler lane
Worktree: `/workspace/sounio`
Branch: `coord/lane-8c-dossier`

Observed:

```text
./bin/souc run /tmp/neurodyn_oct_mul_arg_probe.sio
FAIL arg probe

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run /tmp/neurodyn_oct_mul_arg_probe.sio
OK arg probe
```

The probe calls a function with sixteen `f64` arguments and expects the callee to
observe `a2 = 1.0` and `b5 = 1.0`. The default Madaros path fails this minimal
argument-passing probe, while `lean_single` passes and a direct non-callee local
expression probe passes under Madaros.

Impact:

- Do not treat default-Madaros `do_oct_mul` runtime behavior as validated until
  this is fixed or the NeuroDyn multiplication call shape is refactored away
  from sixteen scalar `f64` arguments.
- The source-level sign fix is still correct and verified by Python plus the
  `lean_single` compiled path.

Acceptance gate:

1. A minimal multi-`f64` argument probe passes under default `./bin/souc run`.
2. The scalar compiled octonion harness passes under default `./bin/souc run`.
3. NeuroDyn `do_oct_mul` runtime evidence states which engine was used.

Next action:

Either repair Madaros multi-`f64` argument lowering or refactor `do_oct_mul` /
`do_o_step_mul` to use a runtime call shape already validated by Madaros, then
rerun the compiled proof under the default path.

## Algebra-A/B Status

The corrected local artifacts prove that the old Algebra-A/B artifact basis is
invalid: the old table had only 122 retained non-associative triples after the
broken exchangeability filter and produced impossible real-part-dominant
associators; the corrected table has 168 non-associative triples and only
imaginary dominant associator dimensions.

This report does not claim a full Algebra-A/B numerical rerun. Full re-audit
requires regenerating the historical artifact set and rerunning the relevant
decision gates on the corrected product, preferably off the main workspace.

## Algebra-C Status

Algebra-C remains blocked until all of the following are true:

1. Corrected A/B artifacts have been regenerated and re-audited.
2. `BLK-20260707-madaros-f64-arg-abi-oct-mul` is resolved or the model runtime
   path is refactored and revalidated.
3. The existing Algebra-C controls remain satisfied: genuinely continuous target
   support/tie audit, generic capacity controls including `gru_wide`, associative
   projection claim boundary, retrain nulls, and circularity ceiling.
