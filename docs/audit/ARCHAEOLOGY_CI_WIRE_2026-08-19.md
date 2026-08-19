<!-- docs:meta
topic_id: repo.docs.audit.archaeology-ci-wire-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.archaeology-ci-wire-2026-08-19
-->

# Archaeology CI wire — semantic declaration

Written **before** any edit to `.github/workflows/ci.yml`.

A ladder the CI does not name is a document. This lane does not change a
TypeKind, an effect, or a registry concept. It changes whether a derived
position is allowed to rot unseen.

```
Semantic-Lane-ID: archaeology-ci-wire-20260819
Owner: grok-cli5
Concept-IDs: none
Intent-Preserved: a refuse that starts to pass is a fallen blocker; a
  pass that starts to fail is a regression; both must stop the merge.
  Positions remain derived from fixtures. Wiring does not rewrite them.
Transformation: the gates that measured green on origin/main become
  named steps in .github/workflows/ci.yml, by the same explicit-list
  pattern as "Canonical lean_single fixed point"
  (canonical_compiler_gate.sh). No glob. No new mechanism. A gate that
  measured red is named in this declaration and is NOT wired.
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: the named steps are reachable because a deliberate
  failing commit made CI red on those steps, and that failure was
  reverted before merge.
Claims-Forbidden: "the archaeology is on CI" after only a grep of
  ci.yml; "SKIP is PASS"; wiring a gate that was red on origin/main
  (that would paint main red and blame the wirer).
Assumptions: origin/main at measurement is 1dc0df549d. Madaros is the
  engine. Inherited SOUC_BIN is poison and is unset. No compiler
  rebuild on the pod; fixtures run on the committed ELF. Heavy rebuilds
  go through Slurm/Foundry if they become necessary (they should not
  for this wire).
Write-Set: .github/workflows/ci.yml
  docs/audit/ARCHAEOLOGY_CI_WIRE_2026-08-19.md
Read-Set: scripts/ci/effect_archaeology_gate.sh
  scripts/ci/typekind_archaeology_gate.sh
  scripts/ci/typekind_archaeology_c.sh
  scripts/ci/canonical_compiler_gate.sh
  .github/workflows/ci.yml (Contracts job, explicit run: list)
Positive-Witness: on origin/main, each gate that will be wired exits 0
  (measured this turn, before the yaml edit).
Negative-Witness: a commit that breaks one named fixture makes the
  corresponding CI step red. Reverted before merge.
Acceptance-Gate: each green gate appears as a literal
  `run: bash scripts/ci/…` line in ci.yml AND the negative CI proof
  exists on the PR. The red gate is absent from ci.yml.
Integration-Target: origin/main
Authoritative-Only-If: the negative proof (CI red because of the new
  step) is on the PR, then reverted.
```

## Order (binding)

1. Measure the three gates on current `origin/main`. Report. Do not wire a red gate.
2. Wire only the green ones, same pattern as the lean_single / canonical compiler step.
3. Prove reachability with a failing commit; revert the failure before merge.

## Measurement

**SHA:** `1dc0df549dbf098dd7bfb18ad07921f0ecb1faeb` (`origin/main`)  
**Engine:** this worktree `bin/souc` → Madaros v0.80.0. `SOUC_BIN` unset. No compiler rebuild.

| gate | rc | wall | what it said |
|---|---:|---:|---|
| `scripts/ci/effect_archaeology_gate.sh` | **1** | 21 s | `EFFECT_ARCHAEOLOGY_SUMMARY rows=16 failures=1`. Accuses `Chaotic`: `PASS_REGRESSION` on `tests/effects/archaeology/chaotic_pass.sio` rc=1. Re-run: typecheck OK, then `Failed to write native binary … main.elf rc=12` / `native-v2 bridge compilation failed`. Not wired. Criterion not lowered. |
| `scripts/ci/typekind_archaeology_gate.sh` | **0** | 11 s | 17 rows, 0 failures. Claim-ready primitives; F128/F256 Reserved; Family F Garden. |
| `scripts/ci/typekind_archaeology_c.sh` | **0** | 15 s | 12 Garden rows; `NoSuchType` ghost control OK. |

Positive control that a reachable gate **is** named: `.github/workflows/ci.yml` step `Canonical lean_single fixed point` → `bash scripts/ci/canonical_compiler_gate.sh` (the comment on that step is the same disease). The three archaeology scripts have **zero** hits under `.github/` on this SHA.

## Disposition after measurement

| gate | measured | wired |
|---|---|---|
| `effect_archaeology_gate.sh` | red — Chaotic `PASS_REGRESSION`, native ELF write rc=12 | **no**. Criterion not lowered. |
| `typekind_archaeology_gate.sh` | green | yes — step `TypeKind archaeology ladder` |
| `typekind_archaeology_c.sh` | green | yes — step `TypeKind archaeology family C` |

Wiring the red gate would paint `main` red and blame the wirer. The Chaotic fixture typechecks; `souc run` then dies at native-v2 ELF write. The gate uses `run` on purpose. Changing it to `check` would hide the regression.

## Negative proof

Not a grep. Two GitHub Actions runs on this PR, each red because of one named new step, then reverted.

| step | run | job | step # | conclusion | what the log said | fail SHA | revert SHA |
|---|---|---|---:|---|---|---|---|
| TypeKind archaeology ladder | [32237891458](https://github.com/Sounio-lang/sounio/actions/runs/32237891458) | [96022381923](https://github.com/Sounio-lang/sounio/actions/runs/32237891458/job/96022381923) | 44 | **failure** | `TYPEKIND_ARCHAEOLOGY_PASS_REGRESSION kind=TyI64 fixture=tests/typekind/i64/pass.sio rc=1` | `128adca981` | `77e705ce12` |
| TypeKind archaeology family C | [32238689476](https://github.com/Sounio-lang/sounio/actions/runs/32238689476) | [96025011882](https://github.com/Sounio-lang/sounio/actions/runs/32238689476/job/96025011882) | 45 | **failure** | `TYPEKIND_ARCHAEOLOGY_C_FAIL reason=layer_drift kind=Distribution indexed=parser computed=checker` | `3967e00d3d` | `be78863328` |

On the ladder-proof run, family C was **skipped** by Actions fail-fast after step 44. That is why a second fail-commit existed: SKIP is not PASS. On the family-C-proof run, step 44 (ladder) was **success** and step 45 (family C) was **failure**.

`effect_archaeology_gate.sh` has no row here. It was red on `origin/main` and was not wired.
