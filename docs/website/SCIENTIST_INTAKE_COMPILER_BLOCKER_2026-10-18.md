<!-- docs:meta
topic_id: repo.docs.website.scientist-intake-compiler-blocker-2026-10-18
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.website.scientist-intake-compiler-blocker-2026-10-18
-->

# Scientist intake: residual public-main compiler blocker

## Blocker and ownership

- Blocker-ID: `SCIENTIST-PUBLIC-MAIN-2966`.
- Status: open residual; documented, not fixed. This handoff does not declare the failure nonblocking or public main healthy.
- Severity: public-main CI blocker.
- Class: compiler checker capacity / self-compile regression; separate diagnostic-reporting defect in the harness.
- Evidence level: exact failed CI job logs, directly retrieved postmerge artifact diagnostic, and repository source inspection; no local compiler execution.
- Owner: [issue #2966](https://github.com/Sounio-lang/sounio/issues/2966), OPEN with no assignee at inspection. A dedicated compiler repair owner remains unassigned. The scientist intake parent owns the intake behavior repair, not this compiler-capacity repair.
- Worktree: `/tmp/sounio-scientist-intake-20261018`; branch `fix/scientist-intake-gate-20261018`; inspected base `c4f1d3ce73987de9e2694af393711180f6a57fb1`.
- Coordination: the inspected local `sounio-coord` snapshot showed the parent's intake claims and no compiler repair claim. This is local coordination evidence, not proof that no remote agent is working. The sidecar claims only this handoff document.

## Exact failure evidence

[Postmerge run 39058562474, job 117471944458](https://github.com/Sounio-lang/sounio/actions/runs/39058562474/job/117471944458) is a failed push run at `c4f1d3ce73987de9e2694af393711180f6a57fb1`. The job is named **Madaros Current-Source f64 Lowering**, but its failing step is `scripts/ci/madaros_fixed_point_gate.sh`:

```text
[rung check] gen1 typechecks self-hosted/compiler/main.sio
           rc=1 errors=0
reached  none
expected run
MADAROS_FIXED_POINT_FAIL: stopped at rung 'none' but this tree is recorded as reaching 'run'.
```

Artifact `madaros-current-source-f64-lowering`, ID `12111241477`, was retrieved directly through the GitHub API. Its `fixed-point/check.log` contains:

```text
run_check_mode: about to check 130 modules
error[E-USE-MOD-CAP] at 1895..1920: module-qualified-call registry is full (1024 distinct lowercase `use` segments)
run_check_mode: verdict=1
```

The decisive evidence is the error code and nonzero verdict.

## Cause and misleading counter

At the inspected base, `self-hosted/check/check.sio:4835` defines the registry count and parallel tables with capacity 1024. `checker_use_module_known` deduplicates by module ID and segment text. `checker_use_module_note` at line 4916 sets `CHECKER_USE_MODULE_OVERFLOWED` when the count reaches the cap. `checker_report_use_module_overflow_inplace` at line 5170 increments the checker error count, sets `had_error`, and emits `error[E-USE-MOD-CAP]`. This is a compile-wide registry capacity wall, not evidence that a particular source span is the first overflowing entry.

`scripts/ci/madaros_fixed_point_gate.sh:208` counts only `error\[E[0-9]+\]`; line 212 uses the same matcher for its code summary. The hyphenated diagnostic is missed, producing misleading `errors=0`. Line 215 still rejects `CHECK_RC != 0`, so this is a real red gate, not a successful check. The gen2 first-failure matcher at line 227 has the same numeric-code restriction.

A minimal harness recommendation is to recognize hyphenated diagnostic codes consistently in those matchers, for example `error\[E[A-Za-z0-9_-]+\]`, and expose the first check diagnostic in the CI output. That improves reporting only; it does not repair the capacity failure or satisfy acceptance. Compiler repair must preserve overflow refusal and audit companion specializer/lowerer registries and backing buffers before changing capacities. No such changes are authorized in this intake scope.

## Prior-main comparison and lineage

[Prior run 39045939815, job 117438084161](https://github.com/Sounio-lang/sounio/actions/runs/39045939815/job/117438084161) failed at `106f7a988d5b007472d0eead724f58ac50f92200`. Its exact job log has the same `check rc=1 errors=0`, `reached none`, expected `run` stop. Artifact ID `12107025685` was identified, but retrieval of its underlying check log had not completed when the diagnosis was reported. Its diagnostic text is therefore not independently confirmed here.

The [existing #2966 diagnostic comment](https://github.com/Sounio-lang/sounio/issues/2966#issuecomment-6236067416) already records `E-USE-MOD-CAP` on earlier main `67e14779d`, run `39038505755`. Together with the directly read postmerge artifact, this establishes an existing capacity-failure lineage. It is distinct from earlier #2966 reports that passed check and failed gen2 while lowering `orbit_is_contested` with `ir_into_acc_failed`.

The inspected `git diff 106f7a988 c4f1d3ce73987de9e2694af393711180f6a57fb1 --stat` contains scientist documentation and website changes, with no compiler or gate source changes. This supports separation from the intake repair; it does not excuse leaving main red. `.github/workflows/ci.yml:1447` documents the postmerge-only rung policy, and the step at line 1463 runs on eligible push/schedule/manual events. A passing PR check does not establish that this postmerge rung passed.

## Acceptance gate and next action

Acceptance requires a dedicated compiler lane to produce current-source executable evidence on the candidate integrated SHA:

1. The rebuilt current-source Madaros checks `self-hosted/compiler/main.sio` with exit status zero and no checker errors, including no `E-USE-MOD-CAP`.
2. `scripts/ci/madaros_fixed_point_gate.sh` reaches the recorded `run` rung: gen2 compilation succeeds, the emitted executable answers as Madaros, and `into_acc_done` meets the CI floor of 122 without tripping the IR-capacity guard.
3. The gate reports `MADAROS_FIXED_POINT_OK`, and the exact integrated main SHA has a successful **Madaros Current-Source f64 Lowering** CI job. Reaching `run` does not establish gen3 success or executable fixed-point equality.

Next action: assign an owner on #2966, claim a separate compiler write set, complete the prior artifact comparison if needed, and repair the compile-wide registry capacity coherently. Improve the harness diagnostic independently, then validate through the current-source/Foundry or Slurm path and exact-SHA main CI. Do not reduce the expected rung, suppress overflow, or treat a counter-only fix as closure.

The parent continues the scientist intake behavior repair: split stages, mechanical rejection, and 24 real model probes. This document records no results for those probes. This sidecar performed diagnosis and documentation only: no compiler execution, gate run, compiler edits, commits, or offload reviews. It contains no mathematical or clinical claims. The referenced `.claude/PARALLEL_BLOCKER_CONTRACT.md` was absent from the inspected fresh worktree; the explicit handoff fields above preserve the unresolved status.
