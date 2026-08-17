<!-- docs:meta
topic_id: repo.docs.governance.ci-trust-contract
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.governance.ci-trust-contract
-->

# CI Trust Contract

Date: 2026-08-17
Authoring lane: codex-2 / ci-trust-contract
Current anchor: `ed9dd2b903` (`origin/main` after #1773)

## Contract

The authoritative CI question in this repository is:

> Did `CI Decision` pass for the selected job set?

Raw check count is not evidence. A five-check run can be complete when Impact
correctly selects a narrow formal/docs surface. A sixteen-check run can still
be meaningless if the selected set is wrong, a selected job aborts before
evaluating, or a selected gate returns success without measuring its witness.

The trust chain is:

1. `Impact` classifies the changed paths.
2. Workflow `if:` clauses select jobs from those Impact outputs.
3. Each selected job must run its own gates honestly.
4. `CI Decision` evaluates the selected job results and fails unless every
   selected job succeeded.

Only the whole chain is a merge signal.

## Observable States

### `CI Decision` passes

A reader may conclude:

- The Impact job ran.
- The evaluator script ran.
- `contracts` succeeded.
- Every job that `evaluate_ci_decision.py` says was selected by Impact ended in
  `success`.
- Jobs that were not selected ended in `success` or `skipped`.

A reader may not conclude:

- Every possible CI job ran.
- The number of visible checks is meaningful.
- Every individual gate measured a real witness.
- Every fixture needed by every gate exists.
- The selected set was semantically sufficient for the PR.

### `CI Decision` fails

A reader may conclude:

- At least one selected job failed, was cancelled, was skipped, or was missing;
  or an unselected job ended in a state other than `success` or `skipped`.
- The run is not merge-trustworthy.

A reader may not immediately conclude:

- The code under test is wrong.

False-red causes remain possible. Examples found on 2026-08-17:

- A gate aborted before evaluation because a base ref such as `origin/main` was
  unavailable in a shallow checkout.
- `compiler_lane_status_gate.sh` encoded the authoring PR's context as a
  universal invariant and failed when main legitimately touched compiler files.

### `Impact` passes

A reader may conclude:

- The changed-path classifier completed.
- Its self-test completed.
- Impact outputs were produced.

A reader may not conclude:

- The selected job set is sufficient.
- The PR is safe.
- Downstream selected jobs ran or passed.

### A job is skipped

A reader may conclude:

- Nothing by itself.

`skipped` is correct only when `CI Decision` passes and the evaluator agrees the
job was not selected. A selected job that is skipped is a failure.

### A job passes

A reader may conclude:

- That job's process returned success.

A reader may not conclude:

- The job measured the intended witness.
- Its fixtures were present.
- It did not pass on empty input.
- Its gate assertions were invariant across arbitrary commits.

This is where the complementary censuses apply:

- base-ref/checkout defects: gates that abort before evaluating;
- missing-fixture defects: suites that are absent rather than red;
- vacuity defects: gates that return green without measuring;
- context-dependent-red defects: gates that return red without a code defect.

### Raw check count

A reader may conclude:

- Nothing authoritative.

Check count is a UI artifact. It changes with Impact selection, skipped jobs,
GitHub rendering, external integrations, merge queue shape, and whether a
workflow reached the decision job. It is not the repo's CI contract.

## Selected Jobs

As of `ed9dd2b903`, `scripts/ci/evaluate_ci_decision.py` treats these jobs as
authoritative:

- Always selected: `contracts`
- Selected for `compiler`, `runtime`, `stdlib`, `tests`, or `full`:
  `native-selfhost-linux-x86_64`, `full-test-suite`,
  `madaros-witness-gate`
- Selected for `compiler` or `full`:
  `source-bootstrap-selfhost-linux-x86_64`,
  `native-selfhost-macos-arm64`
- Selected for `compiler`, `tests`, or `full`:
  `madaros-current-source-deref-f64`
- Selected for `compiler`, `stdlib`, `tests`, `sio`, or `full`:
  `sounio-lint`
- Selected for `lean` or `full`:
  `lean-proofs`
- Selected for `website` or `full`:
  `website`

The workflow `ci-decision` job also lists `madaros-witness-gate` in `needs`,
and the evaluator includes it in the required map. A selected Madaros Witness
Gate failure is therefore not ignored by CI Decision at this anchor.

## Can CI Decision Be Fooled?

Yes, but not by raw check count.

CI Decision can be fooled if any layer below its job-result model lies or is
incomplete:

- Impact under-selects the PR because a changed path is misclassified.
- A job returns success after a gate passes on empty input, absent fixtures, or
  a skipped/failed probe that was converted into success.
- A gate aborts before evaluating but the job masks that abort.
- A job's workflow condition differs from the evaluator's required-map
  condition.
- GitHub never creates the CI Decision job, for example because the workflow
  file is syntactically invalid before the evaluator can run.

Mitigations already present:

- Unknown paths select `full` in `classify_ci_impact.sh`.
- Non-PR events select `full`.
- Changes to workflow files, the Impact classifier, the CI evaluator, and the
  Impact self-test select `full`.
- `contracts` is always required.
- `impact_ci_selftest.sh` tests representative Impact classifications and
  evaluator rejection of failed selected jobs.
- `impact_ci_selftest.sh` also verifies that every non-Impact job in
  `ci-decision.needs` is represented in `evaluate_ci_decision.py`, and that the
  evaluator does not name jobs absent from `ci-decision.needs`.

Remaining hardening:

- Extend the workflow/evaluator parity test from job-id coverage to condition
  coverage, so selector drift is caught mechanically rather than by review.
- Make each substantial gate expose a witness-count or measurement receipt, so
  CI Decision can eventually consume more than job result strings.
- Keep the complementary censuses separate: abort-before-evaluation,
  missing-fixture, vacuous-green, and context-dependent-red are distinct
  failure modes.

## Reading Rule

Use this rule when reviewing PRs or main:

- `CI Decision` pass on the expected SHA is the merge/readiness signal.
- Individual job pass is supporting evidence only.
- Check count is never evidence.
- Missing `CI Decision` means no authoritative CI verdict, regardless of how
  many other checks are green.
- `CI Decision` fail means stop and classify before blaming the code.
