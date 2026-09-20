# CI layers and canonical compiler custody

This changes scheduling and artifact custody, not compiler semantics or oracle
thresholds. Targets are feedback objectives, **not measured latency guarantees**:
Fast PR Gate 10–15 minutes; Deep Compiler Gate 20–30 minutes. Timeouts retain
headroom and fail the job; they never convert incomplete work into success.

## Baseline and causal diagnosis

Source snapshot: `46802325cfaa33300113aa6450d1709063af6f77`.
Measured GitHub run: [#2553 CI, 35475067400](https://github.com/Sounio-lang/sounio/actions/runs/35475067400),
head `eb494dc277a46ea4fe0a8a1911df7e3c0b5628b8`, observed 2026-09-20 UTC.
These are a single-run sample, not p50/p95 statistics. The measured PR and the
implementation base are different commits.

| Existing job/step | Observed elapsed | Consequence |
| --- | ---: | --- |
| Impact | 1m19s | Existing checkout/classification overhead |
| Contracts | 28m13s | Serializes unrelated contracts |
| Multi-ontology composition | 13m48s | Move intact to independent Deep ontology contracts |
| Ontology frontiers | 1m29s | Same move; still runs on every PR |
| Witness fresh Madaros build | 5m47s | Share canonical producer |
| Wave 0 fresh Madaros build | 5m09s | Share canonical producer |
| Current-Source build plus f64 gates | 12m09s | Already supports explicit binary; reuse producer |
| Witness whole job | 13m41s | Keep all witnesses in Deep |
| Native Linux self-host | 3m59s | Distinct seed-chain oracle, not redundant Madaros |
| Source-bootstrap Linux | 4m19s | Distinct seed-chain/cross-target proof |
| Lean Proofs | 10m33s | Keep Lean-change PR selection; exhaustive always |
| Full Test Suite | 23m26s, **failure** | Preserve failure criteria; exhaustive qualification |
| Madaros compiles Madaros | Still running at observation | Existing workflow documents 48–54m rung; isolate |

Subtracting the two ontology steps from the observed Contracts critical path
leaves approximately 12m56s, before Impact and new runner variability. Eliminating
two fresh Madaros builds saves roughly 10–12 runner-minutes in this sample; it
does not subtract that much from wall time because they previously ran in parallel.
The large PR latency reduction comes from moving gen2 off the ordinary PR path.

## Where every class of oracle runs

| Oracle | Ordinary PR | Merge queue / push to main / dispatch | Nightly |
| --- | --- | --- | --- |
| Impact classifier + negative controls | Fast | Yes | Yes |
| Existing Contracts except two ontology steps | Fast, original step selectors | All | All |
| Backend claims, package support, receipt/provenance, canonical lean_single fixed point, boot4 fixed point, AArch64 refusal parity | Remain in Contracts | Yes | Yes |
| Ontology frontiers + multi-ontology composition | Deep, **every PR**, unchanged commands/default engine | Yes | Yes |
| Source-built canonical Madaros + Wave 0 smoke/negative controls | Build on every PR (Correlated requires it); Wave 0 on compiler/runtime/stdlib/tests/full | Yes | Yes |
| Current-source f64, self-parse, IR capacity, DCE, changed Madaros tests, e-graph, warnings | Deep for compiler/stdlib/tests/full | Yes | Yes |
| Correlated effect + slot-identity sabotage | Deep, **every PR**, unchanged oracles | Yes | Yes |
| Full Witness Gate (including KL and sabotage/refusal controls) | Deep for compiler/runtime/stdlib/tests/full | Yes | Yes |
| Lint / website | Fast with existing impact selectors | Yes | Yes |
| Lean proofs and existing adjacent checks | Deep on Lean/full changes | Yes | Yes |
| Native Linux complete self-host, source-bootstrap, macOS execution/cross-target witnesses | Deferred | Exhaustive | Exhaustive |
| Full Test Suite + qd128 accuracy replay | Deferred | Exhaustive | Exhaustive |
| Madaros gen2 fixed-point rung (minimum 122 unchanged) | Deferred | Exhaustive | Exhaustive |
| Chemistry golden probes | Existing chemistry path-filtered PR workflow retained | Exhaustive (reusable workflow) | Exhaustive |
| R6 corpus sweep | Existing nightly-only policy | Dispatch with `nightly=true` | Yes |

Unknown paths and CI changes classify `full=true` and run **exhaustive even on a
PR**. This rollout PR therefore deliberately takes the full qualification path.
Known compiler/seed changes receive Fast + Deep and the existing small canonical
seed fixed-point checks; the costly self-host ladders/gen2 run on merge-group and
on every main push. A docs-only PR still retains all existing unconditional
Contracts and ontology contracts. This conservative first cut avoids guessing
which scientific contracts can safely be path-filtered.

Distribution at this base is covered by the existing package-support and
prebuilt-receipt contracts, now also explicitly included in exhaustive runs.
The pinned external-distribution implementation in PR #2553 was **not merged at
inspection**. Its future focused packaging workflow must be integrated on landing;
this change does not claim to validate code absent from this base or publish a
release. Chemistry deliberately retains its committed lean_single engine and
byte-for-byte goldens; replacing it with Madaros would change the scientific oracle.

## Artifact custody

One `canonical-madaros` producer per selected CI run/commit derives the current
lean_single seed and builds modular Madaros. Current-Source, Witness, Wave 0, Correlated effect and
the exhaustive fixed-point job consume that ELF. The former standalone
`correlated-effect.yml` job now lives in this graph, retaining every PR/main/merge
trigger and gaining nightly coverage. Its manual entry is the CI dispatch. No global mutable compiler
cache, cross-run artifact lookup, or prebuilt fallback is introduced.

The artifact name includes the event SHA and producer attempt. All consumers
check the separate producer job's SHA-256, manifest SHA-256, checkout SHA/tree,
repository, run ID, bootstrap hash, build-script hash, and ELF architecture.
The checkout is the normal PR **merge commit**, not mislabeled as the PR head.
A partial retry may reuse its successful producer from the same run; rerunning
all jobs creates a new artifact name. Missing files, bad metadata, dirty tracked
inputs, and digest/architecture mismatches fail before compiler execution.
The provenance receipt is an integrity/custody record, not a cryptographic
attestation or an independent reproducibility proof. Independent seed/fixed-point
builds remain independent because those compilations are the oracle itself.

GitHub's automatic artifact digest mismatch is a warning, so the consumer adds
an explicit fatal check ([GitHub artifact documentation](https://docs.github.com/en/actions/tutorials/store-and-share-data)).
The workflow has read-only repository permissions and no privileged PR trigger.

## Remaining independent builds

`engine-parity-nightly.yml` still builds Madaros in its own nightly/dispatch run
and compares engines against its existing baseline. `release-gate.yml` still
constructs native seed/self-host ladders for its independent release receipt.
`madaros-prebuilt-refresh.yml` retains its publishing/provenance path. None blocks
ordinary compiler PR feedback with another gen2 rung. Cross-workflow reuse is
intentionally outside this first change: sharing a PR artifact with privileged
release/publishing workflows would require a separately reviewed trust boundary.
This is one canonical build per CI run/commit, not a repository-wide cache or a
claim that all rebuilds across independent workflows have disappeared.

## Verdicts, rollout and rollback

`Fast PR Gate` and `Deep Compiler Gate` are separate early receipts. `CI Decision`
still waits for every selected job, including Chemistry and fixed-point when
exhaustive, and both early verdicts. It rejects missing, failed, cancelled, and
unexpectedly skipped selected jobs; malformed/missing impact classification;
and disagreement between event and exhaustive selection. A failed unselected
job is also fatal. Workflow conditions and evaluator selection are tested across
PR/compiler/runtime/stdlib/tests/Lean/full and all supported event combinations.

Concurrency groups include the event. A main push cannot cancel a nightly run;
only superseded PR runs are cancelled. GitHub may still replace older *pending*
runs within one concurrency group; this is not a guarantee that every historical
push gets exhaustive execution.

On inspection, the main branch-protection endpoint returned **Branch not
protected**; the only listed ruleset targeted `canon/madaros-v2-sota`. No settings
are changed here. To enforce exhaustive qualification **before** integration,
maintainers must require `CI Decision` and use the merge queue (the workflow
already handles `merge_group`). Direct merges only get exhaustive evidence
**after** integration via main push. Fast success alone is never full qualification.

Acceptance: local unit/negative controls, impact self-test, actionlint, script
reference and gate-reference ratchets; then inspect exact-head Actions results.
Measure Fast/Deep completion and queue delay over multiple representative PRs
before claiming the target achieved. Do not waive pre-existing Full Suite failures.
Rollback the workflow/evaluator switch together; the standalone artifact helper
commit can remain unused without changing existing compiler resolution.
