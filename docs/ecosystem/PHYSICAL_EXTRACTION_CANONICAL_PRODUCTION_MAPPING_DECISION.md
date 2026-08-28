<!-- docs:meta
topic_id: repo.docs.ecosystem.physical-extraction-canonical-production-mapping-decision
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ecosystem.physical-extraction-canonical-production-mapping-decision
-->

# Physical Extraction Canonical Production Mapping Decision

Status: executable R3 non-authorizing mapping-selection processing; repository creation, production approval, and cutover execution remain absent.

`tools/science_boundary/canonical_production_mapping_decision_processor.py`
turns one reviewed transcription of an explicit mapping selection into a
deterministic processing receipt. It emits a target mapping proposal only when
every governed target selects an exact, available repository from the bound
point-in-time catalog.

The contract fixes:

```text
authority_scope = mapping-proposal-preparation-only
execution_authority = none
canonical_cutover_execution_status = not-executed
proposal_status = proposed-not-approved
```

It has no hosting API client and performs no repository creation, source
materialization, source removal, or Git ref update.

## Decision Evidence

The input is a JSON transcription governed by
`sounio.physical-extraction-canonical-production-mapping-decision.v1`. Its
source evidence records:

- the decision-request issue URL;
- the exact response URL;
- a responder label;
- the SHA-256 of the raw response body;
- the response timestamp; and
- `evidence_status = transcribed-not-authenticated`.

These fields preserve provenance but do not authenticate a person or prove
organizational authority. The transcription must be independently reviewed
against the linked response before use. A generic continuation instruction is
not a mapping selection and must not be transcribed as one.

The record is bound to the exact catalog identity and observation timestamp,
canonical repository ID, local branch, and local HEAD. The processor also
rebuilds the current science-ring and ownership inventory, so every
`extract-planned` target path, ID, and owner must match the governed files.

The only authorized operation is:

```text
draft-proposed-not-approved-mapping
```

The record explicitly prohibits repository creation or modification, source
materialization or removal, Git ref changes, canonical-production approval, and
cutover approval or execution.

## Target Actions

Every planned target must have exactly one action. Rows are strictly sorted by
`target_id`, and non-null repository IDs and URLs must be unique across rows.

| Action | Required repository fields | Result |
|---|---|---|
| `reuse-observed` | catalog-exact ID, `.git` URL, and default branch; visibility and rationale are null | reusable only when the catalog row is non-archived, non-empty, has a valid head, and reports `WRITE`, `MAINTAIN`, or `ADMIN` |
| `request-new` | desired absent ID, `.git` URL, branch, and visibility; rationale is null | no proposal; repository provisioning, new catalog observation, and new human selection are required |
| `revise-target` | repository fields are null; rationale is required | no proposal; the governed ownership target must be revised before another selection |

`request-new` is a request classification, not permission for the processor to
create a repository. If its desired ID or URL already occurs in the bound
catalog, processing refuses instead of silently changing the action to reuse.

`reuse-observed` never infers a repository from a similar name. The input ID,
URL, and branch must match one catalog row exactly. Visibility and expected HEAD
are copied from that row into the receipt; the expected HEAD is also copied into
the proposal rather than supplied by the response.

## Processing Results

The receipt uses one of three ordered states:

| State | Condition | Next action |
|---|---|---|
| `ownership-policy-review-required` | one or more `revise-target` rows | revise the governed target and repeat the selection |
| `destination-repository-creation-required` | no revision, but one or more `request-new` rows | provision outside this tool, reobserve the catalog, and obtain a reconfirmed selection |
| `proposal-input-complete` | every row is catalog-exact `reuse-observed` | review the emitted `proposed-not-approved` mapping |

Revision takes precedence over creation because an ownership-policy change can
alter the target set. A proposal is emitted only for
`proposal-input-complete`. Even then, the existing production-gap assessor
still reports `production-evidence-and-human-decision-required`: a mapping
selection is not the later explicit decision to execute a cutover.

The receipt is the completion marker when a proposal is emitted. Both outputs
are staged and occupied paths are refused without overwrite. The contract does
not claim crash-atomic promotion across two files; verification requires the
receipt and its exact proposal identity together.

## Commands

Process a reviewed selection:

```bash
python3 tools/science_boundary/canonical_production_mapping_decision_processor.py process \
  --repo-root /path/to/clean/canonical-snapshot \
  --repository-catalog /path/to/repository-catalog.json \
  --mapping-decision /path/to/reviewed-mapping-decision.json \
  --canonical-repository-id sounio \
  --remote-name origin \
  --receipt-output /path/to/mapping-decision-receipt.json \
  --proposal-output /path/to/mapping-proposal.json
```

Omit `--proposal-output` for a selection containing `request-new` or
`revise-target`. Supplying it in those states refuses. Conversely, an all-reuse
selection refuses without a proposal output path.

Verify a completed all-reuse result:

```bash
python3 tools/science_boundary/canonical_production_mapping_decision_processor.py verify \
  --repo-root /path/to/clean/canonical-snapshot \
  --repository-catalog /path/to/repository-catalog.json \
  --mapping-decision /path/to/reviewed-mapping-decision.json \
  --canonical-repository-id sounio \
  --remote-name origin \
  --receipt /path/to/mapping-decision-receipt.json \
  --mapping-proposal /path/to/mapping-proposal.json
```

Verification reconstructs the inventory, catalog validation, local Git
observation, action results, proposal, receipt, and both identities. Catalog,
branch, HEAD, policy, source bytes, response evidence, or output drift refuses.
Rehashing a modified artifact does not make it match the inputs.

## Current Sounio State

Issue [#1122](https://github.com/Sounio-lang/sounio/issues/1122) asks for one
explicit action for each of these governed targets:

| Source | Target |
|---|---|
| `packages/epistemic-core` | `distribution:epistemic-core` |
| `packages/sounio-formats` | `distribution:sounio-formats` |
| `packages/sounio-io-primitives` | `distribution:sounio-io-primitives` |
| `packages/sounio-units` | `distribution:sounio-units` |
| `examples` | `distribution:sounio-research-examples` |

At `2026-07-18T16:57:25Z`, `agourakis82` posted a complete response on issue
[#1122](https://github.com/Sounio-lang/sounio/issues/1122#issuecomment-5012124187).
It requests four new public repositories (`epistemic-core`, `sounio-formats`,
`sounio-io-primitives`, and `sounio-units`) and explicitly selects the observed
public `sounio-examples` repository for `distribution:sounio-research-examples`.

The source response is 2647 bytes with SHA-256
`f2e01687686dfa09df5acf173e67aa9c9d73fe22988302a72926e5d16c39408b`.
The deterministic transcription has identity
`246cd09179f1b0a49aebb2d87d65d33f48dbdd68d17df76d245d34bca7a034de`.
Process and verify modes independently reconstructed receipt identity
`f74837d7d0ae83c6ba3d8d13a317c6024d0aab5c83bfda1069aa4b97f42567b3`
with status `destination-repository-creation-required`, proposal output
`not-emitted`, and execution authority `none`.

The catalog, source response, decision, receipt, and their exact bindings are
preserved under
`artifacts/r3/canonical-production/20260718T165725Z/`. Issue comment
[`5012139001`](https://github.com/Sounio-lang/sounio/issues/1122#issuecomment-5012139001)
records the same processing result. No repository was created or modified.

## Acceptance Gate

The focused gate is
`scripts/ci/physical_extraction_canonical_production_mapping_decision_gate.py`.
It uses temporary standalone repositories and local bare remotes. Equivalent
physical roots emit byte-identical decisions, receipts, and proposals.

The gate passes 204 assertions across all three processing states. It covers
exact target coverage, action-dependent fields, catalog and local-HEAD drift,
unavailable reuse rows, request collisions, unique destinations, response
evidence, deterministic identities, occupied outputs, source preservation, and
forged or rehashed decision, receipt, and proposal refusal. Its all-reuse
proposal is passed back through the production-gap assessor, which keeps
execution authority `none` and the explicit cutover decision missing.

The composed shell gate runs the complete production-gap stack first and
forwards one current-source Madaros input through every compiler-bound gate.

The composed current-source witness is Slurm job `6635` on
`gpuorangefs-r770-proxmox`. Commit
`aa0e50c6af32f55819d16191735344da5bd1c840`, compressed source archive
`ad22c9eca1dd6458a97f55f9063e6f346f70b2cf00470e910ac1a0261a925868`
(339734094 bytes), current-source Madaros
`6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88`,
and Git 2.43.0 passed 178 R0-R2, 65 R2.5, 82 R2.6, 141 inventory, 167
materialization, 527 authorization, 164 local execution, 172 cutover approval,
81 cutover execution, 90 production-gap, and 204 mapping-decision checks. The
job completed with exit `0:0` in 60 seconds. Slurm accounting was unavailable,
so no `MaxRSS` is claimed.

Stdout is 4109 bytes with SHA-256
`8eedbc7c041abf4c2087fab4843eb1b637d366dc1aee4bdb36f5e33dc1ab4f73`.
Stderr contains only the two `srun` allocation messages, is 92 bytes, and has
SHA-256
`37b49d592edaf7aecf7611b86b0d178381e60d6e6434e071fa06ebc5ebe44e5e`.
The streamed input payload is 438497280 bytes with SHA-256
`877fff07323908461ea2eb813c68ef4ce73861c1e95c501dea347b2381091fc7`.

Three batch attempts did not reach the gate: job `6629` inherited a missing
`/tmp` workdir, while jobs `6631` and `6633` were stopped before shell startup
by the cluster's batch environment retrieval failure. The successful witness
used a synchronous Slurm `srun`, streamed the exact archive and compiler into a
node-local root, and used node-local extraction, fixture repositories, remotes,
home, and temporary files. This was a harness-routing fallback only; no
implementation fallback or real hosting operation ran.

## Remaining Boundary

The executable processor has recorded the human selection without emitting a
mapping proposal. The next state is separately authorized provisioning of the
four requested repositories, followed by a fresh organization catalog and
canonical `main` observation and a complete reconfirmation of all five targets.
Production materialization evidence, recovery policy, approval, explicit
cutover decision, and execution remain later and separate interfaces.
