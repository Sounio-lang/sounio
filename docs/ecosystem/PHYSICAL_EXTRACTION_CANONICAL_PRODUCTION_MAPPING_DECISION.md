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

At `2026-07-18T02:42:17Z`, the issue remained open with no response comments.
Therefore no Sounio production mapping decision record, receipt, or proposal
has been authored. The observed public `sounio-examples` repository remains
only a candidate that a human may explicitly select or reject; this tool does
not infer that choice.

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
The 204-assertion focused result is local evidence until a new immutable Slurm
witness is recorded.

## Remaining Boundary

The executable processor closes the transcription-to-proposal preparation gap,
not the human decision itself. A response on issue #1122 is still required.
Depending on that response, the next state is ownership-policy review,
repository provisioning plus reobservation, or review of a non-approved mapping
proposal. Production materialization evidence, recovery policy, approval,
explicit cutover decision, and execution remain later and separate interfaces.
