<!-- docs:meta
topic_id: repo.docs.ecosystem.physical-extraction-canonical-production-mapping-decision
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ecosystem.physical-extraction-canonical-production-mapping-decision
-->

# Physical Extraction Canonical Production Mapping Decision

Status: executable R3 non-authorizing mapping-selection processing; all five
targets were explicitly reconfirmed and emitted as a reviewed
`proposed-not-approved` mapping, while production approval and cutover execution
remain absent.

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
records the same processing result. At that processing checkpoint, no
repository had been created or modified.

On `2026-07-19`, separate interactive authorization was used only to create the
four requested repositories. `epistemic-core`, `sounio-formats`,
`sounio-io-primitives`, and `sounio-units` are public and empty, report
configured default branch `main`, and have no Git refs. Creation used
`auto_init=false`; no package content or other source was materialized. The
post-creation organization catalog was observed at `2026-07-19T00:39:21Z`,
contains 14 repositories, and has identity
`46ef6e4ecde6063e3a1c744a499bc3cdca905a7334405d955cd120171142f0c6`.
The provisioning receipt identity is
`93c70857c13f0e5572a7870689d3c57ec5cd004d48445e6af89007f206254569`.
Issue comment
[`5013552386`](https://github.com/Sounio-lang/sounio/issues/1122#issuecomment-5013552386)
records the same result, and the observation set is preserved under
`artifacts/r3/canonical-production/20260719T003921Z/`.
The requested `sounio-scientific-packages-maintainers` team slug was not
observed, so no team or team permission was created.

Later on `2026-07-19`, a separate interactive instruction authorized only the
exact materialization of the four package roots, initial `main` commits, pushes,
and verification. The bound source is remote review-branch commit
`317e6d085ad5304c8fac185eee03552a6b916123`; correction PR
[#1176](https://github.com/Sounio-lang/sounio/pull/1176) remained draft and
unmerged. Independent math review had caught and prompted repair of a wrong GUM
example and zero-central-value multiplication uncertainty before the copy.

The materialized commits are:

- `epistemic-core`: `732b3fbf7ff1d596cf591124b475791fe5e1add9`;
- `sounio-formats`: `c412c0d1e7ef276d3ad9d1e662d681369e3e384c`;
- `sounio-io-primitives`: `8e593615072e7ad9962ab27c0e316a8be521457d`;
- `sounio-units`: `229d310f676d2a3a1e183983764da2ddd63f6fe0`.

Source inventory identity
`f03458beb2ed07380e7d4a7b1242bb7b32e3c609a47f204e44c34eca64a429e5`
binds all package files and tree hashes. Four fresh clones reproduced those
hashes and commits after push. The post-materialization organization catalog
was observed at `2026-07-19T02:48:12Z`, contains 14 repositories, and has
identity
`095de409e315ff0c716c4877274c8b2d439310bd255233cf1558f42f2b19be2c`.
Issue comment
[`5013887205`](https://github.com/Sounio-lang/sounio/issues/1122#issuecomment-5013887205)
records the result byte-identically, and the evidence is preserved under
`artifacts/r3/canonical-production/20260719T024812Z/`.

This operation did not materialize `sounio-examples`, remove source paths,
repair manifest repository fields, create a team or branch rule, emit a mapping
proposal, or approve production or cutover. At that checkpoint, while PR #1176
was unmerged, the public copies were deliberately bound to its remote source
commit rather than to `origin/main`; any substantive PR rewrite required a new
binding before canonical use.

At `2026-07-19T03:50:45Z`, `agourakis82` explicitly reconfirmed all five
destinations in issue comment
[`5014112002`](https://github.com/Sounio-lang/sounio/issues/1122#issuecomment-5014112002).
The response is 1896 bytes with SHA-256
`22207dcecb8ba8ec7377e9957a36cbcb91fc7fff8110ce3266c221cbe177fea3`.
It authorizes only processing and review of a `proposed-not-approved` mapping
and explicitly withholds source removal, production approval, and cutover.

The selection binds catalog observation `2026-07-19T03:39:58Z`, catalog
identity
`cef66e6c59e9b7f4b35a5d4dd0637bfd71865a93b049a8af6dd4471ade8ad55a`,
and clean canonical `sounio/main` snapshot
`88530f217bab58cac6a9a7c31160f75415b77d68`. The four package destination
commits and the `sounio-examples` commit were unchanged from the preceding
materialization catalog.

The deterministic decision identity is
`63be89d31b54dd21617c27abfdcde0b598d65c74b60b40af965e89da9a736bed`.
Process and verify modes independently reconstructed receipt identity
`d67863e77e2b432221b8c741807a102301c39afc5e91859b34b398e0432a5f87`
with status `proposal-input-complete`, five `reuse-observed` rows, no
`request-new` or `revise-target` rows, and proposal identity
`a32de28e879ea03370f90382f0d67a3651a53b4108d8c45ed0403b1106921f2d`.
Every emitted mapping remains `proposed-not-approved`; execution authority is
`none` and canonical cutover is `not-executed`.

Contract-bound review by xAI/Grok 4.3 and Z.AI/GLM-5.2 found no BLOCKER or
MAJOR inconsistency in the proposal. The downstream production-gap assessor
accepted and independently verified it as assessment identity
`c050015eac9fa7cf794f1ff989cfb114e801ca575d55e28f811b6488a7a28a1d`,
retaining status `production-evidence-and-human-decision-required`. Issue
comment
[`5014152829`](https://github.com/Sounio-lang/sounio/issues/1122#issuecomment-5014152829)
records the result byte-identically. The complete evidence is preserved under
`artifacts/r3/canonical-production/20260719T033958Z/`.

PR #1176 subsequently merged into canonical `main` as
`d380146ffeabd3d18e71182ac4c03132f0788cf2` at
`2026-07-19T13:05:58Z`. Exact comparison of fresh destination clones against
that merge found `sounio-formats`, `sounio-io-primitives`, and `sounio-units`
already byte-identical. Only `epistemic-core/src/lib.sio` differed, in two
comment lines that narrow unsupported claim wording. An ordinary fast-forward
updated `epistemic-core/main` from `732b3fbf7ff1d596cf591124b475791fe5e1add9`
to `3e7d49fb84c7b8c74b8fd4b1cc39660772d9c7d1`; no other destination or ref
changed.

Source inventory identity
`022772f58a51cc4273d5f02043690398085e93b0b767c7e16b91e06d131a7014`
and reconciliation receipt identity
`1ceceb68ea56593770e94f94dfbf4433bdad827d3e90c02220c677603a1b300e`
bind that operation. A fresh post-push clone reproduced the exact six-file
source tree with SHA-256
`5dcea277263dbb656b9c8cfa32ab8f8f148109e8f3a82cb76e33cd6fdd6fa114`.
Issue comment
[`5015935195`](https://github.com/Sounio-lang/sounio/issues/1122#issuecomment-5015935195)
records the result byte-identically, and evidence is preserved under
`artifacts/r3/canonical-production/20260719T133147Z/`.

This reconciliation clears the PR #1176 source-binding dependency for the
public package copies, but it does not rewrite the earlier proposal. The
proposal remains bound to catalog identity
`cef66e6c59e9b7f4b35a5d4dd0637bfd71865a93b049a8af6dd4471ade8ad55a`
and its observed `epistemic-core` head `732b3fbf...`. The later operational
catalog and destination head are evidence of point-in-time drift, not an
implicit amendment, reconfirmation, or approval.

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

The complete post-materialization selection and proposal review are recorded,
but the proposal is not an approval. The maintainer team and branch-rule
evidence remain absent. Production materialization evidence supplied to the gap
assessor, source-removal authorization, canonical-production approval, an
execution policy, and an explicit human cutover decision remain missing and
separate interfaces. PR #1176 is merged and its public package source binding
has been reconciled, but that operation advanced `epistemic-core/main` beyond
the proposal's catalog-bound head. No later head is silently substituted.
Catalog or governed-source drift requires a new selection record before
downstream use.
