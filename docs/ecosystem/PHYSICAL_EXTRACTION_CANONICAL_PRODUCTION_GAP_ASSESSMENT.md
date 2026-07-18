<!-- docs:meta
topic_id: repo.docs.ecosystem.physical-extraction-canonical-production-gap-assessment
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ecosystem.physical-extraction-canonical-production-gap-assessment
-->

# Physical Extraction Canonical Production Gap Assessment

Status: executable R3 non-authorizing prerequisite observation; production policy, approval, human decision, and execution remain absent.

`tools/science_boundary/canonical_production_gap_assessor.py` reconstructs the
current R3 ownership and file inventory, observes one local Git worktree against
a supplied point-in-time repository catalog, and optionally verifies one exact
target-to-repository mapping proposal. It emits a deterministic gap assessment.

The assessment contract fixes:

```text
authority_scope = prerequisite-observation-only
execution_authority = none
canonical_cutover_execution_status = not-executed
```

No accepted assessment status is `ready`, `approved`, or `authorized`. A
complete mapping and available repository fixture reaches only
`production-evidence-and-human-decision-required`.

## Inputs

The assessor consumes:

1. The canonical worktree, `science-rings.tsv`, and the complete physical
   extraction ownership policy. It rebuilds the inventory rather than trusting
   supplied target counts or tree identities.
2. A closed repository catalog with an observation timestamp, exact default
   branch and head object, visibility, archived/empty flags, observed access
   label, and deterministic catalog identity.
3. An optional mapping proposal that covers every `extract-planned` target
   exactly once and binds the supplied catalog identity, target owner,
   repository ID, URL, branch, and expected head.

The catalog is supplied evidence, not a live hosting attestation. An observed
access label does not prove human identity, organizational authority, branch
protection state, or that a future push will succeed. The mapping proposal is
fixed as `proposed-not-approved` and is neither destination-owner approval nor
cutover permission.

Inventory reconstruction and Git/catalog observation are sequential, not one
atomic snapshot. The assessment is a deterministic report over the observed
inputs, not a lock or transaction over the canonical repository or hosting
service.

## Assess

```bash
python3 tools/science_boundary/canonical_production_gap_assessor.py assess \
  --repo-root /exact/source-worktree \
  --rings science-rings.tsv \
  --ownership docs/ecosystem/science-physical-extraction-ownership.tsv \
  --repository-catalog /observed/repository-catalog.json \
  --canonical-repository-id sounio \
  --remote-name origin \
  --output /external/canonical-production-gap-assessment.json
```

Add `--mapping-proposal /reviewed/mapping-proposal.json` only after a separate
proposal has been authored. There is no default proposal, target repository,
or inferred name match.

The output path must be unoccupied. Assessment does not write inside the
canonical worktree, create repositories, copy or remove sources, or update any
Git ref.

## Status Ordering

The deterministic status identifies the first unresolved infrastructure
boundary:

1. `mapping-proposal-required`: no exact target-to-repository proposal was
   supplied.
2. `destination-repositories-required`: the proposal exists, but one or more
   mapped repositories are absent, empty, archived, or differ in URL, branch,
   or head from the catalog.
3. `canonical-source-snapshot-required`: mappings and destinations are
   observed, but the local worktree is dirty or differs from the cataloged
   canonical default branch, head, or remote URL.
4. `production-evidence-and-human-decision-required`: the observable
   repository prerequisites match, but materialization, removal authorization,
   canonical approval, execution policy, and explicit human decision remain
   separate and absent from this assessment.

The latter five permission-bearing prerequisites are always reported missing
by this v1 assessor. They must be authored and validated through their own
interfaces; they cannot be inferred from repository availability.

## Verify

```bash
python3 tools/science_boundary/canonical_production_gap_assessor.py verify \
  --assessment /external/canonical-production-gap-assessment.json \
  --repo-root /exact/source-worktree \
  --rings science-rings.tsv \
  --ownership docs/ecosystem/science-physical-extraction-ownership.tsv \
  --repository-catalog /observed/repository-catalog.json \
  --canonical-repository-id sounio \
  --remote-name origin
```

Verification rebuilds the inventory, Git observation, targets, prerequisites,
summary, and identity. Changed source bytes, Git state, catalog, proposal, or
assessment refuse. Rehashing a forged assessment does not make it valid.

## Current Sounio Observation

At `2026-07-17T21:25:23Z`, the GitHub organization catalog contained ten
repositories. None was explicitly mapped by the ownership policy to the five
planned targets:

| Source | Planned target |
|---|---|
| `packages/epistemic-core` | `distribution:epistemic-core` |
| `packages/sounio-formats` | `distribution:sounio-formats` |
| `packages/sounio-io-primitives` | `distribution:sounio-io-primitives` |
| `packages/sounio-units` | `distribution:sounio-units` |
| `examples` | `distribution:sounio-research-examples` |

The existing `Sounio-lang/sounio-examples` repository is not silently treated
as `distribution:sounio-research-examples`; accepting that reuse would require
an explicit mapping proposal and review. No repository was created or modified
by the observation.

The cataloged `main` head was
`aff3d4010b462af0d4e79ebc141eb6c39c4eaa50`. The assessed clean stacked source
head was `0a88da8cf1c165940cc9aa07f6832992b1206a22` on
`codex/physical-extraction-canonical-cutover-execution-r3-20260717`, so the
canonical default-branch prerequisite also remained absent. A production
snapshot must be reobserved after the reviewed stack lands.

The current assessment reports:

```text
assessment_identity = 0fe82728ea24520af7792d4b5cf45c6c20e62c47a09138d0c4b81207e998e816
catalog_identity = 6dae5a00fb0cf176bed2b7e1e9420cede8591a1175a3a58b5d3a555a9844460e
readiness_status = mapping-proposal-required
planned_target_count = 5
mapped_target_count = 0
observed_available_destination_count = 0
missing_prerequisite_count = 8
execution_authority = none
canonical_cutover_execution_status = not-executed
```

The catalog file SHA-256 is
`ea3285fb3f788f547de5cf4de55930a399f034f21e11763e25e7b46a2460b8c7`;
the assessment file SHA-256 is
`5d61566bff517177d2088b6e327f4d67dbbc14cdc1aaa02d0369ab24762fbcb3`.
No mapping proposal or human decision record was authored. Issue
[#1122](https://github.com/Sounio-lang/sounio/issues/1122) now requests the
five explicit target actions. The non-authorizing processor documented in
`PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_MAPPING_DECISION.md` can validate a
reviewed transcription after a response exists; it does not supply that
response or infer a repository by name.

## Acceptance Gate

The focused gate is
`scripts/ci/physical_extraction_canonical_production_gap_gate.py`. It uses only
temporary standalone Git repositories and local bare remotes. Two equivalent
physical roots emit byte-identical absent-proposal and complete-proposal
assessments. The positive proposal fixture stops at
`production-evidence-and-human-decision-required` with authority `none`.

The gate currently passes 90 assertions covering catalog/proposal identity,
exact target coverage, deterministic roots, clean/default Git comparison,
missing, archived and changed destinations, dirty source state, occupied-output
preservation, source and catalog mutation, and forged or rehashed assessment
refusal. It explicitly refuses `approved` proposals and `authorized` mappings.

The composed shell gate runs the complete R0-R3 cutover execution stack first.

The composed current-source witness is Slurm job `6615` on
`gpuorangefs-r770-proxmox`. Commit
`4dc8749a7be001ab0f9d80e5723fc292078c1527`, compressed source archive
`484fde3c4881905d60fbf601d8d42a7ba5d389cb6392b9657022c2267a791ede`
(339700729 bytes), current-source Madaros
`6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88`,
and Git 2.43.0 passed 178 R0-R2, 65 R2.5, 82 R2.6, 141 inventory, 167
materialization, 527 authorization, 164 local execution, 172 cutover approval,
81 cutover execution, and 90 production-gap checks in 55 seconds with
`MaxRSS=1390896K`. Stdout is 3580 bytes with SHA-256
`94b7da4b7222ed882248dcdf5557348de1a025dc248b623de732ac521a1624ee`;
stderr is empty with SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The immutable archive, compiler, extraction, transaction, fixture repositories,
bare remotes, and logs used node-local `/tmp`. The new compiler environment
variable was forwarded into the complete prior stack, so every compiler-bound
gate used the same Madaros witness. No implementation fallback ran, no real
repository catalog was consumed by the Slurm fixture gate, and no canonical
source or remote ref changed.

## Remaining Decision Boundary

This assessment closes no permission-bearing prerequisite. The deterministic
mapping-selection processor closes only the future transcription-to-proposal
preparation step; issue #1122 still has no response. The next semantic interface
remains `r3-physical-extraction-canonical-production-policy-and-human-decision`.
It requires a reviewed mapping choice, provisioned and materialized destination
repositories, a new clean canonical default-branch snapshot, production
recovery and approval evidence, and an explicit human decision. A generic
continuation command is not that decision.
