<!-- docs:meta
topic_id: repo.docs.ecosystem.physical-extraction-canonical-production-reconciliation-proposal
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ecosystem.physical-extraction-canonical-production-reconciliation-proposal
-->

# Physical Extraction Canonical Production Reconciliation Proposal

Status: executable R3 read-only proposal contract; reconciliation remains
`proposed-not-approved`, destination write authority remains `none`, and
canonical production cutover remains `not-executed`.

`tools/science_boundary/canonical_production_reconciliation_proposal.py`
turns one evidence-bound parity gap into a complete path-level proposal. It
reobserves the exact source and destination Git snapshots, reinventories every
tracked byte, verifies the prior evidence row, and describes the conditional
post-state needed for an exact source mirror.

The contract fixes:

```text
authority_scope = reconciliation-description-and-review-only
proposal_status = proposed-not-approved
execution_authority = none
destination_write_authority = none
source_removal_authority = none
canonical_production_approval = not-approved
canonical_cutover_execution_status = not-executed
```

The tool has no execution subcommand. It cannot copy, replace, delete, commit,
push, approve, or cut over a repository.

## Inputs

The builder requires:

1. A regular evidence-set JSON with a valid canonical identity, fixed
   non-authorizing state, and exactly one selected `parity-gap-observed` row.
2. The clean canonical source worktree at the evidence-bound branch, head,
   remote, and URL.
3. The clean mapped destination worktree at the evidence-bound branch, head,
   remote, and URL.
4. The selected target ID.
5. The exact expected evidence identity as an explicit review pin.

The source subtree and destination root are scanned independently. Their
regular-file inventories must equal their complete Git tracked-file sets.
Symlinks, untracked or ignored content, dirty worktrees, detached or stale
heads, wrong remotes, byte drift, forged evidence, and occupied output paths
refuse.

## Path Plan

The output contains one sorted row for every relative path in the union of the
source subtree and destination tree. Each row carries source state,
destination-before state, a conditional destination-after state, and one
disposition:

| Disposition | Meaning |
|---|---|
| `add-source-byte-copy` | path exists only in source |
| `replace-with-source-byte-copy` | shared path differs by size or SHA-256 |
| `remove-destination-only` | path exists only in destination |
| `retain-identical` | path, size, and SHA-256 already match |

Every row fixes `operation_authority = none`. For a destination-only path the
conditional after-state is `null`, but that is a reviewed exact-mirror
description, not deletion authority. Review may instead reject removal or
request a preservation strategy; either choice requires a new proposal.

The proposal binds both a SHA-256 over the complete canonical path-plan array
and a canonical proposal identity. It never truncates the plan to samples.

## Build

```bash
python3 tools/science_boundary/canonical_production_reconciliation_proposal.py build \
  --evidence /reviewed/production-evidence-set.v1.json \
  --source-root /exact/evidence-bound/source \
  --destination-root /exact/evidence-bound/sounio-examples \
  --target-id distribution:sounio-research-examples \
  --expected-evidence-identity 7d62a39d1dec79aa76780608da6e93182b53703daf5f85fa663cab782429f971 \
  --output /unoccupied/production-reconciliation-proposal.v1.json
```

Only the requested JSON output is created. Source and destination paths are
never written.

## Verify

```bash
python3 tools/science_boundary/canonical_production_reconciliation_proposal.py verify \
  --proposal /reviewed/production-reconciliation-proposal.v1.json \
  --evidence /reviewed/production-evidence-set.v1.json \
  --source-root /exact/evidence-bound/source \
  --destination-root /exact/evidence-bound/sounio-examples \
  --target-id distribution:sounio-research-examples \
  --expected-evidence-identity 7d62a39d1dec79aa76780608da6e93182b53703daf5f85fa663cab782429f971
```

Verification reconstructs every Git binding, file identity, disposition,
count, hash, limitation, precondition, and identity. Rehashing a forged
evidence set or proposal does not make it valid.

## Current Proposal

The proposal under
`artifacts/r3/canonical-production/20260719T213906Z/` is bound to evidence
identity
`7d62a39d1dec79aa76780608da6e93182b53703daf5f85fa663cab782429f971`,
source head `e19af3279a040a6a707967d786be657bdf0d4203`, and destination
head `a22f66e0060ba6d007b8b69012ecadee7e9345bd`.

Its proposal identity is
`ef9b3401af36bebf57fc960eb9108c6d313a15df5d9ada9324dd6138d8ad43f0`;
the complete path-plan SHA-256 is
`86f7ebcb327d17760df4a761e44e99fcf417705696f9e10e6dcd3e08cacd903b`.

| Class | Count |
|---|---:|
| proposed additions | 1,029 |
| proposed replacements | 3 |
| proposed destination-only removals | 7 |
| identical retained paths | 2 |
| total union paths | 1,041 |
| mutation paths | 1,039 |

The three replacement paths are `README.md`, `hello.sio`, and
`uncertainty.sio`. The seven destination-only paths are recorded explicitly in
the durable README and the complete JSON. No one of those changes is
authorized.

The evidence-bound source snapshot is historical. At proposal time the remote
`main` branch was separately observed at
`8fd67e4d70893e1b94bb4c89bb0e03d16526f90e`, not the bound `e19af3279...`
head. The proposal is useful for exact review but is not execution-consumable;
a current catalog, mapping, evidence set, and path plan must be reissued after
any drift.

## Acceptance Gate

The focused gate is
`scripts/ci/physical_extraction_canonical_production_reconciliation_proposal_gate.py`.
It uses only temporary standalone repositories and local bare remotes. It
covers deterministic equivalent roots; add, replace, remove, and retain rows;
complete plan hashing; fixed authority fields; output no-clobber; forged and
rehashed evidence; forged and rehashed proposal; dirty destination; missing
target; overlapping roots; source/destination immutability; and staging-file
cleanup.

The package-support gate runs this focused gate after the production-evidence
gate.

## Remaining Boundary

The next permitted action under the present authorization is review of this
proposal and, because the source head has advanced, preparation of a fresh
point-in-time evidence chain. Destination mutation, source removal, production
approval, an execution/recovery policy, and cutover remain separate
permission-bearing decisions and are absent here.
