<!-- docs:meta
topic_id: repo.docs.ecosystem.physical-extraction-materialization
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ecosystem.physical-extraction-materialization
-->

# Physical Extraction Materialization

Status: executable R3 local exact-copy boundary; canonical repository extraction is not executed.

## Scope

The materialization interface consumes one fully verified
`sounio.physical-extraction-inventory.v1` and one separately authored
`sounio.physical-extraction-destination-policy.v1`. It copies every
`extract-planned` unit into a preexisting approved local destination, verifies
the copied byte identities, and emits one deterministic
`sounio.physical-extraction-materialization.v1` receipt.

The receipt type and statuses are exactly:

```text
materialization_type = verified-local-exact-copy
materialization_status = copied-and-verified
source_removal_status = not-authorized
assurance_level = identity-only
```

This is a physical copy boundary. It is not proof of a remote Git repository,
a push, a package publication, changed maintainership, or permission to delete
the source tree.

## Required Inputs

Materialization requires all of the following:

1. The original repository snapshot, `science-rings.tsv`, and ownership TSV
   still reproduce the supplied R3 inventory exactly.
2. The destination policy covers every `extract-planned` target exactly once
   and no retained or blocked target.
3. Every policy row is `approved`, binds the target ID, target owner, one local
   destination key, one content path, and at least one repository-local
   approval-evidence file by size and SHA-256.
4. The destination root is outside the source repository. Each destination is
   a preexisting, direct child directory on the same filesystem.
5. Each destination contains an exact
   `.sounio-destination-approval.json` marker whose byte hash is bound by the
   destination policy.
6. Every final content path and the requested receipt path are absent.

The policy contract is
`schemas/sounio.physical-extraction-destination-policy.v1.schema.json`. A
policy identity detects change; it is not a signature and does not establish
legal or hosted-service authority.

## Destination Marker

Each destination container carries this closed marker shape before copying:

```json
{
  "schema": "sounio.physical-extraction-destination.v1",
  "marker_type": "preexisting-approved-destination",
  "target_id": "distribution:example",
  "target_owner": "example-maintainers",
  "destination_key": "example-destination",
  "content_path": "payload",
  "approval_state": "approved",
  "source_inventory_identity_sha256": "<inventory identity>"
}
```

The marker is stored outside the copied content. Its exact serialized file
hash must equal `destination_marker_sha256` in the policy. A missing, changed,
symlinked, misplaced, or inventory-mismatched marker refuses before copying.

## Materialize And Verify

```bash
python3 tools/science_boundary/physical_extraction_materializer.py materialize \
  --repo-root . \
  --rings science-rings.tsv \
  --ownership docs/ecosystem/science-physical-extraction-ownership.tsv \
  --inventory target/science-physical-extraction-inventory.json \
  --destination-policy /approved/materialization-policy.json \
  --destinations-root /approved/destinations \
  --receipt /approved/materialization-receipt.json

python3 tools/science_boundary/physical_extraction_materializer.py verify \
  --repo-root . \
  --rings science-rings.tsv \
  --ownership docs/ecosystem/science-physical-extraction-ownership.tsv \
  --inventory target/science-physical-extraction-inventory.json \
  --destination-policy /approved/materialization-policy.json \
  --destinations-root /approved/destinations \
  --receipt /approved/materialization-receipt.json
```

Each source-root member is copied to the corresponding content directory with
its path relative to that source root. The tool copies regular-file bytes only,
refuses symbolic links, and checks every size and SHA-256 against the verified
inventory. The receipt deliberately does not claim preservation of file modes,
ownership bits, timestamps, ACLs, extended attributes, empty directories, or
other metadata absent from the R3 inventory.

Verification reconstructs the inventory and receipt from the current sources,
policies, markers, and destinations. Added, removed, mutated, symlinked,
forged, or rehashed content refuses.

## Promotion And Failure Semantics

All unit copies are staged and verified before any final content path is
promoted. Each unit is promoted by a same-filesystem directory rename. The
receipt is promoted last and acts as the completed-operation marker. Expected
failures before promotion leave every final content path absent; promotion
failures trigger a rollback attempt.

There is no cross-directory filesystem primitive that makes several separate
destination renames crash-atomic as one transaction. The receipt therefore
states `does_not_guarantee_crash_atomicity_across_multiple_destinations`.
After process or host interruption, the verifier is the authority for whether
all destinations and the receipt form one complete materialization. Partial
content without a valid receipt is not an accepted transfer.

## Evidence Boundary

A passing receipt establishes:

- one verified source inventory was consumed;
- every planned unit had an approved and preexisting local destination;
- each destination contains the exact inventoried regular-file bytes;
- retained and blocked units were not copied by this interface;
- the source-removal state remains `not-authorized`.

It does not establish:

- deletion or absence of the source files;
- remote repository creation, commit, push, branch, or default-branch state;
- ownership or maintainership transfer;
- registry or package publication;
- independent replay or environment capture;
- scientific truth, clinical validation, or clinical authority.

The adversarial gate is
`scripts/ci/physical_extraction_materialization_gate.py`. The composed shell
gate also executes R0-R2, R2.5, R2.6, and the R3 inventory gate against the
same source snapshot before accepting materialization behavior.

## Canonical Repository Status

The Sounio repository does not currently contain an approved production
destination policy or a production materialization receipt. Consequently, all
canonical package and research sources remain in place and the repository's
physical extraction is still operationally not executed. Passing fixture and
Slurm gates makes the interface executable; it does not invent the missing
approvals or perform a remote migration.

## Next Interface

`r3-physical-extraction-source-removal-authorization` now requires a valid
materialization receipt, two distinct review evidence records, an exact removal
scope, byte-bound repository repairs, and post-removal gates. It proves that
scope only in a temporary copy and emits `authorized-not-executed`; it does not
remove a Sounio source file. Actual removal remains the separate
`r3-physical-extraction-source-removal-execution` interface.
