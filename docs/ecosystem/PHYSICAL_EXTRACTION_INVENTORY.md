<!-- docs:meta
topic_id: repo.docs.ecosystem.physical-extraction-inventory
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ecosystem.physical-extraction-inventory
-->

# Physical Extraction Inventory

Status: executable R3 ownership and file-identity inventory; physical extraction is not executed.

## Scope

R3 begins the physical separation of programming-language core, scientific
packages, and research artifacts. This first interface makes the proposed
ownership boundary executable before any source is removed or copied to a new
distribution.

The inventory type is exactly:

```text
physical-extraction-planning-snapshot
```

Its extraction status is exactly `not-executed`, and its authority scope is
`repository-file-identity-and-ownership-plan`. A passing snapshot establishes
which regular files existed under each declared root, their hashes, and the
retain, extract, or blocked disposition recorded by the local ownership policy.
It does not establish that a destination exists, that ownership changed, or
that a transfer completed.

## Ownership Policy

The policy is
`docs/ecosystem/science-physical-extraction-ownership.tsv`. It covers every
root in `science-rings.tsv` exactly once and has these fixed columns:

```text
source_path ring current_owner target_kind target_id target_owner
disposition migration_state ownership_evidence extraction_gate
```

Owner and target identifiers are planning labels in this repository. They are
not legal ownership statements, GitHub permissions, registry namespaces, or
evidence that the named destination has been created.

The v1 disposition rules are intentionally closed:

| Ring | Target kind | Disposition | Migration state |
|---|---|---|---|
| `pl-core` | `same-repository` | `retain-core` | `retained` |
| `scientific-package` | `separate-distribution` | `extract-planned` | `planned` |
| `research` | `separate-distribution` | `extract-planned` | `planned` |
| candidate or unresolved | `unassigned` | `hold-unresolved` | `blocked-classification` |

Consequently, the current `self-hosted` root remains in the Sounio core. The
four explicitly classified package roots and `examples` have separate future
distribution identifiers. The mixed `stdlib` remains blocked until its
ring-by-ring inventory is complete; R3 does not infer a classification for it.

## Exact File Snapshot

Each declared root must be a non-overlapping repository-relative directory.
The tool recursively inventories every regular file, recording:

- repository-relative path;
- byte size;
- SHA-256 digest;
- deterministic per-unit tree digest.

Symbolic links, escaping paths, empty roots, duplicate roots, overlapping
roots, missing ownership rows, extra ownership rows, ring mismatches, and
ring-incompatible dispositions refuse. No exclusion pattern silently removes
content from a declared root.

The complete JSON contract is
`schemas/sounio.physical-extraction-inventory.v1.schema.json`.

## Emit And Verify

```bash
python3 tools/science_boundary/physical_extraction_inventory.py inventory \
  --repo-root . \
  --rings science-rings.tsv \
  --ownership docs/ecosystem/science-physical-extraction-ownership.tsv \
  --output target/science-physical-extraction-inventory.json

python3 tools/science_boundary/physical_extraction_inventory.py verify \
  --inventory target/science-physical-extraction-inventory.json \
  --repo-root . \
  --rings science-rings.tsv \
  --ownership docs/ecosystem/science-physical-extraction-ownership.tsv
```

Emission is deterministic across output and policy locations when content is
unchanged. The final JSON is promoted through a sibling staging file and an
existing output is never overwritten. Verification reconstructs the complete
snapshot from the current repository and both TSV inputs. Added, deleted,
changed, forged, or rehashed content refuses.

## Evidence Boundary

The snapshot uses `identity-only` assurance. It establishes file identity,
coverage, and consistency with a local ownership plan. It does not:

- move or delete source files;
- create a repository or distribution;
- transfer ownership or maintainership;
- publish a package or registry entry;
- establish independent replay;
- establish scientific truth, clinical validation, or clinical authority.

The adversarial gate is
`scripts/ci/physical_extraction_inventory_gate.py`. The composed shell gate
also executes R0-R2, R2.5, and R2.6 before accepting this R3 inventory.

## Materialization Interface

`r3-physical-extraction-materialization` consumes one verified inventory,
requires explicitly approved preexisting local destinations, copies the exact
planned regular-file bytes, and verifies destination identity. Its receipt
keeps source removal `not-authorized`; none of those copy claims are implied by
an inventory alone. The complete contract is
`docs/ecosystem/PHYSICAL_EXTRACTION_MATERIALIZATION.md`.
