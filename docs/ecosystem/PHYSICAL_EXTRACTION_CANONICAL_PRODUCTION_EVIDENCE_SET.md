<!-- docs:meta
topic_id: repo.docs.ecosystem.physical-extraction-canonical-production-evidence-set
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ecosystem.physical-extraction-canonical-production-evidence-set
-->

# Physical Extraction Canonical Production Evidence Set

Status: executable R3 read-only evidence contract; proposal remains
`proposed-not-approved`, production remains `not-approved`, and cutover remains
`not-executed`.

`tools/science_boundary/canonical_production_evidence_set.py` binds a clean
canonical source snapshot, a closed repository catalog, an exact mapping
proposal, supplied validation observations, and local clones of every mapped
destination. It inventories and compares every regular tracked file under each
planned extraction unit and emits a deterministic evidence draft.

The contract fixes:

```text
authority_scope = evidence-observation-only
proposal_status = proposed-not-approved
execution_authority = none
source_removal_authority = none
canonical_production_approval = not-approved
canonical_cutover_execution_status = not-executed
```

No evidence status means `ready`, `approved`, `authorized`, or `executed`.
Exact parity satisfies only the evidence-bearing materialization prerequisite;
it does not satisfy any permission-bearing prerequisite.

## Inputs

The builder consumes:

1. A clean canonical Git worktree at the cataloged default branch and head.
2. `science-rings.tsv` and the complete physical-extraction ownership policy.
   The inventory is rebuilt from current bytes rather than accepted as a count
   or hash supplied by the caller.
3. A validated point-in-time repository catalog and a complete
   `proposed-not-approved` mapping proposal bound to that catalog.
4. A destinations root whose direct children are the mapped repository IDs.
   Every clone must be clean and match the proposed remote, branch, and head.
5. A validation-observation manifest bound to the exact canonical source head.
   The manifest records command outcome and stdout/stderr hashes; the evidence
   verifier validates the record but does not replay the command.

Source and destination inventories must equal their respective Git tracked-file
sets. Symlinks, special files, dirty worktrees, ignored or untracked content,
stale heads, wrong branches, wrong remotes, incomplete mappings, invalid
identities, and occupied output paths refuse.

The catalog and proposal bindings contain two complementary hashes. A
`*_file_sha256` hashes the exact formatted input bytes. A
`*_identity_sha256` hashes canonical JSON with its own identity field omitted;
it remains stable across insignificant JSON formatting changes. The validation
manifest uses the same identity rule.

The reviewed mapping intentionally binds
`distribution:sounio-research-examples` to repository ID `sounio-examples`.
The different names are proposal data, not an inferred name match.

## Build

```bash
python3 tools/science_boundary/canonical_production_evidence_set.py build \
  --repo-root /exact/catalog-bound/source \
  --rings science-rings.tsv \
  --ownership docs/ecosystem/science-physical-extraction-ownership.tsv \
  --repository-catalog /reviewed/repository-catalog.json \
  --mapping-proposal /reviewed/mapping-proposal.json \
  --validation-observations /reviewed/validation-observations.json \
  --destinations-root /read-only/local/clones \
  --canonical-repository-id sounio \
  --output /unoccupied/production-evidence-set.json
```

The builder performs no copy, deletion, commit, push, ref update, tag, release,
registry operation, approval, or cutover. It reads the supplied repositories
and creates only the requested JSON output.

## Parity Model

For each target the output binds:

- source inventory file count, bytes, original inventory tree identity, and a
  relative-path comparison tree identity;
- destination repository remote, branch, head, and clean state;
- destination file count, bytes, and comparison tree identity;
- exact counts for matching, missing, extra, and changed paths;
- up to 20 sorted examples from each difference class and one completeness flag
  for each class.

`exact-copy-verified` requires identical path, size, and SHA-256 rows across the
complete trees. Otherwise the target is `parity-gap-observed`. Byte equality is
strong materialization evidence, but it does not prove scientific truth,
empirical validity, destination-owner approval, or clinical authority.

The overall status is:

- `production-evidence-draft-exact-parity` only when every target has exact
  parity and every supplied validation observation passed;
- `production-evidence-draft-gaps-observed` when any parity or validation gap
  exists.

## Verify

```bash
python3 tools/science_boundary/canonical_production_evidence_set.py verify \
  --evidence /reviewed/production-evidence-set.json \
  --repo-root /exact/catalog-bound/source \
  --rings science-rings.tsv \
  --ownership docs/ecosystem/science-physical-extraction-ownership.tsv \
  --repository-catalog /reviewed/repository-catalog.json \
  --mapping-proposal /reviewed/mapping-proposal.json \
  --validation-observations /reviewed/validation-observations.json \
  --destinations-root /read-only/local/clones \
  --canonical-repository-id sounio
```

Verification rebuilds the source inventory, all Git observations, all
destination inventories, parity rows, validation bindings, summary, governance
gaps, proposed execution plan, limitations, and evidence identity. Rehashing a
forged artifact does not make it valid.

## Current Evidence

The evidence under
`artifacts/r3/canonical-production/20260719T200128Z/` is bound to canonical
source head `e19af3279a040a6a707967d786be657bdf0d4203`, catalog identity
`243517f90deda6afc8c704bc5e0813302f67b9d9e91375f00c3e8821ef9894dc`,
and proposal identity
`44f3a2f91534ca17fc0cd8e6794a78989629e5660256375464f33e48b743e069`.

The exact-source `package_import_science_gate.sh` observation passed with exit
code 0, stdout SHA-256
`90889e4dadc967ec0f99f529ebb5b31b18f55ad166d01dc266652ede5c58d824`,
and empty stderr.

| Target | Destination head | Parity |
|---|---|---|
| `distribution:epistemic-core` | `3e7d49fb84c7b8c74b8fd4b1cc39660772d9c7d1` | exact, 6 files |
| `distribution:sounio-formats` | `c412c0d1e7ef276d3ad9d1e662d681369e3e384c` | exact, 6 files |
| `distribution:sounio-io-primitives` | `8e593615072e7ad9962ab27c0e316a8be521457d` | exact, 4 files |
| `distribution:sounio-research-examples` | `a22f66e0060ba6d007b8b69012ecadee7e9345bd` | gap: 1,029 missing, 7 extra, 3 changed |
| `distribution:sounio-units` | `229d310f676d2a3a1e183983764da2ddd63f6fe0` | exact, 5 files |

Evidence identity is
`7d62a39d1dec79aa76780608da6e93182b53703daf5f85fa663cab782429f971`.
The status is `production-evidence-draft-gaps-observed`; four permission-bearing
prerequisites remain explicitly missing.

## Acceptance Gate

The focused gate is
`scripts/ci/physical_extraction_canonical_production_evidence_set_gate.py`.
It uses only temporary standalone Git repositories and local bare remotes. It
passes 57 assertions covering deterministic equivalent roots, exact parity,
missing/extra/changed files, failed validation, stale validation head, invalid
identity, dirty destination, occupied output preservation, forged and rehashed
evidence, source/destination immutability, and fixed non-authorizing fields.

The composed evidence shell gate first invokes the composed
canonical-production gap gate, which includes the prior extraction stack, and
then invokes the focused evidence gate. The package-support gate runs the same
focused evidence gate directly after its focused production-gap step.

## Remaining Boundary

The separate path-complete reconciliation proposal for
`Sounio-lang/sounio-examples` is preserved under
`artifacts/r3/canonical-production/20260719T213906Z/`. It remains
`proposed-not-approved`, describes 1,029 additions, 3 replacements, and 7
destination-only removals, and grants no destination-write authority. The
evidence-bound source head is now historical relative to remote `main`, so the
next evidence action is a fresh catalog, mapping, evidence, and reconciliation
proposal chain. This evidence set and the separate proposal do not authorize
repository modification.

Even after exact parity, source removal, canonical production approval,
execution/recovery policy, and an explicit human cutover decision remain
separate permission-bearing interfaces. The embedded execution sequence is
`draft-not-authorized` and cannot itself be executed.
