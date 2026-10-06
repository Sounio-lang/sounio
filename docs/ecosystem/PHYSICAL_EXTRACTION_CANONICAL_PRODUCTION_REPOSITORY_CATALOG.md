<!-- docs:meta
topic_id: repo.docs.ecosystem.physical-extraction-canonical-production-repository-catalog
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ecosystem.physical-extraction-canonical-production-repository-catalog
-->

# Physical Extraction Canonical Production Repository Catalog

Status: executable R3 point-in-time repository metadata processing. The
catalog is an observation, not repository ownership evidence, mapping approval,
or production authority.

`tools/science_boundary/canonical_production_repository_catalog.py` converts a
saved GitHub GraphQL response into the repository catalog consumed by the
canonical-production gap and mapping-decision contracts. The tool has only
`build` and `verify` modes. It does not contain a GitHub client and cannot
create repositories, update refs, approve mappings, or execute cutover.

## Required Observation

The saved GraphQL response must contain exactly:

```text
data.organization.login
data.organization.repositories.totalCount
data.organization.repositories.nodes[]
```

Each node must contain `name`, `nameWithOwner`, `url`, `visibility`,
`isArchived`, `isEmpty`, `viewerPermission`, and `defaultBranchRef` with its
target OID. Responses containing GraphQL errors or unexpected fields are
refused. `totalCount` must equal the number of supplied nodes, so a paginated or
truncated observation cannot silently become a complete catalog.

The v1 processor accepts one complete connection observation. GitHub
connections whose `totalCount` exceeds the collected page must be handled by a
future pagination-bundle contract; v1 refuses them instead of treating the
first page as complete.

The organization login, owner/name relation, canonical GitHub URL, default
branch, head OID, visibility, flags, and permission enum are validated before
emission. A null viewer permission is preserved as `UNKNOWN`. Empty
repositories are refused because the v1 catalog requires an exact default
branch binding.

## Deterministic Processing

```bash
python3 tools/science_boundary/canonical_production_repository_catalog.py build \
  --graphql-observation /path/to/repository-observation.graphql.json \
  --organization Sounio-lang \
  --observed-at-utc 2026-07-20T01:31:38Z \
  --output /path/to/repository-catalog.v1.json

python3 tools/science_boundary/canonical_production_repository_catalog.py verify \
  --graphql-observation /path/to/repository-observation.graphql.json \
  --organization Sounio-lang \
  --observed-at-utc 2026-07-20T01:31:38Z \
  --catalog /path/to/repository-catalog.v1.json
```

Rows are sorted by repository ID. `catalog_identity_sha256` hashes canonical
JSON after removing only the identity field. The emitted file uses one fixed
ASCII serialization. Build refuses leaf symlinks and existing outputs, and
publishes through an atomic same-directory no-clobber link. Verify reconstructs
the complete payload and requires exact bytes.

## Authority Boundary

The output fixes:

```text
catalog_type = observed-hosting-repository-catalog
authority_scope = supplied-repository-metadata-observation
```

Observed `ADMIN`, `MAINTAIN`, or `WRITE` metadata does not prove a human's or
organization's authority. A catalog does not select target repositories. Any
mapping decision remains a separate permission-bearing input, and the existing
mapping contract requires a new selection record after catalog or canonical
source drift.

The adversarial gate is
`scripts/ci/physical_extraction_canonical_production_repository_catalog_gate.py`
and is composed into `scripts/ci/sounio_package_support_gate.sh`.

## Re-measured 2026-08-27 on rebase onto `origin/main@055825a3f9`

This lane's artifacts were recorded against canonical head
`5cf8be05b96c0a5c2ab101e022b36019dd61ebef` (catalog `7bc56947…`, observed
2026-07-20) and last integrated `origin/main@22111d11`. Rebasing onto
`055825a3f9` re-runs the same two read-only checks the
`20260720T030329Z/drift-observation.v1.json` recorded. **One of them no longer
holds, and it is named here rather than by editing the point-in-time record,
which was accurate when taken and is bound by `SHA256SUMS`.**

Destination repositories — **all five still match the catalog**, unchanged:

| Target | Repository | Cataloged head | Live head 2026-08-27 |
|---|---|---|---|
| `distribution:epistemic-core` | `Sounio-lang/epistemic-core` | `3e7d49fb…` | `3e7d49fb…` |
| `distribution:sounio-formats` | `Sounio-lang/sounio-formats` | `c412c0d1…` | `c412c0d1…` |
| `distribution:sounio-io-primitives` | `Sounio-lang/sounio-io-primitives` | `8e593615…` | `8e593615…` |
| `distribution:sounio-research-examples` | `Sounio-lang/sounio-examples` | `a22f66e0…` | `a22f66e0…` |
| `distribution:sounio-units` | `Sounio-lang/sounio-units` | `229d310f…` | `229d310f…` |

Governed source trees — **four of five unchanged; `examples` has drifted**:

| Source path | Bound tree OID | `origin/main@055825a3f9` | |
|---|---|---|---|
| `packages/epistemic-core` | `41bb77d1704b…` | `41bb77d1704b…` | unchanged |
| `packages/sounio-formats` | `a08b1e7cf51e…` | `a08b1e7cf51e…` | unchanged |
| `packages/sounio-io-primitives` | `d923d77a9df3…` | `d923d77a9df3…` | unchanged |
| `packages/sounio-units` | `44cf83512b96…` | `44cf83512b96…` | unchanged |
| `examples` | `fd977569a836…` | `e8b39d89ffb0…` | **CHANGED** |

`5cf8be05` is still an ancestor of `055825a3f9`, so the lineage the catalog
records is intact; what moved is the content of one governed unit.

The drifted unit is the one mapped to `distribution:sounio-research-examples`.
By this document's own Authority Boundary — *"the existing mapping contract
requires a new selection record after catalog or canonical source drift"* — the
five-row mapping in `20260720T030329Z/mapping-proposal.v1.json` may **not** be
carried forward to today's source on the strength of the 2026-07-20 selection
alone. It remains `proposed-not-approved`, execution authority `none`, and it is
now also **out of date with respect to one of its five source units**. A fresh
catalog, inventory and human selection are required before this proposal is read
as describing the current tree.

Re-measured with:

```bash
git rev-parse 055825a3f9:examples          # e8b39d89ffb0…
gh api repos/Sounio-lang/epistemic-core/commits/main --jq .sha
```
