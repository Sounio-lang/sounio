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
