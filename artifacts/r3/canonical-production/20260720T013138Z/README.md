# R3 Canonical Production Catalog Refresh Observation

This directory preserves the read-only observation needed before a new
canonical-production mapping reconfirmation. It records no mapping selection
and emits no reconciliation proposal.

## Bound Observation

```text
observed at UTC = 2026-07-20T01:31:38Z
organization = Sounio-lang
repository count = 14
catalog identity = 7bc569476058987386b096336256eef08eb4f2ac56d6c693c02cdb8ee7e933d6
catalog file SHA-256 = 1bdabd9dd2606c022bff4e0c0ddbf4ea93b7dbba7598dfe688d1dedd76e08c6e
GraphQL observation SHA-256 = ae6015ab39f87c98389d8ff3069f40041b1920b62c017ff69e223f4480ae11af
canonical source head = 5cf8be05b96c0a5c2ab101e022b36019dd61ebef
source inventory identity = e26c4dbbc19d127c13051213a156f7e323c7d3c4a4424a2b0c2f40600309bb67
source inventory file SHA-256 = cae392e7586a4b7cf100644be7426d2365b189e3ceeca76eb0be883461d1a9c4
```

The only catalog-row change from the prior post-reconciliation catalog
`243517f90deda6afc8c704bc5e0813302f67b9d9e91375f00c3e8821ef9894dc`
is the canonical `sounio` head, from `e19af3279...` to `5cf8be05...`.

## Governed Unit Continuity

The five extraction-planned source units are byte-identical to the prior
inventory:

| Target | Files | Bytes | Tree SHA-256 |
|---|---:|---:|---|
| `distribution:epistemic-core` | 6 | 25,179 | `5dcea277263dbb656b9c8cfa32ab8f8f148109e8f3a82cb76e33cd6fdd6fa114` |
| `distribution:sounio-formats` | 6 | 33,959 | `138783d224d8b0bb395b3fb1188773bf27ce44ed73d91d68553c1eb382804e76` |
| `distribution:sounio-io-primitives` | 4 | 8,639 | `95cb458790992fe514d2be97f39f719d6ecd8750bedea09384931f4d0996e35a` |
| `distribution:sounio-units` | 5 | 5,025 | `4665c96e4184b39fef442a15354eb1643dd4c88a46783f3908648f7500fe5310` |
| `distribution:sounio-research-examples` | 1,034 | 11,886,809 | `7c20223655534c853511a4652a5e458e0295f7764ec4a36cef85b067d2a474e4` |

`governed-source-unit-drift.diff` is the exact Git diff between source heads
`e19af3279...` and `5cf8be05...` for those five roots. It is zero bytes with
SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The five mapped destination heads are also unchanged:

```text
epistemic-core = 3e7d49fb84c7b8c74b8fd4b1cc39660772d9c7d1
sounio-formats = c412c0d1e7ef276d3ad9d1e662d681369e3e384c
sounio-io-primitives = 8e593615072e7ad9962ab27c0e316a8be521457d
sounio-units = 229d310f676d2a3a1e183983764da2ddd63f6fe0
sounio-examples = a22f66e0060ba6d007b8b69012ecadee7e9345bd
```

## Files

- `repository-observation.graphql.json`: saved complete GraphQL response.
- `repository-catalog.v1.json`: deterministic schema-bound catalog.
- `source-inventory.v1.json`: current physical-extraction inventory.
- `governed-source-unit-drift.diff`: empty exact diff for the five governed roots.
- `mapping-reconfirmation-request.md`: exact request for the next human input.
- `SHA256SUMS`: hashes for every other file in this directory.

## Boundary

The prior mapping-decision contract requires reconfirmation after catalog or
canonical source drift even when governed bytes are unchanged. Therefore this
phase does not reuse the old selection as authority and does not emit a new
mapping proposal, evidence set, or reconciliation proposal.

No observed source or destination repository was modified. No source or
destination file was copied, replaced, removed, committed, pushed, approved,
or cut over.
