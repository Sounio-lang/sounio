# R3 Public Package Materialization Evidence

This directory preserves the separately authorized exact-copy materialization
of four reviewed package roots into the public repositories provisioned earlier
on 2026-07-19.

The materialization source is Sounio commit
`317e6d085ad5304c8fac185eee03552a6b916123` on branch
`agent/r3-epistemic-materialization-corrections-20260719`. Source correction PR
[#1176](https://github.com/Sounio-lang/sounio/pull/1176) was still draft and
unmerged at observation time. The package source paths remain present at that
commit; no source removal or canonical cutover was authorized.

## Identities

- source inventory identity:
  `f03458beb2ed07380e7d4a7b1242bb7b32e3c609a47f204e44c34eca64a429e5`
- materialization receipt identity:
  `9d1240fb0d69458b60d93d0d37c7c0f49a2518b0c7ca994cf71a982edf9f197a`
- post-materialization catalog identity:
  `095de409e315ff0c716c4877274c8b2d439310bd255233cf1558f42f2b19be2c`
- public issue receipt body SHA-256:
  `3766198f23ca65dbe43333f69802562da73f034821146af7b8ac95e1baa943e6`

The catalog was observed at `2026-07-19T02:48:12Z`, contains 14
repositories, and records canonical `Sounio-lang/sounio` `main` at
`d05f8069cbb88ec49fd837b8706a6015c9676996`.

## Destination Commits

| Repository | `main` commit | Files | Inventory tree SHA-256 |
|---|---|---:|---|
| `epistemic-core` | `732b3fbf7ff1d596cf591124b475791fe5e1add9` | 6 | `a8add81263e2ffa3f658d9284a2d9e155a4e7466e813c87ef2d15c71b4720291` |
| `sounio-formats` | `c412c0d1e7ef276d3ad9d1e662d681369e3e384c` | 6 | `138783d224d8b0bb395b3fb1188773bf27ce44ed73d91d68553c1eb382804e76` |
| `sounio-io-primitives` | `8e593615072e7ad9962ab27c0e316a8be521457d` | 4 | `95cb458790992fe514d2be97f39f719d6ecd8750bedea09384931f4d0996e35a` |
| `sounio-units` | `229d310f676d2a3a1e183983764da2ddd63f6fe0` | 5 | `4665c96e4184b39fef442a15354eb1643dd4c88a46783f3908648f7500fe5310` |

Each root commit binds the Sounio source commit, inventory identity, unit tree
hash, `exact-copy-no-source-removal` scope, and the mandatory xAI and Z.AI
review provenance.

## Public Receipt

Issue comment
[`5013887205`](https://github.com/Sounio-lang/sounio/issues/1122#issuecomment-5013887205)
records the result. Its local and API-returned bodies are byte-identical: 5656
bytes with SHA-256
`3766198f23ca65dbe43333f69802562da73f034821146af7b8ac95e1baa943e6`.

## Evidence Files

- `source-inventory.v1.json`: verified seven-unit source inventory containing
  the four materialized package units and their per-file hashes.
- `repository-observation.graphql.json`: organization-wide GraphQL
  observation.
- `repository-catalog.v1.json`: validated 14-row post-materialization catalog.
- `git-ref-observation.json`: fresh unauthenticated main-ref observation for
  all four destinations.
- `repository-materialization-receipt.v1.json`: deterministic, scope-limited
  materialization receipt.
- `issue-1122-materialization-receipt.md`: exact public receipt body.
- `issue-1122-materialization-receipt.api.json`: GitHub API observation of the
  published receipt.

Run `sha256sum -c SHA256SUMS` from this directory for a fail-closed byte check.

## Verification

The bound source passed `package_import_science_gate.sh`, including 9/9
`epistemic-core`, 5/5 GUM, all remaining package tests, and both package-import
witnesses. Inventory emission, verification, and its 141-assertion gate passed.
All destination bytes were checked before push; four fresh post-push clones
then reproduced the file hashes and expected commits with clean status and
`git fsck --full --no-dangling` success.

## Remaining Boundary

The requested `sounio-scientific-packages-maintainers` team slug remains
unobserved, and no team, branch rule, tag, or release was created. Existing
manifest repository fields were preserved. `sounio-examples` was not changed.
The next action is an explicit human reconfirmation of all five target mappings
against catalog identity
`095de409e315ff0c716c4877274c8b2d439310bd255233cf1558f42f2b19be2c`.
This evidence is not that reconfirmation and does not emit a proposal or approve
canonical production, source removal, registry publication, or cutover.
