# R3 Public Package Source-Binding Reconciliation Evidence

This directory preserves the scope-limited reconciliation of the existing
public package copies after source PR
[#1176](https://github.com/Sounio-lang/sounio/pull/1176) merged as
`d380146ffeabd3d18e71182ac4c03132f0788cf2`.

The operation compared four fresh destination clones to the exact merged
source. `sounio-formats`, `sounio-io-primitives`, and `sounio-units` were
already byte-identical. Only `epistemic-core/src/lib.sio` differed, in two
comment lines. One ordinary fast-forward commit reconciled that repository;
no other destination or ref changed.

## Identities

- source inventory identity:
  `022772f58a51cc4273d5f02043690398085e93b0b767c7e16b91e06d131a7014`
- reconciliation receipt identity:
  `1ceceb68ea56593770e94f94dfbf4433bdad827d3e90c02220c677603a1b300e`
- post-reconciliation catalog identity:
  `243517f90deda6afc8c704bc5e0813302f67b9d9e91375f00c3e8821ef9894dc`
- public issue receipt body SHA-256:
  `0030bb020e087ded0735e527c78431ec241c491d46128bcfe1b33843fa5beaa5`

## Destination Result

| Repository | Pre-operation `main` | Post-operation `main` | Result | Source tree SHA-256 |
|---|---|---|---|---|
| `epistemic-core` | `732b3fbf7ff1d596cf591124b475791fe5e1add9` | `3e7d49fb84c7b8c74b8fd4b1cc39660772d9c7d1` | reconciled, 6 files / 25,179 bytes | `5dcea277263dbb656b9c8cfa32ab8f8f148109e8f3a82cb76e33cd6fdd6fa114` |
| `sounio-formats` | `c412c0d1e7ef276d3ad9d1e662d681369e3e384c` | unchanged | already exact | `138783d224d8b0bb395b3fb1188773bf27ce44ed73d91d68553c1eb382804e76` |
| `sounio-io-primitives` | `8e593615072e7ad9962ab27c0e316a8be521457d` | unchanged | already exact | `95cb458790992fe514d2be97f39f719d6ecd8750bedea09384931f4d0996e35a` |
| `sounio-units` | `229d310f676d2a3a1e183983764da2ddd63f6fe0` | unchanged | already exact | `4665c96e4184b39fef442a15354eb1643dd4c88a46783f3908648f7500fe5310` |

`sounio-examples` was outside the copy scope and remained at
`a22f66e0060ba6d007b8b69012ecadee7e9345bd`.

## Verification

The exact bound source passed `package_import_science_gate.sh`, including
`epistemic-core` 9/9, GUM 5/5, MessagePack 11/11, all remaining package tests,
the package-import witnesses, and the negative fixture. Inventory emission and
verification passed, as did the 141-test adversarial inventory gate.

Before the push, the updated destination reproduced the source file set and
bytes, `git diff --check` passed, the worktree contained only the intended
file change, and `git fsck --full` passed. After the push, a separate fresh
clone reproduced commit `3e7d49fb...`, the exact six-file source tree, clean
status, and a passing `git fsck --full`.

The post-operation catalog was observed at `2026-07-19T13:35:50Z`. The
monorepo `main` had advanced to `e19af3279a040a6a707967d786be657bdf0d4203`,
but the bound merge is its ancestor and the four package roots are unchanged
between those commits.

The directory label `20260719T133147Z` records the operation start, before the
destination commit at `2026-07-19T13:33:47Z`, catalog observation, and public
receipt. The pre-operation `epistemic-core` tree hash is independently bound by
the earlier materialization receipt under `20260719T024812Z/`.

## Public Receipt

Issue comment
[`5015935195`](https://github.com/Sounio-lang/sounio/issues/1122#issuecomment-5015935195)
records the result. Its local and API-returned bodies are byte-identical: 4,992
bytes with SHA-256
`0030bb020e087ded0735e527c78431ec241c491d46128bcfe1b33843fa5beaa5`.

## Evidence Files

- `source-inventory.v1.json`: verified seven-unit source inventory.
- `epistemic-core-comment-drift.diff`: exact two-hunk, comment-only destination
  diff from the pre-operation to post-operation commit.
- `repository-observation.graphql.json`: raw organization observation.
- `repository-catalog.v1.json`: validated 14-row point-in-time catalog.
- `git-ref-observation.v1.json`: source and five destination ref observation.
- `source-binding-reconciliation-receipt.v1.json`: deterministic operation
  receipt with source, drift, push, verification, catalog, mapping, and public
  comment bindings.
- `issue-1122-source-binding-reconciliation-receipt.md`: exact public body.
- `issue-1122-source-binding-reconciliation-receipt.api.json`: GitHub API
  response used for byte equality.

Run `sha256sum -c SHA256SUMS` from this directory for a fail-closed byte check.

## Authority Boundary

All monorepo source paths remain present. No force push, examples update, tag,
release, team, branch rule, registry operation, ownership transfer, mapping
reconfirmation, production approval, or cutover was performed.

The mapping proposal remains `proposed-not-approved`, identity
`a32de28e879ea03370f90382f0d67a3651a53b4108d8c45ed0403b1106921f2d`,
with authority `none` and its explicit catalog binding
`cef66e6c59e9b7f4b35a5d4dd0637bfd71865a93b049a8af6dd4471ade8ad55a`.
The newer operational catalog is observation evidence only and does not amend
or approve that proposal.
