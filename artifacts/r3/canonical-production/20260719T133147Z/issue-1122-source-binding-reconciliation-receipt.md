## R3 public package source-binding reconciliation receipt - copied and verified

This receipt records a narrow exact-copy reconciliation after source PR
[#1176](https://github.com/Sounio-lang/sounio/pull/1176) merged. It does not
approve the proposed target mapping, canonical production, source removal,
registry publication, a release, or cutover.

### Source binding

- Canonical integration commit:
  [`d380146ffeabd3d18e71182ac4c03132f0788cf2`](https://github.com/Sounio-lang/sounio/commit/d380146ffeabd3d18e71182ac4c03132f0788cf2),
  the merge commit for PR #1176 at `2026-07-19T13:05:58Z`
- Verified source inventory identity:
  `022772f58a51cc4273d5f02043690398085e93b0b767c7e16b91e06d131a7014`
- Inventory status: `not-executed`; the inventory proves file identity and
  coverage, while this separate receipt records the remote reconciliation
- The four source paths remain present in the monorepo; source removal remains
  `not-authorized-not-executed`
- At the post-operation observation, `Sounio-lang/sounio` `main` was
  `e19af3279a040a6a707967d786be657bdf0d4203`. The bound merge is its ancestor,
  and all four package roots are unchanged between the bound merge and that
  observed head.

### Exact drift and destination result

Fresh clones were compared recursively against the bound source snapshot
before any destination write. Three repositories were already byte-identical.
Only `epistemic-core/src/lib.sio` differed, in exactly two comment lines that
remove unsupported claim wording; executable code was unchanged.

| Destination | Pre-operation `main` | Post-operation `main` | Result | Source tree SHA-256 |
|---|---|---|---|---|
| [`Sounio-lang/epistemic-core`](https://github.com/Sounio-lang/epistemic-core) | `732b3fbf7ff1d596cf591124b475791fe5e1add9` | [`3e7d49fb84c7b8c74b8fd4b1cc39660772d9c7d1`](https://github.com/Sounio-lang/epistemic-core/commit/3e7d49fb84c7b8c74b8fd4b1cc39660772d9c7d1) | reconciled: 1 file, 2 comment substitutions; final 6 files / 25,179 bytes | `5dcea277263dbb656b9c8cfa32ab8f8f148109e8f3a82cb76e33cd6fdd6fa114` |
| [`Sounio-lang/sounio-formats`](https://github.com/Sounio-lang/sounio-formats) | `c412c0d1e7ef276d3ad9d1e662d681369e3e384c` | unchanged | already exact: 6 files / 33,959 bytes | `138783d224d8b0bb395b3fb1188773bf27ce44ed73d91d68553c1eb382804e76` |
| [`Sounio-lang/sounio-io-primitives`](https://github.com/Sounio-lang/sounio-io-primitives) | `8e593615072e7ad9962ab27c0e316a8be521457d` | unchanged | already exact: 4 files / 8,639 bytes | `95cb458790992fe514d2be97f39f719d6ecd8750bedea09384931f4d0996e35a` |
| [`Sounio-lang/sounio-units`](https://github.com/Sounio-lang/sounio-units) | `229d310f676d2a3a1e183983764da2ddd63f6fe0` | unchanged | already exact: 5 files / 5,025 bytes | `4665c96e4184b39fef442a15354eb1643dd4c88a46783f3908648f7500fe5310` |

`Sounio-lang/sounio-examples` was outside the authorized copy scope and remains
unchanged at `a22f66e0060ba6d007b8b69012ecadee7e9345bd`.

### Verification evidence

- `bash scripts/ci/package_import_science_gate.sh`: PASS on the exact bound
  merge, including `epistemic-core` 9/9, GUM 5/5, MessagePack 11/11, all
  remaining package tests, package-import witnesses, and the negative fixture
- physical extraction inventory emission and verification: PASS
- `python3 scripts/ci/physical_extraction_inventory_gate.py`: PASS, 141 tests
- pre-push destination verification: exact file set and bytes matched the
  source inventory; `git diff --check` and `git fsck --full` passed
- remote update: one ordinary fast-forward push of `epistemic-core/main` from
  `732b3fbf...` to `3e7d49fb...`; no force push and no other ref was created
- post-push verification: a new independent clone reproduced the expected
  commit, exact six-file source tree, clean status, and `git fsck --full` PASS
- post-operation catalog observed at `2026-07-19T13:35:50Z`: 14 repositories,
  identity `243517f90deda6afc8c704bc5e0813302f67b9d9e91375f00c3e8821ef9894dc`
- mandatory LLM offload: the exact destination diff contained no mathematical
  content beyond narrowing two comments; the external receipt was withheld
  from publication until the required provider review completed

### Authority boundary

Executed here: compare four public package copies to the merged source,
reconcile only the single drifted copy, fast-forward one `main` ref, reclone
and verify it, and observe the resulting public refs.

Not executed: source deletion, changes to `sounio-examples`, tags, releases,
teams, branch rules, registry operations, ownership transfer, mapping
reconfirmation, production approval, or cutover.

The five-target proposal remains **`proposed-not-approved`**, with proposal
identity `a32de28e879ea03370f90382f0d67a3651a53b4108d8c45ed0403b1106921f2d`
and its authorized catalog binding
`cef66e6c59e9b7f4b35a5d4dd0637bfd71865a93b049a8af6dd4471ade8ad55a`.
This newer operational catalog observation does not amend, reconfirm, or
approve that proposal and supplies no execution authority.
