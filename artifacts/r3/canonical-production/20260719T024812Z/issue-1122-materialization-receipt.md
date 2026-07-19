## R3 public package materialization receipt — executed and verified

This receipt records exact-copy materialization of four previously provisioned empty public repositories. It does not approve a canonical mapping, source removal, registry publication, release, ownership transfer, or cutover.

### Source binding

- Sounio source commit: [`317e6d085ad5304c8fac185eee03552a6b916123`](https://github.com/Sounio-lang/sounio/commit/317e6d085ad5304c8fac185eee03552a6b916123)
- Source correction PR: [#1176](https://github.com/Sounio-lang/sounio/pull/1176), still **draft** and not merged at receipt time
- Parent `origin/main` observed for that source commit: `d05f8069cbb88ec49fd837b8706a6015c9676996`
- Inventory identity: `f03458beb2ed07380e7d4a7b1242bb7b32e3c609a47f204e44c34eca64a429e5`
- Inventory status: `not-executed` (identity/coverage evidence; materialization execution is recorded by this separate receipt)
- Source paths retained: `packages/epistemic-core`, `packages/sounio-formats`, `packages/sounio-io-primitives`, and `packages/sounio-units` all remain present at the bound source commit

The source correction was required before publication: independent math review caught a wrong README result and a zero-central-value uncertainty defect. The corrected package gate passed before this materialization.

Because PR #1176 is still draft, `origin/main` and the public package copies temporarily differ on those corrections. The destination commits are intentionally bound to the remote review-branch commit above. If that PR is rejected, rewritten, or changed materially, the public copies require a new source binding and verification receipt before any canonical use; there is no automatic reconciliation claim here.

### Destination results

| Source path | Public destination | `main` commit | Files | Bytes | Inventory tree SHA-256 |
|---|---|---:|---:|---:|---|
| `packages/epistemic-core` | [`Sounio-lang/epistemic-core`](https://github.com/Sounio-lang/epistemic-core) | [`732b3fbf7ff1d596cf591124b475791fe5e1add9`](https://github.com/Sounio-lang/epistemic-core/commit/732b3fbf7ff1d596cf591124b475791fe5e1add9) | 6 | 25,206 | `a8add81263e2ffa3f658d9284a2d9e155a4e7466e813c87ef2d15c71b4720291` |
| `packages/sounio-formats` | [`Sounio-lang/sounio-formats`](https://github.com/Sounio-lang/sounio-formats) | [`c412c0d1e7ef276d3ad9d1e662d681369e3e384c`](https://github.com/Sounio-lang/sounio-formats/commit/c412c0d1e7ef276d3ad9d1e662d681369e3e384c) | 6 | 33,959 | `138783d224d8b0bb395b3fb1188773bf27ce44ed73d91d68553c1eb382804e76` |
| `packages/sounio-io-primitives` | [`Sounio-lang/sounio-io-primitives`](https://github.com/Sounio-lang/sounio-io-primitives) | [`8e593615072e7ad9962ab27c0e316a8be521457d`](https://github.com/Sounio-lang/sounio-io-primitives/commit/8e593615072e7ad9962ab27c0e316a8be521457d) | 4 | 8,639 | `95cb458790992fe514d2be97f39f719d6ecd8750bedea09384931f4d0996e35a` |
| `packages/sounio-units` | [`Sounio-lang/sounio-units`](https://github.com/Sounio-lang/sounio-units) | [`229d310f676d2a3a1e183983764da2ddd63f6fe0`](https://github.com/Sounio-lang/sounio-units/commit/229d310f676d2a3a1e183983764da2ddd63f6fe0) | 5 | 5,025 | `4665c96e4184b39fef442a15354eb1643dd4c88a46783f3908648f7500fe5310` |

Every root commit includes the bound source commit, inventory identity, unit tree hash, `exact-copy-no-source-removal` scope, and LLM-review provenance in its commit message.

### Verification evidence

- `bash scripts/ci/package_import_science_gate.sh`: PASS on the bound source snapshot
  - `epistemic-core`: 9/9
  - `epistemic-core` GUM suite: 5/5
  - MessagePack: 11/11
  - remaining package tests and both package-import witnesses: PASS
- physical extraction inventory emit + verify: PASS
- `scripts/ci/physical_extraction_inventory_gate.py`: PASS, 141 assertions
- pre-push verification: exact file set, size, per-file SHA-256, and inventory tree identity for all four destinations
- post-push verification: four fresh clones, identical hashes, expected `main` OIDs, clean status, and `git fsck --full --no-dangling` PASS
- mandatory offload review: xAI/Grok 4.3 + Z.AI/GLM-5.2; both final public-artifact reviews returned `PROCEED`

### Fresh hosting observation

- Observed at: `2026-07-19T02:48:12Z`
- Repository catalog identity: `095de409e315ff0c716c4877274c8b2d439310bd255233cf1558f42f2b19be2c`
- Catalog: 14 organization repositories; all four destinations observed `PUBLIC`, non-empty, default branch `main`, permission `ADMIN`, and with the exact commits above
- `Sounio-lang/sounio-examples` remained unchanged at `a22f66e0060ba6d007b8b69012ecadee7e9345bd`
- Requested team `sounio-scientific-packages-maintainers`: not observed (`HTTP 404`); no team was created or assigned

### Authority boundary and next step

Executed here: exact package-byte copy, initial `main` commits, pushes, fresh-clone verification, and post-materialization catalog observation.

Not executed: tags, releases, teams, branch rules, source deletion, mapping proposal, canonical approval, registry publication, or cutover. Maintainer-team creation/assignment and branch-rule evidence remain explicit deferred prerequisites, not inferred organization defaults. Existing manifest repository fields were preserved byte-for-byte; repairs to those fields are a separate future phase.

The next action still requires an explicit human reconfirmation of all five target mappings (`epistemic-core`, `sounio-formats`, `sounio-io-primitives`, `sounio-units`, and the already-existing `sounio-examples`) against catalog identity `095de409e315ff0c716c4877274c8b2d439310bd255233cf1558f42f2b19be2c`. This receipt is not that reconfirmation.
