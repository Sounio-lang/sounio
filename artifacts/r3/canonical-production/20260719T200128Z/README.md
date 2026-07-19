# R3 Canonical Production Evidence Set v1

This directory preserves a read-only production-evidence draft bound to the
post-reconciliation catalog and five-row `proposed-not-approved` mapping.

## Result

```text
canonical source head = e19af3279a040a6a707967d786be657bdf0d4203
catalog identity = 243517f90deda6afc8c704bc5e0813302f67b9d9e91375f00c3e8821ef9894dc
proposal identity = 44f3a2f91534ca17fc0cd8e6794a78989629e5660256375464f33e48b743e069
evidence identity = 7d62a39d1dec79aa76780608da6e93182b53703daf5f85fa663cab782429f971
evidence status = production-evidence-draft-gaps-observed
exact parity targets = 4/5
execution authority = none
source removal authority = none
canonical production approval = not-approved
cutover = not-executed
```

`epistemic-core`, `sounio-formats`, `sounio-io-primitives`, and `sounio-units`
match their mapped destination repositories exactly by relative path, file
size, and SHA-256. `sounio-examples` does not match the source `examples` tree:
1,029 source files are missing, 7 destination files are extra, and 3 shared
paths differ.

The exact-source package import science gate passed with code 0. Its complete
stdout and empty stderr are preserved and bound by the validation manifest.

## Files

- `production-evidence-set.v1.json`: deterministic Git, inventory, parity,
  validation, proposed-sequence, and governance-gap evidence.
- `validation-observations.v1.json`: exact-source validation observation bound
  to the canonical source head.
- `package-import-science.stdout`: complete passing gate output.
- `package-import-science.stderr`: empty gate stderr.
- `SHA256SUMS`: fail-closed file hashes for this directory.

Run `sha256sum -c SHA256SUMS` from this directory. Re-run the evidence tool's
`verify` command with the catalog-bound source and five destination clones to
reconstruct the evidence identity from current bytes and Git observations.

## Boundary

No source or destination repository was modified, no Git ref was updated, and
no source removal, production approval, execution policy, human cutover
decision, or cutover execution was supplied. The observed `sounio-examples`
gap requires a separate reviewed materialization proposal before any write.
