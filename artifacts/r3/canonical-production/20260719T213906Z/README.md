# R3 `sounio-examples` Reconciliation Proposal v1

This directory preserves a path-complete, non-authorizing reconciliation
proposal for `distribution:sounio-research-examples` mapped to
`Sounio-lang/sounio-examples`.

## Result

```text
evidence identity = 7d62a39d1dec79aa76780608da6e93182b53703daf5f85fa663cab782429f971
source head = e19af3279a040a6a707967d786be657bdf0d4203
destination head = a22f66e0060ba6d007b8b69012ecadee7e9345bd
proposal identity = ef9b3401af36bebf57fc960eb9108c6d313a15df5d9ada9324dd6138d8ad43f0
path-plan SHA-256 = 86f7ebcb327d17760df4a761e44e99fcf417705696f9e10e6dcd3e08cacd903b
proposal status = proposed-not-approved
execution authority = none
destination write authority = none
source removal authority = none
canonical production approval = not-approved
cutover = not-executed
```

The complete plan contains 1,041 sorted path rows: 1,029 proposed additions,
3 proposed replacements, 7 destination-only paths described as proposed
removals, and 2 identical retained paths. All 1,039 mutation rows carry
`operation_authority = none`.

The proposed replacements are `README.md`, `hello.sio`, and
`uncertainty.sio`. The destination-only paths are `bubble_sort.sio`,
`csv_stats.sio`, `effect_demo.sio`, `hex_encode.sio`, `higher_order.sio`,
`monte_carlo_pi.sio`, and `newton_root.sio`. Their removal is described only
as the exact-mirror candidate; it is not authorized. `fibonacci.sio` and
`structs.sio` are identical retained paths.

## Files

- `production-reconciliation-proposal.v1.json`: complete source, destination,
  before-state, conditional after-state, disposition, and SHA-256 row for every
  path in the union of the two tracked trees.
- `SHA256SUMS`: fail-closed hashes for the files in this directory.

## Currency Boundary

The proposal deliberately reconstructs the evidence-bound source head
`e19af3279...`; it does not claim that this historical snapshot is the current
remote default-branch head. At proposal time, `origin/main` was separately
observed at `8fd67e4d70893e1b94bb4c89bb0e03d16526f90e`. Therefore this artifact is
review evidence only and must be reissued after a fresh source, destination,
catalog, mapping, and evidence observation before any separately authorized
operation could be considered.

## Boundary

The builder read two clean local clones and wrote only the proposal JSON. It
did not modify either repository, create a commit, push a ref, remove a source
file, approve production, or execute cutover. This directory records no
permission-bearing decision.
