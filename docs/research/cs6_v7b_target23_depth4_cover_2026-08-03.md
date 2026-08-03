# CS6 V7-B target-23 depth-4 sibling cover

**Status:** executed and independently verified. The parent was fully evaluated,
but it does not have a complete probe cover at depth delta `+4`.

## Plain question

The previous ladder found that the lower-left descendant of target cell `23`
emits `PROBE_PASS=true` at depth delta `+4`, for both tested carriers. One
passing descendant does not tell us what happens in its 255 siblings.

This experiment evaluates the entire `16 x 16` dyadic partition at that same
depth. Every one of the 256 children is run with both carriers, producing 512
required attempts.

## Verdicts

The result must distinguish three levels:

- `PARENT_COVER_EVALUATED=true` means all 512 required attempts produced a
  classified result with no unknown failures;
- `PARENT_PROBE_COVER_PASS=true` requires `PROBE_PASS=true` in every attempt;
- `PARENT_CERTIFICATE_COVER_PASS=true` requires `CERTIFICATE_PASS=true` in
  every attempt.

A partial collection of passing children is only a localization map. It cannot
establish a parent cover. Even a complete probe cover does not by itself enable
V7-B eligibility. The parent V7-B contract independently requires a nonempty
intersection of the Liouville, C1, C2, and section-resident determinant
enclosures, followed by its frozen winner criterion. This experiment does not
evaluate that joint intersection or select a winner.

## Frozen matrix

```text
PARENT_NODE=U03-0000000006_S04-0000000010
DEPTH_DELTA=4
CHILD_U_DEPTH=7
CHILD_S_DEPTH=8
CHILD_OFFSETS=0..15 x 0..15
CHILD_COUNT=256
CARRIERS=2
ATTEMPT_COUNT=512
```

Heavy execution is routed through Slurm with 32 CPU workers. No FPGA is used.
The preferred r770 worker currently fails before launching even a trivial batch
command (`RaisedSignal:53`). The frozen job therefore permits the r740 worker as
an explicit fallback. Payloads remain hash-bound in worker-local staging, and
the result archive returns as one framed TCP stream carrying its byte count and
SHA-256; the receiver must independently check both before accepting evidence.

## Local protocol gate

```bash
bash scripts/ci/cs6_v7b_target23_depth4_cover_gate.sh
```

The gate validates the exact partition, coordinate uniqueness, attempt count,
claim fences, Python syntax, and Slurm job syntax. It does not execute CAPD and
does not claim a scientific result.

## Result

Slurm job `8519` completed on `gpuorangefs-multi-r740-proxmox` in `00:04:40`.
All 512 attempts were classified, with no timeout or unknown failure:

```text
CHILD_CELLS_EVALUATED=256
ATTEMPTS_COMPLETED=512
PROBE_PASS_ATTEMPTS=462
PROBE_REJECTED_ATTEMPTS=50
BOTH_CARRIERS_PROBE_PASS_CELLS=231
BOTH_CARRIERS_PROBE_REJECT_CELLS=25
MIXED_CARRIER_CELLS=0
CERTIFICATE_PASS_ATTEMPTS=0
PARENT_COVER_EVALUATED=true
PARENT_PROBE_COVER_PASS=false
PARENT_CERTIFICATE_COVER_PASS=false
V7_B_ELIGIBILITY=false
OPEN_PROBLEM_SOLVED=false
```

The two carriers agree on every child. The 25 rejected children form a compact
staircase at the lower-left boundary rather than a scattered pattern:

```text
    S offset -> 0123456789abcdef
U 00           ................
  01           ................
  02           ................
  03           X...............
  04           X...............
  05           X...............
  06           X...............
  07           X...............
  08           XX..............
  09           XX..............
  10           XX..............
  11           XX..............
  12           XXX.............
  13           XXX.............
  14           XXX.............
  15           XXX.............
```

`X` means both carriers returned `DESCENDANT_PROBE_REJECTED`; `.` means both
returned `DESCENDANT_PROBE_PASS`. The next evidence-producing step is therefore
not another broad sweep. It is one additional dyadic subdivision of these 25
boundary cells: 100 grandchildren and 200 carrier attempts. That can determine
whether the staircase is a shrinking boundary layer or a persistent obstruction.

## Evidence binding

The cluster and local verifier outputs are byte-identical. The returned raw tar
has 20,510,720 bytes and SHA-256
`9c9d1615a318275e23cc98749a8ce436f4b6e49665154e129bcf237a5eafb1d6`.
The complete compressed archive, the 512-row result ledger, the coordinate
manifest, both verification outputs, and final Slurm state are retained under
`scripts/research/receipts/cs6_v7b_target23_depth4_cover_v1/`.
