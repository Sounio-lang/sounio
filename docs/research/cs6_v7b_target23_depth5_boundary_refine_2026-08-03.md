# CS6 V7-B target-23 depth-5 boundary refinement

**Status:** pre-execution frozen protocol.

## Plain question

The complete depth-4 sweep evaluated all 256 target-23 children. Both carriers
passed on 231 cells and rejected the same 25 cells. Their exact coordinates and
lower-left staircase map are recorded in
`scripts/research/receipts/cs6_v7b_target23_depth4_cover_v1/boundary-map.txt`.
This experiment subdivides every rejected cell once in each axis.

Each of the 25 source cells contributes four grandchildren. The resulting 100
grandchildren are each evaluated with both carriers, for 200 required attempts.
No already passing depth-4 cell is rerun.

## Exact inference boundary

If all 200 attempts pass, the existing 231 passing depth-4 cells and these 100
passing depth-5 grandchildren form an adaptive probe cover with 331 leaves.
That is a probe-level result only. It does not establish a certificate cover,
the C1/C2/determinant joint intersection, a V7-B winner, promotion, or a solved
open problem.

If any grandchild is rejected, the experiment instead returns a smaller map of
the persistent boundary obstruction. Partial passing grandchildren cannot be
promoted to an adaptive parent cover.

```text
SOURCE_REJECTED_PARENTS=25
GRANDCHILDREN_PER_PARENT=4
GRANDCHILD_CELLS=100
CARRIERS=2
ATTEMPTS=200
SOURCE_ORIGINAL_PARENT_DEPTH_DELTA=4
REFINED_ORIGINAL_PARENT_DEPTH_DELTA=5
REFINEMENT_STEP_DELTA=1
FPGA_EXECUTION=false
```

Heavy execution uses the proven r740 Slurm batch path with 32 CPU workers and
hash-bound worker-local staging. The result returns through one framed TCP
stream carrying its byte count and SHA-256.
