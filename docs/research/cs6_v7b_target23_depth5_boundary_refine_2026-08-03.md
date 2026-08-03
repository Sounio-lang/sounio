# CS6 V7-B target-23 depth-5 boundary refinement

**Status:** executed and independently verified. The adaptive parent probe cover
passes; certificate cover and V7-B eligibility remain false.

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

## Result

Slurm job `8523` completed on `gpuorangefs-multi-r740-proxmox` in `00:02:06`.
All 200 attempts passed the probe and no attempt timed out or failed:

```text
SOURCE_DEPTH4_PASS_CELLS=231
SOURCE_REJECTED_PARENT_CELLS=25
GRANDCHILD_CELLS_EVALUATED=100
ATTEMPTS_COMPLETED=200
PROBE_PASS_ATTEMPTS=200
PROBE_REJECTED_ATTEMPTS=0
UNKNOWN_FAILURE=0
BOTH_CARRIERS_PROBE_PASS_CELLS=100
REFINED_PARENTS_FULL_PROBE_COVER=25
REFINED_PARENTS_WITH_REJECTION=0
ADAPTIVE_PARENT_PROBE_COVER_EVALUATED=true
ADAPTIVE_PARENT_PROBE_COVER_PASS=true
ADAPTIVE_COVER_LEAF_CELLS=331
ADAPTIVE_PARENT_CERTIFICATE_COVER_PASS=false
V7_B_ELIGIBILITY=false
OPEN_PROBLEM_SOLVED=false
```

This closes the depth-localization question at probe level: the 25 rejected
depth-4 cells are each partitioned into four passing depth-5 cells. Together
with the unchanged 231 passing depth-4 cells, they form a disjoint adaptive
probe cover of the original target parent.

The remaining obstruction is not spatial coverage. Every one of the 200 new
attempts reports certified event charts and a valid homogeneous computation,
but also `C1_ORIENTATION_UNRESOLVED=true`,
`C2_HULL_ORIENTATION_UNRESOLVED=true`, and `CERTIFICATE_PASS=false`. The next
scientific target is therefore the orientation/determinant enclosure, not
further blind spatial refinement.

## Evidence binding

The cluster and local verification outputs are byte-identical. The returned raw
tar has 8,017,920 bytes and SHA-256
`a20da8b62a53ce61c2310d557f8859674942637cfc8a01f1cda269e30646cd97`.
The complete archive and compact ledgers are retained under
`scripts/research/receipts/cs6_v7b_target23_depth5_boundary_refine_v1/`.
