# Math review request: CS6 TM2R composability falsifier

Review the new files:

- `scripts/research/cs6_v7b_target23_arb_tm2r_composability_carrier_worker.py`
- `scripts/research/cs6_v7b_target23_arb_tm2r_composability_contract_v1.txt`
- `scripts/research/receipts/cs6_v7b_target23_arb_tm2r_composability_v1/result_report.md`

The source h-set uses the chart

```
U = nx*x + ny*y
S = -ny*x + nx*y
```

with rational `nx,ny`, centers `Uc,Sc`, and radii `Ur,Sr`.  The worker
reconstructs its affine TM2R state using determinant `nx^2+ny^2`:

```
x0 = (nx*Uc - ny*Sc)/det
y0 = (ny*Uc + nx*Sc)/det
x_xi = nx*Ur/det       y_xi = ny*Ur/det
x_eta = -ny*Sr/det     y_eta = nx*Sr/det
```

It then transports four source tiles through the same Poincare map `P^2`.
The first positive-return horizon is enlarged from 2000 to 2600 fixed steps.
The local event-slab search adds radius `1/128` after the old maximum `1/256`.
Every slab is still accepted only when:

1. backward and forward signed Picard boxes close;
2. the event derivative has the required strict sign;
3. the parametric predictor is strictly inside the slab;
4. the interval-Newton-corrected event time is strictly inside the slab.

When a downward Newton projection or upward pre-target orientation test refuses,
the worker splits the pre-event TM2R state in the dominant source/residual
variable, reconstructs both children, preserves the corresponding global
`xi,eta` subdomain, and retries.  It emits a tile receipt only after every child
has completed.

The partial local run was interrupted under the heavy-validation policy after
five subbranches completed, producing ten internal carriers.  No JSON tile
receipt was emitted.  The report therefore states that neither complete tile
support, `B -> C`, recurrence, nor chaos is certified.

Please identify any BLOCKER or MAJOR issue, especially:

1. a sign or inverse error in the source chart;
2. any reason the larger slab could weaken rigor despite the strict Picard and
   interval-Newton tests;
3. any loss of coverage caused by splitting the pre-event state using a
   variable selected from the post-projection carrier;
4. any overclaim in the partial-run report;
5. any provenance field needed before these receipts can support a covering
   relation.
