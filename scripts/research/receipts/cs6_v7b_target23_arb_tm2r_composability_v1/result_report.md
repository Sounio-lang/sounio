# CS6 TM2R composability falsifier v1

## Question

Can the certified first target h-set `B` be transported by the same map `P^2`
to a second rigorously certified h-set `C`, preserving the exit, entry, degree,
and determinant conditions needed for a composable covering relation?

## Source h-set

`B` is reconstructed from the exact rational chart and radii emitted by
`cs6_v7b_target23_arb_tm2r_covector_qr_v1`.  The worker uses a degree-2 Taylor
model with source variables `xi,eta` and four explicit residual variables.  It
does not replace the carrier by an interval box.

The nonrigorous center scout places `P^2(B)` near UPO coordinates
`(u,s)=(9.573241893902707,-0.4549345840649371)`.  This is only a locator for a
prospective node `C`; it is not used as proof data.

## Rigorous local run

The `XLEL` source tile certified its first positive return and interval-Newton
projection at step `2191`.  The original `2000`-step horizon was therefore a
strictly temporal refusal, not loss of source dependence.

The downward event needed a slab larger than the former maximum `1/256`.
The experiment tries `1/128` as the next larger alternative, not as an additive
margin, but retains signed Picard closure, strict event derivative sign, and
strict interval-Newton containment.  The root carrier then needed a residual
split in `RHO0`.

The upward search subsequently split in `RHO0` and `ETA` under
`SECOND_PRIOR_ORIENTATION_UNRESOLVED`.  Before the local run was stopped under
the repository's heavy-validation policy, five of the still-incomplete family
of subbranches completed the second event projection and produced ten
provisional internal carriers.  The log is `partial_XLEL.stderr.txt`; its exact
pre-domain-audit worker hash and interrupted exit status are bound separately in
`partial_execution_context.txt`.

These provisional objects are not standalone carrier receipts: the worker emits
JSON atomically only after its entire source tile closes.  Therefore they do
not certify even the `XLEL` support by themselves.

Every future complete receipt records bounds for all six symbolic variables,
not only `xi,eta`.  Each child is checked as an exact rational half-partition of
its parent, and the terminal split lineages must recursively form a complete
binary cover of the source tile before JSON emission.

## Verdict

- `P^2` second passage exists for certified subbranches: **observed rigorously**.
- Complete `XLEL` source-tile support: **not yet certified**.
- Complete four-tile support of `B`: **not yet certified**.
- Target h-set `C`: **not yet constructed from rigorous full support**.
- Covering relation `B -> C`: **false as a current claim**.
- Recurrent covering graph: **false as a current claim**.
- Chaos: **not proved**.

The quantitative blocker is residual wrapping in the pre-target orientation
test.  Cartesian splitting does make progress, but the full tree is a heavy
validation job.  The next execution should run each source tile independently
through `cs6_v7b_target23_arb_tm2r_composability_run.sh` on the Compiler Foundry
or Slurm path.  If the tree reaches its depth/node cap, the next mathematical
change is QR transport/reanchoring of the residual basis at the downward event,
not a relaxation of the orientation inequality.
