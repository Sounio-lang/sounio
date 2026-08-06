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

## Foundry/Slurm execution

The four base tiles were launched on the three `gpu-orangefs` nodes from Git
commit `8e00e0a7dc3b170ade3a74199c063708c56d858e`.  No atomic tile receipt was
emitted.

- `XLEL` failed after `02:54:13` at the exact split-depth cap `8`.
- `XLEH` and the slow `XHEL` run reached the six-hour operational timeout.
- A deterministic `XHEL` duplicate on the r770 node failed at the same base
  split budget after `02:54:34`.
- The base `XHEH` job was replaced after 32 minutes by the audited retry profile.

The retry profile at commit
`8674961617d4686ede9e2230a7bfe870e5f23e33` changed only the explicit search
budget from depth `8`, 63 nodes to depth `12`, 255 nodes.  It retained the same
TM2R worker, exact binary substitutions, signed Picard closure, strict event
derivative, strict interval-Newton containment, point-fallback refusal, and
box-flattening refusal.  Its operational Slurm limit was extended to 24 hours;
the mathematical budget did not change.

`XLEL` crossed the former failing depth.  One depth-10 sibling certified its
second event, but the remaining child continued to return `PREDICTOR_ESCAPED`
for every tested slab radius from `2^-18` through `2^-7`.  It reached depth 12
on the branch

```
DOWN_RHO0L/DOWN_ETAH/DOWN_RHO0L/DOWN_ETAL/
DOWN_RHO0L/DOWN_ETAL/DOWN_RHO0L/DOWN_RHO1H/
DOWN_ETAH/DOWN_RHO0L/DOWN_RHO1H/DOWN_ETAH
```

and failed closed with `EVENT_SLAB_UNRESOLVED` after `06:17:31`.  The other
retry jobs were then cancelled because no completion of those tiles could
repair the missing `XLEL` support.

The exact local refusal receipts are
`foundry_XLEL_base.incomplete.stderr.txt` and
`foundry_XLEL_retry.incomplete.stderr.txt`.  Their SHA-256 hashes and the full
Slurm accounting are bound by `foundry_execution_context.txt`; the staged
archive and retry hashes are retained under `foundry_provenance/`.

## Foundry verdict

- Complete atomic tile receipts: **0 of 4**.
- Full rigorous support of `P^2(B)`: **not certified**.
- Target h-set `C` from full support: **not derivable**.
- Exit-face inequalities for `C`: **not certified**.
- Entry inequalities for `C`: **not certified**.
- Covering degree and return determinant for `B -> C`: **not certified**.
- Local covering relation `B -> C`: **false as a current claim**.
- Recurrent covering graph and chaos: **not proved**.

The fail-closed analyzer, verifier, mutation audit, and gate for a future full
support family are implemented but deliberately cannot run without all four
atomic JSON receipts.  Increasing Cartesian split depth again is not the next
credible attack: the quantitative failure persisted from depth 8 through 12.
The next experiment should discriminate accumulated event-chart drift, the
event criterion, and an implementation defect.  One candidate is event-local
chart reanchoring, with QR justified only under the hypothesis that frame drift
causes the offset; an alternative is a Taylor-model event chart that directly
tests validated predictor containment.  Preserving all six symbolic variables
is a design requirement on either falsifier, not a consequence of this failure
evidence.
