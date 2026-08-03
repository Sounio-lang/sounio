# Target-23 independent Decimal center-orbit replay

**Date:** 2026-08-03

**Executed commit:** `84d16dd2c283cc0ba49d7c09332e33a30514c8e3`

**Slurm job:** `8535` (`COMPLETED`, 12 seconds, 32 CPU workers)

**Verdict:** bounded pointwise falsification scout completed; no interval or
global certificate follows.

## Question

The retained CAPD experiment gives strict-negative Liouville determinant
enclosures for two carriers on a frozen adaptive cover of 331 target-23 leaves.
Could a separately authored numerical route quickly expose a sign, scale,
section, orbit, or transcription error at the center of any leaf?

This replay was frozen before execution as a falsification experiment. It was
not designed to turn non-rigorous trajectories into a proof.

## Independent numerical route

The worker uses only Python's standard-library `Decimal` arithmetic and a
fixed-step classical RK4 integrator. It does not import CAPD and is not given
the CAPD result table. For each frozen dyadic leaf center it integrates the
three-dimensional flow plus the auxiliary divergence integral, localizes the
first two negative-to-nonnegative `w=0` returns with 48 bisections, and computes
the oriented pointwise Poincare determinant

```text
exp(ell(T)) * normal_velocity(0) / normal_velocity(T) * det(DQ0).
```

The coarse route uses 50 Decimal digits and step `2^-9`; the fine route uses 80
digits and step `2^-10`. The runner compares their point values with both
pre-existing CAPD Liouville enclosures only after the worker exits. A separate
verifier reconstructs the frozen plan, source center, initial normal velocity,
oriented source-frame area, Liouville formula, raw hashes, commands, challenges,
and Slurm provenance.

## Result

- `331/331` center orbits completed at both resolutions.
- `331/331` computed fine determinant estimates were negative at the tested
  precision.
- In exact comparisons of their emitted Decimal values, both coarse and fine
  point estimates lay strictly inside both retained CAPD Liouville enclosures
  on `331/331` leaves.
- The maximum absolute coarse/fine discrepancy was
  `1.6698218812420758892472547931309744777e-18`.
- The smallest fine-point distance to any CAPD endpoint was
  `5.070949335831988725619300086e-15`.
- The largest discrepancy was about `0.0001646` of the corresponding CAPD
  enclosure width.
- Cluster and local verification outputs were byte-identical.
- The verifier rejected `14/14` deliberate mutations covering summary counts
  and verdicts, challenge and attempt bindings, determinant and resolution
  values, containment flags, row deletion, raw stdout, Slurm identity, runtime
  independence, source commit, and forbidden scientific claims.

The observed fine determinants ranged from approximately
`-2.84248640765454e-11` to `-2.83188908986135e-11`. No center trajectory
contradicted the retained CAPD sign or enclosure at the tested resolutions.
The coarse/fine discrepancy is a self-consistency diagnostic, not an RK4 error
bound. Likewise, the endpoint margin is conditional on the retained CAPD
outputs; it neither validates those endpoints nor encloses the true point value.
The exact CAPD endpoints used for every comparison are retained in each row of
`results.tsv` and bound by the execution hashes.

## Failed attempts retained

Job `8533` stopped before any orbit because the Slurm Python build exposed an
empty `sys.executable`. Job `8534` computed 331 runner-pass rows but the frozen
verifier rejected textual `/usr/bin/python3` versus `/usr/bin/python3.12`
provenance. Neither attempt is counted as scientific evidence. Both failure
classifications are retained; job `8535` reran the complete experiment after
the portability repairs.

## What this establishes

This is independent implementation diversity at the level of point-orbit
integration and formula reconstruction. Across every frozen leaf center, it
failed to falsify the retained CAPD Liouville enclosures and found close
coarse/fine numerical agreement.

That is useful evidence against simple implementation and transcription bugs.
It is stronger than replaying the same CAPD executable under new audit bindings.

## What this does not establish

Decimal arithmetic plus RK4 supplies no rigorous truncation-error enclosure.
A computed point lying inside a CAPD interval does not certify the point's true
trajectory or prove the CAPD interval. This is not an independently implemented
interval engine, a leaf-wide proof, a global H-PG certificate, V7-B eligibility,
a carrier winner, novelty, promotion evidence, or a solution of an open problem.
No FPGA was used.

## Next falsification window

The complementary next check is not another precision increase of the same RK4
route. It is an independently implemented validated integrator, preferably with
Taylor models or interval Taylor integration, run first on the leaf with the
smallest observed CAPD margin and then on the full 331-leaf cover. The present
scout targets simple point-orbit and transcription errors; that experiment
would target implementation diversity at the enclosure level.

## Durable evidence

The receipt directory is
`scripts/research/receipts/cs6_v7b_target23_decimal_center_replay_v1/`.
It retains the complete compressed Slurm result, all 331 raw worker transcripts,
the exact result table, cluster and local verifier outputs, mutation results,
failed-attempt records, execution hashes, and pre-execution LLM reviews.
Those reviews drove explicit divergence and section conventions, source-frame
reconstruction, strict field and containment checks, and the tighter
computed-value wording above; disagreements and unavailable-provider legs are
retained alongside the reviews.
