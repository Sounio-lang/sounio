# Pre-QR witness-local upward-event result

## Result

The exact depth-8 witness remains uncertified after extending only the time
refinement from depth 10 to depth 18. Foundry job 9697 completed on
`gpuorangefs-r770-proxmox` in 00:08:44 and classified the result as
`WITNESS_TRANSVERSALITY_UNRESOLVED`.

This is a local certificate refusal, not evidence that the true trajectory or
the full leaf lacks an upward return. It does discriminate the next method:
additional time bisection is not the useful axis for this witness.

## Controls

- The exact frozen path
  `RHO1L,RHO0L,RHO1L,ETAL,RHO0L,RHO2L,RHO1L,ETAL` replayed.
- The exact six-variable rational domain replayed.
- The production point-coefficient policy was checked before the frozen replay.
- The lineage-preserving policy was checked after raw projection capture and
  before the new witness-local integration.
- The production depth-10 refusal boundary was reproduced.
- All 39 implementation checks passed.
- The independent verifier passed and all 22 negative mutations were rejected.
- No point fallback, box flattening, full transport, h-set construction, or
  covering-relation test was used.

Positive aggregate upper weights for the six original variables at the raw
projection mean represented possible symbolic dependence only. No terminal
event carrier was accepted, so final symbolic preservation at an event remains
unassessed.

## Quantitative discriminator

At the frozen production boundary, the full-tube event derivative was
approximately:

```text
dw/dt in [-297.3597262378533, 340.8649518016975]
time depth = 10
step = 1/262144
```

At the final diagnostic boundary it was approximately:

```text
dw/dt in [-297.3314596775189, 340.8367761012897]
time depth = 18
step = 1/67108864
```

The derivative width fell by only `0.008843634958699553%`; its terminal width
was `0.999911563650413` times the production width. Time refinement therefore
left essentially all of the obstruction intact.

The terminal derivative has midpoint about `21.75265821188538` and radius
about `319.0841178894043`. With a fixed midpoint, strict positive
transversality requires reducing that radius by more than
`14.668741391572123` times. This is consistent with, but does not independently
prove the sufficiency of, the earlier approximately 18-fold directional-radius
target.

Time refinement did reduce the upper endpoint of the section tube from about
`4.013867308462356e-4` to `7.541529784291522e-7`, but its lower endpoint
remained near `-12.5694`. No strictly positive endpoint was found.

## Newton attempts

The worker made 38 endpoint projection attempts: both candidate endpoints at
every time depth from 0 through 18. Every attempt refused with
`SECOND_EVENT_COVER_UNRESOLVED`; each internal cover exhausted 255 split nodes
and left 256 unresolved leaves with first cause
`UPWARD_PREFILTER_UNRESOLVED`.

This rules out the proposed event-bookkeeping explanation under the tested
extended budget. It does not rule out a narrower spatial subdivision,
derivative-aligned reconditioning, or a different rigorous carrier geometry.

## Initialization failures

Jobs 9682 and 9685 exposed an early recondition-policy error before the new
diagnostic began. Job 9686 showed that the first corrective patch changed the
wrong assignment. Neither job contains a scientific result. Their retained
tracebacks and the final dual policy controls are included in this packet.

## Next experiment

Do not increase the time-refinement depth again. At the production and terminal
candidate states, build the TM2R derivative model

```text
D(xi,eta,rho0,rho1,rho2,rho3) = u*v - w - z_s
```

and emit an exact width budget separated into linear, quadratic, mixed, and
interval-remainder contributions for each original symbolic direction. Apply
one level of exact domain bisection in the dominant derivative direction and
recompute the budget without integrating the full trajectory.

The next falsifier is quantitative: either a targeted split or a transported
QR/doubleton direction reduces the derivative radius toward the required
factor greater than 14.6687, or the residual interval term remains dominant
and the carrier representation, rather than the event criterion, must change.

No complete next return, h-set, covering relation, recurrent graph, chaos, or
open-problem solution is claimed.
