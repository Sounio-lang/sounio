# Pre-QR next-return transport result

## Result

The accepted raw pre-QR XLEL carrier does not yield a certified complete next
upward return under the frozen lineage-preserving TM2R method and its audited
depth-8, 255-node subdivision budget.

Foundry job 9492 completed on `gpuorangefs-r770-proxmox` in 00:16:13. The
independent verifier classified the receipt as
`PRERECOND_NEXT_RETURN_REFUSED`; all implementation checks passed and all 22
negative mutations were rejected.

The executed source is Git commit `80442bb0d6`. The computation is
deterministic and uses no random seed. Its evidence kernel is Arb interval
arithmetic plus the checked Python TM2R algorithms and an independent
`Fraction`-based receipt verifier; this report makes no Lean proof claim.

This is a refusal of the attempted certificate, not evidence that the true
dynamics lacks the return.

## What survived

- The frozen predictor-centered preconditioned event replayed at residual power
  12.
- The raw projected carrier remained exactly on `w=0`.
- At the accepted raw projection before the attempted next return, aggregate
  dependence on all six original variables
  `xi, eta, rho0, rho1, rho2, rho3` was positive.
- No QR renumbering, point fallback, or box flattening was used.
- The exact six-variable split lineage and every rational parent/child domain
  were retained and independently reconstructed.

The refusal was classified as orientation ambiguity, not by the worker's
terminal symbolic-dependence-loss classifier. Because no terminal carrier was
accepted, preservation of all six variables at the refused event remains
unassessed.

An unsplit variable can retain positive polynomial dependence because split
selection chooses only the largest current width contribution. Thus `XI=0`
and `RHO3=0` in the later split counts do not contradict the positive weights
measured at the raw projection; neither do those counts prove preservation at
the refused event.

## First unresolved witness

The new transport split path was:

```text
RHO1L -> RHO0L -> RHO1L -> ETAL -> RHO0L -> RHO2L -> RHO1L -> ETAL
```

At depth 8, `seek_upward_return` refused with:

```text
failure_class=SECOND_PRIOR_ORIENTATION_UNRESOLVED
detail=a pre-target tube remained section-ambiguous
```

The retained normalized symbolic domain is:

| variable | exact bounds |
|---|---|
| `xi` | `[-1, 0]` |
| `eta` | `[-13/32, -51/128]` |
| `rho0` | `[-1, -63/64]` |
| `rho1` | `[1/2, 9/16]` |
| `rho2` | `[-1, 0]` |
| `rho3` | `[-1, 1]` |

There were eight split nodes with counts `RHO1=3`, `RHO0=2`, `ETA=2`,
`RHO2=1`, `XI=0`, and `RHO3=0`. No terminal event carrier was accepted, so no
terminal domain cover or aggregate event hull was emitted.

## Exhaustive precursor

Job 9491 used the original enumerate-all refusal policy. It reproduced the same
failure class through 27 recorded split decisions over 00:58:32. Once the first
depth-8 witness had made a complete cover impossible, its partial log was
retained and the job was cancelled so job 9492 could produce a closed,
fail-fast receipt on the same node.

## Consequence

This experiment does not provide the requested complete next return. Therefore
within the planned construction it cannot define the next target h-set,
certify a second covering edge, or close a recurrent graph. No abstract
impossibility theorem about covering relations, recurrence, or chaos is claimed.

## Next experiment

Freeze the exact witness domain above and instrument the final pre-target phase
locally. Retain the last strictly negative `w` endpoint, the first ambiguous
tube, its full section derivative, and both candidate reference times. Then run
a witness-local interval-Newton event projection with adaptive time bisection
while preserving the same six symbolic variables.

This is a proposed fail-closed diagnostic, not a claimed termination or
soundness theorem: if its interval inclusions or sign tests do not close, it
must retain another refusal rather than accept a return.

The discriminator is:

1. strict negative departure plus a positive endpoint exists, but the current
   phase criterion refuses: repair the event criterion;
2. the endpoint sign remains ambiguous after local time refinement: the carrier
   enclosure still needs reconditioning or a narrower witness split;
3. the derivative loses strict positivity: the obstruction is transversality,
   not event bookkeeping.

Do not launch another whole-domain transport until this witness-local test
chooses among those three cases.
