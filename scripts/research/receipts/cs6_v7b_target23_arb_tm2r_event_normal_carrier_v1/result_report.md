# Event-normal carrier preflight

## Construction

The carrier preserves the complete degree-2 Taylor model in the six primary
variables `xi`, `eta`, and `rho0` through `rho3`. Four independent coordinates
`sigma0` through `sigma3` are appended for the dynamically reconditioned
remainder. The first carrier column is transverse to the exact rational event
covector `(mid(y), mid(x), -1, 0)`; the other three columns lie exactly in its
kernel. The basis inverse and kernel pairings are checked with exact rational
arithmetic, and every projected Arb generator must reconstruct an enclosure of
the original generator.

This gives an inductive enclosure invariant. A flow step may create mixed or
quadratic `sigma` monomials; the next reconditioning ranges those monomials on
the normalized cube, adds their enclosures to the generator family, and restores
the primary-TM2-plus-linear-carrier normal form with zero component remainder.
This is deliberately an enclosure operation, not an algebraic identity.

## Preflight result

The frozen raw event projection was replayed exactly and the carrier was
installed before one new bounded flow step. Both candidates strictly reduced
the event-derivative width relative to the lineage-preserving control:

| mode | control width | carrier width | improvement factor |
| --- | ---: | ---: | ---: |
| event-normal doubleton | 9.704881887288939e-05 | 9.704823469292023e-05 | 1.0000060194806326 |
| event-normal tripleton | 9.704881887288939e-05 | 9.704823457923340e-05 | 1.0000060206520864 |

The tripleton is the first full-transport candidate because it is marginally
better. The reduction is only about 0.000602 percent, far below the operational
target of 18 times. This preflight therefore validates the implementation and
candidate ordering; it does not validate the quantitative hypothesis.

The strict width margins are computed from the exact rational receipt
endpoints. They exceed the verifier's `2^-230` endpoint-rounding allowance, so
the sign of the tiny comparison is certified. Its small size still makes it a
weak practical signal and not evidence that the full transport will succeed.

Applying the carrier after the production or terminal remainder was already
flattened into component boxes made the derivative enclosure worse. That path
is retained only as a negative control: post-hoc reconditioning cannot recover
correlation that an earlier box projection erased.

## Verification

- All implementation checks passed.
- Both exact generator-reconstruction certificates passed.
- Every reconditioning restored the certified carrier normal form.
- Maximum exact basis-inverse row sum was approximately 1.068125.
- The independent verifier accepted the receipt.
- All 35 receipt mutations were rejected.
- Local stderr was empty.

The mandatory hostile math review initially found that the inductive carrier
invariant was implicit rather than certified. The invariant, verifier, and
mutations were strengthened; Grok's closure review then returned `PASS`. Z.AI
was unavailable because its weekly limit was exhausted, and Qwen and Mistral
fallbacks returned credit errors, so the review is recorded as single-provider
degraded rather than dual-provider closure.

## Decision

Run the tripleton transport from the frozen witness to the refused event on
Foundry/Slurm. Run the doubleton second as a control if the allocation permits.
The event gate remains strict whole-tube transversality, with an 18-times
directional reduction as the target. No full transport was attempted in the
workspace pod.

This receipt certifies no section crossing, covering relation, recurrent graph,
chaos result, priority claim, or solution of an open problem.
