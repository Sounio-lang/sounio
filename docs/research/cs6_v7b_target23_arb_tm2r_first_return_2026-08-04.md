# CS6 target-23 Arb TM2R: rigorous first full-leaf return

**Date:** 2026-08-04

**Status:** the complete critical leaf 331 reaches one rigorously bracketed,
strictly transverse negative-to-positive `w=0` return under an independent Arb
TM2 carrier with QR-derived residual transport. This is a first-return result
only. It is not a second-return, determinant, covering-relation, chaos,
attractor, novelty, or open-problem certificate.

## Carrier

The carrier keeps two fixed source variables for the exact leaf coordinates
and four residual variables for transported uncertainty. All polynomials have
total degree at most two. After every validated time step:

1. monomials involving residual variables become vector generators;
2. even residual monomials use a half-center plus half-generator enclosure;
3. interval remainders become independent axis generators;
4. generator midpoint vectors determine a Gram-Schmidt QR-derived basis;
5. the floating basis is frozen as exact decimal rationals and inverted by
   exact `Fraction` Gauss-Jordan elimination;
6. every generator is reconstructed with outward-rounded Arb `Q*Q^-1` before
   its coordinate radii enter the new zonotope.

Pure `xi,eta` monomials remain explicit throughout. No CAPD value, sampled
point, center-orbit replacement, or FPGA result enters the worker.

## Retained result

With Arb 0.8.0 at 256 bits, time Taylor order 12, and step `2^-8`:

```text
completed steps                 617
Picard containments             617
endpoint-in-Picard containments 617
reconditionings                 617
generator reconstructions       15810
initial departure tubes         1
prior downward crossing tubes   1
prior section-free tubes        614
first-return bracket            [616/256, 617/256]
events validated                1
```

The complete pre-step `w` interval is strictly negative, the complete
post-step interval is strictly positive, and both `x*y-zs` and the full
derivative `w'=x*y-w-zs` are strictly positive on the Picard event tube. The
strict derivative makes the target-step zero unique.

All 616 earlier step tubes are also classified: 614 exclude `w=0`, one is the
strictly positive initial departure from the section, and one has strictly
negative `w'` throughout and is therefore the downward crossing. Consequently
there is no earlier positive return hidden between mesh points, closing the
first-return objection raised by the initial independent review.

## Mathematical boundary

The first-return time is bracketed by a full step. No interval Newton refinement
of the event time or section projection is claimed. No second return has been
transported, so no return-map determinant or covering relation follows from
this artifact.

The next falsifier is to continue from the reconditioned post-event carrier,
with an interval-Newton event projection that preserves the two source
variables, until a second strictly transverse return is enclosed.

```text
FULL_LEAF_FIRST_RETURN_CERTIFICATE=true
FULL_LEAF_SECOND_RETURN_CERTIFICATE=false
GLOBAL_HPG_CERTIFICATE=false
V7_B_ELIGIBILITY=false
CHAOS_PROVED=false
CHAOTIC_ATTRACTOR_PROVED=false
OPEN_PROBLEM_SOLVED=false
```
