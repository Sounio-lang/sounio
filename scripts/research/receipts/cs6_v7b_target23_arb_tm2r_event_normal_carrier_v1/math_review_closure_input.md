# Closure review: event-normal TM2R carrier

Re-review the revised files listed in `math_review_input.md`. The initial Grok
review raised two BLOCKERs and four other objections. The implementation and
certificate now expose the following obligations explicitly.

## Inductive enclosure invariant

After every reconditioning, each component contains only:

1. degree-at-most-two monomials supported on the original six primary
   variables;
2. four pure linear carrier monomials `sigma0` through `sigma3`;
3. zero component remainder.

A flow step may create arbitrary degree-at-most-two sigma-bearing monomials and
an interval remainder. At the next reconditioning:

- each odd or mixed normalized monomial is enclosed by `[-1,1]`;
- each all-even nonconstant normalized monomial is enclosed by `[0,1]`, encoded
  as one-half constant plus one-half symmetric generator;
- each coefficient radius is an independent axis generator;
- each remainder midpoint enters the constant term and each remainder radius is
  an independent axis generator.

All sigma-bearing dependencies may therefore be discarded, which widens the
set but does not invalidate enclosure. The worker now refuses unless every
reconditioning restores the normal form. The receipt records one normal-form
check per reconditioning, and the independent verifier rejects mixed carrier
monomials, nonlinear carrier monomials, nonzero remainders, or incorrect counts.

## Rational basis and interval action

The basis and inverse are matrices of exact `Fraction` values. The generator
coordinates are evaluated in Arb using rational balls. The verifier checks the
exact rational identity `Q * Q^-1 = I`; the worker separately checks in Arb that
the reconstructed interval vector contains every original interval generator.
No exact inverse of an interval matrix is claimed.

Kernel membership is claimed only for the four columns of this midpoint event
covector basis. It is not claimed for the full nonlinear Taylor model, and it
is not a transversality certificate.

## Strict comparison

The one-step widths are serialized as exact rational endpoint bounds. Their
positive margins are approximately `5.84e-10`. The verifier's allowed endpoint
rounding discrepancy is `2^-230`; the margins are more than `10^60` times that
allowance. Thus the strict ordering of the two computed enclosures is
certified, although the small contraction remains a weak engineering signal.

## Claim boundary

The initial review's final statement is accepted: the preflight is only a
falsifier and candidate selector. It proves no multi-step transversality,
section crossing, covering relation, recurrence, chaos, or open-problem result.
Those claims remain false in the receipt.

Return `PASS`, `BLOCKER`, or `MAJOR`. In particular, decide whether the explicit
normal-form restoration and generator hull give a sound inductive enclosure
for full transport. Do not require preservation of sigma dependencies; this
method intentionally replaces them by a larger independent generator hull.
