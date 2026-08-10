[WRONG] "reconstructed carrier enclosure to contain every original generator"
  Arb interval reconstruction after exact rational projection on a degree-2 TM cannot guarantee enclosure of all generators once sigma-bearing monomials are reconditioned; the carrier variables are introduced as independent, breaking the original quadratic dependencies.

[OVERREACH] "exact rational inverse" of coefficient radii and remainders
  Radii and remainders are floating-point intervals; an exact rational inverse exists only for the nominal point, not the enclosure. The subsequent Arb containment check therefore rests on an unproven interval extension of that inverse.

[WRONG] "first basis column transverse … remaining three columns lie exactly in its kernel"
  The four carrier directions are asserted to satisfy an exact linear condition on the event covector, yet the quadratic monomials already present in the original six variables are not re-projected; the claimed kernel membership therefore fails for the full TM2R object.

[WRONG] one-step width reduction (factors ≈1.000006)
  The observed contraction is smaller than the unit roundoff of the Arb precision used; the reported “strict” improvement is indistinguishable from asymmetric outward rounding or from the control reconstruction discarding a single dependence that the carrier version retains by construction.

[BLOCKER] "preserving all monomials supported on the original six variables while reconditioning every sigma-bearing monomial"
  After the first reconditioning step the sigma monomials are no longer pure functions of the original variables; subsequent flow steps therefore operate on an algebraically inconsistent object. No inductive invariant is stated that would restore coherence.

[BLOCKER] preflight vs. full Foundry/Slurm transport
  The supplied one-step numeric check supplies neither a rigorous enclosure proof nor a counter-example; it therefore functions only as an additional falsifier. No theorem about the multi-step carrier TM2R follows from it.
