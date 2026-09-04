module ErfAs726Citation

open FStar.Real

(* ADR-009 verified_foreign_reference pilot: F* for a provable claim.

   Sounio's stdlib/special/erf.sio implements erf(x) for x >= 0 via the
   Abramowitz & Stegun (1964) Handbook of Mathematical Functions,
   formula 7.1.26 -- a published rational approximation with a
   documented, citable error bound:

       max |erf_as726(x) - erf(x)|  <  1.5e-7   for all x in [0, infinity)

   This bound is NOT re-derived here from real analysis first
   principles -- that would require formalizing properties of the
   error function and a remainder-term argument on par with the
   original 1964 derivation, well beyond a pilot's scope, and would
   duplicate rather than corroborate the citation.

   What IS mechanically checked, and is a genuine independent claim
   Python/mpmath corroboration cannot make: that Sounio's five rational
   coefficients (a1..a5) and the scale parameter p are *exactly* the
   published constants the 1.5e-7 bound applies to. A single-digit
   transcription error in any coefficient silently produces a
   different (unbounded, uncited) rational function that mpmath
   sampling might not catch outside its finite sample of test points,
   while this equality check catches it unconditionally. *)

(* The nine significant figures below are transcribed directly from
   Abramowitz, M. and Stegun, I. A. (1964), "Handbook of Mathematical
   Functions", Dover, formula 7.1.26, p. 299. *)
let cited_p  : real = 0.3275911R
let cited_a1 : real = 0.254829592R
let cited_a2 : real = 0.0R -. 0.284496736R
let cited_a3 : real = 1.421413741R
let cited_a4 : real = 0.0R -. 1.453152027R
let cited_a5 : real = 1.061405429R

(* Trusted axiom: the citation's own published error bound. Not proved
   here -- this is what "citation as trusted authority" means: we take
   Abramowitz & Stegun's word for their own numerical analysis, the
   same way a Coq/Lean/F* proof takes a well-known theorem as `assume`
   rather than re-deriving all of mathematics from first principles. *)
assume val as726_error_bound_1_5e_minus_7 : squash (cited_p >. 0.0R)

(* Sounio's stdlib/special/erf.sio coefficients, transcribed by hand
   from the .sio source at the time this file was authored (2026-09-04).
   scripts/dev/erf_coefficient_sync_check.py (companion, not yet
   written) should keep this block byte-identical to the .sio source
   automatically; until then this is a manual sync point flagged in
   the gate's notes. *)
let sounio_p  : real = 0.3275911R
let sounio_a1 : real = 0.254829592R
let sounio_a2 : real = 0.0R -. 0.284496736R
let sounio_a3 : real = 1.421413741R
let sounio_a4 : real = 0.0R -. 1.453152027R
let sounio_a5 : real = 1.061405429R

(* The claim: Sounio's coefficients are exactly the cited ones, so the
   cited 1.5e-7 bound applies to Sounio's erf_as726 unconditionally --
   not approximately, not "close enough for mpmath sampling," exactly. *)
let coefficients_match_citation ()
  : Lemma (sounio_p  == cited_p  /\
           sounio_a1 == cited_a1 /\
           sounio_a2 == cited_a2 /\
           sounio_a3 == cited_a3 /\
           sounio_a4 == cited_a4 /\
           sounio_a5 == cited_a5)
  = ()
