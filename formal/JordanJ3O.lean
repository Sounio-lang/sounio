/-!
# Sounio.JordanJ3O — exceptional Jordan algebra J₃(𝕆) cubic norm (obligations)

Formal companion to `examples/physics/jordan_j3o.sio` (native Sounio, lean_single, all
residuals 0.000000) and `experiments/non_assoc_connectomics/scripts/j3o_foundation.py`
(SymPy, all PASS). Not machine-checked in this environment (no Lean toolchain on the path
used for the empirical work; same caveat as `OctonionAssociator.lean`). The executable
checks are the Sounio program and the SymPy script.

This file states the obligations as **prose**, not as content-bearing axioms — asserting
them via `axiom` would defeat the purpose of separating proof from assumption (the same
discipline used in `DynkinSwapMassLadder.lean` §3). A full Lean development would replace
each with a theorem over `Sounio.OctonionAlgebra`.

## The Freudenthal cubic norm
For a Hermitian J₃(𝕆) element with real diagonal (a,b,c) and octonion off-diagonals
(x,y,z):

  N(X) = a·b·c − a·‖x‖² − b·‖y‖² − c·‖z‖² + 2·Re((z·x)·y)

## Obligations (proved numerically/symbolically; to be discharged in Lean over
`Sounio.OctonionAlgebra`)

1. **Well-definedness.** The cubic term Re((z·x)·y) is unambiguous: Re((z·x)·y) =
   Re(z·(x·y)). (Provable by `ext`/`ring` from the multiplication table; it follows from
   Re(p·q)=Re(q·p) plus the alternative laws.) Verified: `jordan_j3o.sio` line [1] = 0;
   `j3o_foundation.py` `det_well_defined` PASS.

2. **Cyclic (generation) symmetry.** N is invariant under (a,b,c;x,y,z) ↦ (b,c,a;y,z,x) —
   the S₃ that organises the three fermion generations. (Provable by `ext`/`ring`.)
   Verified: `jordan_j3o.sio` line [2] = 0; `j3o_foundation.py` `cyclic_slot_symmetry` PASS.

3. **G₂-invariance.** For any φ ∈ G₂ = Aut(𝕆) applied entrywise to (x,y,z) (diagonal fixed),
   N(φ·X) = N(X). (Provable by `decide` on the finite/explicit automorphism.) Verified for
   φ = (negate e₄..e₇) in `jordan_j3o.sio` line [3] = 0; `j3o_foundation.py`
   `phi_is_G2_automorphism` + `automorphism_preserves_det` PASS.

These three, together with `Triality.lean`, are the verified foundation
(octonions → J₃(𝕆) → triality → three generations). The remaining open step is the E₆
Dynkin-Z₂ mass-ladder realisation, stated as an obligation in `DynkinSwapMassLadder.lean`.
-/

namespace Sounio.JordanJ3O
end Sounio.JordanJ3O
