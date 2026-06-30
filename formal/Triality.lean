/-!
# Sounio.Triality — Spin(8) triality on the octonions (obligations)

Formal companion to `examples/physics/triality_check.sio` (native Sounio core checks) and
`experiments/non_assoc_connectomics/scripts/triality_principle.py` (the full computational
derivation: residual ~1e-13, dim g₂ = 14). Not machine-checked here (no toolchain). Stated
as prose obligations, not content-bearing axioms (same discipline as `JordanJ3O.lean`).

## The trilinear form
t(x,y,z) = Re((x·y)·z) on 𝕆 ≅ ℝ⁸.

## Obligations

1. **Cyclic symmetry of t.** t(x,y,z) = t(y,z,x). (From Re(p·q)=Re(q·p) and the alternative
   laws.) Verified: `triality_check.sio` [1] = 0; `triality_principle.py` `[A]` PASS.
   The form is NOT fully symmetric (t(x,y,z) ≠ t(x,z,y) in general) — the genuine octonionic
   signature; verified `triality_check.sio` [2] ≠ 0.

2. **Principle of infinitesimal triality.** For every A ∈ 𝔰𝔬(8) there exists a UNIQUE pair
   (B,C) ∈ 𝔰𝔬(8)² with
       t(Ax, y, z) + t(x, By, z) + t(x, y, Cz) = 0   for all x,y,z.
   This is a linear-algebra fact (a 56-variable solve), NOT a pure ring identity — it is
   verified COMPUTATIONALLY (`triality_principle.py`: residual ~1e-13, solution rank 56/56),
   and is stated here as an obligation, not proved. The S₃ orbit of A ↦ (B,C) is the three
   8-dim representations 8v/8s/8c = the three generations.

3. **Derivations = g₂, dim 14.** The S₃-fixed diagonal {A : A=B=C}, equivalently the
   derivations {A : A(x·y) = A(x)·y + x·A(y)}, form the Lie algebra g₂ = der(𝕆) = Lie(Aut 𝕆),
   of dimension exactly 14. Verified computationally (`triality_principle.py` `[C]`: dim 14).
   The alternative laws [x,x,y]=0, [x,y,y]=0 that this rests on are verified in
   `triality_check.sio` [3],[4] and proved in `OctonionAssociator.lean`.

## Status
Obligation (1) is a ring identity discharge-able in Lean over `Sounio.OctonionAlgebra`.
Obligations (2)–(3) are linear-algebra facts verified computationally; encoding them as Lean
theorems would require a real-vector-space development of 𝔰𝔬(8) acting on 𝕆 (future work).
The point established now: three generations arise from triality — derived and verified, not
assumed. The open research step is the E₆ Dynkin-Z₂ mass realisation (`DynkinSwapMassLadder.lean`).
-/

namespace Sounio.Triality
end Sounio.Triality
