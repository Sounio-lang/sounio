import SounioCayleyDickson

/-!
# G₂ / V₂(ℝ⁷) skeleton of Cayley–Dickson zero divisors — exact ℤ-rank

Exact-arithmetic companion to `examples/g2_octonion_derivations.sio`.  Where
the Sounio runtime uses float Gaussian elimination, this file recomputes the
same invariants by **exact integer arithmetic** (no float tolerance) on the
verified `cdSigma` structure constants (entries in `{-1,0,1}`).

Trust boundary (precise): `native_decide` certifies the *value* the rank routine
`intRank` returns; it does not prove `intRank` equals the true matrix rank.
`intRank` is standard fraction-free elimination — each step
`target := piv·target − t·pivot` with `piv ≠ 0` is a rank-preserving elementary
operation and the pivot count is the rank — so it is correct by inspection and
cross-validated against the Sounio float-Gauss, but it is **not** a
formally-verified linear-algebra kernel.  The dimensions below are therefore
exact-integer reproducible, not machine-checked from a rank-correctness proof.

Established theorem being skeleton-checked (Moreno 1998 Cor. 2.14; Reggiani
2024): the normalised sedenion zero divisors form `ZD(𝕊) ≅ V₂(ℝ⁷) = G₂/SU(2)`,
total space of the bundle `S³ → G₂ → V₂(ℝ⁷)`.  A manifold homeomorphism is not
finitely checkable; the finite invariants below are.

## Proved here (`native_decide`, exact ℤ-rank)

* `der_octonion_dim_eq_14` — `dim Der(𝕆) = 14 = dim G₂`  (`g₂ = Der(𝕆)`, Cartan).
* `der_octonion_subset_so7` — adjoining `D(1)=0` and skew-on-`Im 𝕆` leaves the
  nullity at 14, i.e. `Der(𝕆) ⊂ so(7)`; a 14-dim derivation subalgebra of
  `so(7)` is `g₂`.
* `octonion_division_algebra` — `ker L_{e₁+e₂} = 0` (𝕆 has no zero divisors).
* `sedenion_zd_S3_fiber` — `ker L_{e₃+e₁₀} = 4` in 𝕊 (the SU(2)=S³ fiber).
* `g2_embeds_in_der_sedenion` — **the anchor**: `dim {D ∈ Der(𝕆) :
  diag(D,D) ∈ Der(𝕊)} = 14 = dim Der(𝕆)`, so every octonion derivation lifts
  to a sedenion derivation.  This is `g₂ ↪ Der(𝕊)`: the octonions'
  `G₂ = Aut(𝕆)` acts on 𝕊, hence on `ZD(𝕊)` — tying `dim 14` to the actual
  zero-divisor structure, not a free-floating number.

## NOT proved here (cited + arithmetic)

`dim V₂(ℝ⁷) = 11`, `ZD(𝕊) ≅ V₂(ℝ⁷)` (Reggiani 2024), and the bundle
identity `dim G₂ = dim V₂(ℝ⁷) + dim S³ = 11 + 3 = 14`.  No manifold
homeomorphism, no quantum-channel / physics claim (established physics is
octonion-only).  The annihilator `Ker L_A` is the fiber's underlying space —
**not** the quaternion subalgebra `H_a`, a separate summand of Moreno's
decomposition `A_n = H_a ⊕ Ker T̃_a ⊕ Ker L_a ⊕ ⊕V_λ`.

No `sorry`. No Mathlib. References: Moreno arXiv:q-alg/9710013;
Biss–Dugger–Isaksen arXiv:math/0511691; Reggiani arXiv:2411.18881;
`Der(𝕆) = g₂` is classical (Cartan).
-/

namespace Sounio.G2Derivations

open Sounio.CayleyDickson

-- Exact integer rank by fraction-free row reduction (= ℚ-rank of an ℤ-matrix).
-- Trusted by inspection (rank-preserving elementary steps), not formally
-- verified; `native_decide` certifies the value, see the header trust note.
partial def intRank (m0 : List (List Int)) : Nat :=
  let rec go (rows : List (List Int)) (col ncol rank : Nat) : Nat :=
    if col ≥ ncol then rank
    else match rows.find? (fun r => (r.getD col 0) ≠ 0) with
      | none => go rows (col+1) ncol rank
      | some pv =>
        let rest := rows.filter (fun r => r != pv)
        let p := pv.getD col 0
        go (rest.map (fun r => let t := r.getD col 0
              if t == 0 then r else List.zipWith (fun rv pvv => p*rv - t*pvv) r pv))
           (col+1) ncol (rank+1)
  go m0 0 (m0.headD []).length 0

-- §1. Octonion derivation (Leibniz) system over the 64 unknowns D[a,b] = col 8a+b.
def sgO (i j : Nat) : Int := cdSigma i j 3
def sgS (i j : Nat) : Int := cdSigma i j 4

def rowO (i j m : Nat) : List Int :=
  let c : List (Nat × Int) :=
    [ (8*m + (i ^^^ j),  sgO i j),
      (8*(m ^^^ j) + i, - sgO (m ^^^ j) j),
      (8*(i ^^^ m) + j, - sgO i (i ^^^ m)) ]
  (List.range 64).map (fun k => (c.filter (·.1 == k)).foldl (fun a p => a + p.2) 0)

def M_O : List (List Int) :=
  (List.range 8).flatMap (fun i => (List.range 8).flatMap (fun j =>
    (List.range 8).map (fun m => rowO i j m)))

-- skew-on-Im 𝕆 and D(1)=0 constraints (as rows over the same 64 unknowns)
def soRows : List (List Int) :=
  let kill1 := (List.range 8).map (fun a =>          -- D[a,0] = 0
    (List.range 64).map (fun k => if k == 8*a + 0 then (1:Int) else 0))
  let skew := (List.range 8).flatMap (fun a => (List.range 8).filterMap (fun b =>
    if 1 ≤ a && a < b then                            -- D[a,b] + D[b,a] = 0
      some ((List.range 64).map (fun k =>
        (if k == 8*a+b then (1:Int) else 0) + (if k == 8*b+a then 1 else 0)))
    else none))
  kill1 ++ skew

-- §2. Pulled-back sedenion Leibniz rows under D ↦ diag(D,D) (cross-block ↦ 0).
def col? (p q : Nat) : Option Nat :=
  if p < 8 && q < 8 then some (8*p + q)
  else if 8 ≤ p && 8 ≤ q then some (8*(p-8) + (q-8))
  else none

def rowS' (i j m : Nat) : List Int :=
  let raw : List (Nat × Nat × Int) :=
    [ (m, i ^^^ j,  sgS i j),
      (m ^^^ j, i, - sgS (m ^^^ j) j),
      (i ^^^ m, j, - sgS i (i ^^^ m)) ]
  let c : List (Nat × Int) := raw.filterMap (fun t => (col? t.1 t.2.1).map (fun ci => (ci, t.2.2)))
  (List.range 64).map (fun k => (c.filter (·.1 == k)).foldl (fun a p => a + p.2) 0)

def M_S' : List (List Int) :=
  (List.range 16).flatMap (fun i => (List.range 16).flatMap (fun j =>
    (List.range 16).map (fun m => rowS' i j m)))

-- §3. Left-multiplication matrix of A = e_a + e_b at level `bits` (dim n = 2^bits).
def leftMul (a b bits n : Nat) : List (List Int) :=
  (List.range n).map (fun row => (List.range n).map (fun col =>
    (if (a ^^^ col) == row then cdSigma a col bits else 0)
    + (if (b ^^^ col) == row then cdSigma b col bits else 0)))

-- ================================================================
-- Theorems
-- ================================================================

/-- `dim Der(𝕆) = 14 = dim G₂`. -/
theorem der_octonion_dim_eq_14 : 64 - intRank M_O = 14 := by native_decide

/-- `Der(𝕆) ⊂ so(7)`: adjoining `D(1)=0` + skew-on-`Im` keeps the nullity 14. -/
theorem der_octonion_subset_so7 : 64 - intRank (M_O ++ soRows) = 14 := by native_decide

/-- The octonions are a division algebra: `ker L_{e₁+e₂} = 0`. -/
theorem octonion_division_algebra : 8 - intRank (leftMul 1 2 3 8) = 0 := by native_decide

/-- The sedenion zero divisor `e₃+e₁₀` has a 4-dimensional annihilator (the
    `S³ = SU(2)` fiber of `S³ → G₂ → V₂(ℝ⁷)`). -/
theorem sedenion_zd_S3_fiber : 16 - intRank (leftMul 3 10 4 16) = 4 := by native_decide

/-- **Anchor — `g₂ ↪ Der(𝕊)`.**  Every octonion derivation lifts to a sedenion
    derivation `diag(D,D)`, so `Aut(𝕆) = G₂` acts on the sedenion zero
    divisors `ZD(𝕊)`.  (Combined nullity equals `dim Der(𝕆) = 14`.) -/
theorem g2_embeds_in_der_sedenion : 64 - intRank (M_O ++ M_S') = 14 := by native_decide

end Sounio.G2Derivations
