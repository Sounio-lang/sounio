import SounioCayleyDickson

/-!
# Cayley–Dickson erasure ladder — the `2^(n-1) − 4` law

This file formalises the *native-erasure ladder* measured at the Sounio
runtime in `examples/{sedenion,pathion,chingon,routon,…}_projective_
measurement.sio` and `examples/cd_l{8,9,10,11}_projective_measurement.sio`.

For a two-support generator `A = e_a + e_b` in the Cayley–Dickson algebra
`C_n` (real dimension `2^n`), left multiplication
  `L_A : C_n → C_n,  L_A(h) := A · h`
is a real-linear map.  Its matrix in the basis `{e_k}` has every entry in
`{-1, 0, +1}` — each entry is a value of the verified sign function
`cdSigma` (`SounioCayleyDickson.lean`), because
  `L_A(e_k) = σ(a,k)·e_{a⊕k} + σ(b,k)·e_{b⊕k}`.
Therefore `L_A` is an **integer matrix**, and its rank over `ℚ` equals its
rank over `ℝ`.  We compute that rank *exactly* in `ℤ` and obtain the
kernel dimension `dim ker L_A = 2^n − rank`.

The runtime (float Gaussian elimination) measured, for the native family
`A = e_3 + e_(2^(n-1)+2)`:

    n   dim    ker     erasure        closed form
    4   16     4       25.000000%     2^3  − 4
    5   32     12      37.500000%     2^4  − 4
    6   64     28      43.750000%     2^5  − 4
    7   128    60      46.875000%     2^6  − 4
    8   256    124     48.437500%     2^7  − 4
    9   512    252     49.218750%     2^8  − 4
    10  1024   508     49.609375%     2^9  − 4
    11  2048   1020    49.804688%     2^10 − 4

## Relation to the published literature (this is a re-derivation, not new math)

The annihilator-dimension theory of zero divisors in Cayley–Dickson
algebras is a developed published subject; this file *verifies* known
structure, it does not discover it.  A Q1 literature check (2026-06-02)
established:

* `dim ker L_A = dim Ann(A)` is the standard annihilator dimension; Moreno
  (1998) proved any zero divisor is imaginary with `dim Ann ≡ 0 (mod 4)`
  and `dim Ann ≤ 2^n − 4` (Cor. 1.17).
* The **sharp maximum** is `2^n − 4n + 4` (Biss–Dugger–Isaksen, *Large
  annihilators in Cayley–Dickson algebras*, Thm 1.2), and every multiple
  of 4 up to that bound is realised.  The native value `2^(n-1) − 4`
  (4, 12, 28, 60, 124, …) equals this maximum **only at n = 4**; for n ≥ 5
  it is strictly **sub-maximal** (the maxima are 4, 16, 44, 104, 228).
* `2^(n-1) − 4` is itself a published structural quantity — the
  top-dimensional eigentheory increment (Biss–Christensen–Dugger–Isaksen
  2009, Cor. 6.5) and the `{a,0}` combination term
  `dim Ann{a,0} = dim Ann a + 2^(n-1) − 4` (BCDI II, Prop. 5.11).
* The `(a,0)` tower-lift exactly *doubles* the annihilator (BDI Thm 10.1,
  `Ann(a,0) = Ann(a) × Ann(a)`).
* The generator `e_3 + e_(2^(n-1)+2)` is **not** a distinguished family: a
  direct ℤ-rank sweep finds 126 (n=5) / 342 (n=6) two-support generators
  with the same kernel dimension.

The contribution here is therefore the *formal, exact verification*
(integer-rank certificate over the verified `cdSigma` table) of these
known quantities at L4–L8 — not a new theorem.

## What this file PROVES

**(1) Algebra-level exact kernels, L4–L8 (`native_decide`, exact ℤ-rank).**
`native_family_kernels` evaluates the *verified* `cdSigma` product, builds
each `L_A` matrix and reduces it over `ℤ`, certifying that the kernel
dimension of the native generator equals the closed form `2^(n-1) − 4`
for `n = 4,5,6,7,8`.  This is the actual finrank (integer matrix ⇒
ℚ-rank = ℝ-rank), reproducing the runtime float-Gauss measurements
independently and exactly.  `lifted_family_kernels_L8` does the same for
all five tower-lifted families coexisting at L8 (kernels 64/96/112/120/124),
and `control_e1_invertible` certifies `L_{e_1}` is invertible (kernel 0) —
the method separates zero-divisors from units.

**(2) The arithmetic law of the ladder (core Lean, `omega`).**
`dim_eq_two_ker_add_eight`:  `dim n = 2·ker(n) + 8`  (n ≥ 4), i.e. the
kernel is *exactly* half the dimension minus 4, so the erasure fraction is
`1/2 − 4/2^n`, strictly below `1/2`, with headroom halving each rung.
`ker_recurrence`:  `ker(n+1) = 2·ker(n) + 4`.
`ker_strictMono`:  the kernel sequence is strictly increasing.
`liftedKer_succ`:  the tower-lift doubling shadow `lifted(k+1) = 2·lifted(k)`.

## What this file does **NOT** claim

* **The `∀n` closed form is established theory, not ours to conjecture.**
  `dim ker L_A = 2^(n-1) − 4` for the native generator follows from the
  published annihilator/eigentheory results cited above; this file proves
  it *exactly* only at L4–L8 (`native_decide`), with L9–L11 as runtime
  measurements.  A general-`n` Lean proof would need a formalised
  Cayley–Dickson algebra with `finrank` (Mathlib), not loaded here.  This
  is a verification of known structure, **not** a novelty claim.

* **The lifted-doubling is stated as arithmetic, not as an operator identity.**
  The structural reason a tower-lifted ZD doubles its kernel is
  `L_{(a,0)} = L_a ⊕ R_a` on `C_{n+1} = (C_n, C_n)`.  We do not formalise
  that ℝ-linear-operator decomposition; `liftedKer_succ` proves only its
  arithmetic shadow, and `lifted_family_kernels_L8` verifies the resulting
  numbers at L8 by direct computation.

* **No physics claim.**  `L_A` is real-linear and, for these zero-divisor
  generators, singular — hence non-unitary and **not** a quantum channel.
  A "projective measurement / erasure" reading is *not* supported by the
  literature (the Q1 check found no such framing) and is deliberately
  withheld.  Established physics in this tower is confined to the octonions
  (n = 3, a division algebra); sedenions and above (n ≥ 4) have no
  established physical role.  The genuine mathematical-physics depth is the
  G₂ connection: the normalised sedenion zero divisors are homeomorphic to
  `G₂ = Aut(𝕆)` (Moreno Cor. 2.14), with `ZD(S) ≅ V₂(ℝ⁷) = G₂/SU(2)`
  (Reggiani 2024) — not exercised by this file.

No `sorry`. No Mathlib. Algebra-level facts by `native_decide` (exact
ℤ-rank of the verified `cdSigma` product); arithmetic facts by `omega`.

References:
  - `SounioCayleyDickson.lean` (the `cdSigma` sign table this builds on)
  - `stdlib/algebra/cayley_dickson.sio` (runtime `cd_sigma`, sign-aligned)
  - `examples/cd_l{8,9,10,11}_projective_measurement.sio` (runtime witnesses)
  - Moreno (1998), *The zero divisors of the Cayley–Dickson algebras over
    the real numbers*, Bol. Soc. Mat. Mexicana 4 — arXiv:q-alg/9710013
    (Cor. 1.17 bound `2^n−4`; Cor. 2.14 `ZD(S) ≅ G₂`)
  - Biss, Dugger, Isaksen, *Large annihilators in Cayley–Dickson algebras*
    — arXiv:math/0511691 (Thm 1.2 max `2^n−4n+4`; Thm 10.1 `(a,0)` doubling)
  - Biss, Christensen, Dugger, Isaksen, eigentheory — `eigen.pdf`
    (Cor. 6.5 increment `2^(n-1)−4`); BCDI II — arXiv:math/0702075 (Prop 5.11)
  - Reggiani (2024) — arXiv:2411.18881 (`ZD(S) ≅ V₂(ℝ⁷) = G₂/SU(2)`)
  - Schafer (1954), Cawagas (2004)
-/

namespace Sounio.CayleyDicksonErasure

open Sounio.CayleyDickson

-- ================================================================
-- §1. The integer left-multiplication matrix and its exact rank
-- ================================================================

/-- Ambient real dimension of the level-`bits` Cayley–Dickson algebra. -/
def dimOf (bits : Nat) : Nat := 2 ^ bits

/-- Matrix of `L_A` for `A = e_a + e_b` at level `bits`, in basis `{e_k}`.
    Column `k` is `L_A(e_k) = σ(a,k)·e_{a⊕k} + σ(b,k)·e_{b⊕k}`, so every
    entry is a `cdSigma` value in `{-1,0,1}` — an integer matrix. -/
def leftMulMatrix (a b bits : Nat) : List (List Int) :=
  let n := dimOf bits
  (List.range n).map (fun row =>
    (List.range n).map (fun col =>
      (if (a ^^^ col) == row then cdSigma a col bits else 0)
      + (if (b ^^^ col) == row then cdSigma b col bits else 0)))

/-- Matrix of `L_A` for a single unit `A = e_a` (used for the control). -/
def leftMul1Matrix (a bits : Nat) : List (List Int) :=
  let n := dimOf bits
  (List.range n).map (fun row =>
    (List.range n).map (fun col =>
      if (a ^^^ col) == row then cdSigma a col bits else 0))

/-- Exact rank of an integer matrix by fraction-free row reduction:
    eliminate with `target := piv·target − t·pivotRow`, which stays in `ℤ`
    and preserves rank; the rank is the number of pivots.  Over `ℤ ⊆ ℚ`
    this is the exact ℚ-rank, hence the ℝ-rank of `L_A`. -/
partial def intRank (m0 : List (List Int)) : Nat :=
  let rec go (rows : List (List Int)) (col ncol rank : Nat) : Nat :=
    if col ≥ ncol then rank
    else match rows.find? (fun r => (r.getD col 0) ≠ 0) with
      | none => go rows (col+1) ncol rank
      | some pivotRow =>
        let rest := rows.filter (fun r => r != pivotRow)
        let piv := pivotRow.getD col 0
        let elim := rest.map (fun r =>
          let t := r.getD col 0
          if t == 0 then r
          else (List.zipWith (fun rv pv => piv * rv - t * pv) r pivotRow))
        go elim (col+1) ncol (rank+1)
  go m0 0 (m0.headD []).length 0

/-- Exact kernel dimension of `L_A` for `A = e_a + e_b` at level `bits`. -/
def kerExact (a b bits : Nat) : Nat := dimOf bits - intRank (leftMulMatrix a b bits)

-- ================================================================
-- §2. The closed-form law
-- ================================================================

/-- Conjectured native-erasure kernel dimension at level `n` for the
    family `A = e_3 + e_(2^(n-1)+2)`. -/
def kerNative (n : Nat) : Nat := 2 ^ (n - 1) - 4

-- helpers (core Lean only)
private theorem two_pow_split (n : Nat) (h : 1 ≤ n) : 2 ^ n = 2 * 2 ^ (n - 1) := by
  have e : 2 ^ ((n - 1) + 1) = 2 ^ (n - 1) * 2 := Nat.pow_succ 2 (n - 1)
  rw [show (n - 1) + 1 = n from by omega] at e
  rw [e, Nat.mul_comm]

private theorem four_le_pow (n : Nat) (h : 4 ≤ n) : 4 ≤ 2 ^ (n - 1) := by
  calc 4 = 2 ^ 2 := by decide
    _ ≤ 2 ^ (n - 1) := Nat.pow_le_pow_right (by omega) (by omega)

-- ================================================================
-- §3. Algebra-level exact kernels (native_decide over verified cdSigma)
-- ================================================================

/-- **Bridge L4–L8.**  The exact ℤ-rank of the `cdSigma` product certifies
    that the native generator's real kernel dimension equals the closed
    form `2^(n-1) − 4` at levels 4..8 — the genuine finrank, computed in
    `ℤ`, matching the runtime float-Gauss measurements (4/12/28/60/124). -/
theorem native_family_kernels :
    [kerExact 3 10 4, kerExact 3 18 5, kerExact 3 34 6, kerExact 3 66 7, kerExact 3 130 8]
      = [kerNative 4, kerNative 5, kerNative 6, kerNative 7, kerNative 8] := by
  native_decide

/-- **All five tower-lifted families coexist at L8** with the measured
    kernels: sedenion×4 = 64, pathion×3 = 96, chingon×2 = 112,
    routon×1 = 120, and the L8-native = 124. -/
theorem lifted_family_kernels_L8 :
    [kerExact 3 10 8, kerExact 3 18 8, kerExact 3 34 8, kerExact 3 66 8, kerExact 3 130 8]
      = [64, 96, 112, 120, 124] := by
  native_decide

/-- **Control.**  The unit `e_1` has invertible left multiplication
    (kernel 0): the exact-rank method separates zero-divisors from units. -/
theorem control_e1_invertible : dimOf 8 - intRank (leftMul1Matrix 1 8) = 0 := by
  native_decide

-- ================================================================
-- §4. The arithmetic law of the ladder (core Lean, omega)
-- ================================================================

/-- **Kernel is half-dimension minus 4.**  `dim n = 2·ker(n) + 8` for
    `n ≥ 4`, so erasure `= 1/2 − 4/2^n`: strictly below `1/2`, with the
    `4/2^n` headroom halving each rung as the dimension doubles. -/
theorem dim_eq_two_ker_add_eight (n : Nat) (h : 4 ≤ n) :
    dimOf n = 2 * kerNative n + 8 := by
  unfold dimOf kerNative
  have hp := two_pow_split n (by omega)
  have hge := four_le_pow n h
  omega

/-- **Doubling recurrence.**  `ker(n+1) = 2·ker(n) + 4` for `n ≥ 4`. -/
theorem ker_recurrence (n : Nat) (h : 4 ≤ n) :
    kerNative (n + 1) = 2 * kerNative n + 4 := by
  unfold kerNative
  have hp : 2 ^ n = 2 * 2 ^ (n - 1) := two_pow_split n (by omega)
  have hge := four_le_pow n h
  simp only [Nat.add_sub_cancel]
  omega

/-- **Strict monotonicity** of the kernel sequence. -/
theorem ker_strictMono (n : Nat) (h : 4 ≤ n) : kerNative n < kerNative (n + 1) := by
  rw [ker_recurrence n h]; have := four_le_pow n h; unfold kerNative; omega

/-- Arithmetic shadow of the tower-lift: a kernel of size `base` lifted
    one rung doubles.  (The operator reason `L_{(a,0)} = L_a ⊕ R_a` is
    stated in the module docstring as motivation; only this arithmetic
    identity is formalised.) -/
def liftedKer (base k : Nat) : Nat := 2 ^ k * base

theorem liftedKer_succ (base k : Nat) :
    liftedKer base (k + 1) = 2 * liftedKer base k := by
  unfold liftedKer
  rw [Nat.pow_succ, Nat.mul_comm (2 ^ k) 2, Nat.mul_assoc]

end Sounio.CayleyDicksonErasure
