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

* **The closed form for all `n` is a conjecture, not a theorem.**
  `kerNative n = 2^(n-1) − 4` is verified at eight runtime points and proved
  exactly (algebra-level) at five of them (L4–L8); L9–L11 are runtime
  measurements only.  We do **not** prove `dim ker L_A = 2^(n-1) − 4` for
  *all* `n ≥ 4`; that would require a general kernel-dimension theorem over
  a formalised Cayley–Dickson algebra (Mathlib `LinearMap.ker` /
  `FiniteDimensional.finrank`), not loaded here.

* **The lifted-doubling is stated as arithmetic, not as an operator identity.**
  The structural reason a tower-lifted ZD doubles its kernel is
  `L_{(a,0)} = L_a ⊕ R_a` on `C_{n+1} = (C_n, C_n)`.  We do not formalise
  that ℝ-linear-operator decomposition; `liftedKer_succ` proves only its
  arithmetic shadow, and `lifted_family_kernels_L8` verifies the resulting
  numbers at L8 by direct computation.

* **No Heisenberg claim.**  `σ_x σ_p ≥ ℏ/2` is a theorem about unitary
  evolution on a complex Hilbert space; `L_A` is real-linear and
  non-unitary by construction.

No `sorry`. No Mathlib. Algebra-level facts by `native_decide` (exact
ℤ-rank of the verified `cdSigma` product); arithmetic facts by `omega`.

References:
  - `SounioCayleyDickson.lean` (the `cdSigma` sign table this builds on)
  - `stdlib/algebra/cayley_dickson.sio` (runtime `cd_sigma`, sign-aligned)
  - `examples/cd_l{8,9,10,11}_projective_measurement.sio` (runtime witnesses)
  - Schafer (1954), Moreno (1998), Cawagas (2004)
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
