/-!
# SounioSedenionProjector — EISA Global Dominance & Moreno–Lagrange Sedenion Projector

Formal Lean 4 proofs (100% Mathlib-free, zero unproven placeholders):
- Section 1: 7B EISA Global Dominance & Cone Pruning Soundness (over `Int`).
- Section 2: 8A Moreno–Lagrange Sedenion Rank-4 Projector over `ℤ` (`by native_decide`).

Signs align 100% with `stdlib/algebra/sedenion.sio` and `SounioSedenionAssociator1848.lean`.
-/

-- =============================================================================
-- Section 1 — 7B EISA Global Dominance & Cone Pruning Soundness (over `Int`)
-- =============================================================================

/-- EISA global certificate soundness: If an interval enclosure `[z_hat - err, z_hat + err]`
    holds for all `j < n`, and the upper bound of every non-champion candidate `j ≠ i_star`
    is strictly strictly less than the lower bound of candidate `i_star`, then `i_star`
    is the strict global maximizer of the true score vector `z`. -/
theorem eisa_global_cert_sound (n : Nat) (i_star : Nat) (z z_hat err : Nat → Int)
    (h_istar : i_star < n)
    (h_enclosure : ∀ j, j < n → z_hat j - err j ≤ z j ∧ z j ≤ z_hat j + err j)
    (h_dom : ∀ j, j < n → j ≠ i_star → z_hat j + err j < z_hat i_star - err i_star) :
    ∀ j, j < n → j ≠ i_star → z j < z i_star := by
  intro j hj hne
  have hj_enc := h_enclosure j hj
  have hi_enc := h_enclosure i_star h_istar
  have h_sep := h_dom j hj hne
  omega

/-- EISA cone pruning soundness: Under an interval enclosure for candidate scores,
    any candidate `j` whose upper bound lies strictly below candidate `i_star`'s
    lower bound cannot dominate `i_star` (in fact `z j < z i_star`), justifying
    immediate pruning from the search cone. -/
theorem eisa_cone_prune_sound (n : Nat) (i_star : Nat) (z z_hat err : Nat → Int)
    (h_istar : i_star < n)
    (h_enclosure : ∀ j, j < n → z_hat j - err j ≤ z j ∧ z j ≤ z_hat j + err j)
    (j : Nat) (hj : j < n)
    (h_prune : z_hat j + err j < z_hat i_star - err i_star) :
    z j < z i_star := by
  have hj_enc := h_enclosure j hj
  have hi_enc := h_enclosure i_star h_istar
  omega

-- =============================================================================
-- Section 2 — 8A Moreno–Lagrange Sedenion Rank-4 Projector over `ℤ`
-- =============================================================================

/-- Recursive Cayley-Dickson sign function verbatim from `SounioSedenionAssociator1848.lean`. -/
def cdSigma (a b : Nat) : Nat → Int
  | 0 => -1
  | 1 => if a == 0 || b == 0 then 1 else -1
  | (n+2) =>
      if a == 0 || b == 0 then 1
      else
        let half := 2 ^ (n+1)
        let aHi := a ≥ half; let bHi := b ≥ half; let aLo := a % half; let bLo := b % half
        if !aHi && !bHi then cdSigma aLo bLo (n+1)
        else if !aHi && bHi then cdSigma bLo aLo (n+1)
        else if aHi && !bHi then (if bLo == 0 then cdSigma aLo 0 (n+1) else - cdSigma aLo bLo (n+1))
        else (if bLo == 0 then - cdSigma 0 aLo (n+1) else cdSigma bLo aLo (n+1))

/-- Sedenion basis multiplication sign: `e_a * e_b = sig(a, b) * e_{a ^^^ b}`. -/
def sig (a b : Nat) : Int := cdSigma a b 4

/-- Flat 256-entry array representation of a 16×16 integer matrix for high-speed evaluation. -/
abbrev Mat16 := Array Int

/-- Left multiplication matrix for the two-support element `e_a + s * e_b`.
    Column `c` represents `(e_a + s * e_b) * e_c = sig(a, c) * e_{a ^^^ c} + s * sig(b, c) * e_{b ^^^ c}`. -/
def leftMat2 (a b : Nat) (s : Int) : Mat16 := Id.run do
  let mut arr : Array Int := Array.replicate 256 (0 : Int)
  for c in [0:16] do
    let ra := a ^^^ c
    let rb := b ^^^ c
    arr := arr.set! (ra * 16 + c) (arr[ra * 16 + c]! + sig a c)
    arr := arr.set! (rb * 16 + c) (arr[rb * 16 + c]! + s * sig b c)
  return arr

/-- Matrix addition for 16×16 matrices. -/
def matAdd (a b : Mat16) : Mat16 := Id.run do
  let mut arr : Array Int := Array.replicate 256 (0 : Int)
  for i in [0:256] do
    arr := arr.set! i (a[i]! + b[i]!)
  return arr

/-- Scalar multiplication for 16×16 matrices. -/
def matScale (s : Int) (m : Mat16) : Mat16 := Id.run do
  let mut arr : Array Int := Array.replicate 256 (0 : Int)
  for i in [0:256] do
    arr := arr.set! i (s * m[i]!)
  return arr

/-- Matrix multiplication for 16×16 matrices. -/
def matMul (a b : Mat16) : Mat16 := Id.run do
  let mut arr : Array Int := Array.replicate 256 (0 : Int)
  for r in [0:16] do
    for c in [0:16] do
      let mut sum : Int := 0
      for k in [0:16] do
        sum := sum + a[r * 16 + k]! * b[k * 16 + c]!
      arr := arr.set! (r * 16 + c) sum
  return arr

/-- The 16×16 identity matrix. -/
def id16 : Mat16 := Id.run do
  let mut arr : Array Int := Array.replicate 256 (0 : Int)
  for i in [0:16] do
    arr := arr.set! (i * 16 + i) 1
  return arr

/-- Moreno–Lagrange rank-4 projector polynomial over ℤ:
    `Q(M) = 8 * I_16 + 6 * M^2 + M^4`.
    Scales the exact orthogonal rank-4 projector `P_4 = I + (1/2)*M^2*(3I + M^2)` on `||Z||^2 = 2`
    by 8 to avoid rational denominators. -/
def projQ (m : Mat16) : Mat16 :=
  let m2 := matMul m m
  matAdd (matScale 8 id16) (matAdd (matScale 6 m2) (matMul m2 m2))

/-- Trace of a 16×16 matrix. -/
def matTrace (m : Mat16) : Int := Id.run do
  let mut tr : Int := 0
  for i in [0:16] do
    tr := tr + m[i * 16 + i]!
  return tr

/-- Symmetry check `M == M^T`. -/
def isSym (m : Mat16) : Bool := Id.run do
  for r in [0:16] do
    for c in [0:16] do
      if m[r * 16 + c]! != m[c * 16 + r]! then
        return false
  return true

/-- Exact elementwise equality of two 16×16 matrices. -/
def matEq (a b : Mat16) : Bool := Id.run do
  for i in [0:256] do
    if a[i]! != b[i]! then
      return false
  return true

/-- Zero-matrix test. -/
def isZero (m : Mat16) : Bool := Id.run do
  for i in [0:256] do
    if m[i]! != 0 then
      return false
  return true

def signs : List Int := [-1, 1]
def units7 : List Nat := [1, 2, 3, 4, 5, 6, 7]

/-- The 84 primitive sedenion zero-divisors:
    `Z = e_a + s * e_{8+b}` with `1 ≤ a, b ≤ 7`, `a ≠ b`, and `s ∈ {-1, 1}`. -/
def zd84 : List Mat16 :=
  units7.flatMap (fun a =>
    units7.flatMap (fun b =>
      if a != b then
        signs.map (fun s => leftMat2 a (8 + b) s)
      else []))

/-- The 42 two-support imaginary octonions:
    `X = e_a + s * e_b` with `1 ≤ a < b ≤ 7`, and `s ∈ {-1, 1}`. -/
def oct42 : List Mat16 :=
  units7.flatMap (fun a =>
    units7.flatMap (fun b =>
      if a < b then
        signs.map (fun s => leftMat2 a b s)
      else []))

/-- 1) Exactly 84 primitive sedenion zero-divisors. -/
theorem zd84_len : zd84.length = 84 := by native_decide

/-- 2) Q(M) is symmetric for all 84 primitive zero-divisors. -/
theorem zd84_symmetric : zd84.all (fun m => isSym (projQ m)) = true := by native_decide

/-- 3) Q(M) is an orthogonal projector up to factor 8: Q(M)^2 = 8 * Q(M). -/
theorem zd84_idempotent : zd84.all (fun m => matEq (matMul (projQ m) (projQ m)) (matScale 8 (projQ m))) = true := by native_decide

/-- 4) Q(M) has rank exactly 4: Trace(Q(M)) = 8 * 4 = 32. -/
theorem zd84_trace_32 : zd84.all (fun m => matTrace (projQ m) == 32) = true := by native_decide

/-- 5) Q(M) annihilates the left multiplication operator L_Z in both directions. -/
theorem zd84_annihilates_lz : zd84.all (fun m => isZero (matMul (projQ m) m) && isZero (matMul m (projQ m))) = true := by native_decide

/-- 6) Exactly 42 two-support imaginary octonions. -/
theorem oct42_len : oct42.length = 42 := by native_decide

/-- 7) Q(M) vanishes identically on octonions (octonions have no zero-divisors). -/
theorem oct42_vanishes : oct42.all (fun m => isZero (projQ m)) = true := by native_decide

-- =============================================================================
-- Section 3 — The 2D Immortal Subspace Theorem (e_0, e_8)
-- =============================================================================

/-- Check whether column `c` of a 16×16 matrix is all zeros. -/
def colIsZero (m : Mat16) (c : Nat) : Bool := Id.run do
  for r in [0:16] do
    if m[r * 16 + c]! != 0 then
      return false
  return true

/-- Check whether row `r` of a 16×16 matrix is all zeros. -/
def rowIsZero (m : Mat16) (r : Nat) : Bool := Id.run do
  for c in [0:16] do
    if m[r * 16 + c]! != 0 then
      return false
  return true

/-- 8) Q(M) has zero columns at 0 and 8 for all 84 primitive zero-divisors. -/
theorem zd84_immortal_cols_0_8 : zd84.all (fun m => colIsZero (projQ m) 0 && colIsZero (projQ m) 8) = true := by native_decide

/-- 9) Q(M) has zero rows at 0 and 8 for all 84 primitive zero-divisors. -/
theorem zd84_immortal_rows_0_8 : zd84.all (fun m => rowIsZero (projQ m) 0 && rowIsZero (projQ m) 8) = true := by native_decide

/-- 10) Coordinates 0 and 8 are the exact immortal coordinates annihilated across all 84 zero-divisors. -/
theorem zd84_exact_immortal_coords : (List.range 16).filter (fun c => zd84.all (fun m => colIsZero (projQ m) c)) = [0, 8] := by native_decide

