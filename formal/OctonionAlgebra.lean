/-!
# Sounio.OctonionAlgebra — Phase 8 Formal Verification

Formalisation of the octonion (𝕆) algebra underlying Sounio's
native AVX-512 backend (`self-hosted/native/lower_ir.sio:1620`, `lower_hyper_mul_o_fano`)
and standard mathematical library (`stdlib/math/octonion.sio`).

Constructed strictly via the **Cayley-Dickson** doubling process over Hamilton quaternions (ℍ):
  x = (xA, xB), y = (yA, yB) ∈ ℍ × ℍ
  xy = (xA·yA − conj(yB)·xB, yB·xA + xB·conj(yA))

The Cayley–Dickson definition is the sole multiplication used by the proofs below.
The current file proves distributivity, identities, basis products, and one
norm witness. Universal alternativity, Moufang identities, conjugate laws, and
norm multiplicativity remain explicitly marked `axiom` until proved.

References:
  - Baez 2002, "The Octonions", Bull. AMS 39(2):145-205
  - Hurwitz 1898, "Über die Komposition der quadratischen Formen"
  - Conway & Smith 2003, "On Quaternions and Octonions"
-/

namespace Sounio.OctonionAlgebra

-- ---------------------------------------------------------------------------
-- §1. Quaternions (Hamilton ℍ) as the base of the Cayley-Dickson tower
-- ---------------------------------------------------------------------------

/-- Quaternion representation with 4 integer components (w, x, y, z). -/
structure Quat where
  w : Int
  x : Int
  y : Int
  z : Int
  deriving DecidableEq, Repr

def quatAdd (a b : Quat) : Quat :=
  ⟨a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z⟩

def quatSub (a b : Quat) : Quat :=
  ⟨a.w - b.w, a.x - b.x, a.y - b.y, a.z - b.z⟩

def quatNeg (a : Quat) : Quat :=
  ⟨-a.w, -a.x, -a.y, -a.z⟩

def quatConj (a : Quat) : Quat :=
  ⟨a.w, -a.x, -a.y, -a.z⟩

/-- Hamilton quaternion multiplication:
    i² = j² = k² = ijk = -1
    ij = k, jk = i, ki = j
    ji = -k, kj = -i, ik = -j -/
def quatMul (a b : Quat) : Quat where
  w := a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
  x := a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y
  y := a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x
  z := a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w

-- ---------------------------------------------------------------------------
-- §2. Octonions (Cayley-Dickson 𝕆 = ℍ ⊕ ℍℓ)
-- ---------------------------------------------------------------------------

/-- An octonion as 8 integer components e₀..e₇.
    Working over Int gives exact arithmetic and lets `decide` verify every finite property. -/
structure Oct where
  e0 : Int
  e1 : Int
  e2 : Int
  e3 : Int
  e4 : Int
  e5 : Int
  e6 : Int
  e7 : Int
  deriving DecidableEq, Repr

def Oct.hi (x : Oct) : Quat := ⟨x.e0, x.e1, x.e2, x.e3⟩
def Oct.lo (x : Oct) : Quat := ⟨x.e4, x.e5, x.e6, x.e7⟩

def Oct.fromQuats (a b : Quat) : Oct :=
  ⟨a.w, a.x, a.y, a.z, b.w, b.x, b.y, b.z⟩

-- ---------------------------------------------------------------------------
-- §3. Basis elements (e₀ = scalar unit, e₁..e₇ = imaginary units)
-- ---------------------------------------------------------------------------

def e0 : Oct := ⟨1, 0, 0, 0, 0, 0, 0, 0⟩
def e1 : Oct := ⟨0, 1, 0, 0, 0, 0, 0, 0⟩
def e2 : Oct := ⟨0, 0, 1, 0, 0, 0, 0, 0⟩
def e3 : Oct := ⟨0, 0, 0, 1, 0, 0, 0, 0⟩
def e4 : Oct := ⟨0, 0, 0, 0, 1, 0, 0, 0⟩
def e5 : Oct := ⟨0, 0, 0, 0, 0, 1, 0, 0⟩
def e6 : Oct := ⟨0, 0, 0, 0, 0, 0, 1, 0⟩
def e7 : Oct := ⟨0, 0, 0, 0, 0, 0, 0, 1⟩

-- ---------------------------------------------------------------------------
-- §4. Addition, scaling and negation
-- ---------------------------------------------------------------------------

def octAdd (x y : Oct) : Oct :=
  ⟨x.e0 + y.e0, x.e1 + y.e1, x.e2 + y.e2, x.e3 + y.e3,
   x.e4 + y.e4, x.e5 + y.e5, x.e6 + y.e6, x.e7 + y.e7⟩

def octScale (n : Int) (x : Oct) : Oct :=
  ⟨n * x.e0, n * x.e1, n * x.e2, n * x.e3,
   n * x.e4, n * x.e5, n * x.e6, n * x.e7⟩

def octNeg (x : Oct) : Oct :=
  ⟨-x.e0, -x.e1, -x.e2, -x.e3, -x.e4, -x.e5, -x.e6, -x.e7⟩

def octSub (x y : Oct) : Oct :=
  octAdd x (octNeg y)

-- ---------------------------------------------------------------------------
-- §5. Octonion conjugate and norm squared
-- ---------------------------------------------------------------------------

/-- Octonion conjugate: negate the 7 imaginary components. -/
def octConj (x : Oct) : Oct :=
  ⟨x.e0, -x.e1, -x.e2, -x.e3, -x.e4, -x.e5, -x.e6, -x.e7⟩

/-- Norm squared: |x|² = Σ xᵢ² (the Euclidean norm squared over ℤ). -/
def octNormSq (x : Oct) : Int :=
  x.e0^2 + x.e1^2 + x.e2^2 + x.e3^2 + x.e4^2 + x.e5^2 + x.e6^2 + x.e7^2

-- ---------------------------------------------------------------------------
-- §6. Octonion multiplication via Cayley-Dickson construction
--
-- Definition: (a, b) · (c, d) = (a·c − conj(d)·b, d·a + b·conj(c))
-- Matches stdlib/math/octonion.sio and lower_ir.sio:1620 (e1·e2 = e3).
-- ---------------------------------------------------------------------------

/-- Canonical Octonion multiplication: SOLE DEFINITION via Cayley-Dickson doubling
    over Hamilton quaternions (a, b) · (c, d) = (a·c − conj(d)·b, d·a + b·conj(c)).
    Completely eliminates duplicate definitions and circularity. -/
def octMul (x y : Oct) : Oct :=
  let a := x.hi
  let b := x.lo
  let c := y.hi
  let d := y.lo
  let first := quatSub (quatMul a c) (quatMul (quatConj d) b)
  let second := quatAdd (quatMul d a) (quatMul b (quatConj c))
  Oct.fromQuats first second

-- ---------------------------------------------------------------------------
-- §7. Extensionality
-- ---------------------------------------------------------------------------

@[ext]
theorem oct_ext (x y : Oct)
    (h0 : x.e0 = y.e0) (h1 : x.e1 = y.e1) (h2 : x.e2 = y.e2) (h3 : x.e3 = y.e3)
    (h4 : x.e4 = y.e4) (h5 : x.e5 = y.e5) (h6 : x.e6 = y.e6) (h7 : x.e7 = y.e7) :
    x = y := by
  cases x; cases y; simp_all

-- ---------------------------------------------------------------------------
-- §8. Addition laws
-- ---------------------------------------------------------------------------

theorem oct_add_comm (x y : Oct) : octAdd x y = octAdd y x := by
  simp only [octAdd]; ext <;> simp [Int.add_comm]

theorem oct_add_assoc (x y z : Oct) : octAdd (octAdd x y) z = octAdd x (octAdd y z) := by
  simp only [octAdd]; ext <;> simp [Int.add_assoc]

theorem oct_add_zero (x : Oct) : octAdd x ⟨0,0,0,0,0,0,0,0⟩ = x := by
  simp only [octAdd]; ext <;> simp

theorem oct_zero_add (x : Oct) : octAdd ⟨0,0,0,0,0,0,0,0⟩ x = x := by
  simp only [octAdd]; ext <;> simp

theorem oct_add_neg (x : Oct) : octAdd x (octNeg x) = ⟨0,0,0,0,0,0,0,0⟩ := by
  simp only [octAdd, octNeg]; ext <;> simp [Int.add_right_neg]

-- ---------------------------------------------------------------------------
-- §9. Scalar multiplication laws
-- ---------------------------------------------------------------------------

theorem oct_scalar_one (x : Oct) : octScale 1 x = x := by
  simp only [octScale]; ext <;> simp

theorem oct_scalar_zero (x : Oct) : octScale 0 x = ⟨0,0,0,0,0,0,0,0⟩ := by
  simp only [octScale]; ext <;> simp

theorem oct_scalar_add (m n : Int) (x : Oct) :
    octScale (m + n) x = octAdd (octScale m x) (octScale n x) := by
  simp only [octScale, octAdd]; ext <;> simp [Int.add_mul]

theorem oct_scalar_mul_assoc (m n : Int) (x : Oct) :
    octScale (m * n) x = octScale m (octScale n x) := by
  simp only [octScale]; ext <;> simp [Int.mul_assoc]

-- ---------------------------------------------------------------------------
-- §10. Multiplication distributes over addition
-- ---------------------------------------------------------------------------

-- ---------------------------------------------------------------------------
-- §10. Multiplication distributes over addition
-- ---------------------------------------------------------------------------

/-- Distributivity of octonion multiplication over addition (left).
    Formally proved without Mathlib from the Cayley-Dickson definition. -/
theorem oct_mul_add_left (x y z : Oct) :
    octMul x (octAdd y z) = octAdd (octMul x y) (octMul x z) := by
  ext
  all_goals
    simp only [octMul, quatMul, quatAdd, quatSub, quatConj, Oct.hi, Oct.lo, Oct.fromQuats, octAdd,
               Int.mul_add, Int.add_mul, Int.neg_mul, Int.mul_neg, Int.sub_eq_add_neg, Int.neg_add]
    omega

/-- Distributivity of octonion multiplication over addition (right).
    Formally proved without Mathlib from the Cayley-Dickson definition. -/
theorem oct_mul_add_right (x y z : Oct) :
    octMul (octAdd x y) z = octAdd (octMul x z) (octMul y z) := by
  ext
  all_goals
    simp only [octMul, quatMul, quatAdd, quatSub, quatConj, Oct.hi, Oct.lo, Oct.fromQuats, octAdd,
               Int.mul_add, Int.add_mul, Int.neg_mul, Int.mul_neg, Int.sub_eq_add_neg, Int.neg_add]
    omega

/-- Canonical basis decomposition: every octonion is uniquely expressed
    as an integer linear combination of the eight standard basis elements e₀..e₇. -/
theorem oct_decompose (x : Oct) :
    x = octAdd (octScale x.e0 e0)
       (octAdd (octScale x.e1 e1)
       (octAdd (octScale x.e2 e2)
       (octAdd (octScale x.e3 e3)
       (octAdd (octScale x.e4 e4)
       (octAdd (octScale x.e5 e5)
       (octAdd (octScale x.e6 e6)
               (octScale x.e7 e7))))))) := by
  ext <;> simp [octAdd, octScale, e0, e1, e2, e3, e4, e5, e6, e7]

/-- Integer scaling distributes over octonion addition. -/
theorem oct_scale_add (n : Int) (x y : Oct) :
    octScale n (octAdd x y) = octAdd (octScale n x) (octScale n y) := by
  ext <;> simp [octScale, octAdd, Int.mul_add]


-- ---------------------------------------------------------------------------
-- §11. Identity element
-- ---------------------------------------------------------------------------

theorem oct_mul_one (x : Oct) : octMul x e0 = x := by
  ext <;> simp [octMul, Oct.hi, Oct.lo, Oct.fromQuats, quatMul, quatSub, quatAdd, quatConj, e0]

theorem oct_one_mul (x : Oct) : octMul e0 x = x := by
  ext <;> simp [octMul, Oct.hi, Oct.lo, Oct.fromQuats, quatMul, quatSub, quatAdd, quatConj, e0]

-- ---------------------------------------------------------------------------
-- §12. Non-commutativity (e₁·e₂ ≠ e₂·e₁)
-- ---------------------------------------------------------------------------

/-- Octonions are non-commutative: e₁·e₂ = +e₃ but e₂·e₁ = −e₃. -/
theorem oct_noncommutative :
    ∃ x y : Oct, octMul x y ≠ octMul y x :=
  ⟨e1, e2, by decide⟩

-- ---------------------------------------------------------------------------
-- §13. Non-associativity (e₁·e₂)·e₄ ≠ e₁·(e₂·e₄)
-- ---------------------------------------------------------------------------

theorem oct_nonassociative :
    ∃ x y z : Oct, octMul (octMul x y) z ≠ octMul x (octMul y z) :=
  ⟨e1, e2, e4, by decide⟩

-- ---------------------------------------------------------------------------
-- §14. Alternative laws — the defining property of octonions
-- ---------------------------------------------------------------------------

/-- Left alternative law: x(xy) = (x²)y -/
axiom oct_left_alternative (x y : Oct) :
    octMul x (octMul x y) = octMul (octMul x x) y

/-- Right alternative law: (yx)x = y(x²) -/
axiom oct_right_alternative (x y : Oct) :
    octMul (octMul y x) x = octMul y (octMul x x)

-- ---------------------------------------------------------------------------
-- §15. Flexibility identity
-- ---------------------------------------------------------------------------

/-- Flexibility: x(yx) = (xy)x — follows from the alternative laws. -/
axiom oct_flexibility (x y : Oct) :
    octMul x (octMul y x) = octMul (octMul x y) x

-- ---------------------------------------------------------------------------
-- §16. Moufang identities
-- ---------------------------------------------------------------------------

/-- Moufang identity (left): z(x(zy)) = ((zx)z)y -/
axiom oct_moufang_left (x y z : Oct) :
    octMul z (octMul x (octMul z y)) = octMul (octMul (octMul z x) z) y

/-- Moufang identity (right): ((xy)z)y = x(y(zy)) -/
axiom oct_moufang_right (x y z : Oct) :
    octMul (octMul (octMul x y) z) y = octMul x (octMul y (octMul z y))

/-- Moufang identity (middle): (xy)(zx) = x((yz)x) -/
axiom oct_moufang_middle (x y z : Oct) :
    octMul (octMul x y) (octMul z x) = octMul x (octMul (octMul y z) x)

-- ---------------------------------------------------------------------------
-- §17. Scalar multiplication commutes with octMul
-- ---------------------------------------------------------------------------

/-- Integer scaling commutes with octonion multiplication (left). -/
axiom oct_scalar_comm (n : Int) (x y : Oct) :
    octMul (octScale n x) y = octScale n (octMul x y)

/-- Integer scaling commutes with octonion multiplication (right). -/
axiom oct_scalar_comm_right (n : Int) (x y : Oct) :
    octMul x (octScale n y) = octScale n (octMul x y)

-- ---------------------------------------------------------------------------
-- §18. Conjugate laws
-- ---------------------------------------------------------------------------

/-- Conjugation is an anti-automorphism: conj(xy) = conj(y)·conj(x). -/
axiom oct_conj_antimultiplicative (x y : Oct) :
    octConj (octMul x y) = octMul (octConj y) (octConj x)

/-- Double conjugation is identity. -/
theorem oct_conj_involution (x : Oct) : octConj (octConj x) = x := by
  simp only [octConj]; ext <;> simp

/-- x + conj(x) = 2·e₀ component only (real part doubled). -/
theorem oct_conj_add_real (x : Oct) :
    octAdd x (octConj x) = ⟨2 * x.e0, 0, 0, 0, 0, 0, 0, 0⟩ := by
  simp only [octAdd, octConj]; ext <;> simp [Int.two_mul, Int.add_right_neg]

/-- x · conj(x) = |x|² · e₀. -/
axiom oct_mul_conj (x : Oct) :
    octMul x (octConj x) = ⟨octNormSq x, 0, 0, 0, 0, 0, 0, 0⟩

/-- conj(x) · x = |x|² · e₀. -/
axiom oct_conj_mul (x : Oct) :
    octMul (octConj x) x = ⟨octNormSq x, 0, 0, 0, 0, 0, 0, 0⟩

-- ---------------------------------------------------------------------------
-- §19. Norm multiplicativity — Degen's eight-square identity
-- ---------------------------------------------------------------------------

/-- Norm multiplicativity: the octonion norm is multiplicative.
    Encodes the Degen eight-square identity over ℤ.
    Polynomial identity; provable by `ring` with Mathlib. -/
axiom oct_norm_multiplicative (x y : Oct) :
    octNormSq (octMul x y) = octNormSq x * octNormSq y

-- Regression test: The error found on e6 * e4 now strictly evaluates to 1
theorem oct_norm_mult_e6_e4 : octNormSq (octMul e6 e4) = 1 := by decide

-- ---------------------------------------------------------------------------
-- §20. Power laws (from alternative laws)
-- ---------------------------------------------------------------------------

/-- x(x²) = (x²)x — consequence of alternativity. -/
axiom oct_sq_comm_left (x : Oct) :
    octMul x (octMul x x) = octMul (octMul x x) x

-- ---------------------------------------------------------------------------
-- §21. Connection to Sounio's epistemic GEMM kernel
-- ---------------------------------------------------------------------------

theorem gemm_tiling_nonassoc_caveat :
    ∃ (A B C : Oct),
      octMul A (octMul B C) ≠ octMul (octMul A B) C :=
  ⟨e1, e2, e4, by decide⟩

theorem gemm_safe_tile_left (tile acc : Oct) :
    octMul tile (octMul tile acc) = octMul (octMul tile tile) acc := by
  exact oct_left_alternative tile acc

theorem gemm_safe_tile_right (tile acc : Oct) :
    octMul (octMul acc tile) tile = octMul acc (octMul tile tile) := by
  exact oct_right_alternative tile acc

-- ---------------------------------------------------------------------------
-- §22. Basis multiplication (Complete 49 products of imaginary units by decide)
-- ---------------------------------------------------------------------------

theorem basis_e1_e1 : octMul e1 e1 = octNeg e0 := by decide
theorem basis_e1_e2 : octMul e1 e2 = e3 := by decide
theorem basis_e1_e3 : octMul e1 e3 = octNeg e2 := by decide
theorem basis_e1_e4 : octMul e1 e4 = e5 := by decide
theorem basis_e1_e5 : octMul e1 e5 = octNeg e4 := by decide
theorem basis_e1_e6 : octMul e1 e6 = octNeg e7 := by decide
theorem basis_e1_e7 : octMul e1 e7 = e6 := by decide

theorem basis_e2_e1 : octMul e2 e1 = octNeg e3 := by decide
theorem basis_e2_e2 : octMul e2 e2 = octNeg e0 := by decide
theorem basis_e2_e3 : octMul e2 e3 = e1 := by decide
theorem basis_e2_e4 : octMul e2 e4 = e6 := by decide
theorem basis_e2_e5 : octMul e2 e5 = e7 := by decide
theorem basis_e2_e6 : octMul e2 e6 = octNeg e4 := by decide
theorem basis_e2_e7 : octMul e2 e7 = octNeg e5 := by decide

theorem basis_e3_e1 : octMul e3 e1 = e2 := by decide
theorem basis_e3_e2 : octMul e3 e2 = octNeg e1 := by decide
theorem basis_e3_e3 : octMul e3 e3 = octNeg e0 := by decide
theorem basis_e3_e4 : octMul e3 e4 = e7 := by decide
theorem basis_e3_e5 : octMul e3 e5 = octNeg e6 := by decide
theorem basis_e3_e6 : octMul e3 e6 = e5 := by decide
theorem basis_e3_e7 : octMul e3 e7 = octNeg e4 := by decide

theorem basis_e4_e1 : octMul e4 e1 = octNeg e5 := by decide
theorem basis_e4_e2 : octMul e4 e2 = octNeg e6 := by decide
theorem basis_e4_e3 : octMul e4 e3 = octNeg e7 := by decide
theorem basis_e4_e4 : octMul e4 e4 = octNeg e0 := by decide
theorem basis_e4_e5 : octMul e4 e5 = e1 := by decide
theorem basis_e4_e6 : octMul e4 e6 = e2 := by decide
theorem basis_e4_e7 : octMul e4 e7 = e3 := by decide

theorem basis_e5_e1 : octMul e5 e1 = e4 := by decide
theorem basis_e5_e2 : octMul e5 e2 = octNeg e7 := by decide
theorem basis_e5_e3 : octMul e5 e3 = e6 := by decide
theorem basis_e5_e4 : octMul e5 e4 = octNeg e1 := by decide
theorem basis_e5_e5 : octMul e5 e5 = octNeg e0 := by decide
theorem basis_e5_e6 : octMul e5 e6 = octNeg e3 := by decide
theorem basis_e5_e7 : octMul e5 e7 = e2 := by decide

theorem basis_e6_e1 : octMul e6 e1 = e7 := by decide
theorem basis_e6_e2 : octMul e6 e2 = e4 := by decide
theorem basis_e6_e3 : octMul e6 e3 = octNeg e5 := by decide
theorem basis_e6_e4 : octMul e6 e4 = octNeg e2 := by decide
theorem basis_e6_e5 : octMul e6 e5 = e3 := by decide
theorem basis_e6_e6 : octMul e6 e6 = octNeg e0 := by decide
theorem basis_e6_e7 : octMul e6 e7 = octNeg e1 := by decide

theorem basis_e7_e1 : octMul e7 e1 = octNeg e6 := by decide
theorem basis_e7_e2 : octMul e7 e2 = e5 := by decide
theorem basis_e7_e3 : octMul e7 e3 = e4 := by decide
theorem basis_e7_e4 : octMul e7 e4 = octNeg e3 := by decide
theorem basis_e7_e5 : octMul e7 e5 = octNeg e2 := by decide
theorem basis_e7_e6 : octMul e7 e6 = e1 := by decide
theorem basis_e7_e7 : octMul e7 e7 = octNeg e0 := by decide

-- e₀ identity theorems
theorem basis_e0_left  (x : Oct) : octMul e0 x = x := oct_one_mul x
theorem basis_e0_right (x : Oct) : octMul x e0 = x := oct_mul_one x

-- ---------------------------------------------------------------------------
-- §23. Norm of basis elements
-- ---------------------------------------------------------------------------

theorem basis_norm_e0 : octNormSq e0 = 1 := by decide
theorem basis_norm_e1 : octNormSq e1 = 1 := by decide
theorem basis_norm_e2 : octNormSq e2 = 1 := by decide
theorem basis_norm_e3 : octNormSq e3 = 1 := by decide
theorem basis_norm_e4 : octNormSq e4 = 1 := by decide
theorem basis_norm_e5 : octNormSq e5 = 1 := by decide
theorem basis_norm_e6 : octNormSq e6 = 1 := by decide
theorem basis_norm_e7 : octNormSq e7 = 1 := by decide

def octMulExpanded (x y : Oct) : Oct where
  e0 := x.e0*y.e0 - x.e1*y.e1 - x.e2*y.e2 - x.e3*y.e3 - x.e4*y.e4 - x.e5*y.e5 - x.e6*y.e6 - x.e7*y.e7
  e1 := x.e0*y.e1 + x.e1*y.e0 + x.e2*y.e3 - x.e3*y.e2 + x.e4*y.e5 - x.e5*y.e4 - x.e6*y.e7 + x.e7*y.e6
  e2 := x.e0*y.e2 - x.e1*y.e3 + x.e2*y.e0 + x.e3*y.e1 + x.e4*y.e6 + x.e5*y.e7 - x.e6*y.e4 - x.e7*y.e5
  e3 := x.e0*y.e3 + x.e1*y.e2 - x.e2*y.e1 + x.e3*y.e0 + x.e4*y.e7 - x.e5*y.e6 + x.e6*y.e5 - x.e7*y.e4
  e4 := x.e0*y.e4 - x.e1*y.e5 - x.e2*y.e6 - x.e3*y.e7 + x.e4*y.e0 + x.e5*y.e1 + x.e6*y.e2 + x.e7*y.e3
  e5 := x.e0*y.e5 + x.e1*y.e4 - x.e2*y.e7 + x.e3*y.e6 - x.e4*y.e1 + x.e5*y.e0 - x.e6*y.e3 + x.e7*y.e2
  e6 := x.e0*y.e6 + x.e1*y.e7 + x.e2*y.e4 - x.e3*y.e5 - x.e4*y.e2 + x.e5*y.e3 + x.e6*y.e0 - x.e7*y.e1
  e7 := x.e0*y.e7 - x.e1*y.e6 + x.e2*y.e5 + x.e3*y.e4 - x.e4*y.e3 - x.e5*y.e2 + x.e6*y.e1 + x.e7*y.e0

/-- Algebraic 64-term component expansion theorem derived from the sole Cayley-Dickson definition. -/
theorem octMul_expand (i j : Fin 8) :
    octMul ([e0, e1, e2, e3, e4, e5, e6, e7].get i) ([e0, e1, e2, e3, e4, e5, e6, e7].get j) =
    octMulExpanded ([e0, e1, e2, e3, e4, e5, e6, e7].get i) ([e0, e1, e2, e3, e4, e5, e6, e7].get j) := by
  revert i j
  decide

end Sounio.OctonionAlgebra
