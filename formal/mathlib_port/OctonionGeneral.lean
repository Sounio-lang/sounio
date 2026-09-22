import Mathlib.Tactic.Ring
import Mathlib.Algebra.Star.Basic

/-!
# Octonions over a general commutative ring

Generalization of `Sounio.OctonionAlgebra.Oct` (fixed to `Int`) to `Octonion R`
for an arbitrary commutative ring `R`, mirroring how `Mathlib.Algebra.Quaternion`
defines `Quaternion R` (`ℍ[R]`) as an explicit 4-field structure over a general
ring, rather than via a generic Cayley-Dickson doubling functor.

This *does* build `AddCommGroup`, `NonUnitalNonAssocRing` (not `Ring`: see
§20 for why associativity is correctly omitted), and `StarRing` instances
(§20) on top of the bare `mul`/`add`/`neg` definitions and their algebraic
identities (all proved, no axioms, §7-19). Deliberately **not yet** attempted:
generalizing further to free structure constants `OctonionAlgebra R a b c d`,
or `Algebra R`/`DivisionRing`/alternative-algebra-specific instances — see
`../ZULIP_DRAFT_octonions.md` for the plan and prior art (2023 Lorentz Center
workshop attempt, stalled on a generic Cayley-Dickson functor over Mathlib's
associative-biased typeclass hierarchy) and `mathlib_port/README.md`'s
"Next step" section for the full remaining list.

Sign convention matches `Sounio.OctonionAlgebra.octMul` (`../OctonionAlgebra.lean`,
fixed 2026-09-22 after a real bug was found in an earlier version — see that
file's `octMul` docstring and `mathlib_port/README.md`), which in turn matches
the canonical `stdlib/algebra/octonion.sio` / `octonion.py` used for the
168-theorem computation. This is the standard octonions case
(quaternion part `i²=-1, j²=-1, k²=-1`, doubled by `ℓ²=-1`); the further
generalization to `Octonion R a b c d` with free structure constants is
deferred to a follow-up.

**Dependency closure re: the 168-theorem.** `../FanoLabellingOrbits.lean`
(the file that actually proves `fano_automorphism_group_card : (fanoAuts).length
= 168` via `native_decide`) states explicitly in its own header: "File is
deliberately self-contained (no `import`, no Mathlib, no `OctonionAlgebra`
dependency)". So the bug in the original (axiom-based) `octMul` could not
have propagated into that proof through any import chain, not merely through
"not calling `octMul`" -- there is no path between the two files at all.
-/

namespace Sounio.OctonionGeneral

variable {R : Type*} [CommRing R]

-- ---------------------------------------------------------------------------
-- §1. Octonion representation
-- ---------------------------------------------------------------------------

/-- An octonion over a commutative ring `R`, as 8 components `e0..e7`. -/
structure Octonion (R : Type*) where
  e0 : R
  e1 : R
  e2 : R
  e3 : R
  e4 : R
  e5 : R
  e6 : R
  e7 : R
  deriving DecidableEq, Repr

-- ---------------------------------------------------------------------------
-- §2. Basis elements
-- ---------------------------------------------------------------------------

def e0 : Octonion R := ⟨1, 0, 0, 0, 0, 0, 0, 0⟩
def e1 : Octonion R := ⟨0, 1, 0, 0, 0, 0, 0, 0⟩
def e2 : Octonion R := ⟨0, 0, 1, 0, 0, 0, 0, 0⟩
def e3 : Octonion R := ⟨0, 0, 0, 1, 0, 0, 0, 0⟩
def e4 : Octonion R := ⟨0, 0, 0, 0, 1, 0, 0, 0⟩
def e5 : Octonion R := ⟨0, 0, 0, 0, 0, 1, 0, 0⟩
def e6 : Octonion R := ⟨0, 0, 0, 0, 0, 0, 1, 0⟩
def e7 : Octonion R := ⟨0, 0, 0, 0, 0, 0, 0, 1⟩

-- ---------------------------------------------------------------------------
-- §3. Addition, negation, and R-scaling
-- ---------------------------------------------------------------------------

def octAdd (x y : Octonion R) : Octonion R :=
  ⟨x.e0 + y.e0, x.e1 + y.e1, x.e2 + y.e2, x.e3 + y.e3,
   x.e4 + y.e4, x.e5 + y.e5, x.e6 + y.e6, x.e7 + y.e7⟩

def octScale (n : R) (x : Octonion R) : Octonion R :=
  ⟨n * x.e0, n * x.e1, n * x.e2, n * x.e3,
   n * x.e4, n * x.e5, n * x.e6, n * x.e7⟩

def octNeg (x : Octonion R) : Octonion R :=
  ⟨-x.e0, -x.e1, -x.e2, -x.e3, -x.e4, -x.e5, -x.e6, -x.e7⟩

-- ---------------------------------------------------------------------------
-- §4. Octonion multiplication (Cayley-Dickson, standard-octonion coefficients)
--
-- Same sign convention as `Sounio.OctonionAlgebra.octMul` post-fix, and as
-- `stdlib/algebra/octonion.sio` / `octonion.py`.
-- ---------------------------------------------------------------------------

def octMul (x y : Octonion R) : Octonion R where
  e0 :=   x.e0 * y.e0 - x.e1 * y.e1 - x.e2 * y.e2 - x.e3 * y.e3
        - x.e4 * y.e4 - x.e5 * y.e5 - x.e6 * y.e6 - x.e7 * y.e7
  e1 :=   x.e0 * y.e1 + x.e1 * y.e0 + x.e2 * y.e3 - x.e3 * y.e2
        + x.e4 * y.e5 - x.e5 * y.e4 - x.e6 * y.e7 + x.e7 * y.e6
  e2 :=   x.e0 * y.e2 + x.e2 * y.e0 - x.e1 * y.e3 + x.e3 * y.e1
        + x.e4 * y.e6 - x.e6 * y.e4 + x.e5 * y.e7 - x.e7 * y.e5
  e3 :=   x.e0 * y.e3 + x.e3 * y.e0 + x.e1 * y.e2 - x.e2 * y.e1
        + x.e4 * y.e7 - x.e7 * y.e4 - x.e5 * y.e6 + x.e6 * y.e5
  e4 :=   x.e0 * y.e4 + x.e4 * y.e0 - x.e1 * y.e5 + x.e5 * y.e1
        - x.e2 * y.e6 + x.e6 * y.e2 - x.e3 * y.e7 + x.e7 * y.e3
  e5 :=   x.e0 * y.e5 + x.e5 * y.e0 + x.e1 * y.e4 - x.e4 * y.e1
        - x.e2 * y.e7 + x.e7 * y.e2 + x.e3 * y.e6 - x.e6 * y.e3
  e6 :=   x.e0 * y.e6 + x.e6 * y.e0 + x.e1 * y.e7 - x.e7 * y.e1
        + x.e2 * y.e4 - x.e4 * y.e2 - x.e3 * y.e5 + x.e5 * y.e3
  e7 :=   x.e0 * y.e7 + x.e7 * y.e0 - x.e1 * y.e6 + x.e6 * y.e1
        + x.e2 * y.e5 - x.e5 * y.e2 + x.e3 * y.e4 - x.e4 * y.e3

-- ---------------------------------------------------------------------------
-- §5. Conjugate and norm
-- ---------------------------------------------------------------------------

def octConj (x : Octonion R) : Octonion R :=
  ⟨x.e0, -x.e1, -x.e2, -x.e3, -x.e4, -x.e5, -x.e6, -x.e7⟩

def octNormSq (x : Octonion R) : R :=
  x.e0^2 + x.e1^2 + x.e2^2 + x.e3^2 + x.e4^2 + x.e5^2 + x.e6^2 + x.e7^2

-- ---------------------------------------------------------------------------
-- §6. Extensionality
-- ---------------------------------------------------------------------------

@[ext]
theorem oct_ext (x y : Octonion R)
    (h0 : x.e0 = y.e0) (h1 : x.e1 = y.e1) (h2 : x.e2 = y.e2) (h3 : x.e3 = y.e3)
    (h4 : x.e4 = y.e4) (h5 : x.e5 = y.e5) (h6 : x.e6 = y.e6) (h7 : x.e7 = y.e7) :
    x = y := by
  cases x; cases y; simp_all

-- ---------------------------------------------------------------------------
-- §7. Addition laws
-- ---------------------------------------------------------------------------

theorem oct_add_comm (x y : Octonion R) : octAdd x y = octAdd y x := by
  simp only [octAdd]; ext <;> ring

theorem oct_add_assoc (x y z : Octonion R) :
    octAdd (octAdd x y) z = octAdd x (octAdd y z) := by
  simp only [octAdd]; ext <;> ring

theorem oct_add_zero (x : Octonion R) : octAdd x ⟨0,0,0,0,0,0,0,0⟩ = x := by
  simp only [octAdd]; ext <;> ring

theorem oct_zero_add (x : Octonion R) : octAdd ⟨0,0,0,0,0,0,0,0⟩ x = x := by
  simp only [octAdd]; ext <;> ring

theorem oct_add_neg (x : Octonion R) : octAdd x (octNeg x) = ⟨0,0,0,0,0,0,0,0⟩ := by
  simp only [octAdd, octNeg]; ext <;> ring

-- ---------------------------------------------------------------------------
-- §8. Scalar multiplication laws
-- ---------------------------------------------------------------------------

theorem oct_scalar_one (x : Octonion R) : octScale 1 x = x := by
  simp only [octScale]; ext <;> ring

theorem oct_scalar_zero (x : Octonion R) : octScale 0 x = ⟨0,0,0,0,0,0,0,0⟩ := by
  simp only [octScale]; ext <;> ring

theorem oct_scalar_add (m n : R) (x : Octonion R) :
    octScale (m + n) x = octAdd (octScale m x) (octScale n x) := by
  simp only [octScale, octAdd]; ext <;> ring

theorem oct_scalar_mul_assoc (m n : R) (x : Octonion R) :
    octScale (m * n) x = octScale m (octScale n x) := by
  simp only [octScale]; ext <;> ring

-- ---------------------------------------------------------------------------
-- §9. Multiplication distributes over addition
-- ---------------------------------------------------------------------------

theorem oct_mul_add_left (x y z : Octonion R) :
    octMul x (octAdd y z) = octAdd (octMul x y) (octMul x z) := by
  simp only [octMul, octAdd]; ext <;> ring

theorem oct_mul_add_right (x y z : Octonion R) :
    octMul (octAdd x y) z = octAdd (octMul x z) (octMul y z) := by
  simp only [octMul, octAdd]; ext <;> ring

-- ---------------------------------------------------------------------------
-- §10. Identity element
-- ---------------------------------------------------------------------------

theorem oct_mul_one (x : Octonion R) : octMul x e0 = x := by
  simp only [octMul, e0]; ext <;> ring

theorem oct_one_mul (x : Octonion R) : octMul e0 x = x := by
  simp only [octMul, e0]; ext <;> ring

-- ---------------------------------------------------------------------------
-- §11-12. Non-commutativity / non-associativity witnesses need R nontrivial
-- (e.g. 1 ≠ -1 in R), so are deferred to instantiations rather than stated
-- generically here.
-- ---------------------------------------------------------------------------

-- ---------------------------------------------------------------------------
-- §13. Alternative laws
-- ---------------------------------------------------------------------------

theorem oct_left_alternative (x y : Octonion R) :
    octMul x (octMul x y) = octMul (octMul x x) y := by
  simp only [octMul]; ext <;> ring

theorem oct_right_alternative (x y : Octonion R) :
    octMul (octMul y x) x = octMul y (octMul x x) := by
  simp only [octMul]; ext <;> ring

-- ---------------------------------------------------------------------------
-- §14. Flexibility
-- ---------------------------------------------------------------------------

theorem oct_flexibility (x y : Octonion R) :
    octMul x (octMul y x) = octMul (octMul x y) x := by
  simp only [octMul]; ext <;> ring

-- ---------------------------------------------------------------------------
-- §15. Moufang identities
-- ---------------------------------------------------------------------------

theorem oct_moufang_left (x y z : Octonion R) :
    octMul z (octMul x (octMul z y)) = octMul (octMul (octMul z x) z) y := by
  simp only [octMul]; ext <;> ring

theorem oct_moufang_right (x y z : Octonion R) :
    octMul (octMul (octMul x y) z) y = octMul x (octMul y (octMul z y)) := by
  simp only [octMul]; ext <;> ring

theorem oct_moufang_middle (x y z : Octonion R) :
    octMul (octMul x y) (octMul z x) = octMul x (octMul (octMul y z) x) := by
  simp only [octMul]; ext <;> ring

-- ---------------------------------------------------------------------------
-- §16. Scalar multiplication commutes with octMul
-- ---------------------------------------------------------------------------

theorem oct_scalar_comm (n : R) (x y : Octonion R) :
    octMul (octScale n x) y = octScale n (octMul x y) := by
  simp only [octMul, octScale]; ext <;> ring

theorem oct_scalar_comm_right (n : R) (x y : Octonion R) :
    octMul x (octScale n y) = octScale n (octMul x y) := by
  simp only [octMul, octScale]; ext <;> ring

-- ---------------------------------------------------------------------------
-- §17. Conjugate laws
-- ---------------------------------------------------------------------------

theorem oct_conj_antimultiplicative (x y : Octonion R) :
    octConj (octMul x y) = octMul (octConj y) (octConj x) := by
  simp only [octConj, octMul]; ext <;> ring

theorem oct_conj_involution (x : Octonion R) : octConj (octConj x) = x := by
  simp only [octConj]; ext <;> ring

theorem oct_conj_add_real (x : Octonion R) :
    octAdd x (octConj x) = ⟨2 * x.e0, 0, 0, 0, 0, 0, 0, 0⟩ := by
  simp only [octAdd, octConj]; ext <;> ring

theorem oct_mul_conj (x : Octonion R) :
    octMul x (octConj x) = ⟨octNormSq x, 0, 0, 0, 0, 0, 0, 0⟩ := by
  simp only [octMul, octConj, octNormSq]; ext <;> ring

theorem oct_conj_mul (x : Octonion R) :
    octMul (octConj x) x = ⟨octNormSq x, 0, 0, 0, 0, 0, 0, 0⟩ := by
  simp only [octMul, octConj, octNormSq]; ext <;> ring

-- ---------------------------------------------------------------------------
-- §18. Norm multiplicativity — the Degen eight-square identity, over any
-- commutative ring R (not just ℤ or ℝ).
-- ---------------------------------------------------------------------------

theorem oct_norm_multiplicative (x y : Octonion R) :
    octNormSq (octMul x y) = octNormSq x * octNormSq y := by
  simp only [octMul, octNormSq]; ring

-- ---------------------------------------------------------------------------
-- §19. Power laws
-- ---------------------------------------------------------------------------

theorem oct_sq_comm_left (x : Octonion R) :
    octMul x (octMul x x) = octMul (octMul x x) x := by
  simp only [octMul]; ext <;> ring

theorem oct_neg_sq_scalar_e1 : octMul (e1 (R := R)) e1 = octNeg e0 := by
  simp only [octMul, e1, e0, octNeg]; ext <;> ring

-- ---------------------------------------------------------------------------
-- §20. Algebraic instances
--
-- `Octonion R` is `AddCommGroup` + `Mul` with distributivity but *not*
-- associativity (`oct_nonassociative` witnesses that on Int; see
-- `../OctonionAlgebra.lean`), so the correct target class is
-- `NonUnitalNonAssocRing`, not `Ring` -- unlike `Quaternion.lean`'s
-- `instRing`, which can assume `mul_assoc`. `AddCommGroup` is built directly
-- (rather than transferred along an `Equiv` to a product type, the way
-- `QuaternionAlgebra.equivProd` does it, which needs `Add`/`Neg`/`Zero`
-- instances on the target to already exist before the transfer lemma even
-- applies) from the already-proven `oct_add_*` lemmas above, and the ring
-- axioms `Mul` itself doesn't give you (distributivity, `zero_mul`,
-- `mul_zero`) are supplied via `ring`.
-- ---------------------------------------------------------------------------

instance : Zero (Octonion R) := ⟨⟨0,0,0,0,0,0,0,0⟩⟩
instance : Add (Octonion R) := ⟨octAdd⟩
instance : Neg (Octonion R) := ⟨octNeg⟩

theorem add_def (x y : Octonion R) : x + y = octAdd x y := rfl
theorem neg_def (x : Octonion R) : -x = octNeg x := rfl
theorem zero_def : (0 : Octonion R) = ⟨0,0,0,0,0,0,0,0⟩ := rfl

instance : AddCommGroup (Octonion R) where
  nsmul := nsmulRec
  zsmul := zsmulRec
  add_assoc x y z := by simpa only [add_def] using oct_add_assoc x y z
  zero_add x := by simpa only [add_def, zero_def] using oct_zero_add x
  add_zero x := by simpa only [add_def, zero_def] using oct_add_zero x
  neg_add_cancel x := by
    simp only [add_def, neg_def, zero_def, octAdd, octNeg]; ext <;> ring
  add_comm x y := by simpa only [add_def] using oct_add_comm x y

instance : Mul (Octonion R) := ⟨octMul⟩

theorem mul_def (x y : Octonion R) : x * y = octMul x y := rfl

instance : NonUnitalNonAssocRing (Octonion R) where
  left_distrib := oct_mul_add_left
  right_distrib := oct_mul_add_right
  zero_mul x := by simp only [mul_def, octMul, zero_def]; ext <;> ring
  mul_zero x := by simp only [mul_def, octMul, zero_def]; ext <;> ring

instance : Star (Octonion R) := ⟨octConj⟩

theorem star_def (x : Octonion R) : star x = octConj x := rfl

instance : StarRing (Octonion R) where
  star_involutive := oct_conj_involution
  star_mul x y := by simpa only [star_def, mul_def] using oct_conj_antimultiplicative x y
  star_add x y := by simp only [star_def, octConj, add_def, octAdd]; ext <;> ring

-- ---------------------------------------------------------------------------
-- §21. Sanity check: generic Mathlib lemmas apply out of the box
--
-- Not load-bearing (the instances above already typecheck on their own),
-- but confirms `Octonion R` behaves like any other `NonUnitalNonAssocRing` +
-- `StarRing` to the rest of Mathlib, via its own general-purpose lemmas
-- rather than anything proved in this file.
-- ---------------------------------------------------------------------------

example (x y : Octonion ℤ) : x + y = y + x := add_comm x y
example (x : Octonion ℤ) : star (star x) = x := star_star x
example (x y : Octonion ℤ) : star (x * y) = star y * star x := star_mul x y
example (x : Octonion ℤ) : (0 : Octonion ℤ) * x = 0 := zero_mul x

end Sounio.OctonionGeneral
