import SounioSqrtField
import SounioMultiquadRing

set_option maxHeartbeats 1000000

/-!
# SounioMultiquadHom — the QF → SqrtField numerator ring-homomorphism core

The deepest algebraic step of the `QF ↪ ℝ` programme, Mathlib-free. We build a minimal
finite-sum library (`fsum`) over an abstract `SqrtField`, define the numerator evaluation
`evalNum l = Σ_{i<16} ofInt(lᵢ) · r i`, and prove the **multiplicative core**

    evalNum (qmul x y).1 = mul (evalNum x.1) (evalNum y.1)

i.e. the 16-dimensional multiquadratic convolution `qmul` is sent to the product of the
radical-sums. The proof promotes the per-generator `generator_law`
(`rᵢ·rⱼ = ofNatProd(i∧j)·r_{i⊕j}`) to a full bilinear identity via the XOR reindexing
permutation `perm_range_xor` — with no `Finset` machinery. This is the den-free heart of the
eventual fraction homomorphism `φ : QF → F`.
-/

namespace SounioMultiquadHom

open SounioSqrt SounioSqrt.RootedField MultiquadRing

variable {R : RootedField}

/-! ## Local re-derivations of private `RootedField` helpers -/

private theorem add_zero_left (a : R.F) : R.add R.zero a = a := by rw [R.add_comm, R.add_zero]

private theorem add_left_cancel_local (a b c : R.F) (h : R.add a b = R.add a c) : b = c := by
  have h2 : R.add (R.neg a) (R.add a b) = R.add (R.neg a) (R.add a c) := by rw [h]
  rwa [← R.add_assoc, ← R.add_assoc, R.add_comm (R.neg a) a, R.add_neg, add_zero_left,
    add_zero_left] at h2

private theorem mul_zero_right (a : R.F) : R.mul a R.zero = R.zero := by
  have hd : R.add (R.mul a R.zero) (R.mul a R.zero) = R.add (R.mul a R.zero) R.zero := by
    rw [← R.left_distrib, add_zero_left, R.add_zero]
  exact add_left_cancel_local (R.mul a R.zero) _ _ hd

private theorem mul_zero_left (a : R.F) : R.mul R.zero a = R.zero := by
  rw [R.mul_comm, mul_zero_right]

private theorem mul4comm (a b c d : R.F) :
    R.mul (R.mul a b) (R.mul c d) = R.mul (R.mul a c) (R.mul b d) := by
  have hbc : R.mul b (R.mul c d) = R.mul c (R.mul b d) := by
    rw [← R.mul_assoc, R.mul_comm b c, R.mul_assoc]
  rw [R.mul_assoc, hbc, ← R.mul_assoc]

/-! ## Finite sum over a list (Mathlib-free `Finset.sum` substitute) -/

def fsum {ι : Type} (l : List ι) (f : ι → R.F) : R.F :=
  l.foldr (fun i acc => R.add (f i) acc) R.zero

theorem fsum_nil {ι} (f : ι → R.F) : fsum ([] : List ι) f = R.zero := rfl

theorem fsum_cons {ι} (a : ι) (l : List ι) (f : ι → R.F) :
    fsum (a :: l) f = R.add (f a) (fsum l f) := rfl

private theorem add4comm (a b c d : R.F) :
    R.add (R.add a b) (R.add c d) = R.add (R.add a c) (R.add b d) := by
  have hbc : R.add b (R.add c d) = R.add c (R.add b d) := by
    rw [← R.add_assoc, R.add_comm b c, R.add_assoc]
  rw [R.add_assoc, hbc, ← R.add_assoc]

theorem fsum_add {ι} (l : List ι) (f g : ι → R.F) :
    fsum l (fun i => R.add (f i) (g i)) = R.add (fsum l f) (fsum l g) := by
  induction l with
  | nil => exact (R.add_zero R.zero).symm
  | cons a l ih =>
    rw [fsum_cons, fsum_cons, fsum_cons, ih, add4comm]

theorem fsum_zero {ι} (l : List ι) : fsum l (fun _ => R.zero) = (R.zero : R.F) := by
  induction l with
  | nil => rfl
  | cons a l ih => rw [fsum_cons, ih]; exact R.add_zero R.zero

theorem fsum_congr {ι} (l : List ι) (f g : ι → R.F) (h : ∀ i ∈ l, f i = g i) :
    fsum l f = fsum l g := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    rw [fsum_cons, fsum_cons, h a (by simp),
      ih (fun i hi => h i (List.mem_cons_of_mem a hi))]

theorem mul_fsum_left {ι} (c : R.F) (l : List ι) (f : ι → R.F) :
    R.mul c (fsum l f) = fsum l (fun i => R.mul c (f i)) := by
  induction l with
  | nil => rw [fsum_nil, fsum_nil, mul_zero_right]
  | cons a l ih => rw [fsum_cons, fsum_cons, R.left_distrib, ih]

theorem mul_fsum_right {ι} (c : R.F) (l : List ι) (f : ι → R.F) :
    R.mul (fsum l f) c = fsum l (fun i => R.mul (f i) c) := by
  induction l with
  | nil => rw [fsum_nil, fsum_nil, mul_zero_left]
  | cons a l ih => rw [fsum_cons, fsum_cons, R.right_distrib, ih]

theorem fsum_mul_fsum {ι κ} (l : List ι) (m : List κ) (f : ι → R.F) (g : κ → R.F) :
    R.mul (fsum l f) (fsum m g) = fsum l (fun i => fsum m (fun j => R.mul (f i) (g j))) := by
  rw [mul_fsum_right]
  apply fsum_congr
  intro i _
  rw [mul_fsum_left]

theorem fsum_map {ι κ} (l : List ι) (φ : ι → κ) (f : κ → R.F) :
    fsum (l.map φ) f = fsum l (fun i => f (φ i)) := by
  induction l with
  | nil => rfl
  | cons a l ih => rw [List.map_cons, fsum_cons, fsum_cons, ih]

theorem fsum_perm {ι} {l l' : List ι} (f : ι → R.F) (h : l.Perm l') : fsum l f = fsum l' f := by
  induction h with
  | nil => rfl
  | cons a _ ih => rw [fsum_cons, fsum_cons, ih]
  | swap x y l =>
    rw [fsum_cons, fsum_cons, fsum_cons, fsum_cons, ← R.add_assoc, ← R.add_assoc,
      R.add_comm (f y) (f x)]
  | trans _ _ ih1 ih2 => rw [ih1, ih2]

theorem fsum_comm {ι κ} (l : List ι) (m : List κ) (h : ι → κ → R.F) :
    fsum l (fun i => fsum m (fun j => h i j)) = fsum m (fun j => fsum l (fun i => h i j)) := by
  induction l with
  | nil =>
    rw [fsum_nil,
      show (fun j => fsum ([] : List ι) (fun i => h i j)) = (fun _ : κ => (R.zero : R.F)) from rfl,
      fsum_zero]
  | cons a l ih =>
    rw [fsum_cons]
    simp only [fsum_cons]
    rw [ih, ← fsum_add]

/-! ## XOR reindexing of the range-16 sum -/

theorem fsum_xor (k : Nat) (hk : k < 16) (G : Nat → R.F) :
    fsum (List.range 16) G = fsum (List.range 16) (fun i => G (i ^^^ k)) := by
  rw [show fsum (List.range 16) (fun i => G (i ^^^ k))
        = fsum ((List.range 16).map (fun i => i ^^^ k)) G from (fsum_map _ _ _).symm]
  exact fsum_perm G (perm_range_xor k hk).symm

/-! ## `ofInt` over the integer fold-sum -/

theorem ofInt_fsum {ι} (l : List ι) (g : ι → Int) :
    ofInt (l.foldr (fun i acc => g i + acc) 0) = fsum l (fun i => (ofInt (g i) : R.F)) := by
  induction l with
  | nil => rfl
  | cons a l ih => rw [List.foldr_cons, ofInt_add, fsum_cons, ih]

theorem foldl_add_int {ι} (l : List ι) (g : ι → Int) (a : Int) :
    l.foldl (fun acc i => acc + g i) a = a + l.foldr (fun i acc => g i + acc) 0 := by
  induction l generalizing a with
  | nil => simp
  | cons x l ih => simp only [List.foldl_cons, List.foldr_cons]; rw [ih (a + g x)]; omega

/-! ## bcoeff bridge: the two `bcoeff` definitions agree -/

private theorem bcoeff_int_eq : ∀ m ∈ List.range 16,
    MultiquadRing.bcoeff m = Int.ofNat (SounioSqrt.RootedField.bcoeff m) := by decide

theorem ofNatProd_eq (m : Nat) (hm : m < 16) :
    (ofNatProd m : R.F) = ofInt (MultiquadRing.bcoeff m) := by
  rw [show MultiquadRing.bcoeff m = Int.ofNat (SounioSqrt.RootedField.bcoeff m)
        from bcoeff_int_eq m (List.mem_range.mpr hm)]
  rfl

/-! ## The numerator evaluation map -/

def evalNum (l : List Int) : R.F :=
  fsum (List.range 16) (fun i => R.mul (ofInt (gi l i)) (r i))

/-! ## Per-pair and per-coefficient lemmas -/

private theorem land_xor_lt (i idx : Nat) (hi : i < 16) : Nat.land i (i ^^^ idx) < 16 :=
  Nat.lt_of_le_of_lt Nat.and_le_left hi

private theorem xor_cancel (i idx : Nat) : i ^^^ (idx ^^^ i) = idx := by
  rw [Nat.xor_comm idx i, ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]

private theorem xor_cancel2 (i idx : Nat) : Nat.xor i (i ^^^ idx) = idx := by
  show i ^^^ (i ^^^ idx) = idx
  rw [← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]

/-- One reindexed inner term equals `ofInt (qmulTerm …)`. -/
private theorem W_eq (x y : QF) (i idx : Nat) (hi : i < 16) :
    R.mul (R.mul (ofInt (gi x.1 i)) (ofInt (gi y.1 (i ^^^ idx))))
          (ofNatProd (Nat.land i (i ^^^ idx)))
      = ofInt (qmulTerm x y i idx) := by
  rw [ofNatProd_eq (Nat.land i (i ^^^ idx)) (land_xor_lt i idx hi), ← ofInt_mul, ← ofInt_mul]
  rfl

/-- The qmul coefficient as an `fsum` of `ofInt` of its terms. -/
private theorem ofInt_qmulCoeff (x y : QF) (idx : Nat) :
    ofInt (qmulCoeff x y idx) = fsum (List.range 16) (fun i => (ofInt (qmulTerm x y i idx) : R.F)) := by
  unfold qmulCoeff
  rw [foldl_add_int, Int.zero_add, ofInt_fsum]

/-! ## The multiplicative core: `evalNum (qmul x y) = mul (evalNum x) (evalNum y)` -/

theorem evalNum_qmul (x y : QF) :
    (evalNum (qmul x y).1 : R.F) = R.mul (evalNum x.1) (evalNum y.1) := by
  symm
  unfold evalNum
  rw [fsum_mul_fsum]
  calc
    fsum (List.range 16) (fun i => fsum (List.range 16)
        (fun j => R.mul (R.mul (ofInt (gi x.1 i)) (r i)) (R.mul (ofInt (gi y.1 j)) (r j))))
      = fsum (List.range 16) (fun i => fsum (List.range 16)
          (fun j => R.mul (R.mul (R.mul (ofInt (gi x.1 i)) (ofInt (gi y.1 j)))
                                 (ofNatProd (Nat.land i j))) (r (Nat.xor i j)))) := by
        apply fsum_congr; intro i _; apply fsum_congr; intro j _
        rw [mul4comm, generator_law, ← R.mul_assoc]
    _ = fsum (List.range 16) (fun i => fsum (List.range 16)
          (fun idx => R.mul (R.mul (R.mul (ofInt (gi x.1 i)) (ofInt (gi y.1 (i ^^^ idx))))
                                   (ofNatProd (Nat.land i (i ^^^ idx)))) (r idx))) := by
        apply fsum_congr; intro i hi
        rw [fsum_xor i (List.mem_range.mp hi)]
        apply fsum_congr; intro idx _
        rw [Nat.xor_comm idx i, xor_cancel2 i idx]
    _ = fsum (List.range 16) (fun idx => fsum (List.range 16)
          (fun i => R.mul (R.mul (R.mul (ofInt (gi x.1 i)) (ofInt (gi y.1 (i ^^^ idx))))
                                 (ofNatProd (Nat.land i (i ^^^ idx)))) (r idx))) :=
        fsum_comm (List.range 16) (List.range 16) _
    _ = fsum (List.range 16) (fun idx =>
          R.mul (fsum (List.range 16)
            (fun i => R.mul (R.mul (ofInt (gi x.1 i)) (ofInt (gi y.1 (i ^^^ idx))))
                            (ofNatProd (Nat.land i (i ^^^ idx))))) (r idx)) := by
        apply fsum_congr; intro idx _
        rw [mul_fsum_right]
    _ = fsum (List.range 16) (fun idx =>
          R.mul (fsum (List.range 16) (fun i => ofInt (qmulTerm x y i idx))) (r idx)) := by
        apply fsum_congr; intro idx _
        congr 1
        apply fsum_congr; intro i hi
        exact W_eq x y i idx (List.mem_range.mp hi)
    _ = fsum (List.range 16) (fun idx => R.mul (ofInt (qmulCoeff x y idx)) (r idx)) := by
        apply fsum_congr; intro idx _
        rw [ofInt_qmulCoeff]
    _ = fsum (List.range 16) (fun idx => R.mul (ofInt (gi (qmul x y).1 idx)) (r idx)) := by
        apply fsum_congr; intro idx hidx
        rw [gi_eq_getElem (qmul x y).1 idx (by rw [qmul_list_len]; exact List.mem_range.mp hidx),
          qmul_getElem x y idx (List.mem_range.mp hidx)]

/-! ## The fraction homomorphism `φ : QF → F` and its ring-hom laws (guarded by `den ≠ 0`)

`φ(c, d) = (Σ cᵢ rᵢ) · inv(ofInt d)`. Under the well-formedness guard `den ≠ 0`, `φ` is a ring
homomorphism: it carries `qmul`/`qadd`/`qsub` to `mul`/`add`/`sub` in `F`. The multiplicative law
is exactly `evalNum_qmul` + the char-0 inverse toolkit (`sf_inv_mul_inv`); the additive laws are
the field fraction-addition identity. This is the den-aware completion of `evalNum_qmul`. -/

private theorem neg_zero_local : R.neg R.zero = (R.zero : R.F) := by
  rw [← add_zero_left (R.neg R.zero), R.add_neg]

private theorem neg_add_local (a b : R.F) : R.neg (R.add a b) = R.add (R.neg a) (R.neg b) := by
  apply add_left_cancel_local (R.add a b)
  rw [R.add_neg, add4comm, R.add_neg, R.add_neg, R.add_zero]

private theorem neg_mul_local (a b : R.F) : R.mul (R.neg a) b = R.neg (R.mul a b) := by
  apply add_left_cancel_local (R.mul a b)
  rw [← R.right_distrib, R.add_neg, mul_zero_left, R.add_neg]

private theorem ofInt_sub (a b : Int) :
    (ofInt (a - b) : R.F) = R.add (ofInt a) (R.neg (ofInt b)) := by
  rw [show a - b = a + (-b) from by omega, ofInt_add, ofInt_neg]

theorem fsum_neg {ι} (l : List ι) (f : ι → R.F) :
    fsum l (fun i => R.neg (f i)) = R.neg (fsum l f) := by
  induction l with
  | nil => exact neg_zero_local.symm
  | cons a l ih => rw [fsum_cons, ih, fsum_cons, neg_add_local]

theorem evalNum_mul_right (l : List Int) (c : R.F) :
    R.mul (evalNum l) c = fsum (List.range 16) (fun i => R.mul (R.mul (ofInt (gi l i)) c) (r i)) := by
  unfold evalNum
  rw [mul_fsum_right]
  apply fsum_congr; intro i _
  rw [R.mul_assoc, R.mul_comm (r i) c, ← R.mul_assoc]

private theorem qsub_list_len' (x y : QF) : (qsub x y).1.length = 16 := by
  simp [qsub, List.length_map, List.length_range]

private theorem qsub_getElem' (x y : QF) (i : Nat) (hi : i < 16) :
    (qsub x y).1[i]'(by rw [qsub_list_len']; exact hi) = gi x.1 i * y.2 - gi y.1 i * x.2 := by
  simp [qsub, gi, List.getElem_map, List.getElem_range (by simp; exact hi)]

theorem evalNum_qadd (x y : QF) :
    evalNum (qadd x y).1
      = R.add (R.mul (evalNum x.1) (ofInt y.2)) (R.mul (evalNum y.1) (ofInt x.2)) := by
  rw [evalNum_mul_right, evalNum_mul_right, ← fsum_add]
  unfold evalNum
  apply fsum_congr; intro i hi
  rw [show gi (qadd x y).1 i = gi x.1 i * y.2 + gi y.1 i * x.2 from by
        rw [gi_eq_getElem (qadd x y).1 i (by rw [qadd_list_len]; exact List.mem_range.mp hi),
          qadd_getElem x y i (List.mem_range.mp hi)],
      ofInt_add, ofInt_mul, ofInt_mul, R.right_distrib]

theorem evalNum_qsub (x y : QF) :
    evalNum (qsub x y).1
      = R.add (R.mul (evalNum x.1) (ofInt y.2)) (R.neg (R.mul (evalNum y.1) (ofInt x.2))) := by
  rw [evalNum_mul_right, evalNum_mul_right, ← fsum_neg, ← fsum_add]
  unfold evalNum
  apply fsum_congr; intro i hi
  rw [show gi (qsub x y).1 i = gi x.1 i * y.2 - gi y.1 i * x.2 from by
        rw [gi_eq_getElem (qsub x y).1 i (by rw [qsub_list_len']; exact List.mem_range.mp hi),
          qsub_getElem' x y i (List.mem_range.mp hi)],
      ofInt_sub, ofInt_mul, ofInt_mul, R.right_distrib, neg_mul_local]

theorem frac_add (a1 a2 d1 d2 : R.F) (h1 : d1 ≠ R.zero) (h2 : d2 ≠ R.zero) :
    R.mul (R.add (R.mul a1 d2) (R.mul a2 d1)) (R.inv (R.mul d1 d2))
      = R.add (R.mul a1 (R.inv d1)) (R.mul a2 (R.inv d2)) := by
  rw [sf_inv_mul_inv h1 h2, R.right_distrib]
  congr 1
  · rw [mul4comm, R.mul_inv d2 h2, R.mul_one]
  · rw [R.mul_comm (R.inv d1) (R.inv d2), mul4comm, R.mul_inv d1 h1, R.mul_one]

theorem frac_sub (a1 a2 d1 d2 : R.F) (h1 : d1 ≠ R.zero) (h2 : d2 ≠ R.zero) :
    R.mul (R.add (R.mul a1 d2) (R.neg (R.mul a2 d1))) (R.inv (R.mul d1 d2))
      = R.add (R.mul a1 (R.inv d1)) (R.neg (R.mul a2 (R.inv d2))) := by
  rw [sf_inv_mul_inv h1 h2, R.right_distrib]
  congr 1
  · rw [mul4comm, R.mul_inv d2 h2, R.mul_one]
  · rw [neg_mul_local, R.mul_comm (R.inv d1) (R.inv d2), mul4comm, R.mul_inv d1 h1, R.mul_one]

/-- The fraction map `φ(c, d) = (Σ cᵢ rᵢ) · inv(ofInt d)`. -/
def phi (x : QF) : R.F := R.mul (evalNum x.1) (R.inv (ofInt x.2))

theorem phi_qmul (x y : QF) (hx : x.2 ≠ 0) (hy : y.2 ≠ 0) :
    phi (qmul x y) = R.mul (phi x) (phi y) := by
  unfold phi
  rw [show (qmul x y).2 = x.2 * y.2 from rfl, ofInt_mul, evalNum_qmul,
    sf_inv_mul_inv (ofInt_ne_zero hx) (ofInt_ne_zero hy), mul4comm]

theorem phi_qadd (x y : QF) (hx : x.2 ≠ 0) (hy : y.2 ≠ 0) :
    phi (qadd x y) = R.add (phi x) (phi y) := by
  unfold phi
  rw [show (qadd x y).2 = x.2 * y.2 from rfl, ofInt_mul, evalNum_qadd]
  exact frac_add (evalNum x.1) (evalNum y.1) (ofInt x.2) (ofInt y.2)
    (ofInt_ne_zero hx) (ofInt_ne_zero hy)

theorem phi_qsub (x y : QF) (hx : x.2 ≠ 0) (hy : y.2 ≠ 0) :
    phi (qsub x y) = R.add (phi x) (R.neg (phi y)) := by
  unfold phi
  rw [show (qsub x y).2 = x.2 * y.2 from rfl, ofInt_mul, evalNum_qsub]
  exact frac_sub (evalNum x.1) (evalNum y.1) (ofInt x.2) (ofInt y.2)
    (ofInt_ne_zero hx) (ofInt_ne_zero hy)

/-- A summand-isolation lemma: if `f` vanishes off a single index `a ∈ l` (and `l` has no
    duplicates), the whole `fsum` collapses to `f a`. -/
theorem fsum_single {ι} (a : ι) :
    ∀ (l : List ι) (f : ι → R.F), a ∈ l → l.Nodup →
      (∀ i ∈ l, i ≠ a → f i = R.zero) → fsum l f = f a := by
  intro l
  induction l with
  | nil => intro f ha; exact absurd ha (by simp)
  | cons b t ih =>
    intro f ha hnd hz
    rw [fsum_cons]
    rcases List.nodup_cons.mp hnd with ⟨hbt, hndt⟩
    by_cases hab : a = b
    · subst hab
      have htz : fsum t f = R.zero := by
        have heq : fsum t f = fsum t (fun _ => R.zero) := by
          apply fsum_congr; intro i hi
          exact hz i (List.mem_cons_of_mem _ hi) (fun h => hbt (h ▸ hi))
        rw [heq, fsum_zero]
      rw [htz, R.add_zero]
    · have hbz : f b = R.zero := hz b (by simp) (fun h => hab h.symm)
      have hat : a ∈ t := (List.mem_cons.mp ha).resolve_left (fun h => hab h)
      rw [hbz, add_zero_left]
      exact ih f hat hndt (fun i hi hia => hz i (List.mem_cons_of_mem _ hi) hia)

/-- **`φ` is unital.** Any QF value that *represents* the rational `1` — coefficient `0` equal
    to the denominator, all other coefficients `0` — with nonzero denominator maps to `R.one`.
    This is the den-aware mathematical content of the `hunit` law of a guarded `QFTransfer`:
    `evalNum` collapses to `ofInt d.2` (only the `r 0 = 1` term survives), then `mul_inv` closes. -/
theorem phi_unit (d : QF) (h0 : gi d.1 0 = d.2)
    (hrest : ∀ i, i < 16 → i ≠ 0 → gi d.1 i = 0) (hd : d.2 ≠ 0) :
    phi d = R.one := by
  have hev : (evalNum d.1 : R.F) = ofInt d.2 := by
    unfold evalNum
    rw [fsum_single 0 (List.range 16) (fun i => R.mul (ofInt (gi d.1 i)) (r i))
        (by simp) List.nodup_range
        (fun i hi hi0 => by
          show R.mul (ofInt (gi d.1 i)) (r i) = R.zero
          rw [hrest i (List.mem_range.mp hi) hi0,
            show (ofInt (0 : Int) : R.F) = R.zero from rfl, mul_zero_left]),
      h0, r_zero, R.mul_one]
  unfold phi
  rw [hev, R.mul_inv _ (ofInt_ne_zero hd)]

#print axioms SounioMultiquadHom.evalNum_qmul
#print axioms SounioMultiquadHom.phi_unit
#print axioms SounioMultiquadHom.phi_qmul
#print axioms SounioMultiquadHom.phi_qadd
#print axioms SounioMultiquadHom.phi_qsub

#eval IO.println "SounioMultiquadHom: PROVED evalNum_qmul + fraction hom φ:QF→F as a UNITAL ring homomorphism — guarded laws phi_qmul/phi_qadd/phi_qsub + phi_unit (φ sends any QF representing 1 with nonzero denominator to R.one). Mathlib-free fsum library + generator_law + perm_range_xor reindex + char-0 inverse toolkit."
