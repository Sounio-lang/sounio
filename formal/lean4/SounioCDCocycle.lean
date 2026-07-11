/-
  SounioCDCocycle — the forward zero-divisor obstruction for ALL Cayley-Dickson levels (not just the
  dims verified by native_decide in SounioCDTowerSeam). The obstruction reduces to the OPERATOR
  identity L_i²=−I, i.e. σ(i,j)·σ(i,i⊕j)=−1 ∀j. On paper this closes by a simultaneous induction over
  the CD sign's four branches, using {L (=L²), R (=R²), antisym (basis units anticommute),
  diag (e_i²=−1)}. Here the CD sign is reformulated on explicit bit-lists `sgn` (MSB first; XOR is
  structural `xorL`, no Nat arithmetic), verified to agree with the Nat `cdSigma` at dim 16 and 32.

  PROVED for ALL n, Mathlib-free, no sorry, no native_decide (axioms: [propext, Quot.sound]):
    • diag        : e_i² = −1 for every imaginary basis unit.
    • cocycle_bundle : the 4-property conjunction {diag, antisym, L, R}, by a SINGLE simultaneous
        induction on the length of two equal-length bit-lists (bundle_aux over n). The bundling is
        essential — L's (true,false) head-case needs antisym at the SHIFTED pair (as, xorL as bs),
        which only a length-indexed ∀-pairs induction supplies.
    • Lsq         : L_i²=−I headline, i.e. sgn i j · sgn i (xorL i j) = −1  (corollary).
    • antisym     : basis units anticommute for ALL n (corollary; the standing open piece of #718).
  Cross-checked at width 4 and 5 by native_decide (agree4/agree5/xor_agree4).

  BRIDGE to the canonical Nat sign `cdSigma` (closes the native_decide-only agreement gap; the four
  new theorems are Mathlib-free, no sorry, no native_decide — axioms [propext, Quot.sound]):
    • isZ_bitsOf     : isZ (bitsOf n i) = true ↔ i = 0                          (∀ n, i < 2^n).
    • xorL_bitsOf    : xorL (bitsOf n i) (bitsOf n j) = bitsOf n (i ^^^ j)      (generalizes xor_agree4).
    • sgn_eq_cdSigma : sgn (bitsOf n i) (bitsOf n j) = cdSigma i j n            (∀ n ≥ 1; generalizes
        agree4/agree5 — n=0 is the unreachable dummy CD level where cdSigma 0 0 0 = -1 ≠ 1 = sgn [] []).
    • cdSigma_cocycle: cdSigma i j n · cdSigma i (i ^^^ j) n = -1              (L_i²=−I on the canonical
        Nat sign, ALL levels n and every imaginary i; via sgn_eq_cdSigma ∘ xorL_bitsOf ∘ isZ_bitsOf ∘ Lsq).
-/
namespace SounioCDCocycle
-- bit-list Cayley-Dickson sign, MSB first (head = top bit). Structural XOR, no Nat arithmetic.
def isZ : List Bool → Bool
  | [] => true
  | b :: bs => (!b) && isZ bs
def sgn : List Bool → List Bool → Int
  | [], [] => 1
  | [a], [b] => if a && b then -1 else 1
  | a :: as, b :: bs =>
      if isZ (a :: as) || isZ (b :: bs) then 1
      else
        match a, b with
        | false, false => sgn as bs
        | false, true  => sgn bs as
        | true,  false => - sgn as bs
        | true,  true  => if isZ bs then -1 else sgn bs as
  | _, _ => 0
termination_by a b => a.length + b.length
decreasing_by all_goals (simp_wf; omega)

def xorL : List Bool → List Bool → List Bool
  | a :: as, b :: bs => (xor a b) :: xorL as bs
  | _, _ => []
-- zeros of a given length
def zeros : Nat → List Bool
  | 0 => []
  | n+1 => false :: zeros n

-- sanity: matches the Nat cdSigma at small dims? convert Nat->bits (MSB first, fixed width) and compare.
def bitsOf : Nat → Nat → List Bool     -- width, value
  | 0, _ => []
  | w+1, v => (decide (v ≥ 2^w)) :: bitsOf w (v % 2^w)
def cdSigma (a b : Nat) : Nat → Int
  | 0 => -1
  | 1 => if a == 0 || b == 0 then 1 else -1
  | (n+2) =>
      if a == 0 || b == 0 then 1
      else
        let half := 2 ^ (n+1)
        if !(a ≥ half) && !(b ≥ half) then cdSigma (a%half) (b%half) (n+1)
        else if !(a ≥ half) && (b ≥ half) then cdSigma (b%half) (a%half) (n+1)
        else if (a ≥ half) && !(b ≥ half) then (if b%half == 0 then cdSigma (a%half) 0 (n+1) else - cdSigma (a%half) (b%half) (n+1))
        else (if b%half == 0 then - cdSigma 0 (a%half) (n+1) else cdSigma (b%half) (a%half) (n+1))
-- agreement at width 4 (dim 16): sgn on bit-lists == cdSigma on Nats
theorem agree4 : (List.range 16).all (fun i => (List.range 16).all (fun j =>
  sgn (bitsOf 4 i) (bitsOf 4 j) == cdSigma i j 4)) = true := by native_decide
theorem agree5 : (List.range 32).all (fun i => (List.range 32).all (fun j =>
  sgn (bitsOf 5 i) (bitsOf 5 j) == cdSigma i j 5)) = true := by native_decide
-- xorL matches Nat xor at width 4
theorem xor_agree4 : (List.range 16).all (fun i => (List.range 16).all (fun j =>
  xorL (bitsOf 4 i) (bitsOf 4 j) == bitsOf 4 (i ^^^ j))) = true := by native_decide

-- diag: e_i^2 = -1 for every imaginary basis unit, ALL n (structural induction).
theorem diag : ∀ i : List Bool, isZ i = false → sgn i i = -1
  | [], h => by simp [isZ] at h
  | [a], h => by cases a <;> simp_all [isZ, sgn]
  | a :: b :: as, h => by
      have hz : isZ (a :: b :: as) = false := h
      cases a with
      | false =>
        have hb : isZ (b :: as) = false := by simpa [isZ] using hz
        simp only [sgn, hz, Bool.or_self, if_false]
        exact diag (b :: as) hb
      | true =>
        by_cases hbz : isZ (b :: as) = true
        · simp [sgn, hz, hbz]
        · have hb : isZ (b :: as) = false := by simpa using hbz
          simp only [sgn, hz, Bool.or_self, hb]
          exact diag (b :: as) hb

-- ===== branch reduction lemmas =====
theorem sgn_ff (as bs : List Bool) (ha : isZ as = false) (hb : isZ bs = false) :
    sgn (false :: as) (false :: bs) = sgn as bs := by
  have hia : isZ (false :: as) = false := by simpa [isZ] using ha
  have hib : isZ (false :: bs) = false := by simpa [isZ] using hb
  match as, bs with
  | a' :: as', b' :: bs' =>
    simp only [sgn, hia, hib, Bool.or_self]
    rw [if_neg (by decide)]

theorem sgn_ft (as bs : List Bool) (ha : isZ as = false) :
    sgn (false :: as) (true :: bs) = sgn bs as := by
  have hia : isZ (false :: as) = false := by simpa [isZ] using ha
  have hib : isZ (true :: bs) = false := by simp [isZ]
  match as with
  | a' :: as' =>
    simp only [sgn, hia, hib, Bool.or_false]
    rw [if_neg (by decide)]

theorem sgn_tf (as bs : List Bool) (hb : isZ bs = false) :
    sgn (true :: as) (false :: bs) = - sgn as bs := by
  have hia : isZ (true :: as) = false := by simp [isZ]
  have hib : isZ (false :: bs) = false := by simpa [isZ] using hb
  match bs with
  | b' :: bs' =>
    simp only [sgn, hia, hib, Bool.false_or]
    rw [if_neg (by decide)]

theorem sgn_tt (as bs : List Bool) (hlen : as.length = bs.length) :
    sgn (true :: as) (true :: bs) = (if isZ bs then -1 else sgn bs as) := by
  cases as with
  | nil => cases bs with
    | nil => simp [sgn, isZ]
    | cons b' bs' => simp only [List.length_cons, List.length_nil] at hlen; omega
  | cons a' as' => cases bs with
    | nil => simp only [List.length_cons, List.length_nil] at hlen; omega
    | cons b' bs' =>
      have hia : isZ (true :: a' :: as') = false := by simp [isZ]
      have hib : isZ (true :: b' :: bs') = false := by simp [isZ]
      simp only [sgn, hia, hib, Bool.or_self]
      rw [if_neg (by decide)]

-- ===== xorL helpers =====
theorem xorL_length (as bs : List Bool) (h : as.length = bs.length) :
    (xorL as bs).length = as.length := by
  induction as generalizing bs with
  | nil => simp [xorL]
  | cons a as ih =>
    cases bs with
    | nil => simp only [List.length_cons, List.length_nil] at h; omega
    | cons b bs =>
      simp only [xorL, List.length_cons]
      rw [ih bs (by simpa using h)]

theorem xorL_comm (as bs : List Bool) : xorL as bs = xorL bs as := by
  induction as generalizing bs with
  | nil => cases bs <;> simp [xorL]
  | cons a as ih =>
    cases bs with
    | nil => simp [xorL]
    | cons b bs => simp only [xorL, ih bs, Bool.xor_comm]

theorem xorL_zero_right (x y : List Bool) (hl : x.length = y.length) (hy : isZ y = true) :
    xorL x y = x := by
  induction x generalizing y with
  | nil => simp [xorL]
  | cons a x ih =>
    cases y with
    | nil => simp only [List.length_cons, List.length_nil] at hl; omega
    | cons b y =>
      cases b with
      | true => simp [isZ] at hy
      | false =>
        simp only [isZ, Bool.not_false, Bool.true_and] at hy
        simp only [xorL, Bool.xor_false]
        rw [ih y (by simpa using hl) hy]

theorem xorL_zero_left (x y : List Bool) (hl : x.length = y.length) (hx : isZ x = true) :
    xorL x y = y := by
  rw [xorL_comm]; exact xorL_zero_right y x hl.symm hx

theorem xorL_isZ_iff (x y : List Bool) (hl : x.length = y.length) :
    isZ (xorL x y) = true ↔ x = y := by
  induction x generalizing y with
  | nil => cases y with
    | nil => simp [xorL, isZ]
    | cons b y => simp only [List.length_cons, List.length_nil] at hl; omega
  | cons a x ih =>
    cases y with
    | nil => simp only [List.length_cons, List.length_nil] at hl; omega
    | cons b y =>
      simp only [xorL, isZ]
      constructor
      · intro h
        rw [Bool.and_eq_true] at h
        obtain ⟨h1, h2⟩ := h
        have hab : a = b := by cases a <;> cases b <;> simp_all
        have hxy : x = y := (ih y (by simpa using hl)).mp h2
        rw [hab, hxy]
      · intro h
        have hab : a = b := by rw [List.cons.injEq] at h; exact h.1
        have hxy : x = y := by rw [List.cons.injEq] at h; exact h.2
        rw [Bool.and_eq_true]
        refine ⟨?_, (ih y (by simpa using hl)).mpr hxy⟩
        cases a <;> cases b <;> simp_all

theorem xorL_eq_self_iff (x y : List Bool) (hl : x.length = y.length) :
    xorL x y = x ↔ isZ y = true := by
  induction x generalizing y with
  | nil => cases y with
    | nil => simp [xorL, isZ]
    | cons b y => simp only [List.length_cons, List.length_nil] at hl; omega
  | cons a x ih =>
    cases y with
    | nil => simp only [List.length_cons, List.length_nil] at hl; omega
    | cons b y =>
      simp only [xorL, isZ, List.cons.injEq]
      constructor
      · intro h
        obtain ⟨h1, h2⟩ := h
        have hb : b = false := by cases a <;> cases b <;> simp_all
        rw [hb]
        simp only [Bool.not_false, Bool.true_and]
        exact (ih y (by simpa using hl)).mp h2
      · intro h
        rw [Bool.and_eq_true] at h
        obtain ⟨h1, h2⟩ := h
        have hb : b = false := by cases b <;> simp_all
        refine ⟨?_, (ih y (by simpa using hl)).mpr h2⟩
        rw [hb]; cases a <;> rfl

theorem isZ_unique (x y : List Bool) (hl : x.length = y.length)
    (hx : isZ x = true) (hy : isZ y = true) : x = y := by
  induction x generalizing y with
  | nil => cases y with | nil => rfl | cons b y => simp only [List.length_cons, List.length_nil] at hl; omega
  | cons a x ih =>
    cases y with
    | nil => simp only [List.length_cons, List.length_nil] at hl; omega
    | cons b y =>
      cases a with
      | true => simp [isZ] at hx
      | false => cases b with
        | true => simp [isZ] at hy
        | false =>
          simp only [isZ, Bool.not_false, Bool.true_and] at hx hy
          rw [ih y (by simpa using hl) hx hy]

-- ===== sgn value lemmas =====
theorem sgn_guard (a b : Bool) (as bs : List Bool) (hlen : as.length = bs.length)
    (hg : (isZ (a :: as) || isZ (b :: bs)) = true) :
    sgn (a :: as) (b :: bs) = 1 := by
  cases as with
  | nil => cases bs with
    | nil => cases a <;> cases b <;> simp_all [sgn, isZ]
    | cons b' bs' => simp only [List.length_cons, List.length_nil] at hlen; omega
  | cons a' as' => cases bs with
    | nil => simp only [List.length_cons, List.length_nil] at hlen; omega
    | cons b' bs' => cases a <;> cases b <;> (simp only [sgn]; rw [if_pos hg])

theorem sgn_isZ (x y : List Bool) (hl : x.length = y.length)
    (hz : isZ x = true ∨ isZ y = true) : sgn x y = 1 := by
  cases x with
  | nil => cases y with | nil => simp [sgn] | cons b bs => simp only [List.length_cons, List.length_nil] at hl; omega
  | cons a as => cases y with
    | nil => simp only [List.length_cons, List.length_nil] at hl; omega
    | cons b bs =>
      refine sgn_guard a b as bs (by simpa using hl) ?_
      rcases hz with h | h
      · rw [h]; rfl
      · rw [h]; simp

-- micro Int lemma
theorem nmn (p q : Int) : (-p) * (-q) = p * q := by
  rw [Int.neg_mul, Int.mul_neg, Int.neg_neg]

theorem bundle_aux : ∀ (n : Nat) (i j : List Bool), i.length = n → j.length = n →
    (isZ i = false → sgn i i = -1)
  ∧ (isZ i = false → isZ j = false → i ≠ j → sgn i j = - sgn j i)
  ∧ (isZ i = false → sgn i j * sgn i (xorL i j) = -1)
  ∧ (isZ i = false → sgn j i * sgn (xorL j i) i = -1) := by
  intro n
  induction n with
  | zero =>
    intro i j hi hj
    have ei : i = [] := by cases i with | nil => rfl | cons => simp only [List.length_cons, List.length_nil] at hi; omega
    have ej : j = [] := by cases j with | nil => rfl | cons => simp only [List.length_cons, List.length_nil] at hj; omega
    subst ei; subst ej
    refine ⟨?_, ?_, ?_, ?_⟩ <;> intro h <;> simp [isZ] at h
  | succ m ih =>
    intro i j hi hj
    cases i with
    | nil => simp only [List.length_cons, List.length_nil] at hi; omega
    | cons a as =>
    cases j with
    | nil => simp only [List.length_cons, List.length_nil] at hj; omega
    | cons b bs =>
    have has : as.length = m := by simpa using hi
    have hbs : bs.length = m := by simpa using hj
    have hlen : as.length = bs.length := by rw [has, hbs]
    have hxl : (xorL as bs).length = m := (xorL_length as bs hlen).trans has
    have hxl2 : (xorL bs as).length = m := (xorL_length bs as hlen.symm).trans hbs
    refine ⟨fun h => diag (a :: as) h, ?anti, ?L, ?R⟩
    -- ================= ANTISYM =================
    · intro hia hjb hne
      cases a <;> cases b
      · -- false false
        have hasf : isZ as = false := by simpa [isZ] using hia
        have hbsf : isZ bs = false := by simpa [isZ] using hjb
        have hasne : as ≠ bs := fun h => hne (by rw [h])
        rw [sgn_ff as bs hasf hbsf, sgn_ff bs as hbsf hasf]
        exact (ih as bs has hbs).2.1 hasf hbsf hasne
      · -- false true
        have hasf : isZ as = false := by simpa [isZ] using hia
        rw [sgn_ft as bs hasf, sgn_tf bs as hasf]; omega
      · -- true false
        have hbsf : isZ bs = false := by simpa [isZ] using hjb
        rw [sgn_tf as bs hbsf, sgn_ft bs as hbsf]
      · -- true true
        rw [sgn_tt as bs hlen, sgn_tt bs as hlen.symm]
        have hasne : as ≠ bs := fun h => hne (by rw [h])
        by_cases hbz : isZ bs = true
        · by_cases haz : isZ as = true
          · exact absurd (isZ_unique as bs hlen haz hbz) hasne
          · have haf : isZ as = false := by simpa using haz
            rw [if_pos hbz, if_neg haz]
            have h1 : sgn as bs = 1 := sgn_isZ as bs hlen (Or.inr hbz)
            omega
        · have hbsf : isZ bs = false := by simpa using hbz
          rw [if_neg hbz]
          by_cases haz : isZ as = true
          · rw [if_pos haz]
            have h1 : sgn bs as = 1 := sgn_isZ bs as hlen.symm (Or.inr haz)
            omega
          · have haf : isZ as = false := by simpa using haz
            rw [if_neg haz]
            exact (ih bs as hbs has).2.1 hbsf haf (fun h => hasne h.symm)
    -- ================= L =================
    · intro hia
      cases a <;> cases b
      · -- false false
        have hasf : isZ as = false := by simpa [isZ] using hia
        simp only [xorL, Bool.xor_false]
        by_cases hbz : isZ bs = true
        · have hxeq : xorL as bs = as := xorL_zero_right as bs hlen hbz
          rw [hxeq, sgn_isZ (false :: as) (false :: bs) (by simp [has, hbs]) (Or.inr (by simpa [isZ] using hbz)), diag (false :: as) hia]
          decide
        · have hbsf : isZ bs = false := by simpa using hbz
          rw [sgn_ff as bs hasf hbsf]
          by_cases hxz : isZ (xorL as bs) = true
          · have heq : as = bs := (xorL_isZ_iff as bs hlen).mp hxz
            rw [sgn_isZ (false :: as) (false :: xorL as bs) (by simp [has, hxl]) (Or.inr (by simpa [isZ] using hxz)), heq, diag bs hbsf]
            decide
          · have hxf : isZ (xorL as bs) = false := by simpa using hxz
            rw [sgn_ff as (xorL as bs) hasf hxf]
            exact (ih as bs has hbs).2.2.1 hasf
      · -- false true
        have hasf : isZ as = false := by simpa [isZ] using hia
        simp only [xorL, Bool.false_xor]
        rw [sgn_ft as bs hasf, sgn_ft as (xorL as bs) hasf, xorL_comm as bs]
        exact (ih as bs has hbs).2.2.2 hasf
      · -- true false
        simp only [xorL, Bool.xor_false]
        have hlen_ax : as.length = (xorL as bs).length := by rw [has, hxl]
        by_cases hbz : isZ bs = true
        · have hxeq : xorL as bs = as := xorL_zero_right as bs hlen hbz
          rw [sgn_isZ (true :: as) (false :: bs) (by simp [has, hbs]) (Or.inr (by simpa [isZ] using hbz)), hxeq, sgn_tt as as rfl]
          by_cases haz : isZ as = true
          · rw [if_pos haz]; decide
          · have haf : isZ as = false := by simpa using haz
            rw [if_neg haz, diag as haf]; decide
        · have hbsf : isZ bs = false := by simpa using hbz
          rw [sgn_tf as bs hbsf, sgn_tt as (xorL as bs) hlen_ax]
          by_cases hxz : isZ (xorL as bs) = true
          · have heq : as = bs := (xorL_isZ_iff as bs hlen).mp hxz
            rw [if_pos hxz, heq, diag bs hbsf]; decide
          · have hxf : isZ (xorL as bs) = false := by simpa using hxz
            rw [if_neg hxz]
            by_cases haz : isZ as = true
            · have hxeq : xorL as bs = bs := xorL_zero_left as bs hlen haz
              rw [hxeq, sgn_isZ as bs hlen (Or.inl haz), sgn_isZ bs as hlen.symm (Or.inr haz)]
              decide
            · have haf : isZ as = false := by simpa using haz
              have hne2 : as ≠ xorL as bs := by
                intro h
                have h2 : isZ bs = true := (xorL_eq_self_iff as bs hlen).mp h.symm
                rw [hbsf] at h2; simp at h2
              have hanti : sgn as (xorL as bs) = - sgn (xorL as bs) as :=
                (ih as (xorL as bs) has hxl).2.1 haf hxf hne2
              rw [show sgn (xorL as bs) as = - sgn as (xorL as bs) from by rw [hanti, Int.neg_neg], nmn]
              exact (ih as bs has hbs).2.2.1 haf
      · -- true true
        simp only [xorL, Bool.xor_self]
        by_cases hbz : isZ bs = true
        · have hxeq : xorL as bs = as := xorL_zero_right as bs hlen hbz
          rw [sgn_tt as bs hlen, if_pos hbz, hxeq]
          by_cases haz : isZ as = true
          · rw [sgn_isZ (true :: as) (false :: as) (by simp [has]) (Or.inr (by simpa [isZ] using haz))]
            decide
          · have haf : isZ as = false := by simpa using haz
            rw [sgn_tf as as haf, diag as haf]; decide
        · have hbsf : isZ bs = false := by simpa using hbz
          rw [sgn_tt as bs hlen, if_neg hbz]
          by_cases hxz : isZ (xorL as bs) = true
          · have heq : as = bs := (xorL_isZ_iff as bs hlen).mp hxz
            have haf : isZ as = false := by rw [heq]; exact hbsf
            rw [sgn_isZ (true :: as) (false :: xorL as bs) (by simp [has, hxl]) (Or.inr (by simpa [isZ] using hxz)), ← heq, diag as haf]
            decide
          · have hxf : isZ (xorL as bs) = false := by simpa using hxz
            rw [sgn_tf as (xorL as bs) hxf]
            by_cases haz : isZ as = true
            · have hxeq : xorL as bs = bs := xorL_zero_left as bs hlen haz
              rw [hxeq, sgn_isZ bs as hlen.symm (Or.inr haz), sgn_isZ as bs hlen (Or.inl haz)]
              decide
            · have haf : isZ as = false := by simpa using haz
              have hne3 : as ≠ bs := by
                intro h
                have h2 : isZ (xorL as bs) = true := (xorL_isZ_iff as bs hlen).mpr h
                rw [h2] at hxf; simp at hxf
              rw [(ih bs as hbs has).2.1 hbsf haf (fun h => hne3 h.symm), nmn]
              exact (ih as bs has hbs).2.2.1 haf
    -- ================= R =================
    · intro hia
      cases a <;> cases b
      · -- false false
        have hasf : isZ as = false := by simpa [isZ] using hia
        simp only [xorL, Bool.xor_false]
        by_cases hbz : isZ bs = true
        · have hxeq : xorL bs as = as := xorL_zero_left bs as hlen.symm hbz
          rw [sgn_isZ (false :: bs) (false :: as) (by simp [has, hbs]) (Or.inl (by simpa [isZ] using hbz)), hxeq, diag (false :: as) hia]
          decide
        · have hbsf : isZ bs = false := by simpa using hbz
          rw [sgn_ff bs as hbsf hasf]
          by_cases hxz : isZ (xorL bs as) = true
          · have heq : bs = as := (xorL_isZ_iff bs as hlen.symm).mp hxz
            rw [sgn_isZ (false :: xorL bs as) (false :: as) (by simp [has, hxl2]) (Or.inl (by simpa [isZ] using hxz)), ← heq, diag bs hbsf]
            decide
          · have hxf : isZ (xorL bs as) = false := by simpa using hxz
            rw [sgn_ff (xorL bs as) as hxf hasf]
            exact (ih as bs has hbs).2.2.2 hasf
      · -- false true
        have hasf : isZ as = false := by simpa [isZ] using hia
        simp only [xorL, Bool.xor_false]
        rw [sgn_tf bs as hasf, sgn_tf (xorL bs as) as hasf, nmn]
        exact (ih as bs has hbs).2.2.2 hasf
      · -- true false
        simp only [xorL, Bool.false_xor]
        have hlen_xa : (xorL bs as).length = as.length := by rw [hxl2, has]
        by_cases hbz : isZ bs = true
        · have hxeq : xorL bs as = as := xorL_zero_left bs as hlen.symm hbz
          rw [sgn_isZ (false :: bs) (true :: as) (by simp [has, hbs]) (Or.inl (by simpa [isZ] using hbz)), hxeq, sgn_tt as as rfl]
          by_cases haz : isZ as = true
          · rw [if_pos haz]; decide
          · have haf : isZ as = false := by simpa using haz
            rw [if_neg haz, diag as haf]; decide
        · have hbsf : isZ bs = false := by simpa using hbz
          rw [sgn_ft bs as hbsf, sgn_tt (xorL bs as) as hlen_xa]
          by_cases haz : isZ as = true
          · rw [if_pos haz, sgn_isZ as bs hlen (Or.inl haz)]; decide
          · have haf : isZ as = false := by simpa using haz
            rw [if_neg haz, xorL_comm bs as]
            exact (ih as bs has hbs).2.2.1 haf
      · -- true true
        simp only [xorL, Bool.xor_self]
        have hlen_xa : (xorL bs as).length = as.length := by rw [hxl2, has]
        rw [sgn_tt bs as hlen.symm]
        by_cases haz : isZ as = true
        · rw [if_pos haz]
          have hxeq : xorL bs as = bs := xorL_zero_right bs as hlen.symm haz
          rw [hxeq]
          by_cases hbz : isZ bs = true
          · rw [sgn_isZ (false :: bs) (true :: as) (by simp [has, hbs]) (Or.inl (by simpa [isZ] using hbz))]
            decide
          · have hbsf : isZ bs = false := by simpa using hbz
            rw [sgn_ft bs as hbsf, sgn_isZ as bs hlen (Or.inl haz)]
            decide
        · have haf : isZ as = false := by simpa using haz
          rw [if_neg haz]
          by_cases hxz : isZ (xorL bs as) = true
          · have heq : bs = as := (xorL_isZ_iff bs as hlen.symm).mp hxz
            rw [sgn_isZ (false :: xorL bs as) (true :: as) (by simp [has, hxl2]) (Or.inl (by simpa [isZ] using hxz)), heq, diag as haf]
            decide
          · have hxf : isZ (xorL bs as) = false := by simpa using hxz
            rw [sgn_ft (xorL bs as) as hxf, xorL_comm bs as]
            exact (ih as bs has hbs).2.2.1 haf

theorem cocycle_bundle : ∀ (i j : List Bool), i.length = j.length →
    (isZ i = false → sgn i i = -1)
  ∧ (isZ i = false → isZ j = false → i ≠ j → sgn i j = - sgn j i)
  ∧ (isZ i = false → sgn i j * sgn i (xorL i j) = -1)
  ∧ (isZ i = false → sgn j i * sgn (xorL j i) i = -1) :=
  fun i j h => bundle_aux i.length i j rfl h.symm

theorem Lsq (i j : List Bool) (h : i.length = j.length) (hi : isZ i = false) :
    sgn i j * sgn i (xorL i j) = -1 :=
  (cocycle_bundle i j h).2.2.1 hi

theorem antisym (i j : List Bool) (h : i.length = j.length)
    (hi : isZ i = false) (hj : isZ j = false) (hij : i ≠ j) :
    sgn i j = - sgn j i :=
  (cocycle_bundle i j h).2.1 hi hj hij

-- ==================================================================================
-- BRIDGE: the bit-list sign `sgn` equals the canonical Nat sign `cdSigma` for ALL n.
-- Then the cocycle identity L_i²=−I transfers to `cdSigma` for all n (∀ dims), not just
-- the native_decide anchors at dim 16/32.
-- ==================================================================================

-- (0) bitsOf has the advertised width.
theorem bitsOf_length : ∀ (n v : Nat), (bitsOf n v).length = n
  | 0, _ => rfl
  | w+1, v => by simp only [bitsOf, List.length_cons, bitsOf_length w (v % 2^w)]

-- (0') the head bit `decide (v ≥ 2^w)` of a width-(w+1) number equals `Nat.testBit v w`.
theorem testBit_eq_decide (w x : Nat) (h : x < 2^(w+1)) :
    x.testBit w = decide (x ≥ 2^w) := by
  by_cases hge : x ≥ 2^w
  · rw [Nat.testBit_of_two_pow_le_and_two_pow_add_one_gt hge h]; simp [hge]
  · have hlt : x < 2^w := by omega
    rw [Nat.testBit_lt_two_pow hlt]; simp [hge]

-- (0'') XOR commutes with truncation to the low w bits.
theorem xor_mod_two_pow (w i j : Nat) : (i % 2^w) ^^^ (j % 2^w) = (i ^^^ j) % 2^w := by
  apply Nat.eq_of_testBit_eq
  intro k
  simp only [Nat.testBit_xor, Nat.testBit_mod_two_pow]
  by_cases hk : k < w <;> simp [hk]

-- (1) isZ ∘ bitsOf detects the value 0.
theorem isZ_bitsOf : ∀ (n i : Nat), i < 2^n → (isZ (bitsOf n i) = true ↔ i = 0)
  | 0, i, hi => by
      have h0 : i = 0 := by simpa using hi
      subst h0; simp [bitsOf, isZ]
  | w+1, i, hi => by
      have hp : 0 < 2^w := Nat.two_pow_pos w
      have hmod : i % 2^w < 2^w := Nat.mod_lt _ hp
      have ih := isZ_bitsOf w (i % 2^w) hmod
      show isZ (decide (i ≥ 2^w) :: bitsOf w (i % 2^w)) = true ↔ i = 0
      simp only [isZ]
      by_cases hge : i ≥ 2^w
      · have hd : decide (i ≥ 2^w) = true := by simp [hge]
        rw [hd]; simp only [Bool.not_true, Bool.false_and]
        constructor
        · intro h; exact absurd h (by decide)
        · intro h; exfalso; omega
      · have hlt : i < 2^w := by omega
        have hd : decide (i ≥ 2^w) = false := by simp [hge]
        rw [hd]; simp only [Bool.not_false, Bool.true_and, ih]
        rw [Nat.mod_eq_of_lt hlt]

-- (2) xorL on bit-lists tracks Nat XOR (generalizes xor_agree4).
theorem xorL_bitsOf : ∀ (n i j : Nat), i < 2^n → j < 2^n →
    xorL (bitsOf n i) (bitsOf n j) = bitsOf n (i ^^^ j)
  | 0, _, _, _, _ => rfl
  | w+1, i, j, hi, hj => by
      have hp : 0 < 2^w := Nat.two_pow_pos w
      have hij : i ^^^ j < 2^(w+1) := Nat.xor_lt_two_pow hi hj
      have ih := xorL_bitsOf w (i % 2^w) (j % 2^w) (Nat.mod_lt _ hp) (Nat.mod_lt _ hp)
      show xorL (decide (i ≥ 2^w) :: bitsOf w (i % 2^w)) (decide (j ≥ 2^w) :: bitsOf w (j % 2^w))
          = decide (i ^^^ j ≥ 2^w) :: bitsOf w ((i ^^^ j) % 2^w)
      simp only [xorL]
      rw [ih, xor_mod_two_pow]
      congr 1
      -- head: xor (decide (i≥2^w)) (decide (j≥2^w)) = decide (i^^^j ≥ 2^w)
      rw [← testBit_eq_decide w i hi, ← testBit_eq_decide w j hj,
          ← testBit_eq_decide w (i ^^^ j) hij, Nat.testBit_xor]

-- (1') corollary: bitsOf of a nonzero value is not the isZ list.
theorem isZ_bitsOf_false (n i : Nat) (hi : i < 2^n) (h0 : i ≠ 0) :
    isZ (bitsOf n i) = false := by
  cases hb : isZ (bitsOf n i) with
  | false => rfl
  | true => exact absurd ((isZ_bitsOf n i hi).mp hb) h0

-- ===== cdSigma branch-reduction lemmas (mirror sgn_ff/ft/tf/tt on the Nat sign) =====
theorem cdSigma_zero_left (L b : Nat) : cdSigma 0 b (L+1) = 1 := by
  cases L <;> rfl

theorem cdSigma_ff (m i j : Nat) (hi0 : i ≠ 0) (hj0 : j ≠ 0)
    (hai : ¬ (i ≥ 2^(m+1))) (hbj : ¬ (j ≥ 2^(m+1))) :
    cdSigma i j (m+2) = cdSigma (i % 2^(m+1)) (j % 2^(m+1)) (m+1) := by
  simp only [cdSigma]
  rw [if_neg (by simp [hi0, hj0]), if_pos (by simp [hai, hbj])]

theorem cdSigma_ft (m i j : Nat) (hi0 : i ≠ 0) (hj0 : j ≠ 0)
    (hai : ¬ (i ≥ 2^(m+1))) (hbj : j ≥ 2^(m+1)) :
    cdSigma i j (m+2) = cdSigma (j % 2^(m+1)) (i % 2^(m+1)) (m+1) := by
  simp only [cdSigma]
  rw [if_neg (by simp [hi0, hj0]), if_neg (by simp [hbj]), if_pos (by simp [hai, hbj])]

theorem cdSigma_tf (m i j : Nat) (hi0 : i ≠ 0) (hj0 : j ≠ 0)
    (hai : i ≥ 2^(m+1)) (hbj : ¬ (j ≥ 2^(m+1))) :
    cdSigma i j (m+2) = - cdSigma (i % 2^(m+1)) (j % 2^(m+1)) (m+1) := by
  have hjlt : j < 2^(m+1) := by omega
  simp only [cdSigma]
  rw [if_neg (by simp [hi0, hj0]), if_neg (by simp [hai]), if_neg (by simp [hai]),
      if_pos (by simp [hai, hbj]), if_neg (by rw [Nat.mod_eq_of_lt hjlt]; simp [hj0])]

theorem cdSigma_tt_raw (m i j : Nat) (hi0 : i ≠ 0) (hj0 : j ≠ 0)
    (hai : i ≥ 2^(m+1)) (hbj : j ≥ 2^(m+1)) :
    cdSigma i j (m+2)
      = (if j % 2^(m+1) == 0 then - cdSigma 0 (i % 2^(m+1)) (m+1)
         else cdSigma (j % 2^(m+1)) (i % 2^(m+1)) (m+1)) := by
  simp only [cdSigma]
  rw [if_neg (by simp [hi0, hj0]), if_neg (by simp [hai]), if_neg (by simp [hai]),
      if_neg (by simp [hbj])]

-- (3) the bridge: `sgn` on bit-lists equals the canonical Nat sign `cdSigma`, ALL levels ≥ 1.
--     (n = 0 is genuinely FALSE: sgn [] [] = 1 but cdSigma 0 0 0 = -1, an unreachable dummy
--      level; the CD recursion bottoms at level 1, hence the `1 ≤ n` hypothesis.)
theorem sgn_eq_cdSigma_aux : ∀ (m i j : Nat), i < 2^(m+1) → j < 2^(m+1) →
    sgn (bitsOf (m+1) i) (bitsOf (m+1) j) = cdSigma i j (m+1)
  | 0, i, j, hi, hj => by
      have hi2 : i < 2 := by simpa using hi
      have hj2 : j < 2 := by simpa using hj
      have hi01 : i = 0 ∨ i = 1 := by omega
      have hj01 : j = 0 ∨ j = 1 := by omega
      rcases hi01 with rfl | rfl <;> rcases hj01 with rfl | rfl <;>
        (simp only [bitsOf, sgn, cdSigma] <;> decide)
  | m+1, i, j, hi, hj => by
      have hpos : 0 < 2^(m+1) := Nat.two_pow_pos (m+1)
      have hiLo : i % 2^(m+1) < 2^(m+1) := Nat.mod_lt i hpos
      have hjLo : j % 2^(m+1) < 2^(m+1) := Nat.mod_lt j hpos
      have eqi : bitsOf (m+2) i = decide (i ≥ 2^(m+1)) :: bitsOf (m+1) (i % 2^(m+1)) := rfl
      have eqj : bitsOf (m+2) j = decide (j ≥ 2^(m+1)) :: bitsOf (m+1) (j % 2^(m+1)) := rfl
      rw [eqi, eqj]
      by_cases hg : i = 0 ∨ j = 0
      · -- GUARD: one operand is zero → both signs are 1
        have hlen : (decide (i ≥ 2^(m+1)) :: bitsOf (m+1) (i % 2^(m+1))).length
                  = (decide (j ≥ 2^(m+1)) :: bitsOf (m+1) (j % 2^(m+1))).length := by
          simp only [List.length_cons, bitsOf_length]
        have hz : isZ (decide (i ≥ 2^(m+1)) :: bitsOf (m+1) (i % 2^(m+1))) = true
                ∨ isZ (decide (j ≥ 2^(m+1)) :: bitsOf (m+1) (j % 2^(m+1))) = true := by
          rcases hg with h | h
          · left;  rw [← eqi]; exact (isZ_bitsOf (m+2) i hi).mpr h
          · right; rw [← eqj]; exact (isZ_bitsOf (m+2) j hj).mpr h
        rw [sgn_isZ _ _ hlen hz]
        have hgb : (i == 0 || j == 0) = true := by
          rcases hg with h | h
          · subst h; rfl
          · subst h; exact Bool.or_true _
        rw [show cdSigma i j (m+2) = 1 from by simp only [cdSigma]; rw [if_pos hgb]]
      · -- non-guard: i ≠ 0 and j ≠ 0
        have hi0 : i ≠ 0 := fun h => hg (Or.inl h)
        have hj0 : j ≠ 0 := fun h => hg (Or.inr h)
        by_cases hai : i ≥ 2^(m+1)
        · by_cases hbj : j ≥ 2^(m+1)
          · -- true/true
            have hda : decide (i ≥ 2^(m+1)) = true := by simp [hai]
            have hdb : decide (j ≥ 2^(m+1)) = true := by simp [hbj]
            rw [hda, hdb]
            have hlen : (bitsOf (m+1) (i % 2^(m+1))).length
                      = (bitsOf (m+1) (j % 2^(m+1))).length := by
              rw [bitsOf_length, bitsOf_length]
            rw [sgn_tt _ _ hlen, cdSigma_tt_raw m i j hi0 hj0 hai hbj]
            by_cases hz : j % 2^(m+1) = 0
            · have hbljT : isZ (bitsOf (m+1) (j % 2^(m+1))) = true :=
                (isZ_bitsOf (m+1) _ hjLo).mpr hz
              have hzb : (j % 2^(m+1) == 0) = true := by rw [hz]; rfl
              rw [if_pos hbljT, if_pos hzb, cdSigma_zero_left m (i % 2^(m+1))]
            · have hbljF : isZ (bitsOf (m+1) (j % 2^(m+1))) = false :=
                isZ_bitsOf_false _ _ hjLo hz
              have hzb : ¬ ((j % 2^(m+1) == 0) = true) := by simp [hz]
              rw [if_neg (by rw [hbljF]; decide), if_neg hzb]
              exact sgn_eq_cdSigma_aux m (j % 2^(m+1)) (i % 2^(m+1)) hjLo hiLo
          · -- true/false
            have hda : decide (i ≥ 2^(m+1)) = true := by simp [hai]
            have hdb : decide (j ≥ 2^(m+1)) = false := by simp [hbj]
            rw [hda, hdb]
            have hjlt : j < 2^(m+1) := by omega
            have hblj : isZ (bitsOf (m+1) (j % 2^(m+1))) = false :=
              isZ_bitsOf_false _ _ hjLo (by rw [Nat.mod_eq_of_lt hjlt]; exact hj0)
            rw [sgn_tf _ _ hblj, cdSigma_tf m i j hi0 hj0 hai hbj,
                sgn_eq_cdSigma_aux m (i % 2^(m+1)) (j % 2^(m+1)) hiLo hjLo]
        · by_cases hbj : j ≥ 2^(m+1)
          · -- false/true
            have hda : decide (i ≥ 2^(m+1)) = false := by simp [hai]
            have hdb : decide (j ≥ 2^(m+1)) = true := by simp [hbj]
            rw [hda, hdb]
            have hilt : i < 2^(m+1) := by omega
            have hbli : isZ (bitsOf (m+1) (i % 2^(m+1))) = false :=
              isZ_bitsOf_false _ _ hiLo (by rw [Nat.mod_eq_of_lt hilt]; exact hi0)
            rw [sgn_ft _ _ hbli, cdSigma_ft m i j hi0 hj0 hai hbj]
            exact sgn_eq_cdSigma_aux m (j % 2^(m+1)) (i % 2^(m+1)) hjLo hiLo
          · -- false/false
            have hda : decide (i ≥ 2^(m+1)) = false := by simp [hai]
            have hdb : decide (j ≥ 2^(m+1)) = false := by simp [hbj]
            rw [hda, hdb]
            have hilt : i < 2^(m+1) := by omega
            have hjlt : j < 2^(m+1) := by omega
            have hbli : isZ (bitsOf (m+1) (i % 2^(m+1))) = false :=
              isZ_bitsOf_false _ _ hiLo (by rw [Nat.mod_eq_of_lt hilt]; exact hi0)
            have hblj : isZ (bitsOf (m+1) (j % 2^(m+1))) = false :=
              isZ_bitsOf_false _ _ hjLo (by rw [Nat.mod_eq_of_lt hjlt]; exact hj0)
            rw [sgn_ff _ _ hbli hblj, cdSigma_ff m i j hi0 hj0 hai hbj]
            exact sgn_eq_cdSigma_aux m (i % 2^(m+1)) (j % 2^(m+1)) hiLo hjLo

theorem sgn_eq_cdSigma (n i j : Nat) (hn : 1 ≤ n) (hi : i < 2^n) (hj : j < 2^n) :
    sgn (bitsOf n i) (bitsOf n j) = cdSigma i j n := by
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  exact sgn_eq_cdSigma_aux m i j hi hj

-- (4) THE PAYOFF: L_i²=−I on the canonical Nat sign, for ALL levels n and every imaginary i.
theorem cdSigma_cocycle (n i j : Nat) (hi : i < 2^n) (hj : j < 2^n) (hi0 : i ≠ 0) :
    cdSigma i j n * cdSigma i (i ^^^ j) n = -1 := by
  cases n with
  | zero =>
      have h1 : i < 1 := by simpa using hi
      exact absurd (by omega : i = 0) hi0
  | succ m =>
      have hn : 1 ≤ m + 1 := by omega
      have hij : i ^^^ j < 2^(m+1) := Nat.xor_lt_two_pow hi hj
      rw [← sgn_eq_cdSigma (m+1) i j hn hi hj,
          ← sgn_eq_cdSigma (m+1) i (i ^^^ j) hn hi hij,
          ← xorL_bitsOf (m+1) i j hi hj]
      exact Lsq (bitsOf (m+1) i) (bitsOf (m+1) j)
        (by rw [bitsOf_length, bitsOf_length]) (isZ_bitsOf_false (m+1) i hi hi0)

end SounioCDCocycle
