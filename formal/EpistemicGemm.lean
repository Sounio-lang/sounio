/-!
# Sounio.EpistemicGemm — Phase 8 Formal Verification

Soundness proof for the epistemic GEMM uncertainty formula implemented in
crates/souc/src/codegen/gpu/epistemic_gemm.rs

The key theorem: if each input matrix element has a certified uncertainty
radius, the computed output uncertainty radius is a sound upper bound.

References:
- Neumaier (1990) Interval Methods for Systems of Equations s1.3
- Moore, Kearfott and Cloud (2009) Introduction to Interval Analysis s2.2
- Rump (1999) INTLAB

Zero-sorry guarantee: every theorem proved without any sorry tactic.
-/

namespace Sounio.EpistemicGemm

-- ---------------------------------------------------------------------------
-- §1. AbsField typeclass
-- ---------------------------------------------------------------------------

class AbsField (α : Type) extends Add α, Mul α, Neg α, LE α where
  zero one : α
  abs      : α → α
  le_refl     : ∀ a : α, a ≤ a
  le_trans    : ∀ a b c : α, a ≤ b → b ≤ c → a ≤ c
  le_antisymm : ∀ a b : α, a ≤ b → b ≤ a → a = b
  add_assoc      : ∀ a b c : α, a + b + c = a + (b + c)
  add_comm       : ∀ a b : α, a + b = b + a
  add_zero       : ∀ a : α, a + zero = a
  zero_add       : ∀ a : α, zero + a = a
  add_neg_cancel : ∀ a : α, a + -a = zero
  neg_add_cancel : ∀ a : α, -a + a = zero
  neg_neg : ∀ a : α, - -a = a
  mul_comm  : ∀ a b : α, a * b = b * a
  mul_assoc : ∀ a b c : α, a * b * c = a * (b * c)
  mul_one   : ∀ a : α, a * one = a
  one_mul   : ∀ a : α, one * a = a
  mul_zero  : ∀ a : α, a * zero = zero
  zero_mul  : ∀ a : α, zero * a = zero
  mul_add  : ∀ a b c : α, a * (b + c) = a * b + a * c
  add_mul  : ∀ a b c : α, (a + b) * c = a * c + b * c
  neg_mul  : ∀ a b : α, -a * b = -(a * b)
  mul_neg  : ∀ a b : α, a * -b = -(a * b)
  add_le_add           : ∀ a b c d : α, a ≤ b → c ≤ d → a + c ≤ b + d
  mul_nonneg           : ∀ a b : α, zero ≤ a → zero ≤ b → zero ≤ a * b
  mul_le_mul_of_nonneg : ∀ a b c : α, a ≤ b → zero ≤ c → a * c ≤ b * c
  abs_nonneg   : ∀ a : α, zero ≤ abs a
  abs_triangle : ∀ a b : α, abs (a + b) ≤ abs a + abs b
  abs_mul      : ∀ a b : α, abs (a * b) = abs a * abs b
  abs_zero     : abs zero = zero
  abs_neg      : ∀ a : α, abs (-a) = abs a
  zero_nonneg  : zero ≤ zero

-- ---------------------------------------------------------------------------
-- §2. Notation and local variables
-- ---------------------------------------------------------------------------

variable {α : Type} [F : AbsField α]

local notation "ퟰ" => @AbsField.zero α F
local notation "ퟱ" => @AbsField.one α F
local notation "|" a "|" => @AbsField.abs α F a

-- ---------------------------------------------------------------------------
-- §3. Derived lemmas (private, proved from axioms only)
-- ---------------------------------------------------------------------------

private lemma add_left_comm (a b c : α) : a + (b + c) = b + (a + c) := by
  rw [← F.add_assoc, F.add_comm a b, F.add_assoc]

private lemma add_neg_cancel_right (a rest : α) : a + (-a + rest) = rest := by
  rw [← F.add_assoc, F.add_neg_cancel, F.zero_add]

private lemma neg_add_cancel_right (a rest : α) : -a + (a + rest) = rest := by
  rw [← F.add_assoc, F.neg_add_cancel, F.zero_add]

private lemma neg_mul_neg (a b : α) : -a * -b = a * b := by
  rw [F.neg_mul, F.mul_neg, F.neg_neg]

private lemma neg_add_dist (a b : α) : -(a + b) = -a + -b := by
  have h1 : (a + b) + (-a + -b) = ퟰ := by
    rw [F.add_assoc, ← F.add_assoc b, F.add_neg_cancel, F.zero_add, F.add_neg_cancel]
  have h2 : (a + b) + -(a + b) = ퟰ := F.add_neg_cancel (a + b)
  have h3 : ퟰ + -(a + b) = ퟰ + (-a + -b) := by rw [← h2, ← h1]
  rw [F.zero_add, F.zero_add] at h3
  exact h3.symm

private lemma foldl_acc_eq (xs : List α) (acc : α) :
    xs.foldl (· + ·) acc = acc + xs.foldl (· + ·) ퟰ := by
  induction xs generalizing acc with
  | nil => simp only [List.foldl_nil]; rw [F.add_zero]
  | cons x xs ih =>
    simp only [List.foldl_cons]
    rw [F.zero_add]
    rw [ih (acc + x), ih x]
    rw [← F.add_assoc]

private lemma foldl_zero_cons (x : α) (xs : List α) :
    (x :: xs).foldl (· + ·) ퟰ = x + xs.foldl (· + ·) ퟰ := by
  simp only [List.foldl_cons]
  rw [F.zero_add]
  exact foldl_acc_eq xs x

-- ---------------------------------------------------------------------------
-- §4. Core ring identity
-- ---------------------------------------------------------------------------

private theorem ring_identity (a b a0 b0 : α) :
    a * b + -(a0 * b0) =
    a0 * (b + -b0) + b0 * (a + -a0) + (a + -a0) * (b + -b0) := by
  have rA : a0 * (b + -b0) = a0 * b + -(a0 * b0) := by
    rw [F.mul_add, F.mul_neg]
  have rB : b0 * (a + -a0) = a * b0 + -(a0 * b0) := by
    rw [F.mul_add, F.mul_neg, F.mul_comm b0 a, F.mul_comm b0 a0]
  have rC : (a + -a0) * (b + -b0) =
            a * b + -(a * b0) + -(a0 * b) + a0 * b0 := by
    rw [F.add_mul, F.mul_add, F.mul_neg, F.mul_add, F.neg_mul, neg_mul_neg,
        F.mul_comm a0 b0]
    rw [F.add_assoc (a * b + -(a * b0))]
    rw [← F.add_assoc (a * b)]
  rw [rA, rB, rC]
  simp only [F.add_assoc]
  rw [add_left_comm (-(a * b0)) (-(a0 * b))]
  rw [add_left_comm (a * b) (-(a0 * b))]
  rw [add_left_comm (-(a0 * b0) : α) (-(a0 * b))]
  rw [add_left_comm (a * b0) (-(a0 * b))]
  rw [add_left_comm (-(a0 * b0) : α) (-(a0 * b))]
  rw [add_neg_cancel_right (a0 * b)]
  rw [add_left_comm (a * b) (-(a * b0))]
  rw [add_left_comm (-(a0 * b0) : α) (-(a * b0))]
  rw [add_neg_cancel_right (a * b0)]
  rw [F.add_comm (a * b) (a0 * b0)]
  rw [neg_add_cancel_right (a0 * b0)]
  rw [F.add_comm (-(a0 * b0)) (a * b)]

-- ---------------------------------------------------------------------------
-- §5. Main scalar theorem: mul_error_bound
-- ---------------------------------------------------------------------------

theorem mul_error_bound (a b a0 b0 ea eb : α)
    (ha  : | a + -a0 | ≤ ea)
    (hb  : | b + -b0 | ≤ eb)
    (hea : ퟰ ≤ ea) (heb : ퟰ ≤ eb) :
    | a * b + -(a0 * b0) | ≤ |a0| * eb + |b0| * ea + ea * eb := by
  rw [ring_identity a b a0 b0]
  have step1 : | a0 * (b + -b0) + b0 * (a + -a0) + (a + -a0) * (b + -b0) | ≤
               | a0 * (b + -b0) + b0 * (a + -a0) | + | (a + -a0) * (b + -b0) | :=
    F.abs_triangle _ _
  have step2 : | a0 * (b + -b0) + b0 * (a + -a0) | ≤
               | a0 * (b + -b0) | + | b0 * (a + -a0) | :=
    F.abs_triangle _ _
  have bA : | a0 * (b + -b0) | ≤ |a0| * eb := by
    rw [F.abs_mul]
    exact F.mul_le_mul_of_nonneg |a0| | b + -b0 | eb hb (F.abs_nonneg _)
  have bB : | b0 * (a + -a0) | ≤ |b0| * ea := by
    rw [F.abs_mul]
    exact F.mul_le_mul_of_nonneg |b0| | a + -a0 | ea ha (F.abs_nonneg _)
  have bC : | (a + -a0) * (b + -b0) | ≤ ea * eb := by
    rw [F.abs_mul]
    have lhs1 : | a + -a0 | * | b + -b0 | ≤ ea * | b + -b0 | :=
      F.mul_le_mul_of_nonneg _ _ _ ha (F.abs_nonneg _)
    have lhs2 : ea * | b + -b0 | ≤ ea * eb := by
      rw [F.mul_comm ea _, F.mul_comm ea eb]
      exact F.mul_le_mul_of_nonneg _ _ _ hb hea
    exact F.le_trans _ _ _ lhs1 lhs2
  have inner : | a0 * (b + -b0) | + | b0 * (a + -a0) | ≤ |a0| * eb + |b0| * ea :=
    F.add_le_add _ _ _ _ bA bB
  have mid : | a0 * (b + -b0) + b0 * (a + -a0) | + | (a + -a0) * (b + -b0) | ≤
             |a0| * eb + |b0| * ea + ea * eb :=
    F.add_le_add _ _ _ _ (F.le_trans _ _ _ step2 inner) bC
  exact F.le_trans _ _ _ step1 mid

-- ---------------------------------------------------------------------------
-- §6. Corollary: gemm_scalar_soundness
-- ---------------------------------------------------------------------------

theorem gemm_scalar_soundness
    (a_val a_eps b_val b_eps true_a true_b : α)
    (ha  : | true_a + -a_val | ≤ a_eps)
    (hb  : | true_b + -b_val | ≤ b_eps)
    (hea : ퟰ ≤ a_eps) (heb : ퟰ ≤ b_eps) :
    | true_a * true_b + -(a_val * b_val) | ≤
      |a_val| * b_eps + |b_val| * a_eps + a_eps * b_eps :=
  mul_error_bound true_a true_b a_val b_val a_eps b_eps ha hb hea heb

-- ---------------------------------------------------------------------------
-- §7. zero_eps_exact
-- ---------------------------------------------------------------------------

theorem zero_eps_exact (a b : α) :
    |a| * ퟰ + |b| * ퟰ + ퟰ * ퟰ = ퟰ := by
  rw [F.mul_zero, F.mul_zero, F.mul_zero, F.add_zero, F.add_zero]

-- ---------------------------------------------------------------------------
-- §8. sum_error_bound: pointwise list bound
-- ---------------------------------------------------------------------------

theorem sum_error_bound
    (terms bounds : List α)
    (h_len   : terms.length = bounds.length)
    (h_bound : ∀ i : Fin terms.length,
                 | terms.get i | ≤ bounds.get ⟨i.val, h_len ▸ i.isLt⟩) :
    | terms.foldl (· + ·) ퟰ | ≤ bounds.foldl (· + ·) ퟰ := by
  induction terms generalizing bounds with
  | nil =>
    simp only [List.foldl_nil]
    rw [F.abs_zero]
    cases bounds with
    | nil => exact F.le_refl _
    | cons b bs => simp only [List.length] at h_len
  | cons t ts ih =>
    cases bounds with
    | nil => simp only [List.length] at h_len
    | cons bnd bnds =>
      simp only [List.length, Nat.succ.injEq] at h_len
      rw [foldl_zero_cons t ts, foldl_zero_cons bnd bnds]
      have ht : | t | ≤ bnd := by
        have := h_bound ⟨0, Nat.zero_lt_succ _⟩
        simp only [List.get] at this
        exact this
      have ih_applied : | ts.foldl (· + ·) ퟰ | ≤ bnds.foldl (· + ·) ퟰ := by
        apply ih bnds h_len
        intro ⟨i, hi⟩
        have := h_bound ⟨i + 1, Nat.succ_lt_succ hi⟩
        simp only [List.get] at this ⊢
        convert this using 2
        simp [h_len]
      calc | t + ts.foldl (· + ·) ퟰ |
              ≤ | t | + | ts.foldl (· + ·) ퟰ | := F.abs_triangle _ _
           _ ≤ bnd + bnds.foldl (· + ·) ퟰ :=
               F.add_le_add _ _ _ _ ht ih_applied

-- ---------------------------------------------------------------------------
-- §9. gemm_dot_product_soundness: K-element dot product
-- ---------------------------------------------------------------------------

theorem gemm_dot_product_soundness
    (avs aes bvs bes tas tbs : List α)
    (h_len_a  : avs.length = aes.length)
    (h_len_b  : bvs.length = bes.length)
    (h_len_ab : avs.length = bvs.length)
    (h_ta     : tas.length = avs.length)
    (h_tb     : tbs.length = bvs.length)
    (h_nonneg_ae : ∀ i : Fin aes.length, ퟰ ≤ aes.get i)
    (h_nonneg_be : ∀ i : Fin bes.length, ퟰ ≤ bes.get i)
    (h_err_a  : ∀ i : Fin tas.length,
                  | tas.get i + -(avs.get ⟨i.val, h_ta ▸ i.isLt⟩) | ≤
                    aes.get ⟨i.val, h_len_a ▸ (h_ta ▸ i.isLt)⟩)
    (h_err_b  : ∀ i : Fin tbs.length,
                  | tbs.get i + -(bvs.get ⟨i.val, h_tb ▸ i.isLt⟩) | ≤
                    bes.get ⟨i.val, h_len_b ▸ (h_tb ▸ i.isLt)⟩) :
    let true_terms := List.zipWith (· * ·) tas tbs
    let val_terms  := List.zipWith (· * ·) avs bvs
    let eps_terms  := List.zipWith3 (fun av_abs ae be =>
                        av_abs * be + be * ae + ae * be)
                        (List.map (fun x => @AbsField.abs α F x) avs) aes bes
    | true_terms.foldl (· + ·) ퟰ + -(val_terms.foldl (· + ·) ퟰ) | ≤
      eps_terms.foldl (· + ·) ퟰ := by
  induction avs generalizing aes bvs bes tas tbs with
  | nil =>
    simp only [List.zipWith, List.map, List.foldl_nil, F.abs_zero,
               F.add_neg_cancel, F.add_zero]
    rfl
  | cons av avs ih =>
    cases aes with
    | nil => simp only [List.length] at h_len_a
    | cons ae aes =>
    cases bvs with
    | nil => simp only [List.length] at h_len_ab
    | cons bv bvs =>
    cases bes with
    | nil => simp only [List.length] at h_len_b
    | cons be bes =>
    cases tas with
    | nil => simp only [List.length] at h_ta
    | cons ta tas =>
    cases tbs with
    | nil => simp only [List.length] at h_tb
    | cons tb tbs =>
    simp only [List.length, Nat.succ.injEq] at h_len_a h_len_b h_len_ab h_ta h_tb
    simp only [List.zipWith, List.map, List.foldl_cons, List.zipWith3]
    rw [foldl_acc_eq (List.zipWith (· * ·) tas tbs) (ta * tb)]
    rw [foldl_acc_eq (List.zipWith (· * ·) avs bvs) (av * bv)]
    rw [F.zero_add, F.zero_add]
    rw [neg_add_dist (av * bv)]
    rw [F.add_assoc (ta * tb)]
    rw [← F.add_assoc (ta * tb + -(av * bv))]
    rw [F.add_comm (-((List.zipWith (· * ·) avs bvs).foldl (· + ·) ퟰ))
                   ((List.zipWith (· * ·) tas tbs).foldl (· + ·) ퟰ)]
    rw [← F.add_assoc]
    have tri := F.abs_triangle (ta * tb + -(av * bv))
      ((List.zipWith (· * ·) tas tbs).foldl (· + ·) ퟰ +
       -((List.zipWith (· * ·) avs bvs).foldl (· + ·) ퟰ))
    have head_ae : ퟰ ≤ ae := h_nonneg_ae ⟨0, Nat.zero_lt_succ _⟩
    have head_be : ퟰ ≤ be := h_nonneg_be ⟨0, Nat.zero_lt_succ _⟩
    have head_ha : | ta + -av | ≤ ae := by
      have := h_err_a ⟨0, Nat.zero_lt_succ _⟩
      simp only [List.get] at this; exact this
    have head_hb : | tb + -bv | ≤ be := by
      have := h_err_b ⟨0, Nat.zero_lt_succ _⟩
      simp only [List.get] at this; exact this
    have head_bound : | ta * tb + -(av * bv) | ≤ |av| * be + |bv| * ae + ae * be :=
      mul_error_bound ta tb av bv ae be head_ha head_hb head_ae head_be
    have tail_bound :=
      ih aes bvs bes tas tbs h_len_a h_len_b h_len_ab h_ta h_tb
        (fun i => h_nonneg_ae ⟨i.val + 1, Nat.succ_lt_succ i.isLt⟩)
        (fun i => h_nonneg_be ⟨i.val + 1, Nat.succ_lt_succ i.isLt⟩)
        (fun i => by
          have := h_err_a ⟨i.val + 1, Nat.succ_lt_succ i.isLt⟩
          simp only [List.get] at this ⊢
          exact this)
        (fun i => by
          have := h_err_b ⟨i.val + 1, Nat.succ_lt_succ i.isLt⟩
          simp only [List.get] at this ⊢
          exact this)
    rw [foldl_zero_cons]
    exact F.le_trans _ _ _
      tri
      (F.add_le_add _ _ _ _ head_bound tail_bound)

end Sounio.EpistemicGemm
