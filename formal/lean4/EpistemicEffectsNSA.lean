/-!
# EpistemicEffectsNSA — Anti-Garbling with TWO certificates, one calculus

The fusion theorem: an uncertainty-typed operator calculus over the Cayley–Dickson
algebra `CD(n)` (n=0 ℝ, 1 ℂ, 2 ℍ, 3 𝕆, 4 𝕊, …) in which a Knowledge type carries
TWO support over-approximations,

  * `N` — the noise-symbol support (axis 2, the NS certificate of `EpistemicEffectsNS`), and
  * `Q` — the basis-element support (axis 1, the NEW associativity certificate),

and the two identity-imposing operations a compiler performs each check ONE of them:

  * combining (`kadd`/`kmul`, the independence assumption) checks `nsDisjoint N₁ N₂`;
  * re-associating (`(xy)z ↦ x(yz)`, the associativity assumption) checks `assocCert Qx Qy Qz`.

Theorems (all sorry-free, Mathlib-free, `decide` only — no `native_decide`):

  * Axis 2 generalized to every `CD(n)` (`typed_agfree`, `exact_preservation`,
    `soundness_star`) for the SENSITIVITY propagator `gMulMeta`.
  * Axis 1: `assoc_zero_of_cert` — the type-level certificate forces the associator of
    every covered triple to vanish (support induction + trilinearity + 7 Fano lines /
    full associativity for n ≤ 2, kernel-decided); `reassoc_sound` — a certified
    re-association preserves value, true form and reported variance; `reassoc_payload_gap`
    and `reassoc_sensitivity_gap` — the uncertified gap is EXACTLY the associator
    (value) and the SUM OF THREE ASSOCIATORS (per-source sensitivity), in the calculus.
  * Orthogonality, syntactic: `nsDisjoint_reassoc_invariant` (axis-2 premises are
    invariant under axis-1 rewrites) and the two kernel witnesses that fail exactly one
    certificate each (`w1_*`: NS-clean, non-Fano; `w2_*`: associative, NS-rejected).
  * The THIRD axis (a correction to the 2026-08-23 "exactly two" claim): the GUM
    variance SHORTCUT `‖y‖²·Var x + ‖x‖²·Var y` imposes norm-multiplicativity, which
    fails beyond Hurwitz. `sed_shortcut_understates` — at n=4, disjoint sources, a single
    product, no re-association: shortcut reports 4, true first-order variance is 8. The
    sensitivity propagator reports 8. So "(C) exactly two" is a theorem for propagators
    that carry sensitivities, and FALSE for variance-only propagators over non-composition
    algebras.

Companion: docs/research/ANTIGARBLING_FUSION_THEOREM_2026-09-01.md
-/

namespace Sounio.EpistemicEffectsNSA

-- ================================================================
-- §A. The Cayley–Dickson carrier CD(n) on integer coordinates
-- ================================================================

/-- Cayley–Dickson sign: `e_a · e_b = cdSigma a b n · e_{a ⊕ b}`. Same function as
    `SounioCayleyDickson.cdSigma` / `cd_sigma_ct`, written by STRUCTURAL recursion on the
    bit-width so the kernel can unfold it under `decide` (no `native_decide` on this host). -/
def cdSigma (a b : Nat) : Nat → Int
  | 0 => if a = 0 ∨ b = 0 then 1 else -1
  | 1 => if a = 0 ∨ b = 0 then 1 else -1
  | bits+2 =>
    if a = 0 ∨ b = 0 then 1
    else
      let half := 2 ^ (bits + 1)
      let aLo := a % half
      let bLo := b % half
      if a < half ∧ b < half then cdSigma aLo bLo (bits + 1)
      else if a < half ∧ ¬ b < half then cdSigma bLo aLo (bits + 1)
      else if ¬ a < half ∧ b < half then
        if bLo = 0 then cdSigma aLo bLo (bits + 1) else -(cdSigma aLo bLo (bits + 1))
      else
        if bLo = 0 then -(cdSigma bLo aLo (bits + 1)) else cdSigma bLo aLo (bits + 1)

/-- Coordinate vectors; only indices `< 2^n` are live at level `n`. -/
abbrev Vec := Nat → Int

def sumR : Nat → (Nat → Int) → Int
  | 0, _ => 0
  | m+1, f => sumR m f + f m

theorem sumR_add (m : Nat) (f g : Nat → Int) :
    sumR m (fun i => f i + g i) = sumR m f + sumR m g := by
  induction m with
  | zero => rfl
  | succ m ih => simp only [sumR]; rw [ih]; omega

theorem sumR_smul (m : Nat) (c : Int) (f : Nat → Int) :
    sumR m (fun i => c * f i) = c * sumR m f := by
  induction m with
  | zero => simp [sumR]
  | succ m ih => simp only [sumR]; rw [ih, Int.mul_add]

theorem sumR_congr (m : Nat) (f g : Nat → Int) (h : ∀ i, i < m → f i = g i) :
    sumR m f = sumR m g := by
  induction m with
  | zero => rfl
  | succ m ih =>
    simp only [sumR]
    rw [ih (fun i hi => h i (Nat.lt_succ_of_lt hi)), h m (Nat.lt_succ_self m)]

theorem sumR_zero (m : Nat) : sumR m (fun _ => 0) = 0 := by
  induction m with
  | zero => rfl
  | succ m ih => simp only [sumR]; rw [ih]; rfl

theorem sumR_eq_zero_of (m : Nat) (f : Nat → Int) (h : ∀ i, i < m → f i = 0) : sumR m f = 0 := by
  rw [sumR_congr m f (fun _ => 0) h]; exact sumR_zero m

theorem sumR_ne_zero_exists (m : Nat) (f : Nat → Int) (h : sumR m f ≠ 0) :
    ∃ i, i < m ∧ f i ≠ 0 := by
  induction m with
  | zero => exact absurd rfl h
  | succ m ih =>
    simp only [sumR] at h
    by_cases hm : f m = 0
    · rw [hm, Int.add_zero] at h
      rcases ih h with ⟨i, hi, hf⟩
      exact ⟨i, Nat.lt_succ_of_lt hi, hf⟩
    · exact ⟨m, Nat.lt_succ_self m, hm⟩

def vzero : Vec := fun _ => 0
def vadd (a b : Vec) : Vec := fun k => a k + b k
def vsub (a b : Vec) : Vec := fun k => a k - b k
def smul (c : Int) (a : Vec) : Vec := fun k => c * a k
/-- Basis vector `e_i`. -/
def e (i : Nat) : Vec := fun k => if k = i then 1 else 0

theorem vadd_zero (a : Vec) : vadd a vzero = a := by funext k; simp [vadd, vzero]
theorem zero_vadd (a : Vec) : vadd vzero a = a := by funext k; simp [vadd, vzero]
theorem vadd_assoc (a b c : Vec) : vadd (vadd a b) c = vadd a (vadd b c) := by
  funext k; simp only [vadd]; omega
theorem vadd_comm (a b : Vec) : vadd a b = vadd b a := by funext k; simp only [vadd]; omega

/-- Extensional equality on the live coordinates. Decidable on concrete data. -/
def eqOn (m : Nat) (a b : Vec) : Prop := ∀ k, k < m → a k = b k
instance (m : Nat) (a b : Vec) : Decidable (eqOn m a b) := Nat.decidableBallLT m _

theorem eqOn_refl (m : Nat) (a : Vec) : eqOn m a a := fun _ _ => rfl
theorem eqOn_symm {m : Nat} {a b : Vec} (h : eqOn m a b) : eqOn m b a := fun k hk => (h k hk).symm
theorem eqOn_trans {m : Nat} {a b c : Vec} (h₁ : eqOn m a b) (h₂ : eqOn m b c) : eqOn m a c :=
  fun k hk => (h₁ k hk).trans (h₂ k hk)
theorem eqOn_of_eq {m : Nat} {a b : Vec} (h : a = b) : eqOn m a b := by subst h; exact eqOn_refl m a
theorem eqOn_vadd {m : Nat} {a a' b b' : Vec} (ha : eqOn m a a') (hb : eqOn m b b') :
    eqOn m (vadd a b) (vadd a' b') := fun k hk => by simp only [vadd]; rw [ha k hk, hb k hk]
theorem eqOn_smul {m : Nat} (c : Int) {a a' : Vec} (ha : eqOn m a a') :
    eqOn m (smul c a) (smul c a') := fun k hk => by simp only [smul]; rw [ha k hk]
theorem eqOn_vadd_zero {m : Nat} {a b : Vec} (ha : eqOn m a vzero) (hb : eqOn m b vzero) :
    eqOn m (vadd a b) vzero := fun k hk => by
  simp only [vadd, vzero]; rw [ha k hk, hb k hk]; rfl
theorem eqOn_smul_zero {m : Nat} (c : Int) {a : Vec} (ha : eqOn m a vzero) :
    eqOn m (smul c a) vzero := fun k hk => by
  simp only [smul, vzero]; rw [ha k hk]; exact Int.mul_zero c

/-- The Cayley–Dickson product at level `n`:
    `(a·b)_k = Σ_{i<2^n} σ(i, i⊕k) · a_i · b_{i⊕k}`. -/
def cdMul (n : Nat) (a b : Vec) : Vec :=
  fun k => sumR (2^n) (fun i => cdSigma i (i ^^^ k) n * a i * b (i ^^^ k))

theorem cdMul_add_right (n : Nat) (a b c : Vec) :
    cdMul n a (vadd b c) = vadd (cdMul n a b) (cdMul n a c) := by
  funext k; simp only [cdMul, vadd]; rw [← sumR_add]
  apply sumR_congr; intro i _; exact Int.mul_add _ _ _

theorem cdMul_add_left (n : Nat) (a a' b : Vec) :
    cdMul n (vadd a a') b = vadd (cdMul n a b) (cdMul n a' b) := by
  funext k; simp only [cdMul, vadd]; rw [← sumR_add]
  apply sumR_congr; intro i _; rw [Int.mul_add, Int.add_mul]

theorem cdMul_smul_left (n : Nat) (c : Int) (a b : Vec) :
    cdMul n (smul c a) b = smul c (cdMul n a b) := by
  funext k; simp only [cdMul, smul]; rw [← sumR_smul]
  apply sumR_congr; intro i _
  rw [Int.mul_left_comm (cdSigma i (i ^^^ k) n) c (a i), Int.mul_assoc]

theorem cdMul_smul_right (n : Nat) (c : Int) (a b : Vec) :
    cdMul n a (smul c b) = smul c (cdMul n a b) := by
  funext k; simp only [cdMul, smul]; rw [← sumR_smul]
  apply sumR_congr; intro i _
  exact Int.mul_left_comm _ c _

theorem cdMul_zero_left (n : Nat) (b : Vec) : cdMul n vzero b = vzero := by
  funext k; simp only [cdMul, vzero]
  apply sumR_eq_zero_of; intro i _; rw [Int.mul_zero, Int.zero_mul]

theorem cdMul_zero_right (n : Nat) (a : Vec) : cdMul n a vzero = vzero := by
  funext k; simp only [cdMul, vzero]
  apply sumR_eq_zero_of; intro i _; exact Int.mul_zero _

/-- The product reads its LEFT operand only on live coordinates (full equality). -/
theorem cdMul_congr_left (n : Nat) {a a' : Vec} (b : Vec) (h : eqOn (2^n) a a') :
    cdMul n a b = cdMul n a' b := by
  funext k; simp only [cdMul]; apply sumR_congr; intro i hi; rw [h i hi]

/-- …and its RIGHT operand only on live coordinates, for live outputs. -/
theorem cdMul_congr_right (n : Nat) (a : Vec) {b b' : Vec} (h : eqOn (2^n) b b') :
    eqOn (2^n) (cdMul n a b) (cdMul n a b') := by
  intro k hk; simp only [cdMul]; apply sumR_congr; intro i hi
  rw [h (i ^^^ k) (Nat.xor_lt_two_pow hi hk)]

/-- Euclidean inner product on the first `m` coordinates, and the squared norm. -/
def inner (m : Nat) (a b : Vec) : Int := sumR m (fun i => a i * b i)
def normSq (m : Nat) (a : Vec) : Int := inner m a a

theorem inner_comm (m : Nat) (a b : Vec) : inner m a b = inner m b a := by
  unfold inner; apply sumR_congr; intro i _; exact Int.mul_comm _ _
theorem inner_add_left (m : Nat) (a a' b : Vec) :
    inner m (vadd a a') b = inner m a b + inner m a' b := by
  unfold inner; rw [← sumR_add]; apply sumR_congr; intro i _; exact Int.add_mul _ _ _
theorem inner_add_right (m : Nat) (a b b' : Vec) :
    inner m a (vadd b b') = inner m a b + inner m a b' := by
  unfold inner; rw [← sumR_add]; apply sumR_congr; intro i _; exact Int.mul_add _ _ _
theorem inner_zero_left (m : Nat) (b : Vec) : inner m vzero b = 0 := by
  unfold inner; apply sumR_eq_zero_of; intro i _; exact Int.zero_mul _
theorem inner_zero_right (m : Nat) (a : Vec) : inner m a vzero = 0 := by
  unfold inner; apply sumR_eq_zero_of; intro i _; exact Int.mul_zero _
theorem inner_congr {m : Nat} {a a' b b' : Vec} (ha : eqOn m a a') (hb : eqOn m b b') :
    inner m a b = inner m a' b' := by
  unfold inner; apply sumR_congr; intro i hi; rw [ha i hi, hb i hi]
theorem inner_zero_right_of {m : Nat} (a : Vec) {b : Vec} (hb : eqOn m b vzero) : inner m a b = 0 := by
  rw [inner_congr (eqOn_refl m a) hb]; exact inner_zero_right m a

/-- `‖a+b‖² = ‖a‖² + ‖b‖² + 2⟨a,b⟩` — the term the independence assumption drops. -/
theorem normSq_add (m : Nat) (a b : Vec) :
    normSq m (vadd a b) = normSq m a + normSq m b + 2 * inner m a b := by
  unfold normSq
  rw [inner_add_left, inner_add_right, inner_add_right, inner_comm m b a]; omega

-- ================================================================
-- §B. The associator, its trilinearity, and the support induction
-- ================================================================

/-- `[x,y,z] = (x·y)·z − x·(y·z)` at level `n`. -/
def assoc (n : Nat) (x y z : Vec) : Vec :=
  vsub (cdMul n (cdMul n x y) z) (cdMul n x (cdMul n y z))

theorem assoc_add1 (n : Nat) (x x' y z : Vec) :
    assoc n (vadd x x') y z = vadd (assoc n x y z) (assoc n x' y z) := by
  funext k; simp only [assoc, vsub, vadd]
  rw [cdMul_add_left, cdMul_add_left, cdMul_add_left]; simp only [vadd]; omega
theorem assoc_add2 (n : Nat) (x y y' z : Vec) :
    assoc n x (vadd y y') z = vadd (assoc n x y z) (assoc n x y' z) := by
  funext k; simp only [assoc, vsub, vadd]
  rw [cdMul_add_right, cdMul_add_left, cdMul_add_left, cdMul_add_right]; simp only [vadd]; omega
theorem assoc_add3 (n : Nat) (x y z z' : Vec) :
    assoc n x y (vadd z z') = vadd (assoc n x y z) (assoc n x y z') := by
  funext k; simp only [assoc, vsub, vadd]
  rw [cdMul_add_right, cdMul_add_right, cdMul_add_right]; simp only [vadd]; omega
theorem assoc_smul1 (n : Nat) (c : Int) (x y z : Vec) :
    assoc n (smul c x) y z = smul c (assoc n x y z) := by
  funext k; simp only [assoc, vsub, smul]
  rw [cdMul_smul_left, cdMul_smul_left, cdMul_smul_left]; simp only [smul]; rw [Int.mul_sub]
theorem assoc_smul2 (n : Nat) (c : Int) (x y z : Vec) :
    assoc n x (smul c y) z = smul c (assoc n x y z) := by
  funext k; simp only [assoc, vsub, smul]
  rw [cdMul_smul_right, cdMul_smul_left, cdMul_smul_left, cdMul_smul_right]; simp only [smul]
  rw [Int.mul_sub]
theorem assoc_smul3 (n : Nat) (c : Int) (x y z : Vec) :
    assoc n x y (smul c z) = smul c (assoc n x y z) := by
  funext k; simp only [assoc, vsub, smul]
  rw [cdMul_smul_right, cdMul_smul_right, cdMul_smul_right]; simp only [smul]; rw [Int.mul_sub]

theorem assoc_zero1 (n : Nat) {x : Vec} (y z : Vec) (hx : eqOn (2^n) x vzero) :
    eqOn (2^n) (assoc n x y z) vzero := by
  have h1 : cdMul n x y = vzero := by rw [cdMul_congr_left n y hx, cdMul_zero_left]
  have h2 : cdMul n x (cdMul n y z) = vzero := by rw [cdMul_congr_left n _ hx, cdMul_zero_left]
  intro k hk; simp only [assoc, vsub]; rw [h1, h2, cdMul_zero_left]; rfl
theorem assoc_zero2 (n : Nat) (x : Vec) {y : Vec} (z : Vec) (hy : eqOn (2^n) y vzero) :
    eqOn (2^n) (assoc n x y z) vzero := by
  have h1 : eqOn (2^n) (cdMul n x y) vzero := by
    have := cdMul_congr_right n x hy; rw [cdMul_zero_right] at this; exact this
  have h2 : eqOn (2^n) (cdMul n y z) vzero := by
    rw [cdMul_congr_left n z hy, cdMul_zero_left]; exact eqOn_refl _ _
  have h3 : eqOn (2^n) (cdMul n x (cdMul n y z)) vzero := by
    have := cdMul_congr_right n x h2; rw [cdMul_zero_right] at this; exact this
  intro k hk; simp only [assoc, vsub]
  rw [cdMul_congr_left n z h1, cdMul_zero_left, h3 k hk]; rfl
theorem assoc_zero3 (n : Nat) (x y : Vec) {z : Vec} (hz : eqOn (2^n) z vzero) :
    eqOn (2^n) (assoc n x y z) vzero := by
  have h1 : eqOn (2^n) (cdMul n (cdMul n x y) z) vzero := by
    have := cdMul_congr_right n (cdMul n x y) hz; rw [cdMul_zero_right] at this; exact this
  have h2 : eqOn (2^n) (cdMul n y z) vzero := by
    have := cdMul_congr_right n y hz; rw [cdMul_zero_right] at this; exact this
  have h3 : eqOn (2^n) (cdMul n x (cdMul n y z)) vzero := by
    have := cdMul_congr_right n x h2; rw [cdMul_zero_right] at this; exact this
  intro k hk; simp only [assoc, vsub]; rw [h1 k hk, h3 k hk]; rfl

/-- Basis-support cover on a list: every live nonzero coordinate is listed
    (the Axis-1 twin of `Covers` for noise symbols). -/
def qCoversL (n : Nat) (L : List Nat) (v : Vec) : Prop := ∀ k, k < 2^n → v k ≠ 0 → k ∈ L

/-- Peel coordinate `i` off `v`: `v = v_i·e_i + v'` with `v' i = 0`. -/
def peel (i : Nat) (v : Vec) : Vec := fun k => if k = i then 0 else v k

theorem peel_decomp (i : Nat) (v : Vec) : v = vadd (smul (v i) (e i)) (peel i v) := by
  funext k; simp only [vadd, smul, e, peel]
  by_cases h : k = i
  · subst h; simp
  · simp [h]

theorem qCoversL_peel {n : Nat} {L : List Nat} {i : Nat} {v : Vec}
    (h : qCoversL n (i :: L) v) : qCoversL n L (peel i v) := by
  intro k hk hne
  simp only [peel] at hne
  by_cases hki : k = i
  · subst hki; simp at hne
  · simp only [hki, if_false] at hne
    rcases List.mem_cons.mp (h k hk hne) with h1 | h1
    · exact absurd h1 hki
    · exact h1

theorem eqOn_zero_of_qCoversL_nil {n : Nat} {v : Vec} (h : qCoversL n [] v) : eqOn (2^n) v vzero := by
  intro k hk
  by_cases hv : v k = 0
  · exact hv
  · exact absurd (h k hk hv) (List.not_mem_nil)

/-- Support induction, slot 1: if the associator vanishes on the basis elements of `L` (with
    `y`, `z` fixed), it vanishes on every vector supported in `L`. -/
theorem assoc_slot1 (n : Nat) (L : List Nat) (y z : Vec) :
    ∀ x, qCoversL n L x → (∀ i ∈ L, eqOn (2^n) (assoc n (e i) y z) vzero) →
      eqOn (2^n) (assoc n x y z) vzero := by
  induction L with
  | nil => intro x hx _; exact assoc_zero1 n y z (eqOn_zero_of_qCoversL_nil hx)
  | cons i L ih =>
    intro x hx hb
    rw [peel_decomp i x, assoc_add1, assoc_smul1]
    exact eqOn_vadd_zero (eqOn_smul_zero _ (hb i (List.mem_cons_self))) 
      (ih (peel i x) (qCoversL_peel hx) (fun j hj => hb j (List.mem_cons_of_mem i hj)))

theorem assoc_slot2 (n : Nat) (L : List Nat) (x z : Vec) :
    ∀ y, qCoversL n L y → (∀ j ∈ L, eqOn (2^n) (assoc n x (e j) z) vzero) →
      eqOn (2^n) (assoc n x y z) vzero := by
  induction L with
  | nil => intro y hy _; exact assoc_zero2 n x z (eqOn_zero_of_qCoversL_nil hy)
  | cons j L ih =>
    intro y hy hb
    rw [peel_decomp j y, assoc_add2, assoc_smul2]
    exact eqOn_vadd_zero (eqOn_smul_zero _ (hb j (List.mem_cons_self)))
      (ih (peel j y) (qCoversL_peel hy) (fun l hl => hb l (List.mem_cons_of_mem j hl)))

theorem assoc_slot3 (n : Nat) (L : List Nat) (x y : Vec) :
    ∀ z, qCoversL n L z → (∀ l ∈ L, eqOn (2^n) (assoc n x y (e l)) vzero) →
      eqOn (2^n) (assoc n x y z) vzero := by
  induction L with
  | nil => intro z hz _; exact assoc_zero3 n x y (eqOn_zero_of_qCoversL_nil hz)
  | cons l L ih =>
    intro z hz hb
    rw [peel_decomp l z, assoc_add3, assoc_smul3]
    exact eqOn_vadd_zero (eqOn_smul_zero _ (hb l (List.mem_cons_self)))
      (ih (peel l z) (qCoversL_peel hz) (fun l' hl' => hb l' (List.mem_cons_of_mem l hl')))

/-- The associator vanishes on all basis triples drawn from `L` (decidable). -/
def assocZeroOn (n : Nat) (L : List Nat) : Prop :=
  ∀ i ∈ L, ∀ j ∈ L, ∀ l ∈ L, eqOn (2^n) (assoc n (e i) (e j) (e l)) vzero
instance (n : Nat) (L : List Nat) : Decidable (assocZeroOn n L) := by
  unfold assocZeroOn; infer_instance

/-- TRILINEARITY THEOREM: an associative basis-index set is an associative subalgebra —
    the associator of any three vectors supported in `L` vanishes. This is the Axis-1
    analogue of `inner_zero_of_ns`: the syntactic cover discharges the semantic condition. -/
theorem assoc_zero_of_qCoversL (n : Nat) (L : List Nat) {x y z : Vec}
    (hx : qCoversL n L x) (hy : qCoversL n L y) (hz : qCoversL n L z)
    (hL : assocZeroOn n L) : eqOn (2^n) (assoc n x y z) vzero :=
  assoc_slot1 n L y z x hx (fun i hi =>
    assoc_slot2 n L (e i) z y hy (fun j hj =>
      assoc_slot3 n L (e i) (e j) z hz (fun l hl => hL i hi j hj l hl)))

/-- The seven Fano lines of the octonions, each with `e_0`: the quaternionic subalgebras. -/
def fanoLines : List (List Nat) :=
  [[0,1,2,3],[0,1,4,5],[0,1,6,7],[0,2,4,6],[0,2,5,7],[0,3,4,7],[0,3,5,6]]

theorem fano_lines_assoc : ∀ L ∈ fanoLines, assocZeroOn 3 L := by decide
theorem real_assoc : assocZeroOn 0 [0] := by decide
theorem complex_assoc : assocZeroOn 1 [0,1] := by decide
theorem quaternion_assoc : assocZeroOn 2 [0,1,2,3] := by decide
/-- …and the non-Fano triple `(1,2,4)` is NOT associative: the lever of Axis 1. -/
theorem non_fano_124 : ¬ eqOn 8 (assoc 3 (e 1) (e 2) (e 4)) vzero := by decide
theorem non_fano_124_value : eqOn 8 (assoc 3 (e 1) (e 2) (e 4)) (smul 2 (e 7)) := by decide


-- ================================================================
-- §C. First-order affine forms with VECTOR coefficients (the true noise content)
-- ================================================================

/-- `[(s, d), …]`: source `s` carries sensitivity vector `d`. Repeats allowed (exact `++`). -/
abbrev Aff := List (Nat × Vec)

def coeff : Aff → Nat → Vec
  | [], _ => vzero
  | (t, c) :: r, s => if t = s then vadd c (coeff r s) else coeff r s

/-- `Σ_{(s,c)∈a} ⟨c, coeff b s⟩` — the kernel's duplicate-tolerant inner product, lifted. -/
def innerA (m : Nat) : Aff → Aff → Int
  | [], _ => 0
  | (s, c) :: r, b => inner m c (coeff b s) + innerA m r b

/-- True first-order variance `Σ_s ‖∂_s‖²` (Lemma 0), as `innerA a a`. -/
def trueVar (m : Nat) (a : Aff) : Int := innerA m a a

theorem coeff_append (a b : Aff) (s : Nat) : coeff (a ++ b) s = vadd (coeff a s) (coeff b s) := by
  induction a with
  | nil => simp [coeff, zero_vadd]
  | cons p r ih =>
    rcases p with ⟨t, c⟩
    simp only [List.cons_append, coeff]
    by_cases h : t = s
    · simp only [h, if_true]; rw [ih, vadd_assoc]
    · simp only [h, if_false]; exact ih

theorem innerA_append_left (m : Nat) (a b c : Aff) :
    innerA m (a ++ b) c = innerA m a c + innerA m b c := by
  induction a with
  | nil => simp [innerA]
  | cons p r ih => rcases p with ⟨t, d⟩; simp only [List.cons_append, innerA]; rw [ih]; omega

theorem innerA_append_right (m : Nat) (a b c : Aff) :
    innerA m a (b ++ c) = innerA m a b + innerA m a c := by
  induction a with
  | nil => rfl
  | cons p r ih =>
    rcases p with ⟨t, d⟩; simp only [innerA]; rw [ih, coeff_append, inner_add_right]; omega

theorem innerA_single_right (m : Nat) (b : Aff) (s : Nat) (c : Vec) :
    innerA m b [(s, c)] = inner m (coeff b s) c := by
  induction b with
  | nil => simp [innerA, coeff, inner_zero_left]
  | cons p r ih =>
    rcases p with ⟨t, d⟩
    simp only [innerA, coeff]
    rw [ih]
    by_cases h : t = s
    · subst h; simp only [if_true]; rw [vadd_zero, inner_add_left]
    · have h' : ¬ s = t := fun e => h e.symm
      simp only [h, h', if_false]; rw [inner_zero_right]; omega

theorem innerA_comm (m : Nat) (a b : Aff) : innerA m a b = innerA m b a := by
  induction a with
  | nil =>
    induction b with
    | nil => rfl
    | cons p r ih =>
      rcases p with ⟨t, d⟩
      simp only [innerA, coeff] at ih ⊢
      rw [inner_zero_right, ← ih]; rfl
  | cons p r ih =>
    rcases p with ⟨s, c⟩
    have : innerA m b ((s, c) :: r) = innerA m b [(s, c)] + innerA m b r :=
      innerA_append_right m b [(s, c)] r
    rw [this, innerA_single_right, inner_comm]; simp only [innerA]; rw [ih]

/-- `Var(a ++ b) = Var a + Var b + 2⟨a,b⟩` — the exact sum's variance. -/
theorem trueVar_append (m : Nat) (a b : Aff) :
    trueVar m (a ++ b) = trueVar m a + trueVar m b + 2 * innerA m a b := by
  unfold trueVar
  rw [innerA_append_left, innerA_append_right, innerA_append_right, innerA_comm m b a]; omega

theorem coeff_absent (b : Aff) (s : Nat) (h : ∀ p ∈ b, p.1 ≠ s) : coeff b s = vzero := by
  induction b with
  | nil => rfl
  | cons p r ih =>
    rcases p with ⟨t, c⟩
    have ht : t ≠ s := h (t, c) (List.mem_cons_self)
    simp only [coeff, ht, if_false]
    exact ih (fun q hq => h q (List.mem_cons_of_mem _ hq))

/-- Index-disjoint forms have zero cross term (Lemma 3, vector-valued). -/
theorem innerA_disjoint (m : Nat) (a b : Aff) (h : ∀ p ∈ a, ∀ q ∈ b, p.1 ≠ q.1) :
    innerA m a b = 0 := by
  induction a with
  | nil => rfl
  | cons p r ih =>
    rcases p with ⟨s, c⟩
    simp only [innerA]
    rw [coeff_absent b s (fun q hq => (h (s, c) (List.mem_cons_self) q hq).symm), inner_zero_right]
    rw [ih (fun p hp q hq => h p (List.mem_cons_of_mem _ hp) q hq)]; rfl

/-- First-order Leibniz scalings: `∂_s(x·y) = ∂_s x · y₀ + x₀ · ∂_s y`. -/
def scaleR (n : Nat) (a : Aff) (y : Vec) : Aff := a.map (fun p => (p.1, cdMul n p.2 y))
def scaleL (n : Nat) (x : Vec) (b : Aff) : Aff := b.map (fun p => (p.1, cdMul n x p.2))

theorem coeff_scaleR (n : Nat) (a : Aff) (y : Vec) (s : Nat) :
    coeff (scaleR n a y) s = cdMul n (coeff a s) y := by
  induction a with
  | nil => simp [scaleR, coeff, cdMul_zero_left]
  | cons p r ih =>
    rcases p with ⟨t, c⟩
    simp only [scaleR, List.map_cons, coeff]
    by_cases h : t = s
    · simp only [h, if_true]; rw [cdMul_add_left]; simp only [scaleR] at ih; rw [ih]
    · simp only [h, if_false]; simp only [scaleR] at ih; exact ih

theorem coeff_scaleL (n : Nat) (x : Vec) (b : Aff) (s : Nat) :
    coeff (scaleL n x b) s = cdMul n x (coeff b s) := by
  induction b with
  | nil => simp [scaleL, coeff, cdMul_zero_right]
  | cons p r ih =>
    rcases p with ⟨t, c⟩
    simp only [scaleL, List.map_cons, coeff]
    by_cases h : t = s
    · simp only [h, if_true]; rw [cdMul_add_right]; simp only [scaleL] at ih; rw [ih]
    · simp only [h, if_false]; simp only [scaleL] at ih; exact ih

theorem mem_scaleR_idx {n : Nat} {a : Aff} {y : Vec} {p : Nat × Vec} (h : p ∈ scaleR n a y) :
    ∃ q ∈ a, q.1 = p.1 := by
  rcases List.mem_map.mp h with ⟨q, hq, rfl⟩; exact ⟨q, hq, rfl⟩
theorem mem_scaleL_idx {n : Nat} {x : Vec} {b : Aff} {p : Nat × Vec} (h : p ∈ scaleL n x b) :
    ∃ q ∈ b, q.1 = p.1 := by
  rcases List.mem_map.mp h with ⟨q, hq, rfl⟩; exact ⟨q, hq, rfl⟩

/-- Pairwise-equal forms (same indices in the same order, coefficients equal on live coords). -/
def AffEqOn (m : Nat) : Aff → Aff → Prop
  | [], [] => True
  | (s, c) :: r, (s', c') :: r' => s = s' ∧ eqOn m c c' ∧ AffEqOn m r r'
  | _, _ => False

theorem coeff_congr {m : Nat} {a a' : Aff} (h : AffEqOn m a a') (s : Nat) :
    eqOn m (coeff a s) (coeff a' s) := by
  induction a generalizing a' with
  | nil => cases a' with
    | nil => exact eqOn_refl m _
    | cons p r => exact absurd h (by simp [AffEqOn])
  | cons p r ih =>
    rcases p with ⟨t, c⟩
    cases a' with
    | nil => exact absurd h (by simp [AffEqOn])
    | cons p' r' =>
      rcases p' with ⟨t', c'⟩
      rcases h with ⟨rfl, hc, hr⟩
      simp only [coeff]
      by_cases ht : t = s
      · simp only [ht, if_true]; exact eqOn_vadd hc (ih hr)
      · simp only [ht, if_false]; exact ih hr

theorem innerA_congr {m : Nat} {a a' b b' : Aff} (ha : AffEqOn m a a') (hb : AffEqOn m b b') :
    innerA m a b = innerA m a' b' := by
  induction a generalizing a' with
  | nil => cases a' with
    | nil => rfl
    | cons p r => exact absurd ha (by simp [AffEqOn])
  | cons p r ih =>
    rcases p with ⟨t, c⟩
    cases a' with
    | nil => exact absurd ha (by simp [AffEqOn])
    | cons p' r' =>
      rcases p' with ⟨t', c'⟩
      rcases ha with ⟨rfl, hc, hr⟩
      simp only [innerA]
      rw [inner_congr hc (coeff_congr hb t), ih hr]

theorem trueVar_congr {m : Nat} {a a' : Aff} (h : AffEqOn m a a') : trueVar m a = trueVar m a' :=
  innerA_congr h h

theorem affEqOn_refl (m : Nat) (a : Aff) : AffEqOn m a a := by
  induction a with
  | nil => trivial
  | cons p r ih => rcases p with ⟨s, c⟩; exact ⟨rfl, eqOn_refl m c, ih⟩

theorem affEqOn_append {m : Nat} {a a' b b' : Aff} (ha : AffEqOn m a a') (hb : AffEqOn m b b') :
    AffEqOn m (a ++ b) (a' ++ b') := by
  induction a generalizing a' with
  | nil => cases a' with
    | nil => simpa using hb
    | cons p r => exact absurd ha (by simp [AffEqOn])
  | cons p r ih =>
    rcases p with ⟨t, c⟩
    cases a' with
    | nil => exact absurd ha (by simp [AffEqOn])
    | cons p' r' =>
      rcases p' with ⟨t', c'⟩
      rcases ha with ⟨rfl, hc, hr⟩
      exact ⟨rfl, hc, ih hr⟩

-- ================================================================
-- §D. Axis 2 — the noise-set lattice `N` (ported from EpistemicEffectsNS)
-- ================================================================

abbrev NS := Option (List Nat)
def nsTop : NS := none
def nsEmpty : NS := some []
def nsSingle (s : Nat) : NS := some [s]
def nsUnion : NS → NS → NS
  | some a, some b => some (a ++ b)
  | _, _ => none
def nsMem (s : Nat) : NS → Prop
  | none => True
  | some l => s ∈ l
def nsDisjoint : NS → NS → Bool
  | some a, some b => a.all (fun s => decide (s ∉ b))
  | _, _ => false

theorem nsDisjoint_sound {la lb : List Nat} (h : nsDisjoint (some la) (some lb) = true) :
    ∀ s, s ∈ la → s ∉ lb := by
  intro s hs
  have := List.all_eq_true.mp h s hs
  exact of_decide_eq_true this
theorem nsDisjoint_top_left (N : NS) : nsDisjoint none N = false := rfl
theorem nsDisjoint_top_right (N : NS) : nsDisjoint N none = false := by cases N <;> rfl
theorem nsDisjoint_of_shared {s : Nat} {Na Nb : NS} (ha : nsMem s Na) (hb : nsMem s Nb) :
    nsDisjoint Na Nb = false := by
  cases Na with
  | none => rfl
  | some la => cases Nb with
    | none => rfl
    | some lb =>
      cases h : nsDisjoint (some la) (some lb) with
      | false => rfl
      | true => exact absurd hb (nsDisjoint_sound h s ha)

/-- `N` covers `a`: every source of the true form is tracked (Lemma 2 invariant). -/
def Covers (N : NS) (a : Aff) : Prop := ∀ p ∈ a, nsMem p.1 N

theorem covers_top (a : Aff) : Covers nsTop a := fun _ _ => trivial
theorem covers_empty : Covers nsEmpty [] := fun _ h => by simp at h
theorem covers_single (s : Nat) (c : Vec) : Covers (nsSingle s) [(s, c)] := by
  intro p hp; simp at hp; subst hp; simp [nsMem, nsSingle]
theorem nsMem_union_left {s : Nat} {Na Nb : NS} (h : nsMem s Na) : nsMem s (nsUnion Na Nb) := by
  cases Na with
  | none => cases Nb <;> trivial
  | some la => cases Nb with
    | none => trivial
    | some lb => exact List.mem_append.mpr (Or.inl h)
theorem nsMem_union_right {s : Nat} {Na Nb : NS} (h : nsMem s Nb) : nsMem s (nsUnion Na Nb) := by
  cases Na with
  | none => cases Nb <;> trivial
  | some la => cases Nb with
    | none => trivial
    | some lb => exact List.mem_append.mpr (Or.inr h)
theorem covers_union {Na Nb : NS} {a b : Aff} (ha : Covers Na a) (hb : Covers Nb b) :
    Covers (nsUnion Na Nb) (a ++ b) := by
  intro p hp
  rcases List.mem_append.mp hp with h | h
  · exact nsMem_union_left (ha p h)
  · exact nsMem_union_right (hb p h)
theorem covers_scaleR {N : NS} {a : Aff} (n : Nat) (y : Vec) (h : Covers N a) :
    Covers N (scaleR n a y) := by
  intro p hp; rcases mem_scaleR_idx hp with ⟨q, hq, hqp⟩; rw [← hqp]; exact h q hq
theorem covers_scaleL {N : NS} {b : Aff} (n : Nat) (x : Vec) (h : Covers N b) :
    Covers N (scaleL n x b) := by
  intro p hp; rcases mem_scaleL_idx hp with ⟨q, hq, hqp⟩; rw [← hqp]; exact h q hq

/-- The NS certificate discharges the semantic condition: disjoint tracked sets ⇒ zero
    cross term between the TRUE forms (Axis 2 soundness). -/
theorem innerA_zero_of_ns (m : Nat) {Na Nb : NS} {a b : Aff} (ha : Covers Na a) (hb : Covers Nb b)
    (hd : nsDisjoint Na Nb = true) : innerA m a b = 0 := by
  cases Na with
  | none => exact absurd hd (by simp [nsDisjoint])
  | some la => cases Nb with
    | none => exact absurd hd (by simp [nsDisjoint])
    | some lb =>
      apply innerA_disjoint
      intro p hp q hq heq
      exact nsDisjoint_sound hd p.1 (ha p hp) (heq ▸ hb q hq)

/-- Pairwise disjointness is what both parenthesizations of a triple need — and it is the
    SAME condition (Axis 2 is invariant under Axis-1 rewrites). -/
theorem nsDisjoint_union_right {N N1 N2 : NS} :
    nsDisjoint N (nsUnion N1 N2) = true ↔ nsDisjoint N N1 = true ∧ nsDisjoint N N2 = true := by
  cases N with
  | none => simp [nsDisjoint]
  | some l => cases N1 with
    | none => simp [nsUnion, nsDisjoint]
    | some l1 => cases N2 with
      | none => simp [nsUnion, nsDisjoint]
      | some l2 =>
        simp only [nsUnion, nsDisjoint, List.all_eq_true, decide_eq_true_eq, List.mem_append]
        constructor
        · intro h; exact ⟨fun s hs h1 => h s hs (Or.inl h1), fun s hs h2 => h s hs (Or.inr h2)⟩
        · rintro ⟨h1, h2⟩ s hs h; rcases h with h | h
          · exact h1 s hs h
          · exact h2 s hs h
theorem nsDisjoint_comm (Na Nb : NS) : nsDisjoint Na Nb = nsDisjoint Nb Na := by
  cases Na with
  | none => cases Nb <;> rfl
  | some la => cases Nb with
    | none => rfl
    | some lb =>
      cases h1 : nsDisjoint (some la) (some lb) <;> cases h2 : nsDisjoint (some lb) (some la) <;> try rfl
      · exfalso
        have h2' := nsDisjoint_sound h2
        have : ¬ (la.all (fun s => decide (s ∉ lb)) = true) := by simpa [nsDisjoint] using h1
        apply this; apply List.all_eq_true.mpr; intro s hs; apply decide_eq_true
        intro hsb; exact h2' s hsb hs
      · exfalso
        have h1' := nsDisjoint_sound h1
        have : ¬ (lb.all (fun s => decide (s ∉ la)) = true) := by simpa [nsDisjoint] using h2
        apply this; apply List.all_eq_true.mpr; intro s hs; apply decide_eq_true
        intro hsa; exact h1' s hsa hs
theorem nsDisjoint_union_left {N1 N2 N : NS} :
    nsDisjoint (nsUnion N1 N2) N = true ↔ nsDisjoint N1 N = true ∧ nsDisjoint N2 N = true := by
  rw [nsDisjoint_comm, nsDisjoint_union_right, nsDisjoint_comm N N1, nsDisjoint_comm N N2]

/-- SYNTACTIC ORTHOGONALITY (Axis 2 ⟂ Axis 1): the NS premises of `(xy)z` and of `x(yz)`
    are equivalent — re-association neither creates nor destroys a support certificate. -/
theorem nsDisjoint_reassoc_invariant (Nx Ny Nz : NS) :
    (nsDisjoint Nx Ny = true ∧ nsDisjoint (nsUnion Nx Ny) Nz = true) ↔
    (nsDisjoint Ny Nz = true ∧ nsDisjoint Nx (nsUnion Ny Nz) = true) := by
  rw [nsDisjoint_union_left, nsDisjoint_union_right]
  constructor
  · rintro ⟨hxy, hxz, hyz⟩; exact ⟨hyz, hxy, hxz⟩
  · rintro ⟨hyz, hxy, hxz⟩; exact ⟨hxy, hxz, hyz⟩

-- ================================================================
-- §E. Axis 1 — the basis-support lattice `Q` (the twin) and the certificate
-- ================================================================

abbrev QS := Option (List Nat)
def qTop : QS := none
def qCovers (n : Nat) : QS → Vec → Prop
  | none, _ => True
  | some L, v => qCoversL n L v
def qCoversAff (n : Nat) (Q : QS) (a : Aff) : Prop := ∀ p ∈ a, qCovers n Q p.2

instance (n : Nat) (L : List Nat) (v : Vec) : Decidable (qCoversL n L v) := by
  unfold qCoversL; infer_instance
instance (n : Nat) (Q : QS) (v : Vec) : Decidable (qCovers n Q v) :=
  match Q with
  | none => isTrue trivial
  | some L => inferInstanceAs (Decidable (qCoversL n L v))
def qUnion : QS → QS → QS
  | some a, some b => some (a ++ b)
  | _, _ => none
/-- Basis support of a product: `{i ⊕ j : i ∈ Qa, j ∈ Qb}`. -/
def qProd : QS → QS → QS
  | some a, some b => some (a.flatMap (fun i => b.map (fun j => i ^^^ j)))
  | _, _ => none

theorem qCoversL_mono {n : Nat} {L L' : List Nat} (hL : ∀ k ∈ L, k ∈ L') {v : Vec}
    (h : qCoversL n L v) : qCoversL n L' v := fun k hk hv => hL k (h k hk hv)

theorem qCovers_vadd {n : Nat} {Qa Qb : QS} {a b : Vec} (ha : qCovers n Qa a) (hb : qCovers n Qb b) :
    qCovers n (qUnion Qa Qb) (vadd a b) := by
  cases Qa with
  | none => cases Qb <;> trivial
  | some La => cases Qb with
    | none => trivial
    | some Lb =>
      intro k hk hne
      simp only [vadd] at hne
      by_cases hak : a k = 0
      · have hbk : b k ≠ 0 := by intro h; apply hne; rw [hak, h]; rfl
        exact List.mem_append.mpr (Or.inr (hb k hk hbk))
      · exact List.mem_append.mpr (Or.inl (ha k hk hak))

theorem qCovers_cdMul {n : Nat} {Qa Qb : QS} {a b : Vec} (ha : qCovers n Qa a) (hb : qCovers n Qb b) :
    qCovers n (qProd Qa Qb) (cdMul n a b) := by
  cases Qa with
  | none => cases Qb <;> trivial
  | some La => cases Qb with
    | none => trivial
    | some Lb =>
      intro k hk hne
      simp only [cdMul] at hne
      rcases sumR_ne_zero_exists _ _ hne with ⟨i, hi, hf⟩
      have hai : a i ≠ 0 := by intro h; apply hf; rw [h, Int.mul_zero, Int.zero_mul]
      have hbj : b (i ^^^ k) ≠ 0 := by intro h; apply hf; rw [h, Int.mul_zero]
      have hiL : i ∈ La := ha i hi hai
      have hjL : (i ^^^ k) ∈ Lb := hb (i ^^^ k) (Nat.xor_lt_two_pow hi hk) hbj
      apply List.mem_flatMap.mpr
      refine ⟨i, hiL, List.mem_map.mpr ⟨i ^^^ k, hjL, ?_⟩⟩
      rw [← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]

theorem qCoversAff_append {n : Nat} {Q : QS} {a b : Aff}
    (ha : qCoversAff n Q a) (hb : qCoversAff n Q b) : qCoversAff n Q (a ++ b) := by
  intro p hp; rcases List.mem_append.mp hp with h | h
  · exact ha p h
  · exact hb p h

theorem qCovers_qUnion_left {n : Nat} {Qa Qb : QS} {v : Vec} (h : qCovers n Qa v) :
    qCovers n (qUnion Qa Qb) v := by
  cases Qa with
  | none => cases Qb <;> trivial
  | some La => cases Qb with
    | none => trivial
    | some Lb => exact qCoversL_mono (fun k hk => List.mem_append.mpr (Or.inl hk)) h
theorem qCovers_qUnion_right {n : Nat} {Qa Qb : QS} {v : Vec} (h : qCovers n Qb v) :
    qCovers n (qUnion Qa Qb) v := by
  cases Qa with
  | none => cases Qb <;> trivial
  | some La => cases Qb with
    | none => trivial
    | some Lb => exact qCoversL_mono (fun k hk => List.mem_append.mpr (Or.inr hk)) h
theorem qCovers_vadd_same {n : Nat} {Q : QS} {a b : Vec} (ha : qCovers n Q a) (hb : qCovers n Q b) :
    qCovers n Q (vadd a b) := by
  cases Q with
  | none => trivial
  | some L =>
    intro k hk hne
    simp only [vadd] at hne
    by_cases hak : a k = 0
    · have hbk : b k ≠ 0 := by intro h; apply hne; rw [hak, h]; rfl
      exact hb k hk hbk
    · exact ha k hk hak
theorem qCovers_vzero (n : Nat) (Q : QS) : qCovers n Q vzero := by
  cases Q with
  | none => trivial
  | some L => intro k _ h; exact absurd rfl h
theorem qCovers_coeff {n : Nat} {Q : QS} {a : Aff} (h : qCoversAff n Q a) (s : Nat) :
    qCovers n Q (coeff a s) := by
  induction a with
  | nil => exact qCovers_vzero n Q
  | cons p r ih =>
    rcases p with ⟨t, c⟩
    simp only [coeff]
    have ih' := ih (fun q hq => h q (List.mem_cons_of_mem _ hq))
    by_cases ht : t = s
    · simp only [ht, if_true]; exact qCovers_vadd_same (h (t, c) (List.mem_cons_self)) ih'
    · simp only [ht, if_false]; exact ih'

theorem qCoversAff_scaleR {n : Nat} {Qa Qb : QS} {a : Aff} {y : Vec}
    (ha : qCoversAff n Qa a) (hy : qCovers n Qb y) : qCoversAff n (qProd Qa Qb) (scaleR n a y) := by
  intro p hp
  rcases List.mem_map.mp hp with ⟨q, hq, rfl⟩
  exact qCovers_cdMul (ha q hq) hy
theorem qCoversAff_scaleL {n : Nat} {Qa Qb : QS} {x : Vec} {b : Aff}
    (hx : qCovers n Qa x) (hb : qCoversAff n Qb b) : qCoversAff n (qProd Qa Qb) (scaleL n x b) := by
  intro p hp
  rcases List.mem_map.mp hp with ⟨q, hq, rfl⟩
  exact qCovers_cdMul hx (hb q hq)

/-- THE AXIS-1 CERTIFICATE (type-level, decidable). Re-association of a triple is licensed
    when the three basis supports lie in one associative subalgebra: everything at n ≤ 2
    (ℝ, ℂ, ℍ — mirrors `SounioCayleyDickson.canReassociate` / `ir_can_reassociate_triple`),
    or, at n = 3, one quaternionic Fano line. ⊤ is never certified beyond n ≤ 2. -/
def assocCert (n : Nat) (Qx Qy Qz : QS) : Bool :=
  decide (n ≤ 2) ||
  (n == 3 && match Qx, Qy, Qz with
    | some Lx, some Ly, some Lz =>
        fanoLines.any (fun L => (Lx ++ Ly ++ Lz).all (fun k => decide (k ∈ L)))
    | _, _, _ => false)

theorem qCoversL_range (n : Nat) (v : Vec) : qCoversL n (List.range (2^n)) v :=
  fun _ hk _ => List.mem_range.mpr hk

theorem assocZeroOn_range_le2 (n : Nat) (hn : n ≤ 2) : assocZeroOn n (List.range (2^n)) := by
  match n, hn with
  | 0, _ => exact real_assoc
  | 1, _ => exact complex_assoc
  | 2, _ => exact quaternion_assoc

/-- CERTIFICATE SOUNDNESS (Axis 1): a certified triple of TYPES forces the associator of every
    covered triple of VALUES to vanish on the live coordinates. -/
theorem assoc_zero_of_cert {n : Nat} {Qx Qy Qz : QS} {x y z : Vec}
    (hc : assocCert n Qx Qy Qz = true)
    (hx : qCovers n Qx x) (hy : qCovers n Qy y) (hz : qCovers n Qz z) :
    eqOn (2^n) (assoc n x y z) vzero := by
  unfold assocCert at hc
  rcases Bool.or_eq_true_iff.mp hc with h2 | h3
  · have hn : n ≤ 2 := of_decide_eq_true h2
    exact assoc_zero_of_qCoversL n _ (qCoversL_range n x) (qCoversL_range n y) (qCoversL_range n z)
      (assocZeroOn_range_le2 n hn)
  · rcases Bool.and_eq_true_iff.mp h3 with ⟨hn3, hm⟩
    have hn : n = 3 := by simpa using hn3
    subst hn
    cases Qx with
    | none => exact absurd hm (by simp)
    | some Lx => cases Qy with
      | none => exact absurd hm (by simp)
      | some Ly => cases Qz with
        | none => exact absurd hm (by simp)
        | some Lz =>
          rcases List.any_eq_true.mp (show fanoLines.any
              (fun L => (Lx ++ Ly ++ Lz).all (fun k => decide (k ∈ L))) = true from hm)
            with ⟨L, hL, hall⟩
          have hsub : ∀ k ∈ Lx ++ Ly ++ Lz, k ∈ L := fun k hk =>
            of_decide_eq_true (List.all_eq_true.mp hall k hk)
          have hx' : qCoversL 3 L x := qCoversL_mono
            (fun k hk => hsub k (List.mem_append.mpr (Or.inl (List.mem_append.mpr (Or.inl hk))))) hx
          have hy' : qCoversL 3 L y := qCoversL_mono
            (fun k hk => hsub k (List.mem_append.mpr (Or.inl (List.mem_append.mpr (Or.inr hk))))) hy
          have hz' : qCoversL 3 L z := qCoversL_mono
            (fun k hk => hsub k (List.mem_append.mpr (Or.inr hk))) hz
          exact assoc_zero_of_qCoversL 3 L hx' hy' hz' (fano_lines_assoc L hL)


-- ================================================================
-- §F. The operator calculus over CD(n): terms, types ⟨N, Q⟩, reduction
-- ================================================================

structure KMeta where
  gumVar : Int
  conf   : Int
  deriving Repr, DecidableEq

def kvalid (m : KMeta) : Prop := 0 ≤ m.conf ∧ m.conf ≤ 1000

/-- Terms. `kraw v m a`: a runtime Knowledge value with payload `v`, REPORTED metadata `m`,
    and TRUE first-order form `a`. `measure v c conf s`: measure center `v` with sensitivity
    vector `c` on source `s`. -/
inductive Expr where
  | kraw    : Vec → KMeta → Aff → Expr
  | measure : Vec → Vec → Int → Nat → Expr
  | certain : Vec → Expr
  | opaque  : Expr → Expr
  | kadd    : Expr → Expr → Expr
  | kmul    : Expr → Expr → Expr

inductive IsValue : Expr → Prop where
  | v_kraw : ∀ v m a, IsValue (.kraw v m a)

/-- A Knowledge type carries BOTH supports: noise symbols `N` and basis elements `Q`. -/
structure Ty where
  N : NS
  Q : QS

/-- Typing at level `n`. (Add-Indep)/(Mul-Indep) check the Axis-2 certificate `nsDisjoint`;
    `Q` is propagated by union / XOR-product. The Axis-1 certificate is checked NOT here but
    at the re-association rewrite (§H) — the two certificates guard two different relations. -/
inductive HasTy (n : Nat) : Expr → Ty → Prop where
  | t_kraw : ∀ v m a N Q, kvalid m → Covers N a → qCovers n Q v → qCoversAff n Q a →
      HasTy n (.kraw v m a) ⟨N, Q⟩
  | t_measure : ∀ v c conf s Q, 0 ≤ conf → conf ≤ 1000 → qCovers n Q v → qCovers n Q c →
      HasTy n (.measure v c conf s) ⟨nsSingle s, Q⟩
  | t_certain : ∀ v Q, qCovers n Q v → HasTy n (.certain v) ⟨nsEmpty, Q⟩
  | t_opaque : ∀ e N Q, HasTy n e ⟨N, Q⟩ → HasTy n (.opaque e) ⟨nsTop, qTop⟩
  | t_kadd : ∀ a b Na Qa Nb Qb, HasTy n a ⟨Na, Qa⟩ → HasTy n b ⟨Nb, Qb⟩ →
      nsDisjoint Na Nb = true → HasTy n (.kadd a b) ⟨nsUnion Na Nb, qUnion Qa Qb⟩
  | t_kmul : ∀ a b Na Qa Nb Qb, HasTy n a ⟨Na, Qa⟩ → HasTy n b ⟨Nb, Qb⟩ →
      nsDisjoint Na Nb = true → HasTy n (.kmul a b) ⟨nsUnion Na Nb, qProd Qa Qb⟩

/-- Reported metadata of a sum (no covariance term). -/
def gAddMeta (ma mb : KMeta) : KMeta :=
  { gumVar := ma.gumVar + mb.gumVar
  , conf := if ma.conf ≤ mb.conf then ma.conf else mb.conf }

/-- The SENSITIVITY (affine / Leibniz) propagator, independence-assuming: it carries the
    per-source sensitivity vectors and sums the two Leibniz halves' variances, dropping their
    cross term. Exact iff the halves are source-disjoint (Axis 2). No norm identity used. -/
def gMulMeta (n : Nat) (x : Vec) (a : Aff) (y : Vec) (b : Aff) (ma mb : KMeta) : KMeta :=
  { gumVar := trueVar (2^n) (scaleR n a y) + trueVar (2^n) (scaleL n x b)
  , conf := if ma.conf ≤ mb.conf then ma.conf else mb.conf }

/-- The GUM VARIANCE SHORTCUT `‖y‖²·Var x + ‖x‖²·Var y` — `EpistemicEffectsNS.gMulMeta` at
    n = 0 — which silently imposes `‖d·y‖² = ‖d‖²‖y‖²` (norm multiplicativity). -/
def gMulShortcut (n : Nat) (x y : Vec) (ma mb : KMeta) : KMeta :=
  { gumVar := normSq (2^n) y * ma.gumVar + normSq (2^n) x * mb.gumVar
  , conf := if ma.conf ≤ mb.conf then ma.conf else mb.conf }

theorem gAddMeta_valid {ma mb : KMeta} (ha : kvalid ma) (hb : kvalid mb) : kvalid (gAddMeta ma mb) := by
  unfold kvalid gAddMeta at *; simp only
  by_cases h : ma.conf ≤ mb.conf <;> simp only [h, if_true, if_false] <;> omega
theorem gMulMeta_valid {n x a y b} {ma mb : KMeta} (ha : kvalid ma) (hb : kvalid mb) :
    kvalid (gMulMeta n x a y b ma mb) := by
  unfold kvalid gMulMeta at *; simp only
  by_cases h : ma.conf ≤ mb.conf <;> simp only [h, if_true, if_false] <;> omega

/-- Small-step CBV reduction at level `n`, carrying true forms. -/
inductive Step (n : Nat) : Expr → Expr → Prop where
  | meas_red : ∀ v c cf s,
      Step n (.measure v c cf s) (.kraw v ⟨normSq (2^n) c, cf⟩ [(s, c)])
  | cert_red : ∀ v, Step n (.certain v) (.kraw v ⟨0, 1000⟩ [])
  | opaque_red : ∀ v m a, Step n (.opaque (.kraw v m a)) (.kraw v m a)
  | opaque_arg : ∀ e e', Step n e e' → Step n (.opaque e) (.opaque e')
  | kadd_red : ∀ x ma a y mb b,
      Step n (.kadd (.kraw x ma a) (.kraw y mb b)) (.kraw (vadd x y) (gAddMeta ma mb) (a ++ b))
  | kadd_l : ∀ e e' r, Step n e e' → Step n (.kadd e r) (.kadd e' r)
  | kadd_r : ∀ v e e', IsValue v → Step n e e' → Step n (.kadd v e) (.kadd v e')
  | kmul_red : ∀ x ma a y mb b,
      Step n (.kmul (.kraw x ma a) (.kraw y mb b))
        (.kraw (cdMul n x y) (gMulMeta n x a y b ma mb) (scaleR n a y ++ scaleL n x b))
  | kmul_l : ∀ e e' r, Step n e e' → Step n (.kmul e r) (.kmul e' r)
  | kmul_r : ∀ v e e', IsValue v → Step n e e' → Step n (.kmul v e) (.kmul v e')

inductive Steps (n : Nat) : Expr → Expr → Prop where
  | refl : ∀ e, Steps n e e
  | step : ∀ {e e' e''}, Step n e e' → Steps n e' e'' → Steps n e e''

-- ================================================================
-- §G. Axis 2 in CD(n): exactness and no anti-garbling (generalizing EpistemicEffectsNS)
-- ================================================================

/-- Every runtime value reports its TRUE first-order variance. -/
def Exact (n : Nat) : Expr → Prop
  | .kraw _ m a => m.gumVar = trueVar (2^n) a
  | .measure _ _ _ _ => True
  | .certain _ => True
  | .opaque e => Exact n e
  | .kadd a b => Exact n a ∧ Exact n b
  | .kmul a b => Exact n a ∧ Exact n b

/-- No independence-assuming operator on runtime values has correlated operands. -/
def AGFree (n : Nat) : Expr → Prop
  | .kadd a b =>
      (∀ x ma a' y mb b', a = .kraw x ma a' → b = .kraw y mb b' → innerA (2^n) a' b' = 0) ∧
      AGFree n a ∧ AGFree n b
  | .kmul a b =>
      (∀ x ma a' y mb b', a = .kraw x ma a' → b = .kraw y mb b' → innerA (2^n) a' b' = 0) ∧
      AGFree n a ∧ AGFree n b
  | .opaque e => AGFree n e
  | .kraw _ _ _ => True
  | .measure _ _ _ _ => True
  | .certain _ => True

instance (n : Nat) (v : Vec) (m : KMeta) (a : Aff) : Decidable (Exact n (.kraw v m a)) :=
  inferInstanceAs (Decidable (m.gumVar = trueVar (2^n) a))

theorem typed_agfree {n : Nat} {e : Expr} {T : Ty} (h : HasTy n e T) : AGFree n e := by
  induction h with
  | t_kraw => trivial
  | t_measure => trivial
  | t_certain => trivial
  | t_opaque _ _ _ _ ih => exact ih
  | t_kadd a b Na Qa Nb Qb ha hb hd iha ihb =>
    refine ⟨?_, iha, ihb⟩
    intro x ma a' y mb b' hea heb; subst hea; subst heb
    cases ha with | t_kraw _ _ _ _ _ _ hca _ _ =>
    cases hb with | t_kraw _ _ _ _ _ _ hcb _ _ =>
    exact innerA_zero_of_ns _ hca hcb hd
  | t_kmul a b Na Qa Nb Qb ha hb hd iha ihb =>
    refine ⟨?_, iha, ihb⟩
    intro x ma a' y mb b' hea heb; subst hea; subst heb
    cases ha with | t_kraw _ _ _ _ _ _ hca _ _ =>
    cases hb with | t_kraw _ _ _ _ _ _ hcb _ _ =>
    exact innerA_zero_of_ns _ hca hcb hd

theorem qCoversAff_single {n : Nat} {Q : QS} {s : Nat} {c : Vec} (h : qCovers n Q c) :
    qCoversAff n Q [(s, c)] := by
  intro p hp; simp at hp; subst hp; exact h
theorem qCoversAff_nil (n : Nat) (Q : QS) : qCoversAff n Q [] := fun _ h => by simp at h
theorem qCovers_top (n : Nat) (v : Vec) : qCovers n qTop v := trivial
theorem qCoversAff_top (n : Nat) (a : Aff) : qCoversAff n qTop a := fun _ _ => trivial

/-- Preservation: the `⟨N, Q⟩` type is preserved by reduction. -/
theorem preservation {n : Nat} {e e' : Expr} (hs : Step n e e') :
    ∀ {T : Ty}, HasTy n e T → HasTy n e' T := by
  induction hs with
  | meas_red v c cf s =>
    intro T h; cases h with | t_measure _ _ _ _ Q h0 h1 hv hc =>
    exact .t_kraw _ _ _ _ _ ⟨h0, h1⟩ (covers_single s c) hv (qCoversAff_single hc)
  | cert_red v =>
    intro T h; cases h with | t_certain _ Q hv =>
    exact .t_kraw _ _ _ _ _ ⟨by decide, by decide⟩ covers_empty hv (qCoversAff_nil n Q)
  | opaque_red v m a =>
    intro T h; cases h with | t_opaque _ N Q h' =>
    cases h' with | t_kraw _ _ _ _ _ hm _ _ _ =>
    exact .t_kraw _ _ _ _ _ hm (covers_top a) (qCovers_top n v) (qCoversAff_top n a)
  | opaque_arg e e' _ ih =>
    intro T h; cases h with | t_opaque _ N Q h' => exact .t_opaque _ _ _ (ih h')
  | kadd_red x ma a y mb b =>
    intro T h; cases h with | t_kadd _ _ Na Qa Nb Qb ha hb hd =>
    cases ha with | t_kraw _ _ _ _ _ hma hca hqa hqaa =>
    cases hb with | t_kraw _ _ _ _ _ hmb hcb hqb hqab =>
    exact .t_kraw _ _ _ _ _ (gAddMeta_valid hma hmb) (covers_union hca hcb) (qCovers_vadd hqa hqb)
      (qCoversAff_append (fun p hp => qCovers_qUnion_left (hqaa p hp))
                         (fun p hp => qCovers_qUnion_right (hqab p hp)))
  | kadd_l e e' r _ ih =>
    intro T h; cases h with | t_kadd _ _ Na Qa Nb Qb ha hb hd => exact .t_kadd _ _ _ _ _ _ (ih ha) hb hd
  | kadd_r v e e' _ _ ih =>
    intro T h; cases h with | t_kadd _ _ Na Qa Nb Qb ha hb hd => exact .t_kadd _ _ _ _ _ _ ha (ih hb) hd
  | kmul_red x ma a y mb b =>
    intro T h; cases h with | t_kmul _ _ Na Qa Nb Qb ha hb hd =>
    cases ha with | t_kraw _ _ _ _ _ hma hca hqa hqaa =>
    cases hb with | t_kraw _ _ _ _ _ hmb hcb hqb hqab =>
    exact .t_kraw _ _ _ _ _ (gMulMeta_valid hma hmb)
      (covers_union (covers_scaleR n y hca) (covers_scaleL n x hcb))
      (qCovers_cdMul hqa hqb)
      (qCoversAff_append (qCoversAff_scaleR hqaa hqb) (qCoversAff_scaleL hqa hqab))
  | kmul_l e e' r _ ih =>
    intro T h; cases h with | t_kmul _ _ Na Qa Nb Qb ha hb hd => exact .t_kmul _ _ _ _ _ _ (ih ha) hb hd
  | kmul_r v e e' _ _ ih =>
    intro T h; cases h with | t_kmul _ _ Na Qa Nb Qb ha hb hd => exact .t_kmul _ _ _ _ _ _ ha (ih hb) hd

theorem trueVar_single (m : Nat) (s : Nat) (c : Vec) : trueVar m [(s, c)] = normSq m c := by
  simp only [trueVar, innerA, coeff, if_true, normSq]; rw [vadd_zero]; omega

/-- EXACT PRESERVATION (Axis 2, every level `n`): under NS typing, every reduct reports its
    true first-order variance — with the sensitivity propagator, on ANY Cayley–Dickson algebra
    (no composition-algebra hypothesis). -/
theorem exact_preservation {n : Nat} {e e' : Expr} (hs : Step n e e') :
    ∀ {T : Ty}, HasTy n e T → Exact n e → Exact n e' := by
  induction hs with
  | meas_red v c cf s => intro T _ _; show _ = trueVar _ _; rw [trueVar_single]
  | cert_red v => intro T _ _; rfl
  | opaque_red v m a => intro T _ hx; exact hx
  | opaque_arg e e' _ ih =>
    intro T h hx; cases h with | t_opaque _ N Q h' => exact ih h' hx
  | kadd_red x ma a y mb b =>
    intro T h hx
    cases h with | t_kadd _ _ Na Qa Nb Qb ha hb hd =>
    cases ha with | t_kraw _ _ _ _ _ _ hca _ _ =>
    cases hb with | t_kraw _ _ _ _ _ _ hcb _ _ =>
    rcases hx with ⟨hxa, hxb⟩
    show (gAddMeta ma mb).gumVar = trueVar (2^n) (a ++ b)
    rw [trueVar_append, innerA_zero_of_ns _ hca hcb hd]
    simp only [gAddMeta]; rw [hxa, hxb]; omega
  | kadd_l e e' r _ ih =>
    intro T h hx; cases h with | t_kadd _ _ Na Qa Nb Qb ha hb hd => exact ⟨ih ha hx.1, hx.2⟩
  | kadd_r v e e' _ _ ih =>
    intro T h hx; cases h with | t_kadd _ _ Na Qa Nb Qb ha hb hd => exact ⟨hx.1, ih hb hx.2⟩
  | kmul_red x ma a y mb b =>
    intro T h _
    cases h with | t_kmul _ _ Na Qa Nb Qb ha hb hd =>
    cases ha with | t_kraw _ _ _ _ _ _ hca _ _ =>
    cases hb with | t_kraw _ _ _ _ _ _ hcb _ _ =>
    show (gMulMeta n x a y b ma mb).gumVar = trueVar (2^n) (scaleR n a y ++ scaleL n x b)
    rw [trueVar_append, innerA_zero_of_ns _ (covers_scaleR n y hca) (covers_scaleL n x hcb) hd]
    simp only [gMulMeta]; omega
  | kmul_l e e' r _ ih =>
    intro T h hx; cases h with | t_kmul _ _ Na Qa Nb Qb ha hb hd => exact ⟨ih ha hx.1, hx.2⟩
  | kmul_r v e e' _ _ ih =>
    intro T h hx; cases h with | t_kmul _ _ Na Qa Nb Qb ha hb hd => exact ⟨hx.1, ih hb hx.2⟩

/-- Theorem 6.4 lifted to CD(n): along every evaluation of a well-typed, initially-exact
    program, typing, exactness and anti-garbling-freedom hold at every reduct. -/
theorem soundness_star {n : Nat} {e e' : Expr} (hs : Steps n e e') :
    ∀ {T : Ty}, HasTy n e T → Exact n e → HasTy n e' T ∧ Exact n e' ∧ AGFree n e' := by
  induction hs with
  | refl e => intro T h hx; exact ⟨h, hx, typed_agfree h⟩
  | step s1 _ ih => intro T h hx; exact ih (preservation s1 h) (exact_preservation s1 h hx)

-- ================================================================
-- §H. Axis 1 in the calculus: re-association, its exact gap, and the certified rewrite
-- ================================================================

theorem scaleR_scaleR (n : Nat) (a : Aff) (y z : Vec) :
    scaleR n (scaleR n a y) z = a.map (fun p => (p.1, cdMul n (cdMul n p.2 y) z)) := by
  simp [scaleR, List.map_map, Function.comp]
theorem scaleR_scaleL (n : Nat) (x : Vec) (a : Aff) (z : Vec) :
    scaleR n (scaleL n x a) z = a.map (fun p => (p.1, cdMul n (cdMul n x p.2) z)) := by
  simp [scaleR, scaleL, List.map_map, Function.comp]
theorem scaleL_scaleR (n : Nat) (x : Vec) (a : Aff) (z : Vec) :
    scaleL n x (scaleR n a z) = a.map (fun p => (p.1, cdMul n x (cdMul n p.2 z))) := by
  simp [scaleR, scaleL, List.map_map, Function.comp]
theorem scaleL_scaleL (n : Nat) (x y : Vec) (a : Aff) :
    scaleL n x (scaleL n y a) = a.map (fun p => (p.1, cdMul n x (cdMul n y p.2))) := by
  simp [scaleL, List.map_map, Function.comp]
theorem scaleR_append (n : Nat) (a b : Aff) (y : Vec) :
    scaleR n (a ++ b) y = scaleR n a y ++ scaleR n b y := by simp [scaleR, List.map_append]
theorem scaleL_append (n : Nat) (x : Vec) (a b : Aff) :
    scaleL n x (a ++ b) = scaleL n x a ++ scaleL n x b := by simp [scaleL, List.map_append]

/-- The two parenthesizations of a triple product of VALUES, and what they reduce to. -/
def srcTriple (x : Vec) (mx : KMeta) (ax : Aff) (y : Vec) (my : KMeta) (ay : Aff)
    (z : Vec) (mz : KMeta) (az : Aff) : Expr :=
  .kmul (.kmul (.kraw x mx ax) (.kraw y my ay)) (.kraw z mz az)
def tgtTriple (x : Vec) (mx : KMeta) (ax : Aff) (y : Vec) (my : KMeta) (ay : Aff)
    (z : Vec) (mz : KMeta) (az : Aff) : Expr :=
  .kmul (.kraw x mx ax) (.kmul (.kraw y my ay) (.kraw z mz az))

/-- True form of `(xy)z`: `∂((xy)z) = (∂x·y)·z + (x·∂y)·z + (xy)·∂z`. -/
def srcForm (n : Nat) (x : Vec) (ax : Aff) (y : Vec) (ay : Aff) (z : Vec) (az : Aff) : Aff :=
  scaleR n (scaleR n ax y ++ scaleL n x ay) z ++ scaleL n (cdMul n x y) az
/-- True form of `x(yz)`: `∂(x(yz)) = ∂x·(yz) + x·(∂y·z) + x·(y·∂z)`. -/
def tgtForm (n : Nat) (x : Vec) (ax : Aff) (y : Vec) (ay : Aff) (z : Vec) (az : Aff) : Aff :=
  scaleR n ax (cdMul n y z) ++ scaleL n x (scaleR n ay z ++ scaleL n y az)

def srcMeta (n : Nat) (x : Vec) (mx : KMeta) (ax : Aff) (y : Vec) (my : KMeta) (ay : Aff)
    (z : Vec) (mz : KMeta) (az : Aff) : KMeta :=
  gMulMeta n (cdMul n x y) (scaleR n ax y ++ scaleL n x ay) z az (gMulMeta n x ax y ay mx my) mz
def tgtMeta (n : Nat) (x : Vec) (mx : KMeta) (ax : Aff) (y : Vec) (my : KMeta) (ay : Aff)
    (z : Vec) (mz : KMeta) (az : Aff) : KMeta :=
  gMulMeta n x ax (cdMul n y z) (scaleR n ay z ++ scaleL n y az) mx (gMulMeta n y ay z az my mz)

theorem src_steps (n : Nat) (x mx ax y my ay z mz az) :
    Steps n (srcTriple x mx ax y my ay z mz az)
      (.kraw (cdMul n (cdMul n x y) z) (srcMeta n x mx ax y my ay z mz az)
             (srcForm n x ax y ay z az)) :=
  .step (.kmul_l _ _ _ (.kmul_red _ _ _ _ _ _)) (.step (.kmul_red _ _ _ _ _ _) (.refl _))
theorem tgt_steps (n : Nat) (x mx ax y my ay z mz az) :
    Steps n (tgtTriple x mx ax y my ay z mz az)
      (.kraw (cdMul n x (cdMul n y z)) (tgtMeta n x mx ax y my ay z mz az)
             (tgtForm n x ax y ay z az)) :=
  .step (.kmul_r _ _ _ (.v_kraw _ _ _) (.kmul_red _ _ _ _ _ _)) (.step (.kmul_red _ _ _ _ _ _) (.refl _))

/-- THE PAYLOAD GAP is the associator (by definition — stated for the record). -/
theorem reassoc_payload_gap (n : Nat) (x y z : Vec) :
    vsub (cdMul n (cdMul n x y) z) (cdMul n x (cdMul n y z)) = assoc n x y z := rfl

/-- THE SENSITIVITY GAP (§3B identity, in the calculus): per source `s`, the true forms of
    `(xy)z` and `x(yz)` differ by EXACTLY the sum of three associators, one per slot. -/
theorem reassoc_sensitivity_gap (n : Nat) (x : Vec) (ax : Aff) (y : Vec) (ay : Aff)
    (z : Vec) (az : Aff) (s : Nat) :
    vsub (coeff (srcForm n x ax y ay z az) s) (coeff (tgtForm n x ax y ay z az) s)
      = vadd (vadd (assoc n (coeff ax s) y z) (assoc n x (coeff ay s) z)) (assoc n x y (coeff az s)) := by
  unfold srcForm tgtForm
  rw [coeff_append, coeff_append, coeff_scaleR, coeff_scaleL, coeff_scaleR, coeff_scaleL,
      coeff_append, coeff_append, coeff_scaleR, coeff_scaleL, coeff_scaleR, coeff_scaleL,
      cdMul_add_left, cdMul_add_right]
  funext k; simp only [vsub, vadd, assoc]; omega

theorem affEqOn_map {m : Nat} (a : Aff) (f g : Nat × Vec → Vec)
    (h : ∀ p ∈ a, eqOn m (f p) (g p)) :
    AffEqOn m (a.map (fun p => (p.1, f p))) (a.map (fun p => (p.1, g p))) := by
  induction a with
  | nil => trivial
  | cons p r ih =>
    exact ⟨rfl, h p (List.mem_cons_self), ih (fun q hq => h q (List.mem_cons_of_mem _ hq))⟩

/-- A certified triple has pairwise-equal true forms (each entry differs by ONE covered
    associator, which the certificate kills). -/
theorem reassoc_forms_eq {n : Nat} {Qx Qy Qz : QS} {x y z : Vec} {ax ay az : Aff}
    (hc : assocCert n Qx Qy Qz = true)
    (hx : qCovers n Qx x) (hy : qCovers n Qy y) (hz : qCovers n Qz z)
    (hax : qCoversAff n Qx ax) (hay : qCoversAff n Qy ay) (haz : qCoversAff n Qz az) :
    AffEqOn (2^n) (srcForm n x ax y ay z az) (tgtForm n x ax y ay z az) := by
  unfold srcForm tgtForm
  rw [scaleR_append, scaleR_scaleR, scaleR_scaleL, scaleL_append, scaleL_scaleR, scaleL_scaleL,
      List.append_assoc]
  apply affEqOn_append
  · apply affEqOn_map; intro p hp k hk
    have h0 := assoc_zero_of_cert hc (hax p hp) hy hz k hk
    simp only [assoc, vsub, vzero] at h0; omega
  apply affEqOn_append
  · apply affEqOn_map; intro p hp k hk
    have h0 := assoc_zero_of_cert hc hx (hay p hp) hz k hk
    simp only [assoc, vsub, vzero] at h0; omega
  · apply affEqOn_map; intro p hp k hk
    have h0 := assoc_zero_of_cert hc hx hy (haz p hp) k hk
    simp only [assoc, vsub, vzero] at h0; omega

/-- THE FUSION THEOREM (certified re-association is garbling-free). If the three operands are
    well-typed at `⟨Nx,Qx⟩, ⟨Ny,Qy⟩, ⟨Nz,Qz⟩`, the source `(xy)z` passes the Axis-2 checks,
    and the Axis-1 certificate `assocCert Qx Qy Qz` holds, then the target `x(yz)`:
      (1) is well-typed (with the re-associated `⟨N, Q⟩`);
      (2) evaluates to the SAME payload on live coordinates;
      (3) evaluates to a pairwise-EQUAL true form;
      (4) REPORTS the same variance — and both reports are exact.
    Neither certificate alone suffices: drop Axis 2 and (4) fails (§I `w2`); drop Axis 1 and
    (2) fails (§I `w1`). -/
theorem reassoc_sound {n : Nat} {Nx Ny Nz : NS} {Qx Qy Qz : QS}
    {x y z : Vec} {mx my mz : KMeta} {ax ay az : Aff}
    (hc : assocCert n Qx Qy Qz = true)
    (hX : HasTy n (.kraw x mx ax) ⟨Nx, Qx⟩)
    (hY : HasTy n (.kraw y my ay) ⟨Ny, Qy⟩)
    (hZ : HasTy n (.kraw z mz az) ⟨Nz, Qz⟩)
    (hxy : nsDisjoint Nx Ny = true) (hxyz : nsDisjoint (nsUnion Nx Ny) Nz = true)
    (hex : Exact n (.kraw x mx ax)) (hey : Exact n (.kraw y my ay)) (hez : Exact n (.kraw z mz az)) :
    HasTy n (tgtTriple x mx ax y my ay z mz az) ⟨nsUnion Nx (nsUnion Ny Nz), qProd Qx (qProd Qy Qz)⟩ ∧
    eqOn (2^n) (cdMul n (cdMul n x y) z) (cdMul n x (cdMul n y z)) ∧
    AffEqOn (2^n) (srcForm n x ax y ay z az) (tgtForm n x ax y ay z az) ∧
    (srcMeta n x mx ax y my ay z mz az).gumVar = (tgtMeta n x mx ax y my ay z mz az).gumVar ∧
    (srcMeta n x mx ax y my ay z mz az).gumVar = trueVar (2^n) (srcForm n x ax y ay z az) := by
  have hX' := hX; have hY' := hY; have hZ' := hZ
  cases hX' with | t_kraw _ _ _ _ _ _ _ hqx hqax =>
  cases hY' with | t_kraw _ _ _ _ _ _ _ hqy hqay =>
  cases hZ' with | t_kraw _ _ _ _ _ _ _ hqz hqaz =>
  -- (1) typing of the target: Axis-2 premises are re-association invariant
  have hnd := (nsDisjoint_reassoc_invariant Nx Ny Nz).mp ⟨hxy, hxyz⟩
  have hT : HasTy n (tgtTriple x mx ax y my ay z mz az)
      ⟨nsUnion Nx (nsUnion Ny Nz), qProd Qx (qProd Qy Qz)⟩ :=
    .t_kmul _ _ _ _ _ _ hX (.t_kmul _ _ _ _ _ _ hY hZ hnd.1) hnd.2
  have hS : HasTy n (srcTriple x mx ax y my ay z mz az)
      ⟨nsUnion (nsUnion Nx Ny) Nz, qProd (qProd Qx Qy) Qz⟩ :=
    .t_kmul _ _ _ _ _ _ (.t_kmul _ _ _ _ _ _ hX hY hxy) hZ hxyz
  -- (2) payload
  have hp : eqOn (2^n) (cdMul n (cdMul n x y) z) (cdMul n x (cdMul n y z)) := by
    intro k hk
    have h0 := assoc_zero_of_cert hc hqx hqy hqz k hk
    simp only [assoc, vsub, vzero] at h0; omega
  -- (3) true forms
  have hf := reassoc_forms_eq hc hqx hqy hqz hqax hqay hqaz
  -- (4) reported: both sides are exact (Axis 2), and the true forms agree (Axis 1)
  have hsx : Exact n (srcTriple x mx ax y my ay z mz az) := ⟨⟨hex, hey⟩, hez⟩
  have htx : Exact n (tgtTriple x mx ax y my ay z mz az) := ⟨hex, ⟨hey, hez⟩⟩
  have es := (soundness_star (src_steps n x mx ax y my ay z mz az) hS hsx).2.1
  have et := (soundness_star (tgt_steps n x mx ax y my ay z mz az) hT htx).2.1
  have es' : (srcMeta n x mx ax y my ay z mz az).gumVar = trueVar (2^n) (srcForm n x ax y ay z az) := es
  have et' : (tgtMeta n x mx ax y my ay z mz az).gumVar = trueVar (2^n) (tgtForm n x ax y ay z az) := et
  refine ⟨hT, hp, hf, ?_, es'⟩
  rw [es', et', trueVar_congr hf]

-- ================================================================
-- §I. Orthogonality witnesses (kernel-checked): each certificate fails ALONE
-- ================================================================

/-- W1 (Axis 2 clean, Axis 1 fails): octonion basis centers `e₁, e₂, e₄` on DISJOINT sources
    0, 1, 2. NS admits the program; the triple is non-Fano, so `assocCert` refuses, and the
    two parenthesizations really do differ (by `2·e₇`). -/
def w1x : Expr := .kraw (e 1) ⟨1, 1000⟩ [(0, e 1)]
def w1y : Expr := .kraw (e 2) ⟨1, 1000⟩ [(1, e 2)]
def w1z : Expr := .kraw (e 4) ⟨1, 1000⟩ [(2, e 4)]

theorem w1_typable :
    HasTy 3 (.kmul (.kmul w1x w1y) w1z)
      ⟨nsUnion (nsUnion (nsSingle 0) (nsSingle 1)) (nsSingle 2),
       qProd (qProd (some [1]) (some [2])) (some [4])⟩ := by
  apply HasTy.t_kmul _ _ _ _ _ _ _ _ (by decide)
  · apply HasTy.t_kmul _ _ _ _ _ _ _ _ (by decide)
    · exact .t_kraw _ _ _ _ _ ⟨by decide, by decide⟩ (covers_single 0 (e 1)) (by decide)
        (qCoversAff_single (by decide))
    · exact .t_kraw _ _ _ _ _ ⟨by decide, by decide⟩ (covers_single 1 (e 2)) (by decide)
        (qCoversAff_single (by decide))
  · exact .t_kraw _ _ _ _ _ ⟨by decide, by decide⟩ (covers_single 2 (e 4)) (by decide)
      (qCoversAff_single (by decide))

theorem w1_cert_refused : assocCert 3 (some [1]) (some [2]) (some [4]) = false := by decide
theorem w1_reassoc_changes_value :
    ¬ eqOn 8 (cdMul 3 (cdMul 3 (e 1) (e 2)) (e 4)) (cdMul 3 (e 1) (cdMul 3 (e 2) (e 4))) := by decide
/-- …and the sensitivity of the product also changes: source 0's coefficient differs by `[e₁,e₂,e₄]`. -/
theorem w1_sensitivity_changes :
    ¬ eqOn 8 (coeff (srcForm 3 (e 1) [(0, e 1)] (e 2) [(1, e 2)] (e 4) [(2, e 4)]) 0)
             (coeff (tgtForm 3 (e 1) [(0, e 1)] (e 2) [(1, e 2)] (e 4) [(2, e 4)]) 0) := by decide

/-- W1′ (both certificates hold): the Fano triple `e₁, e₂, e₃` on disjoint sources. Certified,
    and the fusion theorem applies. -/
theorem w1'_cert : assocCert 3 (some [1]) (some [2]) (some [3]) = true := by decide
theorem w1'_reassoc_sound :
    eqOn 8 (cdMul 3 (cdMul 3 (e 1) (e 2)) (e 3)) (cdMul 3 (e 1) (cdMul 3 (e 2) (e 3))) ∧
    (srcMeta 3 (e 1) ⟨1,1000⟩ [(0, e 1)] (e 2) ⟨1,1000⟩ [(1, e 2)] (e 3) ⟨1,1000⟩ [(2, e 3)]).gumVar
      = (tgtMeta 3 (e 1) ⟨1,1000⟩ [(0, e 1)] (e 2) ⟨1,1000⟩ [(1, e 2)] (e 3) ⟨1,1000⟩ [(2, e 3)]).gumVar := by
  have hX : HasTy 3 (.kraw (e 1) ⟨1,1000⟩ [(0, e 1)]) ⟨nsSingle 0, some [1]⟩ :=
    .t_kraw _ _ _ _ _ ⟨by decide, by decide⟩ (covers_single 0 (e 1)) (by decide) (qCoversAff_single (by decide))
  have hY : HasTy 3 (.kraw (e 2) ⟨1,1000⟩ [(1, e 2)]) ⟨nsSingle 1, some [2]⟩ :=
    .t_kraw _ _ _ _ _ ⟨by decide, by decide⟩ (covers_single 1 (e 2)) (by decide) (qCoversAff_single (by decide))
  have hZ : HasTy 3 (.kraw (e 3) ⟨1,1000⟩ [(2, e 3)]) ⟨nsSingle 2, some [3]⟩ :=
    .t_kraw _ _ _ _ _ ⟨by decide, by decide⟩ (covers_single 2 (e 3)) (by decide) (qCoversAff_single (by decide))
  have h := reassoc_sound w1'_cert hX hY hZ (by decide) (by decide) (by decide) (by decide) (by decide)
  exact ⟨h.2.1, h.2.2.2.1⟩

/-- W2 (Axis 1 clean, Axis 2 fails): at n = 0 (ℝ) EVERYTHING is associative — `assocCert` is
    unconditionally `true` — yet `x · x` on a shared source is untypable (E230) and understates. -/
theorem assocCert_level0 (Qx Qy Qz : QS) : assocCert 0 Qx Qy Qz = true := by
  simp [assocCert]
def w2x : Expr := .kraw (fun _ => 10) ⟨1, 1000⟩ [(0, fun _ => 1)]
theorem w2_untypable : ∀ T, ¬ HasTy 0 (.kmul w2x w2x) T := by
  intro T h
  cases h with | t_kmul _ _ Na Qa Nb Qb ha hb hd =>
  cases ha with | t_kraw _ _ _ _ _ _ hca _ _ =>
  cases hb with | t_kraw _ _ _ _ _ _ hcb _ _ =>
  have ma : nsMem 0 Na := hca (0, fun _ => 1) (List.mem_cons_self)
  have mb : nsMem 0 Nb := hcb (0, fun _ => 1) (List.mem_cons_self)
  rw [nsDisjoint_of_shared ma mb] at hd; exact Bool.noConfusion hd
theorem w2_understates :
    (gMulMeta 0 (fun _ => 10) [(0, fun _ => 1)] (fun _ => 10) [(0, fun _ => 1)] ⟨1,1000⟩ ⟨1,1000⟩).gumVar = 200 ∧
    trueVar 1 (scaleR 0 [(0, fun _ => (1:Int))] (fun _ => 10) ++ scaleL 0 (fun _ => 10) [(0, fun _ => 1)]) = 400 := by
  decide

-- ================================================================
-- §J. The THIRD axis: the variance shortcut imposes norm-multiplicativity (Hurwitz boundary)
-- ================================================================

/-- Sedenion witness (n = 4). Disjoint sources (X on source 0, Y exact), ONE product, NO
    re-association — both certificates hold — and the GUM shortcut STILL understates:
    reported `‖y‖²·Var x = 2·2 = 4`, true first-order variance `‖d·y‖² = 8`. The sensitivity
    propagator reports 8. The shortcut's silent identity is `‖d·y‖² = ‖d‖²‖y‖²`, false in 𝕊. -/
def sedD : Vec := vadd (e 1) (e 10)
def sedY : Vec := vadd (e 4) (e 15)
theorem sed_shortcut_understates :
    (gMulShortcut 4 vzero sedY ⟨normSq 16 sedD, 1000⟩ ⟨0, 1000⟩).gumVar = 4 ∧
    (gMulMeta 4 vzero [(0, sedD)] sedY [] ⟨normSq 16 sedD, 1000⟩ ⟨0, 1000⟩).gumVar = 8 ∧
    trueVar 16 (scaleR 4 [(0, sedD)] sedY ++ scaleL 4 vzero []) = 8 := by
  decide
theorem sed_x_typable :
    HasTy 4 (.kmul (.kraw vzero ⟨normSq 16 sedD, 1000⟩ [(0, sedD)]) (.kraw sedY ⟨0, 1000⟩ []))
      ⟨nsUnion (nsSingle 0) nsEmpty, qProd (some [0, 1, 10]) (some [4, 15])⟩ := by
  apply HasTy.t_kmul _ _ _ _ _ _ _ _ (by decide)
  · exact .t_kraw _ _ _ _ _ ⟨by decide, by decide⟩ (covers_single 0 sedD) (by decide)
      (qCoversAff_single (by decide))
  · exact .t_kraw _ _ _ _ _ ⟨by decide, by decide⟩ covers_empty (by decide) (qCoversAff_nil 4 _)

/-- Octonion control (n = 3, a composition algebra): the same shape, and the shortcut is exact
    — `‖d·y‖² = 4 = ‖d‖²‖y‖²`. The third axis is exactly the Hurwitz boundary. -/
def octD : Vec := vadd (e 1) (e 2)
def octY : Vec := vadd (e 4) (e 7)
theorem oct_shortcut_exact :
    (gMulShortcut 3 vzero octY ⟨normSq 8 octD, 1000⟩ ⟨0, 1000⟩).gumVar = 4 ∧
    trueVar 8 (scaleR 3 [(0, octD)] octY ++ scaleL 3 vzero []) = 4 := by
  decide

/-- Zero-divisor control: the shortcut is not even monotone — on `(e₁+e₁₀)(e₅+e₁₄) = 0` it
    OVER-states (reports 4, true 0). Non-multiplicative norms garble in both directions. -/
theorem sed_shortcut_overstates :
    (gMulShortcut 4 vzero (vadd (e 5) (e 14)) ⟨2, 1000⟩ ⟨0, 1000⟩).gumVar = 4 ∧
    trueVar 16 (scaleR 4 [(0, sedD)] (vadd (e 5) (e 14)) ++ scaleL 4 vzero []) = 0 := by
  decide

/-- At n = 0 (ℝ) the shortcut IS the sensitivity propagator — which is why
    `EpistemicEffectsNS` never met the third axis. -/
theorem inner1 (a b : Vec) : inner 1 a b = a 0 * b 0 := by simp [inner, sumR]
theorem cdMul0 (a b : Vec) (k : Nat) : cdMul 0 a b k = cdSigma 0 k 0 * a 0 * b k := by
  simp [cdMul, sumR, Nat.zero_xor]
theorem cdMul0_0 (a b : Vec) : cdMul 0 a b 0 = a 0 * b 0 := by
  rw [cdMul0]; simp [cdSigma]
theorem mul_swap4 (A Y B : Int) : A * Y * (B * Y) = Y * Y * (A * B) := by
  rw [Int.mul_assoc, Int.mul_left_comm Y B Y, ← Int.mul_assoc, Int.mul_comm (A * B) (Y * Y)]
theorem mul_swap4' (X A B : Int) : X * A * (X * B) = X * X * (A * B) := by
  rw [Int.mul_assoc, Int.mul_left_comm A X B, ← Int.mul_assoc]
theorem innerA1_scaleR (a b : Aff) (y : Vec) :
    innerA 1 (scaleR 0 a y) (scaleR 0 b y) = y 0 * y 0 * innerA 1 a b := by
  induction a with
  | nil => simp [scaleR, innerA]
  | cons p r ih =>
    rcases p with ⟨s, c⟩
    have hcons : scaleR 0 ((s, c) :: r) y = (s, cdMul 0 c y) :: scaleR 0 r y := rfl
    rw [hcons]; simp only [innerA]
    rw [ih, coeff_scaleR, inner1, inner1, cdMul0_0, cdMul0_0, mul_swap4, Int.mul_add]
theorem innerA1_scaleL (x : Vec) (a b : Aff) :
    innerA 1 (scaleL 0 x a) (scaleL 0 x b) = x 0 * x 0 * innerA 1 a b := by
  induction a with
  | nil => simp [scaleL, innerA]
  | cons p r ih =>
    rcases p with ⟨s, c⟩
    have hcons : scaleL 0 x ((s, c) :: r) = (s, cdMul 0 x c) :: scaleL 0 x r := rfl
    rw [hcons]; simp only [innerA]
    rw [ih, coeff_scaleL, inner1, inner1, cdMul0_0, cdMul0_0, mul_swap4', Int.mul_add]
theorem shortcut_eq_sensitivity_level0 (x : Vec) (a : Aff) (y : Vec) (b : Aff) (ma mb : KMeta)
    (hx : ma.gumVar = trueVar 1 a) (hy : mb.gumVar = trueVar 1 b) :
    (gMulShortcut 0 x y ma mb).gumVar = (gMulMeta 0 x a y b ma mb).gumVar := by
  simp only [gMulShortcut, gMulMeta, trueVar, normSq]
  rw [hx, hy, innerA1_scaleR, innerA1_scaleL, inner1, inner1]; simp only [trueVar]


-- ================================================================
-- §K. Hurwitz in the kernel: norm multiplicativity from the polarized sign identity,
--     for every level where it holds (≤ 3) — and its failure at 4. This is what makes the
--     variance SHORTCUT exact on ℝ,ℂ,ℍ,𝕆 and unsound on 𝕊 (the third axis, as a theorem).
-- ================================================================

theorem inner_smul_left (m : Nat) (c : Int) (a b : Vec) : inner m (smul c a) b = c * inner m a b := by
  unfold inner; rw [← sumR_smul]; apply sumR_congr; intro i _; simp only [smul]; exact Int.mul_assoc _ _ _
theorem inner_smul_right (m : Nat) (c : Int) (a b : Vec) : inner m a (smul c b) = c * inner m a b := by
  rw [inner_comm, inner_smul_left, inner_comm]
theorem inner_zero_left_of {m : Nat} {a : Vec} (b : Vec) (ha : eqOn m a vzero) : inner m a b = 0 := by
  rw [inner_congr ha (eqOn_refl m b)]; exact inner_zero_left m b

/-- GENERIC SUPPORT INDUCTION: an additive, ℤ-homogeneous functional that reads only live
    coordinates and kills every basis vector kills every vector. (Abstracts the
    `assoc_slot*` pattern.) -/
theorem lin_zero_of_basis (n : Nat) (F : Vec → Int)
    (hadd : ∀ u v, F (vadd u v) = F u + F v)
    (hsmul : ∀ c u, F (smul c u) = c * F u)
    (hlive : ∀ u, eqOn (2^n) u vzero → F u = 0)
    (hb : ∀ i, i < 2^n → F (e i) = 0) : ∀ v, F v = 0 := by
  have key : ∀ L : List Nat, (∀ i ∈ L, F (e i) = 0) → ∀ v, qCoversL n L v → F v = 0 := by
    intro L
    induction L with
    | nil => intro _ v hv; exact hlive v (eqOn_zero_of_qCoversL_nil hv)
    | cons i L ih =>
      intro hL v hv
      rw [peel_decomp i v, hadd, hsmul, hL i (List.mem_cons_self),
          ih (fun j hj => hL j (List.mem_cons_of_mem i hj)) _ (qCoversL_peel hv)]
      simp
  intro v
  exact key (List.range (2^n)) (fun i hi => hb i (List.mem_range.mp hi)) v (qCoversL_range n v)

/-- The doubly-polarized composition form on basis LEFT factors `e_i, e_i'` and arbitrary
    right factors `b, b'`:  ⟨e_i b, e_i' b'⟩ + ⟨e_i b', e_i' b⟩ − 2·[i=i']·⟨b,b'⟩. -/
def polar (n : Nat) (i i' : Nat) (b b' : Vec) : Int :=
  inner (2^n) (cdMul n (e i) b) (cdMul n (e i') b') + inner (2^n) (cdMul n (e i) b') (cdMul n (e i') b)
    - (if i = i' then 2 * inner (2^n) b b' else 0)

/-- The composition identity on BASIS quadruples — decidable at each level. -/
def polarBasis (n : Nat) : Prop :=
  ∀ i, i < 2^n → ∀ i', i' < 2^n → ∀ l, l < 2^n → ∀ l', l' < 2^n → polar n i i' (e l) (e l') = 0
instance (n : Nat) : Decidable (polarBasis n) := by unfold polarBasis; infer_instance

theorem polarBasis0 : polarBasis 0 := by decide
theorem polarBasis1 : polarBasis 1 := by decide
theorem polarBasis2 : polarBasis 2 := by decide
theorem polarBasis3 : polarBasis 3 := by decide
/-- …and it FAILS for the sedenions: the quadruple behind `sed_shortcut_understates`. -/
theorem not_polarBasis4 : ¬ polarBasis 4 := by
  intro h
  have := h 1 (by decide) 10 (by decide) 4 (by decide) 15 (by decide)
  revert this; decide

theorem polar_add_right (n i i' : Nat) (b c b' : Vec) :
    polar n i i' (vadd b c) b' = polar n i i' b b' + polar n i i' c b' := by
  unfold polar
  rw [cdMul_add_right, cdMul_add_right, inner_add_left, inner_add_right, inner_add_left]
  by_cases h : i = i' <;> simp only [h, if_true, if_false] <;> omega
theorem polar_smul_right (n i i' : Nat) (c : Int) (b b' : Vec) :
    polar n i i' (smul c b) b' = c * polar n i i' b b' := by
  unfold polar
  rw [cdMul_smul_right, cdMul_smul_right, inner_smul_left, inner_smul_right, inner_smul_left]
  by_cases h : i = i' <;> simp only [h, if_true, if_false]
  · rw [Int.mul_sub, Int.mul_add, Int.mul_left_comm c 2]
  · rw [Int.mul_sub, Int.mul_add, Int.mul_zero]
theorem polar_live_right (n i i' : Nat) {b : Vec} (b' : Vec) (hb : eqOn (2^n) b vzero) :
    polar n i i' b b' = 0 := by
  unfold polar
  have h1 : eqOn (2^n) (cdMul n (e i) b) vzero := by
    have := cdMul_congr_right n (e i) hb; rw [cdMul_zero_right] at this; exact this
  have h2 : eqOn (2^n) (cdMul n (e i') b) vzero := by
    have := cdMul_congr_right n (e i') hb; rw [cdMul_zero_right] at this; exact this
  rw [inner_zero_left_of _ h1, inner_zero_right_of _ h2, inner_zero_left_of _ hb]
  by_cases h : i = i' <;> simp [h]
theorem polar_comm (n i i' : Nat) (b b' : Vec) : polar n i i' b b' = polar n i i' b' b := by
  unfold polar; rw [inner_comm (2^n) b' b]; omega

/-- From the basis identity to all `b, b'` (two support inductions). -/
theorem polar_zero_of_polarBasis {n : Nat} (hp : polarBasis n) {i i' : Nat}
    (hi : i < 2^n) (hi' : i' < 2^n) : ∀ b b', polar n i i' b b' = 0 := by
  have step1 : ∀ l, l < 2^n → ∀ b', polar n i i' (e l) b' = 0 := by
    intro l hl
    apply lin_zero_of_basis n (fun b' => polar n i i' (e l) b')
    · intro u v; rw [polar_comm, polar_add_right, polar_comm n i i' u, polar_comm n i i' v]
    · intro c u; rw [polar_comm, polar_smul_right, polar_comm]
    · intro u hu; rw [polar_comm]; exact polar_live_right n i i' (e l) hu
    · intro l' hl'; exact hp i hi i' hi' l hl l' hl'
  intro b b'
  apply lin_zero_of_basis n (fun b => polar n i i' b b')
  · intro u v; exact polar_add_right n i i' u v b'
  · intro c u; exact polar_smul_right n i i' c u b'
  · intro u hu; exact polar_live_right n i i' b' hu
  · intro l hl; exact step1 l hl b'

/-- `⟨e_i b, e_i' b⟩ = [i=i']·‖b‖²` — the right-multiplication matrix is ‖b‖-orthogonal. -/
theorem basis_bil_zero {n : Nat} (hp : polarBasis n) {i i' : Nat} (hi : i < 2^n) (hi' : i' < 2^n) (b : Vec) :
    inner (2^n) (cdMul n (e i) b) (cdMul n (e i') b) - (if i = i' then normSq (2^n) b else 0) = 0 := by
  have h := polar_zero_of_polarBasis hp hi hi' b b
  unfold polar at h; unfold normSq
  by_cases hii : i = i' <;> simp only [hii, if_true, if_false] at h ⊢ <;> omega

theorem sumR_ite_eq (m x : Nat) (hx : x < m) (g : Nat → Int) :
    sumR m (fun k => if k = x then g k else 0) = g x := by
  induction m with
  | zero => exact absurd hx (Nat.not_lt_zero x)
  | succ m ih =>
    simp only [sumR]
    by_cases h : x = m
    · have hz : sumR m (fun k => if k = x then g k else 0) = 0 :=
        sumR_eq_zero_of m _ (fun i hi => if_neg (by omega))
      rw [hz]; simp [h]
    · have hx' : x < m := Nat.lt_of_le_of_ne (Nat.le_of_lt_succ hx) h
      rw [ih hx']; simp [Ne.symm h]

theorem inner_e_e (m i j : Nat) (hi : i < m) : inner m (e i) (e j) = if i = j then 1 else 0 := by
  unfold inner
  have : (fun k => e i k * e j k) = (fun k => if k = i then e j k else 0) := by
    funext k; simp only [e]; by_cases h : k = i <;> simp [h]
  rw [this, sumR_ite_eq m i hi]; simp only [e]

/-- The bilinear defect `⟨a b, a' b⟩ − ⟨a,a'⟩‖b‖²`, killed on every pair by two more
    support inductions. -/
def bil (n : Nat) (b a a' : Vec) : Int :=
  inner (2^n) (cdMul n a b) (cdMul n a' b) - inner (2^n) a a' * normSq (2^n) b

theorem bil_add_left (n : Nat) (b u v a' : Vec) : bil n b (vadd u v) a' = bil n b u a' + bil n b v a' := by
  unfold bil; rw [cdMul_add_left, inner_add_left, inner_add_left, Int.add_mul]; omega
theorem bil_smul_left (n : Nat) (b : Vec) (c : Int) (u a' : Vec) : bil n b (smul c u) a' = c * bil n b u a' := by
  unfold bil; rw [cdMul_smul_left, inner_smul_left, inner_smul_left, Int.mul_sub, Int.mul_assoc]
theorem bil_live_left (n : Nat) (b : Vec) {u : Vec} (a' : Vec) (hu : eqOn (2^n) u vzero) : bil n b u a' = 0 := by
  unfold bil
  rw [cdMul_congr_left n b hu, cdMul_zero_left, inner_zero_left, inner_zero_left_of _ hu]; simp
theorem bil_comm (n : Nat) (b a a' : Vec) : bil n b a a' = bil n b a' a := by
  unfold bil; rw [inner_comm (2^n) (cdMul n a b), inner_comm (2^n) a a']

theorem bil_zero_of_polarBasis {n : Nat} (hp : polarBasis n) (b : Vec) : ∀ a a', bil n b a a' = 0 := by
  have step1 : ∀ i, i < 2^n → ∀ a', bil n b (e i) a' = 0 := by
    intro i hi
    apply lin_zero_of_basis n (fun a' => bil n b (e i) a')
    · intro u v; rw [bil_comm, bil_add_left, bil_comm n b u, bil_comm n b v]
    · intro c u; rw [bil_comm, bil_smul_left, bil_comm]
    · intro u hu; rw [bil_comm]; exact bil_live_left n b (e i) hu
    · intro i' hi'
      have h := basis_bil_zero hp hi hi' b
      unfold bil; rw [inner_e_e (2^n) i i' hi]
      by_cases hii : i = i' <;> simp only [hii, if_true, if_false] at h ⊢ <;> omega
  intro a a'
  apply lin_zero_of_basis n (fun a => bil n b a a')
  · intro u v; exact bil_add_left n b u v a'
  · intro c u; exact bil_smul_left n b c u a'
  · intro u hu; exact bil_live_left n b a' hu
  · intro i hi; exact step1 i hi a'

/-- HURWITZ (kernel form): wherever the basis composition identity holds, the norm is
    multiplicative on ALL vectors. Levels 0–3 qualify (`polarBasis0..3`); level 4 does not. -/
theorem norm_mult_of_polarBasis {n : Nat} (hp : polarBasis n) (a b : Vec) :
    normSq (2^n) (cdMul n a b) = normSq (2^n) a * normSq (2^n) b := by
  have h := bil_zero_of_polarBasis hp b a a
  unfold bil at h; unfold normSq at h ⊢; omega

theorem octonion_norm_multiplicative (a b : Vec) : normSq 8 (cdMul 3 a b) = normSq 8 a * normSq 8 b :=
  norm_mult_of_polarBasis polarBasis3 a b
theorem quaternion_norm_multiplicative (a b : Vec) : normSq 4 (cdMul 2 a b) = normSq 4 a * normSq 4 b :=
  norm_mult_of_polarBasis polarBasis2 a b
/-- The sedenion counter-witness, restated against the general theorem's hypothesis. -/
theorem sedenion_norm_not_multiplicative :
    ¬ (∀ a b : Vec, normSq 16 (cdMul 4 a b) = normSq 16 a * normSq 16 b) := by
  intro h
  have := h sedD sedY
  revert this; decide

/-- Bilinear consequence (polarization): `⟨d·y, d'·y⟩ = ⟨d,d'⟩·‖y‖²` and `⟨x·d, x·d'⟩ = ‖x‖²·⟨d,d'⟩`. -/
theorem inner_mulR_eq {n : Nat} (hp : polarBasis n) (d d' y : Vec) :
    inner (2^n) (cdMul n d y) (cdMul n d' y) = inner (2^n) d d' * normSq (2^n) y := by
  have h := bil_zero_of_polarBasis hp y d d'; unfold bil at h; omega
theorem inner_mulL_eq {n : Nat} (hp : polarBasis n) (x d d' : Vec) :
    inner (2^n) (cdMul n x d) (cdMul n x d') = normSq (2^n) x * inner (2^n) d d' := by
  have h1 := norm_mult_of_polarBasis hp x (vadd d d')
  have h2 := norm_mult_of_polarBasis hp x d
  have h3 := norm_mult_of_polarBasis hp x d'
  rw [cdMul_add_right, normSq_add, normSq_add, h2, h3, Int.mul_add, Int.mul_add] at h1
  have := Int.mul_left_comm (normSq (2^n) x) 2 (inner (2^n) d d')
  omega

theorem trueVar_scaleR_eq {n : Nat} (hp : polarBasis n) (a : Aff) (y : Vec) :
    trueVar (2^n) (scaleR n a y) = normSq (2^n) y * trueVar (2^n) a := by
  unfold trueVar
  suffices h : ∀ b : Aff, innerA (2^n) (scaleR n a y) (scaleR n b y) = normSq (2^n) y * innerA (2^n) a b from h a
  intro b
  induction a with
  | nil => simp [scaleR, innerA]
  | cons p r ih =>
    rcases p with ⟨s, c⟩
    have hcons : scaleR n ((s, c) :: r) y = (s, cdMul n c y) :: scaleR n r y := rfl
    rw [hcons]; simp only [innerA]
    rw [ih, coeff_scaleR, inner_mulR_eq hp, Int.mul_add, Int.mul_comm (inner (2^n) c (coeff b s))]
theorem trueVar_scaleL_eq {n : Nat} (hp : polarBasis n) (x : Vec) (a : Aff) :
    trueVar (2^n) (scaleL n x a) = normSq (2^n) x * trueVar (2^n) a := by
  unfold trueVar
  suffices h : ∀ b : Aff, innerA (2^n) (scaleL n x a) (scaleL n x b) = normSq (2^n) x * innerA (2^n) a b from h a
  intro b
  induction a with
  | nil => simp [scaleL, innerA]
  | cons p r ih =>
    rcases p with ⟨s, c⟩
    have hcons : scaleL n x ((s, c) :: r) = (s, cdMul n x c) :: scaleL n x r := rfl
    rw [hcons]; simp only [innerA]
    rw [ih, coeff_scaleL, inner_mulL_eq hp, Int.mul_add]

/-- THE THIRD AXIS, AS A THEOREM. On every level where the composition identity holds
    (ℝ, ℂ, ℍ, 𝕆), the GUM variance shortcut coincides with the sensitivity propagator on
    exact operands — so it inherits Axis-2 exactness. Beyond Hurwitz (𝕊) the hypothesis
    fails (`not_polarBasis4`) and so does the conclusion (`sed_shortcut_understates`). -/
theorem shortcut_eq_sensitivity_of_polarBasis {n : Nat} (hp : polarBasis n)
    (x : Vec) (a : Aff) (y : Vec) (b : Aff) (ma mb : KMeta)
    (hx : ma.gumVar = trueVar (2^n) a) (hy : mb.gumVar = trueVar (2^n) b) :
    (gMulShortcut n x y ma mb).gumVar = (gMulMeta n x a y b ma mb).gumVar := by
  simp only [gMulShortcut, gMulMeta]
  rw [hx, hy, trueVar_scaleR_eq hp, trueVar_scaleL_eq hp]

theorem octonion_shortcut_exact (x : Vec) (a : Aff) (y : Vec) (b : Aff) (ma mb : KMeta)
    (hx : ma.gumVar = trueVar 8 a) (hy : mb.gumVar = trueVar 8 b) :
    (gMulShortcut 3 x y ma mb).gumVar = (gMulMeta 3 x a y b ma mb).gumVar :=
  shortcut_eq_sensitivity_of_polarBasis polarBasis3 x a y b ma mb hx hy

end Sounio.EpistemicEffectsNSA

-- ================================================================
-- Axiom footprint (reproduce: `lean --threads=1 EpistemicEffectsNSA.lean`)
-- ================================================================
#print axioms Sounio.EpistemicEffectsNSA.assoc_zero_of_qCoversL
#print axioms Sounio.EpistemicEffectsNSA.assoc_zero_of_cert
#print axioms Sounio.EpistemicEffectsNSA.innerA_zero_of_ns
#print axioms Sounio.EpistemicEffectsNSA.nsDisjoint_reassoc_invariant
#print axioms Sounio.EpistemicEffectsNSA.typed_agfree
#print axioms Sounio.EpistemicEffectsNSA.preservation
#print axioms Sounio.EpistemicEffectsNSA.exact_preservation
#print axioms Sounio.EpistemicEffectsNSA.soundness_star
#print axioms Sounio.EpistemicEffectsNSA.reassoc_sensitivity_gap
#print axioms Sounio.EpistemicEffectsNSA.reassoc_sound
#print axioms Sounio.EpistemicEffectsNSA.w1_typable
#print axioms Sounio.EpistemicEffectsNSA.w1_reassoc_changes_value
#print axioms Sounio.EpistemicEffectsNSA.w1'_reassoc_sound
#print axioms Sounio.EpistemicEffectsNSA.w2_untypable
#print axioms Sounio.EpistemicEffectsNSA.w2_understates
#print axioms Sounio.EpistemicEffectsNSA.sed_shortcut_understates
#print axioms Sounio.EpistemicEffectsNSA.oct_shortcut_exact
#print axioms Sounio.EpistemicEffectsNSA.shortcut_eq_sensitivity_level0
#print axioms Sounio.EpistemicEffectsNSA.lin_zero_of_basis
#print axioms Sounio.EpistemicEffectsNSA.norm_mult_of_polarBasis
#print axioms Sounio.EpistemicEffectsNSA.octonion_norm_multiplicative
#print axioms Sounio.EpistemicEffectsNSA.sedenion_norm_not_multiplicative
#print axioms Sounio.EpistemicEffectsNSA.shortcut_eq_sensitivity_of_polarBasis
#print axioms Sounio.EpistemicEffectsNSA.not_polarBasis4
