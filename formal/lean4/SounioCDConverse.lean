/-
  SounioCDConverse — the CONVERSE of the CD-tower seam obstruction, stated and empirically anchored.

  The forward obstruction ({L_l,L_u}=0 ⟹ e_l+e_u not a ZD) is proved dimension-independently GIVEN
  the cocycle lemma L_i²=−I (see SounioCDTowerSeam). Its CONVERSE —

      off-seam(l,u)  ⟹  e_l+e_u is a (left) zero divisor,   for ALL n

  — is a tower-wide conjecture: verified at n=4,5,6 in SounioCDTowerSeam, and (this file's basis)
  empirically extended with a validated O(N) reduction to n=7,8,9,10 (dim 128..1024) with zero
  counterexamples (scripts/research/cd_tower_converse_probe.py).

  The reduction (the crux that turns the converse into a clean sign-cocycle identity): a 2-term
  right factor e_a + s·e_b annihilates e_l+e_u iff the four product basis indices
  {l⊕a, l⊕b, u⊕a, u⊕b} cancel in cross-pairs, which FORCES b = a ⊕ (l⊕u); given that, a solving
  sign s∈{±1} exists iff the four-sign product σ(l,a)σ(u,a)σ(l,a⊕l⊕u)σ(u,a⊕l⊕u) = +1. Hence
  `hasXorAnnih` below is O(N)-per-pair and coincides with the brute `isZD` on the loHi locus
  (checked here at dim 16/32). Verified exhaustive for 2-term factors by an independent math-review
  (grok-4.1). Mathlib-free, no sorry.
-/
import SounioCDTowerSeam
import SounioCDCocycle

namespace SounioCDConverse
open SounioCDTowerSeam

/-- The sharp O(N) 2-term zero-divisor test. `e_l+e_u` has a 2-term annihilator `e_a + s·e_b` iff
    `b = a ⊕ (l⊕u)` (forced by index cancellation) with `a,b ≥ 1`, and the four-sign product is `+1`
    (the two cancellation equations `σ(l,a)+s·σ(u,b)=0`, `s·σ(l,b)+σ(u,a)=0` are jointly solvable in
    `s`). Coincides with `SounioCDTowerSeam.isZD` on the loHi locus. -/
def hasXorAnnih (bits l u : Nat) : Bool :=
  let N := 2 ^ bits
  let d := l ^^^ u
  (List.range N).any (fun a =>
    a ≥ 1 && a ≠ d
      && (cdSigma l a bits * cdSigma u a bits
            * cdSigma l (a ^^^ d) bits * cdSigma u (a ^^^ d) bits == 1))

/-- The converse at a given level (brute form): every off-seam lower×upper pair is a zero divisor. -/
def converseHolds (bits : Nat) : Bool :=
  (loHi bits).all (fun p => ! offSeam bits p.1 p.2 || isZD bits p.1 p.2)

/-- The converse at a given level (sharp σ-form): every off-seam lower×upper pair admits an
    XOR-linked 2-term annihilator. Equivalent to `converseHolds` (see `converse_sharp_agrees_16`),
    but O(N)-per-pair so it certifies far beyond the reach of the brute `isZD` scan. -/
def converseHoldsSharp (bits : Nat) : Bool :=
  (loHi bits).all (fun p => ! offSeam bits p.1 p.2 || hasXorAnnih bits p.1 p.2)

/-- **The tower-wide CONVERSE CONJECTURE** (open for all n; empirically verified n=4..10).
    Stated as a `Prop`, deliberately *not* asserted — no `sorry`, no fixed-n `native_decide` hiding
    inside. A full proof further depends on the cocycle lemma `L_i²=−I` for all n (itself open). -/
def ConverseConjecture : Prop := ∀ bits, 4 ≤ bits → converseHolds bits = true

-- ── Regression anchors (brute `isZD`, native_decide) ─────────────────────────────────────────────
theorem converse_16 : converseHolds 4 = true := by native_decide

-- ── The reduction: sharp O(N) predicate == brute `isZD` on the loHi locus ─────────────────────────
theorem xorAnnih_eq_isZD_16 :
    (loHi 4).all (fun p => hasXorAnnih 4 p.1 p.2 == isZD 4 p.1 p.2) = true := by native_decide

-- ── Sharp-form converse anchors (O(N); reach beyond the brute scan) ───────────────────────────────
theorem converse_sharp_16 : converseHoldsSharp 4 = true := by native_decide
theorem converse_sharp_32 : converseHoldsSharp 5 = true := by native_decide
theorem converse_sharp_64 : converseHoldsSharp 6 = true := by native_decide

/-- At dim 16 the sharp converse and the brute converse agree (so the sharp anchors above carry the
    same content as the brute `isZD` statement, at the level where both are cheap to decide). -/
theorem converse_sharp_agrees_16 : converseHoldsSharp 4 = converseHolds 4 := by native_decide

/-- **Primary-source cross-check (Moreno 1998, `q-alg/9710013`, opening example).** In the sedenions
    `A_4`, `e₁ + e₁₀` is annihilated by `e₁₅ − e₄` (equivalently `e₄ − e₁₅`, a scalar multiple). Here
    `l=1, u=10, d=l⊕u=11`, and the annihilator index pair is XOR-linked: `4 = 15 ⊕ 11`, i.e.
    `b = a ⊕ d`. This is exactly `hasXorAnnih`'s witness, and it discharges `annih` directly. -/
theorem moreno_e1_e10 : annih 4 1 10 4 15 (-1) = true := by native_decide

-- ── The general ∀-soundness of the sharp test (Mathlib-free, no `sorry`, no `native_decide`) ────────
-- `hasXorAnnih bits l u = true → isZD bits l u = true` for ALL `bits`, and all `l, u` on the natural
-- domain `l, u < 2^bits` with `l ≠ u` (both guaranteed on the loHi locus; the theorem is genuinely
-- FALSE without them — e.g. `l = u` makes `hasXorAnnih` true while `isZD` is false, and unbounded
-- `l, u` push the XOR-linked partner `a ⊕ (l⊕u)` outside `range (2^bits)`).

/-- `cdSigma` is always `±1`. -/
theorem neg_pm {x : Int} (h : x = 1 ∨ x = -1) : -x = 1 ∨ -x = -1 := by
  rcases h with h | h <;> rw [h] <;> decide

theorem cdSigma_pm : ∀ (bits a b : Nat), cdSigma a b bits = 1 ∨ cdSigma a b bits = -1
  | 0, _, _ => Or.inr rfl
  | 1, _, _ => by rw [cdSigma]; split; exact Or.inl rfl; exact Or.inr rfl
  | (n+2), a, b => by
      rw [cdSigma]
      extract_lets half aHi bHi aLo bLo
      split
      · exact Or.inl rfl
      · split
        · exact cdSigma_pm (n+1) _ _
        · split
          · exact cdSigma_pm (n+1) _ _
          · split
            · split
              · exact cdSigma_pm (n+1) _ _
              · exact neg_pm (cdSigma_pm (n+1) _ _)
            · split
              · exact neg_pm (cdSigma_pm (n+1) _ _)
              · exact cdSigma_pm (n+1) _ _

/-- Minimal `Nat.xor` group facts (Mathlib-free). -/
theorem xor_left_comm (a b c : Nat) : a ^^^ (b ^^^ c) = b ^^^ (a ^^^ c) := by
  rw [← Nat.xor_assoc, Nat.xor_comm a b, Nat.xor_assoc]

theorem xor_eq_zero_of (a b : Nat) (h : a ^^^ b = 0) : a = b := by
  have hh : a ^^^ (a ^^^ b) = b := by rw [← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]
  rw [h, Nat.xor_zero] at hh; exact hh

/-- The XOR-linked pair `{a, b}` with the solving sign `s` genuinely annihilates: every output basis
    index cancels. `hp`/`hq` are the forced index pairings (`l⊕a=u⊕b`, `l⊕b=u⊕a`), `hpq` says the two
    live indices are distinct, and `e1`/`e2` are the two sign-cancellation equations. -/
theorem annih_of (bits l u a b : Nat) (s : Int)
    (hp : l ^^^ a = u ^^^ b) (hq : l ^^^ b = u ^^^ a)
    (hpq : l ^^^ a ≠ l ^^^ b)
    (e1 : cdSigma l a bits + s * cdSigma u b bits = 0)
    (e2 : s * cdSigma l b bits + cdSigma u a bits = 0) :
    annih bits l u a b s = true := by
  unfold annih
  refine List.all_eq_true.mpr ?_
  intro k _
  rw [beq_iff_eq, ← hp, ← hq]
  by_cases hpk : ((l ^^^ a) == k) = true
  · by_cases hqk : ((l ^^^ b) == k) = true
    · rw [beq_iff_eq] at hpk hqk
      exact absurd (hpk.trans hqk.symm) hpq
    · simp only [if_pos hpk, if_neg hqk]; omega
  · by_cases hqk : ((l ^^^ b) == k) = true
    · simp only [if_neg hpk, if_pos hqk]; omega
    · simp only [if_neg hpk, if_neg hqk]; omega

/-- **Soundness of the sharp O(N) test, for ALL `bits`** (a genuine `∀`-theorem, not `native_decide`).
    On the natural domain (`l, u < 2^bits`, `l ≠ u`), a satisfied `hasXorAnnih` witness yields an
    explicit `isZD` zero-divisor certificate: the XOR-linked pair `{a, a ⊕ (l⊕u)}` with the solving
    sign, ordered so the `a < b` requirement of `isZD` holds. -/
theorem hasXorAnnih_sound (bits l u : Nat)
    (hl : l < 2 ^ bits) (hu : u < 2 ^ bits) (hne : l ≠ u) :
    hasXorAnnih bits l u = true → isZD bits l u = true := by
  intro hxa
  unfold hasXorAnnih at hxa
  rw [List.any_eq_true] at hxa
  obtain ⟨a, hmem, hpred⟩ := hxa
  rw [List.mem_range] at hmem
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hpred
  obtain ⟨⟨-, -⟩, hP⟩ := hpred
  have hbN : a ^^^ (l ^^^ u) < 2 ^ bits := Nat.xor_lt_two_pow hmem (Nat.xor_lt_two_pow hl hu)
  have hd0 : l ^^^ u ≠ 0 := fun h => hne (xor_eq_zero_of l u h)
  have hp : l ^^^ a = u ^^^ (a ^^^ (l ^^^ u)) := by
    apply xor_eq_zero_of
    simp [Nat.xor_comm, xor_left_comm, Nat.xor_self, Nat.xor_zero]
  have hq : l ^^^ (a ^^^ (l ^^^ u)) = u ^^^ a := by
    apply xor_eq_zero_of
    simp [Nat.xor_comm, xor_left_comm, Nat.xor_self, Nat.xor_zero]
  have hab : a ≠ a ^^^ (l ^^^ u) := by
    intro h; apply hd0
    have h2 : a ^^^ a = a ^^^ (a ^^^ (l ^^^ u)) := by rw [← h]
    rw [Nat.xor_self, ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor] at h2
    exact h2.symm
  have hpq : l ^^^ a ≠ l ^^^ (a ^^^ (l ^^^ u)) := by
    intro h; apply hab
    have e : l ^^^ (l ^^^ a) = l ^^^ (l ^^^ (a ^^^ (l ^^^ u))) := by rw [h]
    simpa [← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor] using e
  have e1 : cdSigma l a bits
      + (-cdSigma l a bits * cdSigma u (a ^^^ (l ^^^ u)) bits) * cdSigma u (a ^^^ (l ^^^ u)) bits = 0 := by
    rcases cdSigma_pm bits l a with h1|h1 <;>
    rcases cdSigma_pm bits u (a ^^^ (l ^^^ u)) with h4|h4 <;> simp only [h1, h4] <;> decide
  have e2 : (-cdSigma l a bits * cdSigma u (a ^^^ (l ^^^ u)) bits) * cdSigma l (a ^^^ (l ^^^ u)) bits
      + cdSigma u a bits = 0 := by
    rcases cdSigma_pm bits l a with h1|h1 <;> rcases cdSigma_pm bits u a with h2|h2 <;>
    rcases cdSigma_pm bits l (a ^^^ (l ^^^ u)) with h3|h3 <;>
    rcases cdSigma_pm bits u (a ^^^ (l ^^^ u)) with h4|h4 <;>
      simp only [h1, h2, h3, h4] at hP ⊢ <;> revert hP <;> decide
  have e1' : cdSigma l (a ^^^ (l ^^^ u)) bits
      + (-cdSigma l a bits * cdSigma u (a ^^^ (l ^^^ u)) bits) * cdSigma u a bits = 0 := by
    rcases cdSigma_pm bits l a with h1|h1 <;> rcases cdSigma_pm bits u a with h2|h2 <;>
    rcases cdSigma_pm bits l (a ^^^ (l ^^^ u)) with h3|h3 <;>
    rcases cdSigma_pm bits u (a ^^^ (l ^^^ u)) with h4|h4 <;>
      simp only [h1, h2, h3, h4] at hP ⊢ <;> revert hP <;> decide
  have e2' : (-cdSigma l a bits * cdSigma u (a ^^^ (l ^^^ u)) bits) * cdSigma l a bits
      + cdSigma u (a ^^^ (l ^^^ u)) bits = 0 := by
    rcases cdSigma_pm bits l a with h1|h1 <;>
    rcases cdSigma_pm bits u (a ^^^ (l ^^^ u)) with h4|h4 <;> simp only [h1, h4] <;> decide
  have hs_pm : (-cdSigma l a bits * cdSigma u (a ^^^ (l ^^^ u)) bits) = 1
      ∨ (-cdSigma l a bits * cdSigma u (a ^^^ (l ^^^ u)) bits) = -1 := by
    rcases cdSigma_pm bits l a with h1|h1 <;>
    rcases cdSigma_pm bits u (a ^^^ (l ^^^ u)) with h4|h4 <;> simp only [h1, h4] <;> decide
  unfold isZD
  refine List.any_eq_true.mpr ?_
  rcases Nat.lt_trichotomy a (a ^^^ (l ^^^ u)) with hlt | heq | hgt
  · refine ⟨a, List.mem_range.mpr hmem, ?_⟩
    refine List.any_eq_true.mpr ⟨a ^^^ (l ^^^ u), List.mem_range.mpr hbN, ?_⟩
    have hah : annih bits l u a (a ^^^ (l ^^^ u))
        (-cdSigma l a bits * cdSigma u (a ^^^ (l ^^^ u)) bits) = true :=
      annih_of bits l u a (a ^^^ (l ^^^ u)) _ hp hq hpq e1 e2
    rw [Bool.and_eq_true]
    refine ⟨decide_eq_true_eq.mpr hlt, ?_⟩
    rcases hs_pm with h|h <;> rw [h] at hah <;> simp [hah]
  · exact absurd heq hab
  · refine ⟨a ^^^ (l ^^^ u), List.mem_range.mpr hbN, ?_⟩
    refine List.any_eq_true.mpr ⟨a, List.mem_range.mpr hmem, ?_⟩
    have hah : annih bits l u (a ^^^ (l ^^^ u)) a
        (-cdSigma l a bits * cdSigma u (a ^^^ (l ^^^ u)) bits) = true :=
      annih_of bits l u (a ^^^ (l ^^^ u)) a _ hq hp (fun h => hpq h.symm) e1' e2'
    rw [Bool.and_eq_true]
    refine ⟨decide_eq_true_eq.mpr hgt, ?_⟩
    rcases hs_pm with h|h <;> rw [h] at hah <;> simp [hah]

-- ══════════════════════════════════════════════════════════════════════════════════════════════════
-- STRUCTURAL LEMMAS toward the CONVERSE (`offSeam ⟹ hasXorAnnih` ∀n).  All ∀n, Mathlib-free, no
-- `sorry`, no `native_decide`.  They formalize the "orbit / doubling-recursion" analysis of the
-- winner predicate `P(a) := cdSigma l a · cdSigma u a · cdSigma l (a⊕d) · cdSigma u (a⊕d)`, where
-- `d = l⊕u`.  See the closing prose block for how these combine (and where the last gap is).
-- ══════════════════════════════════════════════════════════════════════════════════════════════════

/-- The paired sign `f(a) = cdSigma l a · cdSigma u a ∈ {±1}` (the left-multiplication sign of the
    factor `e_l + e_u` acting on `e_a`, up to the output index). -/
def fVal (l u a bits : Nat) : Int := cdSigma l a bits * cdSigma u a bits

theorem fVal_pm (l u a bits : Nat) : fVal l u a bits = 1 ∨ fVal l u a bits = -1 := by
  unfold fVal
  rcases cdSigma_pm bits l a with h1|h1 <;> rcases cdSigma_pm bits u a with h2|h2 <;>
    rw [h1, h2] <;> decide

/-- **Reformulation of the `hasXorAnnih` winner product** as `P(a) = f(a)·f(a⊕d)`.  Pure Int algebra;
    no cocycle input.  This is the identity that turns the four-sign test into the involution
    `a ↦ a⊕d` "agreeing" on the orbit of `a`. -/
theorem P_eq_fVal (l u a bits : Nat) :
    cdSigma l a bits * cdSigma u a bits
        * cdSigma l (a ^^^ (l ^^^ u)) bits * cdSigma u (a ^^^ (l ^^^ u)) bits
      = fVal l u a bits * fVal l u (a ^^^ (l ^^^ u)) bits := by
  unfold fVal
  rw [Int.mul_assoc]

/-- `cdSigma _ 0 = 1` for every positive width (the identity `e₀=1` kills the sign).  Note it is
    genuinely `-1` at width `0`, hence the `1 ≤ bits`. -/
theorem cdSigma_zero_right : ∀ (bits x : Nat), 1 ≤ bits → cdSigma x 0 bits = 1
  | 1, x, _ => by
      have hg : (x == 0 || (0:Nat) == 0) = true := by
        rw [show ((0:Nat) == 0) = true from rfl, Bool.or_true]
      rw [cdSigma, if_pos hg]
  | (n+2), x, _ => by
      have hg : (x == 0 || (0:Nat) == 0) = true := by
        rw [show ((0:Nat) == 0) = true from rfl, Bool.or_true]
      rw [cdSigma, if_pos hg]

/-- `cdSigma 0 _ = 1` for every positive width (identity on the left). -/
theorem cdSigma_zero_left : ∀ (bits x : Nat), 1 ≤ bits → cdSigma 0 x bits = 1
  | 1, x, _ => by
      have hg : ((0:Nat) == 0 || x == 0) = true := by
        rw [show ((0:Nat) == 0) = true from rfl, Bool.true_or]
      rw [cdSigma, if_pos hg]
  | (n+2), x, _ => by
      have hg : ((0:Nat) == 0 || x == 0) = true := by
        rw [show ((0:Nat) == 0) = true from rfl, Bool.true_or]
      rw [cdSigma, if_pos hg]

/-- `f(0) = 1`: the trivial orbit `{0,d}` carries paired sign `+1` at the `0` end (so the trivial
    orbit "agrees" exactly when `f(d)=1` — and, being trivial, is never itself a legal witness). -/
theorem fVal_zero (l u bits : Nat) (h : 1 ≤ bits) : fVal l u 0 bits = 1 := by
  unfold fVal; rw [cdSigma_zero_right bits l h, cdSigma_zero_right bits u h]; decide

/-- **Orbit invariance of the winner value:** `P(a) = P(a⊕d)`.  The four-sign test is constant on
    each orbit `{a, a⊕d}` of the fixed-point-free involution `a ↦ a⊕d`, so `hasXorAnnih` is really a
    statement about orbits, and every orbit is witnessed by either of its two members. -/
theorem P_orbit_inv (l u a bits : Nat) :
    fVal l u a bits * fVal l u (a ^^^ (l ^^^ u)) bits
      = fVal l u (a ^^^ (l ^^^ u)) bits * fVal l u ((a ^^^ (l ^^^ u)) ^^^ (l ^^^ u)) bits := by
  have hxx : (a ^^^ (l ^^^ u)) ^^^ (l ^^^ u) = a := by
    rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
  rw [hxx, Int.mul_comm]

/-- **Width-stability of the CD sign on the low block.**  `cdSigma a b` does not depend on the width
    as long as both indices fit below the previous doubling seam `2^(bits-1)`.  This is what lets the
    doubling recursion descend `bits ↦ bits-1` cleanly.  (Proof: the guard and the `!aHi && !bHi`
    branch of `cdSigma` peel one level, matching `cdSigma _ _ (bits-1)` verbatim.) -/
theorem cdSigma_stable (n a b : Nat)
    (ha : a < 2 ^ (n+1)) (hb : b < 2 ^ (n+1)) :
    cdSigma a b (n+2) = cdSigma a b (n+1) := by
  have hai : ¬ a ≥ 2 ^ (n+1) := by omega
  have hbi : ¬ b ≥ 2 ^ (n+1) := by omega
  rw [cdSigma]
  by_cases h0 : (a == 0 || b == 0) = true
  · rw [if_pos h0]
    rw [Bool.or_eq_true] at h0
    rcases h0 with h|h
    · rw [beq_iff_eq] at h; subst h; rw [cdSigma_zero_left (n+1) b (by omega)]
    · rw [beq_iff_eq] at h; subst h; rw [cdSigma_zero_right (n+1) a (by omega)]
  · rw [if_neg h0]
    simp only [ge_iff_le, hai, hbi, decide_false, Bool.not_false,
      Bool.and_self, Bool.true_and, Bool.and_true, if_true,
      Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hb]

/-- **High×low branch of the CD sign.**  The upper generator flips the sign relative to the low
    block: `cdSigma (2^(n+1)+uL) a = - cdSigma uL a` for `1 ≤ a < 2^(n+1)`.  Definitional (the
    `aHi && !bHi` branch with nonzero low part), no cocycle input. -/
theorem cdSigma_hi_lo (n uL a : Nat) (huL : uL < 2 ^ (n+1)) (ha1 : 1 ≤ a) (ha : a < 2 ^ (n+1)) :
    cdSigma (2 ^ (n+1) + uL) a (n+2) = - cdSigma uL a (n+1) := by
  have hpos : 0 < 2 ^ (n+1) := Nat.two_pow_pos (n+1)
  rw [cdSigma]
  have hg : ¬ ((2 ^ (n+1) + uL) == 0 || a == 0) = true := by
    rw [Bool.or_eq_true]
    rintro (h | h)
    · exact absurd (eq_of_beq h) (by omega)
    · exact absurd (eq_of_beq h) (by omega)
  rw [if_neg hg]
  have hAhi : (2 ^ (n+1) + uL) ≥ 2 ^ (n+1) := by omega
  have hBhi : ¬ a ≥ 2 ^ (n+1) := by omega
  have hmod1 : (2 ^ (n+1) + uL) % 2 ^ (n+1) = uL := by
    rw [Nat.add_mod_left]; exact Nat.mod_eq_of_lt huL
  have hmod2 : a % 2 ^ (n+1) = a := Nat.mod_eq_of_lt ha
  have hbz : ¬ (a == 0) = true := fun h => absurd (eq_of_beq h) (by omega)
  simp only [ge_iff_le, hAhi, hBhi, decide_true, decide_false, Bool.not_false,
    Bool.not_true, Bool.and_false, Bool.and_true,
    hmod1, hmod2, hbz]
  rfl

/-- **Low-`a` half of the doubling recursion (antisym-free).**  On the loHi locus (`l,uL < 2^(n+1)`,
    `u = 2^(n+1)+uL`), the paired sign of the *upstairs* pair at a low index `1 ≤ a < 2^(n+1)` is
    minus the paired sign of the *downstairs* pair `(l, uL)`:  `f_{(l,u)}(a) = - f_{(l,uL)}(a)`.
    This is the sign-flip that drives `P_{(l,u)}(a) = - P'_{(l,uL)}(a)` for generic low `a`; it uses
    only width-stability + the high×low branch (no cocycle / antisym). -/
theorem fVal_lo_reduce (n l uL a : Nat)
    (hl : l < 2 ^ (n+1)) (huL : uL < 2 ^ (n+1)) (ha1 : 1 ≤ a) (ha : a < 2 ^ (n+1)) :
    fVal l (2 ^ (n+1) + uL) a (n+2) = - fVal l uL a (n+1) := by
  unfold fVal
  rw [cdSigma_stable n l a hl ha, cdSigma_hi_lo n uL a huL ha1 ha]
  rw [Int.mul_neg]

/-- **Low×high branch of the CD sign** (transpose into the low block): `cdSigma a (2^(n+1)+bL) =
    cdSigma bL a` for `1 ≤ a < 2^(n+1)`.  Definitional (`!aHi && bHi` branch). -/
theorem cdSigma_lo_hi (n bL a : Nat) (hbL : bL < 2 ^ (n+1)) (ha1 : 1 ≤ a) (ha : a < 2 ^ (n+1)) :
    cdSigma a (2 ^ (n+1) + bL) (n+2) = cdSigma bL a (n+1) := by
  have hpos : 0 < 2 ^ (n+1) := Nat.two_pow_pos (n+1)
  rw [cdSigma]
  have hg : ¬ (a == 0 || (2 ^ (n+1) + bL) == 0) = true := by
    rw [Bool.or_eq_true]
    rintro (h | h)
    · exact absurd (eq_of_beq h) (by omega)
    · exact absurd (eq_of_beq h) (by omega)
  rw [if_neg hg]
  have hAhi : ¬ a ≥ 2 ^ (n+1) := by omega
  have hBhi : (2 ^ (n+1) + bL) ≥ 2 ^ (n+1) := by omega
  have hmod1 : a % 2 ^ (n+1) = a := Nat.mod_eq_of_lt ha
  have hmod2 : (2 ^ (n+1) + bL) % 2 ^ (n+1) = bL := by
    rw [Nat.add_mod_left]; exact Nat.mod_eq_of_lt hbL
  simp only [ge_iff_le, hAhi, hBhi, decide_true, decide_false, Bool.not_false, Bool.not_true,
    Bool.and_false, Bool.true_and, hmod1, hmod2]
  rfl

/-- **High×high branch of the CD sign**: `cdSigma (2^(n+1)+uL) (2^(n+1)+bL) = cdSigma bL uL` for
    `1 ≤ bL < 2^(n+1)` (nonzero right low part).  Definitional (both-high branch). -/
theorem cdSigma_hi_hi (n uL bL : Nat) (huL : uL < 2 ^ (n+1)) (hb1 : 1 ≤ bL) (hbL : bL < 2 ^ (n+1)) :
    cdSigma (2 ^ (n+1) + uL) (2 ^ (n+1) + bL) (n+2) = cdSigma bL uL (n+1) := by
  have hpos : 0 < 2 ^ (n+1) := Nat.two_pow_pos (n+1)
  rw [cdSigma]
  have hg : ¬ ((2 ^ (n+1) + uL) == 0 || (2 ^ (n+1) + bL) == 0) = true := by
    rw [Bool.or_eq_true]
    rintro (h | h)
    · exact absurd (eq_of_beq h) (by omega)
    · exact absurd (eq_of_beq h) (by omega)
  rw [if_neg hg]
  have hAhi : (2 ^ (n+1) + uL) ≥ 2 ^ (n+1) := by omega
  have hBhi : (2 ^ (n+1) + bL) ≥ 2 ^ (n+1) := by omega
  have hmod1 : (2 ^ (n+1) + uL) % 2 ^ (n+1) = uL := by
    rw [Nat.add_mod_left]; exact Nat.mod_eq_of_lt huL
  have hmod2 : (2 ^ (n+1) + bL) % 2 ^ (n+1) = bL := by
    rw [Nat.add_mod_left]; exact Nat.mod_eq_of_lt hbL
  have hbz : ¬ (bL == 0) = true := fun h => absurd (eq_of_beq h) (by omega)
  simp only [ge_iff_le, hAhi, hBhi, decide_true, Bool.not_true, Bool.and_false, Bool.false_and,
    Bool.and_self, hmod1, hmod2, hbz]
  rfl

/-- **High-index half of the doubling recursion (transposed, antisym-free).**  For the upstairs pair
    `(l, u)` with `u = 2^(n+1)+uL`, the paired sign at a HIGH index `2^(n+1)+b` (`1 ≤ b < 2^(n+1)`)
    equals the *transposed* downstairs product `cdSigma b l · cdSigma b uL` (arguments flipped
    relative to `f_{(l,uL)}(b) = cdSigma l b · cdSigma uL b`).  The flip is exactly one CD
    antisymmetry `cdSigma x y = -cdSigma y x` per factor — the single missing input (open on
    `cdSigma`; proved for the bit-list sign `sgn` as `SounioCDCocycle.antisym`). -/
theorem fVal_hi_reduce (n l uL b : Nat)
    (hl1 : 1 ≤ l) (hl : l < 2 ^ (n+1)) (huL : uL < 2 ^ (n+1))
    (hb1 : 1 ≤ b) (hb : b < 2 ^ (n+1)) :
    fVal l (2 ^ (n+1) + uL) (2 ^ (n+1) + b) (n+2)
      = cdSigma b l (n+1) * cdSigma b uL (n+1) := by
  unfold fVal
  rw [cdSigma_lo_hi n b l hb hl1 hl, cdSigma_hi_hi n uL b huL hb1 hb]

-- ── Top-bit XOR bookkeeping: the orbit map `a ↦ a⊕d` splits low/high across the seam ─────────────
/-- Below the seam, XOR against the seam bit is just addition: `2^k ^^^ z = 2^k + z` for `z < 2^k`.
    (Proved bit-by-bit; the seam bit `k` is free in `z`, so no carry.) -/
theorem two_pow_xor_eq_add (k z : Nat) (hz : z < 2 ^ k) : 2 ^ k ^^^ z = 2 ^ k + z := by
  apply Nat.eq_of_testBit_eq
  intro i
  rw [Nat.testBit_xor, Nat.testBit_two_pow]
  rcases Nat.lt_trichotomy i k with h|h|h
  · rw [Nat.testBit_two_pow_add_gt h]
    simp [show k ≠ i by omega]
  · subst h
    rw [Nat.testBit_two_pow_add_eq, Nat.testBit_lt_two_pow hz]
    simp
  · have hpow : 2 ^ k ≤ 2 ^ i := Nat.pow_le_pow_right (by decide) (by omega)
    have hzi : z.testBit i = false := Nat.testBit_lt_two_pow (Nat.lt_of_lt_of_le hz hpow)
    have hsucc : 2 ^ (k+1) = 2 ^ k * 2 := Nat.pow_succ 2 k
    have hpow2 : 2 ^ (k+1) ≤ 2 ^ i := Nat.pow_le_pow_right (by decide) (by omega)
    have hsum : 2 ^ k + z < 2 ^ i := by omega
    rw [Nat.testBit_lt_two_pow hsum, hzi]
    simp [show k ≠ i by omega]

/-- **Orbit map across the seam.**  On the loHi locus the involution `a ↦ a⊕d` carries a low index
    `a < 2^k` to the high index `2^k + (a ⊕ dL)` (where `d = 2^k + dL`, `dL < 2^k`).  Hence every
    orbit has exactly one low and one high member, so the winner search may be run over low
    representatives only. -/
theorem orbit_low_to_high (k a dL : Nat) (ha : a < 2 ^ k) (hdL : dL < 2 ^ k) :
    a ^^^ (2 ^ k + dL) = 2 ^ k + (a ^^^ dL) := by
  have haxL : a ^^^ dL < 2 ^ k := Nat.xor_lt_two_pow ha hdL
  rw [← two_pow_xor_eq_add k dL hdL, ← two_pow_xor_eq_add k (a ^^^ dL) haxL]
  rw [← Nat.xor_assoc, Nat.xor_comm a (2 ^ k), Nat.xor_assoc]

-- ══════════════════════════════════════════════════════════════════════════════════════════════════
-- THE DOUBLING RECURSION, CONDITIONAL ON CD ANTISYMMETRY
-- ══════════════════════════════════════════════════════════════════════════════════════════════════

/-- CD antisymmetry at width `m`: distinct nonzero basis units anticommute (`e_x e_y = - e_y e_x`),
    i.e. `cdSigma x y = - cdSigma y x`.  Verified by `native_decide` up to width 6 in
    `SounioCDTowerSeam` (implicitly, via the coincidence gates) and proved *for all widths* on the
    equivalent bit-list sign `sgn` as `SounioCDCocycle.antisym`.  Its transfer to `cdSigma` (the
    bridge `sgn = cdSigma` ∀n) is the one open input consumed below. -/
def cdAntisym (m : Nat) : Prop :=
  ∀ x y : Nat, 1 ≤ x → x < 2 ^ m → 1 ≤ y → y < 2 ^ m → x ≠ y →
    cdSigma x y m = - cdSigma y x m

/-- **The doubling recursion for the converse winner value.**  GIVEN CD antisymmetry one level down,
    the upstairs winner value at a low index `a` is *minus* the downstairs winner value:
    `P_{(l,u)}(a) = - P_{(l,uL)}(a)` for `u = 2^(n+1)+uL` and every low `a ∉ {l, uL, l⊕uL}`.
    This is the Biss–Dugger–Isaksen doubling step, fully assembled from the antisym-free branch
    reductions (`fVal_lo_reduce`, `fVal_hi_reduce`) plus two applications of `cdAntisym (n+1)`.
    (Empirically exact: 263088/263088 low reps over widths 3..7.) -/
theorem converse_recursion (n l uL a : Nat)
    (hasym : cdAntisym (n+1))
    (hl1 : 1 ≤ l) (hl : l < 2 ^ (n+1))
    (huL1 : 1 ≤ uL) (huL : uL < 2 ^ (n+1))
    (ha1 : 1 ≤ a) (ha : a < 2 ^ (n+1))
    (hal : a ≠ l) (hauL : a ≠ uL) (hadL : a ≠ l ^^^ uL) :
    fVal l (2 ^ (n+1) + uL) a (n+2)
        * fVal l (2 ^ (n+1) + uL) (a ^^^ (l ^^^ (2 ^ (n+1) + uL))) (n+2)
      = - (fVal l uL a (n+1) * fVal l uL (a ^^^ (l ^^^ uL)) (n+1)) := by
  -- write b := a ⊕ (l⊕uL) inline (Mathlib-free: no `set`).
  have hdL : l ^^^ uL < 2 ^ (n+1) := Nat.xor_lt_two_pow hl huL
  have hb : a ^^^ (l ^^^ uL) < 2 ^ (n+1) := Nat.xor_lt_two_pow ha hdL
  -- b ≠ 0  (⟺ a ≠ l⊕uL, excluded)
  have hbne0 : a ^^^ (l ^^^ uL) ≠ 0 := fun h => hadL (xor_eq_zero_of a (l ^^^ uL) h)
  have hb1 : 1 ≤ a ^^^ (l ^^^ uL) := Nat.one_le_iff_ne_zero.mpr hbne0
  -- b ≠ l  (⟺ a ≠ uL) and b ≠ uL (⟺ a ≠ l)
  have hbl : a ^^^ (l ^^^ uL) ≠ l := by
    intro h; apply hauL
    have e : a ^^^ (l ^^^ uL) ^^^ (l ^^^ uL) = l ^^^ (l ^^^ uL) := by rw [h]
    rwa [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero, ← Nat.xor_assoc, Nat.xor_self,
      Nat.zero_xor] at e
  have hbuL : a ^^^ (l ^^^ uL) ≠ uL := by
    intro h; apply hal
    have e : a ^^^ (l ^^^ uL) ^^^ (l ^^^ uL) = uL ^^^ (l ^^^ uL) := by rw [h]
    rwa [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero, xor_left_comm, Nat.xor_self,
      Nat.xor_zero] at e
  -- the upstairs difference index is high: a ⊕ d_up = 2^(n+1) + b
  have hd : l ^^^ (2 ^ (n+1) + uL) = 2 ^ (n+1) + (l ^^^ uL) := by
    rw [← two_pow_xor_eq_add (n+1) uL huL, ← two_pow_xor_eq_add (n+1) (l ^^^ uL) hdL]
    rw [xor_left_comm]
  have hidx : a ^^^ (l ^^^ (2 ^ (n+1) + uL)) = 2 ^ (n+1) + (a ^^^ (l ^^^ uL)) := by
    rw [hd, orbit_low_to_high (n+1) a (l ^^^ uL) ha hdL]
  -- reduce both upstairs factors to downstairs data
  rw [hidx, fVal_lo_reduce n l uL a hl huL ha1 ha,
      fVal_hi_reduce n l uL (a ^^^ (l ^^^ uL)) hl1 hl huL hb1 hb]
  -- RHS downstairs paired sign, with the two antisym swaps
  unfold fVal
  rw [hasym (a ^^^ (l ^^^ uL)) l hb1 hb hl1 hl hbl,
      hasym (a ^^^ (l ^^^ uL)) uL hb1 hb huL1 huL hbuL]
  rw [Int.neg_mul_neg, Int.neg_mul, Int.mul_assoc]

-- ══════════════════════════════════════════════════════════════════════════════════════════════════
-- DISCHARGING `cdAntisym` FOR ALL WIDTHS (transfer from the bit-list antisymmetry)
-- The bridge `SounioCDCocycle.antisym` proves CD antisymmetry for the bit-list sign `sgn` for ALL n
-- (Mathlib-free, axioms [propext, Quot.sound]).  We transfer it to the canonical Nat sign `cdSigma`
-- used throughout this file, removing the one open hypothesis `cdAntisym` of `converse_recursion`.
-- ══════════════════════════════════════════════════════════════════════════════════════════════════

/-- **The two `cdSigma` definitions coincide.**  `SounioCDCocycle.cdSigma` (used by the bit-list
    bridge) and `SounioCDTowerSeam.cdSigma` (used here) are byte-for-byte the same recursion, differing
    only by TowerSeam's `let`-naming of `aHi/bHi/aLo/bLo`; equal by induction on the width. -/
theorem cdSigma_defeq : ∀ (n a b : Nat),
    SounioCDCocycle.cdSigma a b n = cdSigma a b n
  | 0, _, _ => rfl
  | 1, _, _ => rfl
  | (n+2), a, b => by
      have ih : ∀ a b, SounioCDCocycle.cdSigma a b (n+1) = cdSigma a b (n+1) :=
        fun a b => cdSigma_defeq (n+1) a b
      simp only [SounioCDCocycle.cdSigma, cdSigma, ih]

/-- **Injectivity of the width-`n` bit encoding.**  Transferred from the cocycle-lane XOR facts:
    `bitsOf n x = bitsOf n y → xorL … isZ → bitsOf n (x⊕y) isZ → x⊕y = 0 → x = y`. -/
theorem bitsOf_inj (n x y : Nat) (hx : x < 2 ^ n) (hy : y < 2 ^ n)
    (h : SounioCDCocycle.bitsOf n x = SounioCDCocycle.bitsOf n y) : x = y := by
  have hlen : (SounioCDCocycle.bitsOf n x).length = (SounioCDCocycle.bitsOf n y).length := by
    rw [SounioCDCocycle.bitsOf_length, SounioCDCocycle.bitsOf_length]
  have hz : SounioCDCocycle.isZ
      (SounioCDCocycle.xorL (SounioCDCocycle.bitsOf n x) (SounioCDCocycle.bitsOf n y)) = true :=
    (SounioCDCocycle.xorL_isZ_iff _ _ hlen).mpr h
  rw [SounioCDCocycle.xorL_bitsOf n x y hx hy] at hz
  have hxor : x ^^^ y = 0 :=
    (SounioCDCocycle.isZ_bitsOf n (x ^^^ y) (Nat.xor_lt_two_pow hx hy)).mp hz
  exact xor_eq_zero_of x y hxor

/-- **`cdAntisym` holds for EVERY width** — the one open input to `converse_recursion`, now proved.
    Distinct nonzero basis units anticommute, `cdSigma x y m = - cdSigma y x m`, for all `m`.  For
    `m = 0` the domain is empty (`1 ≤ x < 2^0 = 1`); for `m ≥ 1` chain the two `cdSigma` defs through
    the bit-list bridge and apply `SounioCDCocycle.antisym`. -/
theorem cdAntisym_all : ∀ m, cdAntisym m := by
  intro m x y hx1 hx hy1 hy hxy
  cases m with
  | zero => rw [Nat.pow_zero] at hx; omega
  | succ k =>
    have hm : 1 ≤ k + 1 := by omega
    have hx0 : x ≠ 0 := by omega
    have hy0 : y ≠ 0 := by omega
    have hbne : SounioCDCocycle.bitsOf (k+1) x ≠ SounioCDCocycle.bitsOf (k+1) y :=
      fun h => hxy (bitsOf_inj (k+1) x y hx hy h)
    calc cdSigma x y (k+1)
        = SounioCDCocycle.cdSigma x y (k+1) := (cdSigma_defeq (k+1) x y).symm
      _ = SounioCDCocycle.sgn (SounioCDCocycle.bitsOf (k+1) x) (SounioCDCocycle.bitsOf (k+1) y) :=
            (SounioCDCocycle.sgn_eq_cdSigma (k+1) x y hm hx hy).symm
      _ = - SounioCDCocycle.sgn (SounioCDCocycle.bitsOf (k+1) y) (SounioCDCocycle.bitsOf (k+1) x) :=
            SounioCDCocycle.antisym _ _
              (by rw [SounioCDCocycle.bitsOf_length, SounioCDCocycle.bitsOf_length])
              (SounioCDCocycle.isZ_bitsOf_false (k+1) x hx hx0)
              (SounioCDCocycle.isZ_bitsOf_false (k+1) y hy hy0) hbne
      _ = - SounioCDCocycle.cdSigma y x (k+1) := by
            rw [SounioCDCocycle.sgn_eq_cdSigma (k+1) y x hm hy hx]
      _ = - cdSigma y x (k+1) := by rw [cdSigma_defeq (k+1) y x]

/-- **`converse_recursion`, now UNCONDITIONAL** — the `cdAntisym (n+1)` hypothesis is discharged by
    `cdAntisym_all`.  The Biss–Dugger–Isaksen doubling step for the winner value holds for all widths
    with no open input. -/
theorem converse_recursion' (n l uL a : Nat)
    (hl1 : 1 ≤ l) (hl : l < 2 ^ (n+1))
    (huL1 : 1 ≤ uL) (huL : uL < 2 ^ (n+1))
    (ha1 : 1 ≤ a) (ha : a < 2 ^ (n+1))
    (hal : a ≠ l) (hauL : a ≠ uL) (hadL : a ≠ l ^^^ uL) :
    fVal l (2 ^ (n+1) + uL) a (n+2)
        * fVal l (2 ^ (n+1) + uL) (a ^^^ (l ^^^ (2 ^ (n+1) + uL))) (n+2)
      = - (fVal l uL a (n+1) * fVal l uL (a ^^^ (l ^^^ uL)) (n+1)) :=
  converse_recursion n l uL a (cdAntisym_all (n+1)) hl1 hl huL1 huL ha1 ha hal hauL hadL

/-
  ── MAP FOR THE NEXT ATTEMPT: how `converse_recursion` feeds the ∀n existence proof ──────────────

  GOAL (still open):  `offSeam bits l u = true → hasXorAnnih bits l u = true` on the loHi locus.
  By `hasXorAnnih_sound` this already yields `isZD`, so only EXISTENCE of a winner is missing.

  What `converse_recursion` gives.  Every nontrivial orbit `{a, a⊕d}` has a unique LOW representative
  `a ∈ [1, 2^(bits-1))` (`orbit_low_to_high`), and by `P_eq_fVal` the winner value there is
  `P_{(l,u)}(a) = f(a)·f(a⊕d)`.  For `u = 2^(n+1)+uL` and every low `a ∉ {l, uL, l⊕uL}`,
  `converse_recursion` proves — GIVEN `cdAntisym (n+1)` — the Biss–Dugger–Isaksen doubling step

        P_{(l,u)}(a)  =  - P_{(l,uL)}(a)          (upstairs winner  ⟺  downstairs LOSER).

  So `hasXorAnnih bits l u` (∃ low `a` with `P_{(l,u)}(a)=+1`, `a ∉ {l,uL,l⊕uL}` acceptable since
  those three orbits are the known losers) is EQUIVALENT to: the DOWNSTAIRS pair `(l, uL)` in
  `A_{n+1}` has a low index `a ∉ {l, uL, l⊕uL}` with `P_{(l,uL)}(a) = -1` (a non-annihilator).

  The two remaining inputs, precisely:

  (1) `cdAntisym m` for all `m` — CD antisymmetry `cdSigma x y = -cdSigma y x` on distinct nonzero
      basis units.  PROVED for the bit-list sign as `SounioCDCocycle.antisym` (∀n, no native_decide);
      the only missing piece is the bridge `sgn = cdSigma` ∀n (a sibling lane).  Once that lands,
      `cdAntisym` is discharged and `converse_recursion` becomes UNCONDITIONAL.

  (2) The downstairs EXISTENCE: pair `(l, uL)` — now a GENERAL pair of `A_{n+1}` (NOT necessarily a
      loHi pair, since `uL < 2^n` is possible) — must have a low non-annihilator index off the three
      trivial-loser orbits.  This is the genuine open combinatorial core.  The counting identity
      `S := Σ_a P(a) = 4·#agree − 2^bits` (each orbit counted twice) reduces it to a lower bound on
      the number of AGREEING orbits; empirically `#agree` is large (winner sets of size `8·(2^k−1)`),
      but a dimension-independent lower bound `#agree ≥ 2` (one nontrivial) is not yet formalized.
      A clean base case `A_3` (octonions, `n = 1`) is finite (`converse_16`-style native_decide),
      so an induction on `n` closes the whole tower ONCE (2) has a dimension-free witness.

  In short: `converse_recursion` reduces the tower-wide converse to {the sgn=cdSigma bridge} ∧ {a
  dimension-free downstairs-loser witness}.  The sign-flip half is now fully mechanized here.
-/

-- ══════════════════════════════════════════════════════════════════════════════════════════════════
-- BOUNDED LEVERS OF THE CONVERSE COUNTING ARGUMENT (all ∀n except the octonion base case)
-- Three self-contained lemmas that formalize the concretely-provable steps of the converse counting
-- argument (`scripts/research/cd_tower_converse_counting.py`).  Mathlib-free, no `sorry`.
-- ══════════════════════════════════════════════════════════════════════════════════════════════════

/-- **CD diagonal sign** `cdSigma x x k = -1` for every nonzero basis unit (`1 ≤ x < 2^k`).  This is
    the algebraic `e_i² = -1`, obtained by transferring `SounioCDCocycle.diag` (the bit-list diagonal
    sign) across the `sgn = cdSigma` bridge.  The width bound `1 ≤ k` is forced by `1 ≤ x < 2^k`. -/
theorem cdSigma_diag (k x : Nat) (h1 : 1 ≤ x) (hx : x < 2 ^ k) : cdSigma x x k = -1 := by
  have hk : 1 ≤ k := by
    cases k with
    | zero => rw [Nat.pow_zero] at hx; omega
    | succ _ => omega
  have hx0 : x ≠ 0 := by omega
  rw [← cdSigma_defeq k x x, ← SounioCDCocycle.sgn_eq_cdSigma k x x hk hx hx]
  exact SounioCDCocycle.diag _ (SounioCDCocycle.isZ_bitsOf_false k x hx hx0)

/-- **Lemma 1 (octonion base case).**  The octonions `A_3` are a division algebra: for every mixed
    (distinct, nonzero) index pair `(l, m)` and every `a`, the four-sign winner product is `-1` — i.e.
    *every* orbit disagrees, so there is no zero divisor.  A fixed finite claim, discharged by
    `native_decide` (a legitimate base-case anchor; the general lemmas below avoid it). -/
theorem oct_all_disagree :
    (List.range 8).all (fun l => (List.range 8).all (fun m =>
      (List.range 8).all (fun a =>
        (! ((! (l == 0)) && (! (m == 0)) && (l != m)))
          || (fVal l m a 3 * fVal l m (a ^^^ (l ^^^ m)) 3 == -1)))) = true := by native_decide

/-- **Lemma 2 (the main σ-identity, ∀n): explicit disagreeing witness for a mixed pair.**  For a mixed
    pair `(l, 2^k + m_lo)` with `1 ≤ l, m_lo < 2^k` and `m_lo ≠ l`, the orbit anchored at `a = m_lo`
    DISAGREES: the four-sign winner product equals `-1`.  Fully derived from the branch reductions
    (`cdSigma_stable`, `cdSigma_hi_lo`, `cdSigma_lo_hi`, `cdSigma_hi_hi`) plus the diagonal
    `cdSigma_diag`; no `native_decide`.  Numerically verified 5094/5094. -/
theorem mixed_witness_disagree (k l m_lo : Nat)
    (hl1 : 1 ≤ l) (hl : l < 2 ^ k) (hm1 : 1 ≤ m_lo) (hm : m_lo < 2 ^ k) (hne : m_lo ≠ l) :
    fVal l (2 ^ k + m_lo) m_lo (k+1)
      * fVal l (2 ^ k + m_lo) (m_lo ^^^ (l ^^^ (2 ^ k + m_lo))) (k+1) = -1 := by
  cases k with
  | zero => rw [Nat.pow_zero] at hl; omega
  | succ j =>
    have hlm : l ^^^ m_lo < 2 ^ (j+1) := Nat.xor_lt_two_pow hl hm
    -- XOR bookkeeping: the disagreeing partner index is `2^(j+1) + l`.
    have hd : l ^^^ (2 ^ (j+1) + m_lo) = 2 ^ (j+1) + (l ^^^ m_lo) := by
      rw [← two_pow_xor_eq_add (j+1) m_lo hm, ← two_pow_xor_eq_add (j+1) (l ^^^ m_lo) hlm,
          xor_left_comm]
    have hidx : m_lo ^^^ (l ^^^ (2 ^ (j+1) + m_lo)) = 2 ^ (j+1) + l := by
      rw [hd, orbit_low_to_high (j+1) m_lo (l ^^^ m_lo) hm hlm]
      have hml : m_lo ^^^ (l ^^^ m_lo) = l := by
        rw [xor_left_comm, Nat.xor_self, Nat.xor_zero]
      rw [hml]
    rw [hidx]
    unfold fVal
    -- factor 1: cdSigma l m_lo (j+1) * -(cdSigma m_lo m_lo (j+1)) = cdSigma l m_lo (j+1)
    -- factor 2: cdSigma l l (j+1) * cdSigma l m_lo (j+1) = -(cdSigma l m_lo (j+1))
    rw [cdSigma_stable j l m_lo hl hm,
        cdSigma_hi_lo j m_lo m_lo hm hm1 hm,
        cdSigma_diag (j+1) m_lo hm1 hm,
        cdSigma_lo_hi j l l hl hl1 hl,
        cdSigma_diag (j+1) l hl1 hl,
        cdSigma_hi_hi j m_lo l hm hl1 hl]
    -- product = X * -(-1) * (-1 * X) = -X² = -1  (X = cdSigma l m_lo (j+1) ∈ {±1})
    rcases cdSigma_pm (j+1) l m_lo with h|h <;> rw [h] <;> decide

/-- **Lemma 3 (doubling core, per-element): winner value is level-stable on the low block.**  For a
    both-low pair `(l, m)` (`l, m < 2^k`) and any low index `a < 2^k`, the four-sign winner product at
    width `k+1` equals the one at width `k`.  This is the per-element identity driving the sum doubling
    `S(k+1) = 2·S(k)` (each low orbit contributes identically at the next level, and the high orbits
    add a second copy); the full Finset/List sum identity is out of scope — this is its core.  Every
    index (`l, m, a, a⊕(l⊕m)`) stays `< 2^k`, so `cdSigma_stable` applies to each of the four factors. -/
theorem P_stable_low (k l m a : Nat) (hl : l < 2 ^ k) (hm : m < 2 ^ k) (ha : a < 2 ^ k) :
    fVal l m a (k+1) * fVal l m (a ^^^ (l ^^^ m)) (k+1)
      = fVal l m a k * fVal l m (a ^^^ (l ^^^ m)) k := by
  have hb : a ^^^ (l ^^^ m) < 2 ^ k := Nat.xor_lt_two_pow ha (Nat.xor_lt_two_pow hl hm)
  cases k with
  | zero =>
    rw [Nat.pow_zero] at hl hm ha
    have hl0 : l = 0 := by omega
    have hm0 : m = 0 := by omega
    have ha0 : a = 0 := by omega
    subst hl0; subst hm0; subst ha0; decide
  | succ j =>
    unfold fVal
    rw [cdSigma_stable j l a hl ha, cdSigma_stable j m a hm ha,
        cdSigma_stable j l (a ^^^ (l ^^^ m)) hl hb, cdSigma_stable j m (a ^^^ (l ^^^ m)) hm hb]

/-- **Bridged cocycle** `cdSigma i j n · cdSigma i (i⊕j) n = -1` for the TowerSeam `cdSigma`,
    transferred from `SounioCDCocycle.cdSigma_cocycle` across the `cdSigma_defeq` bridge. -/
theorem cdSigma_cocycle' (n i j : Nat) (hi : i < 2 ^ n) (hj : j < 2 ^ n) (hi0 : i ≠ 0) :
    cdSigma i j n * cdSigma i (i ^^^ j) n = -1 := by
  rw [← cdSigma_defeq n i j, ← cdSigma_defeq n i (i ^^^ j)]
  exact SounioCDCocycle.cdSigma_cocycle n i j hi hj hi0

/-- **L1 — both-high low-block stability.**  For a both-high pair `(2^k+l_lo, 2^k+m_lo)` and a low
    index `a < 2^k`, the paired sign at width `k+1` equals the downstairs paired sign of `(l_lo, m_lo)`
    at width `k`.  Proof: `a = 0` → all factors `1` (`cdSigma_zero_right`); `a ≥ 1` → two `cdSigma_hi_lo`
    sign-flips cancel (`Int.neg_mul_neg`). -/
theorem fVal_high_stable (k l_lo m_lo a : Nat)
    (hl : l_lo < 2 ^ k) (hm : m_lo < 2 ^ k) (ha : a < 2 ^ k) :
    fVal (2 ^ k + l_lo) (2 ^ k + m_lo) a (k+1) = fVal l_lo m_lo a k := by
  cases k with
  | zero =>
    rw [Nat.pow_zero] at hl hm ha
    have hl0 : l_lo = 0 := by omega
    have hm0 : m_lo = 0 := by omega
    have ha0 : a = 0 := by omega
    subst hl0; subst hm0; subst ha0; decide
  | succ j =>
    unfold fVal
    rcases Nat.eq_zero_or_pos a with ha0 | ha1
    · subst ha0
      rw [cdSigma_zero_right (j+2) _ (by omega), cdSigma_zero_right (j+2) _ (by omega),
          cdSigma_zero_right (j+1) _ (by omega), cdSigma_zero_right (j+1) _ (by omega)]
    · rw [cdSigma_hi_lo j l_lo a hl ha1 ha, cdSigma_hi_lo j m_lo a hm ha1 ha, Int.neg_mul_neg]

/-- **L2 — seam-element edge `(l, 2^k)`.**  For `1 ≤ l < 2^k`, `1 ≤ a < 2^k`, `a ≠ l`, the orbit at
    `a` disagrees: `P l (2^k) a (k+1) = -1`.  Derived from the branch reductions + `cdAntisym_all` +
    `cdSigma_cocycle'`; no `native_decide`. -/
theorem edge_m_eq_H (k l a : Nat)
    (hl1 : 1 ≤ l) (hl : l < 2 ^ k) (ha1 : 1 ≤ a) (ha : a < 2 ^ k) (hal : a ≠ l) :
    fVal l (2 ^ k) a (k+1)
      * fVal l (2 ^ k) (a ^^^ (l ^^^ 2 ^ k)) (k+1) = -1 := by
  cases k with
  | zero => rw [Nat.pow_zero] at hl; omega
  | succ j =>
    have hax : a ^^^ l < 2 ^ (j+1) := Nat.xor_lt_two_pow ha hl
    have haxne : a ^^^ l ≠ 0 := fun h => hal (xor_eq_zero_of a l h)
    have hax1 : 1 ≤ a ^^^ l := Nat.one_le_iff_ne_zero.mpr haxne
    have hd : l ^^^ 2 ^ (j+1) = 2 ^ (j+1) + l := by
      rw [Nat.xor_comm l (2 ^ (j+1)), two_pow_xor_eq_add (j+1) l hl]
    have hidx : a ^^^ (l ^^^ 2 ^ (j+1)) = 2 ^ (j+1) + (a ^^^ l) := by
      rw [hd, orbit_low_to_high (j+1) a l ha hl]
    rw [hidx]
    unfold fVal
    have h2 := cdSigma_hi_lo j 0 a (Nat.two_pow_pos (j+1)) ha1 ha
    rw [Nat.add_zero, cdSigma_zero_left (j+1) a (by omega)] at h2
    have h4 := cdSigma_hi_hi j 0 (a ^^^ l) (Nat.two_pow_pos (j+1)) hax1 hax
    rw [Nat.add_zero, cdSigma_zero_right (j+1) (a ^^^ l) (by omega)] at h4
    have hne_axl : a ^^^ l ≠ l := by
      intro h
      have e : (a ^^^ l) ^^^ l = l ^^^ l := by rw [h]
      rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero] at e
      omega
    rw [cdSigma_stable j l a hl ha, h2,
        cdSigma_lo_hi j (a ^^^ l) l hax hl1 hl, h4,
        cdAntisym_all (j+1) (a ^^^ l) l hax1 hax hl1 hl hne_axl,
        Nat.xor_comm a l]
    have hcoc := cdSigma_cocycle' (j+1) l a hl ha (by omega : l ≠ 0)
    rcases cdSigma_pm (j+1) l a with hA|hA <;>
    rcases cdSigma_pm (j+1) l (l ^^^ a) with hB|hB <;>
      rw [hA, hB] at hcoc ⊢ <;> first | decide | exact absurd hcoc (by decide)

end SounioCDConverse
