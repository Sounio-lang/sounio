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
      A clean base case `A_3` (octonions, `n = 1`) is now proved structurally (`Q_base`, no
      `native_decide`), so an induction on `n` closes the whole tower ONCE (2) has a dimension-free witness.

  In short: `converse_recursion` reduces the tower-wide converse to {the sgn=cdSigma bridge} ∧ {a
  dimension-free downstairs-loser witness}.  The sign-flip half is now fully mechanized here.
-/

-- ══════════════════════════════════════════════════════════════════════════════════════════════════
-- BOUNDED LEVERS OF THE CONVERSE COUNTING ARGUMENT (ALL ∀n — the octonion base is now structural too)
-- Self-contained lemmas that formalize the concretely-provable steps of the converse counting
-- argument (`scripts/research/cd_tower_converse_counting.py`).  Mathlib-free, no `sorry`,
-- no `native_decide` anywhere in the ∀n chain (the octonion base `Q_base` is proved structurally).
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

/-- **L3 — seam-element edge `(l, 2^k+l)`.**  For `1 ≤ l < 2^k`, `1 ≤ a < 2^k`, the orbit at `a`
    disagrees: `P l (2^k+l) a (k+1) = -1`.  Here `d = l⊕(2^k+l) = 2^k`; low factor `= -1`, high
    factor `= 1`. -/
theorem edge_m_eq_H_plus_l (k l a : Nat)
    (hl1 : 1 ≤ l) (hl : l < 2 ^ k) (ha1 : 1 ≤ a) (ha : a < 2 ^ k) :
    fVal l (2 ^ k + l) a (k+1)
      * fVal l (2 ^ k + l) (a ^^^ (l ^^^ (2 ^ k + l))) (k+1) = -1 := by
  cases k with
  | zero => rw [Nat.pow_zero] at hl; omega
  | succ j =>
    have hd : l ^^^ (2 ^ (j+1) + l) = 2 ^ (j+1) := by
      rw [← two_pow_xor_eq_add (j+1) l hl, xor_left_comm, Nat.xor_self, Nat.xor_zero]
    have hidx : a ^^^ (l ^^^ (2 ^ (j+1) + l)) = 2 ^ (j+1) + a := by
      rw [hd, Nat.xor_comm a (2 ^ (j+1)), two_pow_xor_eq_add (j+1) a ha]
    rw [hidx]
    unfold fVal
    rw [cdSigma_stable j l a hl ha, cdSigma_hi_lo j l a hl ha1 ha,
        cdSigma_lo_hi j a l ha hl1 hl, cdSigma_hi_hi j l a hl ha1 ha]
    rcases cdSigma_pm (j+1) l a with hA|hA <;>
    rcases cdSigma_pm (j+1) a l with hB|hB <;> rw [hA, hB] <;> decide

/-- **L4 — seam-element edge `(2^k, 2^k+m_lo)`.**  For `1 ≤ m_lo < 2^k`, `1 ≤ a < 2^k`, `a ≠ m_lo`,
    the orbit at `a` disagrees: `P (2^k) (2^k+m_lo) a (k+1) = -1`.  Here `d = m_lo` (low); each factor
    is `(-1)·(-cdSigma m_lo · k)`, closed by `cdSigma_cocycle'` with `i = m_lo`. -/
theorem edge_l_eq_H (k m_lo a : Nat)
    (hm1 : 1 ≤ m_lo) (hm : m_lo < 2 ^ k) (ha1 : 1 ≤ a) (ha : a < 2 ^ k) (ham : a ≠ m_lo) :
    fVal (2 ^ k) (2 ^ k + m_lo) a (k+1)
      * fVal (2 ^ k) (2 ^ k + m_lo) (a ^^^ (2 ^ k ^^^ (2 ^ k + m_lo))) (k+1) = -1 := by
  cases k with
  | zero => rw [Nat.pow_zero] at hm; omega
  | succ j =>
    have hax : a ^^^ m_lo < 2 ^ (j+1) := Nat.xor_lt_two_pow ha hm
    have haxne : a ^^^ m_lo ≠ 0 := fun h => ham (xor_eq_zero_of a m_lo h)
    have hax1 : 1 ≤ a ^^^ m_lo := Nat.one_le_iff_ne_zero.mpr haxne
    have hd : (2 ^ (j+1)) ^^^ (2 ^ (j+1) + m_lo) = m_lo := by
      rw [← two_pow_xor_eq_add (j+1) m_lo hm, ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]
    have hidx : a ^^^ ((2 ^ (j+1)) ^^^ (2 ^ (j+1) + m_lo)) = a ^^^ m_lo := by rw [hd]
    rw [hidx]
    unfold fVal
    have h1 := cdSigma_hi_lo j 0 a (Nat.two_pow_pos (j+1)) ha1 ha
    rw [Nat.add_zero, cdSigma_zero_left (j+1) a (by omega)] at h1
    have h3 := cdSigma_hi_lo j 0 (a ^^^ m_lo) (Nat.two_pow_pos (j+1)) hax1 hax
    rw [Nat.add_zero, cdSigma_zero_left (j+1) (a ^^^ m_lo) (by omega)] at h3
    rw [h1, cdSigma_hi_lo j m_lo a hm ha1 ha, h3,
        cdSigma_hi_lo j m_lo (a ^^^ m_lo) hm hax1 hax, Nat.xor_comm a m_lo]
    have hcoc := cdSigma_cocycle' (j+1) m_lo a hm ha (by omega : m_lo ≠ 0)
    rcases cdSigma_pm (j+1) m_lo a with hA|hA <;>
    rcases cdSigma_pm (j+1) m_lo (m_lo ^^^ a) with hB|hB <;>
      rw [hA, hB] at hcoc ⊢ <;> first | decide | exact absurd hcoc (by decide)

/-- **Core disagreement identity.**  The transposed product of a distinct nonzero basis pair is `-1`:
    `cdSigma a b · cdSigma b a = -1`.  One antisymmetry (`cdSigma b a = -cdSigma a b`) then `(±1)²=1`.
    No `native_decide`. -/
theorem cdSigma_cross_neg (k a b : Nat)
    (ha1 : 1 ≤ a) (ha : a < 2 ^ k) (hb1 : 1 ≤ b) (hb : b < 2 ^ k) (hab : a ≠ b) :
    cdSigma a b k * cdSigma b a k = -1 := by
  rw [cdAntisym_all k b a hb1 hb ha1 ha (fun h => hab h.symm)]
  rcases cdSigma_pm k a b with h | h <;> rw [h] <;> decide

/-- **Both-low witness (base-case, no IH).**  For a both-low mixed pair `(l,m)` (`1≤l,m<2^(n+1)`,
    `l≠m`) the high orbit anchored at `a = 2^(n+1)+l` — i.e. `{2^(n+1)+l, 2^(n+1)+m}` — DISAGREES:
    the four-sign product is `-1`.  Each factor collapses through `cdSigma_lo_hi` (nonzero low part)
    plus `cdSigma_diag`, and the residue is the core identity `cdSigma_cross_neg`.  No `native_decide`,
    no induction hypothesis. -/
theorem both_low_witness_disagree (n l m : Nat)
    (hl1 : 1 ≤ l) (hl : l < 2 ^ (n+1)) (hm1 : 1 ≤ m) (hm : m < 2 ^ (n+1)) (hlm : l ≠ m) :
    fVal l m (2 ^ (n+1) + l) (n+2)
      * fVal l m ((2 ^ (n+1) + l) ^^^ (l ^^^ m)) (n+2) = -1 := by
  have hidx : (2 ^ (n+1) + l) ^^^ (l ^^^ m) = 2 ^ (n+1) + m := by
    rw [← two_pow_xor_eq_add (n+1) l hl, ← two_pow_xor_eq_add (n+1) m hm,
        Nat.xor_assoc, ← Nat.xor_assoc l l m, Nat.xor_self, Nat.zero_xor]
  rw [hidx]
  unfold fVal
  rw [cdSigma_lo_hi n l l hl hl1 hl, cdSigma_lo_hi n l m hl hm1 hm,
      cdSigma_lo_hi n m l hm hl1 hl, cdSigma_lo_hi n m m hm hm1 hm,
      cdSigma_diag (n+1) l hl1 hl, cdSigma_diag (n+1) m hm1 hm]
  have hcn := cdSigma_cross_neg (n+1) l m hl1 hl hm1 hm hlm
  rcases cdSigma_pm (n+1) l m with h | h <;> rcases cdSigma_pm (n+1) m l with h2 | h2 <;>
    rw [h, h2] at hcn ⊢ <;> first | decide | exact absurd hcn (by decide)

/-- **Both-high witness (base-case, no IH).**  For a both-high mixed pair
    `(2^(n+1)+llo, 2^(n+1)+mlo)` (`llo≠mlo`, both `1≤·<2^(n+1)`) the low orbit `{llo, mlo}` DISAGREES.
    `fVal_high_stable` collapses both factors to level `n+1`, leaving the `{llo,mlo}` orbit which is
    `-1` by `cdSigma_diag` + `cdSigma_cross_neg`.  No `native_decide`, no induction hypothesis. -/
theorem both_high_witness_disagree (n llo mlo : Nat)
    (hl1 : 1 ≤ llo) (hl : llo < 2 ^ (n+1)) (hm1 : 1 ≤ mlo) (hm : mlo < 2 ^ (n+1)) (hlm : llo ≠ mlo) :
    fVal (2 ^ (n+1) + llo) (2 ^ (n+1) + mlo) llo (n+2)
      * fVal (2 ^ (n+1) + llo) (2 ^ (n+1) + mlo)
          (llo ^^^ ((2 ^ (n+1) + llo) ^^^ (2 ^ (n+1) + mlo))) (n+2) = -1 := by
  have hd : (2 ^ (n+1) + llo) ^^^ (2 ^ (n+1) + mlo) = llo ^^^ mlo := by
    rw [← two_pow_xor_eq_add (n+1) llo hl, ← two_pow_xor_eq_add (n+1) mlo hm,
        Nat.xor_assoc, xor_left_comm llo (2 ^ (n+1)) mlo, ← Nat.xor_assoc,
        Nat.xor_self, Nat.zero_xor]
  have hidx : llo ^^^ ((2 ^ (n+1) + llo) ^^^ (2 ^ (n+1) + mlo)) = mlo := by
    rw [hd, ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]
  rw [hidx, fVal_high_stable (n+1) llo mlo llo hl hm hl,
      fVal_high_stable (n+1) llo mlo mlo hl hm hm]
  unfold fVal
  rw [cdSigma_diag (n+1) llo hl1 hl, cdSigma_diag (n+1) mlo hm1 hm]
  have hcn := cdSigma_cross_neg (n+1) mlo llo hm1 hm hl1 hl (fun h => hlm h.symm)
  rcases cdSigma_pm (n+1) mlo llo with h | h <;> rcases cdSigma_pm (n+1) llo mlo with h2 | h2 <;>
    rw [h, h2] at hcn ⊢ <;> first | decide | exact absurd hcn (by decide)

-- ══════════════════════════════════════════════════════════════════════════════════════════════════
-- Q: a non-exceptional disagreeing witness exists for EVERY mixed pair (ORDINARY induction on level).
-- ══════════════════════════════════════════════════════════════════════════════════════════════════

/-- `fVal` is symmetric in the pair `(l,m)` (the paired sign `cdSigma l a · cdSigma m a`). -/
theorem fVal_comm (l m a bits : Nat) : fVal l m a bits = fVal m l a bits := by
  unfold fVal; rw [Int.mul_comm]

/-- The winner-existence statement at level `k`: every mixed pair has a non-exceptional
    (`a ∉ {0, l⊕m, l, m}`) orbit whose four-sign product disagrees (`= -1`). -/
def Qstmt (k : Nat) : Prop :=
  ∀ l m, 1 ≤ l → l < 2 ^ k → 1 ≤ m → m < 2 ^ k → l ≠ m →
    ∃ a, a < 2 ^ k ∧ a ≠ 0 ∧ a ≠ l ^^^ m ∧ a ≠ l ∧ a ≠ m ∧
         fVal l m a k * fVal l m (a ^^^ (l ^^^ m)) k = -1

/-- The witness set is symmetric under swapping the pair: a witness for `(l,m)` is a witness for
    `(m,l)` (guards symmetric, `l⊕m = m⊕l`, `fVal` symmetric). -/
theorem exists_witness_symm (k l m : Nat)
    (h : ∃ a, a < 2 ^ k ∧ a ≠ 0 ∧ a ≠ l ^^^ m ∧ a ≠ l ∧ a ≠ m ∧
         fVal l m a k * fVal l m (a ^^^ (l ^^^ m)) k = -1) :
    ∃ a, a < 2 ^ k ∧ a ≠ 0 ∧ a ≠ m ^^^ l ∧ a ≠ m ∧ a ≠ l ∧
         fVal m l a k * fVal m l (a ^^^ (m ^^^ l)) k = -1 := by
  obtain ⟨a, h1, h2, h3, h4, h5, h6⟩ := h
  refine ⟨a, h1, h2, ?_, h5, h4, ?_⟩
  · rw [Nat.xor_comm m l]; exact h3
  · rw [Nat.xor_comm m l, fVal_comm m l, fVal_comm m l]; exact h6

/-- **Base case `Qstmt 3` (octonions) — structural, NO `native_decide`.**  Every mixed pair in `A_3`
    has a non-exceptional disagreeing orbit, by the SAME six-way seam split as the inductive step but
    with the two doubling cases discharged *without* an IH: both-low takes the high witness
    `a = 2^2+l` (`both_low_witness_disagree`), both-high takes the low witness `a = llo`
    (`both_high_witness_disagree`), and the edges/mixed-non-edge reuse L2/L3/L4/`mixed_witness_disagree`
    at `W = 2`.  This removes the last `native_decide` from the ∀n coincidence chain — every downstream
    theorem is now `[propext, Quot.sound]`. -/
theorem Q_base : Qstmt 3 := by
  have h4 : (2 : Nat) ^ 2 = 4 := by decide
  have h8 : (2 : Nat) ^ 3 = 8 := by decide
  have core : ∀ l m, 1 ≤ l → l < m → m < 2 ^ 3 →
      ∃ a, a < 2 ^ 3 ∧ a ≠ 0 ∧ a ≠ l ^^^ m ∧ a ≠ l ∧ a ≠ m ∧
           fVal l m a 3 * fVal l m (a ^^^ (l ^^^ m)) 3 = -1 := by
    intro l m hl1 hlm hm8
    have hm1 : 1 ≤ m := by omega
    by_cases hmlt : m < 2 ^ 2
    · -- CASE 4: both low — high witness a = 2^2 + l
      have hl4 : l < 2 ^ 2 := by omega
      have hdlt : l ^^^ m < 2 ^ 2 := Nat.xor_lt_two_pow hl4 hmlt
      refine ⟨2 ^ 2 + l, by omega, by omega, by omega, by omega, by omega, ?_⟩
      exact both_low_witness_disagree 1 l m hl1 hl4 hm1 hmlt (by omega)
    · by_cases hllt : l < 2 ^ 2
      · by_cases hmH : m = 2 ^ 2
        · -- CASE 1: m = 2^2
          subst hmH
          obtain ⟨a, ha1, haH, hal⟩ : ∃ a, 1 ≤ a ∧ a < 2 ^ 2 ∧ a ≠ l := by
            by_cases hle : l = 1
            · exact ⟨2, by omega, by omega, by omega⟩
            · exact ⟨1, by omega, by omega, by omega⟩
          have hlmv : l ^^^ 2 ^ 2 = 2 ^ 2 + l := by
            rw [Nat.xor_comm l (2 ^ 2), two_pow_xor_eq_add 2 l hllt]
          refine ⟨a, by omega, by omega, ?_, hal, by omega, ?_⟩
          · rw [hlmv]; omega
          · exact edge_m_eq_H 2 l a hl1 hllt ha1 haH hal
        · -- m > 2^2
          obtain ⟨mlo, rfl⟩ : ∃ mlo, m = 2 ^ 2 + mlo := ⟨m - 2 ^ 2, by omega⟩
          have hmloH : mlo < 2 ^ 2 := by omega
          have hmlo1 : 1 ≤ mlo := by omega
          by_cases hmlol : mlo = l
          · -- CASE 2: m = 2^2 + l
            obtain ⟨a, ha1, haH, hal⟩ : ∃ a, 1 ≤ a ∧ a < 2 ^ 2 ∧ a ≠ l := by
              by_cases hle : l = 1
              · exact ⟨2, by omega, by omega, by omega⟩
              · exact ⟨1, by omega, by omega, by omega⟩
            have hlmv : l ^^^ (2 ^ 2 + mlo) = 2 ^ 2 := by
              rw [hmlol, ← two_pow_xor_eq_add 2 l hllt, xor_left_comm, Nat.xor_self, Nat.xor_zero]
            refine ⟨a, by omega, by omega, ?_, hal, by omega, ?_⟩
            · rw [hlmv]; omega
            · rw [hmlol]; exact edge_m_eq_H_plus_l 2 l a hl1 hllt ha1 haH
          · -- CASE 5: mixed non-edge
            have hlmv : l ^^^ (2 ^ 2 + mlo) = 2 ^ 2 + (l ^^^ mlo) := by
              rw [← two_pow_xor_eq_add 2 mlo hmloH, xor_left_comm,
                  two_pow_xor_eq_add 2 (l ^^^ mlo) (Nat.xor_lt_two_pow hllt hmloH)]
            refine ⟨mlo, by omega, by omega, ?_, hmlol, by omega, ?_⟩
            · rw [hlmv]; omega
            · exact mixed_witness_disagree 2 l mlo hl1 hllt hmlo1 hmloH hmlol
      · -- l ≥ 2^2, both high (l < m ⟹ m ≥ 2^2)
        by_cases hlH : l = 2 ^ 2
        · -- CASE 3: l = 2^2
          subst hlH
          obtain ⟨mlo, rfl⟩ : ∃ mlo, m = 2 ^ 2 + mlo := ⟨m - 2 ^ 2, by omega⟩
          have hmloH : mlo < 2 ^ 2 := by omega
          have hmlo1 : 1 ≤ mlo := by omega
          obtain ⟨a, ha1, haH, ham⟩ : ∃ a, 1 ≤ a ∧ a < 2 ^ 2 ∧ a ≠ mlo := by
            by_cases hle : mlo = 1
            · exact ⟨2, by omega, by omega, by omega⟩
            · exact ⟨1, by omega, by omega, by omega⟩
          have hlmv : (2 ^ 2) ^^^ (2 ^ 2 + mlo) = mlo := by
            rw [← two_pow_xor_eq_add 2 mlo hmloH, ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]
          refine ⟨a, by omega, by omega, ?_, ?_, by omega, ?_⟩
          · rw [hlmv]; exact ham
          · omega
          · exact edge_l_eq_H 2 mlo a hmlo1 hmloH ha1 haH ham
        · -- CASE 6: both high, l ≠ 2^2 — low witness a = llo
          obtain ⟨llo, rfl⟩ : ∃ llo, l = 2 ^ 2 + llo := ⟨l - 2 ^ 2, by omega⟩
          obtain ⟨mlo, rfl⟩ : ∃ mlo, m = 2 ^ 2 + mlo := ⟨m - 2 ^ 2, by omega⟩
          have hlloH : llo < 2 ^ 2 := by omega
          have hmloH : mlo < 2 ^ 2 := by omega
          have hllo1 : 1 ≤ llo := by omega
          have hmlo1 : 1 ≤ mlo := by omega
          have hlomlo : llo ≠ mlo := by omega
          have hd : (2 ^ 2 + llo) ^^^ (2 ^ 2 + mlo) = llo ^^^ mlo := by
            rw [← two_pow_xor_eq_add 2 llo hlloH, ← two_pow_xor_eq_add 2 mlo hmloH,
                Nat.xor_assoc, xor_left_comm llo (2 ^ 2) mlo, ← Nat.xor_assoc,
                Nat.xor_self, Nat.zero_xor]
          have hguard : llo ≠ (2 ^ 2 + llo) ^^^ (2 ^ 2 + mlo) := by
            rw [hd]; intro h
            have : mlo = 0 := by
              calc mlo = llo ^^^ (llo ^^^ mlo) := by
                        rw [← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]
                _ = llo ^^^ llo := by rw [← h]
                _ = 0 := Nat.xor_self llo
            omega
          refine ⟨llo, by omega, by omega, hguard, by omega, by omega, ?_⟩
          exact both_high_witness_disagree 1 llo mlo hllo1 hlloH hmlo1 hmloH hlomlo
  intro l m hl1 hl hm1 hm hlm
  rcases Nat.lt_or_ge l m with hlt | hge
  · exact core l m hl1 hlt hm
  · exact exists_witness_symm 3 m l (core m l hm1 (by omega) hl)

/-- **The inductive step `Qstmt W → Qstmt (W+1)`** (ORDINARY induction, `W ≥ 3`).  Six-case split on
    the seam position of the pair; edges by L2/L3/L4, both-low by `P_stable_low`+IH, mixed non-edge by
    `mixed_witness_disagree`, both-high by L1+IH.  WLOG `l < m` via `exists_witness_symm`. -/
theorem Qstep (W : Nat) (hW : 3 ≤ W) (ih : Qstmt W) : Qstmt (W+1) := by
  have hpow : 2 ^ (W+1) = 2 ^ W * 2 := Nat.pow_succ 2 W
  have h8 : 8 ≤ 2 ^ W := by
    have h : 2 ^ 3 ≤ 2 ^ W := Nat.pow_le_pow_right (by decide) (show 3 ≤ W by omega)
    exact h
  have core : ∀ l m, 1 ≤ l → l < m → m < 2 ^ (W+1) →
      ∃ a, a < 2 ^ (W+1) ∧ a ≠ 0 ∧ a ≠ l ^^^ m ∧ a ≠ l ∧ a ≠ m ∧
           fVal l m a (W+1) * fVal l m (a ^^^ (l ^^^ m)) (W+1) = -1 := by
    intro l m hl1 hlm hmW
    have hm1 : 1 ≤ m := by omega
    by_cases hmlt : m < 2 ^ W
    · -- CASE 4: both low
      have hlW' : l < 2 ^ W := by omega
      obtain ⟨a, ha, ha0, had, hal, ham, haP⟩ := ih l m hl1 hlW' hm1 hmlt (by omega)
      exact ⟨a, by omega, ha0, had, hal, ham,
        by rw [P_stable_low W l m a hlW' hmlt ha]; exact haP⟩
    · by_cases hllt : l < 2 ^ W
      · -- ¬hi_l, hi_m
        by_cases hmH : m = 2 ^ W
        · -- CASE 1: m = 2^W
          subst hmH
          have hlW' : l < 2 ^ W := by omega
          obtain ⟨a, ha1, haH, hal⟩ : ∃ a, 1 ≤ a ∧ a < 2 ^ W ∧ a ≠ l := by
            by_cases hle : l = 1
            · exact ⟨2, by omega, by omega, by omega⟩
            · exact ⟨1, by omega, by omega, by omega⟩
          have hlmv : l ^^^ 2 ^ W = 2 ^ W + l := by
            rw [Nat.xor_comm l (2 ^ W), two_pow_xor_eq_add W l hlW']
          refine ⟨a, by omega, by omega, by omega, hal, by omega, ?_⟩
          exact edge_m_eq_H W l a hl1 hlW' ha1 haH hal
        · -- m > 2^W
          obtain ⟨mlo, rfl⟩ : ∃ mlo, m = 2 ^ W + mlo := ⟨m - 2 ^ W, by omega⟩
          have hmloH : mlo < 2 ^ W := by omega
          have hmlo1 : 1 ≤ mlo := by omega
          by_cases hmlol : mlo = l
          · -- CASE 2: m = 2^W + l
            obtain ⟨a, ha1, haH, hal⟩ : ∃ a, 1 ≤ a ∧ a < 2 ^ W ∧ a ≠ l := by
              by_cases hle : l = 1
              · exact ⟨2, by omega, by omega, by omega⟩
              · exact ⟨1, by omega, by omega, by omega⟩
            have hlmv : l ^^^ (2 ^ W + mlo) = 2 ^ W := by
              rw [hmlol, ← two_pow_xor_eq_add W l hllt, xor_left_comm, Nat.xor_self, Nat.xor_zero]
            refine ⟨a, by omega, by omega, ?_, hal, by omega, ?_⟩
            · rw [hlmv]; omega
            · rw [hmlol]; exact edge_m_eq_H_plus_l W l a hl1 hllt ha1 haH
          · -- CASE 5: mixed non-edge
            have hmlol' : mlo ≠ l := hmlol
            have hlmv : l ^^^ (2 ^ W + mlo) = 2 ^ W + (l ^^^ mlo) := by
              rw [← two_pow_xor_eq_add W mlo hmloH, xor_left_comm,
                  two_pow_xor_eq_add W (l ^^^ mlo) (Nat.xor_lt_two_pow hllt hmloH)]
            refine ⟨mlo, by omega, by omega, ?_, hmlol', by omega, ?_⟩
            · rw [hlmv]; omega
            · exact mixed_witness_disagree W l mlo hl1 hllt hmlo1 hmloH hmlol'
      · -- l ≥ 2^W, both high
        by_cases hlH : l = 2 ^ W
        · -- CASE 3: l = 2^W
          subst hlH
          obtain ⟨mlo, rfl⟩ : ∃ mlo, m = 2 ^ W + mlo := ⟨m - 2 ^ W, by omega⟩
          have hmloH : mlo < 2 ^ W := by omega
          have hmlo1 : 1 ≤ mlo := by omega
          obtain ⟨a, ha1, haH, ham⟩ : ∃ a, 1 ≤ a ∧ a < 2 ^ W ∧ a ≠ mlo := by
            by_cases hle : mlo = 1
            · exact ⟨2, by omega, by omega, by omega⟩
            · exact ⟨1, by omega, by omega, by omega⟩
          have hlmv : (2 ^ W) ^^^ (2 ^ W + mlo) = mlo := by
            rw [← two_pow_xor_eq_add W mlo hmloH, ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]
          refine ⟨a, by omega, by omega, ?_, ?_, by omega, ?_⟩
          · rw [hlmv]; exact ham
          · omega
          · exact edge_l_eq_H W mlo a hmlo1 hmloH ha1 haH ham
        · -- CASE 6: both high, l ≠ 2^W
          obtain ⟨llo, rfl⟩ : ∃ llo, l = 2 ^ W + llo := ⟨l - 2 ^ W, by omega⟩
          obtain ⟨mlo, rfl⟩ : ∃ mlo, m = 2 ^ W + mlo := ⟨m - 2 ^ W, by omega⟩
          have hlloH : llo < 2 ^ W := by omega
          have hmloH : mlo < 2 ^ W := by omega
          have hllo1 : 1 ≤ llo := by omega
          have hmlo1 : 1 ≤ mlo := by omega
          have hlomlo : llo ≠ mlo := by omega
          obtain ⟨a, ha, ha0, had, hal, ham, haP⟩ := ih llo mlo hllo1 hlloH hmlo1 hmloH hlomlo
          have hxor : (2 ^ W + llo) ^^^ (2 ^ W + mlo) = llo ^^^ mlo := by
            rw [← two_pow_xor_eq_add W llo hlloH, ← two_pow_xor_eq_add W mlo hmloH,
                Nat.xor_assoc, xor_left_comm llo (2 ^ W) mlo, ← Nat.xor_assoc,
                Nat.xor_self, Nat.zero_xor]
          have hbH : a ^^^ (llo ^^^ mlo) < 2 ^ W :=
            Nat.xor_lt_two_pow ha (Nat.xor_lt_two_pow hlloH hmloH)
          refine ⟨a, by omega, ha0, ?_, by omega, by omega, ?_⟩
          · rw [hxor]; exact had
          · rw [hxor, fVal_high_stable W llo mlo a hlloH hmloH ha,
                fVal_high_stable W llo mlo (a ^^^ (llo ^^^ mlo)) hlloH hmloH hbH]
            exact haP
  intro l m hl1 hlW hm1 hmW hlm
  rcases Nat.lt_or_ge l m with hlt | hge
  · exact core l m hl1 hlt hmW
  · exact exists_witness_symm (W+1) m l (core m l hm1 (by omega) hlW)

/-- **Q for all levels `≥ 3`** (ORDINARY induction on the offset): every mixed pair has a
    non-exceptional disagreeing orbit. -/
theorem Q_all : ∀ k, Qstmt (k + 3)
  | 0 => Q_base
  | (k+1) => Qstep (k + 3) (by omega) (Q_all k)

/-- Convenience repackaging: `Qstmt W` for every `W ≥ 3`. -/
theorem Q_ge3 (W : Nat) (hW : 3 ≤ W) : Qstmt W := by
  obtain ⟨k, rfl⟩ : ∃ k, W = k + 3 := ⟨W - 3, by omega⟩
  exact Q_all k

-- ══════════════════════════════════════════════════════════════════════════════════════════════════
-- THE CONNECTION: offSeam ⟹ hasXorAnnih on the loHi locus, for ALL bits ≥ 4.
-- ══════════════════════════════════════════════════════════════════════════════════════════════════

/-- **The tower-wide CONVERSE, on the loHi locus (`l < 2^(bits-1) ≤ u < 2^bits`), for all `bits ≥ 4`.**
    An off-seam lower×upper pair admits an XOR-linked 2-term annihilator: `offSeam ⟹ hasXorAnnih`.
    Proof: `Q_ge3` at level `bits-1` produces a non-exceptional disagreeing witness `a` for the
    downstairs pair `(l, u - 2^(bits-1))`; `converse_recursion'` flips its `-1` to `+1` upstairs,
    which is exactly the `hasXorAnnih` winner product.  The `4 ≤ bits` bound is essential — at
    `bits = 3` the octonions have no zero divisors, so off-seam pairs (e.g. `(1,6)`) genuinely fail
    `hasXorAnnih` (matching `ConverseConjecture`'s `4 ≤ bits`). -/
theorem converse_holds (bits l u : Nat) (hb : 4 ≤ bits)
    (hl1 : 1 ≤ l) (hl : l < 2 ^ (bits-1)) (hu1 : 2 ^ (bits-1) ≤ u) (hu : u < 2 ^ bits)
    (hoff : offSeam bits l u = true) :
    hasXorAnnih bits l u = true := by
  obtain ⟨uL, rfl⟩ : ∃ uL, u = 2 ^ (bits-1) + uL := ⟨u - 2 ^ (bits-1), by omega⟩
  have hpow : 2 ^ bits = 2 ^ (bits-1) * 2 := by
    have hps : 2 ^ ((bits-1)+1) = 2 ^ (bits-1) * 2 := Nat.pow_succ 2 (bits-1)
    rw [Nat.sub_add_cancel (show 1 ≤ bits by omega)] at hps
    exact hps
  have huL : uL < 2 ^ (bits-1) := by omega
  have hdval : l ^^^ (2 ^ (bits-1) + uL) = 2 ^ (bits-1) + (l ^^^ uL) := by
    rw [← two_pow_xor_eq_add (bits-1) uL huL, xor_left_comm,
        two_pow_xor_eq_add (bits-1) (l ^^^ uL) (Nat.xor_lt_two_pow hl huL)]
  -- parse offSeam
  rw [offSeam, Bool.not_or, Bool.and_eq_true] at hoff
  obtain ⟨h1, h2⟩ := hoff
  simp only [Bool.not_eq_true', beq_eq_false_iff_ne] at h1 h2
  have huL0 : uL ≠ 0 := by intro h; apply h1; omega
  have hluL : l ≠ uL := by
    intro h; apply h2; rw [hdval, h, Nat.xor_self, Nat.add_zero]
  -- Q downstairs at level bits-1
  obtain ⟨a, haW, ha0, hadxu, hal, hauL, haP⟩ :=
    Q_ge3 (bits-1) (by omega) l uL hl1 hl (by omega) huL hluL
  -- doubling recursion upstairs
  have hbb1 : bits - 2 + 1 = bits - 1 := by omega
  have hbb2 : bits - 2 + 2 = bits := by omega
  have hrec := converse_recursion' (bits-2) l uL a hl1
    (by rw [hbb1]; exact hl) (by omega) (by rw [hbb1]; exact huL)
    (by omega) (by rw [hbb1]; exact haW) hal hauL hadxu
  rw [hbb1, hbb2] at hrec
  have hprod : fVal l (2 ^ (bits-1) + uL) a bits
      * fVal l (2 ^ (bits-1) + uL) (a ^^^ (l ^^^ (2 ^ (bits-1) + uL))) bits = 1 := by
    rw [hrec, haP]; decide
  -- assemble hasXorAnnih
  have had : a ≠ l ^^^ (2 ^ (bits-1) + uL) := by rw [hdval]; omega
  unfold hasXorAnnih
  refine List.any_eq_true.mpr ⟨a, List.mem_range.mpr (by omega), ?_⟩
  rw [Bool.and_eq_true, Bool.and_eq_true, decide_eq_true_eq, decide_eq_true_eq, beq_iff_eq]
  refine ⟨⟨by omega, had⟩, ?_⟩
  rw [P_eq_fVal]; exact hprod

/-- **Membership in `loHi bits`** pins the exact locus: `1 ≤ l < 2^(bits-1) ≤ u < 2^bits`
    (in particular `l ≥ 1` — the identity `e₀` is excluded). -/
theorem loHi_mem (bits : Nat) (hb : 1 ≤ bits) (p : Nat × Nat) (hp : p ∈ loHi bits) :
    1 ≤ p.1 ∧ p.1 < 2 ^ (bits-1) ∧ 2 ^ (bits-1) ≤ p.2 ∧ p.2 < 2 ^ bits := by
  have htop : 1 ≤ 2 ^ (bits-1) := Nat.two_pow_pos (bits-1)
  have hpow : 2 ^ bits = 2 ^ (bits-1) * 2 := by
    have hps : 2 ^ ((bits-1)+1) = 2 ^ (bits-1) * 2 := Nat.pow_succ 2 (bits-1)
    rw [Nat.sub_add_cancel hb] at hps; exact hps
  simp only [loHi, List.mem_flatMap, List.mem_map, List.mem_range] at hp
  obtain ⟨l, hl, u, hu, rfl⟩ := hp
  refine ⟨?_, ?_, ?_, ?_⟩ <;> (dsimp only; omega)

/-- **THE TOWER-WIDE CONVERSE, DISCHARGED** — `ConverseConjecture` is now a theorem: for every level
    `bits ≥ 4`, every off-seam lower×upper pair is a zero divisor.  Composition of `converse_holds`
    (off-seam ⟹ `hasXorAnnih`, ∀n, proved by the octonion-base induction) with `hasXorAnnih_sound`
    (`hasXorAnnih` ⟹ `isZD`, ∀n).  Kernel axioms `[propext, Quot.sound]` — fully anchor-free (the
    octonion base `Q_base` is now structural, no `native_decide`). -/
theorem converse_conjecture_proved : ConverseConjecture := by
  intro bits hb
  unfold converseHolds
  rw [List.all_eq_true]
  intro p hp
  obtain ⟨hp1, hp2, hp3, hp4⟩ := loHi_mem bits (by omega) p hp
  have hlt : p.1 < 2 ^ bits := by
    have hpow : 2 ^ bits = 2 ^ (bits-1) * 2 := by
      have hps : 2 ^ ((bits-1)+1) = 2 ^ (bits-1) * 2 := Nat.pow_succ 2 (bits-1)
      rw [Nat.sub_add_cancel (show 1 ≤ bits by omega)] at hps; exact hps
    have htop : 1 ≤ 2 ^ (bits-1) := Nat.two_pow_pos (bits-1)
    omega
  by_cases hoff : offSeam bits p.1 p.2 = true
  · have hxa := converse_holds bits p.1 p.2 hb hp1 hp2 hp3 hp4 hoff
    have hne : p.1 ≠ p.2 := by omega
    rw [hasXorAnnih_sound bits p.1 p.2 hlt hp4 hne hxa, Bool.or_true]
  · rw [Bool.not_eq_true] at hoff
    rw [hoff]; rfl

-- ══════════════════════════════════════════════════════════════════════════════════════════════════
-- TARGET 1 (necessity): isZD ⟹ hasXorAnnih on the loHi locus, ∀ bits ≥ 4.
-- ══════════════════════════════════════════════════════════════════════════════════════════════════

/-- **High×pure-seam branch of the CD sign**: `cdSigma (2^(n+1)+uL) (2^(n+1)) = -1`.  The right index is
    the pure seam bit `2^(n+1)` (so `bLo = 0`); the both-high branch takes `- cdSigma 0 uL (n+1) = -1`.
    Definitional + `cdSigma_zero_left`; no cocycle input.  (Cross-checked `#eval`: `-1` for all `uL`.) -/
theorem cdSigma_hi_pow (n uL : Nat) (huL : uL < 2 ^ (n+1)) :
    cdSigma (2 ^ (n+1) + uL) (2 ^ (n+1)) (n+2) = -1 := by
  have hpos : 0 < 2 ^ (n+1) := Nat.two_pow_pos (n+1)
  have hg : ¬ ((2 ^ (n+1) + uL) == 0 || (2 ^ (n+1)) == 0) = true := by
    rw [Bool.or_eq_true]
    rintro (h | h)
    · exact absurd (eq_of_beq h) (by omega)
    · exact absurd (eq_of_beq h) (by omega)
  have hAhi : (2 ^ (n+1) + uL) ≥ 2 ^ (n+1) := by omega
  have hBhi : (2 ^ (n+1)) ≥ 2 ^ (n+1) := by omega
  have hmod1 : (2 ^ (n+1) + uL) % 2 ^ (n+1) = uL := by
    rw [Nat.add_mod_left]; exact Nat.mod_eq_of_lt huL
  have hmod2 : (2 ^ (n+1)) % 2 ^ (n+1) = 0 := Nat.mod_self _
  have hstep : cdSigma (2 ^ (n+1) + uL) (2 ^ (n+1)) (n+2) = - cdSigma 0 uL (n+1) := by
    rw [cdSigma]
    rw [if_neg hg]
    simp only [ge_iff_le, hAhi, hBhi, decide_true, Bool.not_true, Bool.and_false, Bool.false_and,
      Bool.and_self, hmod1, hmod2]
    rfl
  rw [hstep, cdSigma_zero_left (n+1) uL (by omega)]

/-- XOR left-cancellation (Mathlib-free): `c ⊕ x = c ⊕ y → x = y`. -/
theorem xor_cancel_left (c x y : Nat) (h : c ^^^ x = c ^^^ y) : x = y := by
  have h2 : c ^^^ (c ^^^ x) = c ^^^ (c ^^^ y) := by rw [h]
  rwa [← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor, ← Nat.xor_assoc, Nat.xor_self,
    Nat.zero_xor] at h2

/-- XOR right-cancellation (Mathlib-free): `x ⊕ c = y ⊕ c → x = y`. -/
theorem xor_cancel_right (x y c : Nat) (h : x ^^^ c = y ^^^ c) : x = y := by
  have h2 : x ^^^ c ^^^ c = y ^^^ c ^^^ c := by rw [h]
  rwa [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero] at h2

/-- **Reduction extraction (necessity core): every annihilator is XOR-linked with sign product `+1`.**
    The reverse of `annih_of`: if `e_a + s·e_b` annihilates `e_l+e_u` (`annih … = true`), then the
    partner is forced, `b = a ⊕ (l⊕u)`, and the four-sign product is `+1`.  Mathlib-free, no cocycle
    input beyond `cdSigma_pm`. -/
theorem annih_forces (bits l u a b : Nat) (s : Int)
    (hl : l < 2 ^ bits) (hu : u < 2 ^ bits) (ha : a < 2 ^ bits) (hne : l ≠ u) (hab : a ≠ b)
    (hh : annih bits l u a b s = true) :
    b = a ^^^ (l ^^^ u) ∧
    cdSigma l a bits * cdSigma u a bits * cdSigma l b bits * cdSigma u b bits = 1 := by
  unfold annih at hh
  rw [List.all_eq_true] at hh
  have hk : ∀ k, k < 2 ^ bits →
      (if (l ^^^ a) == k then cdSigma l a bits else 0)
      + (if (l ^^^ b) == k then s * cdSigma l b bits else 0)
      + (if (u ^^^ a) == k then cdSigma u a bits else 0)
      + (if (u ^^^ b) == k then s * cdSigma u b bits else 0) = 0 := by
    intro k hkk
    have hkm := hh k (List.mem_range.mpr hkk)
    rwa [beq_iff_eq] at hkm
  -- guards that hold at any k (index distinctness independent of k)
  have hla : l ^^^ a < 2 ^ bits := Nat.xor_lt_two_pow hl ha
  -- ── Step 1: b = a ⊕ (l⊕u) ───────────────────────────────────────────────────────────────────────
  have hb : b = a ^^^ (l ^^^ u) := by
    by_cases hbe : b = a ^^^ (l ^^^ u)
    · exact hbe
    · exfalso
      -- evaluate hk at k = l⊕a; only term1 fires
      have g1 : ((l ^^^ a) == (l ^^^ a)) = true := beq_iff_eq.mpr rfl
      have g2 : ¬ ((l ^^^ b) == (l ^^^ a)) = true :=
        fun hc => hab (xor_cancel_left l b a (beq_iff_eq.mp hc)).symm
      have g3 : ¬ ((u ^^^ a) == (l ^^^ a)) = true :=
        fun hc => hne (xor_cancel_right u l a (beq_iff_eq.mp hc)).symm
      have g4 : ¬ ((u ^^^ b) == (l ^^^ a)) = true := by
        intro hc
        apply hbe
        have hub : u ^^^ b = l ^^^ a := beq_iff_eq.mp hc
        exact xor_cancel_left u b (a ^^^ (l ^^^ u)) (by
          rw [hub, xor_left_comm u a (l ^^^ u), xor_left_comm u l u, Nat.xor_self, Nat.xor_zero,
            Nat.xor_comm a l])
      have he := hk (l ^^^ a) hla
      rw [if_pos g1, if_neg g2, if_neg g3, if_neg g4] at he
      rcases cdSigma_pm bits l a with h1 | h1 <;> rw [h1] at he <;> revert he <;> decide
  -- ── Step 2: sign product = 1 ────────────────────────────────────────────────────────────────────
  refine ⟨hb, ?_⟩
  -- folding: u⊕b = l⊕a and l⊕b = u⊕a
  have hf1 : u ^^^ b = l ^^^ a := by
    rw [hb, xor_left_comm u a (l ^^^ u), xor_left_comm u l u, Nat.xor_self, Nat.xor_zero,
      Nat.xor_comm a l]
  have hf2 : l ^^^ b = u ^^^ a := by
    rw [hb, xor_left_comm l a (l ^^^ u), ← Nat.xor_assoc l l u, Nat.xor_self, Nat.zero_xor,
      Nat.xor_comm a u]
  have hlb : l ^^^ b < 2 ^ bits := by rw [hf2]; exact Nat.xor_lt_two_pow hu ha
  -- guards at k1 = l⊕a
  have g1a : ((l ^^^ a) == (l ^^^ a)) = true := beq_iff_eq.mpr rfl
  have g2a : ¬ ((l ^^^ b) == (l ^^^ a)) = true :=
    fun hc => hab (xor_cancel_left l b a (beq_iff_eq.mp hc)).symm
  have g3a : ¬ ((u ^^^ a) == (l ^^^ a)) = true :=
    fun hc => hne (xor_cancel_right u l a (beq_iff_eq.mp hc)).symm
  have g4a : ((u ^^^ b) == (l ^^^ a)) = true := beq_iff_eq.mpr hf1
  have e1 := hk (l ^^^ a) hla
  rw [if_pos g1a, if_neg g2a, if_neg g3a, if_pos g4a] at e1
  -- e1 : cdSigma l a + 0 + 0 + s*cdSigma u b = 0
  -- guards at k2 = l⊕b
  have g1b : ¬ ((l ^^^ a) == (l ^^^ b)) = true :=
    fun hc => hab (xor_cancel_left l a b (beq_iff_eq.mp hc))
  have g2b : ((l ^^^ b) == (l ^^^ b)) = true := beq_iff_eq.mpr rfl
  have g3b : ((u ^^^ a) == (l ^^^ b)) = true := beq_iff_eq.mpr hf2.symm
  have g4b : ¬ ((u ^^^ b) == (l ^^^ b)) = true := by
    intro hc
    have hcc : u ^^^ b = l ^^^ b := beq_iff_eq.mp hc
    rw [hf1] at hcc
    exact hab (xor_cancel_left l a b hcc)
  have e2 := hk (l ^^^ b) hlb
  rw [if_neg g1b, if_pos g2b, if_pos g3b, if_neg g4b] at e2
  -- e2 : 0 + s*cdSigma l b + cdSigma u a + 0 = 0
  -- solve: s = ±1 then product = s² = 1
  have hs : s = 1 ∨ s = -1 := by
    rcases cdSigma_pm bits l a with h1 | h1 <;> rcases cdSigma_pm bits u b with h4 | h4 <;>
      rw [h1, h4] at e1 <;> omega
  rcases hs with hs | hs <;>
    rcases cdSigma_pm bits l a with h1 | h1 <;> rcases cdSigma_pm bits u a with h2 | h2 <;>
    rcases cdSigma_pm bits l b with h3 | h3 <;> rcases cdSigma_pm bits u b with h4 | h4 <;>
    simp only [hs, h1, h2, h3, h4] at e1 e2 ⊢ <;> revert e1 e2 <;> decide

/-- **`P(0) = −1` for EVERY distinct nonzero pair** (`1≤l,u<2^bits`, `l≠u`) — the general form of the
    `a=0` corner, with no loHi/on-seam hypotheses.  The `{0,d}` orbit *always* disagrees: the four-sign
    winner product at `a=0` is `−1`, so `a=0` can never be an annihilator's low index.  Proof is a pure
    σ-identity: `cdSigma _ 0 = 1` (twice), then the two cocycles `σ(l,d)·σ(l,u)=−1`, `σ(u,d)·σ(u,l)=−1`
    (`d=l⊕u`, so `l⊕d=u`, `u⊕d=l`) combine with `σ(l,u)·σ(u,l)=−1` (`cdSigma_cross_neg`) to force
    `σ(l,d)·σ(u,d)=−1`.  No loHi, no `native_decide`.  This is what makes the `isZD ⟹ hasXorAnnih`
    necessity extend to the *full* distinct-nonzero box (not just loHi): the `a=0` branch is vacuous. -/
theorem P0_neg_general (bits l u : Nat)
    (hl1 : 1 ≤ l) (hl : l < 2 ^ bits) (hu1 : 1 ≤ u) (hu : u < 2 ^ bits) (hne : l ≠ u) :
    cdSigma l 0 bits * cdSigma u 0 bits
      * cdSigma l (l ^^^ u) bits * cdSigma u (l ^^^ u) bits = -1 := by
  have hbits : 1 ≤ bits := by
    cases bits with
    | zero => rw [Nat.pow_zero] at hl; omega
    | succ _ => omega
  have hd : l ^^^ u < 2 ^ bits := Nat.xor_lt_two_pow hl hu
  have hlxd : l ^^^ (l ^^^ u) = u := by rw [← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]
  have huxd : u ^^^ (l ^^^ u) = l := by
    rw [Nat.xor_comm l u, ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]
  have hc1 := cdSigma_cocycle' bits l (l ^^^ u) hl hd (by omega)
  rw [hlxd] at hc1
  have hc2 := cdSigma_cocycle' bits u (l ^^^ u) hu hd (by omega)
  rw [huxd] at hc2
  have hcn := cdSigma_cross_neg bits l u hl1 hl hu1 hu hne
  rw [cdSigma_zero_right bits l (by omega), cdSigma_zero_right bits u (by omega)]
  rcases cdSigma_pm bits l (l ^^^ u) with h1 | h1 <;>
  rcases cdSigma_pm bits u (l ^^^ u) with h2 | h2 <;>
  rcases cdSigma_pm bits l u with h3 | h3 <;>
  rcases cdSigma_pm bits u l with h4 | h4 <;>
    rw [h1] at hc1 <;> rw [h2] at hc2 <;> rw [h3] at hc1 hcn <;> rw [h4] at hc2 hcn <;>
    rw [h1, h2] <;>
    first | decide | exact absurd hc1 (by decide) | exact absurd hc2 (by decide)
          | exact absurd hcn (by decide)

/-- **On-seam ⟹ `P(0) = −1`** (the `a = 0` corner).  For an on-seam loHi pair (`offSeam = false`),
    the four-sign winner product at `a = 0` is `−1` — so `a = 0` is NOT a `hasXorAnnih` winner.  Two
    subcases (`u = top` / `l⊕u = top`), each computed from the branch lemmas; no cocycle input beyond
    `cdSigma_diag`.  (Per-case σ values `#eval`-confirmed: `u=top→(−1,1)`, `d=top→(1,−1)`.) -/
theorem P0_neg_of_onSeam (bits l u : Nat) (hb : 2 ≤ bits)
    (hl1 : 1 ≤ l) (hl : l < 2 ^ (bits-1)) (hu1 : 2 ^ (bits-1) ≤ u) (hu : u < 2 ^ bits)
    (hon : offSeam bits l u = false) :
    cdSigma l 0 bits * cdSigma u 0 bits
      * cdSigma l (l ^^^ u) bits * cdSigma u (l ^^^ u) bits = -1 := by
  have hbb1 : bits - 2 + 1 = bits - 1 := by omega
  have hbb2 : bits - 2 + 2 = bits := by omega
  rw [cdSigma_zero_right bits l (by omega), cdSigma_zero_right bits u (by omega)]
  have hon' : u = 2 ^ (bits-1) ∨ l ^^^ u = 2 ^ (bits-1) := by
    simp only [offSeam] at hon
    have hcond : (u == 2 ^ (bits-1) || l ^^^ u == 2 ^ (bits-1)) = true := by
      cases hx : (u == 2 ^ (bits-1) || l ^^^ u == 2 ^ (bits-1)) with
      | true => rfl
      | false => rw [hx] at hon; exact absurd hon (by decide)
    rw [Bool.or_eq_true, beq_iff_eq, beq_iff_eq] at hcond
    exact hcond
  rcases hon' with hutop | hdtop
  · -- u = top; d = l⊕u = top + l ;  σ_l d = -1 (diag), σ_u d = 1 (hi_hi to zero_right)
    subst hutop
    have hd : l ^^^ 2 ^ (bits-1) = 2 ^ (bits-1) + l := by
      rw [Nat.xor_comm l (2 ^ (bits-1)), two_pow_xor_eq_add (bits-1) l hl]
    have e_ld : cdSigma l (2 ^ (bits-1) + l) bits = cdSigma l l (bits-1) := by
      have h := cdSigma_lo_hi (bits-2) l l (by rw [hbb1]; exact hl) hl1 (by rw [hbb1]; exact hl)
      rw [hbb1, hbb2] at h; exact h
    have e_ud : cdSigma (2 ^ (bits-1)) (2 ^ (bits-1) + l) bits = cdSigma l 0 (bits-1) := by
      have h := cdSigma_hi_hi (bits-2) 0 l (Nat.two_pow_pos _) hl1 (by rw [hbb1]; exact hl)
      rw [hbb1, hbb2, Nat.add_zero] at h; exact h
    rw [hd, e_ld, cdSigma_diag (bits-1) l hl1 hl, e_ud, cdSigma_zero_right (bits-1) l (by omega)]
    decide
  · -- l⊕u = top; u = top + l ;  σ_l d = 1 (lo_hi to zero_left), σ_u d = -1 (hi_pow)
    have h1 : u = l ^^^ 2 ^ (bits-1) := by
      rw [← hdtop, ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]
    have hu_eq : u = 2 ^ (bits-1) + l := by
      rw [h1, Nat.xor_comm l (2 ^ (bits-1)), two_pow_xor_eq_add (bits-1) l hl]
    subst hu_eq
    have e_ld : cdSigma l (2 ^ (bits-1)) bits = cdSigma 0 l (bits-1) := by
      have h := cdSigma_lo_hi (bits-2) 0 l (Nat.two_pow_pos _) hl1 (by rw [hbb1]; exact hl)
      rw [hbb1, hbb2, Nat.add_zero] at h; exact h
    have e_ud : cdSigma (2 ^ (bits-1) + l) (2 ^ (bits-1)) bits = -1 := by
      have h := cdSigma_hi_pow (bits-2) l (by rw [hbb1]; exact hl)
      rw [hbb1, hbb2] at h; exact h
    rw [hdtop, e_ld, cdSigma_zero_left (bits-1) l (by omega), e_ud]
    decide

/-- **NECESSITY (Target 1): `isZD ⟹ hasXorAnnih` on the loHi locus, ∀ bits ≥ 4.**  The sharp O(N)
    predicate captures *every* zero divisor, not just the XOR-linked witnesses: any `isZD` certificate,
    via `annih_forces`, is XOR-linked with sign product `+1`.  The `a ≥ 1` witness injects directly;
    the `a = 0` corner cannot be a winner on-seam (`P0_neg_of_onSeam`), so the pair is off-seam and the
    already-proved `converse_holds` supplies a genuine winner. -/
theorem hasXorAnnih_complete (bits l u : Nat) (hb : 4 ≤ bits)
    (hl1 : 1 ≤ l) (hl : l < 2 ^ (bits-1)) (hu1 : 2 ^ (bits-1) ≤ u) (hu : u < 2 ^ bits)
    (hzd : isZD bits l u = true) : hasXorAnnih bits l u = true := by
  have hle : 2 ^ (bits-1) ≤ 2 ^ bits := Nat.pow_le_pow_right (by decide) (by omega)
  have hlt : l < 2 ^ bits := by omega
  have hne : l ≠ u := by omega
  simp only [isZD] at hzd
  rw [List.any_eq_true] at hzd
  obtain ⟨a, ha_mem, hzd⟩ := hzd
  rw [List.any_eq_true] at hzd
  obtain ⟨b, hb_mem, hzd⟩ := hzd
  rw [Bool.and_eq_true, decide_eq_true_eq, Bool.or_eq_true] at hzd
  obtain ⟨hab_lt, hann⟩ := hzd
  have hltA : a < 2 ^ bits := List.mem_range.mp ha_mem
  obtain ⟨s, hannih⟩ : ∃ s : Int, annih bits l u a b s = true := by
    rcases hann with h | h
    · exact ⟨1, h⟩
    · exact ⟨-1, h⟩
  obtain ⟨hbe, hP⟩ :=
    annih_forces bits l u a b s hlt hu hltA hne (Nat.ne_of_lt hab_lt) hannih
  rcases Nat.eq_zero_or_pos a with ha0 | hapos
  · -- a = 0: P(0) = 1 forces off-seam, then converse_holds gives the winner
    subst ha0
    rw [hbe, Nat.zero_xor] at hP
    have hoff : offSeam bits l u = true := by
      cases ho : offSeam bits l u with
      | true => rfl
      | false =>
        have hneg := P0_neg_of_onSeam bits l u (by omega) hl1 hl hu1 hu ho
        rw [hP] at hneg
        exact absurd hneg (by decide)
    exact converse_holds bits l u hb hl1 hl hu1 hu hoff
  · -- a ≥ 1: inject a directly as the XOR-linked winner
    have hane : a ≠ l ^^^ u := by
      intro h; rw [h, Nat.xor_self] at hbe; omega
    rw [hbe] at hP
    unfold hasXorAnnih
    refine List.any_eq_true.mpr ⟨a, List.mem_range.mpr hltA, ?_⟩
    rw [Bool.and_eq_true, Bool.and_eq_true, decide_eq_true_eq, decide_eq_true_eq, beq_iff_eq]
    exact ⟨⟨hapos, hane⟩, hP⟩

/-- **`hasXorAnnih == isZD` on the loHi locus, ∀ bits ≥ 4** — the ∀n generalization of
    `xorAnnih_eq_isZD_16`.  Soundness (`⟹`) by `hasXorAnnih_sound`, completeness (`⟸`) by
    `hasXorAnnih_complete`.  Kernel axioms `[propext, Quot.sound]` — anchor-free. -/
theorem xorAnnih_eq_isZD_all (bits : Nat) (hb : 4 ≤ bits) :
    (loHi bits).all (fun p => hasXorAnnih bits p.1 p.2 == isZD bits p.1 p.2) = true := by
  rw [List.all_eq_true]
  intro p hp
  obtain ⟨hp1, hp2, hp3, hp4⟩ := loHi_mem bits (by omega) p hp
  have hle : 2 ^ (bits-1) ≤ 2 ^ bits := Nat.pow_le_pow_right (by decide) (by omega)
  have hlt : p.1 < 2 ^ bits := by omega
  have hne : p.1 ≠ p.2 := by omega
  rw [beq_iff_eq]
  cases hI : isZD bits p.1 p.2 with
  | true => exact hasXorAnnih_complete bits p.1 p.2 hb hp1 hp2 hp3 hp4 hI
  | false =>
    cases hX : hasXorAnnih bits p.1 p.2 with
    | false => rfl
    | true =>
      have hsd := hasXorAnnih_sound bits p.1 p.2 hlt hp4 hne hX
      rw [hI] at hsd; exact absurd hsd (by decide)

-- ══════════════════════════════════════════════════════════════════════════════════════════════════
-- WIDEN PAST loHi: `isZD = hasXorAnnih` on the FULL distinct-nonzero box, ∀ bits (any `1≤l,u<2^bits`,
--   `l≠u` — NOT just lower×upper).  The geometric `offSeam` predicate is loHi-specific and does NOT
--   characterize `isZD` off the locus (e.g. `e₁+e₂` in the sedenions is off-seam by that test but not a
--   ZD); the correct all-pairs characterization is the `hasXorAnnih` one.  The necessity `isZD ⟹
--   hasXorAnnih` extends here for free: `annih_forces` is already general, and the `a=0` corner (the only
--   loHi-dependent step) is VACUOUS everywhere — `P0_neg_general` gives `P(0)=−1` for every pair.
-- ══════════════════════════════════════════════════════════════════════════════════════════════════

/-- **Necessity `isZD ⟹ hasXorAnnih` on the full box, ∀ bits.**  Every 2-term zero divisor is
    XOR-linked with sign product `+1`, for *any* distinct nonzero pair (not just loHi).  Same argument
    as `hasXorAnnih_complete` for the `a≥1` witness (which was never loHi-specific); the `a=0` corner is
    now dispatched by `P0_neg_general` (`P(0)=−1` always) rather than the off-seam detour. -/
theorem hasXorAnnih_complete_full (bits l u : Nat)
    (hl1 : 1 ≤ l) (hl : l < 2 ^ bits) (hu1 : 1 ≤ u) (hu : u < 2 ^ bits) (hne : l ≠ u)
    (hzd : isZD bits l u = true) : hasXorAnnih bits l u = true := by
  simp only [isZD] at hzd
  rw [List.any_eq_true] at hzd
  obtain ⟨a, ha_mem, hzd⟩ := hzd
  rw [List.any_eq_true] at hzd
  obtain ⟨b, hb_mem, hzd⟩ := hzd
  rw [Bool.and_eq_true, decide_eq_true_eq, Bool.or_eq_true] at hzd
  obtain ⟨hab_lt, hann⟩ := hzd
  have hltA : a < 2 ^ bits := List.mem_range.mp ha_mem
  obtain ⟨s, hannih⟩ : ∃ s : Int, annih bits l u a b s = true := by
    rcases hann with h | h
    · exact ⟨1, h⟩
    · exact ⟨-1, h⟩
  obtain ⟨hbe, hP⟩ :=
    annih_forces bits l u a b s hl hu hltA hne (Nat.ne_of_lt hab_lt) hannih
  rcases Nat.eq_zero_or_pos a with ha0 | hapos
  · -- a = 0 is VACUOUS: annih_forces gives P(0)=+1, but P0_neg_general says P(0)=−1.
    subst ha0
    rw [hbe, Nat.zero_xor] at hP
    have hneg := P0_neg_general bits l u hl1 hl hu1 hu hne
    rw [hP] at hneg
    exact absurd hneg (by decide)
  · -- a ≥ 1: inject a directly as the XOR-linked winner
    have hane : a ≠ l ^^^ u := by
      intro h; rw [h, Nat.xor_self] at hbe; omega
    rw [hbe] at hP
    unfold hasXorAnnih
    refine List.any_eq_true.mpr ⟨a, List.mem_range.mpr hltA, ?_⟩
    rw [Bool.and_eq_true, Bool.and_eq_true, decide_eq_true_eq, decide_eq_true_eq, beq_iff_eq]
    exact ⟨⟨hapos, hane⟩, hP⟩

/-- **The zero-divisor characterization on the FULL box, ∀ bits** — `isZD = hasXorAnnih` for *every*
    distinct nonzero basis pair `(l,u)` (`1≤l,u<2^bits`, `l≠u`), not just lower×upper.  Soundness (`⟸`)
    by `hasXorAnnih_sound`, necessity (`⟹`) by `hasXorAnnih_complete_full`.  So `e_l+e_u` is a 2-term
    zero divisor **iff** it has a non-exceptional XOR-linked agreeing orbit — the honest all-pairs
    widening of `xorAnnih_eq_isZD_all` (which was confined to loHi).  Axioms `[propext, Quot.sound]`. -/
theorem isZD_eq_hasXorAnnih_full (bits l u : Nat)
    (hl1 : 1 ≤ l) (hl : l < 2 ^ bits) (hu1 : 1 ≤ u) (hu : u < 2 ^ bits) (hne : l ≠ u) :
    isZD bits l u = hasXorAnnih bits l u := by
  cases hI : isZD bits l u with
  | true => exact (hasXorAnnih_complete_full bits l u hl1 hl hu1 hu hne hI).symm
  | false =>
    cases hX : hasXorAnnih bits l u with
    | false => rfl
    | true =>
      have hsd := hasXorAnnih_sound bits l u hl hu hne hX
      rw [hI] at hsd; exact absurd hsd (by decide)

/-- **Full-box regression (decided, dim 16):** `isZD == hasXorAnnih` for *every* distinct nonzero pair
    in `A_4` — a concrete cross-check of `isZD_eq_hasXorAnnih_full` over the whole box (both-low,
    both-high, and mixed), complementing the loHi-only `xorAnnih_eq_isZD_16`.  `native_decide`, no
    `sorry`; a fixed-n anchor, outside the ∀n chain. -/
theorem isZD_eq_hasXorAnnih_box_16 :
    ((List.range 16).all (fun l => (List.range 16).all (fun u =>
      (l == 0) || (u == 0) || (l == u) || (isZD 4 l u == hasXorAnnih 4 l u)))) = true := by
  native_decide

-- ══════════════════════════════════════════════════════════════════════════════════════════════════
-- TARGET 2 (seam coincidence): anti0 == ! offSeam on the loHi locus, ∀ bits ≥ 4.
-- The bridge `Q(c) = P(c)`: the anti0 four-sign product equals the converse winner product, so anti0
-- reduces to `∀c P(c) = -1`, closed off-seam by `converse_holds` and on-seam by the edge lemmas.
-- ══════════════════════════════════════════════════════════════════════════════════════════════════

/-- XOR cross-cancellation (Mathlib-free): `(a⊕b)⊕(a⊕c) = b⊕c`. -/
theorem xor_cross_cancel (a b c : Nat) : (a ^^^ b) ^^^ (a ^^^ c) = b ^^^ c := by
  rw [Nat.xor_assoc, xor_left_comm b a c, ← Nat.xor_assoc a a (b ^^^ c), Nat.xor_self, Nat.zero_xor]

/-- **THE UNLOCK — `Q(c) = P(c)`.**  The anti0 four-sign product `σ(l,u⊕c)·σ(u,c)·σ(u,l⊕c)·σ(l,c)`
    equals the converse winner product `σ(l,c)·σ(u,c)·σ(l,c⊕d)·σ(u,c⊕d)` (`d = l⊕u`), for `l,u ≠ 0`.
    Two cocycle rewrites (`σ(l,u⊕c) = -σ(l,c⊕d)`, `σ(u,l⊕c) = -σ(u,c⊕d)`); the two `-1`s cancel. -/
theorem anti0_QP (bits l u c : Nat) (hl0 : l ≠ 0) (hu0 : u ≠ 0)
    (hl : l < 2 ^ bits) (hu : u < 2 ^ bits) (hc : c < 2 ^ bits) :
    cdSigma l (u ^^^ c) bits * cdSigma u c bits * cdSigma u (l ^^^ c) bits * cdSigma l c bits
      = cdSigma l c bits * cdSigma u c bits
          * cdSigma l (c ^^^ (l ^^^ u)) bits * cdSigma u (c ^^^ (l ^^^ u)) bits := by
  have hd : l ^^^ u < 2 ^ bits := Nat.xor_lt_two_pow hl hu
  have hx : c ^^^ (l ^^^ u) < 2 ^ bits := Nat.xor_lt_two_pow hc hd
  have hlx : l ^^^ (c ^^^ (l ^^^ u)) = u ^^^ c := by
    rw [xor_left_comm l c (l ^^^ u), ← Nat.xor_assoc l l u, Nat.xor_self, Nat.zero_xor,
      Nat.xor_comm c u]
  have hux : u ^^^ (c ^^^ (l ^^^ u)) = l ^^^ c := by
    rw [xor_left_comm u c (l ^^^ u), xor_left_comm u l u, Nat.xor_self, Nat.xor_zero,
      Nat.xor_comm c l]
  have hcoc_l := cdSigma_cocycle' bits l (c ^^^ (l ^^^ u)) hl hx hl0
  rw [hlx] at hcoc_l
  have hcoc_u := cdSigma_cocycle' bits u (c ^^^ (l ^^^ u)) hu hx hu0
  rw [hux] at hcoc_u
  -- σ(l,c⊕d) = -σ(l,u⊕c) and σ(u,c⊕d) = -σ(u,l⊕c)
  have e5 : cdSigma l (c ^^^ (l ^^^ u)) bits = - cdSigma l (u ^^^ c) bits := by
    rcases cdSigma_pm bits l (c ^^^ (l ^^^ u)) with h|h <;>
    rcases cdSigma_pm bits l (u ^^^ c) with h'|h' <;>
      rw [h, h'] at hcoc_l ⊢ <;> first | decide | exact absurd hcoc_l (by decide)
  have e6 : cdSigma u (c ^^^ (l ^^^ u)) bits = - cdSigma u (l ^^^ c) bits := by
    rcases cdSigma_pm bits u (c ^^^ (l ^^^ u)) with h|h <;>
    rcases cdSigma_pm bits u (l ^^^ c) with h'|h' <;>
      rw [h, h'] at hcoc_u ⊢ <;> first | decide | exact absurd hcoc_u (by decide)
  rw [e5, e6]
  rcases cdSigma_pm bits l (u ^^^ c) with h1|h1 <;>
  rcases cdSigma_pm bits u c with h2|h2 <;>
  rcases cdSigma_pm bits u (l ^^^ c) with h3|h3 <;>
  rcases cdSigma_pm bits l c with h4|h4 <;>
    rw [h1, h2, h3, h4] <;> decide

/-- Per-`c` bridge: the anti0 vanishing condition `A(c)+B(c) = 0` is equivalent to `P(c) = -1`. -/
theorem anti0_term_iff (bits l u c : Nat) (hl0 : l ≠ 0) (hu0 : u ≠ 0)
    (hl : l < 2 ^ bits) (hu : u < 2 ^ bits) (hc : c < 2 ^ bits) :
    (cdSigma l (u ^^^ c) bits * cdSigma u c bits + cdSigma u (l ^^^ c) bits * cdSigma l c bits = 0)
      ↔ (cdSigma l c bits * cdSigma u c bits
          * cdSigma l (c ^^^ (l ^^^ u)) bits * cdSigma u (c ^^^ (l ^^^ u)) bits = -1) := by
  rw [← anti0_QP bits l u c hl0 hu0 hl hu hc]
  rcases cdSigma_pm bits l (u ^^^ c) with h1|h1 <;>
  rcases cdSigma_pm bits u c with h2|h2 <;>
  rcases cdSigma_pm bits u (l ^^^ c) with h3|h3 <;>
  rcases cdSigma_pm bits l c with h4|h4 <;>
    rw [h1, h2, h3, h4] <;> decide

/-- **Bridge lemma:** `anti0 = true ↔ ∀ c < 2^bits, P(c) = -1` (the converse winner product). -/
theorem anti0_iff (bits l u : Nat) (hl0 : l ≠ 0) (hu0 : u ≠ 0)
    (hl : l < 2 ^ bits) (hu : u < 2 ^ bits) :
    anti0 bits l u = true ↔ ∀ c, c < 2 ^ bits →
      cdSigma l c bits * cdSigma u c bits
        * cdSigma l (c ^^^ (l ^^^ u)) bits * cdSigma u (c ^^^ (l ^^^ u)) bits = -1 := by
  simp only [anti0]
  rw [List.all_eq_true]
  constructor
  · intro h c hc
    have hcc := h c (List.mem_range.mpr hc)
    rw [beq_iff_eq] at hcc
    exact (anti0_term_iff bits l u c hl0 hu0 hl hu hc).mp hcc
  · intro h c hc
    rw [List.mem_range] at hc
    rw [beq_iff_eq]
    exact (anti0_term_iff bits l u c hl0 hu0 hl hu hc).mpr (h c hc)

/-- **Orbit reduction of the winner product:** `P(c) = P(r)` whenever `c ⊕ d = r` (`d = l⊕u`).  The
    involution `c ↦ c⊕d` fixes the four-sign product, so every orbit is witnessed by either member. -/
theorem P_eq_of_xor (bits l u c r : Nat) (h1 : c ^^^ (l ^^^ u) = r) :
    cdSigma l c bits * cdSigma u c bits
        * cdSigma l (c ^^^ (l ^^^ u)) bits * cdSigma u (c ^^^ (l ^^^ u)) bits
      = cdSigma l r bits * cdSigma u r bits
          * cdSigma l (r ^^^ (l ^^^ u)) bits * cdSigma u (r ^^^ (l ^^^ u)) bits := by
  have h2 : r ^^^ (l ^^^ u) = c := by rw [← h1, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
  rw [h1, h2]
  rcases cdSigma_pm bits l c with a1|a1 <;> rcases cdSigma_pm bits u c with a2|a2 <;>
  rcases cdSigma_pm bits l r with a3|a3 <;> rcases cdSigma_pm bits u r with a4|a4 <;>
    rw [a1, a2, a3, a4] <;> decide

/-- **On-seam corner `P(l) = -1`** for the `u = top` pair.  The orbit `{l, top}` has low rep `l`,
    excluded from `edge_m_eq_H`; here `σ(l,l)σ(top,l)σ(l,top)σ(top,top) = (-1)(-1)(1)(-1) = -1`. -/
theorem P_corner_l (bits l : Nat) (hb : 2 ≤ bits) (hl1 : 1 ≤ l) (hl : l < 2 ^ (bits-1)) :
    cdSigma l l bits * cdSigma (2 ^ (bits-1)) l bits
        * cdSigma l (l ^^^ (l ^^^ 2 ^ (bits-1))) bits
        * cdSigma (2 ^ (bits-1)) (l ^^^ (l ^^^ 2 ^ (bits-1))) bits = -1 := by
  have hbb1 : bits - 2 + 1 = bits - 1 := by omega
  have hbb2 : bits - 2 + 2 = bits := by omega
  have htop1 : 1 ≤ 2 ^ (bits-1) := Nat.two_pow_pos _
  have hpow : 2 ^ bits = 2 ^ (bits-1) * 2 := by
    have hps : 2 ^ ((bits-1)+1) = 2 ^ (bits-1) * 2 := Nat.pow_succ 2 (bits-1)
    rw [Nat.sub_add_cancel (show 1 ≤ bits by omega)] at hps; exact hps
  have hlbits : l < 2 ^ bits := by omega
  have htopbits : 2 ^ (bits-1) < 2 ^ bits := by omega
  have hidx : l ^^^ (l ^^^ 2 ^ (bits-1)) = 2 ^ (bits-1) := by
    rw [← Nat.xor_assoc l l (2 ^ (bits-1)), Nat.xor_self, Nat.zero_xor]
  rw [hidx]
  have e2 : cdSigma (2 ^ (bits-1)) l bits = -1 := by
    have h := cdSigma_hi_lo (bits-2) 0 l (Nat.two_pow_pos _) hl1 (by rw [hbb1]; exact hl)
    rw [hbb1, hbb2, Nat.add_zero, cdSigma_zero_left (bits-1) l (by omega)] at h
    exact h
  have e3 : cdSigma l (2 ^ (bits-1)) bits = 1 := by
    have h := cdSigma_lo_hi (bits-2) 0 l (Nat.two_pow_pos _) hl1 (by rw [hbb1]; exact hl)
    rw [hbb1, hbb2, Nat.add_zero, cdSigma_zero_left (bits-1) l (by omega)] at h
    exact h
  rw [cdSigma_diag bits l hl1 hlbits, e2, e3, cdSigma_diag bits (2 ^ (bits-1)) htop1 htopbits]
  decide

/-- Both indices above the seam ⟹ their XOR falls below it: `2^k ≤ a,d < 2^k+2^k → a⊕d < 2^k`. -/
theorem xor_high_high_lt (k a d : Nat) (hak : 2 ^ k ≤ a) (hal : a < 2 ^ k + 2 ^ k)
    (hdk : 2 ^ k ≤ d) (hdl : d < 2 ^ k + 2 ^ k) : a ^^^ d < 2 ^ k := by
  obtain ⟨aLo, haLo, rfl⟩ : ∃ x, x < 2 ^ k ∧ a = 2 ^ k + x := ⟨a - 2 ^ k, by omega, by omega⟩
  obtain ⟨dLo, hdLo, rfl⟩ : ∃ x, x < 2 ^ k ∧ d = 2 ^ k + x := ⟨d - 2 ^ k, by omega, by omega⟩
  rw [← two_pow_xor_eq_add k aLo haLo, ← two_pow_xor_eq_add k dLo hdLo, xor_cross_cancel]
  exact Nat.xor_lt_two_pow haLo hdLo

/-- **On-seam ⟹ `P(r) = -1` for every low index `r < top`.**  The three low reps split into the
    corners `r=0` (`P0_neg_of_onSeam`), `r=l` in the `u=top` case (`P_corner_l`), and the generic
    edge (`edge_m_eq_H` / `edge_m_eq_H_plus_l`).  No `native_decide`. -/
theorem P_low_neg (bits l u : Nat) (hb : 4 ≤ bits)
    (hl1 : 1 ≤ l) (hl : l < 2 ^ (bits-1)) (hu1 : 2 ^ (bits-1) ≤ u) (hu : u < 2 ^ bits)
    (hon : offSeam bits l u = false) (r : Nat) (hr : r < 2 ^ (bits-1)) :
    cdSigma l r bits * cdSigma u r bits
        * cdSigma l (r ^^^ (l ^^^ u)) bits * cdSigma u (r ^^^ (l ^^^ u)) bits = -1 := by
  have hk1 : bits - 1 + 1 = bits := by omega
  have hon' : u = 2 ^ (bits-1) ∨ l ^^^ u = 2 ^ (bits-1) := by
    simp only [offSeam] at hon
    have hcond : (u == 2 ^ (bits-1) || l ^^^ u == 2 ^ (bits-1)) = true := by
      cases hx : (u == 2 ^ (bits-1) || l ^^^ u == 2 ^ (bits-1)) with
      | true => rfl
      | false => rw [hx] at hon; exact absurd hon (by decide)
    rw [Bool.or_eq_true, beq_iff_eq, beq_iff_eq] at hcond
    exact hcond
  by_cases hr0 : r = 0
  · subst hr0
    rw [Nat.zero_xor]
    exact P0_neg_of_onSeam bits l u (by omega) hl1 hl hu1 hu hon
  · have hr1 : 1 ≤ r := by omega
    rcases hon' with hutop | hdtop
    · subst hutop
      by_cases hrl : r = l
      · rw [hrl]
        exact P_corner_l bits l (by omega) hl1 hl
      · rw [P_eq_fVal l (2 ^ (bits-1)) r bits]
        have H := edge_m_eq_H (bits-1) l r hl1 hl hr1 hr hrl
        rw [hk1] at H
        exact H
    · have h1 : u = l ^^^ 2 ^ (bits-1) := by
        rw [← hdtop, ← Nat.xor_assoc, Nat.xor_self, Nat.zero_xor]
      have hu_eq : u = 2 ^ (bits-1) + l := by
        rw [h1, Nat.xor_comm l (2 ^ (bits-1)), two_pow_xor_eq_add (bits-1) l hl]
      subst hu_eq
      rw [P_eq_fVal l (2 ^ (bits-1) + l) r bits]
      have H := edge_m_eq_H_plus_l (bits-1) l r hl1 hl hr1 hr
      rw [hk1] at H
      exact H

/-- **On-seam ⟹ `∀c P(c) = -1`.**  Every `c` reduces (via the orbit involution `P_eq_of_xor`) to a low
    rep `< top`: if `c` is already low use it; otherwise `c` and `d=l⊕u` both sit above the seam so
    `c⊕d < top` (`xor_high_high_lt`).  Then `P_low_neg`. -/
theorem on_seam_P_neg (bits l u : Nat) (hb : 4 ≤ bits)
    (hl1 : 1 ≤ l) (hl : l < 2 ^ (bits-1)) (hu1 : 2 ^ (bits-1) ≤ u) (hu : u < 2 ^ bits)
    (hon : offSeam bits l u = false) (c : Nat) (hc : c < 2 ^ bits) :
    cdSigma l c bits * cdSigma u c bits
        * cdSigma l (c ^^^ (l ^^^ u)) bits * cdSigma u (c ^^^ (l ^^^ u)) bits = -1 := by
  have htop1 : 1 ≤ 2 ^ (bits-1) := Nat.two_pow_pos _
  have hpow : 2 ^ bits = 2 ^ (bits-1) * 2 := by
    have hps : 2 ^ ((bits-1)+1) = 2 ^ (bits-1) * 2 := Nat.pow_succ 2 (bits-1)
    rw [Nat.sub_add_cancel (show 1 ≤ bits by omega)] at hps; exact hps
  have hle : 2 ^ (bits-1) ≤ 2 ^ bits := by omega
  have hlt : l < 2 ^ bits := by omega
  have hon' : u = 2 ^ (bits-1) ∨ l ^^^ u = 2 ^ (bits-1) := by
    simp only [offSeam] at hon
    have hcond : (u == 2 ^ (bits-1) || l ^^^ u == 2 ^ (bits-1)) = true := by
      cases hx : (u == 2 ^ (bits-1) || l ^^^ u == 2 ^ (bits-1)) with
      | true => rfl
      | false => rw [hx] at hon; exact absurd hon (by decide)
    rw [Bool.or_eq_true, beq_iff_eq, beq_iff_eq] at hcond
    exact hcond
  have hdlt : l ^^^ u < 2 ^ bits := Nat.xor_lt_two_pow hlt hu
  have hdhi : 2 ^ (bits-1) ≤ l ^^^ u := by
    rcases hon' with h|h
    · rw [h, Nat.xor_comm l (2 ^ (bits-1)), two_pow_xor_eq_add (bits-1) l hl]; omega
    · exact Nat.le_of_eq h.symm
  by_cases hlow : c < 2 ^ (bits-1)
  · exact P_low_neg bits l u hb hl1 hl hu1 hu hon c hlow
  · have hchi : 2 ^ (bits-1) ≤ c := by omega
    have hr_lt : c ^^^ (l ^^^ u) < 2 ^ (bits-1) :=
      xor_high_high_lt (bits-1) c (l ^^^ u) hchi (by omega) hdhi (by omega)
    rw [P_eq_of_xor bits l u c (c ^^^ (l ^^^ u)) rfl]
    exact P_low_neg bits l u hb hl1 hl hu1 hu hon (c ^^^ (l ^^^ u)) hr_lt

/-- **TARGET 2 — `anti0 = ! offSeam` on the loHi locus, ∀ bits ≥ 4.**  The seam coincidence for the
    operator/anti-commutator member: `{L_l,L_u}=0 ⟺ NOT off-seam`.  Off-seam ⟹ the converse winner
    exists (`converse_holds`), so some `P(a)=+1` breaks the anti0 test (`anti0=false`); on-seam ⟹
    `∀c P(c)=-1` (`on_seam_P_neg`), so anti0 holds.  Via the `Q=P` bridge (`anti0_iff`).  Kernel axioms
    `[propext, Quot.sound]` — anchor-free (the octonion base is structural). -/
theorem seam_eq_anti0 (bits l u : Nat) (hb : 4 ≤ bits)
    (hl1 : 1 ≤ l) (hl : l < 2 ^ (bits-1)) (hu1 : 2 ^ (bits-1) ≤ u) (hu : u < 2 ^ bits) :
    anti0 bits l u = ! offSeam bits l u := by
  have htop1 : 1 ≤ 2 ^ (bits-1) := Nat.two_pow_pos _
  have hle : 2 ^ (bits-1) ≤ 2 ^ bits := Nat.pow_le_pow_right (by decide) (by omega)
  have hlt : l < 2 ^ bits := by omega
  have hl0 : l ≠ 0 := by omega
  have hu0 : u ≠ 0 := by omega
  cases hoff : offSeam bits l u with
  | false =>
    show anti0 bits l u = true
    rw [anti0_iff bits l u hl0 hu0 hlt hu]
    intro c hc
    exact on_seam_P_neg bits l u hb hl1 hl hu1 hu hoff c hc
  | true =>
    show anti0 bits l u = false
    have hxa := converse_holds bits l u hb hl1 hl hu1 hu hoff
    simp only [hasXorAnnih] at hxa
    rw [List.any_eq_true] at hxa
    obtain ⟨a, hmem, hpred⟩ := hxa
    rw [List.mem_range] at hmem
    rw [Bool.and_eq_true, Bool.and_eq_true, decide_eq_true_eq, decide_eq_true_eq, beq_iff_eq]
      at hpred
    obtain ⟨⟨_, _⟩, hPa⟩ := hpred
    cases hanti : anti0 bits l u with
    | false => rfl
    | true =>
      exfalso
      have hall := (anti0_iff bits l u hl0 hu0 hlt hu).mp hanti
      have hneg := hall a hmem
      exact absurd (hPa.symm.trans hneg) (by decide)

/-- **Corollary generalizing `coincidence_16`/`coincidence_32`'s `anti0 == ! offSeam` to ∀n over the
    loHi locus** (`bits ≥ 4`), via `loHi_mem` + `seam_eq_anti0`. -/
theorem anti0_eq_offSeam_all (bits : Nat) (hb : 4 ≤ bits) :
    (loHi bits).all (fun p => anti0 bits p.1 p.2 == ! offSeam bits p.1 p.2) = true := by
  rw [List.all_eq_true]
  intro p hp
  obtain ⟨hp1, hp2, hp3, hp4⟩ := loHi_mem bits (by omega) p hp
  rw [beq_iff_eq]
  exact seam_eq_anti0 bits p.1 p.2 hb hp1 hp2 hp3 hp4

/-- **The OPERATOR characterization also widens: `anti0 = ¬isZD` on the full box, ∀ bits.**  For *every*
    distinct nonzero pair, `{L_l,L_u}=0` ⟺ `e_l+e_u` is not a zero divisor — no loHi hypothesis.  Unlike
    the *geometric* `offSeam` (which is loHi-specific and false off the locus), the operator predicate
    `anti0` is an honest all-pairs ZD detector.  Proof: `anti0 ⟺ ∀c P(c)=−1` (`anti0_iff`), while
    `¬hasXorAnnih ⟺ ∀ non-exceptional a, P(a)=−1`; the two exceptional orbits `{0,d}` are handled by
    `P0_neg_general` (`P(0)=P(d)=−1`), so `anti0 = ¬hasXorAnnih = ¬isZD` (`isZD_eq_hasXorAnnih_full`). -/
theorem anti0_eq_not_isZD_full (bits l u : Nat)
    (hl1 : 1 ≤ l) (hl : l < 2 ^ bits) (hu1 : 1 ≤ u) (hu : u < 2 ^ bits) (hne : l ≠ u) :
    anti0 bits l u = ! isZD bits l u := by
  have hbits : 1 ≤ bits := by
    cases bits with
    | zero => rw [Nat.pow_zero] at hl; omega
    | succ _ => omega
  have hl0 : l ≠ 0 := by omega
  have hu0 : u ≠ 0 := by omega
  -- the {0,d} orbit product is -1 (from P0_neg_general, cancelling the two `cdSigma _ 0 = 1` factors)
  have hddu : cdSigma l (l ^^^ u) bits * cdSigma u (l ^^^ u) bits = -1 := by
    have h0 := P0_neg_general bits l u hl1 hl hu1 hu hne
    rw [cdSigma_zero_right bits l hbits, cdSigma_zero_right bits u hbits,
        Int.one_mul, Int.one_mul] at h0
    exact h0
  rw [isZD_eq_hasXorAnnih_full bits l u hl1 hl hu1 hu hne]
  cases hX : hasXorAnnih bits l u with
  | true =>
    rw [Bool.not_true]
    cases hA : anti0 bits l u with
    | false => rfl
    | true =>
      exfalso
      have hall := (anti0_iff bits l u hl0 hu0 hl hu).mp hA
      unfold hasXorAnnih at hX
      rw [List.any_eq_true] at hX
      obtain ⟨a, ha_mem, hpred⟩ := hX
      rw [Bool.and_eq_true, Bool.and_eq_true, decide_eq_true_eq, decide_eq_true_eq, beq_iff_eq] at hpred
      obtain ⟨⟨_, _⟩, hPa⟩ := hpred
      have hc := hall a (List.mem_range.mp ha_mem)
      rw [hPa] at hc
      exact absurd hc (by decide)
  | false =>
    rw [Bool.not_false, anti0_iff bits l u hl0 hu0 hl hu]
    intro c hc
    by_cases hc0 : c = 0
    · subst hc0; rw [Nat.zero_xor]
      exact P0_neg_general bits l u hl1 hl hu1 hu hne
    · by_cases hcd : c = l ^^^ u
      · subst hcd
        rw [Nat.xor_self, cdSigma_zero_right bits l hbits, cdSigma_zero_right bits u hbits,
            Int.mul_one, Int.mul_one]
        exact hddu
      · have hc1 : 1 ≤ c := by omega
        have hPc : cdSigma l c bits * cdSigma u c bits
            * cdSigma l (c ^^^ (l ^^^ u)) bits * cdSigma u (c ^^^ (l ^^^ u)) bits ≠ 1 := by
          intro hP1
          have hxt : hasXorAnnih bits l u = true := by
            unfold hasXorAnnih
            refine List.any_eq_true.mpr ⟨c, List.mem_range.mpr hc, ?_⟩
            rw [Bool.and_eq_true, Bool.and_eq_true, decide_eq_true_eq, decide_eq_true_eq, beq_iff_eq]
            exact ⟨⟨hc1, hcd⟩, hP1⟩
          rw [hX] at hxt; exact absurd hxt (by decide)
        rcases cdSigma_pm bits l c with h1 | h1 <;>
        rcases cdSigma_pm bits u c with h2 | h2 <;>
        rcases cdSigma_pm bits l (c ^^^ (l ^^^ u)) with h3 | h3 <;>
        rcases cdSigma_pm bits u (c ^^^ (l ^^^ u)) with h4 | h4 <;>
          rw [h1, h2, h3, h4] at hPc ⊢ <;>
          first | rfl | exact absurd rfl hPc

-- ══════════════════════════════════════════════════════════════════════════════════════════════════
-- THE FULL SEAM COINCIDENCE ON loHi, ∀ bits≥4 — every link now a proved ∀n theorem.
--   offSeam  ⟺  isZD  ⟺  hasXorAnnih  ⟺  ¬anti0
-- (forward obstruction `¬offSeam ⟹ ¬isZD` included; NO `native_decide` in the ∀n chain — fully
--  anchor-free; the 7 fixed-n regression anchors near the top of the file retain `native_decide`).
-- ══════════════════════════════════════════════════════════════════════════════════════════════════

/-- **On-seam loHi pairs have NO XOR-winner** (`hasXorAnnih = false`): every orbit loses (`on_seam_P_neg`),
    so no index passes the winner test.  This is the converse-side half of the forward obstruction. -/
theorem hasXorAnnih_false_of_onSeam (bits l u : Nat) (hb : 4 ≤ bits)
    (hl1 : 1 ≤ l) (hl : l < 2 ^ (bits-1)) (hu1 : 2 ^ (bits-1) ≤ u) (hu : u < 2 ^ bits)
    (hon : offSeam bits l u = false) :
    hasXorAnnih bits l u = false := by
  rw [Bool.eq_false_iff]
  intro hxa
  unfold hasXorAnnih at hxa
  rw [List.any_eq_true] at hxa
  obtain ⟨a, hmem, hpred⟩ := hxa
  rw [List.mem_range] at hmem
  rw [Bool.and_eq_true, Bool.and_eq_true, decide_eq_true_eq, decide_eq_true_eq, beq_iff_eq] at hpred
  obtain ⟨⟨-, -⟩, hP⟩ := hpred
  have hneg := on_seam_P_neg bits l u hb hl1 hl hu1 hu hon a hmem
  rw [hP] at hneg
  exact absurd hneg (by decide)

/-- **Forward obstruction + converse, unified**: on the loHi locus (`∀ bits≥4`) a lower×upper pair is a
    2-term zero divisor **iff** it is off-seam.  `isZD = offSeam`.  Off-seam ⟹ `isZD` is the converse
    (`converse_holds`+`hasXorAnnih_sound`); on-seam ⟹ `¬isZD` is the forward obstruction
    (`hasXorAnnih_false_of_onSeam`+`hasXorAnnih_complete`). Kernel-clean `[propext, Quot.sound]`,
    anchor-free. -/
theorem isZD_eq_offSeam (bits l u : Nat) (hb : 4 ≤ bits)
    (hl1 : 1 ≤ l) (hl : l < 2 ^ (bits-1)) (hu1 : 2 ^ (bits-1) ≤ u) (hu : u < 2 ^ bits) :
    isZD bits l u = offSeam bits l u := by
  have hle : 2 ^ (bits-1) ≤ 2 ^ bits := Nat.pow_le_pow_right (by decide) (by omega)
  have hlt : l < 2 ^ bits := by omega
  have hne : l ≠ u := by omega
  cases hoff : offSeam bits l u with
  | true =>
    exact hasXorAnnih_sound bits l u hlt hu hne (converse_holds bits l u hb hl1 hl hu1 hu hoff)
  | false =>
    have hnx := hasXorAnnih_false_of_onSeam bits l u hb hl1 hl hu1 hu hoff
    rw [Bool.eq_false_iff]
    intro hzd
    rw [hasXorAnnih_complete bits l u hb hl1 hl hu1 hu hzd] at hnx
    exact absurd hnx (by decide)

/-- **The full seam coincidence, ∀ bits≥4 on loHi** — the four predicates coincide, each link a proved
    ∀n theorem: `isZD = offSeam`, `hasXorAnnih = offSeam`, `anti0 = ¬offSeam`. -/
theorem seam_coincidence (bits l u : Nat) (hb : 4 ≤ bits)
    (hl1 : 1 ≤ l) (hl : l < 2 ^ (bits-1)) (hu1 : 2 ^ (bits-1) ≤ u) (hu : u < 2 ^ bits) :
    isZD bits l u = offSeam bits l u
      ∧ hasXorAnnih bits l u = offSeam bits l u
      ∧ anti0 bits l u = ! offSeam bits l u := by
  refine ⟨isZD_eq_offSeam bits l u hb hl1 hl hu1 hu, ?_, seam_eq_anti0 bits l u hb hl1 hl hu1 hu⟩
  cases hoff : offSeam bits l u with
  | true => exact converse_holds bits l u hb hl1 hl hu1 hu hoff
  | false => exact hasXorAnnih_false_of_onSeam bits l u hb hl1 hl hu1 hu hoff

-- ══════════════════════════════════════════════════════════════════════════════════════════════════
-- THE LAST COINCIDENCE MEMBER, ∀n:  anti0 ⟺ llsqNegI   ({L_l,L_u}=0  ⟺  (L_l L_u)²=−I)
-- Both reduce to the SAME converse condition `∀c P(c)=−1` (via the cocycle), so they coincide for every
-- l,u ≠ 0 — not just loHi.  Generalizes the `anti0==llsqNegI` member of `coincidence_16/32` to all n.
-- ══════════════════════════════════════════════════════════════════════════════════════════════════

/-- **`R(c) = P(c)`**: the `(L_l L_u)²=−I` four-sign product equals the converse winner product `P(c)`,
    via two `cdSigma_cocycle'` rewrites (`σ(l,l⊕c)=−σ(l,c)`, `σ(l,u⊕c)=−σ(l,c⊕d)`).  Needs `l ≠ 0`. -/
theorem llsq_QP (bits l u c : Nat) (hl0 : l ≠ 0)
    (hl : l < 2 ^ bits) (hu : u < 2 ^ bits) (hc : c < 2 ^ bits) :
    cdSigma l (l ^^^ c) bits * cdSigma u (l ^^^ u ^^^ c) bits
        * cdSigma l (u ^^^ c) bits * cdSigma u c bits
      = cdSigma l c bits * cdSigma u c bits
        * cdSigma l (c ^^^ (l ^^^ u)) bits * cdSigma u (c ^^^ (l ^^^ u)) bits := by
  have hx : c ^^^ (l ^^^ u) < 2 ^ bits := Nat.xor_lt_two_pow hc (Nat.xor_lt_two_pow hl hu)
  have hidx : l ^^^ u ^^^ c = c ^^^ (l ^^^ u) := Nat.xor_comm (l ^^^ u) c
  -- σ(l,l⊕c) = −σ(l,c)
  have hcoc1 := cdSigma_cocycle' bits l c hl hc hl0
  have hb2 : cdSigma l (l ^^^ c) bits = - cdSigma l c bits := by
    rcases cdSigma_pm bits l c with h|h <;>
    rcases cdSigma_pm bits l (l ^^^ c) with h'|h' <;>
      rw [h, h'] at hcoc1 ⊢ <;> first | decide | exact absurd hcoc1 (by decide)
  -- σ(l,u⊕c) = −σ(l,c⊕d)  (l ⊕ (c⊕d) = u⊕c)
  have hlx : l ^^^ (c ^^^ (l ^^^ u)) = u ^^^ c := by
    rw [xor_left_comm l c (l ^^^ u), ← Nat.xor_assoc l l u, Nat.xor_self, Nat.zero_xor,
      Nat.xor_comm c u]
  have hcoc2 := cdSigma_cocycle' bits l (c ^^^ (l ^^^ u)) hl hx hl0
  rw [hlx] at hcoc2
  have hc2 : cdSigma l (u ^^^ c) bits = - cdSigma l (c ^^^ (l ^^^ u)) bits := by
    rcases cdSigma_pm bits l (c ^^^ (l ^^^ u)) with h|h <;>
    rcases cdSigma_pm bits l (u ^^^ c) with h'|h' <;>
      rw [h, h'] at hcoc2 ⊢ <;> first | decide | exact absurd hcoc2 (by decide)
  rw [hidx, hb2, hc2]
  rcases cdSigma_pm bits l c with h1|h1 <;>
  rcases cdSigma_pm bits u c with h2|h2 <;>
  rcases cdSigma_pm bits l (c ^^^ (l ^^^ u)) with h3|h3 <;>
  rcases cdSigma_pm bits u (c ^^^ (l ^^^ u)) with h4|h4 <;>
    rw [h1, h2, h3, h4] <;> decide

/-- **Bridge:** `llsqNegI = true ↔ ∀ c < 2^bits, P(c) = −1` — the SAME condition `anti0_iff` gives. -/
theorem llsqNegI_iff (bits l u : Nat) (hl0 : l ≠ 0)
    (hl : l < 2 ^ bits) (hu : u < 2 ^ bits) :
    llsqNegI bits l u = true ↔ ∀ c, c < 2 ^ bits →
      cdSigma l c bits * cdSigma u c bits
        * cdSigma l (c ^^^ (l ^^^ u)) bits * cdSigma u (c ^^^ (l ^^^ u)) bits = -1 := by
  simp only [llsqNegI]
  rw [List.all_eq_true]
  constructor
  · intro h c hc
    have hcc := h c (List.mem_range.mpr hc)
    rw [beq_iff_eq] at hcc
    rw [← llsq_QP bits l u c hl0 hl hu hc]; exact hcc
  · intro h c hc
    rw [List.mem_range] at hc
    rw [beq_iff_eq, llsq_QP bits l u c hl0 hl hu hc]
    exact h c hc

/-- **The last member, proved ∀n**: `{L_l,L_u}=0 ⟺ (L_l L_u)²=−I`, i.e. `anti0 = llsqNegI`, for every
    `l,u ≠ 0` (`< 2^bits`) — not just loHi.  Both sides `↔ ∀c P(c)=−1` (`anti0_iff`/`llsqNegI_iff`). -/
theorem anti0_eq_llsqNegI (bits l u : Nat) (hl0 : l ≠ 0) (hu0 : u ≠ 0)
    (hl : l < 2 ^ bits) (hu : u < 2 ^ bits) :
    anti0 bits l u = llsqNegI bits l u := by
  have key := (anti0_iff bits l u hl0 hu0 hl hu).trans (llsqNegI_iff bits l u hl0 hl hu).symm
  cases hA : anti0 bits l u <;> cases hB : llsqNegI bits l u
  · rfl
  · rw [hA, hB] at key; exact absurd (key.mpr rfl) (by decide)
  · rw [hA, hB] at key; exact absurd (key.mp rfl) (by decide)
  · rfl

/-- Corollary over `loHi`, generalizing the `anti0==llsqNegI` member of `coincidence_16/32` to all n. -/
theorem anti0_eq_llsqNegI_all (bits : Nat) (hb : 4 ≤ bits) :
    (loHi bits).all (fun p => anti0 bits p.1 p.2 == llsqNegI bits p.1 p.2) = true := by
  rw [List.all_eq_true]
  intro p hp
  obtain ⟨hp1, hp2, hp3, hp4⟩ := loHi_mem bits (by omega) p hp
  have htop : 1 ≤ 2 ^ (bits-1) := Nat.two_pow_pos (bits-1)
  have hle : 2 ^ (bits-1) ≤ 2 ^ bits := Nat.pow_le_pow_right (by decide) (by omega)
  rw [beq_iff_eq]
  exact anti0_eq_llsqNegI bits p.1 p.2 (by omega) (by omega) (by omega) (by omega)

end SounioCDConverse
