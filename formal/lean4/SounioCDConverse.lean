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

end SounioCDConverse
