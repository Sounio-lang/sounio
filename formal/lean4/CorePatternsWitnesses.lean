/-
  CorePatternsWitnesses — compile-verified companions to CORE_PROOF_PATTERNS.md.

  Every pattern documented in that file has a witness here. If this file
  compiles with `lean CorePatternsWitnesses.lean` (exit 0, no `sorry`
  warning), every POSITIVE claim in the document is true of the pinned
  toolchain. The NEGATIVE claims — what core does not have — are recorded
  in the document with the probe output that produced them, since a
  non-existent name cannot be witnessed by a file that compiles.

  Companion to `README.md` + `TACTICS_CORE_WITNESSES.lean` (kimi-cli1),
  which own the tactic-name → core-equivalent table. This file deliberately
  does NOT repeat those rows. It covers the failures where the tactic name
  is not the problem.

  Mathlib-free. Lean 4.33.0.
-/
set_option linter.unusedSimpArgs false

namespace Sounio.CorePatterns

/-! ## P1. Fold invariance under permutation

Mathlib route: `List.Perm.foldl_eq`, or reindex a `Finset.sum` with
`Finset.sum_bij`. Neither exists in core.

Core route: induct on the `List.Perm` DERIVATION. It has exactly four
constructors — `nil`, `cons`, `swap`, `trans` — so the proof never
destructs the list itself. That property is what makes it safe over a
concrete `List.range n`: see P2. -/

def iSum (l : List Nat) (f : Nat → Int) : Int :=
  l.foldl (fun acc i => acc + f i) 0

theorem foldl_add_init (l : List Nat) (g : Nat → Int) (init : Int) :
    l.foldl (fun acc i => acc + g i) init
      = init + l.foldl (fun acc i => acc + g i) 0 := by
  induction l generalizing init with
  | nil => simp
  | cons i l ih =>
    simp only [List.foldl, show (0 : Int) + g i = g i from by simp]
    calc List.foldl (fun acc i => acc + g i) init (i :: l)
        = List.foldl (fun acc i => acc + g i) (init + g i) l := rfl
      _ = init + g i + List.foldl (fun acc i => acc + g i) 0 l := ih (init + g i)
      _ = init + List.foldl (fun acc i => acc + g i) (g i) l := by
          rw [Int.add_assoc]
          congr 1
          exact (ih (g i)).symm

theorem iSum_cons (i : Nat) (l : List Nat) (f : Nat → Int) :
    iSum (i :: l) f = f i + iSum l f := by
  simp [iSum, List.foldl, show (0 : Int) + f i = f i from by simp]
  exact foldl_add_init l f (f i)

/-- The pattern. Four constructor cases, no list destruction. -/
theorem iSum_perm {l₁ l₂ : List Nat} (f : Nat → Int) (h : List.Perm l₁ l₂) :
    iSum l₁ f = iSum l₂ f := by
  induction h with
  | nil => rfl
  | cons x _ ih => rw [iSum_cons, iSum_cons, ih]
  | swap x y l => rw [iSum_cons, iSum_cons, iSum_cons, iSum_cons]; omega
  | trans _ _ ih₁ ih₂ => rw [ih₁, ih₂]

theorem iSum_map (l : List Nat) (g : Nat → Nat) (f : Nat → Int) :
    iSum (l.map g) f = iSum l (fun i => f (g i)) := by
  induction l with
  | nil => rfl
  | cons x l ih => rw [List.map_cons, iSum_cons, iSum_cons, ih]

/-- Re-index along `g` when `g` permutes the index list. The list stays a
    variable throughout, so instantiating at `List.range 16` is free. -/
theorem iSum_reindex (l : List Nat) (g : Nat → Nat) (f : Nat → Int)
    (hp : List.Perm (l.map g) l) :
    iSum l (fun j => f (g j)) = iSum l f := by
  rw [← iSum_map l g f]
  exact iSum_perm f hp

/-! ## P2. Instantiating a general lemma at a concrete range

The point of P1 is that the lemma is proved about a VARIABLE `l` and only
then instantiated. Instantiation is cheap; rewriting inside the concrete
range is what expands it into a literal (see CORE_PROOF_PATTERNS.md §2). -/

theorem iSum_reindex_range16 (g : Nat → Nat) (f : Nat → Int)
    (hp : List.Perm ((List.range 16).map g) (List.range 16)) :
    iSum (List.range 16) (fun j => f (g j)) = iSum (List.range 16) f :=
  iSum_reindex _ g f hp

/-! ## P3. Cancellation lemmas core does not ship

`Nat.xor_cancel_right`, `Nat.xor_cancel_left` and `Nat.xor_right_injective`
are all `unknown constant`. Core ships only the algebraic skeleton
(`Nat.xor_self`, `Nat.xor_assoc`, `Nat.xor_comm`, `Nat.zero_xor`,
`Nat.xor_zero`), which is enough to derive the rest in two lines each.

The general habit: when a cancellation lemma is missing, look for the
associativity/identity/involution triple and derive it. -/

theorem xor_undo (a b : Nat) : (a ^^^ b) ^^^ b = a := by
  rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]

theorem xor_mid (a b : Nat) : (a ^^^ b) ^^^ a = b := by
  rw [Nat.xor_comm a b, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]

theorem xor_ne_self {t : Nat} (ht : t ≠ 0) (j : Nat) : j ^^^ t ≠ j := by
  intro hc
  apply ht
  have h2 : (j ^^^ t) ^^^ j = t := by
    rw [Nat.xor_comm j t]
    exact xor_undo t j
  rw [hc, Nat.xor_self] at h2
  exact h2.symm

/-! ## P4. Choosing a witness out of a hypothesis

Mathlib route: the `choose` tactic. Core has `Classical.choose` /
`Classical.choose_spec` as TERMS, but writing `if h : ∃ …` to guard them
fails: `dite` needs `Decidable` and a bare `Prop` has no instance.

Core route: totalise the existential first, so the guard is a decidable
arithmetic condition rather than the Prop itself. -/

theorem choose_total (n : Nat) (Q : Nat → Nat → Prop)
    (hex : ∀ i, i < n → ∃ k, Q i k) :
    ∃ ks : Nat → Nat, ∀ i, i < n → Q i (ks i) := by
  have htot : ∀ i, ∃ k, i < n → Q i k := by
    intro i
    if hi : i < n then
      obtain ⟨k, hk⟩ := hex i hi
      exact ⟨k, fun _ => hk⟩
    else
      exact ⟨0, fun h => absurd h hi⟩
  exact ⟨fun i => Classical.choose (htot i), fun i hi => Classical.choose_spec (htot i) hi⟩

/-! ## P5. Bounded case analysis

Mathlib route: `interval_cases n`. Unknown tactic in core.

Core route: `match` on numeral patterns with a catch-all successor case
that `omega` discharges. Each branch is a closed instance, so `decide`
finishes it — kernel `decide`, not `native_decide`. -/

theorem lt_four_cases (n : Nat) (h : n < 4) :
    n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 := by
  match n with
  | 0 => decide
  | 1 => decide
  | 2 => decide
  | 3 => decide
  | _ + 4 => omega

/-! ## P6. Counting with an involution, without `Finset.card`

Mathlib route: `Finset.card_eq_two_mul_iff`-style pairing, or
`Finset.sum_involution`. `Finset` does not exist in core at all.

Core route: sort the index set by which side of the involution it sits on,
then reindex one side onto the other with P1. The lemma below is the
reusable form; it needs no bit-position argument and no explicit pairing. -/

def nSum (l : List Nat) (f : Nat → Nat) : Nat :=
  l.foldl (fun acc i => acc + f i) 0

theorem foldl_add_init_nat (l : List Nat) (g : Nat → Nat) (init : Nat) :
    l.foldl (fun acc i => acc + g i) init
      = init + l.foldl (fun acc i => acc + g i) 0 := by
  induction l generalizing init with
  | nil => simp
  | cons i l ih =>
    simp [List.foldl]
    rw [ih (init + g i), ih (g i)]
    omega

theorem nSum_cons (i : Nat) (l : List Nat) (f : Nat → Nat) :
    nSum (i :: l) f = f i + nSum l f := by
  simp [nSum, List.foldl]
  exact foldl_add_init_nat l f (f i)

theorem nSum_add (l : List Nat) (f g : Nat → Nat) :
    nSum l (fun x => f x + g x) = nSum l f + nSum l g := by
  induction l with
  | nil => simp [nSum]
  | cons x l ih => rw [nSum_cons, nSum_cons, nSum_cons, ih]; omega

theorem nSum_pointwise (l : List Nat) (f g : Nat → Nat)
    (h : ∀ x, x ∈ l → f x = g x) : nSum l f = nSum l g := by
  induction l with
  | nil => rfl
  | cons x l ih =>
    rw [nSum_cons, nSum_cons, h x (by simp), ih (fun y hy => h y (by simp [hy]))]

theorem nSum_perm {l₁ l₂ : List Nat} (f : Nat → Nat) (h : List.Perm l₁ l₂) :
    nSum l₁ f = nSum l₂ f := by
  induction h with
  | nil => rfl
  | cons x _ ih => rw [nSum_cons, nSum_cons, ih]
  | swap x y l => rw [nSum_cons, nSum_cons, nSum_cons, nSum_cons]; omega
  | trans _ _ ih₁ ih₂ => rw [ih₁, ih₂]

theorem nSum_map (l : List Nat) (g f : Nat → Nat) :
    nSum (l.map g) f = nSum l (fun i => f (g i)) := by
  induction l with
  | nil => rfl
  | cons x l ih => rw [List.map_cons, nSum_cons, nSum_cons, ih]

theorem nSum_reindex (l : List Nat) (g f : Nat → Nat)
    (hp : List.Perm (l.map g) l) :
    nSum l (fun j => f (g j)) = nSum l f := by
  rw [← nSum_map l g f]
  exact nSum_perm f hp

private def sideLo (g : Nat → Nat) (P : Nat → Bool) (j : Nat) : Nat :=
  if decide (j < g j) && P j then 1 else 0

private def sideHi (g : Nat → Nat) (P : Nat → Bool) (j : Nat) : Nat :=
  if decide (g j < j) && P j then 1 else 0

/-- A `P`-set closed under a fixed-point-free involution has even size.
    `hperm` says `g` permutes the index list; `hinv` that it is an
    involution; `hfree` that it has no fixed point; `hP` that it preserves
    the predicate. -/
theorem count_even_of_involution (l : List Nat) (g : Nat → Nat) (P : Nat → Bool)
    (hperm : List.Perm (l.map g) l)
    (hinv : ∀ j, g (g j) = j)
    (hfree : ∀ j, g j ≠ j)
    (hP : ∀ j, P (g j) = P j) :
    nSum l (fun j => if P j then 1 else 0) % 2 = 0 := by
  have hsplit : nSum l (fun j => if P j then 1 else 0)
      = nSum l (sideLo g P) + nSum l (sideHi g P) := by
    rw [← nSum_add]
    refine nSum_pointwise _ _ _ (fun j _ => ?_)
    have hne : g j ≠ j := hfree j
    cases hc : P j
    · simp [sideLo, sideHi, hc]
    · rcases Nat.lt_or_ge j (g j) with h | h
      · have h2 : ¬ (g j < j) := by omega
        simp [sideLo, sideHi, hc, h, h2]
      · have h1 : g j < j := by omega
        have h2 : ¬ (j < g j) := by omega
        simp [sideLo, sideHi, hc, h1, h2]
  have hpt : ∀ j, sideLo g P j = sideHi g P (g j) := by
    intro j
    unfold sideLo sideHi
    rw [hinv j, hP j]
  have hswap : nSum l (sideLo g P) = nSum l (sideHi g P) :=
    calc nSum l (sideLo g P)
        = nSum l (fun j => sideHi g P (g j)) :=
          nSum_pointwise _ _ _ (fun j _ => hpt j)
      _ = nSum l (sideHi g P) := nSum_reindex _ _ _ hperm
  rw [hsplit, hswap]
  omega

/-- Non-vacuity witness for `count_even_of_involution`: its four hypotheses
    are simultaneously satisfiable, and the conclusion computes. Bit-flip on
    `{0,1,2,3}` with the always-true predicate gives a count of 4. Without
    this, contradictory hypotheses would make the lemma above unusable
    while still compiling. -/
example : nSum (List.range 4) (fun _ => if (fun _ : Nat => true) 0 then 1 else 0) % 2 = 0 :=
  count_even_of_involution (List.range 4) (fun j => j ^^^ 1) (fun _ => true)
    (by decide)
    (fun j => xor_undo j 1)
    (fun j => xor_ne_self (by decide) j)
    (fun _ => rfl)

/-- …and the count really is 4, not 0 — the instance is not empty. -/
example : nSum (List.range 4) (fun _ => 1) = 4 := by decide

/-! ## P7. `omega` crosses `Nat → Int` casts

No `push_cast` in core (see the tactic table in `README.md`). For pure
arithmetic goals `omega` handles the cast directly, which is stronger than
reaching for `simp` and hoping the right cast lemma fires. Note that
`Nat.cast_add` and `Int.ofNat_add` are not addressable names, so
`simp only [Nat.cast_add]` cannot be written at all. -/

theorem iSum_ofNat (l : List Nat) (f : Nat → Nat) :
    iSum l (fun i => (f i : Int)) = (nSum l f : Int) := by
  induction l with
  | nil => rfl
  | cons x l ih => rw [iSum_cons, nSum_cons, ih]; omega

/-! ## P8. `omega` also does `Int` remainder

Parity arguments over `Int` need no special tactic. `%` on `Int` is
`Int.emod`, which is non-negative for a positive modulus, so `(-1) % 2 = 1`
and parity reasoning behaves as expected. -/

theorem neg_one_emod_two : ((0 : Int) - 1) % 2 = 1 := by decide

theorem iSum_mod2_congr (l : List Nat) (f g : Nat → Int)
    (h : ∀ a, a ∈ l → f a % 2 = g a % 2) : iSum l f % 2 = iSum l g % 2 := by
  induction l with
  | nil => rfl
  | cons x l ih =>
    rw [iSum_cons, iSum_cons]
    have hx := h x (by simp)
    have hr := ih (fun a ha => h a (by simp [ha]))
    omega

/-! ## P9. Telescoping a closed walk without a rotation permutation

The obvious route to `Σ_i (c i − c ((i+1) % n)) = 0` is to show
`i ↦ (i+1) % n` permutes `List.range n`, which for a general `n` is real
work. Splitting the last step off instead turns the modular index into a
plain one on `List.range m`, where `Nat.mod_eq_of_lt` applies pointwise. -/

theorem iSum_congr (l : List Nat) (f g : Nat → Int)
    (h : ∀ a, a ∈ l → f a = g a) : iSum l f = iSum l g := by
  induction l with
  | nil => rfl
  | cons x l ih =>
    rw [iSum_cons, iSum_cons, h x (by simp), ih (fun y hy => h y (by simp [hy]))]

theorem iSum_range_succ (n : Nat) (f : Nat → Int) :
    iSum (List.range (n + 1)) f = iSum (List.range n) f + f n := by
  rw [List.range_succ, iSum, List.foldl_append]
  simp [List.foldl]
  rw [foldl_add_init]
  simp [iSum]

theorem telescope (m : Nat) (g : Nat → Int) :
    iSum (List.range m) (fun i => g i - g (i + 1)) = g 0 - g m := by
  induction m with
  | zero => simp [iSum]
  | succ m ih => rw [iSum_range_succ, ih]; omega

theorem closed_walk_sum_zero (m : Nat) (c : Nat → Int) :
    iSum (List.range (m + 1)) (fun i => c i - c ((i + 1) % (m + 1))) = 0 := by
  rw [iSum_range_succ]
  have hinner : iSum (List.range m) (fun i => c i - c ((i + 1) % (m + 1)))
      = iSum (List.range m) (fun i => c i - c (i + 1)) := by
    refine iSum_congr _ _ _ (fun i hi => ?_)
    have hi' : i < m := List.mem_range.mp hi
    rw [Nat.mod_eq_of_lt (by omega)]
  rw [hinner, telescope]
  simp only [Nat.mod_self]
  omega

/-! ## P10. Witnesses falsifying stale "core lacks X" comments

See CORE_PROOF_PATTERNS.md §10 for the audit table. The comment at
`SounioRealCauchy.lean:63` says deriving `Rat.div_pos` from `Rat.mul_pos`
and `Rat.inv_pos` is "non-trivial without `ring`/`field_simp` (Mathlib)".
Both inputs exist in core and the derivation is two lines. -/

theorem rat_div_pos {a b : Rat} (ha : 0 < a) (hb : 0 < b) : 0 < a / b := by
  rw [Rat.div_def]
  exact Rat.mul_pos ha (Rat.inv_pos.mpr hb)

/-- The concrete instance that comment says it needs. -/
example {e : Rat} (he : 0 < e) : 0 < e / 2 := rat_div_pos he (by decide)

/-! ## P11. Mathlib-only tactics and their core-only replacements

Each Mathlib-only tactic below was confirmed absent by a negative probe that
errors with `unknown tactic` under core 4.33.0; the replacements here are the
positive witnesses. See CORE_PROOF_PATTERNS.md §11 for the table.

Confirmed Mathlib-only: `by_contra`, `set`, `nlinarith`, `linarith`,
`positivity`, `ring`, `interval_cases`, `norm_num`, `field_simp`.

Confirmed present in core, contrary to a widespread assumption: `push_cast`,
`norm_cast`, `exact_mod_cast`. What is missing is the Mathlib *lemma names*
(`Nat.cast_add`, `Int.ofNat_add`), not the tactics. -/

theorem core_by_contra (p : Prop) (h : ¬¬p) : p :=
  Classical.byContradiction h

theorem core_by_contra_tac (n : Nat) (h : n ≠ 0) : 0 < n := by
  match n with
  | 0 => exact absurd rfl h
  | _ + 1 => omega

/-- Replacement for `set`: a top-level `def`, or `generalize` when the
abbreviation is only needed for one goal. -/
def shorthand (a b : Nat) : Nat := a + b

theorem core_set (a b : Nat) : shorthand a b = shorthand b a := by
  unfold shorthand; omega

theorem core_set_generalize (a b : Nat) : (a + b) + 0 = a + b := by
  generalize a + b = s
  omega

theorem core_nlinarith (a b : Nat) (h : a ≤ b) : a * a ≤ b * b :=
  Nat.mul_le_mul h h

theorem core_linarith (a b c : Nat) (h1 : a ≤ b) (h2 : b ≤ c) : a ≤ c := by omega

theorem core_positivity (x : Nat) : 0 < x + 1 := Nat.succ_pos x

theorem core_ring (a b : Nat) : (a + b) * (a + b) = a*a + 2*(a*b) + b*b := by
  simp [Nat.mul_add, Nat.mul_comm]
  omega

theorem core_interval_cases (n : Nat) (h : n < 3) : n = 0 ∨ n = 1 ∨ n = 2 := by
  match n with
  | 0 => decide
  | 1 => decide
  | 2 => decide
  | _ + 3 => omega

theorem core_norm_num : (2 : Nat) + 2 = 4 := by decide

/-- `omega` crosses `Nat → Int` unaided, including truncated subtraction. -/
theorem core_cast_omega (a b : Nat) (h : a ≤ b) :
    ((b - a : Nat) : Int) = (b : Int) - (a : Int) := by omega

/-- `push_cast` is available in core and does fire here; `rfl` alone fails on
this goal, so the tactic is doing real work rather than being a no-op. -/
theorem core_cast_pushcast (a b : Nat) (h : a ≤ b) :
    ((b - a : Nat) : Int) = (b : Int) - (a : Int) := by
  push_cast [h]
  omega

end Sounio.CorePatterns
