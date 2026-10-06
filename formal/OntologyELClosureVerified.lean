/-
# Sounio.OntologyELClosureVerified — the boolean closure is exactly `SubDer`

Follow-up closure of round 4 of the *ontology frontiers* research line:
`OntologyELReasoner.lean` §6 defines a *computational* boolean closure
(`subPairs` / `transStep` / `closeSubIter` / `closeSub` / `subB` /
`conflictB` over `Fin n`) alongside the *inductive* deductive system
`SubDer`, but only spot-checks their agreement by `native_decide` on the
`Fin 8` biomedical instance. This file proves the agreement **in general**,
for every TBox over `Fin n`:

1. `transStep_justified` — **one-step soundness**: if every pair of a
   relation `R` is `SubDer`-derivable, so is every pair of `transStep R`
   (the pairs added by one transitivity-propagation step are justified by
   the `SubDer.trans` constructor applied to the two premise derivations).
   Iterating (`closeSubIter_justified`) lifts the invariant to any number
   of iterations.
2. `subB_sound` — **closure soundness**: `subB t c d = true → SubDer t c d`,
   because the seed relation (subsumption edges + diagonal) is justified
   (`seed_justified`) and the invariant is preserved.
3. `subB_complete` — **completeness**: `SubDer t c d → subB t c d = true`.
   The derivation is first linearised into a walk (`subDer_walkL`); a walk
   of `m` edges is contained in `closeSubIter m` of the seed
   (`walkL_mem_closeSubIter`); and any walk can be shortened to one whose
   node list has no duplicates (`walkL_shorten_aux`, the chain-shortening /
   loop-cutting lemma: if the head node reappears later, cut the loop),
   hence has at most `n` nodes, i.e. at most `n - 1 ≤ n + 1` edges, which
   is within the `n + 1` iterations `closeSub` performs.
4. `subB_iff_subDer` — the equivalence: `subB t c d = true ↔ SubDer t c d`.
5. `conflictB_sound`, `conflictB_complete`, `conflictB_iff` — the boolean
   conflict oracle is exactly the semantic `DerivedConflict`:
   `conflictB t c c' = true ↔ DerivedConflict t c c'`.

Everything is proved constructively over the general `Fin n` TBox; the
`Fin 8` biomedical instance of `OntologyELReasoner.lean` §7 is re-checked
through the new theorems (`native_decide` for the booleans, then routed
through `subB_sound` / `conflictB_sound` and the contrapositive of
`subB_complete`).

Self-contained. No Mathlib. Zero sorry. No new axioms.
-/

import OntologyELReasoner

namespace Sounio.OntologyELReasoner

variable {n : Nat}

-- ---------------------------------------------------------------------------
-- §1. Membership characterisations of the list combinators
-- ---------------------------------------------------------------------------

/-- Membership in `dedupPairs` is membership in the original list. -/
theorem mem_dedupPairs {x : Fin n × Fin n} {l : List (Fin n × Fin n)} :
    x ∈ dedupPairs l ↔ x ∈ l := by
  induction l with
  | nil => exact ⟨fun h => h, fun h => h⟩
  | cons a l ih =>
      show x ∈ (if (dedupPairs l).elem a then dedupPairs l else a :: dedupPairs l) ↔
        x ∈ a :: l
      by_cases h : (dedupPairs l).elem a = true
      · rw [if_pos h]
        constructor
        · intro hx
          exact List.mem_cons_of_mem _ (ih.mp hx)
        · intro hx
          cases List.mem_cons.mp hx with
          | inl he => subst he; exact List.mem_of_elem_eq_true h
          | inr hl => exact ih.mpr hl
      · rw [if_neg h]
        constructor
        · intro hx
          cases List.mem_cons.mp hx with
          | inl he => subst he; exact List.mem_cons_self
          | inr hl => exact List.mem_cons_of_mem _ (ih.mp hl)
        · intro hx
          cases List.mem_cons.mp hx with
          | inl he => subst he; exact List.mem_cons_self
          | inr hl => exact List.mem_cons_of_mem _ (ih.mpr hl)

/-- Membership in the inner composition loop of `transStep`: composing the
    fixed pair `p` with every pair of `R`. -/
theorem mem_transStep_inner {x p : Fin n × Fin n} {acc R : List (Fin n × Fin n)} :
    x ∈ R.foldr (fun q acc' => if p.2 == q.1 then (p.1, q.2) :: acc' else acc') acc ↔
      x ∈ acc ∨ ∃ q ∈ R, p.2 == q.1 ∧ x = (p.1, q.2) := by
  induction R generalizing acc with
  | nil =>
      simp only [List.foldr_nil]
      constructor
      · intro h; exact Or.inl h
      · intro h
        cases h with
        | inl h => exact h
        | inr h => obtain ⟨q, hq, -, -⟩ := h; exact absurd hq List.not_mem_nil
  | cons a l ih =>
      simp only [List.foldr_cons]
      by_cases h : p.2 == a.1
      · rw [if_pos h]
        constructor
        · intro hx
          cases List.mem_cons.mp hx with
          | inl he => exact Or.inr ⟨a, List.mem_cons_self, h, he⟩
          | inr hxr =>
              cases ih.mp hxr with
              | inl hacc => exact Or.inl hacc
              | inr h2 =>
                  obtain ⟨q, hq, hcond, hxe⟩ := h2
                  exact Or.inr ⟨q, List.mem_cons_of_mem _ hq, hcond, hxe⟩
        · intro hx
          cases hx with
          | inl hacc => exact List.mem_cons_of_mem _ (ih.mpr (Or.inl hacc))
          | inr h2 =>
              obtain ⟨q, hq, hcond, hxe⟩ := h2
              cases List.mem_cons.mp hq with
              | inl hqa => subst hqa; subst hxe; exact List.mem_cons_self
              | inr hql =>
                  exact List.mem_cons_of_mem _ (ih.mpr (Or.inr ⟨q, hql, hcond, hxe⟩))
      · rw [if_neg h]
        constructor
        · intro hx
          cases ih.mp hx with
          | inl hacc => exact Or.inl hacc
          | inr h2 =>
              obtain ⟨q, hq, hcond, hxe⟩ := h2
              exact Or.inr ⟨q, List.mem_cons_of_mem _ hq, hcond, hxe⟩
        · intro hx
          apply ih.mpr
          cases hx with
          | inl hacc => exact Or.inl hacc
          | inr h2 =>
              obtain ⟨q, hq, hcond, hxe⟩ := h2
              cases List.mem_cons.mp hq with
              | inl hqa => subst hqa; exact absurd hcond h
              | inr hql => exact Or.inr ⟨q, hql, hcond, hxe⟩

/-- Membership in the outer composition loop of `transStep`: composing every
    pair of `P` with every pair of `R`. -/
theorem mem_transStep_new {x : Fin n × Fin n} {acc P R : List (Fin n × Fin n)} :
    x ∈ P.foldr (fun p acc => R.foldr
        (fun q acc' => if p.2 == q.1 then (p.1, q.2) :: acc' else acc') acc) acc ↔
      x ∈ acc ∨ ∃ p ∈ P, ∃ q ∈ R, p.2 == q.1 ∧ x = (p.1, q.2) := by
  induction P generalizing acc with
  | nil =>
      simp only [List.foldr_nil]
      constructor
      · intro h; exact Or.inl h
      · intro h
        cases h with
        | inl h => exact h
        | inr h => obtain ⟨p, hp, -⟩ := h; exact absurd hp List.not_mem_nil
  | cons a l ih =>
      simp only [List.foldr_cons]
      rw [mem_transStep_inner, ih]
      constructor
      · intro hx
        cases hx with
        | inl hx =>
            cases hx with
            | inl hacc => exact Or.inl hacc
            | inr h2 =>
                obtain ⟨p, hp, q, hq, hcond, hxe⟩ := h2
                exact Or.inr ⟨p, List.mem_cons_of_mem _ hp, q, hq, hcond, hxe⟩
        | inr h2 =>
            obtain ⟨q, hq, hcond, hxe⟩ := h2
            exact Or.inr ⟨a, List.mem_cons_self, q, hq, hcond, hxe⟩
      · intro hx
        cases hx with
        | inl hacc => exact Or.inl (Or.inl hacc)
        | inr h2 =>
            obtain ⟨p, hp, q, hq, hcond, hxe⟩ := h2
            cases List.mem_cons.mp hp with
            | inl hpa => subst hpa; exact Or.inr ⟨q, hq, hcond, hxe⟩
            | inr hpl => exact Or.inl (Or.inr ⟨p, hpl, q, hq, hcond, hxe⟩)

/-- Exact membership characterisation of one transitivity-propagation step:
    the old pairs, plus one composed pair per composable pair of pairs. -/
theorem mem_transStep {x : Fin n × Fin n} {R : List (Fin n × Fin n)} :
    x ∈ transStep R ↔ x ∈ R ∨ ∃ p ∈ R, ∃ q ∈ R, p.2 = q.1 ∧ x = (p.1, q.2) := by
  unfold transStep
  rw [mem_dedupPairs, List.mem_append, mem_transStep_new]
  constructor
  · intro h
    cases h with
    | inl h => exact Or.inl h
    | inr h =>
        cases h with
        | inl h => exact absurd h List.not_mem_nil
        | inr h =>
            obtain ⟨p, hp, q, hq, hb, hx⟩ := h
            exact Or.inr ⟨p, hp, q, hq, beq_iff_eq.mp hb, hx⟩
  · intro h
    cases h with
    | inl h => exact Or.inl h
    | inr h =>
        obtain ⟨p, hp, q, hq, hb, hx⟩ := h
        exact Or.inr (Or.inr ⟨p, hp, q, hq, beq_iff_eq.mpr hb, hx⟩)

/-- `transStep` only ever adds pairs: `R` is preserved. -/
theorem subset_transStep {R : List (Fin n × Fin n)} {x : Fin n × Fin n}
    (hx : x ∈ R) : x ∈ transStep R :=
  mem_transStep.mpr (Or.inl hx)

-- ---------------------------------------------------------------------------
-- §2. Iteration structure of `closeSubIter`
-- ---------------------------------------------------------------------------

/-- Unrolling `closeSubIter` from the right. -/
theorem closeSubIter_succ_right (k : Nat) (R : List (Fin n × Fin n)) :
    closeSubIter (k + 1) R = transStep (closeSubIter k R) := by
  induction k generalizing R with
  | zero => rfl
  | succ k ih => exact ih (transStep R)

/-- Iterations compose by addition. -/
theorem closeSubIter_add (a b : Nat) (R : List (Fin n × Fin n)) :
    closeSubIter (a + b) R = closeSubIter a (closeSubIter b R) := by
  induction b generalizing R with
  | zero => rfl
  | succ k ih => exact ih (transStep R)

/-- Every pair of the seed survives `k` iterations (pairs are only added). -/
theorem subset_closeSubIter (k : Nat) {R : List (Fin n × Fin n)} :
    ∀ x ∈ R, x ∈ closeSubIter k R := by
  induction k generalizing R with
  | zero => intro x hx; exact hx
  | succ k ih =>
      intro x hx
      exact ih x (subset_transStep hx)

/-- `closeSubIter k` is monotone in its seed relation. -/
theorem closeSubIter_mono (k : Nat) {R S : List (Fin n × Fin n)}
    (h : ∀ x ∈ R, x ∈ S) : ∀ x ∈ closeSubIter k R, x ∈ closeSubIter k S := by
  induction k generalizing R S with
  | zero => exact h
  | succ k ih =>
      apply ih
      intro x hx
      rw [mem_transStep] at hx ⊢
      cases hx with
      | inl hx => exact Or.inl (h x hx)
      | inr hx =>
          obtain ⟨p, hp, q, hq, h1, h2⟩ := hx
          exact Or.inr ⟨p, h p hp, q, h q hq, h1, h2⟩

/-- More iterations only ever add pairs. -/
theorem closeSubIter_grow (a b : Nat) {R : List (Fin n × Fin n)} {x : Fin n × Fin n}
    (h : x ∈ closeSubIter a R) : x ∈ closeSubIter (a + b) R := by
  rw [closeSubIter_add]
  exact closeSubIter_mono a (subset_closeSubIter b) x h

-- ---------------------------------------------------------------------------
-- §3. The seed relation and its justification
-- ---------------------------------------------------------------------------

/-- The seed relation of `closeSub`, named so proofs can rewrite with it. -/
def seedPairs (t : List (Axiom (Fin n))) : List (Fin n × Fin n) :=
  subPairs t ++ (List.finRange n).map fun c => (c, c)

/-- `closeSub` unfolds to `n + 1` iterations over the seed. -/
theorem closeSub_eq (t : List (Axiom (Fin n))) :
    closeSub t = closeSubIter (n + 1) (seedPairs t) := rfl

/-- A `sub` axiom of the TBox appears as a seed edge. -/
theorem sub_mem_subPairs {t : List (Axiom (Fin n))} {c d : Fin n}
    (h : Axiom.sub c d ∈ t) : (c, d) ∈ subPairs t := by
  rw [subPairs, List.mem_filterMap]
  exact ⟨Axiom.sub c d, h, rfl⟩

/-- Every pair of `subPairs t` comes from a `sub` axiom. -/
theorem mem_subPairs_sub {t : List (Axiom (Fin n))} {c d : Fin n}
    (h : (c, d) ∈ subPairs t) : Axiom.sub c d ∈ t := by
  rw [subPairs, List.mem_filterMap] at h
  obtain ⟨a, ha, hsome⟩ := h
  cases a with
  | sub c' d' =>
      simp only at hsome
      injection hsome with hpd
      injection hpd with hc hd
      subst hc; subst hd
      exact ha
  | disj c' d' =>
      simp only at hsome
      nomatch hsome

/-- A disjointness axiom appears as a directed seed pair (direction 1). -/
theorem disj_mem_disjPairs {t : List (Axiom (Fin n))} {a b : Fin n}
    (h : Axiom.disj a b ∈ t) : (a, b) ∈ disjPairs t := by
  induction t with
  | nil => exact absurd h List.not_mem_nil
  | cons x t ih =>
      cases List.mem_cons.mp h with
      | inl he =>
          subst he
          exact List.mem_cons_self
      | inr ht =>
          cases x with
          | sub c d => exact ih ht
          | disj c d =>
              exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (ih ht))

/-- A disjointness axiom appears as a directed seed pair (direction 2). -/
theorem disj_mem_disjPairs' {t : List (Axiom (Fin n))} {a b : Fin n}
    (h : Axiom.disj b a ∈ t) : (a, b) ∈ disjPairs t := by
  induction t with
  | nil => exact absurd h List.not_mem_nil
  | cons x t ih =>
      cases List.mem_cons.mp h with
      | inl he =>
          subst he
          exact List.mem_cons_of_mem _ (List.mem_cons_self)
      | inr ht =>
          cases x with
          | sub c d => exact ih ht
          | disj c d =>
              exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (ih ht))

/-- Every directed disjointness pair comes from a disjointness axiom, in one
    of the two orientations. -/
theorem mem_disjPairs {t : List (Axiom (Fin n))} {a b : Fin n}
    (h : (a, b) ∈ disjPairs t) : Axiom.disj a b ∈ t ∨ Axiom.disj b a ∈ t := by
  induction t with
  | nil => exact absurd h List.not_mem_nil
  | cons x t ih =>
      cases x with
      | sub c d =>
          cases ih h with
          | inl hd => exact Or.inl (List.mem_cons_of_mem _ hd)
          | inr hd => exact Or.inr (List.mem_cons_of_mem _ hd)
      | disj c d =>
          cases List.mem_cons.mp h with
          | inl he =>
              injection he with hac hbd
              subst hac; subst hbd
              exact Or.inl List.mem_cons_self
          | inr ht =>
              cases List.mem_cons.mp ht with
              | inl he =>
                  injection he with had hbc
                  subst had; subst hbc
                  exact Or.inr List.mem_cons_self
              | inr h3 =>
                  cases ih h3 with
                  | inl hd => exact Or.inl (List.mem_cons_of_mem _ hd)
                  | inr hd => exact Or.inr (List.mem_cons_of_mem _ hd)

-- ---------------------------------------------------------------------------
-- §4. Soundness: every computed pair is `SubDer`-derivable
-- ---------------------------------------------------------------------------

/-- A relation is *justified* when every pair in it is derivable. -/
def Justified (t : List (Axiom (Fin n))) (R : List (Fin n × Fin n)) : Prop :=
  ∀ p ∈ R, SubDer t p.1 p.2

/-- The seed relation is justified: edges by `SubDer.ofAxiom`, diagonal by
    `SubDer.refl`. -/
theorem seed_justified {t : List (Axiom (Fin n))} : Justified t (seedPairs t) := by
  intro p hp
  cases List.mem_append.mp hp with
  | inl h =>
      obtain ⟨c, d⟩ := p
      exact SubDer.ofAxiom (mem_subPairs_sub h)
  | inr h =>
      rw [List.mem_map] at h
      obtain ⟨c, -, hcp⟩ := h
      subst hcp
      exact SubDer.refl

/-- **(1) One-step soundness**: if every pair of `R` is derivable, so is
    every pair added by one transitivity step — the composed pair
    `(p.1, q.2)` is justified by `SubDer.trans` applied to the derivations
    of `p` and `q`. -/
theorem transStep_justified {t : List (Axiom (Fin n))} {R : List (Fin n × Fin n)}
    (h : Justified t R) : Justified t (transStep R) := by
  intro x hx
  rw [mem_transStep] at hx
  cases hx with
  | inl hx => exact h x hx
  | inr hx =>
      obtain ⟨p, hpR, q, hqR, hb, hx⟩ := hx
      subst hx
      have h1 : SubDer t p.1 p.2 := h p hpR
      have h2 : SubDer t q.1 q.2 := h q hqR
      rw [← hb] at h2
      exact SubDer.trans h1 h2

/-- Alias spelling out the reading of (1). -/
theorem transStep_sound {t : List (Axiom (Fin n))} {R : List (Fin n × Fin n)}
    (h : ∀ p ∈ R, SubDer t p.1 p.2) : ∀ p ∈ transStep R, SubDer t p.1 p.2 :=
  transStep_justified h

/-- The justification invariant is preserved by any number of iterations. -/
theorem closeSubIter_justified {t : List (Axiom (Fin n))} (k : Nat)
    {R : List (Fin n × Fin n)} (h : Justified t R) :
    Justified t (closeSubIter k R) := by
  induction k generalizing R with
  | zero => exact h
  | succ k ih => exact ih (transStep_justified h)

/-- Every pair of the full closure is derivable. -/
theorem closeSub_justified {t : List (Axiom (Fin n))} : Justified t (closeSub t) := by
  rw [closeSub_eq]
  exact closeSubIter_justified _ seed_justified

/-- **(2) Closure soundness**: the boolean check only accepts derivable
    subsumptions. -/
theorem subB_sound {t : List (Axiom (Fin n))} {c d : Fin n} (h : subB t c d = true) :
    SubDer t c d :=
  closeSub_justified (c, d) (List.mem_of_elem_eq_true h)

-- ---------------------------------------------------------------------------
-- §5. Walks: linearising `SubDer` derivations
-- ---------------------------------------------------------------------------

/-- A concrete walk through `sub` axioms: `WalkL t c l d` reads "starting
    from `c`, following `sub` axioms through the node list `l` ends at `d`".
    `l` contains every node after `c`, so its last element (when nonempty)
    is `d`. -/
inductive WalkL (t : List (Axiom (Fin n))) : Fin n → List (Fin n) → Fin n → Prop
  | nil (c : Fin n) : WalkL t c [] c
  | cons (c d e : Fin n) (l : List (Fin n)) :
      Axiom.sub c d ∈ t → WalkL t d l e → WalkL t c (d :: l) e

/-- Walks concatenate. -/
theorem walkL_append {t : List (Axiom (Fin n))} {c d e : Fin n}
    {l1 l2 : List (Fin n)} (w1 : WalkL t c l1 d) :
    WalkL t d l2 e → WalkL t c (l1 ++ l2) e := by
  induction w1 with
  | nil c => exact id
  | cons c d' e' l' hax w ih =>
      intro w2
      exact WalkL.cons c d' e (l' ++ l2) hax (ih w2)

/-- Every derivation linearises to a walk. -/
theorem subDer_walkL {t : List (Axiom (Fin n))} {c d : Fin n} (h : SubDer t c d) :
    ∃ l, WalkL t c l d := by
  induction h with
  | ofAxiom hmem => exact ⟨[_], WalkL.cons _ _ _ _ hmem (WalkL.nil _)⟩
  | refl => exact ⟨[], WalkL.nil _⟩
  | trans h1 h2 ih1 ih2 =>
      obtain ⟨l1, w1⟩ := ih1
      obtain ⟨l2, w2⟩ := ih2
      exact ⟨l1 ++ l2, walkL_append w1 w2⟩

/-- If the node `x` occurs inside a walk, the suffix from that occurrence is
    a strictly shorter walk from `x` to the end. -/
theorem walkL_split {t : List (Axiom (Fin n))} {a b x : Fin n} {l : List (Fin n)}
    (w : WalkL t a l b) :
    x ∈ l → ∃ l₂, WalkL t x l₂ b ∧ l₂.length < l.length ∧ ∀ y ∈ l₂, y ∈ l := by
  induction w with
  | nil c =>
      intro hx
      exact absurd hx List.not_mem_nil
  | cons c d e l hax w ih =>
      intro hx
      cases List.mem_cons.mp hx with
      | inl hxd =>
          subst hxd
          exact ⟨l, w, Nat.lt_succ_self _, fun y hy => List.mem_cons_of_mem _ hy⟩
      | inr hxl =>
          obtain ⟨l₂, w₂, hlen, hsub⟩ := ih hxl
          exact ⟨l₂, w₂, Nat.lt_trans hlen (Nat.lt_succ_self _), fun y hy =>
            List.mem_cons_of_mem _ (hsub y hy)⟩

/-- **Chain shortening**: every walk can be replaced by a walk whose node
    list `c :: l'` has no duplicates (loops are cut via `walkL_split`), with
    nodes drawn from the original walk. Proved by strong induction on the
    walk length (fuel `m`). -/
theorem walkL_shorten_aux {t : List (Axiom (Fin n))} :
    ∀ (m : Nat) {c d : Fin n} {l : List (Fin n)},
      WalkL t c l d → l.length ≤ m →
      ∃ l', WalkL t c l' d ∧ (c :: l').Nodup ∧ ∀ y ∈ l', y ∈ l := by
  intro m
  induction m with
  | zero =>
      intro c d l w hlen
      cases w with
      | nil c =>
          exact ⟨[], WalkL.nil _, List.nodup_cons.mpr ⟨List.not_mem_nil, List.nodup_nil⟩,
            fun y hy => absurd hy List.not_mem_nil⟩
      | cons _ d' _ l' hax w =>
          simp only [List.length_cons] at hlen
          exact absurd hlen (Nat.not_succ_le_zero _)
  | succ m ih =>
      intro c d l w hlen
      cases w with
      | nil c =>
          exact ⟨[], WalkL.nil _, List.nodup_cons.mpr ⟨List.not_mem_nil, List.nodup_nil⟩,
            fun y hy => absurd hy List.not_mem_nil⟩
      | cons _ d' _ l' hax w =>
          simp only [List.length_cons] at hlen
          have hlen' : l'.length ≤ m := Nat.le_of_succ_le_succ hlen
          by_cases hcd : c = d'
          · subst hcd
            obtain ⟨l₂, w₂, hnd, hsub⟩ := ih w hlen'
            exact ⟨l₂, w₂, hnd, fun y hy => List.mem_cons_of_mem _ (hsub y hy)⟩
          · by_cases hc : c ∈ l'
            · obtain ⟨l₂, w₂, hlt, hsub⟩ := walkL_split w hc
              have hlt' : l₂.length ≤ m := Nat.le_of_lt (Nat.lt_of_lt_of_le hlt hlen')
              obtain ⟨l₃, w₃, hnd, hsub₃⟩ := ih w₂ hlt'
              exact ⟨l₃, w₃, hnd, fun y hy =>
                List.mem_cons_of_mem _ (hsub _ (hsub₃ y hy))⟩
            · obtain ⟨l₂, w₂, hnd, hsub⟩ := ih w hlen'
              refine ⟨d' :: l₂, WalkL.cons c d' d l₂ hax w₂, ?_, ?_⟩
              · rw [List.nodup_cons]
                constructor
                · intro hmem
                  cases List.mem_cons.mp hmem with
                  | inl heq => exact hcd heq
                  | inr hin => exact hc (hsub _ hin)
                · exact hnd
              · intro y hy
                cases List.mem_cons.mp hy with
                | inl heq => subst heq; exact List.mem_cons_self
                | inr hin => exact List.mem_cons_of_mem _ (hsub _ hin)

-- ---------------------------------------------------------------------------
-- §6. A duplicate-free list over `Fin n` has at most `n` elements
-- ---------------------------------------------------------------------------

/-- A duplicate-free list whose elements all come from `m` is no longer
    than `m`. -/
theorem nodup_sub_length {α : Type} [BEq α] [LawfulBEq α] {l m : List α}
    (h : l.Nodup) (hs : ∀ x ∈ l, x ∈ m) : l.length ≤ m.length := by
  induction l generalizing m with
  | nil => exact Nat.zero_le _
  | cons a l ih =>
      have ha : a ∈ m := hs a (List.mem_cons_self)
      have hnl : l.Nodup := (List.nodup_cons.mp h).2
      have han : a ∉ l := (List.nodup_cons.mp h).1
      have hsub : ∀ x ∈ l, x ∈ m.erase a := fun x hx =>
        (List.mem_erase_of_ne (fun he : x = a => han (he ▸ hx))).mpr
          (hs x (List.mem_cons_of_mem _ hx))
      have hlen := ih hnl hsub
      rw [List.length_erase_of_mem ha] at hlen
      have hm : 0 < m.length := List.length_pos_of_mem ha
      simp only [List.length_cons]
      omega

/-- A duplicate-free list of `Fin n` has at most `n` elements. -/
theorem nodup_length_le_finRange {l : List (Fin n)} (h : l.Nodup) : l.length ≤ n := by
  have h2 := nodup_sub_length h (m := List.finRange n) (fun x _ => List.mem_finRange x)
  rwa [List.length_finRange] at h2

-- ---------------------------------------------------------------------------
-- §7. Completeness: every derivation is computed
-- ---------------------------------------------------------------------------

/-- A walk of `m` edges lands in the `m`-fold iteration of the seed:
    each `cons` step prepends a seed edge to a chain already present, and one
    `transStep` composes them. -/
theorem walkL_mem_closeSubIter {t : List (Axiom (Fin n))} {c d : Fin n}
    {l : List (Fin n)} (w : WalkL t c l d) :
    (c, d) ∈ closeSubIter l.length (seedPairs t) := by
  induction w with
  | nil c =>
      show (c, c) ∈ seedPairs t
      exact List.mem_append_right _ (List.mem_map.mpr ⟨c, List.mem_finRange c, rfl⟩)
  | cons c d e l hax w ih =>
      show (c, e) ∈ closeSubIter (l.length + 1) (seedPairs t)
      rw [closeSubIter_succ_right]
      apply mem_transStep.mpr
      apply Or.inr
      refine ⟨(c, d), ?_, (d, e), ?_, rfl, rfl⟩
      · exact subset_closeSubIter _ _ (List.mem_append_left _ (sub_mem_subPairs hax))
      · exact ih

/-- **(3) Completeness**: every derivable subsumption is computed by `subB`.
    The derivation linearises to a walk; the walk shortens to a
    duplicate-free one, which has at most `n` nodes, i.e. at most `n - 1`
    edges — within the `n + 1` iterations of `closeSub`. -/
theorem subB_complete {t : List (Axiom (Fin n))} {c d : Fin n} (h : SubDer t c d) :
    subB t c d = true := by
  obtain ⟨l, w⟩ := subDer_walkL h
  obtain ⟨l', w', hnd, -⟩ := walkL_shorten_aux l.length w (Nat.le_refl l.length)
  have hlen : (c :: l').length ≤ n := nodup_length_le_finRange hnd
  have hll : l'.length ≤ n + 1 := by
    simp only [List.length_cons] at hlen
    omega
  have hmem : (c, d) ∈ closeSubIter l'.length (seedPairs t) := walkL_mem_closeSubIter w'
  have hmem2 : (c, d) ∈ closeSubIter (n + 1) (seedPairs t) := by
    have hg := closeSubIter_grow l'.length ((n + 1) - l'.length) hmem
    rwa [Nat.add_sub_cancel' hll] at hg
  have hmem3 : (c, d) ∈ closeSub t := closeSub_eq t ▸ hmem2
  exact List.elem_eq_true_of_mem hmem3

/-- Convenience: a derivation gives membership in the computed closure. -/
theorem mem_closeSub_of_subDer {t : List (Axiom (Fin n))} {c d : Fin n}
    (h : SubDer t c d) : (c, d) ∈ closeSub t :=
  List.mem_of_elem_eq_true (subB_complete h)

/-- **(4) The boolean closure is exactly the deductive system.** -/
theorem subB_iff_subDer {t : List (Axiom (Fin n))} {c d : Fin n} :
    subB t c d = true ↔ SubDer t c d :=
  ⟨subB_sound, subB_complete⟩

-- ---------------------------------------------------------------------------
-- §8. The boolean conflict oracle is exactly `DerivedConflict`
-- ---------------------------------------------------------------------------

/-- **(5) Oracle soundness, boolean side**: a computed conflict is a genuine
    semantic conflict (via `subB_sound` on both closure pairs). -/
theorem conflictB_sound {t : List (Axiom (Fin n))} {c c' : Fin n}
    (h : conflictB t c c' = true) : DerivedConflict t c c' := by
  have h' : ((closeSub t).any fun p => (closeSub t).any fun q =>
      (p.1 == c) && (q.1 == c') && (disjPairs t).elem (p.2, q.2)) = true := h
  rw [List.any_eq_true] at h'
  obtain ⟨p, hpR, h'⟩ := h'
  rw [List.any_eq_true] at h'
  obtain ⟨q, hqR, h'⟩ := h'
  simp only [Bool.and_eq_true, beq_iff_eq] at h'
  have hpc : p.1 = c := h'.1.1
  have hqc : q.1 = c' := h'.1.2
  have hD : (disjPairs t).elem (p.2, q.2) = true := h'.2
  have h1 : SubDer t c p.2 := by
    have hs := closeSub_justified p hpR
    rw [hpc] at hs
    exact hs
  have h2 : SubDer t c' q.2 := by
    have hs := closeSub_justified q hqR
    rw [hqc] at hs
    exact hs
  have hDmem : (p.2, q.2) ∈ disjPairs t := List.mem_of_elem_eq_true hD
  cases mem_disjPairs hDmem with
  | inl hd => exact ⟨p.2, q.2, h1, h2, Or.inl hd⟩
  | inr hd => exact ⟨p.2, q.2, h1, h2, Or.inr hd⟩

/-- **Oracle completeness, boolean side**: every semantic conflict is
    computed (via `subB_complete` on both derivations). -/
theorem conflictB_complete {t : List (Axiom (Fin n))} {c c' : Fin n}
    (h : DerivedConflict t c c') : conflictB t c c' = true := by
  obtain ⟨d₁, d₂, h1, h2, hd⟩ := h
  have hp : (c, d₁) ∈ closeSub t := mem_closeSub_of_subDer h1
  have hq : (c', d₂) ∈ closeSub t := mem_closeSub_of_subDer h2
  have hD : (d₁, d₂) ∈ disjPairs t := by
    cases hd with
    | inl h12 => exact disj_mem_disjPairs h12
    | inr h21 => exact disj_mem_disjPairs' h21
  show ((closeSub t).any fun p => (closeSub t).any fun q =>
      (p.1 == c) && (q.1 == c') && (disjPairs t).elem (p.2, q.2)) = true
  rw [List.any_eq_true]
  refine ⟨(c, d₁), hp, ?_⟩
  rw [List.any_eq_true]
  refine ⟨(c', d₂), hq, ?_⟩
  show ((c == c) && (c' == c') && (disjPairs t).elem (d₁, d₂)) = true
  rw [Bool.and_eq_true, Bool.and_eq_true]
  exact ⟨⟨beq_self_eq_true _, beq_self_eq_true _⟩, List.elem_eq_true_of_mem hD⟩

/-- **The boolean oracle is exactly the semantic conflict relation.** -/
theorem conflictB_iff {t : List (Axiom (Fin n))} {c c' : Fin n} :
    conflictB t c c' = true ↔ DerivedConflict t c c' :=
  ⟨conflictB_sound, conflictB_complete⟩

-- ---------------------------------------------------------------------------
-- §9. The `Fin 8` biomedical instance, re-checked through the verification
-- ---------------------------------------------------------------------------

section BiomedVerified

/-- Lymphokine ⊑* Molecule is computed (boolean side). -/
theorem cv_subB_0_2 : subB biomedTBox 0 2 = true := by native_decide

/-- The same fact, now *derived* through the verified soundness direction —
    the boolean computation and the two-hop `SubDer` derivation of
    `OntologyELReasoner.lean` §7 agree. -/
theorem cv_subder_0_2 : SubDer biomedTBox 0 2 := subB_sound cv_subB_0_2

/-- TherapeuticFusionProtein ⊑* Molecule, via the boolean side. -/
theorem cv_subder_7_2 : SubDer biomedTBox 7 2 :=
  subB_sound (by native_decide)

/-- Organ ⊑* SkeletalMuscleOnly is NOT computed, and (by the contrapositive
    of completeness) NOT derivable either: disjointness is not subsumption. -/
theorem cv_subB_5_6_false : subB biomedTBox 5 6 = false := by native_decide

theorem cv_no_sub_5_6 : ¬ SubDer biomedTBox 5 6 := by
  intro h
  have h1 := subB_complete h
  rw [cv_subB_5_6_false] at h1
  exact Bool.noConfusion h1

/-- The fusion protein vs lymphokine conflict is computed... -/
theorem cv_conflictB_7_0 : conflictB biomedTBox 7 0 = true := by native_decide

/-- ...and is a genuine semantic conflict, re-derived through
    `conflictB_sound` rather than the constructor-built witness of §7. -/
theorem cv_derived_7_0 : DerivedConflict biomedTBox 7 0 :=
  conflictB_sound cv_conflictB_7_0

/-- The incoherent fusion protein self-conflicts, boolean side. -/
theorem cv_derived_7_7 : DerivedConflict biomedTBox 7 7 :=
  conflictB_sound (by native_decide)

/-- Lymphokine vs Heart: no conflict, on either side. -/
theorem cv_conflictB_0_4 : conflictB biomedTBox 0 4 = false := by native_decide

theorem cv_no_conflict_0_4 : ¬ DerivedConflict biomedTBox 0 4 := by
  intro h
  have h1 := conflictB_complete h
  rw [cv_conflictB_0_4] at h1
  exact Bool.noConfusion h1

/-- Symmetry through the boolean oracle, matching `check_conflictB_0_7`. -/
theorem cv_derived_0_7 : DerivedConflict biomedTBox 0 7 :=
  conflictB_sound (by native_decide)

end BiomedVerified

end Sounio.OntologyELReasoner
