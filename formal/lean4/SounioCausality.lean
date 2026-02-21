import SounioEffects

/-!
# Sounio Causal Type Theory — Lean 4 Formalization

Formalizes Pearl's structural causal models (SCMs) as a type-level discipline for
the Sounio language, connecting causal inference primitives to the algebraic
effect system defined in `SounioEffects.lean`.

References:
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. 2nd ed. Cambridge.
- Pearl, J. (2012). "The Do-Calculus Revisited." UAI.
- Spirtes, P., Glymour, C., Scheines, R. (2000). *Causation, Prediction, and Search*.
- Bareinboim, E. and Pearl, J. (2016). "Causal inference and the data-fusion problem."

All proofs complete — no `sorry`, no Mathlib.
-/

open Sounio.Effects

namespace Sounio.Causality

variable {n : Nat}

-- ================================================================
-- §1. Causal Variables and Graph Structure
-- ================================================================

abbrev CVar (n : Nat) := Fin n
abbrev EdgeSet (n : Nat) := Fin n → Fin n → Bool

structure SCM (n : Nat) where
  edges : EdgeSet n

def isTopological (edges : EdgeSet n) : Prop :=
  ∀ i j, edges i j = true → i.val < j.val

-- ================================================================
-- §2. Basic Graph Properties
-- ================================================================

def parents (edges : EdgeSet n) (j : Fin n) : Fin n → Bool := fun i => edges i j
def children (edges : EdgeSet n) (i : Fin n) : Fin n → Bool := fun j => edges i j
def isParent (edges : EdgeSet n) (i j : Fin n) : Prop := edges i j = true

theorem topo_no_self_loop (edges : EdgeSet n) (htopo : isTopological edges)
    (i : Fin n) : edges i i = false := by
  cases h : edges i i with
  | false => rfl
  | true => exact absurd (htopo i i h) (Nat.lt_irrefl _)

theorem topo_no_cycle_2 (edges : EdgeSet n) (htopo : isTopological edges)
    (i j : Fin n) (hij : edges i j = true) (hji : edges j i = true) : False :=
  Nat.lt_asymm (htopo i j hij) (htopo j i hji)

theorem parent_lt (edges : EdgeSet n) (htopo : isTopological edges)
    (i j : Fin n) (hp : isParent edges i j) : i.val < j.val :=
  htopo i j hp

-- ================================================================
-- §3. Reachability (Ancestral Relation)
-- ================================================================

inductive Reachable (edges : EdgeSet n) : Fin n → Fin n → Prop where
  | direct {i j : Fin n} : edges i j = true → Reachable edges i j
  | step {i k j : Fin n} : Reachable edges i k → edges k j = true →
      Reachable edges i j

theorem reachable_trans (edges : EdgeSet n) {i k j : Fin n}
    (hik : Reachable edges i k) (hkj : Reachable edges k j) :
    Reachable edges i j := by
  induction hkj generalizing i with
  | direct h => exact Reachable.step hik h
  | step _ hmj ih => exact Reachable.step (ih hik) hmj

theorem reachable_lt (edges : EdgeSet n) (htopo : isTopological edges)
    {i j : Fin n} (hr : Reachable edges i j) : i.val < j.val := by
  induction hr with
  | direct h => exact htopo _ _ h
  | step _ hkj ih => exact Nat.lt_trans ih (htopo _ _ hkj)

theorem reachable_not_self (edges : EdgeSet n) (htopo : isTopological edges)
    (i : Fin n) : ¬Reachable edges i i :=
  fun hr => Nat.lt_irrefl i.val (reachable_lt edges htopo hr)

theorem reachable_asymm (edges : EdgeSet n) (htopo : isTopological edges)
    {i j : Fin n} (hij : Reachable edges i j) : ¬Reachable edges j i :=
  fun hji => Nat.lt_asymm (reachable_lt edges htopo hij)
                           (reachable_lt edges htopo hji)

theorem edge_implies_reachable (edges : EdgeSet n) {i j : Fin n}
    (h : edges i j = true) : Reachable edges i j :=
  Reachable.direct h

theorem reachable_monotone {e1 e2 : EdgeSet n} {i j : Fin n}
    (hsub : ∀ a b, e1 a b = true → e2 a b = true)
    (hr : Reachable e1 i j) : Reachable e2 i j := by
  induction hr with
  | direct h => exact Reachable.direct (hsub _ _ h)
  | step _ hkj ih => exact Reachable.step ih (hsub _ _ hkj)

-- ================================================================
-- §4. Intervention (do-Operator)
-- ================================================================

def intervene (edges : EdgeSet n) (t : Fin n) : EdgeSet n :=
  fun i j => if j = t then false else edges i j

theorem intervene_removes_parents (edges : EdgeSet n) (t i : Fin n) :
    intervene edges t i t = false := by simp [intervene]

theorem intervene_preserves_other (edges : EdgeSet n) (t i j : Fin n)
    (hne : j ≠ t) : intervene edges t i j = edges i j := by
  simp [intervene, hne]

theorem intervene_idempotent (edges : EdgeSet n) (t : Fin n) :
    intervene (intervene edges t) t = intervene edges t := by
  funext i j; simp only [intervene]; split <;> simp

theorem intervene_comm (edges : EdgeSet n) (s t : Fin n) :
    intervene (intervene edges s) t = intervene (intervene edges t) s := by
  funext i j; simp only [intervene]
  by_cases hjt : j = t <;> by_cases hjs : j = s <;> simp [hjt, hjs]

theorem intervene_preserves_topo (edges : EdgeSet n) (t : Fin n)
    (htopo : isTopological edges) : isTopological (intervene edges t) := by
  intro i j h; unfold intervene at h
  by_cases hj : j = t
  · simp [hj] at h
  · simp [hj] at h; exact htopo i j h

theorem intervene_no_parents (edges : EdgeSet n) (t i : Fin n) :
    parents (intervene edges t) t i = false := by simp [parents, intervene]

theorem intervene_preserves_children (edges : EdgeSet n) (t j : Fin n)
    (hne : j ≠ t) : children (intervene edges t) t j = children edges t j := by
  simp [children, intervene, hne]

/-- Nothing can reach the target of an intervention (last edge to t is always false). -/
theorem intervene_blocks_incoming (edges : EdgeSet n) (t : Fin n)
    {x : Fin n} (hr : Reachable (intervene edges t) x t) : False := by
  cases hr with
  | direct h => simp [intervene] at h
  | step _ hkt => simp [intervene] at hkt

theorem intervene_noop_no_parents (edges : EdgeSet n) (t : Fin n)
    (h : ∀ i, edges i t = false) : intervene edges t = edges := by
  funext i j; simp only [intervene]
  by_cases hj : j = t
  · subst hj; simp [h]
  · simp [hj]

/-- Intervention only removes edges. -/
theorem intervene_sub (edges : EdgeSet n) (t i j : Fin n) :
    intervene edges t i j = true → edges i j = true := by
  unfold intervene; by_cases hj : j = t <;> simp [hj]

-- ================================================================
-- §5. Variable Sets and d-Separation
-- ================================================================

abbrev VarSet (n : Nat) := Fin n → Bool
def emptySet : VarSet n := fun _ => false
def singletonSet (v : Fin n) : VarSet n := fun w => decide (w = v)
def setUnion (s₁ s₂ : VarSet n) : VarSet n := fun v => s₁ v || s₂ v

def dSep (edges : EdgeSet n) (x y : Fin n) (z : VarSet n) : Prop :=
  ¬Reachable (fun i j => edges i j && !z j) x y

theorem dsep_empty_no_path (edges : EdgeSet n) (x y : Fin n) :
    dSep edges x y emptySet ↔ ¬Reachable edges x y := by
  constructor
  · intro hdsep hreach
    apply hdsep
    exact reachable_monotone (fun a b h => by simp [emptySet, h]) hreach
  · intro hnr hreach
    apply hnr
    exact reachable_monotone (fun a b h => by simp [emptySet] at h; exact h) hreach

theorem dsep_after_intervene (edges : EdgeSet n) (t x : Fin n) :
    dSep (intervene edges t) x t emptySet := by
  intro hreach
  exact intervene_blocks_incoming edges t
    (reachable_monotone (fun a b h => by simp [emptySet] at h; exact h) hreach)

-- ================================================================
-- §6. Counterfactual Worlds
-- ================================================================

def counterfactualWorld (edges : EdgeSet n) (interventions : List (Fin n)) :
    EdgeSet n :=
  interventions.foldl intervene edges

theorem counterfactual_empty (edges : EdgeSet n) :
    counterfactualWorld edges [] = edges := rfl

theorem counterfactual_single (edges : EdgeSet n) (t : Fin n) :
    counterfactualWorld edges [t] = intervene edges t := rfl

theorem foldl_intervene_preserves_topo (edges : EdgeSet n) (ts : List (Fin n))
    (htopo : isTopological edges) :
    isTopological (ts.foldl intervene edges) := by
  induction ts generalizing edges with
  | nil => exact htopo
  | cons t ts ih => exact ih (intervene edges t) (intervene_preserves_topo edges t htopo)

theorem counterfactual_preserves_topo (edges : EdgeSet n)
    (interventions : List (Fin n)) (htopo : isTopological edges) :
    isTopological (counterfactualWorld edges interventions) :=
  foldl_intervene_preserves_topo edges interventions htopo

theorem counterfactual_cons (edges : EdgeSet n) (t : Fin n) (ts : List (Fin n)) :
    counterfactualWorld edges (t :: ts) =
    counterfactualWorld (intervene edges t) ts := rfl

theorem counterfactual_append_single (edges : EdgeSet n) (ts : List (Fin n))
    (t : Fin n) :
    counterfactualWorld edges (ts ++ [t]) =
    intervene (counterfactualWorld edges ts) t := by
  simp [counterfactualWorld, List.foldl_append]

theorem counterfactual_pair_comm (edges : EdgeSet n) (s t : Fin n) :
    counterfactualWorld edges [s, t] = counterfactualWorld edges [t, s] := by
  simp [counterfactualWorld, List.foldl]
  exact intervene_comm edges s t

/-- foldl intervene can only remove edges. -/
theorem foldl_intervene_sub (edges : EdgeSet n) (ts : List (Fin n)) (i j : Fin n) :
    (ts.foldl intervene edges) i j = true → edges i j = true := by
  induction ts generalizing edges with
  | nil => exact id
  | cons t ts ih =>
    intro h
    exact intervene_sub edges t i j (ih (intervene edges t) h)

/-- foldl intervene kills all incoming edges to listed targets. -/
theorem foldl_intervene_removes (edges : EdgeSet n) (ts : List (Fin n))
    (t : Fin n) (hmem : t ∈ ts) (i : Fin n) :
    (ts.foldl intervene edges) i t = false := by
  induction ts generalizing edges with
  | nil => contradiction
  | cons s rest ih =>
    simp only [List.foldl]
    cases hmem with
    | head =>
      cases h : (rest.foldl intervene (intervene edges t)) i t with
      | false => rfl
      | true =>
        have := foldl_intervene_sub (intervene edges t) rest i t h
        simp [intervene] at this
    | tail _ hmem' =>
      exact ih (intervene edges s) hmem'

/-- Nothing reaches any intervened variable in a counterfactual world. -/
theorem counterfactual_blocks_all (edges : EdgeSet n) (ts : List (Fin n))
    (t : Fin n) (hmem : t ∈ ts) {x : Fin n}
    (hr : Reachable (counterfactualWorld edges ts) x t) : False := by
  have hf := foldl_intervene_removes edges ts t hmem
  unfold counterfactualWorld at hr
  cases hr with
  | direct h => rw [hf] at h; exact Bool.noConfusion h
  | step _ hkt => rw [hf] at hkt; exact Bool.noConfusion hkt

-- ================================================================
-- §7. Pearl's Causal Hierarchy
-- ================================================================

def rule1Condition (edges : EdgeSet n) (x z : Fin n) : Prop :=
  dSep (intervene edges x) x z emptySet

def rule3Condition (edges : EdgeSet n) (x z : Fin n) : Prop :=
  ¬Reachable edges x z

theorem rule1_self (edges : EdgeSet n) (x : Fin n) :
    rule1Condition edges x x :=
  dsep_after_intervene edges x x

theorem rule3_implies_rule1 (edges : EdgeSet n)
    (x z : Fin n) (hr3 : rule3Condition edges x z) :
    rule1Condition edges x z := by
  unfold rule1Condition dSep
  intro hreach
  apply hr3
  exact reachable_monotone
    (fun a b h => by
      simp [emptySet] at h
      exact intervene_sub edges x a b h)
    hreach

-- ================================================================
-- §8. Causal Strength
-- ================================================================

def causalStrength (edges : EdgeSet n) (i j : Fin n) : Nat :=
  if edges i j = true then 1 else 0

theorem causal_strength_zero (edges : EdgeSet n) (i j : Fin n)
    (h : edges i j = false) : causalStrength edges i j = 0 := by
  simp [causalStrength, h]

theorem causal_strength_one (edges : EdgeSet n) (i j : Fin n)
    (h : edges i j = true) : causalStrength edges i j = 1 := by
  simp [causalStrength, h]

theorem intervene_removes_strength (edges : EdgeSet n) (t i : Fin n) :
    causalStrength (intervene edges t) i t = 0 := by
  simp [causalStrength, intervene]

theorem intervene_preserves_strength (edges : EdgeSet n) (t i j : Fin n)
    (hne : j ≠ t) :
    causalStrength (intervene edges t) i j = causalStrength edges i j := by
  simp [causalStrength, intervene, hne]

-- ================================================================
-- §9. Effect System Connection
-- ================================================================

def causalEffectRow : EffectRow := singleRow .Mut
def observationalRow : EffectRow := pureRow

theorem intervention_requires_mut : .Mut ∈ᵣ causalEffectRow :=
  singleRow_member .Mut

theorem observational_subrow_causal :
    effectSubrow observationalRow causalEffectRow :=
  pure_is_subrow causalEffectRow

theorem observational_is_pure : observationalRow = pureRow := rfl

theorem handle_intervention_pure :
    mask causalEffectRow .Mut = pureRow :=
  single_mask_pure .Mut

theorem intervention_no_io : ¬(.IO ∈ᵣ causalEffectRow) :=
  singleRow_not_member .IO .Mut (by decide)

-- ================================================================
-- §10. Graph Construction
-- ================================================================

def emptyGraph : EdgeSet n := fun _ _ => false

def addEdge (edges : EdgeSet n) (i j : Fin n) : EdgeSet n :=
  fun a b => if a = i && b = j then true else edges a b

theorem emptyGraph_no_edges (i j : Fin n) :
    (emptyGraph : EdgeSet n) i j = false := rfl

theorem emptyGraph_topo : isTopological (emptyGraph : EdgeSet n) := by
  intro i j h; simp [emptyGraph] at h

theorem addEdge_creates_edge (edges : EdgeSet n) (i j : Fin n) :
    addEdge edges i j i j = true := by simp [addEdge]

theorem addEdge_preserves (edges : EdgeSet n) (i j a b : Fin n)
    (h : edges a b = true) : addEdge edges i j a b = true := by
  simp only [addEdge]; split <;> simp [h]

theorem addEdge_preserves_topo (edges : EdgeSet n) (i j : Fin n)
    (htopo : isTopological edges) (hlt : i.val < j.val) :
    isTopological (addEdge edges i j) := by
  intro a b h
  unfold addEdge at h
  split at h
  · next hab =>
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hab
    rw [hab.1, hab.2]; exact hlt
  · exact htopo a b h

-- ================================================================
-- §11. Graph Removal and Isolation
-- ================================================================

def removeOutgoing (edges : EdgeSet n) (s : Fin n) : EdgeSet n :=
  fun i j => if i = s then false else edges i j

theorem removeOutgoing_preserves_topo (edges : EdgeSet n) (s : Fin n)
    (htopo : isTopological edges) :
    isTopological (removeOutgoing edges s) := by
  intro i j h; unfold removeOutgoing at h
  by_cases hi : i = s
  · simp [hi] at h
  · simp [hi] at h; exact htopo i j h

theorem removeOutgoing_sub (edges : EdgeSet n) (s i j : Fin n) :
    removeOutgoing edges s i j = true → edges i j = true := by
  unfold removeOutgoing; by_cases hi : i = s <;> simp [hi]

/-- Nothing can reach any node from s when s's outgoing edges are removed. -/
theorem removeOutgoing_no_direct (edges : EdgeSet n) (s j : Fin n) :
    removeOutgoing edges s s j = false := by simp [removeOutgoing]

/-- Nothing is reachable from s after removing all outgoing edges of s. -/
theorem removeOutgoing_blocks (edges : EdgeSet n) (s : Fin n)
    {j : Fin n} (hr : Reachable (removeOutgoing edges s) s j) : False := by
  induction hr with
  | direct h => simp [removeOutgoing] at h
  | step _ _ ih => exact ih

theorem isolate_comm (edges : EdgeSet n) (s : Fin n) :
    intervene (removeOutgoing edges s) s =
    removeOutgoing (intervene edges s) s := by
  funext i j; simp only [intervene, removeOutgoing]
  by_cases hj : j = s <;> by_cases hi : i = s <;> simp [hj, hi]

-- ================================================================
-- §12. Confounders
-- ================================================================

def isConfounder (edges : EdgeSet n) (c x y : Fin n) : Prop :=
  Reachable edges c x ∧ Reachable edges c y

theorem confounder_lt_both (edges : EdgeSet n) (htopo : isTopological edges)
    (c x y : Fin n) (hconf : isConfounder edges c x y) :
    c.val < x.val ∧ c.val < y.val :=
  And.intro (reachable_lt edges htopo hconf.1) (reachable_lt edges htopo hconf.2)

theorem no_self_confounding (edges : EdgeSet n) (htopo : isTopological edges)
    (x y : Fin n) : ¬isConfounder edges x x y :=
  fun hconf => reachable_not_self edges htopo x hconf.1

theorem nothing_confounds_intervened_first (edges : EdgeSet n) (t : Fin n)
    (c y : Fin n) : ¬isConfounder (intervene edges t) c t y :=
  fun hconf => intervene_blocks_incoming edges t hconf.1

theorem nothing_confounds_intervened_second (edges : EdgeSet n) (t : Fin n)
    (c x : Fin n) : ¬isConfounder (intervene edges t) c x t :=
  fun hconf => intervene_blocks_incoming edges t hconf.2

-- ================================================================
-- §13. Markov Blanket
-- ================================================================

def markovBlanket (edges : EdgeSet n) (v : Fin n) : VarSet n :=
  fun w => edges w v || edges v w

theorem child_in_parent_blanket (edges : EdgeSet n) (i j : Fin n)
    (h : edges i j = true) : markovBlanket edges i j = true := by
  simp [markovBlanket, h]

theorem parent_in_child_blanket (edges : EdgeSet n) (i j : Fin n)
    (h : edges i j = true) : markovBlanket edges j i = true := by
  simp [markovBlanket, h]

theorem isolated_empty_blanket (edges : EdgeSet n) (v : Fin n)
    (hno_in : ∀ i, edges i v = false) (hno_out : ∀ j, edges v j = false) :
    markovBlanket edges v = emptySet := by
  funext w; simp [markovBlanket, emptySet, hno_in, hno_out]

-- ================================================================
-- §14. Identifiability
-- ================================================================

def isIdentifiable (edges : EdgeSet n) (x y : Fin n) : Prop :=
  ∀ c, ¬isConfounder edges c x y

theorem intervene_makes_identifiable (edges : EdgeSet n) (x y : Fin n) :
    isIdentifiable (intervene edges x) x y :=
  fun _ hconf => intervene_blocks_incoming edges x hconf.1

theorem emptyGraph_identifiable (x y : Fin n) :
    isIdentifiable (emptyGraph : EdgeSet n) x y := by
  intro _ hconf
  cases hconf.1 with
  | direct h => simp [emptyGraph] at h
  | step _ hkj => simp [emptyGraph] at hkj

-- ================================================================
-- §15. Causal Equivalence
-- ================================================================

def causalEquiv (e1 e2 : EdgeSet n) : Prop :=
  ∀ i j, Reachable e1 i j ↔ Reachable e2 i j

theorem causalEquiv_refl (edges : EdgeSet n) : causalEquiv edges edges :=
  fun _ _ => Iff.rfl

theorem causalEquiv_symm {e1 e2 : EdgeSet n} (h : causalEquiv e1 e2) :
    causalEquiv e2 e1 :=
  fun i j => (h i j).symm

theorem causalEquiv_trans {e1 e2 e3 : EdgeSet n}
    (h12 : causalEquiv e1 e2) (h23 : causalEquiv e2 e3) :
    causalEquiv e1 e3 :=
  fun i j => Iff.trans (h12 i j) (h23 i j)

theorem eq_implies_causalEquiv {e1 e2 : EdgeSet n} (h : e1 = e2) :
    causalEquiv e1 e2 := by subst h; exact causalEquiv_refl e1

-- ================================================================
-- §16. Mediators
-- ================================================================

def isMediator (edges : EdgeSet n) (x m y : Fin n) : Prop :=
  edges x m = true ∧ edges m y = true

theorem mediator_creates_path (edges : EdgeSet n) {x m y : Fin n}
    (hmed : isMediator edges x m y) : Reachable edges x y :=
  Reachable.step (Reachable.direct hmed.1) hmed.2

theorem mediator_ordered (edges : EdgeSet n) (htopo : isTopological edges)
    {x m y : Fin n} (hmed : isMediator edges x m y) :
    x.val < m.val ∧ m.val < y.val :=
  And.intro (htopo x m hmed.1) (htopo m y hmed.2)

-- ================================================================
-- §17. Composition and Chains
-- ================================================================

theorem causal_composition (edges : EdgeSet n) {x m y : Fin n}
    (hxm : edges x m = true) (hmy : edges m y = true) :
    Reachable edges x y :=
  Reachable.step (Reachable.direct hxm) hmy

theorem causal_chain_ordered (edges : EdgeSet n) (htopo : isTopological edges)
    {x m y : Fin n} (hxm : edges x m = true) (hmy : edges m y = true) :
    x.val < m.val ∧ m.val < y.val :=
  And.intro (htopo x m hxm) (htopo m y hmy)

theorem chain_3_reachable (edges : EdgeSet n) {a b c d : Fin n}
    (hab : edges a b = true) (hbc : edges b c = true)
    (hcd : edges c d = true) : Reachable edges a d :=
  Reachable.step (Reachable.step (Reachable.direct hab) hbc) hcd

-- ================================================================
-- §18. Subgraph Properties
-- ================================================================

def isSubgraph (e1 e2 : EdgeSet n) : Prop :=
  ∀ i j, e1 i j = true → e2 i j = true

theorem isSubgraph_refl (edges : EdgeSet n) : isSubgraph edges edges :=
  fun _ _ h => h

theorem isSubgraph_trans {e1 e2 e3 : EdgeSet n}
    (h12 : isSubgraph e1 e2) (h23 : isSubgraph e2 e3) :
    isSubgraph e1 e3 :=
  fun i j h => h23 i j (h12 i j h)

theorem emptyGraph_subgraph (edges : EdgeSet n) :
    isSubgraph emptyGraph edges := by
  intro i j h; simp [emptyGraph] at h

theorem intervene_is_subgraph (edges : EdgeSet n) (t : Fin n) :
    isSubgraph (intervene edges t) edges :=
  fun i j h => intervene_sub edges t i j h

theorem subgraph_reachable {e1 e2 : EdgeSet n} (hsub : isSubgraph e1 e2)
    {i j : Fin n} (hr : Reachable e1 i j) : Reachable e2 i j :=
  reachable_monotone hsub hr

theorem subgraph_topo {e1 e2 : EdgeSet n} (hsub : isSubgraph e1 e2)
    (htopo : isTopological e2) : isTopological e1 :=
  fun i j h => htopo i j (hsub i j h)

theorem emptyGraph_no_reachable {i j : Fin n}
    (hr : Reachable (emptyGraph : EdgeSet n) i j) : False := by
  cases hr with
  | direct h => simp [emptyGraph] at h
  | step _ hkj => simp [emptyGraph] at hkj

end Sounio.Causality
