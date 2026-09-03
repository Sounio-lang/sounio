-- formal/lean4/SounioMercyfulScheduler.lean
/-!
# Mercyful Learning Scheduler — Lean 4 Correctness Formalization

Formalizes the three core correctness claims of the Mercyful Learning
scheduler (`stdlib/clinical/mercyful.sio`, mirrored by the pure-Python
reference `scripts/research/mercyful_runtime_contract.py`), on the
synthetic MIMIC-IV vancomycin TDM graph of
`scripts/research/mercyful_mimic_iv_vancomycin_contract.py` (V1..V7,
V_GREEN) and `docs/research/mimic_iv_mercyful_validation_2026-07-26.md`.

Spec: `docs/research/mercyful_scheduler_lean_spec_2026-07-26.md`.
CI gate: `scripts/ci/mercyful_lean_gate.sh` (also a `@[default_target]`
in `formal/lean4/lakefile.lean`, hence built by the CI `lean-proofs` job).

## The three claims

1. **T2 — Therapeutic-window selection (anti-Goodhart sufficiency).**
   The Mercyful scheduler is a constrained argmin over *feasible* paths
   (valid, start-anchored, target-reaching, within budget). Whenever a
   feasible path exists, the scheduler returns one
   (`mercyful_selects_therapeutic_window`); in particular its pick always
   reaches the target (`mercyful_reaches_target`). Under-dosing —
   selecting a course that never reaches the therapeutic target — is
   excluded by construction of the feasible set.

2. **T3 — Naive toxicity minimizer.** The naive minimizer is argmin of a
   toxicity-only metric over arms; it is target-blind *by construction*
   (its type has no target parameter) and optimal for its own metric
   (`naive_minimizer_optimal`). On the MIMIC-IV synthetic graph it
   selects `FIXED_LOW` — the sub-therapeutic arm with toxicity exactly 0
   that has no path to `TARGET` (`vanco_naive_underdoses`).

3. **T4 — Anti-Goodhart necessity and sufficiency.** With the target
   constraint, under-dosing is impossible (sufficiency, T2). Without it,
   if a zero-suffering trivial course exists (the `[start]` path) and
   every target-reaching course costs positive suffering, the
   unconstrained minimizer provably selects a non-target path
   (`goodhart_trap`). Packed as
   `anti_goodhart_necessary_and_sufficient`. The counterfactual
   `vanco_gate_is_causal` shows that with the verification gate opened
   (unverified fixed dosing admitted to `TARGET`), the same scheduler
   switches its optimum to the non-TDM arm — the gate is what makes the
   TDM route optimal (V4 of the runtime contract).

## Modeling conventions (matching the runtime contracts)

- Unit edge lengths (all lengths are 1.0 in the MIMIC-IV contract).
- Integral suffering charges the *source* state of each traversed edge;
  the peak additionally includes the final state — exactly
  `MercyGraph.path_cost` in `mercyful_runtime_contract.py`.
- Costs are `Rat` (exact). Suffering values are the contract's exact
  printed values as rationals: 0.6 = 3/5, 0.7 = 7/10,
  0.675679 = 675679/1000000, 0.059420 = 59420/1000000.
- The candidate-path set is explicit (`cands : List Path`), so the
  abstract theorems quantify over *any* enumeration. The concrete
  MIMIC-IV theorems use the bounded simple-path enumerator `pathsFrom`
  (fuel 8 > 6 states, so every simple path of the 6-state graph is
  enumerated; completeness of the enumerator itself is the only
  unformalized component — see the spec, "Scoped out").

Mathlib-free. No `sorry`. Abstract theorems (S- and T-sections) are pure
Lean proofs; concrete MIMIC-IV instance theorems (C-section) use
`native_decide` (axiom `Lean.ofReduceBool` for those theorems only).

Clinical scope: this is a synthetic graph scheduler for research
infrastructure. It is not medical guidance, not a treatment
recommendation, and not a clinical decision-support tool.

Relationship to the runtime: this file proves properties of the ABSTRACT
scheduler model (constrained argmin over an explicit candidate list) used
by the runtime contracts. It is NOT a refinement proof of the BFS queue
implementation in `stdlib/clinical/mercyful.sio` (spec §4, scoped out).
-/

namespace Sounio.Mercyful

/-- A path is a list of states (vertices). -/
abbrev Path := List Nat

/-- Finite directed graph with a suffering field over states. -/
structure MercyGraph where
  stateCount : Nat
  edges : List (Nat × Nat)
  suffering : Nat → Rat

/-- Scheduler parameters: endpoints, peak-aversion weight μ, and an edge
    budget L0 (number of traversed edges; unit lengths). -/
structure SchedParams where
  start : Nat
  target : Nat
  mu : Rat
  budget : Nat

-- ================================================================
-- S1. Cost model (mirrors MercyGraph.path_cost, unit lengths)
-- ================================================================

/-- Integral suffering along a path: source-state suffering per edge. -/
def pathIntegral (g : MercyGraph) : Path → Rat
  | [] => 0
  | [_] => 0
  | u :: v :: rest => g.suffering u + pathIntegral g (v :: rest)

/-- Rational max (defined locally to keep the file self-contained). -/
def rmax (a b : Rat) : Rat := if a ≤ b then b else a

/-- Peak suffering over the states of a path (includes the final state). -/
def pathPeak (g : MercyGraph) : Path → Rat
  | [] => 0
  | u :: rest => rmax (g.suffering u) (pathPeak g rest)

/-- Total path cost: integral + μ · peak. -/
def pathCost (g : MercyGraph) (mu : Rat) (p : Path) : Rat :=
  pathIntegral g p + mu * pathPeak g p

theorem le_rmax_left (a b : Rat) : a ≤ rmax a b := by
  unfold rmax
  split
  · next h => exact h
  · next _ => exact Rat.le_refl

theorem le_rmax_right (a b : Rat) : b ≤ rmax a b := by
  unfold rmax
  split
  · next _ => exact Rat.le_refl
  · next h =>
    exact (Rat.le_total (a := a) (b := b)).elim (fun hab => absurd hab h) (fun hba => hba)

/-- Nonnegative suffering fields give nonnegative integrals. -/
theorem pathIntegral_nonneg (g : MercyGraph) (h : ∀ v, 0 ≤ g.suffering v) :
    ∀ p, 0 ≤ pathIntegral g p
  | [] => Rat.le_refl
  | [_] => Rat.le_refl
  | _ :: v :: rest => Rat.add_nonneg (h _) (pathIntegral_nonneg g h (v :: rest))

/-- Nonnegative suffering fields give nonnegative peaks. -/
theorem pathPeak_nonneg (g : MercyGraph) (h : ∀ v, 0 ≤ g.suffering v) :
    ∀ p, 0 ≤ pathPeak g p
  | [] => Rat.le_refl
  | u :: rest =>
    Rat.le_trans (h u) (le_rmax_left (g.suffering u) (pathPeak g rest))

/-- Nonnegative suffering field and μ ≥ 0 give nonnegative path costs. -/
theorem pathCost_nonneg (g : MercyGraph) (h_s : ∀ v, 0 ≤ g.suffering v)
    (h_mu : 0 ≤ mu) (p : Path) : 0 ≤ pathCost g mu p :=
  Rat.add_nonneg (pathIntegral_nonneg g h_s p)
    (Rat.mul_nonneg h_mu (pathPeak_nonneg g h_s p))

theorem rmax_zero_zero : rmax 0 0 = (0 : Rat) := by
  unfold rmax
  rw [if_pos Rat.le_refl]

/-- The trivial path `[u]` on a zero-suffering state costs exactly 0 —
    the Goodhart trap: not treating is the global unconstrained optimum. -/
theorem pathCost_singleton_zero (g : MercyGraph) (mu : Rat) (u : Nat)
    (h : g.suffering u = 0) : pathCost g mu [u] = 0 := by
  show (0 : Rat) + mu * rmax (g.suffering u) 0 = 0
  rw [h, rmax_zero_zero, Rat.mul_zero, Rat.add_zero]

-- ================================================================
-- S2. Argmin over a finite candidate list
-- ================================================================

/-- Strict-improvement argmin accumulator: keeps the FIRST element
    attaining the minimal cost (matches the runtime's `total < best_cost`). -/
def argminAux (cost : α → Rat) (best : α) : List α → α
  | [] => best
  | x :: xs => if cost x < cost best then argminAux cost x xs else argminAux cost best xs

/-- Argmin over a list; `none` iff the list is empty. -/
def argmin (cost : α → Rat) : List α → Option α
  | [] => none
  | x :: xs => some (argminAux cost x xs)

theorem argminAux_mem (cost : α → Rat) (b : α) (xs : List α) :
    argminAux cost b xs = b ∨ argminAux cost b xs ∈ xs := by
  induction xs generalizing b with
  | nil => exact Or.inl rfl
  | cons x xs ih =>
    by_cases h : cost x < cost b
    · simp only [argminAux, if_pos h]
      cases ih x with
      | inl heq =>
        apply Or.inr
        rw [heq]
        exact List.mem_cons_self
      | inr hmem => exact Or.inr (List.mem_cons_of_mem x hmem)
    · simp only [argminAux, if_neg h]
      cases ih b with
      | inl heq => exact Or.inl heq
      | inr hmem => exact Or.inr (List.mem_cons_of_mem x hmem)

theorem argminAux_le (cost : α → Rat) (b : α) (xs : List α) :
    cost (argminAux cost b xs) ≤ cost b ∧
    ∀ x ∈ xs, cost (argminAux cost b xs) ≤ cost x := by
  induction xs generalizing b with
  | nil => exact ⟨Rat.le_refl, fun x hx => nomatch hx⟩
  | cons x xs ih =>
    by_cases h : cost x < cost b
    · simp only [argminAux, if_pos h]
      have ⟨h1, h2⟩ := ih x
      refine ⟨Rat.le_trans h1 (Rat.le_of_lt h), fun y hy => ?_⟩
      cases hy with
      | head => exact h1
      | tail _ hymem => exact h2 y hymem
    · simp only [argminAux, if_neg h]
      have ⟨h1, h2⟩ := ih b
      refine ⟨h1, fun y hy => ?_⟩
      cases hy with
      | head => exact Rat.le_trans h1 (Rat.not_lt.mp h)
      | tail _ hymem => exact h2 y hymem

/-- The argmin of a list is a member of the list. -/
theorem argmin_some_mem {cost : α → Rat} {xs : List α} {m : α}
    (h : argmin cost xs = some m) : m ∈ xs := by
  cases xs with
  | nil => simp [argmin] at h
  | cons x xs' =>
    simp only [argmin] at h
    have hm : argminAux cost x xs' = m := Option.some.inj h
    rw [← hm]
    cases argminAux_mem cost x xs' with
    | inl heq =>
      rw [heq]
      exact List.mem_cons_self
    | inr hmem => exact List.mem_cons_of_mem x hmem

/-- The argmin of a list attains the minimum cost over the list. -/
theorem argmin_some_min {cost : α → Rat} {xs : List α} {m : α}
    (h : argmin cost xs = some m) : ∀ y ∈ xs, cost m ≤ cost y := by
  cases xs with
  | nil => simp [argmin] at h
  | cons x xs' =>
    simp only [argmin] at h
    have hm : argminAux cost x xs' = m := Option.some.inj h
    rw [← hm]
    have ⟨h1, h2⟩ := argminAux_le cost x xs'
    intro y hy
    cases hy with
    | head => exact h1
    | tail _ hymem => exact h2 y hymem

/-- Tie-breaking policy, recorded explicitly: on a cost TIE the
    accumulator is kept (strict-`<` update), so the FIRST list element
    attaining the minimum is selected — matching the runtime's
    `total < best_cost` loop. -/
theorem argminAux_tie_keeps (cost : α → Rat) (b x : α) (xs : List α)
    (h : cost x = cost b) :
    argminAux cost b (x :: xs) = argminAux cost b xs := by
  have hnl : ¬ cost x < cost b := by
    intro hlt
    rw [h] at hlt
    exact Rat.lt_irrefl hlt
  simp only [argminAux, if_neg hnl]

-- ================================================================
-- S3. Feasibility (the anti-Goodhart constraint)
-- ================================================================

/-- Path starts at the required start state. -/
def startsAt (sp : SchedParams) (p : Path) : Bool := p.head? == some sp.start

/-- Path reaches the therapeutic target — the anti-Goodhart clause. -/
def reachesTarget (sp : SchedParams) (p : Path) : Bool := p.getLast? == some sp.target

/-- Path stays within the edge budget (k states ⇒ k−1 edges ≤ L0). -/
def withinBudget (sp : SchedParams) (p : Path) : Bool := decide (p.length ≤ sp.budget + 1)

/-- Consecutive states are edges of the graph. -/
def isPathB (g : MercyGraph) : Path → Bool
  | [] => false
  | [_] => true
  | u :: v :: rest => g.edges.elem (u, v) && isPathB g (v :: rest)

/-- A feasible path: starts at `start`, reaches `target`, within budget,
    and valid in the graph. Target-reaching is what excludes
    under-dosing; dropping it recovers the naive/unconstrained scheduler. -/
def FeasibleB (g : MercyGraph) (sp : SchedParams) (p : Path) : Bool :=
  startsAt sp p && reachesTarget sp p && withinBudget sp p && isPathB g p

/-- **Mercyful scheduler**: argmin over feasible paths only. -/
def mercyfulSchedule (g : MercyGraph) (sp : SchedParams) (cands : List Path) :
    Option Path :=
  argmin (pathCost g sp.mu) (cands.filter (FeasibleB g sp))

/-- **Unconstrained (raw) scheduler**: argmin over ALL candidate paths,
    no target constraint — the Goodhart-vulnerable baseline (V2). -/
def unconstrainedSchedule (g : MercyGraph) (sp : SchedParams) (cands : List Path) :
    Option Path :=
  argmin (pathCost g sp.mu) cands

/-- **Naive toxicity minimizer**: argmin of a toxicity-only metric over
    arms. Target-blind BY CONSTRUCTION — the objective has no target
    parameter, so the pick cannot depend on the therapeutic target (V1). -/
def naiveToxPick (tox : α → Rat) (arms : List α) : Option α := argmin tox arms

-- ================================================================
-- T1. Feasible selection
-- ================================================================

/-- The Mercyful scheduler returns a feasible path whose cost is minimal
    among all feasible candidates. -/
theorem mercyful_feasible_selection (g : MercyGraph) (sp : SchedParams)
    (cands : List Path) (p : Path)
    (h : mercyfulSchedule g sp cands = some p) :
    FeasibleB g sp p = true ∧
    (∀ q ∈ cands, FeasibleB g sp q = true →
      pathCost g sp.mu p ≤ pathCost g sp.mu q) := by
  have hmem : p ∈ cands.filter (FeasibleB g sp) := argmin_some_mem h
  have hmin := argmin_some_min h
  rw [List.mem_filter] at hmem
  exact ⟨hmem.2, fun q hq hfq => hmin q (List.mem_filter.mpr ⟨hq, hfq⟩)⟩

-- ================================================================
-- T2. Therapeutic-window selection (anti-Goodhart sufficiency)
-- ================================================================

/-- **Sufficiency.** The Mercyful scheduler's pick always reaches the
    therapeutic target: under-dosing is excluded by the constraint. -/
theorem mercyful_reaches_target (g : MercyGraph) (sp : SchedParams)
    (cands : List Path) (p : Path)
    (h : mercyfulSchedule g sp cands = some p) :
    reachesTarget sp p = true := by
  have hf := (mercyful_feasible_selection g sp cands p h).1
  unfold FeasibleB at hf
  have ⟨h1, _⟩ := Bool.and_eq_true_iff.mp hf
  have ⟨h2, _⟩ := Bool.and_eq_true_iff.mp h1
  have ⟨_, hr⟩ := Bool.and_eq_true_iff.mp h2
  exact hr

/-- Whenever a feasible path exists among the candidates, the Mercyful
    scheduler returns one. -/
theorem mercyful_exists_of_feasible (g : MercyGraph) (sp : SchedParams)
    (cands : List Path)
    (h : ∃ p ∈ cands, FeasibleB g sp p = true) :
    ∃ p, mercyfulSchedule g sp cands = some p := by
  obtain ⟨p, hp, hfp⟩ := h
  have hne : cands.filter (FeasibleB g sp) ≠ [] := by
    intro hnil
    have hm : p ∈ cands.filter (FeasibleB g sp) := List.mem_filter.mpr ⟨hp, hfp⟩
    rw [hnil] at hm
    nomatch hm
  unfold mercyfulSchedule
  cases hf : cands.filter (FeasibleB g sp) with
  | nil => exact absurd hf hne
  | cons x xs => exact ⟨argminAux (pathCost g sp.mu) x xs, rfl⟩

/-- **Therapeutic-window selection.** If the therapeutic window exists and
    is feasible (some feasible candidate), the Mercyful scheduler selects
    a course that reaches the target, at minimal cost among feasible
    courses. -/
theorem mercyful_selects_therapeutic_window (g : MercyGraph) (sp : SchedParams)
    (cands : List Path)
    (h : ∃ p ∈ cands, FeasibleB g sp p = true) :
    ∃ p, mercyfulSchedule g sp cands = some p ∧
      reachesTarget sp p = true ∧
      (∀ q ∈ cands, FeasibleB g sp q = true →
        pathCost g sp.mu p ≤ pathCost g sp.mu q) := by
  obtain ⟨p, hp⟩ := mercyful_exists_of_feasible g sp cands h
  exact ⟨p, hp, mercyful_reaches_target g sp cands p hp,
    (mercyful_feasible_selection g sp cands p hp).2⟩

-- ================================================================
-- T3. Naive toxicity minimizer
-- ================================================================

/-- The naive toxicity minimizer is optimal for its own (target-blind)
    metric: no arm has strictly lower toxicity than the pick. -/
theorem naive_minimizer_optimal (tox : α → Rat) (arms : List α) (a : α)
    (h : naiveToxPick tox arms = some a) : ∀ b ∈ arms, tox a ≤ tox b :=
  argmin_some_min h

-- ================================================================
-- T4. Anti-Goodhart: necessity and sufficiency
-- ================================================================

/-- **Necessity (the Goodhart trap).** Without the target constraint, if
    the trivial untreated course `[start]` is a candidate and costs 0
    (zero suffering at `start`, nonnegative field, μ ≥ 0), while every
    target-reaching candidate costs strictly positive suffering, then the
    unconstrained scheduler's pick NEVER reaches the target: dropping the
    anti-Goodhart constraint admits under-dosing optima. -/
theorem goodhart_trap (g : MercyGraph) (sp : SchedParams) (cands : List Path)
    (p : Path)
    (h_trivial : [sp.start] ∈ cands)
    (h_s0 : g.suffering sp.start = 0)
    (h_nonneg : ∀ v, 0 ≤ g.suffering v)
    (h_mu : 0 ≤ sp.mu)
    (h_expensive : ∀ q ∈ cands, reachesTarget sp q = true →
      0 < pathCost g sp.mu q)
    (h_pick : unconstrainedSchedule g sp cands = some p) :
    reachesTarget sp p = false := by
  have hcost0 : pathCost g sp.mu [sp.start] = 0 :=
    pathCost_singleton_zero g sp.mu sp.start h_s0
  have hle : pathCost g sp.mu p ≤ 0 := by
    rw [← hcost0]
    exact argmin_some_min h_pick [sp.start] h_trivial
  have hge : 0 ≤ pathCost g sp.mu p := pathCost_nonneg g h_nonneg h_mu p
  have h0 : pathCost g sp.mu p = 0 := Rat.le_antisymm hle hge
  cases hr : reachesTarget sp p with
  | false => rfl
  | true =>
    have hmem : p ∈ cands := argmin_some_mem h_pick
    have hpos := h_expensive p hmem hr
    rw [h0] at hpos
    exact absurd hpos Rat.lt_irrefl

/-- **Necessary and sufficient.** Packing T2 and T4: the anti-Goodhart
    (target-reaching) constraint is sufficient to exclude under-dosing
    picks, and — under the trap conditions — necessary, because the
    unconstrained scheduler provably under-doses. -/
theorem anti_goodhart_necessary_and_sufficient (g : MercyGraph)
    (sp : SchedParams) (cands : List Path) :
    (∀ p, mercyfulSchedule g sp cands = some p → reachesTarget sp p = true) ∧
    (([sp.start] ∈ cands) → (g.suffering sp.start = 0) →
     (∀ v, 0 ≤ g.suffering v) → 0 ≤ sp.mu →
     (∀ q ∈ cands, reachesTarget sp q = true → 0 < pathCost g sp.mu q) →
     ∀ p, unconstrainedSchedule g sp cands = some p →
       reachesTarget sp p = false) :=
  ⟨fun p hp => mercyful_reaches_target g sp cands p hp,
   fun htriv hs0 hnn hmu hexp p hp =>
     goodhart_trap g sp cands p htriv hs0 hnn hmu hexp hp⟩

-- ================================================================
-- S4. Bounded simple-path enumerator (for the concrete instance)
-- ================================================================

/-- Out-neighbors of a state. -/
def edgesFrom (g : MercyGraph) (u : Nat) : List Nat :=
  (g.edges.filter (fun e => e.1 == u)).map Prod.snd

/-- All simple paths from `u` avoiding `visited`, bounded by `fuel`. -/
def pathsFromAux (g : MercyGraph) (visited : List Nat) (fuel : Nat) (u : Nat) :
    List Path :=
  match fuel with
  | 0 => []
  | fuel + 1 =>
    [u] :: ((edgesFrom g u).flatMap fun v =>
      if visited.elem v then [] else (pathsFromAux g (u :: visited) fuel v).map (u :: ·))

/-- All simple paths starting at `start`, including the trivial path
    `[start]` (the untreated course). Fuel ≥ state count enumerates every
    simple path; the MIMIC-IV graph has 6 states and we use fuel 8. -/
def pathsFrom (g : MercyGraph) (fuel : Nat) (start : Nat) : List Path :=
  pathsFromAux g [start] fuel start

-- ================================================================
-- C. Concrete instance: MIMIC-IV vancomycin TDM synthetic graph
-- ================================================================

namespace VancoMimicIV

/-- State ids (mirroring mercyful_mimic_iv_vancomycin_contract.py). -/
def START := 0
def FIXED_LOW := 1
def FIXED_STD := 2
def VANCO_PRE := 3
def TDM_GUIDED := 4
def TARGET := 5

/-- Suffering field (exact rationals): 0.6 = s_win([4,9]) pure efficacy
    shortfall, 0.7 = s_win([6,26]) straddling, 0.675679 pre-TDM and
    0.059420 post-TDM measured on the repo's clinical twin (clause C3). -/
def suffering : Nat → Rat
  | 0 => 0
  | 1 => 3 / 5
  | 2 => 7 / 10
  | 3 => 675679 / 1000000
  | 4 => 59420 / 1000000
  | _ => 0

/-- The gated graph: only the window-verified TDM course is admitted to
    TARGET (G_VERIFY). FIXED_LOW and FIXED_STD have no edge to TARGET. -/
def graph : MercyGraph := {
  stateCount := 6
  edges := [(0, 0), (0, 1), (0, 2), (0, 3), (3, 4), (4, 5)]
  suffering := suffering
}

/-- Counterfactual graph with the verification gate opened: unverified
    fixed standard dosing admitted to TARGET (V4 counterfactual). -/
def graphOpen : MercyGraph := {
  stateCount := 6
  edges := [(0, 0), (0, 1), (0, 2), (0, 3), (3, 4), (2, 5), (4, 5)]
  suffering := suffering
}

/-- Contract parameters: START → TARGET, μ = 1, L0 = 10 (unit lengths). -/
def params : SchedParams := { start := 0, target := 5, mu := 1, budget := 10 }

/-- Toxicity-only metric (supra-therapeutic component of s_win):
    FIXED_LOW has toxicity exactly 0; FIXED_STD has 0.3 = max(0, 26−20)/20. -/
def tox : Nat → Rat
  | 1 => 0
  | 2 => 3 / 10
  | _ => 0

/-- Instance check for the abstract theorems' non-negativity hypothesis:
    the concrete suffering table is everywhere nonnegative, so
    `pathCost_nonneg` and `goodhart_trap` apply to this graph. -/
theorem vanco_suffering_nonneg (v : Nat) : 0 ≤ suffering v := by
  unfold suffering
  split <;> native_decide

/-- C1: the TDM-guided route is the UNIQUE feasible course among the
    enumerated candidates — the same candidate set the runtime scheduler
    optimizes over (fuel 8 > 6 states, so every simple path is
    enumerated; the elementary completeness argument is in spec §4,
    its mechanization scoped out there). -/
theorem vanco_feasible_unique :
    (pathsFrom graph 8 START).filter (FeasibleB graph params)
      = [[START, VANCO_PRE, TDM_GUIDED, TARGET]] := by
  native_decide

/-- C2 (V5): the Mercyful scheduler selects the TDM-guided course. -/
theorem vanco_mercyful_selects_tdm :
    mercyfulSchedule graph params (pathsFrom graph 8 START)
      = some [START, VANCO_PRE, TDM_GUIDED, TARGET] := by
  native_decide

/-- C3 (V5 canonical numbers): the selected course costs exactly
    ∫s = 0.735099, peak = 0.675679, total = 1.410778 at μ = 1 — matching
    the clinical twin's healthy scenario (clause C1) and the runtime
    contract's printed values. -/
theorem vanco_tdm_route_cost :
    pathIntegral graph [START, VANCO_PRE, TDM_GUIDED, TARGET] = 735099 / 1000000 ∧
    pathPeak graph [START, VANCO_PRE, TDM_GUIDED, TARGET] = 675679 / 1000000 ∧
    pathCost graph 1 [START, VANCO_PRE, TDM_GUIDED, TARGET] = 1410778 / 1000000 := by
  native_decide

/-- C4 (V1): the naive toxicity minimizer selects FIXED_LOW — toxicity
    exactly 0 — and FIXED_LOW cannot reach TARGET at all (no enumerated
    path from it reaches the target): under-dosing. -/
theorem vanco_naive_underdoses :
    naiveToxPick tox [FIXED_LOW, FIXED_STD] = some FIXED_LOW ∧
    (pathsFrom graph 8 FIXED_LOW).filter (reachesTarget params) = [] := by
  native_decide

/-- C5 (V2): the unconstrained raw minimizer selects the trivial
    untreated course `[START]` (cost 0) — never treats. -/
theorem vanco_unconstrained_traps :
    unconstrainedSchedule graph params (pathsFrom graph 8 START)
      = some [START] := by
  native_decide

/-- C6 (V4): with the verification gate opened, the SAME scheduler
    switches its optimum to the non-TDM arm FIXED_STD (total 1.4 <
    1.410778) — the anti-Goodhart gate is what makes the TDM route the
    unique feasible optimum. -/
theorem vanco_gate_is_causal :
    mercyfulSchedule graphOpen params (pathsFrom graphOpen 8 START)
      = some [START, FIXED_STD, TARGET] := by
  native_decide

end VancoMimicIV

end Sounio.Mercyful
