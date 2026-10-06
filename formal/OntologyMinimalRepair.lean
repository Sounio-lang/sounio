import OntologyEvolutionRepair

/-!
# Sounio.OntologyMinimalRepair — minimal repair against MULTIPLE partners

Formal companion to
`artifacts/ontology-frontiers/consistent-ontology-evolution/minimal_repair_demo.sio`
(frontier: `consistent-ontology-evolution`; see the `FRONTIER.md` there).

## Setting

`OntologyEvolutionRepair.repair_retry` closes the single-partner case: when
the guard rejects `add a` with conflict witness `k` and `k` is the ONLY
conflicting partner of `a` in the version, removing `k` unblocks the edit.
This file closes the remaining cross-frontier gap: the candidate `a`
conflicts with **several** partners of the current version `v`
(the `partners` set), and every partner *independently* blocks `a`.

Because each partner blocks `a` on its own, admitting `a` forces the removal
of ALL partners — the removal set is uniquely determined. The only real
choice left is binary, and it is made on **epistemic mass** (confidences, in
Nat per-mille as in the other frontier files):

- **ADMIT(a)** — remove all partners, then add `a`. The contested mass that
  survives the conflict is `conf a` alone: the partners are gone.
- **REJECT(a)** — keep `v` unchanged. The contested mass that survives is
  the SUM of the partner confidences (mass convention: we take the sum, not
  the max — all partners stay active, so the version retains all of their
  epistemic weight).

`decide` admits iff `conf a > sum of partner confidences`; on a tie it
**rejects** (documented convention: an incoming candidate must strictly
outweigh the established mass it would displace).

## What is proved here

1. `admit_succeeds` — (a) after removing ALL partners, `a` conflicts with
   nothing left, and the admitted version `a :: (v minus partners)` is
   consistent (via the general sublist lemma `consistent_sublist`).
2. `reject_consistent` — (b) rejecting keeps `v`, hence is consistent.
3. `decide_optimal` — (c) `decide` yields retained mass ≥ the other
   option's (case analysis on the strict comparison).
4. `partner_not_mem_of_admissible` / `admissible_sublist_partnerfree` —
   (d) **necessity / unique minimality**: no partner of `a` may survive in
   ANY kept sublist that admits `a`; hence removing all partners is the
   unique minimal removal set, and the partner-free remainder is the unique
   maximal admissible kept set.
5. Concrete `Fin 7` instance (§5) in the style of the `OntologyEvolution`
   example: candidate 4 conflicts with TWO partners (1 and 3) of {3,2,1};
   an admit-winning confidence profile (`conf 4 = 900 > 400 + 400`) and a
   reject-winning one (`conf 4 = 300 < 400 + 400`), both computed by
   `native_decide`, plus the necessity direction (removing only ONE partner
   still blocks the candidate).

Depends on `OntologyEvolutionRepair.lean` (and transitively on
`OntologyEvolution.lean` and `OntologyAlignmentRepair.lean`). No Mathlib.
Zero sorry. No new axioms.
-/

namespace Sounio.OntologyMinimalRepair

open Sounio.OntologyEvolution (Consistent)
open Sounio.OntologyAlignmentRepair (conflictsAny conflictsAny_false)
open Sounio.OntologyEvolutionRepair (consistent_sublist)

variable {α : Type}

-- ---------------------------------------------------------------------------
-- §1. Partners, mass, and the admit/reject decision
-- ---------------------------------------------------------------------------

/-- The partners of candidate `a` in version `v`: the elements of `v` that
    conflict with `a`. Each partner independently blocks `a`. -/
def partners (C : α → α → Bool) (a : α) (v : List α) : List α :=
  v.filter (fun x => C a x)

/-- Sum of a list of natural-number masses (per-mille confidences). -/
def sum : List Nat → Nat
  | [] => 0
  | x :: xs => x + sum xs

/-- The contested mass retained by REJECT(a): the sum of the partner
    confidences (all partners stay active). Mass convention: SUM, not max —
    the version keeps every partner, hence all of their epistemic weight. -/
def partnerMass (C : α → α → Bool) (conf : α → Nat) (a : α) (v : List α) : Nat :=
  sum ((partners C a v).map conf)

/-- The binary repair decision for a candidate with multiple partners. -/
inductive Decision where
  | admit
  | reject
deriving DecidableEq

/-- Mass-optimal decision: ADMIT iff `conf a` STRICTLY outweighs the summed
    partner mass; ties reject (an incoming candidate must strictly outweigh
    the established mass it would displace). -/
def decide (C : α → α → Bool) (conf : α → Nat) (a : α) (v : List α) : Decision :=
  if partnerMass C conf a v < conf a then .admit else .reject

/-- Applying a decision: ADMIT removes ALL partners and adds `a`; REJECT
    keeps the version unchanged. -/
def applyDecision (C : α → α → Bool) (a : α) (v : List α) : Decision → List α
  | .admit => a :: v.filter (fun x => !C a x)
  | .reject => v

/-- The contested mass retained by each option: `conf a` under ADMIT (the
    partners are gone), the summed partner mass under REJECT. -/
def retainedMass (C : α → α → Bool) (conf : α → Nat) (a : α) (v : List α) :
    Decision → Nat
  | .admit => conf a
  | .reject => partnerMass C conf a v

-- ---------------------------------------------------------------------------
-- §2. (a) ADMIT succeeds; (b) REJECT is consistent
-- ---------------------------------------------------------------------------

/-- After removing ALL partners, nothing left conflicts with `a`. -/
theorem admit_conflictsAny_false {C : α → α → Bool} {a : α} {v : List α} :
    conflictsAny C a (v.filter (fun x => !C a x)) = false := by
  rw [conflictsAny_false]
  intro x hx
  obtain ⟨_, hxnot⟩ := List.mem_filter.mp hx
  cases hcx : C a x with
  | false => rfl
  | true => simp [hcx] at hxnot

/-- (a) **ADMIT succeeds**: removing all partners unblocks `a`, and the
    admitted version `a :: (v minus partners)` is consistent (the remainder
    is a sublist of `v`, so the general sublist lemma applies). Note the
    contrast with `repair_retry`: NO uniqueness hypothesis on partners is
    needed — removing the whole partner set handles any multiplicity. -/
theorem admit_succeeds {C : α → α → Bool} {a : α} {v : List α}
    (hv : Consistent C v) :
    conflictsAny C a (v.filter (fun x => !C a x)) = false ∧
      Consistent C (applyDecision C a v .admit) := by
  refine ⟨admit_conflictsAny_false, ?_⟩
  show Consistent C (a :: v.filter (fun x => !C a x))
  exact ⟨conflictsAny_false.mp admit_conflictsAny_false,
    consistent_sublist List.filter_sublist hv⟩

/-- (b) **REJECT is consistent**: rejecting keeps `v`, which was assumed
    consistent. -/
theorem reject_consistent {C : α → α → Bool} {a : α} {v : List α}
    (hv : Consistent C v) : Consistent C (applyDecision C a v .reject) := hv

-- ---------------------------------------------------------------------------
-- §3. (c) Optimality of the mass-based decision
-- ---------------------------------------------------------------------------

/-- (c) **Optimality**: `decide` retains at least as much contested mass as
    either option — by case analysis on the strict comparison. -/
theorem decide_optimal {C : α → α → Bool} {conf : α → Nat} {a : α} {v : List α}
    (d : Decision) :
    retainedMass C conf a v d ≤ retainedMass C conf a v (decide C conf a v) := by
  by_cases h : partnerMass C conf a v < conf a
  · have hd : decide C conf a v = .admit := by simp [decide, h]
    rw [hd]
    cases d with
    | admit => exact Nat.le_refl _
    | reject =>
        show partnerMass C conf a v ≤ conf a
        exact Nat.le_of_lt h
  · have hd : decide C conf a v = .reject := by simp [decide, h]
    rw [hd]
    cases d with
    | admit =>
        show conf a ≤ partnerMass C conf a v
        exact Nat.le_of_not_lt h
    | reject => exact Nat.le_refl _

-- ---------------------------------------------------------------------------
-- §4. (d) Necessity: removing ALL partners is the UNIQUE minimal repair
-- ---------------------------------------------------------------------------

/-- **Necessity, pointwise**: a conflicting partner of `a` cannot survive in
    ANY kept set that admits `a`. (If one did, `conflictsAny` would still be
    `true` and the guarded edit would reject `a` again.) -/
theorem partner_not_mem_of_admissible {C : α → α → Bool} {a x : α} {w : List α}
    (hcf : conflictsAny C a w = false) (hcx : C a x = true) : x ∉ w := by
  intro hxw
  have h := conflictsAny_false.mp hcf x hxw
  rw [h] at hcx
  exact Bool.noConfusion hcx

/-- Auxiliary: a sublist whose elements all satisfy `p` is also a sublist of
    the `p`-filtered list. -/
theorem sublist_filter_of_forall {p : α → Bool} {w v : List α}
    (hsub : w.Sublist v) (hw : ∀ x ∈ w, p x = true) :
    w.Sublist (v.filter p) := by
  induction hsub with
  | slnil => exact List.Sublist.slnil
  | cons b _ ih =>
      have ih' := ih hw
      rw [List.filter_cons]
      cases hb : p b with
      | true =>
          simp
          exact List.Sublist.cons _ ih'
      | false =>
          simp
          exact ih'
  | cons_cons a _ ih =>
      have hpa : p a = true := hw a List.mem_cons_self
      have htail : ∀ x ∈ _, p x = true :=
        fun x hx => hw x (List.mem_cons_of_mem _ hx)
      have ih' := ih htail
      rw [List.filter_cons]
      simp [hpa]
      exact ih'

/-- (d) **Unique minimality, positive form**: every kept sublist of `v` that
    admits `a` is a sublist of the partner-free remainder. Combined with
    `admit_succeeds` (the remainder itself is admissible), this says the
    partner-free remainder is the UNIQUE MAXIMAL admissible kept set — and
    dually, the partner set is the UNIQUE MINIMAL removal set: nothing less
    suffices (by `partner_not_mem_of_admissible`) and nothing more is needed
    (by `admit_succeeds`). -/
theorem admissible_sublist_partnerfree {C : α → α → Bool} {a : α} {v w : List α}
    (hsub : w.Sublist v) (hcf : conflictsAny C a w = false) :
    w.Sublist (v.filter (fun x => !C a x)) := by
  apply sublist_filter_of_forall hsub
  intro x hx
  have h := conflictsAny_false.mp hcf x hx
  simp [h]

-- ---------------------------------------------------------------------------
-- §5. Concrete instance: candidate 4 vs TWO partners (1 and 3) of {3,2,1}
-- ---------------------------------------------------------------------------

/-- Axioms over `Fin 7`; candidate axiom 4 contradicts the established
    axioms 1 AND 3 (symmetric conflict oracle, in the style of
    `OntologyEvolution.exC`). -/
def exC7 : Fin 7 → Fin 7 → Bool := fun a b =>
  (a.val == 4 && b.val == 1) || (a.val == 1 && b.val == 4) ||
  (a.val == 4 && b.val == 3) || (a.val == 3 && b.val == 4)

/-- Confidence profile where ADMIT wins: `conf 4 = 900 > 400 + 400`. -/
def confA : Fin 7 → Nat := fun x =>
  match x.val with
  | 1 => 400
  | 3 => 400
  | 4 => 900
  | _ => 500

/-- Confidence profile where REJECT wins: `conf 4 = 300 < 400 + 400`. -/
def confR : Fin 7 → Nat := fun x =>
  match x.val with
  | 1 => 400
  | 3 => 400
  | 4 => 300
  | _ => 500

/-- Candidate 4 has exactly TWO partners in {3,2,1}: axioms 3 and 1. -/
theorem ex7_partners : partners exC7 4 [3, 2, 1] = [3, 1] := by
  native_decide

/-- The version {3,2,1} is consistent (via the addition-only invariant of
    `OntologyEvolution`, not by enumeration). -/
theorem ex7_v321_consistent : Consistent exC7 [3, 2, 1] :=
  Sounio.OntologyEvolution.mem_versions_consistent (v := []) (edits := [1, 2, 3])
    (by trivial) (by native_decide)

/-- ADMIT branch: `conf 4 = 900` strictly outweighs the summed partner mass
    `400 + 400 = 800`. -/
theorem ex7_decide_admit : decide exC7 confA 4 [3, 2, 1] = .admit := by
  native_decide

/-- The admitted version removes BOTH partners and adds 4: exactly {4,2}. -/
theorem ex7_admit_version : applyDecision exC7 4 [3, 2, 1] .admit = [4, 2] := by
  native_decide

/-- The admitted version is consistent — via the general theorem, not by
    enumeration. -/
theorem ex7_admit_consistent : Consistent exC7 (applyDecision exC7 4 [3, 2, 1] .admit) :=
  (admit_succeeds ex7_v321_consistent).2

/-- REJECT branch: `conf 4 = 300` does not outweigh `800`. -/
theorem ex7_decide_reject : decide exC7 confR 4 [3, 2, 1] = .reject := by
  native_decide

/-- The rejected version is exactly the original {3,2,1}. -/
theorem ex7_reject_version : applyDecision exC7 4 [3, 2, 1] .reject = [3, 2, 1] := by
  native_decide

/-- Optimality on the instance, ADMIT branch: the decision retains 900,
    the other option would retain 800. -/
theorem ex7_optimal_admit (d : Decision) :
    retainedMass exC7 confA 4 [3, 2, 1] d ≤
      retainedMass exC7 confA 4 [3, 2, 1] (decide exC7 confA 4 [3, 2, 1]) :=
  decide_optimal d

/-- Optimality on the instance, REJECT branch: the decision retains 800,
    the other option would retain 300. -/
theorem ex7_optimal_reject (d : Decision) :
    retainedMass exC7 confR 4 [3, 2, 1] d ≤
      retainedMass exC7 confR 4 [3, 2, 1] (decide exC7 confR 4 [3, 2, 1]) :=
  decide_optimal d

/-- Necessity on the instance, negative form: removing only ONE partner
    (axiom 3) still leaves partner 1, and the candidate remains blocked. -/
theorem ex7_remove_one_still_blocked :
    conflictsAny exC7 4 ([3, 2, 1].filter (fun x => x != 3)) = true := by
  native_decide

/-- Necessity on the instance, via the general theorem: in ANY kept set
    that admits candidate 4, neither partner 1 nor partner 3 may survive. -/
theorem ex7_necessity_pointwise {w : List (Fin 7)}
    (hcf : conflictsAny exC7 4 w = false) : (1 : Fin 7) ∉ w ∧ (3 : Fin 7) ∉ w :=
  ⟨partner_not_mem_of_admissible hcf (by native_decide),
   partner_not_mem_of_admissible hcf (by native_decide)⟩

/-- Maximality on the instance: the partner-free remainder is exactly {2},
    and every admissible kept sublist (e.g. {2} itself) sits inside it. -/
theorem ex7_maximal_kept :
    ([2] : List (Fin 7)).Sublist ([3, 2, 1].filter (fun x => !exC7 4 x)) :=
  admissible_sublist_partnerfree
    (List.Sublist.cons _ (List.Sublist.cons_cons _ (List.Sublist.cons _ List.Sublist.slnil)))
    (by native_decide)

end Sounio.OntologyMinimalRepair
