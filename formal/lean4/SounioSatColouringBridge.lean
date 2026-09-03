/-
# Sounio — soundness of the graph-colouring SAT encoding (the SAT-leg bridge)

This file closes the gap between a machine-checked CNF refutation
(`<cnf>.Unsat`, produced by souc_sat + Lean core's verified LRAT checker, see
`SounioSatK76.lean`) and the *graph-theoretic* statement used by the de Grey
χ(ℝ²) ≥ 5 reduction (`SounioDeGreyChi5.lean`):

    (colourCNF n k edges).Unsat  →  ¬ Colourable (Fin n) k edges.

The encoding `colourCNF` is the standard one emitted by `souc_sat`:
  * variable `v*k + c`  ≙  "vertex `v` receives colour `c`";
  * one *at-least-one* clause per vertex;
  * for every edge and every colour, a binary clause forbidding both endpoints
    from taking that colour.

The proof is pure logic + Nat arithmetic — **no Mathlib, no `native_decide`, no
`sorry`**. It is scale-independent: it converts *any* in-Lean `Unsat` of this
encoding (K₇/6 today, G₅₂₉ once the proof term scales) into a chromatic bound.
-/
import Std.Tactic.BVDecide.LRAT.Checker

open Std.Sat

namespace SounioSatColouring

/-! ## 1. The colouring CNF encoding (matches souc_sat's emitter) -/

/-- At-least-one clause for each vertex `v < n`: `(v*k+0) ∨ … ∨ (v*k+(k-1))`. -/
def atLeastOne (n k : Nat) : List (CNF.Clause Nat) :=
  (List.range n).map (fun v => (List.range k).map (fun c => (v * k + c, true)))

/-- For each edge `(a,b)` and colour `c`: `¬(a*k+c) ∨ ¬(b*k+c)`. -/
def edgeClauses (k : Nat) (edges : List (Nat × Nat)) : List (CNF.Clause Nat) :=
  edges.flatMap (fun e => (List.range k).map
    (fun c => [(e.1 * k + c, false), (e.2 * k + c, false)]))

/-- Full colouring CNF: at-least-one clauses followed by edge clauses. -/
def colourCNF (n k : Nat) (edges : List (Nat × Nat)) : CNF Nat :=
  { clauses := (atLeastOne n k ++ edgeClauses k edges).toArray }

/-! ## 2. The assignment induced by a colouring -/

/-- Boolean assignment from a colouring `col`: variable `x` is true iff vertex
`x / k` has colour `x % k`. -/
def asg (k : Nat) (col : Nat → Nat) : Nat → Bool :=
  fun x => decide (col (x / k) = x % k)

variable {n k : Nat} {edges : List (Nat × Nat)} {col : Nat → Nat}

/-- Index arithmetic: for `c < k`, the variable `v*k+c` decodes to vertex `v`. -/
private theorem div_idx (v c : Nat) (hk : 0 < k) (hc : c < k) : (v * k + c) / k = v := by
  rw [Nat.mul_comm v k, Nat.mul_add_div hk, Nat.div_eq_of_lt hc, Nat.add_zero]

private theorem mod_idx (v c : Nat) (hk : 0 < k) (hc : c < k) : (v * k + c) % k = c := by
  rw [Nat.mul_comm v k, Nat.mul_add_mod, Nat.mod_eq_of_lt hc]

/-- The induced assignment is true exactly on the chosen-colour variable. -/
theorem asg_iff (v c : Nat) (hk : 0 < k) (hc : c < k) :
    asg k col (v * k + c) = true ↔ col v = c := by
  simp only [asg, decide_eq_true_eq, div_idx v c hk hc, mod_idx v c hk hc]

/-! ## 3. A proper colouring satisfies the encoding -/

/-- `col` is a proper `k`-colouring of `(n, edges)`: in range, and no edge
monochromatic (edge endpoints assumed `< n`). -/
structure Proper (n k : Nat) (edges : List (Nat × Nat)) (col : Nat → Nat) : Prop where
  range  : ∀ v, v < n → col v < k
  endpts : ∀ e ∈ edges, e.1 < n ∧ e.2 < n
  proper : ∀ e ∈ edges, col e.1 ≠ col e.2

theorem sat_atLeastOne (hk : 0 < k) (hp : Proper n k edges col) :
    ∀ cl ∈ atLeastOne n k, CNF.Clause.eval (asg k col) cl = true := by
  intro cl hcl
  simp only [atLeastOne, List.mem_map, List.mem_range] at hcl
  obtain ⟨v, hv, rfl⟩ := hcl
  have hcolv : col v < k := hp.range v hv
  simp only [CNF.Clause.eval, List.any_map, List.any_eq_true, List.mem_range, Function.comp]
  exact ⟨col v, hcolv, by simp [(asg_iff v (col v) hk hcolv).mpr rfl]⟩

theorem sat_edgeClauses (hk : 0 < k) (hp : Proper n k edges col) :
    ∀ cl ∈ edgeClauses k edges, CNF.Clause.eval (asg k col) cl = true := by
  intro cl hcl
  simp only [edgeClauses, List.mem_flatMap, List.mem_map, List.mem_range] at hcl
  obtain ⟨e, he, c, hc, rfl⟩ := hcl
  -- not both endpoints take colour c, else they share a colour (edge ⇒ proper)
  have hne : col e.1 ≠ col e.2 := hp.proper e he
  have key : ¬ (asg k col (e.1 * k + c) = true ∧ asg k col (e.2 * k + c) = true) := by
    rintro ⟨h1, h2⟩
    exact hne (by rw [(asg_iff e.1 c hk hc).mp h1, (asg_iff e.2 c hk hc).mp h2])
  simp only [CNF.Clause.eval, List.any_cons, List.any_nil, Bool.or_false]
  cases h1 : asg k col (e.1 * k + c) <;> cases h2 : asg k col (e.2 * k + c) <;> simp_all

/-- **Soundness of the encoding (Nat level).** If the colouring CNF is
unsatisfiable, no proper `k`-colouring exists. -/
theorem not_proper_of_unsat (hk : 0 < k)
    (h : (colourCNF n k edges).Unsat) : ¬ ∃ col : Nat → Nat, Proper n k edges col := by
  rintro ⟨col, hp⟩
  have hsat : CNF.eval (asg k col) (colourCNF n k edges) = true := by
    simp only [colourCNF, CNF.eval]
    rw [Array.all_eq_true_iff_forall_mem]
    intro cl hcl
    rw [List.mem_toArray, List.mem_append] at hcl
    rcases hcl with hcl | hcl
    · exact sat_atLeastOne hk hp cl hcl
    · exact sat_edgeClauses hk hp cl hcl
  have hfalse := h (asg k col)
  rw [hsat] at hfalse
  exact Bool.noConfusion hfalse

/-- **Soundness, `Fin`-colouring form (the SAT leg).** If the colouring CNF is
unsatisfiable then the graph has **no** proper `k`-colouring `Fin n → Fin k`.
This is the statement consumed by the de Grey χ(ℝ²) ≥ 5 reduction. -/
theorem not_colourable_of_unsat (hk : 0 < k)
    (hedges : ∀ e ∈ edges, e.1 < n ∧ e.2 < n)
    (h : (colourCNF n k edges).Unsat) :
    ¬ ∃ c : Fin n → Fin k,
        ∀ e ∈ edges, ∀ (h1 : e.1 < n) (h2 : e.2 < n), c ⟨e.1, h1⟩ ≠ c ⟨e.2, h2⟩ := by
  rintro ⟨c, hc⟩
  apply not_proper_of_unsat hk h
  refine ⟨fun v => if hv : v < n then (c ⟨v, hv⟩).val else 0, ?_, hedges, ?_⟩
  · intro v hv
    simp only [hv, dif_pos]
    exact (c ⟨v, hv⟩).isLt
  · intro e he
    obtain ⟨h1, h2⟩ := hedges e he
    simp only [h1, h2, dif_pos]
    intro heq
    exact hc e he h1 h2 (Fin.eq_of_val_eq heq)

#print axioms not_proper_of_unsat
#print axioms not_colourable_of_unsat

end SounioSatColouring
