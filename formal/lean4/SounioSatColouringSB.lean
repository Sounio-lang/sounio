/-
# Sounio — triangle-precolour symmetry break is satisfiability-preserving (k = 4)

souc_sat refutes the de Grey 4-colouring CNF *augmented* with three unit clauses
that fix a triangle `a,b,c` to colours `0,1,2` (a complete colour-symmetry break
for k=4: residual S₁). The reflective Lean proof (`SounioSatG529`) therefore gives
`(colourCNFsb a b c n edges).Unsat`.

This file closes the WLOG gap: for a triangle (`a,b,c` pairwise adjacent), that
SB-augmented `Unsat` implies the graph has **no** proper 4-colouring at all —
unconditional χ ≥ 5. The argument is the standard colour-permutation WLOG, made
fully constructive: given any proper colouring `f`, relabel colours by the
explicit bijection `relabel4` so the triangle lands on `0,1,2`; the relabelled
colouring still satisfies the plain encoding *and* the three SB units, so the
SB-augmented CNF would be SAT — contradiction.

`relabel4`'s bijectivity facts are decided by exhaustion over `Fin 4` (no Mathlib,
no `native_decide`). Axioms: `[propext, Quot.sound]`.
-/
import SounioSatColouringBridge

open Std.Sat
open SounioSatColouring

namespace SounioSatColouringSB

/-- Colour relabelling that sends `ca,cb,cc` to `0,1,2` and the remaining colour
to `3`. For pairwise-distinct `ca,cb,cc` this is a bijection of `Fin 4`. -/
def relabel4 (ca cb cc x : Fin 4) : Fin 4 :=
  if x = ca then 0 else if x = cb then 1 else if x = cc then 2 else 3

theorem relabel4_a : ∀ ca cb cc : Fin 4, relabel4 ca cb cc ca = 0 := by decide

theorem relabel4_b : ∀ ca cb cc : Fin 4, ca ≠ cb → relabel4 ca cb cc cb = 1 := by decide

theorem relabel4_c : ∀ ca cb cc : Fin 4, ca ≠ cc → cb ≠ cc → relabel4 ca cb cc cc = 2 := by decide

/-- On pairwise-distinct `ca,cb,cc`, `relabel4` is injective on `Fin 4`. -/
theorem relabel4_inj : ∀ ca cb cc x y : Fin 4,
    ca ≠ cb → ca ≠ cc → cb ≠ cc → x ≠ y →
    relabel4 ca cb cc x ≠ relabel4 ca cb cc y := by decide

/-- The recoloured Nat-colouring induced by `f` and a relabelling. -/
def recolour {n : Nat} (f : Fin n → Fin 4) (ca cb cc : Fin 4) (v : Nat) : Nat :=
  if hv : v < n then (relabel4 ca cb cc (f ⟨v, hv⟩)).val else 0

theorem recolour_app {n : Nat} (f : Fin n → Fin 4) (ca cb cc : Fin 4) {v : Nat}
    (hv : v < n) : recolour f ca cb cc v = (relabel4 ca cb cc (f ⟨v, hv⟩)).val := by
  simp only [recolour, hv, dif_pos]

/-- The de Grey 4-colouring CNF augmented with souc_sat's three triangle-precolour
unit clauses (`a→0`, `b→1`, `c→2`); clause order matches souc_sat's emitter. -/
def colourCNFsb (a b c n : Nat) (edges : List (Nat × Nat)) : CNF Nat :=
  { clauses := (atLeastOne n 4 ++ edgeClauses 4 edges
                ++ [[(a * 4 + 0, true)], [(b * 4 + 1, true)], [(c * 4 + 2, true)]]).toArray }

/-- **Triangle-precolour WLOG (k = 4).** If `a,b,c` are pairwise adjacent and the
SB-augmented 4-colouring CNF is unsatisfiable, then the graph has no proper
4-colouring `Fin n → Fin 4` — i.e. its chromatic number is ≥ 5. -/
theorem not_colourable_of_unsat_tri
    {n : Nat} {edges : List (Nat × Nat)} {a b c : Nat}
    (ha : a < n) (hb : b < n) (hc : c < n)
    (hab : (a, b) ∈ edges) (hac : (a, c) ∈ edges) (hbc : (b, c) ∈ edges)
    (hedges : ∀ e ∈ edges, e.1 < n ∧ e.2 < n)
    (hU : (colourCNFsb a b c n edges).Unsat) :
    ¬ ∃ f : Fin n → Fin 4,
        ∀ e ∈ edges, ∀ (h1 : e.1 < n) (h2 : e.2 < n), f ⟨e.1, h1⟩ ≠ f ⟨e.2, h2⟩ := by
  rintro ⟨f, hf⟩
  -- colours of the triangle vertices are pairwise distinct (they are adjacent)
  have dab : f ⟨a, ha⟩ ≠ f ⟨b, hb⟩ := hf (a, b) hab ha hb
  have dac : f ⟨a, ha⟩ ≠ f ⟨c, hc⟩ := hf (a, c) hac ha hc
  have dbc : f ⟨b, hb⟩ ≠ f ⟨c, hc⟩ := hf (b, c) hbc hb hc
  have hk : (0 : Nat) < 4 := by decide
  -- the relabelled Nat-colouring
  have hproper : Proper n 4 edges (recolour f (f ⟨a, ha⟩) (f ⟨b, hb⟩) (f ⟨c, hc⟩)) := by
    refine ⟨?_, hedges, ?_⟩
    · intro v hv
      rw [recolour_app f _ _ _ hv]
      exact (relabel4 _ _ _ (f ⟨v, hv⟩)).isLt
    · intro e he
      obtain ⟨h1, h2⟩ := hedges e he
      rw [recolour_app f _ _ _ h1, recolour_app f _ _ _ h2]
      intro hval
      exact relabel4_inj _ _ _ (f ⟨e.1, h1⟩) (f ⟨e.2, h2⟩) dab dac dbc
        (hf e he h1 h2) (Fin.eq_of_val_eq hval)
  -- the recolouring fixes the triangle to 0,1,2
  have hna : recolour f (f ⟨a, ha⟩) (f ⟨b, hb⟩) (f ⟨c, hc⟩) a = 0 := by
    rw [recolour_app f _ _ _ ha, relabel4_a]; rfl
  have hnb : recolour f (f ⟨a, ha⟩) (f ⟨b, hb⟩) (f ⟨c, hc⟩) b = 1 := by
    rw [recolour_app f _ _ _ hb, relabel4_b _ _ _ dab]; rfl
  have hnc : recolour f (f ⟨a, ha⟩) (f ⟨b, hb⟩) (f ⟨c, hc⟩) c = 2 := by
    rw [recolour_app f _ _ _ hc, relabel4_c _ _ _ dac dbc]; rfl
  -- the relabelled assignment satisfies the SB-augmented CNF
  have hsat : CNF.eval (asg 4 (recolour f (f ⟨a, ha⟩) (f ⟨b, hb⟩) (f ⟨c, hc⟩)))
      (colourCNFsb a b c n edges) = true := by
    simp only [colourCNFsb, CNF.eval]
    rw [Array.all_eq_true_iff_forall_mem]
    intro cl hcl
    rw [List.mem_toArray, List.mem_append, List.mem_append] at hcl
    rcases hcl with (hcl | hcl) | hcl
    · exact sat_atLeastOne hk hproper cl hcl
    · exact sat_edgeClauses hk hproper cl hcl
    · -- the three precolour unit clauses
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hcl
      rcases hcl with rfl | rfl | rfl
      · simp only [CNF.Clause.eval, List.any_cons, List.any_nil, Bool.or_false]
        rw [(asg_iff a 0 hk (by decide)).mpr hna]; rfl
      · simp only [CNF.Clause.eval, List.any_cons, List.any_nil, Bool.or_false]
        rw [(asg_iff b 1 hk (by decide)).mpr hnb]; rfl
      · simp only [CNF.Clause.eval, List.any_cons, List.any_nil, Bool.or_false]
        rw [(asg_iff c 2 hk (by decide)).mpr hnc]; rfl
  have hfalse := hU (asg 4 (recolour f (f ⟨a, ha⟩) (f ⟨b, hb⟩) (f ⟨c, hc⟩)))
  rw [hsat] at hfalse
  exact Bool.noConfusion hfalse

#print axioms not_colourable_of_unsat_tri

/-! ## The same triangle-precolour WLOG for `k = 5`.

This is the SAT-side shape needed by a future planar chi>=6 attempt: a no-5-colouring
certificate may safely precolour any triangle to colours `0,1,2`, leaving a residual
swap of colours `3,4`.
-/

def firstRemaining5 (ca cb cc : Fin 5) : Fin 5 :=
  if (0 : Fin 5) ≠ ca ∧ (0 : Fin 5) ≠ cb ∧ (0 : Fin 5) ≠ cc then 0
  else if (1 : Fin 5) ≠ ca ∧ (1 : Fin 5) ≠ cb ∧ (1 : Fin 5) ≠ cc then 1
  else if (2 : Fin 5) ≠ ca ∧ (2 : Fin 5) ≠ cb ∧ (2 : Fin 5) ≠ cc then 2
  else if (3 : Fin 5) ≠ ca ∧ (3 : Fin 5) ≠ cb ∧ (3 : Fin 5) ≠ cc then 3
  else 4

/-- Colour relabelling for `Fin 5` that sends `ca,cb,cc` to `0,1,2`, and sends the
first remaining colour to `3` and the last remaining colour to `4`. -/
def relabel5 (ca cb cc x : Fin 5) : Fin 5 :=
  if x = ca then 0
  else if x = cb then 1
  else if x = cc then 2
  else if x = firstRemaining5 ca cb cc then 3
  else 4

theorem relabel5_a : ∀ ca cb cc : Fin 5, relabel5 ca cb cc ca = 0 := by decide

theorem relabel5_b : ∀ ca cb cc : Fin 5, ca ≠ cb → relabel5 ca cb cc cb = 1 := by decide

theorem relabel5_c : ∀ ca cb cc : Fin 5, ca ≠ cc → cb ≠ cc → relabel5 ca cb cc cc = 2 := by decide

/-- On pairwise-distinct `ca,cb,cc`, `relabel5` is injective on `Fin 5`. -/
theorem relabel5_inj : ∀ ca cb cc x y : Fin 5,
    ca ≠ cb → ca ≠ cc → cb ≠ cc → x ≠ y →
    relabel5 ca cb cc x ≠ relabel5 ca cb cc y := by decide

/-- The recoloured Nat-colouring induced by a `Fin 5` colouring and a relabelling. -/
def recolour5 {n : Nat} (f : Fin n → Fin 5) (ca cb cc : Fin 5) (v : Nat) : Nat :=
  if hv : v < n then (relabel5 ca cb cc (f ⟨v, hv⟩)).val else 0

theorem recolour5_app {n : Nat} (f : Fin n → Fin 5) (ca cb cc : Fin 5) {v : Nat}
    (hv : v < n) : recolour5 f ca cb cc v = (relabel5 ca cb cc (f ⟨v, hv⟩)).val := by
  simp only [recolour5, hv, dif_pos]

/-- A 5-colouring CNF augmented with triangle-precolour units (`a→0`, `b→1`, `c→2`). -/
def colourCNFsb5 (a b c n : Nat) (edges : List (Nat × Nat)) : CNF Nat :=
  { clauses := (atLeastOne n 5 ++ edgeClauses 5 edges
                ++ [[(a * 5 + 0, true)], [(b * 5 + 1, true)], [(c * 5 + 2, true)]]).toArray }

/-- **Triangle-precolour WLOG (k = 5).** If `a,b,c` are pairwise adjacent and the
SB-augmented 5-colouring CNF is unsatisfiable, then the graph has no proper
5-colouring `Fin n → Fin 5`. This is the SAT-side bridge for a future chi>=6 witness. -/
theorem not_colourable5_of_unsat_tri
    {n : Nat} {edges : List (Nat × Nat)} {a b c : Nat}
    (ha : a < n) (hb : b < n) (hc : c < n)
    (hab : (a, b) ∈ edges) (hac : (a, c) ∈ edges) (hbc : (b, c) ∈ edges)
    (hedges : ∀ e ∈ edges, e.1 < n ∧ e.2 < n)
    (hU : (colourCNFsb5 a b c n edges).Unsat) :
    ¬ ∃ f : Fin n → Fin 5,
        ∀ e ∈ edges, ∀ (h1 : e.1 < n) (h2 : e.2 < n), f ⟨e.1, h1⟩ ≠ f ⟨e.2, h2⟩ := by
  rintro ⟨f, hf⟩
  have dab : f ⟨a, ha⟩ ≠ f ⟨b, hb⟩ := hf (a, b) hab ha hb
  have dac : f ⟨a, ha⟩ ≠ f ⟨c, hc⟩ := hf (a, c) hac ha hc
  have dbc : f ⟨b, hb⟩ ≠ f ⟨c, hc⟩ := hf (b, c) hbc hb hc
  have hk : (0 : Nat) < 5 := by decide
  have hproper : Proper n 5 edges (recolour5 f (f ⟨a, ha⟩) (f ⟨b, hb⟩) (f ⟨c, hc⟩)) := by
    refine ⟨?_, hedges, ?_⟩
    · intro v hv
      rw [recolour5_app f _ _ _ hv]
      exact (relabel5 _ _ _ (f ⟨v, hv⟩)).isLt
    · intro e he
      obtain ⟨h1, h2⟩ := hedges e he
      rw [recolour5_app f _ _ _ h1, recolour5_app f _ _ _ h2]
      intro hval
      exact relabel5_inj _ _ _ (f ⟨e.1, h1⟩) (f ⟨e.2, h2⟩) dab dac dbc
        (hf e he h1 h2) (Fin.eq_of_val_eq hval)
  have hna : recolour5 f (f ⟨a, ha⟩) (f ⟨b, hb⟩) (f ⟨c, hc⟩) a = 0 := by
    rw [recolour5_app f _ _ _ ha, relabel5_a]; rfl
  have hnb : recolour5 f (f ⟨a, ha⟩) (f ⟨b, hb⟩) (f ⟨c, hc⟩) b = 1 := by
    rw [recolour5_app f _ _ _ hb, relabel5_b _ _ _ dab]; rfl
  have hnc : recolour5 f (f ⟨a, ha⟩) (f ⟨b, hb⟩) (f ⟨c, hc⟩) c = 2 := by
    rw [recolour5_app f _ _ _ hc, relabel5_c _ _ _ dac dbc]; rfl
  have hsat : CNF.eval (asg 5 (recolour5 f (f ⟨a, ha⟩) (f ⟨b, hb⟩) (f ⟨c, hc⟩)))
      (colourCNFsb5 a b c n edges) = true := by
    simp only [colourCNFsb5, CNF.eval]
    rw [Array.all_eq_true_iff_forall_mem]
    intro cl hcl
    rw [List.mem_toArray, List.mem_append, List.mem_append] at hcl
    rcases hcl with (hcl | hcl) | hcl
    · exact sat_atLeastOne hk hproper cl hcl
    · exact sat_edgeClauses hk hproper cl hcl
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at hcl
      rcases hcl with rfl | rfl | rfl
      · simp only [CNF.Clause.eval, List.any_cons, List.any_nil, Bool.or_false]
        rw [(asg_iff a 0 hk (by decide)).mpr hna]; rfl
      · simp only [CNF.Clause.eval, List.any_cons, List.any_nil, Bool.or_false]
        rw [(asg_iff b 1 hk (by decide)).mpr hnb]; rfl
      · simp only [CNF.Clause.eval, List.any_cons, List.any_nil, Bool.or_false]
        rw [(asg_iff c 2 hk (by decide)).mpr hnc]; rfl
  have hfalse := hU (asg 5 (recolour5 f (f ⟨a, ha⟩) (f ⟨b, hb⟩) (f ⟨c, hc⟩)))
  rw [hsat] at hfalse
  exact Bool.noConfusion hfalse

#print axioms not_colourable5_of_unsat_tri

end SounioSatColouringSB
