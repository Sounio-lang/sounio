/-
# Sounio.EpistemicTensorBound — soundness of the type-level uncertainty bound.

Mathlib-free (Lean 4 core + `Rat`). The runtime epistemic tensor
(`stdlib/linalg/epistemic_tensor_dyn.sio`) propagates per-element GUM variance
through a matmul:

    var(C_ij) = Σ_k ( b_kj² · u(a_ik)²  +  a_ik² · u(b_kj)² )

The type-level rule (`stdlib/linalg/epi_type_bounds.sio`, `eb_matmul`) propagates a
single static envelope per term, summed over the K contractions:

    var_bound = K · ( Mb²·Ea²  +  Ma²·Eb² )     (then unc = √var_bound)

SOUNDNESS THEOREM (`matmul_var_sound`): if every contraction element respects its
leaf square-bounds, the runtime output variance ≤ the type-level envelope sum.
Companion lemmas `add_var_sound` (bias-add) and `act_lipschitz_sound` (activation)
extend soundness to the rest of the forward pass at the VARIANCE level.

SCOPE (honest, per adversarial review):
  * These lemmas bound VARIANCE. The runtime `unc` = √variance; since √ is
    monotone on ℚ≥0, variance-soundness lifts to `unc`-soundness. The √-lift is
    a standard monotonicity fact, stated here, not re-formalized (no √ over ℚ).
  * `act_lipschitz_sound` takes the activation's Lipschitz constant L as a
    hypothesis. For sigmoid L=¼ (σ' = σ(1-σ) ≤ ¼, elementary calculus); for relu
    L=1. The runtime computes the exact L'≤L; the lemma proves L'·u ≤ L·u.
  * OUT OF SCOPE: the runtime `confs` channel (×0.98/×0.99 multiplicative decay,
    `epistemic_tensor_dyn.sio:141-145`) is a heuristic confidence model, NOT a GUM
    quantity, and is deliberately not covered by `EpiBound` or these theorems.
This is the machine-checked core of the `typed_epi_graph_soundness.sio` witness
(which confirms the numeric envelope K·(Mb²Ea²+Ma²Eb²) at runtime, tight at matmul).
-/

namespace Sounio.EpistemicTensorBound

/-- One contraction element: value/uncertainty of the two operands a, b. -/
structure Elt where
  a : Rat
  b : Rat
  ua : Rat
  ub : Rat

/-- Runtime GUM variance contribution of one element: b²·ua² + a²·ub². -/
def vRun (e : Elt) : Rat := e.b * e.b * (e.ua * e.ua) + e.a * e.a * (e.ub * e.ub)

/-- Per-term type-level envelope: Mb²·Ea² + Ma²·Eb². -/
def vEnv (Ma Mb Ea Eb : Rat) : Rat := Mb * Mb * (Ea * Ea) + Ma * Ma * (Eb * Eb)

/-- The leaf square-bounds an element must respect (the `(mag, unc)` refinement). -/
def Bounded (Ma Mb Ea Eb : Rat) (e : Elt) : Prop :=
  e.a * e.a ≤ Ma * Ma ∧ 0 ≤ Ma * Ma ∧
  e.b * e.b ≤ Mb * Mb ∧ 0 ≤ Mb * Mb ∧
  e.ua * e.ua ≤ Ea * Ea ∧ 0 ≤ e.ua * e.ua ∧
  e.ub * e.ub ≤ Eb * Eb ∧ 0 ≤ e.ub * e.ub

/-- PER-TERM SOUNDNESS: a bounded element's runtime variance ≤ the envelope. -/
theorem term_bound (Ma Mb Ea Eb : Rat) (e : Elt) (h : Bounded Ma Mb Ea Eb e) :
    vRun e ≤ vEnv Ma Mb Ea Eb := by
  obtain ⟨ha2, hMa, hb2, hMb, hua2, hua0, hub2, hub0⟩ := h
  -- b²·ua² ≤ Mb²·ua² ≤ Mb²·Ea²
  have h1 : e.b * e.b * (e.ua * e.ua) ≤ Mb * Mb * (Ea * Ea) :=
    Rat.le_trans
      (Rat.mul_le_mul_of_nonneg_right hb2 hua0)
      (Rat.mul_le_mul_of_nonneg_left hua2 hMb)
  -- a²·ub² ≤ Ma²·ub² ≤ Ma²·Eb²
  have h2 : e.a * e.a * (e.ub * e.ub) ≤ Ma * Ma * (Eb * Eb) :=
    Rat.le_trans
      (Rat.mul_le_mul_of_nonneg_right ha2 hub0)
      (Rat.mul_le_mul_of_nonneg_left hub2 hMa)
  exact Rat.le_trans (Rat.add_le_add_right.mpr h1) (Rat.add_le_add_left.mpr h2)

/-- Pointwise ≤ on paired lists lifts to a ≤ on their sums (cast-free induction). -/
theorem sum_le_sum (l : List (Rat × Rat)) (h : ∀ p ∈ l, p.1 ≤ p.2) :
    (l.map Prod.fst).sum ≤ (l.map Prod.snd).sum := by
  induction l with
  | nil => exact Rat.le_refl
  | cons p t ih =>
    have hp : p.1 ≤ p.2 := h p (List.mem_cons_self ..)
    have ht : ∀ q ∈ t, q.1 ≤ q.2 := fun q hq => h q (List.mem_cons_of_mem p hq)
    simp only [List.map_cons, List.sum_cons]
    exact Rat.le_trans (Rat.add_le_add_right.mpr hp) (Rat.add_le_add_left.mpr (ih ht))

/-- MATMUL SOUNDNESS: for a contraction `data` of bounded elements, the runtime
    output variance Σ vRun ≤ the type-level envelope sum Σ vEnv. The type bound is
    therefore a sound over-approximation of the true runtime uncertainty. -/
theorem matmul_var_sound (Ma Mb Ea Eb : Rat) (data : List Elt)
    (hb : ∀ e ∈ data, Bounded Ma Mb Ea Eb e) :
    (data.map vRun).sum ≤ (data.map (fun _ => vEnv Ma Mb Ea Eb)).sum := by
  -- pair each element's runtime variance with its envelope, then lift to sums
  have key := sum_le_sum (data.map (fun e => (vRun e, vEnv Ma Mb Ea Eb)))
    (by
      intro p hp
      simp only [List.mem_map] at hp
      obtain ⟨e, he, rfl⟩ := hp
      exact term_bound Ma Mb Ea Eb e (hb e he))
  simpa [List.map_map, Function.comp] using key

/-- BIAS-ADD SOUNDNESS (variance level): the runtime add-variance cu²+bu² is
    ≤ the type envelope Ec²+Eb² when each input variance is within its bound. -/
theorem add_var_sound (cu bu Ec Eb : Rat)
    (hc : cu * cu ≤ Ec * Ec) (hb : bu * bu ≤ Eb * Eb) :
    cu * cu + bu * bu ≤ Ec * Ec + Eb * Eb :=
  Rat.le_trans (Rat.add_le_add_right.mpr hc) (Rat.add_le_add_left.mpr hb)

/-- ACTIVATION SOUNDNESS LIFT: for an activation with Lipschitz constant `L`
    (sigmoid L=¼, relu L=1), the runtime output uncertainty `Lp * u` — where `Lp`
    is the exact local derivative magnitude with `Lp ≤ L` — is ≤ the type bound
    `L * u`. (The fact that |σ'| ≤ ¼ and |relu'| ≤ 1 is elementary and supplies the
    hypothesis `hLp : Lp ≤ L`; the runtime computes `Lp` exactly.) -/
theorem act_lipschitz_sound (Lp L u : Rat) (hLp : Lp ≤ L) (hu : 0 ≤ u) :
    Lp * u ≤ L * u :=
  Rat.mul_le_mul_of_nonneg_right hLp hu

end Sounio.EpistemicTensorBound
