/-!
# Sounio.OntologyClaimStatus — verified epistemic status propagation for claims

Formal companion to
`artifacts/ontology-frontiers/epistemic-claim-status/claim_status.sio`
(frontier: `epistemic-claim-status`; see the `FRONTIER.md` there for the
literature evidence: arXiv:2602.15353, arXiv:2601.21116, arXiv:2604.11759,
arXiv:2603.28444).

## Setting

Knowledge-graph claims carry no machine-checkable epistemic status. This
file fixes the two propagation rules used by the prototype and proves their
contracts, with confidences represented in **per-mille** (`Nat`, 0–1000) so
that all arithmetic is exact and decidable:

1. **Weakest link (derivation).** A claim derived from premises gets the
   minimum premise confidence. Proved: the derived confidence never exceeds
   any premise (`chainConf_le_acc`, `chainConf_le_mem`), and a threshold
   satisfied by every premise is preserved along chains of arbitrary length
   (`chainConf_ge`).

2. **Dempster-Shafer fusion (independent sources).** For per-mille
   confidences `a, b ≤ 1000`, the scaled fusion numerator
   `dsNum a b = 1000·1000 - (1000-a)·(1000-b)` (i.e. `1000·ds(a,b)`) never
   drops below the best single source: `1000·max a b ≤ dsNum a b`
   (`dsNum_ge_max`).

Concrete instances reproduce the numbers of the `.sio` prototype and are
checked by `native_decide`/`decide`.

Self-contained. No Mathlib. Zero sorry. No new axioms.
-/

namespace Sounio.OntologyClaimStatus

-- ---------------------------------------------------------------------------
-- §1. Weakest-link derivation
-- ---------------------------------------------------------------------------

/-- Weakest-link combination: a derived claim inherits the minimum
    confidence of its premises. -/
abbrev weakestLink (a b : Nat) : Nat := min a b

theorem weakestLink_le_left (a b : Nat) : weakestLink a b ≤ a :=
  Nat.min_le_left a b

theorem weakestLink_le_right (a b : Nat) : weakestLink a b ≤ b :=
  Nat.min_le_right a b

/-- Threshold preservation for a single derivation step. -/
theorem weakestLink_ge {t a b : Nat} (ha : t ≤ a) (hb : t ≤ b) :
    t ≤ weakestLink a b := by
  show t ≤ min a b
  omega

-- ---------------------------------------------------------------------------
-- §2. Derivation chains of arbitrary length
-- ---------------------------------------------------------------------------

/-- Confidence of a derivation chain: fold the premises through `min`,
    starting from the confidence `acc` of the first premise. -/
def chainConf (acc : Nat) (l : List Nat) : Nat := l.foldl min acc

/-- The chain confidence never exceeds the first premise's confidence. -/
theorem chainConf_le_acc (acc : Nat) (l : List Nat) : chainConf acc l ≤ acc := by
  induction l generalizing acc with
  | nil => exact Nat.le_refl acc
  | cons x xs ih =>
      have h1 : chainConf acc (x :: xs) = chainConf (min acc x) xs := rfl
      rw [h1]
      exact Nat.le_trans (ih (min acc x)) (Nat.min_le_left acc x)

/-- The chain confidence never exceeds any premise in the chain. -/
theorem chainConf_le_mem (acc : Nat) (l : List Nat) {x : Nat} (hx : x ∈ l) :
    chainConf acc l ≤ x := by
  induction l generalizing acc with
  | nil => cases hx
  | cons y ys ih =>
      rw [List.mem_cons] at hx
      have h1 : chainConf acc (y :: ys) = chainConf (min acc y) ys := rfl
      rw [h1]
      cases hx with
      | inl hxy =>
          have h2 : min acc y ≤ x := by
            rw [hxy]
            exact Nat.min_le_right acc y
          exact Nat.le_trans (chainConf_le_acc (min acc y) ys) h2
      | inr hmem => exact ih (min acc y) hmem

/-- **Threshold preservation**: if the first premise and every premise in
    the chain meet threshold `t`, so does the derived claim. -/
theorem chainConf_ge {t : Nat} (acc : Nat) (l : List Nat)
    (hacc : t ≤ acc) (hl : ∀ x ∈ l, t ≤ x) :
    t ≤ chainConf acc l := by
  induction l generalizing acc with
  | nil => exact hacc
  | cons y ys ih =>
      have h1 : chainConf acc (y :: ys) = chainConf (min acc y) ys := rfl
      rw [h1]
      have hy : t ≤ y := hl y (List.mem_cons.mpr (Or.inl rfl))
      apply ih
      · omega
      · intro x hx
        exact hl x (List.mem_cons.mpr (Or.inr hx))

-- ---------------------------------------------------------------------------
-- §3. Dempster-Shafer fusion of independent sources
-- ---------------------------------------------------------------------------

/-- Scaled Dempster-Shafer numerator for per-mille confidences:
    `dsNum a b = 1000·ds(a,b)` where `ds(a,b) = 1-(1-a/1000)(1-b/1000)`.
    Comparisons against `1000·c` are exact, avoiding division. -/
def dsNum (a b : Nat) : Nat := 1000 * 1000 - (1000 - a) * (1000 - b)

theorem dsNum_comm (a b : Nat) : dsNum a b = dsNum b a := by
  unfold dsNum
  rw [Nat.mul_comm (1000 - a) (1000 - b)]

/-- Fusion never drops below the left source. -/
theorem dsNum_ge_left {a b : Nat} (ha : a ≤ 1000) :
    1000 * a ≤ dsNum a b := by
  have h1 : (1000 - a) * (1000 - b) ≤ (1000 - a) * 1000 :=
    Nat.mul_le_mul (Nat.le_refl (1000 - a)) (Nat.sub_le 1000 b)
  have h2 : (1000 - a) * 1000 = 1000 * 1000 - a * 1000 :=
    Nat.sub_mul 1000 a 1000
  have h3 : 1000 * 1000 - ((1000 - a) * 1000) = a * 1000 := by
    rw [h2]
    have ha2 : a * 1000 ≤ 1000 * 1000 :=
      Nat.mul_le_mul ha (Nat.le_refl 1000)
    exact Nat.sub_sub_self ha2
  calc 1000 * a = a * 1000 := Nat.mul_comm 1000 a
    _ = 1000 * 1000 - ((1000 - a) * 1000) := h3.symm
    _ ≤ 1000 * 1000 - (1000 - a) * (1000 - b) :=
        Nat.sub_le_sub_left h1 (1000 * 1000)
    _ = dsNum a b := rfl

/-- Fusion never drops below the right source. -/
theorem dsNum_ge_right {a b : Nat} (hb : b ≤ 1000) :
    1000 * b ≤ dsNum a b := by
  rw [dsNum_comm]
  exact dsNum_ge_left hb

/-- **DS monotonicity**: fusion never drops below the best single source. -/
theorem dsNum_ge_max {a b : Nat} (ha : a ≤ 1000) (hb : b ≤ 1000) :
    1000 * max a b ≤ dsNum a b := by
  cases Nat.le_total a b with
  | inl hab =>
      rw [Nat.max_eq_right hab]
      exact dsNum_ge_right hb
  | inr hba =>
      rw [Nat.max_eq_left hba]
      exact dsNum_ge_left ha

-- ---------------------------------------------------------------------------
-- §4. Concrete instances (the `.sio` prototype's numbers)
-- ---------------------------------------------------------------------------

/-- Weakest link of the curated pair {0.95, 0.90} is 0.90. -/
theorem ex_weakest : weakestLink 950 900 = 900 := rfl

/-- Chain {0.95, 0.90, 0.88} bottoms out at 0.88. -/
theorem ex_chain : chainConf 950 [900, 880] = 880 := by native_decide

/-- DS(0.60, 0.55) = 0.82 exactly (scaled). -/
theorem ex_ds : dsNum 600 550 = 820000 := by native_decide

/-- The fused claim clears the 0.80 high-status threshold even though each
    single source is below it. -/
theorem ex_ds_lifts :
    550 < 800 ∧ 600 < 800 ∧ 800 * 1000 ≤ dsNum 600 550 := by
  refine ⟨by decide, by decide, ?_⟩
  rw [ex_ds]
  decide

end Sounio.OntologyClaimStatus
