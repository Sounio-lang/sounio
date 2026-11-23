/-!
# Sounio — Discharge Trichotomy (kernel of the epistemic decision)

Mathlib-free. Every quantity is an integer in one common fixed-point unit.

Three channels stand between a computed number and a decision about the world:

* `rad` — numerical error. `y` is the exactly-computed value; an enclosure
  `E = (mid, rad)` is sound when `|y - mid| ≤ rad`. Computation shrinks `rad`.
* `ep`  — epistemic uncertainty. The true centre `θ` satisfies `|θ - y| ≤ ep`.
  Only a new measurement shrinks `ep`.
* `al`  — aleatory variability. The realised outcome `η` satisfies
  `|η - θ| ≤ al`. Nothing shrinks `al`.

The decision is whether `η` lies above or below a threshold `c`.

Main results
* soundness of `Above` / `Below`                     (`above_sound`, `below_sound`)
* totality with slack `d`                            (`total`)
* no flip under sound refinement                     (`no_flip`)
* an unsound bound is detectable                     (`contradiction_detectable`)
* computation decides  iff outside the `ep + al` band (`compute_irreducible`, `compute_sufficient`)
* measurement decides  iff outside the `al` band      (`measure_irreducible`, `measure_sufficient`)
* inside a band, abstention is forced, not cautious   (`abstention_necessary`, `aleatory_witness`)

STATUS: written but NOT yet compiled by the author of this file. The eleven
statements were checked exhaustively on a bounded integer box (C++23,
229 603 instances, no counterexample). Run `lean SounioDischargeTrichotomy.lean`
and `#print axioms` before relying on it.
-/

namespace Sounio.Discharge

/-- Numerical enclosure of the exactly-computed value. -/
structure Encl where
  mid : Int
  rad : Int

/-- `E` is sound for `y` when `|y - mid| ≤ rad`. -/
abbrev Encl.Sound (E : Encl) (y : Int) : Prop :=
  E.mid - E.rad ≤ y ∧ y ≤ E.mid + E.rad

/-- `a` lies within `w` of `b`. -/
abbrev Near (a b w : Int) : Prop :=
  b - w ≤ a ∧ a ≤ b + w

/-- Certified above the threshold, all three channels accounted for. -/
abbrev Above (c ep al : Int) (E : Encl) : Prop :=
  c < E.mid - E.rad - ep - al

/-- Certified below the threshold. -/
abbrev Below (c ep al : Int) (E : Encl) : Prop :=
  E.mid + E.rad + ep + al < c

/-- Abstention region, widened by a slack `d` so that the three regions overlap. -/
abbrev Abstain (c ep al d : Int) (E : Encl) : Prop :=
  c - ep - al - d < E.mid - E.rad ∧ E.mid + E.rad < c + ep + al + d

/-! ## Soundness -/

theorem above_sound {c ep al y θ η : Int} {E : Encl}
    (hs : E.Sound y) (hθ : Near θ y ep) (hη : Near η θ al)
    (ha : Above c ep al E) : c < η := by
  obtain ⟨s1, _⟩ := hs
  obtain ⟨t1, _⟩ := hθ
  obtain ⟨e1, _⟩ := hη
  have ha' : c < E.mid - E.rad - ep - al := ha
  omega

theorem below_sound {c ep al y θ η : Int} {E : Encl}
    (hs : E.Sound y) (hθ : Near θ y ep) (hη : Near η θ al)
    (hb : Below c ep al E) : η < c := by
  obtain ⟨_, s2⟩ := hs
  obtain ⟨_, t2⟩ := hθ
  obtain ⟨_, e2⟩ := hη
  have hb' : E.mid + E.rad + ep + al < c := hb
  omega

/-! ## Totality: refinement always ends in one of three verdicts -/

theorem total {c ep al d : Int} {E : Encl} (h : 2 * E.rad < d) :
    Above c ep al E ∨ Below c ep al E ∨ Abstain c ep al d E := by
  by_cases h1 : c < E.mid - E.rad - ep - al
  · exact Or.inl h1
  · by_cases h2 : E.mid + E.rad + ep + al < c
    · exact Or.inr (Or.inl h2)
    · refine Or.inr (Or.inr ?_)
      show c - ep - al - d < E.mid - E.rad ∧ E.mid + E.rad < c + ep + al + d
      constructor
      · omega
      · omega

/-! ## Monotonicity is a theorem, given a fixed referent -/

theorem no_flip {c ep al y : Int} {E₁ E₂ : Encl}
    (h1 : E₁.Sound y) (h2 : E₂.Sound y) (hep : 0 ≤ ep) (hal : 0 ≤ al)
    (ha : Above c ep al E₁) (hb : Below c ep al E₂) : False := by
  obtain ⟨a1, _⟩ := h1
  obtain ⟨_, b2⟩ := h2
  have ha' : c < E₁.mid - E₁.rad - ep - al := ha
  have hb' : E₂.mid + E₂.rad + ep + al < c := hb
  omega

/-- Two disjoint enclosures cannot both be sound: a lying bound is observable. -/
theorem contradiction_detectable {y : Int} {E₁ E₂ : Encl}
    (hd : E₁.mid + E₁.rad < E₂.mid - E₂.rad)
    (h1 : E₁.Sound y) (h2 : E₂.Sound y) : False := by
  obtain ⟨_, a2⟩ := h1
  obtain ⟨b1, _⟩ := h2
  omega

/-! ## What computation can and cannot discharge -/

/-- Inside the `ep + al` band no sound enclosure, however tight, certifies. -/
theorem compute_irreducible {c ep al y : Int} {E : Encl}
    (hlo : c - ep - al ≤ y) (hhi : y ≤ c + ep + al) (hs : E.Sound y) :
    ¬ Above c ep al E ∧ ¬ Below c ep al E := by
  obtain ⟨s1, s2⟩ := hs
  constructor
  · intro ha
    have ha' : c < E.mid - E.rad - ep - al := ha
    omega
  · intro hb
    have hb' : E.mid + E.rad + ep + al < c := hb
    omega

/-- Outside the band, a tight enough sound enclosure certifies. -/
theorem compute_sufficient {c ep al y : Int} {E : Encl}
    (hs : E.Sound y)
    (htight : 2 * E.rad < y - (c + ep + al)) : Above c ep al E := by
  obtain ⟨_, s2⟩ := hs
  show c < E.mid - E.rad - ep - al
  omega

/-! ## What measurement can and cannot discharge -/

/-- Inside the aleatory band no measurement and no computation certifies. -/
theorem measure_irreducible {c ep al y θ : Int} {E : Encl}
    (hlo : c - al ≤ θ) (hhi : θ ≤ c + al)
    (hθ : Near θ y ep) (hs : E.Sound y) :
    ¬ Above c ep al E ∧ ¬ Below c ep al E := by
  obtain ⟨s1, s2⟩ := hs
  obtain ⟨t1, t2⟩ := hθ
  constructor
  · intro ha
    have ha' : c < E.mid - E.rad - ep - al := ha
    omega
  · intro hb
    have hb' : E.mid + E.rad + ep + al < c := hb
    omega

/-- Outside the aleatory band, a precise enough measurement plus computation certifies. -/
theorem measure_sufficient {c ep al y θ : Int} {E : Encl}
    (hθ : Near θ y ep) (hs : E.Sound y)
    (htight : 2 * (ep + E.rad) < θ - (c + al)) : Above c ep al E := by
  obtain ⟨_, s2⟩ := hs
  obtain ⟨_, t2⟩ := hθ
  show c < E.mid - E.rad - ep - al
  omega

/-! ## Abstention is forced: both outcomes are consistent with everything known -/

theorem abstention_necessary {c ep al y : Int}
    (hep : 0 ≤ ep) (hal : 0 ≤ al)
    (hlo : c - ep - al ≤ y) (hhi : y ≤ c + ep + al) :
    (∃ θ η, Near θ y ep ∧ Near η θ al ∧ c ≤ η) ∧
    (∃ θ η, Near θ y ep ∧ Near η θ al ∧ η ≤ c) := by
  constructor
  · refine ⟨y + ep, y + ep + al, ⟨?_, ?_⟩, ⟨?_, ?_⟩, ?_⟩ <;> omega
  · refine ⟨y - ep, y - ep - al, ⟨?_, ?_⟩, ⟨?_, ?_⟩, ?_⟩ <;> omega

/-- Even with the centre known exactly, inside the aleatory band both outcomes occur. -/
theorem aleatory_witness {c al θ : Int}
    (hlo : c - al ≤ θ) (hhi : θ ≤ c + al) :
    (∃ η, Near η θ al ∧ c ≤ η) ∧ (∃ η, Near η θ al ∧ η ≤ c) := by
  constructor
  · refine ⟨θ + al, ⟨?_, ?_⟩, ?_⟩ <;> omega
  · refine ⟨θ - al, ⟨?_, ?_⟩, ?_⟩ <;> omega

end Sounio.Discharge

#print axioms Sounio.Discharge.total
#print axioms Sounio.Discharge.no_flip
#print axioms Sounio.Discharge.compute_irreducible
#print axioms Sounio.Discharge.measure_irreducible
#print axioms Sounio.Discharge.abstention_necessary
