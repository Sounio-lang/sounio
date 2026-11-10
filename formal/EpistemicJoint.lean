import EpistemicSafety
import EpistemicStat

/-!
# Joint certified + statistical layers, and the statistical contest (no Mathlib)

A `Knowledge` value carries both layers: a certified interval (`Epi.Iv`) and a
statistical source `x ± U` (`Epi.Stat.Src`). `meet` accepts only when both
layers accept, and the refusal says which layer refused, certified first.

The statistical contest is resolved by a third source under the same policy,
and succeeds only when the evidence is consistent with exactly one side.
-/

namespace Epi.Joint

open Epi.Stat (Policy Src consistent fuse Fused)

structure K where
  iv : Epi.Iv
  s : Src

inductive Refusal
  | certified      -- certified intervals are disjoint
  | statistical    -- E_n above the declared policy
  | notIdentifiable
  | noSupport
deriving DecidableEq

structure Out where
  iv : Epi.Iv
  s : Fused

def meet (p : Policy) (a b : K) : Except Refusal Out :=
  if Epi.compat a.iv b.iv then
    if consistent p a.s b.s then .ok ⟨Epi.inter a.iv b.iv, fuse a.s b.s⟩
    else .error .statistical
  else .error .certified

/-- A joint `meet` succeeds only when both layers accept. -/
theorem meet_sound {p : Policy} {a b : K} {v : Out} (h : meet p a b = .ok v) :
    Epi.compat a.iv b.iv ∧ consistent p a.s b.s := by
  unfold meet at h
  split at h
  · split at h
    · exact ⟨by assumption, by assumption⟩
    · cases h
  · cases h

/-- The certified interval of a successful joint `meet` is never a contradiction. -/
theorem meet_ok_interval {p : Policy} {a b : K} {v : Out} (h : meet p a b = .ok v) :
    v.iv.ok := by
  unfold meet at h
  split at h
  · split at h
    · cases h; exact Epi.inter_ok (by assumption)
    · cases h
  · cases h

/-- Disjoint certified intervals refuse at the certified layer, whatever the statistics say. -/
theorem meet_refuses_certified {p : Policy} {a b : K} (h : ¬ Epi.compat a.iv b.iv) :
    meet p a b = .error .certified := by
  simp [meet, h]

/-- Overlapping intervals but E_n above the policy refuse at the statistical layer. -/
theorem meet_refuses_statistical {p : Policy} {a b : K}
    (hc : Epi.compat a.iv b.iv) (hs : ¬ consistent p a.s b.s) :
    meet p a b = .error .statistical := by
  simp [meet, hc, hs]

/-- Statistical contest: two sources kept apart, resolved by a third under policy `p`. -/
def resolve (p : Policy) (a b x : Src) : Except Refusal Fused :=
  if consistent p a x then
    if consistent p b x then .error .notIdentifiable else .ok (fuse a x)
  else if consistent p b x then .ok (fuse b x) else .error .noSupport

/-- Identifiability: resolution succeeds only when the third source is
    consistent with exactly one side under the declared policy. -/
theorem resolve_identifiable {p : Policy} {a b x : Src} {v : Fused}
    (h : resolve p a b x = .ok v) :
    (consistent p a x ∧ ¬ consistent p b x) ∨ (¬ consistent p a x ∧ consistent p b x) := by
  unfold resolve at h
  by_cases hax : consistent p a x <;> by_cases hbx : consistent p b x <;> simp_all

/-- With a third source consistent with both sides, the conflict stays unresolved. -/
theorem resolve_refuses_both {p : Policy} {a b x : Src}
    (hax : consistent p a x) (hbx : consistent p b x) :
    resolve p a b x = .error .notIdentifiable := by
  simp [resolve, hax, hbx]

/-- Witness: labs A = 100 ± 4 and B = 112 ± 5 conflict under the ISO 13528 policy;
    a third lab C = 111 ± 4 is consistent with B only, so the contest resolves to B. -/
example : ¬ consistent Epi.Stat.iso13528 ⟨100, 4⟩ ⟨112, 5⟩ := by decide
example : ¬ consistent Epi.Stat.iso13528 ⟨100, 4⟩ ⟨111, 4⟩ := by decide
example : consistent Epi.Stat.iso13528 ⟨112, 5⟩ ⟨111, 4⟩ := by decide

end Epi.Joint
