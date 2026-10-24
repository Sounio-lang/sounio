/-!
# Epistemic safety for a minimal Knowledge/Contest calculus (no Mathlib)

Certified layer only: a `Knowledge` value is an exact integer interval.
Refusal is an explicit outcome, never a value.

Theorems:
* `safety`              — a well-typed `Knowledge` term never evaluates to a contradiction;
* `meet_no_silent`      — a successful `meet` implies its inputs intersect;
* `meet_refuses`        — disjoint inputs make `meet` refuse explicitly;
* `resolve_identifiable`— resolving a contest succeeds only when the evidence
                          is compatible with exactly one side.
-/

namespace Epi

/-- Interval `[lo, hi]` over `Int`. -/
structure Iv where
  lo : Int
  hi : Int
deriving DecidableEq, Repr

def Iv.ok (x : Iv) : Prop := x.lo ≤ x.hi

def inter (a b : Iv) : Iv := ⟨max a.lo b.lo, min a.hi b.hi⟩

/-- Two intervals are compatible iff their intersection is non-empty. -/
def compat (a b : Iv) : Prop := max a.lo b.lo ≤ min a.hi b.hi

instance (a b : Iv) : Decidable (compat a b) := by unfold compat; exact inferInstance

inductive Ty | know | contest
deriving DecidableEq

inductive Val
  | know (x : Iv)
  | contest (a b : Iv)
deriving DecidableEq

inductive Refusal
  | malformed        -- a literal with lo > hi
  | inconsistent     -- meet of disjoint sources
  | notIdentifiable  -- evidence compatible with both sides
  | noSupport        -- evidence compatible with neither side
  | stuck            -- ill-typed use
deriving DecidableEq

inductive Tm
  | lit (x : Iv)
  | meet (t u : Tm)
  | contest (t u : Tm)
  | resolve (c e : Tm)

inductive HasTy : Tm → Ty → Prop
  | lit {x} : HasTy (.lit x) .know
  | meet {t u} : HasTy t .know → HasTy u .know → HasTy (.meet t u) .know
  | contest {t u} : HasTy t .know → HasTy u .know → HasTy (.contest t u) .contest
  | resolve {c e} : HasTy c .contest → HasTy e .know → HasTy (.resolve c e) .know

def eval : Tm → Except Refusal Val
  | .lit x => if x.lo ≤ x.hi then .ok (.know x) else .error .malformed
  | .meet t u =>
    match eval t, eval u with
    | .ok (.know a), .ok (.know b) =>
      if compat a b then .ok (.know (inter a b)) else .error .inconsistent
    | .error r, _ => .error r
    | _, .error r => .error r
    | _, _ => .error .stuck
  | .contest t u =>
    match eval t, eval u with
    | .ok (.know a), .ok (.know b) => .ok (.contest a b)
    | .error r, _ => .error r
    | _, .error r => .error r
    | _, _ => .error .stuck
  | .resolve c e =>
    match eval c, eval e with
    | .ok (.contest a b), .ok (.know x) =>
      if compat a x then
        if compat b x then .error .notIdentifiable else .ok (.know (inter a x))
      else if compat b x then .ok (.know (inter b x)) else .error .noSupport
    | .error r, _ => .error r
    | _, .error r => .error r
    | _, _ => .error .stuck

/-- Well-formed values: every interval inside is non-empty. -/
def Val.ok : Val → Prop
  | .know x => x.ok
  | .contest a b => a.ok ∧ b.ok

theorem inter_ok {a b : Iv} (h : compat a b) : (inter a b).ok := h

/-- Type preservation with well-formedness: `know` terms yield valid
    intervals, `contest` terms yield a pair of valid intervals. -/
theorem preservation : ∀ {t : Tm} {T : Ty} {v : Val},
    HasTy t T → eval t = .ok v →
    match T with
    | .know => ∃ x, v = .know x ∧ x.ok
    | .contest => ∃ a b, v = .contest a b ∧ a.ok ∧ b.ok
  | .lit x, _, v, .lit, h => by
    simp only [eval] at h
    split at h
    · cases h; exact ⟨x, rfl, by assumption⟩
    · cases h
  | .meet t u, _, v, .meet ht hu, h => by
    have iht := @preservation t .know
    have ihu := @preservation u .know
    simp only [eval] at h
    split at h
    · rename_i a b hta hub
      obtain ⟨_, ha1, ha2⟩ := iht ht hta
      split at h
      · cases h; rename_i hc; exact ⟨_, rfl, inter_ok hc⟩
      · cases h
    all_goals (first | cases h | skip)
    all_goals
      rename_i hne
      first
        | (rename_i x y hx hy; obtain ⟨_, h1, _⟩ := iht ht hx; obtain ⟨_, h2, _⟩ := ihu hu hy
           subst h1; subst h2; exact absurd rfl (hne _ _ rfl rfl))
        | (exact absurd h (by simp))
  | .contest t u, _, v, .contest ht hu, h => by
    have iht := @preservation t .know
    have ihu := @preservation u .know
    simp only [eval] at h
    split at h
    · rename_i a b hta hub
      cases h
      obtain ⟨_, ha1, ha2⟩ := iht ht hta
      obtain ⟨_, hb1, hb2⟩ := ihu hu hub
      cases ha1; cases hb1
      exact ⟨a, b, rfl, ha2, hb2⟩
    all_goals (first | cases h | skip)
    all_goals
      rename_i hne
      first
        | (rename_i x y hx hy; obtain ⟨_, h1, _⟩ := iht ht hx; obtain ⟨_, h2, _⟩ := ihu hu hy
           subst h1; subst h2; exact absurd rfl (hne _ _ rfl rfl))
        | (exact absurd h (by simp))
  | .resolve c e, _, v, .resolve hc he, h => by
    have ihc := @preservation c .contest
    have ihe := @preservation e .know
    simp only [eval] at h
    split at h
    · rename_i a b x hca hex
      split at h
      · split at h
        · cases h
        · cases h; rename_i hcx _; exact ⟨_, rfl, inter_ok hcx⟩
      · split at h
        · cases h; rename_i _ hbx; exact ⟨_, rfl, inter_ok hbx⟩
        · cases h
    all_goals (first | cases h | skip)
    all_goals
      rename_i hne
      first
        | (rename_i y z hy hz; obtain ⟨_, _, h1, _⟩ := ihc hc hy; obtain ⟨_, h2, _⟩ := ihe he hz
           subst h1; subst h2; exact absurd rfl (hne _ _ _ rfl rfl))
        | (exact absurd h (by simp))

/-- Safety: a well-typed `Knowledge` term never evaluates to a contradiction. -/
theorem safety {t : Tm} {v : Val} (ht : HasTy t .know) (h : eval t = .ok v) :
    ∃ x, v = .know x ∧ x.ok :=
  preservation ht h

/-- No silent fusion: a successful `meet` of two evaluated sources means they intersect. -/
theorem meet_no_silent {t u : Tm} {a b x : Iv}
    (ha : eval t = .ok (.know a)) (hb : eval u = .ok (.know b))
    (h : eval (.meet t u) = .ok (.know x)) : compat a b ∧ x = inter a b := by
  simp only [eval, ha, hb] at h
  split at h
  · cases h; exact ⟨by assumption, rfl⟩
  · cases h

/-- Disjoint sources make `meet` refuse explicitly. -/
theorem meet_refuses {t u : Tm} {a b : Iv}
    (ha : eval t = .ok (.know a)) (hb : eval u = .ok (.know b)) (hd : ¬ compat a b) :
    eval (.meet t u) = .error .inconsistent := by
  simp [eval, ha, hb, hd]

/-- Identifiability: resolving succeeds only if the evidence is compatible
    with exactly one side of the contest. -/
theorem resolve_identifiable {c e : Tm} {a b x : Iv} {v : Val}
    (hc : eval c = .ok (.contest a b)) (he : eval e = .ok (.know x))
    (h : eval (.resolve c e) = .ok v) :
    (compat a x ∧ ¬ compat b x) ∨ (¬ compat a x ∧ compat b x) := by
  simp only [eval, hc, he] at h
  by_cases hax : compat a x <;> by_cases hbx : compat b x <;> simp_all

/-- With evidence compatible with both sides, resolution is refused. -/
theorem resolve_refuses_both {c e : Tm} {a b x : Iv}
    (hc : eval c = .ok (.contest a b)) (he : eval e = .ok (.know x))
    (hax : compat a x) (hbx : compat b x) :
    eval (.resolve c e) = .error .notIdentifiable := by
  simp [eval, hc, he, hax, hbx]

end Epi
