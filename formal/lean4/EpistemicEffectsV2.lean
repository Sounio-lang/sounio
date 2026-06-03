import EpistemicEffects

/-!
# Epistemic-Effect Calculus V2 — value-carrying `Knowledge<T>`

Successor to `EpistemicEffects.lean`.  The runtime Knowledge value `kraw` now
**stores a value of type `T`** (plus scalar GUM metadata), so generic
`Knowledge<T>` is sound — fixing the machine-checked subject-reduction gap of
the scalar-cell calculus (see `docs/research/epistemic_calculus_value_carrying_
redesign.md` and `EpistemicPreservationWIP_counterexample.lean`).

This module is developed independently; the committed Progress proof in
`EpistemicEffects.lean` stays green.  Reuses the effect lattice and `Ty` from
the original module; redefines `Expr`/`HasTy`/`Step` with the value-carrying
`kraw`.

Design: `kvalue : Knowledge<T> → T` returns the stored value (sound for any T);
`meas_red` needs no type annotation (the value carries T); `kadd`/`kmul`
(GUM arithmetic) are restricted to `Knowledge<ℝ>` — numeric by nature.
-/

namespace Sounio.EpistemicEffectsV2

open Sounio.EpistemicEffects
  (Effect Ty EffectSet emptyE singleE unionE subE TyCtx lookupCtx)

-- scalar GUM metadata (variance, confidence) — the value lives in the payload now
structure KMeta where
  gumVar : Int
  conf   : Int
  deriving Repr, DecidableEq

def kvalid (m : KMeta) : Prop := 0 ≤ m.gumVar ∧ 0 ≤ m.conf ∧ m.conf ≤ 1000

-- §1. Syntax — `measure`/`kraw` carry metadata; `kraw` STORES a value Expr.
inductive Expr where
  | lit_nat  : Nat → Expr
  | lit_real : Int → Expr
  | var      : Nat → Expr
  | lam      : Ty → EffectSet → Expr → Expr
  | app      : Expr → Expr → Expr
  | measure  : Expr → KMeta → Expr
  | kvalue   : Expr → Expr
  | kunc     : Expr → Expr
  | kconf    : Expr → Expr
  | kadd     : Expr → Expr → Expr
  | kmul     : Expr → Expr → Expr
  | letE     : Expr → Expr → Expr
  | kraw     : Expr → KMeta → Expr    -- runtime Knowledge: payload value + metadata

-- §2. Values
inductive IsValue : Expr → Prop where
  | v_nat   : ∀ n, IsValue (.lit_nat n)
  | v_real  : ∀ z, IsValue (.lit_real z)
  | v_lam   : ∀ T E e, IsValue (.lam T E e)
  | v_kraw  : ∀ {v} m, IsValue v → IsValue (.kraw v m)

-- §3. Typing  Γ ⊢ e : T ! E
inductive HasTy : TyCtx → Expr → Ty → EffectSet → Prop where
  | t_lit_nat  : ∀ Γ n, HasTy Γ (.lit_nat n) .tnat emptyE
  | t_lit_real : ∀ Γ z, HasTy Γ (.lit_real z) .treal emptyE
  | t_var      : ∀ Γ n T, lookupCtx Γ n = some T → HasTy Γ (.var n) T emptyE
  | t_lam      : ∀ Γ T₁ T₂ E body,
      HasTy (T₁ :: Γ) body T₂ E → HasTy Γ (.lam T₁ E body) (.tarrow T₁ E T₂) emptyE
  | t_app      : ∀ Γ T₁ T₂ Ef Ec Ecaller f a,
      HasTy Γ f (.tarrow T₁ Ef T₂) Ec → HasTy Γ a T₁ Ec →
      Ef ⊆ₑ Ecaller → Ec ⊆ₑ Ecaller → HasTy Γ (.app f a) T₂ Ecaller
  | t_measure  : ∀ Γ T e m,
      HasTy Γ e T emptyE → kvalid m → HasTy Γ (.measure e m) (.tknow T) (singleE .eObserve)
  | t_kvalue   : ∀ Γ T E e,
      HasTy Γ e (.tknow T) E → HasTy Γ (.kvalue e) T E
  | t_kunc     : ∀ Γ T E e,
      HasTy Γ e (.tknow T) E → HasTy Γ (.kunc e) .treal E
  | t_kconf    : ∀ Γ T E e,
      HasTy Γ e (.tknow T) E → HasTy Γ (.kconf e) .treal E
  | t_kadd     : ∀ Γ E₁ E₂ a b,
      HasTy Γ a (.tknow .treal) E₁ → HasTy Γ b (.tknow .treal) E₂ →
      HasTy Γ (.kadd a b) (.tknow .treal) (unionE E₁ E₂)
  | t_kmul     : ∀ Γ E₁ E₂ a b,
      HasTy Γ a (.tknow .treal) E₁ → HasTy Γ b (.tknow .treal) E₂ →
      HasTy Γ (.kmul a b) (.tknow .treal) (unionE E₁ E₂)
  | t_let      : ∀ Γ T₁ T₂ E₁ E₂ e body,
      HasTy Γ e T₁ E₁ → HasTy (T₁ :: Γ) body T₂ E₂ →
      HasTy Γ (.letE e body) T₂ (unionE E₁ E₂)
  | t_kraw     : ∀ Γ T v m,
      HasTy Γ v T emptyE → IsValue v → kvalid m →
      HasTy Γ (.kraw v m) (.tknow T) emptyE
  | t_sub      : ∀ Γ e T E E',
      HasTy Γ e T E → E ⊆ₑ E' → HasTy Γ e T E'

-- §4. GUM metadata combinators (numeric; values are reals)
def gAddMeta (ma mb : KMeta) : KMeta :=
  { gumVar := ma.gumVar + mb.gumVar, conf := if ma.conf ≤ mb.conf then ma.conf else mb.conf }
def gMulMeta (x : Int) (ma : KMeta) (y : Int) (mb : KMeta) : KMeta :=
  { gumVar := y * y * ma.gumVar + x * x * mb.gumVar
  , conf := if ma.conf ≤ mb.conf then ma.conf else mb.conf }

-- §5. de Bruijn shift / subst — `kraw` payload is a recursive sub-term
def shift (cutoff : Nat) (d : Int) : Expr → Expr
  | .var k =>
      if k < cutoff then .var k
      else match d with
        | .ofNat dn => .var (k + dn)
        | .negSucc dn => .var (k - (dn + 1))
  | .lit_nat n => .lit_nat n
  | .lit_real z => .lit_real z
  | .lam T E b => .lam T E (shift (cutoff + 1) d b)
  | .app f a => .app (shift cutoff d f) (shift cutoff d a)
  | .measure e m => .measure (shift cutoff d e) m
  | .kvalue e => .kvalue (shift cutoff d e)
  | .kunc e => .kunc (shift cutoff d e)
  | .kconf e => .kconf (shift cutoff d e)
  | .kadd a b => .kadd (shift cutoff d a) (shift cutoff d b)
  | .kmul a b => .kmul (shift cutoff d a) (shift cutoff d b)
  | .letE e b => .letE (shift cutoff d e) (shift (cutoff + 1) d b)
  | .kraw v m => .kraw (shift cutoff d v) m

def subst (n : Nat) (w : Expr) : Expr → Expr
  | .var k => if k = n then w else if k > n then .var (k - 1) else .var k
  | .lit_nat m => .lit_nat m
  | .lit_real z => .lit_real z
  | .lam T E b => .lam T E (subst (n + 1) (shift 0 1 w) b)
  | .app f a => .app (subst n w f) (subst n w a)
  | .measure e m => .measure (subst n w e) m
  | .kvalue e => .kvalue (subst n w e)
  | .kunc e => .kunc (subst n w e)
  | .kconf e => .kconf (subst n w e)
  | .kadd a b => .kadd (subst n w a) (subst n w b)
  | .kmul a b => .kmul (subst n w a) (subst n w b)
  | .letE e b => .letE (subst n w e) (subst (n + 1) (shift 0 1 w) b)
  | .kraw v m => .kraw (subst n w v) m

-- §6. Small-step CBV reduction
inductive Step : Expr → Expr → Prop where
  | beta       : IsValue v → Step (.app (.lam T E body) v) (subst 0 v body)
  | app_l      : Step f f' → Step (.app f a) (.app f' a)
  | app_r      : IsValue f → Step a a' → Step (.app f a) (.app f a')
  | meas_red   : IsValue v → Step (.measure v m) (.kraw v m)
  | meas_arg   : Step e e' → Step (.measure e m) (.measure e' m)
  | kvalue_red : Step (.kvalue (.kraw v m)) v
  | kvalue_arg : Step e e' → Step (.kvalue e) (.kvalue e')
  | kunc_red   : Step (.kunc (.kraw v m)) (.lit_real m.gumVar)
  | kunc_arg   : Step e e' → Step (.kunc e) (.kunc e')
  | kconf_red  : Step (.kconf (.kraw v m)) (.lit_real m.conf)
  | kconf_arg  : Step e e' → Step (.kconf e) (.kconf e')
  | kadd_red   : Step (.kadd (.kraw (.lit_real x) ma) (.kraw (.lit_real y) mb))
                      (.kraw (.lit_real (x + y)) (gAddMeta ma mb))
  | kadd_l     : Step e e' → Step (.kadd e r) (.kadd e' r)
  | kadd_r     : IsValue v → Step e e' → Step (.kadd v e) (.kadd v e')
  | kmul_red   : Step (.kmul (.kraw (.lit_real x) ma) (.kraw (.lit_real y) mb))
                      (.kraw (.lit_real (x * y)) (gMulMeta x ma y mb))
  | kmul_l     : Step e e' → Step (.kmul e r) (.kmul e' r)
  | kmul_r     : IsValue v → Step e e' → Step (.kmul v e) (.kmul v e')
  | let_red    : IsValue v → Step (.letE v body) (subst 0 v body)
  | let_step   : Step e e' → Step (.letE e b) (.letE e' b)

infix:50 " ⇒ " => Step

-- ================================================================
-- §7. Generation, canonical forms, and Progress
-- ================================================================

theorem genNat {Γ e T E} (h : HasTy Γ e T E) {n} (he : e = .lit_nat n) : T = .tnat := by
  induction h with
  | t_lit_nat => rfl
  | t_sub _ _ _ _ _ _ _ ih => exact ih he
  | _ => exact Expr.noConfusion he

theorem genReal {Γ e T E} (h : HasTy Γ e T E) {z} (he : e = .lit_real z) : T = .treal := by
  induction h with
  | t_lit_real => rfl
  | t_sub _ _ _ _ _ _ _ ih => exact ih he
  | _ => exact Expr.noConfusion he

theorem genLam {Γ e T E S F b} (h : HasTy Γ e T E) (he : e = .lam S F b) :
    ∃ T₂, T = .tarrow S F T₂ := by
  induction h with
  | t_lam Γ T₁ T₂ E body _ => injection he with h1 h2 h3; subst h1; subst h2; exact ⟨T₂, rfl⟩
  | t_sub _ _ _ _ _ _ _ ih => exact ih he
  | _ => exact Expr.noConfusion he

theorem genKraw {Γ e T E} (h : HasTy Γ e T E) {v m} (he : e = .kraw v m) :
    ∃ T', T = .tknow T' := by
  induction h with
  | t_kraw Γ T' v' m' _ _ _ => exact ⟨T', rfl⟩
  | t_sub _ _ _ _ _ _ _ ih => exact ih he
  | _ => exact Expr.noConfusion he

/-- `kraw` inversion: recover the payload type, its typing, and its value-ness. -/
theorem invKraw {Γ e T E} (h : HasTy Γ e T E) {v m} (he : e = .kraw v m) :
    ∃ T', T = .tknow T' ∧ HasTy Γ v T' emptyE ∧ IsValue v := by
  induction h with
  | t_kraw Γ' T' v' m' hv' hval' hk' =>
    injection he with h1 h2; subst h1; subst h2; exact ⟨T', rfl, hv', hval'⟩
  | t_sub Γ' e' T0 E1 E2 h0 hsub ih => exact ih he
  | _ => exact Expr.noConfusion he

theorem canon_arrow {v S F T₂ E} (hv : IsValue v) (ht : HasTy [] v (.tarrow S F T₂) E) :
    ∃ S' F' b, v = .lam S' F' b := by
  cases hv with
  | v_nat n  => exact Ty.noConfusion (genNat ht rfl)
  | v_real z => exact Ty.noConfusion (genReal ht rfl)
  | v_lam T E0 e0 => exact ⟨T, E0, e0, rfl⟩
  | @v_kraw w m hp => rcases genKraw ht rfl with ⟨T', hT⟩; exact Ty.noConfusion hT

theorem canon_know {v T E} (hv : IsValue v) (ht : HasTy [] v (.tknow T) E) :
    ∃ w m, v = .kraw w m := by
  cases hv with
  | @v_kraw w m hp => exact ⟨w, m, rfl⟩
  | v_nat n  => exact Ty.noConfusion (genNat ht rfl)
  | v_real z => exact Ty.noConfusion (genReal ht rfl)
  | v_lam T0 E0 e0 => rcases genLam ht rfl with ⟨T₂, hT⟩; exact Ty.noConfusion hT

theorem canon_real {v E} (hv : IsValue v) (ht : HasTy [] v .treal E) :
    ∃ z, v = .lit_real z := by
  cases hv with
  | v_real z => exact ⟨z, rfl⟩
  | v_nat n  => exact Ty.noConfusion (genNat ht rfl)
  | v_lam T0 E0 e0 => rcases genLam ht rfl with ⟨T₂, hT⟩; exact Ty.noConfusion hT
  | @v_kraw w m hp => rcases genKraw ht rfl with ⟨T', hT⟩; exact Ty.noConfusion hT

theorem progress' {Γ e T E} (h : HasTy Γ e T E) (hΓ : Γ = []) :
    IsValue e ∨ ∃ e', e ⇒ e' := by
  induction h with
  | t_lit_nat Γ n => exact Or.inl (.v_nat n)
  | t_lit_real Γ z => exact Or.inl (.v_real z)
  | t_var Γ n T hlk => subst hΓ; simp [lookupCtx] at hlk
  | t_lam Γ T₁ T₂ E body _ => exact Or.inl (.v_lam _ _ _)
  | t_app Γ T₁ T₂ Ef Ec Ecaller f a hf ha _ _ ihf iha =>
    subst hΓ
    rcases ihf rfl with hvf | ⟨f', hf'⟩
    · rcases iha rfl with hva | ⟨a', ha'⟩
      · rcases canon_arrow hvf hf with ⟨S, F, b, rfl⟩
        exact Or.inr ⟨subst 0 a b, .beta hva⟩
      · exact Or.inr ⟨.app f a', .app_r hvf ha'⟩
    · exact Or.inr ⟨.app f' a, .app_l hf'⟩
  | t_measure Γ T e m he _ ihe =>
    subst hΓ
    rcases ihe rfl with hv | ⟨e', he'⟩
    · exact Or.inr ⟨.kraw e m, .meas_red hv⟩
    · exact Or.inr ⟨.measure e' m, .meas_arg he'⟩
  | t_kvalue Γ T E e he ihe =>
    subst hΓ
    rcases ihe rfl with hv | ⟨e', he'⟩
    · rcases canon_know hv he with ⟨w, m, rfl⟩
      exact Or.inr ⟨w, .kvalue_red⟩
    · exact Or.inr ⟨.kvalue e', .kvalue_arg he'⟩
  | t_kunc Γ T E e he ihe =>
    subst hΓ
    rcases ihe rfl with hv | ⟨e', he'⟩
    · rcases canon_know hv he with ⟨w, m, rfl⟩
      exact Or.inr ⟨.lit_real m.gumVar, .kunc_red⟩
    · exact Or.inr ⟨.kunc e', .kunc_arg he'⟩
  | t_kconf Γ T E e he ihe =>
    subst hΓ
    rcases ihe rfl with hv | ⟨e', he'⟩
    · rcases canon_know hv he with ⟨w, m, rfl⟩
      exact Or.inr ⟨.lit_real m.conf, .kconf_red⟩
    · exact Or.inr ⟨.kconf e', .kconf_arg he'⟩
  | t_kadd Γ E₁ E₂ a b ha hb iha ihb =>
    subst hΓ
    rcases iha rfl with hva | ⟨a', ha'⟩
    · rcases canon_know hva ha with ⟨wa, ma, rfl⟩
      rcases invKraw ha rfl with ⟨Ta, hTa, hwaty, hwaval⟩
      injection hTa with hTa'; subst hTa'
      rcases canon_real hwaval hwaty with ⟨x, rfl⟩
      rcases ihb rfl with hvb | ⟨b', hb'⟩
      · rcases canon_know hvb hb with ⟨wb, mb, rfl⟩
        rcases invKraw hb rfl with ⟨Tb, hTb, hwbty, hwbval⟩
        injection hTb with hTb'; subst hTb'
        rcases canon_real hwbval hwbty with ⟨y, rfl⟩
        exact Or.inr ⟨_, .kadd_red⟩
      · exact Or.inr ⟨_, .kadd_r (.v_kraw ma (.v_real x)) hb'⟩
    · exact Or.inr ⟨_, .kadd_l ha'⟩
  | t_kmul Γ E₁ E₂ a b ha hb iha ihb =>
    subst hΓ
    rcases iha rfl with hva | ⟨a', ha'⟩
    · rcases canon_know hva ha with ⟨wa, ma, rfl⟩
      rcases invKraw ha rfl with ⟨Ta, hTa, hwaty, hwaval⟩
      injection hTa with hTa'; subst hTa'
      rcases canon_real hwaval hwaty with ⟨x, rfl⟩
      rcases ihb rfl with hvb | ⟨b', hb'⟩
      · rcases canon_know hvb hb with ⟨wb, mb, rfl⟩
        rcases invKraw hb rfl with ⟨Tb, hTb, hwbty, hwbval⟩
        injection hTb with hTb'; subst hTb'
        rcases canon_real hwbval hwbty with ⟨y, rfl⟩
        exact Or.inr ⟨_, .kmul_red⟩
      · exact Or.inr ⟨_, .kmul_r (.v_kraw ma (.v_real x)) hb'⟩
    · exact Or.inr ⟨_, .kmul_l ha'⟩
  | t_let Γ T₁ T₂ E₁ E₂ e body he hbody ihe _ =>
    subst hΓ
    rcases ihe rfl with hv | ⟨e', he'⟩
    · exact Or.inr ⟨subst 0 e body, .let_red hv⟩
    · exact Or.inr ⟨.letE e' body, .let_step he'⟩
  | t_kraw Γ T v m hv hval hk _ => exact Or.inl (.v_kraw m hval)
  | t_sub Γ e T E E' h0 hsub ih => exact ih hΓ

/-- **Progress** for the value-carrying calculus: a well-typed closed term is a
    value or steps. -/
theorem effect_progress {e T E} (ht : HasTy [] e T E) : IsValue e ∨ ∃ e', e ⇒ e' :=
  progress' ht rfl

end Sounio.EpistemicEffectsV2
