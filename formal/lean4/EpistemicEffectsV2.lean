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
  | lit_mg   : Int → Expr    -- milligram quantity; integer scale, like lit_real
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
  | v_mg    : ∀ z, IsValue (.lit_mg z)
  | v_lam   : ∀ T E e, IsValue (.lam T E e)
  | v_kraw  : ∀ {v} m, IsValue v → IsValue (.kraw v m)

-- §3. Typing  Γ ⊢ e : T ! E
inductive HasTy : TyCtx → Expr → Ty → EffectSet → Prop where
  | t_lit_nat  : ∀ Γ n, HasTy Γ (.lit_nat n) .tnat emptyE
  | t_lit_real : ∀ Γ z, HasTy Γ (.lit_real z) .treal emptyE
  | t_lit_mg   : ∀ Γ z, HasTy Γ (.lit_mg z) .tmg emptyE
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
  | .lit_mg z => .lit_mg z
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
  | .lit_mg z => .lit_mg z
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

theorem genMg {Γ e T E} (h : HasTy Γ e T E) {z} (he : e = .lit_mg z) : T = .tmg := by
  induction h with
  | t_lit_mg => rfl
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

/-- `kraw` inversion: recover the payload type, its typing, value-ness, and the
    metadata validity. -/
theorem invKraw {Γ e T E} (h : HasTy Γ e T E) {v m} (he : e = .kraw v m) :
    ∃ T', T = .tknow T' ∧ HasTy Γ v T' emptyE ∧ IsValue v ∧ kvalid m := by
  induction h with
  | t_kraw Γ' T' v' m' hv' hval' hk' =>
    injection he with h1 h2; subst h1; subst h2; exact ⟨T', rfl, hv', hval', hk'⟩
  | t_sub Γ' e' T0 E1 E2 h0 hsub ih => exact ih he
  | _ => exact Expr.noConfusion he

theorem canon_arrow {v S F T₂ E} (hv : IsValue v) (ht : HasTy [] v (.tarrow S F T₂) E) :
    ∃ S' F' b, v = .lam S' F' b := by
  cases hv with
  | v_nat n  => exact Ty.noConfusion (genNat ht rfl)
  | v_real z => exact Ty.noConfusion (genReal ht rfl)
  | v_mg z => exact Ty.noConfusion (genMg ht rfl)
  | v_lam T E0 e0 => exact ⟨T, E0, e0, rfl⟩
  | @v_kraw w m hp => rcases genKraw ht rfl with ⟨T', hT⟩; exact Ty.noConfusion hT

theorem canon_know {v T E} (hv : IsValue v) (ht : HasTy [] v (.tknow T) E) :
    ∃ w m, v = .kraw w m := by
  cases hv with
  | @v_kraw w m hp => exact ⟨w, m, rfl⟩
  | v_nat n  => exact Ty.noConfusion (genNat ht rfl)
  | v_real z => exact Ty.noConfusion (genReal ht rfl)
  | v_mg z => exact Ty.noConfusion (genMg ht rfl)
  | v_lam T0 E0 e0 => rcases genLam ht rfl with ⟨T₂, hT⟩; exact Ty.noConfusion hT

theorem canon_real {v E} (hv : IsValue v) (ht : HasTy [] v .treal E) :
    ∃ z, v = .lit_real z := by
  cases hv with
  | v_real z => exact ⟨z, rfl⟩
  | v_nat n  => exact Ty.noConfusion (genNat ht rfl)
  | v_mg z => exact Ty.noConfusion (genMg ht rfl)
  | v_lam T0 E0 e0 => rcases genLam ht rfl with ⟨T₂, hT⟩; exact Ty.noConfusion hT
  | @v_kraw w m hp => rcases genKraw ht rfl with ⟨T', hT⟩; exact Ty.noConfusion hT

theorem progress' {Γ e T E} (h : HasTy Γ e T E) (hΓ : Γ = []) :
    IsValue e ∨ ∃ e', e ⇒ e' := by
  induction h with
  | t_lit_nat Γ n => exact Or.inl (.v_nat n)
  | t_lit_real Γ z => exact Or.inl (.v_real z)
  | t_lit_mg Γ z => exact Or.inl (.v_mg z)
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
      rcases invKraw ha rfl with ⟨Ta, hTa, hwaty, hwaval, _⟩
      injection hTa with hTa'; subst hTa'
      rcases canon_real hwaval hwaty with ⟨x, rfl⟩
      rcases ihb rfl with hvb | ⟨b', hb'⟩
      · rcases canon_know hvb hb with ⟨wb, mb, rfl⟩
        rcases invKraw hb rfl with ⟨Tb, hTb, hwbty, hwbval, _⟩
        injection hTb with hTb'; subst hTb'
        rcases canon_real hwbval hwbty with ⟨y, rfl⟩
        exact Or.inr ⟨_, .kadd_red⟩
      · exact Or.inr ⟨_, .kadd_r (.v_kraw ma (.v_real x)) hb'⟩
    · exact Or.inr ⟨_, .kadd_l ha'⟩
  | t_kmul Γ E₁ E₂ a b ha hb iha ihb =>
    subst hΓ
    rcases iha rfl with hva | ⟨a', ha'⟩
    · rcases canon_know hva ha with ⟨wa, ma, rfl⟩
      rcases invKraw ha rfl with ⟨Ta, hTa, hwaty, hwaval, _⟩
      injection hTa with hTa'; subst hTa'
      rcases canon_real hwaval hwaty with ⟨x, rfl⟩
      rcases ihb rfl with hvb | ⟨b', hb'⟩
      · rcases canon_know hvb hb with ⟨wb, mb, rfl⟩
        rcases invKraw hb rfl with ⟨Tb, hTb, hwbty, hwbval, _⟩
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

-- ================================================================
-- §8. Preservation (subject reduction)
-- ================================================================

-- shifting / substituting preserves value-ness (kraw payload recurses)
theorem shift_value {v} (hv : IsValue v) : ∀ c d, IsValue (shift c d v) := by
  induction hv with
  | v_nat n => intro c d; exact .v_nat n
  | v_real z => intro c d; exact .v_real z
  | v_mg z => intro c d; exact .v_mg z
  | v_lam T E e => intro c d; exact .v_lam _ _ _
  | v_kraw m hp ih => intro c d; exact .v_kraw m (ih c d)

theorem subst_value {w} (hw : IsValue w) : ∀ n u, IsValue (subst n u w) := by
  induction hw with
  | v_nat n => intro n' u; exact .v_nat n
  | v_real z => intro n' u; exact .v_real z
  | v_mg z => intro n' u; exact .v_mg z
  | v_lam T E e => intro n' u; exact .v_lam _ _ _
  | v_kraw m hp ih => intro n' u; exact .v_kraw m (ih n' u)

-- GUM metadata combinators preserve validity
theorem gAddMeta_valid {ma mb} (ha : kvalid ma) (hb : kvalid mb) : kvalid (gAddMeta ma mb) := by
  obtain ⟨ha1, ha2, ha3⟩ := ha
  obtain ⟨hb1, hb2, hb3⟩ := hb
  refine ⟨?_, ?_, ?_⟩
  · show 0 ≤ ma.gumVar + mb.gumVar
    omega
  · show 0 ≤ (if ma.conf ≤ mb.conf then ma.conf else mb.conf)
    split <;> omega
  · show (if ma.conf ≤ mb.conf then ma.conf else mb.conf) ≤ 1000
    split <;> omega

theorem int_sq_nonneg (x : Int) : 0 ≤ x * x := by
  rcases Int.le_total 0 x with h | h
  · exact Int.mul_nonneg h h
  · have h2 : 0 ≤ -x := by omega
    have h3 := Int.mul_nonneg h2 h2
    rwa [Int.neg_mul_neg] at h3

theorem gMulMeta_valid {x y ma mb} (ha : kvalid ma) (hb : kvalid mb) :
    kvalid (gMulMeta x ma y mb) := by
  have hx : 0 ≤ x * x := int_sq_nonneg x
  have hy : 0 ≤ y * y := int_sq_nonneg y
  obtain ⟨ha1, ha2, ha3⟩ := ha
  obtain ⟨hb1, hb2, hb3⟩ := hb
  refine ⟨?_, ?_, ?_⟩
  · show 0 ≤ y * y * ma.gumVar + x * x * mb.gumVar
    have := Int.mul_nonneg hy ha1; have := Int.mul_nonneg hx hb1; omega
  · show 0 ≤ (if ma.conf ≤ mb.conf then ma.conf else mb.conf)
    split <;> omega
  · show (if ma.conf ≤ mb.conf then ma.conf else mb.conf) ≤ 1000
    split <;> omega

theorem lookup_some_lt {Γ : TyCtx} {n : Nat} {T : Ty}
    (h : lookupCtx Γ n = some T) : n < Γ.length := by
  induction Γ generalizing n with
  | nil => simp [lookupCtx] at h
  | cons hd tl ih =>
    cases n with
    | zero => simp [List.length]
    | succ n' => simp [lookupCtx] at h; have := ih h; simp [List.length]; omega

theorem wellScoped {Γ e T E} (h : HasTy Γ e T E) :
    ∀ c d, Γ.length ≤ c → shift c d e = e := by
  induction h with
  | t_lit_nat => intro c d _; rfl
  | t_lit_real => intro c d _; rfl
  | t_lit_mg => intro c d _; rfl
  | t_var Γ n T hlk =>
    intro c d hc
    have hn : n < Γ.length := lookup_some_lt hlk
    have hnc : n < c := by omega
    simp [shift, hnc]
  | t_lam Γ T₁ T₂ E body _ ih =>
    intro c d hc
    have hb : (T₁ :: Γ).length ≤ c + 1 := by simp [List.length] at hc ⊢; omega
    simp [shift, ih (c+1) d hb]
  | t_app Γ T₁ T₂ Ef Ec Ecaller f a _ _ _ _ ihf iha =>
    intro c d hc; simp [shift, ihf c d hc, iha c d hc]
  | t_measure Γ T e m _ _ ih => intro c d hc; simp [shift, ih c d hc]
  | t_kvalue Γ T E e _ ih => intro c d hc; simp [shift, ih c d hc]
  | t_kunc Γ T E e _ ih => intro c d hc; simp [shift, ih c d hc]
  | t_kconf Γ T E e _ ih => intro c d hc; simp [shift, ih c d hc]
  | t_kadd Γ E₁ E₂ a b _ _ iha ihb => intro c d hc; simp [shift, iha c d hc, ihb c d hc]
  | t_kmul Γ E₁ E₂ a b _ _ iha ihb => intro c d hc; simp [shift, iha c d hc, ihb c d hc]
  | t_let Γ T₁ T₂ E₁ E₂ e body _ _ ihe ihb =>
    intro c d hc
    have hb : (T₁ :: Γ).length ≤ c + 1 := by simp [List.length] at hc ⊢; omega
    simp [shift, ihe c d hc, ihb (c+1) d hb]
  | t_kraw Γ T v m _ _ _ ih => intro c d hc; simp [shift, ih c d hc]
  | t_sub Γ e T E E' _ _ ih => intro c d hc; exact ih c d hc

theorem lookup_insert_lt {Γ : TyCtx} {k n : Nat} {τ T : Ty}
    (hn : n < k) (h : lookupCtx Γ n = some T) :
    lookupCtx (Γ.insertIdx k τ) n = some T := by
  induction Γ generalizing k n with
  | nil => cases n <;> simp [lookupCtx] at h
  | cons hd tl ih =>
    cases k with
    | zero => omega
    | succ k' =>
      cases n with
      | zero => simp [List.insertIdx, lookupCtx] at h ⊢; exact h
      | succ n' => simp [List.insertIdx, lookupCtx] at h ⊢; exact ih (by omega) h

theorem lookup_insert_ge {Γ : TyCtx} {k n : Nat} {τ T : Ty}
    (hn : k ≤ n) (h : lookupCtx Γ n = some T) :
    lookupCtx (Γ.insertIdx k τ) (n + 1) = some T := by
  induction Γ generalizing k n with
  | nil => cases n <;> simp [lookupCtx] at h
  | cons hd tl ih =>
    cases k with
    | zero => simp [List.insertIdx, lookupCtx] at h ⊢; exact h
    | succ k' =>
      cases n with
      | zero => omega
      | succ n' => simp [List.insertIdx, lookupCtx] at h ⊢; exact ih (by omega) h

theorem weakening {Γ e T E} (h : HasTy Γ e T E) :
    ∀ k τ, HasTy (Γ.insertIdx k τ) (shift k 1 e) T E := by
  induction h with
  | t_lit_nat Γ n => intro k τ; exact .t_lit_nat _ _
  | t_lit_real Γ z => intro k τ; exact .t_lit_real _ _
  | t_lit_mg Γ z => intro k τ; exact .t_lit_mg _ _
  | t_var Γ n T hlk =>
    intro k τ
    by_cases hnk : n < k
    · have : shift k 1 (.var n) = .var n := by simp [shift, hnk]
      rw [this]; exact .t_var _ _ _ (lookup_insert_lt hnk hlk)
    · have : shift k 1 (.var n) = .var (n+1) := by simp [shift, hnk]
      rw [this]; exact .t_var _ _ _ (lookup_insert_ge (by omega) hlk)
  | t_lam Γ T₁ T₂ E body _ ih =>
    intro k τ
    have : shift k 1 (.lam T₁ E body) = .lam T₁ E (shift (k+1) 1 body) := by simp [shift]
    rw [this]
    have hctx : (T₁ :: Γ).insertIdx (k+1) τ = T₁ :: Γ.insertIdx k τ := by simp [List.insertIdx]
    have hb := ih (k+1) τ; rw [hctx] at hb
    exact .t_lam _ _ _ _ _ hb
  | t_app Γ T₁ T₂ Ef Ec Ecaller f a _ _ hEf hEc ihf iha =>
    intro k τ; exact .t_app _ _ _ _ _ _ _ _ (ihf k τ) (iha k τ) hEf hEc
  | t_measure Γ T e m _ hk ih =>
    intro k τ; exact .t_measure _ _ _ _ (ih k τ) hk
  | t_kvalue Γ T E e _ ih => intro k τ; exact .t_kvalue _ _ _ _ (ih k τ)
  | t_kunc Γ T E e _ ih => intro k τ; exact .t_kunc _ _ _ _ (ih k τ)
  | t_kconf Γ T E e _ ih => intro k τ; exact .t_kconf _ _ _ _ (ih k τ)
  | t_kadd Γ E₁ E₂ a b _ _ iha ihb => intro k τ; exact .t_kadd _ _ _ _ _ (iha k τ) (ihb k τ)
  | t_kmul Γ E₁ E₂ a b _ _ iha ihb => intro k τ; exact .t_kmul _ _ _ _ _ (iha k τ) (ihb k τ)
  | t_let Γ T₁ T₂ E₁ E₂ e body _ _ ihe ihb =>
    intro k τ
    have : shift k 1 (.letE e body) = .letE (shift k 1 e) (shift (k+1) 1 body) := by simp [shift]
    rw [this]
    have hctx : (T₁ :: Γ).insertIdx (k+1) τ = T₁ :: Γ.insertIdx k τ := by simp [List.insertIdx]
    have hb := ihb (k+1) τ; rw [hctx] at hb
    exact .t_let _ _ _ _ _ _ _ (ihe k τ) hb
  | t_kraw Γ T v m _ hval hk ih =>
    intro k τ
    have : shift k 1 (.kraw v m) = .kraw (shift k 1 v) m := by simp [shift]
    rw [this]; exact .t_kraw _ _ _ _ (ih k τ) (shift_value hval k 1) hk
  | t_sub Γ e T E E' _ hsub ih => intro k τ; exact .t_sub _ _ _ _ _ (ih k τ) hsub

theorem closedWeaken {v σ E} (h : HasTy [] v σ E) : ∀ Δ, HasTy Δ v σ E := by
  intro Δ
  induction Δ with
  | nil => exact h
  | cons τ Δ' ih =>
    have hw := weakening ih 0 τ
    have hsh : shift 0 1 v = v := wellScoped h 0 1 (by simp [List.length])
    rw [hsh] at hw
    simpa [List.insertIdx] using hw

theorem lookup_append_lt {Δ Γ' : TyCtx} {m : Nat} (hm : m < Δ.length) :
    lookupCtx (Δ ++ Γ') m = lookupCtx Δ m := by
  induction Δ generalizing m with
  | nil => simp [List.length] at hm
  | cons hd tl ih =>
    cases m with
    | zero => simp [lookupCtx]
    | succ m' => simp [lookupCtx]; exact ih (by simp [List.length] at hm; omega)

theorem lookup_append_eq {Δ Γ' : TyCtx} {σ : Ty} :
    lookupCtx (Δ ++ σ :: Γ') Δ.length = some σ := by
  induction Δ with
  | nil => simp [lookupCtx]
  | cons hd tl ih => simp [lookupCtx, List.length]; exact ih

theorem invLam {Γ e T E} (h : HasTy Γ e T E) {S F b} (he : e = .lam S F b) :
    ∃ T₂, T = .tarrow S F T₂ ∧ HasTy (S :: Γ) b T₂ F := by
  induction h with
  | t_lam Γ' T₁ T₂ E' body hb ihb =>
    injection he with h1 h2 h3; subst h1; subst h2; subst h3; exact ⟨T₂, rfl, hb⟩
  | t_sub Γ' e' T' E1 E2 h0 hsub ih => exact ih he
  | _ => exact Expr.noConfusion he

theorem value_emptyE {Γ v T E} (hv : IsValue v) (h : HasTy Γ v T E) :
    HasTy Γ v T emptyE := by
  cases hv with
  | v_nat n => rw [genNat h rfl]; exact .t_lit_nat _ _
  | v_real z => rw [genReal h rfl]; exact .t_lit_real _ _
  | v_mg z => rw [genMg h rfl]; exact .t_lit_mg _ _
  | v_lam S F b => rcases invLam h rfl with ⟨T₂, hT, hb⟩; subst hT; exact .t_lam _ _ _ _ _ hb
  | @v_kraw w m hp =>
    rcases invKraw h rfl with ⟨T', hT, hwty, hwval, hk⟩; subst hT
    exact .t_kraw _ _ _ _ hwty hwval hk

theorem substClosed {v σ} (hv : HasTy [] v σ emptyE) {Γ0 e T E} (h : HasTy Γ0 e T E) :
    ∀ Δ, Γ0 = Δ ++ σ :: [] → HasTy Δ (subst Δ.length v e) T E := by
  induction h with
  | t_lit_nat Γ' n => intro Δ hΓ; exact .t_lit_nat _ _
  | t_lit_real Γ' z => intro Δ hΓ; exact .t_lit_real _ _
  | t_lit_mg Γ' z => intro Δ hΓ; exact .t_lit_mg _ _
  | t_var Γ' n T' hlk =>
    intro Δ hΓ; subst hΓ
    by_cases h1 : n = Δ.length
    · subst h1
      have hσ : T' = σ := by
        have he := lookup_append_eq (Δ := Δ) (Γ' := ([] : TyCtx)) (σ := σ)
        rw [he] at hlk; injection hlk with hh; exact hh.symm
      subst hσ
      have : subst Δ.length v (.var Δ.length) = v := by simp [subst]
      rw [this]; exact closedWeaken hv Δ
    · by_cases h2 : n > Δ.length
      · exfalso
        have hlen : (Δ ++ σ :: ([] : TyCtx)).length = Δ.length + 1 := by simp [List.length]
        have := lookup_some_lt hlk; rw [hlen] at this; omega
      · have hn : n < Δ.length := by omega
        have hs : subst Δ.length v (.var n) = .var n := by simp [subst, h1, h2]
        rw [hs]
        exact .t_var _ _ _ (by rw [← lookup_append_lt hn]; exact hlk)
  | t_lam Γ' T₁ T₂ E' body hb ihb =>
    intro Δ hΓ; subst hΓ
    have hsh : shift 0 1 v = v := wellScoped hv 0 1 (by simp [List.length])
    have key : subst Δ.length v (.lam T₁ E' body) = .lam T₁ E' (subst (Δ.length + 1) v body) := by
      simp [subst, hsh]
    rw [key]; exact .t_lam _ _ _ _ _ (ihb (T₁ :: Δ) rfl)
  | t_app Γ' T₁ T₂ Ef Ec Ecaller f a hf ha hEf hEc ihf iha =>
    intro Δ hΓ; subst hΓ
    exact .t_app _ _ _ _ _ _ _ _ (ihf Δ rfl) (iha Δ rfl) hEf hEc
  | t_measure Γ' T e m he hk ih =>
    intro Δ hΓ; subst hΓ; exact .t_measure _ _ _ _ (ih Δ rfl) hk
  | t_kvalue Γ' T E e _ ih => intro Δ hΓ; subst hΓ; exact .t_kvalue _ _ _ _ (ih Δ rfl)
  | t_kunc Γ' T E e _ ih => intro Δ hΓ; subst hΓ; exact .t_kunc _ _ _ _ (ih Δ rfl)
  | t_kconf Γ' T E e _ ih => intro Δ hΓ; subst hΓ; exact .t_kconf _ _ _ _ (ih Δ rfl)
  | t_kadd Γ' E₁ E₂ a b _ _ iha ihb =>
    intro Δ hΓ; subst hΓ; exact .t_kadd _ _ _ _ _ (iha Δ rfl) (ihb Δ rfl)
  | t_kmul Γ' E₁ E₂ a b _ _ iha ihb =>
    intro Δ hΓ; subst hΓ; exact .t_kmul _ _ _ _ _ (iha Δ rfl) (ihb Δ rfl)
  | t_let Γ' T₁ T₂ E₁ E₂ e body _ _ ihe ihb =>
    intro Δ hΓ; subst hΓ
    have hsh : shift 0 1 v = v := wellScoped hv 0 1 (by simp [List.length])
    have key : subst Δ.length v (.letE e body) = .letE (subst Δ.length v e) (subst (Δ.length + 1) v body) := by
      simp [subst, hsh]
    rw [key]; exact .t_let _ _ _ _ _ _ _ (ihe Δ rfl) (ihb (T₁ :: Δ) rfl)
  | t_kraw Γ' T w m hw hval hk ih =>
    intro Δ hΓ; subst hΓ
    have key : subst Δ.length v (.kraw w m) = .kraw (subst Δ.length v w) m := by simp [subst]
    rw [key]; exact .t_kraw _ _ _ _ (ih Δ rfl) (subst_value hval Δ.length v) hk
  | t_sub Γ' e T E E' h0 hsub ih => intro Δ hΓ; exact .t_sub _ _ _ _ _ (ih Δ hΓ) hsub

/-- **Preservation (subject reduction)** for the value-carrying calculus:
    a closed well-typed term keeps its type (and effect) under reduction. -/
theorem preservation' {Γ e T E} (h : HasTy Γ e T E) :
    ∀ {e'}, e ⇒ e' → Γ = [] → HasTy Γ e' T E := by
  induction h with
  | t_lit_nat Γ n => intro e' hs hΓ; cases hs
  | t_lit_real Γ z => intro e' hs hΓ; cases hs
  | t_lit_mg Γ z => intro e' hs hΓ; cases hs
  | t_var Γ n T hlk => intro e' hs hΓ; cases hs
  | t_lam Γ T₁ T₂ E body _ => intro e' hs hΓ; cases hs
  | t_app Γ T₁ T₂ Ef Ec Ecaller f a hf ha hEf hEc ihf iha =>
    intro e' hs hΓ; subst hΓ
    cases hs with
    | beta hvarg =>
      rcases invLam hf rfl with ⟨T2', hTeq, hbody⟩
      injection hTeq with e1 e2 e3; subst e1; subst e2; subst e3
      have hv0 := value_emptyE hvarg ha
      exact .t_sub _ _ _ _ _ (substClosed hv0 hbody [] rfl) hEf
    | app_l hf' => exact .t_app _ _ _ _ _ _ _ _ (ihf hf' rfl) ha hEf hEc
    | app_r _ ha' => exact .t_app _ _ _ _ _ _ _ _ hf (iha ha' rfl) hEf hEc
  | t_measure Γ T e m he hk ihe =>
    intro e' hs hΓ; subst hΓ
    cases hs with
    | meas_red hv =>
      exact .t_sub _ _ _ _ _ (.t_kraw _ _ _ _ he hv hk) (Sounio.EpistemicEffects.emptyE_sub _)
    | meas_arg he' => exact .t_measure _ _ _ _ (ihe he' rfl) hk
  | t_kvalue Γ T E e he ihe =>
    intro e' hs hΓ; subst hΓ
    cases hs with
    | kvalue_red =>
      rcases invKraw he rfl with ⟨T', hT, hwty, hwval, hk⟩
      injection hT with hT'; subst hT'; exact .t_sub _ _ _ _ _ hwty (Sounio.EpistemicEffects.emptyE_sub _)
    | kvalue_arg he' => exact .t_kvalue _ _ _ _ (ihe he' rfl)
  | t_kunc Γ T E e he ihe =>
    intro e' hs hΓ; subst hΓ
    cases hs with
    | kunc_red => exact .t_sub _ _ _ _ _ (.t_lit_real _ _) (Sounio.EpistemicEffects.emptyE_sub _)
    | kunc_arg he' => exact .t_kunc _ _ _ _ (ihe he' rfl)
  | t_kconf Γ T E e he ihe =>
    intro e' hs hΓ; subst hΓ
    cases hs with
    | kconf_red => exact .t_sub _ _ _ _ _ (.t_lit_real _ _) (Sounio.EpistemicEffects.emptyE_sub _)
    | kconf_arg he' => exact .t_kconf _ _ _ _ (ihe he' rfl)
  | t_kadd Γ E₁ E₂ a b ha hb iha ihb =>
    intro e' hs hΓ; subst hΓ
    cases hs with
    | kadd_red =>
      rcases invKraw ha rfl with ⟨Ta, hTa, _, _, hka⟩
      rcases invKraw hb rfl with ⟨Tb, hTb, _, _, hkb⟩
      exact .t_sub _ _ _ _ _ (.t_kraw _ _ _ _ (.t_lit_real _ _) (.v_real _) (gAddMeta_valid hka hkb)) (Sounio.EpistemicEffects.emptyE_sub _)
    | kadd_l ha' => exact .t_kadd _ _ _ _ _ (iha ha' rfl) hb
    | kadd_r _ hb' => exact .t_kadd _ _ _ _ _ ha (ihb hb' rfl)
  | t_kmul Γ E₁ E₂ a b ha hb iha ihb =>
    intro e' hs hΓ; subst hΓ
    cases hs with
    | kmul_red =>
      rcases invKraw ha rfl with ⟨Ta, hTa, _, _, hka⟩
      rcases invKraw hb rfl with ⟨Tb, hTb, _, _, hkb⟩
      exact .t_sub _ _ _ _ _ (.t_kraw _ _ _ _ (.t_lit_real _ _) (.v_real _) (gMulMeta_valid hka hkb)) (Sounio.EpistemicEffects.emptyE_sub _)
    | kmul_l ha' => exact .t_kmul _ _ _ _ _ (iha ha' rfl) hb
    | kmul_r _ hb' => exact .t_kmul _ _ _ _ _ ha (ihb hb' rfl)
  | t_let Γ T₁ T₂ E₁ E₂ e body he hbody ihe ihb =>
    intro e' hs hΓ; subst hΓ
    cases hs with
    | let_red hv => exact .t_sub _ _ _ _ _ (substClosed (value_emptyE hv he) hbody [] rfl) (Sounio.EpistemicEffects.sub_union_right E₁ E₂)
    | let_step he' => exact .t_let _ _ _ _ _ _ _ (ihe he' rfl) hbody
  | t_kraw Γ T v m hv hval hk ih => intro e' hs hΓ; cases hs
  | t_sub Γ e T E E' h0 hsub ih => intro e' hs hΓ; exact .t_sub _ _ _ _ _ (ih hs hΓ) hsub

/-- Closed-term subject reduction. -/
theorem preservation {e T E} (h : HasTy [] e T E) {e'} (hs : e ⇒ e') : HasTy [] e' T E :=
  preservation' h hs rfl

end Sounio.EpistemicEffectsV2
