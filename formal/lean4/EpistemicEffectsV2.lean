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

end Sounio.EpistemicEffectsV2
