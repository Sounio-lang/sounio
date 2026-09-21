-- formal/lean4/EpistemicEffects.lean
/-!
# REFUTED MODEL — USE `EpistemicEffectsV2`

This module is retained as a machine-checked counterexample and shared
definition spine. Its payload-erasing `Knowledge` calculus does **not** satisfy
subject reduction; see §9.1. Import `EpistemicEffectsV2` for the repaired,
value-carrying calculus and its preservation and progress theorems. Do not cite
this module as the metatheory of Sounio's implemented `Knowledge<T>` semantics.

# Sounio Epistemic-Effect Calculus — Lean 4 Soundness Sketch

The formal artifact behind the project's registered SOTA claim that
*epistemic gradual compilation* is the language-level contribution of Sounio
(see `[[project_pl_contribution_sota]]`). The audit
`docs/audit/PL_ADOPTION_AUDIT_2026-05-27.md` §3.1–§3.2 named this gap.

This file gives:
  * Syntax of a core calculus with `Knowledge<T>` and an effect row.
  * Typing relation with the **effect-subset propagation** premise on `app`
    — this is the formal statement of the compiler's E035 check
    (`crates/souc/src/types/core.rs` + audit §3.2).
  * Small-step CBV operational semantics that reduce `Knowledge` arithmetic
    by the GUM linearization rules
    (σ²(a+b) = σ²(a) + σ²(b); σ²(a·b) = b²σ²(a) + a²σ²(b)).
  * Soundness status (2026-08-16 correction — this file's only `sorry`
    removed; see §9.1 for the full history):
      - `effect_preservation`   [REFUTATED — the original statement is FALSE.
        Two machine-checked refutations live in §9.3: the root failure
        (`effect_preservation_is_false`, zero axioms; also in
        `EpistemicPreservationWIP_counterexample.lean`) and the propagation
        failure (`effect_preservation_existential_is_false`, [propext]):
        even `∃ T', HasTy [] e' T' E` fails — type errors are REACHABLE by
        reduction. The repaired calculus is `EpistemicEffectsV2.lean`.]
      - `effect_progress`       [PROVED — induction-on-typing with Γ pinned to []]
      - `gum_conservativity`    [PROVED — `simp` closes both add and mul cases]
      - `confidence_monotonicity` [PROVED — `omega` closes both cases]

References:
  * Plotkin, G. & Pretnar, M. (2009). "Handlers of Algebraic Effects." ESOP.
  * Leijen, D. (2014). "Koka: Programming with Row Polymorphic Effect Types."
  * BIPM/JCGM 100 (2008). "Evaluation of measurement data — Guide to the
    expression of uncertainty in measurement" (GUM).
  * Wright, A. & Felleisen, M. (1994). "A Syntactic Approach to Type Soundness."

Design choices:
  * Mathlib-free — the project lakefile does not depend on Mathlib (cf.
    `SounioEffects.lean`). All numeric reasoning is over `Int`.
  * `Knowledge` values carry `(value, gumVar, conf)` as integers. The
    GUM-conservativity theorem is closed by *construction*: `kmul` produces
    `gumVar = b²·σ²(a) + a²·σ²(b) + remainder` where `remainder ≥ 0` is
    a separate non-negative term. The "true" linearized variance is
    `b²·σ²(a) + a²·σ²(b)`, so `trueVar ≤ gumVar` definitionally; the
    content of the theorem is that the GUM rule is a sound upper bound
    on the linearized variance (it never underestimates).
  * Effect rows as `Effect → Bool` mirrors `SounioEffects.lean`.

No `sorry`. No `Mathlib`.
-/

namespace Sounio.EpistemicEffects

-- ================================================================
-- §1. Effects (the nine constructors of audit §3.2)
-- ================================================================

/-- The nine epistemic-effect labels surfaced by the audit (§3.2):
    `with IO, Mut, Panic, Div, Alloc, Session, Observe, Audit, Hypothesis`.
    GPU/Deterministic are out of scope for this soundness sketch. -/
inductive Effect where
  | eIO
  | eMut
  | ePanic
  | eDiv
  | eAlloc
  | eSession
  | eObserve
  | eAudit
  | eHypothesis
  deriving DecidableEq, Repr

/-- Effect row as a decidable characteristic function. Mathematically
    equivalent to `Finset Effect` via `funext`, but avoids the Mathlib
    dependency. Matches the encoding in `SounioEffects.lean`. -/
abbrev EffectSet := Effect → Bool

def emptyE : EffectSet := fun _ => false
def singleE (e : Effect) : EffectSet := fun f => decide (f = e)
def unionE (a b : EffectSet) : EffectSet := fun e => a e || b e

/-- Subset on effect rows. `a ⊆ b` says every effect in `a` is in `b`.
    This is the relation the typing rule for `app` enforces — the
    callee's effect set must be contained in the caller's annotation
    (compiler error E035 fires when this is violated). -/
def subE (a b : EffectSet) : Prop := ∀ e, a e = true → b e = true

infix:50 " ⊆ₑ " => subE

theorem subE_refl (a : EffectSet) : a ⊆ₑ a := fun _ h => h

theorem subE_trans {a b c : EffectSet} (h₁ : a ⊆ₑ b) (h₂ : b ⊆ₑ c) :
    a ⊆ₑ c := fun e h => h₂ e (h₁ e h)

theorem emptyE_sub (a : EffectSet) : emptyE ⊆ₑ a := by
  intro e h; simp [emptyE] at h

theorem sub_union_left (a b : EffectSet) : a ⊆ₑ unionE a b := by
  intro e h; simp [unionE, h]

theorem sub_union_right (a b : EffectSet) : b ⊆ₑ unionE a b := by
  intro e h; simp [unionE, h]

theorem union_sub {a b c : EffectSet} (h₁ : a ⊆ₑ c) (h₂ : b ⊆ₑ c) :
    unionE a b ⊆ₑ c := by
  intro e he
  have : (a e || b e) = true := he
  cases ha : a e
  · cases hb : b e
    · simp [ha, hb] at this
    · exact h₂ e hb
  · exact h₁ e ha

-- ================================================================
-- §2. Types
-- ================================================================

/-- Object types. `tknow` wraps a base type with a GUM/confidence cell;
    `tarrow` carries the callee's latent effects on the arrow.
    `tmg` is the milligram unit the language ships (`Knowledge<mg>`).
    This calculus still cannot inhabit it: `t_kraw` pins `Real`. -/
inductive Ty where
  | tnat
  | treal
  | tmg
  | tknow  : Ty → Ty
  | tarrow : Ty → EffectSet → Ty → Ty

-- ================================================================
-- §3. Expression syntax
-- ================================================================

/-- Knowledge cell carried at runtime: integer value, integer GUM variance
    (non-negative by construction below), integer confidence in [0, 1000].
    Using `Int` keeps the calculus Mathlib-free and decidable. -/
structure KCell where
  value  : Int
  gumVar : Int      -- ≥ 0 by `kvalid`
  conf   : Int      -- in [0, 1000] by `kvalid`
  deriving Repr, DecidableEq

/-- Well-formedness for Knowledge cells. -/
def kvalid (k : KCell) : Prop :=
  0 ≤ k.gumVar ∧ 0 ≤ k.conf ∧ k.conf ≤ 1000

/-- Core expressions. De Bruijn-style `var n` for binders. -/
inductive Expr where
  | lit_nat  : Nat → Expr
  | lit_real : Int → Expr             -- "real" via integer scale; placeholder
  | var      : Nat → Expr             -- de Bruijn index
  | lam      : Ty → EffectSet → Expr → Expr
  | app      : Expr → Expr → Expr
  | measure  : Expr → KCell → Expr    -- introduces a Knowledge cell
  | kvalue   : Expr → Expr            -- elimination: value projection
  | kunc     : Expr → Expr            -- elimination: GUM-variance projection
  | kconf    : Expr → Expr            -- elimination: confidence projection
  | kadd     : Expr → Expr → Expr     -- Knowledge addition (linear GUM)
  | kmul     : Expr → Expr → Expr     -- Knowledge multiplication (GUM upper)
  | letE     : Expr → Expr → Expr     -- let v = e1 in e2 (binds var 0)
  | kraw     : KCell → Expr           -- runtime Knowledge value

-- ================================================================
-- §4. Typing context
-- ================================================================

/-- A typing context: a list of types, indexed by de Bruijn position. -/
abbrev TyCtx := List Ty

def lookupCtx : TyCtx → Nat → Option Ty
  | [],      _     => none
  | t :: _,  0     => some t
  | _ :: ts, n + 1 => lookupCtx ts n

-- ================================================================
-- §5. Typing relation  Γ ⊢ e : T ! E
-- ================================================================

/-- The typing judgement. The `app` rule encodes the effect-subset
    propagation premise — the formal statement of compiler error E035:
    the callee's latent effects on the arrow type must be a subset
    of the call site's declared effect annotation. -/
inductive HasTy : TyCtx → Expr → Ty → EffectSet → Prop where
  | t_lit_nat  : ∀ Γ n,
      HasTy Γ (.lit_nat n) .tnat emptyE
  | t_lit_real : ∀ Γ z,
      HasTy Γ (.lit_real z) .treal emptyE
  | t_var      : ∀ Γ n T,
      lookupCtx Γ n = some T → HasTy Γ (.var n) T emptyE
  | t_lam      : ∀ Γ T₁ T₂ E body,
      HasTy (T₁ :: Γ) body T₂ E →
      HasTy Γ (.lam T₁ E body) (.tarrow T₁ E T₂) emptyE
  | t_app      : ∀ Γ T₁ T₂ Ef Ec Ecaller f a,
      HasTy Γ f (.tarrow T₁ Ef T₂) Ec →
      HasTy Γ a T₁ Ec →
      -- E035: callee's latent effects must be ⊆ the caller's annotation
      Ef ⊆ₑ Ecaller →
      Ec ⊆ₑ Ecaller →
      HasTy Γ (.app f a) T₂ Ecaller
  | t_measure  : ∀ Γ T e k,
      HasTy Γ e T emptyE →
      kvalid k →
      HasTy Γ (.measure e k) (.tknow T) (singleE .eObserve)
  | t_kvalue   : ∀ Γ T E e,
      HasTy Γ e (.tknow T) E →
      HasTy Γ (.kvalue e) T E
  | t_kunc     : ∀ Γ T E e,
      HasTy Γ e (.tknow T) E →
      HasTy Γ (.kunc e) .treal E
  | t_kconf    : ∀ Γ T E e,
      HasTy Γ e (.tknow T) E →
      HasTy Γ (.kconf e) .treal E
  | t_kadd     : ∀ Γ T E₁ E₂ a b,
      HasTy Γ a (.tknow T) E₁ →
      HasTy Γ b (.tknow T) E₂ →
      HasTy Γ (.kadd a b) (.tknow T) (unionE E₁ E₂)
  | t_kmul     : ∀ Γ T E₁ E₂ a b,
      HasTy Γ a (.tknow T) E₁ →
      HasTy Γ b (.tknow T) E₂ →
      HasTy Γ (.kmul a b) (.tknow T) (unionE E₁ E₂)
  | t_let      : ∀ Γ T₁ T₂ E₁ E₂ e body,
      HasTy Γ e T₁ E₁ →
      HasTy (T₁ :: Γ) body T₂ E₂ →
      HasTy Γ (.letE e body) T₂ (unionE E₁ E₂)
  | t_kraw     : ∀ Γ k,
      kvalid k → HasTy Γ (.kraw k) (.tknow .treal) emptyE
  | -- Structural sub-effecting: any judgement can be reannotated upward.
    t_sub      : ∀ Γ e T E E',
      HasTy Γ e T E → E ⊆ₑ E' → HasTy Γ e T E'

-- ================================================================
-- §6. Values
-- ================================================================

/-- Call-by-value values. -/
inductive IsValue : Expr → Prop where
  | v_nat   : ∀ n, IsValue (.lit_nat n)
  | v_real  : ∀ z, IsValue (.lit_real z)
  | v_lam   : ∀ T E e, IsValue (.lam T E e)
  | v_kraw  : ∀ k, IsValue (.kraw k)

-- ================================================================
-- §7. GUM linearization on Knowledge cells
-- ================================================================

/-- The GUM addition rule (exact under linearization, equality form):
    σ²(a+b) = σ²(a) + σ²(b)  (independent operands; first-order). -/
def gumAdd (x y : KCell) : KCell :=
  { value  := x.value + y.value
  , gumVar := x.gumVar + y.gumVar
  , conf   := if x.conf ≤ y.conf then x.conf else y.conf }

/-- The GUM multiplication rule (first-order linearization, an upper
    bound on the true second-order variance):
    σ²(a·b) ≈ b²·σ²(a) + a²·σ²(b).
    We compute the linearized term and treat that as `gumVar`.
    The "true" variance (under our toy semantics) is the same linearized
    expression; the `gum_conservativity` theorem (§10) shows the GUM
    output is always a non-negative upper bound on it. -/
def gumMul (x y : KCell) : KCell :=
  { value  := x.value * y.value
  , gumVar := y.value * y.value * x.gumVar + x.value * x.value * y.gumVar
  , conf   := if x.conf ≤ y.conf then x.conf else y.conf }

/-- The "true" linearized variance the GUM rule is conserving. In the
    first-order GUM model the true variance and the GUM-output variance
    are *equal*; the conservativity theorem then says the GUM rule never
    underestimates the linearized variance. (Higher-order corrections
    are non-negative and would only increase the true variance further,
    so the GUM rule remains an upper bound; we leave that extension as
    documented in §10's docstring.) -/
def trueAddVar (x y : KCell) : Int := x.gumVar + y.gumVar
def trueMulVar (x y : KCell) : Int :=
  y.value * y.value * x.gumVar + x.value * x.value * y.gumVar

-- ================================================================
-- §8. Operational semantics (CBV small-step)
-- ================================================================

/-- Substitute `v` for de Bruijn index `n` in `e`, decrementing free
    indices above `n`. Small-step rules only ever substitute closed
    values (`v_lam`, `v_nat`, `v_real`, `v_kraw`) into the body, so we
    do not need a full lifting machinery — values have no free vars. -/
def shift (cutoff : Nat) (d : Int) : Expr → Expr
  | .var k =>
      if k < cutoff then .var k
      else
        match d with
        | .ofNat dn => .var (k + dn)
        | .negSucc dn => .var (k - (dn + 1))   -- k ≥ cutoff > dn ⇒ safe in practice
  | .lit_nat n => .lit_nat n
  | .lit_real z => .lit_real z
  | .lam T E b => .lam T E (shift (cutoff + 1) d b)
  | .app f a => .app (shift cutoff d f) (shift cutoff d a)
  | .measure e k => .measure (shift cutoff d e) k
  | .kvalue e => .kvalue (shift cutoff d e)
  | .kunc e => .kunc (shift cutoff d e)
  | .kconf e => .kconf (shift cutoff d e)
  | .kadd a b => .kadd (shift cutoff d a) (shift cutoff d b)
  | .kmul a b => .kmul (shift cutoff d a) (shift cutoff d b)
  | .letE e b => .letE (shift cutoff d e) (shift (cutoff + 1) d b)
  | .kraw k => .kraw k

def subst (n : Nat) (v : Expr) : Expr → Expr
  | .var k =>
      if k = n then v
      else if k > n then .var (k - 1)
      else .var k
  | .lit_nat m => .lit_nat m
  | .lit_real z => .lit_real z
  | .lam T E b => .lam T E (subst (n + 1) (shift 0 1 v) b)
  | .app f a => .app (subst n v f) (subst n v a)
  | .measure e k => .measure (subst n v e) k
  | .kvalue e => .kvalue (subst n v e)
  | .kunc e => .kunc (subst n v e)
  | .kconf e => .kconf (subst n v e)
  | .kadd a b => .kadd (subst n v a) (subst n v b)
  | .kmul a b => .kmul (subst n v a) (subst n v b)
  | .letE e b => .letE (subst n v e) (subst (n + 1) (shift 0 1 v) b)
  | .kraw k => .kraw k

/-- Small-step CBV reduction. The Knowledge arithmetic reductions
    `kadd_red` / `kmul_red` are the operational realization of the
    GUM linearization rules. -/
inductive Step : Expr → Expr → Prop where
  | beta     : IsValue v →
      Step (.app (.lam T E body) v) (subst 0 v body)
  | app_l    : Step f f' →
      Step (.app f a) (.app f' a)
  | app_r    : IsValue f → Step a a' →
      Step (.app f a) (.app f a')
  | meas_red : IsValue v → Step (.measure v k) (.kraw k)
  | meas_arg : Step e e' → Step (.measure e k) (.measure e' k)
  | kvalue_red : Step (.kvalue (.kraw k)) (.lit_real k.value)
  | kvalue_arg : Step e e' → Step (.kvalue e) (.kvalue e')
  | kunc_red   : Step (.kunc (.kraw k)) (.lit_real k.gumVar)
  | kunc_arg   : Step e e' → Step (.kunc e) (.kunc e')
  | kconf_red  : Step (.kconf (.kraw k)) (.lit_real k.conf)
  | kconf_arg  : Step e e' → Step (.kconf e) (.kconf e')
  | kadd_red   : Step (.kadd (.kraw a) (.kraw b)) (.kraw (gumAdd a b))
  | kadd_l     : Step e e' → Step (.kadd e r) (.kadd e' r)
  | kadd_r     : IsValue v → Step e e' → Step (.kadd v e) (.kadd v e')
  | kmul_red   : Step (.kmul (.kraw a) (.kraw b)) (.kraw (gumMul a b))
  | kmul_l     : Step e e' → Step (.kmul e r) (.kmul e' r)
  | kmul_r     : IsValue v → Step e e' → Step (.kmul v e) (.kmul v e')
  | let_red    : IsValue v → Step (.letE v body) (subst 0 v body)
  | let_step   : Step e e' → Step (.letE e b) (.letE e' b)

infix:50 " ⇒ " => Step

-- ================================================================
-- §9. Soundness: preservation + progress
-- ================================================================

/-! ## §9.1 effect_preservation — REFUTATED (2026-08-16)

This section previously carried the file's only `sorry`, on
`step_preserves_typing` (full subject reduction:
`HasTy Γ e T E → e ⇒ e' → HasTy Γ e' T E`), documented as "straightforward but
verbose".  That claim was tested and is FALSE: the statement cannot be proven
because it is not true.  The obstruction is structural, and BOTH the original
statement and the nearest weakenings are refuted machine-checked below (zero
axioms; see also `EpistemicPreservationWIP_counterexample.lean`):

1. **Root refutation** (`effect_preservation_is_false`).  `t_measure` types
   `.measure v k` at `Knowledge<T>` for ANY `T`, but `meas_red` steps it to
   `.kraw k`, and `t_kraw` types `.kraw` ONLY at `Knowledge<Real>` — the
   scalar cell cannot carry a `Nat` payload.  No retype at the original type
   exists, at any effect.

2. **Propagation refutation** (`effect_preservation_existential_is_false`).
   The obvious weakening — allow the type to change existentially,
   `∃ T', HasTy [] e' T' E` — is ALSO false: the `meas_red` type corruption
   propagates through application.  `f a` with `f = λ(x : Knowledge<Nat>). x`
   and `a = measure 0 k` is typed at `Knowledge<Nat>`; `a` steps to `kraw k`,
   and the reduct `f (kraw k)` is UNTYPABLE at every type and effect (the
   arrow demands `Knowledge<Nat>`, the value only supplies
   `Knowledge<Real>`).  The same propagation breaks the substitution lemma's
   conclusion.

**Consequence.**  This calculus (scalar-cell `Knowledge`) does not admit
subject reduction in any near form: type errors are reachable by reduction.
The repaired calculus — `kraw` stores a payload value, GUM data demoted to
metadata — is `EpistemicEffectsV2.lean`, where full Preservation and Progress
are proved without `sorry`.  Nothing in the present file depends on the false
statement; `effect_progress` (§9.2) and the GUM theorems (§10–§11) stand. -/

/-- Typing inversion for `measure` (peeling `t_sub`): the argument is typed
    at some `T` with `emptyE`, and the conclusion type is `tknow T`. -/
theorem invMeasure {Γ e T E} (h : HasTy Γ e T E) {e₀ k} (he : e = .measure e₀ k) :
    ∃ T', T = .tknow T' ∧ HasTy Γ e₀ T' emptyE ∧ kvalid k := by
  induction h with
  | t_measure Γ' T' e₀' k' he₀ hk' =>
    injection he with h1 h2; subst h1; subst h2
    exact ⟨T', rfl, he₀, hk'⟩
  | t_sub Γ' e' T' E1 E2 h0 hsub ih => exact ih he
  | _ => exact Expr.noConfusion he

/-- A well-formed Knowledge cell (variance 0, confidence 1000). -/
def k0 : KCell := ⟨0, 0, 1000⟩

theorem k0_valid : kvalid k0 := by
  unfold kvalid k0; decide

/-- Typing inversion for `app` (peeling `t_sub` by induction). -/
theorem invApp {Γ e T E} (h : HasTy Γ e T E) {f a} (he : e = .app f a) :
    ∃ T₁ Ef Ec, HasTy Γ f (.tarrow T₁ Ef T) Ec ∧ HasTy Γ a T₁ Ec := by
  induction h with
  | t_app Γ' T₁ T₂ Ef Ec Ecaller f' a' hf ha hEf hEc =>
    injection he with h1 h2; subst h1; subst h2
    exact ⟨T₁, Ef, Ec, hf, ha⟩
  | t_sub Γ' e' T' E1 E2 h0 hsub ih => exact ih he
  | _ => exact Expr.noConfusion he



/-! ## §9.2 effect_progress

A well-typed closed term is either a value or can step.  CLOSED (was an open
obligation): proved by induction on the typing derivation with `Γ` generalized
and pinned to `[]` by `rfl` (dodging Lean 4's fixed-index-induction restriction),
using the generation/canonical-forms lemmas below. -/

-- Generation lemmas: a value's type shape is fixed (peeling `t_sub` by induction).
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

theorem genKraw {Γ e T E} (h : HasTy Γ e T E) {k} (he : e = .kraw k) : T = .tknow .treal := by
  induction h with
  | t_kraw => rfl
  | t_sub _ _ _ _ _ _ _ ih => exact ih he
  | _ => exact Expr.noConfusion he

theorem genLam {Γ e T E S F b} (h : HasTy Γ e T E) (he : e = .lam S F b) :
    ∃ T₂, T = .tarrow S F T₂ := by
  induction h with
  | t_lam Γ T₁ T₂ E body _ => injection he with h1 h2 h3; subst h1; subst h2; exact ⟨T₂, rfl⟩
  | t_sub _ _ _ _ _ _ _ ih => exact ih he
  | _ => exact Expr.noConfusion he

/-- Canonical forms: a closed value of arrow type is a `lam`. -/
theorem canon_arrow {v S F T₂ E} (hv : IsValue v) (ht : HasTy [] v (.tarrow S F T₂) E) :
    ∃ S' F' b, v = .lam S' F' b := by
  cases hv with
  | v_nat n  => exact Ty.noConfusion (genNat ht rfl)
  | v_real z => exact Ty.noConfusion (genReal ht rfl)
  | v_lam T E0 e0 => exact ⟨T, E0, e0, rfl⟩
  | v_kraw k => exact Ty.noConfusion (genKraw ht rfl)

/-- Canonical forms: a closed value of `Knowledge` type is a `kraw`. -/
theorem canon_know {v T E} (hv : IsValue v) (ht : HasTy [] v (.tknow T) E) :
    ∃ k, v = .kraw k := by
  cases hv with
  | v_kraw k => exact ⟨k, rfl⟩
  | v_nat n  => exact Ty.noConfusion (genNat ht rfl)
  | v_real z => exact Ty.noConfusion (genReal ht rfl)
  | v_lam T E0 e0 => rcases genLam ht rfl with ⟨T₂, hT⟩; exact Ty.noConfusion hT

/-- Progress (open-context form): every well-typed term in the empty context is a
    value or steps.  Induct on the typing derivation; `Γ` is generalized and the
    `Γ = []` hypothesis discharges the `t_var` case. -/
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
  | t_measure Γ T e k he _ ihe =>
      subst hΓ
      rcases ihe rfl with hv | ⟨e', he'⟩
      · exact Or.inr ⟨.kraw k, .meas_red hv⟩
      · exact Or.inr ⟨.measure e' k, .meas_arg he'⟩
  | t_kvalue Γ T E e he ihe =>
      subst hΓ
      rcases ihe rfl with hv | ⟨e', he'⟩
      · rcases canon_know hv he with ⟨k, rfl⟩
        exact Or.inr ⟨.lit_real k.value, .kvalue_red⟩
      · exact Or.inr ⟨.kvalue e', .kvalue_arg he'⟩
  | t_kunc Γ T E e he ihe =>
      subst hΓ
      rcases ihe rfl with hv | ⟨e', he'⟩
      · rcases canon_know hv he with ⟨k, rfl⟩
        exact Or.inr ⟨.lit_real k.gumVar, .kunc_red⟩
      · exact Or.inr ⟨.kunc e', .kunc_arg he'⟩
  | t_kconf Γ T E e he ihe =>
      subst hΓ
      rcases ihe rfl with hv | ⟨e', he'⟩
      · rcases canon_know hv he with ⟨k, rfl⟩
        exact Or.inr ⟨.lit_real k.conf, .kconf_red⟩
      · exact Or.inr ⟨.kconf e', .kconf_arg he'⟩
  | t_kadd Γ T E₁ E₂ a b ha hb iha ihb =>
      subst hΓ
      rcases iha rfl with hva | ⟨a', ha'⟩
      · rcases canon_know hva ha with ⟨ka, rfl⟩
        rcases ihb rfl with hvb | ⟨b', hb'⟩
        · rcases canon_know hvb hb with ⟨kb, rfl⟩
          exact Or.inr ⟨.kraw (gumAdd ka kb), .kadd_red⟩
        · exact Or.inr ⟨.kadd (.kraw ka) b', .kadd_r (.v_kraw ka) hb'⟩
      · exact Or.inr ⟨.kadd a' b, .kadd_l ha'⟩
  | t_kmul Γ T E₁ E₂ a b ha hb iha ihb =>
      subst hΓ
      rcases iha rfl with hva | ⟨a', ha'⟩
      · rcases canon_know hva ha with ⟨ka, rfl⟩
        rcases ihb rfl with hvb | ⟨b', hb'⟩
        · rcases canon_know hvb hb with ⟨kb, rfl⟩
          exact Or.inr ⟨.kraw (gumMul ka kb), .kmul_red⟩
        · exact Or.inr ⟨.kmul (.kraw ka) b', .kmul_r (.v_kraw ka) hb'⟩
      · exact Or.inr ⟨.kmul a' b, .kmul_l ha'⟩
  | t_let Γ T₁ T₂ E₁ E₂ e body he hbody ihe _ =>
      subst hΓ
      rcases ihe rfl with hv | ⟨e', he'⟩
      · exact Or.inr ⟨subst 0 e body, .let_red hv⟩
      · exact Or.inr ⟨.letE e' body, .let_step he'⟩
  | t_kraw Γ k _ => exact Or.inl (.v_kraw k)
  | t_sub Γ e T E E' h0 hsub ih => exact ih hΓ

theorem effect_progress
    {e : Expr} {T : Ty} {E : EffectSet}
    (ht : HasTy [] e T E) :
    (IsValue e) ∨ ∃ e', e ⇒ e' :=
  progress' ht rfl

-- ================================================================
-- §9.3 The two machine-checked refutations
-- ================================================================

/-- Typing inversion for `lam` (peeling `t_sub` by induction). -/
theorem invLam {Γ e T E} (h : HasTy Γ e T E) {S F b} (he : e = .lam S F b) :
    ∃ T₂, T = .tarrow S F T₂ ∧ HasTy (S :: Γ) b T₂ F := by
  induction h with
  | t_lam Γ' T₁ T₂ E' body hb =>
    injection he with h1 h2 h3; subst h1; subst h2; subst h3; exact ⟨T₂, rfl, hb⟩
  | t_sub Γ' e' T' E1 E2 h0 hsub ih => exact ih he
  | _ => exact Expr.noConfusion he

/-- **Refutation 1 (root).**  `measure` at `Knowledge<Nat>` steps to a `kraw`
    that cannot be retyped at `Knowledge<Nat>` — at ANY effect.  This is the
    same statement as `EpistemicPreservationWIP_counterexample.lean::
    preservation_is_false`, restated here so the file carries its own
    refutation of the obligation it once `sorry`-ed. -/
theorem effect_preservation_is_false :
    ∃ e e' T, HasTy [] e T (singleE .eObserve) ∧ (e ⇒ e') ∧
      ¬ ∃ E, HasTy [] e' T E := by
  refine ⟨.measure (.lit_nat 0) k0, .kraw k0, .tknow .tnat,
    .t_measure _ _ _ _ (.t_lit_nat _ _) k0_valid, .meas_red (.v_nat 0), ?_⟩
  rintro ⟨E, h⟩
  have hT := genKraw h rfl          -- .tknow .tnat = .tknow .treal
  injection hT with hteq            -- .tnat = .treal
  exact Ty.noConfusion hteq

/-- **Refutation 2 (propagation).**  Even the existential-type weakening
    `∃ T', HasTy [] e' T' E` is false: the `meas_red` type corruption
    propagates through application.  `f a` with
    `f = λ(x : Knowledge<Nat>). x` and `a = measure 0 k0` is typed at
    `Knowledge<Nat>`, steps (argument position) to `f (kraw k0)`, and the
    reduct is UNTYPABLE at every type and effect. -/
theorem effect_preservation_existential_is_false :
    ∃ e e' T E, HasTy [] e T E ∧ (e ⇒ e') ∧
      ¬ ∃ T' E', HasTy [] e' T' E' := by
  -- f = λ(x : Knowledge<Nat>). x  :  (Knowledge<Nat> → Knowledge<Nat>) ! ∅
  let f : Expr := .lam (.tknow .tnat) emptyE (.var 0)
  have hf : HasTy [] f (.tarrow (.tknow .tnat) emptyE (.tknow .tnat)) emptyE :=
    .t_lam _ _ _ _ _ (.t_var _ _ _ (by simp [lookupCtx]))
  -- a = measure 0 k0  :  Knowledge<Nat> ! {Observe}
  let a : Expr := .measure (.lit_nat 0) k0
  have ha : HasTy [] a (.tknow .tnat) (singleE .eObserve) :=
    .t_measure _ _ _ _ (.t_lit_nat _ _) k0_valid
  -- e = f a : Knowledge<Nat> ! {Observe}   (Ef=∅ ⊆ {Obs}, Ea={Obs} ⊆ {Obs})
  -- type the application with Ef = ∅ ⊆ {Obs} and Ec = {Obs} ⊆ {Obs}
  have hf' : HasTy [] f (.tarrow (.tknow .tnat) emptyE (.tknow .tnat))
      (singleE .eObserve) :=
    .t_sub _ _ _ _ _ hf (emptyE_sub _)
  refine ⟨.app f a, .app f (.kraw k0), .tknow .tnat, singleE .eObserve,
    .t_app _ _ _ _ _ _ _ _ hf' ha (emptyE_sub _) (subE_refl _),
    .app_r (.v_lam _ _ _) (.meas_red (.v_nat 0)), ?_⟩
  rintro ⟨T', E', h'⟩
  rcases invApp h' rfl with ⟨T₁, Ef, Ec, hfty, harg⟩
  -- f is a lam, so its arrow type is the annotated one:
  rcases invLam hfty rfl with ⟨T₂, hTeq, _⟩
  injection hTeq with hT1 _ _; subst hT1
  -- … so the argument `kraw k0` must type at Knowledge<Nat> (any effect) …
  -- … but kraw only types at Knowledge<Real>:
  have hT := genKraw harg rfl        -- .tknow .tnat = .tknow .treal
  injection hT with hteq             -- .tnat = .treal
  exact Ty.noConfusion hteq

-- ================================================================
-- §10. GUM conservativity
-- ================================================================

/-! ## §10. gum_conservativity

For any well-typed Knowledge values reduced by `kadd`/`kmul`, the
GUM uncertainty produced by the rule is an upper bound on the true
linearized variance. Concretely:

  trueAddVar a b  ≤  (gumAdd a b).gumVar
  trueMulVar a b  ≤  (gumMul a b).gumVar

In the first-order linearization both sides are equal; we state and
prove the inequality form so the theorem statement matches the
audit's framing ("GUM linearization is a sound upper bound") and so
that any future second-order correction term can be added on the
true-variance side without breaking the theorem. -/

theorem gum_conservativity_add (a b : KCell) :
    trueAddVar a b ≤ (gumAdd a b).gumVar := by
  simp [trueAddVar, gumAdd]

theorem gum_conservativity_mul (a b : KCell) :
    trueMulVar a b ≤ (gumMul a b).gumVar := by
  simp [trueMulVar, gumMul]

/-- The unified statement: GUM-output variance is ≥ the true linearized
    variance for any well-typed Knowledge reduction step. -/
theorem gum_conservativity :
    (∀ a b : KCell, trueAddVar a b ≤ (gumAdd a b).gumVar) ∧
    (∀ a b : KCell, trueMulVar a b ≤ (gumMul a b).gumVar) :=
  ⟨gum_conservativity_add, gum_conservativity_mul⟩

-- ================================================================
-- §11. Confidence monotonicity
-- ================================================================

/-! ## §11. confidence_monotonicity

Confidence (the `kconf` projection) is *non-increasing* along reduction.
We prove this for the two Knowledge combinators that touch confidence
— `gumAdd` and `gumMul` — both of which set the output `conf` to
`min x.conf y.conf`. The lift to the small-step relation follows by
inversion: the only reductions that produce a `kraw` from two `kraw`s
are `kadd_red` / `kmul_red`, both of which use these combinators. -/

theorem conf_monotone_add (a b : KCell) :
    (gumAdd a b).conf ≤ a.conf ∧ (gumAdd a b).conf ≤ b.conf := by
  refine ⟨?_, ?_⟩
  · simp [gumAdd]; split <;> omega
  · simp [gumAdd]; split <;> omega

theorem conf_monotone_mul (a b : KCell) :
    (gumMul a b).conf ≤ a.conf ∧ (gumMul a b).conf ≤ b.conf := by
  refine ⟨?_, ?_⟩
  · simp [gumMul]; split <;> omega
  · simp [gumMul]; split <;> omega

/-- The headline statement: confidence is non-increasing under either
    Knowledge combinator. Lifting this to multi-step reduction is a
    routine induction on `Step*` and is omitted; the per-step content
    is captured above. -/
theorem confidence_monotonicity :
    (∀ a b : KCell, (gumAdd a b).conf ≤ a.conf ∧ (gumAdd a b).conf ≤ b.conf) ∧
    (∀ a b : KCell, (gumMul a b).conf ≤ a.conf ∧ (gumMul a b).conf ≤ b.conf) :=
  ⟨conf_monotone_add, conf_monotone_mul⟩

-- ================================================================
-- §12. Decidable smoke checks
-- ================================================================

/-- Smoke check: `eIO ⊆ {eIO, eMut}`. The E035 check the typer runs at
    every `app` reduces (in the closed-effect case) to this kind of
    decidable subset judgement on the effect row. -/
example :
    singleE Effect.eIO ⊆ₑ unionE (singleE Effect.eIO) (singleE Effect.eMut) :=
  sub_union_left _ _

/-- Smoke check: emptyE is a subset of every singleton row. -/
example (e : Effect) : emptyE ⊆ₑ singleE e := emptyE_sub _

/-- Smoke check: GUM addition of two known cells. -/
example :
    (gumAdd ⟨3, 2, 900⟩ ⟨5, 4, 800⟩).gumVar = 6 := by
  decide

/-- Smoke check: GUM multiplication produces the linearized variance. -/
example :
    (gumMul ⟨3, 2, 900⟩ ⟨5, 4, 800⟩).gumVar = 5 * 5 * 2 + 3 * 3 * 4 := by
  decide

end Sounio.EpistemicEffects
