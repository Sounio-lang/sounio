import EpistemicEffects

/-!
# Epistemic-Effect Calculus NS — `Knowledge⟨T, N⟩` with a noise-symbol source-set

The NS extension of `EpistemicEffectsV2.lean` (Paper A, §5–§6): the Knowledge type
carries a **noise-set** `N ∈ L = 𝒫(S) ∪ {⊤}`, and the independence-assuming operators
`kadd`/`kmul` carry the **checked precondition** `disjoint(Nₐ, N_b)` (E230 when it fails).

What is mechanized here, and how it composes into Paper A's Theorem 6.4
("no first-order anti-garbling"):

* **Lemma 1 — general form, Mathlib-free.** Runtime Knowledge values carry their true
  first-order *affine form* `a : Aff` (a formal sum of `c·ε_s` monomials over measurement
  sources `s`). `trueVar_append` proves `Var(a + b) = Var a + Var b + 2⟨a,b⟩` for **all**
  forms (the model file `SounioAntiGarblingModel.lean` only discharged Int witnesses), and
  `trueVar_mul` the delta-method analogue `Var(y·a + x·b) = y²Var a + x²Var b + 2xy⟨a,b⟩`.
  `inner_disjoint` proves disjoint support ⟹ `⟨a,b⟩ = 0`.
* **Lemma 2 — support over-approximation.** `Covers N a` (every source of `a` is in `N`,
  `⊤` trivially) is part of the typing of runtime values (`t_kraw`) and is preserved by
  every transfer (`covers_union`, `covers_scale`); `support_over_approx` extracts it from
  any typing derivation, and preservation carries it along every reduction step.
* **NS-extended type safety.** `progress` and `preservation` for the `N`-annotated
  calculus — the row Paper A §6.4 marked `[pending wire]`.
&  its *true first-order* variance (`m.gumVar = trueVar a`). Type safety alone does not give this
  (§6.1); `exact_preservation` proves that under NS typing it is an invariant of
  reduction — at `kadd_red`/`kmul_red` the disjointness premise + Lemma 2 + Lemma 1 make
  the defective scalar combinators `gAddMeta`/`gMulMeta` *exact*.
* **Theorem 6.4 as stated.** `typed_agfree`: no independence-assuming operator inside a
  well-typed term *whose operands are already runtime values* has nonzero covariance —
  `AGFree` is value-restricted by design, because an operator only computes at a redex;
  `soundness_star` re-establishes it on every term along `Steps` (⇒*), so the check is live
  at every operator actually *reached*.
* **The sabotage witness, in the kernel.** `x + x` on one source: the (unchanged,
  defective) operational semantics steps from an exact term to an inexact one
  (`x_plus_x_understates`, gap `= 2⟨x,x⟩`), and the NS type system rejects that very
  term for *every* annotation `N` (`x_plus_x_untypable`); `x + opaque(y)` is rejected purely by the ⊤ clause
  (`x_plus_top_untypable`, with `x + y` itself admitted); `measure s + measure s` is untypable at
  source level (`measure_plus_measure_untypable`); `x + y` on disjoint sources stays exact.

The operational semantics is **deliberately the defective one** (`gAddMeta` = `ep_add`,
`gMulMeta` = `ep_mul`, no covariance term): the discipline lives entirely in the types.

Design notes. Sources are `Nat` labels; each `measure` site names its source. **Modelling
axiom (honest labelling), assumed not proved:** distinct labels are distinct physical sources.
Under that axiom two sites sharing a label only over-approximate; WITHOUT it — two physically
correlated sites given distinct labels — the calculus under-approximates the true covariance,
and every theorem below is relative to the axiom (Paper A §6.4 residual (iv)). The type
system tracks sources; it does not discover them. **Scope:** `trueVar a = ⟨a,a⟩` is the
variance of `Σ c_s ε_s` under independent unit-variance symbols *by definition*; no
distributional / sampling semantics is modelled — Lemma 1 and Theorem 6.4 are algebraic. `Aff` allows
duplicate sources — `coeff` sums them — so `a ++ b` is exact affine addition and no
canonical form is needed. Nothing in this file is `noncomputable`; every witness is `decide`.

STATUS: see the `#print axioms` block at the end. sorry-free. Lean 4.33.1, Mathlib-free.
-/

namespace Sounio.EpistemicEffectsNS

open Sounio.EpistemicEffects (Effect EffectSet emptyE singleE unionE subE)

-- ================================================================
-- §A. First-order affine forms: the TRUE noise-symbol content of a value
-- ================================================================

/-- A first-order affine form: a formal sum of monomials `c · ε_s` (source, coefficient).
    Duplicated sources are allowed; `coeff` sums them. -/
abbrev Aff := List (Nat × Int)

/-- The coefficient of noise symbol `ε_s` in `a`. -/
def coeff (a : Aff) (s : Nat) : Int :=
  match a with
  | [] => 0
  | (t, c) :: r => (if t = s then c else 0) + coeff r s

/-- Covariance ⟨a,b⟩ = Σ_s coeff a s · coeff b s, computed as Σ_{(s,c)∈a} c · coeff b s. -/
def inner (a b : Aff) : Int :=
  match a with
  | [] => 0
  | (s, c) :: r => c * coeff b s + inner r b

/-- The true first-order variance: ⟨a,a⟩. -/
def trueVar (a : Aff) : Int := inner a a

/-- Scalar multiple `k·a` (the delta-method linearisation coefficient). -/
def scale (k : Int) (a : Aff) : Aff :=
  match a with
  | [] => []
  | (s, c) :: r => (s, k * c) :: scale k r

theorem inner_nil_right (a : Aff) : inner a [] = 0 := by
  induction a with
  | nil => rfl
  | cons p r ih => cases p; simp [inner, coeff, ih]

theorem coeff_append (a b : Aff) (s : Nat) : coeff (a ++ b) s = coeff a s + coeff b s := by
  induction a with
  | nil => simp [coeff]
  | cons p r ih => cases p; simp [coeff, ih]; omega

theorem inner_append_left (a b c : Aff) : inner (a ++ b) c = inner a c + inner b c := by
  induction a with
  | nil => simp [inner]
  | cons p r ih => cases p; simp [inner, ih]; omega

theorem inner_append_right (a b c : Aff) : inner a (b ++ c) = inner a b + inner a c := by
  induction a with
  | nil => simp [inner]
  | cons p r ih => cases p with | mk s d => simp [inner, ih, coeff_append, Int.mul_add]; omega

theorem coeff_single (s t : Nat) (c : Int) : coeff [(s, c)] t = if s = t then c else 0 := by
  simp [coeff]

theorem inner_single_right (b : Aff) (s : Nat) (c : Int) : inner b [(s, c)] = c * coeff b s := by
  induction b with
  | nil => simp [inner, coeff]
  | cons p r ih =>
    cases p with | mk t d =>
    simp [inner, coeff, ih]
    by_cases h : t = s
    · subst h; simp [Int.mul_add, Int.mul_comm]
    · have h' : ¬ s = t := fun e => h e.symm
      simp [h, h']

theorem inner_comm (a b : Aff) : inner a b = inner b a := by
  induction a with
  | nil => simp [inner, inner_nil_right]
  | cons p r ih =>
    cases p with | mk s c =>
    have : inner b ((s, c) :: r) = inner b ([(s, c)] ++ r) := rfl
    rw [this, inner_append_right, inner_single_right]
    simp [inner, ih]

/-- **Lemma 1 (general form).** `Var(a + b) = Var a + Var b + 2⟨a,b⟩`.
    The naive scalar add (`gAddMeta`, = `ep_add`) reports `Var a + Var b`; the gap is
    exactly `2⟨a,b⟩`, and vanishes iff the covariance does. -/
theorem trueVar_append (a b : Aff) :
    trueVar (a ++ b) = trueVar a + trueVar b + 2 * inner a b := by
  unfold trueVar
  rw [inner_append_left, inner_append_right, inner_append_right, inner_comm b a]
  omega

theorem coeff_scale (k : Int) (a : Aff) (s : Nat) : coeff (scale k a) s = k * coeff a s := by
  induction a with
  | nil => simp [scale, coeff]
  | cons p r ih => cases p with | mk t c => simp [scale, coeff, ih, Int.mul_add]; split <;> simp

theorem inner_scale_left (k : Int) (a b : Aff) : inner (scale k a) b = k * inner a b := by
  induction a with
  | nil => simp [scale, inner]
  | cons p r ih => cases p with | mk s c => simp [scale, inner, ih, Int.mul_add, Int.mul_assoc]

theorem inner_scale_right (k : Int) (a b : Aff) : inner a (scale k b) = k * inner a b := by
  rw [inner_comm, inner_scale_left, inner_comm]

/-- **Lemma 1 for products (delta method).** The first-order form of `a·b` at the point
    `(x, y)` is `y·a + x·b`, whose variance is `y²Var a + x²Var b + 2xy⟨a,b⟩`; `gMulMeta`
    (= `ep_mul`) reports the first two terms. -/
theorem trueVar_mul (x y : Int) (a b : Aff) :
    trueVar (scale y a ++ scale x b)
      = y * y * trueVar a + x * x * trueVar b + 2 * (x * y) * inner a b := by
  rw [trueVar_append]
  unfold trueVar
  rw [inner_scale_left, inner_scale_right, inner_scale_left, inner_scale_right,
      inner_scale_left, inner_scale_right]
  simp [Int.mul_assoc, Int.mul_comm, Int.mul_left_comm]

theorem coeff_absent (b : Aff) (s : Nat) (h : ∀ p ∈ b, p.1 ≠ s) : coeff b s = 0 := by
  induction b with
  | nil => rfl
  | cons p r ih =>
    cases p with | mk t d =>
    have ht : t ≠ s := h (t, d) (by simp)
    simp [coeff, ht]; exact ih (fun q hq => h q (by simp [hq]))

/-- Disjoint true supports ⟹ zero covariance (the DISJ side of Crux #1). -/
theorem inner_disjoint (a b : Aff) (h : ∀ p ∈ a, ∀ q ∈ b, p.1 ≠ q.1) : inner a b = 0 := by
  induction a with
  | nil => rfl
  | cons p r ih =>
    cases p with | mk s c =>
    have hc : coeff b s = 0 := coeff_absent b s (fun q hq => (h (s, c) (by simp) q hq).symm)
    simp [inner, hc]; exact ih (fun p hp q hq => h p (by simp [hp]) q hq)

-- ================================================================
-- §B. The noise-set lattice  L = 𝒫(S) ∪ {⊤}  and the abstraction `Covers`
-- ================================================================

/-- `none` = ⊤ (unknown source-set, conservative); `some l` = a finite set of sources. -/
abbrev NS := Option (List Nat)

def nsTop : NS := none
def nsEmpty : NS := some []
def nsSingle (s : Nat) : NS := some [s]

/-- Join of `L`: union, with ⊤ absorbing. -/
def nsUnion : NS → NS → NS
  | some a, some b => some (a ++ b)
  | _, _ => none

/-- Membership; everything is (potentially) in ⊤. -/
def nsMem (s : Nat) : NS → Prop
  | none => True
  | some l => s ∈ l

/-- The checked precondition of (Add-Indep): both finite and set-disjoint.
    **⊤ is never disjoint from anything.** Decidable — this is the E230 test. -/
def nsDisjoint : NS → NS → Bool
  | some a, some b => a.all (fun s => decide (¬ s ∈ b))
  | _, _ => false

theorem nsDisjoint_sound {la lb : List Nat} (h : nsDisjoint (some la) (some lb) = true) :
    ∀ s ∈ la, ¬ s ∈ lb := by
  intro s hs
  have := (List.all_eq_true.mp h) s hs
  exact of_decide_eq_true this

theorem nsDisjoint_top_left (N : NS) : nsDisjoint none N = false := rfl
theorem nsDisjoint_top_right (N : NS) : nsDisjoint N none = false := by cases N <;> rfl

/-- A shared member refutes disjointness — for every annotation, including ⊤. -/
theorem nsDisjoint_of_shared {s : Nat} {Na Nb : NS} (ha : nsMem s Na) (hb : nsMem s Nb) :
    nsDisjoint Na Nb = false := by
  cases Na with
  | none => rfl
  | some la =>
    cases Nb with
    | none => rfl
    | some lb =>
      simp only [nsMem] at ha hb
      simp only [nsDisjoint]
      rw [Bool.eq_false_iff]
      intro h
      have hs := (List.all_eq_true.mp h) s ha
      exact (of_decide_eq_true hs) hb

/-- **The abstraction.** `Covers N a`: every source that actually carries uncertainty
    into `a` is a member of the tracked noise-set `N` (⊤ trivially covers). -/
def Covers (N : NS) (a : Aff) : Prop := ∀ p ∈ a, nsMem p.1 N

theorem covers_top (a : Aff) : Covers nsTop a := fun _ _ => trivial
theorem covers_empty : Covers nsEmpty [] := fun _ h => by simp at h
theorem covers_single (s : Nat) (c : Int) : Covers (nsSingle s) [(s, c)] := by
  intro p hp; simp at hp; subst hp; simp [nsSingle, nsMem]

theorem nsMem_union_left {s : Nat} {Na Nb : NS} (h : nsMem s Na) : nsMem s (nsUnion Na Nb) := by
  cases Na with
  | none => cases Nb <;> trivial
  | some la =>
    cases Nb with
    | none => trivial
    | some lb => simp only [nsMem, nsUnion] at h ⊢; exact List.mem_append_left _ h

theorem nsMem_union_right {s : Nat} {Na Nb : NS} (h : nsMem s Nb) : nsMem s (nsUnion Na Nb) := by
  cases Na with
  | none => cases Nb <;> trivial
  | some la =>
    cases Nb with
    | none => trivial
    | some lb => simp only [nsMem, nsUnion] at h ⊢; exact List.mem_append_right _ h

/-- Transfer soundness for the join: `∪` covers affine addition. -/
theorem covers_union {Na Nb : NS} {a b : Aff} (ha : Covers Na a) (hb : Covers Nb b) :
    Covers (nsUnion Na Nb) (a ++ b) := by
  intro p hp
  rcases List.mem_append.mp hp with h | h
  · exact nsMem_union_left (ha p h)
  · exact nsMem_union_right (hb p h)

/-- Transfer soundness for scaling: linearisation does not change the support. -/
theorem covers_scale {N : NS} {a : Aff} (k : Int) (h : Covers N a) : Covers N (scale k a) := by
  intro p hp
  induction a with
  | nil => simp [scale] at hp
  | cons q r ih =>
    cases q with | mk s c =>
    simp only [scale, List.mem_cons] at hp
    rcases hp with hp | hp
    · subst hp; exact h (s, c) (by simp)
    · exact ih (fun q hq => h q (by simp [hq])) hp

/-- The membership-based `Covers` entails the coefficient-based containment Lemma 2 is
    worded on: every source with a NONZERO coefficient in `a` is a member of `N`.
    (xai review 2026-08-30, item 2: `Covers` is the stricter, conservative invariant.) -/
theorem covers_coeff {N : NS} {a : Aff} (h : Covers N a) : ∀ s, coeff a s ≠ 0 → nsMem s N := by
  intro s hs
  induction a with
  | nil => simp [coeff] at hs
  | cons p r ih =>
    cases p with | mk t c =>
    by_cases ht : t = s
    · subst ht; exact h (t, c) (by simp)
    · simp [coeff, ht] at hs
      exact ih (fun q hq => h q (by simp [hq])) hs

/-- **Lemma 2 ⟹ Crux #1.** Tracked sets disjoint + both covered ⟹ true supports disjoint. -/
theorem covers_disjoint {Na Nb : NS} {a b : Aff} (ha : Covers Na a) (hb : Covers Nb b)
    (hd : nsDisjoint Na Nb = true) : ∀ p ∈ a, ∀ q ∈ b, p.1 ≠ q.1 := by
  intro p hp q hq heq
  cases Na with
  | none => simp [nsDisjoint] at hd
  | some la =>
    cases Nb with
    | none => simp [nsDisjoint] at hd
    | some lb =>
      have hpa : p.1 ∈ la := ha p hp
      have hqb : q.1 ∈ lb := hb q hq
      exact nsDisjoint_sound hd p.1 hpa (heq ▸ hqb)

/-- The composed local criterion: NS-disjoint tracked sets ⟹ zero true covariance. -/
theorem inner_zero_of_ns {Na Nb : NS} {a b : Aff} (ha : Covers Na a) (hb : Covers Nb b)
    (hd : nsDisjoint Na Nb = true) : inner a b = 0 :=
  inner_disjoint a b (covers_disjoint ha hb hd)

-- ================================================================
-- §C. Types, terms, values
-- ================================================================

/-- Types: the Knowledge type carries its noise-set. -/
inductive Ty where
  | tnat   : Ty
  | treal  : Ty
  | tarrow : Ty → EffectSet → Ty → Ty
  | tknow  : Ty → NS → Ty

abbrev TyCtx := List Ty

def lookupCtx : TyCtx → Nat → Option Ty
  | [],      _     => none
  | t :: _,  0     => some t
  | _ :: ts, n + 1 => lookupCtx ts n

/-- Scalar GUM metadata (as in V2): reported variance and confidence. -/
structure KMeta where
  gumVar : Int
  conf   : Int
  deriving Repr, DecidableEq

def kvalid (m : KMeta) : Prop := 0 ≤ m.gumVar ∧ 0 ≤ m.conf ∧ m.conf ≤ 1000

/-- Terms. `measure e c conf s`: measure `e` with standard uncertainty `c` on source `s`.
    `certain e`: an exact constant (no source). `kraw v m a`: a runtime Knowledge value —
    payload `v`, reported metadata `m`, and its TRUE affine form `a`. -/
inductive Expr where
  | lit_nat  : Nat → Expr
  | lit_real : Int → Expr
  | var      : Nat → Expr
  | lam      : Ty → EffectSet → Expr → Expr
  | app      : Expr → Expr → Expr
  | measure  : Expr → Int → Int → Nat → Expr
  | certain  : Expr → Expr
  | opaque   : Expr → Expr
  | kvalue   : Expr → Expr
  | kunc     : Expr → Expr
  | kconf    : Expr → Expr
  | kadd     : Expr → Expr → Expr
  | kmul     : Expr → Expr → Expr
  | letE     : Expr → Expr → Expr
  | kraw     : Expr → KMeta → Aff → Expr

inductive IsValue : Expr → Prop where
  | v_nat   : ∀ n, IsValue (.lit_nat n)
  | v_real  : ∀ z, IsValue (.lit_real z)
  | v_lam   : ∀ T E e, IsValue (.lam T E e)
  | v_kraw  : ∀ {v} m a, IsValue v → IsValue (.kraw v m a)

-- ================================================================
-- §D. Typing  Γ ⊢ e : T ! E   (Paper A §5: Measure, Exact, Add-Indep, Mul-Indep)
-- ================================================================

inductive HasTy : TyCtx → Expr → Ty → EffectSet → Prop where
  | t_lit_nat  : ∀ Γ n, HasTy Γ (.lit_nat n) .tnat emptyE
  | t_lit_real : ∀ Γ z, HasTy Γ (.lit_real z) .treal emptyE
  | t_var      : ∀ Γ n T, lookupCtx Γ n = some T → HasTy Γ (.var n) T emptyE
  | t_lam      : ∀ Γ T₁ T₂ E body,
      HasTy (T₁ :: Γ) body T₂ E → HasTy Γ (.lam T₁ E body) (.tarrow T₁ E T₂) emptyE
  | t_app      : ∀ Γ T₁ T₂ Ef Ec Ecaller f a,
      HasTy Γ f (.tarrow T₁ Ef T₂) Ec → HasTy Γ a T₁ Ec →
      Ef ⊆ₑ Ecaller → Ec ⊆ₑ Ecaller → HasTy Γ (.app f a) T₂ Ecaller
  /-- (Measure): a measurement seeds the singleton source-set `{s}`. -/
  | t_measure  : ∀ Γ T e c conf s,
      HasTy Γ e T emptyE → 0 ≤ conf → conf ≤ 1000 →
      HasTy Γ (.measure e c conf s) (.tknow T (nsSingle s)) (singleE .eObserve)
  /-- (Exact): a constant carries the empty source-set. -/
  | t_certain  : ∀ Γ T e,
      HasTy Γ e T emptyE → HasTy Γ (.certain e) (.tknow T nsEmpty) emptyE
  /-- (Opaque): provenance erased — the result is typed at ⊤, the conservative top. This is
      the paper's `opaque_knowledge()` fixture; ⊤ is never disjoint, so `x + opaque(y)` is
      rejected even when `y`'s true source is disjoint from `x`'s. -/
  | t_opaque   : ∀ Γ T N E e,
      HasTy Γ e (.tknow T N) E → HasTy Γ (.opaque e) (.tknow T nsTop) E
  | t_kvalue   : ∀ Γ T N E e,
      HasTy Γ e (.tknow T N) E → HasTy Γ (.kvalue e) T E
  | t_kunc     : ∀ Γ T N E e,
      HasTy Γ e (.tknow T N) E → HasTy Γ (.kunc e) .treal E
  | t_kconf    : ∀ Γ T N E e,
      HasTy Γ e (.tknow T N) E → HasTy Γ (.kconf e) .treal E
  /-- (Add-Indep): the independence-assuming add REQUIRES provably disjoint supports.
      Failure of the premise is E230. The result unions the supports. -/
  | t_kadd     : ∀ Γ Na Nb E₁ E₂ a b,
      HasTy Γ a (.tknow .treal Na) E₁ → HasTy Γ b (.tknow .treal Nb) E₂ →
      nsDisjoint Na Nb = true →
      HasTy Γ (.kadd a b) (.tknow .treal (nsUnion Na Nb)) (unionE E₁ E₂)
  /-- (Mul-Indep): identical side condition (Paper A §4, corollary). -/
  | t_kmul     : ∀ Γ Na Nb E₁ E₂ a b,
      HasTy Γ a (.tknow .treal Na) E₁ → HasTy Γ b (.tknow .treal Nb) E₂ →
      nsDisjoint Na Nb = true →
      HasTy Γ (.kmul a b) (.tknow .treal (nsUnion Na Nb)) (unionE E₁ E₂)
  | t_let      : ∀ Γ T₁ T₂ E₁ E₂ e body,
      HasTy Γ e T₁ E₁ → HasTy (T₁ :: Γ) body T₂ E₂ →
      HasTy Γ (.letE e body) T₂ (unionE E₁ E₂)
  /-- Runtime Knowledge values: the tracked `N` must COVER the true form `a` (Lemma 2 as
      a typing invariant of values). -/
  | t_kraw     : ∀ Γ T N v m a,
      HasTy Γ v T emptyE → IsValue v → kvalid m → Covers N a →
      HasTy Γ (.kraw v m a) (.tknow T N) emptyE
  | t_sub      : ∀ Γ e T E E',
      HasTy Γ e T E → E ⊆ₑ E' → HasTy Γ e T E'

-- ================================================================
-- §E. The DEFECTIVE scalar combinators (= `ep_add` / `ep_mul`, no covariance term)
-- ================================================================

def gAddMeta (ma mb : KMeta) : KMeta :=
  { gumVar := ma.gumVar + mb.gumVar
  , conf := if ma.conf ≤ mb.conf then ma.conf else mb.conf }

def gMulMeta (x : Int) (ma : KMeta) (y : Int) (mb : KMeta) : KMeta :=
  { gumVar := y * y * ma.gumVar + x * x * mb.gumVar
  , conf := if ma.conf ≤ mb.conf then ma.conf else mb.conf }

-- ================================================================
-- §F. de Bruijn shift / subst
-- ================================================================

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
  | .measure e c cf s => .measure (shift cutoff d e) c cf s
  | .certain e => .certain (shift cutoff d e)
  | .opaque e => .opaque (shift cutoff d e)
  | .kvalue e => .kvalue (shift cutoff d e)
  | .kunc e => .kunc (shift cutoff d e)
  | .kconf e => .kconf (shift cutoff d e)
  | .kadd a b => .kadd (shift cutoff d a) (shift cutoff d b)
  | .kmul a b => .kmul (shift cutoff d a) (shift cutoff d b)
  | .letE e b => .letE (shift cutoff d e) (shift (cutoff + 1) d b)
  | .kraw v m a => .kraw (shift cutoff d v) m a

def subst (n : Nat) (w : Expr) : Expr → Expr
  | .var k => if k = n then w else if k > n then .var (k - 1) else .var k
  | .lit_nat m => .lit_nat m
  | .lit_real z => .lit_real z
  | .lam T E b => .lam T E (subst (n + 1) (shift 0 1 w) b)
  | .app f a => .app (subst n w f) (subst n w a)
  | .measure e c cf s => .measure (subst n w e) c cf s
  | .certain e => .certain (subst n w e)
  | .opaque e => .opaque (subst n w e)
  | .kvalue e => .kvalue (subst n w e)
  | .kunc e => .kunc (subst n w e)
  | .kconf e => .kconf (subst n w e)
  | .kadd a b => .kadd (subst n w a) (subst n w b)
  | .kmul a b => .kmul (subst n w a) (subst n w b)
  | .letE e b => .letE (subst n w e) (subst (n + 1) (shift 0 1 w) b)
  | .kraw v m a => .kraw (subst n w v) m a

-- ================================================================
-- §G. Small-step CBV reduction — the DEFECTIVE semantics, now carrying true forms
-- ================================================================

inductive Step : Expr → Expr → Prop where
  | beta       : IsValue v → Step (.app (.lam T E body) v) (subst 0 v body)
  | app_l      : Step f f' → Step (.app f a) (.app f' a)
  | app_r      : IsValue f → Step a a' → Step (.app f a) (.app f a')
  /-- A measurement's true form is the single monomial `c·ε_s`; reported variance `c²`. -/
  | meas_red   : IsValue v →
      Step (.measure v c cf s) (.kraw v ⟨c * c, cf⟩ [(s, c)])
  | meas_arg   : Step e e' → Step (.measure e c cf s) (.measure e' c cf s)
  | cert_red   : IsValue v → Step (.certain v) (.kraw v ⟨0, 1000⟩ [])
  | cert_arg   : Step e e' → Step (.certain e) (.certain e')
  /-- Erasing provenance is a no-op on the value; only the TYPE forgets. -/
  | opaque_red : Step (.opaque (.kraw v m a)) (.kraw v m a)
  | opaque_arg : Step e e' → Step (.opaque e) (.opaque e')
  | kvalue_red : Step (.kvalue (.kraw v m a)) v
  | kvalue_arg : Step e e' → Step (.kvalue e) (.kvalue e')
  | kunc_red   : Step (.kunc (.kraw v m a)) (.lit_real m.gumVar)
  | kunc_arg   : Step e e' → Step (.kunc e) (.kunc e')
  | kconf_red  : Step (.kconf (.kraw v m a)) (.lit_real m.conf)
  | kconf_arg  : Step e e' → Step (.kconf e) (.kconf e')
  /-- Reported: `gAddMeta` (no covariance). True form: `a ++ b` (exact affine sum). -/
  | kadd_red   : Step (.kadd (.kraw (.lit_real x) ma a) (.kraw (.lit_real y) mb b))
                      (.kraw (.lit_real (x + y)) (gAddMeta ma mb) (a ++ b))
  | kadd_l     : Step e e' → Step (.kadd e r) (.kadd e' r)
  | kadd_r     : IsValue v → Step e e' → Step (.kadd v e) (.kadd v e')
  /-- Reported: `gMulMeta` (no covariance). True first-order form: `y·a + x·b`. -/
  | kmul_red   : Step (.kmul (.kraw (.lit_real x) ma a) (.kraw (.lit_real y) mb b))
                      (.kraw (.lit_real (x * y)) (gMulMeta x ma y mb) (scale y a ++ scale x b))
  | kmul_l     : Step e e' → Step (.kmul e r) (.kmul e' r)
  | kmul_r     : IsValue v → Step e e' → Step (.kmul v e) (.kmul v e')
  | let_red    : IsValue v → Step (.letE v body) (subst 0 v body)
  | let_step   : Step e e' → Step (.letE e b) (.letE e' b)

infix:50 " ⇒ " => Step

/-- Reflexive-transitive closure: every term reached during evaluation. -/
inductive Steps : Expr → Expr → Prop where
  | refl : ∀ e, Steps e e
  | step : ∀ {e e' e''}, Step e e' → Steps e' e'' → Steps e e''

infix:50 " ⇒* " => Steps

-- ================================================================
-- §H. The soundness invariant: reported variance = true variance
-- ================================================================

/-- `Exact e`: every runtime Knowledge value inside `e` reports its TRUE variance.
    This is the property type safety does NOT give (Paper A §6.1) and NS typing does. -/
def Exact : Expr → Prop
  | .kraw v m a => Exact v ∧ m.gumVar = trueVar a
  | .lam _ _ b => Exact b
  | .app f a => Exact f ∧ Exact a
  | .measure e _ _ _ => Exact e
  | .certain e => Exact e
  | .opaque e => Exact e
  | .kvalue e => Exact e
  | .kunc e => Exact e
  | .kconf e => Exact e
  | .kadd a b => Exact a ∧ Exact b
  | .kmul a b => Exact a ∧ Exact b
  | .letE e b => Exact e ∧ Exact b
  | .lit_nat _ => True
  | .lit_real _ => True
  | .var _ => True

/-- `AGFree e`: no independence-assuming operator in `e` WHOSE OPERANDS ARE RUNTIME VALUES
    has nonzero covariance. Value-restricted on purpose: an operator computes only at a redex,
    and `soundness_star` re-derives `AGFree` on every reduct, so the restriction loses no
    reached operator. -/
def AGFree : Expr → Prop
  | .kadd a b =>
      (∀ x ma a' y mb b', a = .kraw (.lit_real x) ma a' → b = .kraw (.lit_real y) mb b' →
        inner a' b' = 0) ∧ AGFree a ∧ AGFree b
  | .kmul a b =>
      (∀ x ma a' y mb b', a = .kraw (.lit_real x) ma a' → b = .kraw (.lit_real y) mb b' →
        inner a' b' = 0) ∧ AGFree a ∧ AGFree b
  | .kraw v _ _ => AGFree v
  | .lam _ _ b => AGFree b
  | .app f a => AGFree f ∧ AGFree a
  | .measure e _ _ _ => AGFree e
  | .certain e => AGFree e
  | .opaque e => AGFree e
  | .kvalue e => AGFree e
  | .kunc e => AGFree e
  | .kconf e => AGFree e
  | .letE e b => AGFree e ∧ AGFree b
  | .lit_nat _ => True
  | .lit_real _ => True
  | .var _ => True

-- ================================================================
-- §I. Generation, canonical forms, inversion
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

theorem genKraw {Γ e T E} (h : HasTy Γ e T E) {v m a} (he : e = .kraw v m a) :
    ∃ T' N, T = .tknow T' N := by
  induction h with
  | t_kraw Γ T' N v' m' a' _ _ _ _ => exact ⟨T', N, rfl⟩
  | t_sub _ _ _ _ _ _ _ ih => exact ih he
  | _ => exact Expr.noConfusion he

/-- `kraw` inversion: payload type/typing, value-ness, metadata validity, and — the
    abstraction fact — the tracked `N` covers the true form. -/
theorem invKraw {Γ e T E} (h : HasTy Γ e T E) {v m a} (he : e = .kraw v m a) :
    ∃ T' N, T = .tknow T' N ∧ HasTy Γ v T' emptyE ∧ IsValue v ∧ kvalid m ∧ Covers N a := by
  induction h with
  | t_kraw Γ' T' N v' m' a' hv' hval' hk' hc' =>
    injection he with h1 h2 h3; subst h1; subst h2; subst h3
    exact ⟨T', N, rfl, hv', hval', hk', hc'⟩
  | t_sub Γ' e' T0 E1 E2 h0 hsub ih => exact ih he
  | _ => exact Expr.noConfusion he

/-- **Lemma 2 (support over-approximation), extracted.** Any typed runtime Knowledge value
    has its true support inside its tracked noise-set. -/
theorem support_over_approx {Γ v m a T N E} (h : HasTy Γ (.kraw v m a) (.tknow T N) E) :
    ∀ p ∈ a, nsMem p.1 N := by
  rcases invKraw h rfl with ⟨T', N', hT, _, _, _, hc⟩
  injection hT with h1 h2; subst h2; exact hc

theorem canon_arrow {v S F T₂ E} (hv : IsValue v) (ht : HasTy [] v (.tarrow S F T₂) E) :
    ∃ S' F' b, v = .lam S' F' b := by
  cases hv with
  | v_nat n  => exact Ty.noConfusion (genNat ht rfl)
  | v_real z => exact Ty.noConfusion (genReal ht rfl)
  | v_lam T E0 e0 => exact ⟨T, E0, e0, rfl⟩
  | @v_kraw w m a hp => rcases genKraw ht rfl with ⟨T', N, hT⟩; exact Ty.noConfusion hT

theorem canon_know {v T N E} (hv : IsValue v) (ht : HasTy [] v (.tknow T N) E) :
    ∃ w m a, v = .kraw w m a := by
  cases hv with
  | @v_kraw w m a hp => exact ⟨w, m, a, rfl⟩
  | v_nat n  => exact Ty.noConfusion (genNat ht rfl)
  | v_real z => exact Ty.noConfusion (genReal ht rfl)
  | v_lam T0 E0 e0 => rcases genLam ht rfl with ⟨T₂, hT⟩; exact Ty.noConfusion hT

theorem canon_real {v E} (hv : IsValue v) (ht : HasTy [] v .treal E) :
    ∃ z, v = .lit_real z := by
  cases hv with
  | v_real z => exact ⟨z, rfl⟩
  | v_nat n  => exact Ty.noConfusion (genNat ht rfl)
  | v_lam T0 E0 e0 => rcases genLam ht rfl with ⟨T₂, hT⟩; exact Ty.noConfusion hT
  | @v_kraw w m a hp => rcases genKraw ht rfl with ⟨T', N, hT⟩; exact Ty.noConfusion hT

theorem invLam {Γ e T E} (h : HasTy Γ e T E) {S F b} (he : e = .lam S F b) :
    ∃ T₂, T = .tarrow S F T₂ ∧ HasTy (S :: Γ) b T₂ F := by
  induction h with
  | t_lam Γ' T₁ T₂ E' body hb ihb =>
    injection he with h1 h2 h3; subst h1; subst h2; subst h3; exact ⟨T₂, rfl, hb⟩
  | t_sub Γ' e' T' E1 E2 h0 hsub ih => exact ih he
  | _ => exact Expr.noConfusion he

-- ================================================================
-- §J. Progress
-- ================================================================

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
  | t_measure Γ T e c cf s he _ _ ihe =>
    subst hΓ
    rcases ihe rfl with hv | ⟨e', he'⟩
    · exact Or.inr ⟨_, .meas_red hv⟩
    · exact Or.inr ⟨_, .meas_arg he'⟩
  | t_certain Γ T e he ihe =>
    subst hΓ
    rcases ihe rfl with hv | ⟨e', he'⟩
    · exact Or.inr ⟨_, .cert_red hv⟩
    · exact Or.inr ⟨_, .cert_arg he'⟩
  | t_opaque Γ T N E e he ihe =>
    subst hΓ
    rcases ihe rfl with hv | ⟨e', he'⟩
    · rcases canon_know hv he with ⟨w, m, a, rfl⟩
      exact Or.inr ⟨_, .opaque_red⟩
    · exact Or.inr ⟨_, .opaque_arg he'⟩
  | t_kvalue Γ T N E e he ihe =>
    subst hΓ
    rcases ihe rfl with hv | ⟨e', he'⟩
    · rcases canon_know hv he with ⟨w, m, a, rfl⟩
      exact Or.inr ⟨w, .kvalue_red⟩
    · exact Or.inr ⟨.kvalue e', .kvalue_arg he'⟩
  | t_kunc Γ T N E e he ihe =>
    subst hΓ
    rcases ihe rfl with hv | ⟨e', he'⟩
    · rcases canon_know hv he with ⟨w, m, a, rfl⟩
      exact Or.inr ⟨.lit_real m.gumVar, .kunc_red⟩
    · exact Or.inr ⟨.kunc e', .kunc_arg he'⟩
  | t_kconf Γ T N E e he ihe =>
    subst hΓ
    rcases ihe rfl with hv | ⟨e', he'⟩
    · rcases canon_know hv he with ⟨w, m, a, rfl⟩
      exact Or.inr ⟨.lit_real m.conf, .kconf_red⟩
    · exact Or.inr ⟨.kconf e', .kconf_arg he'⟩
  | t_kadd Γ Na Nb E₁ E₂ a b ha hb _ iha ihb =>
    subst hΓ
    rcases iha rfl with hva | ⟨a', ha'⟩
    · rcases canon_know hva ha with ⟨wa, ma, fa, rfl⟩
      rcases invKraw ha rfl with ⟨Ta, Na', hTa, hwaty, hwaval, _, _⟩
      injection hTa with hTa' hNa'; subst hTa'
      rcases canon_real hwaval hwaty with ⟨x, rfl⟩
      rcases ihb rfl with hvb | ⟨b', hb'⟩
      · rcases canon_know hvb hb with ⟨wb, mb, fb, rfl⟩
        rcases invKraw hb rfl with ⟨Tb, Nb', hTb, hwbty, hwbval, _, _⟩
        injection hTb with hTb' hNb'; subst hTb'
        rcases canon_real hwbval hwbty with ⟨y, rfl⟩
        exact Or.inr ⟨_, .kadd_red⟩
      · exact Or.inr ⟨_, .kadd_r (.v_kraw ma fa (.v_real x)) hb'⟩
    · exact Or.inr ⟨_, .kadd_l ha'⟩
  | t_kmul Γ Na Nb E₁ E₂ a b ha hb _ iha ihb =>
    subst hΓ
    rcases iha rfl with hva | ⟨a', ha'⟩
    · rcases canon_know hva ha with ⟨wa, ma, fa, rfl⟩
      rcases invKraw ha rfl with ⟨Ta, Na', hTa, hwaty, hwaval, _, _⟩
      injection hTa with hTa' hNa'; subst hTa'
      rcases canon_real hwaval hwaty with ⟨x, rfl⟩
      rcases ihb rfl with hvb | ⟨b', hb'⟩
      · rcases canon_know hvb hb with ⟨wb, mb, fb, rfl⟩
        rcases invKraw hb rfl with ⟨Tb, Nb', hTb, hwbty, hwbval, _, _⟩
        injection hTb with hTb' hNb'; subst hTb'
        rcases canon_real hwbval hwbty with ⟨y, rfl⟩
        exact Or.inr ⟨_, .kmul_red⟩
      · exact Or.inr ⟨_, .kmul_r (.v_kraw ma fa (.v_real x)) hb'⟩
    · exact Or.inr ⟨_, .kmul_l ha'⟩
  | t_let Γ T₁ T₂ E₁ E₂ e body he hbody ihe _ =>
    subst hΓ
    rcases ihe rfl with hv | ⟨e', he'⟩
    · exact Or.inr ⟨subst 0 e body, .let_red hv⟩
    · exact Or.inr ⟨.letE e' body, .let_step he'⟩
  | t_kraw Γ T N v m a hv hval hk hc _ => exact Or.inl (.v_kraw m a hval)
  | t_sub Γ e T E E' h0 hsub ih => exact ih hΓ

/-- **Progress** for the NS calculus. -/
theorem progress {e T E} (ht : HasTy [] e T E) : IsValue e ∨ ∃ e', e ⇒ e' :=
  progress' ht rfl

-- ================================================================
-- §K. Substitution infrastructure (ported from V2)
-- ================================================================

theorem shift_value {v} (hv : IsValue v) : ∀ c d, IsValue (shift c d v) := by
  induction hv with
  | v_nat n => intro c d; exact .v_nat n
  | v_real z => intro c d; exact .v_real z
  | v_lam T E e => intro c d; exact .v_lam _ _ _
  | v_kraw m a hp ih => intro c d; exact .v_kraw m a (ih c d)

theorem subst_value {w} (hw : IsValue w) : ∀ n u, IsValue (subst n u w) := by
  induction hw with
  | v_nat n => intro n' u; exact .v_nat n
  | v_real z => intro n' u; exact .v_real z
  | v_lam T E e => intro n' u; exact .v_lam _ _ _
  | v_kraw m a hp ih => intro n' u; exact .v_kraw m a (ih n' u)

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
  | t_measure Γ T e cc cf s _ _ _ ih => intro c d hc; simp [shift, ih c d hc]
  | t_certain Γ T e _ ih => intro c d hc; simp [shift, ih c d hc]
  | t_opaque Γ T N E e _ ih => intro c d hc; simp [shift, ih c d hc]
  | t_kvalue Γ T N E e _ ih => intro c d hc; simp [shift, ih c d hc]
  | t_kunc Γ T N E e _ ih => intro c d hc; simp [shift, ih c d hc]
  | t_kconf Γ T N E e _ ih => intro c d hc; simp [shift, ih c d hc]
  | t_kadd Γ Na Nb E₁ E₂ a b _ _ _ iha ihb => intro c d hc; simp [shift, iha c d hc, ihb c d hc]
  | t_kmul Γ Na Nb E₁ E₂ a b _ _ _ iha ihb => intro c d hc; simp [shift, iha c d hc, ihb c d hc]
  | t_let Γ T₁ T₂ E₁ E₂ e body _ _ ihe ihb =>
    intro c d hc
    have hb : (T₁ :: Γ).length ≤ c + 1 := by simp [List.length] at hc ⊢; omega
    simp [shift, ihe c d hc, ihb (c+1) d hb]
  | t_kraw Γ T N v m a _ _ _ _ ih => intro c d hc; simp [shift, ih c d hc]
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
  | t_measure Γ T e cc cf s _ h1 h2 ih =>
    intro k τ; exact .t_measure _ _ _ _ _ _ (ih k τ) h1 h2
  | t_certain Γ T e _ ih => intro k τ; exact .t_certain _ _ _ (ih k τ)
  | t_opaque Γ T N E e _ ih => intro k τ; exact .t_opaque _ _ _ _ _ (ih k τ)
  | t_kvalue Γ T N E e _ ih => intro k τ; exact .t_kvalue _ _ _ _ _ (ih k τ)
  | t_kunc Γ T N E e _ ih => intro k τ; exact .t_kunc _ _ _ _ _ (ih k τ)
  | t_kconf Γ T N E e _ ih => intro k τ; exact .t_kconf _ _ _ _ _ (ih k τ)
  | t_kadd Γ Na Nb E₁ E₂ a b _ _ hd iha ihb =>
    intro k τ; exact .t_kadd _ _ _ _ _ _ _ (iha k τ) (ihb k τ) hd
  | t_kmul Γ Na Nb E₁ E₂ a b _ _ hd iha ihb =>
    intro k τ; exact .t_kmul _ _ _ _ _ _ _ (iha k τ) (ihb k τ) hd
  | t_let Γ T₁ T₂ E₁ E₂ e body _ _ ihe ihb =>
    intro k τ
    have : shift k 1 (.letE e body) = .letE (shift k 1 e) (shift (k+1) 1 body) := by simp [shift]
    rw [this]
    have hctx : (T₁ :: Γ).insertIdx (k+1) τ = T₁ :: Γ.insertIdx k τ := by simp [List.insertIdx]
    have hb := ihb (k+1) τ; rw [hctx] at hb
    exact .t_let _ _ _ _ _ _ _ (ihe k τ) hb
  | t_kraw Γ T N v m a _ hval hk hc ih =>
    intro k τ
    have : shift k 1 (.kraw v m a) = .kraw (shift k 1 v) m a := by simp [shift]
    rw [this]; exact .t_kraw _ _ _ _ _ _ (ih k τ) (shift_value hval k 1) hk hc
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

theorem value_emptyE {Γ v T E} (hv : IsValue v) (h : HasTy Γ v T E) :
    HasTy Γ v T emptyE := by
  cases hv with
  | v_nat n => rw [genNat h rfl]; exact .t_lit_nat _ _
  | v_real z => rw [genReal h rfl]; exact .t_lit_real _ _
  | v_lam S F b => rcases invLam h rfl with ⟨T₂, hT, hb⟩; subst hT; exact .t_lam _ _ _ _ _ hb
  | @v_kraw w m a hp =>
    rcases invKraw h rfl with ⟨T', N, hT, hwty, hwval, hk, hc⟩; subst hT
    exact .t_kraw _ _ _ _ _ _ hwty hwval hk hc

theorem substClosed {v σ} (hv : HasTy [] v σ emptyE) {Γ0 e T E} (h : HasTy Γ0 e T E) :
    ∀ Δ, Γ0 = Δ ++ σ :: [] → HasTy Δ (subst Δ.length v e) T E := by
  induction h with
  | t_lit_nat Γ' n => intro Δ hΓ; exact .t_lit_nat _ _
  | t_lit_real Γ' z => intro Δ hΓ; exact .t_lit_real _ _
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
    have key : subst Δ.length v (.lam T₁ E' body)
        = .lam T₁ E' (subst (Δ.length + 1) v body) := by simp [subst, hsh]
    rw [key]; exact .t_lam _ _ _ _ _ (ihb (T₁ :: Δ) rfl)
  | t_app Γ' T₁ T₂ Ef Ec Ecaller f a hf ha hEf hEc ihf iha =>
    intro Δ hΓ; subst hΓ
    exact .t_app _ _ _ _ _ _ _ _ (ihf Δ rfl) (iha Δ rfl) hEf hEc
  | t_measure Γ' T e cc cf s he h1 h2 ih =>
    intro Δ hΓ; subst hΓ; exact .t_measure _ _ _ _ _ _ (ih Δ rfl) h1 h2
  | t_certain Γ' T e _ ih => intro Δ hΓ; subst hΓ; exact .t_certain _ _ _ (ih Δ rfl)
  | t_opaque Γ' T N E e _ ih => intro Δ hΓ; subst hΓ; exact .t_opaque _ _ _ _ _ (ih Δ rfl)
  | t_kvalue Γ' T N E e _ ih => intro Δ hΓ; subst hΓ; exact .t_kvalue _ _ _ _ _ (ih Δ rfl)
  | t_kunc Γ' T N E e _ ih => intro Δ hΓ; subst hΓ; exact .t_kunc _ _ _ _ _ (ih Δ rfl)
  | t_kconf Γ' T N E e _ ih => intro Δ hΓ; subst hΓ; exact .t_kconf _ _ _ _ _ (ih Δ rfl)
  | t_kadd Γ' Na Nb E₁ E₂ a b _ _ hd iha ihb =>
    intro Δ hΓ; subst hΓ; exact .t_kadd _ _ _ _ _ _ _ (iha Δ rfl) (ihb Δ rfl) hd
  | t_kmul Γ' Na Nb E₁ E₂ a b _ _ hd iha ihb =>
    intro Δ hΓ; subst hΓ; exact .t_kmul _ _ _ _ _ _ _ (iha Δ rfl) (ihb Δ rfl) hd
  | t_let Γ' T₁ T₂ E₁ E₂ e body _ _ ihe ihb =>
    intro Δ hΓ; subst hΓ
    have hsh : shift 0 1 v = v := wellScoped hv 0 1 (by simp [List.length])
    have key : subst Δ.length v (.letE e body)
        = .letE (subst Δ.length v e) (subst (Δ.length + 1) v body) := by simp [subst, hsh]
    rw [key]; exact .t_let _ _ _ _ _ _ _ (ihe Δ rfl) (ihb (T₁ :: Δ) rfl)
  | t_kraw Γ' T N w m a hw hval hk hc ih =>
    intro Δ hΓ; subst hΓ
    have key : subst Δ.length v (.kraw w m a) = .kraw (subst Δ.length v w) m a := by simp [subst]
    rw [key]; exact .t_kraw _ _ _ _ _ _ (ih Δ rfl) (subst_value hval Δ.length v) hk hc
  | t_sub Γ' e T E E' h0 hsub ih => intro Δ hΓ; exact .t_sub _ _ _ _ _ (ih Δ hΓ) hsub

-- ================================================================
-- §L. Preservation for the NS calculus (the `[pending wire]` row of Paper A §6.4)
-- ================================================================

theorem preservation' {Γ e T E} (h : HasTy Γ e T E) :
    ∀ {e'}, e ⇒ e' → Γ = [] → HasTy Γ e' T E := by
  induction h with
  | t_lit_nat Γ n => intro e' hs hΓ; cases hs
  | t_lit_real Γ z => intro e' hs hΓ; cases hs
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
  | t_measure Γ T e c cf s he h1 h2 ihe =>
    intro e' hs hΓ; subst hΓ
    cases hs with
    | meas_red hv =>
      have hk : kvalid ⟨c * c, cf⟩ := ⟨int_sq_nonneg c, h1, h2⟩
      exact .t_sub _ _ _ _ _ (.t_kraw _ _ _ _ _ _ he hv hk (covers_single s c))
        (Sounio.EpistemicEffects.emptyE_sub _)
    | meas_arg he' => exact .t_measure _ _ _ _ _ _ (ihe he' rfl) h1 h2
  | t_certain Γ T e he ihe =>
    intro e' hs hΓ; subst hΓ
    cases hs with
    | cert_red hv =>
      have hk : kvalid ⟨0, 1000⟩ := ⟨Int.le_refl 0, by decide, by decide⟩
      exact .t_kraw _ _ _ _ _ _ he hv hk covers_empty
    | cert_arg he' => exact .t_certain _ _ _ (ihe he' rfl)
  | t_opaque Γ T N E e he ihe =>
    intro e' hs hΓ; subst hΓ
    cases hs with
    | opaque_red =>
      rcases invKraw he rfl with ⟨T', N', hT, hwty, hwval, hk, _⟩
      injection hT with hT1 hT2; subst hT1
      exact .t_sub _ _ _ _ _ (.t_kraw _ _ _ _ _ _ hwty hwval hk (covers_top _))
        (Sounio.EpistemicEffects.emptyE_sub _)
    | opaque_arg he' => exact .t_opaque _ _ _ _ _ (ihe he' rfl)
  | t_kvalue Γ T N E e he ihe =>
    intro e' hs hΓ; subst hΓ
    cases hs with
    | kvalue_red =>
      rcases invKraw he rfl with ⟨T', N', hT, hwty, _, _, _⟩
      injection hT with hT' hN'; subst hT'
      exact .t_sub _ _ _ _ _ hwty (Sounio.EpistemicEffects.emptyE_sub _)
    | kvalue_arg he' => exact .t_kvalue _ _ _ _ _ (ihe he' rfl)
  | t_kunc Γ T N E e he ihe =>
    intro e' hs hΓ; subst hΓ
    cases hs with
    | kunc_red => exact .t_sub _ _ _ _ _ (.t_lit_real _ _) (Sounio.EpistemicEffects.emptyE_sub _)
    | kunc_arg he' => exact .t_kunc _ _ _ _ _ (ihe he' rfl)
  | t_kconf Γ T N E e he ihe =>
    intro e' hs hΓ; subst hΓ
    cases hs with
    | kconf_red => exact .t_sub _ _ _ _ _ (.t_lit_real _ _) (Sounio.EpistemicEffects.emptyE_sub _)
    | kconf_arg he' => exact .t_kconf _ _ _ _ _ (ihe he' rfl)
  | t_kadd Γ Na Nb E₁ E₂ a b ha hb hd iha ihb =>
    intro e' hs hΓ; subst hΓ
    cases hs with
    | kadd_red =>
      rcases invKraw ha rfl with ⟨Ta, Na', hTa, _, _, hka, hca⟩
      rcases invKraw hb rfl with ⟨Tb, Nb', hTb, _, _, hkb, hcb⟩
      injection hTa with hTa1 hTa2; subst hTa2
      injection hTb with hTb1 hTb2; subst hTb2
      exact .t_sub _ _ _ _ _
        (.t_kraw _ _ _ _ _ _ (.t_lit_real _ _) (.v_real _) (gAddMeta_valid hka hkb)
          (covers_union hca hcb))
        (Sounio.EpistemicEffects.emptyE_sub _)
    | kadd_l ha' => exact .t_kadd _ _ _ _ _ _ _ (iha ha' rfl) hb hd
    | kadd_r _ hb' => exact .t_kadd _ _ _ _ _ _ _ ha (ihb hb' rfl) hd
  | t_kmul Γ Na Nb E₁ E₂ a b ha hb hd iha ihb =>
    intro e' hs hΓ; subst hΓ
    cases hs with
    | kmul_red =>
      rcases invKraw ha rfl with ⟨Ta, Na', hTa, _, _, hka, hca⟩
      rcases invKraw hb rfl with ⟨Tb, Nb', hTb, _, _, hkb, hcb⟩
      injection hTa with hTa1 hTa2; subst hTa2
      injection hTb with hTb1 hTb2; subst hTb2
      exact .t_sub _ _ _ _ _
        (.t_kraw _ _ _ _ _ _ (.t_lit_real _ _) (.v_real _) (gMulMeta_valid hka hkb)
          (covers_union (covers_scale _ hca) (covers_scale _ hcb)))
        (Sounio.EpistemicEffects.emptyE_sub _)
    | kmul_l ha' => exact .t_kmul _ _ _ _ _ _ _ (iha ha' rfl) hb hd
    | kmul_r _ hb' => exact .t_kmul _ _ _ _ _ _ _ ha (ihb hb' rfl) hd
  | t_let Γ T₁ T₂ E₁ E₂ e body he hbody ihe ihb =>
    intro e' hs hΓ; subst hΓ
    cases hs with
    | let_red hv =>
      exact .t_sub _ _ _ _ _ (substClosed (value_emptyE hv he) hbody [] rfl)
        (Sounio.EpistemicEffects.sub_union_right E₁ E₂)
    | let_step he' => exact .t_let _ _ _ _ _ _ _ (ihe he' rfl) hbody
  | t_kraw Γ T N v m a hv hval hk hc ih => intro e' hs hΓ; cases hs
  | t_sub Γ e T E E' h0 hsub ih => intro e' hs hΓ; exact .t_sub _ _ _ _ _ (ih hs hΓ) hsub

/-- **Preservation (NS-extended subject reduction).** Typing — including the noise-set
    annotation `N` and, through `t_kraw`, the support over-approximation — is preserved. -/
theorem preservation {e T E} (h : HasTy [] e T E) {e'} (hs : e ⇒ e') : HasTy [] e' T E :=
  preservation' h hs rfl

-- ================================================================
-- §M. Exactness is preserved under NS typing — the soundness theorem
-- ================================================================

theorem exact_shift {e} (h : Exact e) : ∀ c d, Exact (shift c d e) := by
  induction e with
  | var k =>
    intro c d; simp only [shift]
    split
    · trivial
    · split <;> trivial
  | lit_nat n => intro c d; trivial
  | lit_real z => intro c d; trivial
  | lam T E b ih => intro c d; exact ih h (c+1) d
  | app f a ihf iha => intro c d; exact ⟨ihf h.1 c d, iha h.2 c d⟩
  | measure e cc cf s ih => intro c d; exact ih h c d
  | certain e ih => intro c d; exact ih h c d
  | «opaque» e ih => intro c d; exact ih h c d
  | kvalue e ih => intro c d; exact ih h c d
  | kunc e ih => intro c d; exact ih h c d
  | kconf e ih => intro c d; exact ih h c d
  | kadd a b iha ihb => intro c d; exact ⟨iha h.1 c d, ihb h.2 c d⟩
  | kmul a b iha ihb => intro c d; exact ⟨iha h.1 c d, ihb h.2 c d⟩
  | letE e b ihe ihb => intro c d; exact ⟨ihe h.1 c d, ihb h.2 (c+1) d⟩
  | kraw v m a ih => intro c d; exact ⟨ih h.1 c d, h.2⟩

theorem exact_subst {e} (h : Exact e) : ∀ n w, Exact w → Exact (subst n w e) := by
  induction e with
  | var k =>
    intro n w hw; simp only [subst]
    split
    · exact hw
    · split <;> trivial
  | lit_nat n => intro n w hw; trivial
  | lit_real z => intro n w hw; trivial
  | lam T E b ih => intro n w hw; exact ih h (n+1) _ (exact_shift hw 0 1)
  | app f a ihf iha => intro n w hw; exact ⟨ihf h.1 n w hw, iha h.2 n w hw⟩
  | measure e cc cf s ih => intro n w hw; exact ih h n w hw
  | certain e ih => intro n w hw; exact ih h n w hw
  | «opaque» e ih => intro n w hw; exact ih h n w hw
  | kvalue e ih => intro n w hw; exact ih h n w hw
  | kunc e ih => intro n w hw; exact ih h n w hw
  | kconf e ih => intro n w hw; exact ih h n w hw
  | kadd a b iha ihb => intro n w hw; exact ⟨iha h.1 n w hw, ihb h.2 n w hw⟩
  | kmul a b iha ihb => intro n w hw; exact ⟨iha h.1 n w hw, ihb h.2 n w hw⟩
  | letE e b ihe ihb => intro n w hw; exact ⟨ihe h.1 n w hw, ihb h.2 (n+1) _ (exact_shift hw 0 1)⟩
  | kraw v m a ih => intro n w hw; exact ⟨ih h.1 n w hw, h.2⟩

/-- A single measurement is exact: reported `c²` = true `⟨c·ε_s, c·ε_s⟩`. -/
theorem trueVar_single (s : Nat) (c : Int) : trueVar [(s, c)] = c * c := by
  simp [trueVar, inner, coeff]

/-- **Exactness preservation.** Under NS typing, if every Knowledge value in `e` reports
    its true variance, so does every Knowledge value after a step — *including* the one
    produced by the defective `gAddMeta`/`gMulMeta`, because the disjointness premise
    (via Lemma 2) zeroes the covariance term of Lemma 1. -/
theorem exact_preservation' {Γ e T E} (h : HasTy Γ e T E) :
    ∀ {e'}, e ⇒ e' → Γ = [] → Exact e → Exact e' := by
  induction h with
  | t_lit_nat Γ n => intro e' hs hΓ hx; cases hs
  | t_lit_real Γ z => intro e' hs hΓ hx; cases hs
  | t_var Γ n T hlk => intro e' hs hΓ hx; cases hs
  | t_lam Γ T₁ T₂ E body _ => intro e' hs hΓ hx; cases hs
  | t_app Γ T₁ T₂ Ef Ec Ecaller f a hf ha hEf hEc ihf iha =>
    intro e' hs hΓ hx; subst hΓ
    cases hs with
    | @beta v T E body hvarg => exact exact_subst (e := body) hx.1 0 _ hx.2
    | app_l hf' => exact ⟨ihf hf' rfl hx.1, hx.2⟩
    | app_r _ ha' => exact ⟨hx.1, iha ha' rfl hx.2⟩
  | t_measure Γ T e c cf s he h1 h2 ihe =>
    intro e' hs hΓ hx; subst hΓ
    cases hs with
    | meas_red hv => exact ⟨hx, (trueVar_single s c).symm⟩
    | meas_arg he' => have hh := ihe he' rfl hx; exact hh
  | t_certain Γ T e he ihe =>
    intro e' hs hΓ hx; subst hΓ
    cases hs with
    | cert_red hv => exact ⟨hx, rfl⟩
    | cert_arg he' => have hh := ihe he' rfl hx; exact hh
  | t_opaque Γ T N E e he ihe =>
    intro e' hs hΓ hx; subst hΓ
    cases hs with
    | opaque_red => exact hx
    | opaque_arg he' => have hh := ihe he' rfl hx; exact hh
  | t_kvalue Γ T N E e he ihe =>
    intro e' hs hΓ hx; subst hΓ
    cases hs with
    | kvalue_red => exact hx.1
    | kvalue_arg he' => have hh := ihe he' rfl hx; exact hh
  | t_kunc Γ T N E e he ihe =>
    intro e' hs hΓ hx; subst hΓ
    cases hs with
    | kunc_red => trivial
    | kunc_arg he' => have hh := ihe he' rfl hx; exact hh
  | t_kconf Γ T N E e he ihe =>
    intro e' hs hΓ hx; subst hΓ
    cases hs with
    | kconf_red => trivial
    | kconf_arg he' => have hh := ihe he' rfl hx; exact hh
  | t_kadd Γ Na Nb E₁ E₂ a b ha hb hd iha ihb =>
    intro e' hs hΓ hx; subst hΓ
    cases hs with
    | kadd_red =>
      rcases invKraw ha rfl with ⟨Ta, Na', hTa, _, _, _, hca⟩
      rcases invKraw hb rfl with ⟨Tb, Nb', hTb, _, _, _, hcb⟩
      injection hTa with hTa1 hTa2; subst hTa2
      injection hTb with hTb1 hTb2; subst hTb2
      have hzero : inner _ _ = 0 := inner_zero_of_ns hca hcb hd
      refine ⟨trivial, ?_⟩
      show _ + _ = trueVar (_ ++ _)
      rw [trueVar_append, hzero, hx.1.2, hx.2.2]; omega
    | kadd_l ha' => exact ⟨iha ha' rfl hx.1, hx.2⟩
    | kadd_r _ hb' => exact ⟨hx.1, ihb hb' rfl hx.2⟩
  | t_kmul Γ Na Nb E₁ E₂ a b ha hb hd iha ihb =>
    intro e' hs hΓ hx; subst hΓ
    cases hs with
    | kmul_red =>
      rcases invKraw ha rfl with ⟨Ta, Na', hTa, _, _, _, hca⟩
      rcases invKraw hb rfl with ⟨Tb, Nb', hTb, _, _, _, hcb⟩
      injection hTa with hTa1 hTa2; subst hTa2
      injection hTb with hTb1 hTb2; subst hTb2
      have hzero : inner _ _ = 0 := inner_zero_of_ns hca hcb hd
      refine ⟨trivial, ?_⟩
      show _ * _ * _ + _ * _ * _ = trueVar (scale _ _ ++ scale _ _)
      rw [trueVar_mul, hzero, hx.1.2, hx.2.2]; simp
    | kmul_l ha' => exact ⟨iha ha' rfl hx.1, hx.2⟩
    | kmul_r _ hb' => exact ⟨hx.1, ihb hb' rfl hx.2⟩
  | t_let Γ T₁ T₂ E₁ E₂ e body he hbody ihe ihb =>
    intro e' hs hΓ hx; subst hΓ
    cases hs with
    | let_red hv => exact exact_subst hx.2 0 _ hx.1
    | let_step he' => exact ⟨ihe he' rfl hx.1, hx.2⟩
  | t_kraw Γ T N v m a hv hval hk hc ih => intro e' hs hΓ hx; cases hs
  | t_sub Γ e T E E' h0 hsub ih => intro e' hs hΓ hx; exact ih hs hΓ hx

theorem exact_preservation {e T E} (h : HasTy [] e T E) {e'} (hs : e ⇒ e') (hx : Exact e) :
    Exact e' :=
  exact_preservation' h hs rfl hx

-- ================================================================
-- §N. Theorem 6.4 — no first-order anti-garbling at any reached operator
-- ================================================================

/-- Every independence-assuming operator inside a well-typed term whose operands are
    runtime values has zero-covariance operands. Pure consequence of the typing premise
    `nsDisjoint` + Lemma 2 + `inner_disjoint`; no reduction involved. -/
theorem typed_agfree {Γ e T E} (h : HasTy Γ e T E) : AGFree e := by
  induction h with
  | t_lit_nat Γ n => trivial
  | t_lit_real Γ z => trivial
  | t_var Γ n T hlk => trivial
  | t_lam Γ T₁ T₂ E body _ ih => exact ih
  | t_app Γ T₁ T₂ Ef Ec Ecaller f a _ _ _ _ ihf iha => exact ⟨ihf, iha⟩
  | t_measure Γ T e c cf s _ _ _ ih => exact ih
  | t_certain Γ T e _ ih => exact ih
  | t_opaque Γ T N E e _ ih => exact ih
  | t_kvalue Γ T N E e _ ih => exact ih
  | t_kunc Γ T N E e _ ih => exact ih
  | t_kconf Γ T N E e _ ih => exact ih
  | t_kadd Γ Na Nb E₁ E₂ a b ha hb hd iha ihb =>
    refine ⟨?_, iha, ihb⟩
    intro x ma a' y mb b' hea heb; subst hea; subst heb
    rcases invKraw ha rfl with ⟨Ta, Na', hTa, _, _, _, hca⟩
    rcases invKraw hb rfl with ⟨Tb, Nb', hTb, _, _, _, hcb⟩
    injection hTa with hTa1 hTa2; subst hTa2
    injection hTb with hTb1 hTb2; subst hTb2
    exact inner_zero_of_ns hca hcb hd
  | t_kmul Γ Na Nb E₁ E₂ a b ha hb hd iha ihb =>
    refine ⟨?_, iha, ihb⟩
    intro x ma a' y mb b' hea heb; subst hea; subst heb
    rcases invKraw ha rfl with ⟨Ta, Na', hTa, _, _, _, hca⟩
    rcases invKraw hb rfl with ⟨Tb, Nb', hTb, _, _, _, hcb⟩
    injection hTa with hTa1 hTa2; subst hTa2
    injection hTb with hTb1 hTb2; subst hTb2
    exact inner_zero_of_ns hca hcb hd
  | t_let Γ T₁ T₂ E₁ E₂ e body _ _ ihe ihb => exact ⟨ihe, ihb⟩
  | t_kraw Γ T N v m a _ _ _ _ ih => exact ih
  | t_sub Γ e T E E' _ _ ih => exact ih

/-- **Theorem (no first-order anti-garbling), Paper A §6.4, mechanized.**
    Along every evaluation of a closed, well-typed, initially-exact program: typing is
    preserved (with the noise-sets), every Knowledge value reports its true variance, and
    no reached independence-assuming operator has correlated operands. -/
theorem soundness_star' {e e'} (hs : e ⇒* e') :
    ∀ {T E}, HasTy [] e T E → Exact e → HasTy [] e' T E ∧ Exact e' ∧ AGFree e' := by
  induction hs with
  | refl e => intro T E h hx; exact ⟨h, hx, typed_agfree h⟩
  | step s1 _ ih => intro T E h hx; exact ih (preservation h s1) (exact_preservation h s1 hx)

theorem soundness_star {e T E} (h : HasTy [] e T E) (hx : Exact e) {e'} (hs : e ⇒* e') :
    HasTy [] e' T E ∧ Exact e' ∧ AGFree e' :=
  soundness_star' hs h hx

-- ================================================================
-- §O. The sabotage witness, kernel-checked (Paper A §8.2 controls, in the calculus)
-- ================================================================

/-- `x`: one measurement, value 10, standard uncertainty 1, source 0. -/
def xk : Expr := .kraw (.lit_real 10) ⟨1, 1000⟩ [(0, 1)]
/-- `y`: an independent measurement on source 1. -/
def yk : Expr := .kraw (.lit_real 20) ⟨4, 1000⟩ [(1, 2)]
/-- `mx`: the SOURCE-LEVEL measurement term (not yet reduced) on source 0. -/
def mx : Expr := .measure (.lit_real 10) 1 1000 0

theorem xk_exact : Exact xk := by simp [xk, Exact, trueVar, inner, coeff]
theorem yk_exact : Exact yk := by simp [yk, Exact, trueVar, inner, coeff]

/-- The defective semantics DOES step `x + x` (it never asks about sources)… -/
theorem x_plus_x_steps :
    (.kadd xk xk) ⇒ (.kraw (.lit_real 20) ⟨2, 1000⟩ [(0, 1), (0, 1)]) := Step.kadd_red

/-- …and the result UNDER-states the variance: reported 2, true 4. Anti-garbling. -/
theorem x_plus_x_understates :
    ¬ Exact (.kraw (.lit_real 20) ⟨2, 1000⟩ [(0, 1), (0, 1)]) := by
  simp [Exact, trueVar, inner, coeff]

/-- The gap is exactly `2⟨x,x⟩` (Lemma 1 instance). -/
theorem x_plus_x_gap :
    trueVar ([(0, 1)] ++ [(0, 1)]) - (gAddMeta ⟨1, 1000⟩ ⟨1, 1000⟩).gumVar
      = 2 * inner [(0, 1)] [(0, 1)] := by decide

/-- **E230 in the kernel.** `x + x` is untypable under NS — for EVERY choice of the
    operands' annotations `Nₐ, N_b` (finite or ⊤): both must cover source 0, so they are
    never disjoint. The naive rule (no premise) would accept it; the exact term it
    produces is `x_plus_x_understates`. -/
theorem x_plus_x_untypable' {Γ e T E} (h : HasTy Γ e T E) (he : e = .kadd xk xk) : False := by
  induction h with
  | t_kadd Γ Na Nb E₁ E₂ a b ha hb hd _ _ =>
    injection he with hea heb; subst hea; subst heb
    have ma : nsMem 0 Na := support_over_approx ha (0, 1) (by simp [xk])
    have mb : nsMem 0 Nb := support_over_approx hb (0, 1) (by simp [xk])
    have := nsDisjoint_of_shared ma mb
    rw [this] at hd; exact Bool.noConfusion hd
  | t_sub _ _ _ _ _ _ _ ih => exact ih he
  | _ => exact Expr.noConfusion he

/-- **E230 in the kernel** — closed statement: no context, type, or effect row types `x + x`. -/
theorem x_plus_x_untypable : ∀ Γ T E, ¬ HasTy Γ (.kadd xk xk) T E :=
  fun _ _ _ h => x_plus_x_untypable' h rfl

theorem invOpaque {Γ e T E} (h : HasTy Γ e T E) {u} (he : e = .opaque u) :
    ∃ T', T = .tknow T' nsTop := by
  induction h with
  | t_opaque Γ' T' N E' e' _ _ => exact ⟨T', rfl⟩
  | t_sub _ _ _ _ _ _ _ ih => exact ih he
  | _ => exact Expr.noConfusion he

/-- **The ⊤ clause in isolation.** `x + opaque(y)`: `y` is on source 1, DISJOINT from `x`'s
    source 0 — `x + y` itself is admitted (`x_plus_y_typable`) — but `opaque` erases `y`'s
    provenance to ⊤, and ⊤ is never disjoint. Rejected purely by the ⊤ clause of
    (Add-Indep). (Grok 4.6 review 2026-08-31, items 7/9.) -/
theorem x_plus_top_untypable' {Γ e T E} (h : HasTy Γ e T E) (he : e = .kadd xk (.opaque yk)) :
    False := by
  induction h with
  | t_kadd Γ Na Nb E₁ E₂ a b ha hb hd _ _ =>
    injection he with hea heb; subst hea; subst heb
    rcases invOpaque hb rfl with ⟨Tb, hTb⟩
    injection hTb with hTb1 hTb2; subst hTb2
    have hf : nsDisjoint Na nsTop = false := nsDisjoint_top_right Na
    rw [hf] at hd; exact Bool.noConfusion hd
  | t_sub _ _ _ _ _ _ _ ih => exact ih he
  | _ => exact Expr.noConfusion he

theorem x_plus_top_untypable : ∀ Γ T E, ¬ HasTy Γ (.kadd xk (.opaque yk)) T E :=
  fun _ _ _ h => x_plus_top_untypable' h rfl

/-- …while `opaque(y)` alone is perfectly well-typed (at ⊤), so the rejection is the sum's. -/
theorem opaque_y_typable : HasTy [] (.opaque yk) (.tknow .treal nsTop) emptyE :=
  .t_opaque _ _ _ _ _ (.t_kraw _ _ _ _ _ _ (.t_lit_real _ _) (.v_real _) ⟨by decide, by decide, by decide⟩ (covers_single 1 2))

theorem invMeasure {Γ e T E} (h : HasTy Γ e T E) {u c cf s} (he : e = .measure u c cf s) :
    ∃ T', T = .tknow T' (nsSingle s) := by
  induction h with
  | t_measure Γ' T' e' c' cf' s' _ _ _ _ =>
    injection he with h1 h2 h3 h4; subst h4; exact ⟨T', rfl⟩
  | t_sub _ _ _ _ _ _ _ ih => exact ih he
  | _ => exact Expr.noConfusion he

/-- **Source-level E230.** `measure(·, s) + measure(·, s)` — the unreduced program text, not a
    runtime value — is untypable: both operands are forced to `{s}` by (Measure), and
    `{s}` is not disjoint from itself. (Grok 4.6 review 2026-08-31, item 9.) -/
theorem measure_plus_measure_untypable' {Γ e T E} (h : HasTy Γ e T E) (he : e = .kadd mx mx) :
    False := by
  induction h with
  | t_kadd Γ Na Nb E₁ E₂ a b ha hb hd _ _ =>
    injection he with hea heb; subst hea; subst heb
    rcases invMeasure ha rfl with ⟨Ta, hTa⟩
    injection hTa with hTa1 hTa2; subst hTa2
    rcases invMeasure hb rfl with ⟨Tb, hTb⟩
    injection hTb with hTb1 hTb2; subst hTb2
    have hf : nsDisjoint (nsSingle 0) (nsSingle 0) = false := by decide
    rw [hf] at hd; exact Bool.noConfusion hd
  | t_sub _ _ _ _ _ _ _ ih => exact ih he
  | _ => exact Expr.noConfusion he

theorem measure_plus_measure_untypable : ∀ Γ T E, ¬ HasTy Γ (.kadd mx mx) T E :=
  fun _ _ _ h => measure_plus_measure_untypable' h rfl

theorem invVar {Γ e T E} (h : HasTy Γ e T E) {n} (he : e = .var n) : lookupCtx Γ n = some T := by
  induction h with
  | t_var Γ' n' T' hlk => injection he with h1; subst h1; exact hlk
  | t_sub _ _ _ _ _ _ _ ih => exact ih he
  | _ => exact Expr.noConfusion he

theorem invLet {Γ e T E} (h : HasTy Γ e T E) {u b} (he : e = .letE u b) :
    ∃ T₁ E₁ E₂, HasTy Γ u T₁ E₁ ∧ HasTy (T₁ :: Γ) b T E₂ := by
  induction h with
  | t_let Γ' T₁ T₂ E₁ E₂ e' body hu hb _ _ =>
    injection he with h1 h2; subst h1; subst h2; exact ⟨T₁, E₁, E₂, hu, hb⟩
  | t_sub _ _ _ _ _ _ _ ih => exact ih he
  | _ => exact Expr.noConfusion he

/-- The shared-VARIABLE body `x + x` (de Bruijn `var 0 + var 0`) is untypable whenever the
    bound variable carries a Knowledge type — both operands look up the SAME `N`, which is
    never disjoint from itself unless empty… and here it is `{0}`. -/
theorem var_plus_var_untypable' {Γ e T E} (h : HasTy Γ e T E)
    (he : e = .kadd (.var 0) (.var 0)) (hΓ : ∃ Γ' T', Γ = (.tknow T' (nsSingle 0)) :: Γ') : False := by
  induction h with
  | t_kadd Γ Na Nb E₁ E₂ a b ha hb hd _ _ =>
    injection he with hea heb; subst hea; subst heb
    rcases hΓ with ⟨Γ', T', rfl⟩
    have la := invVar ha rfl
    have lb := invVar hb rfl
    simp only [lookupCtx, Option.some.injEq] at la lb
    injection la with la1 la2; injection lb with lb1 lb2
    subst la2; subst lb2
    have hf : nsDisjoint (nsSingle 0) (nsSingle 0) = false := by decide
    rw [hf] at hd; exact Bool.noConfusion hd
  | t_sub _ _ _ _ _ _ _ ih => exact ih he hΓ
  | _ => exact Expr.noConfusion he

/-- **The §8.2 shared-variable control, at source level.** `let x = measure(·, s) in x + x`
    is untypable: (Measure) fixes `x : Knowledge⟨ℝ, {s}⟩`, both uses of `x` look up that
    same `{s}`, and `{s}` is not disjoint from itself. (Grok 4.6 round 3, items 9/10b.) -/
theorem let_x_plus_x_untypable : ∀ Γ T E, ¬ HasTy Γ (.letE mx (.kadd (.var 0) (.var 0))) T E := by
  intro Γ T E h
  rcases invLet h rfl with ⟨T₁, E₁, E₂, hu, hb⟩
  rcases invMeasure hu rfl with ⟨T', hT₁⟩; subst hT₁
  exact var_plus_var_untypable' hb rfl ⟨Γ, T', rfl⟩

/-- `x + y` on disjoint sources: admitted, with result set `{0} ∪ {1}`. -/
theorem x_plus_y_typable :
    HasTy [] (.kadd xk yk) (.tknow .treal (nsUnion (nsSingle 0) (nsSingle 1)))
      (unionE emptyE emptyE) := by
  apply HasTy.t_kadd
  · exact .t_kraw _ _ _ _ _ _ (.t_lit_real _ _) (.v_real _) ⟨by decide, by decide, by decide⟩ (covers_single 0 1)
  · exact .t_kraw _ _ _ _ _ _ (.t_lit_real _ _) (.v_real _) ⟨by decide, by decide, by decide⟩ (covers_single 1 2)
  · decide

/-- …and its (defective-combinator) result is EXACT: reported 1 + 4 = 5 = true variance. -/
theorem x_plus_y_exact :
    Exact (.kraw (.lit_real 30) (gAddMeta ⟨1, 1000⟩ ⟨4, 1000⟩) ([(0, 1)] ++ [(1, 2)])) := by
  simp [Exact, gAddMeta, trueVar, inner, coeff]

/-- The same for the product: `x · y` admitted, and `gMulMeta` is exact to first order. -/
theorem x_times_y_exact :
    Exact (.kraw (.lit_real 200) (gMulMeta 10 ⟨1, 1000⟩ 20 ⟨4, 1000⟩)
      (scale 20 [(0, 1)] ++ scale 10 [(1, 2)])) := by
  simp [Exact, gMulMeta, trueVar, inner, coeff, scale]

/-- The product on a SHARED source is not exact (`x · x`: reported 2·10²·1 = 200, true 400). -/
theorem x_times_x_understates :
    ¬ Exact (.kraw (.lit_real 100) (gMulMeta 10 ⟨1, 1000⟩ 10 ⟨1, 1000⟩)
      (scale 10 [(0, 1)] ++ scale 10 [(0, 1)])) := by
  simp [Exact, gMulMeta, trueVar, inner, coeff, scale]

end Sounio.EpistemicEffectsNS

-- ================================================================
-- Axiom footprint (reproduce: `lake env lean EpistemicEffectsNS.lean`)
-- ================================================================
#print axioms Sounio.EpistemicEffectsNS.trueVar_append
#print axioms Sounio.EpistemicEffectsNS.inner_zero_of_ns
#print axioms Sounio.EpistemicEffectsNS.covers_coeff
#print axioms Sounio.EpistemicEffectsNS.progress
#print axioms Sounio.EpistemicEffectsNS.preservation
#print axioms Sounio.EpistemicEffectsNS.exact_preservation
#print axioms Sounio.EpistemicEffectsNS.typed_agfree
#print axioms Sounio.EpistemicEffectsNS.soundness_star
#print axioms Sounio.EpistemicEffectsNS.x_plus_x_untypable
#print axioms Sounio.EpistemicEffectsNS.x_plus_top_untypable
#print axioms Sounio.EpistemicEffectsNS.measure_plus_measure_untypable
#print axioms Sounio.EpistemicEffectsNS.let_x_plus_x_untypable
#print axioms Sounio.EpistemicEffectsNS.x_plus_x_understates
