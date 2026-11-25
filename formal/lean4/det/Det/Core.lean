/-
  formal/lean4/det/Det/Core.lean

  Core calculus for deterministic floating-point values (`f32@Det`).
  Mathlib-free, no `sorry`, no custom axioms.

  Design summary:
  - Types: `f32`, `f32@Det`, `bool`, `bool@Det`.
  - Subtyping: `T@Det <: T` (reflexive, upward-only); there is no rule `T -> T@Det`.
  - Floats are abstract bit patterns (`Bits := UInt32`). A reference semantics
    `ref : Prim -> List Bits -> Bits` abstracts IEEE-754 arithmetic together with
    FTZ/DAZ and NaN canonicalization (all NaNs map to a single canonical bit pattern).
  - A backend `B` supplies `B.prim : Prim -> List Bits -> Bits` and an arbitrary
    non-contract `B.vendor : Nat -> List Bits -> Bits`.
  - Conformance `Conf ref B` is a hypothesis (`∀ p args, B.prim p args = ref p args`),
    never an axiom.
  - `ite` enforces the implicit-flow rule: a non-Det condition makes the result non-Det.
  - `detReduce` reduces a list of floats with a fixed reduction order (`seq` or `balancedTree`).
-/

namespace Det

/-- Abstract 32-bit floating-point bit pattern. IEEE-754 internals are not modeled. -/
abbrev Bits := UInt32

/-- Variable identifier. -/
abbrev Var := Nat

/-- Base scalar types. -/
inductive BaseTy where
  | f32
  | bool
  deriving DecidableEq, Repr

/-- Determinism qualifier: `@Det` vs plain (non-Det). -/
inductive DetQual where
  | det
  | plain
  deriving DecidableEq, Repr

/-- Lattice join on qualifiers: only `det ⊔ det = det`. -/
def DetQual.join : DetQual → DetQual → DetQual
  | .det, .det => .det
  | _, _ => .plain

theorem DetQual.join_eq_det {q1 q2 : DetQual} (h : DetQual.join q1 q2 = .det) :
    q1 = .det ∧ q2 = .det := by
  cases q1 <;> cases q2 <;> simp [DetQual.join] at h
  exact ⟨rfl, rfl⟩

/-- Qualified scalar type. -/
structure Ty where
  base : BaseTy
  qual : DetQual
  deriving DecidableEq, Repr

abbrev Ty.f32 : Ty := ⟨.f32, .plain⟩
abbrev Ty.f32Det : Ty := ⟨.f32, .det⟩
abbrev Ty.bool : Ty := ⟨.bool, .plain⟩
abbrev Ty.boolDet : Ty := ⟨.bool, .det⟩

/-- Qualifier subsumption: `@Det <: plain`, with no reverse rule `plain -> @Det`. -/
inductive SubQual : DetQual → DetQual → Prop where
  | refl (q : DetQual) : SubQual q q
  | det_le_plain : SubQual .det .plain

theorem SubQual.eq_det_of_le_det {q : DetQual} (h : SubQual q .det) : q = .det := by
  cases h with
  | refl => rfl

theorem SubQual.eq_plain_of_plain_le {q : DetQual} (h : SubQual .plain q) : q = .plain := by
  cases h with
  | refl => rfl

theorem SubQual.not_plain_le_det : ¬ SubQual .plain .det := by
  intro h
  cases h

/-- Type subsumption: `T@Det <: T`, preserving base type. -/
inductive SubTy : Ty → Ty → Prop where
  | mk {b : BaseTy} {q1 q2 : DetQual} : SubQual q1 q2 → SubTy ⟨b, q1⟩ ⟨b, q2⟩

theorem SubTy.eq_det_of_le_det {T1 T2 : Ty} (hsub : SubTy T1 T2) (hq : T2.qual = .det) :
    T1.qual = .det := by
  cases hsub with
  | mk hqsub =>
    cases hq
    exact SubQual.eq_det_of_le_det hqsub

theorem SubTy.eq_plain_of_plain_le {T1 T2 : Ty} (hsub : SubTy T1 T2) (hq : T1.qual = .plain) :
    T2.qual = .plain := by
  cases hsub with
  | mk hqsub =>
    cases hq
    exact SubQual.eq_plain_of_plain_le hqsub

theorem SubTy.not_plain_to_det (b : BaseTy) : ¬ SubTy ⟨b, .plain⟩ ⟨b, .det⟩ := by
  intro h
  cases h with
  | mk hq => exact SubQual.not_plain_le_det hq

/-- Contract floating-point primitives. -/
inductive Prim where
  | add
  | sub
  | mul
  | fma
  | div
  | sqrt
  | compare
  | neg
  | abs
  | copysign
  deriving DecidableEq, Repr

def Prim.arity : Prim → Nat
  | .sqrt | .neg | .abs => 1
  | .add | .sub | .mul | .div | .compare | .copysign => 2
  | .fma => 3

def Prim.retBase : Prim → BaseTy
  | .compare => .bool
  | _ => .f32

/-- Fixed reduction order for `det_reduce`. -/
inductive Order where
  | seq
  | balancedTree
  deriving DecidableEq, Repr

/-- Reference semantics for contract primitives (already includes FTZ/DAZ and NaN canonicalization). -/
abbrev RefSem := Prim → List Bits → Bits

/-- A backend provides implementations for contract primitives and arbitrary non-contract vendor ops. -/
structure Backend where
  prim : Prim → List Bits → Bits
  vendor : Nat → List Bits → Bits

/-- Backend conformance predicate (always passed as a hypothesis, never an axiom). -/
def Conf (ref : RefSem) (B : Backend) : Prop :=
  ∀ (p : Prim) (args : List Bits), B.prim p args = ref p args

/-- Core calculus terms. -/
inductive Term where
  | f32Lit (b : Bits)
  | boolLit (b : Bool)
  | var (x : Var)
  | letIn (x : Var) (e1 e2 : Term)
  | prim (p : Prim) (args : List Term)
  | vendor (tag : Nat) (args : List Term)
  | ite (cond thn els : Term)
  | detReduce (ord : Order) (op : Prim) (init : Term) (xs : List Term)
  deriving Repr

/-- Runtime values. -/
inductive Val where
  | f32 (b : Bits)
  | bool (b : Bool)
  | err
  deriving DecidableEq, Repr

/-- Decode a comparison result bit-pattern to `Bool`. -/
def bitsToBool (b : Bits) : Bool := b != 0

/-- Runtime environment. -/
abbrev Env := Var → Val

def Env.empty : Env := fun _ => .err

def Env.extend (env : Env) (x : Var) (v : Val) : Env :=
  fun y => if y == x then v else env y

/-- Typing environment. -/
abbrev TyEnv := Var → Option Ty

def TyEnv.empty : TyEnv := fun _ => none

def TyEnv.extend (Γ : TyEnv) (x : Var) (T : Ty) : TyEnv :=
  fun y => if y == x then some T else Γ y

/-- Extract a list of `f32` bit-patterns from evaluated argument values. -/
def extractF32s : List Val → Option (List Bits)
  | [] => some []
  | .f32 b :: rest =>
    match extractF32s rest with
    | some bs => some (b :: bs)
    | none => none
  | _ :: _ => none

/-- Single deterministic pairwise pass for balanced-tree reduction. -/
def pairPass (f : Bits → Bits → Bits) : List Bits → List Bits
  | [] => []
  | [a] => [a]
  | a :: b :: rest => f a b :: pairPass f rest

theorem pairPass_length_le (f : Bits → Bits → Bits) :
    ∀ xs : List Bits, (pairPass f xs).length ≤ xs.length
  | [] => Nat.le_refl 0
  | [_] => Nat.le_refl 1
  | _ :: _ :: rest => by
    simp only [pairPass, List.length]
    have h := pairPass_length_le f rest
    omega

theorem pairPass_congr {f1 f2 : Bits → Bits → Bits}
    (hf : ∀ a b, f1 a b = f2 a b) :
    ∀ xs : List Bits, pairPass f1 xs = pairPass f2 xs
  | [] => rfl
  | [_] => rfl
  | a :: b :: rest => by
    simp only [pairPass, hf a b, pairPass_congr hf rest]

/-- Deterministic balanced-tree reduction over a list of bit-patterns. -/
def reduceTree (f : Bits → Bits → Bits) (init : Bits) : List Bits → Bits
  | [] => init
  | [a] => a
  | a :: b :: rest =>
    reduceTree f init (pairPass f (a :: b :: rest))
termination_by xs => xs.length
decreasing_by
  simp only [pairPass, List.length]
  have h := pairPass_length_le f rest
  omega

theorem reduceTree_congr {f1 f2 : Bits → Bits → Bits}
    (hf : ∀ a b, f1 a b = f2 a b) (init : Bits) :
    ∀ xs : List Bits, reduceTree f1 init xs = reduceTree f2 init xs
  | [] => by simp [reduceTree]
  | [_] => by simp [reduceTree]
  | a :: b :: rest => by
    rw [reduceTree, reduceTree]
    rw [pairPass_congr hf (a :: b :: rest)]
    exact reduceTree_congr hf init (pairPass f2 (a :: b :: rest))
termination_by xs => xs.length
decreasing_by
  simp only [pairPass, List.length]
  have h := pairPass_length_le f2 rest
  omega

theorem foldl_congr {f1 f2 : Bits → Bits → Bits}
    (hf : ∀ a b, f1 a b = f2 a b) :
    ∀ (init : Bits) (xs : List Bits), xs.foldl f1 init = xs.foldl f2 init
  | _, [] => rfl
  | init, a :: rest => by
    simp only [List.foldl, hf init a]
    exact foldl_congr hf (f2 init a) rest

/-- Deterministic reduction parameterized by `Order`. -/
def reduceWithOrder (ord : Order) (f : Bits → Bits → Bits) (init : Bits) (xs : List Bits) : Bits :=
  match ord with
  | .seq => xs.foldl f init
  | .balancedTree => reduceTree f init xs

theorem reduceWithOrder_congr (ord : Order) {f1 f2 : Bits → Bits → Bits}
    (hf : ∀ a b, f1 a b = f2 a b) (init : Bits) (xs : List Bits) :
    reduceWithOrder ord f1 init xs = reduceWithOrder ord f2 init xs := by
  cases ord with
  | seq => exact foldl_congr hf init xs
  | balancedTree => exact reduceTree_congr hf init xs

/- Total functional evaluator parameterized by backend `B` and environment `env`. -/
mutual
  def eval (B : Backend) (env : Env) : Term → Val
    | .f32Lit b => .f32 b
    | .boolLit b => .bool b
    | .var x => env x
    | .letIn x e1 e2 =>
      let v1 := eval B env e1
      eval B (env.extend x v1) e2
    | .prim p args =>
      match extractF32s (evalList B env args) with
      | some bs =>
        match p.retBase with
        | .f32 => .f32 (B.prim p bs)
        | .bool => .bool (bitsToBool (B.prim p bs))
      | none => .err
    | .vendor tag args =>
      match extractF32s (evalList B env args) with
      | some bs => .f32 (B.vendor tag bs)
      | none => .err
    | .ite cond thn els =>
      match eval B env cond with
      | .bool true => eval B env thn
      | .bool false => eval B env els
      | _ => .err
    | .detReduce ord op init xs =>
      match eval B env init, extractF32s (evalList B env xs) with
      | .f32 v0, some vs =>
        .f32 (reduceWithOrder ord (fun a b => B.prim op [a, b]) v0 vs)
      | _, _ => .err

  def evalList (B : Backend) (env : Env) : List Term → List Val
    | [] => []
    | e :: es => eval B env e :: evalList B env es
end

/- Typing relation for the `f32@Det` core calculus, including subsumption and implicit-flow. -/
mutual
  inductive HasType : TyEnv → Term → Ty → Prop where
    | f32Lit {Γ b} :
        HasType Γ (.f32Lit b) Ty.f32Det
    | boolLit {Γ b} :
        HasType Γ (.boolLit b) Ty.boolDet
    | var {Γ x T} :
        Γ x = some T →
        HasType Γ (.var x) T
    | letIn {Γ x e1 e2 T1 T2} :
        HasType Γ e1 T1 →
        HasType (Γ.extend x T1) e2 T2 →
        HasType Γ (.letIn x e1 e2) T2
    | prim {Γ p args q} :
        args.length = p.arity →
        HasTypeList Γ args ⟨.f32, q⟩ →
        HasType Γ (.prim p args) ⟨p.retBase, q⟩
    | vendor {Γ tag args} :
        HasTypeList Γ args Ty.f32 →
        HasType Γ (.vendor tag args) Ty.f32
    | ite {Γ cond thn els b qc qb} :
        HasType Γ cond ⟨.bool, qc⟩ →
        HasType Γ thn ⟨b, qb⟩ →
        HasType Γ els ⟨b, qb⟩ →
        HasType Γ (.ite cond thn els) ⟨b, DetQual.join qc qb⟩
    | detReduce {Γ ord op init xs q} :
        op.arity = 2 →
        op.retBase = .f32 →
        HasType Γ init ⟨.f32, q⟩ →
        HasTypeList Γ xs ⟨.f32, q⟩ →
        HasType Γ (.detReduce ord op init xs) ⟨.f32, q⟩
    | sub {Γ e T1 T2} :
        HasType Γ e T1 →
        SubTy T1 T2 →
        HasType Γ e T2

  inductive HasTypeList : TyEnv → List Term → Ty → Prop where
    | nil {Γ T} :
        HasTypeList Γ [] T
    | cons {Γ e es T} :
        HasType Γ e T →
        HasTypeList Γ es T →
        HasTypeList Γ (e :: es) T
end

end Det
