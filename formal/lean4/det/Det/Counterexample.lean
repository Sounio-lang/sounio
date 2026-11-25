/-
  formal/lean4/det/Det/Counterexample.lean

  Theorem 3: Necessity of the implicit-flow rule.
  In a variant type system `HasTypeNoImplicitFlow` identical to `HasType` except that
  `ite` does not taint the result qualifier with the condition qualifier, `det_agree`
  fails: a single vendor-valued condition distinguishes two conforming backends on a
  term typed `f32@Det`. We also prove that the full type system `HasType` rejects the
  same witness term.
-/

import Det.Core
import Det.Theorems

namespace Det

/- Variant typing relation without the implicit-flow rule on `ite`. -/
mutual
  inductive HasTypeNoImplicitFlow : TyEnv → Term → Ty → Prop where
    | f32Lit {Γ b} :
        HasTypeNoImplicitFlow Γ (.f32Lit b) Ty.f32Det
    | boolLit {Γ b} :
        HasTypeNoImplicitFlow Γ (.boolLit b) Ty.boolDet
    | var {Γ x T} :
        Γ x = some T →
        HasTypeNoImplicitFlow Γ (.var x) T
    | letIn {Γ x e1 e2 T1 T2} :
        HasTypeNoImplicitFlow Γ e1 T1 →
        HasTypeNoImplicitFlow (Γ.extend x T1) e2 T2 →
        HasTypeNoImplicitFlow Γ (.letIn x e1 e2) T2
    | prim {Γ p args q} :
        args.length = p.arity →
        HasTypeNoImplicitFlowList Γ args ⟨.f32, q⟩ →
        HasTypeNoImplicitFlow Γ (.prim p args) ⟨p.retBase, q⟩
    | vendor {Γ tag args} :
        HasTypeNoImplicitFlowList Γ args Ty.f32 →
        HasTypeNoImplicitFlow Γ (.vendor tag args) Ty.f32
    | ite {Γ cond thn els b qc qb} :
        HasTypeNoImplicitFlow Γ cond ⟨.bool, qc⟩ →
        HasTypeNoImplicitFlow Γ thn ⟨b, qb⟩ →
        HasTypeNoImplicitFlow Γ els ⟨b, qb⟩ →
        HasTypeNoImplicitFlow Γ (.ite cond thn els) ⟨b, qb⟩
    | detReduce {Γ ord op init xs q} :
        op.arity = 2 →
        op.retBase = .f32 →
        HasTypeNoImplicitFlow Γ init ⟨.f32, q⟩ →
        HasTypeNoImplicitFlowList Γ xs ⟨.f32, q⟩ →
        HasTypeNoImplicitFlow Γ (.detReduce ord op init xs) ⟨.f32, q⟩
    | sub {Γ e T1 T2} :
        HasTypeNoImplicitFlow Γ e T1 →
        SubTy T1 T2 →
        HasTypeNoImplicitFlow Γ e T2

  inductive HasTypeNoImplicitFlowList : TyEnv → List Term → Ty → Prop where
    | nil {Γ T} :
        HasTypeNoImplicitFlowList Γ [] T
    | cons {Γ e es T} :
        HasTypeNoImplicitFlow Γ e T →
        HasTypeNoImplicitFlowList Γ es T →
        HasTypeNoImplicitFlowList Γ (e :: es) T
end

/-- Concrete reference semantics where `compare [a, b]` returns `1` iff `a == b`, else `0`. -/
def witnessRef : RefSem
  | .compare, [a, b] => if a == b then 1 else 0
  | _, _ => 0

/-- Conforming backend A whose vendor op returns `0`. -/
def backendA : Backend :=
  { prim := witnessRef
    vendor := fun _ _ => 0 }

/-- Conforming backend B whose vendor op returns `1`. -/
def backendB : Backend :=
  { prim := witnessRef
    vendor := fun _ _ => 1 }

theorem conf_backendA : Conf witnessRef backendA := fun _ _ => rfl

theorem conf_backendB : Conf witnessRef backendB := fun _ _ => rfl

/-- Closed witness term: `if (compare (vendor 0 []) 0) then 1 else 2`. -/
def implicitFlowWitness : Term :=
  .ite (.prim .compare [.vendor 0 [], .f32Lit 0]) (.f32Lit 1) (.f32Lit 2)

theorem witness_typed_without_implicit_flow :
    HasTypeNoImplicitFlow TyEnv.empty implicitFlowWitness Ty.f32Det := by
  unfold implicitFlowWitness
  apply HasTypeNoImplicitFlow.ite (qc := .plain) (qb := .det)
  · apply HasTypeNoImplicitFlow.prim (p := .compare) (q := .plain) rfl
    apply HasTypeNoImplicitFlowList.cons
    · exact HasTypeNoImplicitFlow.vendor .nil
    · apply HasTypeNoImplicitFlowList.cons
      · exact HasTypeNoImplicitFlow.sub .f32Lit (.mk .det_le_plain)
      · exact .nil
  · exact .f32Lit
  · exact .f32Lit

theorem witness_diverges :
    eval backendA Env.empty implicitFlowWitness ≠
    eval backendB Env.empty implicitFlowWitness := by
  intro h
  have hA : eval backendA Env.empty implicitFlowWitness = Val.f32 1 := rfl
  have hB : eval backendB Env.empty implicitFlowWitness = Val.f32 2 := rfl
  rw [hA, hB] at h
  cases h

/-- Theorem 3: Without the implicit-flow rule, `det_agree` is false. -/
theorem implicit_flow_required :
    ∃ (ref : RefSem) (B1 B2 : Backend) (e : Term),
      Conf ref B1 ∧
      Conf ref B2 ∧
      HasTypeNoImplicitFlow TyEnv.empty e Ty.f32Det ∧
      eval B1 Env.empty e ≠ eval B2 Env.empty e :=
  ⟨witnessRef, backendA, backendB, implicitFlowWitness,
   conf_backendA, conf_backendB, witness_typed_without_implicit_flow, witness_diverges⟩

private theorem hasType_compare_vendor_plain_aux :
    ∀ {Γ : TyEnv} {e : Term} {T : Ty},
    HasType Γ e T → ∀ {tag : Nat} {vargs rest : List Term},
    e = .prim .compare (.vendor tag vargs :: rest) → T.qual = .plain
  | _, _, _, .prim _ hargs, _, _, _, he => by
    cases he
    cases hargs with
    | cons hv _ => have hp := hasType_vendor_plain hv; exact hp
  | _, _, _, .sub h1 hsub, _, _, _, he =>
    SubTy.eq_plain_of_plain_le hsub (hasType_compare_vendor_plain_aux h1 he)
  | _, _, _, .f32Lit, _, _, _, he => by cases he
  | _, _, _, .boolLit, _, _, _, he => by cases he
  | _, _, _, .var _, _, _, _, he => by cases he
  | _, _, _, .letIn _ _, _, _, _, he => by cases he
  | _, _, _, .vendor _, _, _, _, he => by cases he
  | _, _, _, .ite _ _ _, _, _, _, he => by cases he
  | _, _, _, .detReduce _ _ _ _, _, _, _, he => by cases he

private theorem hasType_witness_plain_aux :
    ∀ {Γ : TyEnv} {e : Term} {T : Ty},
    HasType Γ e T → e = implicitFlowWitness → T.qual = .plain
  | _, _, _, .ite hc _ _, he => by
    unfold implicitFlowWitness at he
    cases he
    have hcond_plain := hasType_compare_vendor_plain_aux hc rfl
    cases hcond_plain
    rfl
  | _, _, _, .sub h1 hsub, he =>
    SubTy.eq_plain_of_plain_le hsub (hasType_witness_plain_aux h1 he)
  | _, _, _, .f32Lit, he => by cases he
  | _, _, _, .boolLit, he => by cases he
  | _, _, _, .var _, he => by cases he
  | _, _, _, .letIn _ _, he => by cases he
  | _, _, _, .prim _ _, he => by cases he
  | _, _, _, .vendor _, he => by cases he
  | _, _, _, .detReduce _ _ _ _, he => by cases he

private theorem hasType_witness_plain {T : Ty}
    (ht : HasType TyEnv.empty implicitFlowWitness T) : T.qual = .plain :=
  hasType_witness_plain_aux ht rfl

/-- Sanity check: the full type system `HasType` (with implicit-flow) rejects `implicitFlowWitness` at `f32@Det`. -/
theorem implicit_flow_witness_rejected :
    ¬ HasType TyEnv.empty implicitFlowWitness Ty.f32Det := by
  intro ht
  have hp := hasType_witness_plain ht
  cases hp

end Det
