/-
  formal/lean4/det/Det/Theorems.lean

  Main metatheory for `f32@Det`:
  1. `det_agree`: every closed well-typed `T@Det` term evaluates to the exact same
     value on any two conforming backends `B1`, `B2`.
  2. `no_forge`: changing the vendor implementation alone cannot affect any
     closed `T@Det` result, even without a conformance hypothesis. The auxiliary
     syntax-replacement theorems also cover replacement by literal bits.
-/

import Det.Core

namespace Det

/-- Two runtime environments agree on every `@Det`-qualified variable in `Γ`. -/
def EnvAgreeDet (Γ : TyEnv) (env1 env2 : Env) : Prop :=
  ∀ (x : Var) (b : BaseTy), Γ x = some ⟨b, .det⟩ → env1 x = env2 x

theorem EnvAgreeDet.empty : EnvAgreeDet TyEnv.empty Env.empty Env.empty := by
  intro _ _ h
  cases h

theorem EnvAgreeDet.extend_det {Γ : TyEnv} {env1 env2 : Env}
    (h : EnvAgreeDet Γ env1 env2) (x : Var) (b : BaseTy) {v1 v2 : Val} (hv : v1 = v2) :
    EnvAgreeDet (Γ.extend x ⟨b, .det⟩) (env1.extend x v1) (env2.extend x v2) := by
  intro y by_ hy
  unfold TyEnv.extend at hy
  unfold Env.extend
  cases hxy : (y == x)
  · simp [hxy] at hy ⊢
    exact h y by_ hy
  · simp
    exact hv

theorem EnvAgreeDet.extend_plain {Γ : TyEnv} {env1 env2 : Env}
    (h : EnvAgreeDet Γ env1 env2) (x : Var) (b : BaseTy) (v1 v2 : Val) :
    EnvAgreeDet (Γ.extend x ⟨b, .plain⟩) (env1.extend x v1) (env2.extend x v2) := by
  intro y by_ hy
  unfold TyEnv.extend at hy
  unfold Env.extend
  cases hxy : (y == x)
  · simp [hxy] at hy ⊢
    exact h y by_ hy
  · simp [hxy] at hy

/- Mutual structural induction on `HasType` / `HasTypeList` for cross-backend agreement. -/
mutual
  theorem det_agree_open {ref : RefSem} {B1 B2 : Backend}
      (hB1 : Conf ref B1) (hB2 : Conf ref B2) :
      ∀ {Γ : TyEnv} {e : Term} {T : Ty},
      HasType Γ e T → T.qual = .det →
      ∀ {env1 env2 : Env}, EnvAgreeDet Γ env1 env2 →
      eval B1 env1 e = eval B2 env2 e
    | _, _, _, .f32Lit, _, _, _, _ => rfl
    | _, _, _, .boolLit, _, _, _, _ => rfl
    | _, _, ⟨b, .det⟩, .var hx, _, _, _, henv => henv _ b hx
    | _, _, _, .letIn (x := x) (T1 := ⟨b1, .det⟩) h1 h2, hq, _, _, henv =>
      let hv1 := det_agree_open hB1 hB2 h1 rfl henv
      det_agree_open hB1 hB2 h2 hq (henv.extend_det x b1 hv1)
    | _, _, _, .letIn (x := x) (T1 := ⟨b1, .plain⟩) _ h2, hq, _, _, henv =>
      det_agree_open hB1 hB2 h2 hq (henv.extend_plain x b1 _ _)
    | _, _, _, .prim (p := p) (args := args) (q := .det) _ hargs, _, env1, env2, henv => by
      have hlist := det_agree_list hB1 hB2 hargs rfl henv
      simp only [eval, hlist]
      cases extractF32s (evalList B2 env2 args) with
      | none => rfl
      | some bs =>
        have hp : B1.prim p bs = B2.prim p bs := (hB1 p bs).trans (hB2 p bs).symm
        cases p.retBase <;> simp [hp]
    | _, _, _, .vendor _, hq, _, _, _ => by cases hq
    | _, _, _, .ite hc ht' he', hq, env1, env2, henv => by
      have ⟨hqc, hqb⟩ := DetQual.join_eq_det hq
      have hcond := det_agree_open hB1 hB2 hc hqc henv
      simp only [eval, hcond]
      cases eval B2 env2 _ with
      | bool bval =>
        cases bval with
        | true => exact det_agree_open hB1 hB2 ht' hqb henv
        | false => exact det_agree_open hB1 hB2 he' hqb henv
      | f32 _ => rfl
      | err => rfl
    | _, _, _, .detReduce (ord := ord) (op := op) (init := init) (xs := xs) (q := .det) _ _ hinit hxs,
        _, env1, env2, henv => by
      have hi := det_agree_open hB1 hB2 hinit rfl henv
      have hl := det_agree_list hB1 hB2 hxs rfl henv
      simp only [eval, hi, hl]
      cases eval B2 env2 init with
      | f32 v0 =>
        cases extractF32s (evalList B2 env2 xs) with
        | none => rfl
        | some vs =>
          dsimp only
          congr 1
          exact reduceWithOrder_congr ord (fun a b => (hB1 op [a, b]).trans (hB2 op [a, b]).symm) v0 vs
      | bool _ => rfl
      | err => rfl
    | _, _, _, .sub h1 hsub, hq, _, _, henv =>
      det_agree_open hB1 hB2 h1 (SubTy.eq_det_of_le_det hsub hq) henv

  theorem det_agree_list {ref : RefSem} {B1 B2 : Backend}
      (hB1 : Conf ref B1) (hB2 : Conf ref B2) :
      ∀ {Γ : TyEnv} {es : List Term} {T : Ty},
      HasTypeList Γ es T → T.qual = .det →
      ∀ {env1 env2 : Env}, EnvAgreeDet Γ env1 env2 →
      evalList B1 env1 es = evalList B2 env2 es
    | _, _, _, .nil, _, _, _, _ => rfl
    | _, _, _, .cons h1 hs, hq, _, _, henv => by
      simp only [evalList]
      rw [det_agree_open hB1 hB2 h1 hq henv, det_agree_list hB1 hB2 hs hq henv]
end

/-- Theorem 1: Cross-backend bit-identity for closed well-typed `@Det` terms. -/
theorem det_agree {ref : RefSem} {B1 B2 : Backend}
    (hB1 : Conf ref B1) (hB2 : Conf ref B2)
    {e : Term} {b : BaseTy}
    (ht : HasType TyEnv.empty e ⟨b, .det⟩) :
    eval B1 Env.empty e = eval B2 Env.empty e :=
  det_agree_open hB1 hB2 ht rfl EnvAgreeDet.empty

/-! ### Theorem 2: Non-forgeability (`no_forge`) -/

/-- Open-term version: even plain variable bindings may vary, provided all
    Det variable bindings agree and the contract implementation is fixed. -/
theorem no_forge_open (B : Backend) (vendor' : Nat → List Bits → Bits)
    {Γ : TyEnv} {env1 env2 : Env} {e : Term} {b : BaseTy}
    (ht : HasType Γ e ⟨b, .det⟩) (henv : EnvAgreeDet Γ env1 env2) :
    eval B env1 e = eval { B with vendor := vendor' } env2 e :=
  det_agree_open (ref := B.prim) (B1 := B) (B2 := { B with vendor := vendor' })
    (fun _ _ => rfl) (fun _ _ => rfl) ht rfl henv

/-- Vendor noninterference: replace the entire vendor implementation by any
    function of the operation tag and evaluated arguments. The contract
    primitive implementation is held fixed; no IEEE conformance is needed. -/
theorem no_forge (B : Backend) (vendor' : Nat → List Bits → Bits)
    {e : Term} {b : BaseTy}
    (ht : HasType TyEnv.empty e ⟨b, .det⟩) :
    eval B Env.empty e = eval { B with vendor := vendor' } Env.empty e :=
  no_forge_open B vendor' ht EnvAgreeDet.empty

private theorem hasType_vendor_plain_aux :
    ∀ {Γ : TyEnv} {e : Term} {T : Ty},
    HasType Γ e T → ∀ {tag : Nat} {args : List Term}, e = .vendor tag args → T.qual = .plain
  | _, _, _, .vendor _, _, _, _ => rfl
  | _, _, _, .sub h1 hsub, _, _, he => SubTy.eq_plain_of_plain_le hsub (hasType_vendor_plain_aux h1 he)
  | _, _, _, .f32Lit, _, _, he => by cases he
  | _, _, _, .boolLit, _, _, he => by cases he
  | _, _, _, .var _, _, _, he => by cases he
  | _, _, _, .letIn _ _, _, _, he => by cases he
  | _, _, _, .prim _ _, _, _, he => by cases he
  | _, _, _, .ite _ _ _, _, _, he => by cases he
  | _, _, _, .detReduce _ _ _ _, _, _, he => by cases he

/-- Any derivation for a `vendor` term must have `plain` (non-Det) qualifier. -/
theorem hasType_vendor_plain {Γ : TyEnv} {tag : Nat} {args : List Term} {T : Ty}
    (ht : HasType Γ (.vendor tag args) T) : T.qual = .plain :=
  hasType_vendor_plain_aux ht rfl

/-- Theorem 2a (direct): no derivation can assign `f32@Det` to a `vendor` operation. -/
theorem no_forge_direct {Γ : TyEnv} {tag : Nat} {args : List Term} :
    ¬ HasType Γ (.vendor tag args) Ty.f32Det := by
  intro h
  have hp := hasType_vendor_plain h
  cases hp

/- Replace every `vendor` subterm with an arbitrary constant bit-pattern determined by `r`. -/
mutual
  def replaceVendor (r : Nat → List Term → Bits) : Term → Term
    | .f32Lit b => .f32Lit b
    | .boolLit b => .boolLit b
    | .var x => .var x
    | .letIn x e1 e2 => .letIn x (replaceVendor r e1) (replaceVendor r e2)
    | .prim p args => .prim p (replaceVendorList r args)
    | .vendor tag args => .f32Lit (r tag args)
    | .ite cond thn els => .ite (replaceVendor r cond) (replaceVendor r thn) (replaceVendor r els)
    | .detReduce ord op init xs =>
      .detReduce ord op (replaceVendor r init) (replaceVendorList r xs)

  def replaceVendorList (r : Nat → List Term → Bits) : List Term → List Term
    | [] => []
    | e :: es => replaceVendor r e :: replaceVendorList r es
end

theorem replaceVendorList_length (r : Nat → List Term → Bits) :
    ∀ xs : List Term, (replaceVendorList r xs).length = xs.length
  | [] => rfl
  | _ :: xs => by
    simp only [replaceVendorList, List.length, replaceVendorList_length r xs]

/- Theorem 2b (typing preservation under vendor replacement):
   replacing every `vendor` subterm with any bit-pattern preserves typing. -/
mutual
  theorem no_forge_typing (r : Nat → List Term → Bits) :
      ∀ {Γ : TyEnv} {e : Term} {T : Ty},
      HasType Γ e T → HasType Γ (replaceVendor r e) T
    | _, _, _, .f32Lit => .f32Lit
    | _, _, _, .boolLit => .boolLit
    | _, _, _, .var hx => .var hx
    | _, _, _, .letIn h1 h2 => .letIn (no_forge_typing r h1) (no_forge_typing r h2)
    | _, _, _, .prim hlen hargs =>
      .prim (by rw [replaceVendorList_length, hlen]) (no_forge_typing_list r hargs)
    | _, _, _, .vendor _ => .sub .f32Lit (.mk .det_le_plain)
    | _, _, _, .ite hc ht' he' =>
      .ite (no_forge_typing r hc) (no_forge_typing r ht') (no_forge_typing r he')
    | _, _, _, .detReduce ha hr hinit hxs =>
      .detReduce ha hr (no_forge_typing r hinit) (no_forge_typing_list r hxs)
    | _, _, _, .sub h1 hsub => .sub (no_forge_typing r h1) hsub

  theorem no_forge_typing_list (r : Nat → List Term → Bits) :
      ∀ {Γ : TyEnv} {es : List Term} {T : Ty},
      HasTypeList Γ es T → HasTypeList Γ (replaceVendorList r es) T
    | _, _, _, .nil => .nil
    | _, _, _, .cons h1 hs => .cons (no_forge_typing r h1) (no_forge_typing_list r hs)
end

/- Semantic non-interference for open `@Det` terms under vendor replacement. -/
mutual
  theorem no_forge_eval_open (B : Backend) (r : Nat → List Term → Bits) :
      ∀ {Γ : TyEnv} {e : Term} {T : Ty},
      HasType Γ e T → T.qual = .det →
      ∀ {env1 env2 : Env}, EnvAgreeDet Γ env1 env2 →
      eval B env1 (replaceVendor r e) = eval B env2 e
    | _, _, _, .f32Lit, _, _, _, _ => rfl
    | _, _, _, .boolLit, _, _, _, _ => rfl
    | _, _, ⟨b, .det⟩, .var hx, _, _, _, henv => henv _ b hx
    | _, _, _, .letIn (x := x) (T1 := ⟨b1, .det⟩) h1 h2, hq, _, _, henv =>
      let hv1 := no_forge_eval_open B r h1 rfl henv
      no_forge_eval_open B r h2 hq (henv.extend_det x b1 hv1)
    | _, _, _, .letIn (x := x) (T1 := ⟨b1, .plain⟩) _ h2, hq, _, _, henv =>
      no_forge_eval_open B r h2 hq (henv.extend_plain x b1 _ _)
    | _, _, _, .prim (q := .det) _ hargs, _, env1, env2, henv => by
      have hl := no_forge_eval_list B r hargs rfl henv
      simp only [replaceVendor, eval, hl]
    | _, _, _, .vendor _, hq, _, _, _ => by cases hq
    | _, _, _, .ite hc ht' he', hq, env1, env2, henv => by
      have ⟨hqc, hqb⟩ := DetQual.join_eq_det hq
      have hcond := no_forge_eval_open B r hc hqc henv
      simp only [replaceVendor, eval, hcond]
      cases eval B env2 _ with
      | bool bval =>
        cases bval with
        | true => exact no_forge_eval_open B r ht' hqb henv
        | false => exact no_forge_eval_open B r he' hqb henv
      | f32 _ => rfl
      | err => rfl
    | _, _, _, .detReduce (q := .det) _ _ hinit hxs, _, env1, env2, henv => by
      have hi := no_forge_eval_open B r hinit rfl henv
      have hl := no_forge_eval_list B r hxs rfl henv
      simp only [replaceVendor, eval, hi, hl]
    | _, _, _, .sub h1 hsub, hq, _, _, henv =>
      no_forge_eval_open B r h1 (SubTy.eq_det_of_le_det hsub hq) henv

  theorem no_forge_eval_list (B : Backend) (r : Nat → List Term → Bits) :
      ∀ {Γ : TyEnv} {es : List Term} {T : Ty},
      HasTypeList Γ es T → T.qual = .det →
      ∀ {env1 env2 : Env}, EnvAgreeDet Γ env1 env2 →
      evalList B env1 (replaceVendorList r es) = evalList B env2 es
    | _, _, _, .nil, _, _, _, _ => rfl
    | _, _, _, .cons h1 hs, hq, _, _, henv => by
      simp only [replaceVendorList, evalList]
      rw [no_forge_eval_open B r h1 hq henv, no_forge_eval_list B r hs hq henv]
end

/-- Theorem 2c (semantic non-forgeability): for any closed `@Det` term (in particular `f32@Det`),
    replacing every `vendor` subterm with any arbitrary bit-pattern preserves both typing
    and the evaluated result — no `vendor` value can ever reach a `@Det` result. -/
theorem no_forge_replacement (B : Backend) (r : Nat → List Term → Bits)
    {e : Term} {b : BaseTy}
    (ht : HasType TyEnv.empty e ⟨b, .det⟩) :
    HasType TyEnv.empty (replaceVendor r e) ⟨b, .det⟩ ∧
    eval B Env.empty (replaceVendor r e) = eval B Env.empty e :=
  ⟨no_forge_typing r ht, no_forge_eval_open B r ht rfl EnvAgreeDet.empty⟩

end Det
