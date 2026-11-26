/- Well-typed closed terms evaluate to values of their declared base type,
   so cross-backend agreement is not merely equality of error sentinels. -/
import Det.Theorems

namespace Det

/-- Values of a scalar base type. The error sentinel has no base type. -/
inductive Val.HasBase : Val → BaseTy → Prop where
  | f32 (bits : Bits) : HasBase (.f32 bits) .f32
  | bool (b : Bool) : HasBase (.bool b) .bool

/-- A runtime environment supplies values of the declared base types. -/
def EnvTyped (Γ : TyEnv) (env : Env) : Prop :=
  ∀ x T, Γ x = some T → (env x).HasBase T.base

theorem EnvTyped.empty : EnvTyped TyEnv.empty Env.empty := by
  intro _ _ h
  cases h

theorem EnvTyped.extend {Γ : TyEnv} {env : Env} (h : EnvTyped Γ env)
    {x : Var} {v : Val} {T : Ty} (hv : v.HasBase T.base) :
    EnvTyped (Γ.extend x T) (env.extend x v) := by
  intro y U hy
  unfold TyEnv.extend at hy
  unfold Env.extend
  split at hy <;> rename_i he
  · simp [he] at hy ⊢
    cases hy
    exact hv
  · simp [he] at ⊢
    exact h y U hy

mutual
  theorem eval_has_base (B : Backend) :
      ∀ {Γ : TyEnv} {e : Term} {T : Ty}, HasType Γ e T →
      ∀ {env : Env}, EnvTyped Γ env → (eval B env e).HasBase T.base
    | _, _, _, .f32Lit, _, _ => .f32 _
    | _, _, _, .boolLit, _, _ => .bool _
    | _, _, _, .var hx, _, henv => henv _ _ hx
    | _, _, _, .letIn h1 h2, _, henv =>
      eval_has_base B h2 (henv.extend (eval_has_base B h1 henv))
    | _, _, _, .prim (p := p) _ hs, _, henv => by
      obtain ⟨bs, hbs⟩ := eval_list_f32 B hs rfl henv
      simp only [eval, hbs]
      cases p.retBase
      · exact .f32 _
      · exact .bool _
    | _, _, _, .vendor hs, _, henv => by
      obtain ⟨bs, hbs⟩ := eval_list_f32 B hs rfl henv
      simp only [eval, hbs]
      exact .f32 _
    | _, _, _, .ite (cond := cond) hc ht he, env, henv => by
      have hcond := eval_has_base B hc henv
      cases h : eval B env cond with
      | err => rw [h] at hcond; cases hcond
      | f32 _ => rw [h] at hcond; cases hcond
      | bool b =>
        cases b
        · simp only [eval, h]; exact eval_has_base B he henv
        · simp only [eval, h]; exact eval_has_base B ht henv
    | _, _, _, .detReduce (init := init) _ _ hi hs, env, henv => by
      have hv := eval_has_base B hi henv
      obtain ⟨bs, hbs⟩ := eval_list_f32 B hs rfl henv
      cases h : eval B env init with
      | err => rw [h] at hv; cases hv
      | bool _ => rw [h] at hv; cases hv
      | f32 _ => simp only [eval, h, hbs]; exact .f32 _
    | _, _, _, .sub ht (.mk _), _, henv => eval_has_base B ht henv

  theorem eval_list_f32 (B : Backend) :
      ∀ {Γ : TyEnv} {es : List Term} {T : Ty}, HasTypeList Γ es T →
      T.base = .f32 → ∀ {env : Env}, EnvTyped Γ env →
      ∃ bs, extractF32s (evalList B env es) = some bs
    | _, _, _, .nil, _, _, _ => ⟨[], rfl⟩
    | _, _, _, .cons ht hs, hb, _, henv => by
      have hv := eval_has_base B ht henv
      rw [hb] at hv
      obtain ⟨bs, hbs⟩ := eval_list_f32 B hs hb henv
      cases h : eval B _ _ with
      | err => rw [h] at hv; cases hv
      | bool _ => rw [h] at hv; cases hv
      | f32 b => exact ⟨b :: bs, by simp only [evalList, h, extractF32s, hbs]⟩
end

/-- The precise f32 bit-identity corollary: there exists a common bit pattern,
    not just a common evaluator result. -/
theorem det_f32_bits {ref : RefSem} {B1 B2 : Backend}
    (hB1 : Conf ref B1) (hB2 : Conf ref B2) {e : Term}
    (ht : HasType TyEnv.empty e Ty.f32Det) :
    ∃ bits : Bits, eval B1 Env.empty e = .f32 bits ∧ eval B2 Env.empty e = .f32 bits := by
  have hv := eval_has_base B1 ht EnvTyped.empty
  have he := det_agree hB1 hB2 ht
  cases h : eval B1 Env.empty e with
  | err => rw [h] at hv; cases hv
  | bool _ => rw [h] at hv; cases hv
  | f32 b => exact ⟨b, rfl, he.symm.trans h⟩

end Det
