inductive HasTy : TyCtx → Expr → Ty → EffectSet → Prop where
  | t_kraw : ∀ Γ T v m,
      HasTy Γ v T emptyE → IsValue v → kvalid m →
      HasTy Γ (.kraw v m) (.tknow .treal) emptyE
  | t_sub : ∀ Γ e T E E',
      HasTy Γ e T E → E ⊆ₑ E' → HasTy Γ e T E'
