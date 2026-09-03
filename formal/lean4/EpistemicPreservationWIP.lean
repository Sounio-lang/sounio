import EpistemicEffects
namespace Sounio.EpistemicEffects

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
  | t_measure Γ T e k _ _ ih => intro c d hc; simp [shift, ih c d hc]
  | t_kvalue Γ T E e _ ih => intro c d hc; simp [shift, ih c d hc]
  | t_kunc Γ T E e _ ih => intro c d hc; simp [shift, ih c d hc]
  | t_kconf Γ T E e _ ih => intro c d hc; simp [shift, ih c d hc]
  | t_kadd Γ T E₁ E₂ a b _ _ iha ihb => intro c d hc; simp [shift, iha c d hc, ihb c d hc]
  | t_kmul Γ T E₁ E₂ a b _ _ iha ihb => intro c d hc; simp [shift, iha c d hc, ihb c d hc]
  | t_let Γ T₁ T₂ E₁ E₂ e body _ _ ihe ihb =>
    intro c d hc
    have hb : (T₁ :: Γ).length ≤ c + 1 := by simp [List.length] at hc ⊢; omega
    simp [shift, ihe c d hc, ihb (c+1) d hb]
  | t_kraw => intro c d _; rfl
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
    have := ih (k+1) τ
    rw [hctx] at this
    exact .t_lam _ _ _ _ _ this
  | t_app Γ T₁ T₂ Ef Ec Ecaller f a _ _ hEf hEc ihf iha =>
    intro k τ; exact .t_app _ _ _ _ _ _ _ _ (ihf k τ) (iha k τ) hEf hEc
  | t_measure Γ T e kc _ hk ih =>
    intro k τ; exact .t_measure _ _ _ _ (ih k τ) hk
  | t_kvalue Γ T E e _ ih => intro k τ; exact .t_kvalue _ _ _ _ (ih k τ)
  | t_kunc Γ T E e _ ih => intro k τ; exact .t_kunc _ _ _ _ (ih k τ)
  | t_kconf Γ T E e _ ih => intro k τ; exact .t_kconf _ _ _ _ (ih k τ)
  | t_kadd Γ T E₁ E₂ a b _ _ iha ihb => intro k τ; exact .t_kadd _ _ _ _ _ _ (iha k τ) (ihb k τ)
  | t_kmul Γ T E₁ E₂ a b _ _ iha ihb => intro k τ; exact .t_kmul _ _ _ _ _ _ (iha k τ) (ihb k τ)
  | t_let Γ T₁ T₂ E₁ E₂ e body _ _ ihe ihb =>
    intro k τ
    have : shift k 1 (.letE e body) = .letE (shift k 1 e) (shift (k+1) 1 body) := by simp [shift]
    rw [this]
    have hctx : (T₁ :: Γ).insertIdx (k+1) τ = T₁ :: Γ.insertIdx k τ := by simp [List.insertIdx]
    have hb := ihb (k+1) τ
    rw [hctx] at hb
    exact .t_let _ _ _ _ _ _ _ (ihe k τ) hb
  | t_kraw Γ kc hk => intro k τ; exact .t_kraw _ _ hk
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

-- `invLam` was moved into `EpistemicEffects.lean` (§9.3) on 2026-08-16;
-- this file now imports it.

theorem invKraw {Γ e T E} (h : HasTy Γ e T E) {k} (he : e = .kraw k) :
    kvalid k ∧ T = .tknow .treal := by
  induction h with
  | t_kraw Γ' kc hk => injection he with h1; subst h1; exact ⟨hk, rfl⟩
  | t_sub Γ' e' T' E1 E2 h0 hsub ih => exact ih he
  | _ => exact Expr.noConfusion he

theorem value_emptyE {Γ v T E} (hv : IsValue v) (h : HasTy Γ v T E) :
    HasTy Γ v T emptyE := by
  cases hv with
  | v_nat n => rw [genNat h rfl]; exact .t_lit_nat _ _
  | v_real z => rw [genReal h rfl]; exact .t_lit_real _ _
  | v_lam S F b => rcases invLam h rfl with ⟨T₂, hT, hb⟩; subst hT; exact .t_lam _ _ _ _ _ hb
  | v_kraw k => rcases invKraw h rfl with ⟨hk, hT⟩; subst hT; exact .t_kraw _ _ hk

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
    have key : subst Δ.length v (.lam T₁ E' body) = .lam T₁ E' (subst (Δ.length + 1) v body) := by
      simp [subst, hsh]
    rw [key]
    exact .t_lam _ _ _ _ _ (ihb (T₁ :: Δ) rfl)
  | t_app Γ' T₁ T₂ Ef Ec Ecaller f a hf ha hEf hEc ihf iha =>
    intro Δ hΓ; subst hΓ
    exact .t_app _ _ _ _ _ _ _ _ (ihf Δ rfl) (iha Δ rfl) hEf hEc
  | t_measure Γ' T e kc he hk ih =>
    intro Δ hΓ; subst hΓ
    exact .t_measure _ _ _ _ (ih Δ rfl) hk
  | t_kvalue Γ' T E e _ ih => intro Δ hΓ; subst hΓ; exact .t_kvalue _ _ _ _ (ih Δ rfl)
  | t_kunc Γ' T E e _ ih => intro Δ hΓ; subst hΓ; exact .t_kunc _ _ _ _ (ih Δ rfl)
  | t_kconf Γ' T E e _ ih => intro Δ hΓ; subst hΓ; exact .t_kconf _ _ _ _ (ih Δ rfl)
  | t_kadd Γ' T E₁ E₂ a b _ _ iha ihb =>
    intro Δ hΓ; subst hΓ; exact .t_kadd _ _ _ _ _ _ (iha Δ rfl) (ihb Δ rfl)
  | t_kmul Γ' T E₁ E₂ a b _ _ iha ihb =>
    intro Δ hΓ; subst hΓ; exact .t_kmul _ _ _ _ _ _ (iha Δ rfl) (ihb Δ rfl)
  | t_let Γ' T₁ T₂ E₁ E₂ e body he hbody ihe ihb =>
    intro Δ hΓ; subst hΓ
    have hsh : shift 0 1 v = v := wellScoped hv 0 1 (by simp [List.length])
    have key : subst Δ.length v (.letE e body) = .letE (subst Δ.length v e) (subst (Δ.length + 1) v body) := by
      simp [subst, hsh]
    rw [key]
    exact .t_let _ _ _ _ _ _ _ (ihe Δ rfl) (ihb (T₁ :: Δ) rfl)
  | t_kraw Γ' kc hk => intro Δ hΓ; exact .t_kraw _ _ hk
  | t_sub Γ' e T E E' h0 hsub ih => intro Δ hΓ; exact .t_sub _ _ _ _ _ (ih Δ hΓ) hsub
