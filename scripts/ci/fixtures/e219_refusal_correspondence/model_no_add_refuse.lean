-- Mutant: Check types an unimplemented call as refuse but add does not infect.
-- Without add_refuse_l, abs+1 can still be a value — E219 as ornament.
inductive Check : Expr → Observation → Prop where
  | lit : ∀ n, Check (.lit n) (.value n)
  | add_v : ∀ a b n m,
      Check a (.value n) → Check b (.value m) →
      Check (.add a b) (.value (n + m))
  | call_ok : ∀ n, allowlisted n = true →
      Check (.call n) (.value (oracle n))
  | call_refuse : ∀ n, allowlisted n = false →
      Check (.call n) .refuse
