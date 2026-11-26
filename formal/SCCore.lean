/-
  SCCore.lean — ADR-032 step 1: the Lean fragment of Sounio Core (SC).

  Scope of this fragment (docs/core/SC_GRAMMAR_AND_TYPING.md §5):
    literals, `let`, linear and bilinear arithmetic, Knowledge values whose
    Knightian (K) source sets are tracked, named Knightian sources with
    declared bounds, calls to earlier-defined functions instantiated per call
    site, opaque calls (recursion / extern) summarised as `Unknown`, and the
    precise requirement `require`.

  Not in this fragment (stated, not hidden): aleatoric (A) sets, fields and
  projection, Core Lint, semantic refinement (theorem d), summaries as data.

  Theorems:
    (a) `sc_sound` — a well-typed expression never gets stuck, and its value
        has its type.
    (b) the same theorem, read for the refusals: a well-typed `require` never
        fails at run time as imprecise (E230 is sound), and a program whose
        named-bound table is consistent never meets two runtime terms of one
        named source with different bounds (E314 is sound).

  No `sorry`, no `axiom`, no `native_decide`.
-/

namespace SCCore

/-! ## 1. Sources, sets, types -/

/-- A Knightian source. `named n` is a declared source (`knight("n", lo, hi)`);
    `fresh node path` is a residue created by the bilinear node `node` under the
    call path `path` (innermost call site first). -/
inductive Src where
  | named (n : Nat)
  | fresh (node : Nat) (path : List Nat)
  deriving DecidableEq, Repr

/-- The static K set: `known S` (every runtime source is in `S`) or `unknown`. -/
inductive SSet where
  | known (s : List Src)
  | unknown
  deriving DecidableEq, Repr

/-- Join of the lattice Known(S) ⊑ Known(S ∪ T) ⊑ Unknown. -/
def SSet.join : SSet → SSet → SSet
  | .known a, .known b => .known (a ++ b)
  | _, _ => .unknown

inductive Ty where
  | num
  | kn (k : SSet)
  deriving DecidableEq, Repr

/-! ## 2. Expressions -/

inductive Expr where
  | lit (v : Int)
  | knight (n : Nat) (lo hi : Int)
  | var (x : Nat)
  | let_ (x : Nat) (e b : Expr)
  | add (a b : Expr)
  | sub (a b : Expr)
  | scale (c : Int) (a : Expr)
  | mulk (node : Nat) (a b : Expr)
  /-- Call of function `f` (one argument, bound to variable 0) at site `site`. -/
  | call (site f : Nat) (a : Expr)
  /-- Call whose body is not analysed (recursion, extern): summary `Unknown`. -/
  | opq (f : Nat) (a : Expr)
  /-- Precise requirement: the value must carry no Knightian source. -/
  | require (a : Expr)
  deriving DecidableEq, Repr

/-- The named-source declarations occurring in an expression. -/
def knights : Expr → List (Nat × Int × Int)
  | .lit _ => []
  | .knight n lo hi => [(n, lo, hi)]
  | .var _ => []
  | .let_ _ e b => knights e ++ knights b
  | .add a b => knights a ++ knights b
  | .sub a b => knights a ++ knights b
  | .scale _ a => knights a
  | .mulk _ a b => knights a ++ knights b
  | .call _ _ a => knights a
  | .opq _ a => knights a
  | .require a => knights a

/-! ## 3. Values and dynamic semantics -/

structure Term where
  src : Src
  lo : Int
  hi : Int
  c : Int
  deriving DecidableEq, Repr

inductive Val where
  | num (v : Int)
  | kn (c : Int) (ts : List Term)
  deriving DecidableEq, Repr

inductive Err where
  | stuck
  | fuel
  | imprecise
  | conflict
  deriving DecidableEq, Repr

/-- Two terms of one named source with different bounds. -/
def clash (t u : Term) : Bool :=
  match t.src, u.src with
  | .named n, .named m => n == m && (t.lo != u.lo || t.hi != u.hi)
  | _, _ => false

def conflict (ts : List Term) : Bool :=
  ts.any (fun t => ts.any (fun u => clash t u))

def Term.scale (k : Int) (t : Term) : Term := { t with c := k * t.c }

def absI (x : Int) : Int := if x < 0 then -x else x

def rad : List Term → Int
  | [] => 0
  | t :: ts => absI t.c * (absI t.lo + absI t.hi) + rad ts

def mkKn (c : Int) (ts : List Term) : Except Err Val :=
  if conflict ts then .error .conflict else .ok (.kn c ts)

def addV : Val → Val → Except Err Val
  | .num a, .num b => .ok (.num (a + b))
  | .num a, .kn c ts => .ok (.kn (a + c) ts)
  | .kn c ts, .num b => .ok (.kn (c + b) ts)
  | .kn c ts, .kn d us => mkKn (c + d) (ts ++ us)

def negV : Val → Val
  | .num a => .num (-a)
  | .kn c ts => .kn (-c) (ts.map (Term.scale (-1)))

def scaleV (k : Int) : Val → Val
  | .num a => .num (k * a)
  | .kn c ts => .kn (k * c) (ts.map (Term.scale k))

def mulV (node : Nat) (p : List Nat) : Val → Val → Except Err Val
  | .num a, .num b => .ok (.num (a * b))
  | .num a, v@(.kn _ _) => .ok (scaleV a v)
  | v@(.kn _ _), .num b => .ok (scaleV b v)
  | .kn c ts, .kn d us =>
      let r := rad ts * rad us
      mkKn (c * d) (ts.map (Term.scale d) ++ us.map (Term.scale c)
                     ++ [{ src := .fresh node p, lo := -r, hi := r, c := 1 }])

def lookupE {α : Type} (x : Nat) : List (Nat × α) → Option α
  | [] => none
  | (y, v) :: rest => if x = y then some v else lookupE x rest

/-- Big-step evaluation with fuel. `fns` is the function table (body of
    function `f` at index `f`, argument bound to variable 0); `orc` gives the
    value of an opaque call. -/
def eval (orc : Nat → Val → Val) (fns : List Expr) :
    Nat → List (Nat × Val) → List Nat → Expr → Except Err Val
  | 0, _, _, _ => .error .fuel
  | fuel + 1, ρ, p, e =>
    match e with
    | .lit v => .ok (.num v)
    | .knight n lo hi =>
        if lo ≤ hi then .ok (.kn 0 [{ src := .named n, lo := lo, hi := hi, c := 1 }])
        else .error .conflict
    | .var x =>
        match lookupE x ρ with
        | some v => .ok v
        | none => .error .stuck
    | .let_ x e1 b =>
        match eval orc fns fuel ρ p e1 with
        | .error er => .error er
        | .ok v => eval orc fns fuel ((x, v) :: ρ) p b
    | .add a b =>
        match eval orc fns fuel ρ p a, eval orc fns fuel ρ p b with
        | .ok va, .ok vb => addV va vb
        | .error er, _ => .error er
        | _, .error er => .error er
    | .sub a b =>
        match eval orc fns fuel ρ p a, eval orc fns fuel ρ p b with
        | .ok va, .ok vb => addV va (negV vb)
        | .error er, _ => .error er
        | _, .error er => .error er
    | .scale k a =>
        match eval orc fns fuel ρ p a with
        | .ok v => .ok (scaleV k v)
        | .error er => .error er
    | .mulk node a b =>
        match eval orc fns fuel ρ p a, eval orc fns fuel ρ p b with
        | .ok va, .ok vb => mulV node p va vb
        | .error er, _ => .error er
        | _, .error er => .error er
    | .call site f a =>
        match eval orc fns fuel ρ p a with
        | .error er => .error er
        | .ok va =>
          match fns[f]? with
          | some body => eval orc fns fuel [(0, va)] (site :: p) body
          | none => .error .stuck
    | .opq f a =>
        match eval orc fns fuel ρ p a with
        | .error er => .error er
        | .ok va => .ok (orc f va)
    | .require a =>
        match eval orc fns fuel ρ p a with
        | .error er => .error er
        | .ok (.num v) => .ok (.num v)
        | .ok (.kn c ts) => if ts.isEmpty then .ok (.num c) else .error .imprecise

/-! ## 4. Static typing (the checker's view, instantiated per call path) -/

def tyAdd : Ty → Ty → Ty
  | .num, .num => .num
  | .num, .kn k => .kn k
  | .kn k, .num => .kn k
  | .kn k, .kn k' => .kn (k.join k')

def tyMul (node : Nat) (p : List Nat) : Ty → Ty → Ty
  | .num, .num => .num
  | .num, .kn k => .kn k
  | .kn k, .num => .kn k
  | .kn k, .kn k' => .kn ((k.join k').join (.known [.fresh node p]))

def typeOf (fns : List Expr) :
    Nat → List (Nat × Ty) → List Nat → Expr → Option Ty
  | 0, _, _, _ => none
  | fuel + 1, Γ, p, e =>
    match e with
    | .lit _ => some .num
    | .knight n lo hi => if lo ≤ hi then some (.kn (.known [.named n])) else none
    | .var x => lookupE x Γ
    | .let_ x e1 b =>
        match typeOf fns fuel Γ p e1 with
        | none => none
        | some t => typeOf fns fuel ((x, t) :: Γ) p b
    | .add a b =>
        match typeOf fns fuel Γ p a, typeOf fns fuel Γ p b with
        | some ta, some tb => some (tyAdd ta tb)
        | _, _ => none
    | .sub a b =>
        match typeOf fns fuel Γ p a, typeOf fns fuel Γ p b with
        | some ta, some tb => some (tyAdd ta tb)
        | _, _ => none
    | .scale _ a => typeOf fns fuel Γ p a
    | .mulk node a b =>
        match typeOf fns fuel Γ p a, typeOf fns fuel Γ p b with
        | some ta, some tb => some (tyMul node p ta tb)
        | _, _ => none
    | .call site f a =>
        match typeOf fns fuel Γ p a with
        | none => none
        | some ta =>
          match fns[f]? with
          | some body => typeOf fns fuel [(0, ta)] (site :: p) body
          | none => none
    | .opq _ a =>
        match typeOf fns fuel Γ p a with
        | none => none
        | some _ => some (.kn .unknown)
    | .require a =>
        match typeOf fns fuel Γ p a with
        | some .num => some .num
        | some (.kn (.known [])) => some .num
        | _ => none

/-! ## 5. The invariant -/

def terms : Val → List Term
  | .num _ => []
  | .kn _ ts => ts

def HasTy : Val → Ty → Prop
  | .num _, .num => True
  | .kn _ ts, .kn (.known S) => ∀ t ∈ ts, t.src ∈ S
  | .kn _ _, .kn .unknown => True
  | _, _ => False

/-- The type of a Knowledge value depends on its terms, not on its centre. -/
theorem hasTy_center {c d : Int} {ts : List Term} {t : Ty}
    (h : HasTy (.kn c ts) t) : HasTy (.kn d ts) t := by
  cases t with
  | num => exact absurd h (by simp [HasTy])
  | kn k => cases k <;> exact h

/-- Every named term of `v` carries bounds declared somewhere in `occ`. -/
def NamedOK (occ : List (Nat × Int × Int)) (v : Val) : Prop :=
  ∀ t ∈ terms v, ∀ n, t.src = .named n → (n, t.lo, t.hi) ∈ occ

/-- E314 as a property of the program's declaration table. -/
def Consistent (occ : List (Nat × Int × Int)) : Prop :=
  ∀ n l h l' h', (n, l, h) ∈ occ → (n, l', h') ∈ occ → l = l' ∧ h = h'

inductive EnvRel (occ : List (Nat × Int × Int)) : List (Nat × Ty) → List (Nat × Val) → Prop
  | nil : EnvRel occ [] []
  | cons {Γ ρ x t v} : HasTy v t → NamedOK occ v → EnvRel occ Γ ρ →
      EnvRel occ ((x, t) :: Γ) ((x, v) :: ρ)

/-- An opaque call returns a Knowledge value built from fresh residues only. -/
def OrcOK (orc : Nat → Val → Val) : Prop :=
  ∀ f v, ∃ c ts, orc f v = .kn c ts ∧ ∀ t ∈ ts, ∃ nd q, t.src = .fresh nd q

/-! ## 6. Lemmas -/

theorem lookup_rel {occ : List (Nat × Int × Int)} {Γ : List (Nat × Ty)} {ρ : List (Nat × Val)} (h : EnvRel occ Γ ρ) (x : Nat) (t : Ty)
    (ht : lookupE x Γ = some t) :
    ∃ v, lookupE x ρ = some v ∧ HasTy v t ∧ NamedOK occ v := by
  induction h with
  | nil => simp [lookupE] at ht
  | @cons Γ ρ y t' v hv hn _ ih =>
    by_cases hxy : x = y
    · subst hxy
      simp [lookupE] at ht
      subst ht
      exact ⟨v, by simp [lookupE], hv, hn⟩
    · simp [lookupE, hxy] at ht ⊢
      exact ih ht

theorem conflict_false {occ : List (Nat × Int × Int)} (hc : Consistent occ) (ts : List Term)
    (hn : ∀ t ∈ ts, ∀ n, t.src = .named n → (n, t.lo, t.hi) ∈ occ) :
    conflict ts = false := by
  unfold conflict
  rw [Bool.eq_false_iff]
  intro h
  rw [List.any_eq_true] at h
  obtain ⟨t, ht, h⟩ := h
  rw [List.any_eq_true] at h
  obtain ⟨u, hu, h⟩ := h
  unfold clash at h
  cases hts : t.src with
  | fresh _ _ => simp [hts] at h
  | named n =>
    cases hus : u.src with
    | fresh _ _ => simp [hts, hus] at h
    | named m =>
      simp [hts, hus] at h
      obtain ⟨hnm, hb⟩ := h
      subst hnm
      have h1 := hn t ht n hts
      have h2 := hn u hu n hus
      obtain ⟨hl, hh⟩ := hc n _ _ _ _ h1 h2
      cases hb with
      | inl hb => exact hb hl
      | inr hb => exact hb hh

theorem mkKn_ok {occ : List (Nat × Int × Int)} (hc : Consistent occ) (c : Int) (ts : List Term)
    (hn : ∀ t ∈ ts, ∀ n, t.src = .named n → (n, t.lo, t.hi) ∈ occ) :
    mkKn c ts = .ok (.kn c ts) := by
  unfold mkKn
  rw [conflict_false hc ts hn]
  rfl

theorem namedOK_append {occ : List (Nat × Int × Int)} {c d : Int} {ts us : List Term}
    (h1 : NamedOK occ (.kn c ts)) (h2 : NamedOK occ (.kn d us)) :
    ∀ t ∈ ts ++ us, ∀ n, t.src = .named n → (n, t.lo, t.hi) ∈ occ := by
  intro t ht n hs
  rcases List.mem_append.mp ht with h | h
  · exact h1 t h n hs
  · exact h2 t h n hs

theorem namedOK_map_scale {occ : List (Nat × Int × Int)} {ts : List Term} (k : Int)
    (h : ∀ t ∈ ts, ∀ n, t.src = .named n → (n, t.lo, t.hi) ∈ occ) :
    ∀ t ∈ ts.map (Term.scale k), ∀ n, t.src = .named n → (n, t.lo, t.hi) ∈ occ := by
  intro t ht n hs
  rw [List.mem_map] at ht
  obtain ⟨u, hu, rfl⟩ := ht
  exact h u hu n hs

theorem src_map_scale {ts : List Term} {S : List Src} (k : Int)
    (h : ∀ t ∈ ts, t.src ∈ S) : ∀ t ∈ ts.map (Term.scale k), t.src ∈ S := by
  intro t ht
  rw [List.mem_map] at ht
  obtain ⟨u, hu, rfl⟩ := ht
  exact h u hu

theorem scaleV_ok {occ : List (Nat × Int × Int)} {v : Val} {t : Ty} (k : Int) (hv : HasTy v t) (hn : NamedOK occ v) :
    HasTy (scaleV k v) t ∧ NamedOK occ (scaleV k v) := by
  cases v with
  | num a =>
    cases t with
    | num => exact ⟨trivial, by intro t ht; simp [scaleV, terms] at ht⟩
    | kn _ => exact absurd hv (by simp [HasTy])
  | kn c ts =>
    refine ⟨?_, namedOK_map_scale k hn⟩
    cases t with
    | num => exact absurd hv (by simp [HasTy])
    | kn k' =>
      cases k' with
      | unknown => trivial
      | known S => exact src_map_scale k hv

theorem negV_ok {occ : List (Nat × Int × Int)} {v : Val} {t : Ty} (hv : HasTy v t) (hn : NamedOK occ v) :
    HasTy (negV v) t ∧ NamedOK occ (negV v) := by
  have h := scaleV_ok (occ := occ) (-1) hv hn
  cases v with
  | num a =>
    refine ⟨?_, by intro t ht; simp [negV, terms] at ht⟩
    cases t with
    | num => trivial
    | kn _ => exact absurd hv (by simp [HasTy])
  | kn c ts =>
    have e : negV (.kn c ts) = scaleV (-1) (.kn c ts) := by simp [negV, scaleV]
    rw [e]; exact h

theorem addV_ok {occ : List (Nat × Int × Int)} (hc : Consistent occ) {va vb : Val} {ta tb : Ty}
    (ha : HasTy va ta) (hna : NamedOK occ va)
    (hb : HasTy vb tb) (hnb : NamedOK occ vb) :
    ∃ v, addV va vb = .ok v ∧ HasTy v (tyAdd ta tb) ∧ NamedOK occ v := by
  cases va with
  | num a =>
    cases vb with
    | num b =>
      cases ta <;> cases tb <;> simp_all [HasTy, addV, tyAdd, NamedOK, terms]
    | kn d us =>
      cases ta with
      | kn _ => exact absurd ha (by simp [HasTy])
      | num =>
        cases tb with
        | num => exact absurd hb (by simp [HasTy])
        | kn kb => exact ⟨.kn (a + d) us, rfl, hasTy_center hb, hnb⟩
  | kn c ts =>
    cases ta with
    | num => exact absurd ha (by simp [HasTy])
    | kn ka =>
      cases vb with
      | num b =>
        cases tb with
        | kn _ => exact absurd hb (by simp [HasTy])
        | num => exact ⟨.kn (c + b) ts, rfl, hasTy_center ha, hna⟩
      | kn d us =>
        cases tb with
        | num => exact absurd hb (by simp [HasTy])
        | kn kb =>
          refine ⟨.kn (c + d) (ts ++ us), mkKn_ok hc _ _ (namedOK_append hna hnb), ?_,
            namedOK_append hna hnb⟩
          cases ka with
          | unknown => simp [tyAdd, SSet.join, HasTy]
          | known S =>
            cases kb with
            | unknown => simp [tyAdd, SSet.join, HasTy]
            | known T =>
              simp only [tyAdd, SSet.join, HasTy]
              intro t ht
              rcases List.mem_append.mp ht with h | h
              · exact List.mem_append_left _ (ha t h)
              · exact List.mem_append_right _ (hb t h)

theorem mulV_ok {occ : List (Nat × Int × Int)} (hc : Consistent occ) (node : Nat) (p : List Nat) {va vb : Val} {ta tb : Ty}
    (ha : HasTy va ta) (hna : NamedOK occ va)
    (hb : HasTy vb tb) (hnb : NamedOK occ vb) :
    ∃ v, mulV node p va vb = .ok v ∧ HasTy v (tyMul node p ta tb) ∧ NamedOK occ v := by
  cases va with
  | num a =>
    cases ta with
    | kn _ => exact absurd ha (by simp [HasTy])
    | num =>
      cases vb with
      | num b =>
        cases tb with
        | kn _ => exact absurd hb (by simp [HasTy])
        | num => exact ⟨_, rfl, trivial, by intro t ht; simp [terms] at ht⟩
      | kn d us =>
        cases tb with
        | num => exact absurd hb (by simp [HasTy])
        | kn kb =>
          have h := scaleV_ok (occ := occ) a hb hnb
          exact ⟨_, rfl, h.1, h.2⟩
  | kn c ts =>
    cases ta with
    | num => exact absurd ha (by simp [HasTy])
    | kn ka =>
      cases vb with
      | num b =>
        cases tb with
        | kn _ => exact absurd hb (by simp [HasTy])
        | num =>
          have h := scaleV_ok (occ := occ) b ha hna
          exact ⟨_, rfl, h.1, h.2⟩
      | kn d us =>
        cases tb with
        | num => exact absurd hb (by simp [HasTy])
        | kn kb =>
          let r := rad ts * rad us
          let fr : Term := { src := .fresh node p, lo := -r, hi := r, c := 1 }
          have hn : ∀ t ∈ ts.map (Term.scale d) ++ us.map (Term.scale c) ++ [fr],
              ∀ n, t.src = .named n → (n, t.lo, t.hi) ∈ occ := by
            intro t ht n hs
            rcases List.mem_append.mp ht with h | h
            · rcases List.mem_append.mp h with h | h
              · exact namedOK_map_scale d hna t h n hs
              · exact namedOK_map_scale c hnb t h n hs
            · simp at h
              subst h
              simp [fr] at hs
          refine ⟨_, mkKn_ok hc _ _ hn, ?_, hn⟩
          cases ka with
          | unknown => simp [tyMul, SSet.join, HasTy]
          | known S =>
            cases kb with
            | unknown => simp [tyMul, SSet.join, HasTy]
            | known T =>
              simp only [tyMul, SSet.join, HasTy]
              intro t ht
              rcases List.mem_append.mp ht with h | h
              · rcases List.mem_append.mp h with h | h
                · exact List.mem_append_left _
                    (List.mem_append_left _ (src_map_scale d ha t h))
                · exact List.mem_append_left _
                    (List.mem_append_right _ (src_map_scale c hb t h))
              · simp at h
                subst h
                simp

/-! ## 7. Theorems (a) and (b) -/

/-- Soundness of SC typing for this fragment.

    If the checker types `e` (under an environment whose values have their
    types), then evaluation either runs out of fuel or returns a value that has
    the inferred type. In particular it is never `stuck` (theorem a), never
    `imprecise` (the E230 refusal at a precise requirement is sound) and never
    `conflict` (the E314 refusal of inconsistent named bounds is sound). -/
theorem sc_sound (orc : Nat → Val → Val) (hOrc : OrcOK orc)
    (fns : List Expr) (occ : List (Nat × Int × Int)) (hc : Consistent occ)
    (hfns : ∀ body ∈ fns, ∀ k ∈ knights body, k ∈ occ) :
    ∀ fuel Γ ρ p e t, EnvRel occ Γ ρ → (∀ k ∈ knights e, k ∈ occ) →
      typeOf fns fuel Γ p e = some t →
      (eval orc fns fuel ρ p e = .error .fuel ∨
        ∃ v, eval orc fns fuel ρ p e = .ok v ∧ HasTy v t ∧ NamedOK occ v) := by
  intro fuel
  induction fuel with
  | zero => intro Γ ρ p e t _ _ ht; simp [typeOf] at ht
  | succ fuel ih =>
    intro Γ ρ p e t hΓ hk ht
    cases e with
    | lit v =>
      simp [typeOf] at ht; subst ht
      exact Or.inr ⟨.num v, by simp [eval], trivial, by intro t ht; simp [terms] at ht⟩
    | knight n lo hi =>
      simp only [typeOf] at ht
      by_cases hle : lo ≤ hi
      · simp [hle] at ht; subst ht
        refine Or.inr ⟨.kn 0 [{ src := .named n, lo := lo, hi := hi, c := 1 }], by simp [eval, hle], ?_, ?_⟩
        · intro t ht; simp at ht; subst ht; simp
        · intro t ht n' hs
          simp [terms] at ht; subst ht
          simp at hs; subst hs
          exact hk _ (by simp [knights])
      · simp [hle] at ht
    | var x =>
      simp only [typeOf] at ht
      obtain ⟨v, hv, hty, hn⟩ := lookup_rel hΓ x t ht
      exact Or.inr ⟨v, by simp [eval, hv], hty, hn⟩
    | let_ x e1 b =>
      simp only [typeOf] at ht
      cases h1 : typeOf fns fuel Γ p e1 with
      | none => simp [h1] at ht
      | some t1 =>
        simp only [h1] at ht
        have hk1 : ∀ k ∈ knights e1, k ∈ occ := fun k h => hk k (by simp [knights, h])
        have hkb : ∀ k ∈ knights b, k ∈ occ := fun k h => hk k (by simp [knights, h])
        rcases ih Γ ρ p e1 t1 hΓ hk1 h1 with he | ⟨v1, he, hv1, hn1⟩
        · exact Or.inl (by simp [eval, he])
        · have := ih ((x, t1) :: Γ) ((x, v1) :: ρ) p b t (EnvRel.cons hv1 hn1 hΓ) hkb ht
          simpa [eval, he] using this
    | add a b | sub a b =>
      simp only [typeOf] at ht
      have hka : ∀ k ∈ knights a, k ∈ occ := fun k h => hk k (by simp [knights, h])
      have hkb : ∀ k ∈ knights b, k ∈ occ := fun k h => hk k (by simp [knights, h])
      cases h1 : typeOf fns fuel Γ p a with
      | none => simp [h1] at ht
      | some ta =>
        cases h2 : typeOf fns fuel Γ p b with
        | none => simp [h1, h2] at ht
        | some tb =>
          simp [h1, h2] at ht; subst ht
          rcases ih Γ ρ p a ta hΓ hka h1 with ea | ⟨va, ea, hva, hna⟩
          · exact Or.inl (by simp [eval, ea])
          · rcases ih Γ ρ p b tb hΓ hkb h2 with eb | ⟨vb, eb, hvb, hnb⟩
            · exact Or.inl (by simp [eval, ea, eb])
            · first
              | (obtain ⟨v, hv, hty, hn⟩ := addV_ok hc hva hna hvb hnb
                 have he : eval orc fns (fuel + 1) ρ p (.add a b) = .ok v := by
                   simp [eval, ea, eb, hv]
                 exact Or.inr ⟨v, he, hty, hn⟩)
              | (obtain ⟨hvb', hnb'⟩ := negV_ok hvb hnb
                 obtain ⟨v, hv, hty, hn⟩ := addV_ok hc hva hna hvb' hnb'
                 have he : eval orc fns (fuel + 1) ρ p (.sub a b) = .ok v := by
                   simp [eval, ea, eb, hv]
                 exact Or.inr ⟨v, he, hty, hn⟩)
    | scale k a =>
      simp only [typeOf] at ht
      have hka : ∀ k ∈ knights a, k ∈ occ := fun k' h => hk k' (by simp [knights, h])
      rcases ih Γ ρ p a t hΓ hka ht with ea | ⟨va, ea, hva, hna⟩
      · exact Or.inl (by simp [eval, ea])
      · obtain ⟨h1, h2⟩ := scaleV_ok (occ := occ) k hva hna
        exact Or.inr ⟨scaleV k va, by simp [eval, ea], h1, h2⟩
    | mulk node a b =>
      simp only [typeOf] at ht
      have hka : ∀ k ∈ knights a, k ∈ occ := fun k h => hk k (by simp [knights, h])
      have hkb : ∀ k ∈ knights b, k ∈ occ := fun k h => hk k (by simp [knights, h])
      cases h1 : typeOf fns fuel Γ p a with
      | none => simp [h1] at ht
      | some ta =>
        cases h2 : typeOf fns fuel Γ p b with
        | none => simp [h1, h2] at ht
        | some tb =>
          simp [h1, h2] at ht; subst ht
          rcases ih Γ ρ p a ta hΓ hka h1 with ea | ⟨va, ea, hva, hna⟩
          · exact Or.inl (by simp [eval, ea])
          · rcases ih Γ ρ p b tb hΓ hkb h2 with eb | ⟨vb, eb, hvb, hnb⟩
            · exact Or.inl (by simp [eval, ea, eb])
            · obtain ⟨v, hv, hty, hn⟩ := mulV_ok hc node p hva hna hvb hnb
              exact Or.inr ⟨v, by simp [eval, ea, eb, hv], hty, hn⟩
    | call site f a =>
      simp only [typeOf] at ht
      have hka : ∀ k ∈ knights a, k ∈ occ := fun k h => hk k (by simp [knights, h])
      cases h1 : typeOf fns fuel Γ p a with
      | none => simp [h1] at ht
      | some ta =>
        simp only [h1] at ht
        cases hf : fns[f]? with
        | none => simp [hf] at ht
        | some body =>
          simp only [hf] at ht
          have hkbody : ∀ k ∈ knights body, k ∈ occ :=
            hfns body (List.mem_of_getElem? hf)
          rcases ih Γ ρ p a ta hΓ hka h1 with ea | ⟨va, ea, hva, hna⟩
          · exact Or.inl (by simp [eval, ea])
          · have := ih [(0, ta)] [(0, va)] (site :: p) body t
              (EnvRel.cons hva hna EnvRel.nil) hkbody ht
            simpa [eval, ea, hf] using this
    | opq f a =>
      simp only [typeOf] at ht
      have hka : ∀ k ∈ knights a, k ∈ occ := fun k h => hk k (by simp [knights, h])
      cases h1 : typeOf fns fuel Γ p a with
      | none => simp [h1] at ht
      | some ta =>
        simp [h1] at ht; subst ht
        rcases ih Γ ρ p a ta hΓ hka h1 with ea | ⟨va, ea, _, _⟩
        · exact Or.inl (by simp [eval, ea])
        · obtain ⟨c, ts, horc, hfr⟩ := hOrc f va
          refine Or.inr ⟨orc f va, by simp [eval, ea], by rw [horc]; trivial, ?_⟩
          intro t ht n hs
          rw [horc] at ht
          obtain ⟨nd, q, hq⟩ := hfr t ht
          rw [hq] at hs; cases hs
    | require a =>
      simp only [typeOf] at ht
      have hka : ∀ k ∈ knights a, k ∈ occ := fun k h => hk k (by simp [knights, h])
      cases h1 : typeOf fns fuel Γ p a with
      | none => simp [h1] at ht
      | some ta =>
        rcases ih Γ ρ p a ta hΓ hka h1 with ea | ⟨va, ea, hva, hna⟩
        · exact Or.inl (by simp [eval, ea])
        · cases ta with
          | num =>
            simp [h1] at ht; subst ht
            cases va with
            | num v =>
              exact Or.inr ⟨.num v, by simp [eval, ea], trivial,
                by intro t ht; simp [terms] at ht⟩
            | kn _ _ => exact absurd hva (by simp [HasTy])
          | kn k =>
            cases k with
            | unknown => simp [h1] at ht
            | known S =>
              cases S with
              | cons s S => simp [h1] at ht
              | nil =>
                simp [h1] at ht; subst ht
                cases va with
                | num v => exact absurd hva (by simp [HasTy])
                | kn c ts =>
                  have hts : ts = [] := by
                    cases ts with
                    | nil => rfl
                    | cons t ts' => exact absurd (hva t (by simp)) (by simp)
                  subst hts
                  exact Or.inr ⟨.num c, by simp [eval, ea], trivial,
                    by intro t ht; simp [terms] at ht⟩

/-! ## 8. Corollaries: the refusals, stated directly -/

/-- (b, E230) A `require` the checker accepts never fails as imprecise. -/
theorem require_never_imprecise (orc : Nat → Val → Val) (hOrc : OrcOK orc)
    (fns : List Expr) (occ : List (Nat × Int × Int)) (hc : Consistent occ)
    (hfns : ∀ body ∈ fns, ∀ k ∈ knights body, k ∈ occ)
    (fuel : Nat) (p : List Nat) (a : Expr) (t : Ty)
    (hk : ∀ k ∈ knights a, k ∈ occ)
    (ht : typeOf fns fuel [] p (.require a) = some t) :
    eval orc fns fuel [] p (.require a) ≠ .error .imprecise := by
  rcases sc_sound orc hOrc fns occ hc hfns fuel [] [] p (.require a) t EnvRel.nil
      (by simpa [knights] using hk) ht with h | ⟨v, h, _, _⟩
  · rw [h]; intro h'; cases h'
  · rw [h]; intro h'; cases h'

/-- (b, E314) A program whose named-bound table is consistent never meets two
    runtime terms of one named source with different bounds. -/
theorem consistent_never_conflict (orc : Nat → Val → Val) (hOrc : OrcOK orc)
    (fns : List Expr) (occ : List (Nat × Int × Int)) (hc : Consistent occ)
    (hfns : ∀ body ∈ fns, ∀ k ∈ knights body, k ∈ occ)
    (fuel : Nat) (p : List Nat) (e : Expr) (t : Ty)
    (hk : ∀ k ∈ knights e, k ∈ occ)
    (ht : typeOf fns fuel [] p e = some t) :
    eval orc fns fuel [] p e ≠ .error .conflict := by
  rcases sc_sound orc hOrc fns occ hc hfns fuel [] [] p e t EnvRel.nil hk ht with h | ⟨v, h, _, _⟩
  · rw [h]; intro h'; cases h'
  · rw [h]; intro h'; cases h'

/-- (b, Unknown) An opaque (recursive / extern) result never satisfies a
    precise requirement: the checker refuses it. -/
theorem opaque_refused_at_require (fns : List Expr) (fuel : Nat) (Γ : List (Nat × Ty))
    (p : List Nat) (f : Nat) (a : Expr) :
    typeOf fns (fuel + 2) Γ p (.require (.opq f a)) = none := by
  simp only [typeOf]
  cases typeOf fns fuel Γ p a <;> rfl

/-! ## 9. The decidable E314 check, and import order -/

def consistentB (occ : List (Nat × Int × Int)) : Bool :=
  occ.all (fun a => occ.all (fun b => a.1 != b.1 || (a.2.1 == b.2.1 && a.2.2 == b.2.2)))

theorem consistentB_sound (occ : List (Nat × Int × Int)) (h : consistentB occ = true) :
    Consistent occ := by
  intro n l hh l' h' m1 m2
  unfold consistentB at h
  rw [List.all_eq_true] at h
  have h1 := h _ m1
  rw [List.all_eq_true] at h1
  have h2 := h1 _ m2
  simp at h2
  exact h2

/-- Source identity does not depend on module (import) order: the E314 table
    is a set, so any permutation of the declarations is consistent iff the
    original is. -/
theorem consistent_perm {occ occ' : List (Nat × Int × Int)} (hp : occ.Perm occ')
    (h : Consistent occ) : Consistent occ' := by
  intro n l hh l' h' m1 m2
  exact h n l hh l' h' (hp.mem_iff.mpr m1) (hp.mem_iff.mpr m2)

/-! ## 10. Step-1 witness programs (run by the kernel, `decide`/`rfl`) -/

/-- Function 0: the identity. Function 1: `x * x` at node 7 (a fresh residue). -/
def wFns : List Expr := [.var 0, .mulk 7 (.var 0) (.var 0)]

def oracle : Nat → Val → Val := fun _ _ => .kn 0 []

def k3 : Expr := .knight 3 0 1

/-- Repeated identity calls preserve one shared source: `id(k) - id(k)` carries
    two terms of the one source `named 3`, which cancel coefficient-wise. -/
example : eval oracle wFns 10 [] [] (.sub (.call 1 0 k3) (.call 2 0 k3)) =
    .ok (.kn 0 [{ src := .named 3, lo := 0, hi := 1, c := 1 },
                { src := .named 3, lo := 0, hi := 1, c := -1 }]) := by rfl

/-- Two calls that create fresh residues do not cancel: the residues of
    `sq(k)` at call sites 1 and 2 are distinct sources, statically and at run
    time. -/
example : typeOf wFns 10 [] [] (.sub (.call 1 1 k3) (.call 2 1 k3)) =
    some (.kn (.known [.named 3, .named 3, .fresh 7 [1], .named 3, .named 3, .fresh 7 [2]])) := by
  rfl

example : Src.fresh 7 [1] ≠ Src.fresh 7 [2] := by decide

/-- An imported named source with the same bounds agrees; with conflicting
    bounds the table is refused (E314). -/
example : consistentB [(3, 0, 1), (3, 0, 1)] = true := by decide
example : consistentB [(3, 0, 1), (3, 2, 3)] = false := by decide

/-- Reversed bounds are refused by the checker. -/
example : typeOf wFns 10 [] [] (.knight 3 2 1) = none := by rfl

/-- A precise requirement on a Knightian value is refused; on a constant it is
    accepted. -/
example : typeOf wFns 10 [] [] (.require k3) = none := by rfl
example : typeOf wFns 10 [] [] (.require (.lit 4)) = some .num := by rfl

end SCCore
