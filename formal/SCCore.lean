/-
  SCCore.lean — ADR-032 step 1: the Lean fragment of Sounio Core (SC).

  Scope of this fragment (docs/compiler/SC_GRAMMAR_AND_TYPING.md §5):
    literals, `let`, linear and bilinear arithmetic, Knowledge values whose
    aleatory (A) and Knightian (K) source sets are tracked separately, named
    Knightian sources with declared bounds, named aleatory sources, pairs with
    per-field projection, calls to earlier-defined functions instantiated per
    call site, opaque calls (recursion / extern) summarised as `Unknown`, the
    precise requirement `require` (no Knightian source) and the exact
    requirement `requireExact` (no source at all).

  Not in this fragment (stated, not hidden): Core Lint, semantic refinement
  (theorem d), summaries as data (calls are typed by instantiating the body).

  Theorems:
    (a) `sc_sound` — a well-typed expression never gets stuck, and its value
        has its type: each runtime term lies in the static set of its kind.
    (b) the same theorem, read for the refusals: a well-typed `require` never
        fails at run time as imprecise (E230 is sound), a well-typed
        `requireExact` never either, and a program whose named-bound table is
        consistent never meets two runtime terms of one named Knightian source
        with different bounds (E314 is sound).
    `proj_carries_field_set`: a projection's type is the field's type, not a
    join over the aggregate.

  No `sorry`, no `axiom`, no `native_decide`.
-/

namespace SCCore

/-! ## 1. Sources, sets, types -/

/-- A source. `named n` is a declared source; `fresh node path` is a residue
    created by the bilinear node `node` under the call path `path` (innermost
    call site first). Whether a source is aleatory or Knightian is carried by
    the runtime term (`Term.kn`), not by the name. -/
inductive Src where
  | named (n : Nat)
  | fresh (node : Nat) (path : List Nat)
  deriving DecidableEq, Repr

/-- A static set: `known S` (every runtime source of that kind is in `S`) or
    `unknown`. -/
inductive SSet where
  | known (s : List Src)
  | unknown
  deriving DecidableEq, Repr

/-- Join of the lattice Known(S) ⊑ Known(S ∪ T) ⊑ Unknown. -/
def SSet.join : SSet → SSet → SSet
  | .known a, .known b => .known (a ++ b)
  | _, _ => .unknown

/-- Membership in a static set; everything is a member of `unknown`. -/
def inSet : SSet → Src → Prop
  | .known S, s => s ∈ S
  | .unknown, _ => True

theorem inSet_join_l {a b : SSet} {s : Src} (h : inSet a s) : inSet (a.join b) s := by
  cases a <;> cases b <;> simp_all [SSet.join, inSet]

theorem inSet_join_r {a b : SSet} {s : Src} (h : inSet b s) : inSet (a.join b) s := by
  cases a <;> cases b <;> simp_all [SSet.join, inSet]

inductive Ty where
  | num
  /-- Knowledge with an aleatory set `a` and a Knightian set `k`. -/
  | kn (a k : SSet)
  | pair (t u : Ty)
  deriving DecidableEq, Repr

/-! ## 2. Expressions -/

inductive Expr where
  | lit (v : Int)
  /-- A named Knightian source with declared bounds. -/
  | knight (n : Nat) (lo hi : Int)
  /-- A named aleatory source with standard uncertainty `sd`. -/
  | gauss (n : Nat) (sd : Int)
  | var (x : Nat)
  | let_ (x : Nat) (e b : Expr)
  | add (a b : Expr)
  | sub (a b : Expr)
  | scale (c : Int) (a : Expr)
  | mulk (node : Nat) (a b : Expr)
  | pair (a b : Expr)
  | fst (a : Expr)
  | snd (a : Expr)
  /-- Call of function `f` (one argument, bound to variable 0) at site `site`. -/
  | call (site f : Nat) (a : Expr)
  /-- Call whose body is not analysed (recursion, extern): summary `Unknown`. -/
  | opq (f : Nat) (a : Expr)
  /-- Precise requirement: the value must carry no Knightian source. -/
  | require (a : Expr)
  /-- Exact requirement: the value must carry no source at all. -/
  | requireExact (a : Expr)
  deriving DecidableEq, Repr

/-- The named Knightian declarations occurring in an expression. -/
def knights : Expr → List (Nat × Int × Int)
  | .lit _ => []
  | .knight n lo hi => [(n, lo, hi)]
  | .gauss _ _ => []
  | .var _ => []
  | .let_ _ e b => knights e ++ knights b
  | .add a b => knights a ++ knights b
  | .sub a b => knights a ++ knights b
  | .scale _ a => knights a
  | .mulk _ a b => knights a ++ knights b
  | .pair a b => knights a ++ knights b
  | .fst a => knights a
  | .snd a => knights a
  | .call _ _ a => knights a
  | .opq _ a => knights a
  | .require a => knights a
  | .requireExact a => knights a

/-! ## 3. Values and dynamic semantics -/

structure Term where
  src : Src
  lo : Int
  hi : Int
  c : Int
  /-- `true`: Knightian; `false`: aleatory. -/
  kn : Bool
  deriving DecidableEq, Repr

inductive Val where
  | num (v : Int)
  | kn (c : Int) (ts : List Term)
  | pair (v w : Val)
  deriving DecidableEq, Repr

inductive Err where
  | stuck
  | fuel
  | imprecise
  | conflict
  deriving DecidableEq, Repr

/-- Two Knightian terms of one named source with different bounds. -/
def clash (t u : Term) : Bool :=
  match t.src, u.src with
  | .named n, .named m => t.kn && u.kn && n == m && (t.lo != u.lo || t.hi != u.hi)
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
  | _, _ => .error .stuck

def scaleV (k : Int) : Val → Except Err Val
  | .num a => .ok (.num (k * a))
  | .kn c ts => .ok (.kn (k * c) (ts.map (Term.scale k)))
  | .pair _ _ => .error .stuck

def mulV (node : Nat) (p : List Nat) : Val → Val → Except Err Val
  | .num a, .num b => .ok (.num (a * b))
  | .num a, .kn c ts => .ok (.kn (a * c) (ts.map (Term.scale a)))
  | .kn c ts, .num b => .ok (.kn (b * c) (ts.map (Term.scale b)))
  | .kn c ts, .kn d us =>
      let r := rad ts * rad us
      mkKn (c * d) (ts.map (Term.scale d) ++ us.map (Term.scale c)
                     ++ [{ src := .fresh node p, lo := -r, hi := r, c := 1,
                           kn := (ts ++ us).any (·.kn) }])
  | _, _ => .error .stuck

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
        if lo ≤ hi then .ok (.kn 0 [{ src := .named n, lo := lo, hi := hi, c := 1, kn := true }])
        else .error .conflict
    | .gauss n sd => .ok (.kn 0 [{ src := .named n, lo := -sd, hi := sd, c := 1, kn := false }])
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
        | .ok va, .ok vb =>
            match scaleV (-1) vb with
            | .ok nb => addV va nb
            | .error er => .error er
        | .error er, _ => .error er
        | _, .error er => .error er
    | .scale k a =>
        match eval orc fns fuel ρ p a with
        | .ok v => scaleV k v
        | .error er => .error er
    | .mulk node a b =>
        match eval orc fns fuel ρ p a, eval orc fns fuel ρ p b with
        | .ok va, .ok vb => mulV node p va vb
        | .error er, _ => .error er
        | _, .error er => .error er
    | .pair a b =>
        match eval orc fns fuel ρ p a, eval orc fns fuel ρ p b with
        | .ok va, .ok vb => .ok (.pair va vb)
        | .error er, _ => .error er
        | _, .error er => .error er
    | .fst a =>
        match eval orc fns fuel ρ p a with
        | .ok (.pair v _) => .ok v
        | .ok _ => .error .stuck
        | .error er => .error er
    | .snd a =>
        match eval orc fns fuel ρ p a with
        | .ok (.pair _ w) => .ok w
        | .ok _ => .error .stuck
        | .error er => .error er
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
        | .ok (.kn c ts) => if ts.any (·.kn) then .error .imprecise else .ok (.num c)
        | .ok (.pair _ _) => .error .stuck
    | .requireExact a =>
        match eval orc fns fuel ρ p a with
        | .error er => .error er
        | .ok (.num v) => .ok (.num v)
        | .ok (.kn c ts) => if ts.isEmpty then .ok (.num c) else .error .imprecise
        | .ok (.pair _ _) => .error .stuck

/-! ## 4. Static typing (the checker's view, instantiated per call path) -/

def tyAdd : Ty → Ty → Option Ty
  | .num, .num => some .num
  | .num, .kn a k => some (.kn a k)
  | .kn a k, .num => some (.kn a k)
  | .kn a k, .kn a' k' => some (.kn (a.join a') (k.join k'))
  | _, _ => none

def tyScale : Ty → Option Ty
  | .num => some .num
  | .kn a k => some (.kn a k)
  | .pair _ _ => none

/-- The residue of a bilinear node always enters the A set. It enters the K set
    too unless both K sets are provably empty, in which case no Knightian term
    can reach the product and the residue is aleatory. -/
def tyMul (node : Nat) (p : List Nat) : Ty → Ty → Option Ty
  | .num, .num => some .num
  | .num, .kn a k => some (.kn a k)
  | .kn a k, .num => some (.kn a k)
  | .kn a k, .kn a' k' =>
      let fr : SSet := .known [.fresh node p]
      some (.kn ((a.join a').join fr)
        (if k = .known [] ∧ k' = .known [] then k.join k' else (k.join k').join fr))
  | _, _ => none

def typeOf (fns : List Expr) :
    Nat → List (Nat × Ty) → List Nat → Expr → Option Ty
  | 0, _, _, _ => none
  | fuel + 1, Γ, p, e =>
    match e with
    | .lit _ => some .num
    | .knight n lo hi =>
        if lo ≤ hi then some (.kn (.known []) (.known [.named n])) else none
    | .gauss n _ => some (.kn (.known [.named n]) (.known []))
    | .var x => lookupE x Γ
    | .let_ x e1 b =>
        match typeOf fns fuel Γ p e1 with
        | none => none
        | some t => typeOf fns fuel ((x, t) :: Γ) p b
    | .add a b | .sub a b =>
        match typeOf fns fuel Γ p a, typeOf fns fuel Γ p b with
        | some ta, some tb => tyAdd ta tb
        | _, _ => none
    | .scale _ a =>
        match typeOf fns fuel Γ p a with
        | some ta => tyScale ta
        | none => none
    | .mulk node a b =>
        match typeOf fns fuel Γ p a, typeOf fns fuel Γ p b with
        | some ta, some tb => tyMul node p ta tb
        | _, _ => none
    | .pair a b =>
        match typeOf fns fuel Γ p a, typeOf fns fuel Γ p b with
        | some ta, some tb => some (.pair ta tb)
        | _, _ => none
    | .fst a =>
        match typeOf fns fuel Γ p a with
        | some (.pair t _) => some t
        | _ => none
    | .snd a =>
        match typeOf fns fuel Γ p a with
        | some (.pair _ u) => some u
        | _ => none
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
        | some _ => some (.kn .unknown .unknown)
    | .require a =>
        match typeOf fns fuel Γ p a with
        | some .num => some .num
        | some (.kn _ (.known [])) => some .num
        | _ => none
    | .requireExact a =>
        match typeOf fns fuel Γ p a with
        | some .num => some .num
        | some (.kn (.known []) (.known [])) => some .num
        | _ => none

/-! ## 5. The invariant -/

def terms : Val → List Term
  | .num _ => []
  | .kn _ ts => ts
  | .pair v w => terms v ++ terms w

/-- Each runtime term lies in the static set of its own kind. -/
def HasTy : Val → Ty → Prop
  | .num _, .num => True
  | .kn _ ts, .kn a k => ∀ t ∈ ts, inSet (if t.kn then k else a) t.src
  | .pair v w, .pair t u => HasTy v t ∧ HasTy w u
  | _, _ => False

/-- Every named Knightian term of `v` carries bounds declared in `occ`. -/
def NamedOK (occ : List (Nat × Int × Int)) (v : Val) : Prop :=
  ∀ t ∈ terms v, t.kn = true → ∀ n, t.src = .named n → (n, t.lo, t.hi) ∈ occ

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

theorem lookup_rel {occ : List (Nat × Int × Int)} {Γ : List (Nat × Ty)} {ρ : List (Nat × Val)}
    (h : EnvRel occ Γ ρ) (x : Nat) (t : Ty) (ht : lookupE x Γ = some t) :
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
    (hn : ∀ t ∈ ts, t.kn = true → ∀ n, t.src = .named n → (n, t.lo, t.hi) ∈ occ) :
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
      obtain ⟨⟨⟨htk, huk⟩, hnm⟩, hb⟩ := h
      subst hnm
      have h1 := hn t ht htk n hts
      have h2 := hn u hu huk n hus
      obtain ⟨hl, hh⟩ := hc n _ _ _ _ h1 h2
      cases hb with
      | inl hb => exact hb hl
      | inr hb => exact hb hh

theorem mkKn_ok {occ : List (Nat × Int × Int)} (hc : Consistent occ) (c : Int) (ts : List Term)
    (hn : ∀ t ∈ ts, t.kn = true → ∀ n, t.src = .named n → (n, t.lo, t.hi) ∈ occ) :
    mkKn c ts = .ok (.kn c ts) := by
  unfold mkKn
  rw [conflict_false hc ts hn]
  rfl

theorem named_map_scale {occ : List (Nat × Int × Int)} {ts : List Term} (k : Int)
    (h : ∀ t ∈ ts, t.kn = true → ∀ n, t.src = .named n → (n, t.lo, t.hi) ∈ occ) :
    ∀ t ∈ ts.map (Term.scale k), t.kn = true → ∀ n, t.src = .named n → (n, t.lo, t.hi) ∈ occ := by
  intro t ht hk n hs
  rw [List.mem_map] at ht
  obtain ⟨u, hu, rfl⟩ := ht
  exact h u hu hk n hs

theorem inset_map_scale {ts : List Term} {a k : SSet} (s : Int)
    (h : ∀ t ∈ ts, inSet (if t.kn then k else a) t.src) :
    ∀ t ∈ ts.map (Term.scale s), inSet (if t.kn then k else a) t.src := by
  intro t ht
  rw [List.mem_map] at ht
  obtain ⟨u, hu, rfl⟩ := ht
  exact h u hu

/-- Monotonicity of the kind-indexed membership in both sets. -/
theorem inset_mono {t : Term} {a k a' k' : SSet}
    (ha : ∀ s, inSet a s → inSet a' s) (hk : ∀ s, inSet k s → inSet k' s)
    (h : inSet (if t.kn then k else a) t.src) : inSet (if t.kn then k' else a') t.src := by
  cases hb : t.kn <;> simp [hb] at h ⊢
  · exact ha _ h
  · exact hk _ h

theorem scaleV_ok {occ : List (Nat × Int × Int)} {v : Val} {t t' : Ty} (k : Int)
    (hv : HasTy v t) (hn : NamedOK occ v) (ht : tyScale t = some t') :
    ∃ v', scaleV k v = .ok v' ∧ HasTy v' t' ∧ NamedOK occ v' := by
  cases v with
  | num a =>
    cases t with
    | num =>
      simp [tyScale] at ht; subst ht
      exact ⟨_, rfl, trivial, by intro t ht; simp [terms] at ht⟩
    | kn _ _ => exact absurd hv (by simp [HasTy])
    | pair _ _ => exact absurd hv (by simp [HasTy])
  | kn c ts =>
    cases t with
    | num => exact absurd hv (by simp [HasTy])
    | kn a kk =>
      simp [tyScale] at ht; subst ht
      exact ⟨_, rfl, inset_map_scale k hv, named_map_scale k hn⟩
    | pair _ _ => exact absurd hv (by simp [HasTy])
  | pair _ _ =>
    cases t with
    | pair _ _ => simp [tyScale] at ht
    | num => exact absurd hv (by simp [HasTy])
    | kn _ _ => exact absurd hv (by simp [HasTy])

theorem addV_ok {occ : List (Nat × Int × Int)} (hc : Consistent occ) {va vb : Val} {ta tb t : Ty}
    (ha : HasTy va ta) (hna : NamedOK occ va)
    (hb : HasTy vb tb) (hnb : NamedOK occ vb) (ht : tyAdd ta tb = some t) :
    ∃ v, addV va vb = .ok v ∧ HasTy v t ∧ NamedOK occ v := by
  cases va with
  | pair _ _ =>
    cases ta with
    | pair _ _ => cases tb <;> simp [tyAdd] at ht
    | num => exact absurd ha (by simp [HasTy])
    | kn _ _ => exact absurd ha (by simp [HasTy])
  | num x =>
    cases ta with
    | kn _ _ => exact absurd ha (by simp [HasTy])
    | pair _ _ => exact absurd ha (by simp [HasTy])
    | num =>
      cases vb with
      | num y =>
        cases tb with
        | num =>
          simp [tyAdd] at ht; subst ht
          exact ⟨_, rfl, trivial, by intro t ht; simp [terms] at ht⟩
        | kn _ _ => exact absurd hb (by simp [HasTy])
        | pair _ _ => exact absurd hb (by simp [HasTy])
      | kn d us =>
        cases tb with
        | kn a k =>
          simp [tyAdd] at ht; subst ht
          exact ⟨_, rfl, hb, hnb⟩
        | num => exact absurd hb (by simp [HasTy])
        | pair _ _ => exact absurd hb (by simp [HasTy])
      | pair _ _ =>
        cases tb with
        | pair _ _ => simp [tyAdd] at ht
        | num => exact absurd hb (by simp [HasTy])
        | kn _ _ => exact absurd hb (by simp [HasTy])
  | kn c ts =>
    cases ta with
    | num => exact absurd ha (by simp [HasTy])
    | pair _ _ => exact absurd ha (by simp [HasTy])
    | kn a k =>
      cases vb with
      | num y =>
        cases tb with
        | num =>
          simp [tyAdd] at ht; subst ht
          exact ⟨_, rfl, ha, hna⟩
        | kn _ _ => exact absurd hb (by simp [HasTy])
        | pair _ _ => exact absurd hb (by simp [HasTy])
      | pair _ _ =>
        cases tb with
        | pair _ _ => simp [tyAdd] at ht
        | num => exact absurd hb (by simp [HasTy])
        | kn _ _ => exact absurd hb (by simp [HasTy])
      | kn d us =>
        cases tb with
        | num => exact absurd hb (by simp [HasTy])
        | pair _ _ => exact absurd hb (by simp [HasTy])
        | kn a' k' =>
          simp [tyAdd] at ht; subst ht
          have hn : ∀ t ∈ ts ++ us, t.kn = true → ∀ n, t.src = .named n → (n, t.lo, t.hi) ∈ occ := by
            intro t ht hk n hs
            rcases List.mem_append.mp ht with h | h
            · exact hna t h hk n hs
            · exact hnb t h hk n hs
          refine ⟨.kn (c + d) (ts ++ us), mkKn_ok hc _ _ hn, ?_, hn⟩
          intro t ht
          rcases List.mem_append.mp ht with h | h
          · exact inset_mono (fun _ => inSet_join_l) (fun _ => inSet_join_l) (ha t h)
          · exact inset_mono (fun _ => inSet_join_r) (fun _ => inSet_join_r) (hb t h)

/-- No Knightian term can live in a Knowledge value whose K set is `known []`. -/
theorem no_kn_of_empty {c : Int} {ts : List Term} {a : SSet}
    (h : HasTy (.kn c ts) (.kn a (.known []))) : ts.any (·.kn) = false := by
  rw [Bool.eq_false_iff]
  intro hany
  rw [List.any_eq_true] at hany
  obtain ⟨t, ht, hk⟩ := hany
  have := h t ht
  simp [hk, inSet] at this

theorem mulV_ok {occ : List (Nat × Int × Int)} (hc : Consistent occ) (node : Nat) (p : List Nat)
    {va vb : Val} {ta tb t : Ty}
    (ha : HasTy va ta) (hna : NamedOK occ va)
    (hb : HasTy vb tb) (hnb : NamedOK occ vb) (ht : tyMul node p ta tb = some t) :
    ∃ v, mulV node p va vb = .ok v ∧ HasTy v t ∧ NamedOK occ v := by
  cases va with
  | pair _ _ =>
    cases ta with
    | pair _ _ => cases tb <;> simp [tyMul] at ht
    | num => exact absurd ha (by simp [HasTy])
    | kn _ _ => exact absurd ha (by simp [HasTy])
  | num x =>
    cases ta with
    | kn _ _ => exact absurd ha (by simp [HasTy])
    | pair _ _ => exact absurd ha (by simp [HasTy])
    | num =>
      cases vb with
      | num y =>
        cases tb with
        | num =>
          simp [tyMul] at ht; subst ht
          exact ⟨_, rfl, trivial, by intro t ht; simp [terms] at ht⟩
        | kn _ _ => exact absurd hb (by simp [HasTy])
        | pair _ _ => exact absurd hb (by simp [HasTy])
      | kn d us =>
        cases tb with
        | kn a k =>
          simp [tyMul] at ht; subst ht
          exact ⟨_, rfl, inset_map_scale x hb, named_map_scale x hnb⟩
        | num => exact absurd hb (by simp [HasTy])
        | pair _ _ => exact absurd hb (by simp [HasTy])
      | pair _ _ =>
        cases tb with
        | pair _ _ => simp [tyMul] at ht
        | num => exact absurd hb (by simp [HasTy])
        | kn _ _ => exact absurd hb (by simp [HasTy])
  | kn c ts =>
    cases ta with
    | num => exact absurd ha (by simp [HasTy])
    | pair _ _ => exact absurd ha (by simp [HasTy])
    | kn a k =>
      cases vb with
      | num y =>
        cases tb with
        | num =>
          simp [tyMul] at ht; subst ht
          exact ⟨_, rfl, inset_map_scale y ha, named_map_scale y hna⟩
        | kn _ _ => exact absurd hb (by simp [HasTy])
        | pair _ _ => exact absurd hb (by simp [HasTy])
      | pair _ _ =>
        cases tb with
        | pair _ _ => simp [tyMul] at ht
        | num => exact absurd hb (by simp [HasTy])
        | kn _ _ => exact absurd hb (by simp [HasTy])
      | kn d us =>
        cases tb with
        | num => exact absurd hb (by simp [HasTy])
        | pair _ _ => exact absurd hb (by simp [HasTy])
        | kn a' k' =>
          simp only [tyMul, Option.some.injEq] at ht
          subst ht
          let r := rad ts * rad us
          let fr : Term := { src := .fresh node p, lo := -r, hi := r, c := 1,
                             kn := (ts ++ us).any (·.kn) }
          have hn : ∀ t ∈ ts.map (Term.scale d) ++ us.map (Term.scale c) ++ [fr],
              t.kn = true → ∀ n, t.src = .named n → (n, t.lo, t.hi) ∈ occ := by
            intro t ht hk n hs
            rcases List.mem_append.mp ht with h | h
            · rcases List.mem_append.mp h with h | h
              · exact named_map_scale d hna t h hk n hs
              · exact named_map_scale c hnb t h hk n hs
            · simp at h
              subst h
              simp [fr] at hs
          refine ⟨_, mkKn_ok hc _ _ hn, ?_, hn⟩
          -- Both K sets of the product dominate k and k'.
          have hKl : ∀ s, inSet k s → inSet (if k = .known [] ∧ k' = .known [] then k.join k'
              else (k.join k').join (.known [.fresh node p])) s := by
            intro s hs
            split
            · exact inSet_join_l hs
            · exact inSet_join_l (inSet_join_l hs)
          have hKr : ∀ s, inSet k' s → inSet (if k = .known [] ∧ k' = .known [] then k.join k'
              else (k.join k').join (.known [.fresh node p])) s := by
            intro s hs
            split
            · exact inSet_join_r hs
            · exact inSet_join_l (inSet_join_r hs)
          intro t ht
          rcases List.mem_append.mp ht with h | h
          · rcases List.mem_append.mp h with h | h
            · exact inset_mono (fun _ hs => inSet_join_l (inSet_join_l hs)) hKl
                (inset_map_scale d ha t h)
            · exact inset_mono (fun _ hs => inSet_join_l (inSet_join_r hs)) hKr
                (inset_map_scale c hb t h)
          · simp at h
            subst h
            have hfr : inSet (SSet.known [Src.fresh node p]) (Src.fresh node p) := by simp [inSet]
            dsimp only
            by_cases hkind : ((ts.any fun x => x.kn) || us.any fun x => x.kn) = true
            · rw [if_pos hkind]
              by_cases hboth : k = .known [] ∧ k' = .known []
              · exfalso
                obtain ⟨hk1, hk2⟩ := hboth
                subst hk1; subst hk2
                have e1 := no_kn_of_empty ha
                have e2 := no_kn_of_empty hb
                rw [e1, e2] at hkind
                simp at hkind
              · rw [if_neg hboth]; exact inSet_join_r hfr
            · rw [if_neg hkind]; exact inSet_join_r hfr

theorem hasTy_kn_any {c : Int} {ts : List Term} {a : SSet} {k : SSet}
    (h : HasTy (.kn c ts) (.kn a k)) (hk : k = .known []) : ts.any (·.kn) = false := by
  subst hk; exact no_kn_of_empty h

/-! ## 7. Theorems (a) and (b) -/

/-- Soundness of SC typing for this fragment.

    If the checker types `e` (under an environment whose values have their
    types), then evaluation either runs out of fuel or returns a value that has
    the inferred type. In particular it is never `stuck` (theorem a), never
    `imprecise` (the E230 refusals at `require`/`requireExact` are sound) and
    never `conflict` (the E314 refusal of inconsistent named bounds is sound). -/
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
        refine Or.inr ⟨.kn 0 [{ src := .named n, lo := lo, hi := hi, c := 1, kn := true }],
          by simp [eval, hle], ?_, ?_⟩
        · intro t ht; simp at ht; subst ht; simp [inSet]
        · intro t ht _ n' hs
          simp [terms] at ht; subst ht
          simp at hs; subst hs
          exact hk _ (by simp [knights])
      · simp [hle] at ht
    | gauss n sd =>
      simp [typeOf] at ht; subst ht
      refine Or.inr ⟨.kn 0 [{ src := .named n, lo := -sd, hi := sd, c := 1, kn := false }],
        by simp [eval], ?_, ?_⟩
      · intro t ht; simp at ht; subst ht; simp [inSet]
      · intro t ht hkn; simp [terms] at ht; subst ht; simp at hkn
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
    | add a b =>
      simp only [typeOf] at ht
      have hka : ∀ k ∈ knights a, k ∈ occ := fun k h => hk k (by simp [knights, h])
      have hkb : ∀ k ∈ knights b, k ∈ occ := fun k h => hk k (by simp [knights, h])
      cases h1 : typeOf fns fuel Γ p a with
      | none => simp [h1] at ht
      | some ta =>
        cases h2 : typeOf fns fuel Γ p b with
        | none => simp [h1, h2] at ht
        | some tb =>
          simp only [h1, h2] at ht
          rcases ih Γ ρ p a ta hΓ hka h1 with ea | ⟨va, ea, hva, hna⟩
          · exact Or.inl (by simp [eval, ea])
          · rcases ih Γ ρ p b tb hΓ hkb h2 with eb | ⟨vb, eb, hvb, hnb⟩
            · exact Or.inl (by simp [eval, ea, eb])
            · obtain ⟨v, hv, hty, hn⟩ := addV_ok hc hva hna hvb hnb ht
              have he : eval orc fns (fuel + 1) ρ p (.add a b) = .ok v := by
                simp [eval, ea, eb, hv]
              exact Or.inr ⟨v, he, hty, hn⟩
    | sub a b =>
      simp only [typeOf] at ht
      have hka : ∀ k ∈ knights a, k ∈ occ := fun k h => hk k (by simp [knights, h])
      have hkb : ∀ k ∈ knights b, k ∈ occ := fun k h => hk k (by simp [knights, h])
      cases h1 : typeOf fns fuel Γ p a with
      | none => simp [h1] at ht
      | some ta =>
        cases h2 : typeOf fns fuel Γ p b with
        | none => simp [h1, h2] at ht
        | some tb =>
          simp only [h1, h2] at ht
          rcases ih Γ ρ p a ta hΓ hka h1 with ea | ⟨va, ea, hva, hna⟩
          · exact Or.inl (by simp [eval, ea])
          · rcases ih Γ ρ p b tb hΓ hkb h2 with eb | ⟨vb, eb, hvb, hnb⟩
            · exact Or.inl (by simp [eval, ea, eb])
            · -- tyAdd succeeded, so tb is num or kn and scaling it keeps its type.
              have hsc : tyScale tb = some tb := by
                cases ta <;> cases tb <;> simp_all [tyAdd, tyScale]
              obtain ⟨nb, hnbv, hnbt, hnbn⟩ := scaleV_ok (occ := occ) (-1) hvb hnb hsc
              obtain ⟨v, hv, hty, hn⟩ := addV_ok hc hva hna hnbt hnbn ht
              have he : eval orc fns (fuel + 1) ρ p (.sub a b) = .ok v := by
                simp [eval, ea, eb, hnbv, hv]
              exact Or.inr ⟨v, he, hty, hn⟩
    | scale k a =>
      simp only [typeOf] at ht
      have hka : ∀ k ∈ knights a, k ∈ occ := fun k' h => hk k' (by simp [knights, h])
      cases h1 : typeOf fns fuel Γ p a with
      | none => simp [h1] at ht
      | some ta =>
        simp only [h1] at ht
        rcases ih Γ ρ p a ta hΓ hka h1 with ea | ⟨va, ea, hva, hna⟩
        · exact Or.inl (by simp [eval, ea])
        · obtain ⟨v, hv, hty, hn⟩ := scaleV_ok (occ := occ) k hva hna ht
          exact Or.inr ⟨v, by simp [eval, ea, hv], hty, hn⟩
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
          simp only [h1, h2] at ht
          rcases ih Γ ρ p a ta hΓ hka h1 with ea | ⟨va, ea, hva, hna⟩
          · exact Or.inl (by simp [eval, ea])
          · rcases ih Γ ρ p b tb hΓ hkb h2 with eb | ⟨vb, eb, hvb, hnb⟩
            · exact Or.inl (by simp [eval, ea, eb])
            · obtain ⟨v, hv, hty, hn⟩ := mulV_ok hc node p hva hna hvb hnb ht
              exact Or.inr ⟨v, by simp [eval, ea, eb, hv], hty, hn⟩
    | pair a b =>
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
            · refine Or.inr ⟨.pair va vb, by simp [eval, ea, eb], ⟨hva, hvb⟩, ?_⟩
              intro t ht
              simp only [terms] at ht
              rcases List.mem_append.mp ht with h | h
              · exact hna t h
              · exact hnb t h
    | fst a =>
      simp only [typeOf] at ht
      have hka : ∀ k ∈ knights a, k ∈ occ := fun k h => hk k (by simp [knights, h])
      cases h1 : typeOf fns fuel Γ p a with
      | none => simp [h1] at ht
      | some ta =>
        cases ta with
        | num => simp [h1] at ht
        | kn _ _ => simp [h1] at ht
        | pair t1 t2 =>
          simp [h1] at ht; subst ht
          rcases ih Γ ρ p a _ hΓ hka h1 with ea | ⟨va, ea, hva, hna⟩
          · exact Or.inl (by simp [eval, ea])
          · cases va with
            | pair v w =>
              refine Or.inr ⟨v, by simp [eval, ea], hva.1, ?_⟩
              intro t ht; exact hna t (by simp [terms, ht])
            | num _ => exact absurd hva (by simp [HasTy])
            | kn _ _ => exact absurd hva (by simp [HasTy])
    | snd a =>
      simp only [typeOf] at ht
      have hka : ∀ k ∈ knights a, k ∈ occ := fun k h => hk k (by simp [knights, h])
      cases h1 : typeOf fns fuel Γ p a with
      | none => simp [h1] at ht
      | some ta =>
        cases ta with
        | num => simp [h1] at ht
        | kn _ _ => simp [h1] at ht
        | pair t1 t2 =>
          simp [h1] at ht; subst ht
          rcases ih Γ ρ p a _ hΓ hka h1 with ea | ⟨va, ea, hva, hna⟩
          · exact Or.inl (by simp [eval, ea])
          · cases va with
            | pair v w =>
              refine Or.inr ⟨w, by simp [eval, ea], hva.2, ?_⟩
              intro t ht; exact hna t (by simp [terms, ht])
            | num _ => exact absurd hva (by simp [HasTy])
            | kn _ _ => exact absurd hva (by simp [HasTy])
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
          refine Or.inr ⟨orc f va, by simp [eval, ea], ?_, ?_⟩
          · rw [horc]; intro t _; cases t.kn <;> simp [inSet]
          · intro t ht _ n hs
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
        · have hnum : ∀ v, (∀ t ∈ terms (.num v), t.kn = true → ∀ n, t.src = .named n →
              (n, t.lo, t.hi) ∈ occ) := by intro v t ht; simp [terms] at ht
          cases ta with
          | num =>
            simp [h1] at ht; subst ht
            cases va with
            | num v => exact Or.inr ⟨.num v, by simp [eval, ea], trivial, hnum v⟩
            | kn _ _ => exact absurd hva (by simp [HasTy])
            | pair _ _ => exact absurd hva (by simp [HasTy])
          | pair _ _ => simp [h1] at ht
          | kn aa k =>
            cases k with
            | unknown => simp [h1] at ht
            | known S =>
              cases S with
              | cons s S => simp [h1] at ht
              | nil =>
                simp [h1] at ht; subst ht
                cases va with
                | num v => exact absurd hva (by simp [HasTy])
                | pair _ _ => exact absurd hva (by simp [HasTy])
                | kn c ts =>
                  have hany := no_kn_of_empty hva
                  exact Or.inr ⟨.num c, by simp [eval, ea, hany], trivial, hnum c⟩
    | requireExact a =>
      simp only [typeOf] at ht
      have hka : ∀ k ∈ knights a, k ∈ occ := fun k h => hk k (by simp [knights, h])
      cases h1 : typeOf fns fuel Γ p a with
      | none => simp [h1] at ht
      | some ta =>
        rcases ih Γ ρ p a ta hΓ hka h1 with ea | ⟨va, ea, hva, hna⟩
        · exact Or.inl (by simp [eval, ea])
        · have hnum : ∀ v, (∀ t ∈ terms (.num v), t.kn = true → ∀ n, t.src = .named n →
              (n, t.lo, t.hi) ∈ occ) := by intro v t ht; simp [terms] at ht
          cases ta with
          | num =>
            simp [h1] at ht; subst ht
            cases va with
            | num v => exact Or.inr ⟨.num v, by simp [eval, ea], trivial, hnum v⟩
            | kn _ _ => exact absurd hva (by simp [HasTy])
            | pair _ _ => exact absurd hva (by simp [HasTy])
          | pair _ _ => simp [h1] at ht
          | kn aa k =>
            cases aa with
            | unknown => cases k <;> simp [h1] at ht
            | known A =>
              cases k with
              | unknown => cases A <;> simp [h1] at ht
              | known S =>
                cases A with
                | cons _ _ => cases S <;> simp [h1] at ht
                | nil =>
                  cases S with
                  | cons _ _ => simp [h1] at ht
                  | nil =>
                    simp [h1] at ht; subst ht
                    cases va with
                    | num v => exact absurd hva (by simp [HasTy])
                    | pair _ _ => exact absurd hva (by simp [HasTy])
                    | kn c ts =>
                      have hts : ts = [] := by
                        cases ts with
                        | nil => rfl
                        | cons t ts' =>
                          have := hva t (by simp)
                          cases hb : t.kn <;> simp [hb, inSet] at this
                      subst hts
                      exact Or.inr ⟨.num c, by simp [eval, ea], trivial, hnum c⟩

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

/-- (b, exact) A `requireExact` the checker accepts never fails as imprecise. -/
theorem requireExact_never_imprecise (orc : Nat → Val → Val) (hOrc : OrcOK orc)
    (fns : List Expr) (occ : List (Nat × Int × Int)) (hc : Consistent occ)
    (hfns : ∀ body ∈ fns, ∀ k ∈ knights body, k ∈ occ)
    (fuel : Nat) (p : List Nat) (a : Expr) (t : Ty)
    (hk : ∀ k ∈ knights a, k ∈ occ)
    (ht : typeOf fns fuel [] p (.requireExact a) = some t) :
    eval orc fns fuel [] p (.requireExact a) ≠ .error .imprecise := by
  rcases sc_sound orc hOrc fns occ hc hfns fuel [] [] p (.requireExact a) t EnvRel.nil
      (by simpa [knights] using hk) ht with h | ⟨v, h, _, _⟩
  · rw [h]; intro h'; cases h'
  · rw [h]; intro h'; cases h'

/-- (b, E314) A program whose named-bound table is consistent never meets two
    runtime terms of one named Knightian source with different bounds. -/
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

/-- Per-field projection: the type of `fst e` is exactly the type of the first
    field, not a join over the aggregate. -/
theorem proj_carries_field_set (fns : List Expr) (fuel : Nat) (Γ : List (Nat × Ty))
    (p : List Nat) (e : Expr) (t u : Ty)
    (h : typeOf fns fuel Γ p e = some (.pair t u)) :
    typeOf fns (fuel + 1) Γ p (.fst e) = some t ∧ typeOf fns (fuel + 1) Γ p (.snd e) = some u := by
  constructor <;> simp [typeOf, h]

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
def g5 : Expr := .gauss 5 2

/-- Repeated identity calls preserve one shared source: `id(k) - id(k)` carries
    two terms of the one source `named 3`, which cancel coefficient-wise. -/
example : eval oracle wFns 10 [] [] (.sub (.call 1 0 k3) (.call 2 0 k3)) =
    .ok (.kn 0 [{ src := .named 3, lo := 0, hi := 1, c := 1, kn := true },
                { src := .named 3, lo := 0, hi := 1, c := -1, kn := true }]) := by rfl

/-- Two calls that create fresh residues do not cancel: the residues of
    `sq(k)` at call sites 1 and 2 are distinct sources. -/
example : Src.fresh 7 [1] ≠ Src.fresh 7 [2] := by decide

/-- The A and K sets are kept apart: a Knightian and an aleatory source. -/
example : typeOf wFns 10 [] [] (.add k3 g5) =
    some (.kn (.known [.named 5]) (.known [.named 3])) := by rfl

/-- `require` refuses a Knightian value and accepts an aleatory one (a GUM
    uncertainty is a precise bound); `requireExact` refuses both. -/
example : typeOf wFns 10 [] [] (.require k3) = none := by rfl
example : typeOf wFns 10 [] [] (.require g5) = some .num := by rfl
example : typeOf wFns 10 [] [] (.requireExact g5) = none := by rfl
example : typeOf wFns 10 [] [] (.requireExact (.lit 4)) = some .num := by rfl

/-- The square of an aleatory value has an aleatory residue only: its K set
    stays empty, so `require` still accepts it. -/
example : typeOf wFns 10 [] [] (.require (.call 1 1 g5)) = some .num := by rfl

/-- The square of a Knightian value puts its residue in the K set too. -/
example : typeOf wFns 10 [] [] (.require (.call 1 1 k3)) = none := by rfl

/-- Per-field projection: field 0 is Knightian, field 1 is a literal; the
    projection of field 1 satisfies `require`, field 0 does not. -/
def rec01 : Expr := .pair k3 (.lit 7)
example : typeOf wFns 10 [] [] (.require (.snd rec01)) = some .num := by rfl
example : typeOf wFns 10 [] [] (.require (.fst rec01)) = none := by rfl
example : eval oracle wFns 10 [] [] (.require (.snd rec01)) = .ok (.num 7) := by rfl

/-- An imported named source with the same bounds agrees; with conflicting
    bounds the table is refused (E314). -/
example : consistentB [(3, 0, 1), (3, 0, 1)] = true := by decide
example : consistentB [(3, 0, 1), (3, 2, 3)] = false := by decide

/-- Reversed bounds are refused by the checker. -/
example : typeOf wFns 10 [] [] (.knight 3 2 1) = none := by rfl

end SCCore
