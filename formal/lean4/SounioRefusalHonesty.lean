/-
  SounioRefusalHonesty — a well-typed program produces a value or refuses;
  it never fabricates a zero.

  STATUS (2026-08-17): NO `sorry`. Mathlib-free. Kernel-checked under
  leanprover/lean4:v4.33.0. No `native_decide`. The thesis theorems
  (`refusal_is_not_zero`, `well_typed_value_or_refuse`,
  `unimplemented_call_not_zero`, `fabricate_iff_unimplemented`,
  `check_deterministic`, `legacy_total`) depend on NO axioms.
  `honest_agree` / `unimplemented_disagrees` use `propext` only.

  THIS IS THE CANDIDATE THAT WAS REACHABLE. Of the three compiler-level
  firsts proposed (refusal soundness; GUM-through-MLI variance at codegen;
  associator as a graded invariant of lowering), this is the one that has
  a discrete, already-implemented decision in the compiler and does not
  require a model of x86 emission that does not exist. The other two are
  named as out of scope at the bottom, not as future work disguised as
  progress.

  WHAT WOULD BE A FIRST, AND WHAT THIS FILE ACTUALLY IS.
  No mainstream compiler treats "I decline to answer" as a first-class
  outcome of compilation: LLVM, GCC and Cranelift will emit *something*
  for an unresolved external — a relocation, a stub, a zeroed register —
  and the program continues. P0-F / #1622 made Madaros do the other
  thing: a call to an `extern "C"` name the native backend does not
  implement is E219 at the CALL, and the program does not become a
  binary that returns 0. The historical behaviour, documented in
  `tests/compile-fail/extern_c_unimplemented_builtin.sio` and in
  `self-hosted/check/check.sio:13807-13809`, was: body-less extern →
  empty stub → every call silently reads 0. `malloc(64)` was a null
  pointer; `abs(-7)` was 0; both with exit status 0 and no diagnostic.

  A Lean proof that the *current* judgment never returns FabricateZero
  is tautological if FabricateZero is not a constructor of that
  judgment. The content is a CORRESPONDENCE between two relations:

    Legacy  — the historical backend: unimplemented call ↦ 0, and
              arithmetic continues (`abs(-7) + 1` is 1).
    Check   — the E219 judgment: unimplemented call ↦ Refuse, and
              Refuse infects every context (`abs(-7) + 1` is Refuse).

  The distinguishing program is `abs + 1`. Legacy gives 1. Check
  refuses. Those are not the same observation, and the theorem that
  says so is the formal statement of "refusal is not a zero".

  SCOPE — read before quoting.
  (a) This is a MODEL of the E219 fragment. The allow-list below is a
      snapshot of `name_is_native_backend_builtin` dated 2026-08-17
      (34 names: 27 original builtins + 7 P0-F POSIX). Drift is a CI
      concern (`scripts/ci/extern_builtin_mirror_gate.sh`), not a
      theorem.
  (b) Allow-listed calls are modelled as returning the dummy `1`, not
      as the actual syscall. This file does not prove that `getpid`
      returns a pid. Execution witnesses in CI own that.
  (c) The expression language is `lit | add | call`. Enough to make
      refuse-versus-zero distinguishable, not a model of Sounio.
  (d) Typing is "the name is declared", matching the compiler: E219
      fires at the CALL, never at the declaration
      (`ffi_libm_call.sio` declares `tgamma` and calls only `sqrt`).
  (e) This is not Wright–Felleisen progress. Progress says value-or-
      step. Honesty says value-or-refuse. A program that is well-typed
      and mentions an unimplemented extern does not get stuck at 0;
      it is refused.
  (f) lean_single does not implement E219. The theorem is a statement
      about the Madaros judgment, not the seed compiler.

  IMPLEMENTATION CORRESPONDENCE (2026-08-17, after the measured gap).
  An audit against `origin/main` found three divergences. Two are closed
  in the compiler; one is named and left.

  Closed:
    · Expression infection. After E219 the checker now returns `ty_error()`
      (`refused_unimplemented`) instead of the declared return type, so
      `abs() + 1` is not typed as an integer. That is `add_refuse_l`.
    · Live Legacy. An empty IR stub that is not a builtin and not BSS
      used to compile as prologue+ret (call reads rax, usually 0).
      `native_v2_empty_stub_would_fabricate` now emits `ud2` instead.
      Declaration-without-call still has a symbol; a missed E219 cannot
      return a fabricated 0. Gate:
      `scripts/ci/e219_refusal_correspondence_gate.sh`.

  Still open, stated plainly:
    · lean_single has no E219, and a seed E219 written in house
      style would be ornament. `strip_extern_blocks` rewrites
      `extern "C"` into ordinary `fn` stubs before check, so the
      Madaros predicate (`is_extern && !allowlisted`) has no object.
      The distinguishing name of this file, `abs`, is a first-class
      stub (`__native_abs_i64`). Measured 2026-08-17:
      `bin/souc-lean-single-x86_64` emits an ELF for
      `extern_c_unimplemented_builtin.sio` (compile exit 0) and
      `abs(-7)` returns 7. `tc_error` is a warning (253 sites) and
      never sets `TYPECHECK_FAILED`; `tc_error_hard` goes through
      `tc_mark_failed`, which swallows `from_import` (bit 2048).
      Only arity and unbalanced braces punch through. A real refuse
      would need a surviving unresolved-extern bit, a CALL-site
      check, an unconditional `TYPECHECK_FAILED`, and a second
      allow-list — a different judgment, not this one. Left open
      as the #1798 engine-split.

  WHY NOT THE OTHER TWO CANDIDATES, STATED PLAINLY.
  GUM-through-MLI: `gum_conservativity` is already proved for a sketch
  calculus in `EpistemicEffects.lean`; the first would be that the
  *emitted* propagation matches the analytic rule. That needs a model
  of MLI lowering to machine code. `self-hosted/mli/**` is under an
  active claim by another lane, and no such model exists here.
  Associator-under-lowering: the algebra side is theorem-backed
  (`SounioSedenionAssociator1848`, `SounioSkewCategory`); the compiler
  side (associator-aware lowering of `oct_mul`/`sed_mul`) is not a
  Lean object. Connecting them would be inventing the lowering, not
  proving an invariant of it.

  Compiler sources this models:
    self-hosted/check/check.sio          E219 at 7334, 7422, 21053
    self-hosted/check/check.sio:13801    the allow-list and the fabricate note
    tests/compile-fail/extern_c_unimplemented_builtin.sio
-/
set_option linter.unusedSimpArgs false

namespace Sounio.RefusalHonesty

/-! ## §1. The allow-list — snapshot of `name_is_native_backend_builtin`

    Adding a name here, in the model, is the same act as adding it in
    `check.sio`: it REMOVES the refuse guard. If the model allow-lists a
    name the backend cannot emit, the honesty theorem would still hold
    *of the model* and would be a lie about the compiler. The list is
    therefore copied, not invented. -/

def allowlisted (n : String) : Bool :=
  n == "print_int" || n == "print_char" || n == "print" ||
  n == "get_arg_count" || n == "get_arg" ||
  n == "str_len" || n == "str_eq" || n == "str_slice" ||
  n == "starts_with" || n == "str_concat" ||
  n == "read_file" || n == "write_file" || n == "file_size" ||
  n == "sqrt" || n == "print_f64" ||
  n == "exp" || n == "log" || n == "sin" || n == "cos" ||
  n == "str_char_at" || n == "assert" || n == "str_from_bytes" ||
  n == "heap_alloc" || n == "heap_free" ||
  n == "f64_to_bits" || n == "bits_to_f64" || n == "syscall6" ||
  n == "malloc" || n == "free" ||
  n == "getpid" || n == "getppid" || n == "exit" || n == "abort" ||
  n == "system"

/-- Dummy result of an allow-listed call. Distinguishes "the backend
    returned something" from "the stub left 0 in rax". Not a syscall. -/
def oracle (_n : String) : Int := 1

/-! ## §2. The E219 fragment -/

inductive Expr where
  | lit  : Int → Expr
  | add  : Expr → Expr → Expr
  | call : String → Expr
  deriving DecidableEq, Repr

inductive Ty where
  | tint
  deriving DecidableEq, Repr

/-- First-class observations of the CURRENT compiler. There is no
    `fabricate` constructor. That is the design, not a gap: fabrication
    is not an outcome this judgment can name. It lives on `Legacy`. -/
inductive Observation where
  | value  : Int → Observation
  | refuse : Observation
  deriving DecidableEq, Repr

/-- A name is declared (an `extern "C"` signature exists). Declaration
    is not implementation. -/
abbrev Decls := List String

inductive HasTy : Decls → Expr → Ty → Prop where
  | lit  : ∀ Δ n, HasTy Δ (.lit n) .tint
  | add  : ∀ Δ a b,
      HasTy Δ a .tint → HasTy Δ b .tint → HasTy Δ (.add a b) .tint
  | call : ∀ Δ n, n ∈ Δ → HasTy Δ (.call n) .tint

/-! ## §3. Two compilation relations -/

/-- The E219 judgment. Unimplemented calls refuse; refuse infects add. -/
inductive Check : Expr → Observation → Prop where
  | lit : ∀ n, Check (.lit n) (.value n)
  | add_v : ∀ a b n m,
      Check a (.value n) → Check b (.value m) →
      Check (.add a b) (.value (n + m))
  | add_refuse_l : ∀ a b,
      Check a .refuse → Check (.add a b) .refuse
  | add_refuse_r : ∀ a b n,
      Check a (.value n) → Check b .refuse → Check (.add a b) .refuse
  | call_ok : ∀ n, allowlisted n = true →
      Check (.call n) (.value (oracle n))
  | call_refuse : ∀ n, allowlisted n = false →
      Check (.call n) .refuse

/-- The historical backend. Unimplemented calls become 0 and arithmetic
    continues. This relation cannot refuse: every expression produces an
    integer. That is what "fabricate" means. -/
inductive Legacy : Expr → Int → Prop where
  | lit : ∀ n, Legacy (.lit n) n
  | add : ∀ a b n m,
      Legacy a n → Legacy b m → Legacy (.add a b) (n + m)
  | call_ok : ∀ n, allowlisted n = true →
      Legacy (.call n) (oracle n)
  | call_fab : ∀ n, allowlisted n = false →
      Legacy (.call n) 0

/-! ## §4. Allow-list facts, by `decide` — no `native_decide` -/

theorem sqrt_allowlisted : allowlisted "sqrt" = true := by decide
theorem malloc_allowlisted : allowlisted "malloc" = true := by decide
theorem abs_not_allowlisted : allowlisted "abs" = false := by decide
theorem tgamma_not_allowlisted : allowlisted "tgamma" = false := by decide
theorem fabs_not_allowlisted : allowlisted "fabs" = false := by decide

theorem oracle_ne_zero (n : String) : oracle n ≠ 0 := by
  unfold oracle
  decide

/-! ## §5. Inversion and determinism -/

theorem check_lit_inv {n : Int} {o : Observation}
    (h : Check (.lit n) o) : o = .value n := by
  cases h; rfl

theorem check_call_inv {n : String} {o : Observation}
    (h : Check (.call n) o) :
    (allowlisted n = true ∧ o = .value (oracle n)) ∨
    (allowlisted n = false ∧ o = .refuse) := by
  cases h with
  | call_ok _ hok => exact Or.inl ⟨hok, rfl⟩
  | call_refuse _ href => exact Or.inr ⟨href, rfl⟩

theorem legacy_call_inv {n : String} {k : Int}
    (h : Legacy (.call n) k) :
    (allowlisted n = true ∧ k = oracle n) ∨
    (allowlisted n = false ∧ k = 0) := by
  cases h with
  | call_ok _ hok => exact Or.inl ⟨hok, rfl⟩
  | call_fab _ hf => exact Or.inr ⟨hf, rfl⟩

theorem check_deterministic {e : Expr} {o₁ o₂ : Observation}
    (h₁ : Check e o₁) (h₂ : Check e o₂) : o₁ = o₂ := by
  induction h₁ generalizing o₂ with
  | lit n =>
    cases h₂; rfl
  | add_v a b n m ha hb iha ihb =>
    cases h₂ with
    | add_v _ _ n' m' ha' hb' =>
      have hn : Observation.value n = .value n' := iha ha'
      have hm : Observation.value m = .value m' := ihb hb'
      injection hn with hn'; injection hm with hm'
      subst hn'; subst hm'; rfl
    | add_refuse_l _ _ ha' =>
      have : Observation.value n = .refuse := iha ha'
      cases this
    | add_refuse_r _ _ n' ha' hb' =>
      have : Observation.value m = .refuse := ihb hb'
      cases this
  | add_refuse_l a b ha iha =>
    cases h₂ with
    | add_v _ _ n m ha' _ =>
      have : Observation.refuse = .value n := iha ha'
      cases this
    | add_refuse_l _ _ _ => rfl
    | add_refuse_r _ _ n ha' _ =>
      have : Observation.refuse = .value n := iha ha'
      cases this
  | add_refuse_r a b n ha hb iha ihb =>
    cases h₂ with
    | add_v _ _ n' m ha' hb' =>
      have : Observation.refuse = .value m := ihb hb'
      cases this
    | add_refuse_l _ _ ha' =>
      have : Observation.value n = .refuse := iha ha'
      cases this
    | add_refuse_r _ _ _ _ _ => rfl
  | call_ok n hok =>
    cases h₂ with
    | call_ok _ _ => rfl
    | call_refuse _ href =>
      have : true = false := Eq.trans hok.symm href
      cases this
  | call_refuse n href =>
    cases h₂ with
    | call_ok _ hok =>
      have : false = true := Eq.trans href.symm hok
      cases this
    | call_refuse _ _ => rfl

theorem legacy_deterministic {e : Expr} {k₁ k₂ : Int}
    (h₁ : Legacy e k₁) (h₂ : Legacy e k₂) : k₁ = k₂ := by
  induction h₁ generalizing k₂ with
  | lit n => cases h₂; rfl
  | add a b n m ha hb iha ihb =>
    cases h₂ with
    | add _ _ n' m' ha' hb' =>
      have hn := iha ha'; have hm := ihb hb'
      subst hn; subst hm; rfl
  | call_ok n hok =>
    cases h₂ with
    | call_ok _ _ => rfl
    | call_fab _ href =>
      have : true = false := Eq.trans hok.symm href
      cases this
  | call_fab n href =>
    cases h₂ with
    | call_ok _ hok =>
      have : false = true := Eq.trans href.symm hok
      cases this
    | call_fab _ _ => rfl

/-! ## §6. The honesty theorems

    Three statements, in increasing strength.

    1. An unimplemented call refuses, and that refusal is not the
       observation `value 0`.
    2. The historical backend fabricates 0 on the same call, and then
       continues; `abs + 1` is 1 under Legacy and Refuse under Check.
    3. Every well-typed term in the fragment is `value n` or `refuse`
       under Check — never stuck, never fabricated. -/

theorem unimplemented_call_refuses (n : String)
    (h : allowlisted n = false) :
    Check (.call n) .refuse :=
  .call_refuse n h

theorem unimplemented_call_not_zero (n : String)
    (h : allowlisted n = false) :
    ¬ Check (.call n) (.value 0) := by
  intro hv
  have hdet := check_deterministic (unimplemented_call_refuses n h) hv
  cases hdet

theorem legacy_fabricates (n : String)
    (h : allowlisted n = false) :
    Legacy (.call n) 0 :=
  .call_fab n h

/-- Coverage: every name the historical backend would fabricate, the
    current judgment refuses. This is the E219 correspondence, and it
    is the non-tautological content: two relations, one constructor of
    each, tied by the same allow-list predicate. -/
theorem refuse_covers_fabrication (n : String)
    (_hL : Legacy (.call n) 0) (hA : allowlisted n = false) :
    Check (.call n) .refuse :=
  unimplemented_call_refuses n hA

/-- Fabrication under Legacy is exactly the unimplemented names, because
    the oracle is never 0. If a future revision set `oracle n = 0` for
    some allow-listed n, this biconditional would fail — correctly,
    because then Legacy could not tell a real zero from a stub. -/
theorem fabricate_iff_unimplemented (n : String) :
    Legacy (.call n) 0 ↔ allowlisted n = false := by
  constructor
  · intro h
    rcases legacy_call_inv h with hok | hfab
    · have hne : oracle n ≠ 0 := oracle_ne_zero n
      exact absurd hok.2.symm hne
    · exact hfab.1
  · exact legacy_fabricates n

/-! ## §7. The distinguishing program: `abs + 1`

    Under a compiler that stubs to zero, `abs(-7) + 1` is 1 and the
    process exits 0. Under E219 the whole program is refused. The two
    observations are not equal, so refusal is not a zero. -/

def absCall : Expr := .call "abs"
def absPlusOne : Expr := .add absCall (.lit 1)

theorem abs_plus_one_legacy : Legacy absPlusOne 1 :=
  .add _ _ 0 1 (.call_fab "abs" abs_not_allowlisted) (.lit 1)

theorem abs_plus_one_check : Check absPlusOne .refuse :=
  .add_refuse_l _ _ (.call_refuse "abs" abs_not_allowlisted)

theorem abs_plus_one_not_one : ¬ Check absPlusOne (.value 1) := by
  intro hv
  have hdet := check_deterministic abs_plus_one_check hv
  cases hdet

/-- The gap, as a single statement: the same closed program is `1`
    under the historical backend and `refuse` under E219. A compiler
    that treated refuse as zero would make these coincide. -/
theorem refusal_is_not_zero :
    Legacy absPlusOne 1 ∧ Check absPlusOne .refuse ∧
      ¬ Check absPlusOne (.value 1) ∧ ¬ Check absPlusOne (.value 0) :=
  ⟨abs_plus_one_legacy, abs_plus_one_check, abs_plus_one_not_one, by
    intro hv
    have hdet := check_deterministic abs_plus_one_check hv
    cases hdet⟩

/-! ## §8. Declaration is not a call

    The compiler types an unimplemented extern that is never called.
    Honesty is a property of evaluation, not of the signature table. -/

theorem declare_abs_ok : HasTy ["abs"] absCall .tint :=
  .call _ _ (List.Mem.head _)

theorem declare_tgamma_ok : HasTy ["tgamma", "sqrt"] (.call "tgamma") .tint :=
  .call _ _ (List.Mem.head _)

theorem declare_and_call_sqrt :
    HasTy ["tgamma", "sqrt"] (.call "sqrt") .tint :=
  .call _ _ (List.Mem.tail _ (List.Mem.head _))

theorem declared_unimplemented_refuses :
    HasTy ["abs"] absCall .tint ∧ Check absCall .refuse :=
  ⟨declare_abs_ok, unimplemented_call_refuses "abs" abs_not_allowlisted⟩

theorem declared_implemented_values :
    HasTy ["sqrt"] (.call "sqrt") .tint ∧
      Check (.call "sqrt") (.value 1) :=
  ⟨.call _ _ (List.Mem.head _), .call_ok "sqrt" sqrt_allowlisted⟩

/-! ## §9. Honesty progress: well-typed ⇒ value ∨ refuse

    Every well-typed term of the fragment has a Check observation, and
    that observation is never a hidden third thing. Combined with
    determinism, this is the formal statement of epistemic honesty at
    the language level for this fragment: the compiler either produces
    a value or declines; it does not fabricate. -/

theorem check_total_on_typed {Δ : Decls} {e : Expr} {T : Ty}
    (h : HasTy Δ e T) :
    Check e .refuse ∨ ∃ n, Check e (.value n) := by
  induction h with
  | lit n =>
    exact Or.inr ⟨n, .lit n⟩
  | add a b ha hb iha ihb =>
    rcases iha with ra | ⟨n, va⟩
    · exact Or.inl (.add_refuse_l a b ra)
    · rcases ihb with rb | ⟨m, vb⟩
      · exact Or.inl (.add_refuse_r a b n va rb)
      · exact Or.inr ⟨n + m, .add_v a b n m va vb⟩
  | call n hn =>
    cases hA : allowlisted n
    · exact Or.inl (.call_refuse n hA)
    · exact Or.inr ⟨oracle n, .call_ok n hA⟩

/-- Named form of the thesis, restricted to the fragment the file
    actually models. A well-typed program produces a value or refuses.
    It does not produce a fabricated zero: the only way Check says
    `value 0` is if the program computed 0 from literals and allow-
    listed calls, which the oracle never contributes. -/
theorem well_typed_value_or_refuse {Δ : Decls} {e : Expr}
    (h : HasTy Δ e .tint) :
    (∃ n, Check e (.value n)) ∨ Check e .refuse := by
  rcases check_total_on_typed h with href | hv
  · exact Or.inr href
  · exact Or.inl hv

/-! ## §10. The two relations coincide exactly when there is nothing
    to refuse. If they coincided always, E219 would be ornament. -/

def mentionsUnimplemented : Expr → Bool
  | .lit _ => false
  | .add a b => mentionsUnimplemented a || mentionsUnimplemented b
  | .call n => !allowlisted n

theorem check_value_to_legacy {e : Expr} {n : Int}
    (h : Check e (.value n)) : Legacy e n := by
  induction e generalizing n with
  | lit k =>
    have hk := check_lit_inv h
    injection hk with hk'
    rw [hk']
    exact .lit k
  | add a b iha ihb =>
    cases h with
    | add_v _ _ n' m' ha hb =>
      exact .add a b n' m' (iha ha) (ihb hb)
  | call k =>
    rcases check_call_inv h with hok | href
    · injection hok.2 with hn
      rw [hn]
      exact .call_ok k hok.1
    · cases href.2

theorem honest_agree {e : Expr} (h : mentionsUnimplemented e = false) :
    ∀ n, Check e (.value n) ↔ Legacy e n := by
  induction e with
  | lit k =>
    intro n
    constructor
    · exact check_value_to_legacy
    · intro hL
      cases hL
      exact .lit k
  | add a b iha ihb =>
    intro n
    simp [mentionsUnimplemented] at h
    have ha : mentionsUnimplemented a = false := by
      cases ha : mentionsUnimplemented a
      · rfl
      · simp [ha] at h
    have hb : mentionsUnimplemented b = false := by
      cases hb : mentionsUnimplemented b
      · rfl
      · simp [ha, hb] at h
    constructor
    · exact check_value_to_legacy
    · intro hL
      cases hL with
      | add _ _ n' m' hLa hLb =>
        have hCa : Check a (.value n') := (iha ha n').mpr hLa
        have hCb : Check b (.value m') := (ihb hb m').mpr hLb
        exact .add_v a b n' m' hCa hCb
  | call k =>
    intro n
    simp [mentionsUnimplemented] at h
    have hA : allowlisted k = true := by
      cases hA : allowlisted k
      · simp [hA] at h
      · rfl
    constructor
    · exact check_value_to_legacy
    · intro hL
      rcases legacy_call_inv hL with hok | hfab
      · have : n = oracle k := hok.2
        subst this
        exact .call_ok k hA
      · have : true = false := Eq.trans hA.symm hfab.1
        cases this

/-- Legacy is total on the fragment: every expression produces an integer.
    That totality *is* the fabrication bug — there is no hole where a
    missing implementation could refuse. -/
theorem legacy_total (e : Expr) : ∃ k, Legacy e k := by
  induction e with
  | lit n => exact ⟨n, .lit n⟩
  | add a b iha ihb =>
    rcases iha with ⟨n, hn⟩
    rcases ihb with ⟨m, hm⟩
    exact ⟨n + m, .add a b n m hn hm⟩
  | call k =>
    cases hA : allowlisted k
    · exact ⟨0, .call_fab k hA⟩
    · exact ⟨oracle k, .call_ok k hA⟩

/-- When the program mentions an unimplemented name, the relations
    disagree on the distinguishing observation: Legacy still has an
    integer, Check refuses. -/
theorem unimplemented_disagrees {e : Expr}
    (h : mentionsUnimplemented e = true) :
    Check e .refuse ∧ ∃ k, Legacy e k := by
  induction e with
  | lit _ =>
    simp [mentionsUnimplemented] at h
  | add a b iha ihb =>
    simp [mentionsUnimplemented] at h
    have hor : mentionsUnimplemented a = true ∨ mentionsUnimplemented b = true := by
      cases ha : mentionsUnimplemented a <;> cases hb : mentionsUnimplemented b
      · simp [ha, hb] at h
      · exact Or.inr rfl
      · exact Or.inl rfl
      · exact Or.inl rfl
    rcases hor with ha | hb
    · have ⟨href, _⟩ := iha ha
      rcases legacy_total (.add a b) with ⟨k, hk⟩
      exact ⟨.add_refuse_l a b href, ⟨k, hk⟩⟩
    · rcases legacy_total a with ⟨n, hLa⟩
      rcases legacy_total b with ⟨m, hLb⟩
      have hLab : Legacy (.add a b) (n + m) := .add a b n m hLa hLb
      have ⟨hrefB, _⟩ := ihb hb
      cases hA : mentionsUnimplemented a
      · have hCa : Check a (.value n) := (honest_agree hA n).mpr hLa
        exact ⟨.add_refuse_r a b n hCa hrefB, ⟨n + m, hLab⟩⟩
      · have ⟨hrefA, _⟩ := iha hA
        exact ⟨.add_refuse_l a b hrefA, ⟨n + m, hLab⟩⟩
  | call k =>
    simp [mentionsUnimplemented] at h
    have hA : allowlisted k = false := by
      cases hA : allowlisted k
      · rfl
      · simp [hA] at h
    exact ⟨.call_refuse k hA, ⟨0, .call_fab k hA⟩⟩

end Sounio.RefusalHonesty
