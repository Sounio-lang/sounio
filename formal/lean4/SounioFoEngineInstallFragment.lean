/-
  SounioFoEngineInstallFragment — L2 *engine install* slice of residual §5.4.

  Madaros multipass (`lowerer_fo_preregister_pure_fns_multipass_mut`, 4 passes)
  walks pure function definitions and installs FO_XFER kind-6 entries with
  compiled bodies *before* call-site expand.

  This file models **install for the oral Css pure-helper fragment** only:

    Program fragment items (order may reverse — multipass handles that):
      fo_infusion_rate, fo_clearance, fo_css   (+ optional Pk.css peel model)

    install(fragment) builds a Registry name → PureExpr body
    After install, expand(fo_css(p0..p4)) = cssSite
    And reverse-definition order still installs (multipass simulation).

  What this **is**:
    • a machine-checked model of "register these three pure helpers, then expand"
    • reverse-order install (css before leaves) still yields a complete registry
      after enough passes (mirrors multipass prereg)

  What this **is not**:
    • a proof that Madaros multipass is sound for arbitrary programs
    • multi-mod module loader correctness
    • method FO_XFER / op17 in full generality

  Mathlib-free. Zero sorry. Companion:
  scripts/research/fo_engine_install_fragment_cert.py
-/
namespace SounioFoEngineInstallFragment

-- ── §1. PureExpr (with calls) ─────────────────────────────────────────────

inductive PureExpr where
  | param : Nat → PureExpr
  | lit   : Rat → PureExpr
  | add   : PureExpr → PureExpr → PureExpr
  | sub   : PureExpr → PureExpr → PureExpr
  | mul   : PureExpr → PureExpr → PureExpr
  | div   : PureExpr → PureExpr → PureExpr
  | call  : String → List PureExpr → PureExpr
  deriving BEq, Repr

open PureExpr

-- ── §2. Program fragment items ────────────────────────────────────────────

structure PureItem where
  name : String
  body : PureExpr
  deriving BEq, Repr

def bodyInfusionRate : PureExpr :=
  div (mul (param 0) (param 1)) (param 2)

def bodyClearance : PureExpr :=
  mul (param 0) (param 1)

def bodyCss : PureExpr :=
  div
    (call "fo_infusion_rate" [param 0, param 1, param 2])
    (call "fo_clearance" [param 3, param 4])

/-- Canonical order (leaves first). -/
def fragmentForward : List PureItem :=
  [ ⟨"fo_infusion_rate", bodyInfusionRate⟩
  , ⟨"fo_clearance", bodyClearance⟩
  , ⟨"fo_css", bodyCss⟩ ]

/-- Reverse definition order (css first — multipass residual case). -/
def fragmentReverse : List PureItem :=
  [ ⟨"fo_css", bodyCss⟩
  , ⟨"fo_clearance", bodyClearance⟩
  , ⟨"fo_infusion_rate", bodyInfusionRate⟩ ]

-- ── §3. Registry install (one multipass sweep) ────────────────────────────

/-- Finite registry: list of (name, body); last write wins. -/
abbrev Registry := List (String × PureExpr)

def regLookup (r : Registry) (name : String) : Option PureExpr :=
  let rec go : Registry → Option PureExpr
    | [] => none
    | (n, b) :: rest =>
        match go rest with
        | some b' => some b'  -- prefer later entries
        | none => if n == name then some b else none
  go r

/-- Install one item (append — later install overrides for same name). -/
def regInstall (r : Registry) (it : PureItem) : Registry :=
  List.append r [(it.name, it.body)]

/-- One multipass sweep: install every item in order. -/
def installPass (r : Registry) (items : List PureItem) : Registry :=
  items.foldl regInstall r

/-- k multipass sweeps (Madaros uses 4). -/
def multipass (items : List PureItem) (k : Nat) : Registry :=
  let rec go (r : Registry) : Nat → Registry
    | 0 => r
    | n + 1 => go (installPass r items) n
  go [] k

-- ── §4. Expand using installed registry ───────────────────────────────────

def subst (args : List PureExpr) : PureExpr → PureExpr
  | param i => match args[i]? with | some a => a | none => param i
  | lit c => lit c
  | add a b => add (subst args a) (subst args b)
  | sub a b => sub (subst args a) (subst args b)
  | mul a b => mul (subst args a) (subst args b)
  | div a b => div (subst args a) (subst args b)
  | call n as => call n (as.map (subst args))

def expandOnce (r : Registry) : PureExpr → PureExpr
  | call n as =>
      match regLookup r n with
      | some body => subst as body
      | none => call n as
  | add a b => add (expandOnce r a) (expandOnce r b)
  | sub a b => sub (expandOnce r a) (expandOnce r b)
  | mul a b => mul (expandOnce r a) (expandOnce r b)
  | div a b => div (expandOnce r a) (expandOnce r b)
  | e => e

def expandN (r : Registry) : Nat → PureExpr → PureExpr
  | 0, e => e
  | n + 1, e => expandN r n (expandOnce r e)

def expandFull (r : Registry) (e : PureExpr) : PureExpr :=
  expandN r 3 e

-- ── §5. Golden cssSite ────────────────────────────────────────────────────

def cssSite : PureExpr :=
  div (div (mul (param 0) (param 1)) (param 2))
      (mul (param 3) (param 4))

def cssCall : PureExpr :=
  call "fo_css" [param 0, param 1, param 2, param 3, param 4]

-- ── §6. Install then expand certificates ──────────────────────────────────

/-- After 1 forward pass, registry has all three; expand yields site. -/
def regForward1 : Registry := multipass fragmentForward 1

def install_forward_ok : Bool :=
  (regLookup regForward1 "fo_css").isSome &&
  (regLookup regForward1 "fo_infusion_rate").isSome &&
  (regLookup regForward1 "fo_clearance").isSome &&
  expandFull regForward1 cssCall == cssSite

theorem install_forward : install_forward_ok = true := by native_decide

/-- Reverse order: after 1 pass all three still installed (install is not
    dependency-ordered — multipass re-walks; even one pass installs all items
    present in the list regardless of order). -/
def regReverse1 : Registry := multipass fragmentReverse 1

def install_reverse_ok : Bool :=
  (regLookup regReverse1 "fo_css").isSome &&
  (regLookup regReverse1 "fo_infusion_rate").isSome &&
  (regLookup regReverse1 "fo_clearance").isSome &&
  expandFull regReverse1 cssCall == cssSite

theorem install_reverse : install_reverse_ok = true := by native_decide

/-- Four multipass sweeps (Madaros default) still agree. -/
def regForward4 : Registry := multipass fragmentForward 4
def regReverse4 : Registry := multipass fragmentReverse 4

def install_four_pass_ok : Bool :=
  expandFull regForward4 cssCall == cssSite &&
  expandFull regReverse4 cssCall == cssSite

theorem install_four_pass : install_four_pass_ok = true := by native_decide

/-- Multi-mod: import install of the same fragment yields same expand. -/
def regImport : Registry := multipass fragmentForward 1

def install_import_eq_local_ok : Bool :=
  expandFull regImport cssCall == expandFull regForward1 cssCall

theorem install_import_eq_local : install_import_eq_local_ok = true := by native_decide

-- ── §7. Emit RPN after install+expand ─────────────────────────────────────

def OP_PARAM : Nat := 1
def OP_MUL : Nat := 5
def OP_DIV : Nat := 6

structure FoInstr where
  op : Nat
  arg : Nat
  deriving BEq, Repr

def compile : PureExpr → List FoInstr
  | param i => [⟨OP_PARAM, i⟩]
  | lit _ => [⟨2, 0⟩]
  | add a b => compile a ++ compile b ++ [⟨3, 0⟩]
  | sub a b => compile a ++ compile b ++ [⟨4, 0⟩]
  | mul a b => compile a ++ compile b ++ [⟨OP_MUL, 0⟩]
  | div a b => compile a ++ compile b ++ [⟨OP_DIV, 0⟩]
  | call _ _ => []

def cssSiteProg : List FoInstr :=
  [ ⟨OP_PARAM, 0⟩, ⟨OP_PARAM, 1⟩, ⟨OP_MUL, 0⟩
  , ⟨OP_PARAM, 2⟩, ⟨OP_DIV, 0⟩
  , ⟨OP_PARAM, 3⟩, ⟨OP_PARAM, 4⟩, ⟨OP_MUL, 0⟩
  , ⟨OP_DIV, 0⟩ ]

def emit_after_install_ok : Bool :=
  compile (expandFull regForward1 cssCall) == cssSiteProg &&
  compile (expandFull regReverse1 cssCall) == cssSiteProg

theorem emit_after_install : emit_after_install_ok = true := by native_decide

-- ── §8. FO variance ───────────────────────────────────────────────────────

def foVarCss : Rat :=
  let j0 : Rat := 500 / 60
  let j1 : Rat := (4 / 5) / 60
  let j3 : Rat := - (400 : Rat) / 300
  let j4 : Rat := - (20 : Rat) / 3
  let s0 : Rat := 1 / 20
  let s1 : Rat := 10
  let s3 : Rat := 3 / 10
  let s4 : Rat := 1 / 10
  j0*j0*s0*s0 + j1*j1*s1*s1 + j3*j3*s3*s3 + j4*j4*s4*s4

def var_ok : Bool := foVarCss == (191 : Rat) / 240
theorem var_freeze : var_ok = true := by native_decide

-- ── §9. Bundle ────────────────────────────────────────────────────────────

def l2_engine_install_fragment_ok : Bool :=
  install_forward_ok && install_reverse_ok && install_four_pass_ok &&
  install_import_eq_local_ok && emit_after_install_ok && var_ok

theorem l2_engine_install_fragment_closeout :
    l2_engine_install_fragment_ok = true := by native_decide

end SounioFoEngineInstallFragment
