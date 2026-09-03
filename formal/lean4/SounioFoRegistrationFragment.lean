/-
  SounioFoRegistrationFragment — L2 *registration* slice of residual §5.4.

  Madaros multipass FO registration installs pure helpers as FO_XFER entries
  (kind-6 bytecode). Call-site expand (`fo_bc_expand_xfer_call` /
  `fo_bc_inline_xfer_bytecode`) substitutes PARAM i with the i-th call argument
  and inlines the body (lower.sio ~9623–9914).

  This file models that semantics for the **oral Css pure-helper fragment**:

    Registry (local ≡ multi-mod import — same name, same body):
      fo_infusion_rate(F,Dose,τ)  := (F·Dose)/τ
      fo_clearance(CL0,eEta)      := CL0·eEta
      fo_css(F,Dose,τ,CL0,eEta)   := fo_infusion_rate(...) / fo_clearance(...)

    expand(fo_css, [p0..p4])  =  cssSite   (after recursive helper expand)

    Method surface (Pk.css after FO method xfer + field peel):
      self.cl0 → param CL0; rate/clearance methods expand to the same pure tree.

  Proved:
    • recursive expand of fo_css args = cssSite (definitional / native_decide)
    • local registry = import registry (same bodies) ⇒ multi-mod name-identity
    • method peel model = cssSite
    • compile(expand(...)) = cssSiteProg (links L2 pure-emit golden RPN)

  Still open (true L2 registration residual):
    • Madaros multipass *always* installs these bodies before call sites
    • multi-mod frontend prepass correctness for all modules
    • op17 LOAD_PARAM_FIELD / method recv shift in full generality
    • EXP op-20 vs eEta channel (FO-var equivalent at η=0)

  Mathlib-free. Zero sorry. Companion:
  scripts/research/fo_registration_fragment_cert.py
-/
namespace SounioFoRegistrationFragment

-- ── §1. PureExpr + helper calls (pre-expand) ──────────────────────────────

inductive PureExpr where
  | param : Nat → PureExpr
  | lit   : Rat → PureExpr
  | add   : PureExpr → PureExpr → PureExpr
  | sub   : PureExpr → PureExpr → PureExpr
  | mul   : PureExpr → PureExpr → PureExpr
  | div   : PureExpr → PureExpr → PureExpr
  /-- Named pure-helper call (FO_XFER target) before expand. -/
  | call  : String → List PureExpr → PureExpr
  deriving BEq, Repr

open PureExpr

-- ── §2. Registry bodies (stdlib/epistemic/fo.sio + multipass install) ─────

/-- Body of fo_infusion_rate: params 0,1,2 = F, Dose, τ. -/
def bodyInfusionRate : PureExpr :=
  div (mul (param 0) (param 1)) (param 2)

/-- Body of fo_clearance: params 0,1 = CL0, eEta. -/
def bodyClearance : PureExpr :=
  mul (param 0) (param 1)

/-- Body of fo_css: nested *calls* (as registered before expand). -/
def bodyCss : PureExpr :=
  div
    (call "fo_infusion_rate" [param 0, param 1, param 2])
    (call "fo_clearance" [param 3, param 4])

/-- Local-module registry lookup. -/
def lookupLocal (name : String) : Option PureExpr :=
  if name == "fo_infusion_rate" then some bodyInfusionRate
  else if name == "fo_clearance" then some bodyClearance
  else if name == "fo_css" then some bodyCss
  else none

/-- Multi-mod import registry — same pure bodies (name-identity). -/
def lookupImport (name : String) : Option PureExpr :=
  lookupLocal name

theorem registries_agree (name : String) :
    lookupLocal name = lookupImport name := rfl

-- ── §3. PARAM substitution (inline XFER) ──────────────────────────────────

/-- Substitute call args for param indices (Madaros LOAD_PARAM → arg subtree). -/
def subst (args : List PureExpr) : PureExpr → PureExpr
  | param i =>
      match args[i]? with
      | some a => a
      | none => param i
  | lit c => lit c
  | add a b => add (subst args a) (subst args b)
  | sub a b => sub (subst args a) (subst args b)
  | mul a b => mul (subst args a) (subst args b)
  | div a b => div (subst args a) (subst args b)
  | call n as => call n (as.map (subst args))

/-- One expand step: replace outermost calls using the registry. -/
def expandOnce (lookup : String → Option PureExpr) : PureExpr → PureExpr
  | call n as =>
      match lookup n with
      | some body => subst as body
      | none => call n as
  | add a b => add (expandOnce lookup a) (expandOnce lookup b)
  | sub a b => sub (expandOnce lookup a) (expandOnce lookup b)
  | mul a b => mul (expandOnce lookup a) (expandOnce lookup b)
  | div a b => div (expandOnce lookup a) (expandOnce lookup b)
  | e => e

/-- Multipass expand (bounded — oral Css needs ≤ 2 passes: fo_css then leaves). -/
def expandN (lookup : String → Option PureExpr) : Nat → PureExpr → PureExpr
  | 0, e => e
  | n + 1, e => expandN lookup n (expandOnce lookup e)

def expandFull (lookup : String → Option PureExpr) (e : PureExpr) : PureExpr :=
  expandN lookup 3 e

-- ── §4. Oral Css site golden ──────────────────────────────────────────────

def cssSite : PureExpr :=
  div (div (mul (param 0) (param 1)) (param 2))
      (mul (param 3) (param 4))

/-- Call fo_css with params 0..4 (import or local call site). -/
def cssCallSite : PureExpr :=
  call "fo_css" [param 0, param 1, param 2, param 3, param 4]

/-- After multipass expand under local registry. -/
def cssExpandedLocal : PureExpr :=
  expandFull lookupLocal cssCallSite

/-- After multipass expand under import registry. -/
def cssExpandedImport : PureExpr :=
  expandFull lookupImport cssCallSite

def expand_local_ok : Bool := cssExpandedLocal == cssSite
def expand_import_ok : Bool := cssExpandedImport == cssSite

theorem expand_local_eq_site : expand_local_ok = true := by native_decide
theorem expand_import_eq_site : expand_import_ok = true := by native_decide

def expand_local_eq_import_ok : Bool := cssExpandedLocal == cssExpandedImport
theorem expand_local_eq_import : expand_local_eq_import_ok = true := by native_decide

-- ── §5. Method surface (Pk.css after peel) ─────────────────────────────────
--
-- Method FO_XFER mangles Pk_css; self is param0; self.cl0 is op17 peel to FO
-- of cl0 seed. Model: after peel, body is pure tree with CL0 = param 3,
-- eEta = param 4, F/Dose/τ = free args 0..2 — same as cssSite param layout
-- used by science drivers (peels then call).

def cssMethodPeeled : PureExpr := cssSite

theorem method_peel_eq_site : cssMethodPeeled = cssSite := rfl

-- ── §6. Link to pure-emit golden RPN ──────────────────────────────────────

def OP_PARAM : Nat := 1
def OP_MUL   : Nat := 5
def OP_DIV   : Nat := 6

structure FoInstr where
  op  : Nat
  arg : Nat
  deriving BEq, Repr

/-- compile pure (no Call) — same as SounioFoEmitPure.compile. -/
def compile : PureExpr → List FoInstr
  | param i => [⟨OP_PARAM, i⟩]
  | lit _ => [⟨2, 0⟩]
  | add a b => compile a ++ compile b ++ [⟨3, 0⟩]
  | sub a b => compile a ++ compile b ++ [⟨4, 0⟩]
  | mul a b => compile a ++ compile b ++ [⟨OP_MUL, 0⟩]
  | div a b => compile a ++ compile b ++ [⟨OP_DIV, 0⟩]
  | call _ _ => []  -- must expand first

def cssSiteProg : List FoInstr :=
  [ ⟨OP_PARAM, 0⟩, ⟨OP_PARAM, 1⟩, ⟨OP_MUL, 0⟩
  , ⟨OP_PARAM, 2⟩, ⟨OP_DIV, 0⟩
  , ⟨OP_PARAM, 3⟩, ⟨OP_PARAM, 4⟩, ⟨OP_MUL, 0⟩
  , ⟨OP_DIV, 0⟩ ]

def emit_after_expand_ok : Bool :=
  compile cssExpandedLocal == cssSiteProg

theorem emit_after_expand : emit_after_expand_ok = true := by native_decide

def emit_after_import_expand_ok : Bool :=
  compile cssExpandedImport == cssSiteProg

theorem emit_after_import_expand : emit_after_import_expand_ok = true := by native_decide

-- ── §7. FO variance freeze ────────────────────────────────────────────────

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

-- ── §8. Bundle ────────────────────────────────────────────────────────────

def l2_registration_fragment_ok : Bool :=
  expand_local_ok && expand_import_ok && expand_local_eq_import_ok &&
  emit_after_expand_ok && emit_after_import_expand_ok && var_ok

theorem l2_registration_fragment_closeout :
    l2_registration_fragment_ok = true := by native_decide

end SounioFoRegistrationFragment
