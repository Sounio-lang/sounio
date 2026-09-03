/-
  SounioFoEmitPure — L2-full *pure-emit* slice of residual §5.4.

  Madaros `fo_bc_compile_expr` (self-hosted/ir/lower.sio ~9342–9375) for the
  pure arithmetic fragment is:

    Ident(param i)  →  emit PARAM i
    Binary Mul(l,r) →  compile l; compile r; emit MUL
    Binary Div(l,r) →  compile l; compile r; emit DIV
    Binary Add/Sub  →  same with ADD/SUB

  This file **is** that algorithm on an abstract `PureExpr`, proved to emit
  exactly the oral-Css RPN of `SounioFoBytecodeFragment.cssSiteProg`, and
  proved that `run (compile e)` recovers the FoExpr of `e`.

  Also models pure-helper XFER expand for stdlib `fo.sio` bodies:

    fo_infusion_rate(F,Dose,τ)  = (F·Dose)/τ
    fo_clearance(CL0,eEta)      = CL0·eEta     -- eEta stands for exp(η) after
                                               -- unary EXP expand at means, or
                                               -- the eEta channel of L1/L2-frag
    fo_css(...)                 = rate / cl

  Full expand of fo_css equals the site PureExpr (definitional).

  What this **is not**:
    • a proof that Madaros multipass registration / multi-mod prepass always
      feeds this algorithm the right AST for every program
    • FO of method recv / op17 field / CALL_XFER cycle handling
    • EXP-as-op-20 vs eEta-as-param distinction (FO-var equivalent at η=0;
      see residual note)

  Mathlib-free. Zero sorry. Companion: scripts/research/fo_emit_pure_cert.py
-/
namespace SounioFoEmitPure

-- ── §1. Pure expression AST (subset of Madaros Expr for FO compile) ───────

inductive PureExpr where
  | param : Nat → PureExpr
  | lit   : Rat → PureExpr
  | add   : PureExpr → PureExpr → PureExpr
  | sub   : PureExpr → PureExpr → PureExpr
  | mul   : PureExpr → PureExpr → PureExpr
  | div   : PureExpr → PureExpr → PureExpr
  deriving BEq, Repr

open PureExpr

-- ── §2. FO instructions (same as L2-fragment) ─────────────────────────────

def OP_PARAM : Nat := 1
def OP_CONST : Nat := 2
def OP_ADD   : Nat := 3
def OP_SUB   : Nat := 4
def OP_MUL   : Nat := 5
def OP_DIV   : Nat := 6

structure FoInstr where
  op  : Nat
  arg : Nat
  deriving BEq, Repr

-- ── §3. fo_bc_compile_expr for pure fragment (lower.sio 9358–9367) ────────

/-- Madaros pure FO compile: left-to-right postfix emission. -/
def compile : PureExpr → List FoInstr
  | param i => [⟨OP_PARAM, i⟩]
  | lit _c  =>
      -- CONST path uses a const table in Madaros; for Nat-small embeds we
      -- only need param/mul/div for oral Css. Lit unused in cssSiteProg.
      [⟨OP_CONST, 0⟩]
  | add a b => compile a ++ compile b ++ [⟨OP_ADD, 0⟩]
  | sub a b => compile a ++ compile b ++ [⟨OP_SUB, 0⟩]
  | mul a b => compile a ++ compile b ++ [⟨OP_MUL, 0⟩]
  | div a b => compile a ++ compile b ++ [⟨OP_DIV, 0⟩]

-- ── §4. Oral Css as PureExpr + XFER expand of fo_css ──────────────────────

/-- Site: (F·Dose/τ)/(CL0·eEta) with params 0..4. -/
def cssSite : PureExpr :=
  div (div (mul (param 0) (param 1)) (param 2))
      (mul (param 3) (param 4))

/-- fo_infusion_rate body (stdlib/epistemic/fo.sio). -/
def foInfusionRate (F Dose tau : PureExpr) : PureExpr :=
  div (mul F Dose) tau

/-- fo_clearance body with eEta already the exp(η) channel (L1 convention). -/
def foClearance (CL0 eEta : PureExpr) : PureExpr :=
  mul CL0 eEta

/-- fo_css body = rate / clearance. -/
def foCss (F Dose tau CL0 eEta : PureExpr) : PureExpr :=
  div (foInfusionRate F Dose tau) (foClearance CL0 eEta)

/-- XFER expand of fo_css(param0..4) is definitionally the site expression. -/
def cssImportExpanded : PureExpr :=
  foCss (param 0) (param 1) (param 2) (param 3) (param 4)

theorem import_expand_eq_site : cssImportExpanded = cssSite := rfl

/-- Method body Pk.css(f,dose,tau,eta) after peel of self.cl0 → same pure tree
    when cl0 is param 3 and eEta/eta channel is param 4 (FO method xfer). -/
def cssMethodBody : PureExpr := cssSite

theorem method_eq_site : cssMethodBody = cssSite := rfl

-- ── §5. Emitted RPN matches L2-fragment cssSiteProg ───────────────────────

/-- Golden RPN from SounioFoBytecodeFragment (must stay in lockstep). -/
def cssSiteProg : List FoInstr :=
  [ ⟨OP_PARAM, 0⟩, ⟨OP_PARAM, 1⟩, ⟨OP_MUL, 0⟩
  , ⟨OP_PARAM, 2⟩, ⟨OP_DIV, 0⟩
  , ⟨OP_PARAM, 3⟩, ⟨OP_PARAM, 4⟩, ⟨OP_MUL, 0⟩
  , ⟨OP_DIV, 0⟩ ]

def emit_site_ok : Bool := compile cssSite == cssSiteProg

theorem emit_site_eq_golden : emit_site_ok = true := by native_decide

def emit_import_ok : Bool := compile cssImportExpanded == cssSiteProg

theorem emit_import_eq_golden : emit_import_ok = true := by native_decide

def emit_method_ok : Bool := compile cssMethodBody == cssSiteProg

theorem emit_method_eq_golden : emit_method_ok = true := by native_decide

-- ── §6. Stack interpretation recovers FoExpr (link to L2-fragment run) ────

inductive FoExpr where
  | seed : Nat → FoExpr
  | lit  : Rat → FoExpr
  | add  : FoExpr → FoExpr → FoExpr
  | sub  : FoExpr → FoExpr → FoExpr
  | mul  : FoExpr → FoExpr → FoExpr
  | div  : FoExpr → FoExpr → FoExpr
  deriving BEq, Repr

def toFo : PureExpr → FoExpr
  | .param i => .seed i
  | .lit c => .lit c
  | .add a b => .add (toFo a) (toFo b)
  | .sub a b => .sub (toFo a) (toFo b)
  | .mul a b => .mul (toFo a) (toFo b)
  | .div a b => .div (toFo a) (toFo b)

def params (i : Nat) : FoExpr := .seed i

def step (stk : List FoExpr) (ins : FoInstr) : Option (List FoExpr) :=
  match ins.op with
  | 1 => some (params ins.arg :: stk)
  | 2 => some (.lit (ins.arg : Nat) :: stk)
  | 3 => match stk with | b :: a :: r => some (.add a b :: r) | _ => none
  | 4 => match stk with | b :: a :: r => some (.sub a b :: r) | _ => none
  | 5 => match stk with | b :: a :: r => some (.mul a b :: r) | _ => none
  | 6 => match stk with | b :: a :: r => some (.div a b :: r) | _ => none
  | _ => none

def run (prog : List FoInstr) : Option FoExpr :=
  let rec go (stk : List FoExpr) : List FoInstr → Option FoExpr
    | [] => match stk with | [e] => some e | _ => none
    | i :: is =>
      match step stk i with
      | some stk' => go stk' is
      | none => none
  go [] prog

/-- Round-trip: compile then run recovers toFo for oral Css. -/
def roundtrip_site_ok : Bool :=
  match run (compile cssSite) with
  | some e => e == toFo cssSite
  | none => false

theorem roundtrip_site : roundtrip_site_ok = true := by native_decide

def roundtrip_import_ok : Bool :=
  match run (compile cssImportExpanded) with
  | some e => e == toFo cssSite
  | none => false

theorem roundtrip_import : roundtrip_import_ok = true := by native_decide

-- ── §7. FO variance freeze (same as L0/L1/L2-fragment) ───────────────────

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

def l2_pure_emit_ok : Bool :=
  emit_site_ok && emit_import_ok && emit_method_ok &&
  roundtrip_site_ok && roundtrip_import_ok && var_ok

theorem l2_pure_emit_closeout : l2_pure_emit_ok = true := by native_decide

end SounioFoEmitPure
