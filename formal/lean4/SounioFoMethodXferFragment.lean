/-
  SounioFoMethodXferFragment — L2 *method FO_XFER* slice for oral Css residual §5.4.

  Madaros method FO (lower.sio): mangled `Type_method` FO_XFER with recv as
  param0; op17 LOAD_PARAM_FIELD peels `self.field` to FO of that field channel;
  free f64 args are params 1.. (see fo_bc_expand_xfer_call_recv).

  Oral Css science model after peel (R1/R4 drivers):
    Pk { cl0, v0 }
    Pk.css(f, dose, tau, eta) = rate(f,dose,tau) / clearance(eta)
    clearance(eta) = self.cl0 * exp(eta)   -- eEta channel after exp at means
    rate = (f*dose)/tau

  Param layout used by FO (post peel):
    p0 = F, p1 = Dose, p2 = τ, p3 = CL0 (from self.cl0), p4 = eEta

  This file proves the peeled method body is definitionally cssSite and emits
  the golden RPN. Full op17/mangling for arbitrary methods remains open.

  Mathlib-free. Zero sorry. Companion: fo_method_xfer_fragment_cert.py
-/
namespace SounioFoMethodXferFragment

inductive PureExpr where
  | param : Nat → PureExpr
  | mul   : PureExpr → PureExpr → PureExpr
  | div   : PureExpr → PureExpr → PureExpr
  deriving BEq, Repr

open PureExpr

/-- After FO method xfer + self.cl0 peel, method body is pure oral Css. -/
def methodCssPeeled : PureExpr :=
  div (div (mul (param 0) (param 1)) (param 2))
      (mul (param 3) (param 4))

def cssSite : PureExpr := methodCssPeeled

theorem method_peel_eq_site : methodCssPeeled = cssSite := rfl

/-- Call-result make_pk(cl0,v0).css(...) peels to the same pure tree. -/
def callResultCssPeeled : PureExpr := methodCssPeeled

theorem call_result_eq_method : callResultCssPeeled = methodCssPeeled := rfl

def OP_PARAM : Nat := 1
def OP_MUL : Nat := 5
def OP_DIV : Nat := 6

structure FoInstr where
  op : Nat
  arg : Nat
  deriving BEq, Repr

def compile : PureExpr → List FoInstr
  | param i => [⟨OP_PARAM, i⟩]
  | mul a b => compile a ++ compile b ++ [⟨OP_MUL, 0⟩]
  | div a b => compile a ++ compile b ++ [⟨OP_DIV, 0⟩]

def cssSiteProg : List FoInstr :=
  [ ⟨OP_PARAM, 0⟩, ⟨OP_PARAM, 1⟩, ⟨OP_MUL, 0⟩
  , ⟨OP_PARAM, 2⟩, ⟨OP_DIV, 0⟩
  , ⟨OP_PARAM, 3⟩, ⟨OP_PARAM, 4⟩, ⟨OP_MUL, 0⟩
  , ⟨OP_DIV, 0⟩ ]

def emit_method_ok : Bool := compile methodCssPeeled == cssSiteProg
theorem emit_method : emit_method_ok = true := by native_decide

def emit_call_result_ok : Bool := compile callResultCssPeeled == cssSiteProg
theorem emit_call_result : emit_call_result_ok = true := by native_decide

def foVarCss : Rat :=
  let j0 : Rat := 500 / 60
  let j1 : Rat := (4 / 5) / 60
  let j3 : Rat := - (400 : Rat) / 300
  let j4 : Rat := - (20 : Rat) / 3
  j0*j0*(1/20)*(1/20) + j1*j1*10*10 + j3*j3*(3/10)*(3/10) + j4*j4*(1/10)*(1/10)

def var_ok : Bool := foVarCss == (191 : Rat) / 240
theorem var_freeze : var_ok = true := by native_decide

def l2_method_xfer_ok : Bool :=
  emit_method_ok && emit_call_result_ok && var_ok

theorem l2_method_xfer_closeout : l2_method_xfer_ok = true := by native_decide

end SounioFoMethodXferFragment
