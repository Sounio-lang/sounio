/-
  SounioFoMultimodFragment — L2 *multi-mod FO prepass* slice for oral Css.

  Madaros multi-mod FO prepass walks all loaded Programs and preregisters pure
  helpers before the seed body's lower (module-frontend FO prepass). For oral
  Css science, stdlib `epistemic::fo` items install with the same pure bodies
  as a local module would.

  Model:
    modLocal  = [fo_infusion_rate, fo_clearance, fo_css]
    modImport = same three items (multi-mod source)
    install(modLocal ∪ modImport)  -- prepass over both programs
    expand(fo_css(...)) = cssSite
    local-only install ≡ import-only install ≡ union install

  Full multi-mod loader / name resolution for arbitrary modules remains open.

  Mathlib-free. Zero sorry. Companion: fo_multimod_fragment_cert.py
-/
namespace SounioFoMultimodFragment

inductive PureExpr where
  | param : Nat → PureExpr
  | mul   : PureExpr → PureExpr → PureExpr
  | div   : PureExpr → PureExpr → PureExpr
  | call  : String → List PureExpr → PureExpr
  deriving BEq, Repr

open PureExpr

structure PureItem where
  name : String
  body : PureExpr
  deriving BEq, Repr

def bodyInfusion : PureExpr := div (mul (param 0) (param 1)) (param 2)
def bodyClearance : PureExpr := mul (param 0) (param 1)
def bodyCss : PureExpr :=
  div (call "fo_infusion_rate" [param 0, param 1, param 2])
      (call "fo_clearance" [param 3, param 4])

def modLocal : List PureItem :=
  [ ⟨"fo_infusion_rate", bodyInfusion⟩
  , ⟨"fo_clearance", bodyClearance⟩
  , ⟨"fo_css", bodyCss⟩ ]

/-- Multi-mod import of epistemic::fo — same pure bodies. -/
def modImport : List PureItem := modLocal

abbrev Registry := List (String × PureExpr)

def regInstall (r : Registry) (it : PureItem) : Registry :=
  List.append r [(it.name, it.body)]

def installAll (items : List PureItem) : Registry :=
  items.foldl regInstall []

def regLookup (r : Registry) (name : String) : Option PureExpr :=
  let rec go : Registry → Option PureExpr
    | [] => none
    | (n, b) :: rest =>
        match go rest with
        | some b' => some b'
        | none => if n == name then some b else none
  go r

def subst (args : List PureExpr) : PureExpr → PureExpr
  | param i => match args[i]? with | some a => a | none => param i
  | mul a b => mul (subst args a) (subst args b)
  | div a b => div (subst args a) (subst args b)
  | call n as => call n (as.map (subst args))

def expandOnce (r : Registry) : PureExpr → PureExpr
  | call n as => match regLookup r n with | some b => subst as b | none => call n as
  | mul a b => mul (expandOnce r a) (expandOnce r b)
  | div a b => div (expandOnce r a) (expandOnce r b)
  | e => e

def expandFull (r : Registry) (e : PureExpr) : PureExpr :=
  expandOnce r (expandOnce r (expandOnce r e))

def cssSite : PureExpr :=
  div (div (mul (param 0) (param 1)) (param 2))
      (mul (param 3) (param 4))

def cssCall : PureExpr :=
  call "fo_css" [param 0, param 1, param 2, param 3, param 4]

def regLocal : Registry := installAll modLocal
def regImport : Registry := installAll modImport
def regUnion : Registry := installAll (modLocal ++ modImport)

def local_ok : Bool := expandFull regLocal cssCall == cssSite
def import_ok : Bool := expandFull regImport cssCall == cssSite
def union_ok : Bool := expandFull regUnion cssCall == cssSite
def local_eq_import_ok : Bool :=
  expandFull regLocal cssCall == expandFull regImport cssCall
def union_eq_local_ok : Bool :=
  expandFull regUnion cssCall == expandFull regLocal cssCall

theorem multimod_local : local_ok = true := by native_decide
theorem multimod_import : import_ok = true := by native_decide
theorem multimod_union : union_ok = true := by native_decide
theorem multimod_local_eq_import : local_eq_import_ok = true := by native_decide
theorem multimod_union_eq_local : union_eq_local_ok = true := by native_decide

def foVarCss : Rat :=
  let j0 : Rat := 500 / 60
  let j1 : Rat := (4 / 5) / 60
  let j3 : Rat := - (400 : Rat) / 300
  let j4 : Rat := - (20 : Rat) / 3
  j0*j0*(1/20)*(1/20) + j1*j1*10*10 + j3*j3*(3/10)*(3/10) + j4*j4*(1/10)*(1/10)

def var_ok : Bool := foVarCss == (191 : Rat) / 240
theorem var_freeze : var_ok = true := by native_decide

def l2_multimod_ok : Bool :=
  local_ok && import_ok && union_ok && local_eq_import_ok && union_eq_local_ok && var_ok

theorem l2_multimod_closeout : l2_multimod_ok = true := by native_decide

end SounioFoMultimodFragment
