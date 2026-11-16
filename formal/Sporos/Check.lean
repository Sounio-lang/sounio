/-
The stack-effect checker of Sporos and its soundness.

Each word declares its effect `(ins -- outs)` and its locals, and the heights
of its stacks at every jump target (`Word.ann`). `infer` fills in the height at
every other instruction by following fall-through; it is not trusted. `checkWord`
checks the whole table: every instruction has the operands it takes, every
fall-through and every jump lands on the height declared there, `ret` leaves
exactly `outs` values. `step_safe`: in a checked program, a step from a state
that agrees with the table never fails for a reason the checker rules out
(stack or auxiliary underflow, a bad local, a jump outside the word, a call to a
missing word, no frame), and the next state agrees with the table again. What
remains possible at run time: bounds, division by zero, a bad table, `demand`.
-/
import Sporos.Step

namespace Sporos

/-- Height after an instruction: `none` if it is ill-typed, `some none` if it
does not fall through. -/
def hstep (pr : Prog) (wd : Word) (h : Height) : Instr → Option (Option Height)
  | .prim p =>
    if p.eff.1 ≤ h.d then
      (if p = .exit then some none else some (some ⟨h.d - p.eff.1 + p.eff.2, h.a⟩))
    else none
  | .lit _ => some (some ⟨h.d + 1, h.a⟩)
  | .str _ => some (some ⟨h.d + 2, h.a⟩)
  | .pick n => if n < h.d then some (some ⟨h.d + 1, h.a⟩) else none
  | .tor => if 1 ≤ h.d then some (some ⟨h.d - 1, h.a + 1⟩) else none
  | .fromr => if 1 ≤ h.a then some (some ⟨h.d + 1, h.a - 1⟩) else none
  | .rfetch => if 1 ≤ h.a then some (some ⟨h.d + 1, h.a⟩) else none
  | .locget i => if i < wd.locals then some (some ⟨h.d + 1, h.a⟩) else none
  | .locset i => if 1 ≤ h.d ∧ i < wd.locals then some (some ⟨h.d - 1, h.a⟩) else none
  | .jmp t => if wd.ann.lookup t = some h then some none else none
  | .jz t | .jnz t =>
    if 1 ≤ h.d ∧ wd.ann.lookup t = some ⟨h.d - 1, h.a⟩ then some (some ⟨h.d - 1, h.a⟩) else none
  | .call w =>
    match pr.words[w]? with
    | some c => if c.ins ≤ h.d then some (some ⟨h.d - c.ins + c.outs, h.a⟩) else none
    | none => none
  | .ret => if h = ⟨wd.outs, 0⟩ then some none else none

/-- Heights by fall-through from the entry, switching to the declared height at
an annotated instruction. Untrusted: `checkWord` checks what it returns. -/
def infer (pr : Prog) (wd : Word) : List Height :=
  let rec go : List Instr → Nat → Height → List Height
    | [], _, _ => []
    | ins :: rest, i, cur =>
      let h := (wd.ann.lookup i).getD cur
      let nxt := match hstep pr wd h ins with
        | some (some h') => h'
        | _ => (wd.ann.lookup (i + 1)).getD ⟨0, 0⟩
      h :: go rest (i + 1) nxt
  go wd.code 0 ⟨wd.ins, 0⟩

def checkWord (pr : Prog) (wd : Word) (H : List Height) : Bool :=
  H.length == wd.code.length &&
  !wd.code.isEmpty && H[0]? == some ⟨wd.ins, 0⟩ &&
  (List.range wd.code.length).all (fun i =>
    match wd.code[i]?, H[i]? with
    | some ins, some h =>
      match hstep pr wd h ins with
      | none => false
      | some none => true
      | some (some h') => i + 1 < wd.code.length && H[i + 1]? == some h'
    | _, _ => false) &&
  wd.ann.all (fun (t, a) => t < wd.code.length && H[t]? == some a)

def checkProg (pr : Prog) : Bool :=
  (pr.words.all fun wd => checkWord pr wd (infer pr wd)) &&
  (match pr.words[pr.main]? with | some m => m.ins == 0 | none => false)

/-! ### The invariant -/

/-- The heights table of word `w`. -/
def tbl (pr : Prog) (w : Nat) : List Height :=
  match pr.words[w]? with
  | some wd => infer pr wd
  | none => []

/-- Frame `f` sits at a height the table declares. -/
def FrameOK (pr : Prog) (f : Frame) (dlen alen : Nat) : Prop :=
  ∃ wd h, pr.words[f.word]? = some wd ∧ (tbl pr f.word)[f.pc]? = some h ∧ f.pc < wd.code.length ∧
    dlen = f.base + h.d ∧ alen = f.abase + h.a ∧ f.locs.length = wd.locals

/-- The frames below the current one: each caller resumes, after its `call`, at
the height the callee's return produces. -/
def Chain (pr : Prog) : Frame → List Frame → Prop
  | _, [] => True
  | u, g :: rest =>
    (∃ wu, pr.words[u.word]? = some wu ∧ FrameOK pr g (u.base + wu.outs) u.abase) ∧ Chain pr g rest

def Inv (pr : Prog) (s : St) : Prop :=
  match s.rs with
  | [] => False
  | f :: rest => FrameOK pr f s.ds.length s.aux.length ∧ Chain pr f rest

/-- The failures the checker rules out. -/
def Err.static : Err → Bool
  | .underflow | .auxUnderflow | .badLocal | .badJump | .badWord | .noFrame => true
  | _ => false

end Sporos
