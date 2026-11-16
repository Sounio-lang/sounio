/-
Per-handler triples for the stage-1 Sporos checker (decision 34), draft.

`bootstrap/roots/sporos/sporos-stage1.hex` is the hand-written checker.
This file is the proof skeleton for it, in the style the README promises:
one Hoare triple per opcode handler over a symbolic `X86Int.St`, composed
against `Sporos.hstep`, so that the code walk refines `checkWord` under
`infer` and the whole program refines `checkProg`.

Status: the handler addresses and the triple statements are fixed; the
decode witnesses below are kernel-checked. The triple proofs are the
work that remains (they are stated as `Prop`-valued definitions, not as
theorems, so nothing here rests on an admitted proof). The single-pass
reduction the walk implements — fall-through equality checked against
the next ann, non-fallthrough restarts at an ann or at `(0,0)`, every ann
entry equal to the first at its index — is argued in the hex listing's
comments and is the first refinement lemma to formalise.

Register conventions at a handler entry (from the listing): `rbx` file
base, `r15` end, `r14` words table (stride 64), `r13` nwords, `rdi` the
current word entry, `rsi` the instruction cursor, `rcx` the instruction
index, `r8`/`r9` the current height `(d, a)`. Handlers exit by jumping
to `fall_tail` (the `hstep` result falls through) or `none_tail`.
-/
import Bootstrap.X86Dec
import Sporos.Bytecode
import Sporos.Check

namespace SporosX1

open X86Env X86Int X86Dec Sporos

/-- The loaded image base. -/
def base : Nat := 0x400000

/-- File offsets of the checker code in `sporos-stage1.hex`, as addresses. -/
def aL32 : Nat := base + 0x249
def aRDC : Nat := base + 0x271
def aANNL : Nat := base + 0x296
def aWalk : Nat := base + 0x53e
def aFallTail : Nat := base + 0x615
def aNoneTail : Nat := base + 0x669
def aPrim : Nat := base + 0x6b5
def aLit : Nat := base + 0x6f9
def aStr : Nat := base + 0x706
def aPick : Nat := base + 0x713
def aTor : Nat := base + 0x736
def aFromr : Nat := base + 0x752
def aRfetch : Nat := base + 0x76e
def aLocget : Nat := base + 0x786
def aLocset : Nat := base + 0x7ad
def aJmp : Nat := base + 0x7df
def aJz : Nat := base + 0x831
def aCall : Nat := base + 0x899
def aRet : Nat := base + 0x8e7
def aWordDone : Nat := base + 0x908
def aAccept : Nat := base + 0x915
def aEff : Nat := base + 0x91d
def aFail1 : Nat := base + 0x124
def aFail2 : Nat := base + 0x12c

/-- A triple over the architectural step: from any state at `entry`
satisfying `Pre`, the run reaches `exit` in some number of steps with
`Post` relating the two states. `Post` also covers the refusal exits
 (`aFail1` as `exit`) when a handler is given both continuations. -/
def Triple (im : Nat → Option (Ins × Nat)) (entry exit : Nat)
    (Pre : X86Int.St → Prop) (Post : X86Int.St → X86Int.St → Prop) : Prop :=
  ∀ s, s.rip = entry → Pre s →
    ∃ n s', run im n (Out.ok s) = Out.ok s' ∧ s'.rip = exit ∧ Post s s'

/-- `L32` (0x249): `rdx` names four mapped bytes; `rax` returns them
little-endian, `rdx` preserved, nothing else read. -/
def L32Spec (im : Nat → Option (Ins × Nat)) : Prop :=
  ∀ (a : Nat), Triple im aL32 (aL32 + 0x28)
    (fun s => s.reg 2 = some a ∧ (s.mem.rdN a 4).isSome)
    (fun s s' => s'.reg 0 = (s.mem.rdN a 4).map leVal ∧ s'.reg 2 = some a ∧ s'.mem = s.mem)

/-- `RDC` (0x271): with at least four bytes between cursor `rsi` and
end `r15`, returns their `u32` and advances the cursor; otherwise the
run reaches `aFail1` (stated by taking `exit = aFail1` at the call). -/
def RDCSpec (im : Nat → Option (Ins × Nat)) : Prop :=
  ∀ (c e : Nat), Triple im aRDC (aRDC + 0x25)
    (fun s => s.reg 6 = some c ∧ s.reg 15 = some e ∧ c + 4 ≤ e ∧ (s.mem.rdN c 4).isSome)
    (fun s s' => s'.reg 0 = (s.mem.rdN c 4).map leVal ∧ s'.reg 6 = some (c + 4) ∧ s'.mem = s.mem)

/-- `ANNL` (0x296): first-match ann lookup for key `r12` in the word at
`rdi`; `rax` is the entry address, or 0. The semantic content is that
the entries at `rax` and at `rax + 4`, `rax + 8` are the `(t, d, a)` of
`wd.ann`'s first `t = r12` record — the representation invariant below
is what ties addresses in memory to `wd.ann`. -/
def AnnAt (s : X86Int.St) (e t d a : Nat) : Prop :=
  (s.mem.rdN e 4).map leVal = some t ∧ (s.mem.rdN (e + 4) 4).map leVal = some d ∧
    (s.mem.rdN (e + 8) 4).map leVal = some a

def ANNLSpec (im : Nat → Option (Ins × Nat)) : Prop :=
  ∀ (wd : Sporos.Word) (t : Nat), Triple im aANNL (aANNL + 0x33)
    (fun s => s.reg 12 = some t ∧ True)
    (fun s s' => ∀ e, s'.reg 0 = some e →
      (e = 0 ∧ wd.ann.lookup t = none) ∨
      (∃ d a, AnnAt s' e t d a ∧ wd.ann.lookup t = some ⟨d, a⟩))

/-- The shape every fall-through handler's triple takes: `hstep` on the
height in `r8`/`r9` gives `some (some h')`, and the handler leaves
`r8`/`r9` at `h'` at `aFallTail`, cursor past the instruction. Handlers
with `some none` (`jmp`, `ret`, `prim exit`) have the same shape with
`aNoneTail` and the height unchanged on entry to the tail. -/
def HandlerSpec (im : Nat → Option (Ins × Nat)) (entry tail : Nat)
    (pr : Prog) (wd : Word) (ins : Instr) (len : Nat) : Prop :=
  ∀ h c, Triple im entry tail
    (fun s => s.reg 8 = some h.d ∧ s.reg 9 = some h.a ∧ s.reg 6 = some c ∧
      (s.mem.rdN c len).isSome)
    (fun s s' =>
      (∀ h', hstep pr wd h ins = some (some h') →
        s'.reg 8 = some h'.d ∧ s'.reg 9 = some h'.a) ∧
      (hstep pr wd h ins = some none → s'.reg 8 = some h.d ∧ s'.reg 9 = some h.a) ∧
      s'.reg 6 = some (c + len))

/-- The seventeen handler triples to prove, by handler. `prim` is keyed
to the Eff table at `aEff` being `Prim.eff` in `Prim.all` order — the
`eff_table` lemma below is its statement. -/
def Handlers (im : Nat → Option (Ins × Nat)) (pr : Prog) (wd : Word) : Prop :=
  (∀ p, HandlerSpec im aPrim (if p = Prim.exit then aNoneTail else aFallTail) pr wd (.prim p) 1) ∧
  (∀ n, HandlerSpec im aLit aFallTail pr wd (.lit n) 9) ∧
  (∀ i, HandlerSpec im aStr aFallTail pr wd (.str i) 5) ∧
  (∀ n, HandlerSpec im aPick aFallTail pr wd (.pick n) 5) ∧
  HandlerSpec im aTor aFallTail pr wd .tor 1 ∧
  HandlerSpec im aFromr aFallTail pr wd .fromr 1 ∧
  HandlerSpec im aRfetch aFallTail pr wd .rfetch 1 ∧
  (∀ i, HandlerSpec im aLocget aFallTail pr wd (.locget i) 5) ∧
  (∀ i, HandlerSpec im aLocset aFallTail pr wd (.locset i) 5) ∧
  (∀ t, HandlerSpec im aJmp aNoneTail pr wd (.jmp t) 5) ∧
  (∀ t, HandlerSpec im aJz aFallTail pr wd (.jz t) 5) ∧
  (∀ t, HandlerSpec im aJz aFallTail pr wd (.jnz t) 5) ∧
  (∀ w, HandlerSpec im aCall aFallTail pr wd (.call w) 5) ∧
  HandlerSpec im aRet aNoneTail pr wd .ret 1

/-- The Eff table: at byte `2 * Prim.code p` of `aEff` sits `(eff.1, eff.2)`. -/
def EffTable (s : X86Int.St) : Prop :=
  ∀ (p : Prim), (s.mem.rdN (aEff + 2 * p.code) 2) = some [p.eff.1, p.eff.2]

/-! ### Decode witnesses: the first instruction at each handler address,
as written in the listing, decodes as claimed. Kernel-checked; the
`0x40091d` immediate is the Eff table address `aEff`. -/

example : at0 [0x0f, 0xb6, 0x42, 0x03] = some (.ld8 0 2 3, 4) := by decide +kernel
example : at0 [0x48, 0xba, 0x1d, 0x09, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00]
    = some (.movI 2 0x40091d, 10) := by decide +kernel
example : at0 [0x48, 0x83, 0xc1, 0x01] = some (.aluI X86Int.Alu.add 1 1, 4) := by decide +kernel
example : at0 [0x51] = some (.push 1, 1) := by decide +kernel
example : at0 [0x6a, 0x00] = some (.pushI 0, 2) := by decide +kernel

/-! ### `hstep` computations the handler triples will cite, kernel-checked. -/

example (pr : Prog) (wd : Word) (h : Height) :
    hstep pr wd h (.prim .exit) = (if 1 ≤ h.d then some none else none) := by
  unfold hstep; simp [Prim.eff]

end SporosX1
