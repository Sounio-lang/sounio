/-
Per-handler triples for the stage-1 Sporos checker (decision 34), draft.

`bootstrap/roots/sporos/sporos-stage1.hex` is the hand-written checker.
This file is the proof skeleton for it, in the style the README promises:
one Hoare triple per opcode handler over a symbolic `X86Int.St`, composed
against `Sporos.hstep`, so that the code walk refines `checkWord` under
`infer` and the whole program refines `checkProg`. All specs are
parametric in entry address, memory region and stack room; no spec
names a test-oracle mapping or stack address.

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
def aRDC : Nat := base + 0x24c
def aANNL : Nat := base + 0x271
def aWalk : Nat := base + 0x519
def aFallTail : Nat := base + 0x5f0
def aNoneTail : Nat := base + 0x644
def aPrim : Nat := base + 0x690
def aLit : Nat := base + 0x6d4
def aStr : Nat := base + 0x6e1
def aPick : Nat := base + 0x6ee
def aTor : Nat := base + 0x711
def aFromr : Nat := base + 0x72d
def aRfetch : Nat := base + 0x749
def aLocget : Nat := base + 0x761
def aLocset : Nat := base + 0x788
def aJmp : Nat := base + 0x7ba
def aJz : Nat := base + 0x80c
def aCall : Nat := base + 0x874
def aRet : Nat := base + 0x8c2
def aWordDone : Nat := base + 0x8e3
def aAccept : Nat := base + 0x8f0
def aEff : Nat := base + 0x8f8
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

/-- `k` writable words below `rsp`: each slot at `sp - 8*(i+1)` is
writable. Stated per slot, parametric in `sp`; the proof layer that
composes nested calls supplies it for its depth (six words suffice
for this checker's deepest chain, ANNL → L32 under a pushed `rcx`). -/
def StackRoom (s : X86Int.St) (k : Nat) : Prop :=
  ∀ sp, s.reg 4 = some sp → ∀ i, i < k → ∀ v,
    (wr64 s.mem ((sp + W - 8 * (i + 1)) % W) v).isSome

/-- `L32`: `rdx` names four mapped bytes; `rax` returns them
little-endian, `rdx` preserved, nothing else read. Parametric in the
entry (the listing puts it at `aL32`); the exit is the `ret`
instruction itself, `entry + 2`. -/
def L32Spec (im : Nat → Option (Ins × Nat)) (entry : Nat) : Prop :=
  ∀ (a : Nat), a < W → Triple im entry (entry + 2)
    (fun s => s.reg 2 = some a ∧ (s.mem.rdN a 4).isSome)
    (fun s s' => s'.reg 0 = (s.mem.rdN a 4).map leVal ∧ s'.reg 2 = some a ∧ s'.mem = s.mem ∧
      (∀ i, i ≠ 0 → s'.reg i = s.reg i) ∧ s'.regs.length = s.regs.length)

/-- `RDC`: with at least four bytes between cursor `rsi` and end
`r15`, returns their `u32` and advances the cursor; otherwise the
run goes to the refusal exit (a separate triple). The last
precondition is the frame fact that the return-address slot does
not overlap the four bytes read. Parametric in the entry; the
exit is its `ret` at `entry + 0x24`, and its `call` needs one word
of stack room. -/
def RDCSpec (im : Nat → Option (Ins × Nat)) (entry : Nat) : Prop :=
  ∀ (c e : Nat), c < W → e < W → Triple im entry (entry + 0x24)
    (fun s => s.reg 6 = some c ∧ s.reg 15 = some e ∧ c + 4 ≤ e ∧
      (s.mem.rdN c 4).isSome ∧ StackRoom s 1 ∧ (∃ sp, s.reg 4 = some sp) ∧
      (∀ sp m, s.reg 4 = some sp → wr64 s.mem ((sp + W - 8) % W) (entry + 32) = some m →
        m.rdN c 4 = s.mem.rdN c 4))
    (fun s s' => s'.reg 0 = (s.mem.rdN c 4).map leVal ∧ s'.reg 6 = some (c + 4) ∧
      (s'.mem.rdN c 4 = s.mem.rdN c 4))

/-- `ANNL`: first-match ann lookup for key `r12` in the word at
`rdi`; `rax` is the entry address, or 0. The semantic content is that
the entries at `rax` and at `rax + 4`, `rax + 8` are the `(t, d, a)` of
`wd.ann`'s first `t = r12` record — the representation invariant below
is what ties addresses in memory to `wd.ann`. -/
def AnnAt (s : X86Int.St) (e t d a : Nat) : Prop :=
  (s.mem.rdN e 4).map leVal = some t ∧ (s.mem.rdN (e + 4) 4).map leVal = some d ∧
    (s.mem.rdN (e + 8) 4).map leVal = some a

/-- The ann array of a word laid out at `p`: entry `i` at `p + 12*i`
holds `(index, d, a)` as three `u32`s, indices below `W`. -/
def AnnLayout (s : X86Int.St) (ann : List (Nat × Height)) (p : Nat) : Prop :=
  ∀ i (hi : i < ann.length),
    AnnAt s (p + 12 * i) ann[i].1 ann[i].2.d ann[i].2.a ∧ ann[i].1 < W

/-- What `ANNL` leaves in `rax`, as a statement about `List.lookup`. -/
def AnnPost (ann : List (Nat × Height)) (t : Nat) (s' : X86Int.St) : Prop :=
  ∀ e, s'.reg 0 = some e →
    (e = 0 ∧ ann.lookup t = none) ∨
    (∃ d a, e ≠ 0 ∧ AnnAt s' e t d a ∧ ann.lookup t = some ⟨d, a⟩)

/-- `ANNL`: first-match ann lookup for key `r12` in the word at `rdi`.
The word entry at `w` holds the ann count at `w + 32` and the ann
pointer at `w + 48` (the words-table layout); the stack's three
words below `rsp` are disjoint from the word entry and the ann
array. `ANNL` has two `ret` instructions — refusal at `entry + 45`,
found at `entry + 50` — so the exit is a disjunction. `rcx` is
restored modulo `W` (what `push`/`pop` round-trips), `rsp` and
`r12` are restored exactly. Parametric in both entries, the word
address, the ann pointer and the stack address throughout. -/
def ANNLSpec (im : Nat → Option (Ins × Nat)) (entry l32 : Nat) : Prop :=
  ∀ (wd : Sporos.Word) (t w p sp cv : Nat), t < W → 0 < p → w + 56 ≤ W →
    p + 12 * wd.ann.length < W → 24 ≤ sp → sp < W →
    ∀ s, s.rip = entry → s.reg 12 = some t → s.reg 7 = some w →
      s.reg 4 = some sp → s.reg 1 = some cv →
      rd64 s.mem (w + 48) = some p → rd64 s.mem (w + 32) = some wd.ann.length →
      AnnLayout s wd.ann p → StackRoom s 3 →
      (sp ≤ w ∨ w + 56 ≤ sp - 24) → (sp ≤ p ∨ p + 12 * wd.ann.length ≤ sp - 24) →
      ∃ n s', run im n (Out.ok s) = Out.ok s' ∧
        (s'.rip = entry + 45 ∨ s'.rip = entry + 50) ∧ AnnPost wd.ann t s' ∧
        s'.reg 1 = some (cv % W) ∧ s'.reg 4 = some sp ∧ s'.reg 12 = some t ∧
        s'.reg 7 = some w ∧ AnnLayout s' wd.ann p ∧
        (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → s'.reg i = s.reg i) ∧
        (∀ x n, x + n ≤ sp - 24 ∨ sp ≤ x → s'.mem.rdN x n = s.mem.rdN x n) ∧
        (∃ e, s'.reg 0 = some e)

/-- The shape every fall-through handler's triple takes: `hstep` on the
height in `r8`/`r9` gives `some (some h')`, and the handler leaves
`r8`/`r9` at `h'` at `aFallTail`, cursor past the instruction. Handlers
with `some none` (`jmp`, `ret`, `prim exit`) have the same shape with
`aNoneTail` and the height unchanged on entry to the tail. The spec
is the success half: it assumes `hstep` succeeds (the handler's own
checks are exactly `hstep`'s guards, so on heights where `hstep`
refuses the run goes to the refusal exit instead — composed by the
walk), and heights and cursor stay below `W` so the machine's
modular arithmetic is the semantics' arithmetic. -/
def HandlerSpec (im : Nat → Option (Ins × Nat)) (entry tail : Nat)
    (pr : Prog) (wd : Word) (ins : Instr) (len : Nat) : Prop :=
  ∀ h c, (hstep pr wd h ins).isSome → h.d + 2 < W → h.a + 1 < W → c + len < W →
    Triple im entry tail
    (fun s => s.reg 8 = some h.d ∧ s.reg 9 = some h.a ∧ s.reg 6 = some c ∧
      (s.mem.rdN c len).isSome)
    (fun _s s' =>
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
`0x4008f8` immediate is the Eff table address `aEff`. -/

example : at0 [0x0f, 0xb6, 0x42, 0x03] = some (.ld8 0 2 3, 4) := by decide +kernel
example : at0 [0x48, 0xba, 0xf8, 0x08, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00]
    = some (.movI 2 0x4008f8, 10) := by decide +kernel
example : at0 [0x48, 0x83, 0xc1, 0x01] = some (.aluI X86Int.Alu.add 1 1, 4) := by decide +kernel
example : at0 [0x51] = some (.push 1, 1) := by decide +kernel
example : at0 [0x6a, 0x00] = some (.pushI 0, 2) := by decide +kernel

/-! ### `hstep` computations the handler triples will cite, kernel-checked. -/

example (pr : Prog) (wd : Word) (h : Height) :
    hstep pr wd h (.prim .exit) = (if 1 ≤ h.d then some none else none) := by
  unfold hstep; simp [Prim.eff]

/-! ### Proof infrastructure: run composition, register lemmas, step lemmas -/

theorem run_add (im : Nat → Option (Ins × Nat)) (n m : Nat) (o : Out) :
    run im (n + m) o = run im m (run im n o) := by
  induction n generalizing o with
  | zero => simp [run]
  | succ n ih =>
    cases o with
    | ok s =>
      have h : n + 1 + m = (n + m) + 1 := by omega
      rw [h]
      show run im (n + m) (X86Int.step im s) = run im m (run im (n + 1) (Out.ok s))
      rw [ih (o := X86Int.step im s)]
      congr 1
    | exit c => cases m <;> rfl
    | stuck => cases m <;> rfl

/-- One more step at the front of a run. -/
theorem run_succ_ok (im : Nat → Option (Ins × Nat)) (n : Nat) (s : X86Int.St) :
    run im (n + 1) (Out.ok s) = run im n (X86Int.step im s) := rfl

theorem run_zero (im : Nat → Option (Ins × Nat)) (o : Out) : run im 0 o = o := rfl

theorem reg_set_self {s : X86Int.St} {i : Nat} {v : Option Nat} (h : i < s.regs.length) :
    (s.set i v).reg i = v := by
  simp only [St.reg, St.set, List.getElem?_set, h, ↓reduceIte, Option.getD]

theorem reg_set_other {s : X86Int.St} {i j : Nat} {v : Option Nat} (h : j ≠ i) :
    (s.set i v).reg j = s.reg j := by
  simp only [St.reg, St.set, List.getElem?_set]
  rw [if_neg (Ne.symm h)]

theorem reg_some_len {s : X86Int.St} {i v : Nat} (h : s.reg i = some v) :
    i < s.regs.length := by
  unfold St.reg at h
  cases hg : s.regs[i]? with
  | none => simp [hg] at h
  | some x =>
    have h2 := List.getElem?_eq_some_iff.mp hg
    exact h2.1

theorem reg_with_rip (t : X86Int.St) (x i : Nat) : ({ t with rip := x }).reg i = t.reg i := rfl
theorem len_set (t : X86Int.St) (i : Nat) (v : Option Nat) :
    (t.set i v).regs.length = t.regs.length := by simp [St.set]
theorem mem_set (t : X86Int.St) (i : Nat) (v : Option Nat) : (t.set i v).mem = t.mem := rfl
theorem mem_with_rip (t : X86Int.St) (x : Nat) : ({ t with rip := x }).mem = t.mem := rfl

theorem step_ld32 {im : Nat → Option (Ins × Nat)} {s : X86Int.St} {d b disp len x v : Nat}
    (hi : im s.rip = some (.ld32 d b disp, len)) (hx : s.reg b = some x)
    (hv : rd32 s.mem (ea x disp) = some v) :
    X86Int.step im s = .ok { s.set d (some v) with rip := s.rip + len } := by
  simp only [X86Int.step, exec, hi, hx, hv, lift, bind, Option.bind, pure]

theorem step_ret {im : Nat → Option (Ins × Nat)} {s : X86Int.St} {len sp t : Nat}
    (hi : im s.rip = some (.ret, len)) (hsp : s.reg 4 = some sp)
    (ht : rd64 s.mem sp = some t) :
    X86Int.step im s = .ok { s.set 4 (some ((sp + 8) % W)) with rip := t } := by
  simp only [X86Int.step, exec, hi, hsp, ht, lift, bind, Option.bind, pure]

theorem step_call {im : Nat → Option (Ins × Nat)} {s : X86Int.St} {len t sp : Nat} {m : Mem}
    (hi : im s.rip = some (.call t, len)) (hsp : s.reg 4 = some sp)
    (hm : wr64 s.mem ((sp + W - 8) % W) (s.rip + len) = some m) :
    X86Int.step im s = .ok { s.set 4 (some ((sp + W - 8) % W)) with mem := m, rip := t } := by
  simp only [X86Int.step, exec, hi, hsp, hm, lift, bind, Option.bind, pure]

/-! ### Memory frame lemmas: what one write does to `Mem.find` and reads -/

/-- Replacing region `i` by one with the same base, extent and `rw`
changes `Mem.find` only by returning the new region at `i`. -/
theorem find_set_data : ∀ (m : Mem) (x k i : Nat) (r r' : Region),
    m[i]? = some r → r'.base = r.base → r'.hi = r.hi → r'.rw = r.rw →
    Mem.find (m.set i r') x k =
      (Mem.find m x k).map (fun p => if p.1 - k = i then (p.1, r') else p)
  | [], _, _, _, _, _, h, _, _, _ => by simp at h
  | s :: rest, x, k, 0, r, r', h, hb, hh, _ => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at h
    subst h
    simp only [List.set_cons_zero, Mem.find]
    by_cases hc : s.base ≤ x ∧ x < s.hi
    · rw [if_pos hc, if_pos ⟨by rw [hb]; exact hc.1, by rw [hh]; exact hc.2⟩]
      simp
    · rw [if_neg hc, if_neg (by rwa [hb, hh])]
      cases hf : Mem.find rest x (k + 1) with
      | none => rfl
      | some p =>
        have hge := (find_spec rest x (k + 1) hf).1
        simp only [Option.map_some]
        rw [if_neg (by omega)]
  | s :: rest, x, k, i + 1, r, r', h, hb, hh, hrw => by
    have h' : rest[i]? = some r := by simpa using h
    simp only [List.set_cons_succ, Mem.find]
    by_cases hc : s.base ≤ x ∧ x < s.hi
    · rw [if_pos hc, if_pos hc]
      simp
    · rw [if_neg hc, if_neg hc, find_set_data rest x (k + 1) i r r' h' hb hh hrw]
      cases hf : Mem.find rest x (k + 1) with
      | none => rfl
      | some p =>
        have hge := (find_spec rest x (k + 1) hf).1
        simp only [Option.map_some]
        by_cases hp : p.1 - (k + 1) = i
        · rw [if_pos hp, if_pos (by omega)]
        · rw [if_neg hp, if_neg (by omega)]

/-- The region `Mem.find` returns contains the address. -/
theorem find_range : ∀ (m : Mem) (a k : Nat) {j : Nat} {r : Region},
    Mem.find m a k = some (j, r) → r.base ≤ a ∧ a < r.hi
  | [], _, _, _, _, h => by simp [Mem.find] at h
  | s :: rest, a, k, j, r, h => by
    simp only [Mem.find] at h
    split at h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨_, rfl⟩ := h
      assumption
    · exact find_range rest a (k + 1) h

/-- A successful write reads back at its own address. -/
theorem rd_wr_same {m m' : Mem} {a v : Nat} (h : m.wr a v = some m') :
    m'.rd a = some (v % 256) := by
  simp only [Mem.wr, bind, Option.bind] at h
  split at h
  · contradiction
  · rename_i p hp
    obtain ⟨i, r⟩ := p
    have hr := (find_spec m a 0 hp).2
    simp only [Nat.sub_zero] at hr
    have hrng := find_range m a 0 hp
    dsimp only at h
    split at h
    · simp only [Option.some.injEq, pure] at h
      subst h
      have hfind : Mem.find (m.set i { r with data := r.data.set (a - r.base) (v % 256) }) a 0
          = some (i, { r with data := r.data.set (a - r.base) (v % 256) }) := by
        rw [find_set_data m a 0 i r { r with data := r.data.set (a - r.base) (v % 256) }
          hr rfl (by simp [Region.hi]) rfl, hp]
        simp
      have hlt : a - r.base < r.data.length := by
        have := hrng.2
        unfold Region.hi at this
        omega
      simp only [Mem.rd, hfind]
      show (r.data.set (a - r.base) (v % 256))[a - r.base]? = some (v % 256)
      simp [hlt]
    · contradiction

/-- A write at `a` does not change reads at other addresses. -/
theorem rd_wr_other {m m' : Mem} {a v x : Nat} (h : m.wr a v = some m') (hx : x ≠ a) :
    m'.rd x = m.rd x := by
  simp only [Mem.wr, bind, Option.bind] at h
  split at h
  · contradiction
  · rename_i p hp
    obtain ⟨i, r⟩ := p
    have hr := (find_spec m a 0 hp).2
    simp only [Nat.sub_zero] at hr
    dsimp only at h
    split at h
    · simp only [Option.some.injEq, pure] at h
      subst h
      have hfind := find_set_data m x 0 i r { r with data := r.data.set (a - r.base) (v % 256) }
        hr rfl (by simp [Region.hi]) rfl
      simp only [Mem.rd, hfind]
      cases hf : Mem.find m x 0 with
      | none => rfl
      | some p =>
        obtain ⟨j, q⟩ := p
        have hrngx := find_range m x 0 hf
        have hrnga := find_range m a 0 hp
        simp only [Option.map_some]
        by_cases hj : j - 0 = i
        · rw [if_pos hj]
          have hji : j = i := by omega
          have hq : q = r := by
            have h1 := (find_spec m x 0 hf).2
            have h2 := hr
            simp only [Nat.sub_zero] at h1
            rw [hji] at h1
            rw [h1] at h2
            exact Option.some.inj h2
          subst hq
          show (q.data.set (a - q.base) (v % 256))[x - q.base]? = q.data[x - q.base]?
          simp only [List.getElem?_set]
          rw [if_neg (fun hc => hx (by omega))]
        · rw [if_neg hj]
    · contradiction

/-- Writes at `a, a+1, …` do not change a read outside that range. -/
theorem rd_wrN_other : ∀ (bs : List Nat) (m : Mem) (a x : Nat) {m' : Mem},
    m.wrN a bs = some m' → (∀ i, i < bs.length → x ≠ a + i) → m'.rd x = m.rd x
  | [], m, _, _, m', h, _ => by
    simp only [Mem.wrN, Option.some.injEq] at h
    rw [← h]
  | b :: bs, m, a, x, m', h, hx => by
    simp only [Mem.wrN, bind, Option.bind] at h
    split at h
    · contradiction
    · rename_i m1 hm1
      have e1 : m'.rd x = m1.rd x :=
        rd_wrN_other bs m1 (a + 1) x h (fun i hi => by
          have := hx (i + 1) (by simp at hi ⊢; omega)
          rwa [show a + (i + 1) = a + 1 + i from by omega] at this)
      have e2 : m1.rd x = m.rd x :=
        rd_wr_other hm1 (by have := hx 0 (by simp); simpa using this)
      rw [e1, e2]

/-- A multi-byte write reads back as the bytes written. -/
theorem rdN_wrN_same : ∀ (bs : List Nat) (m : Mem) (a : Nat) {m' : Mem},
    m.wrN a bs = some m' → m'.rdN a bs.length = some (bs.map (· % 256))
  | [], m, _, m', h => by
    simp only [Mem.wrN, Option.some.injEq] at h
    subst h
    rfl
  | b :: bs, m, a, m', h => by
    simp only [Mem.wrN, bind, Option.bind] at h
    split at h
    · contradiction
    · rename_i m1 hm1
      have ih := rdN_wrN_same bs m1 (a + 1) h
      have hda : m'.rd a = some (b % 256) := by
        rw [rd_wrN_other bs m1 (a + 1) a h (fun i _ => by omega)]
        exact rd_wr_same hm1
      show m'.rdN a (bs.length + 1) = _
      rw [Mem.rdN]
      simp only [hda, ih, bind, Option.bind, pure, List.map_cons]

/-- `leVal` over an append splits at the boundary. -/
theorem leVal_append : ∀ (xs ys : List Nat),
    leVal (xs ++ ys) = leVal xs + 256 ^ xs.length * leVal ys
  | [], _ => by simp [leVal]
  | x :: xs, ys => by
    simp only [List.cons_append, leVal, List.length_cons]
    rw [leVal_append xs ys, Nat.mul_add]
    have hm : 256 * (256 ^ xs.length * leVal ys) = 256 ^ (xs.length + 1) * leVal ys := by
      rw [Nat.pow_succ, ← Nat.mul_assoc 256 (256 ^ xs.length) (leVal ys),
        Nat.mul_comm 256 (256 ^ xs.length)]
    rw [hm, Nat.add_assoc]

/-- The digits of `v` in base 256 evaluate to `v` modulo `256^n`. -/
theorem leVal_digits : ∀ (n v : Nat),
    leVal ((List.range n).map (fun i => v / 256 ^ i % 256)) = v % 256 ^ n
  | 0, v => by
    show leVal ([] : List Nat) = v % 1
    rw [Nat.mod_one]
    rfl
  | n + 1, v => by
    rw [List.range_succ, List.map_append, List.map_singleton, leVal_append,
      List.length_map, List.length_range, leVal_digits n v]
    simp only [leVal, Nat.add_zero, Nat.mul_zero]
    have h3 : 256 ^ (n + 1) = 256 ^ n * 256 := Nat.pow_succ 256 n
    rw [h3, Nat.mod_mul]

/-- `rd64` after `wr64` at the same address returns the value. -/
theorem rd64_wr64 {m m' : Mem} {a v : Nat} (h : wr64 m a v = some m') :
    rd64 m' a = some (v % W) := by
  unfold wr64 at h
  unfold rd64
  have hr := rdN_wrN_same (bytes8 v) m a h
  have hlen : (bytes8 v).length = 8 := by simp [bytes8]
  rw [hlen] at hr
  rw [hr]
  simp only [Option.map_some]
  have hmap : (bytes8 v).map (· % 256) = bytes8 v := by
    simp [bytes8]
  rw [hmap]
  have : leVal (bytes8 v) = v % W := by
    have hd := leVal_digits 8 v
    have hw : 256 ^ 8 = W := rfl
    rw [hw] at hd
    exact hd
  rw [this]

/-! ### The `L32` triple, proved: with `Ins.ld32` it is one step. -/

/-- The decode fact for `L32` at an arbitrary entry: `mov eax,[rdx]`,
with the caller's `ret` at `entry + 2` the triple's exit. -/
structure L32Code (im : Nat → Option (Ins × Nat)) (entry : Nat) : Prop where
  d0 : im entry = some (.ld32 0 2 0, 2)

theorem L32_triple {im : Nat → Option (Ins × Nat)} {entry : Nat}
    (hc : L32Code im entry) : L32Spec im entry := by
  intro a ha s hrip hpre
  obtain ⟨hrdx, hsome⟩ := hpre
  obtain ⟨bs, hbs⟩ := Option.isSome_iff_exists.mp hsome
  have hrd : rd32 s.mem (ea a 0) = some (leVal bs) := by
    unfold rd32 ea
    rw [Nat.add_zero, Nat.mod_eq_of_lt ha, hbs]
    rfl
  have hstep : X86Int.step im s = .ok { s.set 0 (some (leVal bs)) with rip := s.rip + 2 } :=
    step_ld32 (by rw [hrip]; exact hc.d0) hrdx hrd
  refine ⟨1, { s.set 0 (some (leVal bs)) with rip := s.rip + 2 }, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · show run im 1 (Out.ok s) = _
    rw [run_succ_ok im 0 s, hstep]
    rfl
  · show s.rip + 2 = entry + 2
    rw [hrip]
  · have hl : 0 < s.regs.length := Nat.lt_of_lt_of_le (by decide) (reg_some_len hrdx)
    show ({ s.set 0 (some (leVal bs)) with rip := s.rip + 2 }).reg 0 = _
    rw [reg_with_rip, reg_set_self hl]
    simp [hbs]
  · show ({ s.set 0 (some (leVal bs)) with rip := s.rip + 2 }).reg 2 = some a
    rw [reg_with_rip, reg_set_other (by decide), hrdx]
  · show ({ s.set 0 (some (leVal bs)) with rip := s.rip + 2 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  · intro i hi
    show ({ s.set 0 (some (leVal bs)) with rip := s.rip + 2 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]
  · show (s.set 0 (some (leVal bs))).regs.length = s.regs.length
    rw [len_set]


theorem reg_with_fl (t : X86Int.St) (f : Flags) (i : Nat) : ({ t with fl := f }).reg i = t.reg i := rfl
theorem reg_with_mem (t : X86Int.St) (m : Mem) (i : Nat) : ({ t with mem := m }).reg i = t.reg i := rfl
theorem len_with_fl (t : X86Int.St) (f : Flags) : ({ t with fl := f }).regs.length = t.regs.length := rfl
theorem len_with_mem (t : X86Int.St) (m : Mem) : ({ t with mem := m }).regs.length = t.regs.length := rfl
theorem len_with_rip (t : X86Int.St) (x : Nat) : ({ t with rip := x }).regs.length = t.regs.length := rfl
theorem mem_with_fl (t : X86Int.St) (f : Flags) : ({ t with fl := f }).mem = t.mem := rfl

theorem step_mov {im : Nat → Option (Ins × Nat)} {s : X86Int.St} {d r len v : Nat}
    (hi : im s.rip = some (.mov d r, len)) (hv : s.reg r = some v) :
    X86Int.step im s = .ok { s.set d (some v) with rip := s.rip + len } := by
  simp only [X86Int.step, exec, hi, hv, lift, bind, Option.bind, pure]

theorem step_alu_sub {im : Nat → Option (Ins × Nat)} {s : X86Int.St} {d r len a b : Nat}
    (hi : im s.rip = some (.alu .sub d r, len)) (ha : s.reg d = some a) (hb : s.reg r = some b) :
    X86Int.step im s = .ok { s.set d (some (aluF .sub a b).1) with
      fl := (aluF .sub a b).2, rip := s.rip + len } := by
  simp only [X86Int.step, exec, hi, ha, hb, lift, bind, Option.bind, pure]

theorem step_cmpI {im : Nat → Option (Ins × Nat)} {s : X86Int.St} {r imm len x : Nat}
    (hi : im s.rip = some (.cmpI r imm, len)) (hx : s.reg r = some x) :
    X86Int.step im s = .ok { s with fl := (aluF .sub x (imm % W)).2, rip := s.rip + len } := by
  simp only [X86Int.step, exec, hi, hx, lift, bind, Option.bind, pure]

theorem step_jcc {im : Nat → Option (Ins × Nat)} {s : X86Int.St} {cc t len : Nat} {c : Bool}
    (hi : im s.rip = some (.jcc cc t, len)) (hc : X86Int.cond s.fl cc = some c) :
    X86Int.step im s = .ok { s with rip := if c then t else s.rip + len } := by
  simp only [X86Int.step, exec, hi, hc, lift, bind, Option.bind, pure]

theorem step_aluI_add {im : Nat → Option (Ins × Nat)} {s : X86Int.St} {d imm len a : Nat}
    (hi : im s.rip = some (.aluI .add d imm, len)) (ha : s.reg d = some a) :
    X86Int.step im s = .ok { s.set d (some (aluF .add a (imm % W)).1) with
      fl := (aluF .add a (imm % W)).2, rip := s.rip + len } := by
  simp only [X86Int.step, exec, hi, ha, lift, bind, Option.bind, pure]

theorem cond_cc3 {f : Flags} {b : Bool} (h : f.cf = some b) : X86Int.cond f 3 = some (!b) := by
  simp [X86Int.cond, h]

theorem aluF_sub_fst (a b : Nat) : (aluF .sub a b).1 = (a + W - b) % W := rfl

theorem aluF_sub_cf {a b : Nat} (h : ¬ a < b) : (aluF .sub a b).2.cf = some false := by
  have hlt : decide (a < b) = false := by simp [h]
  simp [aluF, hlt]

theorem aluF_add_fst (a b : Nat) : (aluF .add a b).1 = (a + b) % W := rfl

/-! ### The `RDC` triple, proved. -/

/-- The decode facts for `RDC` at an arbitrary entry, calling `L32`
at `l32` (whose `ret` sits at `l32 + 2`). Offsets are the listing's:
mov 3, sub 3, jae 2, cmp 4 at +13, jae 2 at +17, mov 3 at +24,
call 5 at +27, add 4 at +32, ret at +36. -/
structure RDCCode (im : Nat → Option (Ins × Nat)) (entry l32 : Nat) : Prop where
  d0 : im entry = some (.mov 2 15, 3)
  d1 : im (entry + 3) = some (.alu .sub 2 6, 3)
  d2 : im (entry + 6) = some (.jcc 3 (entry + 13), 2)
  d3 : im (entry + 13) = some (.cmpI 2 4, 4)
  d4 : im (entry + 17) = some (.jcc 3 (entry + 24), 2)
  d5 : im (entry + 24) = some (.mov 2 6, 3)
  d6 : im (entry + 27) = some (.call l32, 5)
  d7 : im (entry + 32) = some (.aluI .add 6 4, 4)
  dret : im (l32 + 2) = some (.ret, 1)

theorem RDC_triple {im : Nat → Option (Ins × Nat)} {entry l32 : Nat}
    (hc : RDCCode im entry l32) (hl : L32Code im l32) (hentry : entry + 36 < W) :
    RDCSpec im entry := by
  intro c e hcW heW s hrip hpre
  obtain ⟨hr6, hr15, hle, hsome, hroom, ⟨sp, hr4⟩, hframe⟩ := hpre
  have hlen15 : 15 < s.regs.length := reg_some_len hr15
  have hlen : ∀ i, i ≤ 15 → i < s.regs.length := fun i hi => by omega
  -- step 1: mov rdx, r15
  have e1 : X86Int.step im s = .ok { s.set 2 (some e) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact hc.d0) hr15
  obtain ⟨s1, hs1⟩ : ∃ t : X86Int.St, t = { s.set 2 (some e) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have hs1rip : s1.rip = entry + 3 := by rw [hs1]; show s.rip + 3 = entry + 3; rw [hrip]
  have hs1r2 : s1.reg 2 = some e := by
    rw [hs1]; show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 2 = some e
    rw [reg_with_rip, reg_set_self (hlen 2 (by decide))]
  have hs1r6 : s1.reg 6 = some c := by
    rw [hs1]; show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr6
  -- step 2: sub rdx, rsi
  have e2 : X86Int.step im s1 = .ok { s1.set 2 (some (aluF .sub e c).1) with
      fl := (aluF .sub e c).2, rip := s1.rip + 3 } :=
    step_alu_sub (by rw [hs1rip]; exact hc.d1) hs1r2 hs1r6
  obtain ⟨s2, hs2⟩ : ∃ t : X86Int.St, t = { s1.set 2 (some (aluF .sub e c).1) with
    fl := (aluF .sub e c).2, rip := s1.rip + 3 } := ⟨_, rfl⟩
  have hsub : (aluF .sub e c).1 = e - c := by
    rw [aluF_sub_fst, show e + W - c = (e - c) + W by omega, Nat.add_mod_right,
      Nat.mod_eq_of_lt (by omega)]
  have hs2rip : s2.rip = entry + 6 := by
    rw [hs2]; show s1.rip + 3 = entry + 6; rw [hs1rip]
  have hlen1 : s1.regs.length = s.regs.length := by
    rw [hs1]; show ({ s.set 2 (some e) with rip := s.rip + 3 }).regs.length = s.regs.length
    rw [len_with_rip, len_set]
  have hs2r2 : s2.reg 2 = some (e - c) := by
    rw [hs2]
    show (s1.set 2 (some (aluF .sub e c).1)).reg 2 = some (e - c)
    rw [reg_set_self (by rw [hlen1]; exact hlen 2 (by decide)), hsub]
  have hs2r6 : s2.reg 6 = some c := by
    rw [hs2]
    show (s1.set 2 (some (aluF .sub e c).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact hs1r6
  have hs2fl : X86Int.cond s2.fl 3 = some true := by
    have hcf : s2.fl.cf = some false := by
      rw [hs2]; exact aluF_sub_cf (by omega)
    have := cond_cc3 hcf
    simpa using this
  -- step 3: jae entry+13
  have e3 : X86Int.step im s2 = .ok { s2 with rip := entry + 13 } := by
    have h := step_jcc (im := im) (s := s2) (by rw [hs2rip]; exact hc.d2) hs2fl
    simpa using h
  obtain ⟨s3, hs3⟩ : ∃ t : X86Int.St, t = { s2 with rip := entry + 13 } := ⟨_, rfl⟩
  have hs3rip : s3.rip = entry + 13 := by rw [hs3]
  have hs3r2 : s3.reg 2 = some (e - c) := by
    rw [hs3]; show ({ s2 with rip := entry + 13 }).reg 2 = some (e - c)
    rw [reg_with_rip]; exact hs2r2
  have hs3r6 : s3.reg 6 = some c := by
    rw [hs3]; show ({ s2 with rip := entry + 13 }).reg 6 = some c
    rw [reg_with_rip]; exact hs2r6
  -- step 4: cmp rdx, 4
  have e4 : X86Int.step im s3 = .ok { s3 with
      fl := (aluF .sub (e - c) (4 % W)).2, rip := s3.rip + 4 } :=
    step_cmpI (by rw [hs3rip]; exact hc.d3) hs3r2
  obtain ⟨s4, hs4⟩ : ∃ t : X86Int.St, t = { s3 with fl := (aluF .sub (e - c) (4 % W)).2, rip := s3.rip + 4 } := ⟨_, rfl⟩
  have hs4rip : s4.rip = entry + 17 := by rw [hs4]; show s3.rip + 4 = entry + 17; rw [hs3rip]
  have hs4r6 : s4.reg 6 = some c := by
    rw [hs4]
    show s3.reg 6 = some c
    exact hs3r6
  have h4W : 4 % W = 4 := Nat.mod_eq_of_lt (by unfold W; decide)
  have hs4fl : X86Int.cond s4.fl 3 = some true := by
    have hcf : s4.fl.cf = some false := by
      rw [hs4]
      show (aluF .sub (e - c) (4 % W)).2.cf = some false
      rw [h4W]; exact aluF_sub_cf (by omega)
    have := cond_cc3 hcf
    simpa using this
  -- step 5: jae entry+24
  have e5 : X86Int.step im s4 = .ok { s4 with rip := entry + 24 } := by
    have h := step_jcc (im := im) (s := s4) (by rw [hs4rip]; exact hc.d4) hs4fl
    simpa using h
  obtain ⟨s5, hs5⟩ : ∃ t : X86Int.St, t = { s4 with rip := entry + 24 } := ⟨_, rfl⟩
  have hs5rip : s5.rip = entry + 24 := by rw [hs5]
  have hs5r6 : s5.reg 6 = some c := by
    rw [hs5]; show ({ s4 with rip := entry + 24 }).reg 6 = some c
    rw [reg_with_rip]; exact hs4r6
  have hs5r4 : s5.reg 4 = some sp := by
    have a1 : s1.reg 4 = some sp := by
      rw [hs1]; show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 4 = some sp
      rw [reg_with_rip, reg_set_other (by decide : (4:Nat) ≠ 2)]; exact hr4
    have a2 : s2.reg 4 = some sp := by
      rw [hs2]; show (s1.set 2 (some (aluF .sub e c).1)).reg 4 = some sp
      rw [reg_set_other (by decide : (4:Nat) ≠ 2)]; exact a1
    have a3 : s3.reg 4 = some sp := by
      rw [hs3]; show ({ s2 with rip := entry + 13 }).reg 4 = some sp
      rw [reg_with_rip]; exact a2
    have a4 : s4.reg 4 = some sp := by
      rw [hs4]; show s3.reg 4 = some sp; exact a3
    rw [hs5]; show ({ s4 with rip := entry + 24 }).reg 4 = some sp
    rw [reg_with_rip]; exact a4
  have hs5mem : s5.mem = s.mem := by
    have m1 : s1.mem = s.mem := by
      rw [hs1]; show ({ s.set 2 (some e) with rip := s.rip + 3 }).mem = s.mem
      rw [mem_with_rip, mem_set]
    have m2 : s2.mem = s.mem := by
      rw [hs2]; show (s1.set 2 (some (aluF .sub e c).1)).mem = s.mem
      rw [mem_set]; exact m1
    have m3 : s3.mem = s.mem := by
      rw [hs3]; show ({ s2 with rip := entry + 13 }).mem = s.mem
      rw [mem_with_rip]; exact m2
    have m4 : s4.mem = s.mem := by
      rw [hs4]; show s3.mem = s.mem; exact m3
    rw [hs5]; show ({ s4 with rip := entry + 24 }).mem = s.mem
    rw [mem_with_rip]; exact m4
  -- step 6: mov rdx, rsi
  have e6 : X86Int.step im s5 = .ok { s5.set 2 (some c) with rip := s5.rip + 3 } :=
    step_mov (by rw [hs5rip]; exact hc.d5) hs5r6
  obtain ⟨s6, hs6⟩ : ∃ t : X86Int.St, t = { s5.set 2 (some c) with rip := s5.rip + 3 } := ⟨_, rfl⟩
  have hs6rip : s6.rip = entry + 27 := by rw [hs6]; show s5.rip + 3 = entry + 27; rw [hs5rip]
  have hs6r2 : s6.reg 2 = some c := by
    rw [hs6]; show ({ s5.set 2 (some c) with rip := s5.rip + 3 }).reg 2 = some c
    have hl5 : 2 < s5.regs.length := by
      have l2 : s2.regs.length = s.regs.length := by
        rw [hs2]; show (s1.set 2 (some (aluF .sub e c).1)).regs.length = s.regs.length
        rw [len_set, hlen1]
      have l3 : s3.regs.length = s.regs.length := by
        rw [hs3]; show ({ s2 with rip := entry + 13 }).regs.length = s.regs.length
        rw [len_with_rip]; exact l2
      have l4 : s4.regs.length = s.regs.length := by
        rw [hs4]; show s3.regs.length = s.regs.length; exact l3
      have l5 : s5.regs.length = s.regs.length := by
        rw [hs5]; show ({ s4 with rip := entry + 24 }).regs.length = s.regs.length
        rw [len_with_rip]; exact l4
      rw [l5]; exact hlen 2 (by decide)
    rw [reg_with_rip, reg_set_self hl5]
  have hs6r4 : s6.reg 4 = some sp := by
    rw [hs6]; show ({ s5.set 2 (some c) with rip := s5.rip + 3 }).reg 4 = some sp
    rw [reg_with_rip, reg_set_other (by decide : (4:Nat) ≠ 2)]; exact hs5r4
  have hs6r6 : s6.reg 6 = some c := by
    rw [hs6]; show ({ s5.set 2 (some c) with rip := s5.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide : (6:Nat) ≠ 2)]; exact hs5r6
  have hs6mem : s6.mem = s.mem := by
    rw [hs6]; show ({ s5.set 2 (some c) with rip := s5.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact hs5mem
  -- step 7: call l32
  obtain ⟨m, hm⟩ := Option.isSome_iff_exists.mp (hroom sp hr4 0 (by decide) (entry + 32))
  have hm' : wr64 s6.mem ((sp + W - 8) % W) (s6.rip + 5) = some m := by
    have hslot : (sp + W - 8 * (0 + 1)) % W = (sp + W - 8) % W := by simp
    rw [hs6mem, hs6rip]
    have := hm
    rwa [hslot] at this
  have e7 : X86Int.step im s6 = .ok { s6.set 4 (some ((sp + W - 8) % W)) with
      mem := m, rip := l32 } :=
    step_call (by rw [hs6rip]; exact hc.d6) hs6r4 hm'
  obtain ⟨s7, hs7⟩ : ∃ t : X86Int.St, t = { s6.set 4 (some ((sp + W - 8) % W)) with
    mem := m, rip := l32 } := ⟨_, rfl⟩
  have hs7rip : s7.rip = l32 := by rw [hs7]
  have hs7r2 : s7.reg 2 = some c := by
    rw [hs7]
    show (s6.set 4 (some ((sp + W - 8) % W))).reg 2 = some c
    rw [reg_set_other (by decide : (2:Nat) ≠ 4)]; exact hs6r2
  have hs7r4 : s7.reg 4 = some ((sp + W - 8) % W) := by
    rw [hs7]
    show (s6.set 4 (some ((sp + W - 8) % W))).reg 4 = some ((sp + W - 8) % W)
    have hl6 : 4 < s6.regs.length := reg_some_len hs6r4
    rw [reg_set_self hl6]
  have hs7r6 : s7.reg 6 = some c := by
    rw [hs7]
    show (s6.set 4 (some ((sp + W - 8) % W))).reg 6 = some c
    rw [reg_set_other (by decide : (6:Nat) ≠ 4)]; exact hs6r6
  have hs7mem : s7.mem = m := by rw [hs7]
  have hframem : m.rdN c 4 = s.mem.rdN c 4 := hframe sp m hr4 (by
    have hh := hm
    rwa [show (sp + W - 8 * (0 + 1)) % W = (sp + W - 8) % W from by simp] at hh)
  -- steps 8..: the L32 triple as a black box
  obtain ⟨n8, s8, hrun8, hs8rip, hs8r0, hs8r2, hs8mem, hs8other, -⟩ :=
    L32_triple hl c hcW s7 hs7rip ⟨hs7r2, by rw [hs7mem, hframem]; exact hsome⟩
  -- ret at l32 + 2
  have hs8r4 : s8.reg 4 = some ((sp + W - 8) % W) := by
    rw [hs8other 4 (by decide)]; exact hs7r4
  have hrd64 : rd64 s8.mem ((sp + W - 8) % W) = some (entry + 32) := by
    have h1 : rd64 m ((sp + W - 8) % W) = some ((entry + 32) % W) := by
      apply rd64_wr64
      have hh := hm
      rwa [show (sp + W - 8 * (0 + 1)) % W = (sp + W - 8) % W from by simp] at hh
    rw [hs8mem, hs7mem, h1, Nat.mod_eq_of_lt (by omega)]
  have e9 : X86Int.step im s8 = .ok { s8.set 4 (some (((sp + W - 8) % W + 8) % W)) with
      rip := entry + 32 } :=
    step_ret (by rw [hs8rip]; exact hc.dret) hs8r4 hrd64
  obtain ⟨s9, hs9⟩ : ∃ t : X86Int.St, t = { s8.set 4 (some (((sp + W - 8) % W + 8) % W)) with
    rip := entry + 32 } := ⟨_, rfl⟩
  have hs9rip : s9.rip = entry + 32 := by rw [hs9]
  have hs9r6 : s9.reg 6 = some c := by
    rw [hs9]
    show ({ s8.set 4 (some (((sp + W - 8) % W + 8) % W)) with rip := entry + 32 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide : (6:Nat) ≠ 4), hs8other 6 (by decide)]
    exact hs7r6
  have hs9r0 : s9.reg 0 = (s.mem.rdN c 4).map leVal := by
    rw [hs9]
    show ({ s8.set 4 (some (((sp + W - 8) % W + 8) % W)) with rip := entry + 32 }).reg 0
      = (s.mem.rdN c 4).map leVal
    rw [reg_with_rip, reg_set_other (by decide : (0:Nat) ≠ 4)]
    have h0 : s8.reg 0 = (s7.mem.rdN c 4).map leVal := hs8r0
    rwa [hs7mem, hframem] at h0
  have hs9mem : s9.mem = m := by
    rw [hs9]
    show ({ s8.set 4 (some (((sp + W - 8) % W + 8) % W)) with rip := entry + 32 }).mem = m
    rw [mem_with_rip, mem_set, hs8mem, hs7mem]
  -- step 10: add rsi, 4
  have e10 : X86Int.step im s9 = .ok { s9.set 6 (some (aluF .add c (4 % W)).1) with
      fl := (aluF .add c (4 % W)).2, rip := s9.rip + 4 } :=
    step_aluI_add (by rw [hs9rip]; exact hc.d7) hs9r6
  obtain ⟨s10, hs10⟩ : ∃ t : X86Int.St, t = { s9.set 6 (some (aluF .add c (4 % W)).1) with
    fl := (aluF .add c (4 % W)).2, rip := s9.rip + 4 } := ⟨_, rfl⟩
  have hadd : (aluF .add c (4 % W)).1 = c + 4 := by
    rw [aluF_add_fst, h4W, Nat.mod_eq_of_lt (by omega)]
  have r1 : run im 1 (Out.ok s) = Out.ok s1 := by rw [run_succ_ok im 0 s, e1, ← hs1, run_zero]
  have r2 : run im 1 (Out.ok s1) = Out.ok s2 := by rw [run_succ_ok im 0 s1, e2, ← hs2, run_zero]
  have r3 : run im 1 (Out.ok s2) = Out.ok s3 := by rw [run_succ_ok im 0 s2, e3, ← hs3, run_zero]
  have r4 : run im 1 (Out.ok s3) = Out.ok s4 := by rw [run_succ_ok im 0 s3, e4, ← hs4, run_zero]
  have r5 : run im 1 (Out.ok s4) = Out.ok s5 := by rw [run_succ_ok im 0 s4, e5, ← hs5, run_zero]
  have r6 : run im 1 (Out.ok s5) = Out.ok s6 := by rw [run_succ_ok im 0 s5, e6, ← hs6, run_zero]
  have r7 : run im 1 (Out.ok s6) = Out.ok s7 := by rw [run_succ_ok im 0 s6, e7, ← hs7, run_zero]
  have r9 : run im 1 (Out.ok s8) = Out.ok s9 := by rw [run_succ_ok im 0 s8, e9, ← hs9, run_zero]
  have r10 : run im 1 (Out.ok s9) = Out.ok s10 := by rw [run_succ_ok im 0 s9, e10, ← hs10, run_zero]
  have h7run : run im 7 (Out.ok s) = Out.ok s7 := by
    rw [show (7:Nat) = 1 + 1 + 1 + 1 + 1 + 1 + 1 from rfl, run_add im 1 6, r1, run_add im 1 5,
      r2, run_add im 1 4, r3, run_add im 1 3, r4, run_add im 1 2, r5, run_add im 1 1, r6, r7]
  refine ⟨7 + (n8 + (1 + 1)), s10, ?_, ?_, ?_, ?_, ?_⟩
  · rw [run_add im 7 (n8 + (1 + 1)), h7run, run_add im n8 (1 + 1), hrun8, run_add im 1 1, r9, r10]
  · rw [hs10]; show s9.rip + 4 = entry + 0x24; rw [hs9rip]
  · rw [hs10]
    show (s9.set 6 (some (aluF .add c (4 % W)).1)).reg 0 = (s.mem.rdN c 4).map leVal
    rw [reg_set_other (by decide : (0:Nat) ≠ 6)]; exact hs9r0
  · rw [hs10]
    show (s9.set 6 (some (aluF .add c (4 % W)).1)).reg 6 = some (c + 4)
    have hl9 : 6 < s9.regs.length := reg_some_len hs9r6
    rw [reg_set_self hl9, hadd]
  · rw [hs10]
    show (s9.set 6 (some (aluF .add c (4 % W)).1)).mem.rdN c 4 = s.mem.rdN c 4
    rw [mem_set, hs9mem]; exact hframem


/-! ### Frame lemmas for multi-byte reads and preserved writability -/

/-- A write outside `[x, x+n)` does not change an `n`-byte read there. -/
theorem rdN_wrN_other : ∀ (n : Nat) (bs : List Nat) (m : Mem) (a x : Nat) {m' : Mem},
    m.wrN a bs = some m' → (∀ i, i < bs.length → ∀ j, j < n → x + j ≠ a + i) →
    m'.rdN x n = m.rdN x n
  | 0, _, _, _, _, _, _, _ => rfl
  | n + 1, bs, m, a, x, m', h, hx => by
    rw [Mem.rdN, Mem.rdN]
    have e1 : m'.rd x = m.rd x :=
      rd_wrN_other bs m a x h (fun i hi => by
        have := hx i hi 0 (by omega)
        simpa using this)
    have e2 : m'.rdN (x + 1) n = m.rdN (x + 1) n :=
      rdN_wrN_other n bs m a (x + 1) h (fun i hi j hj => by
        have := hx i hi (j + 1) (by omega)
        rwa [show x + (j + 1) = x + 1 + j from by omega] at this)
    rw [e1, e2]

/-- The write-relevant shape of a memory: base, extent and `rw` per
region. Writability of any address depends only on it. -/
def wshape (m : Mem) : List (Nat × Nat × Bool) := m.map fun r => (r.base, r.hi, r.rw)

theorem wr_wshape {m m' : Mem} {a v : Nat} (h : m.wr a v = some m') :
    wshape m' = wshape m := by
  simp only [Mem.wr, bind, Option.bind] at h
  split at h
  · contradiction
  · rename_i p hp
    obtain ⟨i, r⟩ := p
    have hr := (find_spec m a 0 hp).2
    simp only [Nat.sub_zero] at hr
    dsimp only at h
    split at h
    · simp only [Option.some.injEq, pure] at h
      subst h
      unfold wshape
      rw [List.map_set]
      apply List.ext_getElem?
      intro j
      rw [List.getElem?_set]
      by_cases hj : i = j
      · subst hj
        obtain ⟨hi, he⟩ := List.getElem?_eq_some_iff.1 hr
        simp [hi, he, Region.hi, List.length_map]
      · simp [hj]
    · contradiction

theorem wrN_wshape : ∀ (bs : List Nat) (m : Mem) (a : Nat) {m' : Mem},
    m.wrN a bs = some m' → wshape m' = wshape m
  | [], m, _, m', h => by
    simp only [Mem.wrN, Option.some.injEq] at h
    rw [← h]
  | b :: bs, m, a, m', h => by
    simp only [Mem.wrN, bind, Option.bind] at h
    split at h
    · contradiction
    · rename_i m1 hm1
      rw [wrN_wshape bs m1 (a + 1) h, wr_wshape hm1]

/-- `Mem.find`'s index and `rw` depend only on the write shape. -/
theorem find_wshape : ∀ (m m' : Mem), wshape m' = wshape m → ∀ (x k : Nat),
    (Mem.find m' x k).map (fun p => (p.1, p.2.rw)) =
      (Mem.find m x k).map (fun p => (p.1, p.2.rw))
  | [], [], _, _, _ => rfl
  | [], _ :: _, h, _, _ => by simp [wshape] at h
  | _ :: _, [], h, _, _ => by simp [wshape] at h
  | s :: rest, s' :: rest', h, x, k => by
    have hh : (s'.base, s'.hi, s'.rw) = (s.base, s.hi, s.rw) := by
      have := h
      simp only [wshape, List.map_cons, List.cons.injEq] at this
      exact this.1
    have hrest : wshape rest' = wshape rest := by
      have := h
      simp only [wshape, List.map_cons, List.cons.injEq] at this
      exact this.2
    have hb : s'.base = s.base := congrArg Prod.fst hh
    have hhi : s'.hi = s.hi := congrArg (fun p => p.2.1) hh
    have hrw : s'.rw = s.rw := congrArg (fun p => p.2.2) hh
    simp only [Mem.find]
    by_cases hc : s.base ≤ x ∧ x < s.hi
    · rw [if_pos hc, if_pos ⟨by rw [hb]; exact hc.1, by rw [hhi]; exact hc.2⟩]
      simp [hrw]
    · rw [if_neg hc, if_neg (by rwa [hb, hhi])]
      exact find_wshape rest rest' hrest x (k + 1)

/-- Writability transfers across memories of the same write shape. -/
theorem wr_some_of_wshape {m m' : Mem} (hsh : wshape m' = wshape m) {x u : Nat}
    (hx : (m.wr x u).isSome) : (m'.wr x u).isSome := by
  obtain ⟨mx, hmx⟩ := Option.isSome_iff_exists.mp hx
  simp only [Mem.wr, bind, Option.bind] at hmx
  split at hmx
  · contradiction
  · rename_i q hq
    obtain ⟨j, rq⟩ := q
    have hrw : rq.rw = true := by
      dsimp only at hmx
      split at hmx
      · assumption
      · contradiction
    have hf := find_wshape m m' hsh x 0
    rw [hq] at hf
    cases hf' : Mem.find m' x 0 with
    | none => simp [hf'] at hf
    | some q' =>
      obtain ⟨j', rq'⟩ := q'
      rw [hf'] at hf
      simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at hf
      unfold Mem.wr
      rw [hf']
      show (if rq'.rw = true then
        some (m'.set j' { rq' with data := rq'.data.set (x - rq'.base) (u % 256) })
        else none).isSome = true
      rw [if_pos (by rw [hf.2]; exact hrw)]
      rfl

/-- Multi-byte writability transfers across the same write shape. -/
theorem wrN_some_of_wshape : ∀ (cs : List Nat) (m m' : Mem), wshape m' = wshape m →
    ∀ (x : Nat), (m.wrN x cs).isSome → (m'.wrN x cs).isSome
  | [], _, _, _, _, hx => hx
  | c :: cs, m, m', hsh, x, hx => by
    cases hm1 : m.wr x c with
    | none => simp [Mem.wrN, hm1] at hx
    | some m1 =>
      have hrest : (m1.wrN (x + 1) cs).isSome := by
        simpa [Mem.wrN, hm1] using hx
      obtain ⟨m1', hm1'⟩ := Option.isSome_iff_exists.mp (wr_some_of_wshape hsh (x := x) (u := c)
        (Option.isSome_iff_exists.mpr ⟨m1, hm1⟩))
      have hgoal : m'.wrN x (c :: cs) = m1'.wrN (x + 1) cs := by
        simp [Mem.wrN, hm1']
      rw [hgoal]
      exact wrN_some_of_wshape cs m1 m1'
        (by rw [wr_wshape hm1', wr_wshape hm1]; exact hsh) (x + 1) hrest

/-- `wr64` writability survives any successful `wr64`. -/
theorem wr64_some_of_wr64 {m m' : Mem} {a v x u : Nat} (h : wr64 m a v = some m')
    (hx : (wr64 m x u).isSome) : (wr64 m' x u).isSome :=
  wrN_some_of_wshape _ m m' (wrN_wshape _ m a h) x hx


theorem step_push {im : Nat → Option (Ins × Nat)} {s : X86Int.St} {r len v sp : Nat} {m : Mem}
    (hi : im s.rip = some (.push r, len)) (hv : s.reg r = some v) (hsp : s.reg 4 = some sp)
    (hm : wr64 s.mem ((sp + W - 8) % W) v = some m) :
    X86Int.step im s = .ok { s.set 4 (some ((sp + W - 8) % W)) with mem := m, rip := s.rip + len } := by
  simp only [X86Int.step, exec, hi, hv, hsp, hm, lift, bind, Option.bind, pure]

theorem step_pop {im : Nat → Option (Ins × Nat)} {s : X86Int.St} {d len sp v : Nat}
    (hi : im s.rip = some (.pop d, len)) (hsp : s.reg 4 = some sp)
    (hv : rd64 s.mem sp = some v) :
    X86Int.step im s = .ok { (s.set 4 (some ((sp + 8) % W))).set d (some v) with
      rip := s.rip + len } := by
  simp only [X86Int.step, exec, hi, hsp, hv, lift, bind, Option.bind, pure]

theorem step_ld {im : Nat → Option (Ins × Nat)} {s : X86Int.St} {d b disp len x v : Nat}
    (hi : im s.rip = some (.ld d b disp, len)) (hx : s.reg b = some x)
    (hv : rd64 s.mem (ea x disp) = some v) :
    X86Int.step im s = .ok { s.set d (some v) with rip := s.rip + len } := by
  simp only [X86Int.step, exec, hi, hx, hv, lift, bind, Option.bind, pure]

theorem step_test {im : Nat → Option (Ins × Nat)} {s : X86Int.St} {a b len x y : Nat}
    (hi : im s.rip = some (.test a b, len)) (hx : s.reg a = some x) (hy : s.reg b = some y) :
    X86Int.step im s = .ok { s with fl := (aluF .and x y).2, rip := s.rip + len } := by
  simp only [X86Int.step, exec, hi, hx, hy, lift, bind, Option.bind, pure]

theorem step_cmp {im : Nat → Option (Ins × Nat)} {s : X86Int.St} {a b len x y : Nat}
    (hi : im s.rip = some (.cmp a b, len)) (hx : s.reg a = some x) (hy : s.reg b = some y) :
    X86Int.step im s = .ok { s with fl := (aluF .sub x y).2, rip := s.rip + len } := by
  simp only [X86Int.step, exec, hi, hx, hy, lift, bind, Option.bind, pure]

theorem step_jmp {im : Nat → Option (Ins × Nat)} {s : X86Int.St} {t len : Nat}
    (hi : im s.rip = some (.jmp t, len)) :
    X86Int.step im s = .ok { s with rip := t } := by
  simp only [X86Int.step, exec, hi]

theorem step_aluI_sub {im : Nat → Option (Ins × Nat)} {s : X86Int.St} {d imm len a : Nat}
    (hi : im s.rip = some (.aluI .sub d imm, len)) (ha : s.reg d = some a) :
    X86Int.step im s = .ok { s.set d (some (aluF .sub a (imm % W)).1) with
      fl := (aluF .sub a (imm % W)).2, rip := s.rip + len } := by
  simp only [X86Int.step, exec, hi, ha, lift, bind, Option.bind, pure]

theorem step_alu_xor {im : Nat → Option (Ins × Nat)} {s : X86Int.St} {d r len a b : Nat}
    (hi : im s.rip = some (.alu .xor d r, len)) (ha : s.reg d = some a) (hb : s.reg r = some b) :
    X86Int.step im s = .ok { s.set d (some (aluF .xor a b).1) with
      fl := (aluF .xor a b).2, rip := s.rip + len } := by
  simp only [X86Int.step, exec, hi, ha, hb, lift, bind, Option.bind, pure]

theorem cond_cc4 {f : Flags} {b : Bool} (h : f.zf = some b) : X86Int.cond f 4 = some b := by
  simp [X86Int.cond, h]

theorem aluF_zf (op : Alu) (a b : Nat) :
    (aluF op a b).2.zf = some ((aluF op a b).1 == 0) := rfl

theorem aluF_and_fst_self (a : Nat) : (aluF .and a a).1 = a := by
  simp [aluF]

theorem aluF_xor_fst_self (a : Nat) : (aluF .xor a a).1 = 0 := by
  simp [aluF]

/-- The stack slot `k` words below a non-wrapping `rsp`. -/
theorem slot_sub {sp k : Nat} (hk : 8 * k ≤ sp) (hsp : sp < W) :
    (sp + W - 8 * k) % W = sp - 8 * k := by
  rw [show sp + W - 8 * k = (sp - 8 * k) + W by omega, Nat.add_mod_right,
    Nat.mod_eq_of_lt (by omega)]

theorem slot_add8 {sp : Nat} (h8 : 8 ≤ sp) (hsp : sp < W) : (sp - 8 + 8) % W = sp := by
  rw [show sp - 8 + 8 = sp by omega, Nat.mod_eq_of_lt hsp]


/-! ### Support for the `ANNL` proof -/

theorem subW_eq_zero {a b : Nat} (ha : a < W) (hb : b < W) :
    ((a + W - b) % W == 0) = (a == b) := by
  by_cases h : a = b
  · subst h
    simp
  · have hne : (a + W - b) % W ≠ 0 := by
      by_cases hle : b ≤ a
      · rw [show a + W - b = (a - b) + W by omega, Nat.add_mod_right,
          Nat.mod_eq_of_lt (by omega)]
        omega
      · rw [Nat.mod_eq_of_lt (by omega)]
        omega
    have e1 : ((a + W - b) % W == 0) = false := by simp [hne]
    have e2 : (a == b) = false := by simp [h]
    rw [e1, e2]

theorem AnnAt_of_mem {s s' : X86Int.St} (h : s'.mem = s.mem) {e t d a : Nat}
    (ha : AnnAt s e t d a) : AnnAt s' e t d a := by
  obtain ⟨h1, h2, h3⟩ := ha
  exact ⟨by rw [h]; exact h1, by rw [h]; exact h2, by rw [h]; exact h3⟩

theorem AnnLayout_of_mem {s s' : X86Int.St} (h : s'.mem = s.mem)
    {ann : List (Nat × Height)} {p : Nat} (hl : AnnLayout s ann p) : AnnLayout s' ann p :=
  fun i hi => ⟨AnnAt_of_mem h (hl i hi).1, (hl i hi).2⟩

theorem rdN_wr64_other {m : Mem} {a v x n : Nat} {m' : Mem} (hm : wr64 m a v = some m')
    (hd : a + 8 ≤ x ∨ x + n ≤ a) : m'.rdN x n = m.rdN x n := by
  have hl : (bytes8 v).length = 8 := by simp [bytes8]
  exact rdN_wrN_other n (bytes8 v) m a x hm (fun i hi j hj => by
    rw [hl] at hi
    omega)

theorem rd64_wr64_other {m : Mem} {a v x : Nat} {m' : Mem} (hm : wr64 m a v = some m')
    (hd : a + 8 ≤ x ∨ x + 8 ≤ a) : rd64 m' x = rd64 m x := by
  unfold rd64
  rw [rdN_wr64_other hm hd]

/-- A stack-slot write disjoint from the ann array keeps the layout. -/
theorem AnnLayout_wr64 {s : X86Int.St} {ann : List (Nat × Height)} {p a v : Nat} {m : Mem}
    (hl : AnnLayout s ann p) (hm : wr64 s.mem a v = some m)
    (hd : a + 8 ≤ p ∨ p + 12 * ann.length ≤ a) :
    AnnLayout { s with mem := m } ann p := by
  intro i hi
  obtain ⟨⟨h1, h2, h3⟩, hw⟩ := hl i hi
  refine ⟨⟨?_, ?_, ?_⟩, hw⟩
  · show (m.rdN (p + 12 * i) 4).map leVal = some ann[i].1
    rw [rdN_wr64_other hm (by omega)]
    exact h1
  · show (m.rdN (p + 12 * i + 4) 4).map leVal = some ann[i].2.d
    rw [rdN_wr64_other hm (by omega)]
    exact h2
  · show (m.rdN (p + 12 * i + 8) 4).map leVal = some ann[i].2.a
    rw [rdN_wr64_other hm (by omega)]
    exact h3

theorem AnnLayout_cons {s : X86Int.St} {x : Nat × Height} {xs : List (Nat × Height)} {p : Nat} :
    AnnLayout s (x :: xs) p ↔
      (AnnAt s p x.1 x.2.d x.2.a ∧ x.1 < W) ∧ AnnLayout s xs (p + 12) := by
  constructor
  · intro h
    constructor
    · have h0 := h 0 (by simp)
      simpa using h0
    · intro i hi
      have hs := h (i + 1) (by simp [hi])
      have e1 : p + 12 * (i + 1) = p + 12 + 12 * i := by omega
      rw [e1] at hs
      simpa only [List.getElem_cons_succ] using hs
  · rintro ⟨hx, hxs⟩ i hi
    cases i with
    | zero => simpa using hx
    | succ i =>
      have hs := hxs i (by
        have : i + 1 < (x :: xs).length := hi
        simpa using this)
      have e1 : p + 12 * (i + 1) = p + 12 + 12 * i := by omega
      rw [e1]
      simpa only [List.getElem_cons_succ] using hs

theorem lookup_cons_eq {t : Nat} {x : Nat × Height} {xs : List (Nat × Height)} (h : x.1 = t) :
    (x :: xs).lookup t = some x.2 := by
  simp [List.lookup, h]

theorem lookup_cons_ne {t : Nat} {x : Nat × Height} {xs : List (Nat × Height)} (h : x.1 ≠ t) :
    (x :: xs).lookup t = xs.lookup t := by
  have hb : (t == x.1) = false := by simp [show t ≠ x.1 from fun hh => h hh.symm]
  show (match t == x.1 with | true => some x.2 | false => xs.lookup t) = xs.lookup t
  rw [hb]

theorem lookup_nil {t : Nat} : (List.lookup t [] : Option Height) = none := rfl


/-- The decode facts for `ANNL` at an arbitrary entry, calling `L32`
at `l32`. Offsets are the listing's (entry `0x271`, loop at `+9`,
none at `+42`, found at `+46`, rets at `+45` and `+50`). -/
structure ANNLCode (im : Nat → Option (Ins × Nat)) (entry l32 : Nat) : Prop where
  d0 : im entry = some (.push 1, 1)
  d1 : im (entry + 1) = some (.ld 0 7 48, 4)
  d2 : im (entry + 5) = some (.ld 1 7 32, 4)
  d3 : im (entry + 9) = some (.test 1 1, 3)
  d4 : im (entry + 12) = some (.jcc 4 (entry + 42), 2)
  d5 : im (entry + 14) = some (.mov 2 0, 3)
  d6 : im (entry + 17) = some (.push 0, 1)
  d7 : im (entry + 18) = some (.call l32, 5)
  d8 : im (entry + 23) = some (.pop 2, 1)
  d9 : im (entry + 24) = some (.cmp 0 12, 3)
  d10 : im (entry + 27) = some (.jcc 4 (entry + 46), 2)
  d11 : im (entry + 29) = some (.mov 0 2, 3)
  d12 : im (entry + 32) = some (.aluI .add 0 12, 4)
  d13 : im (entry + 36) = some (.aluI .sub 1 1, 4)
  d14 : im (entry + 40) = some (.jmp (entry + 9), 2)
  d15 : im (entry + 42) = some (.alu .xor 0 0, 2)
  d16 : im (entry + 44) = some (.pop 1, 1)
  d17 : im (entry + 46) = some (.mov 0 2, 3)
  d18 : im (entry + 49) = some (.pop 1, 1)
  dret : im (l32 + 2) = some (.ret, 1)
  dret_none : im (entry + 45) = some (.ret, 1)
  dret_found : im (entry + 50) = some (.ret, 1)

theorem len_of_reg {s : X86Int.St} {t : Nat} (h : s.reg 12 = some t) {i : Nat}
    (hi : i ≤ 12) : i < s.regs.length := by
  have := reg_some_len h
  omega


/-- The ANNL loop, by induction on the remaining ann list. Entry is
the loop head at `entry + 9` with `rax` the current entry pointer,
`rcx` the remaining count, `rsp` one word below the caller's (the
saved `rcx` sits there). The two deeper stack words are writable;
reads outside the three stack words are preserved. -/
theorem annl_loop {im : Nat → Option (Ins × Nat)} {entry l32 : Nat}
    (hc : ANNLCode im entry l32) (hl : L32Code im l32) (hentry : entry + 50 < W) :
    ∀ (ann : List (Nat × Height)) (t p sp sv : Nat) (s : X86Int.St),
      t < W → 0 < p → p + 12 * ann.length < W → 24 ≤ sp → sp < W →
      s.rip = entry + 9 → s.reg 0 = some p → s.reg 1 = some ann.length →
      s.reg 12 = some t → s.reg 4 = some (sp - 8) →
      rd64 s.mem (sp - 8) = some sv → AnnLayout s ann p →
      (∀ v, (wr64 s.mem (sp - 16) v).isSome) →
      (∀ v, (wr64 s.mem (sp - 24) v).isSome) →
      (sp ≤ p ∨ p + 12 * ann.length ≤ sp - 24) →
      ∃ n s', run im n (Out.ok s) = Out.ok s' ∧
        (s'.rip = entry + 45 ∨ s'.rip = entry + 50) ∧ AnnPost ann t s' ∧
        s'.reg 1 = some sv ∧ s'.reg 4 = some sp ∧ s'.reg 12 = some t ∧
        (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → s'.reg i = s.reg i) ∧
        (∀ x n, x + n ≤ sp - 24 ∨ sp ≤ x → s'.mem.rdN x n = s.mem.rdN x n) ∧
        (∃ e, s'.reg 0 = some e) := by
  intro ann
  induction ann with
  | nil =>
    intro t p sp sv s htW hp0 hbound hsp24 hspW hrip hr0 hr1 hr12 hr4 hsave hlay hw16 hw24 hdisj
    -- test rcx, rcx with rcx = 0
    have hr1' : s.reg 1 = some 0 := hr1
    have e1 : X86Int.step im s = .ok { s with fl := (aluF .and 0 0).2, rip := s.rip + 3 } :=
      step_test (by rw [hrip]; exact hc.d3) hr1' hr1'
    obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s with fl := (aluF .and 0 0).2, rip := s.rip + 3 } := ⟨_, rfl⟩
    have ha1rip : a1.rip = entry + 12 := by rw [ha1]; show s.rip + 3 = entry + 12; rw [hrip]
    have ha1fl : X86Int.cond a1.fl 4 = some true := by
      have hz : a1.fl.zf = some true := by
        rw [ha1]
        show (aluF .and 0 0).2.zf = some true
        rw [aluF_zf, aluF_and_fst_self]
        decide
      have := cond_cc4 hz
      simpa using this
    have ha1r0 : a1.reg 0 = some p := by rw [ha1]; show s.reg 0 = some p; exact hr0
    have ha1r4 : a1.reg 4 = some (sp - 8) := by rw [ha1]; show s.reg 4 = some (sp - 8); exact hr4
    have ha1r12 : a1.reg 12 = some t := by rw [ha1]; show s.reg 12 = some t; exact hr12
    have ha1mem : a1.mem = s.mem := by rw [ha1]
    have ga1 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → a1.reg i = s.reg i := by
      intro i _ _ _ _
      rw [ha1]
      show s.reg i = s.reg i
      rfl
    -- jz none
    have e2 : X86Int.step im a1 = .ok { a1 with rip := entry + 42 } := by
      have h := step_jcc (im := im) (s := a1) (by rw [ha1rip]; exact hc.d4) ha1fl
      simpa using h
    obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1 with rip := entry + 42 } := ⟨_, rfl⟩
    have ha2rip : a2.rip = entry + 42 := by rw [ha2]
    have ha2r0 : a2.reg 0 = some p := by
      rw [ha2]; show ({ a1 with rip := entry + 42 }).reg 0 = some p
      rw [reg_with_rip]; exact ha1r0
    have ha2r4 : a2.reg 4 = some (sp - 8) := by
      rw [ha2]; show ({ a1 with rip := entry + 42 }).reg 4 = some (sp - 8)
      rw [reg_with_rip]; exact ha1r4
    have ha2r12 : a2.reg 12 = some t := by
      rw [ha2]; show ({ a1 with rip := entry + 42 }).reg 12 = some t
      rw [reg_with_rip]; exact ha1r12
    have ha2mem : a2.mem = s.mem := by
      rw [ha2]; show ({ a1 with rip := entry + 42 }).mem = s.mem
      rw [mem_with_rip]; exact ha1mem
    have ga2 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → a2.reg i = s.reg i := by
      intro i h0 h1 h2 h4
      rw [ha2]; show ({ a1 with rip := entry + 42 }).reg i = s.reg i
      rw [reg_with_rip]; exact ga1 i h0 h1 h2 h4
    -- xor eax, eax
    have e3 : X86Int.step im a2 = .ok { a2.set 0 (some (aluF .xor p p).1) with
        fl := (aluF .xor p p).2, rip := a2.rip + 2 } :=
      step_alu_xor (by rw [ha2rip]; exact hc.d15) ha2r0 ha2r0
    obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2.set 0 (some (aluF .xor p p).1) with
      fl := (aluF .xor p p).2, rip := a2.rip + 2 } := ⟨_, rfl⟩
    have ha3rip : a3.rip = entry + 44 := by rw [ha3]; show a2.rip + 2 = entry + 44; rw [ha2rip]
    have ha3r0 : a3.reg 0 = some 0 := by
      rw [ha3]
      show (a2.set 0 (some (aluF .xor p p).1)).reg 0 = some 0
      rw [reg_set_self (len_of_reg ha2r12 (by decide)), aluF_xor_fst_self]
    have ha3r4 : a3.reg 4 = some (sp - 8) := by
      rw [ha3]
      show (a2.set 0 (some (aluF .xor p p).1)).reg 4 = some (sp - 8)
      rw [reg_set_other (by decide)]; exact ha2r4
    have ha3r12 : a3.reg 12 = some t := by
      rw [ha3]
      show (a2.set 0 (some (aluF .xor p p).1)).reg 12 = some t
      rw [reg_set_other (by decide)]; exact ha2r12
    have ha3mem : a3.mem = s.mem := by
      rw [ha3]
      show (a2.set 0 (some (aluF .xor p p).1)).mem = s.mem
      rw [mem_set]; exact ha2mem
    have ga3 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → a3.reg i = s.reg i := by
      intro i h0 h1 h2 h4
      rw [ha3]
      show (a2.set 0 (some (aluF .xor p p).1)).reg i = s.reg i
      rw [reg_set_other h0]; exact ga2 i h0 h1 h2 h4
    -- pop rcx
    have hsave3 : rd64 a3.mem (sp - 8) = some sv := by rw [ha3mem]; exact hsave
    have e4 : X86Int.step im a3 = .ok { (a3.set 4 (some ((sp - 8 + 8) % W))).set 1 (some sv) with
        rip := a3.rip + 1 } :=
      step_pop (by rw [ha3rip]; exact hc.d16) ha3r4 hsave3
    obtain ⟨a4, ha4⟩ : ∃ u : X86Int.St, u = { (a3.set 4 (some ((sp - 8 + 8) % W))).set 1 (some sv) with
      rip := a3.rip + 1 } := ⟨_, rfl⟩
    have r1 : run im 1 (Out.ok s) = Out.ok a1 := by rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
    have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
    have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
    have r4 : run im 1 (Out.ok a3) = Out.ok a4 := by rw [run_succ_ok im 0 a3, e4, ← ha4, run_zero]
    refine ⟨1 + 1 + 1 + 1, a4, ?_, Or.inl ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [run_add im 1 3, r1, run_add im 1 2, r2, run_add im 1 1, r3, r4]
    · rw [ha4]; show a3.rip + 1 = entry + 45; rw [ha3rip]
    · intro e he
      have he0 : a4.reg 0 = some 0 := by
        rw [ha4]
        show ((a3.set 4 (some ((sp - 8 + 8) % W))).set 1 (some sv)).reg 0 = some 0
        rw [reg_set_other (by decide), reg_set_other (by decide)]; exact ha3r0
      have : e = 0 := Option.some.inj (he.symm.trans he0)
      exact Or.inl ⟨this, rfl⟩
    · rw [ha4]
      show ((a3.set 4 (some ((sp - 8 + 8) % W))).set 1 (some sv)).reg 1 = some sv
      have hl1 : 1 < (a3.set 4 (some ((sp - 8 + 8) % W))).regs.length :=
        len_of_reg (s := a3.set 4 (some ((sp - 8 + 8) % W)))
          (by show (a3.set 4 (some ((sp - 8 + 8) % W))).reg 12 = some t
              rw [reg_set_other (by decide)]; exact ha3r12) (by decide)
      rw [reg_set_self hl1]
    · rw [ha4]
      show ((a3.set 4 (some ((sp - 8 + 8) % W))).set 1 (some sv)).reg 4 = some sp
      rw [reg_set_other (by decide),
        reg_set_self (len_of_reg ha3r12 (by decide)), slot_add8 (by omega) hspW]
    · rw [ha4]
      show ((a3.set 4 (some ((sp - 8 + 8) % W))).set 1 (some sv)).reg 12 = some t
      rw [reg_set_other (by decide), reg_set_other (by decide)]; exact ha3r12
    · intro i h0 h1 h2 h4
      rw [ha4]
      show ((a3.set 4 (some ((sp - 8 + 8) % W))).set 1 (some sv)).reg i = s.reg i
      rw [reg_set_other h1, reg_set_other h4]; exact ga3 i h0 h1 h2 h4
    · intro x n _
      have : a4.mem = s.mem := by
        rw [ha4]
        show ((a3.set 4 (some ((sp - 8 + 8) % W))).set 1 (some sv)).mem = s.mem
        rw [mem_set, mem_set]; exact ha3mem
      rw [this]
    · refine ⟨0, ?_⟩
      rw [ha4]
      show ((a3.set 4 (some ((sp - 8 + 8) % W))).set 1 (some sv)).reg 0 = some 0
      rw [reg_set_other (by decide), reg_set_other (by decide)]; exact ha3r0
  | cons x xs ih =>
    intro t p sp sv s htW hp0 hbound hsp24 hspW hrip hr0 hr1 hr12 hr4 hsave hlay hw16 hw24 hdisj
    have hb : p + 12 * (xs.length + 1) < W := hbound
    have hpW : p < W := by omega
    obtain ⟨⟨hxat, hxW⟩, hxlay⟩ := AnnLayout_cons.mp hlay
    -- test rcx, rcx with rcx = length ≥ 1
    have e1 : X86Int.step im s = .ok { s with
        fl := (aluF .and (x :: xs).length (x :: xs).length).2, rip := s.rip + 3 } :=
      step_test (by rw [hrip]; exact hc.d3) hr1 hr1
    obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s with
      fl := (aluF .and (x :: xs).length (x :: xs).length).2, rip := s.rip + 3 } := ⟨_, rfl⟩
    have ha1rip : a1.rip = entry + 12 := by rw [ha1]; show s.rip + 3 = entry + 12; rw [hrip]
    have ha1fl : X86Int.cond a1.fl 4 = some false := by
      have hz : a1.fl.zf = some false := by
        rw [ha1]
        show (aluF .and (x :: xs).length (x :: xs).length).2.zf = some false
        rw [aluF_zf, aluF_and_fst_self]
        simp
      exact cond_cc4 hz
    have ha1r0 : a1.reg 0 = some p := by rw [ha1]; show s.reg 0 = some p; exact hr0
    have ha1r1 : a1.reg 1 = some (x :: xs).length := by
      rw [ha1]; show s.reg 1 = some (x :: xs).length; exact hr1
    have ha1r4 : a1.reg 4 = some (sp - 8) := by rw [ha1]; show s.reg 4 = some (sp - 8); exact hr4
    have ha1r12 : a1.reg 12 = some t := by rw [ha1]; show s.reg 12 = some t; exact hr12
    have ha1mem : a1.mem = s.mem := by rw [ha1]
    have ga1 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → a1.reg i = s.reg i := by
      intro i _ _ _ _
      rw [ha1]
      show s.reg i = s.reg i
      rfl
    -- jz not taken
    have e2 : X86Int.step im a1 = .ok { a1 with rip := entry + 14 } := by
      have h := step_jcc (im := im) (s := a1) (by rw [ha1rip]; exact hc.d4) ha1fl
      simpa [ha1rip] using h
    obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1 with rip := entry + 14 } := ⟨_, rfl⟩
    have ha2rip : a2.rip = entry + 14 := by rw [ha2]
    have ha2r0 : a2.reg 0 = some p := by
      rw [ha2]; show ({ a1 with rip := entry + 14 }).reg 0 = some p
      rw [reg_with_rip]; exact ha1r0
    have ha2r1 : a2.reg 1 = some (x :: xs).length := by
      rw [ha2]; show ({ a1 with rip := entry + 14 }).reg 1 = some (x :: xs).length
      rw [reg_with_rip]; exact ha1r1
    have ha2r4 : a2.reg 4 = some (sp - 8) := by
      rw [ha2]; show ({ a1 with rip := entry + 14 }).reg 4 = some (sp - 8)
      rw [reg_with_rip]; exact ha1r4
    have ha2r12 : a2.reg 12 = some t := by
      rw [ha2]; show ({ a1 with rip := entry + 14 }).reg 12 = some t
      rw [reg_with_rip]; exact ha1r12
    have ha2mem : a2.mem = s.mem := by
      rw [ha2]; show ({ a1 with rip := entry + 14 }).mem = s.mem
      rw [mem_with_rip]; exact ha1mem
    have ga2 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → a2.reg i = s.reg i := by
      intro i h0 h1 h2 h4
      rw [ha2]; show ({ a1 with rip := entry + 14 }).reg i = s.reg i
      rw [reg_with_rip]; exact ga1 i h0 h1 h2 h4
    -- mov rdx, rax
    have e3 : X86Int.step im a2 = .ok { a2.set 2 (some p) with rip := a2.rip + 3 } :=
      step_mov (by rw [ha2rip]; exact hc.d5) ha2r0
    obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2.set 2 (some p) with rip := a2.rip + 3 } := ⟨_, rfl⟩
    have ha3rip : a3.rip = entry + 17 := by rw [ha3]; show a2.rip + 3 = entry + 17; rw [ha2rip]
    have ha3r2 : a3.reg 2 = some p := by
      rw [ha3]; show ({ a2.set 2 (some p) with rip := a2.rip + 3 }).reg 2 = some p
      rw [reg_with_rip, reg_set_self (len_of_reg ha2r12 (by decide))]
    have ha3r0 : a3.reg 0 = some p := by
      rw [ha3]; show ({ a2.set 2 (some p) with rip := a2.rip + 3 }).reg 0 = some p
      rw [reg_with_rip, reg_set_other (by decide)]; exact ha2r0
    have ha3r1 : a3.reg 1 = some (x :: xs).length := by
      rw [ha3]; show ({ a2.set 2 (some p) with rip := a2.rip + 3 }).reg 1 = some (x :: xs).length
      rw [reg_with_rip, reg_set_other (by decide)]; exact ha2r1
    have ha3r4 : a3.reg 4 = some (sp - 8) := by
      rw [ha3]; show ({ a2.set 2 (some p) with rip := a2.rip + 3 }).reg 4 = some (sp - 8)
      rw [reg_with_rip, reg_set_other (by decide)]; exact ha2r4
    have ha3r12 : a3.reg 12 = some t := by
      rw [ha3]; show ({ a2.set 2 (some p) with rip := a2.rip + 3 }).reg 12 = some t
      rw [reg_with_rip, reg_set_other (by decide)]; exact ha2r12
    have ha3mem : a3.mem = s.mem := by
      rw [ha3]; show ({ a2.set 2 (some p) with rip := a2.rip + 3 }).mem = s.mem
      rw [mem_with_rip, mem_set]; exact ha2mem
    have ga3 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → a3.reg i = s.reg i := by
      intro i h0 h1 h2 h4
      rw [ha3]; show ({ a2.set 2 (some p) with rip := a2.rip + 3 }).reg i = s.reg i
      rw [reg_with_rip, reg_set_other h2]; exact ga2 i h0 h1 h2 h4
    -- push rax
    obtain ⟨m3, hm3s⟩ := Option.isSome_iff_exists.mp (hw16 p)
    have hm3 : wr64 a3.mem ((sp - 8 + W - 8) % W) p = some m3 := by
      rw [ha3mem, show sp - 8 + W - 8 = sp + W - 8 * 2 by omega, slot_sub (by omega) hspW]
      exact hm3s
    have e4 : X86Int.step im a3 = .ok { a3.set 4 (some ((sp - 8 + W - 8) % W)) with
        mem := m3, rip := a3.rip + 1 } :=
      step_push (by rw [ha3rip]; exact hc.d6) ha3r0 ha3r4 hm3
    obtain ⟨a4, ha4⟩ : ∃ u : X86Int.St, u = { a3.set 4 (some ((sp - 8 + W - 8) % W)) with
      mem := m3, rip := a3.rip + 1 } := ⟨_, rfl⟩
    have hslot16 : (sp - 8 + W - 8) % W = sp - 16 := by
      rw [show sp - 8 + W - 8 = sp + W - 8 * 2 by omega, slot_sub (by omega) hspW]
    have ha4rip : a4.rip = entry + 18 := by rw [ha4]; show a3.rip + 1 = entry + 18; rw [ha3rip]
    have ha4r4 : a4.reg 4 = some (sp - 16) := by
      rw [ha4]
      show (a3.set 4 (some ((sp - 8 + W - 8) % W))).reg 4 = some (sp - 16)
      rw [reg_set_self (len_of_reg ha3r12 (by decide)), hslot16]
    have ha4r2 : a4.reg 2 = some p := by
      rw [ha4]
      show (a3.set 4 (some ((sp - 8 + W - 8) % W))).reg 2 = some p
      rw [reg_set_other (by decide)]; exact ha3r2
    have ha4r0 : a4.reg 0 = some p := by
      rw [ha4]
      show (a3.set 4 (some ((sp - 8 + W - 8) % W))).reg 0 = some p
      rw [reg_set_other (by decide)]; exact ha3r0
    have ha4r1 : a4.reg 1 = some (x :: xs).length := by
      rw [ha4]
      show (a3.set 4 (some ((sp - 8 + W - 8) % W))).reg 1 = some (x :: xs).length
      rw [reg_set_other (by decide)]; exact ha3r1
    have ha4r12 : a4.reg 12 = some t := by
      rw [ha4]
      show (a3.set 4 (some ((sp - 8 + W - 8) % W))).reg 12 = some t
      rw [reg_set_other (by decide)]; exact ha3r12
    have ha4mem : a4.mem = m3 := by rw [ha4]
    have ga4 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → a4.reg i = s.reg i := by
      intro i h0 h1 h2 h4
      rw [ha4]
      show (a3.set 4 (some ((sp - 8 + W - 8) % W))).reg i = s.reg i
      rw [reg_set_other h4]; exact ga3 i h0 h1 h2 h4
    have lay4 : AnnLayout a4 (x :: xs) p := by
      have h1 : AnnLayout { a3 with mem := m3 } (x :: xs) p :=
        AnnLayout_wr64 (AnnLayout_of_mem ha3mem hlay)
          (by rw [ha3mem]; exact hm3s)
          (by cases hdisj with
              | inl h => exact Or.inl (by omega)
              | inr h => exact Or.inr (by omega))
      exact AnnLayout_of_mem (by rw [ha4]) h1
    have sv4 : rd64 m3 (sp - 8) = some sv := by
      rw [rd64_wr64_other hm3s (Or.inl (by omega))]
      exact hsave
    have pp4 : rd64 m3 (sp - 16) = some p := by
      have h1 := rd64_wr64 hm3s
      rwa [Nat.mod_eq_of_lt hpW] at h1
    have hw24' : ∀ v, (wr64 m3 (sp - 24) v).isSome :=
      fun v => wr64_some_of_wr64 hm3s (hw24 v)
    have hw16' : ∀ v, (wr64 m3 (sp - 16) v).isSome :=
      fun v => wr64_some_of_wr64 hm3s (hw16 v)
    -- call L32
    obtain ⟨m4, hm4⟩ := Option.isSome_iff_exists.mp (hw24' (entry + 23))
    have hm4' : wr64 a4.mem ((sp - 16 + W - 8) % W) (a4.rip + 5) = some m4 := by
      rw [ha4mem, ha4rip, show sp - 16 + W - 8 = sp + W - 8 * 3 by omega,
        slot_sub (by omega) hspW]
      exact hm4
    have e5 : X86Int.step im a4 = .ok { a4.set 4 (some ((sp - 16 + W - 8) % W)) with
        mem := m4, rip := l32 } :=
      step_call (by rw [ha4rip]; exact hc.d7) ha4r4 hm4'
    obtain ⟨a5, ha5⟩ : ∃ u : X86Int.St, u = { a4.set 4 (some ((sp - 16 + W - 8) % W)) with
      mem := m4, rip := l32 } := ⟨_, rfl⟩
    have hslot24 : (sp - 16 + W - 8) % W = sp - 24 := by
      rw [show sp - 16 + W - 8 = sp + W - 8 * 3 by omega, slot_sub (by omega) hspW]
    have ha5rip : a5.rip = l32 := by rw [ha5]
    have ha5r4 : a5.reg 4 = some (sp - 24) := by
      rw [ha5]
      show (a4.set 4 (some ((sp - 16 + W - 8) % W))).reg 4 = some (sp - 24)
      rw [reg_set_self (len_of_reg ha4r12 (by decide)), hslot24]
    have ha5r2 : a5.reg 2 = some p := by
      rw [ha5]
      show (a4.set 4 (some ((sp - 16 + W - 8) % W))).reg 2 = some p
      rw [reg_set_other (by decide)]; exact ha4r2
    have ha5r1 : a5.reg 1 = some (x :: xs).length := by
      rw [ha5]
      show (a4.set 4 (some ((sp - 16 + W - 8) % W))).reg 1 = some (x :: xs).length
      rw [reg_set_other (by decide)]; exact ha4r1
    have ha5r12 : a5.reg 12 = some t := by
      rw [ha5]
      show (a4.set 4 (some ((sp - 16 + W - 8) % W))).reg 12 = some t
      rw [reg_set_other (by decide)]; exact ha4r12
    have ha5mem : a5.mem = m4 := by rw [ha5]
    have ga5 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → a5.reg i = s.reg i := by
      intro i h0 h1 h2 h4
      rw [ha5]
      show (a4.set 4 (some ((sp - 16 + W - 8) % W))).reg i = s.reg i
      rw [reg_set_other h4]; exact ga4 i h0 h1 h2 h4
    have lay5 : AnnLayout a5 (x :: xs) p := by
      have h1 : AnnLayout { a4 with mem := m4 } (x :: xs) p :=
        AnnLayout_wr64 lay4 (by rw [ha4mem]; exact hm4)
          (by cases hdisj with
              | inl h => exact Or.inl (by omega)
              | inr h => exact Or.inr (by omega))
      exact AnnLayout_of_mem (by rw [ha5]) h1
    have hdann : (m4.rdN p 4).map leVal = some x.1 := by
      have h1 := (AnnLayout_cons.mp lay5).1.1
      obtain ⟨h11, -, -⟩ := h1
      rwa [ha5mem] at h11
    have sv5 : rd64 m4 (sp - 8) = some sv := by
      rw [rd64_wr64_other hm4 (Or.inl (by omega))]
      exact sv4
    have pp5 : rd64 m4 (sp - 16) = some p := by
      rw [rd64_wr64_other hm4 (Or.inl (by omega))]
      exact pp4
    have hw16'' : ∀ v, (wr64 m4 (sp - 16) v).isSome :=
      fun v => wr64_some_of_wr64 hm4 (hw16' v)
    have hw24'' : ∀ v, (wr64 m4 (sp - 24) v).isSome :=
      fun v => wr64_some_of_wr64 hm4 (hw24' v)
    have hframe4 : ∀ y n, y + n ≤ sp - 24 ∨ sp ≤ y → m4.rdN y n = s.mem.rdN y n := by
      intro y n hy
      rw [rdN_wr64_other hm4 (by cases hy with
        | inl h => exact Or.inr (by omega)
        | inr h => exact Or.inl (by omega))]
      exact rdN_wr64_other hm3s (by cases hy with
        | inl h => exact Or.inr (by omega)
        | inr h => exact Or.inl (by omega))
    -- L32 as a black box
    have hrdsome : (a5.mem.rdN p 4).isSome := by
      rw [ha5mem]
      cases h : m4.rdN p 4 with
      | none => simp [h] at hdann
      | some bs => rfl
    obtain ⟨n8, a6, hrun8, ha6rip, ha6r0, -, ha6mem, ha6other, ha6len⟩ :=
      L32_triple hl p hpW a5 ha5rip ⟨ha5r2, hrdsome⟩
    have ha6r0' : a6.reg 0 = some x.1 := by
      rw [ha6r0, ha5mem]
      exact hdann
    have ha6mem' : a6.mem = m4 := by rw [ha6mem]; exact ha5mem
    have ha6r4 : a6.reg 4 = some (sp - 24) := by
      rw [ha6other 4 (by decide)]; exact ha5r4
    have ha6r1 : a6.reg 1 = some (x :: xs).length := by
      rw [ha6other 1 (by decide)]; exact ha5r1
    have ha6r2 : a6.reg 2 = some p := by
      rw [ha6other 2 (by decide)]; exact ha5r2
    have ha6r12 : a6.reg 12 = some t := by
      rw [ha6other 12 (by decide)]; exact ha5r12
    have ga6 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → a6.reg i = s.reg i := by
      intro i h0 h1 h2 h4
      rw [ha6other i h0]; exact ga5 i h0 h1 h2 h4
    -- ret from L32
    have hrd64 : rd64 a6.mem (sp - 24) = some (entry + 23) := by
      have h1 := rd64_wr64 hm4
      rw [Nat.mod_eq_of_lt (by omega)] at h1
      rwa [ha6mem']
    have e7 : X86Int.step im a6 = .ok { a6.set 4 (some ((sp - 24 + 8) % W)) with
        rip := entry + 23 } :=
      step_ret (by rw [ha6rip]; exact hc.dret) ha6r4 hrd64
    obtain ⟨a7, ha7⟩ : ∃ u : X86Int.St, u = { a6.set 4 (some ((sp - 24 + 8) % W)) with
      rip := entry + 23 } := ⟨_, rfl⟩
    have ha7rip : a7.rip = entry + 23 := by rw [ha7]
    have hslotm16 : (sp - 24 + 8) % W = sp - 16 := by
      rw [show sp - 24 + 8 = sp - 16 by omega, Nat.mod_eq_of_lt (by omega)]
    have ha7r4 : a7.reg 4 = some (sp - 16) := by
      rw [ha7]
      show ({ a6.set 4 (some ((sp - 24 + 8) % W)) with rip := entry + 23 }).reg 4 = some (sp - 16)
      rw [reg_with_rip, reg_set_self (len_of_reg ha6r12 (by decide)), hslotm16]
    have ha7r0 : a7.reg 0 = some x.1 := by
      rw [ha7]
      show ({ a6.set 4 (some ((sp - 24 + 8) % W)) with rip := entry + 23 }).reg 0 = some x.1
      rw [reg_with_rip, reg_set_other (by decide)]; exact ha6r0'
    have ha7r1 : a7.reg 1 = some (x :: xs).length := by
      rw [ha7]
      show ({ a6.set 4 (some ((sp - 24 + 8) % W)) with rip := entry + 23 }).reg 1 = some (x :: xs).length
      rw [reg_with_rip, reg_set_other (by decide)]; exact ha6r1
    have ha7r2 : a7.reg 2 = some p := by
      rw [ha7]
      show ({ a6.set 4 (some ((sp - 24 + 8) % W)) with rip := entry + 23 }).reg 2 = some p
      rw [reg_with_rip, reg_set_other (by decide)]; exact ha6r2
    have ha7r12 : a7.reg 12 = some t := by
      rw [ha7]
      show ({ a6.set 4 (some ((sp - 24 + 8) % W)) with rip := entry + 23 }).reg 12 = some t
      rw [reg_with_rip, reg_set_other (by decide)]; exact ha6r12
    have ha7mem : a7.mem = m4 := by
      rw [ha7]
      show ({ a6.set 4 (some ((sp - 24 + 8) % W)) with rip := entry + 23 }).mem = m4
      rw [mem_with_rip, mem_set]; exact ha6mem'
    have ga7 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → a7.reg i = s.reg i := by
      intro i h0 h1 h2 h4
      rw [ha7]
      show ({ a6.set 4 (some ((sp - 24 + 8) % W)) with rip := entry + 23 }).reg i = s.reg i
      rw [reg_with_rip, reg_set_other h4]; exact ga6 i h0 h1 h2 h4
    -- pop rdx
    have hpp7 : rd64 a7.mem (sp - 16) = some p := by rw [ha7mem]; exact pp5
    have e8 : X86Int.step im a7 = .ok { (a7.set 4 (some ((sp - 16 + 8) % W))).set 2 (some p) with
        rip := a7.rip + 1 } :=
      step_pop (by rw [ha7rip]; exact hc.d8) ha7r4 hpp7
    obtain ⟨a8, ha8⟩ : ∃ u : X86Int.St, u = { (a7.set 4 (some ((sp - 16 + 8) % W))).set 2 (some p) with
      rip := a7.rip + 1 } := ⟨_, rfl⟩
    have ha8rip : a8.rip = entry + 24 := by rw [ha8]; show a7.rip + 1 = entry + 24; rw [ha7rip]
    have hslotm8 : (sp - 16 + 8) % W = sp - 8 := by
      rw [show sp - 16 + 8 = sp - 8 by omega, Nat.mod_eq_of_lt (by omega)]
    have ha8r4 : a8.reg 4 = some (sp - 8) := by
      rw [ha8]
      show ((a7.set 4 (some ((sp - 16 + 8) % W))).set 2 (some p)).reg 4 = some (sp - 8)
      rw [reg_set_other (by decide), reg_set_self (len_of_reg ha7r12 (by decide)), hslotm8]
    have ha8r2 : a8.reg 2 = some p := by
      rw [ha8]
      show ((a7.set 4 (some ((sp - 16 + 8) % W))).set 2 (some p)).reg 2 = some p
      have hl2 : 2 < (a7.set 4 (some ((sp - 16 + 8) % W))).regs.length :=
        len_of_reg (s := a7.set 4 (some ((sp - 16 + 8) % W)))
          (by show (a7.set 4 (some ((sp - 16 + 8) % W))).reg 12 = some t
              rw [reg_set_other (by decide)]; exact ha7r12) (by decide)
      rw [reg_set_self hl2]
    have ha8r0 : a8.reg 0 = some x.1 := by
      rw [ha8]
      show ((a7.set 4 (some ((sp - 16 + 8) % W))).set 2 (some p)).reg 0 = some x.1
      rw [reg_set_other (by decide), reg_set_other (by decide)]; exact ha7r0
    have ha8r1 : a8.reg 1 = some (x :: xs).length := by
      rw [ha8]
      show ((a7.set 4 (some ((sp - 16 + 8) % W))).set 2 (some p)).reg 1 = some (x :: xs).length
      rw [reg_set_other (by decide), reg_set_other (by decide)]; exact ha7r1
    have ha8r12 : a8.reg 12 = some t := by
      rw [ha8]
      show ((a7.set 4 (some ((sp - 16 + 8) % W))).set 2 (some p)).reg 12 = some t
      rw [reg_set_other (by decide), reg_set_other (by decide)]; exact ha7r12
    have ha8mem : a8.mem = m4 := by
      rw [ha8]
      show ((a7.set 4 (some ((sp - 16 + 8) % W))).set 2 (some p)).mem = m4
      rw [mem_set, mem_set]; exact ha7mem
    have ga8 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → a8.reg i = s.reg i := by
      intro i h0 h1 h2 h4
      rw [ha8]
      show ((a7.set 4 (some ((sp - 16 + 8) % W))).set 2 (some p)).reg i = s.reg i
      rw [reg_set_other h2, reg_set_other h4]; exact ga7 i h0 h1 h2 h4
    -- cmp rax, r12
    have e9 : X86Int.step im a8 = .ok { a8 with fl := (aluF .sub x.1 t).2, rip := a8.rip + 3 } :=
      step_cmp (by rw [ha8rip]; exact hc.d9) ha8r0 ha8r12
    obtain ⟨a9, ha9⟩ : ∃ u : X86Int.St, u = { a8 with fl := (aluF .sub x.1 t).2, rip := a8.rip + 3 } := ⟨_, rfl⟩
    have ha9rip : a9.rip = entry + 27 := by rw [ha9]; show a8.rip + 3 = entry + 27; rw [ha8rip]
    have ha9fl : X86Int.cond a9.fl 4 = some (x.1 == t) := by
      have hz : a9.fl.zf = some (x.1 == t) := by
        rw [ha9]
        show (aluF .sub x.1 t).2.zf = some (x.1 == t)
        rw [aluF_zf, aluF_sub_fst, subW_eq_zero hxW htW]
      exact cond_cc4 hz
    have ha9r0 : a9.reg 0 = some x.1 := by rw [ha9]; show a8.reg 0 = some x.1; exact ha8r0
    have ha9r2 : a9.reg 2 = some p := by rw [ha9]; show a8.reg 2 = some p; exact ha8r2
    have ha9r1 : a9.reg 1 = some (x :: xs).length := by
      rw [ha9]; show a8.reg 1 = some (x :: xs).length; exact ha8r1
    have ha9r4 : a9.reg 4 = some (sp - 8) := by rw [ha9]; show a8.reg 4 = some (sp - 8); exact ha8r4
    have ha9r12 : a9.reg 12 = some t := by rw [ha9]; show a8.reg 12 = some t; exact ha8r12
    have ha9mem : a9.mem = m4 := by rw [ha9]; show a8.mem = m4; exact ha8mem
    have ga9 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → a9.reg i = s.reg i := by
      intro i h0 h1 h2 h4
      rw [ha9]
      show a8.reg i = s.reg i
      exact ga8 i h0 h1 h2 h4
    -- run prefixes
    have r1 : run im 1 (Out.ok s) = Out.ok a1 := by rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
    have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
    have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
    have r4 : run im 1 (Out.ok a3) = Out.ok a4 := by rw [run_succ_ok im 0 a3, e4, ← ha4, run_zero]
    have r5 : run im 1 (Out.ok a4) = Out.ok a5 := by rw [run_succ_ok im 0 a4, e5, ← ha5, run_zero]
    have r7 : run im 1 (Out.ok a6) = Out.ok a7 := by rw [run_succ_ok im 0 a6, e7, ← ha7, run_zero]
    have r8 : run im 1 (Out.ok a7) = Out.ok a8 := by rw [run_succ_ok im 0 a7, e8, ← ha8, run_zero]
    have r9 : run im 1 (Out.ok a8) = Out.ok a9 := by rw [run_succ_ok im 0 a8, e9, ← ha9, run_zero]
    have hpre : run im (5 + n8 + 3) (Out.ok s) = Out.ok a9 := by
      rw [run_add im (5 + n8) 3,
        show run im (5 + n8) (Out.ok s) = Out.ok a6 from by
          rw [run_add im 5 n8,
            show run im 5 (Out.ok s) = Out.ok a5 from by
              rw [run_add im 1 4, r1, run_add im 1 3, r2, run_add im 1 2, r3,
                run_add im 1 1, r4, r5],
            hrun8],
        run_add im 1 2, r7, run_add im 1 1, r8, r9]
    -- jz: split on the comparison
    by_cases hxt : x.1 = t
    · -- found
      have hb2 : (x.1 == t) = true := by simp [hxt]
      have e10 : X86Int.step im a9 = .ok { a9 with rip := entry + 46 } := by
        have h := step_jcc (im := im) (s := a9) (by rw [ha9rip]; exact hc.d10) ha9fl
        rwa [hb2] at h
      obtain ⟨a10, ha10⟩ : ∃ u : X86Int.St, u = { a9 with rip := entry + 46 } := ⟨_, rfl⟩
      have ha10rip : a10.rip = entry + 46 := by rw [ha10]
      have ha10r2 : a10.reg 2 = some p := by
        rw [ha10]; show ({ a9 with rip := entry + 46 }).reg 2 = some p
        rw [reg_with_rip]; exact ha9r2
      have ha10r4 : a10.reg 4 = some (sp - 8) := by
        rw [ha10]; show ({ a9 with rip := entry + 46 }).reg 4 = some (sp - 8)
        rw [reg_with_rip]; exact ha9r4
      have ha10r12 : a10.reg 12 = some t := by
        rw [ha10]; show ({ a9 with rip := entry + 46 }).reg 12 = some t
        rw [reg_with_rip]; exact ha9r12
      have ha10mem : a10.mem = m4 := by
        rw [ha10]; show ({ a9 with rip := entry + 46 }).mem = m4
        rw [mem_with_rip]; exact ha9mem
      have ga10 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → a10.reg i = s.reg i := by
        intro i h0 h1 h2 h4
        rw [ha10]; show ({ a9 with rip := entry + 46 }).reg i = s.reg i
        rw [reg_with_rip]; exact ga9 i h0 h1 h2 h4
      -- mov rax, rdx
      have e11 : X86Int.step im a10 = .ok { a10.set 0 (some p) with rip := a10.rip + 3 } :=
        step_mov (by rw [ha10rip]; exact hc.d17) ha10r2
      obtain ⟨a11, ha11⟩ : ∃ u : X86Int.St, u = { a10.set 0 (some p) with rip := a10.rip + 3 } := ⟨_, rfl⟩
      have ha11rip : a11.rip = entry + 49 := by rw [ha11]; show a10.rip + 3 = entry + 49; rw [ha10rip]
      have ha11r0 : a11.reg 0 = some p := by
        rw [ha11]; show ({ a10.set 0 (some p) with rip := a10.rip + 3 }).reg 0 = some p
        rw [reg_with_rip, reg_set_self (len_of_reg ha10r12 (by decide))]
      have ha11r4 : a11.reg 4 = some (sp - 8) := by
        rw [ha11]; show ({ a10.set 0 (some p) with rip := a10.rip + 3 }).reg 4 = some (sp - 8)
        rw [reg_with_rip, reg_set_other (by decide)]; exact ha10r4
      have ha11r12 : a11.reg 12 = some t := by
        rw [ha11]; show ({ a10.set 0 (some p) with rip := a10.rip + 3 }).reg 12 = some t
        rw [reg_with_rip, reg_set_other (by decide)]; exact ha10r12
      have ha11mem : a11.mem = m4 := by
        rw [ha11]; show ({ a10.set 0 (some p) with rip := a10.rip + 3 }).mem = m4
        rw [mem_with_rip, mem_set]; exact ha10mem
      have ga11 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → a11.reg i = s.reg i := by
        intro i h0 h1 h2 h4
        rw [ha11]; show ({ a10.set 0 (some p) with rip := a10.rip + 3 }).reg i = s.reg i
        rw [reg_with_rip, reg_set_other h0]; exact ga10 i h0 h1 h2 h4
      -- pop rcx
      have hsv11 : rd64 a11.mem (sp - 8) = some sv := by rw [ha11mem]; exact sv5
      have e12 : X86Int.step im a11 = .ok { (a11.set 4 (some ((sp - 8 + 8) % W))).set 1 (some sv) with
          rip := a11.rip + 1 } :=
        step_pop (by rw [ha11rip]; exact hc.d18) ha11r4 hsv11
      obtain ⟨a12, ha12⟩ : ∃ u : X86Int.St, u = { (a11.set 4 (some ((sp - 8 + 8) % W))).set 1 (some sv) with
        rip := a11.rip + 1 } := ⟨_, rfl⟩
      have r10 : run im 1 (Out.ok a9) = Out.ok a10 := by rw [run_succ_ok im 0 a9, e10, ← ha10, run_zero]
      have r11 : run im 1 (Out.ok a10) = Out.ok a11 := by rw [run_succ_ok im 0 a10, e11, ← ha11, run_zero]
      have r12 : run im 1 (Out.ok a11) = Out.ok a12 := by rw [run_succ_ok im 0 a11, e12, ← ha12, run_zero]
      have ha12mem : a12.mem = m4 := by
        rw [ha12]
        show ((a11.set 4 (some ((sp - 8 + 8) % W))).set 1 (some sv)).mem = m4
        rw [mem_set, mem_set]; exact ha11mem
      refine ⟨5 + n8 + 3 + 3, a12, ?_, Or.inr ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · rw [run_add im (5 + n8 + 3) 3, hpre, run_add im 1 2, r10, run_add im 1 1, r11, r12]
      · rw [ha12]; show a11.rip + 1 = entry + 50; rw [ha11rip]
      · intro e he
        have he0 : a12.reg 0 = some p := by
          rw [ha12]
          show ((a11.set 4 (some ((sp - 8 + 8) % W))).set 1 (some sv)).reg 0 = some p
          rw [reg_set_other (by decide), reg_set_other (by decide)]; exact ha11r0
        have hep : e = p := Option.some.inj (he.symm.trans he0)
        refine Or.inr ⟨x.2.d, x.2.a, by rw [hep]; omega, ?_, lookup_cons_eq hxt⟩
        have h1 : AnnAt a5 p x.1 x.2.d x.2.a := (AnnLayout_cons.mp lay5).1.1
        have h2 : AnnAt a12 p x.1 x.2.d x.2.a := AnnAt_of_mem (by rw [ha12mem, ha5mem]) h1
        rw [hxt] at h2
        rwa [hep]
      · rw [ha12]
        show ((a11.set 4 (some ((sp - 8 + 8) % W))).set 1 (some sv)).reg 1 = some sv
        have hl1 : 1 < (a11.set 4 (some ((sp - 8 + 8) % W))).regs.length :=
          len_of_reg (s := a11.set 4 (some ((sp - 8 + 8) % W)))
            (by show (a11.set 4 (some ((sp - 8 + 8) % W))).reg 12 = some t
                rw [reg_set_other (by decide)]; exact ha11r12) (by decide)
        rw [reg_set_self hl1]
      · rw [ha12]
        show ((a11.set 4 (some ((sp - 8 + 8) % W))).set 1 (some sv)).reg 4 = some sp
        rw [reg_set_other (by decide),
          reg_set_self (len_of_reg ha11r12 (by decide)), slot_add8 (by omega) hspW]
      · rw [ha12]
        show ((a11.set 4 (some ((sp - 8 + 8) % W))).set 1 (some sv)).reg 12 = some t
        rw [reg_set_other (by decide), reg_set_other (by decide)]; exact ha11r12
      · intro i h0 h1 h2 h4
        rw [ha12]
        show ((a11.set 4 (some ((sp - 8 + 8) % W))).set 1 (some sv)).reg i = s.reg i
        rw [reg_set_other h1, reg_set_other h4]; exact ga11 i h0 h1 h2 h4
      · intro y n hy
        rw [ha12mem]
        exact hframe4 y n hy
      · refine ⟨p, ?_⟩
        rw [ha12]
        show ((a11.set 4 (some ((sp - 8 + 8) % W))).set 1 (some sv)).reg 0 = some p
        rw [reg_set_other (by decide), reg_set_other (by decide)]; exact ha11r0
    · -- not found: advance and recurse
      have hb2 : (x.1 == t) = false := by simp [hxt]
      have e10 : X86Int.step im a9 = .ok { a9 with rip := entry + 29 } := by
        have h := step_jcc (im := im) (s := a9) (by rw [ha9rip]; exact hc.d10) ha9fl
        rw [hb2] at h
        simpa [ha9rip] using h
      obtain ⟨a10, ha10⟩ : ∃ u : X86Int.St, u = { a9 with rip := entry + 29 } := ⟨_, rfl⟩
      have ha10rip : a10.rip = entry + 29 := by rw [ha10]
      have ha10r2 : a10.reg 2 = some p := by
        rw [ha10]; show ({ a9 with rip := entry + 29 }).reg 2 = some p
        rw [reg_with_rip]; exact ha9r2
      have ha10r1 : a10.reg 1 = some (x :: xs).length := by
        rw [ha10]; show ({ a9 with rip := entry + 29 }).reg 1 = some (x :: xs).length
        rw [reg_with_rip]; exact ha9r1
      have ha10r4 : a10.reg 4 = some (sp - 8) := by
        rw [ha10]; show ({ a9 with rip := entry + 29 }).reg 4 = some (sp - 8)
        rw [reg_with_rip]; exact ha9r4
      have ha10r12 : a10.reg 12 = some t := by
        rw [ha10]; show ({ a9 with rip := entry + 29 }).reg 12 = some t
        rw [reg_with_rip]; exact ha9r12
      have ha10mem : a10.mem = m4 := by
        rw [ha10]; show ({ a9 with rip := entry + 29 }).mem = m4
        rw [mem_with_rip]; exact ha9mem
      have ga10 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → a10.reg i = s.reg i := by
        intro i h0 h1 h2 h4
        rw [ha10]; show ({ a9 with rip := entry + 29 }).reg i = s.reg i
        rw [reg_with_rip]; exact ga9 i h0 h1 h2 h4
      -- mov rax, rdx
      have e11 : X86Int.step im a10 = .ok { a10.set 0 (some p) with rip := a10.rip + 3 } :=
        step_mov (by rw [ha10rip]; exact hc.d11) ha10r2
      obtain ⟨a11, ha11⟩ : ∃ u : X86Int.St, u = { a10.set 0 (some p) with rip := a10.rip + 3 } := ⟨_, rfl⟩
      have ha11rip : a11.rip = entry + 32 := by rw [ha11]; show a10.rip + 3 = entry + 32; rw [ha10rip]
      have ha11r0 : a11.reg 0 = some p := by
        rw [ha11]; show ({ a10.set 0 (some p) with rip := a10.rip + 3 }).reg 0 = some p
        rw [reg_with_rip, reg_set_self (len_of_reg ha10r12 (by decide))]
      have ha11r1 : a11.reg 1 = some (x :: xs).length := by
        rw [ha11]; show ({ a10.set 0 (some p) with rip := a10.rip + 3 }).reg 1 = some (x :: xs).length
        rw [reg_with_rip, reg_set_other (by decide)]; exact ha10r1
      have ha11r4 : a11.reg 4 = some (sp - 8) := by
        rw [ha11]; show ({ a10.set 0 (some p) with rip := a10.rip + 3 }).reg 4 = some (sp - 8)
        rw [reg_with_rip, reg_set_other (by decide)]; exact ha10r4
      have ha11r12 : a11.reg 12 = some t := by
        rw [ha11]; show ({ a10.set 0 (some p) with rip := a10.rip + 3 }).reg 12 = some t
        rw [reg_with_rip, reg_set_other (by decide)]; exact ha10r12
      have ha11mem : a11.mem = m4 := by
        rw [ha11]; show ({ a10.set 0 (some p) with rip := a10.rip + 3 }).mem = m4
        rw [mem_with_rip, mem_set]; exact ha10mem
      have ga11 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → a11.reg i = s.reg i := by
        intro i h0 h1 h2 h4
        rw [ha11]; show ({ a10.set 0 (some p) with rip := a10.rip + 3 }).reg i = s.reg i
        rw [reg_with_rip, reg_set_other h0]; exact ga10 i h0 h1 h2 h4
      -- add rax, 12
      have h12W : 12 % W = 12 := Nat.mod_eq_of_lt (by unfold W; decide)
      have hadd : (aluF .add p (12 % W)).1 = p + 12 := by
        rw [aluF_add_fst, h12W, Nat.mod_eq_of_lt (by omega)]
      have e12 : X86Int.step im a11 = .ok { a11.set 0 (some (aluF .add p (12 % W)).1) with
          fl := (aluF .add p (12 % W)).2, rip := a11.rip + 4 } :=
        step_aluI_add (by rw [ha11rip]; exact hc.d12) ha11r0
      obtain ⟨a12, ha12⟩ : ∃ u : X86Int.St, u = { a11.set 0 (some (aluF .add p (12 % W)).1) with
        fl := (aluF .add p (12 % W)).2, rip := a11.rip + 4 } := ⟨_, rfl⟩
      have ha12rip : a12.rip = entry + 36 := by rw [ha12]; show a11.rip + 4 = entry + 36; rw [ha11rip]
      have ha12r0 : a12.reg 0 = some (p + 12) := by
        rw [ha12]
        show (a11.set 0 (some (aluF .add p (12 % W)).1)).reg 0 = some (p + 12)
        rw [reg_set_self (len_of_reg ha11r12 (by decide)), hadd]
      have ha12r1 : a12.reg 1 = some (x :: xs).length := by
        rw [ha12]
        show (a11.set 0 (some (aluF .add p (12 % W)).1)).reg 1 = some (x :: xs).length
        rw [reg_set_other (by decide)]; exact ha11r1
      have ha12r4 : a12.reg 4 = some (sp - 8) := by
        rw [ha12]
        show (a11.set 0 (some (aluF .add p (12 % W)).1)).reg 4 = some (sp - 8)
        rw [reg_set_other (by decide)]; exact ha11r4
      have ha12r12 : a12.reg 12 = some t := by
        rw [ha12]
        show (a11.set 0 (some (aluF .add p (12 % W)).1)).reg 12 = some t
        rw [reg_set_other (by decide)]; exact ha11r12
      have ha12mem : a12.mem = m4 := by
        rw [ha12]
        show (a11.set 0 (some (aluF .add p (12 % W)).1)).mem = m4
        rw [mem_set]; exact ha11mem
      have ga12 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → a12.reg i = s.reg i := by
        intro i h0 h1 h2 h4
        rw [ha12]
        show (a11.set 0 (some (aluF .add p (12 % W)).1)).reg i = s.reg i
        rw [reg_set_other h0]; exact ga11 i h0 h1 h2 h4
      -- sub rcx, 1
      have h1W : 1 % W = 1 := Nat.mod_eq_of_lt (by unfold W; decide)
      have hsubv : (aluF .sub (x :: xs).length (1 % W)).1 = xs.length := by
        rw [aluF_sub_fst, h1W]
        have hlx : (x :: xs).length = xs.length + 1 := rfl
        rw [hlx, show xs.length + 1 + W - 1 = xs.length + W by omega, Nat.add_mod_right,
          Nat.mod_eq_of_lt (by omega)]
      have e13 : X86Int.step im a12 = .ok { a12.set 1 (some (aluF .sub (x :: xs).length (1 % W)).1) with
          fl := (aluF .sub (x :: xs).length (1 % W)).2, rip := a12.rip + 4 } :=
        step_aluI_sub (by rw [ha12rip]; exact hc.d13) ha12r1
      obtain ⟨a13, ha13⟩ : ∃ u : X86Int.St, u = { a12.set 1 (some (aluF .sub (x :: xs).length (1 % W)).1) with
        fl := (aluF .sub (x :: xs).length (1 % W)).2, rip := a12.rip + 4 } := ⟨_, rfl⟩
      have ha13rip : a13.rip = entry + 40 := by rw [ha13]; show a12.rip + 4 = entry + 40; rw [ha12rip]
      have ha13r1 : a13.reg 1 = some xs.length := by
        rw [ha13]
        show (a12.set 1 (some (aluF .sub (x :: xs).length (1 % W)).1)).reg 1 = some xs.length
        rw [reg_set_self (len_of_reg ha12r12 (by decide)), hsubv]
      have ha13r0 : a13.reg 0 = some (p + 12) := by
        rw [ha13]
        show (a12.set 1 (some (aluF .sub (x :: xs).length (1 % W)).1)).reg 0 = some (p + 12)
        rw [reg_set_other (by decide)]; exact ha12r0
      have ha13r4 : a13.reg 4 = some (sp - 8) := by
        rw [ha13]
        show (a12.set 1 (some (aluF .sub (x :: xs).length (1 % W)).1)).reg 4 = some (sp - 8)
        rw [reg_set_other (by decide)]; exact ha12r4
      have ha13r12 : a13.reg 12 = some t := by
        rw [ha13]
        show (a12.set 1 (some (aluF .sub (x :: xs).length (1 % W)).1)).reg 12 = some t
        rw [reg_set_other (by decide)]; exact ha12r12
      have ha13mem : a13.mem = m4 := by
        rw [ha13]
        show (a12.set 1 (some (aluF .sub (x :: xs).length (1 % W)).1)).mem = m4
        rw [mem_set]; exact ha12mem
      have ga13 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → a13.reg i = s.reg i := by
        intro i h0 h1 h2 h4
        rw [ha13]
        show (a12.set 1 (some (aluF .sub (x :: xs).length (1 % W)).1)).reg i = s.reg i
        rw [reg_set_other h1]; exact ga12 i h0 h1 h2 h4
      -- jmp loop
      have e14 : X86Int.step im a13 = .ok { a13 with rip := entry + 9 } :=
        step_jmp (by rw [ha13rip]; exact hc.d14)
      obtain ⟨a14, ha14⟩ : ∃ u : X86Int.St, u = { a13 with rip := entry + 9 } := ⟨_, rfl⟩
      have ha14rip : a14.rip = entry + 9 := by rw [ha14]
      have ha14r0 : a14.reg 0 = some (p + 12) := by
        rw [ha14]; show ({ a13 with rip := entry + 9 }).reg 0 = some (p + 12)
        rw [reg_with_rip]; exact ha13r0
      have ha14r1 : a14.reg 1 = some xs.length := by
        rw [ha14]; show ({ a13 with rip := entry + 9 }).reg 1 = some xs.length
        rw [reg_with_rip]; exact ha13r1
      have ha14r4 : a14.reg 4 = some (sp - 8) := by
        rw [ha14]; show ({ a13 with rip := entry + 9 }).reg 4 = some (sp - 8)
        rw [reg_with_rip]; exact ha13r4
      have ha14r12 : a14.reg 12 = some t := by
        rw [ha14]; show ({ a13 with rip := entry + 9 }).reg 12 = some t
        rw [reg_with_rip]; exact ha13r12
      have ha14mem : a14.mem = m4 := by
        rw [ha14]; show ({ a13 with rip := entry + 9 }).mem = m4
        rw [mem_with_rip]; exact ha13mem
      have ga14 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → a14.reg i = s.reg i := by
        intro i h0 h1 h2 h4
        rw [ha14]; show ({ a13 with rip := entry + 9 }).reg i = s.reg i
        rw [reg_with_rip]; exact ga13 i h0 h1 h2 h4
      have lay14 : AnnLayout a14 xs (p + 12) := by
        have h1 : AnnLayout a14 (x :: xs) p := AnnLayout_of_mem (by rw [ha14mem, ha5mem]) lay5
        exact (AnnLayout_cons.mp h1).2
      have hsv14 : rd64 a14.mem (sp - 8) = some sv := by rw [ha14mem]; exact sv5
      have hw16a : ∀ v, (wr64 a14.mem (sp - 16) v).isSome := by
        intro v; rw [ha14mem]; exact hw16'' v
      have hw24a : ∀ v, (wr64 a14.mem (sp - 24) v).isSome := by
        intro v; rw [ha14mem]; exact hw24'' v
      obtain ⟨n', s', hrun', hrip', hpost', hr1', hr4', hr12', hgen', hframe', hr0e'⟩ :=
        ih t (p + 12) sp sv a14 htW (by omega) (by omega) hsp24 hspW ha14rip ha14r0 ha14r1
          ha14r12 ha14r4 hsv14 lay14 hw16a hw24a
          (by cases hdisj with
              | inl h => exact Or.inl (by omega)
              | inr h => exact Or.inr (by
                  have h' : p + 12 * (xs.length + 1) ≤ sp - 24 := h
                  omega))
      have r10 : run im 1 (Out.ok a9) = Out.ok a10 := by rw [run_succ_ok im 0 a9, e10, ← ha10, run_zero]
      have r11 : run im 1 (Out.ok a10) = Out.ok a11 := by rw [run_succ_ok im 0 a10, e11, ← ha11, run_zero]
      have r12 : run im 1 (Out.ok a11) = Out.ok a12 := by rw [run_succ_ok im 0 a11, e12, ← ha12, run_zero]
      have r13 : run im 1 (Out.ok a12) = Out.ok a13 := by rw [run_succ_ok im 0 a12, e13, ← ha13, run_zero]
      have r14 : run im 1 (Out.ok a13) = Out.ok a14 := by rw [run_succ_ok im 0 a13, e14, ← ha14, run_zero]
      refine ⟨5 + n8 + 3 + 5 + n', s', ?_, hrip', ?_, hr1', hr4', hr12', ?_, ?_, hr0e'⟩
      · rw [run_add im (5 + n8 + 3 + 5) n',
          show run im (5 + n8 + 3 + 5) (Out.ok s) = Out.ok a14 from by
            rw [run_add im (5 + n8 + 3) 5, hpre, run_add im 1 4, r10, run_add im 1 3, r11,
              run_add im 1 2, r12, run_add im 1 1, r13, r14],
          hrun']
      · intro e he
        cases hpost' e he with
        | inl h => exact Or.inl ⟨h.1, by rw [lookup_cons_ne hxt]; exact h.2⟩
        | inr h =>
          obtain ⟨d, a, hne, hat, hlk⟩ := h
          exact Or.inr ⟨d, a, hne, hat, by rw [lookup_cons_ne hxt]; exact hlk⟩
      · intro i h0 h1 h2 h4
        rw [hgen' i h0 h1 h2 h4]
        exact ga14 i h0 h1 h2 h4
      · intro y n hy
        rw [hframe' y n hy, ha14mem]
        exact hframe4 y n hy



/-- The `ANNL` triple, proved: push the saved count, load the ann
pointer and count from the word entry, then the loop. -/
theorem ANNL_triple {im : Nat → Option (Ins × Nat)} {entry l32 : Nat}
    (hc : ANNLCode im entry l32) (hl : L32Code im l32) (hentry : entry + 50 < W) :
    ANNLSpec im entry l32 := by
  intro wd t w p sp cv htW hp0 hw56 hbound hsp24 hspW s hrip hr12 hr7 hr4 hr1 hfp hfn
    hlay hroom hdisjw hdisjp
  have hw8 : ∀ v, (wr64 s.mem (sp - 8) v).isSome := by
    intro v
    have h := hroom sp hr4 0 (by decide) v
    rwa [slot_sub (k := 0 + 1) (by omega) hspW] at h
  have hw16 : ∀ v, (wr64 s.mem (sp - 16) v).isSome := by
    intro v
    have h := hroom sp hr4 1 (by decide) v
    rwa [slot_sub (k := 1 + 1) (by omega) hspW] at h
  have hw24 : ∀ v, (wr64 s.mem (sp - 24) v).isSome := by
    intro v
    have h := hroom sp hr4 2 (by decide) v
    rwa [slot_sub (k := 2 + 1) (by omega) hspW] at h
  have hslot8 : (sp + W - 8) % W = sp - 8 := by
    rw [show sp + W - 8 = sp + W - 8 * 1 by omega, slot_sub (by omega) hspW]
  -- push rcx
  obtain ⟨m1, hm1s⟩ := Option.isSome_iff_exists.mp (hw8 cv)
  have hm1 : wr64 s.mem ((sp + W - 8) % W) cv = some m1 := by
    rw [hslot8]; exact hm1s
  have e1 : X86Int.step im s = .ok { s.set 4 (some ((sp + W - 8) % W)) with
      mem := m1, rip := s.rip + 1 } :=
    step_push (by rw [hrip]; exact hc.d0) hr1 hr4 hm1
  obtain ⟨b1, hb1⟩ : ∃ u : X86Int.St, u = { s.set 4 (some ((sp + W - 8) % W)) with
    mem := m1, rip := s.rip + 1 } := ⟨_, rfl⟩
  have hb1rip : b1.rip = entry + 1 := by rw [hb1]; show s.rip + 1 = entry + 1; rw [hrip]
  have hb1r4 : b1.reg 4 = some (sp - 8) := by
    rw [hb1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
    rw [reg_set_self (len_of_reg hr12 (by decide)), hslot8]
  have hb1r7 : b1.reg 7 = some w := by
    rw [hb1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 7 = some w
    rw [reg_set_other (by decide)]; exact hr7
  have hb1r12 : b1.reg 12 = some t := by
    rw [hb1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 12 = some t
    rw [reg_set_other (by decide)]; exact hr12
  have hb1mem : b1.mem = m1 := by rw [hb1]
  have gb1 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → b1.reg i = s.reg i := by
    intro i _ _ _ h4
    rw [hb1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg i = s.reg i
    rw [reg_set_other h4]
  have hfp1 : rd64 m1 (w + 48) = some p := by
    rw [rd64_wr64_other hm1s (by cases hdisjw with
      | inl h => exact Or.inl (by omega)
      | inr h => exact Or.inr (by omega))]
    exact hfp
  have hfn1 : rd64 m1 (w + 32) = some wd.ann.length := by
    rw [rd64_wr64_other hm1s (by cases hdisjw with
      | inl h => exact Or.inl (by omega)
      | inr h => exact Or.inr (by omega))]
    exact hfn
  have lay1 : AnnLayout b1 wd.ann p := by
    have h1 : AnnLayout { s with mem := m1 } wd.ann p :=
      AnnLayout_wr64 hlay hm1s
        (by cases hdisjp with
            | inl h => exact Or.inl (by omega)
            | inr h => exact Or.inr (by omega))
    exact AnnLayout_of_mem (by rw [hb1]) h1
  have hsave1 : rd64 m1 (sp - 8) = some (cv % W) := rd64_wr64 hm1s
  have hw16' : ∀ v, (wr64 m1 (sp - 16) v).isSome :=
    fun v => wr64_some_of_wr64 hm1s (hw16 v)
  have hw24' : ∀ v, (wr64 m1 (sp - 24) v).isSome :=
    fun v => wr64_some_of_wr64 hm1s (hw24 v)
  -- mov rax, [rdi+48]
  have hea48 : ea w 48 = w + 48 := Nat.mod_eq_of_lt (by omega)
  have e2 : X86Int.step im b1 = .ok { b1.set 0 (some p) with rip := b1.rip + 4 } :=
    step_ld (by rw [hb1rip]; exact hc.d1) hb1r7 (by rw [hb1mem, hea48]; exact hfp1)
  obtain ⟨b2, hb2⟩ : ∃ u : X86Int.St, u = { b1.set 0 (some p) with rip := b1.rip + 4 } := ⟨_, rfl⟩
  have hb2rip : b2.rip = entry + 5 := by rw [hb2]; show b1.rip + 4 = entry + 5; rw [hb1rip]
  have hb2r0 : b2.reg 0 = some p := by
    rw [hb2]; show ({ b1.set 0 (some p) with rip := b1.rip + 4 }).reg 0 = some p
    rw [reg_with_rip, reg_set_self (len_of_reg hb1r12 (by decide))]
  have hb2r7 : b2.reg 7 = some w := by
    rw [hb2]; show ({ b1.set 0 (some p) with rip := b1.rip + 4 }).reg 7 = some w
    rw [reg_with_rip, reg_set_other (by decide)]; exact hb1r7
  have hb2r4 : b2.reg 4 = some (sp - 8) := by
    rw [hb2]; show ({ b1.set 0 (some p) with rip := b1.rip + 4 }).reg 4 = some (sp - 8)
    rw [reg_with_rip, reg_set_other (by decide)]; exact hb1r4
  have hb2r12 : b2.reg 12 = some t := by
    rw [hb2]; show ({ b1.set 0 (some p) with rip := b1.rip + 4 }).reg 12 = some t
    rw [reg_with_rip, reg_set_other (by decide)]; exact hb1r12
  have hb2mem : b2.mem = m1 := by
    rw [hb2]; show ({ b1.set 0 (some p) with rip := b1.rip + 4 }).mem = m1
    rw [mem_with_rip, mem_set]; exact hb1mem
  have gb2 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → b2.reg i = s.reg i := by
    intro i h0 h1 h2 h4
    rw [hb2]; show ({ b1.set 0 (some p) with rip := b1.rip + 4 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other h0]; exact gb1 i h0 h1 h2 h4
  -- mov rcx, [rdi+32]
  have hea32 : ea w 32 = w + 32 := Nat.mod_eq_of_lt (by omega)
  have e3 : X86Int.step im b2 = .ok { b2.set 1 (some wd.ann.length) with rip := b2.rip + 4 } :=
    step_ld (by rw [hb2rip]; exact hc.d2) hb2r7 (by rw [hb2mem, hea32]; exact hfn1)
  obtain ⟨b3, hb3⟩ : ∃ u : X86Int.St, u = { b2.set 1 (some wd.ann.length) with
    rip := b2.rip + 4 } := ⟨_, rfl⟩
  have hb3rip : b3.rip = entry + 9 := by rw [hb3]; show b2.rip + 4 = entry + 9; rw [hb2rip]
  have hb3r1 : b3.reg 1 = some wd.ann.length := by
    rw [hb3]; show ({ b2.set 1 (some wd.ann.length) with rip := b2.rip + 4 }).reg 1 = some wd.ann.length
    rw [reg_with_rip, reg_set_self (len_of_reg hb2r12 (by decide))]
  have hb3r0 : b3.reg 0 = some p := by
    rw [hb3]; show ({ b2.set 1 (some wd.ann.length) with rip := b2.rip + 4 }).reg 0 = some p
    rw [reg_with_rip, reg_set_other (by decide)]; exact hb2r0
  have hb3r4 : b3.reg 4 = some (sp - 8) := by
    rw [hb3]; show ({ b2.set 1 (some wd.ann.length) with rip := b2.rip + 4 }).reg 4 = some (sp - 8)
    rw [reg_with_rip, reg_set_other (by decide)]; exact hb2r4
  have hb3r7 : b3.reg 7 = some w := by
    rw [hb3]; show ({ b2.set 1 (some wd.ann.length) with rip := b2.rip + 4 }).reg 7 = some w
    rw [reg_with_rip, reg_set_other (by decide)]; exact hb2r7
  have hb3r12 : b3.reg 12 = some t := by
    rw [hb3]; show ({ b2.set 1 (some wd.ann.length) with rip := b2.rip + 4 }).reg 12 = some t
    rw [reg_with_rip, reg_set_other (by decide)]; exact hb2r12
  have hb3mem : b3.mem = m1 := by
    rw [hb3]; show ({ b2.set 1 (some wd.ann.length) with rip := b2.rip + 4 }).mem = m1
    rw [mem_with_rip, mem_set]; exact hb2mem
  have gb3 : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → b3.reg i = s.reg i := by
    intro i h0 h1 h2 h4
    rw [hb3]; show ({ b2.set 1 (some wd.ann.length) with rip := b2.rip + 4 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other h1]; exact gb2 i h0 h1 h2 h4
  have lay3 : AnnLayout b3 wd.ann p := AnnLayout_of_mem (by rw [hb3mem, hb1mem]) lay1
  -- the loop
  obtain ⟨n, s', hrun, hrip', hpost, hr1', hr4', hr12', hgen, hframe, hr0e⟩ :=
    annl_loop hc hl hentry wd.ann t p sp (cv % W) b3 htW hp0 hbound hsp24 hspW hb3rip hb3r0
      hb3r1 hb3r12 hb3r4 (by rw [hb3mem]; exact hsave1) lay3
      (by intro v; rw [hb3mem]; exact hw16' v) (by intro v; rw [hb3mem]; exact hw24' v) hdisjp
  have lay' : AnnLayout s' wd.ann p := by
    intro i hi
    obtain ⟨⟨h1, h2, h3⟩, hw⟩ := lay3 i hi
    refine ⟨⟨?_, ?_, ?_⟩, hw⟩
    · show (s'.mem.rdN (p + 12 * i) 4).map leVal = some wd.ann[i].1
      rw [hframe (p + 12 * i) 4 (by cases hdisjp with
        | inl h => exact Or.inr (by omega)
        | inr h => exact Or.inl (by omega))]
      exact h1
    · show (s'.mem.rdN (p + 12 * i + 4) 4).map leVal = some wd.ann[i].2.d
      rw [hframe (p + 12 * i + 4) 4 (by cases hdisjp with
        | inl h => exact Or.inr (by omega)
        | inr h => exact Or.inl (by omega))]
      exact h2
    · show (s'.mem.rdN (p + 12 * i + 8) 4).map leVal = some wd.ann[i].2.a
      rw [hframe (p + 12 * i + 8) 4 (by cases hdisjp with
        | inl h => exact Or.inr (by omega)
        | inr h => exact Or.inl (by omega))]
      exact h3
  have r1 : run im 1 (Out.ok s) = Out.ok b1 := by rw [run_succ_ok im 0 s, e1, ← hb1, run_zero]
  have r2 : run im 1 (Out.ok b1) = Out.ok b2 := by rw [run_succ_ok im 0 b1, e2, ← hb2, run_zero]
  have r3 : run im 1 (Out.ok b2) = Out.ok b3 := by rw [run_succ_ok im 0 b2, e3, ← hb3, run_zero]
  refine ⟨3 + n, s', ?_, hrip', hpost, hr1', hr4', hr12', ?_, lay', ?_, ?_, hr0e⟩
  · rw [run_add im 3 n,
      show run im 3 (Out.ok s) = Out.ok b3 from by
        rw [run_add im 1 2, r1, run_add im 1 1, r2, r3],
      hrun]
  · rw [hgen 7 (by decide) (by decide) (by decide) (by decide)]
    exact hb3r7
  · intro i h0 h1 h2 h4
    rw [hgen i h0 h1 h2 h4]
    exact gb3 i h0 h1 h2 h4
  · intro x n hx
    rw [hframe x n hx, hb3mem]
    exact rdN_wr64_other hm1s (by cases hx with
      | inl h => exact Or.inr (by omega)
      | inr h => exact Or.inl (by omega))


/-! ### Handler proofs: the straight-line arithmetic handlers -/

theorem aluI_add_val {a k : Nat} (hk : k < W) (h : a + k < W) :
    (aluF .add a (k % W)).1 = a + k := by
  rw [aluF_add_fst, Nat.mod_eq_of_lt hk, Nat.mod_eq_of_lt h]

theorem aluI_sub_val {a k : Nat} (hk : k < W) (hle : k ≤ a) (ha : a < W) :
    (aluF .sub a (k % W)).1 = a - k := by
  rw [aluF_sub_fst, Nat.mod_eq_of_lt hk, show a + W - k = (a - k) + W by omega,
    Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]

theorem cmpI_cf_false {a k : Nat} (hk : k < W) (h : ¬ a < k) :
    (aluF .sub a (k % W)).2.cf = some false := by
  rw [Nat.mod_eq_of_lt hk]
  exact aluF_sub_cf h

theorem len_of_r9 {s : X86Int.St} {v : Nat} (h : s.reg 9 = some v) {i : Nat}
    (hi : i ≤ 9) : i < s.regs.length := by
  have := reg_some_len h
  omega

/-- Decode facts for `h_lit`: add r8,1; add rsi,9; jmp tail. -/
structure LitCode (im : Nat → Option (Ins × Nat)) (entry tail : Nat) : Prop where
  d0 : im entry = some (.aluI .add 8 1, 4)
  d1 : im (entry + 4) = some (.aluI .add 6 9, 4)
  d2 : im (entry + 8) = some (.jmp tail, 5)

theorem Lit_triple {im : Nat → Option (Ins × Nat)} {entry tail : Nat}
    (hc : LitCode im entry tail) (pr : Prog) (wd : Word) (n : Nat) :
    HandlerSpec im entry tail pr wd (.lit n) 9 := by
  intro h c _hok hdW _haW hcW s hrip hpre
  obtain ⟨hr8, hr9, hr6, -⟩ := hpre
  have h1W : 1 < W := by unfold W; decide
  have h9W : 9 < W := by unfold W; decide
  have e1 : X86Int.step im s = .ok { s.set 8 (some (aluF .add h.d (1 % W)).1) with
      fl := (aluF .add h.d (1 % W)).2, rip := s.rip + 4 } :=
    step_aluI_add (by rw [hrip]; exact hc.d0) hr8
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s.set 8 (some (aluF .add h.d (1 % W)).1) with
    fl := (aluF .add h.d (1 % W)).2, rip := s.rip + 4 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = entry + 4 := by rw [ha1]; show s.rip + 4 = entry + 4; rw [hrip]
  have hadd1 : (aluF .add h.d (1 % W)).1 = h.d + 1 := aluI_add_val h1W (by omega)
  have ha1r8 : a1.reg 8 = some (h.d + 1) := by
    rw [ha1]
    show (s.set 8 (some (aluF .add h.d (1 % W)).1)).reg 8 = some (h.d + 1)
    rw [reg_set_self (len_of_r9 hr9 (by decide)), hadd1]
  have ha1r9 : a1.reg 9 = some h.a := by
    rw [ha1]
    show (s.set 8 (some (aluF .add h.d (1 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact hr9
  have ha1r6 : a1.reg 6 = some c := by
    rw [ha1]
    show (s.set 8 (some (aluF .add h.d (1 % W)).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact hr6
  have e2 : X86Int.step im a1 = .ok { a1.set 6 (some (aluF .add c (9 % W)).1) with
      fl := (aluF .add c (9 % W)).2, rip := a1.rip + 4 } :=
    step_aluI_add (by rw [ha1rip]; exact hc.d1) ha1r6
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1.set 6 (some (aluF .add c (9 % W)).1) with
    fl := (aluF .add c (9 % W)).2, rip := a1.rip + 4 } := ⟨_, rfl⟩
  have ha2rip : a2.rip = entry + 8 := by rw [ha2]; show a1.rip + 4 = entry + 8; rw [ha1rip]
  have hadd9 : (aluF .add c (9 % W)).1 = c + 9 := aluI_add_val h9W hcW
  have ha2r6 : a2.reg 6 = some (c + 9) := by
    rw [ha2]
    show (a1.set 6 (some (aluF .add c (9 % W)).1)).reg 6 = some (c + 9)
    rw [reg_set_self (len_of_r9 ha1r9 (by decide)), hadd9]
  have ha2r8 : a2.reg 8 = some (h.d + 1) := by
    rw [ha2]
    show (a1.set 6 (some (aluF .add c (9 % W)).1)).reg 8 = some (h.d + 1)
    rw [reg_set_other (by decide)]; exact ha1r8
  have ha2r9 : a2.reg 9 = some h.a := by
    rw [ha2]
    show (a1.set 6 (some (aluF .add c (9 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha1r9
  have e3 : X86Int.step im a2 = .ok { a2 with rip := tail } :=
    step_jmp (by rw [ha2rip]; exact hc.d2)
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2 with rip := tail } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
  refine ⟨1 + 1 + 1, a3, ?_, by rw [ha3], ?_, ?_, ?_⟩
  · rw [run_add im 1 2, r1, run_add im 1 1, r2, r3]
  · intro h' hh
    have hh' : h' = ⟨h.d + 1, h.a⟩ := by
      simp only [hstep, Option.some.injEq] at hh
      exact hh.symm
    constructor
    · rw [ha3]
      show ({ a2 with rip := tail }).reg 8 = some h'.d
      rw [reg_with_rip, hh']
      exact ha2r8
    · rw [ha3]
      show ({ a2 with rip := tail }).reg 9 = some h'.a
      rw [reg_with_rip, hh']
      exact ha2r9
  · intro hh
    simp [hstep] at hh
  · rw [ha3]
    show ({ a2 with rip := tail }).reg 6 = some (c + 9)
    rw [reg_with_rip]
    exact ha2r6


/-- Decode facts for `h_str`: add r8,2; add rsi,5; jmp tail. -/
structure StrCode (im : Nat → Option (Ins × Nat)) (entry tail : Nat) : Prop where
  d0 : im entry = some (.aluI .add 8 2, 4)
  d1 : im (entry + 4) = some (.aluI .add 6 5, 4)
  d2 : im (entry + 8) = some (.jmp tail, 5)

theorem Str_triple {im : Nat → Option (Ins × Nat)} {entry tail : Nat}
    (hc : StrCode im entry tail) (pr : Prog) (wd : Word) (n : Nat) :
    HandlerSpec im entry tail pr wd (.str n) 5 := by
  intro h c _hok hdW _haW hcW s hrip hpre
  obtain ⟨hr8, hr9, hr6, -⟩ := hpre
  have h2W : 2 < W := by unfold W; decide
  have h5W : 5 < W := by unfold W; decide
  have e1 : X86Int.step im s = .ok { s.set 8 (some (aluF .add h.d (2 % W)).1) with
      fl := (aluF .add h.d (2 % W)).2, rip := s.rip + 4 } :=
    step_aluI_add (by rw [hrip]; exact hc.d0) hr8
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s.set 8 (some (aluF .add h.d (2 % W)).1) with
    fl := (aluF .add h.d (2 % W)).2, rip := s.rip + 4 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = entry + 4 := by rw [ha1]; show s.rip + 4 = entry + 4; rw [hrip]
  have hadd2 : (aluF .add h.d (2 % W)).1 = h.d + 2 := aluI_add_val h2W (by omega)
  have ha1r8 : a1.reg 8 = some (h.d + 2) := by
    rw [ha1]
    show (s.set 8 (some (aluF .add h.d (2 % W)).1)).reg 8 = some (h.d + 2)
    rw [reg_set_self (len_of_r9 hr9 (by decide)), hadd2]
  have ha1r9 : a1.reg 9 = some h.a := by
    rw [ha1]
    show (s.set 8 (some (aluF .add h.d (2 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact hr9
  have ha1r6 : a1.reg 6 = some c := by
    rw [ha1]
    show (s.set 8 (some (aluF .add h.d (2 % W)).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact hr6
  have e2 : X86Int.step im a1 = .ok { a1.set 6 (some (aluF .add c (5 % W)).1) with
      fl := (aluF .add c (5 % W)).2, rip := a1.rip + 4 } :=
    step_aluI_add (by rw [ha1rip]; exact hc.d1) ha1r6
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1.set 6 (some (aluF .add c (5 % W)).1) with
    fl := (aluF .add c (5 % W)).2, rip := a1.rip + 4 } := ⟨_, rfl⟩
  have ha2rip : a2.rip = entry + 8 := by rw [ha2]; show a1.rip + 4 = entry + 8; rw [ha1rip]
  have hadd5 : (aluF .add c (5 % W)).1 = c + 5 := aluI_add_val h5W hcW
  have ha2r6 : a2.reg 6 = some (c + 5) := by
    rw [ha2]
    show (a1.set 6 (some (aluF .add c (5 % W)).1)).reg 6 = some (c + 5)
    rw [reg_set_self (len_of_r9 ha1r9 (by decide)), hadd5]
  have ha2r8 : a2.reg 8 = some (h.d + 2) := by
    rw [ha2]
    show (a1.set 6 (some (aluF .add c (5 % W)).1)).reg 8 = some (h.d + 2)
    rw [reg_set_other (by decide)]; exact ha1r8
  have ha2r9 : a2.reg 9 = some h.a := by
    rw [ha2]
    show (a1.set 6 (some (aluF .add c (5 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha1r9
  have e3 : X86Int.step im a2 = .ok { a2 with rip := tail } :=
    step_jmp (by rw [ha2rip]; exact hc.d2)
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2 with rip := tail } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
  refine ⟨1 + 1 + 1, a3, ?_, by rw [ha3], ?_, ?_, ?_⟩
  · rw [run_add im 1 2, r1, run_add im 1 1, r2, r3]
  · intro h' hh
    have hh' : h' = ⟨h.d + 2, h.a⟩ := by
      simp only [hstep, Option.some.injEq] at hh
      exact hh.symm
    constructor
    · rw [ha3]
      show ({ a2 with rip := tail }).reg 8 = some h'.d
      rw [reg_with_rip, hh']
      exact ha2r8
    · rw [ha3]
      show ({ a2 with rip := tail }).reg 9 = some h'.a
      rw [reg_with_rip, hh']
      exact ha2r9
  · intro hh
    simp [hstep] at hh
  · rw [ha3]
    show ({ a2 with rip := tail }).reg 6 = some (c + 5)
    rw [reg_with_rip]
    exact ha2r6

/-- Decode facts for `h_tor`: cmp r8,1; jae; sub r8,1; add r9,1;
add rsi,1; jmp tail. -/
structure TorCode (im : Nat → Option (Ins × Nat)) (entry tail : Nat) : Prop where
  d0 : im entry = some (.cmpI 8 1, 4)
  d1 : im (entry + 4) = some (.jcc 3 (entry + 11), 2)
  d2 : im (entry + 11) = some (.aluI .sub 8 1, 4)
  d3 : im (entry + 15) = some (.aluI .add 9 1, 4)
  d4 : im (entry + 19) = some (.aluI .add 6 1, 4)
  d5 : im (entry + 23) = some (.jmp tail, 5)

theorem Tor_triple {im : Nat → Option (Ins × Nat)} {entry tail : Nat}
    (hc : TorCode im entry tail) (pr : Prog) (wd : Word) :
    HandlerSpec im entry tail pr wd .tor 1 := by
  intro h c hok hdW haW hcW s hrip hpre
  obtain ⟨hr8, hr9, hr6, -⟩ := hpre
  have h1W : 1 < W := by unfold W; decide
  have hd1 : 1 ≤ h.d := by
    by_cases hd : 1 ≤ h.d
    · exact hd
    · simp only [hstep, if_neg hd, Option.isSome_none, Bool.false_eq_true] at hok
  have hhe : hstep pr wd h .tor = some (some ⟨h.d - 1, h.a + 1⟩) := by
    simp [hstep, hd1]
  have e1 : X86Int.step im s = .ok { s with fl := (aluF .sub h.d (1 % W)).2, rip := s.rip + 4 } :=
    step_cmpI (by rw [hrip]; exact hc.d0) hr8
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s with fl := (aluF .sub h.d (1 % W)).2, rip := s.rip + 4 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = entry + 4 := by rw [ha1]; show s.rip + 4 = entry + 4; rw [hrip]
  have ha1fl : X86Int.cond a1.fl 3 = some true := by
    have hcf : a1.fl.cf = some false := by
      rw [ha1]
      show (aluF .sub h.d (1 % W)).2.cf = some false
      exact cmpI_cf_false h1W (by omega)
    have := cond_cc3 hcf
    simpa using this
  have ha1r8 : a1.reg 8 = some h.d := by rw [ha1]; show s.reg 8 = some h.d; exact hr8
  have ha1r9 : a1.reg 9 = some h.a := by rw [ha1]; show s.reg 9 = some h.a; exact hr9
  have ha1r6 : a1.reg 6 = some c := by rw [ha1]; show s.reg 6 = some c; exact hr6
  have e2 : X86Int.step im a1 = .ok { a1 with rip := entry + 11 } := by
    have hh := step_jcc (im := im) (s := a1) (by rw [ha1rip]; exact hc.d1) ha1fl
    simpa using hh
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1 with rip := entry + 11 } := ⟨_, rfl⟩
  have ha2rip : a2.rip = entry + 11 := by rw [ha2]
  have ha2r8 : a2.reg 8 = some h.d := by
    rw [ha2]; show ({ a1 with rip := entry + 11 }).reg 8 = some h.d
    rw [reg_with_rip]; exact ha1r8
  have ha2r9 : a2.reg 9 = some h.a := by
    rw [ha2]; show ({ a1 with rip := entry + 11 }).reg 9 = some h.a
    rw [reg_with_rip]; exact ha1r9
  have ha2r6 : a2.reg 6 = some c := by
    rw [ha2]; show ({ a1 with rip := entry + 11 }).reg 6 = some c
    rw [reg_with_rip]; exact ha1r6
  have hsub : (aluF .sub h.d (1 % W)).1 = h.d - 1 := aluI_sub_val h1W hd1 (by omega)
  have e3 : X86Int.step im a2 = .ok { a2.set 8 (some (aluF .sub h.d (1 % W)).1) with
      fl := (aluF .sub h.d (1 % W)).2, rip := a2.rip + 4 } :=
    step_aluI_sub (by rw [ha2rip]; exact hc.d2) ha2r8
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2.set 8 (some (aluF .sub h.d (1 % W)).1) with
    fl := (aluF .sub h.d (1 % W)).2, rip := a2.rip + 4 } := ⟨_, rfl⟩
  have ha3rip : a3.rip = entry + 15 := by rw [ha3]; show a2.rip + 4 = entry + 15; rw [ha2rip]
  have ha3r8 : a3.reg 8 = some (h.d - 1) := by
    rw [ha3]
    show (a2.set 8 (some (aluF .sub h.d (1 % W)).1)).reg 8 = some (h.d - 1)
    rw [reg_set_self (len_of_r9 ha2r9 (by decide)), hsub]
  have ha3r9 : a3.reg 9 = some h.a := by
    rw [ha3]
    show (a2.set 8 (some (aluF .sub h.d (1 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha2r9
  have ha3r6 : a3.reg 6 = some c := by
    rw [ha3]
    show (a2.set 8 (some (aluF .sub h.d (1 % W)).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha2r6
  have hadd9 : (aluF .add h.a (1 % W)).1 = h.a + 1 := aluI_add_val h1W haW
  have e4 : X86Int.step im a3 = .ok { a3.set 9 (some (aluF .add h.a (1 % W)).1) with
      fl := (aluF .add h.a (1 % W)).2, rip := a3.rip + 4 } :=
    step_aluI_add (by rw [ha3rip]; exact hc.d3) ha3r9
  obtain ⟨a4, ha4⟩ : ∃ u : X86Int.St, u = { a3.set 9 (some (aluF .add h.a (1 % W)).1) with
    fl := (aluF .add h.a (1 % W)).2, rip := a3.rip + 4 } := ⟨_, rfl⟩
  have ha4rip : a4.rip = entry + 19 := by rw [ha4]; show a3.rip + 4 = entry + 19; rw [ha3rip]
  have ha4r9 : a4.reg 9 = some (h.a + 1) := by
    rw [ha4]
    show (a3.set 9 (some (aluF .add h.a (1 % W)).1)).reg 9 = some (h.a + 1)
    rw [reg_set_self (len_of_r9 ha3r9 (by decide)), hadd9]
  have ha4r8 : a4.reg 8 = some (h.d - 1) := by
    rw [ha4]
    show (a3.set 9 (some (aluF .add h.a (1 % W)).1)).reg 8 = some (h.d - 1)
    rw [reg_set_other (by decide)]; exact ha3r8
  have ha4r6 : a4.reg 6 = some c := by
    rw [ha4]
    show (a3.set 9 (some (aluF .add h.a (1 % W)).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha3r6
  have hadd6 : (aluF .add c (1 % W)).1 = c + 1 := aluI_add_val h1W hcW
  have e5 : X86Int.step im a4 = .ok { a4.set 6 (some (aluF .add c (1 % W)).1) with
      fl := (aluF .add c (1 % W)).2, rip := a4.rip + 4 } :=
    step_aluI_add (by rw [ha4rip]; exact hc.d4) ha4r6
  obtain ⟨a5, ha5⟩ : ∃ u : X86Int.St, u = { a4.set 6 (some (aluF .add c (1 % W)).1) with
    fl := (aluF .add c (1 % W)).2, rip := a4.rip + 4 } := ⟨_, rfl⟩
  have ha5rip : a5.rip = entry + 23 := by rw [ha5]; show a4.rip + 4 = entry + 23; rw [ha4rip]
  have ha5r6 : a5.reg 6 = some (c + 1) := by
    rw [ha5]
    show (a4.set 6 (some (aluF .add c (1 % W)).1)).reg 6 = some (c + 1)
    rw [reg_set_self (len_of_r9 ha4r9 (by decide)), hadd6]
  have ha5r8 : a5.reg 8 = some (h.d - 1) := by
    rw [ha5]
    show (a4.set 6 (some (aluF .add c (1 % W)).1)).reg 8 = some (h.d - 1)
    rw [reg_set_other (by decide)]; exact ha4r8
  have ha5r9 : a5.reg 9 = some (h.a + 1) := by
    rw [ha5]
    show (a4.set 6 (some (aluF .add c (1 % W)).1)).reg 9 = some (h.a + 1)
    rw [reg_set_other (by decide)]; exact ha4r9
  have e6 : X86Int.step im a5 = .ok { a5 with rip := tail } :=
    step_jmp (by rw [ha5rip]; exact hc.d5)
  obtain ⟨a6, ha6⟩ : ∃ u : X86Int.St, u = { a5 with rip := tail } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
  have r4 : run im 1 (Out.ok a3) = Out.ok a4 := by rw [run_succ_ok im 0 a3, e4, ← ha4, run_zero]
  have r5 : run im 1 (Out.ok a4) = Out.ok a5 := by rw [run_succ_ok im 0 a4, e5, ← ha5, run_zero]
  have r6 : run im 1 (Out.ok a5) = Out.ok a6 := by rw [run_succ_ok im 0 a5, e6, ← ha6, run_zero]
  refine ⟨1 + 1 + 1 + 1 + 1 + 1, a6, ?_, by rw [ha6], ?_, ?_, ?_⟩
  · rw [run_add im 1 5, r1, run_add im 1 4, r2, run_add im 1 3, r3, run_add im 1 2, r4,
      run_add im 1 1, r5, r6]
  · intro h' hh
    rw [hhe] at hh
    have hh' : h' = ⟨h.d - 1, h.a + 1⟩ := (Option.some.inj (Option.some.inj hh)).symm
    constructor
    · rw [ha6]
      show ({ a5 with rip := tail }).reg 8 = some h'.d
      rw [reg_with_rip, hh']
      exact ha5r8
    · rw [ha6]
      show ({ a5 with rip := tail }).reg 9 = some h'.a
      rw [reg_with_rip, hh']
      exact ha5r9
  · intro hh
    rw [hhe] at hh
    simp only [Option.some.injEq] at hh
    cases hh
  · rw [ha6]
    show ({ a5 with rip := tail }).reg 6 = some (c + 1)
    rw [reg_with_rip]
    exact ha5r6


/-- Decode facts for `h_fromr`: cmp r9,1; jae; add r8,1; sub r9,1;
add rsi,1; jmp tail. -/
structure FromrCode (im : Nat → Option (Ins × Nat)) (entry tail : Nat) : Prop where
  d0 : im entry = some (.cmpI 9 1, 4)
  d1 : im (entry + 4) = some (.jcc 3 (entry + 11), 2)
  d2 : im (entry + 11) = some (.aluI .add 8 1, 4)
  d3 : im (entry + 15) = some (.aluI .sub 9 1, 4)
  d4 : im (entry + 19) = some (.aluI .add 6 1, 4)
  d5 : im (entry + 23) = some (.jmp tail, 5)

theorem Fromr_triple {im : Nat → Option (Ins × Nat)} {entry tail : Nat}
    (hc : FromrCode im entry tail) (pr : Prog) (wd : Word) :
    HandlerSpec im entry tail pr wd .fromr 1 := by
  intro h c hok hdW haW hcW s hrip hpre
  obtain ⟨hr8, hr9, hr6, -⟩ := hpre
  have h1W : 1 < W := by unfold W; decide
  have ha1le : 1 ≤ h.a := by
    by_cases hd : 1 ≤ h.a
    · exact hd
    · simp only [hstep, if_neg hd, Option.isSome_none, Bool.false_eq_true] at hok
  have hhe : hstep pr wd h .fromr = some (some ⟨h.d + 1, h.a - 1⟩) := by
    simp [hstep, ha1le]
  have e1 : X86Int.step im s = .ok { s with fl := (aluF .sub h.a (1 % W)).2, rip := s.rip + 4 } :=
    step_cmpI (by rw [hrip]; exact hc.d0) hr9
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s with fl := (aluF .sub h.a (1 % W)).2, rip := s.rip + 4 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = entry + 4 := by rw [ha1]; show s.rip + 4 = entry + 4; rw [hrip]
  have ha1fl : X86Int.cond a1.fl 3 = some true := by
    have hcf : a1.fl.cf = some false := by
      rw [ha1]
      show (aluF .sub h.a (1 % W)).2.cf = some false
      exact cmpI_cf_false h1W (by omega)
    have := cond_cc3 hcf
    simpa using this
  have ha1r8 : a1.reg 8 = some h.d := by rw [ha1]; show s.reg 8 = some h.d; exact hr8
  have ha1r9 : a1.reg 9 = some h.a := by rw [ha1]; show s.reg 9 = some h.a; exact hr9
  have ha1r6 : a1.reg 6 = some c := by rw [ha1]; show s.reg 6 = some c; exact hr6
  have e2 : X86Int.step im a1 = .ok { a1 with rip := entry + 11 } := by
    have hh := step_jcc (im := im) (s := a1) (by rw [ha1rip]; exact hc.d1) ha1fl
    simpa using hh
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1 with rip := entry + 11 } := ⟨_, rfl⟩
  have ha2rip : a2.rip = entry + 11 := by rw [ha2]
  have ha2r8 : a2.reg 8 = some h.d := by
    rw [ha2]; show ({ a1 with rip := entry + 11 }).reg 8 = some h.d
    rw [reg_with_rip]; exact ha1r8
  have ha2r9 : a2.reg 9 = some h.a := by
    rw [ha2]; show ({ a1 with rip := entry + 11 }).reg 9 = some h.a
    rw [reg_with_rip]; exact ha1r9
  have ha2r6 : a2.reg 6 = some c := by
    rw [ha2]; show ({ a1 with rip := entry + 11 }).reg 6 = some c
    rw [reg_with_rip]; exact ha1r6
  have hadd8 : (aluF .add h.d (1 % W)).1 = h.d + 1 := aluI_add_val h1W (by omega)
  have e3 : X86Int.step im a2 = .ok { a2.set 8 (some (aluF .add h.d (1 % W)).1) with
      fl := (aluF .add h.d (1 % W)).2, rip := a2.rip + 4 } :=
    step_aluI_add (by rw [ha2rip]; exact hc.d2) ha2r8
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2.set 8 (some (aluF .add h.d (1 % W)).1) with
    fl := (aluF .add h.d (1 % W)).2, rip := a2.rip + 4 } := ⟨_, rfl⟩
  have ha3rip : a3.rip = entry + 15 := by rw [ha3]; show a2.rip + 4 = entry + 15; rw [ha2rip]
  have ha3r8 : a3.reg 8 = some (h.d + 1) := by
    rw [ha3]
    show (a2.set 8 (some (aluF .add h.d (1 % W)).1)).reg 8 = some (h.d + 1)
    rw [reg_set_self (len_of_r9 ha2r9 (by decide)), hadd8]
  have ha3r9 : a3.reg 9 = some h.a := by
    rw [ha3]
    show (a2.set 8 (some (aluF .add h.d (1 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha2r9
  have ha3r6 : a3.reg 6 = some c := by
    rw [ha3]
    show (a2.set 8 (some (aluF .add h.d (1 % W)).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha2r6
  have hsub9 : (aluF .sub h.a (1 % W)).1 = h.a - 1 := aluI_sub_val h1W ha1le (by omega)
  have e4 : X86Int.step im a3 = .ok { a3.set 9 (some (aluF .sub h.a (1 % W)).1) with
      fl := (aluF .sub h.a (1 % W)).2, rip := a3.rip + 4 } :=
    step_aluI_sub (by rw [ha3rip]; exact hc.d3) ha3r9
  obtain ⟨a4, ha4⟩ : ∃ u : X86Int.St, u = { a3.set 9 (some (aluF .sub h.a (1 % W)).1) with
    fl := (aluF .sub h.a (1 % W)).2, rip := a3.rip + 4 } := ⟨_, rfl⟩
  have ha4rip : a4.rip = entry + 19 := by rw [ha4]; show a3.rip + 4 = entry + 19; rw [ha3rip]
  have ha4r9 : a4.reg 9 = some (h.a - 1) := by
    rw [ha4]
    show (a3.set 9 (some (aluF .sub h.a (1 % W)).1)).reg 9 = some (h.a - 1)
    rw [reg_set_self (len_of_r9 ha3r9 (by decide)), hsub9]
  have ha4r8 : a4.reg 8 = some (h.d + 1) := by
    rw [ha4]
    show (a3.set 9 (some (aluF .sub h.a (1 % W)).1)).reg 8 = some (h.d + 1)
    rw [reg_set_other (by decide)]; exact ha3r8
  have ha4r6 : a4.reg 6 = some c := by
    rw [ha4]
    show (a3.set 9 (some (aluF .sub h.a (1 % W)).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha3r6
  have hadd6 : (aluF .add c (1 % W)).1 = c + 1 := aluI_add_val h1W hcW
  have e5 : X86Int.step im a4 = .ok { a4.set 6 (some (aluF .add c (1 % W)).1) with
      fl := (aluF .add c (1 % W)).2, rip := a4.rip + 4 } :=
    step_aluI_add (by rw [ha4rip]; exact hc.d4) ha4r6
  obtain ⟨a5, ha5⟩ : ∃ u : X86Int.St, u = { a4.set 6 (some (aluF .add c (1 % W)).1) with
    fl := (aluF .add c (1 % W)).2, rip := a4.rip + 4 } := ⟨_, rfl⟩
  have ha5rip : a5.rip = entry + 23 := by rw [ha5]; show a4.rip + 4 = entry + 23; rw [ha4rip]
  have ha5r6 : a5.reg 6 = some (c + 1) := by
    rw [ha5]
    show (a4.set 6 (some (aluF .add c (1 % W)).1)).reg 6 = some (c + 1)
    rw [reg_set_self (len_of_r9 ha4r9 (by decide)), hadd6]
  have ha5r8 : a5.reg 8 = some (h.d + 1) := by
    rw [ha5]
    show (a4.set 6 (some (aluF .add c (1 % W)).1)).reg 8 = some (h.d + 1)
    rw [reg_set_other (by decide)]; exact ha4r8
  have ha5r9 : a5.reg 9 = some (h.a - 1) := by
    rw [ha5]
    show (a4.set 6 (some (aluF .add c (1 % W)).1)).reg 9 = some (h.a - 1)
    rw [reg_set_other (by decide)]; exact ha4r9
  have e6 : X86Int.step im a5 = .ok { a5 with rip := tail } :=
    step_jmp (by rw [ha5rip]; exact hc.d5)
  obtain ⟨a6, ha6⟩ : ∃ u : X86Int.St, u = { a5 with rip := tail } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
  have r4 : run im 1 (Out.ok a3) = Out.ok a4 := by rw [run_succ_ok im 0 a3, e4, ← ha4, run_zero]
  have r5 : run im 1 (Out.ok a4) = Out.ok a5 := by rw [run_succ_ok im 0 a4, e5, ← ha5, run_zero]
  have r6 : run im 1 (Out.ok a5) = Out.ok a6 := by rw [run_succ_ok im 0 a5, e6, ← ha6, run_zero]
  refine ⟨1 + 1 + 1 + 1 + 1 + 1, a6, ?_, by rw [ha6], ?_, ?_, ?_⟩
  · rw [run_add im 1 5, r1, run_add im 1 4, r2, run_add im 1 3, r3, run_add im 1 2, r4,
      run_add im 1 1, r5, r6]
  · intro h' hh
    rw [hhe] at hh
    have hh' : h' = ⟨h.d + 1, h.a - 1⟩ := (Option.some.inj (Option.some.inj hh)).symm
    constructor
    · rw [ha6]
      show ({ a5 with rip := tail }).reg 8 = some h'.d
      rw [reg_with_rip, hh']
      exact ha5r8
    · rw [ha6]
      show ({ a5 with rip := tail }).reg 9 = some h'.a
      rw [reg_with_rip, hh']
      exact ha5r9
  · intro hh
    rw [hhe] at hh
    simp only [Option.some.injEq] at hh
    cases hh
  · rw [ha6]
    show ({ a5 with rip := tail }).reg 6 = some (c + 1)
    rw [reg_with_rip]
    exact ha5r6

/-- Decode facts for `h_rfetch`: cmp r9,1; jae; add r8,1; add rsi,1;
jmp tail. -/
structure RfetchCode (im : Nat → Option (Ins × Nat)) (entry tail : Nat) : Prop where
  d0 : im entry = some (.cmpI 9 1, 4)
  d1 : im (entry + 4) = some (.jcc 3 (entry + 11), 2)
  d2 : im (entry + 11) = some (.aluI .add 8 1, 4)
  d3 : im (entry + 15) = some (.aluI .add 6 1, 4)
  d4 : im (entry + 19) = some (.jmp tail, 5)

theorem Rfetch_triple {im : Nat → Option (Ins × Nat)} {entry tail : Nat}
    (hc : RfetchCode im entry tail) (pr : Prog) (wd : Word) :
    HandlerSpec im entry tail pr wd .rfetch 1 := by
  intro h c hok hdW _haW hcW s hrip hpre
  obtain ⟨hr8, hr9, hr6, -⟩ := hpre
  have h1W : 1 < W := by unfold W; decide
  have ha1le : 1 ≤ h.a := by
    by_cases hd : 1 ≤ h.a
    · exact hd
    · simp only [hstep, if_neg hd, Option.isSome_none, Bool.false_eq_true] at hok
  have hhe : hstep pr wd h .rfetch = some (some ⟨h.d + 1, h.a⟩) := by
    simp [hstep, ha1le]
  have e1 : X86Int.step im s = .ok { s with fl := (aluF .sub h.a (1 % W)).2, rip := s.rip + 4 } :=
    step_cmpI (by rw [hrip]; exact hc.d0) hr9
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s with fl := (aluF .sub h.a (1 % W)).2, rip := s.rip + 4 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = entry + 4 := by rw [ha1]; show s.rip + 4 = entry + 4; rw [hrip]
  have ha1fl : X86Int.cond a1.fl 3 = some true := by
    have hcf : a1.fl.cf = some false := by
      rw [ha1]
      show (aluF .sub h.a (1 % W)).2.cf = some false
      exact cmpI_cf_false h1W (by omega)
    have := cond_cc3 hcf
    simpa using this
  have ha1r8 : a1.reg 8 = some h.d := by rw [ha1]; show s.reg 8 = some h.d; exact hr8
  have ha1r9 : a1.reg 9 = some h.a := by rw [ha1]; show s.reg 9 = some h.a; exact hr9
  have ha1r6 : a1.reg 6 = some c := by rw [ha1]; show s.reg 6 = some c; exact hr6
  have e2 : X86Int.step im a1 = .ok { a1 with rip := entry + 11 } := by
    have hh := step_jcc (im := im) (s := a1) (by rw [ha1rip]; exact hc.d1) ha1fl
    simpa using hh
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1 with rip := entry + 11 } := ⟨_, rfl⟩
  have ha2rip : a2.rip = entry + 11 := by rw [ha2]
  have ha2r8 : a2.reg 8 = some h.d := by
    rw [ha2]; show ({ a1 with rip := entry + 11 }).reg 8 = some h.d
    rw [reg_with_rip]; exact ha1r8
  have ha2r9 : a2.reg 9 = some h.a := by
    rw [ha2]; show ({ a1 with rip := entry + 11 }).reg 9 = some h.a
    rw [reg_with_rip]; exact ha1r9
  have ha2r6 : a2.reg 6 = some c := by
    rw [ha2]; show ({ a1 with rip := entry + 11 }).reg 6 = some c
    rw [reg_with_rip]; exact ha1r6
  have hadd8 : (aluF .add h.d (1 % W)).1 = h.d + 1 := aluI_add_val h1W (by omega)
  have e3 : X86Int.step im a2 = .ok { a2.set 8 (some (aluF .add h.d (1 % W)).1) with
      fl := (aluF .add h.d (1 % W)).2, rip := a2.rip + 4 } :=
    step_aluI_add (by rw [ha2rip]; exact hc.d2) ha2r8
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2.set 8 (some (aluF .add h.d (1 % W)).1) with
    fl := (aluF .add h.d (1 % W)).2, rip := a2.rip + 4 } := ⟨_, rfl⟩
  have ha3rip : a3.rip = entry + 15 := by rw [ha3]; show a2.rip + 4 = entry + 15; rw [ha2rip]
  have ha3r8 : a3.reg 8 = some (h.d + 1) := by
    rw [ha3]
    show (a2.set 8 (some (aluF .add h.d (1 % W)).1)).reg 8 = some (h.d + 1)
    rw [reg_set_self (len_of_r9 ha2r9 (by decide)), hadd8]
  have ha3r9 : a3.reg 9 = some h.a := by
    rw [ha3]
    show (a2.set 8 (some (aluF .add h.d (1 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha2r9
  have ha3r6 : a3.reg 6 = some c := by
    rw [ha3]
    show (a2.set 8 (some (aluF .add h.d (1 % W)).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha2r6
  have hadd6 : (aluF .add c (1 % W)).1 = c + 1 := aluI_add_val h1W hcW
  have e4 : X86Int.step im a3 = .ok { a3.set 6 (some (aluF .add c (1 % W)).1) with
      fl := (aluF .add c (1 % W)).2, rip := a3.rip + 4 } :=
    step_aluI_add (by rw [ha3rip]; exact hc.d3) ha3r6
  obtain ⟨a4, ha4⟩ : ∃ u : X86Int.St, u = { a3.set 6 (some (aluF .add c (1 % W)).1) with
    fl := (aluF .add c (1 % W)).2, rip := a3.rip + 4 } := ⟨_, rfl⟩
  have ha4rip : a4.rip = entry + 19 := by rw [ha4]; show a3.rip + 4 = entry + 19; rw [ha3rip]
  have ha4r6 : a4.reg 6 = some (c + 1) := by
    rw [ha4]
    show (a3.set 6 (some (aluF .add c (1 % W)).1)).reg 6 = some (c + 1)
    rw [reg_set_self (len_of_r9 ha3r9 (by decide)), hadd6]
  have ha4r8 : a4.reg 8 = some (h.d + 1) := by
    rw [ha4]
    show (a3.set 6 (some (aluF .add c (1 % W)).1)).reg 8 = some (h.d + 1)
    rw [reg_set_other (by decide)]; exact ha3r8
  have ha4r9 : a4.reg 9 = some h.a := by
    rw [ha4]
    show (a3.set 6 (some (aluF .add c (1 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha3r9
  have e5 : X86Int.step im a4 = .ok { a4 with rip := tail } :=
    step_jmp (by rw [ha4rip]; exact hc.d4)
  obtain ⟨a5, ha5⟩ : ∃ u : X86Int.St, u = { a4 with rip := tail } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
  have r4 : run im 1 (Out.ok a3) = Out.ok a4 := by rw [run_succ_ok im 0 a3, e4, ← ha4, run_zero]
  have r5 : run im 1 (Out.ok a4) = Out.ok a5 := by rw [run_succ_ok im 0 a4, e5, ← ha5, run_zero]
  refine ⟨1 + 1 + 1 + 1 + 1, a5, ?_, by rw [ha5], ?_, ?_, ?_⟩
  · rw [run_add im 1 4, r1, run_add im 1 3, r2, run_add im 1 2, r3, run_add im 1 1, r4, r5]
  · intro h' hh
    rw [hhe] at hh
    have hh' : h' = ⟨h.d + 1, h.a⟩ := (Option.some.inj (Option.some.inj hh)).symm
    constructor
    · rw [ha5]
      show ({ a4 with rip := tail }).reg 8 = some h'.d
      rw [reg_with_rip, hh']
      exact ha4r8
    · rw [ha5]
      show ({ a4 with rip := tail }).reg 9 = some h'.a
      rw [reg_with_rip, hh']
      exact ha4r9
  · intro hh
    rw [hhe] at hh
    simp only [Option.some.injEq] at hh
    cases hh
  · rw [ha5]
    show ({ a4 with rip := tail }).reg 6 = some (c + 1)
    rw [reg_with_rip]
    exact ha4r6


theorem cond_cc2 {f : Flags} {b : Bool} (h : f.cf = some b) : X86Int.cond f 2 = some b := by
  simp [X86Int.cond, h]

theorem aluF_sub_cf_true {a b : Nat} (h : a < b) : (aluF .sub a b).2.cf = some true := by
  have hlt : decide (a < b) = true := by simp [h]
  simp [aluF, hlt]

/-- The `pick` handler's spec, in the unfolded height form: the
immediate at cursor+1 reads as `n`, below the data height, and the
handler reaches the fall-through tail with height `d+1`, cursor
past the instruction, `rsp` restored, and every read outside the
one stack word preserved. Parametric in all addresses. -/
def PickSpec (im : Nat → Option (Ins × Nat)) (entry tail l32 : Nat) : Prop :=
  ∀ (h : Height) (c n sp : Nat), n < h.d → h.d + 1 < W → c + 5 < W → 8 ≤ sp → sp < W →
    ∀ s, s.rip = entry → s.reg 8 = some h.d → s.reg 9 = some h.a → s.reg 6 = some c →
      s.reg 4 = some sp →
      (s.mem.rdN (c + 1) 4).map leVal = some n →
      (∀ v, (wr64 s.mem (sp - 8) v).isSome) →
      (∀ m, wr64 s.mem (sp - 8) (entry + 12) = some m →
        m.rdN (c + 1) 4 = s.mem.rdN (c + 1) 4) →
      ∃ k s', run im k (Out.ok s) = Out.ok s' ∧ s'.rip = tail ∧
        s'.reg 8 = some (h.d + 1) ∧ s'.reg 9 = some h.a ∧ s'.reg 6 = some (c + 5) ∧
        s'.reg 4 = some sp ∧
        (∀ x m, x + m ≤ sp - 8 ∨ sp ≤ x → s'.mem.rdN x m = s.mem.rdN x m)

/-- Decode facts for `h_pick`: mov rdx,rsi; add rdx,1; call L32;
cmp rax,r8; jb; add r8,1; add rsi,5; jmp tail. -/
structure PickCode (im : Nat → Option (Ins × Nat)) (entry tail l32 : Nat) : Prop where
  d0 : im entry = some (.mov 2 6, 3)
  d1 : im (entry + 3) = some (.aluI .add 2 1, 4)
  d2 : im (entry + 7) = some (.call l32, 5)
  d3 : im (entry + 12) = some (.cmp 0 8, 3)
  d4 : im (entry + 15) = some (.jcc 2 (entry + 22), 2)
  d5 : im (entry + 22) = some (.aluI .add 8 1, 4)
  d6 : im (entry + 26) = some (.aluI .add 6 5, 4)
  d7 : im (entry + 30) = some (.jmp tail, 5)
  dret : im (l32 + 2) = some (.ret, 1)

theorem Pick_triple {im : Nat → Option (Ins × Nat)} {entry tail l32 : Nat}
    (hc : PickCode im entry tail l32) (hl : L32Code im l32) (hentry : entry + 35 < W) :
    PickSpec im entry tail l32 := by
  intro h c n sp hn hdW hcW hsp8 hspW s hrip hr8 hr9 hr6 hr4 hrd hw hframe
  have h1W : 1 < W := by unfold W; decide
  have h5W : 5 < W := by unfold W; decide
  -- mov rdx, rsi
  have e1 : X86Int.step im s = .ok { s.set 2 (some c) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact hc.d0) hr6
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s.set 2 (some c) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = entry + 3 := by rw [ha1]; show s.rip + 3 = entry + 3; rw [hrip]
  have ha1r2 : a1.reg 2 = some c := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 2 = some c
    rw [reg_with_rip, reg_set_self (len_of_r9 hr9 (by decide))]
  have ha1r8 : a1.reg 8 = some h.d := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 8 = some h.d
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr8
  have ha1r9 : a1.reg 9 = some h.a := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 9 = some h.a
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr9
  have ha1r6 : a1.reg 6 = some c := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr6
  have ha1r4 : a1.reg 4 = some sp := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 4 = some sp
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr4
  have ha1mem : a1.mem = s.mem := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  -- add rdx, 1
  have hadd2 : (aluF .add c (1 % W)).1 = c + 1 := aluI_add_val h1W (by omega)
  have e2 : X86Int.step im a1 = .ok { a1.set 2 (some (aluF .add c (1 % W)).1) with
      fl := (aluF .add c (1 % W)).2, rip := a1.rip + 4 } :=
    step_aluI_add (by rw [ha1rip]; exact hc.d1) ha1r2
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1.set 2 (some (aluF .add c (1 % W)).1) with
    fl := (aluF .add c (1 % W)).2, rip := a1.rip + 4 } := ⟨_, rfl⟩
  have ha2rip : a2.rip = entry + 7 := by rw [ha2]; show a1.rip + 4 = entry + 7; rw [ha1rip]
  have ha2r2 : a2.reg 2 = some (c + 1) := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 2 = some (c + 1)
    rw [reg_set_self (len_of_r9 ha1r9 (by decide)), hadd2]
  have ha2r8 : a2.reg 8 = some h.d := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact ha1r8
  have ha2r9 : a2.reg 9 = some h.a := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha1r9
  have ha2r6 : a2.reg 6 = some c := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha1r6
  have ha2r4 : a2.reg 4 = some sp := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact ha1r4
  have ha2mem : a2.mem = s.mem := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).mem = s.mem
    rw [mem_set]; exact ha1mem
  -- call L32
  have hslot8 : (sp + W - 8) % W = sp - 8 := by
    rw [show sp + W - 8 = sp + W - 8 * 1 by omega, slot_sub (by omega) hspW]
  obtain ⟨m1, hm1s⟩ := Option.isSome_iff_exists.mp (hw (entry + 12))
  have hm1 : wr64 a2.mem ((sp + W - 8) % W) (a2.rip + 5) = some m1 := by
    rw [ha2mem, ha2rip, hslot8]
    exact hm1s
  have e3 : X86Int.step im a2 = .ok { a2.set 4 (some ((sp + W - 8) % W)) with
      mem := m1, rip := l32 } :=
    step_call (by rw [ha2rip]; exact hc.d2) ha2r4 hm1
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2.set 4 (some ((sp + W - 8) % W)) with
    mem := m1, rip := l32 } := ⟨_, rfl⟩
  have ha3rip : a3.rip = l32 := by rw [ha3]
  have ha3r4 : a3.reg 4 = some (sp - 8) := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
    rw [reg_set_self (len_of_r9 ha2r9 (by decide)), hslot8]
  have ha3r2 : a3.reg 2 = some (c + 1) := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 2 = some (c + 1)
    rw [reg_set_other (by decide)]; exact ha2r2
  have ha3r8 : a3.reg 8 = some h.d := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact ha2r8
  have ha3r9 : a3.reg 9 = some h.a := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha2r9
  have ha3r6 : a3.reg 6 = some c := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha2r6
  have ha3mem : a3.mem = m1 := by rw [ha3]
  have hrdm1 : (m1.rdN (c + 1) 4).map leVal = some n := by
    rw [hframe m1 hm1s]
    exact hrd
  -- L32 as a black box
  have hrdsome : (a3.mem.rdN (c + 1) 4).isSome := by
    rw [ha3mem]
    cases h : m1.rdN (c + 1) 4 with
    | none => simp [h] at hrdm1
    | some bs => rfl
  obtain ⟨n8, a4, hrun8, ha4rip, ha4r0, -, ha4mem, ha4other, -⟩ :=
    L32_triple hl (c + 1) (by omega) a3 ha3rip ⟨ha3r2, hrdsome⟩
  have ha4r0' : a4.reg 0 = some n := by
    rw [ha4r0, ha3mem]
    exact hrdm1
  have ha4mem' : a4.mem = m1 := by rw [ha4mem]; exact ha3mem
  have ha4r4 : a4.reg 4 = some (sp - 8) := by rw [ha4other 4 (by decide)]; exact ha3r4
  have ha4r8 : a4.reg 8 = some h.d := by rw [ha4other 8 (by decide)]; exact ha3r8
  have ha4r9 : a4.reg 9 = some h.a := by rw [ha4other 9 (by decide)]; exact ha3r9
  have ha4r6 : a4.reg 6 = some c := by rw [ha4other 6 (by decide)]; exact ha3r6
  -- ret from L32
  have hrd64 : rd64 a4.mem (sp - 8) = some (entry + 12) := by
    have h1 := rd64_wr64 hm1s
    rw [Nat.mod_eq_of_lt (by omega)] at h1
    rwa [ha4mem']
  have e5 : X86Int.step im a4 = .ok { a4.set 4 (some ((sp - 8 + 8) % W)) with
      rip := entry + 12 } :=
    step_ret (by rw [ha4rip]; exact hc.dret) ha4r4 hrd64
  obtain ⟨a5, ha5⟩ : ∃ u : X86Int.St, u = { a4.set 4 (some ((sp - 8 + 8) % W)) with
    rip := entry + 12 } := ⟨_, rfl⟩
  have ha5rip : a5.rip = entry + 12 := by rw [ha5]
  have ha5r4 : a5.reg 4 = some sp := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 4 = some sp
    rw [reg_with_rip, reg_set_self (len_of_r9 ha4r9 (by decide)), slot_add8 hsp8 hspW]
  have ha5r0 : a5.reg 0 = some n := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 0 = some n
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha4r0'
  have ha5r8 : a5.reg 8 = some h.d := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 8 = some h.d
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha4r8
  have ha5r9 : a5.reg 9 = some h.a := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 9 = some h.a
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha4r9
  have ha5r6 : a5.reg 6 = some c := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha4r6
  have ha5mem : a5.mem = m1 := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).mem = m1
    rw [mem_with_rip, mem_set]; exact ha4mem'
  -- cmp rax, r8
  have e6 : X86Int.step im a5 = .ok { a5 with fl := (aluF .sub n h.d).2, rip := a5.rip + 3 } :=
    step_cmp (by rw [ha5rip]; exact hc.d3) ha5r0 ha5r8
  obtain ⟨a6, ha6⟩ : ∃ u : X86Int.St, u = { a5 with fl := (aluF .sub n h.d).2, rip := a5.rip + 3 } := ⟨_, rfl⟩
  have ha6rip : a6.rip = entry + 15 := by rw [ha6]; show a5.rip + 3 = entry + 15; rw [ha5rip]
  have ha6fl : X86Int.cond a6.fl 2 = some true := by
    have hcf : a6.fl.cf = some true := by
      rw [ha6]
      show (aluF .sub n h.d).2.cf = some true
      exact aluF_sub_cf_true hn
    exact cond_cc2 hcf
  have ha6r8 : a6.reg 8 = some h.d := by rw [ha6]; show a5.reg 8 = some h.d; exact ha5r8
  have ha6r9 : a6.reg 9 = some h.a := by rw [ha6]; show a5.reg 9 = some h.a; exact ha5r9
  have ha6r6 : a6.reg 6 = some c := by rw [ha6]; show a5.reg 6 = some c; exact ha5r6
  have ha6r4 : a6.reg 4 = some sp := by rw [ha6]; show a5.reg 4 = some sp; exact ha5r4
  have ha6mem : a6.mem = m1 := by rw [ha6]; show a5.mem = m1; exact ha5mem
  -- jb taken
  have e7 : X86Int.step im a6 = .ok { a6 with rip := entry + 22 } := by
    have hh := step_jcc (im := im) (s := a6) (by rw [ha6rip]; exact hc.d4) ha6fl
    simpa using hh
  obtain ⟨a7, ha7⟩ : ∃ u : X86Int.St, u = { a6 with rip := entry + 22 } := ⟨_, rfl⟩
  have ha7rip : a7.rip = entry + 22 := by rw [ha7]
  have ha7r8 : a7.reg 8 = some h.d := by
    rw [ha7]; show ({ a6 with rip := entry + 22 }).reg 8 = some h.d
    rw [reg_with_rip]; exact ha6r8
  have ha7r9 : a7.reg 9 = some h.a := by
    rw [ha7]; show ({ a6 with rip := entry + 22 }).reg 9 = some h.a
    rw [reg_with_rip]; exact ha6r9
  have ha7r6 : a7.reg 6 = some c := by
    rw [ha7]; show ({ a6 with rip := entry + 22 }).reg 6 = some c
    rw [reg_with_rip]; exact ha6r6
  have ha7r4 : a7.reg 4 = some sp := by
    rw [ha7]; show ({ a6 with rip := entry + 22 }).reg 4 = some sp
    rw [reg_with_rip]; exact ha6r4
  have ha7mem : a7.mem = m1 := by
    rw [ha7]; show ({ a6 with rip := entry + 22 }).mem = m1
    rw [mem_with_rip]; exact ha6mem
  -- add r8, 1
  have hadd8 : (aluF .add h.d (1 % W)).1 = h.d + 1 := aluI_add_val h1W hdW
  have e8 : X86Int.step im a7 = .ok { a7.set 8 (some (aluF .add h.d (1 % W)).1) with
      fl := (aluF .add h.d (1 % W)).2, rip := a7.rip + 4 } :=
    step_aluI_add (by rw [ha7rip]; exact hc.d5) ha7r8
  obtain ⟨a8, ha8⟩ : ∃ u : X86Int.St, u = { a7.set 8 (some (aluF .add h.d (1 % W)).1) with
    fl := (aluF .add h.d (1 % W)).2, rip := a7.rip + 4 } := ⟨_, rfl⟩
  have ha8rip : a8.rip = entry + 26 := by rw [ha8]; show a7.rip + 4 = entry + 26; rw [ha7rip]
  have ha8r8 : a8.reg 8 = some (h.d + 1) := by
    rw [ha8]
    show (a7.set 8 (some (aluF .add h.d (1 % W)).1)).reg 8 = some (h.d + 1)
    rw [reg_set_self (len_of_r9 ha7r9 (by decide)), hadd8]
  have ha8r9 : a8.reg 9 = some h.a := by
    rw [ha8]
    show (a7.set 8 (some (aluF .add h.d (1 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha7r9
  have ha8r6 : a8.reg 6 = some c := by
    rw [ha8]
    show (a7.set 8 (some (aluF .add h.d (1 % W)).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha7r6
  have ha8r4 : a8.reg 4 = some sp := by
    rw [ha8]
    show (a7.set 8 (some (aluF .add h.d (1 % W)).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact ha7r4
  have ha8mem : a8.mem = m1 := by
    rw [ha8]
    show (a7.set 8 (some (aluF .add h.d (1 % W)).1)).mem = m1
    rw [mem_set]; exact ha7mem
  -- add rsi, 5
  have hadd6 : (aluF .add c (5 % W)).1 = c + 5 := aluI_add_val h5W hcW
  have e9 : X86Int.step im a8 = .ok { a8.set 6 (some (aluF .add c (5 % W)).1) with
      fl := (aluF .add c (5 % W)).2, rip := a8.rip + 4 } :=
    step_aluI_add (by rw [ha8rip]; exact hc.d6) ha8r6
  obtain ⟨a9, ha9⟩ : ∃ u : X86Int.St, u = { a8.set 6 (some (aluF .add c (5 % W)).1) with
    fl := (aluF .add c (5 % W)).2, rip := a8.rip + 4 } := ⟨_, rfl⟩
  have ha9rip : a9.rip = entry + 30 := by rw [ha9]; show a8.rip + 4 = entry + 30; rw [ha8rip]
  have ha9r6 : a9.reg 6 = some (c + 5) := by
    rw [ha9]
    show (a8.set 6 (some (aluF .add c (5 % W)).1)).reg 6 = some (c + 5)
    rw [reg_set_self (len_of_r9 ha8r9 (by decide)), hadd6]
  have ha9r8 : a9.reg 8 = some (h.d + 1) := by
    rw [ha9]
    show (a8.set 6 (some (aluF .add c (5 % W)).1)).reg 8 = some (h.d + 1)
    rw [reg_set_other (by decide)]; exact ha8r8
  have ha9r9 : a9.reg 9 = some h.a := by
    rw [ha9]
    show (a8.set 6 (some (aluF .add c (5 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha8r9
  have ha9r4 : a9.reg 4 = some sp := by
    rw [ha9]
    show (a8.set 6 (some (aluF .add c (5 % W)).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact ha8r4
  have ha9mem : a9.mem = m1 := by
    rw [ha9]
    show (a8.set 6 (some (aluF .add c (5 % W)).1)).mem = m1
    rw [mem_set]; exact ha8mem
  -- jmp tail
  have e10 : X86Int.step im a9 = .ok { a9 with rip := tail } :=
    step_jmp (by rw [ha9rip]; exact hc.d7)
  obtain ⟨a10, ha10⟩ : ∃ u : X86Int.St, u = { a9 with rip := tail } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
  have r5 : run im 1 (Out.ok a4) = Out.ok a5 := by rw [run_succ_ok im 0 a4, e5, ← ha5, run_zero]
  have r6 : run im 1 (Out.ok a5) = Out.ok a6 := by rw [run_succ_ok im 0 a5, e6, ← ha6, run_zero]
  have r7 : run im 1 (Out.ok a6) = Out.ok a7 := by rw [run_succ_ok im 0 a6, e7, ← ha7, run_zero]
  have r8 : run im 1 (Out.ok a7) = Out.ok a8 := by rw [run_succ_ok im 0 a7, e8, ← ha8, run_zero]
  have r9 : run im 1 (Out.ok a8) = Out.ok a9 := by rw [run_succ_ok im 0 a8, e9, ← ha9, run_zero]
  have r10 : run im 1 (Out.ok a9) = Out.ok a10 := by rw [run_succ_ok im 0 a9, e10, ← ha10, run_zero]
  refine ⟨3 + n8 + 6, a10, ?_, by rw [ha10], ?_, ?_, ?_, ?_, ?_⟩
  · rw [run_add im (3 + n8) 6,
      show run im (3 + n8) (Out.ok s) = Out.ok a4 from by
        rw [run_add im 3 n8,
          show run im 3 (Out.ok s) = Out.ok a3 from by
            rw [run_add im 1 2, r1, run_add im 1 1, r2, r3],
          hrun8],
      run_add im 1 5, r5, run_add im 1 4, r6, run_add im 1 3, r7, run_add im 1 2, r8,
      run_add im 1 1, r9, r10]
  · rw [ha10]
    show ({ a9 with rip := tail }).reg 8 = some (h.d + 1)
    rw [reg_with_rip]; exact ha9r8
  · rw [ha10]
    show ({ a9 with rip := tail }).reg 9 = some h.a
    rw [reg_with_rip]; exact ha9r9
  · rw [ha10]
    show ({ a9 with rip := tail }).reg 6 = some (c + 5)
    rw [reg_with_rip]; exact ha9r6
  · rw [ha10]
    show ({ a9 with rip := tail }).reg 4 = some sp
    rw [reg_with_rip]; exact ha9r4
  · intro x m hx
    have hm10 : a10.mem = m1 := by
      rw [ha10]
      show ({ a9 with rip := tail }).mem = m1
      rw [mem_with_rip]; exact ha9mem
    rw [hm10]
    exact rdN_wr64_other hm1s (by cases hx with
      | inl h => exact Or.inr (by omega)
      | inr h => exact Or.inl (by omega))


/-- The `locget` handler's spec: the immediate at cursor+1 reads as
`n`, below the word's `locals` count at `w + 16`, and the handler
reaches the tail with `d+1`. The stack word is disjoint from the
immediate bytes and the locals field. Parametric throughout. -/
def LocgetSpec (im : Nat → Option (Ins × Nat)) (entry tail l32 : Nat) (wd : Word) : Prop :=
  ∀ (h : Height) (c n sp w : Nat), n < wd.locals → h.d + 1 < W → c + 5 < W →
    w + 24 < W → 8 ≤ sp → sp < W →
    (sp ≤ c + 1 ∨ c + 5 ≤ sp - 8) → (sp ≤ w + 16 ∨ w + 24 ≤ sp - 8) →
    ∀ s, s.rip = entry → s.reg 8 = some h.d → s.reg 9 = some h.a → s.reg 6 = some c →
      s.reg 4 = some sp → s.reg 7 = some w → 16 ≤ s.regs.length →
      (s.mem.rdN (c + 1) 4).map leVal = some n →
      rd64 s.mem (w + 16) = some wd.locals →
      (∀ v, (wr64 s.mem (sp - 8) v).isSome) →
      ∃ k s', run im k (Out.ok s) = Out.ok s' ∧ s'.rip = tail ∧
        s'.reg 8 = some (h.d + 1) ∧ s'.reg 9 = some h.a ∧ s'.reg 6 = some (c + 5) ∧
        s'.reg 4 = some sp ∧
        (∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → s'.reg i = s.reg i) ∧
        (∀ x m, x + m ≤ sp - 8 ∨ sp ≤ x → s'.mem.rdN x m = s.mem.rdN x m)

/-- Decode facts for `h_locget`. -/
structure LocgetCode (im : Nat → Option (Ins × Nat)) (entry tail l32 : Nat) : Prop where
  d0 : im entry = some (.mov 2 6, 3)
  d1 : im (entry + 3) = some (.aluI .add 2 1, 4)
  d2 : im (entry + 7) = some (.call l32, 5)
  d3 : im (entry + 12) = some (.ld 10 7 16, 4)
  d4 : im (entry + 16) = some (.cmp 0 10, 3)
  d5 : im (entry + 19) = some (.jcc 2 (entry + 26), 2)
  d6 : im (entry + 26) = some (.aluI .add 8 1, 4)
  d7 : im (entry + 30) = some (.aluI .add 6 5, 4)
  d8 : im (entry + 34) = some (.jmp tail, 5)
  dret : im (l32 + 2) = some (.ret, 1)

theorem Locget_triple {im : Nat → Option (Ins × Nat)} {entry tail l32 : Nat} {wd : Word}
    (hc : LocgetCode im entry tail l32) (hl : L32Code im l32) (hentry : entry + 39 < W) :
    LocgetSpec im entry tail l32 wd := by
  intro h c n sp w hn hdW hcW hwW hsp8 hspW hdisjc hdisjw s hrip hr8 hr9 hr6 hr4 hr7 hlen hrd hloc hw
  have h1W : 1 < W := by unfold W; decide
  have h5W : 5 < W := by unfold W; decide
  have e1 : X86Int.step im s = .ok { s.set 2 (some c) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact hc.d0) hr6
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s.set 2 (some c) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = entry + 3 := by rw [ha1]; show s.rip + 3 = entry + 3; rw [hrip]
  have ha1r2 : a1.reg 2 = some c := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 2 = some c
    rw [reg_with_rip, reg_set_self (len_of_r9 hr9 (by decide))]
  have ha1r8 : a1.reg 8 = some h.d := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 8 = some h.d
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr8
  have ha1r9 : a1.reg 9 = some h.a := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 9 = some h.a
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr9
  have ha1r6 : a1.reg 6 = some c := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr6
  have ha1r4 : a1.reg 4 = some sp := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 4 = some sp
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr4
  have ha1r7 : a1.reg 7 = some w := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 7 = some w
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr7
  have ha1mem : a1.mem = s.mem := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ga1 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a1.reg i = s.reg i := by
    intro i _ h2 _ _ _ _
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other h2]
  have hadd2 : (aluF .add c (1 % W)).1 = c + 1 := aluI_add_val h1W (by omega)
  have e2 : X86Int.step im a1 = .ok { a1.set 2 (some (aluF .add c (1 % W)).1) with
      fl := (aluF .add c (1 % W)).2, rip := a1.rip + 4 } :=
    step_aluI_add (by rw [ha1rip]; exact hc.d1) ha1r2
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1.set 2 (some (aluF .add c (1 % W)).1) with
    fl := (aluF .add c (1 % W)).2, rip := a1.rip + 4 } := ⟨_, rfl⟩
  have ha2rip : a2.rip = entry + 7 := by rw [ha2]; show a1.rip + 4 = entry + 7; rw [ha1rip]
  have ha2r2 : a2.reg 2 = some (c + 1) := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 2 = some (c + 1)
    rw [reg_set_self (len_of_r9 ha1r9 (by decide)), hadd2]
  have ha2r8 : a2.reg 8 = some h.d := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact ha1r8
  have ha2r9 : a2.reg 9 = some h.a := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha1r9
  have ha2r6 : a2.reg 6 = some c := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha1r6
  have ha2r4 : a2.reg 4 = some sp := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact ha1r4
  have ha2r7 : a2.reg 7 = some w := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 7 = some w
    rw [reg_set_other (by decide)]; exact ha1r7
  have ha2mem : a2.mem = s.mem := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).mem = s.mem
    rw [mem_set]; exact ha1mem
  have ga2 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a2.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg i = s.reg i
    rw [reg_set_other h2]; exact ga1 i h0 h2 h4 h6 h8 h10
  have hslot8 : (sp + W - 8) % W = sp - 8 := by
    rw [show sp + W - 8 = sp + W - 8 * 1 by omega, slot_sub (by omega) hspW]
  obtain ⟨m1, hm1s⟩ := Option.isSome_iff_exists.mp (hw (entry + 12))
  have hm1 : wr64 a2.mem ((sp + W - 8) % W) (a2.rip + 5) = some m1 := by
    rw [ha2mem, ha2rip, hslot8]
    exact hm1s
  have e3 : X86Int.step im a2 = .ok { a2.set 4 (some ((sp + W - 8) % W)) with
      mem := m1, rip := l32 } :=
    step_call (by rw [ha2rip]; exact hc.d2) ha2r4 hm1
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2.set 4 (some ((sp + W - 8) % W)) with
    mem := m1, rip := l32 } := ⟨_, rfl⟩
  have ha3rip : a3.rip = l32 := by rw [ha3]
  have ha3r4 : a3.reg 4 = some (sp - 8) := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
    rw [reg_set_self (len_of_r9 ha2r9 (by decide)), hslot8]
  have ha3r2 : a3.reg 2 = some (c + 1) := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 2 = some (c + 1)
    rw [reg_set_other (by decide)]; exact ha2r2
  have ha3r8 : a3.reg 8 = some h.d := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact ha2r8
  have ha3r9 : a3.reg 9 = some h.a := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha2r9
  have ha3r6 : a3.reg 6 = some c := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha2r6
  have ha3r7 : a3.reg 7 = some w := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 7 = some w
    rw [reg_set_other (by decide)]; exact ha2r7
  have ha3mem : a3.mem = m1 := by rw [ha3]
  have ga3 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a3.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg i = s.reg i
    rw [reg_set_other h4]; exact ga2 i h0 h2 h4 h6 h8 h10
  have hfr1 : m1.rdN (c + 1) 4 = s.mem.rdN (c + 1) 4 :=
    rdN_wr64_other hm1s (by cases hdisjc with
      | inl h => exact Or.inl (by omega)
      | inr h => exact Or.inr (by omega))
  have hloc1 : rd64 m1 (w + 16) = some wd.locals := by
    rw [rd64_wr64_other hm1s (by cases hdisjw with
      | inl h => exact Or.inl (by omega)
      | inr h => exact Or.inr (by omega))]
    exact hloc
  have hrdm1 : (m1.rdN (c + 1) 4).map leVal = some n := by
    rw [hfr1]
    exact hrd
  have hrdsome : (a3.mem.rdN (c + 1) 4).isSome := by
    rw [ha3mem]
    cases h : m1.rdN (c + 1) 4 with
    | none => simp [h] at hrdm1
    | some bs => rfl
  obtain ⟨n8, a4, hrun8, ha4rip, ha4r0, -, ha4mem, ha4other, ha4len⟩ :=
    L32_triple hl (c + 1) (by omega) a3 ha3rip ⟨ha3r2, hrdsome⟩
  have ha4r0' : a4.reg 0 = some n := by
    rw [ha4r0, ha3mem]
    exact hrdm1
  have ha4mem' : a4.mem = m1 := by rw [ha4mem]; exact ha3mem
  have ha4r4 : a4.reg 4 = some (sp - 8) := by rw [ha4other 4 (by decide)]; exact ha3r4
  have ha4r8 : a4.reg 8 = some h.d := by rw [ha4other 8 (by decide)]; exact ha3r8
  have ha4r9 : a4.reg 9 = some h.a := by rw [ha4other 9 (by decide)]; exact ha3r9
  have ha4r6 : a4.reg 6 = some c := by rw [ha4other 6 (by decide)]; exact ha3r6
  have ha4r7 : a4.reg 7 = some w := by rw [ha4other 7 (by decide)]; exact ha3r7
  have ga4 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a4.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10
    rw [ha4other i h0]; exact ga3 i h0 h2 h4 h6 h8 h10
  have hrd64 : rd64 a4.mem (sp - 8) = some (entry + 12) := by
    have h1 := rd64_wr64 hm1s
    rw [Nat.mod_eq_of_lt (by omega)] at h1
    rwa [ha4mem']
  have e5 : X86Int.step im a4 = .ok { a4.set 4 (some ((sp - 8 + 8) % W)) with
      rip := entry + 12 } :=
    step_ret (by rw [ha4rip]; exact hc.dret) ha4r4 hrd64
  obtain ⟨a5, ha5⟩ : ∃ u : X86Int.St, u = { a4.set 4 (some ((sp - 8 + 8) % W)) with
    rip := entry + 12 } := ⟨_, rfl⟩
  have ha5rip : a5.rip = entry + 12 := by rw [ha5]
  have ha5r4 : a5.reg 4 = some sp := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 4 = some sp
    rw [reg_with_rip, reg_set_self (len_of_r9 ha4r9 (by decide)), slot_add8 hsp8 hspW]
  have ha5r0 : a5.reg 0 = some n := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 0 = some n
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha4r0'
  have ha5r8 : a5.reg 8 = some h.d := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 8 = some h.d
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha4r8
  have ha5r9 : a5.reg 9 = some h.a := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 9 = some h.a
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha4r9
  have ha5r6 : a5.reg 6 = some c := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha4r6
  have ha5r7 : a5.reg 7 = some w := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 7 = some w
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha4r7
  have ha5mem : a5.mem = m1 := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).mem = m1
    rw [mem_with_rip, mem_set]; exact ha4mem'
  have ga5 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a5.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other h4]; exact ga4 i h0 h2 h4 h6 h8 h10
  -- mov r10, [rdi+16]
  have hea16 : ea w 16 = w + 16 := Nat.mod_eq_of_lt (by omega)
  have e6 : X86Int.step im a5 = .ok { a5.set 10 (some wd.locals) with rip := a5.rip + 4 } :=
    step_ld (by rw [ha5rip]; exact hc.d3) ha5r7 (by rw [ha5mem, hea16]; exact hloc1)
  obtain ⟨a6, ha6⟩ : ∃ u : X86Int.St, u = { a5.set 10 (some wd.locals) with rip := a5.rip + 4 } := ⟨_, rfl⟩
  have ha6rip : a6.rip = entry + 16 := by rw [ha6]; show a5.rip + 4 = entry + 16; rw [ha5rip]
  have hlen3 : a3.regs.length = s.regs.length := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).regs.length = s.regs.length
    rw [len_set, ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).regs.length = s.regs.length
    rw [len_set, ha1]
    show ({ s.set 2 (some c) with rip := s.rip + 3 }).regs.length = s.regs.length
    rw [len_with_rip, len_set]
  have hlen5 : a5.regs.length = s.regs.length := by
    rw [ha5]
    show (a4.set 4 (some ((sp - 8 + 8) % W))).regs.length = s.regs.length
    rw [len_set, ha4len, hlen3]
  have ha6r10 : a6.reg 10 = some wd.locals := by
    rw [ha6]
    show ({ a5.set 10 (some wd.locals) with rip := a5.rip + 4 }).reg 10 = some wd.locals
    rw [reg_with_rip, reg_set_self (by omega)]
  have ha6r0 : a6.reg 0 = some n := by
    rw [ha6]
    show ({ a5.set 10 (some wd.locals) with rip := a5.rip + 4 }).reg 0 = some n
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha5r0
  have ha6r8 : a6.reg 8 = some h.d := by
    rw [ha6]
    show ({ a5.set 10 (some wd.locals) with rip := a5.rip + 4 }).reg 8 = some h.d
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha5r8
  have ha6r9 : a6.reg 9 = some h.a := by
    rw [ha6]
    show ({ a5.set 10 (some wd.locals) with rip := a5.rip + 4 }).reg 9 = some h.a
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha5r9
  have ha6r6 : a6.reg 6 = some c := by
    rw [ha6]
    show ({ a5.set 10 (some wd.locals) with rip := a5.rip + 4 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha5r6
  have ha6r4 : a6.reg 4 = some sp := by
    rw [ha6]
    show ({ a5.set 10 (some wd.locals) with rip := a5.rip + 4 }).reg 4 = some sp
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha5r4
  have ha6mem : a6.mem = m1 := by
    rw [ha6]
    show ({ a5.set 10 (some wd.locals) with rip := a5.rip + 4 }).mem = m1
    rw [mem_with_rip, mem_set]; exact ha5mem
  have ga6 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a6.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10
    rw [ha6]
    show ({ a5.set 10 (some wd.locals) with rip := a5.rip + 4 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other h10]; exact ga5 i h0 h2 h4 h6 h8 h10
  -- cmp rax, r10
  have e7 : X86Int.step im a6 = .ok { a6 with fl := (aluF .sub n wd.locals).2, rip := a6.rip + 3 } :=
    step_cmp (by rw [ha6rip]; exact hc.d4) ha6r0 ha6r10
  obtain ⟨a7, ha7⟩ : ∃ u : X86Int.St, u = { a6 with fl := (aluF .sub n wd.locals).2, rip := a6.rip + 3 } := ⟨_, rfl⟩
  have ha7rip : a7.rip = entry + 19 := by rw [ha7]; show a6.rip + 3 = entry + 19; rw [ha6rip]
  have ha7fl : X86Int.cond a7.fl 2 = some true := by
    have hcf : a7.fl.cf = some true := by
      rw [ha7]
      show (aluF .sub n wd.locals).2.cf = some true
      exact aluF_sub_cf_true hn
    exact cond_cc2 hcf
  have ha7r8 : a7.reg 8 = some h.d := by rw [ha7]; show a6.reg 8 = some h.d; exact ha6r8
  have ha7r9 : a7.reg 9 = some h.a := by rw [ha7]; show a6.reg 9 = some h.a; exact ha6r9
  have ha7r6 : a7.reg 6 = some c := by rw [ha7]; show a6.reg 6 = some c; exact ha6r6
  have ha7r4 : a7.reg 4 = some sp := by rw [ha7]; show a6.reg 4 = some sp; exact ha6r4
  have ha7mem : a7.mem = m1 := by rw [ha7]; show a6.mem = m1; exact ha6mem
  have ga7 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a7.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10
    rw [ha7]
    show a6.reg i = s.reg i
    exact ga6 i h0 h2 h4 h6 h8 h10
  -- jb taken
  have e8 : X86Int.step im a7 = .ok { a7 with rip := entry + 26 } := by
    have hh := step_jcc (im := im) (s := a7) (by rw [ha7rip]; exact hc.d5) ha7fl
    simpa using hh
  obtain ⟨a8, ha8⟩ : ∃ u : X86Int.St, u = { a7 with rip := entry + 26 } := ⟨_, rfl⟩
  have ha8rip : a8.rip = entry + 26 := by rw [ha8]
  have ha8r8 : a8.reg 8 = some h.d := by
    rw [ha8]; show ({ a7 with rip := entry + 26 }).reg 8 = some h.d
    rw [reg_with_rip]; exact ha7r8
  have ha8r9 : a8.reg 9 = some h.a := by
    rw [ha8]; show ({ a7 with rip := entry + 26 }).reg 9 = some h.a
    rw [reg_with_rip]; exact ha7r9
  have ha8r6 : a8.reg 6 = some c := by
    rw [ha8]; show ({ a7 with rip := entry + 26 }).reg 6 = some c
    rw [reg_with_rip]; exact ha7r6
  have ha8r4 : a8.reg 4 = some sp := by
    rw [ha8]; show ({ a7 with rip := entry + 26 }).reg 4 = some sp
    rw [reg_with_rip]; exact ha7r4
  have ha8mem : a8.mem = m1 := by
    rw [ha8]; show ({ a7 with rip := entry + 26 }).mem = m1
    rw [mem_with_rip]; exact ha7mem
  have ga8 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a8.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10
    rw [ha8]; show ({ a7 with rip := entry + 26 }).reg i = s.reg i
    rw [reg_with_rip]; exact ga7 i h0 h2 h4 h6 h8 h10
  -- add r8, 1
  have hadd8 : (aluF .add h.d (1 % W)).1 = h.d + 1 := aluI_add_val h1W hdW
  have e9 : X86Int.step im a8 = .ok { a8.set 8 (some (aluF .add h.d (1 % W)).1) with
      fl := (aluF .add h.d (1 % W)).2, rip := a8.rip + 4 } :=
    step_aluI_add (by rw [ha8rip]; exact hc.d6) ha8r8
  obtain ⟨a9, ha9⟩ : ∃ u : X86Int.St, u = { a8.set 8 (some (aluF .add h.d (1 % W)).1) with
    fl := (aluF .add h.d (1 % W)).2, rip := a8.rip + 4 } := ⟨_, rfl⟩
  have ha9rip : a9.rip = entry + 30 := by rw [ha9]; show a8.rip + 4 = entry + 30; rw [ha8rip]
  have ha9r8 : a9.reg 8 = some (h.d + 1) := by
    rw [ha9]
    show (a8.set 8 (some (aluF .add h.d (1 % W)).1)).reg 8 = some (h.d + 1)
    rw [reg_set_self (len_of_r9 ha8r9 (by decide)), hadd8]
  have ha9r9 : a9.reg 9 = some h.a := by
    rw [ha9]
    show (a8.set 8 (some (aluF .add h.d (1 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha8r9
  have ha9r6 : a9.reg 6 = some c := by
    rw [ha9]
    show (a8.set 8 (some (aluF .add h.d (1 % W)).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha8r6
  have ha9r4 : a9.reg 4 = some sp := by
    rw [ha9]
    show (a8.set 8 (some (aluF .add h.d (1 % W)).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact ha8r4
  have ha9mem : a9.mem = m1 := by
    rw [ha9]
    show (a8.set 8 (some (aluF .add h.d (1 % W)).1)).mem = m1
    rw [mem_set]; exact ha8mem
  have ga9 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a9.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10
    rw [ha9]
    show (a8.set 8 (some (aluF .add h.d (1 % W)).1)).reg i = s.reg i
    rw [reg_set_other h8]; exact ga8 i h0 h2 h4 h6 h8 h10
  -- add rsi, 5
  have hadd6 : (aluF .add c (5 % W)).1 = c + 5 := aluI_add_val h5W hcW
  have e10 : X86Int.step im a9 = .ok { a9.set 6 (some (aluF .add c (5 % W)).1) with
      fl := (aluF .add c (5 % W)).2, rip := a9.rip + 4 } :=
    step_aluI_add (by rw [ha9rip]; exact hc.d7) ha9r6
  obtain ⟨a10, ha10⟩ : ∃ u : X86Int.St, u = { a9.set 6 (some (aluF .add c (5 % W)).1) with
    fl := (aluF .add c (5 % W)).2, rip := a9.rip + 4 } := ⟨_, rfl⟩
  have ha10rip : a10.rip = entry + 34 := by rw [ha10]; show a9.rip + 4 = entry + 34; rw [ha9rip]
  have ha10r6 : a10.reg 6 = some (c + 5) := by
    rw [ha10]
    show (a9.set 6 (some (aluF .add c (5 % W)).1)).reg 6 = some (c + 5)
    rw [reg_set_self (len_of_r9 ha9r9 (by decide)), hadd6]
  have ha10r8 : a10.reg 8 = some (h.d + 1) := by
    rw [ha10]
    show (a9.set 6 (some (aluF .add c (5 % W)).1)).reg 8 = some (h.d + 1)
    rw [reg_set_other (by decide)]; exact ha9r8
  have ha10r9 : a10.reg 9 = some h.a := by
    rw [ha10]
    show (a9.set 6 (some (aluF .add c (5 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha9r9
  have ha10r4 : a10.reg 4 = some sp := by
    rw [ha10]
    show (a9.set 6 (some (aluF .add c (5 % W)).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact ha9r4
  have ha10mem : a10.mem = m1 := by
    rw [ha10]
    show (a9.set 6 (some (aluF .add c (5 % W)).1)).mem = m1
    rw [mem_set]; exact ha9mem
  have ga10 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a10.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10
    rw [ha10]
    show (a9.set 6 (some (aluF .add c (5 % W)).1)).reg i = s.reg i
    rw [reg_set_other h6]; exact ga9 i h0 h2 h4 h6 h8 h10
  -- jmp tail
  have e11 : X86Int.step im a10 = .ok { a10 with rip := tail } :=
    step_jmp (by rw [ha10rip]; exact hc.d8)
  obtain ⟨a11, ha11⟩ : ∃ u : X86Int.St, u = { a10 with rip := tail } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
  have r5 : run im 1 (Out.ok a4) = Out.ok a5 := by rw [run_succ_ok im 0 a4, e5, ← ha5, run_zero]
  have r6 : run im 1 (Out.ok a5) = Out.ok a6 := by rw [run_succ_ok im 0 a5, e6, ← ha6, run_zero]
  have r7 : run im 1 (Out.ok a6) = Out.ok a7 := by rw [run_succ_ok im 0 a6, e7, ← ha7, run_zero]
  have r8 : run im 1 (Out.ok a7) = Out.ok a8 := by rw [run_succ_ok im 0 a7, e8, ← ha8, run_zero]
  have r9 : run im 1 (Out.ok a8) = Out.ok a9 := by rw [run_succ_ok im 0 a8, e9, ← ha9, run_zero]
  have r10 : run im 1 (Out.ok a9) = Out.ok a10 := by rw [run_succ_ok im 0 a9, e10, ← ha10, run_zero]
  have r11 : run im 1 (Out.ok a10) = Out.ok a11 := by rw [run_succ_ok im 0 a10, e11, ← ha11, run_zero]
  refine ⟨3 + n8 + 7, a11, ?_, by rw [ha11], ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [run_add im (3 + n8) 7,
      show run im (3 + n8) (Out.ok s) = Out.ok a4 from by
        rw [run_add im 3 n8,
          show run im 3 (Out.ok s) = Out.ok a3 from by
            rw [run_add im 1 2, r1, run_add im 1 1, r2, r3],
          hrun8],
      run_add im 1 6, r5, run_add im 1 5, r6, run_add im 1 4, r7, run_add im 1 3, r8,
      run_add im 1 2, r9, run_add im 1 1, r10, r11]
  · rw [ha11]
    show ({ a10 with rip := tail }).reg 8 = some (h.d + 1)
    rw [reg_with_rip]; exact ha10r8
  · rw [ha11]
    show ({ a10 with rip := tail }).reg 9 = some h.a
    rw [reg_with_rip]; exact ha10r9
  · rw [ha11]
    show ({ a10 with rip := tail }).reg 6 = some (c + 5)
    rw [reg_with_rip]; exact ha10r6
  · rw [ha11]
    show ({ a10 with rip := tail }).reg 4 = some sp
    rw [reg_with_rip]; exact ha10r4
  · intro i h0 h2 h4 h6 h8 h10
    rw [ha11]
    show ({ a10 with rip := tail }).reg i = s.reg i
    rw [reg_with_rip]; exact ga10 i h0 h2 h4 h6 h8 h10
  · intro x m hx
    have hm11 : a11.mem = m1 := by
      rw [ha11]
      show ({ a10 with rip := tail }).mem = m1
      rw [mem_with_rip]; exact ha10mem
    rw [hm11]
    exact rdN_wr64_other hm1s (by cases hx with
      | inl h => exact Or.inr (by omega)
      | inr h => exact Or.inl (by omega))


/-- The `locset` handler's spec: like `locget`, with the additional
value-to-store guard, leaving `d-1` at the tail. -/
def LocsetSpec (im : Nat → Option (Ins × Nat)) (entry tail l32 : Nat) (wd : Word) : Prop :=
  ∀ (h : Height) (c n sp w : Nat), 1 ≤ h.d → n < wd.locals → h.d < W → c + 5 < W →
    w + 24 < W → 8 ≤ sp → sp < W →
    (sp ≤ c + 1 ∨ c + 5 ≤ sp - 8) → (sp ≤ w + 16 ∨ w + 24 ≤ sp - 8) →
    ∀ s, s.rip = entry → s.reg 8 = some h.d → s.reg 9 = some h.a → s.reg 6 = some c →
      s.reg 4 = some sp → s.reg 7 = some w → 16 ≤ s.regs.length →
      (s.mem.rdN (c + 1) 4).map leVal = some n →
      rd64 s.mem (w + 16) = some wd.locals →
      (∀ v, (wr64 s.mem (sp - 8) v).isSome) →
      ∃ k s', run im k (Out.ok s) = Out.ok s' ∧ s'.rip = tail ∧
        s'.reg 8 = some (h.d - 1) ∧ s'.reg 9 = some h.a ∧ s'.reg 6 = some (c + 5) ∧
        s'.reg 4 = some sp ∧
        (∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → s'.reg i = s.reg i) ∧
        (∀ x m, x + m ≤ sp - 8 ∨ sp ≤ x → s'.mem.rdN x m = s.mem.rdN x m)

/-- Decode facts for `h_locset`. -/
structure LocsetCode (im : Nat → Option (Ins × Nat)) (entry tail l32 : Nat) : Prop where
  d0 : im entry = some (.mov 2 6, 3)
  d1 : im (entry + 3) = some (.aluI .add 2 1, 4)
  d2 : im (entry + 7) = some (.call l32, 5)
  d3 : im (entry + 12) = some (.cmpI 8 1, 4)
  d4 : im (entry + 16) = some (.jcc 3 (entry + 23), 2)
  d5 : im (entry + 23) = some (.ld 10 7 16, 4)
  d6 : im (entry + 27) = some (.cmp 0 10, 3)
  d7 : im (entry + 30) = some (.jcc 2 (entry + 37), 2)
  d8 : im (entry + 37) = some (.aluI .sub 8 1, 4)
  d9 : im (entry + 41) = some (.aluI .add 6 5, 4)
  d10 : im (entry + 45) = some (.jmp tail, 5)
  dret : im (l32 + 2) = some (.ret, 1)

theorem Locset_triple {im : Nat → Option (Ins × Nat)} {entry tail l32 : Nat} {wd : Word}
    (hc : LocsetCode im entry tail l32) (hl : L32Code im l32) (hentry : entry + 50 < W) :
    LocsetSpec im entry tail l32 wd := by
  intro h c n sp w hd1 hn hdW hcW hwW hsp8 hspW hdisjc hdisjw s hrip hr8 hr9 hr6 hr4 hr7 hlen hrd hloc hw
  have h1W : 1 < W := by unfold W; decide
  have h5W : 5 < W := by unfold W; decide
  have e1 : X86Int.step im s = .ok { s.set 2 (some c) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact hc.d0) hr6
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s.set 2 (some c) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = entry + 3 := by rw [ha1]; show s.rip + 3 = entry + 3; rw [hrip]
  have ha1r2 : a1.reg 2 = some c := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 2 = some c
    rw [reg_with_rip, reg_set_self (len_of_r9 hr9 (by decide))]
  have ha1r8 : a1.reg 8 = some h.d := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 8 = some h.d
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr8
  have ha1r9 : a1.reg 9 = some h.a := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 9 = some h.a
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr9
  have ha1r6 : a1.reg 6 = some c := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr6
  have ha1r4 : a1.reg 4 = some sp := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 4 = some sp
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr4
  have ha1r7 : a1.reg 7 = some w := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 7 = some w
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr7
  have ha1mem : a1.mem = s.mem := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ga1 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a1.reg i = s.reg i := by
    intro i _ h2 _ _ _ _
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other h2]
  have hadd2 : (aluF .add c (1 % W)).1 = c + 1 := aluI_add_val h1W (by omega)
  have e2 : X86Int.step im a1 = .ok { a1.set 2 (some (aluF .add c (1 % W)).1) with
      fl := (aluF .add c (1 % W)).2, rip := a1.rip + 4 } :=
    step_aluI_add (by rw [ha1rip]; exact hc.d1) ha1r2
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1.set 2 (some (aluF .add c (1 % W)).1) with
    fl := (aluF .add c (1 % W)).2, rip := a1.rip + 4 } := ⟨_, rfl⟩
  have ha2rip : a2.rip = entry + 7 := by rw [ha2]; show a1.rip + 4 = entry + 7; rw [ha1rip]
  have ha2r2 : a2.reg 2 = some (c + 1) := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 2 = some (c + 1)
    rw [reg_set_self (len_of_r9 ha1r9 (by decide)), hadd2]
  have ha2r8 : a2.reg 8 = some h.d := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact ha1r8
  have ha2r9 : a2.reg 9 = some h.a := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha1r9
  have ha2r6 : a2.reg 6 = some c := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha1r6
  have ha2r4 : a2.reg 4 = some sp := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact ha1r4
  have ha2r7 : a2.reg 7 = some w := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 7 = some w
    rw [reg_set_other (by decide)]; exact ha1r7
  have ha2mem : a2.mem = s.mem := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).mem = s.mem
    rw [mem_set]; exact ha1mem
  have ga2 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a2.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg i = s.reg i
    rw [reg_set_other h2]; exact ga1 i h0 h2 h4 h6 h8 h10
  have hslot8 : (sp + W - 8) % W = sp - 8 := by
    rw [show sp + W - 8 = sp + W - 8 * 1 by omega, slot_sub (by omega) hspW]
  obtain ⟨m1, hm1s⟩ := Option.isSome_iff_exists.mp (hw (entry + 12))
  have hm1 : wr64 a2.mem ((sp + W - 8) % W) (a2.rip + 5) = some m1 := by
    rw [ha2mem, ha2rip, hslot8]
    exact hm1s
  have e3 : X86Int.step im a2 = .ok { a2.set 4 (some ((sp + W - 8) % W)) with
      mem := m1, rip := l32 } :=
    step_call (by rw [ha2rip]; exact hc.d2) ha2r4 hm1
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2.set 4 (some ((sp + W - 8) % W)) with
    mem := m1, rip := l32 } := ⟨_, rfl⟩
  have ha3rip : a3.rip = l32 := by rw [ha3]
  have ha3r4 : a3.reg 4 = some (sp - 8) := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
    rw [reg_set_self (len_of_r9 ha2r9 (by decide)), hslot8]
  have ha3r2 : a3.reg 2 = some (c + 1) := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 2 = some (c + 1)
    rw [reg_set_other (by decide)]; exact ha2r2
  have ha3r8 : a3.reg 8 = some h.d := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact ha2r8
  have ha3r9 : a3.reg 9 = some h.a := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha2r9
  have ha3r6 : a3.reg 6 = some c := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha2r6
  have ha3r7 : a3.reg 7 = some w := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 7 = some w
    rw [reg_set_other (by decide)]; exact ha2r7
  have ha3mem : a3.mem = m1 := by rw [ha3]
  have ga3 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a3.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg i = s.reg i
    rw [reg_set_other h4]; exact ga2 i h0 h2 h4 h6 h8 h10
  have hfr1 : m1.rdN (c + 1) 4 = s.mem.rdN (c + 1) 4 :=
    rdN_wr64_other hm1s (by cases hdisjc with
      | inl h => exact Or.inl (by omega)
      | inr h => exact Or.inr (by omega))
  have hloc1 : rd64 m1 (w + 16) = some wd.locals := by
    rw [rd64_wr64_other hm1s (by cases hdisjw with
      | inl h => exact Or.inl (by omega)
      | inr h => exact Or.inr (by omega))]
    exact hloc
  have hrdm1 : (m1.rdN (c + 1) 4).map leVal = some n := by
    rw [hfr1]
    exact hrd
  have hrdsome : (a3.mem.rdN (c + 1) 4).isSome := by
    rw [ha3mem]
    cases h : m1.rdN (c + 1) 4 with
    | none => simp [h] at hrdm1
    | some bs => rfl
  obtain ⟨n8, a4, hrun8, ha4rip, ha4r0, -, ha4mem, ha4other, ha4len⟩ :=
    L32_triple hl (c + 1) (by omega) a3 ha3rip ⟨ha3r2, hrdsome⟩
  have ha4r0' : a4.reg 0 = some n := by
    rw [ha4r0, ha3mem]
    exact hrdm1
  have ha4mem' : a4.mem = m1 := by rw [ha4mem]; exact ha3mem
  have ha4r4 : a4.reg 4 = some (sp - 8) := by rw [ha4other 4 (by decide)]; exact ha3r4
  have ha4r8 : a4.reg 8 = some h.d := by rw [ha4other 8 (by decide)]; exact ha3r8
  have ha4r9 : a4.reg 9 = some h.a := by rw [ha4other 9 (by decide)]; exact ha3r9
  have ha4r6 : a4.reg 6 = some c := by rw [ha4other 6 (by decide)]; exact ha3r6
  have ha4r7 : a4.reg 7 = some w := by rw [ha4other 7 (by decide)]; exact ha3r7
  have ga4 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a4.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10
    rw [ha4other i h0]; exact ga3 i h0 h2 h4 h6 h8 h10
  have hrd64 : rd64 a4.mem (sp - 8) = some (entry + 12) := by
    have h1 := rd64_wr64 hm1s
    rw [Nat.mod_eq_of_lt (by omega)] at h1
    rwa [ha4mem']
  have e5 : X86Int.step im a4 = .ok { a4.set 4 (some ((sp - 8 + 8) % W)) with
      rip := entry + 12 } :=
    step_ret (by rw [ha4rip]; exact hc.dret) ha4r4 hrd64
  obtain ⟨a5, ha5⟩ : ∃ u : X86Int.St, u = { a4.set 4 (some ((sp - 8 + 8) % W)) with
    rip := entry + 12 } := ⟨_, rfl⟩
  have ha5rip : a5.rip = entry + 12 := by rw [ha5]
  have ha5r4 : a5.reg 4 = some sp := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 4 = some sp
    rw [reg_with_rip, reg_set_self (len_of_r9 ha4r9 (by decide)), slot_add8 hsp8 hspW]
  have ha5r0 : a5.reg 0 = some n := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 0 = some n
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha4r0'
  have ha5r8 : a5.reg 8 = some h.d := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 8 = some h.d
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha4r8
  have ha5r9 : a5.reg 9 = some h.a := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 9 = some h.a
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha4r9
  have ha5r6 : a5.reg 6 = some c := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha4r6
  have ha5r7 : a5.reg 7 = some w := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 7 = some w
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha4r7
  have ha5mem : a5.mem = m1 := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).mem = m1
    rw [mem_with_rip, mem_set]; exact ha4mem'
  have ga5 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a5.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other h4]; exact ga4 i h0 h2 h4 h6 h8 h10
  have hlen3 : a3.regs.length = s.regs.length := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).regs.length = s.regs.length
    rw [len_set, ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).regs.length = s.regs.length
    rw [len_set, ha1]
    show ({ s.set 2 (some c) with rip := s.rip + 3 }).regs.length = s.regs.length
    rw [len_with_rip, len_set]
  have hlen5 : a5.regs.length = s.regs.length := by
    rw [ha5]
    show (a4.set 4 (some ((sp - 8 + 8) % W))).regs.length = s.regs.length
    rw [len_set, ha4len, hlen3]
  -- cmp r8, 1
  have e6 : X86Int.step im a5 = .ok { a5 with fl := (aluF .sub h.d (1 % W)).2, rip := a5.rip + 4 } :=
    step_cmpI (by rw [ha5rip]; exact hc.d3) ha5r8
  obtain ⟨a6, ha6⟩ : ∃ u : X86Int.St, u = { a5 with fl := (aluF .sub h.d (1 % W)).2, rip := a5.rip + 4 } := ⟨_, rfl⟩
  have ha6rip : a6.rip = entry + 16 := by rw [ha6]; show a5.rip + 4 = entry + 16; rw [ha5rip]
  have ha6fl : X86Int.cond a6.fl 3 = some true := by
    have hcf : a6.fl.cf = some false := by
      rw [ha6]
      show (aluF .sub h.d (1 % W)).2.cf = some false
      exact cmpI_cf_false h1W (by omega)
    have := cond_cc3 hcf
    simpa using this
  have ha6r8 : a6.reg 8 = some h.d := by rw [ha6]; show a5.reg 8 = some h.d; exact ha5r8
  have ha6r9 : a6.reg 9 = some h.a := by rw [ha6]; show a5.reg 9 = some h.a; exact ha5r9
  have ha6r6 : a6.reg 6 = some c := by rw [ha6]; show a5.reg 6 = some c; exact ha5r6
  have ha6r4 : a6.reg 4 = some sp := by rw [ha6]; show a5.reg 4 = some sp; exact ha5r4
  have ha6r0 : a6.reg 0 = some n := by rw [ha6]; show a5.reg 0 = some n; exact ha5r0
  have ha6r7 : a6.reg 7 = some w := by rw [ha6]; show a5.reg 7 = some w; exact ha5r7
  have ha6mem : a6.mem = m1 := by rw [ha6]; show a5.mem = m1; exact ha5mem
  have ga6 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a6.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10
    rw [ha6]
    show a5.reg i = s.reg i
    exact ga5 i h0 h2 h4 h6 h8 h10
  have hlen6 : a6.regs.length = s.regs.length := by
    rw [ha6]; show a5.regs.length = s.regs.length; exact hlen5
  -- jae taken
  have e7 : X86Int.step im a6 = .ok { a6 with rip := entry + 23 } := by
    have hh := step_jcc (im := im) (s := a6) (by rw [ha6rip]; exact hc.d4) ha6fl
    simpa using hh
  obtain ⟨a7, ha7⟩ : ∃ u : X86Int.St, u = { a6 with rip := entry + 23 } := ⟨_, rfl⟩
  have ha7rip : a7.rip = entry + 23 := by rw [ha7]
  have ha7r8 : a7.reg 8 = some h.d := by
    rw [ha7]; show ({ a6 with rip := entry + 23 }).reg 8 = some h.d
    rw [reg_with_rip]; exact ha6r8
  have ha7r9 : a7.reg 9 = some h.a := by
    rw [ha7]; show ({ a6 with rip := entry + 23 }).reg 9 = some h.a
    rw [reg_with_rip]; exact ha6r9
  have ha7r6 : a7.reg 6 = some c := by
    rw [ha7]; show ({ a6 with rip := entry + 23 }).reg 6 = some c
    rw [reg_with_rip]; exact ha6r6
  have ha7r4 : a7.reg 4 = some sp := by
    rw [ha7]; show ({ a6 with rip := entry + 23 }).reg 4 = some sp
    rw [reg_with_rip]; exact ha6r4
  have ha7r0 : a7.reg 0 = some n := by
    rw [ha7]; show ({ a6 with rip := entry + 23 }).reg 0 = some n
    rw [reg_with_rip]; exact ha6r0
  have ha7r7 : a7.reg 7 = some w := by
    rw [ha7]; show ({ a6 with rip := entry + 23 }).reg 7 = some w
    rw [reg_with_rip]; exact ha6r7
  have ha7mem : a7.mem = m1 := by
    rw [ha7]; show ({ a6 with rip := entry + 23 }).mem = m1
    rw [mem_with_rip]; exact ha6mem
  have ga7 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a7.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10
    rw [ha7]; show ({ a6 with rip := entry + 23 }).reg i = s.reg i
    rw [reg_with_rip]; exact ga6 i h0 h2 h4 h6 h8 h10
  have hlen7 : a7.regs.length = s.regs.length := by
    rw [ha7]; show ({ a6 with rip := entry + 23 }).regs.length = s.regs.length
    rw [len_with_rip]; exact hlen6
  -- mov r10, [rdi+16]
  have hea16 : ea w 16 = w + 16 := Nat.mod_eq_of_lt (by omega)
  have e8 : X86Int.step im a7 = .ok { a7.set 10 (some wd.locals) with rip := a7.rip + 4 } :=
    step_ld (by rw [ha7rip]; exact hc.d5) ha7r7 (by rw [ha7mem, hea16]; exact hloc1)
  obtain ⟨a8, ha8⟩ : ∃ u : X86Int.St, u = { a7.set 10 (some wd.locals) with rip := a7.rip + 4 } := ⟨_, rfl⟩
  have ha8rip : a8.rip = entry + 27 := by rw [ha8]; show a7.rip + 4 = entry + 27; rw [ha7rip]
  have ha8r10 : a8.reg 10 = some wd.locals := by
    rw [ha8]
    show ({ a7.set 10 (some wd.locals) with rip := a7.rip + 4 }).reg 10 = some wd.locals
    rw [reg_with_rip, reg_set_self (by omega)]
  have ha8r0 : a8.reg 0 = some n := by
    rw [ha8]
    show ({ a7.set 10 (some wd.locals) with rip := a7.rip + 4 }).reg 0 = some n
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha7r0
  have ha8r8 : a8.reg 8 = some h.d := by
    rw [ha8]
    show ({ a7.set 10 (some wd.locals) with rip := a7.rip + 4 }).reg 8 = some h.d
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha7r8
  have ha8r9 : a8.reg 9 = some h.a := by
    rw [ha8]
    show ({ a7.set 10 (some wd.locals) with rip := a7.rip + 4 }).reg 9 = some h.a
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha7r9
  have ha8r6 : a8.reg 6 = some c := by
    rw [ha8]
    show ({ a7.set 10 (some wd.locals) with rip := a7.rip + 4 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha7r6
  have ha8r4 : a8.reg 4 = some sp := by
    rw [ha8]
    show ({ a7.set 10 (some wd.locals) with rip := a7.rip + 4 }).reg 4 = some sp
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha7r4
  have ha8mem : a8.mem = m1 := by
    rw [ha8]
    show ({ a7.set 10 (some wd.locals) with rip := a7.rip + 4 }).mem = m1
    rw [mem_with_rip, mem_set]; exact ha7mem
  have ga8 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a8.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10
    rw [ha8]
    show ({ a7.set 10 (some wd.locals) with rip := a7.rip + 4 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other h10]; exact ga7 i h0 h2 h4 h6 h8 h10
  -- cmp rax, r10
  have e9 : X86Int.step im a8 = .ok { a8 with fl := (aluF .sub n wd.locals).2, rip := a8.rip + 3 } :=
    step_cmp (by rw [ha8rip]; exact hc.d6) ha8r0 ha8r10
  obtain ⟨a9, ha9⟩ : ∃ u : X86Int.St, u = { a8 with fl := (aluF .sub n wd.locals).2, rip := a8.rip + 3 } := ⟨_, rfl⟩
  have ha9rip : a9.rip = entry + 30 := by rw [ha9]; show a8.rip + 3 = entry + 30; rw [ha8rip]
  have ha9fl : X86Int.cond a9.fl 2 = some true := by
    have hcf : a9.fl.cf = some true := by
      rw [ha9]
      show (aluF .sub n wd.locals).2.cf = some true
      exact aluF_sub_cf_true hn
    exact cond_cc2 hcf
  have ha9r8 : a9.reg 8 = some h.d := by rw [ha9]; show a8.reg 8 = some h.d; exact ha8r8
  have ha9r9 : a9.reg 9 = some h.a := by rw [ha9]; show a8.reg 9 = some h.a; exact ha8r9
  have ha9r6 : a9.reg 6 = some c := by rw [ha9]; show a8.reg 6 = some c; exact ha8r6
  have ha9r4 : a9.reg 4 = some sp := by rw [ha9]; show a8.reg 4 = some sp; exact ha8r4
  have ha9mem : a9.mem = m1 := by rw [ha9]; show a8.mem = m1; exact ha8mem
  have ga9 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a9.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10
    rw [ha9]
    show a8.reg i = s.reg i
    exact ga8 i h0 h2 h4 h6 h8 h10
  -- jb taken
  have e10 : X86Int.step im a9 = .ok { a9 with rip := entry + 37 } := by
    have hh := step_jcc (im := im) (s := a9) (by rw [ha9rip]; exact hc.d7) ha9fl
    simpa using hh
  obtain ⟨a10, ha10⟩ : ∃ u : X86Int.St, u = { a9 with rip := entry + 37 } := ⟨_, rfl⟩
  have ha10rip : a10.rip = entry + 37 := by rw [ha10]
  have ha10r8 : a10.reg 8 = some h.d := by
    rw [ha10]; show ({ a9 with rip := entry + 37 }).reg 8 = some h.d
    rw [reg_with_rip]; exact ha9r8
  have ha10r9 : a10.reg 9 = some h.a := by
    rw [ha10]; show ({ a9 with rip := entry + 37 }).reg 9 = some h.a
    rw [reg_with_rip]; exact ha9r9
  have ha10r6 : a10.reg 6 = some c := by
    rw [ha10]; show ({ a9 with rip := entry + 37 }).reg 6 = some c
    rw [reg_with_rip]; exact ha9r6
  have ha10r4 : a10.reg 4 = some sp := by
    rw [ha10]; show ({ a9 with rip := entry + 37 }).reg 4 = some sp
    rw [reg_with_rip]; exact ha9r4
  have ha10mem : a10.mem = m1 := by
    rw [ha10]; show ({ a9 with rip := entry + 37 }).mem = m1
    rw [mem_with_rip]; exact ha9mem
  have ga10 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a10.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10
    rw [ha10]; show ({ a9 with rip := entry + 37 }).reg i = s.reg i
    rw [reg_with_rip]; exact ga9 i h0 h2 h4 h6 h8 h10
  -- sub r8, 1
  have hsub8 : (aluF .sub h.d (1 % W)).1 = h.d - 1 := aluI_sub_val h1W hd1 hdW
  have e11 : X86Int.step im a10 = .ok { a10.set 8 (some (aluF .sub h.d (1 % W)).1) with
      fl := (aluF .sub h.d (1 % W)).2, rip := a10.rip + 4 } :=
    step_aluI_sub (by rw [ha10rip]; exact hc.d8) ha10r8
  obtain ⟨a11, ha11⟩ : ∃ u : X86Int.St, u = { a10.set 8 (some (aluF .sub h.d (1 % W)).1) with
    fl := (aluF .sub h.d (1 % W)).2, rip := a10.rip + 4 } := ⟨_, rfl⟩
  have ha11rip : a11.rip = entry + 41 := by rw [ha11]; show a10.rip + 4 = entry + 41; rw [ha10rip]
  have ha11r8 : a11.reg 8 = some (h.d - 1) := by
    rw [ha11]
    show (a10.set 8 (some (aluF .sub h.d (1 % W)).1)).reg 8 = some (h.d - 1)
    rw [reg_set_self (len_of_r9 ha10r9 (by decide)), hsub8]
  have ha11r9 : a11.reg 9 = some h.a := by
    rw [ha11]
    show (a10.set 8 (some (aluF .sub h.d (1 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha10r9
  have ha11r6 : a11.reg 6 = some c := by
    rw [ha11]
    show (a10.set 8 (some (aluF .sub h.d (1 % W)).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha10r6
  have ha11r4 : a11.reg 4 = some sp := by
    rw [ha11]
    show (a10.set 8 (some (aluF .sub h.d (1 % W)).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact ha10r4
  have ha11mem : a11.mem = m1 := by
    rw [ha11]
    show (a10.set 8 (some (aluF .sub h.d (1 % W)).1)).mem = m1
    rw [mem_set]; exact ha10mem
  have ga11 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a11.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10
    rw [ha11]
    show (a10.set 8 (some (aluF .sub h.d (1 % W)).1)).reg i = s.reg i
    rw [reg_set_other h8]; exact ga10 i h0 h2 h4 h6 h8 h10
  -- add rsi, 5
  have hadd6 : (aluF .add c (5 % W)).1 = c + 5 := aluI_add_val h5W hcW
  have e12 : X86Int.step im a11 = .ok { a11.set 6 (some (aluF .add c (5 % W)).1) with
      fl := (aluF .add c (5 % W)).2, rip := a11.rip + 4 } :=
    step_aluI_add (by rw [ha11rip]; exact hc.d9) ha11r6
  obtain ⟨a12, ha12⟩ : ∃ u : X86Int.St, u = { a11.set 6 (some (aluF .add c (5 % W)).1) with
    fl := (aluF .add c (5 % W)).2, rip := a11.rip + 4 } := ⟨_, rfl⟩
  have ha12rip : a12.rip = entry + 45 := by rw [ha12]; show a11.rip + 4 = entry + 45; rw [ha11rip]
  have ha12r6 : a12.reg 6 = some (c + 5) := by
    rw [ha12]
    show (a11.set 6 (some (aluF .add c (5 % W)).1)).reg 6 = some (c + 5)
    rw [reg_set_self (len_of_r9 ha11r9 (by decide)), hadd6]
  have ha12r8 : a12.reg 8 = some (h.d - 1) := by
    rw [ha12]
    show (a11.set 6 (some (aluF .add c (5 % W)).1)).reg 8 = some (h.d - 1)
    rw [reg_set_other (by decide)]; exact ha11r8
  have ha12r9 : a12.reg 9 = some h.a := by
    rw [ha12]
    show (a11.set 6 (some (aluF .add c (5 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha11r9
  have ha12r4 : a12.reg 4 = some sp := by
    rw [ha12]
    show (a11.set 6 (some (aluF .add c (5 % W)).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact ha11r4
  have ha12mem : a12.mem = m1 := by
    rw [ha12]
    show (a11.set 6 (some (aluF .add c (5 % W)).1)).mem = m1
    rw [mem_set]; exact ha11mem
  have ga12 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → a12.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10
    rw [ha12]
    show (a11.set 6 (some (aluF .add c (5 % W)).1)).reg i = s.reg i
    rw [reg_set_other h6]; exact ga11 i h0 h2 h4 h6 h8 h10
  -- jmp tail
  have e13 : X86Int.step im a12 = .ok { a12 with rip := tail } :=
    step_jmp (by rw [ha12rip]; exact hc.d10)
  obtain ⟨a13, ha13⟩ : ∃ u : X86Int.St, u = { a12 with rip := tail } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
  have r5 : run im 1 (Out.ok a4) = Out.ok a5 := by rw [run_succ_ok im 0 a4, e5, ← ha5, run_zero]
  have r6 : run im 1 (Out.ok a5) = Out.ok a6 := by rw [run_succ_ok im 0 a5, e6, ← ha6, run_zero]
  have r7 : run im 1 (Out.ok a6) = Out.ok a7 := by rw [run_succ_ok im 0 a6, e7, ← ha7, run_zero]
  have r8 : run im 1 (Out.ok a7) = Out.ok a8 := by rw [run_succ_ok im 0 a7, e8, ← ha8, run_zero]
  have r9 : run im 1 (Out.ok a8) = Out.ok a9 := by rw [run_succ_ok im 0 a8, e9, ← ha9, run_zero]
  have r10 : run im 1 (Out.ok a9) = Out.ok a10 := by rw [run_succ_ok im 0 a9, e10, ← ha10, run_zero]
  have r11 : run im 1 (Out.ok a10) = Out.ok a11 := by rw [run_succ_ok im 0 a10, e11, ← ha11, run_zero]
  have r12 : run im 1 (Out.ok a11) = Out.ok a12 := by rw [run_succ_ok im 0 a11, e12, ← ha12, run_zero]
  have r13 : run im 1 (Out.ok a12) = Out.ok a13 := by rw [run_succ_ok im 0 a12, e13, ← ha13, run_zero]
  refine ⟨3 + n8 + 9, a13, ?_, by rw [ha13], ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [run_add im (3 + n8) 9,
      show run im (3 + n8) (Out.ok s) = Out.ok a4 from by
        rw [run_add im 3 n8,
          show run im 3 (Out.ok s) = Out.ok a3 from by
            rw [run_add im 1 2, r1, run_add im 1 1, r2, r3],
          hrun8],
      run_add im 1 8, r5, run_add im 1 7, r6, run_add im 1 6, r7, run_add im 1 5, r8,
      run_add im 1 4, r9, run_add im 1 3, r10, run_add im 1 2, r11, run_add im 1 1, r12, r13]
  · rw [ha13]
    show ({ a12 with rip := tail }).reg 8 = some (h.d - 1)
    rw [reg_with_rip]; exact ha12r8
  · rw [ha13]
    show ({ a12 with rip := tail }).reg 9 = some h.a
    rw [reg_with_rip]; exact ha12r9
  · rw [ha13]
    show ({ a12 with rip := tail }).reg 6 = some (c + 5)
    rw [reg_with_rip]; exact ha12r6
  · rw [ha13]
    show ({ a12 with rip := tail }).reg 4 = some sp
    rw [reg_with_rip]; exact ha12r4
  · intro i h0 h2 h4 h6 h8 h10
    rw [ha13]
    show ({ a12 with rip := tail }).reg i = s.reg i
    rw [reg_with_rip]; exact ga12 i h0 h2 h4 h6 h8 h10
  · intro x m hx
    have hm13 : a13.mem = m1 := by
      rw [ha13]
      show ({ a12 with rip := tail }).mem = m1
      rw [mem_with_rip]; exact ha12mem
    rw [hm13]
    exact rdN_wr64_other hm1s (by cases hx with
      | inl h => exact Or.inr (by omega)
      | inr h => exact Or.inl (by omega))


theorem step_movI {im : Nat → Option (Ins × Nat)} {s : X86Int.St} {d imm len : Nat}
    (hi : im s.rip = some (.movI d imm, len)) :
    X86Int.step im s = .ok { s.set d (some (imm % W)) with rip := s.rip + len } := by
  simp only [X86Int.step, exec, hi]

theorem step_ld8 {im : Nat → Option (Ins × Nat)} {s : X86Int.St} {d b disp len x v : Nat}
    (hi : im s.rip = some (.ld8 d b disp, len)) (hx : s.reg b = some x)
    (hv : s.mem.rd (ea x disp) = some v) :
    X86Int.step im s = .ok { s.set d (some v) with rip := s.rip + len } := by
  simp only [X86Int.step, exec, hi, hx, hv, lift, bind, Option.bind, pure]

theorem step_alu_add {im : Nat → Option (Ins × Nat)} {s : X86Int.St} {d r len a b : Nat}
    (hi : im s.rip = some (.alu .add d r, len)) (ha : s.reg d = some a) (hb : s.reg r = some b) :
    X86Int.step im s = .ok { s.set d (some (aluF .add a b).1) with
      fl := (aluF .add a b).2, rip := s.rip + len } := by
  simp only [X86Int.step, exec, hi, ha, hb, lift, bind, Option.bind, pure]

theorem step_alu_sub_reg {im : Nat → Option (Ins × Nat)} {s : X86Int.St} {d r len a b : Nat}
    (hi : im s.rip = some (.alu .sub d r, len)) (ha : s.reg d = some a) (hb : s.reg r = some b) :
    X86Int.step im s = .ok { s.set d (some (aluF .sub a b).1) with
      fl := (aluF .sub a b).2, rip := s.rip + len } :=
  step_alu_sub hi ha hb

theorem cond_cc5 {f : Flags} {b : Bool} (h : f.zf = some b) : X86Int.cond f 5 = some (!b) := by
  simp [X86Int.cond, h]

/-- A two-byte read is its two single-byte reads. -/
theorem rdN_two {m : Mem} {x b1 b2 : Nat} (h : m.rdN x 2 = some [b1, b2]) :
    m.rd x = some b1 ∧ m.rd (x + 1) = some b2 := by
  cases hb : m.rd x with
  | none =>
    exfalso
    have hz : m.rdN x 2 = none := by
      show (do let b ← m.rd x; let r ← m.rdN (x + 1) 1; pure (b :: r)) = none
      rw [hb]
      rfl
    rw [hz] at h
    cases h
  | some b =>
    cases hb2 : m.rd (x + 1) with
    | none =>
      exfalso
      have h1 : m.rdN (x + 1) 1 = none := by
        show (do let b ← m.rd (x + 1); let r ← m.rdN (x + 1 + 1) 0; pure (b :: r)) = none
        rw [hb2]
        rfl
      have hz : m.rdN x 2 = none := by
        show (do let b ← m.rd x; let r ← m.rdN (x + 1) 1; pure (b :: r)) = none
        rw [hb, h1]
        rfl
      rw [hz] at h
      cases h
    | some b' =>
      have h1 : m.rdN (x + 1) 1 = some [b'] := by
        show (do let b ← m.rd (x + 1); let r ← m.rdN (x + 1 + 1) 0; pure (b :: r)) = some [b']
        rw [hb2]
        rfl
      have h2 : m.rdN x 2 = some [b, b'] := by
        show (do let b ← m.rd x; let r ← m.rdN (x + 1) 1; pure (b :: r)) = some [b, b']
        rw [hb, h1]
        rfl
      rw [h2] at h
      simp only [Option.some.injEq, List.cons.injEq] at h
      refine ⟨?_, ?_⟩
      · rw [h.1]
      · rw [h.2.1]


/-- The fall-through `prim` handler's spec: entered with the opcode
in `rax`, the Eff entry for `p` in memory at `eff + 2 * p.code`, it
reaches the fall-through tail with `d - eff.1 + eff.2`, memory
untouched. The exit prim (`p.code = 50`) is the companion spec. -/
def PrimFallSpec (im : Nat → Option (Ins × Nat)) (entry fall eff : Nat) : Prop :=
  ∀ (p : Prim) (h : Height) (c : Nat), p.code ≠ 50 → p.eff.1 ≤ h.d → h.d < W →
    h.d - p.eff.1 + p.eff.2 < W → c + 1 < W → eff + 2 * p.code + 2 ≤ W →
    ∀ s, s.rip = entry → s.reg 0 = some p.code → s.reg 8 = some h.d → s.reg 9 = some h.a →
      s.reg 6 = some c → 16 ≤ s.regs.length →
      s.mem.rdN (eff + 2 * p.code) 2 = some [p.eff.1, p.eff.2] →
      ∃ k s', run im k (Out.ok s) = Out.ok s' ∧ s'.rip = fall ∧
        s'.reg 8 = some (h.d - p.eff.1 + p.eff.2) ∧ s'.reg 9 = some h.a ∧
        s'.reg 6 = some (c + 1) ∧ s'.mem = s.mem ∧
        (∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → s'.reg i = s.reg i)

/-- Decode facts for `h_prim` (both exits share the prefix). -/
structure PrimCode (im : Nat → Option (Ins × Nat)) (entry fall none eff : Nat) : Prop where
  d0 : im entry = some (.movI 2 eff, 10)
  d1 : im (entry + 10) = some (.alu .add 2 0, 3)
  d2 : im (entry + 13) = some (.alu .add 2 0, 3)
  d3 : im (entry + 16) = some (.ld8 10 2 0, 4)
  d4 : im (entry + 20) = some (.cmp 8 10, 3)
  d5 : im (entry + 23) = some (.jcc 3 (entry + 30), 2)
  d6 : im (entry + 30) = some (.cmpI 0 50, 4)
  d7 : im (entry + 34) = some (.jcc 5 (entry + 45), 2)
  d8x : im (entry + 36) = some (.aluI .add 6 1, 4)
  d9x : im (entry + 40) = some (.jmp none, 5)
  d8 : im (entry + 45) = some (.aluI .add 2 1, 4)
  d9 : im (entry + 49) = some (.ld8 11 2 0, 4)
  d10 : im (entry + 53) = some (.alu .sub 8 10, 3)
  d11 : im (entry + 56) = some (.alu .add 8 11, 3)
  d12 : im (entry + 59) = some (.aluI .add 6 1, 4)
  d13 : im (entry + 63) = some (.jmp fall, 5)

theorem PrimFall_triple {im : Nat → Option (Ins × Nat)} {entry fall none eff : Nat}
    (hc : PrimCode im entry fall none eff) :
    PrimFallSpec im entry fall eff := by
  intro p h c hcode hguard hdW hbound hcW heffW s hrip hr0 hr8 hr9 hr6 hlen htab
  have h1W : 1 < W := by unfold W; decide
  have h50W : 50 < W := by unfold W; decide
  have heff : eff % W = eff := Nat.mod_eq_of_lt (by omega)
  have hcodeW : p.code < W := by omega
  obtain ⟨hrd1, hrd2⟩ := rdN_two htab
  -- mov rdx, eff
  have e1 : X86Int.step im s = .ok { s.set 2 (some (eff % W)) with rip := s.rip + 10 } :=
    step_movI (by rw [hrip]; exact hc.d0)
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s.set 2 (some (eff % W)) with rip := s.rip + 10 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = entry + 10 := by rw [ha1]; show s.rip + 10 = entry + 10; rw [hrip]
  have ha1r2 : a1.reg 2 = some eff := by
    rw [ha1]; show ({ s.set 2 (some (eff % W)) with rip := s.rip + 10 }).reg 2 = some eff
    rw [reg_with_rip, reg_set_self (len_of_r9 hr9 (by decide)), heff]
  have ha1r0 : a1.reg 0 = some p.code := by
    rw [ha1]; show ({ s.set 2 (some (eff % W)) with rip := s.rip + 10 }).reg 0 = some p.code
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr0
  have ha1r8 : a1.reg 8 = some h.d := by
    rw [ha1]; show ({ s.set 2 (some (eff % W)) with rip := s.rip + 10 }).reg 8 = some h.d
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr8
  have ha1r9 : a1.reg 9 = some h.a := by
    rw [ha1]; show ({ s.set 2 (some (eff % W)) with rip := s.rip + 10 }).reg 9 = some h.a
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr9
  have ha1r6 : a1.reg 6 = some c := by
    rw [ha1]; show ({ s.set 2 (some (eff % W)) with rip := s.rip + 10 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr6
  have ha1mem : a1.mem = s.mem := by
    rw [ha1]; show ({ s.set 2 (some (eff % W)) with rip := s.rip + 10 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ga1 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a1.reg i = s.reg i := by
    intro i h2 _ _ _ _
    rw [ha1]; show ({ s.set 2 (some (eff % W)) with rip := s.rip + 10 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other h2]
  have hlen1 : a1.regs.length = s.regs.length := by
    rw [ha1]; show ({ s.set 2 (some (eff % W)) with rip := s.rip + 10 }).regs.length = s.regs.length
    rw [len_with_rip, len_set]
  -- add rdx, rax (twice)
  have hadd1 : (aluF .add eff p.code).1 = eff + p.code := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt (by omega)]
  have e2 : X86Int.step im a1 = .ok { a1.set 2 (some (aluF .add eff p.code).1) with
      fl := (aluF .add eff p.code).2, rip := a1.rip + 3 } :=
    step_alu_add (by rw [ha1rip]; exact hc.d1) ha1r2 ha1r0
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1.set 2 (some (aluF .add eff p.code).1) with
    fl := (aluF .add eff p.code).2, rip := a1.rip + 3 } := ⟨_, rfl⟩
  have ha2rip : a2.rip = entry + 13 := by rw [ha2]; show a1.rip + 3 = entry + 13; rw [ha1rip]
  have ha2r2 : a2.reg 2 = some (eff + p.code) := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add eff p.code).1)).reg 2 = some (eff + p.code)
    rw [reg_set_self (len_of_r9 ha1r9 (by decide)), hadd1]
  have ha2r0 : a2.reg 0 = some p.code := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add eff p.code).1)).reg 0 = some p.code
    rw [reg_set_other (by decide)]; exact ha1r0
  have ha2r8 : a2.reg 8 = some h.d := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add eff p.code).1)).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact ha1r8
  have ha2r9 : a2.reg 9 = some h.a := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add eff p.code).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha1r9
  have ha2r6 : a2.reg 6 = some c := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add eff p.code).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha1r6
  have ha2mem : a2.mem = s.mem := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add eff p.code).1)).mem = s.mem
    rw [mem_set]; exact ha1mem
  have ga2 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a2.reg i = s.reg i := by
    intro i h2 h6 h8 h10 h11
    rw [ha2]
    show (a1.set 2 (some (aluF .add eff p.code).1)).reg i = s.reg i
    rw [reg_set_other h2]; exact ga1 i h2 h6 h8 h10 h11
  have hlen2 : a2.regs.length = s.regs.length := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add eff p.code).1)).regs.length = s.regs.length
    rw [len_set]; exact hlen1
  have hadd2 : (aluF .add (eff + p.code) p.code).1 = eff + 2 * p.code := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt (by omega)]
    omega
  have e3 : X86Int.step im a2 = .ok { a2.set 2 (some (aluF .add (eff + p.code) p.code).1) with
      fl := (aluF .add (eff + p.code) p.code).2, rip := a2.rip + 3 } :=
    step_alu_add (by rw [ha2rip]; exact hc.d2) ha2r2 ha2r0
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2.set 2 (some (aluF .add (eff + p.code) p.code).1) with
    fl := (aluF .add (eff + p.code) p.code).2, rip := a2.rip + 3 } := ⟨_, rfl⟩
  have ha3rip : a3.rip = entry + 16 := by rw [ha3]; show a2.rip + 3 = entry + 16; rw [ha2rip]
  have ha3r2 : a3.reg 2 = some (eff + 2 * p.code) := by
    rw [ha3]
    show (a2.set 2 (some (aluF .add (eff + p.code) p.code).1)).reg 2 = some (eff + 2 * p.code)
    rw [reg_set_self (len_of_r9 ha2r9 (by decide)), hadd2]
  have ha3r0 : a3.reg 0 = some p.code := by
    rw [ha3]
    show (a2.set 2 (some (aluF .add (eff + p.code) p.code).1)).reg 0 = some p.code
    rw [reg_set_other (by decide)]; exact ha2r0
  have ha3r8 : a3.reg 8 = some h.d := by
    rw [ha3]
    show (a2.set 2 (some (aluF .add (eff + p.code) p.code).1)).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact ha2r8
  have ha3r9 : a3.reg 9 = some h.a := by
    rw [ha3]
    show (a2.set 2 (some (aluF .add (eff + p.code) p.code).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha2r9
  have ha3r6 : a3.reg 6 = some c := by
    rw [ha3]
    show (a2.set 2 (some (aluF .add (eff + p.code) p.code).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha2r6
  have ha3mem : a3.mem = s.mem := by
    rw [ha3]
    show (a2.set 2 (some (aluF .add (eff + p.code) p.code).1)).mem = s.mem
    rw [mem_set]; exact ha2mem
  have ga3 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a3.reg i = s.reg i := by
    intro i h2 h6 h8 h10 h11
    rw [ha3]
    show (a2.set 2 (some (aluF .add (eff + p.code) p.code).1)).reg i = s.reg i
    rw [reg_set_other h2]; exact ga2 i h2 h6 h8 h10 h11
  have hlen3 : a3.regs.length = s.regs.length := by
    rw [ha3]
    show (a2.set 2 (some (aluF .add (eff + p.code) p.code).1)).regs.length = s.regs.length
    rw [len_set]; exact hlen2
  -- movzx r10d, byte [rdx]
  have hea0 : ea (eff + 2 * p.code) 0 = eff + 2 * p.code := Nat.mod_eq_of_lt (by omega)
  have e4 : X86Int.step im a3 = .ok { a3.set 10 (some p.eff.1) with rip := a3.rip + 4 } :=
    step_ld8 (by rw [ha3rip]; exact hc.d3) ha3r2 (by rw [ha3mem, hea0]; exact hrd1)
  obtain ⟨a4, ha4⟩ : ∃ u : X86Int.St, u = { a3.set 10 (some p.eff.1) with rip := a3.rip + 4 } := ⟨_, rfl⟩
  have ha4rip : a4.rip = entry + 20 := by rw [ha4]; show a3.rip + 4 = entry + 20; rw [ha3rip]
  have ha4r10 : a4.reg 10 = some p.eff.1 := by
    rw [ha4]
    show ({ a3.set 10 (some p.eff.1) with rip := a3.rip + 4 }).reg 10 = some p.eff.1
    rw [reg_with_rip, reg_set_self (by omega)]
  have ha4r8 : a4.reg 8 = some h.d := by
    rw [ha4]
    show ({ a3.set 10 (some p.eff.1) with rip := a3.rip + 4 }).reg 8 = some h.d
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha3r8
  have ha4r9 : a4.reg 9 = some h.a := by
    rw [ha4]
    show ({ a3.set 10 (some p.eff.1) with rip := a3.rip + 4 }).reg 9 = some h.a
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha3r9
  have ha4r6 : a4.reg 6 = some c := by
    rw [ha4]
    show ({ a3.set 10 (some p.eff.1) with rip := a3.rip + 4 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha3r6
  have ha4r0 : a4.reg 0 = some p.code := by
    rw [ha4]
    show ({ a3.set 10 (some p.eff.1) with rip := a3.rip + 4 }).reg 0 = some p.code
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha3r0
  have ha4r2 : a4.reg 2 = some (eff + 2 * p.code) := by
    rw [ha4]
    show ({ a3.set 10 (some p.eff.1) with rip := a3.rip + 4 }).reg 2 = some (eff + 2 * p.code)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha3r2
  have ha4mem : a4.mem = s.mem := by
    rw [ha4]
    show ({ a3.set 10 (some p.eff.1) with rip := a3.rip + 4 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ha3mem
  have ga4 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a4.reg i = s.reg i := by
    intro i h2 h6 h8 h10 h11
    rw [ha4]
    show ({ a3.set 10 (some p.eff.1) with rip := a3.rip + 4 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other h10]; exact ga3 i h2 h6 h8 h10 h11
  have hlen4 : a4.regs.length = s.regs.length := by
    rw [ha4]
    show ({ a3.set 10 (some p.eff.1) with rip := a3.rip + 4 }).regs.length = s.regs.length
    rw [len_with_rip, len_set]; exact hlen3
  -- cmp r8, r10
  have e5 : X86Int.step im a4 = .ok { a4 with fl := (aluF .sub h.d p.eff.1).2, rip := a4.rip + 3 } :=
    step_cmp (by rw [ha4rip]; exact hc.d4) ha4r8 ha4r10
  obtain ⟨a5, ha5⟩ : ∃ u : X86Int.St, u = { a4 with fl := (aluF .sub h.d p.eff.1).2, rip := a4.rip + 3 } := ⟨_, rfl⟩
  have ha5rip : a5.rip = entry + 23 := by rw [ha5]; show a4.rip + 3 = entry + 23; rw [ha4rip]
  have ha5fl : X86Int.cond a5.fl 3 = some true := by
    have hcf : a5.fl.cf = some false := by
      rw [ha5]
      show (aluF .sub h.d p.eff.1).2.cf = some false
      exact aluF_sub_cf (by omega)
    have := cond_cc3 hcf
    simpa using this
  have ha5r8 : a5.reg 8 = some h.d := by rw [ha5]; show a4.reg 8 = some h.d; exact ha4r8
  have ha5r9 : a5.reg 9 = some h.a := by rw [ha5]; show a4.reg 9 = some h.a; exact ha4r9
  have ha5r6 : a5.reg 6 = some c := by rw [ha5]; show a4.reg 6 = some c; exact ha4r6
  have ha5r0 : a5.reg 0 = some p.code := by rw [ha5]; show a4.reg 0 = some p.code; exact ha4r0
  have ha5r2 : a5.reg 2 = some (eff + 2 * p.code) := by
    rw [ha5]; show a4.reg 2 = some (eff + 2 * p.code); exact ha4r2
  have ha5r10 : a5.reg 10 = some p.eff.1 := by
    rw [ha5]; show a4.reg 10 = some p.eff.1; exact ha4r10
  have ha5mem : a5.mem = s.mem := by rw [ha5]; show a4.mem = s.mem; exact ha4mem
  have ga5 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a5.reg i = s.reg i := by
    intro i h2 h6 h8 h10 h11
    rw [ha5]
    show a4.reg i = s.reg i
    exact ga4 i h2 h6 h8 h10 h11
  have hlen5 : a5.regs.length = s.regs.length := by
    rw [ha5]; show a4.regs.length = s.regs.length; exact hlen4
  -- jae taken
  have e6 : X86Int.step im a5 = .ok { a5 with rip := entry + 30 } := by
    have hh := step_jcc (im := im) (s := a5) (by rw [ha5rip]; exact hc.d5) ha5fl
    simpa using hh
  obtain ⟨a6, ha6⟩ : ∃ u : X86Int.St, u = { a5 with rip := entry + 30 } := ⟨_, rfl⟩
  have ha6rip : a6.rip = entry + 30 := by rw [ha6]
  have ha6r8 : a6.reg 8 = some h.d := by
    rw [ha6]; show ({ a5 with rip := entry + 30 }).reg 8 = some h.d
    rw [reg_with_rip]; exact ha5r8
  have ha6r9 : a6.reg 9 = some h.a := by
    rw [ha6]; show ({ a5 with rip := entry + 30 }).reg 9 = some h.a
    rw [reg_with_rip]; exact ha5r9
  have ha6r6 : a6.reg 6 = some c := by
    rw [ha6]; show ({ a5 with rip := entry + 30 }).reg 6 = some c
    rw [reg_with_rip]; exact ha5r6
  have ha6r0 : a6.reg 0 = some p.code := by
    rw [ha6]; show ({ a5 with rip := entry + 30 }).reg 0 = some p.code
    rw [reg_with_rip]; exact ha5r0
  have ha6r2 : a6.reg 2 = some (eff + 2 * p.code) := by
    rw [ha6]; show ({ a5 with rip := entry + 30 }).reg 2 = some (eff + 2 * p.code)
    rw [reg_with_rip]; exact ha5r2
  have ha6r10 : a6.reg 10 = some p.eff.1 := by
    rw [ha6]; show ({ a5 with rip := entry + 30 }).reg 10 = some p.eff.1
    rw [reg_with_rip]; exact ha5r10
  have ha6mem : a6.mem = s.mem := by
    rw [ha6]; show ({ a5 with rip := entry + 30 }).mem = s.mem
    rw [mem_with_rip]; exact ha5mem
  have ga6 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a6.reg i = s.reg i := by
    intro i h2 h6 h8 h10 h11
    rw [ha6]; show ({ a5 with rip := entry + 30 }).reg i = s.reg i
    rw [reg_with_rip]; exact ga5 i h2 h6 h8 h10 h11
  have hlen6 : a6.regs.length = s.regs.length := by
    rw [ha6]; show ({ a5 with rip := entry + 30 }).regs.length = s.regs.length
    rw [len_with_rip]; exact hlen5
  -- cmp rax, 50
  have e7 : X86Int.step im a6 = .ok { a6 with fl := (aluF .sub p.code (50 % W)).2, rip := a6.rip + 4 } :=
    step_cmpI (by rw [ha6rip]; exact hc.d6) ha6r0
  obtain ⟨a7, ha7⟩ : ∃ u : X86Int.St, u = { a6 with fl := (aluF .sub p.code (50 % W)).2, rip := a6.rip + 4 } := ⟨_, rfl⟩
  have ha7rip : a7.rip = entry + 34 := by rw [ha7]; show a6.rip + 4 = entry + 34; rw [ha6rip]
  have ha7fl : X86Int.cond a7.fl 5 = some true := by
    have hz : a7.fl.zf = some false := by
      rw [ha7]
      show (aluF .sub p.code (50 % W)).2.zf = some false
      rw [aluF_zf]
      have hv : (aluF .sub p.code (50 % W)).1 = (p.code + W - 50) % W := by
        rw [aluF_sub_fst, Nat.mod_eq_of_lt h50W]
      rw [hv, subW_eq_zero hcodeW h50W]
      simp [hcode]
    have := cond_cc5 hz
    simpa using this
  have ha7r8 : a7.reg 8 = some h.d := by rw [ha7]; show a6.reg 8 = some h.d; exact ha6r8
  have ha7r9 : a7.reg 9 = some h.a := by rw [ha7]; show a6.reg 9 = some h.a; exact ha6r9
  have ha7r6 : a7.reg 6 = some c := by rw [ha7]; show a6.reg 6 = some c; exact ha6r6
  have ha7r2 : a7.reg 2 = some (eff + 2 * p.code) := by
    rw [ha7]; show a6.reg 2 = some (eff + 2 * p.code); exact ha6r2
  have ha7r10 : a7.reg 10 = some p.eff.1 := by
    rw [ha7]; show a6.reg 10 = some p.eff.1; exact ha6r10
  have ha7mem : a7.mem = s.mem := by rw [ha7]; show a6.mem = s.mem; exact ha6mem
  have ga7 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a7.reg i = s.reg i := by
    intro i h2 h6 h8 h10 h11
    rw [ha7]
    show a6.reg i = s.reg i
    exact ga6 i h2 h6 h8 h10 h11
  have hlen7 : a7.regs.length = s.regs.length := by
    rw [ha7]; show a6.regs.length = s.regs.length; exact hlen6
  -- jnz prim_fall
  have e8 : X86Int.step im a7 = .ok { a7 with rip := entry + 45 } := by
    have hh := step_jcc (im := im) (s := a7) (by rw [ha7rip]; exact hc.d7) ha7fl
    simpa using hh
  obtain ⟨a8, ha8⟩ : ∃ u : X86Int.St, u = { a7 with rip := entry + 45 } := ⟨_, rfl⟩
  have ha8rip : a8.rip = entry + 45 := by rw [ha8]
  have ha8r8 : a8.reg 8 = some h.d := by
    rw [ha8]; show ({ a7 with rip := entry + 45 }).reg 8 = some h.d
    rw [reg_with_rip]; exact ha7r8
  have ha8r9 : a8.reg 9 = some h.a := by
    rw [ha8]; show ({ a7 with rip := entry + 45 }).reg 9 = some h.a
    rw [reg_with_rip]; exact ha7r9
  have ha8r6 : a8.reg 6 = some c := by
    rw [ha8]; show ({ a7 with rip := entry + 45 }).reg 6 = some c
    rw [reg_with_rip]; exact ha7r6
  have ha8r2 : a8.reg 2 = some (eff + 2 * p.code) := by
    rw [ha8]; show ({ a7 with rip := entry + 45 }).reg 2 = some (eff + 2 * p.code)
    rw [reg_with_rip]; exact ha7r2
  have ha8r10 : a8.reg 10 = some p.eff.1 := by
    rw [ha8]; show ({ a7 with rip := entry + 45 }).reg 10 = some p.eff.1
    rw [reg_with_rip]; exact ha7r10
  have ha8mem : a8.mem = s.mem := by
    rw [ha8]; show ({ a7 with rip := entry + 45 }).mem = s.mem
    rw [mem_with_rip]; exact ha7mem
  have ga8 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a8.reg i = s.reg i := by
    intro i h2 h6 h8 h10 h11
    rw [ha8]; show ({ a7 with rip := entry + 45 }).reg i = s.reg i
    rw [reg_with_rip]; exact ga7 i h2 h6 h8 h10 h11
  have hlen8 : a8.regs.length = s.regs.length := by
    rw [ha8]; show ({ a7 with rip := entry + 45 }).regs.length = s.regs.length
    rw [len_with_rip]; exact hlen7
  -- add rdx, 1
  have hadd3 : (aluF .add (eff + 2 * p.code) (1 % W)).1 = eff + 2 * p.code + 1 :=
    aluI_add_val h1W (by omega)
  have e9 : X86Int.step im a8 = .ok { a8.set 2 (some (aluF .add (eff + 2 * p.code) (1 % W)).1) with
      fl := (aluF .add (eff + 2 * p.code) (1 % W)).2, rip := a8.rip + 4 } :=
    step_aluI_add (by rw [ha8rip]; exact hc.d8) ha8r2
  obtain ⟨a9, ha9⟩ : ∃ u : X86Int.St, u = { a8.set 2 (some (aluF .add (eff + 2 * p.code) (1 % W)).1) with
    fl := (aluF .add (eff + 2 * p.code) (1 % W)).2, rip := a8.rip + 4 } := ⟨_, rfl⟩
  have ha9rip : a9.rip = entry + 49 := by rw [ha9]; show a8.rip + 4 = entry + 49; rw [ha8rip]
  have ha9r2 : a9.reg 2 = some (eff + 2 * p.code + 1) := by
    rw [ha9]
    show (a8.set 2 (some (aluF .add (eff + 2 * p.code) (1 % W)).1)).reg 2 = some (eff + 2 * p.code + 1)
    rw [reg_set_self (len_of_r9 ha8r9 (by decide)), hadd3]
  have ha9r8 : a9.reg 8 = some h.d := by
    rw [ha9]
    show (a8.set 2 (some (aluF .add (eff + 2 * p.code) (1 % W)).1)).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact ha8r8
  have ha9r9 : a9.reg 9 = some h.a := by
    rw [ha9]
    show (a8.set 2 (some (aluF .add (eff + 2 * p.code) (1 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha8r9
  have ha9r6 : a9.reg 6 = some c := by
    rw [ha9]
    show (a8.set 2 (some (aluF .add (eff + 2 * p.code) (1 % W)).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha8r6
  have ha9r10 : a9.reg 10 = some p.eff.1 := by
    rw [ha9]
    show (a8.set 2 (some (aluF .add (eff + 2 * p.code) (1 % W)).1)).reg 10 = some p.eff.1
    rw [reg_set_other (by decide)]; exact ha8r10
  have ha9mem : a9.mem = s.mem := by
    rw [ha9]
    show (a8.set 2 (some (aluF .add (eff + 2 * p.code) (1 % W)).1)).mem = s.mem
    rw [mem_set]; exact ha8mem
  have ga9 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a9.reg i = s.reg i := by
    intro i h2 h6 h8 h10 h11
    rw [ha9]
    show (a8.set 2 (some (aluF .add (eff + 2 * p.code) (1 % W)).1)).reg i = s.reg i
    rw [reg_set_other h2]; exact ga8 i h2 h6 h8 h10 h11
  have hlen9 : a9.regs.length = s.regs.length := by
    rw [ha9]
    show (a8.set 2 (some (aluF .add (eff + 2 * p.code) (1 % W)).1)).regs.length = s.regs.length
    rw [len_set]; exact hlen8
  -- movzx r11d, byte [rdx]
  have hea1 : ea (eff + 2 * p.code + 1) 0 = eff + 2 * p.code + 1 := Nat.mod_eq_of_lt (by omega)
  have e10 : X86Int.step im a9 = .ok { a9.set 11 (some p.eff.2) with rip := a9.rip + 4 } :=
    step_ld8 (by rw [ha9rip]; exact hc.d9) ha9r2 (by rw [ha9mem, hea1]; exact hrd2)
  obtain ⟨a10, ha10⟩ : ∃ u : X86Int.St, u = { a9.set 11 (some p.eff.2) with rip := a9.rip + 4 } := ⟨_, rfl⟩
  have ha10rip : a10.rip = entry + 53 := by rw [ha10]; show a9.rip + 4 = entry + 53; rw [ha9rip]
  have ha10r11 : a10.reg 11 = some p.eff.2 := by
    rw [ha10]
    show ({ a9.set 11 (some p.eff.2) with rip := a9.rip + 4 }).reg 11 = some p.eff.2
    rw [reg_with_rip, reg_set_self (by omega)]
  have ha10r8 : a10.reg 8 = some h.d := by
    rw [ha10]
    show ({ a9.set 11 (some p.eff.2) with rip := a9.rip + 4 }).reg 8 = some h.d
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha9r8
  have ha10r9 : a10.reg 9 = some h.a := by
    rw [ha10]
    show ({ a9.set 11 (some p.eff.2) with rip := a9.rip + 4 }).reg 9 = some h.a
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha9r9
  have ha10r6 : a10.reg 6 = some c := by
    rw [ha10]
    show ({ a9.set 11 (some p.eff.2) with rip := a9.rip + 4 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha9r6
  have ha10r10 : a10.reg 10 = some p.eff.1 := by
    rw [ha10]
    show ({ a9.set 11 (some p.eff.2) with rip := a9.rip + 4 }).reg 10 = some p.eff.1
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha9r10
  have ha10mem : a10.mem = s.mem := by
    rw [ha10]
    show ({ a9.set 11 (some p.eff.2) with rip := a9.rip + 4 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ha9mem
  have ga10 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a10.reg i = s.reg i := by
    intro i h2 h6 h8 h10 h11
    rw [ha10]
    show ({ a9.set 11 (some p.eff.2) with rip := a9.rip + 4 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other h11]; exact ga9 i h2 h6 h8 h10 h11
  -- sub r8, r10
  have hsubv : (aluF .sub h.d p.eff.1).1 = h.d - p.eff.1 := by
    rw [aluF_sub_fst, show h.d + W - p.eff.1 = (h.d - p.eff.1) + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have e11 : X86Int.step im a10 = .ok { a10.set 8 (some (aluF .sub h.d p.eff.1).1) with
      fl := (aluF .sub h.d p.eff.1).2, rip := a10.rip + 3 } :=
    step_alu_sub (by rw [ha10rip]; exact hc.d10) ha10r8 ha10r10
  obtain ⟨a11, ha11⟩ : ∃ u : X86Int.St, u = { a10.set 8 (some (aluF .sub h.d p.eff.1).1) with
    fl := (aluF .sub h.d p.eff.1).2, rip := a10.rip + 3 } := ⟨_, rfl⟩
  have ha11rip : a11.rip = entry + 56 := by rw [ha11]; show a10.rip + 3 = entry + 56; rw [ha10rip]
  have ha11r8 : a11.reg 8 = some (h.d - p.eff.1) := by
    rw [ha11]
    show (a10.set 8 (some (aluF .sub h.d p.eff.1).1)).reg 8 = some (h.d - p.eff.1)
    rw [reg_set_self (len_of_r9 ha10r9 (by decide)), hsubv]
  have ha11r9 : a11.reg 9 = some h.a := by
    rw [ha11]
    show (a10.set 8 (some (aluF .sub h.d p.eff.1).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha10r9
  have ha11r6 : a11.reg 6 = some c := by
    rw [ha11]
    show (a10.set 8 (some (aluF .sub h.d p.eff.1).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha10r6
  have ha11r11 : a11.reg 11 = some p.eff.2 := by
    rw [ha11]
    show (a10.set 8 (some (aluF .sub h.d p.eff.1).1)).reg 11 = some p.eff.2
    rw [reg_set_other (by decide)]; exact ha10r11
  have ha11mem : a11.mem = s.mem := by
    rw [ha11]
    show (a10.set 8 (some (aluF .sub h.d p.eff.1).1)).mem = s.mem
    rw [mem_set]; exact ha10mem
  have ga11 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a11.reg i = s.reg i := by
    intro i h2 h6 h8 h10 h11
    rw [ha11]
    show (a10.set 8 (some (aluF .sub h.d p.eff.1).1)).reg i = s.reg i
    rw [reg_set_other h8]; exact ga10 i h2 h6 h8 h10 h11
  -- add r8, r11
  have haddv : (aluF .add (h.d - p.eff.1) p.eff.2).1 = h.d - p.eff.1 + p.eff.2 := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt hbound]
  have e12 : X86Int.step im a11 = .ok { a11.set 8 (some (aluF .add (h.d - p.eff.1) p.eff.2).1) with
      fl := (aluF .add (h.d - p.eff.1) p.eff.2).2, rip := a11.rip + 3 } :=
    step_alu_add (by rw [ha11rip]; exact hc.d11) ha11r8 ha11r11
  obtain ⟨a12, ha12⟩ : ∃ u : X86Int.St, u = { a11.set 8 (some (aluF .add (h.d - p.eff.1) p.eff.2).1) with
    fl := (aluF .add (h.d - p.eff.1) p.eff.2).2, rip := a11.rip + 3 } := ⟨_, rfl⟩
  have ha12rip : a12.rip = entry + 59 := by rw [ha12]; show a11.rip + 3 = entry + 59; rw [ha11rip]
  have ha12r8 : a12.reg 8 = some (h.d - p.eff.1 + p.eff.2) := by
    rw [ha12]
    show (a11.set 8 (some (aluF .add (h.d - p.eff.1) p.eff.2).1)).reg 8 = some (h.d - p.eff.1 + p.eff.2)
    rw [reg_set_self (len_of_r9 ha11r9 (by decide)), haddv]
  have ha12r9 : a12.reg 9 = some h.a := by
    rw [ha12]
    show (a11.set 8 (some (aluF .add (h.d - p.eff.1) p.eff.2).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha11r9
  have ha12r6 : a12.reg 6 = some c := by
    rw [ha12]
    show (a11.set 8 (some (aluF .add (h.d - p.eff.1) p.eff.2).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha11r6
  have ha12mem : a12.mem = s.mem := by
    rw [ha12]
    show (a11.set 8 (some (aluF .add (h.d - p.eff.1) p.eff.2).1)).mem = s.mem
    rw [mem_set]; exact ha11mem
  have ga12 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a12.reg i = s.reg i := by
    intro i h2 h6 h8 h10 h11
    rw [ha12]
    show (a11.set 8 (some (aluF .add (h.d - p.eff.1) p.eff.2).1)).reg i = s.reg i
    rw [reg_set_other h8]; exact ga11 i h2 h6 h8 h10 h11
  -- add rsi, 1
  have hadd6 : (aluF .add c (1 % W)).1 = c + 1 := aluI_add_val h1W hcW
  have e13 : X86Int.step im a12 = .ok { a12.set 6 (some (aluF .add c (1 % W)).1) with
      fl := (aluF .add c (1 % W)).2, rip := a12.rip + 4 } :=
    step_aluI_add (by rw [ha12rip]; exact hc.d12) ha12r6
  obtain ⟨a13, ha13⟩ : ∃ u : X86Int.St, u = { a12.set 6 (some (aluF .add c (1 % W)).1) with
    fl := (aluF .add c (1 % W)).2, rip := a12.rip + 4 } := ⟨_, rfl⟩
  have ha13rip : a13.rip = entry + 63 := by rw [ha13]; show a12.rip + 4 = entry + 63; rw [ha12rip]
  have ha13r6 : a13.reg 6 = some (c + 1) := by
    rw [ha13]
    show (a12.set 6 (some (aluF .add c (1 % W)).1)).reg 6 = some (c + 1)
    rw [reg_set_self (len_of_r9 ha12r9 (by decide)), hadd6]
  have ha13r8 : a13.reg 8 = some (h.d - p.eff.1 + p.eff.2) := by
    rw [ha13]
    show (a12.set 6 (some (aluF .add c (1 % W)).1)).reg 8 = some (h.d - p.eff.1 + p.eff.2)
    rw [reg_set_other (by decide)]; exact ha12r8
  have ha13r9 : a13.reg 9 = some h.a := by
    rw [ha13]
    show (a12.set 6 (some (aluF .add c (1 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha12r9
  have ha13mem : a13.mem = s.mem := by
    rw [ha13]
    show (a12.set 6 (some (aluF .add c (1 % W)).1)).mem = s.mem
    rw [mem_set]; exact ha12mem
  have ga13 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a13.reg i = s.reg i := by
    intro i h2 h6 h8 h10 h11
    rw [ha13]
    show (a12.set 6 (some (aluF .add c (1 % W)).1)).reg i = s.reg i
    rw [reg_set_other h6]; exact ga12 i h2 h6 h8 h10 h11
  -- jmp fall
  have e14 : X86Int.step im a13 = .ok { a13 with rip := fall } :=
    step_jmp (by rw [ha13rip]; exact hc.d13)
  obtain ⟨a14, ha14⟩ : ∃ u : X86Int.St, u = { a13 with rip := fall } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
  have r4 : run im 1 (Out.ok a3) = Out.ok a4 := by rw [run_succ_ok im 0 a3, e4, ← ha4, run_zero]
  have r5 : run im 1 (Out.ok a4) = Out.ok a5 := by rw [run_succ_ok im 0 a4, e5, ← ha5, run_zero]
  have r6 : run im 1 (Out.ok a5) = Out.ok a6 := by rw [run_succ_ok im 0 a5, e6, ← ha6, run_zero]
  have r7 : run im 1 (Out.ok a6) = Out.ok a7 := by rw [run_succ_ok im 0 a6, e7, ← ha7, run_zero]
  have r8 : run im 1 (Out.ok a7) = Out.ok a8 := by rw [run_succ_ok im 0 a7, e8, ← ha8, run_zero]
  have r9 : run im 1 (Out.ok a8) = Out.ok a9 := by rw [run_succ_ok im 0 a8, e9, ← ha9, run_zero]
  have r10 : run im 1 (Out.ok a9) = Out.ok a10 := by rw [run_succ_ok im 0 a9, e10, ← ha10, run_zero]
  have r11 : run im 1 (Out.ok a10) = Out.ok a11 := by rw [run_succ_ok im 0 a10, e11, ← ha11, run_zero]
  have r12 : run im 1 (Out.ok a11) = Out.ok a12 := by rw [run_succ_ok im 0 a11, e12, ← ha12, run_zero]
  have r13 : run im 1 (Out.ok a12) = Out.ok a13 := by rw [run_succ_ok im 0 a12, e13, ← ha13, run_zero]
  have r14 : run im 1 (Out.ok a13) = Out.ok a14 := by rw [run_succ_ok im 0 a13, e14, ← ha14, run_zero]
  refine ⟨14, a14, ?_, by rw [ha14], ?_, ?_, ?_, ?_, ?_⟩
  · show run im (1 + 13) (Out.ok s) = Out.ok a14
    rw [run_add im 1 13, r1, run_add im 1 12, r2, run_add im 1 11, r3, run_add im 1 10, r4,
      run_add im 1 9, r5, run_add im 1 8, r6, run_add im 1 7, r7, run_add im 1 6, r8,
      run_add im 1 5, r9, run_add im 1 4, r10, run_add im 1 3, r11, run_add im 1 2, r12,
      run_add im 1 1, r13, r14]
  · rw [ha14]
    show ({ a13 with rip := fall }).reg 8 = some (h.d - p.eff.1 + p.eff.2)
    rw [reg_with_rip]; exact ha13r8
  · rw [ha14]
    show ({ a13 with rip := fall }).reg 9 = some h.a
    rw [reg_with_rip]; exact ha13r9
  · rw [ha14]
    show ({ a13 with rip := fall }).reg 6 = some (c + 1)
    rw [reg_with_rip]; exact ha13r6
  · rw [ha14]
    show ({ a13 with rip := fall }).mem = s.mem
    rw [mem_with_rip]; exact ha13mem
  · intro i h2 h6 h8 h10 h11
    rw [ha14]
    show ({ a13 with rip := fall }).reg i = s.reg i
    rw [reg_with_rip]; exact ga13 i h2 h6 h8 h10 h11


/-- The exit `prim` handler's spec: the prim whose code is 50 reads
its Eff entry's first byte as the guard, then goes to the
non-fallthrough tail with heights unchanged. -/
def PrimExitSpec (im : Nat → Option (Ins × Nat)) (entry none eff : Nat) : Prop :=
  ∀ (p : Prim) (h : Height) (c : Nat), p.code = 50 → p.eff.1 ≤ h.d → h.d < W →
    c + 1 < W → eff + 2 * p.code + 2 ≤ W →
    ∀ s, s.rip = entry → s.reg 0 = some p.code → s.reg 8 = some h.d → s.reg 9 = some h.a →
      s.reg 6 = some c → 16 ≤ s.regs.length →
      s.mem.rdN (eff + 2 * p.code) 2 = some [p.eff.1, p.eff.2] →
      ∃ k s', run im k (Out.ok s) = Out.ok s' ∧ s'.rip = none ∧
        s'.reg 8 = some h.d ∧ s'.reg 9 = some h.a ∧
        s'.reg 6 = some (c + 1) ∧ s'.mem = s.mem ∧
        (∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → s'.reg i = s.reg i)

theorem PrimExit_triple {im : Nat → Option (Ins × Nat)} {entry fall none eff : Nat}
    (hc : PrimCode im entry fall none eff) :
    PrimExitSpec im entry none eff := by
  intro p h c hcode hguard hdW hcW heffW s hrip hr0 hr8 hr9 hr6 hlen htab
  have h1W : 1 < W := by unfold W; decide
  have h50W : 50 < W := by unfold W; decide
  have heff : eff % W = eff := Nat.mod_eq_of_lt (by omega)
  have hcodeW : p.code < W := by omega
  obtain ⟨hrd1, hrd2⟩ := rdN_two htab
  -- mov rdx, eff
  have e1 : X86Int.step im s = .ok { s.set 2 (some (eff % W)) with rip := s.rip + 10 } :=
    step_movI (by rw [hrip]; exact hc.d0)
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s.set 2 (some (eff % W)) with rip := s.rip + 10 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = entry + 10 := by rw [ha1]; show s.rip + 10 = entry + 10; rw [hrip]
  have ha1r2 : a1.reg 2 = some eff := by
    rw [ha1]; show ({ s.set 2 (some (eff % W)) with rip := s.rip + 10 }).reg 2 = some eff
    rw [reg_with_rip, reg_set_self (len_of_r9 hr9 (by decide)), heff]
  have ha1r0 : a1.reg 0 = some p.code := by
    rw [ha1]; show ({ s.set 2 (some (eff % W)) with rip := s.rip + 10 }).reg 0 = some p.code
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr0
  have ha1r8 : a1.reg 8 = some h.d := by
    rw [ha1]; show ({ s.set 2 (some (eff % W)) with rip := s.rip + 10 }).reg 8 = some h.d
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr8
  have ha1r9 : a1.reg 9 = some h.a := by
    rw [ha1]; show ({ s.set 2 (some (eff % W)) with rip := s.rip + 10 }).reg 9 = some h.a
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr9
  have ha1r6 : a1.reg 6 = some c := by
    rw [ha1]; show ({ s.set 2 (some (eff % W)) with rip := s.rip + 10 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr6
  have ha1mem : a1.mem = s.mem := by
    rw [ha1]; show ({ s.set 2 (some (eff % W)) with rip := s.rip + 10 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ga1 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a1.reg i = s.reg i := by
    intro i h2 _ _ _ _
    rw [ha1]; show ({ s.set 2 (some (eff % W)) with rip := s.rip + 10 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other h2]
  have hlen1 : a1.regs.length = s.regs.length := by
    rw [ha1]; show ({ s.set 2 (some (eff % W)) with rip := s.rip + 10 }).regs.length = s.regs.length
    rw [len_with_rip, len_set]
  -- add rdx, rax (twice)
  have hadd1 : (aluF .add eff p.code).1 = eff + p.code := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt (by omega)]
  have e2 : X86Int.step im a1 = .ok { a1.set 2 (some (aluF .add eff p.code).1) with
      fl := (aluF .add eff p.code).2, rip := a1.rip + 3 } :=
    step_alu_add (by rw [ha1rip]; exact hc.d1) ha1r2 ha1r0
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1.set 2 (some (aluF .add eff p.code).1) with
    fl := (aluF .add eff p.code).2, rip := a1.rip + 3 } := ⟨_, rfl⟩
  have ha2rip : a2.rip = entry + 13 := by rw [ha2]; show a1.rip + 3 = entry + 13; rw [ha1rip]
  have ha2r2 : a2.reg 2 = some (eff + p.code) := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add eff p.code).1)).reg 2 = some (eff + p.code)
    rw [reg_set_self (len_of_r9 ha1r9 (by decide)), hadd1]
  have ha2r0 : a2.reg 0 = some p.code := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add eff p.code).1)).reg 0 = some p.code
    rw [reg_set_other (by decide)]; exact ha1r0
  have ha2r8 : a2.reg 8 = some h.d := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add eff p.code).1)).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact ha1r8
  have ha2r9 : a2.reg 9 = some h.a := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add eff p.code).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha1r9
  have ha2r6 : a2.reg 6 = some c := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add eff p.code).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha1r6
  have ha2mem : a2.mem = s.mem := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add eff p.code).1)).mem = s.mem
    rw [mem_set]; exact ha1mem
  have ga2 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a2.reg i = s.reg i := by
    intro i h2 h6 h8 h10 h11
    rw [ha2]
    show (a1.set 2 (some (aluF .add eff p.code).1)).reg i = s.reg i
    rw [reg_set_other h2]; exact ga1 i h2 h6 h8 h10 h11
  have hlen2 : a2.regs.length = s.regs.length := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add eff p.code).1)).regs.length = s.regs.length
    rw [len_set]; exact hlen1
  have hadd2 : (aluF .add (eff + p.code) p.code).1 = eff + 2 * p.code := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt (by omega)]
    omega
  have e3 : X86Int.step im a2 = .ok { a2.set 2 (some (aluF .add (eff + p.code) p.code).1) with
      fl := (aluF .add (eff + p.code) p.code).2, rip := a2.rip + 3 } :=
    step_alu_add (by rw [ha2rip]; exact hc.d2) ha2r2 ha2r0
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2.set 2 (some (aluF .add (eff + p.code) p.code).1) with
    fl := (aluF .add (eff + p.code) p.code).2, rip := a2.rip + 3 } := ⟨_, rfl⟩
  have ha3rip : a3.rip = entry + 16 := by rw [ha3]; show a2.rip + 3 = entry + 16; rw [ha2rip]
  have ha3r2 : a3.reg 2 = some (eff + 2 * p.code) := by
    rw [ha3]
    show (a2.set 2 (some (aluF .add (eff + p.code) p.code).1)).reg 2 = some (eff + 2 * p.code)
    rw [reg_set_self (len_of_r9 ha2r9 (by decide)), hadd2]
  have ha3r0 : a3.reg 0 = some p.code := by
    rw [ha3]
    show (a2.set 2 (some (aluF .add (eff + p.code) p.code).1)).reg 0 = some p.code
    rw [reg_set_other (by decide)]; exact ha2r0
  have ha3r8 : a3.reg 8 = some h.d := by
    rw [ha3]
    show (a2.set 2 (some (aluF .add (eff + p.code) p.code).1)).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact ha2r8
  have ha3r9 : a3.reg 9 = some h.a := by
    rw [ha3]
    show (a2.set 2 (some (aluF .add (eff + p.code) p.code).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha2r9
  have ha3r6 : a3.reg 6 = some c := by
    rw [ha3]
    show (a2.set 2 (some (aluF .add (eff + p.code) p.code).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha2r6
  have ha3mem : a3.mem = s.mem := by
    rw [ha3]
    show (a2.set 2 (some (aluF .add (eff + p.code) p.code).1)).mem = s.mem
    rw [mem_set]; exact ha2mem
  have ga3 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a3.reg i = s.reg i := by
    intro i h2 h6 h8 h10 h11
    rw [ha3]
    show (a2.set 2 (some (aluF .add (eff + p.code) p.code).1)).reg i = s.reg i
    rw [reg_set_other h2]; exact ga2 i h2 h6 h8 h10 h11
  have hlen3 : a3.regs.length = s.regs.length := by
    rw [ha3]
    show (a2.set 2 (some (aluF .add (eff + p.code) p.code).1)).regs.length = s.regs.length
    rw [len_set]; exact hlen2
  -- movzx r10d, byte [rdx]
  have hea0 : ea (eff + 2 * p.code) 0 = eff + 2 * p.code := Nat.mod_eq_of_lt (by omega)
  have e4 : X86Int.step im a3 = .ok { a3.set 10 (some p.eff.1) with rip := a3.rip + 4 } :=
    step_ld8 (by rw [ha3rip]; exact hc.d3) ha3r2 (by rw [ha3mem, hea0]; exact hrd1)
  obtain ⟨a4, ha4⟩ : ∃ u : X86Int.St, u = { a3.set 10 (some p.eff.1) with rip := a3.rip + 4 } := ⟨_, rfl⟩
  have ha4rip : a4.rip = entry + 20 := by rw [ha4]; show a3.rip + 4 = entry + 20; rw [ha3rip]
  have ha4r10 : a4.reg 10 = some p.eff.1 := by
    rw [ha4]
    show ({ a3.set 10 (some p.eff.1) with rip := a3.rip + 4 }).reg 10 = some p.eff.1
    rw [reg_with_rip, reg_set_self (by omega)]
  have ha4r8 : a4.reg 8 = some h.d := by
    rw [ha4]
    show ({ a3.set 10 (some p.eff.1) with rip := a3.rip + 4 }).reg 8 = some h.d
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha3r8
  have ha4r9 : a4.reg 9 = some h.a := by
    rw [ha4]
    show ({ a3.set 10 (some p.eff.1) with rip := a3.rip + 4 }).reg 9 = some h.a
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha3r9
  have ha4r6 : a4.reg 6 = some c := by
    rw [ha4]
    show ({ a3.set 10 (some p.eff.1) with rip := a3.rip + 4 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha3r6
  have ha4r0 : a4.reg 0 = some p.code := by
    rw [ha4]
    show ({ a3.set 10 (some p.eff.1) with rip := a3.rip + 4 }).reg 0 = some p.code
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha3r0
  have ha4r2 : a4.reg 2 = some (eff + 2 * p.code) := by
    rw [ha4]
    show ({ a3.set 10 (some p.eff.1) with rip := a3.rip + 4 }).reg 2 = some (eff + 2 * p.code)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha3r2
  have ha4mem : a4.mem = s.mem := by
    rw [ha4]
    show ({ a3.set 10 (some p.eff.1) with rip := a3.rip + 4 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ha3mem
  have ga4 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a4.reg i = s.reg i := by
    intro i h2 h6 h8 h10 h11
    rw [ha4]
    show ({ a3.set 10 (some p.eff.1) with rip := a3.rip + 4 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other h10]; exact ga3 i h2 h6 h8 h10 h11
  have hlen4 : a4.regs.length = s.regs.length := by
    rw [ha4]
    show ({ a3.set 10 (some p.eff.1) with rip := a3.rip + 4 }).regs.length = s.regs.length
    rw [len_with_rip, len_set]; exact hlen3
  -- cmp r8, r10
  have e5 : X86Int.step im a4 = .ok { a4 with fl := (aluF .sub h.d p.eff.1).2, rip := a4.rip + 3 } :=
    step_cmp (by rw [ha4rip]; exact hc.d4) ha4r8 ha4r10
  obtain ⟨a5, ha5⟩ : ∃ u : X86Int.St, u = { a4 with fl := (aluF .sub h.d p.eff.1).2, rip := a4.rip + 3 } := ⟨_, rfl⟩
  have ha5rip : a5.rip = entry + 23 := by rw [ha5]; show a4.rip + 3 = entry + 23; rw [ha4rip]
  have ha5fl : X86Int.cond a5.fl 3 = some true := by
    have hcf : a5.fl.cf = some false := by
      rw [ha5]
      show (aluF .sub h.d p.eff.1).2.cf = some false
      exact aluF_sub_cf (by omega)
    have := cond_cc3 hcf
    simpa using this
  have ha5r8 : a5.reg 8 = some h.d := by rw [ha5]; show a4.reg 8 = some h.d; exact ha4r8
  have ha5r9 : a5.reg 9 = some h.a := by rw [ha5]; show a4.reg 9 = some h.a; exact ha4r9
  have ha5r6 : a5.reg 6 = some c := by rw [ha5]; show a4.reg 6 = some c; exact ha4r6
  have ha5r0 : a5.reg 0 = some p.code := by rw [ha5]; show a4.reg 0 = some p.code; exact ha4r0
  have ha5r2 : a5.reg 2 = some (eff + 2 * p.code) := by
    rw [ha5]; show a4.reg 2 = some (eff + 2 * p.code); exact ha4r2
  have ha5r10 : a5.reg 10 = some p.eff.1 := by
    rw [ha5]; show a4.reg 10 = some p.eff.1; exact ha4r10
  have ha5mem : a5.mem = s.mem := by rw [ha5]; show a4.mem = s.mem; exact ha4mem
  have ga5 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a5.reg i = s.reg i := by
    intro i h2 h6 h8 h10 h11
    rw [ha5]
    show a4.reg i = s.reg i
    exact ga4 i h2 h6 h8 h10 h11
  have hlen5 : a5.regs.length = s.regs.length := by
    rw [ha5]; show a4.regs.length = s.regs.length; exact hlen4
  -- jae taken
  have e6 : X86Int.step im a5 = .ok { a5 with rip := entry + 30 } := by
    have hh := step_jcc (im := im) (s := a5) (by rw [ha5rip]; exact hc.d5) ha5fl
    simpa using hh
  obtain ⟨a6, ha6⟩ : ∃ u : X86Int.St, u = { a5 with rip := entry + 30 } := ⟨_, rfl⟩
  have ha6rip : a6.rip = entry + 30 := by rw [ha6]
  have ha6r8 : a6.reg 8 = some h.d := by
    rw [ha6]; show ({ a5 with rip := entry + 30 }).reg 8 = some h.d
    rw [reg_with_rip]; exact ha5r8
  have ha6r9 : a6.reg 9 = some h.a := by
    rw [ha6]; show ({ a5 with rip := entry + 30 }).reg 9 = some h.a
    rw [reg_with_rip]; exact ha5r9
  have ha6r6 : a6.reg 6 = some c := by
    rw [ha6]; show ({ a5 with rip := entry + 30 }).reg 6 = some c
    rw [reg_with_rip]; exact ha5r6
  have ha6r0 : a6.reg 0 = some p.code := by
    rw [ha6]; show ({ a5 with rip := entry + 30 }).reg 0 = some p.code
    rw [reg_with_rip]; exact ha5r0
  have ha6r2 : a6.reg 2 = some (eff + 2 * p.code) := by
    rw [ha6]; show ({ a5 with rip := entry + 30 }).reg 2 = some (eff + 2 * p.code)
    rw [reg_with_rip]; exact ha5r2
  have ha6r10 : a6.reg 10 = some p.eff.1 := by
    rw [ha6]; show ({ a5 with rip := entry + 30 }).reg 10 = some p.eff.1
    rw [reg_with_rip]; exact ha5r10
  have ha6mem : a6.mem = s.mem := by
    rw [ha6]; show ({ a5 with rip := entry + 30 }).mem = s.mem
    rw [mem_with_rip]; exact ha5mem
  have ga6 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a6.reg i = s.reg i := by
    intro i h2 h6 h8 h10 h11
    rw [ha6]; show ({ a5 with rip := entry + 30 }).reg i = s.reg i
    rw [reg_with_rip]; exact ga5 i h2 h6 h8 h10 h11
  have hlen6 : a6.regs.length = s.regs.length := by
    rw [ha6]; show ({ a5 with rip := entry + 30 }).regs.length = s.regs.length
    rw [len_with_rip]; exact hlen5
  -- cmp rax, 50
  have e7 : X86Int.step im a6 = .ok { a6 with fl := (aluF .sub p.code (50 % W)).2, rip := a6.rip + 4 } :=
    step_cmpI (by rw [ha6rip]; exact hc.d6) ha6r0
  obtain ⟨a7, ha7⟩ : ∃ u : X86Int.St, u = { a6 with fl := (aluF .sub p.code (50 % W)).2, rip := a6.rip + 4 } := ⟨_, rfl⟩
  have ha7rip : a7.rip = entry + 34 := by rw [ha7]; show a6.rip + 4 = entry + 34; rw [ha6rip]
  have ha7fl : X86Int.cond a7.fl 5 = some false := by
    have hz : a7.fl.zf = some true := by
      rw [ha7]
      show (aluF .sub p.code (50 % W)).2.zf = some true
      rw [aluF_zf]
      have hv : (aluF .sub p.code (50 % W)).1 = (p.code + W - 50) % W := by
        rw [aluF_sub_fst, Nat.mod_eq_of_lt h50W]
      rw [hv, subW_eq_zero hcodeW h50W]
      simp [hcode]
    exact cond_cc5 hz
  have ha7r8 : a7.reg 8 = some h.d := by rw [ha7]; show a6.reg 8 = some h.d; exact ha6r8
  have ha7r9 : a7.reg 9 = some h.a := by rw [ha7]; show a6.reg 9 = some h.a; exact ha6r9
  have ha7r6 : a7.reg 6 = some c := by rw [ha7]; show a6.reg 6 = some c; exact ha6r6
  have ha7r2 : a7.reg 2 = some (eff + 2 * p.code) := by
    rw [ha7]; show a6.reg 2 = some (eff + 2 * p.code); exact ha6r2
  have ha7r10 : a7.reg 10 = some p.eff.1 := by
    rw [ha7]; show a6.reg 10 = some p.eff.1; exact ha6r10
  have ha7mem : a7.mem = s.mem := by rw [ha7]; show a6.mem = s.mem; exact ha6mem
  have ga7 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a7.reg i = s.reg i := by
    intro i h2 h6 h8 h10 h11
    rw [ha7]
    show a6.reg i = s.reg i
    exact ga6 i h2 h6 h8 h10 h11
  have hlen7 : a7.regs.length = s.regs.length := by
    rw [ha7]; show a6.regs.length = s.regs.length; exact hlen6
  -- jnz not taken
  have e8 : X86Int.step im a7 = .ok { a7 with rip := entry + 36 } := by
    have hh := step_jcc (im := im) (s := a7) (by rw [ha7rip]; exact hc.d7) ha7fl
    simpa [ha7rip] using hh
  obtain ⟨a8, ha8⟩ : ∃ u : X86Int.St, u = { a7 with rip := entry + 36 } := ⟨_, rfl⟩
  have ha8rip : a8.rip = entry + 36 := by rw [ha8]
  have ha8r8 : a8.reg 8 = some h.d := by
    rw [ha8]; show ({ a7 with rip := entry + 36 }).reg 8 = some h.d
    rw [reg_with_rip]; exact ha7r8
  have ha8r9 : a8.reg 9 = some h.a := by
    rw [ha8]; show ({ a7 with rip := entry + 36 }).reg 9 = some h.a
    rw [reg_with_rip]; exact ha7r9
  have ha8r6 : a8.reg 6 = some c := by
    rw [ha8]; show ({ a7 with rip := entry + 36 }).reg 6 = some c
    rw [reg_with_rip]; exact ha7r6
  have ha8mem : a8.mem = s.mem := by
    rw [ha8]; show ({ a7 with rip := entry + 36 }).mem = s.mem
    rw [mem_with_rip]; exact ha7mem
  have ga8 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a8.reg i = s.reg i := by
    intro i h2 h6 h8 h10 h11
    rw [ha8]; show ({ a7 with rip := entry + 36 }).reg i = s.reg i
    rw [reg_with_rip]; exact ga7 i h2 h6 h8 h10 h11
  -- add rsi, 1
  have hadd6 : (aluF .add c (1 % W)).1 = c + 1 := aluI_add_val h1W hcW
  have e9 : X86Int.step im a8 = .ok { a8.set 6 (some (aluF .add c (1 % W)).1) with
      fl := (aluF .add c (1 % W)).2, rip := a8.rip + 4 } :=
    step_aluI_add (by rw [ha8rip]; exact hc.d8x) ha8r6
  obtain ⟨a9, ha9⟩ : ∃ u : X86Int.St, u = { a8.set 6 (some (aluF .add c (1 % W)).1) with
    fl := (aluF .add c (1 % W)).2, rip := a8.rip + 4 } := ⟨_, rfl⟩
  have ha9rip : a9.rip = entry + 40 := by rw [ha9]; show a8.rip + 4 = entry + 40; rw [ha8rip]
  have ha9r6 : a9.reg 6 = some (c + 1) := by
    rw [ha9]
    show (a8.set 6 (some (aluF .add c (1 % W)).1)).reg 6 = some (c + 1)
    rw [reg_set_self (len_of_r9 ha8r9 (by decide)), hadd6]
  have ha9r8 : a9.reg 8 = some h.d := by
    rw [ha9]
    show (a8.set 6 (some (aluF .add c (1 % W)).1)).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact ha8r8
  have ha9r9 : a9.reg 9 = some h.a := by
    rw [ha9]
    show (a8.set 6 (some (aluF .add c (1 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha8r9
  have ha9mem : a9.mem = s.mem := by
    rw [ha9]
    show (a8.set 6 (some (aluF .add c (1 % W)).1)).mem = s.mem
    rw [mem_set]; exact ha8mem
  have ga9 : ∀ i, i ≠ 2 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a9.reg i = s.reg i := by
    intro i h2 h6 h8 h10 h11
    rw [ha9]
    show (a8.set 6 (some (aluF .add c (1 % W)).1)).reg i = s.reg i
    rw [reg_set_other h6]; exact ga8 i h2 h6 h8 h10 h11
  -- jmp none tail
  have e10 : X86Int.step im a9 = .ok { a9 with rip := none } :=
    step_jmp (by rw [ha9rip]; exact hc.d9x)
  obtain ⟨a10, ha10⟩ : ∃ u : X86Int.St, u = { a9 with rip := none } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
  have r4 : run im 1 (Out.ok a3) = Out.ok a4 := by rw [run_succ_ok im 0 a3, e4, ← ha4, run_zero]
  have r5 : run im 1 (Out.ok a4) = Out.ok a5 := by rw [run_succ_ok im 0 a4, e5, ← ha5, run_zero]
  have r6 : run im 1 (Out.ok a5) = Out.ok a6 := by rw [run_succ_ok im 0 a5, e6, ← ha6, run_zero]
  have r7 : run im 1 (Out.ok a6) = Out.ok a7 := by rw [run_succ_ok im 0 a6, e7, ← ha7, run_zero]
  have r8 : run im 1 (Out.ok a7) = Out.ok a8 := by rw [run_succ_ok im 0 a7, e8, ← ha8, run_zero]
  have r9 : run im 1 (Out.ok a8) = Out.ok a9 := by rw [run_succ_ok im 0 a8, e9, ← ha9, run_zero]
  have r10 : run im 1 (Out.ok a9) = Out.ok a10 := by rw [run_succ_ok im 0 a9, e10, ← ha10, run_zero]
  refine ⟨10, a10, ?_, by rw [ha10], ?_, ?_, ?_, ?_, ?_⟩
  · show run im (1 + 9) (Out.ok s) = Out.ok a10
    rw [run_add im 1 9, r1, run_add im 1 8, r2, run_add im 1 7, r3, run_add im 1 6, r4,
      run_add im 1 5, r5, run_add im 1 4, r6, run_add im 1 3, r7, run_add im 1 2, r8,
      run_add im 1 1, r9, r10]
  · rw [ha10]
    show ({ a9 with rip := none }).reg 8 = some h.d
    rw [reg_with_rip]; exact ha9r8
  · rw [ha10]
    show ({ a9 with rip := none }).reg 9 = some h.a
    rw [reg_with_rip]; exact ha9r9
  · rw [ha10]
    show ({ a9 with rip := none }).reg 6 = some (c + 1)
    rw [reg_with_rip]; exact ha9r6
  · rw [ha10]
    show ({ a9 with rip := none }).mem = s.mem
    rw [mem_with_rip]; exact ha9mem
  · intro i h2 h6 h8 h10 h11
    rw [ha10]
    show ({ a9 with rip := none }).reg i = s.reg i
    rw [reg_with_rip]; exact ga9 i h2 h6 h8 h10 h11


/-- The `ret` handler's spec: at height exactly `(wd.outs, 0)` with
the word entry's outs field at `w + 8`, it reaches the
non-fallthrough tail with heights unchanged. -/
def RetSpec (im : Nat → Option (Ins × Nat)) (entry none : Nat) (wd : Word) : Prop :=
  ∀ (c w : Nat), wd.outs < W → c + 1 < W → w + 16 < W →
    ∀ s, s.rip = entry → s.reg 8 = some wd.outs → s.reg 9 = some 0 → s.reg 6 = some c →
      s.reg 7 = some w → rd64 s.mem (w + 8) = some wd.outs →
      ∃ k s', run im k (Out.ok s) = Out.ok s' ∧ s'.rip = none ∧
        s'.reg 8 = some wd.outs ∧ s'.reg 9 = some 0 ∧ s'.reg 6 = some (c + 1) ∧
        s'.mem = s.mem ∧
        (∀ i, i ≠ 0 → i ≠ 6 → s'.reg i = s.reg i)

/-- Decode facts for `h_ret`. -/
structure RetCode (im : Nat → Option (Ins × Nat)) (entry none : Nat) : Prop where
  d0 : im entry = some (.ld 0 7 8, 4)
  d1 : im (entry + 4) = some (.cmp 8 0, 3)
  d2 : im (entry + 7) = some (.jcc 4 (entry + 14), 2)
  d3 : im (entry + 14) = some (.test 9 9, 3)
  d4 : im (entry + 17) = some (.jcc 4 (entry + 24), 2)
  d5 : im (entry + 24) = some (.aluI .add 6 1, 4)
  d6 : im (entry + 28) = some (.jmp none, 5)

theorem Ret_triple {im : Nat → Option (Ins × Nat)} {entry none : Nat} {wd : Word}
    (hc : RetCode im entry none) :
    RetSpec im entry none wd := by
  intro c w houtW hcW hwW s hrip hr8 hr9 hr6 hr7 houts
  have h1W : 1 < W := by unfold W; decide
  have hea8 : ea w 8 = w + 8 := Nat.mod_eq_of_lt (by omega)
  have e1 : X86Int.step im s = .ok { s.set 0 (some wd.outs) with rip := s.rip + 4 } :=
    step_ld (by rw [hrip]; exact hc.d0) hr7 (by rw [hea8]; exact houts)
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s.set 0 (some wd.outs) with rip := s.rip + 4 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = entry + 4 := by rw [ha1]; show s.rip + 4 = entry + 4; rw [hrip]
  have ha1r0 : a1.reg 0 = some wd.outs := by
    rw [ha1]; show ({ s.set 0 (some wd.outs) with rip := s.rip + 4 }).reg 0 = some wd.outs
    rw [reg_with_rip, reg_set_self (len_of_r9 hr9 (by decide))]
  have ha1r8 : a1.reg 8 = some wd.outs := by
    rw [ha1]; show ({ s.set 0 (some wd.outs) with rip := s.rip + 4 }).reg 8 = some wd.outs
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr8
  have ha1r9 : a1.reg 9 = some 0 := by
    rw [ha1]; show ({ s.set 0 (some wd.outs) with rip := s.rip + 4 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr9
  have ha1r6 : a1.reg 6 = some c := by
    rw [ha1]; show ({ s.set 0 (some wd.outs) with rip := s.rip + 4 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr6
  have ha1mem : a1.mem = s.mem := by
    rw [ha1]; show ({ s.set 0 (some wd.outs) with rip := s.rip + 4 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ga1 : ∀ i, i ≠ 0 → i ≠ 6 → a1.reg i = s.reg i := by
    intro i h0 _
    rw [ha1]; show ({ s.set 0 (some wd.outs) with rip := s.rip + 4 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other h0]
  have e2 : X86Int.step im a1 = .ok { a1 with fl := (aluF .sub wd.outs wd.outs).2, rip := a1.rip + 3 } :=
    step_cmp (by rw [ha1rip]; exact hc.d1) ha1r8 ha1r0
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1 with fl := (aluF .sub wd.outs wd.outs).2, rip := a1.rip + 3 } := ⟨_, rfl⟩
  have ha2rip : a2.rip = entry + 7 := by rw [ha2]; show a1.rip + 3 = entry + 7; rw [ha1rip]
  have ha2fl : X86Int.cond a2.fl 4 = some true := by
    have hz : a2.fl.zf = some true := by
      rw [ha2]
      show (aluF .sub wd.outs wd.outs).2.zf = some true
      rw [aluF_zf, aluF_sub_fst, subW_eq_zero houtW houtW]
      simp
    exact cond_cc4 hz
  have ha2r8 : a2.reg 8 = some wd.outs := by rw [ha2]; show a1.reg 8 = some wd.outs; exact ha1r8
  have ha2r9 : a2.reg 9 = some 0 := by rw [ha2]; show a1.reg 9 = some 0; exact ha1r9
  have ha2r6 : a2.reg 6 = some c := by rw [ha2]; show a1.reg 6 = some c; exact ha1r6
  have ha2mem : a2.mem = s.mem := by rw [ha2]; show a1.mem = s.mem; exact ha1mem
  have ga2 : ∀ i, i ≠ 0 → i ≠ 6 → a2.reg i = s.reg i := by
    intro i h0 h6
    rw [ha2]
    show a1.reg i = s.reg i
    exact ga1 i h0 h6
  have e3 : X86Int.step im a2 = .ok { a2 with rip := entry + 14 } := by
    have hh := step_jcc (im := im) (s := a2) (by rw [ha2rip]; exact hc.d2) ha2fl
    simpa using hh
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2 with rip := entry + 14 } := ⟨_, rfl⟩
  have ha3rip : a3.rip = entry + 14 := by rw [ha3]
  have ha3r8 : a3.reg 8 = some wd.outs := by
    rw [ha3]; show ({ a2 with rip := entry + 14 }).reg 8 = some wd.outs
    rw [reg_with_rip]; exact ha2r8
  have ha3r9 : a3.reg 9 = some 0 := by
    rw [ha3]; show ({ a2 with rip := entry + 14 }).reg 9 = some 0
    rw [reg_with_rip]; exact ha2r9
  have ha3r6 : a3.reg 6 = some c := by
    rw [ha3]; show ({ a2 with rip := entry + 14 }).reg 6 = some c
    rw [reg_with_rip]; exact ha2r6
  have ha3mem : a3.mem = s.mem := by
    rw [ha3]; show ({ a2 with rip := entry + 14 }).mem = s.mem
    rw [mem_with_rip]; exact ha2mem
  have ga3 : ∀ i, i ≠ 0 → i ≠ 6 → a3.reg i = s.reg i := by
    intro i h0 h6
    rw [ha3]; show ({ a2 with rip := entry + 14 }).reg i = s.reg i
    rw [reg_with_rip]; exact ga2 i h0 h6
  have e4 : X86Int.step im a3 = .ok { a3 with fl := (aluF .and 0 0).2, rip := a3.rip + 3 } :=
    step_test (by rw [ha3rip]; exact hc.d3) ha3r9 ha3r9
  obtain ⟨a4, ha4⟩ : ∃ u : X86Int.St, u = { a3 with fl := (aluF .and 0 0).2, rip := a3.rip + 3 } := ⟨_, rfl⟩
  have ha4rip : a4.rip = entry + 17 := by rw [ha4]; show a3.rip + 3 = entry + 17; rw [ha3rip]
  have ha4fl : X86Int.cond a4.fl 4 = some true := by
    have hz : a4.fl.zf = some true := by
      rw [ha4]
      show (aluF .and 0 0).2.zf = some true
      rw [aluF_zf, aluF_and_fst_self]
      decide
    exact cond_cc4 hz
  have ha4r8 : a4.reg 8 = some wd.outs := by rw [ha4]; show a3.reg 8 = some wd.outs; exact ha3r8
  have ha4r9 : a4.reg 9 = some 0 := by rw [ha4]; show a3.reg 9 = some 0; exact ha3r9
  have ha4r6 : a4.reg 6 = some c := by rw [ha4]; show a3.reg 6 = some c; exact ha3r6
  have ha4mem : a4.mem = s.mem := by rw [ha4]; show a3.mem = s.mem; exact ha3mem
  have ga4 : ∀ i, i ≠ 0 → i ≠ 6 → a4.reg i = s.reg i := by
    intro i h0 h6
    rw [ha4]
    show a3.reg i = s.reg i
    exact ga3 i h0 h6
  have e5 : X86Int.step im a4 = .ok { a4 with rip := entry + 24 } := by
    have hh := step_jcc (im := im) (s := a4) (by rw [ha4rip]; exact hc.d4) ha4fl
    simpa using hh
  obtain ⟨a5, ha5⟩ : ∃ u : X86Int.St, u = { a4 with rip := entry + 24 } := ⟨_, rfl⟩
  have ha5rip : a5.rip = entry + 24 := by rw [ha5]
  have ha5r8 : a5.reg 8 = some wd.outs := by
    rw [ha5]; show ({ a4 with rip := entry + 24 }).reg 8 = some wd.outs
    rw [reg_with_rip]; exact ha4r8
  have ha5r9 : a5.reg 9 = some 0 := by
    rw [ha5]; show ({ a4 with rip := entry + 24 }).reg 9 = some 0
    rw [reg_with_rip]; exact ha4r9
  have ha5r6 : a5.reg 6 = some c := by
    rw [ha5]; show ({ a4 with rip := entry + 24 }).reg 6 = some c
    rw [reg_with_rip]; exact ha4r6
  have ha5mem : a5.mem = s.mem := by
    rw [ha5]; show ({ a4 with rip := entry + 24 }).mem = s.mem
    rw [mem_with_rip]; exact ha4mem
  have ga5 : ∀ i, i ≠ 0 → i ≠ 6 → a5.reg i = s.reg i := by
    intro i h0 h6
    rw [ha5]; show ({ a4 with rip := entry + 24 }).reg i = s.reg i
    rw [reg_with_rip]; exact ga4 i h0 h6
  have hadd6 : (aluF .add c (1 % W)).1 = c + 1 := aluI_add_val h1W hcW
  have e6 : X86Int.step im a5 = .ok { a5.set 6 (some (aluF .add c (1 % W)).1) with
      fl := (aluF .add c (1 % W)).2, rip := a5.rip + 4 } :=
    step_aluI_add (by rw [ha5rip]; exact hc.d5) ha5r6
  obtain ⟨a6, ha6⟩ : ∃ u : X86Int.St, u = { a5.set 6 (some (aluF .add c (1 % W)).1) with
    fl := (aluF .add c (1 % W)).2, rip := a5.rip + 4 } := ⟨_, rfl⟩
  have ha6rip : a6.rip = entry + 28 := by rw [ha6]; show a5.rip + 4 = entry + 28; rw [ha5rip]
  have ha6r6 : a6.reg 6 = some (c + 1) := by
    rw [ha6]
    show (a5.set 6 (some (aluF .add c (1 % W)).1)).reg 6 = some (c + 1)
    rw [reg_set_self (len_of_r9 ha5r9 (by decide)), hadd6]
  have ha6r8 : a6.reg 8 = some wd.outs := by
    rw [ha6]
    show (a5.set 6 (some (aluF .add c (1 % W)).1)).reg 8 = some wd.outs
    rw [reg_set_other (by decide)]; exact ha5r8
  have ha6r9 : a6.reg 9 = some 0 := by
    rw [ha6]
    show (a5.set 6 (some (aluF .add c (1 % W)).1)).reg 9 = some 0
    rw [reg_set_other (by decide)]; exact ha5r9
  have ha6mem : a6.mem = s.mem := by
    rw [ha6]
    show (a5.set 6 (some (aluF .add c (1 % W)).1)).mem = s.mem
    rw [mem_set]; exact ha5mem
  have ga6 : ∀ i, i ≠ 0 → i ≠ 6 → a6.reg i = s.reg i := by
    intro i h0 h6
    rw [ha6]
    show (a5.set 6 (some (aluF .add c (1 % W)).1)).reg i = s.reg i
    rw [reg_set_other h6]; exact ga5 i h0 h6
  have e7 : X86Int.step im a6 = .ok { a6 with rip := none } :=
    step_jmp (by rw [ha6rip]; exact hc.d6)
  obtain ⟨a7, ha7⟩ : ∃ u : X86Int.St, u = { a6 with rip := none } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
  have r4 : run im 1 (Out.ok a3) = Out.ok a4 := by rw [run_succ_ok im 0 a3, e4, ← ha4, run_zero]
  have r5 : run im 1 (Out.ok a4) = Out.ok a5 := by rw [run_succ_ok im 0 a4, e5, ← ha5, run_zero]
  have r6 : run im 1 (Out.ok a5) = Out.ok a6 := by rw [run_succ_ok im 0 a5, e6, ← ha6, run_zero]
  have r7 : run im 1 (Out.ok a6) = Out.ok a7 := by rw [run_succ_ok im 0 a6, e7, ← ha7, run_zero]
  refine ⟨7, a7, ?_, by rw [ha7], ?_, ?_, ?_, ?_, ?_⟩
  · show run im (1 + 6) (Out.ok s) = Out.ok a7
    rw [run_add im 1 6, r1, run_add im 1 5, r2, run_add im 1 4, r3, run_add im 1 3, r4,
      run_add im 1 2, r5, run_add im 1 1, r6, r7]
  · rw [ha7]
    show ({ a6 with rip := none }).reg 8 = some wd.outs
    rw [reg_with_rip]; exact ha6r8
  · rw [ha7]
    show ({ a6 with rip := none }).reg 9 = some 0
    rw [reg_with_rip]; exact ha6r9
  · rw [ha7]
    show ({ a6 with rip := none }).reg 6 = some (c + 1)
    rw [reg_with_rip]; exact ha6r6
  · rw [ha7]
    show ({ a6 with rip := none }).mem = s.mem
    rw [mem_with_rip]; exact ha6mem
  · intro i h0 h6
    rw [ha7]
    show ({ a6 with rip := none }).reg i = s.reg i
    rw [reg_with_rip]; exact ga6 i h0 h6


/-- The `call` handler's spec: the immediate at cursor+1 reads as
the callee index `n`, in range of the word count at `r13`; the
callee entry at `tbl + 64*n` declares `(ins, outs)`, and the
handler reaches the fall-through tail with `d - ins + outs`. -/
def CallSpec (im : Nat → Option (Ins × Nat)) (entry fall l32 : Nat) : Prop :=
  ∀ (cw : Word) (h : Height) (c n sp tbl nw : Nat), n < nw → cw.ins ≤ h.d → h.d < W →
    h.d - cw.ins + cw.outs < W → c + 5 < W → 8 ≤ sp → sp < W →
    tbl + n * 64 + 16 < W → (sp ≤ c + 1 ∨ c + 5 ≤ sp - 8) →
    (sp ≤ tbl + n * 64 ∨ tbl + n * 64 + 16 ≤ sp - 8) →
    ∀ s, s.rip = entry → s.reg 8 = some h.d → s.reg 9 = some h.a → s.reg 6 = some c →
      s.reg 4 = some sp → s.reg 13 = some nw → s.reg 14 = some tbl →
      16 ≤ s.regs.length →
      (s.mem.rdN (c + 1) 4).map leVal = some n →
      rd64 s.mem (tbl + n * 64) = some cw.ins →
      rd64 s.mem (tbl + n * 64 + 8) = some cw.outs →
      (∀ v, (wr64 s.mem (sp - 8) v).isSome) →
      ∃ k s', run im k (Out.ok s) = Out.ok s' ∧ s'.rip = fall ∧
        s'.reg 8 = some (h.d - cw.ins + cw.outs) ∧ s'.reg 9 = some h.a ∧
        s'.reg 6 = some (c + 5) ∧ s'.reg 4 = some sp ∧
        (∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → s'.reg i = s.reg i) ∧
        (∀ x m, x + m ≤ sp - 8 ∨ sp ≤ x → s'.mem.rdN x m = s.mem.rdN x m)

/-- Decode facts for `h_call`. -/
structure CallCode (im : Nat → Option (Ins × Nat)) (entry fall l32 : Nat) : Prop where
  d0 : im entry = some (.mov 2 6, 3)
  d1 : im (entry + 3) = some (.aluI .add 2 1, 4)
  d2 : im (entry + 7) = some (.call l32, 5)
  d3 : im (entry + 12) = some (.cmp 0 13, 3)
  d4 : im (entry + 15) = some (.jcc 2 (entry + 22), 2)
  d5 : im (entry + 22) = some (.mov 2 0, 3)
  d6 : im (entry + 25) = some (.alu .add 2 2, 3)
  d7 : im (entry + 28) = some (.alu .add 2 2, 3)
  d8 : im (entry + 31) = some (.alu .add 2 2, 3)
  d9 : im (entry + 34) = some (.alu .add 2 2, 3)
  d10 : im (entry + 37) = some (.alu .add 2 2, 3)
  d11 : im (entry + 40) = some (.alu .add 2 2, 3)
  d12 : im (entry + 43) = some (.alu .add 2 14, 3)
  d13 : im (entry + 46) = some (.ld 10 2 0, 3)
  d14 : im (entry + 49) = some (.cmp 8 10, 3)
  d15 : im (entry + 52) = some (.jcc 3 (entry + 59), 2)
  d16 : im (entry + 59) = some (.ld 11 2 8, 4)
  d17 : im (entry + 63) = some (.alu .sub 8 10, 3)
  d18 : im (entry + 66) = some (.alu .add 8 11, 3)
  d19 : im (entry + 69) = some (.aluI .add 6 5, 4)
  d20 : im (entry + 73) = some (.jmp fall, 5)
  dret : im (l32 + 2) = some (.ret, 1)

theorem Call_triple {im : Nat → Option (Ins × Nat)} {entry fall l32 : Nat}
    (hc : CallCode im entry fall l32) (hl : L32Code im l32) (hentry : entry + 78 < W) :
    CallSpec im entry fall l32 := by
  intro cw h c n sp tbl nw hnw hins hdW hbound hcW hsp8 hspW htbl hdisjc hdisjt s hrip hr8 hr9 hr6
    hr4 hr13 hr14 hlen hrd hinsm houtsm hw
  have h1W : 1 < W := by unfold W; decide
  have h5W : 5 < W := by unfold W; decide
  -- mov rdx, rsi
  have e1 : X86Int.step im s = .ok { s.set 2 (some c) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact hc.d0) hr6
  obtain ⟨a1, ha1⟩ : ∃ u : X86Int.St, u = { s.set 2 (some c) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = entry + 3 := by rw [ha1]; show s.rip + 3 = entry + 3; rw [hrip]
  have ha1r2 : a1.reg 2 = some c := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 2 = some c
    rw [reg_with_rip, reg_set_self (len_of_r9 hr9 (by decide))]
  have ha1r8 : a1.reg 8 = some h.d := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 8 = some h.d
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr8
  have ha1r9 : a1.reg 9 = some h.a := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 9 = some h.a
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr9
  have ha1r6 : a1.reg 6 = some c := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr6
  have ha1r4 : a1.reg 4 = some sp := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 4 = some sp
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr4
  have ha1mem : a1.mem = s.mem := by
    rw [ha1]; show ({ s.set 2 (some c) with rip := s.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  -- add rdx, 1
  have hadd2 : (aluF .add c (1 % W)).1 = c + 1 := aluI_add_val h1W (by omega)
  have e2 : X86Int.step im a1 = .ok { a1.set 2 (some (aluF .add c (1 % W)).1) with
      fl := (aluF .add c (1 % W)).2, rip := a1.rip + 4 } :=
    step_aluI_add (by rw [ha1rip]; exact hc.d1) ha1r2
  obtain ⟨a2, ha2⟩ : ∃ u : X86Int.St, u = { a1.set 2 (some (aluF .add c (1 % W)).1) with
    fl := (aluF .add c (1 % W)).2, rip := a1.rip + 4 } := ⟨_, rfl⟩
  have ha2rip : a2.rip = entry + 7 := by rw [ha2]; show a1.rip + 4 = entry + 7; rw [ha1rip]
  have ha2r2 : a2.reg 2 = some (c + 1) := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 2 = some (c + 1)
    rw [reg_set_self (len_of_r9 ha1r9 (by decide)), hadd2]
  have ha2r8 : a2.reg 8 = some h.d := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact ha1r8
  have ha2r9 : a2.reg 9 = some h.a := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha1r9
  have ha2r6 : a2.reg 6 = some c := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha1r6
  have ha2r4 : a2.reg 4 = some sp := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact ha1r4
  have ha2mem : a2.mem = s.mem := by
    rw [ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).mem = s.mem
    rw [mem_set]; exact ha1mem
  -- call L32
  have hslot8 : (sp + W - 8) % W = sp - 8 := by
    rw [show sp + W - 8 = sp + W - 8 * 1 by omega, slot_sub (by omega) hspW]
  obtain ⟨m1, hm1s⟩ := Option.isSome_iff_exists.mp (hw (entry + 12))
  have hm1 : wr64 a2.mem ((sp + W - 8) % W) (a2.rip + 5) = some m1 := by
    rw [ha2mem, ha2rip, hslot8]
    exact hm1s
  have e3 : X86Int.step im a2 = .ok { a2.set 4 (some ((sp + W - 8) % W)) with
      mem := m1, rip := l32 } :=
    step_call (by rw [ha2rip]; exact hc.d2) ha2r4 hm1
  obtain ⟨a3, ha3⟩ : ∃ u : X86Int.St, u = { a2.set 4 (some ((sp + W - 8) % W)) with
    mem := m1, rip := l32 } := ⟨_, rfl⟩
  have ha3rip : a3.rip = l32 := by rw [ha3]
  have ha3r4 : a3.reg 4 = some (sp - 8) := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
    rw [reg_set_self (len_of_r9 ha2r9 (by decide)), hslot8]
  have ha3r2 : a3.reg 2 = some (c + 1) := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 2 = some (c + 1)
    rw [reg_set_other (by decide)]; exact ha2r2
  have ha3r8 : a3.reg 8 = some h.d := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact ha2r8
  have ha3r9 : a3.reg 9 = some h.a := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha2r9
  have ha3r6 : a3.reg 6 = some c := by
    rw [ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha2r6
  have ha3mem : a3.mem = m1 := by rw [ha3]
  have hfr1 : m1.rdN (c + 1) 4 = s.mem.rdN (c + 1) 4 :=
    rdN_wr64_other hm1s (by cases hdisjc with
      | inl h => exact Or.inl (by omega)
      | inr h => exact Or.inr (by omega))
  have hrdm1 : (m1.rdN (c + 1) 4).map leVal = some n := by
    rw [hfr1]
    exact hrd
  -- L32 as a black box
  have hrdsome : (a3.mem.rdN (c + 1) 4).isSome := by
    rw [ha3mem]
    cases h : m1.rdN (c + 1) 4 with
    | none => simp [h] at hrdm1
    | some bs => rfl
  obtain ⟨n8, a4, hrun8, ha4rip, ha4r0, -, ha4mem, ha4other, ha4len⟩ :=
    L32_triple hl (c + 1) (by omega) a3 ha3rip ⟨ha3r2, hrdsome⟩
  have ha4r0' : a4.reg 0 = some n := by
    rw [ha4r0, ha3mem]
    exact hrdm1
  have ha4mem' : a4.mem = m1 := by rw [ha4mem]; exact ha3mem
  have ha4r4 : a4.reg 4 = some (sp - 8) := by rw [ha4other 4 (by decide)]; exact ha3r4
  have ha4r8 : a4.reg 8 = some h.d := by rw [ha4other 8 (by decide)]; exact ha3r8
  have ha4r9 : a4.reg 9 = some h.a := by rw [ha4other 9 (by decide)]; exact ha3r9
  have ha4r6 : a4.reg 6 = some c := by rw [ha4other 6 (by decide)]; exact ha3r6
  -- ret from L32
  have hrd64 : rd64 a4.mem (sp - 8) = some (entry + 12) := by
    have h1 := rd64_wr64 hm1s
    rw [Nat.mod_eq_of_lt (by omega)] at h1
    rwa [ha4mem']
  have e5 : X86Int.step im a4 = .ok { a4.set 4 (some ((sp - 8 + 8) % W)) with
      rip := entry + 12 } :=
    step_ret (by rw [ha4rip]; exact hc.dret) ha4r4 hrd64
  obtain ⟨a5, ha5⟩ : ∃ u : X86Int.St, u = { a4.set 4 (some ((sp - 8 + 8) % W)) with
    rip := entry + 12 } := ⟨_, rfl⟩
  have ha5rip : a5.rip = entry + 12 := by rw [ha5]
  have ha5r4 : a5.reg 4 = some sp := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 4 = some sp
    rw [reg_with_rip, reg_set_self (len_of_r9 ha4r9 (by decide)), slot_add8 hsp8 hspW]
  have ha5r0 : a5.reg 0 = some n := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 0 = some n
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha4r0'
  have ha5r8 : a5.reg 8 = some h.d := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 8 = some h.d
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha4r8
  have ha5r9 : a5.reg 9 = some h.a := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 9 = some h.a
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha4r9
  have ha5r6 : a5.reg 6 = some c := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact ha4r6
  have ha5mem : a5.mem = m1 := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).mem = m1
    rw [mem_with_rip, mem_set]; exact ha4mem'
  have ha5r13 : a5.reg 13 = some nw := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 13 = some nw
    rw [reg_with_rip, reg_set_other (by decide), ha4other 13 (by decide), ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 13 = some nw
    rw [reg_set_other (by decide), ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 13 = some nw
    rw [reg_set_other (by decide), ha1]
    show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 13 = some nw
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr13
  have ha5r14 : a5.reg 14 = some tbl := by
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg 14 = some tbl
    rw [reg_with_rip, reg_set_other (by decide), ha4other 14 (by decide), ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg 14 = some tbl
    rw [reg_set_other (by decide), ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg 14 = some tbl
    rw [reg_set_other (by decide), ha1]
    show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg 14 = some tbl
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr14
  have ga5 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a5.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10 h11
    rw [ha5]
    show ({ a4.set 4 (some ((sp - 8 + 8) % W)) with rip := entry + 12 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other h4, ha4other i h0, ha3]
    show (a2.set 4 (some ((sp + W - 8) % W))).reg i = s.reg i
    rw [reg_set_other h4, ha2]
    show (a1.set 2 (some (aluF .add c (1 % W)).1)).reg i = s.reg i
    rw [reg_set_other h2, ha1]
    show ({ s.set 2 (some c) with rip := s.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other h2]
  -- cmp rax, r13
  have e6 : X86Int.step im a5 = .ok { a5 with fl := (aluF .sub n nw).2, rip := a5.rip + 3 } :=
    step_cmp (by rw [ha5rip]; exact hc.d3) ha5r0 ha5r13
  obtain ⟨a6, ha6⟩ : ∃ u : X86Int.St, u = { a5 with fl := (aluF .sub n nw).2, rip := a5.rip + 3 } := ⟨_, rfl⟩
  have ha6rip : a6.rip = entry + 15 := by rw [ha6]; show a5.rip + 3 = entry + 15; rw [ha5rip]
  have ha6fl : X86Int.cond a6.fl 2 = some true := by
    have hcf : a6.fl.cf = some true := by
      rw [ha6]
      show (aluF .sub n nw).2.cf = some true
      exact aluF_sub_cf_true hnw
    exact cond_cc2 hcf
  have ha6r0 : a6.reg 0 = some n := by rw [ha6]; show a5.reg 0 = some n; exact ha5r0
  have ha6r8 : a6.reg 8 = some h.d := by rw [ha6]; show a5.reg 8 = some h.d; exact ha5r8
  have ha6r9 : a6.reg 9 = some h.a := by rw [ha6]; show a5.reg 9 = some h.a; exact ha5r9
  have ha6r6 : a6.reg 6 = some c := by rw [ha6]; show a5.reg 6 = some c; exact ha5r6
  have ha6r4 : a6.reg 4 = some sp := by rw [ha6]; show a5.reg 4 = some sp; exact ha5r4
  have ha6r14 : a6.reg 14 = some tbl := by rw [ha6]; show a5.reg 14 = some tbl; exact ha5r14
  have ha6mem : a6.mem = m1 := by rw [ha6]; show a5.mem = m1; exact ha5mem
  have ga6 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a6.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10 h11
    rw [ha6]; show a5.reg i = s.reg i; exact ga5 i h0 h2 h4 h6 h8 h10 h11
  -- jb taken
  have e7 : X86Int.step im a6 = .ok { a6 with rip := entry + 22 } := by
    have hh := step_jcc (im := im) (s := a6) (by rw [ha6rip]; exact hc.d4) ha6fl
    simpa using hh
  obtain ⟨a7, ha7⟩ : ∃ u : X86Int.St, u = { a6 with rip := entry + 22 } := ⟨_, rfl⟩
  have ha7rip : a7.rip = entry + 22 := by rw [ha7]
  have ha7r0 : a7.reg 0 = some n := by
    rw [ha7]
    show ({ a6 with rip := entry + 22 }).reg 0 = some n
    rw [reg_with_rip]; exact ha6r0
  have ha7r8 : a7.reg 8 = some h.d := by
    rw [ha7]
    show ({ a6 with rip := entry + 22 }).reg 8 = some h.d
    rw [reg_with_rip]; exact ha6r8
  have ha7r9 : a7.reg 9 = some h.a := by
    rw [ha7]
    show ({ a6 with rip := entry + 22 }).reg 9 = some h.a
    rw [reg_with_rip]; exact ha6r9
  have ha7r6 : a7.reg 6 = some c := by
    rw [ha7]
    show ({ a6 with rip := entry + 22 }).reg 6 = some c
    rw [reg_with_rip]; exact ha6r6
  have ha7r4 : a7.reg 4 = some sp := by
    rw [ha7]
    show ({ a6 with rip := entry + 22 }).reg 4 = some sp
    rw [reg_with_rip]; exact ha6r4
  have ha7r14 : a7.reg 14 = some tbl := by
    rw [ha7]
    show ({ a6 with rip := entry + 22 }).reg 14 = some tbl
    rw [reg_with_rip]; exact ha6r14
  have ha7mem : a7.mem = m1 := by
    rw [ha7]
    show ({ a6 with rip := entry + 22 }).mem = m1
    rw [mem_with_rip]; exact ha6mem
  have ga7 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a7.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10 h11
    rw [ha7]
    show ({ a6 with rip := entry + 22 }).reg i = s.reg i
    rw [reg_with_rip]; exact ga6 i h0 h2 h4 h6 h8 h10 h11
  -- mov rdx, rax
  have e8 : X86Int.step im a7 = .ok { a7.set 2 (some n) with rip := a7.rip + 3 } :=
    step_mov (by rw [ha7rip]; exact hc.d5) ha7r0
  obtain ⟨a8, ha8⟩ : ∃ u : X86Int.St, u = { a7.set 2 (some n) with rip := a7.rip + 3 } := ⟨_, rfl⟩
  have ha8rip : a8.rip = entry + 25 := by rw [ha8]; show a7.rip + 3 = entry + 25; rw [ha7rip]
  have ha8r2 : a8.reg 2 = some n := by
    rw [ha8]
    show (a7.set 2 (some n)).reg 2 = some n
    rw [reg_set_self (len_of_r9 ha7r9 (by decide))]
  have ha8r8 : a8.reg 8 = some h.d := by
    rw [ha8]
    show (a7.set 2 (some n)).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact ha7r8
  have ha8r9 : a8.reg 9 = some h.a := by
    rw [ha8]
    show (a7.set 2 (some n)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha7r9
  have ha8r6 : a8.reg 6 = some c := by
    rw [ha8]
    show (a7.set 2 (some n)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha7r6
  have ha8r4 : a8.reg 4 = some sp := by
    rw [ha8]
    show (a7.set 2 (some n)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact ha7r4
  have ha8r14 : a8.reg 14 = some tbl := by
    rw [ha8]
    show (a7.set 2 (some n)).reg 14 = some tbl
    rw [reg_set_other (by decide)]; exact ha7r14
  have ha8mem : a8.mem = m1 := by
    rw [ha8]
    show (a7.set 2 (some n)).mem = m1
    rw [mem_set]; exact ha7mem
  have ga8 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → a8.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10 h11
    rw [ha8]
    show (a7.set 2 (some n)).reg i = s.reg i
    rw [reg_set_other h2]; exact ga7 i h0 h2 h4 h6 h8 h10 h11
  -- add rdx, rdx (1)
  have hdbl1 : (aluF .add n n).1 = n * 2 := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt (by omega)]; omega
  have eb1 : X86Int.step im a8 = .ok { a8.set 2 (some (aluF .add n n).1) with
      fl := (aluF .add n n).2, rip := a8.rip + 3 } :=
    step_alu_add (by rw [ha8rip]; exact hc.d6) ha8r2 ha8r2
  obtain ⟨b1, hb1⟩ : ∃ u : X86Int.St, u = { a8.set 2 (some (aluF .add n n).1) with
    fl := (aluF .add n n).2, rip := a8.rip + 3 } := ⟨_, rfl⟩
  have hb1rip : b1.rip = entry + 28 := by rw [hb1]; show a8.rip + 3 = entry + 28; rw [ha8rip]
  have hb1r2 : b1.reg 2 = some (n * 2) := by
    rw [hb1]
    show (a8.set 2 (some (aluF .add n n).1)).reg 2 = some (n * 2)
    rw [reg_set_self (len_of_r9 ha8r9 (by decide)), hdbl1]
  have hb1r8 : b1.reg 8 = some h.d := by
    rw [hb1]
    show (a8.set 2 (some (aluF .add n n).1)).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact ha8r8
  have hb1r9 : b1.reg 9 = some h.a := by
    rw [hb1]
    show (a8.set 2 (some (aluF .add n n).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact ha8r9
  have hb1r6 : b1.reg 6 = some c := by
    rw [hb1]
    show (a8.set 2 (some (aluF .add n n).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ha8r6
  have hb1r4 : b1.reg 4 = some sp := by
    rw [hb1]
    show (a8.set 2 (some (aluF .add n n).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact ha8r4
  have hb1r14 : b1.reg 14 = some tbl := by
    rw [hb1]
    show (a8.set 2 (some (aluF .add n n).1)).reg 14 = some tbl
    rw [reg_set_other (by decide)]; exact ha8r14
  have hb1mem : b1.mem = m1 := by
    rw [hb1]
    show (a8.set 2 (some (aluF .add n n).1)).mem = m1
    rw [mem_set]; exact ha8mem
  have gb1 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → b1.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10 h11
    rw [hb1]
    show (a8.set 2 (some (aluF .add n n).1)).reg i = s.reg i
    rw [reg_set_other h2]; exact ga8 i h0 h2 h4 h6 h8 h10 h11
  -- add rdx, rdx (2)
  have hdbl2 : (aluF .add (n * 2) (n * 2)).1 = n * 4 := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt (by omega)]; omega
  have eb2 : X86Int.step im b1 = .ok { b1.set 2 (some (aluF .add (n * 2) (n * 2)).1) with
      fl := (aluF .add (n * 2) (n * 2)).2, rip := b1.rip + 3 } :=
    step_alu_add (by rw [hb1rip]; exact hc.d7) hb1r2 hb1r2
  obtain ⟨b2, hb2⟩ : ∃ u : X86Int.St, u = { b1.set 2 (some (aluF .add (n * 2) (n * 2)).1) with
    fl := (aluF .add (n * 2) (n * 2)).2, rip := b1.rip + 3 } := ⟨_, rfl⟩
  have hb2rip : b2.rip = entry + 31 := by rw [hb2]; show b1.rip + 3 = entry + 31; rw [hb1rip]
  have hb2r2 : b2.reg 2 = some (n * 4) := by
    rw [hb2]
    show (b1.set 2 (some (aluF .add (n * 2) (n * 2)).1)).reg 2 = some (n * 4)
    rw [reg_set_self (len_of_r9 hb1r9 (by decide)), hdbl2]
  have hb2r8 : b2.reg 8 = some h.d := by
    rw [hb2]
    show (b1.set 2 (some (aluF .add (n * 2) (n * 2)).1)).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact hb1r8
  have hb2r9 : b2.reg 9 = some h.a := by
    rw [hb2]
    show (b1.set 2 (some (aluF .add (n * 2) (n * 2)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact hb1r9
  have hb2r6 : b2.reg 6 = some c := by
    rw [hb2]
    show (b1.set 2 (some (aluF .add (n * 2) (n * 2)).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact hb1r6
  have hb2r4 : b2.reg 4 = some sp := by
    rw [hb2]
    show (b1.set 2 (some (aluF .add (n * 2) (n * 2)).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact hb1r4
  have hb2r14 : b2.reg 14 = some tbl := by
    rw [hb2]
    show (b1.set 2 (some (aluF .add (n * 2) (n * 2)).1)).reg 14 = some tbl
    rw [reg_set_other (by decide)]; exact hb1r14
  have hb2mem : b2.mem = m1 := by
    rw [hb2]
    show (b1.set 2 (some (aluF .add (n * 2) (n * 2)).1)).mem = m1
    rw [mem_set]; exact hb1mem
  have gb2 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → b2.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10 h11
    rw [hb2]
    show (b1.set 2 (some (aluF .add (n * 2) (n * 2)).1)).reg i = s.reg i
    rw [reg_set_other h2]; exact gb1 i h0 h2 h4 h6 h8 h10 h11
  -- add rdx, rdx (3)
  have hdbl3 : (aluF .add (n * 4) (n * 4)).1 = n * 8 := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt (by omega)]; omega
  have eb3 : X86Int.step im b2 = .ok { b2.set 2 (some (aluF .add (n * 4) (n * 4)).1) with
      fl := (aluF .add (n * 4) (n * 4)).2, rip := b2.rip + 3 } :=
    step_alu_add (by rw [hb2rip]; exact hc.d8) hb2r2 hb2r2
  obtain ⟨b3, hb3⟩ : ∃ u : X86Int.St, u = { b2.set 2 (some (aluF .add (n * 4) (n * 4)).1) with
    fl := (aluF .add (n * 4) (n * 4)).2, rip := b2.rip + 3 } := ⟨_, rfl⟩
  have hb3rip : b3.rip = entry + 34 := by rw [hb3]; show b2.rip + 3 = entry + 34; rw [hb2rip]
  have hb3r2 : b3.reg 2 = some (n * 8) := by
    rw [hb3]
    show (b2.set 2 (some (aluF .add (n * 4) (n * 4)).1)).reg 2 = some (n * 8)
    rw [reg_set_self (len_of_r9 hb2r9 (by decide)), hdbl3]
  have hb3r8 : b3.reg 8 = some h.d := by
    rw [hb3]
    show (b2.set 2 (some (aluF .add (n * 4) (n * 4)).1)).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact hb2r8
  have hb3r9 : b3.reg 9 = some h.a := by
    rw [hb3]
    show (b2.set 2 (some (aluF .add (n * 4) (n * 4)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact hb2r9
  have hb3r6 : b3.reg 6 = some c := by
    rw [hb3]
    show (b2.set 2 (some (aluF .add (n * 4) (n * 4)).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact hb2r6
  have hb3r4 : b3.reg 4 = some sp := by
    rw [hb3]
    show (b2.set 2 (some (aluF .add (n * 4) (n * 4)).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact hb2r4
  have hb3r14 : b3.reg 14 = some tbl := by
    rw [hb3]
    show (b2.set 2 (some (aluF .add (n * 4) (n * 4)).1)).reg 14 = some tbl
    rw [reg_set_other (by decide)]; exact hb2r14
  have hb3mem : b3.mem = m1 := by
    rw [hb3]
    show (b2.set 2 (some (aluF .add (n * 4) (n * 4)).1)).mem = m1
    rw [mem_set]; exact hb2mem
  have gb3 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → b3.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10 h11
    rw [hb3]
    show (b2.set 2 (some (aluF .add (n * 4) (n * 4)).1)).reg i = s.reg i
    rw [reg_set_other h2]; exact gb2 i h0 h2 h4 h6 h8 h10 h11
  -- add rdx, rdx (4)
  have hdbl4 : (aluF .add (n * 8) (n * 8)).1 = n * 16 := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt (by omega)]; omega
  have eb4 : X86Int.step im b3 = .ok { b3.set 2 (some (aluF .add (n * 8) (n * 8)).1) with
      fl := (aluF .add (n * 8) (n * 8)).2, rip := b3.rip + 3 } :=
    step_alu_add (by rw [hb3rip]; exact hc.d9) hb3r2 hb3r2
  obtain ⟨b4, hb4⟩ : ∃ u : X86Int.St, u = { b3.set 2 (some (aluF .add (n * 8) (n * 8)).1) with
    fl := (aluF .add (n * 8) (n * 8)).2, rip := b3.rip + 3 } := ⟨_, rfl⟩
  have hb4rip : b4.rip = entry + 37 := by rw [hb4]; show b3.rip + 3 = entry + 37; rw [hb3rip]
  have hb4r2 : b4.reg 2 = some (n * 16) := by
    rw [hb4]
    show (b3.set 2 (some (aluF .add (n * 8) (n * 8)).1)).reg 2 = some (n * 16)
    rw [reg_set_self (len_of_r9 hb3r9 (by decide)), hdbl4]
  have hb4r8 : b4.reg 8 = some h.d := by
    rw [hb4]
    show (b3.set 2 (some (aluF .add (n * 8) (n * 8)).1)).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact hb3r8
  have hb4r9 : b4.reg 9 = some h.a := by
    rw [hb4]
    show (b3.set 2 (some (aluF .add (n * 8) (n * 8)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact hb3r9
  have hb4r6 : b4.reg 6 = some c := by
    rw [hb4]
    show (b3.set 2 (some (aluF .add (n * 8) (n * 8)).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact hb3r6
  have hb4r4 : b4.reg 4 = some sp := by
    rw [hb4]
    show (b3.set 2 (some (aluF .add (n * 8) (n * 8)).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact hb3r4
  have hb4r14 : b4.reg 14 = some tbl := by
    rw [hb4]
    show (b3.set 2 (some (aluF .add (n * 8) (n * 8)).1)).reg 14 = some tbl
    rw [reg_set_other (by decide)]; exact hb3r14
  have hb4mem : b4.mem = m1 := by
    rw [hb4]
    show (b3.set 2 (some (aluF .add (n * 8) (n * 8)).1)).mem = m1
    rw [mem_set]; exact hb3mem
  have gb4 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → b4.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10 h11
    rw [hb4]
    show (b3.set 2 (some (aluF .add (n * 8) (n * 8)).1)).reg i = s.reg i
    rw [reg_set_other h2]; exact gb3 i h0 h2 h4 h6 h8 h10 h11
  -- add rdx, rdx (5)
  have hdbl5 : (aluF .add (n * 16) (n * 16)).1 = n * 32 := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt (by omega)]; omega
  have eb5 : X86Int.step im b4 = .ok { b4.set 2 (some (aluF .add (n * 16) (n * 16)).1) with
      fl := (aluF .add (n * 16) (n * 16)).2, rip := b4.rip + 3 } :=
    step_alu_add (by rw [hb4rip]; exact hc.d10) hb4r2 hb4r2
  obtain ⟨b5, hb5⟩ : ∃ u : X86Int.St, u = { b4.set 2 (some (aluF .add (n * 16) (n * 16)).1) with
    fl := (aluF .add (n * 16) (n * 16)).2, rip := b4.rip + 3 } := ⟨_, rfl⟩
  have hb5rip : b5.rip = entry + 40 := by rw [hb5]; show b4.rip + 3 = entry + 40; rw [hb4rip]
  have hb5r2 : b5.reg 2 = some (n * 32) := by
    rw [hb5]
    show (b4.set 2 (some (aluF .add (n * 16) (n * 16)).1)).reg 2 = some (n * 32)
    rw [reg_set_self (len_of_r9 hb4r9 (by decide)), hdbl5]
  have hb5r8 : b5.reg 8 = some h.d := by
    rw [hb5]
    show (b4.set 2 (some (aluF .add (n * 16) (n * 16)).1)).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact hb4r8
  have hb5r9 : b5.reg 9 = some h.a := by
    rw [hb5]
    show (b4.set 2 (some (aluF .add (n * 16) (n * 16)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact hb4r9
  have hb5r6 : b5.reg 6 = some c := by
    rw [hb5]
    show (b4.set 2 (some (aluF .add (n * 16) (n * 16)).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact hb4r6
  have hb5r4 : b5.reg 4 = some sp := by
    rw [hb5]
    show (b4.set 2 (some (aluF .add (n * 16) (n * 16)).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact hb4r4
  have hb5r14 : b5.reg 14 = some tbl := by
    rw [hb5]
    show (b4.set 2 (some (aluF .add (n * 16) (n * 16)).1)).reg 14 = some tbl
    rw [reg_set_other (by decide)]; exact hb4r14
  have hb5mem : b5.mem = m1 := by
    rw [hb5]
    show (b4.set 2 (some (aluF .add (n * 16) (n * 16)).1)).mem = m1
    rw [mem_set]; exact hb4mem
  have gb5 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → b5.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10 h11
    rw [hb5]
    show (b4.set 2 (some (aluF .add (n * 16) (n * 16)).1)).reg i = s.reg i
    rw [reg_set_other h2]; exact gb4 i h0 h2 h4 h6 h8 h10 h11
  -- add rdx, rdx (6)
  have hdbl6 : (aluF .add (n * 32) (n * 32)).1 = n * 64 := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt (by omega)]; omega
  have eb6 : X86Int.step im b5 = .ok { b5.set 2 (some (aluF .add (n * 32) (n * 32)).1) with
      fl := (aluF .add (n * 32) (n * 32)).2, rip := b5.rip + 3 } :=
    step_alu_add (by rw [hb5rip]; exact hc.d11) hb5r2 hb5r2
  obtain ⟨b6, hb6⟩ : ∃ u : X86Int.St, u = { b5.set 2 (some (aluF .add (n * 32) (n * 32)).1) with
    fl := (aluF .add (n * 32) (n * 32)).2, rip := b5.rip + 3 } := ⟨_, rfl⟩
  have hb6rip : b6.rip = entry + 43 := by rw [hb6]; show b5.rip + 3 = entry + 43; rw [hb5rip]
  have hb6r2 : b6.reg 2 = some (n * 64) := by
    rw [hb6]
    show (b5.set 2 (some (aluF .add (n * 32) (n * 32)).1)).reg 2 = some (n * 64)
    rw [reg_set_self (len_of_r9 hb5r9 (by decide)), hdbl6]
  have hb6r8 : b6.reg 8 = some h.d := by
    rw [hb6]
    show (b5.set 2 (some (aluF .add (n * 32) (n * 32)).1)).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact hb5r8
  have hb6r9 : b6.reg 9 = some h.a := by
    rw [hb6]
    show (b5.set 2 (some (aluF .add (n * 32) (n * 32)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact hb5r9
  have hb6r6 : b6.reg 6 = some c := by
    rw [hb6]
    show (b5.set 2 (some (aluF .add (n * 32) (n * 32)).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact hb5r6
  have hb6r4 : b6.reg 4 = some sp := by
    rw [hb6]
    show (b5.set 2 (some (aluF .add (n * 32) (n * 32)).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact hb5r4
  have hb6r14 : b6.reg 14 = some tbl := by
    rw [hb6]
    show (b5.set 2 (some (aluF .add (n * 32) (n * 32)).1)).reg 14 = some tbl
    rw [reg_set_other (by decide)]; exact hb5r14
  have hb6mem : b6.mem = m1 := by
    rw [hb6]
    show (b5.set 2 (some (aluF .add (n * 32) (n * 32)).1)).mem = m1
    rw [mem_set]; exact hb5mem
  have gb6 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → b6.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10 h11
    rw [hb6]
    show (b5.set 2 (some (aluF .add (n * 32) (n * 32)).1)).reg i = s.reg i
    rw [reg_set_other h2]; exact gb5 i h0 h2 h4 h6 h8 h10 h11
  -- add rdx, r14
  have haddt : (aluF .add (n * 64) tbl).1 = tbl + n * 64 := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt (by omega), Nat.add_comm]
  have eb7 : X86Int.step im b6 = .ok { b6.set 2 (some (aluF .add (n * 64) tbl).1) with
      fl := (aluF .add (n * 64) tbl).2, rip := b6.rip + 3 } :=
    step_alu_add (by rw [hb6rip]; exact hc.d12) hb6r2 hb6r14
  obtain ⟨b7, hb7⟩ : ∃ u : X86Int.St, u = { b6.set 2 (some (aluF .add (n * 64) tbl).1) with
    fl := (aluF .add (n * 64) tbl).2, rip := b6.rip + 3 } := ⟨_, rfl⟩
  have hb7rip : b7.rip = entry + 46 := by rw [hb7]; show b6.rip + 3 = entry + 46; rw [hb6rip]
  have hb7r2 : b7.reg 2 = some (tbl + n * 64) := by
    rw [hb7]
    show (b6.set 2 (some (aluF .add (n * 64) tbl).1)).reg 2 = some (tbl + n * 64)
    rw [reg_set_self (len_of_r9 hb6r9 (by decide)), haddt]
  have hb7r14 : b7.reg 14 = some tbl := by
    rw [hb7]
    show (b6.set 2 (some (aluF .add (n * 64) tbl).1)).reg 14 = some tbl
    rw [reg_set_other (by decide)]; exact hb6r14
  have hb7len : 14 < b7.regs.length := reg_some_len hb7r14
  have hb7r8 : b7.reg 8 = some h.d := by
    rw [hb7]
    show (b6.set 2 (some (aluF .add (n * 64) tbl).1)).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact hb6r8
  have hb7r9 : b7.reg 9 = some h.a := by
    rw [hb7]
    show (b6.set 2 (some (aluF .add (n * 64) tbl).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact hb6r9
  have hb7r6 : b7.reg 6 = some c := by
    rw [hb7]
    show (b6.set 2 (some (aluF .add (n * 64) tbl).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact hb6r6
  have hb7r4 : b7.reg 4 = some sp := by
    rw [hb7]
    show (b6.set 2 (some (aluF .add (n * 64) tbl).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact hb6r4
  have hb7mem : b7.mem = m1 := by
    rw [hb7]
    show (b6.set 2 (some (aluF .add (n * 64) tbl).1)).mem = m1
    rw [mem_set]; exact hb6mem
  have gb7 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → b7.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10 h11
    rw [hb7]
    show (b6.set 2 (some (aluF .add (n * 64) tbl).1)).reg i = s.reg i
    rw [reg_set_other h2]; exact gb6 i h0 h2 h4 h6 h8 h10 h11
  have hins1 : rd64 m1 (tbl + n * 64) = some cw.ins := by
    rw [show rd64 m1 (tbl + n * 64) = rd64 s.mem (tbl + n * 64) from
      rd64_wr64_other hm1s (by cases hdisjt with
        | inl hh => exact Or.inl (by omega)
        | inr hh => exact Or.inr (by omega))]
    exact hinsm
  have houts1 : rd64 m1 (tbl + n * 64 + 8) = some cw.outs := by
    rw [show rd64 m1 (tbl + n * 64 + 8) = rd64 s.mem (tbl + n * 64 + 8) from
      rd64_wr64_other hm1s (by cases hdisjt with
        | inl hh => exact Or.inl (by omega)
        | inr hh => exact Or.inr (by omega))]
    exact houtsm
  have hea0 : ea (tbl + n * 64) 0 = tbl + n * 64 := Nat.mod_eq_of_lt (by omega)
  have hea8 : ea (tbl + n * 64) 8 = tbl + n * 64 + 8 := Nat.mod_eq_of_lt (by omega)
  -- mov r10, [rdx]
  have eb8 : X86Int.step im b7 = .ok { b7.set 10 (some cw.ins) with rip := b7.rip + 3 } :=
    step_ld (by rw [hb7rip]; exact hc.d13) hb7r2 (by rw [hb7mem, hea0]; exact hins1)
  obtain ⟨b8, hb8⟩ : ∃ u : X86Int.St, u = { b7.set 10 (some cw.ins) with rip := b7.rip + 3 } := ⟨_, rfl⟩
  have hb8rip : b8.rip = entry + 49 := by rw [hb8]; show b7.rip + 3 = entry + 49; rw [hb7rip]
  have hb8r10 : b8.reg 10 = some cw.ins := by
    rw [hb8]
    show (b7.set 10 (some cw.ins)).reg 10 = some cw.ins
    rw [reg_set_self (by omega)]
  have hb8r2 : b8.reg 2 = some (tbl + n * 64) := by
    rw [hb8]
    show (b7.set 10 (some cw.ins)).reg 2 = some (tbl + n * 64)
    rw [reg_set_other (by decide)]; exact hb7r2
  have hb8r8 : b8.reg 8 = some h.d := by
    rw [hb8]
    show (b7.set 10 (some cw.ins)).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact hb7r8
  have hb8r9 : b8.reg 9 = some h.a := by
    rw [hb8]
    show (b7.set 10 (some cw.ins)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact hb7r9
  have hb8r6 : b8.reg 6 = some c := by
    rw [hb8]
    show (b7.set 10 (some cw.ins)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact hb7r6
  have hb8r4 : b8.reg 4 = some sp := by
    rw [hb8]
    show (b7.set 10 (some cw.ins)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact hb7r4
  have hb8mem : b8.mem = m1 := by
    rw [hb8]
    show (b7.set 10 (some cw.ins)).mem = m1
    rw [mem_set]; exact hb7mem
  have hb8len : b8.regs.length = b7.regs.length := by
    rw [hb8]; show (b7.set 10 (some cw.ins)).regs.length = b7.regs.length; rw [len_set]
  have gb8 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → b8.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10 h11
    rw [hb8]
    show (b7.set 10 (some cw.ins)).reg i = s.reg i
    rw [reg_set_other h10]; exact gb7 i h0 h2 h4 h6 h8 h10 h11
  -- cmp r8, r10
  have eb9 : X86Int.step im b8 = .ok { b8 with fl := (aluF .sub h.d cw.ins).2, rip := b8.rip + 3 } :=
    step_cmp (by rw [hb8rip]; exact hc.d14) hb8r8 hb8r10
  obtain ⟨b9, hb9⟩ : ∃ u : X86Int.St, u = { b8 with fl := (aluF .sub h.d cw.ins).2, rip := b8.rip + 3 } := ⟨_, rfl⟩
  have hb9rip : b9.rip = entry + 52 := by rw [hb9]; show b8.rip + 3 = entry + 52; rw [hb8rip]
  have hb9fl : X86Int.cond b9.fl 3 = some true := by
    have hcf : b9.fl.cf = some false := by
      rw [hb9]
      show (aluF .sub h.d cw.ins).2.cf = some false
      exact aluF_sub_cf (by omega)
    exact cond_cc3 hcf
  have hb9r2 : b9.reg 2 = some (tbl + n * 64) := by rw [hb9]; show b8.reg 2 = some (tbl + n * 64); exact hb8r2
  have hb9r10 : b9.reg 10 = some cw.ins := by rw [hb9]; show b8.reg 10 = some cw.ins; exact hb8r10
  have hb9r8 : b9.reg 8 = some h.d := by rw [hb9]; show b8.reg 8 = some h.d; exact hb8r8
  have hb9r9 : b9.reg 9 = some h.a := by rw [hb9]; show b8.reg 9 = some h.a; exact hb8r9
  have hb9r6 : b9.reg 6 = some c := by rw [hb9]; show b8.reg 6 = some c; exact hb8r6
  have hb9r4 : b9.reg 4 = some sp := by rw [hb9]; show b8.reg 4 = some sp; exact hb8r4
  have hb9mem : b9.mem = m1 := by rw [hb9]; show b8.mem = m1; exact hb8mem
  have hb9len : b9.regs.length = b7.regs.length := by
    rw [hb9]; show b8.regs.length = b7.regs.length; exact hb8len
  have gb9 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → b9.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10 h11
    rw [hb9]; show b8.reg i = s.reg i; exact gb8 i h0 h2 h4 h6 h8 h10 h11
  -- jae taken
  have eb10 : X86Int.step im b9 = .ok { b9 with rip := entry + 59 } := by
    have hh := step_jcc (im := im) (s := b9) (by rw [hb9rip]; exact hc.d15) hb9fl
    simpa using hh
  obtain ⟨b10, hb10⟩ : ∃ u : X86Int.St, u = { b9 with rip := entry + 59 } := ⟨_, rfl⟩
  have hb10rip : b10.rip = entry + 59 := by rw [hb10]
  have hb10r2 : b10.reg 2 = some (tbl + n * 64) := by
    rw [hb10]
    show ({ b9 with rip := entry + 59 }).reg 2 = some (tbl + n * 64)
    rw [reg_with_rip]; exact hb9r2
  have hb10r10 : b10.reg 10 = some cw.ins := by
    rw [hb10]
    show ({ b9 with rip := entry + 59 }).reg 10 = some cw.ins
    rw [reg_with_rip]; exact hb9r10
  have hb10r8 : b10.reg 8 = some h.d := by
    rw [hb10]
    show ({ b9 with rip := entry + 59 }).reg 8 = some h.d
    rw [reg_with_rip]; exact hb9r8
  have hb10r9 : b10.reg 9 = some h.a := by
    rw [hb10]
    show ({ b9 with rip := entry + 59 }).reg 9 = some h.a
    rw [reg_with_rip]; exact hb9r9
  have hb10r6 : b10.reg 6 = some c := by
    rw [hb10]
    show ({ b9 with rip := entry + 59 }).reg 6 = some c
    rw [reg_with_rip]; exact hb9r6
  have hb10r4 : b10.reg 4 = some sp := by
    rw [hb10]
    show ({ b9 with rip := entry + 59 }).reg 4 = some sp
    rw [reg_with_rip]; exact hb9r4
  have hb10mem : b10.mem = m1 := by
    rw [hb10]
    show ({ b9 with rip := entry + 59 }).mem = m1
    rw [mem_with_rip]; exact hb9mem
  have hb10len : b10.regs.length = b7.regs.length := by
    rw [hb10]; show b9.regs.length = b7.regs.length; exact hb9len
  have gb10 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → b10.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10 h11
    rw [hb10]
    show ({ b9 with rip := entry + 59 }).reg i = s.reg i
    rw [reg_with_rip]; exact gb9 i h0 h2 h4 h6 h8 h10 h11
  -- mov r11, [rdx+8]
  have eb11 : X86Int.step im b10 = .ok { b10.set 11 (some cw.outs) with rip := b10.rip + 4 } :=
    step_ld (by rw [hb10rip]; exact hc.d16) hb10r2 (by rw [hb10mem, hea8]; exact houts1)
  obtain ⟨b11, hb11⟩ : ∃ u : X86Int.St, u = { b10.set 11 (some cw.outs) with rip := b10.rip + 4 } := ⟨_, rfl⟩
  have hb11rip : b11.rip = entry + 63 := by rw [hb11]; show b10.rip + 4 = entry + 63; rw [hb10rip]
  have hb11r11 : b11.reg 11 = some cw.outs := by
    rw [hb11]
    show (b10.set 11 (some cw.outs)).reg 11 = some cw.outs
    rw [reg_set_self (by rw [hb10len]; omega)]
  have hb11r10 : b11.reg 10 = some cw.ins := by
    rw [hb11]
    show (b10.set 11 (some cw.outs)).reg 10 = some cw.ins
    rw [reg_set_other (by decide)]; exact hb10r10
  have hb11r8 : b11.reg 8 = some h.d := by
    rw [hb11]
    show (b10.set 11 (some cw.outs)).reg 8 = some h.d
    rw [reg_set_other (by decide)]; exact hb10r8
  have hb11r9 : b11.reg 9 = some h.a := by
    rw [hb11]
    show (b10.set 11 (some cw.outs)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact hb10r9
  have hb11r6 : b11.reg 6 = some c := by
    rw [hb11]
    show (b10.set 11 (some cw.outs)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact hb10r6
  have hb11r4 : b11.reg 4 = some sp := by
    rw [hb11]
    show (b10.set 11 (some cw.outs)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact hb10r4
  have hb11mem : b11.mem = m1 := by
    rw [hb11]
    show (b10.set 11 (some cw.outs)).mem = m1
    rw [mem_set]; exact hb10mem
  have gb11 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → b11.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10 h11
    rw [hb11]
    show (b10.set 11 (some cw.outs)).reg i = s.reg i
    rw [reg_set_other h11]; exact gb10 i h0 h2 h4 h6 h8 h10 h11
  -- sub r8, r10
  have hsubv : (aluF .sub h.d cw.ins).1 = h.d - cw.ins := by
    rw [aluF_sub_fst, show h.d + W - cw.ins = (h.d - cw.ins) + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have eb12 : X86Int.step im b11 = .ok { b11.set 8 (some (aluF .sub h.d cw.ins).1) with
      fl := (aluF .sub h.d cw.ins).2, rip := b11.rip + 3 } :=
    step_alu_sub (by rw [hb11rip]; exact hc.d17) hb11r8 hb11r10
  obtain ⟨b12, hb12⟩ : ∃ u : X86Int.St, u = { b11.set 8 (some (aluF .sub h.d cw.ins).1) with
    fl := (aluF .sub h.d cw.ins).2, rip := b11.rip + 3 } := ⟨_, rfl⟩
  have hb12rip : b12.rip = entry + 66 := by rw [hb12]; show b11.rip + 3 = entry + 66; rw [hb11rip]
  have hb12r8 : b12.reg 8 = some (h.d - cw.ins) := by
    rw [hb12]
    show (b11.set 8 (some (aluF .sub h.d cw.ins).1)).reg 8 = some (h.d - cw.ins)
    rw [reg_set_self (len_of_r9 hb11r9 (by decide)), hsubv]
  have hb12r11 : b12.reg 11 = some cw.outs := by
    rw [hb12]
    show (b11.set 8 (some (aluF .sub h.d cw.ins).1)).reg 11 = some cw.outs
    rw [reg_set_other (by decide)]; exact hb11r11
  have hb12r9 : b12.reg 9 = some h.a := by
    rw [hb12]
    show (b11.set 8 (some (aluF .sub h.d cw.ins).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact hb11r9
  have hb12r6 : b12.reg 6 = some c := by
    rw [hb12]
    show (b11.set 8 (some (aluF .sub h.d cw.ins).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact hb11r6
  have hb12r4 : b12.reg 4 = some sp := by
    rw [hb12]
    show (b11.set 8 (some (aluF .sub h.d cw.ins).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact hb11r4
  have hb12mem : b12.mem = m1 := by
    rw [hb12]
    show (b11.set 8 (some (aluF .sub h.d cw.ins).1)).mem = m1
    rw [mem_set]; exact hb11mem
  have gb12 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → b12.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10 h11
    rw [hb12]
    show (b11.set 8 (some (aluF .sub h.d cw.ins).1)).reg i = s.reg i
    rw [reg_set_other h8]; exact gb11 i h0 h2 h4 h6 h8 h10 h11
  -- add r8, r11
  have haddv : (aluF .add (h.d - cw.ins) cw.outs).1 = h.d - cw.ins + cw.outs := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt hbound]
  have eb13 : X86Int.step im b12 = .ok { b12.set 8 (some (aluF .add (h.d - cw.ins) cw.outs).1) with
      fl := (aluF .add (h.d - cw.ins) cw.outs).2, rip := b12.rip + 3 } :=
    step_alu_add (by rw [hb12rip]; exact hc.d18) hb12r8 hb12r11
  obtain ⟨b13, hb13⟩ : ∃ u : X86Int.St, u = { b12.set 8 (some (aluF .add (h.d - cw.ins) cw.outs).1) with
    fl := (aluF .add (h.d - cw.ins) cw.outs).2, rip := b12.rip + 3 } := ⟨_, rfl⟩
  have hb13rip : b13.rip = entry + 69 := by rw [hb13]; show b12.rip + 3 = entry + 69; rw [hb12rip]
  have hb13r8 : b13.reg 8 = some (h.d - cw.ins + cw.outs) := by
    rw [hb13]
    show (b12.set 8 (some (aluF .add (h.d - cw.ins) cw.outs).1)).reg 8 = some (h.d - cw.ins + cw.outs)
    rw [reg_set_self (len_of_r9 hb12r9 (by decide)), haddv]
  have hb13r9 : b13.reg 9 = some h.a := by
    rw [hb13]
    show (b12.set 8 (some (aluF .add (h.d - cw.ins) cw.outs).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact hb12r9
  have hb13r6 : b13.reg 6 = some c := by
    rw [hb13]
    show (b12.set 8 (some (aluF .add (h.d - cw.ins) cw.outs).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact hb12r6
  have hb13r4 : b13.reg 4 = some sp := by
    rw [hb13]
    show (b12.set 8 (some (aluF .add (h.d - cw.ins) cw.outs).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact hb12r4
  have hb13mem : b13.mem = m1 := by
    rw [hb13]
    show (b12.set 8 (some (aluF .add (h.d - cw.ins) cw.outs).1)).mem = m1
    rw [mem_set]; exact hb12mem
  have gb13 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → b13.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10 h11
    rw [hb13]
    show (b12.set 8 (some (aluF .add (h.d - cw.ins) cw.outs).1)).reg i = s.reg i
    rw [reg_set_other h8]; exact gb12 i h0 h2 h4 h6 h8 h10 h11
  -- add rsi, 5
  have hadd6 : (aluF .add c (5 % W)).1 = c + 5 := aluI_add_val h5W hcW
  have eb14 : X86Int.step im b13 = .ok { b13.set 6 (some (aluF .add c (5 % W)).1) with
      fl := (aluF .add c (5 % W)).2, rip := b13.rip + 4 } :=
    step_aluI_add (by rw [hb13rip]; exact hc.d19) hb13r6
  obtain ⟨b14, hb14⟩ : ∃ u : X86Int.St, u = { b13.set 6 (some (aluF .add c (5 % W)).1) with
    fl := (aluF .add c (5 % W)).2, rip := b13.rip + 4 } := ⟨_, rfl⟩
  have hb14rip : b14.rip = entry + 73 := by rw [hb14]; show b13.rip + 4 = entry + 73; rw [hb13rip]
  have hb14r6 : b14.reg 6 = some (c + 5) := by
    rw [hb14]
    show (b13.set 6 (some (aluF .add c (5 % W)).1)).reg 6 = some (c + 5)
    rw [reg_set_self (len_of_r9 hb13r9 (by decide)), hadd6]
  have hb14r8 : b14.reg 8 = some (h.d - cw.ins + cw.outs) := by
    rw [hb14]
    show (b13.set 6 (some (aluF .add c (5 % W)).1)).reg 8 = some (h.d - cw.ins + cw.outs)
    rw [reg_set_other (by decide)]; exact hb13r8
  have hb14r9 : b14.reg 9 = some h.a := by
    rw [hb14]
    show (b13.set 6 (some (aluF .add c (5 % W)).1)).reg 9 = some h.a
    rw [reg_set_other (by decide)]; exact hb13r9
  have hb14r4 : b14.reg 4 = some sp := by
    rw [hb14]
    show (b13.set 6 (some (aluF .add c (5 % W)).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact hb13r4
  have hb14mem : b14.mem = m1 := by
    rw [hb14]
    show (b13.set 6 (some (aluF .add c (5 % W)).1)).mem = m1
    rw [mem_set]; exact hb13mem
  have gb14 : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 8 → i ≠ 10 → i ≠ 11 → b14.reg i = s.reg i := by
    intro i h0 h2 h4 h6 h8 h10 h11
    rw [hb14]
    show (b13.set 6 (some (aluF .add c (5 % W)).1)).reg i = s.reg i
    rw [reg_set_other h6]; exact gb13 i h0 h2 h4 h6 h8 h10 h11
  -- jmp fall_tail
  have eb15 : X86Int.step im b14 = .ok { b14 with rip := fall } :=
    step_jmp (by rw [hb14rip]; exact hc.d20)
  obtain ⟨b15, hb15⟩ : ∃ u : X86Int.St, u = { b14 with rip := fall } := ⟨_, rfl⟩
  have r1 : run im 1 (Out.ok s) = Out.ok a1 := by rw [run_succ_ok im 0 s, e1, ← ha1, run_zero]
  have r2 : run im 1 (Out.ok a1) = Out.ok a2 := by rw [run_succ_ok im 0 a1, e2, ← ha2, run_zero]
  have r3 : run im 1 (Out.ok a2) = Out.ok a3 := by rw [run_succ_ok im 0 a2, e3, ← ha3, run_zero]
  have r5 : run im 1 (Out.ok a4) = Out.ok a5 := by rw [run_succ_ok im 0 a4, e5, ← ha5, run_zero]
  have r6 : run im 1 (Out.ok a5) = Out.ok a6 := by rw [run_succ_ok im 0 a5, e6, ← ha6, run_zero]
  have r7 : run im 1 (Out.ok a6) = Out.ok a7 := by rw [run_succ_ok im 0 a6, e7, ← ha7, run_zero]
  have r8 : run im 1 (Out.ok a7) = Out.ok a8 := by rw [run_succ_ok im 0 a7, e8, ← ha8, run_zero]
  have rb1 : run im 1 (Out.ok a8) = Out.ok b1 := by rw [run_succ_ok im 0 a8, eb1, ← hb1, run_zero]
  have rb2 : run im 1 (Out.ok b1) = Out.ok b2 := by rw [run_succ_ok im 0 b1, eb2, ← hb2, run_zero]
  have rb3 : run im 1 (Out.ok b2) = Out.ok b3 := by rw [run_succ_ok im 0 b2, eb3, ← hb3, run_zero]
  have rb4 : run im 1 (Out.ok b3) = Out.ok b4 := by rw [run_succ_ok im 0 b3, eb4, ← hb4, run_zero]
  have rb5 : run im 1 (Out.ok b4) = Out.ok b5 := by rw [run_succ_ok im 0 b4, eb5, ← hb5, run_zero]
  have rb6 : run im 1 (Out.ok b5) = Out.ok b6 := by rw [run_succ_ok im 0 b5, eb6, ← hb6, run_zero]
  have rb7 : run im 1 (Out.ok b6) = Out.ok b7 := by rw [run_succ_ok im 0 b6, eb7, ← hb7, run_zero]
  have rb8 : run im 1 (Out.ok b7) = Out.ok b8 := by rw [run_succ_ok im 0 b7, eb8, ← hb8, run_zero]
  have rb9 : run im 1 (Out.ok b8) = Out.ok b9 := by rw [run_succ_ok im 0 b8, eb9, ← hb9, run_zero]
  have rb10 : run im 1 (Out.ok b9) = Out.ok b10 := by rw [run_succ_ok im 0 b9, eb10, ← hb10, run_zero]
  have rb11 : run im 1 (Out.ok b10) = Out.ok b11 := by rw [run_succ_ok im 0 b10, eb11, ← hb11, run_zero]
  have rb12 : run im 1 (Out.ok b11) = Out.ok b12 := by rw [run_succ_ok im 0 b11, eb12, ← hb12, run_zero]
  have rb13 : run im 1 (Out.ok b12) = Out.ok b13 := by rw [run_succ_ok im 0 b12, eb13, ← hb13, run_zero]
  have rb14 : run im 1 (Out.ok b13) = Out.ok b14 := by rw [run_succ_ok im 0 b13, eb14, ← hb14, run_zero]
  have rb15 : run im 1 (Out.ok b14) = Out.ok b15 := by rw [run_succ_ok im 0 b14, eb15, ← hb15, run_zero]
  refine ⟨3 + n8 + 19, b15, ?_, by rw [hb15], ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [run_add im (3 + n8) 19,
      show run im (3 + n8) (Out.ok s) = Out.ok a4 from by
        rw [run_add im 3 n8,
          show run im 3 (Out.ok s) = Out.ok a3 from by
            rw [run_add im 1 2, r1, run_add im 1 1, r2, r3],
          hrun8],
      run_add im 1 18, r5, run_add im 1 17, r6, run_add im 1 16, r7, run_add im 1 15, r8,
      run_add im 1 14, rb1, run_add im 1 13, rb2, run_add im 1 12, rb3, run_add im 1 11, rb4,
      run_add im 1 10, rb5, run_add im 1 9, rb6, run_add im 1 8, rb7, run_add im 1 7, rb8,
      run_add im 1 6, rb9, run_add im 1 5, rb10, run_add im 1 4, rb11, run_add im 1 3, rb12,
      run_add im 1 2, rb13, run_add im 1 1, rb14, rb15]
  · rw [hb15]
    show ({ b14 with rip := fall }).reg 8 = some (h.d - cw.ins + cw.outs)
    rw [reg_with_rip]; exact hb14r8
  · rw [hb15]
    show ({ b14 with rip := fall }).reg 9 = some h.a
    rw [reg_with_rip]; exact hb14r9
  · rw [hb15]
    show ({ b14 with rip := fall }).reg 6 = some (c + 5)
    rw [reg_with_rip]; exact hb14r6
  · rw [hb15]
    show ({ b14 with rip := fall }).reg 4 = some sp
    rw [reg_with_rip]; exact hb14r4
  · intro i h0 h2 h4 h6 h8 h10 h11
    rw [hb15]
    show ({ b14 with rip := fall }).reg i = s.reg i
    rw [reg_with_rip]; exact gb14 i h0 h2 h4 h6 h8 h10 h11
  · intro x m hx
    have hm15 : b15.mem = m1 := by
      rw [hb15]
      show ({ b14 with rip := fall }).mem = m1
      rw [mem_with_rip]; exact hb14mem
    rw [hm15]
    exact rdN_wr64_other hm1s (by cases hx with
      | inl hh => exact Or.inr (by omega)
      | inr hh => exact Or.inl (by omega))

end SporosX1
