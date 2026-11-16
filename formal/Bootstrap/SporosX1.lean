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
      ∀ i, i ≠ 0 → s'.reg i = s.reg i)

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

def ANNLSpec (im : Nat → Option (Ins × Nat)) (entry : Nat) : Prop :=
  ∀ (wd : Sporos.Word) (t : Nat), Triple im entry (entry + 0x32)
    (fun s => s.reg 12 = some t ∧ StackRoom s 2)
    (fun _s s' => ∀ e, s'.reg 0 = some e →
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
  refine ⟨1, { s.set 0 (some (leVal bs)) with rip := s.rip + 2 }, ?_, ?_, ?_, ?_, ?_, ?_⟩
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
  obtain ⟨n8, s8, hrun8, hs8rip, hs8r0, hs8r2, hs8mem, hs8other⟩ :=
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

end SporosX1
