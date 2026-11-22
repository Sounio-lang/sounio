/-
SporosCheckBinExec: one lemma per instruction form of spr1_check, in the shape
the loop proofs use (registers read through `St.reg`, the next state as a record
update), the input copy (`Buf`, `u32`), and the step tactic `xs`.
-/
import Bootstrap.SporosCheckBinCore

namespace SporosCheckBin

open X86Env X86Int

theorem W_eq : W = 2 ^ 64 := rfl

theorem reg_def (s : St) (j : Nat) : s.reg j = (s.regs[j]?).getD none := rfl

@[simp] theorem set_regs (s : St) (i : Nat) (v : Option Nat) : (s.set i v).regs = s.regs.set i v := rfl

/-! ### The instruction forms -/

section
variable {s : St} {len : Nat}

theorem exec_movI {d imm : Nat} :
    exec (.movI d imm) len s = .ok { s.set d (some (imm % W)) with rip := s.rip + len } := rfl

theorem exec_mov {d r v : Nat} (h : s.reg r = some v) :
    exec (.mov d r) len s = .ok { s.set d (some v) with rip := s.rip + len } := by
  simp [exec, h, lift, bind, Option.bind, pure]

theorem exec_mov32 {d r v : Nat} (h : s.reg r = some v) :
    exec (.mov32 d r) len s = .ok { s.set d (some (v % 2 ^ 32)) with rip := s.rip + len } := by
  simp [exec, h, lift, bind, Option.bind, pure]

theorem exec_alu {op : Alu} {d r a b : Nat} (ha : s.reg d = some a) (hb : s.reg r = some b) :
    exec (.alu op d r) len s =
      .ok { s.set d (some (aluF op a b).1) with fl := (aluF op a b).2, rip := s.rip + len } := by
  simp [exec, ha, hb, lift, bind, Option.bind, pure]

theorem exec_aluI {op : Alu} {d imm a : Nat} (ha : s.reg d = some a) :
    exec (.aluI op d imm) len s =
      .ok { s.set d (some (aluF op a (imm % W)).1) with fl := (aluF op a (imm % W)).2, rip := s.rip + len } := by
  simp [exec, ha, lift, bind, Option.bind, pure]

theorem exec_cmp {a b x y : Nat} (hx : s.reg a = some x) (hy : s.reg b = some y) :
    exec (.cmp a b) len s = .ok { s with fl := (aluF .sub x y).2, rip := s.rip + len } := by
  simp [exec, hx, hy, lift, bind, Option.bind, pure]

theorem exec_cmpI {a imm x : Nat} (hx : s.reg a = some x) :
    exec (.cmpI a imm) len s = .ok { s with fl := (aluF .sub x (imm % W)).2, rip := s.rip + len } := by
  simp [exec, hx, lift, bind, Option.bind, pure]

theorem exec_test {a b x y : Nat} (hx : s.reg a = some x) (hy : s.reg b = some y) :
    exec (.test a b) len s = .ok { s with fl := (aluF .and x y).2, rip := s.rip + len } := by
  simp [exec, hx, hy, lift, bind, Option.bind, pure]

theorem exec_shlI {d c a : Nat} (ha : s.reg d = some a) :
    exec (.shlI d c) len s =
      .ok { s.set d (some (a * 2 ^ (c % 64) % W)) with fl := Flags.undef, rip := s.rip + len } := by
  simp [exec, ha, lift, bind, Option.bind, pure]

theorem exec_shrI {d c a : Nat} (ha : s.reg d = some a) :
    exec (.shrI d c) len s =
      .ok { s.set d (some (a / 2 ^ (c % 64))) with fl := Flags.undef, rip := s.rip + len } := by
  simp [exec, ha, lift, bind, Option.bind, pure]

theorem exec_ld {d b disp a v : Nat} (ha : s.reg b = some a) (hv : rd64 s.mem (ea a disp) = some v) :
    exec (.ld d b disp) len s = .ok { s.set d (some v) with rip := s.rip + len } := by
  simp [exec, ha, hv, lift, bind, Option.bind, pure]

theorem exec_ld32 {d b disp a v : Nat} (ha : s.reg b = some a) (hv : rd32 s.mem (ea a disp) = some v) :
    exec (.ld32 d b disp) len s = .ok { s.set d (some v) with rip := s.rip + len } := by
  simp [exec, ha, hv, lift, bind, Option.bind, pure]

theorem exec_ld8 {d b disp a v : Nat} (ha : s.reg b = some a) (hv : s.mem.rd (ea a disp) = some v) :
    exec (.ld8 d b disp) len s = .ok { s.set d (some v) with rip := s.rip + len } := by
  simp [exec, ha, hv, lift, bind, Option.bind, pure]

theorem exec_lea {d b disp a : Nat} (ha : s.reg b = some a) :
    exec (.lea d b disp) len s = .ok { s.set d (some (ea a disp)) with rip := s.rip + len } := by
  simp [exec, ha, lift, bind, Option.bind, pure]

theorem exec_jmp {t : Nat} : exec (.jmp t) len s = .ok { s with rip := t } := rfl

theorem exec_jcc {cc t : Nat} {c : Bool} (h : X86Int.cond s.fl cc = some c) :
    exec (.jcc cc t) len s = .ok { s with rip := if c then t else s.rip + len } := by
  simp [exec, h, lift, bind, Option.bind, pure]

end


/-! ### The same, as rewrite rules for `simp` (register reads left as matches) -/

section
variable {s : St} {len : Nat}

theorem xmov {d r : Nat} : exec (.mov d r) len s =
    match s.reg r with | some v => .ok { s.set d (some v) with rip := s.rip + len } | none => .stuck := by
  cases h : s.reg r <;> simp [exec, h, lift, bind, Option.bind, pure]

theorem xmov32 {d r : Nat} : exec (.mov32 d r) len s =
    match s.reg r with
    | some v => .ok { s.set d (some (v % 2 ^ 32)) with rip := s.rip + len } | none => .stuck := by
  cases h : s.reg r <;> simp [exec, h, lift, bind, Option.bind, pure]

theorem xalu {op : Alu} {d r : Nat} : exec (.alu op d r) len s =
    match s.reg d, s.reg r with
    | some a, some b => .ok { s.set d (some (aluF op a b).1) with fl := (aluF op a b).2, rip := s.rip + len }
    | _, _ => .stuck := by
  cases h1 : s.reg d <;> cases h2 : s.reg r <;> simp [exec, h1, h2, lift, bind, Option.bind, pure]

theorem xaluI {op : Alu} {d imm : Nat} : exec (.aluI op d imm) len s =
    match s.reg d with
    | some a => .ok { s.set d (some (aluF op a (imm % W)).1) with fl := (aluF op a (imm % W)).2, rip := s.rip + len }
    | none => .stuck := by
  cases h1 : s.reg d <;> simp [exec, h1, lift, bind, Option.bind, pure]

theorem xcmp {a b : Nat} : exec (.cmp a b) len s =
    match s.reg a, s.reg b with
    | some x, some y => .ok { s with fl := (aluF .sub x y).2, rip := s.rip + len }
    | _, _ => .stuck := by
  cases h1 : s.reg a <;> cases h2 : s.reg b <;> simp [exec, h1, h2, lift, bind, Option.bind, pure]

theorem xcmpI {a imm : Nat} : exec (.cmpI a imm) len s =
    match s.reg a with
    | some x => .ok { s with fl := (aluF .sub x (imm % W)).2, rip := s.rip + len }
    | none => .stuck := by
  cases h1 : s.reg a <;> simp [exec, h1, lift, bind, Option.bind, pure]

theorem xtest {a b : Nat} : exec (.test a b) len s =
    match s.reg a, s.reg b with
    | some x, some y => .ok { s with fl := (aluF .and x y).2, rip := s.rip + len }
    | _, _ => .stuck := by
  cases h1 : s.reg a <;> cases h2 : s.reg b <;> simp [exec, h1, h2, lift, bind, Option.bind, pure]

theorem xshlI {d c : Nat} : exec (.shlI d c) len s =
    match s.reg d with
    | some a => .ok { s.set d (some (a * 2 ^ (c % 64) % W)) with fl := Flags.undef, rip := s.rip + len }
    | none => .stuck := by
  cases h1 : s.reg d <;> simp [exec, h1, lift, bind, Option.bind, pure]

theorem xshrI {d c : Nat} : exec (.shrI d c) len s =
    match s.reg d with
    | some a => .ok { s.set d (some (a / 2 ^ (c % 64))) with fl := Flags.undef, rip := s.rip + len }
    | none => .stuck := by
  cases h1 : s.reg d <;> simp [exec, h1, lift, bind, Option.bind, pure]

theorem xld {d b disp : Nat} : exec (.ld d b disp) len s =
    match s.reg b with
    | some a => (match rd64 s.mem (ea a disp) with
      | some v => .ok { s.set d (some v) with rip := s.rip + len } | none => .stuck)
    | none => .stuck := by
  cases h1 : s.reg b with
  | none => simp [exec, h1, lift, bind, Option.bind, pure]
  | some a => cases h2 : rd64 s.mem (ea a disp) <;> simp [exec, h1, h2, lift, bind, Option.bind, pure]

theorem xld32 {d b disp : Nat} : exec (.ld32 d b disp) len s =
    match s.reg b with
    | some a => (match rd32 s.mem (ea a disp) with
      | some v => .ok { s.set d (some v) with rip := s.rip + len } | none => .stuck)
    | none => .stuck := by
  cases h1 : s.reg b with
  | none => simp [exec, h1, lift, bind, Option.bind, pure]
  | some a => cases h2 : rd32 s.mem (ea a disp) <;> simp [exec, h1, h2, lift, bind, Option.bind, pure]

theorem xld8 {d b disp : Nat} : exec (.ld8 d b disp) len s =
    match s.reg b with
    | some a => (match s.mem.rd (ea a disp) with
      | some v => .ok { s.set d (some v) with rip := s.rip + len } | none => .stuck)
    | none => .stuck := by
  cases h1 : s.reg b with
  | none => simp [exec, h1, lift, bind, Option.bind, pure]
  | some a => cases h2 : s.mem.rd (ea a disp) <;> simp [exec, h1, h2, lift, bind, Option.bind, pure]

theorem xlea {d b disp : Nat} : exec (.lea d b disp) len s =
    match s.reg b with
    | some a => .ok { s.set d (some (ea a disp)) with rip := s.rip + len } | none => .stuck := by
  cases h1 : s.reg b <;> simp [exec, h1, lift, bind, Option.bind, pure]

theorem xjcc {cc t : Nat} : exec (.jcc cc t) len s =
    match X86Int.cond s.fl cc with
    | some c => .ok { s with rip := if c then t else s.rip + len } | none => .stuck := by
  cases h1 : X86Int.cond s.fl cc <;> simp [exec, h1, lift, bind, Option.bind, pure]

end

theorem reach_at {s : St} {a : Nat} {ins : Ins} {len : Nat} (hf : im a = some (ins, len)) (ha : s.rip = a)
    {P : St → Prop} (h : match exec ins len s with | .ok s' => Reaches s' P | _ => False) :
    Reaches s P := reach_step (by rw [ha]; exact hf) h

@[simp] theorem regs_mk (regs : List (Option Nat)) (rip : Nat) (fl : Flags) (mem : Mem) (k : Kern) :
    (St.mk regs rip fl mem k).regs = regs := rfl
@[simp] theorem rip_mk (regs : List (Option Nat)) (rip : Nat) (fl : Flags) (mem : Mem) (k : Kern) :
    (St.mk regs rip fl mem k).rip = rip := rfl
@[simp] theorem fl_mk (regs : List (Option Nat)) (rip : Nat) (fl : Flags) (mem : Mem) (k : Kern) :
    (St.mk regs rip fl mem k).fl = fl := rfl
@[simp] theorem mem_mk (regs : List (Option Nat)) (rip : Nat) (fl : Flags) (mem : Mem) (k : Kern) :
    (St.mk regs rip fl mem k).mem = mem := rfl
@[simp] theorem k_mk (regs : List (Option Nat)) (rip : Nat) (fl : Flags) (mem : Mem) (k : Kern) :
    (St.mk regs rip fl mem k).k = k := rfl

@[simp] theorem reg_mk_self (s : St) (rip : Nat) (fl : Flags) (mem : Mem) (k : Kern) (j : Nat) :
    (St.mk s.regs rip fl mem k).reg j = s.reg j := rfl
@[simp] theorem reg_mk_set_eq (regs : List (Option Nat)) (rip : Nat) (fl : Flags) (mem : Mem) (k : Kern)
    {i : Nat} {v : Option Nat} (h : i < regs.length) :
    (St.mk (regs.set i v) rip fl mem k).reg i = v := by simp [St.reg, h]
@[simp] theorem reg_mk_set_ne (regs : List (Option Nat)) (rip : Nat) (fl : Flags) (mem : Mem) (k : Kern)
    {i j : Nat} {v : Option Nat} (h : i ≠ j) :
    (St.mk (regs.set i v) rip fl mem k).reg j = (St.mk regs rip fl mem k).reg j := by simp [St.reg, h]

/-- One instruction: fetch it with `f` (its address must equal rip), execute it,
and normalise the state; `hs` are the facts (register values, memory reads,
small arithmetic) that resolve the reads. -/
syntax "xs " term:max (" using " Lean.Parser.Tactic.simpArg,*)? : tactic
macro_rules
  | `(tactic| xs $f) =>
    `(tactic| (refine reach_at $f (by first | rfl | omega | (dsimp only; omega)) ?_;
               simp (config := { decide := true }) only [exec_movI, xmov, xmov32, xalu, xaluI, xcmp, xcmpI, xtest, xshlI, xshrI, xld,
                 xld32, xld8, xlea, exec_jmp, xjcc, reg_set_eq, reg_set_ne, set_regs, set_regs_length,
                 regs_mk, rip_mk, fl_mk, mem_mk, k_mk, reg_mk_self, reg_mk_set_eq, reg_mk_set_ne, List.length_set,
                 set_rip, set_mem, set_k, set_fl,
                 ↓reduceIte, reduceCtorEq, W_eq, Nat.reducePow, Nat.reduceMod, decide_true, decide_false,
                 Bool.false_eq_true, Bool.true_eq_false, decide_eq_true_eq, decide_eq_false_iff_not,
                 ne_eq, not_false_eq_true]))
  | `(tactic| xs $f using $hs,*) =>
    `(tactic| (refine reach_at $f (by first | rfl | omega | (dsimp only; omega)) ?_;
               simp (config := { decide := true }) only [exec_movI, xmov, xmov32, xalu, xaluI, xcmp, xcmpI, xtest, xshlI, xshrI, xld,
                 xld32, xld8, xlea, exec_jmp, xjcc, reg_set_eq, reg_set_ne, set_regs, set_regs_length,
                 regs_mk, rip_mk, fl_mk, mem_mk, k_mk, reg_mk_self, reg_mk_set_eq, reg_mk_set_ne, List.length_set,
                 set_rip, set_mem, set_k, set_fl,
                 ↓reduceIte, reduceCtorEq, W_eq, Nat.reducePow, Nat.reduceMod, decide_true, decide_false,
                 Bool.false_eq_true, Bool.true_eq_false, decide_eq_true_eq, decide_eq_false_iff_not,
                 ne_eq, not_false_eq_true, $hs,*]))

/-- One step: the fetch fact `f` at an address equal to rip, and the instruction's
outcome `e` as computed by one of the `exec_*` lemmas. -/
theorem reach_x {s : St} {a : Nat} {ins : Ins} {len : Nat} {s1 : St} (f : im a = some (ins, len))
    (ha : s.rip = a) (e : exec ins len s = .ok s1) {P : St → Prop} (h : Reaches s1 P) : Reaches s P :=
  reach_step (by rw [ha]; exact f) (by rw [e]; exact h)

/-! ### Register-file and arithmetic simplification -/

/-- Normal form for register reads of a state built by updates. -/
syntax "rsimp" (" using " Lean.Parser.Tactic.simpArg,*)? : tactic
macro_rules
  | `(tactic| rsimp) =>
    `(tactic| simp (config := { decide := true }) only [reg_set_eq, reg_set_ne, set_regs, set_regs_length,
        regs_mk, rip_mk, fl_mk, mem_mk, k_mk, reg_mk_self, reg_mk_set_eq, reg_mk_set_ne, List.length_set,
        set_rip, set_mem, set_k, set_fl, Option.some.injEq, W_eq, Nat.reducePow, Nat.reduceMod] at *)
  | `(tactic| rsimp using $hs,*) =>
    `(tactic| simp (config := { decide := true }) only [reg_set_eq, reg_set_ne, set_regs, set_regs_length,
        regs_mk, rip_mk, fl_mk, mem_mk, k_mk, reg_mk_self, reg_mk_set_eq, reg_mk_set_ne, List.length_set,
        set_rip, set_mem, set_k, set_fl, Option.some.injEq, W_eq, Nat.reducePow, Nat.reduceMod, $hs,*] at *)

theorem ea_small {a d : Nat} (h : a + d < 2 ^ 64) : ea a d = a + d := Nat.mod_eq_of_lt h

theorem add_small {a d : Nat} (h : a + d < 2 ^ 64) : (aluF .add a d).1 = a + d := Nat.mod_eq_of_lt h

theorem sub_small {a d : Nat} (ha : a < 2 ^ 64) (h : d ≤ a) : (aluF .sub a d).1 = a - d := by
  simp only [aluF, W_eq]
  rw [show a + 2 ^ 64 - d = (a - d) + 2 ^ 64 by omega, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]

/-! ### Reading the input copy -/

/-- The input copy: byte `i` of `bs` at address `b + i`. -/
def Buf (m : Mem) (b : Nat) (bs : List Nat) : Prop := ∀ i (h : i < bs.length), m.rd (b + i) = some bs[i]

/-- Little-endian `u32` at `pos` (zero past the end). -/
def u32 (bs : List Nat) (pos : Nat) : Nat := leVal ((bs.drop pos).take 4)

theorem rdN_buf {m : Mem} {b : Nat} {bs : List Nat} (hb : Buf m b bs) :
    ∀ (n pos : Nat), pos + n ≤ bs.length → m.rdN (b + pos) n = some ((bs.drop pos).take n)
  | 0, pos, _ => by simp [Mem.rdN]
  | n + 1, pos, h => by
    have h1 : pos < bs.length := by omega
    have ih := rdN_buf hb n (pos + 1) (by omega)
    rw [show b + (pos + 1) = b + pos + 1 by omega] at ih
    simp only [Mem.rdN, hb pos h1, ih, bind, Option.bind, pure]
    rw [List.drop_eq_getElem_cons h1]
    rfl

theorem rd32_buf {m : Mem} {b : Nat} {bs : List Nat} (hb : Buf m b bs) {pos : Nat}
    (h : pos + 4 ≤ bs.length) : rd32 m (b + pos) = some (u32 bs pos) := by
  simp [rd32, rdN_buf hb 4 pos h, u32]

theorem leVal_lt : ∀ (l : List Nat), (∀ x ∈ l, x < 256) → leVal l < 256 ^ l.length
  | [], _ => by simp [leVal]
  | x :: l, h => by
    have hx := h x (List.mem_cons_self ..)
    have ih := leVal_lt l (fun y hy => h y (List.mem_cons_of_mem _ hy))
    simp only [leVal, List.length_cons, Nat.pow_succ]
    have : 256 * leVal l ≤ 256 * (256 ^ l.length - 1) := Nat.mul_le_mul_left _ (by omega)
    have h2 : 0 < 256 ^ l.length := Nat.pow_pos (by decide)
    rw [Nat.mul_sub, Nat.mul_one] at this
    rw [Nat.mul_comm]
    omega

theorem u32_lt {bs : List Nat} (hbs : ∀ x ∈ bs, x < 256) (pos : Nat) : u32 bs pos < 2 ^ 32 := by
  have h := leVal_lt ((bs.drop pos).take 4)
    (fun x hx => hbs x (List.mem_of_mem_drop (List.mem_of_mem_take hx)))
  have hl : ((bs.drop pos).take 4).length ≤ 4 := by
    simp only [List.length_take]; exact Nat.min_le_left _ _
  have : 256 ^ ((bs.drop pos).take 4).length ≤ 256 ^ 4 := Nat.pow_le_pow_right (by decide) hl
  unfold u32
  have : (256 : Nat) ^ 4 = 2 ^ 32 := by decide
  omega

/-- Hypotheses common to the loops of a pass: the input copy `bs` at `b`, its
end in rbp, small enough that no address wraps. -/
structure Env (s : St) (b : Nat) (bs : List Nat) : Prop where
  buf : Buf s.mem b bs
  bytes : ∀ x ∈ bs, x < 256
  small : b + bs.length < 2 ^ 48
  len : s.regs.length = 16
  rbp : s.reg 5 = some (b + bs.length)

theorem frame_regs {S : List Nat} {s s' : St} (hr : ∀ j, j ∉ S → s'.regs[j]? = s.regs[j]?)
    (hl : s'.regs.length = s.regs.length) (hm : s'.mem = s.mem) (hk : s'.k = s.k) : Frame S s s' :=
  ⟨fun j hj => by simp only [reg_def, hr j hj], hl, hm, hk⟩

def RegsFrame (S : List Nat) (r r' : List (Option Nat)) : Prop :=
  ∀ j, j ∉ S → (r'[j]?).getD none = (r[j]?).getD none

theorem RegsFrame.refl (S : List Nat) (r : List (Option Nat)) : RegsFrame S r r := fun _ _ => rfl

theorem RegsFrame.set {S : List Nat} {r r' : List (Option Nat)} (h : RegsFrame S r r')
    {i : Nat} (hi : i ∈ S) (v : Option Nat) : RegsFrame S r (r'.set i v) := by
  intro j hj
  have hne : i ≠ j := by rintro rfl; exact hj hi
  simp only [List.getElem?_set_ne hne]
  exact h j hj

theorem Frame.of_mk {S : List Nat} {s : St} {regs : List (Option Nat)} {rip : Nat} {fl : Flags}
    (hr : RegsFrame S s.regs regs) (hl : regs.length = s.regs.length) :
    Frame S s ⟨regs, rip, fl, s.mem, s.k⟩ :=
  ⟨hr, hl, rfl, rfl⟩

/-- A frame: only the registers in the list were written. -/
macro "frame_tac" : tactic =>
  `(tactic| (try dsimp only;
             first
             | (apply Frame.of_mk
                · repeat (apply RegsFrame.set _ (by decide))
                  exact RegsFrame.refl _ _
                · simp only [List.length_set])
             | exact Frame.refl _ _))

theorem Frame.trans_same {S : List Nat} {s s1 s2 : St} (h1 : Frame S s s1) (h2 : Frame S s1 s2) :
    Frame S s s2 :=
  (h1.trans h2).mono (by intro x hx; simp only [List.mem_append] at hx; rcases hx with hx | hx <;> exact hx)

/-- Close a loop iteration: the frame of the iteration, then the rest. -/
theorem reach_frame {S : List Nat} {s s1 : St} {P : St → Prop} (h1 : Frame S s s1)
    (h : Reaches s1 (fun s' => Frame S s1 s' ∧ P s')) : Reaches s1 (fun s' => Frame S s s' ∧ P s') :=
  h.mono (fun _ h' => ⟨h1.trans_same h'.1, h'.2⟩)

theorem Env.frame {s s' : St} {b : Nat} {bs : List Nat} {S : List Nat} (h : Env s b bs)
    (hf : Frame S s s') (h5 : 5 ∉ S) : Env s' b bs :=
  ⟨by rw [hf.mem]; exact h.buf, h.bytes, h.small, by rw [hf.len]; exact h.len,
    by rw [hf.regs 5 h5]; exact h.rbp⟩

end SporosCheckBin
