/-
SporosCheckBinTail: end-to-end execution proofs for initialization (`initTable`),
triple-copy read loop, vote (`coreTable_36`), dual guards, and exit codes.
Composes `init_spec`, `p1_spec`, `p2_spec`, `p3_spec`, and `tail_spec` into
unconditional and good-oracle machine-code theorems against `checkFlatA`.
-/
import Bootstrap.SporosCheckBinPass
import Bootstrap.TMR

namespace SporosCheckBin

open X86Env X86Int

/-! ### Core-tail and init code lookup -/

private theorem mem_coreTable_tail {e : Nat × Ins × Nat} (he : e ∈ coreTable_36) :
    e ∈ coreTable := by
  unfold coreTable
  simp only [List.mem_append]
  exact Or.inr he

theorem core36_codeAt : CodeAt coreTable_36 :=
  fun _ he => table_ok _ (List.mem_append_left initTable (mem_coreTable_tail he))

theorem init_codeAt : CodeAt initTable :=
  fun _ he => table_ok _ (List.mem_append_right coreTable he)

theorem init0_codeAt : CodeAt initTable_0 :=
  fun _ he => init_codeAt _ (List.mem_append_left initTable_1 he)

theorem init1_codeAt : CodeAt initTable_1 :=
  fun _ he => init_codeAt _ (List.mem_append_right initTable_0 he)

private theorem codeAt_of_core36_sub {c : List (Nat × Ins × Nat)}
    (h : subOK c coreTable_36 = true) : CodeAt c :=
  fun e he => core36_codeAt e (subOK_sound c coreTable_36 h e he)

private theorem codeAt_of_init0_sub {c : List (Nat × Ins × Nat)}
    (h : subOK c initTable_0 = true) : CodeAt c :=
  fun e he => init0_codeAt e (subOK_sound c initTable_0 h e he)

private theorem codeAt_of_init1_sub {c : List (Nat × Ins × Nat)}
    (h : subOK c initTable_1 = true) : CodeAt c :=
  fun e he => init1_codeAt e (subOK_sound c initTable_1 h e he)

syntax "rsimpG" (" [" Lean.Parser.Tactic.simpArg,* "]")? : tactic
macro_rules
  | `(tactic| rsimpG) =>
    `(tactic| simp (config := { decide := true }) only [reg_set_eq, reg_set_ne, set_regs, set_regs_length,
        regs_mk, rip_mk, fl_mk, mem_mk, k_mk, reg_mk_self, reg_mk_set_eq, reg_mk_set_ne, List.length_set,
        set_rip, set_mem, set_k, set_fl, Option.some.injEq, W_eq, Nat.reducePow, Nat.reduceMod])
  | `(tactic| rsimpG [$hs,*]) =>
    `(tactic| simp (config := { decide := true }) only [reg_set_eq, reg_set_ne, set_regs, set_regs_length,
        regs_mk, rip_mk, fl_mk, mem_mk, k_mk, reg_mk_self, reg_mk_set_eq, reg_mk_set_ne, List.length_set,
        set_rip, set_mem, set_k, set_fl, Option.some.injEq, W_eq, Nat.reducePow, Nat.reduceMod, $hs,*])

/-! ### Register-definedness invariant (all 16 registers except r11) -/

def RegDefExcept11 (s : St) : Prop :=
  s.regs.length = 16 ∧ ∀ j, j < 16 → j ≠ 11 → (s.reg j).isSome

theorem regDef_get {s : St} (h : RegDefExcept11 s) (j : Nat)
    (hj : j < 16 := by decide) (hj11 : j ≠ 11 := by decide) :
    ∃ v, s.reg j = some v := by
  have hs := h.2 j hj hj11
  cases hr : s.reg j with
  | none => simp [hr] at hs
  | some v => exact ⟨v, rfl⟩

private theorem regDef_set_some {s : St} (h : RegDefExcept11 s) (d v : Nat) :
    RegDefExcept11 (s.set d (some v)) := by
  refine ⟨by simp [h.1], fun j hj hj11 => ?_⟩
  by_cases hjd : j = d
  · subst hjd; rw [reg_set_eq (by rw [h.1]; exact hj)]; rfl
  · rw [reg_set_ne (Ne.symm hjd)]; exact h.2 j hj hj11

private theorem regDef_set_11_none {s : St} (h : RegDefExcept11 s) :
    RegDefExcept11 (s.set 11 none) := by
  refine ⟨by simp [h.1], fun j hj hj11 => ?_⟩
  rw [reg_set_ne (Ne.symm hj11)]
  exact h.2 j hj hj11

theorem exec_preserves_regDefExcept11 {ins : Ins} {len : Nat} {s s' : St}
    (h : RegDefExcept11 s) (he : exec ins len s = .ok s') : RegDefExcept11 s' := by
  cases ins with
  | movI d imm =>
    simp only [exec, Out.ok.injEq] at he
    subst he; exact regDef_set_some h d _
  | mov d r =>
    simp only [exec] at he
    cases hr : s.reg r <;> simp [hr, lift, bind, Option.bind, pure] at he
    subst he; exact regDef_set_some h d _
  | mov32 d r =>
    simp only [exec] at he
    cases hr : s.reg r <;> simp [hr, lift, bind, Option.bind, pure] at he
    subst he; exact regDef_set_some h d _
  | alu op d r =>
    simp only [exec] at he
    cases hd : s.reg d <;> cases hr : s.reg r <;> simp [hd, hr, lift, bind, Option.bind, pure] at he
    subst he; exact regDef_set_some h d _
  | aluI op d imm =>
    simp only [exec] at he
    cases hd : s.reg d <;> simp [hd, lift, bind, Option.bind, pure] at he
    subst he; exact regDef_set_some h d _
  | cmp a b =>
    simp only [exec] at he
    cases ha : s.reg a <;> cases hb : s.reg b <;> simp [ha, hb, lift, bind, Option.bind, pure] at he
    subst he; exact h
  | cmpI a imm =>
    simp only [exec] at he
    cases ha : s.reg a <;> simp [ha, lift, bind, Option.bind, pure] at he
    subst he; exact h
  | test a b =>
    simp only [exec] at he
    cases ha : s.reg a <;> cases hb : s.reg b <;> simp [ha, hb, lift, bind, Option.bind, pure] at he
    subst he; exact h
  | shlI d c =>
    simp only [exec] at he
    cases hd : s.reg d <;> simp [hd, lift, bind, Option.bind, pure] at he
    subst he; exact regDef_set_some h d _
  | shrI d c =>
    simp only [exec] at he
    cases hd : s.reg d <;> simp [hd, lift, bind, Option.bind, pure] at he
    subst he; exact regDef_set_some h d _
  | ld d b disp =>
    simp only [exec] at he
    cases hb : s.reg b with
    | none => simp [hb, lift, bind, Option.bind] at he
    | some a =>
      cases hrd : rd64 s.mem (ea a disp) <;> simp [hb, hrd, lift, bind, Option.bind, pure] at he
      subst he; exact regDef_set_some h d _
  | st b disp r =>
    simp only [exec] at he
    cases hb : s.reg b with
    | none => simp [hb, lift, bind, Option.bind] at he
    | some a =>
      cases hr : s.reg r with
      | none => simp [hb, hr, lift, bind, Option.bind] at he
      | some v =>
        cases hwr : wr64 s.mem (ea a disp) v <;> simp [hb, hr, hwr, lift, bind, Option.bind, pure] at he
        subst he; exact h
  | ld32 d b disp =>
    simp only [exec] at he
    cases hb : s.reg b with
    | none => simp [hb, lift, bind, Option.bind] at he
    | some a =>
      cases hrd : rd32 s.mem (ea a disp) <;> simp [hb, hrd, lift, bind, Option.bind, pure] at he
      subst he; exact regDef_set_some h d _
  | ld8 d b disp =>
    simp only [exec] at he
    cases hb : s.reg b with
    | none => simp [hb, lift, bind, Option.bind] at he
    | some a =>
      cases hrd : s.mem.rd (ea a disp) <;> simp [hb, hrd, lift, bind, Option.bind, pure] at he
      subst he; exact regDef_set_some h d _
  | st8 b disp r =>
    simp only [exec] at he
    cases hb : s.reg b with
    | none => simp [hb, lift, bind, Option.bind] at he
    | some a =>
      cases hr : s.reg r with
      | none => simp [hb, hr, lift, bind, Option.bind] at he
      | some v =>
        cases hwr : s.mem.wr (ea a disp) (v % 256) <;> simp [hb, hr, hwr, lift, bind, Option.bind, pure] at he
        subst he; exact h
  | lea d b disp =>
    simp only [exec] at he
    cases hb : s.reg b <;> simp [hb, lift, bind, Option.bind, pure] at he
    subst he; exact regDef_set_some h d _
  | jmp t =>
    simp only [exec, Out.ok.injEq] at he
    subst he; exact h
  | jcc cc t =>
    simp only [exec] at he
    cases hc : X86Int.cond s.fl cc <;> simp [hc, lift, bind, Option.bind, pure] at he
    subst he; exact h
  | jtab b i disp =>
    simp only [exec] at he
    cases hb : s.reg b with
    | none => simp [hb, lift, bind, Option.bind] at he
    | some x =>
      cases hi : s.reg i with
      | none => simp [hb, hi, lift, bind, Option.bind] at he
      | some y =>
        cases hrd : rd64 s.mem ((x + 8 * y + disp) % W) <;> simp [hb, hi, hrd, lift, bind, Option.bind, pure] at he
        subst he; exact h
  | call t =>
    simp only [exec] at he
    cases hsp : s.reg 4 with
    | none => simp [hsp, lift, bind, Option.bind] at he
    | some sp =>
      cases hwr : wr64 s.mem ((sp + W - 8) % W) (s.rip + len) <;>
        simp [hsp, hwr, lift, bind, Option.bind, pure] at he
      subst he; exact regDef_set_some h 4 _
  | ret =>
    simp only [exec] at he
    cases hsp : s.reg 4 with
    | none => simp [hsp, lift, bind, Option.bind] at he
    | some sp =>
      cases hrd : rd64 s.mem sp <;> simp [hsp, hrd, lift, bind, Option.bind, pure] at he
      subst he; exact regDef_set_some h 4 _
  | jmpR r =>
    simp only [exec] at he
    cases hr : s.reg r <;> simp [hr, lift, bind, Option.bind, pure] at he
    subst he; exact h
  | pushI imm =>
    simp only [exec] at he
    cases hsp : s.reg 4 with
    | none => simp [hsp, lift, bind, Option.bind] at he
    | some sp =>
      cases hwr : wr64 s.mem ((sp + W - 8) % W) (imm % W) <;>
        simp [hsp, hwr, lift, bind, Option.bind, pure] at he
      subst he; exact regDef_set_some h 4 _
  | push r =>
    simp only [exec] at he
    cases hr : s.reg r with
    | none => simp [hr, lift, bind, Option.bind] at he
    | some v =>
      cases hsp : s.reg 4 with
      | none => simp [hr, hsp, lift, bind, Option.bind] at he
      | some sp =>
        cases hwr : wr64 s.mem ((sp + W - 8) % W) v <;>
          simp [hr, hsp, hwr, lift, bind, Option.bind, pure] at he
        subst he; exact regDef_set_some h 4 _
  | pop d =>
    simp only [exec] at he
    cases hsp : s.reg 4 with
    | none => simp [hsp, lift, bind, Option.bind] at he
    | some sp =>
      cases hrd : rd64 s.mem sp <;> simp [hsp, hrd, lift, bind, Option.bind, pure] at he
      subst he; exact regDef_set_some (regDef_set_some h 4 _) d _
  | syscall =>
    simp only [exec] at he
    split at he
    · split at he
      · cases he
        exact regDef_set_11_none (regDef_set_some (regDef_set_some h 0 _) 1 _)
      · cases he
      · cases he
    · cases he

theorem run_preserves_regDefExcept11 :
    ∀ {n : Nat} {s s' : St}, RegDefExcept11 s → run im n (.ok s) = .ok s' → RegDefExcept11 s'
  | 0, s, s', h, hr => by simp only [run, Out.ok.injEq] at hr; rw [← hr]; exact h
  | n + 1, s, s', h, hr => by
    simp only [run, step] at hr
    cases hf : im s.rip with
    | none => simp [hf] at hr; rw [run_stuck] at hr; cases hr
    | some p =>
      rcases p with ⟨ins, len⟩
      simp only [hf] at hr
      cases he : exec ins len s with
      | ok s1 =>
        simp only [he] at hr
        exact run_preserves_regDefExcept11 (exec_preserves_regDefExcept11 h he) hr
      | exit c => simp [he] at hr; rw [run_exit] at hr; cases hr
      | stuck => simp [he] at hr; rw [run_stuck] at hr; cases hr

theorem reaches_regDefExcept11 {s : St} {P : St → Prop} (hs : RegDefExcept11 s)
    (h : Reaches s P) : Reaches s (fun s' => P s' ∧ RegDefExcept11 s') := by
  rcases h with ⟨n, s', hr, hp⟩
  exact ⟨n, s', hr, hp, run_preserves_regDefExcept11 hs hr⟩

/-! ### Instruction and flag lemmas for tail and init -/

theorem xst {b disp r len : Nat} {s : St} : exec (.st b disp r) len s =
    match s.reg b, s.reg r with
    | some a, some v => (match wr64 s.mem (ea a disp) v with
      | some m => .ok { s with mem := m, rip := s.rip + len } | none => .stuck)
    | _, _ => .stuck := by
  cases hb : s.reg b with
  | none => simp [exec, hb, lift, bind, Option.bind]
  | some a =>
    cases hr : s.reg r with
    | none => simp [exec, hb, hr, lift, bind, Option.bind]
    | some v =>
      cases hw : wr64 s.mem (ea a disp) v <;> simp [exec, hb, hr, hw, lift, bind, Option.bind, pure]

theorem xst8 {b disp r len : Nat} {s : St} : exec (.st8 b disp r) len s =
    match s.reg b, s.reg r with
    | some a, some v => (match s.mem.wr (ea a disp) (v % 256) with
      | some m => .ok { s with mem := m, rip := s.rip + len } | none => .stuck)
    | _, _ => .stuck := by
  cases hb : s.reg b with
  | none => simp [exec, hb, lift, bind, Option.bind]
  | some a =>
    cases hr : s.reg r with
    | none => simp [exec, hb, hr, lift, bind, Option.bind]
    | some v =>
      cases hw : s.mem.wr (ea a disp) (v % 256) <;> simp [exec, hb, hr, hw, lift, bind, Option.bind, pure]

theorem exec_syscall_exit {len st a2 a3 a4 a5 a6 : Nat} {s : St}
    (h0 : s.reg 0 = some 60) (h7 : s.reg 7 = some st)
    (h6 : s.reg 6 = some a2) (h2 : s.reg 2 = some a3)
    (h10 : s.reg 10 = some a4) (h8 : s.reg 8 = some a5) (h9 : s.reg 9 = some a6) :
    exec .syscall len s = .exit (st % 256) := by
  simp only [exec, h0, h7, h6, h2, h10, h8, h9, syscall]

theorem ends_syscall_exit {a len st : Nat} {s : St}
    (hf : im a = some (.syscall, len)) (hrip : s.rip = a)
    (hd : RegDefExcept11 s) (h0 : s.reg 0 = some 60) (h7 : s.reg 7 = some st) :
    Ends s (· = .exit (st % 256)) := by
  obtain ⟨v6, h6⟩ := regDef_get hd 6
  obtain ⟨v2, h2⟩ := regDef_get hd 2
  obtain ⟨v10, h10⟩ := regDef_get hd 10
  obtain ⟨v8, h8⟩ := regDef_get hd 8
  obtain ⟨v9, h9⟩ := regDef_get hd 9
  refine ends_step (by rw [hrip]; exact hf) ?_
  rw [exec_syscall_exit h0 h7 h6 h2 h10 h8 h9]

theorem exec_syscall_ret {len nr a1 a2 a3 a4 a5 a6 rax : Nat} {k' : Kern} {m' : Mem} {s : St}
    (h0 : s.reg 0 = some nr) (h7 : s.reg 7 = some a1) (h6 : s.reg 6 = some a2)
    (h2 : s.reg 2 = some a3) (h10 : s.reg 10 = some a4) (h8 : s.reg 8 = some a5)
    (h9 : s.reg 9 = some a6)
    (hsys : syscall s.k s.mem nr a1 a2 a3 a4 a5 a6 = some (.ret rax k' m')) :
    exec .syscall len s = .ok { ((s.set 0 (some rax)).set 1 (some (s.rip + len))).set 11 none with
      k := k', mem := m', rip := s.rip + len } := by
  simp only [exec, h0, h7, h6, h2, h10, h8, h9, hsys]

theorem exec_syscall_none {len nr a1 a2 a3 a4 a5 a6 : Nat} {s : St}
    (h0 : s.reg 0 = some nr) (h7 : s.reg 7 = some a1) (h6 : s.reg 6 = some a2)
    (h2 : s.reg 2 = some a3) (h10 : s.reg 10 = some a4) (h8 : s.reg 8 = some a5)
    (h9 : s.reg 9 = some a6)
    (hsys : syscall s.k s.mem nr a1 a2 a3 a4 a5 a6 = none) :
    exec .syscall len s = .stuck := by
  simp only [exec, h0, h7, h6, h2, h10, h8, h9, hsys]

theorem cond_test_s {x : Nat} : X86Int.cond (aluF .and x x).2 8 = some (msb x) := by
  simp [X86Int.cond, aluF]

theorem msb_lt {x : Nat} (h : x < 2 ^ 63) : msb x = false := by
  simp [msb, Nat.div_eq_of_lt h]

theorem msb_nomem : msb (W - ENOMEM) = true := by decide

theorem cond_add_neg_acc {x : Nat} (hx : x < W) :
    X86Int.cond (aluF .add 11903883045027300202 x).2 5 = some (decide (x ≠ ACCEPT)) := by
  have hacc : ACCEPT = 6542861028682251414 := rfl
  rw [W_eq] at hx
  by_cases h : x = ACCEPT
  · subst h
    have hsum : (11903883045027300202 + ACCEPT) % W = 0 := by decide
    simp [X86Int.cond, aluF, hsum]
  · have hne : (11903883045027300202 + x) % W ≠ 0 := by
      rw [W_eq]
      have h' : x ≠ 6542861028682251414 := by rwa [← hacc]
      by_cases hlt : x < 6542861028682251414
      · rw [Nat.mod_eq_of_lt (by omega)]; omega
      · rw [show 11903883045027300202 + x = (x - 6542861028682251414) + 2 ^ 64 by omega,
          Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
        omega
    simp [X86Int.cond, aluF, beq_false_of_ne hne, h]

theorem cond_add_neg_ovf {x : Nat} (hx : x < W) :
    X86Int.cond (aluF .add 17605398745929497691 x).2 5 = some (decide (x ≠ OVERFLOW)) := by
  have hovf : OVERFLOW = 841345327780053925 := rfl
  rw [W_eq] at hx
  by_cases h : x = OVERFLOW
  · subst h
    have hsum : (17605398745929497691 + OVERFLOW) % W = 0 := by decide
    simp [X86Int.cond, aluF, hsum]
  · have hne : (17605398745929497691 + x) % W ≠ 0 := by
      rw [W_eq]
      have h' : x ≠ 841345327780053925 := by rwa [← hovf]
      by_cases hlt : x < 841345327780053925
      · rw [Nat.mod_eq_of_lt (by omega)]; omega
      · rw [show 17605398745929497691 + x = (x - 841345327780053925) + 2 ^ 64 by omega,
          Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
        omega
    simp [X86Int.cond, aluF, beq_false_of_ne hne, h]

theorem accept_exit_mod : ea ACCEPT 18446744073709551466 % 256 = 0 := by decide

/-! ### Voter, guard, and exit specifications (`L_vote .. L_core_end`) -/

def guardExit (a b : Nat) : Nat :=
  if a = ACCEPT ∧ b = ACCEPT then 0 else if a = OVERFLOW then 6 else 5

def tailExit (a b c : Nat) : Nat :=
  match TMR.vote a b c with
  | none => 3
  | some v => if v = ACCEPT then 0 else if v = OVERFLOW then 6 else 5

theorem tailExit_uniform (v : Nat) :
    tailExit v v v = if v = ACCEPT then 0 else if v = OVERFLOW then 6 else 5 := by
  simp [tailExit, TMR.vote]

theorem verdictWord_lt_W (r : Option Bool) : verdictWord r < W := by
  cases r with
  | none => decide
  | some b => cases b <;> decide

theorem nomem_spec {s : St} (hrip : s.rip = L_nomem) (hd : RegDefExcept11 s) :
    Ends s (· = .exit 6) := by
  have hc : CodeAt [(L_nomem, .movI 7 6, 5), (L_nomem + 5, .movI 0 60, 5),
      (L_nomem + 10, .movI 7 6, 5), (L_nomem + 15, .syscall, 2)] :=
    codeAt_of_core36_sub (by decide +kernel)
  simp only [CodeAt_cons, CodeAt_nil, and_true] at hc
  obtain ⟨f0, f5, f10, f15⟩ := hc
  refine Reaches.ends (P := fun s' => s'.rip = L_nomem + 15 ∧ RegDefExcept11 s' ∧
      s'.reg 0 = some 60 ∧ s'.reg 7 = some 6) ?_ ?_
  · xs f0
    xs f5
    xs f10
    exact Reaches.refl ⟨by dsimp only; omega,
      regDef_set_some (regDef_set_some (regDef_set_some hd 7 6) 0 60) 7 6,
      by rsimpG [hd.1], by rsimpG [hd.1]⟩
  · rintro s1 ⟨hrip1, hd1, h0, h7⟩
    simpa using ends_syscall_exit f15 hrip1 hd1 h0 h7

theorem rej_exit_spec {s : St} (hrip : s.rip = L_rej_exit) (hd : RegDefExcept11 s) :
    Ends s (· = .exit 5) := by
  have hc : CodeAt [(L_rej_exit, .movI 7 5, 5), (L_rej_exit + 5, .movI 0 60, 5),
      (L_rej_exit + 10, .movI 7 5, 5), (L_rej_exit + 15, .syscall, 2)] :=
    codeAt_of_core36_sub (by decide +kernel)
  simp only [CodeAt_cons, CodeAt_nil, and_true] at hc
  obtain ⟨f0, f5, f10, f15⟩ := hc
  refine Reaches.ends (P := fun s' => s'.rip = L_rej_exit + 15 ∧ RegDefExcept11 s' ∧
      s'.reg 0 = some 60 ∧ s'.reg 7 = some 5) ?_ ?_
  · xs f0
    xs f5
    xs f10
    exact Reaches.refl ⟨by dsimp only; omega,
      regDef_set_some (regDef_set_some (regDef_set_some hd 7 5) 0 60) 7 5,
      by rsimpG [hd.1], by rsimpG [hd.1]⟩
  · rintro s1 ⟨hrip1, hd1, h0, h7⟩
    simpa using ends_syscall_exit f15 hrip1 hd1 h0 h7

theorem not_acc_spec {a : Nat} (ha : a < W) {s : St} (hrip : s.rip = L_not_acc)
    (hd : RegDefExcept11 s) (h0 : s.reg 0 = some a) :
    Ends s (· = .exit (if a = OVERFLOW then 6 else 5)) := by
  have hc : CodeAt [(L_not_acc, .movI 7 5, 5), (L_not_acc + 5, .movI 2 17605398745929497691, 10),
      (L_not_acc + 15, .alu .add 2 0, 3), (L_not_acc + 18, .jcc 5 L_rej_exit, 2)] :=
    codeAt_of_core36_sub (by decide +kernel)
  simp only [CodeAt_cons, CodeAt_nil, and_true] at hc
  obtain ⟨f0, f5, f15, f18⟩ := hc
  have hc5 := cond_add_neg_ovf ha
  by_cases hovf : a = OVERFLOW
  · subst hovf
    have hc5_f : X86Int.cond (aluF .add 17605398745929497691 OVERFLOW).2 5 = some false := by
      simpa using hc5
    refine Reaches.ends (P := fun s' => s'.rip = L_nomem ∧ RegDefExcept11 s') ?_ ?_
    · xs f0
      xs f5
      xs f15 using hd.1, h0
      xs f18 using hc5_f
      exact Reaches.refl ⟨by dsimp only; rw [hrip]; decide,
        regDef_set_some (regDef_set_some (regDef_set_some hd 7 5) 2 _) 2 _⟩
    · rintro s1 ⟨hr1, hd1⟩
      simpa using nomem_spec hr1 hd1
  · have hc5_t : X86Int.cond (aluF .add 17605398745929497691 a).2 5 = some true := by
      simpa [hovf] using hc5
    refine Reaches.ends (P := fun s' => s'.rip = L_rej_exit ∧ RegDefExcept11 s') ?_ ?_
    · xs f0
      xs f5
      xs f15 using hd.1, h0
      xs f18 using hc5_t
      exact Reaches.refl ⟨by rfl,
        regDef_set_some (regDef_set_some (regDef_set_some hd 7 5) 2 _) 2 _⟩
    · rintro s1 ⟨hr1, hd1⟩
      simpa [hovf] using rej_exit_spec hr1 hd1

theorem guard_spec {a b : Nat} (ha : a < W) (hb : b < W) {s : St}
    (hrip : s.rip = L_guard) (hd : RegDefExcept11 s)
    (h0 : s.reg 0 = some a) (h3 : s.reg 3 = some b) :
    Ends s (· = .exit (guardExit a b)) := by
  have hc : CodeAt [(L_guard, .movI 7 5, 5), (L_guard + 5, .movI 2 11903883045027300202, 10),
      (L_guard + 15, .alu .add 2 0, 3), (L_guard + 18, .jcc 5 L_not_acc, 2),
      (L_guard + 20, .movI 6 11903883045027300202, 10), (L_guard + 30, .alu .add 6 3, 3),
      (L_guard + 33, .jcc 5 L_not_acc, 2), (L_guard + 35, .movI 0 60, 5),
      (L_guard + 40, .lea 7 3 18446744073709551466, 7), (L_guard + 47, .syscall, 2)] :=
    codeAt_of_core36_sub (by decide +kernel)
  simp only [CodeAt_cons, CodeAt_nil, and_true] at hc
  obtain ⟨f0, f5, f15, f18, f20, f30, f33, f35, f40, f47⟩ := hc
  have hca := cond_add_neg_acc ha
  have hcb := cond_add_neg_acc hb
  by_cases haa : a = ACCEPT
  · subst haa
    have hca_f : X86Int.cond (aluF .add 11903883045027300202 ACCEPT).2 5 = some false := by
      simpa using hca
    by_cases hbb : b = ACCEPT
    · subst hbb
      have hcb_f : X86Int.cond (aluF .add 11903883045027300202 ACCEPT).2 5 = some false := by
        simpa using hcb
      refine Reaches.ends (P := fun s' => s'.rip = L_guard + 47 ∧ RegDefExcept11 s' ∧
          s'.reg 0 = some 60 ∧ s'.reg 7 = some (ea ACCEPT 18446744073709551466)) ?_ ?_
      · xs f0
        xs f5
        xs f15 using hd.1, h0
        xs f18 using hca_f
        xs f20
        xs f30 using hd.1, h3
        xs f33 using hcb_f
        xs f35
        xs f40 using hd.1, h3
        refine Reaches.refl ⟨by dsimp only; omega, ?_, by rsimpG [hd.1], by rsimpG [hd.1]⟩
        exact regDef_set_some (regDef_set_some (regDef_set_some (regDef_set_some (regDef_set_some
          (regDef_set_some (regDef_set_some hd 7 5) 2 _) 2 _) 6 _) 6 _) 0 60) 7 _
      · rintro s1 ⟨hr1, hd1, h01, h71⟩
        have he := ends_syscall_exit f47 hr1 hd1 h01 h71
        simpa [guardExit, accept_exit_mod] using he
    · have hcb_t : X86Int.cond (aluF .add 11903883045027300202 b).2 5 = some true := by
        simpa [hbb] using hcb
      refine Reaches.ends (P := fun s' => s'.rip = L_not_acc ∧ RegDefExcept11 s' ∧
          s'.reg 0 = some ACCEPT) ?_ ?_
      · xs f0
        xs f5
        xs f15 using hd.1, h0
        xs f18 using hca_f
        xs f20
        xs f30 using hd.1, h3
        xs f33 using hcb_t
        refine Reaches.refl ⟨by rfl, ?_, by rsimpG [hd.1]; exact h0⟩
        exact regDef_set_some (regDef_set_some (regDef_set_some (regDef_set_some
          (regDef_set_some hd 7 5) 2 _) 2 _) 6 _) 6 _
      · rintro s1 ⟨hr1, hd1, h01⟩
        have he := not_acc_spec ha hr1 hd1 h01
        have hne : ACCEPT ≠ OVERFLOW := by decide
        simpa [guardExit, hbb, hne] using he
  · have hca_t : X86Int.cond (aluF .add 11903883045027300202 a).2 5 = some true := by
      simpa [haa] using hca
    refine Reaches.ends (P := fun s' => s'.rip = L_not_acc ∧ RegDefExcept11 s' ∧
        s'.reg 0 = some a) ?_ ?_
    · xs f0
      xs f5
      xs f15 using hd.1, h0
      xs f18 using hca_t
      refine Reaches.refl ⟨by rfl, ?_, by rsimpG [hd.1]; exact h0⟩
      exact regDef_set_some (regDef_set_some (regDef_set_some hd 7 5) 2 _) 2 _
    · rintro s1 ⟨hr1, hd1, h01⟩
      have he := not_acc_spec ha hr1 hd1 h01
      simpa [guardExit, haa] using he

theorem tail_spec {a b c : Nat} (ha : a < W) (hb : b < W) (hc : c < W) {s : St}
    (hrip : s.rip = L_vote) (hd : RegDefExcept11 s)
    (h0 : s.reg 0 = some a) (h3 : s.reg 3 = some b) (h1 : s.reg 1 = some c) :
    Ends s (· = .exit (tailExit a b c)) := by
  have hcode : CodeAt [(L_vote, .cmp 0 3, 3), (L_vote + 3, .jcc 4 (L_vote + 18), 2),
      (L_vote + 5, .cmp 0 1, 3), (L_vote + 8, .jcc 4 (L_vote + 18), 2),
      (L_vote + 10, .cmp 3 1, 3), (L_vote + 13, .jcc 5 (L_vote + 26), 2),
      (L_vote + 15, .mov 0 3, 3), (L_vote + 18, .mov 3 0, 3),
      (L_vote + 21, .mov 1 0, 3), (L_vote + 24, .jmp L_guard, 2),
      (L_vote + 26, .movI 7 3, 5), (L_vote + 31, .movI 0 60, 5), (L_vote + 36, .syscall, 2)] :=
    codeAt_of_core36_sub (by decide +kernel)
  simp only [CodeAt_cons, CodeAt_nil, and_true] at hcode
  obtain ⟨f0, f3, f5, f8, f10, f13, f15, f18, f21, f24, f26, f31, f36⟩ := hcode
  have hab_c := (cond_sub ha hb).2.2.1
  have hac_c := (cond_sub ha hc).2.2.1
  have hbc_c := (cond_sub hb hc).2.2.2.1
  by_cases hab : a = b
  · have hab_t : X86Int.cond (aluF .sub a b).2 4 = some true := by simpa [hab] using hab_c
    refine Reaches.ends (P := fun s' => s'.rip = L_guard ∧ RegDefExcept11 s' ∧
        s'.reg 0 = some a ∧ s'.reg 3 = some a) ?_ ?_
    · xs f0 using h0, h3
      xs f3 using hab_t
      xs f18 using h0
      xs f21 using h0
      xs f24
      exact Reaches.refl ⟨rfl, regDef_set_some (regDef_set_some hd 3 a) 1 a,
        by rsimpG [hd.1]; exact h0, by rsimpG [hd.1]⟩
    · rintro s1 ⟨hr1, hd1, h01, h31⟩
      have he := guard_spec ha ha hr1 hd1 h01 h31
      simpa [tailExit, guardExit, TMR.vote, hab] using he
  · have hab_f : X86Int.cond (aluF .sub a b).2 4 = some false := by simpa [hab] using hab_c
    by_cases hac : a = c
    · have hac_t : X86Int.cond (aluF .sub a c).2 4 = some true := by simpa [hac] using hac_c
      refine Reaches.ends (P := fun s' => s'.rip = L_guard ∧ RegDefExcept11 s' ∧
          s'.reg 0 = some a ∧ s'.reg 3 = some a) ?_ ?_
      · xs f0 using h0, h3
        xs f3 using hab_f
        xs f5 using h0, h1
        xs f8 using hac_t
        xs f18 using h0
        xs f21 using h0
        xs f24
        exact Reaches.refl ⟨rfl, regDef_set_some (regDef_set_some hd 3 a) 1 a,
          by rsimpG [hd.1]; exact h0, by rsimpG [hd.1]⟩
      · rintro s1 ⟨hr1, hd1, h01, h31⟩
        have he := guard_spec ha ha hr1 hd1 h01 h31
        simpa [tailExit, guardExit, TMR.vote, hab, hac] using he
    · have hac_f : X86Int.cond (aluF .sub a c).2 4 = some false := by simpa [hac] using hac_c
      by_cases hbc : b = c
      · have hbc_f : X86Int.cond (aluF .sub b c).2 5 = some false := by simpa [hbc] using hbc_c
        refine Reaches.ends (P := fun s' => s'.rip = L_guard ∧ RegDefExcept11 s' ∧
            s'.reg 0 = some b ∧ s'.reg 3 = some b) ?_ ?_
        · xs f0 using h0, h3
          xs f3 using hab_f
          xs f5 using h0, h1
          xs f8 using hac_f
          xs f10 using h3, h1
          xs f13 using hbc_f
          xs f15 using h3
          xs f18 using hd.1
          xs f21 using hd.1
          xs f24
          exact Reaches.refl ⟨rfl, regDef_set_some (regDef_set_some (regDef_set_some hd 0 b) 3 b) 1 b,
            by rsimpG [hd.1], by rsimpG [hd.1]⟩
        · rintro s1 ⟨hr1, hd1, h01, h31⟩
          have he := guard_spec hb hb hr1 hd1 h01 h31
          simpa [tailExit, guardExit, TMR.vote, hab, hac, hbc] using he
      · have hbc_t : X86Int.cond (aluF .sub b c).2 5 = some true := by simpa [hbc] using hbc_c
        refine Reaches.ends (P := fun s' => s'.rip = L_vote + 36 ∧ RegDefExcept11 s' ∧
            s'.reg 0 = some 60 ∧ s'.reg 7 = some 3) ?_ ?_
        · xs f0 using h0, h3
          xs f3 using hab_f
          xs f5 using h0, h1
          xs f8 using hac_f
          xs f10 using h3, h1
          xs f13 using hbc_t
          xs f26
          xs f31
          exact Reaches.refl ⟨rfl, regDef_set_some (regDef_set_some hd 7 3) 0 60,
            by rsimpG [hd.1], by rsimpG [hd.1]⟩
        · rintro s1 ⟨hr1, hd1, h01, h71⟩
          have he := ends_syscall_exit f36 hr1 hd1 h01 h71
          simpa [tailExit, TMR.vote, hab, hac, hbc] using he

/-! ### Memory region and write/read lemmas -/

private theorem mem_find_aux : ∀ (m : Mem) (i0 i : Nat) {r : Region} {a : Nat},
    m.Pairwise (fun r s => r.hi ≤ s.base ∨ s.hi ≤ r.base) →
    m[i]? = some r → r.base ≤ a → a < r.hi →
    Mem.find m a i0 = some (i0 + i, r)
  | [], _, _, _, _, _, hri, _, _ => by simp at hri
  | s :: rest, i0, i, r, a, hpw, hri, ha1, ha2 => by
    rw [List.pairwise_cons] at hpw
    cases i with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hri
      subst hri
      simp [Mem.find, ha1, ha2]
    | succ j =>
      simp only [List.getElem?_cons_succ] at hri
      have hmem : r ∈ rest := List.mem_of_getElem? hri
      have hdisj := hpw.1 r hmem
      have hnot : ¬ (s.base ≤ a ∧ a < s.hi) := by omega
      simp only [Mem.find, hnot, ↓reduceIte]
      have ih := mem_find_aux rest (i0 + 1) j hpw.2 hri ha1 ha2
      rwa [show i0 + 1 + j = i0 + (j + 1) by omega] at ih

theorem mem_find_of_wf {m : Mem} (hw : m.wf) {i : Nat} {r : Region}
    (hri : m[i]? = some r) {a : Nat} (ha1 : r.base ≤ a) (ha2 : a < r.hi) :
    Mem.find m a 0 = some (i, r) := by
  simpa using mem_find_aux m 0 i hw.2 hri ha1 ha2

theorem mem_rd_of_wf {m : Mem} (hw : m.wf) {i : Nat} {r : Region}
    (hri : m[i]? = some r) {a : Nat} (ha1 : r.base ≤ a) (ha2 : a < r.hi) :
    m.rd a = r.data[a - r.base]? := by
  simp [Mem.rd, mem_find_of_wf hw hri ha1 ha2]

theorem mem_wr_of_wf {m : Mem} (hw : m.wf) {i : Nat} {r : Region}
    (hri : m[i]? = some r) (hrw : r.rw = true) {a v : Nat} (ha1 : r.base ≤ a) (ha2 : a < r.hi) :
    m.wr a v = some (m.set i { r with data := r.data.set (a - r.base) (v % 256) }) := by
  simp [Mem.wr, mem_find_of_wf hw hri ha1 ha2, hrw]

private theorem list_set_eq_self : ∀ (m : Mem) (i : Nat) (r : Region),
    m[i]? = some r → m.set i r = m
  | [], _, _, h => by simp at h
  | x :: xs, 0, r, h => by simp only [List.getElem?_cons_zero, Option.some.injEq] at h; simp [h]
  | x :: xs, i + 1, r, h => by
    simp only [List.getElem?_cons_succ] at h
    simp [list_set_eq_self xs i r h]

private theorem mem_wrN_aux : ∀ (bs : List Nat) (a : Nat) (m : Mem), m.wf → ∀ (i : Nat) (r : Region),
    m[i]? = some r → r.rw = true → r.base ≤ a → a + bs.length ≤ r.hi →
    (∀ x ∈ bs, x < 256) →
    ∃ d : List Nat, m.wrN a bs = some (m.set i { r with data := d }) ∧
      d.length = r.data.length ∧
      (∀ k, k < r.data.length →
        d[k]? = if a - r.base ≤ k ∧ k < a - r.base + bs.length then bs[k - (a - r.base)]? else r.data[k]?) := by
  intro bs
  induction bs with
  | nil =>
    intro a m _ i r hri _ _ _ _
    refine ⟨r.data, ?_, rfl, ?_⟩
    · simp only [Mem.wrN, Option.some.injEq]
      exact (list_set_eq_self m i r hri).symm
    · intro k _
      simp only [List.length_nil, Nat.add_zero]
      rw [if_neg (by omega)]
  | cons b bs ih =>
    intro a m hw i r hri hrw ha1 ha2 hby
    have hb : b < 256 := hby b (List.mem_cons_self ..)
    have hmod : b % 256 = b := Nat.mod_eq_of_lt hb
    have ha_lt : a < r.hi := by simp only [Region.hi, List.length_cons] at ha2 ⊢; omega
    let r1 : Region := { r with data := r.data.set (a - r.base) (b % 256) }
    let m1 : Mem := m.set i r1
    have hwr1 : m.wr a b = some m1 := mem_wr_of_wf (v := b) hw hri hrw ha1 ha_lt
    have hw1 : m1.wf := wf_of_shape (wr_shape hwr1) hw
    have hi_lt : i < m.length := (List.getElem?_eq_some_iff.1 hri).1
    have hri1 : m1[i]? = some r1 := by simp [m1, List.getElem?_set_self hi_lt]
    have ha1' : r1.base ≤ a + 1 := by simp only [r1]; omega
    have ha2' : a + 1 + bs.length ≤ r1.hi := by
      simp only [r1, Region.hi, List.length_set] at ha2 ⊢
      simp only [List.length_cons] at ha2
      omega
    obtain ⟨d, hwr2, hlen, hd⟩ := ih (a + 1) m1 hw1 i r1 hri1 hrw ha1' ha2'
      (fun x hx => hby x (List.mem_cons_of_mem _ hx))
    refine ⟨d, ?_, by simpa [r1, List.length_set] using hlen, ?_⟩
    · simp only [Mem.wrN, hwr1, bind, Option.bind]
      change m1.wrN (a + 1) bs = _
      rw [hwr2]
      simp only [m1, List.set_set, r1]
    · intro k hk
      have hk1 : k < r1.data.length := by simpa [r1, List.length_set] using hk
      have hdk := hd k hk1
      rw [hdk]
      have hbase : r1.base = r.base := rfl
      rw [hbase]
      have hsub : a + 1 - r.base = (a - r.base) + 1 := by omega
      rw [hsub]
      by_cases hkab : a - r.base ≤ k ∧ k < a - r.base + (b :: bs).length
      · rw [if_pos hkab]
        by_cases hkeq : k = a - r.base
        · subst hkeq
          rw [if_neg (by omega)]
          simp [r1, List.getElem?_set_self (show a - r.base < r.data.length by
            simp only [Region.hi] at ha_lt; omega), hmod]
        · rw [if_pos (by simp only [List.length_cons] at hkab; omega)]
          have : k - (a - r.base) = (k - (a - r.base + 1)) + 1 := by omega
          rw [this, List.getElem?_cons_succ]
      · rw [if_neg hkab, if_neg (by simp only [List.length_cons] at hkab; omega)]
        simp [r1, List.getElem?_set_ne (show a - r.base ≠ k by
          intro h; apply hkab; simp only [List.length_cons]; omega)]

theorem mem_wrN_of_wf {bs : List Nat} {a : Nat} {m : Mem} (hw : m.wf) {i : Nat} {r : Region}
    (hri : m[i]? = some r) (hrw : r.rw = true) (ha1 : r.base ≤ a) (ha2 : a + bs.length ≤ r.hi)
    (hby : ∀ x ∈ bs, x < 256) :
    ∃ d : List Nat, m.wrN a bs = some (m.set i { r with data := d }) ∧
      d.length = r.data.length ∧
      (∀ k, k < r.data.length →
        d[k]? = if a - r.base ≤ k ∧ k < a - r.base + bs.length then bs[k - (a - r.base)]? else r.data[k]?) :=
  mem_wrN_aux bs a m hw i r hri hrw ha1 ha2 hby

theorem leVal_bytes8 {v : Nat} (hv : v < W) : leVal (bytes8 v) = v := by
  have hW : W = 2 ^ 64 := rfl
  simp only [bytes8, List.range_succ, List.range_zero, List.nil_append,
    List.map_cons, List.map_nil, List.cons_append, List.nil_append, leVal]
  simp only [Nat.pow_zero, Nat.div_one, Nat.pow_one]
  omega

theorem bytes8_lt_256 (v : Nat) : ∀ x ∈ bytes8 v, x < 256 := by
  intro x hx
  simp only [bytes8, List.mem_map] at hx
  obtain ⟨i, _, rfl⟩ := hx
  exact Nat.mod_lt _ (by decide)

theorem bytes8_len (v : Nat) : (bytes8 v).length = 8 := by simp [bytes8]

theorem rdN_of_wf {m : Mem} (hw : m.wf) {i : Nat} {r : Region} (hri : m[i]? = some r)
    {a : Nat} (ha1 : r.base ≤ a) :
    ∀ (n : Nat), a + n ≤ r.hi →
      m.rdN a n = some ((List.range n).map fun j => (r.data[a - r.base + j]?).getD 0)
  | 0, _ => by simp [Mem.rdN]
  | n + 1, ha2 => by
    have halt : a < r.hi := by simp only [Region.hi] at ha2 ⊢; omega
    have hrd := mem_rd_of_wf hw hri ha1 halt
    have hk : a - r.base < r.data.length := by simp only [Region.hi] at halt; omega
    rw [List.getElem?_eq_getElem hk] at hrd
    have ih := rdN_of_wf hw hri (show r.base ≤ a + 1 by omega) n (by simp only [Region.hi] at ha2 ⊢; omega)
    simp only [Mem.rdN, hrd, ih, bind, Option.bind, pure]
    congr 1
    rw [List.range_succ_eq_map, List.map_cons, List.map_map]
    simp only [Nat.add_zero, List.getElem?_eq_getElem hk, Option.getD_some]
    congr 1
    apply List.map_congr_left
    intro j _
    simp only [Function.comp_def, show a + 1 - r.base + j = a - r.base + (j + 1) by omega]

theorem wr64_region_step {m : Mem} (hw : m.wf) {ib : Nat} {rb : Region}
    (hrib : m[ib]? = some rb) (hrw : rb.rw = true) {a v : Nat}
    (ha1 : rb.base ≤ a) (ha2 : a + 8 ≤ rb.hi) :
    ∃ db : List Nat,
      wr64 m a v = some (m.set ib { rb with data := db }) ∧
      db.length = rb.data.length ∧
      (∀ k, k < rb.data.length →
        db[k]? = if a - rb.base ≤ k ∧ k < a - rb.base + 8 then (bytes8 v)[k - (a - rb.base)]?
          else rb.data[k]?) := by
  rw [wr64]
  exact mem_wrN_of_wf hw hrib hrw ha1 (by rw [bytes8_len]; exact ha2) (bytes8_lt_256 v)

theorem rd64_wr64_slot {m : Mem} (hw : m.wf) {ib : Nat} {rb : Region}
    (hrib : m[ib]? = some rb) {off v : Nat}
    (hoff : off + 8 ≤ rb.data.length) (hv : v < W)
    (hslice : ∀ j, j < 8 → rb.data[off + j]? = (bytes8 v)[j]?) :
    rd64 m (rb.base + off) = some v := by
  have h1 : rb.base ≤ rb.base + off := by omega
  have h2 : rb.base + off + 8 ≤ rb.hi := by simp only [Region.hi]; omega
  rw [rd64, rdN_of_wf hw hrib h1 8 h2]
  simp only [Nat.add_sub_cancel_left]
  have hlist : ((List.range 8).map fun j => (rb.data[off + j]?).getD 0) = bytes8 v := by
    apply List.ext_getElem?
    intro j
    by_cases hj : j < 8
    · rw [List.getElem?_eq_getElem (by simp; exact hj)]
      simp only [List.getElem_map, List.getElem_range]
      have hs := hslice j hj
      have hlen2 : j < (bytes8 v).length := by rw [bytes8_len]; exact hj
      rw [hs, List.getElem?_eq_getElem hlen2, Option.getD_some]
    · have hj2 : (bytes8 v).length ≤ j := by rw [bytes8_len]; omega
      rw [List.getElem?_eq_none (by simp; omega), List.getElem?_eq_none hj2]
  rw [hlist, Option.map_some, leVal_bytes8 hv]

private theorem getElem?_take_of_ge {α : Type} {l : List α} {n i : Nat} (h : n ≤ i) :
    (l.take n)[i]? = none := by
  rw [List.getElem?_eq_none (Nat.le_trans (List.length_take_le _ _) h)]

private theorem list_take_succ_set {d bs : List Nat} {idx v : Nat}
    (hi : idx < d.length) (hb : idx < bs.length)
    (ht : d.take idx = bs.take idx) (hv : bs[idx] = v) :
    (d.set idx v).take (idx + 1) = bs.take (idx + 1) := by
  apply List.ext_getElem?
  intro k
  by_cases hlt : k < idx + 1
  · rw [List.getElem?_take_of_lt hlt, List.getElem?_take_of_lt hlt]
    by_cases hke : k = idx
    · subst hke; rw [List.getElem?_set_self hi, List.getElem?_eq_getElem hb, hv]
    · rw [List.getElem?_set_ne (Ne.symm hke)]
      have hki : k < idx := by omega
      have := congrArg (·[k]?) ht
      simpa [List.getElem?_take_of_lt hki] using this
  · rw [getElem?_take_of_ge (by omega), getElem?_take_of_ge (by omega)]

private theorem list_take_wrN {d pfx bs : List Nat} {len : Nat}
    (hlen : len + bs.length ≤ d.length)
    (htake : d.take len = pfx)
    (hpl : pfx.length = len)
    {d' : List Nat} (_hdlen : d'.length = d.length)
    (hd : ∀ k, k < d.length →
      d'[k]? = if len ≤ k ∧ k < len + bs.length then bs[k - len]? else d[k]?) :
    d'.take (len + bs.length) = pfx ++ bs := by
  apply List.ext_getElem?
  intro k
  by_cases hk : k < len + bs.length
  · rw [List.getElem?_take_of_lt hk]
    have hkd : k < d.length := by omega
    rw [hd k hkd]
    by_cases hkl : k < len
    · have hnot : ¬ (len ≤ k ∧ k < len + bs.length) := by omega
      rw [if_neg hnot, List.getElem?_append_left (by omega)]
      have ht := congrArg (·[k]?) htake
      simpa [List.getElem?_take_of_lt hkl] using ht
    · have hin : len ≤ k ∧ k < len + bs.length := by omega
      rw [if_pos hin, List.getElem?_append_right (by omega), hpl]
  · rw [getElem?_take_of_ge (by omega), List.getElem?_eq_none (by simp [hpl]; omega)]

/-! ### Shape abstraction, BSS predicate, and buffer predicates -/

def freeS (sh : List (Nat × Nat)) (a n : Nat) : Bool :=
  sh.all fun p => a + n ≤ p.1 || p.2 ≤ a

def placeS (sh : List (Nat × Nat)) (c : Option Choice) (n : Nat) : Option (Option Nat) :=
  match c with
  | some .nomem => some none
  | some (.at a) =>
    if a % page = 0 ∧ 0 < a ∧ a + n ≤ 2 ^ 47 ∧ freeS sh a n then some (some a) else none
  | _ => none

theorem mem_free_eq_freeS (m : Mem) (a n : Nat) : m.free a n = freeS (shape m) a n := by
  induction m with
  | nil => rfl
  | cons r rest ih =>
    change ((decide (a + n ≤ r.base) || decide (r.hi ≤ a)) && Mem.free rest a n) =
      ((decide (a + n ≤ r.base) || decide (r.hi ≤ a)) && freeS (shape rest) a n)
    rw [ih]

theorem place_eq_placeS (m : Mem) (c : Option Choice) (n : Nat) :
    place m c n = placeS (shape m) c n := by
  cases c with
  | none => rfl
  | some ch =>
    cases ch with
    | nomem => rfl
    | count _ => rfl
    | «at» a => simp only [place, placeS, mem_free_eq_freeS]

theorem shape_append_region (m : Mem) (r : Region) :
    shape (m ++ [r]) = shape m ++ [(r.base, r.hi)] := by
  simp [shape]

theorem shape_eraseIdx (m : Mem) (i : Nat) :
    shape (m.eraseIdx i) = (shape m).eraseIdx i := by
  induction m generalizing i with
  | nil => simp [shape]
  | cons x xs ih =>
    cases i with
    | zero => rfl
    | succ n =>
      change (x.base, x.hi) :: shape (xs.eraseIdx n) = (x.base, x.hi) :: (shape xs).eraseIdx n
      rw [ih n]

def HasBss (m : Mem) : Prop :=
  ∃ (ib : Nat) (r : Region), m[ib]? = some r ∧ r.base = 0x600000 ∧ r.data.length = 0x3000 ∧ r.rw = true

def HasBufRW (m : Mem) (i b cap : Nat) (pfx : List Nat) : Prop :=
  ∃ (r : Region), m[i]? = some r ∧ r.base = b ∧ r.data.length = cap ∧ r.rw = true ∧
    0 < b ∧ b + cap ≤ 2 ^ 47 ∧ b % page = 0 ∧ cap % page = 0 ∧ 65536 ≤ cap ∧
    b ≠ 0x600000 ∧ pfx.length ≤ cap ∧ r.data.take pfx.length = pfx

private theorem pages_ge (n : Nat) : n ≤ pages n := by
  simp only [pages, page]; omega

private theorem pages_of_dvd {n : Nat} (h : n % page = 0) : pages n = n := by
  simp only [pages, page] at h ⊢; omega

private theorem pages_dvd (n : Nat) : pages n % page = 0 := by
  simp only [pages, page]; omega

theorem place_spec_full {m : Mem} {c : Option Choice} {n a : Nat}
    (h : place m c n = some (some a)) :
    a % page = 0 ∧ 0 < a ∧ a + n ≤ 2 ^ 47 ∧ m.free a n = true := by
  unfold place at h
  split at h
  · simp at h
  · split at h
    · rename_i hc
      simp only [Option.some.injEq] at h
      subst h
      exact hc
    · contradiction
  · contradiction

private theorem base_ne_of_free {m : Mem} {i : Nat} {r : Region} {a n : Nat}
    (hri : m[i]? = some r) (hn : 0 < n) (hrlen : 0 < r.data.length) (hf : m.free a n = true) :
    a ≠ r.base := by
  have hmem := List.mem_of_getElem? hri
  have := free_spec hf r hmem
  rcases this with h | h <;> (try simp only [Region.hi] at h) <;> omega

private theorem hasBss_append {m : Mem} (hb : HasBss m) (r : Region) : HasBss (m ++ [r]) := by
  obtain ⟨ib, rb, hrib, h1, h2, h3⟩ := hb
  have hlt : ib < m.length := (List.getElem?_eq_some_iff.1 hrib).1
  exact ⟨ib, rb, by rw [List.getElem?_append_left hlt]; exact hrib, h1, h2, h3⟩

private theorem hasBss_set_ne {m : Mem} (hb : HasBss m) {i : Nat} {r0 r1 : Region}
    (hi : m[i]? = some r0) (hne : r0.base ≠ 0x600000) : HasBss (m.set i r1) := by
  obtain ⟨ib, rb, hrib, h1, h2, h3⟩ := hb
  have hne_i : i ≠ ib := by
    intro h; subst h; rw [hi] at hrib; cases hrib; exact hne h1
  exact ⟨ib, rb, by rw [List.getElem?_set_ne hne_i]; exact hrib, h1, h2, h3⟩

private theorem hasBss_eraseIdx_append {m : Mem} (hb : HasBss m) {i : Nat} {r0 r1 : Region}
    (hi : m[i]? = some r0) (hne : r0.base ≠ 0x600000) :
    HasBss (m.eraseIdx i ++ [r1]) := by
  obtain ⟨ib, rb, hrib, h1, h2, h3⟩ := hb
  have hne_i : i ≠ ib := by
    intro h; subst h; rw [hi] at hrib; cases hrib; exact hne h1
  rcases Nat.lt_trichotomy ib i with hlt | heq | hgt
  · have hlen : ib < (m.eraseIdx i).length := by
      have hi_lt := (List.getElem?_eq_some_iff.1 hi).1
      rw [List.length_eraseIdx, if_pos hi_lt]; omega
    refine ⟨ib, rb, ?_, h1, h2, h3⟩
    rw [List.getElem?_append_left hlen, List.getElem?_eraseIdx_of_lt hlt]
    exact hrib
  · exact absurd heq.symm hne_i
  · have hi_lt : i < m.length := (List.getElem?_eq_some_iff.1 hi).1
    have hib_lt : ib < m.length := (List.getElem?_eq_some_iff.1 hrib).1
    have hlen : ib - 1 < (m.eraseIdx i).length := by
      rw [List.length_eraseIdx, if_pos hi_lt]; omega
    refine ⟨ib - 1, rb, ?_, h1, h2, h3⟩
    rw [List.getElem?_append_left hlen, List.getElem?_eraseIdx_of_ge (by omega)]
    rw [show ib - 1 + 1 = ib by omega]
    exact hrib

theorem hasBufRW_to_Buf {m : Mem} {i b cap : Nat} {pfx : List Nat}
    (hw : m.wf) (hb : HasBufRW m i b cap pfx) : Buf m b pfx := by
  obtain ⟨r, hri, rfl, hcap, _, _, _, _, _, _, _, hlen, htake⟩ := hb
  intro k hk
  have ha1 : r.base ≤ r.base + k := by omega
  have ha2 : r.base + k < r.hi := by simp only [Region.hi]; omega
  rw [mem_rd_of_wf hw hri ha1 ha2]
  simp only [Nat.add_sub_cancel_left]
  have helem : r.data[k]? = (r.data.take pfx.length)[k]? := by
    rw [List.getElem?_take_of_lt hk]
  rw [helem, htake, List.getElem?_eq_getElem hk]

private theorem hasBufRW_append {m : Mem} {i b cap : Nat} {pfx : List Nat}
    (hb : HasBufRW m i b cap pfx) (r : Region) : HasBufRW (m ++ [r]) i b cap pfx := by
  obtain ⟨rb, hri, hrest⟩ := hb
  have hlt : i < m.length := (List.getElem?_eq_some_iff.1 hri).1
  exact ⟨rb, by rw [List.getElem?_append_left hlt]; exact hri, hrest⟩

private theorem hasBufRW_set_ne {m : Mem} {i b cap : Nat} {pfx : List Nat}
    (hb : HasBufRW m i b cap pfx) {j : Nat} {r0 r1 : Region}
    (hj : m[j]? = some r0) (hne : r0.base ≠ b) : HasBufRW (m.set j r1) i b cap pfx := by
  obtain ⟨rb, hri, hb_eq, hrest⟩ := hb
  have hij : j ≠ i := by intro h; subst h; rw [hj] at hri; cases hri; exact hne hb_eq
  exact ⟨rb, by rw [List.getElem?_set_ne hij]; exact hri, hb_eq, hrest⟩

private theorem buf_append {m : Mem} {b : Nat} {bs : List Nat}
    (hb : Buf m b bs) (r : Region) : Buf (m ++ [r]) b bs := by
  intro k hk
  have hk_rd := hb k hk
  simp only [Mem.rd, bind, Option.bind] at hk_rd ⊢
  cases hf0 : m.find (b + k) 0 with
  | none => simp [hf0] at hk_rd
  | some p =>
    rcases p with ⟨i, r0⟩
    have hfind_app : (m ++ [r]).find (b + k) 0 = some (i, r0) := by
      have aux : ∀ (l : Mem) (i0 : Nat), l.find (b + k) i0 = some (i, r0) →
          (l ++ [r]).find (b + k) i0 = some (i, r0) := by
        intro l
        induction l with
        | nil => intro i0 h; simp [Mem.find] at h
        | cons x xs ih =>
          intro i0 h
          simp only [List.cons_append, Mem.find] at h ⊢
          split at h
          · rw [if_pos (by assumption)]; exact h
          · rw [if_neg (by assumption)]; exact ih (i0 + 1) h
      exact aux m 0 hf0
    rw [hfind_app]
    simpa [hf0] using hk_rd

private theorem pairwise_getElem {α : Type} {R : α → α → Prop} {l : List α}
    (hp : l.Pairwise R) : ∀ {i j : Nat} {a b : α},
    l[i]? = some a → l[j]? = some b → i ≠ j → R a b ∨ R b a := by
  induction hp with
  | nil => intro i j a b hi; simp at hi
  | @cons x xs hxs _ ih =>
    intro i j a b hi hj hne
    cases i with
    | zero =>
      cases j with
      | zero => exact absurd rfl hne
      | succ j' =>
        simp only [List.getElem?_cons_zero, Option.some.injEq] at hi
        simp only [List.getElem?_cons_succ] at hj
        subst hi
        exact Or.inl (hxs b (List.mem_of_getElem? hj))
    | succ i' =>
      cases j with
      | zero =>
        simp only [List.getElem?_cons_succ] at hi
        simp only [List.getElem?_cons_zero, Option.some.injEq] at hj
        subst hj
        exact Or.inr (hxs a (List.mem_of_getElem? hi))
      | succ j' =>
        simp only [List.getElem?_cons_succ] at hi hj
        exact ih hi hj (fun h => hne (by omega))

private theorem buf_set_ne_base {m : Mem} {b : Nat} {bs : List Nat} (hw : m.wf)
    {ib ibuf : Nat} {rb rbuf : Region}
    (hrib : m[ib]? = some rb) (hbufi : m[ibuf]? = some rbuf) (hne : ib ≠ ibuf)
    (hbase : rbuf.base = b) (hblen : b + bs.length ≤ rbuf.hi)
    {db : List Nat} (hdl : db.length = rb.data.length) (hb : Buf m b bs) :
    Buf (m.set ib { rb with data := db }) b bs := by
  intro k hk
  have hk_rd := hb k hk
  simp only [Mem.rd, bind, Option.bind] at hk_rd ⊢
  have hdisj : ¬ (rb.base ≤ b + k ∧ b + k < rb.hi) := by
    have hp := pairwise_getElem hw.2 hrib hbufi hne
    intro hmem
    simp only [Region.hi] at hp hmem hblen
    rcases hp with h | h <;> omega
  have hfind_eq : Mem.find (m.set ib { rb with data := db }) (b + k) 0 = Mem.find m (b + k) 0 := by
    suffices h : ∀ (l : Mem) (idx i0 : Nat),
        l[idx]? = some rb →
        Mem.find (l.set idx { rb with data := db }) (b + k) i0 = Mem.find l (b + k) i0 from
      h m ib 0 hrib
    intro l
    induction l with
    | nil => intro idx i0 h; simp at h
    | cons x xs ih =>
      intro idx i0 hx
      cases idx with
      | zero =>
        simp only [List.getElem?_cons_zero, Option.some.injEq] at hx
        have hnot' : ¬ (({ rb with data := db } : Region).base ≤ b + k ∧
            b + k < ({ rb with data := db } : Region).hi) := by
          simp only [Region.hi, hdl]; exact hdisj
        rw [hx]
        simp [Mem.find, hdisj, hnot']
      | succ j =>
        simp only [List.getElem?_cons_succ] at hx
        simp only [List.set_cons_succ, Mem.find]
        split
        · rfl
        · exact ih j (i0 + 1) hx
  rw [hfind_eq]
  exact hk_rd

/-! ### CPS result predicate for initialization and read loop -/

def ReachesInit (s : St) (P : St → Prop) (allow6 allowStuck : Bool) : Prop :=
  ∀ (Q : Out → Prop),
    (∀ s', P s' → Ends s' Q) →
    (allow6 = true → Q (.exit 6)) →
    (allowStuck = true → Q .stuck) →
    Ends s Q

theorem ReachesInit.refl {s : St} {P : St → Prop} {a6 as : Bool} (h : P s) :
    ReachesInit s P a6 as := fun _ hP _ _ => hP s h

theorem ReachesInit.trans {s : St} {P1 P2 : St → Prop} {a6 as : Bool}
    (h1 : ReachesInit s P1 a6 as) (h2 : ∀ s', P1 s' → ReachesInit s' P2 a6 as) :
    ReachesInit s P2 a6 as :=
  fun Q hQ h6 hs => h1 Q (fun s' hs' => h2 s' hs' Q hQ h6 hs) h6 hs

theorem Reaches.toInit {s : St} {P : St → Prop} (a6 as : Bool) (h : Reaches s P) :
    ReachesInit s P a6 as :=
  fun _ hP _ _ => h.ends hP

theorem ReachesInit.exit6 {s : St} {P : St → Prop} {as : Bool}
    (he : Ends s (· = .exit 6)) : ReachesInit s P true as := by
  intro Q _ h6 _
  obtain ⟨n, hn⟩ := he
  exact ⟨n, by rw [hn]; exact h6 rfl⟩

theorem ReachesInit.stuck1 {s : St} {P : St → Prop} {a6 : Bool} {ins : Ins} {len : Nat}
    (hf : im s.rip = some (ins, len)) (hs : exec ins len s = .stuck) :
    ReachesInit s P a6 true := by
  intro Q _ _ hst
  exact ⟨1, by simp [run, step, hf, hs]; exact hst rfl⟩

/-! ### Copy-loop proof (`4210539..4210561` and `4210611..4210633`) -/

def CopyCode (L L_end : Nat) : Prop :=
  CodeAt [
    (L, .test 11 11, 3),
    (L + 3, .jcc 4 L_end, 2),
    (L + 5, .ld8 2 6 0, 3),
    (L + 8, .st8 7 0 2, 2),
    (L + 10, .aluI .add 6 1, 4),
    (L + 14, .aluI .add 7 1, 4),
    (L + 18, .aluI .sub 11 1, 4),
    (L + 22, .jmp L, 2)
  ]

theorem copy_loop_spec {L L_end : Nat} (hc : CopyCode L L_end)
    {i1 b1 b2 cap1 : Nat} {bs : List Nat}
    (hby : ∀ x ∈ bs, x < 256) (hb1s : b1 + bs.length < 2 ^ 48) (hb2s : b2 + bs.length < 2 ^ 48)
    (hne_12 : b2 ≠ b1) (hne_bss2 : b2 ≠ 0x600000) :
    ∀ (rem idx : Nat) (s : St),
      idx + rem = bs.length →
      s.rip = L →
      RegDefExcept11 s →
      s.mem.wf →
      HasBss s.mem →
      HasBufRW s.mem i1 b1 cap1 bs →
      (∃ (i2 : Nat) (r2 : Region), s.mem[i2]? = some r2 ∧ r2.base = b2 ∧ r2.rw = true ∧
        bs.length ≤ r2.data.length ∧ 0 < r2.data.length ∧ r2.data.take idx = bs.take idx) →
      s.reg 6 = some (b1 + idx) →
      s.reg 7 = some (b2 + idx) →
      s.reg 11 = some rem →
      Reaches s (fun s' =>
        s'.rip = L_end ∧
        RegDefExcept11 s' ∧
        s'.mem.wf ∧
        shape s'.mem = shape s.mem ∧
        HasBss s'.mem ∧
        HasBufRW s'.mem i1 b1 cap1 bs ∧
        (∃ (i2 : Nat) (r2 : Region), s'.mem[i2]? = some r2 ∧ r2.base = b2 ∧ r2.rw = true ∧
          bs.length ≤ r2.data.length ∧ 0 < r2.data.length ∧ Buf s'.mem b2 bs) ∧
        (∀ (j : Nat) (rj : Region), s.mem[j]? = some rj → rj.base ≠ b2 → s'.mem[j]? = some rj) ∧
        s'.reg 14 = s.reg 14 ∧ s'.reg 15 = s.reg 15 ∧
        s'.reg 5 = s.reg 5 ∧ s'.reg 13 = s.reg 13 ∧ s'.reg 8 = s.reg 8 ∧
        s'.k = s.k)
  | 0, idx, s, hsum, hrip, hd, hwf, hbss, hb1, hr2, h6, h7, h11 => by
    have hidx : idx = bs.length := by omega
    subst hidx
    simp only [CopyCode, CodeAt_cons, CodeAt_nil, and_true] at hc
    obtain ⟨f0, f3, _, _, _, _, _, _⟩ := hc
    have hct : X86Int.cond (aluF .and 0 0).2 4 = some true := by simpa using (cond_test_e (x := 0))
    refine Reaches.trans (P := fun s' => s'.rip = L_end ∧ s'.regs = s.regs ∧
        s'.mem = s.mem ∧ s'.k = s.k) ?_ ?_
    · xs f0 using h11
      xs f3 using hct
      exact Reaches.refl ⟨rfl, rfl, rfl, rfl⟩
    · rintro s' ⟨hr', hregs', hmem', hk'⟩
      obtain ⟨i2, r2, hr2i, rfl, hr2w, hlen2, hr2_pos, htake2⟩ := hr2
      have hd' : RegDefExcept11 s' := ⟨by rw [hregs']; exact hd.1, fun j hj hj11 => by
        simp only [St.reg, hregs']; exact hd.2 j hj hj11⟩
      have hbuf2 : Buf s'.mem r2.base bs := by
        rw [hmem']
        intro k hk
        have ha1 : r2.base ≤ r2.base + k := by omega
        have ha2 : r2.base + k < r2.hi := by simp only [Region.hi]; omega
        rw [mem_rd_of_wf hwf hr2i ha1 ha2]
        simp only [Nat.add_sub_cancel_left]
        have heq : r2.data[k]? = (r2.data.take bs.length)[k]? := by
          rw [List.getElem?_take_of_lt hk]
        rw [heq, htake2, List.take_length, List.getElem?_eq_getElem hk]
      refine Reaches.refl ⟨hr', hd', by rwa [hmem'], by rw [hmem'], by rwa [hmem'], by rwa [hmem'],
        ⟨i2, r2, by rwa [hmem'], rfl, hr2w, hlen2, hr2_pos, hbuf2⟩,
        fun j rj hj _ => by rwa [hmem'],
        by simp [St.reg, hregs'], by simp [St.reg, hregs'], by simp [St.reg, hregs'],
        by simp [St.reg, hregs'], by simp [St.reg, hregs'], hk'⟩
  | rem + 1, idx, s, hsum, hrip, hd, hwf, hbss, hb1, hr2, h6, h7, h11 => by
    have hc' := hc
    simp only [CopyCode, CodeAt_cons, CodeAt_nil, and_true] at hc'
    obtain ⟨f0, f3, f5, f8, f10, f14, f18, f22⟩ := hc'
    have hct : X86Int.cond (aluF .and (rem + 1) (rem + 1)).2 4 = some false := by
      simpa using (cond_test_e (x := rem + 1))
    obtain ⟨i2, r2, hr2i, rfl, hr2w, hlen2, hr2_pos, htake2⟩ := hr2
    have hb1_buf : Buf s.mem b1 bs := hasBufRW_to_Buf hwf hb1
    have hidx_lt : idx < bs.length := by omega
    have hrd8 : s.mem.rd (ea (b1 + idx) 0) = some bs[idx] := by
      rw [ea_small (by omega), Nat.add_zero]
      exact hb1_buf idx hidx_lt
    have hv_lt : bs[idx] < 256 := hby _ (List.getElem_mem hidx_lt)
    have ha1 : r2.base ≤ r2.base + idx := by omega
    have ha2 : r2.base + idx < r2.hi := by simp only [Region.hi]; omega
    have hwr1 := mem_wr_of_wf hwf hr2i hr2w (v := bs[idx]) ha1 ha2
    simp only [Nat.add_sub_cancel_left] at hwr1
    have hwr_eq : s.mem.wr (ea (r2.base + idx) 0) (bs[idx] % 256) =
        some (s.mem.set i2 { r2 with data := r2.data.set idx (bs[idx] % 256) }) := by
      rw [ea_small (by omega), Nat.add_zero]
      simpa [Nat.mod_eq_of_lt hv_lt] using hwr1
    let d2' := r2.data.set idx (bs[idx] % 256)
    let r2' : Region := { r2 with data := d2' }
    let m' : Mem := s.mem.set i2 r2'
    have hsh' : shape m' = shape s.mem := wr_shape hwr1
    have hwf' : m'.wf := wf_of_shape hsh' hwf
    have hbss' : HasBss m' := hasBss_set_ne hbss hr2i hne_bss2
    have hb1' : HasBufRW m' i1 b1 cap1 bs := hasBufRW_set_ne hb1 hr2i hne_12
    have hi2_lt : i2 < s.mem.length := (List.getElem?_eq_some_iff.1 hr2i).1
    have htake2' : d2'.take (idx + 1) = bs.take (idx + 1) := by
      dsimp only [d2']
      rw [Nat.mod_eq_of_lt hv_lt]
      exact list_take_succ_set (by omega) hidx_lt htake2 rfl
    have hr2' : ∃ (i2' : Nat) (r2'' : Region), m'[i2']? = some r2'' ∧ r2''.base = r2.base ∧ r2''.rw = true ∧
        bs.length ≤ r2''.data.length ∧ 0 < r2''.data.length ∧ r2''.data.take (idx + 1) = bs.take (idx + 1) :=
      ⟨i2, r2', by simp [m', List.getElem?_set_self hi2_lt], rfl, hr2w,
        by simp [r2', d2', hlen2], by simp [r2', d2', hr2_pos], htake2'⟩
    have hother1 : ∀ (j : Nat) (rj : Region), s.mem[j]? = some rj → rj.base ≠ r2.base → m'[j]? = some rj := by
      intro j rj hj hne
      have hij : i2 ≠ j := by intro h; subst h; rw [hr2i] at hj; cases hj; exact hne rfl
      simpa [m', List.getElem?_set_ne hij] using hj
    have hadd6 : (aluF .add (b1 + idx) 1).1 = b1 + (idx + 1) := by
      rw [add_small (by omega)]; omega
    have hadd7 : (aluF .add (r2.base + idx) 1).1 = r2.base + (idx + 1) := by
      rw [add_small (by omega)]; omega
    have hsub11 : (aluF .sub (rem + 1) 1).1 = rem := by
      rw [sub_small (by omega) (by omega)]; omega
    have hstep1 : Reaches s (fun s1 =>
        s1.rip = L ∧
        RegDefExcept11 s1 ∧
        s1.mem = m' ∧
        s1.reg 6 = some (b1 + (idx + 1)) ∧
        s1.reg 7 = some (r2.base + (idx + 1)) ∧
        s1.reg 11 = some rem ∧
        s1.reg 14 = s.reg 14 ∧ s1.reg 15 = s.reg 15 ∧
        s1.reg 5 = s.reg 5 ∧ s1.reg 13 = s.reg 13 ∧ s1.reg 8 = s.reg 8 ∧
        s1.k = s.k) := by
      xs f0 using h11
      xs f3 using hct
      xs f5 using h6, hrd8
      xs f8 using xst8, hd.1, h7, hwr_eq
      xs f10 using h6
      xs f14 using h7
      xs f18 using h11
      xs f22
      refine Reaches.refl ⟨rfl, ?_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, rfl⟩
      · exact regDef_set_some (regDef_set_some (regDef_set_some (regDef_set_some hd 2 _) 6 _) 7 _) 11 _
      · rsimpG [hd.1, hadd6]
      · rsimpG [hd.1, hadd7]
      · rsimpG [hd.1, hsub11]
      all_goals rsimpG [hd.1]
    refine hstep1.trans fun s1 ⟨hr1, hd1, hm1, h61, h71, h111, h14, h15, h5, h13, h8, hk1⟩ => ?_
    have ih := copy_loop_spec hc hby hb1s hb2s hne_12 hne_bss2 rem (idx + 1) s1
      (by omega) hr1 hd1 (by rwa [hm1]) (by rwa [hm1]) (by rwa [hm1]) (by rwa [hm1]) h61 h71 h111
    refine ih.mono fun s' ⟨ha, hb, hc', hsh, hd', he, hf, hoth, hg14, hg15, hg5, hg13, hg8, hk'⟩ =>
      ⟨ha, hb, hc', by rw [hsh, hm1, hsh'], hd', he, hf,
        fun j rj hj hne => hoth j rj (by rw [hm1]; exact hother1 j rj hj hne) hne,
        hg14.trans h14, hg15.trans h15, hg5.trans h5, hg13.trans h13, hg8.trans h8, hk'.trans hk1⟩

/-! ### Oracle predicates (unconditional and good-oracle unification) -/

def countStep (o : List Choice) (want avail : Nat) : Nat × List Choice :=
  let full := min want avail
  match o with
  | .count n :: o' => if 0 < n ∧ n ≤ full then (n, o') else (full, o')
  | _ => (full, o)

private theorem kern_count_eq_countStep (k : Kern) (want avail : Nat) :
    (k.count want avail).1 = (countStep k.oracle want avail).1 ∧
    (k.count want avail).2.oracle = (countStep k.oracle want avail).2 ∧
    (k.count want avail).2.stdin = k.stdin := by
  cases ho : k.oracle with
  | nil => simp [Kern.count, Kern.next, countStep, ho]
  | cons c rest =>
    cases c with
    | count n =>
      by_cases hc : 0 < n ∧ n ≤ min want avail <;>
        simp [Kern.count, Kern.next, countStep, ho, hc]
    | _ => simp [Kern.count, Kern.next, countStep, ho]

private theorem countStep_bounds (o : List Choice) (want avail : Nat) :
    (countStep o want avail).1 ≤ want ∧ (countStep o want avail).1 ≤ avail := by
  simp only [countStep]
  cases o with
  | nil => exact ⟨Nat.min_le_left _ _, Nat.min_le_right _ _⟩
  | cons c rest =>
    cases c with
    | count n =>
      by_cases hc : 0 < n ∧ n ≤ min want avail
      · simp only [if_pos hc]
        exact ⟨Nat.le_trans hc.2 (Nat.min_le_left _ _), Nat.le_trans hc.2 (Nat.min_le_right _ _)⟩
      · simp only [if_neg hc]
        exact ⟨Nat.min_le_left _ _, Nat.min_le_right _ _⟩
    | _ => exact ⟨Nat.min_le_left _ _, Nat.min_le_right _ _⟩

private theorem countStep_zero_iff (o : List Choice) {want avail : Nat} (hw : 0 < want) :
    (countStep o want avail).1 = 0 ↔ avail = 0 := by
  simp only [countStep]
  cases o with
  | nil => simp only [Nat.min_def]; split <;> omega
  | cons c rest =>
    cases c with
    | count n =>
      by_cases hc : 0 < n ∧ n ≤ min want avail
      · simp only [if_pos hc]
        rcases hc with ⟨hn, hmin⟩
        have ha : 0 < avail := by
          by_cases h : want ≤ avail <;> (simp only [Nat.min_def, h, ↓reduceIte] at hmin; omega)
        constructor <;> omega
      · simp only [if_neg hc]
        rw [Nat.min_def]
        split <;> omega
    | _ => simp only [Nat.min_def]; split <;> omega

inductive GoodTailOrac (sh : List (Nat × Nat)) (len : Nat) : List Choice → Prop where
  | mk {b2 b3 o}
      (hp2 : placeS sh o.head? (pages (len + 1)) = some (some b2))
      (hp3 : placeS (sh ++ [(b2, b2 + pages (len + 1))]) (o.drop 1).head? (pages (len + 1)) =
        some (some b3)) :
      GoodTailOrac sh len o

inductive GoodRdOrac : List (Nat × Nat) → Nat → Nat → Nat → Nat → List Choice → Prop where
  | grow {sh iBuf cap rem a' o}
      (hp : placeS (sh.eraseIdx iBuf) o.head? (2 * cap) = some (some a'))
      (hnext : GoodRdOrac (sh.eraseIdx iBuf ++ [(a', a' + 2 * cap)]) (sh.eraseIdx iBuf).length
        cap (2 * cap) rem (o.drop 1)) :
      GoodRdOrac sh iBuf cap cap rem o
  | done {sh iBuf len cap o}
      (hlt : len < cap)
      (htail : GoodTailOrac sh len (countStep o (cap - len) 0).2) :
      GoodRdOrac sh iBuf len cap 0 o
  | step {sh iBuf len cap rem o}
      (hlt : len < cap)
      (hpos : 0 < rem)
      (hnext :
        let p := countStep o (cap - len) rem
        GoodRdOrac sh iBuf (len + p.1) cap (rem - p.1) p.2) :
      GoodRdOrac sh iBuf len cap rem o

inductive GoodInitOrac (sh : List (Nat × Nat)) (rem : Nat) (o : List Choice) : Prop where
  | mk {a0}
      (hp : placeS sh o.head? 65536 = some (some a0))
      (hrd : GoodRdOrac (sh ++ [(a0, a0 + 65536)]) sh.length 0 65536 rem (o.drop 1)) :
      GoodInitOrac sh rem o

private theorem head?_eq_next (k : Kern) : k.next.1 = k.oracle.head? ∧ k.next.2.oracle = k.oracle.drop 1 := by
  cases ho : k.oracle <;> simp [Kern.next, ho]

/-! ### Init execution: mremap, read loop, eof triplication, start -/

theorem init_mmap1_spec {s : St} {a6 as : Bool}
    (hrip : s.rip = L_start) (hd : RegDefExcept11 s)
    (hwf : s.mem.wf) (hbss : HasBss s.mem)
    (hor : a6 = false ∨ as = false → GoodInitOrac (shape s.mem) s.k.stdin.length s.k.oracle) :
    ReachesInit s (fun s1 =>
      s1.rip = L_rd ∧
      RegDefExcept11 s1 ∧
      s1.mem.wf ∧
      HasBss s1.mem ∧
      ∃ i1 b1, HasBufRW s1.mem i1 b1 65536 [] ∧
        shape s1.mem = shape s.mem ++ [(b1, b1 + 65536)] ∧
        i1 = (shape s.mem).length ∧
        s1.reg 14 = some b1 ∧ s1.reg 12 = some 65536 ∧ s1.reg 13 = some 0 ∧
        s1.reg 8 = some (W - 1) ∧
        s1.k.stdin = s.k.stdin ∧
        (a6 = false ∨ as = false →
          GoodRdOrac (shape s1.mem) i1 0 65536 s1.k.stdin.length s1.k.oracle)) a6 as := by
  have hcode : CodeAt [(L_start, .alu .xor 7 7, 2), (L_start + 2, .movI 6 65536, 5),
      (L_start + 7, .movI 2 3, 5), (L_start + 12, .movI 10 34, 6),
      (L_start + 18, .movI 8 (W - 1), 10), (L_start + 28, .alu .xor 9 9, 3),
      (L_start + 31, .movI 0 9, 5), (L_start + 36, .syscall, 2),
      (L_start + 38, .test 0 0, 3), (L_start + 41, .jcc 8 L_nomem, 6),
      (L_start + 47, .mov 14 0, 3), (L_start + 50, .movI 12 65536, 6),
      (L_start + 56, .alu .xor 13 13, 3)] :=
    codeAt_of_init0_sub (by decide +kernel)
  simp only [CodeAt_cons, CodeAt_nil, and_true] at hcode
  obtain ⟨f0, f2, f7, f12, f18, f28, f31, f36, f38, f41, f47, f50, f56⟩ := hcode
  obtain ⟨v7, hv7⟩ := regDef_get hd 7
  obtain ⟨v9, hv9⟩ := regDef_get hd 9
  have hsetup : Reaches s (fun s1 =>
      s1.rip = L_start + 36 ∧ RegDefExcept11 s1 ∧
      s1.mem = s.mem ∧ s1.k = s.k ∧
      s1.reg 0 = some 9 ∧ s1.reg 7 = some 0 ∧ s1.reg 6 = some 65536 ∧
      s1.reg 2 = some 3 ∧ s1.reg 10 = some 34 ∧ s1.reg 8 = some (W - 1) ∧ s1.reg 9 = some 0) := by
    xs f0 using hv7
    xs f2
    xs f7
    xs f12
    xs f18
    xs f28 using hv9
    xs f31
    refine Reaches.refl ⟨by dsimp only; omega, ?_, rfl, rfl, by rsimpG [hd.1],
      ?_, by rsimpG [hd.1], by rsimpG [hd.1], by rsimpG [hd.1],
      by rsimpG [hd.1], ?_⟩
    · exact regDef_set_some (regDef_set_some (regDef_set_some (regDef_set_some (regDef_set_some
        (regDef_set_some (regDef_set_some hd 7 _) 6 _) 2 _) 10 _) 8 _) 9 _) 0 _
    · rsimpG [hd.1]; exact alu_xor_self v7
    · rsimpG [hd.1]; exact alu_xor_self v9
  refine ReachesInit.trans (hsetup.toInit a6 as) ?_
  rintro s1 ⟨hrip1, hd1, hm1, hk1, h0, h7, h6, h2, h10, h8, h9⟩
  have hor1 : a6 = false ∨ as = false → GoodInitOrac (shape s1.mem) s1.k.stdin.length s1.k.oracle :=
    fun h => by rw [hm1, hk1]; exact hor h
  clear hor
  have hsys_pre : syscall s1.k s1.mem 9 0 65536 3 34 (W - 1) 0 =
      (match place s1.mem s1.k.next.1 65536 with
      | none => none
      | some none => some (.ret (neg ENOMEM) s1.k.next.2 s1.mem)
      | some (some a) => some (.ret a s1.k.next.2
          (s1.mem ++ [⟨a, List.replicate 65536 0, true⟩]))) := by
    simp only [syscall, PROT_RW, MAP_PRIVATE_ANON, show pages 65536 = 65536 by decide,
      bind, Option.bind, pure]
    rcases s1.k.next with ⟨c, k'⟩
    cases place s1.mem c 65536 with
    | none => rfl
    | some o => cases o <;> rfl
  cases hpl : place s1.mem s1.k.next.1 65536 with
  | none =>
    cases as
    · exfalso
      obtain ⟨hp, _⟩ := hor1 (Or.inr rfl)
      rw [place_eq_placeS, (head?_eq_next s1.k).1] at hpl
      rw [hp] at hpl; cases hpl
    · exact ReachesInit.stuck1 (by rw [hrip1]; exact f36)
        (exec_syscall_none h0 h7 h6 h2 h10 h8 h9 (by rw [hsys_pre, hpl]))
  | some o =>
    cases o with
    | none =>
      cases a6
      · exfalso
        obtain ⟨hp, _⟩ := hor1 (Or.inl rfl)
        rw [place_eq_placeS, (head?_eq_next s1.k).1] at hpl
        rw [hp] at hpl; cases hpl
      · have hsys : syscall s1.k s1.mem 9 0 65536 3 34 (W - 1) 0 =
            some (.ret (neg ENOMEM) s1.k.next.2 s1.mem) := by
          rw [hsys_pre, hpl]
        have he := exec_syscall_ret (len := 2) h0 h7 h6 h2 h10 h8 h9 hsys
        let s1' : St := { ((s1.set 0 (some (neg ENOMEM))).set 1 (some (L_start + 38))).set 11 none with
          k := s1.k.next.2, mem := s1.mem, rip := L_start + 38 }
        have he' : exec .syscall 2 s1 = .ok s1' := by rw [he, hrip1]
        have hs1' : Reaches s1 (fun s2 => s2.rip = L_nomem ∧ RegDefExcept11 s2) := by
          refine reach_step (by rw [hrip1]; exact f36) ?_
          rw [he']
          have h0' : s1'.reg 0 = some (W - ENOMEM) := by simp only [s1']; rsimpG [hd1.1]
          have hcc : X86Int.cond (aluF .and (W - ENOMEM) (W - ENOMEM)).2 8 = some true := by
            rw [cond_test_s, msb_nomem]
          xs f38 using h0'
          xs f41 using hcc
          exact Reaches.refl ⟨by dsimp only; rfl,
            regDef_set_11_none (regDef_set_some (regDef_set_some hd1 0 _) 1 _)⟩
        exact ReachesInit.exit6 (hs1'.ends fun s2 ⟨hr2, hd2⟩ => nomem_spec hr2 hd2)
    | some b1 =>
      have hpf := place_spec_full hpl
      have hsys : syscall s1.k s1.mem 9 0 65536 3 34 (W - 1) 0 =
          some (.ret b1 s1.k.next.2 (s1.mem ++ [⟨b1, List.replicate 65536 0, true⟩])) := by
        rw [hsys_pre, hpl]
      have he := exec_syscall_ret (len := 2) h0 h7 h6 h2 h10 h8 h9 hsys
      let m2 := s1.mem ++ [⟨b1, List.replicate 65536 0, true⟩]
      have hwf2 : m2.wf := syscall_wf (by rw [hm1]; exact hwf) hsys
      have hbss2 : HasBss m2 := hasBss_append (by rw [hm1]; exact hbss) _
      have hb1_ne : b1 ≠ 0x600000 := by
        have hbss_1 : HasBss s1.mem := by rw [hm1]; exact hbss
        obtain ⟨ib, rb, hrib, hb_eq, hblen, _⟩ := hbss_1
        rw [← hb_eq]
        exact base_ne_of_free hrib (by decide) (by omega) hpf.2.2.2
      have hbuf2 : HasBufRW m2 s1.mem.length b1 65536 [] :=
        ⟨⟨b1, List.replicate 65536 0, true⟩,
          by rw [List.getElem?_append_right (Nat.le_refl _), Nat.sub_self]; rfl,
          rfl, List.length_replicate, rfl,
          hpf.2.1, hpf.2.2.1, hpf.1, by decide, by decide, hb1_ne,
          by simp only [List.length_nil, Nat.zero_le],
          by simp only [List.length_nil, List.take_zero]⟩
      have hsh2 : shape m2 = shape s1.mem ++ [(b1, b1 + 65536)] := by
        simp only [m2, shape_append_region, Region.hi, List.length_replicate]
      have hor2 : a6 = false ∨ as = false →
          GoodRdOrac (shape m2) (shape s1.mem).length 0 65536 s1.k.next.2.stdin.length s1.k.next.2.oracle := by
        intro h
        obtain ⟨hp, hrd⟩ := hor1 h
        rw [place_eq_placeS, (head?_eq_next s1.k).1] at hpl
        rw [hp] at hpl
        cases hpl
        have hdrop := (head?_eq_next s1.k).2
        have hstdin : s1.k.next.2.stdin = s1.k.stdin := by cases ho : s1.k.oracle <;> simp [Kern.next, ho]
        rw [hsh2, hdrop, hstdin]
        simpa [shape] using hrd
      have hmsb : msb b1 = false := msb_lt (by omega)
      have hcc : X86Int.cond (aluF .and b1 b1).2 8 = some false := by rw [cond_test_s, hmsb]
      refine Reaches.toInit a6 as ?_
      refine reach_step (by rw [hrip1]; exact f36) ?_
      let s1' : St := { ((s1.set 0 (some b1)).set 1 (some (L_start + 38))).set 11 none with
        k := s1.k.next.2, mem := m2, rip := L_start + 38 }
      have he' : exec .syscall 2 s1 = .ok s1' := by rw [he, hrip1]
      rw [he']
      obtain ⟨v13, hv13⟩ : ∃ v, s1.reg 13 = some v := regDef_get hd1 13
      have hv13' : s1'.reg 13 = some v13 := by simp only [s1']; rsimpG [hd1.1]; exact hv13
      have h0' : s1'.reg 0 = some b1 := by simp only [s1']; rsimpG [hd1.1]
      have h8' : s1'.reg 8 = some (W - 1) := by simp only [s1']; rsimpG [hd1.1]; exact h8
      have hd1' : RegDefExcept11 s1' :=
        regDef_set_11_none (regDef_set_some (regDef_set_some hd1 0 _) 1 _)
      xs f38 using h0'
      xs f41 using hcc
      xs f47 using h0'
      xs f50
      xs f56 using hv13'
      refine Reaches.refl ⟨by rfl, ?_, by rsimpG [hd1'.1]; exact hwf2,
        by rsimpG [hd1'.1]; exact hbss2,
        s1.mem.length, b1, by rsimpG [hd1'.1]; exact hbuf2,
        by rsimpG [hd1'.1]; rw [← hm1]; exact hsh2,
        by simp [shape, hm1], by rsimpG [hd1'.1], by rsimpG [hd1'.1],
        by rsimpG [hd1'.1]; exact alu_xor_self v13,
        by rsimpG [hd1'.1]; exact h8',
        by simp only [s1']; rw [hk1]; cases ho : s.k.oracle <;> simp [Kern.next, ho],
        fun h => by
          simp only [s1']
          simpa [hm1, hk1, shape] using hor2 h⟩
      exact regDef_set_some (regDef_set_some (regDef_set_some hd1' 14 _) 12 _) 13 _

theorem step_mremap_gen {s : St} {i1 b1 cap : Nat} {pfx : List Nat} {a6 as : Bool}
    (hrip : s.rip = L_rd) (hd : RegDefExcept11 s)
    (hwf : s.mem.wf) (hbss : HasBss s.mem) (hb1 : HasBufRW s.mem i1 b1 cap pfx)
    (hpl : pfx.length = cap)
    (h14 : s.reg 14 = some b1) (h12 : s.reg 12 = some cap) (h13 : s.reg 13 = some cap)
    (h8 : s.reg 8 = some (W - 1))
    (hor : a6 = false ∨ as = false → GoodRdOrac (shape s.mem) i1 cap cap s.k.stdin.length s.k.oracle) :
    ReachesInit s (fun s1 =>
      s1.rip = L_rd_go ∧
      RegDefExcept11 s1 ∧
      s1.mem.wf ∧
      HasBss s1.mem ∧
      ∃ i1' b1', HasBufRW s1.mem i1' b1' (2 * cap) pfx ∧
        s1.reg 14 = some b1' ∧ s1.reg 12 = some (2 * cap) ∧ s1.reg 13 = some cap ∧
        s1.reg 8 = some (W - 1) ∧
        s1.k.stdin = s.k.stdin ∧
        (a6 = false ∨ as = false →
          GoodRdOrac (shape s1.mem) i1' cap (2 * cap) s1.k.stdin.length s1.k.oracle)) a6 as := by
  have hcode : CodeAt [(L_rd, .cmp 13 12, 3), (L_rd + 3, .jcc 5 L_rd_go, 2),
      (L_rd + 5, .mov 7 14, 3), (L_rd + 8, .mov 6 12, 3), (L_rd + 11, .mov 2 12, 3),
      (L_rd + 14, .shlI 2 1, 4), (L_rd + 18, .movI 10 1, 6), (L_rd + 24, .movI 0 25, 5),
      (L_rd + 29, .syscall, 2), (L_rd + 31, .test 0 0, 3), (L_rd + 34, .jcc 8 L_nomem, 6),
      (L_rd + 40, .mov 14 0, 3), (L_rd + 43, .shlI 12 1, 4)] :=
    codeAt_of_init0_sub (by decide +kernel)
  simp only [CodeAt_cons, CodeAt_nil, and_true] at hcode
  obtain ⟨f0, f3, f5, f8, f11, f14, f18, f24, f29, f31, f34, f40, f43⟩ := hcode
  obtain ⟨r, hri, rfl, hrcap, hrw, hb1_pos, hb1_47, hb1_pg, hcap_pg, hcap_64, hb1_bss, _, htake⟩ := hb1
  have hcap_W : cap < W := by rw [W_eq]; omega
  have h2cap_W : 2 * cap < W := by rw [W_eq]; omega
  have hc5 : X86Int.cond (aluF .sub cap cap).2 5 = some false := by
    simpa using (cond_sub hcap_W hcap_W).2.2.2.1
  have hshl : (cap * 2 ^ (1 % 64)) % W = 2 * cap := by
    rw [show 1 % 64 = 1 by decide, Nat.pow_one, Nat.mul_comm cap 2, Nat.mod_eq_of_lt h2cap_W]
  have hshl' : cap * 2 % 18446744073709551616 = 2 * cap := by
    rw [W_eq] at h2cap_W; omega
  have hsetup : Reaches s (fun s1 =>
      s1.rip = L_rd + 29 ∧ RegDefExcept11 s1 ∧
      s1.mem = s.mem ∧ s1.k = s.k ∧
      s1.reg 0 = some 25 ∧ s1.reg 7 = some r.base ∧ s1.reg 6 = some cap ∧
      s1.reg 2 = some (2 * cap) ∧ s1.reg 10 = some 1 ∧
      s1.reg 14 = some r.base ∧ s1.reg 12 = some cap ∧ s1.reg 13 = some cap ∧
      s1.reg 8 = some (W - 1)) := by
    xs f0 using h13, h12
    xs f3 using hc5
    xs f5 using h14
    xs f8 using h12
    xs f11 using h12
    xs f14 using hd.1
    xs f18
    xs f24
    refine Reaches.refl ⟨by simp only []; omega, ?_, rfl, rfl, by rsimpG [hd.1],
      by rsimpG [hd.1], by rsimpG [hd.1],
      by rsimpG [hd.1, hshl'], by rsimpG [hd.1], by rsimpG [hd.1]; exact h14,
      by rsimpG [hd.1]; exact h12, by rsimpG [hd.1]; exact h13,
      by rsimpG [hd.1]; exact h8⟩
    exact regDef_set_some (regDef_set_some (regDef_set_some (regDef_set_some (regDef_set_some
      (regDef_set_some hd 7 _) 6 _) 2 _) 2 _) 10 _) 0 _
  refine ReachesInit.trans (hsetup.toInit a6 as) ?_
  rintro s1 ⟨hrip1, hd1, hm1, hk1, h0, h7, h6, h2, h10, _, h12_1, h13_1, h8_1⟩
  have hor1 : a6 = false ∨ as = false → GoodRdOrac (shape s1.mem) i1 cap cap s1.k.stdin.length s1.k.oracle :=
    fun h => by rw [hm1, hk1]; exact hor h
  clear hor
  obtain ⟨v9, hv9⟩ := regDef_get hd1 9
  have hfind : s1.mem.find r.base 0 = some (i1, r) := by
    rw [hm1]
    exact mem_find_of_wf hwf hri (by omega) (by simp only [Region.hi]; omega)
  have hpg_cap : pages cap = cap := pages_of_dvd hcap_pg
  have h2cap_pg : (2 * cap) % page = 0 := by
    rw [Nat.mul_mod, hcap_pg, Nat.mul_zero, Nat.zero_mod]
  have hpg_2cap : pages (2 * cap) = 2 * cap := pages_of_dvd h2cap_pg
  have hnot_lt : ¬ (2 * cap < r.data.length) := by omega
  have hsys_pre : syscall s1.k s1.mem 25 r.base cap (2 * cap) 1 (W - 1) v9 =
      (match place (s1.mem.eraseIdx i1) s1.k.next.1 (2 * cap) with
      | none => none
      | some none => some (.ret (neg ENOMEM) s1.k.next.2 s1.mem)
      | some (some a) => some (.ret a s1.k.next.2
          (s1.mem.eraseIdx i1 ++ [⟨a, r.data ++ List.replicate (2 * cap - cap) 0, true⟩]))) := by
    simp only [syscall, MREMAP_MAYMOVE, hfind, hrcap, hrw, hpg_cap, hpg_2cap, bind, Option.bind, pure]
    rw [if_neg (by decide), if_neg (by simp), if_neg (by omega)]
    rcases s1.k.next with ⟨c, k'⟩
    cases place (s1.mem.eraseIdx i1) c (2 * cap) with
    | none => rfl
    | some o => cases o <;> rfl
  cases hpl' : place (s1.mem.eraseIdx i1) s1.k.next.1 (2 * cap) with
  | none =>
    cases as
    · exfalso
      have hg := hor1 (Or.inr rfl)
      generalize hrem_s : s1.k.stdin.length = rem_s at hg
      cases hg with
      | grow hp _ =>
        rw [place_eq_placeS, shape_eraseIdx, (head?_eq_next s1.k).1] at hpl'
        rw [hp] at hpl'; cases hpl'
      | done hlt _ => omega
      | step hlt _ _ => omega
    · exact ReachesInit.stuck1 (by rw [hrip1]; exact f29)
        (exec_syscall_none h0 h7 h6 h2 h10 h8_1 hv9 (by rw [hsys_pre, hpl']))
  | some o =>
    cases o with
    | none =>
      cases a6
      · exfalso
        have hg := hor1 (Or.inl rfl)
        generalize hrem_s : s1.k.stdin.length = rem_s at hg
        cases hg with
        | grow hp _ =>
          rw [place_eq_placeS, shape_eraseIdx, (head?_eq_next s1.k).1] at hpl'
          rw [hp] at hpl'; cases hpl'
        | done hlt _ => omega
        | step hlt _ _ => omega
      · have hsys : syscall s1.k s1.mem 25 r.base cap (2 * cap) 1 (W - 1) v9 =
            some (.ret (neg ENOMEM) s1.k.next.2 s1.mem) := by
          rw [hsys_pre, hpl']
        have he := exec_syscall_ret (len := 2) h0 h7 h6 h2 h10 h8_1 hv9 hsys
        let s1' : St := { ((s1.set 0 (some (neg ENOMEM))).set 1 (some (L_rd + 31))).set 11 none with
          k := s1.k.next.2, mem := s1.mem, rip := L_rd + 31 }
        have he' : exec .syscall 2 s1 = .ok s1' := by rw [he, hrip1]
        have hs1' : Reaches s1 (fun s2 => s2.rip = L_nomem ∧ RegDefExcept11 s2) := by
          refine reach_step (by rw [hrip1]; exact f29) ?_
          rw [he']
          have h0' : s1'.reg 0 = some (W - ENOMEM) := by simp only [s1']; rsimpG [hd1.1]
          have hcc : X86Int.cond (aluF .and (W - ENOMEM) (W - ENOMEM)).2 8 = some true := by
            rw [cond_test_s, msb_nomem]
          xs f31 using h0'
          xs f34 using hcc
          exact Reaches.refl ⟨by rfl,
            regDef_set_11_none (regDef_set_some (regDef_set_some hd1 0 _) 1 _)⟩
        exact ReachesInit.exit6 (hs1'.ends fun s2 ⟨hr2, hd2⟩ => nomem_spec hr2 hd2)
    | some b1' =>
      have hpf := place_spec_full hpl'
      have hsys : syscall s1.k s1.mem 25 r.base cap (2 * cap) 1 (W - 1) v9 =
          some (.ret b1' s1.k.next.2
            (s1.mem.eraseIdx i1 ++ [⟨b1', r.data ++ List.replicate (2 * cap - cap) 0, true⟩])) := by
        rw [hsys_pre, hpl']
      have he := exec_syscall_ret (len := 2) h0 h7 h6 h2 h10 h8_1 hv9 hsys
      let m2 : Mem := s1.mem.eraseIdx i1 ++ [⟨b1', r.data ++ List.replicate (2 * cap - cap) 0, true⟩]
      have hri1 : s1.mem[i1]? = some r := by rw [hm1]; exact hri
      have hwf2 : m2.wf := syscall_wf (by rw [hm1]; exact hwf) hsys
      have hbss2 : HasBss m2 := hasBss_eraseIdx_append (by rw [hm1]; exact hbss) hri1 hb1_bss
      have hb1'_ne : b1' ≠ 0x600000 := by
        have hbss_1 : HasBss s1.mem := by rw [hm1]; exact hbss
        obtain ⟨ib0, rb0, hrib0, hb0_eq, hb0_len, _⟩ := hbss_1
        have hne_i : i1 ≠ ib0 := by
          intro h; subst h; rw [hri1] at hrib0; cases hrib0; exact hb1_bss hb0_eq
        rcases Nat.lt_trichotomy ib0 i1 with hlt | heq | hgt
        · have hlt_er : ib0 < (s1.mem.eraseIdx i1).length := by
            have := (List.getElem?_eq_some_iff.1 hri1).1
            rw [List.length_eraseIdx, if_pos this]; omega
          have hr_er : (s1.mem.eraseIdx i1)[ib0]? = some rb0 := by
            rw [List.getElem?_eraseIdx_of_lt hlt]; exact hrib0
          rw [← hb0_eq]
          exact base_ne_of_free hr_er (by omega) (by omega) hpf.2.2.2
        · exact absurd heq.symm hne_i
        · have hi_lt := (List.getElem?_eq_some_iff.1 hri1).1
          have hib_lt := (List.getElem?_eq_some_iff.1 hrib0).1
          have hlt_er : ib0 - 1 < (s1.mem.eraseIdx i1).length := by
            rw [List.length_eraseIdx, if_pos hi_lt]; omega
          have hr_er : (s1.mem.eraseIdx i1)[ib0 - 1]? = some rb0 := by
            rw [List.getElem?_eraseIdx_of_ge (by omega), show ib0 - 1 + 1 = ib0 by omega]; exact hrib0
          rw [← hb0_eq]
          exact base_ne_of_free hr_er (by omega) (by omega) hpf.2.2.2
      have hdlen2 : (r.data ++ List.replicate (2 * cap - cap) 0).length = 2 * cap := by
        simp only [List.length_append, List.length_replicate, hrcap]; omega
      have htake2 : (r.data ++ List.replicate (2 * cap - cap) 0).take pfx.length = pfx := by
        rw [List.take_append_of_le_length (by omega)]; exact htake
      have hbuf2 : HasBufRW m2 (s1.mem.eraseIdx i1).length b1' (2 * cap) pfx :=
        ⟨⟨b1', r.data ++ List.replicate (2 * cap - cap) 0, true⟩,
          by rw [List.getElem?_append_right (Nat.le_refl _), Nat.sub_self]; rfl,
          rfl, hdlen2, rfl,
          hpf.2.1, by omega, hpf.1, h2cap_pg, by omega, hb1'_ne, by omega, htake2⟩
      have hor2 : a6 = false ∨ as = false →
          GoodRdOrac (shape m2) (s1.mem.eraseIdx i1).length cap (2 * cap)
            s1.k.next.2.stdin.length s1.k.next.2.oracle := by
        intro h
        have hg := hor1 h
        generalize hrem_s : s1.k.stdin.length = rem_s at hg
        cases hg with
        | grow hp hnext =>
          rw [place_eq_placeS, shape_eraseIdx, (head?_eq_next s1.k).1] at hpl'
          rw [hp] at hpl'; cases hpl'
          have hsh2 : shape m2 = (shape s1.mem).eraseIdx i1 ++ [(b1', b1' + 2 * cap)] := by
            simp only [m2, shape_append_region, shape_eraseIdx, Region.hi, hdlen2]
          have hdrop := (head?_eq_next s1.k).2
          have hstdin : s1.k.next.2.stdin = s1.k.stdin := by cases ho : s1.k.oracle <;> simp [Kern.next, ho]
          have hlen_er : ((shape s1.mem).eraseIdx i1).length = (s1.mem.eraseIdx i1).length := by
            simp only [shape, List.length_eraseIdx, List.length_map]
          rw [hsh2, hdrop, hstdin, ← hlen_er, hrem_s]
          exact hnext
        | done hlt _ => omega
        | step hlt _ _ => omega
      have hmsb : msb b1' = false := msb_lt (by omega)
      have hcc : X86Int.cond (aluF .and b1' b1').2 8 = some false := by rw [cond_test_s, hmsb]
      refine Reaches.toInit a6 as ?_
      refine reach_step (by rw [hrip1]; exact f29) ?_
      let s1' : St := { ((s1.set 0 (some b1')).set 1 (some (L_rd + 31))).set 11 none with
        k := s1.k.next.2, mem := m2, rip := L_rd + 31 }
      have he' : exec .syscall 2 s1 = .ok s1' := by rw [he, hrip1]
      rw [he']
      have h0' : s1'.reg 0 = some b1' := by simp only [s1']; rsimpG [hd1.1]
      have h12' : s1'.reg 12 = some cap := by simp only [s1']; rsimpG [hd1.1]; exact h12_1
      have h13' : s1'.reg 13 = some cap := by simp only [s1']; rsimpG [hd1.1]; exact h13_1
      have h8' : s1'.reg 8 = some (W - 1) := by simp only [s1']; rsimpG [hd1.1]; exact h8_1
      have hd1' : RegDefExcept11 s1' :=
        regDef_set_11_none (regDef_set_some (regDef_set_some hd1 0 _) 1 _)
      xs f31 using h0'
      xs f34 using hcc
      xs f40 using h0'
      xs f43 using h12'
      refine Reaches.refl ⟨by rfl, ?_, by rsimpG [hd1'.1]; exact hwf2,
        by rsimpG [hd1'.1]; exact hbss2,
        (s1.mem.eraseIdx i1).length, b1', by rsimpG [hd1'.1]; exact hbuf2,
        by rsimpG [hd1'.1], by rsimpG [hd1'.1, hshl'],
        by rsimpG [hd1'.1]; exact h13', by rsimpG [hd1'.1]; exact h8',
        by simp only [s1']; rw [hk1]; cases ho : s.k.oracle <;> simp [Kern.next, ho],
        fun h => by
          simp only [s1']
          simpa [hm1, hk1] using hor2 h⟩
      exact regDef_set_some (regDef_set_some hd1' 14 _) 12 _

theorem step_read_gen {s : St} {i1 b1 cap : Nat} {pfx : List Nat} {a6 as : Bool}
    (hrip : s.rip = L_rd_go) (hd : RegDefExcept11 s)
    (hwf : s.mem.wf) (hbss : HasBss s.mem) (hb1 : HasBufRW s.mem i1 b1 cap pfx)
    (hlt : pfx.length < cap) (hby : ∀ x ∈ s.k.stdin, x < 256)
    (h14 : s.reg 14 = some b1) (h12 : s.reg 12 = some cap) (h13 : s.reg 13 = some pfx.length)
    (h8 : s.reg 8 = some (W - 1))
    (hor : a6 = false ∨ as = false →
      GoodRdOrac (shape s.mem) i1 pfx.length cap s.k.stdin.length s.k.oracle) :
    Reaches s (fun s1 =>
      (s.k.stdin = [] ∧
        s1.rip = L_eof ∧ RegDefExcept11 s1 ∧ s1.mem.wf ∧ HasBss s1.mem ∧
        HasBufRW s1.mem i1 b1 cap pfx ∧
        s1.reg 14 = some b1 ∧ s1.reg 13 = some pfx.length ∧ s1.reg 8 = some (W - 1) ∧
        s1.k.stdin = [] ∧
        (a6 = false ∨ as = false → GoodTailOrac (shape s1.mem) pfx.length s1.k.oracle)) ∨
      (s.k.stdin ≠ [] ∧
        ∃ pfx', pfx ++ s.k.stdin = pfx' ++ s1.k.stdin ∧ pfx.length < pfx'.length ∧
        s1.k.stdin.length < s.k.stdin.length ∧
        s1.rip = L_rd ∧ RegDefExcept11 s1 ∧ s1.mem.wf ∧ HasBss s1.mem ∧
        HasBufRW s1.mem i1 b1 cap pfx' ∧
        s1.reg 14 = some b1 ∧ s1.reg 12 = some cap ∧ s1.reg 13 = some pfx'.length ∧
        s1.reg 8 = some (W - 1) ∧
        (a6 = false ∨ as = false →
          GoodRdOrac (shape s1.mem) i1 pfx'.length cap s1.k.stdin.length s1.k.oracle))) := by
  have hcode : CodeAt [(L_rd_go, .alu .xor 7 7, 2), (L_rd_go + 2, .mov 6 14, 3),
      (L_rd_go + 5, .alu .add 6 13, 3), (L_rd_go + 8, .mov 2 12, 3),
      (L_rd_go + 11, .alu .sub 2 13, 3), (L_rd_go + 14, .alu .xor 0 0, 2),
      (L_rd_go + 16, .syscall, 2), (L_rd_go + 18, .test 0 0, 3),
      (L_rd_go + 21, .jcc 8 L_nomem, 6), (L_rd_go + 27, .jcc 4 L_eof, 2),
      (L_rd_go + 29, .alu .add 13 0, 3), (L_rd_go + 32, .jmp L_rd, 2)] :=
    codeAt_of_init0_sub (by decide +kernel)
  simp only [CodeAt_cons, CodeAt_nil, and_true] at hcode
  obtain ⟨f0, f2, f5, f8, f11, f14, f16, f18, f21, f27, f29, f32⟩ := hcode
  obtain ⟨r, hri, rfl, hrcap, hrw, hb1_pos, hb1_47, hb1_pg, hcap_pg, hcap_64, hb1_bss, hpl_le, htake⟩ := hb1
  obtain ⟨v7, hv7⟩ := regDef_get hd 7
  obtain ⟨v0, hv0⟩ := regDef_get hd 0
  have hcap_W : cap < W := by rw [W_eq]; omega
  have hadd6 : (aluF .add r.base pfx.length).1 = r.base + pfx.length := by
    rw [add_small (by omega)]
  have hsub2 : (aluF .sub cap pfx.length).1 = cap - pfx.length := by
    rw [sub_small hcap_W (by omega)]
  have hsetup : Reaches s (fun s1 =>
      s1.rip = L_rd_go + 16 ∧ RegDefExcept11 s1 ∧
      s1.mem = s.mem ∧ s1.k = s.k ∧
      s1.reg 0 = some 0 ∧ s1.reg 7 = some 0 ∧ s1.reg 6 = some (r.base + pfx.length) ∧
      s1.reg 2 = some (cap - pfx.length) ∧
      s1.reg 14 = some r.base ∧ s1.reg 12 = some cap ∧ s1.reg 13 = some pfx.length ∧
      s1.reg 8 = some (W - 1)) := by
    xs f0 using hv7
    xs f2 using h14
    xs f5 using hd.1, h13
    xs f8 using h12
    xs f11 using hd.1, h13
    xs f14 using hv0
    refine Reaches.refl ⟨by simp only []; omega, ?_, rfl, rfl,
      by rsimpG [hd.1]; exact alu_xor_self v0, by rsimpG [hd.1]; exact alu_xor_self v7,
      by rsimpG [hd.1, hadd6], by rsimpG [hd.1, hsub2],
      by rsimpG [hd.1]; exact h14, by rsimpG [hd.1]; exact h12,
      by rsimpG [hd.1]; exact h13, by rsimpG [hd.1]; exact h8⟩
    exact regDef_set_some (regDef_set_some (regDef_set_some (regDef_set_some (regDef_set_some
      (regDef_set_some hd 7 _) 6 _) 6 _) 2 _) 2 _) 0 _
  refine hsetup.trans fun s1 ⟨hrip1, hd1, hm1, hk1, h0, h7, h6, h2, h14_1, h12_1, h13_1, h8_1⟩ => ?_
  have hor1 : a6 = false ∨ as = false →
      GoodRdOrac (shape s1.mem) i1 pfx.length cap s1.k.stdin.length s1.k.oracle :=
    fun h => by rw [hm1, hk1]; exact hor h
  clear hor
  obtain ⟨v10, hv10⟩ := regDef_get hd1 10
  obtain ⟨v9, hv9⟩ := regDef_get hd1 9
  let n := (s1.k.count (cap - pfx.length) s1.k.stdin.length).1
  let k' := (s1.k.count (cap - pfx.length) s1.k.stdin.length).2
  have hcnt_eq := kern_count_eq_countStep s1.k (cap - pfx.length) s1.k.stdin.length
  have hcnt_b := countStep_bounds s1.k.oracle (cap - pfx.length) s1.k.stdin.length
  have hn_le_c : n ≤ cap - pfx.length := by dsimp only [n]; rw [hcnt_eq.1]; exact hcnt_b.1
  have hn_le_s : n ≤ s1.k.stdin.length := by dsimp only [n]; rw [hcnt_eq.1]; exact hcnt_b.2
  have htake_len : (s1.k.stdin.take n).length = n := by simp [hn_le_s]
  have hby_take : ∀ x ∈ s1.k.stdin.take n, x < 256 := fun x hx =>
    hby x (by rw [← hk1]; exact List.mem_of_mem_take hx)
  have hwf1 : s1.mem.wf := by rw [hm1]; exact hwf
  have hri1 : s1.mem[i1]? = some r := by rw [hm1]; exact hri
  obtain ⟨d', hwrN, hdlen, hd'⟩ := mem_wrN_of_wf (bs := s1.k.stdin.take n) (a := r.base + pfx.length)
    hwf1 hri1 hrw (by omega)
    (by simp only [Region.hi]; omega) hby_take
  let m2 : Mem := s1.mem.set i1 { r with data := d' }
  have hsh2 : shape m2 = shape s1.mem := wrN_shape s1.mem _ _ hwrN
  have hwf2 : m2.wf := wf_of_shape hsh2 hwf1
  have hbss2 : HasBss m2 := hasBss_set_ne (by rw [hm1]; exact hbss) hri1 hb1_bss
  have hsys : syscall s1.k s1.mem 0 0 (r.base + pfx.length) (cap - pfx.length) v10 (W - 1) v9 =
      some (.ret n { k' with stdin := s1.k.stdin.drop n } m2) := by
    simp only [syscall, ↓reduceIte]
    change (do let m' ← s1.mem.wrN (r.base + pfx.length) (s1.k.stdin.take n)
               pure (SysRes.ret n { k' with stdin := s1.k.stdin.drop n } m')) = _
    rw [hwrN]; rfl
  have he := exec_syscall_ret (len := 2) h0 h7 h6 h2 hv10 h8_1 hv9 hsys
  have hn_lt63 : n < 2 ^ 63 := by omega
  have hcc_s : X86Int.cond (aluF .and n n).2 8 = some false := by rw [cond_test_s, msb_lt hn_lt63]
  refine reach_step (by rw [hrip1]; exact f16) ?_
  let s1' : St := { ((s1.set 0 (some n)).set 1 (some (L_rd_go + 18))).set 11 none with
    k := { k' with stdin := s1.k.stdin.drop n }, mem := m2, rip := L_rd_go + 18 }
  have he' : exec .syscall 2 s1 = .ok s1' := by rw [he, hrip1]
  rw [he']
  have h0' : s1'.reg 0 = some n := by simp only [s1']; rsimpG [hd1.1]
  have h13' : s1'.reg 13 = some pfx.length := by simp only [s1']; rsimpG [hd1.1]; exact h13_1
  have h14' : s1'.reg 14 = some r.base := by simp only [s1']; rsimpG [hd1.1]; exact h14_1
  have h12' : s1'.reg 12 = some cap := by simp only [s1']; rsimpG [hd1.1]; exact h12_1
  have h8' : s1'.reg 8 = some (W - 1) := by simp only [s1']; rsimpG [hd1.1]; exact h8_1
  have hd1' : RegDefExcept11 s1' :=
    regDef_set_11_none (regDef_set_some (regDef_set_some hd1 0 _) 1 _)
  have hi1_lt : i1 < s1.mem.length := (List.getElem?_eq_some_iff.1 hri1).1
  have hbuf_take : d'.take (pfx.length + n) = pfx ++ s1.k.stdin.take n := by
    conv => lhs; rw [← htake_len]
    refine list_take_wrN (d := r.data) (bs := s1.k.stdin.take n) (by rw [htake_len]; omega) htake rfl hdlen ?_
    intro k hk
    have hk' := hd' k hk
    simpa [Nat.add_sub_cancel_left] using hk'
  have hbuf2 : HasBufRW m2 i1 r.base cap (pfx ++ s1.k.stdin.take n) :=
    ⟨{ r with data := d' }, by simp [m2, List.getElem?_set_self hi1_lt], rfl,
      by simp [hdlen, hrcap], hrw, hb1_pos, hb1_47, hb1_pg, hcap_pg, hcap_64, hb1_bss,
      by simp [htake_len]; omega, by simpa [htake_len] using hbuf_take⟩
  have hcc_z : X86Int.cond (aluF .and n n).2 4 = some (decide (n = 0)) := cond_test_e
  by_cases hn0 : n = 0
  · have hstdin_nil : s1.k.stdin = [] := by
      have hlen0 : s1.k.stdin.length = 0 :=
        (countStep_zero_iff s1.k.oracle (want := cap - pfx.length) (by omega)).1 (by rw [← hcnt_eq.1]; exact hn0)
      exact List.length_eq_zero_iff.1 hlen0
    have htake_nil : s1.k.stdin.take n = [] := by simp [hn0]
    have hbuf2' : HasBufRW m2 i1 r.base cap pfx := by simpa [htake_nil] using hbuf2
    have hor2 : a6 = false ∨ as = false → GoodTailOrac (shape m2) pfx.length k'.oracle := by
      intro h
      have hlen0 : s1.k.stdin.length = 0 := List.length_eq_zero_iff.2 hstdin_nil
      have hg := hor1 h
      rw [hlen0] at hg
      cases hg with
      | grow => omega
      | done _ htail =>
        have hcs : countStep s1.k.oracle (cap - pfx.length) 0 =
            countStep s1.k.oracle (cap - pfx.length) s1.k.stdin.length := by rw [hlen0]
        rw [hcs, ← hcnt_eq.2.1] at htail
        rw [hsh2]
        exact htail
      | step _ hpos _ => omega
    xs f18 using h0'
    xs f21 using hcc_s
    xs f27 using hn0, hcc_z
    refine Reaches.refl (Or.inl ⟨by rw [← hk1]; exact hstdin_nil, by rfl, hd1',
      by rsimpG [hd1'.1]; exact hwf2, by rsimpG [hd1'.1]; exact hbss2,
      by rsimpG [hd1'.1]; exact hbuf2',
      by rsimpG [hd1'.1]; exact h14', by rsimpG [hd1'.1]; exact h13',
      by rsimpG [hd1'.1]; exact h8',
      by simp [s1', hstdin_nil], fun h => by
        simp only [s1']
        exact hor2 h⟩)
  · have hstdin_ne : s.k.stdin ≠ [] := by
      intro heq; rw [hk1, heq, List.length_nil] at hn_le_s; exact hn0 (by omega)
    let pfx' := pfx ++ s1.k.stdin.take n
    have hlen_pfx' : pfx'.length = pfx.length + n := by simp [pfx', htake_len]
    have hadd13 : (aluF .add pfx.length n).1 = pfx'.length := by
      rw [hlen_pfx', add_small (by omega)]
    have hor2 : a6 = false ∨ as = false →
        GoodRdOrac (shape m2) i1 pfx'.length cap (s1.k.stdin.drop n).length k'.oracle := by
      intro h
      have hg := hor1 h
      generalize hrem_s : s1.k.stdin.length = rem_s at hg
      cases hg with
      | grow => omega
      | done _ _ => omega
      | step _ _ hnext =>
        dsimp only at hnext
        rw [← hrem_s] at hnext
        rw [← hcnt_eq.1, ← hcnt_eq.2.1] at hnext
        rw [hsh2, hlen_pfx', List.length_drop]
        exact hnext
    xs f18 using h0'
    xs f21 using hcc_s
    xs f27 using hn0, hcc_z
    xs f29 using h13', h0'
    xs f32
    refine Reaches.refl (Or.inr ⟨hstdin_ne, pfx', ?_, by rw [hlen_pfx']; omega,
      by simp [s1']; rw [← hk1]; omega, by rfl, regDef_set_some hd1' 13 _,
      by rsimpG [hd1'.1]; exact hwf2, by rsimpG [hd1'.1]; exact hbss2,
      by rsimpG [hd1'.1]; exact hbuf2,
      by rsimpG [hd1'.1]; exact h14', by rsimpG [hd1'.1]; exact h12',
      by rsimpG [hd1'.1, hadd13],
      by rsimpG [hd1'.1]; exact h8', fun h => by
        simp only [s1']
        simpa using hor2 h⟩)
    simp only [s1']
    rw [← hk1]
    symm
    dsimp only [pfx']
    rw [List.append_assoc, List.take_append_drop]

theorem rd_loop_gen {a6 as : Bool} (bs0 : List Nat) (hby : ∀ x ∈ bs0, x < 256) :
    ∀ (rem : Nat) (pfx : List Nat) (i1 b1 cap : Nat) (s : St),
      pfx ++ s.k.stdin = bs0 →
      s.k.stdin.length = rem →
      s.rip = L_rd →
      RegDefExcept11 s →
      s.mem.wf →
      HasBss s.mem →
      HasBufRW s.mem i1 b1 cap pfx →
      s.reg 14 = some b1 → s.reg 12 = some cap → s.reg 13 = some pfx.length →
      s.reg 8 = some (W - 1) →
      (a6 = false ∨ as = false →
        GoodRdOrac (shape s.mem) i1 pfx.length cap s.k.stdin.length s.k.oracle) →
      ReachesInit s (fun s1 =>
        s1.rip = L_eof ∧ RegDefExcept11 s1 ∧ s1.mem.wf ∧ HasBss s1.mem ∧
        ∃ i1' b1' cap', HasBufRW s1.mem i1' b1' cap' bs0 ∧
          s1.reg 14 = some b1' ∧ s1.reg 13 = some bs0.length ∧ s1.reg 8 = some (W - 1) ∧
          (a6 = false ∨ as = false → GoodTailOrac (shape s1.mem) bs0.length s1.k.oracle)) a6 as := by
  intro rem
  induction rem using Nat.strongRecOn with
  | ind rem ih =>
    intro pfx i1 b1 cap s heq hrem hrip hd hwf hbss hb1 h14 h12 h13 h8 hor
    have hby_s : ∀ x ∈ s.k.stdin, x < 256 := fun x hx =>
      hby x (by rw [← heq]; exact List.mem_append_right _ hx)
    have hle_cap : pfx.length ≤ cap := by
      obtain ⟨_, _, _, _, _, _, _, _, _, _, _, h, _⟩ := hb1; exact h
    by_cases hfull : pfx.length = cap
    · have hmremap := step_mremap_gen hrip hd hwf hbss hb1 hfull h14 h12
        (by rw [← hfull]; exact h13) h8 (fun h => by have hh := hor h; rw [hfull] at hh; exact hh)
      refine hmremap.trans ?_
      rintro s1 ⟨hrip1, hd1, hwf1, hbss1, i1', b1', hb1', h14_1, h12_1, h13_1, h8_1, hk1, hor1⟩
      have hlen_lt : pfx.length < 2 * cap := by
        obtain ⟨_, _, _, _, _, _, _, _, _, h64, _, _, _⟩ := hb1'; omega
      have hby1 : ∀ x ∈ s1.k.stdin, x < 256 := fun x hx => by rw [hk1] at hx; exact hby_s x hx
      have hrd := step_read_gen hrip1 hd1 hwf1 hbss1 hb1' hlen_lt hby1 h14_1 h12_1
        (by rw [hfull]; exact h13_1) h8_1 (fun h => by rw [hfull]; exact hor1 h)
      refine (hrd.toInit a6 as).trans ?_
      rintro s2 (⟨hnil, hr2, hd2, hwf2, hbss2, hb1_2, h14_2, h13_2, h8_2, _, hor2⟩ |
          ⟨_, pfx', heq', _, hlt_len, hr2, hd2, hwf2, hbss2, hb1_2, h14_2, h12_2, h13_2, h8_2, hor2⟩)
      · have hpfx_eq : pfx = bs0 := by rw [← hk1, hnil, List.append_nil] at heq; exact heq
        subst hpfx_eq
        exact ReachesInit.refl ⟨hr2, hd2, hwf2, hbss2, i1', b1', 2 * cap, hb1_2, h14_2, h13_2, h8_2, hor2⟩
      · have heq'' : pfx' ++ s2.k.stdin = bs0 := by rw [← heq', hk1, heq]
        have hlt' : s2.k.stdin.length < rem := by rw [← hrem, ← hk1]; exact hlt_len
        exact ih s2.k.stdin.length hlt' pfx' i1' b1' (2 * cap) s2 heq'' rfl hr2 hd2 hwf2 hbss2
          hb1_2 h14_2 h12_2 h13_2 h8_2 hor2
    · have hlt : pfx.length < cap := Nat.lt_of_le_of_ne hle_cap hfull
      have hcode : CodeAt [(L_rd, .cmp 13 12, 3), (L_rd + 3, .jcc 5 L_rd_go, 2)] :=
        codeAt_of_init0_sub (by decide +kernel)
      simp only [CodeAt_cons, CodeAt_nil, and_true] at hcode
      obtain ⟨f0, f3⟩ := hcode
      have hcap_W : cap < W := by
        obtain ⟨_, _, _, _, _, _, hb47, _, _, _, _, _, _⟩ := hb1; rw [W_eq]; omega
      have hc5 : X86Int.cond (aluF .sub pfx.length cap).2 5 = some true := by
        simpa [hfull] using (cond_sub (by omega : pfx.length < W) hcap_W).2.2.2.1
      have hjmp : Reaches s (fun s1 =>
          s1.rip = L_rd_go ∧ RegDefExcept11 s1 ∧ s1.mem = s.mem ∧ s1.k = s.k ∧
          s1.reg 14 = some b1 ∧ s1.reg 12 = some cap ∧ s1.reg 13 = some pfx.length ∧
          s1.reg 8 = some (W - 1)) := by
        xs f0 using h13, h12
        xs f3 using hc5
        exact Reaches.refl ⟨rfl, hd, rfl, rfl, h14, h12, h13, h8⟩
      refine (hjmp.toInit a6 as).trans ?_
      rintro s1 ⟨hr1, hd1, hm1, hk1, h14_1, h12_1, h13_1, h8_1⟩
      have hby1 : ∀ x ∈ s1.k.stdin, x < 256 := fun x hx => by rw [hk1] at hx; exact hby_s x hx
      have hor1 : a6 = false ∨ as = false →
          GoodRdOrac (shape s1.mem) i1 pfx.length cap s1.k.stdin.length s1.k.oracle := fun h => by
        rw [hm1, hk1]; exact hor h
      have hrd := step_read_gen hr1 hd1 (by rwa [hm1]) (by rwa [hm1]) (by rwa [hm1]) hlt hby1
        h14_1 h12_1 h13_1 h8_1 hor1
      refine (hrd.toInit a6 as).trans ?_
      rintro s2 (⟨hnil, hr2, hd2, hwf2, hbss2, hb1_2, h14_2, h13_2, h8_2, _, hor2⟩ |
          ⟨_, pfx', heq', _, hlt_len, hr2, hd2, hwf2, hbss2, hb1_2, h14_2, h12_2, h13_2, h8_2, hor2⟩)
      · have hpfx_eq : pfx = bs0 := by rw [← hk1, hnil, List.append_nil] at heq; exact heq
        subst hpfx_eq
        exact ReachesInit.refl ⟨hr2, hd2, hwf2, hbss2, i1, b1, cap, hb1_2, h14_2, h13_2, h8_2, hor2⟩
      · have heq'' : pfx' ++ s2.k.stdin = bs0 := by rw [← heq', hk1, heq]
        have hlt' : s2.k.stdin.length < rem := by rw [← hrem, ← hk1]; exact hlt_len
        exact ih s2.k.stdin.length hlt' pfx' i1 b1 cap s2 heq'' rfl hr2 hd2 hwf2 hbss2
          hb1_2 h14_2 h12_2 h13_2 h8_2 hor2

private theorem wr64_bss_slot {m : Mem} {ib : Nat} {rb : Region}
    (hwf : m.wf) (hrib : m[ib]? = some rb) (hbase : rb.base = 0x600000)
    (hlen : rb.data.length = 0x3000) (hrw : rb.rw = true)
    (off v : Nat) (hoff : off + 8 ≤ 0x3000) :
    ∃ db : List Nat,
      wr64 m (0x600000 + off) v = some (m.set ib { rb with data := db }) ∧
      db.length = 0x3000 ∧
      (∀ j, j < 8 → db[off + j]? = (bytes8 v)[j]?) ∧
      (∀ k, k < 0x3000 → ¬ (off ≤ k ∧ k < off + 8) → db[k]? = rb.data[k]?) := by
  have ha1 : rb.base ≤ 0x600000 + off := by omega
  have ha2 : 0x600000 + off + 8 ≤ rb.hi := by simp only [Region.hi]; omega
  obtain ⟨db, hwr, hdlen, hd⟩ := wr64_region_step hwf hrib hrw ha1 ha2
  refine ⟨db, hwr, by simpa [hlen] using hdlen, ?_, ?_⟩
  · intro j hj
    have hlt : off + j < rb.data.length := by omega
    have hdk := hd (off + j) hlt
    have hin : 0x600000 + off - rb.base ≤ off + j ∧ off + j < 0x600000 + off - rb.base + 8 := by omega
    rw [hdk, if_pos hin, show off + j - (0x600000 + off - rb.base) = j by omega]
  · intro k hk hnot
    have hlt : k < rb.data.length := by omega
    have hdk := hd k hlt
    have hnot' : ¬ (0x600000 + off - rb.base ≤ k ∧ k < 0x600000 + off - rb.base + 8) := by omega
    rw [hdk, if_neg hnot']

private theorem buf_of_region_eq {m m' : Mem} (hw : m.wf) (hw' : m'.wf) {i : Nat} {r : Region}
    (h1 : m[i]? = some r) (h2 : m'[i]? = some r)
    {b : Nat} (hb : r.base = b) {bs : List Nat} (hle : b + bs.length ≤ r.hi)
    (hbuf : Buf m b bs) : Buf m' b bs := by
  intro k hk
  have ha1 : r.base ≤ b + k := by omega
  have ha2 : b + k < r.hi := by omega
  rw [mem_rd_of_wf hw' h2 ha1 ha2, ← mem_rd_of_wf hw h1 ha1 ha2]
  exact hbuf k hk

private theorem exec_wr64_bss_pair {L off ib rPtr : Nat}
    (hc : CodeAt [(L, .movI 7 (0x600000 + off), 5), (L + 5, .st 7 0 rPtr, 3), (L + 8, .st 7 8 13, 4)])
    (hrPtr_ne : rPtr ≠ 7) (h13_ne : (13 : Nat) ≠ 7)
    (hoff : off + 16 ≤ 0x3000) {s : St} {rb : Region} {b len : Nat}
    (hrip : s.rip = L) (hd : RegDefExcept11 s)
    (hwf : s.mem.wf) (hrib : s.mem[ib]? = some rb) (hbase : rb.base = 0x600000)
    (hlen_rb : rb.data.length = 0x3000) (hrw : rb.rw = true)
    (hrp : s.reg rPtr = some b) (h13 : s.reg 13 = some len) :
    Reaches s (fun s' =>
      s'.rip = L + 12 ∧ RegDefExcept11 s' ∧ s'.mem.wf ∧
      (∃ db : List Nat, s'.mem = s.mem.set ib { rb with data := db } ∧ db.length = 0x3000 ∧
        (∀ j, j < 8 → db[off + j]? = (bytes8 b)[j]?) ∧
        (∀ j, j < 8 → db[(off + 8) + j]? = (bytes8 len)[j]?) ∧
        (∀ k, k < 0x3000 → ¬ (off ≤ k ∧ k < off + 16) → db[k]? = rb.data[k]?)) ∧
      (∀ r, r ≠ 7 → s'.reg r = s.reg r)) := by
  simp only [CodeAt_cons, CodeAt_nil, and_true] at hc
  obtain ⟨f0, f5, f8⟩ := hc
  have hlt_ib : ib < s.mem.length := (List.getElem?_eq_some_iff.1 hrib).1
  obtain ⟨db1, hwr1, hdlen1, hslot1, hrest1⟩ :=
    wr64_bss_slot hwf hrib hbase hlen_rb hrw off b (by omega)
  let rb1 : Region := { rb with data := db1 }
  let m1 : Mem := s.mem.set ib rb1
  have hwf1 : m1.wf := wf_of_shape (wrN_shape _ _ _ hwr1) hwf
  have hrib1 : m1[ib]? = some rb1 := by simp [m1, List.getElem?_set_self hlt_ib]
  obtain ⟨db2, hwr2, hdlen2, hslot2, hrest2⟩ :=
    wr64_bss_slot hwf1 hrib1 hbase hdlen1 hrw (off + 8) len (by omega)
  have hwf2 : Mem.wf (m1.set ib { rb1 with data := db2 }) := wf_of_shape (wrN_shape _ _ _ hwr2) hwf1
  have hea0 : ea (0x600000 + off) 0 = 0x600000 + off := by
    rw [ea_small (by omega)]; omega
  have hea8 : ea (0x600000 + off) 8 = 0x600000 + (off + 8) := by
    rw [ea_small (by omega)]; omega
  have hmod : (0x600000 + off) % 18446744073709551616 = 0x600000 + off :=
    Nat.mod_eq_of_lt (by omega)
  have hne_7p : (7 : Nat) ≠ rPtr := Ne.symm hrPtr_ne
  have hne_713 : (7 : Nat) ≠ 13 := Ne.symm h13_ne
  have hwr1' : wr64 s.mem (ea (0x600000 + off) 0) b = some m1 := by rwa [hea0]
  have hwr2' : wr64 m1 (ea (0x600000 + off) 8) len =
      some (s.mem.set ib { rb with data := db2 }) := by
    rw [hea8, hwr2]
    simp [m1, List.set_set, rb1]
  have hval : (0x600000 + off) % W = 0x600000 + off := by rw [W_eq]; exact hmod
  have hr7_1 : (s.set 7 (some (0x600000 + off))).reg 7 = some (0x600000 + off) :=
    reg_set_eq (by rw [hd.1]; omega)
  have hrp1 : (s.set 7 (some (0x600000 + off))).reg rPtr = some b :=
    (reg_set_ne hne_7p).trans hrp
  have h13_1 : (s.set 7 (some (0x600000 + off))).reg 13 = some len :=
    (reg_set_ne hne_713).trans h13
  let s1 : St := { s.set 7 (some (0x600000 + off)) with rip := L + 5 }
  have hs1_eq : exec (.movI 7 (0x600000 + off)) 5 s = .ok s1 := by
    rw [exec_movI, hrip, hval]
  let s2 : St := { s1 with mem := m1, rip := L + 5 + 3 }
  have h7s1 : s1.reg 7 = some (0x600000 + off) := hr7_1
  have hps1 : s1.reg rPtr = some b := hrp1
  have hm1s : s1.mem = s.mem := rfl
  have hs2_eq : exec (.st 7 0 rPtr) 3 s1 = .ok s2 := by
    simp only [xst, h7s1, hps1, hm1s, hwr1']; rfl
  let s3 : St := { s2 with mem := s.mem.set ib { rb with data := db2 }, rip := L + 5 + 3 + 4 }
  have h7s2 : s2.reg 7 = some (0x600000 + off) := hr7_1
  have h13s2 : s2.reg 13 = some len := h13_1
  have hm2s : s2.mem = m1 := rfl
  have hs3_eq : exec (.st 7 8 13) 4 s2 = .ok s3 := by
    simp only [xst, h7s2, h13s2, hm2s, hwr2']; rfl
  refine reach_step (by rw [hrip]; exact f0) ?_
  rw [hs1_eq]
  refine reach_step f5 ?_
  rw [hs2_eq]
  refine reach_step (by change im (L + 5 + 3) = _; rw [show L + 5 + 3 = L + 8 by omega]; exact f8) ?_
  rw [hs3_eq]
  refine Reaches.refl ⟨by change L + 5 + 3 + 4 = L + 12; omega, regDef_set_some hd 7 _,
    by change Mem.wf (s.mem.set ib { rb with data := db2 }); simpa [m1, List.set_set, rb1] using hwf2,
    ⟨db2, rfl, hdlen2, ?_, hslot2, ?_⟩, ?_⟩
  · intro j hj
    rw [hrest2 (off + j) (by omega) (by omega)]
    exact hslot1 j hj
  · intro k hk hnot
    rw [hrest2 k hk (by omega), hrest1 k hk (by omega)]
  · intro r hr
    exact reg_set_ne (Ne.symm hr)

theorem eof_to_tripled_gen {a6 as : Bool} {bs : List Nat} (hby : ∀ x ∈ bs, x < 256)
    {i1 b1 cap : Nat} {s : St}
    (hrip : s.rip = L_eof) (hd : RegDefExcept11 s)
    (hwf : s.mem.wf) (hbss : HasBss s.mem) (hb1 : HasBufRW s.mem i1 b1 cap bs)
    (h14 : s.reg 14 = some b1) (h13 : s.reg 13 = some bs.length) (h8 : s.reg 8 = some (W - 1))
    (hor : a6 = false ∨ as = false → GoodTailOrac (shape s.mem) bs.length s.k.oracle) :
    ReachesInit s (fun s' =>
      s'.rip = L_tripled ∧
      RegDefExcept11 s' ∧
      s'.mem.wf ∧
      s'.reg 0 = some REJECT ∧ s'.reg 3 = some REJECT ∧ s'.reg 1 = some REJECT ∧
      s'.reg 8 = some (W - 1) ∧
      ∃ b2 b3,
        Buf s'.mem b1 bs ∧ Desc s'.mem 0x600000 b1 bs.length ∧ b1 + bs.length < 2 ^ 48 ∧
        Buf s'.mem b2 bs ∧ Desc s'.mem 0x601000 b2 bs.length ∧ b2 + bs.length < 2 ^ 48 ∧
        Buf s'.mem b3 bs ∧ Desc s'.mem 0x602000 b3 bs.length ∧ b3 + bs.length < 2 ^ 48) a6 as := by
  rw [show L_eof = 4210491 by rfl] at hrip
  have hb1_cp := hb1
  obtain ⟨r1, hr1i, rfl, hr1cap, _, hb1_pos, hb1_47, _, _, hb1_64, hb1_bss, hbs_le, _⟩ := hb1_cp
  have hbs_47 : bs.length < 2 ^ 47 := by omega
  have hbs_W : bs.length < W := by rw [W_eq]; omega
  have hea_len : ea bs.length 1 = bs.length + 1 := ea_small (by omega)
  have hpg_nz : pages (bs.length + 1) ≠ 0 := by
    have := pages_ge (bs.length + 1); omega
  have hlen1_pos : bs.length + 1 ≠ 0 := by omega
  have hmmap_len : (List.replicate (pages (bs.length + 1)) 0).length = pages (bs.length + 1) :=
    List.length_replicate
  -- Code facts for init_0 tail (4210491 .. 4210592) and init_1 (4210583 .. 4210701)
  have hcode0 : CodeAt [(4210491, .alu .xor 7 7, 2), (4210493, .lea 6 13 1, 4),
      (4210497, .movI 2 3, 5), (4210502, .movI 10 34, 6), (4210508, .alu .xor 9 9, 3),
      (4210511, .movI 0 9, 5), (4210516, .syscall, 2), (4210518, .test 0 0, 3),
      (4210521, .jcc 8 L_nomem, 6), (4210527, .mov 15 0, 3), (4210530, .mov 6 14, 3),
      (4210533, .mov 7 15, 3), (4210536, .mov 11 13, 3),
      (4210539, .test 11 11, 3), (4210539 + 3, .jcc 4 4210563, 2),
      (4210539 + 5, .ld8 2 6 0, 3), (4210539 + 8, .st8 7 0 2, 2),
      (4210539 + 10, .aluI .add 6 1, 4), (4210539 + 14, .aluI .add 7 1, 4),
      (4210539 + 18, .aluI .sub 11 1, 4), (4210539 + 22, .jmp 4210539, 2),
      (4210563, .alu .xor 7 7, 2), (4210565, .lea 6 13 1, 4), (4210569, .movI 2 3, 5),
      (4210574, .movI 10 34, 6), (4210580, .alu .xor 9 9, 3)] :=
    codeAt_of_init0_sub (by decide +kernel)
  simp only [CodeAt_cons, CodeAt_nil, and_true] at hcode0
  obtain ⟨f0, f2, f6, f11, f17, f20, f25, f27, f30, f36, f39, f42, f45,
    c1a, c1b, c1c, c1d, c1e, c1f, c1g, c1h, g0, g2, g6, g11, g17⟩ := hcode0
  have hcp1 : CopyCode 4210539 4210563 := by
    simp only [CopyCode, CodeAt_cons, CodeAt_nil, and_true]
    exact ⟨c1a, c1b, c1c, c1d, c1e, c1f, c1g, c1h⟩
  have hcode1 : CodeAt [(4210583, .movI 0 9, 5), (4210588, .syscall, 2),
      (4210590, .test 0 0, 3), (4210593, .jcc 8 L_nomem, 6), (4210599, .mov 5 0, 3),
      (4210602, .mov 6 14, 3), (4210605, .mov 7 5, 3), (4210608, .mov 11 13, 3),
      (4210611, .test 11 11, 3), (4210611 + 3, .jcc 4 4210635, 2),
      (4210611 + 5, .ld8 2 6 0, 3), (4210611 + 8, .st8 7 0 2, 2),
      (4210611 + 10, .aluI .add 6 1, 4), (4210611 + 14, .aluI .add 7 1, 4),
      (4210611 + 18, .aluI .sub 11 1, 4), (4210611 + 22, .jmp 4210611, 2),
      (4210635, .movI 7 6291456, 5), (4210640, .st 7 0 14, 3), (4210643, .st 7 8 13, 4),
      (4210647, .movI 7 6295552, 5), (4210652, .st 7 0 15, 3), (4210655, .st 7 8 13, 4),
      (4210659, .movI 7 6299648, 5), (4210664, .st 7 0 5, 3), (4210667, .st 7 8 13, 4),
      (4210671, .movI 0 REJECT, 10), (4210681, .movI 3 REJECT, 10),
      (4210691, .movI 1 REJECT, 10), (4210701, .jmp L_tripled, 5)] :=
    codeAt_of_init1_sub (by decide +kernel)
  simp only [CodeAt_cons, CodeAt_nil, and_true] at hcode1
  obtain ⟨g20, g25, g27, g30, g36, g39, g42, g45,
    c2a, c2b, c2c, c2d, c2e, c2f, c2g, c2h,
    d0, d5, d8, d12, d17, d20, d24, d29, d32, d36, d46, d56, d66⟩ := hcode1
  have hcp2 : CopyCode 4210611 4210635 := by
    simp only [CopyCode, CodeAt_cons, CodeAt_nil, and_true]
    exact ⟨c2a, c2b, c2c, c2d, c2e, c2f, c2g, c2h⟩
  -- Phase 1: mmap copy 2 (b2)
  obtain ⟨v7, hv7⟩ := regDef_get hd 7
  obtain ⟨v9, hv9⟩ := regDef_get hd 9
  have hsetup1 : Reaches s (fun s1 =>
      s1.rip = 4210516 ∧ RegDefExcept11 s1 ∧
      s1.mem = s.mem ∧ s1.k = s.k ∧
      s1.reg 0 = some 9 ∧ s1.reg 7 = some 0 ∧ s1.reg 6 = some (bs.length + 1) ∧
      s1.reg 2 = some 3 ∧ s1.reg 10 = some 34 ∧ s1.reg 8 = some (W - 1) ∧ s1.reg 9 = some 0 ∧
      s1.reg 14 = some r1.base ∧ s1.reg 13 = some bs.length) := by
    xs f0 using hv7
    xs f2 using h13
    xs f6
    xs f11
    xs f17 using hv9
    xs f20
    refine Reaches.refl ⟨by dsimp only; omega, ?_, rfl, rfl, by rsimpG [hd.1],
      by rsimpG [hd.1]; exact alu_xor_self v7,
      by rsimpG [hd.1, hea_len], by rsimpG [hd.1], by rsimpG [hd.1],
      by rsimpG [hd.1]; exact h8,
      by rsimpG [hd.1]; exact alu_xor_self v9, by rsimpG [hd.1]; exact h14,
      by rsimpG [hd.1]; exact h13⟩
    exact regDef_set_some (regDef_set_some (regDef_set_some (regDef_set_some (regDef_set_some
      (regDef_set_some hd 7 _) 6 _) 2 _) 10 _) 9 _) 0 _
  refine ReachesInit.trans (hsetup1.toInit a6 as) ?_
  rintro s1 ⟨hrip1, hd1, hm1, hk1, h0_1, h7_1, h6_1, h2_1, h10_1, h8_1, h9_1, h14_1, h13_1⟩
  have hor1 : a6 = false ∨ as = false → GoodTailOrac (shape s1.mem) bs.length s1.k.oracle :=
    fun h => by rw [hm1, hk1]; exact hor h
  clear hor
  have hsys1_pre : syscall s1.k s1.mem 9 0 (bs.length + 1) 3 34 (W - 1) 0 =
      (match place s1.mem s1.k.next.1 (pages (bs.length + 1)) with
      | none => none
      | some none => some (.ret (neg ENOMEM) s1.k.next.2 s1.mem)
      | some (some a) => some (.ret a s1.k.next.2
          (s1.mem ++ [⟨a, List.replicate (pages (bs.length + 1)) 0, true⟩]))) := by
    simp only [syscall, PROT_RW, MAP_PRIVATE_ANON, hlen1_pos, ↓reduceIte, bind, Option.bind, pure]
    rcases s1.k.next with ⟨c, k'⟩
    cases place s1.mem c (pages (bs.length + 1)) with
    | none => rfl
    | some o => cases o <;> rfl
  cases hpl2 : place s1.mem s1.k.next.1 (pages (bs.length + 1)) with
  | none =>
    cases as
    · exfalso
      obtain ⟨hp2, _⟩ := hor1 (Or.inr rfl)
      rw [place_eq_placeS, (head?_eq_next s1.k).1] at hpl2
      rw [hp2] at hpl2; cases hpl2
    · exact ReachesInit.stuck1 (by rw [hrip1]; exact f25)
        (exec_syscall_none h0_1 h7_1 h6_1 h2_1 h10_1 h8_1 h9_1 (by rw [hsys1_pre, hpl2]))
  | some o2 =>
    cases o2 with
    | none =>
      cases a6
      · exfalso
        obtain ⟨hp2, _⟩ := hor1 (Or.inl rfl)
        rw [place_eq_placeS, (head?_eq_next s1.k).1] at hpl2
        rw [hp2] at hpl2; cases hpl2
      · have hsys : syscall s1.k s1.mem 9 0 (bs.length + 1) 3 34 (W - 1) 0 =
            some (.ret (neg ENOMEM) s1.k.next.2 s1.mem) := by
          rw [hsys1_pre, hpl2]
        have he := exec_syscall_ret (len := 2) h0_1 h7_1 h6_1 h2_1 h10_1 h8_1 h9_1 hsys
        let s1' : St := { ((s1.set 0 (some (neg ENOMEM))).set 1 (some 4210518)).set 11 none with
          k := s1.k.next.2, mem := s1.mem, rip := 4210518 }
        have he' : exec .syscall 2 s1 = .ok s1' := by rw [he, hrip1]
        have hs1' : Reaches s1 (fun s2 => s2.rip = L_nomem ∧ RegDefExcept11 s2) := by
          refine reach_step (by rw [hrip1]; exact f25) ?_
          rw [he']
          have h0' : s1'.reg 0 = some (W - ENOMEM) := by simp only [s1']; rsimpG [hd1.1]
          have hcc : X86Int.cond (aluF .and (W - ENOMEM) (W - ENOMEM)).2 8 = some true := by
            rw [cond_test_s, msb_nomem]
          xs f27 using h0'
          xs f30 using hcc
          exact Reaches.refl ⟨by rfl,
            regDef_set_11_none (regDef_set_some (regDef_set_some hd1 0 _) 1 _)⟩
        exact ReachesInit.exit6 (hs1'.ends fun s2 ⟨hr2, hd2⟩ => nomem_spec hr2 hd2)
    | some b2 =>
      have hpf2 := place_spec_full hpl2
      have hsys1 : syscall s1.k s1.mem 9 0 (bs.length + 1) 3 34 (W - 1) 0 =
          some (.ret b2 s1.k.next.2
            (s1.mem ++ [⟨b2, List.replicate (pages (bs.length + 1)) 0, true⟩])) := by
        rw [hsys1_pre, hpl2]
      have he1 := exec_syscall_ret (len := 2) h0_1 h7_1 h6_1 h2_1 h10_1 h8_1 h9_1 hsys1
      let m2 : Mem := s1.mem ++ [⟨b2, List.replicate (pages (bs.length + 1)) 0, true⟩]
      have hwf2 : m2.wf := syscall_wf (by rw [hm1]; exact hwf) hsys1
      have hbss2 : HasBss m2 := hasBss_append (by rw [hm1]; exact hbss) _
      have hb1_2 : HasBufRW m2 i1 r1.base cap bs := hasBufRW_append (by rw [hm1]; exact hb1) _
      have hne_12 : b2 ≠ r1.base := by
        have h1m : s1.mem[i1]? = some r1 := by rw [hm1]; exact hr1i
        exact base_ne_of_free h1m (by omega) (by omega) hpf2.2.2.2
      have hne_bss2 : b2 ≠ 0x600000 := by
        have hbss_1 : HasBss s1.mem := by rw [hm1]; exact hbss
        obtain ⟨ib, rb, hrib, hb_eq, hblen, _⟩ := hbss_1
        rw [← hb_eq]
        exact base_ne_of_free hrib (by omega) (by omega) hpf2.2.2.2
      have hb2_len : bs.length ≤ (List.replicate (pages (bs.length + 1)) 0).length := by
        rw [hmmap_len]; have := pages_ge (bs.length + 1); omega
      have hb2_lt48 : b2 + bs.length < 2 ^ 48 := by
        have := pages_ge (bs.length + 1); omega
      have hdrop2 : s1.k.next.2.oracle = s1.k.oracle.drop 1 := (head?_eq_next s1.k).2
      have hor2 : a6 = false ∨ as = false →
          placeS (shape m2) s1.k.next.2.oracle.head? (pages (bs.length + 1)) =
            placeS (shape s1.mem ++ [(b2, b2 + pages (bs.length + 1))]) (s1.k.oracle.drop 1).head?
              (pages (bs.length + 1)) := by
        intro _
        rw [hdrop2, shape_append_region, Region.hi, hmmap_len]
      have hp3_of : a6 = false ∨ as = false →
          ∃ b3, placeS (shape m2) s1.k.next.2.oracle.head? (pages (bs.length + 1)) = some (some b3) := by
        intro h
        obtain ⟨hp2, hp3⟩ := hor1 h
        rw [place_eq_placeS, (head?_eq_next s1.k).1] at hpl2
        rw [hp2] at hpl2; cases hpl2
        rw [hor2 h]
        exact ⟨_, hp3⟩
      have hmsb2 : msb b2 = false := msb_lt (by omega)
      have hcc2 : X86Int.cond (aluF .and b2 b2).2 8 = some false := by rw [cond_test_s, hmsb2]
      let s1' : St := { ((s1.set 0 (some b2)).set 1 (some 4210518)).set 11 none with
        k := s1.k.next.2, mem := m2, rip := 4210518 }
      have he1' : exec .syscall 2 s1 = .ok s1' := by rw [he1, hrip1]
      have hpre_cp1 : Reaches s1 (fun s2 =>
          s2.rip = 4210539 ∧ RegDefExcept11 s2 ∧ s2.mem = m2 ∧ s2.k = s1.k.next.2 ∧
          s2.reg 14 = some r1.base ∧ s2.reg 15 = some b2 ∧ s2.reg 13 = some bs.length ∧
          s2.reg 8 = some (W - 1) ∧ s2.reg 6 = some (r1.base + 0) ∧
          s2.reg 7 = some (b2 + 0) ∧ s2.reg 11 = some bs.length) := by
        refine reach_step (by rw [hrip1]; exact f25) ?_
        rw [he1']
        have h0' : s1'.reg 0 = some b2 := by simp only [s1']; rsimpG [hd1.1]
        have h14' : s1'.reg 14 = some r1.base := by simp only [s1']; rsimpG [hd1.1]; exact h14_1
        have h13' : s1'.reg 13 = some bs.length := by simp only [s1']; rsimpG [hd1.1]; exact h13_1
        have h8' : s1'.reg 8 = some (W - 1) := by simp only [s1']; rsimpG [hd1.1]; exact h8_1
        have hd1' : RegDefExcept11 s1' :=
          regDef_set_11_none (regDef_set_some (regDef_set_some hd1 0 _) 1 _)
        xs f27 using h0'
        xs f30 using hcc2
        xs f36 using h0'
        xs f39 using h14'
        xs f42 using hd1'.1
        xs f45 using h13'
        refine Reaches.refl ⟨by rfl, ?_, rfl, rfl,
          by rsimpG [hd1'.1]; exact h14', by rsimpG [hd1'.1],
          by rsimpG [hd1'.1]; exact h13', by rsimpG [hd1'.1]; exact h8',
          by simp only [Nat.add_zero]; rsimpG [hd1'.1],
          by simp only [Nat.add_zero]; rsimpG [hd1'.1],
          by rsimpG [hd1'.1]⟩
        exact regDef_set_some (regDef_set_some (regDef_set_some (regDef_set_some hd1' 15 _) 6 _) 7 _) 11 _
      have hcp1_done : Reaches s1 (fun s2 =>
          s2.rip = 4210563 ∧ RegDefExcept11 s2 ∧ s2.mem.wf ∧ shape s2.mem = shape m2 ∧
          HasBss s2.mem ∧ HasBufRW s2.mem i1 r1.base cap bs ∧
          (∃ (i2 : Nat) (r2 : Region), s2.mem[i2]? = some r2 ∧ r2.base = b2 ∧ r2.rw = true ∧
            bs.length ≤ r2.data.length ∧ 0 < r2.data.length ∧ Buf s2.mem b2 bs) ∧
          s2.reg 14 = some r1.base ∧ s2.reg 15 = some b2 ∧
          s2.reg 13 = some bs.length ∧ s2.reg 8 = some (W - 1) ∧ s2.k = s1.k.next.2) := by
        refine hpre_cp1.trans fun s2 ⟨hr2, hd2, hm2, hk2, h14_2, h15_2, h13_2, h8_2, h6_2, h7_2, h11_2⟩ => ?_
        have htake2 : (List.replicate (pages (bs.length + 1)) 0).take 0 = bs.take 0 := by simp
        have hcp := copy_loop_spec hcp1 hby (by omega) hb2_lt48 hne_12 hne_bss2 bs.length 0 s2
          (Nat.zero_add _) hr2 hd2 (by rwa [hm2]) (by rwa [hm2]) (by rwa [hm2])
          ⟨s1.mem.length, ⟨b2, List.replicate (pages (bs.length + 1)) 0, true⟩,
            by rw [hm2]; simp [m2], rfl, rfl, hb2_len, by rw [hmmap_len]; omega, htake2⟩
          h6_2 h7_2 h11_2
        refine hcp.mono fun s3 ⟨hr3, hd3, hwf3, hsh3, hbss3, hb1_3, hr2_3, _, h14_3, h15_3, _, h13_3, h8_3, hk3⟩ => ?_
        exact ⟨hr3, hd3, hwf3, by rw [hsh3, hm2], hbss3, hb1_3, hr2_3,
          h14_3.trans h14_2, h15_3.trans h15_2, h13_3.trans h13_2, h8_3.trans h8_2, hk3.trans hk2⟩
      -- Phase 2: mmap copy 3 (b3)
      refine ReachesInit.trans (hcp1_done.toInit a6 as) ?_
      rintro s2 ⟨hr2, hd2, hwf2, hsh2, hbss2, hb1_2, ⟨i2, r2, hr2i, r2base, r2rw, hr2len, hr2pos, hbuf2⟩,
          h14_2, h15_2, h13_2, h8_2, hk2⟩
      obtain ⟨v7_2, hv7_2⟩ := regDef_get hd2 7
      obtain ⟨v9_2, hv9_2⟩ := regDef_get hd2 9
      have hsetup2 : Reaches s2 (fun s3 =>
          s3.rip = 4210588 ∧ RegDefExcept11 s3 ∧ s3.mem = s2.mem ∧ s3.k = s2.k ∧
          s3.reg 0 = some 9 ∧ s3.reg 7 = some 0 ∧ s3.reg 6 = some (bs.length + 1) ∧
          s3.reg 2 = some 3 ∧ s3.reg 10 = some 34 ∧ s3.reg 8 = some (W - 1) ∧ s3.reg 9 = some 0 ∧
          s3.reg 14 = some r1.base ∧ s3.reg 15 = some b2 ∧ s3.reg 13 = some bs.length) := by
        xs g0 using hv7_2
        xs g2 using h13_2
        xs g6
        xs g11
        xs g17 using hv9_2
        xs g20
        refine Reaches.refl ⟨by dsimp only; omega, ?_, rfl, rfl, by rsimpG [hd2.1],
          by rsimpG [hd2.1]; exact alu_xor_self v7_2,
          by rsimpG [hd2.1, hea_len], by rsimpG [hd2.1], by rsimpG [hd2.1],
          by rsimpG [hd2.1]; exact h8_2,
          by rsimpG [hd2.1]; exact alu_xor_self v9_2,
          by rsimpG [hd2.1]; exact h14_2, by rsimpG [hd2.1]; exact h15_2,
          by rsimpG [hd2.1]; exact h13_2⟩
        exact regDef_set_some (regDef_set_some (regDef_set_some (regDef_set_some (regDef_set_some
          (regDef_set_some hd2 7 _) 6 _) 2 _) 10 _) 9 _) 0 _
      refine ReachesInit.trans (hsetup2.toInit a6 as) ?_
      rintro s3 ⟨hrip3, hd3, hm3, hk3, h0_3, h7_3, h6_3, h2_3, h10_3, h8_3, h9_3, h14_3, h15_3, h13_3⟩
      have hsys2_pre : syscall s3.k s3.mem 9 0 (bs.length + 1) 3 34 (W - 1) 0 =
          (match place s3.mem s3.k.next.1 (pages (bs.length + 1)) with
          | none => none
          | some none => some (.ret (neg ENOMEM) s3.k.next.2 s3.mem)
          | some (some a) => some (.ret a s3.k.next.2
              (s3.mem ++ [⟨a, List.replicate (pages (bs.length + 1)) 0, true⟩]))) := by
        simp only [syscall, PROT_RW, MAP_PRIVATE_ANON, hlen1_pos, ↓reduceIte, bind, Option.bind, pure]
        rcases s3.k.next with ⟨c, k'⟩
        cases place s3.mem c (pages (bs.length + 1)) with
        | none => rfl
        | some o => cases o <;> rfl
      cases hpl3 : place s3.mem s3.k.next.1 (pages (bs.length + 1)) with
      | none =>
        cases as
        · exfalso
          obtain ⟨b3, hp3⟩ := hp3_of (Or.inr rfl)
          rw [place_eq_placeS, hm3, hsh2, hk3, hk2, (head?_eq_next s1.k.next.2).1] at hpl3
          rw [hp3] at hpl3; cases hpl3
        · exact ReachesInit.stuck1 (by rw [hrip3]; exact g25)
            (exec_syscall_none h0_3 h7_3 h6_3 h2_3 h10_3 h8_3 h9_3 (by rw [hsys2_pre, hpl3]))
      | some o3 =>
        cases o3 with
        | none =>
          cases a6
          · exfalso
            obtain ⟨b3, hp3⟩ := hp3_of (Or.inl rfl)
            rw [place_eq_placeS, hm3, hsh2, hk3, hk2, (head?_eq_next s1.k.next.2).1] at hpl3
            rw [hp3] at hpl3; cases hpl3
          · have hsys : syscall s3.k s3.mem 9 0 (bs.length + 1) 3 34 (W - 1) 0 =
                some (.ret (neg ENOMEM) s3.k.next.2 s3.mem) := by
              rw [hsys2_pre, hpl3]
            have he := exec_syscall_ret (len := 2) h0_3 h7_3 h6_3 h2_3 h10_3 h8_3 h9_3 hsys
            let s3' : St := { ((s3.set 0 (some (neg ENOMEM))).set 1 (some 4210590)).set 11 none with
              k := s3.k.next.2, mem := s3.mem, rip := 4210590 }
            have he' : exec .syscall 2 s3 = .ok s3' := by rw [he, hrip3]
            have hs3' : Reaches s3 (fun s4 => s4.rip = L_nomem ∧ RegDefExcept11 s4) := by
              refine reach_step (by rw [hrip3]; exact g25) ?_
              rw [he']
              have h0' : s3'.reg 0 = some (W - ENOMEM) := by simp only [s3']; rsimpG [hd3.1]
              have hcc : X86Int.cond (aluF .and (W - ENOMEM) (W - ENOMEM)).2 8 = some true := by
                rw [cond_test_s, msb_nomem]
              xs g27 using h0'
              xs g30 using hcc
              exact Reaches.refl ⟨by rfl,
                regDef_set_11_none (regDef_set_some (regDef_set_some hd3 0 _) 1 _)⟩
            exact ReachesInit.exit6 (hs3'.ends fun s4 ⟨hr4, hd4⟩ => nomem_spec hr4 hd4)
        | some b3 =>
          have hpf3 := place_spec_full hpl3
          have hsys2 : syscall s3.k s3.mem 9 0 (bs.length + 1) 3 34 (W - 1) 0 =
              some (.ret b3 s3.k.next.2
                (s3.mem ++ [⟨b3, List.replicate (pages (bs.length + 1)) 0, true⟩])) := by
            rw [hsys2_pre, hpl3]
          have he2 := exec_syscall_ret (len := 2) h0_3 h7_3 h6_3 h2_3 h10_3 h8_3 h9_3 hsys2
          let m4 : Mem := s3.mem ++ [⟨b3, List.replicate (pages (bs.length + 1)) 0, true⟩]
          have hwf4 : m4.wf := syscall_wf (by rw [hm3]; exact hwf2) hsys2
          have hbss4 : HasBss m4 := hasBss_append (by rw [hm3]; exact hbss2) _
          have hb1_4 : HasBufRW m4 i1 r1.base cap bs := hasBufRW_append (by rw [hm3]; exact hb1_2) _
          have hbuf2_4 : Buf m4 b2 bs := buf_append (by rw [hm3]; exact hbuf2) _
          have hr2i_4 : m4[i2]? = some r2 := by
            have hlt : i2 < s3.mem.length := by rw [hm3]; exact (List.getElem?_eq_some_iff.1 hr2i).1
            rw [List.getElem?_append_left hlt, hm3]; exact hr2i
          have hne_13 : b3 ≠ r1.base := by
            obtain ⟨r1', h1m, h1base, h1cap, _, _, _, _, _, h1_64, _⟩ := hb1_2
            rw [← h1base]
            exact base_ne_of_free (by rw [hm3]; exact h1m) (by omega) (by omega) hpf3.2.2.2
          have hne_23 : b3 ≠ b2 := by
            rw [← r2base]
            exact base_ne_of_free (by rw [hm3]; exact hr2i) (by omega) hr2pos hpf3.2.2.2
          have hne_bss3 : b3 ≠ 0x600000 := by
            have hbss_3 : HasBss s3.mem := by rw [hm3]; exact hbss2
            obtain ⟨ib, rb, hrib, hb_eq, hblen, _⟩ := hbss_3
            rw [← hb_eq]
            exact base_ne_of_free hrib (by omega) (by omega) hpf3.2.2.2
          have hb3_lt48 : b3 + bs.length < 2 ^ 48 := by
            have := pages_ge (bs.length + 1); omega
          have hmsb3 : msb b3 = false := msb_lt (by omega)
          have hcc3 : X86Int.cond (aluF .and b3 b3).2 8 = some false := by rw [cond_test_s, hmsb3]
          let s3' : St := { ((s3.set 0 (some b3)).set 1 (some 4210590)).set 11 none with
            k := s3.k.next.2, mem := m4, rip := 4210590 }
          have he2' : exec .syscall 2 s3 = .ok s3' := by rw [he2, hrip3]
          have hpre_cp2 : Reaches s3 (fun s4 =>
              s4.rip = 4210611 ∧ RegDefExcept11 s4 ∧ s4.mem = m4 ∧
              s4.reg 14 = some r1.base ∧ s4.reg 15 = some b2 ∧ s4.reg 5 = some b3 ∧
              s4.reg 13 = some bs.length ∧ s4.reg 8 = some (W - 1) ∧
              s4.reg 6 = some (r1.base + 0) ∧ s4.reg 7 = some (b3 + 0) ∧
              s4.reg 11 = some bs.length) := by
            refine reach_step (by rw [hrip3]; exact g25) ?_
            rw [he2']
            have h0' : s3'.reg 0 = some b3 := by simp only [s3']; rsimpG [hd3.1]
            have h14' : s3'.reg 14 = some r1.base := by simp only [s3']; rsimpG [hd3.1]; exact h14_3
            have h15' : s3'.reg 15 = some b2 := by simp only [s3']; rsimpG [hd3.1]; exact h15_3
            have h13' : s3'.reg 13 = some bs.length := by simp only [s3']; rsimpG [hd3.1]; exact h13_3
            have h8' : s3'.reg 8 = some (W - 1) := by simp only [s3']; rsimpG [hd3.1]; exact h8_3
            have hd3' : RegDefExcept11 s3' :=
              regDef_set_11_none (regDef_set_some (regDef_set_some hd3 0 _) 1 _)
            xs g27 using h0'
            xs g30 using hcc3
            xs g36 using h0'
            xs g39 using h14'
            xs g42 using hd3'.1
            xs g45 using h13'
            refine Reaches.refl ⟨by rfl, ?_, rfl,
              by rsimpG [hd3'.1]; exact h14', by rsimpG [hd3'.1]; exact h15',
              by rsimpG [hd3'.1], by rsimpG [hd3'.1]; exact h13',
              by rsimpG [hd3'.1]; exact h8',
              by simp only [Nat.add_zero]; rsimpG [hd3'.1],
              by simp only [Nat.add_zero]; rsimpG [hd3'.1],
              by rsimpG [hd3'.1]⟩
            exact regDef_set_some (regDef_set_some (regDef_set_some (regDef_set_some hd3' 5 _) 6 _) 7 _) 11 _
          -- Run copy_loop 2 and write all 6 descriptor words + REJECT sentinels + jmp L_tripled
          refine Reaches.toInit a6 as ?_
          refine hpre_cp2.trans fun s4 ⟨hr4, hd4, hm4, h14_4, h15_4, h5_4, h13_4, h8_4, h6_4, h7_4, h11_4⟩ => ?_
          have htake3 : (List.replicate (pages (bs.length + 1)) 0).take 0 = bs.take 0 := by simp
          have hcp2_done := copy_loop_spec hcp2 hby (by omega) hb3_lt48 hne_13 hne_bss3 bs.length 0 s4
            (Nat.zero_add _) hr4 hd4 (by rwa [hm4]) (by rwa [hm4]) (by rwa [hm4])
            ⟨s3.mem.length, ⟨b3, List.replicate (pages (bs.length + 1)) 0, true⟩,
              by rw [hm4]; simp [m4], rfl, rfl, hb2_len, by rw [hmmap_len]; omega, htake3⟩
            h6_4 h7_4 h11_4
          refine hcp2_done.trans fun s5 ⟨hr5, hd5, hwf5, hsh5, hbss5, hb1_5,
              ⟨i3, r3, hr3i, r3base, r3rw, hr3len, hr3pos, hbuf3⟩,
              hoth2, h14_5, h15_5, h5_5, h13_5, h8_5, _⟩ => ?_
          have h14_5' : s5.reg 14 = some r1.base := h14_5.trans h14_4
          have h15_5' : s5.reg 15 = some b2 := h15_5.trans h15_4
          have h5_5' : s5.reg 5 = some b3 := h5_5.trans h5_4
          have h13_5' : s5.reg 13 = some bs.length := h13_5.trans h13_4
          have h8_5' : s5.reg 8 = some (W - 1) := h8_5.trans h8_4
          have hr2i_4' : s4.mem[i2]? = some r2 := by rw [hm4]; exact hr2i_4
          have hr2i_5 : s5.mem[i2]? = some r2 := hoth2 i2 r2 hr2i_4' (by rw [r2base]; exact hne_23.symm)
          have hbuf2_5 : Buf s5.mem b2 bs := by
            have hb4 : Buf s4.mem b2 bs := by rw [hm4]; exact hbuf2_4
            exact buf_of_region_eq (by rw [hm4]; exact hwf4) hwf5 hr2i_4' hr2i_5 r2base
              (by simp only [Region.hi]; omega) hb4
          obtain ⟨r1_5, hr1i_5, hr1base_5, hr1cap_5, _, _, hr1_47, _, _, _, hr1bss_5, hbs_le_5, _⟩ := hb1_5
          have hbuf1_5 : Buf s5.mem r1.base bs := hasBufRW_to_Buf hwf5
            ⟨r1_5, hr1i_5, hr1base_5, hr1cap_5, by assumption, by assumption, hr1_47,
             by assumption, by assumption, by assumption, hr1bss_5, hbs_le_5, by assumption⟩
          obtain ⟨ib, rb, hrib, hb_base, hb_len, hb_rw⟩ := hbss5
          have hc_p1 : CodeAt [(4210635, .movI 7 (0x600000 + 0), 5),
              (4210635 + 5, .st 7 0 14, 3), (4210635 + 8, .st 7 8 13, 4)] := by
            simp only [CodeAt_cons, CodeAt_nil, and_true]; exact ⟨d0, d5, d8⟩
          have hc_p2 : CodeAt [(4210647, .movI 7 (0x600000 + 0x1000), 5),
              (4210647 + 5, .st 7 0 15, 3), (4210647 + 8, .st 7 8 13, 4)] := by
            simp only [CodeAt_cons, CodeAt_nil, and_true]; exact ⟨d12, d17, d20⟩
          have hc_p3 : CodeAt [(4210659, .movI 7 (0x600000 + 0x2000), 5),
              (4210659 + 5, .st 7 0 5, 3), (4210659 + 8, .st 7 8 13, 4)] := by
            simp only [CodeAt_cons, CodeAt_nil, and_true]; exact ⟨d24, d29, d32⟩
          have hp1_reach := exec_wr64_bss_pair hc_p1 (by decide) (by decide) (by decide) hr5 hd5 hwf5
            hrib hb_base hb_len hb_rw h14_5' h13_5'
          refine hp1_reach.trans fun s6 ⟨hr6, hd6, hwf6, ⟨db1, hm6, hdl1, hs1a, hs1b, hrest1⟩, hreg6⟩ => ?_
          have hrib6 : s6.mem[ib]? = some { rb with data := db1 } := by
            rw [hm6, List.getElem?_set_self (List.getElem?_eq_some_iff.1 hrib).1]
          have hp2_reach := exec_wr64_bss_pair hc_p2 (by decide) (by decide) (by decide) hr6 hd6 hwf6
            hrib6 hb_base hdl1 hb_rw (by rw [hreg6 15 (by decide)]; exact h15_5')
            (by rw [hreg6 13 (by decide)]; exact h13_5')
          refine hp2_reach.trans fun s7 ⟨hr7, hd7, hwf7, ⟨db2, hm7, hdl2, hs2a, hs2b, hrest2⟩, hreg7⟩ => ?_
          have hrib7 : s7.mem[ib]? = some { rb with data := db2 } := by
            rw [hm7, hm6, List.set_set, List.getElem?_set_self (List.getElem?_eq_some_iff.1 hrib).1]
          have hp3_reach := exec_wr64_bss_pair hc_p3 (by decide) (by decide) (by decide) hr7 hd7 hwf7
            hrib7 hb_base hdl2 hb_rw
            (by rw [hreg7 5 (by decide), hreg6 5 (by decide)]; exact h5_5')
            (by rw [hreg7 13 (by decide), hreg6 13 (by decide)]; exact h13_5')
          refine hp3_reach.trans fun s8 ⟨hr8, hd8, hwf8, ⟨db3, hm8, hdl3, hs3a, hs3b, hrest3⟩, hreg8⟩ => ?_
          have hm8' : s8.mem = s5.mem.set ib { rb with data := db3 } := by
            rw [hm8, hm7, hm6, List.set_set, List.set_set]
          have hrib8 : s8.mem[ib]? = some { rb with data := db3 } := by
            rw [hm8', List.getElem?_set_self (List.getElem?_eq_some_iff.1 hrib).1]
          have h8_8 : s8.reg 8 = some (W - 1) := by
            rw [hreg8 8 (by decide), hreg7 8 (by decide), hreg6 8 (by decide)]; exact h8_5'
          have hrej_W : REJECT < W := verdictWord_lt_W (some false)
          have hrej_mod : REJECT % 18446744073709551616 = REJECT :=
            Nat.mod_eq_of_lt (by rw [W_eq] at hrej_W; exact hrej_W)
          have hne_ib1 : ib ≠ i1 := by
            intro h; subst h; rw [hr1i_5] at hrib; cases hrib; exact hr1bss_5 (hr1base_5.symm.trans hb_base)
          have hne_ib2 : ib ≠ i2 := by
            intro h; subst h; rw [hr2i_5] at hrib; cases hrib; exact hne_bss2 (by rw [← r2base, hb_base])
          have hne_ib3 : ib ≠ i3 := by
            intro h; subst h; rw [hr3i] at hrib; cases hrib; exact hne_bss3 (by rw [← r3base, hb_base])
          have hbuf1_8 : Buf s8.mem r1.base bs := buf_of_region_eq hwf5 hwf8 hr1i_5
            (by rw [hm8', List.getElem?_set_ne hne_ib1]; exact hr1i_5) hr1base_5
            (by simp only [Region.hi]; omega) hbuf1_5
          have hbuf2_8 : Buf s8.mem b2 bs := buf_of_region_eq hwf5 hwf8 hr2i_5
            (by rw [hm8', List.getElem?_set_ne hne_ib2]; exact hr2i_5) r2base
            (by simp only [Region.hi]; omega) hbuf2_5
          have hbuf3_8 : Buf s8.mem b3 bs := buf_of_region_eq hwf5 hwf8 hr3i
            (by rw [hm8', List.getElem?_set_ne hne_ib3]; exact hr3i) r3base
            (by simp only [Region.hi]; omega) hbuf3
          have hb1_W : r1.base < W := by rw [W_eq]; omega
          have hb2_W : b2 < W := by rw [W_eq]; omega
          have hb3_W : b3 < W := by rw [W_eq]; omega
          have hd_600 : Desc s8.mem 0x600000 r1.base bs.length := by
            refine ⟨?_, ?_⟩
            · rw [ea_small (by omega), show (0x600000 : Nat) + 0 = rb.base + 0 by omega]
              refine rd64_wr64_slot hwf8 hrib8 (off := 0) (by simp [hdl3]) hb1_W fun j hj => ?_
              rw [hrest3 (0 + j) (by omega) (by omega), hrest2 (0 + j) (by omega) (by omega)]
              exact hs1a j hj
            · rw [ea_small (by omega), show (0x600000 : Nat) + 8 = rb.base + 8 by omega]
              refine rd64_wr64_slot hwf8 hrib8 (off := 8) (by simp [hdl3]) hbs_W fun j hj => ?_
              rw [hrest3 (8 + j) (by omega) (by omega), hrest2 (8 + j) (by omega) (by omega)]
              exact hs1b j hj
          have hd_601 : Desc s8.mem 0x601000 b2 bs.length := by
            refine ⟨?_, ?_⟩
            · rw [ea_small (by omega), show (0x601000 : Nat) + 0 = rb.base + 0x1000 by omega]
              refine rd64_wr64_slot hwf8 hrib8 (off := 0x1000) (by simp [hdl3]) hb2_W fun j hj => ?_
              rw [hrest3 (0x1000 + j) (by omega) (by omega)]
              exact hs2a j hj
            · rw [ea_small (by omega), show (0x601000 : Nat) + 8 = rb.base + 0x1008 by omega]
              refine rd64_wr64_slot hwf8 hrib8 (off := 0x1008) (by simp [hdl3]) hbs_W fun j hj => ?_
              rw [hrest3 (0x1008 + j) (by omega) (by omega)]
              exact hs2b j hj
          have hd_602 : Desc s8.mem 0x602000 b3 bs.length := by
            refine ⟨?_, ?_⟩
            · rw [ea_small (by omega), show (0x602000 : Nat) + 0 = rb.base + 0x2000 by omega]
              refine rd64_wr64_slot hwf8 hrib8 (off := 0x2000) (by simp [hdl3]) hb3_W fun j hj => ?_
              exact hs3a j hj
            · rw [ea_small (by omega), show (0x602000 : Nat) + 8 = rb.base + 0x2008 by omega]
              refine rd64_wr64_slot hwf8 hrib8 (off := 0x2008) (by simp [hdl3]) hbs_W fun j hj => ?_
              exact hs3b j hj
          xs d36
          xs d46
          xs d56
          xs d66
          refine Reaches.refl ⟨rfl, regDef_set_some (regDef_set_some (regDef_set_some hd8 0 _) 3 _) 1 _,
            by rsimpG [hd8.1]; exact hwf8,
            by rsimpG [hd8.1, hrej_mod], by rsimpG [hd8.1, hrej_mod],
            by rsimpG [hd8.1, hrej_mod], by rsimpG [hd8.1]; exact h8_8,
            b2, b3, by rsimpG [hd8.1]; exact hbuf1_8, by rsimpG [hd8.1]; exact hd_600, by omega,
            by rsimpG [hd8.1]; exact hbuf2_8, by rsimpG [hd8.1]; exact hd_601, hb2_lt48,
            by rsimpG [hd8.1]; exact hbuf3_8, by rsimpG [hd8.1]; exact hd_602, hb3_lt48⟩

/-! ### Complete init + 3 passes + voter/guard composition -/

theorem init_spec_gen {s : St} {a6 as : Bool}
    (hby : ∀ x ∈ s.k.stdin, x < 256)
    (hrip : s.rip = L_start) (hd : RegDefExcept11 s)
    (hwf : s.mem.wf) (hbss : HasBss s.mem)
    (hor : a6 = false ∨ as = false → GoodInitOrac (shape s.mem) s.k.stdin.length s.k.oracle) :
    ReachesInit s (fun s' =>
      s'.rip = L_tripled ∧
      RegDefExcept11 s' ∧
      s'.mem.wf ∧
      s'.reg 0 = some REJECT ∧ s'.reg 3 = some REJECT ∧ s'.reg 1 = some REJECT ∧
      s'.reg 8 = some (W - 1) ∧
      ∃ b1 b2 b3,
        Buf s'.mem b1 s.k.stdin ∧ Desc s'.mem 0x600000 b1 s.k.stdin.length ∧ b1 + s.k.stdin.length < 2 ^ 48 ∧
        Buf s'.mem b2 s.k.stdin ∧ Desc s'.mem 0x601000 b2 s.k.stdin.length ∧ b2 + s.k.stdin.length < 2 ^ 48 ∧
        Buf s'.mem b3 s.k.stdin ∧ Desc s'.mem 0x602000 b3 s.k.stdin.length ∧ b3 + s.k.stdin.length < 2 ^ 48)
      a6 as := by
  refine (init_mmap1_spec hrip hd hwf hbss hor).trans ?_
  rintro s1 ⟨hr1, hd1, hwf1, hbss1, i1, b1, hb1, _, _, h14, h12, h13, h8, hk1, hor1⟩
  have hby1 : ∀ x ∈ s.k.stdin, x < 256 := hby
  have hrd := rd_loop_gen s.k.stdin hby1 s1.k.stdin.length [] i1 b1 65536 s1
    (by simpa using hk1) rfl hr1 hd1 hwf1 hbss1 hb1 h14 h12 h13 (by simpa using h8) hor1
  refine hrd.trans ?_
  rintro s2 ⟨hr2, hd2, hwf2, hbss2, i1', b1', cap', hb1', h14', h13', h8', hor2⟩
  refine (eof_to_tripled_gen hby hr2 hd2 hwf2 hbss2 hb1' h14' h13' h8' hor2).trans ?_
  rintro s3 ⟨hr3, hd3, hwf3, h0, h3, h1, h8_3, b2, b3, hb_1, hd_1, hl_1, hb_2, hd_2, hl_2, hb_3, hd_3, hl_3⟩
  exact ReachesInit.refl ⟨hr3, hd3, hwf3, h0, h3, h1, h8_3, b1', b2, b3,
    hb_1, hd_1, hl_1, hb_2, hd_2, hl_2, hb_3, hd_3, hl_3⟩

def flatExit (r : Option Bool) : Nat :=
  match r with
  | some true => 0
  | some false => 5
  | none => 6

theorem flatExit_eq_tailExit (r : Option Bool) :
    flatExit r = tailExit (verdictWord r) (verdictWord r) (verdictWord r) := by
  rw [tailExit_uniform]
  cases r with
  | none => decide
  | some b => cases b <;> decide

theorem tripled_to_exit {bs : List Nat} {s : St}
    (hby : ∀ x ∈ bs, x < 256)
    (hrip : s.rip = L_tripled) (hd : RegDefExcept11 s)
    (hb1 : ∃ b1, Buf s.mem b1 bs ∧ Desc s.mem 0x600000 b1 bs.length ∧ b1 + bs.length < 2 ^ 48)
    (hb2 : ∃ b2, Buf s.mem b2 bs ∧ Desc s.mem 0x601000 b2 bs.length ∧ b2 + bs.length < 2 ^ 48)
    (hb3 : ∃ b3, Buf s.mem b3 bs ∧ Desc s.mem 0x602000 b3 bs.length ∧ b3 + bs.length < 2 ^ 48) :
    Ends s (· = .exit (flatExit (checkFlatA bs))) := by
  obtain ⟨b1, hbuf1, hdesc1, hsm1⟩ := hb1
  obtain ⟨b2, hbuf2, hdesc2, hsm2⟩ := hb2
  obtain ⟨b3, hbuf3, hdesc3, hsm3⟩ := hb3
  obtain ⟨v8, hv8⟩ := regDef_get hd 8
  have hp1 := reaches_regDefExcept11 hd (p1_spec hbuf1 hby hsm1 hdesc1 hd.1 hrip hv8)
  refine hp1.ends fun s1 ⟨⟨hf1, hrip1, hr0_1, v8_1, hv8_1⟩, hd1⟩ => ?_
  have hbuf2_1 : Buf s1.mem b2 bs := by rw [hf1.mem]; exact hbuf2
  have hdesc2_1 : Desc s1.mem 0x601000 b2 bs.length := ⟨by rw [hf1.mem]; exact hdesc2.ptr, by rw [hf1.mem]; exact hdesc2.len⟩
  have hbuf3_1 : Buf s1.mem b3 bs := by rw [hf1.mem]; exact hbuf3
  have hdesc3_1 : Desc s1.mem 0x602000 b3 bs.length := ⟨by rw [hf1.mem]; exact hdesc3.ptr, by rw [hf1.mem]; exact hdesc3.len⟩
  have hp2 := reaches_regDefExcept11 hd1 (p2_spec hbuf2_1 hby hsm2 hdesc2_1 hd1.1 hrip1 hv8_1)
  refine hp2.ends fun s2 ⟨⟨hf2, hrip2, hr3_2, v8_2, hv8_2⟩, hd2⟩ => ?_
  have hr0_2 : s2.reg 0 = some (verdictWord (checkFlatA bs)) := by
    rw [hf2.regs 0 (by decide)]; exact hr0_1
  have hbuf3_2 : Buf s2.mem b3 bs := by rw [hf2.mem]; exact hbuf3_1
  have hdesc3_2 : Desc s2.mem 0x602000 b3 bs.length := ⟨by rw [hf2.mem]; exact hdesc3_1.ptr, by rw [hf2.mem]; exact hdesc3_1.len⟩
  have hp3 := reaches_regDefExcept11 hd2 (p3_spec hbuf3_2 hby hsm3 hdesc3_2 hd2.1 hrip2 hv8_2)
  refine hp3.ends fun s3 ⟨⟨hf3, hrip3, hr1_3, _, _⟩, hd3⟩ => ?_
  have hr0_3 : s3.reg 0 = some (verdictWord (checkFlatA bs)) := by
    rw [hf3.regs 0 (by decide)]; exact hr0_2
  have hr3_3 : s3.reg 3 = some (verdictWord (checkFlatA bs)) := by
    rw [hf3.regs 3 (by decide)]; exact hr3_2
  have hvW := verdictWord_lt_W (checkFlatA bs)
  have htail := tail_spec hvW hvW hvW hrip3 hd3 hr0_3 hr3_3 hr1_3
  rw [flatExit_eq_tailExit]
  exact htail

/-! ### Initial process state, canonical oracle, and top-level theorems -/

def initCodeRegion : Region := ⟨0x400000, spr1Bytes, false⟩
def initBssRegion : Region := ⟨0x600000, List.replicate 0x3000 0, true⟩

def initMem : Mem :=
  let (stk, _) := X86Env.initStack 0x7ffffffff000 []
  [initCodeRegion, initBssRegion, stk]

def initState (k : Kern) : St :=
  let (_, sp) := X86Env.initStack 0x7ffffffff000 []
  { regs := (List.replicate 16 (some 0)).set 4 (some sp)
  , rip := L_start
  , fl := Flags.undef
  , mem := initMem
  , k := k }

private theorem spr1Bytes_len_bound : spr1Bytes.length ≤ 0x10000 := by decide +kernel

private theorem initStack_nil_ok :
    (X86Env.initStack 0x7ffffffff000 []).1.base = 0x7fffffffd000 ∧
    (X86Env.initStack 0x7ffffffff000 []).1.data.length = 8192 := by decide +kernel

theorem initMem_wf : initMem.wf := by
  have _ := spr1Bytes_len_bound
  have hstk_base : (let (stk, _) := X86Env.initStack 0x7ffffffff000 []; stk).base = 0x7fffffffd000 :=
    initStack_nil_ok.1
  have hstk_val := initStack_nil_ok.1
  simp only [X86Env.initStack] at hstk_val
  refine ⟨?_, ?_⟩
  · intro r hr
    simp only [initMem, List.mem_cons, List.not_mem_nil, or_false] at hr
    rcases hr with rfl | rfl | rfl
    · decide
    · decide
    · rw [hstk_base]; decide
  · refine List.Pairwise.cons ?_ (List.Pairwise.cons ?_
        (List.Pairwise.cons (fun _ h => by cases h) List.Pairwise.nil))
    · intro r hr
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
      rcases hr with rfl | rfl
      · left; simp only [initCodeRegion, initBssRegion, Region.hi]; omega
      · left; simp only [initCodeRegion, Region.hi]; rw [hstk_val]; omega
    · intro r hr
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
      rw [hr]; left; simp only [initBssRegion, Region.hi, List.length_replicate]; rw [hstk_val]; omega

theorem initMem_hasBss : HasBss initMem :=
  ⟨1, initBssRegion, rfl, rfl, List.length_replicate, rfl⟩

theorem initState_regDef (k : Kern) : RegDefExcept11 (initState k) := by
  have hrep : ∀ i < 16, ((List.replicate 16 (some (0 : Nat)))[i]?.getD none).isSome = true := by
    decide
  refine ⟨by simp [initState], fun j hj _ => ?_⟩
  simp only [initState, St.reg]
  by_cases hj4 : j = 4
  · subst hj4; simp
  · rw [List.getElem?_set_ne (Ne.symm hj4)]
    exact hrep j hj

theorem initMem_shape :
    shape initMem =
      [(0x400000, 0x400000 + spr1Bytes.length), (0x600000, 0x603000), (0x7fffffffd000, 0x7ffffffff000)] := by
  have hstk := initStack_nil_ok
  simp only [initMem, shape, initCodeRegion, initBssRegion, Region.hi, List.length_replicate,
    hstk.1, hstk.2, List.map_cons, List.map_nil]

def growCountAux : Nat → Nat → Nat → Nat
  | 0, _, _ => 0
  | fuel + 1, cap, total => if total < cap then 0 else 1 + growCountAux fuel (2 * cap) total

def growCount (total : Nat) : Nat := growCountAux 45 65536 total

def canonOracle (bs : List Nat) : List Choice :=
  List.replicate (1 + growCount bs.length) (.at 0x100000000000) ++
    [.at 0x300000000000, .at 0x500000000000]

private theorem freeS_canon_a1 {sh : List (Nat × Nat)}
    (hsh : ∀ p ∈ sh, p.2 ≤ 0x100000000000 ∨ 0x600000000000 ≤ p.1)
    {n : Nat} (hn : n ≤ 2 ^ 45) :
    freeS sh 0x100000000000 n = true := by
  rw [freeS, List.all_eq_true]
  intro p hp
  have h1 := hsh p hp
  have hcap : (0x100000000000 : Nat) + 2 ^ 45 < 0x600000000000 := by decide
  rcases h1 with h | h <;> simp [Bool.or_eq_true] <;> omega

private theorem freeS_canon_b2 {sh : List (Nat × Nat)}
    (hsh : ∀ p ∈ sh, p.2 ≤ 0x300000000000 ∨ 0x600000000000 ≤ p.1)
    {n : Nat} (hn : n ≤ 2 ^ 44) :
    freeS sh 0x300000000000 n = true := by
  rw [freeS, List.all_eq_true]
  intro p hp
  have h1 := hsh p hp
  have hcap : (0x300000000000 : Nat) + 2 ^ 44 ≤ 0x600000000000 := by decide
  rcases h1 with h | h <;> simp [Bool.or_eq_true] <;> omega

private theorem freeS_canon_b3 {sh : List (Nat × Nat)}
    (hsh : ∀ p ∈ sh, p.2 ≤ 0x500000000000 ∨ 0x600000000000 ≤ p.1)
    {n : Nat} (hn : n ≤ 2 ^ 44) :
    freeS sh 0x500000000000 n = true := by
  rw [freeS, List.all_eq_true]
  intro p hp
  have h1 := hsh p hp
  have hcap : (0x500000000000 : Nat) + 2 ^ 44 ≤ 0x600000000000 := by decide
  rcases h1 with h | h <;> simp [Bool.or_eq_true] <;> omega

private theorem countStep_no_count (t : Nat) (want avail : Nat) :
    countStep (List.replicate t (.at 0x100000000000) ++ [.at 0x300000000000, .at 0x500000000000]) want avail =
      (min want avail, List.replicate t (.at 0x100000000000) ++ [.at 0x300000000000, .at 0x500000000000]) := by
  cases t <;> simp [countStep, List.replicate_succ]

private theorem pow2_grow_bound : ∀ (f cap total : Nat),
    total < 2 ^ 44 → 65536 ≤ cap → total < cap * 2 ^ f →
    growCountAux f cap total = 0 → total < cap
  | 0, cap, total, _, _, hlt, _ => by simpa using hlt
  | f + 1, cap, total, ht44, h64, hlt, hg => by
    simp only [growCountAux] at hg
    split at hg
    · assumption
    · omega

private theorem pages_len_le_pow44 {len : Nat} (hlen : len < 2 ^ 44) : pages (len + 1) ≤ 2 ^ 44 := by
  simp only [pages, page]
  have hpow : (2 : Nat) ^ 44 = 4294967296 * 4096 := by decide
  omega

private theorem canon_tail_good {sh : List (Nat × Nat)} {iBuf len cap : Nat}
    (h44 : len < 2 ^ 44) (h45 : cap ≤ 2 ^ 45)
    (hsh : ∀ p ∈ sh.eraseIdx iBuf, p.2 ≤ 0x100000000000 ∨ 0x600000000000 ≤ p.1)
    (hi : sh[iBuf]? = some (0x100000000000, 0x100000000000 + cap)) :
    GoodTailOrac sh len [Choice.at 0x300000000000, Choice.at 0x500000000000] := by
  have hpg := pages_len_le_pow44 (len := len) h44
  have hsh_b2 : ∀ p ∈ sh, p.2 ≤ 0x300000000000 ∨ 0x600000000000 ≤ p.1 := by
    intro p hp
    rcases List.mem_iff_getElem?.1 hp with ⟨j, hj⟩
    by_cases hji : j = iBuf
    · subst hji; rw [hi] at hj; cases hj; left; omega
    · have hp_er : p ∈ sh.eraseIdx iBuf := by
        rcases Nat.lt_trichotomy j iBuf with hjlt | hjeq | hjgt
        · exact List.mem_of_getElem? (by rw [List.getElem?_eraseIdx_of_lt hjlt]; exact hj)
        · exact absurd hjeq hji
        · have hj_er : (sh.eraseIdx iBuf)[j - 1]? = some p := by
            rw [List.getElem?_eraseIdx_of_ge (by omega), show j - 1 + 1 = j by omega]
            exact hj
          exact List.mem_of_getElem? hj_er
      rcases hsh p hp_er with h | h
      · left; omega
      · right; exact h
  have hfree2 : freeS sh 0x300000000000 (pages (len + 1)) = true := freeS_canon_b2 hsh_b2 hpg
  have hb2 : placeS sh ([Choice.at 0x300000000000, Choice.at 0x500000000000].head?) (pages (len + 1)) =
      some (some 0x300000000000) := by
    simp [placeS, page, hfree2]; omega
  have hsh_b3 : ∀ p ∈ sh ++ [(0x300000000000, 0x300000000000 + pages (len + 1))],
      p.2 ≤ 0x500000000000 ∨ 0x600000000000 ≤ p.1 := by
    intro p hp
    rw [List.mem_append] at hp
    rcases hp with hp | hp
    · rcases hsh_b2 p hp with h | h
      · left; omega
      · right; exact h
    · simp at hp; subst hp; left; omega
  have hfree3 : freeS (sh ++ [(0x300000000000, 0x300000000000 + pages (len + 1))])
      0x500000000000 (pages (len + 1)) = true := freeS_canon_b3 hsh_b3 hpg
  have hb3 : placeS (sh ++ [(0x300000000000, 0x300000000000 + pages (len + 1))])
      (([Choice.at 0x300000000000, Choice.at 0x500000000000].drop 1).head?) (pages (len + 1)) =
      some (some 0x500000000000) := by
    simp [placeS, page, hfree3]; omega
  exact GoodTailOrac.mk hb2 hb3

private theorem canon_rd_orac_aux :
    ∀ (f rem : Nat) (sh : List (Nat × Nat)) (iBuf len cap : Nat),
      len + rem < 2 ^ 44 →
      65536 ≤ cap → cap ≤ 2 ^ 45 →
      len ≤ cap →
      len + rem < cap * 2 ^ f →
      (∀ p ∈ sh.eraseIdx iBuf, p.2 ≤ 0x100000000000 ∨ 0x600000000000 ≤ p.1) →
      sh[iBuf]? = some (0x100000000000, 0x100000000000 + cap) →
      GoodRdOrac sh iBuf len cap rem
        (List.replicate (growCountAux f cap (len + rem)) (.at 0x100000000000) ++
          [.at 0x300000000000, .at 0x500000000000])
  | f, rem => by
    induction f generalizing rem with
    | zero =>
      intro sh iBuf len cap h44 h64 h45 hle hltf hsh hi
      induction rem using Nat.strongRecOn generalizing len with
      | ind rem ih_rem =>
        have hltc : len < cap := by simp only [Nat.pow_zero, Nat.mul_one] at hltf; omega
        have hg0 : growCountAux 0 cap (len + rem) = 0 := rfl
        rw [hg0]
        by_cases hrem0' : rem = 0
        · subst hrem0'
          rw [List.replicate_zero, List.nil_append]
          refine GoodRdOrac.done hltc ?_
          rw [show (countStep [.at 0x300000000000, .at 0x500000000000] (cap - len) 0).2 =
              [.at 0x300000000000, .at 0x500000000000] from rfl]
          exact canon_tail_good (by omega) h45 hsh hi
        · have hpos : 0 < rem := by omega
          refine GoodRdOrac.step hltc hpos ?_
          dsimp only
          rw [countStep_no_count 0 (cap - len) rem]
          have hmin : min (cap - len) rem = rem := by simp only [Nat.pow_zero, Nat.mul_one] at hltf; omega
          rw [hmin]
          have hlt_r : rem - rem < rem := by omega
          have hltf' : len + rem < cap := by simpa using hltf
          have hltf0 : len + rem + (rem - rem) < cap * 2 ^ 0 := by simpa using hltf'
          simpa [hg0] using ih_rem (rem - rem) hlt_r (len + rem) (by omega) (by omega) hltf0
    | succ f ih_f =>
      intro sh iBuf len cap h44 h64 h45 hle hltf hsh hi
      induction rem using Nat.strongRecOn generalizing sh iBuf len with
      | ind rem ih_rem =>
        by_cases hfull : len = cap
        · have h44' : cap + rem < 2 ^ 44 := by omega
          rw [hfull] at hltf
          have hltf' : cap + rem < (2 * cap) * 2 ^ f := by
            rw [Nat.pow_succ] at hltf
            simpa [Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm] using hltf
          rw [hfull]
          have hnot : ¬ (cap + rem < cap) := by omega
          have hg : growCountAux (f + 1) cap (cap + rem) = 1 + growCountAux f (2 * cap) (cap + rem) := by
            simp [growCountAux, hnot]
          rw [hg, show 1 + growCountAux f (2 * cap) (cap + rem) = growCountAux f (2 * cap) (cap + rem) + 1 by omega,
            List.replicate_succ, List.cons_append]
          have h2cap_le : 2 * cap ≤ 2 ^ 45 := by omega
          have hfree1 : freeS (sh.eraseIdx iBuf) 0x100000000000 (2 * cap) = true :=
            freeS_canon_a1 hsh h2cap_le
          have hpl : ∀ t : List Choice, placeS (sh.eraseIdx iBuf) (Choice.at 0x100000000000 :: t).head? (2 * cap) =
              some (some 0x100000000000) := by
            intro t; simp [placeS, page, hfree1]; omega
          refine GoodRdOrac.grow (hpl _) ?_
          dsimp only [List.drop_succ_cons, List.drop_zero]
          have hsh' : ∀ p ∈ (sh.eraseIdx iBuf ++ [(0x100000000000, 0x100000000000 + 2 * cap)]).eraseIdx
              (sh.eraseIdx iBuf).length, p.2 ≤ 0x100000000000 ∨ 0x600000000000 ≤ p.1 := by
            rw [List.eraseIdx_append_of_length_le (Nat.le_refl _), Nat.sub_self]
            simpa using hsh
          have hi' : (sh.eraseIdx iBuf ++ [(0x100000000000, 0x100000000000 + 2 * cap)])[(sh.eraseIdx iBuf).length]? =
              some (0x100000000000, 0x100000000000 + 2 * cap) := by simp
          exact ih_f rem (sh.eraseIdx iBuf ++ [(0x100000000000, 0x100000000000 + 2 * cap)])
            (sh.eraseIdx iBuf).length cap (2 * cap) h44' (by omega) h2cap_le (by omega)
            hltf' hsh' hi'
        · have hltc : len < cap := Nat.lt_of_le_of_ne hle hfull
          by_cases hrem0' : rem = 0
          · subst hrem0'
            have hg0 : growCountAux (f + 1) cap (len + 0) = 0 := by simp [growCountAux, hltc]
            rw [hg0, List.replicate_zero, List.nil_append]
            refine GoodRdOrac.done hltc ?_
            rw [show (countStep [.at 0x300000000000, .at 0x500000000000] (cap - len) 0).2 =
                [.at 0x300000000000, .at 0x500000000000] from rfl]
            exact canon_tail_good (by omega) h45 hsh hi
          · have hpos : 0 < rem := by omega
            refine GoodRdOrac.step hltc hpos ?_
            dsimp only
            rw [countStep_no_count]
            have hn_pos : 0 < min (cap - len) rem := by omega
            have hlt_r : rem - min (cap - len) rem < rem := by omega
            have hsum_eq : (len + min (cap - len) rem) + (rem - min (cap - len) rem) = len + rem := by omega
            have h44_n : (len + min (cap - len) rem) + (rem - min (cap - len) rem) < 2 ^ 44 := by
              rw [hsum_eq]; exact h44
            have hle_n : len + min (cap - len) rem ≤ cap := by
              have := Nat.min_le_left (cap - len) rem; omega
            have hltf_n : (len + min (cap - len) rem) + (rem - min (cap - len) rem) < cap * 2 ^ (f + 1) := by
              rw [hsum_eq]; exact hltf
            have ih := ih_rem (rem - min (cap - len) rem) hlt_r sh iBuf (len + min (cap - len) rem)
              h44_n hle_n hltf_n hsh hi
            rwa [hsum_eq] at ih

theorem canonOracle_good {bs : List Nat} (h44 : bs.length < 2 ^ 44) :
    GoodInitOrac (shape initMem) bs.length (canonOracle bs) := by
  have hsh0 : ∀ p ∈ shape initMem, p.2 ≤ 0x100000000000 ∨ 0x600000000000 ≤ p.1 := by
    rw [initMem_shape]
    have hlen := spr1Bytes_len_bound
    intro p hp
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl | rfl
    · left; omega
    · left; omega
    · right; omega
  have hfree0 : freeS (shape initMem) 0x100000000000 65536 = true :=
    freeS_canon_a1 hsh0 (by decide)
  have hp0 : placeS (shape initMem) (canonOracle bs).head? 65536 = some (some 0x100000000000) := by
    simp [canonOracle, show 1 + growCount bs.length = growCount bs.length + 1 by omega,
      List.replicate_succ, placeS, page, hfree0]
  refine GoodInitOrac.mk hp0 ?_
  have hdrop : (canonOracle bs).drop 1 =
      List.replicate (growCount bs.length) (.at 0x100000000000) ++ [.at 0x300000000000, .at 0x500000000000] := by
    simp [canonOracle, show 1 + growCount bs.length = growCount bs.length + 1 by omega, List.replicate_succ]
  rw [hdrop]
  have hsh1 : ∀ p ∈ (shape initMem ++ [(0x100000000000, 0x100000000000 + 65536)]).eraseIdx (shape initMem).length,
      p.2 ≤ 0x100000000000 ∨ 0x600000000000 ≤ p.1 := by
    rw [List.eraseIdx_append_of_length_le (Nat.le_refl _), Nat.sub_self]
    simpa using hsh0
  have hi1 : (shape initMem ++ [(0x100000000000, 0x100000000000 + 65536)])[(shape initMem).length]? =
      some (0x100000000000, 0x100000000000 + 65536) := by simp
  have hlt45 : 0 + bs.length < 65536 * 2 ^ 45 := by
    have : (2 : Nat) ^ 44 < 65536 * 2 ^ 45 := by decide
    omega
  simpa [growCount] using canon_rd_orac_aux 45 bs.length
    (shape initMem ++ [(0x100000000000, 0x100000000000 + 65536)])
    (shape initMem).length 0 65536 (by omega) (by decide) (by decide) (by omega) hlt45 hsh1 hi1

/-! ### Top-level flat binary refinement theorems -/

theorem spr1_run_flat_uncond {k : Kern} (hby : ∀ x ∈ k.stdin, x < 256) :
    ∀ {n c : Nat}, run im n (.ok (initState k)) = .exit c →
      (c = 0 → checkFlatA k.stdin = some true) ∧
      (c = 5 → checkFlatA k.stdin = some false) ∧
      c ≠ 3 ∧ (c = 0 ∨ c = 5 ∨ c = 6) := by
  have hinit := init_spec_gen (a6 := true) (as := true) hby rfl (initState_regDef k) initMem_wf
    initMem_hasBss (fun h => by simp at h)
  have hends := hinit (fun o =>
      o = .stuck ∨
      ∃ c', o = .exit c' ∧
        (c' = 0 → checkFlatA k.stdin = some true) ∧
        (c' = 5 → checkFlatA k.stdin = some false) ∧
        c' ≠ 3 ∧ (c' = 0 ∨ c' = 5 ∨ c' = 6))
    (fun s' ⟨hrip, hd, _, _, _, _, _, b1, b2, b3, hb1, hd1, hl1, hb2, hd2, hl2, hb3, hd3, hl3⟩ => by
      obtain ⟨m, hm⟩ := tripled_to_exit hby hrip hd ⟨b1, hb1, hd1, hl1⟩ ⟨b2, hb2, hd2, hl2⟩ ⟨b3, hb3, hd3, hl3⟩
      refine ⟨m, Or.inr ⟨_, hm, ?_⟩⟩
      cases hcf : checkFlatA k.stdin with
      | none => simp [flatExit]
      | some b => cases b <;> simp [flatExit])
    (fun _ => Or.inr ⟨6, rfl, fun h => False.elim (by omega), fun h => False.elim (by omega),
      by decide, Or.inr (Or.inr rfl)⟩)
    (fun _ => Or.inl rfl)
  obtain ⟨m, hm⟩ := hends
  intro n c hnc
  rcases hm with hst | ⟨c', hm', hprop⟩
  · by_cases hle : m ≤ n
    · have ha := run_add m (n - m) (.ok (initState k))
      rw [show m + (n - m) = n by omega, hnc, hst, run_stuck] at ha
      cases ha
    · have ha := run_add n (m - n) (.ok (initState k))
      rw [show n + (m - n) = m by omega, hnc, run_exit, hst] at ha
      cases ha
  · have hsame : c' = c := by
      by_cases hle : m ≤ n
      · have ha := run_add m (n - m) (.ok (initState k))
        rw [show m + (n - m) = n by omega, hnc, hm', run_exit] at ha
        cases ha; rfl
      · have ha := run_add n (m - n) (.ok (initState k))
        rw [show n + (m - n) = m by omega, hnc, hm', run_exit] at ha
        cases ha; rfl
    rwa [← hsame]

theorem spr1_run_flat_good {k : Kern} (hby : ∀ x ∈ k.stdin, x < 256)
    (hor : GoodInitOrac (shape initMem) k.stdin.length k.oracle) :
    Ends (initState k) (· = .exit (flatExit (checkFlatA k.stdin))) := by
  have hinit := init_spec_gen (a6 := false) (as := false) hby rfl (initState_regDef k) initMem_wf
    initMem_hasBss (fun _ => hor)
  exact hinit (· = .exit (flatExit (checkFlatA k.stdin)))
    (fun s' ⟨hrip, hd, _, _, _, _, _, b1, b2, b3, hb1, hd1, hl1, hb2, hd2, hl2, hb3, hd3, hl3⟩ =>
      tripled_to_exit hby hrip hd ⟨b1, hb1, hd1, hl1⟩ ⟨b2, hb2, hd2, hl2⟩ ⟨b3, hb3, hd3, hl3⟩)
    (fun h => by simp at h)
    (fun h => by simp at h)

theorem spr1_run_flat_canon {bs : List Nat} (hby : ∀ x ∈ bs, x < 256) (h44 : bs.length < 2 ^ 44) :
    Ends (initState ⟨[], [], bs, [], canonOracle bs⟩) (· = .exit (flatExit (checkFlatA bs))) :=
  spr1_run_flat_good hby (canonOracle_good h44)

end SporosCheckBin
