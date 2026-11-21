/-
SporosCheckArm64BinTail: bridges the 192-byte tail `[L_vote, L_core_end)` of
`spr1_check_arm64` to `SporosCheckArm64Flat.tailImAt` / `tailAltsAt`, defines
`TripledState`, `initState`, and `goodOracle`, and proves `vote_guard_*`,
`nomem_spec`, `three_pass_ends`, `initMem_wf`, memory region read/write
lemmas, and `copy_loop_spec`.
-/
import Bootstrap.SporosCheckArm64BinWords

namespace SporosCheckArm64Bin

open AArch64Env AArch64Phys SporosCheckArm64Flat

/-! ### Full-image fetch alternatives and `[L_vote, L_core_end)` bridge -/

/-- Hamming-1 fetch alternatives for the full `spr1_check_arm64` ELF image. -/
def alts : Alts := imageAlts spr1Arm64Bytes BASE

theorem L_vote_base_ok : base_ok L_vote := by
  unfold base_ok L_vote AArch64Env.W; decide

private theorem tailBytes_eq_tailCode : tailBytes = tailCode := by
  decide +kernel

theorem tail_bytes_eq : (spr1Arm64Bytes.drop (L_vote - BASE)).take 192 = tailCode := by
  rw [drop_vote_eq, List.take_append_of_le_length (by rw [tailBytes_len]; exact Nat.le_refl _),
    show (192 : Nat) = tailBytes.length from tailBytes_len.symm, List.take_length, tailBytes_eq_tailCode]

theorem spr1_tail_byte {j : Nat} (hj : j < 192) :
    spr1Arm64Bytes[L_vote - BASE + j]? = tailCode[j]? := by
  have h : ((spr1Arm64Bytes.drop (L_vote - BASE)).take 192)[j]? = tailCode[j]? :=
    congrArg (fun l => l[j]?) tail_bytes_eq
  rw [List.getElem?_take, if_pos hj, List.getElem?_drop] at h
  exact h

theorem im_eq_tailImAt {a : Nat} (hlo : L_vote ≤ a) (hhi : a + 4 ≤ L_vote + 192) :
    im a = tailImAt L_vote a := by
  unfold im tailImAt imageOf
  rw [spr1Arm64Bytes_len, tailCode_length]
  have h1 : BASE ≤ a := by unfold BASE L_vote at *; omega
  have h2 : a + 4 ≤ BASE + 47636 := by unfold BASE L_vote at *; omega
  have hcond : (BASE ≤ a && a + 4 ≤ BASE + 47636 && BASE % 4 == 0 && a % 4 == 0) =
      (L_vote ≤ a && a + 4 ≤ L_vote + 192 && L_vote % 4 == 0 && a % 4 == 0) := by
    simp only [h1, h2, hlo, hhi, decide_true, Bool.true_and,
      show (BASE % 4 == 0) = true by decide, show (L_vote % 4 == 0) = true by decide]
  rw [hcond]
  split
  · apply AArch64Phys.decode_reads_only_4_bytes
    intro i hi
    have hidx : a - BASE + i = L_vote - BASE + (a - L_vote + i) := by
      unfold BASE L_vote at *; omega
    rw [hidx]
    exact spr1_tail_byte (by omega)
  · rfl

private theorem altsAt_eq_of_agree {b1 b2 : Nat → Option Nat} {pc : Nat}
    (hag : ∀ i < 4, b1 i = b2 i) (hpc : pc % 4 = 0) : altsAt b1 = altsAt b2 := by
  unfold altsAt
  apply List.map_congr_left
  intro k _
  exact (AArch64Phys.fetch_fault_fixed_boundary hag hpc [k]).1

theorem alts_eq_tailAltsAt {a : Nat} (hlo : L_vote ≤ a) (hhi : a + 4 ≤ L_vote + 192) :
    alts a = tailAltsAt L_vote a := by
  unfold alts tailAltsAt imageAlts
  rw [spr1Arm64Bytes_len, tailCode_length]
  have h1 : BASE ≤ a := by unfold BASE L_vote at *; omega
  have h2 : a + 4 ≤ BASE + 47636 := by unfold BASE L_vote at *; omega
  have hcond : (BASE ≤ a && a + 4 ≤ BASE + 47636 && BASE % 4 == 0 && a % 4 == 0) =
      (L_vote ≤ a && a + 4 ≤ L_vote + 192 && L_vote % 4 == 0 && a % 4 == 0) := by
    simp only [h1, h2, hlo, hhi, decide_true, Bool.true_and,
      show (BASE % 4 == 0) = true by decide, show (L_vote % 4 == 0) = true by decide]
  rw [hcond]
  split
  · next hc =>
    have ha4 : a % 4 = 0 := by
      simp only [Bool.and_eq_true, beq_iff_eq] at hc; exact hc.2
    have hag : ∀ i < 4, spr1Arm64Bytes[a - BASE + i]? = tailCode[a - L_vote + i]? := by
      intro i hi
      have hidx : a - BASE + i = L_vote - BASE + (a - L_vote + i) := by
        unfold BASE L_vote at *; omega
      rw [hidx]; exact spr1_tail_byte (by omega)
    exact altsAt_eq_of_agree hag ha4
  · rfl

theorem tailImAt_some_bounds {a : Nat} {p : Ins × Nat} (h : tailImAt L_vote a = some p) :
    L_vote ≤ a ∧ a + 4 ≤ L_vote + 192 ∧ a % 4 = 0 := by
  unfold tailImAt imageOf at h
  rw [tailCode_length] at h
  split at h
  · next hc =>
    simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hc
    exact ⟨hc.1.1.1, hc.1.1.2, hc.2⟩
  · cases h

private theorem step_eq_of_img_eq {im1 im2 : Img} {s : St} (h : im1 s.pc = im2 s.pc) :
    step im1 s = step im2 s := by
  unfold step; rw [h]

private theorem run_stuck_any (img : Img) : ∀ n, run img n .stuck = .stuck
  | 0 => rfl
  | _ + 1 => rfl

private theorem run_img_eq_of_agree {im1 im2 : Img} {P : Nat → Prop}
    (hagree : ∀ a, P a → im1 a = im2 a)
    (hclosed : ∀ a p, im2 a = some p → P a) :
    ∀ {n c : Nat} {o : Out}, run im2 n o = .exit c → run im1 n o = .exit c
  | 0, _, _, h => h
  | _ + 1, _, .exit _, h => h
  | _ + 1, _, .stuck, h => by rw [run_stuck_any] at h; cases h
  | _ + 1, _, .ok s, h => by
    simp only [run] at h ⊢
    cases ht : im2 s.pc with
    | none =>
      have hst : step im2 s = .stuck := by unfold step; rw [ht]
      rw [hst, run_stuck_any] at h; cases h
    | some p =>
      have hstep : step im1 s = step im2 s :=
        step_eq_of_img_eq (hagree s.pc (hclosed s.pc p ht))
      rw [hstep]
      exact run_img_eq_of_agree hagree hclosed h

theorem run_im_eq_exit_of_tail {n c : Nat} {o : Out}
    (h : run (tailImAt L_vote) n o = .exit c) : run im n o = .exit c :=
  run_img_eq_of_agree (P := fun a => L_vote ≤ a ∧ a + 4 ≤ L_vote + 192)
    (fun _ ha => im_eq_tailImAt ha.1 ha.2)
    (fun _ _ ht => let ⟨h1, h2, _⟩ := tailImAt_some_bounds ht; ⟨h1, h2⟩) h

/-! ### `vote_guard_spec` and `nomem_spec` -/

theorem voterRegs_eq_regs {s : St} {a b c : Nat} (hl : s.regs.length = 31)
    (h19 : s.reg 19 = some a) (h20 : s.reg 20 = some b) (h21 : s.reg 21 = some c) :
    TMRVoterArm64.voterRegs s a b c = s.regs := by
  have hvr : TMRVoterArm64.voterRegs s a b c = (List.range 31).map (fun i => s.reg i) := by
    unfold TMRVoterArm64.voterRegs
    rw [← h19, ← h20, ← h21]
    rfl
  rw [hvr, ← hl]
  apply List.ext_getElem?
  intro j
  by_cases hj : j < s.regs.length
  · simp [St.reg, List.getElem?_map, List.getElem?_range hj, List.getElem?_eq_getElem hj]
  · rw [List.getElem?_eq_none (by simp; omega), List.getElem?_eq_none (by omega)]

theorem tail_step0_eq_init {s : St} {a b c : Nat} (hl : s.regs.length = 31) (hpc : s.pc = L_vote)
    (h19 : s.reg 19 = some a) (h20 : s.reg 20 = some b) (h21 : s.reg 21 = some c) :
    step (tailImAt L_vote) s = step (tailImAt L_vote) (tailInit L_vote s a b c) := by
  have him : tailImAt L_vote L_vote = some (.addSubReg true true 64 31 19 20 .lsl 0, 4) := by
    decide +kernel
  have hregs := voterRegs_eq_regs hl h19 h20 h21
  have hst : tailInit L_vote s a b c = { s with fl := Flags.undef } := by
    cases s; dsimp only at hpc hregs; subst hpc; simp [tailInit, hregs]
  rw [hst]
  rw [step, step, hpc, him]
  cases s; dsimp only at hpc; subst hpc; rfl

theorem tail_run_eq_init {n : Nat} {s : St} {a b c : Nat} (hn : 0 < n)
    (hl : s.regs.length = 31) (hpc : s.pc = L_vote)
    (h19 : s.reg 19 = some a) (h20 : s.reg 20 = some b) (h21 : s.reg 21 = some c) :
    run (tailImAt L_vote) n (.ok s) = run (tailImAt L_vote) n (.ok (tailInit L_vote s a b c)) := by
  cases n with
  | zero => omega
  | succ k => simp only [run, tail_step0_eq_init hl hpc h19 h20 h21]

theorem vote_guard_accept {s : St} {a b c : Nat} (hl : s.regs.length = 31) (hpc : s.pc = L_vote)
    (h19 : s.reg 19 = some a) (h20 : s.reg 20 = some b) (h21 : s.reg 21 = some c)
    (ha : a < AArch64Env.W) (hb : b < AArch64Env.W) (hc : c < AArch64Env.W)
    (hv : TMR.vote a b c = some ACCEPT) :
    run im 29 (.ok s) = .exit 0 := by
  apply run_im_eq_exit_of_tail
  rw [tail_run_eq_init (by decide) hl hpc h19 h20 h21]
  exact tail_arch_accept L_vote_base_ok s ha hb hc hv

theorem vote_guard_overflow {s : St} {a b c : Nat} (hl : s.regs.length = 31) (hpc : s.pc = L_vote)
    (h19 : s.reg 19 = some a) (h20 : s.reg 20 = some b) (h21 : s.reg 21 = some c)
    (ha : a < AArch64Env.W) (hb : b < AArch64Env.W) (hc : c < AArch64Env.W)
    (hv : TMR.vote a b c = some OVERFLOW) :
    run im 29 (.ok s) = .exit 6 := by
  apply run_im_eq_exit_of_tail
  rw [tail_run_eq_init (by decide) hl hpc h19 h20 h21]
  exact tail_arch_overflow L_vote_base_ok s ha hb hc hv

theorem vote_guard_reject {s : St} {w a b c : Nat} (hl : s.regs.length = 31) (hpc : s.pc = L_vote)
    (h19 : s.reg 19 = some a) (h20 : s.reg 20 = some b) (h21 : s.reg 21 = some c)
    (ha : a < AArch64Env.W) (hb : b < AArch64Env.W) (hc : c < AArch64Env.W)
    (hv : TMR.vote a b c = some w) (hneA : w ≠ ACCEPT) (hneO : w ≠ OVERFLOW) :
    run im 29 (.ok s) = .exit 5 := by
  apply run_im_eq_exit_of_tail
  rw [tail_run_eq_init (by decide) hl hpc h19 h20 h21]
  exact tail_arch_reject L_vote_base_ok s ha hb hc hv hneA hneO

theorem vote_guard_refuse {s : St} {a b c : Nat} (hl : s.regs.length = 31) (hpc : s.pc = L_vote)
    (h19 : s.reg 19 = some a) (h20 : s.reg 20 = some b) (h21 : s.reg 21 = some c)
    (ha : a < AArch64Env.W) (hb : b < AArch64Env.W) (hc : c < AArch64Env.W)
    (hv : TMR.vote a b c = none) :
    run im 8 (.ok s) = .exit 3 := by
  apply run_im_eq_exit_of_tail
  rw [tail_run_eq_init (by decide) hl hpc h19 h20 h21]
  exact tail_arch_refuse L_vote_base_ok s ha hb hc hv

private theorem nomem_ins :
    tailImAt L_vote L_nomem = some (.movWide 2 64 0 6 0, 4) ∧
    tailImAt L_vote (L_nomem + 4) = some (.movWide 2 64 2 6 0, 4) ∧
    tailImAt L_vote (L_nomem + 8) = some (.logicReg .orr false 64 0 0 2 .lsl 0, 4) ∧
    tailImAt L_vote (L_nomem + 12) = some (.movWide 2 64 8 93 0, 4) ∧
    tailImAt L_vote (L_nomem + 16) = some (.svc 0, 4) := by
  decide +kernel

theorem nomem_spec {s : St} (hl : s.regs.length = 31) (hpc : s.pc = L_nomem) :
    run im 5 (.ok s) = .exit 6 := by
  apply run_im_eq_exit_of_tail
  obtain ⟨f0, f1, f2, f3, f4⟩ := nomem_ins
  have h1 : step (tailImAt L_vote) s = .ok ((s.withPc (L_nomem + 4)).setReg 0 (some 6)) := by
    rw [step, hpc, f0]; simp [exec, hpc, St.withPc, St.wrZR, pow2, L_nomem, AArch64Env.W]
  have h2 : step (tailImAt L_vote) ((s.withPc (L_nomem + 4)).setReg 0 (some 6)) =
      .ok (((s.withPc (L_nomem + 8)).setReg 0 (some 6)).setReg 2 (some 6)) := by
    rw [step]; simp only [St.setReg, St.withPc]; rw [f1]
    simp [exec, St.wrZR, St.setReg, pow2, L_nomem, AArch64Env.W]
  have h3 : step (tailImAt L_vote) (((s.withPc (L_nomem + 8)).setReg 0 (some 6)).setReg 2 (some 6)) =
      .ok (((((s.withPc (L_nomem + 12)).setReg 0 (some 6)).setReg 2 (some 6))).setReg 0 (some 6)) := by
    rw [step]; simp only [St.setReg, St.withPc]; rw [f2]
    simp [exec, lift, St.rdZR, St.wrZR, St.setReg, St.reg, hl,
      shiftVal, pow2, L_nomem, AArch64Env.W]
  have h4 : step (tailImAt L_vote) (((((s.withPc (L_nomem + 12)).setReg 0 (some 6)).setReg 2 (some 6))).setReg 0 (some 6)) =
      .ok (((((((s.withPc (L_nomem + 16)).setReg 0 (some 6)).setReg 2 (some 6))).setReg 0 (some 6))).setReg 8 (some 93)) := by
    rw [step]; simp only [St.setReg, St.withPc]; rw [f3]
    simp [exec, St.wrZR, St.setReg, pow2, L_nomem, AArch64Env.W]
  have h5 : step (tailImAt L_vote) (((((((s.withPc (L_nomem + 16)).setReg 0 (some 6)).setReg 2 (some 6))).setReg 0 (some 6))).setReg 8 (some 93)) =
      .exit 6 := by
    rw [step]; simp only [St.setReg, St.withPc]; rw [f4]
    simp [exec, execSvc, applySys, syscall, St.reg, hl]
  simp only [run, h1, h2, h3, h4, h5]

/-! ### `TripledState` and end-to-end 3-pass + voter execution -/

/-- Descriptor base addresses extracted from BSS descriptors `0x600000`, `0x601000`, `0x602000`. -/
def tripledB1 (s : St) : Nat := (rdMem s.mem 0x600000 .x8).getD 0
def tripledB2 (s : St) : Nat := (rdMem s.mem 0x601000 .x8).getD 0
def tripledB3 (s : St) : Nat := (rdMem s.mem 0x602000 .x8).getD 0

/-- Post-triplication machine state at `L_tripled = 0x401100`: three independent
pairwise-disjoint buffers `b1`, `b2`, `b3` holding `bs`, their descriptors at
`0x600000`, `0x601000`, `0x602000` (disjoint from the buffers), verdict
registers `x19 = x20 = x21 = REJECT`, `VoterScratchClean s`, page-zero unmapped
(`noLowMem`), byte-bounded descriptors (`descBytes`), and BSS/code disjointness. -/
structure TripledState (bs : List Nat) (s : St) : Prop where
  pc        : s.pc = L_tripled
  regsLen   : s.regs.length = 31
  memWf     : s.mem.wf
  noLowMem  : ∀ r ∈ s.mem, 4096 ≤ r.base
  descBytes : ∀ d ∈ ([0x600000, 0x601000, 0x602000] : List Nat),
                ∀ j < 16, ∀ x, s.mem.rd (d + j) = some x → x < 256
  oracleWf  : OracleWf s.k.oracle
  bytes     : ∀ b ∈ bs, b < 256
  x19       : s.reg 19 = some REJECT
  x20       : s.reg 20 = some REJECT
  x21       : s.reg 21 = some REJECT
  scratch   : VoterScratchClean s
  hD1_ptr   : rdMem s.mem 0x600000 .x8 = some (tripledB1 s)
  hD1_len   : rdMem s.mem (0x600000 + 8) .x8 = some bs.length
  hD2_ptr   : rdMem s.mem 0x601000 .x8 = some (tripledB2 s)
  hD2_len   : rdMem s.mem (0x601000 + 8) .x8 = some bs.length
  hD3_ptr   : rdMem s.mem 0x602000 .x8 = some (tripledB3 s)
  hD3_len   : rdMem s.mem (0x602000 + 8) .x8 = some bs.length
  hBuf1     : Buf s.mem (tripledB1 s) bs
  hBuf2     : Buf s.mem (tripledB2 s) bs
  hBuf3     : Buf s.mem (tripledB3 s) bs
  hSm1      : tripledB1 s + bs.length < HMAX
  hSm2      : tripledB2 s + bs.length < HMAX
  hSm3      : tripledB3 s + bs.length < HMAX
  disj12    : tripledB1 s + bs.length ≤ tripledB2 s ∨ tripledB2 s + bs.length ≤ tripledB1 s
  disj13    : tripledB1 s + bs.length ≤ tripledB3 s ∨ tripledB3 s + bs.length ≤ tripledB1 s
  disj23    : tripledB2 s + bs.length ≤ tripledB3 s ∨ tripledB3 s + bs.length ≤ tripledB2 s
  disjD1    : ∀ d ∈ [0x600000, 0x601000, 0x602000], tripledB1 s + bs.length ≤ d ∨ d + 16 ≤ tripledB1 s
  disjD2    : ∀ d ∈ [0x600000, 0x601000, 0x602000], tripledB2 s + bs.length ≤ d ∨ d + 16 ≤ tripledB2 s
  disjD3    : ∀ d ∈ [0x600000, 0x601000, 0x602000], tripledB3 s + bs.length ≤ d ∨ d + 16 ≤ tripledB3 s
  disjBss   : ∀ b ∈ ([tripledB1 s, tripledB2 s, tripledB3 s] : List Nat), b + bs.length ≤ 0x600000 ∨ 0x603000 ≤ b
  disjCode  : ∀ b ∈ ([tripledB1 s, tripledB2 s, tripledB3 s] : List Nat), b + bs.length ≤ BASE ∨ BASE + 47636 ≤ b
  hiB1      : 4096 ≤ tripledB1 s
  hiB2      : 4096 ≤ tripledB2 s
  hiB3      : 4096 ≤ tripledB3 s

def TripledState.b1 {bs : List Nat} {s : St} (_ : TripledState bs s) : Nat := tripledB1 s
def TripledState.b2 {bs : List Nat} {s : St} (_ : TripledState bs s) : Nat := tripledB2 s
def TripledState.b3 {bs : List Nat} {s : St} (_ : TripledState bs s) : Nat := tripledB3 s

theorem voterScratchClean_of_frame {S : List Nat} {s s' : St}
    (hf : Frame S s s') (hdisj : ∀ r ∈ VoterScratchRegs, r ∉ S)
    (hscr : VoterScratchClean s) : VoterScratchClean s' := by
  intro r hr v hv
  rw [hf.regs r (hdisj r hr)] at hv
  exact hscr r hr v hv

theorem three_pass_reach_vote {bs : List Nat} {s : St} (ht : TripledState bs s) :
    Reaches s fun s' =>
      s'.pc = L_vote ∧ s'.regs.length = 31 ∧ s'.mem = s.mem ∧
      s'.reg 19 = some (checkFlatWord bs) ∧ s'.reg 20 = some (checkFlatWord bs) ∧
      s'.reg 21 = some (checkFlatWord bs) ∧ VoterScratchClean s' := by
  refine (p1_spec ht.hD1_ptr ht.hD1_len ht.hBuf1 ht.bytes ht.hSm1 ht.regsLen ht.pc rfl).trans
    fun s1 ⟨hf1, hpc1, h19_1, hl1, hm1⟩ => ?_
  have hscr1 := voterScratchClean_of_frame hf1 (by decide) ht.scratch
  refine (p2_spec ht.hD2_ptr ht.hD2_len ht.hBuf2 ht.bytes ht.hSm2 hl1 hpc1 hm1).trans
    fun s2 ⟨hf2, hpc2, h20_2, hl2, hm2⟩ => ?_
  have h19_2 : s2.reg 19 = some (checkFlatWord bs) := by rw [hf2.regs 19 (by decide)]; exact h19_1
  have hscr2 := voterScratchClean_of_frame hf2 (by decide) hscr1
  refine (p3_spec ht.hD3_ptr ht.hD3_len ht.hBuf3 ht.bytes ht.hSm3 hl2 hpc2 hm2).mono
    fun s3 ⟨hf3, hpc3, h21_3, hl3, hm3⟩ => ?_
  have h19_3 : s3.reg 19 = some (checkFlatWord bs) := by rw [hf3.regs 19 (by decide)]; exact h19_2
  have h20_3 : s3.reg 20 = some (checkFlatWord bs) := by rw [hf3.regs 20 (by decide)]; exact h20_2
  have hscr3 := voterScratchClean_of_frame hf3 (by decide) hscr2
  exact ⟨hpc3, hl3, hm3, h19_3, h20_3, h21_3, hscr3⟩

/-- Expected exit status of `spr1_check_arm64` on post-triplication bytes `bs`. -/
def expectedExit (bs : List Nat) : Nat :=
  match checkFlat bs with
  | some true => 0
  | some false => 5
  | none => 6

theorem three_pass_ends {bs : List Nat} {s : St} (ht : TripledState bs s) :
    Ends s (· = .exit (expectedExit bs)) := by
  refine (three_pass_reach_vote ht).ends fun s' ⟨hpc, hl, _, h19, h20, h21, _⟩ => ?_
  have hlt := checkFlatWord_lt_W bs
  have hv : TMR.vote (checkFlatWord bs) (checkFlatWord bs) (checkFlatWord bs) = some (checkFlatWord bs) := by
    unfold TMR.vote; simp
  unfold expectedExit checkFlatWord at *
  cases hcf : checkFlat bs with
  | none =>
    rw [hcf] at h19 h20 h21 hlt hv
    exact ⟨29, vote_guard_overflow hl hpc h19 h20 h21 hlt hlt hlt hv⟩
  | some b =>
    cases b
    · rw [hcf] at h19 h20 h21 hlt hv
      exact ⟨29, vote_guard_reject hl hpc h19 h20 h21 hlt hlt hlt hv (by decide) (by decide)⟩
    · rw [hcf] at h19 h20 h21 hlt hv
      exact ⟨29, vote_guard_accept hl hpc h19 h20 h21 hlt hlt hlt hv⟩

/-! ### ELF initial state, memory region helpers, and byte-copy loop -/

def codeRegion : Region := ⟨BASE, spr1Arm64Bytes, false⟩
def bssRegion  : Region := ⟨0x600000, List.replicate 0x3000 0, true⟩
def initStk    : Region × Nat := AArch64Env.initStack 0x7000000000 []
def initMem    : Mem := [codeRegion, bssRegion, initStk.1]

def initState (bs : List Nat) (o : List Choice) : St :=
  ⟨List.replicate 31 (some 0), some initStk.2, L_start, Flags.undef, initMem, ⟨[], [], bs, [], o⟩⟩

theorem initStk_props : initStk.1.base % AArch64Env.page = 0 ∧
    0x603000 ≤ initStk.1.base ∧ initStk.1.hi = 0x7000000000 := by
  have h1 : initStk.1.base = 0x7000000000 - (AArch64Env.stackRoom + AArch64Env.pages 41) := rfl
  have h2 : initStk.1.data.length = AArch64Env.stackRoom + AArch64Env.pages 41 := by
    simp only [initStk, AArch64Env.initStack, AArch64Env.layStrings, List.map_nil, List.foldl_nil,
      List.length_nil, List.flatten_nil, List.append_nil, List.length_append, List.length_replicate,
      AArch64Env.le64]
    decide
  simp only [Region.hi, h1, h2, AArch64Env.stackRoom, AArch64Env.pages, AArch64Env.page]
  decide

theorem initMem_wf : initMem.wf := by
  have ⟨hsp, hlo, _⟩ := initStk_props
  refine ⟨?_, ?_⟩
  · intro r hr
    simp only [initMem, List.mem_cons, List.not_mem_nil, or_false] at hr
    rcases hr with rfl | rfl | rfl
    · simp only [codeRegion]; decide
    · simp only [bssRegion]; decide
    · exact hsp
  · simp only [initMem, List.pairwise_cons, List.mem_cons, List.not_mem_nil, or_false,
      List.Pairwise.nil, and_true, forall_eq_or_imp, forall_eq]
    have hc : codeRegion.hi = BASE + 47636 := by
      simp only [codeRegion, Region.hi, spr1Arm64Bytes_len]
    have hb : bssRegion.hi = 0x603000 := by
      simp only [bssRegion, Region.hi, List.length_replicate]
    refine ⟨⟨Or.inl (by rw [hc]; decide), Or.inl (by rw [hc]; unfold BASE; omega)⟩,
      Or.inl (by rw [hb]; exact hlo), fun _ h => h.elim⟩

theorem Mem.find_at_idx : ∀ (m : Mem) (idx i a : Nat) {r : Region},
    m.wf → m[idx]? = some r → r.base ≤ a → a < r.hi → Mem.find m a i = some (i + idx, r)
  | [], _, _, _, _, _, h, _, _ => by simp at h
  | r0 :: rest, 0, i, a, r, _, hget, hlo, hhi => by
    simp only [List.getElem?_cons_zero, Option.some.injEq] at hget
    subst hget; simp [Mem.find, hlo, hhi]
  | r0 :: rest, k + 1, i, a, r, hwf, hget, hlo, hhi => by
    have hdisj : r0.hi ≤ r.base ∨ r.hi ≤ r0.base :=
      (List.pairwise_cons.1 hwf.2).1 r (List.mem_of_getElem? (by simpa using hget))
    have hnot : ¬ (r0.base ≤ a ∧ a < r0.hi) := by omega
    simp only [Mem.find, hnot, ↓reduceIte]
    have hwf' : Mem.wf rest := ⟨fun x hx => hwf.1 x (List.mem_cons_of_mem _ hx), (List.pairwise_cons.1 hwf.2).2⟩
    have ih := Mem.find_at_idx rest k (i + 1) a hwf' (by simpa using hget) hlo hhi
    rwa [show i + 1 + k = i + (k + 1) by omega] at ih

theorem Mem.rd_at_idx {m : Mem} {idx a : Nat} {r : Region}
    (hwf : m.wf) (hget : m[idx]? = some r) (hlo : r.base ≤ a) (hhi : a < r.hi) :
    m.rd a = r.data[a - r.base]? := by
  simp only [Mem.rd, bind, Option.bind, Mem.find_at_idx m idx 0 a hwf hget hlo hhi]

theorem Mem.wr_at_idx {m : Mem} {idx a v : Nat} {r : Region}
    (hwf : m.wf) (hget : m[idx]? = some r) (hrw : r.rw = true) (hlo : r.base ≤ a) (hhi : a < r.hi) :
    m.wr a v = some (m.set idx { r with data := r.data.set (a - r.base) (v % 256) }) := by
  simp only [Mem.wr, bind, Option.bind, Mem.find_at_idx m idx 0 a hwf hget hlo hhi, hrw, ↓reduceIte, pure, Nat.zero_add]

theorem Mem.wrN_at_idx : ∀ (bs : List Nat) (m : Mem) (idx off : Nat) (r : Region),
    m.wf → m[idx]? = some r → r.rw = true → off + bs.length ≤ r.data.length →
    ∃ data', m.wrN (r.base + off) bs = some (m.set idx { r with data := data' }) ∧
      data'.length = r.data.length ∧
      (∀ j < off, data'[j]? = r.data[j]?) ∧
      (∀ j < bs.length, data'[off + j]? = some (bs[j]! % 256)) ∧
      (∀ j, off + bs.length ≤ j → data'[j]? = r.data[j]?)
  | [], m, idx, off, r, _, hget, _, _ => by
    refine ⟨r.data, ?_, rfl, fun _ _ => rfl, (fun _ h => by cases h), fun _ _ => rfl⟩
    have hs : m.set idx r = m := by
      apply List.ext_getElem?; intro j
      by_cases hj : idx = j
      · subst hj; have hlt := (List.getElem?_eq_some_iff.1 hget).1
        rw [List.getElem?_set_self hlt, hget]
      · rw [List.getElem?_set_ne hj]
    simpa [Mem.wrN] using congrArg some hs.symm
  | b :: rest, m, idx, off, r, hwf, hget, hrw, hle => by
    have hlo : r.base ≤ r.base + off := Nat.le_add_right _ _
    have hhi : r.base + off < r.hi := by simp only [Region.hi, List.length_cons] at *; omega
    have hwr1 := Mem.wr_at_idx (v := b) hwf hget hrw hlo hhi
    rw [Nat.add_sub_cancel_left] at hwr1
    let r1 : Region := { r with data := r.data.set off (b % 256) }
    let m1 : Mem := m.set idx r1
    have hwf1 : m1.wf := wf_of_shape (wr_shape hwr1) hwf
    have hidx_lt : idx < m.length := (List.getElem?_eq_some_iff.1 hget).1
    have hget1 : m1[idx]? = some r1 := List.getElem?_set_self hidx_lt
    have hle1 : off + 1 + rest.length ≤ r1.data.length := by
      simp only [r1, List.length_set, List.length_cons] at hle ⊢; omega
    obtain ⟨data', hwrN, hlen', hlt', hmid', hge'⟩ :=
      Mem.wrN_at_idx rest m1 idx (off + 1) r1 hwf1 hget1 hrw hle1
    have hset_set : m1.set idx { r1 with data := data' } = m.set idx { r with data := data' } := by
      simp [m1, r1, List.set_set]
    rw [hset_set] at hwrN
    refine ⟨data', ?_, by rw [hlen']; simp [r1], ?_, ?_, ?_⟩
    · simp only [Mem.wrN, bind, Option.bind, hwr1, show r.base + off + 1 = r.base + (off + 1) by omega]
      exact hwrN
    · intro j hj
      rw [hlt' j (by omega), show r1.data[j]? = r.data[j]? by simp [r1, List.getElem?_set_ne (by omega : off ≠ j)]]
    · intro j hj
      cases j with
      | zero =>
        rw [Nat.add_zero, hlt' off (by omega)]
        dsimp only [r1]
        rw [List.getElem?_set_self (by simp only [List.length_cons] at hle; omega)]
        rfl
      | succ j' =>
        rw [show off + (j' + 1) = off + 1 + j' by omega]
        exact hmid' j' (by simp only [List.length_cons] at hj; omega)
    · intro j hj
      simp only [List.length_cons] at hj
      rw [hge' j (by omega)]
      simp [r1, List.getElem?_set_ne (by omega : off ≠ j)]

theorem buf_of_region {m : Mem} {idx : Nat} {r : Region} {bs : List Nat}
    (hwf : m.wf) (hget : m[idx]? = some r) (hlen : bs.length ≤ r.data.length)
    (heq : ∀ j < bs.length, r.data[j]? = some bs[j]!) :
    Buf m r.base bs := by
  intro j hj
  rw [Mem.rd_at_idx hwf hget (Nat.le_add_right _ _) (by simp [Region.hi]; omega),
    Nat.add_sub_cancel_left, heq j hj]
  simp [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem hj]

theorem leVal_bytesN : ∀ (k v : Nat), leVal (bytesN v k) = v % 256 ^ k
  | 0, _ => by simp [bytesN, leVal, Nat.mod_one]
  | k + 1, v => by
    simp only [bytesN, leVal, leVal_bytesN k (v / 256)]
    rw [Nat.pow_succ']
    have h1 : v = 256 * (v / 256) + v % 256 := (Nat.div_add_mod v 256).symm
    have h2 : v / 256 = 256 ^ k * (v / 256 / 256 ^ k) + v / 256 % 256 ^ k :=
      (Nat.div_add_mod (v / 256) (256 ^ k)).symm
    have hlt : v % 256 + 256 * (v / 256 % 256 ^ k) < 256 * 256 ^ k := by
      have hm1 : v % 256 < 256 := Nat.mod_lt v (by decide)
      have hm2 : v / 256 % 256 ^ k < 256 ^ k := Nat.mod_lt _ (Nat.pow_pos (by decide))
      omega
    have hmul : v = (v % 256 + 256 * (v / 256 % 256 ^ k)) + 256 * 256 ^ k * (v / 256 / 256 ^ k) := by
      conv => lhs; rw [h1, h2]
      have : 256 * (256 ^ k * (v / 256 / 256 ^ k) + v / 256 % 256 ^ k) =
          256 * 256 ^ k * (v / 256 / 256 ^ k) + 256 * (v / 256 % 256 ^ k) := by
        rw [Nat.mul_add, Nat.mul_assoc]
      omega
    conv => rhs; rw [hmul]
    rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hlt]

theorem leVal_bytesN8 (v : Nat) : leVal (bytesN v 8) = v % 2 ^ 64 :=
  leVal_bytesN 8 v

theorem rdMem_x8_of_region {m : Mem} {idx off v : Nat} {r : Region}
    (hwf : m.wf) (hget : m[idx]? = some r) (hle : off + 8 ≤ r.data.length) (hv : v < 2 ^ 64)
    (hbytes : ∀ j < 8, r.data[off + j]? = some ((bytesN v 8)[j]! % 256)) :
    rdMem m (r.base + off) .x8 = some v := by
  have hrd : ∀ j < 8, m.rd (r.base + off + j) = some (bytesN v 8)[j]! := by
    intro j hj
    rw [show r.base + off + j = r.base + (off + j) by omega,
      Mem.rd_at_idx hwf hget (Nat.le_add_right _ _) (by simp [Region.hi]; omega),
      Nat.add_sub_cancel_left, hbytes j hj]
    have hlt256 : (bytesN v 8)[j]! < 256 := by
      have hb_lt : ∀ k v j, j < k → (bytesN v k)[j]! < 256 := by
        intro k
        induction k with
        | zero => intro _ _ hj'; cases hj'
        | succ k ih =>
          intro v' j' hj'
          cases j' with
          | zero => simp [bytesN]; exact Nat.mod_lt v' (by decide)
          | succ j'' => simpa [bytesN] using ih (v' / 256) j'' (Nat.lt_of_succ_lt_succ hj')
      exact hb_lt 8 v j hj
    rw [Nat.mod_eq_of_lt hlt256]
  have h0 : m.rd (r.base + off) = some (bytesN v 8)[0]! := by simpa using hrd 0 (by decide)
  have h1 : m.rd (r.base + off + 1) = some (bytesN v 8)[1]! := hrd 1 (by decide)
  have h2 : m.rd (r.base + off + 1 + 1) = some (bytesN v 8)[2]! := by
    rw [show r.base + off + 1 + 1 = r.base + off + 2 by omega]; exact hrd 2 (by decide)
  have h3 : m.rd (r.base + off + 1 + 1 + 1) = some (bytesN v 8)[3]! := by
    rw [show r.base + off + 1 + 1 + 1 = r.base + off + 3 by omega]; exact hrd 3 (by decide)
  have h4 : m.rd (r.base + off + 1 + 1 + 1 + 1) = some (bytesN v 8)[4]! := by
    rw [show r.base + off + 1 + 1 + 1 + 1 = r.base + off + 4 by omega]; exact hrd 4 (by decide)
  have h5 : m.rd (r.base + off + 1 + 1 + 1 + 1 + 1) = some (bytesN v 8)[5]! := by
    rw [show r.base + off + 1 + 1 + 1 + 1 + 1 = r.base + off + 5 by omega]; exact hrd 5 (by decide)
  have h6 : m.rd (r.base + off + 1 + 1 + 1 + 1 + 1 + 1) = some (bytesN v 8)[6]! := by
    rw [show r.base + off + 1 + 1 + 1 + 1 + 1 + 1 = r.base + off + 6 by omega]; exact hrd 6 (by decide)
  have h7 : m.rd (r.base + off + 1 + 1 + 1 + 1 + 1 + 1 + 1) = some (bytesN v 8)[7]! := by
    rw [show r.base + off + 1 + 1 + 1 + 1 + 1 + 1 + 1 = r.base + off + 7 by omega]; exact hrd 7 (by decide)
  have hrdN : m.rdN (r.base + off) 8 = some (bytesN v 8) := by
    simp only [Mem.rdN, bind, Option.bind, pure, h0, h1, h2, h3, h4, h5, h6, h7]
    rfl
  simp [rdMem, MemSize.bytes, hrdN, leVal_bytesN8, Nat.mod_eq_of_lt hv]

/-- Outcome of executing `_start`: stuck, `-ENOMEM` (`exit 6`), or valid `TripledState`. -/
inductive InitOutcome (bs : List Nat) (s : St) : Prop
  | stuck   : Ends s (· = .stuck) → InitOutcome bs s
  | nomem   : Ends s (· = .exit 6) → InitOutcome bs s
  | tripled : Reaches s (TripledState bs) → InitOutcome bs s

theorem InitOutcome.of_reaches {bs : List Nat} {s s' : St}
    (hr : Reaches s (· = s')) (ho : InitOutcome bs s') : InitOutcome bs s := by
  rcases ho with h | h | h
  · exact .stuck (hr.ends fun _ heq => heq ▸ h)
  · exact .nomem (hr.ends fun _ heq => heq ▸ h)
  · exact .tripled (hr.trans fun _ heq => heq ▸ h)

theorem init_codeAt : CodeAt initTable :=
  fun e he => table_ok e (List.mem_append.2 (Or.inr he))

theorem cond_cmp_zero_lt63 {v : Nat} (hv : v < 2 ^ 63) :
    AArch64Phys.cond (addSubOp true 64 (v % 2 ^ 64) 0).2 4 = some false ∧
    AArch64Phys.cond (addSubOp true 64 (v % 2 ^ 64) 0).2 0 = some (v == 0) := by
  have hlt64 : v < 2 ^ 64 := by omega
  have hmsb : msb 64 v = false := by
    simp [msb, pow2]; omega
  simp [AArch64Phys.cond, addSubOp, pow2_64, Nat.mod_eq_of_lt hlt64, hmsb]

theorem cond_cmp_zero_enomem :
    AArch64Phys.cond (addSubOp true 64 ((AArch64Env.neg ENOMEM) % AArch64Env.W % 2 ^ 64) 0).2 4 =
      some true := by
  decide

theorem copy_loop_spec {L src dst idxS idxD : Nat} {bs : List Nat}
    (f0 : im L = some (.cbz false 64 11 28, 4))
    (f1 : im (L + 4) = some (.ldr .b1 10 6 0, 4))
    (f2 : im (L + 8) = some (.str .b1 10 7 0, 4))
    (f3 : im (L + 12) = some (.addSubImm false false 64 6 6 1, 4))
    (f4 : im (L + 16) = some (.addSubImm false false 64 7 7 1, 4))
    (f5 : im (L + 20) = some (.addSubImm true false 64 11 11 1, 4))
    (f6 : im (L + 24) = some (.b false 18446744073709551592, 4))
    (hL : L + 28 < 2 ^ 64) (hne : idxS ≠ idxD)
    (hby : ∀ b ∈ bs, b < 256) (hsLen : bs.length < 2 ^ 48)
    (hsrc_lt : src + bs.length < 2 ^ 48) (hdst_lt : dst + bs.length < 2 ^ 48) :
    ∀ (rem off : Nat) (s : St) (rS rD : Region),
      off + rem = bs.length →
      s.regs.length = 31 → s.pc = L → s.mem.wf →
      s.mem[idxS]? = some rS → s.mem[idxD]? = some rD →
      rS.base = src → rD.base = dst → rD.rw = true →
      bs.length ≤ rS.data.length → bs.length ≤ rD.data.length →
      (∀ j < bs.length, rS.data[j]? = some bs[j]!) →
      (∀ j < off, rD.data[j]? = some bs[j]!) →
      s.reg 6 = some (src + off) → s.reg 7 = some (dst + off) → s.reg 11 = some rem →
      Reaches s fun s' =>
        s'.pc = L + 28 ∧ s'.regs.length = 31 ∧ s'.sp = s.sp ∧ s'.k = s.k ∧
        (∀ j, j ∉ ([6, 7, 10, 11] : List Nat) → s'.reg j = s.reg j) ∧
        s'.mem.wf ∧
        ∃ dataD', s'.mem = s.mem.set idxD { rD with data := dataD' } ∧
          dataD'.length = rD.data.length ∧ ∀ j < bs.length, dataD'[j]? = some bs[j]!
  | 0, off, s, rS, rD, hsum, hl, hpc, hwf, _, hgetD, _, _, _, _, _, _, hdone, _, _, h11 => by
    have hoff : off = bs.length := by omega
    subst hoff
    have hr0 : (L + 28) % AArch64Env.W = L + 28 := modW_small hL
    xs f0 using h11, hr0
    have hset : s.mem = s.mem.set idxD { rD with data := rD.data } := by
      apply List.ext_getElem?; intro j
      by_cases hj : idxD = j
      · subst hj; rw [List.getElem?_set_self (List.getElem?_eq_some_iff.1 hgetD).1, hgetD]
      · rw [List.getElem?_set_ne hj]
    exact Reaches.refl ⟨rfl, hl, rfl, rfl, fun _ _ => rfl, hwf, rD.data, hset, rfl, hdone⟩
  | k + 1, off, s, rS, rD, hsum, hl, hpc, hwf, hgetS, hgetD, hbS, hbD, hrwD, hleS, hleD, hdataS, hdone, h6, h7, h11 => by
    have hk1 : (k + 1) % 2 ^ 64 ≠ 0 := by omega
    have hoff_lt : off < bs.length := by omega
    have hb_lt : bs[off]! < 256 := by
      have : bs[off]! = bs[off] := by simp [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem hoff_lt]
      rw [this]; exact hby _ (List.getElem_mem hoff_lt)
    have hrd : rdMem s.mem (((src + off) % 2 ^ 64 + 0) % AArch64Env.W) .b1 = some bs[off]! := by
      rw [ea_mod_small (by omega) (by omega), Nat.add_zero]
      have h1 := Mem.rd_at_idx hwf hgetS (by omega : rS.base ≤ src + off) (by simp [Region.hi]; omega)
      rw [hbS, Nat.add_sub_cancel_left, hdataS off hoff_lt] at h1
      simp [rdMem, MemSize.bytes, Mem.rdN, h1, leVal]
    let rD1 : Region := { rD with data := rD.data.set off bs[off]! }
    let m1 : Mem := s.mem.set idxD rD1
    have hwr1 : s.mem.wr (dst + off) bs[off]! = some m1 := by
      have h1 := Mem.wr_at_idx (v := bs[off]!) hwf hgetD hrwD
        (by omega : rD.base ≤ dst + off) (by simp [Region.hi]; omega)
      have hsub : dst + off - rD.base = off := by omega
      rw [hsub, Nat.mod_eq_of_lt hb_lt] at h1
      exact h1
    have hwr : wrMem s.mem (((dst + off) % 2 ^ 64 + 0) % AArch64Env.W) (bs[off]! % 2 ^ 32) .b1 = some m1 := by
      rw [ea_mod_small (by omega) (by omega), Nat.add_zero, Nat.mod_eq_of_lt (by omega : bs[off]! < 2 ^ 32)]
      simp only [wrMem, MemSize.bytes, bytesN, Mem.wrN, bind, Option.bind, Nat.mod_eq_of_lt hb_lt, hwr1]
    have hwf1 : m1.wf := wf_of_shape (wr_shape hwr1) hwf
    have h6' : ((src + off) % 2 ^ 64 + 1 % 2 ^ 64) % 2 ^ 64 = src + (off + 1) := by omega
    have h7' : ((dst + off) % 2 ^ 64 + 1 % 2 ^ 64) % 2 ^ 64 = dst + (off + 1) := by omega
    have h11' : ((k + 1) % 2 ^ 64 + 2 ^ 64 - 1 % 2 ^ 64) % 2 ^ 64 = k := by omega
    have hLadd : ∀ a d, a + d ≤ 28 → (L + a + d) % AArch64Env.W = L + (a + d) := fun a d hd => by
      rw [W_eq, Nat.add_assoc]; exact Nat.mod_eq_of_lt (by omega)
    have hLback : (L + 24 + 18446744073709551592) % AArch64Env.W = L := by
      rw [W_eq, Nat.add_assoc, show 24 + 18446744073709551592 = 2 ^ 64 by decide, Nat.add_mod_right]
      exact Nat.mod_eq_of_lt (by omega)
    xs f0 using h11, hk1, (show (L + 4) % AArch64Env.W = L + 4 by rw [W_eq]; exact Nat.mod_eq_of_lt (by omega))
    xs f1 using hl, h6, hrd, (by omega : bs[off]! < 2 ^ 32), hLadd 4 4 (by decide)
    xs f2 using hl, h7, hwr, hLadd 8 4 (by decide)
    xs f3 using hl, h6, h6', hLadd 12 4 (by decide)
    xs f4 using hl, h7, h7', hLadd 16 4 (by decide)
    xs f5 using hl, h11, h11', hLadd 20 4 (by decide)
    xs f6 using hLback
    have hgetS1 : m1[idxS]? = some rS := by rw [List.getElem?_set_ne hne.symm]; exact hgetS
    have hgetD1 : m1[idxD]? = some rD1 := List.getElem?_set_self (List.getElem?_eq_some_iff.1 hgetD).1
    have hdone1 : ∀ j < off + 1, rD1.data[j]? = some bs[j]! := by
      intro j hj
      by_cases heq : off = j
      · subst heq; dsimp only [rD1]; rw [List.getElem?_set_self (by omega)]
      · rw [show rD1.data[j]? = rD.data[j]? by simp [rD1, List.getElem?_set_ne heq]]
        exact hdone j (by omega)
    refine (copy_loop_spec f0 f1 f2 f3 f4 f5 f6 hL hne hby hsLen hsrc_lt hdst_lt
      k (off + 1) _ rS rD1 (by omega) (by simp [hl]) rfl hwf1 hgetS1 hgetD1 hbS hbD hrwD hleS
      (by simp [rD1, hleD]) hdataS hdone1 (by simp [hl]) (by simp [hl]) (by simp [hl])).mono
      fun s' ⟨hpc', hl', hsp', hk', hreg', hwf', dataD', hm', hlen', hd'⟩ => ?_
    refine ⟨hpc', hl', hsp', hk', ?_, hwf', dataD', ?_, by simpa [rD1] using hlen', hd'⟩
    · intro j hj
      rw [hreg' j hj]
      have h6ne : 6 ≠ j := fun h => hj (by simp [← h])
      have h7ne : 7 ≠ j := fun h => hj (by simp [← h])
      have h10ne : 10 ≠ j := fun h => hj (by simp [← h])
      have h11ne : 11 ≠ j := fun h => hj (by simp [← h])
      simp [h6ne, h7ne, h10ne, h11ne]
    · simpa [m1, rD1, List.set_set] using hm'

/-- Number of `mremap` doublings needed from `cap` to strictly exceed `len` (with fuel `f`). -/
def mremapNeedAux : Nat → Nat → Nat → Nat
  | 0, _, _ => 0
  | f + 1, len, cap => if len < cap then 0 else 1 + mremapNeedAux f len (cap * 2)

/-- Non-`-ENOMEM`, full-read oracle placing buffers 1, 2, 3 in three disjoint `2^46`-byte high-VA arenas. -/
def goodOracle (bs : List Nat) : List Choice :=
  List.replicate (1 + mremapNeedAux 30 bs.length 0x10000) (.at 0x400000000000) ++
  [.at 0x800000000000, .at 0xc00000000000]

theorem goodOracle_wf (bs : List Nat) : OracleWf (goodOracle bs) := by
  intro c hc
  simp only [goodOracle, List.mem_append, List.mem_replicate, List.mem_cons, List.not_mem_nil, or_false] at hc
  rcases hc with ⟨_, rfl⟩ | rfl | rfl <;>
    simp [Choice.wf, AArch64Env.page, AArch64Env.userMax]

end SporosCheckArm64Bin
