/-
SporosCheckArm64BinInitEof: verifies the post-EOF triplication phase of `_start`
(`L_eof = 0x40b914` through `b L_tripled` at `0x40ba10`): allocating buffers
`b2` and `b3`, copying `bs` from `b1` into both, storing the three page-separated
descriptors at `0x600000`, `0x601000`, `0x602000`, initializing `x19 = x20 = x21
= REJECT`, and jumping to `L_tripled` in a valid `TripledState bs`.
-/
import Bootstrap.SporosCheckArm64BinTail

namespace SporosCheckArm64Bin

open AArch64Env AArch64Phys SporosCheckArm64Flat

/-- Update only kernel and memory on a state, preserving `regs`, `sp`, `pc`, `fl`. -/
def _root_.AArch64Phys.St.withKM (s : St) (k : Kern) (m : Mem) : St := { s with k := k, mem := m }

@[simp] theorem withKM_regs (s : St) (k : Kern) (m : Mem) : (s.withKM k m).regs = s.regs := rfl
@[simp] theorem withKM_sp (s : St) (k : Kern) (m : Mem) : (s.withKM k m).sp = s.sp := rfl
@[simp] theorem withKM_pc (s : St) (k : Kern) (m : Mem) : (s.withKM k m).pc = s.pc := rfl
@[simp] theorem withKM_fl (s : St) (k : Kern) (m : Mem) : (s.withKM k m).fl = s.fl := rfl
@[simp] theorem withKM_mem (s : St) (k : Kern) (m : Mem) : (s.withKM k m).mem = m := rfl
@[simp] theorem withKM_k (s : St) (k : Kern) (m : Mem) : (s.withKM k m).k = k := rfl
@[simp] theorem withKM_reg (s : St) (k : Kern) (m : Mem) (i : Nat) : (s.withKM k m).reg i = s.reg i := rfl

/-- Parameterized `_start` outcome: under a good-oracle predicate `G`, `.stuck`
and `.nomem` are ruled out, leaving `Reaches s (TripledState bs)`. -/
inductive InitOutcomeG (G : Prop) (bs : List Nat) (s : St) : Prop
  | stuck   : ¬ G → Ends s (· = .stuck) → InitOutcomeG G bs s
  | nomem   : ¬ G → Ends s (· = .exit 6) → InitOutcomeG G bs s
  | tripled : Reaches s (TripledState bs) → InitOutcomeG G bs s

theorem InitOutcomeG.toOutcome {G : Prop} {bs : List Nat} {s : St}
    (h : InitOutcomeG G bs s) : InitOutcome bs s := by
  rcases h with ⟨_, h⟩ | ⟨_, h⟩ | h
  · exact .stuck h
  · exact .nomem h
  · exact .tripled h

theorem InitOutcomeG.of_good {G : Prop} {bs : List Nat} {s : St}
    (h : InitOutcomeG G bs s) (hG : G) : Reaches s (TripledState bs) := by
  rcases h with ⟨hn, _⟩ | ⟨hn, _⟩ | h
  · exact absurd hG hn
  · exact absurd hG hn
  · exact h

theorem InitOutcomeG.of_reaches {G G' : Prop} {bs : List Nat} {s s' : St}
    (himp : G → G') (hr : Reaches s (· = s')) (ho : InitOutcomeG G' bs s') :
    InitOutcomeG G bs s := by
  rcases ho with ⟨hn, h⟩ | ⟨hn, h⟩ | h
  · exact .stuck (fun hG => hn (himp hG)) (hr.ends fun _ heq => heq ▸ h)
  · exact .nomem (fun hG => hn (himp hG)) (hr.ends fun _ heq => heq ▸ h)
  · exact .tripled (hr.trans fun _ heq => heq ▸ h)

/-- Good-oracle invariant at `L_eof`: `b1 = 0x400000000000`, `cap ≤ 2^46`,
`bs.length < 2^45`, and remaining oracle `[.at 0x800000000000, .at 0xc00000000000]`. -/
def EofOracleOk (b1 cap len : Nat) (o : List Choice) : Prop :=
  b1 = 0x400000000000 ∧ cap ≤ 2 ^ 46 ∧ len < 2 ^ 45 ∧
  o = [.at 0x800000000000, .at 0xc00000000000]

theorem initMem_hi_lt_arena :
    codeRegion.hi ≤ 0x400000000000 ∧ bssRegion.hi ≤ 0x400000000000 ∧
    initStk.1.hi ≤ 0x400000000000 := by
  have ⟨_, _, hhi⟩ := initStk_props
  refine ⟨?_, ?_, by rw [hhi]; decide⟩
  · unfold Region.hi
    have h1 : codeRegion.base = BASE := rfl
    have h2 : codeRegion.data.length = 47636 := spr1Arm64Bytes_len
    rw [h1, h2]; decide
  · unfold Region.hi
    have h1 : bssRegion.base = 0x600000 := rfl
    have h2 : bssRegion.data.length = 0x3000 := List.length_replicate
    rw [h1, h2]; decide

theorem pages_lt_pow46 {len : Nat} (hlen : len < 2 ^ 45) : pages (len + 1) ≤ 2 ^ 46 := by
  unfold pages page; omega

private theorem eof_noLowMem {rCode rStk : Region} {d6 data1 data2 data3 : List Nat} {b1 b2 b3 : Nat}
    (hCode_base : rCode.base = BASE) (hStk_lo : 4096 ≤ rStk.base)
    (hb1_lo : 4096 ≤ b1) (hb2_lo : 4096 ≤ b2) (hb3_lo : 4096 ≤ b3) :
    ∀ r ∈ ([rCode, ⟨0x600000, d6, true⟩, rStk, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] : Mem),
      4096 ≤ r.base := by
  intro r hr
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
  rcases hr with rfl | rfl | rfl | rfl | rfl | rfl
  · rw [hCode_base]; decide
  · dsimp only; decide
  · exact hStk_lo
  · exact hb1_lo
  · exact hb2_lo
  · exact hb3_lo

private theorem eof_descBytes {rCode rStk : Region} {bs data1 data2 data3 d6 : List Nat} {b1 b2 b3 : Nat}
    (hwf_6 : Mem.wf [rCode, ⟨0x600000, d6, true⟩, rStk, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩])
    (hd6_l : d6.length = 0x3000)
    (h1p : ∀ j < 8, d6[0 + j]? = some ((bytesN b1 8)[j]! % 256))
    (h1l : ∀ j < 8, d6[8 + j]? = some ((bytesN bs.length 8)[j]! % 256))
    (h2p : ∀ j < 8, d6[0x1000 + j]? = some ((bytesN b2 8)[j]! % 256))
    (h2l : ∀ j < 8, d6[0x1008 + j]? = some ((bytesN bs.length 8)[j]! % 256))
    (h3p : ∀ j < 8, d6[0x2000 + j]? = some ((bytesN b3 8)[j]! % 256))
    (h3l : ∀ j < 8, d6[0x2008 + j]? = some ((bytesN bs.length 8)[j]! % 256)) :
    ∀ d ∈ ([0x600000, 0x601000, 0x602000] : List Nat),
      ∀ j < 16, ∀ x,
        Mem.rd [rCode, ⟨0x600000, d6, true⟩, rStk, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] (d + j) = some x → x < 256 := by
  intro d hd j hj x hx
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hd
  have hreg1 : ([rCode, ⟨0x600000, d6, true⟩, rStk, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] : Mem)[1]? =
      some ⟨0x600000, d6, true⟩ := rfl
  rcases hd with rfl | rfl | rfl
  · rw [Mem.rd_at_idx hwf_6 hreg1 (by dsimp only; omega) (by simp only [Region.hi, hd6_l]; omega)] at hx
    simp only [Nat.add_sub_cancel_left] at hx
    by_cases hj8 : j < 8
    · rw [show j = 0 + j by omega, h1p j hj8] at hx; cases hx; exact Nat.mod_lt _ (by decide)
    · rw [show j = 8 + (j - 8) by omega, h1l (j - 8) (by omega)] at hx; cases hx; exact Nat.mod_lt _ (by decide)
  · rw [Mem.rd_at_idx hwf_6 hreg1 (by dsimp only; omega) (by simp only [Region.hi, hd6_l]; omega)] at hx
    rw [show 0x601000 + j - 0x600000 = 0x1000 + j by omega] at hx
    by_cases hj8 : j < 8
    · rw [h2p j hj8] at hx; cases hx; exact Nat.mod_lt _ (by decide)
    · rw [show 0x1000 + j = 0x1008 + (j - 8) by omega, h2l (j - 8) (by omega)] at hx; cases hx; exact Nat.mod_lt _ (by decide)
  · rw [Mem.rd_at_idx hwf_6 hreg1 (by dsimp only; omega) (by simp only [Region.hi, hd6_l]; omega)] at hx
    rw [show 0x602000 + j - 0x600000 = 0x2000 + j by omega] at hx
    by_cases hj8 : j < 8
    · rw [h3p j hj8] at hx; cases hx; exact Nat.mod_lt _ (by decide)
    · rw [show 0x2000 + j = 0x2008 + (j - 8) by omega, h3l (j - 8) (by omega)] at hx; cases hx; exact Nat.mod_lt _ (by decide)

private structure EofMemSpec (m : Mem) (bs : List Nat) (b1 b2 b3 : Nat) : Prop where
  wf      : m.wf
  noLow   : ∀ r ∈ m, 4096 ≤ r.base
  descBy  : ∀ d ∈ ([0x600000, 0x601000, 0x602000] : List Nat), ∀ j < 16, ∀ x, m.rd (d + j) = some x → x < 256
  hD1_p   : rdMem m 0x600000 .x8 = some b1
  hD1_l   : rdMem m 0x600008 .x8 = some bs.length
  hD2_p   : rdMem m 0x601000 .x8 = some b2
  hD2_l   : rdMem m 0x601008 .x8 = some bs.length
  hD3_p   : rdMem m 0x602000 .x8 = some b3
  hD3_l   : rdMem m 0x602008 .x8 = some bs.length
  hBuf1   : Buf m b1 bs
  hBuf2   : Buf m b2 bs
  hBuf3   : Buf m b3 bs
  hSm1    : b1 + bs.length < HMAX
  hSm2    : b2 + bs.length < HMAX
  hSm3    : b3 + bs.length < HMAX
  disj12  : b1 + bs.length ≤ b2 ∨ b2 + bs.length ≤ b1
  disj13  : b1 + bs.length ≤ b3 ∨ b3 + bs.length ≤ b1
  disj23  : b2 + bs.length ≤ b3 ∨ b3 + bs.length ≤ b2
  disjD1  : ∀ d ∈ ([0x600000, 0x601000, 0x602000] : List Nat), b1 + bs.length ≤ d ∨ d + 16 ≤ b1
  disjD2  : ∀ d ∈ ([0x600000, 0x601000, 0x602000] : List Nat), b2 + bs.length ≤ d ∨ d + 16 ≤ b2
  disjD3  : ∀ d ∈ ([0x600000, 0x601000, 0x602000] : List Nat), b3 + bs.length ≤ d ∨ d + 16 ≤ b3
  disjBss : ∀ b ∈ ([b1, b2, b3] : List Nat), b + bs.length ≤ 0x600000 ∨ 0x603000 ≤ b
  disjCode : ∀ b ∈ ([b1, b2, b3] : List Nat), b + bs.length ≤ BASE ∨ BASE + 47636 ≤ b
  hiB1    : 4096 ≤ b1
  hiB2    : 4096 ≤ b2
  hiB3    : 4096 ≤ b3

private theorem eof_mk_disj {rCode rStk : Region} {d6 data1 data2 data3 : List Nat} {len b1 b2 b3 : Nat}
    (hCode_base : rCode.base = BASE) (hCode_len : rCode.data.length = 47636) (hd6_l : d6.length = 0x3000)
    (hwf_6 : Mem.wf [rCode, ⟨0x600000, d6, true⟩, rStk, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩])
    (hd1_len : len ≤ data1.length) (hd2_len : len ≤ data2.length) (hd3_len : len ≤ data3.length) :
    (b1 + len ≤ b2 ∨ b2 + len ≤ b1) ∧
    (b1 + len ≤ b3 ∨ b3 + len ≤ b1) ∧
    (b2 + len ≤ b3 ∨ b3 + len ≤ b2) ∧
    (∀ d ∈ ([0x600000, 0x601000, 0x602000] : List Nat), b1 + len ≤ d ∨ d + 16 ≤ b1) ∧
    (∀ d ∈ ([0x600000, 0x601000, 0x602000] : List Nat), b2 + len ≤ d ∨ d + 16 ≤ b2) ∧
    (∀ d ∈ ([0x600000, 0x601000, 0x602000] : List Nat), b3 + len ≤ d ∨ d + 16 ≤ b3) ∧
    (∀ b ∈ ([b1, b2, b3] : List Nat), b + len ≤ 0x600000 ∨ 0x603000 ≤ b) ∧
    (∀ b ∈ ([b1, b2, b3] : List Nat), b + len ≤ BASE ∨ BASE + 47636 ≤ b) := by
  have ⟨hc_all, hpw1⟩ := List.pairwise_cons.1 hwf_6.2
  have ⟨hb_all, hpw2⟩ := List.pairwise_cons.1 hpw1
  have ⟨_, hpw3⟩ := List.pairwise_cons.1 hpw2
  have ⟨h1_all, hpw4⟩ := List.pairwise_cons.1 hpw3
  have ⟨h2_all, _⟩ := List.pairwise_cons.1 hpw4
  have hBss_imp : ∀ {b : Nat} {dat : List Nat}, len ≤ dat.length →
      (0x600000 + d6.length ≤ b ∨ b + dat.length ≤ 0x600000) →
      (b + len ≤ 0x600000 ∨ 0x603000 ≤ b) := fun {b} {_} hlen h =>
    h.elim (fun h1 => Or.inr (by rw [hd6_l] at h1; exact h1))
      (fun h2 => Or.inl (Nat.le_trans (Nat.add_le_add_left hlen b) h2))
  have hDisjD_imp : ∀ {b : Nat}, (b + len ≤ 0x600000 ∨ 0x603000 ≤ b) →
      ∀ d ∈ ([0x600000, 0x601000, 0x602000] : List Nat), b + len ≤ d ∨ d + 16 ≤ b := by
    intro b h d hd
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hd
    rcases hd with rfl | rfl | rfl
    · exact h.imp id (fun h2 => Nat.le_trans (by decide) h2)
    · exact h.imp (fun h1 => Nat.le_trans h1 (by decide)) (fun h2 => Nat.le_trans (by decide) h2)
    · exact h.imp (fun h1 => Nat.le_trans h1 (by decide)) (fun h2 => Nat.le_trans (by decide) h2)
  have hBss1' := hBss_imp hd1_len (hb_all ⟨b1, data1, true⟩ (by simp))
  have hBss2' := hBss_imp hd2_len (hb_all ⟨b2, data2, true⟩ (by simp))
  have hBss3' := hBss_imp hd3_len (hb_all ⟨b3, data3, true⟩ (by simp))
  have hCode_imp : ∀ {b : Nat} {dat : List Nat}, len ≤ dat.length →
      (rCode.base + rCode.data.length ≤ b ∨ b + dat.length ≤ rCode.base) →
      (b + len ≤ BASE ∨ BASE + 47636 ≤ b) := fun {b} {_} hlen h =>
    h.elim (fun h1 => Or.inr (by rw [hCode_base, hCode_len] at h1; exact h1))
      (fun h2 => Or.inl (Nat.le_trans (Nat.add_le_add_left hlen b) (by rw [hCode_base] at h2; exact h2)))
  have hCode1' := hCode_imp hd1_len (hc_all ⟨b1, data1, true⟩ (by simp))
  have hCode2' := hCode_imp hd2_len (hc_all ⟨b2, data2, true⟩ (by simp))
  have hCode3' := hCode_imp hd3_len (hc_all ⟨b3, data3, true⟩ (by simp))
  refine ⟨(h1_all ⟨b2, data2, true⟩ (by simp)).imp (Nat.le_trans (Nat.add_le_add_left hd1_len _)) (Nat.le_trans (Nat.add_le_add_left hd2_len _)),
    (h1_all ⟨b3, data3, true⟩ (by simp)).imp (Nat.le_trans (Nat.add_le_add_left hd1_len _)) (Nat.le_trans (Nat.add_le_add_left hd3_len _)),
    (h2_all ⟨b3, data3, true⟩ (by simp)).imp (Nat.le_trans (Nat.add_le_add_left hd2_len _)) (Nat.le_trans (Nat.add_le_add_left hd3_len _)),
    hDisjD_imp hBss1', hDisjD_imp hBss2', hDisjD_imp hBss3', ?_, ?_⟩
  · intro b hb; simp only [List.mem_cons, List.not_mem_nil, or_false] at hb
    rcases hb with rfl | rfl | rfl
    · exact hBss1'
    · exact hBss2'
    · exact hBss3'
  · intro b hb; simp only [List.mem_cons, List.not_mem_nil, or_false] at hb
    rcases hb with rfl | rfl | rfl
    · exact hCode1'
    · exact hCode2'
    · exact hCode3'

private theorem eof_mk_tripled_of_mem {m : Mem} {bs : List Nat} {b1 b2 b3 : Nat} {sF : St}
    (hmF : sF.mem = m) (hby : ∀ b ∈ bs, b < 256)
    (hpcF : sF.pc = L_tripled) (hlF : sF.regs.length = 31)
    (hoF : OracleWf sF.k.oracle) (h19F : sF.reg 19 = some REJECT)
    (h20F : sF.reg 20 = some REJECT) (h21F : sF.reg 21 = some REJECT)
    (hscrF' : VoterScratchClean sF) (hs : EofMemSpec m bs b1 b2 b3) : TripledState bs sF := by
  have htb1 : tripledB1 sF = b1 := by unfold tripledB1; rw [hmF]; exact congrArg (·.getD 0) hs.hD1_p
  have htb2 : tripledB2 sF = b2 := by unfold tripledB2; rw [hmF]; exact congrArg (·.getD 0) hs.hD2_p
  have htb3 : tripledB3 sF = b3 := by unfold tripledB3; rw [hmF]; exact congrArg (·.getD 0) hs.hD3_p
  exact
    { pc := hpcF, regsLen := hlF, memWf := hmF ▸ hs.wf
      noLowMem := hmF ▸ hs.noLow, descBytes := hmF ▸ hs.descBy
      oracleWf := hoF, bytes := hby, x19 := h19F, x20 := h20F, x21 := h21F, scratch := hscrF'
      hD1_ptr := by rw [hmF, htb1]; exact hs.hD1_p
      hD1_len := hmF ▸ hs.hD1_l
      hD2_ptr := by rw [hmF, htb2]; exact hs.hD2_p
      hD2_len := hmF ▸ hs.hD2_l
      hD3_ptr := by rw [hmF, htb3]; exact hs.hD3_p
      hD3_len := hmF ▸ hs.hD3_l
      hBuf1 := by rw [hmF, htb1]; exact hs.hBuf1
      hBuf2 := by rw [hmF, htb2]; exact hs.hBuf2
      hBuf3 := by rw [hmF, htb3]; exact hs.hBuf3
      hSm1 := htb1 ▸ hs.hSm1, hSm2 := htb2 ▸ hs.hSm2, hSm3 := htb3 ▸ hs.hSm3
      disj12 := by rw [htb1, htb2]; exact hs.disj12
      disj13 := by rw [htb1, htb3]; exact hs.disj13
      disj23 := by rw [htb2, htb3]; exact hs.disj23
      disjD1 := htb1 ▸ hs.disjD1, disjD2 := htb2 ▸ hs.disjD2, disjD3 := htb3 ▸ hs.disjD3
      disjBss := by rw [htb1, htb2, htb3]; exact hs.disjBss
      disjCode := by rw [htb1, htb2, htb3]; exact hs.disjCode
      hiB1 := htb1 ▸ hs.hiB1, hiB2 := htb2 ▸ hs.hiB2, hiB3 := htb3 ▸ hs.hiB3 }

private theorem eof_rd_bss {rCode rStk : Region} {data1 data2 data3 d6 : List Nat}
    {b1 b2 b3 off addr v : Nat}
    (hwf_6 : Mem.wf [rCode, ⟨0x600000, d6, true⟩, rStk, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩])
    (hd6_l : d6.length = 0x3000)
    (haddr : 0x600000 + off = addr) (hoff : off + 8 ≤ 0x3000) (hv : v < 2 ^ 64)
    (hp : ∀ j < 8, d6[off + j]? = some ((bytesN v 8)[j]! % 256)) :
    rdMem [rCode, ⟨0x600000, d6, true⟩, rStk, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] addr .x8 = some v := by
  have h := rdMem_x8_of_region (idx := 1) (off := off) (v := v) hwf_6 rfl
    (by dsimp only; rw [hd6_l]; exact hoff) hv hp
  simpa only [haddr] using h

theorem eof_to_tripled {bs : List Nat} {b1 cap : Nat} {data1 : List Nat} {s : St}
    (hby : ∀ b ∈ bs, b < 256) (ho : OracleWf s.k.oracle)
    (hl : s.regs.length = 31) (hpc : s.pc = L_eof)
    (hm : s.mem = [codeRegion, bssRegion, initStk.1, ⟨b1, data1, true⟩]) (hwf : s.mem.wf)
    (hb1_lo : 4096 ≤ b1)
    (hb1_hi : b1 + cap ≤ userMax) (hlen_lt : bs.length < cap) (hd1_len : data1.length = cap)
    (hd1_bs : ∀ j < bs.length, data1[j]? = some bs[j]!)
    (h13 : s.reg 13 = some bs.length) (h14 : s.reg 14 = some b1)
    (hzero : ∀ r ∈ ([16, 17, 18, 22, 23, 27, 28, 29] : List Nat), s.reg r = some 0) :
    InitOutcomeG (EofOracleOk b1 cap bs.length s.k.oracle) bs s := by
  have hc := init_codeAt
  simp only [initTable, initTable_0, initTable_1, List.cons_append, List.nil_append,
    CodeAt_cons] at hc
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _,
    f35, f36, f37, f38, f39, f40, f41, f42, f43, f44, f45, f46, f47, f48,
    c1_0, c1_1, c1_2, c1_3, c1_4, c1_5, c1_6,
    f56, f57, f58, f59, f60, f61, f62, f63, f64, f65, f66, f67, f68, f69,
    c2_0, c2_1, c2_2, c2_3, c2_4, c2_5, c2_6,
    fd0, fd1, fd2, fd3, fd4, fd5, fd6, fd7, fd8,
    fr0, fr1, fr2, fr3, fr4, fr5, fr6, fr7, fr8, fr9, fr10, fr11, fb_tripled, _⟩ := hc
  have hlen_lt48 : bs.length < 2 ^ 48 := by unfold userMax at hb1_hi; omega
  have hlen1 : (bs.length % 2 ^ 64 + 1 % 2 ^ 64) % 2 ^ 64 = bs.length + 1 := by omega
  have hb1_lt48 : b1 + bs.length < HMAX := by unfold HMAX userMax at *; omega
  let sA :=
    St.setReg (St.setReg (St.setReg (St.setReg (St.setReg (St.setReg (St.setReg
      (s.withPc 4241712) 0 (some 0)) 1 (some (bs.length + 1))) 2 (some 3))
      3 (some 34)) 4 (some (2 ^ 64 - 1))) 5 (some 0)) 8 (some 222)
  have hpre2 : Reaches s (· = sA) := by
    have hpc0 : s.pc = 4241684 := hpc
    xs f35 using hpc0, hl, (by decide : (4241684 + 4) % AArch64Env.W = 4241688)
    xs f36 using hl, h13, hlen1, (by decide : (4241688 + 4) % AArch64Env.W = 4241692)
    xs f37 using hl, (by decide : (4241692 + 4) % AArch64Env.W = 4241696)
    xs f38 using hl, (by decide : (4241696 + 4) % AArch64Env.W = 4241700)
    xs f39 using hl, (by decide : (4241700 + 4) % AArch64Env.W = 4241704)
    xs f40 using hl, (by decide : (4241704 + 4) % AArch64Env.W = 4241708)
    xs f41 using hl, (by decide : (4241708 + 4) % AArch64Env.W = 4241712)
    exact Reaches.refl rfl
  refine InitOutcomeG.of_reaches id hpre2 ?_
  have hsA_l : sA.regs.length = 31 := by simp [sA, hl]
  have hsA_pc : sA.pc = 4241712 := rfl
  have hsA_0 : sA.reg 0 = some 0 := by simp [sA, hl]
  have hsA_1 : sA.reg 1 = some (bs.length + 1) := by simp [sA, hl]
  have hsA_2 : sA.reg 2 = some 3 := by simp [sA, hl]
  have hsA_3 : sA.reg 3 = some 34 := by simp [sA, hl]
  have hsA_4 : sA.reg 4 = some (2 ^ 64 - 1) := by simp [sA, hl]
  have hsA_5 : sA.reg 5 = some 0 := by simp [sA, hl]
  have hsA_8 : sA.reg 8 = some 222 := by simp [sA, hl]
  cases hn2 : s.k.next with
  | mk oc2 k2 =>
    have ho2 : OracleWf k2.oracle := kern_next_oracle_wf ho hn2
    cases hp2 : place s.mem oc2 (pages (bs.length + 1)) with
    | none =>
      refine .stuck ?_ ⟨1, ?_⟩
      · rintro ⟨rfl, hcap46, hlen45, hor⟩
        have hoc : oc2 = some (.at 0x800000000000) := by
          unfold Kern.next at hn2; rw [hor] at hn2; cases hn2; rfl
        have hpg46 := pages_lt_pow46 hlen45
        have ⟨hcH, hbH, hsH⟩ := initMem_hi_lt_arena
        have hfree : s.mem.free 0x800000000000 (pages (bs.length + 1)) = true := by
          rw [hm]; simp only [Mem.free, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true,
            Bool.or_eq_true, decide_eq_true_eq]
          exact ⟨Or.inr (by omega), Or.inr (by omega), Or.inr (by omega),
            Or.inr (by simp only [Region.hi, hd1_len]; omega)⟩
        rw [hoc] at hp2
        simp only [place, show (0x800000000000 : Nat) % page = 0 by decide,
          show (0 : Nat) < 0x800000000000 by decide,
          show 0x800000000000 + pages (bs.length + 1) ≤ userMax by unfold userMax; omega,
          hfree, and_self, ↓reduceIte] at hp2
        cases hp2
      · simp [run, step, hsA_pc, f42, exec, execSvc, applySys, syscall,
          hsA_0, hsA_1, hsA_2, hsA_3, hsA_4, hsA_5, hsA_8,
          show sA.k = s.k from rfl, show sA.mem = s.mem from rfl, hn2, hp2,
          PROT_RW, MAP_PRIVATE_ANON, AArch64Env.W]
    | some ob2 =>
      cases ob2 with
      | none =>
        refine .nomem ?_ ?_
        · rintro ⟨_, _, _, hor⟩
          have hoc : oc2 = some (.at 0x800000000000) := by
            unfold Kern.next at hn2; rw [hor] at hn2; cases hn2; rfl
          rw [hoc] at hp2; simp [place] at hp2
        · let sA' := ((sA.withPc 4241716).setReg 0 (some (AArch64Env.neg ENOMEM % AArch64Env.W))).withKM k2 s.mem
          have hs_svc : step im sA = .ok sA' := by
            simp [step, hsA_pc, f42, exec, execSvc, applySys, syscall,
              hsA_0, hsA_1, hsA_2, hsA_3, hsA_4, hsA_5, hsA_8,
              show sA.k = s.k from rfl, show sA.mem = s.mem from rfl, hn2, hp2,
              PROT_RW, MAP_PRIVATE_ANON, AArch64Env.W, sA', St.withKM]
          have hto_nm : Reaches sA' (fun t => t.regs.length = 31 ∧ t.pc = L_nomem) := by
            have hpcA' : sA'.pc = 4241716 := rfl
            have hlA' : sA'.regs.length = 31 := by simp [sA', hsA_l]
            have hr0A' : sA'.reg 0 = some (AArch64Env.neg ENOMEM % AArch64Env.W) := by simp [sA', hsA_l]
            xs f43 using hpcA', hlA', hr0A', (by decide : (4241716 + 4) % AArch64Env.W = 4241720)
            xs f44 using cond_cmp_zero_enomem, (by decide : (4241720 + 18446744073709534760) % AArch64Env.W = L_nomem)
            exact Reaches.refl ⟨by simp [hlA'], rfl⟩
          obtain ⟨n', t', hrun', htl', htpc'⟩ := hto_nm
          refine ⟨1 + n' + 5, ?_⟩
          rw [run_add, run_add, show run im 1 (.ok sA) = step im sA from rfl, hs_svc, hrun', nomem_spec htl' htpc']
      | some b2 =>
        have ⟨hb2_pg, hb2_pos, hb2_hi, hb2_free⟩ := place_spec hp2
        have hb2_lo : 4096 ≤ b2 := by unfold page at hb2_pg; omega
        have hpg2_ge : bs.length + 1 ≤ pages (bs.length + 1) := by unfold pages page; omega
        have hb2_lt48 : b2 + bs.length < HMAX := by unfold HMAX userMax at *; omega
        let r2 : Region := ⟨b2, List.replicate (pages (bs.length + 1)) 0, true⟩
        let m2 : Mem := s.mem ++ [r2]
        have hm2_wf : m2.wf := wf_append hwf hb2_pg (by simpa [r2] using hb2_free)
        have hm2_eq : m2 = [codeRegion, bssRegion, initStk.1, ⟨b1, data1, true⟩, r2] := by simp [m2, hm]
        let sA' := ((sA.withPc 4241716).setReg 0 (some b2)).withKM k2 m2
        have hs_svc2 : step im sA = .ok sA' := by
          have hb2_m : b2 % 2 ^ 64 = b2 := Nat.mod_eq_of_lt (by unfold userMax at hb2_hi; omega)
          simp [step, hsA_pc, f42, exec, execSvc, applySys, syscall,
            hsA_0, hsA_1, hsA_2, hsA_3, hsA_4, hsA_5, hsA_8,
            show sA.k = s.k from rfl, show sA.mem = s.mem from rfl, hn2, hp2,
            PROT_RW, MAP_PRIVATE_ANON, AArch64Env.W, hb2_m, sA', St.withKM, m2, r2]
        have hstep_b2 : Reaches sA fun t =>
            t.pc = 4241796 ∧ t.regs.length = 31 ∧ t.k = k2 ∧ t.mem.wf ∧
            t.reg 0 = some 0 ∧ t.reg 1 = some (bs.length + 1) ∧ t.reg 2 = some 3 ∧
            t.reg 3 = some 34 ∧ t.reg 4 = some (2 ^ 64 - 1) ∧ t.reg 5 = some 0 ∧ t.reg 8 = some 222 ∧
            t.reg 13 = some bs.length ∧ t.reg 14 = some b1 ∧ t.reg 15 = some b2 ∧
            (∀ r ∈ ([16, 17, 18, 22, 23, 27, 28, 29] : List Nat), t.reg r = some 0) ∧
            ∃ data2, t.mem = [codeRegion, bssRegion, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩] ∧
              data2.length = pages (bs.length + 1) ∧ ∀ j < bs.length, data2[j]? = some bs[j]! := by
          refine Reaches.trans (P := (· = sA')) ⟨1, sA', by simp [run, hs_svc2], rfl⟩ fun t0 ht0 => ?_
          subst ht0
          have ht0_pc : sA'.pc = 4241716 := rfl
          have ht0_l : sA'.regs.length = 31 := by simp [sA', hsA_l]
          have ht0_0 : sA'.reg 0 = some b2 := by simp [sA', hsA_l]
          have ht0_13 : sA'.reg 13 = some bs.length := by simp [sA', sA, h13]
          have ht0_14 : sA'.reg 14 = some b1 := by simp [sA', sA, h14]
          have hcmp2 : AArch64Phys.cond (addSubOp true 64 (b2 % 18446744073709551616) 0).2 4 = some false :=
            (cond_cmp_zero_lt63 (by unfold userMax at hb2_hi; omega : b2 < 2 ^ 63)).1
          have hb2_m64 : b2 % 18446744073709551616 = b2 := Nat.mod_eq_of_lt (by unfold userMax at hb2_hi; omega)
          have hb1_m64 : b1 % 18446744073709551616 = b1 := Nat.mod_eq_of_lt (by unfold userMax at hb1_hi; omega)
          have hlen_m64 : bs.length % 18446744073709551616 = bs.length := Nat.mod_eq_of_lt (by omega)
          xs f43 using ht0_pc, ht0_l, ht0_0, (by decide : (4241716 + 4) % AArch64Env.W = 4241720)
          xs f44 using hcmp2, (by decide : (4241720 + 4) % AArch64Env.W = 4241724)
          xs f45 using ht0_l, ht0_0, hb2_m64, (by decide : (4241724 + 4) % AArch64Env.W = 4241728)
          xs f46 using ht0_l, ht0_14, hb1_m64, (by decide : (4241728 + 4) % AArch64Env.W = 4241732)
          xs f47 using ht0_l, hb2_m64, (by decide : (4241732 + 4) % AArch64Env.W = 4241736)
          xs f48 using ht0_l, ht0_13, hlen_m64, (by decide : (4241736 + 4) % AArch64Env.W = 4241740)
          refine (copy_loop_spec c1_0 c1_1 c1_2 c1_3 c1_4 c1_5 c1_6 (by decide) (by decide : (3 : Nat) ≠ 4)
            hby hlen_lt48 (by unfold HMAX at hb1_lt48; omega) (by unfold HMAX at hb2_lt48; omega)
            bs.length 0 _ ⟨b1, data1, true⟩ r2 (by omega) (by simp [ht0_l]) rfl hm2_wf
            (by change m2[3]? = _; rw [hm2_eq]; rfl) (by change m2[4]? = _; rw [hm2_eq]; rfl) rfl rfl rfl
            (by simp only [hd1_len]; omega) (by simp only [r2, List.length_replicate]; omega)
            hd1_bs (fun _ h => by cases h)
            (by simp [ht0_l]) (by simp [ht0_l, r2]) (by simp [ht0_l])).trans
            fun t1 ⟨hpc1, hl1, _, hk1, hreg1, hwf1, data2, hm1, hlen2, hd2⟩ => ?_
          have ht1_13 : t1.reg 13 = some bs.length := by rw [hreg1 13 (by decide)]; simp [ht0_13]
          have ht1_14 : t1.reg 14 = some b1 := by rw [hreg1 14 (by decide)]; simp [ht0_14]
          have ht1_15 : t1.reg 15 = some b2 := by rw [hreg1 15 (by decide)]; simp [ht0_l]
          have ht1_z : ∀ r ∈ ([16, 17, 18, 22, 23, 27, 28, 29] : List Nat), t1.reg r = some 0 := by
            intro r hr
            have hne : r ∉ ([6, 7, 10, 11] : List Nat) := by
              simp only [List.mem_cons, List.not_mem_nil, or_false] at hr ⊢
              rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide
            rw [hreg1 r hne]
            simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
            rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
              simpa [sA', sA] using hzero _ (by simp)
          xs f56 using hl1, (by decide : (4241768 + 4) % AArch64Env.W = 4241772)
          xs f57 using hl1, ht1_13, hlen1, (by decide : (4241772 + 4) % AArch64Env.W = 4241776)
          xs f58 using hl1, (by decide : (4241776 + 4) % AArch64Env.W = 4241780)
          xs f59 using hl1, (by decide : (4241780 + 4) % AArch64Env.W = 4241784)
          xs f60 using hl1, (by decide : (4241784 + 4) % AArch64Env.W = 4241788)
          xs f61 using hl1, (by decide : (4241788 + 4) % AArch64Env.W = 4241792)
          xs f62 using hl1, (by decide : (4241792 + 4) % AArch64Env.W = 4241796)
          refine Reaches.refl ⟨rfl, by simp [hl1], hk1, hwf1, by simp [hl1], by simp [hl1], by simp [hl1],
            by simp [hl1], by simp [hl1], by simp [hl1], by simp [hl1], by simp [ht1_13], by simp [ht1_14],
            by simp [ht1_15], ?_, data2, by simp only [setReg_mem, withPc_mem]; rw [hm1]; change m2.set 4 _ = _; rw [hm2_eq]; rfl,
            by simpa [r2] using hlen2, hd2⟩
          intro r hr
          simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
          rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
            simpa using ht1_z _ (by simp)
        obtain ⟨nB, sB, hrunB, hpcB, hlB, hkB, hwfB, hB0, hB1, hB2, hB3, hB4, hB5, hB8,
          hB13, hB14, hB15, hBz, data2, hmB, hlen2, hd2⟩ := hstep_b2
        refine InitOutcomeG.of_reaches id ⟨nB, sB, hrunB, rfl⟩ ?_
        -- mmap #3 for b3
        cases hn3 : k2.next with
        | mk oc3 k3 =>
          have ho3 : OracleWf k3.oracle := kern_next_oracle_wf ho2 hn3
          cases hp3 : place sB.mem oc3 (pages (bs.length + 1)) with
          | none =>
            refine .stuck ?_ ⟨1, ?_⟩
            · rintro ⟨rfl, hcap46, hlen45, hor⟩
              have hpair2 : oc2 = some (.at 0x800000000000) ∧ k2.oracle = [.at 0xc00000000000] := by
                unfold Kern.next at hn2; rw [hor] at hn2; cases hn2; exact ⟨rfl, rfl⟩
              have hb2_eq : b2 = 0x800000000000 := by
                rw [hpair2.1] at hp2; simp only [place] at hp2
                split at hp2 <;> cases hp2; rfl
              have hoc3 : oc3 = some (.at 0xc00000000000) := by
                unfold Kern.next at hn3; rw [hpair2.2] at hn3; cases hn3; rfl
              have hpg46 := pages_lt_pow46 hlen45
              have ⟨hcH, hbH, hsH⟩ := initMem_hi_lt_arena
              have hfree3 : sB.mem.free 0xc00000000000 (pages (bs.length + 1)) = true := by
                rw [hmB, hb2_eq]; simp only [Mem.free, List.all_cons, List.all_nil, Bool.and_true,
                  Bool.and_eq_true, Bool.or_eq_true, decide_eq_true_eq]
                exact ⟨Or.inr (by omega), Or.inr (by omega), Or.inr (by omega),
                  Or.inr (by simp only [Region.hi, hd1_len]; omega),
                  Or.inr (by simp only [Region.hi, hlen2]; omega)⟩
              rw [hoc3] at hp3
              simp only [place, show (0xc00000000000 : Nat) % page = 0 by decide,
                show (0 : Nat) < 0xc00000000000 by decide,
                show 0xc00000000000 + pages (bs.length + 1) ≤ userMax by unfold userMax; omega,
                hfree3, and_self, ↓reduceIte] at hp3
              cases hp3
            · simp [run, step, hpcB, f63, exec, execSvc, applySys, syscall, hB0, hB1, hB2, hB3, hB4, hB5, hB8,
                hkB, hn3, hp3, PROT_RW, MAP_PRIVATE_ANON, AArch64Env.W]
          | some ob3 =>
            cases ob3 with
            | none =>
              refine .nomem ?_ ?_
              · rintro ⟨_, _, _, hor⟩
                have hk2_or : k2.oracle = [.at 0xc00000000000] := by
                  unfold Kern.next at hn2; rw [hor] at hn2; cases hn2; rfl
                have hoc3 : oc3 = some (.at 0xc00000000000) := by
                  unfold Kern.next at hn3; rw [hk2_or] at hn3; cases hn3; rfl
                rw [hoc3] at hp3; simp [place] at hp3
              · let sB' := ((sB.withPc 4241800).setReg 0 (some (AArch64Env.neg ENOMEM % AArch64Env.W))).withKM k3 sB.mem
                have hs_svc : step im sB = .ok sB' := by
                  simp [step, hpcB, f63, exec, execSvc, applySys, syscall, hB0, hB1, hB2, hB3, hB4, hB5, hB8,
                    hkB, hn3, hp3, PROT_RW, MAP_PRIVATE_ANON, AArch64Env.W, sB', St.withKM]
                have hto_nm : Reaches sB' (fun t => t.regs.length = 31 ∧ t.pc = L_nomem) := by
                  have hpcB' : sB'.pc = 4241800 := rfl
                  have hlB' : sB'.regs.length = 31 := by simp [sB', hlB]
                  have hr0B' : sB'.reg 0 = some (AArch64Env.neg ENOMEM % AArch64Env.W) := by simp [sB', hlB]
                  xs f64 using hpcB', hlB', hr0B', (by decide : (4241800 + 4) % AArch64Env.W = 4241804)
                  xs f65 using cond_cmp_zero_enomem, (by decide : (4241804 + 18446744073709534676) % AArch64Env.W = L_nomem)
                  exact Reaches.refl ⟨by simp [hlB'], rfl⟩
                obtain ⟨n', t', hrun', htl', htpc'⟩ := hto_nm
                refine ⟨1 + n' + 5, ?_⟩
                rw [run_add, run_add, show run im 1 (.ok sB) = step im sB from rfl, hs_svc, hrun', nomem_spec htl' htpc']
            | some b3 =>
              have ⟨hb3_pg, hb3_pos, hb3_hi, hb3_free⟩ := place_spec hp3
              have hb3_lo : 4096 ≤ b3 := by unfold page at hb3_pg; omega
              have hb3_lt48 : b3 + bs.length < HMAX := by unfold HMAX userMax at *; omega
              let r3 : Region := ⟨b3, List.replicate (pages (bs.length + 1)) 0, true⟩
              let m3 : Mem := sB.mem ++ [r3]
              have hm3_wf : m3.wf := wf_append hwfB hb3_pg (by simpa [r3] using hb3_free)
              have hm3_eq : m3 = [codeRegion, bssRegion, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, r3] := by
                simp [m3, hmB]
              let sC := ((sB.withPc 4241800).setReg 0 (some b3)).withKM k3 m3
              have hs_svc3 : step im sB = .ok sC := by
                have hb3_m : b3 % 2 ^ 64 = b3 := Nat.mod_eq_of_lt (by unfold userMax at hb3_hi; omega)
                simp [step, hpcB, f63, exec, execSvc, applySys, syscall, hB0, hB1, hB2, hB3, hB4, hB5, hB8,
                  hkB, hn3, hp3, PROT_RW, MAP_PRIVATE_ANON, AArch64Env.W, hb3_m, sC, St.withKM, m3, r3]
              refine .tripled ?_
              refine Reaches.trans (P := (· = sC)) ⟨1, sC, by simp [run, hs_svc3], rfl⟩ fun t0 ht0 => ?_
              subst ht0
              have hpcC : sC.pc = 4241800 := rfl
              have hlC : sC.regs.length = 31 := by simp [sC, hlB]
              have hr0C : sC.reg 0 = some b3 := by simp [sC, hlB]
              have hr13C : sC.reg 13 = some bs.length := by simp [sC, hB13]
              have hr14C : sC.reg 14 = some b1 := by simp [sC, hB14]
              have hcmp3 : AArch64Phys.cond (addSubOp true 64 (b3 % 18446744073709551616) 0).2 4 = some false :=
                (cond_cmp_zero_lt63 (by unfold userMax at hb3_hi; omega : b3 < 2 ^ 63)).1
              have hb3_m64 : b3 % 18446744073709551616 = b3 := Nat.mod_eq_of_lt (by unfold userMax at hb3_hi; omega)
              have hb2_m64 : b2 % 18446744073709551616 = b2 := Nat.mod_eq_of_lt (by unfold userMax at hb2_hi; omega)
              have hb1_m64 : b1 % 18446744073709551616 = b1 := Nat.mod_eq_of_lt (by unfold userMax at hb1_hi; omega)
              have hlen_m64 : bs.length % 18446744073709551616 = bs.length := Nat.mod_eq_of_lt (by omega)
              xs f64 using hpcC, hlC, hr0C, (by decide : (4241800 + 4) % AArch64Env.W = 4241804)
              xs f65 using hcmp3, (by decide : (4241804 + 4) % AArch64Env.W = 4241808)
              xs f66 using hlC, hr0C, hb3_m64, (by decide : (4241808 + 4) % AArch64Env.W = 4241812)
              xs f67 using hlC, hr14C, hb1_m64, (by decide : (4241812 + 4) % AArch64Env.W = 4241816)
              xs f68 using hlC, hb3_m64, (by decide : (4241816 + 4) % AArch64Env.W = 4241820)
              xs f69 using hlC, hr13C, hlen_m64, (by decide : (4241820 + 4) % AArch64Env.W = 4241824)
              refine (copy_loop_spec c2_0 c2_1 c2_2 c2_3 c2_4 c2_5 c2_6 (by decide) (by decide : (3 : Nat) ≠ 5)
                hby hlen_lt48 (by unfold HMAX at hb1_lt48; omega) (by unfold HMAX at hb3_lt48; omega)
                bs.length 0 _ ⟨b1, data1, true⟩ r3 (by omega) (by simp [hlC]) rfl hm3_wf
                (by change m3[3]? = _; rw [hm3_eq]; rfl) (by change m3[5]? = _; rw [hm3_eq]; rfl) rfl rfl rfl
                (by simp only [hd1_len]; omega) (by simp only [r3, List.length_replicate]; omega)
                hd1_bs (fun _ h => by cases h)
                (by simp [hlC]) (by simp [hlC, r3]) (by simp [hlC])).trans
                fun t2 ⟨hpc2, hl2, _, hk2, hreg2, hwf2, data3, hm2, hlen3, hd3⟩ => ?_
              have hm2_list : t2.mem = [codeRegion, bssRegion, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] := by
                rw [hm2]; change m3.set 5 _ = _; rw [hm3_eq]; rfl
              have ht2_13 : t2.reg 13 = some bs.length := by rw [hreg2 13 (by decide)]; simp [hr13C]
              have ht2_14 : t2.reg 14 = some b1 := by rw [hreg2 14 (by decide)]; simp [hr14C]
              have ht2_15 : t2.reg 15 = some b2 := by rw [hreg2 15 (by decide)]; simp [sC, hB15]
              have ht2_24 : t2.reg 24 = some b3 := by rw [hreg2 24 (by decide)]; simp [hlC]
              have ht2_3 : t2.reg 3 = some 34 := by rw [hreg2 3 (by decide)]; simp [sC, hB3]
              have ht2_4 : t2.reg 4 = some (2 ^ 64 - 1) := by rw [hreg2 4 (by decide)]; simp [sC, hB4]
              have ht2_5 : t2.reg 5 = some 0 := by rw [hreg2 5 (by decide)]; simp [sC, hB5]
              have ht2_z : ∀ r ∈ ([16, 17, 18, 22, 23, 27, 28, 29] : List Nat), t2.reg r = some 0 := by
                intro r hr
                have hne : r ∉ ([6, 7, 10, 11] : List Nat) := by
                  simp only [List.mem_cons, List.not_mem_nil, or_false] at hr ⊢
                  rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide
                rw [hreg2 r hne]
                simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
                rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
                  simpa [sC] using hBz _ (by simp)
              have hwr_bss : ∀ (mCur : Mem) (dCur : List Nat) (off v : Nat),
                  mCur.wf → mCur = [codeRegion, ⟨0x600000, dCur, true⟩, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] →
                  dCur.length = 0x3000 → off + 8 ≤ 0x3000 →
                  ∃ dNext, wrMem mCur (0x600000 + off) v .x8 =
                      some [codeRegion, ⟨0x600000, dNext, true⟩, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] ∧
                    Mem.wf [codeRegion, ⟨0x600000, dNext, true⟩, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] ∧
                    dNext.length = 0x3000 ∧
                    (∀ j < off, dNext[j]? = dCur[j]?) ∧
                    (∀ j < 8, dNext[off + j]? = some ((bytesN v 8)[j]! % 256)) ∧
                    (∀ j, off + 8 ≤ j → dNext[j]? = dCur[j]?) := by
                intro mCur dCur off v hwfC hmC hdLen hoff
                obtain ⟨dN, hwrN, hlenN, hltN, hmidN, hgeN⟩ :=
                  Mem.wrN_at_idx (bytesN v 8) mCur 1 off ⟨0x600000, dCur, true⟩ hwfC
                    (by rw [hmC]; rfl) rfl (by simp [bytesN, hdLen]; omega)
                have hwfN : Mem.wf (mCur.set 1 ⟨0x600000, dN, true⟩) := wf_of_shape (wrN_shape _ _ _ hwrN) hwfC
                have hset_eq : mCur.set 1 ⟨0x600000, dN, true⟩ =
                    [codeRegion, ⟨0x600000, dN, true⟩, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] := by
                  rw [hmC]; rfl
                rw [hset_eq] at hwrN hwfN
                exact ⟨dN, by simpa [wrMem, MemSize.bytes] using hwrN, hwfN, by simpa [hdLen] using hlenN, hltN, hmidN, hgeN⟩
              obtain ⟨d1, hw1, hwf_1, hd1_l, _, hd1_m, _⟩ :=
                hwr_bss t2.mem (List.replicate 0x3000 0) 0 b1 hwf2 hm2_list List.length_replicate (by decide)
              obtain ⟨d2, hw2, hwf_2, hd2_l, hd2_lt, hd2_m, _⟩ :=
                hwr_bss _ d1 8 bs.length hwf_1 rfl hd1_l (by decide)
              obtain ⟨d3, hw3, hwf_3, hd3_l, hd3_lt, hd3_m, _⟩ :=
                hwr_bss _ d2 0x1000 b2 hwf_2 rfl hd2_l (by decide)
              obtain ⟨d4, hw4, hwf_4, hd4_l, hd4_lt, hd4_m, _⟩ :=
                hwr_bss _ d3 0x1008 bs.length hwf_3 rfl hd3_l (by decide)
              obtain ⟨d5, hw5, hwf_5, hd5_l, hd5_lt, hd5_m, _⟩ :=
                hwr_bss _ d4 0x2000 b3 hwf_4 rfl hd4_l (by decide)
              obtain ⟨d6, hw6, hwf_6, hd6_l, hd6_lt, hd6_m, _⟩ :=
                hwr_bss _ d5 0x2008 bs.length hwf_5 rfl hd5_l (by decide)
              have hw1' : wrMem t2.mem 0x600000 b1 .x8 =
                  some [codeRegion, ⟨0x600000, d1, true⟩, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] := hw1
              have hw2' : wrMem [codeRegion, ⟨0x600000, d1, true⟩, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] 0x600008 bs.length .x8 =
                  some [codeRegion, ⟨0x600000, d2, true⟩, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] := hw2
              have hw3' : wrMem [codeRegion, ⟨0x600000, d2, true⟩, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] 0x601000 b2 .x8 =
                  some [codeRegion, ⟨0x600000, d3, true⟩, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] := hw3
              have hw4' : wrMem [codeRegion, ⟨0x600000, d3, true⟩, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] 0x601008 bs.length .x8 =
                  some [codeRegion, ⟨0x600000, d4, true⟩, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] := hw4
              have hw5' : wrMem [codeRegion, ⟨0x600000, d4, true⟩, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] 0x602000 b3 .x8 =
                  some [codeRegion, ⟨0x600000, d5, true⟩, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] := hw5
              have hw6' : wrMem [codeRegion, ⟨0x600000, d5, true⟩, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] 0x602008 bs.length .x8 =
                  some [codeRegion, ⟨0x600000, d6, true⟩, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] := hw6
              xs fd0 using hpc2, hl2, (by decide : (4241852 / 4096 * 4096 + 2052096) % AArch64Env.W = 0x600000),
                (by decide : (4241852 + 4) % AArch64Env.W = 4241856)
              xs fd1 using hl2, ht2_14, hb1_m64, (by decide : (0x600000 + 0) % AArch64Env.W = 0x600000),
                hw1', (by decide : (4241856 + 4) % AArch64Env.W = 4241860)
              xs fd2 using hl2, ht2_13, hlen_m64, (by decide : (0x600000 + 8) % AArch64Env.W = 0x600008),
                hw2', (by decide : (4241860 + 4) % AArch64Env.W = 4241864)
              xs fd3 using hl2, (by decide : (4241864 / 4096 * 4096 + 2056192) % AArch64Env.W = 0x601000),
                (by decide : (4241864 + 4) % AArch64Env.W = 4241868)
              xs fd4 using hl2, ht2_15, hb2_m64, (by decide : (0x601000 + 0) % AArch64Env.W = 0x601000),
                hw3', (by decide : (4241868 + 4) % AArch64Env.W = 4241872)
              xs fd5 using hl2, ht2_13, hlen_m64, (by decide : (0x601000 + 8) % AArch64Env.W = 0x601008),
                hw4', (by decide : (4241872 + 4) % AArch64Env.W = 4241876)
              xs fd6 using hl2, (by decide : (4241876 / 4096 * 4096 + 2060288) % AArch64Env.W = 0x602000),
                (by decide : (4241876 + 4) % AArch64Env.W = 4241880)
              xs fd7 using hl2, ht2_24, hb3_m64, (by decide : (0x602000 + 0) % AArch64Env.W = 0x602000),
                hw5', (by decide : (4241880 + 4) % AArch64Env.W = 4241884)
              xs fd8 using hl2, ht2_13, hlen_m64, (by decide : (0x602000 + 8) % AArch64Env.W = 0x602008),
                hw6', (by decide : (4241884 + 4) % AArch64Env.W = 4241888)
              have hrej_0 : 1 * pow2 (0 * 16) % 18446744073709551616 = 1 := by decide
              have hrej_1 : (1 % pow2 (1 * 16) + 4369 * pow2 (1 * 16) + 1 / pow2 (1 * 16 + 16) * pow2 (1 * 16 + 16)) % 18446744073709551616 = 0x11110001 := by decide
              have hrej_2 : (0x11110001 % pow2 (2 * 16) + 48879 * pow2 (2 * 16) + 0x11110001 / pow2 (2 * 16 + 16) * pow2 (2 * 16 + 16)) % 18446744073709551616 = 0xbeef11110001 := by decide
              have hrej_3 : (0xbeef11110001 % pow2 (3 * 16) + 57005 * pow2 (3 * 16) + 0xbeef11110001 / pow2 (3 * 16 + 16) * pow2 (3 * 16 + 16)) % 18446744073709551616 = REJECT := by decide
              xs fr0 using hl2, hrej_0, (by decide : (4241888 + 4) % AArch64Env.W = 4241892)
              xs fr1 using hl2, hrej_1, (by decide : (4241892 + 4) % AArch64Env.W = 4241896)
              xs fr2 using hl2, hrej_2, (by decide : (4241896 + 4) % AArch64Env.W = 4241900)
              xs fr3 using hl2, hrej_3, (by decide : (4241900 + 4) % AArch64Env.W = 4241904)
              xs fr4 using hl2, hrej_0, (by decide : (4241904 + 4) % AArch64Env.W = 4241908)
              xs fr5 using hl2, hrej_1, (by decide : (4241908 + 4) % AArch64Env.W = 4241912)
              xs fr6 using hl2, hrej_2, (by decide : (4241912 + 4) % AArch64Env.W = 4241916)
              xs fr7 using hl2, hrej_3, (by decide : (4241916 + 4) % AArch64Env.W = 4241920)
              xs fr8 using hl2, hrej_0, (by decide : (4241920 + 4) % AArch64Env.W = 4241924)
              xs fr9 using hl2, hrej_1, (by decide : (4241924 + 4) % AArch64Env.W = 4241928)
              xs fr10 using hl2, hrej_2, (by decide : (4241928 + 4) % AArch64Env.W = 4241932)
              xs fr11 using hl2, hrej_3, (by decide : (4241932 + 4) % AArch64Env.W = 4241936)
              xs fb_tripled using (by decide : (4241936 + 18446744073709508336) % AArch64Env.W = L_tripled)
              have h1p : ∀ j < 8, d6[0 + j]? = some ((bytesN b1 8)[j]! % 256) :=
                fun j hj => by rw [hd6_lt _ (by omega), hd5_lt _ (by omega), hd4_lt _ (by omega), hd3_lt _ (by omega), hd2_lt _ (by omega), hd1_m j hj]
              have h1l : ∀ j < 8, d6[8 + j]? = some ((bytesN bs.length 8)[j]! % 256) :=
                fun j hj => by rw [hd6_lt _ (by omega), hd5_lt _ (by omega), hd4_lt _ (by omega), hd3_lt _ (by omega), hd2_m j hj]
              have h2p : ∀ j < 8, d6[0x1000 + j]? = some ((bytesN b2 8)[j]! % 256) :=
                fun j hj => by rw [hd6_lt _ (by omega), hd5_lt _ (by omega), hd4_lt _ (by omega), hd3_m j hj]
              have h2l : ∀ j < 8, d6[0x1008 + j]? = some ((bytesN bs.length 8)[j]! % 256) :=
                fun j hj => by rw [hd6_lt _ (by omega), hd5_lt _ (by omega), hd4_m j hj]
              have h3p : ∀ j < 8, d6[0x2000 + j]? = some ((bytesN b3 8)[j]! % 256) :=
                fun j hj => by rw [hd6_lt _ (by omega), hd5_m j hj]
              have h3l : ∀ j < 8, d6[0x2008 + j]? = some ((bytesN bs.length 8)[j]! % 256) :=
                fun j hj => hd6_m j hj
              have hlen1' : bs.length ≤ data1.length := by simp only [hd1_len]; omega
              have hlen2' : bs.length ≤ data2.length := by simp only [hlen2]; omega
              have hlen3' : bs.length ≤ data3.length := by simp only [hlen3, r3, List.length_replicate]; omega
              have hlen64 : bs.length < 2 ^ 64 := Nat.lt_trans hlen_lt48 (by decide)
              have hb1_64 : b1 < 2 ^ 64 := by unfold userMax at hb1_hi; omega
              have hb2_64 : b2 < 2 ^ 64 := by unfold userMax at hb2_hi; omega
              have hb3_64 : b3 < 2 ^ 64 := by unfold userMax at hb3_hi; omega
              have r1p := eof_rd_bss (rCode := codeRegion) (rStk := initStk.1) hwf_6 hd6_l rfl (by decide) hb1_64 h1p
              have r1l := eof_rd_bss (rCode := codeRegion) (rStk := initStk.1) hwf_6 hd6_l rfl (by decide) hlen64 h1l
              have r2p := eof_rd_bss (rCode := codeRegion) (rStk := initStk.1) hwf_6 hd6_l rfl (by decide) hb2_64 h2p
              have r2l := eof_rd_bss (rCode := codeRegion) (rStk := initStk.1) hwf_6 hd6_l rfl (by decide) hlen64 h2l
              have r3p := eof_rd_bss (rCode := codeRegion) (rStk := initStk.1) hwf_6 hd6_l rfl (by decide) hb3_64 h3p
              have r3l := eof_rd_bss (rCode := codeRegion) (rStk := initStk.1) hwf_6 hd6_l rfl (by decide) hlen64 h3l
              have ⟨d12, d13, d23, dD1, dD2, dD3, dBss, dCode⟩ := eof_mk_disj (rCode := codeRegion) (rStk := initStk.1)
                rfl spr1Arm64Bytes_len hd6_l hwf_6 hlen1' hlen2' hlen3'
              have hms : EofMemSpec [codeRegion, ⟨0x600000, d6, true⟩, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] bs b1 b2 b3 :=
                { wf := hwf_6
                  noLow := eof_noLowMem (rCode := codeRegion) (rStk := initStk.1) (d6 := d6) (data1 := data1) (data2 := data2) (data3 := data3)
                    rfl (Nat.le_trans (by decide) initStk_props.2.1) hb1_lo hb2_lo hb3_lo
                  descBy := eof_descBytes (rCode := codeRegion) (rStk := initStk.1) (bs := bs) hwf_6 hd6_l h1p h1l h2p h2l h3p h3l
                  hD1_p := r1p, hD1_l := r1l, hD2_p := r2p, hD2_l := r2l, hD3_p := r3p, hD3_l := r3l
                  hBuf1 := buf_of_region (bs := bs) hwf_6 (by rfl : ([codeRegion, ⟨0x600000, d6, true⟩, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] : Mem)[3]? = some ⟨b1, data1, true⟩) hlen1' hd1_bs
                  hBuf2 := buf_of_region (bs := bs) hwf_6 (by rfl : ([codeRegion, ⟨0x600000, d6, true⟩, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] : Mem)[4]? = some ⟨b2, data2, true⟩) hlen2' hd2
                  hBuf3 := buf_of_region (bs := bs) hwf_6 (by rfl : ([codeRegion, ⟨0x600000, d6, true⟩, initStk.1, ⟨b1, data1, true⟩, ⟨b2, data2, true⟩, ⟨b3, data3, true⟩] : Mem)[5]? = some ⟨b3, data3, true⟩) hlen3' hd3
                  hSm1 := hb1_lt48, hSm2 := hb2_lt48, hSm3 := hb3_lt48
                  disj12 := d12, disj13 := d13, disj23 := d23
                  disjD1 := dD1, disjD2 := dD2, disjD3 := dD3
                  disjBss := dBss, disjCode := dCode
                  hiB1 := hb1_lo, hiB2 := hb2_lo, hiB3 := hb3_lo }
              refine Reaches.refl (eof_mk_tripled_of_mem rfl hby rfl (by simp [hl2]) (by simpa [hk2, sC] using ho3)
                (by simp [hl2]; rfl) (by simp [hl2]; rfl) (by simp [hl2]; rfl) ?_ hms)
              intro r hr v hv
              simp only [VoterScratchRegs, List.mem_cons, List.not_mem_nil, or_false] at hr
              rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
              · simp [ht2_3] at hv; subst hv; unfold valNotAcc; decide
              · simp [ht2_4] at hv; subst hv; unfold valNotAcc; decide
              · simp [ht2_5] at hv; subst hv; unfold valNotAcc; decide
              · simp [ht2_z 16 (by simp)] at hv; subst hv; unfold valNotAcc; decide
              · simp [ht2_z 17 (by simp)] at hv; subst hv; unfold valNotAcc; decide
              · simp [ht2_z 18 (by simp)] at hv; subst hv; unfold valNotAcc; decide
              · simp [ht2_z 22 (by simp)] at hv; subst hv; unfold valNotAcc; decide
              · simp [ht2_z 23 (by simp)] at hv; subst hv; unfold valNotAcc; decide
              · simp [ht2_z 27 (by simp)] at hv; subst hv; unfold valNotAcc; decide
              · simp [ht2_z 28 (by simp)] at hv; subst hv; unfold valNotAcc; decide
              · simp [ht2_z 29 (by simp)] at hv; subst hv; unfold valNotAcc; decide

end SporosCheckArm64Bin
