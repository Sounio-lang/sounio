/-
SporosCheckArm64BinInit: verifies the initial `mmap` and `read`+`mremap` loop
of `_start` (`L_start = 0x40b888` through `L_eof = 0x40b914`), proving both:
- `init_spec`: for ANY byte input `bs` and ANY well-formed kernel oracle `o`,
  `initState bs o` either gets `.stuck`, exits `6` (`L_nomem`), or reaches a
  valid `TripledState bs`;
- `init_spec_complete`: for ANY byte input `bs` with `bs.length < 2^45`, the
  well-formed oracle `goodOracle bs` deterministically reaches `TripledState bs`.
-/
import Bootstrap.SporosCheckArm64BinInitEof

namespace SporosCheckArm64Bin

open AArch64Env AArch64Phys SporosCheckArm64Flat

theorem mremapNeedAux_lt {f len cap : Nat} (hlt : len < cap) : mremapNeedAux f len cap = 0 := by
  cases f with
  | zero => rfl
  | succ f' => simp [mremapNeedAux, hlt]

theorem mremapNeedAux_ge {f len cap : Nat} (hle : cap ≤ len) (hfit : len < cap * 2 ^ f) :
    0 < f ∧ mremapNeedAux f len cap = 1 + mremapNeedAux (f - 1) len (cap * 2) := by
  cases f with
  | zero => simp at hfit; omega
  | succ f' =>
    have hnot : ¬ len < cap := by omega
    exact ⟨Nat.succ_pos _, by simp [mremapNeedAux, hnot]⟩

/-- Good-oracle invariant during the `read`+`mremap` loop at `L_rd` / `L_rd_go`. -/
def RdOracleOk (b1 cap bsLen : Nat) (o : List Choice) : Prop :=
  ∃ f, b1 = 0x400000000000 ∧ bsLen < 2 ^ 45 ∧ cap ≤ 2 ^ 46 ∧ bsLen < cap * 2 ^ f ∧
    o = List.replicate (mremapNeedAux f bsLen cap) (.at 0x400000000000) ++
        [.at 0x800000000000, .at 0xc00000000000]

private theorem kern_count_at_oracle {k k' : Kern} {want avail n : Nat}
    (hc : k.count want avail = (n, k'))
    (hat : ∃ a rest, k.oracle = .at a :: rest) :
    k'.oracle = k.oracle := by
  obtain ⟨a, rest, hor⟩ := hat
  unfold Kern.count Kern.next at hc
  rw [hor] at hc
  cases hc
  rfl

private theorem rdOracle_head_at {b1 cap bsLen : Nat} {o : List Choice}
    (h : RdOracleOk b1 cap bsLen o) : ∃ a rest, o = .at a :: rest := by
  obtain ⟨f, _, _, _, _, hor⟩ := h
  cases hn : mremapNeedAux f bsLen cap with
  | zero => rw [hn] at hor; exact ⟨0x800000000000, [.at 0xc00000000000], by simpa using hor⟩
  | succ k => rw [hn] at hor; exact ⟨0x400000000000, _, by simpa [List.replicate_succ] using hor⟩

private theorem read_loop_spec {bs : List Nat} (hby : ∀ b ∈ bs, b < 256) :
    ∀ (rem : Nat) (b1 cap len : Nat) (data1 : List Nat) (s : St),
      s.k.stdin.length = rem → s.k.stdin = bs.drop len → len + rem = bs.length →
      OracleWf s.k.oracle → s.regs.length = 31 → s.pc = L_rd →
      s.mem = [codeRegion, bssRegion, initStk.1, ⟨b1, data1, true⟩] → s.mem.wf →
      4096 ≤ b1 → b1 + cap ≤ userMax → len ≤ cap → 0 < cap → cap % page = 0 → data1.length = cap →
      (∀ j < len, data1[j]? = some bs[j]!) →
      s.reg 12 = some cap → s.reg 13 = some len → s.reg 14 = some b1 →
      (∀ r ∈ ([16, 17, 18, 22, 23, 27, 28, 29] : List Nat), s.reg r = some 0) →
      InitOutcomeG (RdOracleOk b1 cap bs.length s.k.oracle) bs s := by
  intro rem
  induction rem using Nat.strongRecOn with
  | ind rem ih =>
    intro b1 cap len data1 s hrem hstdin hsum ho hl hpc hm hwf hb1_lo hb1_hi hlen_le hcap_pos hcap_pg hd1_len hd1_bs
      h12 h13 h14 hzero
    have hc := init_codeAt
    simp only [initTable, initTable_0, initTable_1, List.cons_append, List.nil_append,
      CodeAt_cons] at hc
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _,
      f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24,
      f25, f26, f27, f28, f29, f30, f31, f32, f33, f34, _⟩ := hc
    have hcap_lt64 : cap < 2 ^ 64 := by unfold userMax at hb1_hi; omega
    have hlen_lt64 : len < 2 ^ 64 := by omega
    have hb1_lt64 : b1 < 2 ^ 64 := by unfold userMax at hb1_hi; omega
    have hcs_eq := cond_sub (show len < pow2 64 by rw [pow2_64]; omega) (show cap < pow2 64 by rw [pow2_64]; omega)
    -- Step from `L_rd_go` (`read` syscall)
    have hstep_to_rd :
        ∀ (b1' cap' : Nat) (data1' : List Nat) (t : St),
          t.k.stdin = bs.drop len → OracleWf t.k.oracle → t.regs.length = 31 → t.pc = L_rd_go →
          t.mem = [codeRegion, bssRegion, initStk.1, ⟨b1', data1', true⟩] → t.mem.wf →
          4096 ≤ b1' → b1' + cap' ≤ userMax → len < cap' → 0 < cap' → cap' % page = 0 → data1'.length = cap' →
          (∀ j < len, data1'[j]? = some bs[j]!) →
          t.reg 12 = some cap' → t.reg 13 = some len → t.reg 14 = some b1' →
          (∀ r ∈ ([16, 17, 18, 22, 23, 27, 28, 29] : List Nat), t.reg r = some 0) →
          InitOutcomeG (RdOracleOk b1' cap' bs.length t.k.oracle) bs t := by
      intro b1' cap' data1' t ht_in ht_o ht_l ht_pc ht_m ht_wf hb1'_lo hb1'_hi hlen_lt' hcap'_pos hcap'_pg hd1'_len hd1'_bs
        ht12 ht13 ht14 htz
      have hadd_b : (b1' % 18446744073709551616 + len % 18446744073709551616) % 18446744073709551616 = b1' + len := by unfold userMax at hb1'_hi; omega
      have hsub_c : (cap' % 18446744073709551616 + 18446744073709551616 - len % 18446744073709551616) % 18446744073709551616 = cap' - len := by unfold userMax at hb1'_hi; omega
      let tR :=
        St.setReg (St.setReg (St.setReg (St.setReg (t.withPc 4241660) 0 (some 0))
          1 (some (b1' + len))) 2 (some (cap' - len))) 8 (some 63)
      have hpre_rd : Reaches t (· = tR) := by
        have ht_pc0 : t.pc = 4241644 := ht_pc
        xs f25 using ht_pc0, ht_l, (by decide : (4241644 + 4) % AArch64Env.W = 4241648)
        xs f26 using ht_l, ht14, ht13, hadd_b, (by decide : (4241648 + 4) % AArch64Env.W = 4241652)
        xs f27 using ht_l, ht12, ht13, hsub_c, (by decide : (4241652 + 4) % AArch64Env.W = 4241656)
        xs f28 using ht_l, (by decide : (4241656 + 4) % AArch64Env.W = 4241660)
        exact Reaches.refl rfl
      refine InitOutcomeG.of_reaches id hpre_rd ?_
      have htR_pc : tR.pc = 4241660 := rfl
      have htR_0 : tR.reg 0 = some 0 := by simp [tR, ht_l]
      have htR_1 : tR.reg 1 = some (b1' + len) := by simp [tR, ht_l]
      have htR_2 : tR.reg 2 = some (cap' - len) := by simp [tR, ht_l]
      have htR_8 : tR.reg 8 = some 63 := by simp [tR, ht_l]
      cases hcnt : t.k.count (cap' - len) t.k.stdin.length with
      | mk n k' =>
        have hk'_o : OracleWf k'.oracle := kern_count_oracle_wf ht_o hcnt
        have hn_le : n ≤ min (cap' - len) t.k.stdin.length := by
          unfold Kern.count at hcnt; dsimp only at hcnt
          cases hn : t.k.next with
          | mk oc k1 =>
            rw [hn] at hcnt
            cases oc with
            | none => cases hcnt; exact Nat.le_refl _
            | some c =>
              cases c with
              | count m => dsimp only at hcnt; split at hcnt <;> cases hcnt <;> omega
              | _ => cases hcnt; exact Nat.le_refl _
        have htake_len : (t.k.stdin.take n).length = n := by simp; omega
        obtain ⟨dN, hwrN, hlenN, hltN, hmidN, _⟩ :=
          Mem.wrN_at_idx (t.k.stdin.take n) t.mem 3 len ⟨b1', data1', true⟩ ht_wf
            (by rw [ht_m]; rfl) rfl (by dsimp only; omega)
        have hwfN : Mem.wf (t.mem.set 3 ⟨b1', dN, true⟩) := wf_of_shape (wrN_shape _ _ _ hwrN) ht_wf
        have hsetN : t.mem.set 3 ⟨b1', dN, true⟩ = [codeRegion, bssRegion, initStk.1, ⟨b1', dN, true⟩] := by
          rw [ht_m]; rfl
        rw [hsetN] at hwrN hwfN
        let tAfter := ((tR.withPc 4241664).setReg 0 (some n)).withKM
          { k' with stdin := t.k.stdin.drop n } [codeRegion, bssRegion, initStk.1, ⟨b1', dN, true⟩]
        have hn_lt63 : n < 2 ^ 63 := by unfold userMax at hb1'_hi; omega
        have hn_m : n % 2 ^ 64 = n := Nat.mod_eq_of_lt (by omega)
        have hs_rd : step im tR = .ok tAfter := by
          simp [step, htR_pc, f29, exec, execSvc, applySys, syscall,
            htR_0, htR_1, htR_2, htR_8, show tR.k = t.k from rfl, show tR.mem = t.mem from rfl,
            hcnt, hwrN, hn_m, AArch64Env.W, tAfter, St.withKM]
        have hkeep_ok : RdOracleOk b1' cap' bs.length t.k.oracle → RdOracleOk b1' cap' bs.length k'.oracle := by
          intro hok
          rw [kern_count_at_oracle hcnt (rdOracle_head_at hok)]
          exact hok
        refine InitOutcomeG.of_reaches hkeep_ok ⟨1, tAfter, by simp [run, hs_rd], rfl⟩ ?_
        have hcmp_n : AArch64Phys.cond (addSubOp true 64 (n % 18446744073709551616) 0).2 4 = some false ∧
            AArch64Phys.cond (addSubOp true 64 (n % 18446744073709551616) 0).2 0 = some (n == 0) :=
          cond_cmp_zero_lt63 hn_lt63
        have hpcA : tAfter.pc = 4241664 := rfl
        have hlA : tAfter.regs.length = 31 := by simp [tAfter, tR, ht_l]
        have hr0A : tAfter.reg 0 = some n := by simp [tAfter, tR, ht_l]
        have hr13A : tAfter.reg 13 = some len := by simp [tAfter, tR, ht13]
        by_cases hn0 : n = 0
        · have hin_empty : t.k.stdin = [] := by
            by_cases he : t.k.stdin.length = 0
            · exact List.eq_nil_of_length_eq_zero he
            · exfalso
              unfold Kern.count at hcnt; dsimp only at hcnt
              cases hn : t.k.next with
              | mk oc k1 =>
                rw [hn] at hcnt
                cases oc with
                | none => cases hcnt; omega
                | some c =>
                  cases c with
                  | count m => dsimp only at hcnt; split at hcnt <;> cases hcnt <;> omega
                  | _ => cases hcnt; omega
          have hlen_eq : len = bs.length := by
            have := congrArg List.length ht_in
            rw [hin_empty, List.length_nil, List.length_drop] at this; omega
          subst hlen_eq
          let tEof := (tAfter.withPc L_eof).withFl (addSubOp true 64 (0 % 18446744073709551616) 0).2
          have hreach_eof : Reaches tAfter (· = tEof) := by
            rw [hn0] at hcmp_n hr0A
            xs f30 using hpcA, hlA, hr0A, (by decide : (4241664 + 4) % AArch64Env.W = 4241668)
            xs f31 using hcmp_n.1, (by decide : (4241668 + 4) % AArch64Env.W = 4241672)
            xs f32 using (by simpa using hcmp_n.2), (by decide : (4241672 + 12) % AArch64Env.W = L_eof)
            exact Reaches.refl rfl
          have himp_eof : RdOracleOk b1' cap' bs.length k'.oracle → EofOracleOk b1' cap' bs.length k'.oracle := by
            rintro ⟨f, hb1eq, h45, h46, _, hor⟩
            rw [mremapNeedAux_lt hlen_lt'] at hor
            exact ⟨hb1eq, h46, h45, by simpa using hor⟩
          refine InitOutcomeG.of_reaches himp_eof hreach_eof ?_
          refine eof_to_tripled (data1 := dN) (s := tEof) hby hk'_o (by simp [tEof, hlA]) rfl rfl hwfN hb1'_lo hb1'_hi hlen_lt'
            (by simpa [hd1'_len] using hlenN) (fun j hj => by rw [hltN j hj]; exact hd1'_bs j hj)
            (by simp [tEof, hr13A]) (by simp [tEof, tAfter, tR, ht14]) ?_
          intro r hr
          simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
          rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
            simpa [tEof, tAfter, tR] using htz _ (by simp)
        · have hlen_add : (len % 18446744073709551616 + n % 18446744073709551616) % 18446744073709551616 = len + n := by
            unfold userMax at hb1'_hi; omega
          let tBack := ((tAfter.withPc L_rd).withFl (addSubOp true 64 (n % 18446744073709551616) 0).2).setReg 13 (some (len + n))
          have hreach_back : Reaches tAfter (· = tBack) := by
            xs f30 using hpcA, hlA, hr0A, (by decide : (4241664 + 4) % AArch64Env.W = 4241668)
            xs f31 using hcmp_n.1, (by decide : (4241668 + 4) % AArch64Env.W = 4241672)
            xs f32 using (by simpa [beq_false_of_ne hn0] using hcmp_n.2), (by decide : (4241672 + 4) % AArch64Env.W = 4241676)
            xs f33 using hlA, hr13A, hr0A, hlen_add, (by decide : (4241676 + 4) % AArch64Env.W = 4241680)
            xs f34 using (by decide : (4241680 + 18446744073709551532) % AArch64Env.W = L_rd)
            exact Reaches.refl rfl
          refine InitOutcomeG.of_reaches id hreach_back ?_
          have hdrop_eq : t.k.stdin.drop n = bs.drop (len + n) := by rw [ht_in, List.drop_drop]
          have hrem' : (t.k.stdin.drop n).length < rem := by
            rw [ht_in, List.length_drop] at hn_le
            rw [List.length_drop, ht_in, List.length_drop]; omega
          have hdN_bs : ∀ j < len + n, dN[j]? = some bs[j]! := by
            intro j hj
            by_cases hjl : j < len
            · rw [hltN j hjl]; exact hd1'_bs j hjl
            · have hk : j - len < (t.k.stdin.take n).length := by omega
              have hmid := hmidN (j - len) hk
              rw [show len + (j - len) = j by omega] at hmid
              rw [hmid]
              have hlt_bs : j < bs.length := by rw [ht_in, List.length_drop] at hn_le; omega
              have htake_j : (t.k.stdin.take n)[j - len]! = bs[j]! := by
                simp only [List.getElem!_eq_getElem?_getD, List.getElem?_take, if_pos (by omega : j - len < n),
                  ht_in, List.getElem?_drop, show len + (j - len) = j by omega]
              rw [htake_j, Nat.mod_eq_of_lt]
              simpa [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem hlt_bs] using hby _ (List.getElem_mem hlt_bs)
          refine ih (t.k.stdin.drop n).length hrem' b1' cap' (len + n) dN tBack rfl hdrop_eq
            (by rw [ht_in, List.length_drop] at hn_le; rw [List.length_drop, ht_in, List.length_drop]; omega) hk'_o (by simp [tBack, hlA]) rfl rfl hwfN
            hb1'_lo hb1'_hi (by omega) hcap'_pos hcap'_pg (by simpa [hd1'_len] using hlenN) hdN_bs
            (by simp [tBack, tAfter, tR, ht12]) (by simp [tBack, hlA]) (by simp [tBack, tAfter, tR, ht14]) ?_
          intro r hr
          simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
          rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
            simpa [tBack, tAfter, tR] using htz _ (by simp)
    -- Dispatch `len = cap` (`mremap`) vs `len < cap` (direct `read`)
    by_cases hfull : len = cap
    · subst hfull
      have hfl14 : AArch64Phys.cond (addSubOp true 64 (len % 18446744073709551616) (len % 18446744073709551616)).2 1 = some false := by
        rw [Nat.mod_eq_of_lt (by omega : len < 18446744073709551616), hcs_eq.2.1]; simp
      have hcap2 : len % 2 ^ 64 * 2 ^ 1 % 2 ^ 64 = 2 * len := by
        unfold userMax at hb1_hi; omega
      have hb1_m64 : b1 % 18446744073709551616 = b1 := Nat.mod_eq_of_lt (by omega)
      have hlen_m64 : len % 18446744073709551616 = len := Nat.mod_eq_of_lt (by omega)
      let sM :=
        St.setReg (St.setReg (St.setReg (St.setReg (St.setReg
          ((s.withPc 4241624).withFl (addSubOp true 64 len len).2)
          0 (some b1)) 1 (some len)) 2 (some (2 * len))) 3 (some 1)) 8 (some 216)
      have hpre_mr : Reaches s (· = sM) := by
        have hpc0 : s.pc = 4241596 := hpc
        xs f13 using hpc0, hl, h13, h12, (by decide : (4241596 + 4) % AArch64Env.W = 4241600)
        xs f14 using hfl14, (by decide : (4241600 + 4) % AArch64Env.W = 4241604)
        xs f15 using hl, h14, hb1_m64, (by decide : (4241604 + 4) % AArch64Env.W = 4241608)
        xs f16 using hl, h12, hlen_m64, (by decide : (4241608 + 4) % AArch64Env.W = 4241612)
        xs f17 using hl, h12, (by decide : (4241612 + 4) % AArch64Env.W = 4241616)
        xs f18 using hl, (by decide : (4241616 + 4) % AArch64Env.W = 4241620)
        xs f19 using hl, (by decide : (4241620 + 4) % AArch64Env.W = 4241624)
        exact Reaches.refl (by simp [sM, hcap2])
      refine InitOutcomeG.of_reaches id hpre_mr ?_
      have hsM_pc : sM.pc = 4241624 := rfl
      have hsM_0 : sM.reg 0 = some b1 := by simp [sM, hl]
      have hsM_1 : sM.reg 1 = some len := by simp [sM, hl]
      have hsM_2 : sM.reg 2 = some (2 * len) := by simp [sM, hl]
      have hsM_3 : sM.reg 3 = some 1 := by simp [sM, hl]
      have hsM_8 : sM.reg 8 = some 216 := by simp [sM, hl]
      have hpg_len : pages len = len := by unfold pages; unfold page at hcap_pg ⊢; omega
      have hpg_2len : pages (2 * len) = 2 * len := by unfold pages; unfold page at hcap_pg ⊢; omega
      have hfind3 : s.mem.find b1 0 = some (3, ⟨b1, data1, true⟩) :=
        Mem.find_at_idx s.mem 3 0 b1 hwf (by rw [hm]; rfl) (Nat.le_refl _) (by simp [Region.hi]; omega)
      have herase3 : s.mem.eraseIdx 3 = [codeRegion, bssRegion, initStk.1] := by rw [hm]; rfl
      cases hnM : s.k.next with
      | mk ocM kM =>
        have hoM : OracleWf kM.oracle := kern_next_oracle_wf ho hnM
        have hstep_ok : ∀ hok : RdOracleOk b1 len bs.length s.k.oracle,
            ocM = some (.at 0x400000000000) ∧ 2 * len ≤ 2 ^ 46 ∧
            RdOracleOk 0x400000000000 (2 * len) bs.length kM.oracle := by
          rintro ⟨f, _, h45, _, hfit, hor⟩
          have ⟨hfpos, hmreq⟩ := mremapNeedAux_ge (by omega : len ≤ bs.length) hfit
          rw [hmreq, Nat.add_comm 1, List.replicate_succ] at hor
          simp only [List.cons_append] at hor
          have hpair : ocM = some (.at 0x400000000000) ∧
              kM.oracle = List.replicate (mremapNeedAux (f - 1) bs.length (len * 2)) (.at 0x400000000000) ++
                [.at 0x800000000000, .at 0xc00000000000] := by
            unfold Kern.next at hnM; rw [hor] at hnM; cases hnM; exact ⟨rfl, rfl⟩
          have hpow : 2 * len * 2 ^ (f - 1) = len * 2 ^ f := by
            have hf : 2 ^ f = 2 ^ (f - 1) * 2 := by rw [← Nat.pow_succ]; exact congrArg (2 ^ ·) (by omega)
            rw [hf, Nat.mul_comm 2 len, Nat.mul_assoc, Nat.mul_comm (2 ^ (f - 1)) 2]
          exact ⟨hpair.1, by omega, f - 1, rfl, h45, by omega, by omega, by simpa [Nat.mul_comm 2 len] using hpair.2⟩
        cases hpM : place [codeRegion, bssRegion, initStk.1] ocM (2 * len) with
        | none =>
          refine .stuck ?_ ⟨1, ?_⟩
          · intro hok
            have ⟨hoc, h2len, _⟩ := hstep_ok hok
            have ⟨hcH, hbH, hsH⟩ := initMem_hi_lt_arena
            have hfree : Mem.free [codeRegion, bssRegion, initStk.1] 0x400000000000 (2 * len) = true := by
              simp only [Mem.free, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true,
                Bool.or_eq_true, decide_eq_true_eq]
              exact ⟨Or.inr hcH, Or.inr hbH, Or.inr hsH⟩
            rw [hoc] at hpM
            simp only [place, show (0x400000000000 : Nat) % page = 0 by decide,
              show (0 : Nat) < 0x400000000000 by decide,
              show 0x400000000000 + 2 * len ≤ userMax by unfold userMax; omega,
              hfree, and_self, ↓reduceIte] at hpM
            cases hpM
          · simp [run, step, hsM_pc, f20, exec, execSvc, applySys, syscall,
              hsM_0, hsM_1, hsM_2, hsM_3, hsM_8,
              show sM.k = s.k from rfl, show sM.mem = s.mem from rfl, hfind3, herase3,
              hpg_len, hpg_2len, hd1_len, hnM, hpM, MREMAP_MAYMOVE]
        | some ob1' =>
          cases ob1' with
          | none =>
            refine .nomem ?_ ?_
            · intro hok
              have ⟨hoc, _, _⟩ := hstep_ok hok
              rw [hoc] at hpM; simp [place] at hpM
            · let sM' := ((sM.withPc 4241628).setReg 0 (some (AArch64Env.neg ENOMEM % AArch64Env.W))).withKM kM s.mem
              have hs_mr : step im sM = .ok sM' := by
                simp [step, hsM_pc, f20, exec, execSvc, applySys, syscall,
                  hsM_0, hsM_1, hsM_2, hsM_3, hsM_8,
                  show sM.k = s.k from rfl, show sM.mem = s.mem from rfl, hfind3, herase3,
                  hpg_len, hpg_2len, hd1_len, hnM, hpM, show ¬ (2 * len < len) by omega,
                  MREMAP_MAYMOVE, AArch64Env.W, sM', St.withKM]
              have hto_nm : Reaches sM' (fun t => t.regs.length = 31 ∧ t.pc = L_nomem) := by
                have hpcM' : sM'.pc = 4241628 := rfl
                have hlM' : sM'.regs.length = 31 := by simp [sM', sM, hl]
                have hr0M' : sM'.reg 0 = some (AArch64Env.neg ENOMEM % AArch64Env.W) := by simp [sM', sM, hl]
                xs f21 using hpcM', hlM', hr0M', (by decide : (4241628 + 4) % AArch64Env.W = 4241632)
                xs f22 using cond_cmp_zero_enomem, (by decide : (4241632 + 18446744073709534848) % AArch64Env.W = L_nomem)
                exact Reaches.refl ⟨by simp [hlM'], rfl⟩
              obtain ⟨n', t', hrun', htl', htpc'⟩ := hto_nm
              refine ⟨1 + n' + 5, ?_⟩
              rw [run_add, run_add, show run im 1 (.ok sM) = step im sM from rfl, hs_mr, hrun', nomem_spec htl' htpc']
          | some b1' =>
            have ⟨hb1'_pg, hb1'_pos, hb1'_hi, hb1'_free⟩ := place_spec hpM
            have hb1'_lo : 4096 ≤ b1' := by unfold page at hb1'_pg; omega
            let data1' := data1 ++ List.replicate (2 * len - len) 0
            let m' : Mem := [codeRegion, bssRegion, initStk.1, ⟨b1', data1', true⟩]
            have hm'_wf : m'.wf := wf_append initMem_wf hb1'_pg (by
              change Mem.free [codeRegion, bssRegion, initStk.1] b1' data1'.length = true
              simpa [data1', hd1_len, show len + (2 * len - len) = 2 * len by omega] using hb1'_free)
            have hb1'_m : b1' % 2 ^ 64 = b1' := Nat.mod_eq_of_lt (by unfold userMax at hb1'_hi; omega)
            let sM' := ((sM.withPc 4241628).setReg 0 (some b1')).withKM kM m'
            have hs_mr : step im sM = .ok sM' := by
              simp [step, hsM_pc, f20, exec, execSvc, applySys, syscall,
                hsM_0, hsM_1, hsM_2, hsM_3, hsM_8,
                show sM.k = s.k from rfl, show sM.mem = s.mem from rfl, hfind3, herase3,
                hpg_len, hpg_2len, hd1_len, hnM, hpM, hb1'_m, show ¬ (2 * len < len) by omega,
                MREMAP_MAYMOVE, AArch64Env.W, sM', St.withKM, m', data1']
            have himp_mr : RdOracleOk b1 len bs.length s.k.oracle → RdOracleOk b1' (2 * len) bs.length kM.oracle := by
              intro hok
              have ⟨hoc, _, hnext⟩ := hstep_ok hok
              have hb1'_eq : b1' = 0x400000000000 := by
                rw [hoc] at hpM; simp only [place] at hpM; split at hpM <;> cases hpM; rfl
              rw [hb1'_eq]; exact hnext
            refine InitOutcomeG.of_reaches himp_mr ⟨1, sM', by simp [run, hs_mr], rfl⟩ ?_
            have hcmp' : AArch64Phys.cond (addSubOp true 64 (b1' % 18446744073709551616) 0).2 4 = some false :=
              (cond_cmp_zero_lt63 (by unfold userMax at hb1'_hi; omega : b1' < 2 ^ 63)).1
            have hpcM' : sM'.pc = 4241628 := rfl
            have hlM' : sM'.regs.length = 31 := by simp [sM', sM, hl]
            have hr0M' : sM'.reg 0 = some b1' := by simp [sM', sM, hl]
            have hr12M' : sM'.reg 12 = some len := by simp [sM', sM, h12]
            have hb1'_m64 : b1' % 18446744073709551616 = b1' := Nat.mod_eq_of_lt (by unfold userMax at hb1'_hi; omega)
            let sRd1 :=
              St.setReg (St.setReg ((sM'.withPc L_rd_go).withFl (addSubOp true 64 b1' 0).2)
                14 (some b1')) 12 (some (2 * len))
            have hto_rd : Reaches sM' (· = sRd1) := by
              xs f21 using hpcM', hlM', hr0M', (by decide : (4241628 + 4) % AArch64Env.W = 4241632)
              xs f22 using hcmp', (by decide : (4241632 + 4) % AArch64Env.W = 4241636)
              xs f23 using hlM', hr0M', hb1'_m64, (by decide : (4241636 + 4) % AArch64Env.W = 4241640)
              xs f24 using hlM', hr12M', (by decide : (4241640 + 4) % AArch64Env.W = L_rd_go)
              exact Reaches.refl (by simp [sRd1, hcap2])
            refine InitOutcomeG.of_reaches id hto_rd ?_
            have hkM_in : kM.stdin = bs.drop len := by
              unfold Kern.next at hnM; cases h : s.k.oracle <;> rw [h] at hnM <;> cases hnM <;> exact hstdin
            refine hstep_to_rd b1' (2 * len) data1' sRd1 hkM_in hoM (by simp [sRd1, hlM']) rfl rfl hm'_wf
              hb1'_lo hb1'_hi (by omega) (by omega) (by unfold page at hcap_pg ⊢; omega) (by simp only [data1', List.length_append, List.length_replicate, hd1_len]; omega)
              (fun j hj => by simp only [data1', List.getElem?_append_left (by omega : j < data1.length), hd1_bs j hj])
              (by simp [sRd1, hlM']) (by simp [sRd1, sM', sM, h13]) (by simp [sRd1, hlM']) ?_
            intro r hr
            simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
            rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
              simpa [sRd1, sM', sM] using hzero _ (by simp)
    · have hfl14 : AArch64Phys.cond (addSubOp true 64 (len % 18446744073709551616) (cap % 18446744073709551616)).2 1 = some true := by
        rw [Nat.mod_eq_of_lt (by omega : len < 18446744073709551616),
            Nat.mod_eq_of_lt (by omega : cap < 18446744073709551616), hcs_eq.2.1]; simp [hfull]
      let sRd2 := (s.withPc L_rd_go).withFl (addSubOp true 64 (len % 18446744073709551616) (cap % 18446744073709551616)).2
      have hto_rd : Reaches s (· = sRd2) := by
        have hpc0 : s.pc = 4241596 := hpc
        xs f13 using hpc0, hl, h13, h12, (by decide : (4241596 + 4) % AArch64Env.W = 4241600)
        xs f14 using hfl14, (by decide : (4241600 + 44) % AArch64Env.W = L_rd_go)
        exact Reaches.refl rfl
      refine InitOutcomeG.of_reaches id hto_rd ?_
      refine hstep_to_rd b1 cap data1 sRd2 hstdin ho (by simp [sRd2, hl]) rfl hm hwf hb1_lo hb1_hi (by omega)
        hcap_pos hcap_pg hd1_len hd1_bs (by simp [sRd2, h12]) (by simp [sRd2, h13]) (by simp [sRd2, h14])
        (fun r hr => by simpa [sRd2] using hzero r hr)

/-- Full `_start` theorem from ELF entry `initState bs o`, simultaneously proving
general soundness (`InitOutcome`) and completeness under `goodOracle bs` (`bs.length < 2^45`). -/
theorem init_spec_gen {bs : List Nat} {o : List Choice}
    (hby : ∀ b ∈ bs, b < 256) (ho : OracleWf o) :
    InitOutcomeG (bs.length < 2 ^ 45 ∧ o = goodOracle bs) bs (initState bs o) := by
  have hc := init_codeAt
  simp only [initTable, initTable_0, initTable_1, List.cons_append, List.nil_append,
    CodeAt_cons] at hc
  obtain ⟨f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, _⟩ := hc
  let s0 := initState bs o
  have hl0 : s0.regs.length = 31 := rfl
  let sA :=
    St.setReg (St.setReg (St.setReg (St.setReg (St.setReg (St.setReg (St.setReg
      (s0.withPc 4241572) 0 (some 0)) 1 (some 0x10000)) 2 (some 3))
      3 (some 34)) 4 (some (2 ^ 64 - 1))) 5 (some 0)) 8 (some 222)
  have hpre1 : Reaches s0 (· = sA) := by
    have hpc0 : s0.pc = 4241544 := rfl
    xs f0 using hpc0, hl0, (by decide : (4241544 + 4) % AArch64Env.W = 4241548)
    xs f1 using hl0, (by decide : 1 * pow2 (1 * 16) % 18446744073709551616 = 0x10000),
      (by decide : (4241548 + 4) % AArch64Env.W = 4241552)
    xs f2 using hl0, (by decide : (4241552 + 4) % AArch64Env.W = 4241556)
    xs f3 using hl0, (by decide : (4241556 + 4) % AArch64Env.W = 4241560)
    xs f4 using hl0, (by decide : (4241560 + 4) % AArch64Env.W = 4241564)
    xs f5 using hl0, (by decide : (4241564 + 4) % AArch64Env.W = 4241568)
    xs f6 using hl0, (by decide : (4241568 + 4) % AArch64Env.W = 4241572)
    exact Reaches.refl rfl
  refine InitOutcomeG.of_reaches id hpre1 ?_
  have hsA_pc : sA.pc = 4241572 := rfl
  have hsA_0 : sA.reg 0 = some 0 := by simp [sA, hl0]
  have hsA_1 : sA.reg 1 = some 0x10000 := by simp [sA, hl0]
  have hsA_2 : sA.reg 2 = some 3 := by simp [sA, hl0]
  have hsA_3 : sA.reg 3 = some 34 := by simp [sA, hl0]
  have hsA_4 : sA.reg 4 = some (2 ^ 64 - 1) := by simp [sA, hl0]
  have hsA_5 : sA.reg 5 = some 0 := by simp [sA, hl0]
  have hsA_8 : sA.reg 8 = some 222 := by simp [sA, hl0]
  cases hn1 : s0.k.next with
  | mk oc1 k1 =>
    have ho1 : OracleWf k1.oracle := kern_next_oracle_wf ho hn1
    have hk1_in : k1.stdin = bs := by
      unfold Kern.next at hn1; cases h : o <;> simp [s0, initState, h] at hn1 <;> rcases hn1 with ⟨_, rfl⟩ <;> rfl
    have hgood1 : (bs.length < 2 ^ 45 ∧ o = goodOracle bs) →
        oc1 = some (.at 0x400000000000) ∧ RdOracleOk 0x400000000000 0x10000 bs.length k1.oracle := by
      rintro ⟨h45, hor⟩
      have hrep : goodOracle bs = .at 0x400000000000 ::
          (List.replicate (mremapNeedAux 30 bs.length 0x10000) (.at 0x400000000000) ++
            [.at 0x800000000000, .at 0xc00000000000]) := by
        simp [goodOracle, Nat.add_comm 1, List.replicate_succ]
      unfold Kern.next at hn1
      simp only [s0, initState, hor, hrep] at hn1
      cases hn1
      exact ⟨rfl, 30, rfl, h45, by decide, by omega, rfl⟩
    cases hp1 : place initMem oc1 (pages 0x10000) with
    | none =>
      refine .stuck ?_ ⟨1, ?_⟩
      · intro hg
        have ⟨hoc, _⟩ := hgood1 hg
        have ⟨hcH, hbH, hsH⟩ := initMem_hi_lt_arena
        have hpg0 : pages 0x10000 = 0x10000 := by decide
        have hfree : initMem.free 0x400000000000 0x10000 = true := by
          simp only [initMem, Mem.free, List.all_cons, List.all_nil, Bool.and_true, Bool.and_eq_true,
            Bool.or_eq_true, decide_eq_true_eq]
          exact ⟨Or.inr hcH, Or.inr hbH, Or.inr hsH⟩
        rw [hoc, hpg0] at hp1
        simp only [place, show (0x400000000000 : Nat) % page = 0 by decide,
          show (0 : Nat) < 0x400000000000 by decide,
          show 0x400000000000 + 0x10000 ≤ userMax by decide, hfree, and_self, ↓reduceIte] at hp1
        cases hp1
      · simp [run, step, hsA_pc, f7, exec, execSvc, applySys, syscall,
          hsA_0, hsA_1, hsA_2, hsA_3, hsA_4, hsA_5, hsA_8,
          show sA.k = s0.k from rfl, show sA.mem = initMem from rfl, hn1, hp1,
          PROT_RW, MAP_PRIVATE_ANON, AArch64Env.W]
    | some ob1 =>
      cases ob1 with
      | none =>
        refine .nomem ?_ ?_
        · intro hg
          have ⟨hoc, _⟩ := hgood1 hg
          rw [hoc] at hp1; simp [place] at hp1
        · let sA' := ((sA.withPc 4241576).setReg 0 (some (AArch64Env.neg ENOMEM % AArch64Env.W))).withKM k1 initMem
          have hs1 : step im sA = .ok sA' := by
            simp [step, hsA_pc, f7, exec, execSvc, applySys, syscall,
              hsA_0, hsA_1, hsA_2, hsA_3, hsA_4, hsA_5, hsA_8,
              show sA.k = s0.k from rfl, show sA.mem = initMem from rfl, hn1, hp1,
              PROT_RW, MAP_PRIVATE_ANON, AArch64Env.W, sA', St.withKM]
          have hto_nm : Reaches sA' (fun t => t.regs.length = 31 ∧ t.pc = L_nomem) := by
            have hpcA' : sA'.pc = 4241576 := rfl
            have hlA' : sA'.regs.length = 31 := by simp [sA', sA, hl0]
            have hr0A' : sA'.reg 0 = some (AArch64Env.neg ENOMEM % AArch64Env.W) := by simp [sA', sA, hl0]
            xs f8 using hpcA', hlA', hr0A', (by decide : (4241576 + 4) % AArch64Env.W = 4241580)
            xs f9 using cond_cmp_zero_enomem, (by decide : (4241580 + 18446744073709534900) % AArch64Env.W = L_nomem)
            exact Reaches.refl ⟨by simp [hlA'], rfl⟩
          obtain ⟨n', t', hrun', htl', htpc'⟩ := hto_nm
          refine ⟨1 + n' + 5, ?_⟩
          rw [run_add, run_add, show run im 1 (.ok sA) = step im sA from rfl, hs1, hrun', nomem_spec htl' htpc']
      | some b1 =>
        have ⟨hb1_pg, hb1_pos, hb1_hi, hb1_free⟩ := place_spec hp1
        have hb1_lo : 4096 ≤ b1 := by unfold page at hb1_pg; omega
        have hpg0 : pages 0x10000 = 0x10000 := by decide
        rw [hpg0] at hb1_hi hb1_free
        let m1 : Mem := [codeRegion, bssRegion, initStk.1, ⟨b1, List.replicate 0x10000 0, true⟩]
        have hm1_wf : m1.wf := wf_append initMem_wf hb1_pg (by simp only [List.length_replicate]; exact hb1_free)
        have hb1_m : b1 % 18446744073709551616 = b1 := Nat.mod_eq_of_lt (by unfold userMax at hb1_hi; omega)
        let sA' := ((sA.withPc 4241576).setReg 0 (some b1)).withKM k1 m1
        have hm1_app : initMem ++ [⟨b1, List.replicate 0x10000 0, true⟩] = m1 := rfl
        have hp1' : place initMem oc1 0x10000 = some (some b1) := hpg0 ▸ hp1
        have hs1 : step im sA = .ok sA' := by
          simp (config := { decide := true }) only [step, hsA_pc, f7, exec, execSvc, applySys, syscall,
            hsA_0, hsA_1, hsA_2, hsA_3, hsA_4, hsA_5, hsA_8,
            show sA.k = s0.k from rfl, show sA.mem = initMem from rfl, hn1, hpg0, hp1',
            W_eq, Nat.reducePow, ↓reduceIte, bind, Option.bind, pure,
            hm1_app, hb1_m, (by decide : (4241572 + 4) % 18446744073709551616 = 4241576),
            sA', St.withKM, St.withPc, St.setReg]
        have himp1 : (bs.length < 2 ^ 45 ∧ o = goodOracle bs) → RdOracleOk b1 0x10000 bs.length k1.oracle := by
          intro hg
          have ⟨hoc, hrd⟩ := hgood1 hg
          have hb1_eq : b1 = 0x400000000000 := by
            rw [hoc] at hp1; simp only [place] at hp1; split at hp1 <;> cases hp1; rfl
          rw [hb1_eq]; exact hrd
        refine InitOutcomeG.of_reaches himp1 ⟨1, sA', by simp [run, hs1], rfl⟩ ?_
        have hcmp1 : AArch64Phys.cond (addSubOp true 64 (b1 % 18446744073709551616) 0).2 4 = some false :=
          (cond_cmp_zero_lt63 (by unfold userMax at hb1_hi; omega : b1 < 2 ^ 63)).1
        have hpcA' : sA'.pc = 4241576 := rfl
        have hlA' : sA'.regs.length = 31 := by simp [sA', sA, hl0]
        have hr0A' : sA'.reg 0 = some b1 := by simp [sA', sA, hl0]
        have hb1_m64 : b1 % 18446744073709551616 = b1 := Nat.mod_eq_of_lt (by unfold userMax at hb1_hi; omega)
        let sLoop :=
          St.setReg (St.setReg (St.setReg
            ((sA'.withPc L_rd).withFl (addSubOp true 64 b1 0).2)
            14 (some b1)) 12 (some 0x10000)) 13 (some 0)
        have hto_rd : Reaches sA' (· = sLoop) := by
          xs f8 using hpcA', hlA', hr0A', (by decide : (4241576 + 4) % AArch64Env.W = 4241580)
          xs f9 using hcmp1, (by decide : (4241580 + 4) % AArch64Env.W = 4241584)
          xs f10 using hlA', hr0A', hb1_m64, (by decide : (4241584 + 4) % AArch64Env.W = 4241588)
          xs f11 using hlA', (by decide : 1 * pow2 (1 * 16) % 18446744073709551616 = 0x10000),
            (by decide : (4241588 + 4) % AArch64Env.W = 4241592)
          xs f12 using hlA', (by decide : (4241592 + 4) % AArch64Env.W = L_rd)
          exact Reaches.refl rfl
        refine InitOutcomeG.of_reaches id hto_rd ?_
        refine read_loop_spec hby bs.length b1 0x10000 0 (List.replicate 0x10000 0) sLoop
          (by simp [sLoop, sA', hk1_in]) (by simp [sLoop, sA', hk1_in]) (by omega) ho1 (by simp [sLoop, hlA']) rfl rfl hm1_wf
          hb1_lo hb1_hi (by omega) (by decide) (by decide) List.length_replicate (fun _ h => by cases h)
          (by simp [sLoop, hlA']) (by simp [sLoop, hlA']) (by simp [sLoop, hlA']) ?_
        intro r hr
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hr
        rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
          simp [sLoop, sA', sA, s0, initState, St.reg]

theorem init_spec {bs : List Nat} {o : List Choice}
    (hby : ∀ b ∈ bs, b < 256) (ho : OracleWf o) :
    InitOutcome bs (initState bs o) :=
  (init_spec_gen hby ho).toOutcome

theorem init_spec_complete {bs : List Nat} (hby : ∀ b ∈ bs, b < 256) (hlen : bs.length < 2 ^ 45) :
    Reaches (initState bs (goodOracle bs)) (TripledState bs) :=
  (init_spec_gen hby (goodOracle_wf bs)).of_good ⟨hlen, rfl⟩

/-- Explicit `-ENOMEM` witness from ELF entry: if the first `mmap` returns `-ENOMEM`,
`_start` deterministically routes to `L_nomem` and exits `6`. -/
theorem init_nomem_exit6 (bs : List Nat) :
    Ends (initState bs [.nomem]) (· = .exit 6) := by
  have hc := init_codeAt
  simp only [initTable, initTable_0, initTable_1, List.cons_append, List.nil_append,
    CodeAt_cons] at hc
  obtain ⟨f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, _⟩ := hc
  let s0 := initState bs [.nomem]
  have hl0 : s0.regs.length = 31 := rfl
  let sA :=
    St.setReg (St.setReg (St.setReg (St.setReg (St.setReg (St.setReg (St.setReg
      (s0.withPc 4241572) 0 (some 0)) 1 (some 0x10000)) 2 (some 3))
      3 (some 34)) 4 (some (2 ^ 64 - 1))) 5 (some 0)) 8 (some 222)
  have hpre : Reaches s0 (· = sA) := by
    have hpc0 : s0.pc = 4241544 := rfl
    xs f0 using hpc0, hl0, (by decide : (4241544 + 4) % AArch64Env.W = 4241548)
    xs f1 using hl0, (by decide : 1 * pow2 (1 * 16) % 18446744073709551616 = 0x10000),
      (by decide : (4241548 + 4) % AArch64Env.W = 4241552)
    xs f2 using hl0, (by decide : (4241552 + 4) % AArch64Env.W = 4241556)
    xs f3 using hl0, (by decide : (4241556 + 4) % AArch64Env.W = 4241560)
    xs f4 using hl0, (by decide : (4241560 + 4) % AArch64Env.W = 4241564)
    xs f5 using hl0, (by decide : (4241564 + 4) % AArch64Env.W = 4241568)
    xs f6 using hl0, (by decide : (4241568 + 4) % AArch64Env.W = 4241572)
    exact Reaches.refl rfl
  refine hpre.ends fun sA_t hsA => ?_
  subst hsA
  have hsA_pc : sA.pc = 4241572 := rfl
  have hsA_0 : sA.reg 0 = some 0 := by simp [sA, hl0]
  have hsA_1 : sA.reg 1 = some 0x10000 := by simp [sA, hl0]
  have hsA_2 : sA.reg 2 = some 3 := by simp [sA, hl0]
  have hsA_3 : sA.reg 3 = some 34 := by simp [sA, hl0]
  have hsA_4 : sA.reg 4 = some (2 ^ 64 - 1) := by simp [sA, hl0]
  have hsA_5 : sA.reg 5 = some 0 := by simp [sA, hl0]
  have hsA_8 : sA.reg 8 = some 222 := by simp [sA, hl0]
  let k1 : Kern := ⟨[], [], bs, [], []⟩
  let sA' := ((sA.withPc 4241576).setReg 0 (some (AArch64Env.neg ENOMEM % AArch64Env.W))).withKM k1 initMem
  have hs1 : step im sA = .ok sA' := by
    simp [step, hsA_pc, f7, exec, execSvc, applySys, syscall, Kern.next, place,
      hsA_0, hsA_1, hsA_2, hsA_3, hsA_4, hsA_5, hsA_8,
      show sA.k = s0.k from rfl, show sA.mem = initMem from rfl, s0, initState,
      PROT_RW, MAP_PRIVATE_ANON, AArch64Env.W, sA', St.withKM, k1]
  have hto_nm : Reaches sA' (fun t => t.regs.length = 31 ∧ t.pc = L_nomem) := by
    have hpcA' : sA'.pc = 4241576 := rfl
    have hlA' : sA'.regs.length = 31 := by simp [sA', sA, hl0]
    have hr0A' : sA'.reg 0 = some (AArch64Env.neg ENOMEM % AArch64Env.W) := by simp [sA', sA, hl0]
    xs f8 using hpcA', hlA', hr0A', (by decide : (4241576 + 4) % AArch64Env.W = 4241580)
    xs f9 using cond_cmp_zero_enomem, (by decide : (4241580 + 18446744073709534900) % AArch64Env.W = L_nomem)
    exact Reaches.refl ⟨by simp [hlA'], rfl⟩
  obtain ⟨n', t', hrun', htl', htpc'⟩ := hto_nm
  refine ⟨1 + n' + 5, ?_⟩
  rw [run_add, run_add, show run im 1 (.ok sA) = step im sA from rfl, hs1, hrun', nomem_spec htl' htpc']

end SporosCheckArm64Bin
