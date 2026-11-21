/-
SporosCheckArm64BinWalk: instruction dispatch and CFG walk binary
specifications (`p1_spec`, `p2_spec`, `p3_spec`) for `spr1_check_arm64` over
`AArch64Phys` / `AArch64Env`, matching `SporosCheckArm64Flat.checkFlatWord`.
-/
import Bootstrap.SporosCheckArm64BinPass

namespace SporosCheckArm64Bin

open AArch64Env AArch64Phys SporosCheckArm64Flat

theorem rdMem_w4_at {m : Mem} {b off disp : Nat} {bs : List Nat}
    (hbuf : Buf m b bs) (hsm : b + bs.length < HMAX) (hw : off + disp + 4 ≤ bs.length) :
    rdMem m (((b + off) % 2 ^ 64 + disp) % AArch64Env.W) .w4 = some (u32 bs (off + disp)) := by
  have h1 : b + off < 2 ^ 64 := by unfold HMAX at hsm; omega
  have h2 : b + off + disp < 2 ^ 64 := by unfold HMAX at hsm; omega
  rw [ea_mod_small h1 h2, show b + off + disp = b + (off + disp) by omega]
  exact rdMem_w4_buf hbuf hw

theorem rdMem_b1_at {m : Mem} {b off : Nat} {bs : List Nat}
    (hbuf : Buf m b bs) (hsm : b + bs.length < HMAX) (hp : off < bs.length) :
    rdMem m (((b + off) % 2 ^ 64 + 0) % AArch64Env.W) .b1 = some bs[off] := by
  have h1 : b + off < 2 ^ 64 := by unfold HMAX at hsm; omega
  rw [ea_mod_small h1 (by omega), Nat.add_zero]
  exact rdMem_b1_buf hbuf hp

theorem relOff_tgt {pc tgt : Nat} (hpc : pc < 2 ^ 64) (htgt : tgt < 2 ^ 64) :
    (pc + relOff pc tgt) % AArch64Env.W = tgt := by
  unfold relOff; rw [W_eq, Nat.mod_eq_of_lt hpc]
  by_cases h : pc ≤ tgt
  · have : (tgt + 2 ^ 64 - pc) % 2 ^ 64 = tgt - pc := by omega
    rw [this]; omega
  · have : (tgt + 2 ^ 64 - pc) % 2 ^ 64 = tgt + 2 ^ 64 - pc := by omega
    rw [this]; omega

private theorem modW_add {B k d : Nat} (hB : B + 2728 < 2 ^ 64) (hk : k + d ≤ 2728) :
    (B + k + d) % AArch64Env.W = B + (k + d) := by
  rw [W_eq, Nat.add_assoc]
  exact Nat.mod_eq_of_lt (Nat.lt_of_le_of_lt (Nat.add_le_add_left hk B) hB)

private theorem modW_back {B k off tgt : Nat} (hB : B + 2728 < 2 ^ 64)
    (h1 : k + off = 2 ^ 64 + tgt) (h2 : tgt ≤ 2728) :
    (B + k + off) % AArch64Env.W = B + tgt := by
  rw [W_eq, Nat.add_assoc, h1, show B + (2 ^ 64 + tgt) = (B + tgt) + 2 ^ 64 by omega,
    Nat.add_mod_right]
  exact Nat.mod_eq_of_lt (Nat.lt_of_le_of_lt (Nat.add_le_add_left h2 B) hB)

private theorem relOff_w {B k tgt : Nat} (hB : B + 2728 < 2 ^ 64)
    (hk : k ≤ 2728) (ht : tgt ≤ 2728) :
    (B + k + relOff (B + k) (B + tgt)) % AArch64Env.W = B + tgt :=
  relOff_tgt (Nat.lt_of_le_of_lt (Nat.add_le_add_left hk B) hB)
    (Nat.lt_of_le_of_lt (Nat.add_le_add_left ht B) hB)

theorem skipCode_succ_inv {bs : List Nat} {rem p endP : Nat}
    (h : skipCode bs (rem + 1) p = some endP) :
    ∃ op ilen, bs[p]? = some op ∧ instrLen op = some ilen ∧
      p + ilen ≤ bs.length ∧ skipCode bs rem (p + ilen) = some endP := by
  simp only [skipCode, bind, Option.bind] at h
  cases hi : skipInstr bs p with
  | none => simp only [hi] at h; cases h
  | some q =>
    simp only [hi] at h
    obtain ⟨_, hqle, op, ilen, hop, hil, rfl⟩ := skipInstr_bounds hi
    exact ⟨op, ilen, hop, hil, hqle, h⟩

theorem readWordAt_fields {bs : List Nat} {off : Nat} {wm : WordSlice}
    (hby : ∀ x ∈ bs, x < 256) (h : readWordAt bs off = some wm) :
    off + 20 ≤ bs.length ∧
    u32 bs off = wm.ins ∧ u32 bs (off + 4) = wm.outs ∧
    u32 bs (off + 8) = wm.locals ∧ u32 bs (off + 12) = wm.nc ∧
    u32 bs (off + 16) = wm.na ∧ wm.codeOff = off + 20 ∧
    skipCode bs wm.nc (off + 20) = some wm.annOff ∧
    wm.annOff ≤ bs.length ∧ wm.nextOff = wm.annOff + wm.na * 12 ∧
    wm.nextOff ≤ bs.length := by
  unfold readWordAt at h
  split at h <;> try cases h
  rename_i hp20
  simp only [bind, Option.bind,
    rdU32_eq_some_u32 hby (by omega : off + 4 ≤ bs.length),
    rdU32_eq_some_u32 hby (by omega : off + 4 + 4 ≤ bs.length),
    rdU32_eq_some_u32 hby (by omega : off + 8 + 4 ≤ bs.length),
    rdU32_eq_some_u32 hby (by omega : off + 12 + 4 ≤ bs.length),
    rdU32_eq_some_u32 hby (by omega : off + 16 + 4 ≤ bs.length)] at h
  cases hsc : skipCode bs (u32 bs (off + 12)) (off + 20) with
  | none => simp only [hsc] at h; cases h
  | some annOff =>
    simp only [hsc] at h
    split at h
    · rename_i hann
      cases h
      exact ⟨hp20, rfl, rfl, rfl, rfl, rfl, rfl, hsc, (skipCode_bounds _ _ _ hsc).2 hp20, rfl, hann⟩
    · cases h

private theorem instrLen_cases {op ilen : Nat} (h : instrLen op = some ilen) :
    (op < 57 ∧ ilen = 1) ∨
    (op = 0x80 ∧ ilen = 9) ∨ (op = 0x81 ∧ ilen = 5) ∨ (op = 0x82 ∧ ilen = 5) ∨
    (op = 0x83 ∧ ilen = 1) ∨ (op = 0x84 ∧ ilen = 1) ∨ (op = 0x85 ∧ ilen = 1) ∨
    (op = 0x86 ∧ ilen = 5) ∨ (op = 0x87 ∧ ilen = 5) ∨ (op = 0x88 ∧ ilen = 5) ∨
    (op = 0x89 ∧ ilen = 5) ∨ (op = 0x8a ∧ ilen = 5) ∨ (op = 0x8b ∧ ilen = 5) ∨
    (op = 0x8c ∧ ilen = 1) := by
  unfold instrLen at h
  by_cases h57 : op < 57
  · simp [h57] at h; exact Or.inl ⟨h57, h.symm⟩
  · simp only [h57, ↓reduceIte] at h
    by_cases h0 : op = 0x80
    · subst h0; simp at h; exact Or.inr (Or.inl ⟨rfl, h.symm⟩)
    · by_cases h1 : op = 0x81
      · subst h1; simp at h; exact Or.inr (Or.inr (Or.inl ⟨rfl, h.symm⟩))
      · by_cases h2 : op = 0x82
        · subst h2; simp at h; exact Or.inr (Or.inr (Or.inr (Or.inl ⟨rfl, h.symm⟩)))
        · by_cases h3 : op = 0x83
          · subst h3; simp at h; exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨rfl, h.symm⟩))))
          · by_cases h4 : op = 0x84
            · subst h4; simp at h; exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨rfl, h.symm⟩)))))
            · by_cases h5 : op = 0x85
              · subst h5; simp at h; exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨rfl, h.symm⟩))))))
              · by_cases h6 : op = 0x86
                · subst h6; simp at h; exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨rfl, h.symm⟩)))))))
                · by_cases h7 : op = 0x87
                  · subst h7; simp at h; exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨rfl, h.symm⟩))))))))
                  · by_cases h8 : op = 0x88
                    · subst h8; simp at h; exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨rfl, h.symm⟩)))))))))
                    · by_cases h9 : op = 0x89
                      · subst h9; simp at h; exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨rfl, h.symm⟩))))))))))
                      · by_cases ha : op = 0x8a
                        · subst ha; simp at h; exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨rfl, h.symm⟩)))))))))))
                        · by_cases hb : op = 0x8b
                          · subst hb; simp at h; exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨rfl, h.symm⟩))))))))))))
                          · by_cases hc : op = 0x8c
                            · subst hc; simp at h; exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨rfl, h.symm⟩))))))))))))
                            · simp [h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, ha, hb, hc] at h

theorem hstep_dispatch_spec {B adrImm R nxt b ws nw wordOff outs locals annOff na p hd ha op ilen : Nat}
    {bs : List Nat} {m : Mem} {s : St}
    (pb : PassBlocks B adrImm R nxt) (hB : B + 2728 < 2 ^ 64)
    (hbuf : Buf m b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX)
    (hhd : hd < HMAX) (hha : ha < HMAX)
    (hw : wordOff + 20 ≤ bs.length) (hna : u32 bs (wordOff + 16) = na) (hann : annOff + na * 12 ≤ bs.length)
    (houts : u32 bs (wordOff + 4) = outs) (hlocs : u32 bs (wordOff + 8) = locals)
    (hsw : skipWords bs nw ws = some bs.length) (hws_le : ws ≤ bs.length) (hnw : nw < 2 ^ 32)
    (hop : bs[p]? = some op) (hil : instrLen op = some ilen) (hpfit : p + ilen ≤ bs.length)
    (hm : s.mem = m) (hl : s.regs.length = 31) (hpc : s.pc = B + 776)
    (h1 : s.reg 1 = some nw) (h6 : s.reg 6 = some (b + p)) (h7 : s.reg 7 = some ha)
    (h9 : s.reg 9 = some (b + annOff)) (h12 : s.reg 12 = some (b + wordOff))
    (h13 : s.reg 13 = some hd) (h24 : s.reg 24 = some (b + bs.length)) (h25 : s.reg 25 = some (b + ws)) :
    Reaches s fun s' => Frame [7, 10, 11, 13, 14, 15] s s' ∧
      let imm := if ilen == 5 then u32 bs (p + 1) else 0
      match hstepF bs nw ws outs locals annOff na hd ha op imm with
      | .reject => s'.pc = B + 2708
      | .overflow => s'.pc = B + 2688
      | .noFall => s'.pc = B + 2448 ∧ s'.regs.length = 31 ∧ s'.mem = m ∧ s'.reg 15 = some ilen
      | .next nd na' => s'.pc = B + 2348 ∧ s'.regs.length = 31 ∧ s'.mem = m ∧
          s'.reg 13 = some nd ∧ s'.reg 7 = some na' ∧ s'.reg 15 = some ilen ∧ nd < HMAX ∧ na' < HMAX := by
  have hp_lt : p < bs.length := by
    cases hgt : bs[p]? with
    | none => rw [hgt] at hop; cases hop
    | some _ => exact (List.getElem?_eq_some_iff.1 hgt).1
  have hop_eq : bs[p] = op := by
    have := List.getElem?_eq_getElem hp_lt; rw [this] at hop; cases hop; rfl
  have hop_by : op < 256 := hop_eq ▸ hby bs[p] (List.getElem_mem hp_lt)
  have hop32 : op < 2 ^ 32 := by omega
  have hcPre := pb.cDispPre; simp only [CodeAt_cons] at hcPre; obtain ⟨_, _, _, _, f776, f780, f784, _⟩ := hcPre
  have hcDE := pb.cDispEnd; simp only [CodeAt_cons] at hcDE; obtain ⟨f892, _, _, _⟩ := hcDE
  have hrd := rdMem_b1_at hbuf hsm hp_lt; rw [hop_eq] at hrd
  have hr776 : (B + 776 + 4) % AArch64Env.W = B + 780 := modW_small (by omega)
  have hr780 : (B + 780 + 4) % AArch64Env.W = B + 784 := modW_small (by omega)
  xs f776 using hl, hm, h6, hrd, hop32, hr776
  have hcs57 := cond_sub (show op < pow2 64 by rw [pow2_64]; omega) (show 57 < pow2 64 by rw [pow2_64]; decide)
  xs f780 using hl, hr780
  rcases instrLen_cases hil with ⟨h57, rfl⟩ | hhi
  · have hfl : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (57 % 2 ^ 64)).2 3 = some true := by
      rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by decide), hcs57.2.2.2.1]; simp [h57]
    have hr784 : (B + 784 + 112) % AArch64Env.W = B + 896 := modW_small (by omega)
    xs f784 using hfl, hr784
    refine (hstep_prim_spec (bs := bs) (m := m) (nw := nw) (ws := ws) (outs := outs) (locals := locals)
      (annOff := annOff) (na := na) pb hB hhd hha h57 (by simp [hm])
      (by simp [hl]) rfl (by simp [h13]) (by simp [h7]) (by simp [hl])).mono fun s' ⟨hf, hres⟩ => ?_
    exact ⟨Frame.trans_same (by frame_tac) hf, hres⟩
  · have h57 : ¬ op < 57 := by rcases hhi with ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ <;> decide
    have hfl : AArch64Phys.cond (addSubOp true 64 (op % 2 ^ 64) (57 % 2 ^ 64)).2 3 = some false := by
      rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by decide), hcs57.2.2.2.1]; simp [h57]
    xs f784 using hfl, modW_add hB (k := 784) (d := 4) (by decide)
    refine (dispatch_loop (defTgt := B + 2708) _ (B + 788) _ pb.cDisp (by simpa using f892)
      (by intro q hq; simp at hq; rcases hq with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> omega)
      (by omega) (by simp) (by omega) (by omega) hB (by simp [hl]) rfl (by simp [hl]) hop_by).trans
      fun s1 ⟨hf1, hpc1, hs1_len, _⟩ => ?_
    have hs1_mem : s1.mem = m := by rw [hf1.mem]; exact hm
    have hs1_1 : s1.reg 1 = some nw := by rw [hf1.regs 1 (by decide)]; simp [h1]
    have hs1_6 : s1.reg 6 = some (b + p) := by rw [hf1.regs 6 (by decide)]; simp [h6]
    have hs1_7 : s1.reg 7 = some ha := by rw [hf1.regs 7 (by decide)]; simp [h7]
    have hs1_9 : s1.reg 9 = some (b + annOff) := by rw [hf1.regs 9 (by decide)]; simp [h9]
    have hs1_12 : s1.reg 12 = some (b + wordOff) := by rw [hf1.regs 12 (by decide)]; simp [h12]
    have hs1_13 : s1.reg 13 = some hd := by rw [hf1.regs 13 (by decide)]; simp [h13]
    have hs1_24 : s1.reg 24 = some (b + bs.length) := by rw [hf1.regs 24 (by decide)]; simp [h24]
    have hs1_25 : s1.reg 25 = some (b + ws) := by rw [hf1.regs 25 (by decide)]; simp [h25]
    rcases hhi with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | hlc
    · -- 0x80..0x85
      refine (hstep_push_rstack_spec (nw := nw) (ws := ws) (outs := outs) (locals := locals)
        (annOff := annOff) (na := na) pb hB hbuf hby hsm hhd hha (Or.inr (Or.inl rfl))
        (Or.inl rfl) hs1_mem hs1_len (by simpa [dispatchTgt] using hpc1) hs1_6 hs1_13 hs1_7).mono fun s2 ⟨hf2, hres⟩ => ?_
      refine ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) hf2), ?_⟩
      simp at hres ⊢; cases hst : hstepF bs nw ws outs locals annOff na hd ha 128 0 <;> (rw [hst] at hres; first | exact hres.elim | exact hres)
    · refine (hstep_push_rstack_spec (nw := nw) (ws := ws) (outs := outs) (locals := locals)
        (annOff := annOff) (na := na) pb hB hbuf hby hsm hhd hha (Or.inl hpfit)
        (Or.inr (Or.inl rfl)) hs1_mem hs1_len (by simpa [dispatchTgt] using hpc1) hs1_6 hs1_13 hs1_7).mono fun s2 ⟨hf2, hres⟩ => ?_
      refine ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) hf2), ?_⟩
      have h81 : hstepF bs nw ws outs locals annOff na hd ha 0x81 (u32 bs (p + 1)) =
          hstepF bs nw ws outs locals annOff na hd ha 0x81 0 := by simp [hstepF]
      simp at hres ⊢; rw [h81]; cases hst : hstepF bs nw ws outs locals annOff na hd ha 129 0 <;> (rw [hst] at hres; first | exact hres.elim | exact hres)
    · refine (hstep_push_rstack_spec (nw := nw) (ws := ws) (outs := outs) (locals := locals)
        (annOff := annOff) (na := na) pb hB hbuf hby hsm hhd hha (Or.inl hpfit)
        (Or.inr (Or.inr (Or.inl rfl))) hs1_mem hs1_len (by simpa [dispatchTgt] using hpc1) hs1_6 hs1_13 hs1_7).mono fun s2 ⟨hf2, hres⟩ => ?_
      refine ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) hf2), ?_⟩
      simp at hres ⊢; cases hst : hstepF bs nw ws outs locals annOff na hd ha 130 (u32 bs (p + 1)) <;> (rw [hst] at hres; first | exact hres.elim | exact hres)
    · refine (hstep_push_rstack_spec (nw := nw) (ws := ws) (outs := outs) (locals := locals)
        (annOff := annOff) (na := na) pb hB hbuf hby hsm hhd hha (Or.inr (Or.inr (Or.inl rfl)))
        (Or.inr (Or.inr (Or.inr (Or.inl rfl)))) hs1_mem hs1_len (by simpa [dispatchTgt] using hpc1) hs1_6 hs1_13 hs1_7).mono fun s2 ⟨hf2, hres⟩ => ?_
      refine ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) hf2), ?_⟩
      simp at hres ⊢; cases hst : hstepF bs nw ws outs locals annOff na hd ha 131 0 <;> (rw [hst] at hres; first | exact hres.elim | exact hres)
    · refine (hstep_push_rstack_spec (nw := nw) (ws := ws) (outs := outs) (locals := locals)
        (annOff := annOff) (na := na) pb hB hbuf hby hsm hhd hha (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
        (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))) hs1_mem hs1_len (by simpa [dispatchTgt] using hpc1) hs1_6 hs1_13 hs1_7).mono fun s2 ⟨hf2, hres⟩ => ?_
      refine ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) hf2), ?_⟩
      simp at hres ⊢; cases hst : hstepF bs nw ws outs locals annOff na hd ha 132 0 <;> (rw [hst] at hres; first | exact hres.elim | exact hres)
    · refine (hstep_push_rstack_spec (nw := nw) (ws := ws) (outs := outs) (locals := locals)
        (annOff := annOff) (na := na) pb hB hbuf hby hsm hhd hha (Or.inr (Or.inr (Or.inr (Or.inr rfl))))
        (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr rfl))))) hs1_mem hs1_len (by simpa [dispatchTgt] using hpc1) hs1_6 hs1_13 hs1_7).mono fun s2 ⟨hf2, hres⟩ => ?_
      refine ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) hf2), ?_⟩
      simp at hres ⊢; cases hst : hstepF bs nw ws outs locals annOff na hd ha 133 0 <;> (rw [hst] at hres; first | exact hres.elim | exact hres)
    · -- 0x86..0x8c
      have hlc_op : op = 0x86 ∨ op = 0x87 ∨ op = 0x88 ∨ op = 0x89 ∨ op = 0x8a ∨ op = 0x8b ∨ op = 0x8c := by
        rcases hlc with ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ <;> simp
      have hlc_hp : p + 5 ≤ bs.length ∨ op = 0x8c := by
        rcases hlc with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
        all_goals first | exact Or.inl hpfit | exact Or.inr rfl
      have hlc_pc : s1.pc = if op = 0x86 then B + 1840 else if op = 0x87 then B + 1880 else if op = 0x88 then B + 1912
          else if op = 0x89 ∨ op = 0x8a then B + 2000 else if op = 0x8b then B + 2096 else B + 2324 := by
        rcases hlc with ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ | ⟨rfl, _⟩ <;>
          simpa [dispatchTgt] using hpc1
      refine (hstep_loc_ctl_spec pb hB hbuf hby hsm hhd hha hw hna hann houts hlocs hsw hws_le hnw hlc_hp hlc_op
        hs1_mem hs1_len hlc_pc hs1_1 hs1_6 hs1_7 hs1_9 hs1_12 hs1_13 hs1_24 hs1_25).mono fun s2 ⟨hf2, hres⟩ => ?_
      refine ⟨Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide)) hf2), ?_⟩
      rcases hlc with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;>
        (simp at hres ⊢; cases hst : hstepF bs nw ws outs locals annOff na hd ha _ _ <;> (rw [hst] at hres; exact hres))

theorem walk_spec {B adrImm R nxt b ws nw wordOff outs locals nc annOff na : Nat} {bs : List Nat} {m : Mem}
    (pb : PassBlocks B adrImm R nxt) (hB : B + 2728 < 2 ^ 64)
    (hbuf : Buf m b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < HMAX)
    (hw : wordOff + 20 ≤ bs.length) (houts : u32 bs (wordOff + 4) = outs)
    (hlocs : u32 bs (wordOff + 8) = locals) (hnc : u32 bs (wordOff + 12) = nc)
    (hna : u32 bs (wordOff + 16) = na) (hann : annOff + na * 12 ≤ bs.length)
    (hsw : skipWords bs nw ws = some bs.length) (hws_le : ws ≤ bs.length) (hnw : nw < 2 ^ 32) :
    ∀ (rem i p curD curA : Nat) (s : St),
      rem + i = nc → skipCode bs rem p = some annOff → p ≤ bs.length →
      s.mem = m → s.regs.length = 31 → s.pc = B + 704 →
      s.reg 1 = some nw → s.reg 6 = some (b + p) → s.reg 8 = some i →
      s.reg 9 = some (b + annOff) → s.reg 12 = some (b + wordOff) →
      s.reg 24 = some (b + bs.length) → s.reg 25 = some (b + ws) →
      ((lookup bs annOff na i = none → s.reg 13 = some curD ∧ s.reg 7 = some curA) ∧
        ∃ d0 a0, s.reg 13 = some d0 ∧ s.reg 7 = some a0 ∧ d0 < HMAX ∧ a0 < HMAX) →
      Reaches s fun s' => Frame [6, 7, 8, 10, 11, 13, 14, 15] s s' ∧
        match walk bs nw ws outs locals nc annOff na rem i p curD curA with
        | .ok => s'.pc = B + 2528 ∧ s'.regs.length = 31 ∧ s'.mem = m
        | .reject => s'.pc = B + 2708
        | .overflow => s'.pc = B + 2688
  | 0, i, p, curD, curA, s, hsum, _, _, hm, hl, hpc, _, _, h8, _, h12, _, _, _ => by
    have hi : i = nc := by omega
    subst hi
    have hnc_lt := u32_lt hby (wordOff + 12); rw [hnc] at hnc_lt
    have hcW := pb.cWalk0; simp only [CodeAt_cons] at hcW
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, f704, f708, f712, _⟩ := hcW
    have hrd := rdMem_w4_at hbuf hsm (by omega : wordOff + 12 + 4 ≤ bs.length); rw [hnc] at hrd
    have hcs := cond_sub (show i < pow2 64 by rw [pow2_64]; omega) (show i < pow2 64 by rw [pow2_64]; omega)
    xs f704 using hl, hm, h12, hrd, hnc_lt, modW_add hB (k := 704) (d := 4) (by decide)
    xs f708 using hl, h8, modW_add hB (k := 708) (d := 4) (by decide)
    have hfl : AArch64Phys.cond (addSubOp true 64 (i % 2 ^ 64) (i % 2 ^ 64)).2 0 = some true := by
      rw [Nat.mod_eq_of_lt (by omega), hcs.1]; simp
    xs f712 using hfl, modW_add hB (k := 712) (d := 1816) (by decide)
    exact Reaches.refl ⟨by frame_tac, rfl, by simp [hl], hm⟩
  | r + 1, i, p, curD, curA, s, hsum, hsc, hp_le, hm, hl, hpc, h1, h6, h8, h9, h12, h24, h25, hinv => by
    have hnc_lt := u32_lt hby (wordOff + 12); rw [hnc] at hnc_lt
    have hne_i : i ≠ nc := by omega
    have hcW := pb.cWalk0; simp only [CodeAt_cons] at hcW
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, f704, f708, f712, _⟩ := hcW
    have hrd := rdMem_w4_at hbuf hsm (by omega : wordOff + 12 + 4 ≤ bs.length); rw [hnc] at hrd
    have hcs := cond_sub (show i < pow2 64 by rw [pow2_64]; omega) (show nc < pow2 64 by rw [pow2_64]; omega)
    xs f704 using hl, hm, h12, hrd, hnc_lt, modW_add hB (k := 704) (d := 4) (by decide)
    xs f708 using hl, h8, modW_add hB (k := 708) (d := 4) (by decide)
    have hfl : AArch64Phys.cond (addSubOp true 64 (i % 2 ^ 64) (nc % 2 ^ 64)).2 0 = some false := by
      rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs.1]; simp [hne_i]
    xs f712 using hfl, modW_add hB (k := 712) (d := 4) (by decide)
    obtain ⟨op, ilen, hop, hil, hpfit, hsc_next⟩ := skipCode_succ_inv hsc
    refine (lookup_spec pb.cLuI (by omega) (Or.inl rfl) (by simpa [hm] using hbuf) hby hsm hw hna hann
      (by omega : i < 2 ^ 64) (by simp [hl]) rfl (by simp [h12]) (by simp [h9]) (by simp [h8])).trans
      fun s1 ⟨hf1, hpc1, hs1_11, hs1_len, hlu⟩ => ?_
    have hs1_mem : s1.mem = m := by rw [hf1.mem]; exact hm
    have hs1_1 : s1.reg 1 = some nw := by rw [hf1.regs 1 (by decide)]; simp [h1]
    have hs1_6 : s1.reg 6 = some (b + p) := by rw [hf1.regs 6 (by decide)]; simp [h6]
    have hs1_8 : s1.reg 8 = some i := by rw [hf1.regs 8 (by decide)]; simp [h8]
    have hs1_9 : s1.reg 9 = some (b + annOff) := by rw [hf1.regs 9 (by decide)]; simp [h9]
    have hs1_12 : s1.reg 12 = some (b + wordOff) := by rw [hf1.regs 12 (by decide)]; simp [h12]
    have hs1_24 : s1.reg 24 = some (b + bs.length) := by rw [hf1.regs 24 (by decide)]; simp [h24]
    have hs1_25 : s1.reg 25 = some (b + ws) := by rw [hf1.regs 25 (by decide)]; simp [h25]
    have hstep776 : Reaches s1 fun s2 => Frame [7, 10, 11, 13, 14, 15] s1 s2 ∧
        s2.pc = B + 776 ∧ s2.regs.length = 31 ∧ s2.mem = m ∧
        s2.reg 13 = some ((lookup bs annOff na i).getD (curD, curA)).1 ∧
        s2.reg 7 = some ((lookup bs annOff na i).getD (curD, curA)).2 ∧
        ((lookup bs annOff na i).getD (curD, curA)).1 < HMAX ∧
        ((lookup bs annOff na i).getD (curD, curA)).2 < HMAX := by
      have hcDP := pb.cDispPre; simp only [CodeAt_cons] at hcDP; obtain ⟨f760, f764, f768, f772, _⟩ := hcDP
      cases hlu_eq : lookup bs annOff na i with
      | none =>
        rw [hlu_eq] at hlu
        obtain ⟨hd_eq, ha_eq⟩ := hinv.1 hlu_eq
        obtain ⟨d0, a0, hd0, ha0, hd_lt, ha_lt⟩ := hinv.2
        rw [hd_eq] at hd0; cases hd0; rw [ha_eq] at ha0; cases ha0
        have hs1_13 : s1.reg 13 = some curD := by rw [hf1.regs 13 (by decide)]; simp [hd_eq]
        have hs1_7 : s1.reg 7 = some curA := by rw [hf1.regs 7 (by decide)]; simp [ha_eq]
        have hcs0 := cond_sub (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
          (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
        xs f760 using hs1_len, hlu, hs1_11, modW_add hB (k := 760) (d := 4) (by decide)
        have hfl0 : AArch64Phys.cond (addSubOp true 64 ((b + (annOff + na * 12)) % 2 ^ 64)
            ((b + (annOff + na * 12)) % 2 ^ 64)).2 0 = some true := by
          rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs0.1]; simp
        xs f764 using hfl0, modW_add hB (k := 764) (d := 12) (by decide)
        exact Reaches.refl ⟨by frame_tac, rfl, hs1_len, hs1_mem, hs1_13, hs1_7, hd_lt, ha_lt⟩
      | some pa =>
        rcases pa with ⟨ad, aa⟩
        rw [hlu_eq] at hlu
        obtain ⟨q, _, hq12, hs1_10, hqne, rfl, rfl⟩ := hlu
        have had := u32_lt hby (q + 4); have haa := u32_lt hby (q + 8)
        have hcs0 := cond_sub (show b + q < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
          (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
        have hrd4 := rdMem_w4_at (m := s1.mem) (by simpa [hs1_mem] using hbuf) hsm (by omega : q + 4 + 4 ≤ bs.length)
        have hrd8 := rdMem_w4_at (m := s1.mem) (by simpa [hs1_mem] using hbuf) hsm (by omega : q + 8 + 4 ≤ bs.length)
        xs f760 using hs1_len, hs1_10, hs1_11, modW_add hB (k := 760) (d := 4) (by decide)
        have hfl0 : AArch64Phys.cond (addSubOp true 64 ((b + q) % 2 ^ 64)
            ((b + (annOff + na * 12)) % 2 ^ 64)).2 0 = some false := by
          rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
            Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs0.1]
          simp [hqne]
        xs f764 using hfl0, modW_add hB (k := 764) (d := 4) (by decide)
        xs f768 using hs1_len, hs1_10, hrd4, had, modW_add hB (k := 768) (d := 4) (by decide)
        xs f772 using hs1_len, hs1_10, hrd8, haa, modW_add hB (k := 772) (d := 4) (by decide)
        exact Reaches.refl ⟨by frame_tac, rfl, by simp [hs1_len], hs1_mem, by simp [hs1_len], by simp [hs1_len],
          u32_lt_HMAX hby (q + 4), u32_lt_HMAX hby (q + 8)⟩
    refine hstep776.trans fun s2 ⟨hf2, hpc2, hs2_len, hs2_mem, hs2_13, hs2_7, hhd_lt, hha_lt⟩ => ?_
    have hs2_1 : s2.reg 1 = some nw := by rw [hf2.regs 1 (by decide)]; exact hs1_1
    have hs2_6 : s2.reg 6 = some (b + p) := by rw [hf2.regs 6 (by decide)]; exact hs1_6
    have hs2_8 : s2.reg 8 = some i := by rw [hf2.regs 8 (by decide)]; exact hs1_8
    have hs2_9 : s2.reg 9 = some (b + annOff) := by rw [hf2.regs 9 (by decide)]; exact hs1_9
    have hs2_12 : s2.reg 12 = some (b + wordOff) := by rw [hf2.regs 12 (by decide)]; exact hs1_12
    have hs2_24 : s2.reg 24 = some (b + bs.length) := by rw [hf2.regs 24 (by decide)]; exact hs1_24
    have hs2_25 : s2.reg 25 = some (b + ws) := by rw [hf2.regs 25 (by decide)]; exact hs1_25
    have himm_eq : (if ilen == 5 then rdU32 bs (p + 1) else some 0) =
        some (if ilen == 5 then u32 bs (p + 1) else 0) := by
      by_cases h5 : ilen = 5
      · subst h5; simp [rdU32_eq_some_u32 hby (by omega : p + 1 + 4 ≤ bs.length)]
      · simp [beq_false_of_ne h5]
    refine (hstep_dispatch_spec pb hB hbuf hby hsm hhd_lt hha_lt hw hna hann houts hlocs hsw hws_le hnw
      hop hil hpfit hs2_mem hs2_len hpc2 hs2_1 hs2_6 hs2_7 hs2_9 hs2_12 hs2_13 hs2_24 hs2_25).trans
      fun s3 ⟨hf3, hres⟩ => ?_
    have hf_pre : Frame [6, 7, 8, 10, 11, 13, 14, 15] s s3 :=
      Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf1 (by decide))
        (Frame.trans_same (Frame.mono hf2 (by decide)) (Frame.mono hf3 (by decide))))
    have hw_unfold : walk bs nw ws outs locals nc annOff na (r + 1) i p curD curA =
        match hstepF bs nw ws outs locals annOff na
          ((lookup bs annOff na i).getD (curD, curA)).1 ((lookup bs annOff na i).getD (curD, curA)).2
          op (if ilen == 5 then u32 bs (p + 1) else 0) with
        | .overflow => .overflow
        | .reject => .reject
        | .noFall => walk bs nw ws outs locals nc annOff na r (i + 1) (p + ilen) 0 0
        | .next nd na' =>
          if i + 1 ≥ nc then .reject
          else if (match lookup bs annOff na (i + 1) with
              | none => true | some (ad, aa) => ad == nd && aa == na') then
            walk bs nw ws outs locals nc annOff na r (i + 1) (p + ilen) nd na'
          else .reject := by
      rw [walk]; simp only [hop, hil, hpfit, decide_true, Bool.not_true, Bool.false_eq_true, ↓reduceIte, himm_eq]; rfl
    rw [hw_unfold]
    dsimp only at hres
    cases hst : hstepF bs nw ws outs locals annOff na
        ((lookup bs annOff na i).getD (curD, curA)).1 ((lookup bs annOff na i).getD (curD, curA)).2
        op (if ilen == 5 then u32 bs (p + 1) else 0) with
    | reject => rw [hst] at hres; exact Reaches.refl ⟨hf_pre, hres⟩
    | overflow => rw [hst] at hres; exact Reaches.refl ⟨hf_pre, hres⟩
    | noFall =>
      rw [hst] at hres
      obtain ⟨hpc3, hs3_len, hs3_mem, hs3_15⟩ := hres
      have hcFE := pb.cFallEnd; simp only [CodeAt_cons] at hcFE
      obtain ⟨_, _, _, _, _, _, _, _, f2440, f2444, _⟩ := hcFE
      have hcNP := pb.cNofallPre; simp only [CodeAt_cons] at hcNP; obtain ⟨f2448, _⟩ := hcNP
      have hcNE := pb.cNofallEnd; simp only [CodeAt_cons] at hcNE
      obtain ⟨f2496, f2500, f2504, f2508, f2512, f2516, f2520, f2524, _⟩ := hcNE
      have hs3_1 : s3.reg 1 = some nw := by rw [hf3.regs 1 (by decide)]; exact hs2_1
      have hs3_6 : s3.reg 6 = some (b + p) := by rw [hf3.regs 6 (by decide)]; exact hs2_6
      have hs3_8 : s3.reg 8 = some i := by rw [hf3.regs 8 (by decide)]; exact hs2_8
      have hs3_9 : s3.reg 9 = some (b + annOff) := by rw [hf3.regs 9 (by decide)]; exact hs2_9
      have hs3_12 : s3.reg 12 = some (b + wordOff) := by rw [hf3.regs 12 (by decide)]; exact hs2_12
      have hs3_24 : s3.reg 24 = some (b + bs.length) := by rw [hf3.regs 24 (by decide)]; exact hs2_24
      have hs3_25 : s3.reg 25 = some (b + ws) := by rw [hf3.regs 25 (by decide)]; exact hs2_25
      have hi1 : (i % 2 ^ 64 + 1) % 2 ^ 64 = i + 1 := by omega
      xs f2448 using hs3_len, hs3_8, hi1, modW_add hB (k := 2448) (d := 4) (by decide)
      refine (lookup_spec pb.cLuNofall (by omega) (Or.inl rfl) (by simpa [hs3_mem] using hbuf) hby hsm hw hna hann
        (by omega : i + 1 < 2 ^ 64) (by simp [hs3_len]) rfl (by simp [hs3_12]) (by simp [hs3_9]) (by simp [hs3_len])).trans
        fun s4 ⟨hf4, hpc4, hs4_11, hs4_len, hlu4⟩ => ?_
      have hs4_mem : s4.mem = m := by rw [hf4.mem]; exact hs3_mem
      have hs4_1 : s4.reg 1 = some nw := by rw [hf4.regs 1 (by decide)]; simp [hs3_1]
      have hs4_6 : s4.reg 6 = some (b + p) := by rw [hf4.regs 6 (by decide)]; simp [hs3_6]
      have hs4_8 : s4.reg 8 = some (i + 1) := by rw [hf4.regs 8 (by decide)]; simp [hs3_len]
      have hs4_9 : s4.reg 9 = some (b + annOff) := by rw [hf4.regs 9 (by decide)]; simp [hs3_9]
      have hs4_12 : s4.reg 12 = some (b + wordOff) := by rw [hf4.regs 12 (by decide)]; simp [hs3_12]
      have hs4_15 : s4.reg 15 = some ilen := by rw [hf4.regs 15 (by decide)]; simp [hs3_15]
      have hs4_24 : s4.reg 24 = some (b + bs.length) := by rw [hf4.regs 24 (by decide)]; simp [hs3_24]
      have hs4_25 : s4.reg 25 = some (b + ws) := by rw [hf4.regs 25 (by decide)]; simp [hs3_25]
      have hbp_nxt : ((b + p) % 2 ^ 64 + ilen % 2 ^ 64) % 2 ^ 64 = b + (p + ilen) := by
        unfold HMAX at hsm; omega
      cases hlu4_eq : lookup bs annOff na (i + 1) with
      | none =>
        rw [hlu4_eq] at hlu4
        have hcs0 := cond_sub (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
          (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
        xs f2496 using hs4_len, hlu4, hs4_11, modW_add hB (k := 2496) (d := 4) (by decide)
        have hfl0 : AArch64Phys.cond (addSubOp true 64 ((b + (annOff + na * 12)) % 2 ^ 64)
            ((b + (annOff + na * 12)) % 2 ^ 64)).2 0 = some true := by
          rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs0.1]; simp
        xs f2500 using hfl0, modW_add hB (k := 2500) (d := 16) (by decide)
        xs f2516 using hs4_len, modW_add hB (k := 2516) (d := 4) (by decide)
        xs f2520 using hs4_len, modW_add hB (k := 2520) (d := 4) (by decide)
        xs f2524 using modW_back hB (k := 2524) (off := 18446744073709551532) (tgt := 2440) (by decide) (by decide)
        xs f2440 using hs4_len, hs4_6, hs4_15, hbp_nxt, modW_add hB (k := 2440) (d := 4) (by decide)
        xs f2444 using modW_back hB (k := 2444) (off := 18446744073709549876) (tgt := 704) (by decide) (by decide)
        refine (walk_spec pb hB hbuf hby hsm hw houts hlocs hnc hna hann hsw hws_le hnw r (i + 1) (p + ilen) 0 0 _
          (by omega) hsc_next hpfit (by simp [hs4_mem]) (by simp [hs4_len]) rfl (by simp [hs4_1]) (by simp [hs4_len])
          (by simp [hs4_8]) (by simp [hs4_9]) (by simp [hs4_12]) (by simp [hs4_24]) (by simp [hs4_25])
          ⟨fun _ => ⟨by simp [hs4_len], by simp [hs4_len]⟩, 0, 0, by simp [hs4_len], by simp [hs4_len],
            by decide, by decide⟩).mono fun s5 ⟨hf5, hres5⟩ => ?_
        exact ⟨Frame.trans_same hf_pre (Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf4 (by decide))
          (Frame.trans_same (by frame_tac) hf5))), hres5⟩
      | some pa =>
        rcases pa with ⟨ad, aa⟩
        rw [hlu4_eq] at hlu4
        obtain ⟨q, _, hq12, hs4_10, hqne, rfl, rfl⟩ := hlu4
        have had := u32_lt hby (q + 4); have haa := u32_lt hby (q + 8)
        have hcs0 := cond_sub (show b + q < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
          (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
        have hrd4 := rdMem_w4_at (m := s4.mem) (by simpa [hs4_mem] using hbuf) hsm (by omega : q + 4 + 4 ≤ bs.length)
        have hrd8 := rdMem_w4_at (m := s4.mem) (by simpa [hs4_mem] using hbuf) hsm (by omega : q + 8 + 4 ≤ bs.length)
        xs f2496 using hs4_len, hs4_10, hs4_11, modW_add hB (k := 2496) (d := 4) (by decide)
        have hfl0 : AArch64Phys.cond (addSubOp true 64 ((b + q) % 2 ^ 64)
            ((b + (annOff + na * 12)) % 2 ^ 64)).2 0 = some false := by
          rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
            Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs0.1]
          simp [hqne]
        xs f2500 using hfl0, modW_add hB (k := 2500) (d := 4) (by decide)
        xs f2504 using hs4_len, hs4_10, hrd4, had, modW_add hB (k := 2504) (d := 4) (by decide)
        xs f2508 using hs4_len, hs4_10, hrd8, haa, modW_add hB (k := 2508) (d := 4) (by decide)
        xs f2512 using modW_back hB (k := 2512) (off := 18446744073709551544) (tgt := 2440) (by decide) (by decide)
        xs f2440 using hs4_len, hs4_6, hs4_15, hbp_nxt, modW_add hB (k := 2440) (d := 4) (by decide)
        xs f2444 using modW_back hB (k := 2444) (off := 18446744073709549876) (tgt := 704) (by decide) (by decide)
        refine (walk_spec pb hB hbuf hby hsm hw houts hlocs hnc hna hann hsw hws_le hnw r (i + 1) (p + ilen) 0 0 _
          (by omega) hsc_next hpfit (by simp [hs4_mem]) (by simp [hs4_len]) rfl (by simp [hs4_1]) (by simp [hs4_len])
          (by simp [hs4_8]) (by simp [hs4_9]) (by simp [hs4_12]) (by simp [hs4_24]) (by simp [hs4_25])
          ⟨(fun hnone => by rw [hlu4_eq] at hnone; cases hnone),
            u32 bs (q + 4), u32 bs (q + 8), by simp [hs4_len], by simp [hs4_len],
            u32_lt_HMAX hby (q + 4), u32_lt_HMAX hby (q + 8)⟩).mono fun s5 ⟨hf5, hres5⟩ => ?_
        exact ⟨Frame.trans_same hf_pre (Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf4 (by decide))
          (Frame.trans_same (by frame_tac) hf5))), hres5⟩
    | next nd na' =>
      rw [hst] at hres
      obtain ⟨hpc3, hs3_len, hs3_mem, hs3_13, hs3_7, hs3_15, hnd_lt, hna'_lt⟩ := hres
      have hcFP := pb.cFallPre; simp only [CodeAt_cons] at hcFP; obtain ⟨f2348, f2352, f2356, f2360, _⟩ := hcFP
      have hcFE := pb.cFallEnd; simp only [CodeAt_cons] at hcFE
      obtain ⟨f2408, f2412, f2416, f2420, f2424, f2428, f2432, f2436, f2440, f2444, _⟩ := hcFE
      have hs3_1 : s3.reg 1 = some nw := by rw [hf3.regs 1 (by decide)]; exact hs2_1
      have hs3_6 : s3.reg 6 = some (b + p) := by rw [hf3.regs 6 (by decide)]; exact hs2_6
      have hs3_8 : s3.reg 8 = some i := by rw [hf3.regs 8 (by decide)]; exact hs2_8
      have hs3_9 : s3.reg 9 = some (b + annOff) := by rw [hf3.regs 9 (by decide)]; exact hs2_9
      have hs3_12 : s3.reg 12 = some (b + wordOff) := by rw [hf3.regs 12 (by decide)]; exact hs2_12
      have hs3_24 : s3.reg 24 = some (b + bs.length) := by rw [hf3.regs 24 (by decide)]; exact hs2_24
      have hs3_25 : s3.reg 25 = some (b + ws) := by rw [hf3.regs 25 (by decide)]; exact hs2_25
      have hi1 : (i % 2 ^ 64 + 1) % 2 ^ 64 = i + 1 := by omega
      have hrd_nc := rdMem_w4_at (m := s3.mem) (by simpa [hs3_mem] using hbuf) hsm (by omega : wordOff + 12 + 4 ≤ bs.length)
      rw [hnc] at hrd_nc
      xs f2348 using hs3_len, hs3_8, hi1, modW_add hB (k := 2348) (d := 4) (by decide)
      xs f2352 using hs3_len, hs3_12, hrd_nc, hnc_lt, modW_add hB (k := 2352) (d := 4) (by decide)
      have hcs_ge := cond_sub (show i + 1 < pow2 64 by rw [pow2_64]; omega) (show nc < pow2 64 by rw [pow2_64]; omega)
      xs f2356 using hs3_len, modW_add hB (k := 2356) (d := 4) (by decide)
      by_cases hge : nc ≤ i + 1
      · have hfl : AArch64Phys.cond (addSubOp true 64 ((i + 1) % 2 ^ 64) (nc % 2 ^ 64)).2 2 = some true := by
          rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs_ge.2.2.1]; simp [hge]
        xs f2360 using hfl, relOff_w hB (k := 2360) (tgt := 2708) (by decide) (by decide)
        simp only [hge, ↓reduceIte]
        exact Reaches.refl ⟨Frame.trans_same hf_pre (by frame_tac), rfl⟩
      · have hfl : AArch64Phys.cond (addSubOp true 64 ((i + 1) % 2 ^ 64) (nc % 2 ^ 64)).2 2 = some false := by
          rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega), hcs_ge.2.2.1]; simp [hge]
        xs f2360 using hfl, modW_add hB (k := 2360) (d := 4) (by decide)
        simp only [hge, ↓reduceIte]
        refine (lookup_spec pb.cLuFall (by omega) (Or.inl rfl) (by simpa [hs3_mem] using hbuf) hby hsm hw hna hann
          (by omega : i + 1 < 2 ^ 64) (by simp [hs3_len]) rfl (by simp [hs3_12]) (by simp [hs3_9]) (by simp [hs3_len])).trans
          fun s4 ⟨hf4, hpc4, hs4_11, hs4_len, hlu4⟩ => ?_
        have hs4_mem : s4.mem = m := by rw [hf4.mem]; exact hs3_mem
        have hs4_1 : s4.reg 1 = some nw := by rw [hf4.regs 1 (by decide)]; simp [hs3_1]
        have hs4_6 : s4.reg 6 = some (b + p) := by rw [hf4.regs 6 (by decide)]; simp [hs3_6]
        have hs4_7 : s4.reg 7 = some na' := by rw [hf4.regs 7 (by decide)]; simp [hs3_7]
        have hs4_8 : s4.reg 8 = some (i + 1) := by rw [hf4.regs 8 (by decide)]; simp [hs3_len]
        have hs4_9 : s4.reg 9 = some (b + annOff) := by rw [hf4.regs 9 (by decide)]; simp [hs3_9]
        have hs4_12 : s4.reg 12 = some (b + wordOff) := by rw [hf4.regs 12 (by decide)]; simp [hs3_12]
        have hs4_13 : s4.reg 13 = some nd := by rw [hf4.regs 13 (by decide)]; simp [hs3_13]
        have hs4_15 : s4.reg 15 = some ilen := by rw [hf4.regs 15 (by decide)]; simp [hs3_15]
        have hs4_24 : s4.reg 24 = some (b + bs.length) := by rw [hf4.regs 24 (by decide)]; simp [hs3_24]
        have hs4_25 : s4.reg 25 = some (b + ws) := by rw [hf4.regs 25 (by decide)]; simp [hs3_25]
        have hbp_nxt : ((b + p) % 2 ^ 64 + ilen % 2 ^ 64) % 2 ^ 64 = b + (p + ilen) := by
          unfold HMAX at hsm; omega
        cases hlu4_eq : lookup bs annOff na (i + 1) with
        | none =>
          rw [hlu4_eq] at hlu4
          have hcs0 := cond_sub (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
            (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
          xs f2408 using hs4_len, hlu4, hs4_11, modW_add hB (k := 2408) (d := 4) (by decide)
          have hfl0 : AArch64Phys.cond (addSubOp true 64 ((b + (annOff + na * 12)) % 2 ^ 64)
              ((b + (annOff + na * 12)) % 2 ^ 64)).2 0 = some true := by
            rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs0.1]; simp
          xs f2412 using hfl0, modW_add hB (k := 2412) (d := 28) (by decide)
          xs f2440 using hs4_len, hs4_6, hs4_15, hbp_nxt, modW_add hB (k := 2440) (d := 4) (by decide)
          xs f2444 using modW_back hB (k := 2444) (off := 18446744073709549876) (tgt := 704) (by decide) (by decide)
          refine (walk_spec pb hB hbuf hby hsm hw houts hlocs hnc hna hann hsw hws_le hnw r (i + 1) (p + ilen) nd na' _
            (by omega) hsc_next hpfit (by simp [hs4_mem]) (by simp [hs4_len]) rfl (by simp [hs4_1]) (by simp [hs4_len])
            (by simp [hs4_8]) (by simp [hs4_9]) (by simp [hs4_12]) (by simp [hs4_24]) (by simp [hs4_25])
            ⟨fun _ => ⟨by simp [hs4_13], by simp [hs4_7]⟩, nd, na', by simp [hs4_13], by simp [hs4_7],
              hnd_lt, hna'_lt⟩).mono fun s5 ⟨hf5, hres5⟩ => ?_
          exact ⟨Frame.trans_same hf_pre (Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf4 (by decide))
            (Frame.trans_same (by frame_tac) hf5))), hres5⟩
        | some pa =>
          rcases pa with ⟨ad, aa⟩
          rw [hlu4_eq] at hlu4
          obtain ⟨q, _, hq12, hs4_10, hqne, rfl, rfl⟩ := hlu4
          have had := u32_lt hby (q + 4); have haa := u32_lt hby (q + 8)
          have hcs0 := cond_sub (show b + q < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
            (show b + (annOff + na * 12) < pow2 64 by rw [pow2_64]; unfold HMAX at hsm; omega)
          have hrd4 := rdMem_w4_at (m := s4.mem) (by simpa [hs4_mem] using hbuf) hsm (by omega : q + 4 + 4 ≤ bs.length)
          xs f2408 using hs4_len, hs4_10, hs4_11, modW_add hB (k := 2408) (d := 4) (by decide)
          have hfl0 : AArch64Phys.cond (addSubOp true 64 ((b + q) % 2 ^ 64)
              ((b + (annOff + na * 12)) % 2 ^ 64)).2 0 = some false := by
            rw [Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega),
              Nat.mod_eq_of_lt (by unfold HMAX at hsm; omega), hcs0.1]
            simp [hqne]
          xs f2412 using hfl0, modW_add hB (k := 2412) (d := 4) (by decide)
          xs f2416 using hs4_len, hs4_10, hrd4, had, modW_add hB (k := 2416) (d := 4) (by decide)
          have hcs1 := cond_sub (show u32 bs (q + 4) < pow2 64 by rw [pow2_64]; omega)
            (show nd < pow2 64 by rw [pow2_64]; unfold HMAX at hnd_lt; omega)
          xs f2420 using hs4_len, hs4_13, modW_add hB (k := 2420) (d := 4) (by decide)
          by_cases heqd : u32 bs (q + 4) = nd
          · have hfl1 : AArch64Phys.cond (addSubOp true 64 (u32 bs (q + 4) % 2 ^ 64) (nd % 2 ^ 64)).2 1 = some false := by
              rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by unfold HMAX at hnd_lt; omega), hcs1.2.1]; simp [heqd]
            have hrd8 := rdMem_w4_at (m := s4.mem) (by simpa [hs4_mem] using hbuf) hsm (by omega : q + 8 + 4 ≤ bs.length)
            xs f2424 using hfl1, modW_add hB (k := 2424) (d := 4) (by decide)
            xs f2428 using hs4_len, hs4_10, hrd8, haa, modW_add hB (k := 2428) (d := 4) (by decide)
            have hcs2 := cond_sub (show u32 bs (q + 8) < pow2 64 by rw [pow2_64]; omega)
              (show na' < pow2 64 by rw [pow2_64]; unfold HMAX at hna'_lt; omega)
            xs f2432 using hs4_len, hs4_7, modW_add hB (k := 2432) (d := 4) (by decide)
            by_cases heqa : u32 bs (q + 8) = na'
            · have hfl2 : AArch64Phys.cond (addSubOp true 64 (u32 bs (q + 8) % 2 ^ 64) (na' % 2 ^ 64)).2 1 = some false := by
                rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by unfold HMAX at hna'_lt; omega), hcs2.2.1]; simp [heqa]
              xs f2436 using hfl2, modW_add hB (k := 2436) (d := 4) (by decide)
              xs f2440 using hs4_len, hs4_6, hs4_15, hbp_nxt, modW_add hB (k := 2440) (d := 4) (by decide)
              xs f2444 using modW_back hB (k := 2444) (off := 18446744073709549876) (tgt := 704) (by decide) (by decide)
              simp only [heqd, heqa, beq_self_eq_true, Bool.and_self, ↓reduceIte]
              refine (walk_spec pb hB hbuf hby hsm hw houts hlocs hnc hna hann hsw hws_le hnw r (i + 1) (p + ilen) nd na' _
                (by omega) hsc_next hpfit (by simp [hs4_mem]) (by simp [hs4_len]) rfl (by simp [hs4_1]) (by simp [hs4_len])
                (by simp [hs4_8]) (by simp [hs4_9]) (by simp [hs4_12]) (by simp [hs4_24]) (by simp [hs4_25])
                ⟨fun _ => ⟨by simp [hs4_13], by simp [hs4_7]⟩, nd, na', by simp [hs4_13], by simp [hs4_7],
                  hnd_lt, hna'_lt⟩).mono fun s5 ⟨hf5, hres5⟩ => ?_
              exact ⟨Frame.trans_same hf_pre (Frame.trans_same (by frame_tac) (Frame.trans_same (Frame.mono hf4 (by decide))
                (Frame.trans_same (by frame_tac) hf5))), hres5⟩
            · have hfl2 : AArch64Phys.cond (addSubOp true 64 (u32 bs (q + 8) % 2 ^ 64) (na' % 2 ^ 64)).2 1 = some true := by
                rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by unfold HMAX at hna'_lt; omega), hcs2.2.1]; simp [heqa]
              xs f2436 using hfl2, relOff_w hB (k := 2436) (tgt := 2708) (by decide) (by decide)
              simp only [heqd, beq_self_eq_true, beq_false_of_ne heqa, Bool.and_false, Bool.false_eq_true, ↓reduceIte]
              exact Reaches.refl ⟨Frame.trans_same hf_pre (Frame.trans_same (by frame_tac)
                (Frame.trans_same (Frame.mono hf4 (by decide)) (by frame_tac))), rfl⟩
          · have hfl1 : AArch64Phys.cond (addSubOp true 64 (u32 bs (q + 4) % 2 ^ 64) (nd % 2 ^ 64)).2 1 = some true := by
              rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by unfold HMAX at hnd_lt; omega), hcs1.2.1]; simp [heqd]
            xs f2424 using hfl1, relOff_w hB (k := 2424) (tgt := 2708) (by decide) (by decide)
            simp only [beq_false_of_ne heqd, Bool.false_and, Bool.false_eq_true, ↓reduceIte]
            exact Reaches.refl ⟨Frame.trans_same hf_pre (Frame.trans_same (by frame_tac)
              (Frame.trans_same (Frame.mono hf4 (by decide)) (by frame_tac))), rfl⟩


end SporosCheckArm64Bin
