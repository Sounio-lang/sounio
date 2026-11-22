/-
SporosCheckBinLoops: correctness proofs for `skipdataT`, `chkT`, `effT`, and `lookupT`.
-/
import Bootstrap.SporosCheckBinSpec

namespace SporosCheckBin

open X86Env X86Int

/-! ### skipData -/

theorem skipdata_spec {L rej : Nat} (hc : CodeAt (skipdataT L rej)) {b : Nat} {bs : List Nat} :
    ∀ (n pos : Nat) (s : St), Env s b bs → s.rip = L → s.reg 10 = some (b + pos) →
      s.reg 15 = some n → pos ≤ bs.length → n < 2 ^ 32 →
      Reaches s (fun s' => Frame [10, 11, 14, 15] s s' ∧
        match skipDataA bs n pos with
        | some pos' => s'.rip = L + 51 ∧ s'.reg 10 = some (b + pos')
        | none => s'.rip = rej) := by
  simp only [skipdataT, CodeAt_cons, CodeAt_nil, and_true] at hc
  obtain ⟨f0, f3, f5, f8, f12, f15, f21, f24, f27, f30, f33, f36, f42, f45, f49⟩ := hc
  intro n
  induction n with
  | zero =>
    intro pos s he hr h10 h15 hp hn
    have h5 := he.rbp
    have hl := he.len
    xs f0 using h15
    xs f3 using cond_test_e
    refine Reaches.refl ⟨frame_regs (fun j _ => rfl) rfl rfl rfl, ?_⟩
    exact ⟨rfl, h10⟩
  | succ n ih =>
    intro pos s he hr h10 h15 hp hn
    have h5 := he.rbp
    have hl := he.len
    have hsm := he.small
    xs f0 using h15
    xs f3 using cond_test_e, Nat.add_one_ne_zero
    xs f5 using h10, hl
    xs f8 using hl, add_small (show b + pos + 4 < 2 ^ 64 by omega)
    xs f12 using hl, h5
    have hc7 := (cond_sub (show b + pos + 4 < W by rw [W_eq]; omega)
      (show b + bs.length < W by rw [W_eq]; omega)).2.2.2.2.2
    by_cases hlt : bs.length < pos + 4
    · xs f15 using hc7, show b + bs.length < b + pos + 4 by omega
      exact Reaches.refl ⟨by frame_tac, by simp [skipDataA, hlt]⟩
    · have hp4 : pos + 4 ≤ bs.length := by omega
      have hu := u32_lt he.bytes pos
      xs f15 using hc7, show ¬ (b + bs.length < b + pos + 4) by omega
      xs f21 using hl, h10, ea_small (show b + pos + 0 < 2 ^ 64 by omega), Nat.add_zero, rd32_buf he.buf hp4
      xs f24 using hl
      xs f27 using hl, h5
      xs f30 using hl, sub_small (show b + bs.length < 2 ^ 64 by omega) (show b + pos + 4 ≤ b + bs.length by omega)
      have hc2 := (cond_sub (show b + bs.length - (b + pos + 4) < W by rw [W_eq]; omega)
        (show u32 bs pos < W by rw [W_eq]; omega)).1
      xs f33 using hl
      by_cases hlt2 : bs.length - (pos + 4) < u32 bs pos
      · xs f36 using hc2, show b + bs.length - (b + pos + 4) < u32 bs pos by omega
        exact Reaches.refl ⟨by frame_tac, by simp [skipDataA, hlt, hlt2]⟩
      · xs f36 using hc2, show ¬ (b + bs.length - (b + pos + 4) < u32 bs pos) by omega
        xs f42 using hl, add_small (show b + pos + 4 + u32 bs pos < 2 ^ 64 by omega)
        xs f45 using hl, h15, sub_small (show n + 1 < 2 ^ 64 by omega) (show 1 ≤ n + 1 by omega),
          Nat.add_sub_cancel
        xs f49
        have hstep : skipDataA bs (n + 1) pos = skipDataA bs n (pos + 4 + u32 bs pos) := by
          simp [skipDataA, hlt, hlt2]
        rw [hstep]
        refine reach_frame (by frame_tac) (ih (pos + 4 + u32 bs pos) _ ?_ rfl ?_ ?_ (by omega) (by omega))
        · exact he.frame (S := [10, 11, 14, 15]) (by frame_tac) (by decide)
        · rsimp using hl; omega
        · rsimp using hl

/-! ### Height check (`chkT`) -/

theorem chk_spec {L r ovf : Nat} (hc : CodeAt (chkT L r ovf))
    {s : St} {v : Nat} (hl : s.regs.length = 16) (hrip : s.rip = L) (hr : s.reg r = some v)
    (hv : v < 2 ^ 64) :
    Reaches s (fun s' => Frame [10] s s' ∧
      if HMAX ≤ v then s'.rip = ovf else s'.rip = L + 17) := by
  simp only [chkT, CodeAt_cons, CodeAt_nil, and_true] at hc
  obtain ⟨f0, f3, f7, f10, f12⟩ := hc
  xs f0 using hr
  xs f3 using hl
  xs f7 using hl
  by_cases hov : HMAX ≤ v
  · have hnz : v / 2 ^ 48 ≠ 0 := by unfold HMAX at hov; omega
    xs f10 using cond_test_e, hnz
    xs f12
    exact Reaches.refl ⟨by frame_tac, by simp [hov]⟩
  · have hz : v / 2 ^ 48 = 0 := by unfold HMAX at hov; exact Nat.div_eq_of_lt (by omega)
    xs f10 using cond_test_e, hz
    exact Reaches.refl ⟨by frame_tac, by simp [hov]⟩

/-! ### Effect binary search tree (`effT`) -/

def EffTree.small : EffTree → Bool
  | .leaf i o => i < 2 ^ 64 && o < 2 ^ 64
  | .node k l r => k < 2 ^ 64 && l.small && r.small

theorem primEffTree_small : primEffTree.small = true := by decide +kernel

theorem effTree_spec {L tgt : Nat} : ∀ (t : EffTree) (off : Nat),
    CodeAt (EffTree.emit L tgt off t).2 → t.small = true →
    ∀ (s : St) (op : Nat), s.regs.length = 16 → s.rip = L + off → s.reg 15 = some op → op < 2 ^ 64 →
      Reaches s (fun s' => Frame [10, 11] s s' ∧ s'.rip = tgt ∧
        s'.reg 10 = some (t.eval op).1 ∧ s'.reg 11 = some (t.eval op).2)
  | .leaf i o, off, hc, hs, s, op, hl, hrip, _, _ => by
    simp only [EffTree.emit, CodeAt_cons, CodeAt_nil, and_true] at hc
    obtain ⟨f0, f6, f12⟩ := hc
    simp only [EffTree.small, Bool.and_eq_true, decide_eq_true_eq] at hs
    xs f0
    xs f6
    xs f12
    refine Reaches.refl ⟨by frame_tac, rfl, ?_, ?_⟩ <;>
      simp (config := { decide := true }) only [EffTree.eval, reg_mk_set_eq, reg_mk_set_ne,
        List.length_set, hl, Nat.mod_eq_of_lt hs.1, Nat.mod_eq_of_lt hs.2]
  | .node k l r, off, hc, hs, s, op, hl, hrip, h15, hop => by
    simp only [EffTree.emit] at hc
    rw [CodeAt_append, CodeAt_append, CodeAt_cons, CodeAt_cons, CodeAt_nil] at hc
    obtain ⟨⟨⟨f0, f4, _⟩, hcL⟩, hcR⟩ := hc
    simp only [EffTree.small, Bool.and_eq_true, decide_eq_true_eq] at hs
    obtain ⟨⟨hk, hsl⟩, hsr⟩ := hs
    xs f0 using h15
    have hc3 := (cond_sub (show op < W by rw [W_eq]; exact hop)
      (show k < W by rw [W_eq]; exact hk)).2.1
    by_cases hlt : op < k
    · xs f4 using hc3, Nat.mod_eq_of_lt hk, show ¬ (k ≤ op) by omega
      refine (effTree_spec l (off + 10) hcL hsl _ op ?_ ?_ ?_ hop).mono
        fun _ ⟨hf, h1, h2, h3⟩ => ⟨Frame.trans_same (by frame_tac) hf, h1,
          by simpa [EffTree.eval, hlt] using h2, by simpa [EffTree.eval, hlt] using h3⟩
      · rsimp; exact hl
      · rsimp; omega
      · rsimp; exact h15
    · xs f4 using hc3, Nat.mod_eq_of_lt hk, show k ≤ op by omega
      refine (effTree_spec r (EffTree.emit L tgt (off + 10) l).1 hcR hsr _ op ?_ ?_ ?_ hop).mono
        fun _ ⟨hf, h1, h2, h3⟩ => ⟨Frame.trans_same (by frame_tac) hf, h1,
          by simpa [EffTree.eval, hlt] using h2, by simpa [EffTree.eval, hlt] using h3⟩
      · rsimp; exact hl
      · rsimp
      · rsimp; exact h15

theorem eff_spec {L tgt : Nat} (hc : CodeAt (effT L tgt)) {s : St} {op : Nat}
    (hl : s.regs.length = 16) (hrip : s.rip = L) (h15 : s.reg 15 = some op) (hop : op < 57) :
    Reaches s (fun s' => Frame [10, 11] s s' ∧ s'.rip = tgt ∧
      s'.reg 10 = some (primEffTree.eval op).1 ∧ s'.reg 11 = some (primEffTree.eval op).2) := by
  rw [← primEffTree_emit] at hc
  exact effTree_spec primEffTree 0 hc primEffTree_small s op hl (by simpa using hrip) h15 (by omega)

/-! ### Annotation lookup (`lookupT`) -/

theorem lookup_loop_spec {L t annEnd b : Nat} {bs : List Nat} (ht : t < 16)
    (ht10 : t ≠ 10) (ht11 : t ≠ 11) (ht14 : t ≠ 14)
    (f25 : im (L + 25) = some (.cmp 10 11, 3))
    (f28 : im (L + 28) = some (.jcc 4 (L + 44), 2))
    (f30 : im (L + 30) = some (.ld32 14 10 0, 3))
    (f33 : im (L + 33) = some (.cmp 14 t, 3))
    (f36 : im (L + 36) = some (.jcc 4 (L + 44), 2))
    (f38 : im (L + 38) = some (.aluI .add 10 12, 4))
    (f42 : im (L + 42) = some (.jmp (L + 25), 2)) :
    ∀ (k q : Nat) (s : St) (idx : Nat),
      Buf s.mem b bs → (∀ x ∈ bs, x < 256) → b + bs.length < 2 ^ 48 →
      s.regs.length = 16 → s.rip = L + 25 →
      s.reg 10 = some (b + q) → s.reg 11 = some (b + annEnd) →
      s.reg t = some idx → idx < 2 ^ 64 →
      annEnd = q + 12 * k → annEnd ≤ bs.length →
      Reaches s (fun s' => Frame [10, 14] s s' ∧ s'.rip = L + 44 ∧
        match lookupA bs idx k q with
        | some q' => s'.reg 10 = some (b + q') ∧ q' + 12 ≤ annEnd
        | none => s'.reg 10 = some (b + annEnd))
  | 0, q, s, idx, hbuf, hby, hsm, hl, hrip, h10, h11, hrt, hidx, hae, hle => by
    have hq : q = annEnd := by omega
    rw [hq] at h10 ⊢
    xs f25 using h10, h11
    have hc4 := (cond_sub (show b + annEnd < W by rw [W_eq]; omega)
      (show b + annEnd < W by rw [W_eq]; omega)).2.2.1
    xs f28 using hc4
    exact Reaches.refl ⟨by frame_tac, rfl, h10⟩
  | k + 1, q, s, idx, hbuf, hby, hsm, hl, hrip, h10, h11, hrt, hidx, hae, hle => by
    xs f25 using h10, h11
    have hc4 := (cond_sub (show b + q < W by rw [W_eq]; omega)
      (show b + annEnd < W by rw [W_eq]; omega)).2.2.1
    xs f28 using hc4, show b + q ≠ b + annEnd by omega
    have hq4 : q + 4 ≤ bs.length := by omega
    have hu := u32_lt hby q
    xs f30 using hl, h10, ea_small (show b + q + 0 < 2 ^ 64 by omega), Nat.add_zero, rd32_buf hbuf hq4
    xs f33 using hl, hrt, ht14, ht14.symm
    have hc4' := (cond_sub (show u32 bs q < W by rw [W_eq]; omega)
      (show idx < W by rw [W_eq]; exact hidx)).2.2.1
    by_cases heq : u32 bs q = idx
    · have hc4eq : X86Int.cond (aluF .sub idx idx).snd 4 = some true := by
        simpa using (cond_sub (show idx < W by rw [W_eq]; exact hidx)
          (show idx < W by rw [W_eq]; exact hidx)).2.2.1
      xs f36 using hc4eq, heq
      exact Reaches.refl ⟨by frame_tac, rfl, by simp [lookupA, heq]; exact ⟨by rsimp; exact h10, by omega⟩⟩
    · xs f36 using hc4', heq
      xs f38 using hl, h10, add_small (show b + q + 12 < 2 ^ 64 by omega)
      xs f42
      have hstep : lookupA bs idx (k + 1) q = lookupA bs idx k (q + 12) := by simp [lookupA, heq]
      rw [hstep]
      refine reach_frame (by frame_tac)
        (lookup_loop_spec ht ht10 ht11 ht14 f25 f28 f30 f33 f36 f38 f42
          k (q + 12) _ idx ?_ hby hsm ?_ rfl ?_ ?_ ?_ hidx (by omega) hle)
      · rsimp; exact hbuf
      · rsimp; exact hl
      · rsimp using hl; omega
      · rsimp using hl; exact h11
      · simp (config := { decide := true }) only [reg_mk_set_ne, reg_mk_self, ne_eq, not_false_eq_true,
          ht10.symm, ht14.symm, hrt]

theorem lookup_spec {L t : Nat} (hc : CodeAt (lookupT L t))
    (ht : t < 16) (ht10 : t ≠ 10) (ht11 : t ≠ 11) (ht14 : t ≠ 14)
    {b : Nat} {bs : List Nat} {s : St} {wordOff annOff idx : Nat}
    (hbuf : Buf s.mem b bs) (hby : ∀ x ∈ bs, x < 256) (hsm : b + bs.length < 2 ^ 48)
    (hl : s.regs.length = 16) (hrip : s.rip = L)
    (h12 : s.reg 12 = some (b + wordOff)) (h9 : s.reg 9 = some (b + annOff))
    (hrt : s.reg t = some idx) (hidx : idx < 2 ^ 64)
    (hw20 : wordOff + 20 ≤ bs.length)
    (hann : annOff + 12 * u32 bs (wordOff + 16) ≤ bs.length) :
    Reaches s (fun s' => Frame [10, 11, 14] s s' ∧ s'.rip = L + 44 ∧
      s'.reg 11 = some (b + annOff + 12 * u32 bs (wordOff + 16)) ∧
      match lookupA bs idx (u32 bs (wordOff + 16)) annOff with
      | some q' => s'.reg 10 = some (b + q') ∧ q' + 12 ≤ annOff + 12 * u32 bs (wordOff + 16)
      | none => s'.reg 10 = some (b + annOff + 12 * u32 bs (wordOff + 16))) := by
  simp only [lookupT, CodeAt_cons, CodeAt_nil, and_true] at hc
  obtain ⟨f0, f5, f8, f12, f16, f19, f22, f25, f28, f30, f33, f36, f38, f42⟩ := hc
  have hna : u32 bs (wordOff + 16) < 2 ^ 32 := u32_lt hby (wordOff + 16)
  xs f0 using hl, h12, ea_small (show b + wordOff + 16 < 2 ^ 64 by omega),
    show b + wordOff + 16 = b + (wordOff + 16) by omega, rd32_buf hbuf hw20
  xs f5 using hl
  xs f8 using hl, show u32 bs (wordOff + 16) * 8 % 18446744073709551616 = u32 bs (wordOff + 16) * 8 by omega
  xs f12 using hl, show u32 bs (wordOff + 16) * 4 % 18446744073709551616 = u32 bs (wordOff + 16) * 4 by omega
  xs f16 using hl, add_small (show u32 bs (wordOff + 16) * 8 + u32 bs (wordOff + 16) * 4 < 2 ^ 64 by omega)
  xs f19 using hl, h9, add_small (show u32 bs (wordOff + 16) * 8 + u32 bs (wordOff + 16) * 4 + (b + annOff) < 2 ^ 64 by omega)
  xs f22 using hl, h9
  have h12na : u32 bs (wordOff + 16) * 8 + u32 bs (wordOff + 16) * 4 + (b + annOff) = b + (annOff + 12 * u32 bs (wordOff + 16)) := by omega
  refine (lookup_loop_spec ht ht10 ht11 ht14 f25 f28 f30 f33 f36 f38 f42
    (u32 bs (wordOff + 16)) annOff _ idx ?_ hby hsm ?_ ?_ ?_ ?_ ?_ hidx rfl hann).mono ?_
  · rsimp; exact hbuf
  · rsimp; exact hl
  · rsimp; omega
  · rsimp using hl
  · rsimp using hl; exact h12na
  · simp (config := { decide := true }) only [reg_mk_set_ne, reg_mk_self, ne_eq, not_false_eq_true,
      ht10.symm, ht11.symm, ht14.symm, hrt]
  · intro s' ⟨hf, hrip', hm⟩
    refine ⟨(Frame.trans (S := [10, 11, 14]) (by frame_tac) hf).mono ?_, hrip', ?_, ?_⟩
    · intro x hx; simp only [List.mem_cons, List.not_mem_nil, List.mem_append, or_false] at hx ⊢
      rcases hx with (rfl | rfl | rfl) | (rfl | rfl) <;> decide
    · rw [hf.regs 11 (by decide)]
      rsimp using hl; omega
    · simpa [Nat.add_assoc] using hm

end SporosCheckBin
