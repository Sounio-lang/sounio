import Bootstrap.SeedA
import Bootstrap.SeedB
import Bootstrap.X86Uni

/-!
Lifting `SeedA` (`271B`) and `SeedB` (`283B`) correctness and refusal theorems
over every byte list `L : List Nat` (`∀ c ∈ L, c < 256`) from `X86Min` to the
unified single physical x86-64 machine `X86Uni.{run,prun}`.
-/

namespace X86Uni

open X86Env PDist

private theorem seedA_load_meta :
    X86Min.SeedA.imA.vbase = 0x400000 ∧ X86Min.SeedA.s0.rip = 0x400078 := by
  have hchk :
      (match X86Min.load X86Min.Data.seedA seedSp [] with
       | some (im, st) => im.vbase == 0x400000 && st.rip == 0x400078
       | none => false) = true := by
    decide +kernel
  rw [show seedSp = 0x7ffffffde000 from rfl, X86Min.SeedA.load_A []] at hchk
  simpa [Bool.and_eq_true, beq_iff_eq] using hchk

private theorem seedB_load_meta :
    X86Min.SeedB.imB.vbase = 0x400000 ∧ X86Min.SeedB.s0.rip = 0x400078 := by
  have hchk :
      (match X86Min.load X86Min.Data.seedB seedSp [] with
       | some (im, st) => im.vbase == 0x400000 && st.rip == 0x400078
       | none => false) = true := by
    decide +kernel
  rw [show seedSp = 0x7ffffffde000 from rfl, X86Min.SeedB.load_B []] at hchk
  simpa [Bool.and_eq_true, beq_iff_eq] using hchk

private theorem seedA_exec_sim {F : Nat} {L : List Nat} (hL : ∀ c ∈ L, c < 256)
    {code : Nat} {out : List Nat}
    (hx : X86Min.exec F X86Min.Data.seedA L = .exit code out) :
    run seedAImage F (.ok (initSt 0x400078 L)) = .exit code ∧
    ∃ k < F, ∃ if_,
      run seedAImage k (.ok (initSt 0x400078 L)) = .ok if_ ∧
      step seedAImage if_ = .exit code ∧
      if_.k.out = out := by
  have hload : X86Min.load X86Min.Data.seedA seedSp L =
      some (X86Min.SeedA.imA, { X86Min.SeedA.s0 with inp := L }) := by
    simpa [seedSp] using X86Min.SeedA.load_A L
  have him_eq : X86Min.SeedA.imA =
      X86Min.Img.mk' X86Min.Data.seedA 0x400000 X86Min.Data.seedA.length := by
    have hm := (load_facts hload).1
    rwa [seedA_load_meta.1] at hm
  have hcanon : X86Min.SeedA.imA.vbase + X86Min.SeedA.imA.len + 16 < W := by
    have hlenA : X86Min.SeedA.imA.len = X86Min.Data.seedA.length :=
      congrArg X86Min.Img.len (load_facts hload).1
    rw [seedA_load_meta.1, hlenA, seedA_len_eq]; decide
  have hno11 : ∀ a ins len, X86Min.SeedA.imA.decodeAt a = some (ins, len) →
      ∀ w, ins ≠ .rr .xor w 11 11 := by
    intro a ins len hdec w
    rw [him_eq] at hdec
    exact seedA_imageOk.noXor11 a ins len hdec w
  have hnoJmp : ∀ a r len, X86Min.SeedA.imA.decodeAt a = some (.jmpReg r, len) → False := by
    intro a r len hdec
    rw [him_eq] at hdec
    exact seedA_imageOk.noJmpReg a r len hdec
  have hsim := exec_min_sim seedA_imageOk.bytesLt256 hL hload hcanon hno11 hnoJmp hx
  rw [seedA_load_meta.1] at hsim
  change run seedAImage F (.ok (initSt X86Min.SeedA.s0.rip L)) = .exit code ∧
    ∃ k < F, ∃ mf if_,
      X86Min.runS X86Min.SeedA.imA k { X86Min.SeedA.s0 with inp := L } = .inl mf ∧
      run seedAImage k (.ok (initSt X86Min.SeedA.s0.rip L)) = .ok if_ ∧
      Sim mf if_ ∧
      X86Min.step X86Min.SeedA.imA mf = .inr (.exit code out) ∧
      step seedAImage if_ = .exit code ∧
      if_.k.out = out at hsim
  rw [seedA_load_meta.2] at hsim
  obtain ⟨hrun, k, hk, _, if_, _, hrun_k, _, _, hstep_if, hout_if⟩ := hsim
  exact ⟨hrun, k, hk, if_, hrun_k, hstep_if, hout_if⟩

private theorem seedB_exec_sim {F : Nat} {L : List Nat} (hL : ∀ c ∈ L, c < 256)
    {code : Nat} {out : List Nat}
    (hx : X86Min.exec F X86Min.Data.seedB L = .exit code out) :
    run seedBImage F (.ok (initSt 0x400078 L)) = .exit code ∧
    ∃ k < F, ∃ if_,
      run seedBImage k (.ok (initSt 0x400078 L)) = .ok if_ ∧
      step seedBImage if_ = .exit code ∧
      if_.k.out = out := by
  have hload : X86Min.load X86Min.Data.seedB seedSp L =
      some (X86Min.SeedB.imB, { X86Min.SeedB.s0 with inp := L }) := by
    simpa [seedSp] using X86Min.SeedB.load_B L
  have him_eq : X86Min.SeedB.imB =
      X86Min.Img.mk' X86Min.Data.seedB 0x400000 X86Min.Data.seedB.length := by
    have hm := (load_facts hload).1
    rwa [seedB_load_meta.1] at hm
  have hcanon : X86Min.SeedB.imB.vbase + X86Min.SeedB.imB.len + 16 < W := by
    have hlenB : X86Min.SeedB.imB.len = X86Min.Data.seedB.length :=
      congrArg X86Min.Img.len (load_facts hload).1
    rw [seedB_load_meta.1, hlenB, seedB_len_eq]; decide
  have hno11 : ∀ a ins len, X86Min.SeedB.imB.decodeAt a = some (ins, len) →
      ∀ w, ins ≠ .rr .xor w 11 11 := by
    intro a ins len hdec w
    rw [him_eq] at hdec
    exact seedB_imageOk.noXor11 a ins len hdec w
  have hnoJmp : ∀ a r len, X86Min.SeedB.imB.decodeAt a = some (.jmpReg r, len) → False := by
    intro a r len hdec
    rw [him_eq] at hdec
    exact seedB_imageOk.noJmpReg a r len hdec
  have hsim := exec_min_sim seedB_imageOk.bytesLt256 hL hload hcanon hno11 hnoJmp hx
  rw [seedB_load_meta.1] at hsim
  change run seedBImage F (.ok (initSt X86Min.SeedB.s0.rip L)) = .exit code ∧
    ∃ k < F, ∃ mf if_,
      X86Min.runS X86Min.SeedB.imB k { X86Min.SeedB.s0 with inp := L } = .inl mf ∧
      run seedBImage k (.ok (initSt X86Min.SeedB.s0.rip L)) = .ok if_ ∧
      Sim mf if_ ∧
      X86Min.step X86Min.SeedB.imB mf = .inr (.exit code out) ∧
      step seedBImage if_ = .exit code ∧
      if_.k.out = out at hsim
  rw [seedB_load_meta.2] at hsim
  obtain ⟨hrun, k, hk, _, if_, _, hrun_k, _, _, hstep_if, hout_if⟩ := hsim
  exact ⟨hrun, k, hk, if_, hrun_k, hstep_if, hout_if⟩

private theorem seedA_exec_prun_sim (alts : Alts) (r : Rates) (hr : r.ok)
    {F : Nat} {L : List Nat} (hL : ∀ c ∈ L, c < 256)
    {code : Nat} {out : List Nat}
    (hx : X86Min.exec F X86Min.Data.seedA L = .exit code out) :
    (prun alts (r.state hr) r.fetch seedAImage F (.ok (initSt 0x400078 L))).ok ∧
    fm (prun alts (r.state hr) r.fetch seedAImage F (.ok (initSt 0x400078 L))) ≤ F * r.total ∧
    faultFree (prun alts (r.state hr) r.fetch seedAImage F (.ok (initSt 0x400078 L))) =
      [X86Int.Out.exit code] ∧
    ∃ k < F, ∃ if_,
      run seedAImage k (.ok (initSt 0x400078 L)) = .ok if_ ∧
      faultFree (prun alts (r.state hr) r.fetch seedAImage k (.ok (initSt 0x400078 L))) =
        [X86Int.Out.ok if_] ∧
      if_.k.out = out ∧
      step seedAImage if_ = .exit code := by
  have hload : X86Min.load X86Min.Data.seedA seedSp L =
      some (X86Min.SeedA.imA, { X86Min.SeedA.s0 with inp := L }) := by
    simpa [seedSp] using X86Min.SeedA.load_A L
  have him_eq : X86Min.SeedA.imA =
      X86Min.Img.mk' X86Min.Data.seedA 0x400000 X86Min.Data.seedA.length := by
    have hm := (load_facts hload).1
    rwa [seedA_load_meta.1] at hm
  have hcanon : X86Min.SeedA.imA.vbase + X86Min.SeedA.imA.len + 16 < W := by
    have hlenA : X86Min.SeedA.imA.len = X86Min.Data.seedA.length :=
      congrArg X86Min.Img.len (load_facts hload).1
    rw [seedA_load_meta.1, hlenA, seedA_len_eq]; decide
  have hno11 : ∀ a ins len, X86Min.SeedA.imA.decodeAt a = some (ins, len) →
      ∀ w, ins ≠ .rr .xor w 11 11 := by
    intro a ins len hdec w
    rw [him_eq] at hdec
    exact seedA_imageOk.noXor11 a ins len hdec w
  have hnoJmp : ∀ a r len, X86Min.SeedA.imA.decodeAt a = some (.jmpReg r, len) → False := by
    intro a r len hdec
    rw [him_eq] at hdec
    exact seedA_imageOk.noJmpReg a r len hdec
  have hp := exec_prun_min_sim alts r hr seedA_imageOk.bytesLt256 hL hload hcanon hno11 hnoJmp hx
  rw [seedA_load_meta.1] at hp
  dsimp only at hp
  rw [seedA_load_meta.2] at hp
  exact hp

private theorem seedB_exec_prun_sim (alts : Alts) (r : Rates) (hr : r.ok)
    {F : Nat} {L : List Nat} (hL : ∀ c ∈ L, c < 256)
    {code : Nat} {out : List Nat}
    (hx : X86Min.exec F X86Min.Data.seedB L = .exit code out) :
    (prun alts (r.state hr) r.fetch seedBImage F (.ok (initSt 0x400078 L))).ok ∧
    fm (prun alts (r.state hr) r.fetch seedBImage F (.ok (initSt 0x400078 L))) ≤ F * r.total ∧
    faultFree (prun alts (r.state hr) r.fetch seedBImage F (.ok (initSt 0x400078 L))) =
      [X86Int.Out.exit code] ∧
    ∃ k < F, ∃ if_,
      run seedBImage k (.ok (initSt 0x400078 L)) = .ok if_ ∧
      faultFree (prun alts (r.state hr) r.fetch seedBImage k (.ok (initSt 0x400078 L))) =
        [X86Int.Out.ok if_] ∧
      if_.k.out = out ∧
      step seedBImage if_ = .exit code := by
  have hload : X86Min.load X86Min.Data.seedB seedSp L =
      some (X86Min.SeedB.imB, { X86Min.SeedB.s0 with inp := L }) := by
    simpa [seedSp] using X86Min.SeedB.load_B L
  have him_eq : X86Min.SeedB.imB =
      X86Min.Img.mk' X86Min.Data.seedB 0x400000 X86Min.Data.seedB.length := by
    have hm := (load_facts hload).1
    rwa [seedB_load_meta.1] at hm
  have hcanon : X86Min.SeedB.imB.vbase + X86Min.SeedB.imB.len + 16 < W := by
    have hlenB : X86Min.SeedB.imB.len = X86Min.Data.seedB.length :=
      congrArg X86Min.Img.len (load_facts hload).1
    rw [seedB_load_meta.1, hlenB, seedB_len_eq]; decide
  have hno11 : ∀ a ins len, X86Min.SeedB.imB.decodeAt a = some (ins, len) →
      ∀ w, ins ≠ .rr .xor w 11 11 := by
    intro a ins len hdec w
    rw [him_eq] at hdec
    exact seedB_imageOk.noXor11 a ins len hdec w
  have hnoJmp : ∀ a r len, X86Min.SeedB.imB.decodeAt a = some (.jmpReg r, len) → False := by
    intro a r len hdec
    rw [him_eq] at hdec
    exact seedB_imageOk.noJmpReg a r len hdec
  have hp := exec_prun_min_sim alts r hr seedB_imageOk.bytesLt256 hL hload hcanon hno11 hnoJmp hx
  rw [seedB_load_meta.1] at hp
  dsimp only at hp
  rw [seedB_load_meta.2] at hp
  exact hp

/-- Full lifting of `SeedA` correctness over every byte input `L : List Nat`:
running `X86Min.Data.seedA` (271B) in `X86Uni.run` exits 0 with output `bs` iff `HexA L none bs`. -/
theorem seedA_uni_correct (L bs : List Nat) (hL : ∀ c ∈ L, c < 256) :
    (∃ F, run seedAImage F (.ok (initSt 0x400078 L)) = .exit 0 ∧
      ∃ k < F, ∃ if_,
        run seedAImage k (.ok (initSt 0x400078 L)) = .ok if_ ∧
        step seedAImage if_ = .exit 0 ∧
        if_.k.out = bs) ↔
    X86Min.Formats.HexA L none bs := by
  constructor
  · rintro ⟨F, _, k, _, if_, hk_run, hk_step, hk_out⟩
    rcases Classical.em (∃ bs', X86Min.Formats.HexA L none bs') with ⟨bs', hhex⟩ | hno
    · obtain ⟨F', hx'⟩ := (X86Min.SeedA.seedA_correct L bs' hL).mpr hhex
      obtain ⟨_, k', _, if_', hk_run', hk_step', hk_out'⟩ := seedA_exec_sim hL hx'
      obtain ⟨_, hst_eq, _⟩ := run_exit_step_unique hk_run hk_step hk_run' hk_step'
      have hbs : bs = bs' := by rw [← hk_out, hst_eq, hk_out']
      rwa [hbs]
    · have hforall : ∀ bs', ¬ X86Min.Formats.HexA L none bs' := fun bs' h => hno ⟨bs', h⟩
      obtain ⟨F', x', hx'⟩ := X86Min.SeedA.seedA_refuses L hL hforall
      obtain ⟨_, k', _, if_', hk_run', hk_step', _⟩ := seedA_exec_sim hL hx'
      obtain ⟨_, _, hcode⟩ := run_exit_step_unique hk_run hk_step hk_run' hk_step'
      cases hcode
  · intro hhex
    obtain ⟨F, hx⟩ := (X86Min.SeedA.seedA_correct L bs hL).mpr hhex
    exact ⟨F, seedA_exec_sim hL hx⟩

/-- Full lifting of `SeedA` refusal over every invalid byte input `L : List Nat`:
running `X86Min.Data.seedA` (271B) in `X86Uni.run` exits with code 1. -/
theorem seedA_uni_refuses (L : List Nat) (hL : ∀ c ∈ L, c < 256)
    (h : ∀ bs, ¬ X86Min.Formats.HexA L none bs) :
    ∃ F, run seedAImage F (.ok (initSt 0x400078 L)) = .exit 1 ∧
      ∃ k < F, ∃ if_ out,
        run seedAImage k (.ok (initSt 0x400078 L)) = .ok if_ ∧
        step seedAImage if_ = .exit 1 ∧
        if_.k.out = out := by
  obtain ⟨F, x, hx⟩ := X86Min.SeedA.seedA_refuses L hL h
  obtain ⟨hrun, k, hk, if_, hrun_k, hstep_k, hout_k⟩ := seedA_exec_sim hL hx
  exact ⟨F, hrun, k, hk, if_, x, hrun_k, hstep_k, hout_k⟩

/-- Full lifting of `SeedB` correctness over every byte input `L : List Nat`:
running `X86Min.Data.seedB` (283B) in `X86Uni.run` exits 0 with output `bs` iff `OctB L bs`. -/
theorem seedB_uni_correct (L bs : List Nat) (hL : ∀ c ∈ L, c < 256) :
    (∃ F, run seedBImage F (.ok (initSt 0x400078 L)) = .exit 0 ∧
      ∃ k < F, ∃ if_,
        run seedBImage k (.ok (initSt 0x400078 L)) = .ok if_ ∧
        step seedBImage if_ = .exit 0 ∧
        if_.k.out = bs) ↔
    X86Min.Formats.OctB L bs := by
  constructor
  · rintro ⟨F, _, k, _, if_, hk_run, hk_step, hk_out⟩
    rcases Classical.em (∃ bs', X86Min.Formats.OctB L bs') with ⟨bs', hoct⟩ | hno
    · obtain ⟨F', hx'⟩ := (X86Min.SeedB.seedB_correct L bs' hL).mpr hoct
      obtain ⟨_, k', _, if_', hk_run', hk_step', hk_out'⟩ := seedB_exec_sim hL hx'
      obtain ⟨_, hst_eq, _⟩ := run_exit_step_unique hk_run hk_step hk_run' hk_step'
      have hbs : bs = bs' := by rw [← hk_out, hst_eq, hk_out']
      rwa [hbs]
    · have hforall : ∀ bs', ¬ X86Min.Formats.OctB L bs' := fun bs' h => hno ⟨bs', h⟩
      obtain ⟨F', x', hx'⟩ := X86Min.SeedB.seedB_refuses L hL hforall
      obtain ⟨_, k', _, if_', hk_run', hk_step', _⟩ := seedB_exec_sim hL hx'
      obtain ⟨_, _, hcode⟩ := run_exit_step_unique hk_run hk_step hk_run' hk_step'
      cases hcode
  · intro hoct
    obtain ⟨F, hx⟩ := (X86Min.SeedB.seedB_correct L bs hL).mpr hoct
    exact ⟨F, seedB_exec_sim hL hx⟩

/-- Full lifting of `SeedB` refusal over every invalid byte input `L : List Nat`:
running `X86Min.Data.seedB` (283B) in `X86Uni.run` exits with code 1. -/
theorem seedB_uni_refuses (L : List Nat) (hL : ∀ c ∈ L, c < 256)
    (h : ∀ bs, ¬ X86Min.Formats.OctB L bs) :
    ∃ F, run seedBImage F (.ok (initSt 0x400078 L)) = .exit 1 ∧
      ∃ k < F, ∃ if_ out,
        run seedBImage k (.ok (initSt 0x400078 L)) = .ok if_ ∧
        step seedBImage if_ = .exit 1 ∧
        if_.k.out = out := by
  obtain ⟨F, x, hx⟩ := X86Min.SeedB.seedB_refuses L hL h
  obtain ⟨hrun, k, hk, if_, hrun_k, hstep_k, hout_k⟩ := seedB_exec_sim hL hx
  exact ⟨F, hrun, k, hk, if_, x, hrun_k, hstep_k, hout_k⟩

/-- Physical fault-bound theorem for `SeedA` (271B) in `X86Uni.prun`: for every valid
input `L` with `HexA L none bs`, `X86Uni.prun` is sub-unitary, bounds fault mass by `F * r.total`,
and its fault-free mass exits 0 with final output `bs`. -/
theorem seedA_uni_prun (alts : Alts) (r : Rates) (hr : r.ok)
    (L bs : List Nat) (hL : ∀ c ∈ L, c < 256) (hhex : X86Min.Formats.HexA L none bs) :
    ∃ F,
      (prun alts (r.state hr) r.fetch seedAImage F (.ok (initSt 0x400078 L))).ok ∧
      fm (prun alts (r.state hr) r.fetch seedAImage F (.ok (initSt 0x400078 L))) ≤ F * r.total ∧
      faultFree (prun alts (r.state hr) r.fetch seedAImage F (.ok (initSt 0x400078 L))) =
        [X86Int.Out.exit 0] ∧
      ∃ k < F, ∃ if_,
        run seedAImage k (.ok (initSt 0x400078 L)) = .ok if_ ∧
        faultFree (prun alts (r.state hr) r.fetch seedAImage k (.ok (initSt 0x400078 L))) =
          [X86Int.Out.ok if_] ∧
        if_.k.out = bs ∧
        step seedAImage if_ = .exit 0 := by
  obtain ⟨F, hx⟩ := (X86Min.SeedA.seedA_correct L bs hL).mpr hhex
  exact ⟨F, seedA_exec_prun_sim alts r hr hL hx⟩

/-- Physical fault-bound theorem for `SeedB` (283B) in `X86Uni.prun`: for every valid
input `L` with `OctB L bs`, `X86Uni.prun` is sub-unitary, bounds fault mass by `F * r.total`,
and its fault-free mass exits 0 with final output `bs`. -/
theorem seedB_uni_prun (alts : Alts) (r : Rates) (hr : r.ok)
    (L bs : List Nat) (hL : ∀ c ∈ L, c < 256) (hoct : X86Min.Formats.OctB L bs) :
    ∃ F,
      (prun alts (r.state hr) r.fetch seedBImage F (.ok (initSt 0x400078 L))).ok ∧
      fm (prun alts (r.state hr) r.fetch seedBImage F (.ok (initSt 0x400078 L))) ≤ F * r.total ∧
      faultFree (prun alts (r.state hr) r.fetch seedBImage F (.ok (initSt 0x400078 L))) =
        [X86Int.Out.exit 0] ∧
      ∃ k < F, ∃ if_,
        run seedBImage k (.ok (initSt 0x400078 L)) = .ok if_ ∧
        faultFree (prun alts (r.state hr) r.fetch seedBImage k (.ok (initSt 0x400078 L))) =
          [X86Int.Out.ok if_] ∧
        if_.k.out = bs ∧
        step seedBImage if_ = .exit 0 := by
  obtain ⟨F, hx⟩ := (X86Min.SeedB.seedB_correct L bs hL).mpr hoct
  exact ⟨F, seedB_exec_prun_sim alts r hr hL hx⟩

/-- Physical fault-bound theorem for refused `SeedA` inputs in `X86Uni.prun`. -/
theorem seedA_uni_refuses_prun (alts : Alts) (r : Rates) (hr : r.ok)
    (L : List Nat) (hL : ∀ c ∈ L, c < 256) (h : ∀ bs, ¬ X86Min.Formats.HexA L none bs) :
    ∃ F,
      (prun alts (r.state hr) r.fetch seedAImage F (.ok (initSt 0x400078 L))).ok ∧
      fm (prun alts (r.state hr) r.fetch seedAImage F (.ok (initSt 0x400078 L))) ≤ F * r.total ∧
      faultFree (prun alts (r.state hr) r.fetch seedAImage F (.ok (initSt 0x400078 L))) =
        [X86Int.Out.exit 1] ∧
      ∃ k < F, ∃ if_ out,
        run seedAImage k (.ok (initSt 0x400078 L)) = .ok if_ ∧
        faultFree (prun alts (r.state hr) r.fetch seedAImage k (.ok (initSt 0x400078 L))) =
          [X86Int.Out.ok if_] ∧
        if_.k.out = out ∧
        step seedAImage if_ = .exit 1 := by
  obtain ⟨F, x, hx⟩ := X86Min.SeedA.seedA_refuses L hL h
  obtain ⟨hok, hfm, hff, k, hk, if_, hrun_k, hff_k, hout_k, hstep_k⟩ :=
    seedA_exec_prun_sim alts r hr hL hx
  exact ⟨F, hok, hfm, hff, k, hk, if_, x, hrun_k, hff_k, hout_k, hstep_k⟩

/-- Physical fault-bound theorem for refused `SeedB` inputs in `X86Uni.prun`. -/
theorem seedB_uni_refuses_prun (alts : Alts) (r : Rates) (hr : r.ok)
    (L : List Nat) (hL : ∀ c ∈ L, c < 256) (h : ∀ bs, ¬ X86Min.Formats.OctB L bs) :
    ∃ F,
      (prun alts (r.state hr) r.fetch seedBImage F (.ok (initSt 0x400078 L))).ok ∧
      fm (prun alts (r.state hr) r.fetch seedBImage F (.ok (initSt 0x400078 L))) ≤ F * r.total ∧
      faultFree (prun alts (r.state hr) r.fetch seedBImage F (.ok (initSt 0x400078 L))) =
        [X86Int.Out.exit 1] ∧
      ∃ k < F, ∃ if_ out,
        run seedBImage k (.ok (initSt 0x400078 L)) = .ok if_ ∧
        faultFree (prun alts (r.state hr) r.fetch seedBImage k (.ok (initSt 0x400078 L))) =
          [X86Int.Out.ok if_] ∧
        if_.k.out = out ∧
        step seedBImage if_ = .exit 1 := by
  obtain ⟨F, x, hx⟩ := X86Min.SeedB.seedB_refuses L hL h
  obtain ⟨hok, hfm, hff, k, hk, if_, hrun_k, hff_k, hout_k, hstep_k⟩ :=
    seedB_exec_prun_sim alts r hr hL hx
  exact ⟨F, hok, hfm, hff, k, hk, if_, x, hrun_k, hff_k, hout_k, hstep_k⟩

end X86Uni
