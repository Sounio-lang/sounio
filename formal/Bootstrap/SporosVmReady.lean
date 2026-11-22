import Bootstrap.SporosVmSimBase

/-!
# Native x86-64 VM Ready-State Construction & Non-Vacuity (`SporosVmReady`)

Constructs an explicit initial post-boot `X86Vm.St` (`mkVmReadyState`) and `VmCtx`
(`mkVmReadyCtx`) at `SporosVmBin.L_vm_loop` for any `FlatProgFit` Sporos program,
memory size, input byte stream, and kernel oracle, and proves `mkVmReadyState_rel`:

```
VmRel pr (SporosProvLen.linit pr msize inp) (mkVmReadyState pr msize inp oracle)
```

This establishes non-vacuity of `VmRel` for all fitting programs (including the
concrete examples `echo`, `hello`, `upper`, `fib`, and `provDemo`).

Root A framing: Root A is MUSE's hand-written checker
(`bootstrap/roots/sporos/sporos-stage1.hex`), `spr1_check` is an independent
second check, and `sporos_vm` is the post-admission native x86-64 VM paired
with `Sporos.decode` + `Sporos.checkProg`.
-/

namespace SporosVmReady

open X86Env X86Int X86Vm SporosVmCore SporosVmExec SporosVmSimBase

local instance : Inhabited Region := ⟨⟨0, [], false⟩⟩
local instance : Inhabited SporosProvLen.LVal := ⟨⟨0, 0⟩⟩

/-! ### Helper lemmas for page alignment, zero-padded byte slices, and sorted memories -/

private theorem vmBytes_lt_256 : ∀ b ∈ SporosVmBin.vmBytes, b < 256 := by
  have h : SporosVmBin.vmBytes.all (· < 256) = true := by decide +kernel
  intro b hb
  exact of_decide_eq_true (List.all_eq_true.mp h b hb)

private theorem pages_mod (n : Nat) : X86Env.pages n % X86Env.page = 0 := by
  unfold X86Env.pages X86Env.page
  exact Nat.mul_mod_left _ 4096

private theorem pages_ge (n : Nat) : n ≤ X86Env.pages n := by
  unfold X86Env.pages X86Env.page
  omega

private theorem pages_pos {n : Nat} (hn : 0 < n) : 0 < X86Env.pages n := by
  have := pages_ge n; omega

private theorem leVal_replicate_zero (n : Nat) : leVal (List.replicate n 0) = 0 := by
  induction n with
  | zero => rfl
  | succ k ih => simp [List.replicate_succ, leVal, ih]

private theorem drop_take_replicate_zero (len a k : Nat) :
    leVal (((List.replicate len 0).drop a).take k) = 0 := by
  rw [List.drop_replicate, List.take_replicate, leVal_replicate_zero]

private theorem flatMap_bytes8_lt_256 (qs : List Nat) :
    ∀ b ∈ qs.flatMap bytes8, b < 256 := by
  intro b hb
  rw [List.mem_flatMap] at hb
  obtain ⟨_, _, hb'⟩ := hb
  exact bytes8_lt b hb'

private theorem leVal_flatMap_bytes8_pad (qs : List Nat) (pad : Nat) {j : Nat}
    (hj : j < qs.length) (hW : qs[j]! < X86Env.W) :
    leVal (((qs.flatMap bytes8 ++ List.replicate pad 0).drop (j * 8)).take 8) = qs[j]! := by
  have hlen : (qs.flatMap bytes8).length = qs.length * 8 := flatMap_bytes8_length qs
  have h1 : ((qs.flatMap bytes8 ++ List.replicate pad 0).drop (j * 8)).take 8 =
      ((qs.flatMap bytes8).drop (j * 8)).take 8 := by
    rw [List.drop_append_of_le_length (by omega),
      List.take_append_of_le_length (by rw [List.length_drop]; omega)]
  rw [h1]
  exact flatMap_bytes8_drop_take qs j hj hW

private theorem pairwise_of_chain : ∀ (rs : List Region),
    (∀ i, i + 1 < rs.length → rs[i]!.hi ≤ rs[i + 1]!.base) →
    (∀ r ∈ rs, r.base ≤ r.hi) →
    rs.Pairwise (fun a b => a.hi ≤ b.base ∨ b.hi ≤ a.base)
  | [], _, _ => .nil
  | [_], _, _ => .cons (by simp) .nil
  | r0 :: r1 :: rest, hch, hle => by
    have h01 : r0.hi ≤ r1.base := by simpa using hch 0 (by simp)
    have htl : (r1 :: rest).Pairwise (fun a b => a.hi ≤ b.base ∨ b.hi ≤ a.base) :=
      pairwise_of_chain (r1 :: rest)
        (fun i hi => by simpa using hch (i + 1) (by simp at hi ⊢; omega))
        (fun r hr => hle r (List.mem_cons_of_mem _ hr))
    have hge : ∀ u ∈ (r1 :: rest), r1.base ≤ u.base := by
      have go : ∀ (xs : List Region),
          (∀ i, i + 1 < xs.length → xs[i]!.hi ≤ xs[i + 1]!.base) →
          (∀ r ∈ xs, r.base ≤ r.hi) →
          ∀ (x : Region) (t : List Region), xs = x :: t → ∀ u ∈ x :: t, x.base ≤ u.base := by
        intro xs
        induction xs with
        | nil => intro _ _ _ _ h; cases h
        | cons a as ih =>
          intro hch' hle' x t heq u hu
          cases heq
          rcases List.mem_cons.mp hu with rfl | hu'
          · exact Nat.le_refl _
          · cases as with
            | nil => cases hu'
            | cons b bs =>
              have hab : a.hi ≤ b.base := by simpa using hch' 0 (by simp)
              have ha : a.base ≤ a.hi := hle' a (by simp)
              have hbu : b.base ≤ u.base := ih
                (fun i hi => by simpa using hch' (i + 1) (by simp at hi ⊢; omega))
                (fun r hr => hle' r (List.mem_cons_of_mem _ hr))
                b bs rfl u hu'
              omega
      exact go (r1 :: rest)
        (fun i hi => by simpa using hch (i + 1) (by simp at hi ⊢; omega))
        (fun r hr => hle r (List.mem_cons_of_mem _ hr))
        r1 rest rfl
    exact .cons (fun u hu => Or.inl (Nat.le_trans h01 (hge u hu))) htl

private theorem nodup_of_strict_chain (rs : List Region)
    (hch : ∀ i, i + 1 < rs.length → rs[i]!.hi ≤ rs[i + 1]!.base)
    (hpos : ∀ r ∈ rs, 0 < r.data.length) :
    (rs.map Region.base).Nodup := by
  have hpw := pairwise_of_chain rs hch (fun r _ => by simp only [Region.hi]; omega)
  have hne : rs.Pairwise (fun a b => a.base ≠ b.base) := by
    have go : ∀ (xs : List Region),
        xs.Pairwise (fun a b => a.hi ≤ b.base ∨ b.hi ≤ a.base) →
        (∀ r ∈ xs, 0 < r.data.length) →
        xs.Pairwise (fun a b => a.base ≠ b.base) := by
      intro xs hp hlen
      induction hp with
      | nil => exact .nil
      | @cons a l ha _ ih =>
        refine .cons ?_ (ih (fun r hr => hlen r (List.mem_cons_of_mem _ hr)))
        intro b hb
        have h1 := ha b hb
        have hla := hlen a (by simp)
        have hlb := hlen b (List.mem_cons_of_mem _ hb)
        simp only [Region.hi] at h1
        omega
    exact go rs hpw hpos
  exact List.pairwise_map.mpr hne

/-! ### Explicit ready-state construction -/

def readyInitLocals (pr : Sporos.Prog) : Nat :=
  match pr.words[pr.main]? with
  | some m => m.locals
  | none => 0

def readyCodeLen (pr : Sporos.Prog) : Nat :=
  X86Env.pages (max (encodeFlatProg pr).length 1)

def readyLocPages (pr : Sporos.Prog) : Nat :=
  X86Env.pages (max (readyInitLocals pr * 16) 4096)

def readyMemwLen (msize : Nat) : Nat :=
  X86Env.pages (max msize 1)

def readyMempLen (msize : Nat) : Nat :=
  X86Env.pages (8 * max msize 1)

def readyInpPages (inp : List Nat) : Nat :=
  X86Env.pages (max inp.length 1)

/-- End address of the laid-out ready-state arenas (`≤ 2^47` guarantees all bounds). -/
def readyEndAddr (pr : Sporos.Prog) (msize : Nat) (inp : List Nat) : Nat :=
  0x1000000 + 3 * readyCodeLen pr + 3 * 4096 + readyLocPages pr +
    readyMemwLen msize + readyMempLen msize + 2 * 4096 + readyInpPages inp + 4096

/-- Construct the concrete `VmCtx` corresponding to `SporosProvLen.linit pr msize inp`. -/
def mkVmReadyCtx (pr : Sporos.Prog) (msize : Nat) (inp : List Nat) : VmCtx :=
  let cLen := readyCodeLen pr
  let cData := encodeFlatProg pr ++ List.replicate (cLen - (encodeFlatProg pr).length) 0
  let locPg := readyLocPages pr
  let mwLen := readyMemwLen msize
  let mpLen := readyMempLen msize
  let inPg := readyInpPages inp
  let b_code1 := 0x1000000
  let b_code2 := b_code1 + cLen
  let b_code3 := b_code2 + cLen
  let b_ds := b_code3 + cLen
  let b_aux := b_ds + 4096
  let b_rs := b_aux + 4096
  let b_loc := b_rs + 4096
  let b_memw := b_loc + locPg
  let b_memp := b_memw + mwLen
  let b_tabs := b_memp + mpLen
  let b_kv := b_tabs + 4096
  let b_inp := b_kv + 4096
  let b_out := b_inp + inPg
  let nlocs := readyInitLocals pr
  let locCap := locPg / 16
  let ctxSlots : List Nat :=
    [ b_ds, 0, 256,
      b_aux, 0, 256,
      b_rs, 1, 128,
      b_loc, nlocs, locCap,
      b_memw, b_memp, msize, pr.data.flatten.length,
      b_tabs, 0, 512,
      b_kv, 0, 128,
      b_inp, inp.length, 0,
      b_out, 0, 4096,
      0, b_code1, b_code2, b_code3 ]
  let ctxData := ctxSlots.flatMap bytes8 ++ List.replicate (ctxSize - 256) 0
  let rsData := ([pr.main, 0, 0, nlocs].flatMap bytes8) ++ List.replicate (4096 - 32) 0
  let mwData := (List.range mwLen).map fun a => ((pr.data.flatten)[a]?.getD 0) % 256
  let inpData := inp.map (· % 256) ++ List.replicate (inPg - inp.length) 0
  { stk := ⟨0x700000, List.replicate 4096 0, true⟩
    sp := 0x700f00
    ctxReg := ⟨ctxBase, ctxData, true⟩
    code1 := ⟨b_code1, cData, false⟩
    code2 := ⟨b_code2, cData, false⟩
    code3 := ⟨b_code3, cData, false⟩
    dsReg := ⟨b_ds, List.replicate 4096 0, true⟩
    dsCap := 256
    auxReg := ⟨b_aux, List.replicate 4096 0, true⟩
    auxCap := 256
    rsReg := ⟨b_rs, rsData, true⟩
    rsCap := 128
    locReg := ⟨b_loc, List.replicate locPg 0, true⟩
    locCap := locCap
    memwReg := ⟨b_memw, mwData, true⟩
    mempReg := ⟨b_memp, List.replicate mpLen 0, true⟩
    tabsReg := ⟨b_tabs, List.replicate 4096 0, true⟩
    tabsCap := 512
    kvReg := ⟨b_kv, List.replicate 4096 0, true⟩
    kvLen := 0
    kvCap := 128
    inpReg := ⟨b_inp, inpData, false⟩
    inpLen := inp.length
    outReg := ⟨b_out, List.replicate 4096 0, true⟩
    outCap := 4096 }

/-- Sorted list of the 16 ready-state memory mappings. -/
def mkVmReadyMem (vc : VmCtx) : Mem :=
  [codeRegion, vc.ctxReg, vc.stk, vc.code1, vc.code2, vc.code3,
   vc.dsReg, vc.auxReg, vc.rsReg, vc.locReg, vc.memwReg, vc.mempReg,
   vc.tabsReg, vc.kvReg, vc.inpReg, vc.outReg]

/-- Concrete post-boot `X86Vm.St` at `SporosVmBin.L_vm_loop`. -/
def mkVmReadyState (pr : Sporos.Prog) (msize : Nat) (inp : List Nat)
    (oracle : List X86Env.Choice) : X86Vm.St :=
  let vc := mkVmReadyCtx pr msize inp
  { regs := ((List.replicate 16 (some 0)).set 4 (some vc.sp)).set 15 (some ctxBase)
    rip := SporosVmBin.L_vm_loop
    fl := X86Int.Flags.undef
    mem := mkVmReadyMem vc
    k := { files := [], fds := [], stdin := [], out := [], oracle := oracle } }

private theorem nodup_regions_of_nodup_readyMem (vc : VmCtx)
    (h : ((mkVmReadyMem vc).map Region.base).Nodup) :
    (vc.regions.map Region.base).Nodup := by
  simp only [VmCtx.regions, mkVmReadyMem, List.map_cons, List.map_nil, List.nodup_cons,
    List.mem_cons, List.not_mem_nil, or_false] at h ⊢
  rcases h with ⟨h0, h1, h2, hrest⟩
  refine ⟨?_, ?_, ?_, hrest⟩
  · intro hm; apply h0; rcases hm with hm | hm | hm
    · exact Or.inr (Or.inl hm)
    · exact Or.inl hm
    · exact Or.inr (Or.inr hm)
  · intro hm; rcases hm with hm | hm
    · exact h1 (Or.inl hm.symm)
    · exact h2 hm
  · intro hm; exact h1 (Or.inr hm)

private theorem mem_mkVmReadyMem_of_mem_regions (vc : VmCtx) :
    ∀ r ∈ vc.regions, r ∈ mkVmReadyMem vc := by
  intro r hr
  simp only [VmCtx.regions, mkVmReadyMem, List.mem_cons, List.not_mem_nil, or_false] at hr ⊢
  rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact Or.inl rfl
  · exact Or.inr (Or.inr (Or.inl rfl))
  · exact Or.inr (Or.inl rfl)
  · exact Or.inr (Or.inr (Or.inr (Or.inl rfl)))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))))))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))))))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl)))))))))))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl rfl))))))))))))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr rfl))))))))))))))

/-! ### Main non-vacuity theorem: `mkVmReadyState` satisfies `VmRel` -/

theorem mkVmReadyCtx_wf (pr : Sporos.Prog) (msize : Nat) (inp : List Nat)
    (oracle : List X86Env.Choice)
    (hfit : FlatProgFit pr) (hmsize : msize < 2 ^ 44)
    (hend : readyEndAddr pr msize inp ≤ 2 ^ 47) :
    (mkVmReadyCtx pr msize inp).wf pr (SporosProvLen.linit pr msize inp)
      (mkVmReadyState pr msize inp oracle) := by
  let vc := mkVmReadyCtx pr msize inp
  let s := mkVmReadyState pr msize inp oracle
  let ls := SporosProvLen.linit pr msize inp
  have hcLen_pos : 0 < readyCodeLen pr := pages_pos (by omega)
  have hcLen_ge : (encodeFlatProg pr).length ≤ readyCodeLen pr :=
    Nat.le_trans (Nat.le_max_left _ _) (pages_ge _)
  have hlocPg_pos : 0 < readyLocPages pr := pages_pos (by omega)
  have hlocPg_ge : readyInitLocals pr * 16 ≤ readyLocPages pr :=
    Nat.le_trans (Nat.le_max_left _ _) (pages_ge _)
  have hlocPg_4k : 4096 ≤ readyLocPages pr :=
    Nat.le_trans (Nat.le_max_right _ _) (pages_ge _)
  have hlocPg_mod : readyLocPages pr % 4096 = 0 := pages_mod _
  have hlocCap_mul : readyLocPages pr / 16 * 16 = readyLocPages pr := by omega
  have hmwLen_pos : 0 < readyMemwLen msize := pages_pos (by omega)
  have hmwLen_ge : msize ≤ readyMemwLen msize := Nat.le_trans (Nat.le_max_left _ _) (pages_ge _)
  have hmpLen_pos : 0 < readyMempLen msize := pages_pos (by omega)
  have hmpLen_ge : msize * 8 ≤ readyMempLen msize := by
    unfold readyMempLen; have := pages_ge (8 * max msize 1); omega
  have hinPg_pos : 0 < readyInpPages inp := pages_pos (by omega)
  have hinPg_ge : inp.length ≤ readyInpPages inp := Nat.le_trans (Nat.le_max_left _ _) (pages_ge _)
  have hcodeRegion_len : codeRegion.data.length = 12945 := vmBytes_len_eq
  have hctx_len : vc.ctxReg.data.length = 4096 := by
    simp only [vc, mkVmReadyCtx, List.length_append, flatMap_bytes8_length, List.length_replicate,
      List.length_cons, List.length_nil, ctxSize]
  have hcode1_len : vc.code1.data.length = readyCodeLen pr := by
    simp only [vc, mkVmReadyCtx, List.length_append, List.length_replicate]; omega
  have hrs_len : vc.rsReg.data.length = 4096 := by
    simp only [vc, mkVmReadyCtx, List.length_append, flatMap_bytes8_length, List.length_replicate,
      List.length_cons, List.length_nil]
  have hmw_len : vc.memwReg.data.length = readyMemwLen msize := by
    simp only [vc, mkVmReadyCtx, List.length_map, List.length_range]
  have hinp_len : vc.inpReg.data.length = readyInpPages inp := by
    simp only [vc, mkVmReadyCtx, List.length_append, List.length_map, List.length_replicate]; omega
  have hstk_len : vc.stk.data.length = 4096 := List.length_replicate
  have hds_len : vc.dsReg.data.length = 4096 := List.length_replicate
  have haux_len : vc.auxReg.data.length = 4096 := List.length_replicate
  have hloc_len : vc.locReg.data.length = readyLocPages pr := List.length_replicate
  have hmp_len : vc.mempReg.data.length = readyMempLen msize := List.length_replicate
  have htb_len : vc.tabsReg.data.length = 4096 := List.length_replicate
  have hkv_len : vc.kvReg.data.length = 4096 := List.length_replicate
  have hout_len : vc.outReg.data.length = 4096 := List.length_replicate
  have hmem_chain : ∀ i, i + 1 < (mkVmReadyMem vc).length →
      (mkVmReadyMem vc)[i]!.hi ≤ (mkVmReadyMem vc)[i + 1]!.base := by
    intro i hi
    have hi15 : i < 15 := by simp only [mkVmReadyMem, List.length_cons, List.length_nil] at hi; omega
    have hhi_code : codeRegion.hi = baseAddr + 12945 := by simp only [Region.hi, hcodeRegion_len]; rfl
    have hhi_ctx : vc.ctxReg.hi = ctxBase + 4096 := by simp only [Region.hi, hctx_len]; rfl
    have hhi_stk : vc.stk.hi = 0x700000 + 4096 := by simp only [Region.hi, hstk_len]; rfl
    have hhi_c1 : vc.code1.hi = vc.code1.base + readyCodeLen pr := by simp only [Region.hi, hcode1_len]
    have hhi_c2 : vc.code2.hi = vc.code2.base + readyCodeLen pr := by simp only [Region.hi, vc, mkVmReadyCtx, List.length_append, List.length_replicate]; omega
    have hhi_c3 : vc.code3.hi = vc.code3.base + readyCodeLen pr := by simp only [Region.hi, vc, mkVmReadyCtx, List.length_append, List.length_replicate]; omega
    have hhi_ds : vc.dsReg.hi = vc.dsReg.base + 4096 := by simp only [Region.hi, hds_len]
    have hhi_aux : vc.auxReg.hi = vc.auxReg.base + 4096 := by simp only [Region.hi, haux_len]
    have hhi_rs : vc.rsReg.hi = vc.rsReg.base + 4096 := by simp only [Region.hi, hrs_len]
    have hhi_loc : vc.locReg.hi = vc.locReg.base + readyLocPages pr := by simp only [Region.hi, hloc_len]
    have hhi_mw : vc.memwReg.hi = vc.memwReg.base + readyMemwLen msize := by simp only [Region.hi, hmw_len]
    have hhi_mp : vc.mempReg.hi = vc.mempReg.base + readyMempLen msize := by simp only [Region.hi, hmp_len]
    have hhi_tb : vc.tabsReg.hi = vc.tabsReg.base + 4096 := by simp only [Region.hi, htb_len]
    have hhi_kv : vc.kvReg.hi = vc.kvReg.base + 4096 := by simp only [Region.hi, hkv_len]
    have hhi_in : vc.inpReg.hi = vc.inpReg.base + readyInpPages inp := by simp only [Region.hi, hinp_len]
    match i, hi15 with
    | 0, _ => simp only [mkVmReadyMem, List.getElem!_eq_getElem?_getD, List.getElem?_cons_zero,
        List.getElem?_cons_succ, Option.getD_some, hhi_code]; change baseAddr + 12945 ≤ ctxBase; decide
    | 1, _ => simp only [mkVmReadyMem, List.getElem!_eq_getElem?_getD, List.getElem?_cons_zero,
        List.getElem?_cons_succ, Option.getD_some, hhi_ctx]; change ctxBase + 4096 ≤ 0x700000; decide
    | 2, _ => simp only [mkVmReadyMem, List.getElem!_eq_getElem?_getD, List.getElem?_cons_zero,
        List.getElem?_cons_succ, Option.getD_some, hhi_stk]; change 0x700000 + 4096 ≤ 0x1000000; decide
    | 3, _ => simp only [mkVmReadyMem, List.getElem!_eq_getElem?_getD, List.getElem?_cons_zero,
        List.getElem?_cons_succ, Option.getD_some, hhi_c1]; exact Nat.le_refl _
    | 4, _ => simp only [mkVmReadyMem, List.getElem!_eq_getElem?_getD, List.getElem?_cons_zero,
        List.getElem?_cons_succ, Option.getD_some, hhi_c2]; exact Nat.le_refl _
    | 5, _ => simp only [mkVmReadyMem, List.getElem!_eq_getElem?_getD, List.getElem?_cons_zero,
        List.getElem?_cons_succ, Option.getD_some, hhi_c3]; exact Nat.le_refl _
    | 6, _ => simp only [mkVmReadyMem, List.getElem!_eq_getElem?_getD, List.getElem?_cons_zero,
        List.getElem?_cons_succ, Option.getD_some, hhi_ds]; exact Nat.le_refl _
    | 7, _ => simp only [mkVmReadyMem, List.getElem!_eq_getElem?_getD, List.getElem?_cons_zero,
        List.getElem?_cons_succ, Option.getD_some, hhi_aux]; exact Nat.le_refl _
    | 8, _ => simp only [mkVmReadyMem, List.getElem!_eq_getElem?_getD, List.getElem?_cons_zero,
        List.getElem?_cons_succ, Option.getD_some, hhi_rs]; exact Nat.le_refl _
    | 9, _ => simp only [mkVmReadyMem, List.getElem!_eq_getElem?_getD, List.getElem?_cons_zero,
        List.getElem?_cons_succ, Option.getD_some, hhi_loc]; exact Nat.le_refl _
    | 10, _ => simp only [mkVmReadyMem, List.getElem!_eq_getElem?_getD, List.getElem?_cons_zero,
        List.getElem?_cons_succ, Option.getD_some, hhi_mw]; exact Nat.le_refl _
    | 11, _ => simp only [mkVmReadyMem, List.getElem!_eq_getElem?_getD, List.getElem?_cons_zero,
        List.getElem?_cons_succ, Option.getD_some, hhi_mp]; exact Nat.le_refl _
    | 12, _ => simp only [mkVmReadyMem, List.getElem!_eq_getElem?_getD, List.getElem?_cons_zero,
        List.getElem?_cons_succ, Option.getD_some, hhi_tb]; exact Nat.le_refl _
    | 13, _ => simp only [mkVmReadyMem, List.getElem!_eq_getElem?_getD, List.getElem?_cons_zero,
        List.getElem?_cons_succ, Option.getD_some, hhi_kv]; exact Nat.le_refl _
    | 14, _ => simp only [mkVmReadyMem, List.getElem!_eq_getElem?_getD, List.getElem?_cons_zero,
        List.getElem?_cons_succ, Option.getD_some, hhi_in]; exact Nat.le_refl _
  have hmem_pos : ∀ u ∈ mkVmReadyMem vc, 0 < u.data.length := by
    intro u hu
    simp only [mkVmReadyMem, List.mem_cons, List.not_mem_nil, or_false] at hu
    rcases hu with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · rw [hcodeRegion_len]; decide
    · rw [hctx_len]; decide
    · rw [hstk_len]; decide
    · rw [hcode1_len]; exact hcLen_pos
    · simp only [vc, mkVmReadyCtx, List.length_append, List.length_replicate]; omega
    · simp only [vc, mkVmReadyCtx, List.length_append, List.length_replicate]; omega
    · rw [hds_len]; decide
    · rw [haux_len]; decide
    · rw [hrs_len]; decide
    · rw [hloc_len]; exact hlocPg_pos
    · rw [hmw_len]; exact hmwLen_pos
    · rw [hmp_len]; exact hmpLen_pos
    · rw [htb_len]; decide
    · rw [hkv_len]; decide
    · rw [hinp_len]; exact hinPg_pos
    · rw [hout_len]; decide
  have hmem_wf : (mkVmReadyMem vc).wf := by
    refine ⟨?_, pairwise_of_chain _ hmem_chain (fun r _ => by simp only [Region.hi]; omega)⟩
    intro r hr
    have hmc : readyCodeLen pr % X86Env.page = 0 := pages_mod _
    have hml : readyLocPages pr % X86Env.page = 0 := pages_mod _
    have hmw : readyMemwLen msize % X86Env.page = 0 := pages_mod _
    have hmp : readyMempLen msize % X86Env.page = 0 := pages_mod _
    have hmi : readyInpPages inp % X86Env.page = 0 := pages_mod _
    have hp4 : X86Env.page = 4096 := rfl
    have hcb : ctxBase = 0x600000 := rfl
    have hba : baseAddr = 0x400000 := rfl
    simp only [mkVmReadyMem, List.mem_cons, List.not_mem_nil, or_false] at hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals (simp only [codeRegion, vc, mkVmReadyCtx, hp4, hcb, hba] at hmc hml hmw hmp hmi ⊢; try omega)
  have hcm : CodeMem s.mem := ⟨by simp [s, mkVmReadyState, mkVmReadyMem], hmem_wf⟩
  have hbl : BytesLt s.mem := by
    intro r hr b hb
    simp only [s, mkVmReadyState, mkVmReadyMem, List.mem_cons, List.not_mem_nil, or_false] at hr
    rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · exact vmBytes_lt_256 b hb
    · simp only [mkVmReadyCtx, List.mem_append, List.mem_replicate] at hb
      rcases hb with hb | ⟨_, rfl⟩
      · exact flatMap_bytes8_lt_256 _ b hb
      · decide
    · simp only [mkVmReadyCtx, List.mem_replicate] at hb; rw [hb.2]; decide
    · simp only [mkVmReadyCtx, encodeFlatProg, List.mem_append, List.mem_replicate] at hb
      rcases hb with hb | ⟨_, rfl⟩
      · exact flatMap_bytes8_lt_256 _ b hb
      · decide
    · simp only [mkVmReadyCtx, encodeFlatProg, List.mem_append, List.mem_replicate] at hb
      rcases hb with hb | ⟨_, rfl⟩
      · exact flatMap_bytes8_lt_256 _ b hb
      · decide
    · simp only [mkVmReadyCtx, encodeFlatProg, List.mem_append, List.mem_replicate] at hb
      rcases hb with hb | ⟨_, rfl⟩
      · exact flatMap_bytes8_lt_256 _ b hb
      · decide
    · simp only [mkVmReadyCtx, List.mem_replicate] at hb; rw [hb.2]; decide
    · simp only [mkVmReadyCtx, List.mem_replicate] at hb; rw [hb.2]; decide
    · simp only [mkVmReadyCtx, List.mem_append, List.mem_replicate] at hb
      rcases hb with hb | ⟨_, rfl⟩
      · exact flatMap_bytes8_lt_256 _ b hb
      · decide
    · simp only [mkVmReadyCtx, List.mem_replicate] at hb; rw [hb.2]; decide
    · simp only [mkVmReadyCtx, List.mem_map] at hb
      obtain ⟨_, _, rfl⟩ := hb; exact Nat.mod_lt _ (by decide)
    · simp only [mkVmReadyCtx, List.mem_replicate] at hb; rw [hb.2]; decide
    · simp only [mkVmReadyCtx, List.mem_replicate] at hb; rw [hb.2]; decide
    · simp only [mkVmReadyCtx, List.mem_replicate] at hb; rw [hb.2]; decide
    · simp only [mkVmReadyCtx, List.mem_append, List.mem_map, List.mem_replicate] at hb
      rcases hb with ⟨_, _, rfl⟩ | ⟨_, rfl⟩
      · exact Nat.mod_lt _ (by decide)
      · decide
    · simp only [mkVmReadyCtx, List.mem_replicate] at hb; rw [hb.2]; decide
  have hregs : RegsOk s := by
    refine ⟨by simp [s, mkVmReadyState], ?_⟩
    intro r hr16 _
    simp only [s, mkVmReadyState, X86Int.St.reg]
    by_cases h15 : r = 15
    · subst h15; rfl
    · rw [List.getElem?_set_ne (Ne.symm h15)]
      by_cases h4 : r = 4
      · subst h4; rfl
      · rw [List.getElem?_set_ne (Ne.symm h4), List.getElem?_replicate, if_pos hr16]; rfl
  have hnodup_mem : ((mkVmReadyMem vc).map Region.base).Nodup :=
    nodup_of_strict_chain _ hmem_chain hmem_pos
  have hnodup_reg : (vc.regions.map Region.base).Nodup :=
    nodup_regions_of_nodup_readyMem vc hnodup_mem
  have hmem_all : ∀ r ∈ vc.regions, r ∈ s.mem :=
    mem_mkVmReadyMem_of_mem_regions vc
  have hhi_end : readyEndAddr pr msize inp ≤ 2 ^ 47 := hend
  simp only [readyEndAddr] at hhi_end
  have hW_47 : 2 ^ 47 < X86Env.W := by change 2 ^ 47 < 2 ^ 64; omega
  have hctx_hi : vc.ctxReg.hi ≤ 2 ^ 47 := by rw [Region.hi, hctx_len]; change ctxBase + 4096 ≤ 2 ^ 47; decide
  have hc1_hi : vc.code1.hi ≤ 2 ^ 47 := by rw [Region.hi, hcode1_len]; change 0x1000000 + readyCodeLen pr ≤ 2 ^ 47; omega
  have hc2_hi : vc.code2.hi ≤ 2 ^ 47 := by
    simp only [Region.hi, vc, mkVmReadyCtx, List.length_append, List.length_replicate]; omega
  have hc3_hi : vc.code3.hi ≤ 2 ^ 47 := by
    simp only [Region.hi, vc, mkVmReadyCtx, List.length_append, List.length_replicate]; omega
  have hflat1 : FlatCodeOk pr vc.code1 :=
    ⟨by rw [hcode1_len]; exact hcLen_ge, hc1_hi, by
      simp only [vc, mkVmReadyCtx, List.take_append_of_le_length (Nat.le_refl _), List.take_length]⟩
  have hflat2 : FlatCodeOk pr vc.code2 :=
    ⟨by simp only [vc, mkVmReadyCtx, List.length_append, List.length_replicate]; omega, hc2_hi, by
      simp only [vc, mkVmReadyCtx, List.take_append_of_le_length (Nat.le_refl _), List.take_length]⟩
  have hflat3 : FlatCodeOk pr vc.code3 :=
    ⟨by simp only [vc, mkVmReadyCtx, List.length_append, List.length_replicate]; omega, hc3_hi, by
      simp only [vc, mkVmReadyCtx, List.take_append_of_le_length (Nat.le_refl _), List.take_length]⟩
  have hW_nlocs : readyInitLocals pr < X86Env.W := by omega
  let cLen := readyCodeLen pr
  let locPg := readyLocPages pr
  let mwLen := readyMemwLen msize
  let mpLen := readyMempLen msize
  let inPg := readyInpPages inp
  let b_code1 := 0x1000000
  let b_code2 := b_code1 + cLen
  let b_code3 := b_code2 + cLen
  let b_ds := b_code3 + cLen
  let b_aux := b_ds + 4096
  let b_rs := b_aux + 4096
  let b_loc := b_rs + 4096
  let b_memw := b_loc + locPg
  let b_memp := b_memw + mwLen
  let b_tabs := b_memp + mpLen
  let b_kv := b_tabs + 4096
  let b_inp := b_kv + 4096
  let b_out := b_inp + inPg
  let qs : List Nat :=
    [ b_ds, 0, 256, b_aux, 0, 256, b_rs, 1, 128, b_loc, readyInitLocals pr, locPg / 16,
      b_memw, b_memp, msize, pr.data.flatten.length, b_tabs, 0, 512, b_kv, 0, 128,
      b_inp, inp.length, 0, b_out, 0, 4096, 0, b_code1, b_code2, b_code3 ]
  have hqs_lt : ∀ x ∈ qs, x < X86Env.W := by
    intro x hx
    simp only [qs, List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals first
      | exact hfit.2.2.2.2.1
      | exact hW_nlocs
      | omega
  have hslot : ∀ (j : Nat) (hj : j < 32),
      leVal ((vc.ctxReg.data.drop (j * 8)).take 8) = qs[j]! := by
    intro j hj
    have hlt : qs[j]! < X86Env.W := hqs_lt _ (by
      rw [List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem (by simpa [qs] using hj)]
      exact List.getElem_mem _)
    exact leVal_flatMap_bytes8_pad qs (ctxSize - 256) (by simpa [qs] using hj) hlt
  have hrs_rel : RsLocRel vc.rsReg.data vc.locReg.data (rsLocBase ls.rs) ls.rs := by
    have hlb : rsLocBase ls.rs = readyInitLocals pr := by
      simp [ls, SporosProvLen.linit, rsLocBase, readyInitLocals]; rfl
    have hlr : (SporosProvLen.linit pr msize inp).rs =
        [⟨pr.main, 0, List.replicate (readyInitLocals pr) ⟨0, 0⟩, 0, 0⟩] := rfl
    rw [hlb, hlr]
    simp only [RsLocRel, rsLocBase, List.map_nil, List.foldl_nil, List.length_nil, List.length_replicate,
      Nat.zero_add, Nat.zero_mul]
    have hW0 : 0 < X86Env.W := by change 0 < 2 ^ 64; omega
    refine ⟨trivial, by rw [hrs_len]; decide, by rw [hloc_len]; omega, ?_, ?_, ?_, ?_, ?_, trivial⟩
    · exact leVal_flatMap_bytes8_pad [pr.main, 0, 0, readyInitLocals pr] (4096 - 32) (j := 0) (by simp) hfit.1
    · exact leVal_flatMap_bytes8_pad [pr.main, 0, 0, readyInitLocals pr] (4096 - 32) (j := 1) (by simp) hW0
    · exact leVal_flatMap_bytes8_pad [pr.main, 0, 0, readyInitLocals pr] (4096 - 32) (j := 2) (by simp) hW0
    · exact leVal_flatMap_bytes8_pad [pr.main, 0, 0, readyInitLocals pr] (4096 - 32) (j := 3) (by simp) hW_nlocs
    · intro j hj
      simp only [vc, mkVmReadyCtx, drop_take_replicate_zero]
      have hget : (List.replicate (readyInitLocals pr) (⟨0, 0⟩ : SporosProvLen.LVal))[j]! = ⟨0, 0⟩ := by
        rw [List.getElem!_eq_getElem?_getD, List.getElem?_replicate, if_pos hj]; rfl
      rw [hget]; exact ⟨rfl, rfl⟩
  have hmem_rel : MemRel vc.memwReg.data vc.mempReg.data ls.msize ls.mem := by
    refine ⟨by rw [hmw_len]; exact hmwLen_ge, by rw [hmp_len]; exact hmpLen_ge, ?_⟩
    intro x hx
    have hxl : x < readyMemwLen msize := by
      simp only [ls, SporosProvLen.linit] at hx; omega
    refine ⟨?_, ?_⟩
    · simp only [vc, mkVmReadyCtx, ls, SporosProvLen.linit, SporosProvLen.lloadData,
        List.getElem?_map, List.getElem?_range hxl, Option.map_some]
      cases (pr.data.flatten)[x]? with
      | none => rfl
      | some b => simp
    · simp only [vc, mkVmReadyCtx, ls, SporosProvLen.linit, SporosProvLen.lloadData,
        drop_take_replicate_zero]
      cases (pr.data.flatten)[x]? <;> simp
  have hinp_rel : InpRel vc.inpReg.data ls.ipos vc.inpLen ls.inp := by
    refine ⟨by simp only [ls, SporosProvLen.linit, vc, mkVmReadyCtx, Nat.zero_add], by rw [hinp_len]; exact hinPg_ge, ?_⟩
    intro j hj
    simp only [ls, SporosProvLen.linit, vc, mkVmReadyCtx, Nat.zero_add] at hj ⊢
    rw [List.getElem?_append_left (by simp [hj]), List.getElem?_map,
      List.getElem?_eq_getElem hj, Option.map_some, List.getElem!_eq_getElem?_getD,
      List.getElem?_eq_getElem hj, Option.getD_some]
  have hlb_eq : rsLocBase ls.rs = readyInitLocals pr := by
    simp [ls, SporosProvLen.linit, rsLocBase, readyInitLocals]; rfl
  have hstk_hi : vc.stk.hi = 0x700000 + 4096 := by rw [Region.hi, hstk_len]; rfl
  have hds_hi : vc.dsReg.hi = b_ds + 4096 := by rw [Region.hi, hds_len]; rfl
  have haux_hi : vc.auxReg.hi = b_aux + 4096 := by rw [Region.hi, haux_len]; rfl
  have hrs_hi : vc.rsReg.hi = b_rs + 4096 := by rw [Region.hi, hrs_len]; rfl
  have hloc_hi : vc.locReg.hi = b_loc + locPg := by rw [Region.hi, hloc_len]; rfl
  have hmw_hi : vc.memwReg.hi = b_memw + mwLen := by rw [Region.hi, hmw_len]; rfl
  have hmp_hi : vc.mempReg.hi = b_memp + mpLen := by rw [Region.hi, hmp_len]; rfl
  have htb_hi : vc.tabsReg.hi = b_tabs + 4096 := by rw [Region.hi, htb_len]; rfl
  have hkv_hi : vc.kvReg.hi = b_kv + 4096 := by rw [Region.hi, hkv_len]; rfl
  have hinp_hi : vc.inpReg.hi = b_inp + inPg := by rw [Region.hi, hinp_len]; rfl
  have hout_hi : vc.outReg.hi = b_out + 4096 := by rw [Region.hi, hout_len]; rfl
  have hs4 : s.reg 4 = some vc.sp := by
    change (((List.replicate 16 (some 0)).set 4 (some 0x700f00)).set 15 (some ctxBase))[4]?.join = some 0x700f00
    rfl
  have hs15 : s.reg 15 = some ctxBase := by
    change (((List.replicate 16 (some 0)).set 4 (some 0x700f00)).set 15 (some ctxBase))[15]?.join = some ctxBase
    rfl
  exact ⟨hcm, hbl, hmem_pos, hregs, hs4, hs15,
    hnodup_reg, hmem_all,
    rfl, by change 0x700000 + 32 ≤ 0x700f00; decide,
    by rw [hstk_hi]; change 0x700f00 + 8 ≤ 0x700000 + 4096; decide,
    by change 0x700f00 < 2 ^ 64; omega,
    rfl, rfl, by rw [hctx_len]; decide, hctx_hi,
    hflat1, hflat2, hflat3,
    rfl, by change 4 ≤ 256; decide, by rw [hds_len]; rfl, by change 256 * 16 % X86Env.page = 0; decide, by rw [hds_hi]; omega,
    rfl, by change 4 ≤ 256; decide, by rw [haux_len]; rfl, by change 256 * 16 % X86Env.page = 0; decide, by rw [haux_hi]; omega,
    rfl, by change 0 < 128; decide, by rw [hrs_len]; rfl, by change 128 * 32 % X86Env.page = 0; decide, by rw [hrs_hi]; omega,
    rfl, by change 0 < locPg / 16; omega, by rw [hloc_len]; change locPg = locPg / 16 * 16; omega,
    by change (locPg / 16 * 16) % 4096 = 0; omega, by rw [hloc_hi]; omega,
    rfl, hmsize, hmw_len, by rw [hmw_hi]; omega,
    rfl, by rw [hmp_len]; rfl, by rw [hmp_hi]; omega,
    rfl, by change 0 < 512; decide, by rw [htb_len]; rfl, by change 512 * 8 % X86Env.page = 0; decide, by rw [htb_hi]; omega,
    rfl, by change 0 < 128; decide, by rw [hkv_len]; rfl, by change 128 * 32 % X86Env.page = 0; decide, by rw [hkv_hi]; omega,
    by change 0 ≤ 128; decide,
    by rw [hinp_hi]; omega,
    rfl, by change 8 ≤ 4096; decide, by rw [hout_len]; rfl, by change 4096 % X86Env.page = 0; decide, by rw [hout_hi]; omega,
    hslot 0 (by decide), hslot 1 (by decide), hslot 2 (by decide),
    hslot 3 (by decide), hslot 4 (by decide), hslot 5 (by decide),
    hslot 6 (by decide), hslot 7 (by decide), hslot 8 (by decide),
    hslot 9 (by decide), by rw [hlb_eq]; exact hslot 10 (by decide),
    hslot 11 (by decide), hslot 12 (by decide), hslot 13 (by decide),
    hslot 14 (by decide), hslot 15 (by decide),
    hslot 16 (by decide), hslot 17 (by decide), hslot 18 (by decide),
    hslot 19 (by decide), hslot 20 (by decide), hslot 21 (by decide),
    hslot 22 (by decide), hslot 23 (by decide), hslot 24 (by decide),
    hslot 25 (by decide), hslot 26 (by decide), hslot 27 (by decide),
    hslot 28 (by decide), hslot 29 (by decide), hslot 30 (by decide),
    hslot 31 (by decide),
    ⟨rfl, by rw [hds_len]; simp [SporosProvLen.linit], by intro i hi; simp [SporosProvLen.linit] at hi⟩, by change 0 ≤ 256; decide,
    ⟨rfl, by rw [haux_len]; simp [SporosProvLen.linit], by intro i hi; simp [SporosProvLen.linit] at hi⟩, by change 0 ≤ 256; decide,
    hrs_rel, by change 1 ≤ 128; decide,
    by rw [hlb_eq]; change readyInitLocals pr ≤ locPg / 16; omega,
    hmem_rel, ⟨by simp [SporosProvLen.linit], by intro t ht; simp [SporosProvLen.linit] at ht⟩,
    by simp [SporosProvLen.linit],
    hinp_rel,
    ⟨rfl, by rw [hout_len]; simp [SporosProvLen.linit], by intro j hj; simp [SporosProvLen.linit] at hj⟩, by simp [SporosProvLen.linit],
    SporosProvLen.linit_wordsLt pr msize inp, hfit⟩

/-- Non-vacuity theorem: `mkVmReadyState` satisfies `VmRel` with `SporosProvLen.linit`. -/
theorem mkVmReadyState_rel (pr : Sporos.Prog) (msize : Nat) (inp : List Nat)
    (oracle : List X86Env.Choice)
    (hfit : FlatProgFit pr) (hmsize : msize < 2 ^ 44)
    (hend : readyEndAddr pr msize inp ≤ 2 ^ 47) :
    VmRel pr (SporosProvLen.linit pr msize inp) (mkVmReadyState pr msize inp oracle) :=
  ⟨rfl, mkVmReadyCtx pr msize inp, mkVmReadyCtx_wf pr msize inp oracle hfit hmsize hend⟩

end SporosVmReady
