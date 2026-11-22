import Bootstrap.SporosVmPrimBase

/-!
# Native x86-64 VM Memory & Heap Primitive Proofs (`P_ld8`..`P_strhash`)

Simulation proofs over `X86Vm` for the 15 memory and heap primitives:
- byte/word loads & stores (`P_ld8`, `P_ld16`, `P_ld32`, `P_ld64`,
  `P_st8`, `P_st16`, `P_st32`, `P_st64`),
- block & string operations (`P_copy`, `P_fill`, `P_mcmp`, `P_streq`, `P_strhash`),
- arena heap pointer operations (`P_alloc`, `P_mark`, `P_release`).

Note on Track A architecture: Root A is MUSE's hand-written checker
(`bootstrap/roots/sporos/sporos-stage1.hex`), `spr1_check` is an independent second check,
and `sporos_vm` is the post-admission native x86-64 VM paired with `Sporos.decode` +
`Sporos.checkProg`.
-/

namespace SporosVmPrimMem

open X86Env X86Int X86Vm SporosVmCore SporosVmExec SporosVmSimBase SporosVmPrimBase

local instance : Inhabited SporosProvLen.LVal := ⟨⟨0, 0⟩⟩
local instance : Inhabited SporosProvLen.LFrame := ⟨⟨0, 0, [], 0, 0⟩⟩

/-- Lookup table chunk `k` from `SporosVmBin.vmTableChunks`. -/
def vmTableChunk (k : Nat) : List (Nat × X86Vm.Ins × Nat) :=
  match SporosVmBin.vmTableChunks[k]? with
  | some (_, c) => c
  | none => []

theorem step_at_vc (k : Nat) {s : X86Vm.St} {rip : Nat} {ins : X86Vm.Ins} {len : Nat}
    (hrip : s.rip = rip) (he : (rip, ins, len) ∈ vmTableChunk k) :
    X86Vm.step vmImage s = X86Vm.exec ins len s := by
  unfold vmTableChunk at he
  cases hc : SporosVmBin.vmTableChunks[k]? with
  | none => rw [hc] at he; cases he
  | some p =>
    rw [hc] at he
    exact step_at_chunk k hrip hc he

/-! ### Pure `MemRel`, FNV-1a, boolean range, and byte-slice helpers -/

theorem memRel_ld_val : ∀ {mwData mpData : List Nat} {msize a n : Nat} {mem : SporosProvLen.LMem},
    MemRel mwData mpData msize mem → a + n ≤ msize →
    leVal ((mwData.drop a).take n) =
      (List.range n).foldr (fun i acc => acc * 256 + (mem (a + i)).w % 256) 0
  | _, _, _, _, 0, _, _, _ => by simp [leVal]
  | mwData, mpData, msize, a, k + 1, mem, hm, hle => by
    have ha_lt : a < msize := by omega
    have ha_len : a < mwData.length := by have := hm.1; omega
    have ha_mem : mwData[a]? = some ((mem a).w % 256) := (hm.2.2 a ha_lt).1
    have ha_get : mwData[a]'ha_len = (mem a).w % 256 := by
      have hget := List.getElem?_eq_getElem ha_len
      rw [hget] at ha_mem
      exact Option.some.inj ha_mem
    have hdrop : (mwData.drop a).take (k + 1) =
        (mwData[a]'ha_len) :: ((mwData.drop (a + 1)).take k) :=
      (list_take_drop_succ mwData a k ha_len).symm
    have hih := memRel_ld_val hm (a := a + 1) (n := k) (by omega)
    rw [hdrop, leVal, ha_get, hih, List.range_succ_eq_map, List.foldr_cons, List.foldr_map]
    have hfold : (List.range k).foldr (fun x acc => acc * 256 + (mem (a + (x + 1))).w % 256) 0 =
        (List.range k).foldr (fun i acc => acc * 256 + (mem (a + 1 + i)).w % 256) 0 := by
      congr 1; funext x acc; rw [Nat.add_assoc, Nat.add_comm x 1]
    rw [hfold]
    simp only [Nat.add_zero]
    omega

theorem foldr_ld_lt_pow : ∀ (mem : SporosProvLen.LMem) (a n : Nat),
    (List.range n).foldr (fun i acc => acc * 256 + (mem (a + i)).w % 256) 0 < 256 ^ n
  | _, _, 0 => by simp
  | mem, a, k + 1 => by
    rw [List.range_succ_eq_map, List.foldr_cons, List.foldr_map]
    have hfold : (List.range k).foldr (fun x acc => acc * 256 + (mem (a + (x + 1))).w % 256) 0 =
        (List.range k).foldr (fun i acc => acc * 256 + (mem (a + 1 + i)).w % 256) 0 := by
      congr 1; funext x acc; rw [Nat.add_assoc, Nat.add_comm x 1]
    have hih : (List.range k).foldr (fun x acc => acc * 256 + (mem (a + (x + 1))).w % 256) 0 < 256 ^ k := by
      rw [hfold]; exact foldr_ld_lt_pow mem (a + 1) k
    have hmod : (mem (a + 0)).w % 256 < 256 := Nat.mod_lt _ (by decide)
    have hpow : 256 ^ (k + 1) = 256 ^ k * 256 := Nat.pow_succ _ _
    have hstep : (List.range k).foldr (fun x acc => acc * 256 + (mem (a + x.succ)).w % 256) 0 =
        (List.range k).foldr (fun x acc => acc * 256 + (mem (a + (x + 1))).w % 256) 0 := rfl
    rw [hpow, hstep]; omega

theorem range_succ_foldr_add_mod (f : Nat → Nat) (q i : Nat) :
    (q + (List.range (i + 1)).foldr (fun j acc => f j + acc) 0) % X86Env.W =
      ((q + (List.range i).foldr (fun j acc => f j + acc) 0) % X86Env.W + f i) % X86Env.W := by
  have hsum : ∀ (l : List Nat) (z : Nat),
      l.foldr (fun j acc => f j + acc) z = l.foldr (fun j acc => f j + acc) 0 + z := by
    intro l z
    induction l with
    | nil => simp
    | cons x xs ih => simp only [List.foldr_cons, ih]; omega
  rw [List.range_succ, List.foldr_append, List.foldr_cons, List.foldr_nil, Nat.add_zero, hsum]
  rw [show q + ((List.range i).foldr (fun j acc => f j + acc) 0 + f i) =
      (q + (List.range i).foldr (fun j acc => f j + acc) 0) + f i by omega]
  rw [Nat.add_mod, ← Nat.mod_mod _ X86Env.W, ← Nat.add_mod]
  simp only [Nat.add_mod, Nat.mod_mod]

theorem range_succ_all (p : Nat → Bool) (i : Nat) :
    (List.range (i + 1)).all p = ((List.range i).all p && p i) := by
  rw [List.range_succ, List.all_append]; simp

theorem mod_add_left (a b n : Nat) : (a % n + b) % n = (a + b) % n := by
  rw [Nat.add_mod, Nat.mod_mod, ← Nat.add_mod]

def memWSize : MemW → Nat
  | .w8 => 1
  | .w16 => 2
  | .w32 => 4
  | .w64 => 8

def ldIdxLen : MemW → Nat
  | .w8 | .w16 => 5
  | .w32 | .w64 => 4

def stIdxLen : MemW → Nat
  | .w16 => 5
  | .w8 | .w32 | .w64 => 4

theorem lift_primPostOutcome {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    {s₀ s₁ : X86Vm.St} {p : Sporos.Prim}
    (hrch : Reaches vmImage s₀ s₁) (hk : s₁.k = s₀.k)
    (h : PrimPostOutcome pr ls f rest s₁ p) :
    PrimPostOutcome pr ls f rest s₀ p := by
  dsimp only [PrimPostOutcome] at h ⊢
  cases hlex : SporosProvLen.lexecPrim p (ls.ds.take p.eff.1) ls with
  | ok res ls' => rw [hlex] at h; exact lift_next_outcome hrch hk h
  | halt c => rw [hlex] at h; exact lift_halt_outcome hrch hk h
  | fail e => rw [hlex] at h; exact lift_fail_outcome hrch hk h

def memWBytes : MemW → Nat → List Nat
  | .w8, v => [v % 256]
  | .w16, v => bytes2 v
  | .w32, v => bytes4 v
  | .w64, v => bytes8 v

theorem memWBytes_len (mw : MemW) (v : Nat) : (memWBytes mw v).length = memWSize mw := by
  cases mw <;> simp [memWBytes, memWSize, bytes2_length, bytes4_length, bytes8_length]

theorem memWBytes_lt (mw : MemW) (v : Nat) : ∀ b ∈ memWBytes mw v, b < 256 := by
  cases mw
  · intro b hb; simp only [memWBytes, List.mem_singleton] at hb; subst hb; exact Nat.mod_lt _ (by decide)
  · exact bytes2_lt
  · exact bytes4_lt
  · exact bytes8_lt

theorem memWBytes_get (mw : MemW) (v j : Nat) (hj : j < memWSize mw) :
    (memWBytes mw v)[j]'(by rw [memWBytes_len]; exact hj) % 256 = v / 256 ^ j % 256 := by
  cases mw
  · have : j = 0 := by simp [memWSize] at hj; omega
    subst this; simp [memWBytes]
  · simp [memWSize] at hj
    have hcases : j = 0 ∨ j = 1 := by omega
    rcases hcases with rfl | rfl <;> simp [memWBytes, bytes2]
  · simp [memWSize] at hj
    have hcases : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3 := by omega
    rcases hcases with rfl | rfl | rfl | rfl <;> simp [memWBytes, bytes4]
  · simp [memWSize] at hj
    have hcases : j = 0 ∨ j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 ∨ j = 5 ∨ j = 6 ∨ j = 7 := by omega
    rcases hcases with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> simp [memWBytes, bytes8]

theorem memWBytes_wrMem (mw : MemW) {m : Mem} {r : Region} {a v : Nat}
    (hw : m.wf) (hm : r ∈ m) (hrw : r.rw = true) (hlo : r.base ≤ a)
    (hhi : a + memWSize mw ≤ r.hi) :
    ∃ (i : Nat) (m' : Mem),
      wrMem mw m a v = some m' ∧ m[i]? = some r ∧
      m' = m.set i { r with data := writeBytes r.data (a - r.base) (memWBytes mw v) } ∧
      m'.wf ∧ { r with data := writeBytes r.data (a - r.base) (memWBytes mw v) } ∈ m' ∧
      (∀ s ∈ m, s.base ≠ r.base → s ∈ m') := by
  cases mw
  · exact wrMem_w8_region hw hm hrw hlo hhi
  · exact wrMem_w16_region hw hm hrw hlo hhi
  · exact wrMem_w32_region hw hm hrw hlo hhi
  · exact wrMem_w64_region hw hm hrw hlo hhi

theorem memRel_wr_memw_bytes (mw : MemW) {mwData : List Nat} {msize a v : Nat}
    (hlo : a + memWSize mw ≤ msize) (hlen : msize ≤ mwData.length) :
    let mwData' := writeBytes mwData a (memWBytes mw v)
    (∀ x, x < msize →
      mwData'[x]? =
        some (if a ≤ x ∧ x < a + memWSize mw then v / 256 ^ (x - a) % 256
              else mwData[x]?.getD 0)) := by
  intro mwData' x hx
  by_cases hxa : x < a
  · rw [writeBytes_getElem?_lt _ _ _ _ hxa]
    have hnot : ¬(a ≤ x ∧ x < a + memWSize mw) := by omega
    rw [if_neg hnot]
    have hlt : x < mwData.length := by omega
    rw [List.getElem?_eq_getElem hlt]; rfl
  · by_cases hxhi : a + memWSize mw ≤ x
    · have hj : a + (memWBytes mw v).length ≤ x := by rw [memWBytes_len]; exact hxhi
      rw [writeBytes_getElem?_ge _ _ _ _ hj]
      have hnot : ¬(a ≤ x ∧ x < a + memWSize mw) := by omega
      rw [if_neg hnot]
      have hlt : x < mwData.length := by omega
      rw [List.getElem?_eq_getElem hlt]; rfl
    · have hcond : a ≤ x ∧ x < a + memWSize mw := by omega
      rw [if_pos hcond]
      have hk : x - a < (memWBytes mw v).length := by rw [memWBytes_len]; omega
      have hd : a + (memWBytes mw v).length ≤ mwData.length := by rw [memWBytes_len]; omega
      have hins := writeBytes_getElem?_inside mwData a (x - a) (memWBytes mw v) hk hd
      rw [show a + (x - a) = x by omega] at hins
      rw [hins, memWBytes_get mw v (x - a) (by omega)]

theorem writeBytes_qword_memp {mpData : List Nat} {msize idx q : Nat}
    (hidx : idx < msize) (hlen : msize * 8 ≤ mpData.length) :
    let mpData' := writeBytes mpData (idx * 8) (bytes8 q)
    (leVal ((mpData'.drop (idx * 8)).take 8) = q % X86Env.W) ∧
    (∀ x, x < msize → x ≠ idx →
      leVal ((mpData'.drop (x * 8)).take 8) = leVal ((mpData.drop (x * 8)).take 8)) := by
  refine ⟨leVal_writeBytes_same mpData (idx * 8) q (by omega), ?_⟩
  intro x _hx hne
  have hdisj : x * 8 + 8 ≤ idx * 8 ∨ idx * 8 + 8 ≤ x * 8 := by omega
  rw [writeBytes_drop_take_disjoint mpData (idx * 8) (x * 8) 8 (bytes8 q)
    (by rw [bytes8_length]; exact hdisj) (by omega)]

/-! ### `VmCtx` memory extractors & updaters -/

theorem wf_mem_info {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) :
    (vc.memwReg.rw = true ∧ ls.msize < 2 ^ 44 ∧
      vc.memwReg.data.length = X86Env.pages (max ls.msize 1) ∧ vc.memwReg.hi ≤ 2 ^ 47 ∧
      vc.memwReg.hi < X86Env.W) ∧
    (vc.mempReg.rw = true ∧
      vc.mempReg.data.length = X86Env.pages (8 * max ls.msize 1) ∧ vc.mempReg.hi ≤ 2 ^ 47 ∧
      vc.mempReg.hi < X86Env.W) ∧
    MemRel vc.memwReg.data vc.mempReg.data ls.msize ls.mem ∧
    ls.hp < X86Env.W := by
  have hbl := h.2.1
  have hmctx := vc.mem_ctx h
  obtain ⟨_, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, hsl15,
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
    _, _, _, _, _, _, _, hmemrel, _, _, _, _, _, _, _, _⟩ := h
  have h47 : (2 ^ 47 : Nat) < X86Env.W := by decide
  have hhpW : ls.hp < X86Env.W := by
    rw [← hsl15]
    exact leVal_take8_lt_W (fun b hb =>
      (hbl vc.ctxReg hmctx) b (List.mem_of_mem_drop hb))
  exact ⟨⟨hmw1, hmw2, hmw3, hmw4, by omega⟩, ⟨hmp1, hmp2, hmp3, by omega⟩, hmemrel, hhpW⟩

theorem rd_ctx_memwBase {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : rd64 s.mem (ea ctxBase 96) = some vc.memwReg.base := by
  have hs96 : leVal ((vc.ctxReg.data.drop 96).take 8) = vc.memwReg.base := by
    obtain ⟨_, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _, _, hsl12, _⟩ := h
    exact hsl12
  exact vc.rd64_ctx_of_eq h (by decide) hs96

theorem rd_ctx_mempBase {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) : rd64 s.mem (ea ctxBase 104) = some vc.mempReg.base := by
  have hs104 : leVal ((vc.ctxReg.data.drop 104).take 8) = vc.mempReg.base := by
    obtain ⟨_, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
      _, _, _, _, _, _, _, _, _, _, _, _, _, hsl13, _⟩ := h
    exact hsl13
  exact vc.rd64_ctx_of_eq h (by decide) hs104

theorem wf_write_mem {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s s' : X86Vm.St} {mw' mp' : Region} {mem' : SporosProvLen.LMem}
    (h : vc.wf pr ls s)
    (hcm' : CodeMem s'.mem) (hbl' : BytesLt s'.mem) (hpos' : ∀ u ∈ s'.mem, 0 < u.data.length)
    (hhr' : RegsOk s') (hsp' : s'.reg 4 = some vc.sp) (h15' : s'.reg 15 = some ctxBase)
    (hmw_base : mw'.base = vc.memwReg.base) (hmw_rw : mw'.rw = true)
    (hmw_len : mw'.data.length = vc.memwReg.data.length)
    (hmp_base : mp'.base = vc.mempReg.base) (hmp_rw : mp'.rw = true)
    (hmp_len : mp'.data.length = vc.mempReg.data.length)
    (hmw_mem : mw' ∈ s'.mem) (hmp_mem : mp' ∈ s'.mem)
    (hoth : ∀ u ∈ s.mem, u.base ≠ vc.memwReg.base → u.base ≠ vc.mempReg.base → u ∈ s'.mem)
    (hmemrel' : MemRel mw'.data mp'.data ls.msize mem')
    (hwlt_mem' : ∀ a, (mem' a).w < Sporos.M ∧ (mem' a).plen < Sporos.M) :
    { vc with memwReg := mw', mempReg := mp' }.wf pr { ls with mem := mem' } s' := by
  obtain ⟨_, _, _, _, _, _, hnodup, hmem,
    hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3, hctx4, hf1, hf2, hf3,
    hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
    hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
    _hmw1, hmw2, hmw3, hmw4, _hmp1, hmp2, hmp3,
    htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
    hinp1, hout1, hout2, hout3, hout4, hout5,
    hsl0, hsl1, hsl2, hsl3, hsl4, hsl5, hsl6, hsl7,
    hsl8, hsl9, hsl10, hsl11, hsl12, hsl13, hsl14, hsl15,
    hsl16, hsl17, hsl18, hsl19, hsl20, hsl21, hsl22, hsl23,
    hsl24, hsl25, hsl26, hsl27, hsl28, hsl29, hsl30, hsl31,
    hpds, hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle,
    _hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle, hwlt, hfit⟩ := h
  have hnodup0 : (List.map Region.base vc.regions).Nodup := hnodup
  simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnodup0
  have hnodup' : (List.map Region.base ({ vc with memwReg := mw', mempReg := mp' }).regions).Nodup := by
    simpa only [VmCtx.regions, List.map_cons, List.map_nil, hmw_base, hmp_base] using hnodup
  have hmem' : ∀ r ∈ ({ vc with memwReg := mw', mempReg := mp' }).regions, r ∈ s'.mem := by
    intro r hr_mem
    simp only [VmCtx.regions, List.mem_cons, List.not_mem_nil, or_false] at hr_mem
    rcases hr_mem with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals first
      | exact hmw_mem
      | exact hmp_mem
      | exact hoth _ (hmem _ (by simp [VmCtx.regions]))
          (by intro heq; rw [heq] at hnodup0; simp [List.nodup_cons] at hnodup0)
          (by intro heq; rw [heq] at hnodup0; simp [List.nodup_cons] at hnodup0)
  have hmw4' : mw'.hi ≤ 2 ^ 47 := by
    simpa only [Region.hi, hmw_base, hmw_len] using hmw4
  have hmp3' : mp'.hi ≤ 2 ^ 47 := by
    simpa only [Region.hi, hmp_base, hmp_len] using hmp3
  have hwlt' : ({ ls with mem := mem' } : SporosProvLen.LSt).WordsLt :=
    ⟨hwlt.1, hwlt.2.1, hwlt.2.2.1, hwlt_mem', hwlt.2.2.2.2.1, hwlt.2.2.2.2.2⟩
  refine ⟨hcm', hbl', hpos', hhr', hsp', h15', hnodup', hmem',
    hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3, hctx4, hf1, hf2, hf3,
    hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
    hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
    hmw_rw, hmw2, by rw [hmw_len]; exact hmw3, hmw4',
    hmp_rw, by rw [hmp_len]; exact hmp2, hmp3',
    htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
    hinp1, hout1, hout2, hout3, hout4, hout5,
    hsl0, hsl1, hsl2, hsl3, hsl4, hsl5, hsl6, hsl7,
    hsl8, hsl9, hsl10, hsl11,
    by rw [hmw_base]; exact hsl12, by rw [hmp_base]; exact hsl13, hsl14, hsl15,
    hsl16, hsl17, hsl18, hsl19, hsl20, hsl21, hsl22, hsl23,
    hsl24, hsl25, hsl26, hsl27, hsl28, hsl29, hsl30, hsl31,
    hpds, hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle,
    hmemrel', htabsrel, htabsle, hinprel, houtrel, houtle, hwlt', hfit⟩

theorem step_wr_ctx_hp {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {len r newHp : Nat}
    (hstep : X86Vm.step vmImage s = X86Vm.exec (.ofInt (.st 15 120 r)) len s)
    (h : vc.wf pr ls s) (hr : s.reg r = some newHp) (hW : newHp < X86Env.W) :
    let ctx' := { vc.ctxReg with data := writeBytes vc.ctxReg.data 120 (bytes8 newHp) }
    let vc' := { vc with ctxReg := ctx' }
    ∃ s',
      X86Vm.step vmImage s = .ok s' ∧
      s'.rip = s.rip + len ∧ s'.k = s.k ∧ s'.regs = s.regs ∧ s'.fl = s.fl ∧
      vc'.wf pr { ls with hp := newHp } s' ∧
      (∀ m0 u, (m0 + 8 ≤ 120 ∨ 128 ≤ m0) → m0 + 8 ≤ ctxSize →
        rd64 s.mem (ea ctxBase m0) = some u → rd64 s'.mem (ea ctxBase m0) = some u) := by
  have hb := vc.ctx_bounds h
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have hea : ea ctxBase 120 = ctxBase + 120 := ea_add_eq (by have : ctxSize = 4096 := rfl; omega)
  obtain ⟨idx, m', hwr, hidx, hm', _, hmem', hoth⟩ :=
    wrMem_w64_region h.1.2 (vc.mem_ctx h) hb.2.1 (a := ctxBase + 120)
      (by rw [hb.1]; omega) (by simp only [Region.hi, hb.1]; have : ctxSize = 4096 := rfl; omega)
      (v := newHp)
  have hsub : ctxBase + 120 - vc.ctxReg.base = 120 := by rw [hb.1]; omega
  rw [hsub] at hm' hmem'
  rw [← hea] at hwr
  have hcm' := h.1.wr64_preserve hwr
  have hbl' := bytesLt_wr64 h.2.1 hwr
  have hpos' := len_pos_wrN h.2.2.1 hwr
  let ctx' := { vc.ctxReg with data := writeBytes vc.ctxReg.data 120 (bytes8 newHp) }
  let vc' := { vc with ctxReg := ctx' }
  let s' : X86Vm.St := { regs := s.regs, rip := s.rip + len, fl := s.fl, mem := m', k := s.k }
  have hs : X86Vm.step vmImage s = .ok s' := by rw [hstep, exec_st h15 hr hwr]
  have hle_ctx : ∀ m0 u, m0 + 8 ≤ ctxSize → rd64 s.mem (ea ctxBase m0) = some u →
      leVal ((vc.ctxReg.data.drop m0).take 8) = u :=
    fun m0 _ hmlim hprev => ctx_leVal_of_rd64 h hmlim hprev
  obtain ⟨_hcm, _hblt, _hpos, hregs, hsp, h15s, hnodup, hmem,
    hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3, hctx4, hf1, hf2, hf3,
    hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
    hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
    hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
    htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
    hinp1, hout1, hout2, hout3, hout4, hout5,
    hsl0, hsl1, hsl2, hsl3, hsl4, hsl5, hsl6, hsl7,
    hsl8, hsl9, hsl10, hsl11, hsl12, hsl13, hsl14, _hsl15,
    hsl16, hsl17, hsl18, hsl19, hsl20, hsl21, hsl22, hsl23,
    hsl24, hsl25, hsl26, hsl27, hsl28, hsl29, hsl30, hsl31,
    hpds, hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle,
    hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle, hwlt, hfit⟩ := h
  have hnodup' : (vc'.regions.map Region.base).Nodup := by
    simpa [vc', ctx', VmCtx.regions] using hnodup
  have hmem_all : ∀ u ∈ vc'.regions, u ∈ s'.mem := by
    intro u hu
    simp only [vc', VmCtx.regions, List.mem_cons, List.not_mem_nil, or_false] at hu
    have hnot : ∀ r0 ∈ s.mem, r0.base ≠ vc.ctxReg.base → r0 ∈ s'.mem := by
      intro r0 hr0 hne
      rw [show s'.mem = m' from rfl]; exact hoth r0 hr0 hne
    rcases hu with rfl | rfl | huc | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals first
      | (rw [huc]; exact hmem')
      | exact hnot _ (hmem _ (by simp [VmCtx.regions]))
          (by intro heq; have hnd := hnodup; simp only [VmCtx.regions, List.map_cons, List.map_nil] at hnd
              rw [heq] at hnd; simp [List.nodup_cons] at hnd)
  have hctx3' : ctxSize ≤ ctx'.data.length := by dsimp [ctx']; rw [writeBytes_length]; exact hctx3
  have hctx4' : ctx'.hi ≤ 2 ^ 47 := by simpa only [Region.hi, ctx', writeBytes_length] using hctx4
  have hslot : ∀ j, (j + 8 ≤ 120 ∨ 128 ≤ j) → j + 8 ≤ ctxSize →
      leVal ((ctx'.data.drop j).take 8) = leVal ((vc.ctxReg.data.drop j).take 8) := by
    intro j hj hlim
    dsimp [ctx']
    rw [writeBytes_drop_take_disjoint vc.ctxReg.data 120 j 8 (bytes8 newHp)
      (by rw [bytes8_length]; exact hj) (by have : ctxSize = 4096 := rfl; omega)]
  have hsl15' : leVal ((ctx'.data.drop 120).take 8) = newHp := by
    dsimp [ctx']
    rw [leVal_writeBytes_same vc.ctxReg.data 120 newHp (by have : ctxSize = 4096 := rfl; omega),
      Nat.mod_eq_of_lt hW]
  have hwf' : vc'.wf pr { ls with hp := newHp } s' :=
    ⟨hcm', hbl', hpos', hregs, hsp, h15s, hnodup', hmem_all,
      hstk1, hstk2, hstk3, hstk4, hctx1, hctx2, hctx3', hctx4', hf1, hf2, hf3,
      hds1, hds2, hds3, hds4, hds5, haux1, haux2, haux3, haux4, haux5,
      hrs1, hrs2, hrs3, hrs4, hrs5, hloc1, hloc2, hloc3, hloc4, hloc5,
      hmw1, hmw2, hmw3, hmw4, hmp1, hmp2, hmp3,
      htb1, htb2, htb3, htb4, htb5, hkv1, hkv2, hkv3, hkv4, hkv5, hkv6,
      hinp1, hout1, hout2, hout3, hout4, hout5,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl0,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl1,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl2,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl3,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl4,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl5,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl6,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl7,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl8,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl9,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl10,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl11,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl12,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl13,
      by rw [hslot _ (Or.inl (by decide)) (by decide)]; exact hsl14,
      hsl15',
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl16,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl17,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl18,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl19,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl20,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl21,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl22,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl23,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl24,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl25,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl26,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl27,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl28,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl29,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl30,
      by rw [hslot _ (Or.inr (by decide)) (by decide)]; exact hsl31,
      hpds, hdsle, hpaux, hauxle, hrsrel, hrsle, hlocle,
      hmemrel, htabsrel, htabsle, hinprel, houtrel, houtle,
      ⟨hwlt.1, hwlt.2.1, hwlt.2.2.1, hwlt.2.2.2.1, hwlt.2.2.2.2.1, hwlt.2.2.2.2.2⟩, hfit⟩
  refine ⟨s', hs, rfl, rfl, rfl, rfl, hwf', ?_⟩
  intro m0 u hm0 hmlim hprev
  exact vc'.rd64_ctx_of_eq hwf' hmlim
    (by rw [hslot m0 hm0 hmlim, hle_ctx m0 u hmlim hprev])

/-! ### Shared 6-instruction range check -/

theorem range_check_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {ra rlen a len rip0 : Nat}
    {c0 c1 c2 c3 c4 c5 : Nat}
    (hm0 : (rip0, (.ofInt (.cmp ra 10)), 3) ∈ vmTableChunk c0)
    (hm1 : (rip0 + 3, (.ofInt (.jcc 7 SporosVmBin.L_fail_bounds)), 6) ∈ vmTableChunk c1)
    (hm2 : (rip0 + 9, (.ofInt (.mov 11 10)), 3) ∈ vmTableChunk c2)
    (hm3 : (rip0 + 12, (.ofInt (.alu .sub 11 ra)), 3) ∈ vmTableChunk c3)
    (hm4 : (rip0 + 15, (.ofInt (.cmp rlen 11)), 3) ∈ vmTableChunk c4)
    (hm5 : (rip0 + 18, (.ofInt (.jcc 7 SporosVmBin.L_fail_bounds)), 6) ∈ vmTableChunk c5)
    (hrip : s.rip = rip0) (h : vc.wf pr ls s)
    (hra : s.reg ra = some a) (hrlen : s.reg rlen = some len) (hr10 : s.reg 10 = some ls.msize)
    (haW : a < X86Env.W) (hlenW : len < X86Env.W) (hne11 : ra ≠ 11) (hne11' : rlen ≠ 11) :
    if a + len ≤ ls.msize then
      ∃ s',
        Reaches vmImage s s' ∧ s'.rip = rip0 + 24 ∧ s'.mem = s.mem ∧ s'.k = s.k ∧
        RegsOk s' ∧ s'.reg 4 = some vc.sp ∧ s'.reg 15 = some ctxBase ∧
        s'.reg 11 = some (ls.msize - a) ∧
        (∀ r, r ≠ 11 → s'.reg r = s.reg r)
    else
      ∃ s_exit, Reaches vmImage s s_exit ∧ X86Vm.step vmImage s_exit = .exit 1 ∧ s_exit.k = s.k := by
  have hmsize_lt : ls.msize < 2 ^ 44 := (wf_mem_info h).1.2.1
  have h44 : (2 ^ 44 : Nat) < X86Env.W := by decide
  have hmsizeW : ls.msize < X86Env.W := Nat.lt_trans hmsize_lt h44
  have hreg := h.2.2.2.1
  have h4 := h.2.2.2.2.1
  have h15 := h.2.2.2.2.2.1
  have h0 : X86Vm.step vmImage s =
      .ok { s with fl := (X86Int.aluF .sub a ls.msize).2, rip := rip0 + 3 } := by
    rw [step_at_vc c0 hrip hm0, exec_cmp hra hr10, hrip]
  let s1 := { s with fl := (X86Int.aluF .sub a ls.msize).2, rip := rip0 + 3 }
  by_cases hgt : ls.msize < a
  · have hno : ¬(a + len ≤ ls.msize) := by omega
    rw [if_neg hno]
    have h1 : X86Vm.step vmImage s1 = .ok { s1 with rip := SporosVmBin.L_fail_bounds } := by
      rw [step_at_vc c1 rfl hm1, exec_jcc (cond_aluF_sub_a a ls.msize haW hmsizeW)]
      dsimp only [s1]; simp [hgt]
    let s2 := { s1 with rip := SporosVmBin.L_fail_bounds }
    have hwf2 : vc.wf pr ls s2 := vc.wf_step_misc (s' := s2) h rfl hreg h4 h15
    exact lift_fail_outcome (Reaches.step_trans h0 (Reaches.step h1)) rfl
      (L_fail_from_vc_spec (e := .bounds) rfl hwf2)
  · have hle_a : a ≤ ls.msize := by omega
    have h1 : X86Vm.step vmImage s1 = .ok { s1 with rip := rip0 + 9 } := by
      rw [step_at_vc c1 rfl hm1, exec_jcc (cond_aluF_sub_a a ls.msize haW hmsizeW)]
      dsimp only [s1]; simp [hgt]
    let s2 := { s1 with rip := rip0 + 9 }
    have hs2_10 : s2.reg 10 = some ls.msize := hr10
    have h2 : X86Vm.step vmImage s2 = .ok { s2.set 11 (some ls.msize) with rip := rip0 + 12 } := by
      rw [step_at_vc c2 rfl hm2, exec_mov hs2_10]
    let s3 := { s2.set 11 (some ls.msize) with rip := rip0 + 12 }
    have hr3 : RegsOk s3 := regsOk_set hreg (by decide) ls.msize
    have hs3_11 : s3.reg 11 = some ls.msize := reg_set_self_ok' hreg (by decide) _
    have hs3_ra : s3.reg ra = some a := by
      change (s2.set 11 _).reg ra = some a; rw [reg_set_ne (Ne.symm hne11)]; exact hra
    have hs3_rlen : s3.reg rlen = some len := by
      change (s2.set 11 _).reg rlen = some len; rw [reg_set_ne (Ne.symm hne11')]; exact hrlen
    have hsub_eq : (ls.msize + X86Env.W - a % X86Env.W) % X86Env.W = ls.msize - a := by
      rw [Nat.mod_eq_of_lt haW, show ls.msize + X86Env.W - a = (ls.msize - a) + X86Env.W by omega,
        Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    have h3 : X86Vm.step vmImage s3 =
        .ok { s3.set 11 (some (ls.msize - a)) with fl := (X86Int.aluF .sub ls.msize a).2, rip := rip0 + 15 } := by
      rw [step_at_vc c3 rfl hm3, exec_alu hs3_11 hs3_ra]
      have h1eq : (X86Int.aluF .sub ls.msize a).1 = ls.msize - a := by
        rw [aluF_sub_val]; simpa [Nat.mod_eq_of_lt haW] using hsub_eq
      rw [← h1eq]
    let s4 := { s3.set 11 (some (ls.msize - a)) with fl := (X86Int.aluF .sub ls.msize a).2, rip := rip0 + 15 }
    have hr4 : RegsOk s4 := regsOk_set hr3 (by decide) (ls.msize - a)
    have hs4_11 : s4.reg 11 = some (ls.msize - a) := reg_set_self_ok' hr3 (by decide) _
    have hs4_rlen : s4.reg rlen = some len := by
      change (s3.set 11 _).reg rlen = some len; rw [reg_set_ne (Ne.symm hne11')]; exact hs3_rlen
    have hs4_4 : s4.reg 4 = some vc.sp := by
      change (s3.set 11 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]
      change (s2.set 11 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
    have hs4_15 : s4.reg 15 = some ctxBase := by
      change (s3.set 11 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]
      change (s2.set 11 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
    have h4s : X86Vm.step vmImage s4 =
        .ok { s4 with fl := (X86Int.aluF .sub len (ls.msize - a)).2, rip := rip0 + 18 } := by
      rw [step_at_vc c4 rfl hm4, exec_cmp hs4_rlen hs4_11]
    let s5 := { s4 with fl := (X86Int.aluF .sub len (ls.msize - a)).2, rip := rip0 + 18 }
    have hrch5 : Reaches vmImage s s5 :=
      Reaches.step_trans h0 (Reaches.step_trans h1 (Reaches.step_trans h2 (Reaches.step_trans h3 (Reaches.step h4s))))
    by_cases hgt2 : ls.msize - a < len
    · have hno : ¬(a + len ≤ ls.msize) := by omega
      rw [if_neg hno]
      have h5 : X86Vm.step vmImage s5 = .ok { s5 with rip := SporosVmBin.L_fail_bounds } := by
        rw [step_at_vc c5 rfl hm5, exec_jcc (cond_aluF_sub_a len (ls.msize - a) hlenW (by omega))]
        dsimp only [s5]; simp [hgt2]
      let s6 := { s5 with rip := SporosVmBin.L_fail_bounds }
      have hwf6 : vc.wf pr ls s6 := vc.wf_step_misc (s' := s6) h rfl hr4 hs4_4 hs4_15
      exact lift_fail_outcome (Reaches.trans hrch5 (Reaches.step h5)) rfl
        (L_fail_from_vc_spec (e := .bounds) rfl hwf6)
    · have hyes : a + len ≤ ls.msize := by omega
      rw [if_pos hyes]
      have h5 : X86Vm.step vmImage s5 = .ok { s5 with rip := rip0 + 24 } := by
        rw [step_at_vc c5 rfl hm5, exec_jcc (cond_aluF_sub_a len (ls.msize - a) hlenW (by omega))]
        dsimp only [s5]; simp [hgt2]
      refine ⟨{ s5 with rip := rip0 + 24 }, Reaches.trans hrch5 (Reaches.step h5), rfl, rfl, rfl, hr4, hs4_4, hs4_15, hs4_11, ?_⟩
      intro r hrne
      change (s3.set 11 _).reg r = s.reg r
      rw [reg_set_ne (Ne.symm hrne)]
      change (s2.set 11 _).reg r = s.reg r
      rw [reg_set_ne (Ne.symm hrne)]
      rfl

/-! ### Shared `P_ld{8,16,32,64}` helpers and specs -/

theorem ld_memw_val {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    (h : vc.wf pr ls s) (mw : MemW) {a : Nat} (hle : a + memWSize mw ≤ ls.msize) :
    rdMem mw s.mem (eaIdx vc.memwReg.base a 1 0) =
      some (leVal ((vc.memwReg.data.drop a).take (memWSize mw))) := by
  have hmi := wf_mem_info h
  have hrel := hmi.2.2.1
  have hmreg := vc.mem_memw h
  have hlo : vc.memwReg.base ≤ vc.memwReg.base + a := Nat.le_add_right _ _
  have hea : eaIdx vc.memwReg.base a 1 0 = vc.memwReg.base + a := by
    have hltW : vc.memwReg.base + 1 * a + 0 < X86Env.W := by
      have hle1 : a + 1 ≤ ls.msize := by cases mw <;> simp [memWSize] at hle ⊢ <;> omega
      have := hmi.1.2.2.2.2
      have hmw_len := hrel.1
      simp only [Region.hi] at this
      omega
    rw [eaIdx_add_eq hltW, Nat.one_mul, Nat.add_zero]
  rw [hea]
  have hhi : vc.memwReg.base + a + memWSize mw ≤ vc.memwReg.hi := by
    have := hrel.1; simp only [Region.hi]; omega
  cases mw
  · have hlt_d : a < vc.memwReg.data.length := by
      have := hrel.1; simp [memWSize] at hle; omega
    have hrd := mem_rd_of_mem h.1.2 hmreg hlo hhi
    simp only [rdMem, Nat.add_sub_cancel_left] at hrd ⊢
    rw [hrd]
    have hdrop : (vc.memwReg.data.drop a).take 1 = [vc.memwReg.data[a]'hlt_d] := by
      have h0 := (list_take_drop_succ vc.memwReg.data a 0 hlt_d).symm
      simpa using h0
    have hb_lt : vc.memwReg.data[a]'hlt_d < 256 :=
      h.2.1 vc.memwReg hmreg _ (List.getElem_mem _)
    simp only [memWSize, hdrop, leVal, Nat.add_zero, Nat.mul_zero]
  · exact rdMem_w16_at h.1.2 hmreg hlo hhi (by simp [memWSize, Nat.add_sub_cancel_left])
  · exact rdMem_w32_at h.1.2 hmreg hlo hhi (by simp [memWSize, Nat.add_sub_cancel_left])
  · exact rdMem_w64_at h.1.2 hmreg hlo hhi (by simp [memWSize, Nat.add_sub_cancel_left])

theorem ld_memp_sum_step {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt} {s : X86Vm.St}
    {a i acc rip0 chk : Nat}
    (hrip : s.rip = rip0)
    (hm_ld : (rip0, (.ldIdx .w64 11 9 0 8 (8 * i)), if i = 0 then 4 else 5) ∈ vmTableChunk chk)
    (hm_add : (rip0 + (if i = 0 then 4 else 5), (.ofInt (.alu .add 10 11)), 3) ∈ vmTableChunk chk)
    (h : vc.wf pr ls s) (hlt : a + i < ls.msize) (hreg : RegsOk s)
    (hr9 : s.reg 9 = some vc.mempReg.base) (hr0 : s.reg 0 = some a) (hr10 : s.reg 10 = some acc)
    (hr4 : s.reg 4 = some vc.sp) (hr15 : s.reg 15 = some ctxBase) :
    ∃ s',
      Reaches vmImage s s' ∧ s'.rip = rip0 + (if i = 0 then 7 else 8) ∧ s'.mem = s.mem ∧ s'.k = s.k ∧
      RegsOk s' ∧ s'.reg 4 = some vc.sp ∧ s'.reg 15 = some ctxBase ∧
      s'.reg 0 = some a ∧ s'.reg 9 = some vc.mempReg.base ∧
      s'.reg 10 = some ((acc % X86Env.W + (ls.mem (a + i)).plen) % X86Env.W) := by
  have hmi := wf_mem_info h
  have hrel := hmi.2.2.1
  have hmp_mem := vc.mem_memp h
  have hwlt := (vc.code_flat h).2.2.2.2
  have hp_lt : (ls.mem (a + i)).plen < X86Env.W := (hwlt.2.2.2.1 (a + i)).2
  have hmp_len : ls.msize * 8 ≤ vc.mempReg.data.length := hrel.2.1
  have hhi_ea : vc.mempReg.base + (a + i) * 8 + 8 ≤ vc.mempReg.hi := by
    simp only [Region.hi]; omega
  have hea : eaIdx vc.mempReg.base a 8 (8 * i) = vc.mempReg.base + (a + i) * 8 := by
    have hltW : vc.mempReg.base + 8 * a + 8 * i < X86Env.W := by
      have := hmi.2.1.2.2.2; simp only [Region.hi] at this; omega
    rw [eaIdx_add_eq hltW]; omega
  have hrd : rdMem .w64 s.mem (eaIdx vc.mempReg.base a 8 (8 * i)) = some (ls.mem (a + i)).plen := by
    rw [hea]
    refine rdMem_w64_at h.1.2 hmp_mem (Nat.le_add_right _ _) hhi_ea ?_
    rw [Nat.add_sub_cancel_left, (hrel.2.2 (a + i) hlt).2, Nat.mod_eq_of_lt hp_lt]
  let l1 := if i = 0 then 4 else 5
  have h0 : X86Vm.step vmImage s =
      .ok { s.set 11 (some (ls.mem (a + i)).plen) with rip := rip0 + l1 } := by
    rw [step_at_vc chk hrip hm_ld, exec_ldIdx hr9 hr0 hrd, hrip]
  let s1 := { s.set 11 (some (ls.mem (a + i)).plen) with rip := rip0 + l1 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) (ls.mem (a + i)).plen
  have hs1_11 : s1.reg 11 = some (ls.mem (a + i)).plen := reg_set_self_ok' hreg (by decide) _
  have hs1_10 : s1.reg 10 = some acc := by
    change (s.set 11 _).reg 10 = some acc; rw [reg_set_ne (by decide)]; exact hr10
  have hs1_9 : s1.reg 9 = some vc.mempReg.base := by
    change (s.set 11 _).reg 9 = some vc.mempReg.base; rw [reg_set_ne (by decide)]; exact hr9
  have hs1_0 : s1.reg 0 = some a := by
    change (s.set 11 _).reg 0 = some a; rw [reg_set_ne (by decide)]; exact hr0
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 11 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hr4
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 11 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hr15
  let res := (acc % X86Env.W + (ls.mem (a + i)).plen) % X86Env.W
  have h1 : X86Vm.step vmImage s1 =
      .ok { s1.set 10 (some res) with fl := (X86Int.aluF .add acc (ls.mem (a + i)).plen).2, rip := rip0 + (if i = 0 then 7 else 8) } := by
    rw [step_at_vc chk rfl hm_add, exec_alu hs1_10 hs1_11]
    have hrip_arith : rip0 + l1 + 3 = rip0 + (if i = 0 then 7 else 8) := by cases i <;> simp [l1]
    have hres_eq : (X86Int.aluF .add acc (ls.mem (a + i)).plen).1 = res := by
      rw [aluF_add_val]; dsimp only [res]; rw [Nat.add_mod, Nat.mod_eq_of_lt hp_lt]
    rw [← hres_eq]; change _ = Out.ok { s1.set 10 (some _) with fl := _, rip := _ }
    rw [← hrip_arith]
  let s2 := { s1.set 10 (some res) with fl := (X86Int.aluF .add acc (ls.mem (a + i)).plen).2, rip := rip0 + (if i = 0 then 7 else 8) }
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) res
  refine ⟨s2, Reaches.step_trans h0 (Reaches.step h1), rfl, rfl, rfl, hr2,
    by change (s1.set 10 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs1_4,
    by change (s1.set 10 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15,
    by change (s1.set 10 _).reg 0 = some a; rw [reg_set_ne (by decide)]; exact hs1_0,
    by change (s1.set 10 _).reg 9 = some vc.mempReg.base; rw [reg_set_ne (by decide)]; exact hs1_9,
    reg_set_self_ok' hr1 (by decide) res⟩

theorem ld_prologue_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (mw : MemW) (p : Sporos.Prim) (rip0 : Nat)
    {c_ld0 c_mov1 c_ld10 c0 c1 c2 c3 c4 c5 c_ld8 c_ldidx c_st360 c_ld9 c_ldq : Nat}
    (hp_eff : p.eff.1 = 1 ∧ p.eff.2 = 1 ∧
      (∀ (_v : Nat) (_pm : Nat),
        SporosProvLen.lexecPrim p [ls.ds[0]!] ls =
          if ls.ds[0]!.w + memWSize mw ≤ ls.msize then
            .ok [{ w := (ls.ld ls.ds[0]!.w (memWSize mw)).get!.1,
                   plen := (SporosProvLen.lq [ls.ds[0]!] ls.pc + (ls.ld ls.ds[0]!.w (memWSize mw)).get!.2) % Sporos.M }] ls
          else .fail .bounds))
    (hm_ld0 : (rip0, (.ofInt (.ld 0 15 296)), 7) ∈ vmTableChunk c_ld0)
    (hm_mov1 : (rip0 + 7, (.ofInt (.movI 1 (memWSize mw))), 5) ∈ vmTableChunk c_mov1)
    (hm_ld10 : (rip0 + 12, (.ofInt (.ld 10 15 112)), 4) ∈ vmTableChunk c_ld10)
    (hm_rc0 : (rip0 + 16, (.ofInt (.cmp 0 10)), 3) ∈ vmTableChunk c0)
    (hm_rc1 : (rip0 + 19, (.ofInt (.jcc 7 SporosVmBin.L_fail_bounds)), 6) ∈ vmTableChunk c1)
    (hm_rc2 : (rip0 + 25, (.ofInt (.mov 11 10)), 3) ∈ vmTableChunk c2)
    (hm_rc3 : (rip0 + 28, (.ofInt (.alu .sub 11 0)), 3) ∈ vmTableChunk c3)
    (hm_rc4 : (rip0 + 31, (.ofInt (.cmp 1 11)), 3) ∈ vmTableChunk c4)
    (hm_rc5 : (rip0 + 34, (.ofInt (.jcc 7 SporosVmBin.L_fail_bounds)), 6) ∈ vmTableChunk c5)
    (hm_ld8 : (rip0 + 40, (.ofInt (.ld 8 15 96)), 4) ∈ vmTableChunk c_ld8)
    (hm_ldidx : (rip0 + 44, (.ldIdx mw 2 8 0 1 0), ldIdxLen mw) ∈ vmTableChunk c_ldidx)
    (hm_st360 : (rip0 + 44 + ldIdxLen mw, (.ofInt (.st 15 360 2)), 7) ∈ vmTableChunk c_st360)
    (hm_ld9 : (rip0 + 51 + ldIdxLen mw, (.ofInt (.ld 9 15 104)), 4) ∈ vmTableChunk c_ld9)
    (hm_ldq : (rip0 + 55 + ldIdxLen mw, (.ofInt (.ld 10 15 272)), 7) ∈ vmTableChunk c_ldq)
    (hrip : s.rip = rip0) (hle : 1 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (_hor : OracleAdmits s) (_hrs : ls.rs = f :: rest) (_hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 1) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 1) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (hcont : ∀ (vc' : VmCtx) (s' : X86Vm.St),
      ls.ds[0]!.w + memWSize mw ≤ ls.msize →
      Reaches vmImage s s' → s'.k = s.k →
      s'.rip = rip0 + 62 + ldIdxLen mw →
      vc'.wf pr ls s' → RegsOk s' →
      s'.reg 4 = some vc.sp → s'.reg 15 = some ctxBase →
      s'.reg 0 = some ls.ds[0]!.w → s'.reg 9 = some vc.mempReg.base →
      s'.reg 10 = some (SporosProvLen.lq [ls.ds[0]!] ls.pc) →
      vc'.mempReg = vc.mempReg → vc'.sp = vc.sp →
      rd64 s'.mem (ea ctxBase 280) = some 1 →
      rd64 s'.mem (ea ctxBase 288) = some 1 →
      rd64 s'.mem (ea ctxBase 360) = some (ls.ld ls.ds[0]!.w (memWSize mw)).get!.1 →
      PrimPostOutcome pr ls f rest s' p) :
    PrimPostOutcome pr ls f rest s p := by
  have _ := hp_eff
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have haW : ls.ds[0]!.w < X86Env.W := (wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 0 < ls.ds.length))).1
  have h0 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.ds[0]!.w) with rip := rip0 + 7 } := by
    rw [step_at_vc c_ld0 hrip hm_ld0, exec_ld h15 h296, hrip]
  let s1 := { s.set 0 (some ls.ds[0]!.w) with rip := rip0 + 7 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) _
  have hs1_0 : s1.reg 0 = some ls.ds[0]!.w := reg_set_self_ok' hreg (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hkW : memWSize mw < X86Env.W := by cases mw <;> simp [memWSize] <;> decide
  have h1 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some (memWSize mw % X86Env.W)) with rip := rip0 + 12 } := by
    rw [step_at_vc c_mov1 rfl hm_mov1, exec_movI]
  let s2 := { s1.set 1 (some (memWSize mw)) with rip := rip0 + 12 }
  have h1' : X86Vm.step vmImage s1 = .ok s2 := by simpa [s2, Nat.mod_eq_of_lt hkW] using h1
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) _
  have hs2_1 : s2.reg 1 = some (memWSize mw) := reg_set_self_ok' hr1 (by decide) _
  have hs2_0 : s2.reg 0 = some ls.ds[0]!.w := by
    change (s1.set 1 _).reg 0 = some ls.ds[0]!.w; rw [reg_set_ne (by decide)]; exact hs1_0
  have hs2_15 : s2.reg 15 = some ctxBase := by
    change (s1.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
  have hs2_4 : s2.reg 4 = some vc.sp := by
    change (s1.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs1_4
  have hms2 : rd64 s2.mem (ea ctxBase 112) = some ls.msize := by
    change rd64 s.mem (ea ctxBase 112) = some ls.msize; exact rd_ctx_msize h
  have h2 : X86Vm.step vmImage s2 = .ok { s2.set 10 (some ls.msize) with rip := rip0 + 16 } := by
    rw [step_at_vc c_ld10 rfl hm_ld10, exec_ld hs2_15 hms2]
  let s3 := { s2.set 10 (some ls.msize) with rip := rip0 + 16 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) ls.msize
  have hs3_10 : s3.reg 10 = some ls.msize := reg_set_self_ok' hr2 (by decide) _
  have hs3_0 : s3.reg 0 = some ls.ds[0]!.w := by
    change (s2.set 10 _).reg 0 = some ls.ds[0]!.w; rw [reg_set_ne (by decide)]; exact hs2_0
  have hs3_1 : s3.reg 1 = some (memWSize mw) := by
    change (s2.set 10 _).reg 1 = some (memWSize mw); rw [reg_set_ne (by decide)]; exact hs2_1
  have hs3_4 : s3.reg 4 = some vc.sp := by
    change (s2.set 10 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs2_4
  have hs3_15 : s3.reg 15 = some ctxBase := by
    change (s2.set 10 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs2_15
  have hwf3 : vc.wf pr ls s3 := vc.wf_step_misc (s' := s3) h rfl hr3 hs3_4 hs3_15
  have hrch3 : Reaches vmImage s s3 :=
    Reaches.step_trans h0 (Reaches.step_trans h1' (Reaches.step h2))
  have hrc := range_check_spec (vc := vc) (pr := pr) (ls := ls) (s := s3) (ra := 0) (rlen := 1)
    (a := ls.ds[0]!.w) (len := memWSize mw) (rip0 := rip0 + 16)
    hm_rc0 hm_rc1 hm_rc2 hm_rc3 hm_rc4 hm_rc5 rfl hwf3 hs3_0 hs3_1 hs3_10 haW hkW
    (by decide) (by decide)
  by_cases hbd : ls.ds[0]!.w + memWSize mw ≤ ls.msize
  · rw [if_pos hbd] at hrc
    obtain ⟨s4, hrch4, hrip4, hmem4, hk4, hr4, hs4_4, hs4_15, _, hreg4⟩ := hrc
    have hs4_0 : s4.reg 0 = some ls.ds[0]!.w := by rw [hreg4 0 (by decide)]; exact hs3_0
    have hwf4 : vc.wf pr ls s4 := vc.wf_step_misc (s' := s4) hwf3 hmem4 hr4 hs4_4 hs4_15
    have hrd_mw : rd64 s4.mem (ea ctxBase 96) = some vc.memwReg.base := by
      rw [hmem4]; exact rd_ctx_memwBase hwf3
    have h4s : X86Vm.step vmImage s4 = .ok { s4.set 8 (some vc.memwReg.base) with rip := rip0 + 44 } := by
      rw [step_at_vc c_ld8 hrip4 hm_ld8, exec_ld hs4_15 hrd_mw, hrip4]
    let s5 := { s4.set 8 (some vc.memwReg.base) with rip := rip0 + 44 }
    have hr5 : RegsOk s5 := regsOk_set hr4 (by decide) vc.memwReg.base
    have hs5_8 : s5.reg 8 = some vc.memwReg.base := reg_set_self_ok' hr4 (by decide) _
    have hs5_0 : s5.reg 0 = some ls.ds[0]!.w := by
      change (s4.set 8 _).reg 0 = some ls.ds[0]!.w; rw [reg_set_ne (by decide)]; exact hs4_0
    have hs5_4 : s5.reg 4 = some vc.sp := by
      change (s4.set 8 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs4_4
    have hs5_15 : s5.reg 15 = some ctxBase := by
      change (s4.set 8 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs4_15
    have hwf5 : vc.wf pr ls s5 := vc.wf_step_misc (s' := s5) hwf4 rfl hr5 hs5_4 hs5_15
    have hrd_memw := ld_memw_val hwf5 mw hbd
    have hval_eq : leVal ((vc.memwReg.data.drop ls.ds[0]!.w).take (memWSize mw)) =
        (ls.ld ls.ds[0]!.w (memWSize mw)).get!.1 := by
      rw [memRel_ld_val (wf_mem_info h).2.2.1 hbd]
      simp only [SporosProvLen.LSt.ld, hbd, ↓reduceIte, Option.get!_some]
    rw [hval_eq] at hrd_memw
    have h5 : X86Vm.step vmImage s5 =
        .ok { s5.set 2 (some (ls.ld ls.ds[0]!.w (memWSize mw)).get!.1) with rip := rip0 + 44 + ldIdxLen mw } := by
      rw [step_at_vc c_ldidx rfl hm_ldidx, exec_ldIdx hs5_8 hs5_0 hrd_memw]
    let s6 := { s5.set 2 (some (ls.ld ls.ds[0]!.w (memWSize mw)).get!.1) with rip := rip0 + 44 + ldIdxLen mw }
    have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) _
    have hs6_2 : s6.reg 2 = some (ls.ld ls.ds[0]!.w (memWSize mw)).get!.1 :=
      reg_set_self_ok' hr5 (by decide) _
    have hs6_0 : s6.reg 0 = some ls.ds[0]!.w := by
      change (s5.set 2 _).reg 0 = some ls.ds[0]!.w; rw [reg_set_ne (by decide)]; exact hs5_0
    have hs6_4 : s6.reg 4 = some vc.sp := by
      change (s5.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs5_4
    have hs6_15 : s6.reg 15 = some ctxBase := by
      change (s5.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs5_15
    have hwf6 : vc.wf pr ls s6 := vc.wf_step_misc (s' := s6) hwf5 rfl hr6 hs6_4 hs6_15
    obtain ⟨s7, hs7, hrip7, hk7, hregs7, _, hwf7, h360_7, hoth7⟩ :=
      step_wr_ctx_scratch (off := 360) (step_at_vc c_st360 rfl hm_st360) hwf6 hs6_2 (by decide) (by decide)
    have hld_lt : (ls.ld ls.ds[0]!.w (memWSize mw)).get!.1 < X86Env.W := by
      simp only [SporosProvLen.LSt.ld, hbd, ↓reduceIte, Option.get!_some]
      have hpow_le : 256 ^ memWSize mw ≤ X86Env.W := by cases mw <;> simp [memWSize] <;> decide
      exact Nat.lt_of_lt_of_le (foldr_ld_lt_pow ls.mem ls.ds[0]!.w (memWSize mw)) hpow_le
    rw [Nat.mod_eq_of_lt hld_lt] at h360_7
    have ha7_eq : s7.rip = rip0 + 51 + ldIdxLen mw := by
      rw [hrip7]; dsimp [s6]; omega
    have h272_7 : rd64 s7.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 1) ls.pc) :=
      hoth7 272 _ (Or.inl (by decide)) (by decide) (by change rd64 s4.mem _ = _; rw [hmem4]; exact h272)
    have h280_7 : rd64 s7.mem (ea ctxBase 280) = some 1 :=
      hoth7 280 _ (Or.inl (by decide)) (by decide) (by change rd64 s4.mem _ = _; rw [hmem4]; exact h280)
    have h288_7 : rd64 s7.mem (ea ctxBase 288) = some 1 :=
      hoth7 288 _ (Or.inl (by decide)) (by decide) (by change rd64 s4.mem _ = _; rw [hmem4]; exact h288)
    have hmpb_7 : rd64 s7.mem (ea ctxBase 104) = some vc.mempReg.base :=
      hoth7 104 _ (Or.inl (by decide)) (by decide) (by change rd64 s4.mem _ = _; rw [hmem4]; exact rd_ctx_mempBase hwf3)
    have h7_15 : s7.reg 15 = some ctxBase := hwf7.2.2.2.2.2.1
    have h7_4 : s7.reg 4 = some vc.sp := hwf7.2.2.2.2.1
    have hr7 : RegsOk s7 := hwf7.2.2.2.1
    have h7_0 : s7.reg 0 = some ls.ds[0]!.w := by simpa [X86Int.St.reg, hregs7] using hs6_0
    have h7 : X86Vm.step vmImage s7 = .ok { s7.set 9 (some vc.mempReg.base) with rip := rip0 + 55 + ldIdxLen mw } := by
      rw [step_at_vc c_ld9 ha7_eq hm_ld9, exec_ld h7_15 hmpb_7, ha7_eq]
      have : rip0 + 51 + ldIdxLen mw + 4 = rip0 + 55 + ldIdxLen mw := by omega
      rw [this]
    let s8 := { s7.set 9 (some vc.mempReg.base) with rip := rip0 + 55 + ldIdxLen mw }
    have hr8 : RegsOk s8 := regsOk_set hr7 (by decide) _
    have hs8_9 : s8.reg 9 = some vc.mempReg.base := reg_set_self_ok' hr7 (by decide) _
    have hs8_0 : s8.reg 0 = some ls.ds[0]!.w := by
      change (s7.set 9 _).reg 0 = some ls.ds[0]!.w; rw [reg_set_ne (by decide)]; exact h7_0
    have hs8_15 : s8.reg 15 = some ctxBase := by
      change (s7.set 9 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h7_15
    have hs8_4 : s8.reg 4 = some vc.sp := by
      change (s7.set 9 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h7_4
    have htake1 : ls.ds.take 1 = [ls.ds[0]!] := by
      rcases hds : ls.ds with _ | ⟨x, r⟩ <;> simp [hds] at hle ⊢
    have h272_8 : rd64 s8.mem (ea ctxBase 272) = some (SporosProvLen.lq [ls.ds[0]!] ls.pc) := by
      change rd64 s7.mem (ea ctxBase 272) = _
      rw [← htake1]; exact h272_7
    have h8 : X86Vm.step vmImage s8 =
        .ok { s8.set 10 (some (SporosProvLen.lq [ls.ds[0]!] ls.pc)) with rip := rip0 + 62 + ldIdxLen mw } := by
      rw [step_at_vc c_ldq rfl hm_ldq, exec_ld hs8_15 h272_8]
      have : rip0 + 55 + ldIdxLen mw + 7 = rip0 + 62 + ldIdxLen mw := by omega
      rw [this]
    let s9 := { s8.set 10 (some (SporosProvLen.lq [ls.ds[0]!] ls.pc)) with rip := rip0 + 62 + ldIdxLen mw }
    have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) _
    have hs9_10 : s9.reg 10 = some (SporosProvLen.lq [ls.ds[0]!] ls.pc) :=
      reg_set_self_ok' hr8 (by decide) _
    have hs9_9 : s9.reg 9 = some vc.mempReg.base := by
      change (s8.set 10 _).reg 9 = some vc.mempReg.base; rw [reg_set_ne (by decide)]; exact hs8_9
    have hs9_0 : s9.reg 0 = some ls.ds[0]!.w := by
      change (s8.set 10 _).reg 0 = some ls.ds[0]!.w; rw [reg_set_ne (by decide)]; exact hs8_0
    have hs9_15 : s9.reg 15 = some ctxBase := by
      change (s8.set 10 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs8_15
    have hs9_4 : s9.reg 4 = some vc.sp := by
      change (s8.set 10 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs8_4
    let ctx' := { vc.ctxReg with data := writeBytes vc.ctxReg.data 360 (bytes8 (ls.ld ls.ds[0]!.w (memWSize mw)).get!.1) }
    let vc' := { vc with ctxReg := ctx' }
    have hwf9 : vc'.wf pr ls s9 := VmCtx.wf_step_misc (s' := s9) hwf7 rfl hr9 hs9_4 hs9_15
    have hrch9 : Reaches vmImage s s9 :=
      Reaches.trans hrch3 (Reaches.trans hrch4 (Reaches.step_trans h4s (Reaches.step_trans h5
        (Reaches.step_trans hs7 (Reaches.step_trans h7 (Reaches.step h8))))))
    have hk9 : s9.k = s.k := by
      rw [show s9.k = s7.k from rfl, hk7]
      change s4.k = s.k; exact hk4
    have hfin := hcont vc' s9 hbd hrch9 hk9 rfl hwf9 hr9 hs9_4 hs9_15 hs9_0 hs9_9 hs9_10 rfl rfl
      h280_7 h288_7 h360_7
    exact lift_primPostOutcome hrch9 hk9 hfin
  · rw [if_neg hbd] at hrc
    have hrch_fail := Reaches.trans hrch3 (hrc.choose_spec.1)
    rcases hds_eq : ls.ds with _ | ⟨a, r⟩ <;> try (simp [hds_eq] at hle)
    have hno : ¬(a.w + memWSize mw ≤ ls.msize) := by simpa [hds_eq] using hbd
    dsimp only [PrimPostOutcome]
    rw [hp_eff.1]
    have hpe := hp_eff.2.2 0 0
    simp only [hds_eq, List.take_succ_cons, List.take_zero, List.getElem!_cons_zero] at hpe ⊢
    rw [hpe, if_neg hno]
    exact ⟨hrc.choose, hrch_fail, hrc.choose_spec.2.1, hrc.choose_spec.2.2⟩

theorem ld_finish_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (mw : MemW) (p : Sporos.Prim) (qacc : Nat)
    {c_st c_jmp : Nat}
    (hp_eff : p.eff.1 = 1 ∧ p.eff.2 = 1 ∧
      ∀ (_ _ : Nat), SporosProvLen.lexecPrim p [ls.ds[0]!] ls =
        if ls.ds[0]!.w + memWSize mw ≤ ls.msize then
          .ok [{ w := (ls.ld ls.ds[0]!.w (memWSize mw)).get!.1,
                 plen := (SporosProvLen.lq [ls.ds[0]!] ls.pc + (ls.ld ls.ds[0]!.w (memWSize mw)).get!.2) % Sporos.M }] ls
        else .fail .bounds)
    (hm_st : (s.rip, (.ofInt (.st 15 368 10)), 7) ∈ vmTableChunk c_st)
    (hm_jmp : (s.rip + 7, (.ofInt (.jmp SporosVmBin.L_prim_epilogue)), 5) ∈ vmTableChunk c_jmp)
    (hbd : ls.ds[0]!.w + memWSize mw ≤ ls.msize)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (hle : 1 ≤ ls.ds.length) (hr10 : s.reg 10 = some qacc)
    (hq_eq : qacc % X86Env.W = (SporosProvLen.lq [ls.ds[0]!] ls.pc + (ls.ld ls.ds[0]!.w (memWSize mw)).get!.2) % Sporos.M)
    (h280 : rd64 s.mem (ea ctxBase 280) = some 1) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h360 : rd64 s.mem (ea ctxBase 360) = some (ls.ld ls.ds[0]!.w (memWSize mw)).get!.1) :
    PrimPostOutcome pr ls f rest s p := by
  obtain ⟨s1, hs1, hrip1, hk1, _, _, hwf1, h368_1, hoth1⟩ :=
    step_wr_ctx_scratch (off := 368) (step_at_vc c_st rfl hm_st) h hr10 (by decide) (by decide)
  rw [hq_eq] at h368_1
  have h280_1 := hoth1 280 _ (Or.inl (by decide)) (by decide) h280
  have h288_1 := hoth1 288 _ (Or.inl (by decide)) (by decide) h288
  have h360_1 := hoth1 360 _ (Or.inl (by decide)) (by decide) h360
  let v := (ls.ld ls.ds[0]!.w (memWSize mw)).get!.1
  let pm := (ls.ld ls.ds[0]!.w (memWSize mw)).get!.2
  let qout := (SporosProvLen.lq [ls.ds[0]!] ls.pc + pm) % Sporos.M
  let outVal : SporosProvLen.LVal := ⟨v, qout⟩
  have hvW : v < X86Env.W := by
    simp only [v, SporosProvLen.LSt.ld, hbd, ↓reduceIte, Option.get!_some]
    have hpow_le : 256 ^ memWSize mw ≤ X86Env.W := by cases mw <;> simp [memWSize] <;> decide
    exact Nat.lt_of_lt_of_le (foldr_ld_lt_pow ls.mem ls.ds[0]!.w (memWSize mw)) hpow_le
  have hqW : qout < X86Env.W := by dsimp [qout]; rw [← M_eq_W']; exact lt_W_of_mod _
  have hslots : ∀ (i : Nat) (_ : i < ([outVal] : List SporosProvLen.LVal).length),
      rd64 s1.mem (ea ctxBase (360 + i * 16)) = some ([outVal][i]).w ∧
      rd64 s1.mem (ea ctxBase (368 + i * 16)) = some ([outVal][i]).plen := by
    intro i hi; rcases i with _ | i
    · exact ⟨h360_1, h368_1⟩
    · simp at hi
  have houtsW : ∀ w ∈ ([outVal] : List SporosProvLen.LVal), w.w < X86Env.W ∧ w.plen < X86Env.W := by
    intro w hw; simp only [List.mem_singleton] at hw; subst hw; exact ⟨hvW, hqW⟩
  have hfin := jmp_epilogue0_finish_spec
    (step_at_vc c_jmp hrip1 hm_jmp)
    hwf1 (oracleAdmits_of_reaches hor (Reaches.step hs1)) hrs hpcW hle (by simp) houtsW h280_1 h288_1 hslots
  have hres := lift_next_outcome (Reaches.step hs1) hk1 hfin
  rcases hds_eq : ls.ds with _ | ⟨a, r⟩ <;> try (simp [hds_eq] at hle)
  have hpe := hp_eff.2.2 0 0
  dsimp only [PrimPostOutcome]
  rw [hp_eff.1]
  simp only [hds_eq, List.take_succ_cons, List.take_zero, List.getElem!_cons_zero] at hpe hres ⊢
  rw [hpe, if_pos (by simpa [hds_eq] using hbd)]
  simpa [outVal, v, qout, pm, hds_eq] using hres

theorem P_ld8_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_ld8) (hle : 1 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 1) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 1) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) :
    PrimPostOutcome pr ls f rest s .ld8 := by
  have hp_eff : Sporos.Prim.ld8.eff.1 = 1 ∧ Sporos.Prim.ld8.eff.2 = 1 ∧
      (∀ v pm : Nat, SporosProvLen.lexecPrim .ld8 [ls.ds[0]!] ls =
        if ls.ds[0]!.w + memWSize .w8 ≤ ls.msize then
          .ok [{ w := (ls.ld ls.ds[0]!.w (memWSize .w8)).get!.1,
                 plen := (SporosProvLen.lq [ls.ds[0]!] ls.pc + (ls.ld ls.ds[0]!.w (memWSize .w8)).get!.2) % Sporos.M }] ls
        else .fail .bounds) := by
    refine ⟨rfl, rfl, fun _ _ => ?_⟩
    by_cases hbd : ls.ds[0]!.w + 1 ≤ ls.msize <;>
      simp only [SporosProvLen.lexecPrim, SporosProvLen.LSt.ld, memWSize, hbd, ↓reduceIte, Option.get!_some]
  refine ld_prologue_spec .w8 .ld8 4196587
    (c_ld0 := 6) (c_mov1 := 6) (c_ld10 := 6) (c0 := 6) (c1 := 6) (c2 := 7) (c3 := 7) (c4 := 7) (c5 := 7)
    (c_ld8 := 7) (c_ldidx := 7) (c_st360 := 7) (c_ld9 := 7) (c_ldq := 7)
    hp_eff (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    hrip hle h hor hrs hpcW h272 h280 h288 h296 ?_
  intro vc' s' hbd _hrch _hk hrip' hwf' hr' hs4 hs15 hs0 hs9 hs10 hmp hsp h280' h288' h360'
  obtain ⟨s1, hrch1, hrip1, hmem1, hk1, hr1, hs1_4, hs1_15, _hs1_0, _hs1_9, hs1_10⟩ :=
    ld_memp_sum_step (chk := 7) (i := 0) hrip' (by decide) (by decide) hwf' (by simp only [memWSize] at hbd; omega)
      hr' (by simpa [hmp] using hs9) hs0 hs10 (by simpa [hsp] using hs4) hs15
  have hwf1 : vc'.wf pr ls s1 := vc'.wf_step_misc (s' := s1) hwf' hmem1 hr1 hs1_4 hs1_15
  have hqW : SporosProvLen.lq [ls.ds[0]!] ls.pc < X86Env.W := by
    dsimp [SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
  have hq_eq : ((SporosProvLen.lq [ls.ds[0]!] ls.pc) % X86Env.W + (ls.mem (ls.ds[0]!.w + 0)).plen) % X86Env.W =
      (SporosProvLen.lq [ls.ds[0]!] ls.pc + (ls.ld ls.ds[0]!.w (memWSize .w8)).get!.2) % Sporos.M := by
    have hbd' : ls.ds[0]!.w + 1 ≤ ls.msize := hbd
    rw [Nat.mod_eq_of_lt hqW, M_eq_W']
    simp only [SporosProvLen.LSt.ld, memWSize, hbd', ↓reduceIte, Option.get!_some,
      show List.range 1 = [0] from rfl, List.foldr_cons, List.foldr_nil, Nat.add_zero]
  have hfin := ld_finish_spec (vc := vc') (ls := ls) .w8 .ld8 _ (c_st := 7) (c_jmp := 7)
    hp_eff (by rw [hrip1]; decide) (by rw [hrip1]; decide) hbd hwf1
    (oracleAdmits_of_reaches (oracleAdmits_of_reaches hor _hrch) hrch1) hrs hpcW hle hs1_10
    (by rw [Nat.mod_mod]; exact hq_eq)
    (by simpa [hmem1] using h280') (by simpa [hmem1] using h288') (by simpa [hmem1] using h360')
  exact lift_primPostOutcome hrch1 hk1 hfin

theorem P_ld16_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_ld16) (hle : 1 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 1) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 1) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) :
    PrimPostOutcome pr ls f rest s .ld16 := by
  have hp_eff : Sporos.Prim.ld16.eff.1 = 1 ∧ Sporos.Prim.ld16.eff.2 = 1 ∧
      (∀ v pm : Nat, SporosProvLen.lexecPrim .ld16 [ls.ds[0]!] ls =
        if ls.ds[0]!.w + memWSize .w16 ≤ ls.msize then
          .ok [{ w := (ls.ld ls.ds[0]!.w (memWSize .w16)).get!.1,
                 plen := (SporosProvLen.lq [ls.ds[0]!] ls.pc + (ls.ld ls.ds[0]!.w (memWSize .w16)).get!.2) % Sporos.M }] ls
        else .fail .bounds) := by
    refine ⟨rfl, rfl, fun _ _ => ?_⟩
    by_cases hbd : ls.ds[0]!.w + 2 ≤ ls.msize <;>
      simp only [SporosProvLen.lexecPrim, SporosProvLen.LSt.ld, memWSize, hbd, ↓reduceIte, Option.get!_some]
  refine ld_prologue_spec .w16 .ld16 4196673
    (c_ld0 := 7) (c_mov1 := 7) (c_ld10 := 7) (c0 := 7) (c1 := 7) (c2 := 7) (c3 := 7) (c4 := 7) (c5 := 7)
    (c_ld8 := 7) (c_ldidx := 7) (c_st360 := 7) (c_ld9 := 7) (c_ldq := 7)
    hp_eff (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    hrip hle h hor hrs hpcW h272 h280 h288 h296 ?_
  intro vc' s' hbd _hrch _hk hrip' hwf' hr' hs4 hs15 hs0 hs9 hs10 hmp hsp h280' h288' h360'
  obtain ⟨s1, hrch1, hrip1, hmem1, hk1, hr1, hs1_4, hs1_15, hs1_0, hs1_9, hs1_10⟩ :=
    ld_memp_sum_step (chk := 7) (i := 0) hrip' (by decide) (by decide) hwf' (by simp only [memWSize] at hbd; omega)
      hr' (by simpa [hmp] using hs9) hs0 hs10 (by simpa [hsp] using hs4) hs15
  have hwf1 : vc'.wf pr ls s1 := vc'.wf_step_misc (s' := s1) hwf' hmem1 hr1 hs1_4 hs1_15
  obtain ⟨s2, hrch2, hrip2, hmem2, hk2, hr2, hs2_4, hs2_15, _hs2_0, _hs2_9, hs2_10⟩ :=
    ld_memp_sum_step (chk := 7) (i := 1) hrip1 (by decide) (by decide) hwf1 (by simp only [memWSize] at hbd; omega)
      hr1 hs1_9 hs1_0 hs1_10 hs1_4 hs1_15
  have hwf2 : vc'.wf pr ls s2 := vc'.wf_step_misc (s' := s2) hwf1 hmem2 hr2 hs2_4 hs2_15
  have hqW : SporosProvLen.lq [ls.ds[0]!] ls.pc < X86Env.W := by
    dsimp [SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
  have hq_eq : ((((SporosProvLen.lq [ls.ds[0]!] ls.pc) % X86Env.W + (ls.mem (ls.ds[0]!.w + 0)).plen) % X86Env.W) % X86Env.W +
      (ls.mem (ls.ds[0]!.w + 1)).plen) % X86Env.W =
      (SporosProvLen.lq [ls.ds[0]!] ls.pc + (ls.ld ls.ds[0]!.w (memWSize .w16)).get!.2) % Sporos.M := by
    have hbd' : ls.ds[0]!.w + 2 ≤ ls.msize := hbd
    rw [Nat.mod_eq_of_lt hqW, Nat.mod_mod, M_eq_W', Nat.add_mod, Nat.mod_mod, ← Nat.add_mod]
    simp only [SporosProvLen.LSt.ld, memWSize, hbd', ↓reduceIte, Option.get!_some,
      show List.range 2 = [0, 1] from rfl, List.foldr_cons, List.foldr_nil, Nat.add_zero]
    congr 1; omega
  have hrch12 := Reaches.trans hrch1 hrch2
  have hk12 : s2.k = s'.k := by rw [hk2, hk1]
  have hfin := ld_finish_spec (vc := vc') (ls := ls) .w16 .ld16 _ (c_st := 7) (c_jmp := 7)
    hp_eff (by rw [hrip2]; decide) (by rw [hrip2]; decide) hbd hwf2
    (oracleAdmits_of_reaches (oracleAdmits_of_reaches hor _hrch) hrch12) hrs hpcW hle hs2_10
    (by rw [Nat.mod_mod]; exact hq_eq)
    (by simpa [hmem2, hmem1] using h280') (by simpa [hmem2, hmem1] using h288')
    (by simpa [hmem2, hmem1] using h360')
  exact lift_primPostOutcome hrch12 hk12 hfin

theorem P_ld32_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_ld32) (hle : 1 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 1) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 1) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) :
    PrimPostOutcome pr ls f rest s .ld32 := by
  have hp_eff : Sporos.Prim.ld32.eff.1 = 1 ∧ Sporos.Prim.ld32.eff.2 = 1 ∧
      (∀ v pm : Nat, SporosProvLen.lexecPrim .ld32 [ls.ds[0]!] ls =
        if ls.ds[0]!.w + memWSize .w32 ≤ ls.msize then
          .ok [{ w := (ls.ld ls.ds[0]!.w (memWSize .w32)).get!.1,
                 plen := (SporosProvLen.lq [ls.ds[0]!] ls.pc + (ls.ld ls.ds[0]!.w (memWSize .w32)).get!.2) % Sporos.M }] ls
        else .fail .bounds) := by
    refine ⟨rfl, rfl, fun _ _ => ?_⟩
    by_cases hbd : ls.ds[0]!.w + 4 ≤ ls.msize <;>
      simp only [SporosProvLen.lexecPrim, SporosProvLen.LSt.ld, memWSize, hbd, ↓reduceIte, Option.get!_some]
  refine ld_prologue_spec .w32 .ld32 4196767
    (c_ld0 := 7) (c_mov1 := 7) (c_ld10 := 7) (c0 := 7) (c1 := 7) (c2 := 7) (c3 := 7) (c4 := 7) (c5 := 7)
    (c_ld8 := 7) (c_ldidx := 7) (c_st360 := 7) (c_ld9 := 7) (c_ldq := 7)
    hp_eff (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    hrip hle h hor hrs hpcW h272 h280 h288 h296 ?_
  intro vc' s' hbd _hrch _hk hrip' hwf' hr' hs4 hs15 hs0 hs9 hs10 hmp hsp h280' h288' h360'
  obtain ⟨s1, hrch1, hrip1, hmem1, hk1, hr1, hs1_4, hs1_15, hs1_0, hs1_9, hs1_10⟩ :=
    ld_memp_sum_step (chk := 7) (i := 0) hrip' (by decide) (by decide) hwf' (by simp only [memWSize] at hbd; omega)
      hr' (by simpa [hmp] using hs9) hs0 hs10 (by simpa [hsp] using hs4) hs15
  have hwf1 : vc'.wf pr ls s1 := vc'.wf_step_misc (s' := s1) hwf' hmem1 hr1 hs1_4 hs1_15
  obtain ⟨s2, hrch2, hrip2, hmem2, hk2, hr2, hs2_4, hs2_15, hs2_0, hs2_9, hs2_10⟩ :=
    ld_memp_sum_step (chk := 7) (i := 1) hrip1 (by decide) (by decide) hwf1 (by simp only [memWSize] at hbd; omega)
      hr1 hs1_9 hs1_0 hs1_10 hs1_4 hs1_15
  have hwf2 : vc'.wf pr ls s2 := vc'.wf_step_misc (s' := s2) hwf1 hmem2 hr2 hs2_4 hs2_15
  obtain ⟨s3, hrch3, hrip3, hmem3, hk3, hr3, hs3_4, hs3_15, hs3_0, hs3_9, hs3_10⟩ :=
    ld_memp_sum_step (chk := 7) (i := 2) hrip2 (by decide) (by decide) hwf2 (by simp only [memWSize] at hbd; omega)
      hr2 hs2_9 hs2_0 hs2_10 hs2_4 hs2_15
  have hwf3 : vc'.wf pr ls s3 := vc'.wf_step_misc (s' := s3) hwf2 hmem3 hr3 hs3_4 hs3_15
  obtain ⟨s4, hrch4, hrip4, hmem4, hk4, hr4, hs4_4, hs4_15, _hs4_0, _hs4_9, hs4_10⟩ :=
    ld_memp_sum_step (chk := 7) (i := 3) hrip3 (by decide) (by decide) hwf3 (by simp only [memWSize] at hbd; omega)
      hr3 hs3_9 hs3_0 hs3_10 hs3_4 hs3_15
  have hwf4 : vc'.wf pr ls s4 := vc'.wf_step_misc (s' := s4) hwf3 hmem4 hr4 hs4_4 hs4_15
  have hqW : SporosProvLen.lq [ls.ds[0]!] ls.pc < X86Env.W := by
    dsimp [SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
  have hq_eq : (((((((SporosProvLen.lq [ls.ds[0]!] ls.pc) % X86Env.W +
      (ls.mem (ls.ds[0]!.w + 0)).plen) % X86Env.W) % X86Env.W +
      (ls.mem (ls.ds[0]!.w + 1)).plen) % X86Env.W) % X86Env.W +
      (ls.mem (ls.ds[0]!.w + 2)).plen) % X86Env.W % X86Env.W +
      (ls.mem (ls.ds[0]!.w + 3)).plen) % X86Env.W =
      (SporosProvLen.lq [ls.ds[0]!] ls.pc + (ls.ld ls.ds[0]!.w (memWSize .w32)).get!.2) % Sporos.M := by
    have hbd' : ls.ds[0]!.w + 4 ≤ ls.msize := hbd
    rw [Nat.mod_eq_of_lt hqW, M_eq_W']
    simp only [SporosProvLen.LSt.ld, memWSize, hbd', ↓reduceIte, Option.get!_some,
      show List.range 4 = [0, 1, 2, 3] from rfl, List.foldr_cons, List.foldr_nil, Nat.add_zero]
    simp only [Nat.mod_mod, mod_add_left]
    congr 1; omega
  have hrch14 := Reaches.trans hrch1 (Reaches.trans hrch2 (Reaches.trans hrch3 hrch4))
  have hk14 : s4.k = s'.k := by rw [hk4, hk3, hk2, hk1]
  have hfin := ld_finish_spec (vc := vc') (ls := ls) .w32 .ld32 _ (c_st := 7) (c_jmp := 7)
    hp_eff (by rw [hrip4]; decide) (by rw [hrip4]; decide) hbd hwf4
    (oracleAdmits_of_reaches (oracleAdmits_of_reaches hor _hrch) hrch14) hrs hpcW hle hs4_10
    (by rw [Nat.mod_mod]; exact hq_eq)
    (by simpa [hmem4, hmem3, hmem2, hmem1] using h280')
    (by simpa [hmem4, hmem3, hmem2, hmem1] using h288')
    (by simpa [hmem4, hmem3, hmem2, hmem1] using h360')
  exact lift_primPostOutcome hrch14 hk14 hfin

theorem P_ld64_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_ld64) (hle : 1 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 1) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 1) (h288 : rd64 s.mem (ea ctxBase 288) = some 1)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w) :
    PrimPostOutcome pr ls f rest s .ld64 := by
  have hp_eff : Sporos.Prim.ld64.eff.1 = 1 ∧ Sporos.Prim.ld64.eff.2 = 1 ∧
      (∀ v pm : Nat, SporosProvLen.lexecPrim .ld64 [ls.ds[0]!] ls =
        if ls.ds[0]!.w + memWSize .w64 ≤ ls.msize then
          .ok [{ w := (ls.ld ls.ds[0]!.w (memWSize .w64)).get!.1,
                 plen := (SporosProvLen.lq [ls.ds[0]!] ls.pc + (ls.ld ls.ds[0]!.w (memWSize .w64)).get!.2) % Sporos.M }] ls
        else .fail .bounds) := by
    refine ⟨rfl, rfl, fun _ _ => ?_⟩
    by_cases hbd : ls.ds[0]!.w + 8 ≤ ls.msize <;>
      simp only [SporosProvLen.lexecPrim, SporosProvLen.LSt.ld, memWSize, hbd, ↓reduceIte, Option.get!_some]
  refine ld_prologue_spec .w64 .ld64 4196876
    (c_ld0 := 7) (c_mov1 := 7) (c_ld10 := 7) (c0 := 7) (c1 := 7) (c2 := 7) (c3 := 7) (c4 := 8) (c5 := 8)
    (c_ld8 := 8) (c_ldidx := 8) (c_st360 := 8) (c_ld9 := 8) (c_ldq := 8)
    hp_eff (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    hrip hle h hor hrs hpcW h272 h280 h288 h296 ?_
  intro vc' s' hbd _hrch _hk hrip' hwf' hr' hs4 hs15 hs0 hs9 hs10 hmp hsp h280' h288' h360'
  obtain ⟨s1, hrch1, hrip1, hmem1, hk1, hr1, hs1_4, hs1_15, hs1_0, hs1_9, hs1_10⟩ :=
    ld_memp_sum_step (chk := 8) (i := 0) hrip' (by decide) (by decide) hwf' (by simp only [memWSize] at hbd; omega)
      hr' (by simpa [hmp] using hs9) hs0 hs10 (by simpa [hsp] using hs4) hs15
  have hwf1 : vc'.wf pr ls s1 := vc'.wf_step_misc (s' := s1) hwf' hmem1 hr1 hs1_4 hs1_15
  obtain ⟨s2, hrch2, hrip2, hmem2, hk2, hr2, hs2_4, hs2_15, hs2_0, hs2_9, hs2_10⟩ :=
    ld_memp_sum_step (chk := 8) (i := 1) hrip1 (by decide) (by decide) hwf1 (by simp only [memWSize] at hbd; omega)
      hr1 hs1_9 hs1_0 hs1_10 hs1_4 hs1_15
  have hwf2 : vc'.wf pr ls s2 := vc'.wf_step_misc (s' := s2) hwf1 hmem2 hr2 hs2_4 hs2_15
  obtain ⟨s3, hrch3, hrip3, hmem3, hk3, hr3, hs3_4, hs3_15, hs3_0, hs3_9, hs3_10⟩ :=
    ld_memp_sum_step (chk := 8) (i := 2) hrip2 (by decide) (by decide) hwf2 (by simp only [memWSize] at hbd; omega)
      hr2 hs2_9 hs2_0 hs2_10 hs2_4 hs2_15
  have hwf3 : vc'.wf pr ls s3 := vc'.wf_step_misc (s' := s3) hwf2 hmem3 hr3 hs3_4 hs3_15
  obtain ⟨s4, hrch4, hrip4, hmem4, hk4, hr4, hs4_4, hs4_15, hs4_0, hs4_9, hs4_10⟩ :=
    ld_memp_sum_step (chk := 8) (i := 3) hrip3 (by decide) (by decide) hwf3 (by simp only [memWSize] at hbd; omega)
      hr3 hs3_9 hs3_0 hs3_10 hs3_4 hs3_15
  have hwf4 : vc'.wf pr ls s4 := vc'.wf_step_misc (s' := s4) hwf3 hmem4 hr4 hs4_4 hs4_15
  obtain ⟨s5, hrch5, hrip5, hmem5, hk5, hr5, hs5_4, hs5_15, hs5_0, hs5_9, hs5_10⟩ :=
    ld_memp_sum_step (chk := 8) (i := 4) hrip4 (by decide) (by decide) hwf4 (by simp only [memWSize] at hbd; omega)
      hr4 hs4_9 hs4_0 hs4_10 hs4_4 hs4_15
  have hwf5 : vc'.wf pr ls s5 := vc'.wf_step_misc (s' := s5) hwf4 hmem5 hr5 hs5_4 hs5_15
  obtain ⟨s6, hrch6, hrip6, hmem6, hk6, hr6, hs6_4, hs6_15, hs6_0, hs6_9, hs6_10⟩ :=
    ld_memp_sum_step (chk := 8) (i := 5) hrip5 (by decide) (by decide) hwf5 (by simp only [memWSize] at hbd; omega)
      hr5 hs5_9 hs5_0 hs5_10 hs5_4 hs5_15
  have hwf6 : vc'.wf pr ls s6 := vc'.wf_step_misc (s' := s6) hwf5 hmem6 hr6 hs6_4 hs6_15
  obtain ⟨s7, hrch7, hrip7, hmem7, hk7, hr7, hs7_4, hs7_15, hs7_0, hs7_9, hs7_10⟩ :=
    ld_memp_sum_step (chk := 8) (i := 6) hrip6 (by decide) (by decide) hwf6 (by simp only [memWSize] at hbd; omega)
      hr6 hs6_9 hs6_0 hs6_10 hs6_4 hs6_15
  have hwf7 : vc'.wf pr ls s7 := vc'.wf_step_misc (s' := s7) hwf6 hmem7 hr7 hs7_4 hs7_15
  obtain ⟨s8, hrch8, hrip8, hmem8, hk8, hr8, hs8_4, hs8_15, _hs8_0, _hs8_9, hs8_10⟩ :=
    ld_memp_sum_step (chk := 8) (i := 7) hrip7 (by decide) (by decide) hwf7 (by simp only [memWSize] at hbd; omega)
      hr7 hs7_9 hs7_0 hs7_10 hs7_4 hs7_15
  have hwf8 : vc'.wf pr ls s8 := vc'.wf_step_misc (s' := s8) hwf7 hmem8 hr8 hs8_4 hs8_15
  have hqW : SporosProvLen.lq [ls.ds[0]!] ls.pc < X86Env.W := by
    dsimp [SporosProvLen.lq]; rw [← M_eq_W']; exact lt_W_of_mod _
  have hq_eq : (((((((((((((((SporosProvLen.lq [ls.ds[0]!] ls.pc) % X86Env.W +
      (ls.mem (ls.ds[0]!.w + 0)).plen) % X86Env.W) % X86Env.W +
      (ls.mem (ls.ds[0]!.w + 1)).plen) % X86Env.W) % X86Env.W +
      (ls.mem (ls.ds[0]!.w + 2)).plen) % X86Env.W) % X86Env.W +
      (ls.mem (ls.ds[0]!.w + 3)).plen) % X86Env.W) % X86Env.W +
      (ls.mem (ls.ds[0]!.w + 4)).plen) % X86Env.W) % X86Env.W +
      (ls.mem (ls.ds[0]!.w + 5)).plen) % X86Env.W) % X86Env.W +
      (ls.mem (ls.ds[0]!.w + 6)).plen) % X86Env.W % X86Env.W +
      (ls.mem (ls.ds[0]!.w + 7)).plen) % X86Env.W =
      (SporosProvLen.lq [ls.ds[0]!] ls.pc + (ls.ld ls.ds[0]!.w (memWSize .w64)).get!.2) % Sporos.M := by
    have hbd' : ls.ds[0]!.w + 8 ≤ ls.msize := hbd
    rw [Nat.mod_eq_of_lt hqW, M_eq_W']
    simp only [SporosProvLen.LSt.ld, memWSize, hbd', ↓reduceIte, Option.get!_some,
      show List.range 8 = [0, 1, 2, 3, 4, 5, 6, 7] from rfl, List.foldr_cons, List.foldr_nil, Nat.add_zero]
    simp only [Nat.mod_mod, mod_add_left]
    congr 1; omega
  have hrch18 := Reaches.trans hrch1 (Reaches.trans hrch2 (Reaches.trans hrch3 (Reaches.trans hrch4
    (Reaches.trans hrch5 (Reaches.trans hrch6 (Reaches.trans hrch7 hrch8))))))
  have hk18 : s8.k = s'.k := by rw [hk8, hk7, hk6, hk5, hk4, hk3, hk2, hk1]
  have hfin := ld_finish_spec (vc := vc') (ls := ls) .w64 .ld64 _ (c_st := 8) (c_jmp := 8)
    hp_eff (by rw [hrip8]; decide) (by rw [hrip8]; decide) hbd hwf8
    (oracleAdmits_of_reaches (oracleAdmits_of_reaches hor _hrch) hrch18) hrs hpcW hle hs8_10
    (by rw [Nat.mod_mod]; exact hq_eq)
    (by simpa [hmem8, hmem7, hmem6, hmem5, hmem4, hmem3, hmem2, hmem1] using h280')
    (by simpa [hmem8, hmem7, hmem6, hmem5, hmem4, hmem3, hmem2, hmem1] using h288')
    (by simpa [hmem8, hmem7, hmem6, hmem5, hmem4, hmem3, hmem2, hmem1] using h360')
  exact lift_primPostOutcome hrch18 hk18 hfin

/-! ### Shared `P_st{8,16,32,64}` helpers and specs -/

theorem st_memp_qword_step {vc : VmCtx} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {mw_cur mp_cur : Region} {a i q rip0 chk : Nat}
    (hm_st : (rip0, (.stIdx .w64 9 0 8 (8 * i) 10), if i = 0 then 4 else 5) ∈ vmTableChunk chk)
    (hrip : s.rip = rip0)
    (hcm : CodeMem s.mem) (hbl : BytesLt s.mem) (hpos : ∀ u ∈ s.mem, 0 < u.data.length)
    (hmp_mem : mp_cur ∈ s.mem) (hmw_mem : mw_cur ∈ s.mem) (hctx_mem : vc.ctxReg ∈ s.mem)
    (hmp_base : mp_cur.base = vc.mempReg.base) (hmp_rw : mp_cur.rw = true)
    (hmp_len : mp_cur.data.length = vc.mempReg.data.length)
    (hmw_base : mw_cur.base = vc.memwReg.base) (hctx_base : vc.ctxReg.base = ctxBase)
    (hhi_memp : vc.mempReg.hi < X86Env.W)
    (hne_mp_mw : vc.mempReg.base ≠ vc.memwReg.base)
    (hne_mp_ctx : vc.mempReg.base ≠ ctxBase)
    (hlt : a + i < ls.msize) (hlen_mp : ls.msize * 8 ≤ vc.mempReg.data.length)
    (hr9 : s.reg 9 = some vc.mempReg.base) (hr0 : s.reg 0 = some a) (hr10 : s.reg 10 = some q) :
    let mp' := { mp_cur with data := writeBytes mp_cur.data ((a + i) * 8) (bytes8 q) }
    ∃ s',
      X86Vm.step vmImage s = .ok s' ∧
      s'.rip = rip0 + (if i = 0 then 4 else 5) ∧ s'.regs = s.regs ∧ s'.fl = s.fl ∧ s'.k = s.k ∧
      CodeMem s'.mem ∧ BytesLt s'.mem ∧ (∀ u ∈ s'.mem, 0 < u.data.length) ∧
      mp' ∈ s'.mem ∧ mw_cur ∈ s'.mem ∧ vc.ctxReg ∈ s'.mem ∧
      (∀ u ∈ s.mem, u.base ≠ vc.mempReg.base → u ∈ s'.mem) := by
  have hea : eaIdx vc.mempReg.base a 8 (8 * i) = mp_cur.base + (a + i) * 8 := by
    have hltW : vc.mempReg.base + 8 * a + 8 * i < X86Env.W := by
      simp only [Region.hi] at hhi_memp; omega
    rw [eaIdx_add_eq hltW, hmp_base]; omega
  have hlo : mp_cur.base ≤ mp_cur.base + (a + i) * 8 := Nat.le_add_right _ _
  have hhi : mp_cur.base + (a + i) * 8 + 8 ≤ mp_cur.hi := by
    simp only [Region.hi, hmp_len]; omega
  obtain ⟨idx, m', hwr, _, hm', hm'wf, hmp', hoth⟩ :=
    wrMem_w64_region hcm.2 hmp_mem hmp_rw hlo hhi
  rw [Nat.add_sub_cancel_left] at hm' hmp'
  rw [← hea] at hwr
  have hs : X86Vm.step vmImage s = .ok { s with mem := m', rip := rip0 + (if i = 0 then 4 else 5) } := by
    rw [step_at_vc chk hrip hm_st, exec_stIdx hr9 hr0 hr10 hwr, hrip]
  refine ⟨{ s with mem := m', rip := rip0 + (if i = 0 then 4 else 5) }, hs, rfl, rfl, rfl, rfl,
    hcm.wr64_preserve hwr, bytesLt_wr64 hbl hwr, len_pos_wrN hpos hwr,
    hmp', hoth mw_cur hmw_mem (by rw [hmw_base, hmp_base]; exact Ne.symm hne_mp_mw),
    hoth vc.ctxReg hctx_mem (by rw [hctx_base, hmp_base]; exact Ne.symm hne_mp_ctx),
    fun u hu hne => hoth u hu (by rw [hmp_base]; exact hne)⟩

theorem st_prologue_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (mw : MemW) (p : Sporos.Prim) (rip0 : Nat)
    {c_ld0 c_mov1 c_ld10 c0 c1 c2 c3 c4 c5 c_ld8 c_ld2 c_stidx c_ld9 c_ldq : Nat}
    (hp_eff : p.eff.1 = 2 ∧ p.eff.2 = 0 ∧
      SporosProvLen.lexecPrim p [ls.ds[0]!, ls.ds[1]!] ls =
        match ls.st ls.ds[0]!.w (memWSize mw) ls.ds[1]!.w
            (SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!] ls.pc) with
        | some ls' => .ok [] ls'
        | none => .fail .bounds)
    (hm_ld0 : (rip0, (.ofInt (.ld 0 15 296)), 7) ∈ vmTableChunk c_ld0)
    (hm_mov1 : (rip0 + 7, (.ofInt (.movI 1 (memWSize mw))), 5) ∈ vmTableChunk c_mov1)
    (hm_ld10 : (rip0 + 12, (.ofInt (.ld 10 15 112)), 4) ∈ vmTableChunk c_ld10)
    (hm_rc0 : (rip0 + 16, (.ofInt (.cmp 0 10)), 3) ∈ vmTableChunk c0)
    (hm_rc1 : (rip0 + 19, (.ofInt (.jcc 7 SporosVmBin.L_fail_bounds)), 6) ∈ vmTableChunk c1)
    (hm_rc2 : (rip0 + 25, (.ofInt (.mov 11 10)), 3) ∈ vmTableChunk c2)
    (hm_rc3 : (rip0 + 28, (.ofInt (.alu .sub 11 0)), 3) ∈ vmTableChunk c3)
    (hm_rc4 : (rip0 + 31, (.ofInt (.cmp 1 11)), 3) ∈ vmTableChunk c4)
    (hm_rc5 : (rip0 + 34, (.ofInt (.jcc 7 SporosVmBin.L_fail_bounds)), 6) ∈ vmTableChunk c5)
    (hm_ld8 : (rip0 + 40, (.ofInt (.ld 8 15 96)), 4) ∈ vmTableChunk c_ld8)
    (hm_ld2 : (rip0 + 44, (.ofInt (.ld 2 15 312)), 7) ∈ vmTableChunk c_ld2)
    (hm_stidx : (rip0 + 51, (.stIdx mw 8 0 1 0 2), stIdxLen mw) ∈ vmTableChunk c_stidx)
    (hm_ld9 : (rip0 + 51 + stIdxLen mw, (.ofInt (.ld 9 15 104)), 4) ∈ vmTableChunk c_ld9)
    (hm_ldq : (rip0 + 55 + stIdxLen mw, (.ofInt (.ld 10 15 272)), 7) ∈ vmTableChunk c_ldq)
    (hrip : s.rip = rip0) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (_hor : OracleAdmits s) (_hrs : ls.rs = f :: rest) (_hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (_h280 : rd64 s.mem (ea ctxBase 280) = some 2) (_h288 : rd64 s.mem (ea ctxBase 288) = some 0)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w)
    (hcont : ∀ (s' : X86Vm.St),
      ls.ds[0]!.w + memWSize mw ≤ ls.msize →
      Reaches vmImage s s' → s'.k = s.k →
      s'.rip = rip0 + 62 + stIdxLen mw →
      CodeMem s'.mem → BytesLt s'.mem → (∀ u ∈ s'.mem, 0 < u.data.length) → RegsOk s' →
      { vc.memwReg with data := writeBytes vc.memwReg.data ls.ds[0]!.w (memWBytes mw ls.ds[1]!.w) } ∈ s'.mem →
      vc.mempReg ∈ s'.mem → vc.ctxReg ∈ s'.mem →
      (∀ u ∈ s.mem, u.base ≠ vc.memwReg.base → u ∈ s'.mem) →
      s'.reg 4 = some vc.sp → s'.reg 15 = some ctxBase →
      s'.reg 0 = some ls.ds[0]!.w → s'.reg 9 = some vc.mempReg.base →
      s'.reg 10 = some (SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!] ls.pc) →
      PrimPostOutcome pr ls f rest s p) :
    PrimPostOutcome pr ls f rest s p := by
  have _ := hp_eff
  have h15 : s.reg 15 = some ctxBase := h.2.2.2.2.2.1
  have h4 : s.reg 4 = some vc.sp := h.2.2.2.2.1
  have hreg : RegsOk s := h.2.2.2.1
  have haW : ls.ds[0]!.w < X86Env.W := (wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 0 < ls.ds.length))).1
  have _hvW : ls.ds[1]!.w < X86Env.W := (wf_ds_lt_W h (getElem!_mem_of_lt (by omega : 1 < ls.ds.length))).1
  have h0 : X86Vm.step vmImage s = .ok { s.set 0 (some ls.ds[0]!.w) with rip := rip0 + 7 } := by
    rw [step_at_vc c_ld0 hrip hm_ld0, exec_ld h15 h296, hrip]
  let s1 := { s.set 0 (some ls.ds[0]!.w) with rip := rip0 + 7 }
  have hr1 : RegsOk s1 := regsOk_set hreg (by decide) _
  have hs1_0 : s1.reg 0 = some ls.ds[0]!.w := reg_set_self_ok' hreg (by decide) _
  have hs1_15 : s1.reg 15 = some ctxBase := by
    change (s.set 0 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact h15
  have hs1_4 : s1.reg 4 = some vc.sp := by
    change (s.set 0 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact h4
  have hkW : memWSize mw < X86Env.W := by cases mw <;> simp [memWSize] <;> decide
  have h1 : X86Vm.step vmImage s1 = .ok { s1.set 1 (some (memWSize mw % X86Env.W)) with rip := rip0 + 12 } := by
    rw [step_at_vc c_mov1 rfl hm_mov1, exec_movI]
  let s2 := { s1.set 1 (some (memWSize mw)) with rip := rip0 + 12 }
  have h1' : X86Vm.step vmImage s1 = .ok s2 := by simpa [s2, Nat.mod_eq_of_lt hkW] using h1
  have hr2 : RegsOk s2 := regsOk_set hr1 (by decide) _
  have hs2_1 : s2.reg 1 = some (memWSize mw) := reg_set_self_ok' hr1 (by decide) _
  have hs2_0 : s2.reg 0 = some ls.ds[0]!.w := by
    change (s1.set 1 _).reg 0 = some ls.ds[0]!.w; rw [reg_set_ne (by decide)]; exact hs1_0
  have hs2_15 : s2.reg 15 = some ctxBase := by
    change (s1.set 1 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs1_15
  have hs2_4 : s2.reg 4 = some vc.sp := by
    change (s1.set 1 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs1_4
  have hms2 : rd64 s2.mem (ea ctxBase 112) = some ls.msize := by
    change rd64 s.mem (ea ctxBase 112) = some ls.msize; exact rd_ctx_msize h
  have h2 : X86Vm.step vmImage s2 = .ok { s2.set 10 (some ls.msize) with rip := rip0 + 16 } := by
    rw [step_at_vc c_ld10 rfl hm_ld10, exec_ld hs2_15 hms2]
  let s3 := { s2.set 10 (some ls.msize) with rip := rip0 + 16 }
  have hr3 : RegsOk s3 := regsOk_set hr2 (by decide) ls.msize
  have hs3_10 : s3.reg 10 = some ls.msize := reg_set_self_ok' hr2 (by decide) _
  have hs3_0 : s3.reg 0 = some ls.ds[0]!.w := by
    change (s2.set 10 _).reg 0 = some ls.ds[0]!.w; rw [reg_set_ne (by decide)]; exact hs2_0
  have hs3_1 : s3.reg 1 = some (memWSize mw) := by
    change (s2.set 10 _).reg 1 = some (memWSize mw); rw [reg_set_ne (by decide)]; exact hs2_1
  have hs3_4 : s3.reg 4 = some vc.sp := by
    change (s2.set 10 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs2_4
  have hs3_15 : s3.reg 15 = some ctxBase := by
    change (s2.set 10 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs2_15
  have hwf3 : vc.wf pr ls s3 := vc.wf_step_misc (s' := s3) h rfl hr3 hs3_4 hs3_15
  have hrch3 : Reaches vmImage s s3 :=
    Reaches.step_trans h0 (Reaches.step_trans h1' (Reaches.step h2))
  have hrc := range_check_spec (vc := vc) (pr := pr) (ls := ls) (s := s3) (ra := 0) (rlen := 1)
    (a := ls.ds[0]!.w) (len := memWSize mw) (rip0 := rip0 + 16)
    hm_rc0 hm_rc1 hm_rc2 hm_rc3 hm_rc4 hm_rc5 rfl hwf3 hs3_0 hs3_1 hs3_10 haW hkW
    (by decide) (by decide)
  by_cases hbd : ls.ds[0]!.w + memWSize mw ≤ ls.msize
  · rw [if_pos hbd] at hrc
    obtain ⟨s4, hrch4, hrip4, hmem4, hk4, hr4, hs4_4, hs4_15, _, hreg4⟩ := hrc
    have hs4_0 : s4.reg 0 = some ls.ds[0]!.w := by rw [hreg4 0 (by decide)]; exact hs3_0
    have hwf4 : vc.wf pr ls s4 := vc.wf_step_misc (s' := s4) hwf3 hmem4 hr4 hs4_4 hs4_15
    have hrd_mw4 : rd64 s4.mem (ea ctxBase 96) = some vc.memwReg.base := by
      rw [hmem4]; exact rd_ctx_memwBase hwf3
    have h4s : X86Vm.step vmImage s4 = .ok { s4.set 8 (some vc.memwReg.base) with rip := rip0 + 44 } := by
      rw [step_at_vc c_ld8 hrip4 hm_ld8, exec_ld hs4_15 hrd_mw4, hrip4]
    let s5 := { s4.set 8 (some vc.memwReg.base) with rip := rip0 + 44 }
    have hr5 : RegsOk s5 := regsOk_set hr4 (by decide) _
    have hs5_8 : s5.reg 8 = some vc.memwReg.base := reg_set_self_ok' hr4 (by decide) _
    have hs5_0 : s5.reg 0 = some ls.ds[0]!.w := by
      change (s4.set 8 _).reg 0 = some ls.ds[0]!.w; rw [reg_set_ne (by decide)]; exact hs4_0
    have hs5_4 : s5.reg 4 = some vc.sp := by
      change (s4.set 8 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs4_4
    have hs5_15 : s5.reg 15 = some ctxBase := by
      change (s4.set 8 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs4_15
    have h312_5 : rd64 s5.mem (ea ctxBase 312) = some ls.ds[1]!.w := by
      change rd64 s4.mem (ea ctxBase 312) = some ls.ds[1]!.w
      rw [hmem4]; exact h312
    have h5 : X86Vm.step vmImage s5 = .ok { s5.set 2 (some ls.ds[1]!.w) with rip := rip0 + 51 } := by
      rw [step_at_vc c_ld2 rfl hm_ld2, exec_ld hs5_15 h312_5]
    let s6 := { s5.set 2 (some ls.ds[1]!.w) with rip := rip0 + 51 }
    have hr6 : RegsOk s6 := regsOk_set hr5 (by decide) _
    have hs6_2 : s6.reg 2 = some ls.ds[1]!.w := reg_set_self_ok' hr5 (by decide) _
    have hs6_8 : s6.reg 8 = some vc.memwReg.base := by
      change (s5.set 2 _).reg 8 = some vc.memwReg.base; rw [reg_set_ne (by decide)]; exact hs5_8
    have hs6_0 : s6.reg 0 = some ls.ds[0]!.w := by
      change (s5.set 2 _).reg 0 = some ls.ds[0]!.w; rw [reg_set_ne (by decide)]; exact hs5_0
    have hs6_4 : s6.reg 4 = some vc.sp := by
      change (s5.set 2 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs5_4
    have hs6_15 : s6.reg 15 = some ctxBase := by
      change (s5.set 2 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs5_15
    have hmi := wf_mem_info h
    have hea : eaIdx vc.memwReg.base ls.ds[0]!.w 1 0 = vc.memwReg.base + ls.ds[0]!.w := by
      have hltW : vc.memwReg.base + 1 * ls.ds[0]!.w + 0 < X86Env.W := by
        have hle1 : ls.ds[0]!.w + 1 ≤ ls.msize := by cases mw <;> simp only [memWSize] at hbd ⊢ <;> omega
        have hmw_hi := hmi.1.2.2.2.2; have hmw_len := hmi.2.2.1.1; simp only [Region.hi] at hmw_hi; omega
      rw [eaIdx_add_eq hltW, Nat.one_mul, Nat.add_zero]
    have hwf6 : vc.wf pr ls s6 := vc.wf_step_misc (s' := s6) hwf4 rfl hr6 hs6_4 hs6_15
    have hhi_mw : vc.memwReg.base + ls.ds[0]!.w + memWSize mw ≤ vc.memwReg.hi := by
      have := hmi.2.2.1.1; simp only [Region.hi]; omega
    obtain ⟨_, m7, hwr7, _, hm7, _, hmw7, hoth7⟩ :=
      memWBytes_wrMem mw hwf6.1.2 (vc.mem_memw hwf6) hmi.1.1 (Nat.le_add_right _ _) hhi_mw (v := ls.ds[1]!.w)
    rw [Nat.add_sub_cancel_left] at hm7 hmw7
    rw [← hea] at hwr7
    have h6 : X86Vm.step vmImage s6 = .ok { s6 with mem := m7, rip := rip0 + 51 + stIdxLen mw } := by
      rw [step_at_vc c_stidx rfl hm_stidx, exec_stIdx hs6_8 hs6_0 hs6_2 hwr7]
    let s7 := { s6 with mem := m7, rip := rip0 + 51 + stIdxLen mw }
    have hcm7 : CodeMem m7 := hwf6.1.wrMem_preserve hwr7
    have hbl7 : BytesLt m7 :=
      bytesLt_wrMem hwf6.2.1 hwf6.1.2 (vc.mem_memw hwf6) hmi.1.1 (by rw [hea]; exact Nat.le_add_right _ _)
        (by rw [hea]; cases mw <;> exact hhi_mw) hwr7
    have hpos7 : ∀ u ∈ m7, 0 < u.data.length := len_pos_wrMem hwf6.2.2.1 hwr7
    have hne_mw_mp : vc.mempReg.base ≠ vc.memwReg.base :=
      vc.ne_bases h (i := 11) (j := 10) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
    have hne_mw_ctx : ctxBase ≠ vc.memwReg.base := by
      have hcb1 := (vc.ctx_bounds h).1
      rw [← hcb1]
      exact vc.ne_bases h (i := 2) (j := 10) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
    have hmp7 : vc.mempReg ∈ m7 := hoth7 vc.mempReg (vc.mem_memp hwf6) hne_mw_mp
    have hctx7 : vc.ctxReg ∈ m7 := hoth7 vc.ctxReg (vc.mem_ctx hwf6) (by rw [(vc.ctx_bounds h).1]; exact hne_mw_ctx)
    have hrd_mp : rd64 m7 (ea ctxBase 104) = some vc.mempReg.base := by
      have hs104 : leVal ((vc.ctxReg.data.drop 104).take 8) = vc.mempReg.base := by
        obtain ⟨_, _, _, _, _, _, _, _,
          _, _, _, _, _, _, _, _, _, _, _,
          _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
          _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
          _, _, _, _, _, _, _, _, _, _, _, _, _, hsl13, _⟩ := h
        exact hsl13
      have hcb := vc.ctx_bounds h
      rw [ea_add_eq (by have : ctxSize = 4096 := rfl; omega)]
      exact rd64_of_mem hcm7.2 hctx7 (by rw [hcb.1]; omega)
        (by simp only [Region.hi, hcb.1]; have : ctxSize = 4096 := rfl; omega)
        (by rw [hcb.1, Nat.add_sub_cancel_left]; exact hs104)
    have hrd_272_7 : rd64 m7 (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc) := by
      have hcb := vc.ctx_bounds h
      have hle272 := ctx_leVal_of_rd64 h (by decide) h272
      rw [ea_add_eq (by have : ctxSize = 4096 := rfl; omega)]
      exact rd64_of_mem hcm7.2 hctx7 (by rw [hcb.1]; omega)
        (by simp only [Region.hi, hcb.1]; have : ctxSize = 4096 := rfl; omega)
        (by rw [hcb.1, Nat.add_sub_cancel_left]; exact hle272)
    have hs7_15 : s7.reg 15 = some ctxBase := hs6_15
    have h7 : X86Vm.step vmImage s7 = .ok { s7.set 9 (some vc.mempReg.base) with rip := rip0 + 55 + stIdxLen mw } := by
      rw [step_at_vc c_ld9 rfl hm_ld9, exec_ld hs7_15 hrd_mp]
      have : rip0 + 51 + stIdxLen mw + 4 = rip0 + 55 + stIdxLen mw := by omega
      rw [this]
    let s8 := { s7.set 9 (some vc.mempReg.base) with rip := rip0 + 55 + stIdxLen mw }
    have hr8 : RegsOk s8 := regsOk_set hr6 (by decide) _
    have hs8_9 : s8.reg 9 = some vc.mempReg.base := reg_set_self_ok' hr6 (by decide) _
    have hs8_0 : s8.reg 0 = some ls.ds[0]!.w := by
      change (s7.set 9 _).reg 0 = some ls.ds[0]!.w; rw [reg_set_ne (by decide)]; exact hs6_0
    have hs8_15 : s8.reg 15 = some ctxBase := by
      change (s7.set 9 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs6_15
    have hs8_4 : s8.reg 4 = some vc.sp := by
      change (s7.set 9 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs6_4
    have htake2 : ls.ds.take 2 = [ls.ds[0]!, ls.ds[1]!] := by
      rcases hds : ls.ds with _ | ⟨x, _ | ⟨y, r⟩⟩ <;> simp [hds] at hle ⊢
    have hrd_272_8 : rd64 s8.mem (ea ctxBase 272) = some (SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!] ls.pc) := by
      change rd64 m7 (ea ctxBase 272) = _
      rw [← htake2]; exact hrd_272_7
    have h8 : X86Vm.step vmImage s8 =
        .ok { s8.set 10 (some (SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!] ls.pc)) with rip := rip0 + 62 + stIdxLen mw } := by
      rw [step_at_vc c_ldq rfl hm_ldq, exec_ld hs8_15 hrd_272_8]
      have : rip0 + 55 + stIdxLen mw + 7 = rip0 + 62 + stIdxLen mw := by omega
      rw [this]
    let s9 := { s8.set 10 (some (SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!] ls.pc)) with rip := rip0 + 62 + stIdxLen mw }
    have hr9 : RegsOk s9 := regsOk_set hr8 (by decide) _
    have hs9_10 : s9.reg 10 = some (SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!] ls.pc) :=
      reg_set_self_ok' hr8 (by decide) _
    have hs9_9 : s9.reg 9 = some vc.mempReg.base := by
      change (s8.set 10 _).reg 9 = some vc.mempReg.base; rw [reg_set_ne (by decide)]; exact hs8_9
    have hs9_0 : s9.reg 0 = some ls.ds[0]!.w := by
      change (s8.set 10 _).reg 0 = some ls.ds[0]!.w; rw [reg_set_ne (by decide)]; exact hs8_0
    have hs9_15 : s9.reg 15 = some ctxBase := by
      change (s8.set 10 _).reg 15 = some ctxBase; rw [reg_set_ne (by decide)]; exact hs8_15
    have hs9_4 : s9.reg 4 = some vc.sp := by
      change (s8.set 10 _).reg 4 = some vc.sp; rw [reg_set_ne (by decide)]; exact hs8_4
    have hrch9 : Reaches vmImage s s9 :=
      Reaches.trans hrch3 (Reaches.trans hrch4 (Reaches.step_trans h4s (Reaches.step_trans h5
        (Reaches.step_trans h6 (Reaches.step_trans h7 (Reaches.step h8))))))
    have hk9 : s9.k = s.k := by
      have : s9.k = s4.k := rfl
      rw [this, hk4]; rfl
    have hoth9 : ∀ u ∈ s.mem, u.base ≠ vc.memwReg.base → u ∈ s9.mem := by
      intro u hu hne
      have hu6 : u ∈ s6.mem := by change u ∈ s4.mem; rw [hmem4]; exact hu
      exact hoth7 u hu6 hne
    exact hcont s9 hbd hrch9 hk9 rfl hcm7 hbl7 hpos7 hr9 hmw7 hmp7 hctx7 hoth9
      hs9_4 hs9_15 hs9_0 hs9_9 hs9_10
  · rw [if_neg hbd] at hrc
    have hrch_fail := Reaches.trans hrch3 hrc.choose_spec.1
    rcases hds_eq : ls.ds with _ | ⟨a, _ | ⟨v, r⟩⟩ <;> try (simp [hds_eq] at hle)
    have hno : ¬(a.w + memWSize mw ≤ ls.msize) := by simpa [hds_eq] using hbd
    dsimp only [PrimPostOutcome]
    have hpe := hp_eff.2.2
    rw [hp_eff.1]
    simp only [hds_eq, List.take_succ_cons, List.take_zero, List.getElem!_cons_zero,
      List.getElem!_cons_succ, SporosProvLen.LSt.st, hno, ↓reduceIte] at hpe ⊢
    rw [hpe]
    exact ⟨hrc.choose, hrch_fail, hrc.choose_spec.2.1, hrc.choose_spec.2.2⟩

theorem st_memp_step_inv {vc : VmCtx} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {mw_cur mp_cur : Region} {a i q rip0 chk : Nat}
    (hm_st : (rip0, (.stIdx .w64 9 0 8 (8 * i) 10), if i = 0 then 4 else 5) ∈ vmTableChunk chk)
    (hrip : s.rip = rip0)
    (hcm : CodeMem s.mem) (hbl : BytesLt s.mem) (hpos : ∀ u ∈ s.mem, 0 < u.data.length)
    (hmp_mem : mp_cur ∈ s.mem) (hmw_mem : mw_cur ∈ s.mem) (hctx_mem : vc.ctxReg ∈ s.mem)
    (hmp_base : mp_cur.base = vc.mempReg.base) (hmp_rw : mp_cur.rw = true)
    (hmp_len : mp_cur.data.length = vc.mempReg.data.length)
    (hmw_base : mw_cur.base = vc.memwReg.base) (hctx_base : vc.ctxReg.base = ctxBase)
    (hhi_memp : vc.mempReg.hi < X86Env.W)
    (hne_mp_mw : vc.mempReg.base ≠ vc.memwReg.base)
    (hne_mp_ctx : vc.mempReg.base ≠ ctxBase)
    (hlt : a + i < ls.msize) (hlen_mp : ls.msize * 8 ≤ vc.mempReg.data.length)
    (hr9 : s.reg 9 = some vc.mempReg.base) (hr0 : s.reg 0 = some a) (hr10 : s.reg 10 = some q)
    (hinv : ∀ x, x < ls.msize →
      leVal ((mp_cur.data.drop (x * 8)).take 8) =
        if a ≤ x ∧ x < a + i then q % X86Env.W else (ls.mem x).plen % X86Env.W) :
    let mp' := { mp_cur with data := writeBytes mp_cur.data ((a + i) * 8) (bytes8 q) }
    ∃ s',
      X86Vm.step vmImage s = .ok s' ∧
      s'.rip = rip0 + (if i = 0 then 4 else 5) ∧ s'.regs = s.regs ∧ s'.fl = s.fl ∧ s'.k = s.k ∧
      CodeMem s'.mem ∧ BytesLt s'.mem ∧ (∀ u ∈ s'.mem, 0 < u.data.length) ∧
      mp' ∈ s'.mem ∧ mw_cur ∈ s'.mem ∧ vc.ctxReg ∈ s'.mem ∧
      mp'.base = vc.mempReg.base ∧ mp'.rw = true ∧ mp'.data.length = vc.mempReg.data.length ∧
      (∀ u ∈ s.mem, u.base ≠ vc.mempReg.base → u ∈ s'.mem) ∧
      (∀ x, x < ls.msize →
        leVal ((mp'.data.drop (x * 8)).take 8) =
          if a ≤ x ∧ x < a + (i + 1) then q % X86Env.W else (ls.mem x).plen % X86Env.W) := by
  obtain ⟨s', hstep, hrip', hregs', hfl', hk', hcm', hbl', hpos', hmp', hmw', hctx', hoth'⟩ :=
    st_memp_qword_step (vc := vc) (ls := ls) hm_st hrip hcm hbl hpos hmp_mem hmw_mem hctx_mem
      hmp_base hmp_rw hmp_len hmw_base hctx_base hhi_memp hne_mp_mw hne_mp_ctx hlt hlen_mp hr9 hr0 hr10
  have hlen_cur : ls.msize * 8 ≤ mp_cur.data.length := by rw [hmp_len]; exact hlen_mp
  have hqstep := writeBytes_qword_memp (mpData := mp_cur.data) (msize := ls.msize) (idx := a + i) (q := q)
    hlt hlen_cur
  refine ⟨s', hstep, hrip', hregs', hfl', hk', hcm', hbl', hpos', hmp', hmw', hctx',
    hmp_base, hmp_rw, by simp [writeBytes_length, hmp_len], hoth', ?_⟩
  intro x hx
  by_cases hxi : x = a + i
  · subst hxi
    have hcond : a ≤ a + i ∧ a + i < a + (i + 1) := by omega
    rw [if_pos hcond]
    exact hqstep.1
  · rw [hqstep.2 x hx hxi, hinv x hx]
    have hiff : (a ≤ x ∧ x < a + i) ↔ (a ≤ x ∧ x < a + (i + 1)) := by omega
    by_cases h1 : a ≤ x ∧ x < a + i
    · rw [if_pos h1, if_pos (hiff.mp h1)]
    · rw [if_neg h1, if_neg (fun h2 => h1 (hiff.mpr h2))]

theorem st_finish_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s s_end : X86Vm.St} {mw' mp' : Region} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (mw : MemW) (p : Sporos.Prim) {c_jmp : Nat}
    (hp_eff : p.eff.1 = 2 ∧ p.eff.2 = 0 ∧
      SporosProvLen.lexecPrim p [ls.ds[0]!, ls.ds[1]!] ls =
        match ls.st ls.ds[0]!.w (memWSize mw) ls.ds[1]!.w
            (SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!] ls.pc) with
        | some ls' => .ok [] ls'
        | none => .fail .bounds)
    (hm_jmp : (s_end.rip, (.ofInt (.jmp SporosVmBin.L_prim_epilogue)), 5) ∈ vmTableChunk c_jmp)
    (hbd : ls.ds[0]!.w + memWSize mw ≤ ls.msize)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (hle : 2 ≤ ls.ds.length)
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 0)
    (hrch : Reaches vmImage s s_end) (hk : s_end.k = s.k)
    (hcm' : CodeMem s_end.mem) (hbl' : BytesLt s_end.mem) (hpos' : ∀ u ∈ s_end.mem, 0 < u.data.length)
    (hr' : RegsOk s_end) (hsp' : s_end.reg 4 = some vc.sp) (h15' : s_end.reg 15 = some ctxBase)
    (hmw_eq : mw' = { vc.memwReg with data := writeBytes vc.memwReg.data ls.ds[0]!.w (memWBytes mw ls.ds[1]!.w) })
    (hmp_base : mp'.base = vc.mempReg.base) (hmp_rw : mp'.rw = true)
    (hmp_len : mp'.data.length = vc.mempReg.data.length)
    (hmw_mem : mw' ∈ s_end.mem) (hmp_mem : mp' ∈ s_end.mem) (hctx_mem : vc.ctxReg ∈ s_end.mem)
    (hoth : ∀ u ∈ s.mem, u.base ≠ vc.memwReg.base → u.base ≠ vc.mempReg.base → u ∈ s_end.mem)
    (hinv : ∀ x, x < ls.msize →
      leVal ((mp'.data.drop (x * 8)).take 8) =
        if ls.ds[0]!.w ≤ x ∧ x < ls.ds[0]!.w + memWSize mw then
          (SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!] ls.pc) % X86Env.W
        else (ls.mem x).plen % X86Env.W) :
    PrimPostOutcome pr ls f rest s p := by
  let a := ls.ds[0]!.w
  let v := ls.ds[1]!.w
  let q := SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!] ls.pc
  let mem' : SporosProvLen.LMem := fun x =>
    if a ≤ x ∧ x < a + memWSize mw then ⟨v / 256 ^ (x - a) % 256, q⟩ else ls.mem x
  let ls' : SporosProvLen.LSt := { ls with mem := mem' }
  let vc' : VmCtx := { vc with memwReg := mw', mempReg := mp' }
  have hmi := wf_mem_info h
  have hmw_base : mw'.base = vc.memwReg.base := by rw [hmw_eq]
  have hmw_rw : mw'.rw = true := by rw [hmw_eq]; exact hmi.1.1
  have hmw_len : mw'.data.length = vc.memwReg.data.length := by rw [hmw_eq]; simp [writeBytes_length]
  have hmemrel' : MemRel mw'.data mp'.data ls.msize mem' := by
    refine ⟨by rw [hmw_len]; exact hmi.2.2.1.1, by rw [hmp_len]; exact hmi.2.2.1.2.1, ?_⟩
    intro x hx
    have hbytes := memRel_wr_memw_bytes mw (mwData := vc.memwReg.data) (msize := ls.msize) (a := a) (v := v)
      hbd hmi.2.2.1.1 x hx
    rw [hmw_eq]
    dsimp only
    constructor
    · rw [hbytes]
      by_cases hc : a ≤ x ∧ x < a + memWSize mw
      · dsimp only [mem']; rw [if_pos hc, if_pos hc, Nat.mod_mod]
      · dsimp only [mem']; rw [if_neg hc, if_neg hc]
        have hprev := (hmi.2.2.1.2.2 x hx).1
        rw [hprev]; rfl
    · rw [hinv x hx]
      by_cases hc : a ≤ x ∧ x < a + memWSize mw
      · dsimp only [mem']; rw [if_pos hc, if_pos hc]
      · dsimp only [mem']; rw [if_neg hc, if_neg hc]
  have hqW : q < Sporos.M := by dsimp [q, SporosProvLen.lq]; exact Nat.mod_lt _ (by decide)
  have hwlt_mem' : ∀ x, (mem' x).w < Sporos.M ∧ (mem' x).plen < Sporos.M := by
    intro x
    by_cases hc : a ≤ x ∧ x < a + memWSize mw
    · dsimp only [mem']; rw [if_pos hc]
      exact ⟨Nat.lt_trans (Nat.mod_lt _ (by decide)) (by decide), hqW⟩
    · dsimp only [mem']; rw [if_neg hc]
      have hwl := (vc.code_flat h).2.2.2.2
      exact hwl.2.2.2.1 x
  have hwf' : vc'.wf pr ls' s_end :=
    wf_write_mem h hcm' hbl' hpos' hr' hsp' h15' hmw_base hmw_rw hmw_len hmp_base hmp_rw hmp_len
      hmw_mem hmp_mem hoth hmemrel' hwlt_mem'
  have h280' : rd64 s_end.mem (ea ctxBase 280) = some 2 := by
    have hcb := vc.ctx_bounds h
    have hle280 := ctx_leVal_of_rd64 h (by decide) h280
    rw [ea_add_eq (by have : ctxSize = 4096 := rfl; omega)]
    exact rd64_of_mem hcm'.2 hctx_mem (by rw [hcb.1]; omega)
      (by simp only [Region.hi, hcb.1]; have : ctxSize = 4096 := rfl; omega)
      (by rw [hcb.1, Nat.add_sub_cancel_left]; exact hle280)
  have h288' : rd64 s_end.mem (ea ctxBase 288) = some 0 := by
    have hcb := vc.ctx_bounds h
    have hle288 := ctx_leVal_of_rd64 h (by decide) h288
    rw [ea_add_eq (by have : ctxSize = 4096 := rfl; omega)]
    exact rd64_of_mem hcm'.2 hctx_mem (by rw [hcb.1]; omega)
      (by simp only [Region.hi, hcb.1]; have : ctxSize = 4096 := rfl; omega)
      (by rw [hcb.1, Nat.add_sub_cancel_left]; exact hle288)
  have hfin := jmp_epilogue0_finish_spec (vc := vc') (ls := ls') (outs := [])
    (step_at_vc c_jmp rfl hm_jmp)
    hwf' (oracleAdmits_of_reaches hor hrch) hrs hpcW hle (by simp) (by simp) h280' h288'
    (fun i hi => absurd hi (Nat.not_lt_zero _))
  have hres := lift_next_outcome hrch hk hfin
  rcases hds_eq : ls.ds with _ | ⟨da, _ | ⟨dv, dr⟩⟩ <;> try (simp [hds_eq] at hle)
  have hpe := hp_eff.2.2
  dsimp only [PrimPostOutcome]
  rw [hp_eff.1]
  simp only [hds_eq, List.take_succ_cons, List.take_zero, List.getElem!_cons_zero,
    List.getElem!_cons_succ] at hpe ⊢
  rw [hpe]
  have hbd' : da.w + memWSize mw ≤ ls.msize := by simpa [hds_eq] using hbd
  simp only [SporosProvLen.LSt.st, hbd', ↓reduceIte]
  simpa [ls', mem', a, v, q, hds_eq] using hres

theorem P_st8_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_st8) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 0)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .st8 := by
  have hp_eff : Sporos.Prim.st8.eff.1 = 2 ∧ Sporos.Prim.st8.eff.2 = 0 ∧
      SporosProvLen.lexecPrim .st8 [ls.ds[0]!, ls.ds[1]!] ls =
        match ls.st ls.ds[0]!.w (memWSize .w8) ls.ds[1]!.w
            (SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!] ls.pc) with
        | some ls' => .ok [] ls'
        | none => .fail .bounds := ⟨rfl, rfl, rfl⟩
  refine st_prologue_spec .w8 .st8 4197017
    (c_ld0 := 8) (c_mov1 := 8) (c_ld10 := 8) (c0 := 8) (c1 := 8) (c2 := 8) (c3 := 8) (c4 := 8) (c5 := 8)
    (c_ld8 := 8) (c_ld2 := 8) (c_stidx := 8) (c_ld9 := 8) (c_ldq := 8)
    hp_eff (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    hrip hle h hor hrs hpcW h272 h280 h288 h296 h312 ?_
  intro s0 hbd hrch0 hk0 hrip0 hcm0 hbl0 hpos0 hr0 hmw0 hmp0 hctx0 hoth0 hs0_4 hs0_15 hs0_0 hs0_9 hs0_10
  have hmi := wf_mem_info h
  have hne_mp_mw : vc.mempReg.base ≠ vc.memwReg.base :=
    vc.ne_bases h (i := 11) (j := 10) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hne_mp_ctx : vc.mempReg.base ≠ ctxBase := by
    have hcb := (vc.ctx_bounds h).1; rw [← hcb]
    exact vc.ne_bases h (i := 11) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hinv0 : ∀ x, x < ls.msize →
      leVal ((vc.mempReg.data.drop (x * 8)).take 8) =
        if ls.ds[0]!.w ≤ x ∧ x < ls.ds[0]!.w + 0 then
          (SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!] ls.pc) % X86Env.W
        else (ls.mem x).plen % X86Env.W := by
    intro x hx; rw [if_neg (by omega)]; exact (hmi.2.2.1.2.2 x hx).2
  obtain ⟨s1, hs1, hrip1, hregs1, _, hk1, hcm1, hbl1, hpos1, hmp1, hmw1, hctx1,
      hmp1_b, hmp1_rw, hmp1_l, hoth1, hinv1⟩ :=
    st_memp_step_inv (vc := vc) (ls := ls) (chk := 8) (i := 0) (by decide) hrip0
      hcm0 hbl0 hpos0 hmp0 hmw0 hctx0 rfl hmi.2.1.1 rfl rfl (vc.ctx_bounds h).1
      hmi.2.1.2.2.2 hne_mp_mw hne_mp_ctx (by simp only [memWSize] at hbd; omega) hmi.2.2.1.2.1
      hs0_9 hs0_0 hs0_10 hinv0
  exact st_finish_spec (vc := vc) .w8 .st8 (c_jmp := 8) hp_eff (by rw [hrip1]; decide) hbd h hor hrs hpcW hle
    h280 h288 (Reaches.trans hrch0 (Reaches.step hs1)) (by rw [hk1, hk0]) hcm1 hbl1 hpos1
    (by simpa [RegsOk, X86Int.St.reg, hregs1] using hr0) (by simpa [X86Int.St.reg, hregs1] using hs0_4)
    (by simpa [X86Int.St.reg, hregs1] using hs0_15) rfl hmp1_b hmp1_rw hmp1_l hmw1 hmp1 hctx1
    (fun u hu h1 h2 => hoth1 u (hoth0 u hu h1) h2) hinv1

theorem P_st16_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_st16) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 0)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .st16 := by
  have hp_eff : Sporos.Prim.st16.eff.1 = 2 ∧ Sporos.Prim.st16.eff.2 = 0 ∧
      SporosProvLen.lexecPrim .st16 [ls.ds[0]!, ls.ds[1]!] ls =
        match ls.st ls.ds[0]!.w (memWSize .w16) ls.ds[1]!.w
            (SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!] ls.pc) with
        | some ls' => .ok [] ls'
        | none => .fail .bounds := ⟨rfl, rfl, rfl⟩
  refine st_prologue_spec .w16 .st16 4197092
    (c_ld0 := 8) (c_mov1 := 8) (c_ld10 := 8) (c0 := 8) (c1 := 8) (c2 := 8) (c3 := 8) (c4 := 8) (c5 := 8)
    (c_ld8 := 8) (c_ld2 := 8) (c_stidx := 8) (c_ld9 := 8) (c_ldq := 8)
    hp_eff (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    hrip hle h hor hrs hpcW h272 h280 h288 h296 h312 ?_
  intro s0 hbd hrch0 hk0 hrip0 hcm0 hbl0 hpos0 hr0 hmw0 hmp0 hctx0 hoth0 hs0_4 hs0_15 hs0_0 hs0_9 hs0_10
  have hmi := wf_mem_info h
  have hne_mp_mw : vc.mempReg.base ≠ vc.memwReg.base :=
    vc.ne_bases h (i := 11) (j := 10) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hne_mp_ctx : vc.mempReg.base ≠ ctxBase := by
    have hcb := (vc.ctx_bounds h).1; rw [← hcb]
    exact vc.ne_bases h (i := 11) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hinv0 : ∀ x, x < ls.msize →
      leVal ((vc.mempReg.data.drop (x * 8)).take 8) =
        if ls.ds[0]!.w ≤ x ∧ x < ls.ds[0]!.w + 0 then
          (SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!] ls.pc) % X86Env.W
        else (ls.mem x).plen % X86Env.W := by
    intro x hx; rw [if_neg (by omega)]; exact (hmi.2.2.1.2.2 x hx).2
  obtain ⟨s1, hs1, hrip1, hregs1, _, hk1, hcm1, hbl1, hpos1, hmp1, hmw1, hctx1,
      hmp1_b, hmp1_rw, hmp1_l, hoth1, hinv1⟩ :=
    st_memp_step_inv (vc := vc) (ls := ls) (chk := 8) (i := 0) (by decide) hrip0
      hcm0 hbl0 hpos0 hmp0 hmw0 hctx0 rfl hmi.2.1.1 rfl rfl (vc.ctx_bounds h).1
      hmi.2.1.2.2.2 hne_mp_mw hne_mp_ctx (by simp only [memWSize] at hbd; omega) hmi.2.2.1.2.1
      hs0_9 hs0_0 hs0_10 hinv0
  obtain ⟨s2, hs2, hrip2, hregs2, _, hk2, hcm2, hbl2, hpos2, hmp2, hmw2, hctx2,
      hmp2_b, hmp2_rw, hmp2_l, hoth2, hinv2⟩ :=
    st_memp_step_inv (vc := vc) (ls := ls) (chk := 8) (i := 1) (by decide) hrip1
      hcm1 hbl1 hpos1 hmp1 hmw1 hctx1 hmp1_b hmp1_rw hmp1_l rfl (vc.ctx_bounds h).1
      hmi.2.1.2.2.2 hne_mp_mw hne_mp_ctx (by simp only [memWSize] at hbd; omega) hmi.2.2.1.2.1
      (by simpa [X86Int.St.reg, hregs1] using hs0_9) (by simpa [X86Int.St.reg, hregs1] using hs0_0)
      (by simpa [X86Int.St.reg, hregs1] using hs0_10) hinv1
  exact st_finish_spec (vc := vc) .w16 .st16 (c_jmp := 8) hp_eff (by rw [hrip2]; decide) hbd h hor hrs hpcW hle
    h280 h288 (Reaches.trans hrch0 (Reaches.step_trans hs1 (Reaches.step hs2))) (by rw [hk2, hk1, hk0])
    hcm2 hbl2 hpos2 (by simpa [RegsOk, X86Int.St.reg, hregs2, hregs1] using hr0)
    (by simpa [X86Int.St.reg, hregs2, hregs1] using hs0_4)
    (by simpa [X86Int.St.reg, hregs2, hregs1] using hs0_15) rfl hmp2_b hmp2_rw hmp2_l hmw2 hmp2 hctx2
    (fun u hu h1 h2 => hoth2 u (hoth1 u (hoth0 u hu h1) h2) h2) hinv2

theorem P_st32_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_st32) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 0)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .st32 := by
  have hp_eff : Sporos.Prim.st32.eff.1 = 2 ∧ Sporos.Prim.st32.eff.2 = 0 ∧
      SporosProvLen.lexecPrim .st32 [ls.ds[0]!, ls.ds[1]!] ls =
        match ls.st ls.ds[0]!.w (memWSize .w32) ls.ds[1]!.w
            (SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!] ls.pc) with
        | some ls' => .ok [] ls'
        | none => .fail .bounds := ⟨rfl, rfl, rfl⟩
  refine st_prologue_spec .w32 .st32 4197173
    (c_ld0 := 8) (c_mov1 := 8) (c_ld10 := 8) (c0 := 8) (c1 := 8) (c2 := 8) (c3 := 9) (c4 := 9) (c5 := 9)
    (c_ld8 := 9) (c_ld2 := 9) (c_stidx := 9) (c_ld9 := 9) (c_ldq := 9)
    hp_eff (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    hrip hle h hor hrs hpcW h272 h280 h288 h296 h312 ?_
  intro s0 hbd hrch0 hk0 hrip0 hcm0 hbl0 hpos0 hr0 hmw0 hmp0 hctx0 hoth0 hs0_4 hs0_15 hs0_0 hs0_9 hs0_10
  have hmi := wf_mem_info h
  have hne_mp_mw : vc.mempReg.base ≠ vc.memwReg.base :=
    vc.ne_bases h (i := 11) (j := 10) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hne_mp_ctx : vc.mempReg.base ≠ ctxBase := by
    have hcb := (vc.ctx_bounds h).1; rw [← hcb]
    exact vc.ne_bases h (i := 11) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hinv0 : ∀ x, x < ls.msize →
      leVal ((vc.mempReg.data.drop (x * 8)).take 8) =
        if ls.ds[0]!.w ≤ x ∧ x < ls.ds[0]!.w + 0 then
          (SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!] ls.pc) % X86Env.W
        else (ls.mem x).plen % X86Env.W := by
    intro x hx; rw [if_neg (by omega)]; exact (hmi.2.2.1.2.2 x hx).2
  obtain ⟨s1, hs1, hrip1, hregs1, _, hk1, hcm1, hbl1, hpos1, hmp1, hmw1, hctx1,
      hmp1_b, hmp1_rw, hmp1_l, hoth1, hinv1⟩ :=
    st_memp_step_inv (vc := vc) (ls := ls) (chk := 9) (i := 0) (by decide) hrip0
      hcm0 hbl0 hpos0 hmp0 hmw0 hctx0 rfl hmi.2.1.1 rfl rfl (vc.ctx_bounds h).1
      hmi.2.1.2.2.2 hne_mp_mw hne_mp_ctx (by simp only [memWSize] at hbd; omega) hmi.2.2.1.2.1
      hs0_9 hs0_0 hs0_10 hinv0
  obtain ⟨s2, hs2, hrip2, hregs2, _, hk2, hcm2, hbl2, hpos2, hmp2, hmw2, hctx2,
      hmp2_b, hmp2_rw, hmp2_l, hoth2, hinv2⟩ :=
    st_memp_step_inv (vc := vc) (ls := ls) (chk := 9) (i := 1) (by decide) hrip1
      hcm1 hbl1 hpos1 hmp1 hmw1 hctx1 hmp1_b hmp1_rw hmp1_l rfl (vc.ctx_bounds h).1
      hmi.2.1.2.2.2 hne_mp_mw hne_mp_ctx (by simp only [memWSize] at hbd; omega) hmi.2.2.1.2.1
      (by simpa [X86Int.St.reg, hregs1] using hs0_9) (by simpa [X86Int.St.reg, hregs1] using hs0_0)
      (by simpa [X86Int.St.reg, hregs1] using hs0_10) hinv1
  obtain ⟨s3, hs3, hrip3, hregs3, _, hk3, hcm3, hbl3, hpos3, hmp3, hmw3, hctx3,
      hmp3_b, hmp3_rw, hmp3_l, hoth3, hinv3⟩ :=
    st_memp_step_inv (vc := vc) (ls := ls) (chk := 9) (i := 2) (by decide) hrip2
      hcm2 hbl2 hpos2 hmp2 hmw2 hctx2 hmp2_b hmp2_rw hmp2_l rfl (vc.ctx_bounds h).1
      hmi.2.1.2.2.2 hne_mp_mw hne_mp_ctx (by simp only [memWSize] at hbd; omega) hmi.2.2.1.2.1
      (by simpa [X86Int.St.reg, hregs2, hregs1] using hs0_9)
      (by simpa [X86Int.St.reg, hregs2, hregs1] using hs0_0)
      (by simpa [X86Int.St.reg, hregs2, hregs1] using hs0_10) hinv2
  obtain ⟨s4, hs4, hrip4, hregs4, _, hk4, hcm4, hbl4, hpos4, hmp4, hmw4, hctx4,
      hmp4_b, hmp4_rw, hmp4_l, hoth4, hinv4⟩ :=
    st_memp_step_inv (vc := vc) (ls := ls) (chk := 9) (i := 3) (by decide) hrip3
      hcm3 hbl3 hpos3 hmp3 hmw3 hctx3 hmp3_b hmp3_rw hmp3_l rfl (vc.ctx_bounds h).1
      hmi.2.1.2.2.2 hne_mp_mw hne_mp_ctx (by simp only [memWSize] at hbd; omega) hmi.2.2.1.2.1
      (by simpa [X86Int.St.reg, hregs3, hregs2, hregs1] using hs0_9)
      (by simpa [X86Int.St.reg, hregs3, hregs2, hregs1] using hs0_0)
      (by simpa [X86Int.St.reg, hregs3, hregs2, hregs1] using hs0_10) hinv3
  exact st_finish_spec (vc := vc) .w32 .st32 (c_jmp := 9) hp_eff (by rw [hrip4]; decide) hbd h hor hrs hpcW hle
    h280 h288
    (Reaches.trans hrch0 (Reaches.step_trans hs1 (Reaches.step_trans hs2 (Reaches.step_trans hs3 (Reaches.step hs4)))))
    (by rw [hk4, hk3, hk2, hk1, hk0]) hcm4 hbl4 hpos4
    (by simpa [RegsOk, X86Int.St.reg, hregs4, hregs3, hregs2, hregs1] using hr0)
    (by simpa [X86Int.St.reg, hregs4, hregs3, hregs2, hregs1] using hs0_4)
    (by simpa [X86Int.St.reg, hregs4, hregs3, hregs2, hregs1] using hs0_15) rfl hmp4_b hmp4_rw hmp4_l
    hmw4 hmp4 hctx4 (fun u hu h1 h2 => hoth4 u (hoth3 u (hoth2 u (hoth1 u (hoth0 u hu h1) h2) h2) h2) h2) hinv4

theorem P_st64_spec {vc : VmCtx} {pr : Sporos.Prog} {ls : SporosProvLen.LSt}
    {s : X86Vm.St} {f : SporosProvLen.LFrame} {rest : List SporosProvLen.LFrame}
    (hrip : s.rip = SporosVmBin.P_st64) (hle : 2 ≤ ls.ds.length)
    (h : vc.wf pr ls s) (hor : OracleAdmits s) (hrs : ls.rs = f :: rest) (hpcW : f.pc + 1 < X86Env.W)
    (h272 : rd64 s.mem (ea ctxBase 272) = some (SporosProvLen.lq (ls.ds.take 2) ls.pc))
    (h280 : rd64 s.mem (ea ctxBase 280) = some 2) (h288 : rd64 s.mem (ea ctxBase 288) = some 0)
    (h296 : rd64 s.mem (ea ctxBase 296) = some ls.ds[0]!.w)
    (h312 : rd64 s.mem (ea ctxBase 312) = some ls.ds[1]!.w) :
    PrimPostOutcome pr ls f rest s .st64 := by
  have hp_eff : Sporos.Prim.st64.eff.1 = 2 ∧ Sporos.Prim.st64.eff.2 = 0 ∧
      SporosProvLen.lexecPrim .st64 [ls.ds[0]!, ls.ds[1]!] ls =
        match ls.st ls.ds[0]!.w (memWSize .w64) ls.ds[1]!.w
            (SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!] ls.pc) with
        | some ls' => .ok [] ls'
        | none => .fail .bounds := ⟨rfl, rfl, rfl⟩
  refine st_prologue_spec .w64 .st64 4197263
    (c_ld0 := 9) (c_mov1 := 9) (c_ld10 := 9) (c0 := 9) (c1 := 9) (c2 := 9) (c3 := 9) (c4 := 9) (c5 := 9)
    (c_ld8 := 9) (c_ld2 := 9) (c_stidx := 9) (c_ld9 := 9) (c_ldq := 9)
    hp_eff (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    hrip hle h hor hrs hpcW h272 h280 h288 h296 h312 ?_
  intro s0 hbd hrch0 hk0 hrip0 hcm0 hbl0 hpos0 hr0 hmw0 hmp0 hctx0 hoth0 hs0_4 hs0_15 hs0_0 hs0_9 hs0_10
  have hmi := wf_mem_info h
  have hne_mp_mw : vc.mempReg.base ≠ vc.memwReg.base :=
    vc.ne_bases h (i := 11) (j := 10) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hne_mp_ctx : vc.mempReg.base ≠ ctxBase := by
    have hcb := (vc.ctx_bounds h).1; rw [← hcb]
    exact vc.ne_bases h (i := 11) (j := 2) (by simp [VmCtx.regions]) (by simp [VmCtx.regions]) (by decide)
  have hinv0 : ∀ x, x < ls.msize →
      leVal ((vc.mempReg.data.drop (x * 8)).take 8) =
        if ls.ds[0]!.w ≤ x ∧ x < ls.ds[0]!.w + 0 then
          (SporosProvLen.lq [ls.ds[0]!, ls.ds[1]!] ls.pc) % X86Env.W
        else (ls.mem x).plen % X86Env.W := by
    intro x hx; rw [if_neg (by omega)]; exact (hmi.2.2.1.2.2 x hx).2
  obtain ⟨s1, hs1, hrip1, hregs1, _, hk1, hcm1, hbl1, hpos1, hmp1, hmw1, hctx1,
      hmp1_b, hmp1_rw, hmp1_l, hoth1, hinv1⟩ :=
    st_memp_step_inv (vc := vc) (ls := ls) (chk := 9) (i := 0) (by decide) hrip0
      hcm0 hbl0 hpos0 hmp0 hmw0 hctx0 rfl hmi.2.1.1 rfl rfl (vc.ctx_bounds h).1
      hmi.2.1.2.2.2 hne_mp_mw hne_mp_ctx (by simp only [memWSize] at hbd; omega) hmi.2.2.1.2.1
      hs0_9 hs0_0 hs0_10 hinv0
  obtain ⟨s2, hs2, hrip2, hregs2, _, hk2, hcm2, hbl2, hpos2, hmp2, hmw2, hctx2,
      hmp2_b, hmp2_rw, hmp2_l, hoth2, hinv2⟩ :=
    st_memp_step_inv (vc := vc) (ls := ls) (chk := 9) (i := 1) (by decide) hrip1
      hcm1 hbl1 hpos1 hmp1 hmw1 hctx1 hmp1_b hmp1_rw hmp1_l rfl (vc.ctx_bounds h).1
      hmi.2.1.2.2.2 hne_mp_mw hne_mp_ctx (by simp only [memWSize] at hbd; omega) hmi.2.2.1.2.1
      (by simpa [X86Int.St.reg, hregs1] using hs0_9) (by simpa [X86Int.St.reg, hregs1] using hs0_0)
      (by simpa [X86Int.St.reg, hregs1] using hs0_10) hinv1
  obtain ⟨s3, hs3, hrip3, hregs3, _, hk3, hcm3, hbl3, hpos3, hmp3, hmw3, hctx3,
      hmp3_b, hmp3_rw, hmp3_l, hoth3, hinv3⟩ :=
    st_memp_step_inv (vc := vc) (ls := ls) (chk := 9) (i := 2) (by decide) hrip2
      hcm2 hbl2 hpos2 hmp2 hmw2 hctx2 hmp2_b hmp2_rw hmp2_l rfl (vc.ctx_bounds h).1
      hmi.2.1.2.2.2 hne_mp_mw hne_mp_ctx (by simp only [memWSize] at hbd; omega) hmi.2.2.1.2.1
      (by simpa [X86Int.St.reg, hregs2, hregs1] using hs0_9)
      (by simpa [X86Int.St.reg, hregs2, hregs1] using hs0_0)
      (by simpa [X86Int.St.reg, hregs2, hregs1] using hs0_10) hinv2
  obtain ⟨s4, hs4, hrip4, hregs4, _, hk4, hcm4, hbl4, hpos4, hmp4, hmw4, hctx4,
      hmp4_b, hmp4_rw, hmp4_l, hoth4, hinv4⟩ :=
    st_memp_step_inv (vc := vc) (ls := ls) (chk := 9) (i := 3) (by decide) hrip3
      hcm3 hbl3 hpos3 hmp3 hmw3 hctx3 hmp3_b hmp3_rw hmp3_l rfl (vc.ctx_bounds h).1
      hmi.2.1.2.2.2 hne_mp_mw hne_mp_ctx (by simp only [memWSize] at hbd; omega) hmi.2.2.1.2.1
      (by simpa [X86Int.St.reg, hregs3, hregs2, hregs1] using hs0_9)
      (by simpa [X86Int.St.reg, hregs3, hregs2, hregs1] using hs0_0)
      (by simpa [X86Int.St.reg, hregs3, hregs2, hregs1] using hs0_10) hinv3
  obtain ⟨s5, hs5, hrip5, hregs5, _, hk5, hcm5, hbl5, hpos5, hmp5, hmw5, hctx5,
      hmp5_b, hmp5_rw, hmp5_l, hoth5, hinv5⟩ :=
    st_memp_step_inv (vc := vc) (ls := ls) (chk := 9) (i := 4) (by decide) hrip4
      hcm4 hbl4 hpos4 hmp4 hmw4 hctx4 hmp4_b hmp4_rw hmp4_l rfl (vc.ctx_bounds h).1
      hmi.2.1.2.2.2 hne_mp_mw hne_mp_ctx (by simp only [memWSize] at hbd; omega) hmi.2.2.1.2.1
      (by simpa [X86Int.St.reg, hregs4, hregs3, hregs2, hregs1] using hs0_9)
      (by simpa [X86Int.St.reg, hregs4, hregs3, hregs2, hregs1] using hs0_0)
      (by simpa [X86Int.St.reg, hregs4, hregs3, hregs2, hregs1] using hs0_10) hinv4
  obtain ⟨s6, hs6, hrip6, hregs6, _, hk6, hcm6, hbl6, hpos6, hmp6, hmw6, hctx6,
      hmp6_b, hmp6_rw, hmp6_l, hoth6, hinv6⟩ :=
    st_memp_step_inv (vc := vc) (ls := ls) (chk := 9) (i := 5) (by decide) hrip5
      hcm5 hbl5 hpos5 hmp5 hmw5 hctx5 hmp5_b hmp5_rw hmp5_l rfl (vc.ctx_bounds h).1
      hmi.2.1.2.2.2 hne_mp_mw hne_mp_ctx (by simp only [memWSize] at hbd; omega) hmi.2.2.1.2.1
      (by simpa [X86Int.St.reg, hregs5, hregs4, hregs3, hregs2, hregs1] using hs0_9)
      (by simpa [X86Int.St.reg, hregs5, hregs4, hregs3, hregs2, hregs1] using hs0_0)
      (by simpa [X86Int.St.reg, hregs5, hregs4, hregs3, hregs2, hregs1] using hs0_10) hinv5
  obtain ⟨s7, hs7, hrip7, hregs7, _, hk7, hcm7, hbl7, hpos7, hmp7, hmw7, hctx7,
      hmp7_b, hmp7_rw, hmp7_l, hoth7, hinv7⟩ :=
    st_memp_step_inv (vc := vc) (ls := ls) (chk := 9) (i := 6) (by decide) hrip6
      hcm6 hbl6 hpos6 hmp6 hmw6 hctx6 hmp6_b hmp6_rw hmp6_l rfl (vc.ctx_bounds h).1
      hmi.2.1.2.2.2 hne_mp_mw hne_mp_ctx (by simp only [memWSize] at hbd; omega) hmi.2.2.1.2.1
      (by simpa [X86Int.St.reg, hregs6, hregs5, hregs4, hregs3, hregs2, hregs1] using hs0_9)
      (by simpa [X86Int.St.reg, hregs6, hregs5, hregs4, hregs3, hregs2, hregs1] using hs0_0)
      (by simpa [X86Int.St.reg, hregs6, hregs5, hregs4, hregs3, hregs2, hregs1] using hs0_10) hinv6
  obtain ⟨s8, hs8, hrip8, hregs8, _, hk8, hcm8, hbl8, hpos8, hmp8, hmw8, hctx8,
      hmp8_b, hmp8_rw, hmp8_l, hoth8, hinv8⟩ :=
    st_memp_step_inv (vc := vc) (ls := ls) (chk := 9) (i := 7) (by decide) hrip7
      hcm7 hbl7 hpos7 hmp7 hmw7 hctx7 hmp7_b hmp7_rw hmp7_l rfl (vc.ctx_bounds h).1
      hmi.2.1.2.2.2 hne_mp_mw hne_mp_ctx (by simp only [memWSize] at hbd; omega) hmi.2.2.1.2.1
      (by simpa [X86Int.St.reg, hregs7, hregs6, hregs5, hregs4, hregs3, hregs2, hregs1] using hs0_9)
      (by simpa [X86Int.St.reg, hregs7, hregs6, hregs5, hregs4, hregs3, hregs2, hregs1] using hs0_0)
      (by simpa [X86Int.St.reg, hregs7, hregs6, hregs5, hregs4, hregs3, hregs2, hregs1] using hs0_10) hinv7
  have hregs_eq : s8.regs = s0.regs := by
    rw [hregs8, hregs7, hregs6, hregs5, hregs4, hregs3, hregs2, hregs1]
  exact st_finish_spec (vc := vc) .w64 .st64 (c_jmp := 9) hp_eff (by rw [hrip8]; decide) hbd h hor hrs hpcW hle
    h280 h288
    (Reaches.trans hrch0 (Reaches.step_trans hs1 (Reaches.step_trans hs2 (Reaches.step_trans hs3
      (Reaches.step_trans hs4 (Reaches.step_trans hs5 (Reaches.step_trans hs6 (Reaches.step_trans hs7 (Reaches.step hs8)))))))))
    (by rw [hk8, hk7, hk6, hk5, hk4, hk3, hk2, hk1, hk0]) hcm8 hbl8 hpos8
    (by simpa [RegsOk, X86Int.St.reg, hregs_eq] using hr0)
    (by simpa [X86Int.St.reg, hregs_eq] using hs0_4)
    (by simpa [X86Int.St.reg, hregs_eq] using hs0_15) rfl hmp8_b hmp8_rw hmp8_l
    hmw8 hmp8 hctx8
    (fun u hu h1 h2 => hoth8 u (hoth7 u (hoth6 u (hoth5 u (hoth4 u (hoth3 u (hoth2 u (hoth1 u (hoth0 u hu h1) h2) h2) h2) h2) h2) h2) h2) h2)
    hinv8

end SporosVmPrimMem
