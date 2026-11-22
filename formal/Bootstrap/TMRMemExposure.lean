/-
TMRMemExposure: Decision 50 machine-code exposure tracking on triplicated
page-disjoint memory arenas under `X86Int` architectural and physical fault
semantics (`X86Dec`, `X86Int`, `PDist`, `TMR`, `TMRExposure`).

Closes the open obligations noted in `TMRExposure.lean:20-25`:
  1. Models 3 page-disjoint writable arenas in `X86Env.Mem` (`arena0 = 0x100000`,
     `arena1 = 0x200000`, `arena2 = 0x300000`, `4096` bytes each, separated by
     unmapped gaps) storing `(step, acc_exposure, val[w], last_step[w])` for
     slots `w < N` (`N ≤ 255`).
  2. Instantiates concrete `TMR.Channel` projections for mapped memory words
     under real `X86Int.memFlips` (`Rates.mem`).
  3. Defines the canonical x86-64 machine-code kernel `memExposureKernelCode`
     strictly within `X86Dec.decode` / `X86Int.Ins` (computing `delta^2` via a
     shift-and-add loop with `alu .add`/`.and`/`shrI`/`jcc`), majority-voting
     and repairing `(val[w], last_step[w], step, acc)` across all 3 arenas
     (`exit(3)` on any 3-way split, `exit(2)` on 64-bit overflow or invalid slot),
     and writing `(step + 1, acc + delta^2, v_new, step + 1)` to all 3 arenas.
  4. Proves `memExposureIm_decode`, `memExposure_arch_majority`,
     `memExposure_arch_refuse`, `memExposure_arch_overflow`, `sdc_cert_zero`,
     `memExposure_phys_bound`, and a concrete `PDist.p7` rational witness.
-/
import Bootstrap.PDist
import Bootstrap.TMR
import Bootstrap.TMRExposure
import Bootstrap.TMRVoter
import Bootstrap.X86Env
import Bootstrap.X86Int
import Bootstrap.X86Dec

namespace TMRMemExposure

open X86Env X86Int X86Dec PDist TMR

/-! ### Three page-disjoint 4 KiB arenas in `X86Env.Mem` -/

def arena0 : Nat := 0x100000
def arena1 : Nat := 0x200000
def arena2 : Nat := 0x300000
def arenaSize : Nat := X86Env.page
def maxSlots : Nat := 255

def arenaBase : Nat → Nat
  | 0 => arena0
  | 1 => arena1
  | _ => arena2

/-- Arena layout offsets within each 4096-byte page:
  - `0x00`: `step` (8 bytes)
  - `0x08`: `acc_exposure` (8 bytes)
  - `0x10 + 16 * w`: `val[w]` (8 bytes)
  - `0x18 + 16 * w`: `last_step[w]` (8 bytes) -/
def offStep : Nat := 0
def offAcc : Nat := 8
def offVal (w : Nat) : Nat := 16 + 16 * w
def offLast (w : Nat) : Nat := 24 + 16 * w

def arenaRegion (base : Nat) (data : List Nat) : X86Env.Region :=
  ⟨base, data, true⟩

def arenaMem (d0 d1 d2 : List Nat) : X86Env.Mem :=
  [arenaRegion arena0 d0, arenaRegion arena1 d1, arenaRegion arena2 d2]

def ArenaDataOk (d : List Nat) : Prop :=
  d.length = arenaSize ∧ ∀ b ∈ d, b < 256

theorem arenaSize_eq : arenaSize = 4096 := rfl

theorem arenaMem_wf {d0 d1 d2 : List Nat}
    (h0 : d0.length = arenaSize) (h1 : d1.length = arenaSize) (h2 : d2.length = arenaSize) :
    (arenaMem d0 d1 d2).wf := by
  refine ⟨?_, ?_⟩
  · intro r hr
    simp only [arenaMem, List.mem_cons, List.not_mem_nil, or_false] at hr
    rcases hr with rfl | rfl | rfl <;> simp only [arenaRegion, arena0, arena1, arena2, X86Env.page] <;> decide
  · simp only [arenaMem, List.pairwise_cons, List.mem_cons, List.not_mem_nil, or_false,
      Region.hi, arenaRegion, h0, h1, h2]
    refine ⟨?_, ?_, by simp⟩
    · intro r hr
      rcases hr with rfl | rfl <;> exact Or.inl (by simp [arena0, arena1, arena2, arenaSize, X86Env.page])
    · intro r hr
      rcases hr with rfl
      exact Or.inl (by simp [arena1, arena2, arenaSize, X86Env.page])

theorem arenaMem_gap0_unmapped {d0 d1 d2 : List Nat}
    (h0 : d0.length = arenaSize) {a : Nat}
    (hlo : arena0 + arenaSize ≤ a) (hhi : a < arena1) :
    X86Env.Mem.rd (arenaMem d0 d1 d2) a = none := by
  simp only [arena0, arena1, arenaSize, X86Env.page] at h0 hlo hhi
  have hr0 : ¬ (arena0 ≤ a ∧ a < arena0 + d0.length) := by simp only [arena0]; omega
  have hr1 : ¬ (arena1 ≤ a ∧ a < arena1 + d1.length) := by simp only [arena1]; omega
  have hr2 : ¬ (arena2 ≤ a ∧ a < arena2 + d2.length) := by simp only [arena2]; omega
  simp [X86Env.Mem.rd, X86Env.Mem.find, arenaMem, arenaRegion, Region.hi, hr0, hr1, hr2]

theorem arenaMem_gap1_unmapped {d0 d1 d2 : List Nat}
    (h0 : d0.length = arenaSize) (h1 : d1.length = arenaSize) {a : Nat}
    (hlo : arena1 + arenaSize ≤ a) (hhi : a < arena2) :
    X86Env.Mem.rd (arenaMem d0 d1 d2) a = none := by
  simp only [arena1, arena2, arenaSize, X86Env.page] at h0 h1 hlo hhi
  have hr0 : ¬ (arena0 ≤ a ∧ a < arena0 + d0.length) := by simp only [arena0]; omega
  have hr1 : ¬ (arena1 ≤ a ∧ a < arena1 + d1.length) := by simp only [arena1]; omega
  have hr2 : ¬ (arena2 ≤ a ∧ a < arena2 + d2.length) := by simp only [arena2]; omega
  simp [X86Env.Mem.rd, X86Env.Mem.find, arenaMem, arenaRegion, Region.hi, hr0, hr1, hr2]

/-! ### Pure 64-bit word operations on 4 KiB arena byte buffers -/

def rdBytes (d : List Nat) (off : Nat) : Nat → List Nat
  | 0 => []
  | n + 1 => (d[off]?).getD 0 :: rdBytes d (off + 1) n

def rdWord (d : List Nat) (off : Nat) : Nat :=
  X86Int.leVal (rdBytes d off 8)

def wrBytes (d : List Nat) : Nat → List Nat → List Nat
  | _, [] => d
  | off, b :: bs => wrBytes (d.set off (b % 256)) (off + 1) bs

def wrWord (d : List Nat) (off v : Nat) : List Nat :=
  wrBytes d off (X86Int.bytes8 v)

def setArenaState (d : List Nat) (w step acc val last : Nat) : List Nat :=
  wrWord (wrWord (wrWord (wrWord d offStep step) offAcc acc) (offVal w) val) (offLast w) last

theorem wrBytes_length : ∀ (d : List Nat) (off : Nat) (bs : List Nat),
    (wrBytes d off bs).length = d.length
  | _, _, [] => rfl
  | d, off, _ :: bs => by simp only [wrBytes, wrBytes_length, List.length_set]

theorem wrWord_length (d : List Nat) (off v : Nat) :
    (wrWord d off v).length = d.length :=
  wrBytes_length d off (X86Int.bytes8 v)

theorem setArenaState_length (d : List Nat) (w step acc val last : Nat) :
    (setArenaState d w step acc val last).length = d.length := by
  simp only [setArenaState, wrWord_length]

private theorem mem_set_lt256 {d : List Nat} (hd : ∀ b ∈ d, b < 256)
    (off v : Nat) : ∀ b ∈ d.set off (v % 256), b < 256 := by
  intro b hb
  obtain ⟨i, hi, hget⟩ := List.mem_iff_getElem.1 hb
  rw [List.getElem_set] at hget
  split at hget
  · subst hget; exact Nat.mod_lt _ (by decide)
  · subst hget
    rw [List.length_set] at hi
    exact hd d[i] (List.getElem_mem hi)

theorem wrBytes_bytes_lt : ∀ (d : List Nat) (off : Nat) (bs : List Nat),
    (∀ b ∈ d, b < 256) → ∀ b ∈ wrBytes d off bs, b < 256
  | _, _, [], hd => hd
  | _, off, b :: bs, hd => wrBytes_bytes_lt _ (off + 1) bs (mem_set_lt256 hd off b)

theorem wrWord_bytes_lt {d : List Nat} (hd : ∀ b ∈ d, b < 256) (off v : Nat) :
    ∀ b ∈ wrWord d off v, b < 256 :=
  wrBytes_bytes_lt d off (X86Int.bytes8 v) hd

theorem ArenaDataOk.wrWord {d : List Nat} (hd : ArenaDataOk d) (off v : Nat) :
    ArenaDataOk (wrWord d off v) :=
  ⟨by rw [wrWord_length, hd.1], wrWord_bytes_lt hd.2 off v⟩

theorem ArenaDataOk.setArenaState {d : List Nat} (hd : ArenaDataOk d) (w step acc val last : Nat) :
    ArenaDataOk (setArenaState d w step acc val last) :=
  (((hd.wrWord offStep step).wrWord offAcc acc).wrWord (offVal w) val).wrWord (offLast w) last

theorem rdBytes_length (d : List Nat) : ∀ off n, (rdBytes d off n).length = n
  | _, 0 => rfl
  | off, n + 1 => by simp [rdBytes, rdBytes_length d (off + 1) n]

private theorem leVal_lt_256_pow : ∀ (bs : List Nat), (∀ b ∈ bs, b < 256) →
    X86Int.leVal bs < 256 ^ bs.length
  | [], _ => by simp [X86Int.leVal]
  | b :: bs, hbs => by
    have hb : b < 256 := hbs b (List.mem_cons_self ..)
    have hr : X86Int.leVal bs < 256 ^ bs.length :=
      leVal_lt_256_pow bs (fun x hx => hbs x (List.mem_cons_of_mem _ hx))
    simp only [X86Int.leVal, List.length_cons, Nat.pow_succ]
    omega

theorem rdBytes_lt256 {d : List Nat} (hd : ∀ b ∈ d, b < 256) : ∀ off n,
    ∀ b ∈ rdBytes d off n, b < 256
  | _, 0, _, hb => by simp [rdBytes] at hb
  | off, n + 1, b, hb => by
    simp only [rdBytes, List.mem_cons] at hb
    rcases hb with rfl | hb
    · cases hget : d[off]? with
      | none => simp
      | some x =>
        simp only [Option.getD_some]
        exact hd x (List.mem_of_getElem? hget)
    · exact rdBytes_lt256 hd (off + 1) n b hb

theorem rdWord_lt_W {d : List Nat} (hd : ∀ b ∈ d, b < 256) (off : Nat) :
    rdWord d off < X86Env.W := by
  have hlen : (rdBytes d off 8).length = 8 := rdBytes_length d off 8
  have hlt := leVal_lt_256_pow (rdBytes d off 8) (rdBytes_lt256 hd off 8)
  rw [hlen] at hlt
  simpa [rdWord, X86Env.W] using hlt

private theorem getElem?_set_self' (d : List Nat) (off v : Nat) (h : off < d.length) :
    (d.set off v)[off]? = some v := by
  simp [h]

private theorem getElem?_wrBytes_ne : ∀ (d : List Nat) (off : Nat) (bs : List Nat) (a : Nat),
    (a < off ∨ off + bs.length ≤ a) → (wrBytes d off bs)[a]? = d[a]?
  | _, _, [], _, _ => rfl
  | d, off, b :: bs, a, ha => by
    have ha' : a < off + 1 ∨ (off + 1) + bs.length ≤ a := by
      simp only [List.length_cons] at ha; omega
    have hne : off ≠ a := by
      simp only [List.length_cons] at ha; omega
    rw [wrBytes, getElem?_wrBytes_ne _ (off + 1) bs a ha', List.getElem?_set_ne hne]

private theorem rdBytes_wrBytes_disjoint (d : List Nat) (off1 : Nat) (bs : List Nat) :
    ∀ off2 n, (off1 + bs.length ≤ off2 ∨ off2 + n ≤ off1) →
      rdBytes (wrBytes d off1 bs) off2 n = rdBytes d off2 n
  | _, 0, _ => rfl
  | off2, n + 1, hdisj => by
    have h1 : off2 < off1 ∨ off1 + bs.length ≤ off2 := by omega
    have h2 : off1 + bs.length ≤ off2 + 1 ∨ (off2 + 1) + n ≤ off1 := by omega
    simp only [rdBytes, getElem?_wrBytes_ne d off1 bs off2 h1,
      rdBytes_wrBytes_disjoint d off1 bs (off2 + 1) n h2]

theorem rdWord_wrWord_disjoint {d : List Nat} {off1 off2 : Nat} (v : Nat)
    (hdisj : off1 + 8 ≤ off2 ∨ off2 + 8 ≤ off1) :
    rdWord (wrWord d off1 v) off2 = rdWord d off2 := by
  have hlen : (X86Int.bytes8 v).length = 8 := by simp [X86Int.bytes8]
  unfold rdWord wrWord
  rw [rdBytes_wrBytes_disjoint d off1 (X86Int.bytes8 v) off2 8 (by rw [hlen]; exact hdisj)]

private theorem rdBytes_wrBytes_same_gen (d : List Nat) :
    ∀ (off : Nat) (bs : List Nat),
      (∀ b ∈ bs, b < 256) → off + bs.length ≤ d.length →
      rdBytes (wrBytes d off bs) off bs.length = bs
  | _, [], _, _ => rfl
  | off, b :: bs, hlt, hd => by
    have hb : b < 256 := hlt b (List.mem_cons_self ..)
    have hdisj : off < off + 1 ∨ (off + 1) + bs.length ≤ off := Or.inl (by omega)
    have h0 : (wrBytes (d.set off (b % 256)) (off + 1) bs)[off]? = some (b % 256) := by
      rw [getElem?_wrBytes_ne _ (off + 1) bs off hdisj,
        getElem?_set_self' d off (b % 256) (by simp at hd; omega)]
    have hrest := rdBytes_wrBytes_same_gen (d.set off (b % 256)) (off + 1) bs
      (fun x hx => hlt x (List.mem_cons_of_mem _ hx)) (by rw [List.length_set]; simp at hd; omega)
    simp only [wrBytes, rdBytes]
    rw [h0, Option.getD_some, hrest, Nat.mod_eq_of_lt hb]

private theorem rdBytes_wrBytes_same (d : List Nat) (off : Nat) (bs : List Nat)
    (hbs : bs.length = 8) (hlt : ∀ b ∈ bs, b < 256) (hd : off + 8 ≤ d.length) :
    rdBytes (wrBytes d off bs) off 8 = bs := by
  have h := rdBytes_wrBytes_same_gen d off bs hlt (by omega)
  rwa [hbs] at h

private theorem leVal_bytes8 (v : Nat) (hv : v < X86Env.W) :
    X86Int.leVal (X86Int.bytes8 v) = v := by
  simp only [X86Int.bytes8, List.range_succ, List.range_zero, List.nil_append, List.cons_append,
    List.map_cons, List.map_nil, X86Int.leVal]
  have hW : X86Env.W = 2 ^ 64 := rfl
  omega

private theorem bytes8_lt256 (v : Nat) : ∀ b ∈ X86Int.bytes8 v, b < 256 := by
  intro b hb
  simp only [X86Int.bytes8, List.mem_map, List.mem_range] at hb
  obtain ⟨i, _, rfl⟩ := hb
  exact Nat.mod_lt _ (by decide)

theorem rdWord_wrWord_same {d : List Nat} {off v : Nat}
    (hd : off + 8 ≤ d.length) (hv : v < X86Env.W) :
    rdWord (wrWord d off v) off = v := by
  unfold rdWord wrWord
  rw [rdBytes_wrBytes_same d off (X86Int.bytes8 v) (by simp [X86Int.bytes8]) (bytes8_lt256 v) hd]
  exact leVal_bytes8 v hv

private theorem getElem?_wrBytes_in : ∀ (d : List Nat) (off : Nat) (bs : List Nat) (k : Nat),
    k < bs.length → off + bs.length ≤ d.length →
    (wrBytes d off bs)[off + k]? = some (bs[k]! % 256)
  | d, off, b :: bs, 0, _, hlen => by
    have ha : off < off + 1 ∨ (off + 1) + bs.length ≤ off := Or.inl (by omega)
    show (wrBytes (d.set off (b % 256)) (off + 1) bs)[off + 0]? = some (b % 256)
    rw [show off + 0 = off from Nat.add_zero off,
      getElem?_wrBytes_ne _ (off + 1) bs off ha,
      getElem?_set_self' d off (b % 256) (by simp at hlen; omega)]
  | d, off, b :: bs, k + 1, hk, hlen => by
    have hk' : k < bs.length := by simp at hk; omega
    have hlen' : (off + 1) + bs.length ≤ (d.set off (b % 256)).length := by
      rw [List.length_set]; simp at hlen; omega
    have heq : off + (k + 1) = (off + 1) + k := by omega
    rw [wrBytes, heq, getElem?_wrBytes_in _ (off + 1) bs k hk' hlen']
    rfl

theorem wrBytes_wrBytes_same {d : List Nat} {off : Nat} {bs1 bs2 : List Nat}
    (h1 : bs1.length = 8) (h2 : bs2.length = 8) (hd : off + 8 ≤ d.length) :
    wrBytes (wrBytes d off bs1) off bs2 = wrBytes d off bs2 := by
  apply List.ext_getElem?
  intro i
  by_cases hi : off ≤ i ∧ i < off + 8
  · obtain ⟨k, hk, rfl⟩ : ∃ k < 8, i = off + k := ⟨i - off, by omega, by omega⟩
    have hlen1 : (wrBytes d off bs1).length = d.length := wrBytes_length d off bs1
    rw [getElem?_wrBytes_in _ off bs2 k (by omega) (by rw [hlen1, h2]; exact hd)]
    rw [getElem?_wrBytes_in d off bs2 k (by omega) (by rw [h2]; exact hd)]
  · have hdisj1 : i < off ∨ off + bs1.length ≤ i := by omega
    have hdisj2 : i < off ∨ off + bs2.length ≤ i := by omega
    rw [getElem?_wrBytes_ne _ off bs2 i hdisj2, getElem?_wrBytes_ne d off bs1 i hdisj1,
      getElem?_wrBytes_ne d off bs2 i hdisj2]

theorem wrWord_wrWord_same {d : List Nat} {off v1 v2 : Nat} (hd : off + 8 ≤ d.length) :
    wrWord (wrWord d off v1) off v2 = wrWord d off v2 :=
  wrBytes_wrBytes_same (by simp [X86Int.bytes8]) (by simp [X86Int.bytes8]) hd

theorem wrBytes_wrBytes_comm {d : List Nat} {off1 off2 : Nat} {bs1 bs2 : List Nat}
    (h1 : bs1.length = 8) (h2 : bs2.length = 8)
    (hd1 : off1 + 8 ≤ d.length) (hd2 : off2 + 8 ≤ d.length)
    (hdisj : off1 + 8 ≤ off2 ∨ off2 + 8 ≤ off1) :
    wrBytes (wrBytes d off1 bs1) off2 bs2 = wrBytes (wrBytes d off2 bs2) off1 bs1 := by
  apply List.ext_getElem?
  intro i
  by_cases hi1 : off1 ≤ i ∧ i < off1 + 8
  · obtain ⟨k, hk, rfl⟩ : ∃ k < 8, i = off1 + k := ⟨i - off1, by omega, by omega⟩
    have hdisj2 : off1 + k < off2 ∨ off2 + bs2.length ≤ off1 + k := by omega
    rw [getElem?_wrBytes_ne _ off2 bs2 (off1 + k) hdisj2]
    rw [getElem?_wrBytes_in d off1 bs1 k (by omega) (by rw [h1]; exact hd1)]
    rw [getElem?_wrBytes_in _ off1 bs1 k (by omega) (by rw [wrBytes_length, h1]; exact hd1)]
  · by_cases hi2 : off2 ≤ i ∧ i < off2 + 8
    · obtain ⟨k, hk, rfl⟩ : ∃ k < 8, i = off2 + k := ⟨i - off2, by omega, by omega⟩
      have hdisj1 : off2 + k < off1 ∨ off1 + bs1.length ≤ off2 + k := by omega
      rw [getElem?_wrBytes_in _ off2 bs2 k (by omega) (by rw [wrBytes_length, h2]; exact hd2)]
      rw [getElem?_wrBytes_ne _ off1 bs1 (off2 + k) hdisj1]
      rw [getElem?_wrBytes_in d off2 bs2 k (by omega) (by rw [h2]; exact hd2)]
    · have hdisj1 : i < off1 ∨ off1 + bs1.length ≤ i := by omega
      have hdisj2 : i < off2 ∨ off2 + bs2.length ≤ i := by omega
      rw [getElem?_wrBytes_ne _ off2 bs2 i hdisj2, getElem?_wrBytes_ne _ off1 bs1 i hdisj1,
        getElem?_wrBytes_ne _ off1 bs1 i hdisj1, getElem?_wrBytes_ne _ off2 bs2 i hdisj2]

theorem wrWord_wrWord_comm {d : List Nat} {off1 off2 v1 v2 : Nat}
    (hd1 : off1 + 8 ≤ d.length) (hd2 : off2 + 8 ≤ d.length)
    (hdisj : off1 + 8 ≤ off2 ∨ off2 + 8 ≤ off1) :
    wrWord (wrWord d off1 v1) off2 v2 = wrWord (wrWord d off2 v2) off1 v1 :=
  wrBytes_wrBytes_comm (by simp [X86Int.bytes8]) (by simp [X86Int.bytes8]) hd1 hd2 hdisj

theorem setArenaState_rdStep {d : List Nat} (hd : d.length = arenaSize) {w : Nat}
    {step acc val last : Nat} (hs : step < X86Env.W) :
    rdWord (setArenaState d w step acc val last) offStep = step := by
  have hdisj1 : offLast w + 8 ≤ offStep ∨ offStep + 8 ≤ offLast w := by
    simp only [offStep, offLast]; exact Or.inr (by omega)
  have hdisj2 : offVal w + 8 ≤ offStep ∨ offStep + 8 ≤ offVal w := by
    simp only [offStep, offVal]; exact Or.inr (by omega)
  have hdisj3 : offAcc + 8 ≤ offStep ∨ offStep + 8 ≤ offAcc := by
    simp only [offStep, offAcc]; exact Or.inr (by omega)
  unfold setArenaState
  rw [rdWord_wrWord_disjoint last hdisj1, rdWord_wrWord_disjoint val hdisj2,
    rdWord_wrWord_disjoint acc hdisj3, rdWord_wrWord_same (by simp [offStep, hd, arenaSize, X86Env.page]) hs]

theorem setArenaState_rdAcc {d : List Nat} (hd : d.length = arenaSize) {w : Nat}
    {step acc val last : Nat} (ha : acc < X86Env.W) :
    rdWord (setArenaState d w step acc val last) offAcc = acc := by
  have hdisj1 : offLast w + 8 ≤ offAcc ∨ offAcc + 8 ≤ offLast w := by
    simp only [offAcc, offLast]; exact Or.inr (by omega)
  have hdisj2 : offVal w + 8 ≤ offAcc ∨ offAcc + 8 ≤ offVal w := by
    simp only [offAcc, offVal]; exact Or.inr (by omega)
  unfold setArenaState
  rw [rdWord_wrWord_disjoint last hdisj1, rdWord_wrWord_disjoint val hdisj2,
    rdWord_wrWord_same (by simp [offAcc, wrWord_length, hd, arenaSize, X86Env.page]) ha]

theorem setArenaState_rdVal {d : List Nat} (hd : d.length = arenaSize) {w : Nat} (hw : w < maxSlots)
    {step acc val last : Nat} (hv : val < X86Env.W) :
    rdWord (setArenaState d w step acc val last) (offVal w) = val := by
  have hdisj1 : offLast w + 8 ≤ offVal w ∨ offVal w + 8 ≤ offLast w := by
    simp only [offVal, offLast]; exact Or.inr (by omega)
  unfold setArenaState
  rw [rdWord_wrWord_disjoint last hdisj1,
    rdWord_wrWord_same (by simp only [offVal, wrWord_length, hd, arenaSize, X86Env.page, maxSlots] at *; omega) hv]

theorem setArenaState_rdLast {d : List Nat} (hd : d.length = arenaSize) {w : Nat} (hw : w < maxSlots)
    {step acc val last : Nat} (hl : last < X86Env.W) :
    rdWord (setArenaState d w step acc val last) (offLast w) = last := by
  unfold setArenaState
  rw [rdWord_wrWord_same (by simp only [offLast, wrWord_length, hd, arenaSize, X86Env.page, maxSlots] at *; omega) hl]

theorem setArenaState_rdOtherVal {d : List Nat} {w w' : Nat} (hne : w ≠ w')
    (step acc val last : Nat) :
    rdWord (setArenaState d w step acc val last) (offVal w') = rdWord d (offVal w') := by
  have h1 : offLast w + 8 ≤ offVal w' ∨ offVal w' + 8 ≤ offLast w := by simp [offLast, offVal]; omega
  have h2 : offVal w + 8 ≤ offVal w' ∨ offVal w' + 8 ≤ offVal w := by simp [offVal]; omega
  have h3 : offAcc + 8 ≤ offVal w' ∨ offVal w' + 8 ≤ offAcc := by
    simp only [offAcc, offVal]; exact Or.inl (by omega)
  have h4 : offStep + 8 ≤ offVal w' ∨ offVal w' + 8 ≤ offStep := by
    simp only [offStep, offVal]; exact Or.inl (by omega)
  unfold setArenaState
  rw [rdWord_wrWord_disjoint _ h1, rdWord_wrWord_disjoint _ h2,
    rdWord_wrWord_disjoint _ h3, rdWord_wrWord_disjoint _ h4]

theorem setArenaState_rdOtherLast {d : List Nat} {w w' : Nat} (hne : w ≠ w')
    (step acc val last : Nat) :
    rdWord (setArenaState d w step acc val last) (offLast w') = rdWord d (offLast w') := by
  have h1 : offLast w + 8 ≤ offLast w' ∨ offLast w' + 8 ≤ offLast w := by simp [offLast]; omega
  have h2 : offVal w + 8 ≤ offLast w' ∨ offLast w' + 8 ≤ offVal w := by simp [offVal, offLast]; omega
  have h3 : offAcc + 8 ≤ offLast w' ∨ offLast w' + 8 ≤ offAcc := by
    simp only [offAcc, offLast]; exact Or.inl (by omega)
  have h4 : offStep + 8 ≤ offLast w' ∨ offLast w' + 8 ≤ offStep := by
    simp only [offStep, offLast]; exact Or.inl (by omega)
  unfold setArenaState
  rw [rdWord_wrWord_disjoint _ h1, rdWord_wrWord_disjoint _ h2,
    rdWord_wrWord_disjoint _ h3, rdWord_wrWord_disjoint _ h4]

/-! ### Relating `X86Env.Mem` (`rd64` / `wr64`) to `arenaMem` -/

private theorem rd_arena0 {d0 d1 d2 : List Nat} (h0 : d0.length = arenaSize)
    (off : Nat) (hoff : off < arenaSize) :
    X86Env.Mem.rd (arenaMem d0 d1 d2) (arena0 + off) = some ((d0[off]?).getD 0) := by
  have hlt : off < d0.length := by omega
  have hcond : arena0 ≤ arena0 + off ∧ arena0 + off < arena0 + d0.length := by omega
  simp [X86Env.Mem.rd, X86Env.Mem.find, arenaMem, arenaRegion, Region.hi, hcond,
    Nat.add_sub_cancel_left, hlt]

private theorem rd_arena1 {d0 d1 d2 : List Nat} (h0 : d0.length = arenaSize)
    (h1 : d1.length = arenaSize) (off : Nat) (hoff : off < arenaSize) :
    X86Env.Mem.rd (arenaMem d0 d1 d2) (arena1 + off) = some ((d1[off]?).getD 0) := by
  have hlt : off < d1.length := by omega
  have h1' : arena0 + d0.length ≤ arena1 := by simp [h0, arena0, arena1, arenaSize, X86Env.page]
  have hr0 : ¬ (arena0 ≤ arena1 + off ∧ arena1 + off < arena0 + d0.length) := by omega
  have hr1 : arena1 ≤ arena1 + off ∧ arena1 + off < arena1 + d1.length := by omega
  simp [X86Env.Mem.rd, X86Env.Mem.find, arenaMem, arenaRegion, Region.hi,
    hr0, hr1, Nat.add_sub_cancel_left, hlt]

private theorem rd_arena2 {d0 d1 d2 : List Nat} (h0 : d0.length = arenaSize)
    (h1 : d1.length = arenaSize) (h2 : d2.length = arenaSize) (off : Nat) (hoff : off < arenaSize) :
    X86Env.Mem.rd (arenaMem d0 d1 d2) (arena2 + off) = some ((d2[off]?).getD 0) := by
  have hlt : off < d2.length := by omega
  have h0' : arena0 + d0.length ≤ arena2 := by simp [h0, arena0, arena2, arenaSize, X86Env.page]
  have h1' : arena1 + d1.length ≤ arena2 := by simp [h1, arena1, arena2, arenaSize, X86Env.page]
  have hr0 : ¬ (arena0 ≤ arena2 + off ∧ arena2 + off < arena0 + d0.length) := by omega
  have hr1 : ¬ (arena1 ≤ arena2 + off ∧ arena2 + off < arena1 + d1.length) := by omega
  have hr2 : arena2 ≤ arena2 + off ∧ arena2 + off < arena2 + d2.length := by omega
  simp [X86Env.Mem.rd, X86Env.Mem.find, arenaMem, arenaRegion, Region.hi,
    hr0, hr1, hr2, Nat.add_sub_cancel_left, hlt]

private theorem rdN_arena0 {d0 d1 d2 : List Nat} (h0 : d0.length = arenaSize) :
    ∀ n off, off + n ≤ arenaSize →
      X86Env.Mem.rdN (arenaMem d0 d1 d2) (arena0 + off) n = some (rdBytes d0 off n)
  | 0, _, _ => rfl
  | n + 1, off, h => by
    have h1 := rd_arena0 (d0 := d0) (d1 := d1) (d2 := d2) h0 off (by omega)
    have h2 := rdN_arena0 (d0 := d0) (d1 := d1) (d2 := d2) h0 n (off + 1) (by omega)
    have heq : arena0 + off + 1 = arena0 + (off + 1) := by omega
    simp only [X86Env.Mem.rdN, bind, Option.bind, pure]
    rw [h1, heq]
    dsimp only
    rw [h2]; rfl

private theorem rdN_arena1 {d0 d1 d2 : List Nat} (h0 : d0.length = arenaSize)
    (h1 : d1.length = arenaSize) : ∀ n off, off + n ≤ arenaSize →
      X86Env.Mem.rdN (arenaMem d0 d1 d2) (arena1 + off) n = some (rdBytes d1 off n)
  | 0, _, _ => rfl
  | n + 1, off, h => by
    have ha := rd_arena1 (d0 := d0) (d1 := d1) (d2 := d2) h0 h1 off (by omega)
    have hb := rdN_arena1 (d0 := d0) (d1 := d1) (d2 := d2) h0 h1 n (off + 1) (by omega)
    have heq : arena1 + off + 1 = arena1 + (off + 1) := by omega
    simp only [X86Env.Mem.rdN, bind, Option.bind, pure]
    rw [ha, heq]
    dsimp only
    rw [hb]; rfl

private theorem rdN_arena2 {d0 d1 d2 : List Nat} (h0 : d0.length = arenaSize)
    (h1 : d1.length = arenaSize) (h2 : d2.length = arenaSize) : ∀ n off, off + n ≤ arenaSize →
      X86Env.Mem.rdN (arenaMem d0 d1 d2) (arena2 + off) n = some (rdBytes d2 off n)
  | 0, _, _ => rfl
  | n + 1, off, h => by
    have ha := rd_arena2 (d0 := d0) (d1 := d1) (d2 := d2) h0 h1 h2 off (by omega)
    have hb := rdN_arena2 (d0 := d0) (d1 := d1) (d2 := d2) h0 h1 h2 n (off + 1) (by omega)
    have heq : arena2 + off + 1 = arena2 + (off + 1) := by omega
    simp only [X86Env.Mem.rdN, bind, Option.bind, pure]
    rw [ha, heq]
    dsimp only
    rw [hb]; rfl

theorem rd64_arena0 {d0 d1 d2 : List Nat} (h0 : d0.length = arenaSize)
    {off : Nat} (hoff : off + 8 ≤ arenaSize) :
    X86Int.rd64 (arenaMem d0 d1 d2) (arena0 + off) = some (rdWord d0 off) := by
  simp [X86Int.rd64, rdN_arena0 h0 8 off hoff, rdWord]

theorem rd64_arena1 {d0 d1 d2 : List Nat} (h0 : d0.length = arenaSize)
    (h1 : d1.length = arenaSize) {off : Nat} (hoff : off + 8 ≤ arenaSize) :
    X86Int.rd64 (arenaMem d0 d1 d2) (arena1 + off) = some (rdWord d1 off) := by
  simp [X86Int.rd64, rdN_arena1 h0 h1 8 off hoff, rdWord]

theorem rd64_arena2 {d0 d1 d2 : List Nat} (h0 : d0.length = arenaSize)
    (h1 : d1.length = arenaSize) (h2 : d2.length = arenaSize) {off : Nat} (hoff : off + 8 ≤ arenaSize) :
    X86Int.rd64 (arenaMem d0 d1 d2) (arena2 + off) = some (rdWord d2 off) := by
  simp [X86Int.rd64, rdN_arena2 h0 h1 h2 8 off hoff, rdWord]

private theorem wr_arena0 {d0 d1 d2 : List Nat} (h0 : d0.length = arenaSize)
    (off v : Nat) (hoff : off < arenaSize) :
    X86Env.Mem.wr (arenaMem d0 d1 d2) (arena0 + off) v =
      some (arenaMem (d0.set off (v % 256)) d1 d2) := by
  have hr0 : arena0 ≤ arena0 + off ∧ arena0 + off < arena0 + d0.length := by omega
  simp [X86Env.Mem.wr, X86Env.Mem.find, arenaMem, arenaRegion, Region.hi, hr0,
    Nat.add_sub_cancel_left]

private theorem wr_arena1 {d0 d1 d2 : List Nat} (h0 : d0.length = arenaSize)
    (h1 : d1.length = arenaSize) (off v : Nat) (hoff : off < arenaSize) :
    X86Env.Mem.wr (arenaMem d0 d1 d2) (arena1 + off) v =
      some (arenaMem d0 (d1.set off (v % 256)) d2) := by
  have h1' : arena0 + d0.length ≤ arena1 := by simp [h0, arena0, arena1, arenaSize, X86Env.page]
  have hr0 : ¬ (arena0 ≤ arena1 + off ∧ arena1 + off < arena0 + d0.length) := by omega
  have hr1 : arena1 ≤ arena1 + off ∧ arena1 + off < arena1 + d1.length := by omega
  simp [X86Env.Mem.wr, X86Env.Mem.find, arenaMem, arenaRegion, Region.hi, hr0, hr1,
    Nat.add_sub_cancel_left]

private theorem wr_arena2 {d0 d1 d2 : List Nat} (h0 : d0.length = arenaSize)
    (h1 : d1.length = arenaSize) (h2 : d2.length = arenaSize) (off v : Nat) (hoff : off < arenaSize) :
    X86Env.Mem.wr (arenaMem d0 d1 d2) (arena2 + off) v =
      some (arenaMem d0 d1 (d2.set off (v % 256))) := by
  have h0' : arena0 + d0.length ≤ arena2 := by simp [h0, arena0, arena2, arenaSize, X86Env.page]
  have h1' : arena1 + d1.length ≤ arena2 := by simp [h1, arena1, arena2, arenaSize, X86Env.page]
  have hr0 : ¬ (arena0 ≤ arena2 + off ∧ arena2 + off < arena0 + d0.length) := by omega
  have hr1 : ¬ (arena1 ≤ arena2 + off ∧ arena2 + off < arena1 + d1.length) := by omega
  have hr2 : arena2 ≤ arena2 + off ∧ arena2 + off < arena2 + d2.length := by omega
  simp [X86Env.Mem.wr, X86Env.Mem.find, arenaMem, arenaRegion, Region.hi, hr0, hr1, hr2,
    Nat.add_sub_cancel_left]

private theorem wrN_arena0 : ∀ (bs : List Nat) (d0 d1 d2 : List Nat) (off : Nat),
    d0.length = arenaSize → off + bs.length ≤ arenaSize →
    X86Env.Mem.wrN (arenaMem d0 d1 d2) (arena0 + off) bs =
      some (arenaMem (wrBytes d0 off bs) d1 d2)
  | [], _, _, _, _, _, _ => rfl
  | b :: bs, d0, d1, d2, off, h0, hoff => by
    simp only [List.length_cons] at hoff
    have ha := wr_arena0 (d1 := d1) (d2 := d2) h0 off b (by omega)
    have hb := wrN_arena0 bs (d0.set off (b % 256)) d1 d2 (off + 1) (by rw [List.length_set]; exact h0) (by omega)
    have heq : arena0 + off + 1 = arena0 + (off + 1) := by omega
    simp only [X86Env.Mem.wrN, bind, Option.bind]
    rw [ha, heq]
    dsimp only
    rw [hb]; rfl

private theorem wrN_arena1 : ∀ (bs : List Nat) (d0 d1 d2 : List Nat) (off : Nat),
    d0.length = arenaSize → d1.length = arenaSize → off + bs.length ≤ arenaSize →
    X86Env.Mem.wrN (arenaMem d0 d1 d2) (arena1 + off) bs =
      some (arenaMem d0 (wrBytes d1 off bs) d2)
  | [], _, _, _, _, _, _, _ => rfl
  | b :: bs, d0, d1, d2, off, h0, h1, hoff => by
    simp only [List.length_cons] at hoff
    have ha := wr_arena1 (d2 := d2) h0 h1 off b (by omega)
    have hb := wrN_arena1 bs d0 (d1.set off (b % 256)) d2 (off + 1) h0 (by rw [List.length_set]; exact h1) (by omega)
    have heq : arena1 + off + 1 = arena1 + (off + 1) := by omega
    simp only [X86Env.Mem.wrN, bind, Option.bind]
    rw [ha, heq]
    dsimp only
    rw [hb]; rfl

private theorem wrN_arena2 : ∀ (bs : List Nat) (d0 d1 d2 : List Nat) (off : Nat),
    d0.length = arenaSize → d1.length = arenaSize → d2.length = arenaSize → off + bs.length ≤ arenaSize →
    X86Env.Mem.wrN (arenaMem d0 d1 d2) (arena2 + off) bs =
      some (arenaMem d0 d1 (wrBytes d2 off bs))
  | [], _, _, _, _, _, _, _, _ => rfl
  | b :: bs, d0, d1, d2, off, h0, h1, h2, hoff => by
    simp only [List.length_cons] at hoff
    have ha := wr_arena2 h0 h1 h2 off b (by omega)
    have hb := wrN_arena2 bs d0 d1 (d2.set off (b % 256)) (off + 1) h0 h1 (by rw [List.length_set]; exact h2) (by omega)
    have heq : arena2 + off + 1 = arena2 + (off + 1) := by omega
    simp only [X86Env.Mem.wrN, bind, Option.bind]
    rw [ha, heq]
    dsimp only
    rw [hb]; rfl

theorem wr64_arena0 {d0 d1 d2 : List Nat} (h0 : d0.length = arenaSize)
    {off v : Nat} (hoff : off + 8 ≤ arenaSize) :
    X86Int.wr64 (arenaMem d0 d1 d2) (arena0 + off) v =
      some (arenaMem (wrWord d0 off v) d1 d2) :=
  wrN_arena0 (X86Int.bytes8 v) d0 d1 d2 off h0 (by simpa [X86Int.bytes8] using hoff)

theorem wr64_arena1 {d0 d1 d2 : List Nat} (h0 : d0.length = arenaSize)
    (h1 : d1.length = arenaSize) {off v : Nat} (hoff : off + 8 ≤ arenaSize) :
    X86Int.wr64 (arenaMem d0 d1 d2) (arena1 + off) v =
      some (arenaMem d0 (wrWord d1 off v) d2) :=
  wrN_arena1 (X86Int.bytes8 v) d0 d1 d2 off h0 h1 (by simpa [X86Int.bytes8] using hoff)

theorem wr64_arena2 {d0 d1 d2 : List Nat} (h0 : d0.length = arenaSize)
    (h1 : d1.length = arenaSize) (h2 : d2.length = arenaSize) {off v : Nat} (hoff : off + 8 ≤ arenaSize) :
    X86Int.wr64 (arenaMem d0 d1 d2) (arena2 + off) v =
      some (arenaMem d0 d1 (wrWord d2 off v)) :=
  wrN_arena2 (X86Int.bytes8 v) d0 d1 d2 off h0 h1 h2 (by simpa [X86Int.bytes8] using hoff)

/-! ### Concrete `TMR.Channel` projections under real `X86Int.memFlips` (`Rates.mem`) -/

/-- State-level `X86Int.memFlips` channel at rate `Rates.mem`. -/
def memFlipsStChannel (stepF : X86Int.St → X86Int.St) (r : X86Int.Rates) (hr : r.ok) :
    TMR.Channel X86Int.St where
  f := stepF
  step s := X86Int.unif r.mem (stepF s) (X86Int.memFlips (stepF s))
  q := r.mem
  q0 := hr.2.2.1.1
  ok s := X86Int.unif_ok hr.2.2.1 (stepF s) (X86Int.memFlips (stepF s))
  clean s := X86Int.faultFree_unif r.mem (stepF s) (X86Int.memFlips (stepF s))
  le s := X86Int.fm_unif_le hr.2.2.1 (stepF s) (X86Int.memFlips (stepF s))

/-- Helper to update/read a 64-bit word at `arena0 + off` in `arenaMem d0 d1 d2`. -/
def setArena0Word (s : X86Int.St) (d0 d1 d2 : List Nat) (off v : Nat) : X86Int.St :=
  { s with mem := arenaMem (wrWord d0 off v) d1 d2 }

def getArena0Word (s : X86Int.St) (off fallback : Nat) : Nat :=
  (X86Int.rd64 s.mem (arena0 + off)).getD fallback

/-- Concrete `TMR.Channel Nat` projecting a mapped 64-bit word at `arena0 + off`
under real `X86Int.memFlips` (`Rates.mem`). -/
def arenaWordFlipsChannel (s0 : X86Int.St) (d0 d1 d2 : List Nat)
    (h0 : d0.length = arenaSize) (off : Nat) (hoff : off + 8 ≤ arenaSize)
    (fn : Nat → Nat) (hfn : ∀ v, fn v < X86Env.W)
    (r : X86Int.Rates) (hr : r.ok) : TMR.Channel Nat where
  f := fn
  step v :=
    let s' := setArena0Word s0 d0 d1 d2 off (fn v)
    X86Int.relabel (fun t => getArena0Word t off (fn v))
      (X86Int.unif r.mem s' (X86Int.memFlips s'))
  q := r.mem
  q0 := hr.2.2.1.1
  ok v :=
    let s' := setArena0Word s0 d0 d1 d2 off (fn v)
    X86Int.relabel_ok _ (X86Int.unif_ok hr.2.2.1 s' (X86Int.memFlips s'))
  clean v := by
    dsimp only
    rw [X86Int.faultFree_relabel, X86Int.faultFree_unif]
    simp only [List.map_cons, List.map_nil, getArena0Word, setArena0Word]
    have hlen : (wrWord d0 off (fn v)).length = arenaSize := by rw [wrWord_length, h0]
    rw [rd64_arena0 hlen hoff, rdWord_wrWord_same (by omega) (hfn v)]
    rfl
  le v := by
    dsimp only
    rw [X86Int.fm_relabel]
    exact X86Int.fm_unif_le hr.2.2.1 _ _

/-- General mapped memory word channel on any `Mem` supporting `wr64`/`rd64`
under real `X86Int.memFlips` (`Rates.mem`). -/
def mappedWordFlipsChannel (s0 : X86Int.St) (addr : Nat) (fn : Nat → Nat)
    (hclean : ∀ v, X86Int.rd64 ((X86Int.wr64 s0.mem addr (fn v)).getD s0.mem) addr = some (fn v))
    (r : X86Int.Rates) (hr : r.ok) : TMR.Channel Nat where
  f := fn
  step v :=
    let s' := { s0 with mem := (X86Int.wr64 s0.mem addr (fn v)).getD s0.mem }
    X86Int.relabel (fun t => (X86Int.rd64 t.mem addr).getD (fn v))
      (X86Int.unif r.mem s' (X86Int.memFlips s'))
  q := r.mem
  q0 := hr.2.2.1.1
  ok v :=
    X86Int.relabel_ok _ (X86Int.unif_ok hr.2.2.1 _ _)
  clean v := by
    dsimp only
    rw [X86Int.faultFree_relabel, X86Int.faultFree_unif]
    simp only [List.map_cons, List.map_nil, hclean v, Option.getD_some]
  le v := by
    dsimp only
    rw [X86Int.fm_relabel]
    exact X86Int.fm_unif_le hr.2.2.1 _ _

/-- Idle (unwritten) 64-bit word channel (`v ↦ v % W`) for Decision 50 exposure
tracking under `X86Int.memFlips` at rate `Rates.mem`. -/
def idleArenaWordChannel (s0 : X86Int.St) (d0 d1 d2 : List Nat)
    (h0 : d0.length = arenaSize) (off : Nat) (hoff : off + 8 ≤ arenaSize)
    (r : X86Int.Rates) (hr : r.ok) : TMR.Channel Nat :=
  arenaWordFlipsChannel s0 d0 d1 d2 h0 off hoff (fun v => v % X86Env.W)
    (fun v => Nat.mod_lt v (by decide)) r hr

theorem idleArenaWordChannel_iter {s0 : X86Int.St} {d0 d1 d2 : List Nat}
    {h0 : d0.length = arenaSize} {off : Nat} {hoff : off + 8 ≤ arenaSize}
    {r : X86Int.Rates} {hr : r.ok} {v : Nat} (hv : v < X86Env.W) :
    ∀ k, TMR.iter (idleArenaWordChannel s0 d0 d1 d2 h0 off hoff r hr).f (k + 1) v = v
  | 0 => by simp [TMR.iter, idleArenaWordChannel, arenaWordFlipsChannel, Nat.mod_eq_of_lt hv]
  | k + 1 => by
    have h1 : (idleArenaWordChannel s0 d0 d1 d2 h0 off hoff r hr).f v = v := by
      simp [idleArenaWordChannel, arenaWordFlipsChannel, Nat.mod_eq_of_lt hv]
    show TMR.iter _ (k + 1) ((idleArenaWordChannel s0 d0 d1 d2 h0 off hoff r hr).f v) = v
    rw [h1]
    exact idleArenaWordChannel_iter hv k

/-! ### Canonical 415-byte x86-64 exposure-tracking machine-code kernel -/

def memExposureBase : Nat := 0x400000
def memExposurePark : Nat := memExposureBase + 0x185
def memExposureOverflow : Nat := memExposureBase + 0x187
def memExposureRefuse : Nat := memExposureBase + 0x193
def memExposureFuel : Nat := 450

/-- Canonical 415-byte machine-code kernel at `0x400000`.
Inputs: `rdi` (reg 7) = slot `w`, `rsi` (reg 6) = `v_new`.
Majority-votes `(val[w], last_step[w], step, acc)` across `arena0..arena2`,
repairs any 1-copy mismatch across all 3 arenas (`exit(3)` on any 3-way split),
computes `delta := step - last_step[w] + 1` and `delta^2` by shift-and-add,
checks 64-bit overflow (`exit(2)`), and writes `(step+1, acc+delta^2, v_new, step+1)`
to all 3 arenas before parking at `0x400185`. -/
def memExposureKernelCode : List Nat :=
  -- Stage 0: slot bounds check and arena/slot pointer setup (0x000..0x038)
  [ 0x48, 0x81, 0xff, 0xff, 0x00, 0x00, 0x00  -- 0x000: cmp rdi, 255
  , 0x0f, 0x83, 0x7a, 0x01, 0x00, 0x00        -- 0x007: jae 0x400187 (overflow/bad slot)
  , 0x41, 0xb8, 0x00, 0x00, 0x10, 0x00        -- 0x00d: mov r8d, 0x100000
  , 0x41, 0xb9, 0x00, 0x00, 0x20, 0x00        -- 0x013: mov r9d, 0x200000
  , 0x41, 0xba, 0x00, 0x00, 0x30, 0x00        -- 0x019: mov r10d, 0x300000
  , 0x49, 0x89, 0xfb                          -- 0x01f: mov r11, rdi
  , 0x49, 0xc1, 0xe3, 0x04                    -- 0x022: shl r11, 4
  , 0x4d, 0x89, 0xc6                          -- 0x026: mov r14, r8
  , 0x4d, 0x01, 0xde                          -- 0x029: add r14, r11
  , 0x4d, 0x89, 0xcf                          -- 0x02c: mov r15, r9
  , 0x4d, 0x01, 0xdf                          -- 0x02f: add r15, r11
  , 0x4c, 0x89, 0xd2                          -- 0x032: mov rdx, r10
  , 0x4c, 0x01, 0xda                          -- 0x035: add rdx, r11

  -- Stage 1: load, majority-vote, and repair val[w] (0x038..0x067)
  , 0x49, 0x8b, 0x46, 0x10                    -- 0x038: mov rax, [r14 + 16]
  , 0x49, 0x8b, 0x5f, 0x10                    -- 0x03c: mov rbx, [r15 + 16]
  , 0x48, 0x8b, 0x4a, 0x10                    -- 0x040: mov rcx, [rdx + 16]
  , 0x48, 0x39, 0xd8                          -- 0x044: cmp rax, rbx
  , 0x74, 0x12                                -- 0x047: je  0x40005b
  , 0x48, 0x39, 0xc8                          -- 0x049: cmp rax, rcx
  , 0x74, 0x0d                                -- 0x04c: je  0x40005b
  , 0x48, 0x39, 0xcb                          -- 0x04e: cmp rbx, rcx
  , 0x74, 0x05                                -- 0x051: je  0x400058
  , 0xe9, 0x3b, 0x01, 0x00, 0x00              -- 0x053: jmp 0x400193 (refuse)
  , 0x48, 0x89, 0xd8                          -- 0x058: mov rax, rbx
  , 0x49, 0x89, 0x46, 0x10                    -- 0x05b: mov [r14 + 16], rax
  , 0x49, 0x89, 0x47, 0x10                    -- 0x05f: mov [r15 + 16], rax
  , 0x48, 0x89, 0x42, 0x10                    -- 0x063: mov [rdx + 16], rax

  -- Stage 2: load, majority-vote, and repair last_step[w] into r11 (0x067..0x099)
  , 0x49, 0x8b, 0x46, 0x18                    -- 0x067: mov rax, [r14 + 24]
  , 0x49, 0x8b, 0x5f, 0x18                    -- 0x06b: mov rbx, [r15 + 24]
  , 0x48, 0x8b, 0x4a, 0x18                    -- 0x06f: mov rcx, [rdx + 24]
  , 0x48, 0x39, 0xd8                          -- 0x073: cmp rax, rbx
  , 0x74, 0x12                                -- 0x076: je  0x40008a
  , 0x48, 0x39, 0xc8                          -- 0x078: cmp rax, rcx
  , 0x74, 0x0d                                -- 0x07b: je  0x40008a
  , 0x48, 0x39, 0xcb                          -- 0x07d: cmp rbx, rcx
  , 0x74, 0x05                                -- 0x080: je  0x400087
  , 0xe9, 0x0c, 0x01, 0x00, 0x00              -- 0x082: jmp 0x400193 (refuse)
  , 0x48, 0x89, 0xd8                          -- 0x087: mov rax, rbx
  , 0x49, 0x89, 0x46, 0x18                    -- 0x08a: mov [r14 + 24], rax
  , 0x49, 0x89, 0x47, 0x18                    -- 0x08e: mov [r15 + 24], rax
  , 0x48, 0x89, 0x42, 0x18                    -- 0x092: mov [rdx + 24], rax
  , 0x49, 0x89, 0xc3                          -- 0x096: mov r11, rax

  -- Stage 3: load, majority-vote, and repair step into r12 (0x099..0x0c5)
  , 0x49, 0x8b, 0x00                          -- 0x099: mov rax, [r8]
  , 0x49, 0x8b, 0x19                          -- 0x09c: mov rbx, [r9]
  , 0x49, 0x8b, 0x0a                          -- 0x09f: mov rcx, [r10]
  , 0x48, 0x39, 0xd8                          -- 0x0a2: cmp rax, rbx
  , 0x74, 0x12                                -- 0x0a5: je  0x4000b9
  , 0x48, 0x39, 0xc8                          -- 0x0a7: cmp rax, rcx
  , 0x74, 0x0d                                -- 0x0aa: je  0x4000b9
  , 0x48, 0x39, 0xcb                          -- 0x0ac: cmp rbx, rcx
  , 0x74, 0x05                                -- 0x0af: je  0x4000b6
  , 0xe9, 0xdd, 0x00, 0x00, 0x00              -- 0x0b1: jmp 0x400193 (refuse)
  , 0x48, 0x89, 0xd8                          -- 0x0b6: mov rax, rbx
  , 0x49, 0x89, 0x00                          -- 0x0b9: mov [r8], rax
  , 0x49, 0x89, 0x01                          -- 0x0bc: mov [r9], rax
  , 0x49, 0x89, 0x02                          -- 0x0bf: mov [r10], rax
  , 0x49, 0x89, 0xc4                          -- 0x0c2: mov r12, rax

  -- Stage 4: load, majority-vote, and repair acc into r13 (0x0c5..0x0f7)
  , 0x49, 0x8b, 0x40, 0x08                    -- 0x0c5: mov rax, [r8 + 8]
  , 0x49, 0x8b, 0x59, 0x08                    -- 0x0c9: mov rbx, [r9 + 8]
  , 0x49, 0x8b, 0x4a, 0x08                    -- 0x0cd: mov rcx, [r10 + 8]
  , 0x48, 0x39, 0xd8                          -- 0x0d1: cmp rax, rbx
  , 0x74, 0x12                                -- 0x0d4: je  0x4000e8
  , 0x48, 0x39, 0xc8                          -- 0x0d6: cmp rax, rcx
  , 0x74, 0x0d                                -- 0x0d9: je  0x4000e8
  , 0x48, 0x39, 0xcb                          -- 0x0db: cmp rbx, rcx
  , 0x74, 0x05                                -- 0x0de: je  0x4000e5
  , 0xe9, 0xae, 0x00, 0x00, 0x00              -- 0x0e0: jmp 0x400193 (refuse)
  , 0x48, 0x89, 0xd8                          -- 0x0e5: mov rax, rbx
  , 0x49, 0x89, 0x40, 0x08                    -- 0x0e8: mov [r8 + 8], rax
  , 0x49, 0x89, 0x41, 0x08                    -- 0x0ec: mov [r9 + 8], rax
  , 0x49, 0x89, 0x42, 0x08                    -- 0x0f0: mov [r10 + 8], rax
  , 0x49, 0x89, 0xc5                          -- 0x0f4: mov r13, rax

  -- Stage 5: delta := step - last[w] + 1, step_new := step + 1, overflow checks (0x0f7..0x117)
  , 0x4c, 0x89, 0xe0                          -- 0x0f7: mov rax, r12
  , 0x4c, 0x29, 0xd8                          -- 0x0fa: sub rax, r11
  , 0x0f, 0x82, 0x84, 0x00, 0x00, 0x00        -- 0x0fd: jc  0x400187
  , 0x48, 0x83, 0xc0, 0x01                    -- 0x103: add rax, 1
  , 0x0f, 0x82, 0x7a, 0x00, 0x00, 0x00        -- 0x107: jc  0x400187
  , 0x49, 0x83, 0xc4, 0x01                    -- 0x10d: add r12, 1
  , 0x0f, 0x82, 0x70, 0x00, 0x00, 0x00        -- 0x111: jc  0x400187

  -- Stage 6: shift-and-add loop computing rbx := delta^2 (0x117..0x143)
  , 0x48, 0x31, 0xdb                          -- 0x117: xor rbx, rbx
  , 0x48, 0x89, 0xc1                          -- 0x11a: mov rcx, rax
  , 0x48, 0x89, 0xc5                          -- 0x11d: mov rbp, rax   [sq_loop]
  , 0x48, 0x83, 0xe5, 0x01                    -- 0x120: and rbp, 1
  , 0x74, 0x09                                -- 0x124: jz  0x40012f
  , 0x48, 0x01, 0xcb                          -- 0x126: add rbx, rcx
  , 0x0f, 0x82, 0x58, 0x00, 0x00, 0x00        -- 0x129: jc  0x400187
  , 0x48, 0xc1, 0xe8, 0x01                    -- 0x12f: shr rax, 1     [sq_skip_add]
  , 0x48, 0x85, 0xc0                          -- 0x133: test rax, rax
  , 0x74, 0x0b                                -- 0x136: jz  0x400143
  , 0x48, 0x01, 0xc9                          -- 0x138: add rcx, rcx
  , 0x0f, 0x82, 0x46, 0x00, 0x00, 0x00        -- 0x13b: jc  0x400187
  , 0xeb, 0xda                                -- 0x141: jmp 0x40011d

  -- Stage 7: acc += delta^2, commit to all 3 arenas, normalize scratch, park (0x143..0x187)
  , 0x49, 0x01, 0xdd                          -- 0x143: add r13, rbx   [sq_done]
  , 0x0f, 0x82, 0x3b, 0x00, 0x00, 0x00        -- 0x146: jc  0x400187
  , 0x4d, 0x89, 0x20                          -- 0x14c: mov [r8], r12
  , 0x4d, 0x89, 0x68, 0x08                    -- 0x14f: mov [r8 + 8], r13
  , 0x49, 0x89, 0x76, 0x10                    -- 0x153: mov [r14 + 16], rsi
  , 0x4d, 0x89, 0x66, 0x18                    -- 0x157: mov [r14 + 24], r12
  , 0x4d, 0x89, 0x21                          -- 0x15b: mov [r9], r12
  , 0x4d, 0x89, 0x69, 0x08                    -- 0x15e: mov [r9 + 8], r13
  , 0x49, 0x89, 0x77, 0x10                    -- 0x162: mov [r15 + 16], rsi
  , 0x4d, 0x89, 0x67, 0x18                    -- 0x166: mov [r15 + 24], r12
  , 0x4d, 0x89, 0x22                          -- 0x16a: mov [r10], r12
  , 0x4d, 0x89, 0x6a, 0x08                    -- 0x16d: mov [r10 + 8], r13
  , 0x48, 0x89, 0x72, 0x10                    -- 0x171: mov [rdx + 16], rsi
  , 0x4c, 0x89, 0x62, 0x18                    -- 0x175: mov [rdx + 24], r12
  , 0x4c, 0x89, 0xe8                          -- 0x179: mov rax, r13
  , 0x48, 0x31, 0xdb                          -- 0x17c: xor rbx, rbx
  , 0x48, 0x31, 0xc9                          -- 0x17f: xor rcx, rcx
  , 0x48, 0x31, 0xed                          -- 0x182: xor rbp, rbp
  , 0xeb, 0xfe                                -- 0x185: jmp 0x400185   [park]

  -- Overflow / invalid slot exit(2) (0x187..0x193)
  , 0xbf, 0x02, 0x00, 0x00, 0x00              -- 0x187: mov edi, 2
  , 0xb8, 0x3c, 0x00, 0x00, 0x00              -- 0x18c: mov eax, 60
  , 0x0f, 0x05                                -- 0x191: syscall

  -- 3-way split exit(3) (0x193..0x19f)
  , 0xbf, 0x03, 0x00, 0x00, 0x00              -- 0x193: mov edi, 3
  , 0xb8, 0x3c, 0x00, 0x00, 0x00              -- 0x198: mov eax, 60
  , 0x0f, 0x05                                -- 0x19d: syscall
  ]

theorem memExposureKernelCode_length : memExposureKernelCode.length = 415 := by
  decide +kernel

def memExposureIm : Nat → Option (X86Int.Ins × Nat) :=
  X86Dec.imageOf memExposureKernelCode memExposureBase

def memExposureAlts : X86Int.Alts := fun a =>
  if a < memExposureBase then []
  else match memExposureIm a with
    | some (_, len) => X86Dec.altsAt (fun i => memExposureKernelCode[a - memExposureBase + i]?) a len
    | none => []

theorem memExposureIm_decode_stage0 :
    memExposureIm (memExposureBase + 0x000) = some (.cmpI 7 255, 7) ∧
    memExposureIm (memExposureBase + 0x007) = some (.jcc 3 memExposureOverflow, 6) ∧
    memExposureIm (memExposureBase + 0x00d) = some (.movI 8 arena0, 6) ∧
    memExposureIm (memExposureBase + 0x013) = some (.movI 9 arena1, 6) ∧
    memExposureIm (memExposureBase + 0x019) = some (.movI 10 arena2, 6) ∧
    memExposureIm (memExposureBase + 0x01f) = some (.mov 11 7, 3) ∧
    memExposureIm (memExposureBase + 0x022) = some (.shlI 11 4, 4) ∧
    memExposureIm (memExposureBase + 0x026) = some (.mov 14 8, 3) ∧
    memExposureIm (memExposureBase + 0x029) = some (.alu .add 14 11, 3) ∧
    memExposureIm (memExposureBase + 0x02c) = some (.mov 15 9, 3) ∧
    memExposureIm (memExposureBase + 0x02f) = some (.alu .add 15 11, 3) ∧
    memExposureIm (memExposureBase + 0x032) = some (.mov 2 10, 3) ∧
    memExposureIm (memExposureBase + 0x035) = some (.alu .add 2 11, 3) := by
  decide +kernel

theorem memExposureIm_decode_stage1 :
    memExposureIm (memExposureBase + 0x038) = some (.ld 0 14 16, 4) ∧
    memExposureIm (memExposureBase + 0x03c) = some (.ld 3 15 16, 4) ∧
    memExposureIm (memExposureBase + 0x040) = some (.ld 1 2 16, 4) ∧
    memExposureIm (memExposureBase + 0x044) = some (.cmp 0 3, 3) ∧
    memExposureIm (memExposureBase + 0x047) = some (.jcc 4 (memExposureBase + 0x05b), 2) ∧
    memExposureIm (memExposureBase + 0x049) = some (.cmp 0 1, 3) ∧
    memExposureIm (memExposureBase + 0x04c) = some (.jcc 4 (memExposureBase + 0x05b), 2) ∧
    memExposureIm (memExposureBase + 0x04e) = some (.cmp 3 1, 3) ∧
    memExposureIm (memExposureBase + 0x051) = some (.jcc 4 (memExposureBase + 0x058), 2) ∧
    memExposureIm (memExposureBase + 0x053) = some (.jmp memExposureRefuse, 5) ∧
    memExposureIm (memExposureBase + 0x058) = some (.mov 0 3, 3) ∧
    memExposureIm (memExposureBase + 0x05b) = some (.st 14 16 0, 4) ∧
    memExposureIm (memExposureBase + 0x05f) = some (.st 15 16 0, 4) ∧
    memExposureIm (memExposureBase + 0x063) = some (.st 2 16 0, 4) := by
  decide +kernel

theorem memExposureIm_decode_stage2 :
    memExposureIm (memExposureBase + 0x067) = some (.ld 0 14 24, 4) ∧
    memExposureIm (memExposureBase + 0x06b) = some (.ld 3 15 24, 4) ∧
    memExposureIm (memExposureBase + 0x06f) = some (.ld 1 2 24, 4) ∧
    memExposureIm (memExposureBase + 0x073) = some (.cmp 0 3, 3) ∧
    memExposureIm (memExposureBase + 0x076) = some (.jcc 4 (memExposureBase + 0x08a), 2) ∧
    memExposureIm (memExposureBase + 0x078) = some (.cmp 0 1, 3) ∧
    memExposureIm (memExposureBase + 0x07b) = some (.jcc 4 (memExposureBase + 0x08a), 2) ∧
    memExposureIm (memExposureBase + 0x07d) = some (.cmp 3 1, 3) ∧
    memExposureIm (memExposureBase + 0x080) = some (.jcc 4 (memExposureBase + 0x087), 2) ∧
    memExposureIm (memExposureBase + 0x082) = some (.jmp memExposureRefuse, 5) ∧
    memExposureIm (memExposureBase + 0x087) = some (.mov 0 3, 3) ∧
    memExposureIm (memExposureBase + 0x08a) = some (.st 14 24 0, 4) ∧
    memExposureIm (memExposureBase + 0x08e) = some (.st 15 24 0, 4) ∧
    memExposureIm (memExposureBase + 0x092) = some (.st 2 24 0, 4) ∧
    memExposureIm (memExposureBase + 0x096) = some (.mov 11 0, 3) := by
  decide +kernel

theorem memExposureIm_decode_stage3 :
    memExposureIm (memExposureBase + 0x099) = some (.ld 0 8 0, 3) ∧
    memExposureIm (memExposureBase + 0x09c) = some (.ld 3 9 0, 3) ∧
    memExposureIm (memExposureBase + 0x09f) = some (.ld 1 10 0, 3) ∧
    memExposureIm (memExposureBase + 0x0a2) = some (.cmp 0 3, 3) ∧
    memExposureIm (memExposureBase + 0x0a5) = some (.jcc 4 (memExposureBase + 0x0b9), 2) ∧
    memExposureIm (memExposureBase + 0x0a7) = some (.cmp 0 1, 3) ∧
    memExposureIm (memExposureBase + 0x0aa) = some (.jcc 4 (memExposureBase + 0x0b9), 2) ∧
    memExposureIm (memExposureBase + 0x0ac) = some (.cmp 3 1, 3) ∧
    memExposureIm (memExposureBase + 0x0af) = some (.jcc 4 (memExposureBase + 0x0b6), 2) ∧
    memExposureIm (memExposureBase + 0x0b1) = some (.jmp memExposureRefuse, 5) ∧
    memExposureIm (memExposureBase + 0x0b6) = some (.mov 0 3, 3) ∧
    memExposureIm (memExposureBase + 0x0b9) = some (.st 8 0 0, 3) ∧
    memExposureIm (memExposureBase + 0x0bc) = some (.st 9 0 0, 3) ∧
    memExposureIm (memExposureBase + 0x0bf) = some (.st 10 0 0, 3) ∧
    memExposureIm (memExposureBase + 0x0c2) = some (.mov 12 0, 3) := by
  decide +kernel

theorem memExposureIm_decode_stage4 :
    memExposureIm (memExposureBase + 0x0c5) = some (.ld 0 8 8, 4) ∧
    memExposureIm (memExposureBase + 0x0c9) = some (.ld 3 9 8, 4) ∧
    memExposureIm (memExposureBase + 0x0cd) = some (.ld 1 10 8, 4) ∧
    memExposureIm (memExposureBase + 0x0d1) = some (.cmp 0 3, 3) ∧
    memExposureIm (memExposureBase + 0x0d4) = some (.jcc 4 (memExposureBase + 0x0e8), 2) ∧
    memExposureIm (memExposureBase + 0x0d6) = some (.cmp 0 1, 3) ∧
    memExposureIm (memExposureBase + 0x0d9) = some (.jcc 4 (memExposureBase + 0x0e8), 2) ∧
    memExposureIm (memExposureBase + 0x0db) = some (.cmp 3 1, 3) ∧
    memExposureIm (memExposureBase + 0x0de) = some (.jcc 4 (memExposureBase + 0x0e5), 2) ∧
    memExposureIm (memExposureBase + 0x0e0) = some (.jmp memExposureRefuse, 5) ∧
    memExposureIm (memExposureBase + 0x0e5) = some (.mov 0 3, 3) ∧
    memExposureIm (memExposureBase + 0x0e8) = some (.st 8 8 0, 4) ∧
    memExposureIm (memExposureBase + 0x0ec) = some (.st 9 8 0, 4) ∧
    memExposureIm (memExposureBase + 0x0f0) = some (.st 10 8 0, 4) ∧
    memExposureIm (memExposureBase + 0x0f4) = some (.mov 13 0, 3) := by
  decide +kernel

theorem memExposureIm_decode_stage5 :
    memExposureIm (memExposureBase + 0x0f7) = some (.mov 0 12, 3) ∧
    memExposureIm (memExposureBase + 0x0fa) = some (.alu .sub 0 11, 3) ∧
    memExposureIm (memExposureBase + 0x0fd) = some (.jcc 2 memExposureOverflow, 6) ∧
    memExposureIm (memExposureBase + 0x103) = some (.aluI .add 0 1, 4) ∧
    memExposureIm (memExposureBase + 0x107) = some (.jcc 2 memExposureOverflow, 6) ∧
    memExposureIm (memExposureBase + 0x10d) = some (.aluI .add 12 1, 4) ∧
    memExposureIm (memExposureBase + 0x111) = some (.jcc 2 memExposureOverflow, 6) := by
  decide +kernel

theorem memExposureIm_decode_stage6 :
    memExposureIm (memExposureBase + 0x117) = some (.alu .xor 3 3, 3) ∧
    memExposureIm (memExposureBase + 0x11a) = some (.mov 1 0, 3) ∧
    memExposureIm (memExposureBase + 0x11d) = some (.mov 5 0, 3) ∧
    memExposureIm (memExposureBase + 0x120) = some (.aluI .and 5 1, 4) ∧
    memExposureIm (memExposureBase + 0x124) = some (.jcc 4 (memExposureBase + 0x12f), 2) ∧
    memExposureIm (memExposureBase + 0x126) = some (.alu .add 3 1, 3) ∧
    memExposureIm (memExposureBase + 0x129) = some (.jcc 2 memExposureOverflow, 6) ∧
    memExposureIm (memExposureBase + 0x12f) = some (.shrI 0 1, 4) ∧
    memExposureIm (memExposureBase + 0x133) = some (.test 0 0, 3) ∧
    memExposureIm (memExposureBase + 0x136) = some (.jcc 4 (memExposureBase + 0x143), 2) ∧
    memExposureIm (memExposureBase + 0x138) = some (.alu .add 1 1, 3) ∧
    memExposureIm (memExposureBase + 0x13b) = some (.jcc 2 memExposureOverflow, 6) ∧
    memExposureIm (memExposureBase + 0x141) = some (.jmp (memExposureBase + 0x11d), 2) := by
  decide +kernel

theorem memExposureIm_decode_stage7 :
    memExposureIm (memExposureBase + 0x143) = some (.alu .add 13 3, 3) ∧
    memExposureIm (memExposureBase + 0x146) = some (.jcc 2 memExposureOverflow, 6) ∧
    memExposureIm (memExposureBase + 0x14c) = some (.st 8 0 12, 3) ∧
    memExposureIm (memExposureBase + 0x14f) = some (.st 8 8 13, 4) ∧
    memExposureIm (memExposureBase + 0x153) = some (.st 14 16 6, 4) ∧
    memExposureIm (memExposureBase + 0x157) = some (.st 14 24 12, 4) ∧
    memExposureIm (memExposureBase + 0x15b) = some (.st 9 0 12, 3) ∧
    memExposureIm (memExposureBase + 0x15e) = some (.st 9 8 13, 4) ∧
    memExposureIm (memExposureBase + 0x162) = some (.st 15 16 6, 4) ∧
    memExposureIm (memExposureBase + 0x166) = some (.st 15 24 12, 4) ∧
    memExposureIm (memExposureBase + 0x16a) = some (.st 10 0 12, 3) ∧
    memExposureIm (memExposureBase + 0x16d) = some (.st 10 8 13, 4) ∧
    memExposureIm (memExposureBase + 0x171) = some (.st 2 16 6, 4) ∧
    memExposureIm (memExposureBase + 0x175) = some (.st 2 24 12, 4) := by
  decide +kernel

theorem memExposureIm_decode_exits :
    memExposureIm (memExposureBase + 0x179) = some (.mov 0 13, 3) ∧
    memExposureIm (memExposureBase + 0x17c) = some (.alu .xor 3 3, 3) ∧
    memExposureIm (memExposureBase + 0x17f) = some (.alu .xor 1 1, 3) ∧
    memExposureIm (memExposureBase + 0x182) = some (.alu .xor 5 5, 3) ∧
    memExposureIm memExposurePark = some (.jmp memExposurePark, 2) ∧
    memExposureIm memExposureOverflow = some (.movI 7 2, 5) ∧
    memExposureIm (memExposureOverflow + 5) = some (.movI 0 60, 5) ∧
    memExposureIm (memExposureOverflow + 10) = some (.syscall, 2) ∧
    memExposureIm memExposureRefuse = some (.movI 7 3, 5) ∧
    memExposureIm (memExposureRefuse + 5) = some (.movI 0 60, 5) ∧
    memExposureIm (memExposureRefuse + 10) = some (.syscall, 2) := by
  decide +kernel

theorem memExposureIm_decode :
    (memExposureIm (memExposureBase + 0x000) = some (.cmpI 7 255, 7) ∧
     memExposureIm (memExposureBase + 0x007) = some (.jcc 3 memExposureOverflow, 6) ∧
     memExposureIm (memExposureBase + 0x00d) = some (.movI 8 arena0, 6) ∧
     memExposureIm (memExposureBase + 0x013) = some (.movI 9 arena1, 6) ∧
     memExposureIm (memExposureBase + 0x019) = some (.movI 10 arena2, 6) ∧
     memExposureIm (memExposureBase + 0x01f) = some (.mov 11 7, 3) ∧
     memExposureIm (memExposureBase + 0x022) = some (.shlI 11 4, 4) ∧
     memExposureIm (memExposureBase + 0x026) = some (.mov 14 8, 3) ∧
     memExposureIm (memExposureBase + 0x029) = some (.alu .add 14 11, 3) ∧
     memExposureIm (memExposureBase + 0x02c) = some (.mov 15 9, 3) ∧
     memExposureIm (memExposureBase + 0x02f) = some (.alu .add 15 11, 3) ∧
     memExposureIm (memExposureBase + 0x032) = some (.mov 2 10, 3) ∧
     memExposureIm (memExposureBase + 0x035) = some (.alu .add 2 11, 3)) ∧
    (memExposureIm (memExposureBase + 0x038) = some (.ld 0 14 16, 4) ∧
     memExposureIm (memExposureBase + 0x03c) = some (.ld 3 15 16, 4) ∧
     memExposureIm (memExposureBase + 0x040) = some (.ld 1 2 16, 4) ∧
     memExposureIm (memExposureBase + 0x044) = some (.cmp 0 3, 3) ∧
     memExposureIm (memExposureBase + 0x047) = some (.jcc 4 (memExposureBase + 0x05b), 2) ∧
     memExposureIm (memExposureBase + 0x049) = some (.cmp 0 1, 3) ∧
     memExposureIm (memExposureBase + 0x04c) = some (.jcc 4 (memExposureBase + 0x05b), 2) ∧
     memExposureIm (memExposureBase + 0x04e) = some (.cmp 3 1, 3) ∧
     memExposureIm (memExposureBase + 0x051) = some (.jcc 4 (memExposureBase + 0x058), 2) ∧
     memExposureIm (memExposureBase + 0x053) = some (.jmp memExposureRefuse, 5) ∧
     memExposureIm (memExposureBase + 0x058) = some (.mov 0 3, 3) ∧
     memExposureIm (memExposureBase + 0x05b) = some (.st 14 16 0, 4) ∧
     memExposureIm (memExposureBase + 0x05f) = some (.st 15 16 0, 4) ∧
     memExposureIm (memExposureBase + 0x063) = some (.st 2 16 0, 4)) ∧
    (memExposureIm (memExposureBase + 0x067) = some (.ld 0 14 24, 4) ∧
     memExposureIm (memExposureBase + 0x06b) = some (.ld 3 15 24, 4) ∧
     memExposureIm (memExposureBase + 0x06f) = some (.ld 1 2 24, 4) ∧
     memExposureIm (memExposureBase + 0x073) = some (.cmp 0 3, 3) ∧
     memExposureIm (memExposureBase + 0x076) = some (.jcc 4 (memExposureBase + 0x08a), 2) ∧
     memExposureIm (memExposureBase + 0x078) = some (.cmp 0 1, 3) ∧
     memExposureIm (memExposureBase + 0x07b) = some (.jcc 4 (memExposureBase + 0x08a), 2) ∧
     memExposureIm (memExposureBase + 0x07d) = some (.cmp 3 1, 3) ∧
     memExposureIm (memExposureBase + 0x080) = some (.jcc 4 (memExposureBase + 0x087), 2) ∧
     memExposureIm (memExposureBase + 0x082) = some (.jmp memExposureRefuse, 5) ∧
     memExposureIm (memExposureBase + 0x087) = some (.mov 0 3, 3) ∧
     memExposureIm (memExposureBase + 0x08a) = some (.st 14 24 0, 4) ∧
     memExposureIm (memExposureBase + 0x08e) = some (.st 15 24 0, 4) ∧
     memExposureIm (memExposureBase + 0x092) = some (.st 2 24 0, 4) ∧
     memExposureIm (memExposureBase + 0x096) = some (.mov 11 0, 3)) ∧
    (memExposureIm (memExposureBase + 0x099) = some (.ld 0 8 0, 3) ∧
     memExposureIm (memExposureBase + 0x09c) = some (.ld 3 9 0, 3) ∧
     memExposureIm (memExposureBase + 0x09f) = some (.ld 1 10 0, 3) ∧
     memExposureIm (memExposureBase + 0x0a2) = some (.cmp 0 3, 3) ∧
     memExposureIm (memExposureBase + 0x0a5) = some (.jcc 4 (memExposureBase + 0x0b9), 2) ∧
     memExposureIm (memExposureBase + 0x0a7) = some (.cmp 0 1, 3) ∧
     memExposureIm (memExposureBase + 0x0aa) = some (.jcc 4 (memExposureBase + 0x0b9), 2) ∧
     memExposureIm (memExposureBase + 0x0ac) = some (.cmp 3 1, 3) ∧
     memExposureIm (memExposureBase + 0x0af) = some (.jcc 4 (memExposureBase + 0x0b6), 2) ∧
     memExposureIm (memExposureBase + 0x0b1) = some (.jmp memExposureRefuse, 5) ∧
     memExposureIm (memExposureBase + 0x0b6) = some (.mov 0 3, 3) ∧
     memExposureIm (memExposureBase + 0x0b9) = some (.st 8 0 0, 3) ∧
     memExposureIm (memExposureBase + 0x0bc) = some (.st 9 0 0, 3) ∧
     memExposureIm (memExposureBase + 0x0bf) = some (.st 10 0 0, 3) ∧
     memExposureIm (memExposureBase + 0x0c2) = some (.mov 12 0, 3)) ∧
    (memExposureIm (memExposureBase + 0x0c5) = some (.ld 0 8 8, 4) ∧
     memExposureIm (memExposureBase + 0x0c9) = some (.ld 3 9 8, 4) ∧
     memExposureIm (memExposureBase + 0x0cd) = some (.ld 1 10 8, 4) ∧
     memExposureIm (memExposureBase + 0x0d1) = some (.cmp 0 3, 3) ∧
     memExposureIm (memExposureBase + 0x0d4) = some (.jcc 4 (memExposureBase + 0x0e8), 2) ∧
     memExposureIm (memExposureBase + 0x0d6) = some (.cmp 0 1, 3) ∧
     memExposureIm (memExposureBase + 0x0d9) = some (.jcc 4 (memExposureBase + 0x0e8), 2) ∧
     memExposureIm (memExposureBase + 0x0db) = some (.cmp 3 1, 3) ∧
     memExposureIm (memExposureBase + 0x0de) = some (.jcc 4 (memExposureBase + 0x0e5), 2) ∧
     memExposureIm (memExposureBase + 0x0e0) = some (.jmp memExposureRefuse, 5) ∧
     memExposureIm (memExposureBase + 0x0e5) = some (.mov 0 3, 3) ∧
     memExposureIm (memExposureBase + 0x0e8) = some (.st 8 8 0, 4) ∧
     memExposureIm (memExposureBase + 0x0ec) = some (.st 9 8 0, 4) ∧
     memExposureIm (memExposureBase + 0x0f0) = some (.st 10 8 0, 4) ∧
     memExposureIm (memExposureBase + 0x0f4) = some (.mov 13 0, 3)) ∧
    (memExposureIm (memExposureBase + 0x0f7) = some (.mov 0 12, 3) ∧
     memExposureIm (memExposureBase + 0x0fa) = some (.alu .sub 0 11, 3) ∧
     memExposureIm (memExposureBase + 0x0fd) = some (.jcc 2 memExposureOverflow, 6) ∧
     memExposureIm (memExposureBase + 0x103) = some (.aluI .add 0 1, 4) ∧
     memExposureIm (memExposureBase + 0x107) = some (.jcc 2 memExposureOverflow, 6) ∧
     memExposureIm (memExposureBase + 0x10d) = some (.aluI .add 12 1, 4) ∧
     memExposureIm (memExposureBase + 0x111) = some (.jcc 2 memExposureOverflow, 6)) ∧
    (memExposureIm (memExposureBase + 0x117) = some (.alu .xor 3 3, 3) ∧
     memExposureIm (memExposureBase + 0x11a) = some (.mov 1 0, 3) ∧
     memExposureIm (memExposureBase + 0x11d) = some (.mov 5 0, 3) ∧
     memExposureIm (memExposureBase + 0x120) = some (.aluI .and 5 1, 4) ∧
     memExposureIm (memExposureBase + 0x124) = some (.jcc 4 (memExposureBase + 0x12f), 2) ∧
     memExposureIm (memExposureBase + 0x126) = some (.alu .add 3 1, 3) ∧
     memExposureIm (memExposureBase + 0x129) = some (.jcc 2 memExposureOverflow, 6) ∧
     memExposureIm (memExposureBase + 0x12f) = some (.shrI 0 1, 4) ∧
     memExposureIm (memExposureBase + 0x133) = some (.test 0 0, 3) ∧
     memExposureIm (memExposureBase + 0x136) = some (.jcc 4 (memExposureBase + 0x143), 2) ∧
     memExposureIm (memExposureBase + 0x138) = some (.alu .add 1 1, 3) ∧
     memExposureIm (memExposureBase + 0x13b) = some (.jcc 2 memExposureOverflow, 6) ∧
     memExposureIm (memExposureBase + 0x141) = some (.jmp (memExposureBase + 0x11d), 2)) ∧
    (memExposureIm (memExposureBase + 0x143) = some (.alu .add 13 3, 3) ∧
     memExposureIm (memExposureBase + 0x146) = some (.jcc 2 memExposureOverflow, 6) ∧
     memExposureIm (memExposureBase + 0x14c) = some (.st 8 0 12, 3) ∧
     memExposureIm (memExposureBase + 0x14f) = some (.st 8 8 13, 4) ∧
     memExposureIm (memExposureBase + 0x153) = some (.st 14 16 6, 4) ∧
     memExposureIm (memExposureBase + 0x157) = some (.st 14 24 12, 4) ∧
     memExposureIm (memExposureBase + 0x15b) = some (.st 9 0 12, 3) ∧
     memExposureIm (memExposureBase + 0x15e) = some (.st 9 8 13, 4) ∧
     memExposureIm (memExposureBase + 0x162) = some (.st 15 16 6, 4) ∧
     memExposureIm (memExposureBase + 0x166) = some (.st 15 24 12, 4) ∧
     memExposureIm (memExposureBase + 0x16a) = some (.st 10 0 12, 3) ∧
     memExposureIm (memExposureBase + 0x16d) = some (.st 10 8 13, 4) ∧
     memExposureIm (memExposureBase + 0x171) = some (.st 2 16 6, 4) ∧
     memExposureIm (memExposureBase + 0x175) = some (.st 2 24 12, 4)) ∧
    (memExposureIm (memExposureBase + 0x179) = some (.mov 0 13, 3) ∧
     memExposureIm (memExposureBase + 0x17c) = some (.alu .xor 3 3, 3) ∧
     memExposureIm (memExposureBase + 0x17f) = some (.alu .xor 1 1, 3) ∧
     memExposureIm (memExposureBase + 0x182) = some (.alu .xor 5 5, 3) ∧
     memExposureIm memExposurePark = some (.jmp memExposurePark, 2) ∧
     memExposureIm memExposureOverflow = some (.movI 7 2, 5) ∧
     memExposureIm (memExposureOverflow + 5) = some (.movI 0 60, 5) ∧
     memExposureIm (memExposureOverflow + 10) = some (.syscall, 2) ∧
     memExposureIm memExposureRefuse = some (.movI 7 3, 5) ∧
     memExposureIm (memExposureRefuse + 5) = some (.movI 0 60, 5) ∧
     memExposureIm (memExposureRefuse + 10) = some (.syscall, 2)) :=
  ⟨memExposureIm_decode_stage0,
   memExposureIm_decode_stage1,
   memExposureIm_decode_stage2,
   memExposureIm_decode_stage3,
   memExposureIm_decode_stage4,
   memExposureIm_decode_stage5,
   memExposureIm_decode_stage6,
   memExposureIm_decode_stage7,
   memExposureIm_decode_exits⟩

theorem memExposureAlts_len_000 : (memExposureAlts (memExposureBase + 0x000)).length = 56 := by
  decide +kernel

theorem memExposureAlts_len_park : (memExposureAlts memExposurePark).length = 16 := by
  decide +kernel

/-! ### Architectural verification (`X86Int.run`) -/

def mkRegs (r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 : Option Nat) :
    List (Option Nat) :=
  [r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15]

def memExposureInit (s : X86Int.St) (w v_new : Nat) (d0 d1 d2 : List Nat) : X86Int.St :=
  { regs := mkRegs
      (some ((s.reg 0).getD 0))
      (some ((s.reg 1).getD 0))
      (some ((s.reg 2).getD 0))
      (some ((s.reg 3).getD 0))
      (s.reg 4)
      (some ((s.reg 5).getD 0))
      (some v_new)
      (some w)
      (some ((s.reg 8).getD 0))
      (some ((s.reg 9).getD 0))
      (some ((s.reg 10).getD 0))
      (s.reg 11)
      (s.reg 12)
      (s.reg 13)
      (s.reg 14)
      (s.reg 15)
    rip := memExposureBase
    fl := X86Int.Flags.undef
    mem := arenaMem d0 d1 d2
    k := s.k }

def memExposureDone (s : X86Int.St) (w v_new step_v acc_v last_v : Nat)
    (d0 d1 d2 : List Nat) : X86Int.St :=
  let delta := step_v - last_v + 1
  let step' := step_v + 1
  let acc' := acc_v + delta * delta
  { regs := mkRegs
      (some acc')
      (some 0)
      (some (arena2 + 16 * w))
      (some 0)
      (s.reg 4)
      (some 0)
      (some v_new)
      (some w)
      (some arena0)
      (some arena1)
      (some arena2)
      (some last_v)
      (some step')
      (some acc')
      (some (arena0 + 16 * w))
      (some (arena1 + 16 * w))
    rip := memExposurePark
    fl := TMRVoter.voterFlagsEq
    mem := arenaMem
      (setArenaState d0 w step' acc' v_new step')
      (setArenaState d1 w step' acc' v_new step')
      (setArenaState d2 w step' acc' v_new step')
    k := s.k }

theorem memExposureInit_regs_length (s : X86Int.St) (w v_new : Nat) (d0 d1 d2 : List Nat) :
    (memExposureInit s w v_new d0 d1 d2).regs.length = 16 := rfl

theorem memExposureDone_regs_length (s : X86Int.St) (w v_new step_v acc_v last_v : Nat)
    (d0 d1 d2 : List Nat) :
    (memExposureDone s w v_new step_v acc_v last_v d0 d1 d2).regs.length = 16 := rfl

private theorem run_exit (im : Nat → Option (X86Int.Ins × Nat)) (c : Nat) :
    ∀ n, X86Int.run im n (.exit c) = .exit c
  | 0 => rfl
  | _ + 1 => rfl

private theorem run_stuck (im : Nat → Option (X86Int.Ins × Nat)) :
    ∀ n, X86Int.run im n .stuck = .stuck
  | 0 => rfl
  | _ + 1 => rfl

private theorem run_add (im : Nat → Option (X86Int.Ins × Nat)) :
    ∀ m n o, X86Int.run im (m + n) o = X86Int.run im n (X86Int.run im m o)
  | 0, n, o => by rw [Nat.zero_add]; rfl
  | m + 1, n, .ok s => by
    rw [Nat.add_right_comm]
    simp only [X86Int.run, run_add im m n (X86Int.step im s)]
  | _ + 1, n, .exit c => by simp [X86Int.run, run_exit]
  | _ + 1, n, .stuck => by simp [X86Int.run, run_stuck]

private theorem run_park (s : X86Int.St) (hs : s.rip = memExposurePark) :
    ∀ k, X86Int.run memExposureIm k (.ok s) = .ok s
  | 0 => rfl
  | k + 1 => by
    cases s with
    | mk regs rip fl mem k0 =>
      simp only at hs
      subst hs
      have hstep : X86Int.step memExposureIm ⟨regs, memExposurePark, fl, mem, k0⟩ =
          .ok ⟨regs, memExposurePark, fl, mem, k0⟩ := by
        simp [X86Int.step, memExposureIm_decode_exits.2.2.2.2.1, X86Int.exec]
      simp only [X86Int.run, hstep]
      exact run_park ⟨regs, memExposurePark, fl, mem, k0⟩ rfl k

private theorem run_to_fuel_park {N k : Nat} (hk : k ≤ N) {o : X86Int.Out} {s : X86Int.St}
    (hs : s.rip = memExposurePark) (hrun : X86Int.run memExposureIm k o = .ok s) :
    X86Int.run memExposureIm N o = .ok s := by
  have hN : N = k + (N - k) := by omega
  rw [hN, run_add, hrun, run_park s hs (N - k)]

private theorem run_to_fuel_exit {N k c : Nat} (hk : k ≤ N) {o : X86Int.Out}
    (hrun : X86Int.run memExposureIm k o = .exit c) :
    X86Int.run memExposureIm N o = .exit c := by
  have hN : N = k + (N - k) := by omega
  rw [hN, run_add, hrun, run_exit]

private theorem cond_cf_true {f : Flags} (h : f.cf = some true) : X86Int.cond f 2 = some true := by
  simp [X86Int.cond, h]

private theorem cond_cf_false {f : Flags} (h : f.cf = some false) : X86Int.cond f 2 = some false := by
  simp [X86Int.cond, h]

private theorem cond_ncf_true {f : Flags} (h : f.cf = some true) : X86Int.cond f 3 = some false := by
  simp [X86Int.cond, h]

private theorem cond_ncf_false {f : Flags} (h : f.cf = some false) : X86Int.cond f 3 = some true := by
  simp [X86Int.cond, h]

private theorem cond_zf_true {f : Flags} (h : f.zf = some true) : X86Int.cond f 4 = some true := by
  simp [X86Int.cond, h]

private theorem cond_zf_false {f : Flags} (h : f.zf = some false) : X86Int.cond f 4 = some false := by
  simp [X86Int.cond, h]

private theorem aluF_sub_eq (v : Nat) :
    (X86Int.aluF .sub v v).2 = TMRVoter.voterFlagsEq := by
  simp [X86Int.aluF, X86Int.msb, TMRVoter.voterFlagsEq, Nat.add_sub_cancel_left]

private theorem aluF_sub_zf_ne {x y : Nat} (hx : x < X86Env.W) (hy : y < X86Env.W) (hne : x ≠ y) :
    (X86Int.aluF .sub x y).2.zf = some false := by
  simp only [X86Int.aluF]
  have hpos : 0 < x + X86Env.W - y := by omega
  have hlt : x + X86Env.W - y < 2 * X86Env.W := by omega
  have hneW : x + X86Env.W - y ≠ X86Env.W := by omega
  have hnz : (x + X86Env.W - y) % X86Env.W ≠ 0 := by
    intro hmod
    obtain ⟨k, hk⟩ := Nat.dvd_of_mod_eq_zero hmod
    rcases k with _ | _ | k
    · simp at hk; omega
    · simp at hk; exact hneW hk
    · have hk2 : 2 ≤ k + 2 := by omega
      have hge : 2 * X86Env.W ≤ X86Env.W * (k + 2) := by
        simpa [Nat.mul_comm] using Nat.mul_le_mul_left X86Env.W hk2
      have hkeq : x + X86Env.W - y = X86Env.W * (k + 2) := hk
      omega
  simp [hnz]

private theorem aluF_sub_val {x y : Nat} (hx : x < X86Env.W) (hy : y ≤ x) :
    (X86Int.aluF .sub x y).1 = x - y := by
  simp only [X86Int.aluF]
  have heq : x + X86Env.W - y = (x - y) + X86Env.W := by omega
  rw [heq, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]

private theorem aluF_add_val {x y : Nat} (h : x + y < X86Env.W) :
    (X86Int.aluF .add x y).1 = x + y := by
  simp [X86Int.aluF, Nat.mod_eq_of_lt h]

private theorem aluF_add_cf_false {x y : Nat} (h : x + y < X86Env.W) :
    (X86Int.aluF .add x y).2.cf = some false := by
  simp [X86Int.aluF, show ¬ X86Env.W ≤ x + y from by omega]

private theorem aluF_add_cf_true {x y : Nat} (h : X86Env.W ≤ x + y) :
    (X86Int.aluF .add x y).2.cf = some true := by
  simp [X86Int.aluF, h]

private theorem aluF_xor_self (v : Nat) :
    X86Int.aluF .xor v v = (0, TMRVoter.voterFlagsEq) := by
  simp [X86Int.aluF, X86Int.msb, Nat.xor_self, TMRVoter.voterFlagsEq]

private theorem aluF_and_one_val (m : Nat) :
    (X86Int.aluF .and m 1).1 = m % 2 := by
  simp [X86Int.aluF, Nat.and_one_is_mod]

private theorem aluF_and_one_zf (m : Nat) :
    (X86Int.aluF .and m 1).2.zf = some (m % 2 == 0) := by
  simp [X86Int.aluF, Nat.and_one_is_mod]

private theorem aluF_and_self_zf (m : Nat) :
    (X86Int.aluF .and m m).2.zf = some (m == 0) := by
  simp [X86Int.aluF, Nat.and_self]

private theorem exit2_run {r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 : Nat}
    {fl : Flags} {mem : Mem} {k : Kern} :
    X86Int.run memExposureIm 3
      (.ok ⟨mkRegs (some r0) (some r1) (some r2) (some r3) (some r4) (some r5) (some r6) (some r7)
        (some r8) (some r9) (some r10) (some r11) (some r12) (some r13) (some r14) (some r15),
        memExposureOverflow, fl, mem, k⟩) = .exit 2 := by
  have h1 := memExposureIm_decode_exits.2.2.2.2.2.1
  have h2 := memExposureIm_decode_exits.2.2.2.2.2.2.1
  have h3 := memExposureIm_decode_exits.2.2.2.2.2.2.2.1
  rw [X86Int.run, X86Int.step]; dsimp only; rw [h1]
  simp only [X86Int.exec, St.set, mkRegs, List.set]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [h2]
  simp only [X86Int.exec, St.set, List.set]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [h3]
  simp only [X86Int.exec, St.reg, List.getElem?_cons_zero, List.getElem?_cons_succ,
    Option.getD_some, X86Env.syscall]
  rfl

private theorem exit2_run_r4opt {r0 r2 r6 r7 r8 r9 r10 : Nat}
    {r1 r3 r4 r5 r11 r12 r13 r14 r15 : Option Nat} {fl : Flags} {mem : Mem} {k : Kern} :
    X86Int.run memExposureIm 3
      (.ok ⟨mkRegs (some r0) r1 (some r2) r3 r4 r5 (some r6) (some r7)
        (some r8) (some r9) (some r10) r11 r12 r13 r14 r15,
        memExposureOverflow, fl, mem, k⟩) = .exit 2 := by
  have h1 := memExposureIm_decode_exits.2.2.2.2.2.1
  have h2 := memExposureIm_decode_exits.2.2.2.2.2.2.1
  have h3 := memExposureIm_decode_exits.2.2.2.2.2.2.2.1
  rw [X86Int.run, X86Int.step]; dsimp only; rw [h1]
  simp only [X86Int.exec, St.set, mkRegs, List.set]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [h2]
  simp only [X86Int.exec, St.set, List.set]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [h3]
  simp only [X86Int.exec, St.reg, List.getElem?_cons_zero, List.getElem?_cons_succ,
    Option.getD_some, X86Env.syscall]
  rfl

private theorem exit3_run_r4opt {r0 r1 r2 r3 r5 r6 r7 r8 r9 r10 : Nat}
    {r4 r11 r12 r13 r14 r15 : Option Nat} {fl : Flags} {mem : Mem} {k : Kern} :
    X86Int.run memExposureIm 3
      (.ok ⟨mkRegs (some r0) (some r1) (some r2) (some r3) r4 (some r5) (some r6) (some r7)
        (some r8) (some r9) (some r10) r11 r12 r13 r14 r15,
        memExposureRefuse, fl, mem, k⟩) = .exit 3 := by
  have h1 := memExposureIm_decode_exits.2.2.2.2.2.2.2.2.1
  have h2 := memExposureIm_decode_exits.2.2.2.2.2.2.2.2.2.1
  have h3 := memExposureIm_decode_exits.2.2.2.2.2.2.2.2.2.2
  rw [X86Int.run, X86Int.step]; dsimp only; rw [h1]
  simp only [X86Int.exec, St.set, mkRegs, List.set]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [h2]
  simp only [X86Int.exec, St.set, List.set]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [h3]
  simp only [X86Int.exec, St.reg, List.getElem?_cons_zero, List.getElem?_cons_succ,
    Option.getD_some, X86Env.syscall]
  rfl

private theorem stage0_ok (s : X86Int.St) {w v_new : Nat} (hw : w < maxSlots)
    (d0 d1 d2 : List Nat) :
    X86Int.run memExposureIm 13 (.ok (memExposureInit s w v_new d0 d1 d2)) =
      .ok ⟨mkRegs
          (some ((s.reg 0).getD 0))
          (some ((s.reg 1).getD 0))
          (some (arena2 + 16 * w))
          (some ((s.reg 3).getD 0))
          (s.reg 4)
          (some ((s.reg 5).getD 0))
          (some v_new)
          (some w)
          (some arena0)
          (some arena1)
          (some arena2)
          (some (16 * w))
          (s.reg 12)
          (s.reg 13)
          (some (arena0 + 16 * w))
          (some (arena1 + 16 * w)),
        memExposureBase + 0x038,
        (X86Int.aluF .add arena2 (16 * w)).2,
        arenaMem d0 d1 d2,
        s.k⟩ := by
  have hcf : (X86Int.aluF .sub w (255 % X86Env.W)).2.cf = some true := by
    simp only [X86Int.aluF, maxSlots] at *
    have hlt : w < 255 % X86Env.W := by
      have : 255 % X86Env.W = 255 := by decide
      omega
    simp [hlt]
  have hwshl : w * 2 ^ (4 % 64) % X86Env.W = 16 * w := by
    have hpow : 2 ^ (4 % 64) = 16 := by decide
    have hlt : w * 16 < X86Env.W := by simp [maxSlots, X86Env.W] at *; omega
    rw [hpow, Nat.mod_eq_of_lt hlt, Nat.mul_comm]
  have ha0 : (X86Int.aluF .add arena0 (16 * w)).1 = arena0 + 16 * w :=
    aluF_add_val (by simp [arena0, maxSlots, X86Env.W] at *; omega)
  have ha1 : (X86Int.aluF .add arena1 (16 * w)).1 = arena1 + 16 * w :=
    aluF_add_val (by simp [arena1, maxSlots, X86Env.W] at *; omega)
  have ha2 : (X86Int.aluF .add arena2 (16 * w)).1 = arena2 + 16 * w :=
    aluF_add_val (by simp [arena2, maxSlots, X86Env.W] at *; omega)
  have hm0 : arena0 % X86Env.W = arena0 := by decide
  have hm1 : arena1 % X86Env.W = arena1 := by decide
  have hm2 : arena2 % X86Env.W = arena2 := by decide
  have hd := memExposureIm_decode_stage0
  have hd0 : memExposureIm memExposureBase = some (.cmpI 7 255, 7) := by
    simpa only [Nat.add_zero] using hd.1
  rw [X86Int.run, X86Int.step]; dsimp only [memExposureInit]; rw [hd0]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, mkRegs,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd.2.1]
  simp only [X86Int.exec, cond_ncf_true hcf, lift, bind, Option.bind, pure, Bool.false_eq_true,
    ↓reduceIte]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd.2.2.1]
  simp only [X86Int.exec, St.set, mkRegs, List.set, hm0]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd.2.2.2.1]
  simp only [X86Int.exec, St.set, List.set, hm1]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd.2.2.2.2.1]
  simp only [X86Int.exec, St.set, List.set, hm2]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd.2.2.2.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd.2.2.2.2.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hwshl]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd.2.2.2.2.2.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd.2.2.2.2.2.2.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, ha0]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd.2.2.2.2.2.2.2.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd.2.2.2.2.2.2.2.2.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, ha1]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd.2.2.2.2.2.2.2.2.2.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd.2.2.2.2.2.2.2.2.2.2.2.2]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, ha2]
  rfl

private theorem stage0_bad_slot (s : X86Int.St) {w v_new : Nat} (hw : maxSlots ≤ w)
    (d0 d1 d2 : List Nat) :
    X86Int.run memExposureIm 5 (.ok (memExposureInit s w v_new d0 d1 d2)) = .exit 2 := by
  have hcf : (X86Int.aluF .sub w (255 % X86Env.W)).2.cf = some false := by
    simp only [X86Int.aluF, maxSlots] at *
    have hge : ¬ (w < 255 % X86Env.W) := by
      have : 255 % X86Env.W = 255 := by decide
      omega
    simp [hge]
  have hd := memExposureIm_decode_stage0
  have hd0 : memExposureIm memExposureBase = some (.cmpI 7 255, 7) := by
    simpa only [Nat.add_zero] using hd.1
  rw [show (5 : Nat) = 2 + 3 from rfl, run_add]
  have h2 : X86Int.run memExposureIm 2 (.ok (memExposureInit s w v_new d0 d1 d2)) =
      .ok ⟨mkRegs (some ((s.reg 0).getD 0)) (some ((s.reg 1).getD 0))
          (some ((s.reg 2).getD 0)) (some ((s.reg 3).getD 0)) (s.reg 4)
          (some ((s.reg 5).getD 0)) (some v_new) (some w)
          (some ((s.reg 8).getD 0)) (some ((s.reg 9).getD 0)) (some ((s.reg 10).getD 0))
          (s.reg 11) (s.reg 12) (s.reg 13) (s.reg 14) (s.reg 15),
        memExposureOverflow, (X86Int.aluF .sub w (255 % X86Env.W)).2, arenaMem d0 d1 d2, s.k⟩ := by
    rw [X86Int.run, X86Int.step]; dsimp only [memExposureInit]; rw [hd0]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, mkRegs,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hd.2.1]
    simp only [X86Int.exec, cond_ncf_false hcf, lift, bind, Option.bind, pure, ↓reduceIte]
    rfl
  rw [h2]
  exact exit2_run_r4opt

private theorem voteGadget_majority
    {G : Nat}
    (h00 : memExposureIm G = some (.cmp 0 3, 3))
    (h03 : memExposureIm (G + 0x03) = some (.jcc 4 (G + 0x17), 2))
    (h05 : memExposureIm (G + 0x05) = some (.cmp 0 1, 3))
    (h08 : memExposureIm (G + 0x08) = some (.jcc 4 (G + 0x17), 2))
    (h0a : memExposureIm (G + 0x0a) = some (.cmp 3 1, 3))
    (h0d : memExposureIm (G + 0x0d) = some (.jcc 4 (G + 0x14), 2))
    (h14 : memExposureIm (G + 0x14) = some (.mov 0 3, 3))
    {r2 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 : Option Nat}
    {fl : Flags} {mem : Mem} {k : Kern}
    {a b c v : Nat} (ha : a < X86Env.W) (hb : b < X86Env.W) (hc : c < X86Env.W)
    (hv : TMR.vote a b c = some v) :
    ∃ n ≤ 7, ∃ b' fl',
      X86Int.run memExposureIm n
        (.ok ⟨mkRegs (some a) (some c) r2 (some b) r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15, G, fl, mem, k⟩) =
        .ok ⟨mkRegs (some v) (some c) r2 (some b') r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15, G + 0x17, fl', mem, k⟩ := by
  by_cases hab : a = b
  · subst a
    have hveq : v = b := by unfold TMR.vote at hv; simp at hv; exact hv.symm
    subst v
    refine ⟨2, by decide, b, TMRVoter.voterFlagsEq, ?_⟩
    have hzf_true : TMRVoter.voterFlagsEq.zf = some true := rfl
    rw [X86Int.run, X86Int.step]; dsimp only; rw [h00]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, mkRegs,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, aluF_sub_eq]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [h03]
    simp only [X86Int.exec, cond_zf_true hzf_true, lift, bind, Option.bind, pure, ↓reduceIte]
    rfl
  · have hzf1 : (X86Int.aluF .sub a b).2.zf = some false := aluF_sub_zf_ne ha hb hab
    by_cases hac : a = c
    · subst a
      have hveq : v = c := by unfold TMR.vote at hv; simp at hv; exact hv.symm
      subst v
      have hzf_true : TMRVoter.voterFlagsEq.zf = some true := rfl
      refine ⟨4, by decide, b, TMRVoter.voterFlagsEq, ?_⟩
      rw [X86Int.run, X86Int.step]; dsimp only; rw [h00]
      simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, mkRegs,
        List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
      rw [X86Int.run, X86Int.step]; dsimp only; rw [h03]
      simp only [X86Int.exec, cond_zf_false hzf1, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
      rw [X86Int.run, X86Int.step]; dsimp only; rw [h05]
      simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, mkRegs,
        List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, aluF_sub_eq]
      rw [X86Int.run, X86Int.step]; dsimp only; rw [h08]
      simp only [X86Int.exec, cond_zf_true hzf_true, lift, bind, Option.bind, pure, ↓reduceIte]
      rfl
    · have hbc : b = c := by unfold TMR.vote at hv; simp [hab, hac] at hv; exact hv.1
      subst b
      have hveq : v = c := by unfold TMR.vote at hv; simp [hab] at hv; exact hv.symm
      subst v
      have hzf2 : (X86Int.aluF .sub a c).2.zf = some false := aluF_sub_zf_ne ha hc hac
      have hzf_true : TMRVoter.voterFlagsEq.zf = some true := rfl
      refine ⟨7, by decide, c, TMRVoter.voterFlagsEq, ?_⟩
      rw [X86Int.run, X86Int.step]; dsimp only; rw [h00]
      simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, mkRegs,
        List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
      rw [X86Int.run, X86Int.step]; dsimp only; rw [h03]
      simp only [X86Int.exec, cond_zf_false hzf1, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
      rw [X86Int.run, X86Int.step]; dsimp only; rw [h05]
      simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, mkRegs,
        List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
      rw [X86Int.run, X86Int.step]; dsimp only; rw [h08]
      simp only [X86Int.exec, cond_zf_false hzf2, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
      rw [X86Int.run, X86Int.step]; dsimp only; rw [h0a]
      simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, mkRegs,
        List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, aluF_sub_eq]
      rw [X86Int.run, X86Int.step]; dsimp only; rw [h0d]
      simp only [X86Int.exec, cond_zf_true hzf_true, lift, bind, Option.bind, pure, ↓reduceIte]
      rw [X86Int.run, X86Int.step]; dsimp only; rw [h14]
      simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
        List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
      rfl

private theorem voteGadget_refuse
    {G : Nat}
    (h00 : memExposureIm G = some (.cmp 0 3, 3))
    (h03 : memExposureIm (G + 0x03) = some (.jcc 4 (G + 0x17), 2))
    (h05 : memExposureIm (G + 0x05) = some (.cmp 0 1, 3))
    (h08 : memExposureIm (G + 0x08) = some (.jcc 4 (G + 0x17), 2))
    (h0a : memExposureIm (G + 0x0a) = some (.cmp 3 1, 3))
    (h0d : memExposureIm (G + 0x0d) = some (.jcc 4 (G + 0x14), 2))
    (h0f : memExposureIm (G + 0x0f) = some (.jmp memExposureRefuse, 5))
    {r2 r5 r6 r7 r8 r9 r10 : Nat}
    {r4 r11 r12 r13 r14 r15 : Option Nat}
    {fl : Flags} {mem : Mem} {k : Kern}
    {a b c : Nat} (ha : a < X86Env.W) (hb : b < X86Env.W) (hc : c < X86Env.W)
    (hv : TMR.vote a b c = none) :
    X86Int.run memExposureIm 10
      (.ok ⟨mkRegs (some a) (some c) (some r2) (some b) r4 (some r5) (some r6) (some r7)
        (some r8) (some r9) (some r10) r11 r12 r13 r14 r15, G, fl, mem, k⟩) = .exit 3 := by
  have hab : a ≠ b := by intro h; subst h; unfold TMR.vote at hv; simp at hv
  have hac : a ≠ c := by intro h; subst h; unfold TMR.vote at hv; simp at hv
  have hbc : b ≠ c := by intro h; subst h; unfold TMR.vote at hv; simp [hab] at hv
  have hzf1 : (X86Int.aluF .sub a b).2.zf = some false := aluF_sub_zf_ne ha hb hab
  have hzf2 : (X86Int.aluF .sub a c).2.zf = some false := aluF_sub_zf_ne ha hc hac
  have hzf3 : (X86Int.aluF .sub b c).2.zf = some false := aluF_sub_zf_ne hb hc hbc
  rw [show (10 : Nat) = 7 + 3 from rfl, run_add]
  have h7 : X86Int.run memExposureIm 7
      (.ok ⟨mkRegs (some a) (some c) (some r2) (some b) r4 (some r5) (some r6) (some r7)
        (some r8) (some r9) (some r10) r11 r12 r13 r14 r15, G, fl, mem, k⟩) =
      .ok ⟨mkRegs (some a) (some c) (some r2) (some b) r4 (some r5) (some r6) (some r7)
        (some r8) (some r9) (some r10) r11 r12 r13 r14 r15,
        memExposureRefuse, (X86Int.aluF .sub b c).2, mem, k⟩ := by
    rw [X86Int.run, X86Int.step]; dsimp only; rw [h00]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, mkRegs,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [h03]
    simp only [X86Int.exec, cond_zf_false hzf1, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [h05]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, mkRegs,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [h08]
    simp only [X86Int.exec, cond_zf_false hzf2, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [h0a]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, mkRegs,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [h0d]
    simp only [X86Int.exec, cond_zf_false hzf3, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [h0f]
    simp only [X86Int.exec]
    rfl
  rw [h7]
  exact exit3_run_r4opt

private theorem ea_arena_add (base off1 off2 : Nat) (h : base + off1 + off2 < X86Env.W) :
    X86Int.ea (base + off1) off2 = base + (off1 + off2) := by
  simp only [X86Int.ea, Nat.add_assoc]
  exact Nat.mod_eq_of_lt (by omega)

private theorem getElem?_wrWord_in {d : List Nat} {off v k : Nat} (hk : k < 8)
    (hd : off + 8 ≤ d.length) :
    (wrWord d off v)[off + k]? = some ((X86Int.bytes8 v)[k]! % 256) := by
  have hlen : (X86Int.bytes8 v).length = 8 := by simp [X86Int.bytes8]
  exact getElem?_wrBytes_in d off (X86Int.bytes8 v) k (by omega) (by rw [hlen]; exact hd)

private theorem getElem?_wrWord_ne {d : List Nat} {off v a : Nat}
    (ha : a < off ∨ off + 8 ≤ a) :
    (wrWord d off v)[a]? = d[a]? := by
  have hlen : (X86Int.bytes8 v).length = 8 := by simp [X86Int.bytes8]
  exact getElem?_wrBytes_ne d off (X86Int.bytes8 v) a (by rw [hlen]; exact ha)

private theorem setArenaState_after_repairs {d : List Nat} (hd : d.length = arenaSize)
    {w : Nat} (hw : w < maxSlots) (val_v last_v step_v acc_v step' acc' v_new : Nat) :
    let d1 := wrWord d (offVal w) val_v
    let d2 := wrWord d1 (offLast w) last_v
    let d3 := wrWord d2 offStep step_v
    let d4 := wrWord d3 offAcc acc_v
    wrWord (wrWord (wrWord (wrWord d4 offStep step') offAcc acc') (offVal w) v_new) (offLast w) step' =
      setArenaState d w step' acc' v_new step' := by
  intro d1 d2 d3 d4
  have hsz : ∀ {x : List Nat}, x.length = arenaSize →
      offStep + 8 ≤ x.length ∧ offAcc + 8 ≤ x.length ∧
      offVal w + 8 ≤ x.length ∧ offLast w + 8 ≤ x.length := by
    intro x hx
    simp only [offStep, offAcc, offVal, offLast, arenaSize, X86Env.page, maxSlots] at *
    omega
  have hl0 := hsz hd
  have hl1 : d1.length = arenaSize := by rw [wrWord_length, hd]
  have hl2 : d2.length = arenaSize := by rw [wrWord_length, hl1]
  have hl3 : d3.length = arenaSize := by rw [wrWord_length, hl2]
  have hl4 : d4.length = arenaSize := by rw [wrWord_length, hl3]
  apply List.ext_getElem?
  intro i
  by_cases hiL : offLast w ≤ i ∧ i < offLast w + 8
  · obtain ⟨k, hk, rfl⟩ : ∃ k < 8, i = offLast w + k := ⟨i - offLast w, by omega, by omega⟩
    rw [getElem?_wrWord_in hk (by rw [wrWord_length, wrWord_length, wrWord_length]; exact (hsz hl4).2.2.2)]
    rw [setArenaState, getElem?_wrWord_in hk (by rw [wrWord_length, wrWord_length, wrWord_length]; exact hl0.2.2.2)]
  · by_cases hiV : offVal w ≤ i ∧ i < offVal w + 8
    · obtain ⟨k, hk, rfl⟩ : ∃ k < 8, i = offVal w + k := ⟨i - offVal w, by omega, by omega⟩
      have hdisj : offVal w + k < offLast w ∨ offLast w + 8 ≤ offVal w + k := by
        simp only [offVal, offLast] at *; omega
      rw [getElem?_wrWord_ne hdisj,
        getElem?_wrWord_in hk (by rw [wrWord_length, wrWord_length]; exact (hsz hl4).2.2.1)]
      rw [setArenaState, getElem?_wrWord_ne hdisj,
        getElem?_wrWord_in hk (by rw [wrWord_length, wrWord_length]; exact hl0.2.2.1)]
    · by_cases hiA : offAcc ≤ i ∧ i < offAcc + 8
      · obtain ⟨k, hk, rfl⟩ : ∃ k < 8, i = offAcc + k := ⟨i - offAcc, by omega, by omega⟩
        have hdisjL : offAcc + k < offLast w ∨ offLast w + 8 ≤ offAcc + k := by
          simp only [offAcc, offLast] at *; omega
        have hdisjV : offAcc + k < offVal w ∨ offVal w + 8 ≤ offAcc + k := by
          simp only [offAcc, offVal] at *; omega
        rw [getElem?_wrWord_ne hdisjL, getElem?_wrWord_ne hdisjV,
          getElem?_wrWord_in hk (by rw [wrWord_length]; exact (hsz hl4).2.1)]
        rw [setArenaState, getElem?_wrWord_ne hdisjL, getElem?_wrWord_ne hdisjV,
          getElem?_wrWord_in hk (by rw [wrWord_length]; exact hl0.2.1)]
      · by_cases hiS : offStep ≤ i ∧ i < offStep + 8
        · obtain ⟨k, hk, rfl⟩ : ∃ k < 8, i = offStep + k := ⟨i - offStep, by omega, by omega⟩
          have hdisjL : offStep + k < offLast w ∨ offLast w + 8 ≤ offStep + k := by
            simp only [offStep, offLast] at *; omega
          have hdisjV : offStep + k < offVal w ∨ offVal w + 8 ≤ offStep + k := by
            simp only [offStep, offVal] at *; omega
          have hdisjA : offStep + k < offAcc ∨ offAcc + 8 ≤ offStep + k := by
            simp only [offStep, offAcc] at *; omega
          rw [getElem?_wrWord_ne hdisjL, getElem?_wrWord_ne hdisjV,
            getElem?_wrWord_ne hdisjA, getElem?_wrWord_in hk (hsz hl4).1]
          rw [setArenaState, getElem?_wrWord_ne hdisjL, getElem?_wrWord_ne hdisjV,
            getElem?_wrWord_ne hdisjA, getElem?_wrWord_in hk hl0.1]
        · have dL : i < offLast w ∨ offLast w + 8 ≤ i := by omega
          have dV : i < offVal w ∨ offVal w + 8 ≤ i := by omega
          have dA : i < offAcc ∨ offAcc + 8 ≤ i := by omega
          have dS : i < offStep ∨ offStep + 8 ≤ i := by omega
          rw [getElem?_wrWord_ne dL, getElem?_wrWord_ne dV, getElem?_wrWord_ne dA,
            getElem?_wrWord_ne dS]
          dsimp only [d4, d3, d2, d1]
          rw [getElem?_wrWord_ne dA, getElem?_wrWord_ne dS, getElem?_wrWord_ne dL,
            getElem?_wrWord_ne dV]
          rw [setArenaState, getElem?_wrWord_ne dL, getElem?_wrWord_ne dV,
            getElem?_wrWord_ne dA, getElem?_wrWord_ne dS]

private theorem stage1_ok {w v_new : Nat} (hw : w < maxSlots)
    {d0 d1 d2 : List Nat} (hd0 : ArenaDataOk d0) (hd1 : ArenaDataOk d1) (hd2 : ArenaDataOk d2)
    {r0 r1 r3 r5 : Nat} {r4 r12 r13 : Option Nat} {fl : Flags} {k : Kern}
    {val_v : Nat} (hv : TMR.vote (rdWord d0 (offVal w)) (rdWord d1 (offVal w)) (rdWord d2 (offVal w)) = some val_v) :
    ∃ n ≤ 13, ∃ b' c' fl',
      X86Int.run memExposureIm n
        (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
          (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some (16 * w))
          r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
          memExposureBase + 0x038, fl, arenaMem d0 d1 d2, k⟩) =
        .ok ⟨mkRegs (some val_v) (some c') (some (arena2 + 16 * w)) (some b') r4 (some r5)
          (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some (16 * w))
          r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
          memExposureBase + 0x067, fl',
          arenaMem (wrWord d0 (offVal w) val_v) (wrWord d1 (offVal w) val_v) (wrWord d2 (offVal w) val_v), k⟩ := by
  have hoff : offVal w + 8 ≤ arenaSize := by
    simp only [offVal, arenaSize, X86Env.page, maxSlots] at *; omega
  have hea0 : X86Int.ea (arena0 + 16 * w) 16 = arena0 + offVal w := by
    rw [ea_arena_add arena0 (16 * w) 16 (by simp [arena0, maxSlots, X86Env.W] at *; omega)]
    simp [offVal, Nat.add_comm]
  have hea1 : X86Int.ea (arena1 + 16 * w) 16 = arena1 + offVal w := by
    rw [ea_arena_add arena1 (16 * w) 16 (by simp [arena1, maxSlots, X86Env.W] at *; omega)]
    simp [offVal, Nat.add_comm]
  have hea2 : X86Int.ea (arena2 + 16 * w) 16 = arena2 + offVal w := by
    rw [ea_arena_add arena2 (16 * w) 16 (by simp [arena2, maxSlots, X86Env.W] at *; omega)]
    simp [offVal, Nat.add_comm]
  have hdec := memExposureIm_decode_stage1
  have hld0 := rd64_arena0 (d1 := d1) (d2 := d2) hd0.1 hoff
  have hld1 := rd64_arena1 (d2 := d2) hd0.1 hd1.1 hoff
  have hld2 := rd64_arena2 hd0.1 hd1.1 hd2.1 hoff
  have h3ld : X86Int.run memExposureIm 3
      (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some (16 * w))
        r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x038, fl, arenaMem d0 d1 d2, k⟩) =
      .ok ⟨mkRegs (some (rdWord d0 (offVal w))) (some (rdWord d2 (offVal w)))
        (some (arena2 + 16 * w)) (some (rdWord d1 (offVal w))) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some (16 * w))
        r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x044, fl, arenaMem d0 d1 d2, k⟩ := by
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea0, hld0]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea1, hld1]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea2, hld2]
    rfl
  obtain ⟨m, hm, b', fl', hvote⟩ := voteGadget_majority
    hdec.2.2.2.1 hdec.2.2.2.2.1 hdec.2.2.2.2.2.1 hdec.2.2.2.2.2.2.1
    hdec.2.2.2.2.2.2.2.1 hdec.2.2.2.2.2.2.2.2.1 hdec.2.2.2.2.2.2.2.2.2.2.1
    (rdWord_lt_W hd0.2 _) (rdWord_lt_W hd1.2 _) (rdWord_lt_W hd2.2 _) hv
  have hwr0 := wr64_arena0 (d1 := d1) (d2 := d2) (v := val_v) hd0.1 hoff
  have hwr1 := wr64_arena1 (d0 := wrWord d0 (offVal w) val_v) (d2 := d2) (v := val_v)
    (by rw [wrWord_length, hd0.1]) hd1.1 hoff
  have hwr2 := wr64_arena2 (d0 := wrWord d0 (offVal w) val_v) (d1 := wrWord d1 (offVal w) val_v)
    (d2 := d2) (v := val_v) (by rw [wrWord_length, hd0.1]) (by rw [wrWord_length, hd1.1]) hd2.1 hoff
  have h3st : X86Int.run memExposureIm 3
      (.ok ⟨mkRegs (some val_v) (some (rdWord d2 (offVal w))) (some (arena2 + 16 * w)) (some b')
        r4 (some r5) (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some (16 * w))
        r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x05b, fl', arenaMem d0 d1 d2, k⟩) =
      .ok ⟨mkRegs (some val_v) (some (rdWord d2 (offVal w))) (some (arena2 + 16 * w)) (some b')
        r4 (some r5) (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some (16 * w))
        r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x067, fl',
        arenaMem (wrWord d0 (offVal w) val_v) (wrWord d1 (offVal w) val_v) (wrWord d2 (offVal w) val_v), k⟩ := by
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, mkRegs,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea0, hwr0]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.2.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea1, hwr1]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.2.2.2]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea2, hwr2]
    rfl
  refine ⟨3 + m + 3, by omega, b', rdWord d2 (offVal w), fl', ?_⟩
  rw [run_add, run_add, h3ld]
  rw [show memExposureBase + 0x044 + 0x17 = memExposureBase + 0x05b from by omega] at hvote
  rw [hvote, h3st]

private theorem stage1_refuse {w v_new : Nat} (hw : w < maxSlots)
    {d0 d1 d2 : List Nat} (hd0 : ArenaDataOk d0) (hd1 : ArenaDataOk d1) (hd2 : ArenaDataOk d2)
    {r0 r1 r3 r5 : Nat} {r4 r12 r13 : Option Nat} {fl : Flags} {k : Kern}
    (hv : TMR.vote (rdWord d0 (offVal w)) (rdWord d1 (offVal w)) (rdWord d2 (offVal w)) = none) :
    X86Int.run memExposureIm 13
      (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some (16 * w))
        r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x038, fl, arenaMem d0 d1 d2, k⟩) = .exit 3 := by
  have hoff : offVal w + 8 ≤ arenaSize := by
    simp only [offVal, arenaSize, X86Env.page, maxSlots] at *; omega
  have hea0 : X86Int.ea (arena0 + 16 * w) 16 = arena0 + offVal w := by
    rw [ea_arena_add arena0 (16 * w) 16 (by simp [arena0, maxSlots, X86Env.W] at *; omega)]
    simp [offVal, Nat.add_comm]
  have hea1 : X86Int.ea (arena1 + 16 * w) 16 = arena1 + offVal w := by
    rw [ea_arena_add arena1 (16 * w) 16 (by simp [arena1, maxSlots, X86Env.W] at *; omega)]
    simp [offVal, Nat.add_comm]
  have hea2 : X86Int.ea (arena2 + 16 * w) 16 = arena2 + offVal w := by
    rw [ea_arena_add arena2 (16 * w) 16 (by simp [arena2, maxSlots, X86Env.W] at *; omega)]
    simp [offVal, Nat.add_comm]
  have hdec := memExposureIm_decode_stage1
  have hld0 := rd64_arena0 (d1 := d1) (d2 := d2) hd0.1 hoff
  have hld1 := rd64_arena1 (d2 := d2) hd0.1 hd1.1 hoff
  have hld2 := rd64_arena2 hd0.1 hd1.1 hd2.1 hoff
  rw [show (13 : Nat) = 3 + 10 from rfl, run_add]
  have h3ld : X86Int.run memExposureIm 3
      (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some (16 * w))
        r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x038, fl, arenaMem d0 d1 d2, k⟩) =
      .ok ⟨mkRegs (some (rdWord d0 (offVal w))) (some (rdWord d2 (offVal w)))
        (some (arena2 + 16 * w)) (some (rdWord d1 (offVal w))) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some (16 * w))
        r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x044, fl, arenaMem d0 d1 d2, k⟩ := by
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea0, hld0]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea1, hld1]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea2, hld2]
    rfl
  rw [h3ld]
  exact voteGadget_refuse
    hdec.2.2.2.1 hdec.2.2.2.2.1 hdec.2.2.2.2.2.1 hdec.2.2.2.2.2.2.1
    hdec.2.2.2.2.2.2.2.1 hdec.2.2.2.2.2.2.2.2.1 hdec.2.2.2.2.2.2.2.2.2.1
    (rdWord_lt_W hd0.2 _) (rdWord_lt_W hd1.2 _) (rdWord_lt_W hd2.2 _) hv

private theorem stage2_ok {w v_new : Nat} (hw : w < maxSlots)
    {d0 d1 d2 : List Nat} (hd0 : ArenaDataOk d0) (hd1 : ArenaDataOk d1) (hd2 : ArenaDataOk d2)
    {r0 r1 r3 r5 : Nat} {r4 r12 r13 : Option Nat} {fl : Flags} {k : Kern}
    {last_v : Nat} (hv : TMR.vote (rdWord d0 (offLast w)) (rdWord d1 (offLast w)) (rdWord d2 (offLast w)) = some last_v) :
    ∃ n ≤ 14, ∃ b' c' fl',
      X86Int.run memExposureIm n
        (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
          (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some (16 * w))
          r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
          memExposureBase + 0x067, fl, arenaMem d0 d1 d2, k⟩) =
        .ok ⟨mkRegs (some last_v) (some c') (some (arena2 + 16 * w)) (some b') r4 (some r5)
          (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
          r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
          memExposureBase + 0x099, fl',
          arenaMem (wrWord d0 (offLast w) last_v) (wrWord d1 (offLast w) last_v) (wrWord d2 (offLast w) last_v), k⟩ := by
  have hoff : offLast w + 8 ≤ arenaSize := by
    simp only [offLast, arenaSize, X86Env.page, maxSlots] at *; omega
  have hea0 : X86Int.ea (arena0 + 16 * w) 24 = arena0 + offLast w := by
    rw [ea_arena_add arena0 (16 * w) 24 (by simp [arena0, maxSlots, X86Env.W] at *; omega)]
    simp [offLast, Nat.add_comm]
  have hea1 : X86Int.ea (arena1 + 16 * w) 24 = arena1 + offLast w := by
    rw [ea_arena_add arena1 (16 * w) 24 (by simp [arena1, maxSlots, X86Env.W] at *; omega)]
    simp [offLast, Nat.add_comm]
  have hea2 : X86Int.ea (arena2 + 16 * w) 24 = arena2 + offLast w := by
    rw [ea_arena_add arena2 (16 * w) 24 (by simp [arena2, maxSlots, X86Env.W] at *; omega)]
    simp [offLast, Nat.add_comm]
  have hdec := memExposureIm_decode_stage2
  have hld0 := rd64_arena0 (d1 := d1) (d2 := d2) hd0.1 hoff
  have hld1 := rd64_arena1 (d2 := d2) hd0.1 hd1.1 hoff
  have hld2 := rd64_arena2 hd0.1 hd1.1 hd2.1 hoff
  have h3ld : X86Int.run memExposureIm 3
      (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some (16 * w))
        r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x067, fl, arenaMem d0 d1 d2, k⟩) =
      .ok ⟨mkRegs (some (rdWord d0 (offLast w))) (some (rdWord d2 (offLast w)))
        (some (arena2 + 16 * w)) (some (rdWord d1 (offLast w))) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some (16 * w))
        r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x073, fl, arenaMem d0 d1 d2, k⟩ := by
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea0, hld0]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea1, hld1]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea2, hld2]
    rfl
  obtain ⟨m, hm, b', fl', hvote⟩ := voteGadget_majority
    hdec.2.2.2.1 hdec.2.2.2.2.1 hdec.2.2.2.2.2.1 hdec.2.2.2.2.2.2.1
    hdec.2.2.2.2.2.2.2.1 hdec.2.2.2.2.2.2.2.2.1 hdec.2.2.2.2.2.2.2.2.2.2.1
    (rdWord_lt_W hd0.2 _) (rdWord_lt_W hd1.2 _) (rdWord_lt_W hd2.2 _) hv
  have hwr0 := wr64_arena0 (d1 := d1) (d2 := d2) (v := last_v) hd0.1 hoff
  have hwr1 := wr64_arena1 (d0 := wrWord d0 (offLast w) last_v) (d2 := d2) (v := last_v)
    (by rw [wrWord_length, hd0.1]) hd1.1 hoff
  have hwr2 := wr64_arena2 (d0 := wrWord d0 (offLast w) last_v) (d1 := wrWord d1 (offLast w) last_v)
    (d2 := d2) (v := last_v) (by rw [wrWord_length, hd0.1]) (by rw [wrWord_length, hd1.1]) hd2.1 hoff
  have h4st : X86Int.run memExposureIm 4
      (.ok ⟨mkRegs (some last_v) (some (rdWord d2 (offLast w))) (some (arena2 + 16 * w)) (some b')
        r4 (some r5) (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some (16 * w))
        r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x08a, fl', arenaMem d0 d1 d2, k⟩) =
      .ok ⟨mkRegs (some last_v) (some (rdWord d2 (offLast w))) (some (arena2 + 16 * w)) (some b')
        r4 (some r5) (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x099, fl',
        arenaMem (wrWord d0 (offLast w) last_v) (wrWord d1 (offLast w) last_v) (wrWord d2 (offLast w) last_v), k⟩ := by
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, mkRegs,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea0, hwr0]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.2.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea1, hwr1]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.2.2.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea2, hwr2]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.2.2.2.2]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
    rfl
  refine ⟨3 + m + 4, by omega, b', rdWord d2 (offLast w), fl', ?_⟩
  rw [run_add, run_add, h3ld]
  rw [show memExposureBase + 0x073 + 0x17 = memExposureBase + 0x08a from by omega] at hvote
  rw [hvote, h4st]

private theorem stage2_refuse {w v_new : Nat} (hw : w < maxSlots)
    {d0 d1 d2 : List Nat} (hd0 : ArenaDataOk d0) (hd1 : ArenaDataOk d1) (hd2 : ArenaDataOk d2)
    {r0 r1 r3 r5 : Nat} {r4 r12 r13 : Option Nat} {fl : Flags} {k : Kern}
    (hv : TMR.vote (rdWord d0 (offLast w)) (rdWord d1 (offLast w)) (rdWord d2 (offLast w)) = none) :
    X86Int.run memExposureIm 13
      (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some (16 * w))
        r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x067, fl, arenaMem d0 d1 d2, k⟩) = .exit 3 := by
  have hoff : offLast w + 8 ≤ arenaSize := by
    simp only [offLast, arenaSize, X86Env.page, maxSlots] at *; omega
  have hea0 : X86Int.ea (arena0 + 16 * w) 24 = arena0 + offLast w := by
    rw [ea_arena_add arena0 (16 * w) 24 (by simp [arena0, maxSlots, X86Env.W] at *; omega)]
    simp [offLast, Nat.add_comm]
  have hea1 : X86Int.ea (arena1 + 16 * w) 24 = arena1 + offLast w := by
    rw [ea_arena_add arena1 (16 * w) 24 (by simp [arena1, maxSlots, X86Env.W] at *; omega)]
    simp [offLast, Nat.add_comm]
  have hea2 : X86Int.ea (arena2 + 16 * w) 24 = arena2 + offLast w := by
    rw [ea_arena_add arena2 (16 * w) 24 (by simp [arena2, maxSlots, X86Env.W] at *; omega)]
    simp [offLast, Nat.add_comm]
  have hdec := memExposureIm_decode_stage2
  have hld0 := rd64_arena0 (d1 := d1) (d2 := d2) hd0.1 hoff
  have hld1 := rd64_arena1 (d2 := d2) hd0.1 hd1.1 hoff
  have hld2 := rd64_arena2 hd0.1 hd1.1 hd2.1 hoff
  rw [show (13 : Nat) = 3 + 10 from rfl, run_add]
  have h3ld : X86Int.run memExposureIm 3
      (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some (16 * w))
        r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x067, fl, arenaMem d0 d1 d2, k⟩) =
      .ok ⟨mkRegs (some (rdWord d0 (offLast w))) (some (rdWord d2 (offLast w)))
        (some (arena2 + 16 * w)) (some (rdWord d1 (offLast w))) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some (16 * w))
        r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x073, fl, arenaMem d0 d1 d2, k⟩ := by
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea0, hld0]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea1, hld1]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea2, hld2]
    rfl
  rw [h3ld]
  exact voteGadget_refuse
    hdec.2.2.2.1 hdec.2.2.2.2.1 hdec.2.2.2.2.2.1 hdec.2.2.2.2.2.2.1
    hdec.2.2.2.2.2.2.2.1 hdec.2.2.2.2.2.2.2.2.1 hdec.2.2.2.2.2.2.2.2.2.1
    (rdWord_lt_W hd0.2 _) (rdWord_lt_W hd1.2 _) (rdWord_lt_W hd2.2 _) hv

private theorem stage3_ok {w v_new : Nat}
    {d0 d1 d2 : List Nat} (hd0 : ArenaDataOk d0) (hd1 : ArenaDataOk d1) (hd2 : ArenaDataOk d2)
    {r0 r1 r3 r5 last_v : Nat} {r4 r12 r13 : Option Nat} {fl : Flags} {k : Kern}
    {step_v : Nat} (hv : TMR.vote (rdWord d0 offStep) (rdWord d1 offStep) (rdWord d2 offStep) = some step_v) :
    ∃ n ≤ 14, ∃ b' c' fl',
      X86Int.run memExposureIm n
        (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
          (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
          r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
          memExposureBase + 0x099, fl, arenaMem d0 d1 d2, k⟩) =
        .ok ⟨mkRegs (some step_v) (some c') (some (arena2 + 16 * w)) (some b') r4 (some r5)
          (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
          (some step_v) r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
          memExposureBase + 0x0c5, fl',
          arenaMem (wrWord d0 offStep step_v) (wrWord d1 offStep step_v) (wrWord d2 offStep step_v), k⟩ := by
  have hoff : offStep + 8 ≤ arenaSize := by decide
  have hea0 : X86Int.ea arena0 0 = arena0 + offStep := by decide
  have hea1 : X86Int.ea arena1 0 = arena1 + offStep := by decide
  have hea2 : X86Int.ea arena2 0 = arena2 + offStep := by decide
  have hdec := memExposureIm_decode_stage3
  have hld0 := rd64_arena0 (d1 := d1) (d2 := d2) hd0.1 hoff
  have hld1 := rd64_arena1 (d2 := d2) hd0.1 hd1.1 hoff
  have hld2 := rd64_arena2 hd0.1 hd1.1 hd2.1 hoff
  have h3ld : X86Int.run memExposureIm 3
      (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x099, fl, arenaMem d0 d1 d2, k⟩) =
      .ok ⟨mkRegs (some (rdWord d0 offStep)) (some (rdWord d2 offStep))
        (some (arena2 + 16 * w)) (some (rdWord d1 offStep)) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x0a2, fl, arenaMem d0 d1 d2, k⟩ := by
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea0, hld0]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea1, hld1]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea2, hld2]
    rfl
  obtain ⟨m, hm, b', fl', hvote⟩ := voteGadget_majority
    hdec.2.2.2.1 hdec.2.2.2.2.1 hdec.2.2.2.2.2.1 hdec.2.2.2.2.2.2.1
    hdec.2.2.2.2.2.2.2.1 hdec.2.2.2.2.2.2.2.2.1 hdec.2.2.2.2.2.2.2.2.2.2.1
    (rdWord_lt_W hd0.2 _) (rdWord_lt_W hd1.2 _) (rdWord_lt_W hd2.2 _) hv
  have hwr0 := wr64_arena0 (d1 := d1) (d2 := d2) (v := step_v) hd0.1 hoff
  have hwr1 := wr64_arena1 (d0 := wrWord d0 offStep step_v) (d2 := d2) (v := step_v)
    (by rw [wrWord_length, hd0.1]) hd1.1 hoff
  have hwr2 := wr64_arena2 (d0 := wrWord d0 offStep step_v) (d1 := wrWord d1 offStep step_v)
    (d2 := d2) (v := step_v) (by rw [wrWord_length, hd0.1]) (by rw [wrWord_length, hd1.1]) hd2.1 hoff
  have h4st : X86Int.run memExposureIm 4
      (.ok ⟨mkRegs (some step_v) (some (rdWord d2 offStep)) (some (arena2 + 16 * w)) (some b')
        r4 (some r5) (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x0b9, fl', arenaMem d0 d1 d2, k⟩) =
      .ok ⟨mkRegs (some step_v) (some (rdWord d2 offStep)) (some (arena2 + 16 * w)) (some b')
        r4 (some r5) (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        (some step_v) r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x0c5, fl',
        arenaMem (wrWord d0 offStep step_v) (wrWord d1 offStep step_v) (wrWord d2 offStep step_v), k⟩ := by
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, mkRegs,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea0, hwr0]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.2.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea1, hwr1]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.2.2.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea2, hwr2]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.2.2.2.2]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
    rfl
  refine ⟨3 + m + 4, by omega, b', rdWord d2 offStep, fl', ?_⟩
  rw [run_add, run_add, h3ld]
  rw [show memExposureBase + 0x0a2 + 0x17 = memExposureBase + 0x0b9 from by omega] at hvote
  rw [hvote, h4st]

private theorem stage3_refuse {w v_new : Nat}
    {d0 d1 d2 : List Nat} (hd0 : ArenaDataOk d0) (hd1 : ArenaDataOk d1) (hd2 : ArenaDataOk d2)
    {r0 r1 r3 r5 last_v : Nat} {r4 r12 r13 : Option Nat} {fl : Flags} {k : Kern}
    (hv : TMR.vote (rdWord d0 offStep) (rdWord d1 offStep) (rdWord d2 offStep) = none) :
    X86Int.run memExposureIm 13
      (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x099, fl, arenaMem d0 d1 d2, k⟩) = .exit 3 := by
  have hoff : offStep + 8 ≤ arenaSize := by decide
  have hea0 : X86Int.ea arena0 0 = arena0 + offStep := by decide
  have hea1 : X86Int.ea arena1 0 = arena1 + offStep := by decide
  have hea2 : X86Int.ea arena2 0 = arena2 + offStep := by decide
  have hdec := memExposureIm_decode_stage3
  have hld0 := rd64_arena0 (d1 := d1) (d2 := d2) hd0.1 hoff
  have hld1 := rd64_arena1 (d2 := d2) hd0.1 hd1.1 hoff
  have hld2 := rd64_arena2 hd0.1 hd1.1 hd2.1 hoff
  rw [show (13 : Nat) = 3 + 10 from rfl, run_add]
  have h3ld : X86Int.run memExposureIm 3
      (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x099, fl, arenaMem d0 d1 d2, k⟩) =
      .ok ⟨mkRegs (some (rdWord d0 offStep)) (some (rdWord d2 offStep))
        (some (arena2 + 16 * w)) (some (rdWord d1 offStep)) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        r12 r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x0a2, fl, arenaMem d0 d1 d2, k⟩ := by
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea0, hld0]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea1, hld1]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea2, hld2]
    rfl
  rw [h3ld]
  exact voteGadget_refuse
    hdec.2.2.2.1 hdec.2.2.2.2.1 hdec.2.2.2.2.2.1 hdec.2.2.2.2.2.2.1
    hdec.2.2.2.2.2.2.2.1 hdec.2.2.2.2.2.2.2.2.1 hdec.2.2.2.2.2.2.2.2.2.1
    (rdWord_lt_W hd0.2 _) (rdWord_lt_W hd1.2 _) (rdWord_lt_W hd2.2 _) hv

private theorem stage4_ok {w v_new : Nat}
    {d0 d1 d2 : List Nat} (hd0 : ArenaDataOk d0) (hd1 : ArenaDataOk d1) (hd2 : ArenaDataOk d2)
    {r0 r1 r3 r5 last_v step_v : Nat} {r4 r13 : Option Nat} {fl : Flags} {k : Kern}
    {acc_v : Nat} (hv : TMR.vote (rdWord d0 offAcc) (rdWord d1 offAcc) (rdWord d2 offAcc) = some acc_v) :
    ∃ n ≤ 14, ∃ b' c' fl',
      X86Int.run memExposureIm n
        (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
          (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
          (some step_v) r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
          memExposureBase + 0x0c5, fl, arenaMem d0 d1 d2, k⟩) =
        .ok ⟨mkRegs (some acc_v) (some c') (some (arena2 + 16 * w)) (some b') r4 (some r5)
          (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
          (some step_v) (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
          memExposureBase + 0x0f7, fl',
          arenaMem (wrWord d0 offAcc acc_v) (wrWord d1 offAcc acc_v) (wrWord d2 offAcc acc_v), k⟩ := by
  have hoff : offAcc + 8 ≤ arenaSize := by decide
  have hea0 : X86Int.ea arena0 8 = arena0 + offAcc := by decide
  have hea1 : X86Int.ea arena1 8 = arena1 + offAcc := by decide
  have hea2 : X86Int.ea arena2 8 = arena2 + offAcc := by decide
  have hdec := memExposureIm_decode_stage4
  have hld0 := rd64_arena0 (d1 := d1) (d2 := d2) hd0.1 hoff
  have hld1 := rd64_arena1 (d2 := d2) hd0.1 hd1.1 hoff
  have hld2 := rd64_arena2 hd0.1 hd1.1 hd2.1 hoff
  have h3ld : X86Int.run memExposureIm 3
      (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        (some step_v) r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x0c5, fl, arenaMem d0 d1 d2, k⟩) =
      .ok ⟨mkRegs (some (rdWord d0 offAcc)) (some (rdWord d2 offAcc))
        (some (arena2 + 16 * w)) (some (rdWord d1 offAcc)) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        (some step_v) r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x0d1, fl, arenaMem d0 d1 d2, k⟩ := by
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea0, hld0]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea1, hld1]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea2, hld2]
    rfl
  obtain ⟨m, hm, b', fl', hvote⟩ := voteGadget_majority
    hdec.2.2.2.1 hdec.2.2.2.2.1 hdec.2.2.2.2.2.1 hdec.2.2.2.2.2.2.1
    hdec.2.2.2.2.2.2.2.1 hdec.2.2.2.2.2.2.2.2.1 hdec.2.2.2.2.2.2.2.2.2.2.1
    (rdWord_lt_W hd0.2 _) (rdWord_lt_W hd1.2 _) (rdWord_lt_W hd2.2 _) hv
  have hwr0 := wr64_arena0 (d1 := d1) (d2 := d2) (v := acc_v) hd0.1 hoff
  have hwr1 := wr64_arena1 (d0 := wrWord d0 offAcc acc_v) (d2 := d2) (v := acc_v)
    (by rw [wrWord_length, hd0.1]) hd1.1 hoff
  have hwr2 := wr64_arena2 (d0 := wrWord d0 offAcc acc_v) (d1 := wrWord d1 offAcc acc_v)
    (d2 := d2) (v := acc_v) (by rw [wrWord_length, hd0.1]) (by rw [wrWord_length, hd1.1]) hd2.1 hoff
  have h4st : X86Int.run memExposureIm 4
      (.ok ⟨mkRegs (some acc_v) (some (rdWord d2 offAcc)) (some (arena2 + 16 * w)) (some b')
        r4 (some r5) (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        (some step_v) r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x0e8, fl', arenaMem d0 d1 d2, k⟩) =
      .ok ⟨mkRegs (some acc_v) (some (rdWord d2 offAcc)) (some (arena2 + 16 * w)) (some b')
        r4 (some r5) (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        (some step_v) (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x0f7, fl',
        arenaMem (wrWord d0 offAcc acc_v) (wrWord d1 offAcc acc_v) (wrWord d2 offAcc acc_v), k⟩ := by
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, mkRegs,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea0, hwr0]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.2.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea1, hwr1]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.2.2.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea2, hwr2]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.2.2.2.2]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
    rfl
  refine ⟨3 + m + 4, by omega, b', rdWord d2 offAcc, fl', ?_⟩
  rw [run_add, run_add, h3ld]
  rw [show memExposureBase + 0x0d1 + 0x17 = memExposureBase + 0x0e8 from by omega] at hvote
  rw [hvote, h4st]

private theorem stage4_refuse {w v_new : Nat}
    {d0 d1 d2 : List Nat} (hd0 : ArenaDataOk d0) (hd1 : ArenaDataOk d1) (hd2 : ArenaDataOk d2)
    {r0 r1 r3 r5 last_v step_v : Nat} {r4 r13 : Option Nat} {fl : Flags} {k : Kern}
    (hv : TMR.vote (rdWord d0 offAcc) (rdWord d1 offAcc) (rdWord d2 offAcc) = none) :
    X86Int.run memExposureIm 13
      (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        (some step_v) r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x0c5, fl, arenaMem d0 d1 d2, k⟩) = .exit 3 := by
  have hoff : offAcc + 8 ≤ arenaSize := by decide
  have hea0 : X86Int.ea arena0 8 = arena0 + offAcc := by decide
  have hea1 : X86Int.ea arena1 8 = arena1 + offAcc := by decide
  have hea2 : X86Int.ea arena2 8 = arena2 + offAcc := by decide
  have hdec := memExposureIm_decode_stage4
  have hld0 := rd64_arena0 (d1 := d1) (d2 := d2) hd0.1 hoff
  have hld1 := rd64_arena1 (d2 := d2) hd0.1 hd1.1 hoff
  have hld2 := rd64_arena2 hd0.1 hd1.1 hd2.1 hoff
  rw [show (13 : Nat) = 3 + 10 from rfl, run_add]
  have h3ld : X86Int.run memExposureIm 3
      (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        (some step_v) r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x0c5, fl, arenaMem d0 d1 d2, k⟩) =
      .ok ⟨mkRegs (some (rdWord d0 offAcc)) (some (rdWord d2 offAcc))
        (some (arena2 + 16 * w)) (some (rdWord d1 offAcc)) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        (some step_v) r13 (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x0d1, fl, arenaMem d0 d1 d2, k⟩ := by
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea0, hld0]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea1, hld1]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hea2, hld2]
    rfl
  rw [h3ld]
  exact voteGadget_refuse
    hdec.2.2.2.1 hdec.2.2.2.2.1 hdec.2.2.2.2.2.1 hdec.2.2.2.2.2.2.1
    hdec.2.2.2.2.2.2.2.1 hdec.2.2.2.2.2.2.2.2.1 hdec.2.2.2.2.2.2.2.2.2.1
    (rdWord_lt_W hd0.2 _) (rdWord_lt_W hd1.2 _) (rdWord_lt_W hd2.2 _) hv

private theorem stage5_ok {w v_new r0 r1 r3 r5 last_v step_v acc_v : Nat}
    {r4 : Option Nat} {fl : Flags} {mem : Mem} {k : Kern}
    (hsW : step_v < X86Env.W) (hle : last_v ≤ step_v) (hs1 : step_v + 1 < X86Env.W) :
    let delta := step_v - last_v + 1
    X86Int.run memExposureIm 7
      (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        (some step_v) (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x0f7, fl, mem, k⟩) =
      .ok ⟨mkRegs (some delta) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        (some (step_v + 1)) (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x117, (X86Int.aluF .add step_v 1).2, mem, k⟩ := by
  have hdlt : (step_v - last_v) + 1 < X86Env.W := by omega
  have hsubv : (X86Int.aluF .sub step_v last_v).1 = step_v - last_v := aluF_sub_val hsW hle
  have hsubcf : (X86Int.aluF .sub step_v last_v).2.cf = some false := by
    simp [X86Int.aluF, show ¬ step_v < last_v from by omega]
  have hadd1 : (X86Int.aluF .add (step_v - last_v) (1 % X86Env.W)).1 = step_v - last_v + 1 := by
    rw [Nat.mod_eq_of_lt (by decide)]; exact aluF_add_val hdlt
  have hadd1cf : (X86Int.aluF .add (step_v - last_v) (1 % X86Env.W)).2.cf = some false := by
    rw [Nat.mod_eq_of_lt (by decide)]; exact aluF_add_cf_false hdlt
  have hadds : (X86Int.aluF .add step_v (1 % X86Env.W)).1 = step_v + 1 := by
    rw [Nat.mod_eq_of_lt (by decide)]; exact aluF_add_val hs1
  have haddscf : (X86Int.aluF .add step_v (1 % X86Env.W)).2.cf = some false := by
    rw [Nat.mod_eq_of_lt (by decide)]; exact aluF_add_cf_false hs1
  have hdec := memExposureIm_decode_stage5
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hsubv]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.1]
  simp only [X86Int.exec, cond_cf_false hsubcf, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hadd1]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.1]
  simp only [X86Int.exec, cond_cf_false hadd1cf, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hadds]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2]
  simp only [X86Int.exec, cond_cf_false haddscf, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
  rw [Nat.mod_eq_of_lt (by decide)]
  rfl

private theorem stage5_overflow {w v_new r0 r1 r3 r5 last_v step_v acc_v : Nat}
    {r4 : Option Nat} {fl : Flags} {mem : Mem} {k : Kern}
    (hsW : step_v < X86Env.W)
    (hov : step_v < last_v ∨ X86Env.W ≤ step_v + 1) :
    ∃ n ≤ 10,
      X86Int.run memExposureIm n
        (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
          (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
          (some step_v) (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
          memExposureBase + 0x0f7, fl, mem, k⟩) = .exit 2 := by
  have hdec := memExposureIm_decode_stage5
  by_cases hlt : step_v < last_v
  · have hcf : (X86Int.aluF .sub step_v last_v).2.cf = some true := by
      simp [X86Int.aluF, hlt]
    refine ⟨3 + 3, by decide, ?_⟩
    rw [run_add]
    have h3 : X86Int.run memExposureIm 3
        (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
          (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
          (some step_v) (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
          memExposureBase + 0x0f7, fl, mem, k⟩) =
        .ok ⟨mkRegs (some (X86Int.aluF .sub step_v last_v).1) (some r1) (some (arena2 + 16 * w)) (some r3)
          r4 (some r5) (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
          (some step_v) (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
          memExposureOverflow, (X86Int.aluF .sub step_v last_v).2, mem, k⟩ := by
      rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.1]
      simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
        List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
      rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.1]
      simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
        List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
      rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.1]
      simp only [X86Int.exec, cond_cf_true hcf, lift, bind, Option.bind, pure, ↓reduceIte]
      rfl
    rw [h3]; exact exit2_run_r4opt
  · have hle : last_v ≤ step_v := by omega
    have hs1 : X86Env.W ≤ step_v + 1 := by
      rcases hov with h | h
      · exact absurd h hlt
      · exact h
    have hsubv : (X86Int.aluF .sub step_v last_v).1 = step_v - last_v := aluF_sub_val hsW hle
    have hsubcf : (X86Int.aluF .sub step_v last_v).2.cf = some false := by
      simp [X86Int.aluF, show ¬ step_v < last_v from hlt]
    by_cases hdov : X86Env.W ≤ (step_v - last_v) + 1
    · have hadd1cf : (X86Int.aluF .add (step_v - last_v) (1 % X86Env.W)).2.cf = some true := by
        rw [Nat.mod_eq_of_lt (by decide)]; exact aluF_add_cf_true hdov
      refine ⟨5 + 3, by decide, ?_⟩
      rw [run_add]
      have h5 : X86Int.run memExposureIm 5
          (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
            (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
            (some step_v) (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
            memExposureBase + 0x0f7, fl, mem, k⟩) =
          .ok ⟨mkRegs (some (X86Int.aluF .add (step_v - last_v) (1 % X86Env.W)).1) (some r1)
            (some (arena2 + 16 * w)) (some r3) r4 (some r5) (some v_new) (some w)
            (some arena0) (some arena1) (some arena2) (some last_v)
            (some step_v) (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
            memExposureOverflow, (X86Int.aluF .add (step_v - last_v) (1 % X86Env.W)).2, mem, k⟩ := by
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.1]
        simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
          List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.1]
        simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
          List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hsubv]
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.1]
        simp only [X86Int.exec, cond_cf_false hsubcf, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.1]
        simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
          List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.1]
        simp only [X86Int.exec, cond_cf_true hadd1cf, lift, bind, Option.bind, pure, ↓reduceIte]
        rfl
      rw [h5]; exact exit2_run_r4opt
    · have hdlt : (step_v - last_v) + 1 < X86Env.W := by omega
      have hadd1 : (X86Int.aluF .add (step_v - last_v) (1 % X86Env.W)).1 = step_v - last_v + 1 := by
        rw [Nat.mod_eq_of_lt (by decide)]; exact aluF_add_val hdlt
      have hadd1cf : (X86Int.aluF .add (step_v - last_v) (1 % X86Env.W)).2.cf = some false := by
        rw [Nat.mod_eq_of_lt (by decide)]; exact aluF_add_cf_false hdlt
      have haddscf : (X86Int.aluF .add step_v (1 % X86Env.W)).2.cf = some true := by
        rw [Nat.mod_eq_of_lt (by decide)]; exact aluF_add_cf_true hs1
      refine ⟨7 + 3, by decide, ?_⟩
      rw [run_add]
      have h7 : X86Int.run memExposureIm 7
          (.ok ⟨mkRegs (some r0) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
            (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
            (some step_v) (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
            memExposureBase + 0x0f7, fl, mem, k⟩) =
          .ok ⟨mkRegs (some (step_v - last_v + 1)) (some r1)
            (some (arena2 + 16 * w)) (some r3) r4 (some r5) (some v_new) (some w)
            (some arena0) (some arena1) (some arena2) (some last_v)
            (some (X86Int.aluF .add step_v (1 % X86Env.W)).1) (some acc_v)
            (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
            memExposureOverflow, (X86Int.aluF .add step_v (1 % X86Env.W)).2, mem, k⟩ := by
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.1]
        simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
          List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.1]
        simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
          List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hsubv]
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.1]
        simp only [X86Int.exec, cond_cf_false hsubcf, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.1]
        simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
          List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hadd1]
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.1]
        simp only [X86Int.exec, cond_cf_false hadd1cf, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.1]
        simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
          List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2]
        simp only [X86Int.exec, cond_cf_true haddscf, lift, bind, Option.bind, pure, ↓reduceIte]
        rfl
      rw [h7]; exact exit2_run_r4opt

private theorem sq_loop_ok {w v_new last_v step' acc_v : Nat} {r4 : Option Nat} {mem : Mem} {k : Kern} :
    ∀ (n m sq sh r5 : Nat) (fl : Flags),
      0 < m → m < 2 ^ n → sq + sh * m < X86Env.W →
      ∃ steps ≤ 11 * n, ∃ sh' r5' fl',
        X86Int.run memExposureIm steps
          (.ok ⟨mkRegs (some m) (some sh) (some (arena2 + 16 * w)) (some sq) r4 (some r5)
            (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
            (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
            memExposureBase + 0x11d, fl, mem, k⟩) =
          .ok ⟨mkRegs (some 0) (some sh') (some (arena2 + 16 * w)) (some (sq + sh * m)) r4 (some r5')
            (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
            (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
            memExposureBase + 0x143, fl', mem, k⟩
  | 0, m, _, _, _, _, hm0, hmn, _ => by simp at hmn; omega
  | n + 1, m, sq, sh, r5, fl, hm0, hmn, hlt => by
    have hdec := memExposureIm_decode_stage6
    have hpow2 : 2 ^ (1 % 64) = 2 := by decide
    have hm_div : m / 2 ^ (1 % 64) = m / 2 := by rw [hpow2]
    have hm_eq : m = 2 * (m / 2) + m % 2 := (Nat.div_add_mod m 2).symm
    -- Phase 1 of iteration: compute sq' (at 0x12f)
    have hsq' : ∃ k1 ≤ 5, ∃ sq' fl1,
        (sq' = if m % 2 = 0 then sq else sq + sh) ∧
        sq' + (2 * sh) * (m / 2) = sq + sh * m ∧
        X86Int.run memExposureIm k1
          (.ok ⟨mkRegs (some m) (some sh) (some (arena2 + 16 * w)) (some sq) r4 (some r5)
            (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
            (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
            memExposureBase + 0x11d, fl, mem, k⟩) =
          .ok ⟨mkRegs (some m) (some sh) (some (arena2 + 16 * w)) (some sq') r4 (some (m % 2))
            (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
            (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
            memExposureBase + 0x12f, fl1, mem, k⟩ := by
      by_cases heven : m % 2 = 0
      · have hzf : (X86Int.aluF .and m (1 % X86Env.W)).2.zf = some true := by
          rw [Nat.mod_eq_of_lt (by decide), aluF_and_one_zf, heven]; rfl
        have handv : (X86Int.aluF .and m (1 % X86Env.W)).1 = m % 2 := by
          rw [Nat.mod_eq_of_lt (by decide)]; exact aluF_and_one_val m
        refine ⟨3, by decide, sq, (X86Int.aluF .and m (1 % X86Env.W)).2, by simp [heven], ?_, ?_⟩
        · have hm2 : m = 2 * (m / 2) := by omega
          calc sq + 2 * sh * (m / 2) = sq + sh * (2 * (m / 2)) := by
                rw [Nat.mul_assoc, Nat.mul_left_comm 2 sh (m / 2)]
            _ = sq + sh * m := by rw [← hm2]
        · rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.1]
          simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
            List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
          rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.1]
          simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
            List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, handv]
          rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.1]
          simp only [X86Int.exec, cond_zf_true hzf, lift, bind, Option.bind, pure, ↓reduceIte]
          rfl
      · have hodd : m % 2 = 1 := by omega
        have hzf : (X86Int.aluF .and m (1 % X86Env.W)).2.zf = some false := by
          rw [Nat.mod_eq_of_lt (by decide), aluF_and_one_zf, hodd]; rfl
        have handv : (X86Int.aluF .and m (1 % X86Env.W)).1 = m % 2 := by
          rw [Nat.mod_eq_of_lt (by decide)]; exact aluF_and_one_val m
        have h1le : 1 ≤ m := by omega
        have hsum : sq + sh < X86Env.W := by
          have : sh * 1 ≤ sh * m := Nat.mul_le_mul_left sh h1le
          omega
        have haddv : (X86Int.aluF .add sq sh).1 = sq + sh := aluF_add_val hsum
        have haddcf : (X86Int.aluF .add sq sh).2.cf = some false := aluF_add_cf_false hsum
        refine ⟨5, by decide, sq + sh, (X86Int.aluF .add sq sh).2, by simp [heven], ?_, ?_⟩
        · have hm2 : m = 2 * (m / 2) + 1 := by omega
          calc sq + sh + 2 * sh * (m / 2) = sq + (sh * (2 * (m / 2)) + sh) := by
                rw [Nat.mul_assoc, Nat.mul_left_comm 2 sh (m / 2)]; omega
            _ = sq + sh * (2 * (m / 2) + 1) := by rw [Nat.mul_add, Nat.mul_one]
            _ = sq + sh * m := by rw [← hm2]
        · rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.1]
          simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
            List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
          rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.1]
          simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
            List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, handv]
          rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.1]
          simp only [X86Int.exec, cond_zf_false hzf, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
          rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.1]
          simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
            List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, haddv]
          rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.1]
          simp only [X86Int.exec, cond_cf_false haddcf, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
          rfl
    obtain ⟨k1, hk1, sq', fl1, _, hinv, hrun1⟩ := hsq'
    -- Phase 2 of iteration: shift m, check if done or double sh and loop
    by_cases hm1 : m / 2 = 0
    · have hm_is_1 : m = 1 := by omega
      have hsq'_eq : sq' = sq + sh * m := by rw [hm1] at hinv; simpa using hinv
      have hzf : (X86Int.aluF .and 0 0).2.zf = some true := by rw [aluF_and_self_zf]; rfl
      refine ⟨k1 + 3, by omega, sh, m % 2, (X86Int.aluF .and 0 0).2, ?_⟩
      rw [run_add, hrun1, ← hsq'_eq]
      rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.1]
      simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
        List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hm_div, hm1]
      rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.1]
      simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, mkRegs,
        List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
      rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.1]
      simp only [X86Int.exec, cond_zf_true hzf, lift, bind, Option.bind, pure, ↓reduceIte]
      rfl
    · have hm'_pos : 0 < m / 2 := by omega
      have hsh2_lt : sh + sh < X86Env.W := by
        have hle2 : (2 * sh) * 1 ≤ (2 * sh) * (m / 2) := Nat.mul_le_mul_left (2 * sh) hm'_pos
        omega
      have hzf : (X86Int.aluF .and (m / 2) (m / 2)).2.zf = some false := by
        rw [aluF_and_self_zf]
        have hne0 : (m / 2 == 0) = false := by
          cases h : m / 2 == 0
          · rfl
          · simp at h; omega
        rw [hne0]
      have haddsh : (X86Int.aluF .add sh sh).1 = sh + sh := aluF_add_val hsh2_lt
      have haddshcf : (X86Int.aluF .add sh sh).2.cf = some false := aluF_add_cf_false hsh2_lt
      have hrun2 : X86Int.run memExposureIm 6
          (.ok ⟨mkRegs (some m) (some sh) (some (arena2 + 16 * w)) (some sq') r4 (some (m % 2))
            (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
            (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
            memExposureBase + 0x12f, fl1, mem, k⟩) =
          .ok ⟨mkRegs (some (m / 2)) (some (sh + sh)) (some (arena2 + 16 * w)) (some sq') r4 (some (m % 2))
            (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
            (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
            memExposureBase + 0x11d, (X86Int.aluF .add sh sh).2, mem, k⟩ := by
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.1]
        simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
          List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hm_div]
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.1]
        simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, mkRegs,
          List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.1]
        simp only [X86Int.exec, cond_zf_false hzf, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.1]
        simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
          List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, haddsh]
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.2.1]
        simp only [X86Int.exec, cond_cf_false haddshcf, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.2.2]
        simp only [X86Int.exec]
        rfl
      have hm'_pow : m / 2 < 2 ^ n := by
        rw [Nat.pow_succ] at hmn
        omega
      have hsh_eq : sh + sh = 2 * sh := by omega
      have hnext_lt : sq' + (sh + sh) * (m / 2) < X86Env.W := by
        rw [hsh_eq, hinv]; exact hlt
      obtain ⟨k3, hk3, sh', r5', fl', hrec⟩ :=
        sq_loop_ok n (m / 2) sq' (sh + sh) (m % 2) (X86Int.aluF .add sh sh).2 hm'_pos hm'_pow hnext_lt
      refine ⟨k1 + 6 + k3, by omega, sh', r5', fl', ?_⟩
      rw [run_add, run_add, hrun1, hrun2, hrec, hsh_eq, hinv]

private theorem sq_loop_overflow {w v_new last_v step' acc_v : Nat} {r4 : Option Nat} {mem : Mem} {k : Kern} :
    ∀ (c m sq sh r5 : Nat) (fl : Flags),
      0 < m → sq < X86Env.W → sh < X86Env.W →
      X86Env.W ≤ sh * 2 ^ c → X86Env.W ≤ sq + sh * m →
      ∃ steps ≤ 11 * c + 3,
        X86Int.run memExposureIm steps
          (.ok ⟨mkRegs (some m) (some sh) (some (arena2 + 16 * w)) (some sq) r4 (some r5)
            (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
            (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
            memExposureBase + 0x11d, fl, mem, k⟩) = .exit 2
  | 0, _, _, sh, _, _, _, _, hshW, hcW, _ => by
    simp at hcW; omega
  | c + 1, m, sq, sh, r5, fl, hm0, hsqW, hshW, hcW, hov => by
    have hdec := memExposureIm_decode_stage6
    have hpow2 : 2 ^ (1 % 64) = 2 := by decide
    have hm_div : m / 2 ^ (1 % 64) = m / 2 := by rw [hpow2]
    by_cases hadd_ov : m % 2 = 1 ∧ X86Env.W ≤ sq + sh
    · have hzf : (X86Int.aluF .and m (1 % X86Env.W)).2.zf = some false := by
        rw [Nat.mod_eq_of_lt (by decide), aluF_and_one_zf, hadd_ov.1]; rfl
      have handv : (X86Int.aluF .and m (1 % X86Env.W)).1 = m % 2 := by
        rw [Nat.mod_eq_of_lt (by decide)]; exact aluF_and_one_val m
      have hcf : (X86Int.aluF .add sq sh).2.cf = some true := aluF_add_cf_true hadd_ov.2
      refine ⟨5 + 3, by omega, ?_⟩
      rw [run_add]
      have h5 : X86Int.run memExposureIm 5
          (.ok ⟨mkRegs (some m) (some sh) (some (arena2 + 16 * w)) (some sq) r4 (some r5)
            (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
            (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
            memExposureBase + 0x11d, fl, mem, k⟩) =
          .ok ⟨mkRegs (some m) (some sh) (some (arena2 + 16 * w)) (some (X86Int.aluF .add sq sh).1)
            r4 (some (m % 2)) (some v_new) (some w) (some arena0) (some arena1) (some arena2)
            (some last_v) (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
            memExposureOverflow, (X86Int.aluF .add sq sh).2, mem, k⟩ := by
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.1]
        simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
          List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.1]
        simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
          List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, handv]
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.1]
        simp only [X86Int.exec, cond_zf_false hzf, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.1]
        simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
          List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
        rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.1]
        simp only [X86Int.exec, cond_cf_true hcf, lift, bind, Option.bind, pure, ↓reduceIte]
        rfl
      rw [h5]; exact exit2_run_r4opt
    · -- No overflow in sq + sh; so m/2 > 0
      have hsq' : ∃ k1 ≤ 5, ∃ sq' fl1,
          sq' < X86Env.W ∧
          sq' + (2 * sh) * (m / 2) = sq + sh * m ∧
          X86Int.run memExposureIm k1
            (.ok ⟨mkRegs (some m) (some sh) (some (arena2 + 16 * w)) (some sq) r4 (some r5)
              (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
              (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
              memExposureBase + 0x11d, fl, mem, k⟩) =
            .ok ⟨mkRegs (some m) (some sh) (some (arena2 + 16 * w)) (some sq') r4 (some (m % 2))
              (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
              (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
              memExposureBase + 0x12f, fl1, mem, k⟩ := by
        by_cases heven : m % 2 = 0
        · have hzf : (X86Int.aluF .and m (1 % X86Env.W)).2.zf = some true := by
            rw [Nat.mod_eq_of_lt (by decide), aluF_and_one_zf, heven]; rfl
          have handv : (X86Int.aluF .and m (1 % X86Env.W)).1 = m % 2 := by
            rw [Nat.mod_eq_of_lt (by decide)]; exact aluF_and_one_val m
          refine ⟨3, by decide, sq, (X86Int.aluF .and m (1 % X86Env.W)).2, hsqW, ?_, ?_⟩
          · have hm2 : m = 2 * (m / 2) := by omega
            calc sq + 2 * sh * (m / 2) = sq + sh * (2 * (m / 2)) := by
                  rw [Nat.mul_assoc, Nat.mul_left_comm 2 sh (m / 2)]
              _ = sq + sh * m := by rw [← hm2]
          · rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.1]
            simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
              List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
            rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.1]
            simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
              List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, handv]
            rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.1]
            simp only [X86Int.exec, cond_zf_true hzf, lift, bind, Option.bind, pure, ↓reduceIte]
            rfl
        · have hodd : m % 2 = 1 := by omega
          have hsum : sq + sh < X86Env.W := by
            rcases Nat.lt_or_ge (sq + sh) X86Env.W with hlt | hge
            · exact hlt
            · exact absurd hge (fun h => hadd_ov ⟨hodd, h⟩)
          have hzf : (X86Int.aluF .and m (1 % X86Env.W)).2.zf = some false := by
            rw [Nat.mod_eq_of_lt (by decide), aluF_and_one_zf, hodd]; rfl
          have handv : (X86Int.aluF .and m (1 % X86Env.W)).1 = m % 2 := by
            rw [Nat.mod_eq_of_lt (by decide)]; exact aluF_and_one_val m
          have haddv : (X86Int.aluF .add sq sh).1 = sq + sh := aluF_add_val hsum
          have haddcf : (X86Int.aluF .add sq sh).2.cf = some false := aluF_add_cf_false hsum
          refine ⟨5, by decide, sq + sh, (X86Int.aluF .add sq sh).2, hsum, ?_, ?_⟩
          · have hm2 : m = 2 * (m / 2) + 1 := by omega
            calc sq + sh + 2 * sh * (m / 2) = sq + (sh * (2 * (m / 2)) + sh) := by
                  rw [Nat.mul_assoc, Nat.mul_left_comm 2 sh (m / 2)]; omega
              _ = sq + sh * (2 * (m / 2) + 1) := by rw [Nat.mul_add, Nat.mul_one]
              _ = sq + sh * m := by rw [← hm2]
          · rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.1]
            simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
              List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
            rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.1]
            simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
              List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, handv]
            rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.1]
            simp only [X86Int.exec, cond_zf_false hzf, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
            rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.1]
            simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
              List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, haddv]
            rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.1]
            simp only [X86Int.exec, cond_cf_false haddcf, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
            rfl
      obtain ⟨k1, hk1, sq', fl1, hsq'W, hinv, hrun1⟩ := hsq'
      have hm'_pos : 0 < m / 2 := by
        rcases Nat.eq_zero_or_pos (m / 2) with hm'0 | hpos
        · rw [hm'0, Nat.mul_zero, Nat.add_zero] at hinv
          omega
        · exact hpos
      have hzf : (X86Int.aluF .and (m / 2) (m / 2)).2.zf = some false := by
        rw [aluF_and_self_zf]
        have hne0 : (m / 2 == 0) = false := by
          cases h : m / 2 == 0
          · rfl
          · simp at h; omega
        rw [hne0]
      by_cases hsh_ov : X86Env.W ≤ sh + sh
      · have hcf : (X86Int.aluF .add sh sh).2.cf = some true := aluF_add_cf_true hsh_ov
        refine ⟨k1 + 5 + 3, by omega, ?_⟩
        rw [run_add, run_add, hrun1]
        have h5 : X86Int.run memExposureIm 5
            (.ok ⟨mkRegs (some m) (some sh) (some (arena2 + 16 * w)) (some sq') r4 (some (m % 2))
              (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
              (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
              memExposureBase + 0x12f, fl1, mem, k⟩) =
            .ok ⟨mkRegs (some (m / 2)) (some (X86Int.aluF .add sh sh).1) (some (arena2 + 16 * w)) (some sq')
              r4 (some (m % 2)) (some v_new) (some w) (some arena0) (some arena1) (some arena2)
              (some last_v) (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
              memExposureOverflow, (X86Int.aluF .add sh sh).2, mem, k⟩ := by
          rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.1]
          simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
            List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hm_div]
          rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.1]
          simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, mkRegs,
            List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
          rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.1]
          simp only [X86Int.exec, cond_zf_false hzf, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
          rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.1]
          simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
            List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
          rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.2.1]
          simp only [X86Int.exec, cond_cf_true hcf, lift, bind, Option.bind, pure, ↓reduceIte]
          rfl
        rw [h5]; exact exit2_run_r4opt
      · have hsh2_lt : sh + sh < X86Env.W := by omega
        have haddsh : (X86Int.aluF .add sh sh).1 = sh + sh := aluF_add_val hsh2_lt
        have haddshcf : (X86Int.aluF .add sh sh).2.cf = some false := aluF_add_cf_false hsh2_lt
        have hrun2 : X86Int.run memExposureIm 6
            (.ok ⟨mkRegs (some m) (some sh) (some (arena2 + 16 * w)) (some sq') r4 (some (m % 2))
              (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
              (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
              memExposureBase + 0x12f, fl1, mem, k⟩) =
            .ok ⟨mkRegs (some (m / 2)) (some (sh + sh)) (some (arena2 + 16 * w)) (some sq') r4 (some (m % 2))
              (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
              (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
              memExposureBase + 0x11d, (X86Int.aluF .add sh sh).2, mem, k⟩ := by
          rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.1]
          simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
            List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, hm_div]
          rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.1]
          simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, mkRegs,
            List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
          rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.1]
          simp only [X86Int.exec, cond_zf_false hzf, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
          rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.1]
          simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
            List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, haddsh]
          rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.2.1]
          simp only [X86Int.exec, cond_cf_false haddshcf, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
          rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.2.2.2.2.2.2.2.2.2.2.2]
          simp only [X86Int.exec]
          rfl
        have hcW' : X86Env.W ≤ (sh + sh) * 2 ^ c := by
          rw [Nat.pow_succ] at hcW
          have : (sh + sh) * 2 ^ c = sh * (2 ^ c * 2) := by
            rw [show sh + sh = sh * 2 from by omega, Nat.mul_assoc, Nat.mul_comm 2 (2 ^ c)]
          rw [this]; exact hcW
        have hov' : X86Env.W ≤ sq' + (sh + sh) * (m / 2) := by
          have : sh + sh = 2 * sh := by omega
          rw [this, hinv]; exact hov
        obtain ⟨k3, hk3, hrec⟩ :=
          sq_loop_overflow c (m / 2) sq' (sh + sh) (m % 2) (X86Int.aluF .add sh sh).2
            hm'_pos hsq'W hsh2_lt hcW' hov'
        refine ⟨k1 + 6 + k3, by omega, ?_⟩
        rw [run_add, run_add, hrun1, hrun2, hrec]

private theorem stage6_ok {w v_new delta r1 r3 r5 last_v step' acc_v : Nat}
    {r4 : Option Nat} {fl : Flags} {mem : Mem} {k : Kern}
    (hpos : 0 < delta) (hsq : delta * delta < X86Env.W) :
    ∃ steps ≤ 354, ∃ sh' r5' fl',
      X86Int.run memExposureIm steps
        (.ok ⟨mkRegs (some delta) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
          (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
          (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
          memExposureBase + 0x117, fl, mem, k⟩) =
        .ok ⟨mkRegs (some 0) (some sh') (some (arena2 + 16 * w)) (some (delta * delta)) r4 (some r5')
          (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
          (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
          memExposureBase + 0x143, fl', mem, k⟩ := by
  have hd32 : delta < 2 ^ 32 := by
    rcases Nat.lt_or_ge delta (2 ^ 32) with hlt | h1
    · exact hlt
    · have h2 : 2 ^ 64 ≤ delta * delta := by
        calc (2 ^ 64 : Nat) = 2 ^ 32 * 2 ^ 32 := by decide
          _ ≤ delta * delta := Nat.mul_le_mul h1 h1
      have hW : X86Env.W = 2 ^ 64 := rfl
      omega
  have hdec := memExposureIm_decode_stage6
  have h2 : X86Int.run memExposureIm 2
      (.ok ⟨mkRegs (some delta) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x117, fl, mem, k⟩) =
      .ok ⟨mkRegs (some delta) (some delta) (some (arena2 + 16 * w)) (some 0) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x11d, TMRVoter.voterFlagsEq, mem, k⟩ := by
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, aluF_xor_self]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
    rfl
  obtain ⟨m, hm, sh', r5', fl', hloop⟩ :=
    sq_loop_ok (w := w) (v_new := v_new) (last_v := last_v) (step' := step') (acc_v := acc_v)
      (r4 := r4) (mem := mem) (k := k)
      32 delta 0 delta r5 TMRVoter.voterFlagsEq hpos hd32 (by simpa using hsq)
  refine ⟨2 + m, by omega, sh', r5', fl', ?_⟩
  rw [run_add, h2]
  simpa using hloop

private theorem stage6_overflow {w v_new delta r1 r3 r5 last_v step' acc_v : Nat}
    {r4 : Option Nat} {fl : Flags} {mem : Mem} {k : Kern}
    (hdW : delta < X86Env.W) (hov : X86Env.W ≤ delta * delta) :
    ∃ steps ≤ 357,
      X86Int.run memExposureIm steps
        (.ok ⟨mkRegs (some delta) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
          (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
          (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
          memExposureBase + 0x117, fl, mem, k⟩) = .exit 2 := by
  have hge32 : 2 ^ 32 ≤ delta := by
    rcases Nat.lt_or_ge delta (2 ^ 32) with hlt | hge
    · have h1 : delta ≤ 2 ^ 32 - 1 := by omega
      have h2 : delta * delta ≤ (2 ^ 32 - 1) * (2 ^ 32 - 1) := Nat.mul_le_mul h1 h1
      have h3 : (2 ^ 32 - 1) * (2 ^ 32 - 1) < X86Env.W := by decide
      omega
    · exact hge
  have hcW : X86Env.W ≤ delta * 2 ^ 32 := by
    calc X86Env.W = 2 ^ 32 * 2 ^ 32 := by decide
      _ ≤ delta * 2 ^ 32 := Nat.mul_le_mul_right _ hge32
  have hpos : 0 < delta := by omega
  have hdec := memExposureIm_decode_stage6
  have h2 : X86Int.run memExposureIm 2
      (.ok ⟨mkRegs (some delta) (some r1) (some (arena2 + 16 * w)) (some r3) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x117, fl, mem, k⟩) =
      .ok ⟨mkRegs (some delta) (some delta) (some (arena2 + 16 * w)) (some 0) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x11d, TMRVoter.voterFlagsEq, mem, k⟩ := by
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, aluF_xor_self]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hdec.2.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
    rfl
  obtain ⟨m, hm, hloop⟩ :=
    sq_loop_overflow (w := w) (v_new := v_new) (last_v := last_v) (step' := step') (acc_v := acc_v)
      (r4 := r4) (mem := mem) (k := k)
      32 delta 0 delta r5 TMRVoter.voterFlagsEq hpos (by decide) hdW hcW (by simpa using hov)
  refine ⟨2 + m, by omega, ?_⟩
  rw [run_add, h2, hloop]

private theorem stage7_ok (s : X86Int.St) {w v_new sh r5 step_v acc_v last_v sq : Nat}
    (hw : w < maxSlots) (hacc : acc_v + sq < X86Env.W)
    {fl : Flags} {d0 d1 d2 : List Nat}
    (h0 : d0.length = arenaSize) (h1 : d1.length = arenaSize) (h2 : d2.length = arenaSize)
    (hs4 : s.reg 4 = s.reg 4) :
    X86Int.run memExposureIm 18
      (.ok ⟨mkRegs (some 0) (some sh) (some (arena2 + 16 * w)) (some sq) (s.reg 4) (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        (some (step_v + 1)) (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x143, fl, arenaMem d0 d1 d2, s.k⟩) =
      .ok ⟨mkRegs (some (acc_v + sq)) (some 0) (some (arena2 + 16 * w)) (some 0) (s.reg 4) (some 0)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        (some (step_v + 1)) (some (acc_v + sq)) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposurePark, TMRVoter.voterFlagsEq,
        arenaMem
          (setArenaState d0 w (step_v + 1) (acc_v + sq) v_new (step_v + 1))
          (setArenaState d1 w (step_v + 1) (acc_v + sq) v_new (step_v + 1))
          (setArenaState d2 w (step_v + 1) (acc_v + sq) v_new (step_v + 1)),
        s.k⟩ := by
  have _ := hs4
  have haddv : (X86Int.aluF .add acc_v sq).1 = acc_v + sq := aluF_add_val hacc
  have haddcf : (X86Int.aluF .add acc_v sq).2.cf = some false := aluF_add_cf_false hacc
  have hoffS : offStep + 8 ≤ arenaSize := by decide
  have hoffA : offAcc + 8 ≤ arenaSize := by decide
  have hoffV : offVal w + 8 ≤ arenaSize := by simp [offVal, arenaSize, X86Env.page, maxSlots] at *; omega
  have hoffL : offLast w + 8 ≤ arenaSize := by simp [offLast, arenaSize, X86Env.page, maxSlots] at *; omega
  have heaS0 : X86Int.ea arena0 0 = arena0 + offStep := by decide
  have heaA0 : X86Int.ea arena0 8 = arena0 + offAcc := by decide
  have heaV0 : X86Int.ea (arena0 + 16 * w) 16 = arena0 + offVal w := by
    rw [ea_arena_add arena0 (16 * w) 16 (by simp [arena0, maxSlots, X86Env.W] at *; omega)]
    simp [offVal, Nat.add_comm]
  have heaL0 : X86Int.ea (arena0 + 16 * w) 24 = arena0 + offLast w := by
    rw [ea_arena_add arena0 (16 * w) 24 (by simp [arena0, maxSlots, X86Env.W] at *; omega)]
    simp [offLast, Nat.add_comm]
  have heaS1 : X86Int.ea arena1 0 = arena1 + offStep := by decide
  have heaA1 : X86Int.ea arena1 8 = arena1 + offAcc := by decide
  have heaV1 : X86Int.ea (arena1 + 16 * w) 16 = arena1 + offVal w := by
    rw [ea_arena_add arena1 (16 * w) 16 (by simp [arena1, maxSlots, X86Env.W] at *; omega)]
    simp [offVal, Nat.add_comm]
  have heaL1 : X86Int.ea (arena1 + 16 * w) 24 = arena1 + offLast w := by
    rw [ea_arena_add arena1 (16 * w) 24 (by simp [arena1, maxSlots, X86Env.W] at *; omega)]
    simp [offLast, Nat.add_comm]
  have heaS2 : X86Int.ea arena2 0 = arena2 + offStep := by decide
  have heaA2 : X86Int.ea arena2 8 = arena2 + offAcc := by decide
  have heaV2 : X86Int.ea (arena2 + 16 * w) 16 = arena2 + offVal w := by
    rw [ea_arena_add arena2 (16 * w) 16 (by simp [arena2, maxSlots, X86Env.W] at *; omega)]
    simp [offVal, Nat.add_comm]
  have heaL2 : X86Int.ea (arena2 + 16 * w) 24 = arena2 + offLast w := by
    rw [ea_arena_add arena2 (16 * w) 24 (by simp [arena2, maxSlots, X86Env.W] at *; omega)]
    simp [offLast, Nat.add_comm]
  have hs0' : (wrWord d0 offStep (step_v + 1)).length = arenaSize := by rw [wrWord_length, h0]
  have ha0' : (wrWord (wrWord d0 offStep (step_v + 1)) offAcc (acc_v + sq)).length = arenaSize := by
    rw [wrWord_length, hs0']
  have hv0' : (wrWord (wrWord (wrWord d0 offStep (step_v + 1)) offAcc (acc_v + sq)) (offVal w) v_new).length = arenaSize := by
    rw [wrWord_length, ha0']
  have hl0' : (wrWord (wrWord (wrWord (wrWord d0 offStep (step_v + 1)) offAcc (acc_v + sq)) (offVal w) v_new) (offLast w) (step_v + 1)).length = arenaSize := by
    rw [wrWord_length, hv0']
  have hs1' : (wrWord d1 offStep (step_v + 1)).length = arenaSize := by rw [wrWord_length, h1]
  have ha1' : (wrWord (wrWord d1 offStep (step_v + 1)) offAcc (acc_v + sq)).length = arenaSize := by
    rw [wrWord_length, hs1']
  have hv1' : (wrWord (wrWord (wrWord d1 offStep (step_v + 1)) offAcc (acc_v + sq)) (offVal w) v_new).length = arenaSize := by
    rw [wrWord_length, ha1']
  have hl1' : (wrWord (wrWord (wrWord (wrWord d1 offStep (step_v + 1)) offAcc (acc_v + sq)) (offVal w) v_new) (offLast w) (step_v + 1)).length = arenaSize := by
    rw [wrWord_length, hv1']
  have hs2' : (wrWord d2 offStep (step_v + 1)).length = arenaSize := by rw [wrWord_length, h2]
  have ha2' : (wrWord (wrWord d2 offStep (step_v + 1)) offAcc (acc_v + sq)).length = arenaSize := by
    rw [wrWord_length, hs2']
  have hv2' : (wrWord (wrWord (wrWord d2 offStep (step_v + 1)) offAcc (acc_v + sq)) (offVal w) v_new).length = arenaSize := by
    rw [wrWord_length, ha2']
  have hd7 := memExposureIm_decode_stage7
  have hde := memExposureIm_decode_exits
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd7.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, haddv]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd7.2.1]
  simp only [X86Int.exec, cond_cf_false haddcf, lift, bind, Option.bind, pure, Bool.false_eq_true, ↓reduceIte]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd7.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, heaS0,
    wr64_arena0 h0 hoffS]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd7.2.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, heaA0,
    wr64_arena0 hs0' hoffA]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd7.2.2.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, heaV0,
    wr64_arena0 ha0' hoffV]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd7.2.2.2.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, heaL0,
    wr64_arena0 hv0' hoffL]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd7.2.2.2.2.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, heaS1,
    wr64_arena1 hl0' h1 hoffS]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd7.2.2.2.2.2.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, heaA1,
    wr64_arena1 hl0' hs1' hoffA]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd7.2.2.2.2.2.2.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, heaV1,
    wr64_arena1 hl0' ha1' hoffV]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd7.2.2.2.2.2.2.2.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, heaL1,
    wr64_arena1 hl0' hv1' hoffL]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd7.2.2.2.2.2.2.2.2.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, heaS2,
    wr64_arena2 hl0' hl1' h2 hoffS]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd7.2.2.2.2.2.2.2.2.2.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, heaA2,
    wr64_arena2 hl0' hl1' hs2' hoffA]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd7.2.2.2.2.2.2.2.2.2.2.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, heaV2,
    wr64_arena2 hl0' hl1' ha2' hoffV]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hd7.2.2.2.2.2.2.2.2.2.2.2.2.2]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, heaL2,
    wr64_arena2 hl0' hl1' hv2' hoffL]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hde.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hde.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, aluF_xor_self]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hde.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, aluF_xor_self]
  rw [X86Int.run, X86Int.step]; dsimp only; rw [hde.2.2.2.1]
  simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, List.set,
    List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some, aluF_xor_self]
  rfl

private theorem stage7_overflow {w v_new sh r5 step' acc_v last_v sq : Nat}
    (hacc : X86Env.W ≤ acc_v + sq)
    {r4 : Option Nat} {fl : Flags} {mem : Mem} {k : Kern} :
    X86Int.run memExposureIm 5
      (.ok ⟨mkRegs (some 0) (some sh) (some (arena2 + 16 * w)) (some sq) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x143, fl, mem, k⟩) = .exit 2 := by
  have hcf : (X86Int.aluF .add acc_v sq).2.cf = some true := aluF_add_cf_true hacc
  have hd7 := memExposureIm_decode_stage7
  rw [show (5 : Nat) = 2 + 3 from rfl, run_add]
  have h2 : X86Int.run memExposureIm 2
      (.ok ⟨mkRegs (some 0) (some sh) (some (arena2 + 16 * w)) (some sq) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        (some step') (some acc_v) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureBase + 0x143, fl, mem, k⟩) =
      .ok ⟨mkRegs (some 0) (some sh) (some (arena2 + 16 * w)) (some sq) r4 (some r5)
        (some v_new) (some w) (some arena0) (some arena1) (some arena2) (some last_v)
        (some step') (some (X86Int.aluF .add acc_v sq).1) (some (arena0 + 16 * w)) (some (arena1 + 16 * w)),
        memExposureOverflow, (X86Int.aluF .add acc_v sq).2, mem, k⟩ := by
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hd7.1]
    simp only [X86Int.exec, lift, bind, Option.bind, pure, St.reg, St.set, mkRegs, List.set,
      List.getElem?_cons_zero, List.getElem?_cons_succ, Option.getD_some]
    rw [X86Int.run, X86Int.step]; dsimp only; rw [hd7.2.1]
    simp only [X86Int.exec, cond_cf_true hcf, lift, bind, Option.bind, pure, ↓reduceIte]
    rfl
  rw [h2]; exact exit2_run_r4opt

/-- Architectural correctness on majority without 64-bit overflow:
for any slot `w < maxSlots`, any 64-bit `v_new < W`, and any 3 arena buffers
`d0 d1 d2` whose `(val[w], last_step[w], step, acc)` fields each have a
2-of-3 majority (`val_v, last_v, step_v, acc_v`), `memExposureFuel` steps of
`memExposureIm` repair any 1-copy mismatch and park on `memExposureDone` with
`(step_v + 1, acc_v + delta^2, v_new, step_v + 1)` written to all 3 arenas. -/
theorem memExposure_arch_majority (s : X86Int.St) {w v_new : Nat}
    (hw : w < maxSlots) (_hvnew : v_new < X86Env.W)
    {d0 d1 d2 : List Nat} (hd0 : ArenaDataOk d0) (hd1 : ArenaDataOk d1) (hd2 : ArenaDataOk d2)
    {val_v last_v step_v acc_v : Nat}
    (hval : TMR.vote (rdWord d0 (offVal w)) (rdWord d1 (offVal w)) (rdWord d2 (offVal w)) = some val_v)
    (hlast : TMR.vote (rdWord d0 (offLast w)) (rdWord d1 (offLast w)) (rdWord d2 (offLast w)) = some last_v)
    (hstep : TMR.vote (rdWord d0 offStep) (rdWord d1 offStep) (rdWord d2 offStep) = some step_v)
    (hacc : TMR.vote (rdWord d0 offAcc) (rdWord d1 offAcc) (rdWord d2 offAcc) = some acc_v)
    (hle : last_v ≤ step_v)
    (hs1 : step_v + 1 < X86Env.W)
    (hsq : (step_v - last_v + 1) * (step_v - last_v + 1) < X86Env.W)
    (hac : acc_v + (step_v - last_v + 1) * (step_v - last_v + 1) < X86Env.W) :
    X86Int.run memExposureIm memExposureFuel (.ok (memExposureInit s w v_new d0 d1 d2)) =
      .ok (memExposureDone s w v_new step_v acc_v last_v d0 d1 d2) := by
  have hsW : step_v < X86Env.W := by omega
  have hs0 := stage0_ok s (v_new := v_new) hw d0 d1 d2
  obtain ⟨n1, hn1, b1, c1, fl1, hs1_run⟩ := stage1_ok hw hd0 hd1 hd2
    (r0 := (s.reg 0).getD 0) (r1 := (s.reg 1).getD 0) (r3 := (s.reg 3).getD 0)
    (r4 := s.reg 4) (r5 := (s.reg 5).getD 0) (r12 := s.reg 12) (r13 := s.reg 13)
    (fl := (X86Int.aluF .add arena2 (16 * w)).2) (k := s.k) hval
  let d0_1 := wrWord d0 (offVal w) val_v
  let d1_1 := wrWord d1 (offVal w) val_v
  let d2_1 := wrWord d2 (offVal w) val_v
  have hd0_1 : ArenaDataOk d0_1 := hd0.wrWord (offVal w) val_v
  have hd1_1 : ArenaDataOk d1_1 := hd1.wrWord (offVal w) val_v
  have hd2_1 : ArenaDataOk d2_1 := hd2.wrWord (offVal w) val_v
  have dVL : offVal w + 8 ≤ offLast w ∨ offLast w + 8 ≤ offVal w := Or.inl (by simp [offVal, offLast]; omega)
  have hlast1 : TMR.vote (rdWord d0_1 (offLast w)) (rdWord d1_1 (offLast w)) (rdWord d2_1 (offLast w)) = some last_v := by
    simpa [d0_1, d1_1, d2_1, rdWord_wrWord_disjoint val_v dVL] using hlast
  obtain ⟨n2, hn2, b2, c2, fl2, hs2_run⟩ := stage2_ok hw hd0_1 hd1_1 hd2_1
    (r0 := val_v) (r1 := c1) (r3 := b1) (r4 := s.reg 4) (r5 := (s.reg 5).getD 0)
    (r12 := s.reg 12) (r13 := s.reg 13) (fl := fl1) (k := s.k) hlast1
  let d0_2 := wrWord d0_1 (offLast w) last_v
  let d1_2 := wrWord d1_1 (offLast w) last_v
  let d2_2 := wrWord d2_1 (offLast w) last_v
  have hd0_2 : ArenaDataOk d0_2 := hd0_1.wrWord (offLast w) last_v
  have hd1_2 : ArenaDataOk d1_2 := hd1_1.wrWord (offLast w) last_v
  have hd2_2 : ArenaDataOk d2_2 := hd2_1.wrWord (offLast w) last_v
  have dVS : offVal w + 8 ≤ offStep ∨ offStep + 8 ≤ offVal w := Or.inr (by simp [offStep, offVal]; omega)
  have dLS : offLast w + 8 ≤ offStep ∨ offStep + 8 ≤ offLast w := Or.inr (by simp [offStep, offLast]; omega)
  have hstep2 : TMR.vote (rdWord d0_2 offStep) (rdWord d1_2 offStep) (rdWord d2_2 offStep) = some step_v := by
    simpa [d0_2, d1_2, d2_2, d0_1, d1_1, d2_1, rdWord_wrWord_disjoint last_v dLS,
      rdWord_wrWord_disjoint val_v dVS] using hstep
  obtain ⟨n3, hn3, b3, c3, fl3, hs3_run⟩ := stage3_ok hd0_2 hd1_2 hd2_2
    (w := w) (v_new := v_new) (r0 := last_v) (r1 := c2) (r3 := b2) (r4 := s.reg 4)
    (r5 := (s.reg 5).getD 0) (last_v := last_v) (r12 := s.reg 12) (r13 := s.reg 13)
    (fl := fl2) (k := s.k) hstep2
  let d0_3 := wrWord d0_2 offStep step_v
  let d1_3 := wrWord d1_2 offStep step_v
  let d2_3 := wrWord d2_2 offStep step_v
  have hd0_3 : ArenaDataOk d0_3 := hd0_2.wrWord offStep step_v
  have hd1_3 : ArenaDataOk d1_3 := hd1_2.wrWord offStep step_v
  have hd2_3 : ArenaDataOk d2_3 := hd2_2.wrWord offStep step_v
  have dVA : offVal w + 8 ≤ offAcc ∨ offAcc + 8 ≤ offVal w := Or.inr (by simp [offAcc, offVal])
  have dLA : offLast w + 8 ≤ offAcc ∨ offAcc + 8 ≤ offLast w := Or.inr (by simp [offAcc, offLast]; omega)
  have dSA : offStep + 8 ≤ offAcc ∨ offAcc + 8 ≤ offStep := Or.inl (by simp [offStep, offAcc])
  have hacc3 : TMR.vote (rdWord d0_3 offAcc) (rdWord d1_3 offAcc) (rdWord d2_3 offAcc) = some acc_v := by
    simpa [d0_3, d1_3, d2_3, d0_2, d1_2, d2_2, d0_1, d1_1, d2_1,
      rdWord_wrWord_disjoint step_v dSA, rdWord_wrWord_disjoint last_v dLA,
      rdWord_wrWord_disjoint val_v dVA] using hacc
  obtain ⟨n4, hn4, b4, c4, fl4, hs4_run⟩ := stage4_ok hd0_3 hd1_3 hd2_3
    (w := w) (v_new := v_new) (r0 := step_v) (r1 := c3) (r3 := b3) (r4 := s.reg 4)
    (r5 := (s.reg 5).getD 0) (last_v := last_v) (step_v := step_v) (r13 := s.reg 13)
    (fl := fl3) (k := s.k) hacc3
  let d0_4 := wrWord d0_3 offAcc acc_v
  let d1_4 := wrWord d1_3 offAcc acc_v
  let d2_4 := wrWord d2_3 offAcc acc_v
  have hd0_4 : ArenaDataOk d0_4 := hd0_3.wrWord offAcc acc_v
  have hd1_4 : ArenaDataOk d1_4 := hd1_3.wrWord offAcc acc_v
  have hd2_4 : ArenaDataOk d2_4 := hd2_3.wrWord offAcc acc_v
  have hs5_run := stage5_ok (w := w) (v_new := v_new) (r0 := acc_v) (r1 := c4) (r3 := b4)
    (r4 := s.reg 4) (r5 := (s.reg 5).getD 0) (last_v := last_v) (step_v := step_v) (acc_v := acc_v)
    (fl := fl4) (mem := arenaMem d0_4 d1_4 d2_4) (k := s.k) hsW hle hs1
  obtain ⟨n6, hn6, sh', r5', fl6, hs6_run⟩ := stage6_ok (w := w) (v_new := v_new)
    (delta := step_v - last_v + 1) (r1 := c4) (r3 := b4) (r5 := (s.reg 5).getD 0)
    (last_v := last_v) (step' := step_v + 1) (acc_v := acc_v) (r4 := s.reg 4)
    (fl := (X86Int.aluF .add step_v 1).2) (mem := arenaMem d0_4 d1_4 d2_4) (k := s.k)
    (by omega) hsq
  have hs7_run := stage7_ok s (w := w) (v_new := v_new) (sh := sh') (r5 := r5')
    (step_v := step_v) (acc_v := acc_v) (last_v := last_v)
    (sq := (step_v - last_v + 1) * (step_v - last_v + 1)) hw hac
    (fl := fl6) hd0_4.1 hd1_4.1 hd2_4.1 rfl
  have hrep0 := setArenaState_after_repairs hd0.1 hw val_v last_v step_v acc_v (step_v + 1)
    (acc_v + (step_v - last_v + 1) * (step_v - last_v + 1)) v_new
  have hrep1 := setArenaState_after_repairs hd1.1 hw val_v last_v step_v acc_v (step_v + 1)
    (acc_v + (step_v - last_v + 1) * (step_v - last_v + 1)) v_new
  have hrep2 := setArenaState_after_repairs hd2.1 hw val_v last_v step_v acc_v (step_v + 1)
    (acc_v + (step_v - last_v + 1) * (step_v - last_v + 1)) v_new
  dsimp only at hrep0 hrep1 hrep2
  have hfull : X86Int.run memExposureIm (13 + n1 + n2 + n3 + n4 + 7 + n6 + 18)
      (.ok (memExposureInit s w v_new d0 d1 d2)) =
      .ok (memExposureDone s w v_new step_v acc_v last_v d0 d1 d2) := by
    rw [run_add, run_add, run_add, run_add, run_add, run_add, run_add]
    rw [hs0, hs1_run, hs2_run, hs3_run, hs4_run, hs5_run, hs6_run, hs7_run]
    dsimp only [memExposureDone]
    rw [show setArenaState d0_4 w (step_v + 1) (acc_v + (step_v - last_v + 1) * (step_v - last_v + 1)) v_new (step_v + 1) =
        setArenaState d0 w (step_v + 1) (acc_v + (step_v - last_v + 1) * (step_v - last_v + 1)) v_new (step_v + 1) from hrep0]
    rw [show setArenaState d1_4 w (step_v + 1) (acc_v + (step_v - last_v + 1) * (step_v - last_v + 1)) v_new (step_v + 1) =
        setArenaState d1 w (step_v + 1) (acc_v + (step_v - last_v + 1) * (step_v - last_v + 1)) v_new (step_v + 1) from hrep1]
    rw [show setArenaState d2_4 w (step_v + 1) (acc_v + (step_v - last_v + 1) * (step_v - last_v + 1)) v_new (step_v + 1) =
        setArenaState d2 w (step_v + 1) (acc_v + (step_v - last_v + 1) * (step_v - last_v + 1)) v_new (step_v + 1) from hrep2]
  have hle_fuel : 13 + n1 + n2 + n3 + n4 + 7 + n6 + 18 ≤ memExposureFuel := by
    simp only [memExposureFuel]; omega
  exact run_to_fuel_park hle_fuel rfl hfull

/-- Architectural correctness on 64-bit overflow (or `step < last`): exits with `Out.exit 2`. -/
theorem memExposure_arch_overflow (s : X86Int.St) {w v_new : Nat}
    (hw : w < maxSlots)
    {d0 d1 d2 : List Nat} (hd0 : ArenaDataOk d0) (hd1 : ArenaDataOk d1) (hd2 : ArenaDataOk d2)
    {val_v last_v step_v acc_v : Nat}
    (hval : TMR.vote (rdWord d0 (offVal w)) (rdWord d1 (offVal w)) (rdWord d2 (offVal w)) = some val_v)
    (hlast : TMR.vote (rdWord d0 (offLast w)) (rdWord d1 (offLast w)) (rdWord d2 (offLast w)) = some last_v)
    (hstep : TMR.vote (rdWord d0 offStep) (rdWord d1 offStep) (rdWord d2 offStep) = some step_v)
    (hacc : TMR.vote (rdWord d0 offAcc) (rdWord d1 offAcc) (rdWord d2 offAcc) = some acc_v)
    (hov : step_v < last_v ∨ X86Env.W ≤ step_v + 1 ∨
      X86Env.W ≤ (step_v - last_v + 1) * (step_v - last_v + 1) ∨
      X86Env.W ≤ acc_v + (step_v - last_v + 1) * (step_v - last_v + 1)) :
    X86Int.run memExposureIm memExposureFuel (.ok (memExposureInit s w v_new d0 d1 d2)) = .exit 2 := by
  have hstep_lt_W : step_v < X86Env.W := by
    unfold TMR.vote at hstep
    split at hstep
    · simp at hstep; subst hstep; exact rdWord_lt_W hd0.2 _
    · split at hstep
      · simp at hstep; subst hstep; exact rdWord_lt_W hd1.2 _
      · simp at hstep
  have hs0 := stage0_ok s (v_new := v_new) hw d0 d1 d2
  obtain ⟨n1, hn1, b1, c1, fl1, hs1_run⟩ := stage1_ok hw hd0 hd1 hd2
    (r0 := (s.reg 0).getD 0) (r1 := (s.reg 1).getD 0) (r3 := (s.reg 3).getD 0)
    (r4 := s.reg 4) (r5 := (s.reg 5).getD 0) (r12 := s.reg 12) (r13 := s.reg 13)
    (fl := (X86Int.aluF .add arena2 (16 * w)).2) (k := s.k) hval
  let d0_1 := wrWord d0 (offVal w) val_v
  let d1_1 := wrWord d1 (offVal w) val_v
  let d2_1 := wrWord d2 (offVal w) val_v
  have hd0_1 : ArenaDataOk d0_1 := hd0.wrWord (offVal w) val_v
  have hd1_1 : ArenaDataOk d1_1 := hd1.wrWord (offVal w) val_v
  have hd2_1 : ArenaDataOk d2_1 := hd2.wrWord (offVal w) val_v
  have dVL : offVal w + 8 ≤ offLast w ∨ offLast w + 8 ≤ offVal w := Or.inl (by simp [offVal, offLast]; omega)
  have hlast1 : TMR.vote (rdWord d0_1 (offLast w)) (rdWord d1_1 (offLast w)) (rdWord d2_1 (offLast w)) = some last_v := by
    simpa [d0_1, d1_1, d2_1, rdWord_wrWord_disjoint val_v dVL] using hlast
  obtain ⟨n2, hn2, b2, c2, fl2, hs2_run⟩ := stage2_ok hw hd0_1 hd1_1 hd2_1
    (r0 := val_v) (r1 := c1) (r3 := b1) (r4 := s.reg 4) (r5 := (s.reg 5).getD 0)
    (r12 := s.reg 12) (r13 := s.reg 13) (fl := fl1) (k := s.k) hlast1
  let d0_2 := wrWord d0_1 (offLast w) last_v
  let d1_2 := wrWord d1_1 (offLast w) last_v
  let d2_2 := wrWord d2_1 (offLast w) last_v
  have hd0_2 : ArenaDataOk d0_2 := hd0_1.wrWord (offLast w) last_v
  have hd1_2 : ArenaDataOk d1_2 := hd1_1.wrWord (offLast w) last_v
  have hd2_2 : ArenaDataOk d2_2 := hd2_1.wrWord (offLast w) last_v
  have dVS : offVal w + 8 ≤ offStep ∨ offStep + 8 ≤ offVal w := Or.inr (by simp [offStep, offVal]; omega)
  have dLS : offLast w + 8 ≤ offStep ∨ offStep + 8 ≤ offLast w := Or.inr (by simp [offStep, offLast]; omega)
  have hstep2 : TMR.vote (rdWord d0_2 offStep) (rdWord d1_2 offStep) (rdWord d2_2 offStep) = some step_v := by
    simpa [d0_2, d1_2, d2_2, d0_1, d1_1, d2_1, rdWord_wrWord_disjoint last_v dLS,
      rdWord_wrWord_disjoint val_v dVS] using hstep
  obtain ⟨n3, hn3, b3, c3, fl3, hs3_run⟩ := stage3_ok hd0_2 hd1_2 hd2_2
    (w := w) (v_new := v_new) (r0 := last_v) (r1 := c2) (r3 := b2) (r4 := s.reg 4)
    (r5 := (s.reg 5).getD 0) (last_v := last_v) (r12 := s.reg 12) (r13 := s.reg 13)
    (fl := fl2) (k := s.k) hstep2
  let d0_3 := wrWord d0_2 offStep step_v
  let d1_3 := wrWord d1_2 offStep step_v
  let d2_3 := wrWord d2_2 offStep step_v
  have hd0_3 : ArenaDataOk d0_3 := hd0_2.wrWord offStep step_v
  have hd1_3 : ArenaDataOk d1_3 := hd1_2.wrWord offStep step_v
  have hd2_3 : ArenaDataOk d2_3 := hd2_2.wrWord offStep step_v
  have dVA : offVal w + 8 ≤ offAcc ∨ offAcc + 8 ≤ offVal w := Or.inr (by simp [offAcc, offVal])
  have dLA : offLast w + 8 ≤ offAcc ∨ offAcc + 8 ≤ offLast w := Or.inr (by simp [offAcc, offLast]; omega)
  have dSA : offStep + 8 ≤ offAcc ∨ offAcc + 8 ≤ offStep := Or.inl (by simp [offStep, offAcc])
  have hacc3 : TMR.vote (rdWord d0_3 offAcc) (rdWord d1_3 offAcc) (rdWord d2_3 offAcc) = some acc_v := by
    simpa [d0_3, d1_3, d2_3, d0_2, d1_2, d2_2, d0_1, d1_1, d2_1,
      rdWord_wrWord_disjoint step_v dSA, rdWord_wrWord_disjoint last_v dLA,
      rdWord_wrWord_disjoint val_v dVA] using hacc
  obtain ⟨n4, hn4, b4, c4, fl4, hs4_run⟩ := stage4_ok hd0_3 hd1_3 hd2_3
    (w := w) (v_new := v_new) (r0 := step_v) (r1 := c3) (r3 := b3) (r4 := s.reg 4)
    (r5 := (s.reg 5).getD 0) (last_v := last_v) (step_v := step_v) (r13 := s.reg 13)
    (fl := fl3) (k := s.k) hacc3
  let d0_4 := wrWord d0_3 offAcc acc_v
  let d1_4 := wrWord d1_3 offAcc acc_v
  let d2_4 := wrWord d2_3 offAcc acc_v
  by_cases hov5 : step_v < last_v ∨ X86Env.W ≤ step_v + 1
  · obtain ⟨n5, hn5, hs5_run⟩ := stage5_overflow (w := w) (v_new := v_new) (r0 := acc_v)
      (r1 := c4) (r3 := b4) (r4 := s.reg 4) (r5 := (s.reg 5).getD 0)
      (last_v := last_v) (step_v := step_v) (acc_v := acc_v)
      (fl := fl4) (mem := arenaMem d0_4 d1_4 d2_4) (k := s.k) hstep_lt_W hov5
    refine run_to_fuel_exit (k := 13 + n1 + n2 + n3 + n4 + n5) (by simp [memExposureFuel]; omega) ?_
    rw [run_add, run_add, run_add, run_add, run_add, hs0, hs1_run, hs2_run, hs3_run, hs4_run, hs5_run]
  · have hle : last_v ≤ step_v := by omega
    have hs1 : step_v + 1 < X86Env.W := by omega
    have hs5_run := stage5_ok (w := w) (v_new := v_new) (r0 := acc_v) (r1 := c4) (r3 := b4)
      (r4 := s.reg 4) (r5 := (s.reg 5).getD 0) (last_v := last_v) (step_v := step_v) (acc_v := acc_v)
      (fl := fl4) (mem := arenaMem d0_4 d1_4 d2_4) (k := s.k) hstep_lt_W hle hs1
    by_cases hov6 : X86Env.W ≤ (step_v - last_v + 1) * (step_v - last_v + 1)
    · obtain ⟨n6, hn6, hs6_run⟩ := stage6_overflow (w := w) (v_new := v_new)
        (delta := step_v - last_v + 1) (r1 := c4) (r3 := b4) (r5 := (s.reg 5).getD 0)
        (last_v := last_v) (step' := step_v + 1) (acc_v := acc_v) (r4 := s.reg 4)
        (fl := (X86Int.aluF .add step_v 1).2) (mem := arenaMem d0_4 d1_4 d2_4) (k := s.k)
        (by omega) hov6
      refine run_to_fuel_exit (k := 13 + n1 + n2 + n3 + n4 + 7 + n6) (by simp [memExposureFuel]; omega) ?_
      rw [run_add, run_add, run_add, run_add, run_add, run_add]
      rw [hs0, hs1_run, hs2_run, hs3_run, hs4_run, hs5_run, hs6_run]
    · have hsq : (step_v - last_v + 1) * (step_v - last_v + 1) < X86Env.W := by omega
      have hac_ov : X86Env.W ≤ acc_v + (step_v - last_v + 1) * (step_v - last_v + 1) := by
        rcases hov with h | h | h | h
        · exact absurd h (by omega)
        · exact absurd h (by omega)
        · exact absurd h (by omega)
        · exact h
      obtain ⟨n6, hn6, sh', r5', fl6, hs6_run⟩ := stage6_ok (w := w) (v_new := v_new)
        (delta := step_v - last_v + 1) (r1 := c4) (r3 := b4) (r5 := (s.reg 5).getD 0)
        (last_v := last_v) (step' := step_v + 1) (acc_v := acc_v) (r4 := s.reg 4)
        (fl := (X86Int.aluF .add step_v 1).2) (mem := arenaMem d0_4 d1_4 d2_4) (k := s.k)
        (by omega) hsq
      have hs7_ov := stage7_overflow (w := w) (v_new := v_new) (sh := sh') (r5 := r5')
        (step' := step_v + 1) (acc_v := acc_v) (last_v := last_v)
        (sq := (step_v - last_v + 1) * (step_v - last_v + 1)) hac_ov
        (r4 := s.reg 4) (fl := fl6) (mem := arenaMem d0_4 d1_4 d2_4) (k := s.k)
      refine run_to_fuel_exit (k := 13 + n1 + n2 + n3 + n4 + 7 + n6 + 5) (by simp [memExposureFuel]; omega) ?_
      rw [run_add, run_add, run_add, run_add, run_add, run_add, run_add]
      rw [hs0, hs1_run, hs2_run, hs3_run, hs4_run, hs5_run, hs6_run, hs7_ov]

theorem memExposure_arch_bad_slot (s : X86Int.St) {w v_new : Nat}
    (hw : maxSlots ≤ w) (d0 d1 d2 : List Nat) :
    X86Int.run memExposureIm memExposureFuel (.ok (memExposureInit s w v_new d0 d1 d2)) = .exit 2 :=
  run_to_fuel_exit (k := 5) (by simp [memExposureFuel]) (stage0_bad_slot s hw d0 d1 d2)

/-- Architectural refusal (`Out.exit 3`): whenever any of `(val[w], last_step[w], step, acc)`
exhibits a 3-way split across the 3 arenas, `memExposureIm` halts via Linux `exit(3)`. -/
theorem memExposure_arch_refuse (s : X86Int.St) {w v_new : Nat} (hw : w < maxSlots)
    {d0 d1 d2 : List Nat} (hd0 : ArenaDataOk d0) (hd1 : ArenaDataOk d1) (hd2 : ArenaDataOk d2)
    (hsplit :
      TMR.vote (rdWord d0 (offVal w)) (rdWord d1 (offVal w)) (rdWord d2 (offVal w)) = none ∨
      TMR.vote (rdWord d0 (offLast w)) (rdWord d1 (offLast w)) (rdWord d2 (offLast w)) = none ∨
      TMR.vote (rdWord d0 offStep) (rdWord d1 offStep) (rdWord d2 offStep) = none ∨
      TMR.vote (rdWord d0 offAcc) (rdWord d1 offAcc) (rdWord d2 offAcc) = none) :
    X86Int.run memExposureIm memExposureFuel (.ok (memExposureInit s w v_new d0 d1 d2)) = .exit 3 := by
  have hs0 := stage0_ok s (v_new := v_new) hw d0 d1 d2
  cases hv1 : TMR.vote (rdWord d0 (offVal w)) (rdWord d1 (offVal w)) (rdWord d2 (offVal w)) with
  | none =>
    refine run_to_fuel_exit (k := 13 + 13) (by simp [memExposureFuel]) ?_
    rw [run_add, hs0]
    exact stage1_refuse hw hd0 hd1 hd2 hv1
  | some val_v =>
    obtain ⟨n1, hn1, b1, c1, fl1, hs1_run⟩ := stage1_ok hw hd0 hd1 hd2
      (r0 := (s.reg 0).getD 0) (r1 := (s.reg 1).getD 0) (r3 := (s.reg 3).getD 0)
      (r4 := s.reg 4) (r5 := (s.reg 5).getD 0) (r12 := s.reg 12) (r13 := s.reg 13)
      (fl := (X86Int.aluF .add arena2 (16 * w)).2) (k := s.k) hv1
    let d0_1 := wrWord d0 (offVal w) val_v
    let d1_1 := wrWord d1 (offVal w) val_v
    let d2_1 := wrWord d2 (offVal w) val_v
    have hd0_1 : ArenaDataOk d0_1 := hd0.wrWord (offVal w) val_v
    have hd1_1 : ArenaDataOk d1_1 := hd1.wrWord (offVal w) val_v
    have hd2_1 : ArenaDataOk d2_1 := hd2.wrWord (offVal w) val_v
    have dVL : offVal w + 8 ≤ offLast w ∨ offLast w + 8 ≤ offVal w := Or.inl (by simp [offVal, offLast]; omega)
    cases hv2 : TMR.vote (rdWord d0 (offLast w)) (rdWord d1 (offLast w)) (rdWord d2 (offLast w)) with
    | none =>
      have hlast1 : TMR.vote (rdWord d0_1 (offLast w)) (rdWord d1_1 (offLast w)) (rdWord d2_1 (offLast w)) = none := by
        simpa [d0_1, d1_1, d2_1, rdWord_wrWord_disjoint val_v dVL] using hv2
      refine run_to_fuel_exit (k := 13 + n1 + 13) (by simp [memExposureFuel]; omega) ?_
      rw [run_add, run_add, hs0, hs1_run]
      exact stage2_refuse hw hd0_1 hd1_1 hd2_1 hlast1
    | some last_v =>
      have hlast1 : TMR.vote (rdWord d0_1 (offLast w)) (rdWord d1_1 (offLast w)) (rdWord d2_1 (offLast w)) = some last_v := by
        simpa [d0_1, d1_1, d2_1, rdWord_wrWord_disjoint val_v dVL] using hv2
      obtain ⟨n2, hn2, b2, c2, fl2, hs2_run⟩ := stage2_ok hw hd0_1 hd1_1 hd2_1
        (r0 := val_v) (r1 := c1) (r3 := b1) (r4 := s.reg 4) (r5 := (s.reg 5).getD 0)
        (r12 := s.reg 12) (r13 := s.reg 13) (fl := fl1) (k := s.k) hlast1
      let d0_2 := wrWord d0_1 (offLast w) last_v
      let d1_2 := wrWord d1_1 (offLast w) last_v
      let d2_2 := wrWord d2_1 (offLast w) last_v
      have hd0_2 : ArenaDataOk d0_2 := hd0_1.wrWord (offLast w) last_v
      have hd1_2 : ArenaDataOk d1_2 := hd1_1.wrWord (offLast w) last_v
      have hd2_2 : ArenaDataOk d2_2 := hd2_1.wrWord (offLast w) last_v
      have dVS : offVal w + 8 ≤ offStep ∨ offStep + 8 ≤ offVal w := Or.inr (by simp [offStep, offVal]; omega)
      have dLS : offLast w + 8 ≤ offStep ∨ offStep + 8 ≤ offLast w := Or.inr (by simp [offStep, offLast]; omega)
      cases hv3 : TMR.vote (rdWord d0 offStep) (rdWord d1 offStep) (rdWord d2 offStep) with
      | none =>
        have hstep2 : TMR.vote (rdWord d0_2 offStep) (rdWord d1_2 offStep) (rdWord d2_2 offStep) = none := by
          simpa [d0_2, d1_2, d2_2, d0_1, d1_1, d2_1, rdWord_wrWord_disjoint last_v dLS,
            rdWord_wrWord_disjoint val_v dVS] using hv3
        refine run_to_fuel_exit (k := 13 + n1 + n2 + 13) (by simp [memExposureFuel]; omega) ?_
        rw [run_add, run_add, run_add, hs0, hs1_run, hs2_run]
        exact stage3_refuse hd0_2 hd1_2 hd2_2 hstep2
      | some step_v =>
        have hstep2 : TMR.vote (rdWord d0_2 offStep) (rdWord d1_2 offStep) (rdWord d2_2 offStep) = some step_v := by
          simpa [d0_2, d1_2, d2_2, d0_1, d1_1, d2_1, rdWord_wrWord_disjoint last_v dLS,
            rdWord_wrWord_disjoint val_v dVS] using hv3
        obtain ⟨n3, hn3, b3, c3, fl3, hs3_run⟩ := stage3_ok hd0_2 hd1_2 hd2_2
          (w := w) (v_new := v_new) (r0 := last_v) (r1 := c2) (r3 := b2) (r4 := s.reg 4)
          (r5 := (s.reg 5).getD 0) (last_v := last_v) (r12 := s.reg 12) (r13 := s.reg 13)
          (fl := fl2) (k := s.k) hstep2
        let d0_3 := wrWord d0_2 offStep step_v
        let d1_3 := wrWord d1_2 offStep step_v
        let d2_3 := wrWord d2_2 offStep step_v
        have hd0_3 : ArenaDataOk d0_3 := hd0_2.wrWord offStep step_v
        have hd1_3 : ArenaDataOk d1_3 := hd1_2.wrWord offStep step_v
        have hd2_3 : ArenaDataOk d2_3 := hd2_2.wrWord offStep step_v
        have dVA : offVal w + 8 ≤ offAcc ∨ offAcc + 8 ≤ offVal w := Or.inr (by simp [offAcc, offVal])
        have dLA : offLast w + 8 ≤ offAcc ∨ offAcc + 8 ≤ offLast w := Or.inr (by simp [offAcc, offLast]; omega)
        have dSA : offStep + 8 ≤ offAcc ∨ offAcc + 8 ≤ offStep := Or.inl (by simp [offStep, offAcc])
        have hv4 : TMR.vote (rdWord d0 offAcc) (rdWord d1 offAcc) (rdWord d2 offAcc) = none := by
          rcases hsplit with h | h | h | h
          · rw [hv1] at h; cases h
          · rw [hv2] at h; cases h
          · rw [hv3] at h; cases h
          · exact h
        have hacc3 : TMR.vote (rdWord d0_3 offAcc) (rdWord d1_3 offAcc) (rdWord d2_3 offAcc) = none := by
          simpa [d0_3, d1_3, d2_3, d0_2, d1_2, d2_2, d0_1, d1_1, d2_1,
            rdWord_wrWord_disjoint step_v dSA, rdWord_wrWord_disjoint last_v dLA,
            rdWord_wrWord_disjoint val_v dVA] using hv4
        refine run_to_fuel_exit (k := 13 + n1 + n2 + n3 + 13) (by simp [memExposureFuel]; omega) ?_
        rw [run_add, run_add, run_add, run_add, hs0, hs1_run, hs2_run, hs3_run]
        exact stage4_refuse hd0_3 hd1_3 hd2_3 hacc3

/-! ### Zero Single-Fault Underestimation & Physical Fault Bound (`X86Int.prun`) -/

private theorem wrWord_overwrite_same {d : List Nat} {off v1 v2 : Nat}
    (hoff : off + 8 ≤ d.length) :
    wrWord (wrWord d off v1) off v2 = wrWord d off v2 := by
  apply List.ext_getElem?
  intro a
  by_cases ha : off ≤ a ∧ a < off + 8
  · obtain ⟨k, hk, rfl⟩ : ∃ k < 8, a = off + k := ⟨a - off, by omega, by omega⟩
    have hlen1 : off + 8 ≤ (wrWord d off v1).length := by rw [wrWord_length]; exact hoff
    rw [getElem?_wrWord_in hk hlen1, getElem?_wrWord_in hk hoff]
  · have hne : a < off ∨ off + 8 ≤ a := by omega
    rw [getElem?_wrWord_ne hne, getElem?_wrWord_ne hne, getElem?_wrWord_ne hne]

private theorem setArenaState_wrWord_step {d : List Nat} (hd : d.length = arenaSize)
    (w x step' acc' v_new last' : Nat) :
    setArenaState (wrWord d offStep x) w step' acc' v_new last' =
      setArenaState d w step' acc' v_new last' := by
  unfold setArenaState
  rw [wrWord_overwrite_same (by simp [offStep, hd, arenaSize, X86Env.page])]

private theorem setArenaState_wrWord_acc {d : List Nat} (hd : d.length = arenaSize)
    (w x step' acc' v_new last' : Nat) :
    setArenaState (wrWord d offAcc x) w step' acc' v_new last' =
      setArenaState d w step' acc' v_new last' := by
  have hS : offStep + 8 ≤ d.length := by simp [offStep, hd, arenaSize, X86Env.page]
  have hA : offAcc + 8 ≤ d.length := by simp [offAcc, hd, arenaSize, X86Env.page]
  have hdisj : offStep + 8 ≤ offAcc ∨ offAcc + 8 ≤ offStep := Or.inl (by simp [offStep, offAcc])
  unfold setArenaState
  rw [wrWord_wrWord_comm hA hS hdisj.symm, wrWord_overwrite_same (by rw [wrWord_length]; exact hA)]

private theorem setArenaState_wrWord_val {d : List Nat} (hd : d.length = arenaSize)
    {w : Nat} (hw : w < maxSlots) (x step' acc' v_new last' : Nat) :
    setArenaState (wrWord d (offVal w) x) w step' acc' v_new last' =
      setArenaState d w step' acc' v_new last' := by
  have hsz : ∀ {xs : List Nat}, xs.length = arenaSize →
      offStep + 8 ≤ xs.length ∧ offAcc + 8 ≤ xs.length ∧ offVal w + 8 ≤ xs.length := by
    intro xs hxs
    simp only [offStep, offAcc, offVal, arenaSize, X86Env.page, maxSlots] at *
    omega
  have h0 := hsz hd
  have hV : offVal w + 8 ≤ (wrWord d offStep step').length := by rw [wrWord_length]; exact h0.2.2
  have hA : offAcc + 8 ≤ (wrWord d offStep step').length := by rw [wrWord_length]; exact h0.2.1
  have dVS : offStep + 8 ≤ offVal w ∨ offVal w + 8 ≤ offStep := Or.inl (by simp [offStep, offVal]; omega)
  have dVA : offAcc + 8 ≤ offVal w ∨ offVal w + 8 ≤ offAcc := Or.inl (by simp [offAcc, offVal])
  unfold setArenaState
  rw [wrWord_wrWord_comm h0.2.2 h0.1 dVS.symm,
    wrWord_wrWord_comm hV hA dVA.symm,
    wrWord_overwrite_same (by rw [wrWord_length, wrWord_length]; exact h0.2.2)]

private theorem setArenaState_wrWord_last {d : List Nat} (hd : d.length = arenaSize)
    {w : Nat} (hw : w < maxSlots) (x step' acc' v_new last' : Nat) :
    setArenaState (wrWord d (offLast w) x) w step' acc' v_new last' =
      setArenaState d w step' acc' v_new last' := by
  have hsz : ∀ {xs : List Nat}, xs.length = arenaSize →
      offStep + 8 ≤ xs.length ∧ offAcc + 8 ≤ xs.length ∧
      offVal w + 8 ≤ xs.length ∧ offLast w + 8 ≤ xs.length := by
    intro xs hxs
    simp only [offStep, offAcc, offVal, offLast, arenaSize, X86Env.page, maxSlots] at *
    omega
  have h0 := hsz hd
  have h1 := hsz (by rw [wrWord_length]; exact hd : (wrWord d offStep step').length = arenaSize)
  have h2 := hsz (by rw [wrWord_length, wrWord_length]; exact hd :
    (wrWord (wrWord d offStep step') offAcc acc').length = arenaSize)
  have dLS : offStep + 8 ≤ offLast w ∨ offLast w + 8 ≤ offStep := Or.inl (by simp [offStep, offLast]; omega)
  have dLA : offAcc + 8 ≤ offLast w ∨ offLast w + 8 ≤ offAcc := Or.inl (by simp [offAcc, offLast]; omega)
  have dLV : offVal w + 8 ≤ offLast w ∨ offLast w + 8 ≤ offVal w := Or.inl (by simp [offVal, offLast]; omega)
  unfold setArenaState
  rw [wrWord_wrWord_comm h0.2.2.2 h0.1 dLS.symm,
    wrWord_wrWord_comm h1.2.2.2 h1.2.1 dLA.symm,
    wrWord_wrWord_comm h2.2.2.2 h2.2.2.1 dLV.symm,
    wrWord_overwrite_same (by rw [wrWord_length, wrWord_length, wrWord_length]; exact h0.2.2.2)]

/-- Single-arena fault model relative to clean replicated arena image `d`:
at most one of `(d0, d1, d2)` differs arbitrarily across the 4 KiB page from `d`. -/
def SingleArenaFault (d d0 d1 d2 : List Nat) : Prop :=
  (d1 = d ∧ d2 = d) ∨ (d0 = d ∧ d2 = d) ∨ (d0 = d ∧ d1 = d)

theorem singleArenaFault_refl (d : List Nat) : SingleArenaFault d d d d :=
  Or.inl ⟨rfl, rfl⟩

theorem vote_single_fault {d d0 d1 d2 : List Nat} (hsf : SingleArenaFault d d0 d1 d2) (off : Nat) :
    TMR.vote (rdWord d0 off) (rdWord d1 off) (rdWord d2 off) = some (rdWord d off) := by
  unfold TMR.vote
  rcases hsf with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> split <;> simp_all

/-- Zero Single-Fault Underestimation Certificate (`sdc_cert_zero`):
under any arbitrary 1-arena corruption `SingleArenaFault d d0 d1 d2`, `memExposureIm`
completes cleanly in `memExposureFuel` cycles, writes exact `(step + 1, acc + delta^2, v_new, step + 1)`
to all 3 arenas (`X86Int.rd64` matches `acc'` in every arena with zero underestimation),
and leaves every untouched slot `w' ≠ w` cleanly 2-of-3 recoverable. -/
theorem sdc_cert_zero (s : X86Int.St) {w v_new : Nat} (hw : w < maxSlots)
    (hvnew : v_new < X86Env.W)
    {d d0 d1 d2 : List Nat} (hd : ArenaDataOk d)
    (hd0 : ArenaDataOk d0) (hd1 : ArenaDataOk d1) (hd2 : ArenaDataOk d2)
    (hsf : SingleArenaFault d d0 d1 d2)
    (hle : rdWord d (offLast w) ≤ rdWord d offStep)
    (hs1 : rdWord d offStep + 1 < X86Env.W)
    (hsq : (rdWord d offStep - rdWord d (offLast w) + 1) *
      (rdWord d offStep - rdWord d (offLast w) + 1) < X86Env.W)
    (hac : rdWord d offAcc + (rdWord d offStep - rdWord d (offLast w) + 1) *
      (rdWord d offStep - rdWord d (offLast w) + 1) < X86Env.W) :
    let step_v := rdWord d offStep
    let acc_v := rdWord d offAcc
    let last_v := rdWord d (offLast w)
    let delta := step_v - last_v + 1
    let step' := step_v + 1
    let acc' := acc_v + delta * delta
    let sDone := memExposureDone s w v_new step_v acc_v last_v d0 d1 d2
    X86Int.run memExposureIm memExposureFuel (.ok (memExposureInit s w v_new d0 d1 d2)) =
      .ok sDone ∧
    X86Int.rd64 sDone.mem (arena0 + offAcc) = some acc' ∧
    X86Int.rd64 sDone.mem (arena1 + offAcc) = some acc' ∧
    X86Int.rd64 sDone.mem (arena2 + offAcc) = some acc' ∧
    X86Int.rd64 sDone.mem (arena0 + offStep) = some step' ∧
    X86Int.rd64 sDone.mem (arena1 + offStep) = some step' ∧
    X86Int.rd64 sDone.mem (arena2 + offStep) = some step' ∧
    X86Int.rd64 sDone.mem (arena0 + offVal w) = some v_new ∧
    X86Int.rd64 sDone.mem (arena1 + offVal w) = some v_new ∧
    X86Int.rd64 sDone.mem (arena2 + offVal w) = some v_new ∧
    X86Int.rd64 sDone.mem (arena0 + offLast w) = some step' ∧
    X86Int.rd64 sDone.mem (arena1 + offLast w) = some step' ∧
    X86Int.rd64 sDone.mem (arena2 + offLast w) = some step' ∧
    TMR.vote ((X86Int.rd64 sDone.mem (arena0 + offAcc)).getD 0)
      ((X86Int.rd64 sDone.mem (arena1 + offAcc)).getD 0)
      ((X86Int.rd64 sDone.mem (arena2 + offAcc)).getD 0) = some acc' ∧
    (∀ w' < maxSlots, w' ≠ w →
      TMR.vote ((X86Int.rd64 sDone.mem (arena0 + offVal w')).getD 0)
        ((X86Int.rd64 sDone.mem (arena1 + offVal w')).getD 0)
        ((X86Int.rd64 sDone.mem (arena2 + offVal w')).getD 0) = some (rdWord d (offVal w')) ∧
      TMR.vote ((X86Int.rd64 sDone.mem (arena0 + offLast w')).getD 0)
        ((X86Int.rd64 sDone.mem (arena1 + offLast w')).getD 0)
        ((X86Int.rd64 sDone.mem (arena2 + offLast w')).getD 0) = some (rdWord d (offLast w'))) := by
  have _ := hd
  intro step_v acc_v last_v delta step' acc' sDone
  have hval : TMR.vote (rdWord d0 (offVal w)) (rdWord d1 (offVal w)) (rdWord d2 (offVal w)) =
      some (rdWord d (offVal w)) := vote_single_fault hsf (offVal w)
  have hlast : TMR.vote (rdWord d0 (offLast w)) (rdWord d1 (offLast w)) (rdWord d2 (offLast w)) =
      some last_v := vote_single_fault hsf (offLast w)
  have hstep : TMR.vote (rdWord d0 offStep) (rdWord d1 offStep) (rdWord d2 offStep) =
      some step_v := vote_single_fault hsf offStep
  have hacc : TMR.vote (rdWord d0 offAcc) (rdWord d1 offAcc) (rdWord d2 offAcc) =
      some acc_v := vote_single_fault hsf offAcc
  have hrun := memExposure_arch_majority s hw hvnew hd0 hd1 hd2 hval hlast hstep hacc hle hs1 hsq hac
  let d0' := setArenaState d0 w step' acc' v_new step'
  let d1' := setArenaState d1 w step' acc' v_new step'
  let d2' := setArenaState d2 w step' acc' v_new step'
  have hl0 : d0'.length = arenaSize := by rw [setArenaState_length, hd0.1]
  have hl1 : d1'.length = arenaSize := by rw [setArenaState_length, hd1.1]
  have hl2 : d2'.length = arenaSize := by rw [setArenaState_length, hd2.1]
  have hoffS : offStep + 8 ≤ arenaSize := by decide
  have hoffA : offAcc + 8 ≤ arenaSize := by decide
  have hoffV : offVal w + 8 ≤ arenaSize := by simp [offVal, arenaSize, X86Env.page, maxSlots] at *; omega
  have hoffL : offLast w + 8 ≤ arenaSize := by simp [offLast, arenaSize, X86Env.page, maxSlots] at *; omega
  have ha0 : X86Int.rd64 sDone.mem (arena0 + offAcc) = some acc' := by
    change X86Int.rd64 (arenaMem d0' d1' d2') (arena0 + offAcc) = some acc'
    rw [rd64_arena0 hl0 hoffA, setArenaState_rdAcc hd0.1 hac]
  have ha1 : X86Int.rd64 sDone.mem (arena1 + offAcc) = some acc' := by
    change X86Int.rd64 (arenaMem d0' d1' d2') (arena1 + offAcc) = some acc'
    rw [rd64_arena1 hl0 hl1 hoffA, setArenaState_rdAcc hd1.1 hac]
  have ha2 : X86Int.rd64 sDone.mem (arena2 + offAcc) = some acc' := by
    change X86Int.rd64 (arenaMem d0' d1' d2') (arena2 + offAcc) = some acc'
    rw [rd64_arena2 hl0 hl1 hl2 hoffA, setArenaState_rdAcc hd2.1 hac]
  have hs0 : X86Int.rd64 sDone.mem (arena0 + offStep) = some step' := by
    change X86Int.rd64 (arenaMem d0' d1' d2') (arena0 + offStep) = some step'
    rw [rd64_arena0 hl0 hoffS, setArenaState_rdStep hd0.1 hs1]
  have hs1' : X86Int.rd64 sDone.mem (arena1 + offStep) = some step' := by
    change X86Int.rd64 (arenaMem d0' d1' d2') (arena1 + offStep) = some step'
    rw [rd64_arena1 hl0 hl1 hoffS, setArenaState_rdStep hd1.1 hs1]
  have hs2 : X86Int.rd64 sDone.mem (arena2 + offStep) = some step' := by
    change X86Int.rd64 (arenaMem d0' d1' d2') (arena2 + offStep) = some step'
    rw [rd64_arena2 hl0 hl1 hl2 hoffS, setArenaState_rdStep hd2.1 hs1]
  have hv0 : X86Int.rd64 sDone.mem (arena0 + offVal w) = some v_new := by
    change X86Int.rd64 (arenaMem d0' d1' d2') (arena0 + offVal w) = some v_new
    rw [rd64_arena0 hl0 hoffV, setArenaState_rdVal hd0.1 hw hvnew]
  have hv1 : X86Int.rd64 sDone.mem (arena1 + offVal w) = some v_new := by
    change X86Int.rd64 (arenaMem d0' d1' d2') (arena1 + offVal w) = some v_new
    rw [rd64_arena1 hl0 hl1 hoffV, setArenaState_rdVal hd1.1 hw hvnew]
  have hv2 : X86Int.rd64 sDone.mem (arena2 + offVal w) = some v_new := by
    change X86Int.rd64 (arenaMem d0' d1' d2') (arena2 + offVal w) = some v_new
    rw [rd64_arena2 hl0 hl1 hl2 hoffV, setArenaState_rdVal hd2.1 hw hvnew]
  have hl_0 : X86Int.rd64 sDone.mem (arena0 + offLast w) = some step' := by
    change X86Int.rd64 (arenaMem d0' d1' d2') (arena0 + offLast w) = some step'
    rw [rd64_arena0 hl0 hoffL, setArenaState_rdLast hd0.1 hw hs1]
  have hl_1 : X86Int.rd64 sDone.mem (arena1 + offLast w) = some step' := by
    change X86Int.rd64 (arenaMem d0' d1' d2') (arena1 + offLast w) = some step'
    rw [rd64_arena1 hl0 hl1 hoffL, setArenaState_rdLast hd1.1 hw hs1]
  have hl_2 : X86Int.rd64 sDone.mem (arena2 + offLast w) = some step' := by
    change X86Int.rd64 (arenaMem d0' d1' d2') (arena2 + offLast w) = some step'
    rw [rd64_arena2 hl0 hl1 hl2 hoffL, setArenaState_rdLast hd2.1 hw hs1]
  refine ⟨hrun, ha0, ha1, ha2, hs0, hs1', hs2, hv0, hv1, hv2, hl_0, hl_1, hl_2, ?_, ?_⟩
  · rw [ha0, ha1, ha2]; simp [TMR.vote]
  · intro w' hw' hne
    have hoffV' : offVal w' + 8 ≤ arenaSize := by
      simp [offVal, arenaSize, X86Env.page, maxSlots] at *; omega
    have hoffL' : offLast w' + 8 ≤ arenaSize := by
      simp [offLast, arenaSize, X86Env.page, maxSlots] at *; omega
    have hsf' : SingleArenaFault (setArenaState d w step' acc' v_new step') d0' d1' d2' := by
      rcases hsf with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
      · exact Or.inl ⟨rfl, rfl⟩
      · exact Or.inr (Or.inl ⟨rfl, rfl⟩)
      · exact Or.inr (Or.inr ⟨rfl, rfl⟩)
    constructor
    · change TMR.vote
        ((X86Int.rd64 (arenaMem d0' d1' d2') (arena0 + offVal w')).getD 0)
        ((X86Int.rd64 (arenaMem d0' d1' d2') (arena1 + offVal w')).getD 0)
        ((X86Int.rd64 (arenaMem d0' d1' d2') (arena2 + offVal w')).getD 0) = some (rdWord d (offVal w'))
      rw [rd64_arena0 hl0 hoffV', rd64_arena1 hl0 hl1 hoffV', rd64_arena2 hl0 hl1 hl2 hoffV',
        Option.getD_some, Option.getD_some, Option.getD_some,
        vote_single_fault hsf' (offVal w'), setArenaState_rdOtherVal hne.symm]
    · change TMR.vote
        ((X86Int.rd64 (arenaMem d0' d1' d2') (arena0 + offLast w')).getD 0)
        ((X86Int.rd64 (arenaMem d0' d1' d2') (arena1 + offLast w')).getD 0)
        ((X86Int.rd64 (arenaMem d0' d1' d2') (arena2 + offLast w')).getD 0) = some (rdWord d (offLast w'))
      rw [rd64_arena0 hl0 hoffL', rd64_arena1 hl0 hl1 hoffL', rd64_arena2 hl0 hl1 hl2 hoffL',
        Option.getD_some, Option.getD_some, Option.getD_some,
        vote_single_fault hsf' (offLast w'), setArenaState_rdOtherLast hne.symm]

private theorem arenaDataOk_modify_xor {d : List Nat} (hd : ArenaDataOk d) (j b : Nat) (hb : b < 8) :
    ArenaDataOk (d.modify j fun v => Nat.xor v (2 ^ b)) := by
  refine ⟨by rw [List.length_modify, hd.1], ?_⟩
  intro x hx
  obtain ⟨idx, hidx, hget⟩ := List.mem_iff_getElem.1 hx
  rw [List.getElem_modify] at hget
  split at hget
  · subst hget
    rw [List.length_modify] at hidx
    exact Nat.xor_lt_two_pow (hd.2 d[idx] (List.getElem_mem hidx))
      (Nat.pow_lt_pow_right (by decide) hb)
  · subst hget
    rw [List.length_modify] at hidx
    exact hd.2 d[idx] (List.getElem_mem hidx)

private theorem arenaMem_flipMem {s : X86Int.St} {d0 d1 d2 : List Nat}
    (hs : s.mem = arenaMem d0 d1 d2)
    (hd0 : ArenaDataOk d0) (hd1 : ArenaDataOk d1) (hd2 : ArenaDataOk d2)
    {s' : X86Int.St} (hs' : s' ∈ X86Int.memFlips s) :
    ∃ d0' d1' d2', ArenaDataOk d0' ∧ ArenaDataOk d1' ∧ ArenaDataOk d2' ∧
      ((d1' = d1 ∧ d2' = d2) ∨ (d0' = d0 ∧ d2' = d2) ∨ (d0' = d0 ∧ d1' = d1)) ∧
      s' = { s with mem := arenaMem d0' d1' d2' } := by
  simp only [X86Int.memFlips, List.mem_flatMap, List.mem_map, List.mem_range] at hs'
  obtain ⟨i, hi, j, hj, b, hb, rfl⟩ := hs'
  rw [hs] at hi
  simp only [arenaMem, List.length_cons, List.length_nil] at hi
  rcases (by omega : i = 0 ∨ i = 1 ∨ i = 2) with rfl | rfl | rfl
  · refine ⟨d0.modify j fun v => Nat.xor v (2 ^ b), d1, d2,
      arenaDataOk_modify_xor hd0 j b hb, hd1, hd2, Or.inl ⟨rfl, rfl⟩, ?_⟩
    simp [X86Int.St.flipMem, hs, arenaMem, arenaRegion, List.modify]
  · refine ⟨d0, d1.modify j fun v => Nat.xor v (2 ^ b), d2,
      hd0, arenaDataOk_modify_xor hd1 j b hb, hd2, Or.inr (Or.inl ⟨rfl, rfl⟩), ?_⟩
    simp [X86Int.St.flipMem, hs, arenaMem, arenaRegion, List.modify]
  · refine ⟨d0, d1, d2.modify j fun v => Nat.xor v (2 ^ b),
      hd0, hd1, arenaDataOk_modify_xor hd2 j b hb, Or.inr (Or.inr ⟨rfl, rfl⟩), ?_⟩
    simp [X86Int.St.flipMem, hs, arenaMem, arenaRegion, List.modify]

/-- Physical `X86Int.memFlips` Single-Fault Certificate: for EVERY single-bit memory flip `s'`
in `X86Int.memFlips` hitting clean initial state `memExposureInit s w v_new d d d`, running
`memExposureIm` for `memExposureFuel` steps repairs the bit flip and writes exact `acc'`
to all 3 arenas (`X86Int.rd64` matches `acc_v + delta^2`), with zero underestimation. -/
theorem sdc_cert_zero_memFlips (s : X86Int.St) {w v_new : Nat} (hw : w < maxSlots)
    (hvnew : v_new < X86Env.W) {d : List Nat} (hd : ArenaDataOk d)
    (hle : rdWord d (offLast w) ≤ rdWord d offStep)
    (hs1 : rdWord d offStep + 1 < X86Env.W)
    (hsq : (rdWord d offStep - rdWord d (offLast w) + 1) *
      (rdWord d offStep - rdWord d (offLast w) + 1) < X86Env.W)
    (hac : rdWord d offAcc + (rdWord d offStep - rdWord d (offLast w) + 1) *
      (rdWord d offStep - rdWord d (offLast w) + 1) < X86Env.W)
    {s' : X86Int.St} (hs' : s' ∈ X86Int.memFlips (memExposureInit s w v_new d d d)) :
    ∃ d0' d1' d2',
      let step_v := rdWord d offStep
      let acc_v := rdWord d offAcc
      let last_v := rdWord d (offLast w)
      let delta := step_v - last_v + 1
      let step' := step_v + 1
      let acc' := acc_v + delta * delta
      let sDone := memExposureDone s w v_new step_v acc_v last_v d0' d1' d2'
      X86Int.run memExposureIm memExposureFuel (.ok s') = .ok sDone ∧
      X86Int.rd64 sDone.mem (arena0 + offAcc) = some acc' ∧
      X86Int.rd64 sDone.mem (arena1 + offAcc) = some acc' ∧
      X86Int.rd64 sDone.mem (arena2 + offAcc) = some acc' ∧
      X86Int.rd64 sDone.mem (arena0 + offStep) = some step' ∧
      X86Int.rd64 sDone.mem (arena1 + offStep) = some step' ∧
      X86Int.rd64 sDone.mem (arena2 + offStep) = some step' ∧
      X86Int.rd64 sDone.mem (arena0 + offVal w) = some v_new ∧
      X86Int.rd64 sDone.mem (arena1 + offVal w) = some v_new ∧
      X86Int.rd64 sDone.mem (arena2 + offVal w) = some v_new ∧
      X86Int.rd64 sDone.mem (arena0 + offLast w) = some step' ∧
      X86Int.rd64 sDone.mem (arena1 + offLast w) = some step' ∧
      X86Int.rd64 sDone.mem (arena2 + offLast w) = some step' := by
  obtain ⟨d0', d1', d2', hd0', hd1', hd2', hsf, hsEq⟩ :=
    arenaMem_flipMem rfl hd hd hd hs'
  have hinit : s' = memExposureInit s w v_new d0' d1' d2' := hsEq
  have hcert := sdc_cert_zero s hw hvnew hd hd0' hd1' hd2' hsf hle hs1 hsq hac
  refine ⟨d0', d1', d2', ?_⟩
  rw [hinit]
  exact ⟨hcert.1, hcert.2.1, hcert.2.2.1, hcert.2.2.2.1,
    hcert.2.2.2.2.1, hcert.2.2.2.2.2.1, hcert.2.2.2.2.2.2.1,
    hcert.2.2.2.2.2.2.2.1, hcert.2.2.2.2.2.2.2.2.1, hcert.2.2.2.2.2.2.2.2.2.1,
    hcert.2.2.2.2.2.2.2.2.2.2.1, hcert.2.2.2.2.2.2.2.2.2.2.2.1, hcert.2.2.2.2.2.2.2.2.2.2.2.2.1⟩

/-- Post-retirement single-bit `X86Int.memFlips` certificate on clean `memExposureDone`:
any 1-bit flip across all 3 arenas after retirement leaves `(acc', step', v_new, step')`
majority-recoverable across `(arena0, arena1, arena2)` with zero underestimation. -/
theorem sdc_cert_zero_post_memFlips (s : X86Int.St) {w v_new : Nat} (hw : w < maxSlots)
    (hvnew : v_new < X86Env.W) {d : List Nat} (hd : ArenaDataOk d)
    (hs1 : rdWord d offStep + 1 < X86Env.W)
    (hac : rdWord d offAcc + (rdWord d offStep - rdWord d (offLast w) + 1) *
      (rdWord d offStep - rdWord d (offLast w) + 1) < X86Env.W)
    {s' : X86Int.St}
    (hs' : s' ∈ X86Int.memFlips
      (memExposureDone s w v_new (rdWord d offStep) (rdWord d offAcc) (rdWord d (offLast w)) d d d)) :
    let step_v := rdWord d offStep
    let acc_v := rdWord d offAcc
    let last_v := rdWord d (offLast w)
    let delta := step_v - last_v + 1
    let step' := step_v + 1
    let acc' := acc_v + delta * delta
    TMR.vote ((X86Int.rd64 s'.mem (arena0 + offAcc)).getD 0)
      ((X86Int.rd64 s'.mem (arena1 + offAcc)).getD 0)
      ((X86Int.rd64 s'.mem (arena2 + offAcc)).getD 0) = some acc' ∧
    TMR.vote ((X86Int.rd64 s'.mem (arena0 + offStep)).getD 0)
      ((X86Int.rd64 s'.mem (arena1 + offStep)).getD 0)
      ((X86Int.rd64 s'.mem (arena2 + offStep)).getD 0) = some step' ∧
    TMR.vote ((X86Int.rd64 s'.mem (arena0 + offVal w)).getD 0)
      ((X86Int.rd64 s'.mem (arena1 + offVal w)).getD 0)
      ((X86Int.rd64 s'.mem (arena2 + offVal w)).getD 0) = some v_new ∧
    TMR.vote ((X86Int.rd64 s'.mem (arena0 + offLast w)).getD 0)
      ((X86Int.rd64 s'.mem (arena1 + offLast w)).getD 0)
      ((X86Int.rd64 s'.mem (arena2 + offLast w)).getD 0) = some step' := by
  intro step_v acc_v last_v delta step' acc'
  let dClean := setArenaState d w step' acc' v_new step'
  have hdClean : ArenaDataOk dClean := hd.setArenaState w step' acc' v_new step'
  obtain ⟨d0', d1', d2', hd0', hd1', hd2', hsf, hsEq⟩ :=
    arenaMem_flipMem rfl hdClean hdClean hdClean hs'
  have hmem : s'.mem = arenaMem d0' d1' d2' := by rw [hsEq]
  have hoffS : offStep + 8 ≤ arenaSize := by decide
  have hoffA : offAcc + 8 ≤ arenaSize := by decide
  have hoffV : offVal w + 8 ≤ arenaSize := by simp [offVal, arenaSize, X86Env.page, maxSlots] at *; omega
  have hoffL : offLast w + 8 ≤ arenaSize := by simp [offLast, arenaSize, X86Env.page, maxSlots] at *; omega
  rw [hmem]
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [rd64_arena0 hd0'.1 hoffA, rd64_arena1 hd0'.1 hd1'.1 hoffA, rd64_arena2 hd0'.1 hd1'.1 hd2'.1 hoffA,
      Option.getD_some, Option.getD_some, Option.getD_some,
      vote_single_fault hsf offAcc, setArenaState_rdAcc hd.1 hac]
  · rw [rd64_arena0 hd0'.1 hoffS, rd64_arena1 hd0'.1 hd1'.1 hoffS, rd64_arena2 hd0'.1 hd1'.1 hd2'.1 hoffS,
      Option.getD_some, Option.getD_some, Option.getD_some,
      vote_single_fault hsf offStep, setArenaState_rdStep hd.1 hs1]
  · rw [rd64_arena0 hd0'.1 hoffV, rd64_arena1 hd0'.1 hd1'.1 hoffV, rd64_arena2 hd0'.1 hd1'.1 hd2'.1 hoffV,
      Option.getD_some, Option.getD_some, Option.getD_some,
      vote_single_fault hsf (offVal w), setArenaState_rdVal hd.1 hw hvnew]
  · rw [rd64_arena0 hd0'.1 hoffL, rd64_arena1 hd0'.1 hd1'.1 hoffL, rd64_arena2 hd0'.1 hd1'.1 hd2'.1 hoffL,
      Option.getD_some, Option.getD_some, Option.getD_some,
      vote_single_fault hsf (offLast w), setArenaState_rdLast hd.1 hw hs1]

/-! ### Physical Fault Semantics (`X86Int.prun`) & Exposed Bounds -/

private theorem rates_state_q (r : X86Int.Rates) (hr : r.ok) :
    r.fetch + (r.state hr).q = r.total := rfl

private theorem rates_total_nonneg (r : X86Int.Rates) (hr : r.ok) : 0 ≤ r.total :=
  X86Int.nonneg_add hr.1.1 (X86Int.nonneg_add hr.2.1.1 (X86Int.nonneg_add hr.2.2.1.1 hr.2.2.2.1))

/-- Physical run of `memExposureIm` (`memExposureFuel` cycles) from `memExposureInit`
under `X86Int.prun` (`memExposureAlts`, `Rates.state`, `Rates.fetch`). -/
def memExposurePhys (r : X86Int.Rates) (hr : r.ok) (s : X86Int.St) (w v_new : Nat)
    (d0 d1 d2 : List Nat) : PDist.PDF X86Int.Out :=
  X86Int.prun memExposureAlts (r.state hr) r.fetch memExposureIm memExposureFuel
    (.ok (memExposureInit s w v_new d0 d1 d2))

theorem memExposure_phys_ok (r : X86Int.Rates) (hr : r.ok) (s : X86Int.St) (w v_new : Nat)
    (d0 d1 d2 : List Nat) : (memExposurePhys r hr s w v_new d0 d1 d2).ok :=
  X86Int.prun_ok memExposureAlts (r.state hr) hr.1 memExposureIm memExposureFuel _

theorem memExposure_phys_clean (r : X86Int.Rates) (hr : r.ok) (s : X86Int.St) (w v_new : Nat)
    (d0 d1 d2 : List Nat) :
    PDist.faultFree (memExposurePhys r hr s w v_new d0 d1 d2) =
      [X86Int.run memExposureIm memExposureFuel (.ok (memExposureInit s w v_new d0 d1 d2))] :=
  X86Int.faultFree_prun memExposureAlts (r.state hr) r.fetch memExposureIm memExposureFuel _

theorem memExposure_phys_clean_majority (r : X86Int.Rates) (hr : r.ok) (s : X86Int.St)
    {w v_new : Nat} (hw : w < maxSlots) (hvnew : v_new < X86Env.W)
    {d0 d1 d2 : List Nat} (hd0 : ArenaDataOk d0) (hd1 : ArenaDataOk d1) (hd2 : ArenaDataOk d2)
    {val_v last_v step_v acc_v : Nat}
    (hval : TMR.vote (rdWord d0 (offVal w)) (rdWord d1 (offVal w)) (rdWord d2 (offVal w)) = some val_v)
    (hlast : TMR.vote (rdWord d0 (offLast w)) (rdWord d1 (offLast w)) (rdWord d2 (offLast w)) = some last_v)
    (hstep : TMR.vote (rdWord d0 offStep) (rdWord d1 offStep) (rdWord d2 offStep) = some step_v)
    (hacc : TMR.vote (rdWord d0 offAcc) (rdWord d1 offAcc) (rdWord d2 offAcc) = some acc_v)
    (hle : last_v ≤ step_v) (hs1 : step_v + 1 < X86Env.W)
    (hsq : (step_v - last_v + 1) * (step_v - last_v + 1) < X86Env.W)
    (hac : acc_v + (step_v - last_v + 1) * (step_v - last_v + 1) < X86Env.W) :
    PDist.faultFree (memExposurePhys r hr s w v_new d0 d1 d2) =
      [.ok (memExposureDone s w v_new step_v acc_v last_v d0 d1 d2)] := by
  rw [memExposure_phys_clean, memExposure_arch_majority s hw hvnew hd0 hd1 hd2 hval hlast hstep hacc hle hs1 hsq hac]

theorem memExposure_phys_clean_refuse (r : X86Int.Rates) (hr : r.ok) (s : X86Int.St)
    {w v_new : Nat} (hw : w < maxSlots)
    {d0 d1 d2 : List Nat} (hd0 : ArenaDataOk d0) (hd1 : ArenaDataOk d1) (hd2 : ArenaDataOk d2)
    (hsplit :
      TMR.vote (rdWord d0 (offVal w)) (rdWord d1 (offVal w)) (rdWord d2 (offVal w)) = none ∨
      TMR.vote (rdWord d0 (offLast w)) (rdWord d1 (offLast w)) (rdWord d2 (offLast w)) = none ∨
      TMR.vote (rdWord d0 offStep) (rdWord d1 offStep) (rdWord d2 offStep) = none ∨
      TMR.vote (rdWord d0 offAcc) (rdWord d1 offAcc) (rdWord d2 offAcc) = none) :
    PDist.faultFree (memExposurePhys r hr s w v_new d0 d1 d2) = [.exit 3] := by
  rw [memExposure_phys_clean, memExposure_arch_refuse s hw hd0 hd1 hd2 hsplit]

theorem memExposure_phys_clean_overflow (r : X86Int.Rates) (hr : r.ok) (s : X86Int.St)
    {w v_new : Nat} (hw : w < maxSlots)
    {d0 d1 d2 : List Nat} (hd0 : ArenaDataOk d0) (hd1 : ArenaDataOk d1) (hd2 : ArenaDataOk d2)
    {val_v last_v step_v acc_v : Nat}
    (hval : TMR.vote (rdWord d0 (offVal w)) (rdWord d1 (offVal w)) (rdWord d2 (offVal w)) = some val_v)
    (hlast : TMR.vote (rdWord d0 (offLast w)) (rdWord d1 (offLast w)) (rdWord d2 (offLast w)) = some last_v)
    (hstep : TMR.vote (rdWord d0 offStep) (rdWord d1 offStep) (rdWord d2 offStep) = some step_v)
    (hacc : TMR.vote (rdWord d0 offAcc) (rdWord d1 offAcc) (rdWord d2 offAcc) = some acc_v)
    (hov : step_v < last_v ∨ X86Env.W ≤ step_v + 1 ∨
      X86Env.W ≤ (step_v - last_v + 1) * (step_v - last_v + 1) ∨
      X86Env.W ≤ acc_v + (step_v - last_v + 1) * (step_v - last_v + 1)) :
    PDist.faultFree (memExposurePhys r hr s w v_new d0 d1 d2) = [.exit 2] := by
  rw [memExposure_phys_clean, memExposure_arch_overflow s hw hd0 hd1 hd2 hval hlast hstep hacc hov]

theorem memExposure_phys_fm_le (r : X86Int.Rates) (hr : r.ok) (s : X86Int.St) (w v_new : Nat)
    (d0 d1 d2 : List Nat) :
    PDist.fm (memExposurePhys r hr s w v_new d0 d1 d2) ≤ (memExposureFuel : Rat) * r.total := by
  simpa [memExposurePhys, rates_state_q] using
    X86Int.fm_prun memExposureAlts (r.state hr) hr.1 memExposureIm memExposureFuel
      (.ok (memExposureInit s w v_new d0 d1 d2))

private theorem rdBytes_modify_wrWord_lt256 {d : List Nat} (hd : d.length = arenaSize)
    (off v : Nat) (hoff : off + 8 ≤ arenaSize) (j b : Nat) (hb : b < 8) :
    ∀ n a, off ≤ a → a + n ≤ off + 8 →
      ∀ x ∈ rdBytes ((wrWord d off v).modify j fun u => Nat.xor u (2 ^ b)) a n, x < 256
  | 0, _, _, _, _, hx => by simp [rdBytes] at hx
  | n + 1, a, haLo, haHi, x, hx => by
    simp only [rdBytes, List.mem_cons] at hx
    rcases hx with rfl | hx
    · have hlen : (wrWord d off v).length = arenaSize := by rw [wrWord_length, hd]
      have haLt : a < (wrWord d off v).length := by omega
      have hget : ((wrWord d off v).modify j fun u => Nat.xor u (2 ^ b))[a]? =
          some (if j = a then Nat.xor (wrWord d off v)[a] (2 ^ b) else (wrWord d off v)[a]) := by
        rw [List.getElem?_eq_getElem (by rw [List.length_modify]; exact haLt), List.getElem_modify]
      have hbase_lt : (wrWord d off v)[a] < 256 := by
        obtain ⟨k, hk, hakeq⟩ : ∃ k < 8, a = off + k := ⟨a - off, by omega, by omega⟩
        have hsome : (wrWord d off v)[a]? = some (wrWord d off v)[a] :=
          List.getElem?_eq_getElem haLt
        have hgetw := getElem?_wrWord_in (d := d) (v := v) hk (by rw [hd]; exact hoff)
        rw [← hakeq, hsome, Option.some.injEq] at hgetw
        rw [hgetw]
        exact Nat.mod_lt _ (by decide)
      rw [hget]
      simp only [Option.getD_some]
      split
      · exact Nat.xor_lt_two_pow hbase_lt (Nat.pow_lt_pow_right (by decide) hb)
      · exact hbase_lt
    · exact rdBytes_modify_wrWord_lt256 hd off v hoff j b hb n (a + 1) (by omega) (by omega) x hx

private theorem mem_unif {β : Type} {p : Rat} {c : β} {xs : List β} {y : (β × Bool) × Rat}
    (hy : y ∈ X86Int.unif p c xs) : y.1.1 = c ∨ y.1.1 ∈ xs := by
  cases xs with
  | nil =>
    simp only [X86Int.unif, PDist.pureF, List.mem_cons, List.not_mem_nil, or_false] at hy
    subst hy; exact Or.inl rfl
  | cons x r =>
    simp only [X86Int.unif, List.mem_cons, List.mem_map] at hy
    rcases hy with h | ⟨z, hz, h⟩
    · rw [h]; exact Or.inl rfl
    · rw [← h]; exact Or.inr (by simpa using hz)

theorem idleArenaWordChannel_lt_W {s0 : X86Int.St} {d0 d1 d2 : List Nat}
    {h0 : d0.length = arenaSize} {off : Nat} {hoff : off + 8 ≤ arenaSize}
    {r : X86Int.Rates} {hr : r.ok} (z : Nat) :
    (idleArenaWordChannel s0 d0 d1 d2 h0 off hoff r hr).f z < X86Env.W ∧
    ∀ y ∈ (idleArenaWordChannel s0 d0 d1 d2 h0 off hoff r hr).step z, y.1.1 < X86Env.W := by
  refine ⟨Nat.mod_lt z (by decide), ?_⟩
  intro y hy
  simp only [idleArenaWordChannel, arenaWordFlipsChannel, X86Int.relabel, List.mem_map] at hy
  obtain ⟨u, hu, rfl⟩ := hy
  rcases mem_unif hu with hst | hflips
  · rw [hst]
    simp only [getArena0Word, setArena0Word]
    have hlen : (wrWord d0 off (z % X86Env.W)).length = arenaSize := by rw [wrWord_length, h0]
    rw [rd64_arena0 hlen hoff,
      rdWord_wrWord_same (d := d0) (by rw [h0]; exact hoff) (Nat.mod_lt _ (by decide))]
    exact Nat.mod_lt z (by decide)
  · simp only [X86Int.memFlips, List.mem_flatMap, List.mem_map, List.mem_range] at hflips
    obtain ⟨i, hi, j, hj, b, hb, hflipEq⟩ := hflips
    rw [← hflipEq]
    simp only [setArena0Word, arenaMem, List.length_cons, List.length_nil] at hi
    rcases (by omega : i = 0 ∨ i = 1 ∨ i = 2) with rfl | rfl | rfl
    · have hlenm : ((wrWord d0 off (z % X86Env.W)).modify j fun v => Nat.xor v (2 ^ b)).length =
          arenaSize := by rw [List.length_modify, wrWord_length, h0]
      have hrd : X86Int.rd64
          ((setArena0Word s0 d0 d1 d2 off (z % X86Env.W)).flipMem 0 j b).mem (arena0 + off) =
          some (rdWord ((wrWord d0 off (z % X86Env.W)).modify j fun v => Nat.xor v (2 ^ b)) off) := by
        change X86Int.rd64
          (arenaMem ((wrWord d0 off (z % X86Env.W)).modify j fun v => Nat.xor v (2 ^ b)) d1 d2)
          (arena0 + off) = _
        exact rd64_arena0 hlenm hoff
      simp only [getArena0Word, hrd, Option.getD_some]
      have h256 := rdBytes_modify_wrWord_lt256 h0 off (z % X86Env.W) hoff j b hb 8 off
        (Nat.le_refl _) (by omega)
      have hlt := leVal_lt_256_pow
        (rdBytes ((wrWord d0 off (z % X86Env.W)).modify j fun v => Nat.xor v (2 ^ b)) off 8) h256
      rw [rdBytes_length] at hlt
      simpa [rdWord, X86Env.W] using hlt
    · have hlen : (wrWord d0 off (z % X86Env.W)).length = arenaSize := by rw [wrWord_length, h0]
      have hrd : X86Int.rd64
          ((setArena0Word s0 d0 d1 d2 off (z % X86Env.W)).flipMem 1 j b).mem (arena0 + off) =
          some (rdWord (wrWord d0 off (z % X86Env.W)) off) := by
        change X86Int.rd64
          (arenaMem (wrWord d0 off (z % X86Env.W)) (d1.modify j fun v => Nat.xor v (2 ^ b)) d2)
          (arena0 + off) = _
        exact rd64_arena0 hlen hoff
      simp only [getArena0Word, hrd, Option.getD_some,
        rdWord_wrWord_same (d := d0) (by rw [h0]; exact hoff) (Nat.mod_lt _ (by decide))]
      exact Nat.mod_lt z (by decide)
    · have hlen : (wrWord d0 off (z % X86Env.W)).length = arenaSize := by rw [wrWord_length, h0]
      have hrd : X86Int.rd64
          ((setArena0Word s0 d0 d1 d2 off (z % X86Env.W)).flipMem 2 j b).mem (arena0 + off) =
          some (rdWord (wrWord d0 off (z % X86Env.W)) off) := by
        change X86Int.rd64
          (arenaMem (wrWord d0 off (z % X86Env.W)) d1 (d2.modify j fun v => Nat.xor v (2 ^ b)))
          (arena0 + off) = _
        exact rd64_arena0 hlen hoff
      simp only [getArena0Word, hrd, Option.getD_some,
        rdWord_wrWord_same (d := d0) (by rw [h0]; exact hoff) (Nat.mod_lt _ (by decide))]
      exact Nat.mod_lt z (by decide)

/-- Helper combining 3 independent draws from `D : PDist.PDF Nat` against expected clean
majority `g`, passing actual copies `(a, b, c)` into continuation `cont` and marking any
2-of-3 vote mismatch (`vote a b c ≠ some g`) as faulty. -/
def triRun3 (D : PDist.PDF Nat) (g : Nat) (cont : Nat → Nat → Nat → PDist.PDF X86Int.Out) :
    PDist.PDF X86Int.Out :=
  TMR.bindW D fun a =>
    TMR.bindW D fun b =>
      TMR.bindW D fun c =>
        PDist.tagW (decide (TMR.vote a.1 b.1 c.1 ≠ some g)) 1 (cont a.1 b.1 c.1)

private theorem sumW_add_const {α : Type} (D : PDist.PDF α) (φ : α × Bool → Rat) (k : Rat) :
    TMR.sumW D (fun x => φ x + k) = TMR.sumW D φ + PDist.mass D * k := by
  induction D with
  | nil => show (0 : Rat) = 0 + 0 * k; rw [Rat.zero_mul, Rat.add_zero]
  | cons x r ih =>
    show x.2 * (φ x.1 + k) + TMR.sumW r (fun y => φ y + k) =
      (x.2 * φ x.1 + TMR.sumW r φ) + (x.2 + PDist.mass r) * k
    rw [ih, Rat.mul_add, Rat.add_mul]
    calc (x.2 * φ x.1 + x.2 * k) + (TMR.sumW r φ + PDist.mass r * k)
        = x.2 * φ x.1 + (x.2 * k + (TMR.sumW r φ + PDist.mass r * k)) := Rat.add_assoc _ _ _
      _ = x.2 * φ x.1 + ((x.2 * k + TMR.sumW r φ) + PDist.mass r * k) := by rw [Rat.add_assoc (x.2 * k)]
      _ = x.2 * φ x.1 + ((TMR.sumW r φ + x.2 * k) + PDist.mass r * k) := by rw [Rat.add_comm (x.2 * k)]
      _ = x.2 * φ x.1 + (TMR.sumW r φ + (x.2 * k + PDist.mass r * k)) := by rw [Rat.add_assoc (TMR.sumW r φ)]
      _ = (x.2 * φ x.1 + TMR.sumW r φ) + (x.2 * k + PDist.mass r * k) := (Rat.add_assoc _ _ _).symm

private theorem sumW_add_const_ok {α : Type} {D : PDist.PDF α} (hD : D.ok)
    (φ : α × Bool → Rat) (k : Rat) :
    TMR.sumW D (fun x => φ x + k) = TMR.sumW D φ + k := by
  rw [sumW_add_const, hD.2, Rat.one_mul]

private theorem fm_tri_eq_sumW {σ : Type} [DecidableEq σ] (D : PDist.PDF σ) (g : σ) :
    PDist.fm (TMR.tri D g) =
      TMR.sumW D (fun a => TMR.sumW D (fun b => TMR.sumW D (fun c =>
        TMR.ι (decide (TMR.vote a.1 b.1 c.1 ≠ some g))))) := by
  have hm : ∀ v : Option σ, PDist.fm (TMR.mark g v) = TMR.ι (decide (v ≠ some g)) := fun v => by
    show (if decide (v ≠ some g) = true then (1 : Rat) else 0) + 0 = _
    rw [Rat.add_zero]; rfl
  simp only [TMR.tri, TMR.fm_bindW, hm]

private theorem exists_clean_of_faultFree_eq {α : Type} {D : PDist.PDF α} {g : α}
    (h : PDist.faultFree D = [g]) : ∃ a ∈ D, a.1.1 = g ∧ a.1.2 = false := by
  induction D with
  | nil => simp [PDist.faultFree] at h
  | cons x r ih =>
    obtain ⟨⟨a, t⟩, w⟩ := x
    cases t
    · simp only [PDist.faultFree, Bool.false_eq_true, ↓reduceIte, List.cons.injEq] at h
      exact ⟨((a, false), w), List.mem_cons_self .., h.1, rfl⟩
    · simp only [PDist.faultFree, ↓reduceIte] at h
      obtain ⟨y, hy, ha, ht⟩ := ih h
      exact ⟨y, List.mem_cons_of_mem _ hy, ha, ht⟩

private theorem mem_faultFree_bindW_intro {α β : Type} {D : PDist.PDF α}
    {K : α × Bool → PDist.PDF β} {a : (α × Bool) × Rat} {z : β}
    (ha : a ∈ D) (hz : z ∈ PDist.faultFree (K a.1)) :
    z ∈ PDist.faultFree (TMR.bindW D K) := by
  induction D with
  | nil => simp at ha
  | cons x r ih =>
    simp only [TMR.bindW, PDist.faultFree_append, TMR.faultFree_scaleL, List.mem_append]
    rcases List.mem_cons.1 ha with rfl | hr
    · exact Or.inl hz
    · exact Or.inr (ih hr)

private theorem vote_self (g : Nat) : TMR.vote g g g = some g := by
  unfold TMR.vote; simp

theorem triRun3_ok {D : PDist.PDF Nat} (hD : D.ok) (g : Nat)
    {cont : Nat → Nat → Nat → PDist.PDF X86Int.Out} (hcont : ∀ a b c, (cont a b c).ok) :
    (triRun3 D g cont).ok := by
  refine ⟨?_, ?_⟩
  · refine TMR.bindW_nonneg hD.1 fun a => TMR.bindW_nonneg hD.1 fun b =>
      TMR.bindW_nonneg hD.1 fun c => ?_
    intro y hy
    simp only [PDist.tagW, List.mem_map] at hy
    obtain ⟨z, hz, rfl⟩ := hy
    have hz0 : 0 ≤ z.2 := (hcont a.1 b.1 c.1).1 z hz
    simpa [Rat.one_mul] using hz0
  · have hm : ∀ a b c, PDist.mass
        (PDist.tagW (decide (TMR.vote a b c ≠ some g)) 1 (cont a b c)) = 1 := by
      intro a b c
      rw [PDist.mass_tagW, (hcont a b c).2, Rat.one_mul]
    simp only [triRun3, TMR.mass_bindW, hm, TMR.sumW_const, hD.2, Rat.one_mul]

theorem triRun3_clean {D : PDist.PDF Nat} (hW : ∀ y ∈ D, y.1.1 < X86Env.W) (g : Nat)
    (cont : Nat → Nat → Nat → PDist.PDF X86Int.Out)
    (P : X86Int.Out → Prop)
    (hP : ∀ a b c, a < X86Env.W → b < X86Env.W → c < X86Env.W →
      TMR.vote a b c = some g → ∀ u ∈ PDist.faultFree (cont a b c), P u) :
    ∀ u ∈ PDist.faultFree (triRun3 D g cont), P u := by
  intro u hu
  obtain ⟨a, ha, hu⟩ := TMR.mem_faultFree_bindW hu
  obtain ⟨b, hb, hu⟩ := TMR.mem_faultFree_bindW hu
  obtain ⟨c, hc, hu⟩ := TMR.mem_faultFree_bindW hu
  rw [PDist.faultFree_tagW] at hu
  split at hu
  · simp at hu
  · next hne =>
    have hv : TMR.vote a.1.1 b.1.1 c.1.1 = some g := by simpa using hne
    exact hP a.1.1 b.1.1 c.1.1 (hW a ha) (hW b hb) (hW c hc) hv u hu

theorem triRun3_mem_clean {D : PDist.PDF Nat} {g : Nat} (hc : PDist.faultFree D = [g])
    (cont : Nat → Nat → Nat → PDist.PDF X86Int.Out) {target : X86Int.Out}
    (ht : target ∈ PDist.faultFree (cont g g g)) :
    target ∈ PDist.faultFree (triRun3 D g cont) := by
  obtain ⟨a, ha, ha1, _⟩ := exists_clean_of_faultFree_eq hc
  have hphys : target ∈
      PDist.faultFree (PDist.tagW (decide (TMR.vote a.1.1 a.1.1 a.1.1 ≠ some g)) 1
        (cont a.1.1 a.1.1 a.1.1)) := by
    rw [ha1]
    have htag : decide (TMR.vote g g g ≠ some g) = false := by simp [vote_self]
    rw [PDist.faultFree_tagW, if_neg (by simp [htag])]
    exact ht
  exact mem_faultFree_bindW_intro ha (mem_faultFree_bindW_intro ha (mem_faultFree_bindW_intro ha hphys))

theorem triRun3_bound_le {D : PDist.PDF Nat} (hD : D.ok) (g : Nat)
    (cont : Nat → Nat → Nat → PDist.PDF X86Int.Out)
    (hcont : ∀ a b c, (cont a b c).ok) {qv : Rat} (hqv0 : 0 ≤ qv)
    (hqv : ∀ a b c, PDist.fm (cont a b c) ≤ qv) :
    PDist.fm (triRun3 D g cont) ≤ PDist.fm (TMR.tri D g) + qv := by
  rw [triRun3, TMR.fm_bindW]
  simp only [TMR.fm_bindW]
  have hcell : ∀ a b c : Nat,
      PDist.fm (PDist.tagW (decide (TMR.vote a b c ≠ some g)) 1 (cont a b c)) ≤
        TMR.ι (decide (TMR.vote a b c ≠ some g)) + qv := by
    intro a b c
    rw [PDist.fm_tagW]
    by_cases ht : decide (TMR.vote a b c ≠ some g) = true
    · rw [if_pos ht, Rat.one_mul, (hcont a b c).2]
      simp only [TMR.ι, ht, ↓reduceIte]
      have := add_le_add' (Rat.le_refl (a := (1 : Rat))) hqv0
      rwa [Rat.add_zero] at this
    · have hfalse : decide (TMR.vote a b c ≠ some g) = false := by
        cases h : decide (TMR.vote a b c ≠ some g) <;> simp_all
      rw [if_neg ht, Rat.one_mul]
      simp only [TMR.ι, hfalse, Bool.false_eq_true, ↓reduceIte, Rat.zero_add]
      exact hqv a b c
  have h1 :
      TMR.sumW D (fun a => TMR.sumW D (fun b => TMR.sumW D (fun c =>
        PDist.fm (PDist.tagW (decide (TMR.vote a.1 b.1 c.1 ≠ some g)) 1
          (cont a.1 b.1 c.1))))) ≤
      TMR.sumW D (fun a => TMR.sumW D (fun b => TMR.sumW D (fun c =>
        TMR.ι (decide (TMR.vote a.1 b.1 c.1 ≠ some g)) + qv))) :=
    TMR.sumW_le hD.1 fun a _ => TMR.sumW_le hD.1 fun b _ => TMR.sumW_le hD.1 fun c _ =>
      hcell a.1.1 b.1.1 c.1.1
  refine Rat.le_trans h1 ?_
  have hinner : ∀ a b : Nat × Bool,
      TMR.sumW D (fun c => TMR.ι (decide (TMR.vote a.1 b.1 c.1 ≠ some g)) + qv) =
        TMR.sumW D (fun c => TMR.ι (decide (TMR.vote a.1 b.1 c.1 ≠ some g))) + qv :=
    fun a b => sumW_add_const_ok hD _ _
  have hmid : ∀ a : Nat × Bool,
      TMR.sumW D (fun b => TMR.sumW D (fun c => TMR.ι (decide (TMR.vote a.1 b.1 c.1 ≠ some g)) + qv)) =
        TMR.sumW D (fun b => TMR.sumW D (fun c => TMR.ι (decide (TMR.vote a.1 b.1 c.1 ≠ some g)))) + qv := by
    intro a
    rw [TMR.sumW_congr D (fun b => hinner a b), sumW_add_const_ok hD]
  rw [TMR.sumW_congr D hmid, sumW_add_const_ok hD, fm_tri_eq_sumW]
  exact Rat.le_refl

private theorem mem_bindF {α β : Type} {d : PDist.PDF α} {f : α → PDist.PDF β}
    {y : (β × Bool) × Rat} (hy : y ∈ PDist.bindF d f) :
    ∃ x ∈ d, ∃ z ∈ f x.1.1, y.1.1 = z.1.1 := by
  induction d with
  | nil => simp [PDist.bindF] at hy
  | cons x r ih =>
    simp only [PDist.bindF, List.mem_append] at hy
    rcases hy with hy | hy
    · simp only [PDist.tagW, List.mem_map] at hy
      obtain ⟨z, hz, rfl⟩ := hy
      exact ⟨x, List.mem_cons_self .., z, hz, rfl⟩
    · obtain ⟨w, hw, z, hz, hyeq⟩ := ih hy
      exact ⟨w, List.mem_cons_of_mem _ hw, z, hz, hyeq⟩

private theorem idleK_lt_W_gen (C : TMR.Channel Nat)
    (hstepW : ∀ z, z < X86Env.W → ∀ y ∈ C.step z, y.1.1 < X86Env.W) :
    ∀ k x, x < X86Env.W → ∀ y ∈ TMR.idleK C k x, y.1.1 < X86Env.W
  | 0, x, hx, y, hy => by
    simp only [TMR.idleK, PDist.pureF, List.mem_cons, List.not_mem_nil, or_false] at hy
    subst hy; exact hx
  | k + 1, x, hx, y, hy => by
    simp only [TMR.idleK] at hy
    obtain ⟨a, ha, z, hz, hyeq⟩ := mem_bindF hy
    rw [hyeq]
    exact idleK_lt_W_gen C hstepW k a.1.1 (hstepW x hx a ha) z hz

private theorem exposed_lt_W_gen (C : TMR.Channel Nat)
    (hstepW : ∀ z, z < X86Env.W → ∀ y ∈ C.step z, y.1.1 < X86Env.W)
    (k x : Nat) (hx : x < X86Env.W) :
    ∀ y ∈ TMR.exposed C k x, y.1.1 < X86Env.W := by
  intro y hy
  dsimp only [TMR.exposed] at hy
  obtain ⟨a, ha, z, hz, hyeq⟩ := mem_bindF hy
  rw [hyeq]
  exact hstepW a.1.1 (idleK_lt_W_gen C hstepW k x hx a ha) z hz

private theorem hoffS_le : offStep + 8 ≤ arenaSize := by decide
private theorem hoffA_le : offAcc + 8 ≤ arenaSize := by decide

private theorem hoffV_lt {w : Nat} (hw : w < maxSlots) : offVal w + 8 ≤ arenaSize := by
  have hwm : w < 255 := hw
  unfold offVal arenaSize X86Env.page
  omega

private theorem hoffL_lt {w : Nat} (hw : w < maxSlots) : offLast w + 8 ≤ arenaSize := by
  have hwm : w < 255 := hw
  unfold offLast arenaSize X86Env.page
  omega

private theorem disj_VL (w : Nat) : offVal w + 8 ≤ offLast w ∨ offLast w + 8 ≤ offVal w :=
  Or.inl (by unfold offVal offLast; omega)

private theorem disj_SV (w : Nat) : offStep + 8 ≤ offVal w ∨ offVal w + 8 ≤ offStep :=
  Or.inl (by unfold offStep offVal; omega)

private theorem disj_SL (w : Nat) : offStep + 8 ≤ offLast w ∨ offLast w + 8 ≤ offStep :=
  Or.inl (by unfold offStep offLast; omega)

private theorem disj_AV (w : Nat) : offAcc + 8 ≤ offVal w ∨ offVal w + 8 ≤ offAcc :=
  Or.inl (by unfold offAcc offVal; omega)

private theorem disj_AL (w : Nat) : offAcc + 8 ≤ offLast w ∨ offLast w + 8 ≤ offAcc :=
  Or.inl (by unfold offAcc offLast; omega)

private theorem disj_SA : offStep + 8 ≤ offAcc ∨ offAcc + 8 ≤ offStep :=
  Or.inl (by decide)

/-- Decision-50 exposed physical step: word `val[w]` evolves idly for `k` ticks across
3 independent copies of `idleArenaWordChannel` (under `X86Int.memFlips` at `Rates.mem`),
then runs the 450-cycle physical machine-code kernel `memExposurePhys`. -/
def memExposureExposedPhys (s : X86Int.St) (w v_new : Nat) (d : List Nat) (hd : ArenaDataOk d)
    (hw : w < maxSlots) (val_v k : Nat) (r : X86Int.Rates) (hr : r.ok) :
    PDist.PDF X86Int.Out :=
  let C := idleArenaWordChannel s d d d hd.1 (offVal w) (hoffV_lt hw) r hr
  triRun3 (TMR.exposed C k val_v) val_v fun a b c =>
    memExposurePhys r hr s w v_new
      (wrWord d (offVal w) a) (wrWord d (offVal w) b) (wrWord d (offVal w) c)

private theorem exposure_err_pow (C : TMR.Channel Nat) (k v : Nat) :
    PDist.fm (TMR.tri (TMR.exposed C k v) (TMR.iter C.f (k + 1) v)) ≤
      3 * (((k + 1 : Nat) : Rat) * C.q) ^ 2 := by
  rw [Rat.pow_succ, Rat.pow_one]
  exact TMR.exposure_err C k v

theorem memExposure_exposed_phys_ok (s : X86Int.St) {w v_new : Nat} (hw : w < maxSlots)
    {d : List Nat} (hd : ArenaDataOk d) (val_v k : Nat) (r : X86Int.Rates) (hr : r.ok) :
    (memExposureExposedPhys s w v_new d hd hw val_v k r hr).ok := by
  dsimp only [memExposureExposedPhys]
  exact triRun3_ok
    (TMR.exposed_ok (idleArenaWordChannel s d d d hd.1 (offVal w) (hoffV_lt hw) r hr) k val_v)
    val_v
    (fun a b c => memExposure_phys_ok r hr s w v_new
      (wrWord d (offVal w) a) (wrWord d (offVal w) b) (wrWord d (offVal w) c))

/-- Full physical fault bound (`memExposure_phys_bound`): combining `TMRExposure.exposure_err`
on `idleArenaWordChannel` (`Rates.mem`) with `X86Int.prun` (`memExposureFuel` cycles at `Rates.total`)
yields a valid PDF whose clean branch uniquely reaches `memExposureDone` and whose fault
probability is at most `3 * ((k + 1) * r.mem)^2 + memExposureFuel * r.total`. -/
theorem memExposure_phys_bound (s : X86Int.St) {w v_new : Nat} (hw : w < maxSlots)
    (hvnew : v_new < X86Env.W) {d : List Nat} (hd : ArenaDataOk d)
    (k : Nat) (r : X86Int.Rates) (hr : r.ok)
    (hval_eq : rdWord d (offVal w) < X86Env.W := rdWord_lt_W hd.2 (offVal w))
    (hle : rdWord d (offLast w) ≤ rdWord d offStep)
    (hs1 : rdWord d offStep + 1 < X86Env.W)
    (hsq : (rdWord d offStep - rdWord d (offLast w) + 1) *
      (rdWord d offStep - rdWord d (offLast w) + 1) < X86Env.W)
    (hac : rdWord d offAcc + (rdWord d offStep - rdWord d (offLast w) + 1) *
      (rdWord d offStep - rdWord d (offLast w) + 1) < X86Env.W) :
    let val_v := rdWord d (offVal w)
    let step_v := rdWord d offStep
    let acc_v := rdWord d offAcc
    let last_v := rdWord d (offLast w)
    let P := memExposureExposedPhys s w v_new d hd hw val_v k r hr
    let sDone := memExposureDone s w v_new step_v acc_v last_v d d d
    P.ok ∧
    X86Int.Out.ok sDone ∈ PDist.faultFree P ∧
    (∀ u ∈ PDist.faultFree P, u = .ok sDone) ∧
    PDist.fm P ≤ 3 * (((k + 1 : Nat) : Rat) * r.mem) ^ 2 + (memExposureFuel : Rat) * r.total := by
  intro val_v step_v acc_v last_v P sDone
  let C := idleArenaWordChannel s d d d hd.1 (offVal w) (hoffV_lt hw) r hr
  have hexp_ok : (TMR.exposed C k val_v).ok := TMR.exposed_ok C k val_v
  have hstepW : ∀ z, z < X86Env.W → ∀ y ∈ C.step z, y.1.1 < X86Env.W :=
    fun z _ => (idleArenaWordChannel_lt_W z).2
  have hDW : ∀ y ∈ TMR.exposed C k val_v, y.1.1 < X86Env.W :=
    exposed_lt_W_gen C hstepW k val_v hval_eq
  have hiter : TMR.iter C.f (k + 1) val_v = val_v :=
    idleArenaWordChannel_iter hval_eq k
  have hclean_D : PDist.faultFree (TMR.exposed C k val_v) = [val_v] := by
    simpa [hiter] using TMR.faultFree_exposed C k val_v
  have hdone_eq : ∀ a b c,
      memExposureDone s w v_new step_v acc_v last_v
        (wrWord d (offVal w) a) (wrWord d (offVal w) b) (wrWord d (offVal w) c) = sDone := by
    intro a b c
    have ha := setArenaState_wrWord_val hd.1 hw a (step_v + 1)
      (acc_v + (step_v - last_v + 1) * (step_v - last_v + 1)) v_new (step_v + 1)
    have hb := setArenaState_wrWord_val hd.1 hw b (step_v + 1)
      (acc_v + (step_v - last_v + 1) * (step_v - last_v + 1)) v_new (step_v + 1)
    have hc := setArenaState_wrWord_val hd.1 hw c (step_v + 1)
      (acc_v + (step_v - last_v + 1) * (step_v - last_v + 1)) v_new (step_v + 1)
    dsimp only [sDone, memExposureDone]
    rw [ha, hb, hc]
  have hoffVd : offVal w + 8 ≤ d.length := by rw [hd.1]; exact hoffV_lt hw
  have hcont_clean : ∀ a b c, a < X86Env.W → b < X86Env.W → c < X86Env.W →
      TMR.vote a b c = some val_v →
      PDist.faultFree (memExposurePhys r hr s w v_new
        (wrWord d (offVal w) a) (wrWord d (offVal w) b) (wrWord d (offVal w) c)) = [.ok sDone] := by
    intro a b c ha hb hc hv
    have hvoteV : TMR.vote
        (rdWord (wrWord d (offVal w) a) (offVal w))
        (rdWord (wrWord d (offVal w) b) (offVal w))
        (rdWord (wrWord d (offVal w) c) (offVal w)) = some val_v := by
      rw [rdWord_wrWord_same hoffVd ha, rdWord_wrWord_same hoffVd hb, rdWord_wrWord_same hoffVd hc]
      exact hv
    have hvoteL : TMR.vote
        (rdWord (wrWord d (offVal w) a) (offLast w))
        (rdWord (wrWord d (offVal w) b) (offLast w))
        (rdWord (wrWord d (offVal w) c) (offLast w)) = some last_v := by
      simp [rdWord_wrWord_disjoint _ (disj_VL w), vote_self, last_v]
    have hvoteS : TMR.vote
        (rdWord (wrWord d (offVal w) a) offStep)
        (rdWord (wrWord d (offVal w) b) offStep)
        (rdWord (wrWord d (offVal w) c) offStep) = some step_v := by
      simp [rdWord_wrWord_disjoint _ (disj_SV w).symm, vote_self, step_v]
    have hvoteA : TMR.vote
        (rdWord (wrWord d (offVal w) a) offAcc)
        (rdWord (wrWord d (offVal w) b) offAcc)
        (rdWord (wrWord d (offVal w) c) offAcc) = some acc_v := by
      simp [rdWord_wrWord_disjoint _ (disj_AV w).symm, vote_self, acc_v]
    rw [memExposure_phys_clean_majority r hr s hw hvnew (hd.wrWord _ a) (hd.wrWord _ b) (hd.wrWord _ c)
      hvoteV hvoteL hvoteS hvoteA hle hs1 hsq hac, hdone_eq a b c]
  refine ⟨memExposure_exposed_phys_ok s hw hd val_v k r hr, ?_, ?_, ?_⟩
  · dsimp only [P, memExposureExposedPhys]
    apply triRun3_mem_clean hclean_D
    rw [hcont_clean val_v val_v val_v hval_eq hval_eq hval_eq (vote_self val_v)]
    exact List.mem_singleton.2 rfl
  · dsimp only [P, memExposureExposedPhys]
    refine triRun3_clean hDW val_v _ (fun u => u = .ok sDone) ?_
    intro a b c ha hb hc hv u hu
    rw [hcont_clean a b c ha hb hc hv] at hu
    exact List.mem_singleton.1 hu
  · dsimp only [P, memExposureExposedPhys]
    have hf0 : 0 ≤ (memExposureFuel : Rat) * r.total :=
      Rat.mul_nonneg Rat.natCast_nonneg (rates_total_nonneg r hr)
    have htri := triRun3_bound_le hexp_ok val_v
      (fun a b c => memExposurePhys r hr s w v_new
        (wrWord d (offVal w) a) (wrWord d (offVal w) b) (wrWord d (offVal w) c))
      (fun a b c => memExposure_phys_ok r hr s w v_new
        (wrWord d (offVal w) a) (wrWord d (offVal w) b) (wrWord d (offVal w) c))
      hf0
      (fun a b c => memExposure_phys_fm_le r hr s w v_new
        (wrWord d (offVal w) a) (wrWord d (offVal w) b) (wrWord d (offVal w) c))
    have hexp := exposure_err_pow C k val_v
    rw [hiter] at hexp
    exact Rat.le_trans htri (add_le_add' hexp Rat.le_refl)

def cell4 (d : List Nat) (w va la sa aa : Nat) : List Nat :=
  wrWord (wrWord (wrWord (wrWord d (offVal w) va) (offLast w) la) offStep sa) offAcc aa

private theorem cell4_ok {d : List Nat} (hd : ArenaDataOk d) (w va la sa aa : Nat) :
    ArenaDataOk (cell4 d w va la sa aa) :=
  (((hd.wrWord _ va).wrWord _ la).wrWord _ sa).wrWord _ aa

private theorem setArenaState_wr4 {d : List Nat} (hd : d.length = arenaSize) {w : Nat}
    (hw : w < maxSlots) (va la sa aa step' acc' v_new last' : Nat) :
    setArenaState (cell4 d w va la sa aa) w step' acc' v_new last' =
      setArenaState d w step' acc' v_new last' := by
  have hl1 : (wrWord d (offVal w) va).length = arenaSize := by rw [wrWord_length, hd]
  have hl2 : (wrWord (wrWord d (offVal w) va) (offLast w) la).length = arenaSize := by rw [wrWord_length, hl1]
  have hl3 : (wrWord (wrWord (wrWord d (offVal w) va) (offLast w) la) offStep sa).length = arenaSize := by
    rw [wrWord_length, hl2]
  dsimp only [cell4]
  rw [setArenaState_wrWord_acc hl3, setArenaState_wrWord_step hl2,
    setArenaState_wrWord_last hl1 hw, setArenaState_wrWord_val hd hw]

/-- 4-word simultaneous Decision-50 physical step: all four tracked fields
`(val[w], last_step[w], step, acc)` independently accumulate `(k_v, k_l, k_s, k_a)`
idle exposure ticks across 3 copies of `idleArenaWordChannel` before `memExposurePhys`
executes the 450-cycle machine-code majority voter and repair kernel. -/
def memExposureExposed4Phys (s : X86Int.St) (w v_new : Nat) (d : List Nat) (hd : ArenaDataOk d)
    (hw : w < maxSlots) (k_val k_last k_step k_acc : Nat)
    (r : X86Int.Rates) (hr : r.ok) : PDist.PDF X86Int.Out :=
  let Cv := idleArenaWordChannel s d d d hd.1 (offVal w) (hoffV_lt hw) r hr
  let Cl := idleArenaWordChannel s d d d hd.1 (offLast w) (hoffL_lt hw) r hr
  let Cs := idleArenaWordChannel s d d d hd.1 offStep hoffS_le r hr
  let Ca := idleArenaWordChannel s d d d hd.1 offAcc hoffA_le r hr
  triRun3 (TMR.exposed Cv k_val (rdWord d (offVal w))) (rdWord d (offVal w)) fun va vb vc =>
    triRun3 (TMR.exposed Cl k_last (rdWord d (offLast w))) (rdWord d (offLast w)) fun la lb lc =>
      triRun3 (TMR.exposed Cs k_step (rdWord d offStep)) (rdWord d offStep) fun sa sb sc =>
        triRun3 (TMR.exposed Ca k_acc (rdWord d offAcc)) (rdWord d offAcc) fun aa ab ac =>
          memExposurePhys r hr s w v_new
            (cell4 d w va la sa aa) (cell4 d w vb lb sb ab) (cell4 d w vc lc sc ac)

theorem memExposure_four_word_phys_bound (s : X86Int.St) {w v_new : Nat} (hw : w < maxSlots)
    (hvnew : v_new < X86Env.W) {d : List Nat} (hd : ArenaDataOk d)
    (k_val k_last k_step k_acc : Nat) (r : X86Int.Rates) (hr : r.ok)
    (hle : rdWord d (offLast w) ≤ rdWord d offStep)
    (hs1 : rdWord d offStep + 1 < X86Env.W)
    (hsq : (rdWord d offStep - rdWord d (offLast w) + 1) *
      (rdWord d offStep - rdWord d (offLast w) + 1) < X86Env.W)
    (hac : rdWord d offAcc + (rdWord d offStep - rdWord d (offLast w) + 1) *
      (rdWord d offStep - rdWord d (offLast w) + 1) < X86Env.W) :
    let step_v := rdWord d offStep
    let acc_v := rdWord d offAcc
    let last_v := rdWord d (offLast w)
    let P := memExposureExposed4Phys s w v_new d hd hw k_val k_last k_step k_acc r hr
    let sDone := memExposureDone s w v_new step_v acc_v last_v d d d
    P.ok ∧
    X86Int.Out.ok sDone ∈ PDist.faultFree P ∧
    (∀ u ∈ PDist.faultFree P, u = .ok sDone) ∧
    PDist.fm P ≤
      3 * (((k_val + 1 : Nat) : Rat) * r.mem) ^ 2 +
      (3 * (((k_last + 1 : Nat) : Rat) * r.mem) ^ 2 +
      (3 * (((k_step + 1 : Nat) : Rat) * r.mem) ^ 2 +
      (3 * (((k_acc + 1 : Nat) : Rat) * r.mem) ^ 2 + (memExposureFuel : Rat) * r.total))) := by
  intro step_v acc_v last_v P sDone
  let val_v := rdWord d (offVal w)
  let Cv := idleArenaWordChannel s d d d hd.1 (offVal w) (hoffV_lt hw) r hr
  let Cl := idleArenaWordChannel s d d d hd.1 (offLast w) (hoffL_lt hw) r hr
  let Cs := idleArenaWordChannel s d d d hd.1 offStep hoffS_le r hr
  let Ca := idleArenaWordChannel s d d d hd.1 offAcc hoffA_le r hr
  have hvalW : val_v < X86Env.W := rdWord_lt_W hd.2 (offVal w)
  have hlastW : last_v < X86Env.W := rdWord_lt_W hd.2 (offLast w)
  have hstepW : step_v < X86Env.W := rdWord_lt_W hd.2 offStep
  have haccW : acc_v < X86Env.W := rdWord_lt_W hd.2 offAcc
  let Dv := TMR.exposed Cv k_val val_v
  let Dl := TMR.exposed Cl k_last last_v
  let Ds := TMR.exposed Cs k_step step_v
  let Da := TMR.exposed Ca k_acc acc_v
  have hDvW : ∀ y ∈ Dv, y.1.1 < X86Env.W :=
    exposed_lt_W_gen Cv (fun z _ => (idleArenaWordChannel_lt_W z).2) k_val val_v hvalW
  have hDlW : ∀ y ∈ Dl, y.1.1 < X86Env.W :=
    exposed_lt_W_gen Cl (fun z _ => (idleArenaWordChannel_lt_W z).2) k_last last_v hlastW
  have hDsW : ∀ y ∈ Ds, y.1.1 < X86Env.W :=
    exposed_lt_W_gen Cs (fun z _ => (idleArenaWordChannel_lt_W z).2) k_step step_v hstepW
  have hDaW : ∀ y ∈ Da, y.1.1 < X86Env.W :=
    exposed_lt_W_gen Ca (fun z _ => (idleArenaWordChannel_lt_W z).2) k_acc acc_v haccW
  have hiterV : TMR.iter Cv.f (k_val + 1) val_v = val_v := idleArenaWordChannel_iter hvalW k_val
  have hiterL : TMR.iter Cl.f (k_last + 1) last_v = last_v := idleArenaWordChannel_iter hlastW k_last
  have hiterS : TMR.iter Cs.f (k_step + 1) step_v = step_v := idleArenaWordChannel_iter hstepW k_step
  have hiterA : TMR.iter Ca.f (k_acc + 1) acc_v = acc_v := idleArenaWordChannel_iter haccW k_acc
  have hcDv : PDist.faultFree Dv = [val_v] := by simpa [Dv, hiterV] using TMR.faultFree_exposed Cv k_val val_v
  have hcDl : PDist.faultFree Dl = [last_v] := by simpa [Dl, hiterL] using TMR.faultFree_exposed Cl k_last last_v
  have hcDs : PDist.faultFree Ds = [step_v] := by simpa [Ds, hiterS] using TMR.faultFree_exposed Cs k_step step_v
  have hcDa : PDist.faultFree Da = [acc_v] := by simpa [Da, hiterA] using TMR.faultFree_exposed Ca k_acc acc_v
  have hrdV : ∀ va la sa aa, va < X86Env.W → rdWord (cell4 d w va la sa aa) (offVal w) = va := by
    intro va la sa aa hva
    simp only [cell4]
    rw [rdWord_wrWord_disjoint _ (disj_AV w), rdWord_wrWord_disjoint _ (disj_SV w),
      rdWord_wrWord_disjoint _ (disj_VL w).symm, rdWord_wrWord_same (by rw [hd.1]; exact hoffV_lt hw) hva]
  have hrdL : ∀ va la sa aa, la < X86Env.W → rdWord (cell4 d w va la sa aa) (offLast w) = la := by
    intro va la sa aa hla
    simp only [cell4]
    rw [rdWord_wrWord_disjoint _ (disj_AL w), rdWord_wrWord_disjoint _ (disj_SL w),
      rdWord_wrWord_same (by rw [wrWord_length, hd.1]; exact hoffL_lt hw) hla]
  have hrdS : ∀ va la sa aa, sa < X86Env.W → rdWord (cell4 d w va la sa aa) offStep = sa := by
    intro va la sa aa hsa
    simp only [cell4]
    rw [rdWord_wrWord_disjoint _ disj_SA.symm,
      rdWord_wrWord_same (by rw [wrWord_length, wrWord_length, hd.1]; exact hoffS_le) hsa]
  have hrdA : ∀ va la sa aa, aa < X86Env.W → rdWord (cell4 d w va la sa aa) offAcc = aa := by
    intro va la sa aa haa
    simp only [cell4]
    rw [rdWord_wrWord_same (by rw [wrWord_length, wrWord_length, wrWord_length, hd.1]; exact hoffA_le) haa]
  have hphys_clean : ∀ va vb vc la lb lc sa sb sc aa ab ac,
      va < X86Env.W → vb < X86Env.W → vc < X86Env.W → TMR.vote va vb vc = some val_v →
      la < X86Env.W → lb < X86Env.W → lc < X86Env.W → TMR.vote la lb lc = some last_v →
      sa < X86Env.W → sb < X86Env.W → sc < X86Env.W → TMR.vote sa sb sc = some step_v →
      aa < X86Env.W → ab < X86Env.W → ac < X86Env.W → TMR.vote aa ab ac = some acc_v →
      PDist.faultFree (memExposurePhys r hr s w v_new
        (cell4 d w va la sa aa) (cell4 d w vb lb sb ab) (cell4 d w vc lc sc ac)) = [.ok sDone] := by
    intro va vb vc la lb lc sa sb sc aa ab ac hva hvb hvc hv hla hlb hlc hl hsa hsb hsc hs haa hab hac' ha
    have hv' : TMR.vote (rdWord (cell4 d w va la sa aa) (offVal w))
        (rdWord (cell4 d w vb lb sb ab) (offVal w)) (rdWord (cell4 d w vc lc sc ac) (offVal w)) = some val_v := by
      rw [hrdV va la sa aa hva, hrdV vb lb sb ab hvb, hrdV vc lc sc ac hvc]; exact hv
    have hl' : TMR.vote (rdWord (cell4 d w va la sa aa) (offLast w))
        (rdWord (cell4 d w vb lb sb ab) (offLast w)) (rdWord (cell4 d w vc lc sc ac) (offLast w)) = some last_v := by
      rw [hrdL va la sa aa hla, hrdL vb lb sb ab hlb, hrdL vc lc sc ac hlc]; exact hl
    have hs' : TMR.vote (rdWord (cell4 d w va la sa aa) offStep)
        (rdWord (cell4 d w vb lb sb ab) offStep) (rdWord (cell4 d w vc lc sc ac) offStep) = some step_v := by
      rw [hrdS va la sa aa hsa, hrdS vb lb sb ab hsb, hrdS vc lc sc ac hsc]; exact hs
    have ha' : TMR.vote (rdWord (cell4 d w va la sa aa) offAcc)
        (rdWord (cell4 d w vb lb sb ab) offAcc) (rdWord (cell4 d w vc lc sc ac) offAcc) = some acc_v := by
      rw [hrdA va la sa aa haa, hrdA vb lb sb ab hab, hrdA vc lc sc ac hac']; exact ha
    rw [memExposure_phys_clean_majority r hr s hw hvnew (cell4_ok hd _ _ _ _ _) (cell4_ok hd _ _ _ _ _)
      (cell4_ok hd _ _ _ _ _) hv' hl' hs' ha' hle hs1 hsq hac]
    dsimp only [sDone, memExposureDone]
    rw [setArenaState_wr4 hd.1 hw, setArenaState_wr4 hd.1 hw, setArenaState_wr4 hd.1 hw]
  have hokA : ∀ va vb vc la lb lc sa sb sc,
      (triRun3 Da acc_v fun aa ab ac =>
        memExposurePhys r hr s w v_new (cell4 d w va la sa aa) (cell4 d w vb lb sb ab) (cell4 d w vc lc sc ac)).ok :=
    fun _ _ _ _ _ _ _ _ _ =>
      triRun3_ok (TMR.exposed_ok Ca k_acc acc_v) acc_v fun _ _ _ => memExposure_phys_ok r hr s w v_new _ _ _
  have hokS : ∀ va vb vc la lb lc,
      (triRun3 Ds step_v fun sa sb sc =>
        triRun3 Da acc_v fun aa ab ac =>
          memExposurePhys r hr s w v_new (cell4 d w va la sa aa) (cell4 d w vb lb sb ab) (cell4 d w vc lc sc ac)).ok :=
    fun va vb vc la lb lc =>
      triRun3_ok (TMR.exposed_ok Cs k_step step_v) step_v (hokA va vb vc la lb lc)
  have hokL : ∀ va vb vc,
      (triRun3 Dl last_v fun la lb lc =>
        triRun3 Ds step_v fun sa sb sc =>
          triRun3 Da acc_v fun aa ab ac =>
            memExposurePhys r hr s w v_new (cell4 d w va la sa aa) (cell4 d w vb lb sb ab) (cell4 d w vc lc sc ac)).ok :=
    fun va vb vc =>
      triRun3_ok (TMR.exposed_ok Cl k_last last_v) last_v (hokS va vb vc)
  have hPok : P.ok :=
    triRun3_ok (TMR.exposed_ok Cv k_val val_v) val_v hokL
  have hmem_clean : X86Int.Out.ok sDone ∈ PDist.faultFree P := by
    dsimp only [P, memExposureExposed4Phys]
    refine triRun3_mem_clean hcDv _ (triRun3_mem_clean hcDl _ (triRun3_mem_clean hcDs _
      (triRun3_mem_clean hcDa _ ?_)))
    rw [hphys_clean val_v val_v val_v last_v last_v last_v step_v step_v step_v acc_v acc_v acc_v
      hvalW hvalW hvalW (vote_self _) hlastW hlastW hlastW (vote_self _)
      hstepW hstepW hstepW (vote_self _) haccW haccW haccW (vote_self _)]
    exact List.mem_singleton.2 rfl
  have hall_clean : ∀ u ∈ PDist.faultFree P, u = .ok sDone := by
    dsimp only [P, memExposureExposed4Phys]
    refine triRun3_clean hDvW val_v _ (fun u => u = .ok sDone) ?_
    intro va vb vc hva hvb hvc hv
    refine triRun3_clean hDlW last_v _ (fun u => u = .ok sDone) ?_
    intro la lb lc hla hlb hlc hl
    refine triRun3_clean hDsW step_v _ (fun u => u = .ok sDone) ?_
    intro sa sb sc hsa hsb hsc hs
    refine triRun3_clean hDaW acc_v _ (fun u => u = .ok sDone) ?_
    intro aa ab ac haa hab hac' ha u hu
    rw [hphys_clean va vb vc la lb lc sa sb sc aa ab ac
      hva hvb hvc hv hla hlb hlc hl hsa hsb hsc hs haa hab hac' ha] at hu
    exact List.mem_singleton.1 hu
  refine ⟨hPok, hmem_clean, hall_clean, ?_⟩
  have hrt0 : 0 ≤ (memExposureFuel : Rat) * r.total :=
    Rat.mul_nonneg Rat.natCast_nonneg (rates_total_nonneg r hr)
  have hbA : ∀ va vb vc la lb lc sa sb sc,
      PDist.fm (triRun3 Da acc_v fun aa ab ac =>
        memExposurePhys r hr s w v_new (cell4 d w va la sa aa) (cell4 d w vb lb sb ab) (cell4 d w vc lc sc ac)) ≤
      3 * (((k_acc + 1 : Nat) : Rat) * r.mem) ^ 2 + (memExposureFuel : Rat) * r.total := by
    intro va vb vc la lb lc sa sb sc
    have htr := triRun3_bound_le (TMR.exposed_ok Ca k_acc acc_v) acc_v
      (fun aa ab ac => memExposurePhys r hr s w v_new
        (cell4 d w va la sa aa) (cell4 d w vb lb sb ab) (cell4 d w vc lc sc ac))
      (fun aa ab ac => memExposure_phys_ok r hr s w v_new
        (cell4 d w va la sa aa) (cell4 d w vb lb sb ab) (cell4 d w vc lc sc ac))
      hrt0
      (fun aa ab ac => memExposure_phys_fm_le r hr s w v_new
        (cell4 d w va la sa aa) (cell4 d w vb lb sb ab) (cell4 d w vc lc sc ac))
    have hex := exposure_err_pow Ca k_acc acc_v
    rw [hiterA] at hex
    exact Rat.le_trans htr (add_le_add' hex Rat.le_refl)
  have hqa0 : 0 ≤ 3 * (((k_acc + 1 : Nat) : Rat) * r.mem) ^ 2 + (memExposureFuel : Rat) * r.total :=
    X86Int.nonneg_add (Rat.mul_nonneg (by decide) (by
      rw [Rat.pow_succ, Rat.pow_one]
      exact Rat.mul_nonneg (Rat.mul_nonneg Rat.natCast_nonneg hr.2.2.1.1)
        (Rat.mul_nonneg Rat.natCast_nonneg hr.2.2.1.1))) hrt0
  have hbS : ∀ va vb vc la lb lc,
      PDist.fm (triRun3 Ds step_v fun sa sb sc =>
        triRun3 Da acc_v fun aa ab ac =>
          memExposurePhys r hr s w v_new (cell4 d w va la sa aa) (cell4 d w vb lb sb ab) (cell4 d w vc lc sc ac)) ≤
      3 * (((k_step + 1 : Nat) : Rat) * r.mem) ^ 2 +
        (3 * (((k_acc + 1 : Nat) : Rat) * r.mem) ^ 2 + (memExposureFuel : Rat) * r.total) := by
    intro va vb vc la lb lc
    have htr := triRun3_bound_le (TMR.exposed_ok Cs k_step step_v) step_v _
      (hokA va vb vc la lb lc) hqa0 (hbA va vb vc la lb lc)
    have hex := exposure_err_pow Cs k_step step_v
    rw [hiterS] at hex
    exact Rat.le_trans htr (add_le_add' hex Rat.le_refl)
  have hqs0 : 0 ≤ 3 * (((k_step + 1 : Nat) : Rat) * r.mem) ^ 2 +
      (3 * (((k_acc + 1 : Nat) : Rat) * r.mem) ^ 2 + (memExposureFuel : Rat) * r.total) :=
    X86Int.nonneg_add (Rat.mul_nonneg (by decide) (by
      rw [Rat.pow_succ, Rat.pow_one]
      exact Rat.mul_nonneg (Rat.mul_nonneg Rat.natCast_nonneg hr.2.2.1.1)
        (Rat.mul_nonneg Rat.natCast_nonneg hr.2.2.1.1))) hqa0
  have hbL : ∀ va vb vc,
      PDist.fm (triRun3 Dl last_v fun la lb lc =>
        triRun3 Ds step_v fun sa sb sc =>
          triRun3 Da acc_v fun aa ab ac =>
            memExposurePhys r hr s w v_new (cell4 d w va la sa aa) (cell4 d w vb lb sb ab) (cell4 d w vc lc sc ac)) ≤
      3 * (((k_last + 1 : Nat) : Rat) * r.mem) ^ 2 +
        (3 * (((k_step + 1 : Nat) : Rat) * r.mem) ^ 2 +
        (3 * (((k_acc + 1 : Nat) : Rat) * r.mem) ^ 2 + (memExposureFuel : Rat) * r.total)) := by
    intro va vb vc
    have htr := triRun3_bound_le (TMR.exposed_ok Cl k_last last_v) last_v _
      (hokS va vb vc) hqs0 (hbS va vb vc)
    have hex := exposure_err_pow Cl k_last last_v
    rw [hiterL] at hex
    exact Rat.le_trans htr (add_le_add' hex Rat.le_refl)
  have hql0 : 0 ≤ 3 * (((k_last + 1 : Nat) : Rat) * r.mem) ^ 2 +
      (3 * (((k_step + 1 : Nat) : Rat) * r.mem) ^ 2 +
      (3 * (((k_acc + 1 : Nat) : Rat) * r.mem) ^ 2 + (memExposureFuel : Rat) * r.total)) :=
    X86Int.nonneg_add (Rat.mul_nonneg (by decide) (by
      rw [Rat.pow_succ, Rat.pow_one]
      exact Rat.mul_nonneg (Rat.mul_nonneg Rat.natCast_nonneg hr.2.2.1.1)
        (Rat.mul_nonneg Rat.natCast_nonneg hr.2.2.1.1))) hqs0
  dsimp only [P, memExposureExposed4Phys]
  have htrV := triRun3_bound_le (TMR.exposed_ok Cv k_val val_v) val_v _ hokL hql0 hbL
  have hexV := exposure_err_pow Cv k_val val_v
  rw [hiterV] at hexV
  exact Rat.le_trans htrV (add_le_add' hexV Rat.le_refl)

/-! ### Multi-Write Decision 50 Schedule & Cumulative `acc_exposure` Bound -/

/-- One architectural write update step on arena byte buffer `d`. -/
def nextArena (d : List Nat) (w v_new : Nat) : List Nat :=
  let step_v := rdWord d offStep
  let acc_v := rdWord d offAcc
  let last_v := rdWord d (offLast w)
  let delta := step_v - last_v + 1
  setArenaState d w (step_v + 1) (acc_v + delta * delta) v_new (step_v + 1)

private theorem ArenaDataOk.nextArena {d : List Nat} (hd : ArenaDataOk d) (w v_new : Nat) :
    ArenaDataOk (nextArena d w v_new) :=
  hd.setArenaState w _ _ v_new _

private theorem nextArena_rdAcc {d : List Nat} (hd : d.length = arenaSize) (w v_new : Nat)
    (hac : rdWord d offAcc + (rdWord d offStep - rdWord d (offLast w) + 1) *
      (rdWord d offStep - rdWord d (offLast w) + 1) < X86Env.W) :
    rdWord (nextArena d w v_new) offAcc =
      rdWord d offAcc + (rdWord d offStep - rdWord d (offLast w) + 1) *
        (rdWord d offStep - rdWord d (offLast w) + 1) :=
  setArenaState_rdAcc hd hac

private theorem nextArena_rdStep {d : List Nat} (hd : d.length = arenaSize) (w v_new : Nat)
    (hs1 : rdWord d offStep + 1 < X86Env.W) :
    rdWord (nextArena d w v_new) offStep = rdWord d offStep + 1 :=
  setArenaState_rdStep hd hs1

/-- Exact sequential specification of `d` across a write workload `ops : List (Nat × Nat)`. -/
def execArenaOps (d : List Nat) : List (Nat × Nat) → List Nat
  | [] => d
  | (w, v_new) :: rest => execArenaOps (nextArena d w v_new) rest

/-- Exact retired `X86Int.St` after `ops` cleanly complete from `s` on `d`. -/
def execStateOps (s : X86Int.St) (d : List Nat) : List (Nat × Nat) → X86Int.St
  | [] => s
  | (w, v_new) :: rest =>
    let s' := memExposureDone s w v_new (rdWord d offStep) (rdWord d offAcc) (rdWord d (offLast w)) d d d
    execStateOps s' (nextArena d w v_new) rest

/-- Cumulative Decision-50 sum `∑ (step_i - last_i[w_i] + 1)^2` added to `acc`. -/
def opsExposureSum (d : List Nat) : List (Nat × Nat) → Nat
  | [] => 0
  | (w, v_new) :: rest =>
    let delta := rdWord d offStep - rdWord d (offLast w) + 1
    delta * delta + opsExposureSum (nextArena d w v_new) rest

/-- Causal & non-overflow validity of a write workload `ops` starting at `d`. -/
def OpsOk (d : List Nat) : List (Nat × Nat) → Prop
  | [] => True
  | (w, v_new) :: rest =>
    let step_v := rdWord d offStep
    let acc_v := rdWord d offAcc
    let last_v := rdWord d (offLast w)
    let delta := step_v - last_v + 1
    w < maxSlots ∧
    v_new < X86Env.W ∧
    last_v ≤ step_v ∧
    step_v + 1 < X86Env.W ∧
    delta * delta < X86Env.W ∧
    acc_v + delta * delta < X86Env.W ∧
    OpsOk (nextArena d w v_new) rest

theorem execArenaOps_rdAcc (ops : List (Nat × Nat)) :
    ∀ {d : List Nat}, ArenaDataOk d → OpsOk d ops →
      rdWord (execArenaOps d ops) offAcc = rdWord d offAcc + opsExposureSum d ops := by
  induction ops with
  | nil => intro d _ _; rfl
  | cons p rest ih =>
    intro d hd hops
    rcases p with ⟨w, v_new⟩
    rcases hops with ⟨_, _, _, _, _, hac, hrest⟩
    have hrec := ih (hd.nextArena w v_new) hrest
    have hacc' := nextArena_rdAcc hd.1 w v_new hac
    dsimp only [execArenaOps, opsExposureSum]
    rw [hrec, hacc', Nat.add_assoc]

theorem execArenaOps_rdStep (ops : List (Nat × Nat)) :
    ∀ {d : List Nat}, ArenaDataOk d → OpsOk d ops →
      rdWord (execArenaOps d ops) offStep = rdWord d offStep + ops.length := by
  induction ops with
  | nil => intro d _ _; rfl
  | cons p rest ih =>
    intro d hd hops
    rcases p with ⟨w, v_new⟩
    rcases hops with ⟨_, _, _, hs1, _, _, hrest⟩
    have hrec := ih (hd.nextArena w v_new) hrest
    have hstep' := nextArena_rdStep hd.1 w v_new hs1
    dsimp only [execArenaOps, List.length_cons]
    rw [hrec, hstep']; omega

def memExposureMultiStep (w v_new : Nat) (d : List Nat) (hd : ArenaDataOk d)
    (r : X86Int.Rates) (hr : r.ok) (o : X86Int.Out) : PDist.PDF X86Int.Out :=
  match o with
  | .ok s =>
    if hw : w < maxSlots then
      memExposureExposedPhys s w v_new d hd hw (rdWord d (offVal w))
        (rdWord d offStep - rdWord d (offLast w)) r hr
    else PDist.pureF .stuck
  | o => PDist.pureF o

/-- Multi-write physical Decision 50 execution: sequences each write `(w, v_new)` through
idle exposure on `idleArenaWordChannel` followed by `memExposurePhys` (`450` cycles). -/
def memExposureMultiPhys (d : List Nat) (hd : ArenaDataOk d) (r : X86Int.Rates) (hr : r.ok) :
    List (Nat × Nat) → X86Int.Out → PDist.PDF X86Int.Out
  | [], o => PDist.pureF o
  | (w, v_new) :: rest, o =>
    PDist.bindF (memExposureMultiStep w v_new d hd r hr o)
      (memExposureMultiPhys (nextArena d w v_new) (hd.nextArena w v_new) r hr rest)

private theorem exposure_term_eq (delta : Nat) (q : Rat) :
    3 * (((delta : Rat)) * q) ^ 2 = 3 * q * q * ((delta * delta : Nat) : Rat) := by
  rw [Rat.pow_succ, Rat.pow_one, Rat.natCast_mul]
  have h1 : (((delta : Rat) * q) * ((delta : Rat) * q)) = (q * q) * ((delta : Rat) * (delta : Rat)) := by
    calc ((delta : Rat) * q) * ((delta : Rat) * q)
        = (delta : Rat) * (q * ((delta : Rat) * q)) := Rat.mul_assoc _ _ _
      _ = (delta : Rat) * ((delta : Rat) * (q * q)) := by
          congr 1
          calc q * ((delta : Rat) * q) = (q * (delta : Rat)) * q := (Rat.mul_assoc _ _ _).symm
            _ = ((delta : Rat) * q) * q := by rw [Rat.mul_comm q]
            _ = (delta : Rat) * (q * q) := Rat.mul_assoc _ _ _
      _ = ((delta : Rat) * (delta : Rat)) * (q * q) := (Rat.mul_assoc _ _ _).symm
      _ = (q * q) * ((delta : Rat) * (delta : Rat)) := Rat.mul_comm _ _
  rw [h1, ← Rat.mul_assoc 3 (q * q), ← Rat.mul_assoc 3 q q]

theorem memExposure_multi_step_ok (w v_new : Nat) (d : List Nat) (hd : ArenaDataOk d)
    (r : X86Int.Rates) (hr : r.ok) (o : X86Int.Out) :
    (memExposureMultiStep w v_new d hd r hr o).ok := by
  cases o with
  | ok s =>
    dsimp only [memExposureMultiStep]
    split
    · next hw => exact memExposure_exposed_phys_ok s hw hd _ _ r hr
    · exact PDist.pureF_ok _
  | exit _ => exact PDist.pureF_ok _
  | stuck => exact PDist.pureF_ok _

theorem memExposure_multi_phys_ok :
    ∀ (d : List Nat) (hd : ArenaDataOk d) (r : X86Int.Rates) (hr : r.ok)
      (ops : List (Nat × Nat)) (o : X86Int.Out),
      (memExposureMultiPhys d hd r hr ops o).ok
  | _, _, _, _, [], o => PDist.pureF_ok o
  | d, hd, r, hr, (w, v_new) :: rest, o =>
    PDist.bindF_ok (memExposure_multi_step_ok w v_new d hd r hr o)
      (memExposure_multi_phys_ok (nextArena d w v_new) (hd.nextArena w v_new) r hr rest)

theorem memExposure_multi_write_clean :
    ∀ (d : List Nat) (hd : ArenaDataOk d) (r : X86Int.Rates) (hr : r.ok)
      (ops : List (Nat × Nat)) (s : X86Int.St),
      OpsOk d ops →
      X86Int.Out.ok (execStateOps s d ops) ∈
        PDist.faultFree (memExposureMultiPhys d hd r hr ops (.ok s)) ∧
      ∀ u ∈ PDist.faultFree (memExposureMultiPhys d hd r hr ops (.ok s)),
        u = .ok (execStateOps s d ops)
  | d, hd, r, hr, [], s, _ => by
    simp [memExposureMultiPhys, execStateOps, PDist.faultFree_pureF]
  | d, hd, r, hr, (w, v_new) :: rest, s, hops => by
    rcases hops with ⟨hw, hv, hle, hs1, hsq, hac, hrest⟩
    let s' := memExposureDone s w v_new (rdWord d offStep) (rdWord d offAcc) (rdWord d (offLast w)) d d d
    have hstep1 := memExposure_phys_bound s hw hv hd (rdWord d offStep - rdWord d (offLast w)) r hr
      (rdWord_lt_W hd.2 _) hle hs1 hsq hac
    have hmstep : memExposureMultiStep w v_new d hd r hr (.ok s) =
        memExposureExposedPhys s w v_new d hd hw (rdWord d (offVal w))
          (rdWord d offStep - rdWord d (offLast w)) r hr := by
      simp [memExposureMultiStep, hw]
    have hmem1 : X86Int.Out.ok s' ∈ PDist.faultFree (memExposureMultiStep w v_new d hd r hr (.ok s)) := by
      rw [hmstep]; exact hstep1.2.1
    have hall1 : ∀ v ∈ PDist.faultFree (memExposureMultiStep w v_new d hd r hr (.ok s)),
        v = .ok s' := by
      rw [hmstep]; exact hstep1.2.2.1
    obtain ⟨ihmem, ihall⟩ := memExposure_multi_write_clean (nextArena d w v_new) (hd.nextArena w v_new)
      r hr rest s' hrest
    constructor
    · simp only [memExposureMultiPhys, execStateOps, PDist.faultFree_bindF, List.mem_flatMap]
      exact ⟨.ok s', hmem1, ihmem⟩
    · intro u hu
      simp only [memExposureMultiPhys, PDist.faultFree_bindF, List.mem_flatMap] at hu
      obtain ⟨v, hvMem, hu⟩ := hu
      rw [hall1 v hvMem] at hu
      exact ihall u hu

/-- Multi-write Decision 50 physical bound: cumulative fault probability across `ops`
is at most `3 * r.mem^2 * (opsExposureSum d ops) + ops.length * (memExposureFuel * r.total)`,
literally linking the hardware `acc_exposure` increment `opsExposureSum` to the
physical fault bound over `X86Int.prun`. -/
theorem memExposure_multi_write_bound :
    ∀ (d : List Nat) (hd : ArenaDataOk d) (r : X86Int.Rates) (hr : r.ok)
      (ops : List (Nat × Nat)) (o : X86Int.Out),
      OpsOk d ops →
      PDist.fm (memExposureMultiPhys d hd r hr ops o) ≤
        3 * r.mem * r.mem * ((opsExposureSum d ops : Nat) : Rat) +
          (ops.length : Rat) * ((memExposureFuel : Rat) * r.total)
  | d, hd, r, hr, [], o, _ => by
    rw [memExposureMultiPhys, opsExposureSum, PDist.fm_pureF]
    simp only [List.length_nil, show ((0 : Nat) : Rat) = 0 from rfl, Rat.mul_zero, Rat.zero_mul,
      Rat.add_zero]
    exact Rat.le_refl
  | d, hd, r, hr, (w, v_new) :: rest, o, hops => by
    rcases hops with ⟨hw, hv, hle, hs1, hsq, hac, hrest⟩
    let delta := rdWord d offStep - rdWord d (offLast w) + 1
    let d' := nextArena d w v_new
    let hd' : ArenaDataOk d' := hd.nextArena w v_new
    have hih := memExposure_multi_write_bound d' hd' r hr rest
    have h3m0 : 0 ≤ 3 * r.mem * r.mem :=
      Rat.mul_nonneg (Rat.mul_nonneg (by decide) hr.2.2.1.1) hr.2.2.1.1
    have hft0 : 0 ≤ (memExposureFuel : Rat) * r.total :=
      Rat.mul_nonneg Rat.natCast_nonneg (rates_total_nonneg r hr)
    have hrhs0 : 0 ≤ 3 * r.mem * r.mem * ((opsExposureSum d' rest : Nat) : Rat) +
        (rest.length : Rat) * ((memExposureFuel : Rat) * r.total) :=
      X86Int.nonneg_add (Rat.mul_nonneg h3m0 Rat.natCast_nonneg)
        (Rat.mul_nonneg Rat.natCast_nonneg hft0)
    have hstep_le : PDist.fm (memExposureMultiStep w v_new d hd r hr o) ≤
        3 * r.mem * r.mem * ((delta * delta : Nat) : Rat) + (memExposureFuel : Rat) * r.total := by
      cases o with
      | ok s =>
        simp only [memExposureMultiStep, dif_pos hw]
        have hbd := (memExposure_phys_bound s hw hv hd (rdWord d offStep - rdWord d (offLast w))
          r hr (rdWord_lt_W hd.2 _) hle hs1 hsq hac).2.2.2
        have hdeq : ((rdWord d offStep - rdWord d (offLast w) + 1 : Nat) : Rat) = (delta : Rat) := rfl
        rw [hdeq, exposure_term_eq delta r.mem] at hbd
        exact hbd
      | exit _ =>
        simp only [memExposureMultiStep, PDist.fm_pureF]
        exact X86Int.nonneg_add (Rat.mul_nonneg h3m0 Rat.natCast_nonneg) hft0
      | stuck =>
        simp only [memExposureMultiStep, PDist.fm_pureF]
        exact X86Int.nonneg_add (Rat.mul_nonneg h3m0 Rat.natCast_nonneg) hft0
    have hbind := PDist.fm_bindF_le' (memExposure_multi_step_ok w v_new d hd r hr o) hrhs0
      (memExposure_multi_phys_ok d' hd' r hr rest) (fun u => hih u hrest)
    have htot_le := Rat.le_trans hbind (add_le_add' hstep_le Rat.le_refl)
    have hsplit :
        (3 * r.mem * r.mem * ((delta * delta : Nat) : Rat) + (memExposureFuel : Rat) * r.total) +
          (3 * r.mem * r.mem * ((opsExposureSum d' rest : Nat) : Rat) +
            (rest.length : Rat) * ((memExposureFuel : Rat) * r.total)) =
        3 * r.mem * r.mem * (((delta * delta + opsExposureSum d' rest : Nat) : Rat)) +
          (((w, v_new) :: rest).length : Rat) * ((memExposureFuel : Rat) * r.total) := by
      rw [Rat.natCast_add, Rat.mul_add, List.length_cons, Rat.natCast_add, Rat.add_mul,
        show ((1 : Nat) : Rat) = 1 from rfl, Rat.one_mul, Rat.add_comm ((rest.length : Rat) * _)]
      have hcomm :
          (3 * r.mem * r.mem * ((delta * delta : Nat) : Rat) + (memExposureFuel : Rat) * r.total) +
            (3 * r.mem * r.mem * ((opsExposureSum d' rest : Nat) : Rat) +
              (rest.length : Rat) * ((memExposureFuel : Rat) * r.total)) =
          (3 * r.mem * r.mem * ((delta * delta : Nat) : Rat) +
            3 * r.mem * r.mem * ((opsExposureSum d' rest : Nat) : Rat)) +
            ((memExposureFuel : Rat) * r.total +
              (rest.length : Rat) * ((memExposureFuel : Rat) * r.total)) := by
        calc _ = (3 * r.mem * r.mem * ((delta * delta : Nat) : Rat) +
                ((memExposureFuel : Rat) * r.total +
                  (3 * r.mem * r.mem * ((opsExposureSum d' rest : Nat) : Rat) +
                    (rest.length : Rat) * ((memExposureFuel : Rat) * r.total)))) := Rat.add_assoc _ _ _
          _ = (3 * r.mem * r.mem * ((delta * delta : Nat) : Rat) +
                ((3 * r.mem * r.mem * ((opsExposureSum d' rest : Nat) : Rat) +
                  (memExposureFuel : Rat) * r.total) +
                    (rest.length : Rat) * ((memExposureFuel : Rat) * r.total))) := by
              congr 1; rw [← Rat.add_assoc, Rat.add_comm ((memExposureFuel : Rat) * r.total)]
          _ = (3 * r.mem * r.mem * ((delta * delta : Nat) : Rat) +
                (3 * r.mem * r.mem * ((opsExposureSum d' rest : Nat) : Rat) +
                  ((memExposureFuel : Rat) * r.total +
                    (rest.length : Rat) * ((memExposureFuel : Rat) * r.total)))) := by
              congr 1; rw [Rat.add_assoc]
          _ = _ := (Rat.add_assoc _ _ _).symm
      rw [hcomm, Rat.add_comm ((memExposureFuel : Rat) * r.total)]
    simpa only [memExposureMultiPhys, opsExposureSum] using (hsplit ▸ htot_le)

/-! ### Concrete `PDist.p7` `Rat` witness -/

private theorem p7_step_k0_rat :
    3 * (((0 + 1 : Nat) : Rat) * (1 / 5000000)) ^ 2 + (memExposureFuel : Rat) * (4 / 5000000) =
      9000000003 / 25000000000000 := by
  decide +kernel

private theorem p7_step_k9_rat :
    3 * (((9 + 1 : Nat) : Rat) * (1 / 5000000)) ^ 2 + (memExposureFuel : Rat) * (4 / 5000000) =
      90000003 / 250000000000 := by
  decide +kernel

private theorem p7_step4_k9_rat :
    3 * (((9 + 1 : Nat) : Rat) * (1 / 5000000)) ^ 2 +
      (3 * (((9 + 1 : Nat) : Rat) * (1 / 5000000)) ^ 2 +
      (3 * (((9 + 1 : Nat) : Rat) * (1 / 5000000)) ^ 2 +
      (3 * (((9 + 1 : Nat) : Rat) * (1 / 5000000)) ^ 2 + (memExposureFuel : Rat) * (4 / 5000000)))) =
      22500003 / 62500000000 := by
  decide +kernel

/-- Concrete `PDist.p7` (`[1 / 20000000, 1 / 5000000]`) rational witness for the Decision 50
triplicated memory-exposure kernel (`415` bytes, `memExposureFuel = 450` cycles):
1. Unconditional `X86Int.prun` physical bound under any 1-arena corruption (`sdc_cert_zero` +
   `memExposurePhys`): `≤ 9 / 25000`.
2. Exposed `val[w]` single write at `k = 0` (`delta = 1`): `≤ 9000000003 / 25000000000000`.
3. Exposed `val[w]` single write at `k = 9` (`delta = 10`): `≤ 90000003 / 250000000000`.
4. All 4 words `(val[w], last_step[w], step, acc)` simultaneously exposed `k = 9` ticks:
   `≤ 22500003 / 62500000000`. -/
theorem witness_p7 (r : X86Int.Rates) (hr : r.ok)
    (hf : PDist.p7.mem r.fetch) (hg : PDist.p7.mem r.reg)
    (hm : PDist.p7.mem r.mem) (hl : PDist.p7.mem r.flag)
    (s : X86Int.St) {w v_new : Nat} (hw : w < maxSlots) (hvnew : v_new < X86Env.W)
    {d d0 d1 d2 : List Nat} (hd : ArenaDataOk d)
    (hd0 : ArenaDataOk d0) (hd1 : ArenaDataOk d1) (hd2 : ArenaDataOk d2)
    (hsf : SingleArenaFault d d0 d1 d2)
    (hle : rdWord d (offLast w) ≤ rdWord d offStep)
    (hs1 : rdWord d offStep + 1 < X86Env.W)
    (hsq : (rdWord d offStep - rdWord d (offLast w) + 1) *
      (rdWord d offStep - rdWord d (offLast w) + 1) < X86Env.W)
    (hac : rdWord d offAcc + (rdWord d offStep - rdWord d (offLast w) + 1) *
      (rdWord d offStep - rdWord d (offLast w) + 1) < X86Env.W) :
    let val_v := rdWord d (offVal w)
    let step_v := rdWord d offStep
    let acc_v := rdWord d offAcc
    let last_v := rdWord d (offLast w)
    let sDoneClean := memExposureDone s w v_new step_v acc_v last_v d d d
    (memExposurePhys r hr s w v_new d0 d1 d2).ok ∧
    PDist.faultFree (memExposurePhys r hr s w v_new d0 d1 d2) =
      [.ok (memExposureDone s w v_new step_v acc_v last_v d0 d1 d2)] ∧
    PDist.fm (memExposurePhys r hr s w v_new d0 d1 d2) ≤ 9 / 25000 ∧
    X86Int.Out.ok sDoneClean ∈ PDist.faultFree (memExposureExposedPhys s w v_new d hd hw val_v 0 r hr) ∧
    PDist.fm (memExposureExposedPhys s w v_new d hd hw val_v 0 r hr) ≤ 9000000003 / 25000000000000 ∧
    X86Int.Out.ok sDoneClean ∈ PDist.faultFree (memExposureExposedPhys s w v_new d hd hw val_v 9 r hr) ∧
    PDist.fm (memExposureExposedPhys s w v_new d hd hw val_v 9 r hr) ≤ 90000003 / 250000000000 ∧
    X86Int.Out.ok sDoneClean ∈ PDist.faultFree (memExposureExposed4Phys s w v_new d hd hw 9 9 9 9 r hr) ∧
    PDist.fm (memExposureExposed4Phys s w v_new d hd hw 9 9 9 9 r hr) ≤ 22500003 / 62500000000 := by
  intro val_v step_v acc_v last_v sDoneClean
  have htot : r.total ≤ 4 / 5000000 := by
    have hsum : r.fetch + (r.reg + (r.mem + r.flag)) ≤
        (1 / 5000000 : Rat) + (1 / 5000000 + (1 / 5000000 + 1 / 5000000)) :=
      add_le_add' hf.2 (add_le_add' hg.2 (add_le_add' hm.2 hl.2))
    have h4 : (1 / 5000000 : Rat) + (1 / 5000000 + (1 / 5000000 + 1 / 5000000)) = 4 / 5000000 := by
      decide +kernel
    exact Rat.le_trans hsum (h4 ▸ Rat.le_refl)
  have hfuel : (memExposureFuel : Rat) * r.total ≤ (memExposureFuel : Rat) * (4 / 5000000) :=
    Rat.mul_le_mul_of_nonneg_left htot Rat.natCast_nonneg
  have hfuel_eq : (memExposureFuel : Rat) * (4 / 5000000) = 9 / 25000 := by decide +kernel
  have hcert := sdc_cert_zero s hw hvnew hd hd0 hd1 hd2 hsf hle hs1 hsq hac
  have hclean_phys : PDist.faultFree (memExposurePhys r hr s w v_new d0 d1 d2) =
      [.ok (memExposureDone s w v_new step_v acc_v last_v d0 d1 d2)] := by
    rw [memExposure_phys_clean, hcert.1]
  have hbd_k : ∀ k : Nat,
      3 * (((k + 1 : Nat) : Rat) * r.mem) ^ 2 ≤
        3 * (((k + 1 : Nat) : Rat) * (1 / 5000000)) ^ 2 := by
    intro k
    have hm_le : ((k + 1 : Nat) : Rat) * r.mem ≤ ((k + 1 : Nat) : Rat) * (1 / 5000000) :=
      Rat.mul_le_mul_of_nonneg_left hm.2 Rat.natCast_nonneg
    have hm0 : 0 ≤ ((k + 1 : Nat) : Rat) * r.mem :=
      Rat.mul_nonneg Rat.natCast_nonneg hr.2.2.1.1
    have hsqk : (((k + 1 : Nat) : Rat) * r.mem) ^ 2 ≤ (((k + 1 : Nat) : Rat) * (1 / 5000000)) ^ 2 := by
      rw [Rat.pow_succ, Rat.pow_one, Rat.pow_succ, Rat.pow_one]
      exact TMR.sq_le hm0 hm_le
    exact Rat.mul_le_mul_of_nonneg_left hsqk (by decide)
  have hk0 := memExposure_phys_bound s hw hvnew hd 0 r hr (rdWord_lt_W hd.2 _) hle hs1 hsq hac
  have hk9 := memExposure_phys_bound s hw hvnew hd 9 r hr (rdWord_lt_W hd.2 _) hle hs1 hsq hac
  have hk4 := memExposure_four_word_phys_bound s hw hvnew hd 9 9 9 9 r hr hle hs1 hsq hac
  refine ⟨memExposure_phys_ok r hr s w v_new d0 d1 d2, hclean_phys,
    Rat.le_trans (memExposure_phys_fm_le r hr s w v_new d0 d1 d2) (hfuel_eq ▸ hfuel),
    hk0.2.1, ?_, hk9.2.1, ?_, hk4.2.1, ?_⟩
  · exact Rat.le_trans hk0.2.2.2 (Rat.le_trans (add_le_add' (hbd_k 0) hfuel) (p7_step_k0_rat ▸ Rat.le_refl))
  · exact Rat.le_trans hk9.2.2.2 (Rat.le_trans (add_le_add' (hbd_k 9) hfuel) (p7_step_k9_rat ▸ Rat.le_refl))
  · exact Rat.le_trans hk4.2.2.2 (Rat.le_trans
      (add_le_add' (hbd_k 9) (add_le_add' (hbd_k 9) (add_le_add' (hbd_k 9) (add_le_add' (hbd_k 9) hfuel))))
      (p7_step4_k9_rat ▸ Rat.le_refl))

end TMRMemExposure
