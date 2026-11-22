/-
Stage-1 final proof, definitions phase: the initial process
state, oracle consumption, and the `Room` witness data.  Split
out of `Bootstrap.SporosX1Final` unchanged; the next phase is
`Bootstrap.SporosX1FinalLoader`.
-/
import Bootstrap.SporosX1Image

open X86Env X86Int X86Dec Sporos

namespace SporosX1


/-- The bytes the Linux loader maps for the single `PT_LOAD`
segment: `p_filesz` bytes from `p_offset`, zero-filled to
`p_memsz`.  The field theorems of `Bootstrap.SporosX1Image` pin
these to the whole code, as `loadSegment_eq` records. -/
def loadSegment : List Nat :=
  (stage1Code.drop phOffset).take phFilesz ++
    List.replicate (phMemsz - phFilesz) 0

theorem loadSegment_eq : loadSegment = stage1Code := by
  unfold loadSegment
  rw [phOffset_eq, phFilesz_eq, phMemsz_eq]
  simp

/-- The initial stage-1 process state, as the Linux loader builds
it from the ELF fields: the load segment mapped read-only at
`p_vaddr`, entry at `e_entry`, a Linux initial stack for
`argv = [argv0, path]` ending at `top`, all registers zero except
`rsp`, and a kernel whose only file maps `path` to `f`.  Stdin and
the oracle are parameters: the stage-1 checker verdict must not
depend on them, except that the oracle may refuse memory. -/
def stage1Init (f path argv0 stdin : List Nat) (oracle : List Choice)
    (top : Nat) : X86Int.St :=
  let (stk, sp) := X86Env.initStack top [argv0, path]
  { regs := (List.replicate 16 (some 0)).set 4 (some sp),
    rip := elfEntry,
    fl := Flags.undef,
    mem := [⟨phVaddr, loadSegment, false⟩, stk],
    k := ⟨[(path, f)], [], stdin, [], oracle⟩ }

/-- Hypotheses on the initial state that are not about the program
file's contents: byte-valued file and path, a NUL-free path short
enough for the model's `cstr`, a page-aligned stack top, and code and
stack regions disjoint inside the model's address bound. -/
def Stage1InitOk (f path argv0 : List Nat) (top : Nat) : Prop :=
  (∀ b ∈ f, b < 256) ∧ (∀ b ∈ path, b < 256) ∧
    (∀ b ∈ argv0, b < 256) ∧ 0 ∉ path ∧ 0 ∉ argv0 ∧
    path.length < 4096 ∧ top % page = 0 ∧ top ≤ 2 ^ 47 ∧
    phVaddr + phMemsz ≤
      (X86Env.initStack top [argv0, path]).1.base ∧
    (X86Env.initStack top [argv0, path]).1.hi = top

/-- The state `s` is about to execute an `mmap`/`mremap` syscall. -/
def AtMemSyscall (s : X86Int.St) : Prop :=
  (stage1Im s.rip = some (.syscall, 2)) ∧
    (s.reg 0 = some 9 ∨ s.reg 0 = some 25)

/-- The oracle's next relevant choice at `s` does not make the
model's memory syscall undefined.  Stated directly through
`X86Env.syscall`, at the register values `s` is about to use: the
call returns (a placement, or `-ENOMEM` on a refusal) instead of
`none`, which is how an illegal or missing oracle choice would make
the machine stuck. -/
def OracleLegalAt (s : X86Int.St) : Prop :=
  AtMemSyscall s →
    ∃ nr a1 a2 a3 a4 a5 a6 rax k' m',
      s.reg 0 = some nr ∧ s.reg 7 = some a1 ∧ s.reg 6 = some a2 ∧
      s.reg 2 = some a3 ∧ s.reg 10 = some a4 ∧ s.reg 8 = some a5 ∧
      s.reg 9 = some a6 ∧
      X86Env.syscall s.k s.mem nr a1 a2 a3 a4 a5 a6 =
        some (.ret rax k' m')

/-- Every reachable memory syscall meets a legal oracle choice. -/
def OracleAdmits (s0 : X86Int.St) : Prop :=
  ∀ k s, run stage1Im k (Out.ok s0) = Out.ok s → OracleLegalAt s

/-- The run reaches a memory syscall whose next oracle choice refuses
memory.  This is the only route to exit 2. -/
def HitsNomem (s0 : X86Int.St) : Prop :=
  ∃ k s, run stage1Im k (Out.ok s0) = Out.ok s ∧ AtMemSyscall s ∧
    s.k.oracle.head? = some .nomem

/-! ### Oracle consumption (toward the existence witness)

A syscall consumes at most one oracle choice: `Kern.next` drops the
head, and `Kern.count` drops it only when it is a count.  Hence the
oracle in any reachable state is a drop of the initial oracle, and
an initial oracle with no `nomem` entry can never present `nomem`
at a memory syscall. -/

theorem kern_next_oracle_drop (k : Kern) :
    ∃ n, n ≤ 1 ∧ (Kern.next k).2.oracle = k.oracle.drop n := by
  cases ho : k.oracle with
  | nil => exact ⟨0, by decide, by simp [Kern.next, ho]⟩
  | cons c rest => exact ⟨1, by decide, by simp [Kern.next, ho]⟩

theorem kern_count_oracle_drop (k : Kern) (want avail : Nat) :
    ∃ n, n ≤ 1 ∧ (Kern.count k want avail).2.oracle = k.oracle.drop n := by
  obtain ⟨n, hn, hd⟩ := kern_next_oracle_drop k
  cases hnext : Kern.next k with
  | mk oc k' =>
    rw [hnext] at hd
    dsimp only [] at hd
    cases oc with
    | none =>
      exact ⟨0, by decide, by simp [Kern.count, hnext]⟩
    | some c =>
      cases c <;>
        solve
          | (refine ⟨0, by decide, ?_⟩; simp [Kern.count, hnext])
          | (refine ⟨n, hn, ?_⟩
             unfold Kern.count
             simp only [hnext]
             split <;> dsimp only [] <;> exact hd)

theorem syscall0_oracle_drop {k : Kern} {m : Mem}
    {a1 a2 a3 a4 a5 a6 rax : Nat} {k' : Kern} {m' : Mem}
    (h : X86Env.syscall k m 0 a1 a2 a3 a4 a5 a6 =
      some (.ret rax k' m')) :
    ∃ n, n ≤ 1 ∧ k'.oracle = k.oracle.drop n := by
  by_cases h0 : a1 = 0
  · subst h0
    obtain ⟨n, hn, hd⟩ := kern_count_oracle_drop k a3 k.stdin.length
    refine ⟨n, hn, ?_⟩
    simp only [X86Env.syscall, if_true] at h
    cases hw : m.wrN a2 (k.stdin.take (k.count a3 k.stdin.length).1) with
    | none => simp [hw] at h
    | some m'' =>
      simp [hw] at h
      obtain ⟨-, rfl, -⟩ := h
      exact hd
  · by_cases h3 : 3 ≤ a1
    · cases hfd : k.fds[a1 - 3]? with
      | some ofd =>
        cases ofd with
        | some fd =>
          obtain ⟨n, hn, hd⟩ :=
            kern_count_oracle_drop k a3 (fd.data.length - fd.pos)
          refine ⟨n, hn, ?_⟩
          unfold X86Env.syscall at h
          rw [if_neg h0, if_pos h3, hfd] at h
          cases hw : m.wrN a2 ((fd.data.drop fd.pos).take
              (k.count a3 (fd.data.length - fd.pos)).1) with
          | none => simp [hw] at h
          | some m'' =>
            simp [hw] at h
            obtain ⟨-, rfl, -⟩ := h
            exact hd
        | none =>
          refine ⟨0, by decide, ?_⟩
          simp only [X86Env.syscall] at h
          rw [if_neg h0, if_pos h3, hfd] at h
          simp at h
          obtain ⟨-, rfl, -⟩ := h
          rfl
      | none =>
        refine ⟨0, by decide, ?_⟩
        simp only [X86Env.syscall] at h
        rw [if_neg h0, if_pos h3, hfd] at h
        simp at h
        obtain ⟨-, rfl, -⟩ := h
        rfl
    · refine ⟨0, by decide, ?_⟩
      simp only [X86Env.syscall] at h
      rw [if_neg h0, if_neg h3] at h
      simp at h
      obtain ⟨-, rfl, -⟩ := h
      rfl

theorem syscall1_oracle_drop {k : Kern} {m : Mem}
    {a1 a2 a3 a4 a5 a6 rax : Nat} {k' : Kern} {m' : Mem}
    (h : X86Env.syscall k m 1 a1 a2 a3 a4 a5 a6 =
      some (.ret rax k' m')) :
    ∃ n, n ≤ 1 ∧ k'.oracle = k.oracle.drop n := by
  by_cases h1 : a1 = 1
  · subst h1
    obtain ⟨n, hn, hd⟩ := kern_count_oracle_drop k a3 a3
    refine ⟨n, hn, ?_⟩
    simp only [X86Env.syscall] at h
    cases hw : m.rdN a2 (k.count a3 a3).1 with
    | none => simp [hw] at h
    | some bs =>
      simp [hw] at h
      obtain ⟨-, rfl, -⟩ := h
      exact hd
  · refine ⟨0, by decide, ?_⟩
    simp only [X86Env.syscall] at h
    rw [if_neg h1] at h
    simp at h
    obtain ⟨-, rfl, -⟩ := h
    rfl

theorem syscall2_oracle_drop {k : Kern} {m : Mem}
    {a1 a2 a3 a4 a5 a6 rax : Nat} {k' : Kern} {m' : Mem}
    (h : X86Env.syscall k m 2 a1 a2 a3 a4 a5 a6 =
      some (.ret rax k' m')) :
    ∃ n, n ≤ 1 ∧ k'.oracle = k.oracle.drop n := by
  by_cases h2 : a2 ≠ 0
  · simp [X86Env.syscall, h2] at h
  · refine ⟨0, by decide, ?_⟩
    cases hc : m.cstr a1 4096 with
    | none => simp [X86Env.syscall, h2, hc] at h
    | some name =>
      cases hl : lookup k.files name with
      | none =>
        simp [X86Env.syscall, h2, hc, hl] at h
        obtain ⟨-, rfl, -⟩ := h
        rfl
      | some data =>
        simp [X86Env.syscall, h2, hc, hl] at h
        obtain ⟨-, rfl, -⟩ := h
        rfl

theorem syscall_oracle_drop_of_nr {k : Kern} {m : Mem} {nr : Nat}
    (hnr : nr = 2 ∨ nr = 3 ∨ nr = 11)
    {a1 a2 a3 a4 a5 a6 rax : Nat} {k' : Kern} {m' : Mem}
    (h : X86Env.syscall k m nr a1 a2 a3 a4 a5 a6 =
      some (.ret rax k' m')) :
    ∃ n, n ≤ 1 ∧ k'.oracle = k.oracle.drop n := by
  rcases hnr with rfl | rfl | rfl
  · exact syscall2_oracle_drop h
  all_goals
    (simp only [X86Env.syscall] at h
     repeat' split at h <;>
       (refine ⟨0, by decide, ?_⟩
        simp only [List.drop_zero]
        first
          | (obtain ⟨-, rfl, -⟩ := h; rfl)
          | (cases hc : m.cstr a1 4096 <;> simp only [hc, Option.bind_some] at h <;>
               split at h <;> simp_all)
          | simp_all))

theorem syscall9_oracle_drop {k : Kern} {m : Mem}
    {a1 a2 a3 a4 a5 a6 rax : Nat} {k' : Kern} {m' : Mem}
    (h : X86Env.syscall k m 9 a1 a2 a3 a4 a5 a6 =
      some (.ret rax k' m')) :
    ∃ n, n ≤ 1 ∧ k'.oracle = k.oracle.drop n := by
  by_cases hbad : a1 ≠ 0 ∨ a3 ≠ PROT_RW ∨ a4 ≠ MAP_PRIVATE_ANON ∨
      a5 ≠ W - 1 ∨ a6 ≠ 0
  · unfold X86Env.syscall at h
    rw [if_pos hbad] at h
    simp at h
  · by_cases h0 : a2 = 0
    · refine ⟨0, by decide, ?_⟩
      unfold X86Env.syscall at h
      rw [if_neg hbad, if_pos h0] at h
      simp at h
      obtain ⟨-, rfl, -⟩ := h
      rfl
    · obtain ⟨n, hn, hd⟩ := kern_next_oracle_drop k
      refine ⟨n, hn, ?_⟩
      unfold X86Env.syscall at h
      rw [if_neg hbad, if_neg h0] at h
      cases hnext : Kern.next k with
      | mk c k1 =>
        rw [hnext] at hd
        dsimp only [] at hd
        cases hp : X86Env.place m c (X86Env.pages a2) with
        | none => simp [hnext, hp] at h
        | some oa =>
          cases oa with
          | none =>
            simp [hnext, hp] at h
            obtain ⟨-, rfl, -⟩ := h
            exact hd
          | some a =>
            simp [hnext, hp] at h
            obtain ⟨-, rfl, -⟩ := h
            exact hd

theorem syscall25_oracle_drop {k : Kern} {m : Mem}
    {a1 a2 a3 a4 a5 a6 rax : Nat} {k' : Kern} {m' : Mem}
    (h : X86Env.syscall k m 25 a1 a2 a3 a4 a5 a6 =
      some (.ret rax k' m')) :
    ∃ n, n ≤ 1 ∧ k'.oracle = k.oracle.drop n := by
  by_cases hbad : a4 ≠ MREMAP_MAYMOVE
  · simp only [X86Env.syscall] at h
    rw [if_pos hbad] at h
    simp at h
  · cases hfind : m.find a1 0 with
    | none =>
      simp only [X86Env.syscall] at h
      rw [if_neg hbad, hfind] at h
      simp at h
    | some ir =>
      obtain ⟨i, r⟩ := ir
      by_cases hbad2 : r.base ≠ a1 ∨ r.data.length ≠ X86Env.pages a2 ∨ ¬ r.rw
      · simp [X86Env.syscall, hbad, hfind, hbad2] at h
        rcases hbad2 with h | h | h <;> simp_all
      · by_cases hsmall : X86Env.pages a3 < r.data.length
        · simp [X86Env.syscall, hbad, hfind, hbad2, hsmall] at h
        · obtain ⟨n, hn, hd⟩ := kern_next_oracle_drop k
          refine ⟨n, hn, ?_⟩
          simp only [X86Env.syscall, if_neg hbad, hfind, if_neg hbad2,
            if_neg hsmall] at h
          cases hnext : Kern.next k with
          | mk c k1 =>
            rw [hnext] at hd
            dsimp only [] at hd
            cases hp : X86Env.place (m.eraseIdx i) c (X86Env.pages a3) with
            | none => simp [hnext, hp] at h
            | some oa =>
              cases oa with
              | none =>
                simp [hnext, hp] at h
                obtain ⟨-, rfl, -⟩ := h
                exact hd
              | some a =>
                simp [hnext, hp] at h
                obtain ⟨-, rfl, -⟩ := h
                exact hd

theorem syscall_oracle_drop {k : Kern} {m : Mem}
    {nr a1 a2 a3 a4 a5 a6 rax : Nat} {k' : Kern} {m' : Mem}
    (h : X86Env.syscall k m nr a1 a2 a3 a4 a5 a6 =
      some (.ret rax k' m')) :
    ∃ n, n ≤ 1 ∧ k'.oracle = k.oracle.drop n := by
  by_cases h0 : nr = 0
  · subst h0; exact syscall0_oracle_drop h
  · by_cases h1 : nr = 1
    · subst h1; exact syscall1_oracle_drop h
    · by_cases h2 : nr = 2
      · exact syscall_oracle_drop_of_nr (Or.inl h2) h
      · by_cases h3 : nr = 3
        · exact syscall_oracle_drop_of_nr (Or.inr (Or.inl h3)) h
        · by_cases h9 : nr = 9
          · subst h9; exact syscall9_oracle_drop h
          · by_cases h11 : nr = 11
            · exact syscall_oracle_drop_of_nr (Or.inr (Or.inr h11)) h
            · by_cases h25 : nr = 25
              · subst h25; exact syscall25_oracle_drop h
              · by_cases h60 : nr = 60
                · subst h60
                  simp only [X86Env.syscall] at h
                  simp at h
                · unfold X86Env.syscall at h
                  split at h <;> simp_all

theorem exec_oracle_drop {ins : Ins} {len : Nat} {s s' : X86Int.St}
    (h : X86Int.exec ins len s = .ok s') :
    ∃ n, n ≤ 1 ∧ s'.k.oracle = s.k.oracle.drop n := by
  cases ins with
  | syscall =>
    cases hr0 : s.reg 0 with
    | none => simp [X86Int.exec, hr0] at h
    | some nr =>
    cases hr7 : s.reg 7 with
    | none => simp [X86Int.exec, hr0, hr7] at h
    | some a1 =>
    cases hr6 : s.reg 6 with
    | none => simp [X86Int.exec, hr0, hr7, hr6] at h
    | some a2 =>
    cases hr2 : s.reg 2 with
    | none => simp [X86Int.exec, hr0, hr7, hr6, hr2] at h
    | some a3 =>
    cases hr10 : s.reg 10 with
    | none => simp [X86Int.exec, hr0, hr7, hr6, hr2, hr10] at h
    | some a4 =>
    cases hr8 : s.reg 8 with
    | none => simp [X86Int.exec, hr0, hr7, hr6, hr2, hr10, hr8] at h
    | some a5 =>
    cases hr9 : s.reg 9 with
    | none => simp [X86Int.exec, hr0, hr7, hr6, hr2, hr10, hr8, hr9] at h
    | some a6 =>
      cases hsys : X86Env.syscall s.k s.mem nr a1 a2 a3 a4 a5 a6 with
      | none =>
        simp [X86Int.exec, hr0, hr7, hr6, hr2, hr10, hr8, hr9, hsys] at h
      | some res =>
        cases res with
        | exit c =>
          simp [X86Int.exec, hr0, hr7, hr6, hr2, hr10, hr8, hr9, hsys] at h
        | ret rax k m =>
          obtain ⟨n, hn, hd⟩ := syscall_oracle_drop hsys
          refine ⟨n, hn, ?_⟩
          simp [X86Int.exec, hr0, hr7, hr6, hr2, hr10, hr8, hr9, hsys] at h
          obtain rfl := h
          exact hd
  | _ =>
    refine ⟨0, by decide, ?_⟩
    simp only [List.drop_zero]
    simp only [X86Int.exec] at h
    first
      | (injection h with h; cases h)
      | (have h1 := lift_ok h
         obtain ⟨a, -, h1⟩ := bind_some h1
         cases h1)
      | (have h1 := lift_ok h
         obtain ⟨a, -, h1⟩ := bind_some h1
         obtain ⟨b, -, h1⟩ := bind_some h1
         cases h1)
      | (have h1 := lift_ok h
         obtain ⟨a, -, h1⟩ := bind_some h1
         obtain ⟨b, -, h1⟩ := bind_some h1
         obtain ⟨c, -, h1⟩ := bind_some h1
         cases h1)
      | (split at h <;> (try split at h) <;>
          (first
            | cases h
            | (injection h with hh; cases hh)))
    all_goals rfl

theorem step_oracle_drop {im : Nat → Option (Ins × Nat)}
    {s s' : X86Int.St} (h : X86Int.step im s = .ok s') :
    ∃ n, n ≤ 1 ∧ s'.k.oracle = s.k.oracle.drop n := by
  cases him : im s.rip with
  | none => simp [X86Int.step, him] at h
  | some il =>
    obtain ⟨ins, len⟩ := il
    exact exec_oracle_drop (by simpa [X86Int.step, him] using h)

theorem run_oracle_drop {im : Nat → Option (Ins × Nat)} :
    ∀ (k : Nat) (s s' : X86Int.St),
      X86Int.run im k (.ok s) = .ok s' →
      ∃ n, s'.k.oracle = s.k.oracle.drop n := by
  intro k
  induction k with
  | zero =>
    intro s s' h
    simp [X86Int.run] at h
    obtain rfl := h
    exact ⟨0, rfl⟩
  | succ k ih =>
    intro s s' h
    cases hs : X86Int.step im s with
    | ok s1 =>
      have hr : X86Int.run im k (.ok s1) = .ok s' := by
        have : X86Int.run im (k + 1) (.ok s) = X86Int.run im k (.ok s1) := by
          rw [X86Int.run, hs]
        rw [h] at this
        exact this.symm
      obtain ⟨n1, -, hd1⟩ := step_oracle_drop hs
      obtain ⟨n2, hd2⟩ := ih s1 s' hr
      exact ⟨n1 + n2, by rw [hd2, hd1, List.drop_drop]⟩
    | exit c =>
      cases k <;> simp [X86Int.run, hs] at h
    | stuck =>
      cases k <;> simp [X86Int.run, hs] at h

theorem not_hitsNomem_of_no_nomem {s0 : X86Int.St}
    (hno : ∀ c ∈ s0.k.oracle, c ≠ .nomem) :
    ¬ HitsNomem s0 := by
  rintro ⟨k, s, hr, -, hhead⟩
  obtain ⟨n, hd⟩ := run_oracle_drop k s0 s hr
  have hmem : Choice.nomem ∈ s0.k.oracle := by
    have h1 : Choice.nomem ∈ s.k.oracle := by
      cases hs : s.k.oracle with
      | nil => simp [hs] at hhead
      | cons c rest =>
        simp [hs] at hhead
        rw [← hhead]
        exact List.mem_cons_self
    rw [hd] at h1
    exact List.mem_of_mem_drop h1
  exact hno _ hmem rfl

/-! ### Room, `M f`, and the witness oracle

With the `nwords * 20 ≤ remaining` refusal now in the hex, every
allocation size is bounded by the file length: the file buffer's
final capacity is at most `65536 + 2 * f.length`, the words table is
`64 * nwords ≤ 4 * f.length`, and the lengths scratch is `f.length`.
`stage1Mf` is the sum of those three page-rounded bounds.  `Room`
asks for one free interval of that size; `stage1Oracle` places the
file buffer (initial `mmap` and every growth `mremap`, reusing the
same base after the model erases the old region), then the table,
then the scratch, at successive sub-bases of that interval. -/

/-- Page-rounded byte bound for the file buffer at final capacity. -/
def arenaFileBytes (f : List Nat) : Nat :=
  X86Env.pages (65536 + 2 * f.length)

/-- Page-rounded byte bound for the words table. -/
def arenaTableBytes (f : List Nat) : Nat :=
  X86Env.pages (4 * f.length)

/-- Page-rounded byte bound for the lengths scratch. -/
def arenaScratchBytes (f : List Nat) : Nat :=
  X86Env.pages f.length

/-- `M f`: bytes of address space the witness oracle needs, as an
explicit function of the file size. -/
def stage1Mf (f : List Nat) : Nat :=
  arenaFileBytes f + arenaTableBytes f + arenaScratchBytes f

/-- There is a free, page-aligned interval of `stage1Mf f` bytes
below `2^47`, disjoint from the initial code and stack regions. -/
def Room (f path argv0 : List Nat) (top : Nat) : Prop :=
  ∃ A, A % page = 0 ∧ 0 < A ∧ A + stage1Mf f ≤ 2 ^ 47 ∧
    X86Env.Mem.free (stage1Init f path argv0 [] [] top).mem A
      (stage1Mf f) = true

/-- Number of loader growth `mremap`s for a file of length `L`:
the capacity doubles from 64 KiB until it strictly exceeds `L`
(the loader grows when a read exactly fills the region). -/
def stage1Growths (L : Nat) : Nat :=
  if L < 65536 then 0 else Nat.log2 (L / 65536) + 1

/-- The witness oracle: the file base repeated for the initial
`mmap` and each growth `mremap`, then the table and scratch bases.
No `nomem`, and no counts (reads therefore always return in full
and do not consume an entry). -/
def stage1Oracle (f : List Nat) (A : Nat) : List Choice :=
  List.replicate (stage1Growths f.length + 1) (.at A) ++
    [.at (A + arenaFileBytes f),
     .at (A + (arenaFileBytes f + arenaTableBytes f))]

/-- `stage1Mf` is linear in the file length, with an explicit
constant: at most `65536 + 8 * (f.length + 4096)` bytes. -/
theorem stage1Mf_le (f : List Nat) :
    stage1Mf f ≤ 65536 + 8 * (f.length + 4096) := by
  have key : ∀ n : Nat, X86Env.pages n ≤ n + 4095 := by
    intro n
    unfold X86Env.pages
    exact Nat.div_mul_le_self _ _
  have h1 := key (65536 + 2 * f.length)
  have h2 := key (4 * f.length)
  have h3 := key f.length
  unfold stage1Mf arenaFileBytes arenaTableBytes arenaScratchBytes
  omega

/-- The witness oracle never refuses memory: every entry is a
placement. -/
theorem stage1Oracle_no_nomem {f : List Nat} {A : Nat} {c : Choice}
    (hc : c ∈ stage1Oracle f A) : c ≠ .nomem := by
  simp [stage1Oracle] at hc
  rcases hc with ⟨-, rfl⟩ | rfl | rfl <;> simp

/-- The witness oracle never presents a memory refusal at a memory
syscall: no entry is `nomem`, and reachable oracles are drops of it. -/
theorem stage1Oracle_not_hitsNomem
    (f path argv0 stdin : List Nat) (A top : Nat) :
    ¬ HitsNomem (stage1Init f path argv0 stdin (stage1Oracle f A) top) :=
  not_hitsNomem_of_no_nomem (fun c hc => stage1Oracle_no_nomem hc)

theorem kern_next_of_head {k : Kern} {c : Choice}
    (h : k.oracle.head? = some c) :
    (Kern.next k).1 = some c := by
  cases ho : k.oracle with
  | nil => simp [ho] at h
  | cons x rest =>
    simp [ho] at h
    simp [Kern.next, ho, h]

theorem syscall_mmap_ret_of_place {k : Kern} {m : Mem} {a2 A : Nat}
    (ha2 : a2 ≠ 0) (hhead : k.oracle.head? = some (.at A))
    (hplace : X86Env.place m (some (.at A)) (X86Env.pages a2) =
      some (some A)) :
    ∃ rax k' m', X86Env.syscall k m 9 0 a2 PROT_RW MAP_PRIVATE_ANON
      (W - 1) 0 = some (.ret rax k' m') := by
  refine ⟨A, (Kern.next k).2, m ++ [⟨A, List.replicate (X86Env.pages a2) 0, true⟩], ?_⟩
  have hnext1 : (Kern.next k).1 = some (.at A) := kern_next_of_head hhead
  simp only [X86Env.syscall]
  rw [if_neg (by simp), if_neg ha2]
  cases hnext : Kern.next k with
  | mk c k1 =>
    rw [hnext] at hnext1
    simp only [] at hnext1
    subst hnext1
    simp [hplace]

theorem syscall_mremap_ret_of_place {k : Kern} {m : Mem}
    {a1 a2 a3 A : Nat} {i : Nat} {r : Region}
    (hfind : m.find a1 0 = some (i, r))
    (hbase : r.base = a1) (hlen : r.data.length = X86Env.pages a2)
    (hrw : r.rw = true) (hgrow : ¬ X86Env.pages a3 < r.data.length)
    (hhead : k.oracle.head? = some (.at A))
    (hplace : X86Env.place (m.eraseIdx i) (some (.at A))
      (X86Env.pages a3) = some (some A)) :
    ∃ rax k' m', X86Env.syscall k m 25 a1 a2 a3 MREMAP_MAYMOVE 0 0 =
      some (.ret rax k' m') := by
  refine ⟨A, (Kern.next k).2,
    m.eraseIdx i ++ [⟨A, r.data ++
      List.replicate (X86Env.pages a3 - r.data.length) 0, true⟩], ?_⟩
  have hnext1 : (Kern.next k).1 = some (.at A) := kern_next_of_head hhead
  have hval : ¬ (r.base ≠ a1 ∨ r.data.length ≠ X86Env.pages a2 ∨
      ¬ r.rw) := by simp [hbase, hlen, hrw]
  rw [hlen] at hgrow
  simp only [X86Env.syscall, hfind, hbase, hlen, hrw, hgrow,
    if_false]
  cases hnext : Kern.next k with
  | mk c k1 =>
    rw [hnext] at hnext1
    simp only [] at hnext1
    subst hnext1
    dsimp only []
    rw [hplace]
    simp [hgrow]


end SporosX1
